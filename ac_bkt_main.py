"""
AC-BKT: Asymmetric Consistency Bayesian Knowledge Tracing
==========================================================

A compact, publication-oriented implementation containing only:
1. AC-BKT model definition;
2. leakage-free chronological train/validation/test splitting;
3. validation-only hyperparameter selection;
4. training and held-out prediction;
5. AUC and RMSE reporting.

The implementation is distilled from the authors' experimental notebook and
removes ablation, sensitivity, bootstrap, trajectory, and plotting code.

Default split
-------------
All datasets use a chronological 60/20/20 split by default.

ASSISTments2015 note
--------------------
In our additional split-sensitivity check, ASSISTments2015 showed materially
lower AUC with the 60/20/20 allocation. Keeping the same final 20% test period
but increasing the training portion from 60% to 70% (70/10/20) improved AC-BKT
AUC from about 0.60 to 0.663931. This is consistent with insufficient fitting
under the smaller 60% training partition. For ASSISTments2015, we therefore
recommend running this script with:

    python ac_bkt_main.py --dataset ASSISTments2015 --split 70-10-20

For the other datasets, the default 60/20/20 split is retained unless a
different split is explicitly requested.

Example
-------
    python ac_bkt_main.py --dataset ASSISTments2009
    python ac_bkt_main.py --dataset ASSISTments2015 --split 70-10-20

Dependencies
------------
numpy, pandas, scikit-learn, torch
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, roc_auc_score


EPS = 1e-6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SPLIT_TO_CODE = {"train": 0, "val": 1, "test": 2}
CODE_TO_SPLIT = {value: key for key, value in SPLIT_TO_CODE.items()}


# ---------------------------------------------------------------------------
# Reproducibility and metrics
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_auc(y_true: Sequence[float], y_prob: Sequence[float]) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def compute_metrics(
    y_true: Sequence[float], y_prob: Sequence[float]
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), EPS, 1.0 - EPS)

    if len(y_true) == 0:
        return {"auc": float("nan"), "rmse": float("nan"), "n": 0}

    return {
        "auc": safe_auc(y_true, y_prob),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_prob))),
        "n": int(len(y_true)),
    }


# ---------------------------------------------------------------------------
# Dataset loading and leakage-free chronological splitting
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataConfig:
    path: str
    user_col: str
    skill_col: str
    correct_col: str
    order_col: str
    sep: str = ","
    encoding: str = "utf-8"
    multi_skill_sep: Optional[str] = None
    min_seq_len: int = 10
    train_ratio: float = 0.60
    val_ratio: float = 0.20
    correct_map: Optional[Dict[str, int]] = None
    order_is_datetime: bool = False


@dataclass
class SequenceRecord:
    seq_id: str
    user_id: str
    skill_id: str
    skill_idx: int
    answers: torch.Tensor
    split_codes: torch.Tensor
    positions: torch.Tensor


def load_interactions(cfg: DataConfig) -> pd.DataFrame:
    required = [cfg.user_col, cfg.skill_col, cfg.correct_col, cfg.order_col]

    df = pd.read_csv(
        cfg.path,
        sep=cfg.sep,
        usecols=required,
        low_memory=False,
        encoding=cfg.encoding,
    )
    df = df.rename(
        columns={
            cfg.user_col: "user_id",
            cfg.skill_col: "skill_id",
            cfg.correct_col: "correct",
            cfg.order_col: "order_id",
        }
    )

    df["source_row"] = np.arange(len(df), dtype=np.int64)
    df = df.dropna(
        subset=["user_id", "skill_id", "correct", "order_id"]
    ).copy()

    if cfg.multi_skill_sep:
        df["skill_id"] = (
            df["skill_id"].astype(str).str.split(cfg.multi_skill_sep)
        )
        df = df.explode("skill_id", ignore_index=True)

    df["skill_id"] = df["skill_id"].astype(str).str.strip()
    df = df[
        (df["skill_id"] != "") &
        (df["skill_id"].str.lower() != "nan")
    ].copy()

    if cfg.correct_map:
        normalized = df["correct"].astype(str).str.strip().str.upper()
        mapping = {
            str(key).strip().upper(): int(value)
            for key, value in cfg.correct_map.items()
        }
        df["correct"] = normalized.map(mapping)
    else:
        df["correct"] = pd.to_numeric(df["correct"], errors="coerce")

    if cfg.order_is_datetime:
        dt = pd.to_datetime(df["order_id"], errors="coerce")
        df["order_id"] = dt.astype("int64").where(dt.notna(), np.nan)
    else:
        df["order_id"] = pd.to_numeric(df["order_id"], errors="coerce")

    df = df[
        df["correct"].isin([0, 1]) & df["order_id"].notna()
    ].copy()
    df["correct"] = df["correct"].astype(np.float32)

    # Stable ordering prevents ties from being rearranged nondeterministically.
    df = df.sort_values(
        ["user_id", "skill_id", "order_id", "source_row"],
        kind="mergesort",
    ).reset_index(drop=True)

    sizes = df.groupby(["user_id", "skill_id"], sort=False).size()
    valid_pairs = sizes[sizes >= cfg.min_seq_len].index
    pair_index = pd.MultiIndex.from_frame(df[["user_id", "skill_id"]])
    df = df[pair_index.isin(valid_pairs)].copy()

    if df.empty:
        raise ValueError(
            "No valid interactions remain after preprocessing. "
            "Check file paths, columns, response coding, and min_seq_len."
        )

    skills = sorted(df["skill_id"].astype(str).unique())
    skill_map = {skill: idx for idx, skill in enumerate(skills)}
    df["skill_idx"] = (
        df["skill_id"].astype(str).map(skill_map).astype(np.int64)
    )

    return df.reset_index(drop=True)


def chronological_split(
    df: pd.DataFrame, cfg: DataConfig
) -> pd.DataFrame:
    """
    Split each student-skill sequence chronologically.

    Default: 60% train / 20% validation / 20% test.
    No future interaction can appear in training or validation relative to test.
    """
    if not (0.0 < cfg.train_ratio < 1.0):
        raise ValueError("train_ratio must lie in (0, 1).")
    if not (0.0 < cfg.val_ratio < 1.0):
        raise ValueError("val_ratio must lie in (0, 1).")
    if cfg.train_ratio + cfg.val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.")

    parts: List[pd.DataFrame] = []

    for (user_id, skill_id), group in df.groupby(
        ["user_id", "skill_id"], sort=False
    ):
        group = group.sort_values(
            ["order_id", "source_row"], kind="mergesort"
        ).copy()
        n = len(group)

        n_train = max(2, int(math.floor(n * cfg.train_ratio)))
        n_val = max(1, int(math.floor(n * cfg.val_ratio)))

        # Preserve at least one interaction in each partition for short sequences.
        if n_train + n_val >= n:
            n_train = max(2, n - 2)
            n_val = 1

        n_test = n - n_train - n_val
        if n_test < 1:
            raise ValueError(
                f"Sequence length {n} cannot form train/val/test partitions."
            )

        split = np.array(
            ["train"] * n_train +
            ["val"] * n_val +
            ["test"] * n_test,
            dtype=object,
        )

        group["split"] = split
        group["split_code"] = (
            pd.Series(split).map(SPLIT_TO_CODE).to_numpy()
        )
        group["seq_pos"] = np.arange(n, dtype=np.int64)
        group["seq_id"] = str(user_id) + "||" + str(skill_id)
        parts.append(group)

    out = pd.concat(parts, ignore_index=True)

    # Leakage checks.
    for seq_id, group in out.groupby("seq_id", sort=False):
        codes = (
            group.sort_values("seq_pos")["split_code"].to_numpy()
        )
        if not np.all(codes[:-1] <= codes[1:]):
            raise AssertionError(
                f"Non-chronological split detected for {seq_id}"
            )
        if set(codes.tolist()) != {0, 1, 2}:
            raise AssertionError(
                f"Missing train/val/test partition for {seq_id}"
            )

    return out


def make_sequences(df: pd.DataFrame) -> List[SequenceRecord]:
    records: List[SequenceRecord] = []

    for seq_id, group in df.groupby("seq_id", sort=False):
        group = group.sort_values("seq_pos", kind="mergesort")
        records.append(
            SequenceRecord(
                seq_id=str(seq_id),
                user_id=str(group["user_id"].iloc[0]),
                skill_id=str(group["skill_id"].iloc[0]),
                skill_idx=int(group["skill_idx"].iloc[0]),
                answers=torch.tensor(
                    group["correct"].to_numpy(),
                    dtype=torch.float32,
                ),
                split_codes=torch.tensor(
                    group["split_code"].to_numpy(),
                    dtype=torch.int8,
                ),
                positions=torch.tensor(
                    group["seq_pos"].to_numpy(),
                    dtype=torch.long,
                ),
            )
        )

    return records


# ---------------------------------------------------------------------------
# Cached batches
# ---------------------------------------------------------------------------

@dataclass
class TensorBatch:
    skill_idx: torch.Tensor
    answers: torch.Tensor
    valid_mask: torch.Tensor
    target_mask: torch.Tensor
    positions: torch.Tensor
    split_codes: torch.Tensor
    seq_ids: List[str]
    user_ids: List[str]
    skill_ids: List[str]

    @property
    def target_count(self) -> int:
        return int(self.target_mask.sum().item())

    def to(self, device: torch.device) -> "TensorBatch":
        non_blocking = device.type == "cuda"
        return TensorBatch(
            skill_idx=self.skill_idx.to(
                device, non_blocking=non_blocking
            ),
            answers=self.answers.to(
                device, non_blocking=non_blocking
            ),
            valid_mask=self.valid_mask.to(
                device, non_blocking=non_blocking
            ),
            target_mask=self.target_mask.to(
                device, non_blocking=non_blocking
            ),
            positions=self.positions.to(
                device, non_blocking=non_blocking
            ),
            split_codes=self.split_codes.to(
                device, non_blocking=non_blocking
            ),
            seq_ids=self.seq_ids,
            user_ids=self.user_ids,
            skill_ids=self.skill_ids,
        )


@dataclass
class PreparedData:
    records: List[SequenceRecord]
    train_batches: List[TensorBatch]
    val_batches: List[TensorBatch]
    test_batches: List[TensorBatch]
    num_skills: int


def _cutoff_for_target(
    record: SequenceRecord, target_code: int
) -> int:
    eligible = torch.where(
        record.split_codes <= target_code
    )[0]
    if eligible.numel() == 0:
        return 0
    return int(eligible[-1].item()) + 1


def _collate_records(
    records: Sequence[SequenceRecord],
    target_split: str,
    pin_memory: bool,
) -> TensorBatch:
    target_code = SPLIT_TO_CODE[target_split]
    cutoffs = [
        _cutoff_for_target(record, target_code)
        for record in records
    ]

    max_len = max(cutoffs)
    batch_size = len(records)

    answers = torch.zeros(
        (batch_size, max_len), dtype=torch.float32
    )
    valid_mask = torch.zeros(
        (batch_size, max_len), dtype=torch.bool
    )
    target_mask = torch.zeros(
        (batch_size, max_len), dtype=torch.bool
    )
    positions = torch.full(
        (batch_size, max_len), -1, dtype=torch.long
    )
    split_codes = torch.full(
        (batch_size, max_len), -1, dtype=torch.int8
    )
    skill_idx = torch.empty(batch_size, dtype=torch.long)

    for row, (record, cutoff) in enumerate(zip(records, cutoffs)):
        answers[row, :cutoff] = record.answers[:cutoff]
        valid_mask[row, :cutoff] = True
        split_codes[row, :cutoff] = record.split_codes[:cutoff]
        target_mask[row, :cutoff] = (
            record.split_codes[:cutoff] == target_code
        )
        positions[row, :cutoff] = record.positions[:cutoff]
        skill_idx[row] = record.skill_idx

    if pin_memory and torch.cuda.is_available():
        answers = answers.pin_memory()
        valid_mask = valid_mask.pin_memory()
        target_mask = target_mask.pin_memory()
        positions = positions.pin_memory()
        split_codes = split_codes.pin_memory()
        skill_idx = skill_idx.pin_memory()

    return TensorBatch(
        skill_idx=skill_idx,
        answers=answers,
        valid_mask=valid_mask,
        target_mask=target_mask,
        positions=positions,
        split_codes=split_codes,
        seq_ids=[record.seq_id for record in records],
        user_ids=[record.user_id for record in records],
        skill_ids=[record.skill_id for record in records],
    )


def build_cached_batches(
    records: Sequence[SequenceRecord],
    target_split: str,
    batch_size: int,
    bucket_multiplier: int = 20,
    seed: int = 0,
    pin_memory: bool = True,
) -> List[TensorBatch]:
    target_code = SPLIT_TO_CODE[target_split]

    valid_records = [
        record
        for record in records
        if bool(
            (record.split_codes == target_code).any().item()
        )
    ]

    valid_records.sort(
        key=lambda record: _cutoff_for_target(
            record, target_code
        )
    )

    rng = np.random.default_rng(seed)
    large_bucket_size = max(
        batch_size, batch_size * bucket_multiplier
    )
    grouped: List[List[SequenceRecord]] = []

    for start in range(
        0, len(valid_records), large_bucket_size
    ):
        bucket = valid_records[
            start : start + large_bucket_size
        ]
        permutation = rng.permutation(len(bucket))
        bucket = [bucket[i] for i in permutation]

        grouped.extend(
            [
                bucket[i : i + batch_size]
                for i in range(0, len(bucket), batch_size)
            ]
        )

    return [
        _collate_records(
            group,
            target_split=target_split,
            pin_memory=pin_memory,
        )
        for group in grouped
        if group
    ]


def prepare_data(
    records: List[SequenceRecord],
    num_skills: int,
    train_batch_size: int,
    eval_batch_size: int,
    seed: int,
) -> PreparedData:
    return PreparedData(
        records=records,
        train_batches=build_cached_batches(
            records,
            "train",
            train_batch_size,
            seed=seed,
        ),
        val_batches=build_cached_batches(
            records,
            "val",
            eval_batch_size,
            seed=seed,
        ),
        test_batches=build_cached_batches(
            records,
            "test",
            eval_batch_size,
            seed=seed,
        ),
        num_skills=num_skills,
    )


# ---------------------------------------------------------------------------
# AC-BKT model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ACBKTHyper:
    beta: float = 0.60
    lambda_c: float = 2.0
    sigma: float = 0.50
    alpha_g: float = 0.50
    alpha_s: float = 0.50
    tau: float = 0.25


class ACBKT(nn.Module):
    """
    Batched AC-BKT.

    Prediction at interaction t uses only response information observed
    strictly before y_t.

    c_t = tanh(lambda * (E_pos - E_neg))
    G_t = clip(G_base * (1 - alpha_g * c_t))
    S_t = clip(S_base * (1 + alpha_s * c_t))
    T_t = T_base * [tau + (1-tau) exp(-c_t^2 / (2 sigma^2))]
    """

    def __init__(self, num_skills: int, hp: ACBKTHyper):
        super().__init__()
        self.num_skills = num_skills
        self.hp = hp

        # Same initialization used by the source experimental implementation.
        self.raw_l0 = nn.Parameter(
            torch.empty(num_skills).normal_(0.0, 0.05)
        )
        self.raw_t = nn.Parameter(
            torch.empty(num_skills).normal_(-2.0, 0.05)
        )
        self.raw_s = nn.Parameter(
            torch.empty(num_skills).normal_(-2.2, 0.05)
        )
        self.raw_g = nn.Parameter(
            torch.empty(num_skills).normal_(-1.4, 0.05)
        )

    def base_params(
        self, skill_idx: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        p_l0 = torch.sigmoid(self.raw_l0[skill_idx])
        p_t = 0.50 * torch.sigmoid(
            self.raw_t[skill_idx]
        )
        p_s = 0.50 * torch.sigmoid(
            self.raw_s[skill_idx]
        )
        p_g = 0.50 * torch.sigmoid(
            self.raw_g[skill_idx]
        )
        return p_l0, p_t, p_s, p_g

    def forward_batch(
        self,
        skill_idx: torch.Tensor,
        answers: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        hp = self.hp
        batch_size, max_len = answers.shape

        p_l, p_t_base, p_s_base, p_g_base = (
            self.base_params(skill_idx)
        )

        e_pos = torch.zeros(
            batch_size,
            device=answers.device,
            dtype=answers.dtype,
        )
        e_neg = torch.zeros_like(e_pos)

        predictions = []

        for t in range(max_len):
            active = valid_mask[:, t]
            y = answers[:, t]

            # Pre-response directional consistency.
            c_t = torch.tanh(
                hp.lambda_c * (e_pos - e_neg)
            )

            p_g_t = torch.clamp(
                p_g_base * (1.0 - hp.alpha_g * c_t),
                EPS,
                1.0 - EPS,
            )
            p_s_t = torch.clamp(
                p_s_base * (1.0 + hp.alpha_s * c_t),
                EPS,
                1.0 - EPS,
            )

            sigma = max(float(hp.sigma), EPS)
            gate = (
                hp.tau
                + (1.0 - hp.tau)
                * torch.exp(
                    -(c_t * c_t) /
                    (2.0 * sigma * sigma)
                )
            )
            p_t_t = torch.clamp(
                p_t_base * gate,
                EPS,
                1.0 - EPS,
            )

            pred = torch.clamp(
                p_l * (1.0 - p_s_t)
                + (1.0 - p_l) * p_g_t,
                EPS,
                1.0 - EPS,
            )
            predictions.append(pred)

            posterior_if_correct = (
                p_l * (1.0 - p_s_t) / pred
            )

            wrong_denom = torch.clamp(
                p_l * p_s_t
                + (1.0 - p_l) * (1.0 - p_g_t),
                EPS,
                1.0 - EPS,
            )
            posterior_if_wrong = (
                p_l * p_s_t / wrong_denom
            )

            posterior = torch.where(
                y > 0.5,
                posterior_if_correct,
                posterior_if_wrong,
            )

            next_p_l = torch.clamp(
                posterior
                + (1.0 - posterior) * p_t_t,
                EPS,
                1.0 - EPS,
            )

            # Padding does not modify latent states.
            p_l = torch.where(active, next_p_l, p_l)

            # The current response updates the EMA only after its prediction.
            updated_e_pos = (
                hp.beta * e_pos
                + (1.0 - hp.beta) * y
            )
            updated_e_neg = (
                hp.beta * e_neg
                + (1.0 - hp.beta) * (1.0 - y)
            )

            e_pos = torch.where(
                active, updated_e_pos, e_pos
            )
            e_neg = torch.where(
                active, updated_e_neg, e_neg
            )

        return torch.stack(predictions, dim=1)


# ---------------------------------------------------------------------------
# Training and validation-only hyperparameter search
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 200
    lr: float = 0.03
    weight_decay: float = 1e-5
    patience: int = 20
    min_delta: float = 1e-5
    grad_clip: float = 5.0
    train_batch_size: int = 1024
    eval_batch_size: int = 2048


@dataclass(frozen=True)
class SearchConfig:
    n_candidates: int = 24
    screening_epochs: int = 50
    screening_patience: int = 7
    n_finalists: int = 4
    seed: int = 2026


@dataclass
class FitResult:
    model: ACBKT
    history: pd.DataFrame
    seconds: float
    best_epoch: int
    best_val_nll: float


def _batch_loss_sum(
    model: ACBKT, batch: TensorBatch
) -> Tuple[torch.Tensor, int]:
    batch = batch.to(DEVICE)

    pred = model.forward_batch(
        batch.skill_idx,
        batch.answers,
        batch.valid_mask,
    )
    mask = batch.target_mask

    loss_sum = F.binary_cross_entropy(
        pred[mask],
        batch.answers[mask],
        reduction="sum",
    )

    return loss_sum, int(mask.sum().item())


@torch.no_grad()
def evaluate_nll(
    model: ACBKT, batches: Sequence[TensorBatch]
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0

    for batch in batches:
        loss_sum, count = _batch_loss_sum(model, batch)
        total_loss += float(loss_sum.item())
        total_count += count

    return total_loss / max(total_count, 1)


def fit_acbkt(
    prepared: PreparedData,
    hp: ACBKTHyper,
    cfg: TrainConfig,
    seed: int,
    epochs_override: Optional[int] = None,
    patience_override: Optional[int] = None,
) -> FitResult:
    set_seed(seed)

    model = ACBKT(
        prepared.num_skills, hp
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    epochs = (
        epochs_override
        if epochs_override is not None
        else cfg.epochs
    )
    patience = (
        patience_override
        if patience_override is not None
        else cfg.patience
    )

    total_train_count = sum(
        batch.target_count
        for batch in prepared.train_batches
    )
    if total_train_count == 0:
        raise ValueError(
            "No training observations are available."
        )

    best_state = None
    best_val_nll = float("inf")
    best_epoch = 0
    stale = 0
    history_rows = []
    rng = np.random.default_rng(seed)
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss_sum = 0.0

        order = rng.permutation(
            len(prepared.train_batches)
        )

        for batch_index in order:
            batch = prepared.train_batches[
                int(batch_index)
            ].to(DEVICE)

            pred = model.forward_batch(
                batch.skill_idx,
                batch.answers,
                batch.valid_mask,
            )
            mask = batch.target_mask

            loss_sum = F.binary_cross_entropy(
                pred[mask],
                batch.answers[mask],
                reduction="sum",
            )

            normalized_loss = (
                loss_sum / total_train_count
            )
            normalized_loss.backward()
            train_loss_sum += float(
                loss_sum.detach().item()
            )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.grad_clip
        )
        optimizer.step()

        val_nll = evaluate_nll(
            model, prepared.val_batches
        )
        train_nll = (
            train_loss_sum / total_train_count
        )

        history_rows.append(
            {
                "epoch": epoch,
                "train_nll": train_nll,
                "val_nll": val_nll,
            }
        )

        if val_nll < best_val_nll - cfg.min_delta:
            best_val_nll = val_nll
            best_epoch = epoch
            best_state = copy.deepcopy(
                model.state_dict()
            )
            stale = 0
        else:
            stale += 1

        if stale >= patience:
            break

    if best_state is None:
        raise RuntimeError(
            "Training did not produce a valid checkpoint."
        )

    model.load_state_dict(best_state)

    return FitResult(
        model=model,
        history=pd.DataFrame(history_rows),
        seconds=time.perf_counter() - start,
        best_epoch=best_epoch,
        best_val_nll=best_val_nll,
    )


@torch.no_grad()
def predict_batches(
    model: ACBKT,
    batches: Sequence[TensorBatch],
    target_split: str,
) -> pd.DataFrame:
    model.eval()
    target_code = SPLIT_TO_CODE[target_split]
    rows: List[Dict[str, object]] = []

    for cached_batch in batches:
        batch = cached_batch.to(DEVICE)

        pred = model.forward_batch(
            batch.skill_idx,
            batch.answers,
            batch.valid_mask,
        )

        for row_index, seq_id in enumerate(batch.seq_ids):
            target_indices = torch.where(
                batch.target_mask[row_index]
            )[0].cpu().numpy()

            for t in target_indices:
                split_code = int(
                    batch.split_codes[
                        row_index, t
                    ].item()
                )
                if split_code != target_code:
                    continue

                rows.append(
                    {
                        "seq_id": seq_id,
                        "user_id": batch.user_ids[row_index],
                        "skill_id": batch.skill_ids[row_index],
                        "position": int(
                            batch.positions[
                                row_index, t
                            ].item()
                        ),
                        "split": CODE_TO_SPLIT[
                            split_code
                        ],
                        "y": float(
                            batch.answers[
                                row_index, t
                            ].item()
                        ),
                        "pred": float(
                            pred[
                                row_index, t
                            ].item()
                        ),
                    }
                )

    return pd.DataFrame(rows)


def score_on_split(
    model: ACBKT,
    batches: Sequence[TensorBatch],
    split_name: str,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    pred_df = predict_batches(
        model, batches, split_name
    )

    metrics = compute_metrics(
        pred_df["y"], pred_df["pred"]
    )
    return metrics, pred_df


def generate_hyperparameter_candidates(
    n_candidates: int, seed: int
) -> List[ACBKTHyper]:
    """
    Same candidate ranges as the source experimental notebook.
    A fixed random subset is screened to keep the search tractable.
    """
    grid = list(
        product(
            [0.40, 0.60, 0.80],       # beta
            [1.0, 2.0, 4.0],          # lambda_c
            [0.30, 0.50, 0.80],       # sigma
            [0.25, 0.50, 0.75],       # alpha_g
            [0.25, 0.50, 0.75],       # alpha_s
            [0.10, 0.25, 0.50],       # tau
        )
    )

    rng = np.random.default_rng(seed)
    chosen = rng.choice(
        len(grid),
        size=min(n_candidates, len(grid)),
        replace=False,
    )

    candidates = [
        ACBKTHyper(*grid[int(index)])
        for index in chosen
    ]

    default = ACBKTHyper()
    if default not in candidates:
        candidates[0] = default

    return candidates


def select_hyperparameters(
    prepared: PreparedData,
    train_cfg: TrainConfig,
    search_cfg: SearchConfig,
) -> Tuple[ACBKTHyper, pd.DataFrame]:
    """
    Two-stage validation-only search:
    1. screen n_candidates with shorter training;
    2. fully retrain the best n_finalists;
    3. select the finalist with highest validation AUC.
    """
    candidates = generate_hyperparameter_candidates(
        search_cfg.n_candidates,
        search_cfg.seed,
    )

    rows: List[Dict[str, object]] = []
    screening_results: List[
        Tuple[float, ACBKTHyper]
    ] = []

    for candidate_id, hp in enumerate(candidates):
        fit = fit_acbkt(
            prepared,
            hp,
            train_cfg,
            seed=search_cfg.seed,
            epochs_override=search_cfg.screening_epochs,
            patience_override=search_cfg.screening_patience,
        )

        metrics, _ = score_on_split(
            fit.model,
            prepared.val_batches,
            "val",
        )

        selection_value = (
            -np.inf
            if np.isnan(metrics["auc"])
            else float(metrics["auc"])
        )

        screening_results.append(
            (selection_value, hp)
        )
        rows.append(
            {
                "stage": "screening",
                "candidate_id": candidate_id,
                **asdict(hp),
                **metrics,
                "seconds": fit.seconds,
                "best_epoch": fit.best_epoch,
            }
        )

    screening_results.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    finalists = [
        hp
        for _, hp in screening_results[
            : search_cfg.n_finalists
        ]
    ]

    best_hp = None
    best_auc = -np.inf

    for finalist_id, hp in enumerate(finalists):
        fit = fit_acbkt(
            prepared,
            hp,
            train_cfg,
            seed=search_cfg.seed,
        )

        metrics, _ = score_on_split(
            fit.model,
            prepared.val_batches,
            "val",
        )

        validation_auc = (
            -np.inf
            if np.isnan(metrics["auc"])
            else float(metrics["auc"])
        )

        rows.append(
            {
                "stage": "refinement",
                "candidate_id": finalist_id,
                **asdict(hp),
                **metrics,
                "seconds": fit.seconds,
                "best_epoch": fit.best_epoch,
            }
        )

        if validation_auc > best_auc:
            best_auc = validation_auc
            best_hp = hp

    if best_hp is None:
        raise RuntimeError(
            "Hyperparameter selection failed."
        )

    return best_hp, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Dataset configurations from the source project
# ---------------------------------------------------------------------------

DATASETS = {
    "ASSISTments2015": DataConfig(
        path="2015_100_skill_builders_main_problems.csv",
        user_col="user_id",
        skill_col="sequence_id",
        correct_col="correct",
        order_col="log_id",
        sep=",",
        encoding="latin1",
        min_seq_len=5,
    ),
    "ASSISTments2009": DataConfig(
        path="non_skill_builder_data_new.csv",
        user_col="user_id",
        skill_col="skill_id",
        correct_col="correct",
        order_col="order_id",
        sep=",",
        encoding="latin1",
        min_seq_len=5,
    ),
    "Junyi": DataConfig(
        path="junyi/Log_Problem.csv",
        user_col="uuid",
        skill_col="ucid",
        correct_col="is_correct",
        order_col="timestamp_TW",
        sep=",",
        encoding="utf-8",
        min_seq_len=15,
        correct_map={"TRUE": 1, "FALSE": 0},
        order_is_datetime=True,
    ),
    "Algebra2005": DataConfig(
        path="algebra_2005_2006.txt",
        user_col="Anon Student Id",
        skill_col="KC(Default)",
        correct_col="Correct First Attempt",
        order_col="Row",
        sep="\t",
        encoding="utf-8",
        min_seq_len=8,
        multi_skill_sep="~~",
    ),
}


TRAIN_CONFIG = TrainConfig(
    epochs=200,
    lr=0.03,
    weight_decay=1e-5,
    patience=20,
    train_batch_size=1024,
    eval_batch_size=2048,
)

SEARCH_CONFIG = SearchConfig(
    n_candidates=24,
    screening_epochs=50,
    screening_patience=7,
    n_finalists=4,
    seed=2026,
)

PUBLICATION_SEEDS = [11, 22, 33, 44, 55]


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def parse_split(split_name: str) -> Tuple[float, float]:
    mapping = {
        "60-20-20": (0.60, 0.20),
        "70-10-20": (0.70, 0.10),
        "80-10-10": (0.80, 0.10),
    }
    if split_name not in mapping:
        raise ValueError(
            f"Unknown split {split_name}. "
            f"Choose from {list(mapping)}."
        )
    return mapping[split_name]


def run_acbkt(
    dataset_name: str,
    split_name: str = "60-20-20",
    output_root: str = "results_acbkt_main",
    seeds: Sequence[int] = PUBLICATION_SEEDS,
) -> Dict[str, object]:
    if dataset_name not in DATASETS:
        raise KeyError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(DATASETS)}"
        )

    train_ratio, val_ratio = parse_split(split_name)
    data_cfg = replace(
        DATASETS[dataset_name],
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    if dataset_name == "ASSISTments2015" and split_name == "60-20-20":
        print(
            "\nNOTE: ASSISTments2015 showed clear sensitivity to the "
            "training allocation in our additional check. "
            "For this dataset, 70/10/20 is recommended. "
            "Use --split 70-10-20 for the recommended setting.\n"
        )

    output_dir = (
        Path(output_root) /
        dataset_name /
        split_name
    )
    output_dir.mkdir(
        parents=True, exist_ok=True
    )

    print(f"Device: {DEVICE}")
    print(f"Dataset: {dataset_name}")
    print(f"Split: {split_name}")

    interactions = load_interactions(data_cfg)
    split_df = chronological_split(
        interactions, data_cfg
    )

    split_counts = (
        split_df["split"]
        .value_counts()
        .reindex(["train", "val", "test"])
        .fillna(0)
        .astype(int)
    )
    total = int(split_counts.sum())

    print("\nInteractions by split:")
    for split, count in split_counts.items():
        print(
            f"  {split:5s}: {count:,} "
            f"({count / total:.2%})"
        )

    records = make_sequences(split_df)
    num_skills = int(
        split_df["skill_idx"].max()
    ) + 1

    prepared = prepare_data(
        records=records,
        num_skills=num_skills,
        train_batch_size=TRAIN_CONFIG.train_batch_size,
        eval_batch_size=TRAIN_CONFIG.eval_batch_size,
        seed=SEARCH_CONFIG.seed,
    )

    print(
        f"\nSequences: {len(records):,} | "
        f"Skills: {num_skills:,} | "
        f"Interactions: {len(split_df):,}"
    )

    # Validation-only hyperparameter selection.
    best_hp, search_table = select_hyperparameters(
        prepared,
        TRAIN_CONFIG,
        SEARCH_CONFIG,
    )

    print("\nSelected hyperparameters:")
    print(json.dumps(asdict(best_hp), indent=2))

    search_table.to_csv(
        output_dir / "hyperparameter_search.csv",
        index=False,
    )

    with open(
        output_dir / "selected_hyperparameters.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            asdict(best_hp),
            file,
            indent=2,
        )

    # Final repeated fits and held-out test evaluation.
    run_rows = []
    prediction_tables = []

    for seed in seeds:
        print(f"\nFinal training seed {seed} ...")

        fit = fit_acbkt(
            prepared,
            best_hp,
            TRAIN_CONFIG,
            seed=seed,
        )

        metrics, pred_df = score_on_split(
            fit.model,
            prepared.test_batches,
            "test",
        )

        run_rows.append(
            {
                "dataset": dataset_name,
                "split": split_name,
                "seed": seed,
                "auc": metrics["auc"],
                "rmse": metrics["rmse"],
                "n": metrics["n"],
                "best_epoch": fit.best_epoch,
                "best_val_nll": fit.best_val_nll,
                "seconds": fit.seconds,
            }
        )

        pred_df["seed"] = seed
        prediction_tables.append(pred_df)

        print(
            f"  AUC={metrics['auc']:.6f} | "
            f"RMSE={metrics['rmse']:.6f}"
        )

    runs = pd.DataFrame(run_rows)
    predictions = pd.concat(
        prediction_tables,
        ignore_index=True,
    )

    auc_mean = float(runs["auc"].mean())
    auc_std = float(
        runs["auc"].std(ddof=1)
        if len(runs) > 1
        else 0.0
    )
    rmse_mean = float(runs["rmse"].mean())
    rmse_std = float(
        runs["rmse"].std(ddof=1)
        if len(runs) > 1
        else 0.0
    )

    summary = pd.DataFrame(
        [
            {
                "dataset": dataset_name,
                "split": split_name,
                "n_seeds": len(runs),
                "auc_mean": auc_mean,
                "auc_std": auc_std,
                "rmse_mean": rmse_mean,
                "rmse_std": rmse_std,
                "test_n_per_seed": int(
                    runs["n"].iloc[0]
                ),
            }
        ]
    )

    runs.to_csv(
        output_dir / "test_runs.csv",
        index=False,
    )
    predictions.to_csv(
        output_dir / "test_predictions.csv",
        index=False,
    )
    summary.to_csv(
        output_dir / "metrics_summary.csv",
        index=False,
    )

    # Save the exact row-level split for reproducibility.
    split_df.to_csv(
        output_dir / "split_manifest.csv",
        index=False,
    )

    print("\nFinal held-out test metrics:")
    print(
        f"AUC  = {auc_mean:.6f} ± {auc_std:.6f}"
    )
    print(
        f"RMSE = {rmse_mean:.6f} ± {rmse_std:.6f}"
    )
    print(f"\nResults saved to: {output_dir.resolve()}")

    return {
        "best_hyperparameters": best_hp,
        "search": search_table,
        "runs": runs,
        "predictions": predictions,
        "summary": summary,
        "split_manifest": split_df,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate AC-BKT with a chronological "
            "train/validation/test split."
        )
    )

    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        default="ASSISTments2015",
        help="Dataset configuration to run.",
    )

    parser.add_argument(
        "--split",
        choices=[
            "60-20-20",
            "70-10-20",
            "80-10-10",
        ],
        default="60-20-20",
        help=(
            "Chronological split. Default is 60-20-20. "
            "For ASSISTments2015, 70-10-20 is recommended."
        ),
    )

    parser.add_argument(
        "--output-root",
        default="results_acbkt_main",
        help="Directory used to save metrics and predictions.",
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=PUBLICATION_SEEDS,
        help="Final-fit random seeds.",
    )

    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    run_acbkt(
        dataset_name=args.dataset,
        split_name=args.split,
        output_root=args.output_root,
        seeds=args.seeds,
    )
