"""Train/evaluate the 24-hour Direct-MLP with hourly rolling training and one forward encoder.

This is the clean forward-only ablation of the reverse-week rolling model. Training
uses all 145 hourly origins inside each decoder week, while validation and test keep
the same seven daily 00:00 origins. All losses, schedules, and metrics are unchanged.
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader, TensorDataset

from data_utils import get_data_oncor_load_weekly
from Model_24hr_MLP_forwardonly_rollingtrain import VariationalSeq2Seq_meta


EVAL_FORECAST_INDICES: Tuple[int, ...] = (0, 24, 48, 72, 96, 120, 144)
TRAIN_FORECAST_INDICES: Tuple[int, ...] = tuple(range(145))
# Backward-compatible name used by evaluation/reporting code.
FORECAST_INDICES: Tuple[int, ...] = EVAL_FORECAST_INDICES


@dataclass
class Config:
    encoder_mode: str
    run_name: str
    use_horizon_weighting: bool
    use_peak_loss: bool
    output_root: str = os.getenv(
        "ONCOR_OUTPUT_ROOT",
        "/content/drive/MyDrive/M2oE2_For_Zhe/oncor_reverseweek",
    )
    profile_ids_env: str = os.getenv("ONCOR_XFMR_IDS", "")
    split_years_env: str = os.getenv("ONCOR_SPLIT_YEARS", "")

    seed: int = int(os.getenv("ONCOR_SEED", "42"))
    batch_size: int = int(os.getenv("ONCOR_BATCH_SIZE", "16"))
    total_update_budget: int = int(os.getenv("ONCOR_TOTAL_UPDATES", "6000"))
    lr_warmup_update_budget: int = int(os.getenv("ONCOR_LR_WARMUP_UPDATES", "720"))
    kl_anneal_update_budget: int = int(os.getenv("ONCOR_KL_ANNEAL_UPDATES", "360"))
    moe_warmup_update_budget: int = int(os.getenv("ONCOR_MOE_WARMUP_UPDATES", "480"))
    lr_decay_update_1: int = int(os.getenv("ONCOR_LR_DECAY_UPDATE_1", "1800"))
    lr_decay_update_2: int = int(os.getenv("ONCOR_LR_DECAY_UPDATE_2", "2700"))
    lr_decay_update_3: int = int(os.getenv("ONCOR_LR_DECAY_UPDATE_3", "3600"))
    lr_decay_update_4: int = int(os.getenv("ONCOR_LR_DECAY_UPDATE_4", "4200"))
    continuation_start_update: int = int(os.getenv("ONCOR_CONTINUATION_START_UPDATE", "4800"))
    continuation_lr_switch_update: int = int(os.getenv("ONCOR_CONTINUATION_LR_SWITCH", "5400"))
    continuation_lr_1: float = float(os.getenv("ONCOR_CONTINUATION_LR_1", "1e-5"))
    continuation_lr_2: float = float(os.getenv("ONCOR_CONTINUATION_LR_2", "5e-6"))

    epochs: int = int(os.getenv("ONCOR_EPOCHS", "0"))
    warmup_epochs: int = int(os.getenv("ONCOR_LR_WARMUP_EPOCHS", "0"))
    kl_anneal_epochs: int = int(os.getenv("ONCOR_KL_ANNEAL_EPOCHS", "0"))
    moe_warmup_epochs: int = int(os.getenv("ONCOR_MOE_WARMUP_EPOCHS", "0"))
    base_lr: float = float(os.getenv("ONCOR_LR", "3e-4"))
    kl_weight: float = float(os.getenv("ONCOR_KL_WEIGHT", "1e-4"))
    weight_decay: float = float(os.getenv("ONCOR_WEIGHT_DECAY", "1e-4"))
    grad_clip_norm: float = float(os.getenv("ONCOR_GRAD_CLIP", "1.0"))
    num_workers: int = int(os.getenv("ONCOR_NUM_WORKERS", "0"))

    xprime_dim: int = 16
    hidden_dim: int = 96
    latent_dim: int = 64
    num_layers: int = 1
    output_len: int = 24
    input_dim: int = 1
    output_dim: int = 1
    dropout: float = 0.1
    mlp_hidden: int = 70
    reverse_hidden_dim: int = int(os.getenv("ONCOR_REVERSE_HIDDEN", "48"))
    fusion_bottleneck: int = int(os.getenv("ONCOR_FUSION_BOTTLENECK", "32"))
    residual_scale: float = float(os.getenv("ONCOR_RESIDUAL_SCALE", "0.1"))

    # Horizon-focused objective, applied to every active day-ahead origin.
    early_hours: int = int(os.getenv("ONCOR_EARLY_HOURS", "6"))
    early3_weight: float = float(os.getenv("ONCOR_EARLY3_WEIGHT", "2.0"))
    early6_weight: float = float(os.getenv("ONCOR_EARLY6_WEIGHT", "1.5"))
    mid_start_hour: int = int(os.getenv("ONCOR_MID_START_HOUR", "10"))
    mid_end_hour: int = int(os.getenv("ONCOR_MID_END_HOUR", "16"))
    mid_weight: float = float(os.getenv("ONCOR_MID_WEIGHT", "1.3"))
    horizon_loss_weight: float = float(os.getenv("ONCOR_HORIZON_LOSS_WEIGHT", "2.0"))
    boundary_ramp_hours: int = int(os.getenv("ONCOR_BOUNDARY_RAMP_HOURS", "3"))
    boundary_ramp_weight: float = float(os.getenv("ONCOR_BOUNDARY_RAMP_WEIGHT", "0.5"))
    focus_score_early_weight: float = float(os.getenv("ONCOR_FOCUS_SCORE_EARLY_WEIGHT", "1.0"))
    focus_score_mid_weight: float = float(os.getenv("ONCOR_FOCUS_SCORE_MID_WEIGHT", "0.25"))

    # Warm-started top-k peak refinement.
    peak_top_k: int = int(os.getenv("ONCOR_PEAK_TOP_K", "3"))
    peak_magnitude_weight: float = float(os.getenv("ONCOR_PEAK_MAG_WEIGHT", "0.15"))
    peak_under_weight: float = float(os.getenv("ONCOR_PEAK_UNDER_WEIGHT", "0.10"))
    peak_warmup_start: int = int(os.getenv("ONCOR_PEAK_WARMUP_START", "4800"))
    peak_warmup_end: int = int(os.getenv("ONCOR_PEAK_WARMUP_END", "5400"))
    true_peak_under_weight: float = float(
        os.getenv("ONCOR_TRUE_PEAK_UNDER_WEIGHT", "0.50")
    )
    peak_selection_under_weight: float = float(
        os.getenv("ONCOR_PEAK_SELECTION_UNDER_WEIGHT", "0.5")
    )
    target_peak_selection_under_weight: float = float(
        os.getenv("ONCOR_TARGET_PEAK_SELECTION_UNDER_WEIGHT", "1.0")
    )
    balanced_peak_weight: float = float(os.getenv("ONCOR_BALANCED_PEAK_WEIGHT", "0.45"))

    # Final latent Monte Carlo evaluation. Checkpoint selection stays deterministic.
    mc_latent_samples: int = int(os.getenv("ONCOR_MC_LATENT_SAMPLES", "20"))
    mc_seed: int = int(os.getenv("ONCOR_MC_SEED", "20260714"))
    mc_antithetic: bool = os.getenv("ONCOR_MC_ANTITHETIC", "1") == "1"
    mc_checkpoints_env: str = os.getenv(
        "ONCOR_MC_CHECKPOINTS",
        "mse,focus,balanced,peak",
    )
    save_mc_scenarios: bool = os.getenv("ONCOR_SAVE_MC_SCENARIOS", "1") == "1"

    logvar_min: float = -10.0
    logvar_max: float = -3.8
    top_k: int = int(os.getenv("ONCOR_TOP_K", "4"))

    steps_per_epoch: int = 0

    force_retrain: bool = os.getenv("ONCOR_FORCE_RETRAIN", "0") == "1"
    eval_only: bool = os.getenv("ONCOR_EVAL_ONLY", "0") == "1"
    save_predictions: bool = os.getenv("ONCOR_SAVE_PREDICTIONS", "1") == "1"
    save_latest_full: bool = os.getenv("ONCOR_SAVE_LATEST_FULL", "1") == "1"
    require_midnight_origins: bool = os.getenv("ONCOR_REQUIRE_MIDNIGHT_ORIGINS", "1") == "1"
    resume_checkpoint_env: str = os.getenv("ONCOR_RESUME_CHECKPOINT", "")
    require_resume: bool = os.getenv("ONCOR_REQUIRE_RESUME", "0") == "1"

    def profile_ids(self) -> Optional[List[str]]:
        ids = [v.strip() for v in self.profile_ids_env.split(",") if v.strip()]
        return ids or None

    def requested_years(self) -> Optional[List[int]]:
        years = [int(v.strip()) for v in self.split_years_env.split(",") if v.strip()]
        return sorted(set(years)) or None

    def mc_checkpoint_names(self) -> List[str]:
        return [v.strip() for v in self.mc_checkpoints_env.split(",") if v.strip()]

    def horizon_weights(self) -> Tuple[float, ...]:
        if self.output_len != 24:
            raise ValueError("The default horizon weighting assumes output_len=24")
        if not (1 <= self.early_hours <= self.output_len):
            raise ValueError("early_hours must be in [1, output_len]")
        if not (1 <= self.mid_start_hour <= self.mid_end_hour <= self.output_len):
            raise ValueError("mid-hour bounds must satisfy 1 <= start <= end <= output_len")
        weights = np.ones(self.output_len, dtype=float)
        weights[: min(3, self.early_hours)] = self.early3_weight
        if self.early_hours > 3:
            weights[3:self.early_hours] = self.early6_weight
        weights[self.mid_start_hour - 1:self.mid_end_hour] = np.maximum(
            weights[self.mid_start_hour - 1:self.mid_end_hour], self.mid_weight
        )
        weights /= weights.mean()
        return tuple(float(v) for v in weights)

    def validate(self) -> None:
        if self.total_update_budget <= 0:
            raise ValueError("ONCOR_TOTAL_UPDATES must be positive")
        if not (
            0 < self.lr_warmup_update_budget
            < self.lr_decay_update_1
            < self.lr_decay_update_2
            < self.lr_decay_update_3
            < self.lr_decay_update_4
            < self.continuation_start_update
            < self.continuation_lr_switch_update
            < self.total_update_budget
        ):
            raise ValueError(
                "Require 0 < warmup < decay1 < decay2 < decay3 < decay4 "
                "< continuation_start < continuation_switch < total updates"
            )
        if not (1 <= self.peak_top_k <= self.output_len):
            raise ValueError("ONCOR_PEAK_TOP_K must be in [1, output_len]")
        if not (0 <= self.peak_warmup_start < self.peak_warmup_end):
            raise ValueError("Require 0 <= peak warmup start < peak warmup end")
        if self.mc_latent_samples <= 0:
            raise ValueError("ONCOR_MC_LATENT_SAMPLES must be positive")
        if self.mc_antithetic and self.mc_latent_samples % 2 != 0:
            raise ValueError("Antithetic MC requires an even number of latent samples")


# ============================================================
# Reproducibility and data adapters
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_update_schedule(cfg: Config, steps_per_epoch: int) -> None:
    """Convert the optimizer-update budget into a maximum epoch count."""
    if steps_per_epoch <= 0:
        raise ValueError(f"steps_per_epoch must be positive, got {steps_per_epoch}")
    cfg.steps_per_epoch = int(steps_per_epoch)
    cfg.validate()

    def to_epochs(update_budget: int) -> int:
        return max(1, int(math.ceil(float(update_budget) / steps_per_epoch)))

    if cfg.epochs <= 0:
        cfg.epochs = to_epochs(cfg.total_update_budget)
    if cfg.warmup_epochs <= 0:
        cfg.warmup_epochs = to_epochs(cfg.lr_warmup_update_budget)
    if cfg.kl_anneal_epochs <= 0:
        cfg.kl_anneal_epochs = to_epochs(cfg.kl_anneal_update_budget)
    if cfg.moe_warmup_epochs <= 0:
        cfg.moe_warmup_epochs = to_epochs(cfg.moe_warmup_update_budget)

    print(
        "[SCHEDULE] "
        f"steps/epoch={cfg.steps_per_epoch}, epochs={cfg.epochs} "
        f"(~{cfg.epochs * cfg.steps_per_epoch} updates), "
        f"full_budget={cfg.total_update_budget}, "
        f"lr_warmup={cfg.lr_warmup_update_budget}, "
        f"lr_decay=({cfg.lr_decay_update_1},{cfg.lr_decay_update_2},"
        f"{cfg.lr_decay_update_3},{cfg.lr_decay_update_4}), "
        f"continuation=({cfg.continuation_start_update},"
        f"{cfg.continuation_lr_switch_update}; "
        f"{cfg.continuation_lr_1:g},{cfg.continuation_lr_2:g}), "
        f"kl_anneal={cfg.kl_anneal_update_budget}, "
        f"moe_warmup={cfg.moe_warmup_update_budget}"
    )

def _ensure_week_matrix(arr, name: str, dtype=None):
    a = np.asarray(arr, dtype=object if dtype is None else dtype)
    if a.ndim == 2 and a.shape[1] == 168:
        return np.asarray(a, dtype=dtype) if dtype is not None else a
    if a.ndim == 1:
        rows = [np.asarray(x, dtype=dtype).reshape(-1) for x in a]
        if not rows or any(len(x) != 168 for x in rows):
            raise ValueError(f"{name} must contain exactly 168 values per week")
        return np.stack(rows, axis=0)
    raise ValueError(f"{name} must be [N,168], got {a.shape}")


def split_merged_profiles(times, load, temp, workday, season) -> List[Dict]:
    times_w = _ensure_week_matrix(times, "times")
    load_w = _ensure_week_matrix(load, "load", float)
    temp_w = _ensure_week_matrix(temp, "temp", float)
    wd_w = _ensure_week_matrix(workday, "workday", float)
    season_w = _ensure_week_matrix(season, "season", float)

    starts, ends = [], []
    for row in times_w:
        idx = pd.DatetimeIndex(pd.to_datetime(np.asarray(row).reshape(-1)))
        starts.append(idx[0])
        ends.append(idx[-1])
    boundaries = [0]
    for i in range(1, len(times_w)):
        if starts[i] <= ends[i - 1]:
            boundaries.append(i)
    boundaries.append(len(times_w))

    profiles = []
    for j, (a, b) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        profiles.append({
            "name": f"profile_{j}",
            "times": times_w[a:b],
            "load": load_w[a:b],
            "temp": temp_w[a:b],
            "workday": wd_w[a:b],
            "season": season_w[a:b],
        })
        print(f"[PROFILE] profile_{j}: weeks={b-a}, range={starts[a]} -> {ends[b-1]}")
    return profiles


def load_profiles(profile_ids: Optional[Sequence[str]]) -> List[Dict]:
    if profile_ids:
        out = []
        for pid in profile_ids:
            t, l, temp, wd, season = get_data_oncor_load_weekly(XFMR=str(pid))
            out.append({
                "name": str(pid),
                "times": _ensure_week_matrix(t, f"times[{pid}]"),
                "load": _ensure_week_matrix(l, f"load[{pid}]", float),
                "temp": _ensure_week_matrix(temp, f"temp[{pid}]", float),
                "workday": _ensure_week_matrix(wd, f"workday[{pid}]", float),
                "season": _ensure_week_matrix(season, f"season[{pid}]", float),
            })
            print(f"[PROFILE] {pid}: weeks={out[-1]['load'].shape[0]}")
        return out

    print("[PROFILE] ONCOR_XFMR_IDS is empty; recovering profiles from XFMR='all'.")
    return split_merged_profiles(*get_data_oncor_load_weekly(XFMR="all"))


def _future_path(values: np.ndarray, horizon: int, include_current: bool) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    start = 0 if include_current else 1
    offsets = np.arange(start, start + horizon)[None, :]
    indices = np.clip(np.arange(len(values))[:, None] + offsets, 0, len(values) - 1)
    return values[indices]


def build_frame(profile: Dict, output_len: int) -> pd.DataFrame:
    index = pd.DatetimeIndex(pd.to_datetime(np.asarray(profile["times"]).reshape(-1)))
    load = np.asarray(profile["load"], dtype=float).reshape(-1)
    temp = np.asarray(profile["temp"], dtype=float).reshape(-1)
    workday = np.asarray(profile["workday"], dtype=float).reshape(-1)
    if not (len(index) == len(load) == len(temp) == len(workday)):
        raise ValueError(f"Length mismatch for {profile['name']}")
    if index.has_duplicates:
        raise ValueError(f"Duplicate timestamps for {profile['name']}")
    if not np.isfinite(load).all() or not np.isfinite(temp).all():
        raise ValueError(f"Non-finite load/temperature for {profile['name']}")

    df = pd.DataFrame({"load": load, "temp": temp, "workday": workday}, index=index)
    angle = 2.0 * np.pi * (index.month.to_numpy() - 1) / 12.0
    df["month_sin"] = np.sin(angle)
    df["month_cos"] = np.cos(angle)
    temp_path = _future_path(temp, output_len, include_current=False)
    for h in range(output_len):
        df[f"temp_fc_tplus{h:02d}"] = temp_path[:, h]
    wd_path = _future_path(workday, output_len, include_current=False)
    df["workday_future24_mean"] = wd_path.mean(axis=1)
    return df


def infer_split_years(frames: Dict[str, pd.DataFrame], requested: Optional[List[int]]) -> List[int]:
    if requested:
        return requested
    complete_by_profile = []
    for df in frames.values():
        years = set()
        for year, group in df.groupby(df.index.year):
            if set(range(1, 13)).issubset(set(group.index.month.unique())):
                years.add(int(year))
        complete_by_profile.append(years)
    common = set.intersection(*complete_by_profile) if complete_by_profile else set()
    if common:
        return sorted(common)
    # Fallback: use every observed year and let origin masks remove unavailable months.
    return sorted(set(int(y) for df in frames.values() for y in df.index.year.unique()))


def _inverse_params(scaler: MinMaxScaler) -> Tuple[float, float]:
    lo, hi = float(scaler.data_min_[0]), float(scaler.data_max_[0])
    fr_lo, fr_hi = map(float, scaler.feature_range)
    scale = (hi - lo) / max(fr_hi - fr_lo, 1e-12)
    shift = lo - fr_lo * scale
    return scale, shift


def _classify_target(index_24: pd.DatetimeIndex, split_years: set) -> Optional[str]:
    """Classify a contiguous 24h target without discarding Jan-Oct month crossings."""
    if len(index_24) != 24:
        return None
    values = pd.DatetimeIndex(index_24).to_numpy(dtype="datetime64[ns]")
    if not np.all(np.diff(values) == np.timedelta64(1, "h")):
        return None
    years = set(int(v) for v in index_24.year)
    if len(years) != 1 or next(iter(years)) not in split_years:
        return None
    months = index_24.month.to_numpy()
    if np.all((months >= 1) & (months <= 10)):
        return "train"
    if np.all(months == 11):
        return "val"
    if np.all(months == 12):
        return "test"
    return None


def prepare_data(profiles: List[Dict], cfg: Config, device: torch.device):
    profile_names = [str(p["name"]) for p in profiles]
    if len(set(profile_names)) != len(profile_names):
        raise ValueError("Profile names must be unique")
    profile_to_idx = {name: i for i, name in enumerate(profile_names)}
    profile_keys = [f"profile_onehot_{i:02d}" for i in range(len(profile_names))]

    frames = {str(p["name"]): build_frame(p, cfg.output_len) for p in profiles}
    for name, df in frames.items():
        idx = profile_to_idx[name]
        for j, key in enumerate(profile_keys):
            df[key] = 1.0 if j == idx else 0.0

    split_years = infer_split_years(frames, cfg.requested_years())
    split_year_set = set(split_years)
    print(f"[SPLIT] years={split_years}; train=months 1-10, val=11, test=12")

    temp_keys = ["temp"] + [f"temp_fc_tplus{h:02d}" for h in range(cfg.output_len)]
    workday_keys = ["workday", "workday_future24_mean"]
    month_keys = ["month_sin", "month_cos"]
    ext_keys = temp_keys + workday_keys + month_keys + profile_keys
    offset = 0
    expert_specs = []
    for name, keys in (
        ("temp", temp_keys),
        ("workday", workday_keys),
        ("month", month_keys),
        ("profile", profile_keys),
    ):
        expert_specs.append({"name": name, "indices": list(range(offset, offset + len(keys)))})
        offset += len(keys)

    # Leakage-free scalers: strictly Jan-Oct rows in selected years.
    load_scalers: Dict[str, MinMaxScaler] = {}
    load_inverse: Dict[str, Tuple[float, float]] = {}
    scaler_meta = {"load": {}, "temperature": None}
    for name, df in frames.items():
        mask = df.index.year.isin(split_years) & (df.index.month <= 10)
        if not mask.any():
            raise RuntimeError(f"No Jan-Oct scaler rows for {name}")
        scaler = MinMaxScaler().fit(df.loc[mask, ["load"]].to_numpy(dtype=float))
        load_scalers[name] = scaler
        load_inverse[name] = _inverse_params(scaler)
        scaler_meta["load"][name] = {
            "fit_scope": "selected years, January-October only",
            "data_min": float(scaler.data_min_[0]),
            "data_max": float(scaler.data_max_[0]),
        }
        print(f"[SCALER][LOAD] {name}: min={scaler.data_min_[0]:.6f}, max={scaler.data_max_[0]:.6f}")

    temp_fit = []
    for df in frames.values():
        mask = df.index.year.isin(split_years) & (df.index.month <= 10)
        temp_fit.append(df.loc[mask, ["temp"]].to_numpy(dtype=float))
    temp_scaler = MinMaxScaler().fit(np.concatenate(temp_fit, axis=0))
    scaler_meta["temperature"] = {
        "fit_scope": "shared selected years, January-October current temperature only",
        "data_min": float(temp_scaler.data_min_[0]),
        "data_max": float(temp_scaler.data_max_[0]),
    }
    print(f"[SCALER][TEMP] shared min={temp_scaler.data_min_[0]:.6f}, max={temp_scaler.data_max_[0]:.6f}")

    processed: Dict[str, pd.DataFrame] = {}
    for name, df in frames.items():
        z = pd.DataFrame(index=df.index)
        z["load"] = load_scalers[name].transform(df[["load"]].to_numpy(dtype=float)).reshape(-1)
        for key in temp_keys:
            z[key] = temp_scaler.transform(df[[key]].to_numpy(dtype=float)).reshape(-1)
        for key in workday_keys + month_keys + profile_keys:
            z[key] = df[key].to_numpy(dtype=float)
        processed[name] = z

    stores = {
        split: {
            "enc_l": [], "enc_ext": [], "dec_l": [], "dec_ext": [],
            "target": [], "origin_mask": [], "load_scale": [], "load_shift": [],
            "metadata": [],
        }
        for split in ("train", "val", "test")
    }

    for profile in profiles:
        name = str(profile["name"])
        df = processed[name]
        raw_df = frames[name]
        n_weeks = np.asarray(profile["load"]).shape[0]
        created = {"train": 0, "val": 0, "test": 0}

        for week in range(n_weeks - 1):
            enc_start = week * 168
            enc_end = enc_start + 168
            dec_start = enc_end
            dec_end = dec_start + 168
            if dec_end > len(df):
                continue
            full_index = df.index[enc_start:dec_end]
            expected = pd.date_range(full_index[0], periods=336, freq="h")
            if len(full_index) != 336 or not full_index.equals(expected):
                continue

            dec_index = df.index[dec_start:dec_end]
            origin_splits = []
            origin_times = []
            for origin in TRAIN_FORECAST_INDICES:
                target_index = dec_index[origin:origin + cfg.output_len]
                origin_splits.append(_classify_target(target_index, split_year_set))
                origin_times.append(target_index[0] if len(target_index) else pd.NaT)

            if not any(v is not None for v in origin_splits):
                continue

            if cfg.require_midnight_origins:
                eval_times = [origin_times[i] for i in EVAL_FORECAST_INDICES]
                non_midnight = [
                    ts for ts in eval_times
                    if not pd.isna(ts)
                    and (ts.hour != 0 or ts.minute != 0 or ts.second != 0)
                ]
                if non_midnight:
                    raise ValueError(
                        f"Evaluation origins must be daily 00:00 for {name}; "
                        f"found {non_midnight[:3]}. Rolling training origins are "
                        "intentionally hourly."
                    )

            enc = df.iloc[enc_start:enc_end]
            dec = df.iloc[dec_start:dec_end]
            target = np.stack([
                dec["load"].to_numpy(dtype=float)[origin:origin + cfg.output_len]
                for origin in TRAIN_FORECAST_INDICES
            ], axis=0)[:, :, None]
            enc_l = enc["load"].to_numpy(dtype=float)[:, None]
            enc_ext = enc[ext_keys].to_numpy(dtype=float)
            dec_l = dec["load"].to_numpy(dtype=float)[:144, None]
            dec_ext = dec[ext_keys].to_numpy(dtype=float)[:144]
            scale, shift = load_inverse[name]

            for split in ("train", "val", "test"):
                mask = np.asarray([v == split for v in origin_splits], dtype=np.float32)
                if mask.sum() == 0:
                    continue
                s = stores[split]
                s["enc_l"].append(enc_l)
                s["enc_ext"].append(enc_ext)
                s["dec_l"].append(dec_l)
                s["dec_ext"].append(dec_ext)
                s["target"].append(target)
                s["origin_mask"].append(mask)
                s["load_scale"].append([scale])
                s["load_shift"].append([shift])
                s["metadata"].append({
                    "profile": name,
                    "week": int(week),
                    "encoder_start": str(enc.index[0]),
                    "encoder_end": str(enc.index[-1]),
                    "decoder_start": str(dec.index[0]),
                    "decoder_end": str(dec.index[-1]),
                    "origin_times": [str(v) for v in origin_times],
                    "active_origins": [int(i) for i, active in enumerate(mask) if active > 0],
                })
                created[split] += 1
        print(f"[WINDOWS] {name}: {created}")

    def pack(split: str):
        s = stores[split]
        if not s["enc_l"]:
            raise RuntimeError(f"No samples generated for split={split}")
        tensor = lambda values: torch.tensor(np.asarray(values), dtype=torch.float32, device=device)
        out = {
            "enc_l": tensor(s["enc_l"]),
            "enc_ext": tensor(s["enc_ext"]),
            "dec_l": tensor(s["dec_l"]),
            "dec_ext": tensor(s["dec_ext"]),
            "target": tensor(s["target"]),
            "origin_mask": tensor(s["origin_mask"]),
            "load_scale": tensor(s["load_scale"]),
            "load_shift": tensor(s["load_shift"]),
            "metadata": s["metadata"],
        }
        active = int(out["origin_mask"].sum().item())
        print(
            f"[{split.upper()}] weekly contexts={len(s['metadata'])}, active forecast origins={active}, "
            f"points={active * cfg.output_len}"
        )
        print(
            f"  enc={tuple(out['enc_l'].shape)}, dec={tuple(out['dec_l'].shape)}, "
            f"target={tuple(out['target'].shape)}, ext_dim={out['enc_ext'].shape[-1]}"
        )
        return out

    tensors = {split: pack(split) for split in ("train", "val", "test")}
    metadata_rows = []
    for split, s in stores.items():
        for row in s["metadata"]:
            metadata_rows.append({"split": split, **row})

    return {
        "tensors": tensors,
        "ext_keys": ext_keys,
        "expert_specs": expert_specs,
        "profile_names": profile_names,
        "profile_to_idx": profile_to_idx,
        "split_years": split_years,
        "scaler_meta": scaler_meta,
        "metadata": pd.DataFrame(metadata_rows),
    }


def make_loader(
    data: Dict,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: Optional[int] = None,
) -> DataLoader:
    dataset = TensorDataset(
        data["enc_l"], data["enc_ext"], data["dec_l"], data["dec_ext"],
        data["target"], data["origin_mask"], data["load_scale"], data["load_shift"],
    )
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(seed if seed is not None else 0))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        generator=generator,
    )


# ============================================================
# Losses and metrics
# ============================================================

def masked_mean(values: torch.Tensor, origin_mask: torch.Tensor) -> torch.Tensor:
    mask = origin_mask[:, :, None, None].to(values.dtype).expand_as(values)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def gaussian_nll_masked(mu, logvar, target, origin_mask):
    nll = 0.5 * (
        logvar + math.log(2.0 * math.pi)
        + (target - mu).pow(2) / (torch.exp(logvar) + 1e-12)
    )
    return masked_mean(nll, origin_mask)


def all_origin_horizon_weighted_mse(
    mu: torch.Tensor,
    target: torch.Tensor,
    origin_mask: torch.Tensor,
    horizon_weights: Sequence[float],
) -> torch.Tensor:
    """Weighted MSE over all active origins and all 24 forecast horizons."""
    if len(horizon_weights) != mu.size(2):
        raise ValueError(
            f"Expected {mu.size(2)} horizon weights, got {len(horizon_weights)}"
        )
    w = mu.new_tensor(list(horizon_weights)).view(1, 1, mu.size(2), 1)
    mask = origin_mask[:, :, None, None].to(mu.dtype)
    err2 = (mu - target).pow(2)
    denom = origin_mask.sum() * w.sum() * mu.size(-1)
    return (err2 * w * mask).sum() / denom.clamp_min(1.0)


def all_origin_boundary_ramp_mse(
    mu: torch.Tensor,
    target: torch.Tensor,
    enc_last: torch.Tensor,
    dec_l: torch.Tensor,
    origin_mask: torch.Tensor,
    forecast_indices: Sequence[int],
    hours: int,
) -> torch.Tensor:
    """Match the boundary-to-forecast ramp for every active forecast origin."""
    k = min(int(hours), mu.size(2))
    if k <= 0:
        return mu.new_zeros(())
    if len(forecast_indices) != mu.size(1):
        raise ValueError("forecast_indices must align with the origin dimension")

    previous = []
    for origin in forecast_indices:
        if int(origin) == 0:
            previous.append(enc_last)
        else:
            previous.append(dec_l[:, int(origin) - 1, :])
    previous = torch.stack(previous, dim=1)  # [B,O,D]

    pred_path = torch.cat([previous.unsqueeze(2), mu[:, :, :k, :]], dim=2)
    true_path = torch.cat([previous.unsqueeze(2), target[:, :, :k, :]], dim=2)
    pred_ramp = pred_path[:, :, 1:, :] - pred_path[:, :, :-1, :]
    true_ramp = true_path[:, :, 1:, :] - true_path[:, :, :-1, :]
    mask = origin_mask[:, :, None, None].to(mu.dtype)
    denom = origin_mask.sum() * k * mu.size(-1)
    return ((pred_ramp - true_ramp).pow(2) * mask).sum() / denom.clamp_min(1.0)


def weighted_kl(mu_z, logvar_z, origin_mask):
    lv = torch.clamp(logvar_z, -10.0, 10.0)
    per_sample = -0.5 * torch.sum(1.0 + lv - mu_z.pow(2) - lv.exp(), dim=1)
    weights = origin_mask.sum(dim=1)
    return (per_sample * weights).sum() / weights.sum().clamp_min(1.0)



def topk_peak_losses(mu, target, origin_mask, top_k: int):
    """Normalized top-k peak magnitude and asymmetric underprediction losses."""
    k = min(int(top_k), int(mu.size(2)))
    pred_peak = torch.topk(mu[..., 0], k=k, dim=2).values.mean(dim=2)
    true_peak = torch.topk(target[..., 0], k=k, dim=2).values.mean(dim=2)
    active = origin_mask > 0
    if not active.any():
        zero = mu.new_zeros(())
        return zero, zero
    diff = pred_peak[active] - true_peak[active]
    magnitude = diff.pow(2).mean()
    under = torch.relu(-diff).pow(2).mean()
    return magnitude, under


def target_aligned_peak_losses(mu, target, origin_mask, top_k: int):
    """Losses evaluated at the target curve's own top-k peak hours."""
    k = min(int(top_k), int(target.size(2)))
    true_values, true_indices = torch.topk(target[..., 0], k=k, dim=2)
    pred_at_true_peak = torch.gather(mu[..., 0], dim=2, index=true_indices)
    active = origin_mask > 0
    if not active.any():
        zero = mu.new_zeros(())
        return zero, zero
    diff = pred_at_true_peak[active] - true_values[active]
    mse = diff.pow(2).mean()
    under = torch.relu(-diff).pow(2).mean()
    return mse, under


def peak_warm_factor(cfg: Config, update: int) -> float:
    if update <= cfg.peak_warmup_start:
        return 0.0
    if update >= cfg.peak_warmup_end:
        return 1.0
    return float(
        (update - cfg.peak_warmup_start)
        / max(cfg.peak_warmup_end - cfg.peak_warmup_start, 1)
    )


def _normal_absolute_moment(delta: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """E|N(delta, sigma^2)|, used by exact Gaussian-mixture CRPS."""
    sigma = sigma.clamp_min(1e-8)
    z = delta / sigma
    normal = Normal(torch.zeros_like(z), torch.ones_like(z))
    return 2.0 * sigma * torch.exp(normal.log_prob(z)) + delta * (
        2.0 * normal.cdf(z) - 1.0
    )


def gaussian_mixture_crps(
    component_mu: torch.Tensor,
    component_sigma: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Exact equally weighted Gaussian-mixture CRPS.

    component_mu/component_sigma: [S,B,O,H,D]
    target: [B,O,H,D]
    """
    first = _normal_absolute_moment(
        target.unsqueeze(0) - component_mu,
        component_sigma,
    ).mean(dim=0)
    delta = component_mu[:, None, ...] - component_mu[None, :, ...]
    pair_sigma = torch.sqrt(
        component_sigma[:, None, ...].pow(2)
        + component_sigma[None, :, ...].pow(2)
    )
    second = 0.5 * _normal_absolute_moment(delta, pair_sigma).mean(dim=(0, 1))
    return first - second


def _encode_once(
    model: nn.Module,
    enc_l: torch.Tensor,
    enc_ext: torch.Tensor,
    cfg: Config,
):
    selected = model.decoder._origins(FORECAST_INDICES, 144)
    mu_z, logvar_z, phase_features = model.encoder(
        enc_l,
        enc_ext,
        transform_block=model.transform_enc,
        phase_indices=selected,
        epoch=cfg.moe_warmup_update_budget + 1,
        top_k=cfg.top_k,
        warmup_epochs=cfg.moe_warmup_update_budget,
    )
    return selected, mu_z, logvar_z, phase_features


def _decode_from_z(
    model: nn.Module,
    enc_l: torch.Tensor,
    dec_l: torch.Tensor,
    dec_ext: torch.Tensor,
    z: torch.Tensor,
    phase_features: Optional[torch.Tensor],
    selected: Sequence[int],
    cfg: Config,
):
    return model.decoder(
        dec_l,
        dec_ext,
        z_latent=z,
        phase_features=phase_features,
        phase_fuser=model.encoder.fuse_phase if phase_features is not None else None,
        transform_block=model.transform_dec,
        initial_go=enc_l[:, -1, :],
        epoch=cfg.moe_warmup_update_budget + 1,
        top_k=cfg.top_k,
        warmup_epochs=cfg.moe_warmup_update_budget,
        forecast_indices=selected,
    )


def _latent_epsilons(
    samples: int,
    latent_shape: Sequence[int],
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    antithetic: bool,
) -> torch.Tensor:
    if antithetic:
        half = samples // 2
        eps_half = torch.randn(
            (half, *latent_shape),
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        )
        eps = torch.cat([eps_half, -eps_half], dim=0)
    else:
        eps = torch.randn(
            (samples, *latent_shape),
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        )
    return eps.to(device=device, dtype=dtype)

def gaussian_crps(mu, sigma, target):
    sigma = sigma.clamp_min(1e-8)
    z = (target - mu) / sigma
    normal = Normal(torch.zeros_like(z), torch.ones_like(z))
    phi = torch.exp(normal.log_prob(z))
    Phi = normal.cdf(z)
    return sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    metadata: List[Dict],
    cfg: Config,
    prediction_path: Optional[Path] = None,
    latent_mode: str = "mean",
    scenario_path: Optional[Path] = None,
):
    """Evaluate with z=mu or with antithetic latent Monte Carlo integration."""
    if latent_mode not in {"mean", "mc"}:
        raise ValueError("latent_mode must be 'mean' or 'mc'")
    model.eval()
    sums = {"mse": 0.0, "mae": 0.0, "nll": 0.0, "crps": 0.0}
    raw_abs_sum = raw_sq_sum = raw_y_abs_sum = 0.0
    point_count = 0
    peak_pct_sum = 0.0
    peak_under = peak_count = 0
    topk_peak_sq_norm = topk_under_sq_norm = 0.0
    topk_peak_sq_raw = topk_peak_abs_raw = 0.0
    target_peak_sq_norm = target_peak_under_sq_norm = 0.0
    target_peak_sq_raw = target_peak_abs_raw = 0.0
    target_peak_under_count = target_peak_point_count = 0
    peak_hour_abs_sum = 0.0
    scenario_peak_pct_sum = scenario_peak_under_sum = 0.0
    rows = []
    meta_offset = 0

    scenario_mu_norm_blocks = []
    scenario_target_norm_blocks = []
    scenario_scale_blocks = []
    scenario_shift_blocks = []
    scenario_profiles = []
    scenario_origin_times = []
    scenario_origin_offsets = []

    diag_weight = 0
    residual_ratio_sum = phase_feature_norm_sum = phase_day_diversity_sum = 0.0

    prefix_ks = (1, 3, 6, 24)
    prefix = {
        k: {"norm_sq": 0.0, "norm_abs": 0.0, "raw_sq": 0.0,
            "raw_abs": 0.0, "raw_y_abs": 0.0, "count": 0}
        for k in prefix_ks
    }
    mid = {"norm_sq": 0.0, "norm_abs": 0.0, "raw_sq": 0.0,
           "raw_abs": 0.0, "raw_y_abs": 0.0, "count": 0}

    n_origins = len(FORECAST_INDICES)
    H = cfg.output_len
    horizon_norm_sq = np.zeros(H, dtype=float)
    horizon_raw_sq = np.zeros(H, dtype=float)
    horizon_raw_abs = np.zeros(H, dtype=float)
    horizon_count = np.zeros(H, dtype=float)
    origin_raw_sq = np.zeros(n_origins, dtype=float)
    origin_raw_abs = np.zeros(n_origins, dtype=float)
    origin_count = np.zeros(n_origins, dtype=float)
    origin_first6_raw_sq = np.zeros(n_origins, dtype=float)
    origin_first6_count = np.zeros(n_origins, dtype=float)
    origin_target_peak_raw_sq = np.zeros(n_origins, dtype=float)
    origin_target_peak_under_count = np.zeros(n_origins, dtype=float)
    origin_target_peak_point_count = np.zeros(n_origins, dtype=float)
    origin_peak_hour_abs = np.zeros(n_origins, dtype=float)
    origin_peak_count = np.zeros(n_origins, dtype=float)
    matrix_raw_sq = np.zeros((n_origins, H), dtype=float)
    matrix_raw_abs = np.zeros((n_origins, H), dtype=float)
    matrix_count = np.zeros((n_origins, H), dtype=float)

    mc_generator = torch.Generator(device="cpu")
    mc_generator.manual_seed(cfg.mc_seed)

    for batch in loader:
        enc_l, enc_ext, dec_l, dec_ext, target, origin_mask, scale, shift = [
            x.to(device) for x in batch
        ]
        eval_positions = list(EVAL_FORECAST_INDICES)
        target = target[:, eval_positions, :, :]
        origin_mask = origin_mask[:, eval_positions]
        selected, mu_z, logvar_z, phase_features = _encode_once(
            model, enc_l, enc_ext, cfg
        )
        encoder_diagnostics = model.encoder.booster_diagnostics()
        batch_weight = int(enc_l.size(0))
        phase_feature_norm_sum += encoder_diagnostics["phase_feature_norm"] * batch_weight
        phase_day_diversity_sum += encoder_diagnostics["phase_day_diversity"] * batch_weight
        diag_weight += batch_weight
        batch_residual_ratio = 0.0

        if latent_mode == "mean":
            mu, logvar = _decode_from_z(
                model, enc_l, dec_l, dec_ext, mu_z,
                phase_features, selected, cfg,
            )
            sigma = torch.exp(0.5 * logvar)
            err = mu - target
            nll = 0.5 * (
                logvar + math.log(2.0 * math.pi)
                + err.pow(2) / (torch.exp(logvar) + 1e-12)
            )
            crps = gaussian_crps(mu, sigma, target)
            component_mu = None
            batch_residual_ratio = model.encoder.booster_diagnostics()["residual_ratio"]
        else:
            eps = _latent_epsilons(
                cfg.mc_latent_samples,
                mu_z.shape,
                mc_generator,
                device,
                mu_z.dtype,
                cfg.mc_antithetic,
            )
            std_z = torch.exp(0.5 * torch.clamp(logvar_z, -10.0, 10.0))
            component_mu_list = []
            component_logvar_list = []
            component_residual_ratios = []
            for s in range(cfg.mc_latent_samples):
                z_s = mu_z + std_z * eps[s]
                mu_s, logvar_s = _decode_from_z(
                    model, enc_l, dec_l, dec_ext, z_s,
                    phase_features, selected, cfg,
                )
                component_mu_list.append(mu_s)
                component_logvar_list.append(logvar_s)
                component_residual_ratios.append(
                    model.encoder.booster_diagnostics()["residual_ratio"]
                )
            batch_residual_ratio = float(np.mean(component_residual_ratios))
            component_mu = torch.stack(component_mu_list, dim=0)
            component_logvar = torch.stack(component_logvar_list, dim=0)
            component_var = torch.exp(component_logvar)
            component_sigma = torch.sqrt(component_var.clamp_min(1e-12))
            mu = component_mu.mean(dim=0)
            total_var = (
                component_var + component_mu.pow(2)
            ).mean(dim=0) - mu.pow(2)
            total_var = total_var.clamp_min(1e-12)
            logvar = torch.log(total_var)
            sigma = torch.sqrt(total_var)
            err = mu - target

            component_log_prob = -0.5 * (
                component_logvar
                + math.log(2.0 * math.pi)
                + (target.unsqueeze(0) - component_mu).pow(2)
                / (component_var + 1e-12)
            )
            nll = -(
                torch.logsumexp(component_log_prob, dim=0)
                - math.log(cfg.mc_latent_samples)
            )
            crps = gaussian_mixture_crps(component_mu, component_sigma, target)

        # Read the phase-fusion residual only after decoder fusion has executed.
        residual_ratio_sum += batch_residual_ratio * batch_weight

        mask = origin_mask[:, :, None, None].expand_as(mu).bool()
        n_points = int(mask.sum().item())
        point_count += n_points
        sums["mse"] += err.pow(2)[mask].sum().item()
        sums["mae"] += err.abs()[mask].sum().item()
        sums["nll"] += nll[mask].sum().item()
        sums["crps"] += crps[mask].sum().item()

        scale4 = scale[:, None, None, :]
        shift4 = shift[:, None, None, :]
        mu_raw = mu * scale4 + shift4
        y_raw = target * scale4 + shift4
        sigma_raw = sigma * scale4.abs()
        raw_err = mu_raw - y_raw
        raw_abs_sum += raw_err.abs()[mask].sum().item()
        raw_sq_sum += raw_err.pow(2)[mask].sum().item()
        raw_y_abs_sum += y_raw.abs()[mask].sum().item()

        active_mask = origin_mask > 0
        peak_k = min(cfg.peak_top_k, H)
        pred_topk_norm = torch.topk(mu[..., 0], k=peak_k, dim=2).values.mean(dim=2)
        true_topk_norm = torch.topk(target[..., 0], k=peak_k, dim=2).values.mean(dim=2)
        pred_topk_raw = torch.topk(mu_raw[..., 0], k=peak_k, dim=2).values.mean(dim=2)
        true_topk_raw = torch.topk(y_raw[..., 0], k=peak_k, dim=2).values.mean(dim=2)
        topk_diff_norm = pred_topk_norm[active_mask] - true_topk_norm[active_mask]
        topk_diff_raw = pred_topk_raw[active_mask] - true_topk_raw[active_mask]
        topk_peak_sq_norm += topk_diff_norm.pow(2).sum().item()
        topk_under_sq_norm += torch.relu(-topk_diff_norm).pow(2).sum().item()
        topk_peak_sq_raw += topk_diff_raw.pow(2).sum().item()
        topk_peak_abs_raw += topk_diff_raw.abs().sum().item()

        # Evaluate predictions at the target curve's own top-k peak hours.
        true_peak_values_norm, true_peak_indices = torch.topk(
            target[..., 0], k=peak_k, dim=2
        )
        pred_at_true_peak_norm = torch.gather(
            mu[..., 0], dim=2, index=true_peak_indices
        )
        true_peak_values_raw = torch.gather(
            y_raw[..., 0], dim=2, index=true_peak_indices
        )
        pred_at_true_peak_raw = torch.gather(
            mu_raw[..., 0], dim=2, index=true_peak_indices
        )
        target_diff_norm = (
            pred_at_true_peak_norm[active_mask] - true_peak_values_norm[active_mask]
        )
        target_diff_raw = (
            pred_at_true_peak_raw[active_mask] - true_peak_values_raw[active_mask]
        )
        target_peak_sq_norm += target_diff_norm.pow(2).sum().item()
        target_peak_under_sq_norm += torch.relu(-target_diff_norm).pow(2).sum().item()
        target_peak_sq_raw += target_diff_raw.pow(2).sum().item()
        target_peak_abs_raw += target_diff_raw.abs().sum().item()
        target_peak_under_count += int((target_diff_raw < 0).sum().item())
        target_peak_point_count += int(target_diff_raw.numel())

        pred_peak_hour = mu[..., 0].argmax(dim=2)
        true_peak_hour = target[..., 0].argmax(dim=2)
        peak_hour_abs_sum += (
            pred_peak_hour[active_mask] - true_peak_hour[active_mask]
        ).abs().sum().item()

        for k in prefix_ks:
            kk = min(k, H)
            m = active_mask[:, :, None, None].expand(-1, -1, kk, mu.size(-1))
            e_n = err[:, :, :kk, :][m]
            e_r = raw_err[:, :, :kk, :][m]
            y_r = y_raw[:, :, :kk, :][m]
            cnt = int(e_n.numel())
            prefix[k]["norm_sq"] += e_n.pow(2).sum().item()
            prefix[k]["norm_abs"] += e_n.abs().sum().item()
            prefix[k]["raw_sq"] += e_r.pow(2).sum().item()
            prefix[k]["raw_abs"] += e_r.abs().sum().item()
            prefix[k]["raw_y_abs"] += y_r.abs().sum().item()
            prefix[k]["count"] += cnt

        ms = cfg.mid_start_hour - 1
        me = cfg.mid_end_hour
        m_mid = active_mask[:, :, None, None].expand(-1, -1, me - ms, mu.size(-1))
        e_n_mid = err[:, :, ms:me, :][m_mid]
        e_r_mid = raw_err[:, :, ms:me, :][m_mid]
        y_r_mid = y_raw[:, :, ms:me, :][m_mid]
        mid["norm_sq"] += e_n_mid.pow(2).sum().item()
        mid["norm_abs"] += e_n_mid.abs().sum().item()
        mid["raw_sq"] += e_r_mid.pow(2).sum().item()
        mid["raw_abs"] += e_r_mid.abs().sum().item()
        mid["raw_y_abs"] += y_r_mid.abs().sum().item()
        mid["count"] += int(e_n_mid.numel())

        for h in range(H):
            e_n_h = err[:, :, h, :][active_mask]
            e_r_h = raw_err[:, :, h, :][active_mask]
            horizon_norm_sq[h] += e_n_h.pow(2).sum().item()
            horizon_raw_sq[h] += e_r_h.pow(2).sum().item()
            horizon_raw_abs[h] += e_r_h.abs().sum().item()
            horizon_count[h] += e_r_h.numel()

        for o in range(n_origins):
            active_o = active_mask[:, o]
            if not active_o.any():
                continue
            e_o = raw_err[active_o, o, :, :]
            origin_raw_sq[o] += e_o.pow(2).sum().item()
            origin_raw_abs[o] += e_o.abs().sum().item()
            origin_count[o] += e_o.numel()
            e_o6 = raw_err[active_o, o, :min(6, H), :]
            origin_first6_raw_sq[o] += e_o6.pow(2).sum().item()
            origin_first6_count[o] += e_o6.numel()
            target_diff_o = (
                pred_at_true_peak_raw[active_o, o, :]
                - true_peak_values_raw[active_o, o, :]
            )
            origin_target_peak_raw_sq[o] += target_diff_o.pow(2).sum().item()
            origin_target_peak_under_count[o] += (target_diff_o < 0).sum().item()
            origin_target_peak_point_count[o] += target_diff_o.numel()
            origin_peak_hour_abs[o] += (
                pred_peak_hour[active_o, o] - true_peak_hour[active_o, o]
            ).abs().sum().item()
            origin_peak_count[o] += int(active_o.sum().item())
            for h in range(H):
                e_oh = raw_err[active_o, o, h, :]
                matrix_raw_sq[o, h] += e_oh.pow(2).sum().item()
                matrix_raw_abs[o, h] += e_oh.abs().sum().item()
                matrix_count[o, h] += e_oh.numel()

        component_mu_raw = (
            component_mu * scale[None, :, None, None, :] + shift[None, :, None, None, :]
            if component_mu is not None else None
        )
        for b in range(mu.size(0)):
            meta = metadata[meta_offset + b]
            for origin_pos, origin in enumerate(FORECAST_INDICES):
                if origin_mask[b, origin_pos].item() <= 0:
                    continue
                true_peak = y_raw[b, origin_pos, :, 0].max()
                pred_peak = mu_raw[b, origin_pos, :, 0].max()
                peak_pct_sum += float(
                    (pred_peak - true_peak).abs()
                    / true_peak.abs().clamp_min(1e-6)
                )
                peak_under += int(pred_peak < true_peak)
                peak_count += 1

                if component_mu_raw is not None:
                    scenario_peaks = component_mu_raw[:, b, origin_pos, :, 0].max(dim=1).values
                    scenario_peak_pct_sum += float(
                        (
                            (scenario_peaks - true_peak).abs()
                            / true_peak.abs().clamp_min(1e-6)
                        ).mean()
                    )
                    scenario_peak_under_sum += float(
                        (scenario_peaks < true_peak).to(torch.float32).mean()
                    )

                start_time = pd.Timestamp(meta["origin_times"][origin])
                if prediction_path is not None:
                    for h in range(H):
                        rows.append({
                            "profile": meta["profile"],
                            "week": meta["week"],
                            "origin_index": origin_pos,
                            "origin_offset": origin,
                            "origin_time": str(start_time),
                            "horizon": h + 1,
                            "target_time": str(start_time + pd.Timedelta(hours=h)),
                            "target_norm": float(target[b, origin_pos, h, 0].cpu()),
                            "pred_norm": float(mu[b, origin_pos, h, 0].cpu()),
                            "sigma_norm": float(sigma[b, origin_pos, h, 0].cpu()),
                            "target_raw": float(y_raw[b, origin_pos, h, 0].cpu()),
                            "pred_raw": float(mu_raw[b, origin_pos, h, 0].cpu()),
                            "sigma_raw": float(sigma_raw[b, origin_pos, h, 0].cpu()),
                            "latent_mode": latent_mode,
                            "mc_samples": (
                                cfg.mc_latent_samples if latent_mode == "mc" else 0
                            ),
                        })

                if scenario_path is not None and component_mu is not None:
                    scenario_mu_norm_blocks.append(
                        component_mu[:, b, origin_pos, :, 0].detach().cpu().numpy()[:, None, :]
                    )
                    scenario_target_norm_blocks.append(
                        target[b, origin_pos, :, 0].detach().cpu().numpy()[None, :]
                    )
                    scenario_scale_blocks.append(
                        np.asarray([float(scale[b, 0].detach().cpu())], dtype=np.float32)
                    )
                    scenario_shift_blocks.append(
                        np.asarray([float(shift[b, 0].detach().cpu())], dtype=np.float32)
                    )
                    scenario_profiles.append(str(meta["profile"]))
                    scenario_origin_times.append(str(start_time))
                    scenario_origin_offsets.append(int(origin))
        meta_offset += mu.size(0)

    if point_count == 0:
        raise RuntimeError("Evaluation had zero active points")

    result = {
        "latent_mode": latent_mode,
        "mc_latent_samples": cfg.mc_latent_samples if latent_mode == "mc" else 0,
        "mc_seed": cfg.mc_seed if latent_mode == "mc" else None,
        "mc_antithetic": cfg.mc_antithetic if latent_mode == "mc" else False,
        "mse_norm": sums["mse"] / point_count,
        "rmse_norm": math.sqrt(sums["mse"] / point_count),
        "mae_norm": sums["mae"] / point_count,
        "nll_norm": sums["nll"] / point_count,
        "crps_norm": sums["crps"] / point_count,
        "rmse_raw": math.sqrt(raw_sq_sum / point_count),
        "mae_raw": raw_abs_sum / point_count,
        "wape_raw": raw_abs_sum / max(raw_y_abs_sum, 1e-12),
        "pvpe24": peak_pct_sum / max(peak_count, 1),
        "under24": peak_under / max(peak_count, 1),
        "topk_peak_mse_norm": topk_peak_sq_norm / max(peak_count, 1),
        "topk_peak_under_mse_norm": topk_under_sq_norm / max(peak_count, 1),
        "topk_peak_rmse_raw": math.sqrt(topk_peak_sq_raw / max(peak_count, 1)),
        "topk_peak_mae_raw": topk_peak_abs_raw / max(peak_count, 1),
        "target_peak_mse_norm": target_peak_sq_norm / max(target_peak_point_count, 1),
        "target_peak_under_mse_norm": (
            target_peak_under_sq_norm / max(target_peak_point_count, 1)
        ),
        "target_peak_rmse_raw": math.sqrt(
            target_peak_sq_raw / max(target_peak_point_count, 1)
        ),
        "target_peak_mae_raw": (
            target_peak_abs_raw / max(target_peak_point_count, 1)
        ),
        "target_peak_under_rate": (
            target_peak_under_count / max(target_peak_point_count, 1)
        ),
        "peak_hour_mae": peak_hour_abs_sum / max(peak_count, 1),
        "scenario_peak_pvpe24": (
            scenario_peak_pct_sum / max(peak_count, 1)
            if latent_mode == "mc" else None
        ),
        "scenario_peak_under_probability": (
            scenario_peak_under_sum / max(peak_count, 1)
            if latent_mode == "mc" else None
        ),
        "n_points": point_count,
        "n_day_ahead_origins": peak_count,
        "booster_residual_ratio": residual_ratio_sum / max(diag_weight, 1),
        "phase_feature_norm": phase_feature_norm_sum / max(diag_weight, 1),
        "phase_day_diversity": phase_day_diversity_sum / max(diag_weight, 1),
    }

    for k in prefix_ks:
        cnt = prefix[k]["count"]
        name = f"all_first{k}"
        result[f"{name}_mse_norm"] = prefix[k]["norm_sq"] / cnt
        result[f"{name}_rmse_norm"] = math.sqrt(prefix[k]["norm_sq"] / cnt)
        result[f"{name}_mae_norm"] = prefix[k]["norm_abs"] / cnt
        result[f"{name}_rmse_raw"] = math.sqrt(prefix[k]["raw_sq"] / cnt)
        result[f"{name}_mae_raw"] = prefix[k]["raw_abs"] / cnt
        result[f"{name}_wape_raw"] = prefix[k]["raw_abs"] / max(prefix[k]["raw_y_abs"], 1e-12)
        result[f"{name}_n_points"] = cnt

    mid_name = f"all_mid{cfg.mid_start_hour}_{cfg.mid_end_hour}"
    mid_cnt = mid["count"]
    result[f"{mid_name}_mse_norm"] = mid["norm_sq"] / mid_cnt
    result[f"{mid_name}_rmse_norm"] = math.sqrt(mid["norm_sq"] / mid_cnt)
    result[f"{mid_name}_mae_norm"] = mid["norm_abs"] / mid_cnt
    result[f"{mid_name}_rmse_raw"] = math.sqrt(mid["raw_sq"] / mid_cnt)
    result[f"{mid_name}_mae_raw"] = mid["raw_abs"] / mid_cnt
    result[f"{mid_name}_wape_raw"] = mid["raw_abs"] / max(mid["raw_y_abs"], 1e-12)
    result[f"{mid_name}_n_points"] = mid_cnt

    result["focus_score"] = (
        result["mse_norm"]
        + cfg.focus_score_early_weight * result["all_first6_mse_norm"]
        + cfg.focus_score_mid_weight * result[f"{mid_name}_mse_norm"]
    )
    result["peak_score"] = (
        result["topk_peak_mse_norm"]
        + cfg.peak_selection_under_weight * result["topk_peak_under_mse_norm"]
        + cfg.target_peak_selection_under_weight
        * result["target_peak_under_mse_norm"]
    )
    result["balanced_score"] = (
        result["focus_score"]
        + cfg.balanced_peak_weight * result["peak_score"]
    )
    result["horizon_mse_norm"] = [
        float(horizon_norm_sq[h] / horizon_count[h]) for h in range(H)
    ]
    result["horizon_rmse_raw"] = [
        math.sqrt(horizon_raw_sq[h] / horizon_count[h]) for h in range(H)
    ]
    result["horizon_mae_raw"] = [
        float(horizon_raw_abs[h] / horizon_count[h]) for h in range(H)
    ]
    result["origin_24h_rmse_raw"] = [
        math.sqrt(origin_raw_sq[o] / origin_count[o]) if origin_count[o] > 0 else None
        for o in range(n_origins)
    ]
    result["origin_24h_mae_raw"] = [
        float(origin_raw_abs[o] / origin_count[o]) if origin_count[o] > 0 else None
        for o in range(n_origins)
    ]
    result["origin_first6_rmse_raw"] = [
        math.sqrt(origin_first6_raw_sq[o] / origin_first6_count[o])
        if origin_first6_count[o] > 0 else None
        for o in range(n_origins)
    ]
    result["origin_target_peak_rmse_raw"] = [
        math.sqrt(origin_target_peak_raw_sq[o] / origin_target_peak_point_count[o])
        if origin_target_peak_point_count[o] > 0 else None
        for o in range(n_origins)
    ]
    result["origin_target_peak_under_rate"] = [
        float(origin_target_peak_under_count[o] / origin_target_peak_point_count[o])
        if origin_target_peak_point_count[o] > 0 else None
        for o in range(n_origins)
    ]
    result["origin_peak_hour_mae"] = [
        float(origin_peak_hour_abs[o] / origin_peak_count[o])
        if origin_peak_count[o] > 0 else None
        for o in range(n_origins)
    ]
    result["origin_horizon_rmse_raw"] = [
        [
            math.sqrt(matrix_raw_sq[o, h] / matrix_count[o, h])
            if matrix_count[o, h] > 0 else None
            for h in range(H)
        ]
        for o in range(n_origins)
    ]
    result["origin_horizon_mae_raw"] = [
        [
            float(matrix_raw_abs[o, h] / matrix_count[o, h])
            if matrix_count[o, h] > 0 else None
            for h in range(H)
        ]
        for o in range(n_origins)
    ]

    if prediction_path is not None:
        pd.DataFrame(rows).to_csv(prediction_path, index=False)

    if scenario_path is not None and scenario_mu_norm_blocks:
        scenario_mu_norm = np.concatenate(scenario_mu_norm_blocks, axis=1)
        target_norm = np.concatenate(scenario_target_norm_blocks, axis=0)
        scales = np.concatenate(scenario_scale_blocks, axis=0)
        shifts = np.concatenate(scenario_shift_blocks, axis=0)
        scenario_mu_raw = (
            scenario_mu_norm * scales[None, :, None]
            + shifts[None, :, None]
        )
        target_raw = target_norm * scales[:, None] + shifts[:, None]
        np.savez_compressed(
            scenario_path,
            scenario_mu_norm=scenario_mu_norm.astype(np.float32),
            scenario_mu_raw=scenario_mu_raw.astype(np.float32),
            target_norm=target_norm.astype(np.float32),
            target_raw=target_raw.astype(np.float32),
            profile=np.asarray(scenario_profiles, dtype=str),
            origin_time=np.asarray(scenario_origin_times, dtype=str),
            origin_offset=np.asarray(scenario_origin_offsets, dtype=np.int16),
            mc_seed=np.asarray([cfg.mc_seed], dtype=np.int64),
            antithetic=np.asarray([cfg.mc_antithetic], dtype=bool),
        )
    return result


# ============================================================
# Training
# ============================================================

def build_model(cfg: Config, n_externals: int, expert_specs: Sequence[Dict], device: torch.device):
    n_experts = len(expert_specs)
    if cfg.top_k <= 0 or cfg.top_k > n_experts:
        raise ValueError(
            f"ONCOR_TOP_K must be in [1,{n_experts}] for the grouped experts, got {cfg.top_k}"
        )
    print(f"[MOE] experts={[spec['name'] for spec in expert_specs]}, top_k={cfg.top_k}")
    model = VariationalSeq2Seq_meta(
        xprime_dim=cfg.xprime_dim,
        input_dim=cfg.input_dim,
        hidden_size=cfg.hidden_dim,
        latent_size=cfg.latent_dim,
        output_len=cfg.output_len,
        n_externals=n_externals,
        expert_specs=expert_specs,
        encoder_mode=cfg.encoder_mode,
        output_dim=cfg.output_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        logvar_min=cfg.logvar_min,
        logvar_max=cfg.logvar_max,
        mlp_hidden=cfg.mlp_hidden,
        reverse_hidden_size=cfg.reverse_hidden_dim,
        fusion_bottleneck=cfg.fusion_bottleneck,
        residual_scale=cfg.residual_scale,
    ).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"[MODEL] mode={cfg.encoder_mode}, parameters={total:,}, "
        "reverse_encoder=disabled, chronological_context=168h"
    )
    return model


def _optimizer(model, lr, cfg):
    return torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay
    )


def _scheduled_lr(cfg: Config, update: int) -> Tuple[float, int]:
    if update <= cfg.lr_warmup_update_budget:
        lr = cfg.base_lr * update / max(cfg.lr_warmup_update_budget, 1)
        stage = 0
    elif update < cfg.lr_decay_update_1:
        lr = cfg.base_lr
        stage = 0
    elif update < cfg.lr_decay_update_2:
        lr = cfg.base_lr * 0.5
        stage = 1
    elif update < cfg.lr_decay_update_3:
        lr = cfg.base_lr * 0.25
        stage = 2
    elif update < cfg.lr_decay_update_4:
        lr = cfg.base_lr / 6.0
        stage = 3
    elif update <= cfg.continuation_start_update:
        lr = cfg.base_lr / 12.0
        stage = 4
    elif update < cfg.continuation_lr_switch_update:
        lr = cfg.continuation_lr_1
        stage = 5
    else:
        lr = cfg.continuation_lr_2
        stage = 6
    return float(lr), int(stage)



def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_meta: List[Dict],
    cfg: Config,
    device: torch.device,
    paths: Dict[str, Path],
    history_path: Path,
    resume_checkpoint: Optional[Path] = None,
):
    optimizer = _optimizer(model, cfg.base_lr, cfg)
    global_update = 0
    start_epoch = 0
    resumed = False

    if resume_checkpoint is not None:
        if not resume_checkpoint.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_checkpoint}")
        resume_obj = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        if not isinstance(resume_obj, dict) or "model" not in resume_obj:
            raise ValueError("Resume checkpoint must be a full checkpoint dictionary")
        model.load_state_dict(resume_obj["model"])
        if "optimizer" in resume_obj:
            optimizer.load_state_dict(resume_obj["optimizer"])
        global_update = int(resume_obj.get("global_update", 0))
        start_epoch = int(resume_obj.get("epoch", 0))
        resumed = True
        if global_update >= cfg.total_update_budget:
            raise ValueError(
                f"Resume update {global_update} is not below total budget {cfg.total_update_budget}"
            )

    horizon_weights = cfg.horizon_weights()
    mid_key = f"all_mid{cfg.mid_start_hour}_{cfg.mid_end_hour}_mse_norm"
    print(f"[HORIZON WEIGHTS] {[round(v, 4) for v in horizon_weights]}")
    print(
        f"[OBJECTIVE] horizon_weighting={cfg.use_horizon_weighting}, "
        f"horizon_loss_weight={cfg.horizon_loss_weight:g}, "
        f"boundary_ramp_weight={cfg.boundary_ramp_weight:g}, "
        f"peak_loss={cfg.use_peak_loss}, topk={cfg.peak_top_k}, "
        f"peak_weights=(magnitude={cfg.peak_magnitude_weight:g}, "
        f"aggregate_under={cfg.peak_under_weight:g}, "
        f"target_aligned_under={cfg.true_peak_under_weight:g}), "
        f"peak_warmup=({cfg.peak_warmup_start},{cfg.peak_warmup_end})"
    )
    if resumed:
        print(
            f"[RESUME] {resume_checkpoint} | epoch={start_epoch} "
            f"update={global_update} -> {cfg.total_update_budget}"
        )
    else:
        print(f"[TRAIN FROM SCRATCH] update=0 -> {cfg.total_update_budget}")
    print(
        "[TRAINING POLICY] fixed update schedule; no rollback or early stop. "
        "Validation uses deterministic z=mu and only selects checkpoints."
    )

    best = {
        "best_mse": float("inf"),
        "best_crps": float("inf"),
        "best_early6": float("inf"),
        "best_mid": float("inf"),
        "best_focus": float("inf"),
        "best_peak": float("inf"),
        "best_balanced": float("inf"),
    }
    best_monitor = float("inf")
    bad_epochs = 0
    history = []
    for epoch in range(start_epoch + 1, cfg.epochs + 1):
        model.train()
        loss_sum = nll_sum = kl_sum = horizon_sum = ramp_sum = 0.0
        peak_mag_sum = peak_under_sum = target_peak_mse_sum = 0.0
        target_peak_under_sum = peak_extra_sum = 0.0
        active_sum = 0.0
        started = time.time()
        last_kl_w = last_peak_factor = 0.0
        stage = 5

        for batch in train_loader:
            if global_update >= cfg.total_update_budget:
                break
            update = global_update + 1
            lr, stage = _scheduled_lr(cfg, update)
            for group in optimizer.param_groups:
                group["lr"] = lr

            kl_w = cfg.kl_weight * min(
                1.0, update / max(cfg.kl_anneal_update_budget, 1)
            )
            peak_factor = peak_warm_factor(cfg, update) if cfg.use_peak_loss else 0.0
            last_kl_w = kl_w
            last_peak_factor = peak_factor

            enc_l, enc_ext, dec_l, dec_ext, target, origin_mask, _, _ = [
                x.to(device) for x in batch
            ]
            optimizer.zero_grad(set_to_none=True)
            mu, logvar, mu_z, logvar_z = model(
                enc_l, enc_ext, dec_l, dec_ext,
                epoch=update, top_k=cfg.top_k,
                warmup_epochs=cfg.moe_warmup_update_budget,
                forecast_indices=TRAIN_FORECAST_INDICES,
            )
            nll = gaussian_nll_masked(mu, logvar, target, origin_mask)
            kl = weighted_kl(mu_z, logvar_z, origin_mask)
            horizon_mse = all_origin_horizon_weighted_mse(
                mu, target, origin_mask, horizon_weights
            )
            ramp_mse = all_origin_boundary_ramp_mse(
                mu, target, enc_l[:, -1, :], dec_l, origin_mask,
                TRAIN_FORECAST_INDICES, cfg.boundary_ramp_hours,
            )
            peak_mag, peak_under = topk_peak_losses(
                mu, target, origin_mask, cfg.peak_top_k
            )
            target_peak_mse, target_peak_under = target_aligned_peak_losses(
                mu, target, origin_mask, cfg.peak_top_k
            )

            extra = (
                cfg.horizon_loss_weight * horizon_mse
                + cfg.boundary_ramp_weight * ramp_mse
                if cfg.use_horizon_weighting else mu.new_zeros(())
            )
            peak_extra = peak_factor * (
                cfg.peak_magnitude_weight * peak_mag
                + cfg.peak_under_weight * peak_under
                + cfg.true_peak_under_weight * target_peak_under
            )
            loss = nll + kl_w * kl + extra + peak_extra
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch}, update {update}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            global_update = update

            active = float(origin_mask.sum().item())
            loss_sum += float(loss.detach()) * active
            nll_sum += float(nll.detach()) * active
            kl_sum += float(kl.detach()) * active
            horizon_sum += float(horizon_mse.detach()) * active
            ramp_sum += float(ramp_mse.detach()) * active
            peak_mag_sum += float(peak_mag.detach()) * active
            peak_under_sum += float(peak_under.detach()) * active
            target_peak_mse_sum += float(target_peak_mse.detach()) * active
            target_peak_under_sum += float(target_peak_under.detach()) * active
            peak_extra_sum += float(peak_extra.detach()) * active
            active_sum += active

        if active_sum <= 0:
            break

        val = evaluate(
            model, val_loader, device, val_meta, cfg, latent_mode="mean"
        )
        score_map = {
            "best_mse": val["mse_norm"],
            "best_crps": val["crps_norm"],
            "best_early6": val["all_first6_mse_norm"],
            "best_mid": val[mid_key],
            "best_focus": val["focus_score"],
            "best_peak": val["peak_score"],
            "best_balanced": val["balanced_score"],
        }
        monitor = val["balanced_score"] if cfg.use_peak_loss else val["focus_score"]
        improved_monitor = monitor < best_monitor

        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_update": global_update,
            "val": val,
            "config": asdict(cfg),
            "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
            "is_resume_start": False,
        }
        for name, score in score_map.items():
            if score < best[name]:
                best[name] = score
                torch.save(payload, paths[name])
        if cfg.save_latest_full:
            torch.save(payload, paths["latest_full"])

        if improved_monitor:
            best_monitor = monitor
            bad_epochs = 0
        else:
            bad_epochs += 1

        record = {
            "epoch": epoch,
            "global_update": global_update,
            "stage": stage,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": loss_sum / max(active_sum, 1.0),
            "train_nll": nll_sum / max(active_sum, 1.0),
            "train_kl": kl_sum / max(active_sum, 1.0),
            "train_horizon_weighted_mse": horizon_sum / max(active_sum, 1.0),
            "train_boundary_ramp_mse": ramp_sum / max(active_sum, 1.0),
            "train_topk_peak_mse": peak_mag_sum / max(active_sum, 1.0),
            "train_topk_peak_under_mse": peak_under_sum / max(active_sum, 1.0),
            "train_target_peak_mse": target_peak_mse_sum / max(active_sum, 1.0),
            "train_target_peak_under_mse": (
                target_peak_under_sum / max(active_sum, 1.0)
            ),
            "train_peak_extra": peak_extra_sum / max(active_sum, 1.0),
            "kl_weight": last_kl_w,
            "peak_warm_factor": last_peak_factor,
            "val_mse": val["mse_norm"],
            "val_crps": val["crps_norm"],
            "val_all_h1_mse": val["all_first1_mse_norm"],
            "val_all_first3_mse": val["all_first3_mse_norm"],
            "val_all_first6_mse": val["all_first6_mse_norm"],
            "val_mid_mse": val[mid_key],
            "val_topk_peak_mse": val["topk_peak_mse_norm"],
            "val_topk_peak_under_mse": val["topk_peak_under_mse_norm"],
            "val_target_peak_mse": val["target_peak_mse_norm"],
            "val_target_peak_under_mse": val["target_peak_under_mse_norm"],
            "val_target_peak_under_rate": val["target_peak_under_rate"],
            "val_peak_hour_mae": val["peak_hour_mae"],
            "val_peak_score": val["peak_score"],
            "val_focus_score": val["focus_score"],
            "val_balanced_score": val["balanced_score"],
            "val_monitor_score": monitor,
            "val_residual_ratio": val["booster_residual_ratio"],
            "val_phase_feature_norm": val["phase_feature_norm"],
            "val_phase_day_diversity": val["phase_day_diversity"],
            "bad_epochs_since_monitor_improvement": bad_epochs,
            "seconds": time.time() - started,
        }
        history.append(record)
        pd.DataFrame(history).to_csv(history_path, index=False)

        if (
            epoch == start_epoch + 1
            or epoch % 5 == 0
            or improved_monitor
            or global_update >= cfg.total_update_budget
        ):
            print(
                f"[TRAIN] {epoch:04d}/{cfg.epochs} "
                f"update={global_update:04d}/{cfg.total_update_budget} "
                f"stage={stage} loss={record['train_loss']:.6f} "
                f"valMSE={val['mse_norm']:.7f} "
                f"early6={val['all_first6_mse_norm']:.7f} "
                f"mid={val[mid_key]:.7f} "
                f"peak={val['peak_score']:.7f} "
                f"targetUnder={val['target_peak_under_rate']:.4f} "
                f"peakHourMAE={val['peak_hour_mae']:.3f} "
                f"balanced={val['balanced_score']:.7f} "
                f"monitor={monitor:.7f} best={best_monitor:.7f} "
                f"lr={record['lr']:.2e} peakWarm={last_peak_factor:.3f} "
                f"resid={val['booster_residual_ratio']:.4f} "
                f"dayDiv={val['phase_day_diversity']:.4f} "
                f"bad={bad_epochs}"
            )

        if global_update >= cfg.total_update_budget:
            print(f"[UPDATE BUDGET] reached {global_update} optimizer updates")
            break

    return history


def load_checkpoint(path: Path, model: nn.Module, device: torch.device):
    obj = torch.load(path, map_location=device, weights_only=False)
    state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    model.load_state_dict(state)
    return obj


# ============================================================
# End-to-end orchestration
# ============================================================

def _write_metric_tables(output: Path, checkpoint_name: str, split: str, metrics: Dict) -> None:
    prefix = output / f"{checkpoint_name}_{split}"
    pd.DataFrame({
        "horizon": np.arange(1, len(metrics["horizon_rmse_raw"]) + 1),
        "mse_norm": metrics["horizon_mse_norm"],
        "rmse_raw": metrics["horizon_rmse_raw"],
        "mae_raw": metrics["horizon_mae_raw"],
    }).to_csv(str(prefix) + "_horizon_metrics.csv", index=False)

    pd.DataFrame({
        "origin_index": np.arange(len(FORECAST_INDICES)),
        "origin_offset": list(FORECAST_INDICES),
        "rmse_24h_raw": metrics["origin_24h_rmse_raw"],
        "mae_24h_raw": metrics["origin_24h_mae_raw"],
        "rmse_first6_raw": metrics["origin_first6_rmse_raw"],
        "target_peak_rmse_raw": metrics["origin_target_peak_rmse_raw"],
        "target_peak_under_rate": metrics["origin_target_peak_under_rate"],
        "peak_hour_mae": metrics["origin_peak_hour_mae"],
    }).to_csv(str(prefix) + "_origin_metrics.csv", index=False)

    columns = [f"h{h:02d}" for h in range(1, len(metrics["horizon_rmse_raw"]) + 1)]
    rmse_df = pd.DataFrame(metrics["origin_horizon_rmse_raw"], columns=columns)
    rmse_df.insert(0, "origin_offset", list(FORECAST_INDICES))
    rmse_df.to_csv(str(prefix) + "_origin_horizon_rmse_raw.csv", index=False)
    mae_df = pd.DataFrame(metrics["origin_horizon_mae_raw"], columns=columns)
    mae_df.insert(0, "origin_offset", list(FORECAST_INDICES))
    mae_df.to_csv(str(prefix) + "_origin_horizon_mae_raw.csv", index=False)


def run_experiment(
    encoder_mode: str,
    run_name: str,
    use_horizon_weighting: bool,
    use_peak_loss: bool,
    resume_checkpoint: Optional[str] = None,
) -> None:
    cfg = Config(
        encoder_mode=encoder_mode,
        run_name=run_name,
        use_horizon_weighting=bool(use_horizon_weighting),
        use_peak_loss=bool(use_peak_loss),
    )
    resume_value = resume_checkpoint or cfg.resume_checkpoint_env
    resume_path = Path(resume_value) if resume_value else None
    if cfg.require_resume and resume_path is None:
        raise ValueError(
            "ONCOR_REQUIRE_RESUME=1 but no resume checkpoint was provided. "
            "Set ONCOR_RESUME_CHECKPOINT or pass resume_checkpoint."
        )
    if resume_path is not None and not resume_path.exists():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")
    if device.type == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}")

    output = Path(cfg.output_root) / run_name
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "best_mse": output / "mse.pt",
        "best_crps": output / "crps.pt",
        "best_early6": output / "early6.pt",
        "best_mid": output / "mid.pt",
        "best_focus": output / "focus.pt",
        "best_peak": output / "peak.pt",
        "best_balanced": output / "balanced.pt",
        "latest_full": output / "latest.pt",
    }
    history_path = output / "training_history.csv"

    profiles = load_profiles(cfg.profile_ids())
    prepared = prepare_data(profiles, cfg, device)
    prepared["metadata"].to_csv(output / "split_metadata.csv", index=False)

    train_loader = make_loader(
        prepared["tensors"]["train"], cfg.batch_size, True, cfg.num_workers, seed=cfg.seed
    )
    val_loader = make_loader(
        prepared["tensors"]["val"], cfg.batch_size, False, cfg.num_workers
    )
    test_loader = make_loader(
        prepared["tensors"]["test"], cfg.batch_size, False, cfg.num_workers
    )
    resolve_update_schedule(cfg, len(train_loader))
    model = build_model(cfg, len(prepared["ext_keys"]), prepared["expert_specs"], device)
    set_seed(cfg.seed)

    config_json = {
        **asdict(cfg),
        "forecast_indices": list(EVAL_FORECAST_INDICES),  # plotting compatibility
        "train_forecast_indices": list(TRAIN_FORECAST_INDICES),
        "eval_forecast_indices": list(EVAL_FORECAST_INDICES),
        "horizon_weights_normalized": list(cfg.horizon_weights()),
        "split_years": prepared["split_years"],
        "ext_keys": prepared["ext_keys"],
        "expert_specs": prepared["expert_specs"],
        "profile_names": prepared["profile_names"],
        "scaler_meta": prepared["scaler_meta"],
        "resume_checkpoint": str(resume_path) if resume_path is not None else None,
        "resume_expected_update": (cfg.continuation_start_update if resume_path is not None else None),
        "training_objective": (
            "masked Gaussian NLL + annealed VAE KL"
            + (
                " + all-origin horizon-weighted MSE + boundary-ramp MSE"
                if cfg.use_horizon_weighting else ""
            )
            + (
                " + warm-started top-k magnitude, aggregate-under, and "
                "target-aligned peak-underprediction losses"
                if cfg.use_peak_loss else ""
            )
        ),
        "split_definition": (
            "hourly rolling 24h origins for training; daily 00:00 origins for "
            "validation/test; Jan-Oct train / November validation / December test"
        ),
        "protocol_version": "24h_mlp_forwardonly_rollingtrain",
        "reverse_context_hours": 0,
        "reverse_time_order": None,
        "booster_definition": "disabled; chronological forward encoder only",
        "checkpoint_definition": (
            "mse, crps, early6, mid, focus, peak, balanced, latest; "
            "load an existing checkpoint or train from scratch"
        ),
        "evaluation_definition": (
            "validation/checkpoint selection uses z=mu; final test also reports "
            f"{cfg.mc_latent_samples}-sample antithetic latent MC with exact mixture NLL/CRPS"
        ),
    }
    with (output / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config_json, f, indent=2)

    monitor_path = paths["best_balanced"] if cfg.use_peak_loss else paths["best_focus"]
    checkpoint_candidates = [
        monitor_path,
        paths["best_mse"],
        paths["best_crps"],
        paths["latest_full"],
    ]
    existing_checkpoint = next((p for p in checkpoint_candidates if p.exists()), None)
    if cfg.eval_only:
        if existing_checkpoint is None:
            raise FileNotFoundError(
                f"ONCOR_EVAL_ONLY=1 but no checkpoint exists in {output}"
            )
    elif cfg.force_retrain or existing_checkpoint is None:
        train(
            model, train_loader, val_loader,
            prepared["tensors"]["val"]["metadata"],
            cfg, device, paths, history_path, resume_path,
        )
    else:
        print(f"[LOAD] Existing checkpoint found: {existing_checkpoint}")

    checkpoint_short_names = {
        "best_mse": "mse",
        "best_crps": "crps",
        "best_early6": "early6",
        "best_mid": "mid",
        "best_focus": "focus",
        "best_peak": "peak",
        "best_balanced": "balanced",
    }
    checkpoint_internal_names = {v: k for k, v in checkpoint_short_names.items()}

    deterministic_results = {}
    mc_results = {}
    mid_key = f"all_mid{cfg.mid_start_hour}_{cfg.mid_end_hour}_rmse_raw"
    eval_checkpoint_names = [
        "best_mse", "best_crps", "best_early6", "best_mid",
        "best_focus", "best_peak", "best_balanced",
    ]
    mc_names = {
        checkpoint_internal_names.get(name, name)
        for name in cfg.mc_checkpoint_names()
    }

    for checkpoint_name in eval_checkpoint_names:
        checkpoint_path = paths[checkpoint_name]
        if not checkpoint_path.exists():
            continue
        short_name = checkpoint_short_names[checkpoint_name]
        load_checkpoint(checkpoint_path, model, device)
        deterministic_results[short_name] = {}
        for split, loader in (("val", val_loader), ("test", test_loader)):
            pred_path = (
                output / f"{short_name}_{split}_predictions.csv"
                if cfg.save_predictions else None
            )
            metrics = evaluate(
                model, loader, device,
                prepared["tensors"][split]["metadata"], cfg,
                prediction_path=pred_path,
                latent_mode="mean",
            )
            deterministic_results[short_name][split] = metrics
            _write_metric_tables(output, short_name, split, metrics)
            print(
                f"[DET][{short_name.upper()}][{split.upper()}] "
                f"MSE={metrics['mse_norm']:.7f} CRPS={metrics['crps_norm']:.7f} "
                f"WAPE={metrics['wape_raw']:.4f} "
                f"H1={metrics['all_first1_rmse_raw']:.5f} "
                f"3h={metrics['all_first3_rmse_raw']:.5f} "
                f"6h={metrics['all_first6_rmse_raw']:.5f} "
                f"MID={metrics[mid_key]:.5f} "
                f"PVPE={metrics['pvpe24']:.4f} under={metrics['under24']:.4f} "
                f"targetUnder={metrics['target_peak_under_rate']:.4f} "
                f"peakHourMAE={metrics['peak_hour_mae']:.3f} "
                f"resid={metrics['booster_residual_ratio']:.4f} "
                f"balanced={metrics['balanced_score']:.7f}"
            )

        if checkpoint_name in mc_names:
            pred_path = (
                output / f"{short_name}_test_mc20.csv"
                if cfg.save_predictions else None
            )
            scenario_path = (
                output / f"{short_name}_test_mc20.npz"
                if cfg.save_mc_scenarios else None
            )
            metrics_mc = evaluate(
                model, test_loader, device,
                prepared["tensors"]["test"]["metadata"], cfg,
                prediction_path=pred_path,
                latent_mode="mc",
                scenario_path=scenario_path,
            )
            mc_results[short_name] = {"test": metrics_mc}
            _write_metric_tables(output, f"{short_name}_mc20", "test", metrics_mc)
            print(
                f"[MC20][{short_name.upper()}][TEST] "
                f"MSE={metrics_mc['mse_norm']:.7f} "
                f"CRPS={metrics_mc['crps_norm']:.7f} "
                f"WAPE={metrics_mc['wape_raw']:.4f} "
                f"H1={metrics_mc['all_first1_rmse_raw']:.5f} "
                f"3h={metrics_mc['all_first3_rmse_raw']:.5f} "
                f"6h={metrics_mc['all_first6_rmse_raw']:.5f} "
                f"MID={metrics_mc[mid_key]:.5f} "
                f"PVPE={metrics_mc['pvpe24']:.4f} "
                f"under={metrics_mc['under24']:.4f} "
                f"scenarioUnderP={metrics_mc['scenario_peak_under_probability']:.4f}"
            )

    with (output / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(deterministic_results, f, indent=2)
    with (output / "metrics_mc20.json").open("w", encoding="utf-8") as f:
        json.dump(mc_results, f, indent=2)
    print(f"[DONE] outputs: {output}")



# ============================================================
# Default experiment: hourly rolling 24h training, seven daily evaluation
# forecasts, and one chronological forward encoder
# ============================================================
if __name__ == "__main__":
    run_experiment(
        encoder_mode="forward",
        run_name=os.getenv(
            "ONCOR_RUN_NAME",
            "24hr_mlp_forwardonly_rollingtrain",
        ),
        use_horizon_weighting=True,
        use_peak_loss=False,
        resume_checkpoint=os.getenv("ONCOR_RESUME_CHECKPOINT", "") or None,
    )
