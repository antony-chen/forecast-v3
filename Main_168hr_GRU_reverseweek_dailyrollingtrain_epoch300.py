"""Train and evaluate rolling-origin 168-hour forecasting with a GRU decoder.

For every training origin, the input is the immediately preceding 168 hours and
the target is the immediately following 168 hours. Training origins move hourly.
Validation and test retain the original packed-week 00:00 origins. No future
load is supplied to the model.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader, Dataset

# Use the corrected data_utils.py shipped beside this Main file.  Resolve the
# project root separately for processed_data and output paths.
THIS_DIR = Path(__file__).resolve().parent


def _resolve_project_root() -> Path:
    configured = os.getenv("ONCOR_PROJECT_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    for candidate in [THIS_DIR, *THIS_DIR.parents]:
        if (candidate / "processed_data").is_dir():
            return candidate.resolve()
    return THIS_DIR.parent.resolve()


PROJECT_ROOT = _resolve_project_root()
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from data_utils import get_data_oncor_load_weekly  # noqa: E402
from Model_168hr_GRU_reverseweek_dailyrollingtrain import Direct168GRUModel, WEEK_HOURS  # noqa: E402


@dataclass
class Config:
    run_name: str = os.getenv(
        "ONCOR_RUN_NAME",
        "168hr_gru_reverseweek_dailyrollingtrain_epoch300",
    )
    output_root: str = os.getenv(
        "ONCOR_OUTPUT_ROOT",
        str(PROJECT_ROOT / "oncor_reverseweek"),
    )
    profile_ids_env: str = os.getenv("ONCOR_XFMR_IDS", "")
    split_years_env: str = os.getenv("ONCOR_SPLIT_YEARS", "")

    seed: int = int(os.getenv("ONCOR_SEED", "42"))
    batch_size: int = int(os.getenv("ONCOR_BATCH_SIZE", "16"))
    epochs: int = int(os.getenv("ONCOR_EPOCHS", "300"))
    base_lr: float = float(os.getenv("ONCOR_LR", "3e-4"))
    weight_decay: float = float(os.getenv("ONCOR_WEIGHT_DECAY", "1e-4"))
    grad_clip: float = float(os.getenv("ONCOR_GRAD_CLIP", "1.0"))
    num_workers: int = int(os.getenv("ONCOR_NUM_WORKERS", "0"))
    train_stride_hours: int = int(os.getenv("ONCOR_TRAIN_STRIDE_HOURS", "24"))

    # Epoch-based schedule. No optimizer-update budget is used.
    lr_warmup_epochs: int = int(os.getenv("ONCOR_LR_WARMUP_EPOCHS", "10"))
    lr_decay_epoch_1: int = int(os.getenv("ONCOR_LR_DECAY_EPOCH_1", "90"))
    lr_decay_epoch_2: int = int(os.getenv("ONCOR_LR_DECAY_EPOCH_2", "150"))
    lr_decay_epoch_3: int = int(os.getenv("ONCOR_LR_DECAY_EPOCH_3", "210"))
    lr_decay_epoch_4: int = int(os.getenv("ONCOR_LR_DECAY_EPOCH_4", "260"))
    kl_anneal_epochs: int = int(os.getenv("ONCOR_KL_ANNEAL_EPOCHS", "20"))
    moe_warmup_epochs: int = int(os.getenv("ONCOR_MOE_WARMUP_EPOCHS", "10"))

    xprime_dim: int = 16
    hidden_dim: int = 96
    latent_dim: int = 64
    reverse_hidden_dim: int = 48
    fusion_bottleneck: int = 32
    residual_scale: float = 0.1
    phase_summary_dim: int = int(os.getenv("ONCOR_PHASE_SUMMARY_DIM", "32"))
    future_summary_dim: int = int(os.getenv("ONCOR_FUTURE_SUMMARY_DIM", "32"))
    hourly_hidden: int = int(os.getenv("ONCOR_HOURLY_HIDDEN", "64"))
    dropout: float = float(os.getenv("ONCOR_DROPOUT", "0.1"))
    output_len: int = WEEK_HOURS
    output_dim: int = 1
    input_dim: int = 1
    encoder_mode: str = "reverse_week_dual"
    top_k: int = int(os.getenv("ONCOR_TOP_K", "4"))
    logvar_min: float = -10.0
    logvar_max: float = -3.8

    kl_weight: float = float(os.getenv("ONCOR_KL_WEIGHT", "1e-4"))
    mse_weight: float = float(os.getenv("ONCOR_MSE_WEIGHT", "2.0"))
    ramp_weight: float = float(os.getenv("ONCOR_BOUNDARY_RAMP_WEIGHT", "0.5"))
    ramp_hours: int = int(os.getenv("ONCOR_BOUNDARY_RAMP_HOURS", "3"))

    eval_only: bool = os.getenv("ONCOR_EVAL_ONLY", "0") == "1"
    force_retrain: bool = os.getenv("ONCOR_FORCE_RETRAIN", "0") == "1"
    save_predictions: bool = os.getenv("ONCOR_SAVE_PREDICTIONS", "1") == "1"
    require_midnight_origin: bool = os.getenv("ONCOR_REQUIRE_MIDNIGHT_ORIGIN", "1") == "1"

    def profile_ids(self) -> Optional[List[str]]:
        values = [v.strip() for v in self.profile_ids_env.split(",") if v.strip()]
        return values or None

    def split_years(self) -> Optional[List[int]]:
        values = [int(v.strip()) for v in self.split_years_env.split(",") if v.strip()]
        return sorted(set(values)) or None

    def validate(self) -> None:
        if self.output_len != 168:
            raise ValueError("Direct168 requires output_len=168")
        if self.epochs <= 0:
            raise ValueError("ONCOR_EPOCHS must be positive")
        if not (
            0 < self.lr_warmup_epochs < self.lr_decay_epoch_1
            < self.lr_decay_epoch_2 < self.lr_decay_epoch_3
            < self.lr_decay_epoch_4 < self.epochs
        ):
            raise ValueError(
                "Require 0 < warmup < decay1 < decay2 < decay3 < decay4 < epochs"
            )
        if self.kl_anneal_epochs <= 0:
            raise ValueError("ONCOR_KL_ANNEAL_EPOCHS must be positive")
        if self.moe_warmup_epochs < 0:
            raise ValueError("ONCOR_MOE_WARMUP_EPOCHS must be non-negative")
        if self.ramp_hours < 1 or self.ramp_hours > self.output_len:
            raise ValueError("ramp_hours must be in [1,168]")
        if self.train_stride_hours < 1:
            raise ValueError("ONCOR_TRAIN_STRIDE_HOURS must be positive")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _ensure_week_matrix(arr, name: str, dtype=None):
    values = np.asarray(arr, dtype=dtype)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2 or values.shape[1] != WEEK_HOURS:
        raise ValueError(f"{name} must be [weeks,168], got {values.shape}")
    return values


def split_merged_profiles(times, load, temp, workday, season) -> List[Dict]:
    times = _ensure_week_matrix(times, "times")
    load = _ensure_week_matrix(load, "load", dtype=float)
    temp = _ensure_week_matrix(temp, "temp", dtype=float)
    workday = _ensure_week_matrix(workday, "workday", dtype=float)
    season = _ensure_week_matrix(season, "season", dtype=float)
    if not (times.shape == load.shape == temp.shape == workday.shape == season.shape):
        raise ValueError("Merged weekly arrays must share shape")

    starts, ends = [], []
    for row in times:
        idx = pd.DatetimeIndex(pd.to_datetime(np.asarray(row).reshape(-1)))
        starts.append(idx[0])
        ends.append(idx[-1])

    # Match v9_F: a new profile starts when weekly timestamps reset/overlap.
    boundaries = [0]
    for i in range(1, len(times)):
        if starts[i] <= ends[i - 1]:
            boundaries.append(i)
    boundaries.append(len(times))

    profiles = []
    for idx, (a, b) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        profiles.append({
            "name": f"profile_{idx}",
            "times": times[a:b],
            "load": load[a:b],
            "temp": temp[a:b],
            "workday": workday[a:b],
            "season": season[a:b],
        })
        print(
            f"[PROFILE] profile_{idx}: weeks={b-a}, "
            f"range={starts[a]} -> {ends[b-1]}"
        )
    if not profiles:
        raise RuntimeError("No profiles recovered from merged weekly arrays")
    print(f"[PROFILE] recovered {len(profiles)} profiles from XFMR='all'.")
    return profiles


def load_profiles(profile_ids: Optional[Sequence[str]]) -> List[Dict]:
    if profile_ids:
        profiles = []
        for value in profile_ids:
            arrays = get_data_oncor_load_weekly(XFMR=str(value))
            t, l, temp, wd, season = arrays
            profiles.append({
                "name": str(value),
                "times": _ensure_week_matrix(t, "times"),
                "load": _ensure_week_matrix(l, "load", dtype=float),
                "temp": _ensure_week_matrix(temp, "temp", dtype=float),
                "workday": _ensure_week_matrix(wd, "workday", dtype=float),
                "season": _ensure_week_matrix(season, "season", dtype=float),
            })
        return profiles
    print("[PROFILE] ONCOR_XFMR_IDS is empty; recovering profiles from XFMR='all'.")
    return split_merged_profiles(*get_data_oncor_load_weekly(XFMR="all"))


def _future_path(values: np.ndarray, horizon: int, include_current: bool) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    start = 0 if include_current else 1
    offsets = np.arange(start, start + horizon)[None, :]
    indices = np.clip(np.arange(len(values))[:, None] + offsets, 0, len(values) - 1)
    return values[indices]


def build_frame(profile: Dict) -> pd.DataFrame:
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

    # Use the next 24 temperatures (t+1 ... t+24), aligned with the
    # 24-hour load horizon; workday summary also excludes the current hour.
    temp_path = _future_path(temp, 24, include_current=False)
    for h in range(24):
        df[f"temp_fc_tplus{h:02d}"] = temp_path[:, h]
    wd_path = _future_path(workday, 24, include_current=False)
    df["workday_future24_mean"] = wd_path.mean(axis=1)
    return df


def infer_split_years(frames: Dict[str, pd.DataFrame], requested: Optional[List[int]]) -> List[int]:
    if requested:
        return requested
    complete = []
    for df in frames.values():
        years = set()
        for year, group in df.groupby(df.index.year):
            if set(range(1, 13)).issubset(set(group.index.month.unique())):
                years.add(int(year))
        complete.append(years)
    common = set.intersection(*complete) if complete else set()
    if common:
        return sorted(common)
    return sorted(set(int(y) for df in frames.values() for y in df.index.year.unique()))


def _is_hourly_contiguous(index: pd.DatetimeIndex, expected_len: int) -> bool:
    """Compare timestamp values, independent of pandas datetime unit (s/ns)."""
    idx = pd.DatetimeIndex(index)
    if len(idx) != expected_len:
        return False
    values = idx.to_numpy(dtype="datetime64[ns]")
    if len(values) <= 1:
        return True
    return bool(np.all(np.diff(values) == np.timedelta64(1, "h")))


def classify_target_week(index: pd.DatetimeIndex, split_years: set) -> Optional[str]:
    if not _is_hourly_contiguous(index, WEEK_HOURS):
        return None
    years = set(int(v) for v in index.year)
    if len(years) != 1 or next(iter(years)) not in split_years:
        return None
    months = index.month.to_numpy()
    if np.all((months >= 1) & (months <= 10)):
        return "train"
    if np.all(months == 11):
        return "val"
    if np.all(months == 12):
        return "test"
    return None


def _inverse_params(scaler: MinMaxScaler) -> Tuple[float, float]:
    lo, hi = float(scaler.data_min_[0]), float(scaler.data_max_[0])
    fr_lo, fr_hi = map(float, scaler.feature_range)
    scale = (hi - lo) / max(fr_hi - fr_lo, 1e-12)
    shift = lo - fr_lo * scale
    return scale, shift



class Rolling168WindowDataset(Dataset):
    """Materialize one rolling 168h-to-168h sample only when requested.

    Keeping the profile time series once, rather than copying every overlapping
    168-hour window into a large tensor, substantially reduces host/GPU memory.
    """

    def __init__(self, profile_arrays: Dict[str, Dict[str, np.ndarray]], records: List[Tuple]):
        self.profile_arrays = profile_arrays
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        profile, origin, scale, shift = self.records[index]
        arrays = self.profile_arrays[profile]
        enc_start = origin - WEEK_HOURS
        target_end = origin + WEEK_HOURS

        enc_l = torch.from_numpy(arrays["load"][enc_start:origin])
        enc_ext = torch.from_numpy(arrays["enc_ext"][enc_start:origin])
        future_ext = torch.from_numpy(arrays["future_ext"][origin:target_end])
        target = torch.from_numpy(arrays["load"][origin:target_end])
        scale_t = torch.tensor([scale], dtype=torch.float32)
        shift_t = torch.tensor([shift], dtype=torch.float32)
        return enc_l, enc_ext, future_ext, target, scale_t, shift_t


def prepare_data(profiles: List[Dict], cfg: Config, device: torch.device) -> Dict:
    del device  # samples are kept on CPU and moved batch-by-batch in train/evaluate

    profile_names = [str(p["name"]) for p in profiles]
    profile_to_idx = {name: i for i, name in enumerate(profile_names)}
    profile_keys = [f"profile_onehot_{i:02d}" for i in range(len(profile_names))]

    frames = {str(p["name"]): build_frame(p) for p in profiles}
    for name, df in frames.items():
        idx = profile_to_idx[name]
        for j, key in enumerate(profile_keys):
            df[key] = 1.0 if j == idx else 0.0

    split_years = infer_split_years(frames, cfg.split_years())
    split_year_set = set(split_years)
    print(
        f"[SPLIT] years={split_years}; daily rolling 00:00 origins for Jan-Oct train; "
        "original weekly 00:00 origins for November validation and December test"
    )

    temp_keys = ["temp"] + [f"temp_fc_tplus{h:02d}" for h in range(24)]
    workday_keys = ["workday", "workday_future24_mean"]
    month_keys = ["month_sin", "month_cos"]
    enc_ext_keys = temp_keys + workday_keys + month_keys + profile_keys
    future_ext_keys = ["temp", "workday", "month_sin", "month_cos"] + profile_keys

    expert_specs = []
    offset = 0
    for expert_name, keys in (
        ("temp", temp_keys),
        ("workday", workday_keys),
        ("month", month_keys),
        ("profile", profile_keys),
    ):
        expert_specs.append({
            "name": expert_name,
            "indices": list(range(offset, offset + len(keys))),
        })
        offset += len(keys)

    load_scalers: Dict[str, MinMaxScaler] = {}
    load_inverse: Dict[str, Tuple[float, float]] = {}
    scaler_meta = {"load": {}, "temperature": None}
    for name, df in frames.items():
        mask = df.index.year.isin(split_years) & (df.index.month <= 10)
        if not bool(mask.any()):
            raise RuntimeError(f"No January-October load available to fit scaler for {name}")
        scaler = MinMaxScaler().fit(df.loc[mask, ["load"]].to_numpy(dtype=float))
        load_scalers[name] = scaler
        load_inverse[name] = _inverse_params(scaler)
        scaler_meta["load"][name] = {
            "data_min": float(scaler.data_min_[0]),
            "data_max": float(scaler.data_max_[0]),
            "fit_scope": "selected years January-October only",
        }
        print(
            f"[SCALER][LOAD] {name}: min={scaler.data_min_[0]:.6f}, "
            f"max={scaler.data_max_[0]:.6f}"
        )

    temp_fit = []
    for df in frames.values():
        mask = df.index.year.isin(split_years) & (df.index.month <= 10)
        temp_fit.append(df.loc[mask, ["temp"]].to_numpy(dtype=float))
    temp_scaler = MinMaxScaler().fit(np.concatenate(temp_fit, axis=0))
    scaler_meta["temperature"] = {
        "data_min": float(temp_scaler.data_min_[0]),
        "data_max": float(temp_scaler.data_max_[0]),
        "fit_scope": "shared selected years January-October only",
    }
    print(
        f"[SCALER][TEMP] min={temp_scaler.data_min_[0]:.6f}, "
        f"max={temp_scaler.data_max_[0]:.6f}"
    )

    processed = {}
    profile_arrays: Dict[str, Dict[str, np.ndarray]] = {}
    for name, df in frames.items():
        z = pd.DataFrame(index=df.index)
        z["load"] = load_scalers[name].transform(
            df[["load"]].to_numpy(dtype=float)
        ).reshape(-1)
        for key in temp_keys:
            z[key] = temp_scaler.transform(
                df[[key]].to_numpy(dtype=float)
            ).reshape(-1)
        for key in workday_keys + month_keys + profile_keys:
            z[key] = df[key].to_numpy(dtype=float)
        processed[name] = z
        profile_arrays[name] = {
            "load": np.ascontiguousarray(
                z["load"].to_numpy(dtype=np.float32)[:, None]
            ),
            "enc_ext": np.ascontiguousarray(
                z[enc_ext_keys].to_numpy(dtype=np.float32)
            ),
            "future_ext": np.ascontiguousarray(
                z[future_ext_keys].to_numpy(dtype=np.float32)
            ),
        }

    records = {split: [] for split in ("train", "val", "test")}
    metadata_by_split = {split: [] for split in ("train", "val", "test")}

    for profile in profiles:
        name = str(profile["name"])
        df = processed[name]
        raw_df = frames[name]
        n_weeks = np.asarray(profile["load"]).shape[0]
        # Preserve the exact old evaluation origins: the beginning of each
        # packed target week. Training, in contrast, uses daily 00:00 origins by default.
        weekly_eval_origins = {
            week * WEEK_HOURS
            for week in range(1, n_weeks)
            if week * WEEK_HOURS + WEEK_HOURS <= len(df)
        }
        counts = {"train": 0, "val": 0, "test": 0, "excluded": 0}
        scale, shift = load_inverse[name]

        first_origin = WEEK_HOURS
        last_origin = len(df) - WEEK_HOURS
        for origin in range(first_origin, last_origin + 1):
            target_index = df.index[origin:origin + WEEK_HOURS]
            split = classify_target_week(target_index, split_year_set)
            if split is None:
                counts["excluded"] += 1
                continue

            if split == "train":
                if (origin - first_origin) % cfg.train_stride_hours != 0:
                    continue
                origin_ts = pd.Timestamp(target_index[0])
                if cfg.train_stride_hours == 24 and (
                    origin_ts.hour != 0
                    or origin_ts.minute != 0
                    or origin_ts.second != 0
                ):
                    raise ValueError(
                        f"Daily rolling training origin must start at 00:00 for {name}; "
                        f"found {origin_ts}."
                    )
            else:
                if origin not in weekly_eval_origins:
                    continue
                origin_ts = pd.Timestamp(target_index[0])
                if cfg.require_midnight_origin and (
                    origin_ts.hour != 0
                    or origin_ts.minute != 0
                    or origin_ts.second != 0
                ):
                    raise ValueError(
                        f"The evaluation 168-hour forecast must start at 00:00 for {name}; "
                        f"found {origin_ts}. Set ONCOR_REQUIRE_MIDNIGHT_ORIGIN=0 only "
                        "for an intentional shifted-origin experiment."
                    )

            full_index = df.index[origin - WEEK_HOURS:origin + WEEK_HOURS]
            if not _is_hourly_contiguous(full_index, 2 * WEEK_HOURS):
                counts["excluded"] += 1
                continue

            records[split].append((name, int(origin), float(scale), float(shift)))
            meta = {
                "profile": name,
                "week": int(origin // WEEK_HOURS - 1),
                "origin_position": int(origin),
                "origin_hour_offset_from_packed_week": int(origin % WEEK_HOURS),
                "encoder_start": str(df.index[origin - WEEK_HOURS]),
                "encoder_end": str(df.index[origin - 1]),
                "target_start": str(df.index[origin]),
                "target_end": str(df.index[origin + WEEK_HOURS - 1]),
                "raw_target_min": float(
                    raw_df.iloc[origin:origin + WEEK_HOURS]["load"].min()
                ),
                "raw_target_max": float(
                    raw_df.iloc[origin:origin + WEEK_HOURS]["load"].max()
                ),
                "origin_policy": (
                    f"daily_00:00_rolling_stride_{cfg.train_stride_hours}h"
                    if split == "train"
                    else "packed_week_start_00:00"
                ),
            }
            metadata_by_split[split].append(meta)
            counts[split] += 1
        print(f"[WINDOWS] {name}: {counts}")

    split_data = {}
    for split in ("train", "val", "test"):
        if not records[split]:
            raise RuntimeError(f"No samples generated for {split}")
        dataset = Rolling168WindowDataset(profile_arrays, records[split])
        split_data[split] = {
            "dataset": dataset,
            "metadata": metadata_by_split[split],
        }
        n = len(dataset)
        print(
            f"[{split.upper()}] windows={n}, points={n * WEEK_HOURS}, "
            f"enc=({n},168,1), future_ext=({n},168,{len(future_ext_keys)}), "
            f"target=({n},168,1)"
        )

    metadata = []
    for split in ("train", "val", "test"):
        metadata.extend({"split": split, **row} for row in metadata_by_split[split])

    return {
        "tensors": split_data,
        "enc_ext_keys": enc_ext_keys,
        "future_ext_keys": future_ext_keys,
        "expert_specs": expert_specs,
        "profile_names": profile_names,
        "split_years": split_years,
        "scaler_meta": scaler_meta,
        "metadata": pd.DataFrame(metadata),
    }


def make_loader(data: Dict, batch_size: int, shuffle: bool, num_workers: int, seed: int = 42):
    dataset = data["dataset"]
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator if shuffle else None,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )


def gaussian_nll(mu, logvar, target):
    return 0.5 * (logvar + (target - mu).pow(2) * torch.exp(-logvar)).mean()


def gaussian_crps(mu, sigma, target):
    z = (target - mu) / sigma.clamp_min(1e-8)
    normal = Normal(torch.zeros_like(z), torch.ones_like(z))
    return sigma * (
        z * (2.0 * normal.cdf(z) - 1.0)
        + 2.0 * torch.exp(normal.log_prob(z))
        - 1.0 / math.sqrt(math.pi)
    )


def kl_loss(mu_z, logvar_z):
    return (-0.5 * (1.0 + logvar_z - mu_z.pow(2) - logvar_z.exp()).sum(dim=-1)).mean()


def repeated_daily_horizon_weights(device: torch.device) -> torch.Tensor:
    day = np.ones(24, dtype=np.float32)
    day[:3] = 2.0
    day[3:6] = 1.5
    day[9:16] = np.maximum(day[9:16], 1.3)
    week = np.tile(day, 7)
    week /= week.mean()
    return torch.tensor(week, dtype=torch.float32, device=device).view(1, 168, 1)


def weighted_mse(mu, target, weights):
    return ((mu - target).pow(2) * weights).mean()


def boundary_ramp_loss(mu, target, last_enc, hours: int):
    pred_path = torch.cat([last_enc.unsqueeze(1), mu[:, :hours, :]], dim=1)
    true_path = torch.cat([last_enc.unsqueeze(1), target[:, :hours, :]], dim=1)
    return (torch.diff(pred_path, dim=1) - torch.diff(true_path, dim=1)).pow(2).mean()


def scheduled_lr(cfg: Config, epoch: int) -> float:
    """Piecewise learning-rate schedule indexed only by epoch."""
    if epoch <= cfg.lr_warmup_epochs:
        return cfg.base_lr * epoch / max(cfg.lr_warmup_epochs, 1)
    if epoch < cfg.lr_decay_epoch_1:
        return cfg.base_lr
    if epoch < cfg.lr_decay_epoch_2:
        return cfg.base_lr * 0.5
    if epoch < cfg.lr_decay_epoch_3:
        return cfg.base_lr * 0.25
    if epoch < cfg.lr_decay_epoch_4:
        return cfg.base_lr / 6.0
    return cfg.base_lr / 12.0


def build_model(cfg: Config, n_enc_ext: int, expert_specs, future_ext_dim: int, device):
    model = Direct168GRUModel(
        xprime_dim=cfg.xprime_dim,
        input_dim=cfg.input_dim,
        hidden_size=cfg.hidden_dim,
        latent_size=cfg.latent_dim,
        n_encoder_externals=n_enc_ext,
        encoder_expert_specs=expert_specs,
        future_ext_dim=future_ext_dim,
        output_len=cfg.output_len,
        output_dim=cfg.output_dim,
        encoder_mode=cfg.encoder_mode,
        dropout=cfg.dropout,
        logvar_min=cfg.logvar_min,
        logvar_max=cfg.logvar_max,
        reverse_hidden_size=cfg.reverse_hidden_dim,
        fusion_bottleneck=cfg.fusion_bottleneck,
        residual_scale=cfg.residual_scale,
        phase_summary_size=cfg.phase_summary_dim,
        future_summary_size=cfg.future_summary_dim,
        decoder_hidden_size=cfg.hourly_hidden,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] mode={cfg.encoder_mode}, parameters={n_params:,}, reverse_context=168h full week")
    return model


@torch.no_grad()
def evaluate(model, loader, metadata, cfg: Config, device, prediction_path: Optional[Path] = None):
    model.eval()
    sum_sq = sum_abs = target_abs = 0.0
    crps_sum = nll_sum = 0.0
    count = 0
    horizon_sq = np.zeros(168, dtype=np.float64)
    horizon_abs = np.zeros(168, dtype=np.float64)
    day_sq = np.zeros(7, dtype=np.float64)
    day_count = np.zeros(7, dtype=np.int64)
    week_peak_sq = 0.0
    week_peak_under = 0
    week_peak_hour_abs = 0.0
    week_count = 0
    rows = []
    meta_pos = 0

    for batch in loader:
        enc_l, enc_ext, future_ext, target, scale, shift = [x.to(device) for x in batch]
        mu, logvar, _, _ = model(
            enc_l, enc_ext, future_ext,
            epoch=cfg.epochs,
            top_k=cfg.top_k,
            warmup_epochs=cfg.moe_warmup_epochs,
            latent_mode="mean",
        )
        sigma = torch.exp(0.5 * logvar)
        error = mu - target
        sum_sq += error.pow(2).sum().item()
        sum_abs += error.abs().sum().item()
        count += error.numel()
        nll_sum += float(gaussian_nll(mu, logvar, target)) * error.numel()
        crps_sum += gaussian_crps(mu, sigma, target).sum().item()

        scale3 = scale[:, None, :]
        shift3 = shift[:, None, :]
        mu_raw = mu * scale3 + shift3
        target_raw = target * scale3 + shift3
        sigma_raw = sigma * scale3.abs()
        raw_error = mu_raw - target_raw
        target_abs += target_raw.abs().sum().item()

        horizon_sq += raw_error.pow(2).sum(dim=(0, 2)).cpu().numpy()
        horizon_abs += raw_error.abs().sum(dim=(0, 2)).cpu().numpy()
        for d in range(7):
            e = raw_error[:, d * 24:(d + 1) * 24, :]
            day_sq[d] += e.pow(2).sum().item()
            day_count[d] += e.numel()

        for b in range(mu.size(0)):
            y = target_raw[b, :, 0]
            p = mu_raw[b, :, 0]
            true_peak, true_idx = y.max(dim=0)
            pred_peak, pred_idx = p.max(dim=0)
            diff = pred_peak - true_peak
            week_peak_sq += diff.pow(2).item()
            week_peak_under += int(diff.item() < 0)
            week_peak_hour_abs += abs(int(pred_idx) - int(true_idx))
            week_count += 1

            if prediction_path is not None:
                meta = metadata[meta_pos + b]
                start = pd.Timestamp(meta["target_start"])
                for h in range(168):
                    rows.append({
                        "profile": meta["profile"],
                        "week": meta["week"],
                        "target_start": meta["target_start"],
                        "timestamp": str(start + pd.Timedelta(hours=h)),
                        "horizon": h + 1,
                        "target_norm": float(target[b, h, 0].cpu()),
                        "pred_norm": float(mu[b, h, 0].cpu()),
                        "sigma_norm": float(sigma[b, h, 0].cpu()),
                        "target_raw": float(target_raw[b, h, 0].cpu()),
                        "pred_raw": float(mu_raw[b, h, 0].cpu()),
                        "sigma_raw": float(sigma_raw[b, h, 0].cpu()),
                    })
        meta_pos += mu.size(0)

    if prediction_path is not None:
        pd.DataFrame(rows).to_csv(prediction_path, index=False)

    mse_norm = sum_sq / max(count, 1)
    raw_sq_total = float(horizon_sq.sum())
    raw_abs_total = float(horizon_abs.sum())
    return {
        "mse_norm": mse_norm,
        "crps_norm": crps_sum / max(count, 1),
        "nll_norm": nll_sum / max(count, 1),
        "rmse_raw": math.sqrt(raw_sq_total / max(count, 1)),
        "mae_raw": raw_abs_total / max(count, 1),
        "wape_raw": raw_abs_total / max(target_abs, 1e-12),
        "week_peak_rmse_raw": math.sqrt(week_peak_sq / max(week_count, 1)),
        "week_peak_under_rate": week_peak_under / max(week_count, 1),
        "week_peak_hour_mae": week_peak_hour_abs / max(week_count, 1),
        "day_rmse_raw": [
            math.sqrt(day_sq[d] / max(int(day_count[d]), 1)) for d in range(7)
        ],
        "horizon_rmse_raw": [
            math.sqrt(horizon_sq[h] / max(week_count, 1)) for h in range(168)
        ],
        "horizon_mae_raw": [
            horizon_abs[h] / max(week_count, 1) for h in range(168)
        ],
        "n_weeks": week_count,
        "n_points": count,
        "booster": model.encoder.booster_diagnostics(),
    }


def save_metric_tables(output: Path, name: str, split: str, metrics: Dict):
    pd.DataFrame({
        "horizon": np.arange(1, 169),
        "rmse_raw": metrics["horizon_rmse_raw"],
        "mae_raw": metrics["horizon_mae_raw"],
    }).to_csv(output / f"{name}_{split}_horizon_metrics.csv", index=False)
    pd.DataFrame({
        "day": np.arange(1, 8),
        "rmse_raw": metrics["day_rmse_raw"],
    }).to_csv(output / f"{name}_{split}_day_metrics.csv", index=False)


def train(model, train_loader, val_loader, val_meta, cfg: Config, device, output: Path):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.base_lr, weight_decay=cfg.weight_decay
    )
    weights = repeated_daily_horizon_weights(device)
    steps_per_epoch = len(train_loader)
    print(
        f"[SCHEDULE] steps/epoch={steps_per_epoch}, epochs={cfg.epochs}, "
        f"lr_warmup={cfg.lr_warmup_epochs}, "
        f"lr_decay=({cfg.lr_decay_epoch_1},{cfg.lr_decay_epoch_2},"
        f"{cfg.lr_decay_epoch_3},{cfg.lr_decay_epoch_4}), "
        f"kl_anneal={cfg.kl_anneal_epochs}, moe_warmup={cfg.moe_warmup_epochs}"
    )
    print(
        f"[PROTOCOL] daily rolling-origin 168h; train stride={cfg.train_stride_hours}h; "
        "autoregressive hourly GRU; validation/test use packed-week 00:00 origins"
    )
    print("[TRAINING POLICY] fixed 300-epoch schedule by default; no update budget or early stop; best validation checkpoints are retained.")

    best_mse = float("inf")
    best_crps = float("inf")
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        lr = scheduled_lr(cfg, epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr

        loss_sum = nll_sum = mse_sum = ramp_sum = kl_sum = 0.0
        batches = 0
        started = time.time()
        for batch in train_loader:
            enc_l, enc_ext, future_ext, target, _, _ = [x.to(device) for x in batch]
            optimizer.zero_grad(set_to_none=True)
            mu, logvar, mu_z, logvar_z = model(
                enc_l, enc_ext, future_ext,
                epoch=epoch,
                top_k=cfg.top_k,
                warmup_epochs=cfg.moe_warmup_epochs,
                latent_mode="sample",
            )
            nll = gaussian_nll(mu, logvar, target)
            mse = weighted_mse(mu, target, weights)
            ramp = boundary_ramp_loss(mu, target, enc_l[:, -1, :], cfg.ramp_hours)
            kl = kl_loss(mu_z, logvar_z)
            kl_w = cfg.kl_weight * min(1.0, epoch / max(cfg.kl_anneal_epochs, 1))
            loss = nll + cfg.mse_weight * mse + cfg.ramp_weight * ramp + kl_w * kl
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            loss_sum += float(loss.detach())
            nll_sum += float(nll.detach())
            mse_sum += float(mse.detach())
            ramp_sum += float(ramp.detach())
            kl_sum += float(kl.detach())
            batches += 1

        val = evaluate(model, val_loader, val_meta, cfg, device)
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val": val,
            "config": asdict(cfg),
        }
        if val["mse_norm"] < best_mse:
            best_mse = val["mse_norm"]
            torch.save(payload, output / "mse.pt")
        if val["crps_norm"] < best_crps:
            best_crps = val["crps_norm"]
            torch.save(payload, output / "crps.pt")
        torch.save(payload, output / "latest.pt")

        record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": loss_sum / max(batches, 1),
            "train_nll": nll_sum / max(batches, 1),
            "train_weighted_mse": mse_sum / max(batches, 1),
            "train_boundary_ramp": ramp_sum / max(batches, 1),
            "train_kl": kl_sum / max(batches, 1),
            "val_mse": val["mse_norm"],
            "val_crps": val["crps_norm"],
            "val_rmse_raw": val["rmse_raw"],
            "val_wape_raw": val["wape_raw"],
            "seconds": time.time() - started,
        }
        history.append(record)
        pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)

        if epoch == 1 or epoch % 5 == 0 or epoch == cfg.epochs:
            print(
                f"[TRAIN] epoch={epoch:04d}/{cfg.epochs} "
                f"loss={record['train_loss']:.6f} valMSE={val['mse_norm']:.7f} "
                f"CRPS={val['crps_norm']:.7f} RMSE={val['rmse_raw']:.4f} "
                f"WAPE={val['wape_raw']:.4f} lr={record['lr']:.2e}"
            )


def load_checkpoint(path: Path, model: nn.Module, device):
    obj = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(obj["model"] if isinstance(obj, dict) and "model" in obj else obj)
    return obj


def run() -> None:
    cfg = Config()
    cfg.validate()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")
    if device.type == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
    print(f"[PROJECT_ROOT] {PROJECT_ROOT}")

    # The long-used project data_utils_v2.py resolves processed_data/... relative
    # to the current working directory. Always run its data-loading stage from
    # M2oE2_For_Zhe, even when this script is launched inside the package folder.
    os.chdir(PROJECT_ROOT)
    print(f"[WORKDIR] {Path.cwd()}")

    output = Path(cfg.output_root) / cfg.run_name
    output.mkdir(parents=True, exist_ok=True)
    profiles = load_profiles(cfg.profile_ids())
    prepared = prepare_data(profiles, cfg, device)
    prepared["metadata"].to_csv(output / "split_metadata.csv", index=False)

    train_loader = make_loader(
        prepared["tensors"]["train"], cfg.batch_size, True, cfg.num_workers, cfg.seed
    )
    val_loader = make_loader(
        prepared["tensors"]["val"], cfg.batch_size, False, cfg.num_workers
    )
    test_loader = make_loader(
        prepared["tensors"]["test"], cfg.batch_size, False, cfg.num_workers
    )
    model = build_model(
        cfg,
        len(prepared["enc_ext_keys"]),
        prepared["expert_specs"],
        len(prepared["future_ext_keys"]),
        device,
    )

    config_payload = {
        **asdict(cfg),
        "method": "rolling-origin 168hr GRU forecast",
        "uses_target_week_load": False,
        "uses_same_hour_previous_week_as_input": True,
        "uses_hard_previous_week_skip": False,
        "uses_hourly_future_external_directly": True,
        "uses_explicit_hour_or_dow_encoding": False,
        "input_load_hours": 168,
        "output_hours": 168,
        "train_origin_policy": "daily 00:00 rolling 168h history to 168h target",
        "train_origin_stride_hours": cfg.train_stride_hours,
        "eval_origin_policy": "original packed-week starts at 00:00",
        "enc_ext_keys": prepared["enc_ext_keys"],
        "future_ext_keys": prepared["future_ext_keys"],
        "expert_specs": prepared["expert_specs"],
        "profile_names": prepared["profile_names"],
        "split_years": prepared["split_years"],
        "scaler_meta": prepared["scaler_meta"],
        "split_definition": (
            "daily 00:00 rolling 168h targets wholly inside Jan-Oct=train; original packed-week "
            "00:00 targets wholly inside November=val and December=test; "
            "split-boundary-crossing horizons excluded"
        ),
        "checkpoint_policy": "train for a fixed epoch count and retain best MSE/CRPS plus latest checkpoints",
        "training_control": "epoch_based_only",
        "reverse_context_hours": 168,
        "reverse_time_order": "most recent encoder hour back to oldest encoder hour",
        "booster_definition": "full preceding 168h window encoded in reverse and shared across seven relative 24h segments",
        "decoder_change": (
            "hourly GRU initialized by encoder, phase and future summaries; "
            "each step uses previous prediction, local future external and previous-week same-hour load"
        ),
    }
    with (output / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)

    best_path = output / "mse.pt"
    checkpoint_candidates = [
        best_path,
        output / "crps.pt",
        output / "latest.pt",
    ]
    existing_checkpoint = next((p for p in checkpoint_candidates if p.exists()), None)
    if cfg.eval_only:
        if existing_checkpoint is None:
            raise FileNotFoundError(f"No checkpoint available for eval-only in: {output}")
    elif cfg.force_retrain or existing_checkpoint is None:
        train(
            model,
            train_loader,
            val_loader,
            prepared["tensors"]["val"]["metadata"],
            cfg,
            device,
            output,
        )
    else:
        print(f"[LOAD] existing checkpoint: {existing_checkpoint}")

    results = {}
    for ckpt_name in ("mse", "crps"):
        ckpt = output / f"{ckpt_name}.pt"
        if not ckpt.exists():
            continue
        load_checkpoint(ckpt, model, device)
        results[ckpt_name] = {}
        for split, loader in (("val", val_loader), ("test", test_loader)):
            pred_path = (
                output / f"{ckpt_name}_{split}_predictions.csv"
                if cfg.save_predictions else None
            )
            metrics = evaluate(
                model,
                loader,
                prepared["tensors"][split]["metadata"],
                cfg,
                device,
                prediction_path=pred_path,
            )
            results[ckpt_name][split] = metrics
            save_metric_tables(output, ckpt_name, split, metrics)
            print(
                f"[{ckpt_name.upper()}][{split.upper()}] "
                f"MSE={metrics['mse_norm']:.7f} CRPS={metrics['crps_norm']:.7f} "
                f"RMSE={metrics['rmse_raw']:.5f} WAPE={metrics['wape_raw']:.4f} "
                f"weekPeakRMSE={metrics['week_peak_rmse_raw']:.5f} "
                f"weekPeakUnder={metrics['week_peak_under_rate']:.4f} "
                f"peakHourMAE={metrics['week_peak_hour_mae']:.3f}"
            )

    with (output / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[DONE] {output}")


if __name__ == "__main__":
    run()
