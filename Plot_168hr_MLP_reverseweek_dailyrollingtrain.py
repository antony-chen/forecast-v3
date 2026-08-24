from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.preprocessing import MinMaxScaler

from data_utils import build_predictions_table
from Main_168hr_MLP_reverseweek_dailyrollingtrain_epoch300 import (
    Config,
    WEEK_HOURS,
    build_frame,
    build_model,
    infer_split_years,
    load_checkpoint,
    load_profiles,
    set_seed,
)

plt.switch_backend("Agg")

# ============================================================
# Defaults
# ============================================================
_DEFAULT_CFG = Config()
DEFAULT_OUTPUT_ROOT = Path(os.getenv("ONCOR_OUTPUT_ROOT", _DEFAULT_CFG.output_root))
DEFAULT_RUN_NAME = os.getenv("ONCOR_RUN_NAME", _DEFAULT_CFG.run_name)
DEFAULT_RUN_DIR = DEFAULT_OUTPUT_ROOT / DEFAULT_RUN_NAME

RUN_DIR = Path(os.getenv("ONCOR_PLOT_RUN_DIR", str(DEFAULT_RUN_DIR)))
CHECKPOINT_NAME = os.getenv("ONCOR_PLOT_CHECKPOINT", "mse")
TARGET_PROFILE = os.getenv("ONCOR_PLOT_PROFILE", "").strip()  # default: use last profile
TARGET_YEAR = int(os.getenv("ONCOR_PLOT_YEAR", "2024"))
SHOW_TEMPERATURE = os.getenv("ONCOR_PLOT_SHOW_TEMP", "1") == "1"
OUTPUT_DIR = Path(
    os.getenv(
        "ONCOR_PLOT_OUTPUT_DIR",
        str(RUN_DIR / f"year_monthly_hist_forecast_plots_{CHECKPOINT_NAME}"),
    )
)
FIG_DPI = int(os.getenv("ONCOR_PLOT_DPI", "180"))
# 0 (default): use an automatic load range for each monthly figure.
# 1: use one shared load range for the full year.
UNIFY_YLIM = os.getenv("ONCOR_PLOT_UNIFY_YLIM", "0") == "1"


# ============================================================
# Helpers
# ============================================================
def build_cfg_from_run(run_dir: Path) -> Config:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {run_dir}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg_json = json.load(f)

    cfg = Config()
    field_names = {f.name for f in dataclass_fields(cfg)}
    for k, v in cfg_json.items():
        # Only overwrite actual dataclass fields. config.json also stores
        # informational keys (e.g. "split_years", the list of years used for
        # this run) whose names can collide with Config *methods* of the same
        # name (Config.split_years()) -- hasattr() would match those too and
        # clobber the method with data, breaking every later call to it.
        if k in field_names:
            setattr(cfg, k, v)
    return cfg


def _inverse_params(scaler: MinMaxScaler) -> Tuple[float, float]:
    lo, hi = float(scaler.data_min_[0]), float(scaler.data_max_[0])
    fr_lo, fr_hi = map(float, scaler.feature_range)
    scale = (hi - lo) / max(fr_hi - fr_lo, 1e-12)
    shift = lo - fr_lo * scale
    return scale, shift


def classify_split(ts: pd.Timestamp) -> str:
    m = ts.month
    if 1 <= m <= 10:
        return "train"
    if m == 11:
        return "val"
    return "test"


def prepare_processed_frames(profiles: List[Dict], cfg: Config):
    profile_names = [str(p["name"]) for p in profiles]
    profile_to_idx = {name: i for i, name in enumerate(profile_names)}
    profile_keys = [f"profile_onehot_{i:02d}" for i in range(len(profile_names))]

    frames = {str(p["name"]): build_frame(p) for p in profiles}
    for name, df in frames.items():
        idx = profile_to_idx[name]
        for j, key in enumerate(profile_keys):
            df[key] = 1.0 if j == idx else 0.0

    split_years = infer_split_years(frames, cfg.split_years())

    temp_keys = ["temp"] + [f"temp_fc_tplus{h:02d}" for h in range(24)]
    workday_keys = ["workday", "workday_future24_mean"]
    month_keys = ["month_sin", "month_cos"]
    enc_ext_keys = temp_keys + workday_keys + month_keys + profile_keys
    future_ext_keys = ["temp", "workday", "month_sin", "month_cos"] + profile_keys

    load_scalers = {}
    inverse_params = {}
    for name, df in frames.items():
        mask = df.index.year.isin(split_years) & (df.index.month <= 10)
        scaler = MinMaxScaler().fit(df.loc[mask, ["load"]].to_numpy(dtype=float))
        load_scalers[name] = scaler
        inverse_params[name] = _inverse_params(scaler)

    temp_fit = []
    for df in frames.values():
        mask = df.index.year.isin(split_years) & (df.index.month <= 10)
        temp_fit.append(df.loc[mask, ["temp"]].to_numpy(dtype=float))
    temp_scaler = MinMaxScaler().fit(np.concatenate(temp_fit, axis=0))

    processed = {}
    for name, df in frames.items():
        z = pd.DataFrame(index=df.index)
        z["load"] = load_scalers[name].transform(df[["load"]].to_numpy(dtype=float)).reshape(-1)
        for key in temp_keys:
            z[key] = temp_scaler.transform(df[[key]].to_numpy(dtype=float)).reshape(-1)
        for key in workday_keys + month_keys + profile_keys:
            z[key] = df[key].to_numpy(dtype=float)
        processed[name] = z

    return {
        "frames": frames,
        "processed": processed,
        "enc_ext_keys": enc_ext_keys,
        "future_ext_keys": future_ext_keys,
        "profile_names": profile_names,
        "inverse_params": inverse_params,
    }


@torch.no_grad()
def predict_week(model, cfg: Config, enc_l, enc_ext, future_ext):
    mu, logvar, _, _ = model(
        enc_l,
        enc_ext,
        future_ext,
        epoch=cfg.epochs,
        top_k=cfg.top_k,
        warmup_epochs=cfg.moe_warmup_epochs,
        latent_mode="mean",
    )
    return mu.cpu().numpy()[0, :, 0], logvar.cpu().numpy()[0, :, 0]


def build_week_records(model, cfg: Config, profiles: List[Dict], prepared: Dict, device: torch.device, target_profile: str):
    frames = prepared["frames"]
    processed = prepared["processed"]
    enc_ext_keys = prepared["enc_ext_keys"]
    future_ext_keys = prepared["future_ext_keys"]
    scale, shift = prepared["inverse_params"][target_profile]

    profile_obj = None
    for p in profiles:
        if str(p["name"]) == target_profile:
            profile_obj = p
            break
    if profile_obj is None:
        raise ValueError(f"Profile not found: {target_profile}")

    raw_df = frames[target_profile]
    proc_df = processed[target_profile]
    n_weeks = np.asarray(profile_obj["load"]).shape[0]
    records = []

    for week in range(n_weeks - 1):
        enc_start = week * WEEK_HOURS
        enc_end = enc_start + WEEK_HOURS
        dec_start = enc_end
        dec_end = dec_start + WEEK_HOURS
        if dec_end > len(proc_df):
            continue

        full_index = proc_df.index[enc_start:dec_end]
        expected = pd.date_range(full_index[0], periods=2 * WEEK_HOURS, freq="h")
        if len(full_index) != 2 * WEEK_HOURS or not full_index.equals(expected):
            continue

        enc_proc = proc_df.iloc[enc_start:enc_end]
        dec_proc = proc_df.iloc[dec_start:dec_end]
        enc_raw = raw_df.iloc[enc_start:enc_end]
        dec_raw = raw_df.iloc[dec_start:dec_end]

        enc_l = torch.tensor(enc_proc[["load"]].to_numpy()[None, ...], dtype=torch.float32, device=device)
        enc_ext = torch.tensor(enc_proc[enc_ext_keys].to_numpy()[None, ...], dtype=torch.float32, device=device)
        future_ext = torch.tensor(dec_proc[future_ext_keys].to_numpy()[None, ...], dtype=torch.float32, device=device)

        pred_mu_norm, pred_logvar_norm = predict_week(model, cfg, enc_l, enc_ext, future_ext)
        pred_mu_raw = pred_mu_norm * scale + shift
        pred_sigma_raw = np.exp(0.5 * pred_logvar_norm) * abs(scale)

        true_week = dec_raw["load"].to_numpy(dtype=float)
        err = pred_mu_raw - true_week
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        coverage = float(np.mean((true_week >= pred_mu_raw - pred_sigma_raw) & (true_week <= pred_mu_raw + pred_sigma_raw)))

        start_ts = pd.Timestamp(dec_raw.index[0])
        records.append(
            {
                "week": week,
                "split": classify_split(start_ts),
                "enc_start": enc_raw.index[0],
                "enc_end": enc_raw.index[-1],
                "dec_start": dec_raw.index[0],
                "dec_end": dec_raw.index[-1],
                "history_load": enc_raw["load"].to_numpy(dtype=float),
                "history_temp": enc_raw["temp"].to_numpy(dtype=float),
                "forecast_true": true_week,
                "forecast_temp": dec_raw["temp"].to_numpy(dtype=float),
                "forecast_mu": pred_mu_raw,
                "forecast_sigma": pred_sigma_raw,
                "rmse": rmse,
                "mae": mae,
                "coverage_1sigma": coverage,
                "full_time": pd.date_range(enc_raw.index[0], periods=2 * WEEK_HOURS, freq="h"),
            }
        )
    return records


def _make_legend_handles(show_temp: bool):
    handles = [
        Line2D([0], [0], color="0.45", lw=1.2, label="History (previous 168h)"),
        Line2D([0], [0], color="k", lw=1.5, label="True (forecast week)"),
        Line2D([0], [0], color="red", lw=1.7, label="Prediction mean"),
        Patch(facecolor="#f2a3a3", edgecolor="none", alpha=0.45, label="Prediction +/- 1 sigma"),
    ]
    if show_temp:
        handles.extend(
            [
                Line2D([0], [0], color="#6bb7ff", lw=1.1, ls=":", label="Temp (history)"),
                Line2D([0], [0], color="#f4a261", lw=1.1, ls=":", label="Temp (forecast week)"),
            ]
        )
    return handles


def _nice_load_ylim(month_records: List[Dict]) -> Tuple[float, float]:
    """Return a tight load range without forcing the lower limit to zero."""
    vals = []
    for rec in month_records:
        vals.extend([
            np.asarray(rec["history_load"], dtype=float),
            np.asarray(rec["forecast_true"], dtype=float),
            np.asarray(rec["forecast_mu"] - rec["forecast_sigma"], dtype=float),
            np.asarray(rec["forecast_mu"] + rec["forecast_sigma"], dtype=float),
        ])

    vmax = max(float(np.nanmax(v)) for v in vals)
    vmin = min(float(np.nanmin(v)) for v in vals)
    span = max(vmax - vmin, 1e-6)
    reference = max(abs(vmin), abs(vmax), 1.0)
    pad = max(0.08 * span, 0.015 * reference)
    return vmin - pad, vmax + pad


def plot_month(records: List[Dict], month: int, out_path: Path, profile_name: str, checkpoint_name: str, global_ylim: Tuple[float, float] | None = None):
    n = len(records)
    if n == 0:
        return

    fig_h = max(4.4 * n + 1.8, 6.5)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(17.5, fig_h), sharex=False)
    axes = np.atleast_1d(axes).tolist()
    fig.subplots_adjust(top=0.90, hspace=0.30)

    y_limits = global_ylim if global_ylim is not None else _nice_load_ylim(records)

    for i, rec in enumerate(records):
        ax = axes[i]
        x_hist = np.arange(WEEK_HOURS)
        x_fc = np.arange(WEEK_HOURS, 2 * WEEK_HOURS)

        full_time = pd.DatetimeIndex(rec["full_time"])
        day_ticks = np.arange(0, 2 * WEEK_HOURS, 24)
        day_labels = [pd.Timestamp(full_time[t]).strftime("%m-%d") for t in day_ticks]

        # The history and forecast timestamps are consecutive. Start each
        # forecast-side curve from the final observed history value so the
        # plotted time series is visually continuous.
        x_cont = np.arange(WEEK_HOURS - 1, 2 * WEEK_HOURS)
        history_last = float(rec["history_load"][-1])
        true_cont = np.concatenate(([history_last], rec["forecast_true"]))
        mean_cont = np.concatenate(([history_last], rec["forecast_mu"]))

        ax.axvline(WEEK_HOURS - 0.5, color="0.4", ls="--", lw=1.0)
        ax.plot(x_hist, rec["history_load"], color="0.45", lw=1.2)
        ax.plot(x_cont, true_cont, color="k", lw=1.5)
        ax.plot(x_cont, mean_cont, color="red", lw=1.7)
        ax.fill_between(
            x_fc,
            rec["forecast_mu"] - rec["forecast_sigma"],
            rec["forecast_mu"] + rec["forecast_sigma"],
            color="#f2a3a3",
            alpha=0.45,
            linewidth=0,
        )


        ax.set_ylabel("Load")
        ax.set_ylim(*y_limits)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(0, 2 * WEEK_HOURS - 1)
        ax.set_xticks(day_ticks)
        ax.set_xticklabels(day_labels, rotation=25, ha="right")

        if SHOW_TEMPERATURE:
            ax_t = ax.twinx()
            temp_cont = np.concatenate(([float(rec["history_temp"][-1])], rec["forecast_temp"]))
            ax_t.plot(x_hist, rec["history_temp"], color="#6bb7ff", ls=":", lw=1.1)
            ax_t.plot(x_cont, temp_cont, color="#f4a261", ls=":", lw=1.1)
            ax_t.set_ylabel("Temperature (°F)")
            temp_all = np.concatenate([rec["history_temp"], rec["forecast_temp"]])
            tmin, tmax = float(np.min(temp_all)), float(np.max(temp_all))
            pad = 0.05 * max(tmax - tmin, 1.0)
            ax_t.set_ylim(tmin - pad, tmax + pad)

        title = (
            f"168hr MLP Reverse-Week | Profile {profile_name} | "
            f"Forecast week {pd.Timestamp(rec['dec_start']).strftime('%Y-%m-%d %H:%M')} | "
            f"{rec['split']} | RMSE={rec['rmse']:.3f}, MAE={rec['mae']:.3f}, "
            f"+/-1sigma coverage={100.0 * rec['coverage_1sigma']:.1f}%"
        )
        ax.set_title(title, fontsize=10)

        if i == n - 1:
            ax.set_xlabel("Previous 168h history (reverse encoder) | single-shot 168h forecast for the following week")

    fig.suptitle(
        f"168hr MLP Reverse-Week | Profile {profile_name} | {TARGET_YEAR}-{month:02d} | "
        "single-shot 168h forecast from a fully reverse-encoded 168h history week"
        + (" | with temperature" if SHOW_TEMPERATURE else ""),
        fontsize=14,
        y=0.985,
    )
    fig.legend(
        handles=_make_legend_handles(SHOW_TEMPERATURE),
        loc="upper center",
        ncol=min(7 if SHOW_TEMPERATURE else 5, len(_make_legend_handles(SHOW_TEMPERATURE))),
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, 0.965),
    )
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main():
    if not RUN_DIR.exists():
        raise FileNotFoundError(f"Run directory does not exist: {RUN_DIR}")
    ckpt_path = RUN_DIR / f"{CHECKPOINT_NAME}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    cfg = build_cfg_from_run(RUN_DIR)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")
    if device.type == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}")

    profiles = load_profiles(cfg.profile_ids())
    prepared = prepare_processed_frames(profiles, cfg)
    profile_names = prepared["profile_names"]

    plot_profile = TARGET_PROFILE if TARGET_PROFILE else profile_names[-1]
    if plot_profile not in profile_names:
        raise ValueError(f"Requested profile {plot_profile} not in {profile_names}")
    print(f"[PLOT_PROFILE] {plot_profile}")

    # Build expert specs from saved config when present.
    expert_specs = None
    config_json = json.loads((RUN_DIR / "config.json").read_text(encoding="utf-8"))
    if "expert_specs" in config_json:
        expert_specs = config_json["expert_specs"]

    model = build_model(
        cfg,
        len(prepared["enc_ext_keys"]),
        expert_specs,
        len(prepared["future_ext_keys"]),
        device,
    )
    load_checkpoint(ckpt_path, model, device)
    model.eval()

    records = build_week_records(model, cfg, profiles, prepared, device, plot_profile)
    records = [r for r in records if pd.Timestamp(r["dec_start"]).year == TARGET_YEAR]
    if not records:
        raise RuntimeError(f"No weekly records found for profile={plot_profile}, year={TARGET_YEAR}")

    month_groups = defaultdict(list)
    for rec in records:
        month_groups[pd.Timestamp(rec["dec_start"]).month].append(rec)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    global_ylim = None
    if UNIFY_YLIM:
        global_ylim = _nice_load_ylim(records)

    summary_rows = []
    for month in range(1, 13):
        recs = month_groups.get(month, [])
        if not recs:
            continue
        recs = sorted(recs, key=lambda r: pd.Timestamp(r["dec_start"]))
        out_path = OUTPUT_DIR / f"{plot_profile}_{TARGET_YEAR}_month{month:02d}_{CHECKPOINT_NAME}_probabilistic.png"
        plot_month(recs, month, out_path, plot_profile, CHECKPOINT_NAME, global_ylim=global_ylim)
        predictions_path = OUTPUT_DIR / f"{plot_profile}_{TARGET_YEAR}_month{month:02d}_{CHECKPOINT_NAME}_predictions.csv"
        build_predictions_table(recs, plot_profile).to_csv(predictions_path, index=False)
        summary_rows.append({
            "month": month,
            "weeks_in_figure": len(recs),
            "file": str(out_path),
            "predictions_file": str(predictions_path),
            "first_decoder_start": str(recs[0]["dec_start"]),
            "last_decoder_start": str(recs[-1]["dec_start"]),
            "split_labels": ",".join(sorted({r['split'] for r in recs})),
        })
        print(f"[SAVED] {out_path}")
        print(f"[SAVED] {predictions_path}")

    pd.DataFrame(summary_rows).to_csv(
        OUTPUT_DIR / f"{plot_profile}_{TARGET_YEAR}_{CHECKPOINT_NAME}_monthly_manifest.csv",
        index=False,
    )
    print(f"[DONE] output_dir={OUTPUT_DIR}")


if __name__ == "__main__":
    main()
