#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Station residuals and local uncertainty diagnostics for the coupled run."""

from __future__ import annotations

import importlib.util
import json
import math
import shutil
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN = (
    ROOT
    / "output_sgc_new_events_joint_vp_relocation_fmm_min4_plus_shallow"
)
DATA = RUN / "data"
FIG = RUN / "figures"
MODEL = RUN / "models"
CODE = RUN / "codigos_finales"
JOINT_SCRIPT = ROOT / "codigos" / "run_sgc_new_events_joint_vp_relocation.py"

LOCAL_RADIUS_NODES = 2
RMS_DELTA_THRESHOLD_S = 0.20
RMS_DELTA_TIGHT_S = 0.10


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def robust_mad(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    med = np.median(values)
    return float(1.4826 * np.median(np.abs(values - med)))


def rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(values * values)))


def azimuthal_gap(group: pd.DataFrame) -> float:
    ex = float(group["source_x_km"].iloc[0])
    ey = float(group["source_y_km"].iloc[0])
    dx = group["receiver_x_km"].to_numpy(dtype=float) - ex
    dy = group["receiver_y_km"].to_numpy(dtype=float) - ey
    az = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0
    az = np.unique(np.round(az, 3))
    if az.size < 2:
        return 360.0
    az = np.sort(az)
    gaps = np.diff(np.r_[az, az[0] + 360.0])
    return float(np.max(gaps))


def station_residual_summary(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for station, group in pred.groupby("station"):
        residual = group["residual_final_s"].to_numpy(dtype=float)
        rows.append(
            {
                "station": station,
                "n_picks": int(len(group)),
                "mean_residual_s": float(np.nanmean(residual)),
                "median_residual_s": float(np.nanmedian(residual)),
                "rms_residual_s": rms(residual),
                "mad_residual_s": robust_mad(residual),
                "p90_abs_residual_s": float(np.nanpercentile(np.abs(residual), 90)),
                "station_static_s": float(np.nanmedian(group["station_static_s"])),
            }
        )
    out = pd.DataFrame(rows).sort_values("n_picks", ascending=False)
    path = DATA / "location_station_residual_summary.csv"
    out.to_csv(path, index=False)
    return out


def event_residual_summary(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for event_id, group in pred.groupby("event_id", sort=False):
        residual = group["residual_final_s"].to_numpy(dtype=float)
        rows.append(
            {
                "event_id": event_id,
                "n_picks": int(len(group)),
                "depth_km": float(group["event_depth_km"].iloc[0]),
                "selection_group": str(group["selection_group"].iloc[0]),
                "rms_residual_s": rms(residual),
                "mad_residual_s": robust_mad(residual),
                "p90_abs_residual_s": float(np.nanpercentile(np.abs(residual), 90)),
                "median_residual_s": float(np.nanmedian(residual)),
                "azimuthal_gap_deg": azimuthal_gap(group),
            }
        )
    out = pd.DataFrame(rows)
    path = DATA / "location_event_residual_summary.csv"
    out.to_csv(path, index=False)
    return out


def load_model():
    grid = np.load(MODEL / "sgc_new_events_grid_10km.npz")
    xg, yg, zg = grid["xg"], grid["yg"], grid["zg"]
    vp = np.load(MODEL / "sgc_new_events_vp_final_joint_relocation.npy")
    return xg, yg, zg, vp


def local_uncertainty_diagnostic(joint, qmod, picks: pd.DataFrame, xg, yg, zg, vp: np.ndarray) -> pd.DataFrame:
    station_statics = pd.read_csv(DATA / "sgc_new_events_station_statics.csv").set_index("station")["static_s"].to_dict()
    station_nodes = joint.receiver_nodes(qmod, picks, xg, yg, zg)
    station_cache, cache_seconds = joint.build_station_cache(qmod, vp, station_nodes, xg, yg, zg, return_predecessors=False)
    ny, nx = vp.shape[1], vp.shape[2]
    rows = []
    t0 = time.time()
    for n, (event_id, group) in enumerate(picks.groupby("event_id", sort=False), start=1):
        center = qmod.coord_to_index_xyz(
            float(group["source_x_km"].iloc[0]),
            float(group["source_y_km"].iloc[0]),
            float(group["source_z_km"].iloc[0]),
            xg,
            yg,
            zg,
        )
        obs = group["travel_time_s"].to_numpy(dtype=float)
        stations = group["station"].astype(str).tolist()
        candidates = []
        for cand in joint.candidate_nodes(center, vp.shape, LOCAL_RADIUS_NODES, zg=zg):
            cand_idx = qmod.idx_linear(cand[0], cand[1], cand[2], ny, nx)
            stat = np.array([float(station_statics.get(sta, 0.0)) for sta in stations], dtype=float)
            tcalc = np.array([station_cache[sta]["dist"][cand_idx] for sta in stations], dtype=float) + stat
            if not np.all(np.isfinite(tcalc)):
                continue
            dt0 = float(np.median(obs - tcalc))
            residual = obs - (tcalc + dt0)
            rr = rms(residual)
            x, y, z = qmod.coord_from_kji(cand, xg, yg, zg)
            candidates.append((float(x), float(y), float(z), float(dt0), rr))
        if not candidates:
            continue
        cand_df = pd.DataFrame(candidates, columns=["x_km", "y_km", "z_km", "dt0_s", "rms_s"])
        min_rms = float(cand_df["rms_s"].min())
        accepted = cand_df[cand_df["rms_s"] <= min_rms + RMS_DELTA_THRESHOLD_S].copy()
        tight = cand_df[cand_df["rms_s"] <= min_rms + RMS_DELTA_TIGHT_S].copy()
        center_x, center_y, center_z = qmod.coord_from_kji(center, xg, yg, zg)

        def spans(df: pd.DataFrame) -> tuple[float, float, float]:
            if df.empty:
                return float("nan"), float("nan"), float("nan")
            hdist = np.sqrt((df["x_km"] - center_x) ** 2 + (df["y_km"] - center_y) ** 2)
            return (
                float(np.nanmax(hdist)),
                float(df["z_km"].max() - df["z_km"].min()),
                float(df["dt0_s"].std(ddof=0)) if len(df) > 1 else 0.0,
            )

        horiz_span, depth_span, dt0_std = spans(accepted)
        horiz_tight, depth_tight, dt0_tight = spans(tight)
        rows.append(
            {
                "event_id": event_id,
                "n_picks": int(len(group)),
                "depth_km": float(group["source_z_km"].iloc[0]),
                "azimuthal_gap_deg": azimuthal_gap(group),
                "min_local_rms_s": min_rms,
                "candidate_nodes": int(len(cand_df)),
                "accepted_nodes_delta_0p20s": int(len(accepted)),
                "accepted_nodes_delta_0p10s": int(len(tight)),
                "horizontal_uncertainty_0p20_km": horiz_span,
                "depth_uncertainty_0p20_km": depth_span,
                "origin_time_uncertainty_0p20_s": dt0_std,
                "horizontal_uncertainty_0p10_km": horiz_tight,
                "depth_uncertainty_0p10_km": depth_tight,
                "origin_time_uncertainty_0p10_s": dt0_tight,
            }
        )
        if n % 300 == 0:
            print(f"[uncertainty] {n}/{picks['event_id'].nunique()} events in {time.time() - t0:.1f}s", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "location_local_grid_uncertainty_summary.csv", index=False)
    print(f"[uncertainty] station cache seconds={cache_seconds:.2f}", flush=True)
    return out


def make_figures(sta: pd.DataFrame, evt: pd.DataFrame, unc: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=240)
    order = sta.sort_values("n_picks", ascending=False)["station"].tolist()
    axes[0].bar(order, sta.set_index("station").loc[order, "rms_residual_s"], color="#4c78a8")
    axes[0].set_ylabel("Final residual RMS (s)")
    axes[0].set_xlabel("Station")
    axes[0].set_title("Residual level by station")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].bar(order, sta.set_index("station").loc[order, "station_static_s"], color="#f58518")
    axes[1].axhline(0.0, color="0.25", lw=0.8)
    axes[1].set_ylabel("Station static (s)")
    axes[1].set_xlabel("Station")
    axes[1].set_title("Estimated station statics")
    axes[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    station_fig = FIG / "location_station_residuals_and_statics.png"
    fig.savefig(station_fig, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=240)
    axes[0].hist(evt["rms_residual_s"], bins=40, color="#4c78a8", alpha=0.85)
    axes[0].axvline(evt["rms_residual_s"].median(), color="k", lw=1.0, label="median")
    axes[0].set_xlabel("Event residual RMS (s)")
    axes[0].set_ylabel("Events")
    axes[0].set_title("Event-level fit")
    axes[0].legend(fontsize=8)
    axes[1].scatter(evt["azimuthal_gap_deg"], evt["rms_residual_s"], s=8, alpha=0.35, color="#54a24b")
    axes[1].set_xlabel("Azimuthal gap (deg)")
    axes[1].set_ylabel("Event RMS (s)")
    axes[1].set_title("Geometry vs residual")
    axes[2].scatter(evt["n_picks"], evt["rms_residual_s"], s=10, alpha=0.35, color="#e45756")
    axes[2].set_xlabel("Number of P picks")
    axes[2].set_ylabel("Event RMS (s)")
    axes[2].set_title("Pick count vs residual")
    fig.tight_layout()
    event_fig = FIG / "location_event_uncertainty_diagnostics.png"
    fig.savefig(event_fig, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=240)
    axes[0].hist(unc["horizontal_uncertainty_0p20_km"], bins=np.arange(-0.5, 35.5, 2.5), color="#4c78a8", alpha=0.85)
    axes[0].set_xlabel("Horizontal uncertainty proxy (km)")
    axes[0].set_ylabel("Events")
    axes[0].set_title("Accepted nodes within +0.20 s")
    axes[1].hist(unc["depth_uncertainty_0p20_km"], bins=np.arange(-0.5, 65.5, 5), color="#f58518", alpha=0.85)
    axes[1].set_xlabel("Depth span (km)")
    axes[1].set_title("Depth-time trade-off proxy")
    axes[2].scatter(unc["depth_uncertainty_0p20_km"], unc["origin_time_uncertainty_0p20_s"], s=9, alpha=0.35, color="#b279a2")
    axes[2].set_xlabel("Depth span (km)")
    axes[2].set_ylabel("Origin-time std within accepted nodes (s)")
    axes[2].set_title("Depth vs origin-time ambiguity")
    fig.tight_layout()
    uncertainty_fig = FIG / "location_local_grid_uncertainty.png"
    fig.savefig(uncertainty_fig, bbox_inches="tight")
    plt.close(fig)
    return station_fig, event_fig, uncertainty_fig


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    CODE.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(DATA / "sgc_new_events_final_vp_predictions.csv")
    picks = pd.read_csv(DATA / "sgc_new_events_final_relocated_picks.csv")
    sta = station_residual_summary(pred)
    evt = event_residual_summary(pred)

    joint = import_module(JOINT_SCRIPT, "joint_helpers_for_uncertainty")
    qmod = joint.import_module(joint.Q_FSM_SCRIPT, "q_fsm_helpers_for_uncertainty")
    xg, yg, zg, vp = load_model()
    unc = local_uncertainty_diagnostic(joint, qmod, picks, xg, yg, zg, vp)
    station_fig, event_fig, uncertainty_fig = make_figures(sta, evt, unc)

    metrics = {
        "station_count": int(sta["station"].nunique()),
        "event_count": int(evt["event_id"].nunique()),
        "median_event_rms_s": float(evt["rms_residual_s"].median()),
        "p90_event_rms_s": float(evt["rms_residual_s"].quantile(0.90)),
        "median_event_mad_s": float(evt["mad_residual_s"].median()),
        "median_azimuthal_gap_deg": float(evt["azimuthal_gap_deg"].median()),
        "p90_azimuthal_gap_deg": float(evt["azimuthal_gap_deg"].quantile(0.90)),
        "median_horizontal_uncertainty_0p20_km": float(unc["horizontal_uncertainty_0p20_km"].median()),
        "median_depth_uncertainty_0p20_km": float(unc["depth_uncertainty_0p20_km"].median()),
        "median_origin_time_uncertainty_0p20_s": float(unc["origin_time_uncertainty_0p20_s"].median()),
        "fraction_depth_span_le_20km": float((unc["depth_uncertainty_0p20_km"] <= 20.0).mean()),
        "fraction_horizontal_uncertainty_le_15km": float((unc["horizontal_uncertainty_0p20_km"] <= 15.0).mean()),
        "station_max_abs_static_s": float(sta["station_static_s"].abs().max()),
        "station_max_rms_s": float(sta["rms_residual_s"].max()),
        "figures": {
            "station": str(station_fig),
            "event": str(event_fig),
            "uncertainty": str(uncertainty_fig),
        },
        "tables": {
            "station_residuals": str(DATA / "location_station_residual_summary.csv"),
            "event_residuals": str(DATA / "location_event_residual_summary.csv"),
            "local_grid_uncertainty": str(DATA / "location_local_grid_uncertainty_summary.csv"),
        },
    }
    metrics_path = RUN / "location_uncertainty_station_diagnostics_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    shutil.copy2(Path(__file__), CODE / Path(__file__).name)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
