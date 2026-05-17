#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare the preferred slab-informed run with the current-dataset 1-D run."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


ROOT = Path(__file__).resolve().parents[1]
SENS = ROOT.parent / "output_sgc_new_events_joint_vp_relocation_fmm_min4_plus_shallow_1d_current"
OUT_DATA = ROOT / "data"
OUT_FIG = ROOT / "figures"
OUT_MODEL = ROOT / "models"


def corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float).ravel()
    bb = np.asarray(b, dtype=float).ravel()
    ok = np.isfinite(aa) & np.isfinite(bb)
    if ok.sum() < 3:
        return float("nan")
    aa = aa[ok] - np.nanmean(aa[ok])
    bb = bb[ok] - np.nanmean(bb[ok])
    den = np.sqrt(np.sum(aa * aa) * np.sum(bb * bb))
    return float(np.sum(aa * bb) / den) if den > 0 else float("nan")


def rms(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.sqrt(np.mean(arr * arr))) if arr.size else float("nan")


def event_rms(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = (
        df.groupby("event_id")["residual_final_s"]
        .apply(lambda s: rms(s.to_numpy(dtype=float)))
        .reset_index(name="event_final_rms_s")
    )
    return out


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def smooth_nan(arr: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    values = np.where(mask & np.isfinite(arr), arr, 0.0)
    weights = (mask & np.isfinite(arr)).astype(float)
    smoothed_values = gaussian_filter(values, sigma=sigma, mode="nearest")
    smoothed_weights = gaussian_filter(weights, sigma=sigma, mode="nearest")
    return np.where(smoothed_weights > 1.0e-6, smoothed_values / smoothed_weights, np.nan)


def main() -> None:
    pref_m = load_json(ROOT / "sgc_new_events_joint_vp_relocation_metrics.json")
    sens_m = load_json(SENS / "sgc_new_events_joint_vp_relocation_metrics.json")

    grid = np.load(OUT_MODEL / "sgc_new_events_grid_10km.npz")
    xg, yg, zg = grid["xg"], grid["yg"], grid["zg"]

    pref_init = np.load(OUT_MODEL / "sgc_new_events_vp_initial_from_q.npy")
    pref_final = np.load(OUT_MODEL / "sgc_new_events_vp_final_joint_relocation.npy")
    pref_active = np.load(OUT_MODEL / "sgc_new_events_vp_active_mask.npy").astype(bool)

    sens_init = np.load(SENS / "models" / "sgc_new_events_vp_initial_from_q.npy")
    sens_final = np.load(SENS / "models" / "sgc_new_events_vp_final_joint_relocation.npy")
    sens_active = np.load(SENS / "models" / "sgc_new_events_vp_active_mask.npy").astype(bool)

    common = pref_active & sens_active & np.isfinite(pref_final) & np.isfinite(sens_final)
    pref_update = 100.0 * (pref_final - pref_init) / np.maximum(pref_init, 1.0e-9)
    sens_update = 100.0 * (sens_final - sens_init) / np.maximum(sens_init, 1.0e-9)
    start_lateral = 100.0 * (pref_init - sens_init) / np.maximum(sens_init, 1.0e-9)
    final_diff = 100.0 * (sens_final - pref_final) / np.maximum(pref_final, 1.0e-9)

    pref_er = event_rms(ROOT / "data" / "sgc_new_events_final_vp_predictions.csv")
    sens_er = event_rms(SENS / "data" / "sgc_new_events_final_vp_predictions.csv")
    er = pref_er.merge(sens_er, on="event_id", suffixes=("_preferred", "_1d"))

    pref_reloc = pd.read_csv(ROOT / "data" / "sgc_new_events_relocations_round2.csv")
    sens_reloc = pd.read_csv(SENS / "data" / "sgc_new_events_relocations_round2.csv")
    reloc = pref_reloc.merge(sens_reloc, on="event_id", suffixes=("_preferred", "_1d"))
    dx = reloc["relocated_x_km_1d"] - reloc["relocated_x_km_preferred"]
    dy = reloc["relocated_y_km_1d"] - reloc["relocated_y_km_preferred"]
    dz = reloc["relocated_z_km_1d"] - reloc["relocated_z_km_preferred"]
    reloc["run_horizontal_difference_km"] = np.sqrt(dx * dx + dy * dy)
    reloc["run_vertical_difference_km"] = np.abs(dz)
    reloc["run_total_difference_km"] = np.sqrt(dx * dx + dy * dy + dz * dz)
    intermediate = reloc[
        reloc["relocated_z_km_preferred"].between(70.0, 170.0)
        | reloc["relocated_z_km_1d"].between(70.0, 170.0)
    ].copy()

    smooth_metrics: dict[str, dict[str, float | list[float]]] = {}
    for sigma in [1.0, 1.5, 2.0]:
        pref_update_s = smooth_nan(pref_update, pref_active, sigma)
        sens_update_s = smooth_nan(sens_update, sens_active, sigma)
        pref_final_s = smooth_nan(pref_final, pref_active, sigma)
        sens_final_s = smooth_nan(sens_final, sens_active, sigma)
        smoothed_final_diff = 100.0 * (sens_final_s - pref_final_s) / np.maximum(pref_final_s, 1.0e-9)
        smooth_metrics[f"sigma_{sigma:g}_cells"] = {
            "corr_final_abs_vp_common": corr(pref_final_s[common], sens_final_s[common]),
            "corr_recovered_update_common": corr(pref_update_s[common], sens_update_s[common]),
            "abs_final_difference_percent_p50_p90_common": [
                float(x) for x in np.nanpercentile(np.abs(smoothed_final_diff[common]), [50, 90])
            ],
        }

    metrics = {
        "method": "Current 2022-2026 dataset sensitivity: preferred slab-informed start versus depth-only 1-D start",
        "preferred": {
            "events": pref_m["selected_events"],
            "picks": pref_m["selected_picks"],
            "equations": pref_m["final_equations"],
            "active_cells": pref_m["active_cells"],
            "final_rms_s": pref_m["final_rms_s"],
            "median_event_final_rms_s": float(pref_er["event_final_rms_s"].median()),
            "checkerboard_corr": pref_m["checkerboard"]["checkerboard_corr_active"],
        },
        "one_d": {
            "events": sens_m["selected_events"],
            "picks": sens_m["selected_picks"],
            "equations": sens_m["final_equations"],
            "active_cells": sens_m["active_cells"],
            "final_rms_s": sens_m["final_rms_s"],
            "median_event_final_rms_s": float(sens_er["event_final_rms_s"].median()),
            "checkerboard_corr": sens_m["checkerboard"]["checkerboard_corr_active"],
        },
        "common_active_cells": int(common.sum()),
        "corr_final_abs_vp_common": corr(pref_final[common], sens_final[common]),
        "corr_recovered_update_common": corr(pref_update[common], sens_update[common]),
        "corr_starting_lateral_component_vs_preferred_update_common": corr(start_lateral[common], pref_update[common]),
        "final_difference_percent_p05_p50_p95_common": [
            float(x) for x in np.nanpercentile(final_diff[common], [5, 50, 95])
        ],
        "abs_final_difference_percent_p50_p90_common": [
            float(x) for x in np.nanpercentile(np.abs(final_diff[common]), [50, 90])
        ],
        "smoothed_comparison": smooth_metrics,
        "event_rms_difference_1d_minus_preferred_p50_p90_s": [
            float(x) for x in np.nanpercentile(
                er["event_final_rms_s_1d"] - er["event_final_rms_s_preferred"], [50, 90]
            )
        ],
        "relocation_difference_all_events": {
            "median_horizontal_km": float(reloc["run_horizontal_difference_km"].median()),
            "p90_horizontal_km": float(reloc["run_horizontal_difference_km"].quantile(0.90)),
            "median_vertical_km": float(reloc["run_vertical_difference_km"].median()),
            "p90_vertical_km": float(reloc["run_vertical_difference_km"].quantile(0.90)),
            "median_total_km": float(reloc["run_total_difference_km"].median()),
            "p90_total_km": float(reloc["run_total_difference_km"].quantile(0.90)),
        },
        "relocation_difference_intermediate_events_70_170km": {
            "events": int(len(intermediate)),
            "median_horizontal_km": float(intermediate["run_horizontal_difference_km"].median()),
            "p90_horizontal_km": float(intermediate["run_horizontal_difference_km"].quantile(0.90)),
            "median_vertical_km": float(intermediate["run_vertical_difference_km"].median()),
            "p90_vertical_km": float(intermediate["run_vertical_difference_km"].quantile(0.90)),
            "median_total_km": float(intermediate["run_total_difference_km"].median()),
            "p90_total_km": float(intermediate["run_total_difference_km"].quantile(0.90)),
        },
    }
    (OUT_DATA / "current_dataset_1d_sensitivity_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    reloc.to_csv(OUT_DATA / "current_dataset_1d_sensitivity_relocation_differences.csv", index=False)
    er.to_csv(OUT_DATA / "current_dataset_1d_sensitivity_event_rms.csv", index=False)

    depths = [80.0, 140.0]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), constrained_layout=True)
    vmax_update = 35.0
    vmax_diff = 12.0
    for row, depth in enumerate(depths):
        iz = int(np.argmin(np.abs(zg - depth)))
        panels = [
            (pref_update[iz], pref_active[iz], "Preferred update", vmax_update),
            (sens_update[iz], sens_active[iz], "1-D-start update", vmax_update),
            (final_diff[iz], common[iz], "Final Vp difference\n1-D - preferred", vmax_diff),
        ]
        for col, (arr, mask, title, vmax) in enumerate(panels):
            plot = np.where(mask, arr, np.nan)
            im = axes[row, col].imshow(
                plot,
                origin="lower",
                extent=[xg.min(), xg.max(), yg.min(), yg.max()],
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                interpolation="nearest",
            )
            axes[row, col].set_title(f"{title}, z={zg[iz]:.0f} km", fontsize=10)
            axes[row, col].set_xlabel("x (km)")
            axes[row, col].set_ylabel("y (km)")
            cbar = fig.colorbar(im, ax=axes[row, col], shrink=0.82)
            cbar.set_label("%")
    fig.suptitle("Current-dataset 1-D starting-model sensitivity", fontsize=12)
    fig_path = OUT_FIG / "current_dataset_1d_prior_sensitivity.png"
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)

    print(json.dumps(metrics, indent=2))
    print(fig_path)


if __name__ == "__main__":
    main()
