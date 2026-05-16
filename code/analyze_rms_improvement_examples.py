#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare SGC-reported RMS with final event RMS after relocation/tomography."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve()
if (HERE.parents[1] / "data").exists() and (HERE.parents[1] / "figures").exists():
    RUN = HERE.parents[1]
    ROOT = RUN.parent
else:
    ROOT = HERE.parents[1]
    RUN = ROOT / "output_sgc_new_events_joint_vp_relocation_fmm_min4_plus_shallow"
DATA = RUN / "data"
FIG = RUN / "figures"
CODE = RUN / "codigos_finales"


def rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(values * values)))


def load_event_comparison() -> pd.DataFrame:
    selected = pd.read_csv(DATA / "sgc_new_events_selected_event_summary_min4_non2700_nest300.csv")
    selected = selected.drop_duplicates("event_id").copy()
    selected = selected.rename(
        columns={
            "event_depth_km": "sgc_depth_km",
            "event_lat": "sgc_lat",
            "event_lon": "sgc_lon",
            "rms_s": "sgc_rms_s",
        }
    )
    keep = [
        "event_id",
        "selection_group",
        "sgc_lat",
        "sgc_lon",
        "sgc_depth_km",
        "sgc_rms_s",
        "magnitude",
        "phases",
        "n_picks",
    ]
    selected = selected[[c for c in keep if c in selected.columns]].copy()

    pred = pd.read_csv(DATA / "sgc_new_events_final_vp_predictions.csv")
    final = []
    for event_id, group in pred.groupby("event_id", sort=False):
        residual = group["residual_final_s"].to_numpy(dtype=float)
        final.append(
            {
                "event_id": event_id,
                "final_event_rms_s": rms(residual),
                "final_abs_residual_p90_s": float(np.nanpercentile(np.abs(residual), 90)),
                "final_n_equations": int(len(group)),
                "relocated_lat": float(group["event_lat"].iloc[0]),
                "relocated_lon": float(group["event_lon"].iloc[0]),
                "relocated_depth_km": float(group["event_depth_km"].iloc[0]),
            }
        )
    final = pd.DataFrame(final)

    reloc = pd.read_csv(DATA / "sgc_new_events_relocations_round2.csv")
    reloc = reloc[
        [
            "event_id",
            "rms_before_s",
            "rms_after_s",
            "horizontal_shift_km",
            "vertical_shift_km",
            "total_shift_km",
            "origin_time_shift_s",
        ]
    ].copy()

    out = selected.merge(final, on="event_id", how="inner").merge(reloc, on="event_id", how="left")
    out["rms_improvement_s"] = out["sgc_rms_s"] - out["final_event_rms_s"]
    out["rms_improvement_percent"] = 100.0 * out["rms_improvement_s"] / out["sgc_rms_s"].replace(0, np.nan)
    out["depth_change_km"] = out["relocated_depth_km"] - out["sgc_depth_km"]
    out = out.sort_values("rms_improvement_s", ascending=False).reset_index(drop=True)
    out.to_csv(DATA / "rms_improvement_sgc_vs_final_by_event.csv", index=False)
    out.head(20).to_csv(DATA / "rms_improvement_top20_examples.csv", index=False)
    return out


def make_scatter_figure(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), dpi=250)
    y_cap = 1.15
    imp_min, imp_max = -0.80, 0.95
    final_clipped = np.minimum(df["final_event_rms_s"].to_numpy(dtype=float), y_cap)
    final_out = df["final_event_rms_s"].to_numpy(dtype=float) > y_cap
    axes[0].scatter(df["sgc_depth_km"], df["sgc_rms_s"], s=9, alpha=0.18, c="#1f77b4", label="SGC RMS", rasterized=True)
    axes[0].scatter(df.loc[~final_out, "sgc_depth_km"], final_clipped[~final_out], s=9, alpha=0.20, c="#2ca02c", label="Final RMS", rasterized=True)
    axes[0].scatter(df.loc[final_out, "sgc_depth_km"], final_clipped[final_out], s=30, alpha=0.70, c="#2ca02c", marker="^", label=f"Final RMS > {y_cap:.2f} s", rasterized=True)
    axes[0].set_xlabel("SGC-reported depth (km)")
    axes[0].set_ylabel("RMS (s)")
    axes[0].set_title("Catalog RMS and final event RMS vs SGC depth")
    axes[0].set_ylim(-0.05, y_cap + 0.08)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    improvement = df["rms_improvement_s"].to_numpy(dtype=float)
    improvement_clipped = np.clip(improvement, imp_min, imp_max)
    below = improvement < imp_min
    above = improvement > imp_max
    sc = axes[1].scatter(
        df["sgc_depth_km"],
        improvement_clipped,
        c=df["total_shift_km"],
        cmap="viridis",
        s=np.clip(df["n_picks"].to_numpy(dtype=float), 4, 14) * 4,
        alpha=0.42,
        edgecolor="none",
        rasterized=True,
    )
    if below.any():
        axes[1].scatter(df.loc[below, "sgc_depth_km"], np.full(int(below.sum()), imp_min), marker="v", s=32, c="#444444", alpha=0.7, label=f"< {imp_min:.1f} s")
    if above.any():
        axes[1].scatter(df.loc[above, "sgc_depth_km"], np.full(int(above.sum()), imp_max), marker="^", s=32, c="#444444", alpha=0.7, label=f"> {imp_max:.1f} s")
    axes[1].axhline(0.0, color="0.25", lw=0.9)
    axes[1].set_xlabel("SGC-reported depth (km)")
    axes[1].set_ylabel("SGC RMS - final RMS (s)")
    axes[1].set_title("RMS improvement after relocation + Vp update")
    axes[1].set_ylim(imp_min - 0.05, imp_max + 0.05)
    axes[1].grid(alpha=0.25)
    if below.any() or above.any():
        axes[1].legend(fontsize=8, loc="lower left")
    cbar = fig.colorbar(sc, ax=axes[1], shrink=0.82)
    cbar.set_label("Total relocation shift (km)")
    fig.tight_layout()
    path = FIG / "rms_sgc_vs_final_by_depth.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def make_top_examples_figure(df: pd.DataFrame, n: int = 12) -> Path:
    top = df.head(n).iloc[::-1].copy()
    labels = top["event_id"].astype(str).str.replace("SGC", "", regex=False)
    y = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=250)
    ax.hlines(y, top["final_event_rms_s"], top["sgc_rms_s"], color="0.7", lw=2)
    ax.scatter(top["sgc_rms_s"], y, color="#d62728", s=45, label="SGC RMS")
    ax.scatter(top["final_event_rms_s"], y, color="#2ca02c", s=45, label="Final RMS")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("RMS (s)")
    ax.set_title("Largest event-level RMS improvements")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=8)
    path = FIG / "rms_top_improvement_examples.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def write_latex_table(df: pd.DataFrame, n: int = 8) -> Path:
    top = df.head(n).copy()
    path = DATA / "rms_improvement_top_examples_table.tex"
    lines = [
        r"\begin{table}[p]",
        r"\centering",
        r"\caption{Examples of earthquakes with the largest event-level RMS improvement after relocation and velocity updating. Depth is the SGC-reported catalog depth.}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Event & Depth (km) & Picks & SGC RMS (s) & Final RMS (s) & Improvement (s) & Shift (km) \\",
        r"\midrule",
    ]
    for row in top.itertuples(index=False):
        event = str(row.event_id).replace("_", r"\_")
        lines.append(
            f"{event} & {row.sgc_depth_km:.1f} & {int(row.n_picks)} & {row.sgc_rms_s:.2f} & "
            f"{row.final_event_rms_s:.2f} & {row.rms_improvement_s:.2f} & {row.total_shift_km:.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    CODE.mkdir(parents=True, exist_ok=True)
    df = load_event_comparison()
    fig_scatter = make_scatter_figure(df)
    fig_top = make_top_examples_figure(df)
    table_tex = write_latex_table(df)
    improved = df[df["rms_improvement_s"] > 0].copy()
    metrics = {
        "events_compared": int(len(df)),
        "events_improved": int(len(improved)),
        "fraction_improved": float(len(improved) / max(len(df), 1)),
        "median_sgc_rms_s": float(df["sgc_rms_s"].median()),
        "median_final_event_rms_s": float(df["final_event_rms_s"].median()),
        "median_rms_improvement_s": float(df["rms_improvement_s"].median()),
        "median_rms_improvement_percent": float(df["rms_improvement_percent"].median()),
        "p90_rms_improvement_s": float(df["rms_improvement_s"].quantile(0.90)),
        "max_rms_improvement_s": float(df["rms_improvement_s"].max()),
        "max_improvement_event_id": str(df.iloc[0]["event_id"]),
        "median_sgc_depth_km": float(df["sgc_depth_km"].median()),
        "figures": {
            "rms_depth": str(fig_scatter),
            "top_examples": str(fig_top),
        },
        "tables": {
            "all_events": str(DATA / "rms_improvement_sgc_vs_final_by_event.csv"),
            "top20_csv": str(DATA / "rms_improvement_top20_examples.csv"),
            "latex_table": str(table_tex),
        },
    }
    metrics_path = RUN / "rms_improvement_sgc_vs_final_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    src = Path(__file__).resolve()
    dst = (CODE / Path(__file__).name).resolve()
    if src != dst:
        shutil.copy2(src, dst)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(df.head(10)[["event_id", "sgc_depth_km", "n_picks", "sgc_rms_s", "final_event_rms_s", "rms_improvement_s", "total_shift_km"]].to_string(index=False))


if __name__ == "__main__":
    main()
