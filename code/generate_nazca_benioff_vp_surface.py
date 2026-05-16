#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reconstruct a local Nazca/Benioff surface and sample final Vp anomalies.

The geometry is controlled by relocated intermediate-depth earthquakes. The
tomographic model is used only as an attribute sampled on the reconstructed
surface, so the figure separates "where the Benioff zone is" from "what the
velocity model says along that geometry".
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.interpolate import RBFInterpolator, RegularGridInterpolator
from scipy.spatial import ConvexHull, cKDTree


ROOT = Path(__file__).resolve().parents[1]
RUN = Path(
    r"C:\Users\claudia.parra\Documents\Materias_romulo\investigacion"
    r"\tomografia_localizacion_simultaneas"
    r"\output_sgc_new_events_joint_vp_relocation_fmm_min4_plus_shallow"
)
DATA = RUN / "data"
FIG = RUN / "figures"
MODEL = RUN / "models"
CODE = RUN / "codigos_finales"

LAT0 = 7.0
LON0 = -73.0
KM_PER_DEG = 111.32
COSLAT0 = math.cos(math.radians(LAT0))

MIN_CONTROL_DEPTH_KM = 70.0
MAX_CONTROL_DEPTH_KM = 170.0
MIN_HIT_COUNT = 20
RBF_SMOOTHING = 35.0
SURFACE_NX = 100
SURFACE_NY = 95


def lonlat_to_xy(lon, lat):
    x = (np.asarray(lon, dtype=float) - LON0) * KM_PER_DEG * COSLAT0
    y = (np.asarray(lat, dtype=float) - LAT0) * KM_PER_DEG
    return x, y


def xy_to_lonlat(x, y):
    lon = np.asarray(x, dtype=float) / (KM_PER_DEG * COSLAT0) + LON0
    lat = np.asarray(y, dtype=float) / KM_PER_DEG + LAT0
    return lon, lat


def as_xyz(arr_zyx: np.ndarray) -> np.ndarray:
    return np.transpose(arr_zyx, (2, 1, 0))


def make_interpolator(xg, yg, zg, arr_zyx, nearest=False):
    return RegularGridInterpolator(
        (xg, yg, zg),
        as_xyz(arr_zyx),
        method="nearest" if nearest else "linear",
        bounds_error=False,
        fill_value=np.nan,
    )


def nearest_grid_indices(x, y, z, xg, yg, zg):
    ix = np.clip(np.rint((x - xg[0]) / (xg[1] - xg[0])).astype(int), 0, len(xg) - 1)
    iy = np.clip(np.rint((y - yg[0]) / (yg[1] - yg[0])).astype(int), 0, len(yg) - 1)
    iz = np.clip(np.rint((z - zg[0]) / (zg[1] - zg[0])).astype(int), 0, len(zg) - 1)
    return iz, iy, ix


def load_model():
    grid = np.load(MODEL / "sgc_new_events_grid_10km.npz")
    xg, yg, zg = grid["xg"], grid["yg"], grid["zg"]
    vp0 = np.load(MODEL / "sgc_new_events_vp_initial_from_q.npy")
    vp = np.load(MODEL / "sgc_new_events_vp_final_joint_relocation.npy")
    active = np.load(MODEL / "sgc_new_events_vp_active_mask.npy").astype(bool)
    hits = np.load(MODEL / "sgc_new_events_vp_hit_count.npy")
    dvp = 100.0 * (vp - vp0) / np.maximum(vp0, 1e-6)
    return xg, yg, zg, vp0, vp, dvp, active, hits


def sample_relocated_events(model_tuple):
    xg, yg, zg, _vp0, vp, dvp, active, hits = model_tuple
    events = pd.read_csv(DATA / "sgc_new_events_relocations_round2.csv")
    events = events.rename(
        columns={
            "relocated_lon": "lon",
            "relocated_lat": "lat",
            "relocated_z_km": "depth_km",
            "relocated_x_km": "x_km",
            "relocated_y_km": "y_km",
        }
    )
    iz, iy, ix = nearest_grid_indices(
        events["x_km"].to_numpy(),
        events["y_km"].to_numpy(),
        events["depth_km"].to_numpy(),
        xg,
        yg,
        zg,
    )
    events["vp_final_km_s"] = vp[iz, iy, ix]
    events["dvp_percent"] = dvp[iz, iy, ix]
    events["vp_active"] = active[iz, iy, ix]
    events["hit_count"] = hits[iz, iy, ix]
    sampled_path = DATA / "nazca_benioff_relocated_events_sampled_vp.csv"
    events.to_csv(sampled_path, index=False)
    return events, sampled_path


def select_control_events(events: pd.DataFrame) -> pd.DataFrame:
    control = events[
        events["depth_km"].between(MIN_CONTROL_DEPTH_KM, MAX_CONTROL_DEPTH_KM)
        & events["vp_active"].astype(bool)
        & (events["hit_count"] >= MIN_HIT_COUNT)
        & np.isfinite(events["dvp_percent"])
    ].copy()
    # Avoid over-weighting repeated node locations from grid relocation.
    control["node_key"] = (
        control["x_km"].round(3).astype(str)
        + "_"
        + control["y_km"].round(3).astype(str)
        + "_"
        + control["depth_km"].round(3).astype(str)
    )
    control = control.sort_values(["n_picks", "hit_count"], ascending=[False, False])
    control = control.drop_duplicates("node_key").drop(columns=["node_key"]).reset_index(drop=True)
    control_path = DATA / "nazca_benioff_vp_control_events.csv"
    control.to_csv(control_path, index=False)
    if len(control) < 20:
        raise RuntimeError(f"Too few Benioff control events after filtering: {len(control)}")
    return control


def fit_benioff_surface(control: pd.DataFrame):
    points = control[["x_km", "y_km"]].to_numpy(dtype=float)
    depth = control["depth_km"].to_numpy(dtype=float)
    rbf = RBFInterpolator(points, depth, kernel="thin_plate_spline", smoothing=RBF_SMOOTHING)
    hull = ConvexHull(points)
    hull_path = MplPath(points[hull.vertices])
    xi = np.linspace(points[:, 0].min() - 8.0, points[:, 0].max() + 8.0, SURFACE_NX)
    yi = np.linspace(points[:, 1].min() - 8.0, points[:, 1].max() + 8.0, SURFACE_NY)
    X, Y = np.meshgrid(xi, yi)
    flat = np.column_stack([X.ravel(), Y.ravel()])
    inside = hull_path.contains_points(flat).reshape(X.shape)
    Z = rbf(flat).reshape(X.shape)
    Z = np.where(inside & (Z >= MIN_CONTROL_DEPTH_KM) & (Z <= MAX_CONTROL_DEPTH_KM), Z, np.nan)
    tree = cKDTree(points)
    nearest = tree.query(flat, k=1)[0].reshape(X.shape)

    predicted = rbf(points)
    residual = predicted - depth
    return X, Y, Z, nearest, residual


def sample_surface(X, Y, Z, model_tuple):
    xg, yg, zg, _vp0, vp, dvp, active, hits = model_tuple
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    valid = np.isfinite(pts).all(axis=1)
    interpolators = {
        "vp_final_km_s": make_interpolator(xg, yg, zg, vp),
        "dvp_percent": make_interpolator(xg, yg, zg, dvp),
        "vp_active": make_interpolator(xg, yg, zg, active.astype(float), nearest=True),
        "hit_count": make_interpolator(xg, yg, zg, hits.astype(float), nearest=True),
    }
    out = {}
    for key, interpolator in interpolators.items():
        values = np.full(X.size, np.nan)
        values[valid] = interpolator(pts[valid])
        out[key] = values.reshape(X.shape)
    return out


def save_surface(X, Y, Z, nearest, attrs):
    lon, lat = xy_to_lonlat(X.ravel(), Y.ravel())
    confidence = np.full(X.size, "low", dtype=object)
    finite = np.isfinite(Z.ravel())
    confidence[finite & (nearest.ravel() <= 35.0)] = "moderate"
    confidence[finite & (nearest.ravel() <= 20.0)] = "high"
    confidence[~finite] = "outside"
    surface = pd.DataFrame(
        {
            "lon": lon,
            "lat": lat,
            "x_km": X.ravel(),
            "y_km": Y.ravel(),
            "slab_depth_km": Z.ravel(),
            "nearest_control_event_km": nearest.ravel(),
            "confidence": confidence,
            "vp_final_km_s": attrs["vp_final_km_s"].ravel(),
            "dvp_percent_on_surface": attrs["dvp_percent"].ravel(),
            "vp_active_surface": attrs["vp_active"].ravel() >= 0.5,
            "hit_count_surface": attrs["hit_count"].ravel(),
        }
    )
    surface = surface[np.isfinite(surface["slab_depth_km"])].copy()
    surface["tomography_supported_slab"] = (
        surface["confidence"].isin(["high", "moderate"])
        & surface["vp_active_surface"]
        & (surface["hit_count_surface"] >= MIN_HIT_COUNT)
        & (surface["dvp_percent_on_surface"] >= 0.0)
    )
    surface_path = DATA / "nazca_benioff_vp_surface.csv"
    surface.to_csv(surface_path, index=False)
    return surface, surface_path


def station_table():
    picks = pd.read_csv(DATA / "sgc_new_events_final_relocated_picks.csv", usecols=["station", "station_lat", "station_lon"])
    stations = picks.drop_duplicates("station").rename(columns={"station_lat": "lat", "station_lon": "lon"})
    return stations.sort_values("station").reset_index(drop=True)


def plot_3d(control, surface, stations):
    grid = surface.pivot_table(index="y_km", columns="x_km", values="slab_depth_km")
    dvp_grid = surface.pivot_table(index="y_km", columns="x_km", values="dvp_percent_on_surface")
    X, Y = np.meshgrid(grid.columns.to_numpy(dtype=float), grid.index.to_numpy(dtype=float))
    Z = grid.to_numpy(dtype=float)
    DVP = dvp_grid.to_numpy(dtype=float)
    lon_grid, lat_grid = xy_to_lonlat(X, Y)

    finite_dvp = DVP[np.isfinite(DVP)]
    vmax = float(np.nanpercentile(np.abs(finite_dvp), 95)) if finite_dvp.size else 10.0
    vmax = max(5.0, min(vmax, 35.0))
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)
    face = plt.cm.RdBu_r(norm(DVP))
    face[..., -1] = np.where(np.isfinite(Z), 0.72, 0.0)

    fig = plt.figure(figsize=(12, 9), dpi=260)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(lon_grid, lat_grid, Z, facecolors=face, linewidth=0, antialiased=True, shade=False)
    ax.scatter(
        control["lon"],
        control["lat"],
        control["depth_km"],
        c="black",
        s=15,
        alpha=0.72,
        depthshade=False,
        label="Relocated Benioff events",
    )
    pos = control[control["dvp_percent"] >= 0.0]
    ax.scatter(
        pos["lon"],
        pos["lat"],
        pos["depth_km"],
        facecolors="none",
        edgecolors="#ffd166",
        s=46,
        linewidth=0.7,
        depthshade=False,
        label="Events in positive dVp cells",
    )
    ax.scatter(
        stations["lon"],
        stations["lat"],
        np.full(len(stations), MIN_CONTROL_DEPTH_KM),
        marker="^",
        c="red",
        s=62,
        edgecolor="white",
        linewidth=0.4,
        depthshade=False,
        label="Stations projected at 70 km",
    )
    ax.set_title("Nazca/Benioff surface constrained by relocated seismicity and final Vp anomalies", pad=16)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Depth (km)")
    ax.set_zlim(MAX_CONTROL_DEPTH_KM + 5, MIN_CONTROL_DEPTH_KM - 5)
    ax.view_init(elev=25, azim=-132)
    ax.set_box_aspect((1.2, 1.25, 0.72))
    ax.legend(loc="upper left", fontsize=8)
    sm = plt.cm.ScalarMappable(norm=norm, cmap="RdBu_r")
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.62, pad=0.08)
    cbar.set_label("dVp on reconstructed surface (%)")
    path = FIG / "nazca_benioff_vp_surface_3d.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_map(control, surface, stations):
    grid = surface.pivot_table(index="y_km", columns="x_km", values="slab_depth_km")
    dvp_grid = surface.pivot_table(index="y_km", columns="x_km", values="dvp_percent_on_surface")
    X, Y = np.meshgrid(grid.columns.to_numpy(dtype=float), grid.index.to_numpy(dtype=float))
    Z = grid.to_numpy(dtype=float)
    DVP = dvp_grid.to_numpy(dtype=float)
    lon_grid, lat_grid = xy_to_lonlat(X, Y)

    fig, ax = plt.subplots(figsize=(9, 8), dpi=250)
    depth_levels = np.arange(70, 171, 10)
    im = ax.contourf(lon_grid, lat_grid, Z, levels=depth_levels, cmap="viridis_r", alpha=0.65)
    cs = ax.contour(lon_grid, lat_grid, Z, levels=np.arange(80, 171, 20), colors="k", linewidths=0.7)
    ax.clabel(cs, fmt="%d km", fontsize=7)
    sc = ax.scatter(control["lon"], control["lat"], c=control["dvp_percent"], cmap="RdBu_r", vmin=-25, vmax=25, s=20, edgecolor="white", linewidth=0.2)
    supported = surface[surface["tomography_supported_slab"]]
    ax.scatter(supported["lon"], supported["lat"], s=5, c="#f4a261", alpha=0.45, label="Positive-dVp supported surface")
    ax.scatter(stations["lon"], stations["lat"], marker="^", s=65, c="red", edgecolor="white", linewidth=0.4, label="Stations")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Nazca/Benioff reconstruction: depth contours and Vp anomaly support")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=8)
    fig.colorbar(im, ax=ax, label="Reconstructed slab depth (km)", shrink=0.78)
    fig.colorbar(sc, ax=ax, label="dVp at control events (%)", shrink=0.78, pad=0.02)
    path = FIG / "nazca_benioff_vp_surface_map.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_depth_dvp(control, surface):
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=250)
    ax.scatter(control["dvp_percent"], control["depth_km"], c="black", s=16, alpha=0.5, label="Control events")
    supported = surface[surface["tomography_supported_slab"]]
    ax.scatter(supported["dvp_percent_on_surface"], supported["slab_depth_km"], c="#f4a261", s=8, alpha=0.35, label="Supported surface nodes")
    ax.axvline(0.0, color="0.35", lw=0.9)
    ax.invert_yaxis()
    ax.set_xlabel("dVp (%)")
    ax.set_ylabel("Depth (km)")
    ax.set_title("Velocity anomaly sampled on the Benioff-controlled slab")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    path = FIG / "nazca_benioff_vp_depth_anomaly.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    CODE.mkdir(parents=True, exist_ok=True)

    model_tuple = load_model()
    events, sampled_path = sample_relocated_events(model_tuple)
    control = select_control_events(events)
    X, Y, Z, nearest, residual = fit_benioff_surface(control)
    attrs = sample_surface(X, Y, Z, model_tuple)
    surface, surface_path = save_surface(X, Y, Z, nearest, attrs)
    stations = station_table()

    fig3d = plot_3d(control, surface, stations)
    figmap = plot_map(control, surface, stations)
    figdepth = plot_depth_dvp(control, surface)

    supported = surface[surface["tomography_supported_slab"]]
    metrics = {
        "control_events": int(len(control)),
        "sampled_relocated_events": int(len(events)),
        "control_depth_range_km": [float(control["depth_km"].min()), float(control["depth_km"].max())],
        "surface_nodes": int(len(surface)),
        "surface_depth_range_km": [float(surface["slab_depth_km"].min()), float(surface["slab_depth_km"].max())],
        "high_confidence_nodes": int((surface["confidence"] == "high").sum()),
        "moderate_or_high_confidence_nodes": int(surface["confidence"].isin(["high", "moderate"]).sum()),
        "tomography_supported_nodes": int(len(supported)),
        "tomography_supported_fraction": float(len(supported) / max(len(surface), 1)),
        "fit_rmse_km": float(np.sqrt(np.mean(residual**2))),
        "fit_mae_km": float(np.mean(np.abs(residual))),
        "median_control_dvp_percent": float(np.nanmedian(control["dvp_percent"])),
        "median_surface_dvp_percent": float(np.nanmedian(surface["dvp_percent_on_surface"])),
        "median_supported_surface_dvp_percent": float(np.nanmedian(supported["dvp_percent_on_surface"])) if len(supported) else float("nan"),
        "sampled_events_csv": str(sampled_path),
        "control_events_csv": str(DATA / "nazca_benioff_vp_control_events.csv"),
        "surface_csv": str(surface_path),
        "figures": {
            "3d_surface": str(fig3d),
            "map": str(figmap),
            "depth_anomaly": str(figdepth),
        },
    }
    metrics_path = RUN / "nazca_benioff_vp_surface_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    shutil.copy2(Path(__file__), CODE / Path(__file__).name)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
