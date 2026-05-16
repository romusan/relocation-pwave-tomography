#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a new SGC event set and run coupled Vp tomography + relocation.

This script uses the final 10 km Vp model produced by the Q-tomography work
as the starting model. It then selects events that were not used in that work,
downloads the SGC XML metadata, extracts first P picks for the working station
set, relocates events by a local grid-node search, and alternates relocation
with sparse 3-D slowness updates.

The inversion is intentionally phrased as an iterative coupled workflow rather
than a full dense Thurber system, because the requested 2000-event dataset is
too large for the dense SVD prototype contained in this folder.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import lsmr


ROOT = Path(__file__).resolve().parents[1]
INVESTIGATION = ROOT.parent
Q_ROOT = INVESTIGATION / "tomografia_Q"

Q_FSM_ROOT = Q_ROOT / "output_V2" / "sgc_final_q_extra_station_fsm_tomography"
Q_FSM_SCRIPT = Q_FSM_ROOT / "codigos_finales" / "run_sgc_final_fsm_vp_q_tomography.py"
Q_XML_SCRIPT = Q_ROOT / "download_sgc_waveforms_for_q_spectral.py"

CATALOG_CSV = Q_ROOT / "output_V2" / "data" / "sgc_combined_area_2022_2026_depth20_170_rms_lt1_for_final.csv"
STATIONS_CSV = Q_ROOT / "output_V2" / "data" / "domain_stations_v2_plus_extra_used_for_q.csv"
USED_PICKS_CSV = Q_ROOT / "output_V2" / "sgc_final_q_extra_station_tomography" / "data" / "sgc_1000_vp_picks.csv"
USED_INVENTORY_CSV = Q_ROOT / "output_V2" / "data" / "sgc_waveforms_q_spectral_final_with_nest_and_extra_stations_inventory.csv"

VP_INITIAL_NPY = Q_FSM_ROOT / "models" / "sgc_final_fsm_vp_final.npy"
GRID_NPZ = Q_FSM_ROOT / "models" / "sgc_final_fsm_grid.npz"

OUT = Path(os.environ.get("OUT_DIR", str(ROOT / "output_sgc_new_events_joint_vp_relocation")))
DATA = OUT / "data"
XML_DIR = Path(os.environ.get("XML_CACHE_DIR", str(DATA / "xml_events")))
FIG = OUT / "figures"
MODEL = OUT / "models"
PAPER = OUT / "paper"
CODE = OUT / "codigos_finales"

LAT0 = 7.0
LON0 = -73.0
LON_MIN, LON_MAX = -74.60, -72.55
LAT_MIN, LAT_MAX = 6.20, 8.05
MAX_MODEL_DEPTH_KM = 180.0

CANDIDATE_NON_NEST = int(os.environ.get("CANDIDATE_NON_NEST", "2400"))
CANDIDATE_NEST = int(os.environ.get("CANDIDATE_NEST", "600"))
TARGET_NON_NEST = int(os.environ.get("TARGET_NON_NEST", "1700"))
TARGET_NEST = int(os.environ.get("TARGET_NEST", "300"))
MIN_PICKS_PER_EVENT = int(os.environ.get("MIN_PICKS_PER_EVENT", "3"))
MAX_TRAVEL_TIME_S = 260.0

RELOCATION_RADIUS_NODES = 1
COUPLED_ROUNDS = 2
TRAVEL_SOLVER = os.environ.get("TRAVEL_SOLVER", "fmm").strip().lower()
USE_STATION_STATICS = os.environ.get("USE_STATION_STATICS", "1").strip().lower() not in {"0", "false", "no"}
VP_ALPHA = 0.20
VP_DAMP = 0.25
VP_LAPLACE = 3.0
VP_ITERLIM = 1000
VP_MIN_HITS = 20
STATION_STATIC_DAMP = 0.75
S_MIN = 1.0 / 15.0
S_MAX = 1.0 / 2.0

CHECKERBOARD_SLOWNESS_FRACTION = 0.04
CHECKERBOARD_NOISE_S = 0.03


def ensure_dirs() -> None:
    for path in [OUT, DATA, XML_DIR, FIG, MODEL, PAPER, CODE]:
        path.mkdir(parents=True, exist_ok=True)


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def latlon_to_xy(lat: float, lon: float) -> tuple[float, float]:
    x = (float(lon) - LON0) * 111.32 * math.cos(math.radians(LAT0))
    y = (float(lat) - LAT0) * 111.32
    return float(x), float(y)


def xy_to_latlon(x: float, y: float) -> tuple[float, float]:
    lat = float(y) / 111.32 + LAT0
    lon = float(x) / (111.32 * math.cos(math.radians(LAT0))) + LON0
    return lat, lon


def domain_mask(xg: np.ndarray, yg: np.ndarray, zg: np.ndarray) -> np.ndarray:
    x_min, y_min = latlon_to_xy(LAT_MIN, LON_MIN)
    x_max, y_max = latlon_to_xy(LAT_MAX, LON_MAX)
    xmin, xmax = sorted([x_min, x_max])
    ymin, ymax = sorted([y_min, y_max])
    zz, yy, xx = np.meshgrid(zg, yg, xg, indexing="ij")
    return (
        (xx >= xmin)
        & (xx <= xmax)
        & (yy >= ymin)
        & (yy <= ymax)
        & (zz <= MAX_MODEL_DEPTH_KM)
    )


def rms(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.sqrt(np.mean(arr * arr))) if arr.size else float("nan")


def load_q_grid_and_vp() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grid = np.load(GRID_NPZ)
    vp = np.load(VP_INITIAL_NPY).astype(float)
    return grid["xg"].astype(float), grid["yg"].astype(float), grid["zg"].astype(float), vp


def load_used_event_ids() -> set[str]:
    used: set[str] = set()
    for path in [USED_PICKS_CSV, USED_INVENTORY_CSV]:
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=["event_id"])
        used.update(df["event_id"].astype(str).str.strip().tolist())
    return used


def quality_sort(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["phases", "rms_s", "gap_deg", "magnitude", "error_depth_km"]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["quality_score"] = (
        out["phases"].fillna(0) * 2.0
        - out["rms_s"].fillna(1.0) * 20.0
        - out["gap_deg"].fillna(180.0) / 10.0
        + out["magnitude"].fillna(0.0) * 3.0
        - out["error_depth_km"].fillna(20.0)
    )
    return out.sort_values(["quality_score", "phases", "magnitude"], ascending=[False, False, False])


def select_new_catalog() -> pd.DataFrame:
    selected_path = DATA / f"sgc_new_events_candidate_catalog_non{CANDIDATE_NON_NEST}_nest{CANDIDATE_NEST}.csv"
    if selected_path.exists():
        return pd.read_csv(selected_path)

    catalog = pd.read_csv(CATALOG_CSV)
    catalog["event_id"] = catalog["event_id"].astype(str).str.strip()
    used = load_used_event_ids()
    catalog = catalog[~catalog["event_id"].isin(used)].copy()
    catalog = catalog.drop_duplicates("event_id")
    catalog["depth_km"] = pd.to_numeric(catalog["depth_km"], errors="coerce")
    catalog["lat"] = pd.to_numeric(catalog["lat"], errors="coerce")
    catalog["lon"] = pd.to_numeric(catalog["lon"], errors="coerce")
    catalog = catalog[
        catalog["lat"].between(LAT_MIN, LAT_MAX)
        & catalog["lon"].between(LON_MIN, LON_MAX)
        & catalog["depth_km"].between(20.0, 170.0)
        & catalog["quakeml_url"].notna()
    ].copy()

    non_nest = quality_sort(catalog[catalog["depth_km"].between(20.0, 120.0)]).head(CANDIDATE_NON_NEST)
    nest = quality_sort(catalog[catalog["depth_km"].between(120.0, 170.0)]).head(CANDIDATE_NEST)
    selected = pd.concat([non_nest, nest], ignore_index=True)
    selected["selection_group"] = np.where(selected["depth_km"].between(120.0, 170.0), "bucaramanga_nest", "domain_20_120")
    selected.to_csv(selected_path, index=False)
    (DATA / "sgc_used_event_ids_from_q.txt").write_text("\n".join(sorted(used)), encoding="utf-8")
    return selected


def _download_one_xml(row: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
    event_id = str(row["event_id"])
    out = XML_DIR / f"{event_id}.xml"
    if out.exists() and out.stat().st_size > 500:
        return {"event_id": event_id, "xml_file": str(out), "http_status": 200, "cached": True, "error": ""}
    url = str(row["quakeml_url"])
    try:
        response = requests.get(url, timeout=timeout)
        status = int(response.status_code)
        if status == 200 and len(response.content) > 500:
            out.write_bytes(response.content)
            return {"event_id": event_id, "xml_file": str(out), "http_status": status, "cached": False, "error": ""}
        return {"event_id": event_id, "xml_file": str(out), "http_status": status, "cached": False, "error": f"bytes={len(response.content)}"}
    except Exception as exc:
        return {"event_id": event_id, "xml_file": str(out), "http_status": -1, "cached": False, "error": repr(exc)}


def download_xmls(catalog: pd.DataFrame, max_workers: int = 10) -> pd.DataFrame:
    log_path = DATA / f"sgc_new_events_xml_download_log_non{CANDIDATE_NON_NEST}_nest{CANDIDATE_NEST}.csv"
    if log_path.exists():
        log = pd.read_csv(log_path)
        expected = set(catalog["event_id"].astype(str))
        ok = set(log.loc[log["http_status"].eq(200), "event_id"].astype(str))
        if expected.issubset(ok):
            return log[log["event_id"].astype(str).isin(expected)].copy()

    records = catalog.to_dict("records")
    results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_download_one_xml, row) for row in records]
        for i, fut in enumerate(as_completed(futures), start=1):
            results.append(fut.result())
            if i % 100 == 0:
                print(f"[XML] {i}/{len(records)} completed in {time.time() - t0:.1f}s", flush=True)
    log = pd.DataFrame(results)
    log.to_csv(log_path, index=False)
    return log


def station_table() -> pd.DataFrame:
    stations = pd.read_csv(STATIONS_CSV)
    stations["station"] = stations["station"].astype(str)
    stations = stations.drop_duplicates("station").copy()
    xy = stations.apply(lambda r: latlon_to_xy(r["lat"], r["lon"]), axis=1, result_type="expand")
    stations["receiver_x_km"] = xy[0]
    stations["receiver_y_km"] = xy[1]
    return stations


def origin_depth_km(origin) -> float:
    depth = getattr(origin, "depth", np.nan)
    if depth is None:
        return float("nan")
    depth = float(depth)
    return depth / 1000.0 if abs(depth) > 1000.0 else depth


def build_pick_database(catalog: pd.DataFrame, xml_log: pd.DataFrame, xmlmod) -> pd.DataFrame:
    pick_path = DATA / f"sgc_new_events_vp_picks_raw_non{CANDIDATE_NON_NEST}_nest{CANDIDATE_NEST}.csv"
    if pick_path.exists():
        return pd.read_csv(pick_path)

    stations = station_table()
    station_map = stations.set_index("station").to_dict("index")
    domain_stations = set(stations["station"].astype(str))
    meta = catalog.set_index("event_id").to_dict("index")
    rows = []
    failures = []

    ok_xml = xml_log[xml_log["http_status"].eq(200)].copy()
    for n, item in enumerate(ok_xml.itertuples(index=False), start=1):
        event_id = str(item.event_id)
        xml_path = Path(str(item.xml_file))
        try:
            event, origin, magnitude = xmlmod.parse_event_xml(xml_path)
            picks = xmlmod.first_domain_p_picks(event, domain_stations)
        except Exception as exc:
            failures.append({"event_id": event_id, "error": repr(exc)})
            continue
        if getattr(origin, "time", None) is None:
            failures.append({"event_id": event_id, "error": "missing origin time"})
            continue
        event_lat = float(getattr(origin, "latitude", np.nan))
        event_lon = float(getattr(origin, "longitude", np.nan))
        event_depth = origin_depth_km(origin)
        if not (np.isfinite(event_lat) and np.isfinite(event_lon) and np.isfinite(event_depth)):
            failures.append({"event_id": event_id, "error": "missing origin coordinates"})
            continue
        source_x, source_y = latlon_to_xy(event_lat, event_lon)
        cat = meta.get(event_id, {})
        for pick in picks:
            sta = pick["station"]
            station_row = station_map.get(sta)
            if station_row is None:
                continue
            travel_time = float(pick["pick_time"] - origin.time)
            if not (0.0 < travel_time <= MAX_TRAVEL_TIME_S):
                continue
            rows.append(
                {
                    "event_id": event_id,
                    "selection_group": cat.get("selection_group", ""),
                    "origin_time_utc": str(origin.time),
                    "event_lat": event_lat,
                    "event_lon": event_lon,
                    "event_depth_km": event_depth,
                    "source_x_km": source_x,
                    "source_y_km": source_y,
                    "source_z_km": event_depth,
                    "magnitude": getattr(magnitude, "mag", cat.get("magnitude", np.nan)) if magnitude is not None else cat.get("magnitude", np.nan),
                    "catalog_rms_s": cat.get("rms_s", np.nan),
                    "catalog_phases": cat.get("phases", np.nan),
                    "network": pick["network"],
                    "station": sta,
                    "location": pick["location"],
                    "channel": pick["channel"],
                    "station_lat": station_row["lat"],
                    "station_lon": station_row["lon"],
                    "receiver_x_km": station_row["receiver_x_km"],
                    "receiver_y_km": station_row["receiver_y_km"],
                    "pick_time_utc": str(pick["pick_time"]),
                    "travel_time_s": travel_time,
                    "xml_file": str(xml_path),
                }
            )
        if n % 250 == 0:
            print(f"[Picks] parsed {n}/{len(ok_xml)} XML files", flush=True)

    picks_df = pd.DataFrame(rows)
    picks_df.to_csv(pick_path, index=False)
    pd.DataFrame(failures).to_csv(DATA / "sgc_new_events_xml_parse_failures.csv", index=False)
    return picks_df


def choose_inversion_events(picks_raw: pd.DataFrame) -> pd.DataFrame:
    out_path = DATA / (
        f"sgc_new_events_vp_picks_selected_min{MIN_PICKS_PER_EVENT}_"
        f"non{TARGET_NON_NEST}_nest{TARGET_NEST}_for_joint_inversion.csv"
    )
    if out_path.exists():
        return pd.read_csv(out_path)
    counts = picks_raw.groupby("event_id").size().rename("n_picks").reset_index()
    evt = picks_raw.drop_duplicates("event_id").merge(counts, on="event_id")
    evt = evt[evt["n_picks"] >= MIN_PICKS_PER_EVENT].copy()
    evt["selection_group"] = np.where(evt["event_depth_km"].between(120.0, 170.0), "bucaramanga_nest", "domain_20_120")
    evt = quality_sort(evt.rename(columns={"catalog_rms_s": "rms_s", "catalog_phases": "phases"}))
    nest_ids = evt[evt["selection_group"].eq("bucaramanga_nest")].head(TARGET_NEST)["event_id"].tolist()
    non_ids = evt[evt["selection_group"].ne("bucaramanga_nest")].head(TARGET_NON_NEST)["event_id"].tolist()
    selected_ids = set(non_ids + nest_ids)
    selected = picks_raw[picks_raw["event_id"].isin(selected_ids)].copy()
    selected.to_csv(out_path, index=False)
    evt[evt["event_id"].isin(selected_ids)].to_csv(
        DATA / f"sgc_new_events_selected_event_summary_min{MIN_PICKS_PER_EVENT}_non{TARGET_NON_NEST}_nest{TARGET_NEST}.csv",
        index=False,
    )
    return selected


def unique_source_nodes_from_picks(qmod, picks: pd.DataFrame, xg, yg, zg) -> dict[str, tuple[int, int, int]]:
    nodes = {}
    for row in picks.drop_duplicates("event_id").itertuples(index=False):
        nodes[str(row.event_id)] = qmod.coord_to_index_xyz(float(row.source_x_km), float(row.source_y_km), float(row.source_z_km), xg, yg, zg)
    return nodes


def receiver_nodes(qmod, picks: pd.DataFrame, xg, yg, zg) -> dict[str, tuple[int, int, int]]:
    nodes = {}
    for row in picks.drop_duplicates("station").itertuples(index=False):
        nodes[str(row.station)] = qmod.coord_to_index_xyz(float(row.receiver_x_km), float(row.receiver_y_km), 0.0, xg, yg, zg)
    return nodes


def build_station_cache(qmod, vp: np.ndarray, station_nodes: dict[str, tuple[int, int, int]], xg, yg, zg, return_predecessors: bool):
    """Compute one graph travel-time field per station.

    The graph is undirected, so travel time from event to station equals travel
    time from station to event. With thousands of events and only a handful of
    stations, this is much faster than one Dijkstra solve per source event.
    """
    dx = float(xg[1] - xg[0])
    dy = float(yg[1] - yg[0])
    dz = float(zg[1] - zg[0])
    ny, nx = vp.shape[1], vp.shape[2]
    cache = {}
    t0 = time.time()
    if TRAVEL_SOLVER == "fmm":
        slowness = 1.0 / np.maximum(vp, 1.0e-9)
        for station, kji in station_nodes.items():
            src_idx = qmod.idx_linear(kji[0], kji[1], kji[2], ny, nx)
            T = qmod.fast_sweeping_3d(slowness, kji[0], kji[1], kji[2], dx, dy, dz, qmod.FSM_SWEEPS, qmod.BIG_T)
            cache[station] = {"idx": int(src_idx), "kji": kji, "T": T, "dist": T.ravel()}
    else:
        graph = qmod.build_travel_graph(vp, dx, dy, dz)
        for station, kji in station_nodes.items():
            src_idx = qmod.idx_linear(kji[0], kji[1], kji[2], ny, nx)
            if return_predecessors:
                dist, predecessors = dijkstra(graph, directed=False, indices=src_idx, return_predecessors=True)
                cache[station] = {"idx": int(src_idx), "kji": kji, "dist": dist, "pred": predecessors}
            else:
                dist = dijkstra(graph, directed=False, indices=src_idx, return_predecessors=False)
                cache[station] = {"idx": int(src_idx), "kji": kji, "dist": dist}
    elapsed = time.time() - t0
    return cache, elapsed


def build_vp_rows_station_cache(qmod, vp: np.ndarray, picks: pd.DataFrame, xg, yg, zg):
    dx = float(xg[1] - xg[0])
    dy = float(yg[1] - yg[0])
    dz = float(zg[1] - zg[0])
    ny, nx = vp.shape[1], vp.shape[2]
    station_nodes = receiver_nodes(qmod, picks, xg, yg, zg)
    station_cache, cache_seconds = build_station_cache(qmod, vp, station_nodes, xg, yg, zg, return_predecessors=True)

    rows, obs, pred, row_ids = [], [], [], []
    failures = 0
    for row_id, row in enumerate(picks.itertuples(index=False)):
        station = str(row.station)
        cache = station_cache.get(station)
        if cache is None:
            failures += 1
            continue
        skji = qmod.coord_to_index_xyz(float(row.source_x_km), float(row.source_y_km), float(row.source_z_km), xg, yg, zg)
        src_idx = qmod.idx_linear(skji[0], skji[1], skji[2], ny, nx)
        tcalc = float(cache["dist"][src_idx])
        if "T" in cache:
            ray, ok = qmod.backtrace_build_row(cache["T"], cache["kji"], skji, dx, dy, dz, vp=None, mode="slowness")
        else:
            ray, ok = qmod.backtrace_build_row_graph(
                cache["pred"],
                int(cache["idx"]),
                int(src_idx),
                vp.shape,
                dx,
                dy,
                dz,
                vp=None,
                mode="slowness",
            )
        if (not ok) or (not ray) or (not np.isfinite(tcalc)):
            failures += 1
            continue
        rows.append(ray)
        obs.append(float(row.travel_time_s))
        pred.append(tcalc)
        row_ids.append(int(row_id))
    return rows, np.asarray(obs), np.asarray(pred), np.asarray(row_ids), failures, cache_seconds


def candidate_nodes(center: tuple[int, int, int], shape: tuple[int, int, int], radius: int, zg: np.ndarray | None = None) -> list[tuple[int, int, int]]:
    nz, ny, nx = shape
    ck, cj, ci = center
    out = []
    for dk in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            for di in range(-radius, radius + 1):
                k, j, i = ck + dk, cj + dj, ci + di
                if 0 <= k < nz and 0 <= j < ny and 0 <= i < nx:
                    if zg is not None and not (20.0 <= float(zg[k]) <= 170.0):
                        continue
                    out.append((k, j, i))
    return out


def relocate_events_grid_search(qmod, vp: np.ndarray, picks: pd.DataFrame, xg, yg, zg, round_id: int, station_statics: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ny, nx = vp.shape[1], vp.shape[2]
    rec_nodes = receiver_nodes(qmod, picks, xg, yg, zg)
    station_cache, cache_seconds = build_station_cache(qmod, vp, rec_nodes, xg, yg, zg, return_predecessors=False)
    source_nodes = unique_source_nodes_from_picks(qmod, picks, xg, yg, zg)

    relocated = []
    picks_out = []
    groups = {eid: df.copy() for eid, df in picks.groupby("event_id", sort=False)}
    t0 = time.time()
    for n, (event_id, group) in enumerate(groups.items(), start=1):
        center = source_nodes[str(event_id)]
        stations = group["station"].astype(str).tolist()
        obs = group["travel_time_s"].to_numpy(dtype=float)
        best = None
        current_rms = float("nan")
        for cand in candidate_nodes(center, vp.shape, RELOCATION_RADIUS_NODES, zg=zg):
            cand_idx = qmod.idx_linear(cand[0], cand[1], cand[2], ny, nx)
            stat = np.array([float(station_statics.get(sta, 0.0)) for sta in stations], dtype=float)
            tcalc = np.array([station_cache[sta]["dist"][cand_idx] for sta in stations], dtype=float) + stat
            if not np.all(np.isfinite(tcalc)):
                continue
            dt0 = float(np.median(obs - tcalc))
            residual = obs - (tcalc + dt0)
            rr = rms(residual)
            if cand == center:
                current_rms = rr
            if best is None or rr < best["rms_s"]:
                best = {"kji": cand, "dt0_s": dt0, "rms_s": rr, "residuals": residual, "pred": tcalc}
        if best is None:
            continue
        x, y, z = qmod.coord_from_kji(best["kji"], xg, yg, zg)
        lat, lon = xy_to_latlon(x, y)
        shift_xyz = np.array([x, y, z], dtype=float) - group[["source_x_km", "source_y_km", "source_z_km"]].iloc[0].to_numpy(dtype=float)
        relocated.append(
            {
                "event_id": event_id,
                "round": int(round_id),
                "n_picks": int(len(group)),
                "catalog_x_km": float(group["source_x_km"].iloc[0]),
                "catalog_y_km": float(group["source_y_km"].iloc[0]),
                "catalog_z_km": float(group["source_z_km"].iloc[0]),
                "relocated_x_km": float(x),
                "relocated_y_km": float(y),
                "relocated_z_km": float(z),
                "relocated_lat": float(lat),
                "relocated_lon": float(lon),
                "origin_time_shift_s": float(best["dt0_s"]),
                "rms_before_s": float(current_rms),
                "rms_after_s": float(best["rms_s"]),
                "horizontal_shift_km": float(np.linalg.norm(shift_xyz[:2])),
                "vertical_shift_km": float(shift_xyz[2]),
                "total_shift_km": float(np.linalg.norm(shift_xyz)),
            }
        )
        out = group.copy()
        out["source_x_km"] = float(x)
        out["source_y_km"] = float(y)
        out["source_z_km"] = float(z)
        out["event_lat"] = float(lat)
        out["event_lon"] = float(lon)
        out["event_depth_km"] = float(z)
        out["origin_time_shift_s"] = float(best["dt0_s"])
        out["travel_time_s"] = out["travel_time_s"].astype(float) - float(best["dt0_s"])
        picks_out.append(out)
        if n % 250 == 0:
            print(f"[Reloc {round_id}] {n}/{len(groups)} events; station fields={len(station_cache)}; elapsed={time.time() - t0:.1f}s", flush=True)

    reloc_df = pd.DataFrame(relocated)
    picks_df = pd.concat(picks_out, ignore_index=True) if picks_out else pd.DataFrame()
    stats = {
        "round": int(round_id),
        "events_relocated": int(len(reloc_df)),
        "station_graph_fields": int(len(station_cache)),
        "station_graph_seconds": float(cache_seconds),
        "median_rms_before_s": float(reloc_df["rms_before_s"].median()) if len(reloc_df) else float("nan"),
        "median_rms_after_s": float(reloc_df["rms_after_s"].median()) if len(reloc_df) else float("nan"),
        "median_horizontal_shift_km": float(reloc_df["horizontal_shift_km"].median()) if len(reloc_df) else float("nan"),
        "median_total_shift_km": float(reloc_df["total_shift_km"].median()) if len(reloc_df) else float("nan"),
    }
    return picks_df, reloc_df, stats


def build_active_mask(qmod, vp: np.ndarray, picks: pd.DataFrame, xg, yg, zg, mask_domain: np.ndarray):
    rows, obs, pred, row_ids, failures, elapsed = build_vp_rows_station_cache(qmod, vp, picks, xg, yg, zg)
    hits = qmod.hit_count_from_rows(rows, vp.size).reshape(vp.shape)
    active = mask_domain & (hits >= VP_MIN_HITS)
    return active, hits, (rows, obs, pred, row_ids, failures, elapsed)


def solve_augmented_with_station_statics(qmod, A: sp.csr_matrix, b: np.ndarray, stations_for_rows: np.ndarray, active_mask3: np.ndarray):
    stations = sorted({str(item) for item in stations_for_rows})
    sta_pos = {sta: i for i, sta in enumerate(stations)}
    rr = np.arange(len(stations_for_rows), dtype=int)
    cc = np.array([sta_pos[str(sta)] for sta in stations_for_rows], dtype=int)
    vv = np.ones(len(stations_for_rows), dtype=float)
    S = sp.csr_matrix((vv, (rr, cc)), shape=(len(stations_for_rows), len(stations)), dtype=float)

    blocks = [sp.hstack([A, S], format="csr")]
    rhs = [b]
    ncell = A.shape[1]
    nsta = len(stations)
    if VP_LAPLACE > 0:
        L = qmod.laplacian_active_3d(active_mask3)
        if L.shape[0] > 0:
            blocks.append(sp.hstack([VP_LAPLACE * L, sp.csr_matrix((L.shape[0], nsta))], format="csr"))
            rhs.append(np.zeros(L.shape[0], dtype=float))
    if VP_DAMP > 0:
        blocks.append(sp.hstack([VP_DAMP * sp.eye(ncell, format="csr"), sp.csr_matrix((ncell, nsta))], format="csr"))
        rhs.append(np.zeros(ncell, dtype=float))
    if STATION_STATIC_DAMP > 0:
        blocks.append(sp.hstack([sp.csr_matrix((nsta, ncell)), STATION_STATIC_DAMP * sp.eye(nsta, format="csr")], format="csr"))
        rhs.append(np.zeros(nsta, dtype=float))
    sol = lsmr(sp.vstack(blocks, format="csr"), np.concatenate(rhs), maxiter=VP_ITERLIM)
    x = sol[0]
    return (
        x[:ncell],
        dict(zip(stations, x[ncell:])),
        {"istop": int(sol[1]), "it": int(sol[2]), "normr": float(sol[3]), "acond": float(sol[6])},
    )


def update_velocity_once(qmod, vp: np.ndarray, picks: pd.DataFrame, xg, yg, zg, active: np.ndarray, label: str, station_statics: dict[str, float]):
    rows, obs, pred, row_ids, failures, elapsed = build_vp_rows_station_cache(qmod, vp, picks, xg, yg, zg)
    A, active_idx, kept = qmod.rows_to_csr(rows, active.ravel())
    kept = np.asarray(kept, dtype=int)
    obs_kept = obs[kept]
    pred_kept = pred[kept]
    station_for_rows = picks.iloc[row_ids[kept]]["station"].astype(str).to_numpy()
    current_static = np.array([float(station_statics.get(sta, 0.0)) for sta in station_for_rows], dtype=float)
    residual = obs_kept - pred_kept - current_static
    pre_rms = rms(residual)
    if USE_STATION_STATICS:
        dslow, dstat, info = solve_augmented_with_station_statics(qmod, A, residual, station_for_rows, active)
        station_statics = station_statics.copy()
        for sta, val in dstat.items():
            station_statics[sta] = float(station_statics.get(sta, 0.0) + val)
    else:
        dslow, info = qmod.solve_augmented(A, residual, active, VP_LAPLACE, VP_DAMP, VP_ITERLIM)
        dstat = {}
    svec = (1.0 / np.maximum(vp, 1.0e-9)).ravel()
    snew = svec.copy()
    snew[active_idx] = np.clip(snew[active_idx] + VP_ALPHA * dslow, S_MIN, S_MAX)
    vp_new = (1.0 / np.maximum(snew, 1.0e-12)).reshape(vp.shape)
    print(f"[Vp {label}] solver={TRAVEL_SOLVER} station_statics={USE_STATION_STATICS} equations={A.shape[0]} active={A.shape[1]} pre_rms={pre_rms:.3f}s lsmr_it={info['it']}", flush=True)
    return vp_new, station_statics, {
        "label": label,
        "travel_solver": TRAVEL_SOLVER,
        "station_statics": bool(USE_STATION_STATICS),
        "equations": int(A.shape[0]),
        "active_cells": int(A.shape[1]),
        "data_parameter_ratio_cells_only": float(A.shape[0] / max(A.shape[1], 1)),
        "pre_rms_s": float(pre_rms),
        "fsm_failures": int(failures),
        "fsm_build_seconds": float(elapsed),
        "lsmr_iterations": int(info["it"]),
        "lsmr_istop": int(info["istop"]),
        "lsmr_acond": float(info["acond"]),
        "station_static_l2_s": float(np.linalg.norm(list(station_statics.values()))) if station_statics else 0.0,
        "station_static_max_abs_s": float(max([abs(v) for v in station_statics.values()] or [0.0])),
    }, {"A": A, "active_idx": active_idx, "rows": rows, "obs": obs, "pred": pred, "row_ids": row_ids, "kept": kept, "station_for_rows": station_for_rows}


def final_predictions(qmod, vp: np.ndarray, picks: pd.DataFrame, xg, yg, zg, active: np.ndarray, station_statics: dict[str, float]):
    rows, obs, pred, row_ids, failures, elapsed = build_vp_rows_station_cache(qmod, vp, picks, xg, yg, zg)
    A, active_idx, kept = qmod.rows_to_csr(rows, active.ravel())
    kept = np.asarray(kept, dtype=int)
    out = picks.iloc[row_ids[kept]].copy().reset_index(drop=True)
    stat = out["station"].astype(str).map(lambda sta: float(station_statics.get(sta, 0.0))).to_numpy(dtype=float)
    out["station_static_s"] = stat
    out["pred_final_raw_s"] = pred[kept]
    out["pred_final_s"] = pred[kept] + stat
    out["residual_final_s"] = obs[kept] - out["pred_final_s"].to_numpy(dtype=float)
    return out, {"A": A, "active_idx": active_idx, "kept": kept, "rows": rows, "obs": obs, "pred": pred, "row_ids": row_ids, "failures": failures, "elapsed": elapsed}


def run_checkerboard(qmod, final_payload: dict[str, Any], active: np.ndarray, vp_ref: np.ndarray):
    A = final_payload["A"]
    active_idx = final_payload["active_idx"]
    active_count = A.shape[1]
    if active_count == 0:
        return None, None, {}
    x = np.arange(active_count)
    # Deterministic alternating synthetic model in active-cell order. The model
    # scale is a small slowness perturbation, matching the Q-project convention.
    true = CHECKERBOARD_SLOWNESS_FRACTION * (1.0 / np.nanmedian(vp_ref)) * np.where((x // 7) % 2 == 0, 1.0, -1.0)
    rng = np.random.default_rng(20260514)
    data = A @ true + rng.normal(0.0, CHECKERBOARD_NOISE_S, size=A.shape[0])
    rec, info = qmod.solve_augmented(A, data, active, VP_LAPLACE, VP_DAMP, VP_ITERLIM)
    corr = float(np.corrcoef(true, rec)[0, 1]) if active_count > 1 else float("nan")
    amp_ratio = float(np.std(rec) / np.std(true)) if np.std(true) > 0 else float("nan")
    true_cube = np.full(vp_ref.size, np.nan, dtype=float)
    rec_cube = np.full(vp_ref.size, np.nan, dtype=float)
    true_cube[active_idx] = true
    rec_cube[active_idx] = rec
    metrics = {
        "checkerboard_active_cells": int(active_count),
        "checkerboard_noise_s": CHECKERBOARD_NOISE_S,
        "checkerboard_corr_active": corr,
        "checkerboard_amplitude_ratio_active": amp_ratio,
        "checkerboard_lsmr_iterations": int(info["it"]),
    }
    return true_cube.reshape(vp_ref.shape), rec_cube.reshape(vp_ref.shape), metrics


def run_spike_test(qmod, final_payload: dict[str, Any], active: np.ndarray, vp_ref: np.ndarray):
    A = final_payload["A"]
    active_idx = final_payload["active_idx"]
    if A.shape[1] == 0:
        return None, None, {}
    # Use the active cell nearest to the median active coordinate as a compact
    # point-spread diagnostic.
    coords = np.array([qmod.idx_to_kji(int(idx), vp_ref.shape[1], vp_ref.shape[2]) for idx in active_idx], dtype=float)
    center = np.nanmedian(coords, axis=0)
    spike_col = int(np.argmin(np.sum((coords - center) ** 2, axis=1)))
    true = np.zeros(A.shape[1], dtype=float)
    true[spike_col] = 0.10 * (1.0 / np.nanmedian(vp_ref))
    rng = np.random.default_rng(20260515)
    data = A @ true + rng.normal(0.0, CHECKERBOARD_NOISE_S, size=A.shape[0])
    rec, info = qmod.solve_augmented(A, data, active, VP_LAPLACE, VP_DAMP, VP_ITERLIM)
    recovered_fraction = float(rec[spike_col] / true[spike_col]) if true[spike_col] != 0 else float("nan")
    spread_ratio = float((np.linalg.norm(rec) - abs(rec[spike_col])) / max(np.linalg.norm(rec), 1.0e-12))
    true_cube = np.full(vp_ref.size, np.nan, dtype=float)
    rec_cube = np.full(vp_ref.size, np.nan, dtype=float)
    true_cube[active_idx] = true
    rec_cube[active_idx] = rec
    metrics = {
        "spike_active_cells": int(A.shape[1]),
        "spike_column": int(spike_col),
        "spike_grid_index": [int(item) for item in coords[spike_col]],
        "spike_recovered_fraction": recovered_fraction,
        "spike_spread_ratio": spread_ratio,
        "spike_lsmr_iterations": int(info["it"]),
    }
    return true_cube.reshape(vp_ref.shape), rec_cube.reshape(vp_ref.shape), metrics


def run_lcurve_diagnostic(qmod, final_payload: dict[str, Any], active: np.ndarray, residual: np.ndarray) -> pd.DataFrame:
    A = final_payload["A"]
    rows = []
    for lap in [1.0, 2.0, 3.0, 5.0]:
        for damp in [0.10, 0.25, 0.50, 1.00]:
            sol, info = qmod.solve_augmented(A, residual, active, lap, damp, min(VP_ITERLIM, 600))
            data_norm = float(np.linalg.norm(A @ sol - residual))
            model_norm = float(np.linalg.norm(sol))
            L = qmod.laplacian_active_3d(active)
            lap_norm = float(np.linalg.norm(L @ sol)) if L.shape[0] else float("nan")
            rows.append(
                {
                    "laplace_weight": float(lap),
                    "damp_weight": float(damp),
                    "data_norm": data_norm,
                    "model_norm": model_norm,
                    "laplacian_norm": lap_norm,
                    "lsmr_iterations": int(info["it"]),
                    "lsmr_acond": float(info["acond"]),
                    "selected": bool(abs(lap - VP_LAPLACE) < 1.0e-9 and abs(damp - VP_DAMP) < 1.0e-9),
                }
            )
    return pd.DataFrame(rows)


def plot_area(picks: pd.DataFrame, stations: pd.DataFrame) -> Path:
    events = picks.drop_duplicates("event_id")
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(events["event_lon"], events["event_lat"], s=8, c="black", alpha=0.45, label="New events")
    nest = events[events["event_depth_km"].between(120.0, 170.0)]
    if len(nest):
        ax.scatter(nest["event_lon"], nest["event_lat"], s=12, facecolors="none", edgecolors="0.3", linewidths=0.5, label="Bucaramanga nest")
    ax.scatter(stations["lon"], stations["lat"], s=55, c="red", marker="^", edgecolor="black", linewidth=0.5, label="Stations")
    for _, row in stations.iterrows():
        ax.text(float(row["lon"]) + 0.015, float(row["lat"]) + 0.015, str(row["station"]), fontsize=7)
    ax.set_xlim(LON_MIN - 0.05, LON_MAX + 0.05)
    ax.set_ylim(LAT_MIN - 0.05, LAT_MAX + 0.05)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Tomography area: new SGC events and stations")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    path = FIG / "sgc_new_events_area_stations_events.png"
    fig.savefig(path, dpi=250)
    plt.close(fig)
    return path


def plot_relocation(reloc: pd.DataFrame) -> Path:
    latest = reloc.sort_values("round").drop_duplicates("event_id", keep="last")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(latest["horizontal_shift_km"], bins=30, color="#4778b6", alpha=0.85)
    axes[0].set_xlabel("Horizontal shift (km)")
    axes[0].set_ylabel("Events")
    axes[0].set_title("Relocation horizontal shifts")
    axes[1].scatter(latest["rms_before_s"], latest["rms_after_s"], s=8, c="black", alpha=0.4)
    lim = [0, max(float(latest["rms_before_s"].quantile(0.98)), float(latest["rms_after_s"].quantile(0.98)), 1.0)]
    axes[1].plot(lim, lim, "r--", lw=1)
    axes[1].set_xlim(lim)
    axes[1].set_ylim(lim)
    axes[1].set_xlabel("RMS before relocation (s)")
    axes[1].set_ylabel("RMS after relocation (s)")
    axes[1].set_title("Per-event travel-time fit")
    fig.tight_layout()
    path = FIG / "sgc_new_events_relocation_diagnostics.png"
    fig.savefig(path, dpi=250)
    plt.close(fig)
    return path


def plot_vp_slices(vp0: np.ndarray, vp: np.ndarray, active: np.ndarray, xg, yg, zg) -> Path:
    depths = [40.0, 80.0, 140.0]
    fig, axes = plt.subplots(len(depths), 3, figsize=(12, 10))
    delta = 100.0 * (vp - vp0) / np.maximum(vp0, 1.0e-9)
    for r, depth in enumerate(depths):
        k = int(np.argmin(np.abs(zg - depth)))
        arrays = [vp0[k], vp[k], np.where(active[k], delta[k], np.nan)]
        titles = [f"Initial Vp z={zg[k]:.0f} km", "Final Vp", "Delta Vp (%) active"]
        cmaps = ["viridis", "viridis", "RdBu_r"]
        for c in range(3):
            im = axes[r, c].imshow(
                arrays[c],
                extent=[xg.min(), xg.max(), yg.min(), yg.max()],
                origin="lower",
                aspect="auto",
                cmap=cmaps[c],
            )
            axes[r, c].set_title(titles[c], fontsize=10)
            axes[r, c].set_xlabel("x (km)")
            axes[r, c].set_ylabel("y (km)")
            fig.colorbar(im, ax=axes[r, c], shrink=0.75)
    fig.tight_layout()
    path = FIG / "sgc_new_events_vp_slices_initial_final_delta.png"
    fig.savefig(path, dpi=250)
    plt.close(fig)
    return path


def plot_checker(true_cb: np.ndarray | None, rec_cb: np.ndarray | None, active: np.ndarray, xg, yg, zg) -> Path | None:
    if true_cb is None or rec_cb is None:
        return None
    depth = 80.0
    k = int(np.argmin(np.abs(zg - depth)))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    arrays = [np.where(active[k], true_cb[k], np.nan), np.where(active[k], rec_cb[k], np.nan), active[k].astype(float)]
    titles = ["Imposed slowness", "Recovered slowness", "Active mask"]
    for ax, arr, title in zip(axes, arrays, titles):
        im = ax.imshow(arr, extent=[xg.min(), xg.max(), yg.min(), yg.max()], origin="lower", aspect="auto", cmap="RdBu_r")
        ax.set_title(title)
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    path = FIG / "sgc_new_events_checkerboard_diagnostic.png"
    fig.savefig(path, dpi=250)
    plt.close(fig)
    return path


def plot_spike(true_spike: np.ndarray | None, rec_spike: np.ndarray | None, active: np.ndarray, xg, yg, zg) -> Path | None:
    if true_spike is None or rec_spike is None:
        return None
    finite = np.isfinite(true_spike) & (np.abs(true_spike) > 0)
    if not finite.any():
        return None
    k = int(np.argwhere(finite)[0][0])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    arrays = [np.where(active[k], true_spike[k], np.nan), np.where(active[k], rec_spike[k], np.nan), np.where(active[k], rec_spike[k] - np.nan_to_num(true_spike[k]), np.nan)]
    titles = ["Imposed spike", "Recovered spike", "Difference"]
    for ax, arr, title in zip(axes, arrays, titles):
        im = ax.imshow(arr, extent=[xg.min(), xg.max(), yg.min(), yg.max()], origin="lower", aspect="auto", cmap="RdBu_r")
        ax.set_title(title)
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    path = FIG / "sgc_new_events_spike_diagnostic.png"
    fig.savefig(path, dpi=250)
    plt.close(fig)
    return path


def plot_lcurve(lcurve: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    for damp, grp in lcurve.groupby("damp_weight"):
        ax.plot(grp["laplacian_norm"], grp["data_norm"], marker="o", label=f"damp={damp:g}")
    sel = lcurve[lcurve["selected"]]
    if len(sel):
        ax.scatter(sel["laplacian_norm"], sel["data_norm"], s=80, c="red", zorder=5, label="chosen")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Laplacian norm")
    ax.set_ylabel("Data residual norm")
    ax.set_title("Linearized regularization trade-off")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIG / "sgc_new_events_lcurve_diagnostic.png"
    fig.savefig(path, dpi=250)
    plt.close(fig)
    return path


def write_paper(metrics: dict[str, Any], figures: dict[str, str | None]) -> Path:
    tex = rf"""
\documentclass[review]{{elsarticle}}
\usepackage{{graphicx}}
\usepackage{{amsmath}}
\usepackage{{booktabs}}
\usepackage{{siunitx}}
\journal{{Journal of South American Earth Sciences}}

\begin{{document}}
\begin{{frontmatter}}
\title{{Coupled P-wave velocity tomography and earthquake relocation in the Middle Magdalena Valley using new SGC events}}
\author{{Draft for internal revision}}
\begin{{abstract}}
We update the simultaneous location--tomography experiment for the Middle Magdalena Valley using a new SGC catalog subset independent from the events previously used in the Q-tomography workflow. The starting velocity model is the final 10 km P-wave velocity field from the Q-tomography study, and the same geographical domain and station set are retained. The new dataset contains {metrics['selected_events']} relocated events, including {metrics['selected_nest_events']} Bucaramanga-nest earthquakes between 120 and 170 km depth, and {metrics['selected_picks']} P arrivals. A coupled workflow alternates local grid-node hypocentral relocation and sparse 3-D slowness updates. The final velocity model should be interpreted as a consistency test of the Q-derived velocity field under an independent event set, not as a fully joint Thurber inversion.
\end{{abstract}}
\begin{{keyword}}
Middle Magdalena Valley \sep earthquake relocation \sep P-wave tomography \sep Bucaramanga nest \sep Fast marching
\end{{keyword}}
\end{{frontmatter}}

\section{{Data and method}}
The input catalog was built from the SGC expert catalog for 2022--2026 using the same study area as the Q-tomography work (longitude {LON_MIN} to {LON_MAX}, latitude {LAT_MIN} to {LAT_MAX}). Events previously used in the Q workflow were excluded. From the remaining catalog, we selected high-quality events with RMS $<1.0$ s and depths between 20 and 170 km, forcing the inclusion of 300 events from the Bucaramanga seismic nest. XML metadata were downloaded from SGC and first vertical-component P picks were retained for the station set used in the final Q-tomography run.

Travel times were predicted on the 10 km grid by a 26-neighbour minimum-time graph operator. Hypocenters were relocated by searching neighbouring grid nodes around the catalog hypocenter and estimating a robust origin-time correction for each candidate. After relocation, slowness perturbations were solved with LSMR using damping {VP_DAMP} and Laplacian smoothing {VP_LAPLACE}. This produces an iterative coupled location--velocity workflow with the final Q-derived Vp model as prior.

\section{{Results}}
The selected inversion set contains {metrics['selected_events']} events and {metrics['selected_picks']} P picks from {metrics['stations']} stations. The first relocation round reduced the median event-level RMS from {metrics['relocation_rounds'][0]['median_rms_before_s']:.2f} s to {metrics['relocation_rounds'][0]['median_rms_after_s']:.2f} s, with a median horizontal shift of {metrics['relocation_rounds'][0]['median_horizontal_shift_km']:.1f} km. The final P-wave inversion used {metrics['final_equations']} equations and {metrics['active_cells']} active cells, giving a data-to-parameter ratio of {metrics['data_parameter_ratio']:.2f}. The final RMS over retained equations is {metrics['final_rms_s']:.2f} s.

\begin{{figure}}[p]
\centering
\includegraphics[width=0.85\textwidth]{{{figures['area']}}}
\caption{{Tomography area, new SGC events used in the coupled inversion, Bucaramanga nest events, and stations.}}
\end{{figure}}

\begin{{figure}}[p]
\centering
\includegraphics[width=0.9\textwidth]{{{figures['relocation']}}}
\caption{{Relocation diagnostics for the new SGC event set.}}
\end{{figure}}

\begin{{figure}}[p]
\centering
\includegraphics[width=\textwidth]{{{figures['vp_slices']}}}
\caption{{Initial velocity from the final Q-tomography model, final coupled velocity model, and active-cell perturbations.}}
\end{{figure}}

\section{{Discussion}}
The new event set tests whether the velocity field obtained in the Q-tomography project remains useful when confronted with independent earthquakes. The inclusion of 300 Bucaramanga-nest events strengthens ray coverage at intermediate depths and is therefore particularly relevant for constraining the downgoing Nazca slab geometry. The relocation step absorbs part of the catalog origin-time and hypocentral error before updating velocity, reducing the risk that velocity perturbations simply compensate fixed-hypocenter bias. The model remains limited by the station geometry and by the discrete 10 km relocation grid; therefore, individual 10 km cells should not be overinterpreted. Broad positive Vp perturbations at intermediate depth are most defensibly interpreted as consistency with a high-velocity slab-related corridor, whereas shallow negative perturbations are consistent with sedimentary and upper-crustal heterogeneity in the basin.

\section{{Conclusions}}
Using events independent from the Q-tomography dataset, the coupled workflow provides a direct test of the final Q-derived Vp model as an initial condition for simultaneous relocation and tomography. The larger and independent catalog improves the statistical basis of the velocity branch, especially after adding Bucaramanga-nest events. The result should be used to update the simultaneous-location paper with emphasis on relocated hypocenters, the robustness of broad Vp anomalies, and the role of the nest events in defining the intermediate-depth slab geometry.

\end{{document}}
"""
    path = PAPER / "joint_vp_relocation_sgc_new_events_jsames_draft.tex"
    path.write_text(tex, encoding="utf-8")
    return path


def main() -> None:
    ensure_dirs()
    qmod = import_module(Q_FSM_SCRIPT, "q_fsm_helpers")
    xmlmod = import_module(Q_XML_SCRIPT, "sgc_xml_helpers")

    print("[Setup] Selecting new events not used in Q tomography", flush=True)
    catalog = select_new_catalog()
    xml_log = download_xmls(catalog)
    picks_raw = build_pick_database(catalog, xml_log, xmlmod)
    picks = choose_inversion_events(picks_raw)
    if picks["event_id"].nunique() < 1000:
        raise RuntimeError("Too few events with usable P picks. Inspect XML parse/download logs.")

    stations = station_table()
    xg, yg, zg, vp0 = load_q_grid_and_vp()
    domain = domain_mask(xg, yg, zg)
    print(f"[Setup] Grid={vp0.shape}, dx={xg[1]-xg[0]:.1f} km, events={picks['event_id'].nunique()}, picks={len(picks)}", flush=True)

    area_fig = plot_area(picks, stations)

    vp = vp0.copy()
    current_picks = picks.copy()
    station_statics = {sta: 0.0 for sta in sorted(current_picks["station"].astype(str).unique())}
    relocation_tables = []
    relocation_stats = []
    history = []

    for round_id in range(1, COUPLED_ROUNDS + 1):
        print(f"[Round {round_id}] Relocating events", flush=True)
        current_picks, reloc, stats = relocate_events_grid_search(qmod, vp, current_picks, xg, yg, zg, round_id, station_statics)
        current_picks.to_csv(DATA / f"sgc_new_events_vp_picks_relocated_round{round_id}.csv", index=False)
        reloc.to_csv(DATA / f"sgc_new_events_relocations_round{round_id}.csv", index=False)
        relocation_tables.append(reloc)
        relocation_stats.append(stats)

        print(f"[Round {round_id}] Building active mask and updating Vp", flush=True)
        active, hits, _ = build_active_mask(qmod, vp, current_picks, xg, yg, zg, domain)
        vp, station_statics, hist, payload = update_velocity_once(qmod, vp, current_picks, xg, yg, zg, active, f"round{round_id}", station_statics)
        hist["round"] = int(round_id)
        history.append(hist)

    active, hits, _ = build_active_mask(qmod, vp, current_picks, xg, yg, zg, domain)
    pred_df, final_payload = final_predictions(qmod, vp, current_picks, xg, yg, zg, active, station_statics)
    final_rms = rms(pred_df["residual_final_s"].to_numpy(dtype=float))
    pred_df.to_csv(DATA / "sgc_new_events_final_vp_predictions.csv", index=False)
    current_picks.to_csv(DATA / "sgc_new_events_final_relocated_picks.csv", index=False)
    reloc_all = pd.concat(relocation_tables, ignore_index=True)
    reloc_all.to_csv(DATA / "sgc_new_events_relocations_all_rounds.csv", index=False)
    pd.DataFrame(history).to_csv(DATA / "sgc_new_events_joint_inversion_history.csv", index=False)

    true_cb, rec_cb, cb_metrics = run_checkerboard(qmod, final_payload, active, vp)
    true_spike, rec_spike, spike_metrics = run_spike_test(qmod, final_payload, active, vp)
    lcurve_residual = pred_df["residual_final_s"].to_numpy(dtype=float)
    lcurve = run_lcurve_diagnostic(qmod, final_payload, active, lcurve_residual)

    np.save(MODEL / "sgc_new_events_vp_initial_from_q.npy", vp0)
    np.save(MODEL / "sgc_new_events_vp_final_joint_relocation.npy", vp)
    np.save(MODEL / "sgc_new_events_vp_active_mask.npy", active)
    np.save(MODEL / "sgc_new_events_vp_hit_count.npy", hits)
    if true_cb is not None:
        np.save(MODEL / "sgc_new_events_checkerboard_true.npy", true_cb)
        np.save(MODEL / "sgc_new_events_checkerboard_recovered.npy", rec_cb)
    if true_spike is not None:
        np.save(MODEL / "sgc_new_events_spike_true.npy", true_spike)
        np.save(MODEL / "sgc_new_events_spike_recovered.npy", rec_spike)
    np.savez(MODEL / "sgc_new_events_grid_10km.npz", xg=xg, yg=yg, zg=zg)
    pd.DataFrame([{"station": sta, "static_s": val} for sta, val in sorted(station_statics.items())]).to_csv(DATA / "sgc_new_events_station_statics.csv", index=False)
    lcurve.to_csv(DATA / "sgc_new_events_lcurve_diagnostic.csv", index=False)

    reloc_fig = plot_relocation(reloc_all)
    vp_fig = plot_vp_slices(vp0, vp, active, xg, yg, zg)
    checker_fig = plot_checker(true_cb, rec_cb, active, xg, yg, zg)
    spike_fig = plot_spike(true_spike, rec_spike, active, xg, yg, zg)
    lcurve_fig = plot_lcurve(lcurve)

    metrics = {
        "run_name": "SGC new events coupled Vp tomography and relocation",
        "catalog_candidates": int(len(catalog)),
        "xml_download_ok": int(xml_log["http_status"].eq(200).sum()),
        "raw_pick_rows": int(len(picks_raw)),
        "raw_pick_events": int(picks_raw["event_id"].nunique()),
        "selected_picks": int(len(current_picks)),
        "selected_events": int(current_picks["event_id"].nunique()),
        "selected_nest_events": int(
            current_picks.drop_duplicates("event_id")["selection_group"].astype(str).eq("bucaramanga_nest").sum()
        ),
        "stations": int(current_picks["station"].nunique()),
        "station_list": sorted(current_picks["station"].astype(str).unique().tolist()),
        "grid_shape": list(vp.shape),
        "grid_spacing_km": {
            "dx": float(xg[1] - xg[0]),
            "dy": float(yg[1] - yg[0]),
            "dz": float(zg[1] - zg[0]),
        },
        "domain": {
            "lon_min": LON_MIN,
            "lon_max": LON_MAX,
            "lat_min": LAT_MIN,
            "lat_max": LAT_MAX,
            "max_depth_km": MAX_MODEL_DEPTH_KM,
        },
        "coupled_rounds": int(COUPLED_ROUNDS),
        "travel_solver": TRAVEL_SOLVER,
        "station_statics_enabled": bool(USE_STATION_STATICS),
        "station_statics": {sta: float(val) for sta, val in sorted(station_statics.items())},
        "relocation_rounds": relocation_stats,
        "active_cells": int(active.sum()),
        "final_equations": int(final_payload["A"].shape[0]),
        "data_parameter_ratio": float(final_payload["A"].shape[0] / max(int(active.sum()), 1)),
        "final_rms_s": float(final_rms),
        "regularization": {
            "vp_alpha": VP_ALPHA,
            "vp_damp": VP_DAMP,
            "vp_laplace": VP_LAPLACE,
            "vp_min_hits": VP_MIN_HITS,
        },
        "checkerboard": cb_metrics,
        "spike": spike_metrics,
        "figures": {
            "area": str(area_fig),
            "relocation": str(reloc_fig),
            "vp_slices": str(vp_fig),
            "checkerboard": str(checker_fig) if checker_fig else None,
            "spike": str(spike_fig) if spike_fig else None,
            "lcurve": str(lcurve_fig),
        },
    }
    (OUT / "sgc_new_events_joint_vp_relocation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    figures_for_tex = {
        "area": str(area_fig).replace("\\", "/"),
        "relocation": str(reloc_fig).replace("\\", "/"),
        "vp_slices": str(vp_fig).replace("\\", "/"),
        "checkerboard": str(checker_fig).replace("\\", "/") if checker_fig else None,
        "spike": str(spike_fig).replace("\\", "/") if spike_fig else None,
        "lcurve": str(lcurve_fig).replace("\\", "/"),
    }
    paper = write_paper(metrics, figures_for_tex)

    for src in [Path(__file__), Q_FSM_SCRIPT, Q_XML_SCRIPT]:
        if src.exists():
            shutil.copy2(src, CODE / src.name)

    print("[Done] Metrics:", json.dumps(metrics, indent=2), flush=True)
    print(f"[Done] Draft paper: {paper}", flush=True)


if __name__ == "__main__":
    main()
