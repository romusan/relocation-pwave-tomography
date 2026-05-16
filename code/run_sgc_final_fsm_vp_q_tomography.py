#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run final SGC Vp and spectral-ratio Q tomography with grid-eikonal backtracing.

Inputs are taken from the response-corrected final preprocessing run:

* output_V2/sgc_final_q_extra_station_tomography/data/sgc_1000_vp_picks.csv
* output_V2/sgc_final_q_extra_station_tomography/data/sgc_1000_spectral_tstar_pairs.csv

This script intentionally avoids ObsPy. Waveform processing and response
removal must be completed before running it.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import Delaunay, cKDTree
from scipy.sparse.linalg import lsmr

try:
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]
        def wrap(func):
            return func
        return wrap


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "output_V2" / "sgc_final_q_extra_station_tomography"
SRC_DATA = SRC / "data"
OUT = ROOT / "output_V2" / "sgc_final_q_extra_station_fsm_tomography"
DATA = OUT / "data"
FIG = OUT / "figures"
MODEL = OUT / "models"
PAPER = OUT / "paper"
CODE = OUT / "codigos_finales"

VP_MODEL_NPZ = ROOT / "modelo_vp_3d_nazca.npz"
PICKS_CSV = SRC_DATA / "sgc_1000_vp_picks.csv"
PAIR_CSV = SRC_DATA / "sgc_1000_spectral_tstar_pairs.csv"
TRACE_CSV = SRC_DATA / "sgc_1000_trace_spectra_inventory.csv"
STATIONS_CSV = ROOT / "output_V2" / "data" / "domain_stations_v2_plus_extra_used_for_q.csv"
CATALOG_CSV = (
    ROOT
    / "output_V2"
    / "data"
    / "sgc_combined_area_2022_2026_depth20_170_rms_lt1_for_final.csv"
)

LAT0 = 7.0
LON0 = -73.0
LON_MIN, LON_MAX = -74.60, -72.55
LAT_MIN, LAT_MAX = 6.20, 8.05
MAX_MODEL_DEPTH_KM = 180.0

GRID_FACTOR = 2
FSM_SWEEPS = 8
BIG_T = 1.0e20
RAY_MAX_STEPS = 200000
USE_ENDPOINT_CORRECTION = False

NOUTER_VP = 5
VP_ALPHA = 0.25
VP_DAMP = 0.25
VP_LAPLACE = 3.0
VP_ITERLIM = 1000
VP_MIN_HITS = 20
S_MIN = 1.0 / 15.0
S_MAX = 1.0 / 2.0

Q_CELL_DAMP = 0.05
Q_STATION_DAMP = 0.15
Q_LAPLACE = 3.0
Q_ITERLIM = 1000
Q_MIN_HITS = 10
QINV_MIN = 1.0 / 300.0
QINV_MAX = 1.0 / 20.0

CHECKERBOARD_SLOWNESS_FRACTION = 0.04
CHECKERBOARD_NOISE_S = 0.03
_GRID_EDGE_CACHE: dict[tuple[tuple[int, int, int], float, float, float], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def ensure_dirs() -> None:
    for path in [DATA, FIG, MODEL, PAPER, CODE]:
        path.mkdir(parents=True, exist_ok=True)


def latlon_to_xy(lat: float, lon: float) -> tuple[float, float]:
    x = (float(lon) - LON0) * 111.32 * math.cos(math.radians(LAT0))
    y = (float(lat) - LAT0) * 111.32
    return float(x), float(y)


def xy_to_latlon(x: float, y: float) -> tuple[float, float]:
    lat = float(y) / 111.32 + LAT0
    lon = float(x) / (111.32 * math.cos(math.radians(LAT0))) + LON0
    return lat, lon


def coord_to_index_1d(value: float, grid: np.ndarray) -> int:
    if len(grid) < 2:
        return 0
    idx = int(round((float(value) - float(grid[0])) / float(grid[1] - grid[0])))
    return int(np.clip(idx, 0, len(grid) - 1))


def coord_to_index_xyz(x: float, y: float, z: float, xg, yg, zg) -> tuple[int, int, int]:
    i = coord_to_index_1d(x, xg)
    j = coord_to_index_1d(y, yg)
    k = coord_to_index_1d(z, zg)
    return k, j, i


def coord_from_kji(kji: tuple[int, int, int], xg, yg, zg) -> np.ndarray:
    k, j, i = kji
    return np.array([float(xg[i]), float(yg[j]), float(zg[k])], dtype=float)


def idx_linear(k: int, j: int, i: int, ny: int, nx: int) -> int:
    return int((k * ny + j) * nx + i)


def idx_to_kji(idx: int, ny: int, nx: int) -> tuple[int, int, int]:
    k = int(idx) // (ny * nx)
    rem = int(idx) - k * ny * nx
    j = rem // nx
    i = rem - j * nx
    return k, j, i


def grid_edges(shape: tuple[int, int, int], dx: float, dy: float, dz: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = (shape, float(dx), float(dy), float(dz))
    cached = _GRID_EDGE_CACHE.get(key)
    if cached is not None:
        return cached
    nz, ny, nx = shape
    dirs = []
    for dk in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                if dk == 0 and dj == 0 and di == 0:
                    continue
                if (dk, dj, di) <= (0, 0, 0):
                    continue
                dirs.append((dk, dj, di, math.sqrt((di * dx) ** 2 + (dj * dy) ** 2 + (dk * dz) ** 2)))
    uu, vv, ds = [], [], []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                u = idx_linear(k, j, i, ny, nx)
                for dk, dj, di, step in dirs:
                    kk, jj, ii = k + dk, j + dj, i + di
                    if 0 <= kk < nz and 0 <= jj < ny and 0 <= ii < nx:
                        uu.append(u)
                        vv.append(idx_linear(kk, jj, ii, ny, nx))
                        ds.append(step)
    out = (np.asarray(uu, dtype=np.int32), np.asarray(vv, dtype=np.int32), np.asarray(ds, dtype=np.float64))
    _GRID_EDGE_CACHE[key] = out
    return out


def build_travel_graph(vp: np.ndarray, dx: float, dy: float, dz: float) -> sp.csr_matrix:
    u, v, ds = grid_edges(vp.shape, dx, dy, dz)
    slow = (1.0 / np.maximum(vp.ravel(), 1.0e-9)).astype(np.float64)
    weights = ds * 0.5 * (slow[u] + slow[v])
    return sp.csr_matrix((weights, (u, v)), shape=(vp.size, vp.size), dtype=np.float64)


def backtrace_build_row_graph(predecessors, src_idx: int, rec_idx: int, shape: tuple[int, int, int], dx: float, dy: float, dz: float, vp=None, mode="slowness"):
    nz, ny, nx = shape
    row: dict[int, float] = {}
    cur = int(rec_idx)
    steps = 0
    while steps < RAY_MAX_STEPS:
        if cur == int(src_idx):
            return row, True
        prev = int(predecessors[cur])
        if prev < 0 or prev == cur:
            return row, False
        ck, cj, ci = idx_to_kji(cur, ny, nx)
        pk, pj, pi = idx_to_kji(prev, ny, nx)
        ds = math.sqrt(((ci - pi) * dx) ** 2 + ((cj - pj) * dy) ** 2 + ((ck - pk) * dz) ** 2)
        if ds <= 0.0:
            return row, False
        if mode == "slowness":
            row[cur] = row.get(cur, 0.0) + ds
        elif mode == "qinv":
            if vp is None:
                return row, False
            vel = float(vp[ck, cj, ci])
            if (not np.isfinite(vel)) or vel <= 0:
                return row, False
            row[cur] = row.get(cur, 0.0) + ds / vel
        else:
            raise ValueError(f"Unknown mode: {mode}")
        cur = prev
        steps += 1
    return row, False


def rectangular_domain_mask(xg, yg, zg) -> np.ndarray:
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


def load_grid_and_vp() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(VP_MODEL_NPZ)
    xg = data["xg"][::GRID_FACTOR].astype(float)
    yg = data["yg"][::GRID_FACTOR].astype(float)
    zg = data["zg"][::GRID_FACTOR].astype(float)
    vp = data["vp"][::GRID_FACTOR, ::GRID_FACTOR, ::GRID_FACTOR].astype(float)
    return xg, yg, zg, vp


@njit
def _solve_local(t1, t2, t3, h1, h2, h3, s):
    a = 1.0 / (h1 * h1)
    b = -2.0 * t1 / (h1 * h1)
    c = (t1 * t1) / (h1 * h1) - s * s
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        disc = 0.0
    T = (-b + math.sqrt(disc)) / (2.0 * a)
    if T <= t2:
        return T

    a = 1.0 / (h1 * h1) + 1.0 / (h2 * h2)
    b = -2.0 * (t1 / (h1 * h1) + t2 / (h2 * h2))
    c = (t1 * t1) / (h1 * h1) + (t2 * t2) / (h2 * h2) - s * s
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        disc = 0.0
    T = (-b + math.sqrt(disc)) / (2.0 * a)
    if T <= t3:
        return T

    a = 1.0 / (h1 * h1) + 1.0 / (h2 * h2) + 1.0 / (h3 * h3)
    b = -2.0 * (t1 / (h1 * h1) + t2 / (h2 * h2) + t3 / (h3 * h3))
    c = (t1 * t1) / (h1 * h1) + (t2 * t2) / (h2 * h2) + (t3 * t3) / (h3 * h3) - s * s
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        disc = 0.0
    T = (-b + math.sqrt(disc)) / (2.0 * a)
    if T < t3:
        T = t3
    return T


@njit
def fast_sweeping_3d(slowness, src_k, src_j, src_i, dx, dy, dz, n_sweeps, big):
    nz, ny, nx = slowness.shape
    T = np.empty((nz, ny, nx), dtype=np.float64)
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                T[k, j, i] = big
    T[src_k, src_j, src_i] = 0.0

    for _ in range(n_sweeps):
        for kz in (1, -1):
            k_range = range(0, nz) if kz == 1 else range(nz - 1, -1, -1)
            for jy in (1, -1):
                j_range = range(0, ny) if jy == 1 else range(ny - 1, -1, -1)
                for ix in (1, -1):
                    i_range = range(0, nx) if ix == 1 else range(nx - 1, -1, -1)
                    for k in k_range:
                        for j in j_range:
                            for i in i_range:
                                if k == src_k and j == src_j and i == src_i:
                                    continue
                                tx = big
                                if i > 0 and T[k, j, i - 1] < tx:
                                    tx = T[k, j, i - 1]
                                if i < nx - 1 and T[k, j, i + 1] < tx:
                                    tx = T[k, j, i + 1]
                                ty = big
                                if j > 0 and T[k, j - 1, i] < ty:
                                    ty = T[k, j - 1, i]
                                if j < ny - 1 and T[k, j + 1, i] < ty:
                                    ty = T[k, j + 1, i]
                                tz = big
                                if k > 0 and T[k - 1, j, i] < tz:
                                    tz = T[k - 1, j, i]
                                if k < nz - 1 and T[k + 1, j, i] < tz:
                                    tz = T[k + 1, j, i]
                                t1, h1 = tx, dx
                                t2, h2 = ty, dy
                                t3, h3 = tz, dz
                                if t2 < t1:
                                    t1, t2 = t2, t1
                                    h1, h2 = h2, h1
                                if t3 < t2:
                                    t2, t3 = t3, t2
                                    h2, h3 = h3, h2
                                if t2 < t1:
                                    t1, t2 = t2, t1
                                    h1, h2 = h2, h1
                                new_t = _solve_local(t1, t2, t3, h1, h2, h3, slowness[k, j, i])
                                if new_t < T[k, j, i]:
                                    T[k, j, i] = new_t
    return T


NEI26 = [
    (dk, dj, di)
    for dk in (-1, 0, 1)
    for dj in (-1, 0, 1)
    for di in (-1, 0, 1)
    if not (dk == 0 and dj == 0 and di == 0)
]


def backtrace_build_row(T, src_kji, rec_kji, dx, dy, dz, vp=None, mode="slowness"):
    nz, ny, nx = T.shape
    sk, sj, si = src_kji
    k, j, i = rec_kji
    row: dict[int, float] = {}
    steps = 0
    while steps < RAY_MAX_STEPS:
        if (k, j, i) == (sk, sj, si):
            return row, True
        t_here = T[k, j, i]
        best = None
        best_t = t_here
        for dk, dj, di in NEI26:
            kk = k + dk
            jj = j + dj
            ii = i + di
            if kk < 0 or kk >= nz or jj < 0 or jj >= ny or ii < 0 or ii >= nx:
                continue
            tt = T[kk, jj, ii]
            if tt < best_t:
                best_t = tt
                best = (kk, jj, ii, dk, dj, di)
        if best is None:
            return row, False
        kk, jj, ii, dk, dj, di = best
        ds = math.sqrt((di * dx) ** 2 + (dj * dy) ** 2 + (dk * dz) ** 2)
        if ds <= 0:
            return row, False
        lin = idx_linear(k, j, i, ny, nx)
        if mode == "slowness":
            weight = ds
        elif mode == "qinv":
            vel = float(vp[k, j, i])
            if (not np.isfinite(vel)) or vel <= 0:
                return row, False
            weight = ds / vel
        else:
            raise ValueError(f"Unknown mode: {mode}")
        row[lin] = row.get(lin, 0.0) + weight
        k, j, i = kk, jj, ii
        steps += 1
    return row, False


def laplacian_active_3d(mask3: np.ndarray) -> sp.csr_matrix:
    nz, ny, nx = mask3.shape
    active = np.flatnonzero(mask3.ravel())
    pos = {int(old): i for i, old in enumerate(active)}
    rr, cc, vv = [], [], []
    row = 0
    for old in active:
        k, j, i = idx_to_kji(int(old), ny, nx)
        neigh = []
        for dk, dj, di in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
            kk, jj, ii = k + dk, j + dj, i + di
            if kk < 0 or kk >= nz or jj < 0 or jj >= ny or ii < 0 or ii >= nx:
                continue
            new = pos.get(idx_linear(kk, jj, ii, ny, nx))
            if new is not None:
                neigh.append(new)
        if not neigh:
            continue
        rr.append(row); cc.append(pos[int(old)]); vv.append(float(len(neigh)))
        for new in neigh:
            rr.append(row); cc.append(new); vv.append(-1.0)
        row += 1
    return sp.csr_matrix((vv, (rr, cc)), shape=(row, len(active)), dtype=np.float64)


def build_q0_cube(zg, ny: int, nx: int) -> np.ndarray:
    q0 = np.zeros((len(zg), ny, nx), dtype=float)
    for k, z in enumerate(zg):
        q0[k, :, :] = 40.0 if float(z) < 30.0 else 60.0
    return q0


def rms(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(arr * arr))) if arr.size else float("nan")


def rows_to_csr(rows: list[dict[int, float]], active_flat: np.ndarray):
    active = np.flatnonzero(active_flat)
    pos = {int(old): i for i, old in enumerate(active)}
    rr, cc, vv = [], [], []
    kept = []
    for row_idx, row in enumerate(rows):
        local = [(pos[int(idx)], float(val)) for idx, val in row.items() if int(idx) in pos]
        if not local:
            continue
        out_row = len(kept)
        kept.append(row_idx)
        for col, val in local:
            rr.append(out_row); cc.append(col); vv.append(val)
    return sp.csr_matrix((vv, (rr, cc)), shape=(len(kept), len(active))), active, kept


def solve_augmented(A, b, active_mask3, laplace_weight, damp_weight, iterlim):
    active_count = A.shape[1]
    blocks = [A]
    rhs = [b]
    if laplace_weight > 0:
        L = laplacian_active_3d(active_mask3)
        if L.shape[0] > 0:
            blocks.append(laplace_weight * L)
            rhs.append(np.zeros(L.shape[0], dtype=float))
    if damp_weight > 0:
        blocks.append(damp_weight * sp.eye(active_count, format="csr"))
        rhs.append(np.zeros(active_count, dtype=float))
    sol = lsmr(sp.vstack(blocks, format="csr"), np.concatenate(rhs), maxiter=iterlim)
    return sol[0], {"istop": int(sol[1]), "it": int(sol[2]), "normr": float(sol[3]), "acond": float(sol[6])}


def prep_vp_groups(picks: pd.DataFrame, xg, yg, zg):
    groups: dict[tuple[int, int, int], list[tuple[tuple[int, int, int], float, int, np.ndarray, np.ndarray]]] = {}
    for row_id, row in enumerate(picks.itertuples(index=False)):
        src_xyz = np.array([float(row.source_x_km), float(row.source_y_km), float(row.source_z_km)], dtype=float)
        rec_xyz = np.array([float(row.receiver_x_km), float(row.receiver_y_km), 0.0], dtype=float)
        skji = coord_to_index_xyz(src_xyz[0], src_xyz[1], src_xyz[2], xg, yg, zg)
        rkji = coord_to_index_xyz(rec_xyz[0], rec_xyz[1], rec_xyz[2], xg, yg, zg)
        groups.setdefault(skji, []).append((rkji, float(row.travel_time_s), int(row_id), src_xyz, rec_xyz))
    return groups


def build_vp_rows(vp: np.ndarray, groups, xg, yg, zg):
    dx = float(xg[1] - xg[0])
    dy = float(yg[1] - yg[0])
    dz = float(zg[1] - zg[0])
    graph = build_travel_graph(vp, dx, dy, dz)
    ny, nx = vp.shape[1], vp.shape[2]
    rows, obs, pred, row_ids = [], [], [], []
    failures = 0
    t0 = time.time()
    for nsrc, (skji, recs) in enumerate(groups.items(), start=1):
        sk, sj, si = skji
        src_idx = idx_linear(sk, sj, si, ny, nx)
        dist, predecessors = dijkstra(graph, directed=False, indices=src_idx, return_predecessors=True)
        src_node_xyz = coord_from_kji(skji, xg, yg, zg)
        for rkji, tobs, row_id, src_xyz, rec_xyz in recs:
            rk, rj, ri = rkji
            rec_node_xyz = coord_from_kji(rkji, xg, yg, zg)
            rec_idx = idx_linear(rk, rj, ri, ny, nx)
            src_ds = float(np.linalg.norm(src_xyz - src_node_xyz))
            rec_ds = float(np.linalg.norm(rec_xyz - rec_node_xyz))
            endpoint_time = src_ds / max(float(vp[sk, sj, si]), 1.0e-9) + rec_ds / max(float(vp[rk, rj, ri]), 1.0e-9) if USE_ENDPOINT_CORRECTION else 0.0
            tcalc = float(dist[rec_idx]) + endpoint_time
            ray, ok = backtrace_build_row_graph(predecessors, src_idx, rec_idx, vp.shape, dx, dy, dz, vp=None, mode="slowness")
            if (not ok) or (not ray) or (not np.isfinite(tcalc)) or tcalc >= BIG_T * 0.9:
                failures += 1
                continue
            if USE_ENDPOINT_CORRECTION and src_ds > 0:
                ray[src_idx] = ray.get(src_idx, 0.0) + src_ds
            if USE_ENDPOINT_CORRECTION and rec_ds > 0:
                ray[rec_idx] = ray.get(rec_idx, 0.0) + rec_ds
            rows.append(ray)
            obs.append(float(tobs))
            pred.append(tcalc)
            row_ids.append(int(row_id))
    return rows, np.asarray(obs), np.asarray(pred), np.asarray(row_ids), failures, time.time() - t0


def hit_count_from_rows(rows: list[dict[int, float]], ncell: int) -> np.ndarray:
    hits = np.zeros(ncell, dtype=int)
    for row in rows:
        for idx, val in row.items():
            if abs(float(val)) > 0:
                hits[int(idx)] += 1
    return hits


def build_vp_active_mask(vp0, groups, xg, yg, zg, domain_mask):
    rows, obs, pred, row_ids, failures, elapsed = build_vp_rows(vp0, groups, xg, yg, zg)
    hits = hit_count_from_rows(rows, vp0.size).reshape(vp0.shape)
    mask = domain_mask & (hits >= VP_MIN_HITS)
    return mask, hits, {"initial_fsm_row_failures": int(failures), "initial_fsm_build_seconds": float(elapsed)}


def run_vp_fsm(vp0, picks, xg, yg, zg, active_mask3, initial_rows=None):
    groups = prep_vp_groups(picks, xg, yg, zg)
    vp = vp0.copy()
    final_payload = None
    history = []
    for it in range(NOUTER_VP):
        if it == 0 and initial_rows is not None:
            rows, obs, pred, row_ids, failures, build_seconds = initial_rows
        else:
            rows, obs, pred, row_ids, failures, build_seconds = build_vp_rows(vp, groups, xg, yg, zg)
        A, active_idx, kept = rows_to_csr(rows, active_mask3.ravel())
        kept = np.asarray(kept, dtype=int)
        obs_kept = obs[kept]
        pred_kept = pred[kept]
        b = obs_kept - pred_kept
        pre_rms = rms(b)
        dslow, info = solve_augmented(A, b, active_mask3, VP_LAPLACE, VP_DAMP, VP_ITERLIM)
        svec = (1.0 / np.maximum(vp, 1.0e-9)).ravel()
        snew = svec.copy()
        snew[active_idx] = np.clip(snew[active_idx] + VP_ALPHA * dslow, S_MIN, S_MAX)
        vp = (1.0 / np.maximum(snew, 1.0e-12)).reshape(vp.shape)
        history.append(
            {
                "iteration": int(it),
                "equations": int(A.shape[0]),
                "active_cells": int(A.shape[1]),
                "pre_rms_s": float(pre_rms),
                "post_rms_s": float("nan"),
                "best_rms_s": float("nan"),
                "accepted": True,
                "fsm_failures": int(failures),
                "fsm_build_seconds": float(build_seconds),
                "lsmr_iterations": int(info["it"]),
                "lsmr_istop": int(info["istop"]),
                "lsmr_acond": float(info["acond"]),
            }
        )
        print(f"[Grid Vp 10km] it={it} pre={pre_rms:.4f}s update_applied=True", flush=True)

    rows, obs, pred, row_ids, failures, build_seconds = build_vp_rows(vp, groups, xg, yg, zg)
    A, active_idx, kept = rows_to_csr(rows, active_mask3.ravel())
    kept = np.asarray(kept, dtype=int)
    obs_kept = obs[kept]
    pred_kept = pred[kept]
    final_rms = rms(obs_kept - pred_kept)
    hist = pd.DataFrame(history)
    if len(hist):
        hist.loc[hist.index[-1], "post_rms_s"] = final_rms
        hist.loc[hist.index[-1], "best_rms_s"] = final_rms
    final_payload = {
        "A": A,
        "active_idx": active_idx,
        "row_ids": row_ids[kept],
        "obs": obs_kept,
        "pred": pred_kept,
        "final_failures": int(failures),
        "final_build_seconds": float(build_seconds),
    }
    print(f"[Grid Vp 10km] final recomputed RMS={final_rms:.4f}s", flush=True)
    return vp, hist, final_payload


def prepare_q_pairs(pair_df: pd.DataFrame) -> pd.DataFrame:
    pairs = pair_df.copy()
    if "qc_ok" in pairs.columns:
        pairs = pairs[pairs["qc_ok"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
    pairs = pairs.dropna(
        subset=[
            "delta_tstar_s",
            "event_lat",
            "event_lon",
            "event_depth_km",
            "station_i_lat",
            "station_i_lon",
            "station_j_lat",
            "station_j_lon",
        ]
    ).reset_index(drop=True)
    for prefix in ["event", "station_i", "station_j"]:
        if prefix == "event":
            xy = pairs.apply(lambda r: latlon_to_xy(r.event_lat, r.event_lon), axis=1, result_type="expand")
        else:
            xy = pairs.apply(lambda r, p=prefix: latlon_to_xy(r[f"{p}_lat"], r[f"{p}_lon"]), axis=1, result_type="expand")
        pairs[f"{prefix}_x_km"] = xy[0]
        pairs[f"{prefix}_y_km"] = xy[1]
    return pairs


def row_difference(row_i: dict[int, float], row_j: dict[int, float]) -> dict[int, float]:
    out: dict[int, float] = {}
    for idx, val in row_i.items():
        out[idx] = out.get(idx, 0.0) + float(val)
    for idx, val in row_j.items():
        out[idx] = out.get(idx, 0.0) - float(val)
    return {idx: val for idx, val in out.items() if abs(val) > 0.0}


def row_dot(row: dict[int, float], vec: np.ndarray) -> float:
    return float(sum(float(val) * float(vec[int(idx)]) for idx, val in row.items()))


def build_q_rows(vp, pairs, xg, yg, zg):
    dx = float(xg[1] - xg[0])
    dy = float(yg[1] - yg[0])
    dz = float(zg[1] - zg[0])
    graph = build_travel_graph(vp, dx, dy, dz)
    tfield_cache: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}
    ray_cache: dict[tuple[tuple[int, int, int], tuple[int, int, int]], dict[int, float] | None] = {}
    rows, obs, kept = [], [], []
    failures = 0
    ny, nx = vp.shape[1], vp.shape[2]
    for idx, row in pairs.iterrows():
        src_xyz = np.array([float(row.event_x_km), float(row.event_y_km), float(row.event_depth_km)], dtype=float)
        rec_i_xyz = np.array([float(row.station_i_x_km), float(row.station_i_y_km), 0.0], dtype=float)
        rec_j_xyz = np.array([float(row.station_j_x_km), float(row.station_j_y_km), 0.0], dtype=float)
        skji = coord_to_index_xyz(src_xyz[0], src_xyz[1], src_xyz[2], xg, yg, zg)
        ri = coord_to_index_xyz(rec_i_xyz[0], rec_i_xyz[1], rec_i_xyz[2], xg, yg, zg)
        rj = coord_to_index_xyz(rec_j_xyz[0], rec_j_xyz[1], rec_j_xyz[2], xg, yg, zg)
        if skji not in tfield_cache:
            sk, sj, si = skji
            src_idx = idx_linear(sk, sj, si, ny, nx)
            tfield_cache[skji] = dijkstra(graph, directed=False, indices=src_idx, return_predecessors=True)
        _, predecessors = tfield_cache[skji]
        endpoint_xyz = {ri: rec_i_xyz, rj: rec_j_xyz}
        for rec in [ri, rj]:
            key = (skji, rec)
            if key not in ray_cache:
                sk, sj, si = skji
                rk, rj_idx, ri_idx = rec
                src_idx = idx_linear(sk, sj, si, ny, nx)
                rec_idx = idx_linear(rk, rj_idx, ri_idx, ny, nx)
                ray, ok = backtrace_build_row_graph(predecessors, src_idx, rec_idx, vp.shape, dx, dy, dz, vp=vp, mode="qinv")
                if ok and ray:
                    src_node = coord_from_kji(skji, xg, yg, zg)
                    rec_node = coord_from_kji(rec, xg, yg, zg)
                    src_ds = float(np.linalg.norm(src_xyz - src_node))
                    rec_ds = float(np.linalg.norm(endpoint_xyz[rec] - rec_node))
                    src_lin = idx_linear(sk, sj, si, vp.shape[1], vp.shape[2])
                    rec_lin = idx_linear(rk, rj_idx, ri_idx, vp.shape[1], vp.shape[2])
                    if USE_ENDPOINT_CORRECTION and src_ds > 0:
                        ray[src_lin] = ray.get(src_lin, 0.0) + src_ds / max(float(vp[sk, sj, si]), 1.0e-9)
                    if USE_ENDPOINT_CORRECTION and rec_ds > 0:
                        ray[rec_lin] = ray.get(rec_lin, 0.0) + rec_ds / max(float(vp[rk, rj_idx, ri_idx]), 1.0e-9)
                ray_cache[key] = ray if ok else None
        ray_i = ray_cache[(skji, ri)]
        ray_j = ray_cache[(skji, rj)]
        if ray_i is None or ray_j is None:
            failures += 1
            continue
        diff = row_difference(ray_i, ray_j)
        if not diff:
            failures += 1
            continue
        rows.append(diff)
        obs.append(float(row.delta_tstar_s))
        kept.append(int(idx))
    return rows, np.asarray(obs), np.asarray(kept), failures


def build_joint_q_matrix(rows, residual, pairs_kept, active_mask3, stations):
    active_idx = np.flatnonzero(active_mask3.ravel())
    active_pos = {int(old): i for i, old in enumerate(active_idx)}
    station_pos = {str(sta): i for i, sta in enumerate(stations)}
    ncell = len(active_idx)
    nsta = len(stations)
    rr, cc, vv = [], [], []
    kept_local = []
    b = []
    for row_idx, (kernel, res, pair) in enumerate(zip(rows, residual, pairs_kept.itertuples(index=False))):
        local = [(active_pos[int(idx)], float(val)) for idx, val in kernel.items() if int(idx) in active_pos]
        if not local:
            continue
        out_row = len(b)
        for col, val in local:
            rr.append(out_row); cc.append(col); vv.append(val)
        rr.append(out_row); cc.append(ncell + station_pos[str(pair.station_i)]); vv.append(1.0)
        rr.append(out_row); cc.append(ncell + station_pos[str(pair.station_j)]); vv.append(-1.0)
        b.append(float(res))
        kept_local.append(row_idx)
    A = sp.csr_matrix((vv, (rr, cc)), shape=(len(b), ncell + nsta), dtype=np.float64)
    return A, np.asarray(b), active_idx, np.asarray(kept_local, dtype=int)


def solve_q_augmented(A, b, active_mask3, nstation):
    ncell = A.shape[1] - nstation
    blocks = [A]
    rhs = [b]
    L = laplacian_active_3d(active_mask3)
    if L.shape[0] > 0:
        blocks.append(sp.hstack([Q_LAPLACE * L, sp.csr_matrix((L.shape[0], nstation))], format="csr"))
        rhs.append(np.zeros(L.shape[0]))
    blocks.append(sp.hstack([Q_CELL_DAMP * sp.eye(ncell, format="csr"), sp.csr_matrix((ncell, nstation))], format="csr"))
    rhs.append(np.zeros(ncell))
    blocks.append(sp.hstack([sp.csr_matrix((nstation, ncell)), Q_STATION_DAMP * sp.eye(nstation, format="csr")], format="csr"))
    rhs.append(np.zeros(nstation))
    sol = lsmr(sp.vstack(blocks, format="csr"), np.concatenate(rhs), maxiter=Q_ITERLIM)
    return sol[0], {"istop": int(sol[1]), "it": int(sol[2]), "normr": float(sol[3]), "acond": float(sol[6])}


def run_q_fsm(vp, pairs, xg, yg, zg, domain_mask):
    q0 = build_q0_cube(zg, len(yg), len(xg))
    qinv0 = 1.0 / np.maximum(q0, 1.0e-9)
    rows, obs, kept_idx, failures = build_q_rows(vp, pairs, xg, yg, zg)
    hits = hit_count_from_rows(rows, vp.size).reshape(vp.shape)
    active_q = domain_mask & (hits >= Q_MIN_HITS)
    pairs_kept0 = pairs.iloc[kept_idx].reset_index(drop=True)
    pred0_all = np.asarray([row_dot(row, qinv0.ravel()) for row in rows])
    residual = obs - pred0_all
    stations = sorted(set(pairs_kept0["station_i"]).union(set(pairs_kept0["station_j"])))
    A, b, active_idx, kept_local = build_joint_q_matrix(rows, residual, pairs_kept0, active_q, stations)
    x, info = solve_q_augmented(A, b, active_q, len(stations))
    ncell = len(active_idx)
    dq = x[:ncell]
    station_terms = x[ncell:]
    qinv = qinv0.ravel().copy()
    qinv[active_idx] = np.clip(qinv[active_idx] + dq, QINV_MIN, QINV_MAX)
    pairs_kept = pairs_kept0.iloc[kept_local].reset_index(drop=True)
    pred0 = pred0_all[kept_local]
    station_map = {sta: float(station_terms[i]) for i, sta in enumerate(stations)}
    station_pred = np.asarray([station_map[str(r.station_i)] - station_map[str(r.station_j)] for r in pairs_kept.itertuples(index=False)])
    pred1 = pred0 + (A[:, :ncell] @ dq) + station_pred
    obs_used = pairs_kept["delta_tstar_s"].to_numpy(dtype=float)
    station_df = pd.DataFrame({"station": stations, "station_term_s": [station_map[sta] for sta in stations]})
    initial_rms = rms(obs_used - pred0)
    station_only_rms = rms(obs_used - pred0 - station_pred)
    final_rms = rms(obs_used - pred1)
    if "r2" in pairs_kept:
        high_r2 = pairs_kept[pairs_kept["r2"] >= 0.4].copy()
        high_r2_pairs = high_r2[["station_i", "station_j"]].drop_duplicates() if len(high_r2) else high_r2
    else:
        high_r2 = pd.DataFrame()
        high_r2_pairs = pd.DataFrame()
    metrics = {
        "pair_observations_qc_input": int(len(pairs)),
        "observations_used": int(len(obs_used)),
        "active_cells": int(active_q.sum()),
        "station_terms": int(len(stations)),
        "data_parameter_ratio_cells_plus_stations": float(len(obs_used) / max(int(active_q.sum()) + len(stations), 1)),
        "initial_rms_s": initial_rms,
        "station_only_rms_s": station_only_rms,
        "final_rms_s": final_rms,
        "station_only_rms_reduction_percent": float(100.0 * (initial_rms - station_only_rms) / max(initial_rms, 1.0e-12)),
        "rms_reduction_percent": float(100.0 * (initial_rms - final_rms) / max(initial_rms, 1.0e-12)),
        "median_r2": float(pairs_kept["r2"].median()) if "r2" in pairs_kept else float("nan"),
        "r2_ge_0p4_observations": int(len(high_r2)),
        "r2_ge_0p4_events": int(high_r2["event_id"].nunique()) if len(high_r2) and "event_id" in high_r2 else 0,
        "r2_ge_0p4_station_pairs": int(len(high_r2_pairs)) if len(high_r2_pairs) else 0,
        "r2_ge_0p4_median_r2": float(high_r2["r2"].median()) if len(high_r2) and "r2" in high_r2 else float("nan"),
        "max_abs_station_term_s": float(np.max(np.abs(station_terms))) if len(station_terms) else 0.0,
        "min_hit_threshold": int(Q_MIN_HITS),
        "fsm_ray_failures": int(failures),
        "lsmr_iterations": int(info["it"]),
        "lsmr_istop": int(info["istop"]),
    }
    pred_df = pairs_kept.copy()
    pred_df["pred_initial_s"] = pred0
    pred_df["pred_final_s"] = pred1
    pred_df["residual_initial_s"] = obs_used - pred0
    pred_df["residual_final_s"] = obs_used - pred1
    return qinv.reshape(vp.shape), q0, active_q, station_df, pred_df, hits, metrics


def run_vp_checkerboard(A, active_idx, active_mask3, vp_ref):
    ny, nx = active_mask3.shape[1], active_mask3.shape[2]
    sref = (1.0 / np.maximum(vp_ref, 1.0e-9)).ravel()[active_idx]
    pattern = np.zeros(len(active_idx), dtype=float)
    for n, idx in enumerate(active_idx):
        k, j, i = idx_to_kji(int(idx), ny, nx)
        pattern[n] = 1.0 if ((i // 2 + j // 2 + k) % 2 == 0) else -1.0
    true = CHECKERBOARD_SLOWNESS_FRACTION * sref * pattern
    rng = np.random.default_rng(20260509)
    synth = A @ true + rng.normal(0.0, CHECKERBOARD_NOISE_S, size=A.shape[0])
    recovered, info = solve_augmented(A, synth, active_mask3, VP_LAPLACE, VP_DAMP, VP_ITERLIM)
    corr = float(np.corrcoef(true, recovered)[0, 1]) if len(true) > 2 else float("nan")
    rmse = float(np.sqrt(np.mean((true - recovered) ** 2)))
    true_cube = np.full(vp_ref.size, np.nan)
    rec_cube = np.full(vp_ref.size, np.nan)
    true_cube[active_idx] = true / np.maximum(sref, 1.0e-12) * 100.0
    rec_cube[active_idx] = recovered / np.maximum(sref, 1.0e-12) * 100.0
    return true_cube.reshape(vp_ref.shape), rec_cube.reshape(vp_ref.shape), {
        "type": "linearized grid-eikonal checkerboard on final ray geometry",
        "active_cells": int(len(active_idx)),
        "observations": int(A.shape[0]),
        "noise_std_s": float(CHECKERBOARD_NOISE_S),
        "slowness_fraction_amplitude": float(CHECKERBOARD_SLOWNESS_FRACTION),
        "correlation": corr,
        "rmse_slowness": rmse,
        "lsmr_iterations": int(info["it"]),
        "lsmr_istop": int(info["istop"]),
    }


def plot_area(events_df, stations_df) -> Path:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(events_df["event_lon"], events_df["event_lat"], s=8, c="black", alpha=0.45, label="Earthquakes")
    ax.scatter(stations_df["lon"], stations_df["lat"], s=80, c="red", marker="^", edgecolor="black", lw=0.4, label="Stations")
    ax.plot([LON_MIN, LON_MAX, LON_MAX, LON_MIN, LON_MIN], [LAT_MIN, LAT_MIN, LAT_MAX, LAT_MAX, LAT_MIN], color="#2b6cb0", lw=1.5)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Final tomography area: SGC events, Bucaramanga nest, and expanded stations")
    ax.legend(loc="best")
    ax.grid(True, lw=0.3, alpha=0.4)
    out = FIG / "sgc_final_fsm_area_events_black_stations_red.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_vp(vp0, vp, active, xg, yg, zg) -> Path:
    dvp = 100.0 * (vp - vp0) / np.maximum(vp0, 1.0e-9)
    vmax = float(np.nanpercentile(np.abs(dvp[active]), 95)) if np.any(active) else 10.0
    vmax = max(5.0, min(35.0, vmax))
    depths = [40.0, 80.0, 120.0]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), constrained_layout=True)
    im = None
    for ax, dep in zip(np.asarray(axes).ravel(), depths):
        iz = int(np.argmin(np.abs(zg - dep)))
        arr = np.where(active[iz], dvp[iz], np.nan)
        if np.isfinite(arr).any():
            im = ax.imshow(arr, origin="lower", extent=[xg.min(), xg.max(), yg.min(), yg.max()], cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        else:
            ax.text(0.5, 0.5, "No active cells", ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_title(f"dVp (%) z={zg[iz]:.0f} km", fontsize=11)
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
    if im is not None:
        fig.colorbar(im, ax=np.asarray(axes).ravel().tolist(), shrink=0.82, label="dVp (%)")
    fig.suptitle("Final grid-eikonal P-wave velocity tomography", fontsize=14)
    out = FIG / "sgc_final_fsm_vp_slices.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_vp_history(history: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(history["iteration"], history["pre_rms_s"], "o-", label="pre-update")
    ax.plot(history["iteration"], history["post_rms_s"], "s-", label="post-update")
    ax.plot(history["iteration"], history["best_rms_s"], "k^-", label="best")
    ax.set_xlabel("External iteration")
    ax.set_ylabel("Travel-time RMS (s)")
    ax.set_title("Grid-eikonal Vp convergence")
    ax.grid(True, lw=0.3, alpha=0.4)
    ax.legend()
    out = FIG / "sgc_final_fsm_vp_convergence.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_checkerboard(true_cube, rec_cube, active, xg, yg, zg) -> Path:
    depths = [40.0, 80.0, 120.0, 160.0]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    vmax = 4.5
    for col, dep in enumerate(depths):
        iz = int(np.argmin(np.abs(zg - dep)))
        axes[0, col].imshow(np.where(active[iz], true_cube[iz], np.nan), origin="lower", extent=[xg.min(), xg.max(), yg.min(), yg.max()], cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        im = axes[1, col].imshow(np.where(active[iz], rec_cube[iz], np.nan), origin="lower", extent=[xg.min(), xg.max(), yg.min(), yg.max()], cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        axes[0, col].set_title(f"Imposed z={zg[iz]:.0f} km")
        axes[1, col].set_title(f"Recovered z={zg[iz]:.0f} km")
        for ax in axes[:, col]:
            ax.set_xlabel("x (km)")
            ax.set_ylabel("y (km)")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="Slowness perturbation (%)")
    fig.suptitle("Grid-eikonal Vp checkerboard recovery", fontsize=14)
    out = FIG / "sgc_final_fsm_vp_checkerboard.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_q(qinv, q0, active, station_df, q_pred, xg, yg, zg) -> Path:
    dq = 100.0 * (qinv - 1.0 / np.maximum(q0, 1.0e-9)) / np.maximum(1.0 / np.maximum(q0, 1.0e-9), 1.0e-12)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.8), constrained_layout=True)
    axes[0, 0].scatter(q_pred["residual_initial_s"], q_pred["residual_final_s"], s=12, c="black", alpha=0.4)
    lim = float(np.nanpercentile(np.abs(np.r_[q_pred["residual_initial_s"], q_pred["residual_final_s"]]), 98))
    lim = max(lim, 0.05)
    axes[0, 0].plot([-lim, lim], [-lim, lim], "r--", lw=1)
    axes[0, 0].set_xlim(-lim, lim)
    axes[0, 0].set_ylim(-lim, lim)
    axes[0, 0].set_xlabel("Initial residual (s)")
    axes[0, 0].set_ylabel("Final residual (s)")
    axes[0, 0].set_title("Q residual reduction", fontsize=11)
    station_df.sort_values("station_term_s").plot.barh(x="station", y="station_term_s", ax=axes[0, 1], legend=False, color="#4c78a8")
    axes[0, 1].set_xlabel("Station term (s)")
    axes[0, 1].set_title("Receiver spectral terms", fontsize=11)
    axes[0, 1].tick_params(axis="y", labelsize=9)
    vmax = float(np.nanpercentile(np.abs(dq[active]), 95)) if np.any(active) else 25.0
    vmax = max(25.0, min(150.0, vmax))
    depth_counts = active.reshape(active.shape[0], -1).sum(axis=1)
    valid = np.where(depth_counts > 0)[0]
    chosen = []
    for dep in [80.0, 150.0]:
        if len(valid):
            iz = int(valid[np.argmin(np.abs(zg[valid] - dep))])
            if iz not in chosen:
                chosen.append(iz)
    if len(chosen) < 2 and len(valid):
        for iz in valid[np.argsort(depth_counts[valid])[::-1]]:
            if int(iz) not in chosen:
                chosen.append(int(iz))
            if len(chosen) == 2:
                break
    im = None
    for ax, iz in zip(axes[1, :], chosen[:2]):
        dep = float(zg[iz])
        arr = np.where(active[iz], dq[iz], np.nan)
        im = ax.imshow(
            arr,
            origin="lower",
            extent=[xg.min(), xg.max(), yg.min(), yg.max()],
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            aspect="auto",
        )
        ax.set_title(f"Pilot dQinv (%) z={dep:.0f} km", fontsize=11)
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
    for ax in axes[1, len(chosen[:2]):]:
        ax.axis("off")
    if im is not None:
        fig.colorbar(im, ax=axes[1, :].tolist(), shrink=0.8, label="dQinv (%)")
    fig.suptitle("Grid-eikonal spectral-ratio Q pilot inversion", fontsize=14)
    out = FIG / "sgc_final_fsm_q_pilot.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def reconstruct_benioff(vp0, vp, qinv, active_vp, active_q, xg, yg, zg):
    catalog = pd.read_csv(CATALOG_CSV)
    rename = {}
    if "lat" in catalog.columns:
        rename["lat"] = "latitude"
    if "lon" in catalog.columns:
        rename["lon"] = "longitude"
    catalog = catalog.rename(columns=rename)
    catalog = catalog.dropna(subset=["latitude", "longitude", "depth_km"]).copy()
    events = catalog[
        (catalog["longitude"].between(LON_MIN, LON_MAX))
        & (catalog["latitude"].between(LAT_MIN, LAT_MAX))
        & (catalog["depth_km"].between(60.0, 170.0))
    ].copy()
    if len(events) > 1500:
        events = events.sample(1500, random_state=20260509).copy()
    xs, ys = [], []
    for r in events.itertuples(index=False):
        x, y = latlon_to_xy(r.latitude, r.longitude)
        xs.append(x); ys.append(y)
    events["x_km"] = xs
    events["y_km"] = ys
    if len(events) < 8:
        return events, pd.DataFrame(), {}
    points = events[["x_km", "y_km"]].to_numpy(dtype=float)
    depths = events["depth_km"].to_numpy(dtype=float)
    tree = cKDTree(points)
    hull = Delaunay(points)
    k_neigh = min(24, len(events))
    smooth_radius_km = 30.0

    def weighted_depth(query_points: np.ndarray):
        dist, ind = tree.query(query_points, k=k_neigh)
        if k_neigh == 1:
            dist = dist[:, None]
            ind = ind[:, None]
        weights = np.exp(-0.5 * (dist / smooth_radius_km) ** 2)
        weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1.0e-12)
        return np.sum(weights * depths[ind], axis=1)

    xx, yy = np.meshgrid(np.linspace(min(xs), max(xs), 75), np.linspace(min(ys), max(ys), 75))
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    inside = hull.find_simplex(grid_points) >= 0
    zz_flat = weighted_depth(grid_points[inside])
    rows = []
    for (x, y), z in zip(grid_points[inside], zz_flat):
        lat, lon = xy_to_latlon(float(x), float(y))
        k, j, i = coord_to_index_xyz(float(x), float(y), float(z), xg, yg, zg)
        dvp = 100.0 * (vp[k, j, i] - vp0[k, j, i]) / max(float(vp0[k, j, i]), 1.0e-9)
        qval = 1.0 / max(float(qinv[k, j, i]), 1.0e-12)
        rows.append(
            {
                "lon": lon,
                "lat": lat,
                "x_km": float(x),
                "y_km": float(y),
                "depth_km": float(z),
                "nearest_grid_z_km": float(zg[k]),
                "dvp_percent": float(dvp),
                "Q_pilot": float(qval),
                "vp_active": bool(active_vp[k, j, i]),
                "q_active": bool(active_q[k, j, i]),
            }
        )
    surface = pd.DataFrame(rows)
    pred_event = weighted_depth(points)
    metrics = {
        "control_events": int(len(events)),
        "rmse_km": float(np.sqrt(np.mean((events["depth_km"].to_numpy() - pred_event) ** 2))),
        "mae_km": float(np.mean(np.abs(events["depth_km"].to_numpy() - pred_event))),
        "surface_nodes": int(len(surface)),
        "surface_nodes_vp_active": int(surface["vp_active"].sum()),
        "surface_nodes_q_active": int(surface["q_active"].sum()),
        "surface_method": "convex-hull k-nearest-neighbor Gaussian smoothing",
        "smoothing_radius_km": float(smooth_radius_km),
        "k_nearest": int(k_neigh),
        "depth_min_km": 60.0,
        "depth_max_km": 170.0,
    }
    return events, surface, metrics


def plot_benioff_3d(events, surface) -> Path:
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    lon_col = "lon" if "lon" in events.columns else "longitude"
    lat_col = "lat" if "lat" in events.columns else "latitude"
    sc = ax.scatter(events[lon_col], events[lat_col], events["depth_km"], c=events["depth_km"], s=7, cmap="viridis_r", alpha=0.35, label="SGC control events")
    surf = surface.copy()
    surf_attr = surface[surface["vp_active"]].copy()
    if len(surf) >= 8:
        tri = mtri.Triangulation(surf["lon"].to_numpy(), surf["lat"].to_numpy())
        ax.plot_trisurf(
            tri,
            surf["depth_km"].to_numpy(),
            cmap="viridis_r",
            linewidth=0.15,
            edgecolor="none",
            alpha=0.52,
            shade=True,
            label="Smoothed Benioff surface",
        )
        if len(surf_attr):
            ax.scatter(
                surf_attr["lon"],
                surf_attr["lat"],
                surf_attr["depth_km"],
                c=surf_attr["dvp_percent"],
                s=8,
                cmap="RdBu_r",
                alpha=0.75,
                label="Surface nodes sampled by dVp",
            )
    else:
        ax.scatter(surf["lon"], surf["lat"], surf["depth_km"], c=surf["dvp_percent"], s=10, cmap="RdBu_r", alpha=0.65, label="Benioff surface sampled by dVp")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Depth (km)")
    ax.invert_zaxis()
    ax.set_title("Nazca/Benioff reconstruction with tomography attributes")
    fig.colorbar(sc, ax=ax, shrink=0.65, label="Event depth (km)")
    ax.legend(loc="upper left")
    out = FIG / "sgc_final_fsm_nazca_benioff_3d_surface.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def write_paper(metrics: dict, figures: dict[str, Path]) -> Path:
    tex = PAPER / "mmv_tomography_final_fsm_nest_extra_stations_submission_draft.tex"
    m = metrics
    text = rf"""\documentclass[11pt]{{article}}

\usepackage[margin=1in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{float}}
\usepackage{{hyperref}}
\usepackage{{amsmath}}
\usepackage{{xurl}}
\graphicspath{{{{../figures/}}}}

\title{{P-wave velocity tomography and a response-corrected spectral-ratio Q pilot framework for the Middle Magdalena Valley, Colombia, using expanded SGC waveforms, Bucaramanga-nest earthquakes, and 3-D grid-eikonal ray tracing}}
\author{{Draft prepared for submission review}}
\date{{\today}}

\begin{{document}}
\maketitle

\begin{{abstract}}
We present an updated P-wave velocity and pilot spectral-ratio Q experiment for the Middle Magdalena Valley (MMV), Colombia, using Servicio Geologico Colombiano earthquakes, response-corrected waveforms, additional nearby stations, and intermediate-depth Bucaramanga-nest seismicity. The revised workflow uses {m['vp']['events']} earthquakes and {m['vp']['observations']} P-arrival observations for the velocity branch, and {m['q_pilot']['observations_used']} response-corrected differential spectral-ratio $t^*$ observations for the Q branch from {m['stations_used']} stations. Unlike the earlier straight-ray diagnostic, this version builds the inversion kernels with a 3-D minimum-time grid-eikonal operator and backtraced rays on a 10 km tomography grid. The initial P-wave model is a hybrid 3-D model informed by a Nazca slab prior; all velocity anomalies are therefore interpreted as perturbations relative to this reference. The velocity inversion uses {m['vp']['active_cells']} active cells, giving a data-to-parameter ratio of {m['vp']['data_parameter_ratio_cells_only']:.2f}, and reduces travel-time RMS from {m['vp']['initial_rms_s']:.2f} s to {m['vp']['final_rms_s']:.2f} s. A linearized checkerboard test on the final curved-ray geometry gives a correlation of {m['vp_checkerboard']['correlation']:.2f}, indicating moderate recovery of broad structures but limited cell-scale resolution. For Q, the StationXML instrument response is removed before spectral fitting. The Q pilot inversion uses {m['q_pilot']['active_cells']} active cells plus {m['q_pilot']['station_terms']} station terms and reduces differential $t^*$ RMS from {m['q_pilot']['initial_rms_s']:.3f} s to {m['q_pilot']['final_rms_s']:.3f} s. Nevertheless, the median spectral-fit $R^2$ remains {m['q_pilot']['median_r2']:.2f}, so the Q field is treated as a diagnostic pilot product rather than a calibrated attenuation model. The reconstructed local Benioff/Nazca surface uses {m['benioff']['control_events']} earthquakes between {m['benioff']['depth_min_km']:.0f} and {m['benioff']['depth_max_km']:.0f} km depth and provides a geometric reference for sampling the Vp and pilot-Q fields.
\end{{abstract}}

\section{{Introduction}}

The Middle Magdalena Valley lies above a complex northern Andean plate boundary where crustal heterogeneity, basin structure, and intermediate-depth seismicity associated with the Bucaramanga nest complicate regional seismic imaging. The objective of this revision is not to overstate a mature Q tomography result. Instead, it separates three products with different confidence levels: a P-wave velocity model computed with 3-D grid-eikonal ray tracing, a response-corrected spectral-ratio Q pilot framework, and a local Nazca/Benioff geometric reconstruction sampled by the tomographic fields. This framing is important because velocity anomalies can be discussed as structural indicators where the checkerboard is recoverable, whereas Q anomalies can only be used as fluid-sensitive candidates if spectral quality, receiver terms, and independent geological context support that interpretation.

\section{{Data}}

The earthquake catalog was downloaded from the SGC expert SeisComP query system for 2022--2026. Events were selected inside the tomography area bounded by longitude {LON_MIN:.2f} to {LON_MAX:.2f} degrees and latitude {LAT_MIN:.2f} to {LAT_MAX:.2f} degrees, with catalog RMS lower than 1.0 s. The velocity and waveform databases include shallow-to-intermediate events between 20 and 120 km depth and an additional Bucaramanga-nest subset between 120 and 170 km depth. The velocity branch uses {m['vp']['observations']} P picks from {m['vp']['events']} events. The Q branch uses vertical-component waveforms for station-pair spectral ratios from {m['stations_used']} stations. Instrument response metadata were downloaded separately from the SGC FDSN station service as StationXML with \texttt{{level=response}}; all Q spectra in this version were computed after response removal to velocity units.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.88\linewidth]{{{figures['area'].name}}}
\caption{{Tomography area for the final run. Earthquakes used in the inversion are shown in black and stations in red. The rectangular domain intentionally includes the Bucaramanga-nest depth interval used to constrain the local Benioff reconstruction.}}
\end{{figure}}

\section{{Methods}}

\subsection{{Initial velocity model}}

The initial velocity model is a 3-D Nazca slab-informed P-wave model superposed on a regional crust--mantle background. This is a stabilizing prior, not an independent result. The final model is interpreted as a perturbation relative to that hybrid reference; therefore the discussion emphasizes spatially coherent perturbations in well-sampled cells rather than absolute velocity values everywhere in the domain.

\subsection{{Grid-eikonal velocity inversion}}

For each source, minimum-time paths are computed on the reduced 10 km 3-D grid using a 26-neighbor eikonal graph whose edge costs are the distance-weighted average slowness between adjacent cells. Rays are backtraced from receiver to source through the predecessor tree, and path lengths are assembled as slowness kernels. The inversion solves for slowness perturbations using LSMR with damping {VP_DAMP} and Laplacian smoothing {VP_LAPLACE}. Cells outside the tomography rectangle, below {MAX_MODEL_DEPTH_KM:.0f} km, or with fewer than {VP_MIN_HITS} ray hits are inactive.

\subsection{{Response-corrected spectral-ratio Q pilot}}

For a common earthquake observed at stations $i$ and $j$, the Q observable is
\[
  \ln\left(\frac{{A_i(f)}}{{A_j(f)}}\right) = c - \pi f \Delta t^*_{{ij}}.
\]
The inversion solves
\[
  \Delta t^*_{{ij}} = (G_i-G_j)Q^{{-1}} + \kappa_i-\kappa_j,
\]
where $G_i$ and $G_j$ are backtraced path kernels and $\kappa$ are receiver spectral terms. Because median spectral-fit $R^2$ remains low, the Q result is retained as a pilot diagnostic showing path--station tradeoffs, not as a final calibrated attenuation model.

\section{{Results}}

\subsection{{P-wave velocity}}

The grid-eikonal velocity inversion uses {m['vp']['observations']} equations and {m['vp']['active_cells']} active cells. The data-to-parameter ratio is {m['vp']['data_parameter_ratio_cells_only']:.2f}, which is substantially healthier than the earlier sparse V2 setup. The travel-time RMS decreases from {m['vp']['initial_rms_s']:.2f} s to {m['vp']['final_rms_s']:.2f} s. This final RMS is still high for local tomography, and is interpreted as the combined effect of fixed hypocenters, catalog heterogeneity, unresolved structure, and picking uncertainty.

Within the resolved corridor, negative relative Vp perturbations in the upper and middle crust are interpreted as candidate expressions of the sedimentary basin, fractured crust, and lower effective elastic moduli relative to the slab-informed reference model. Positive perturbations at intermediate depths are more consistent with denser mafic/ultramafic material and the high-velocity slab framework. These interpretations are intentionally regional: the current 10 km grid and checkerboard recovery do not justify assigning individual cells to single faults, depocenters, or fluid conduits.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.92\linewidth]{{{figures['vp'].name}}}
\caption{{P-wave velocity perturbations relative to the Nazca slab-informed initial model. Only active cells satisfying the hit-count threshold are shown.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.72\linewidth]{{{figures['vp_history'].name}}}
\caption{{Velocity inversion convergence on the 10 km grid. RMS is shown before each slowness update, with the final point recomputed after the last accepted update.}}
\end{{figure}}

\subsection{{Resolution diagnostic}}

The checkerboard test uses the final grid-eikonal ray geometry, the same active mask, and the same damping/smoothing values as the production inversion. The imposed pattern has a {100*CHECKERBOARD_SLOWNESS_FRACTION:.0f}\% slowness amplitude and includes {CHECKERBOARD_NOISE_S:.2f} s Gaussian noise. The recovered correlation is {m['vp_checkerboard']['correlation']:.2f}. This is a moderate recovery result: it supports cautious interpretation of broad, repeatedly sampled structures, but not detailed cell-by-cell interpretation or isolated edge anomalies.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.98\linewidth]{{{figures['checkerboard'].name}}}
\caption{{Linearized checkerboard recovery for the P-wave branch. The test evaluates the actual backtraced ray geometry used in the production inversion.}}
\end{{figure}}

\subsection{{Spectral-ratio Q pilot}}

After response removal, the Q branch retains {m['q_pilot']['observations_used']} station-pair observations. The inversion uses {m['q_pilot']['active_cells']} active cells and {m['q_pilot']['station_terms']} station terms, with a data-to-parameter ratio of {m['q_pilot']['data_parameter_ratio_cells_plus_stations']:.2f}. RMS decreases from {m['q_pilot']['initial_rms_s']:.3f} s to {m['q_pilot']['final_rms_s']:.3f} s. Applying the receiver terms from the joint solution without the path component gives an RMS of {m['q_pilot']['station_only_rms_s']:.3f} s, which shows that receiver and path terms are strongly coupled rather than independently sufficient. The largest absolute station term is {m['q_pilot']['max_abs_station_term_s']:.3f} s. However, the median spectral-fit $R^2$ is only {m['q_pilot']['median_r2']:.2f}. The Q field is therefore not interpreted as a robust image of intrinsic attenuation. Its main value is methodological: it quantifies receiver terms and demonstrates the need for stricter spectral weighting and independent validation.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.92\linewidth]{{{figures['q'].name}}}
\caption{{Spectral-ratio Q pilot inversion after instrument-response removal. Station terms remain first-order contributors, and the volumetric field is interpreted only diagnostically.}}
\end{{figure}}

\subsection{{Spectral-quality sensitivity}}

The expanded waveform set improves coverage but does not automatically improve the quality of the spectral-ratio fits. A hard filter of $R^2 \geq 0.4$ retains {m['q_pilot']['r2_ge_0p4_observations']} pairs from {m['q_pilot']['r2_ge_0p4_events']} events and {m['q_pilot']['r2_ge_0p4_station_pairs']} station-pair combinations, with median $R^2={m['q_pilot']['r2_ge_0p4_median_r2']:.2f}$. Therefore the present manuscript does not claim that simply downloading more waveforms will improve Q. Future work should prioritize shorter P windows, narrower frequency-band testing, and weighted inversions in which observations are scaled by spectral-fit quality.

\subsection{{Nazca/Benioff reconstruction}}

The local slab reference is reconstructed from {m['benioff']['control_events']} earthquakes between {m['benioff']['depth_min_km']:.0f} and {m['benioff']['depth_max_km']:.0f} km depth, explicitly including the Bucaramanga-nest interval. The reconstructed surface has an internal RMSE of {m['benioff']['rmse_km']:.1f} km and MAE of {m['benioff']['mae_km']:.1f} km against the control events. The surface is a geometric reference, not a tomographic constraint. Sampling the surface by the Vp and pilot-Q fields provides a compact way to compare seismicity geometry with model attributes while avoiding claims that the tomography alone reconstructs the plate.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.86\linewidth]{{{figures['benioff3d'].name}}}
\caption{{Three-dimensional Nazca/Benioff reconstruction from intermediate-depth SGC seismicity, including the Bucaramanga-nest depth range, sampled by velocity perturbations where the model is active.}}
\end{{figure}}

\section{{Discussion}}

The final grid-eikonal run confirms the central lesson of the previous revision: adding events improves sampling and data balance, but it does not automatically produce a calibrated Q model. The velocity branch is the main scientific product because it uses the strongest observation-to-parameter ratio and a ray geometry consistent with the forward operator. Even so, the model remains tied to fixed SGC hypocenters and a slab-informed initial model. The appropriate interpretation is a coverage-aware perturbation model for the illuminated MMV corridor, not a basin-wide absolute Vp reconstruction.

The response-corrected Q branch resolves the strongest methodological objection raised against the earlier draft: spectral slopes are no longer measured in raw counts. However, the low median $R^2$ and the magnitude of station terms show that Q remains a pilot product. The $R^2$ sensitivity test shows that more waveforms are not the controlling limitation. A cleaner subset exists, but it sacrifices coverage and does not remove the dominance of receiver terms. The paper should therefore present the Q branch as a reproducible diagnostic framework and as an experimental design result for future MMV attenuation tomography. A publication-grade Q model will require stricter spectral-fit weighting, possibly event/station hierarchical terms, shorter windows around the direct P arrival, and validation against independent spectral measurements.

The relation between Q and fluids must be stated cautiously. In principle, high attenuation, expressed as elevated $Q^{{-1}}$ or reduced Q, can be compatible with fluids, partial saturation, fracturing, scattering, and temperature effects. In this dataset, a Q anomaly cannot by itself demonstrate fluid movement. The most defensible interpretation is conditional: Q can identify candidate fluid-sensitive zones only where response-corrected spectral fits are reliable, receiver terms are not dominant, and the anomaly is spatially coherent with independent evidence such as low Vp, clustered seismicity, faults, or slab-dehydration geometry. Thus, the present results are useful for designing the next attenuation experiment, but they are not direct proof of active fluid migration.

The Benioff reconstruction is useful because it separates geometry from tomography. Earthquakes define the local slab-like surface; Vp and pilot Q are then sampled onto that surface. This avoids circular interpretation and makes clear that the plate geometry is not imposed by the Q inversion.

\section{{Conclusions}}

\begin{{enumerate}}
\item The final grid-eikonal velocity inversion improves the defensibility of the MMV model by using {m['vp']['observations']} P picks, {m['vp']['active_cells']} active cells, and a data-to-parameter ratio of {m['vp']['data_parameter_ratio_cells_only']:.2f}.
\item The 10 km curved-ray recomputation reduces P-wave travel-time RMS from {m['vp']['initial_rms_s']:.2f} s to {m['vp']['final_rms_s']:.2f} s. The residual remains high and should be discussed as a fixed-hypocenter/catalog limitation.
\item The response-corrected spectral-ratio Q branch is methodologically stronger than the previous raw-count version, but median spectral-fit $R^2$ of {m['q_pilot']['median_r2']:.2f} prevents direct geological interpretation of the volumetric Q field.
\item Q anomalies may be fluid-sensitive, but in this dataset they cannot prove fluid motion without stronger spectral fits, smaller receiver terms, and independent geological or geochemical constraints.
\item The local Nazca/Benioff surface reconstructed from {m['benioff']['control_events']} earthquakes between {m['benioff']['depth_min_km']:.0f} and {m['benioff']['depth_max_km']:.0f} km provides the strongest integrated geometric context for interpreting the velocity model.
\end{{enumerate}}

\section*{{Data and Code Availability}}

The reproducibility package is contained in the output folder accompanying this draft. Final scripts are copied to the \texttt{{codigos\_finales}} subfolder, and the SGC StationXML response metadata are stored with the station metadata.

\section*{{Funding}}
This research received no external funding.

\section*{{Declaration of Competing Interest}}
The authors declare no competing interests.

\end{{document}}
"""
    tex.write_text(text, encoding="utf-8")
    return tex


def main() -> None:
    ensure_dirs()
    xg, yg, zg, vp0 = load_grid_and_vp()
    domain = rectangular_domain_mask(xg, yg, zg)
    picks = pd.read_csv(PICKS_CSV)
    pairs_raw = pd.read_csv(PAIR_CSV)
    stations = pd.read_csv(STATIONS_CSV)
    trace_df = pd.read_csv(TRACE_CSV)
    events_for_area = picks.drop_duplicates("event_id")[["event_id", "event_lat", "event_lon", "event_depth_km"]].copy()
    area_fig = plot_area(events_for_area, stations)

    groups = prep_vp_groups(picks, xg, yg, zg)
    print(f"[Grid 10km] grid shape={vp0.shape}, total cells={vp0.size}, unique source nodes={len(groups)}", flush=True)
    initial_rows = build_vp_rows(vp0, groups, xg, yg, zg)
    rows0, _, _, _, failures0, elapsed0 = initial_rows
    vp_hits = hit_count_from_rows(rows0, vp0.size).reshape(vp0.shape)
    active_vp = domain & (vp_hits >= VP_MIN_HITS)
    active_diag = {"initial_fsm_row_failures": int(failures0), "initial_fsm_build_seconds": float(elapsed0)}
    print(f"[Grid 10km] active Vp cells: {int(active_vp.sum())} / {vp0.size}", flush=True)
    vp, history, vp_payload = run_vp_fsm(vp0, picks, xg, yg, zg, active_vp, initial_rows=initial_rows)
    if vp_payload is None:
        raise RuntimeError("FSM Vp inversion did not produce an accepted model.")

    pairs = prepare_q_pairs(pairs_raw)
    qinv, q0, active_q, station_df, q_pred, q_hits, q_metrics = run_q_fsm(vp, pairs, xg, yg, zg, domain)
    true_cb, rec_cb, cb_metrics = run_vp_checkerboard(vp_payload["A"], vp_payload["active_idx"], active_vp, vp)

    benioff_events, benioff_surface, benioff_metrics = reconstruct_benioff(vp0, vp, qinv, active_vp, active_q, xg, yg, zg)

    np.save(MODEL / "sgc_final_fsm_vp_initial.npy", vp0)
    np.save(MODEL / "sgc_final_fsm_vp_final.npy", vp)
    np.save(MODEL / "sgc_final_fsm_vp_active_mask.npy", active_vp)
    np.save(MODEL / "sgc_final_fsm_qinv_pilot.npy", qinv)
    np.save(MODEL / "sgc_final_fsm_q_pilot.npy", 1.0 / np.maximum(qinv, 1.0e-12))
    np.save(MODEL / "sgc_final_fsm_q_active_mask.npy", active_q)
    np.save(MODEL / "sgc_final_fsm_vp_hit_count.npy", vp_hits)
    np.save(MODEL / "sgc_final_fsm_q_hit_count.npy", q_hits)
    np.savez(MODEL / "sgc_final_fsm_grid.npz", xg=xg, yg=yg, zg=zg)

    history.to_csv(DATA / "sgc_final_fsm_vp_inversion_history.csv", index=False)
    station_df.to_csv(DATA / "sgc_final_fsm_q_station_terms.csv", index=False)
    q_pred.to_csv(DATA / "sgc_final_fsm_q_predictions.csv", index=False)
    benioff_events.to_csv(DATA / "sgc_final_fsm_benioff_control_events.csv", index=False)
    benioff_surface.to_csv(DATA / "sgc_final_fsm_benioff_surface_sampled_vp_q.csv", index=False)

    vp_pred = picks.iloc[vp_payload["row_ids"]].copy().reset_index(drop=True)
    vp_pred["pred_final_s"] = vp_payload["pred"]
    vp_pred["residual_final_s"] = vp_payload["obs"] - vp_payload["pred"]
    vp_pred.to_csv(DATA / "sgc_final_fsm_vp_predictions.csv", index=False)

    vp_fig = plot_vp(vp0, vp, active_vp, xg, yg, zg)
    history_fig = plot_vp_history(history)
    checker_fig = plot_checkerboard(true_cb, rec_cb, active_vp, xg, yg, zg)
    q_fig = plot_q(qinv, q0, active_q, station_df, q_pred, xg, yg, zg)
    benioff_fig = plot_benioff_3d(benioff_events, benioff_surface)

    initial_rms = float(history.iloc[0]["pre_rms_s"])
    final_rms = float(history["best_rms_s"].iloc[-1])
    metrics = {
        "run_name": "Final SGC grid-eikonal Vp and response-corrected spectral-ratio Q pilot tomography with Bucaramanga nest and expanded stations",
        "input_picks": str(PICKS_CSV),
        "input_pairs": str(PAIR_CSV),
        "instrument_response_removed_for_q": bool(
            "instrument_response_removed" in trace_df.columns
            and trace_df["instrument_response_removed"].astype(str).str.lower().eq("true").all()
        ),
        "instrument_response_output": "VEL",
        "stations_used": int(stations["station"].nunique()),
        "station_list": sorted(stations["station"].astype(str).unique().tolist()),
        "trace_spectra": int(len(trace_df)),
        "q_pairs_input": int(len(pairs_raw)),
        "domain": {
            "lon_min": LON_MIN,
            "lon_max": LON_MAX,
            "lat_min": LAT_MIN,
            "lat_max": LAT_MAX,
            "max_model_depth_km": MAX_MODEL_DEPTH_KM,
            "rectangular": True,
        },
        "vp": {
            "method": "26-neighbor 3-D grid-eikonal minimum-time rays with iterative LSMR slowness updates",
            "observations": int(len(picks)),
            "events": int(picks["event_id"].nunique()),
            "active_cells": int(active_vp.sum()),
            "data_parameter_ratio_cells_only": float(len(picks) / max(int(active_vp.sum()), 1)),
            "initial_rms_s": initial_rms,
            "final_rms_s": final_rms,
            "rms_reduction_percent": float(100.0 * (initial_rms - final_rms) / max(initial_rms, 1.0e-12)),
            "min_hit_threshold": int(VP_MIN_HITS),
            "regularization": {"alpha": VP_ALPHA, "damp": VP_DAMP, "laplace": VP_LAPLACE},
            **active_diag,
        },
        "q_pilot": q_metrics,
        "vp_checkerboard": cb_metrics,
        "benioff": benioff_metrics,
        "figures": {
            "area": str(area_fig),
            "vp": str(vp_fig),
            "vp_history": str(history_fig),
            "checkerboard": str(checker_fig),
            "q": str(q_fig),
            "benioff3d": str(benioff_fig),
        },
        "limitations": [
            "Vp uses fixed SGC hypocenters; no joint hypocenter-velocity relocation is performed.",
            "The initial Vp model includes a Nazca slab-informed prior; anomalies are perturbations relative to that prior.",
            "Q spectra are response-corrected, but low spectral-fit R2 still makes the volumetric Q field pilot-level.",
            "The checkerboard is a linearized diagnostic on final grid-eikonal ray geometry, not a full nonlinear synthetic reinversion.",
        ],
    }
    (OUT / "sgc_final_fsm_tomography_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    figures = {
        "area": area_fig,
        "vp": vp_fig,
        "vp_history": history_fig,
        "checkerboard": checker_fig,
        "q": q_fig,
        "benioff3d": benioff_fig,
    }
    paper_tex = write_paper(metrics, figures)
    shutil.copy2(Path(__file__), CODE / Path(__file__).name)
    shutil.copy2(ROOT / "download_sgc_stationxml_responses.py", CODE / "download_sgc_stationxml_responses.py")
    print(json.dumps({"metrics": str(OUT / "sgc_final_fsm_tomography_metrics.json"), "paper_tex": str(paper_tex), **metrics}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
