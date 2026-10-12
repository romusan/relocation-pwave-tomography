"""Vendored first-order Fast Sweeping operator, reference implementation.
Source: repositorio_q1_tomografia_fmm/code/run_sgc_final_fsm_vp_q_tomography.py
Local migration 2026-10-09; only decorator disk caching was added.
MIT notice: see LICENSE_REFERENCE.txt. Numba is mandatory (no silent fallback).
Known analytical precision gate failures are retained and documented.
"""
import math
import numpy as np
from numba import njit

@njit(cache=True)
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

@njit(cache=True)
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
