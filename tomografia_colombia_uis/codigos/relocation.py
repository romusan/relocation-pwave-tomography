"""Candidate scoring extracted from the reference relocation loop.

Input: predicted times [candidate, pick], observed travel times [pick].
Add station statics to predictions BEFORE calling. No phase association,
field calculation, spatial search, weighting or catalog edits occur here.
The native backend must be selected explicitly and fails if unavailable.
"""
from pathlib import Path
import ctypes
import os
import numpy as np

_LIB = None


def _inputs(times, obs):
    times = np.ascontiguousarray(times, dtype=np.float64)
    obs = np.ascontiguousarray(obs, dtype=np.float64)
    if times.ndim != 2 or obs.ndim != 1 or times.shape[1] != obs.size:
        raise ValueError('Expected times [candidate, pick] and obs [pick]')
    if min(times.shape) < 1 or not np.isfinite(obs).all():
        raise ValueError('Empty inputs or non-finite observations')
    return times, obs


def _native():
    global _LIB
    if _LIB is None:
        path = Path(__file__).resolve().parent / 'native' / (
            'relocation.dll' if os.name == 'nt' else 'relocation.so')
        if not path.exists():
            raise RuntimeError('C++ backend not built: run codigos/native/build.py')
        lib = ctypes.CDLL(str(path))
        array = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        lib.score_candidates.argtypes = [array, array, ctypes.c_int64,
            ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), array, array]
        lib.score_candidates.restype = ctypes.c_int
        _LIB = lib
    return _LIB


def score_candidates(times, obs, backend='numpy'):
    """Return (best index, origin shifts, RMS scores); best=-1 if none valid."""
    times, obs = _inputs(times, obs)
    shifts = np.full(times.shape[0], np.nan)
    scores = np.full(times.shape[0], np.inf)
    if backend == 'cpp':
        best = ctypes.c_int64(-1)
        status = _native().score_candidates(times, obs, *times.shape,
                                            ctypes.byref(best), shifts, scores)
        if status:
            raise RuntimeError(f'C++ scoring failed with status {status}')
        return best.value, shifts, scores
    if backend not in ('reference', 'numpy'):
        raise ValueError(f'Unknown backend: {backend}')
    # Reference operation order is intentionally retained for equivalence.
    with np.errstate(over='ignore', invalid='ignore'):
        if backend == 'reference':
            for c, tcalc in enumerate(times):
                if not np.isfinite(tcalc).all():
                    continue
                shift = float(np.median(obs - tcalc))
                rr = float(np.sqrt(np.mean((obs - (tcalc + shift)) ** 2)))
                if np.isfinite(shift) and np.isfinite(rr):
                    shifts[c], scores[c] = shift, rr
        else:
            valid = np.isfinite(times).all(axis=1)
            t = times[valid]
            shift = np.median(obs[None, :] - t, axis=1)
            rr = np.sqrt(np.mean((obs[None, :] - (t + shift[:, None])) ** 2, axis=1))
            good = np.isfinite(shift) & np.isfinite(rr)
            indices = np.flatnonzero(valid)[good]
            shifts[indices], scores[indices] = shift[good], rr[good]
    best = int(np.argmin(scores))
    return (best if np.isfinite(scores[best]) else -1), shifts, scores
