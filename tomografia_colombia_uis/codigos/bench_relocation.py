"""Compare scoring backends on real P picks and synthetic homogeneous times.

This is a kernel benchmark, NOT a tomography or a relocated catalog.
Times are prepared once and shared; field computation, table preparation,
ray tracing and inversion are excluded. C++ includes ctypes call overhead.
"""
from pathlib import Path
import argparse
import json
import platform
import sys
from time import perf_counter
import numpy as np
import pandas as pd
from relocation import score_candidates

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', type=int, default=500)
    ap.add_argument('--repeats', type=int, default=3)
    a = ap.parse_args()
    if a.events < 1 or a.repeats < 1:
        ap.error('events and repeats must be positive')
    pk = pd.read_csv(ROOT / 'datos' / 'picks_phasenet.csv')
    pk = pk[pk.fase.eq('P')]
    offset = np.array(np.meshgrid(*([np.arange(-2, 3) * 5.] * 3), indexing='ij'))
    offset = offset.reshape(3, -1).T
    cases = []
    for _, g in pk.groupby('event_id', sort=False):
        g = g.drop_duplicates(['net', 'sta'])
        if len(g) < 4:
            continue
        source = np.array([0., 0., g.ev_prof_km.iloc[0]])
        stations = np.column_stack([
            (g.st_lon - g.ev_lon) * 111.32 * np.cos(np.radians(g.ev_lat)),
            (g.st_lat - g.ev_lat) * 111.32, np.zeros(len(g))])
        candidates = source + offset
        candidates = candidates[(candidates[:, 2] >= 0) & (candidates[:, 2] <= 180)]
        times = np.linalg.norm(candidates[:, None, :] - stations[None, :, :], axis=2) / 6.2
        cases.append((times, g.t_viaje_s.to_numpy(dtype=float)))
        if len(cases) >= a.events:
            break
    if not cases:
        raise RuntimeError('No events with four distinct P stations')
    reference = [score_candidates(t, o, 'reference') for t, o in cases]
    max_error = 0.
    for backend in ('numpy', 'cpp'):
        for (t, o), expected in zip(cases, reference):
            actual = score_candidates(t, o, backend)
            if expected[0] != actual[0]:
                raise AssertionError(f'{backend}: different selected candidate')
            for x, y in zip(expected[1:], actual[1:]):
                np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-12, equal_nan=True)
                mask = np.isfinite(x) & np.isfinite(y)
                max_error = max(max_error, float(np.max(np.abs(x[mask] - y[mask]), initial=0)))
    elapsed = {}
    for backend in ('reference', 'numpy', 'cpp'):
        samples = []
        for _ in range(a.repeats):
            start = perf_counter()
            for t, o in cases:
                score_candidates(t, o, backend)
            samples.append(perf_counter() - start)
        elapsed[backend] = {'seconds': samples, 'median_s': float(np.median(samples))}
    result = {'events': len(cases), 'candidates_total': sum(len(t) for t, _ in cases),
        'times': 'synthetic homogeneous 6.2 km/s, with observed P picks; not scientific results',
        'scope': 'candidate scoring only, excludes preparation, travel fields, rays and inversion',
        'max_abs_difference_s': max_error, 'candidate_disagreements': 0, 'timings': elapsed,
        'speedup_cpp_vs_reference': elapsed['reference']['median_s'] / elapsed['cpp']['median_s'],
        'speedup_cpp_vs_numpy': elapsed['numpy']['median_s'] / elapsed['cpp']['median_s'],
        'environment': {'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__},
        'build': json.loads((ROOT / 'codigos/native/build_info.json').read_text(encoding='utf-8'))}
    out = ROOT / 'resultados/revision_2026-10-09/benchmark_relocation.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
