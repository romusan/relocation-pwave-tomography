"""Read-only audit of current inputs; writes only a separate JSON report."""
from pathlib import Path
import hashlib
import json
import platform
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    paths = {key: ROOT / 'datos' / name for key, name in {
        'catalog': 'catalogo_seleccionado.csv', 'picks': 'picks_phasenet.csv',
        'done': 'campana_eventos_hechos.csv', 'validation': 'validacion_phasenet.csv',
        'analyst': 'picks_analista.csv', 'stations': 'estaciones_cm.csv',
    }.items()}
    frames = {key: pd.read_csv(path) for key, path in paths.items()}
    cat, pk, done, val, analyst, stations = [frames[k] for k in paths]
    dom = json.loads((ROOT / 'dominio.json').read_text(encoding='utf-8'))
    keys = ['event_id', 'net', 'sta', 'fase']
    multiplicity = pk.groupby(keys).size()
    per_event = pk.groupby('event_id').size()
    phase = {}
    for f, g in pk.groupby('fase'):
        sub = g.groupby(['event_id', 'net', 'sta']).size()
        phase[f] = {
            'picks': len(g), 'events': int(g.event_id.nunique()),
            'stations': len(g[['net', 'sta']].drop_duplicates()),
            'multiple_candidate_groups': int((sub > 1).sum()),
            'negative_travel_times': int((g.t_viaje_s < 0).sum()),
            'prob_median': float(g.prob.median()),
        }
    matched_validation = {}
    for f in ('P', 'S'):
        v = val[f'res_{f}'].dropna()
        matched_validation[f] = {
            'matched_station_event_rows': len(v),
            'median_s': float(v.median()),
            'mad_s': float(np.median(abs(v - v.median()))),
            'p95_abs_s': float(v.abs().quantile(.95)),
            'warning': 'nearest-to-analyst matching; misses and false positives not measured',
        }
    coords = pk[['event_id', 'ev_lat', 'ev_lon', 'ev_prof_km']].drop_duplicates('event_id')
    # Catalog gap concerns the catalog network. New-pick gap is only a
    # preliminary geometry diagnostic: all candidates, no association/QC.
    gaps = []
    for eid, g in pk.groupby('event_id', sort=False):
        st = g.drop_duplicates(['net', 'sta'])
        dx = (st.st_lon - st.ev_lon) * np.cos(np.radians(st.ev_lat))
        dy = st.st_lat - st.ev_lat
        az = np.sort(np.degrees(np.arctan2(dx, dy)) % 360)
        gap = float(np.diff(np.r_[az, az[0] + 360]).max())
        gaps.append(gap)
    n_unique_p = pk[pk.fase.eq('P')].groupby('event_id').apply(
        lambda g: len(g[['net', 'sta']].drop_duplicates()), include_groups=False)
    stats = {
        'domain': dom,
        'catalog_events': len(cat), 'done_rows': len(done),
        'done_unique_events': int(done.event_id.nunique()),
        'picks_rows': len(pk), 'events_with_picks': int(pk.event_id.nunique()),
        'done_without_picks': len(set(done.event_id) - set(pk.event_id)),
        'exact_duplicate_pick_rows': int(pk.duplicated().sum()),
        'multiple_event_station_phase_groups': int((multiplicity > 1).sum()),
        'extra_candidates_over_one_per_group': int((multiplicity - 1).clip(lower=0).sum()),
        'events_with_at_least_4_distinct_P_stations_before_QC': int((n_unique_p >= 4).sum()),
        'picks_per_event_median': float(per_event.median()), 'phase': phase,
        'preliminary_gap_median_deg': float(np.median(gaps)),
        'preliminary_gap_lt180_events': int((np.array(gaps) < 180).sum()),
        'event_lat_range_with_picks': [float(coords.ev_lat.min()), float(coords.ev_lat.max())],
        'analyst_reference': {'picks': len(analyst), 'events': int(analyst.event_id.nunique()),
            'lat_range': [float(analyst.ev_lat.min()), float(analyst.ev_lat.max())],
            'lon_range': [float(analyst.ev_lon.min()), float(analyst.ev_lon.max())]},
        'validation': matched_validation,
        'station_metadata_rows': len(stations),
        'station_codes_with_multiple_metadata_rows': int((stations.groupby(['net', 'sta']).size() > 1).sum()),
        'full_grid_unknowns': dom['malla_elegida']['celdas'],
        'raw_P_picks_per_full_grid_node': float(pk.fase.eq('P').sum() / dom['malla_elegida']['celdas']),
        'environment': {'python': sys.version, 'platform': platform.platform(),
            'numpy': np.__version__, 'pandas': pd.__version__},
        'inputs_sha256': {str(p.relative_to(ROOT)): sha256(p)
                          for p in [*paths.values(), ROOT / 'dominio.json']},
    }
    out = ROOT / 'resultados' / 'revision_2026-10-09'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'auditoria_datos.json').write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in stats.items() if k not in ('domain', 'inputs_sha256', 'environment')},
                     indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
