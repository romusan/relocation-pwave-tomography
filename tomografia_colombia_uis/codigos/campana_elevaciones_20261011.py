"""2026-10-11 elevation campaign: repeat the 1D vs 1D+placas comparison.

Same observations, split, domain, grids, regularization and rounds as the
original COMPARACION_INICIOS_1D_3D_2026-10-10 experiment. Two changes are
introduced together and applied identically to both arms:

1. The verified exact discrete Godunov Jacobian (adjoint_cpp) replaces the
   approximate 26-neighbour kernel; the solver/adjoint gate is
   codigos/test_adjoint_consistent.py.
2. Station elevations enter as a first-order plane-wave receiver correction
   (config station_elevation=true; gate codigos/test_elevacion_20261011.py).

Outputs are new folders under resultados/campana_elev_20261011; nothing
from the previous campaigns is modified. The shared field cache only gains
new, identity-verified entries for the new trial models.
"""
from pathlib import Path
import argparse, json, shutil, sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codigos'))
from regional.inputs import sha256, load_profile
from regional.geometry import Projection, grids
from regional.pipeline_consistent import run_velocity
from elevaciones_20261011 import with_elevations, INVENTORY

BASE = ROOT / 'resultados/migracion_isc_choco_2026-10-10'
CAMPAIGN = ROOT / 'resultados/campana_elev_20261011'
RUN1D = CAMPAIGN / 'tomografia_1d_elev'
RUN3D = CAMPAIGN / 'tomografia_1d_placas_elev'
PRIOR = ROOT / 'resultados/modelo_1d_placas_continuas_slab2_2026-10-10/vp_inicial_1d_mas_placas_30km.npz'
PROFILE = ROOT / 'datos/perfil_vp_vmm_transferencia.csv'
CODE = ['regional/pipeline_consistent.py', 'regional/pipeline_nurbs.py', 'regional/forward.py',
        'regional/rays_consistent.py', 'regional/rays.py', 'regional/inversion.py',
        'regional/inputs.py', 'regional/geometry.py', 'regional/initial_model.py',
        'native/eikonal_adjoint.cpp', 'native/eikonal_adjoint.dll',
        'test_adjoint_consistent.py', 'test_elevacion_20261011.py']


def common():
    cfg = json.loads((BASE / 'config.json').read_text(encoding='utf-8'))
    cfg['station_elevation'] = True
    cfg['ray_backend'] = 'adjoint_cpp'
    domain = json.loads((BASE / 'domain.json').read_text(encoding='utf-8'))
    pr = Projection(domain)
    grid, coarse = grids(domain, pr, cfg['solver_spacing_km'], cfg['inversion_spacing_km'])
    original = pd.read_csv(BASE / 'associated_picks_selected.csv', low_memory=False)
    selected, missing = with_elevations(original)
    pd.testing.assert_frame_equal(selected.drop(columns='st_elev_km'), original,
                                  check_dtype=False)
    events = pd.read_csv(BASE / 'events_split.csv')
    n_picks = int((selected.st_elev_km > 0).sum())
    return cfg, domain, pr, grid, coarse, selected, events, {
        'stations_total': int(selected.station.nunique()),
        'stations_missing_elevation': missing,
        'picks_with_positive_elevation': n_picks,
        'max_elevation_km': float(selected.st_elev_km.max())}


def prepare(out, selected, events, cfg, domain, extra):
    out.mkdir(parents=True, exist_ok=False)
    manifest = {str(Path(__file__).resolve()): sha256(__file__)}
    for name in ['associated_picks_selected.csv', 'events_split.csv']:
        shutil.copy2(BASE / name, out / name)
        manifest[str(BASE / name)] = sha256(BASE / name)
    for name in ['domain.json', 'config.json']:
        manifest[str(BASE / name)] = sha256(BASE / name)
    for rel in CODE:
        f = ROOT / 'codigos' / rel
        manifest[str(f)] = sha256(f)
    for f in [PROFILE, *INVENTORY]:
        manifest[str(f)] = sha256(f)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (out / 'config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    (out / 'domain.json').write_text(json.dumps(domain, indent=2), encoding='utf-8')
    events.to_csv(out / 'events_split.csv', index=False)
    selected.to_csv(out / 'associated_picks_selected.csv', index=False)
    (out / 'elevation_audit.json').write_text(json.dumps(extra, indent=2), encoding='utf-8')
    return manifest


def run1d():
    cfg, domain, pr, grid, coarse, selected, events, audit = common()
    if (RUN1D / 'summary.json').exists():
        print('run1d already complete', flush=True)
        return
    manifest = prepare(RUN1D, selected, events, cfg, domain, audit)
    print('RUN 1D original + elevacion: cubiertas',
          audit['stations_total'] - len(audit['stations_missing_elevation']),
          'de', audit['stations_total'], 'estaciones; max', round(audit['max_elevation_km'], 2), 'km', flush=True)
    summary, cache, nodes, statics = run_velocity(selected, events, grid, coarse,
        load_profile(PROFILE), pr, domain, cfg, RUN1D, manifest)
    cache.close()
    print('run1d completed:', RUN1D, flush=True)


def run3d():
    cfg, domain, pr, grid, coarse, selected, events, audit = common()
    if (RUN3D / 'summary.json').exists():
        print('run3d already complete', flush=True)
        return
    manifest = prepare(RUN3D, selected, events, cfg, domain, audit)
    manifest[str(PRIOR)] = sha256(PRIOR)
    (RUN3D / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (RUN3D / 'preparation.json').write_text(json.dumps({
        'initial_model': str(PRIOR), 'same_prior_as': 'COMPARACION_INICIOS_1D_3D_2026-10-10',
        'conditioned_restart': True}, indent=2), encoding='utf-8')
    print('RUN 1D+placas + elevacion: max elevacion', round(audit['max_elevation_km'], 2), 'km', flush=True)
    summary, cache, nodes, statics = run_velocity(selected, events, grid, coarse,
        load_profile(PROFILE), pr, domain, cfg, RUN3D, manifest, initial_model=PRIOR)
    cache.close()
    print('run3d completed:', RUN3D, flush=True)


def compare():
    import comparar_elev_20261011 as comp
    out = CAMPAIGN / 'comparacion'
    if (out / 'comparacion.json').exists():
        print('compare already complete', flush=True)
        return
    comp.main_argv(['--old', str(RUN1D), '--new', str(RUN3D), '--prior', str(PRIOR),
                    '--output', str(out)])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['run1d', 'run3d', 'compare', 'all'])
    a = ap.parse_args()
    if a.stage in ('run1d', 'all'):
        run1d()
    if a.stage in ('run3d', 'all'):
        run3d()
    if a.stage in ('compare', 'all'):
        compare()
