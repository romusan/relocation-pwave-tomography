"""Nonlinear synthetic recovery controls WITH station elevations (2026-10-11).

Repeat of codigos/sinteticos_no_lineales_20261011.py on the same event/station
population, the same checkerboard truth, the same jitter/noise/statics seeds
and the same gates. The only change: station elevations enter the synthetic
data generation and the inversion through the first-order plane-wave receiver
correction (config station_elevation=true). Stations without a local
elevation entry are kept at 0 km and disclosed in generation.json.

Outputs go to resultados/robustez_elev_20261011; the previous robustness
package is untouched.
"""
from pathlib import Path
import json, sys, time, types, hashlib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsmr
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codigos'))
from regional.geometry import Grid, candidate_nodes
from regional.forward import FieldCache
from regional.pipeline_consistent import relocate, residual
from regional.inversion import station_design
from nazca_campana_20261010 import projection
from elevaciones_20261011 import with_elevations

OLD = ROOT / 'resultados/tomografia_1d_adjunto_consistente_2026-10-10'
OUT = ROOT / 'resultados/robustez_elev_20261011'
CFG = json.loads((OLD / 'config.json').read_text(encoding='utf-8'))
DOMAIN = json.loads((OLD / 'domain.json').read_text(encoding='utf-8'))
PR = projection(DOMAIN)
F = np.load(OLD / 'vp_final.npz')
GRID = Grid(F['xg'], F['yg'], F['zg'])
COARSE = Grid(F['inversion_xg'], F['inversion_yg'], F['inversion_zg'])
PROFILE = (GRID.z, F['vp_initial'][:, 0, 0])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def metrics(r):
    a = np.abs(r)
    q = np.quantile(a, .95)
    return dict(rms_s=float(np.sqrt(np.mean(r * r))), median_abs_s=float(np.median(a)),
                p90_abs_s=float(np.quantile(a, .9)),
                trimmed95_rms_s=float(np.sqrt(np.mean(r[a <= q] ** 2))))


def load():
    p, missing = with_elevations(pd.read_csv(OLD / 'predictions.csv', low_memory=False))
    return p, missing


def refined(h):
    return Grid(*[np.arange(a[0], a[-1] + h / 2, h) for a in [GRID.x, GRID.y, GRID.z]])


def truth(grid, amp):
    z, y, x = np.meshgrid(grid.z, grid.y, grid.x, indexing='ij')
    sign = np.where((np.floor((x - COARSE.x[0]) / 40) + np.floor((y - COARSE.y[0]) / 40)
                     + np.floor(z / 40)).astype(int) % 2 == 0, 1., -1.)
    base = np.broadcast_to(np.interp(grid.z, *PROFILE)[:, None, None], grid.shape)
    return base * (1 + amp * sign)


def nodes_for(p, grid=GRID):
    nodes = {}
    for eid, g in p.groupby('event_id', sort=False):
        xyz = g[['source_x_km', 'source_y_km', 'ev_prof_km']].iloc[0].to_numpy(float)
        near = candidate_nodes(grid, grid.nodes(xyz), grid.spacing, DOMAIN, PR)
        nodes[eid] = tuple(near[np.argmin(np.linalg.norm(grid.xyz(near) - xyz, axis=1))])
    return nodes


def predict_exact(cache, p):
    pred = np.empty(len(p))
    for sta, ids in p.groupby('station', sort=False).indices.items():
        q = p.iloc[ids]
        interp = RegularGridInterpolator((cache.grid.z, cache.grid.y, cache.grid.x), cache.get(sta))
        xyz = q[['source_x_km', 'source_y_km', 'ev_prof_km']].to_numpy(float)
        nodes = np.asarray(cache.grid.nodes(xyz)).reshape(-1, 3)
        pred[ids] = interp(xyz[:, ::-1]) + cache.elevation_correction(sta, nodes)
    return pred


def bootstrap(p, a, b, seed=20261011):
    labels, inv = np.unique(p.event_id.to_numpy(), return_inverse=True)
    n = np.bincount(inv)
    aa = np.bincount(inv, weights=a * a)
    bb = np.bincount(inv, weights=b * b)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(3000):
        ix = rng.integers(0, len(labels), len(labels))
        values.append(np.sqrt(aa[ix].sum() / n[ix].sum()) - np.sqrt(bb[ix].sum() / n[ix].sum()))
    return dict(delta_rms_3d_minus_1d_s=metrics(a)['rms_s'] - metrics(b)['rms_s'],
                ci95_s=np.quantile(values, [.025, .975]).tolist(), replicates=3000, unit='event')


def engine():
    original = ROOT / 'codigos/regional/pipeline_consistent.py'
    source = original.read_text(encoding='utf-8')
    start = source.index('    diagnostic, imposed, recovered = linear_resolution(')
    stop = source.index('    pred_output = p.copy()', start)
    source = (source[:start] + "    diagnostic={'kind':'not run: nonlinear/spatial experiment'}\n"
              + "    lcurve=pd.DataFrame({'converged':[]})\n" + source[stop:])
    module = types.ModuleType('regional.robustness_engine_elev')
    module.__package__ = 'regional'
    exec(compile(source, str(original) + ' [diagnostics omitted]', 'exec'), module.__dict__)
    return module.run_velocity, source


def fit_control(p, folder, cfg):
    train = p[p.split.eq('train')].reset_index(drop=True)
    stations = p[['station', 'st_x_km', 'st_y_km', 'st_elev_km']].drop_duplicates()
    cache = FieldCache(F['vp_initial'], GRID, stations, cfg['cache_directory'], cfg)
    nodes = nodes_for(p)
    origins = {e: 0. for e in nodes}
    statics = {s: 0. for s in p.station.unique()}
    S, free, ref = station_design(train)
    A = sparse.vstack([S, cfg['station_damping'] * sparse.eye(len(free))], format='csr')
    history = []
    for r in range(1, 4):
        nodes, origins, rel = relocate(cache, train, nodes, origins, statics, DOMAIN, PR, cfg)
        before = residual(cache, train, nodes, origins, statics)
        sol = lsmr(A, np.r_[before, np.zeros(len(free))], atol=cfg['lsmr_atol'],
                   btol=cfg['lsmr_btol'], maxiter=cfg['lsmr_maxiter'])
        assert sol[1] in (0, 1, 2, 4, 5)
        for alpha in (1., .5, .25):
            trial = dict(statics)
            for s, d in zip(free, sol[0]):
                trial[s] += alpha * float(np.clip(d, -cfg['max_static_step_s'], cfg['max_static_step_s']))
            after = residual(cache, train, nodes, origins, trial)
            if np.mean(after ** 2) <= np.mean(before ** 2) + 1e-10:
                statics = trial
                break
        history.append(dict(round=r, train=metrics(residual(cache, train, nodes, origins, statics))))
        rel.to_csv(folder / f'control_relocation_{r}.csv', index=False)
    save(folder / 'control_1d.json', dict(history=history, reference=ref, statics=statics))
    return cache, nodes, statics


def fit_pair(p, folder, cfg):
    folder.mkdir(exist_ok=True)
    save(folder / 'config.json', cfg)
    save(folder / 'domain.json', DOMAIN)
    events = p[['event_id', 'split']].drop_duplicates()
    events.to_csv(folder / 'partition.csv', index=False)
    run, source = engine()
    (folder / 'inversion_engine.py').write_text(source, encoding='utf-8')
    summary, cache, nodes, statics = run(p, events, GRID, COARSE, PROFILE, PR, DOMAIN, cfg, folder,
                                       {'original_engine_sha256': digest(ROOT / 'codigos/regional/pipeline_consistent.py')})
    test = p[p.split.eq('holdout')].reset_index(drop=True)
    r3 = test.observed_s.to_numpy() - predict_exact(cache, test) - test.station.map(statics).to_numpy()
    cache.close()
    c1, n1, s1 = fit_control(p, folder, cfg)
    r1 = test.observed_s.to_numpy() - predict_exact(c1, test) - test.station.map(s1).to_numpy()
    c1.close()
    result = dict(events=len(events), train_events=events.split.eq('train').sum().item(),
        holdout_events=events.split.eq('holdout').sum().item(),
        train_picks=int(p.split.eq('train').sum()), holdout_picks=len(test),
        control_1d=metrics(r1), model_3d=metrics(r3), paired_bootstrap=bootstrap(test, r3, r1),
        test_only_stations=sorted(set(test.station) - set(p[p.split.eq('train')].station)))
    q = test[['event_id', 'station', 'ev_lat', 'ev_lon', 'ev_prof_km']].copy()
    q['residual_1d_s'] = r1
    q['residual_3d_s'] = r3
    q.to_csv(folder / 'exact_holdout_residuals.csv', index=False)
    save(folder / 'comparison.json', result)
    print(json.dumps(result), flush=True)
    return result


def recovery(folder, amp, p):
    f = np.load(folder / 'vp_final.npz')
    vp = f['vp']
    active = f['active']
    imposed = truth(GRID, amp)
    base = F['vp_initial']
    from regional.rays import mapping
    mapflat = mapping(GRID, COARSE)
    count = np.bincount(mapflat, minlength=active.size)
    reduce = lambda a: np.bincount(mapflat, weights=a.ravel(), minlength=active.size) / count
    target = reduce((imposed - base) / base)[active.ravel()]
    got = reduce((vp - base) / base)[active.ravel()]
    corr = float(np.corrcoef(target, got)[0, 1])
    ratio = float(np.dot(target, got) / np.dot(target, target))
    result = dict(amplitude_fraction=amp, active_cells=int(active.sum()), correlation=corr,
        relative_l2_error=float(np.linalg.norm(got - target) / np.linalg.norm(target)),
        least_squares_amplitude_ratio=ratio,
        sign_agreement=float(np.mean(np.sign(target) == np.sign(got))),
        velocity_rmse_active_km_s=float(np.sqrt(np.mean((reduce(vp - imposed)[active.ravel()]) ** 2))))
    rel = pd.read_csv(folder / 'relocated_train_events.csv')
    true = p[['event_id', 'true_x_km', 'true_y_km', 'true_depth_km', 'injected_origin_s']].drop_duplicates('event_id')
    rel = rel.merge(true, on='event_id', validate='one_to_one')
    dr = rel[['x_km', 'y_km', 'depth_km']].to_numpy() - rel[['true_x_km', 'true_y_km', 'true_depth_km']].to_numpy()
    result['hypocenter_3d_rmse_km'] = float(np.sqrt(np.mean(np.sum(dr * dr, axis=1))))
    result['origin_shift_rmse_s'] = float(np.sqrt(np.mean((rel.origin_shift_s - rel.injected_origin_s) ** 2)))
    result['gate'] = dict(correlation_min=.7, amplitude_ratio_range=[.5, 1.5], relative_l2_max=.6)
    result['passes_gate'] = bool(corr >= .7 and .5 <= ratio <= 1.5 and result['relative_l2_error'] <= .6)
    np.savez(folder / 'nonlinear_recovery.npz', truth_vp=imposed, recovered_vp=vp, base_vp=base,
             active=active, xg=GRID.x, yg=GRID.y, zg=GRID.z,
             inversion_xg=COARSE.x, inversion_yg=COARSE.y, inversion_zg=COARSE.z)
    save(folder / 'recovery_metrics.json', result)
    return result


def main():
    OUT.mkdir(exist_ok=True)
    fine = refined(5)
    p, missing = load()
    missing_set = set(missing)
    base_key = lambda s: str(s).split('@')[0]
    stations = sorted(p.station.unique())
    cfg_gen = dict(CFG, cache_directory=str(OUT / 'synthetic_fields'), time_field_memory_limit_mb=192)
    rng = np.random.default_rng(20261011)
    events = p[['event_id', 'source_x_km', 'source_y_km', 'ev_prof_km']].drop_duplicates(
        'event_id').sort_values('event_id').reset_index(drop=True)
    events = events.rename(columns={'source_x_km': 'true_x_km', 'source_y_km': 'true_y_km',
                                    'ev_prof_km': 'true_depth_km'})
    jitter = rng.uniform(-5, 5, (len(events), 3))
    events[['source_x_km', 'source_y_km', 'ev_prof_km']] = events[
        ['true_x_km', 'true_y_km', 'true_depth_km']].to_numpy() + jitter
    la, lo = PR.latlon(events.source_x_km.to_numpy(), events.source_y_km.to_numpy())
    la = np.clip(la, DOMAIN['lat_min'], DOMAIN['lat_max'])
    lo = np.clip(lo, DOMAIN['lon_min'], DOMAIN['lon_max'])
    events['source_x_km'], events['source_y_km'] = PR.xy(la, lo)
    events['ev_lat'] = la
    events['ev_lon'] = lo
    events.ev_prof_km = events.ev_prof_km.clip(0, 180)
    events['injected_origin_s'] = rng.normal(0, .2, len(events))
    bias = dict(zip(stations, rng.normal(0, .15, len(stations))))
    bias[stations[0]] = 0.
    noise = rng.normal(0, .05, len(p))
    p = p.drop(columns=['source_x_km', 'source_y_km', 'ev_prof_km', 'ev_lat', 'ev_lon']).merge(
        events, on='event_id', how='left', validate='many_to_one')
    for amp in [.03, .08]:
        folder = OUT / f'synthetic_{int(amp * 100):02d}pct'
        folder.mkdir(exist_ok=True)
        if (folder / 'recovery_metrics.json').exists():
            continue
        path = folder / 'synthetic_observations.csv'
        if path.exists():
            q = pd.read_csv(path, low_memory=False)
        else:
            vp = truth(fine, amp)
            stations_df = p[['station', 'st_x_km', 'st_y_km', 'st_elev_km']].drop_duplicates()
            cache = FieldCache(vp, fine, stations_df, cfg_gen['cache_directory'], cfg_gen)
            true_p = p.copy()
            true_p['source_x_km'] = p.true_x_km
            true_p['source_y_km'] = p.true_y_km
            true_p['ev_prof_km'] = p.true_depth_km
            t = predict_exact(cache, true_p)
            cache.close()
            q = p.copy()
            q['true_time_s'] = t
            q['injected_station_s'] = q.station.map(bias)
            q['noise_s'] = noise
            q['observed_s'] = t + q.injected_origin_s + q.injected_station_s + noise
            q.to_csv(path, index=False)
        save(folder / 'generation.json', dict(
            generator_grid_km=5, inverse_grid_km=10, cell_grid_km=20, noise_sigma_s=.05,
            amplitude_fraction=amp, block_width_km=40, seed=20261011, all_stations=len(stations),
            events=q.event_id.nunique(), picks=len(q),
            source_position='exact true catalog; inverse catalog perturbed uniformly +/-5 km per axis',
            origin_sigma_s=.2, station_sigma_s=.15,
            station_elevation=True, station_elevation_method='first-order plane-wave t(z=-e)=t(z=0)+e*qz',
            stations_missing_elevation=missing,
            picks_at_stations_missing_elevation=int(q.station.map(lambda s: base_key(s) in missing_set).sum()),
            weights='same sigma=1 s as real-data experiment, not tuned using synthetic noise',
            script_sha256=digest(__file__),
            gate=dict(correlation_min=.7, amplitude_ratio_min=.5, amplitude_ratio_max=1.5,
                      relative_l2_max=.6)))
        print(f'SYNTHETIC {amp * 100:g}% inversion', flush=True)
        fit_pair(q, folder, dict(CFG, station_elevation=True,
                                 cache_directory=str(OUT / 'inverse_synthetic_fields'),
                                 time_field_memory_limit_mb=192))
        result = recovery(folder, amp, q)
        print(json.dumps(result), flush=True)
    save(OUT / 'nonlinear_synthetics.json', dict(experiments=[
        json.loads((OUT / f'synthetic_{int(a * 100):02d}pct/recovery_metrics.json').read_text())
        for a in [.03, .08]], generation_grid_km=5, inversion_grid_km=10,
        operator='nonlinear FSM data and newly recomputed fields/Jacobians each inverse round',
        not_shared_linear_forward_operator=True, station_elevation=True))
    print('synthetic controls completed:', OUT, flush=True)


if __name__ == '__main__':
    main()
