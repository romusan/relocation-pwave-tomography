"""Alternating P-only relocation / velocity updates with fixed held-out events."""
from pathlib import Path
from time import perf_counter
import json
import platform
import sys
import numpy as np
import pandas as pd
import scipy
import numba
import pyproj
from relocation import score_candidates
from .factored import FieldCache
from .geometry import candidate_nodes, gap_deg
from .factored import kernels, mapping, active_mask
from .inversion import station_design, regularized, linear_resolution, lcurve_diagnostic


def residual(cache, picks, nodes, origins, statics):
    pred = cache.predict(picks, nodes)
    correction = np.array([origins[e] + statics.get(s, 0.) for e, s in zip(picks.event_id, picks.station)])
    return picks.observed_s.to_numpy(float) - pred - correction


def rms(values):
    return float(np.sqrt(np.mean(np.asarray(values)**2))) if len(values) else None


def relocate(cache, picks, nodes, origins, statics, domain, projection, config):
    result = []
    new_nodes, new_origins = dict(nodes), dict(origins)
    for eid, g in picks.groupby('event_id', sort=False):
        candidates = candidate_nodes(cache.grid, nodes[eid], config['relocation_radius_km'], domain, projection)
        linear = cache.grid.linear(candidates)
        times = np.column_stack([cache.get(s).ravel()[linear] + cache.elevation_correction(s, candidates)
                                 + statics.get(s, 0.) for s in g.station])
        # Always score against the ORIGINAL observations; origin shift is an
        # absolute correction to stored UTC origin, never subtracted twice.
        best, shift, scores = score_candidates(times, g.observed_s.to_numpy(float), config['scoring_backend'])
        original_scores = scores.copy()
        allowed = abs(shift) <= config['max_origin_shift_s']
        scores = np.where(allowed, scores, np.inf)
        best = int(np.argmin(scores)) if np.isfinite(scores).any() else -1
        if best < 0:
            result.append({'event_id': eid, 'status': 'no_candidate_with_allowed_origin'})
            continue
        center = np.flatnonzero(np.all(candidates == nodes[eid], axis=1))[0]
        before = float(original_scores[center])
        new_nodes[eid] = tuple(candidates[best])
        new_origins[eid] = float(shift[best])
        xyz = cache.grid.xyz(candidates[best])
        lat, lon = projection.latlon(xyz[0], xyz[1])
        result.append({'event_id': eid, 'status': 'relocated', 'origin_shift_s': shift[best],
            'x_km': xyz[0], 'y_km': xyz[1], 'depth_km': xyz[2], 'lat': float(lat), 'lon': float(lon),
            'gap_deg': gap_deg(float(lat), float(lon), g.st_lat_current.to_numpy(), g.st_lon_current.to_numpy()),
            'rms_profiled_center_s': before, 'rms_profiled_best_s': scores[best],
            'displacement_from_previous_node_km': float(np.linalg.norm(xyz-cache.grid.xyz(nodes[eid])))})
    return new_nodes, new_origins, pd.DataFrame(result)


def run_velocity(picks, event_table, grid, coarse, profile, projection, domain, config, out, source_manifest, initial_model=None):
    started = perf_counter()
    out = Path(out)
    p = picks[picks.fase.eq('P')].copy().reset_index(drop=True)
    if p.empty:
        raise ValueError('No associated P observations')
    elevation_on = bool(config.get('station_elevation', False))
    if elevation_on and 'st_elev_km' not in p.columns:
        raise ValueError('station_elevation is enabled but picks lack st_elev_km')
    cols = ['station', 'st_x_km', 'st_y_km'] + (['st_elev_km'] if elevation_on else [])
    stations = p[cols].drop_duplicates()
    if stations.station.duplicated().any():
        raise ValueError('A station changes coordinates across selected epochs; split station IDs by epoch')
    nodes = {}
    for eid, g in p.groupby('event_id', sort=False):
        xyz = np.array([g.source_x_km.iloc[0], g.source_y_km.iloc[0], g.ev_prof_km.iloc[0]])
        rounded = grid.nodes(xyz)
        nearby = candidate_nodes(grid,rounded,grid.spacing,domain,projection)
        if len(nearby)==0:
            raise ValueError(f'No in-domain node near event {eid}')
        nearest = np.argmin(np.linalg.norm(grid.xyz(nearby)-xyz,axis=1))
        nodes[eid] = tuple(nearby[nearest])
    origins = {eid: 0. for eid in nodes}
    statics = {s: 0. for s in p.station.unique()}
    train = p[p.event_id.isin(event_table[event_table.split.eq('train')].event_id)].reset_index(drop=True)
    test = p[p.event_id.isin(event_table[event_table.split.eq('holdout')].event_id)].reset_index(drop=True)
    initial_nodes = dict(nodes)
    vz = np.interp(grid.z, *profile)
    vp = np.broadcast_to(vz[:, None, None], grid.shape).copy()
    if initial_model is not None:
        from .initial_model import load_initial_model
        vp = load_initial_model(initial_model, grid, domain)
    vp_initial = vp.copy()
    cache_directory = config.get('cache_directory',out/'cache')
    cache = FieldCache(vp, grid, stations, cache_directory, config)
    baseline = {'train_rms_fixed_catalog_s': rms(residual(cache, train, nodes, origins, statics)),
                'holdout_rms_fixed_catalog_origin_s': rms(residual(cache, test, initial_nodes, origins, statics))}
    history, last_rel = [], pd.DataFrame()
    S, free, reference_station = station_design(train, config['station_statics'])
    direct_to_coarse = mapping(grid, coarse)
    last_A = last_mask = None
    for round_id in range(1, config['rounds'] + 1):
        round_start = perf_counter()
        nodes, origins, last_rel = relocate(cache, train, nodes, origins, statics, domain, projection, config)
        last_rel.to_csv(out/f'relocation_round_{round_id}.csv', index=False)
        before = residual(cache, train, nodes, origins, statics)
        ray_start = perf_counter()
        G, ray_info = kernels(cache, train, nodes, coarse, config['ray_backend'])
        ray_seconds = perf_counter() - ray_start
        mask, hits = active_mask(G, coarse, domain, projection, config['min_ray_hits'])
        # A fixed observation population is required to compare updates. Failed
        # ray rows are retained with zero cell derivative; never dropped from RMS.
        active = np.flatnonzero(mask)
        if len(active)==0:
            np.savez(out/'coverage_no_active_cells.npz',ray_hits=hits,xg=coarse.x,yg=coarse.y,zg=coarse.z)
            (out/'run_status.json').write_text(json.dumps({'status':'insufficient_coverage',
                'round':round_id,'min_ray_hits':config['min_ray_hits'],'max_ray_hits':int(hits.max()),
                'message':'No active cells on spatially balanced selection. Increase pilot size; do not silently lower the scientific threshold.'},indent=2),encoding='utf-8')
        A = G[:, active]
        solve_start = perf_counter()
        dslow, dstatic, solver_info = regularized(A, before, mask, S,
            np.full(len(train), config['sigma_p_s']), config)
        solve_seconds = perf_counter() - solve_start
        full_delta = np.zeros(int(np.prod(coarse.shape)))
        full_delta[active] = dslow
        delta_direct = full_delta[direct_to_coarse].reshape(grid.shape)
        old_statics = dict(statics)
        accepted = False
        trials = []
        previous_cache_stats = cache.stats()
        previous_rms = rms(before)
        for alpha in (1., .5, .25):
            slowness = np.maximum(1/vp + alpha * delta_direct, 1/config['vp_max_km_s'])
            trial = 1 / slowness
            frac = config['max_velocity_fraction_per_round']
            trial = np.clip(trial, np.maximum(config['vp_min_km_s'], vp*(1-frac)),
                            np.minimum(config['vp_max_km_s'], vp*(1+frac)))
            trial_statics = dict(old_statics)
            for sta, step in zip(free, dstatic):
                trial_statics[sta] += alpha * float(np.clip(step, -config['max_static_step_s'], config['max_static_step_s']))
            trial_cache = FieldCache(trial, grid, stations, cache_directory, config)
            after = residual(trial_cache, train, nodes, origins, trial_statics)
            trial_rms = rms(after)
            trials.append({'alpha': alpha, 'train_rms_s': trial_rms, 'cache': trial_cache.stats()})
            if trial_rms <= previous_rms + 1e-10:
                accepted = True
                vp, statics, cache = trial, trial_statics, trial_cache
                break
        # Held-out event locations and origins are never optimized. Association
        # is fixed before split and uses only the initial 1-D profile.
        holdout_res = test.observed_s.to_numpy()-cache.predict_exact(test)-test.station.map(statics).to_numpy()
        row = {'round': round_id, 'train_rms_before_update_s': previous_rms,
               'train_rms_after_update_s': rms(residual(cache, train, nodes, origins, statics)),
               'holdout_rms_fixed_catalog_origin_s': rms(holdout_res),
               'active_cells': len(active), 'train_equations': len(train),
               'update_accepted': accepted, 'line_search': trials,
               'lsmr': solver_info, 'ray_diagnostic': ray_info,
               'cache_before_update': previous_cache_stats,
               'ray_seconds': ray_seconds, 'solve_seconds': solve_seconds,
               'round_seconds': perf_counter()-round_start,
               'max_abs_station_static_s': float(max(abs(v) for v in statics.values()))}
        history.append(row)
        print(f"Round {round_id}: RMS {previous_rms:.4f} -> {row['train_rms_after_update_s']:.4f} s, active={len(active)}, accepted={accepted}, holdout={row['holdout_rms_fixed_catalog_origin_s']}", flush=True)
        (out/'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        np.savez(out/f'model_round_{round_id}.npz', vp=vp, active=mask, ray_hits=hits,
                 xg=grid.x, yg=grid.y, zg=grid.z, inversion_xg=coarse.x, inversion_yg=coarse.y, inversion_zg=coarse.z)
        last_A, last_mask = A, mask
    final_G, final_ray_info = kernels(cache, train, nodes, coarse, config['ray_backend'])
    last_mask, final_hits = active_mask(final_G, coarse, domain, projection, config['min_ray_hits'])
    last_A = final_G[:, np.flatnonzero(last_mask)]
    diagnostic={'kind':'not run: nonlinear campaign'}
    lcurve=pd.DataFrame({'converged':[]})
    pred_output = p.copy()
    pred_output['split'] = np.where(pred_output.event_id.isin(train.event_id), 'train', 'holdout')
    pred_output['predicted_s'] = cache.predict(p, nodes)
    held=pred_output.split.eq('holdout')
    pred_output.loc[held,'predicted_s']=cache.predict_exact(pred_output[held].reset_index(drop=True))
    pred_output['origin_shift_s'] = pred_output.event_id.map(origins)
    pred_output['station_static_s'] = pred_output.station.map(statics)
    pred_output['residual_s'] = (pred_output.observed_s - pred_output.predicted_s -
                               pred_output.origin_shift_s - pred_output.station_static_s)
    pred_output.to_csv(out/'predictions.csv', index=False)
    last_rel.to_csv(out/'relocated_train_events.csv', index=False)
    pd.DataFrame({'station': list(statics), 'static_s': list(statics.values())}).to_csv(out/'station_statics.csv', index=False)
    np.savez(out/'vp_final.npz', vp=vp, vp_initial=vp_initial, xg=grid.x, yg=grid.y, zg=grid.z,
             active=last_mask, inversion_xg=coarse.x, inversion_yg=coarse.y, inversion_zg=coarse.z)
    summary = {'status': 'factored_FM_actual_station_elevation', 'phases_in_velocity_inversion': ['P'],
        'train_events': train.event_id.nunique(), 'holdout_events': test.event_id.nunique(),
        'train_picks': len(train), 'holdout_picks': len(test), 'direct_shape': list(grid.shape),
        'inversion_shape': list(coarse.shape), 'baseline': baseline, 'history': history,
        'station_static_reference': reference_station, 'linear_resolution': diagnostic,
        'final_ray_diagnostic': final_ray_info,
        'linear_lcurve': {'file':'linear_lcurve.csv','solutions':len(lcurve),
            'converged_solutions':int(lcurve.converged.sum()),
            'limitation':'linearized sensitivity at final model; no retuning on holdout'},
        'projection_audit': projection.audit(domain), 'config': config, 'domain': domain,
        'station_elevation_km': {'enabled': True,'method':'receiver at continuous XYZ, depth=-elevation; factored station-centred mesh'},
        'sources': source_manifest, 'seconds_total': perf_counter()-started,
        'environment': {'python': sys.version, 'platform': platform.platform(),
            'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__,
            'numba': numba.__version__, 'pyproj': pyproj.__version__},
        'limitations': ['Station-centred shifted grids use interpolated slowness; adjoint differentiates this interpolation.', 'Negative depth is extrapolated surface rock, not explicit topography or Earth curvature.', 'Relocation searches 10-km grid nodes; heldout predictions use continuous hypocentres.', 'Mesh and nonlinear resolution are independent gates.']}
    summary['projection_audit']['depth_convention']='positive down; stations at depth=-elevation_km'
    (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary, cache, nodes, statics
