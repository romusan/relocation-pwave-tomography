"""Measured spectral-pair Q inversion using final Vp and station-based rays.

Sign convention: delta_tstar = integral_i(1/Q * dl/Vp) - integral_j(...)
+ site_i - site_j. This module does not infer Q from pick travel times.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.linalg import cholesky, solve_triangular
from .inputs import sha256
from .rays import kernels
from .inversion import active_mask, regularized


def covariance_whitener(covariance, indices):
    selected = np.asarray(covariance)[np.ix_(indices, indices)]
    if not np.isfinite(selected).all() or not np.allclose(selected, selected.T, atol=1e-12):
        raise ValueError('Q covariance must be finite and symmetric')
    lower = cholesky(selected, lower=True)
    return solve_triangular(lower, np.eye(len(indices)), lower=True)


def run_q(path, picks, event_table, cache, nodes, coarse, domain, projection, config, out, covariance_path=None):
    out = Path(out)/'Q'
    out.mkdir(exist_ok=False)
    raw = pd.read_csv(path)
    required = {'event_id', 'station_i', 'station_j', 'delta_tstar_s', 'sigma_delta_tstar_s', 'qc_ok'}
    if required - set(raw):
        raise ValueError(f'Q pairs missing columns: {sorted(required-set(raw))}')
    raw['pair_row'] = np.arange(len(raw))
    raw['reason'] = ''
    if raw[list(required)].isna().any().any():
        raise ValueError('Missing values in required Q columns')
    for c in ('delta_tstar_s', 'sigma_delta_tstar_s'):
        raw[c] = pd.to_numeric(raw[c], errors='coerce')
    valid_numbers = np.isfinite(raw.delta_tstar_s) & np.isfinite(raw.sigma_delta_tstar_s) & (raw.sigma_delta_tstar_s > 0)
    raw.loc[~valid_numbers, 'reason'] = 'invalid_measurement_or_sigma'
    raw.loc[(raw.reason == '') & ~raw.qc_ok.astype(str).str.lower().isin(['true','1','yes']), 'reason'] = 'spectral_qc_failed'
    raw.loc[(raw.reason == '') & (raw.station_i == raw.station_j), 'reason'] = 'identical_endpoints'
    allowed_stations = set(cache.station_nodes)
    raw.loc[(raw.reason == '') & (~raw.event_id.isin(nodes) | ~raw.station_i.isin(allowed_stations) |
                                  ~raw.station_j.isin(allowed_stations)), 'reason'] = 'outside_velocity_dataset'
    raw['_pair'] = [tuple(sorted([i,j])) for i,j in zip(raw.station_i, raw.station_j)]
    duplicate = raw.duplicated(['event_id', '_pair'], keep='first')
    raw.loc[(raw.reason == '') & duplicate, 'reason'] = 'duplicate_or_reversed_pair'
    if covariance_path is None:
        # Reuse of one waveform in several pairs creates correlated errors.
        # Without covariance retain deterministic disjoint pairs per event.
        for _, g in raw[raw.reason.eq('')].groupby('event_id', sort=False):
            used = set()
            for row in g.itertuples():
                if row.station_i in used or row.station_j in used:
                    raw.loc[row.Index, 'reason'] = 'shared_waveform_covariance_missing'
                else:
                    used.update([row.station_i, row.station_j])
    raw.drop(columns='_pair').to_csv(out/'pairs_qc.csv', index=False)
    pairs = raw[raw.reason.eq('')].copy().reset_index(drop=True)
    if pairs.empty:
        raise ValueError('No Q measurements survived explicit pair QC')
    rows = pd.concat([pairs[['event_id','station_i']].rename(columns={'station_i':'station'}),
                      pairs[['event_id','station_j']].rename(columns={'station_j':'station'})], ignore_index=True)
    endpoints = rows.drop_duplicates().reset_index(drop=True)
    E, ray_info = kernels(cache, endpoints, nodes, coarse, config['ray_backend'], q_mode=True)
    position = {(r.event_id,r.station): i for i,r in enumerate(endpoints.itertuples())}
    ia = [position[(r.event_id,r.station_i)] for r in pairs.itertuples()]
    ib = [position[(r.event_id,r.station_j)] for r in pairs.itertuples()]
    G = E[ia] - E[ib]
    failed_endpoints = set(ray_info['failed_rows'])
    valid_ray = np.array([a not in failed_endpoints and b not in failed_endpoints for a,b in zip(ia,ib)])
    raw.loc[pairs.loc[~valid_ray,'pair_row'],'reason'] = 'ray_failure'
    raw.drop(columns='_pair').to_csv(out/'pairs_qc.csv',index=False)
    pairs = pairs[valid_ray].reset_index(drop=True)
    G = G[valid_ray]
    if pairs.empty:
        raise ValueError('No Q pairs with two valid endpoint rays')
    is_train = pairs.event_id.isin(event_table[event_table.split.eq('train')].event_id).to_numpy()
    train_indices = np.flatnonzero(is_train)
    if len(train_indices) == 0:
        raise ValueError('No Q pairs for training events')
    train_G = G[train_indices]
    mask, hits = active_mask(train_G, coarse, domain, projection, config['min_ray_hits'])
    active = np.flatnonzero(mask)
    station_ids = sorted(set(pairs.loc[is_train,'station_i']) | set(pairs.loc[is_train,'station_j']))
    station_position = {s:i for i,s in enumerate(station_ids)}
    ga = [station_position[s] for s in pairs.loc[is_train,'station_i']]
    gb = [station_position[s] for s in pairs.loc[is_train,'station_j']]
    graph = sparse.csr_matrix((np.ones(len(ga)),(ga,gb)),shape=(len(station_ids),len(station_ids)))
    ncomponents, labels = connected_components(graph,directed=False)
    references = [station_ids[int(np.flatnonzero(labels==label)[0])] for label in range(ncomponents)]
    free = [s for s in station_ids if s not in references]
    sta_pos = {s:i for i,s in enumerate(free)}
    rr, cc, vv = [], [], []
    for row, pair in enumerate(pairs.itertuples()):
        for station, sign in ((pair.station_i, 1.), (pair.station_j, -1.)):
            if station in sta_pos:
                rr.append(row); cc.append(sta_pos[station]); vv.append(sign)
    S = sparse.csr_matrix((vv,(rr,cc)),shape=(len(pairs),len(free)))
    qinv0 = np.full(int(np.prod(coarse.shape)), 1/config['q_initial'])
    pred0 = G @ qinv0
    obs = pairs.delta_tstar_s.to_numpy(float)
    b = obs - pred0
    W = None
    if covariance_path:
        covariance = np.load(covariance_path, allow_pickle=False)
        if covariance.shape != (len(raw),len(raw)):
            raise ValueError('Covariance shape must match ORIGINAL Q CSV rows')
        diagonal = np.diag(covariance)[pairs.pair_row.to_numpy(int)]
        if not np.allclose(diagonal,pairs.sigma_delta_tstar_s.to_numpy(float)**2,rtol=1e-5,atol=1e-12):
            raise ValueError('Covariance diagonal disagrees with supplied delta_tstar sigma')
        W = covariance_whitener(covariance, pairs.loc[is_train, 'pair_row'].to_numpy(int))
    dq, site, info = regularized(train_G[:,active], b[is_train], mask, S[train_indices],
        pairs.loc[is_train,'sigma_delta_tstar_s'].to_numpy(float), config, q=True, whitener=W)
    qinv = qinv0.copy()
    qinv[active] = np.clip(qinv[active]+dq, 1/config['q_max'], 1/config['q_min'])
    pred1 = G @ qinv + S @ site  # recompute after bounds, not unclipped dq
    pairs['predicted_initial_s'] = pred0
    pairs['predicted_final_s'] = pred1
    pairs['residual_s'] = obs-pred1
    pairs['split'] = np.where(is_train,'train','holdout')
    pairs.to_csv(out/'predictions.csv',index=False)
    np.savez(out/'q_final.npz',q=(1/qinv).reshape(coarse.shape),active=mask,ray_hits=hits,
             xg=coarse.x,yg=coarse.y,zg=coarse.z)
    pd.DataFrame({'station':[*references,*free],'site_tstar_s':[*([0.]*len(references)),*site]}).to_csv(out/'site_terms.csv',index=False)
    def rms(v):
        return float(np.sqrt(np.mean(v*v))) if len(v) else None
    summary = {'status':'experimental_measured_spectral_pairs','raw_pairs':len(raw),'retained_pairs':len(pairs),
        'train_pairs':int(is_train.sum()),'holdout_pairs':int((~is_train).sum()),'active_cells':len(active),
        'train_rms_before_s':rms(b[is_train]),'train_rms_after_s':rms((obs-pred1)[is_train]),
        'holdout_rms_before_s':rms(b[~is_train]),'holdout_rms_after_s':rms((obs-pred1)[~is_train]),
        'lsmr':info,'ray_diagnostic':ray_info,'site_reference_stations':references,
        'covariance':'provided' if covariance_path else 'disjoint pairs per event, diagonal supplied uncertainty',
        'input_sha256':sha256(path), 'covariance_sha256':sha256(covariance_path) if covariance_path else None,
        'sign':'delta_tstar_i_minus_j = integral_i - integral_j + site_i - site_j',
        'limitation':'Requires independently QCed spectra/response correction; this module does not create spectral measurements.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary
