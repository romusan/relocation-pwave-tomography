"""Deterministic association, station epochs and unchanged raw observations."""
from pathlib import Path
from collections import deque
import hashlib
import numpy as np
import pandas as pd
from .geometry import gap_deg


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load_station_elevations(paths):
    """Combine station inventories (net,sta,elev in meters) into km per base key.

    Epoch suffixes ('NET.STA@hash') are stripped by attach_station_elevations.
    Later inventories win on duplicate keys; rows without elevation are dropped.
    """
    inv = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    inv['station'] = inv.net.astype(str) + '.' + inv.sta.astype(str)
    inv = inv.dropna(subset=['elev']).drop_duplicates('station', keep='last')
    return dict(zip(inv.station, inv.elev.to_numpy(float) / 1000.))


def attach_station_elevations(picks, elevation_km):
    """Add st_elev_km to picks by station key; report keys without elevation.

    Stations missing from the inventory get 0 km and are returned as missing,
    so a campaign can count and disclose them instead of failing silently.
    """
    picks = picks.copy()
    base = picks.station.astype(str).str.replace(r'@.*$', '', regex=True)
    picks['st_elev_km'] = base.map(elevation_km).fillna(0.0).to_numpy(float)
    return picks, sorted(set(base) - set(elevation_km))


def load_profile(path):
    d = pd.read_csv(path).sort_values('depth_km')
    z, v = d.depth_km.to_numpy(float), d.vp_km_s.to_numpy(float)
    if len(z) < 2 or not np.isfinite(z).all() or not np.isfinite(v).all() or (v <= 0).any() or (np.diff(z) <= 0).any():
        raise ValueError('Profile needs increasing unique depths and finite positive Vp')
    return z, v


def straight_time(xyz, stations, profile):
    # Only a broad association prediction; this is NOT a refracted ray.
    fraction = np.linspace(0, 1, 33)
    z = xyz[:, 2, None] * (1 - fraction)
    slow = 1 / np.interp(z, *profile)
    mean_slow = np.trapezoid(slow, fraction, axis=1)
    return np.linalg.norm(xyz - stations, axis=1) * mean_slow


def active_stations(table, time):
    begin = pd.to_datetime(table.inicio, utc=True, errors='coerce', format='mixed')
    end = pd.to_datetime(table.fin, utc=True, errors='coerce', format='mixed')
    t = pd.Timestamp(str(time))
    t = t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')
    active = table[(begin.isna() | (begin <= t)) & (end.isna() | (end >= t))].copy()
    return active.sort_values('inicio', kind='stable').drop_duplicates(['net', 'sta'], keep='last')


def associate(raw, stations, domain, projection, profile, config):
    required = ['event_id', 'origin_time', 'ev_lat', 'ev_lon', 'ev_prof_km',
                'net', 'sta', 'fase', 'pick_time', 'prob']
    if set(required) - set(raw):
        raise ValueError(f'Missing columns: {sorted(set(required) - set(raw))}')
    d = raw.copy().reset_index(drop=True)
    d.insert(0, 'raw_row', np.arange(len(d)))
    d['reason'] = ''
    d['selected'] = False
    d['station'] = d.net.astype(str) + '.' + d.sta.astype(str)
    for c in ('ev_lat', 'ev_lon', 'ev_prof_km', 'prob'):
        d[c] = pd.to_numeric(d[c], errors='coerce')
    pick = pd.to_datetime(d.pick_time, utc=True, errors='coerce', format='mixed')
    origin = pd.to_datetime(d.origin_time, utc=True, errors='coerce', format='mixed')
    d['observed_s'] = (pick - origin).dt.total_seconds()
    invalid = (~np.isfinite(d[['ev_lat', 'ev_lon', 'ev_prof_km', 'prob', 'observed_s']]).all(axis=1)
               | ~d.prob.between(0, 1) | d[['event_id', 'net', 'sta']].isna().any(axis=1))
    d.loc[invalid, 'reason'] = 'invalid_input'
    outside = ~(d.ev_lat.between(domain['lat_min'], domain['lat_max']) &
                d.ev_lon.between(domain['lon_min'], domain['lon_max']) &
                d.ev_prof_km.between(domain['prof_min'], domain['prof_max']))
    d.loc[(d.reason == '') & outside, 'reason'] = 'outside_domain'
    d.loc[(d.reason == '') & (d.observed_s <= 0), 'reason'] = 'nonpositive_travel_time'
    duplicate = d.duplicated(subset=required, keep='first')
    d.loc[(d.reason == '') & duplicate, 'reason'] = 'duplicate'
    d.loc[(d.reason == '') & ~d.fase.isin(['P', 'S']), 'reason'] = 'unsupported_phase'
    threshold = np.where(d.fase == 'P', config['min_probability_p'], config['min_probability_s'])
    d.loc[(d.reason == '') & (d.prob < threshold), 'reason'] = 'below_detection_threshold'
    d['st_x_km'] = np.nan
    d['st_y_km'] = np.nan
    d['st_lat_current'] = np.nan
    d['st_lon_current'] = np.nan
    d['st_elev_km'] = 0.0
    # Vectorized epoch matching avoids reparsing 203 stations per event.
    d['_origin_utc'] = origin
    info = d[d.reason.eq('')].groupby('event_id')[['_origin_utc','ev_lat','ev_lon','ev_prof_km']].nunique()
    conflict = info.index[(info > 1).any(axis=1)]
    d.loc[d.reason.eq('') & d.event_id.isin(conflict), 'reason'] = 'inconsistent_event_origin'
    meta = stations[['net','sta','lat','lon','inicio','fin']].copy()
    meta['_station_key'] = meta.net.astype(str)+'.'+meta.sta.astype(str)
    repeated = meta.duplicated(['net','sta'],keep=False)
    for index in meta.index[repeated]:
        row = meta.loc[index]
        epoch_id = hashlib.sha256(str((row.inicio,row.fin,row.lat,row.lon)).encode()).hexdigest()[:12]
        meta.loc[index,'_station_key'] += '@'+epoch_id
    meta['_begin'] = pd.to_datetime(meta.inicio, utc=True, errors='coerce', format='mixed')
    meta['_end'] = pd.to_datetime(meta.fin, utc=True, errors='coerce', format='mixed')
    for key, parsed in [('inicio','_begin'),('fin','_end')]:
        if (meta[key].notna() & meta[key].astype(str).str.strip().ne('') & meta[parsed].isna()).any():
            raise ValueError(f'Malformed station epoch: {key}')
    matches = d[d.reason.eq('')][['raw_row','net','sta','_origin_utc']].merge(meta,on=['net','sta'],how='inner')
    matches = matches[(matches._begin.isna() | (matches._begin <= matches._origin_utc)) &
                      (matches._end.isna() | (matches._end >= matches._origin_utc))]
    matches = matches.sort_values(['raw_row','_begin'],kind='stable',na_position='first').drop_duplicates('raw_row',keep='last').set_index('raw_row')
    good_ids = matches.index.to_numpy(int)
    d.loc[d.reason.eq('') & ~d.raw_row.isin(good_ids),'reason'] = 'station_inactive_or_missing'
    sx,sy = projection.xy(matches.lat.to_numpy(float), matches.lon.to_numpy(float))
    if not np.isfinite(sx).all() or not np.isfinite(sy).all():
        raise ValueError('Invalid station coordinates')
    d.loc[good_ids,'st_x_km'] = sx
    d.loc[good_ids,'st_y_km'] = sy
    d.loc[good_ids,'st_lat_current'] = matches.lat.to_numpy(float)
    d.loc[good_ids,'st_lon_current'] = matches.lon.to_numpy(float)
    if 'elev' in matches.columns:
        d.loc[good_ids,'st_elev_km'] = matches.elev.to_numpy(float) / 1000.
    d.loc[good_ids,'station'] = matches._station_key.to_numpy(str)
    d.drop(columns='_origin_utc',inplace=True)
    good = d.reason.eq('')
    x, y = projection.xy(d.loc[good, 'ev_lat'], d.loc[good, 'ev_lon'])
    d.loc[good, 'source_x_km'] = x
    d.loc[good, 'source_y_km'] = y
    xyz = d.loc[good, ['source_x_km', 'source_y_km', 'ev_prof_km']].to_numpy(float)
    # Receivers sit at their topographic height: z = -elevation (positive down).
    rec = np.column_stack([d.loc[good, 'st_x_km'], d.loc[good, 'st_y_km'],
                           -d.loc[good, 'st_elev_km']])
    pred = straight_time(xyz, rec, profile)
    pred *= np.where(d.loc[good, 'fase'].eq('S'), config['vp_vs_association'], 1.)
    d.loc[good, 'association_prediction_s'] = pred
    d.loc[good, 'association_residual_s'] = d.loc[good, 'observed_s'] - pred
    window = np.where(d.fase.eq('P'),config['association_window_p_s'],config['association_window_s_s'])
    outside_window = d.reason.eq('') & (d.association_residual_s.abs()>window)
    d.loc[outside_window,'reason'] = 'outside_association_window'
    eligible = d[d.reason.eq('')].copy()
    eligible['_abs'] = eligible.association_residual_s.abs()
    keys = ['event_id','station','fase']
    eligible = eligible.sort_values([*keys,'_abs','prob','raw_row'],
        ascending=[True,True,True,True,False,True],kind='stable')
    eligible['_group'] = eligible.groupby(keys,sort=False).ngroup()
    eligible['_rank'] = eligible.groupby('_group',sort=False).cumcount()
    first = eligible[eligible._rank.eq(0)].set_index('_group',drop=False)
    second = eligible[eligible._rank.eq(1)].set_index('_group',drop=False)
    paired = first[['_abs','observed_s']].join(second[['_abs','observed_s']],lsuffix='_first',rsuffix='_second')
    ambiguous = paired.index[((paired._abs_second-paired._abs_first)<config['ambiguity_margin_s']) &
                              ((paired.observed_s_second-paired.observed_s_first).abs()>.2)]
    d.loc[eligible.raw_row,'reason'] = 'alternative_candidate'
    d.loc[eligible[eligible._group.isin(ambiguous)].raw_row,'reason'] = 'ambiguous_candidates'
    winners = first[~first._group.isin(ambiguous)].raw_row.to_numpy(int)
    d.loc[winners,'selected'] = True
    d.loc[winners,'reason'] = 'retained'
    chosen = d[d.selected]
    ptime = chosen[chosen.fase.eq('P')][['event_id','station','observed_s']]
    stime = chosen[chosen.fase.eq('S')][['event_id','station','observed_s','raw_row']]
    ordered = stime.merge(ptime,on=['event_id','station'],suffixes=('_S','_P'))
    invalid_s = ordered[ordered.observed_s_S<=ordered.observed_s_P].raw_row.to_numpy(int)
    d.loc[invalid_s,'selected'] = False
    d.loc[invalid_s,'reason'] = 'S_not_after_P'
    p = d[d.selected & d.fase.eq('P')]
    counts = p.groupby('event_id').station.nunique()
    valid_ids = counts[counts >= config['min_p_stations']].index
    d.loc[d.selected & ~d.event_id.isin(valid_ids), ['selected', 'reason']] = [False, 'insufficient_distinct_P_stations']
    retained = d[d.selected].copy()
    events = []
    for eid, g in retained[retained.fase.eq('P')].groupby('event_id', sort=False):
        events.append({'event_id': eid, 'n_p_stations': g.station.nunique(),
            'lat': g.ev_lat.iloc[0], 'lon': g.ev_lon.iloc[0], 'depth_km': g.ev_prof_km.iloc[0],
            'gap_deg': gap_deg(g.ev_lat.iloc[0], g.ev_lon.iloc[0], g.st_lat_current.to_numpy(), g.st_lon_current.to_numpy())})
    return retained, d, pd.DataFrame(events, columns=['event_id', 'n_p_stations', 'lat', 'lon', 'depth_km', 'gap_deg'])


def spatial_order(events, projection, cell_km=25.):
    x, y = projection.xy(events.lat, events.lon)
    cells = list(zip(np.floor(x / cell_km).astype(int), np.floor(y / cell_km).astype(int),
                     np.floor(events.depth_km / cell_km).astype(int)))
    e = events.assign(cell=cells).sort_values(['gap_deg', 'event_id'], kind='stable')
    groups = list(e.groupby('cell',sort=True))
    centers = np.asarray([key for key,_ in groups],float)
    if len(centers)==0:
        return []
    # Lexicographic cell order makes a short pilot concentrate in the west.
    # Farthest-point traversal gives short prefixes spatial spread as well as
    # round-robin density balance. Distances use the same 25 km cell units.
    remaining = np.ones(len(centers),bool)
    next_cell = int(np.argmin(np.sum((centers-centers.mean(axis=0))**2,axis=1)))
    closest = np.full(len(centers),np.inf)
    cell_order=[]
    for _ in range(len(centers)):
        cell_order.append(next_cell)
        remaining[next_cell]=False
        closest=np.minimum(closest,np.sum((centers-centers[next_cell])**2,axis=1))
        next_cell=int(np.argmax(np.where(remaining,closest,-1.)))
    queues = [deque(groups[i][1].event_id) for i in cell_order]
    result = []
    while queues:
        for q in queues:
            result.append(q.popleft())
        queues = [q for q in queues if q]
    return result
