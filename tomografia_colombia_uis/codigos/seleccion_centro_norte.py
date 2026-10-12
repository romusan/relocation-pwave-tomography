"""100 central deep events + 300 northern events; preserve previous partition."""
from pathlib import Path
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from regional.inputs import spatial_order

ROOT = Path(__file__).resolve().parents[1]
MESH = ROOT/'resultados/benioff_3d_1500_norte_2026-10-09/fragmentos_geometricos.npz'


def extend_cohort(base, available, catalog, projection, number=400, seed=20261009):
    if number != 400:
        raise ValueError('This experiment requires 100 central + 300 northern additions')
    for d in (base, available, catalog):
        if d.event_id.duplicated().any(): raise ValueError('Duplicate event identifiers')
    if len(base)!=1500 or not set(base.split).issubset({'train','holdout'}):
        raise ValueError('Requires the verified 1500-event cohort')
    d=available[~available.event_id.isin(base.event_id)].merge(
        catalog[['event_id','region']],on='event_id',validate='one_to_one')
    center=d[d.lat.between(4.5,7.2)&d.lon.between(-76.7,-74.3)&(d.depth_km>=40)].copy()
    with np.load(MESH) as m:
        cover=unary_union([Polygon(t[:,:2]) for t in m['xyz'][m['triangles']]])
    x,y=projection.xy(center.lat,center.lon)
    center['distance_to_previous_mesh_km']=[Point(xx,yy).distance(cover) for xx,yy in zip(x,y)]
    unsupported=center[center.distance_to_previous_mesh_km>5]
    order=spatial_order(unsupported,projection)+spatial_order(
        center[~center.event_id.isin(unsupported.event_id)],projection)
    if len(order)<100: raise ValueError('Fewer than 100 central deep candidates')
    central=center.set_index('event_id').loc[order[:100]].reset_index()
    central['addition_group']='Centro profundo: 4.5–7.2 N, -76.7 a -74.3'
    north=d[d.lat.between(7,9)&~d.event_id.isin(central.event_id)].copy()
    if len(north)<300: raise ValueError('Fewer than 300 northern candidates')
    northern=north.set_index('event_id').loc[spatial_order(north,projection)[:300]].reset_index()
    northern['addition_group']='Norte 7–9 N: segunda ampliacion'
    # Fixed 20% holdout separately within each predefined geographic group.
    rng=np.random.default_rng(seed)
    groups=[]
    for extra in (central,northern):
        held=set(rng.permutation(extra.event_id.to_numpy())[:round(len(extra)*.2)])
        extra['split']=np.where(extra.event_id.isin(held),'holdout','train')
        groups.append(extra)
    extra=pd.concat(groups,ignore_index=True)
    events=pd.concat([base,extra],ignore_index=True)
    if events.event_id.duplicated().any(): raise ValueError('Duplicated extension')
    np.testing.assert_array_equal(events.set_index('event_id').loc[base.event_id,'split'],base.split)
    eligible=pd.concat([center.assign(selection_zone='central'),north.assign(selection_zone='north')])
    return events,extra,eligible
