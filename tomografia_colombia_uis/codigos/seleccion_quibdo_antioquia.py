"""Add 300 Choco/Antioquia events: deep + Quibdo + geographic balance."""
import numpy as np
import pandas as pd
from regional.inputs import spatial_order


def extend_cohort(base, available, catalog, projection, number=300, seed=20261009):
    if number!=300 or len(base)!=1900: raise ValueError('Requires 1900 base + 300 additions')
    for d in (base,available,catalog):
        if d.event_id.duplicated().any(): raise ValueError('Duplicate event identifiers')
    if not set(base.split).issubset({'train','holdout'}): raise ValueError('Invalid previous split')
    d=available[~available.event_id.isin(base.event_id)].merge(
        catalog[['event_id','region']],on='event_id',validate='one_to_one')
    target=d[d.region.str.contains(r' - (?:Choc.|Antioquia), Colombia$',regex=True,na=False)].copy()
    if len(target)<number: raise ValueError('Insufficient target observations')
    chosen=[]
    # Priorities use catalog geography/depth, not velocity or plate assignment.
    for pool in [target[target.depth_km>=40],
                 target[target.region.str.match(r'^Quibd. - Choc., Colombia$',na=False)]]:
        chosen.extend(spatial_order(pool[~pool.event_id.isin(chosen)],projection))
    near=target[target.lat.between(4.8,6.8)&target.lon.between(-77.3,-75.3)]
    near_chosen=int(near.event_id.isin(chosen).sum())
    chosen.extend(spatial_order(near[~near.event_id.isin(chosen)],projection)[:max(0,100-near_chosen)])
    chosen.extend(spatial_order(target[~target.event_id.isin(chosen)],projection)[:number-len(chosen)])
    if len(chosen)!=number or len(set(chosen))!=number: raise ValueError('Invalid target count')
    extra=target.set_index('event_id').loc[chosen].reset_index()
    extra['addition_group']=np.where(extra.region.str.contains(' - Antioquia, Colombia'), 'Antioquia','Choco')
    extra['in_quibdo_window']=extra.lat.between(4.8,6.8)&extra.lon.between(-77.3,-75.3)
    # Stratify holdout by depth: 20% of each predeclared depth group.
    rng=np.random.default_rng(seed); held=set()
    for pool in (extra[extra.depth_km>=40],extra[extra.depth_km<40]):
        held.update(rng.permutation(pool.event_id.to_numpy())[:round(len(pool)*.2)])
    extra['split']=np.where(extra.event_id.isin(held),'holdout','train')
    events=pd.concat([base,extra],ignore_index=True)
    if events.event_id.duplicated().any(): raise ValueError('Duplicated cohort')
    np.testing.assert_array_equal(events.set_index('event_id').loc[base.event_id,'split'],base.split)
    return events,extra,target
