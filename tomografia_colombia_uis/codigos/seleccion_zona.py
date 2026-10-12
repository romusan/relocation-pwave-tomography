"""Targeted additions with unchanged base cohort and train/holdout membership."""
import numpy as np
import pandas as pd
from regional.inputs import spatial_order


def extend_cohort(base, available, catalog, projection, number=200, seed=20261009):
    if base.event_id.duplicated().any() or catalog.event_id.duplicated().any():
        raise ValueError('Duplicate event identifiers in base or catalog')
    if not set(base.split).issubset({'train','holdout'}):
        raise ValueError('Base splits must be train/holdout')
    if not set(base.event_id).issubset(set(available.event_id)):
        raise ValueError('Some previous events no longer have associated observations')
    if number <= 0 or int(number) != number:
        raise ValueError('Number of additional events must be a positive integer')
    candidates=available[~available.event_id.isin(base.event_id)].merge(
        catalog[['event_id','region']],on='event_id',validate='one_to_one')
    dept=candidates.region.str.extract(r' - (Valle del Cauca|Cauca), Colombia$',expand=False)
    strict=candidates[dept.notna()].copy()
    strict['addition_group']=dept[dept.notna()]
    order=spatial_order(strict,projection)
    extra=strict.set_index('event_id').loc[order[:number]].reset_index()
    needed=number-len(extra)
    if needed:
        # At most ONE explicitly labeled nearby coastal event; never invent a
        # department assignment or weaken the original observation quality gate.
        coast=candidates[candidates.region.str.contains('Pac.fico|Costa Oeste',case=False,na=False)
            & candidates.lat.between(3.2,4.2) & candidates.lon.between(-78.5,-77.1)].copy()
        coast=coast[~coast.event_id.isin(extra.event_id)]
        if needed != 1 or coast.empty:
            raise ValueError(f'Only {len(strict)} new department events; cannot complete {number} with the allowed one coastal supplement')
        coast['addition_group']='Pacifico cercano a Valle (etiqueta separada)'
        # Quality ranking independent of velocity anomaly and plate hypothesis.
        coast=coast.sort_values(['gap_deg','n_p_stations','event_id'],ascending=[True,False,True])
        extra=pd.concat([extra,coast.head(1)],ignore_index=True)
    ntest=max(1,round(len(extra)*.2))
    rng=np.random.default_rng(seed)
    held=set(rng.permutation(extra.event_id.to_numpy())[:ntest])
    extra['split']=np.where(extra.event_id.isin(held),'holdout','train')
    base=base.copy()
    base['addition_group']='cohorte_1000_conservada'
    result=pd.concat([base,extra],ignore_index=True)
    if result.event_id.duplicated().any():
        raise ValueError('Extension contains duplicate events')
    return result,extra,strict
