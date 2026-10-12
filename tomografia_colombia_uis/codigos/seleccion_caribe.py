"""Spatially balanced northern additions, with preserved previous holdout."""
import numpy as np
import pandas as pd
from regional.inputs import spatial_order


def extend_cohort(base, available, catalog, projection, number=300, seed=20261009):
    for name, table in [('base', base), ('available', available), ('catalog', catalog)]:
        if table.event_id.duplicated().any():
            raise ValueError(f'Duplicate event identifiers in {name}')
    if number <= 0 or int(number) != number:
        raise ValueError('Additional event count must be a positive integer')
    if not set(base.split).issubset({'train', 'holdout'}):
        raise ValueError('Invalid previous split')
    if not set(base.event_id).issubset(set(available.event_id)):
        raise ValueError('Missing previous associated events')
    candidates = available[~available.event_id.isin(base.event_id)
        & available.lat.between(7., 9.) & available.lon.between(-79., -73.)].merge(
        catalog[['event_id', 'region']], on='event_id', validate='one_to_one')
    if len(candidates) < number:
        raise ValueError(f'Only {len(candidates)} northern events available for {number} additions')
    order = spatial_order(candidates, projection)
    extra = candidates.set_index('event_id').loc[order[:number]].reset_index()
    extra['addition_group'] = 'Norte 7–9 N (sin asignacion de placa)'
    ntest = max(1, round(number * .2))
    held = set(np.random.default_rng(seed).permutation(extra.event_id.to_numpy())[:ntest])
    extra['split'] = np.where(extra.event_id.isin(held), 'holdout', 'train')
    result = pd.concat([base, extra], ignore_index=True)
    if result.event_id.duplicated().any():
        raise ValueError('Extension includes duplicate events')
    kept = result.set_index('event_id').loc[base.event_id, 'split'].to_numpy()
    np.testing.assert_array_equal(kept, base.split.to_numpy())
    return result, extra, candidates
