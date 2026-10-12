"""Verify a completed run's provenance and fixed train/holdout population."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from regional.inputs import sha256


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('run',type=Path)
    a=ap.parse_args()
    run=a.run.resolve()
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    changed=[path for path,digest in manifest.items() if sha256(path)!=digest]
    if changed:
        raise AssertionError(f'Sources differ from the run snapshot: {changed}')
    summary=json.loads((run/'summary.json').read_text(encoding='utf-8'))
    predictions=pd.read_csv(run/'predictions.csv')
    split=pd.read_csv(run/'events_split.csv')
    if split.event_id.duplicated().any():
        raise AssertionError('Duplicate selected event')
    train=set(split[split.split.eq('train')].event_id)
    holdout=set(split[split.split.eq('holdout')].event_id)
    if train&holdout or set(predictions.event_id)!=train|holdout:
        raise AssertionError('Train/holdout population mismatch')
    if predictions.duplicated(['event_id','station','fase']).any():
        raise AssertionError('More than one associated phase per event/station')
    if (predictions.observed_s<=0).any():
        raise AssertionError('Nonpositive travel time in inversion')
    if not np.isfinite(predictions[['predicted_s','residual_s','origin_shift_s','station_static_s']]).all().all():
        raise AssertionError('Nonfinite prediction')
    np.testing.assert_allclose(predictions.residual_s,predictions.observed_s-predictions.predicted_s-
        predictions.origin_shift_s-predictions.station_static_s,atol=1e-10)
    held=predictions[predictions.split.eq('holdout')]
    np.testing.assert_array_equal(held.origin_shift_s,np.zeros(len(held)))
    final_rms=float(np.sqrt(np.mean(held.residual_s**2)))
    np.testing.assert_allclose(final_rms,summary['history'][-1]['holdout_rms_fixed_catalog_origin_s'],atol=1e-10)
    with np.load(run/'vp_final.npz') as model:
        vp=model['vp']
        if vp.shape!=tuple(summary['direct_shape']) or not np.isfinite(vp).all():
            raise AssertionError('Invalid final model')
        if vp.min()<summary['config']['vp_min_km_s']-1e-10 or vp.max()>summary['config']['vp_max_km_s']+1e-10:
            raise AssertionError('Vp exceeds physical bounds')
    for row in summary['history']:
        if not row['lsmr']['converged']:
            raise AssertionError('Unconverged solver accepted')
        if row['train_equations']!=summary['train_picks']:
            raise AssertionError('Population changed between rounds')
        if row['update_accepted'] and row['train_rms_after_update_s']>row['train_rms_before_update_s']+1e-10:
            raise AssertionError('Accepted velocity update made fit worse')
    result={'verified':True,'source_files_checked':len(manifest),'train_events':len(train),
        'holdout_events':len(holdout),'predictions_checked':len(predictions),
        'note':'Software/data invariants only; no certification of scientific resolution'}
    (run/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
