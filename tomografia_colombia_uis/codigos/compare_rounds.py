"""Compare successive controls on identical selected events and grid axes."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('earlier',type=Path)
    ap.add_argument('later',type=Path)
    a=ap.parse_args()
    with np.load(a.earlier/'vp_final.npz') as x,np.load(a.later/'vp_final.npz') as y:
        for key in ('xg','yg','zg','inversion_xg','inversion_yg','inversion_zg'):
            np.testing.assert_array_equal(x[key],y[key])
        first=pd.read_csv(a.earlier/'events_split.csv')
        second=pd.read_csv(a.later/'events_split.csv')
        pd.testing.assert_frame_equal(first,second)
        summary=json.loads((a.earlier/'summary.json').read_text(encoding='utf-8'))
        stride=int(round(summary['config']['inversion_spacing_km']/summary['config']['solver_spacing_km']))
        mask=x['active']&y['active']
        rel=100*np.abs(y['vp']/x['vp']-1)
        coarse=rel[::stride,::stride,::stride]
        r=pd.read_csv(a.earlier/'relocated_train_events.csv')
        s=pd.read_csv(a.later/'relocated_train_events.csv')
        m=r.merge(s,on='event_id',suffixes=('_earlier','_later'),validate='one_to_one')
        shift=np.sqrt(sum((m[c+'_later']-m[c+'_earlier'])**2 for c in ('x_km','y_km','depth_km')))
        result={'same_event_population':set(r.event_id)==set(s.event_id),
            'common_final_active_cells':int(mask.sum()),
            'common_active_vp_change_max_percent':float(coarse[mask].max()) if mask.any() else None,
            'common_active_vp_change_median_percent':float(np.median(coarse[mask])) if mask.any() else None,
            'event_nodes_changed_earlier_to_later':int((shift>1e-6).sum()),
            'event_node_shift_max_km':float(shift.max()),
            'note':'Additional round is a convergence control, not a tuned choice using holdout.'}
        output=a.earlier/'control_4r_comparison.json'
        output.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
