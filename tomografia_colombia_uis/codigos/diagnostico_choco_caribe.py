"""Compare shared withheld observations and geographic coverage across domains."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from regional.inputs import sha256
from regional.geometry import Projection
from reconstruir_benioff_ampliado import model_cells

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--run',type=Path,required=True)
    a=ap.parse_args(); old=ROOT/'resultados/migracion_2200_quibdo_antioquia_2026-10-10'
    runs=[old,a.run]; sources={}
    for run in runs:
        for f,h in json.loads((run/'manifest.json').read_text()).items():
            if sha256(f)!=h: raise ValueError('Changed source '+f)
        for name in ['predictions.csv','vp_final.npz','summary.json','events_split.csv']:
            f=run/name; sources[str(f.resolve())]=sha256(f)
    p=[pd.read_csv(r/'predictions.csv') for r in runs]
    h=[d[d.split.eq('holdout')] for d in p]
    keys=['event_id','station','raw_row']
    shared=h[0].merge(h[1],on=keys,suffixes=('_previous','_new'),validate='one_to_one')
    if len(shared)!=len(h[0]): raise ValueError('Lost inherited holdout observations')
    delta=np.abs(shared.observed_s_previous-shared.observed_s_new).max()
    if delta>1e-10: raise ValueError('Changed inherited UTC observations')
    rms=lambda x: float(np.sqrt(np.mean(np.asarray(x)**2))) if len(x) else None
    coverage=[]
    for run in runs:
        domain=json.loads((run/'summary.json').read_text())['domain']
        with np.load(run/'vp_final.npz') as m: c=model_cells(m,Projection(domain))
        e=pd.read_csv(run/'relocated_train_events.csv')
        windows={'Choco_Quibdo':(4.8,6.8,-77.3,-75.3),'Caribe_norte':(9.,13.,-79.,-71.),
                 'centro':(4.5,7.2,-76.7,-74.3)}
        zones={}
        for name,(a0,a1,b0,b1) in windows.items():
            cells=c[c.lat.between(a0,a1)&c.lon.between(b0,b1)]
            rel=e[e.lat.between(a0,a1)&e.lon.between(b0,b1)]
            zones[name]=dict(active_vp_cells=len(cells),active_vp_cells_depth_ge_40=int(cells.depth_km.ge(40).sum()),
                relocated_events=len(rel),relocated_depth_ge_40=int(rel.depth_km.ge(40).sum()))
        coverage.append(dict(run=str(run.resolve()),zones=zones,total_active_cells=len(c)))
    extra=h[1][~h[1].event_id.isin(h[0].event_id)]
    additions=pd.read_csv(a.run/'added_events.csv')
    holdout_groups={}
    for name,g in additions.groupby('addition_group',sort=True):
        observations=extra[extra.event_id.isin(g.event_id)]
        holdout_groups[name]=dict(events=observations.event_id.nunique(),P_observations=len(observations),
            rms_fixed_catalog_origin_s=rms(observations.residual_s))
    shared.to_csv(a.run/'common_holdout_comparison.csv',index=False)
    sources[str(Path(__file__).resolve())]=sha256(__file__)
    result=dict(previous_sources_unchanged=True,common_holdout_events=shared.event_id.nunique(),
        common_holdout_P_observations=len(shared),observed_s_max_difference=float(delta),
        common_holdout_previous_rms_s=rms(shared.residual_s_previous),
        common_holdout_new_rms_s=rms(shared.residual_s_new),
        new_holdout_events=extra.event_id.nunique(),new_holdout_P_observations=len(extra),
        new_holdout_rms_s=rms(extra.residual_s),coverage=coverage,
        holdout_by_added_group=holdout_groups,
        limitation='Domain, projection and station coverage also changed: difference is not attributable to event count alone. Active cells are not demonstrated resolution.',
        input_sha256=sources)
    (a.run/'comparacion_choco_caribe.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'},indent=2))

if __name__=='__main__': main()
