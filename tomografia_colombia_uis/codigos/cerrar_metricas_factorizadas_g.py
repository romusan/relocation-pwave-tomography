from pathlib import Path
import json,numpy as np,pandas as pd
from robustez_20261011 import metrics,bootstrap
root=Path(__file__).resolve().parents[1]/'resultados/campana_factorizada_cotas_20261011g'
save=lambda name,obj:(root/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
synthetics=[]
for n in ['03','08']:
 p=root/f'synthetic_{n}pct/recovery_metrics.json';assert p.is_file(),str(p)
 synthetics.append(json.loads(p.read_text()))
save('nonlinear_synthetics.json',synthetics)
regions=[];tables=[]
for n in [1,2,3]:
 d=root/f'spatial_fold_{n}';assert (d/'comparison.json').is_file(),str(d)
 r=json.loads((d/'comparison.json').read_text());r.update(fold=n,excluded_buffer_events=len(pd.read_csv(d/'buffer_excluded_events.csv')));regions.append(r)
 q=pd.read_csv(d/'exact_holdout_residuals.csv');q['fold']=n;tables.append(q)
save('spatial_validation.json',regions)
p=pd.concat(tables,ignore_index=True);assert p.event_id.nunique()==3621
result=dict(control_1d=metrics(p.residual_1d_s.to_numpy()),model_3d=metrics(p.residual_3d_s.to_numpy()),paired_bootstrap=bootstrap(p,p.residual_3d_s.to_numpy(),p.residual_1d_s.to_numpy()),events=p.event_id.nunique(),picks=len(p),definition='pooled predictions from disjoint spatial holdouts, not a single fitted model')
save('spatial_pooled_comparison.json',result);p.to_csv(root/'spatial_pooled_residuals.csv',index=False)
print(json.dumps(dict(synthetics=synthetics,spatial=result),indent=2))
