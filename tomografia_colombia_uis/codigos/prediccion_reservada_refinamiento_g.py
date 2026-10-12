from pathlib import Path
import json,numpy as np,pandas as pd
root=Path(__file__).resolve().parents[1]/'resultados/campana_factorizada_cotas_20261011g'
mc=json.loads((root/'mesh_new_final_convergence.json').read_text())[0]
p=pd.read_csv(root/'selected_picks.csv');p=p[p.station.isin(mc['station_ids'])].reset_index(drop=True)
ctl=json.loads((root/'real_elevation/control_1d.json').read_text())['statics'];d=pd.read_csv(root/'real_elevation/station_statics.csv');sta=dict(zip(d.station,d.static_s))
held=p.split.eq('holdout').to_numpy();results=[]
for h in [10,5,2.5]:
 one=pd.read_csv(root/f'mesh_initial_1d_{h:g}.csv');three=pd.read_csv(root/f'mesh_new_3d_{h:g}.csv')
 assert np.array_equal(one[['event_id','station']].to_numpy(),p[['event_id','station']].to_numpy())
 assert np.array_equal(three[['event_id','station']].to_numpy(),p[['event_id','station']].to_numpy())
 r1=(p.observed_s.to_numpy()-one.predicted_s.to_numpy()-p.station.map(ctl).to_numpy())[held]
 r3=(p.observed_s.to_numpy()-three.predicted_s.to_numpy()-p.station.map(sta).to_numpy())[held]
 rms=lambda v:float(np.sqrt(np.mean(v*v)))
 results.append(dict(spacing_km=h,held_picks=int(held.sum()),held_events=p.loc[held,'event_id'].nunique(),rms_control_1d_s=rms(r1),rms_3d_s=rms(r3),delta_3d_minus_control_s=rms(r3)-rms(r1)))
report=dict(scope='same six stations selected before refinement; no refit, frozen models/statics, continuous catalog hypocentres, holdout only',station_ids=mc['station_ids'],results=results)
(root/'holdout_refinement_comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
