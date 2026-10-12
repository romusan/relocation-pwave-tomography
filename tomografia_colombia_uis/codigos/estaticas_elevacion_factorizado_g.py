from pathlib import Path
import json,numpy as np,pandas as pd
from regional.geometry import Projection
root=Path(__file__).resolve().parents[1]/'resultados/campana_factorizada_cotas_20261011g'
p=pd.read_csv(root/'selected_picks.csv');train=set(p.loc[p.split.eq('train'),'station'])
c=json.loads((root/'real_elevation/control_1d.json').read_text());reference=c['reference']
station=p[['station','st_elev_km','st_x_km','st_y_km']].drop_duplicates('station')
pr=Projection(json.loads((root/'real_elevation/domain.json').read_text()))
station['st_lat'],station['st_lon']=pr.latlon(station.st_x_km.to_numpy(),station.st_y_km.to_numpy())
station=station[station.station.isin(train)&station.station.ne(reference)].copy()
s=pd.read_csv(root/'real_elevation/station_statics.csv')
station['static_1d_s']=station.station.map(c['statics']);station=station.merge(s[['station','static_s']],on='station',validate='one_to_one').rename(columns={'static_s':'static_3d_s'})
assert not station.isna().any().any()
X=np.column_stack([np.ones(len(station)),station[['st_elev_km','st_lat','st_lon']].to_numpy()]);rng=np.random.default_rng(20261011);reports={}
indices=rng.integers(0,len(station),size=(3000,len(station)))
for key in ['1d','3d']:
 y=station[f'static_{key}_s'].to_numpy();beta=np.linalg.lstsq(X,y,rcond=None)[0]
 bs=[np.linalg.lstsq(X[i],y[i],rcond=None)[0][1] for i in indices]
 reports[key]=dict(stations=len(y),excluded_gauge=reference,coordinates='inverse AEQD of the actual station XY used in the solver',pearson=float(np.corrcoef(station.st_elev_km,y)[0,1]),slope_adjusted_lat_lon_s_per_km=float(beta[1]),ci95_station_bootstrap_s_per_km=np.quantile(bs,[.025,.975]).tolist(),bootstrap_replicates=3000,interpretation='exploratory association; station bootstrap does not account for spatial dependence or prove causality')
station.to_csv(root/'station_elevation_association.csv',index=False)
(root/'station_elevation_association.json').write_text(json.dumps(reports,indent=2),encoding='utf-8');print(json.dumps(reports,indent=2))
