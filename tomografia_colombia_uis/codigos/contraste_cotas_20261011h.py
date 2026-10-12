"""Matched receiver elevation contrast, separate from the regularization protocol."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
import campana_perfil_suavidad_20261011h as h
from factorizado_paralelo_20261011g import BaseCache
# Serial precomputation for this lightweight independent control avoids
# oversubscribing the simultaneous iterative Vp experiment.
h.f.FieldCache=BaseCache
p=h.population();q=p.copy();q['st_elev_km']=0.
plan=dict(changes='only receiver elevation set to zero; same population, factored solver, initial profile, partitions, absolute station damping and training stopping criterion',parent_code_sha256=hashlib.sha256(Path(h.__file__).read_bytes()).hexdigest(),data_sha256=hashlib.sha256((h.GOUT/'selected_picks.csv').read_bytes()).hexdigest(),case_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
path=h.OUT/'elevation_contrast_protocol.json'
if path.exists():assert json.loads(path.read_text())==plan
else:h.save(path,plan)
h.fit(q,'fixed',0.,h.OUT/'fixed_1d_zero_elevation')
if not (h.OUT/'fixed_1d/comparison.json').exists():raise RuntimeError('Physical-height control must finish before paired analysis')
stations=p[p.split.eq('train')][['station','st_x_km','st_y_km','st_elev_km']].drop_duplicates()
lat,lon=h.PR.latlon(stations.st_x_km.to_numpy(),stations.st_y_km.to_numpy())
stations['lat']=lat;stations['lon']=lon
for tag,folder in [('actual','fixed_1d'),('zero','fixed_1d_zero_elevation')]:
    static=pd.read_csv(h.OUT/folder/'station_statics.csv').rename(columns={'static_s':tag})
    stations=stations.merge(static,on='station',validate='one_to_one')
ref=json.loads((h.OUT/'fixed_1d/comparison.json').read_text())['reference_station'];stations=stations[stations.station.ne(ref)].reset_index(drop=True)
X=np.column_stack([np.ones(len(stations)),stations.st_elev_km,stations.lat-stations.lat.mean(),stations.lon-stations.lon.mean()]);Y=stations[['zero','actual']].to_numpy()
beta=np.linalg.lstsq(X,Y,rcond=None)[0][1];rng=np.random.default_rng(20261011);bs=[]
for _ in range(3000):
    ids=rng.integers(len(stations),size=len(stations));bs.append(np.linalg.lstsq(X[ids],Y[ids],rcond=None)[0][1])
bs=np.asarray(bs)
a=pd.read_csv(h.OUT/'fixed_1d/holdout_residuals.csv').rename(columns={'residual_s':'actual'})
b=pd.read_csv(h.OUT/'fixed_1d_zero_elevation/holdout_residuals.csv').rename(columns={'residual_s':'zero'})
assert a[['event_id','station']].equals(b[['event_id','station']])
report=dict(stations=len(stations),reference=ref,zero_slope_s_per_km=float(beta[0]),actual_slope_s_per_km=float(beta[1]),paired_slope_difference_actual_minus_zero_s_per_km=float(beta[1]-beta[0]),paired_ci95_s_per_km=np.quantile(bs[:,1]-bs[:,0],[.025,.975]).tolist(),rms_actual=h.metrics(a.actual.to_numpy()),rms_zero=h.metrics(b.zero.to_numpy()),paired_event_bootstrap=h.bootstrap(a,a.actual.to_numpy(),b.zero.to_numpy()),limitation='Matched computational contrast; no geology covariate; station bootstrap ignores spatial dependence; does not establish the cause of the earlier first-order versus factored RMS difference')
h.save(h.OUT/'elevation_contrast.json',report);stations.to_csv(h.OUT/'elevation_contrast_stations.csv',index=False)
print(json.dumps(report),flush=True)
