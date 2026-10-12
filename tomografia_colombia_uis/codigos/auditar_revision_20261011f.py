"""Post hoc audits only; no changes to the frozen inversions."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from robustez_20261011 import nodes_for, GRID
from robustez_20261011 import COARSE
from regional.rays import mapping

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'resultados/tomografia_1d_adjunto_consistente_2026-10-10'
ROB=ROOT/'resultados/robustez_20261011'
OUT=ROOT/'resultados/revision_independiente_20261011f'
OUT.mkdir(exist_ok=True)
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
sources={}
def read(f):
    sources[str(f)]=sha(f)
    return pd.read_csv(f,low_memory=False)
def rmse(v):return float(np.sqrt(np.mean(np.sum(v*v,axis=1))))
def err(a,true):return a[['x_km','y_km','depth_km']].to_numpy()-true
results={}
all_errors=[]
for amp in ['03','08']:
    folder=ROB/f'synthetic_{amp}pct'
    q=read(folder/'synthetic_observations.csv').drop_duplicates('event_id')
    rel=read(folder/'relocated_train_events.csv')
    ctrl=read(folder/'control_relocation_3.csv')
    q=q.set_index('event_id').loc[rel.event_id].reset_index()
    ctrl=ctrl.set_index('event_id').loc[rel.event_id].reset_index()
    true=q[['true_x_km','true_y_km','true_depth_km']].to_numpy()
    start=q[['source_x_km','source_y_km','ev_prof_km']].to_numpy()-true
    nearest=nodes_for(q)
    initial_nodes=np.array([GRID.xyz(np.array(nearest[e])) for e in q.event_id])
    r=err(rel,true);c=err(ctrl,true)
    dist=lambda v:np.linalg.norm(v,axis=1)
    initial=dist(start);final=dist(r)
    df=pd.DataFrame(dict(event_id=rel.event_id,scenario=amp,initial_km=initial,final_3d_km=final,control_1d_km=dist(c)))
    all_errors.append(df)
    rng=np.random.default_rng(20261011);deltas=[]
    for _ in range(3000):
        ix=rng.integers(0,len(df),len(df));deltas.append(np.sqrt(np.mean(final[ix]**2))-np.sqrt(np.mean(initial[ix]**2)))
    results[amp]=dict(events=len(q),initial_rmse_km=rmse(start),initial_node_rmse_km=rmse(initial_nodes-true),control_1d_rmse_km=rmse(c),final_3d_rmse_km=rmse(r),fraction_worse=float(np.mean(final>initial)),delta_rmse_ci95_km=np.quantile(deltas,[.025,.975]).tolist(),rmse_ratio=rmse(r)/rmse(start),initial_mean_error_km=float(initial.mean()))
pd.concat(all_errors).to_csv(OUT/'errores_hipocentrales_pareados.csv',index=False)
mapflat=mapping(GRID,COARSE);count=np.bincount(mapflat,minlength=np.prod(COARSE.shape))
fig,axes=plt.subplots(2,2,figsize=(10,10),layout='constrained')
for row,amp in enumerate(['03','08']):
    recovery=ROB/f'synthetic_{amp}pct/nonlinear_recovery.npz'
    sources[str(recovery)]=sha(recovery)
    f=np.load(recovery)
    active=f['active'];base=f['base_vp'];iz=int(np.argmin(abs(COARSE.z-60)))
    for col,(key,label) in enumerate([('truth_vp','Referencia agregada'),('recovered_vp','Recuperado')]):
        delta=100*(f[key]-base)/base
        aggregated=(np.bincount(mapflat,weights=delta.ravel(),minlength=active.size)/count).reshape(active.shape)
        panel=np.where(active[iz],aggregated[iz],np.nan)
        ax=axes[row,col]
        im=ax.imshow(panel,origin='lower',extent=(COARSE.x[0]-10,COARSE.x[-1]+10,COARSE.y[0]-10,COARSE.y[-1]+10),cmap='RdBu',vmin=-16,vmax=16,aspect='equal')
        ax.set(xlabel='X proyectada (km)',ylabel='Y proyectada (km)',title=f'{label}: ±{int(amp)} %, 60 km')
fig.colorbar(im,ax=axes,label='Anomalía de Vp respecto al 1D (%)',shrink=.8)
fig.savefig(OUT/'Figura_6_sinteticos_no_lineales.png',dpi=180);plt.close(fig)

r=read(ROOT/'resultados/revision_cientifica_20261011/holdout_ablation_residuals.csv')
rows=[]
for lo,hi in [(0,40),(40,80),(80,120),(120,180)]:
    sub=r[r.ev_prof_km.ge(lo)&(r.ev_prof_km.lt(hi) if hi<180 else r.ev_prof_km.le(hi))]
    row=dict(depth_min_km=lo,depth_max_km=hi,events=sub.event_id.nunique(),picks=len(sub))
    for name,col in [('initial_1d','continuous_1d_no_statics'),('control_1d','continuous_1d_relocated_statics'),('final_3d','continuous_3d_final3d_statics')]:row[name+'_rms_s']=float(np.sqrt(np.mean(sub[col]**2)))
    rows.append(row)
pd.DataFrame(rows).to_csv(OUT/'reserva_por_profundidad_interpolada.csv',index=False)

p=read(OLD/'predictions.csv')
stations=p[['station','st_lat_current','st_lon_current']].drop_duplicates('station')
counts=p[p.split.eq('train')].station.value_counts();stations['train_picks']=stations.station.map(counts).fillna(0).astype(int)
catalogs=[ROOT/'datos/estaciones_cm.csv',ROOT/'datos/caribe_1d_estaciones_completas_2026-10-10/estaciones_todas_regionales.csv',ROOT/'datos/internacional_choco_2026-10-10/stations_regional_isc.csv',ROOT/'datos/nazca_cauca_valle_eje_2026-10-10/estaciones_regionales.csv']
cs=[]
for f in catalogs:
    a=read(f)
    if all(x in a for x in ['net','sta','lat','lon','elev']):
        a['station']=a.net.astype(str)+'.'+a.sta.astype(str);a['source']=str(f.relative_to(ROOT));cs.append(a[['station','lat','lon','elev','source']])
c=pd.concat(cs).drop_duplicates(['station','lat','lon','elev'])
for idx,s in stations.iterrows():
    candidates=c[c.station.eq(s.station)].copy()
    if len(candidates)==0:continue
    candidates['coordinate_difference_deg']=np.hypot(candidates.lat-s.st_lat_current,candidates.lon-s.st_lon_current)
    candidates=candidates.sort_values('coordinate_difference_deg')
    selected=candidates.iloc[0]
    if selected.coordinate_difference_deg>.01:continue
    compatible=candidates[candidates.coordinate_difference_deg.lt(.01)]
    elevations=pd.to_numeric(compatible.elev,errors='coerce').dropna()
    if len(elevations)==0 or elevations.max()-elevations.min()>100:continue
    stations.loc[idx,'elevation_m']=float(selected.elev)
    stations.loc[idx,'metadata_source']=selected.source
    stations.loc[idx,'coordinate_difference_deg']=selected.coordinate_difference_deg
for name,f in [('3d',OLD/'station_statics.csv'),('1d',ROOT/'resultados/revision_cientifica_20261011/1d_relocated_station_statics.csv')]:
    a=read(f).set_index('station').static_s
    stations['static_'+name+'_s']=stations.station.map(a)
stations.to_csv(OUT/'estaticas_y_elevacion.csv',index=False)
stats={}
sub=stations[stations.elevation_m.notna()&stations.train_picks.gt(0)&stations.station.ne('CM.ACH1')].copy()
for name in ['1d','3d']:
    sub1=sub[sub['static_'+name+'_s'].notna()]
    x=sub1.elevation_m.to_numpy()/1000;y=sub1['static_'+name+'_s'].to_numpy()
    X=np.column_stack([np.ones(len(x)),x,sub1.st_lat_current,sub1.st_lon_current])
    beta=np.linalg.lstsq(X,y,rcond=None)[0]
    rng=np.random.default_rng(20261011);boot=[]
    for _ in range(3000):
        ix=rng.integers(0,len(x),len(x));boot.append(np.linalg.lstsq(X[ix],y[ix],rcond=None)[0][1])
    stats[name]=dict(stations=len(x),pearson_r=float(pearsonr(x,y).statistic),pearson_p=float(pearsonr(x,y).pvalue),spearman_r=float(spearmanr(x,y).statistic),spearman_p=float(spearmanr(x,y).pvalue),elevation_slope_adjusted_latlon_s_per_km=float(beta[1]),slope_ci95_station_bootstrap=np.quantile(boot,[.025,.975]).tolist(),ols_r2=float(1-np.sum((y-X@beta)**2)/np.sum((y-y.mean())**2)))
fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
for ax,name in zip(axes,['1d','3d']):
    ax.scatter(sub.elevation_m/1000,sub['static_'+name+'_s'],s=20,alpha=.7)
    ax.set(xlabel='Elevación de estación (km)',ylabel='Estática (s)',title=f'{name.upper()}: r = {stats[name]["pearson_r"]:.3f}, n = {stats[name]["stations"]}')
    ax.axhline(0,color='gray',lw=.7);ax.grid(alpha=.15)
fig.savefig(OUT/'Figura_S6_estaticas_elevacion.png',dpi=180);plt.close(fig)
geo=ROOT/'datos/cartografia/departamentos_dane_2025.geojson';sources[str(geo)]=sha(geo)
features=json.loads(geo.read_text(encoding='utf-8'))['features']
fig,axes=plt.subplots(1,2,figsize=(10,7),layout='constrained')
for ax,name in zip(axes,['1d','3d']):
    for feature in features:
        g=feature['geometry']
        polys=g['coordinates'] if g['type']=='MultiPolygon' else [g['coordinates']]
        for poly in polys:
            for ring in poly:
                arr=np.array(ring);ax.plot(arr[:,0],arr[:,1],color='gray',lw=.4)
    points=ax.scatter(sub.st_lon_current,sub.st_lat_current,c=sub['static_'+name+'_s'],cmap='RdBu',vmin=-1.5,vmax=1.5,s=22,zorder=2)
    ax.set(xlim=(-79,-71),ylim=(.8,13),xlabel='Longitud (°)',ylabel='Latitud (° N)',title=f'Estáticas del modelo {name.upper()}')
    ax.set_aspect('equal');fig.colorbar(points,ax=ax,label='Estática (s)',shrink=.8)
fig.savefig(OUT/'Figura_S7_mapa_estaticas.png',dpi=180);plt.close(fig)
report=dict(synthetic_location=results,depth_holdout=rows,statics_elevation=stats,metadata_coverage=dict(all_stations=len(stations),elevation_matched=int(stations.elevation_m.notna().sum()),analyzed=len(sub)),notes=['Exploratory post hoc station correlation, not causal attribution. Equal station weights; CM.ACH1 reference and test-only stations excluded.','Station coordinate agreement <=0.01 degree; incompatible elevations spanning >100 m excluded.','Latitude/longitude adjusted OLS and station bootstrap do not eliminate spatial dependence.','Synthetic truth permits position-error assessment; no true positions are available for real catalog.'],input_sha256=sources)
(OUT/'auditoria_independiente.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='input_sha256'},indent=2,ensure_ascii=False))
