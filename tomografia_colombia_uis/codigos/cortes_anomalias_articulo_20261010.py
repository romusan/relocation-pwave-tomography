"""Vp anomalies relative to the verified 1D starting volume."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
from scipy.interpolate import RegularGridInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'codigos'))
from nazca_campana_20261010 import projection
RUN=ROOT/'resultados/tomografia_1d_adjunto_consistente_2026-10-10'
OUT=ROOT/'articulo_boletin_geologia_uis_adjunto'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pr=projection(json.loads((RUN/'domain.json').read_text()))
events=pd.read_csv(RUN/'relocated_train_events.csv')
with np.load(RUN/'vp_final.npz') as f:
    initial=RegularGridInterpolator((f['zg'],f['yg'],f['xg']),f['vp_initial'])
    final=RegularGridInterpolator((f['zg'],f['yg'],f['xg']),f['vp'])
    cover=RegularGridInterpolator((f['inversion_zg'],f['inversion_yg'],f['inversion_xg']),f['active'].astype(float),method='nearest',bounds_error=False,fill_value=0)
lat,depth=np.meshgrid(np.arange(.8,13.0001,.18),np.arange(0,180.001,20))
longitudes=[-77,-76,-74];data=[];masks=[];stats=[]
plt.rcParams.update({'font.size':16,'font.family':'DejaVu Sans','axes.labelsize':16,'axes.titlesize':16})
fig,axs=plt.subplots(1,3,figsize=(12,5.5),layout='constrained')
cmap=plt.get_cmap('RdBu').copy();cmap.set_bad('#dcdcdc')
for ax,lon in zip(axs,longitudes):
    px,py=pr.xy(lat.ravel(),np.full(lat.size,lon));q=np.c_[depth.ravel(),py,px]
    v0=initial(q);v1=final(q);supported=cover(q)>.5
    delta=(100*(v1/v0-1)).reshape(lat.shape);mask=supported.reshape(lat.shape)
    data.append(delta);masks.append(mask);shown=delta[mask]
    assert np.isfinite(shown).all() and (np.abs(shown)<=16.000001).all()
    stats.append({'longitude_deg':lon,'active_section_samples':int(mask.sum()),'minimum_percent':float(shown.min()),'maximum_percent':float(shown.max())})
    im=ax.pcolormesh(lat,depth,np.ma.array(delta,mask=~mask),norm=TwoSlopeNorm(vmin=-16,vcenter=0,vmax=16),cmap=cmap,shading='nearest',rasterized=True)
    e=events[events.lon.between(lon-.25,lon+.25)]
    ax.scatter(e.lat,e.depth_km,c='black',s=5,alpha=.55)
    ax.set(title=f'{abs(lon)}° O',xlim=(.8,13),ylim=(180,0),xlabel='Latitud (° N)',ylabel='Profundidad (km)',xticks=[2,6,10],yticks=[0,40,80,120,160,180])
    ax.grid(alpha=.12)
bar=fig.colorbar(im,ax=axs.ravel().tolist(),label='Anomalía de Vp respecto a 1D (%)',ticks=[-16,-8,0,8,16],shrink=.85)
path=OUT/'figuras/Figura_5_cortes_anomalias_Vp.png';assert not path.exists();fig.savefig(path,dpi=300);plt.close(fig)
arrays=OUT/'fuentes/cortes_anomalias_vp.npz';np.savez(arrays,latitude_deg=lat,depth_km=depth,longitude_deg=np.array(longitudes),anomaly_percent=np.array(data),active=np.array(masks))
audit={'definition':'100 * (Vp_final / Vp_initial - 1), at identical spatial coordinates',
    'color_scale_percent':[-16,16],'positive':'blue; faster than initial 1D','negative':'red; slower than initial 1D',
    'inactive':'gray; no active coverage','earthquakes':'relocated SGC training events within longitude +/- 0.25 degrees',
    'sampling':{'latitude_step_deg':.18,'depth_step_km':20},'sections':stats,
    'input_sha256':{str(p):sha(p) for p in [Path(__file__),RUN/'vp_final.npz',RUN/'domain.json',RUN/'relocated_train_events.csv']},
    'output_sha256':{str(p):sha(p) for p in [path,arrays]}}
(OUT/'fuentes/auditoria_cortes_anomalias.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(stats,indent=2),flush=True)
