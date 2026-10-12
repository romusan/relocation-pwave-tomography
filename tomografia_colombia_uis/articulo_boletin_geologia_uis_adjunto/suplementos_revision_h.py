"""Scientific supplemental figures from complete h results only."""
import json,sys
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figuras_revision_h import ROOT,C,OUT,pr,boundaries
sys.path.insert(0,str(ROOT/'codigos'))
from regional.geometry import Grid
from regional.factored import mapping
from controles_factorizados_20261011g import truth
from campana_perfil_suavidad_20261011h import GRID,initial
assert json.loads((C/'completed.json').read_text())['status']=='completed'
assert (C/'elevation_contrast.json').exists()
arms=['fixed_1d','updated_1d','real_smooth_2','real_smooth_20','real_smooth_200']
labels=['1D fijo','1D ajustable','3D λ = 2','3D λ = 20','3D λ = 200']
colors=['#444444','#1b9e77','#d95f02','#7570b3','#e7298a']
fig,axes=plt.subplots(1,2,figsize=(10,4.5),constrained_layout=True)
for arm,label,color in zip(arms,labels,colors):
    q=pd.DataFrame(json.loads((C/arm/'history.json').read_text()))
    for ax,col in zip(axes,['train_rms_s','holdout_rms_s']):ax.plot(q['round'],q[col],'-o',ms=3,label=label,color=color)
for ax,title in zip(axes,['Entrenamiento','Reserva condicionada al catálogo']):ax.set(xlabel='Ronda',ylabel='RMS (s)',title=title);ax.grid(alpha=.2)
axes[1].legend(fontsize=8);fig.savefig(OUT/'RMS_por_ronda.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(6,7),constrained_layout=True)
f=np.load(C/'real_smooth_20/vp_final.npz');fine=Grid(f['xg'],f['yg'],f['zg']);coarse=Grid(f['inversion_xg'],f['inversion_yg'],f['inversion_zg']);active=f['active'].ravel()[mapping(fine,coarse)].reshape(fine.shape)
ax.plot(initial()[:,0,0],GRID.z,'k--',label='Perfil inicial')
for arm,label,color in zip(arms[1:],labels[1:],colors[1:]):
    f=np.load(C/arm/'vp_final.npz');vp=f['vp']
    values=[float(np.median(a[m])) if m.any() else np.nan for a,m in zip(vp,active)]
    ax.plot(values,f['zg'],label=label,color=color)
ax.set(xlabel='Vp (km/s)',ylabel='Profundidad (km)',title='Mediana en nodos del soporte activo',ylim=(180,0));ax.legend(fontsize=9);ax.grid(alpha=.2);fig.savefig(OUT/'Perfiles_velocidad.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(10,9),constrained_layout=True)
vp=truth(GRID,.03);base=initial();arrays=[100*(vp/base-1)]+[100*(np.load(C/f'synthetic_smooth_{w}/vp_final.npz')['vp']/base-1) for w in [2,20,200]]
k=int(np.argmin(abs(GRID.z-80)));x,y=np.meshgrid(GRID.x,GRID.y);lat,lon=pr.latlon(x,y)
titles=['Verdad ±3 %','Recuperado λ = 2','Recuperado λ = 20','Recuperado λ = 200']
recovery_range=max(3.,float(np.ceil(max(abs(a[k][active[k]]).max() for a in arrays[1:]))))
for i,(ax,a,title) in enumerate(zip(axes.ravel(),arrays,titles)):
    lim=3 if i==0 else recovery_range;cmap=plt.get_cmap('RdBu').copy();cmap.set_bad('#cfcfcf')
    im=ax.pcolormesh(lon,lat,np.ma.masked_where(~active[k],a[k]),shading='auto',cmap=cmap,vmin=-lim,vmax=lim)
    boundaries(ax);ax.set(xlabel='Longitud (°)',ylabel='Latitud (°)',title=title+' · 80 km');ax.set_aspect(1)
    fig.colorbar(im,ax=ax,shrink=.7,label='Anomalía (%)')
fig.savefig(OUT/'Damero_efectivo.png',dpi=180);plt.close(fig)
q=pd.read_csv(C/'elevation_contrast_stations.csv')
fig,axes=plt.subplots(1,3,figsize=(12,4.5),constrained_layout=True)
X=np.column_stack([np.ones(len(q)),q.st_elev_km,q.lat-q.lat.mean(),q.lon-q.lon.mean()])
for ax,key,title in zip(axes[:2],['zero','actual'],['Cota cero','Cota real']):
    y=q[key].to_numpy();coef=np.linalg.lstsq(X,y,rcond=None)[0];partial=y-X[:,2:]@coef[2:]
    ax.scatter(q.st_elev_km,partial,s=10,alpha=.6);x=np.linspace(q.st_elev_km.min(),q.st_elev_km.max(),100);ax.plot(x,coef[0]+coef[1]*x,color='darkred')
    ax.set(xlabel='Elevación real (km)',ylabel='Estática descontando latitud y longitud (s)',title=title);ax.grid(alpha=.15)
lim=float(np.max(abs(q.actual)));im=axes[2].scatter(q.lon,q.lat,c=q.actual,cmap='RdBu_r',vmin=-lim,vmax=lim,s=12)
boundaries(axes[2]);axes[2].set(xlabel='Longitud (°)',ylabel='Latitud (°)',title='Estáticas con cota real');axes[2].set_aspect(1);fig.colorbar(im,ax=axes[2],shrink=.7,label='Estática (s)')
fig.savefig(OUT/'Estaticas_elevacion.png',dpi=180);plt.close(fig)
(C/'supplement_figure_audit.json').write_text(json.dumps(dict(checkerboard_depth_km=80,truth_color_range_percent=3,recovery_color_range_percent=recovery_range,grey='outside fixed active parameter support, not resolution',profile='median direct-grid nodes within common active support, not whole-domain median',statics_partial='subtract fitted latitude and longitude contributions; intercept retained'),indent=2),encoding='utf-8')
print(OUT)
