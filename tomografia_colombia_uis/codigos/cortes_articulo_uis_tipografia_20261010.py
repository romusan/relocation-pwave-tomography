"""Redraw unchanged six-panel sections with print-readable labels."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
from scipy.interpolate import RegularGridInterpolator
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'codigos'))
from nazca_campana_20261010 import projection
RUN=ROOT/'resultados/migracion_nazca_cauca_valle_eje_1d_2026-10-10'
OUT=ROOT/'articulo_boletin_geologia_uis'
pr=projection(json.loads((RUN/'domain.json').read_text()))
e=pd.read_csv(RUN/'relocated_train_events.csv')
with np.load(RUN/'vp_final.npz') as f:
    initial=RegularGridInterpolator((f['zg'],f['yg'],f['xg']),f['vp_initial'])
    final=RegularGridInterpolator((f['zg'],f['yg'],f['xg']),f['vp'])
    cover=RegularGridInterpolator((f['inversion_zg'],f['inversion_yg'],f['inversion_xg']),f['active'].astype(float),method='nearest',bounds_error=False,fill_value=0)
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
plt.rcParams.update({'font.size':16,'font.family':'DejaVu Sans','axes.labelsize':16,'axes.titlesize':16})
fig,axs=plt.subplots(2,3,figsize=(12,8),layout='constrained')
lt,zt=np.meshgrid(np.arange(.8,13.0001,.18),np.arange(0,180.001,20))
for j,lon in enumerate([-77,-76,-74]):
    px,py=pr.xy(lt.ravel(),np.full(lt.size,lon));q=np.c_[zt.ravel(),py,px]
    for i,interp in enumerate([initial,final]):
        v=interp(q).reshape(lt.shape)
        if i:v=np.ma.array(v,mask=cover(q).reshape(lt.shape)<.5)
        im=axs[i,j].pcolormesh(lt,zt,v,vmin=4.5,vmax=8.8,cmap='viridis',shading='nearest',rasterized=True)
        if i:
            ev=e[e.lon.between(lon-.25,lon+.25)];axs[i,j].scatter(ev.lat,ev.depth_km,s=3,c='black',alpha=.4)
        axs[i,j].set(xlim=(.8,13),ylim=(180,0),xlabel='Latitud (° N)',ylabel='Profundidad (km)',title=('Inicial 1D' if i==0 else 'Final 3D')+f'   {lon}°',xticks=[2,6,10],yticks=[0,40,80,120,160,180]);axs[i,j].grid(alpha=.15)
bar=fig.colorbar(im,ax=axs.ravel().tolist(),label='Vp (km/s)',shrink=.8)
bar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:.1f}'.replace('.',',')))
fig.savefig(OUT/'figuras/Figura_3_cortes_inicial_final.png',dpi=300);plt.close(fig)
print('Revised labels only; section values and mask unchanged.')
