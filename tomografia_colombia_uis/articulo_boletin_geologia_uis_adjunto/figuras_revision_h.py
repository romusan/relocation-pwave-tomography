from pathlib import Path
import sys,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];C=ROOT/'resultados/campana_perfil_suavidad_20261011h';A=Path(__file__).resolve().parent;OUT=A/'figuras_revision_h';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'codigos'))
from nazca_campana_20261010 import projection
from regional.geometry import Grid
from regional.factored import mapping
domain=json.loads((ROOT/'resultados/campana_factorizada_cotas_20261011g/real_elevation/domain.json').read_text());pr=projection(domain)
geo=json.loads((ROOT/'datos/cartografia/departamentos_dane_2025.geojson').read_text(encoding='utf-8-sig'));lines=[]
for feature in geo['features']:
    g=feature['geometry']
    for poly in g['coordinates'] if g['type']=='MultiPolygon' else [g['coordinates']]:lines.append(np.array(poly[0]))
def boundaries(ax):
    for line in lines:ax.plot(line[:,0],line[:,1],color='.4',lw=.35)
def data():
    p=pd.read_csv(ROOT/'resultados/campana_factorizada_cotas_20261011g/selected_picks.csv',low_memory=False)
    e=pd.read_csv(C/'event_descriptors.csv');sta=p[['st_x_km','st_y_km','st_elev_km']].round(3).drop_duplicates();lat,lon=pr.latlon(sta.st_x_km.to_numpy(),sta.st_y_km.to_numpy())
    fig,axes=plt.subplots(1,3,figsize=(12,5),constrained_layout=True)
    ax=axes[0];im=ax.scatter(e.ev_lon,e.ev_lat,c=e.ev_prof_km,s=5,cmap='viridis_r',vmin=0,vmax=180,alpha=.7,label='Eventos con ID SGC');ax.scatter(lon,lat,marker='^',c='darkred',s=10,label=f'{len(sta)} posiciones de receptor');boundaries(ax);ax.set(xlabel='Longitud (°)',ylabel='Latitud (°)',xlim=(domain['lon_min'],domain['lon_max']),ylim=(domain['lat_min'],domain['lat_max']));ax.legend(fontsize=7,loc='lower left');fig.colorbar(im,ax=ax,shrink=.65,label='Profundidad (km)');ax.set_aspect(1)
    for split,color,label in [('train','#4575b4','Ajuste'),('holdout','#fdae61','Reserva')]:
        q=e[e.split.eq(split)]
        axes[1].hist(q.ev_prof_km,bins=np.arange(0,190,10),histtype='step',color=color,label=label)
        axes[2].hist(q.magnitude.dropna(),bins=np.arange(0,8.5,.25),histtype='step',color=color,label=label)
    axes[1].set(xlabel='Profundidad (km)',ylabel='Número de eventos');axes[2].set(xlabel='Magnitud publicada (tipos mixtos)',ylabel='Número de eventos',title='3606 con magnitud; 15 sin dato');axes[1].legend();axes[2].legend()
    fig.savefig(OUT/'Datos_estaciones_magnitudes_profundidades.png',dpi=180);plt.close(fig)
def velocities():
    # Primary graphical weight fixed to the middle value before any 3D result.
    f=np.load(C/'real_smooth_20/vp_final.npz');fine=Grid(f['xg'],f['yg'],f['zg']);coarse=Grid(f['inversion_xg'],f['inversion_yg'],f['inversion_zg']);active=f['active'].ravel()[mapping(fine,coarse)].reshape(fine.shape)
    x,y=np.meshgrid(f['xg'],f['yg']);lat,lon=pr.latlon(x,y)
    depths=[int(np.argmin(abs(f['zg']-d))) for d in [20,80,140]]
    anomaly=100*(f['vp']/f['vp_initial']-1)
    anomaly_range=max(3.,float(np.ceil(max(abs(anomaly[k][active[k]]).max() for k in depths))))
    for kind in ['velocities','anomalies']:
        fig,axes=plt.subplots(3 if kind=='velocities' else 1,2 if kind=='velocities' else 3,figsize=(9.8,11) if kind=='velocities' else (11,5),squeeze=False,constrained_layout=True)
        cmap=plt.get_cmap('viridis' if kind=='velocities' else 'RdBu').copy();cmap.set_bad('#cfcfcf')
        for row,depth in enumerate([20,80,140]):
            k=np.argmin(abs(f['zg']-depth));arrays=[f['vp_initial'][k],f['vp'][k]] if kind=='velocities' else [100*(f['vp'][k]/f['vp_initial'][k]-1)]
            for col,a in enumerate(arrays):
                ax=axes[row,col] if kind=='velocities' else axes[0,row]
                im=ax.pcolormesh(lon,lat,np.ma.masked_where(~active[k],a),shading='auto',cmap=cmap,vmin=4 if kind=='velocities' else -anomaly_range,vmax=8.8 if kind=='velocities' else anomaly_range)
                boundaries(ax);ax.set(xlabel='Longitud (°)',ylabel='Latitud (°)',xlim=(lon.min(),lon.max()),ylim=(lat.min(),lat.max()),title=f'{depth} km · '+(('Inicial' if col==0 else 'Final λ = 20') if kind=='velocities' else 'Anomalía'));ax.set_aspect(1)
        fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.7,label='Vp (km/s)' if kind=='velocities' else 'ΔVp/Vp inicial (%) · azul rápido, rojo lento')
        fig.savefig(OUT/('Modelos_inicial_final.png' if kind=='velocities' else 'Anomalias_activas.png'),dpi=190);plt.close(fig)
    (C/'figure_support.json').write_text(json.dumps(dict(primary_graphical_lambda=20,selection='predefined middle weight, not chosen by holdout advantage',active_cells=int(f['active'].sum()),anomaly_color_range_percent=anomaly_range,color_rule='symmetric common range enclosing all active values in the three displayed slices; no clipped values',mask='fixed inversion support; not a resolution mask; grey means outside parameter support'),indent=2),encoding='utf-8')
if __name__=='__main__':globals()[sys.argv[1]]()
