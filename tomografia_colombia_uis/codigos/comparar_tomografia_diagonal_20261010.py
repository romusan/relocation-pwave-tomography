"""Identical-data 1D versus training-informed diagonal slab restart."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
from scipy.interpolate import RegularGridInterpolator
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'codigos'))
from regional.inputs import sha256
from nazca_campana_20261010 import projection
from agregar_mapa_superficie_norte import boundaries,lines
from shapely.geometry import box
from shapely.ops import unary_union
BASE=ROOT/'resultados/migracion_nazca_cauca_valle_eje_1d_2026-10-10'
NEW=ROOT/'resultados/tomografia_placas_diagonal_2026-10-10'
MODEL=ROOT/'resultados/modelo_1d_placas_diagonal_2026-10-10'
GEO=ROOT/'resultados/placas_nazca_caribe_diagonal_2026-10-10'

def rms(x):return float(np.sqrt(np.mean(np.asarray(x)**2)))
def main():
    summaries=[json.loads((r/'summary.json').read_text(encoding='utf-8')) for r in [BASE,NEW]]
    for r in [BASE,NEW]:assert json.loads((r/'verification.json').read_text())['verified']
    a,b=[pd.read_csv(r/'predictions.csv',low_memory=False) for r in [BASE,NEW]]
    x=a.merge(b,on=['event_id','station','fase','raw_row'],suffixes=('_1d','_3d'),validate='one_to_one');assert len(x)==len(a)==len(b)
    for c in ['observed_s','ev_lat','ev_lon','ev_prof_km','st_x_km','st_y_km']:
        np.testing.assert_allclose(x[c+'_1d'],x[c+'_3d'],rtol=0,atol=1e-10)
    assert (x.split_1d==x.split_3d).all();assert summaries[0]['config']==summaries[1]['config'];assert summaries[0]['domain']==summaries[1]['domain'];assert summaries[0]['station_static_reference']==summaries[1]['station_static_reference']
    held=x[x.split_1d.eq('holdout')].copy();assert held.origin_shift_s_1d.eq(0).all() and held.origin_shift_s_3d.eq(0).all()
    held.to_csv(NEW/'comparacion_reserva_pareada.csv',index=False)
    g=held.assign(a=held.residual_s_1d**2,b=held.residual_s_3d**2).groupby('event_id').agg(a=('a','sum'),b=('b','sum'),n=('a','size')).to_numpy()
    rng=np.random.default_rng(20261009);draw=rng.integers(0,len(g),size=(3000,len(g)));total=g[draw].sum(axis=1);diff=np.sqrt(total[:,1]/total[:,2])-np.sqrt(total[:,0]/total[:,2]);ci=np.quantile(diff,[.025,.975])
    before,after=rms(held.residual_s_1d),rms(held.residual_s_3d)
    judgment='El cambio de RMS común no demuestra una mejora: el intervalo incluye cero.' if ci[0]<=0<=ci[1] else ('El RMS de la reserva común mejora en este reinicio condicionado.' if ci[1]<0 else 'El RMS de la reserva común empeora; este modelo inicial no mejora la predicción.')
    data=dict(events=int(held.event_id.nunique()),P=len(held),RMS_1d_s=before,RMS_3d_s=after,delta_s=after-before,delta_ci95_s=ci.tolist(),bootstrap_by_event=3000,judgment=judgment,same_observations_split_grid_config=True,training_prior_reused=True,independent_geological_validation=False,baseline_1d=summaries[0]['baseline'],baseline_3d=summaries[1]['baseline'],train_rms_1d_s=summaries[0]['history'][-1]['train_rms_after_update_s'],train_rms_3d_s=summaries[1]['history'][-1]['train_rms_after_update_s'])
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for summary,label,color in zip(summaries,['Inicio 1D','Inicio 1D + placas NURBS'],['#333333','#7926b3']):
        h=summary['history'];rounds=[0]+[r['round'] for r in h]
        axs[0].plot(rounds,[summary['baseline']['holdout_rms_fixed_catalog_origin_s']]+[r['holdout_rms_fixed_catalog_origin_s'] for r in h],'-o',color=color,label=label)
        axs[1].plot(rounds,[summary['baseline']['train_rms_fixed_catalog_s']]+[r['train_rms_after_update_s'] for r in h],'-o',color=color,label=label)
    for ax,label in zip(axs,['Reserva fija: mismas observaciones','Entrenamiento: relocalizaciones y estáticos']):ax.set(xlabel='Ronda (0: antes de ajustes)',ylabel='RMS (s)',title=label,xticks=[0,1,2,3]);ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.suptitle('Comparación controlada: fondo 1D frente a placas diagonales NURBS');fig.savefig(NEW/'comparacion_rms.png',dpi=170);plt.close(fig)
    domain=summaries[1]['domain'];pr=projection(domain);rel=pd.read_csv(NEW/'relocated_train_events.csv')
    with np.load(NEW/'vp_final.npz') as m,np.load(MODEL/'vp_inicial_1d_placas_30km.npz') as initial:
        for k in ['xg','yg','zg']:np.testing.assert_array_equal(m[k],initial[k])
        np.testing.assert_array_equal(m['vp_initial'],initial['vp'])
        original_background=initial['vp_background_1d'].copy()
        pi=RegularGridInterpolator((m['zg'],m['yg'],m['xg']),m['vp'],bounds_error=False,fill_value=np.nan)
        p0=RegularGridInterpolator((m['zg'],m['yg'],m['xg']),initial['vp_background_1d'],bounds_error=False,fill_value=np.nan)
        active=RegularGridInterpolator((m['inversion_zg'],m['inversion_yg'],m['inversion_xg']),m['active'].astype(float),method='nearest',bounds_error=False,fill_value=0)
        zz,yy,xx=np.meshgrid(m['inversion_zg'],m['inversion_yg'],m['inversion_xg'],indexing='ij');mask=m['active'].astype(bool);centers=np.c_[xx[mask],yy[mask],zz[mask]]
        clat,clon=pr.latlon(centers[:,0],centers[:,1]);cellvp=pi(centers[:,::-1]);celldvp=100*(cellvp/p0(centers[:,::-1])-1)
    shapes=[]
    for name,color in [('nazca','#7926b3'),('caribe','#ed831b')]:
        with np.load(GEO/f'{name}_superficie.npz') as sh:mid=sh['middle'];faces=sh['faces'];pts=np.c_[sh['lon'],sh['lat'],mid[:,2]]
        v=pi(mid[:,::-1]);dv=100*(v/p0(mid[:,::-1])-1);support=active(mid[:,::-1])>.5
        pd.DataFrame(dict(lon=pts[:,0],lat=pts[:,1],depth_km=pts[:,2],vp_final=v,dvp_original_1d_percent=dv,active=support)).to_csv(NEW/f'{name}_vp_final_sobre_geometria_inicial.csv',index=False)
        shapes.append((name,color,pts,faces,v,dv))
    path,meta,features=boundaries();polys=[p[1].intersection(box(-79,.8,-71,13)) for p in features.values()]
    borders=[(unary_union(polys).boundary,'Colombia','#25303c',3),(unary_union([p.boundary for p in polys]),'Departamentos','#999999',1)]
    fig,axs=plt.subplots(2,2,figsize=(11,12),layout='constrained')
    for j,run in enumerate([BASE,NEW]):
        with np.load(run/'vp_final.npz') as fm:
            iy,ix=np.meshgrid(fm['inversion_yg'],fm['inversion_xg'],indexing='ij');fl,fo=pr.latlon(ix.ravel(),iy.ravel());fl=fl.reshape(ix.shape);fo=fo.reshape(ix.shape)
            fpi=RegularGridInterpolator((fm['zg'],fm['yg'],fm['xg']),fm['vp']);fbg=RegularGridInterpolator((fm['zg'],fm['yg'],fm['xg']),original_background)
            for i,depth in enumerate([80,120]):
                k=int(np.argmin(abs(fm['inversion_zg']-depth)));zval=fm['inversion_zg'][k];coords=np.c_[np.full(ix.size,zval),iy.ravel(),ix.ravel()]
                delta=(100*(fpi(coords)/fbg(coords)-1)).reshape(ix.shape);delta=np.ma.array(delta,mask=~fm['active'][k].astype(bool))
                ax=axs[i,j];im=ax.pcolormesh(fo,fl,delta,cmap='RdBu_r',vmin=-15,vmax=15,shading='nearest',rasterized=True)
                for geo,name,color,width in borders:
                    for p in lines(geo):ax.plot(p[:,0],p[:,1],color=color,linewidth=width*.2)
                ax.set(xlim=(-79,-71),ylim=(.8,13),xlabel='Longitud WGS84',ylabel='Latitud WGS84',title=('Inicio 1D' if j==0 else 'Inicio 1D + placas')+f': final a {zval:.0f} km');ax.set_aspect('equal')
    fig.colorbar(im,ax=axs.ravel().tolist(),label='Vp final respecto al mismo 1D original (%)',shrink=.7)
    fig.suptitle('Comparación de Vp final en celdas activas\nBlanco: sin cobertura activa; no se interpola entre huecos');fig.savefig(NEW/'comparacion_vp_final_cortes.png',dpi=150);plt.close(fig)
    import plotly.graph_objects as go
    fig=go.Figure()
    for name,color,p,f,v,dv in shapes:fig.add_trace(go.Mesh3d(x=p[:,0],y=p[:,1],z=p[:,2],i=f[:,0],j=f[:,1],k=f[:,2],color=color,opacity=.45,showlegend=True,name=name.capitalize()+' — geometría inicial',customdata=np.c_[v,dv],hovertemplate='Lon %{x:.3f}<br>Lat %{y:.3f}<br>Prof. %{z:.1f} km<br>Vp final %{customdata[0]:.3f} km/s<br>dVp / 1D %{customdata[1]:.1f}%<extra>Geometría inicial candidata</extra>'))
    fig.add_trace(go.Scatter3d(x=clon,y=clat,z=centers[:,2],mode='markers',visible='legendonly',name='Vp final: celdas activas',marker=dict(size=4,symbol='square',color=celldvp,cmin=-15,cmax=15,colorscale='RdBu_r',colorbar=dict(title='dVp / 1D %'),opacity=.8)))
    r=rel[rel.status.eq('relocated')];fig.add_trace(go.Scatter3d(x=r.lon,y=r.lat,z=r.depth_km,mode='markers',name='SGC relocalizados en nueva inversión',marker=dict(size=1.5,color='#333333',opacity=.35),text=r.event_id))
    for geo,name,color,width in borders:
        x,y,z=[],[],[]
        for p in lines(geo):x.extend(list(p[:,0])+[None]);y.extend(list(p[:,1])+[None]);z.extend([0]*len(p)+[None])
        fig.add_trace(go.Scatter3d(x=x,y=y,z=z,mode='lines',name=name,line=dict(color=color,width=width)))
    note='Tomografía iniciada desde 1D + placas diagonales de 30 km. Morado/naranja: geometría inicial candidata, conservada como referencia; no son interfaces inferidas de nuevo. Puntos: nuevas relocalizaciones SGC. Active Vp final desde la leyenda. Comparación condicionada por el modelo previo de entrenamiento.'
    fig.update_layout(title='Tomografía desde placas: Vp final y geometría inicial Nazca–Caribe',height=900,scene=dict(xaxis=dict(title='Longitud WGS84',range=[-79,-71]),yaxis=dict(title='Latitud WGS84',range=[.8,13]),zaxis=dict(title='Profundidad km',range=[180,0]),aspectmode='manual',aspectratio=dict(x=.655,y=1,z=.133)))
    (NEW/'vp_final_placas_3d.html').write_text(fig.to_html(include_plotlyjs=True).replace('<body>','<body><p>'+note+'</p>'),encoding='utf-8')
    inputs=[Path(__file__),BASE/'predictions.csv',NEW/'predictions.csv',BASE/'summary.json',NEW/'summary.json',MODEL/'metodologia.json',MODEL/'vp_inicial_1d_placas_30km.npz',NEW/'vp_final.npz',NEW/'relocated_train_events.csv',GEO/'nazca_superficie.npz',GEO/'caribe_superficie.npz',path,meta]
    data['source_sha256']={str(p):sha256(p) for p in inputs};(NEW/'comparacion_1d_3d.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    method=json.loads((MODEL/'metodologia.json').read_text());verification=json.loads((NEW/'verification.json').read_text())
    report=f'''# Tomografía: inicio 1D frente a 1D + Nazca y Caribe diagonales

{judgment}

| Resultado | Inicio 1D | Inicio 1D + placas |
|---|---:|---:|
| RMS reserva final (s) | {before:.6f} | {after:.6f} |
| RMS entrenamiento final (s) | {data['train_rms_1d_s']:.6f} | {data['train_rms_3d_s']:.6f} |
| RMS reserva antes de ajustar (s) | {summaries[0]['baseline']['holdout_rms_fixed_catalog_origin_s']:.6f} | {summaries[1]['baseline']['holdout_rms_fixed_catalog_origin_s']:.6f} |

Cambio RMS 3D − 1D: **{after-before:+.6f} s**, IC 95 % **[{ci[0]:+.6f}, {ci[1]:+.6f}]**, mediante 3000 remuestreos por evento. Son los mismos **{data['events']} sismos reservados y {data['P']} P**, con hipocentros y orígenes fijos. Se conservan todas las observaciones, estaciones/épocas, mallas, proyección AEQD original, regularización, tres rondas y referencia de estáticos. La selección no se rehace con el modelo 3D.

## Modelo inicial

El fondo es el perfil 1D original exacto. Se añaden bandas de 30 km normales a las superficies Nazca morada y Caribe naranja, con una transición coseno de 15 km. La diagonal del usuario y la continuidad de las superficies son hipótesis. Los contrastes firmados se toman del Vp final de entrenamiento anterior, en celdas activas a ≤40 km; se limitan a ±3 %. En zonas sin apoyo se prolonga la mediana firmada de cada placa: Nazca {method['surfaces'][0]['median_continuation_contrast_percent']:.4f} % y Caribe {method['surfaces'][1]['median_continuation_contrast_percent']:.4f} %. No se obliga a que toda la placa sea rápida. Los contrastes no se suman donde las bandas se superponen.

Cambian {method['changed_nodes']} nodos del fondo. El Vp inicial varía entre {method['vp_range_km_s'][0]:.4f} y {method['vp_range_km_s'][1]:.4f} km/s. Fuera de las bandas el fondo permanece idéntico. Se comprobó igualdad exacta entre este volumen y el Vp inicial realmente cargado por la inversión.

Este reinicio reutiliza geometría y velocidad de entrenamiento de la inversión anterior; no es validación geológica independiente. La reserva no intervino en la construcción de las placas ni en los contrastes. El espesor, el límite diagonal y los huecos interpolados no quedan confirmados por un RMS más bajo. Tampoco esta comparación implica que toda la sismicidad pertenezca a una sola lámina. El solver y los rayos siguen siendo aproximaciones experimentales; los receptores están proyectados a z=0. No hay una nueva inversión de Q ni de Vs.

## Archivos

- Modelo inicial: `resultados/modelo_1d_placas_diagonal_2026-10-10/vp_inicial_1d_placas_30km.npz`.
- [Cortes del modelo inicial](resultados/modelo_1d_placas_diagonal_2026-10-10/modelo_inicial_cortes.png).
- [Comparación de RMS](resultados/tomografia_placas_diagonal_2026-10-10/comparacion_rms.png).
- [Cortes comparados de Vp final](resultados/tomografia_placas_diagonal_2026-10-10/comparacion_vp_final_cortes.png).
- [Visor 3D del resultado](resultados/tomografia_placas_diagonal_2026-10-10/vp_final_placas_3d.html).
- [Comparación pareada y hashes](resultados/tomografia_placas_diagonal_2026-10-10/comparacion_1d_3d.json).

Se verificaron {verification['source_files_checked']} fuentes y {verification['predictions_checked']} predicciones, sin cambios de reserva ni discrepancias de residuales. Las corridas previas permanecen intactas. Si la reserva empeora, se conserva la inicialización 1D como referencia preferida; no se ajusta el contraste repetidamente para perseguir el mismo conjunto reservado.
'''
    (ROOT/'COMPARACION_1D_PLACAS_DIAGONAL_2026-10-10.md').write_text(report,encoding='utf-8')
    print(json.dumps({k:v for k,v in data.items() if k!='source_sha256'},indent=2),flush=True)
if __name__=='__main__':main()
