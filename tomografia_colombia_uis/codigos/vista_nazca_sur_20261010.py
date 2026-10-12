"""WGS84 final 1D-start tomography, SGC seismicity, stations and two borders."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reconstruir_benioff_ampliado import controls,supported_mesh,model_cells
from nazca_campana_20261010 import projection as Projection
from regional.inputs import sha256
from agregar_mapa_superficie_norte import boundaries,lines
from shapely.geometry import box
from shapely.ops import unary_union

def main():
    run=ROOT/'resultados/migracion_nazca_cauca_valle_eje_1d_2026-10-10';out=ROOT/'resultados/nazca_cauca_valle_eje_1d_vista_2026-10-10';out.mkdir(exist_ok=False)
    domain=json.loads((run/'domain.json').read_text());pr=Projection(domain)
    e=pd.read_csv(run/'relocated_train_events.csv');cat=pd.read_csv(run/'catalogo_visualizacion_ampliado.csv');p=pd.read_csv(run/'associated_picks_selected.csv',low_memory=False)
    s=p[p.fase.eq('P')][['station','st_lat_current','st_lon_current']].drop_duplicates();added=set(pd.read_csv(run/'added_events.csv').event_id)
    with np.load(run/'vp_final.npz') as m:c=model_cells(m,pr)
    c.to_csv(out/'vp_celdas_activas.csv',index=False)
    # Local Benioff triangles are visual guides. Do not bridge empty northern areas.
    good=controls(e);good=good[good.accepted];xyz=good[['x_km','y_km','depth_km']].to_numpy();f=supported_mesh(xyz)
    la,lo=pr.latlon(xyz[:,0],xyz[:,1]);pts=np.c_[lo,la,xyz[:,2]]
    np.savez_compressed(out/'fragmentos_finales.npz',xyz=xyz,triangles=f)
    import plotly.graph_objects as go
    fig=go.Figure()
    fig.add_trace(go.Scatter3d(x=cat.lon,y=cat.lat,z=cat.depth_km,mode='markers',name='Catálogo SGC de contexto',visible='legendonly',marker=dict(size=1,color='#8b8b8b',opacity=.12)))
    for selected,name,color in [(~e.event_id.isin(added),'SGC relocalizados anteriores','#222222'),(e.event_id.isin(added),'SGC relocalizados nuevos Cauca Valle y Eje Cafetero','#cc4025')]:
        d=e[selected&e.status.eq('relocated')];fig.add_trace(go.Scatter3d(x=d.lon,y=d.lat,z=d.depth_km,mode='markers',name=name,marker=dict(size=2.5,color=color,opacity=.6),text=d.event_id,hovertemplate='%{text}<br>Lon %{x:.3f}<br>Lat %{y:.3f}<br>Prof. %{z:.1f} km<extra>'+name+'</extra>'))
    fig.add_trace(go.Mesh3d(x=pts[:,0],y=pts[:,1],z=pts[:,2],i=f[:,0],j=f[:,1],k=f[:,2],name='Benioff: fragmentos locales provisionales',showlegend=True,color='#c4a06d',opacity=.22))
    fig.add_trace(go.Scatter3d(x=c.lon,y=c.lat,z=c.depth_km,mode='markers',name='Vp final: celdas activas',marker=dict(size=4,symbol='square',color=c.dvp_percent,colorscale='RdBu_r',cmin=-15,cmax=15,opacity=.8,colorbar=dict(title='dVp / 1D %')),customdata=c[['vp_km_s','dvp_percent']],hovertemplate='Lon %{x:.3f}<br>Lat %{y:.3f}<br>Prof. %{z:.1f} km<br>Vp final %{customdata[0]:.3f} km/s<br>dVp %{customdata[1]:.2f}%<extra>Cobertura activa, no resolución demostrada</extra>'))
    fig.add_trace(go.Scatter3d(x=s.st_lon_current,y=s.st_lat_current,z=np.zeros(len(s)),mode='markers',marker=dict(size=3,symbol='diamond',color='#78428b'),name='Todas las estaciones/épocas con P',text=s.station,hovertemplate='%{text}<br>Lon %{x:.3f}<br>Lat %{y:.3f}<extra>Receptor proyectado a z=0</extra>'))
    path,meta,features=boundaries();polys=[g[1].intersection(box(-79,.8,-71,13)) for g in features.values()];borders=[(unary_union([g.boundary for g in polys]),'Departamentos','#999999',1),(unary_union(polys).boundary,'Colombia','#202e35',3)]
    for geo,name,color,width in borders:
        x,y,z=[],[],[]
        for q in lines(geo):x.extend(q[:,0].tolist()+[None]);y.extend(q[:,1].tolist()+[None]);z.extend([0]*len(q)+[None])
        fig.add_trace(go.Scatter3d(x=x,y=y,z=z,mode='lines',line=dict(color=color,width=width),name=name))
    prep=json.loads((run/'preparation.json').read_text())
    fig.update_layout(title=f"Caribe y Nazca: Vp final desde inicio 1D, {prep['selected_events']} sismos",height=900,scene=dict(xaxis=dict(title='Longitud WGS84',range=[-79,-71]),yaxis=dict(title='Latitud WGS84',range=[.8,13]),zaxis=dict(title='Profundidad km',range=[180,0]),aspectmode='manual',aspectratio=dict(x=.655,y=1,z=.133)))
    note=f"Inicio 1D original sin placas impuestas; receptores CARMA incorporados. Se agregaron {prep['additional_events']} eventos de Cauca Valle y Eje Cafetero y todas sus estaciones regionales utilizables. Colores: dVp final respecto al mismo 1D. Fragmentos Benioff locales: no asignan identidad de placa ni rellenan huecos sin apoyo. SGC y relocalizaciones de entrenamiento; dos capas fronterizas. Modelo experimental: cobertura activa no demuestra resolución."
    (out/'caribe_nazca_vp_final_3d.html').write_text(fig.to_html(include_plotlyjs=True).replace('<body>','<body><p>'+note+'</p>'),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection,Line3DCollection
    fig=plt.figure(figsize=(15,9));ax=fig.add_subplot(projection='3d');ax.set_position([0,.1,.82,.83])
    ax.add_collection3d(Poly3DCollection(pts[f],facecolor='#c4a06d',edgecolor='none',alpha=.25));old=e[~e.event_id.isin(added)];new=e[e.event_id.isin(added)]
    ax.scatter(old.lon,old.lat,old.depth_km,s=2,c='#444444',alpha=.2);ax.scatter(new.lon,new.lat,new.depth_km,s=5,c='#cc4025',alpha=.6)
    cb=ax.scatter(c.lon,c.lat,c.depth_km,s=8,c=c.dvp_percent,cmap='RdBu_r',vmin=-15,vmax=15,marker='s',alpha=.8)
    ax.scatter(s.st_lon_current,s.st_lat_current,np.zeros(len(s)),s=9,marker='^',c='#78428b',alpha=.7)
    for geo,name,color,width in borders:ax.add_collection3d(Line3DCollection([np.c_[q[:,0],q[:,1],np.zeros(len(q))] for q in lines(geo)],colors=color,linewidths=.3*width))
    ax.set(xlabel='Longitud WGS84',ylabel='Latitud WGS84',zlabel='Profundidad km',xlim=(-79,-71),ylim=(.8,13),zlim=(180,0),zticks=[0,60,120,180]);ax.set_box_aspect([.655,1,.399],zoom=1.02);ax.view_init(elev=24,azim=-65)
    cax=fig.add_axes([.86,.27,.018,.4]);fig.colorbar(cb,cax=cax,label='dVp respecto al 1D original %');fig.suptitle('Caribe y Nazca: Vp final desde 1D con ampliación Cauca Valle y Eje Cafetero',fontsize=16)
    fig.text(.03,.035,'Rojo: nuevos sismos relocalizados. Gris: anteriores. Triángulos: estaciones con P. Color: celdas activas de Vp.\nBenioff local sin identidad tectónica asignada. Exageración vertical 3× en esta imagen; visor interactivo sin exageración.',fontsize=10)
    fig.savefig(out/'caribe_nazca_vp_final.png',dpi=170,bbox_inches='tight');plt.close(fig)
    inputs=[Path(__file__),ROOT/'codigos/nazca_campana_20261010.py',run/'vp_final.npz',run/'relocated_train_events.csv',run/'associated_picks_selected.csv',run/'catalogo_visualizacion_ampliado.csv',run/'added_events.csv',path,meta]
    (out/'metodologia.json').write_text(json.dumps(dict(note=note,active_cells=len(c),stations=len(s),faces=len(f),input_sha256={str(p):sha256(p) for p in inputs}),indent=2),encoding='utf-8');print('Final 1D-start geographic viewer exported',flush=True)
if __name__=='__main__':main()
