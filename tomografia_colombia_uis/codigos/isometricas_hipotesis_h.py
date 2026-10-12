"""Publication isometric views of the same final-h NURBS and k-means data."""
from pathlib import Path
import json,sys,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection,Line3DCollection
from shapely.geometry import box
from shapely.ops import unary_union
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'codigos'))
from agregar_mapa_superficie_norte import boundaries,lines
C=ROOT/'resultados/campana_perfil_suavidad_20261011h';H=C/'hipotesis_placas'
A=ROOT/'articulo_boletin_geologia_uis_adjunto';OUT=A/'figuras_revision_h'
path,meta,features=boundaries();polys=[a[1].intersection(box(-79,.8,-71,13)) for a in features.values()]
borders=[(unary_union(polys).boundary,'Colombia','#24313c',.9),(unary_union([a.boundary for a in polys]),'Departamentos','#909090',.3)]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
config=json.loads((ROOT/'datos/limite_diagonal_nazca_caribe_20261010.json').read_text());p0,p1=np.array(config['endpoints_lon_lat'])
ll=np.linspace(-79,-71,160);lat=p0[1]+(ll-p0[0])*(p1[1]-p0[1])/(p1[0]-p0[0])
def base():
    fig=plt.figure(figsize=(10,8));ax=fig.add_subplot(projection='3d');ax.set_position([.02,.15,.96,.82]);ax.set_proj_type('ortho')
    for geo,name,color,width in borders:
        ax.add_collection3d(Line3DCollection([np.c_[q[:,0],q[:,1],np.zeros(len(q))] for q in lines(geo)],colors=color,linewidths=width))
    ax.plot(ll,lat,np.zeros(len(ll)),color='#202020',lw=1.3,ls='--')
    ax.set(xlabel='Longitud (°)',ylabel='Latitud (° N)',zlabel='Profundidad (km)',xlim=(-79,-71),ylim=(.8,13),zlim=(180,0),zticks=[0,40,80,120,160],xticks=[-78,-76,-74,-72],yticks=[2,4,6,8,10,12])
    ax.set_box_aspect([.655,1,.399]);ax.view_init(elev=35.2643897,azim=-55)
    ax.xaxis.labelpad=8;ax.yaxis.labelpad=8;ax.zaxis.labelpad=8
    return fig,ax
events=pd.read_csv(C/'real_smooth_20/relocated_train_events.csv');assert len(events)==2897
fig,ax=base()
for name,color in [('nazca','#7926b3'),('caribe','#ed831b')]:
    a=np.load(H/f'{name}_superficie.npz');pts=np.c_[a['lon'],a['lat'],a['middle'][:,2]];faces=a['faces'];local=a['supported'][faces].all(axis=1)
    for selected,alpha in [(~local,.28),(local,.80)]:
        ax.add_collection3d(Poly3DCollection(pts[faces[selected]],facecolors=color,edgecolors='none',alpha=alpha,rasterized=True))
ax.scatter(events.lon,events.lat,events.depth_km,s=2,c='#202020',alpha=.38,depthshade=False)
fig.legend(handles=[Patch(facecolor='#7926b3',label='Nazca candidata'),Patch(facecolor='#ed831b',label='Caribe candidata'),Line2D([],[],color='black',marker='.',ls='none',label='SGC relocalizados'),Line2D([],[],color='black',ls='--',label='Diagonal propuesta')],loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.03),fontsize=10)
fig.savefig(OUT/'Isometrica_Nazca_Caribe_h.png',dpi=300);plt.close(fig)
groups=pd.read_csv(H/'kmeans_sismos.csv');bis=pd.read_csv(H/'bisectores_kmeans.csv');assert len(groups)==1160
fig,ax=base()
for label,color in [(0,'#7926b3'),(1,'#ed831b')]:
    ev=groups[groups.km_k2.eq(label)];ax.scatter(ev.lon,ev.lat,ev.depth_km,s=6,c=color,alpha=.65,depthshade=False)
for depth,style in [(60,':'),(80,'-.'),(100,'-')]:
    g=bis[bis.metric.eq('km')&bis.depth_km.eq(depth)];ax.plot(g.lon,g.lat,g.depth_km,color='#333333',lw=1.4,ls=style)
fig.legend(handles=[Line2D([],[],color='#7926b3',marker='o',ls='none',label='Grupo 1'),Line2D([],[],color='#ed831b',marker='o',ls='none',label='Grupo 2'),Line2D([],[],color='black',ls='--',label='Diagonal propuesta')]+[Line2D([],[],color='#333333',ls=style,label=f'Bisector a {depth} km') for depth,style in [(60,':'),(80,'-.'),(100,'-')]],loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.025),fontsize=10)
fig.savefig(OUT/'Isometrica_kmeans_h.png',dpi=300);plt.close(fig)
for name in ['Isometrica_Nazca_Caribe_h.png','Isometrica_kmeans_h.png']:(A/'suplementos'/name).write_bytes((OUT/name).read_bytes())
source=[Path(__file__),C/'real_smooth_20/relocated_train_events.csv',H/'nazca_superficie.npz',H/'caribe_superficie.npz',H/'kmeans_sismos.csv',H/'bisectores_kmeans.csv',path]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report=dict(camera=dict(projection='orthographic',elevation_deg=35.2643897,azimuth_deg=-55,vertical_exaggeration_approx=3),nurbs_earthquakes=2897,kmeans_earthquakes=1160,kmeans_fit_cells=706,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in source},output_sha256={name:sha(OUT/name) for name in ['Isometrica_Nazca_Caribe_h.png','Isometrica_kmeans_h.png']})
(H/'isometricas_procedencia.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
