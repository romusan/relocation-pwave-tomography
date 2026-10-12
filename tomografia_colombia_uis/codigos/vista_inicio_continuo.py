"""Show continuous prior geometry, normal thickness and source support in WGS84."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from regional.geometry import Projection
from regional.inputs import sha256
from agregar_mapa_superficie_norte import boundaries,lines,ROOT
from shapely.geometry import box
from shapely.ops import unary_union

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True);a=ap.parse_args()
    run=ROOT/'resultados/migracion_isc_choco_2026-10-10';domain=json.loads((run/'domain.json').read_text());pr=Projection(domain)
    import plotly.graph_objects as go
    fig=go.Figure();shapes=json.loads((a.model/'superficies_nurbs_continuas.json').read_text(encoding='utf-8'))
    colors=['#1581a0','#d75b39','#927149']
    for j,(sh,color) in enumerate(zip(shapes,colors),1):
        with np.load(a.model/f'superficie_{j:02d}.npz') as m:
            f=m['faces'];mid=m['middle'];support=m['support_class'];contrast=m['contrast_percent'];normal=m['normals']
            for offset,opacity,suffix in [(0,.67,'media'),(-15,.10,'superior'),(15,.10,'inferior')]:
                xyz=mid+offset*normal;la,lo=pr.latlon(xyz[:,0],xyz[:,1]);custom=np.c_[support,contrast]
                fig.add_trace(go.Mesh3d(x=lo,y=la,z=xyz[:,2],i=f[:,0],j=f[:,1],k=f[:,2],color=color,opacity=opacity,name=sh['name']+' '+suffix,legendgroup=str(j),showlegend=offset==0,customdata=custom,hovertemplate='Longitud %{x:.3f}<br>Latitud %{y:.3f}<br>Profundidad %{z:.1f} km<br>Apoyo %{customdata[0]}<br>Contraste inicial %{customdata[1]:.2f}%<extra>'+sh['name']+'</extra>'))
            if j==1:
                xyz=mid[support==3][::6];la,lo=pr.latlon(xyz[:,0],xyz[:,1]);fig.add_trace(go.Scatter3d(x=lo,y=la,z=xyz[:,2],mode='markers',marker=dict(size=2,color='#e49c32'),name='Prior Slab2 sin sismos locales cercanos'))
            else:
                xyz=mid[support==2][::4];la,lo=pr.latlon(xyz[:,0],xyz[:,1]);fig.add_trace(go.Scatter3d(x=lo,y=la,z=xyz[:,2],mode='markers',marker=dict(size=2,color='#e49c32'),name='Interpolación '+str(j)))
    e=pd.read_csv(run/'relocated_train_events.csv');fig.add_trace(go.Scatter3d(x=e.lon,y=e.lat,z=e.depth_km,mode='markers',marker=dict(size=2,color='#222222',opacity=.35),name='SGC relocalizados entrenamiento'))
    path,meta,features=boundaries();region=box(-79,2,-71,13);polys=[g[1].intersection(region) for g in features.values()]
    for geo,name,color,width in [(unary_union([g.boundary for g in polys]),'Fronteras departamentales','#999999',1),(unary_union(polys).boundary,'Fronteras de Colombia','#202e35',3)]:
        x,y,z=[],[],[]
        for q in lines(geo):x.extend(q[:,0].tolist()+[None]);y.extend(q[:,1].tolist()+[None]);z.extend([0]*len(q)+[None])
        fig.add_trace(go.Scatter3d(x=x,y=y,z=z,mode='lines',line=dict(color=color,width=width),name=name))
    fig.update_layout(title='Inicio 3D: perfil 1D original y placas continuas de 30 km',height=820,scene=dict(xaxis=dict(title='Longitud WGS84',range=[-79,-71]),yaxis=dict(title='Latitud WGS84',range=[2,13]),zaxis=dict(title='Profundidad km',range=[180,0]),aspectmode='manual',aspectratio=dict(x=.726,y=1,z=.148)))
    note='<p>Modelo inicial hipotético. Espesor normal 30 km. Apoyo 1: cerca de sismos; 2: interpolación de Benioff; 3: prior Slab2. Los colores identifican ramas geométricas y no asignan Nazca o Caribe. El fondo conserva exactamente el perfil 1D original.</p>'
    html=fig.to_html(include_plotlyjs=True);html=html.replace('<body>','<body>'+note);(a.model/'modelo_inicial_continuo_3d.html').write_text(html,encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection,Line3DCollection
    fig=plt.figure(figsize=(13,9));ax=fig.add_subplot(projection='3d');ax.set_position([.03,.10,.81,.76])
    for sh,color in zip(shapes,colors):
        xyz=np.asarray(sh['middle']);f=np.asarray(sh['faces']);la,lo=pr.latlon(xyz[:,0],xyz[:,1]);pts=np.c_[lo,la,xyz[:,2]]
        ax.add_collection3d(Poly3DCollection(pts[f],facecolor=color,edgecolor='none',alpha=.42))
    for geo,color,width in [(unary_union([g.boundary for g in polys]),'#999999',.25),(unary_union(polys).boundary,'#202e35',.9)]:ax.add_collection3d(Line3DCollection([np.c_[q[:,0],q[:,1],np.zeros(len(q))] for q in lines(geo)],colors=color,linewidths=width))
    ax.set(xlabel='Longitud WGS84',ylabel='Latitud WGS84',zlabel='Profundidad km',xlim=(-79,-71),ylim=(2,13),zlim=(180,0));ax.set_box_aspect([.726,1,.148],zoom=.8);ax.view_init(elev=27,azim=-65)
    fig.suptitle('Inicio 3D con superficies continuas Slab2 y Benioff',fontsize=15);fig.text(.04,.035,'Azul: Slab2 SAM y Benioff. Rojo y ocre: ramas de Benioff. Hipótesis geométricas sin identidad tectónica confirmada.',fontsize=9)
    fig.savefig(a.model/'vista_inicio_continuo.png',dpi=160,bbox_inches='tight');plt.close(fig)
    (a.model/'viewer_manifest.json').write_text(json.dumps({str(f.resolve()):sha256(f) for f in [Path(__file__),path,meta,a.model/'superficies_nurbs_continuas.json',a.model/'metodologia.json']},indent=2))
    print('Continuous prior viewer exported; exactly two boundary layers')
if __name__=='__main__':main()
