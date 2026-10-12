from pathlib import Path
import sys,json,hashlib,html
import numpy as np,pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist
import plotly.graph_objects as go
from shapely.geometry import box
from shapely.ops import unary_union
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'codigos'))
from nazca_campana_20261010 import projection
from agregar_mapa_superficie_norte import boundaries,lines
RUN=ROOT/'resultados/tomografia_1d_adjunto_consistente_2026-10-10'
GEO=ROOT/'resultados/placas_adjunto_consistente_2026-10-10'
OUT=ROOT/'articulo_boletin_geologia_uis_adjunto'
SUP=OUT/'suplementos_20261011';SUP.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
domain=json.loads((RUN/'domain.json').read_text());pr=projection(domain)
e=pd.read_csv(RUN/'relocated_train_events.csv');assert len(e)==2987
m=np.load(RUN/'vp_final.npz');axes=(m['zg'],m['yg'],m['xg']);coarse=(m['inversion_zg'],m['inversion_yg'],m['inversion_xg'])
v0=RegularGridInterpolator(axes,m['vp_initial']);v1=RegularGridInterpolator(axes,m['vp'])
cover=RegularGridInterpolator(coarse,m['active'].astype(float),method='nearest',bounds_error=False,fill_value=0)
path,meta,features=boundaries();polys=[a[1].intersection(box(-79,.8,-71,13)) for a in features.values()]
borders=[(unary_union(polys).boundary,'Colombia','#223244',3),(unary_union([a.boundary for a in polys]),'Departamentos','#989898',1)]
def base(title):
    f=go.Figure()
    for geo,name,color,width in borders:
        x,y,z=[],[],[]
        for q in lines(geo):x.extend(q[:,0].tolist()+[None]);y.extend(q[:,1].tolist()+[None]);z.extend([0]*len(q)+[None])
        f.add_trace(go.Scatter3d(x=x,y=y,z=z,mode='lines',name=name,line=dict(color=color,width=width),hoverinfo='name'))
    f.update_layout(title=title,template='plotly_white',height=780,margin=dict(l=0,r=0,t=65,b=0),
        scene=dict(xaxis=dict(title='Longitud (°)',range=[-79,-71]),yaxis=dict(title='Latitud (° N)',range=[.8,13]),
        zaxis=dict(title='Profundidad (km)',range=[180,0]),aspectmode='manual',aspectratio=dict(x=.655,y=1,z=.399),
        camera=dict(projection=dict(type='orthographic'),eye=dict(x=1.5,y=-1.5,z=1.2))),legend=dict(x=0,y=1))
    return f
def earthquakes(f,ev,name='Sismos SGC relocalizados',color='#202020'):
    f.add_trace(go.Scatter3d(x=ev.lon,y=ev.lat,z=ev.depth_km,mode='markers',name=name,
        marker=dict(size=2,color=color,opacity=.65),text=ev.event_id,
        hovertemplate='SGC %{text}<br>Longitud %{x:.3f}°<br>Latitud %{y:.3f}°<br>Profundidad %{z:.1f} km<extra></extra>'))
def export(f,name,note):
    body=f.to_html(full_html=False,include_plotlyjs=True,div_id='scene3d',config=dict(responsive=True,displaylogo=False))
    nav='<nav><a href="figura_1_fragmentos_sgc.html">Fragmentos</a> · <a href="figura_2_nurbs_nazca_caribe_sgc.html">NURBS</a> · <a href="clasificacion_kmeans_sgc.html">K-means</a></nav>'
    content='<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(f.layout.title.text)+'</title><style>body{font:16px Arial;margin:24px;color:#243044}p{max-width:1100px;line-height:1.5}a{color:#3b3b8e}nav{margin-bottom:16px}</style><body>'+nav+'<p>'+note+'</p><p>Arrastre para girar; use la rueda para acercar. Pulse una entrada de la leyenda para mostrar u ocultar una capa. Profundidad positiva hacia abajo; exageración vertical aproximada 3×. Archivo autónomo, sin conexión a internet.</p>'+body+'</body></html>'
    (SUP/name).write_text(content,encoding='utf-8')
f=base('Figura 1 · Fragmentos locales y sismicidad SGC')
with np.load(ROOT/'resultados/fragmentos_adjunto_consistente_2026-10-10/fragmentos_finales.npz') as b:
    xyz=b['xyz'];faces=b['triangles'];faces=faces[(cover(xyz[:,::-1])>.5)[faces].all(axis=1)]
la,lo=pr.latlon(xyz[:,0],xyz[:,1]);assert len(faces)==331
f.add_trace(go.Mesh3d(x=lo,y=la,z=xyz[:,2],i=faces[:,0],j=faces[:,1],k=faces[:,2],name='Fragmentos locales con cobertura Vp',color='#7a8c91',opacity=.55,showlegend=True))
earthquakes(f,e)
export(f,'figura_1_fragmentos_sgc.html','331 triángulos locales con cobertura Vp activa en sus vértices y 2987 eventos SGC de entrenamiento relocalizados. Son fragmentos de la distribución sísmica, no isosuperficies de Vp. Se conserva el mismo experimento iniciado con 1D de la Figura 1 del artículo.')
f=base('Figura 2 · Nazca morada y Caribe naranja con sismicidad SGC')
for name,color in [('nazca','#7926b3'),('caribe','#ed831b')]:
    with np.load(GEO/f'{name}_superficie.npz') as sh:
        faces=sh['faces'];support=sh['supported'][faces].all(axis=1)
        for choose,label,opacity in [(support,'apoyo local',.8),(~support,'continuidad interpolada',.32)]:
            ff=faces[choose];f.add_trace(go.Mesh3d(x=sh['lon'],y=sh['lat'],z=sh['middle'][:,2],i=ff[:,0],j=ff[:,1],k=ff[:,2],name=name.capitalize()+' candidata · '+label,color=color,opacity=opacity,showlegend=True))
earthquakes(f,e)
export(f,'figura_2_nurbs_nazca_caribe_sgc.html','Superficies medias NURBS candidatas posteriores a la inversión, con los mismos 2987 sismos SGC. Morado: Nazca; naranja: Caribe. Mayor opacidad: apoyo local; menor: interpolación. La clasificación diagonal es una hipótesis geográfica; estas superficies no se ajustaron con k-means. El espesor nominal de 30 km no es una medición tomográfica.')
# K-means receives no diagonal or Slab2 classification during fitting.
ev=e[(e.depth_km>=40)&(e.gap_deg<=180)].copy();X=ev[['x_km','y_km','depth_km']].to_numpy()
ll=ev.lon.to_numpy();bb=6+(ll+72.5)*(9.2-6)/(-76.3+72.5);diag=(ev.lat.to_numpy()>=bb).astype(int)
def silhouette(X,lab):
    D=cdist(X,X);a=np.zeros(len(X));b=np.full(len(X),np.inf)
    for group in np.unique(lab):
        ids=lab==group;n=ids.sum();avg=D[:,ids].sum(axis=1)
        a[ids]=avg[ids]/max(1,n-1);b[~ids]=np.minimum(b[~ids],avg[~ids]/n)
    return float(np.mean((b-a)/np.maximum(a,b)))
solutions={};scores=[]
for metric,data in [('km',X),('standardized', (X-X.mean(axis=0))/X.std(axis=0))]:
    for k in range(2,7):
        candidates=[]
        for seed in range(30):
            centers,lab=kmeans2(data,k,iter=100,minit='++',seed=20261011+seed,missing='raise')
            inertia=float(((data-centers[lab])**2).sum());candidates.append((inertia,centers,lab))
        inertia,centers,lab=min(candidates,key=lambda a:a[0]);score=silhouette(data,lab)
        agreement=max(np.mean(lab==diag),np.mean(1-lab==diag)) if k==2 else None
        row=dict(metric=metric,k=k,n=len(data),inertia=inertia,silhouette=score,agreement_diagonal=agreement)
        if k==2:row['starts_agreement_ge99_percent']=int(sum(max(np.mean(c[2]==lab),np.mean(1-c[2]==lab))>=.99 for c in candidates))
        scores.append(row);solutions[(metric,k)]=(centers,lab)
        ev[f'kmeans_{metric}_k{k}']=lab+1
ev['diagonal_group']=diag;ev.to_csv(SUP/'clasificacion_kmeans_sismos.csv',index=False)
pd.DataFrame(scores).to_csv(SUP/'metricas_kmeans.csv',index=False)
chosen=[r for r in scores if r['metric']=='km'];best=max(chosen,key=lambda a:a['silhouette'])
f=base('Clasificación espacial exploratoria de sismos con k-means')
palette=['#7926b3','#ed831b','#0b8a8f','#cc4b63','#4269ad','#7e8d26'];buttons=[]
for k in range(2,7):
    centers,lab=solutions[('km',k)];start=len(f.data)
    for group in range(k):
        earthquakes(f,ev[lab==group],f'Grupo {group+1} · k={k}',palette[group]);f.data[-1].visible=k==2
    buttons.append(dict(label=f'k = {k}',method='update',args=[dict(visible=[True,True]+[kk==k for kk in range(2,7) for g in range(kk)])]))
f.update_layout(updatemenus=[dict(buttons=buttons,x=.78,y=1.08)])
two=next(r for r in chosen if r['k']==2)
export(f,'clasificacion_kmeans_sgc.html',f'K-means con {len(ev)} eventos SGC de profundidad ≥40 km y brecha ≤180°, distancias euclidianas en x, y y profundidad en km, sin utilizar la diagonal al ajustar. Treinta inicializaciones por k. Para k=2: silueta {two["silhouette"]:.3f}, acuerdo máximo con la diagonal {100*two["agreement_diagonal"]:.1f} %. El mayor índice de silueta entre k=2–6 corresponde a k={best["k"]}. Los grupos no llevan nombres tectónicos: k-means no establece la pertenencia a Nazca o Caribe. La tabla CSV incluye el contraste con coordenadas estandarizadas.')
# Numerical review diagnostics at the same 10011 active coarse nodes.
zz,yy,xx=np.meshgrid(*coarse,indexing='ij');active=m['active'].astype(bool);q=np.c_[zz[active],yy[active],xx[active]]
initial=v0(q);final=v1(q);delta=final/initial-1;eps=1e-6
low=np.abs(delta-(.95**3-1))<=eps;upper=np.abs(final-8.8)<=eps;rows=[]
for depth in coarse[0]:
    sel=q[:,0]==depth;rows.append(dict(depth_km=float(depth),active=int(sel.sum()),lower_saturation=int((low&sel).sum()),upper_saturation=int((upper&sel).sum())))
pd.DataFrame(rows).to_csv(SUP/'saturacion_por_profundidad.csv',index=False)
c=np.load(RUN/'linear_checkerboard.npz');im=c['imposed_slowness'];rec=c['recovered_slowness'];assert len(im)==active.sum()
corr=float(np.corrcoef(im,rec)[0,1]);l2=float(np.linalg.norm(rec-im)/np.linalg.norm(im));amp=float(np.std(rec)/np.std(im));slope=float(np.dot(im,rec)/np.dot(im,im))
plt.rcParams.update({'font.size':12})
cg=np.zeros(tuple(c['shape']),dtype=bool);cg.ravel()[c['active']]=True;assert np.array_equal(cg,active)
truth=np.full(cg.shape,np.nan);recovered=truth.copy();truth[cg]=im;recovered[cg]=rec
index=int(np.argmin(abs(coarse[0]-80)));fig,axs=plt.subplots(1,2,figsize=(10,5),layout='constrained');cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#ddd')
for ax,values,title in zip(axs,[truth[index],recovered[index]],['Patrón impuesto','Patrón recuperado']):
    image=ax.pcolormesh(coarse[2],coarse[1],values,vmin=-.003,vmax=.003,cmap=cmap,shading='nearest');ax.set(title=title+' · 80 km',xlabel='Este local (km)',ylabel='Norte local (km)');ax.set_aspect('equal')
fig.colorbar(image,ax=axs,label='Perturbación de lentitud (s/km)',shrink=.8);fig.savefig(OUT/'figuras/Figura_6_checkerboard_revision.png',dpi=300);plt.close(fig)
round3=np.load(RUN/'model_round_3.npz');hits=round3['ray_hits'];coverage=[]
# These counts refer to round 3 before its update, NOT final kernels.
latco,lonco=pr.latlon(xx,yy);inside=(latco>=.8)&(latco<=13)&(lonco>=-79)&(lonco<=-71)
for threshold in [10,20,40]:coverage.append(dict(minimum_contributions=threshold,active_before_round3=int(((hits>=threshold)&inside).sum())))
held=pd.read_csv(OUT/'fuentes/reserva_inicial_final.csv');depthrows=[]
for a,b in [(0,40),(40,80),(80,120),(120,181)]:
    h=held[held.ev_prof_km.ge(a)&held.ev_prof_km.lt(b)];n=len(h)
    if n:depthrows.append(dict(depth_min_km=a,depth_max_km=min(b,180),events=int(h.event_id.nunique()),p_times=n,initial_rms_s=float(np.sqrt(np.mean(h.residual_initial_s**2))),final_rms_s=float(np.sqrt(np.mean(h.residual_s**2)))))
pd.DataFrame(depthrows).to_csv(SUP/'rms_reserva_por_profundidad.csv',index=False)
# Distance to the proposed diagonal, measured in projected kilometres.
ax,ay=pr.xy(np.array([6,9.2]),np.array([-72.5,-76.3]));origin=np.array([ax[0],ay[0]]);direction=np.array([ax[1]-ax[0],ay[1]-ay[0]])
distance=np.abs((X[:,0]-origin[0])*direction[1]-(X[:,1]-origin[1])*direction[0])/np.linalg.norm(direction)
ev['distance_to_diagonal_km']=distance;ev.to_csv(SUP/'clasificacion_kmeans_sismos.csv',index=False)
report=dict(events_kmeans=len(ev),kmeans=scores,best_k_km=best['k'],lower_saturation=int(low.sum()),lower_percent=float(100*low.mean()),upper_saturation=int(upper.sum()),upper_percent=float(100*upper.mean()),saturation_tolerance=eps,lower_fraction=.95**3,
    checkerboard=dict(correlation=corr,relative_l2=l2,std_amplitude_ratio=amp,least_squares_amplitude_ratio=slope),rms_depth=depthrows,
    diagonal_within_25km=int((distance<=25).sum()),diagonal_within_25km_percent=float(100*np.mean(distance<=25)),coverage_before_round3=coverage,
    input_sha256={str(p):sha(p) for p in [Path(__file__),RUN/'vp_final.npz',RUN/'relocated_train_events.csv',RUN/'linear_checkerboard.npz',RUN/'model_round_3.npz',GEO/'nazca_superficie.npz',GEO/'caribe_superficie.npz',path,meta]},
    output_sha256={str(p):sha(p) for p in SUP.iterdir() if p.is_file()})
(SUP/'revision_diagnosticos.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ['input_sha256','output_sha256','kmeans']},indent=2),flush=True)
