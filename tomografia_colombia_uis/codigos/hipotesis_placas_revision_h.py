"""Rebuild geometric hypotheses from the completed h model, without reinversion.

NURBS are cubic B-splines with unit rational weights. Tectonic names encode
the historical geographic hypothesis, independently of the unsupervised fit.
"""
from pathlib import Path
import sys,json,html,hashlib
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from shapely.geometry import MultiPoint,box
from shapely import contains_xy
from shapely.ops import unary_union
import plotly.graph_objects as go
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'codigos'))
from nazca_campana_20261010 import projection
from nazca_nurbs_morada_20261010 import fit_sheet
from modelo_3d_continuo import evaluate,grd
from agregar_mapa_superficie_norte import boundaries,lines
C=ROOT/'resultados/campana_perfil_suavidad_20261011h'
RUN=C/'real_smooth_20';OUT=C/'hipotesis_placas';OUT.mkdir(exist_ok=True)
SUP=ROOT/'articulo_boletin_geologia_uis_adjunto/suplementos'
assert json.loads((C/'completed.json').read_text())['status']=='completed'
domain_path=ROOT/'resultados/campana_factorizada_cotas_20261011g/real_elevation/domain.json'
pr=projection(json.loads(domain_path.read_text()))
e=pd.read_csv(RUN/'relocated_train_events.csv')
descriptors=pd.read_csv(C/'event_descriptors.csv')
assert set(e.event_id)==set(descriptors.query("split=='train'").event_id)
assert e.event_id.str.startswith('SGC').all() and len(e)==2897
m=np.load(RUN/'vp_final.npz');axes=(m['zg'],m['yg'],m['xg'])
vp=RegularGridInterpolator(axes,m['vp'],bounds_error=False,fill_value=np.nan)
v0=RegularGridInterpolator(axes,m['vp_initial'],bounds_error=False,fill_value=np.nan)
coarse=(m['inversion_zg'],m['inversion_yg'],m['inversion_xg'])
support=RegularGridInterpolator(coarse,m['active'].astype(float),method='nearest',bounds_error=False,fill_value=0)
diag_path=ROOT/'datos/limite_diagonal_nazca_caribe_20261010.json'
ends=np.array(json.loads(diag_path.read_text())['endpoints_lon_lat'])
def diagonal(lon):return ends[0,1]+(np.asarray(lon)-ends[0,0])*np.diff(ends[:,1])[0]/np.diff(ends[:,0])[0]
good=e[e.status.eq('relocated')&e.depth_km.ge(40)&e.gap_deg.le(180)&e.lat.between(.8,13)&e.lon.between(-79,-71)].copy()
good[['ix','iy','iz']]=np.floor(good[['x_km','y_km','depth_km']]/20).astype(int)
cells=good.groupby(['ix','iy','iz']).agg(x=('x_km','median'),y=('y_km','median'),z=('depth_km','median'),n=('event_id','size')).reset_index()
la,lo=pr.latlon(cells.x.to_numpy(),cells.y.to_numpy());cells['lat']=la;cells['lon']=lo
path,meta,features=boundaries();polys=[a[1].intersection(box(-79,.8,-71,13)) for a in features.values()]
borders=[(unary_union(polys).boundary,'Colombia','#223244',3),(unary_union([a.boundary for a in polys]),'Departamentos','#989898',1)]
def base(title):
    f=go.Figure()
    for geo,name,color,width in borders:
        xx,yy,zz=[],[],[]
        for q in lines(geo):xx.extend(q[:,0].tolist()+[None]);yy.extend(q[:,1].tolist()+[None]);zz.extend([0]*len(q)+[None])
        f.add_trace(go.Scatter3d(x=xx,y=yy,z=zz,mode='lines',name=name,line=dict(color=color,width=width),hoverinfo='name'))
    f.update_layout(title=dict(text=title.replace(' · ','<br>',1),x=.5,font=dict(size=16)),template='plotly_white',height=850,margin=dict(l=0,r=0,t=85,b=0),
      scene=dict(xaxis=dict(title='Longitud WGS84 (°)',range=[-79,-71]),yaxis=dict(title='Latitud WGS84 (° N)',range=[.8,13]),zaxis=dict(title='Profundidad (km)',range=[180,0]),aspectmode='manual',aspectratio=dict(x=.655,y=1,z=.399),camera=dict(projection=dict(type='orthographic'),eye=dict(x=1.5,y=-1.5,z=1.2))),legend=dict(x=0,y=1,font=dict(size=11)))
    return f
def quakes(f,ev,name='SGC relocalizados · entrenamiento',color='#202020',visible=True):
    f.add_trace(go.Scatter3d(x=ev.lon,y=ev.lat,z=ev.depth_km,mode='markers',name=name,visible=visible,
      marker=dict(size=2,color=color,opacity=.55),text=ev.event_id,hovertemplate='ID %{text}<br>Lon %{x:.3f}°<br>Lat %{y:.3f}°<br>Prof. %{z:.1f} km<extra></extra>'))
def export(f,name,note):
    nav='<nav><a href="figura_1_fragmentos_sgc.html">Anomalías 3D</a> · <a href="figura_2_nurbs_nazca_caribe_sgc.html">Nazca y Caribe NURBS</a> · <a href="clasificacion_kmeans_sgc.html">División k-means</a></nav>'
    content='<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(f.layout.title.text)+'</title><style>body{font:16px Arial;margin:24px;color:#243044}p{max-width:1200px;line-height:1.5}a{color:#503a8c}nav{margin-bottom:16px}</style><body>'+nav+'<p>'+note+'</p><p>Campaña final h, λ = 20. Arrastre para girar, rueda para acercar y leyenda para ocultar capas. Latitud, longitud y profundidad positiva hacia abajo; exageración vertical aproximada 3×. HTML autónomo. Hipótesis de trabajo sometida a evaluación; no constituye una frontera tectónica confirmada.</p>'+f.to_html(full_html=False,include_plotlyjs=True,div_id='scene3d',config=dict(responsive=True,displaylogo=False))+'</body></html>'
    (SUP/name).write_text(content,encoding='utf-8')
    f.write_json(OUT/(Path(name).stem+'.plotly.json'))

# Marching tetrahedra: only cubes with all eight active vertices, no zero fill.
zz,yy,xx=np.meshgrid(*coarse,indexing='ij');xyz=np.stack([xx,yy,zz],axis=-1)
values=100*(vp(xyz[...,::-1])/v0(xyz[...,::-1])-1)
active=m['active'].astype(bool)&(zz>=40)&np.isfinite(values)
corners=np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
idx=np.indices(tuple(np.array(active.shape)-1)).reshape(3,-1).T
ci=idx[:,None,:]+corners[None,:,:]
valid=active[ci[:,:,0],ci[:,:,1],ci[:,:,2]].all(axis=1);ci=ci[valid]
cv=values[ci[:,:,0],ci[:,:,1],ci[:,:,2]];cx=xyz[ci[:,:,0],ci[:,:,1],ci[:,:,2]]
tets=np.array([[0,1,3,7],[0,3,2,7],[0,2,6,7],[0,6,4,7],[0,4,5,7],[0,5,1,7]])
edges=[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
def isomesh(level):
    vertices=[];faces=[]
    for ids in tets:
        tv=cv[:,ids];tx=cx[:,ids];hit=(tv.min(axis=1)<level)&(tv.max(axis=1)>level)
        for v,x in zip(tv[hit],tx[hit]):
            points=[]
            for a,b in edges:
                if (v[a]<level)!=(v[b]<level):points.append(x[a]+(level-v[a])/(v[b]-v[a])*(x[b]-x[a]))
            if len(points)<3:continue
            p=np.asarray(points)
            if len(p)==4:
                center=p.mean(axis=0);u=p[0]-center;u/=np.linalg.norm(u);normal=np.cross(p[1]-p[0],p[2]-p[0]);normal/=np.linalg.norm(normal);w=np.cross(normal,u)
                p=p[np.argsort(np.arctan2((p-center)@w,(p-center)@u))]
            off=len(vertices);vertices.extend(p);faces.append([off,off+1,off+2])
            if len(p)==4:faces.append([off,off+2,off+3])
    return np.asarray(vertices).reshape(-1,3),np.asarray(faces,dtype=int).reshape(-1,3)
f=base('Fragmentos candidatos de las anomalías Vp · modelo final h');iso=[]
for level in [1.,3.,5.,-1.,-3.,-5.]:
    pts,faces=isomesh(level);lat,lon=pr.latlon(pts[:,0],pts[:,1])
    np.savez_compressed(OUT/f'anomalia_{level:+g}.npz',xyz=pts,faces=faces,level_percent=level)
    iso.append(dict(level_percent=level,triangles=len(faces),vertices=len(pts)))
    if len(faces):f.add_trace(go.Mesh3d(x=lon,y=lat,z=pts[:,2],i=faces[:,0],j=faces[:,1],k=faces[:,2],name=f'ΔVp {level:+g}% · '+('rápido' if level>0 else 'lento'),color='#386cb0' if level>0 else '#d73027',opacity=.48,showlegend=True,visible=True if abs(level)==3 else 'legendonly',hovertemplate=f'ΔVp {level:+g}%<br>Lon %{{x:.3f}}<br>Lat %{{y:.3f}}<br>Prof. %{{z:.1f}} km<extra></extra>'))
quakes(f,e)
export(f,'figura_1_fragmentos_sgc.html','Fragmentos extraídos de ΔVp = 100(Vp final/Vp inicial 1D − 1), en niveles ±1, ±3 y ±5 %. Inicialmente se muestran ±3 %; active otros niveles en la leyenda. Triangulación por tetraedros sobre muestreo de 20 km, solo en cubos con sus ocho nodos activos y profundidades ≥40 km, sin rellenar huecos. Azul: rápido; rojo: lento. Se superponen los 2897 eventos de entrenamiento con ID SGC relocalizados. Son anomalías calculadas candidatas, cuya atribución a placas debe contrastarse; el damero no valida su recuperación.')
print('Anomaly fragments',iso,flush=True)

# Unsupervised clustering: one median per occupied 20 km 3D cell.
X=cells[['x','y','z']].to_numpy();mu=X.mean(axis=0);sigma=X.std(axis=0)
def silhouette(data,lab):
    dist=cdist(data,data);a=np.zeros(len(lab));b=np.full(len(lab),np.inf)
    for g in np.unique(lab):
        ids=lab==g;count=ids.sum();s=dist[:,ids].sum(axis=1);a[ids]=s[ids]/max(count-1,1);b[~ids]=np.minimum(b[~ids],s[~ids]/count)
    scores=(b-a)/np.maximum(a,b)
    for g in np.unique(lab):
        if (lab==g).sum()==1:scores[lab==g]=0
    return float(scores.mean())
solutions={};metrics=[];di=(cells.lat.to_numpy()>=diagonal(cells.lon.to_numpy())).astype(int)
for metric,data in [('km',X),('estandarizado',(X-mu)/sigma)]:
    for k in range(2,7):
        attempts=[]
        for start in range(30):
            centers,lab=kmeans2(data,k,iter=100,minit='++',seed=20261012+start,missing='raise')
            attempts.append((float(((data-centers[lab])**2).sum()),centers,lab))
        inertia,centers,lab=min(attempts,key=lambda a:a[0]);solutions[metric,k]=(centers,lab)
        row=dict(metric=metric,k=k,cells=len(X),inertia=inertia,silhouette=silhouette(data,lab))
        if k==2:row.update(agreement_diagonal=float(max(np.mean(lab==di),np.mean(1-lab==di))),stable_starts_99pct=int(sum(max(np.mean(a[2]==lab),np.mean(1-a[2]==lab))>=.99 for a in attempts)))
        metrics.append(row);cells[f'{metric}_k{k}']=lab
cells.to_csv(OUT/'kmeans_celdas.csv',index=False);pd.DataFrame(metrics).to_csv(OUT/'metricas_kmeans.csv',index=False)
centroid_rows=[]
for (metric,k),(centers,_) in solutions.items():
    physical=centers if metric=='km' else centers*sigma+mu
    for group,center in enumerate(physical):centroid_rows.append(dict(metric=metric,k=k,group=group,x_km=center[0],y_km=center[1],depth_km=center[2]))
pd.DataFrame(centroid_rows).to_csv(OUT/'kmeans_centroides.csv',index=False)
f=base('Hipótesis de división espacial · k-means sin etiquetas tectónicas')
ll=np.linspace(-79,-71,160);f.add_trace(go.Scatter3d(x=ll,y=diagonal(ll),z=np.zeros(len(ll)),name='Diagonal histórica propuesta',mode='lines',line=dict(color='black',width=4,dash='dash')))
bisectors=[]
for metric in ['km','estandarizado']:
    centers,_=solutions[metric,2];delta=centers[1]-centers[0];rhs=(centers[1]@centers[1]-centers[0]@centers[0])/2
    if metric=='estandarizado':delta=delta/sigma;rhs+=delta@mu
    for depth in [60,80,100]:
        xs=np.linspace(X[:,0].min(),X[:,0].max(),180);ys=(rhs-delta[0]*xs-delta[2]*depth)/delta[1]
        lat,lon=pr.latlon(xs,ys);ok=(lat>=.8)&(lat<=13)&(lon>=-79)&(lon<=-71)
        f.add_trace(go.Scatter3d(x=lon[ok],y=lat[ok],z=np.full(ok.sum(),depth),mode='lines',visible=metric=='km',name=f'Bisector {metric} k=2 · {depth} km',line=dict(color='#353535',width=4,dash='dash')))
        bisectors.append(pd.DataFrame(dict(lon=lon[ok],lat=lat[ok],depth_km=depth,metric=metric)))
pd.concat(bisectors).to_csv(OUT/'bisectores_kmeans.csv',index=False)
fixed=len(f.data);groups=[];palette=['#7926b3','#ed831b','#238b8d','#cf476a','#4369ad','#78942a']
for metric in ['km','estandarizado']:
    for k in range(2,7):
        centers,lab=solutions[metric,k];data=good[['x_km','y_km','depth_km']].to_numpy();data=data if metric=='km' else (data-mu)/sigma
        assigned=cdist(data,centers).argmin(axis=1);good[f'{metric}_k{k}']=assigned
        for g in range(k):quakes(f,good[assigned==g],f'Grupo {g+1} · k={k} · {metric}',palette[g],metric=='km' and k==2);groups.append((metric,k))
buttons=[]
for metric in ['km','estandarizado']:
    for k in range(2,7):buttons.append(dict(label=f'{metric} · k={k}',method='update',args=[dict(visible=[True,True,True]+[metric=='km' and k==2]*3+[metric=='estandarizado' and k==2]*3+[a==(metric,k) for a in groups])]))
assert fixed==9
f.update_layout(updatemenus=[dict(buttons=buttons,x=.98,y=.96,xanchor='right',yanchor='top')])
good.to_csv(OUT/'kmeans_sismos.csv',index=False)
two=next(r for r in metrics if r['metric']=='km' and r['k']==2);best=max([r for r in metrics if r['metric']=='km'],key=lambda a:a['silhouette'])
export(f,'clasificacion_kmeans_sgc.html',f'K-means ajustado a {len(cells)} celdas de 20 km, con igual peso por celda, a partir de {len(good)} sismos SGC profundos (≥40 km) y brecha ≤180°. Treinta inicios por k, k=2–6; contraste de distancia física en km y coordenadas estandarizadas. La diagonal y Slab2 no intervienen en el ajuste. Para k=2 en km: silueta {two["silhouette"]:.3f}, acuerdo con la diagonal {100*two["agreement_diagonal"]:.1f} %, estabilidad ≥99 % en {two["stable_starts_99pct"]}/30 inicios. Máxima silueta en km: k={best["k"]}. Los puntos reciben el centroide más próximo después del ajuste por celdas; los colores no asignan identidad de placa. Se comparan la diagonal histórica a z=0 y los bisectores a 60, 80 y 100 km. La división depende de la métrica y la profundidad; no prueba contacto físico de placas.')
print('Kmeans',two,'best k',best['k'],flush=True)

# Continuous surfaces retain the historical diagonal, not the k-means labels.
slab=ROOT/'datos/slab2_2026-10-10';sx,sy,sd=grd(slab/'sam_slab2_dep_02.23.18.grd');sd=-sd
_,_,unc=grd(slab/'sam_slab2_unc_02.23.18.grd');slabdepth=RegularGridInterpolator((sy,sx),sd,bounds_error=False,fill_value=np.nan)
slabsig=RegularGridInterpolator((sy,sx),unc,bounds_error=False,fill_value=np.nan)
gx=np.arange(m['xg'].min(),m['xg'].max()+.1,5);gy=np.arange(m['yg'].min(),m['yg'].max()+.1,5)
xx,yy=np.meshgrid(gx,gy);xy=np.c_[xx.ravel(),yy.ravel()];lat,lon=pr.latlon(xy[:,0],xy[:,1])
inside=(lat>=.8)&(lat<=13)&(lon>=-79)&(lon<=-71)
held=descriptors.query("split=='holdout' and ev_prof_km>=40").copy();hx,hy=pr.xy(held.ev_lat.to_numpy(),held.ev_lon.to_numpy());hxy=np.c_[hx,hy]
f=base('Nazca morada y Caribe naranja · NURBS actualizadas con datos h');sheets=[]
for name,color,north,nx,ny,smooth in [('Nazca','#7926b3',False,14,18,.8),('Caribe','#ed831b',True,10,14,2.)]:
    region=(cells.lat.to_numpy()>=diagonal(cells.lon.to_numpy()))==north
    ctrl=cells[region][['x','y','z']].to_numpy();assert len(ctrl)>25
    hull=MultiPoint(ctrl[:,:2]).convex_hull.buffer(75)
    usable=inside&((lat>=diagonal(lon))==north)&contains_xy(hull,xy[:,0],xy[:,1])
    if name=='Nazca':
        ax,ay=np.meshgrid(sx[::3],sy[::3]);az=sd[::3,::3]
        ok=np.isfinite(az)&(az>=0)&(az<=165)&(ax>=-79)&(ax<=-71)&(ay>=.8)&(ay<=13)&(ay<diagonal(ax))
        px,py=pr.xy(ay[ok],ax[ok]);anchors=np.c_[px,py];target=az[ok]+15;weights=.15*8/np.clip(np.nan_to_num(slabsig(np.c_[ay[ok],ax[ok]]),nan=20),5,30)
        raw=slabdepth(np.c_[lat,lon]);usable|=inside&~(lat>=diagonal(lon))&np.isfinite(raw)&(raw>=0)&(raw<=165)
        anchor_note='Slab2 SAM +15 km, weak weight 0.15*8/clip(sigma,5,30); no seismic compatibility filter'
    else:
        anchors=np.array(hull.exterior.coords);anchors=anchors[::max(1,len(anchors)//100)]
        _,ids=cKDTree(ctrl[:,:2]).query(anchors);target=ctrl[ids,2];weights=np.full(len(anchors),.12)
        anchor_note='Nearest seismic cell depths on 75 km buffer, weight 0.12; no Caribbean Slab2 anchors'
    fitxy=np.vstack([anchors,ctrl[:,:2]]);fitz=np.r_[target,ctrl[:,2]];w=np.r_[weights,np.ones(len(ctrl))]
    sh=fit_sheet(fitxy,fitz,w,nx,ny,smooth)
    for _ in range(3):
        predicted,_=evaluate(sh,ctrl[:,:2]);err=np.abs(predicted[:,2]-ctrl[:,2]);w[-len(ctrl):]=np.sqrt(np.minimum(1,15/np.maximum(err,1e-9)));sh=fit_sheet(fitxy,fitz,w,nx,ny,smooth)
    uv=(xy-sh['lo'])/sh['span'];usable&=((uv>=0)&(uv<=1)).all(axis=1)
    mid,normals=evaluate(sh,xy[usable]);assert np.isfinite(mid).all() and (mid[:,2]>=10-1e-6).all() and (mid[:,2]<=180+1e-6).all()
    mapping=np.full(len(xy),-1,int);mapping[usable]=np.arange(usable.sum())
    ids=np.arange(len(xy)).reshape(len(gy),len(gx));quads=np.c_[ids[:-1,:-1].ravel(),ids[:-1,1:].ravel(),ids[1:,:-1].ravel(),ids[1:,1:].ravel()];quads=mapping[quads[usable[quads].all(axis=1)]]
    faces=np.vstack([quads[:,[0,1,2]],quads[:,[1,3,2]]]);assert len(faces)>0
    distance,_=cKDTree(ctrl).query(mid);local=(distance<=40)&(support(mid[:,::-1])>.5)
    ml,mo=pr.latlon(mid[:,0],mid[:,1]);speed=vp(mid[:,::-1]);dvp=100*(speed/v0(mid[:,::-1])-1)
    predicted,_=evaluate(sh,ctrl[:,:2]);rms=float(np.sqrt(np.mean((predicted[:,2]-ctrl[:,2])**2)))
    huv=(hxy-sh['lo'])/sh['span'];hmask=(((held.ev_lat.to_numpy()>=diagonal(held.ev_lon.to_numpy()))==north)&((huv>=0)&(huv<=1)).all(axis=1)&contains_xy(hull,hxy[:,0],hxy[:,1]))
    hp,_=evaluate(sh,hxy[hmask]);held_rms=float(np.sqrt(np.mean((hp[:,2]-held.ev_prof_km.to_numpy()[hmask])**2))) if hmask.any() else None
    np.savez_compressed(OUT/f'{name.lower()}_superficie.npz',middle=mid,normals=normals,faces=faces,lon=mo,lat=ml,supported=local,vp_final=speed,dvp_percent=dvp,control_points=sh['control_points'],weights=sh['weights'],knots_u=sh['knots_u'],knots_v=sh['knots_v'],lo=sh['lo'],span=sh['span'],height=sh['height'],fit_xy=fitxy,fit_depth_km=fitz,fit_row_weights=w)
    pd.DataFrame(dict(lon=mo,lat=ml,depth_km=mid[:,2],vp_final=speed,dvp_percent=dvp,nearest_control_km=distance,active_local_support=local)).to_csv(OUT/f'{name.lower()}_superficie.csv',index=False)
    np.savez_compressed(OUT/f'{name.lower()}_espesor_30km.npz',middle=mid,top=mid-15*normals,bottom=mid+15*normals,faces=faces)
    for select,label,opacity in [(local[faces].all(axis=1),'apoyo local',.82),(~local[faces].all(axis=1),'continuidad interpolada',.30)]:
        fs=faces[select]
        if len(fs):f.add_trace(go.Mesh3d(x=mo,y=ml,z=mid[:,2],i=fs[:,0],j=fs[:,1],k=fs[:,2],name=name+' candidata · '+label,color=color,opacity=opacity,showlegend=True,customdata=np.c_[speed,dvp,distance],hovertemplate='Lon %{x:.3f}<br>Lat %{y:.3f}<br>Prof. %{z:.1f} km<br>Vp h %{customdata[0]:.3f} km/s<br>ΔVp %{customdata[1]:.2f}%<br>Control cercano %{customdata[2]:.1f} km<extra>'+name+' candidata</extra>'))
    sheets.append(dict(name=name,color=color,controls=len(ctrl),control_cells_rms_vertical_km=rms,heldout_catalog_events=int(hmask.sum()),heldout_vertical_rms_km=held_rms,triangles=len(faces),locally_supported_triangles=int(local[faces].all(axis=1).sum()),anchor_note=anchor_note,grid_shape=[nx,ny],smooth=smooth,degree=3,unit_rational_weights=True,nominal_thickness_km=30))
quakes(f,e);f.add_trace(go.Scatter3d(x=ll,y=diagonal(ll),z=np.zeros(len(ll)),name='Diagonal geográfica propuesta · no estimada',mode='lines',line=dict(color='black',width=4,dash='dash')))
export(f,'figura_2_nurbs_nazca_caribe_sgc.html',f'Nazca candidata morada y Caribe candidata naranja, reajustadas a celdas sísmicas profundas de la campaña h; se superponen los 2897 sismos de entrenamiento. NURBS cúbicas con pesos racionales unitarios (caso B-spline), ajustes robustos y profundidades acotadas. La división histórica es un supuesto geográfico independiente de k-means; no se impone contacto en profundidad. Nazca incorpora anclajes débiles Slab2 SAM, sin descartar sismos por compatibilidad con Slab2; Caribe utiliza anclajes sísmicos débiles. Opaco: distancia 3D ≤40 km a controles y Vp activo; translúcido: continuidad interpolada. El espesor nominal de 30 km es un escenario, no una medición. Las muestras Vp/ΔVp pueden consultarse al pasar el cursor. No se cambió el volumen tomográfico ni su RMS.')
print('NURBS',sheets,flush=True)
source=[Path(__file__),RUN/'vp_final.npz',RUN/'relocated_train_events.csv',C/'event_descriptors.csv',domain_path,diag_path,path,slab/'sam_slab2_dep_02.23.18.grd',slab/'sam_slab2_unc_02.23.18.grd',ROOT/'codigos/nazca_nurbs_morada_20261010.py',ROOT/'codigos/modelo_3d_continuo.py']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report=dict(campaign='h',arm='real_smooth_20',train_events=len(e),deep_quality_events=len(good),equal_weight_cells=len(cells),anomaly_sampling_km=20,anomaly_cubes_all_corners_active=len(ci),isosurfaces=iso,kmeans=metrics,best_k_km=best['k'],nurbs=sheets,diagonal_endpoints_lon_lat=ends.tolist(),tectonic_boundary_confirmed=False,tomography_unchanged=True,heldout_geometry_note='Catalog coordinates, depth >=40 km, same geographic side and 75 km data hull buffer; no gap or Slab2 compatibility filter; vertical geometric errors are not location errors or travel-time RMS',source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in source})
(OUT/'metodologia.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
for name in ['metodologia.json','kmeans_celdas.csv','kmeans_sismos.csv','kmeans_centroides.csv','metricas_kmeans.csv','bisectores_kmeans.csv']:
    (SUP/('hipotesis_h_'+name)).write_bytes((OUT/name).read_bytes())
print('Hypotheses rebuilt from final h data',flush=True)
