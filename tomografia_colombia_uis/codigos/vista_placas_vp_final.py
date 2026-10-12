"""Final-relocation sheet reconstruction, colored by final Vp with coverage mask."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator
from modelo_3d_continuo import fit_sheet,evaluate
from regional.geometry import Projection
from regional.inputs import sha256
from agregar_mapa_superficie_norte import boundaries,lines,ROOT
from shapely.geometry import box
from shapely.ops import unary_union

def main():
    run=ROOT/'resultados/tomografia_1d_mas_placas_continuas_2026-10-10'
    prior=ROOT/'resultados/modelo_1d_placas_continuas_slab2_2026-10-10'
    out=ROOT/'resultados/placas_vp_final_2026-10-10';out.mkdir(exist_ok=True)
    pr=Projection(json.loads((run/'domain.json').read_text()))
    e=pd.read_csv(run/'relocated_train_events.csv');e=e[e.status.eq('relocated')&e.depth_km.ge(40)&e.gap_deg.le(180)].copy()
    shapes=json.loads((prior/'superficies_nurbs_continuas.json').read_text(encoding='utf-8'))
    xyz=e[['x_km','y_km','depth_km']].to_numpy()
    distances=np.array([cKDTree(np.asarray(s['middle'])).query(xyz)[0] for s in shapes]);branch=distances.argmin(axis=0)
    m=np.load(run/'vp_final.npz');bg=np.load(prior/'vp_inicial_1d_mas_placas_30km.npz')['vp_background_1d']
    axes=(m['zg'],m['yg'],m['xg']);vpi=RegularGridInterpolator(axes,m['vp'],bounds_error=False,fill_value=np.nan)
    v0i=RegularGridInterpolator(axes,bg,bounds_error=False,fill_value=np.nan)
    coverage=RegularGridInterpolator((m['inversion_zg'],m['inversion_yg'],m['inversion_xg']),m['active'].astype(float),method='nearest',bounds_error=False,fill_value=0)
    rows=[];meshes=[]
    for i,s in enumerate(shapes):
        old=np.asarray(s['middle']);faces=np.asarray(s['faces']);sel=(branch==i)&(distances[i]<=60);events=xyz[sel]
        # Weak inherited geometry stabilizes gaps. Final relocations constrain the update.
        anchors=old[::8];points=np.vstack([anchors,events]);weights=np.r_[np.full(len(anchors),.15),np.ones(len(events))]
        sh=fit_sheet(points[:,:2],points[:,2],weights,*np.asarray(s['height']).shape,.8)
        mid,norm=evaluate(sh,old[:,:2]);vp=vpi(mid[:,::-1]);dvp=100*(vp/v0i(mid[:,::-1])-1)
        supported=(coverage(mid[:,::-1])>.5)&np.isfinite(vp)&(mid[:,2]>=0)&(mid[:,2]<=180)
        valid=np.isfinite(vp)&(mid[:,2]>=0)&(mid[:,2]<=180);faces=faces[valid[faces].all(axis=1)]
        covered=faces[supported[faces].all(axis=1)];uncovered=faces[~supported[faces].all(axis=1)]
        la,lo=pr.latlon(mid[:,0],mid[:,1]);pts=np.c_[lo,la,mid[:,2]]
        name=['A: Slab2 SAM + Benioff','B: rama profunda de Benioff','C: rama somera de Benioff'][i]
        np.savez_compressed(out/f'superficie_final_{i+1:02d}.npz',middle=mid,normals=norm,faces=faces,vp_final=vp,dvp_original_1d_percent=dvp,supported=supported)
        pd.DataFrame(dict(lon=lo,lat=la,depth_km=mid[:,2],vp_final=vp,dvp_percent=dvp,active_support=supported)).to_csv(out/f'superficie_final_{i+1:02d}.csv',index=False)
        rows.append(dict(name=name,final_relocated_controls=int(sel.sum()),covered_triangles=len(covered),uncovered_triangles=len(uncovered),median_depth_change_km=float(np.median(mid[:,2]-old[:,2]))))
        meshes.append((name,pts,covered,uncovered,vp,dvp))
    path,meta,features=boundaries();region=box(-79,2,-71,13);polys=[g[1].intersection(region) for g in features.values()]
    borders=[(unary_union([g.boundary for g in polys]),'Departamentos','#999999',1),(unary_union(polys).boundary,'Colombia','#202e35',3)]
    import plotly.graph_objects as go
    fig=go.Figure()
    for name,pts,covered,uncovered,vp,dvp in meshes:
        for fs,support in [(covered,True),(uncovered,False)]:
            if not len(fs):continue
            kw=dict(intensity=vp,colorscale='Viridis',cmin=5.3,cmax=8.8,showscale=support and name.startswith('A'),colorbar=dict(title='Vp final<br>km/s')) if support else dict(color='#b7b7b7')
            fig.add_trace(go.Mesh3d(x=pts[:,0],y=pts[:,1],z=pts[:,2],i=fs[:,0],j=fs[:,1],k=fs[:,2],opacity=.88 if support else .16,name=name+(' — Vp final activo' if support else ' — geometría heredada sin cobertura'),showlegend=True,customdata=np.c_[vp,dvp],hovertemplate='Lon %{x:.3f}<br>Lat %{y:.3f}<br>Prof. %{z:.1f} km<br>Vp final %{customdata[0]:.3f} km/s<br>dVp/1D %{customdata[1]:.2f}%<extra>'+name+'</extra>',**kw))
    fig.add_trace(go.Scatter3d(x=e.lon,y=e.lat,z=e.depth_km,mode='markers',marker=dict(size=2,color='black',opacity=.35),name='SGC relocalizados finales ≥40 km'))
    for geo,name,color,width in borders:
        xx,yy,zz=[],[],[]
        for q in lines(geo):xx.extend(q[:,0].tolist()+[None]);yy.extend(q[:,1].tolist()+[None]);zz.extend([0]*len(q)+[None])
        fig.add_trace(go.Scatter3d(x=xx,y=yy,z=zz,mode='lines',line=dict(color=color,width=width),name=name))
    note='Reconstrucción posterior a la inversión: NURBS ajustadas a sismos relocalizados finales, estabilizadas por la geometría Slab2/Benioff anterior. Color: Vp del volumen final en celdas activas. Gris: geometría heredada sin cobertura activa. Nazca y Caribe son interpretaciones candidatas; A/B/C no equivalen automáticamente a placas. La superficie no es una interfaz determinada únicamente por Vp.'
    fig.update_layout(title='Nazca y Caribe: superficies candidatas con Vp final',height=900,scene=dict(xaxis=dict(title='Longitud WGS84',range=[-79,-71]),yaxis=dict(title='Latitud WGS84',range=[2,13]),zaxis=dict(title='Profundidad km',range=[180,0]),aspectmode='manual',aspectratio=dict(x=.726,y=1,z=.148)),legend=dict(font=dict(size=11)))
    (out/'placas_vp_final_3d.html').write_text(fig.to_html(include_plotlyjs=True).replace('<body>','<body><p>'+note+'</p>'),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection,Line3DCollection
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    fig=plt.figure(figsize=(15,9));ax=fig.add_subplot(projection='3d');ax.set_position([.00,.10,.85,.83]);norm=Normalize(5.3,8.8);cmap=plt.get_cmap('viridis')
    for name,pts,covered,uncovered,vp,dvp in meshes:
        if len(uncovered):ax.add_collection3d(Poly3DCollection(pts[uncovered],facecolors='#ababab',edgecolors='none',alpha=.26))
        if len(covered):ax.add_collection3d(Poly3DCollection(pts[covered],facecolors=cmap(norm(vp[covered].mean(axis=1))),edgecolors='none',alpha=.92))
    ax.scatter(e.lon,e.lat,e.depth_km,s=2,c='black',alpha=.24)
    for geo,name,color,width in borders:ax.add_collection3d(Line3DCollection([np.c_[q[:,0],q[:,1],np.zeros(len(q))] for q in lines(geo)],colors=color,linewidths=width*.3))
    ax.set(xlabel='Longitud WGS84',ylabel='Latitud WGS84',zlabel='Profundidad km',xlim=(-79,-71),ylim=(2,13),zlim=(180,0),zticks=[0,60,120,180]);ax.set_box_aspect([.726,1,.444],zoom=1.04);ax.view_init(elev=24,azim=-65)
    cb=fig.add_axes([.86,.27,.018,.40]);fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cb,label='Vp final km/s')
    fig.suptitle('Nazca y Caribe: reconstrucción candidata con el modelo Vp final',fontsize=16)
    fig.text(.035,.045,'Color: Vp final con cobertura activa. Gris: geometría heredada sin cobertura. Puntos: SGC relocalizados finales.\nSuperficies ajustadas con Benioff final y apoyo Slab2; identidad tectónica como hipótesis. Exageración vertical 3× en esta imagen.',fontsize=10)
    fig.savefig(out/'placas_vp_final.png',dpi=180,bbox_inches='tight');plt.close(fig)
    sources=[Path(__file__),run/'vp_final.npz',run/'relocated_train_events.csv',prior/'superficies_nurbs_continuas.json',prior/'vp_inicial_1d_mas_placas_30km.npz',ROOT/'codigos/modelo_3d_continuo.py',path,meta]
    (out/'metodologia.json').write_text(json.dumps(dict(description=note,surfaces=rows,source_sha256={str(p):sha256(p) for p in sources}),indent=2),encoding='utf-8')
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
