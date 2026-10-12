"""Continuous candidate Nazca sheet, without changing the final tomography."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator,BSpline
from scipy.optimize import lsq_linear
from scipy.ndimage import binary_fill_holes
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'codigos'))
from modelo_3d_continuo import fit_sheet as unbounded_sheet,evaluate,grd
from nazca_campana_20261010 import projection
from regional.inputs import sha256
from exportar_solidos_nurbs import solid
from agregar_mapa_superficie_norte import boundaries,lines
from shapely.geometry import box
from shapely.ops import unary_union

RUN=ROOT/'resultados/migracion_nazca_cauca_valle_eje_1d_2026-10-10'
OUT=ROOT/'resultados/nazca_nurbs_morada_2026-10-10'
DATA=ROOT/'datos/slab2_2026-10-10'
PURPLE='#7926b3'

def fit_sheet(xy,z,weight,nx,ny,smooth):
    sheet=unbounded_sheet(xy,z,weight,nx,ny,smooth)
    uv=(xy-sheet['lo'])/sheet['span']
    bu=BSpline.design_matrix(uv[:,0],sheet['knots_u'],3).toarray()
    bv=BSpline.design_matrix(uv[:,1],sheet['knots_v'],3).toarray()
    matrix=np.einsum('ni,nj->nij',bu,bv).reshape(-1,nx*ny)
    reg=[]
    for i in range(nx):
        for j in range(ny-2):
            a=np.zeros((nx,ny));a[i,j:j+3]=[1,-2,1];reg.append(a.ravel())
    for i in range(nx-2):
        for j in range(ny):
            a=np.zeros((nx,ny));a[i:i+3,j]=[1,-2,1];reg.append(a.ravel())
    aug=np.vstack([matrix*weight[:,None],smooth*np.asarray(reg)])
    target=np.r_[z*weight,np.zeros(len(reg))]
    # Positive B-spline basis + bounded coefficients guarantees bounded depth
    # everywhere, instead of clipping individual vertices and creating seams.
    fit=lsq_linear(aug,target,bounds=(10,180),tol=1e-7,max_iter=100)
    assert fit.success,fit.message
    sheet['height']=fit.x.reshape(nx,ny);sheet['control_points'][:,:,2]=sheet['height']
    return sheet

def main():
    OUT.mkdir(exist_ok=False)
    domain=json.loads((RUN/'domain.json').read_text());pr=projection(domain)
    e=pd.read_csv(RUN/'relocated_train_events.csv');splits=pd.read_csv(RUN/'events_split.csv')
    assert set(e.event_id)<=set(splits[splits.split.eq('train')].event_id)
    lon,lat,dep=grd(DATA/'sam_slab2_dep_02.23.18.grd');dep=-dep
    _,_,unc=grd(DATA/'sam_slab2_unc_02.23.18.grd')
    prior=RegularGridInterpolator((lat,lon),dep,bounds_error=False,fill_value=np.nan)
    sig=RegularGridInterpolator((lat,lon),unc,bounds_error=False,fill_value=np.nan)
    e['slab2_top_km']=prior(e[['lat','lon']]);e['residual_midplane_km']=e.depth_km-e.slab2_top_km-15
    quality=e.status.eq('relocated')&e.depth_km.ge(40)&e.gap_deg.le(180)
    # Geographic window follows the user's southern Nazca hypothesis; 8 N is
    # an explicit display cutoff, not an estimated Nazca/Caribbean boundary.
    candidate=quality&e.lat.between(.8,8,inclusive='left')&e.lon.between(-79,-71)
    compatible=candidate&e.residual_midplane_km.abs().le(35)&e.slab2_top_km.le(180)
    e['used_for_nazca_fit']=compatible;e.to_csv(OUT/'asignacion_sismos.csv',index=False)
    good=e[compatible].copy();assert len(good)>50
    xx,yy=np.meshgrid(lon[::3],lat[::3]);dd=dep[::3,::3]
    sel=np.isfinite(dd)&(dd>=0)&(dd<=165)&(xx>=-79)&(xx<=-71)&(yy>=.8)&(yy<8)
    ax,ay=pr.xy(yy[sel],xx[sel]);prior_xy=np.c_[ax,ay];prior_z=dd[sel]+15
    uncertainty=np.nan_to_num(sig(np.c_[yy[sel],xx[sel]]),nan=20)
    # Equalize clustered seismic control cells rather than counting repeated
    # hypocenters as independent measurements of an interface.
    g=good.copy();g[['ix','iy','iz']]=np.floor(g[['x_km','y_km','depth_km']]/20).astype(int)
    g=g.groupby(['ix','iy','iz']).agg(x=('x_km','median'),y=('y_km','median'),z=('depth_km','median'),n=('event_id','size')).reset_index()
    controls=g[['x','y','z']].to_numpy();g.to_csv(OUT/'controles_benioff.csv',index=False)
    xy=np.vstack([prior_xy,controls[:,:2]]);z=np.r_[prior_z,controls[:,2]]
    weights=np.r_[8/np.clip(uncertainty,5,30),np.full(len(controls),1.0)]
    sheet=fit_sheet(xy,z,weights,14,18,.8)
    # One cubic tensor spline across all the footprint; unit rational weights.
    # Robustly downweight incompatible control-cell residuals, never the prior.
    for _ in range(3):
        fit,_=evaluate(sheet,controls[:,:2]);res=np.abs(fit[:,2]-controls[:,2])
        weights[-len(controls):]=np.sqrt(np.minimum(1,15/np.maximum(res,1e-9)))
        sheet=fit_sheet(xy,z,weights,14,18,.8)
    with np.load(RUN/'vp_final.npz') as m:
        sx=np.arange(m['xg'].min(),m['xg'].max()+.1,5.);sy=np.arange(m['yg'].min(),m['yg'].max()+.1,5.)
        mx,my=np.meshgrid(sx,sy);sample=np.c_[mx.ravel(),my.ravel()];la,lo=pr.latlon(sample[:,0],sample[:,1])
        raw=prior(np.c_[la,lo]);uv=(sample-sheet['lo'])/sheet['span']
        geographic=(la>=.8)&(la<8)&(lo>=-79)&(lo<=-71)&((uv>=0)&(uv<=1)).all(axis=1)
        valid=np.isfinite(raw)&(raw>=0)&(raw<=165)&geographic
        filled=binary_fill_holes(valid.reshape(len(sy),len(sx))).ravel()&geographic
        middle,norm=evaluate(sheet,sample[filled]);finite=(middle[:,2]>=0)&(middle[:,2]<=180)
        assert finite.all(),'NURBS leaves 0-180 km volume: revise fit, do not clip surface silently'
        kept=np.full(len(sample),-1,int);kept[np.flatnonzero(filled)]=np.arange(filled.sum())
        nx=len(sx);ny=len(sy);ids=np.arange(nx*ny).reshape(ny,nx)
        a=ids[:-1,:-1].ravel();b=ids[:-1,1:].ravel();c=ids[1:,:-1].ravel();d=ids[1:,1:].ravel()
        quads=np.c_[a,b,c,d];quads=quads[filled[quads].all(axis=1)];q=kept[quads]
        faces=np.vstack([q[:,[0,1,2]],q[:,[1,3,2]]]);assert len(faces)>100
        vpi=RegularGridInterpolator((m['zg'],m['yg'],m['xg']),m['vp'],bounds_error=False,fill_value=np.nan)
        v0=RegularGridInterpolator((m['zg'],m['yg'],m['xg']),m['vp_initial'],bounds_error=False,fill_value=np.nan)
        cover=RegularGridInterpolator((m['inversion_zg'],m['inversion_yg'],m['inversion_xg']),m['active'].astype(float),method='nearest',bounds_error=False,fill_value=0)
        vp=vpi(middle[:,::-1]);dvp=100*(vp/v0(middle[:,::-1])-1);active=cover(middle[:,::-1])>.5
    distance,_=cKDTree(controls).query(middle);supported=active&(distance<=40)
    measured=faces[supported[faces].all(axis=1)];interpolated=faces[~supported[faces].all(axis=1)]
    # Check the mesh contains no isolated vertices and has one connected sheet.
    used=np.unique(faces);pairs=np.vstack([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
    graph=coo_matrix((np.ones(2*len(pairs)),(np.r_[pairs[:,0],pairs[:,1]],np.r_[pairs[:,1],pairs[:,0]])),shape=(len(middle),len(middle))).tocsr()
    ncomponents,_=connected_components(graph[used][:,used],directed=False)
    assert ncomponents==1,f'Disconnected sheet: {ncomponents} components'
    fit,_=evaluate(sheet,controls[:,:2]);res=fit[:,2]-controls[:,2];priorfit,_=evaluate(sheet,prior_xy)
    heights=np.asarray(sheet['height']);assert np.isfinite(heights).all()
    vertices,solid_faces=solid(middle,norm,faces,30)
    np.testing.assert_allclose(np.linalg.norm((middle+15*norm)-(middle-15*norm),axis=1),30,atol=1e-8)
    nla,nlo=pr.latlon(middle[:,0],middle[:,1]);pts=np.c_[nlo,nla,middle[:,2]]
    np.savez_compressed(OUT/'nazca_superficie.npz',middle=middle,normals=norm,faces=faces,lon=nlo,lat=nla,supported=supported,distance_to_benioff_km=distance,vp_final=vp,dvp_percent=dvp,slab2_raw_top_km=raw[filled],solid_vertices=vertices,solid_faces=solid_faces)
    pd.DataFrame(dict(lon=nlo,lat=nla,depth_km=middle[:,2],support=supported,distance_to_benioff_km=distance,vp_final=vp,dvp_percent=dvp)).to_csv(OUT/'nazca_superficie.csv',index=False)
    serial={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in sheet.items()}
    serial.update(name='Nazca candidata continua',surface_kind='empirical midplane, not a measured top interface',thickness_visual_km=30)
    (OUT/'nazca_nurbs.json').write_text(json.dumps(serial,indent=2),encoding='utf-8')
    # Slab2-style export: top depth of this display scenario, positive down.
    top=middle-15*norm;tla,tlo=pr.latlon(top[:,0],top[:,1])
    pd.DataFrame(dict(lon=tlo,lat=tla,depth_km=top[:,2],interpolated=~supported)).to_csv(OUT/'nazca_top_depth.csv',index=False)
    with (OUT/'nazca_30km.obj').open('w',encoding='utf-8') as f:
        f.write('# AEQD local km; z positive down; candidate Nazca 30 km scenario\n')
        for p in vertices:f.write('v '+' '.join(f'{a:.6f}' for a in p)+'\n')
        for t in solid_faces:f.write('f '+' '.join(str(a+1) for a in t)+'\n')
    path,meta,features=boundaries();polys=[a[1].intersection(box(-79,.8,-71,13)) for a in features.values()]
    borders=[(unary_union(polys).boundary,'Colombia','#25303c',3),(unary_union([p.boundary for p in polys]),'Departamentos','#999999',1)]
    import plotly.graph_objects as go
    fig=go.Figure()
    for fs,name,opacity in [(measured,'Nazca morada: apoyo local Benioff y Vp',.92),(interpolated,'Nazca morada: continuidad interpolada / apoyo Slab2',.46)]:
        if not len(fs):continue
        fig.add_trace(go.Mesh3d(x=pts[:,0],y=pts[:,1],z=pts[:,2],i=fs[:,0],j=fs[:,1],k=fs[:,2],color=PURPLE,opacity=opacity,name=name,showlegend=True,customdata=np.c_[distance,vp,dvp],hovertemplate='Lon %{x:.3f}<br>Lat %{y:.3f}<br>Prof. %{z:.1f} km<br>Dist. Benioff %{customdata[0]:.1f} km<br>Vp volumen %{customdata[1]:.3f} km/s<br>dVp %{customdata[2]:.1f}%<extra>'+name+'</extra>'))
    bottom=middle+15*norm;bla,blo=pr.latlon(bottom[:,0],bottom[:,1])
    fig.add_trace(go.Mesh3d(x=blo,y=bla,z=bottom[:,2],i=faces[:,0],j=faces[:,1],k=faces[:,2],color=PURPLE,opacity=.18,visible='legendonly',name='Cara inferior: escenario de espesor 30 km',showlegend=True))
    fig.add_trace(go.Scatter3d(x=good.lon,y=good.lat,z=good.depth_km,mode='markers',name='SGC relocalizados compatibles usados',text=good.event_id,marker=dict(size=2,color='#202027',opacity=.6)))
    others=e[~compatible&quality];fig.add_trace(go.Scatter3d(x=others.lon,y=others.lat,z=others.depth_km,mode='markers',name='Otros SGC relocalizados: contexto',visible='legendonly',marker=dict(size=1.5,color='gray',opacity=.2)))
    for geo,name,color,width in borders:
        x,y,z=[],[],[]
        for p in lines(geo):x.extend(list(p[:,0])+[None]);y.extend(list(p[:,1])+[None]);z.extend([0]*len(p)+[None])
        fig.add_trace(go.Scatter3d(x=x,y=y,z=z,mode='lines',name=name,line=dict(color=color,width=width)))
    note='Nazca candidata: una NURBS cúbica continua morada. Morado opaco: a ≤40 km de controles Benioff y en celda Vp activa. Morado translúcido: continuidad interpolada apoyada por Slab2. No modifica Vp ni el RMS. La ventana termina en 8° N por hipótesis de trabajo, no porque se haya resuelto el límite Caribe–Nazca. El escenario de espesor 30 km no es una medición.'
    fig.update_layout(title='Nazca: superficie NURBS continua morada con apoyo Slab2 y SGC',height=900,scene=dict(xaxis=dict(title='Longitud WGS84',range=[-79,-71]),yaxis=dict(title='Latitud WGS84',range=[.8,13]),zaxis=dict(title='Profundidad km',range=[180,0]),aspectmode='manual',aspectratio=dict(x=.655,y=1,z=.133)),legend=dict(font=dict(size=11)))
    (OUT/'nazca_nurbs_3d.html').write_text(fig.to_html(include_plotlyjs=True).replace('<body>','<body><p>'+note+'</p>'),encoding='utf-8')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection,Line3DCollection
    fig=plt.figure(figsize=(15,9));ax=fig.add_subplot(projection='3d');ax.set_position([.04,.12,.9,.8])
    for fs,opacity in [(interpolated,.35),(measured,.9)]:
        if len(fs):ax.add_collection3d(Poly3DCollection(pts[fs],facecolors=PURPLE,edgecolors='none',alpha=opacity))
    ax.scatter(good.lon,good.lat,good.depth_km,s=3,c='#222222',alpha=.5)
    for geo,name,color,width in borders:ax.add_collection3d(Line3DCollection([np.c_[p[:,0],p[:,1],np.zeros(len(p))] for p in lines(geo)],colors=color,linewidths=width*.3))
    ax.set(xlabel='Longitud WGS84',ylabel='Latitud WGS84',zlabel='Profundidad km',xlim=(-79,-71),ylim=(.8,13),zlim=(180,0),zticks=[0,60,120,180]);ax.set_box_aspect([.655,1,.399]);ax.view_init(elev=24,azim=-65)
    fig.suptitle('Nazca: NURBS continua morada — Slab2 SAM y Benioff SGC',fontsize=16)
    fig.text(.045,.025,'Morado opaco: apoyo local de sismos y Vp activo. Morado translúcido: continuidad interpolada.\nPuntos: SGC relocalizados compatibles. Geometría candidata; corte de ventana en 8° N. Exageración vertical 3×.',fontsize=11)
    fig.savefig(OUT/'nazca_nurbs_morada.png',dpi=170);plt.close(fig)
    inputs=[Path(__file__),ROOT/'codigos/modelo_3d_continuo.py',ROOT/'codigos/exportar_solidos_nurbs.py',ROOT/'codigos/nazca_campana_20261010.py',RUN/'vp_final.npz',RUN/'relocated_train_events.csv',RUN/'events_split.csv',RUN/'domain.json',DATA/'sam_slab2_dep_02.23.18.grd',DATA/'sam_slab2_unc_02.23.18.grd',DATA/'descarga.json',path,meta]
    report=dict(description=note,events_used=len(good),control_cells=len(controls),candidate_events=int(candidate.sum()),excluded_incompatible_events=int((candidate&~compatible).sum()),slab2_anchor_points=len(prior_xy),mesh_vertices=len(middle),mesh_faces=len(faces),supported_faces=len(measured),interpolated_faces=len(interpolated),connected_components=ncomponents,unused_vertices=int(len(middle)-len(used)),internal_footprint_holes_filled_vertices=int((filled&~valid).sum()),fit_benioff_rms_km=float(np.sqrt(np.mean(res**2))),fit_benioff_p90_abs_km=float(np.quantile(abs(res),.9)),fit_slab2_midplane_rms_km=float(np.sqrt(np.mean((priorfit[:,2]-prior_z)**2))),depth_range_km=[float(middle[:,2].min()),float(middle[:,2].max())],unit_weights_NURBS=True,degree=[3,3],control_net_shape=list(sheet['control_points'].shape),thickness_visual_km=30,tomography_unchanged=True,projection=domain.get('projection_reference_bounds'),references=[dict(title='Hayes et al. 2018 Slab2',doi='10.1126/science.aat4723',url='https://www.usgs.gov/publications/slab2-a-comprehensive-subduction-zone-geometry-model')],limitations=['Continuity is imposed in this geometric candidate, not newly measured resolution.','Benioff points are an empirical slab midplane proxy, not measured slab interfaces.','Slab2 anchors have a nominal 15 km vertical offset; normal thickness is a 30 km visualization scenario.','All southern events are not assumed Nazca: discordant depths are excluded.','Slab2 is a geometry prior, may share seismic inputs and is not independent confirmation.','The 8 N geographic cutoff is not a tectonic boundary.','Vp values in unsupported portions are only sampled from the existing volume; no new velocity anomaly inferred.'],source_sha256={str(p):sha256(p) for p in inputs})
    (OUT/'metodologia.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (OUT/'README.md').write_text(f"# Nazca NURBS morada\n\n{note}\n\nUna sola superficie conectada: {len(faces)} triángulos, {len(measured)} con apoyo local y {len(interpolated)} interpolados. Se usan {len(good)} sismos SGC de entrenamiento y {len(controls)} celdas de control. RMS geométrico frente a controles Benioff: {report['fit_benioff_rms_km']:.2f} km; no es RMS de tiempos de viaje.\n\nArchivos: nazca_nurbs_3d.html (visor), nazca_nurbs_morada.png (imagen), nazca_nurbs.json (NURBS editable), nazca_superficie.npz/csv (malla y apoyo), nazca_top_depth.csv (cara superior del escenario, profundidad positiva), nazca_30km.obj (sólido en coordenadas locales km).\n\nReferencia: [Hayes et al. 2018 Slab2](https://www.usgs.gov/publications/slab2-a-comprehensive-subduction-zone-geometry-model), DOI 10.1126/science.aat4723.\n\nLa tomografía final permanece intacta. Para usar esta geometría como modelo inicial, debe realizarse otra inversión y comparar las mismas observaciones de reserva.\n",encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2),flush=True)
if __name__=='__main__':main()
