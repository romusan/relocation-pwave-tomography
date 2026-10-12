"""Purple Nazca/orange Caribbean candidate NURBS with an editable diagonal."""
from pathlib import Path
import json,sys
import numpy as np,pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from scipy.ndimage import binary_fill_holes
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from shapely.geometry import MultiPoint,Point,box
from shapely import contains_xy
from shapely.ops import unary_union
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'codigos'))
from nazca_nurbs_morada_20261010 import fit_sheet
from modelo_3d_continuo import evaluate,grd
from nazca_campana_20261010 import projection
from exportar_solidos_nurbs import solid
from regional.inputs import sha256
from agregar_mapa_superficie_norte import boundaries,lines
RUN=ROOT/'resultados/migracion_nazca_cauca_valle_eje_1d_2026-10-10'
OUT=ROOT/'resultados/placas_nazca_caribe_diagonal_2026-10-10'
PARAM=ROOT/'datos/limite_diagonal_nazca_caribe_20261010.json'
DATA=ROOT/'datos/slab2_2026-10-10'

def main():
    OUT.mkdir(exist_ok=False)
    config=json.loads(PARAM.read_text(encoding='utf-8'));start,end=np.asarray(config['endpoints_lon_lat'])
    def boundary(lon):return start[1]+(np.asarray(lon)-start[0])*(end[1]-start[1])/(end[0]-start[0])
    domain=json.loads((RUN/'domain.json').read_text());pr=projection(domain)
    e=pd.read_csv(RUN/'relocated_train_events.csv');s=pd.read_csv(RUN/'events_split.csv')
    assert set(e.event_id)<=set(s[s.split.eq('train')].event_id)
    good=e.status.eq('relocated')&e.depth_km.ge(40)&e.gap_deg.le(180)
    north=e.lat>=boundary(e.lon)
    lon,lat,z=grd(DATA/'sam_slab2_dep_02.23.18.grd');z=-z
    _,_,unc=grd(DATA/'sam_slab2_unc_02.23.18.grd')
    prior=RegularGridInterpolator((lat,lon),z,bounds_error=False,fill_value=np.nan)
    err=RegularGridInterpolator((lat,lon),unc,bounds_error=False,fill_value=np.nan)
    e['slab2_top_km']=prior(e[['lat','lon']]);compatible=(e.depth_km-e.slab2_top_km-15).abs()<=35
    e['geographic_hypothesis']=np.where(north,'Caribe','Nazca')
    e['used_Nazca']=good&~north&compatible;e['used_Caribe']=good&north
    e.to_csv(OUT/'asignacion_sismos.csv',index=False)
    with np.load(RUN/'vp_final.npz') as m:
        sx=np.arange(m['xg'].min(),m['xg'].max()+.1,5);sy=np.arange(m['yg'].min(),m['yg'].max()+.1,5)
        xx,yy=np.meshgrid(sx,sy);xy=np.c_[xx.ravel(),yy.ravel()];la,lo=pr.latlon(xy[:,0],xy[:,1])
        inside=(la>=.8)&(la<=13)&(lo>=-79)&(lo<=-71)
        raw=prior(np.c_[la,lo]);pv=RegularGridInterpolator((m['zg'],m['yg'],m['xg']),m['vp'],bounds_error=False,fill_value=np.nan)
        p0=RegularGridInterpolator((m['zg'],m['yg'],m['xg']),m['vp_initial'],bounds_error=False,fill_value=np.nan)
        coverage=RegularGridInterpolator((m['inversion_zg'],m['inversion_yg'],m['inversion_xg']),m['active'].astype(float),method='nearest',bounds_error=False,fill_value=0)
        sheets=[];reports=[]
        for name,color,flag in [('Nazca','#7926b3','used_Nazca'),('Caribe','#ed831b','used_Caribe')]:
            ev=e[e[flag]].copy();assert len(ev)>50
            ev[['ix','iy','iz']]=np.floor(ev[['x_km','y_km','depth_km']]/20).astype(int)
            grouped=ev.groupby(['ix','iy','iz']).agg(x=('x_km','median'),y=('y_km','median'),z=('depth_km','median'),n=('event_id','size')).reset_index()
            ctrl=grouped[['x','y','z']].to_numpy();grouped.to_csv(OUT/f'{name.lower()}_controles.csv',index=False)
            if name=='Nazca':
                lx,ly=np.meshgrid(lon[::3],lat[::3]);dd=z[::3,::3]
                sel=np.isfinite(dd)&(dd>=0)&(dd<=165)&(lx>=-79)&(lx<=-71)&(ly>=.8)&(ly<=13)&(ly<boundary(lx))
                ax,ay=pr.xy(ly[sel],lx[sel]);anchors=np.c_[ax,ay];az=dd[sel]+15
                sigma=np.nan_to_num(err(np.c_[ly[sel],lx[sel]]),nan=20);aw=8/np.clip(sigma,5,30)
                usable=inside&np.isfinite(raw)&(raw>=0)&(raw<=165)&(la<boundary(lo))
                usable=binary_fill_holes(usable.reshape(len(sy),len(sx))).ravel()&inside&(la<boundary(lo))
                nx,ny=14,18;regularization=.8;anchor_note='Slab2 SAM geometry, nominal +15 km midplane offset'
            else:
                hull=MultiPoint(ctrl[:,:2]).convex_hull.buffer(75)
                usable=inside&(la>=boundary(lo))&contains_xy(hull,xy[:,0],xy[:,1])
                # Weak continuation anchors only within a 75 km lateral buffer
                # of the data hull. Their depth is explicitly extrapolated.
                outline=np.array(hull.exterior.coords)[::max(1,len(hull.exterior.coords)//100)]
                distance,index=cKDTree(ctrl[:,:2]).query(outline)
                anchors=outline;az=ctrl[index,2];aw=np.full(len(anchors),.12)
                nx,ny=10,14;regularization=2.;anchor_note='Weak nearest-control depths on 75 km lateral hull buffer; no Slab2 Caribbean anchors'
            fitxy=np.vstack([anchors,ctrl[:,:2]]);target=np.r_[az,ctrl[:,2]];weights=np.r_[aw,np.ones(len(ctrl))]
            sh=fit_sheet(fitxy,target,weights,nx,ny,regularization)
            for _ in range(3):
                fit,_=evaluate(sh,ctrl[:,:2]);res=np.abs(fit[:,2]-ctrl[:,2]);weights[-len(ctrl):]=np.sqrt(np.minimum(1,15/np.maximum(res,1e-9)))
                sh=fit_sheet(fitxy,target,weights,nx,ny,regularization)
            uv=(xy-sh['lo'])/sh['span'];usable&=((uv>=0)&(uv<=1)).all(axis=1)
            mid,norm=evaluate(sh,xy[usable]);assert ((mid[:,2]>=0)&(mid[:,2]<=180)).all()
            kept=np.full(len(xy),-1,int);kept[np.flatnonzero(usable)]=np.arange(usable.sum())
            ids=np.arange(len(xy)).reshape(len(sy),len(sx));a=ids[:-1,:-1].ravel();b=ids[:-1,1:].ravel();c=ids[1:,:-1].ravel();d=ids[1:,1:].ravel()
            q=np.c_[a,b,c,d];q=kept[q[usable[q].all(axis=1)]];faces=np.vstack([q[:,[0,1,2]],q[:,[1,3,2]]])
            assert len(faces)>100
            pairs=np.vstack([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]);used=np.unique(faces)
            graph=coo_matrix((np.ones(len(pairs)*2),(np.r_[pairs[:,0],pairs[:,1]],np.r_[pairs[:,1],pairs[:,0]])),shape=(len(mid),len(mid))).tocsr()
            components,_=connected_components(graph[used][:,used],directed=False);assert components==1
            dist,_=cKDTree(ctrl).query(mid);vp=pv(mid[:,::-1]);dvp=100*(vp/p0(mid[:,::-1])-1)
            support=(dist<=40)&(coverage(mid[:,::-1])>.5)
            local=faces[support[faces].all(axis=1)];interp=faces[~support[faces].all(axis=1)]
            ml,mo=pr.latlon(mid[:,0],mid[:,1]);pts=np.c_[mo,ml,mid[:,2]]
            verts,sf=solid(mid,norm,faces,30)
            np.testing.assert_allclose(np.linalg.norm((mid+15*norm)-(mid-15*norm),axis=1),30,atol=1e-8)
            np.savez_compressed(OUT/f'{name.lower()}_superficie.npz',middle=mid,normals=norm,faces=faces,lon=mo,lat=ml,supported=support,vp_final=vp,dvp_percent=dvp,solid_vertices=verts,solid_faces=sf)
            pd.DataFrame(dict(lon=mo,lat=ml,depth_km=mid[:,2],supported=support,vp_final=vp,dvp_percent=dvp)).to_csv(OUT/f'{name.lower()}_superficie.csv',index=False)
            serial={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in sh.items()};serial.update(name=name,color=color,thickness_scenario_km=30,geographic_clip=config,geometry_proxy='empirical Benioff midplane, not measured plate interface')
            (OUT/f'{name.lower()}_nurbs.json').write_text(json.dumps(serial,indent=2),encoding='utf-8')
            fit,_=evaluate(sh,ctrl[:,:2]);res=fit[:,2]-ctrl[:,2]
            reports.append(dict(name=name,events=len(ev),control_cells=len(ctrl),vertices=len(mid),faces=len(faces),supported_faces=len(local),interpolated_faces=len(interp),connected_components=components,fit_benioff_rms_km=float(np.sqrt(np.mean(res**2))),fit_benioff_p90_abs_km=float(np.quantile(abs(res),.9)),anchors=anchor_note))
            sheets.append(dict(name=name,color=color,pts=pts,faces=faces,local=local,interp=interp,events=ev,dist=dist,vp=vp,dvp=dvp))
    path,meta,features=boundaries();polys=[a[1].intersection(box(-79,.8,-71,13)) for a in features.values()]
    borders=[(unary_union(polys).boundary,'Colombia','#25303c',3),(unary_union([p.boundary for p in polys]),'Departamentos','#999999',1)]
    import plotly.graph_objects as go
    fig=go.Figure()
    for sh in sheets:
        p=sh['pts']
        for f,label,opacity in [(sh['local'],'apoyo local',.94),(sh['interp'],'continuidad interpolada',.48)]:
            if not len(f):continue
            fig.add_trace(go.Mesh3d(x=p[:,0],y=p[:,1],z=p[:,2],i=f[:,0],j=f[:,1],k=f[:,2],name=sh['name']+' — '+label,color=sh['color'],opacity=opacity,showlegend=True,customdata=np.c_[sh['dist'],sh['vp'],sh['dvp']],hovertemplate='Lon %{x:.3f}<br>Lat %{y:.3f}<br>Prof. %{z:.1f} km<br>Distancia Benioff %{customdata[0]:.1f} km<br>Vp volumen %{customdata[1]:.3f} km/s<extra>'+sh['name']+' candidata — '+label+'</extra>'))
        ev=sh['events'];fig.add_trace(go.Scatter3d(x=ev.lon,y=ev.lat,z=ev.depth_km,mode='markers',name='SGC usados: '+sh['name'],marker=dict(size=2,color=sh['color'],opacity=.7),visible='legendonly',text=ev.event_id))
    for geo,name,color,width in borders:
        x,y,z=[],[],[]
        for p in lines(geo):x.extend(list(p[:,0])+[None]);y.extend(list(p[:,1])+[None]);z.extend([0]*len(p)+[None])
        fig.add_trace(go.Scatter3d(x=x,y=y,z=z,mode='lines',name=name,line=dict(color=color,width=width)))
    ll=np.linspace(start[0],end[0],100);bl=boundary(ll)
    fig.add_trace(go.Scatter3d(x=ll,y=bl,z=np.zeros(len(ll)),mode='lines',name='Límite diagonal propuesto (en superficie)',line=dict(color='#202020',width=5,dash='dash')))
    fig.add_trace(go.Scatter3d(x=[-75.8814],y=[8.748],z=[0],mode='markers+text',text=['Montería'],textposition='top center',name='Referencia Montería',marker=dict(size=4,color='black')))
    note='Nazca morada y Caribe naranja: superficies NURBS candidatas. Límite diagonal propuesto por el usuario, con extremos aproximados oriente (−72,5°, 6,0° N) y norte de Montería (−76,3°, 9,2° N); se prolonga solo para recortar las superficies. La línea es una hipótesis geográfica, no un límite resuelto por la inversión. Opaco: apoyo local de Benioff y Vp activo. Translúcido: interpolación/extrapolación geométrica. No modifica la tomografía ni el RMS.'
    fig.update_layout(title='Nazca morada y Caribe naranja — NURBS y límite diagonal propuesto',height=900,scene=dict(xaxis=dict(title='Longitud WGS84',range=[-79,-71]),yaxis=dict(title='Latitud WGS84',range=[.8,13]),zaxis=dict(title='Profundidad km',range=[180,0]),aspectmode='manual',aspectratio=dict(x=.655,y=1,z=.133)),legend=dict(font=dict(size=11)))
    (OUT/'placas_nazca_caribe_3d.html').write_text(fig.to_html(include_plotlyjs=True).replace('<body>','<body><p>'+note+'</p>'),encoding='utf-8')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection,Line3DCollection
    fig=plt.figure(figsize=(15,9));ax=fig.add_subplot(projection='3d');ax.set_position([.04,.12,.9,.8])
    for sh in sheets:
        for f,opacity in [(sh['interp'],.4),(sh['local'],.94)]:
            if len(f):ax.add_collection3d(Poly3DCollection(sh['pts'][f],facecolors=sh['color'],edgecolors='none',alpha=opacity))
    for geo,name,color,width in borders:ax.add_collection3d(Line3DCollection([np.c_[p[:,0],p[:,1],np.zeros(len(p))] for p in lines(geo)],colors=color,linewidths=width*.3))
    ax.plot(ll,bl,np.zeros(len(ll)),color='black',linestyle='--',linewidth=1.8);ax.scatter([-75.8814],[8.748],[0],c='black',s=12);ax.text(-75.8814,8.748,0,' Montería',fontsize=8)
    ax.set(xlabel='Longitud WGS84',ylabel='Latitud WGS84',zlabel='Profundidad km',xlim=(-79,-71),ylim=(.8,13),zlim=(180,0),zticks=[0,60,120,180]);ax.set_box_aspect([.655,1,.399]);ax.view_init(elev=24,azim=-65)
    fig.suptitle('Nazca morada y Caribe naranja: NURBS continuas candidatas',fontsize=16)
    fig.text(.04,.025,'Colores opacos: apoyo local. Translúcidos: continuidad geométrica. Línea discontinua: hipótesis diagonal en superficie.\nSismos SGC relocalizados; Nazca con apoyo Slab2 SAM. Superficies empíricas, no interfaces medidas. Exageración vertical 3×.',fontsize=11)
    fig.savefig(OUT/'placas_nazca_caribe.png',dpi=170);plt.close(fig)
    # Plan view makes the proposed diagonal and administrative references clear.
    fig,ax=plt.subplots(figsize=(8,10),layout='constrained')
    for sh in sheets:
        p=sh['pts'];ax.tripcolor(p[:,0],p[:,1],sh['faces'],np.zeros(len(p)),cmap=__import__('matplotlib.colors',fromlist=['ListedColormap']).ListedColormap([sh['color']]),alpha=.4)
        ev=sh['events'];ax.scatter(ev.lon,ev.lat,s=4,c=sh['color'],label=sh['name']+' candidata')
    for geo,name,color,width in borders:
        for p in lines(geo):ax.plot(p[:,0],p[:,1],color=color,linewidth=width*.3)
    ax.plot(ll,bl,'k--',label='Diagonal propuesta');ax.scatter([-75.8814],[8.748],c='black',s=20);ax.annotate('Montería',(-75.8814,8.748),xytext=(5,5),textcoords='offset points')
    ax.set(xlim=(-79,-71),ylim=(.8,13),xlabel='Longitud WGS84',ylabel='Latitud WGS84',title='Huellas de las superficies y límite diagonal hipotético');ax.set_aspect('equal');ax.grid(alpha=.15);ax.legend(loc='lower right')
    fig.savefig(OUT/'limite_diagonal_mapa.png',dpi=170);plt.close(fig)
    sources=[Path(__file__),PARAM,ROOT/'codigos/nazca_nurbs_morada_20261010.py',ROOT/'codigos/modelo_3d_continuo.py',ROOT/'codigos/nazca_campana_20261010.py',ROOT/'codigos/exportar_solidos_nurbs.py',RUN/'vp_final.npz',RUN/'relocated_train_events.csv',RUN/'events_split.csv',RUN/'domain.json',DATA/'sam_slab2_dep_02.23.18.grd',DATA/'sam_slab2_unc_02.23.18.grd',DATA/'descarga.json',path,meta]
    report=dict(description=note,surfaces=reports,boundary=config,tomography_unchanged=True,slab2_caribbean_used=False,limitations=['Geographic assignment follows the proposed diagonal and does not validate plate identity.','Caribbean single sheet may average distinct depth branches; large control misfits are reported.','Interpolation uses the data convex hull plus a 75 km buffer; it is not resolved structure.','The two geographic masks exclude overlaps as a display hypothesis; real slabs may overlap.','No shared depth or physical contact is imposed at the proposed line.','30 km normal thickness is an unmeasured visualization scenario.'],source_sha256={str(p):sha256(p) for p in sources})
    (OUT/'metodologia.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (OUT/'README.md').write_text('# Nazca morada y Caribe naranja\n\n'+note+'\n\n'+json.dumps(reports,indent=2,ensure_ascii=False)+'\n\nParámetros editables: '+str(PARAM)+'\n\nLa geometría Caribe se ajusta a sismos de entrenamiento, sin Slab2 Caribe (no cubre esta región colombiana). El espesor de 30 km es un escenario. Una sola profundidad por coordenada horizontal puede promediar ramas distintas; los residuales geométricos no son RMS de tiempos de viaje. La línea no impone contacto entre placas ni excluye su posible superposición real.\n',encoding='utf-8')
    print(json.dumps(reports,indent=2),flush=True)
if __name__=='__main__':main()
