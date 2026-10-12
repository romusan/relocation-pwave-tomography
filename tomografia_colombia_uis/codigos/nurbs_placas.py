"""Trimmed rational tensor surfaces and normal thickness from training hypocenters.

Geometric fragments are not tectonic identities. Unit weights are the B-spline
special case of NURBS; serialized weights can be changed in subsequent fits.
"""
from pathlib import Path
import json,sys,argparse
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline,RegularGridInterpolator
from scipy.spatial import Delaunay,cKDTree,QhullError
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'codigos'))
from regional.geometry import Projection
from regional.inputs import sha256

def basis(t,derivative=0):
    return BSpline([0,0,0,1,1,1],np.eye(3),2)(t) if not derivative else BSpline([0,0,0,1,1,1],np.eye(3),2).derivative()(t)

def evaluate(control,weights,uv):
    """NURBS surface and analytic normals; inputs/outputs use physical km."""
    u,v=np.asarray(uv).T
    bu,bv=basis(u),basis(v)
    b=np.einsum('ni,nj->nij',bu,bv)*weights
    den=b.sum(axis=(1,2));num=np.einsum('nij,ijk->nk',b,control)
    s=num/den[:,None];tangents=[]
    for a,c in [(basis(u,1),bv),(bu,basis(v,1))]:
        db=np.einsum('ni,nj->nij',a,c)*weights
        tangents.append((np.einsum('nij,ijk->nk',db,control)-s*db.sum(axis=(1,2))[:,None])/den[:,None])
    normal=np.cross(*tangents);length=np.linalg.norm(normal,axis=1)
    if np.any(length<1e-9):raise ValueError('Degenerate NURBS normal')
    normal/=length[:,None];normal[normal[:,2]<0]*=-1
    return s,normal

def fragments(events):
    d=events[events.status.eq('relocated')&events.depth_km.ge(40)&events.gap_deg.le(180)].copy()
    # 3D cells preserve distinct depth layers in the same horizontal column.
    xyz=d[['x_km','y_km','depth_km']].to_numpy(float)
    d[['ix','iy','iz']]=np.floor(xyz/20).astype(int)
    c=d.groupby(['ix','iy','iz']).agg(x=('x_km','median'),y=('y_km','median'),z=('depth_km','median'),n=('event_id','size')).reset_index()
    c[['tile_x','tile_y']]=np.floor(c[['x','y']]/150).astype(int)
    patches=[]
    for tile,g in c.groupby(['tile_x','tile_y']):
        p=g[['x','y','z']].to_numpy();pairs=cKDTree(p).query_pairs(60,output_type='ndarray')
        if len(pairs):
            graph=coo_matrix((np.ones(2*len(pairs)),(np.r_[pairs[:,0],pairs[:,1]],np.r_[pairs[:,1],pairs[:,0]])),shape=(len(p),len(p)))
            _,labels=connected_components(graph,directed=False)
        else:labels=np.arange(len(p))
        for label in np.unique(labels):
            pp=p[labels==label];counts=g.n.to_numpy()[labels==label]
            if len(pp)<12:continue
            center=pp.mean(axis=0);_,sing,rotation=np.linalg.svd(pp-center,full_matrices=False)
            if sing[1]/sing[0]<.12:continue
            if np.linalg.det(rotation)<0:rotation[2]*=-1
            local=(pp-center)@rotation.T;lo=local[:,:2].min(axis=0);span=np.ptp(local[:,:2],axis=0)
            uv=(local[:,:2]-lo)/span
            try:tri=Delaunay(uv)
            except QhullError:continue
            faces=tri.simplices;valid=np.ones(len(faces),bool)
            for a,b in [(0,1),(1,2),(2,0)]:valid&=np.linalg.norm(pp[faces[:,a]]-pp[faces[:,b]],axis=1)<=75
            if valid.sum()<4:continue
            weights=np.ones((3,3));matrix=np.einsum('ni,nj->nij',basis(uv[:,0]),basis(uv[:,1])).reshape(-1,9)
            # Smooth second differences in both parameter directions.
            reg=[]
            for axis in [0,1]:
                for j in range(3):
                    a=np.zeros((3,3));a[j,:]=[1,-2,1] if axis==0 else 0
                    if axis==1:a[:,j]=[1,-2,1]
                    reg.append(a.ravel())
            w=np.sqrt(np.minimum(counts,5));aug=np.vstack([matrix*w[:,None],.15*np.asarray(reg)])
            target=np.vstack([pp*w[:,None],np.zeros((6,3))]);ctrl=np.linalg.lstsq(aug,target,rcond=None)[0].reshape(3,3,3)
            fit,_=evaluate(ctrl,weights,uv);error=np.linalg.norm(fit-pp,axis=1)
            if np.quantile(error,.9)>20:continue
            resolution=max(25,int(np.max(span)/3)+1);t=np.linspace(0,1,resolution);u,v=np.meshgrid(t,t)
            sampleuv=np.c_[u.ravel(),v.ravel()];simplex=tri.find_simplex(sampleuv)
            keep=simplex>=0;keep[keep]&=valid[simplex[keep]]
            sampleuv=sampleuv[keep];surface,normals=evaluate(ctrl,weights,sampleuv)
            st=Delaunay(sampleuv);sf=st.simplices;middle=sampleuv[sf].mean(axis=1);ss=tri.find_simplex(middle)
            ok=ss>=0;ok[ok]&=valid[ss[ok]]
            for a,b in [(0,1),(1,2),(2,0)]:ok&=np.linalg.norm(surface[sf[:,a]]-surface[sf[:,b]],axis=1)<=7
            patches.append(dict(id=f'fragmento_{len(patches)+1:02d}',tile=list(map(int,tile)),center=center,rotation=rotation,uv_min=lo,uv_span=span,control_points=ctrl,weights=weights,degree=[2,2],knots_u=[0,0,0,1,1,1],knots_v=[0,0,0,1,1,1],sample_uv=sampleuv,surface=surface,normals=normals,faces=sf[ok],fit_rms_km=float(np.sqrt(np.mean(error**2))),fit_p90_km=float(np.quantile(error,.9)),controls=pp,counts=counts,plate_identity='sin_asignar'))
    return patches,c

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    domain=json.loads((a.run/'domain.json').read_text());pr=Projection(domain)
    e=pd.read_csv(a.run/'relocated_train_events.csv');patches,controls=fragments(e)
    if not patches:raise ValueError('Insufficient support for NURBS patches')
    m=np.load(a.run/'vp_final.npz');x,y,z=m['xg'],m['yg'],m['zg'];shape=m['vp'].shape
    zz,yy,xx=np.meshgrid(z,y,x,indexing='ij');nodes=np.c_[xx.ravel(),yy.ravel(),zz.ravel()]
    cz,cy,cx=np.meshgrid(m['inversion_zg'],m['inversion_yg'],m['inversion_xg'],indexing='ij');active=m['active'].astype(bool)
    loc=np.c_[cz[active],cy[active],cx[active]];sample=RegularGridInterpolator((z,y,x),m['vp'])(loc)
    depths=m['inversion_zg'];reference=m['vp_initial'][:,0,0];profile=np.interp(depths,z,reference);counts=[]
    for j,depth in enumerate(depths):
        values=sample[loc[:,0]==depth];counts.append(len(values))
        if len(values)>=10:profile[j]=np.clip(np.median(values),np.interp(depth,z,reference)*.97,np.interp(depth,z,reference)*1.03)
    pd.DataFrame(dict(depth_km=depths,vp_km_s=profile,n_supported_cells=counts)).to_csv(a.output/'perfil_1d_desde_vp_final.csv',index=False)
    bg=np.broadcast_to(np.interp(z,depths,profile)[:,None,None],shape).copy()
    supported_tree=cKDTree(loc[:,::-1]);final_values=sample;thicknesses=[30,50,70]
    best={h:np.zeros(len(nodes)) for h in thicknesses};bands={h:np.zeros(len(nodes),bool) for h in thicknesses}
    serialized=[]
    for patch in patches:
        s=patch['surface'];normals=patch['normals'];nearest,si=cKDTree(s).query(nodes,workers=-1)
        delta=nodes-s[si];distance=np.abs(np.einsum('ij,ij->i',delta,normals[si]));tangent=np.sqrt(np.maximum(0,nearest**2-distance**2))
        nearactive,ai=supported_tree.query(s);sv=final_values[ai];baseline=np.interp(s[:,2],depths,profile)
        contrast=np.clip(sv/baseline-1,-.03,.03);contrast[nearactive>40]=0
        patch['contrast_median_percent']=float(100*np.median(contrast[nearactive<=40])) if np.any(nearactive<=40) else None
        for h in thicknesses:
            # 15 km cosine transition as in the FMM reference implementation.
            weight=np.where(distance<=h/2,1.,np.where(distance<h/2+15,.5*(1+np.cos(np.pi*(distance-h/2)/15)),0.))
            weight[tangent>4.5]=0 # do not extend a fitted plane outside its trimmed patch
            contribution=contrast[si]*weight
            replace=abs(contribution)>abs(best[h]);best[h][replace]=contribution[replace]
            bands[h]|=(weight>0)&(contrast[si]!=0)
        for h in thicknesses:
            top=s-h/2*normals;bottom=s+h/2*normals
            np.savez_compressed(a.output/f'{patch["id"]}_espesor_{h}.npz',middle=s,top=top,bottom=bottom,normals=normals,faces=patch['faces'])
            np.testing.assert_allclose(np.linalg.norm(bottom-top,axis=1),h,atol=1e-8)
        serialized.append({key:value.tolist() if isinstance(value,np.ndarray) else value for key,value in patch.items()})
    maskgeo=(nodes[:,2]>=0)&(nodes[:,2]<=180)
    lat,lon=pr.latlon(nodes[:,0],nodes[:,1]);maskgeo&=(lat>=2)&(lat<=13)&(lon>=-79)&(lon<=-71)
    scenarios={}
    for h in thicknesses:
        best[h][~maskgeo]=0
        vp=np.clip(bg*(1+best[h].reshape(shape)),4,8.8)
        np.savez_compressed(a.output/f'vp_inicial_nurbs_{h}km.npz',vp=vp,vp_background_1d=bg,xg=x,yg=y,zg=z,domain_json=json.dumps(domain),thickness_km=h,taper_km=15,anomaly_supported=bands[h].reshape(shape)&maskgeo.reshape(shape))
        scenarios[str(h)]=dict(changed_nodes=int(np.count_nonzero(best[h])),vp_min=float(vp.min()),vp_max=float(vp.max()))
    controls.to_csv(a.output/'controles_geometricos_3d.csv',index=False)
    (a.output/'superficies_nurbs.json').write_text(json.dumps(serialized,indent=2),encoding='utf-8')
    sources={str(f.resolve()):sha256(f) for f in [Path(__file__),a.run/'vp_final.npz',a.run/'relocated_train_events.csv',a.run/'events_split.csv',ROOT.parent/'repositorio_q1_tomografia_fmm/v2/build_initial_model.py']}
    report=dict(patches=len(patches),training_only=True,thickness_scenarios_km=thicknesses,default_thickness_km=30,baseline='1D medians of supported final Vp cells, minimum 10 cells per depth; clipped to +/-3 percent of original 1D; sparse depths retain original',plate_contrast='signed final Vp minus derived 1D, nearest supported cell within 40 km, +/-3 percent; no imposed positive slab contrast',overlaps='maximum absolute supported contribution, not sum',geometry='quadratic rational tensor surfaces with unit weights; local PCA and 3D cells permit more than one depth per XY; trimmed patches on 150 km tiles',normal_thickness='upper and lower surfaces separated by h along analytic NURBS normals',grid_distance_approximation='3 km surface sampling and nearest normal, tangent tolerance 4.5 km; voxel discretization 10 km; not exact signed distance to a CAD solid',limitations=['Empirical starting model derived from the same training data; not independent evidence for plate identity','Benioff seismicity does not locate both physical plate interfaces or constrain true thickness','Tiles can introduce patch seams; geometry leaves unsupported gaps and sparse regions open','No South American slab invented from shallow crustal seismicity; host plate represented by background 1D','Q has not been inferred from Vp'],scenarios=scenarios,input_sha256=sources)
    (a.output/'metodologia.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    # Editable plate aliases remain unassigned until geological comparison.
    (a.output/'identidad_placas.json').write_text(json.dumps({p['id']:p['plate_identity'] for p in patches},indent=2))
    import plotly.graph_objects as go
    fig=go.Figure()
    for p in patches:
        for offset,name,opacity in [(0,'superficie media',.65),(-15,'cara superior 30 km',.16),(15,'cara inferior 30 km',.16)]:
            s=p['surface']+offset*p['normals'];la,lo=pr.latlon(s[:,0],s[:,1]);f=p['faces']
            fig.add_trace(go.Mesh3d(x=lo,y=la,z=s[:,2],i=f[:,0],j=f[:,1],k=f[:,2],opacity=opacity,name=p['id']+' '+name,legendgroup=p['id'],showlegend=offset==0))
    fig.add_trace(go.Scatter3d(x=e.lon,y=e.lat,z=e.depth_km,mode='markers',marker=dict(size=2,color='gray',opacity=.35),name='SGC relocalizados entrenamiento'))
    from agregar_mapa_superficie_norte import boundaries,lines
    from shapely.geometry import box
    from shapely.ops import unary_union
    _,_,features=boundaries()
    region=box(domain['lon_min'],domain['lat_min'],domain['lon_max'],domain['lat_max'])
    polys=[v[1].intersection(region) for v in features.values()]
    deps=unary_union([p.boundary for p in polys]);border=unary_union(polys).boundary
    for geo,name,color in [(deps,'Fronteras departamentales','#999999'),(border,'Fronteras de Colombia','black')]:
        lx,ly,lz=[],[],[]
        for pts in lines(geo):lx.extend(pts[:,0].tolist()+[None]);ly.extend(pts[:,1].tolist()+[None]);lz.extend([0.]*len(pts)+[None])
        fig.add_trace(go.Scatter3d(x=lx,y=ly,z=lz,mode='lines',line=dict(color=color,width=2),name=name))
    fig.update_layout(title='Fragmentos NURBS provisionales con espesor de 30 km',scene=dict(xaxis_title='Longitud WGS84',yaxis_title='Latitud WGS84',zaxis=dict(title='Profundidad km',range=[180,0]),aspectmode='manual',aspectratio=dict(x=.726,y=1,z=.148)))
    fig.write_html(a.output/'placas_nurbs_3d.html',include_plotlyjs=True)
    print(json.dumps(report,ensure_ascii=True),flush=True)

if __name__=='__main__':main()
