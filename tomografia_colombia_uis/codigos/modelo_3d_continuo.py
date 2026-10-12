"""Original 1D plus continuous NURBS slabs constrained by Slab2 and training seismicity."""
from pathlib import Path
import argparse,json,sys
import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline,RegularGridInterpolator
from scipy.spatial import cKDTree,Delaunay,QhullError
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'codigos'))
from regional.inputs import sha256
from regional.geometry import Projection
from exportar_solidos_nurbs import solid

def knots(n,p=3):return np.r_[np.zeros(p+1),np.linspace(0,1,n-p+1)[1:-1],np.ones(p+1)]

def fit_sheet(xy,z,weight,nx=12,ny=15,smooth=.5):
    lo=xy.min(axis=0);span=np.ptp(xy,axis=0)
    if np.any(span<10):raise ValueError('Degenerate sheet extent')
    uv=(xy-lo)/span;ku,kv=knots(nx),knots(ny)
    bu=BSpline.design_matrix(uv[:,0],ku,3).toarray();bv=BSpline.design_matrix(uv[:,1],kv,3).toarray()
    matrix=np.einsum('ni,nj->nij',bu,bv).reshape(-1,nx*ny)
    penalty=[]
    for axis in [0,1]:
        for i in range(nx if axis==0 else nx-2):
            for j in range(ny-2 if axis==0 else ny):
                a=np.zeros((nx,ny))
                if axis==0:a[i,j:j+3]=[1,-2,1]
                else:a[i:i+3,j]=[1,-2,1]
                penalty.append(a.ravel())
    aug=np.vstack([matrix*weight[:,None],smooth*np.asarray(penalty)])
    target=np.r_[z*weight,np.zeros(len(penalty))]
    height=np.linalg.lstsq(aug,target,rcond=None)[0].reshape(nx,ny)
    gx=np.array([np.mean(ku[i+1:i+4]) for i in range(nx)])
    gy=np.array([np.mean(kv[i+1:i+4]) for i in range(ny)])
    xx,yy=np.meshgrid(lo[0]+gx*span[0],lo[1]+gy*span[1],indexing='ij')
    return dict(lo=lo,span=span,knots_u=ku,knots_v=kv,height=height,control_points=np.stack([xx,yy,height],axis=-1),weights=np.ones((nx,ny)),degree=[3,3])

def evaluate(sheet,xy):
    uv=(xy-sheet['lo'])/sheet['span'];nx,ny=sheet['height'].shape
    bu=BSpline(sheet['knots_u'],np.eye(nx),3);bv=BSpline(sheet['knots_v'],np.eye(ny),3)
    u,v=uv.T;h=np.einsum('ni,ij,nj->n',bu(u),sheet['height'],bv(v))
    dx=np.einsum('ni,ij,nj->n',bu.derivative()(u),sheet['height'],bv(v))/sheet['span'][0]
    dy=np.einsum('ni,ij,nj->n',bu(u),sheet['height'],bv.derivative()(v))/sheet['span'][1]
    normal=np.c_[-dx,-dy,np.ones(len(h))];normal/=np.linalg.norm(normal,axis=1)[:,None]
    return np.c_[xy,h],normal

def grd(path):
    with h5py.File(path) as d:x=(d['x'][:]+180)%360-180;y=d['y'][:];z=d['z'][:].astype(float)
    return x,y,z

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    run=ROOT/'resultados/migracion_isc_choco_2026-10-10';data=ROOT/'datos/slab2_2026-10-10'
    domain=json.loads((run/'domain.json').read_text());pr=Projection(domain);m=np.load(run/'vp_final.npz')
    e=pd.read_csv(run/'relocated_train_events.csv');train=set(pd.read_csv(run/'events_split.csv').query("split=='train'").event_id)
    assert set(e.event_id)<=train
    e=e[e.status.eq('relocated')&e.depth_km.ge(40)&e.gap_deg.le(180)].copy()
    dep=data/'sam_slab2_dep_02.23.18.grd';unc=data/'sam_slab2_unc_02.23.18.grd';lon,lat,depth=grd(dep);_,_,sigma=grd(unc);depth=-depth
    interpolator=RegularGridInterpolator((lat,lon),depth,bounds_error=False,fill_value=np.nan)
    errors=RegularGridInterpolator((lat,lon),sigma,bounds_error=False,fill_value=np.nan)
    xx,yy=np.meshgrid(lon[::2],lat[::2]);zz=depth[::2,::2];mask=np.isfinite(zz)&(xx>=-79)&(xx<=-71)&(yy>=2)&(yy<=13)&(zz<=200)
    px,py=pr.xy(yy[mask],xx[mask]);prior_xy=np.c_[px,py];prior_z=zz[mask]
    prior_sigma=np.nan_to_num(errors(np.c_[yy[mask],xx[mask]]),nan=20)
    e['slab2_top_km']=interpolator(e[['lat','lon']]);e['slab2_midplane_residual_km']=e.depth_km-e.slab2_top_km-15
    compatible=e.slab2_midplane_residual_km.abs().le(30)&e.slab2_top_km.le(200)
    e['branch']=np.where(compatible,'Slab2_SAM_Benioff',np.where(e.slab2_midplane_residual_km>30,'Benioff_mas_profunda','Benioff_mas_somera'))
    e.loc[e.slab2_top_km.isna(),'branch']='Benioff_fuera_Slab2'
    good=e[compatible]
    sheet=fit_sheet(np.vstack([prior_xy,good[['x_km','y_km']].to_numpy()]),np.r_[prior_z,good.depth_km.to_numpy()-15],np.r_[8/np.clip(prior_sigma,5,30),np.full(len(good),.65)],14,18,.8)
    sheet['name']='Slab2 SAM con ajuste Benioff';sheet['is_interface']=True
    # Sample the shared global spline, never independent 150 km tiles.
    sx=np.arange(m['xg'].min(),m['xg'].max()+1,5.);sy=np.arange(m['yg'].min(),m['yg'].max()+1,5.)
    x2,y2=np.meshgrid(sx,sy);xy=np.c_[x2.ravel(),y2.ravel()];la,lo=pr.latlon(xy[:,0],xy[:,1]);original=interpolator(np.c_[la,lo])
    uv=(xy-sheet['lo'])/sheet['span'];valid=np.isfinite(original)&(original<=180)&(la>=2)&(la<=13)&(lo>=-79)&(lo<=-71)&np.all((uv>=0)&(uv<=1),axis=1)
    top,norm=evaluate(sheet,xy[valid]);middle=top+15*norm
    full_to_kept=np.full(len(xy),-1,int);full_to_kept[np.flatnonzero(valid)]=np.arange(valid.sum());nx=len(sx);ny=len(sy)
    faces=[]
    for j in range(ny-1):
        for i in range(nx-1):
            ids=[j*nx+i,j*nx+i+1,(j+1)*nx+i,(j+1)*nx+i+1]
            if valid[ids].all():faces.extend([[full_to_kept[ids[0]],full_to_kept[ids[1]],full_to_kept[ids[2]]],[full_to_kept[ids[1]],full_to_kept[ids[3]],full_to_kept[ids[2]]]])
    sheet.update(middle=middle,normals=norm,faces=np.asarray(faces),support_xy=prior_xy,slab2_raw_top_km=original[valid]);sheets=[sheet]
    # Discordant depth branches stay separate, so overlapping layers are not
    # reduced to one median per horizontal column.
    rejected=[]
    for branch,g in e[~compatible].groupby('branch'):
        d=g.copy();d[['ix','iy','iz']]=np.floor(d[['x_km','y_km','depth_km']]/20).astype(int)
        c=d.groupby(['ix','iy','iz']).agg(x=('x_km','median'),y=('y_km','median'),z=('depth_km','median'),n=('event_id','size')).reset_index()
        p=c[['x','y','z']].to_numpy();pairs=cKDTree(p).query_pairs(100,output_type='ndarray')
        graph=coo_matrix((np.ones(len(pairs)*2),(np.r_[pairs[:,0],pairs[:,1]],np.r_[pairs[:,1],pairs[:,0]])),shape=(len(p),len(p)))
        _,labels=connected_components(graph,directed=False)
        for label in np.unique(labels):
            q=p[labels==label];count=c.n.to_numpy()[labels==label]
            if len(q)<12 or np.any(np.ptp(q[:,:2],axis=0)<40):rejected.append(dict(branch=branch,controls=len(q)));continue
            nctrl=min(7,max(4,int(np.sqrt(len(q)/2))))
            sh=fit_sheet(q[:,:2],q[:,2],np.sqrt(np.minimum(count,5)),nctrl,nctrl,.7)
            tx=np.arange(q[:,0].min(),q[:,0].max()+.1,5);ty=np.arange(q[:,1].min(),q[:,1].max()+.1,5);x,y=np.meshgrid(tx,ty);sxy=np.c_[x.ravel(),y.ravel()]
            try:tri=Delaunay(q[:,:2])
            except QhullError:continue
            tf=tri.simplices;usable=np.ones(len(tf),bool)
            for aa,bb in [(0,1),(1,2),(2,0)]:usable&=np.linalg.norm(q[tf[:,aa],:2]-q[tf[:,bb],:2],axis=1)<=150
            ix=tri.find_simplex(sxy);keep=ix>=0;keep[keep]&=usable[ix[keep]]
            near,_=cKDTree(q[:,:2]).query(sxy);keep&=near<=100;sxy=sxy[keep]
            if len(sxy)<10:continue
            mid,n=evaluate(sh,sxy);tri2=Delaunay(sxy);f=tri2.simplices;centers=sxy[f].mean(axis=1);ix=tri.find_simplex(centers);accept=ix>=0;accept[accept]&=usable[ix[accept]]
            for aa,bb in [(0,1),(1,2),(2,0)]:accept&=np.linalg.norm(sxy[f[:,aa]]-sxy[f[:,bb]],axis=1)<=8
            sh.update(name=branch+'_'+str(label),is_interface=False,middle=mid,normals=n,faces=f[accept],support_xy=q[:,:2],seismic_controls=q)
            sheets.append(sh)
    e.to_csv(a.output/'asignacion_benioff_entrenamiento.csv',index=False)
    bg=m['vp_initial'].copy()
    assert np.array_equal(bg,np.broadcast_to(bg[:,0,0][:,None,None],bg.shape))
    z,y,x=np.meshgrid(m['zg'],m['yg'],m['xg'],indexing='ij');nodes=np.c_[x.ravel(),y.ravel(),z.ravel()]
    al,ao=pr.latlon(nodes[:,0],nodes[:,1]);inside=(al>=2)&(al<=13)&(ao>=-79)&(ao<=-71)
    score=np.zeros(len(nodes));change=np.zeros(len(nodes));owner=np.zeros(len(nodes),np.int16)
    cz,cy,cx=np.meshgrid(m['inversion_zg'],m['inversion_yg'],m['inversion_xg'],indexing='ij');active=m['active'].astype(bool);points=np.c_[cx[active],cy[active],cz[active]]
    values=RegularGridInterpolator((m['zg'],m['yg'],m['xg']),m['vp'])(points[:,::-1]);tree=cKDTree(points)
    metadata=[]
    for number,sh in enumerate(sheets,1):
        mid=sh['middle'];normal=sh['normals'];dist,indices=cKDTree(mid).query(nodes,workers=-1);delta=nodes-mid[indices];perp=np.abs(np.einsum('ij,ij->i',delta,normal[indices]));tangent=np.sqrt(np.maximum(0,dist**2-perp**2))
        weight=np.where(perp<=15,1.,np.where(perp<30,.5*(1+np.cos(np.pi*(perp-15)/15)),0.));weight[(tangent>5)|(dist>31)|~inside]=0
        distance,idx=tree.query(mid,k=8);vw=1/(distance+5);vw[distance>60]=0;total=vw.sum(axis=1);vel=(vw*values[idx]).sum(axis=1)/np.maximum(total,1e-10)
        vref=np.interp(mid[:,2],m['zg'],bg[:,0,0]);contrast=np.full(len(mid),.03);covered=total>0;contrast[covered]=np.clip(vel[covered]/vref[covered]-1,-.03,.03)
        candidate=weight/(perp+1);take=candidate>score;score[take]=candidate[take];change[take]=contrast[indices[take]]*weight[take];owner[take]=number
        nearest,_=cKDTree(sh['support_xy']).query(mid[:,:2]);support=np.where(nearest<=30,1,np.where(nearest<=100,2,3)) if not sh['is_interface'] else np.where(cKDTree(good[['x_km','y_km']]).query(mid[:,:2])[0]<=40,1,3)
        sh.update(support_class=support,contrast_percent=100*contrast,velocity_from_training=covered)
        np.testing.assert_allclose(np.linalg.norm((mid+15*normal)-(mid-15*normal),axis=1),30,atol=1e-8)
        vertices,solid_faces=solid(mid,normal,sh['faces'],30)
        np.savez_compressed(a.output/f'superficie_{number:02d}.npz',middle=mid,top=mid-15*normal,bottom=mid+15*normal,normals=normal,faces=sh['faces'],support_class=support,contrast_percent=100*contrast,solid_vertices=vertices,solid_faces=solid_faces)
        metadata.append(dict(name=sh['name'],sample_points=len(mid),faces=len(sh['faces']),connected_surface='single global spline per branch',source='Slab2 SAM plus Benioff' if sh['is_interface'] else 'training Benioff interpolation',support_classes={str(k):int(np.sum(support==k)) for k in [1,2,3]},velocity_covered_points=int(covered.sum())))
    vp=np.clip(bg*(1+change.reshape(bg.shape)),4,8.8)
    assert np.array_equal(vp.ravel()[change==0],bg.ravel()[change==0])
    np.savez_compressed(a.output/'vp_inicial_1d_mas_placas_30km.npz',vp=vp,vp_background_1d=bg,xg=m['xg'],yg=m['yg'],zg=m['zg'],domain_json=json.dumps(domain),thickness_km=30,taper_km=15,slab_owner=owner.reshape(bg.shape),anomaly_fraction=change.reshape(bg.shape))
    serial=[{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in sh.items()} for sh in sheets]
    (a.output/'superficies_nurbs_continuas.json').write_text(json.dumps(serial),encoding='utf-8')
    inputs=[Path(__file__),ROOT/'codigos/exportar_solidos_nurbs.py',ROOT/'datos/perfil_vp_vmm_transferencia.csv',run/'vp_final.npz',run/'events_split.csv',run/'relocated_train_events.csv']+list(data.glob('*.grd'))+[data/'descarga.json']
    report=dict(original_1d_background_exact=True,thickness_km=30,taper_km=15,training_only=True,sheets=metadata,rejected_sparse_branches=rejected,changed_nodes=int(np.count_nonzero(change)),velocity_contrast='signed training Vp final relative to original 1D, +/-3 percent; outside supported cells +3 percent hypothesis from FMM precedent, not Slab2 velocity measurement',slab2_depth='negative depths converted to positive down; geometry is top interface, normal offset defines middle and bottom',slab2_caribbean_colombia_coverage=0,overlap='closest normal midplane with finite lateral support; contrasts not summed',geometry_limitations=['Continuity is imposed in the initial hypothesis, not established by new observations','Slab2 and SGC may share seismic inputs; prior is not independent confirmation','Separate Benioff branches are geometric hypotheses, not automatic Nazca/Caribe identification','Slab2 top corrected using seismicity as an approximate middle of a 30 km band; this is not a measured interface','Slab2 footprint respected; extra branches interpolate up to 150 km edges and 100 km distance, labeled as interpolation','5 km surface sampling with normal-distance approximation on a 10 km volume','Host South American lithosphere represented by original 1D background; Q not inferred'],input_sha256={str(f.resolve()):sha256(f) for f in inputs})
    (a.output/'metodologia.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True),flush=True)
if __name__=='__main__':main()
