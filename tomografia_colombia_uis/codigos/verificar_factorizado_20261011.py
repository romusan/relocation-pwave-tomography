"""Analytical, refinement, directional finite-difference and homogeneity gates."""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from regional.geometry import Grid
from regional.factored import Graph,FieldCache,interpolation_weights
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'resultados/campana_factorizada_cotas_20261011g';OUT.mkdir(exist_ok=True)
def main():
    begin=time.perf_counter();report={};rng=np.random.default_rng(20261011)
    source=np.array([50.,50.,0.]);queries=rng.uniform([5,5,0],[95,95,80],(300,3));results=[]
    for h in [10.,5.,2.5]:
        grid=Grid(np.arange(0,100+h/2,h),np.arange(0,100+h/2,h),np.arange(-10,90+h/2,h))
        z,y,x=np.meshgrid(grid.z,grid.y,grid.x,indexing='ij');r=np.sqrt((x-source[0])**2+(y-source[1])**2+(z-source[2])**2)
        # Manufactured T=r*(s0+a*(x-xs)): |gradient T| is known analytically.
        u=.17+.00015*(x-source[0]);dx=np.divide(x-source[0],r,out=np.zeros_like(r),where=r>0);dy=np.divide(y-source[1],r,out=np.zeros_like(r),where=r>0);dz=np.divide(z-source[2],r,out=np.zeros_like(r),where=r>0)
        slow=np.sqrt((u*dx+r*.00015)**2+(u*dy)**2+(u*dz)**2)
        slow[r==0]=.17
        with Graph(np.full(grid.shape,1/6),grid,source) as g:
            pred=RegularGridInterpolator((grid.z,grid.y,grid.x),g.u)(queries[:,::-1])*np.linalg.norm(queries-source,axis=1)
            hom=float(np.max(abs(pred-np.linalg.norm(queries-source,axis=1)/6)))
        with Graph(slow,grid,source) as g:
            pred=RegularGridInterpolator((grid.z,grid.y,grid.x),g.u)(queries[:,::-1])*np.linalg.norm(queries-source,axis=1)
            exact=np.linalg.norm(queries-source,axis=1)*(.17+.00015*(queries[:,0]-source[0]));err=pred-exact
        results.append(dict(h_km=h,homogeneous_max_abs_s=hom,manufactured_rms_s=float(np.sqrt(np.mean(err**2))),manufactured_p95_s=float(np.quantile(abs(err),.95))))
    report['analytical']=results
    grid=Grid(np.arange(0,71,10.),np.arange(0,81,10.),np.arange(-10,71,10.));slow=.16+rng.uniform(0,.04,grid.shape)
    src=np.array([20.,30.,0.]);starts=np.array([grid.linear(np.array([[6,5,5],[4,6,6]]))]).ravel();rr=np.linalg.norm(grid.xyz(np.column_stack(np.unravel_index(starts,grid.shape)))-src,axis=1)
    with Graph(slow,grid,src) as g:
        J=g.adjoint(starts[:,None],rr[:,None],np.arange(slow.size,dtype=np.int64),slow.size);base=g.t.ravel()[starts]
    homerr=float(np.max(abs(J@slow.ravel()-base)));directions=[]
    for amplitude in [1e-5,1e-6,1e-7]:
        d=rng.normal(size=slow.shape);d/=np.linalg.norm(d)
        with Graph(slow+amplitude*d,grid,src) as plus,Graph(slow-amplitude*d,grid,src) as minus:
            fd=(plus.t.ravel()[starts]-minus.t.ravel()[starts])/(2*amplitude)
        j=J@d.ravel();directions.append(dict(epsilon=amplitude,relative_error=float(np.linalg.norm(fd-j)/max(np.linalg.norm(fd),1e-12))))
    report['adjoint']=dict(homogeneity_max_abs_s=homerr,directional_fd=directions)
    stations=pd.DataFrame([dict(station='test',st_x_km=24.3,st_y_km=32.2,st_elev_km=1.8)])
    cache=FieldCache(np.full(grid.shape,6.),grid,stations,OUT/'verification_cache',{})
    p=pd.DataFrame(dict(station=['test']*2,source_x_km=[63.3,28.2],source_y_km=[43.4,37.1],ev_prof_km=[47.6,64.3]))
    xyz=p[['source_x_km','source_y_km','ev_prof_km']].to_numpy();exact=np.linalg.norm(xyz-cache.sources['test'],axis=1)/6
    elevated_error=float(np.max(abs(cache.predict_exact(p)-exact)))
    mesh,P=cache.station_mesh('test');slow=.16+rng.uniform(0,.04,grid.shape);ii,ww=interpolation_weights(mesh,xyz);rr=np.linalg.norm(xyz-cache.sources['test'],axis=1)
    with Graph((P@slow.ravel()).reshape(mesh.shape),mesh,cache.sources['test']) as g:
        J=g.adjoint_projected(ii,ww*rr[:,None],P)
    d=rng.normal(size=slow.shape);d/=np.linalg.norm(d);eps=1e-6
    with Graph((P@(slow+eps*d).ravel()).reshape(mesh.shape),mesh,cache.sources['test']) as gp,Graph((P@(slow-eps*d).ravel()).reshape(mesh.shape),mesh,cache.sources['test']) as gm:
        fd=np.sum((gp.u.ravel()[ii]-gm.u.ravel()[ii])*ww,axis=1)*rr/(2*eps)
    pe=float(np.linalg.norm(fd-J@d.ravel())/np.linalg.norm(fd))
    report['physical_elevation']=dict(homogeneous_exact_xyz_max_abs_s=elevated_error,projected_adjoint_fd_relative_error=pe,station_elevation_km=1.8)
    report['pass']=bool(max(r['homogeneous_max_abs_s'] for r in results)<1e-9 and homerr<1e-9 and max(d['relative_error'] for d in directions)<1e-5 and results[-1]['manufactured_rms_s']<results[0]['manufactured_rms_s'] and elevated_error<1e-9 and pe<1e-5)
    report['seconds']=time.perf_counter()-begin
    (OUT/'solver_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2),flush=True)
    if not report['pass']:raise RuntimeError('Solver/adjoint gates failed; do not launch inversions')
if __name__=='__main__':main()
