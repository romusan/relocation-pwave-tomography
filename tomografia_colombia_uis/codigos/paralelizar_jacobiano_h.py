"""Station-level concurrency of the identical discrete Jacobian, with private caches."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,sys,time,hashlib,builtins,os
import numpy as np
from scipy import sparse
import campana_perfil_suavidad_20261011h as h
from factorizado_paralelo_20261011g import BaseCache
original=h.jacobian

def parallel(cache,picks,nodes,B):
    if B.shape[1]>32:return original(cache,picks,nodes,B)
    stations=picks[['station','st_x_km','st_y_km','st_elev_km']].drop_duplicates().set_index('station',drop=False)
    groups=list(picks.groupby('station',sort=False).indices.items())
    def worker(group):
        sta,ids=group
        # No shared mutable mesh/projection/LRU dictionaries across workers.
        local=BaseCache(cache.vp,cache.grid,stations.loc[[sta]],h.CFG['cache_directory'],cache.config)
        try:
            mesh,P=local.station_mesh(sta);Q=(P@B).tocsr();rows=[];order=[]
            with h.f.Graph((P@(1/cache.vp).ravel()).reshape(mesh.shape),mesh,local.sources[sta],cache.order) as graph:
                for part in np.array_split(ids,max(1,int(np.ceil(len(ids)/64)))):
                    xyz=h.GRID.xyz(np.array([nodes[e] for e in picks.iloc[part].event_id]))
                    ii,ww=h.f.interpolation_weights(mesh,xyz)
                    a=graph.adjoint_projected(ii,ww*np.linalg.norm(xyz-local.sources[sta],axis=1)[:,None],Q)
                    rows.append(sparse.csr_matrix(a));order.extend(part)
            return rows,order
        finally:local.close()
    rows=[];order=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for count,(rr,oo) in enumerate(pool.map(worker,groups),1):
            rows.extend(rr);order.extend(oo)
            if count%20==0:print(f'H parallel station Jacobian {count}/{len(groups)} / {B.shape[1]} parameters',flush=True)
    return sparse.vstack(rows,format='csr')[np.argsort(order)]

def verify():
    p=h.population();names=p.station.value_counts().index[:2]
    p=p[p.station.isin(names)].groupby('station',sort=False).head(24).reset_index(drop=True)
    stations=p[['station','st_x_km','st_y_km','st_elev_km']].drop_duplicates()
    cache=BaseCache(h.initial(),h.GRID,stations,h.CFG['cache_directory'],h.CFG);nodes=h.nodes_for(p)
    mask=np.load(h.GOUT/'real_elevation/model_round_1.npz')['active'];reports=[]
    for kind in ['1d','3d']:
        B,_,_=h.basis(kind,mask)
        t=time.perf_counter();a=original(cache,p,nodes,B);old=time.perf_counter()-t
        t=time.perf_counter();b=parallel(cache,p,nodes,B);new=time.perf_counter()-t
        difference=(a!=b).nnz;assert difference==0,(kind,difference)
        reports.append(dict(basis=kind,rows=len(p),parameters=B.shape[1],bitwise_identical=True,serial_seconds=old,station_parallel_seconds=new))
    cache.close()
    report=dict(cases=reports,workers=2,concurrency_scope='depth basis only; 3D keeps the original implementation',core_sha256=hashlib.sha256(Path(h.__file__).read_bytes()).hexdigest(),helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),numerical_change=False,private_cache_per_station=True)
    h.save(h.OUT/'station_parallel_verification.json',report);print(json.dumps(report),flush=True)

def all():
    r=json.loads((h.OUT/'station_parallel_verification.json').read_text())
    assert r['helper_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert r['core_sha256']==hashlib.sha256(Path(h.__file__).read_bytes()).hexdigest()
    assert builtins.all(c['bitwise_identical'] for c in r['cases'])
    h.save(h.OUT/'execution_backend.json',dict(pid=os.getpid(),core_sha256=r['core_sha256'],helper_sha256=r['helper_sha256'],jacobian='two private station workers for depth basis, original for 3D',bitwise_verification=r['cases'],protocol_change=False))
    h.jacobian=parallel;h.all()

if __name__=='__main__':globals()[sys.argv[1]]()
