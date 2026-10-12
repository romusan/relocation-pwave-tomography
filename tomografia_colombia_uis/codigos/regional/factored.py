"""Factored FM with continuous station XYZ and matching discrete adjoint.

Continuous observations interpolate u, then multiply by analytic source distance.
This reproduces homogeneous travel times exactly, including off-grid endpoints.
The source-cell boundary approximation is audited separately by refinement.
"""
from pathlib import Path
import ctypes,os,hashlib,json,time
from collections import OrderedDict
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy import sparse
from .inversion import active_mask as _active_mask
from .geometry import Grid

LIB=None
def mapping(direct,coarse):
    # Map physical coordinates: direct z includes a negative station layer,
    # coarse z starts at zero. Index-ratio mapping shifts all depth parameters.
    z,y,x=np.meshgrid(direct.z,direct.y,direct.x,indexing='ij')
    n=coarse.nodes(np.column_stack([x.ravel(),y.ravel(),z.ravel()]))
    return coarse.linear(n)
def native():
    global LIB
    if LIB is None:
        lib=ctypes.CDLL(str(Path(__file__).resolve().parents[1]/'native'/('factored_eikonal.dll' if os.name=='nt' else 'factored_eikonal.so')))
        a=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');i=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
        lib.factored_create.argtypes=[a,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_double,a,ctypes.c_int,a,a];lib.factored_create.restype=ctypes.c_void_p
        lib.factored_destroy.argtypes=[ctypes.c_void_p]
        lib.factored_adjoint.argtypes=[ctypes.c_void_p,i,a,ctypes.c_int,ctypes.c_int,i,ctypes.c_int,a];lib.factored_adjoint.restype=ctypes.c_int
        lib.factored_projected.argtypes=[ctypes.c_void_p,i,a,ctypes.c_int,ctypes.c_int,i,i,a,ctypes.c_int,a];lib.factored_projected.restype=ctypes.c_int
        LIB=lib
    return LIB

class Graph:
    def __init__(self,slow,grid,source,order=2):
        self.grid=grid;self.source=np.asarray(source,float);self.u=np.empty(grid.shape);self.t=np.empty(grid.shape)
        src=(self.source-np.array([grid.x[0],grid.y[0],grid.z[0]]))/grid.spacing
        if np.max(abs(src-np.rint(src)))>1e-8:raise ValueError('Use station-centred mesh for off-node sources')
        src=np.rint(src)
        self.lib=native();self.handle=self.lib.factored_create(np.ascontiguousarray(slow),*grid.shape,grid.spacing,np.ascontiguousarray(src),order,self.u,self.t)
        if not self.handle:raise RuntimeError('Factored solver rejected source/grid or did not reach all nodes')
    def adjoint(self,ids,weights,map_flat,nc):
        ids=np.ascontiguousarray(ids,dtype=np.int64).reshape(len(ids),-1);weights=np.ascontiguousarray(weights,dtype=float).reshape(ids.shape)
        out=np.empty((len(ids),nc));m=np.ascontiguousarray(map_flat,dtype=np.int64)
        if self.lib.factored_adjoint(self.handle,ids,weights,*ids.shape,m,nc,out):raise RuntimeError('Factored adjoint failure')
        return out
    def adjoint_projected(self,ids,weights,Q):
        ids=np.ascontiguousarray(ids,dtype=np.int64);weights=np.ascontiguousarray(weights,float);out=np.empty((len(ids),Q.shape[1]))
        ptr=np.ascontiguousarray(Q.indptr,dtype=np.int64);ix=np.ascontiguousarray(Q.indices,dtype=np.int64);v=np.ascontiguousarray(Q.data,float)
        if self.lib.factored_projected(self.handle,ids,weights,*ids.shape,ptr,ix,v,Q.shape[1],out):raise RuntimeError('Projected adjoint failure')
        return out
    def close(self):
        if self.handle:self.lib.factored_destroy(self.handle);self.handle=None
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

class FieldCache:
    def __init__(self,vp,grid,stations,directory,config):
        self.vp=np.ascontiguousarray(vp,float);self.grid=grid;self.config=config;self.order=config.get('factored_order',2)
        if self.vp.shape!=grid.shape or not np.isfinite(vp).all() or np.min(vp)<=0:raise ValueError('Invalid Vp')
        self.sources={str(s.station):np.array([s.st_x_km,s.st_y_km,-s.st_elev_km]) for s in stations.itertuples()}
        self.station_nodes={s:tuple(grid.nodes(xyz)) for s,xyz in self.sources.items()}
        self.meshes={};self.projections=OrderedDict()
        hasher=hashlib.sha256(self.vp.tobytes())
        for axis in [grid.x,grid.y,grid.z]:hasher.update(axis.tobytes())
        hasher.update(Path(__file__).read_bytes());hasher.update((Path(__file__).resolve().parents[1]/'native/factored_eikonal.cpp').read_bytes());hasher.update(str(self.order).encode())
        self.model_key=hasher.hexdigest();self.path=Path(directory)/self.model_key[:16];self.path.mkdir(parents=True,exist_ok=True)
        self.memory=OrderedDict();self.max_fields=max(1,int(config.get('time_field_memory_limit_mb',384)*1024**2/(2*self.vp.nbytes)));self.calculated=self.loaded=0;self.seconds=0.
    def pair(self,station):
        if station in self.memory:self.memory.move_to_end(station);return self.memory[station]
        key=hashlib.sha256(str(station).encode()+self.sources[station].tobytes()).hexdigest();p=self.path/(key[:16]+'.npz');start=time.perf_counter()
        if p.exists():
            with np.load(p) as f:
                if str(f['model_key'])!=self.model_key or str(f['station_key'])!=key:raise RuntimeError('Cache identity mismatch')
                pair=f['u'],f['t']
            self.loaded+=1
        else:
            mesh,P=self.station_mesh(station)
            slow=(P@(1/self.vp).ravel()).reshape(mesh.shape)
            with Graph(slow,mesh,self.sources[station],self.order) as g:
                common=self.common_xyz()
                uu=RegularGridInterpolator((mesh.z,mesh.y,mesh.x),g.u)(common[:,::-1]).reshape(self.grid.shape)
                pair=g.u,uu*np.linalg.norm(common-self.sources[station],axis=1).reshape(self.grid.shape)
            tmp=p.with_suffix('.tmp')
            with tmp.open('wb') as fh:np.savez(fh,u=pair[0],t=pair[1],model_key=self.model_key,station_key=key)
            tmp.replace(p);self.calculated+=1
            if self.calculated%10==0:print(f'Factored FM {self.model_key[:8]}: {self.calculated} fields',flush=True)
        self.seconds+=time.perf_counter()-start;self.memory[station]=pair
        while len(self.memory)>self.max_fields:self.memory.popitem(last=False)
        return pair
    def get(self,station):return self.pair(station)[1]
    def common_xyz(self):
        z,y,x=np.meshgrid(self.grid.z,self.grid.y,self.grid.x,indexing='ij')
        return np.column_stack([x.ravel(),y.ravel(),z.ravel()])
    def station_mesh(self,s):
        if s not in self.projections:
            h=self.grid.spacing;delta=self.sources[s]-self.grid.xyz(np.array(self.station_nodes[s]))
            axes=[np.r_[a[0]-h,a,a[-1]+h]+d for a,d in zip([self.grid.x,self.grid.y,self.grid.z],delta)]
            mesh=Grid(*axes);z,y,x=np.meshgrid(mesh.z,mesh.y,mesh.x,indexing='ij');xyz=np.column_stack([x.ravel(),y.ravel(),z.ravel()])
            ids,w=interpolation_weights(self.grid,xyz,clip=True)
            P=sparse.csr_matrix((w.ravel(),(np.repeat(np.arange(len(xyz)),8),ids.ravel())),shape=(len(xyz),self.vp.size))
            self.meshes[s]=mesh;self.projections[s]=P
            while len(self.projections)>2:self.projections.popitem(last=False)
        return self.meshes[s],self.projections[s]
    def elevation_correction(self,station,nodes):return np.zeros(len(nodes))
    def predict(self,picks,event_nodes):
        result=np.empty(len(picks))
        for s,ids in picks.groupby('station',sort=False).indices.items():
            nodes=np.asarray([event_nodes[e] for e in picks.iloc[ids].event_id]);result[ids]=self.get(s)[tuple(nodes.T)]
        return result
    def predict_exact(self,picks):
        result=np.empty(len(picks))
        for s,ids in picks.groupby('station',sort=False).indices.items():
            xyz=picks.iloc[ids][['source_x_km','source_y_km','ev_prof_km']].to_numpy(float)
            mesh,_=self.station_mesh(s)
            u=RegularGridInterpolator((mesh.z,mesh.y,mesh.x),self.pair(s)[0])(xyz[:,::-1])
            result[ids]=u*np.linalg.norm(xyz-self.sources[s],axis=1)
        return result
    def stats(self):return dict(model_key=self.model_key,fields_calculated=self.calculated,fields_loaded=self.loaded,field_seconds=self.seconds,resident_field_limit=self.max_fields,operator='factored FM; physical receiver elevation')
    def close(self):self.memory.clear()

def interpolation_weights(grid,xyz,clip=False):
    xyz=np.asarray(xyz,float);h=grid.spacing;q=(xyz-np.array([grid.x[0],grid.y[0],grid.z[0]]))/h
    sizes=np.array([len(grid.x),len(grid.y),len(grid.z)])
    if clip:q=np.clip(q,0,sizes-1)
    elif np.any(q< -1e-10) or np.any(q>sizes-1+1e-10):raise ValueError('Interpolation outside grid')
    q=np.clip(q,0,sizes-1);lo=np.minimum(np.floor(q).astype(int),sizes-2);f=q-lo;ids=[];weights=[]
    for dz in [0,1]:
        for dy in [0,1]:
            for dx in [0,1]:
                ijk=lo+np.array([dx,dy,dz]);ids.append((ijk[:,2]*len(grid.y)+ijk[:,1])*len(grid.x)+ijk[:,0]);weights.append((f[:,0] if dx else 1-f[:,0])*(f[:,1] if dy else 1-f[:,1])*(f[:,2] if dz else 1-f[:,2]))
    return np.array(ids).T,np.array(weights).T

def active_mask(matrix,grid,domain,projection,minimum_hits):
    return _active_mask(abs(matrix)>=.1,grid,domain,projection,minimum_hits)

def kernels(cache,picks,nodes,coarse,backend='factored_cpp',q_mode=False):
    if q_mode:raise ValueError('Only P slowness supported')
    m=mapping(cache.grid,coarse);nc=int(np.prod(coarse.shape));blocks=[];ids_all=[];errors=[]
    for count,(s,ids) in enumerate(picks.groupby('station',sort=False).indices.items(),1):
        mesh,P=cache.station_mesh(s)
        # P maps fixed model slowness to station-centred mesh; differentiate P.
        C=sparse.csr_matrix((np.ones(len(m)),(np.arange(len(m)),m)),shape=(len(m),nc))
        Q=(P@C).tocsr()
        with Graph((P@(1/cache.vp).ravel()).reshape(mesh.shape),mesh,cache.sources[s],cache.order) as g:
            for part in np.array_split(ids,max(1,int(np.ceil(len(ids)/32)))):
                n=np.asarray([nodes[e] for e in picks.iloc[part].event_id]);start=cache.grid.linear(n)
                xyz=cache.grid.xyz(n);rr=np.linalg.norm(xyz-cache.sources[s],axis=1)
                ii,ww=interpolation_weights(mesh,xyz)
                # Native accumulation through sparse model projection Q.
                rows=g.adjoint_projected(ii,ww*rr[:,None],Q)
                # Identity uses direct model, avoiding coarse averaging at boundaries.
                # Local directional FD and full identity tested independently.
                blocks.append(sparse.csr_matrix(rows));ids_all.extend(part)
        if count%20==0:print(f'Factored adjoint: {count} stations',flush=True)
    mat=sparse.vstack(blocks,format='csr')[np.argsort(ids_all)]
    return mat,dict(operator='discrete factored FM reverse differentiation including source-cell interpolation',failed_rows=[],matrix_nnz=mat.nnz,kernel_time_minus_field_median_s=None,kernel_time_minus_field_p95_abs_s=None,kernel_time_minus_field_max_abs_s=None,limitation='Signed second-order sensitivities; coverage uses absolute values only, inversion retains signed Jacobian')
