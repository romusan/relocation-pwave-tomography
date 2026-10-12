"""Discrete first-order eikonal Jacobian, not geometrical 26-neighbour rays.

Parent coefficients (T-a)/h² / sum((T-a)/h²), local dT/ds=s/sum.
No kernel normalization or clipping; reverse graph accumulation is exact.
"""
from pathlib import Path
import ctypes,os
import numpy as np
from scipy import sparse
from .rays import mapping
from .inversion import active_mask as _active_mask
def active_mask(matrix,grid,domain,projection,minimum_hits):
    # Coverage only: retain the complete Jacobian for inversion and diagnostics.
    # 0.1 km = 1% of the 10 km forward spacing in this controlled experiment.
    return _active_mask(matrix>=.1,grid,domain,projection,minimum_hits)
_LIB=None
def native():
    global _LIB
    if _LIB is None:
        path=Path(__file__).resolve().parents[1]/'native'/('eikonal_adjoint.dll' if os.name=='nt' else 'eikonal_adjoint.so')
        lib=ctypes.CDLL(str(path));a=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');i=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
        lib.adjoint_create.argtypes=[a,a,i,*([ctypes.c_int64]*4),ctypes.c_double,ctypes.c_int64,ctypes.POINTER(ctypes.c_double)];lib.adjoint_create.restype=ctypes.c_void_p
        lib.adjoint_batch.argtypes=[ctypes.c_void_p,i,ctypes.c_int64,a,a,i];lib.adjoint_batch.restype=ctypes.c_int
        lib.adjoint_destroy.argtypes=[ctypes.c_void_p];lib.adjoint_destroy.restype=None
        _LIB=lib
    return _LIB
class DiscreteGraph:
    def __init__(self,field,slowness,map_flat,source,spacing,ncoarse):
        self.lib=native();self.ncoarse=ncoarse;self.residual=ctypes.c_double()
        self.handle=self.lib.adjoint_create(np.ascontiguousarray(field),np.ascontiguousarray(slowness),np.ascontiguousarray(map_flat,dtype=np.int64),*field.shape,int(source),spacing,ncoarse,ctypes.byref(self.residual))
        if not self.handle:raise RuntimeError('Invalid or unconverged eikonal field: discrete Jacobian rejected')
    def batch(self,starts):
        starts=np.ascontiguousarray(starts,dtype=np.int64);rows=np.empty((len(starts),self.ncoarse));mom=np.empty(len(starts));visits=np.empty(len(starts),dtype=np.int64)
        if self.lib.adjoint_batch(self.handle,starts,len(starts),rows,mom,visits):raise RuntimeError('Discrete adjoint failure')
        return rows,mom,visits
    def close(self):
        if self.handle:self.lib.adjoint_destroy(self.handle);self.handle=None
def kernels(cache,picks,nodes,coarse,backend='adjoint_cpp',q_mode=False):
    if backend!='adjoint_cpp' or q_mode:raise ValueError('Discrete Jacobian supports P slowness inversion only; Q needs a physical attenuation operator')
    direct=cache.grid;map_flat=mapping(direct,coarse);nc=int(np.prod(coarse.shape));matrix_blocks=[];row_ids=[];mismatch=[];max_residual=0;visits_all=[]
    slow=np.ascontiguousarray(1/cache.vp)
    for n,(station,ids) in enumerate(picks.groupby('station',sort=False).indices.items(),1):
        field=cache.get(station);source=int(direct.linear(np.array([cache.station_nodes[station]]))[0]);graph=DiscreteGraph(field,slow,map_flat,source,direct.spacing,nc)
        max_residual=max(max_residual,graph.residual.value)
        try:
            for part in np.array_split(ids,max(1,int(np.ceil(len(ids)/32)))):
                starts=direct.linear(np.array([nodes[e] for e in picks.iloc[part].event_id]))
                rows,mom,v=graph.batch(starts);matrix_blocks.append(sparse.csr_matrix(rows));row_ids.extend(part.tolist());mismatch.extend((mom-field.ravel()[starts]).tolist());visits_all.extend(v.tolist())
        finally:graph.close()
        if n%20==0:print(f'Adjoint: {n} stations, {len(row_ids)} equations',flush=True)
    grouped=sparse.vstack(matrix_blocks,format='csr');order=np.argsort(row_ids);G=grouped[order]
    return G,{'operator':'exact discrete Godunov first-order Jacobian','failed_rows':[],'kernel_time_minus_field_median_s':float(np.median(mismatch)),'kernel_time_minus_field_p95_abs_s':float(np.percentile(np.abs(mismatch),95)),'kernel_time_minus_field_max_abs_s':float(np.max(np.abs(mismatch))),'max_local_eikonal_relative_residual':max_residual,'median_causal_nodes_per_equation':float(np.median(visits_all)),'matrix_nnz':G.nnz,'limitation':'Discrete sensitivities include grid diffusion; not geometrical ray lengths or independent physical accuracy.'}
