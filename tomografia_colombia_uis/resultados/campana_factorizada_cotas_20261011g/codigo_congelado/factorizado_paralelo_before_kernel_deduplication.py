"""Bitwise-equivalent four-thread adjoint; serial forward and frozen protocol."""
from pathlib import Path
import ctypes,json,sys
import numpy as np
import regional.factored as f
from scipy.interpolate import RegularGridInterpolator
from concurrent.futures import ThreadPoolExecutor
BaseCache=f.FieldCache
class ParallelFieldCache(BaseCache):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # Independent station fields, identical arithmetic within each field.
        # Workers return no arrays, avoiding a second retained population.
        def precompute(station):
            local=BaseCache(*args,**kwargs)
            local.max_fields=1
            local.pair(station)
            counts=local.calculated,local.loaded,local.seconds
            local.close()
            return counts
        workers=2 if self.vp.size>1000000 else 4
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for calculated,loaded,seconds in pool.map(precompute,list(self.sources)):
                self.calculated+=calculated;self.loaded+=loaded;self.seconds+=seconds
    def predict_exact(self,picks):
        # Prediction needs mesh axes, not its sparse model Jacobian P.
        result=np.empty(len(picks));grid=self.grid;h=grid.spacing
        for s,ids in picks.groupby('station',sort=False).indices.items():
            delta=self.sources[s]-grid.xyz(np.array(self.station_nodes[s]))
            axes=[np.r_[a[0]-h,a,a[-1]+h]+d for a,d in zip([grid.x,grid.y,grid.z],delta)]
            xyz=picks.iloc[ids][['source_x_km','source_y_km','ev_prof_km']].to_numpy(float)
            u=RegularGridInterpolator(tuple(axes[::-1]),self.pair(s)[0])(xyz[:,::-1])
            result[ids]=u*np.linalg.norm(xyz-self.sources[s],axis=1)
        return result
def enable():
    old=f.native();path=Path(__file__).resolve().parent/'native/factored_eikonal_parallel.dll';lib=ctypes.CDLL(str(path))
    for name in ['factored_create','factored_destroy','factored_adjoint','factored_projected']:
        getattr(lib,name).argtypes=getattr(old,name).argtypes;getattr(lib,name).restype=getattr(old,name).restype
    f.LIB=lib
    f.FieldCache=ParallelFieldCache
    return old,lib
def verify():
    from regional.geometry import Grid
    rng=np.random.default_rng(74);grid=Grid(np.arange(0,110,10.),np.arange(0,130,10.),np.arange(-10,90,10.));slow=.16+rng.random(grid.shape)*.02;source=np.array([20.,30.,0.]);starts=rng.integers(0,slow.size,(12,1));weights=np.ones((12,1));m=np.arange(slow.size,dtype=np.int64)
    old,new=enable();f.LIB=old
    with f.Graph(slow,grid,source) as g:A=g.adjoint(starts,weights,m,slow.size);T=g.t.copy()
    f.LIB=new
    with f.Graph(slow,grid,source) as g:B=g.adjoint(starts,weights,m,slow.size);U=g.t.copy()
    assert np.array_equal(A,B) and np.array_equal(T,U),'Parallel backend changes numerical results'
    import verificar_factorizado_20261011 as test
    test.main()
    out=Path(__file__).resolve().parents[1]/'resultados/campana_factorizada_cotas_20261011g';(out/'parallel_equivalence.json').write_text(json.dumps(dict(bitwise_same_fields=True,bitwise_same_adjoint_rows=True,threads=4,rows=len(starts)),indent=2))
if __name__=='__main__':
    if sys.argv[1]=='verify':verify()
    else:
        enable()
        if sys.argv[1]=='real':
            from campana_factorizada_cotas_20261011g import real
            real()
        else:
            import controles_factorizados_20261011g as controls
            getattr(controls,sys.argv[1])()
