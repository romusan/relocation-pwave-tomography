"""Read existing fields; calculate new fields in RAM without writing disk caches."""
import hashlib,time,json
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from factorizado_paralelo_20261011g import BaseCache
import regional.factored as f

class MemoryCache(BaseCache):
    def __init__(self,vp,grid,stations,directory,config):
        super().__init__(vp,grid,stations,directory,dict(config,time_field_memory_limit_mb=2048))
    def pair(self,station):
        if station in self.memory:
            self.memory.move_to_end(station);return self.memory[station]
        key=hashlib.sha256(str(station).encode()+self.sources[station].tobytes()).hexdigest()
        path=self.path/(key[:16]+'.npz');start=time.perf_counter()
        if path.exists():
            with np.load(path) as q:
                assert str(q['model_key'])==self.model_key and str(q['station_key'])==key
                pair=q['u'],q['t']
            self.loaded+=1
        else:
            mesh,P=self.station_mesh(station)
            slow=(P@(1/self.vp).ravel()).reshape(mesh.shape)
            with f.Graph(slow,mesh,self.sources[station],self.order) as graph:
                xyz=self.common_xyz()
                u=RegularGridInterpolator((mesh.z,mesh.y,mesh.x),graph.u)(xyz[:,::-1]).reshape(self.grid.shape)
                pair=graph.u,u*np.linalg.norm(xyz-self.sources[station],axis=1).reshape(self.grid.shape)
            self.calculated+=1
        self.seconds+=time.perf_counter()-start;self.memory[station]=pair
        while len(self.memory)>self.max_fields:self.memory.popitem(last=False)
        return pair

def verify(h):
    picks=h.population().iloc[:1];station=picks[['station','st_x_km','st_y_km','st_elev_km']]
    vp=h.initial()+1e-7;sta=str(station.station.iloc[0])
    memory=MemoryCache(vp,h.GRID,station,h.OUT/'memory_cache_check',h.CFG)
    expected=BaseCache(vp,h.GRID,station,h.OUT/'memory_cache_reference',h.CFG)
    a=memory.pair(sta);b=expected.pair(sta)
    assert all(np.array_equal(x,y) for x,y in zip(a,b))
    assert not list(memory.path.glob('*.npz'))
    memory.close();expected.close()
    h.save(h.OUT/'memory_cache_verification.json',dict(bitwise_same_factor_and_times=True,new_field_files_written=0,fields_in_memory_mb=2048,module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))

def enable(h):
    report=json.loads((h.OUT/'memory_cache_verification.json').read_text())
    assert report['module_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    f.FieldCache=MemoryCache

if __name__=='__main__':
    import campana_perfil_suavidad_20261011h as h
    verify(h)
