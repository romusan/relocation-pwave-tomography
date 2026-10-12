"""Benchmark C++ ray tracing against migrated reference on the same FSM field."""
from pathlib import Path
import json
from time import perf_counter
import numpy as np
from regional.fsm_reference import fast_sweeping_3d
from regional.rays import trace


def main():
    rng=np.random.default_rng(20261009)
    shape=(31,31,31)
    center=(15,15,15)
    field=fast_sweeping_3d(np.full(shape,1/6.),*center,5.,5.,5.,8,1e6)
    receiver=np.ravel_multi_index(center,shape)
    starts=np.ravel_multi_index(rng.integers(0,31,size=(3,1000)),shape)
    reference=[trace(field,int(s),int(receiver),5.,'reference') for s in starts]
    for s,expected in zip(starts,reference):
        actual=trace(field,int(s),int(receiver),5.,'cpp')
        if expected is None or actual is None:
            if expected is not actual:
                raise AssertionError('Different ray failure status')
        else:
            np.testing.assert_array_equal(expected[0],actual[0])
            np.testing.assert_allclose(expected[1],actual[1],rtol=1e-14,atol=1e-14)
    elapsed={}
    for backend in ('reference','cpp'):
        samples=[]
        for _ in range(3):
            start=perf_counter()
            for s in starts:
                trace(field,int(s),int(receiver),5.,backend)
            samples.append(perf_counter()-start)
        elapsed[backend]={'seconds':samples,'median_s':float(np.median(samples))}
    result={'rays':len(starts),'field':'homogeneous first-order FSM, 31^3 at 5 km',
        'candidate_path_disagreements':0,'timings':elapsed,
        'speedup_cpp':elapsed['reference']['median_s']/elapsed['cpp']['median_s'],
        'scope':'backtrace only, includes ctypes/array allocation; excludes fields, coarse mapping and CSR'}
    out=Path(__file__).resolve().parents[1]/'resultados/revision_2026-10-09/benchmark_rays.json'
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
