"""Bit-equivalent FSM: omit remaining sweeps only at an exact fixed point."""
from pathlib import Path
import sys,runpy,time,json,hashlib
import numpy as np
from numba import njit
from regional.fsm_reference import _solve_local,fast_sweeping_3d as reference

SOURCE=Path(__file__).resolve().parent/'regional/fsm_reference.py'
source=SOURCE.read_text(encoding='utf-8')
source=source[source.index('@njit(cache=True)\ndef fast_sweeping_3d'):]
source=source.replace('@njit(cache=True)','@njit(cache=False)').replace('def fast_sweeping_3d(','def equivalent(')
assert source.count('for _ in range(n_sweeps):')==1
source=source.replace('for _ in range(n_sweeps):','for _ in range(n_sweeps):\n        changed = False')
source=source.replace('T[k, j, i] = new_t','T[k, j, i] = new_t\n                                    changed = True')
source=source.replace('    return T','        if not changed:\n            break\n    return T')
exec(compile(source,str(Path(__file__)),'exec'),globals())

def certify():
    from robustez_20261011 import OUT,CFG,GRID,F,digest
    from convergencia_malla_20261011 import refined
    from sinteticos_no_lineales_20261011 import truth
    rng=np.random.default_rng(20261011);tests=[]
    for label,slow,node,h in [('random',rng.uniform(.10,.22,(21,25,29)),(3,4,7),10.),('real_initial',1/F['vp_initial'],(0,71,48),10.)]:
        a=reference(slow,*node,h,h,h,8,1e6);b=equivalent(slow,*node,h,h,h,8,1e6)
        assert np.array_equal(a,b),label
        tests.append(dict(case=label,bitwise_equal=True,max_abs_diff_s=float(np.max(abs(a-b)))))
    fine=refined(5);vp=truth(fine,.03)
    folder=OUT/'synthetic_fields/936908c3d169074d'
    candidates=list((OUT/'synthetic_fields').glob('936908c3*/*.npy'))
    assert candidates,'Need already completed fine-grid reference fields'
    for path in candidates[:3]:
        meta=json.loads(path.with_suffix('.json').read_text());node=tuple(meta['node'])
        tic=time.perf_counter();b=equivalent(1/vp,*node,5.,5.,5.,8,1e6)
        a=np.load(path,mmap_mode='r');assert np.array_equal(a,b),meta['station']
        tests.append(dict(case='fine_synthetic_'+meta['station'],bitwise_equal=True,max_abs_diff_s=float(np.max(abs(a-b))),optimized_seconds=time.perf_counter()-tic))
        del a,b
    out=dict(tests=tests,proof='After a full sweep makes zero strict decreases, the same deterministic update sees identical inputs in every subsequent sweep; otherwise all eight sweeps execute.',reference_sha256=digest(SOURCE),optimized_sha256=digest(__file__),changes_to_physical_model=False,changes_to_parameters=False)
    (OUT/'fsm_equivalence.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2),flush=True)

if __name__=='__main__':
    if sys.argv[1]=='certify':certify()
    else:
        from robustez_20261011 import OUT,digest
        cert=json.loads((OUT/'fsm_equivalence.json').read_text());assert cert['optimized_sha256']==digest(__file__)
        import regional.fsm_reference,regional.forward
        regional.fsm_reference.fast_sweeping_3d=equivalent;regional.forward.fast_sweeping_3d=equivalent
        target=Path(sys.argv[1]).resolve();sys.argv=[str(target),*sys.argv[2:]]
        runpy.run_path(str(target),run_name='__main__')
