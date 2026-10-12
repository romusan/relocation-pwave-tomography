"""Resume the frozen campaign; retry LSMR iteration limits, never accept stop 7."""
from pathlib import Path
import json, hashlib, os
import paralelizar_jacobiano_h as parallel
h=parallel.h
original_lsmr=h.lsmr
original_fit=h.fit
attempts=[]

def checked_lsmr(A,b,**kwargs):
    answer=original_lsmr(A,b,**kwargs)
    record=dict(rows=A.shape[0],parameters=A.shape[1],initial_stop=int(answer[1]),initial_iterations=int(answer[2]),atol=kwargs['atol'],btol=kwargs['btol'])
    if answer[1]==7:
        for limit in [10000,30000]:
            retry=dict(kwargs,maxiter=limit)
            answer=original_lsmr(A,b,**retry)
            record.setdefault('retries',[]).append(dict(limit=limit,stop=int(answer[1]),iterations=int(answer[2]),normr=float(answer[3]),normar=float(answer[4]),condition=float(answer[6])))
            print('H LSMR retry '+json.dumps(record['retries'][-1]),flush=True)
            if answer[1]!=7:break
    attempts.append(record)
    h.save(h.OUT/'lsmr_resume_audit.json',attempts)
    return answer

def preserve_completed(p,kind,weight,folder,*args,**kwargs):
    complete=Path(folder)/'comparison.json'
    if complete.exists():
        result=json.loads(complete.read_text())
        assert result['kind']==kind and result['laplace_weight']==weight
        assert (Path(folder)/'vp_final.npz').is_file()
        print('H preserve completed '+str(folder),flush=True)
        return result
    return original_fit(p,kind,weight,folder,*args,**kwargs)

verification=json.loads((h.OUT/'station_parallel_verification.json').read_text())
assert verification['helper_sha256']==hashlib.sha256(Path(parallel.__file__).read_bytes()).hexdigest()
assert verification['core_sha256']==hashlib.sha256(Path(h.__file__).read_bytes()).hexdigest()
h.save(h.OUT/'lsmr_execution_amendment.json',dict(pid=os.getpid(),reason='LSMR stop 7 at 1000 iterations in real lambda 200; no result accepted from that solve',retry_limits=[10000,30000],tolerances_unchanged=True,objective_unchanged=True,completed_arms_preserved=True,core_sha256=verification['core_sha256'],driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
h.jacobian=parallel.parallel
h.lsmr=checked_lsmr
h.fit=preserve_completed
h.all()
