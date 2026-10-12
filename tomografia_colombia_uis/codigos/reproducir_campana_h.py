"""Repeat the frozen protocol into a separate directory without deleting files."""
import campana_perfil_suavidad_20261011h as h
h.OUT=h.ROOT/'resultados/reproduccion_perfil_suavidad_h'
h.OUT.mkdir(exist_ok=True)
h.CFG=dict(h.CFG,cache_directory=str(h.OUT/'cache'))
import cache_memoria_h
cache_memoria_h.verify(h)
cache_memoria_h.enable(h)
original_lsmr=h.lsmr
def checked_lsmr(A,b,**kwargs):
    answer=original_lsmr(A,b,**kwargs)
    if answer[1]==7:
        for limit in [10000,30000]:
            answer=original_lsmr(A,b,**dict(kwargs,maxiter=limit))
            print(f'LSMR retry: limit={limit}, stop={answer[1]}, iterations={answer[2]}',flush=True)
            if answer[1]!=7:break
    return answer
h.lsmr=checked_lsmr
h.all()
