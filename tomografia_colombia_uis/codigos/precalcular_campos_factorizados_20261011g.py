"""Precompute the last 100 station fields; same frozen synthetic generator.

Two workers ahead of the main generator. Disjoint tail is completed before
the main generator reaches it. Existing fields retain hash identities.
"""
from factorizado_paralelo_20261011g import enable,BaseCache
enable()
from controles_factorizados_20261011g import *
from concurrent.futures import ThreadPoolExecutor
def main():
    p=protocol();stations=p[['station','st_x_km','st_y_km','st_elev_km']].drop_duplicates();tail=stations.iloc[-100:]
    fine=Grid(np.arange(GRID.x[0],GRID.x[-1]+2.5,5.),np.arange(GRID.y[0],GRID.y[-1]+2.5,5.),np.arange(-10,182.5,5.))
    for amp in [.03,.08]:
        vp=truth(fine,amp)
        def work(s):
            cache=BaseCache(vp,fine,tail[tail.station.eq(s)],OUT/'synthetic_cache',dict(CFG,time_field_memory_limit_mb=32));cache.pair(s);cache.close();return s
        with ThreadPoolExecutor(max_workers=2) as pool:
            for i,s in enumerate(pool.map(work,tail.station),1):print(f'Prefill {amp:g}: {i}/100 {s}',flush=True)
if __name__=='__main__':main()
