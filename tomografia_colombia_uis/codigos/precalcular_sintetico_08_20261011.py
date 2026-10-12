"""Precompute identical 8% generator fields in an isolated worker cache."""
from sinteticos_no_lineales_20261011 import truth,refined
from robustez_20261011 import *
import shutil

def main():
    p=load();g=refined(5);cfg=dict(CFG,time_field_memory_limit_mb=64)
    cache=FieldCache(truth(g,.08),g,p[['station','st_x_km','st_y_km']].drop_duplicates(),OUT/'warm08_isolated',cfg)
    destination=OUT/'synthetic_fields'/cache.path.name;destination.mkdir(parents=True,exist_ok=True)
    identity=cache.path/'model_identity.json'
    if not (destination/identity.name).exists():shutil.copy2(identity,destination/identity.name)
    for sta in p.station.unique():
        key=hashlib.sha256((sta+str(cache.station_nodes[sta])).encode()).hexdigest()[:16]
        target=destination/(key+'.npy')
        if target.exists():continue
        cache.get(sta)
        # Metadata first, then publish the complete identical array atomically.
        shutil.copy2(cache.path/(key+'.json'),destination/(key+'.json'))
        tmp=destination/(key+'.warmtmp');shutil.copy2(cache.path/(key+'.npy'),tmp)
        try:
            if not target.exists():tmp.replace(target)
            else:tmp.unlink()
        except PermissionError:
            if target.exists():tmp.unlink(missing_ok=True)
            else:raise
    stats=cache.stats();cache.close();save(OUT/'warm08_complete.json',dict(cache=stats,script_sha256=digest(__file__),physical_change=False))
if __name__=='__main__':main()
