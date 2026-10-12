"""Identify old h-only regenerable fields; preserve all saved models and g caches."""
from pathlib import Path
import hashlib,json,datetime
import numpy as np
import campana_perfil_suavidad_20261011h as h
root=Path(h.CFG['cache_directory']).resolve()
protected=set()
for result in [h.GOUT,h.OUT]:
    for path in result.rglob('*.npz'):
        if root in path.parents:continue
        try:
            with np.load(path) as q:
                if not all(k in q for k in ['vp','xg','yg','zg']):
                    if path.name=='state.npz' and 'vp' in q:
                        vp=q['vp'];axes=[h.GRID.x,h.GRID.y,h.GRID.z]
                    else:continue
                else:vp=q['vp'];axes=[q['xg'],q['yg'],q['zg']]
                digest=hashlib.sha256(np.ascontiguousarray(vp,float).tobytes())
                for axis in axes:digest.update(axis.tobytes())
                digest.update(Path(h.f.__file__).read_bytes())
                digest.update((h.ROOT/'codigos/native/factored_eikonal.cpp').read_bytes())
                digest.update(b'2');protected.add(digest.hexdigest()[:16])
        except (ValueError,KeyError):continue
cutoff=datetime.datetime(2026,10,11,15,58).timestamp()
recent=datetime.datetime.now().timestamp()-3600
targets=[]
for directory in root.iterdir():
    if not directory.is_dir() or directory.name in protected:continue
    if len(directory.name)!=16 or any(c not in '0123456789abcdef' for c in directory.name):continue
    files=list(directory.iterdir())
    if not files or any(not p.is_file() or p.suffix!='.npz' for p in files):continue
    # Creation after the h controls started; no recent writer in the directory.
    if directory.stat().st_ctime<cutoff or max(p.stat().st_mtime for p in files)>recent:continue
    targets.append(dict(path=str(directory.resolve()),bytes=sum(p.stat().st_size for p in files)))
report=dict(cache_root=str(root),targets=targets,bytes=sum(p['bytes'] for p in targets),protected_model_keys=sorted(protected),scope='only regenerable h intermediate fields, no saved models or early g fields; no file written within one hour')
h.save(h.OUT/'intermediate_cache_cleanup_plan.json',report)
print(json.dumps(dict(directories=len(targets),bytes=report['bytes']),indent=2))
