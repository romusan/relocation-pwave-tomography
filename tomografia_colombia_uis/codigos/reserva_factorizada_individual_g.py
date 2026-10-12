"""Independent spatial fold; same mathematical protocol, separate field cache."""
from pathlib import Path
import sys,inspect,json
import factorizado_paralelo_20261011g as backend
backend.enable()
import controles_factorizados_20261011g as c
fold=int(sys.argv[1]);assert fold in [1,2,3]
c.CFG['cache_directory']=str(c.OUT/f'cache_fold_{fold}')
source=inspect.getsource(c.spatial).replace('for fold in range(3):',f'for fold in [{fold-1}]:').replace("OUT/'spatial_validation.json'",f"OUT/'spatial_fold_report_{fold}.json'")
exec(compile(source,__file__,'exec'),c.__dict__)
c.spatial()
