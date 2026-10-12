"""Portable entry point: change cache location only, not scientific parameters."""
from pathlib import Path
import sys,runpy
import robustez_20261011 as controls

if __name__=='__main__':
    controls.CFG['cache_directory']=str(controls.ROOT/'resultados/cache_reproduccion')
    kind=sys.argv[1]
    if kind=='spatial':controls.spatial()
    else:
        scripts={'mesh':'convergencia_malla_20261011.py','synthetic':'sinteticos_no_lineales_20261011.py','summary':'resumir_robustez_20261011.py'}
        path=Path(__file__).parent/scripts[kind]
        runpy.run_path(str(path),run_name='__main__')
