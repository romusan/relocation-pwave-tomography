from pathlib import Path
import json,numpy as np
from regional.geometry import Grid
from regional.factored import mapping
root=Path(__file__).resolve().parents[1]/'resultados/campana_factorizada_cotas_20261011g'
f=np.load(root/'real_elevation/vp_final.npz');g=Grid(f['xg'],f['yg'],f['zg']);c=Grid(f['inversion_xg'],f['inversion_yg'],f['inversion_zg'])
m=mapping(g,c);active=f['active'].ravel();vp=f['vp'].ravel();base=f['vp_initial'].ravel()
has=lambda condition:np.bincount(m,weights=condition,minlength=active.size)>0
lower_physical=has(vp<=4+1e-8);upper=has(vp>=8.8-1e-8);lower_cumulative=has(np.isclose(vp/base,.95**3,atol=1e-8,rtol=0))
r=dict(active_cells=int(active.sum()),definition='active inversion cells containing at least one direct-grid node at the specified limit',upper_physical_8p8=int((active&upper).sum()),lower_physical_4=int((active&lower_physical).sum()),lower_cumulative_three_5pct_steps=int((active&lower_cumulative).sum()),union_upper_and_cumulative=int((active&(upper|lower_cumulative)).sum()),union_percent=float(100*(active&(upper|lower_cumulative)).sum()/active.sum()),cumulative_lower_relative_vp=float(.95**3))
(root/'velocity_bound_audit.json').write_text(json.dumps(r,indent=2),encoding='utf-8');print(r)
