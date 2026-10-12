from pathlib import Path
import json,sys,time
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'codigos'))
from regional.geometry import grids
from regional.forward import FieldCache
from regional.fsm_reference import fast_sweeping_3d
from regional.rays import mapping
from regional.rays_consistent import DiscreteGraph
from nazca_campana_20261010 import projection
BASE=ROOT/'resultados/migracion_nazca_cauca_valle_eje_1d_2026-10-10'
OUT=ROOT/'resultados/tomografia_1d_adjunto_consistente_2026-10-10'
cfg=json.loads((BASE/'config.json').read_text());domain=json.loads((BASE/'domain.json').read_text());pr=projection(domain);grid,coarse=grids(domain,pr,10,20)
p=pd.read_csv(BASE/'associated_picks_selected.csv',low_memory=False);p=p[p.fase.eq('P')];r=pd.read_csv(BASE/'relocated_train_events.csv');nodes={q.event_id:tuple(grid.nodes([q.x_km,q.y_km,q.depth_km])) for q in r.itertuples()}
p=p[p.event_id.isin(nodes)];stations=p[['station','st_x_km','st_y_km']].drop_duplicates();mp=mapping(grid,coarse);nc=int(np.prod(coarse.shape))
with np.load(BASE/'vp_final.npz') as f:vp=f['vp'].copy()
cache=FieldCache(vp,grid,stations,cfg['cache_directory'],cfg);slow=1/vp;results=[];rng=np.random.default_rng(7731)
for station in p.station.value_counts().index[[0,20,80]]:
    selected=p[p.station.eq(station)].head(12);starts=grid.linear(np.array([nodes[x] for x in selected.event_id]));source=cache.station_nodes[station];src=int(np.ravel_multi_index(source,grid.shape));field=cache.get(station)
    graph=DiscreteGraph(field,slow,mp,src,10,nc);G,mom,_=graph.batch(starts);graph.close()
    for direction_id in range(2):
        d=rng.normal(0,.001,nc);ds=d[mp].reshape(grid.shape);eps=1e-4
        plus=fast_sweeping_3d(slow+eps*ds,*source,10.,10.,10.,8,1e6)
        minus=fast_sweeping_3d(slow-eps*ds,*source,10.,10.,10.,8,1e6)
        fd=(plus.ravel()[starts]-minus.ravel()[starts])/(2*eps);linear=G@d
        error=np.linalg.norm(linear-fd)/max(np.linalg.norm(fd),1e-15)
        results.append(dict(station=station,direction=direction_id,events=len(starts),relative_l2_directional_error=float(error),max_absolute_derivative_error=float(abs(linear-fd).max()),max_time_identity_error_s=float(abs(mom-field.ravel()[starts]).max())))
        assert error<2e-4,(station,error)
cache.close();report={'independent_real_model_fd_checks':results,'relative_error_gate':2e-4,'success':True,'note':'Perturbations recompute full FSM fields, not the Jacobian operator.'}
(OUT/'finite_difference_real_model.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
