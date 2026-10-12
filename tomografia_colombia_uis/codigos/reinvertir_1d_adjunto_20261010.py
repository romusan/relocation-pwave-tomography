from pathlib import Path
import json,shutil,sys,hashlib,argparse,time
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'codigos'))
from regional.geometry import grids
from regional.forward import FieldCache
from regional.rays_consistent import kernels
from regional.pipeline_consistent import run_velocity
from nazca_campana_20261010 import projection
BASE=ROOT/'resultados/migracion_nazca_cauca_valle_eje_1d_2026-10-10'
OUT=ROOT/'resultados/tomografia_1d_adjunto_consistente_2026-10-10'
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--diagnostic-only',action='store_true');args=ap.parse_args()
    cfg=json.loads((BASE/'config.json').read_text());cfg['ray_backend']='adjoint_cpp';cfg['kernel_coverage_min_equivalent_length_km']=.1
    domain=json.loads((BASE/'domain.json').read_text());pr=projection(domain);grid,coarse=grids(domain,pr,10,20)
    picks=pd.read_csv(BASE/'associated_picks_selected.csv',low_memory=False);events=pd.read_csv(BASE/'events_split.csv')
    with np.load(BASE/'vp_final.npz') as f:oldvp=f['vp'].copy()
    stations=picks[picks.fase.eq('P')][['station','st_x_km','st_y_km']].drop_duplicates()
    if args.diagnostic_only:
        rel=pd.read_csv(BASE/'relocated_train_events.csv');nodes={r.event_id:tuple(grid.nodes([r.x_km,r.y_km,r.depth_km])) for r in rel.itertuples()}
        train=picks[picks.fase.eq('P') & picks.event_id.isin(nodes)].reset_index(drop=True)
        best=train.station.value_counts().index[0];sample=train[train.station.eq(best)].head(60).reset_index(drop=True)
        cache=FieldCache(oldvp,grid,stations,cfg['cache_directory'],cfg);start=time.perf_counter();G,info=kernels(cache,sample,nodes,coarse,'adjoint_cpp');info['seconds']=time.perf_counter()-start
        print(json.dumps(info,indent=2),flush=True);cache.close();return
    assert not (OUT/'summary.json').exists(),'Create a new run rather than replace verified results'
    OUT.mkdir(parents=True,exist_ok=True)
    originals=json.loads((BASE/'manifest.json').read_text())
    for name,h in originals.items():assert sha(name)==h,name
    manifest={str(Path(__file__)):sha(__file__)}
    for name in ['associated_picks_selected.csv','events_split.csv','catalogo_visualizacion_ampliado.csv','domain.json']:
        shutil.copy2(BASE/name,OUT/name);manifest[str(BASE/name)]=sha(BASE/name)
    for name in ['regional/pipeline_consistent.py','regional/rays_consistent.py','native/eikonal_adjoint.cpp','native/eikonal_adjoint.dll','test_adjoint_consistent.py']:
        f=ROOT/'codigos'/name;manifest[str(f)]=sha(f)
    manifest[str(BASE/'manifest.json')]=sha(BASE/'manifest.json')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (OUT/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    profile=pd.read_csv(ROOT/'datos/perfil_vp_vmm_transferencia.csv')
    summary,cache,nodes,statics=run_velocity(picks,events,grid,coarse,(profile.depth_km.to_numpy(),profile.vp_km_s.to_numpy()),pr,domain,cfg,OUT,manifest)
    assert abs(summary['baseline']['holdout_rms_fixed_catalog_origin_s']-1.6497264362535538)<1e-10
    for name,h in originals.items():assert sha(name)==h,name
    cache.close();print('Completed consistent inversion:',OUT,flush=True)
if __name__=='__main__':main()
