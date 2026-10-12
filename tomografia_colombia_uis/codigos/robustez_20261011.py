"""Frozen robustness protocol and independent experimental outputs."""
from pathlib import Path
import json,hashlib,sys,types,time
import numpy as np,pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsmr
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from regional.geometry import Grid,candidate_nodes
from regional.forward import FieldCache
from regional.pipeline_consistent import relocate,residual
from regional.inversion import station_design
from nazca_campana_20261010 import projection

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'resultados/tomografia_1d_adjunto_consistente_2026-10-10'
OUT=ROOT/'resultados/robustez_20261011';OUT.mkdir(exist_ok=True)
CFG=json.loads((OLD/'config.json').read_text())
DOMAIN=json.loads((OLD/'domain.json').read_text())
PR=projection(DOMAIN)
F=np.load(OLD/'vp_final.npz')
GRID=Grid(F['xg'],F['yg'],F['zg'])
COARSE=Grid(F['inversion_xg'],F['inversion_yg'],F['inversion_zg'])
PROFILE=(GRID.z,F['vp_initial'][:,0,0])

def load():return pd.read_csv(OLD/'predictions.csv',low_memory=False)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,data):Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
def metrics(r):
    a=np.abs(r);q=np.quantile(a,.95)
    return dict(rms_s=float(np.sqrt(np.mean(r*r))),median_abs_s=float(np.median(a)),p90_abs_s=float(np.quantile(a,.9)),trimmed95_rms_s=float(np.sqrt(np.mean(r[a<=q]**2))))
def nodes_for(p,grid=GRID):
    nodes={}
    for eid,g in p.groupby('event_id',sort=False):
        xyz=g[['source_x_km','source_y_km','ev_prof_km']].iloc[0].to_numpy(float)
        near=candidate_nodes(grid,grid.nodes(xyz),grid.spacing,DOMAIN,PR)
        nodes[eid]=tuple(near[np.argmin(np.linalg.norm(grid.xyz(near)-xyz,axis=1))])
    return nodes
def predict_exact(cache,p):
    pred=np.empty(len(p))
    for sta,ids in p.groupby('station',sort=False).indices.items():
        q=p.iloc[ids]
        pred[ids]=RegularGridInterpolator((cache.grid.z,cache.grid.y,cache.grid.x),cache.get(sta))(
            q[['ev_prof_km','source_y_km','source_x_km']].to_numpy(float))
    return pred
def bootstrap(p,a,b,seed=20261011):
    labels,inv=np.unique(p.event_id.to_numpy(),return_inverse=True)
    n=np.bincount(inv);aa=np.bincount(inv,weights=a*a);bb=np.bincount(inv,weights=b*b)
    rng=np.random.default_rng(seed);values=[]
    for _ in range(3000):
        ix=rng.integers(0,len(labels),len(labels));values.append(np.sqrt(aa[ix].sum()/n[ix].sum())-np.sqrt(bb[ix].sum()/n[ix].sum()))
    return dict(delta_rms_3d_minus_1d_s=metrics(a)['rms_s']-metrics(b)['rms_s'],ci95_s=np.quantile(values,[.025,.975]).tolist(),replicates=3000,unit='event')
def engine():
    # Preserve the production inversion; omit only costly linear diagnostics,
    # which are not the purpose of these independently generated experiments.
    original=ROOT/'codigos/regional/pipeline_consistent.py'
    source=original.read_text(encoding='utf-8')
    start=source.index('    diagnostic, imposed, recovered = linear_resolution(')
    stop=source.index('    pred_output = p.copy()',start)
    source=source[:start]+"    diagnostic={'kind':'not run: nonlinear/spatial experiment'}\n    lcurve=pd.DataFrame({'converged':[]})\n"+source[stop:]
    module=types.ModuleType('regional.robustness_engine');module.__package__='regional'
    exec(compile(source,str(original)+' [diagnostics omitted]','exec'),module.__dict__)
    return module.run_velocity,source
def fit_control(p,folder,cfg):
    train=p[p.split.eq('train')].reset_index(drop=True)
    cache=FieldCache(F['vp_initial'],GRID,p[['station','st_x_km','st_y_km']].drop_duplicates(),cfg['cache_directory'],cfg)
    nodes=nodes_for(p);origins={e:0. for e in nodes};statics={s:0. for s in p.station.unique()}
    S,free,ref=station_design(train);A=sparse.vstack([S,cfg['station_damping']*sparse.eye(len(free))],format='csr');history=[]
    for r in range(1,4):
        nodes,origins,rel=relocate(cache,train,nodes,origins,statics,DOMAIN,PR,cfg)
        before=residual(cache,train,nodes,origins,statics)
        sol=lsmr(A,np.r_[before,np.zeros(len(free))],atol=cfg['lsmr_atol'],btol=cfg['lsmr_btol'],maxiter=cfg['lsmr_maxiter'])
        assert sol[1] in (0,1,2,4,5)
        for alpha in (1.,.5,.25):
            trial=dict(statics)
            for s,d in zip(free,sol[0]):trial[s]+=alpha*float(np.clip(d,-cfg['max_static_step_s'],cfg['max_static_step_s']))
            after=residual(cache,train,nodes,origins,trial)
            if np.mean(after**2)<=np.mean(before**2)+1e-10:statics=trial;break
        history.append(dict(round=r,train=metrics(residual(cache,train,nodes,origins,statics))))
        rel.to_csv(folder/f'control_relocation_{r}.csv',index=False)
    save(folder/'control_1d.json',dict(history=history,reference=ref,statics=statics))
    return cache,nodes,statics
def fit_pair(p,folder,cfg):
    folder.mkdir(exist_ok=True)
    cfg=dict(cfg);save(folder/'config.json',cfg);save(folder/'domain.json',DOMAIN)
    events=p[['event_id','split']].drop_duplicates();events.to_csv(folder/'partition.csv',index=False)
    run,source=engine();(folder/'inversion_engine.py').write_text(source,encoding='utf-8')
    summary,cache,nodes,statics=run(p,events,GRID,COARSE,PROFILE,PR,DOMAIN,cfg,folder,{'original_engine_sha256':digest(ROOT/'codigos/regional/pipeline_consistent.py')})
    test=p[p.split.eq('holdout')].reset_index(drop=True)
    r3=test.observed_s.to_numpy()-predict_exact(cache,test)-test.station.map(statics).to_numpy();cache.close()
    c1,n1,s1=fit_control(p,folder,cfg)
    r1=test.observed_s.to_numpy()-predict_exact(c1,test)-test.station.map(s1).to_numpy();c1.close()
    result=dict(events=len(events),train_events=events.split.eq('train').sum().item(),holdout_events=events.split.eq('holdout').sum().item(),train_picks=int(p.split.eq('train').sum()),holdout_picks=len(test),control_1d=metrics(r1),model_3d=metrics(r3),paired_bootstrap=bootstrap(test,r3,r1),test_only_stations=sorted(set(test.station)-set(p[p.split.eq('train')].station)))
    q=test[['event_id','station','ev_lat','ev_lon','ev_prof_km']].copy();q['residual_1d_s']=r1;q['residual_3d_s']=r3;q.to_csv(folder/'exact_holdout_residuals.csv',index=False)
    save(folder/'comparison.json',result);print(json.dumps(result),flush=True)
    return result
def protocol():
    plan=dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),grid_spacing_km=[10,5,2.5],grid_representatives='two highest-pick-count stations in each of latitude bands <4.5, 4.5–8, >=8; all associated P paths at those six stations; station coordinates locked at original 10-km nodes',grid_models=['constant 6 km/s','initial 1D','final 3D slowness trilinear interpolant'],grid_gate=dict(rms_5_minus_2p5_s=.10,p95_abs_5_minus_2p5_s=.25,require_decreasing_refinement_error=True),spatial_bands_deg=[4.5,8.0],spatial_buffer_km=20,spatial_folds=3,spatial_note='retrospective leave-region-out refits, existing selected/associated dataset; not a newly acquired blind reserve',synthetic=dict(generator_spacing_km=5,inversion_spacing_direct_km=10,inversion_cell_km=20,block_width_km=40,amplitude_fraction=[.03,.08],noise_sigma_s=.05,seed=20261011,source_positions='exact catalog XYZ',catalog_perturbation='uniform +/-5 km per XYZ axis, boundaries clipped, not provided to inversion',origin_perturbation_sigma_s=.2,station_statics_sigma_s=.15,stations='all original station IDs',rounds=3),parameter_selection='all inversion parameters retained; no test-driven tuning',sha256={str(x):digest(x) for x in [OLD/'vp_final.npz',OLD/'predictions.csv',OLD/'config.json',Path(__file__)]})
    path=OUT/'protocol.json'
    if not path.exists():save(path,plan)
    else:
        prior=json.loads(path.read_text());assert prior['sha256']==plan['sha256'],'Protocol/code changed after freezing'
def spatial():
    protocol();p=load();e=p[['event_id','ev_lat','ev_lon','source_x_km','source_y_km']].drop_duplicates('event_id').reset_index(drop=True)
    band=np.digitize(e.ev_lat,[4.5,8.]);e['band']=band;e.to_csv(OUT/'spatial_events.csv',index=False)
    all_results=[]
    for fold in range(3):
        folder=OUT/f'spatial_fold_{fold+1}';folder.mkdir(exist_ok=True)
        if (folder/'comparison.json').exists():all_results.append(json.loads((folder/'comparison.json').read_text()));continue
        test=e[band==fold];train=e[band!=fold].copy()
        distance=cKDTree(test[['source_x_km','source_y_km']].to_numpy()).query(train[['source_x_km','source_y_km']].to_numpy())[0]
        guard=train[distance<20];train=train[distance>=20]
        guard.to_csv(folder/'buffer_excluded_events.csv',index=False)
        q=p[p.event_id.isin(pd.concat([train,test]).event_id)].copy().reset_index(drop=True)
        q['split']=np.where(q.event_id.isin(test.event_id),'holdout','train')
        print(f'SPATIAL {fold+1}: train {len(train)}, heldout {len(test)}, buffer {len(guard)}',flush=True)
        result=fit_pair(q,folder,CFG);result.update(fold=fold+1,buffer_excluded_events=len(guard));save(folder/'comparison.json',result);all_results.append(result)
    save(OUT/'spatial_validation.json',dict(folds=all_results,protocol='leave-latitude-region-out, 20 km horizontal buffer, full nonlinear refits from original 1D'))

if __name__=='__main__':
    if sys.argv[1]=='protocol':protocol()
    elif sys.argv[1]=='spatial':spatial()
