"""Isolated campaign: 1D start, factored operator, actual station elevation.

Training and holdout event identities and all inversion hyperparameters are
inherited; station metadata must match coordinates. No zero imputation.
Model translated to each station mesh by slowness interpolation; its transpose
is included in the adjoint. Negative depths contain extrapolated surface rock.
"""
from pathlib import Path
import sys,json,types,hashlib,time
import numpy as np,pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsmr
from scipy.interpolate import RegularGridInterpolator
from regional.geometry import Grid,candidate_nodes
from regional.factored import FieldCache
from regional.inversion import station_design
from nazca_campana_20261010 import projection
from robustez_20261011 import metrics,bootstrap

ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/'resultados/tomografia_1d_adjunto_consistente_2026-10-10'
OUT=ROOT/'resultados/campana_factorizada_cotas_20261011g';OUT.mkdir(exist_ok=True)
F=np.load(OLD/'vp_final.npz');GRID=Grid(F['xg'],F['yg'],np.r_[-10.,F['zg']]);COARSE=Grid(F['inversion_xg'],F['inversion_yg'],F['inversion_zg'])
PROFILE=(F['zg'],F['vp_initial'][:,0,0]);DOMAIN=json.loads((OLD/'domain.json').read_text());PR=projection(DOMAIN)
CFG=json.loads((OLD/'config.json').read_text());CFG.update(ray_backend='factored_cpp',factored_order=2,station_elevation=True,cache_directory=str(OUT/'cache'))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load():
    p=pd.read_csv(OLD/'predictions.csv',low_memory=False)
    elev=pd.read_csv(ROOT/'resultados/revision_independiente_20261011f/estaticas_y_elevacion.csv')
    valid=elev[elev.elevation_m.notna()][['station','elevation_m','metadata_source','coordinate_difference_deg']]
    p=p.merge(valid,on='station',how='inner',validate='many_to_one');p['st_elev_km']=p.elevation_m/1000
    counts=p.groupby('event_id').station.nunique();good=counts[counts>=CFG['min_p_stations']].index
    p=p[p.event_id.isin(good)].reset_index(drop=True)
    return p
def engine():
    original=ROOT/'codigos/regional/pipeline_consistent.py';source=original.read_text(encoding='utf-8')
    source=source.replace('from .forward import FieldCache','from .factored import FieldCache').replace('from .rays_consistent import kernels, mapping, active_mask','from .factored import kernels, mapping, active_mask')
    source=source.replace("p[['station', 'st_x_km', 'st_y_km']]","p[['station', 'st_x_km', 'st_y_km', 'st_elev_km']]")
    start=source.index('    diagnostic, imposed, recovered = linear_resolution(');stop=source.index('    pred_output = p.copy()',start)
    source=source[:start]+"    diagnostic={'kind':'not run: nonlinear campaign'}\n    lcurve=pd.DataFrame({'converged':[]})\n"+source[stop:]
    # All held-out predictions use exact catalog XYZ, including report output.
    source=source.replace("holdout_res = residual(cache, test, initial_nodes, {e: 0. for e in nodes}, statics)","holdout_res = test.observed_s.to_numpy()-cache.predict_exact(test)-test.station.map(statics).to_numpy()")
    source=source.replace("pred_output['predicted_s'] = cache.predict(p, nodes)","pred_output['predicted_s'] = cache.predict(p, nodes)\n    held=pred_output.split.eq('holdout')\n    pred_output.loc[held,'predicted_s']=cache.predict_exact(pred_output[held].reset_index(drop=True))")
    source=source.replace("'status': 'consistent_discrete_first_order_FSM'","'status': 'factored_FM_actual_station_elevation'")
    start=source.index("        'station_elevation_km': {");stop=source.index("        'sources':",start)
    source=source[:start]+"        'station_elevation_km': {'enabled': True,'method':'receiver at continuous XYZ, depth=-elevation; factored station-centred mesh'},\n"+source[stop:]
    start=source.index("        'limitations': [");stop=source.index("    (out/'summary.json')",start)
    source=source[:start]+"        'limitations': ['Station-centred shifted grids use interpolated slowness; adjoint differentiates this interpolation.', 'Negative depth is extrapolated surface rock, not explicit topography or Earth curvature.', 'Relocation searches 10-km grid nodes; heldout predictions use continuous hypocentres.', 'Mesh and nonlinear resolution are independent gates.']}\n    summary['projection_audit']['depth_convention']='positive down; stations at depth=-elevation_km'\n"+source[stop:]
    module=types.ModuleType('regional.factored_campaign_engine');module.__package__='regional';exec(compile(source,str(original)+' [isolated factored variant]','exec'),module.__dict__)
    return module,source
def protocol():
    check=json.loads((OUT/'solver_verification.json').read_text());assert check['pass']
    p=load();e=p[['event_id','split']].drop_duplicates();p.to_csv(OUT/'selected_picks.csv',index=False);e.to_csv(OUT/'partition.csv',index=False)
    paths=[Path(__file__),ROOT/'codigos/regional/pipeline_consistent.py',ROOT/'codigos/regional/factored.py',ROOT/'codigos/native/factored_eikonal.cpp',ROOT/'codigos/native/factored_eikonal.dll',OLD/'predictions.csv',OLD/'vp_final.npz',ROOT/'resultados/revision_independiente_20261011f/estaticas_y_elevacion.csv']
    plan=dict(operator='second-order factored fast marching; source-aligned translated mesh',station_geometry='continuous X,Y, depth=-elevation; no pairwise additive correction',model_start='same frozen 1D',excluded_station_ids=sorted(set(pd.read_csv(OLD/'predictions.csv',usecols=['station']).station)-set(p.station)),events=len(e),train_events=int(e.split.eq('train').sum()),holdout_events=int(e.split.eq('holdout').sum()),picks=len(p),stations=p.station.nunique(),config=CFG,mesh_spacing_km=[10,5,2.5],mesh_gates=dict(rms_last_refinement_s=.1,p95_last_refinement_s=.25),synthetics=dict(amplitudes=[.03,.08],generator_spacing_km=5,seed=20261011,block_km=40),spatial=dict(bands=[4.5,8.],buffer_km=20),sha256={str(x):sha(x) for x in paths})
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())['sha256']==plan['sha256'],'Frozen code changed'
    else:save(path,plan)
    return p
def initial():return np.broadcast_to(np.interp(GRID.z,*PROFILE)[:,None,None],GRID.shape).copy()
def nodes_for(p):
    nodes={}
    for e,g in p.groupby('event_id',sort=False):
        xyz=g[['source_x_km','source_y_km','ev_prof_km']].iloc[0].to_numpy(float);near=candidate_nodes(GRID,GRID.nodes(xyz),GRID.spacing,DOMAIN,PR)
        nodes[e]=tuple(near[np.argmin(np.linalg.norm(GRID.xyz(near)-xyz,axis=1))])
    return nodes
def fit_control(p,folder,cfg,module):
    train=p[p.split.eq('train')].reset_index(drop=True);stations=p[['station','st_x_km','st_y_km','st_elev_km']].drop_duplicates()
    cache=FieldCache(initial(),GRID,stations,cfg['cache_directory'],cfg);nodes=nodes_for(p);origins={e:0. for e in nodes};statics={s:0. for s in p.station.unique()}
    S,free,ref=station_design(train);A=sparse.vstack([S,cfg['station_damping']*sparse.eye(len(free))],format='csr');history=[]
    for r in range(1,cfg['rounds']+1):
        nodes,origins,rel=module.relocate(cache,train,nodes,origins,statics,DOMAIN,PR,cfg)
        before=module.residual(cache,train,nodes,origins,statics)
        sol=lsmr(A,np.r_[before,np.zeros(len(free))],atol=cfg['lsmr_atol'],btol=cfg['lsmr_btol'],maxiter=cfg['lsmr_maxiter']);assert sol[1] in (0,1,2,4,5)
        for alpha in [1.,.5,.25]:
            trial=dict(statics)
            for s,d in zip(free,sol[0]):trial[s]+=alpha*float(np.clip(d,-cfg['max_static_step_s'],cfg['max_static_step_s']))
            after=module.residual(cache,train,nodes,origins,trial)
            if np.mean(after**2)<=np.mean(before**2)+1e-10:statics=trial;break
        rel.to_csv(folder/f'control_relocation_{r}.csv',index=False);history.append(dict(round=r,train=metrics(module.residual(cache,train,nodes,origins,statics))))
    save(folder/'control_1d.json',dict(history=history,reference=ref,statics=statics));return cache,nodes,statics
def fit_pair(p,folder):
    folder.mkdir(exist_ok=True);cfg=dict(CFG);save(folder/'config.json',cfg);save(folder/'domain.json',DOMAIN)
    module,source=engine();(folder/'inversion_engine.py').write_text(source,encoding='utf-8');events=p[['event_id','split']].drop_duplicates();events.to_csv(folder/'partition.csv',index=False)
    if not (folder/'vp_final.npz').exists():
        summary,c3,n3,s3=module.run_velocity(p,events,GRID,COARSE,PROFILE,PR,DOMAIN,cfg,folder,{'protocol_sha256':sha(OUT/'protocol.json')})
    else:
        model=np.load(folder/'vp_final.npz');c3=FieldCache(model['vp'],GRID,p[['station','st_x_km','st_y_km','st_elev_km']].drop_duplicates(),cfg['cache_directory'],cfg)
        stat=pd.read_csv(folder/'station_statics.csv');s3=dict(zip(stat.station,stat.static_s))
    test=p[p.split.eq('holdout')].reset_index(drop=True);r3=test.observed_s.to_numpy()-c3.predict_exact(test)-test.station.map(s3).to_numpy();c3.close()
    c1,n1,s1=fit_control(p,folder,cfg,module);r1=test.observed_s.to_numpy()-c1.predict_exact(test)-test.station.map(s1).to_numpy();c1.close()
    result=dict(control_1d=metrics(r1),model_3d=metrics(r3),paired_bootstrap=bootstrap(test,r3,r1),train_events=int(events.split.eq('train').sum()),holdout_events=len(test.event_id.unique()),holdout_picks=len(test))
    q=test[['event_id','station','ev_lat','ev_lon','ev_prof_km']].copy();q['residual_1d_s']=r1;q['residual_3d_s']=r3;q.to_csv(folder/'exact_holdout_residuals.csv',index=False)
    save(folder/'comparison.json',result);print(json.dumps(result),flush=True);return result
def real():
    p=protocol();folder=OUT/'real_elevation'
    if not (folder/'comparison.json').exists():fit_pair(p,folder)
def mesh():
    p=protocol();stations=p[['station','st_x_km','st_y_km','st_elev_km']].drop_duplicates();chosen=[]
    for lo,hi in [(-90,4.5),(4.5,8),(8,90)]:chosen+=p[(p.ev_lat>=lo)&(p.ev_lat<hi)].station.value_counts().head(2).index.tolist()
    chosen=list(dict.fromkeys(chosen));p=p[p.station.isin(chosen)].reset_index(drop=True);data=[]
    for model in ['homogeneous','initial_1d','previous_3d']:
        for h in [10.,5.,2.5]:
            path=OUT/f'mesh_{model}_{h:g}.csv'
            if path.exists():continue
            grid=Grid(np.arange(GRID.x[0],GRID.x[-1]+h/2,h),np.arange(GRID.y[0],GRID.y[-1]+h/2,h),np.arange(-10,180+h/2,h))
            if model=='homogeneous':vp=np.full(grid.shape,6.)
            elif model=='initial_1d':vp=np.broadcast_to(np.interp(grid.z,*PROFILE)[:,None,None],grid.shape).copy()
            else:
                z,y,x=np.meshgrid(grid.z,grid.y,grid.x,indexing='ij');sl=RegularGridInterpolator((F['zg'],F['yg'],F['xg']),1/F['vp'])(np.column_stack([np.maximum(z.ravel(),0),y.ravel(),x.ravel()]));vp=(1/sl).reshape(grid.shape)
            cfg=dict(CFG,time_field_memory_limit_mb=128);cache=FieldCache(vp,grid,stations[stations.station.isin(chosen)],OUT/'mesh_cache',cfg)
            q=p[['event_id','station']].copy();q['predicted_s']=cache.predict_exact(p)
            if model=='homogeneous':q['analytic_s']=np.linalg.norm(p[['source_x_km','source_y_km','ev_prof_km']].to_numpy()-np.array([cache.sources[s] for s in p.station]),axis=1)/6
            q.to_csv(path,index=False);cache.close();print(f'MESH {model} {h} complete',flush=True)
        runs=[pd.read_csv(OUT/f'mesh_{model}_{h:g}.csv') for h in [10,5,2.5]];dif=[]
        for i in [0,1]:
            d=runs[i].predicted_s.to_numpy()-runs[i+1].predicted_s.to_numpy();dif.append(dict(rms_s=float(np.sqrt(np.mean(d*d))),p95_s=float(np.quantile(abs(d),.95))))
        entry=dict(model=model,picks=len(p),station_ids=chosen,differences=dif,pass_gate=dif[-1]['rms_s']<=.1 and dif[-1]['p95_s']<=.25)
        if model=='homogeneous':entry['max_analytic_error_s']=max(float(abs(q.predicted_s-q.analytic_s).max()) for q in runs)
        data.append(entry);save(OUT/'mesh_convergence.json',data)
if __name__=='__main__':
    globals()[sys.argv[1]]()
