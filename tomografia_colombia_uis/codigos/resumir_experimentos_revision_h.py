"""Read-only aggregation, permitted only after the numerical campaign ends."""
from pathlib import Path
import json,sys,hashlib
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];C=ROOT/'resultados/campana_perfil_suavidad_20261011h';G=ROOT/'resultados/campana_factorizada_cotas_20261011g'
sys.path.insert(0,str(ROOT/'codigos'))
from robustez_20261011 import bootstrap
from regional.geometry import Grid
from regional.factored import mapping
load=lambda p:json.loads(p.read_text(encoding='utf-8'))
assert load(C/'completed.json')['status']=='completed'
assert (C/'elevation_contrast.json').exists()
arms=['fixed_1d','updated_1d','real_smooth_2','real_smooth_20','real_smooth_200']
control=pd.read_csv(C/'updated_1d/holdout_residuals.csv')
summaries=[];rounds=[]
for arm in arms:
    folder=C/arm;r=load(folder/'comparison.json');a=pd.read_csv(folder/'holdout_residuals.csv')
    assert a[['event_id','station']].equals(control[['event_id','station']])
    r['arm']=arm;r['comparison_to_updated_1d']=bootstrap(a,a.residual_s.to_numpy(),control.residual_s.to_numpy())
    h=load(folder/'history.json');rounds.extend(dict(arm=arm,**v) for v in h)
    f=np.load(folder/'vp_final.npz');fine=Grid(f['xg'],f['yg'],f['zg']);coarse=Grid(f['inversion_xg'],f['inversion_yg'],f['inversion_zg']);m=mapping(fine,coarse);mask=f['active'].ravel()
    vp=f['vp'].ravel();initial=f['vp_initial'].ravel();counts=np.bincount(m,minlength=mask.size)
    assert np.isfinite(vp).all() and vp.min()>=4.-1e-12 and vp.max()<=8.8+1e-12
    if arm in ['fixed_1d','updated_1d']:
        assert np.ptp(f['vp'],axis=(1,2)).max()<1e-12,'1D arm has lateral variation'
    else:
        assert np.array_equal(vp[~mask[m]],initial[~mask[m]]),'Inactive nodes changed'
    residual=pd.read_csv(folder/'train_residuals.csv').residual_s.to_numpy()
    assert abs(np.sqrt(np.mean(residual**2))-r['train']['rms_s'])<1e-12
    assert abs(np.sqrt(np.mean(a.residual_s.to_numpy()**2))-r['holdout']['rms_s'])<1e-12
    monotone=all(h[i]['objective']<=h[i-1]['objective']+1e-7 for i in range(1,len(h)))
    c=np.bincount(m,weights=1/vp-1/initial,minlength=mask.size)/counts
    dif=[]
    model=c.reshape(coarse.shape)
    for axis in range(3):
        ok=np.take(f['active'],range(coarse.shape[axis]-1),axis=axis)&np.take(f['active'],range(1,coarse.shape[axis]),axis=axis)
        dif.extend(np.diff(model,axis=axis)[ok].tolist())
    r['roughness_neighbor_slowness_rms_s_per_km']=float(np.sqrt(np.mean(np.square(dif))))
    bounds={}
    for name,predicate in [('physical_lower',np.isclose(vp,4.,atol=1e-9,rtol=0)),('physical_upper',np.isclose(vp,8.8,atol=1e-9,rtol=0)),('cumulative_lower_step',np.isclose(vp,np.maximum(4.,initial*.95**len(h)),atol=1e-8,rtol=0))]:
        hit=np.bincount(m,weights=predicate.astype(int),minlength=mask.size)>0
        bounds[name]=int((hit&mask).sum())
    r['bounds_active_cells']=bounds;r['active_cells']=int(mask.sum())
    if (folder/'mesh_convergence.json').exists():r['mesh']=load(folder/'mesh_convergence.json')
    r['audit']=dict(finite_physical_velocity_bounds=True,predictions_rms_recomputed=True,monotone_training_objective=monotone,parameter_basis_preserved=True)
    summaries.append(r)
pd.DataFrame(rounds).to_csv(C/'rms_por_ronda.csv',index=False)
synthetic=[];q=pd.read_csv(C/'synthetic_effective_errors.csv',low_memory=False)
truth=q[['event_id','true_x_km','true_y_km','true_depth_km','source_x_km','source_y_km','ev_prof_km','split']].drop_duplicates('event_id');truth=truth[truth.split.eq('train')]
t=truth[['true_x_km','true_y_km','true_depth_km']].to_numpy();x=truth[['source_x_km','source_y_km','ev_prof_km']].to_numpy()
initial_rmse=float(np.sqrt(np.mean(np.sum((x-t)**2,axis=1))))
control=pd.read_csv(C/'synthetic_updated_1d/holdout_residuals.csv')
for arm in ['synthetic_updated_1d','synthetic_smooth_2','synthetic_smooth_20','synthetic_smooth_200']:
    folder=C/arm;r=load(folder/('recovery_metrics.json' if 'smooth' in arm else 'comparison.json'));r['arm']=arm
    rel=pd.read_csv(folder/'state_events.csv').merge(truth,on='event_id',validate='one_to_one')
    assert len(rel)==len(truth)
    f=np.load(folder/'vp_final.npz');grid=Grid(f['xg'],f['yg'],f['zg'])
    d=grid.xyz(rel[['k','j','i']].to_numpy(int))-rel[['true_x_km','true_y_km','true_depth_km']].to_numpy()
    r['initial_catalog_train_hypocentre_rmse_km']=initial_rmse;r['final_train_hypocentre_rmse_km']=float(np.sqrt(np.mean(np.sum(d*d,axis=1))));r['hypocentre_train_events']=len(rel)
    a=pd.read_csv(folder/'holdout_residuals.csv');assert a[['event_id','station']].equals(control[['event_id','station']])
    r['comparison_to_updated_1d']=bootstrap(a,a.residual_s.to_numpy(),control.residual_s.to_numpy())
    synthetic.append(r)
elevation=load(C/'elevation_contrast.json')
elevation['actual_control']=load(C/'fixed_1d/comparison.json')
elevation['zero_control']=load(C/'fixed_1d_zero_elevation/comparison.json')
elevation['paired_event_bootstrap']['delta_rms_actual_minus_zero_s']=elevation['paired_event_bootstrap']['delta_rms_3d_minus_1d_s']
report=dict(real=summaries,synthetic=synthetic,effective_errors=load(C/'synthetic_error_calibration.json'),elevation=elevation,data=load(C/'data_description_audit.json'),arrival_sources=load(C/'arrival_source_audit.json'),verification=load(C/'verification.json'),protocol=load(C/'protocol.json'))
(C/'manuscript_metrics.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(dict(real=[(r['arm'],r['train']['rms_s'],r['holdout']['rms_s'],r['converged']) for r in summaries],synthetic=[(r['arm'],r.get('correlation'),r.get('relative_l2_error')) for r in synthetic]),indent=2))
