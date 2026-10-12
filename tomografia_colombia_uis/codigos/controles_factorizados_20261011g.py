"""Nonlinear recovery and buffered spatial refits for factored campaign."""
from campana_factorizada_cotas_20261011g import *
from scipy.spatial import cKDTree
from regional.factored import mapping

def truth(grid,amp):
    z,y,x=np.meshgrid(grid.z,grid.y,grid.x,indexing='ij')
    sign=np.where((np.floor((x-COARSE.x[0])/40)+np.floor((y-COARSE.y[0])/40)+np.floor(np.maximum(z,0)/40)).astype(int)%2==0,1.,-1.)
    base=np.broadcast_to(np.interp(grid.z,*PROFILE)[:,None,None],grid.shape)
    return base*(1+amp*sign)
def recovery(folder,amp,q):
    f=np.load(folder/'vp_final.npz');vp=f['vp'];mask=f['active'].ravel();imposed=truth(GRID,amp);base=initial();m=mapping(GRID,COARSE);count=np.bincount(m,minlength=mask.size)
    reduce=lambda a:np.bincount(m,weights=a.ravel(),minlength=mask.size)/count
    target=reduce((imposed-base)/base)[mask];got=reduce((vp-base)/base)[mask]
    r=dict(amplitude_fraction=amp,active_cells=int(mask.sum()),correlation=float(np.corrcoef(target,got)[0,1]),relative_l2_error=float(np.linalg.norm(got-target)/np.linalg.norm(target)),least_squares_amplitude_ratio=float(target@got/(target@target)),sign_agreement=float(np.mean(np.sign(target)==np.sign(got))))
    true=q[['event_id','true_x_km','true_y_km','true_depth_km','source_x_km','source_y_km','ev_prof_km']].drop_duplicates('event_id')
    train=q[q.split.eq('train')].event_id.unique();true=true[true.event_id.isin(train)]
    xyztrue=true[['true_x_km','true_y_km','true_depth_km']].to_numpy();xyzcat=true[['source_x_km','source_y_km','ev_prof_km']].to_numpy();r['initial_catalog_rmse_km']=float(np.sqrt(np.mean(np.sum((xyzcat-xyztrue)**2,axis=1))))
    rel=pd.read_csv(folder/'relocated_train_events.csv').merge(true,on='event_id',validate='one_to_one');delta=rel[['x_km','y_km','depth_km']].to_numpy()-rel[['true_x_km','true_y_km','true_depth_km']].to_numpy()
    r['final_3d_hypocentre_rmse_km']=float(np.sqrt(np.mean(np.sum(delta*delta,axis=1))))
    ctrl=pd.read_csv(folder/'control_relocation_3.csv').merge(true,on='event_id',validate='one_to_one');delta=ctrl[['x_km','y_km','depth_km']].to_numpy()-ctrl[['true_x_km','true_y_km','true_depth_km']].to_numpy()
    r['final_1d_hypocentre_rmse_km']=float(np.sqrt(np.mean(np.sum(delta*delta,axis=1))))
    r['passes_gate']=bool(r['correlation']>=.7 and .5<=r['least_squares_amplitude_ratio']<=1.5 and r['relative_l2_error']<=.6)
    np.savez(folder/'nonlinear_recovery.npz',truth_vp=imposed,recovered_vp=vp,base_vp=base,active=f['active'],xg=GRID.x,yg=GRID.y,zg=GRID.z,inversion_xg=COARSE.x,inversion_yg=COARSE.y,inversion_zg=COARSE.z)
    save(folder/'recovery_metrics.json',r);print(json.dumps(r),flush=True)
    return r
def synthetic():
    p=protocol();fine=Grid(np.arange(GRID.x[0],GRID.x[-1]+2.5,5.),np.arange(GRID.y[0],GRID.y[-1]+2.5,5.),np.arange(-10,182.5,5.))
    # Reuse the frozen original perturbations and noise for all retained rows.
    old=pd.read_csv(ROOT/'resultados/robustez_20261011/synthetic_03pct/synthetic_observations.csv',low_memory=False)
    columns=['event_id','station','raw_row','true_x_km','true_y_km','true_depth_km','source_x_km','source_y_km','ev_prof_km','ev_lat','ev_lon','injected_origin_s','injected_station_s','noise_s']
    keys=['event_id','station','raw_row'];replace=[c for c in columns if c not in keys]
    q=p.drop(columns=[c for c in replace if c in p.columns]).merge(old[columns],on=keys,validate='one_to_one')
    assert len(q)==len(p),'Synthetic population mismatch'
    reports=[]
    for amp in [.03,.08]:
        folder=OUT/f'synthetic_{int(100*amp):02d}pct';folder.mkdir(exist_ok=True)
        if (folder/'recovery_metrics.json').exists():reports.append(json.loads((folder/'recovery_metrics.json').read_text()));continue
        path=folder/'synthetic_observations.csv'
        if path.exists():obs=pd.read_csv(path,low_memory=False,float_precision='round_trip')
        else:
            stations=q[['station','st_x_km','st_y_km','st_elev_km']].drop_duplicates();cache=FieldCache(truth(fine,amp),fine,stations,OUT/'synthetic_cache',dict(CFG,time_field_memory_limit_mb=96))
            tp=q.copy();tp['source_x_km']=q.true_x_km;tp['source_y_km']=q.true_y_km;tp['ev_prof_km']=q.true_depth_km
            obs=q.copy();obs['true_time_s']=cache.predict_exact(tp);cache.close();obs['observed_s']=obs.true_time_s+obs.injected_origin_s+obs.injected_station_s+obs.noise_s;obs.to_csv(path,index=False)
        save(folder/'generation.json',dict(generator_spacing_km=5,inversion_spacing_km=10,amplitude=amp,perturbations='same frozen original row noise/origin/locations/statics, filtered to matched station population',reference_sha256=sha(ROOT/'resultados/robustez_20261011/synthetic_03pct/synthetic_observations.csv'),controls_sha256=sha(__file__)))
        fit_pair(obs,folder);reports.append(recovery(folder,amp,obs))
    save(OUT/'nonlinear_synthetics.json',reports)
def spatial():
    p=protocol();e=p[['event_id','ev_lat','source_x_km','source_y_km']].drop_duplicates('event_id');e['band']=np.digitize(e.ev_lat,[4.5,8.]);results=[]
    for fold in range(3):
        folder=OUT/f'spatial_fold_{fold+1}';folder.mkdir(exist_ok=True)
        if (folder/'comparison.json').exists():results.append(json.loads((folder/'comparison.json').read_text()));continue
        test=e[e.band.eq(fold)];train=e[~e.band.eq(fold)];d=cKDTree(test[['source_x_km','source_y_km']].to_numpy()).query(train[['source_x_km','source_y_km']].to_numpy())[0]
        guard=train[d<20];train=train[d>=20];guard.to_csv(folder/'buffer_excluded_events.csv',index=False)
        q=p[p.event_id.isin(pd.concat([test,train]).event_id)].copy().reset_index(drop=True);q['split']=np.where(q.event_id.isin(test.event_id),'holdout','train')
        r=fit_pair(q,folder);r.update(fold=fold+1,excluded_buffer_events=len(guard));save(folder/'comparison.json',r);results.append(r)
    save(OUT/'spatial_validation.json',results)
def all_controls():
    synthetic();spatial()
if __name__=='__main__':globals()[sys.argv[1]]()
