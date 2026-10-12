"""Independent fine-grid nonlinear data, relocation/statics/velocity inversion."""
from robustez_20261011 import *
from convergencia_malla_20261011 import refined

def truth(grid,amp):
    z,y,x=np.meshgrid(grid.z,grid.y,grid.x,indexing='ij')
    sign=np.where((np.floor((x-COARSE.x[0])/40)+np.floor((y-COARSE.y[0])/40)+np.floor(z/40)).astype(int)%2==0,1.,-1.)
    # Perturbations held fixed in physical coordinates on both grids.
    base=np.broadcast_to(np.interp(grid.z,*PROFILE)[:,None,None],grid.shape)
    return base*(1+amp*sign)
def recovery(folder,amp,p):
    f=np.load(folder/'vp_final.npz');vp=f['vp'];active=f['active']
    imposed=truth(GRID,amp);base=F['vp_initial']
    from regional.rays import mapping
    mapflat=mapping(GRID,COARSE)
    count=np.bincount(mapflat,minlength=active.size)
    reduce=lambda a:np.bincount(mapflat,weights=a.ravel(),minlength=active.size)/count
    target=reduce((imposed-base)/base)[active.ravel()];got=reduce((vp-base)/base)[active.ravel()]
    corr=float(np.corrcoef(target,got)[0,1])
    ratio=float(np.dot(target,got)/np.dot(target,target))
    result=dict(amplitude_fraction=amp,active_cells=int(active.sum()),correlation=corr,relative_l2_error=float(np.linalg.norm(got-target)/np.linalg.norm(target)),least_squares_amplitude_ratio=ratio,sign_agreement=float(np.mean(np.sign(target)==np.sign(got))),velocity_rmse_active_km_s=float(np.sqrt(np.mean((reduce(vp-imposed)[active.ravel()])**2))))
    rel=pd.read_csv(folder/'relocated_train_events.csv');true=p[['event_id','true_x_km','true_y_km','true_depth_km','injected_origin_s']].drop_duplicates('event_id')
    rel=rel.merge(true,on='event_id',validate='one_to_one')
    dr=rel[['x_km','y_km','depth_km']].to_numpy()-rel[['true_x_km','true_y_km','true_depth_km']].to_numpy()
    result['hypocenter_3d_rmse_km']=float(np.sqrt(np.mean(np.sum(dr*dr,axis=1))))
    result['origin_shift_rmse_s']=float(np.sqrt(np.mean((rel.origin_shift_s-rel.injected_origin_s)**2)))
    result['gate']=dict(correlation_min=.7,amplitude_ratio_range=[.5,1.5],relative_l2_max=.6)
    result['passes_gate']=bool(corr>=.7 and .5<=ratio<=1.5 and result['relative_l2_error']<=.6)
    np.savez(folder/'nonlinear_recovery.npz',truth_vp=imposed,recovered_vp=vp,base_vp=base,active=active,xg=GRID.x,yg=GRID.y,zg=GRID.z,inversion_xg=COARSE.x,inversion_yg=COARSE.y,inversion_zg=COARSE.z)
    save(folder/'recovery_metrics.json',result);return result
def main():
    protocol();fine=refined(5);cfg=dict(CFG,cache_directory=str(OUT/'synthetic_fields'),time_field_memory_limit_mb=192)
    p=load();rng=np.random.default_rng(20261011)
    events=p[['event_id','source_x_km','source_y_km','ev_prof_km']].drop_duplicates('event_id').sort_values('event_id').reset_index(drop=True)
    events=events.rename(columns={'source_x_km':'true_x_km','source_y_km':'true_y_km','ev_prof_km':'true_depth_km'})
    jitter=rng.uniform(-5,5,(len(events),3));events[['source_x_km','source_y_km','ev_prof_km']]=events[['true_x_km','true_y_km','true_depth_km']].to_numpy()+jitter
    # Bound both geographic location and depth within the original inversion domain.
    la,lo=PR.latlon(events.source_x_km.to_numpy(),events.source_y_km.to_numpy())
    la=np.clip(la,DOMAIN['lat_min'],DOMAIN['lat_max']);lo=np.clip(lo,DOMAIN['lon_min'],DOMAIN['lon_max'])
    events['source_x_km'],events['source_y_km']=PR.xy(la,lo);events['ev_lat']=la;events['ev_lon']=lo
    events.ev_prof_km=events.ev_prof_km.clip(0,180);events['injected_origin_s']=rng.normal(0,.2,len(events))
    stations=sorted(p.station.unique());bias=dict(zip(stations,rng.normal(0,.15,len(stations))));bias[stations[0]]=0.
    noise=rng.normal(0,.05,len(p))
    p=p.drop(columns=['source_x_km','source_y_km','ev_prof_km','ev_lat','ev_lon']).merge(events,on='event_id',how='left',validate='many_to_one')
    for amp in [.03,.08]:
        folder=OUT/f'synthetic_{int(amp*100):02d}pct';folder.mkdir(exist_ok=True)
        if (folder/'recovery_metrics.json').exists():continue
        path=folder/'synthetic_observations.csv'
        if path.exists():q=pd.read_csv(path,low_memory=False)
        else:
            vp=truth(fine,amp);cache=FieldCache(vp,fine,p[['station','st_x_km','st_y_km']].drop_duplicates(),cfg['cache_directory'],cfg)
            true_p=p.copy();true_p['source_x_km']=p.true_x_km;true_p['source_y_km']=p.true_y_km;true_p['ev_prof_km']=p.true_depth_km
            t=predict_exact(cache,true_p);cache.close()
            q=p.copy();q['true_time_s']=t;q['injected_station_s']=q.station.map(bias);q['noise_s']=noise
            q['observed_s']=t+q.injected_origin_s+q.injected_station_s+noise;q.to_csv(path,index=False)
        save(folder/'generation.json',dict(generator_grid_km=5,inverse_grid_km=10,cell_grid_km=20,noise_sigma_s=.05,amplitude_fraction=amp,block_width_km=40,seed=20261011,all_stations=len(stations),events=q.event_id.nunique(),picks=len(q),source_position='exact true catalog; inverse catalog perturbed uniformly +/-5 km per axis',origin_sigma_s=.2,station_sigma_s=.15,weights='same sigma=1 s as real-data experiment, not tuned using synthetic noise',script_sha256=digest(__file__),gate=dict(correlation_min=.7,amplitude_ratio_min=.5,amplitude_ratio_max=1.5,relative_l2_max=.6)))
        print(f'SYNTHETIC {amp*100:g}% inversion',flush=True)
        fit_pair(q,folder,dict(CFG,cache_directory=str(OUT/'inverse_synthetic_fields'),time_field_memory_limit_mb=192))
        result=recovery(folder,amp,q);print(json.dumps(result),flush=True)
    save(OUT/'nonlinear_synthetics.json',dict(experiments=[json.loads((OUT/f'synthetic_{int(a*100):02d}pct/recovery_metrics.json').read_text()) for a in [.03,.08]],generation_grid_km=5,inversion_grid_km=10,operator='nonlinear FSM data and newly recomputed fields/Jacobians each inverse round',not_shared_linear_forward_operator=True))
if __name__=='__main__':main()
