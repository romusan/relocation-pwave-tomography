"""Nested full-domain forward convergence with fixed physical sources/model."""
from robustez_20261011 import *
from regional.fsm_reference import fast_sweeping_3d

def refined(h):
    return Grid(*[np.arange(a[0],a[-1]+h/2,h) for a in [GRID.x,GRID.y,GRID.z]])
def sample_volume(volume,g):
    interp=RegularGridInterpolator((GRID.z,GRID.y,GRID.x),volume)
    result=np.empty(g.shape)
    yy,xx=np.meshgrid(g.y,g.x,indexing='ij')
    for k,z in enumerate(g.z):result[k]=interp(np.column_stack([np.full(xx.size,z),yy.ravel(),xx.ravel()])).reshape(xx.shape)
    return result
def main():
    protocol();folder=OUT/'mesh';folder.mkdir(exist_ok=True)
    p=load();stations=p[['station','st_x_km','st_y_km','st_lat_current']].drop_duplicates('station')
    stations['band']=np.digitize(stations.st_lat_current,[4.5,8.]);stations['picks']=stations.station.map(p.station.value_counts())
    selected=pd.concat([g.sort_values(['picks','station'],ascending=[False,True]).head(2) for _,g in stations.groupby('band')])
    selected.to_csv(folder/'selected_stations.csv',index=False)
    q=p[p.station.isin(selected.station)].copy().reset_index(drop=True)
    q['xyz_count']=q.groupby('station').station.transform('size')
    rows=[];times=[]
    for model in ['homogeneous','initial_1d','final_3d']:
        for h in [10.,5.,2.5]:
            g=refined(h)
            slow=np.full(g.shape,1/6.) if model=='homogeneous' else sample_volume(1/F['vp_initial' if model=='initial_1d' else 'vp'],g)
            for sta in selected.station:
                part=q[q.station.eq(sta)].copy();row=selected[selected.station.eq(sta)].iloc[0]
                xyz=GRID.xyz(GRID.nodes([row.st_x_km,row.st_y_km,0.]))
                node=tuple(g.nodes(xyz));path=folder/f'{model}_{h:g}_{sta}.csv'
                if path.exists():rows.append(pd.read_csv(path));continue
                tic=time.perf_counter();field=fast_sweeping_3d(slow,*node,h,h,h,CFG['fsm_sweeps'],1e6)
                pred=RegularGridInterpolator((g.z,g.y,g.x),field)(part[['ev_prof_km','source_y_km','source_x_km']].to_numpy())
                t=pd.DataFrame(dict(raw_row=part.raw_row,event_id=part.event_id,station=sta,model=model,h_km=h,time_s=pred))
                t['distance_km']=np.linalg.norm(part[['source_x_km','source_y_km','ev_prof_km']].to_numpy()-xyz,axis=1)
                if model=='homogeneous':t['analytic_time_s']=t.distance_km/6
                t.to_csv(path,index=False);rows.append(t)
                timing=dict(model=model,h_km=h,station=sta,shape=g.shape,seconds=time.perf_counter()-tic);times.append(timing)
                print(json.dumps(timing),flush=True)
                del field
            del slow
    allp=pd.concat(rows,ignore_index=True);allp.to_csv(folder/'all_predictions.csv',index=False)
    summary=[]
    for model,d in allp.groupby('model'):
        wide=d.pivot(index=['raw_row','station'],columns='h_km',values='time_s')
        e1=(wide[10]-wide[5]).to_numpy();e2=(wide[5]-wide[2.5]).to_numpy()
        a=metrics(e1);b=metrics(e2);ratio=a['rms_s']/b['rms_s']
        item=dict(model=model,picks=len(wide),stations=len(selected),delta_10_5=a,delta_5_2p5=b,apparent_order=float(np.log2(ratio)),decreasing_error=bool(b['rms_s']<a['rms_s']),p95_abs_5_2p5_s=float(np.quantile(abs(e2),.95)))
        item['passes_frozen_gate']=bool(item['decreasing_error'] and b['rms_s']<=.10 and item['p95_abs_5_2p5_s']<=.25)
        if model=='homogeneous':
            item['analytical_error_by_h']={str(h):metrics((v.time_s-v.analytic_time_s).to_numpy()) for h,v in d.groupby('h_km')}
        summary.append(item)
    save(folder/'summary.json',dict(models=summary,station_source='locked original 10 km station node at all refinements',model_source='same full-domain trilinear slowness interpolant; no refitting',parameters_unchanged=True,gate_scope='numerical forward consistency on six fixed station samples, not whole inversion convergence',timings=times,script_sha256=digest(__file__),input_sha256=json.loads((OUT/'protocol.json').read_text())['sha256']))
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
