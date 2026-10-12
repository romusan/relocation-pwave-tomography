"""1D ablation with the original three relocation/static rounds and bounds."""
from ablacion_revision_20261011 import *
from regional.pipeline_consistent import relocate,residual
from regional.inversion import station_design

def main():
    cfg=json.loads((RUN/'config.json').read_text());domain=json.loads((RUN/'domain.json').read_text());pr=projection(domain)
    p=pd.read_csv(RUN/'predictions.csv',low_memory=False);train=p[p.split.eq('train')].reset_index(drop=True);test=p[p.split.eq('holdout')].reset_index(drop=True)
    f=np.load(RUN/'vp_final.npz');grid=Grid(f['xg'],f['yg'],f['zg']);nodes={}
    for eid,g in p.groupby('event_id',sort=False):
        xyz=g[['source_x_km','source_y_km','ev_prof_km']].iloc[0].to_numpy();near=candidate_nodes(grid,grid.nodes(xyz),grid.spacing,domain,pr)
        nodes[eid]=tuple(near[np.argmin(np.linalg.norm(grid.xyz(near)-xyz,axis=1))])
    cache=FieldCache(f['vp_initial'],grid,p[['station','st_x_km','st_y_km']].drop_duplicates(),cfg['cache_directory'],cfg)
    origins={e:0. for e in nodes};statics={s:0. for s in p.station.unique()};S,free,ref=station_design(train);hist=[]
    A=sparse.vstack([S,cfg['station_damping']*sparse.eye(len(free))],format='csr')
    for r in range(1,4):
        nodes,origins,rel=relocate(cache,train,nodes,origins,statics,domain,pr,cfg)
        before=residual(cache,train,nodes,origins,statics)
        sol=lsmr(A,np.r_[before,np.zeros(len(free))],atol=cfg['lsmr_atol'],btol=cfg['lsmr_btol'],maxiter=cfg['lsmr_maxiter']);assert sol[1] in (1,2,4,5)
        old=dict(statics);accepted=False
        for alpha in [1.,.5,.25]:
            trial=dict(old)
            for s,d in zip(free,sol[0]):trial[s]+=alpha*float(np.clip(d,-cfg['max_static_step_s'],cfg['max_static_step_s']))
            after=residual(cache,train,nodes,origins,trial)
            if np.mean(after**2)<=np.mean(before**2)+1e-10:statics=trial;accepted=True;break
        hist.append(dict(round=r,train=metrics(residual(cache,train,nodes,origins,statics)),holdout_node=metrics(residual(cache,test,nodes,{e:0. for e in nodes},statics)),accepted=accepted))
        print(json.dumps(hist[-1]),flush=True)
        rel.to_csv(OUT/f'1d_relocation_round_{r}.csv',index=False)
    saved=pd.read_csv(OUT/'holdout_ablation_residuals.csv');assert list(saved.raw_row)==list(test.raw_row)
    delta=test.station.map(statics).to_numpy()
    rows=[]
    for coord in ['node','continuous']:
        residuals=saved[coord+'_1d_no_statics'].to_numpy()-delta
        saved[coord+'_1d_relocated_statics']=residuals
        rows.append(dict(coordinates=coord,model='1d_relocated_statics',events=test.event_id.nunique(),picks=len(test),**metrics(residuals)))
    saved.to_csv(OUT/'holdout_ablation_residuals.csv',index=False)
    metrics_file=OUT/'holdout_ablation_metrics.csv';df=pd.read_csv(metrics_file);df=df[df.model.ne('1d_relocated_statics')];pd.concat([df,pd.DataFrame(rows)],ignore_index=True).to_csv(metrics_file,index=False)
    pd.DataFrame(dict(station=list(statics),static_s=list(statics.values()))).to_csv(OUT/'1d_relocated_station_statics.csv',index=False)
    (OUT/'1d_relocated_ablation.json').write_text(json.dumps(dict(history=hist,metrics=rows,reference_station=ref,notes=['Same catalog-node initialization, 10 km local search radius, three rounds, origin bound 8 s, station step bound 0.5 s and damping 2 as 3D experiment.','Velocity held fixed at original 1D throughout; held-out hypocenters and origins never optimized.','Continuous holdout audit changes sampling only; train relocation uses the original node search.'],source_sha256={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [Path(__file__),RUN/'vp_final.npz',RUN/'predictions.csv',RUN/'config.json']}),indent=2),encoding='utf-8')
    cache.close();print(json.dumps(rows,indent=2),flush=True)
if __name__=='__main__':main()
