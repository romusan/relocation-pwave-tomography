"""Independent extended experiment; does not modify or overwrite the g protocol."""
from pathlib import Path
import sys,json,time,hashlib
import numpy as np,pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsmr
from scipy.interpolate import RegularGridInterpolator
from factorizado_paralelo_20261011g import enable
enable()
import regional.factored as f
from campana_factorizada_cotas_20261011g import ROOT,OUT as GOUT,GRID,COARSE,CFG,DOMAIN,PR,initial,nodes_for,engine
from regional.inversion import station_design,laplacian
from robustez_20261011 import metrics,bootstrap
OUT=ROOT/'resultados/campana_perfil_suavidad_20261011h';OUT.mkdir(exist_ok=True)
save=lambda path,obj:Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
CFG=dict(CFG,cache_directory=str(GOUT/'cache'))
MAX_ROUNDS=12
SMOOTH=[2.,20.,200.]
module,_=engine()

def population():
    return pd.read_csv(GOUT/'selected_picks.csv',low_memory=False,float_precision='round_trip')

def basis(kind,mask):
    n=initial().size
    if kind=='1d':
        # Independent depth parameters at every nonnegative direct-grid depth.
        # The negative rock layer shares the z=0 parameter.
        ids=np.repeat(np.maximum(np.arange(len(GRID.z))-1,0),len(GRID.x)*len(GRID.y))
        nc=len(GRID.z)-1
        B=sparse.csr_matrix((np.ones(n),(np.arange(n),ids)),shape=(n,nc))
        rr=[];cc=[];v=[]
        for k in range(nc-2):
            rr.extend([k]*3);cc.extend([k,k+1,k+2]);v.extend([1.,-2.,1.])
        L=sparse.csr_matrix((v,(rr,cc)),shape=(nc-2,nc))
    else:
        ids=f.mapping(GRID,COARSE);active=np.flatnonzero(mask)
        C=sparse.csr_matrix((np.ones(n),(np.arange(n),ids)),shape=(n,mask.size))
        B=C[:,active];L=laplacian(mask)
    count=np.asarray(B.sum(axis=0)).ravel()
    assert (count>0).all()
    return B.tocsr(),L.tocsr(),count

def jacobian(cache,picks,nodes,B):
    rows=[];order=[]
    for count,(sta,ids) in enumerate(picks.groupby('station',sort=False).indices.items(),1):
        mesh,P=cache.station_mesh(sta);Q=(P@B).tocsr()
        with f.Graph((P@(1/cache.vp).ravel()).reshape(mesh.shape),mesh,cache.sources[sta],cache.order) as graph:
            for part in np.array_split(ids,max(1,int(np.ceil(len(ids)/64)))):
                xyz=GRID.xyz(np.array([nodes[e] for e in picks.iloc[part].event_id]))
                ii,ww=f.interpolation_weights(mesh,xyz)
                a=graph.adjoint_projected(ii,ww*np.linalg.norm(xyz-cache.sources[sta],axis=1)[:,None],Q)
                rows.append(sparse.csr_matrix(a));order.extend(part)
        if count%20==0:print(f'H adjoint {count} stations / {B.shape[1]} parameters',flush=True)
    return sparse.vstack(rows,format='csr')[np.argsort(order)]

def fit(p,kind,weight,folder,max_rounds=MAX_ROUNDS):
    folder.mkdir(exist_ok=True)
    if (folder/'comparison.json').exists():return json.loads((folder/'comparison.json').read_text())
    train=p[p.split.eq('train')].reset_index(drop=True);test=p[p.split.eq('holdout')].reset_index(drop=True)
    stations=p[['station','st_x_km','st_y_km','st_elev_km']].drop_duplicates()
    base=initial();vp=base.copy();nodes=nodes_for(p);origins={e:0. for e in nodes};statics={s:0. for s in p.station.unique()}
    S,free,reference=station_design(train,True)
    mask=np.load(GOUT/'real_elevation/model_round_1.npz')['active']
    B,L,count=basis(kind,mask) if kind!='fixed' else (None,None,None)
    history=[];start=1
    if (folder/'state.npz').exists():
        q=np.load(folder/'state.npz');vp=q['vp'];e=pd.read_csv(folder/'state_events.csv');nodes.update({r.event_id:(r.k,r.j,r.i) for r in e.itertuples()});origins.update(dict(zip(e.event_id,e.origin_s)))
        statics.update(json.loads((folder/'state_statics.json').read_text()));history=json.loads((folder/'history.json').read_text());start=len(history)+1
    cache=f.FieldCache(vp,GRID,stations,CFG['cache_directory'],CFG)
    delta=lambda v:np.asarray(B.T@((1/v-1/base).ravel())).ravel()/count
    def objective(res,v,st):
        value=float(res@res/CFG['sigma_p_s']**2)+CFG['station_damping']**2*sum(st[s]**2 for s in free)
        if B is not None:
            c=delta(v);value+=weight**2*float((L@c)@(L@c))+CFG['cell_damping']**2*float(c@c)
        return value
    for r in range(start,max_rounds+1):
        begin=time.perf_counter();oldvp=vp.copy()
        nodes,origins,rel=module.relocate(cache,train,nodes,origins,statics,DOMAIN,PR,CFG)
        before=module.residual(cache,train,nodes,origins,statics)
        nc=B.shape[1] if B is not None else 0
        A=jacobian(cache,train,nodes,B) if nc else sparse.csr_matrix((len(train),0))
        aug=[sparse.hstack([A,S])/CFG['sigma_p_s']];rhs=[before/CFG['sigma_p_s']]
        if nc:
            c=delta(vp)
            aug += [sparse.hstack([weight*L,sparse.csr_matrix((L.shape[0],S.shape[1]))]),sparse.hstack([CFG['cell_damping']*sparse.eye(nc),sparse.csr_matrix((nc,S.shape[1]))])]
            rhs += [-weight*(L@c),-CFG['cell_damping']*c]
        aug += [sparse.hstack([sparse.csr_matrix((len(free),nc)),CFG['station_damping']*sparse.eye(len(free))])]
        rhs += [-CFG['station_damping']*np.array([statics[s] for s in free])]
        sol=lsmr(sparse.vstack(aug,format='csr'),np.concatenate(rhs),atol=1e-6,btol=1e-6,maxiter=1000)
        assert sol[1] in (0,1,2,4,5),f'LSMR {sol[1]}'
        accepted=False;prev=objective(before,vp,statics)
        for alpha in [1.,.5,.25,.125,.0625]:
            trial=(1/np.maximum(1/vp+alpha*(B@sol[0][:nc]).reshape(GRID.shape),1/8.8)) if nc else vp.copy()
            trial=np.clip(trial,np.maximum(4.,.95*vp),np.minimum(8.8,1.05*vp))
            st=dict(statics)
            for sta,step in zip(free,sol[0][nc:]):st[sta]+=alpha*float(np.clip(step,-.5,.5))
            tc=f.FieldCache(trial,GRID,stations,CFG['cache_directory'],CFG) if nc else cache
            after=module.residual(tc,train,nodes,origins,st);obj=objective(after,trial,st)
            if obj<=prev+1e-10:
                if tc is not cache:cache.close()
                cache=tc;vp=trial;statics=st;accepted=True;break
            if tc is not cache:tc.close()
        final=module.residual(cache,train,nodes,origins,statics)
        held=test.observed_s.to_numpy()-cache.predict_exact(test)-test.station.map(statics).to_numpy()
        row=dict(round=r,train_rms_s=metrics(final)['rms_s'],holdout_rms_s=metrics(held)['rms_s'],objective=objective(final,vp,statics),accepted=accepted,max_vp_fraction=float(np.max(abs(vp/oldvp-1))),seconds=time.perf_counter()-begin,lsmr_istop=int(sol[1]))
        history.append(row);save(folder/'history.json',history)
        np.savez(folder/'state.npz',vp=vp)
        e=pd.DataFrame([dict(event_id=e,k=n[0],j=n[1],i=n[2],origin_s=origins[e]) for e,n in nodes.items()]);e.to_csv(folder/'state_events.csv',index=False);save(folder/'state_statics.json',statics)
        rel.to_csv(folder/'relocated_train_events.csv',index=False)
        print(f'H {kind} lambda={weight} '+json.dumps(row),flush=True)
        if r>=4 and len(history)>=3:
            changes=[abs(history[i]['objective']/history[i-1]['objective']-1) for i in [-1,-2]]
            if max(changes)<.001 and max(history[i]['max_vp_fraction'] for i in [-1,-2])<.005:break
    r=history[-1]['round']
    final=module.residual(cache,train,nodes,origins,statics)
    held=test.observed_s.to_numpy()-cache.predict_exact(test)-test.station.map(statics).to_numpy()
    cache.close()
    converged=r>=4 and len(history)>=3 and max(abs(history[i]['objective']/history[i-1]['objective']-1) for i in [-1,-2])<.001 and max(history[i]['max_vp_fraction'] for i in [-1,-2])<.005
    np.savez(folder/'vp_final.npz',vp=vp,vp_initial=base,active=mask,xg=GRID.x,yg=GRID.y,zg=GRID.z,inversion_xg=COARSE.x,inversion_yg=COARSE.y,inversion_zg=COARSE.z)
    pd.DataFrame(dict(station=list(statics),static_s=list(statics.values()))).to_csv(folder/'station_statics.csv',index=False)
    q=test[['event_id','station']].copy();q['residual_s']=held;q.to_csv(folder/'holdout_residuals.csv',index=False)
    trainout=train[['event_id','station']].copy();trainout['residual_s']=final;trainout.to_csv(folder/'train_residuals.csv',index=False)
    result=dict(kind=kind,laplace_weight=weight,train=metrics(final),holdout=metrics(held),rounds=r,converged=bool(converged),stop='training_objective_plateau' if converged else 'predefined_maximum_rounds',parameters=nc,reference_station=reference)
    save(folder/'comparison.json',result)
    return result

def verification():
    # Direct depth-basis derivative, and absolute regularization RHS.
    rng=np.random.default_rng(20261011);g=GRID
    small=f.Grid(np.arange(0.,60.,10.),np.arange(0.,70.,10.),np.arange(-10.,60.,10.))
    slow=np.full(small.shape,.16);source=np.array([20.,30.,0.]);nz=len(small.z)-1
    ids=np.repeat(np.maximum(np.arange(len(small.z))-1,0),len(small.x)*len(small.y))
    B=sparse.csr_matrix((np.ones(slow.size),(np.arange(slow.size),ids)),shape=(slow.size,nz));direction=rng.normal(size=nz)*.001
    ii,ww=f.interpolation_weights(small,np.array([[42.,35.,33.]]));rr=np.linalg.norm(np.array([[42.,35.,33.]])-source,axis=1)
    with f.Graph(slow,small,source) as graph:adj=(graph.adjoint_projected(ii,ww*rr[:,None],B)@direction)[0]
    vals=[]
    for sign in [1,-1]:
        with f.Graph(slow+sign*1e-4*(B@direction).reshape(small.shape),small,source) as graph:vals.append(float((graph.u.ravel()[ii]*ww).sum()*rr[0]))
    fd=(vals[0]-vals[1])/(2e-4);error=abs(adj-fd)/max(abs(fd),1e-12)
    assert error<1e-5,error
    # The optimum of ||d-b||² + w²||c+d||² has an exact scalar solution.
    b=2.;c=3.;w=4.;a=np.array([[1.],[w]]);rhs=np.array([b,-w*c]);d=np.linalg.lstsq(a,rhs,rcond=None)[0][0]
    assert abs(d-(b-w*w*c)/(1+w*w))<1e-12
    save(OUT/'verification.json',dict(depth_basis_relative_fd_error=error,absolute_regularization_scalar_exact=True,pass_gate=True))

def protocol():
    p=population();plan=dict(data_sha256=hashlib.sha256((GOUT/'selected_picks.csv').read_bytes()).hexdigest(),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),arms=['fixed_1d','updated_1d','3d_lambda_2','3d_lambda_20','3d_lambda_200'],smooth_values=SMOOTH,max_rounds=MAX_ROUNDS,stopping='two consecutive training objective relative changes <0.001 and max Vp fractional change <0.005; minimum four rounds; never stop on holdout',regularization='absolute departure slowness from reference, not only update; fixed support from first g round; static absolute damping',synthetic='3 percent truth, effective empirical centered errors from main training catalog residuals only; stress test, not picker uncertainty; all three smooth weights; four rounds',mesh='same six preselected stations and 4597 observations at 10/5/2.5 km',seed=20261011,limitations=['Finite round cap must be reported if convergence fails','Main holdout used for reporting sweep, not choosing an independent unbiased winner','Catalog origin/hypocentre remain conditional; Pn/Pg not separately followed'])
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==plan,'Protocol changed after freeze'
    else:save(path,plan)
    return p

def real():
    p=protocol();results=[fit(p,'fixed',0.,OUT/'fixed_1d'),fit(p,'1d',20.,OUT/'updated_1d')]
    for w in SMOOTH:results.append(fit(p,'3d',w,OUT/f'real_smooth_{int(w)}'))
    save(OUT/'real_comparisons.json',results)

def synthetic():
    p=protocol();path=OUT/'synthetic_effective_errors.csv'
    if path.exists():q=pd.read_csv(path,low_memory=False,float_precision='round_trip')
    else:
        # Training-only empirical residuals at continuous catalog coordinates.
        cache=f.FieldCache(initial(),GRID,p[['station','st_x_km','st_y_km','st_elev_km']].drop_duplicates(),CFG['cache_directory'],CFG)
        train=p[p.split.eq('train')].reset_index(drop=True)
        stat=json.loads((GOUT/'real_elevation/control_1d.json').read_text())['statics']
        err=train.observed_s.to_numpy()-cache.predict_exact(train)-train.station.map(stat).fillna(0).to_numpy();cache.close()
        err-=np.median(err);rng=np.random.default_rng(20261011)
        q=pd.read_csv(GOUT/'synthetic_03pct/synthetic_observations.csv',low_memory=False,float_precision='round_trip')
        noise=rng.choice(err,len(q),replace=True)
        # Effective errors already include catalog-origin/systematic scatter:
        # do not additionally inject the old synthetic origins/statics/noise.
        q['effective_error_s']=noise;q['observed_s']=q.true_time_s+noise
        q.to_csv(path,index=False)
        save(OUT/'synthetic_error_calibration.json',dict(source='training catalog-continuous 1D residuals minus fitted station correction, median centered; effective errors, not independent picker errors',training_samples=len(err),sample_rms_s=float(np.sqrt(np.mean(err**2))),sample_median_abs_s=float(np.median(abs(err))),synthetic_rms_s=float(np.sqrt(np.mean(noise**2))),seed=20261011,explicit_origin_static_noise_injection=False))
    results=[fit(q,'1d',20.,OUT/'synthetic_updated_1d',max_rounds=4)]
    for w in SMOOTH:
        folder=OUT/f'synthetic_smooth_{int(w)}';r=fit(q,'3d',w,folder,max_rounds=4)
        from controles_factorizados_20261011g import truth
        target=(truth(GRID,.03)/initial()-1).ravel();got=(np.load(folder/'vp_final.npz')['vp']/initial()-1).ravel()
        mask=np.load(GOUT/'real_elevation/model_round_1.npz')['active'];m=f.mapping(GRID,COARSE);cnt=np.bincount(m,minlength=mask.size)
        reduce=lambda v:np.bincount(m,weights=v,minlength=mask.size)/cnt
        a=reduce(target)[mask.ravel()];b=reduce(got)[mask.ravel()]
        r.update(correlation=float(np.corrcoef(a,b)[0,1]),relative_l2_error=float(np.linalg.norm(b-a)/np.linalg.norm(a)))
        save(folder/'recovery_metrics.json',r);results.append(r)
    save(OUT/'synthetic_comparisons.json',results)

def refined_times(vp,grid,p):
    result=np.empty(len(p));h=grid.spacing
    interpolation=RegularGridInterpolator((grid.z,grid.y,grid.x),1/vp)
    for sta,ids in p.groupby('station',sort=False).indices.items():
        q=p.iloc[ids];source=np.array([q.st_x_km.iloc[0],q.st_y_km.iloc[0],-q.st_elev_km.iloc[0]])
        delta=source-grid.xyz(grid.nodes(source));mesh=f.Grid(*[np.r_[a[0]-h,a,a[-1]+h]+d for a,d in zip([grid.x,grid.y,grid.z],delta)])
        yy,xx=np.meshgrid(np.clip(mesh.y,grid.y[0],grid.y[-1]),np.clip(mesh.x,grid.x[0],grid.x[-1]),indexing='ij');slow=np.empty(mesh.shape)
        for k,z in enumerate(mesh.z):slow[k]=interpolation(np.column_stack([np.full(xx.size,np.clip(z,grid.z[0],grid.z[-1])),yy.ravel(),xx.ravel()])).reshape(xx.shape)
        xyz=q[['source_x_km','source_y_km','ev_prof_km']].to_numpy()
        with f.Graph(slow,mesh,source,2) as graph:result[ids]=RegularGridInterpolator((mesh.z,mesh.y,mesh.x),graph.u)(xyz[:,::-1])*np.linalg.norm(xyz-source,axis=1)
        print(f'H mesh {h} {sta}',flush=True)
    return result

def mesh():
    p=protocol();ids=json.loads((GOUT/'holdout_refinement_comparison.json').read_text())['station_ids'];q=p[p.station.isin(ids)].reset_index(drop=True)
    for kind in ['updated_1d']+[f'real_smooth_{int(w)}' for w in SMOOTH]:
        folder=OUT/kind;model=np.load(folder/'vp_final.npz');st=pd.read_csv(folder/'station_statics.csv');stat=dict(zip(st.station,st.static_s));runs=[]
        for h in [10.,5.,2.5]:
            path=folder/f'mesh_{h:g}.csv'
            if not path.exists():
                grid=f.Grid(np.arange(GRID.x[0],GRID.x[-1]+h/2,h),np.arange(GRID.y[0],GRID.y[-1]+h/2,h),np.arange(-10,180+h/2,h))
                z,y,x=np.meshgrid(grid.z,grid.y,grid.x,indexing='ij');slow=RegularGridInterpolator((GRID.z,GRID.y,GRID.x),1/model['vp'])(np.column_stack([z.ravel(),y.ravel(),x.ravel()]));vp=(1/slow).reshape(grid.shape)
                result=q[['event_id','station','split','observed_s']].copy();result['time_s']=refined_times(vp,grid,q);result['residual_s']=result.observed_s-result.time_s-result.station.map(stat);result.to_csv(path,index=False)
            runs.append(pd.read_csv(path))
        differences=[]
        for a,b in zip(runs[:-1],runs[1:]):
            d=a.time_s.to_numpy()-b.time_s.to_numpy();differences.append(dict(rms_s=float(np.sqrt(np.mean(d*d))),p95_s=float(np.quantile(abs(d),.95))))
        report=dict(differences=differences,pass_gate=differences[-1]['rms_s']<=.1 and differences[-1]['p95_s']<=.25,held=[metrics(a[a.split.eq('holdout')].residual_s.to_numpy()) for a in runs])
        save(folder/'mesh_convergence.json',report)

def all():
    verification();real();synthetic();mesh()
    save(OUT/'completed.json',dict(status='completed',time_utc=pd.Timestamp.now(tz='UTC').isoformat()))

if __name__=='__main__':globals()[sys.argv[1]]()
