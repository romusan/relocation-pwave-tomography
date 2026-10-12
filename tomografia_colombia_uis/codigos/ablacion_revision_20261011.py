"""Fixed-catalog prediction audit; no held-out fitting or changes to verified runs."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy import sparse
from scipy.sparse.linalg import lsmr
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'codigos'))
from regional.geometry import Grid,candidate_nodes
from regional.forward import FieldCache
from nazca_campana_20261010 import projection
RUN=ROOT/'resultados/tomografia_1d_adjunto_consistente_2026-10-10'
OUT=ROOT/'resultados/revision_cientifica_20261011';OUT.mkdir(exist_ok=True)

def metrics(r):
    a=np.abs(r);q=np.quantile(a,.95)
    return dict(rms_s=float(np.sqrt(np.mean(r*r))),median_abs_s=float(np.median(a)),p90_abs_s=float(np.quantile(a,.9)),trimmed95_rms_s=float(np.sqrt(np.mean(r[a<=q]**2))))

def fit_statics(p,res,cfg):
    # Independent 1D control: train catalog coordinates fixed; train origins
    # nuisance parameters, unpenalized; same reference and station damping.
    eids=sorted(p.event_id.unique());stas=sorted(p.station.unique());free=stas[1:]
    ep={e:i for i,e in enumerate(eids)};sp={s:i for i,s in enumerate(free)}
    rows=np.arange(len(p));E=sparse.csr_matrix((np.ones(len(p)),(rows,p.event_id.map(ep))),shape=(len(p),len(eids)))
    ix=np.flatnonzero(p.station.isin(free));S=sparse.csr_matrix((np.ones(len(ix)),(ix,p.iloc[ix].station.map(sp))),shape=(len(p),len(free)))
    A=sparse.vstack([sparse.hstack([E,S]),sparse.hstack([sparse.csr_matrix((len(free),len(eids))),cfg['station_damping']*sparse.eye(len(free))])],format='csr')
    sol=lsmr(A,np.r_[res,np.zeros(len(free))],atol=1e-10,btol=1e-10,maxiter=5000)
    assert sol[1] in (1,2,4,5),sol[1]
    return {stas[0]:0.,**dict(zip(free,sol[0][len(eids):]))},dict(istop=int(sol[1]),iterations=int(sol[2]),reference_station=stas[0],train_events=len(eids),free_statics=len(free))

def main():
    cfg=json.loads((RUN/'config.json').read_text());domain=json.loads((RUN/'domain.json').read_text());pr=projection(domain)
    p=pd.read_csv(RUN/'predictions.csv',low_memory=False).reset_index(drop=True)
    train=p.split.eq('train').to_numpy();held=~train;static=p.station_static_s.to_numpy()
    f=np.load(RUN/'vp_final.npz');grid=Grid(f['xg'],f['yg'],f['zg']);axes=(grid.z,grid.y,grid.x)
    xyz=p[['source_x_km','source_y_km','ev_prof_km']].to_numpy();q=xyz[:,::-1]
    assert np.all(q>=np.array([a[0] for a in axes])) and np.all(q<=np.array([a[-1] for a in axes]))
    stations=p[['station','st_x_km','st_y_km']].drop_duplicates();nodes={}
    for eid,g in p.groupby('event_id',sort=False):
        pos=xyz[g.index[0]];near=candidate_nodes(grid,grid.nodes(pos),grid.spacing,domain,pr)
        nodes[eid]=tuple(near[np.argmin(np.linalg.norm(grid.xyz(near)-pos,axis=1))])
    predictions={};cache_info={}
    for model,key in [('1d','vp_initial'),('3d','vp')]:
        cache=FieldCache(f[key],grid,stations,cfg['cache_directory'],cfg)
        node=np.empty(len(p));cont=np.empty(len(p))
        for i,(sta,ix) in enumerate(p.groupby('station',sort=False).indices.items()):
            field=cache.get(sta);ns=np.array([nodes[e] for e in p.iloc[ix].event_id]);node[ix]=field[tuple(ns.T)]
            cont[ix]=RegularGridInterpolator(axes,field)(q[ix])
            if i%50==0:print(model,'station',i,'/',len(stations),flush=True)
        predictions[model]=(node,cont);cache_info[model]=cache.stats();cache.close()
    assert np.allclose(predictions['3d'][0][held],p.predicted_s.to_numpy()[held],atol=1e-10)
    obs=p.observed_s.to_numpy();rows=[];residuals=p.loc[held,['raw_row','event_id','station','ev_prof_km','dist_epi_km']].copy();fits={}
    for coord,j in [('node',0),('continuous',1)]:
        t0=predictions['1d'][j];t1=predictions['3d'][j]
        independent,info=fit_statics(p.loc[train].reset_index(drop=True),(obs-t0)[train],cfg)
        fitted=p.station.map(independent).fillna(0.).to_numpy();fits[coord]=info
        pd.DataFrame(dict(station=list(independent),static_s=list(independent.values()))).to_csv(OUT/f'1d_station_statics_{coord}.csv',index=False)
        cases={'1d_no_statics':obs-t0,'1d_final3d_statics':obs-t0-static,'3d_no_statics':obs-t1,'3d_final3d_statics':obs-t1-static,'1d_independent_statics':obs-t0-fitted}
        for name,r in cases.items():
            rows.append(dict(coordinates=coord,model=name,events=int(p.loc[held].event_id.nunique()),picks=int(held.sum()),**metrics(r[held])))
            residuals[f'{coord}_{name}']=r[held]
    pd.DataFrame(rows).to_csv(OUT/'holdout_ablation_metrics.csv',index=False);residuals.to_csv(OUT/'holdout_ablation_residuals.csv',index=False)
    # Paired bootstrap resamples events, preserving all their picks.
    group=residuals.groupby('event_id');rng=np.random.default_rng(20261011);ci={}
    for coord in ['node','continuous']:
        pairs=[('1d_no_statics','3d_final3d_statics'),('1d_final3d_statics','3d_final3d_statics'),('1d_independent_statics','3d_final3d_statics')]
        for base,end in pairs:
            sums=group[[coord+'_'+base,coord+'_'+end]].agg(lambda s:np.sum(s*s)).to_numpy();counts=group.size().to_numpy();vals=[]
            for _ in range(3000):
                ix=rng.integers(0,len(counts),len(counts));r=np.sqrt(sums[ix].sum(axis=0)/counts[ix].sum());vals.append(r[1]-r[0])
            ci[coord+'_'+base+'_vs_'+end]=np.quantile(vals,[.025,.975]).tolist()
    report=dict(metrics=rows,bootstrap_delta_rms_95_ci_s=ci,bootstrap_replicates=3000,cache=cache_info,independent_static_fit=fits,
        notes=['No holdout origins, hypocenters, statics or parameters fitted.','Continuous means trilinear interpolation at exact catalog coordinates; stations remain snapped at z=0.','Final3d statics are plug-in attribution controls, learned jointly with the 3D model.','Independent 1D statics are fitted only to train picks at fixed catalog locations with free train origin shifts, damping 2 and reference CM.ACH1; not a full 1D relocation inversion.','This audit does not establish grid convergence or nonlinear resolution.'],
        source_sha256={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [Path(__file__),RUN/'vp_final.npz',RUN/'predictions.csv',RUN/'config.json',RUN/'domain.json']})
    (OUT/'ablacion.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(rows,indent=2),flush=True)
if __name__=='__main__':main()
