"""Same frozen mesh experiment, bounded-memory forward-only sampling.

No sparse parameter projection is needed for forward refinement. Slowness is
sampled in z slices, giving the same trilinear model as the campaign cache.
"""
from campana_factorizada_cotas_20261011g import *
from regional.factored import Graph
def main():
    p=protocol();chosen=[]
    for lo,hi in [(-90,4.5),(4.5,8),(8,90)]:chosen+=p[(p.ev_lat>=lo)&(p.ev_lat<hi)].station.value_counts().head(2).index.tolist()
    chosen=list(dict.fromkeys(chosen));p=p[p.station.isin(chosen)].reset_index(drop=True);reports=[]
    # Interpolating fixed 10-km final model at refined nodes, then translated
    # nodes, matches the original refinement program's two model operations.
    basegrid=Grid(F['xg'],F['yg'],F['zg'])
    for model in ['homogeneous','initial_1d','previous_3d']:
        for h in [10.,5.,2.5]:
            path=OUT/f'mesh_{model}_{h:g}.csv'
            if path.exists():continue
            grid=Grid(np.arange(GRID.x[0],GRID.x[-1]+h/2,h),np.arange(GRID.y[0],GRID.y[-1]+h/2,h),np.arange(-10,180+h/2,h));out=p[['event_id','station']].copy();out['predicted_s']=np.nan
            for station,ids in p.groupby('station',sort=False).indices.items():
                qs=p.iloc[ids];source=np.array([qs.st_x_km.iloc[0],qs.st_y_km.iloc[0],-qs.st_elev_km.iloc[0]])
                delta=source-grid.xyz(grid.nodes(source));axes=[np.r_[a[0]-h,a,a[-1]+h]+d for a,d in zip([grid.x,grid.y,grid.z],delta)];mesh=Grid(*axes)
                if model=='homogeneous':slow=np.full(mesh.shape,1/6)
                else:
                    # Fixed-grid values, followed by station mesh interpolation.
                    fixed=np.empty(grid.shape)
                    yy,xx=np.meshgrid(grid.y,grid.x,indexing='ij')
                    interp=RegularGridInterpolator((basegrid.z,basegrid.y,basegrid.x),1/F['vp']) if model=='previous_3d' else None
                    for k,z in enumerate(grid.z):
                        if model=='initial_1d':fixed[k]=1/np.interp(z,*PROFILE)
                        else:fixed[k]=interp(np.column_stack([np.full(xx.size,max(z,0)),yy.ravel(),xx.ravel()])).reshape(xx.shape)
                    interp=RegularGridInterpolator((grid.z,grid.y,grid.x),fixed);slow=np.empty(mesh.shape);yy,xx=np.meshgrid(np.clip(mesh.y,grid.y[0],grid.y[-1]),np.clip(mesh.x,grid.x[0],grid.x[-1]),indexing='ij')
                    for k,z in enumerate(mesh.z):slow[k]=interp(np.column_stack([np.full(xx.size,np.clip(z,grid.z[0],grid.z[-1])),yy.ravel(),xx.ravel()])).reshape(xx.shape)
                    del fixed
                xyz=qs[['source_x_km','source_y_km','ev_prof_km']].to_numpy()
                with Graph(slow,mesh,source,CFG['factored_order']) as g:
                    out.loc[ids,'predicted_s']=RegularGridInterpolator((mesh.z,mesh.y,mesh.x),g.u)(xyz[:,::-1])*np.linalg.norm(xyz-source,axis=1)
                if model=='homogeneous':out.loc[ids,'analytic_s']=np.linalg.norm(xyz-source,axis=1)/6
                print(f'MESH {model} {h} {station} complete',flush=True)
            out.to_csv(path,index=False)
        runs=[pd.read_csv(OUT/f'mesh_{model}_{h:g}.csv') for h in [10,5,2.5]];dif=[]
        for a,b in zip(runs[:-1],runs[1:]):
            d=a.predicted_s.to_numpy()-b.predicted_s.to_numpy();dif.append(dict(rms_s=float(np.sqrt(np.mean(d*d))),p95_s=float(np.quantile(abs(d),.95))))
        entry=dict(model=model,picks=len(p),station_ids=chosen,differences=dif,pass_gate=dif[-1]['rms_s']<=.1 and dif[-1]['p95_s']<=.25)
        if model=='homogeneous':entry['max_analytic_error_s']=max(float(abs(q.predicted_s-q.analytic_s).max()) for q in runs)
        reports.append(entry);save(OUT/'mesh_convergence.json',reports)
    save(OUT/'mesh_execution.json',dict(script_sha256=sha(__file__),implementation='same trilinear model interpolation, sampled by slices to avoid sparse projection allocation; no change to forward operator or acceptance gates'))
if __name__=='__main__':main()
