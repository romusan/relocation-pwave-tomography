"""Publication figures and measured metrics for the ONE original-1D experiment."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
from scipy.interpolate import RegularGridInterpolator
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'codigos'))
from regional.geometry import grids,candidate_nodes
from regional.forward import FieldCache
from regional.inputs import sha256
from nazca_campana_20261010 import projection
from agregar_mapa_superficie_norte import boundaries,lines
from shapely.geometry import box
from shapely.ops import unary_union
RUN=ROOT/'resultados/migracion_nazca_cauca_valle_eje_1d_2026-10-10'
GEO=ROOT/'resultados/placas_nazca_caribe_diagonal_2026-10-10'
OLDVIEW=ROOT/'resultados/nazca_cauca_valle_eje_1d_vista_2026-10-10'
OUT=ROOT/'articulo_boletin_geologia_uis';(OUT/'figuras').mkdir(parents=True,exist_ok=True)

def main():
    cfg=json.loads((RUN/'config.json').read_text());domain=json.loads((RUN/'domain.json').read_text());pr=projection(domain)
    summary=json.loads((RUN/'summary.json').read_text());e=pd.read_csv(RUN/'relocated_train_events.csv');p=pd.read_csv(RUN/'associated_picks_selected.csv',low_memory=False);events=pd.read_csv(RUN/'events_split.csv')
    pred=pd.read_csv(RUN/'predictions.csv',low_memory=False)
    with np.load(RUN/'vp_final.npz') as m:
        bg=m['vp_initial'].copy();vp=m['vp'].copy();x,y,z=[m[k].copy() for k in ['xg','yg','zg']];ix,iy,iz=[m[k].copy() for k in ['inversion_xg','inversion_yg','inversion_zg']];active=m['active'].copy().astype(bool)
    initial=RegularGridInterpolator((z,y,x),bg);final=RegularGridInterpolator((z,y,x),vp);cover=RegularGridInterpolator((iz,iy,ix),active.astype(float),method='nearest',bounds_error=False,fill_value=0)
    held=pred[pred.split.eq('holdout')].copy();grid,coarse=grids(domain,pr,10,20)
    nodes={}
    for eid,g in held.groupby('event_id',sort=False):
        xyz=np.array([g.source_x_km.iloc[0],g.source_y_km.iloc[0],g.ev_prof_km.iloc[0]]);rounded=grid.nodes(xyz);nearby=candidate_nodes(grid,rounded,grid.spacing,domain,pr)
        nodes[eid]=tuple(nearby[np.argmin(np.linalg.norm(grid.xyz(nearby)-xyz,axis=1))])
    stations=p[p.fase.eq('P')][['station','st_x_km','st_y_km']].drop_duplicates()
    cache=FieldCache(bg,grid,stations,cfg['cache_directory'],cfg);held['residual_initial_s']=held.observed_s.to_numpy()-cache.predict(held,nodes)
    baseline=float(np.sqrt(np.mean(held.residual_initial_s**2)));np.testing.assert_allclose(baseline,summary['baseline']['holdout_rms_fixed_catalog_origin_s'],atol=1e-10)
    g=held.assign(a=held.residual_initial_s**2,b=held.residual_s**2).groupby('event_id').agg(a=('a','sum'),b=('b','sum'),n=('a','size')).to_numpy()
    rng=np.random.default_rng(20261009);draw=rng.integers(0,len(g),size=(3000,len(g)));tot=g[draw].sum(axis=1);delta=np.sqrt(tot[:,1]/tot[:,2])-np.sqrt(tot[:,0]/tot[:,2]);ci=np.quantile(delta,[.025,.975]).tolist()
    held.to_csv(OUT/'fuentes/reserva_inicial_final.csv',index=False)
    path,meta,features=boundaries();polys=[a[1].intersection(box(-79,.8,-71,13)) for a in features.values()]
    borders=[(unary_union(polys).boundary,'Colombia','#17232c',.9),(unary_union([a.boundary for a in polys]),'Departamentos','#949494',.35)]
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection,Line3DCollection
    plt.rcParams.update({'font.size':11,'font.family':'DejaVu Sans','axes.labelsize':11})
    def setup():
        fig=plt.figure(figsize=(10,8));ax=fig.add_subplot(projection='3d');ax.set_position([.00,.13,.97,.85]);ax.set_proj_type('ortho')
        for geo,name,color,width in borders:ax.add_collection3d(Line3DCollection([np.c_[q[:,0],q[:,1],np.zeros(len(q))] for q in lines(geo)],colors=color,linewidths=width))
        ax.scatter(e.lon,e.lat,e.depth_km,s=3,c='#212121',alpha=.35,depthshade=False)
        ax.set(xlabel='Longitud (°)',ylabel='Latitud (° N)',zlabel='Profundidad (km)',xlim=(-79,-71),ylim=(.8,13),zlim=(180,0),zticks=[0,60,120,180]);ax.set_box_aspect([.655,1,.399]);ax.view_init(elev=35.264,azim=-55)
        return fig,ax
    with np.load(OLDVIEW/'fragmentos_finales.npz') as f:xyz=f['xyz'];faces=f['triangles']
    local=cover(xyz[:,::-1])>.5;faces=faces[local[faces].all(axis=1)];la,lo=pr.latlon(xyz[:,0],xyz[:,1]);points=np.c_[lo,la,xyz[:,2]]
    assert len(faces)>0
    fig,ax=setup();ax.add_collection3d(Poly3DCollection(points[faces],facecolors='#7a8c91',edgecolors='#54676c',linewidths=.15,alpha=.55))
    fig.legend(handles=[Patch(facecolor='#7a8c91',label='Fragmentos locales con cobertura Vp activa'),Line2D([],[],marker='.',color='black',linestyle='None',label='Sismos SGC relocalizados')],loc='lower center',ncol=1,frameon=False,bbox_to_anchor=(.5,.025))
    fig.savefig(OUT/'figuras/Figura_1_fragmentos_SGC.png',dpi=300);plt.close(fig)
    fig,ax=setup()
    for name,color in [('nazca','#7926b3'),('caribe','#ed831b')]:
        with np.load(GEO/f'{name}_superficie.npz') as sh:
            mid=sh['middle'];f=sh['faces'];points=np.c_[sh['lon'],sh['lat'],mid[:,2]];support=sh['supported']
        for ff,opacity in [(f[~support[f].all(axis=1)],.32),(f[support[f].all(axis=1)],.8)]:
            ax.add_collection3d(Poly3DCollection(points[ff],facecolors=color,edgecolors='none',alpha=opacity))
    fig.legend(handles=[Patch(facecolor='#7926b3',label='Nazca candidata'),Patch(facecolor='#ed831b',label='Caribe candidata'),Line2D([],[],marker='.',color='black',linestyle='None',label='Sismos SGC relocalizados')],loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.025))
    fig.savefig(OUT/'figuras/Figura_2_placas_SGC.png',dpi=300);plt.close(fig)
    # Side-by-side initial and final sections, no Vp color cloud in 3D figures.
    fig,axs=plt.subplots(2,3,figsize=(12,8),layout='constrained')
    latgrid=np.arange(.8,13.0001,.18);depthgrid=np.arange(0,180.001,20);lt,zt=np.meshgrid(latgrid,depthgrid)
    for j,longitude in enumerate([-77,-76,-74]):
        px,py=pr.xy(lt.ravel(),np.full(lt.size,longitude));query=np.c_[zt.ravel(),py,px]
        for i,interp in enumerate([initial,final]):
            values=interp(query).reshape(lt.shape)
            if i:values=np.ma.array(values,mask=cover(query).reshape(lt.shape)<.5)
            im=axs[i,j].pcolormesh(lt,zt,values,vmin=5.3,vmax=8.8,cmap='viridis',shading='nearest',rasterized=True)
            if i:
                q=e[e.lon.between(longitude-.25,longitude+.25)];axs[i,j].scatter(q.lat,q.depth_km,c='black',s=3,alpha=.4)
            axs[i,j].set(xlim=(.8,13),ylim=(180,0),xlabel='Latitud (° N)',ylabel='Profundidad (km)',title=('Inicial 1D' if not i else 'Final 3D')+f'   {longitude}°',xticks=[2,4,6,8,10,12]);axs[i,j].grid(alpha=.15)
    fig.colorbar(im,ax=axs.ravel().tolist(),label='Vp (km/s)',shrink=.8)
    fig.savefig(OUT/'figuras/Figura_3_cortes_inicial_final.png',dpi=300);plt.close(fig)
    # Quantify final supported cells rather than classifying their colors as slabs.
    cz,cy,cx=np.meshgrid(iz,iy,ix,indexing='ij');q=np.c_[cz[active],cy[active],cx[active]];v=final(q);dv=100*(v/initial(q)-1)
    cat=pd.read_csv(RUN/'catalogo_visualizacion_ampliado.csv');datecols=[c for c in cat if 'time' in c.lower() or 'fecha' in c.lower() or 'origin' in c.lower()]
    dates={c:[str(cat[c].min()),str(cat[c].max())] for c in datecols if cat[c].notna().any()}
    result=dict(selected_events=len(events),train_events=len(e),holdout_events=held.event_id.nunique(),P=len(pred),train_P=int(pred.split.eq('train').sum()),holdout_P=len(held),S_QC=int(p.fase.eq('S').sum()),station_epochs=len(stations),initial_holdout_RMS_s=baseline,final_holdout_RMS_s=summary['history'][-1]['holdout_rms_fixed_catalog_origin_s'],holdout_delta_ci95_s=ci,initial_train_RMS_s=summary['baseline']['train_rms_fixed_catalog_s'],final_train_RMS_s=summary['history'][-1]['train_rms_after_update_s'],active_cells=int(active.sum()),active_cells_ge40=int((cz[active]>=40).sum()),final_active_vp_range_km_s=[float(v.min()),float(v.max())],final_active_dvp_percentiles=np.quantile(dv,[0,.05,.5,.95,1]).tolist(),fragment_triangles=len(faces),catalog_date_ranges=dates,event_catalog_extents={c:[float(events[c].min()),float(events[c].max())] for c in ['lat','lon','depth_km']},relocated_extents={c:[float(e[c].min()),float(e[c].max())] for c in ['lat','lon','depth_km']},linear_resolution=summary['linear_resolution'],final_ray_diagnostic=summary['final_ray_diagnostic'],station_static_reference=summary['station_static_reference'],static_range_s=[float(pred.station_static_s.min()),float(pred.station_static_s.max())],history=[{k:r[k] for k in ['round','train_rms_before_update_s','train_rms_after_update_s','holdout_rms_fixed_catalog_origin_s','active_cells']} for r in summary['history']],projection_audit=summary['projection_audit'],geometry=json.loads((GEO/'metodologia.json').read_text())['surfaces'])
    result['input_sha256']={str(f):sha256(f) for f in [Path(__file__),RUN/'vp_final.npz',RUN/'predictions.csv',RUN/'relocated_train_events.csv',RUN/'associated_picks_selected.csv',RUN/'events_split.csv',RUN/'summary.json',OLDVIEW/'fragmentos_finales.npz',GEO/'nazca_superficie.npz',GEO/'caribe_superficie.npz',path,meta]}
    (OUT/'fuentes/metricas_articulo.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k not in ['input_sha256','projection_audit']},indent=2),flush=True)
if __name__=='__main__':main()
