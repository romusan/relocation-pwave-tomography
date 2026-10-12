"""Paired holdout comparison and measured final curved-ray illumination."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from regional.geometry import Projection,Grid,candidate_nodes
from regional.forward import FieldCache
from regional.rays import kernels
from regional.inputs import sha256
OLD=ROOT/'resultados/migracion_caribe_1d_carma_2026-10-10'
NEW=ROOT/'resultados/migracion_eje_caribe_1d_2026-10-10'
def rms(x):return float(np.sqrt(np.mean(np.asarray(x)**2)))
def coverage(run):
    domain=json.loads((run/'domain.json').read_text(encoding='utf-8'));pr=Projection(domain);cfg=json.loads((run/'config.json').read_text(encoding='utf-8'))
    picks=pd.read_csv(run/'associated_picks_selected.csv',low_memory=False);e=pd.read_csv(run/'events_split.csv');rel=pd.read_csv(run/'relocated_train_events.csv')
    p=picks[picks.fase.eq('P')&picks.event_id.isin(e[e.split.eq('train')].event_id)].reset_index(drop=True)
    with np.load(run/'vp_final.npz') as a:
        direct=Grid(a['xg'],a['yg'],a['zg']);coarse=Grid(a['inversion_xg'],a['inversion_yg'],a['inversion_zg']);vp=a['vp'].copy();active=a['active'].ravel().copy()
    cache=FieldCache(vp,direct,p[['station','st_x_km','st_y_km']].drop_duplicates(),cfg['cache_directory'],cfg)
    rr=rel[rel.status.eq('relocated')].set_index('event_id');nodes={}
    for eid,g in p.groupby('event_id',sort=False):
        if eid in rr.index:
            r=rr.loc[eid];nodes[eid]=tuple(direct.nodes([r.x_km,r.y_km,r.depth_km]))
        else:
            xyz=np.array([g.source_x_km.iloc[0],g.source_y_km.iloc[0],g.ev_prof_km.iloc[0]])
            nearby=candidate_nodes(direct,direct.nodes(xyz),direct.spacing,domain,pr);nodes[eid]=tuple(nearby[np.argmin(np.linalg.norm(direct.xyz(nearby)-xyz,axis=1))])
    G,ray_info=kernels(cache,p,nodes,coarse,cfg['ray_backend']);B=G.copy();B.data=np.ones_like(B.data)
    hits=np.asarray(B.sum(axis=0)).ravel().astype(int)
    # Global source-to-receiver horizontal orientation; not local ray incidence or resolution.
    source=np.array([direct.xyz(nodes[eid]) for eid in p.event_id]);angle=np.degrees(np.arctan2(p.st_y_km.to_numpy()-source[:,1],p.st_x_km.to_numpy()-source[:,0]))%180
    sector=(angle//30).astype(int);direction=np.array([np.asarray(B[sector==i].sum(axis=0)).ravel() for i in range(6)])
    n_sectors=(direction>0).sum(axis=0)
    xyz=coarse.xyz(np.indices(coarse.shape).reshape(3,-1).T);la,lo=pr.latlon(xyz[:,0],xyz[:,1])
    np.savez_compressed(run/'iluminacion_rayos_final.npz',hits=hits.reshape(coarse.shape),orientation_sectors=n_sectors.reshape(coarse.shape),sector_hits=direction.reshape((6,*coarse.shape)),xg=coarse.x,yg=coarse.y,zg=coarse.z)
    return dict(xyz=xyz,lat=la,lon=lo,hits=hits,active=active,sectors=n_sectors,events=e,picks=picks,ray_info=ray_info)
def main():
    a=pd.read_csv(OLD/'predictions.csv',low_memory=False);b=pd.read_csv(NEW/'predictions.csv',low_memory=False);a=a[a.split.eq('holdout')];h=b[b.split.eq('holdout')]
    pair=a.merge(h,on=['event_id','station','fase','raw_row'],suffixes=('_before','_after'),validate='one_to_one');assert len(pair)==len(a)
    for col in ['observed_s','ev_lat','ev_lon','ev_prof_km']:
        np.testing.assert_allclose(pair[col+'_before'],pair[col+'_after'],atol=1e-10,rtol=0)
    np.testing.assert_array_equal(pair.origin_shift_s_after,np.zeros(len(pair)))
    with np.load(OLD/'vp_final.npz') as a1,np.load(NEW/'vp_final.npz') as b1:np.testing.assert_array_equal(a1['vp_initial'],b1['vp_initial'])
    pair.to_csv(NEW/'common_holdout_comparison.csv',index=False)
    group=pair.assign(a=pair.residual_s_before**2,b=pair.residual_s_after**2).groupby('event_id').agg(a=('a','sum'),b=('b','sum'),n=('a','size')).to_numpy()
    rng=np.random.default_rng(20261009);draw=rng.integers(0,len(group),size=(3000,len(group)));totals=group[draw].sum(axis=1)
    delta=np.sqrt(totals[:,1]/totals[:,2])-np.sqrt(totals[:,0]/totals[:,2])
    result=dict(common_holdout_events=pair.event_id.nunique(),common_holdout_P=len(pair),common_RMS_before_s=rms(pair.residual_s_before),common_RMS_after_s=rms(pair.residual_s_after),delta_bootstrap_95percent_ci_s=np.quantile(delta,[.025,.975]).tolist(),all_new_holdout_RMS_s=rms(h.residual_s),initial_original_1d_exact=True)
    print('Computing actual curved-ray coverage before/after',flush=True)
    cov=[coverage(OLD),coverage(NEW)];zones={'Eje_Cafetero':(4,6.6,-77,-74.5),'Caribe':(8,13,-79,-71),'Transicion_7_9N':(7,9,-79,-71)};counts=[]
    for label,c in zip(['before','after'],cov):
        row=dict(stage=label,zones={})
        for zone,(la0,la1,lo0,lo1) in zones.items():
            cells=(c['lat']>=la0)&(c['lat']<=la1)&(c['lon']>=lo0)&(c['lon']<=lo1);deep=cells&(c['xyz'][:,2]>=40)
            ev=c['events'];ev=ev[ev.lat.between(la0,la1)&ev.lon.between(lo0,lo1)];pp=c['picks'];pp=pp[pp.event_id.isin(ev.event_id)&pp.fase.eq('P')]
            row['zones'][zone]=dict(events=len(ev),events_depth_ge40=int(ev.depth_km.ge(40).sum()),P=len(pp),station_epochs=pp.station.nunique(),active_cells=int((cells&c['active']).sum()),active_deep_cells=int((deep&c['active']).sum()),cells_hits20_sectors3_deep=int((deep&(c['hits']>=20)&(c['sectors']>=3)).sum()))
        counts.append(row)
    previous=cov[0];current=cov[1];centers=previous['xyz'];distance=cKDTree(centers[previous['active']]).query(centers)[0]
    target=(~previous['active'])&(distance<=60)&(centers[:,2]>=40)&(centers[:,2]<=160)
    target&=((previous['lat']>=4)&(previous['lat']<=6.6)&(previous['lon']>=-77)&(previous['lon']<=-74.5))|((previous['lat']>=8)&(previous['lat']<=13)&(previous['lon']>=-79)&(previous['lon']<=-71))
    gained=target&current['active'];lost=previous['active']&~current['active']
    target_zones={}
    for zone,(la0,la1,lo0,lo1) in zones.items():
        z=(previous['lat']>=la0)&(previous['lat']<=la1)&(previous['lon']>=lo0)&(previous['lon']<=lo1)
        target_zones[zone]=dict(previous_inactive_targets=int((z&target).sum()),targets_now_active=int((z&gained).sum()),targets_now_hits20_sectors3=int((z&target&(current['hits']>=20)&(current['sectors']>=3)).sum()),previous_active_cells_lost=int((z&lost).sum()))
    result.update(coverage=counts,previous_inactive_targets=int(target.sum()),targets_now_active=int(gained.sum()),targets_now_hits20_sectors3=int((target&(current['hits']>=20)&(current['sectors']>=3)).sum()),previous_active_cells_lost=int(lost.sum()),limits='Ray counts and global orientation bins measure acquisition, not resolution or geological continuity; losses reported because selection changes ray paths.')
    result['target_zones']=target_zones
    result['source_sha256']={str(p):sha256(p) for p in [Path(__file__),OLD/'vp_final.npz',NEW/'vp_final.npz',OLD/'predictions.csv',NEW/'predictions.csv',OLD/'iluminacion_rayos_final.npz',NEW/'iluminacion_rayos_final.npz']}
    (NEW/'comparacion_iluminacion.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(13,12),layout='constrained')
    for i,(label,c) in enumerate(zip(['Antes','Después'],cov)):
        e=c['events'];p=c['picks'];sta=p[p.fase.eq('P')][['station','st_lat_current','st_lon_current']].drop_duplicates()
        im=axs[0,i].scatter(e.lon,e.lat,c=e.depth_km,s=8,cmap='viridis',vmin=0,vmax=180,alpha=.6);axs[0,i].scatter(sta.st_lon_current,sta.st_lat_current,marker='^',c='red',s=10);axs[0,i].set_title(label+': eventos y estaciones con P')
        deep=(c['xyz'][:,2]>=40)&(c['xyz'][:,2]<=160);cell=c['active']&deep
        axs[1,i].scatter(c['lon'][cell],c['lat'][cell],c=c['xyz'][cell,2],s=12,marker='s',cmap='viridis',vmin=0,vmax=180,alpha=.5)
        hole=target&~c['active'];axs[1,i].scatter(c['lon'][hole],c['lat'][hole],s=13,facecolors='none',edgecolors='#777777',linewidths=.3,label='Objetivos aún inactivos')
        if i==1:axs[1,i].scatter(c['lon'][gained],c['lat'][gained],s=18,facecolors='none',edgecolors='red',linewidths=.7,label='Objetivos ahora activos')
        axs[1,i].set_title(label+': cobertura proyectada, 40–160 km');axs[1,i].legend(fontsize=8)
    for ax in axs.ravel():ax.set(xlim=(-79,-71),ylim=(2,13),xlabel='Longitud WGS84',ylabel='Latitud WGS84');ax.set_aspect('equal');ax.grid(alpha=.2)
    fig.colorbar(im,ax=axs.ravel().tolist(),label='Profundidad (km)',shrink=.6);fig.suptitle('Mismo inicio 1D — cobertura proyectada no equivale a resolución');fig.savefig(NEW/'comparacion_iluminacion.png',dpi=160);plt.close(fig)
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
