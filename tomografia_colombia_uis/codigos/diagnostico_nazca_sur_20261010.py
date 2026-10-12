"""Separate domain effect from data effect with paired holdouts and bent rays."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from regional.inputs import sha256
from nazca_campana_20261010 import projection,departments
import diagnostico_eje_caribe_1d_20261010 as existing
# Adapt the existing ray diagnostic to the explicitly fixed campaign projection.
# The inherited module's file remains unchanged and its SHA is recorded below.
existing.Projection=projection
OLD=ROOT/'resultados/migracion_eje_caribe_1d_2026-10-10'
CONTROL=ROOT/'resultados/migracion_control_sur_1d_2026-10-10'
NEW=ROOT/'resultados/migracion_nazca_cauca_valle_eje_1d_2026-10-10'
BOUNDS={'Cauca':(.8,3.8,-79,-75),'Valle_del_Cauca':(2.8,5.5,-79,-75),'Eje_Cafetero':(3.8,6.3,-77.5,-74.3)}
def paired(a,b,name):
    aa=pd.read_csv(a/'predictions.csv',low_memory=False);aa=aa[aa.split.eq('holdout')]
    bb=pd.read_csv(b/'predictions.csv',low_memory=False);bb=bb[bb.split.eq('holdout')]
    pair=aa.merge(bb,on=['event_id','station','fase','raw_row'],suffixes=('_before','_after'),validate='one_to_one');assert len(pair)==len(aa)
    for col in ['observed_s','ev_lat','ev_lon','ev_prof_km']:np.testing.assert_allclose(pair[col+'_before'],pair[col+'_after'],rtol=0,atol=1e-10)
    assert pair.origin_shift_s_before.eq(0).all() and pair.origin_shift_s_after.eq(0).all()
    pair.to_csv(NEW/f'{name}_holdout.csv',index=False)
    g=pair.assign(a=pair.residual_s_before**2,b=pair.residual_s_after**2).groupby('event_id').agg(a=('a','sum'),b=('b','sum'),n=('a','size')).to_numpy()
    rng=np.random.default_rng(20261009);draw=rng.integers(0,len(g),size=(3000,len(g)));totals=g[draw].sum(axis=1);delta=np.sqrt(totals[:,1]/totals[:,2])-np.sqrt(totals[:,0]/totals[:,2])
    return dict(events=pair.event_id.nunique(),P=len(pair),RMS_before_s=existing.rms(pair.residual_s_before),RMS_after_s=existing.rms(pair.residual_s_after),delta_ci95_s=np.quantile(delta,[.025,.975]).tolist())
def main():
    result=dict(domain_effect=paired(OLD,CONTROL,'domain_effect'),data_effect=paired(CONTROL,NEW,'data_effect'),total_effect=paired(OLD,NEW,'total_effect'))
    with np.load(CONTROL/'vp_final.npz') as a,np.load(NEW/'vp_final.npz') as b:
        for k in ['xg','yg','zg','inversion_xg','inversion_yg','inversion_zg','vp_initial']:np.testing.assert_array_equal(a[k],b[k])
    result['same_expanded_grid_and_1d']=True
    for r in [CONTROL,NEW]:
        rel=pd.read_csv(r/'relocated_train_events.csv');assert rel.status.eq('relocated').all(), 'Incomplete final relocations require explicit previous-node reconstruction'
    print('Computing final bent-ray coverage on identical expanded grids',flush=True)
    cov=[existing.coverage(CONTROL),existing.coverage(NEW)];previous,current=cov
    xyz=previous['xyz'];unique,back=np.unique(xyz[:,:2],axis=0,return_inverse=True);pr=projection(json.loads((NEW/'domain.json').read_text(encoding='utf-8')));la,lo=pr.latlon(unique[:,0],unique[:,1]);dept=departments(la,lo)[back]
    distance=cKDTree(xyz[previous['active']]).query(xyz)[0];deep=(xyz[:,2]>=40)&(xyz[:,2]<=160)
    corridor_masks={z:(previous['lat']>=a)&(previous['lat']<=b)&(previous['lon']>=c)&(previous['lon']<=d) for z,(a,b,c,d) in BOUNDS.items()}
    roi=np.logical_or.reduce(list(corridor_masks.values()));target=(~previous['active'])&deep&roi&(distance<=np.where(previous['lat']<2,160,80))
    gained=target&current['active'];lost=previous['active']&~current['active'];target_counts={}
    for z,mask in corridor_masks.items():target_counts[z]=dict(inactive_targets=int((target&mask).sum()),targets_now_active=int((gained&mask).sum()),targets_now_hits20_sectors3=int((target&mask&(current['hits']>=20)&(current['sectors']>=3)).sum()))
    result.update(targets_before_inactive=int(target.sum()),targets_now_active=int(gained.sum()),targets_now_hits20_sectors3=int((target&(current['hits']>=20)&(current['sectors']>=3)).sum()),previous_active_cells_lost=int(lost.sum()),corridor_counts=target_counts,corridor_counts_overlap=True)
    counts=[]
    for label,c in zip(['expanded_control','new_data'],cov):
        e=c['events'];ez=departments(e.lat,e.lon);row=dict(stage=label,department_zones={})
        for zone in ['Cauca','Valle_del_Cauca','Eje_Cafetero']:
            selected=e[ez==zone];p=c['picks'];p=p[p.event_id.isin(selected.event_id)&p.fase.eq('P')];mask=(dept==zone)&(xyz[:,2]>=40)
            row['department_zones'][zone]=dict(events=len(selected),events_ge40=int(selected.depth_km.ge(40).sum()),P=len(p),station_epochs=p.station.nunique(),active_deep_cells=int((mask&c['active']).sum()),deep_cells_hits20_sectors3=int((mask&(c['hits']>=20)&(c['sectors']>=3)).sum()))
        mask=(c['lat']>=8)&(c['lat']<=13)&(c['lon']>=-79)&(c['lon']<=-71)&(c['xyz'][:,2]>=40)
        row['Caribe_8_13N']=dict(active_deep_cells=int((mask&c['active']).sum()),deep_cells_hits20_sectors3=int((mask&(c['hits']>=20)&(c['sectors']>=3)).sum()))
        counts.append(row)
    result['coverage']=counts;result['limits']='Department zones identify acquisition groups, not plates. Corridor targets overlap. Hits/global direction sectors quantify illumination, not structural resolution. No geological boundary imposed.'
    inputs=[Path(__file__),Path(existing.__file__),ROOT/'codigos/nazca_campana_20261010.py',ROOT/'datos/cartografia/departamentos_dane_2025.geojson']+[r/f for r in [OLD,CONTROL,NEW] for f in ['vp_final.npz','predictions.csv','domain.json']]+[r/'iluminacion_rayos_final.npz' for r in [CONTROL,NEW]]
    result['source_sha256']={str(p):sha256(p) for p in inputs};(NEW/'comparacion_control_iluminacion.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from agregar_mapa_superficie_norte import boundaries,lines
    from shapely.geometry import box
    from shapely.ops import unary_union
    _,_,fs=boundaries();polys=[g[1].intersection(box(-79,.8,-71,13)) for g in fs.values()];borders=[unary_union(polys).boundary,unary_union([g.boundary for g in polys])]
    fig,axs=plt.subplots(1,2,figsize=(12,11),layout='constrained')
    for ax,label,c in zip(axs,['Control de malla ampliada','Con nuevos datos Cauca Valle Eje'],cov):
        m=deep&c['active'];im=ax.scatter(c['lon'][m],c['lat'][m],c=c['xyz'][m,2],s=13,marker='s',cmap='viridis',vmin=40,vmax=160,alpha=.45)
        missing=target&~c['active'];ax.scatter(c['lon'][missing],c['lat'][missing],s=12,facecolors='none',edgecolors='#888888',linewidths=.3,label='Objetivos aún inactivos')
        if label.startswith('Con'):ax.scatter(c['lon'][gained],c['lat'][gained],s=18,facecolors='none',edgecolors='red',linewidths=.6,label='Objetivos ahora activos')
        for i,border in enumerate(borders):
            for line in lines(border):ax.plot(line[:,0],line[:,1],color='#333333' if i==0 else '#999999',linewidth=.7 if i==0 else .25)
        ax.set(xlim=(-79,-71),ylim=(.8,13),xlabel='Longitud WGS84',ylabel='Latitud WGS84',title=label);ax.set_aspect('equal');ax.legend(fontsize=8,loc='upper left');ax.grid(alpha=.15)
    fig.colorbar(im,ax=axs.tolist(),label='Profundidad de celdas activas (km)',shrink=.65);fig.suptitle('Cobertura proyectada de 40–160 km\nMismo inicio 1D y misma malla ampliada; proyección no equivale a superficie de placa')
    fig.savefig(NEW/'comparacion_iluminacion_sur.png',dpi=170);plt.close(fig);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
