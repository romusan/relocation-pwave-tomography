"""Targeted Cauca/Valle coverage, cross-sections and unchanged holdout comparison."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from reconstruir_benioff_3d import Projection,model_cells


def rms(values):
    return float(np.sqrt(np.mean(np.asarray(values)**2))) if len(values) else None


def south(d):
    # Explicit geographic window, not an administrative boundary or plate label.
    return d.lat.between(2,5)&d.lon.between(-78.5,-75.5)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run',type=Path)
    ap.add_argument('--previous',type=Path,default=ROOT/'resultados/migracion_1000_caribe_2026-10-09')
    a=ap.parse_args()
    run,prev=a.run.resolve(),a.previous.resolve()
    old=pd.read_csv(prev/'events_split.csv')
    events=pd.read_csv(run/'events_split.csv')
    added=pd.read_csv(run/'added_events.csv')
    shared=old[['event_id','split']].merge(events[['event_id','split']],on='event_id',suffixes=('_old','_new'),validate='one_to_one')
    if len(shared)!=len(old) or not shared.split_old.eq(shared.split_new).all():
        raise ValueError('Previous cohort or its split changed')
    summaries=[json.loads((r/'summary.json').read_text()) for r in (prev,run)]
    if summaries[0]['domain']!=summaries[1]['domain']:
        raise ValueError('Domain changed')
    projection=Projection(summaries[1]['domain'])
    cells=[]
    for r in (prev,run):
        with np.load(r/'vp_final.npz') as m:
            cells.append(model_cells(m,projection))
    predictions=[pd.read_csv(r/'predictions.csv') for r in (prev,run)]
    held=[d[d.split.eq('holdout')] for d in predictions]
    keys=['event_id','station','raw_row']
    common=held[0][keys+['observed_s','residual_s']].merge(held[1][keys+['observed_s','residual_s']],
        on=keys,suffixes=('_old','_new'),validate='one_to_one')
    if len(common)!=len(held[0]):
        raise ValueError('Old heldout observations did not remain held out')
    np.testing.assert_allclose(common.observed_s_old,common.observed_s_new,atol=1e-10)
    common.to_csv(run/'previous_holdout_comparison.csv',index=False)
    fresh=held[1][held[1].event_id.isin(added.event_id)]
    coverage=[dict(run=r.name,selected_events=len(d),active_vp_cells=len(c),
        south_active_vp_cells=int(south(c).sum()),
        south_deep_active_vp_cells=int((south(c)&c.depth_km.ge(40)).sum()),
        north_active_vp_cells=int(c.lat.ge(7).sum()),
        north_deep_active_vp_cells=int((c.lat.ge(7)&c.depth_km.ge(40)).sum()))
        for r,d,c in zip((prev,run),(old,events),cells)]
    paths=[*[r/f for r in (prev,run) for f in ('events_split.csv','vp_final.npz','summary.json','predictions.csv')],
        run/'added_events.csv',ROOT/'datos/catalogo_seleccionado.csv',Path(__file__),
        ROOT/'reconstruir_benioff_3d.py',ROOT/'codigos/regional/geometry.py']
    comparison=dict(previous_event_count=len(old),added_event_count=len(added),total_events=len(events),
        additions_by_group=added.addition_group.value_counts().to_dict(),
        added_depth_ge_40=int(added.depth_km.ge(40).sum()),previous_split_preserved=True,
        previous_holdout_events=common.event_id.nunique(),previous_holdout_picks=len(common),
        previous_holdout_old_rms_s=rms(common.residual_s_old),previous_holdout_new_rms_s=rms(common.residual_s_new),
        added_holdout_events=fresh.event_id.nunique(),added_holdout_picks=len(fresh),added_holdout_rms_s=rms(fresh.residual_s),
        total_holdout_rms_s=rms(held[1].residual_s),coverage=coverage,
        south_window=dict(lat_min=2,lat_max=5,lon_min=-78.5,lon_max=-75.5,minimum_deep_km=40,
            meaning='Geographic diagnostic window, not exact department boundaries or plate identity'),
        input_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (run/'comparison_1000_1200.json').write_text(json.dumps(comparison,indent=2),encoding='utf-8')
    cat=pd.read_csv(ROOT/'datos/catalogo_seleccionado.csv')
    relocated=pd.read_csv(run/'relocated_train_events.csv')
    old_rel=relocated[~relocated.event_id.isin(added.event_id)]
    new_rel=relocated[relocated.event_id.isin(added.event_id)]
    fig,axs=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    axs[0].scatter(old.lon,old.lat,s=8,c='gray',alpha=.35,label='1.000 eventos conservados')
    axs[0].scatter(added.lon,added.lat,s=15,c='darkorange',label='200 eventos nuevos')
    axs[0].set(xlabel='Longitud (°)',ylabel='Latitud (° N)',xlim=(-79,-73),ylim=(2,9),title='Ampliacion dirigida al sur')
    axs[0].legend(loc='upper left',fontsize=8)
    panels=[('lat','lon',-77.5,-75.7,(2,5),'N–S: longitudes −77,5 a −75,7°'),
            ('lon','lat',3,4.5,(-78.5,-75.5),'O–E: latitudes 3 a 4,5° N')]
    for ax,(horizontal,filter_axis,low,high,lim,title) in zip(axs[1:],panels):
        for d,size,color,alpha,label in [(cat,1,'gray',.12,'Catalogo SGC'),(old_rel,8,'black',.4,'Relocalizados previos'),
                                      (new_rel,15,'darkorange',.8,'Nuevos relocalizados')]:
            s=d[d[filter_axis].between(low,high)&d[horizontal].between(*lim)]
            ax.scatter(s[horizontal],s.depth_km,s=size,c=color,alpha=alpha,label=label)
        c=cells[1]
        s=c[c[filter_axis].between(low,high)&c[horizontal].between(*lim)]
        color=ax.scatter(s[horizontal],s.depth_km,s=35,c=s.dvp_percent,cmap='RdBu_r',
                         vmin=-15,vmax=15,marker='s',label='Vp activo')
        ax.set(xlabel='Latitud (° N)' if horizontal=='lat' else 'Longitud (°)',ylabel='Profundidad (km)',
               ylim=(180,0),xlim=lim,title=title)
        ax.legend(loc='lower left',fontsize=7)
        fig.colorbar(color,ax=ax,label='dVp/Vp0 (%)')
    fig.suptitle('Cauca–Valle: sismicidad y Vp experimental; sin asignacion automatica a Nazca')
    fig.savefig(run/'perfiles_cauca_valle_vp.png',dpi=160)
    fig.savefig(run/'perfiles_cauca_valle_vp.pdf')
    plt.close(fig)
    print(json.dumps({k:v for k,v in comparison.items() if k!='input_sha256'},indent=2),flush=True)


if __name__=='__main__': main()
