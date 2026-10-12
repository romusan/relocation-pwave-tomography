"""Coverage and independent prediction comparison for a larger regional pilot.

Geographic subsets are diagnostic windows, never tectonic plate assignments.
"""
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
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from reconstruir_benioff_3d import model_cells, Projection


def rms(v):
    return float(np.sqrt(np.mean(np.asarray(v)**2))) if len(v) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run',type=Path)
    ap.add_argument('--previous',type=Path,default=ROOT/'resultados/migracion_final_2026-10-09')
    a = ap.parse_args()
    run,old = a.run.resolve(),a.previous.resolve()
    summaries = [json.loads((p/'summary.json').read_text()) for p in (old,run)]
    if summaries[0]['domain'] != summaries[1]['domain']:
        raise ValueError('Comparison requires identical geographic domain')
    projection = Projection(summaries[1]['domain'])
    sets = [pd.read_csv(p/'events_split.csv') for p in (old,run)]
    cells = []
    counts = []
    for p,s,e in zip((old,run),summaries,sets):
        with np.load(p/'vp_final.npz') as m:
            c = model_cells(m,projection)
        cells.append(c)
        counts.append(dict(run=p.name,selected_events=len(e),
            north_events_lat_ge_7=int((e.lat>=7).sum()),
            north_deep_events=int(((e.lat>=7)&(e.depth_km>=40)).sum()),
            active_vp_cells=len(c),north_active_vp_cells=int((c.lat>=7).sum()),
            north_deep_active_vp_cells=int(((c.lat>=7)&(c.depth_km>=40)).sum()),
            baseline_holdout_rms_s=s['baseline']['holdout_rms_fixed_catalog_origin_s'],
            final_holdout_rms_s=s['history'][-1]['holdout_rms_fixed_catalog_origin_s']))
    pred = [pd.read_csv(p/'predictions.csv') for p in (old,run)]
    keys = ['event_id','station','raw_row']
    held = [d[d.split.eq('holdout')][keys+['residual_s']] for d in pred]
    common = held[0].merge(held[1],on=keys,suffixes=('_old','_new'),validate='one_to_one')
    common.to_csv(run/'common_holdout_comparison.csv',index=False)
    comparison = dict(coverage=counts,common_holdout_events=int(common.event_id.nunique()),
        common_holdout_picks=len(common),common_holdout_old_rms_s=rms(common.residual_s_old),
        common_holdout_new_rms_s=rms(common.residual_s_new),
        caveat='Overall RMS uses different holdout cohorts; common comparison uses only events held out in BOTH runs.',
        north_definition='Latitude >= 7 N, within the same domain; not a Caribbean plate classification',
        input_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [*[r/f for r in (old,run) for f in
                      ('events_split.csv','vp_final.npz','predictions.csv','summary.json')],
                      Path(__file__),ROOT/'reconstruir_benioff_3d.py',ROOT/'codigos/regional/geometry.py']})
    (run/'comparison_500_1000.json').write_text(json.dumps(comparison,indent=2),encoding='utf-8')
    fig,axs = plt.subplots(1,3,figsize=(15,5),layout='constrained')
    domain = summaries[1]['domain']
    e = sets[1]
    for split,color,label in [('train','black','Ajuste'),('holdout','orange','Reserva')]:
        d = e[e.split.eq(split)]
        axs[0].scatter(d.lon,d.lat,s=8,c=color,alpha=.5,label=f'{label}: {len(d)}')
    axs[0].axhline(7,color='gray',linestyle='--',lw=.8)
    axs[0].scatter([-73.10],[6.81],c='green',marker='*',s=80,label='Bucaramanga (referencia)')
    axs[0].set(xlabel='Longitud (°)',ylabel='Latitud (° N)',title='1.000 eventos distribuidos',
               xlim=(domain['lon_min'],domain['lon_max']),ylim=(domain['lat_min'],domain['lat_max']))
    axs[0].legend(fontsize=8)
    cat = pd.read_csv(ROOT/'datos/catalogo_seleccionado.csv')
    relocated = pd.read_csv(run/'relocated_train_events.csv')
    panels = [('lat','lon',-74.,1.,'N–S: longitudes −75 a −73°'),
              ('lon','lat',7.,1.,'O–E: latitudes 6 a 8° N')]
    for ax,(horizontal,filter_axis,center,half,title) in zip(axs[1:],panels):
        for d,size,color,alpha,label in [(cat,1,'gray',.12,'Catalogo SGC'),
                                        (relocated,9,'black',.65,'Relocalizados')]:
            s = d[d[filter_axis].between(center-half,center+half)]
            ax.scatter(s[horizontal],s.depth_km,s=size,c=color,alpha=alpha,label=label)
        c = cells[1]
        s = c[c[filter_axis].between(center-half,center+half)]
        color = ax.scatter(s[horizontal],s.depth_km,s=38,c=s.dvp_percent,cmap='RdBu_r',
                           vmin=-15,vmax=15,marker='s',label='Vp activo')
        lim = (domain['lat_min'],domain['lat_max']) if horizontal=='lat' else (domain['lon_min'],domain['lon_max'])
        ax.set(xlabel='Latitud (° N)' if horizontal=='lat' else 'Longitud (°)',
               ylabel='Profundidad (km)',ylim=(180,0),xlim=lim,title=title)
        ax.legend(fontsize=8)
        fig.colorbar(color,ax=ax,label='dVp/Vp0 (%)')
    fig.suptitle('Norte de Colombia: ventanas exploratorias, sin asignacion de placa')
    fig.savefig(run/'perfiles_norte_vp.png',dpi=160)
    fig.savefig(run/'perfiles_norte_vp.pdf')
    plt.close(fig)
    print(json.dumps({k:v for k,v in comparison.items() if k!='input_sha256'},indent=2))


if __name__ == '__main__': main()
