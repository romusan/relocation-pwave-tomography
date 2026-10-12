"""Compare northern additions against the preserved previous cohort."""
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
sys.path.insert(0, str(ROOT))
from reconstruir_benioff_3d import model_cells, Projection


def rms(values):
    return float(np.sqrt(np.mean(np.asarray(values)**2))) if len(values) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run', type=Path)
    ap.add_argument('--previous', type=Path, default=ROOT/'resultados/migracion_1900_centro_norte_2026-10-10')
    a = ap.parse_args()
    old, run = a.previous.resolve(), a.run.resolve()
    summaries = [json.loads((r/'summary.json').read_text()) for r in (old, run)]
    if summaries[0]['domain'] != summaries[1]['domain']:
        raise ValueError('Geographic comparison requires the same domain')
    projection = Projection(summaries[1]['domain'])
    cohorts = [pd.read_csv(r/'events_split.csv') for r in (old, run)]
    oldsplit = cohorts[0].set_index('event_id').split
    new_preserved = cohorts[1].set_index('event_id').loc[oldsplit.index, 'split']
    np.testing.assert_array_equal(oldsplit.to_numpy(), new_preserved.to_numpy())
    extra = pd.read_csv(run/'added_events.csv')
    predictions = [pd.read_csv(r/'predictions.csv') for r in (old, run)]
    key = ['event_id', 'station', 'raw_row']
    held = [d[d.split.eq('holdout')][key+['observed_s', 'residual_s']] for d in predictions]
    common = held[0].merge(held[1], on=key, suffixes=('_old', '_new'), validate='one_to_one')
    if len(common) != len(held[0]):
        raise ValueError('Some original held-out observations are missing')
    np.testing.assert_allclose(common.observed_s_old, common.observed_s_new, atol=0, rtol=0)
    common.to_csv(run/'previous_holdout_comparison.csv', index=False)
    coverage, cells = [], []
    for r, e in zip((old, run), cohorts):
        with np.load(r/'vp_final.npz') as model:
            c = model_cells(model, projection)
        cells.append(c)
        coverage.append(dict(run=r.name, selected_events=len(e), active_cells=len(c),
            northern_events=int((e.lat>=7).sum()),
            northern_deep_events=int(((e.lat>=7)&(e.depth_km>=40)).sum()),
            northern_cells=int((c.lat>=7).sum()),
            northern_deep_cells=int(((c.lat>=7)&(c.depth_km>=40)).sum()),
            central_deep_cells=int((c.lat.between(4.5,7.2)&c.lon.between(-76.7,-74.3)&(c.depth_km>=40)).sum()),
            quibdo_deep_cells=int((c.lat.between(4.8,6.8)&c.lon.between(-77.3,-75.3)&(c.depth_km>=40)).sum()),
            quibdo_cells=int((c.lat.between(4.8,6.8)&c.lon.between(-77.3,-75.3)).sum()),
            southwest_deep_cells=int(((c.lat<=5)&c.lon.between(-78.5,-75.5)&(c.depth_km>=40)).sum())))
    newheld = predictions[1][predictions[1].event_id.isin(extra[extra.split.eq('holdout')].event_id)]
    result = dict(coverage=coverage, added_events=len(extra),
        added_deep_events=int((extra.depth_km>=40).sum()), previous_split_preserved=True,
        previous_holdout_events=int(common.event_id.nunique()), previous_holdout_picks=len(common),
        previous_holdout_old_rms_s=rms(common.residual_s_old), previous_holdout_new_rms_s=rms(common.residual_s_new),
        new_holdout_events=int(newheld.event_id.nunique()), new_holdout_picks=len(newheld),
        new_holdout_rms_s=rms(newheld.residual_s),
        northern_definition='7–9 N inside original domain, geographic window only; no tectonic plate assignment',
        limitation='Catalog/picks end at 9 N and 180 km; full Caribbean coast and deeper slab unavailable',
        input_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [*[r/f for r in (old,run) for f in ('events_split.csv','vp_final.npz','predictions.csv','summary.json')],
             run/'added_events.csv',Path(__file__),ROOT/'reconstruir_benioff_3d.py',ROOT/'codigos/regional/geometry.py']})
    (run/'comparison_1900_2200.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    cat = pd.read_csv(ROOT/'datos/catalogo_seleccionado.csv')
    rel = pd.read_csv(run/'relocated_train_events.csv')
    added = rel[rel.event_id.isin(extra.event_id)]
    fig, axs = plt.subplots(1,3,figsize=(15,5),layout='constrained')
    axs[0].scatter(cat.lon,cat.lat,s=1,c='gray',alpha=.12,label='Catálogo SGC')
    axs[0].scatter(rel.lon,rel.lat,s=7,c='black',alpha=.45,label='Sismos relocalizados')
    axs[0].set(xlim=(-79,-73),ylim=(2,9),xlabel='Longitud (°)',ylabel='Latitud (° N)',title='Catálogo y relocalizaciones')
    axs[0].legend(fontsize=8)
    panels = [('lat','lon',-76.3,1.,'N–S: longitudes −77,3 a −75,3°'),
              ('lon','lat',5.8,1.,'O–E: latitudes 4,8 a 6,8° N')]
    for ax,(horizontal,filter_axis,center,half,title) in zip(axs[1:],panels):
        for d,size,col,alpha,label in [(cat,1,'gray',.12,'Catálogo SGC'),
            (rel,8,'black',.45,'Sismos relocalizados')]:
            s=d[d[filter_axis].between(center-half,center+half)]
            ax.scatter(s[horizontal],s.depth_km,s=size,c=col,alpha=alpha,label=label)
        s=cells[1][cells[1][filter_axis].between(center-half,center+half)]
        color=ax.scatter(s[horizontal],s.depth_km,s=35,c=s.dvp_percent,cmap='RdBu_r',vmin=-15,vmax=15,marker='s',label='Vp activo')
        ax.set(xlim=(4.8,6.8) if horizontal=='lat' else (-78,-73),ylim=(180,0),
            xlabel='Latitud (° N)' if horizontal=='lat' else 'Longitud (°)',ylabel='Profundidad (km)',title=title)
        ax.legend(fontsize=8)
        fig.colorbar(color,ax=ax,label='dVp/Vp₀ (%)')
    fig.suptitle('Norte de Colombia: iluminación de Quibdó, Chocó y Antioquia')
    fig.savefig(run/'perfiles_quibdo_antioquia_vp.png',dpi=160)
    fig.savefig(run/'perfiles_quibdo_antioquia_vp.pdf')
    plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'},indent=2))


if __name__=='__main__': main()
