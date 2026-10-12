"""Static QA figure for an experimental migration run."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('run',type=Path)
    args=ap.parse_args()
    run=args.run.resolve()
    summary=json.loads((run/'summary.json').read_text(encoding='utf-8'))
    events=pd.read_csv(run/'events_split.csv')
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for split,label,color in [('train','Ajuste','#2563eb'),('holdout','Reserva','#ef8b2c')]:
        sub=events[events.split.eq(split)]
        axes[0].scatter(sub.lon,sub.lat,s=15,c=color,label=f'{label}: {len(sub)}')
    domain=summary['domain']
    axes[0].set(xlim=(domain['lon_min'],domain['lon_max']),ylim=(domain['lat_min'],domain['lat_max']),
                xlabel='Longitud (°)',ylabel='Latitud (° N)',title='Selección distribuida por el dominio')
    axes[0].legend(fontsize=8)
    train=[summary['baseline']['train_rms_fixed_catalog_s']]+[r['train_rms_after_update_s'] for r in summary['history']]
    test=[summary['baseline']['holdout_rms_fixed_catalog_origin_s']]+[r['holdout_rms_fixed_catalog_origin_s'] for r in summary['history']]
    axes[1].plot(range(len(train)),train,'o-',label='Ajuste: ubicación + Vp')
    axes[1].plot(range(len(test)),test,'s-',label='Reserva: ubicación/origen fijos')
    axes[1].set(xlabel='Ronda (0 = catálogo + 1-D)',ylabel='RMS P (s)',title='Comparación con los mismos picks')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=.2)
    pred=pd.read_csv(run/'predictions.csv')
    for split,label,color in [('train','Ajuste','#2563eb'),('holdout','Reserva','#ef8b2c')]:
        v=pred[pred.split.eq(split)].residual_s
        axes[2].hist(v,bins=np.linspace(-6,6,40),histtype='step',density=True,color=color,label=label)
    axes[2].set(xlabel='Residuo P final (s)',ylabel='Densidad',title='Cola de residuos (vista ±6 s)')
    axes[2].legend(fontsize=8)
    fig.suptitle('Piloto de migración — FSM de primer orden; resultado experimental',fontsize=12)
    fig.savefig(run/'diagnostico_migracion.png',dpi=160)
    fig.savefig(run/'diagnostico_migracion.pdf')
    plt.close(fig)
    print(run/'diagnostico_migracion.png')


if __name__=='__main__':
    main()
