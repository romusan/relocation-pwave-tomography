"""Geographic profiles of supported Vp cells; no tectonic boundary imposed."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reconstruir_benioff_ampliado import model_cells
from nazca_campana_20261010 import projection as Projection
from regional.inputs import sha256

def main():
    run=ROOT/'resultados/migracion_nazca_cauca_valle_eje_1d_2026-10-10'
    pr=Projection(json.loads((run/'domain.json').read_text(encoding='utf-8')))
    with np.load(run/'vp_final.npz') as m:c=model_cells(m,pr)
    e=pd.read_csv(run/'relocated_train_events.csv');e=e[e.status.eq('relocated')]
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for ax,lon in zip(axs.flat,[-77,-76,-75,-74]):
        # Fixed half-width in longitude; cells are drawn without gap interpolation.
        d=c[(c.lon-lon).abs()<=.25];q=e[(e.lon-lon).abs()<=.25]
        im=ax.scatter(d.lat,d.depth_km,c=d.dvp_percent,s=35,marker='s',cmap='RdBu_r',vmin=-15,vmax=15)
        ax.scatter(q.lat,q.depth_km,s=5,c='black',alpha=.35,label='SGC relocalizados')
        ax.axvspan(7,9,color='gold',alpha=.06,label='Banda de comparación 7–9° N')
        ax.set(xlim=(.8,13),ylim=(180,0),xlabel='Latitud WGS84',ylabel='Profundidad (km)',title=f'Longitud {lon}° ±0.25°')
        ax.grid(alpha=.2);ax.legend(fontsize=8,loc='lower left')
    fig.colorbar(im,ax=axs.ravel().tolist(),label='dVp final respecto al 1D original (%)',shrink=.8)
    fig.suptitle('Perfiles de Nazca y transición Caribe desde inicio 1D\nCeldas activas sin rellenar huecos; la franja amarilla no define un límite de placas')
    fig.savefig(run/'perfiles_transicion.png',dpi=170);plt.close(fig)
    inputs=[Path(__file__),ROOT/'codigos/nazca_campana_20261010.py',run/'vp_final.npz',run/'relocated_train_events.csv']
    (run/'perfiles_transicion_metodo.json').write_text(json.dumps(dict(longitudes=[-77,-76,-75,-74],half_width_deg=.25,meaning='Supported cell centers, no interpolation across missing support; 7–9N is only a comparison band; no plate assignment',sources={str(p):sha256(p) for p in inputs}),indent=2),encoding='utf-8')
    print('Geographic transition profiles exported',flush=True)
if __name__=='__main__':main()
