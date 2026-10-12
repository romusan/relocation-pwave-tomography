"""Export a static scientific companion to the interactive Benioff explorer."""
from pathlib import Path
import argparse
import json
import sys
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from reconstruir_benioff_3d import Projection


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('viewer',type=Path)
    ap.add_argument('run',type=Path)
    a=ap.parse_args()
    p=Projection(json.loads((ROOT/'dominio.json').read_text()))
    catalog_path=ROOT/'datos/catalogo_seleccionado.csv'
    d=pd.read_csv(catalog_path)
    # Reproducible display thinning only; tomography and geometry remain unchanged.
    d=d.sample(min(len(d),15000),random_state=20261009)
    x,y=p.xy(d.lat,d.lon)
    e=pd.read_csv(a.run/'relocated_train_events.csv')
    c=pd.read_csv(a.viewer/'vp_celdas_activas.csv')
    f=plt.figure(figsize=(12,9),layout='constrained')
    ax=f.add_subplot(111,projection='3d')
    ax.scatter(x,y,d.depth_km,s=1,c=d.depth_km,cmap='viridis',vmin=0,vmax=180,
               alpha=.16,label='Catalogo (15.000 puntos de contexto)')
    with np.load(a.viewer/'fragmentos_geometricos.npz') as mesh:
        triangles=mesh['xyz'][mesh['triangles']]
    if len(triangles):
        ax.add_collection3d(Poly3DCollection(triangles,facecolor='gold',edgecolor='none',alpha=.18))
    ax.scatter(e.x_km,e.y_km,e.depth_km,s=5,c='black',alpha=.55,label=f'{len(e)} relocalizaciones')
    color=ax.scatter(c.x_km,c.y_km,c.depth_km,s=16,c=c.dvp_percent,cmap='RdBu_r',
                     vmin=-15,vmax=15,marker='s',alpha=.85,label=f'{len(c)} celdas activas Vp')
    refx,refy=p.xy([3.90,6.81],[-78.20,-73.10])
    ax.plot(refx,refy,[0,0],'g--',lw=1)
    for rx,ry,label in zip(refx,refy,['Pacifico (referencia)','Bucaramanga (referencia)']):
        ax.text(rx,ry,0,label,color='darkgreen',fontsize=8)
    ax.set(xlabel='Este AEQD (km)',ylabel='Norte AEQD (km)',zlabel='Profundidad (km)',
           zlim=(180,0),title='Pacifico–Bucaramanga: sismicidad y Vp experimental\nFragmentos provisionales; sin asignacion de placa')
    ax.set_box_aspect([np.ptp(x),np.ptp(y),180])
    ax.view_init(elev=22,azim=-130)
    ax.legend(loc='upper left',fontsize=8)
    f.colorbar(color,ax=ax,shrink=.55,pad=.06,label='dVp/Vp0 (%)')
    f.savefig(a.viewer/'vista_3d.png',dpi=180)
    plt.close(f)
    inputs=[catalog_path,a.run/'relocated_train_events.csv',a.viewer/'vp_celdas_activas.csv',
            a.viewer/'fragmentos_geometricos.npz',ROOT/'dominio.json',Path(__file__),
            ROOT/'reconstruir_benioff_3d.py',ROOT/'codigos/regional/geometry.py']
    (a.viewer/'vista_3d_manifest.json').write_text(json.dumps(dict(
        static_catalog_display_points=len(d),display_only_seed=20261009,
        input_sha256={str(q.resolve()):hashlib.sha256(q.read_bytes()).hexdigest() for q in inputs}),indent=2))
    print(a.viewer/'vista_3d.png')


if __name__=='__main__': main()
