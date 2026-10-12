"""Measure geometric coverage changes; never fill gaps by relaxing thresholds."""
from pathlib import Path
import argparse,json,hashlib,sys
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from shapely.ops import unary_union
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from regional.geometry import Projection

ROOT=Path(__file__).resolve().parents[1]


def measure(viewer,projection):
    with np.load(viewer/'fragmentos_geometricos.npz') as m:
        xyz=m['xyz']; faces=m['triangles']
    triangles=xyz[faces]
    cover=unary_union([Polygon(t[:,:2]) for t in triangles])
    x,y=projection.xy([4.5,4.5,7.2,7.2],[-76.7,-74.3,-74.3,-76.7])
    window=Polygon(np.column_stack([x,y]))
    pairs=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
    graph=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(len(xyz),len(xyz))).tocsr()
    n,labels=connected_components(graph,directed=False)
    used=np.unique(faces)
    parts=len(np.unique(labels[used]))
    lat,lon=projection.latlon(xyz[:,0],xyz[:,1])
    central=(lat>=4.5)&(lat<=7.2)&(lon>=-76.7)&(lon<=-74.3)
    west=set(labels[central&(lon<=-75.7)&np.isin(np.arange(len(xyz)),used)])
    east=set(labels[central&(lon>=-74.8)&np.isin(np.arange(len(xyz)),used)])
    qc=pd.read_csv(viewer/'controles_geometricos_qc.csv')
    qlat,qlon=projection.latlon(qc.x_km,qc.y_km)
    q=qc[(qlat>=4.5)&(qlat<=7.2)&(qlon>=-76.7)&(qlon<=-74.3)]
    return dict(triangles=len(faces),connected_mesh_components=parts,
        central_projected_area_km2=window.area,central_mesh_covered_area_km2=cover.intersection(window).area,
        central_covered_fraction=cover.intersection(window).area/window.area,
        central_control_columns=len(q),central_rejected_columns=int((~q.accepted).sum()),
        west_east_connected_somewhere_in_mesh=bool(west&east)),np.column_stack([lon,lat])[faces]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--viewer',type=Path,required=True)
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--previous-viewer',type=Path,default=ROOT/'resultados/benioff_3d_1500_norte_2026-10-09')
    a=ap.parse_args(); p=Projection(json.loads((a.run/'summary.json').read_text())['domain'])
    results=[]; meshes=[]
    for v in [a.previous_viewer,a.viewer]:
        r,m=measure(v,p); results.append(r); meshes.append(m)
    fig,axs=plt.subplots(1,2,figsize=(10,6),layout='constrained')
    for ax,m,r,title in zip(axs,meshes,results,['1.500 eventos','1.900 eventos']):
        ax.add_collection(PolyCollection(m,facecolors='gold',edgecolors='#826600',linewidths=.5,alpha=.5))
        ax.set(xlim=(-76.7,-74.3),ylim=(4.5,7.2),xlabel='Longitud (°)',ylabel='Latitud (° N)',
            title=f'{title}: cobertura geométrica {100*r["central_covered_fraction"]:.1f}%')
        ax.set_aspect('equal')
    fig.suptitle('Ventana central: proyección de fragmentos aceptados (no continuidad de placa demostrada)')
    fig.savefig(a.run/'hueco_central_antes_despues.png',dpi=180); plt.close(fig)
    files=[*[v/f for v in [a.previous_viewer,a.viewer] for f in ['fragmentos_geometricos.npz','controles_geometricos_qc.csv']],Path(__file__)]
    result=dict(previous=results[0],current=results[1],window=dict(lat_min=4.5,lat_max=7.2,lon_min=-76.7,lon_max=-74.3),
        caution='Projected triangle area is geometric support, not tomographic resolution or plate identification; component connections may pass outside central window',
        input_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    (a.run/'hueco_central_comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'},indent=2))


if __name__=='__main__': main()
