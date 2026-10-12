"""Shortest existing mesh path between central west/east controls; no new edges."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from regional.geometry import Projection


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--viewer',type=Path,required=True); ap.add_argument('--run',type=Path,required=True)
    a=ap.parse_args(); mesh_path=a.viewer/'fragmentos_geometricos.npz'
    with np.load(mesh_path) as m: xyz=m['xyz']; faces=m['triangles']
    projection=Projection(json.loads((a.run/'summary.json').read_text())['domain'])
    lat,lon=projection.latlon(xyz[:,0],xyz[:,1]); central=(lat>=4.5)&(lat<=7.2)&(lon>=-76.7)&(lon<=-74.3)
    pairs=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
    used=np.unique(faces); west=np.flatnonzero(central&(lon<=-75.7)&np.isin(np.arange(len(xyz)),used))
    east=np.flatnonzero(central&(lon>=-74.8)&np.isin(np.arange(len(xyz)),used)); output={}
    qc_path=a.viewer/'controles_geometricos_qc.csv'; qc=pd.read_csv(qc_path); good=qc[qc.accepted].reset_index(drop=True)
    np.testing.assert_allclose(good[['x_km','y_km','depth_km']].to_numpy(),xyz)
    supported=good.n_events.to_numpy()>=2
    cases=[('whole_mesh',pairs),('central_only',pairs[central[pairs].all(axis=1)]),
           ('central_min_2_events',pairs[(central[pairs]&supported[pairs]).all(axis=1)])]
    for label,edges in cases:
        weight=np.linalg.norm(xyz[edges[:,0]]-xyz[edges[:,1]],axis=1)
        graph=coo_matrix((weight,(edges[:,0],edges[:,1])),shape=(len(xyz),len(xyz))).tocsr()
        distance,prev=dijkstra(graph,directed=False,indices=west,return_predecessors=True)
        sub=distance[:,east]; connected=bool(np.isfinite(sub).any()); output[label]=dict(connected=connected)
        if connected:
            row,col=np.unravel_index(np.argmin(sub),sub.shape); target=int(east[col]); source=int(west[row]); path=[target]
            while path[-1]!=source: path.append(int(prev[row,path[-1]]))
            path=path[::-1]; delta=np.diff(xyz[path],axis=0); lengths=np.linalg.norm(delta,axis=1)
            horizontal=np.linalg.norm(delta[:,:2],axis=1); slopes=abs(delta[:,2])/horizontal
            if np.max(horizontal)>75+1e-8 or np.max(slopes)>1.5+1e-8: raise ValueError('Path exceeds geometric limits')
            frame=pd.DataFrame(dict(control_index=path,lon=lon[path],lat=lat[path],depth_km=xyz[path,2],
                supporting_events=good.iloc[path].n_events.to_numpy(),depth_spread_km=good.iloc[path].depth_spread_km.to_numpy()))
            frame.to_csv(a.run/f'geometric_connection_{label}.csv',index=False)
            output[label].update(path_length_km=float(sub[row,col]),vertices=len(path),
                all_vertices_in_central_window=bool(central[path].all()),
                depth_min_km=float(xyz[path,2].min()),depth_max_km=float(xyz[path,2].max()),
                max_horizontal_edge_km=float(horizontal.max()),max_slope=float(slopes.max()),
                minimum_events_per_control=int(frame.supporting_events.min()))
    inputs=[mesh_path,qc_path,a.run/'summary.json',Path(__file__),Path(__file__).parent/'regional/geometry.py']
    output['input_sha256']={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    output['caution']='Existing geometric connection only; shallow 40–50 km connection does not demonstrate deep slab continuity or plate identity'
    (a.run/'conexion_geometrica_central.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in output.items() if k!='input_sha256'},indent=2))


if __name__=='__main__': main()
