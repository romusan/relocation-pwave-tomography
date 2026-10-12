"""Independent geometric and provenance checks for the h hypotheses."""
from pathlib import Path
import ast,json,hashlib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
ROOT=Path(__file__).resolve().parents[1];C=ROOT/'resultados/campana_perfil_suavidad_20261011h';O=C/'hipotesis_placas'
H=json.loads((O/'metodologia.json').read_text(encoding='utf-8'))
for rel,expected in H['source_sha256'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==expected,rel
# A known oblique plane tests extraction independently of the tomography.
source=ast.parse((ROOT/'codigos/hipotesis_placas_revision_h.py').read_text(encoding='utf-8'))
fn=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='isomesh')
corners=np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])[:,::-1]
ns=dict(np=np,cx=corners[None,:,:].astype(float),cv=(corners@np.array([1,2,3]))[None,:].astype(float),tets=np.array([[0,1,3,7],[0,3,2,7],[0,2,6,7],[0,6,4,7],[0,4,5,7],[0,5,1,7]]),edges=[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)])
exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'isolated_isomesh','exec'),ns)
points,faces=ns['isomesh'](2.5);assert len(faces)>0
np.testing.assert_allclose(points@np.array([1,2,3]),2.5,atol=1e-12)
assert ((points>=0)&(points<=1)).all()
for item in H['isosurfaces']:
    a=np.load(O/f"anomalia_{item['level_percent']:+g}.npz");p=a['xyz'];t=a['faces'];assert len(t)==item['triangles']
    assert np.isfinite(p).all() and (p[:,2]>=40).all() and (p[:,2]<=180).all()
    assert t.min()>=0 and t.max()<len(p)
cells=pd.read_csv(O/'kmeans_celdas.csv');X=cells[['x','y','z']].to_numpy()
centers=pd.read_csv(O/'kmeans_centroides.csv')
for metric in ['km','estandarizado']:
    for k in range(2,7):
        ct=centers[centers.metric.eq(metric)&centers.k.eq(k)][['x_km','y_km','depth_km']].to_numpy()
        distances=cdist(X,ct) if metric=='km' else cdist(X/X.std(axis=0),ct/X.std(axis=0))
        assert np.array_equal(distances.argmin(axis=1),cells[f'{metric}_k{k}'].to_numpy())
for sheet in H['nurbs']:
    a=np.load(O/(sheet['name'].lower()+'_superficie.npz'));assert np.isfinite(a['middle']).all()
    np.testing.assert_allclose(np.linalg.norm(a['normals'],axis=1),1,atol=1e-12)
    assert (a['weights']==1).all() and a['height'].min()>=10-1e-6 and a['height'].max()<=180+1e-6
    a30=np.load(O/(sheet['name'].lower()+'_espesor_30km.npz'))
    np.testing.assert_allclose(np.linalg.norm(a30['top']-a30['bottom'],axis=1),30,atol=1e-10)
    np.testing.assert_allclose(a30['middle'],a['middle'])
assert H['train_events']==2897 and H['equal_weight_cells']==len(cells)==706
report=dict(source_hashes_verified=True,analytic_oblique_plane_pass=True,all_mesh_indices_and_depths_valid=True,kmeans_labels_match_saved_centroids=True,nurbs_unit_weights_and_bounded_depths_verified=True,nominal_normal_thickness_verified=True,tomography_source_unchanged=True)
(O/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
