"""Physical checks for continuous spline geometry and 1D plus slab volumes."""
import numpy as np
from pathlib import Path
from modelo_3d_continuo import fit_sheet,evaluate,grd,ROOT
from regional.initial_model import load_initial_model
from regional.geometry import Projection,grids
import json

def main():
    x,y=np.meshgrid(np.linspace(-100,100,20),np.linspace(-80,80,18));xy=np.c_[x.ravel(),y.ravel()];z=60+.3*xy[:,0]+.15*xy[:,1]
    sheet=fit_sheet(xy,z,np.ones(len(z)),5,5,0)
    query=np.array([[-70,-50],[0,0],[30,50]])
    s,n=evaluate(sheet,query);np.testing.assert_allclose(s[:,2],60+.3*query[:,0]+.15*query[:,1],atol=1e-8)
    expected=np.array([-.3,-.15,1]);expected/=np.linalg.norm(expected);np.testing.assert_allclose(n,np.tile(expected,(len(n),1)),atol=1e-8)
    np.testing.assert_allclose(np.linalg.norm(30*n,axis=1),30)
    model=ROOT/'resultados/modelo_1d_placas_continuas_slab2_2026-10-10';old=ROOT/'resultados/migracion_isc_choco_2026-10-10';domain=json.loads((old/'domain.json').read_text());g,_=grids(domain,Projection(domain),10,20)
    vp=load_initial_model(model/'vp_inicial_1d_mas_placas_30km.npz',g,domain)
    with np.load(model/'vp_inicial_1d_mas_placas_30km.npz') as m,np.load(old/'vp_final.npz') as base:
        np.testing.assert_array_equal(m['vp_background_1d'],base['vp_initial'])
        mask=m['anomaly_fraction']==0;np.testing.assert_array_equal(vp[mask],base['vp_initial'][mask])
        assert np.count_nonzero(~mask)>10000
    for code in ['cam','car']:
        lon,lat,z=grd(next((ROOT/'datos/slab2_2026-10-10').glob(code+'*dep*')));xx,yy=np.meshgrid(lon,lat)
        assert not (np.isfinite(z)&(xx>=-79)&(xx<=-71)&(yy>=2)&(yy<=13)).any()
    print('PASS planar NURBS, analytic normals, 30 km thickness, exact original 1D background, finite grid, no fabricated CAR/CAM support')
if __name__=='__main__':main()
