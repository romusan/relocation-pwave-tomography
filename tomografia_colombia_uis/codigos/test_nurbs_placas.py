"""Check rational evaluation, normal thickness and model-grid rejection."""
import json,tempfile
from pathlib import Path
import numpy as np
from nurbs_placas import evaluate
from regional.initial_model import load_initial_model
from regional.geometry import Grid

def main():
    u,v=np.meshgrid([0,.5,1],[0,.5,1],indexing='ij')
    ctrl=np.stack([u*100,v*80,20+u*30+v*10],axis=-1)
    uv=np.array([[.1,.7],[.5,.5],[.9,.2]])
    for weights in [np.ones((3,3)),np.array([[1,2,1],[1,3,1],[1,2,1]])]:
        s,n=evaluate(ctrl,weights,uv)
        np.testing.assert_allclose(n@np.array([-.3,-.125,1])/np.linalg.norm([-.3,-.125,1]),1,atol=1e-10)
        np.testing.assert_allclose(np.linalg.norm((s+35*n)-(s-35*n),axis=1),70)
        np.testing.assert_allclose(s[:,2],20+.3*s[:,0]+.125*s[:,1],atol=1e-10)
    grid=Grid(np.arange(3.)*10,np.arange(3.)*10,np.arange(3.)*10);domain={'lat_min':2.}
    with tempfile.TemporaryDirectory() as tmp:
        f=Path(tmp)/'m.npz';np.savez(f,vp=np.ones(grid.shape)*6,xg=grid.x,yg=grid.y,zg=grid.z,domain_json=json.dumps(domain))
        assert load_initial_model(f,grid,domain).shape==grid.shape
        for g,d in [(Grid(grid.x+1,grid.y,grid.z),domain),(grid,{'lat_min':3.})]:
            try:load_initial_model(f,g,d)
            except ValueError:pass
            else:raise AssertionError('Incompatible model accepted')
    print('PASS rational plane, analytic normals, normal thickness, grid/domain validation')
if __name__=='__main__':main()
