"""Independent finite-difference checks and failure gates for native Jacobian."""
import unittest,json
from pathlib import Path
import numpy as np
from regional.fsm_reference import fast_sweeping_3d
from regional.rays_consistent import DiscreteGraph
from regional.rays import mapping
from regional.geometry import Grid
class AdjointChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grid=Grid(np.arange(0,130,10.),np.arange(0,150,10.),np.arange(0,90,10.))
        cls.coarse=Grid(np.arange(0,130,20.),np.arange(0,150,20.),np.arange(0,90,20.))
        cls.map=mapping(cls.grid,cls.coarse);cls.nc=int(np.prod(cls.coarse.shape))
        cls.source=(0,7,3);cls.src=int(np.ravel_multi_index(cls.source,cls.grid.shape))
        rng=np.random.default_rng(431);cls.slow=.15+rng.uniform(0,.02,cls.grid.shape)
        cls.field=cls.forward(cls.slow);cls.starts=np.array([np.ravel_multi_index(x,cls.grid.shape) for x in [(8,14,12),(6,3,11),(4,12,8),(0,7,3)]])
    @classmethod
    def forward(cls,s):return fast_sweeping_3d(s,*cls.source,10.,10.,10.,8,1e6)
    def graph(self):return DiscreteGraph(self.field,self.slow,self.map,self.src,10.,self.nc)
    def test_euler_identity_and_zero_source(self):
        g=self.graph()
        try:
            rows,mom,v=g.batch(self.starts)
            np.testing.assert_allclose(mom,self.field.ravel()[self.starts],atol=2e-8,rtol=1e-10)
            self.assertTrue(np.all(rows>=0));self.assertEqual(rows[-1].sum(),0)
        finally:g.close()
    def test_independent_central_finite_difference_directions(self):
        g=self.graph();rows,_,_=g.batch(self.starts);g.close()
        rng=np.random.default_rng(328)
        for _ in range(4):
            direction=rng.normal(0,.001,self.nc);d=direction[self.map].reshape(self.grid.shape)
            eps=1e-4
            fd=(self.forward(self.slow+eps*d).ravel()[self.starts]-self.forward(self.slow-eps*d).ravel()[self.starts])/(2*eps)
            np.testing.assert_allclose(rows@direction,fd,atol=2e-6,rtol=2e-5)
    def test_uniform_slowness_scaling(self):
        scaled=self.forward(self.slow*1.00001)
        np.testing.assert_allclose(scaled,self.field*1.00001,atol=5e-8)
    def test_reject_inconsistent_field(self):
        bad=self.field.copy();bad[4,6,7]+=.1
        with self.assertRaises(RuntimeError):DiscreteGraph(bad,self.slow,self.map,self.src,10.,self.nc)
if __name__=='__main__':unittest.main(verbosity=2)
