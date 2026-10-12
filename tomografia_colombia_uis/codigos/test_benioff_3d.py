"""Scientific guards: no gap bridging or averaging overlapping depth columns."""
import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reconstruir_benioff_3d import supported_mesh, controls, profile_coordinates


class BenioffGuards(unittest.TestCase):
    def test_two_clusters_remain_disconnected(self):
        p = np.array([[0,0,80],[10,0,85],[0,10,85],
                      [200,0,80],[210,0,85],[200,10,85]])
        faces = supported_mesh(p, max_edge=30)
        self.assertEqual(len(faces), 2)
        self.assertTrue(all(np.all(f<3) or np.all(f>=3) for f in faces))

    def test_depth_superposition_is_not_averaged_into_surface(self):
        d = pd.DataFrame(dict(event_id=['a','b','c','d'],status=['relocated']*4,
            x_km=[1,2,21,22],y_km=[1,2,1,2],depth_km=[50,140,80,90],gap_deg=[100]*4))
        c = controls(d)
        self.assertEqual(c.accepted.tolist(), [False,True])
        self.assertEqual(c.depth_spread_km.tolist(), [90,10])

    def test_steep_triangle_rejected_and_degenerate_input_safe(self):
        self.assertEqual(len(supported_mesh([[0,0,50],[10,0,150],[0,10,50]])),0)
        self.assertEqual(len(supported_mesh([[0,0,50],[10,0,60],[20,0,70]])),0)

    def test_section_projection_distance_and_offset(self):
        along,across,length = profile_coordinates([0,50,100],[20,-10,0],np.array([0,0]),np.array([100,0]))
        np.testing.assert_allclose(along,[0,50,100])
        np.testing.assert_allclose(across,[20,-10,0])
        self.assertEqual(length,100)


if __name__ == '__main__': unittest.main()
