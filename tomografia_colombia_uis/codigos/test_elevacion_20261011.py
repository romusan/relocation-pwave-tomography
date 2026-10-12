"""Analytic gates for the first-order station-elevation correction."""
import unittest, tempfile
from pathlib import Path
import numpy as np, pandas as pd
from regional.geometry import Grid
from regional.forward import FieldCache
from regional.rays_consistent import kernels


class ElevationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grid = Grid(np.arange(0., 121., 10.), np.arange(0., 121., 10.), np.arange(0., 61., 10.))
        cls.vp = np.full(cls.grid.shape, 6.0)
        cls.cfg = {'fsm_sweeps': 8, 'time_field_memory_limit_mb': 64}
        cls.tmp = tempfile.TemporaryDirectory()
        cls.sta = 'CM.TEST'
        cls.ij = (6, 6)
        rows = [{'station': cls.sta, 'st_x_km': 60.0, 'st_y_km': 60.0, 'st_elev_km': 2.0}]
        cls.stations_e = pd.DataFrame(rows)
        cls.stations_0 = pd.DataFrame([{**rows[0], 'st_elev_km': 0.0}])
        cls.stations_noe = cls.stations_0.drop(columns='st_elev_km')

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def cache(self, stations):
        c = FieldCache(self.vp, self.grid, stations, Path(self.tmp.name), self.cfg)
        self.addCleanup(c.close)
        return c

    def test_vertical_exact_against_analytic(self):
        # Uniform 6 km/s; receiver 2 km above sea level, source 50 km below.
        # Exact travel time = 52/6 s. The grid (z>=0) gives 50/6; the
        # correction must supply exactly the missing 2/6 s.
        c = self.cache(self.stations_e)
        node = np.array([[5, *self.ij]])
        corr = c.elevation_correction(self.sta, node)
        np.testing.assert_allclose(corr, [2.0 / 6.0], rtol=1e-12)
        t_grid = float(c.get(self.sta)[5, *self.ij])
        np.testing.assert_allclose(t_grid + corr[0], 52.0 / 6.0, rtol=1e-12)

    def test_zero_elevation_and_missing_column_are_noops(self):
        c0 = self.cache(self.stations_0)
        cn = self.cache(self.stations_noe)
        node = np.array([[5, *self.ij], [0, 0, 0]])
        np.testing.assert_allclose(c0.elevation_correction(self.sta, node), 0., atol=1e-15)
        np.testing.assert_allclose(cn.elevation_correction(self.sta, node), 0., atol=1e-15)

    def test_horizontal_event_correction_below_vertical(self):
        # The farther the source is horizontally, the smaller the vertical
        # slowness component: 0 <= corr <= e/v, strictly inside for a
        # horizontally offset source.
        c = self.cache(self.stations_e)
        below = c.elevation_correction(self.sta, np.array([[5, *self.ij]]))[0]
        far = c.elevation_correction(self.sta, np.array([[0, 12, 12]]))[0]
        self.assertGreater(below, 0.)
        self.assertGreater(below, far)
        self.assertGreaterEqual(far, 0.)
        self.assertLessEqual(below, 2.0 / 6.0 + 1e-12)

    def test_jacobian_rows_unaffected_by_elevation(self):
        # Elevation is an additive pair-dependent term; the discrete Jacobian
        # (rows) must be identical with and without it.
        picks = pd.DataFrame({'event_id': ['e1', 'e2'], 'station': [self.sta, self.sta],
                              'observed_s': [8.0, 8.0]})
        coarse = Grid(self.grid.x, self.grid.y, self.grid.z)
        nodes = {'e1': (5, *self.ij), 'e2': (0, 12, 12)}
        g_e, _ = kernels(self.cache(self.stations_e), picks, nodes, coarse, 'adjoint_cpp')
        g_0, _ = kernels(self.cache(self.stations_0), picks, nodes, coarse, 'adjoint_cpp')
        np.testing.assert_allclose(g_e.toarray(), g_0.toarray(), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
