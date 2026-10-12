"""Numerical equivalence gates for the new C++ kernel; no network access."""
import unittest
import numpy as np
from relocation import score_candidates


class Equivalence(unittest.TestCase):
    def check_case(self, times, obs):
        expected = score_candidates(times, obs, 'reference')
        for backend in ('numpy', 'cpp'):
            actual = score_candidates(times, obs, backend)
            self.assertEqual(expected[0], actual[0])
            for e, a in zip(expected[1:], actual[1:]):
                np.testing.assert_allclose(a, e, atol=1e-12, rtol=1e-12,
                                           equal_nan=True)

    def test_random_odd_even_and_static_terms(self):
        rng = np.random.default_rng(20261009)
        for n in (1, 4, 5, 8, 25):
            for c in (1, 27, 125):
                for _ in range(10):
                    times = rng.uniform(0, 150, (c, n))
                    times += rng.normal(0, 0.4, n)  # station statics
                    obs = rng.uniform(0, 150, n)
                    self.check_case(times, obs)

    def test_ties_nonfinite_and_noncontiguous(self):
        self.check_case(np.ones((27, 5)), np.arange(5.))
        self.check_case(np.full((4, 5), np.nan), np.arange(5.))
        t = np.arange(50.).reshape(5, 10)[:, ::2]
        t[0, 1] = np.inf
        t[1, 2] = np.nan
        self.check_case(t, np.arange(10.)[::2])

    def test_invalid_inputs(self):
        for backend in ('reference', 'numpy', 'cpp'):
            for t, obs in ((np.ones((2, 3)), [1, 2]),
                           (np.ones((2, 3)), [1, np.nan, 3]),
                           (np.ones((0, 3)), [1, 2, 3])):
                with self.assertRaises(ValueError):
                    score_candidates(t, obs, backend)

    def test_known_origin_shift_and_unique_hypocenter(self):
        # Candidate 1 explains observations exactly after a 2 s shift;
        # the other candidates have station-dependent discrepancies.
        t = np.array([[9., 11., 14., 20.], [10., 12., 15., 18.],
                      [11., 14., 16., 22.]])
        obs = np.array([12., 14., 17., 20.])
        for backend in ('reference', 'numpy', 'cpp'):
            best, shift, rms = score_candidates(t, obs, backend)
            self.assertEqual(best, 1)
            self.assertEqual(shift[best], 2.)
            self.assertEqual(rms[best], 0.)


if __name__ == '__main__':
    unittest.main()
