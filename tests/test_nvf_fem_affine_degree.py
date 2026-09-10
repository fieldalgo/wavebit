"""Physical controls for local-degree affine Maxwell reference spaces."""
import unittest

import numpy as np

from tests.test_nvf_fem_affine_scattering import airy_slab

try:
    import ngsolve as ng
except ImportError:
    ng = None


@unittest.skipIf(ng is None, 'optional ngsolve reference environment is absent')
class AffineDegreeScatteringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from studies.nvf_fem_reference import solve_affine_rayleigh
        cls.solve = staticmethod(solve_affine_rayleigh)
        ng.SetNumThreads(1)

    def test_local_space_preserves_condensation_and_quadrature_controls(self):
        settings = dict(h=.2, order=2, film_degree=3, film_degree_pad=.05,
                        cut=1, theta=.31, phi=.23)
        baseline = self.solve(**settings)
        self.assertEqual(baseline['max_order'], 3)
        self.assertGreater(baseline['space_refinement']['selected_entities']['cells'], 0)
        self.assertLess(baseline['coupling_dofs'], baseline['dofs'])
        for options in (dict(condense=False), dict(quadrature_bonus=14),
                        dict(recompute_internal=True)):
            result = self.solve(**settings, **options)
            self.assertEqual(result['recompute_internal'], options.get('recompute_internal', False))
            self.assertEqual(result['modes'], baseline['modes'])
            self.assertEqual(result['space_refinement'], baseline['space_refinement'])
            for pol in ('p', 's'):
                self.assertLess(result[pol]['residual'], 1e-9)
                for key in ('R', 'T', 'volume_R', 'volume_T', 'order_R', 'order_T',
                            'energy_budget_residual'):
                    with self.subTest(options=options, polarization=pol, quantity=key):
                        np.testing.assert_allclose(result[pol][key], baseline[pol][key],
                                                   rtol=0, atol=2e-10)
                for side in ('top', 'bottom', 'total'):
                    self.assertAlmostEqual(result[pol]['robin_complement_power'][side],
                                           baseline[pol]['robin_complement_power'][side],
                                           delta=2e-10)

    def test_enriched_driven_slab_matches_independent_airy_powers(self):
        result = self.solve(h=.18, order=4, film_degree=5, film_degree_pad=.05,
                            uniform=4., cut=1, phi=.37)
        self.assertEqual(result['max_order'], 5)
        for pol in ('p', 's'):
            exact = airy_slab(4., polarization=pol)
            for side in ('R', 'T'):
                for quantity in (side, 'volume_'+side):
                    self.assertAlmostEqual(result[pol][quantity], exact[side], delta=5e-4)
            self.assertLess(result[pol]['flux_disagreement'], 5e-4)
            self.assertLess(result[pol]['residual'], 1e-9)


if __name__ == '__main__':
    unittest.main()
