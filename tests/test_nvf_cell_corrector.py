"""Optional NGSolve instrument checks; runnable with standard-library unittest."""
import importlib.util
import unittest

import numpy as np

try:  # the NGSolve environment runs this file under unittest alone, without pytest
    import pytest
    pytestmark = pytest.mark.research
except ImportError:
    pass

AVAILABLE = importlib.util.find_spec('ngsolve') is not None
if AVAILABLE:
    import ngsolve as ng
    from studies.nvf_cell_corrector import make_mesh, correctors, project
    from studies.nvf_cell_bounds import energy_bounds, projected_matrices
    from studies.nvf_fem_reference import periodic_counts


@unittest.skipUnless(AVAILABLE, 'optional NGSolve cell instrument')
class CellCorrectorTests(unittest.TestCase):
    def test_geometry_has_both_periodic_axes_and_correct_material(self):
        mesh = make_mesh(.12, 'tilted')
        counts = periodic_counts(mesh)
        self.assertGreater(min(counts.values()), 0)
        self.assertLess(ng.Compress(ng.Periodic(ng.H1(mesh, order=2))).ndof,
                        ng.H1(mesh, order=2).ndof)
        self.assertAlmostEqual(ng.Integrate(1, mesh, definedon=mesh.Materials('inclusion')),
                               .64*.7/np.sqrt(5), places=12)

    def test_uniform_and_bar_energy_are_analytic(self):
        for fixture, exact in [('uniform', 4*np.eye(2)), ('bar', np.diag([1/.625, 2.5]))]:
            mesh = make_mesh(.12, fixture)
            primal = correctors(mesh, 2, 4.)
            reciprocal = correctors(mesh, 2, 4., reciprocal=True)
            result = energy_bounds(primal.energy, reciprocal.energy)
            for key in ('upper', 'lower'):
                np.testing.assert_allclose(result[key], exact, atol=2e-12, rtol=0)
            self.assertLess(max(primal.residuals+reciprocal.residuals), 1e-12)

    def test_bar_projection_matches_exact_piecewise_field_and_inverse_rule(self):
        cap = 4
        coefficients, metadata = project(cap, .12, 2, 12, 1., 'bar')
        n, m = np.meshgrid(np.arange(-cap, cap+1), np.arange(-cap, cap+1), indexing='ij')
        indicator = .5*np.sinc(.5*m)*np.sinc(n)
        delta = ((m == 0) & (n == 0)).astype(float)
        epsilon, reciprocal = delta+11.25*indicator, delta+(1/12.25-1)*indicator
        harmonic = 1/(.5+.5/12.25)
        expected = np.stack([harmonic*reciprocal, 0*delta, 0*delta, delta,
                              harmonic**2*reciprocal, 0*delta, epsilon, epsilon])
        np.testing.assert_allclose(coefficients, expected, atol=2e-11, rtol=0)
        actual_slots = projected_matrices(coefficients, 2)
        exact_slots = projected_matrices(expected, 2)
        for actual, want in zip(actual_slots, exact_slots):
            np.testing.assert_allclose(actual, want, atol=2e-10, rtol=0)
        self.assertLess(metadata['mean_energy_quadrature_error'], 1e-11)

    def test_tilted_projection_quadrature_is_refined_independently(self):
        first, metadata = project(8, .06, 4, 12, 5.)
        second, _ = project(8, .06, 4, 16, 5.)
        np.testing.assert_allclose(first, second, atol=2e-10, rtol=0)
        self.assertLess(metadata['bounds']['relative_gap'], .001)

    def test_invalid_discretizations_are_rejected(self):
        with self.assertRaises(ValueError):
            make_mesh(.4)
        with self.assertRaises(ValueError):
            project(4, degree=4, quadrature=2)
        with self.assertRaises(ValueError):
            make_mesh(.1, 'unknown')


if __name__ == '__main__':
    unittest.main()
