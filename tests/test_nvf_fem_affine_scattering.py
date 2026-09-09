"""Independent controls for the optional physical affine Maxwell instrument.

The Airy slab calculation below uses scalar Fresnel coefficients and a round
trip through the film, independent of the FEM source and boundary assembly.
Run with standard-library unittest in the optional NGSolve environment.
"""
import unittest
from unittest.mock import patch

import numpy as np

try:
    import ngsolve as ng
except ImportError:
    ng = None


def airy_slab(epsilon, theta=.2, polarization='p'):
    """Unit-incident-power R/T of a .3-micrometer film at wavelength .9."""
    eps = np.array([1., epsilon, 2.25])
    normal = np.sqrt(eps.astype(complex)-np.sin(theta)**2)
    admittance = eps/normal if polarization == 'p' else normal
    first = (admittance[0]-admittance[1])/(admittance[0]+admittance[1])
    second = (admittance[1]-admittance[2])/(admittance[1]+admittance[2])
    phase = np.exp(2j*np.pi/.9*normal[1]*.3)
    denominator = 1+first*second*phase**2
    reflected = (first+second*phase**2)/denominator
    transmitted = (1+first)*(1+second)*phase/denominator
    return dict(R=float(abs(reflected)**2),
                T=float(admittance[2].real/admittance[0].real*abs(transmitted)**2))


@unittest.skipIf(ng is None, 'optional ngsolve reference environment is absent')
class AffineScatteringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from studies import nvf_fem_reference as reference
        cls.reference = reference
        ng.SetNumThreads(1)

    def test_zero_contrast_both_polarizations_and_oblique_incidence(self):
        # The scattered solution vanishes identically, including at nonzero phi.
        for theta, phi, transform, cut in (
                (.2, .37, [[1., .45], [0., .7]], 1),
                (.31, -.23, [[.9, -.25], [.3, 1.1]], 2),
                (0., .53, [[1., .45], [0., .7]], 1)):
            with self.subTest(theta=theta, phi=phi):
                result = self.reference.solve_affine_rayleigh(
                    .2, 1, cut, uniform=1., theta=theta, phi=phi, transform=transform)
                area = .49*np.linalg.det(transform)
                self.assertAlmostEqual(result['area'], area, delta=1e-14)
                for pol in ('p', 's'):
                    exact = airy_slab(1., theta, pol)
                    for quantity in ('T', 'R'):
                        self.assertAlmostEqual(result[pol][quantity], exact[quantity], delta=2e-10)
                        self.assertAlmostEqual(result[pol]['volume_'+quantity], exact[quantity], delta=2e-10)
                    self.assertLess(result[pol]['energy_error'], 2e-10)
                    self.assertLess(result[pol]['residual'], 1e-12)

    def test_driven_slab_matches_independent_airy_powers_and_specular_labels(self):
        events = []
        factorizations = []
        original_inverse = ng.BaseMatrix.Inverse
        def inverse(matrix, *args, **kwargs):
            factorizations.append(matrix)
            return original_inverse(matrix, *args, **kwargs)
        with patch.object(ng.BaseMatrix, 'Inverse', inverse):
            result = self.reference.solve_affine_rayleigh(
                .18, 4, 1, uniform=4., phi=.37, progress=events.append)
        # One physical matrix supplies both independent incident polarizations.
        self.assertEqual(len(factorizations), 1)
        self.assertEqual(sum(event['stage'] == 'factorized' for event in events), 1)
        modes = [tuple(mode) for mode in result['modes']]
        center = modes.index((0, 0))
        for pol in ('p', 's'):
            exact = airy_slab(4., polarization=pol)
            for quantity in ('T', 'R'):
                power = np.asarray(result[pol]['order_'+quantity])
                self.assertEqual(power.shape, (len(modes),))
                self.assertTrue(np.isfinite(power).all())
                self.assertAlmostEqual(result[pol][quantity], exact[quantity], delta=5e-4)
                self.assertAlmostEqual(result[pol]['volume_'+quantity], exact[quantity], delta=5e-4)
                self.assertAlmostEqual(power[center], exact[quantity], delta=5e-4)
                self.assertLess(np.sum(abs(np.delete(power, center))), 5e-4)
            self.assertLess(result[pol]['flux_disagreement'], 5e-4)
            self.assertLess(result[pol]['energy_error'], 5e-4)
            self.assertLess(result[pol]['residual'], 1e-9)

    def test_condensation_preserves_both_physical_solutions(self):
        settings = dict(h=.2, order=2, cut=1, uniform=4., theta=.31, phi=.23)
        full = self.reference.solve_affine_rayleigh(**settings, condense=False)
        condensed = self.reference.solve_affine_rayleigh(**settings, condense=True)
        self.assertEqual(full['modes'], condensed['modes'])
        for pol in ('p', 's'):
            for quantity in ('T', 'R', 'volume_T', 'volume_R'):
                self.assertAlmostEqual(full[pol][quantity], condensed[pol][quantity], delta=2e-10)
            for quantity in ('order_T', 'order_R'):
                np.testing.assert_allclose(full[pol][quantity], condensed[pol][quantity],
                                           rtol=0, atol=2e-10)
            self.assertLess(condensed[pol]['residual'], 1e-9)

    def test_invalid_scattering_inputs_fail_before_meshing(self):
        invalid = [dict(order=value) for value in (True, np.bool_(True), 0, -1, 1.5)]
        invalid += [dict(cut=value) for value in (True, -1, 1.5, 0)]
        invalid += [dict(theta=value) for value in
                    (np.nan, np.inf, np.pi/2, -np.pi/2, np.pi, 1j, [0.], True)]
        invalid += [dict(phi=value) for value in (np.nan, np.inf, 1j, [0.], True)]
        invalid += [dict(quadrature_bonus=value) for value in
                    (True, np.bool_(True), -1, 1.5, 6., np.nan, np.inf, 1j, [6], '6')]
        with patch.object(self.reference, 'make_affine_profile_mesh',
                          side_effect=AssertionError('invalid scattering request reached the mesher')):
            for settings in invalid:
                with self.subTest(settings=settings), self.assertRaises(ValueError):
                    self.reference.solve_affine_rayleigh(**settings)
        invalid_edges = [dict(edge_h=value) for value in
                         (0., -.1, .21, np.nan, np.inf, True, np.bool_(True), 1j, [.1], '.1')]
        invalid_edges += [dict(edge_h=.1, uniform=4.)]
        with patch.object(self.reference.occ, 'OCCGeometry',
                          side_effect=AssertionError('invalid edge sizing reached mesh generation')):
            for settings in invalid_edges:
                with self.subTest(settings=settings), self.assertRaises(ValueError):
                    self.reference.solve_affine_rayleigh(**settings)


if __name__ == '__main__':
    unittest.main()
