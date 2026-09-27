"""Optional Parseval controls for the affine FEM Robin complement diagnostic.

Only analytic coefficient functions are integrated on a small mesh. No Maxwell
matrix or scattering solution is constructed by these tests.
"""
import unittest

import numpy as np

try:  # the NGSolve environment runs this file under unittest alone, without pytest
    import pytest
    pytestmark = pytest.mark.research
except ImportError:
    pass

try:
    import ngsolve as ng
except ImportError:
    ng = None


@unittest.skipIf(ng is None, 'optional ngsolve reference environment is absent')
class FEMBoundaryPowerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from studies import nvf_fem_reference as reference
        cls.reference = reference
        ng.SetNumThreads(1)
        cls.transform = np.array([[1., .45], [0., .7]])
        cls.area = .343
        cls.mesh, _ = reference.make_affine_profile_mesh(uniform=4.)
        cls.theta = .31
        # First row is top, second row bottom, each in physical x/y components.
        cls.constant = np.array([[.3+.1j, -.2+.4j], [.2-.3j, .1+.2j]])
        cls.oscillatory = np.array([[.15-.1j, .2+.25j], [-.25+.05j, .08-.12j]])

    @staticmethod
    def analytic_field(transform, constant, oscillatory):
        reciprocal = 2*np.pi*np.linalg.inv(.7*transform).T
        # (2,1) is outside a cut-one trace and is evanescent in both exteriors.
        g = reciprocal@np.array([2., 1.])
        phase = ng.exp(1j*(g[0]*ng.x+g[1]*ng.y))
        top_weight = (ng.z+.1)/.5
        components = [top_weight*constant[0, i]+(1-top_weight)*constant[1, i]
                      +(top_weight*oscillatory[0, i]+(1-top_weight)*oscillatory[1, i])*phase
                      for i in range(2)]
        # A large normal component must not enter the tangential power norm.
        return ng.CF((*components, 99.+10j))

    def diagnose(self, field, ports, *, mesh=None, area=None):
        return self.reference.robin_complement_power(
            self.mesh if mesh is None else mesh, field, ports,
            area=self.area if area is None else area, theta=self.theta, quadrature_order=24)

    def test_omitted_fourier_mode_has_parseval_power_on_both_sides(self):
        field = self.analytic_field(self.transform, self.constant, self.oscillatory)
        ports = np.sqrt(self.area)*self.constant[:, None, :]
        actual = self.diagnose(field, ports)
        # |b_top|^2=.135, |b_bottom|^2=.0858; area cancels, the substrate
        # Robin weight is sqrt(2.25)=1.5, and the incident flux is cos(theta).
        top, bottom = .135/np.cos(.31), 1.5*.0858/np.cos(.31)
        self.assertAlmostEqual(actual['top'], top, delta=2e-12)
        self.assertAlmostEqual(actual['bottom'], bottom, delta=2e-12)
        self.assertAlmostEqual(actual['total'], top+bottom, delta=3e-12)

    def test_retaining_the_nonconstant_mode_removes_the_complement(self):
        field = self.analytic_field(self.transform, self.constant, self.oscillatory)
        ports = np.sqrt(self.area)*np.stack((self.constant, self.oscillatory), axis=1)
        actual = self.diagnose(field, ports)
        for value in actual.values():
            self.assertAlmostEqual(value, 0., delta=3e-12)

    def test_adding_a_retained_constant_annex_leaves_the_complement_unchanged(self):
        annex = np.array([.2+.3j, -.4+.1j])
        field = self.analytic_field(self.transform, self.constant, self.oscillatory)
        ports = np.sqrt(self.area)*self.constant[:, None, :]
        before = self.diagnose(field, ports)
        after = self.diagnose(field+ng.CF((*annex, 0.)), ports+np.sqrt(self.area)*annex)
        for side in ('top', 'bottom', 'total'):
            self.assertAlmostEqual(before[side], after[side], delta=3e-12)

    def test_physical_rotation_and_common_scale_preserve_normalized_power(self):
        angle, scale = .41, 1.17
        rotation = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
        transform = scale*rotation@self.transform
        area = .343*scale**2
        constant, oscillatory = self.constant@rotation.T, self.oscillatory@rotation.T
        mesh, _ = self.reference.make_affine_profile_mesh(transform=transform, uniform=4.)
        field = self.analytic_field(transform, constant, oscillatory)
        actual = self.diagnose(field, np.sqrt(area)*constant[:, None, :], mesh=mesh, area=area)
        self.assertAlmostEqual(actual['top'], .135/np.cos(.31), delta=2e-12)
        self.assertAlmostEqual(actual['bottom'], 1.5*.0858/np.cos(.31), delta=2e-12)

    def test_projection_norm_overshoot_remains_signed(self):
        constant = np.ones((2, 2), complex)
        field = self.analytic_field(self.transform, constant, np.zeros((2, 2)))
        ports = np.sqrt(self.area)*constant[:, None, :]*(1+1e-8)
        actual = self.diagnose(field, ports)
        top = 2*(1-(1+1e-8)**2)/np.cos(.31)
        self.assertLess(actual['top'], 0.)
        self.assertLess(actual['bottom'], 0.)
        self.assertAlmostEqual(actual['top'], top, delta=2e-13)
        self.assertAlmostEqual(actual['bottom'], 1.5*top, delta=2e-13)

    def test_invalid_diagnostic_inputs_are_rejected(self):
        field = ng.CF((0., 0., 0.))
        options = dict(area=self.area, theta=.31, quadrature_order=12)
        ports = np.zeros((2, 1, 2), complex)
        for name, values in (('area', (0., -1., np.inf, 1j, True)),
                             ('theta', (np.nan, np.pi/2, [0.])),
                             ('quadrature_order', (-1, 1.5, True))):
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    self.reference.robin_complement_power(
                        self.mesh, field, ports, **(options | {name: value}))
        for invalid in (np.zeros((2, 2)), np.zeros((1, 1, 2)), np.zeros((2, 0, 2)),
                        np.full((2, 1, 2), np.nan), np.zeros((2, 1, 2), dtype=bool)):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                self.reference.robin_complement_power(self.mesh, field, invalid, **options)


if __name__ == '__main__':
    unittest.main()
