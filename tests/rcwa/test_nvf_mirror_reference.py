"""Optional independent half-cell FEM controls (run in the NGSolve environment)."""
import importlib.util
import unittest

import numpy as np

try:  # the NGSolve environment runs this file under unittest alone, without pytest
    import pytest
    pytestmark = pytest.mark.research
except ImportError:
    pass


@unittest.skipUnless(importlib.util.find_spec('ngsolve'), 'optional NGSolve environment')
class MirrorReferenceTests(unittest.TestCase):
    def test_closed_laminate_matches_analytic_fourier_fields_and_energy(self):
        from studies.rcwa.nvf_mirror_reference.nvf_mirror_reference import project_half_profile
        cap = 3
        c, metadata = project_half_profile(cap, h=.12, degree=3, quadrature=12, gap=0.)
        edges, eps = np.array([-350., -175., 31., 175., 350.]), np.array([1., 4., 9., 1.])
        def fourier(values):
            result = []
            for mode in range(-cap, cap+1):
                k = 2*np.pi*mode/700
                intervals = ((np.exp(-1j*k*edges[:-1])-np.exp(-1j*k*edges[1:]))/(1j*k*700)
                             if mode else np.diff(edges)/700)
                result.append(intervals@values)
            return np.asarray(result)
        ef, reciprocal = fourier(eps), fourier(1/eps)
        harmonic = 1/reciprocal[cap].real
        expected = np.zeros_like(c)
        expected[0, cap] = harmonic*reciprocal
        expected[3, cap, cap] = 1
        expected[4, cap] = harmonic**2*reciprocal
        expected[6, cap] = expected[7, cap] = ef
        np.testing.assert_allclose(c, expected, atol=2e-11, rtol=0)
        for side in ('upper', 'lower'):
            np.testing.assert_allclose(metadata['bounds'][side], np.diag([harmonic, ef[cap].real]), atol=2e-11, rtol=0)

    def test_positive_gap_obeys_parity_and_converged_fourier_quadrature(self):
        from studies.rcwa.nvf_mirror_reference.nvf_mirror_reference import project_half_profile
        c, metadata = project_half_profile(4, h=.1, degree=4, quadrature=10)
        refined, refined_metadata = project_half_profile(4, h=.1, degree=4, quadrature=14)
        signs = np.array([1, -1, -1, 1, 1, -1, 1, 1])
        np.testing.assert_allclose(c, signs[:, None, None]*c[:, ::-1], atol=2e-14, rtol=0)
        np.testing.assert_allclose(c, c[:, ::-1, ::-1].conj(), atol=2e-14, rtol=0)
        np.testing.assert_allclose(c, refined, atol=1e-10, rtol=0)
        np.testing.assert_allclose(c[:4, 4, 4].reshape(2, 2), np.eye(2), atol=2e-12, rtol=0)
        self.assertLess(metadata['bounds']['relative_gap'], 1e-4)
        self.assertLess(refined_metadata['mean_energy_quadrature_error'], 1e-10)

    def test_half_and_full_cell_energy_enclosures_are_consistent(self):
        from studies.rcwa.nvf_gap_reference.nvf_gap_reference import project_profile
        from studies.rcwa.nvf_mirror_reference.nvf_mirror_reference import project_half_profile
        _, full = project_profile(2, h=.1, degree=4, quadrature=12)
        _, half = project_half_profile(2, h=.1, degree=4, quadrature=12)
        for a, b in ((full, half), (half, full)):
            difference = np.array(a['bounds']['upper'])-np.array(b['bounds']['lower'])
            self.assertGreater(np.linalg.eigvalsh(difference).min(), -1e-10)

    def test_interface_quadrature_uses_its_own_element_and_matches_exact_material_fourier(self):
        from studies.rcwa.nvf_fem_quadrature.nvf_fem_quadrature import triangle_quadrature
        from studies.rcwa.nvf_gap_reference.nvf_gap_reference import make_profile_mesh
        from studies.rcwa.nvf_mirror_reference.nvf_mirror_reference import PERIOD, make_half_mesh
        orders = np.array([-112, -39, 0, 61, 112])
        height = PERIOD-.017
        expected = np.zeros((len(orders), len(orders)), complex)
        expected[2, 2] = 1.
        vertical = height/PERIOD*np.sinc(orders*height/PERIOD)
        for left, right, eps in ((-.175, .031, 4.), (.031, .175, 9.)):
            horizontal = ((right-left)/PERIOD*np.sinc(orders*(right-left)/PERIOD)
                          *np.exp(-1j*np.pi/PERIOD*orders*(right+left)))
            expected += (eps-1)*vertical[:, None]*horizontal[None]
        for fraction, maker in ((.5, make_half_mesh), (1., make_profile_mesh)):
            with self.subTest(area_fraction=fraction):
                mesh, materials = maker(h=.03)
                material = mesh.MaterialCF(materials, default=1.)
                coefficients, total_weight = np.zeros_like(expected), 0.
                for mapped, xy, weights in triangle_quadrature(mesh, 36, period=PERIOD, area_fraction=fraction):
                    self.assertLessEqual(len(weights), 24000)
                    eps = material(mapped).ravel()
                    ex = np.exp(-2j*np.pi/PERIOD*orders[:, None]*xy[:, 0])
                    phase = 2*np.pi/PERIOD*orders[:, None]*xy[:, 1]
                    ey = 2*np.cos(phase) if fraction == .5 else np.exp(-1j*phase)
                    coefficients += (ey*(weights*eps))@ex.T
                    total_weight += weights.sum()
                self.assertAlmostEqual(total_weight, fraction, places=14)
                np.testing.assert_allclose(coefficients, expected, atol=5e-13, rtol=0)

    def test_invalid_geometry_and_material_mapping_are_rejected(self):
        from studies.rcwa.nvf_mirror_reference.nvf_mirror_reference import half_correctors, make_half_mesh, project_half_profile
        for settings in (dict(gap=-1.), dict(gap=700.), dict(seams=[180.]), dict(values=[-1., 9.]),
                         dict(values=[4.+.1j, 9.])):
            with self.assertRaisesRegex(ValueError, 'half-cell profile'):
                make_half_mesh(**settings)
        with self.assertRaisesRegex(ValueError, 'quadrature'):
            project_half_profile(2, degree=4, quadrature=3)
        mesh, _ = make_half_mesh(h=.12)
        with self.assertRaisesRegex(ValueError, 'material mapping'):
            half_correctors(mesh, 3, {'piece_0': 4.})


if __name__ == '__main__':
    unittest.main()
