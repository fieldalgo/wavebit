"""Optional independent FEM controls for affine reference material profiles."""
import importlib.util
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec('ngsolve'), 'optional NGSolve environment')
class AffineReferenceTests(unittest.TestCase):
    def test_uniform_cell_is_isotropic_for_sheared_and_rotated_transforms(self):
        from studies.nvf_cell_corrector import make_mesh, project_mesh
        mesh = make_mesh(.15, 'uniform')
        for transform in (np.array([[1., .45], [0., .7]]), np.array([[.2, -.7], [1., .45]])):
            c, metadata = project_mesh(mesh, 2, 2, 10, 4., transform=transform)
            expected = np.zeros_like(c)
            expected[[0, 3], 2, 2] = 1
            expected[[4, 6, 7], 2, 2] = 4
            np.testing.assert_allclose(c, expected, atol=2e-12, rtol=0)
            np.testing.assert_allclose(metadata['bounds']['upper'], 4*np.eye(2), atol=2e-12, rtol=0)
            np.testing.assert_allclose(metadata['bounds']['lower'], 4*np.eye(2), atol=2e-12, rtol=0)

    def test_closed_affine_laminate_has_exact_fields_energy_and_bounds(self):
        from studies.nvf_gap_reference import project_profile
        transform = np.array([[1., .45], [0., .7]])
        cap = 3
        c, metadata = project_profile(cap, .12, 3, 12, gap=0., transform=transform)
        normal = np.linalg.solve(transform.T, [1., 0.])
        normal /= np.linalg.norm(normal)
        nn, identity = np.outer(normal, normal), np.eye(2)
        n, m = np.meshgrid(np.arange(-cap, cap+1), np.arange(-cap, cap+1), indexing='ij')
        delta = ((n == 0) & (m == 0)).astype(float)
        epsilon, rho = delta.astype(complex), delta.astype(complex)
        for left, right, value in ((-175., 31., 4.), (31., 175., 9.)):
            span, middle = (right-left)/700, (right+left)/1400
            shape = span*np.sinc(m*span)*np.exp(-2j*np.pi*m*middle)*(n == 0)
            epsilon += (value-1)*shape
            rho += (1/value-1)*shape
        mean_rho = rho[cap, cap].real
        field = delta[..., None, None]*identity+(rho/mean_rho-delta)[..., None, None]*nn
        gram = epsilon[..., None, None]*(identity-nn)+(rho/mean_rho**2)[..., None, None]*nn
        expected = np.stack([field[..., 0, 0], field[..., 0, 1], field[..., 1, 0], field[..., 1, 1],
                             gram[..., 0, 0], gram[..., 0, 1], gram[..., 1, 1], epsilon])
        np.testing.assert_allclose(c, expected, atol=3e-11, rtol=0)
        for key in ('upper', 'lower'):
            np.testing.assert_allclose(metadata['bounds'][key], gram[cap, cap], atol=3e-11, rtol=0)
        self.assertEqual(metadata['fixture'], 'affine_partitioned_gap')

    def test_physical_rotation_and_scale_preserve_the_reference_field(self):
        from studies.nvf_gap_reference import make_profile_mesh
        from studies.nvf_cell_corrector import project_mesh
        mesh, materials = make_profile_mesh(.12)
        transform = np.array([[1., .45], [0., .7]])
        angle = .37
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        first, metadata = project_mesh(mesh, 2, 3, 10, materials, transform=transform)
        rotated, rm = project_mesh(mesh, 2, 3, 10, materials, transform=1.7*rotation@transform)
        field = first[:4].reshape(2, 2, 5, 5).transpose(2, 3, 0, 1)
        gram = np.stack([np.stack([first[4], first[5]], -1), np.stack([first[5], first[6]], -1)], -2)
        expected_field, expected_gram = rotation@field@rotation.T, rotation@gram@rotation.T
        expected = np.stack([expected_field[..., 0, 0], expected_field[..., 0, 1], expected_field[..., 1, 0],
                             expected_field[..., 1, 1], expected_gram[..., 0, 0], expected_gram[..., 0, 1],
                             expected_gram[..., 1, 1], first[7]])
        np.testing.assert_allclose(rotated, expected, atol=3e-11, rtol=0)
        for key in ('upper', 'lower'):
            np.testing.assert_allclose(rm['bounds'][key], rotation@metadata['bounds'][key]@rotation.T, atol=3e-11, rtol=0)

    def test_affine_material_projection_and_quadrature_are_independent(self):
        from studies.nvf_gap_reference import make_profile_mesh
        from studies.nvf_cell_corrector import project_mesh
        mesh, materials = make_profile_mesh(.1)
        transform = np.array([[1., .45], [0., .7]])
        first, _ = project_mesh(mesh, 6, 4, 10, materials, transform=transform)
        second, _ = project_mesh(mesh, 6, 4, 14, materials, transform=transform)
        np.testing.assert_allclose(first, second, atol=2e-10, rtol=0)
        n, m = np.meshgrid(np.arange(-6, 7), np.arange(-6, 7), indexing='ij')
        expected = ((m == 0) & (n == 0)).astype(complex)
        height = 1-17/700
        for left, right, value in ((-175., 31., 4.), (31., 175., 9.)):
            span, middle = (right-left)/700, (right+left)/1400
            expected += (value-1)*span*height*np.sinc(m*span)*np.sinc(n*height)*np.exp(-2j*np.pi*m*middle)
        np.testing.assert_allclose(second[7], expected, atol=2e-12, rtol=0)

    def test_invalid_transforms_raise(self):
        from studies.nvf_gap_reference import make_profile_mesh
        from studies.nvf_cell_corrector import correctors
        mesh, materials = make_profile_mesh(.15)
        for transform in (np.eye(3), [[1., 0.], [0., -1.]], [[1., 1.], [0., 0.]],
                          [[1., np.nan], [0., 1.]], [[1., .2j], [0., 1.]]):
            with self.assertRaisesRegex(ValueError, 'transform'):
                correctors(mesh, 2, materials, transform=transform)


if __name__ == '__main__':
    unittest.main()
