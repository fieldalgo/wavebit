"""Optional static FEM controls; runnable with unittest in the FEM environment."""
import importlib.util
import unittest

import numpy as np

try:  # the NGSolve environment runs this file under unittest alone, without pytest
    import pytest
    pytestmark = pytest.mark.research
except ImportError:
    pass


@unittest.skipUnless(importlib.util.find_spec('ngsolve'), 'optional NGSolve environment')
class GapReferenceTests(unittest.TestCase):
    def test_closed_fitted_laminate_recovers_exact_tensor_and_mean_field(self):
        from studies.rcwa.nvf_gap_reference.nvf_gap_reference import project_profile
        c, metadata = project_profile(2, h=.12, degree=3, quadrature=8, gap=0.)
        exact = np.diag([1/(.5+206/700/4+144/700/9), .5+206/700*4+144/700*9])
        np.testing.assert_allclose(c[:4, 2, 2].reshape(2, 2), np.eye(2), atol=2e-12, rtol=0)
        for side in ('upper', 'lower'):
            np.testing.assert_allclose(metadata['bounds'][side], exact, atol=2e-11, rtol=0)

    def test_positive_gap_has_consistent_finite_energy_enclosure(self):
        from studies.rcwa.nvf_gap_reference.nvf_gap_reference import project_profile
        c, metadata = project_profile(2, h=.1, degree=4, quadrature=10)
        np.testing.assert_allclose(c[:4, 2, 2].reshape(2, 2), np.eye(2), atol=2e-12, rtol=0)
        self.assertLess(metadata['bounds']['relative_gap'], 1e-4)
        self.assertLess(metadata['mean_energy_quadrature_error'], 1e-10)

    def test_invalid_geometry_and_incomplete_material_mapping_are_rejected(self):
        from studies.rcwa.nvf_gap_reference.nvf_gap_reference import make_profile_mesh
        from studies.rcwa.nvf_cell_corrector.nvf_cell_corrector import correctors
        with self.assertRaisesRegex(ValueError, 'profile'):
            make_profile_mesh(seams=[180.])
        mesh, _ = make_profile_mesh(h=.15)
        with self.assertRaisesRegex(ValueError, 'mapping'):
            correctors(mesh, 2, {'piece_0': 4.})


if __name__ == '__main__':
    unittest.main()
