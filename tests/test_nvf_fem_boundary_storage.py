"""Small controls for storing only the affine Maxwell boundary response rows.

These use the optional internal NGSolve environment. The driven comparison
has interior H(curl) degrees of freedom, so an incorrect coupling-only
correction in the uncondensed solve cannot pass unnoticed.
"""
import unittest
from unittest.mock import patch

import numpy as np

try:
    import ngsolve as ng
except ImportError:
    ng = None


@unittest.skipIf(ng is None, 'optional ngsolve reference environment is absent')
class FEMBoundaryStorageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from studies import nvf_fem_reference as reference
        cls.reference = reference
        ng.SetNumThreads(1)

    def test_periodic_hcurl_boundary_rows_contain_every_retained_trace(self):
        mesh, _ = self.reference.make_affine_profile_mesh(.2)
        space = ng.Compress(ng.Periodic(ng.HCurl(mesh, order=3, complex=True)))
        test = space.TestFunction()
        boundary = np.asarray(list(space.GetDofs(
            mesh.Boundaries('outer_top|outer_bottom'))), dtype=bool)
        coupling = np.asarray(list(space.FreeDofs(coupling=True)), dtype=bool)
        self.assertTrue(np.any(boundary))
        self.assertFalse(np.any(boundary & ~coupling))
        self.assertLess(np.count_nonzero(coupling), space.ndof)
        reciprocal = 2*np.pi*np.linalg.inv([[.7, .315], [0., .49]]).T
        for side in ('outer_top', 'outer_bottom'):
            side_rows = np.asarray(list(space.GetDofs(mesh.Boundaries(side))), dtype=bool)
            for n in (-1, 0, 1):
                for m in (-1, 0, 1):
                    g = reciprocal@np.array([m, n])
                    phase = ng.exp(1j*(g[0]*ng.x+g[1]*ng.y))/np.sqrt(.343)
                    for component in (0, 1):
                        load = ng.LinearForm(space)
                        load += phase*test.Trace()[component]*ng.ds(side, bonus_intorder=8)
                        load.Assemble()
                        values = load.vec.FV().NumPy()
                        with self.subTest(side=side, mode=(m, n), component=component):
                            self.assertEqual(np.count_nonzero(values[~side_rows]), 0)
                            self.assertEqual(np.count_nonzero(values[~boundary]), 0)
                            self.assertGreater(np.linalg.norm(values), 0.)

    def test_driven_full_and_condensed_solutions_share_one_factorization(self):
        results = []
        original_inverse = ng.BaseMatrix.Inverse
        for condense in (False, True):
            factorizations, applications = [], []

            class CountedInverse:
                def __init__(self, matrix):
                    self.matrix = matrix

                def __mul__(self, rhs):
                    applications.append(None)
                    return self.matrix*rhs

            def inverse(matrix, *args, **kwargs):
                factorizations.append(None)
                return CountedInverse(original_inverse(matrix, *args, **kwargs))

            with patch.object(ng.BaseMatrix, 'Inverse', inverse):
                result = self.reference.solve_affine_rayleigh(
                    .2, 3, 1, theta=.31, phi=.23, condense=condense)
            results.append(result)
            with self.subTest(condense=condense):
                self.assertEqual(len(factorizations), 1)
                # Two sources, one inverse per trace, then two full corrections.
                self.assertEqual(len(applications), result['rank']+4)
                self.assertLess(result['boundary_dofs'], result['coupling_dofs'])
                self.assertLess(result['coupling_dofs'], result['dofs'])
                self.assertEqual(result['boundary_array_bytes'],
                                 2*result['boundary_dofs']*result['rank']*16)
                for pol in ('p', 's'):
                    self.assertLess(result[pol]['residual'], 1e-11)

        full, condensed = results
        self.assertEqual(full['modes'], condensed['modes'])
        for pol in ('p', 's'):
            for key in ('T', 'R', 'volume_T', 'volume_R', 'flux_disagreement',
                        'energy_error', 'energy_budget_residual'):
                self.assertAlmostEqual(full[pol][key], condensed[pol][key], delta=2e-10)
            for key in ('order_T', 'order_R'):
                np.testing.assert_allclose(full[pol][key], condensed[pol][key],
                                           rtol=0, atol=2e-10)
            for side in ('top', 'bottom', 'total'):
                self.assertAlmostEqual(full[pol]['robin_complement_power'][side],
                                       condensed[pol]['robin_complement_power'][side],
                                       delta=2e-10)


if __name__ == '__main__':
    unittest.main()
