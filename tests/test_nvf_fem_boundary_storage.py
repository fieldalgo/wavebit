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
        original_form = ng.BilinearForm
        original_compute = original_form.ComputeInternal
        original_apply = original_form.Apply
        for condense, recompute_internal in ((False, False), (True, False), (True, True)):
            factorizations, applications, forms, internal_calls, apply_calls = [], [], [], [], []
            events = []

            class CountedInverse:
                def __init__(self, matrix):
                    self.matrix = matrix

                def __mul__(self, rhs):
                    applications.append(None)
                    return self.matrix*rhs

            def inverse(matrix, *args, **kwargs):
                factorizations.append(None)
                return CountedInverse(original_inverse(matrix, *args, **kwargs))

            def progress(record):
                events.append((record, len(applications)))

            def form(*args, **kwargs):
                result = original_form(*args, **kwargs)
                forms.append(result)
                return result

            def compute_internal(form, vector, source):
                internal_calls.append((vector.FV().NumPy().copy(), source.FV().NumPy().copy()))
                return original_compute(form, vector, source)

            def apply(form, *args, **kwargs):
                apply_calls.append(None)
                return original_apply(form, *args, **kwargs)

            with (patch.object(ng.BaseMatrix, 'Inverse', inverse),
                  patch.object(ng, 'BilinearForm', form),
                  patch.object(original_form, 'ComputeInternal', compute_internal),
                  patch.object(original_form, 'Apply', apply)):
                result = self.reference.solve_affine_rayleigh(
                    .2, 3, 1, theta=.31, phi=.23, condense=condense,
                    recompute_internal=np.bool_(recompute_internal),
                    progress=progress, boundary_progress_every=np.int64(16) if condense else None)
            results.append(result)
            with self.subTest(condense=condense, recompute_internal=recompute_internal):
                self.assertEqual(len(factorizations), 1)
                self.assertEqual(len(forms), 1)
                # Two sources, one inverse per trace, then two full corrections.
                self.assertEqual(len(applications), result['rank']+4)
                self.assertIs(result['recompute_internal'], recompute_internal)
                self.assertIs(result['settings']['recompute_internal'], recompute_internal)
                self.assertEqual(len(internal_calls), 4 if recompute_internal else 0)
                self.assertEqual(len(apply_calls), 4 if recompute_internal else 2)
                if recompute_internal:
                    for name in ('harmonic_extension', 'harmonic_extension_trans',
                                 'inner_solve', 'inner_matrix'):
                        self.assertIsNone(getattr(forms[0], name), name)
                    for column in range(2):
                        # Source elimination starts from zero, and final
                        # reconstruction must receive the original volume load.
                        self.assertEqual(np.count_nonzero(internal_calls[column][0]), 0)
                        self.assertGreater(np.linalg.norm(internal_calls[column][1]), 0.)
                        np.testing.assert_array_equal(internal_calls[column][1],
                                                      internal_calls[column+2][1])
                elif condense:
                    for name in ('harmonic_extension', 'harmonic_extension_trans', 'inner_solve'):
                        self.assertIsNotNone(getattr(forms[0], name), name)
                self.assertLess(result['boundary_dofs'], result['coupling_dofs'])
                self.assertLess(result['coupling_dofs'], result['dofs'])
                self.assertEqual(result['boundary_array_bytes'],
                                 2*result['boundary_dofs']*result['rank']*16)
                for pol in ('p', 's'):
                    self.assertLess(result[pol]['residual'], 1e-11)
                stages = [record['stage'] for record, _ in events]
                if condense:
                    self.assertEqual(stages, ['mesh', 'factorized']
                                     + ['boundary_progress']*3 + ['boundary'])
                    self.assertEqual([record['boundary_columns_completed']
                                      for record, _ in events[2:-1]], [16, 32, 36])
                    for record, applied in events[2:-1]:
                        self.assertEqual(record['boundary_columns_total'], result['rank'])
                        self.assertEqual(applied, 2+record['boundary_columns_completed'])
                        for key in ('dofs', 'rank', 'settings', 'periodic_pairs'):
                            self.assertEqual(record[key], result[key])
                else:
                    self.assertEqual(stages, ['mesh', 'factorized', 'boundary'])

        full = results[0]
        for condensed in results[1:]:
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

    def test_recomputed_internal_driven_slab_matches_analytic_control(self):
        from tests.test_nvf_fem_affine_scattering import airy_slab

        result = self.reference.solve_affine_rayleigh(
            .18, 4, 1, uniform=4., phi=.37, recompute_internal=True)
        center = result['modes'].index([0, 0])
        for pol in ('p', 's'):
            exact = airy_slab(4., polarization=pol)
            for quantity in ('T', 'R'):
                powers = np.asarray(result[pol]['order_'+quantity])
                self.assertAlmostEqual(result[pol][quantity], exact[quantity], delta=5e-4)
                self.assertAlmostEqual(result[pol]['volume_'+quantity], exact[quantity], delta=5e-4)
                self.assertAlmostEqual(powers[center], exact[quantity], delta=5e-4)
                self.assertLess(np.sum(abs(np.delete(powers, center))), 5e-4)
            self.assertLess(result[pol]['residual'], 1e-11)

    def test_invalid_internal_storage_options_fail_before_meshing(self):
        invalid = [dict(recompute_internal=value)
                   for value in (0, 1, 0., 1., None, 'yes', [], np.nan)]
        invalid += [dict(condense=False, recompute_internal=True),
                    dict(condense=np.bool_(False), recompute_internal=np.bool_(True))]
        with patch.object(self.reference, 'make_affine_profile_mesh') as make_mesh:
            for settings in invalid:
                with self.subTest(settings=settings), self.assertRaisesRegex(
                        ValueError, 'recompute_internal'):
                    self.reference.solve_affine_rayleigh(**settings)
            make_mesh.assert_not_called()

    def test_boundary_callback_can_abort_before_the_next_inverse(self):
        original_inverse = ng.BaseMatrix.Inverse
        factorizations, applications, events = [], [], []

        class AbortReference(RuntimeError):
            pass

        class CountedInverse:
            def __init__(self, matrix):
                self.matrix = matrix

            def __mul__(self, rhs):
                applications.append(None)
                return self.matrix*rhs

        def inverse(matrix, *args, **kwargs):
            factorizations.append(None)
            return CountedInverse(original_inverse(matrix, *args, **kwargs))

        def progress(record):
            events.append(record)
            if record['stage'] == 'boundary_progress':
                raise AbortReference('caller resource guard')

        with patch.object(ng.BaseMatrix, 'Inverse', inverse), self.assertRaises(AbortReference):
            self.reference.solve_affine_rayleigh(
                .2, 2, 1, progress=progress, boundary_progress_every=8)
        self.assertEqual(len(factorizations), 1)
        self.assertEqual(len(applications), 2+8)
        self.assertEqual([record['stage'] for record in events],
                         ['mesh', 'factorized', 'boundary_progress'])
        self.assertEqual(events[-1]['boundary_columns_completed'], 8)
        self.assertEqual(events[-1]['boundary_columns_total'], 36)

    def test_source_callback_stops_after_completed_inverses_before_any_trace(self):
        original_inverse = ng.BaseMatrix.Inverse
        original_assemble = ng.LinearForm.Assemble
        original_compute = ng.BilinearForm.ComputeInternal

        class AbortReference(RuntimeError):
            pass

        for recompute_internal in (False, True):
            for stop_after in (1, 2):
                factorizations, applications, assemblies, reconstructions, events = [], [], [], [], []

                class EagerCountedInverse:
                    def __init__(self, matrix):
                        self.matrix = matrix

                    def __mul__(self, rhs):
                        result = rhs.CreateVector()
                        # Count only after the native solve has actually run;
                        # matrix*rhs alone would return a lazy expression.
                        result.data = self.matrix*rhs
                        applications.append(None)
                        return result

                def inverse(matrix, *args, **kwargs):
                    result = original_inverse(matrix, *args, **kwargs)
                    factorizations.append(None)
                    return EagerCountedInverse(result)

                def assemble(form, *args, **kwargs):
                    result = original_assemble(form, *args, **kwargs)
                    assemblies.append(None)
                    return result

                def compute(form, *args, **kwargs):
                    result = original_compute(form, *args, **kwargs)
                    reconstructions.append(None)
                    return result

                def progress(record):
                    events.append((record, len(applications), len(assemblies)))
                    if (record['stage'] == 'source_progress'
                            and record['source_columns_completed'] == stop_after):
                        raise AbortReference('caller source resource guard')

                with (self.subTest(recompute_internal=recompute_internal, stop_after=stop_after),
                      patch.object(ng.BaseMatrix, 'Inverse', inverse),
                      patch.object(ng.LinearForm, 'Assemble', assemble),
                      patch.object(ng.BilinearForm, 'ComputeInternal', compute),
                      self.assertRaises(AbortReference)):
                    self.reference.solve_affine_rayleigh(
                        .2, 3, 1, recompute_internal=recompute_internal,
                        progress=progress, source_progress=np.bool_(True), boundary_progress_every=1)
                self.assertEqual(len(factorizations), 1)
                self.assertEqual(len(applications), stop_after)
                self.assertEqual(len(reconstructions), stop_after if recompute_internal else 0)
                # Both original volume loads are assembled before factorization.
                # The first trace would be a third LinearForm assembly.
                self.assertEqual(len(assemblies), 2)
                self.assertEqual([record['stage'] for record, _, _ in events],
                                 ['mesh', 'factorized']+['source_progress']*stop_after)
                self.assertGreater(events[0][0]['dofs'], events[0][0]['coupling_dofs'])
                for count, (record, completed_inverses, assembled_loads) in enumerate(events[2:], 1):
                    self.assertEqual(record['source_columns_completed'], count)
                    self.assertEqual(record['source_columns_total'], 2)
                    self.assertEqual(record['polarization'], ('p', 's')[count-1])
                    self.assertEqual(completed_inverses, count)
                    self.assertEqual(assembled_loads, 2)

    def test_invalid_source_progress_flag_is_rejected_before_meshing(self):
        with patch.object(self.reference, 'make_affine_profile_mesh') as make_mesh:
            for enabled in (0, 1, 0., 1., None, 'yes', [], np.nan):
                with self.subTest(source_progress=enabled), self.assertRaisesRegex(ValueError, 'source_progress'):
                    self.reference.solve_affine_rayleigh(source_progress=enabled)
            make_mesh.assert_not_called()

    def test_invalid_boundary_progress_interval_is_rejected_before_meshing(self):
        with patch.object(self.reference, 'make_affine_profile_mesh') as make_mesh:
            for interval in (True, np.bool_(False), 0, -1, 1.5, '8', np.nan):
                with self.subTest(interval=interval), self.assertRaisesRegex(
                        ValueError, 'boundary_progress_every'):
                    self.reference.solve_affine_rayleigh(boundary_progress_every=interval)
            make_mesh.assert_not_called()


if __name__ == '__main__':
    unittest.main()
