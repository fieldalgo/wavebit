"""Optional FEM reference checks; runnable with Python's unittest alone.

Run in the NGSolve environment:
    PYTHONPATH=$PWD python -m unittest discover -s tests -p test_nvf_fem_reference.py
The normal Wavebit test environment can skip this optional dependency.
"""
import unittest

import numpy as np

try:
    import ngsolve as ng
except ImportError:
    ng = None


@unittest.skipIf(ng is None, 'optional ngsolve reference environment is absent')
class FEMReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from studies import nvf_fem_reference as reference
        cls.reference = reference
        ng.SetNumThreads(2)
        cls.mesh = reference.make_mesh(.2)

    def test_actual_periodic_pairs_in_both_directions(self):
        counts = self.reference.periodic_counts(self.mesh)
        self.assertGreater(counts['x'], 0)
        self.assertGreater(counts['y'], 0)
        ordinary = ng.HCurl(self.mesh, order=1, complex=True)
        periodic = ng.Compress(ng.Periodic(ordinary))
        self.assertLess(periodic.ndof, ordinary.ndof)

    def test_empty_identification_request_is_rejected(self):
        from types import SimpleNamespace
        mesh = SimpleNamespace(ngmesh=SimpleNamespace(GetIdentifications=lambda: []))
        with self.assertRaisesRegex(ValueError, 'missing periodic'):
            self.reference.periodic_counts(mesh)

    def test_either_missing_direction_is_rejected(self):
        from types import SimpleNamespace
        original = self.mesh.ngmesh
        for axis in (0, 1):
            pairs = [(a, b) for a, b in original.GetIdentifications()
                     if abs(original[b].p[axis]-original[a].p[axis]) > .6]
            class OneAxisMesh:
                def GetIdentifications(self):
                    return pairs
                def __getitem__(self, point):
                    return original[point]
            with self.assertRaisesRegex(ValueError, 'missing periodic'):
                self.reference.periodic_counts(SimpleNamespace(ngmesh=OneAxisMesh()))

    def test_material_volume_and_tilted_periodic_images(self):
        mesh = self.mesh
        volume = ng.Integrate(1, mesh, definedon=mesh.Materials('inclusion'))
        self.assertAlmostEqual(volume, .64*.7/np.sqrt(5)*.3, delta=1e-11)
        epsilon = mesh.MaterialCF({'inclusion': 12.25}, default=1.)
        co, si = 2/np.sqrt(5), 1/np.sqrt(5)
        # Includes the small inclusion pieces on both sides of the periodic cell.
        for x, y in [(.349, .16), (-.349, -.16), (.2, .1), (.2, -.25), (0., .27)]:
            inside = False
            for shift in (-.7, 0., .7):
                u = co*(x-shift)+si*y
                v = -si*(x-shift)+co*y
                inside |= abs(u) < .32 and abs(v) < .35/np.sqrt(5)
            self.assertAlmostEqual(epsilon(mesh(x, y, .15)), 12.25 if inside else 1.)

    def test_exact_slab_control(self):
        result = self.reference.slab_efficiencies(4.)
        self.assertAlmostEqual(result['T'], .838052025532751, delta=2e-14)
        self.assertAlmostEqual(result['T']+result['R'], 1., delta=2e-14)

    def test_absent_inclusion_reduces_to_bare_interface(self):
        result = self.reference.solve(.25, 1, uniform=1.)
        self.assertAlmostEqual(result['T'], .9621399115785346, delta=2e-10)
        self.assertAlmostEqual(result['R'], .03786008842146582, delta=2e-10)
        self.assertLess(result['residual'], 1e-12)

    def test_driven_slab_agrees_with_analytic(self):
        result = self.reference.solve(.18, 3, uniform=4.)
        self.assertLess(result['analytic_T_error'], .005)
        self.assertLess(result['energy_error'], 5e-4)
        self.assertLess(result['residual'], 1e-10)

    def test_condensed_system_recovers_full_system(self):
        full = self.reference.solve(.25, 2, uniform=4., condense=False)
        condensed = self.reference.solve(.25, 2, uniform=4., condense=True)
        self.assertAlmostEqual(full['T'], condensed['T'], delta=2e-11)
        self.assertAlmostEqual(full['R'], condensed['R'], delta=2e-11)
        self.assertLess(condensed['residual'], 1e-10)

    def test_rayleigh_cut_cannot_omit_propagating_substrate_order(self):
        with self.assertRaisesRegex(ValueError, 'omits a propagating'):
            self.reference.rayleigh_data(0, 2.25)
        modes, blocks = self.reference.rayleigh_data(1, 2.25)
        grazing = blocks[modes.index((1, 0))]
        # This small transverse admittance is why the short PML reflected this
        # order appreciably despite a successful uniform-slab control.
        self.assertAlmostEqual(grazing[1, 1].real, 1.50714457696335, delta=2e-13)
        self.assertTrue(np.all(np.linalg.eigvalsh(grazing) > 0))

    def test_rayleigh_mesh_has_no_absorbing_layers(self):
        mesh = self.reference.make_mesh(.25, boundary='rayleigh')
        self.assertFalse(any('pml' in name for name in mesh.GetMaterials()))
        self.assertIn('outer_top', mesh.GetBoundaries())
        self.assertIn('outer_bottom', mesh.GetBoundaries())
        self.reference.periodic_counts(mesh)

    def test_rayleigh_zero_contrast_and_driven_slab(self):
        zero = self.reference.solve_rayleigh(.25, 1, cut=1, uniform=1.)
        self.assertAlmostEqual(zero['T'], .9621399115785346, delta=2e-10)
        self.assertLess(zero['energy_error'], 2e-10)
        driven = self.reference.solve_rayleigh(.18, 3, cut=1, uniform=4.)
        self.assertLess(driven['analytic_T_error'], 5e-4)
        self.assertLess(driven['flux_disagreement'], 5e-4)
        self.assertLess(driven['energy_error'], 5e-4)

    def test_rayleigh_condensation_recovers_full_boundary_system(self):
        full = self.reference.solve_rayleigh(.25, 2, cut=1, uniform=4., condense=False)
        condensed = self.reference.solve_rayleigh(.25, 2, cut=1, uniform=4.)
        for key in ('T', 'R', 'volume_T', 'volume_R'):
            self.assertAlmostEqual(full[key], condensed[key], delta=2e-10)
        self.assertLess(condensed['residual'], 1e-10)


if __name__ == '__main__':
    unittest.main()
