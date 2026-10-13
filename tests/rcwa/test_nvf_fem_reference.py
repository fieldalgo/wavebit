"""Optional FEM reference checks; runnable with Python's unittest alone.

Run in the NGSolve environment:
    PYTHONPATH=$PWD python -m unittest discover -s tests -p test_nvf_fem_reference.py
The normal Wavebit test environment can skip this optional dependency.
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
class FEMAlgebraicTests(unittest.TestCase):
    """Analytic and small-array controls; no mesh or FEM system is constructed."""

    @classmethod
    def setUpClass(cls):
        from studies.rcwa.nvf_fem_reference import nvf_fem_reference as reference
        cls.reference = reference

    def test_exact_slab_control(self):
        result = self.reference.slab_efficiencies(4.)
        self.assertAlmostEqual(result['T'], .838052025532751, delta=2e-14)
        self.assertAlmostEqual(result['T']+result['R'], 1., delta=2e-14)

    def test_rayleigh_cut_cannot_omit_propagating_substrate_order(self):
        with self.assertRaisesRegex(ValueError, 'omits a propagating'):
            self.reference.rayleigh_data(0, 2.25)
        modes, blocks = self.reference.rayleigh_data(1, 2.25)
        grazing = blocks[modes.index((1, 0))]
        # This small transverse admittance is why the short PML reflected this
        # order appreciably despite a successful uniform-slab control.
        self.assertAlmostEqual(grazing[1, 1].real, 1.50714457696335, delta=2e-13)
        self.assertTrue(np.all(np.linalg.eigvalsh(grazing) > 0))

    def test_affine_propagating_orders_and_plane_wave_admittances(self):
        lattice = [[.7, .315], [0., .49]]
        expected = {1.: {(0, 0): 6.8421554688331865},
                    2.25: {(0, 0): 10.379719428627423,
                           (-1, 0): 4.332808115991115,
                           (-1, -1): 1.5266749945101408}}
        k0 = 2*np.pi/.9
        for epsilon, propagating in expected.items():
            modes, blocks = self.reference.rayleigh_data(1, epsilon, lattice=lattice)
            actual = {mode for mode, block in zip(modes, blocks)
                      if np.linalg.norm(block.imag) < 1e-13}
            self.assertEqual(actual, set(propagating))
            for (m, n), gamma in propagating.items():
                # Components follow a_1=(.7,0), a_2=(.315,.49) explicitly.
                k = np.array([k0*np.sin(.2)+2*np.pi*m/.7,
                              2*np.pi*(n-.45*m)/.49])
                transverse = np.array([-k[1], k[0]])
                block = blocks[modes.index((m, n))]
                # Maxwell's two exterior plane-wave admittances are gamma
                # for transverse E and k0^2*epsilon/gamma for longitudinal E.
                np.testing.assert_allclose(block@transverse, gamma*transverse,
                                           rtol=2e-13, atol=2e-13)
                np.testing.assert_allclose(block@k, k0*k0*epsilon/gamma*k,
                                           rtol=2e-13, atol=2e-13)

    def test_affine_blocks_use_physical_reciprocal_vectors(self):
        k0 = 2*np.pi/.9
        theta, phi, epsilon = .31, .37, 2.25
        modes, blocks = self.reference.rayleigh_data(
            2, epsilon, lattice=[[.7, .315], [0., .49]], theta=theta, phi=phi)
        for m, n in ((0, 0), (-1, -1), (1, 0), (-2, 1)):
            kx = k0*np.sin(theta)*np.cos(phi)+2*np.pi*m/.7
            ky = k0*np.sin(theta)*np.sin(phi)+2*np.pi*(n-.45*m)/.49
            gamma = np.sqrt(complex(k0*k0*epsilon-kx*kx-ky*ky))
            expected = [[gamma+kx*kx/gamma, kx*ky/gamma],
                        [kx*ky/gamma, gamma+ky*ky/gamma]]
            np.testing.assert_allclose(blocks[modes.index((m, n))], expected,
                                       rtol=2e-14, atol=2e-13)

    def test_rayleigh_blocks_rotate_with_lattice_and_incidence(self):
        angle = .53
        rotation = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
        lattice = np.array([[.7, .315], [0., .49]])
        modes, blocks = self.reference.rayleigh_data(
            2, 2.25, lattice=lattice, theta=.31, phi=.17)
        rotated_modes, rotated = self.reference.rayleigh_data(
            2, 2.25, lattice=rotation@lattice, theta=.31, phi=.17+angle)
        self.assertEqual(rotated_modes, modes)
        np.testing.assert_allclose(rotated, rotation@blocks@rotation.T,
                                   rtol=5e-14, atol=5e-13)

    def test_sheared_basis_cannot_hide_propagating_labels_outside_cut(self):
        # This is the square physical lattice in another primitive basis.
        # Its propagating (1,0) physical wave now has integer labels (1,4).
        lattice = [[.7, 2.8], [0., .7]]
        with self.assertRaisesRegex(ValueError, 'omits a propagating'):
            self.reference.rayleigh_data(3, 2.25, lattice=lattice)
        modes, blocks = self.reference.rayleigh_data(4, 2.25, lattice=lattice)
        self.assertAlmostEqual(blocks[modes.index((1, 4))][1, 1].real,
                               1.50714457696335, delta=2e-13)

    def test_explicit_default_lattice_and_incidence_preserve_original_data(self):
        modes, blocks = self.reference.rayleigh_data(1, 2.25)
        explicit_modes, explicit = self.reference.rayleigh_data(
            1, 2.25, lattice=[[.7, 0.], [0., .7]], theta=.2, phi=0.)
        self.assertEqual(modes, explicit_modes)
        np.testing.assert_array_equal(blocks, explicit)

    def test_rayleigh_invalid_inputs_are_rejected(self):
        bad = [dict(cut=value) for value in (True, np.bool_(True), -1, 1.5)]
        bad += [dict(epsilon=value) for value in (0., -1., np.inf, np.nan, 1+1j, [1.], True)]
        bad += [{name: value} for name in ('theta', 'phi')
                for value in (np.inf, np.nan, 1j, [0.], True)]
        bad += [dict(lattice=value) for value in
                (np.eye(3), [1., 2.], [[1., 0.], [0., 0.]],
                 [[1., 2.], [2., 4.]], [[1., np.inf], [0., 1.]],
                 [[1., np.nan], [0., 1.]], np.eye(2, dtype=complex),
                 np.eye(2, dtype=bool))]
        for options in bad:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.reference.rayleigh_data(**(dict(cut=1, epsilon=2.25) | options))

    def test_exactly_grazing_retained_order_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'exactly grazing'):
            self.reference.rayleigh_data(
                1, 1., lattice=[[.9, 0.], [0., .45]], theta=0.)


@unittest.skipIf(ng is None, 'optional ngsolve reference environment is absent')
class FEMReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from studies.rcwa.nvf_fem_reference import nvf_fem_reference as reference
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
