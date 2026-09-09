"""Optional, small-mesh checks for the physical affine FEM geometry.

These controls construct tetrahedra and finite-element spaces, not Maxwell
systems. Run with unittest in the separately installed NGSolve environment.
"""
import unittest
from types import SimpleNamespace

import numpy as np

try:
    import ngsolve as ng
except ImportError:
    ng = None


@unittest.skipIf(ng is None, 'optional ngsolve reference environment is absent')
class AffineFEMGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from studies import nvf_fem_reference as reference
        cls.reference = reference
        ng.SetNumThreads(1)
        cls.transform = np.array([[1., .45], [0., .7]])
        angle = .53
        cls.rotation = np.array([[np.cos(angle), -np.sin(angle)],
                                 [np.sin(angle), np.cos(angle)]])
        cls.mesh, cls.materials = reference.make_affine_profile_mesh()
        cls.rotated, cls.rotated_materials = reference.make_affine_profile_mesh(
            transform=cls.rotation@cls.transform)
        cls.uniform, cls.uniform_materials = reference.make_affine_profile_mesh(uniform=4.)

    def test_material_and_cell_volumes_follow_physical_geometry(self):
        expected = {'piece_0': .7*.206*.683*.3,
                    'piece_1': .7*.144*.683*.3,
                    'background': .343*.3-.7*.350*.683*.3,
                    'substrate': .343*.1, 'superstrate': .343*.1}
        self.assertEqual(self.materials, {'piece_0': 4., 'piece_1': 9.,
                                         'background': 1., 'substrate': 2.25,
                                         'superstrate': 1.})
        for mesh in (self.mesh, self.rotated):
            self.assertEqual(set(mesh.GetMaterials()), set(expected))
            self.assertFalse(any('pml' in material for material in mesh.GetMaterials()))
            for material, volume in expected.items():
                actual = ng.Integrate(1, mesh, definedon=mesh.Materials(material))
                self.assertAlmostEqual(actual, volume, delta=1e-11)
            self.assertAlmostEqual(ng.Integrate(1, mesh), .343*.5, delta=1e-11)

    def test_outer_faces_have_correct_areas_and_heights(self):
        expected = {'outer_top': .343, 'outer_bottom': .343,
                    'periodic_x': 2*np.hypot(.315, .49)*.5,
                    'periodic_y': 2*.7*.5}
        for mesh in (self.mesh, self.rotated):
            for side, area in expected.items():
                actual = ng.Integrate(1, mesh, definedon=mesh.Boundaries(side))
                self.assertAlmostEqual(actual, area, delta=1e-11)
            for side, height in (('outer_top', .4), ('outer_bottom', -.1)):
                moment = ng.Integrate(ng.z, mesh, definedon=mesh.Boundaries(side))
                self.assertAlmostEqual(moment, .343*height, delta=1e-11)

    def test_material_sampling_on_both_sides_of_each_interface(self):
        samples = [(.029, 0., .15, 4.), (.033, 0., .15, 9.),
                   (-.174, 0., .15, 4.), (-.176, 0., .15, 1.),
                   (.174, 0., .15, 9.), (.176, 0., .15, 1.),
                   (0., .3405, .15, 4.), (0., .3425, .15, 1.),
                   (0., -.3405, .15, 4.), (0., -.3425, .15, 1.),
                   (0., 0., -.002, 2.25), (0., 0., .002, 4.),
                   (.08, 0., .298, 9.), (.08, 0., .302, 1.)]
        for mesh, materials, transform in (
                (self.mesh, self.materials, self.transform),
                (self.rotated, self.rotated_materials, self.rotation@self.transform)):
            epsilon = mesh.MaterialCF(materials)
            for x, y, z, expected in samples:
                physical = transform@np.array([x, y])
                self.assertAlmostEqual(epsilon(mesh(*physical, z)), expected)

    def test_actual_pairs_match_signed_primitive_vectors_and_reduce_hcurl_dofs(self):
        for mesh, lattice in ((self.mesh, np.array([[.7, .315], [0., .49]])),
                              (self.rotated, self.rotation@np.array([[.7, .315], [0., .49]]))):
            counts = self.reference.periodic_counts(mesh, lattice)
            self.assertGreater(counts['x'], 0)
            self.assertGreater(counts['y'], 0)
            vectors = [np.r_[lattice[:, i], 0.] for i in range(2)]
            for first, second in mesh.ngmesh.GetIdentifications():
                delta = np.array(mesh.ngmesh[second].p)-np.array(mesh.ngmesh[first].p)
                self.assertTrue(any(np.linalg.norm(delta-sign*vector) < 1e-9
                                    for vector in vectors for sign in (-1, 1)))
            ordinary = ng.HCurl(mesh, order=1, complex=True)
            periodic = ng.Compress(ng.Periodic(ordinary))
            self.assertLess(periodic.ndof, ordinary.ndof)

    def test_uniform_control_fills_the_entire_affine_film(self):
        self.assertEqual(self.uniform_materials,
                         {'film': 4., 'substrate': 2.25, 'superstrate': 1.})
        self.assertEqual(set(self.uniform.GetMaterials()), set(self.uniform_materials))
        self.assertAlmostEqual(ng.Integrate(1, self.uniform,
                                          definedon=self.uniform.Materials('film')),
                               .343*.3, delta=1e-11)
        epsilon = self.uniform.MaterialCF(self.uniform_materials)
        for x, y in ((-.3, -.3), (0., 0.), (.3, .3), (0., .345)):
            physical = self.transform@np.array([x, y])
            self.assertAlmostEqual(epsilon(self.uniform(*physical, .15)), 4.)

    def test_original_square_and_2d_scalar_meshes_keep_periodic_counts(self):
        from studies.nvf_cell_corrector import make_mesh as make_scalar_cell
        from studies.nvf_gap_reference import make_profile_mesh
        square = self.reference.make_mesh(.25, uniform=4., pad=.1, boundary='rayleigh')
        scalar_cell = make_scalar_cell(h=.2, refine=1.)
        scalar, _ = make_profile_mesh(h=.2, refine=1.)
        for mesh in (square, scalar_cell, scalar):
            implicit = self.reference.periodic_counts(mesh)
            explicit = self.reference.periodic_counts(mesh, [[.7, 0.], [0., .7]])
            self.assertEqual(implicit, explicit)
            self.assertGreater(implicit['x'], 0)
            self.assertGreater(implicit['y'], 0)

    def test_pair_orientation_can_reverse_but_not_individual_components(self):
        class PointPairs:
            def __init__(self, deltas):
                self.points = [SimpleNamespace(p=(0., 0., 0.))]
                self.points += [SimpleNamespace(p=delta) for delta in deltas]

            def GetIdentifications(self):
                return [(0, i) for i in range(1, len(self.points))]

            def __getitem__(self, index):
                return self.points[index]

        lattice = [[.7, .315], [0., .49]]
        reversed_pairs = SimpleNamespace(ngmesh=PointPairs([(-.7, 0., 0.), (-.315, -.49, 0.)]))
        self.assertEqual(self.reference.periodic_counts(reversed_pairs, lattice), {'x': 1, 'y': 1})
        wrong_sign = SimpleNamespace(ngmesh=PointPairs([(.7, 0., 0.), (.315, -.49, 0.)]))
        with self.assertRaisesRegex(ValueError, 'unexpected periodic point displacement'):
            self.reference.periodic_counts(wrong_sign, lattice)
        missing = SimpleNamespace(ngmesh=PointPairs([(.7, 0., 0.)]))
        with self.assertRaisesRegex(ValueError, 'missing periodic point identifications'):
            self.reference.periodic_counts(missing, lattice)

    def test_invalid_geometry_is_rejected(self):
        bad = [dict(h=value) for value in (0., -.1, .25, np.inf, True)]
        bad += [{name: value} for name in ('period', 'thickness', 'pad')
                for value in (0., -.1, np.nan, [1.])]
        bad += [dict(width=value) for value in (0., .7, .8)]
        bad += [dict(gap=value) for value in (0., -.01, .7, .8)]
        bad += [dict(seam=value) for value in (-.175, .175, .3)]
        bad += [dict(uniform=value) for value in (0., -1., np.nan, 1j, [4.], True)]
        bad += [dict(transform=value) for value in
                (np.eye(3), [[1., 0.], [0., 0.]], [[1., 0.], [0., -1.]],
                 [[1., np.nan], [0., 1.]], np.eye(2, dtype=complex),
                 np.eye(2, dtype=bool))]
        for options in bad:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.reference.make_affine_profile_mesh(**options)
        for lattice in (np.eye(3), [[1., 0.], [0., 0.]],
                        [[1., np.inf], [0., 1.]], np.eye(2, dtype=complex)):
            with self.subTest(lattice=lattice), self.assertRaises(ValueError):
                self.reference.periodic_counts(self.mesh, lattice)


if __name__ == '__main__':
    unittest.main()
