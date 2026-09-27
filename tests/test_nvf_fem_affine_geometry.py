"""Optional, small-mesh checks for the physical affine FEM geometry.

These controls construct tetrahedra and finite-element spaces, not Maxwell
systems. Run with unittest in the separately installed NGSolve environment.
"""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

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
        cls.refined, cls.refined_materials = reference.make_affine_profile_mesh(edge_h=.08)
        cls.refined_rotated, cls.refined_rotated_materials = reference.make_affine_profile_mesh(
            transform=cls.rotation@cls.transform, edge_h=.08)

    def test_material_and_cell_volumes_follow_physical_geometry(self):
        expected = {'piece_0': .7*.206*.683*.3,
                    'piece_1': .7*.144*.683*.3,
                    'background': .343*.3-.7*.350*.683*.3,
                    'substrate': .343*.1, 'superstrate': .343*.1}
        self.assertEqual(self.materials, {'piece_0': 4., 'piece_1': 9.,
                                         'background': 1., 'substrate': 2.25,
                                         'superstrate': 1.})
        for mesh in (self.mesh, self.rotated, self.refined, self.refined_rotated):
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
        for mesh in (self.mesh, self.rotated, self.refined, self.refined_rotated):
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
                (self.rotated, self.rotated_materials, self.rotation@self.transform),
                (self.refined, self.refined_materials, self.transform),
                (self.refined_rotated, self.refined_rotated_materials,
                 self.rotation@self.transform)):
            epsilon = mesh.MaterialCF(materials)
            for x, y, z, expected in samples:
                physical = transform@np.array([x, y])
                self.assertAlmostEqual(epsilon(mesh(*physical, z)), expected)

    def test_actual_pairs_match_signed_primitive_vectors_and_reduce_hcurl_dofs(self):
        for mesh, lattice in ((self.mesh, np.array([[.7, .315], [0., .49]])),
                              (self.rotated, self.rotation@np.array([[.7, .315], [0., .49]])),
                              (self.refined, np.array([[.7, .315], [0., .49]])),
                              (self.refined_rotated,
                               self.rotation@np.array([[.7, .315], [0., .49]]))):
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

    def test_local_sizing_refines_all_twenty_physical_dielectric_edges_after_rotation(self):
        # Define the physical lines independently of OCC edge labels/selection:
        # six vertical lines and seven horizontal lines on each film interface.
        reference_lines = []
        for x in (-.175, .031, .175):
            for y in (-.3415, .3415):
                reference_lines.append(((x, y, 0.), (x, y, .3)))
        for z in (0., .3):
            for x in (-.175, .031, .175):
                reference_lines.append(((x, -.3415, z), (x, .3415, z)))
            for y in (-.3415, .3415):
                for left, right in ((-.175, .031), (.031, .175)):
                    reference_lines.append(((left, y, z), (right, y, z)))
        self.assertEqual(len(reference_lines), 20)
        for ordinary, refined, transform in (
                (self.mesh, self.refined, self.transform),
                (self.rotated, self.refined_rotated, self.rotation@self.transform)):
            clouds = [np.array([tuple(point.p) for point in mesh.ngmesh.Points()])
                      for mesh in (ordinary, refined)]
            for line in reference_lines:
                endpoints = np.array(line)
                endpoints[:, :2] = endpoints[:, :2]@transform.T
                first, last = endpoints
                direction = last-first
                fractions = []
                for points in clouds:
                    along = (points-first)@direction/(direction@direction)
                    distance = np.linalg.norm(points-first-along[:, None]*direction, axis=1)
                    on_line = (distance < 1e-9) & (along >= -1e-9) & (along <= 1+1e-9)
                    positions = np.sort(along[on_line])
                    self.assertAlmostEqual(positions[0], 0., delta=1e-9)
                    self.assertAlmostEqual(positions[-1], 1., delta=1e-9)
                    fractions.append(positions)
                with self.subTest(transform=transform.tolist(), line=line):
                    self.assertGreater(len(fractions[1]), len(fractions[0]))
                    self.assertLess(np.max(np.diff(fractions[1])),
                                    np.max(np.diff(fractions[0])))

    def test_mesh_stage_reports_local_sizing_and_actual_condensed_space_count(self):
        class MeshStageComplete(Exception):
            pass

        original = self.reference.make_affine_profile_mesh
        meshes, events = [], []

        def capture_mesh(*args, **kwargs):
            result = original(*args, **kwargs)
            meshes.append(result[0])
            return result

        def stop_before_assembly(event):
            events.append(event)
            raise MeshStageComplete

        with patch.object(self.reference, 'make_affine_profile_mesh', capture_mesh):
            with self.assertRaises(MeshStageComplete):
                self.reference.solve_affine_rayleigh(
                    order=3, edge_h=.12, vertex_h=.07,
                    condense=False, progress=stop_before_assembly)
        self.assertEqual(len(events), 1)
        metadata = events[0]
        self.assertEqual(metadata['stage'], 'mesh')
        self.assertEqual(metadata['edge_h'], .12)
        self.assertEqual(metadata['settings']['edge_h'], .12)
        self.assertEqual(metadata['vertex_h'], .07)
        self.assertEqual(metadata['settings']['vertex_h'], .07)
        space = ng.Compress(ng.Periodic(ng.HCurl(meshes[0], order=3, complex=True)))
        self.assertEqual(metadata['dofs'], space.ndof)
        self.assertEqual(metadata['coupling_dofs'], sum(space.FreeDofs(coupling=True)))
        self.assertLess(metadata['coupling_dofs'], metadata['dofs'])

    def test_absent_vertex_target_preserves_existing_mesh_exactly(self):
        for options in ({}, dict(h=.18, pad=.2, edge_h=.12), dict(uniform=4.)):
            ordinary, materials = self.reference.make_affine_profile_mesh(**options)
            explicit, explicit_materials = self.reference.make_affine_profile_mesh(
                **options, vertex_h=None)
            with self.subTest(options=options):
                self.assertEqual(materials, explicit_materials)
                np.testing.assert_array_equal(
                    [tuple(point.p) for point in ordinary.ngmesh.Points()],
                    [tuple(point.p) for point in explicit.ngmesh.Points()])
                for region in (ng.VOL, ng.BND):
                    self.assertEqual(
                        [(element.index, tuple(vertex.nr for vertex in element.vertices))
                         for element in ordinary.Elements(region)],
                        [(element.index, tuple(vertex.nr for vertex in element.vertices))
                         for element in explicit.Elements(region)])
                self.assertEqual(
                    [(first.nr, second.nr) for first, second in ordinary.ngmesh.GetIdentifications()],
                    [(first.nr, second.nr) for first, second in explicit.ngmesh.GetIdentifications()])

    def test_vertex_target_follows_twelve_physical_corners_and_preserves_geometry(self):
        original_glue = self.reference.occ.Glue
        expected_volumes = {'piece_0': .7*.206*.683*.3, 'piece_1': .7*.144*.683*.3,
                            'background': .343*.3-.7*.350*.683*.3,
                            'substrate': .343*.2, 'superstrate': .343*.2}
        reference_corners = np.array([(x, y, z) for x in (-.175, .031, .175)
                                      for y in (-.3415, .3415) for z in (0., .3)])
        for transform in (self.transform, self.rotation@self.transform):
            tagged = set()

            def capture_corners(parts):
                for part in parts:
                    for vertex in part.vertices:
                        if vertex.maxh == .07:
                            tagged.add(tuple(np.round(tuple(vertex.p), 12)))
                return original_glue(parts)

            ordinary, _ = self.reference.make_affine_profile_mesh(
                .18, pad=.2, edge_h=.12, transform=transform)
            with patch.object(self.reference.occ, 'Glue', capture_corners):
                refined, materials = self.reference.make_affine_profile_mesh(
                    .18, pad=.2, edge_h=.12, vertex_h=.07, transform=transform)
            expected_corners = reference_corners.copy()
            expected_corners[:, :2] = expected_corners[:, :2]@transform.T
            self.assertEqual(len(tagged), 12)
            distances = np.linalg.norm(np.asarray(list(tagged))[:, None, :]
                                       -expected_corners[None, :, :], axis=2)
            self.assertTrue(np.all(distances.min(axis=0) < 1e-11))
            self.assertTrue(np.all(distances.min(axis=1) < 1e-11))
            self.assertEqual(materials, self.materials)
            for material, volume in expected_volumes.items():
                self.assertAlmostEqual(ng.Integrate(1, refined,
                                                    definedon=refined.Materials(material)),
                                       volume, delta=1e-11)
            self.assertAlmostEqual(ng.Integrate(1, refined), .343*.7, delta=1e-11)
            for side, height in (('outer_top', .5), ('outer_bottom', -.2)):
                region = refined.Boundaries(side)
                self.assertAlmostEqual(ng.Integrate(1, refined, definedon=region),
                                       .343, delta=1e-11)
                self.assertAlmostEqual(ng.Integrate(ng.z, refined, definedon=region),
                                       .343*height, delta=1e-11)
            counts = self.reference.periodic_counts(refined, .7*transform)
            self.assertGreater(counts['x'], 0)
            self.assertGreater(counts['y'], 0)
            vectors = [np.r_[.7*transform[:, axis], 0.] for axis in (0, 1)]
            for first, second in refined.ngmesh.GetIdentifications():
                delta = np.array(refined.ngmesh[second].p)-np.array(refined.ngmesh[first].p)
                self.assertTrue(any(np.linalg.norm(delta-sign*vector) < 1e-9
                                    for vector in vectors for sign in (-1, 1)))
            # Each corner terminates a vertical dielectric edge. Compare the
            # adjacent physical segment, not the nearest point across the gap.
            steps = []
            for mesh in (ordinary, refined):
                points = np.array([tuple(point.p) for point in mesh.ngmesh.Points()])
                reference_xy = points[:, :2]@np.linalg.inv(transform).T
                adjacent = []
                for x in (-.175, .031, .175):
                    for y in (-.3415, .3415):
                        on_edge = ((abs(reference_xy[:, 0]-x) < 1e-9)
                                   & (abs(reference_xy[:, 1]-y) < 1e-9)
                                   & (points[:, 2] >= -1e-9) & (points[:, 2] <= .3+1e-9))
                        z = np.sort(points[on_edge, 2])
                        self.assertGreaterEqual(len(z), 3)
                        adjacent.extend((z[1]-z[0], z[-1]-z[-2]))
                steps.append(np.asarray(adjacent))
            self.assertTrue(np.all(steps[1] < steps[0]))

    def test_vertex_and_edge_targets_are_independent(self):
        original_glue = self.reference.occ.Glue
        vertex_values, edge_values = [], []

        def capture_targets(parts):
            for part in parts:
                vertex_values.extend(vertex.maxh for vertex in part.vertices if vertex.maxh < 1.)
                edge_values.extend(edge.maxh for edge in part.edges if edge.maxh < 1.)
            return original_glue(parts)

        with patch.object(self.reference.occ, 'Glue', capture_targets):
            self.reference.make_affine_profile_mesh(.18, edge_h=.10, vertex_h=.12)
        self.assertEqual(set(vertex_values), {.12})
        self.assertEqual(set(edge_values), {.10})
        # A vertex target does not require an edge target.
        mesh, _ = self.reference.make_affine_profile_mesh(.18, vertex_h=.12)
        self.assertGreater(mesh.ne, 0)

    def test_invalid_vertex_requests_fail_before_mesh_generation(self):
        invalid = [dict(vertex_h=value) for value in
                   (0., -.1, .21, np.nan, np.inf, True, np.bool_(True), 1j, [.1], '.1')]
        invalid += [dict(vertex_h=.1, uniform=4.)]
        with patch.object(self.reference.occ, 'OCCGeometry',
                          side_effect=AssertionError('invalid vertex target reached mesh generation')):
            for options in invalid:
                for entry in (self.reference.make_affine_profile_mesh,
                              self.reference.solve_affine_rayleigh):
                    with self.subTest(options=options, entry=entry.__name__), self.assertRaises(ValueError):
                        entry(**options)

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
        bad += [dict(edge_h=value) for value in
                (0., -.1, .21, np.nan, np.inf, True, np.bool_(True), 1j, [.1], '.1')]
        bad += [dict(edge_h=.1, uniform=4.)]
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
