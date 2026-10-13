"""Optional mesh/space controls for fixed-mesh local H(curl) degree refinement."""
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
class AffineFEMSpaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from studies.rcwa.nvf_fem_reference.nvf_fem_reference import make_affine_profile_mesh
        from studies.rcwa.nvf_fem_space.nvf_fem_space import make_affine_hcurl_space
        ng.SetNumThreads(1)
        cls.make_space = staticmethod(make_affine_hcurl_space)
        cls.mesh, _ = make_affine_profile_mesh(.18, pad=.2, edge_h=.12)
        cls.base = ng.Compress(ng.Periodic(ng.HCurl(cls.mesh, order=5, complex=True)))
        cls.enriched, cls.metadata = make_affine_hcurl_space(
            cls.mesh, 5, film_degree=6, film_degree_pad=.05)

    def assert_same_space(self, first, second):
        self.assertEqual(first.ndof, second.ndof)
        self.assertEqual(list(first.FreeDofs(coupling=True)),
                         list(second.FreeDofs(coupling=True)))
        for element in self.mesh.Elements(ng.VOL):
            self.assertEqual(first.GetDofNrs(element), second.GetDofNrs(element))
            self.assertEqual(first.GetFE(element).order, second.GetFE(element).order)

    def test_default_and_equal_degree_preserve_global_space(self):
        for options in ({}, dict(film_degree=5), dict(film_degree=5, film_degree_pad=.05)):
            with self.subTest(options=options):
                space, metadata = self.make_space(self.mesh, 5, **options)
                self.assert_same_space(space, self.base)
                self.assertEqual(metadata['base_order'], 5)
                self.assertEqual(metadata['max_order'], 5)
                self.assertEqual(metadata['selected_entities'], dict(edges=0, faces=0, cells=0))

    def test_uniform_enrichment_matches_the_higher_global_space(self):
        space, metadata = self.make_space(self.mesh, 5, film_degree=6, film_degree_pad=1.)
        expected = ng.Compress(ng.Periodic(ng.HCurl(self.mesh, order=6, complex=True)))
        self.assert_same_space(space, expected)
        self.assertEqual(metadata['selected_entities'],
                         dict(edges=self.mesh.nedge, faces=self.mesh.nface, cells=self.mesh.ne))

    def test_local_orders_match_independent_physical_band_and_budget(self):
        z = {vertex.nr: vertex.point[2] for vertex in self.mesh.vertices}
        selected = dict(edges=0, faces=0, cells=0)
        # Entity dimensions for the ordinary Nedelec orders 5 and 6. These
        # measure the actual resulting space, not a helper-side selection list.
        dimensions = {'edges': {5: 6, 6: 7}, 'faces': {5: 24, 6: 35},
                      'cells': {5: 36, 6: 70}}
        for kind, entities in (('edges', self.mesh.edges), ('faces', self.mesh.faces),
                                ('cells', self.mesh.Elements(ng.VOL))):
            for entity in entities:
                center = np.mean([z[vertex.nr] for vertex in entity.vertices])
                degree = 6 if -.05-1e-13 <= center <= .35+1e-13 else 5
                node = ng.NodeId(ng.CELL, entity.nr) if kind == 'cells' else entity
                self.assertEqual(len(self.enriched.GetDofNrs(node)), dimensions[kind][degree])
                selected[kind] += degree == 6
        self.assertEqual(self.metadata['selected_entities'], selected)
        self.assertEqual(self.metadata['base_order'], 5)
        self.assertEqual(self.metadata['max_order'], 6)
        self.assertEqual(self.metadata['dofs'], self.enriched.ndof)
        self.assertEqual(self.metadata['coupling_dofs'],
                         sum(self.enriched.FreeDofs(coupling=True)))
        self.assertGreater(self.enriched.ndof, self.base.ndof)
        self.assertGreater(self.metadata['coupling_dofs'],
                           sum(self.base.FreeDofs(coupling=True)))
        self.assertLess(self.metadata['coupling_dofs'], 60000)

    def test_periodic_entity_orders_and_identified_dofs_agree(self):
        parent = list(range(self.mesh.nv))

        def representative(index):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        for first, second in self.mesh.ngmesh.GetIdentifications():
            parent[representative(first.nr-1)] = representative(second.nr-1)
        for entities in (self.mesh.edges, self.mesh.faces):
            groups = {}
            for entity in entities:
                key = tuple(sorted(representative(vertex.nr) for vertex in entity.vertices))
                groups.setdefault(key, []).append(entity)
            paired = [group for group in groups.values() if len(group) > 1]
            self.assertTrue(paired)
            for group in paired:
                expected = self.enriched.GetDofNrs(group[0])
                for entity in group[1:]:
                    self.assertEqual(self.enriched.GetDofNrs(entity), expected)

    def test_every_element_has_consistent_dofs_and_enriched_order(self):
        orders = set()
        for element in self.mesh.Elements(ng.VOL):
            finite_element = self.enriched.GetFE(element)
            self.assertEqual(finite_element.ndof, len(self.enriched.GetDofNrs(element)))
            edge_orders = [len(self.enriched.GetDofNrs(edge))-1 for edge in element.edges]
            face_orders = [6 if len(self.enriched.GetDofNrs(face)) == 35 else 5
                           for face in element.faces]
            cell = ng.NodeId(ng.CELL, element.nr)
            cell_order = 6 if len(self.enriched.GetDofNrs(cell)) == 70 else 5
            self.assertEqual(finite_element.order, max(edge_orders+face_orders+[cell_order]))
            orders.add(finite_element.order)
        self.assertEqual(orders, {5, 6})

    def test_transition_bulk_and_face_forms_have_sufficient_default_quadrature(self):
        u, v = self.enriched.TnT()
        lateral = ng.CF((.4, .7, 0.))
        bulk = ((ng.curl(u)+1j*ng.Cross(lateral, u))
                *(ng.curl(v)-1j*ng.Cross(lateral, v))-9*u*v)
        boundary = -1j*u.Trace()*v.Trace()
        for region, expression in ((ng.VOL, bulk), (ng.BND, boundary)):
            transitions = [element for element in self.mesh.Elements(region)
                           if {len(self.enriched.GetDofNrs(edge))-1
                               for edge in element.edges} == {5, 6}]
            self.assertTrue(transitions)
            element = transitions[0]
            finite_element = self.enriched.GetFE(element)
            self.assertEqual(finite_element.order, 6)
            matrices = []
            for bonus in (0, 6):
                integrator = ng.SymbolicBFI(expression, region, bonus_intorder=bonus)
                matrices.append(np.array(integrator.CalcElementMatrix(
                    finite_element, self.mesh.GetTrafo(element),
                    heapsize=64000000, complex=True)))
            # A local polynomial-form check, without global assembly or solve.
            # It includes mixed entity degrees, the Bloch curl terms and a
            # tangential Robin form. Explicit field readouts use max_order.
            with self.subTest(region=region):
                self.assertGreater(np.linalg.norm(matrices[1]), 0.)
                self.assertLess(np.linalg.norm(matrices[0]-matrices[1])
                                /np.linalg.norm(matrices[1]), 1e-11)

    def test_invalid_parameters_are_rejected(self):
        invalid = [dict(order=value) for value in
                   (True, np.bool_(True), 0, -1, 1.5, 5., np.nan, np.inf, [5], '5')]
        invalid += [dict(film_degree=value) for value in
                    (True, np.bool_(True), 0, 4, 6., np.nan, np.inf, [6], '6')]
        invalid += [dict(film_degree=6, film_degree_pad=value) for value in
                    (True, np.bool_(True), -.1, np.nan, np.inf, 1j, [.05], '.05')]
        invalid += [dict(thickness=value) for value in
                    (True, 0., -.1, np.nan, np.inf, 1j, [.3], '.3')]
        invalid += [dict(film_degree_pad=.05)]
        for options in invalid:
            parameters = dict(order=5)
            parameters.update(options)
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.make_space(self.mesh, **parameters)


if __name__ == '__main__':
    unittest.main()
