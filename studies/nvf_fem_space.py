"""Optional periodic H(curl) spaces for fixed-mesh affine FEM refinement.

This uses the separately installed NGSolve reference environment. It constructs
spaces only; it does not assemble or solve a Maxwell system.
"""
import numpy as np
import ngsolve as ng


def make_affine_hcurl_space(mesh, order, *, film_degree=None,
                            film_degree_pad=0., thickness=.3):
    """Return a complex periodic H(curl) space and its refinement metadata.

    ``order`` is the positive integer base degree. ``film_degree`` optionally
    raises the degree on edges, faces and tetrahedra whose vertex-average z
    lies in [-film_degree_pad, thickness+film_degree_pad]. Coordinates and
    lengths use the physical mesh units. Periodic identifications are assumed
    to be lateral, as in the affine reference mesh; they preserve this z band.

    ``film_degree`` must be an integer no smaller than ``order``. None or the
    base degree retains the original global-order space exactly. A positive
    ``film_degree_pad`` requires an explicit ``film_degree``. Metadata reports
    the base/max degrees, counts of entities raised, and full/coupling DOFs.
    The max degree must also govern explicit polynomial readout quadrature in
    callers. A small space or a high degree alone is no optical error bound.
    """
    if (isinstance(order, (bool, np.bool_))
            or not isinstance(order, (int, np.integer)) or order < 1):
        raise ValueError('order must be a positive integer')
    order = int(order)
    if film_degree is not None:
        if (isinstance(film_degree, (bool, np.bool_))
                or not isinstance(film_degree, (int, np.integer))
                or film_degree < order):
            raise ValueError('film_degree must be an integer no smaller than order')
        film_degree = int(film_degree)
    for name, value, positive in (('film_degree_pad', film_degree_pad, False),
                                 ('thickness', thickness, True)):
        scalar = np.asarray(value)
        if (scalar.shape != () or scalar.dtype.kind not in 'iuf'
                or not np.isfinite(scalar) or scalar < 0 or (positive and scalar == 0)):
            condition = 'positive' if positive else 'nonnegative'
            raise ValueError(f'{name} must be a finite {condition} real scalar')
    film_degree_pad, thickness = float(film_degree_pad), float(thickness)
    if film_degree is None and film_degree_pad != 0:
        raise ValueError('film_degree_pad requires an explicit film_degree')
    if mesh.dim != 3:
        raise ValueError('the affine H(curl) reference requires a three-dimensional mesh')

    selected = dict(edges=0, faces=0, cells=0)
    if film_degree is None or film_degree == order:
        ordinary = ng.HCurl(mesh, order=order, complex=True)
    else:
        ordinary = ng.HCurl(mesh, order=order, complex=True,
                             order_policy=ng.ORDER_POLICY.VARIABLE)
        z = {vertex.nr: vertex.point[2] for vertex in mesh.vertices}
        tolerance = 32*np.finfo(float).eps*max(thickness, film_degree_pad)
        lower, upper = -film_degree_pad-tolerance, thickness+film_degree_pad+tolerance
        for kind, entities in (('edges', mesh.edges), ('faces', mesh.faces),
                                ('cells', mesh.Elements(ng.VOL))):
            for entity in entities:
                middle = sum(z[vertex.nr] for vertex in entity.vertices)/len(entity.vertices)
                if lower <= middle <= upper:
                    node = ng.NodeId(ng.CELL, entity.nr) if kind == 'cells' else entity
                    ordinary.SetOrder(node, film_degree)
                    selected[kind] += 1
        ordinary.Update()
    space = ng.Compress(ng.Periodic(ordinary))
    metadata = dict(base_order=order,
                    max_order=film_degree if any(selected.values()) else order,
                    film_degree=film_degree, film_degree_pad=film_degree_pad,
                    thickness=thickness, selected_entities=selected,
                    dofs=space.ndof, coupling_dofs=sum(space.FreeDofs(coupling=True)))
    return space, metadata
