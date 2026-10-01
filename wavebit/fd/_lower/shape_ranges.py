"""Fixed body allocations and the restricted public geometry-gradient certificate."""
from ..geometry import Sphere, Cylinder, Box, Prism, _shape_traced


def records(scene):
    """Structure index, kind, exact bounds and any static orientation.

    Bounds are recorded even when two ranges snap to the same cells. A plan reserves the
    range the caller declared, not any later range that happens to have the same allocation.
    Indexing also distinguishes moving a range to another painter in an overlapping scene.
    """
    out = []
    for index, structure in enumerate(scene.structures):
        shape = structure.shape
        if isinstance(shape, (Sphere, Cylinder, Box, Prism)) and shape.range is not None:
            row = (index, type(shape).__name__.lower(), shape.range)
            if isinstance(shape, Prism):
                row += (shape.axis, shape.angle, len(shape.vertices))
            elif isinstance(shape, Box):
                row += (shape.axis, shape.angle)
            elif isinstance(shape, Cylinder):
                row += (shape.axis,)
            out.append(row)
    return tuple(out)


def refuse_mismatch(scene, plan):
    """Protect direct ``lower`` callers too; ``run`` additionally checks the whole scene."""
    if records(scene) != plan.shape_ranges:
        raise ValueError('lower(scene, plan=...): Sphere/Cylinder/Box/Prism(range=) declarations differ '
                         'from the plan. Keep each range on the same structure with the same '
                         'bounds, static orientation and vertex count, or build a new plan '
                         'with fd.plan(scene).')


def leaves(shape):
    if isinstance(shape, Prism):
        return (shape.height, *shape.center, *(v for p in shape.vertices for v in p))
    if isinstance(shape, Box):
        return (*shape.size, *shape.center)
    if isinstance(shape, (Sphere, Cylinder)):
        return ((shape.radius, *shape.center)
                + ((shape.height,) if isinstance(shape, Cylinder) else ()))
    return ()


def traced(scene):
    return any(_shape_traced(v) for s in scene.structures for v in leaves(s.shape))


def ranged_body(shape):
    """A Sphere, Cylinder, Box or Prism that reserves a range: the only shape that may move."""
    return isinstance(shape, (Sphere, Cylinder, Box, Prism)) and shape.range is not None


def _overlap(a, b):
    """Whether two half-open index boxes share a cell (they may meet at an index boundary)."""
    return all(min(hi, d) > max(lo, c) for (lo, hi), (c, d) in zip(a, b))


def _carries(value):
    """Whether a material, a pole or any value inside them carries a derivative."""
    import dataclasses
    if isinstance(value, (tuple, list)):
        return any(_carries(v) for v in value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return any(_carries(getattr(value, f.name)) for f in dataclasses.fields(value))
    return _shape_traced(value)


def name(scene):
    return '/'.join(dict.fromkeys(type(s.shape).__name__ for s in scene.structures
                    if any(_shape_traced(v) for v in leaves(s.shape))))


def remember(scene, plan):
    """The seed holds only concrete numbers, never backend leaves or a captured scene."""
    seeds = tuple((i, s.shape.vertices, s.shape.height, s.shape.center)
                  if isinstance(s.shape, Prism) else
                  (i, s.shape.size, s.shape.center) if isinstance(s.shape, Box) else
                  (i, s.shape.radius, s.shape.center)
                  + ((s.shape.height,) if isinstance(s.shape, Cylinder) else ())
                  for i, s in enumerate(scene.structures)
                  if isinstance(s.shape, (Sphere, Cylinder, Box, Prism)) and s.shape.range is not None)
    if seeds:
        object.__setattr__(plan, '_shape_seeds', seeds)


def planning_scene(scene, plan):
    """A concrete preflight of all unchanged inputs, after certifying the full allocation.

    A seed proves nothing about another geometry. Fixed, interior supports with disjoint
    interpolation halos remove overlap, seam and wall cases for every geometry in every
    range. Runtime guards then check actual sizes, containment, star positivity and CFL.
    All materials and unchanged scene inputs are rechecked on the host. Interacting or
    nested material interfaces and geometry-dependent sources/monitors remain outside this
    certificate.

    Every ranged body, traced or concrete, keeps the full certificate: an isotropic lossless
    Material, a halo strictly inside the domain and clear of absorbers, and pairwise disjoint
    halos. A structure **without** a range is concrete by construction (its shape refuses a
    traced length) and is planned and lowered at its own geometry like any concrete scene's,
    with any material such a scene admits -- a cell-snapped substrate running into a PML, a
    conductive or dispersive layer. It needs only to stay off every traced body's halo: its
    support widened by the same interpolation cell must not share a cell with a traced
    body's halo, so nothing it paints or mixes reaches a cell whose coefficients move. Its
    material values must be concrete; a traced partner material is not certified here. A
    flush junction between two partners is refused beside a traced body, by
    ``junction.refuse_mismatch``, whose signature covers all scene geometry.

    Every source and monitor family a Scene admits has static placement. Plane-wave
    illumination (PlaneWave, TFSF) and the frequency-domain readouts (Flux, Diffraction,
    FarField) are admitted beside Dipole, Record and Field because their own doors already
    keep each sheet, box face and closed surface clear of every structure's support, which
    for a ranged body is its whole reservation, and a Flux plane or Field sample reads the
    fields alone. What remains geometry-dependent is the one plain medium the plane-wave
    algebra reads at a source's entry plane and a Diffraction plane (``_post.medium_at``);
    those cell layers must lie outside every traced body's support (:func:`plane_media`).
    """
    if not traced(scene):
        return scene
    where = name(scene)
    if plan is None or not getattr(plan, '_shape_seeds', ()):
        raise ValueError(f'A traced {where} requires plan= built by fd.plan(scene) from '
                         'concrete geometry before differentiation.')
    refuse_mismatch(scene, plan)
    from ..materials import Material, as_material
    from .._spec import static_float
    if scene.initial is not None:
        raise ValueError(f'A traced {where} supports ranged Sphere/Cylinder/Box/Prism bodies '
                         'with disjoint interpolation halos, beside concrete partners clear of '
                         'them, and no initial field.')
    ranged = [ranged_body(s.shape) for s in scene.structures]
    for material in (as_material(scene.background),
                     *(s.material for s, r in zip(scene.structures, ranged) if r)):
        if type(material) is not Material:
            raise ValueError(f'A traced {where} requires concrete isotropic lossless Material.')
        for triple in (material.eps, material.mu_value):
            values = tuple(static_float(f'traced {where} material', 'value', v) for v in triple)
            if not values[0] == values[1] == values[2]:
                raise ValueError(f'A traced {where} requires concrete isotropic lossless Material.')
    for i, (structure, r) in enumerate(zip(scene.structures, ranged)):
        if not r and _carries(as_material(structure.material)):
            raise ValueError(f'structures[{i}], a concrete partner without a range beside the '
                             f'traced {where}, has a traced material value; partner materials '
                             'must be concrete here.')
    seeds = {row[0]: row for row in plan._shape_seeds}
    # support() includes the occupancy margin; constitutive terms add one interpolation
    # cell (two adjacent half-cell averages). Certify it even for an unturned Box,
    # whose zero cross terms mean constitutive.validate has no term to inspect.
    halos = {}
    structures = []
    for i, structure in enumerate(scene.structures):
        shape = structure.shape
        support = shape.support(plan.grid)
        halo = tuple((lo-1, hi+1) for lo, hi in support)
        if not ranged[i]:
            structures.append(structure)        # checked against traced halos below
            continue
        if any(lo <= 1 or hi >= n - 1 for (lo, hi), n in zip(support, plan.grid.n)):
            raise ValueError(f'A traced {where} range at structures[{i}], including its '
                             'interpolation halo, must stay strictly inside the domain, '
                             'away from boundary masks.')
        for face, cells in plan.absorbing_faces:
            a = 'xyz'.index(face[0])
            if (halo[a][0] < cells if face[1] == '-' else halo[a][1] > plan.grid.n[a]-cells):
                raise ValueError(f'structures[{i}] range interpolation halo meets the '
                                 f'{face} absorbing layer; keep the whole reservation clear.')
        for j, other in halos.items():
            if _overlap(halo, other):
                raise ValueError(f'structures[{j}] and structures[{i}] range interpolation '
                                 'halos overlap. Traced bodies require disjoint reserved '
                                 'halos throughout the sweep, even when their seeds are apart.')
        halos[i] = halo
        if any(_shape_traced(v) for v in leaves(shape)):
            seed = seeds[i]
            if isinstance(shape, Prism):
                index, vertices, height, center = seed
                dimensions = dict(vertices=vertices, height=height, center=center)
            elif isinstance(shape, Box):
                index, size, center = seed
                dimensions = dict(size=size, center=center)
            else:
                index, radius, center, *height = seed
                dimensions = dict(radius=radius, center=center)
                if isinstance(shape, Cylinder):
                    dimensions['height'] = height[0]
            structure = structure.replace(shape=shape.replace(**dimensions))
        structures.append(structure)
    moving = {j: halo for j, halo in halos.items()
              if any(_shape_traced(v) for v in leaves(scene.structures[j].shape))}
    for i, structure in enumerate(scene.structures):
        if ranged[i]:
            continue
        reach = tuple((lo-1, hi+1) for lo, hi in structure.shape.support(plan.grid))
        for j, halo in moving.items():
            if _overlap(reach, halo):
                raise ValueError(
                    f'structures[{i}] ({structure.shape.describe()}), a concrete partner '
                    f'without a range, reaches the interpolation halo of the traced '
                    f'structures[{j}]: its support widened by one cell shares a cell with '
                    f'that halo (cells {tuple(halo)}), where the coefficients move with the '
                    'geometry. Keep every concrete partner one cell clear of each traced '
                    'reservation and its margin.')
    for axis, cells, label in plane_media(scene, plan):
        for i, structure in enumerate(scene.structures):
            if not any(_shape_traced(v) for v in leaves(structure.shape)):
                continue                    # concrete partners are read at their current values
            lo, hi = structure.shape.support(plan.grid)[axis]
            if any(lo <= k < hi for k in cells):
                raise ValueError(
                    f'{label} reads one plain medium in the cell layers {cells} across '
                    f'{"xyz"[axis]}, and the reserved support of the traced structures[{i}] '
                    f'(cells {lo}..{hi - 1}) reaches them: the medium there would move with '
                    'the geometry. Keep that plane outside every traced body\'s range and '
                    'its one-cell margin.')
    return scene.replace(structures=tuple(structures))


def plane_media(scene, plan):
    """``[(axis, cells, label), ...]``: each cell-layer pair ``_post.medium_at`` will read.

    A PlaneWave or TFSF normalisation reads the two layers either side of its column's entry
    plane (``_assemble._incident_split``), which for a PlaneWave is also the sheet layer pair
    a Diffraction monitor checks; a Diffraction monitor reads its own plane's pair.
    ``medium_at`` asks a shape's own fill of a layer only where that shape's support reaches
    it, so a layer outside every traced support is decided by concrete geometry alone.

    A FarField box also reads the pair at its low x face (``monitors.farfield_checks``), but
    the same check requires every structure's support strictly inside the box with a cell to
    spare, so that pair is never in a support and is not listed here. ``_post.layer_share``
    still refuses a traced shape if a later rule ever lets one reach a read layer.
    """
    from ..monitors import Diffraction
    from .monitors import diffraction_layout
    grid = plan.grid
    out = []
    for term in plan.terms:
        if getattr(term, 'kind', None) in ('PlaneWave', 'TFSF'):
            entry = int(round((term.plane_E - grid.low[term.axis]) / grid.dx))
            out.append((term.axis, (entry - 1, entry),
                        f'{term.kind}({term.name!r}) normalisation'))
    for m in scene.monitors:
        if isinstance(m, Diffraction):
            layout, _ = diffraction_layout(m, scene, grid)
            out.append((layout.P, (layout.k - 1, layout.k), f'Diffraction({m.name!r})'))
    return out


def validate(scene, plan):
    """Direct lower() has the same preflight as run(), not just a matching allocation."""
    concrete = planning_scene(scene, plan)
    if concrete is scene:
        return
    from .plan import plan as make_plan
    checked = make_plan(concrete, _shape_trace=True)
    if checked.skeleton != plan.skeleton:
        raise ValueError(f'lower(scene, plan=...): traced {name(scene)} plan belongs to a different '
                         'scene; rebuild fd.plan from concrete geometry.')


def _prism_outline_valid(shape, rt, minimum):
    """The concrete polygon door and hull-width floor, on backend scalar arrays.

    A hull's minimum width occurs perpendicular to one of its edges. Every hull
    edge joins two input vertices, so testing all distinct pairs includes that
    minimum without sorting a traced hull. Extra directions cannot undercut the
    global minimum. Degenerate pairs add infinity, not an unguarded division.
    The crossing signs and inclusive endpoint tests match polygon_shape; neither
    validity nor the width check contributes a derivative to the physical operator.
    """
    xp = rt.xp
    vertices = xp.stack([xp.stack([rt.array(v, 'float64') for v in p])
                         for p in shape.vertices])
    nv = len(shape.vertices)
    after = xp.roll(vertices, -1, axis=0)
    area = xp.sum(vertices[:, 0]*after[:, 1] - after[:, 0]*vertices[:, 1])
    valid = xp.all(xp.isfinite(vertices)) & (area > 0)
    width = rt.array(float('inf'), 'float64')

    def side(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

    def between(a, b, c):
        return xp.all((xp.minimum(a, b) <= c) & (c <= xp.maximum(a, b)))

    for i in range(nv):
        for j in range(i+1, nv):
            delta = vertices[j]-vertices[i]
            norm = xp.sqrt(xp.sum(delta*delta))
            relative = vertices-vertices[i]
            projection = relative[:, 0]*delta[1] - relative[:, 1]*delta[0]
            span = (xp.max(projection)-xp.min(projection))/xp.where(norm > 0, norm, 1.)
            width = xp.minimum(width, xp.where(norm > 0, span, float('inf')))
            if j == i+1 or (i == 0 and j == nv-1):
                continue
            p, q, r, s = vertices[i], after[i], vertices[j], after[j]
            d1, d2, d3, d4 = side(r, s, p), side(r, s, q), side(p, q, r), side(p, q, s)
            crossing = ((d1*d2 < 0) & (d3*d4 < 0)
                        | ((d1 == 0) & between(r, s, p))
                        | ((d2 == 0) & between(r, s, q))
                        | ((d3 == 0) & between(p, q, r))
                        | ((d4 == 0) & between(p, q, s)))
            valid = valid & ~crossing
    return valid & (width >= minimum)


def _valid(shape, plan, rt):
    """One body's containment and existing four-cell resolution door, on arrays."""
    from .checks import CELLS_ACROSS_REFUSED, _ACROSS_SLACK
    xp = rt.xp
    centers = (shape.center,)
    if isinstance(shape, Prism):
        height = rt.array(shape.height, 'float64')
        minimum = plan.dx * CELLS_ACROSS_REFUSED * (1.0 - _ACROSS_SLACK)
        valid = xp.isfinite(height) & (height >= minimum)
        valid = valid & _prism_outline_valid(shape, rt, minimum)
        halves = tuple(height / 2 if a == 'xyz'.index(shape.axis) else 0. for a in range(3))
        centers = shape._range_points()
    elif isinstance(shape, Box):
        sizes = tuple(rt.array(s, 'float64') for s in shape.size)
        valid = xp.ones_like(sizes[0], dtype=xp.bool)
        for size in sizes:
            valid = valid & xp.isfinite(size) & (
                size / plan.dx >= CELLS_ACROSS_REFUSED * (1.0 - _ACROSS_SLACK))
        halves = shape._range_halves()
    else:
        radius = rt.array(shape.radius, 'float64')
        valid = xp.isfinite(radius) & (2 * radius / plan.dx >=
                                      CELLS_ACROSS_REFUSED * (1.0 - _ACROSS_SLACK))
        height = None
        if isinstance(shape, Cylinder):
            height = rt.array(shape.height, 'float64')
            valid = valid & xp.isfinite(height) & (
                height / plan.dx >= CELLS_ACROSS_REFUSED * (1.0 - _ACROSS_SLACK))
        halves = tuple(height / 2 if height is not None and axis == 'xyz'.index(shape.axis)
                       else radius for axis in range(3))
    for center in centers:
        for axis, (c, (lo, hi)) in enumerate(zip(center, shape.range)):
            c = rt.array(c, 'float64')
            half = rt.array(halves[axis], 'float64')
            valid = valid & xp.isfinite(c) & (c - half >= lo) & (c + half <= hi)
    return valid


def guard(scene, plan, rt, previous):
    """Every ranged body's runtime geometry certificate, combined without host truth tests.

    A partner without a range is concrete and was checked on the host like any concrete
    scene's structure; it has no range to be contained in.
    """
    if not traced(scene):
        return previous
    valid = None
    for structure in scene.structures:
        if not ranged_body(structure.shape):
            continue
        current = _valid(structure.shape, plan, rt)
        valid = current if valid is None else valid & current
    value = rt.constant(rt.xp.where(valid, rt.array(1., 'float64'), rt.array(float('nan'), 'float64')))
    return value if previous is None else previous * value
