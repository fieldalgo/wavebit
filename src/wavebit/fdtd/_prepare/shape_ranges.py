"""Fixed body allocations and the restricted public geometry-gradient certificate."""
from wavebit.fdtd._legacy_geometry import Sphere
from wavebit.fdtd._legacy_geometry import Cylinder
from wavebit.fdtd._legacy_geometry import Box
from wavebit.fdtd._legacy_geometry import Prism
from wavebit.fdtd._legacy_geometry import _shape_traced


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
    """The seed holds only concrete numbers, never backend leaves or a captured scene.

    Two seeds: each ranged body's dimensions (``_shape_seeds``), and the values of every
    plain ``fd.Material`` the plan was built with (``_material_seeds``, keyed by structure
    index, ``-1`` for the background), which :func:`_seeded_materials` puts back in place of
    a traced permittivity or permeability for host planning.
    """
    seeds = tuple((i, s.shape.vertices, s.shape.height, s.shape.center)
                  if isinstance(s.shape, Prism) else
                  (i, s.shape.size, s.shape.center) if isinstance(s.shape, Box) else
                  (i, s.shape.radius, s.shape.center)
                  + ((s.shape.height,) if isinstance(s.shape, Cylinder) else ())
                  for i, s in enumerate(scene.structures)
                  if isinstance(s.shape, (Sphere, Cylinder, Box, Prism)) and s.shape.range is not None)
    if seeds:
        object.__setattr__(plan, '_shape_seeds', seeds)
    from wavebit.materials.time_domain import as_material
    materials = tuple((key, *seed) for key, material in _materials(scene)
                      for seed in (_material_seed(as_material(material)),) if seed is not None)
    if materials:
        object.__setattr__(plan, '_material_seeds', materials)


def _materials(scene):
    """``(key, material)`` for the background (key ``-1``) and every structure, in order."""
    return ((-1, scene.background),
            *((i, s.material) for i, s in enumerate(scene.structures)))


def _material_seed(material):
    """``(eps, mu)`` of a plain concrete ``fd.Material`` as host floats (``mu`` None when it
    was not named), or None for any other material or one ``materials.concrete`` cannot
    read -- a JAX tracer or a Torch tensor that asks for a gradient."""
    from wavebit.materials.time_domain import Material
    from wavebit.materials.time_domain import concrete
    if type(material) is not Material:
        return None
    eps = tuple(concrete(v) for v in material.eps)
    mu = None if material.mu is None else tuple(concrete(v) for v in material.mu)
    if None in eps or (mu is not None and None in mu):
        return None
    return eps, mu


def materials_traced(scene):
    """Whether the background or any structure's material carries a derivative."""
    from wavebit.materials.time_domain import as_material
    return any(_carries(as_material(m)) for _, m in _materials(scene))


def _coupled(plan):
    """Whether host planning certifies smoothed coupling from material numbers: a
    diagonal-rule constitutive term (``constitutive.validate``'s star lift and bound).

    A junction alone is not: ``junction.classify`` asks the materials structural questions
    only, so a flush pair of cross-free unturned Boxes plans a traced value live, as it did
    before material seeds existed (and records no seed for it)."""
    from wavebit.fdtd._kernel.stencil import StencilTerm
    return any(not isinstance(t, StencilTerm) for t in plan.constitutive)


def _one_number(triple):
    """A triple given as one scalar (the same object three times, traced or not) or three
    equal concrete numbers: the isotropic reading a traced value allows."""
    from wavebit.materials.time_domain import concrete
    if triple[0] is triple[1] is triple[2]:
        return True
    values = [concrete(v) for v in triple]
    return None not in values and values[0] == values[1] == values[2]


def _read_as_one_number(scene, plan, key):
    """Whether lowering reads medium ``key`` as one number: the background, a smoothed shape,
    a junction's medium, or a structure within a cell of a smoothed shape's support -- where
    ``constitutive.lower`` and ``junction.tables`` read component 0 and ``check_smoothed``'s
    isotropy door was asked of the plan's values only."""
    from wavebit.fdtd._prepare.constitutive import _intersect
    if key < 0:
        return True
    shape = scene.structures[key].shape
    if shape.body() is not None or any(key in r.media for r in plan.junctions):
        return True
    support = shape.support(plan.grid)
    return any(_intersect(support, tuple((lo - 1, hi + 1)
                                         for lo, hi in s.shape.support(plan.grid))) is not None
               for s in scene.structures if s.shape.body() is not None)


def _stabilized_media(scene, plan):
    """``{'structures[i]', ...}``: the media filling a stabilized PML layer the grazing survey
    judges orders in.

    ``grazing._survey`` judges orders only for a scene with exactly one PlaneWave and a
    Diffraction monitor, on the PML faces along that wave's axis, in each medium filling the
    layer (``grazing._layer_media``, whose labels these are). On a pure-stretch face the
    verdict is a threshold in the order's exit cos, which :func:`grazing_guard` holds a
    traced medium to at run time; on a face with ``stabilize > 0`` it is also the face's exact
    discrete return (``grazing._judge``), which has no runtime twin.
    """
    from wavebit.fdtd._legacy_boundaries import PML
    from wavebit.fdtd._legacy_monitors import Diffraction
    from wavebit.fdtd._prepare.grazing import _layer_media
    from wavebit.fdtd._prepare.grazing import _stabilize
    from wavebit.fdtd._prepare.monitors import plane_waves
    waves = plane_waves(scene)
    if len(waves) != 1 or not any(isinstance(m, Diffraction) for m in scene.monitors):
        return set()
    P = waves[0].axis
    out = set()
    for side in ('-', '+'):
        face = scene.boundaries.face('xyz'[P] + side)
        if isinstance(face, PML) and _stabilize(face) > 0.0:
            out |= {label for label, _ in _layer_media(scene, plan.grid, P, side, face.cells)}
    return out


def _mirror_could_couple(scene, plan, term, a, side):
    """Whether ``term``'s cross coefficients in its end layer on the ``side`` face of axis
    ``a`` can be nonzero beside a traced medium: a smoothed body in the term has a fractional
    fill there, and a traced material paints into the term.

    ``constitutive.validate`` refuses cross coefficients on a Symmetry plane, and they are
    ``fill (1 - fill)`` times the contrast: zero where no fill is fractional, whatever the
    values, and zero where the plan's values have no contrast. Only the second can hide a
    coupling a traced value then makes. The fill is asked at all six components' Yee points
    of the layer (``_post.layer_share``'s reading), with the exact clip's round-off.
    """
    import numpy as np
    from wavebit.materials.time_domain import as_material
    from wavebit.fdtd._shapes import occupancy as body_occupancy
    from wavebit.fdtd._prepare.checks import _CLIP_ROUNDOFF
    from wavebit.fdtd._prepare.constitutive import _intersect
    from wavebit.fdtd._prepare.occupancy import yee_points
    grid = plan.grid
    box = list(reversed(term.box))
    box[a] = (box[a][0], box[a][0] + 1) if side == '-' else (box[a][1] - 1, box[a][1])
    box = tuple(box)
    inside = [s for s in scene.structures if _intersect(s.shape.support(grid), box)]
    if not (_carries(as_material(scene.background))
            or any(_carries(as_material(s.material)) for s in inside)):
        return False
    for s in inside:
        body = s.shape.body()
        if body is None:
            continue
        for field in ('E', 'H'):
            for c in range(3):
                points = np.asarray(yee_points(grid, box, field, c)).reshape(-1, 3)
                fill = np.asarray(body_occupancy(body, points, grid.dx)[0])
                if np.any((fill > _CLIP_ROUNDOFF) & (fill < 1.0 - _CLIP_ROUNDOFF)):
                    return True
    return False


def _seeded_materials(scene, plan, where, substitute=True):
    """``scene`` with every traced material value replaced by the plan's own seed value.

    Host planning certifies the plan's materials -- the star lift and the coupled CFL bound
    (``constitutive.validate``), the junction's material scope, the resolution floor, the
    survey of grazing orders -- and those certificates need numbers. The values the plan
    was built with passed them once, so restoring them makes host planning reproduce that
    plan from the unchanged inputs around them, while lowering reads the live values and
    the runtime guards judge those: the time-step bound with the lifted operator
    (``constitutive.guard``, ``params._traced_guard``), positivity
    (``params._material_door_guard``), every star's positivity at its lift
    (``constitutive.lift``), the resolution floor (:func:`resolution_guard`) and the
    grazing rule on a pure-stretch PML face (:func:`grazing_guard`). Only the
    traced leaves are restored; a concrete value beside them is checked as it is.

    What has no runtime twin is refused instead, by name:

    * a traced value outside a plain ``fd.Material``'s ``eps`` and ``mu``;
    * a traced ``eps`` or ``mu`` given per axis on a medium lowering reads as one number
      (:func:`_read_as_one_number`) -- a scalar keeps the triple one object, the isotropy a
      traced value can show; a snapped partner clear of every smoothed body keeps a traced
      diagonal, its slots restored one by one;
    * a traced background in a scene with a PML matched to it (no ``background=`` of its
      own), a PlaneWave or TFSF source, or a ``FarField(propagation='yee')``: it fixes the
      PML's match (``params._cpml``), the plane-wave column's transverse wavevector and drive
      (``sources.plane_wave_drive``), the grazing survey's rows and the lattice far-field map
      (``results`` refuses a traced one after the run), all built on the host (a cell driven
      by Dipoles, with any PML matched to a concrete medium of its own, reads it nowhere
      else);
    * a traced structure filling a stabilized PML layer the grazing survey reads
      (:func:`_stabilized_media`): the rule judges the orders there on the face's exact return
      too, on the host only (on a pure-stretch face :func:`grazing_guard` holds it at run
      time);
    * a traced material in a smoothed region where a fractional fill reaches a ``Symmetry``
      plane (:func:`_mirror_could_couple`): the mirror check reads the cross coefficients,
      which vanish where the plan's own values have no contrast, so a seed equal to its
      surroundings would pass it vacuously;
    * a traced medium at a ``Diffraction`` plane whose ``orders=`` is not given: the window of
      propagating orders is counted at the plan's value on the host, and again from the live
      medium after the run, where a traced one cannot be counted (``monitors.
      diffraction_window``); refused here rather than after the whole run;
    * a material the plan was built without, or whose permeability it named differently.

    ``substitute=False`` runs the refusals alone and returns ``scene``, for a direct
    ``fd.lower`` of concrete geometry, which plans nothing (:func:`validate`); the seed is
    then needed only to compare how ``mu`` was named, and a non-plain traced value -- a
    snapped partner's conductivity, say -- lowers as it always has.
    """
    from wavebit.materials.time_domain import Material
    from wavebit.materials.time_domain import as_material
    from wavebit.fdtd._kernel.stencil import StencilTerm
    from wavebit.fdtd._legacy_monitors import FarField
    from wavebit.fdtd._legacy_sources import PlaneWave
    from wavebit.fdtd._legacy_sources import TFSF
    seeds = {row[0]: row[1:] for row in getattr(plan, '_material_seeds', ())}
    background, structures = None, list(scene.structures)
    grid = plan.grid
    pml = dict(plan.pml_cells)
    stabilized = _stabilized_media(scene, plan) if materials_traced(scene) else set()
    for key, material in _materials(scene):
        material = as_material(material)
        if not _carries(material):
            continue
        label = 'Scene(background=)' if key < 0 else f'structures[{key}]'
        if type(material) is not Material:
            if not substitute:
                continue
            raise ValueError(
                f'{label} has a traced value in an fd.{type(material).__name__}; in a scene '
                f'{where}, only a plain fd.Material\'s eps and mu may be traced. Keep '
                'conductivities and pole parameters concrete.')
        for name, triple in (('eps', material.eps), ('mu', material.mu_value)):
            if (_carries(triple) and not triple[0] is triple[1] is triple[2]
                    and _read_as_one_number(scene, plan, key)):
                raise ValueError(
                    f'{label} has a traced {name} given per axis; in a scene {where}, a '
                    f'traced {name} is one scalar, {name}=value, the same on all three axes.')
        if key < 0:
            matched = any(scene.boundaries.face(face).background is None for face in pml)
            if (matched or any(isinstance(s, (PlaneWave, TFSF)) for s in scene.sources)
                    or any(isinstance(m, FarField) and m.propagation == 'yee'
                           for m in scene.monitors)):
                raise ValueError(
                    f'Scene(background=) has a traced value in a scene {where}, and a PML '
                    'matched to the background, a PlaneWave or TFSF source or a '
                    'FarField(propagation=\'yee\') reads it: the PML match, the plane-wave '
                    'column\'s transverse wavevector and drive, the grazing survey and the '
                    'lattice far-field map are built from it on the host. Give each PML a '
                    'concrete background=, or differentiate a structure\'s material instead.')
        elif label in stabilized:
            raise ValueError(
                f'{label} has a traced material value and fills a stabilized PML layer along '
                f'the PlaneWave\'s axis; in a scene {where}, the grazing rule judges the orders '
                'a Diffraction monitor asks for there on the face\'s exact return, on the host '
                'only, at the values the plan was built with. Keep that material concrete, or '
                'the face a pure stretch (stabilize=0), whose rule is checked at run time.')
        seed = seeds.get(key)
        if not substitute and seed is None:
            continue
        if seed is None or (seed[1] is None) != (material.mu is None):
            raise ValueError(
                f'{label}\'s material carries a traced value, and plan= was not built with a '
                'plain concrete fd.Material there'
                + ('' if seed is None else ' naming mu the same way')
                + '. Build the plan with fd.plan(scene) from the same scene at concrete values.')
        eps = tuple(s if _carries(v) else v for v, s in zip(material.eps, seed[0]))
        mu = None if material.mu is None else tuple(
            s if _carries(v) else v for v, s in zip(material.mu, seed[1]))
        if key < 0:
            background = Material(eps=eps, mu=mu)
        else:
            structures[key] = structures[key].replace(material=Material(eps=eps, mu=mu))
    if any(_carries(as_material(m)) for _, m in _materials(scene)):
        for term in plan.constitutive:
            if isinstance(term, StencilTerm):
                continue
            for a, (lo, hi) in enumerate(reversed(term.box)):
                for side, touches in (('-', lo == 0), ('+', hi == grid.n[a])):
                    if (touches and plan.faces['xyz'[a] + side].kind == 'symmetry'
                            and _mirror_could_couple(scene, plan, term, a, side)):
                        raise ValueError(
                            f'{term.name} reaches the Symmetry plane on {"xyz"[a] + side!r} '
                            f'and a material value is traced; in a scene {where}, the mirror '
                            'check reads cross coefficients that vanish where the plan\'s own '
                            'values have no contrast, and a smoothed surface reaches the '
                            'mirror\'s cell layer here. Keep the surface a cell clear of the '
                            'mirror, or the materials concrete.')
        import wavebit.fdtd._post as _post
        from wavebit.fdtd._legacy_monitors import Diffraction
        from wavebit.fdtd._prepare.monitors import diffraction_layout
        for m in scene.monitors:
            if not isinstance(m, Diffraction) or m.orders is not None:
                continue
            layout, _ = diffraction_layout(m, scene, grid)
            eps, mu = _post.medium_at(scene, grid, layout.P, (layout.k - 1, layout.k),
                                      f'Diffraction({m.name!r})')
            if _carries((eps, mu)):
                raise ValueError(
                    f'Diffraction({m.name!r}) counts its propagating orders in a medium whose '
                    f'material value is traced; in a scene {where}, give orders=(M_u, M_v): '
                    'the window is a static choice, and a traced index cannot be counted '
                    'after the run.')
    if not substitute:
        return scene
    out = scene.replace(structures=tuple(structures))
    return out if background is None else out.replace(background=background)


def planning_scene(scene, plan):
    """A concrete preflight of all unchanged inputs, after certifying the full allocation.

    A seed proves nothing about another geometry. Fixed, interior supports with disjoint
    interpolation halos remove overlap, seam and wall cases for every geometry in every
    range. Runtime guards then check actual sizes, containment, star positivity and CFL.
    All materials and unchanged scene inputs are rechecked on the host. Interacting or
    nested material interfaces and geometry-dependent sources/monitors remain outside this
    certificate.

    Every ranged body, traced or concrete, keeps the full certificate: an isotropic lossless
    Material (each of eps and mu one number, a traced one given as a scalar), a halo strictly
    inside the domain and clear of absorbers, and pairwise disjoint halos. A structure
    **without** a range is concrete by construction (its shape refuses a traced length) and is
    planned and lowered at its own geometry like any concrete scene's, with any material such
    a scene admits -- a cell-snapped substrate running into a PML, a conductive or dispersive
    layer. It needs only to stay off every traced body's halo: its support widened by the
    same interpolation cell must not share a cell with a traced body's halo, so nothing it
    paints or mixes reaches a cell whose coefficients move. A flush junction between two
    concrete partners beside a traced body is planned and lowered as in a concrete scene;
    ``junction.refuse_mismatch`` compares it through this function's concrete scene, the one
    the plan was built from.

    **Traced material values** -- a body's, a partner's or the background's ``eps`` or ``mu``
    -- are planned at the plan's own values and lowered live (:func:`_seeded_materials`,
    which also lists what a host rule reads and so stays concrete). The same substitution
    serves a scene whose geometry is concrete when its plan certifies smoothed coupling from
    material numbers (:func:`_coupled`); every other scene plans its traced values live, as
    it always has.

    **The one partner a traced body may touch** is the face it stands on in a recorded
    junction (:func:`_certify_junction`): an unranged, concrete, unturned smoothed Box, under
    the ``'split'`` rule. The junction's write set then moves with the rim, so it is
    recomputed from the live geometry inside the record's static rows
    (``junction.live_write``), and ``guard`` keeps the base on the face. A traced body is
    never the face of a junction.

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
        if plan is None or not materials_traced(scene) or not _coupled(plan):
            return scene
        return _seeded_materials(scene, plan, 'with smoothed structures')
    where = name(scene)
    if plan is None or not getattr(plan, '_shape_seeds', ()):
        raise ValueError(f'A traced {where} requires plan= built by fd.plan(scene) from '
                         'concrete geometry before differentiation.')
    refuse_mismatch(scene, plan)
    from wavebit.materials.time_domain import Material
    from wavebit.materials.time_domain import as_material
    if scene.initial is not None:
        raise ValueError(f'A traced {where} supports ranged Sphere/Cylinder/Box/Prism bodies '
                         'with disjoint interpolation halos, beside concrete partners clear of '
                         'them, and no initial field.')
    ranged = [ranged_body(s.shape) for s in scene.structures]
    for material in (as_material(scene.background),
                     *(s.material for s, r in zip(scene.structures, ranged) if r)):
        if (type(material) is not Material
                or not all(_one_number(t) for t in (material.eps, material.mu_value))):
            raise ValueError(f'A traced {where} requires isotropic lossless Material: a plain '
                             'fd.Material whose eps and mu are each one number (a traced '
                             'value given as one scalar).')
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
    standing = {}
    for record in plan.junctions:
        if record.face in moving:
            raise ValueError(f'structures[{record.face}] carries traced geometry and is the '
                             f'face structures[{record.standing}] stands on; a junction is '
                             'differentiated through its standing body only. Keep the face '
                             'concrete.')
        if record.standing in moving:
            _certify_junction(scene, plan, record, moving[record.standing])
            standing[record.standing] = record.face
    for i, structure in enumerate(scene.structures):
        if ranged[i]:
            continue
        reach = tuple((lo-1, hi+1) for lo, hi in structure.shape.support(plan.grid))
        for j, halo in moving.items():
            if standing.get(j) == i:
                continue                    # the certified face of j's own junction
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
    return _seeded_materials(scene.replace(structures=tuple(structures)), plan,
                             f'with a traced {where}')


def _certify_junction(scene, plan, record, halo):
    """What ``junction.classify`` proved of the seed, made to hold for the whole range.

    The traced body stands on the face of an unranged, concrete, unturned smoothed Box (the
    record says so for the seed; the shapes and the plane are static). Its rule must be
    ``'split'``: a staircase owner is a static per-point choice. Its reserved halo and the
    record's rows, together, must lie over the face's extent on both transverse axes with
    every voxel's half cell -- so the rim never meets the substrate's edge -- and the face's
    far side must lie a cell beyond both along the face normal, so no written or mixed voxel
    reaches it. The base stays on the plane through ``guard``; with the four-cell height
    floor, the pillar's far end stays outside the rows (at most 3.5 cells from the plane).
    """
    from wavebit.fdtd._prepare.junction import SPLIT
    from wavebit.fdtd._prepare.junction import TRANSVERSE
    grid = plan.grid
    face_index, j = record.face, record.standing
    if record.rule != SPLIT:
        raise ValueError(f'structures[{j}] carries traced geometry and stands flush with '
                         f"junction={record.rule!r}; a traced standing body takes 'split' "
                         'only, whose values are smooth in the geometry. Leave junction= out.')
    # A ranged face never arrives here: two ranged bodies that touch share halo cells, which
    # the pairwise rule above has already refused.
    face = scene.structures[face_index].shape
    lo = [min(a[0], b[0]) for a, b in zip(halo, record.rows)]
    hi = [max(a[1], b[1]) for a, b in zip(halo, record.rows)]
    for b in TRANSVERSE[record.axis]:
        if not (face.low[b] <= grid.low[b] + (lo[b]-.5)*grid.dx
                and face.high[b] >= grid.low[b] + (hi[b]+.5)*grid.dx):
            raise ValueError(
                f'The traced structures[{j}] stands on structures[{face_index}], but its '
                f'reserved halo and junction rows (cells {lo[b]}..{hi[b]-1} along '
                f'{"xyz"[b]}) reach past that face\'s edge: the rim could meet the edge '
                'somewhere in the range. Widen the substrate or narrow the range.')
    a = record.axis
    far = face.low[a] if record.side == 1 else face.high[a]
    cell = (far - grid.low[a]) / grid.dx
    if lo[a] - 1 <= cell <= hi[a] + 1:
        raise ValueError(
            f'The traced structures[{j}] stands on structures[{face_index}], whose far face '
            f'at {"xyz"[a]} = {far:g} lies within a cell of the reserved halo and junction '
            f'rows (cells {lo[a]}..{hi[a]-1}); a voxel there would hold that face too. '
            'Thicken the substrate.')


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
    from wavebit.fdtd._legacy_monitors import Diffraction
    from wavebit.fdtd._prepare.monitors import diffraction_layout
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
    """Direct lower() has the same preflight as run(), not just a matching allocation.

    A scene whose geometry is concrete lowers its traced material values against the
    caller's plan without a host plan rebuilt here, as it always has; beside smoothed
    coupling it first hears the refusals of what no runtime guard checks
    (:func:`_seeded_materials` with ``substitute=False``), which ``fd.run`` asks too.
    """
    if not traced(scene):
        if seeded(scene, plan):
            _seeded_materials(scene, plan, 'with smoothed structures', substitute=False)
        return
    concrete = planning_scene(scene, plan)
    from wavebit.fdtd._prepare.plan import plan as make_plan
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
    from wavebit.fdtd._prepare.checks import CELLS_ACROSS_REFUSED
    from wavebit.fdtd._prepare.checks import _ACROSS_SLACK
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


def seeded(scene, plan):
    """Whether host planning reads ``scene`` through :func:`planning_scene`'s seeds."""
    return traced(scene) or (plan is not None and materials_traced(scene) and _coupled(plan))


def _floor_index(scene, plan):
    """The largest index ``checks.check_wavelength_resolution`` admits, ``w / 8 dx`` at the
    shortest of the sources' ``band.shortest_wavelength`` and every ``Diffraction``
    wavelength; None when no such wavelength is concrete."""
    from wavebit.fdtd._legacy_monitors import Diffraction
    from wavebit.fdtd._prepare.sources import shortest_wavelength
    wavelengths = [w for m in scene.monitors if isinstance(m, Diffraction)
                   for w in m.wavelengths]
    band = shortest_wavelength(scene)
    if band is not None:
        wavelengths.append(band)
    return min(wavelengths) / (8.0 * plan.dx) if wavelengths else None


def grazing_guard(scene, plan, rt, previous):
    """The grazing rule, asked of every traced medium in a judged PML layer at run time.

    ``grazing.enforce`` refuses an order a pure-stretch PML face cannot absorb -- one that
    leaves at an exit cos below what the face accepts -- counting the orders in each medium
    filling the layer at the plan's values. A traced plain Material there is held to the same
    rule here: its live ``min_c eps_c mu_c`` may not fall in any interval
    ``grazing.refused_bands`` finds (orders counted up to the resolution floor,
    :func:`resolution_guard`'s, above which the run is ``nan`` already), or the run's
    coefficients are ``nan``. Built only for a seeded scene with a traced material in such a
    layer, so every other run's ``stable`` is unchanged.
    """
    if not (materials_traced(scene) and seeded(scene, plan)):
        return previous
    from wavebit.materials.time_domain import Material
    from wavebit.materials.time_domain import as_material
    from wavebit.fdtd._prepare.grazing import refused_bands
    floor = _floor_index(scene, plan)
    if floor is None:
        return previous
    bands = refused_bands(scene, plan, floor)
    valid = None
    for key, material in _materials(scene):
        material = as_material(material)
        if type(material) is not Material or not _carries(material):
            continue
        label = 'the background' if key < 0 else f'structures[{key}]'
        intervals = [b for (_, medium), held in bands.items() if medium == label for b in held]
        if not intervals:
            continue
        index = None
        for e, m in zip(material.eps, material.mu_value):
            product = rt.array(e, 'float64') * rt.array(m, 'float64')
            index = product if index is None else rt.xp.minimum(index, product)
        for low, high in intervals:
            ok = ~((index > low) & (index < high))
            valid = ok if valid is None else valid & ok
    if valid is None:
        return previous
    value = rt.constant(rt.xp.where(valid, rt.array(1., 'float64'),
                                    rt.array(float('nan'), 'float64')))
    return value if previous is None else previous * value


def resolution_guard(scene, plan, rt, previous):
    """The plan's resolution door, asked of every traced medium at run time.

    ``checks.check_wavelength_resolution`` refuses a material sampled below 8 cells per
    wavelength at the sources' ``band.shortest_wavelength`` and at each ``Diffraction``
    wavelength: there not even the first digit of an order survives. Host planning read a
    traced value at the plan's seed (:func:`_seeded_materials`), so the live value is held
    to the same floor here: ``max_c eps_c mu_c <= (w / 8 dx)^2`` at the shortest such
    concrete wavelength ``w``, or the run's coefficients are ``nan``. Built only for a
    seeded scene with a traced material, so every other run's ``stable`` is unchanged.
    """
    if not (materials_traced(scene) and seeded(scene, plan)):
        return previous
    from wavebit.materials.time_domain import Material
    from wavebit.materials.time_domain import as_material
    floor = _floor_index(scene, plan)
    if floor is None:
        return previous
    limit = rt.array(floor ** 2, 'float64')
    valid = None
    for _, material in _materials(scene):
        material = as_material(material)
        if type(material) is not Material or not _carries(material):
            continue                    # a medium with poles or loss has its own index
        for e, m in zip(material.eps, material.mu_value):
            ok = rt.array(e, 'float64') * rt.array(m, 'float64') <= limit
            valid = ok if valid is None else valid & ok
    if valid is None:
        return previous                 # only a conductivity or a pole parameter is traced
    value = rt.constant(rt.xp.where(valid, rt.array(1., 'float64'),
                                    rt.array(float('nan'), 'float64')))
    return value if previous is None else previous * value


def guard(scene, plan, rt, previous):
    """Every ranged body's runtime geometry certificate, combined without host truth tests.

    A partner without a range is concrete and was checked on the host like any concrete
    scene's structure; it has no range to be contained in. A body standing in a recorded
    junction must keep its base on the face's plane, to the flush test's own tolerance
    (``junction._roles``): the junction's rows, rule and certificate assume it.
    """
    if not traced(scene):
        return previous
    from wavebit.fdtd._prepare.checks import _CLIP_ROUNDOFF
    valid = None
    for structure in scene.structures:
        if not ranged_body(structure.shape):
            continue
        current = _valid(structure.shape, plan, rt)
        valid = current if valid is None else valid & current
    for record in plan.junctions:
        shape = scene.structures[record.standing].shape
        if not any(_shape_traced(v) for v in leaves(shape)):
            continue
        a = record.axis
        extent = shape.height if isinstance(shape, Cylinder) else shape.size[a]
        base = (rt.array(shape.center[a], 'float64')
                - record.side * rt.array(extent, 'float64') / 2)
        valid = valid & (rt.xp.abs(base - record.plane) <= _CLIP_ROUNDOFF * plan.dx)
    value = rt.constant(rt.xp.where(valid, rt.array(1., 'float64'), rt.array(float('nan'), 'float64')))
    return value if previous is None else previous * value
