"""Plan and lower compact inverse-material cross coefficients.

Public smoothed geometry is Sphere and, from C4, the extruded Box, Cylinder
and Prism. Their cross moment is the nearest-surface tangent normal; for an
unturned Box it is exactly zero. Curved surface moments and each body have
separate accuracy gates.
"""
import math

import numpy as np

from .._kernel.constitutive import ConstitutiveTerm, cross_action
from .._kernel.masks import Region, Slab
from ..materials import Conductor, Dispersive
from .coefficients import _paste, build, materials_of
from .occupancy import smoothed_structures, yee_points


def common_points(grid, box, field):
    """The E node or H cell centre shared by three staggered components."""
    points = yee_points(grid, box, field, 0)
    points[..., 0] += (-0.5 if field == 'E' else 0.5) * grid.dx
    return points


def _intersect(a, b):
    box = tuple((max(u[0], v[0]), min(u[1], v[1])) for u, v in zip(a, b))
    return box if all(hi > lo for lo, hi in box) else None


def cross_free(body):
    """Whether ``body``'s cross coefficients are zero everywhere: an unturned ``fd.Box``.

    Every face of an axis-aligned box has ``n_i n_j = 0`` for ``i != j``, including the voxels
    that hold more than one face, so :func:`lower` writes zero weights for it and a
    ``ConstitutiveTerm`` over it would add exactly nothing -- the term adds only the
    off-diagonal increment. So :func:`terms` plans none for it: no flux memory, no flux work,
    and nothing for an initial field to be refused over. ``angle`` is a Python float in C4; a
    traced one (C6) has to decide this statically. A box turned by 90 or 360 degrees answers
    ``False`` and gets the nearest-surface proxy, as ``fd.Box``'s docstring says.
    """
    from .._shapes import Box
    return isinstance(body, Box) and body.angle == 0


def terms(scene, plan):
    """Merge intersecting supports, including one interpolation-cell halo.

    One cell is the exact reach of the two adjacent half-cell averages, not
    an empirical cutoff. Disjoint regions own disjoint flux memory and writes.
    A body whose cross coefficients are zero everywhere (:func:`cross_free`) owns none.
    """
    boxes = []
    for _, structure, body in smoothed_structures(scene):
        if cross_free(body):
            continue
        box = tuple((max(0, lo-1), min(n, hi+1))
                    for (lo, hi), n in zip(structure.shape.support(plan.grid), plan.grid.n))
        i = 0
        while i < len(boxes):
            if _intersect(box, boxes[i]) is None:
                i += 1
                continue
            other = boxes.pop(i)
            box = tuple((min(u[0], v[0]), max(u[1], v[1])) for u, v in zip(box, other))
            i = 0
        boxes.append(box)
    background, structures = materials_of(scene)
    magnetic = background.is_magnetic or any(s.material.is_magnetic for s in structures)
    out = []
    for i, box in enumerate(sorted(boxes)):
        for field in (('E', 'H') if magnetic else ('E',)):
            masks = []
            for mask in plan.masks:
                if mask.field != field:
                    continue
                slab = mask.slab
                cut = _intersect(box, (slab.x, slab.y, slab.z))
                if cut is not None:
                    x, y, z = tuple((lo-base[0], hi-base[0])
                                    for (lo, hi), base in zip(cut, box))
                    masks.append(Slab(c=slab.c, z=z, y=y, x=x))
            out.append(ConstitutiveTerm(f'constitutive[{field}{i}]', field,
                       Region(z=box[2], y=box[1], x=box[0]), plan.dt/plan.dx, tuple(masks)))
    return tuple(out)


def lower(scene, plan, rt, report=None, proxy_stats=None, proxy_by_index=None, nodes=None):
    """Paint node coefficients in scene order, preserving the two-phase rule.

    ``proxy_stats`` collects the nearest-surface proxy's worst gap per body type (the plan's
    warning), ``proxy_by_index`` the same per structure index (``fd.check``'s block, C5).
    ``nodes``, when given, receives per term the node tensor's diagonal ``(3, nz, ny, nx)``
    and its ``1/<eps>`` ``(nz, ny, nx)``, painted by the same rule as the cross coefficients:
    ``1/<eps> + (<1/eps> - 1/<eps>) n_a^2`` where the body that last reaches a node has a fill
    beyond the clip's round-off (the cross coefficients' own test, and the doors': a fill within
    ``checks._CLIP_ROUNDOFF`` of zero, either sign, leaves the node as the earlier bodies left
    it), ``1/value`` under a cell-snapped shape, ``1/background`` where nothing does.
    With the cross coefficients off its diagonal it is that node's Kottke tensor, what the
    star lift moves each node's star toward (``stars.lift``).
    """
    from .._shapes import Box, Sphere, occupancy
    from .checks import _CLIP_ROUNDOFF
    from .._kernel.stencil import StencilTerm
    background, structures = materials_of(scene)
    xp, out = rt.xp, {}
    for term in plan.constitutive:
        if isinstance(term, StencilTerm):
            # A prepared extended-stencil region has no diagonal-rule cross coefficients:
            # its whole operator, diagonal and off-diagonal, was assembled on the host at
            # plan time (``stencil.prepare``) and reaches the kernel through ``be`` and
            # ``params['stencil']``. Building a cross array here would allocate a
            # region-sized array of zeros that nothing reads.
            continue
        box = tuple(reversed(term.box))
        field = term.field
        prop = lambda material: (material.eps if field == 'E' else material.mu_value)[0]
        held = rt.zeros(term.region.shape, 'float64') + rt.array(prop(background), 'float64')
        cross = rt.zeros((3,) + term.region.shape, 'float64')
        tensor = xp.stack([1.0/held]*3)
        grounds = {}
        for index, structure in enumerate(structures):
            for record in plan.junctions:
                if index == min(record.face, record.standing):
                    grounds[record] = held
            cut = _intersect(box, structure.shape.support(plan.grid))
            if cut is None:
                continue
            local = tuple((lo-base[0], hi-base[0]) for (lo, hi), base in zip(cut, box))
            view = tuple(slice(*s) for s in reversed(local))
            value = rt.array(prop(structure.material), 'float64')
            body = structure.shape.body()
            old = held[view]
            if body is None:
                # A snapped shape replaces whole cells, including any previous cross term.
                fresh = xp.zeros_like(old) + value
                off = rt.zeros((3,) + tuple(hi-lo for lo, hi in reversed(local)), 'float64')
                diagonal = xp.stack([1.0/fresh]*3)
            else:
                points = rt.array(common_points(plan.grid, cut, field), 'float64')
                fill, moment = occupancy(body, points, plan.dx, report=report)
                _, normal = body.surface(points)
                arithmetic = fill*value + (1-fill)*old
                # Algebraically <1/eps>-1/<eps>, factored to avoid cancellation
                # at zero contrast and at exactly empty/full voxels.
                contrast = fill*(1-fill)*(value-old)**2 / (value*old*arithmetic)
                weights = xp.stack([normal[..., i]*normal[..., j]
                                    for i, j in ((0, 1), (0, 2), (1, 2))])
                if cross_free(body):
                    # Every face of an axis-aligned box has n_i*n_j=0 for i!=j,
                    # including voxels that contain more than one face; this one
                    # shares a region some other body's term owns.
                    weights = xp.zeros_like(weights)
                elif proxy_stats is not None and not isinstance(body, Sphere):
                    # This host-only measurement belongs to planning, never a
                    # transformed lowering. It exposes the custom-body proxy
                    # where its supplied moments describe more than one face.
                    active = (fill > _CLIP_ROUNDOFF) & (fill < 1-_CLIP_ROUNDOFF)
                    if plan.junctions and field == "E":
                        from .junction import proxy_mask
                        active = active & proxy_mask(plan, index, cut)
                    gap = np.abs(np.asarray(moment)-np.asarray(normal)**2)
                    worst = float(gap[np.asarray(active)].max(initial=0.))
                    name = type(body).__name__
                    proxy_stats[name] = max(proxy_stats.get(name, 0.), worst)
                    if proxy_by_index is not None:
                        proxy_by_index[index] = max(proxy_by_index.get(index, 0.), worst)
                # A fill within the clip's round-off of zero does not reach the node: the
                # doors read it so, and an exact ``fill == 0`` did not. A cylinder raised half
                # a cell above a smoothed substrate has its base on the plane nodes' voxel
                # face, fill 4e-15 there, and it overwrote the substrate's node tensor: Xi_zz
                # 1/<eps> = 0.615 where the substrate's <1/eps> is 0.722.
                empty = xp.abs(fill) <= _CLIP_ROUNDOFF
                off = xp.where(empty[None], cross[(slice(None),) + view],
                               contrast[None]*weights)
                diagonal = xp.where(empty[None], tensor[(slice(None),) + view],
                                    xp.stack([1.0/arithmetic + contrast*normal[..., a]**2
                                              for a in range(3)]))
                fresh = xp.where(empty, old, arithmetic)
            held = _paste(xp, held, local, fresh)
            cross = _paste(xp, cross, local, off)
            tensor = _paste(xp, tensor, local, diagonal)
        if plan.junctions:
            from .junction import paint_nodes
            for record in plan.junctions:
                tensor, cross, held = paint_nodes(rt, record, scene, plan, term,
                                                   tensor, cross, held, grounds[record])
        out[term.name] = cross
        if nodes is not None:
            nodes[term.name] = (tensor, 1.0/held)
    return out


def lift(plan, params, nodes, rt, lifted=None):
    """Patch ``params['be']`` / ``params['bh']`` with the star lift of every term.

    Returns whether every star of every term is positive definite at its lift (a Python bool
    on concrete values, a runtime bool under a transform), the certificate ``validate`` refuses
    on and ``guard`` turns into NaN. ``nodes`` is what :func:`lower` filled. The diagonal is
    changed only over a region where some star moves (``stars.lift``). ``minus_bh`` is not
    touched here: ``params.lower`` rebuilds it from the final ``bh`` after the guard, which
    every plan with a coupled term has. ``lifted``, when given, receives each term's
    ``stars.Lifted`` by name.

    The change goes into ``be`` and not beside it, for the reason ``params._stencil`` gives: ``be``
    is what the step multiplies the curl by and what ``InteriorEnergy`` reads the medium from.
    """
    from .._kernel.observe import InteriorEnergy
    from .._kernel.stencil import StencilTerm
    from .stars import lift as lift_stars
    xp, certified = rt.xp, True
    for term in plan.constitutive:
        if isinstance(term, StencilTerm):
            continue
        key = 'be' if term.field == 'E' else 'bh'
        cross = params['constitutive'][term.name]
        diagonal = xp.broadcast_to(InteriorEnergy._cut(params[key], slice(None), term.box)
                                   / term.ratio, cross.shape)
        out = lift_stars(term.field, diagonal, cross, *nodes[term.name], rt)
        if lifted is not None:
            lifted[term.name] = out
        certified = certified & out.certified
        if out.delta is None:
            continue
        coefficient = xp.broadcast_to(params[key], (3, plan.nz, plan.ny, plan.nx))
        block = coefficient[term.view] + term.ratio * out.delta
        params[key] = _paste(xp, coefficient, tuple(reversed(term.box)), block)
    return certified


def lifted_on_host(scene, plan):
    """``(params, {term name: stars.Lifted})`` on NumPy: the coefficients with the lift
    patched in and each term's lift, for ``validate`` and for the plan's notes."""
    from .._runtime import runtime
    rt, nodes, lifted = runtime('numpy'), {}, {}
    params = dict(build(scene, plan, rt))
    params['constitutive'] = lower(scene, plan, rt, nodes=nodes)
    lift(plan, params, nodes, rt, lifted)
    return params, lifted


def enclosure(diagonal, cross, xp, field='E'):
    """Gershgorin bounds: diagonal plus/minus the absolute off-diagonal row sum."""
    radius = cross_action(xp.ones_like(diagonal), xp.abs(cross), xp, field, bounded=True)
    return xp.min(diagonal-radius), xp.max(diagonal+radius)


def validate(scene, plan):
    """Refuse unsupported memory regions and uncertified operators once, at plan time.

    Returns ``{term name: stars.Lifted}``, the lift of every diagonal-rule term, which the
    plan's notes report (``checks._lift_notes``); ``None`` when there is no such term.

    Only the diagonal rule's terms: a prepared extended-stencil term carries its own
    certificate, taken on the whole assembled operator before any field is advanced
    (``stencil.prepare`` -- the consistency residual, the minimum eigenvalue, the LDL energy
    bound, the identity halo and its own spectral time-step rate), and the enclosure below
    reads cross coefficients it does not have.
    """
    from .._kernel.stencil import StencilTerm
    coupled = tuple(t for t in plan.constitutive if not isinstance(t, StencilTerm))
    if not coupled:
        return None
    from .._runtime import runtime
    from .._kernel.observe import InteriorEnergy
    background, structures = materials_of(scene)
    for term in coupled:
        if getattr(plan, 'has_' + term.field + '0'):
            raise ValueError(f'InitialField.{term.field} with smoothed constitutive coupling '
                             'needs an inverse material solve to initialize D/B. That solve '
                             'is not implemented. Excite this field with a source instead; '
                             'a diagonal division would initialize the wrong energy.')
        box = tuple(reversed(term.box))
        for name, cells in plan.absorbing_faces:
            a = 'xyz'.index(name[0])
            lo, hi = box[a]
            reaches_layer = lo < cells if name[1] == '-' else hi > plan.grid.n[a]-cells
            if reaches_layer:
                raise ValueError(f'{term.name} interpolation support meets the {name} '
                                 'absorbing layer. Constitutive D/B memory currently needs '
                                 'lossless cells; leave one interpolation cell beyond the '
                                 'smoothed support clear of the layer. Crossing it is C9.')
        for i, structure in enumerate(structures):
            if (_intersect(box, structure.shape.support(plan.grid)) is not None
                    and isinstance(structure.material, (Conductor, Dispersive))):
                raise ValueError(f'{term.name} interpolation support meets structures[{i}] '
                                 'with conductivity or poles. Coupled lossless D/B memory '
                                 'cannot integrate this material; separate their supports. '
                                 'Dispersive smoothing is C8.')
    rt = runtime('numpy')
    try:
        coefficients, lifted = lifted_on_host(scene, plan)
        cross = coefficients['constitutive']
    except np.linalg.LinAlgError as err:
        raise ValueError(f'the star lift could not solve its eigenproblem ({err}). '
                         f'{_reach(background, structures)}') from err
    except (TypeError, ValueError, RuntimeError) as err:
        raise ValueError('fd.plan requires concrete geometry and materials to certify '
                         'constitutive coupling before numerical lowering. Build the plan '
                         f'from concrete values ({type(err).__name__}: {err}).') from err
    upper = {'E': max(1/float(v) for v in background.eps),
             'H': max(1/float(v) for v in background.mu_value)}
    for s in structures:
        for field, values in (('E', s.material.eps), ('H', s.material.mu_value)):
            upper[field] = max(upper[field], *(1/float(v) for v in values))
    for term in coupled:
        c = cross[term.name]
        # E averages towards the low node, H towards the high cell centre.
        # Zero extension is the exact adjoint on a finite box. Internal region
        # ends must carry no coupling to omitted neighbours. PEC/PMC masks are
        # projected into flux memory too; mirrors need a weighted adjoint.
        for a, (lo, hi) in enumerate(reversed(term.box)):
            at = 0 if term.field == 'E' else c.shape[3-a]-1
            outside = lo == 0 if term.field == 'E' else hi == plan.grid.n[a]
            edge = np.take(c, at, axis=3-a)
            if not outside and np.any(edge != 0):
                raise ValueError(f'{term.name} has cross coefficients at an internal '
                                 'support end; enlarge its interpolation support.')
            for side, end in (('-', 0), ('+', c.shape[3-a]-1)):
                face = plan.faces['xyz'[a]+side]
                touches = lo == 0 if side == '-' else hi == plan.grid.n[a]
                if (touches and face.kind == 'symmetry'
                        and np.any(np.take(c, end, axis=3-a) != 0)):
                    raise ValueError(f'{term.name} reaches a Symmetry plane; cross coupling '
                                     'there needs a mirror-weighted adjoint, which is not '
                                     'written. Leave its surface one cell clear of the '
                                     'mirror: of the smoothed bodies only the unturned fd.Box '
                                     'plans no cross term, and it alone may cross a mirror '
                                     '(symmetric about the plane, checks.check_symmetry).')
        key = 'be' if term.field == 'E' else 'bh'
        if not lifted[term.name].certified:
            raise ValueError(f'{term.name}: a star of the lifted inverse-material operator is '
                             'not positive definite. The lift makes every star positive for '
                             f'positive finite phases within float64\'s reach. '
                             f'{_reach(background, structures)} If the contrast is small, a '
                             'material or a fill reaching the lift is not '
                             'a finite positive number.')
        diagonal = np.broadcast_to(InteriorEnergy._cut(coefficients[key], slice(None),
                                                       term.box)/term.ratio, c.shape)
        _, hi = (float(v) for v in enclosure(diagonal, c, np, term.field))
        if not math.isfinite(hi):
            raise ValueError(f'{term.name} inverse-material spectrum has no finite upper '
                             f'bound ({hi!r}); a material value is not finite.')
        upper[term.field] = max(upper[term.field], hi)
    from .checks import cfl_axis_sum
    rate = pole_rate(scene, rt)
    bound = 1 / math.sqrt(cfl_axis_sum(scene, plan.dx)*upper['E']*upper['H'] + float(rate)/4)
    if not plan.dt < bound:
        raise ValueError(f'Constitutive CFL bound dt < {bound:.17g} is not met by '
                         f'dt={plan.dt:.17g}; upper inverse-material eigenvalue bounds '
                         f'are E={upper["E"]:.17g}, H={upper["H"]:.17g}. '
                         f'Lower Run(courant=...) below {bound*math.sqrt(3)/plan.dx:.17g}.')
    return lifted


def _reach(background, structures):
    """The sentence that says how far the scene's contrast is from float64's reach."""
    values = [float(v) for m in [background] + [s.material for s in structures]
              for v in tuple(m.eps) + tuple(m.mu_value)]
    contrast = max(values) / min(values) if min(values) > 0 else float('inf')
    return (f'The phases here span a contrast of {contrast:.3g}; from about 1e12 a fill\'s '
            'round-off (about 1e-13) times the contrast breaks the mix itself, which no lift '
            'can repair (measured on spheres: every contrast up to 1e11 certified, refusals from '
            '1e12; wavebit/fd/_lower/stars.py, "Float64\'s reach").')


def pole_rate(scene, rt):
    """The existing resonance-plus-coupling bound, including remote pole regions."""
    background, structures = materials_of(scene)
    resonance, coupling = rt.array(0., 'float64'), rt.array(0., 'float64')
    for material in [background] + [s.material for s in structures]:
        strength = rt.array(0., 'float64')
        for pole in material.poles:
            frequency2, drive = pole.rate()
            resonance = rt.xp.maximum(resonance, rt.array(frequency2, 'float64'))
            strength = strength + rt.array(drive, 'float64')
        eps_min = rt.xp.min(rt.xp.stack([rt.array(v, 'float64') for v in material.eps]))
        coupling = rt.xp.maximum(coupling, strength/eps_min)
    return resonance + coupling


def guard(scene, plan, params, rt, certified):
    """Whole-operator certificate for prepared plans lowered under differentiation.

    Planning reports concrete failures by name. A transformed leaf has no
    Python truth value; the existing stable=NaN contract makes a failed
    certificate visible in results, without a per-cell exception or warning.
    ``certified`` is :func:`lift`'s: every star positive definite at its lift, which is the
    positivity of the operator. The Gershgorin enclosure is read for its upper end only, the
    time step's bound; its lower end was the certificate before the lift, and a loose one.
    """
    from .._kernel.observe import InteriorEnergy
    from .checks import cfl_axis_sum
    xp = rt.xp
    background, structures = materials_of(scene)
    upper = {}
    for field in ('E', 'H'):
        values = [rt.array(v, 'float64')
                  for m in [background] + [s.material for s in structures]
                  for v in (m.eps if field == 'E' else m.mu_value)]
        upper[field] = 1 / xp.min(xp.stack(values))
    from .._kernel.stencil import StencilTerm
    valid = rt.array(True) & certified
    for term in plan.constitutive:
        if isinstance(term, StencilTerm):
            continue                      # certified on the host at plan time; see validate
        key = 'be' if term.field == 'E' else 'bh'
        cross = params['constitutive'][term.name]
        diagonal = xp.broadcast_to(InteriorEnergy._cut(params[key], slice(None),
                                                       term.box)/term.ratio, cross.shape)
        _, hi = enclosure(diagonal, cross, xp, term.field)
        valid = valid & xp.isfinite(hi)
        upper[term.field] = xp.maximum(upper[term.field], hi)
    valid = valid & (plan.dt**2 * (cfl_axis_sum(scene, plan.dx)*upper['E']*upper['H']
                                 + pole_rate(scene, rt)/4) < 1)
    return rt.constant(xp.where(valid, rt.array(1., 'float64'), rt.array(float('nan'), 'float64')))
