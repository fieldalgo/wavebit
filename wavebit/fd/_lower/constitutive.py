"""Plan and lower compact inverse-material cross coefficients.

Public smoothed geometry currently consists of Sphere. Its cross moment is
the same nearest-surface tangent normal used by the diagonal rule. Curved
surface moments and additional public bodies have separate accuracy gates.
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


def terms(scene, plan):
    """Merge intersecting supports, including one interpolation-cell halo.

    One cell is the exact reach of the two adjacent half-cell averages, not
    an empirical cutoff. Disjoint regions own disjoint flux memory and writes.
    """
    boxes = []
    for _, structure, _ in smoothed_structures(scene):
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


def lower(scene, plan, rt, report=None, proxy_stats=None):
    """Paint node coefficients in scene order, preserving the two-phase rule."""
    from .._shapes import Box, Sphere, occupancy
    background, structures = materials_of(scene)
    xp, out = rt.xp, {}
    for term in plan.constitutive:
        box = tuple(reversed(term.box))
        field = term.field
        prop = lambda material: (material.eps if field == 'E' else material.mu_value)[0]
        held = rt.zeros(term.region.shape, 'float64') + rt.array(prop(background), 'float64')
        cross = rt.zeros((3,) + term.region.shape, 'float64')
        for structure in structures:
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
                if isinstance(body, Box) and body.angle == 0:
                    # Every face of an axis-aligned box has n_i*n_j=0 for i!=j,
                    # including voxels that contain more than one face.
                    weights = xp.zeros_like(weights)
                elif proxy_stats is not None and not isinstance(body, Sphere):
                    # This host-only measurement belongs to planning, never a
                    # transformed lowering. It exposes the custom-body proxy
                    # where its supplied moments describe more than one face.
                    from .checks import _CLIP_ROUNDOFF
                    active = (fill > _CLIP_ROUNDOFF) & (fill < 1-_CLIP_ROUNDOFF)
                    gap = np.abs(np.asarray(moment)-np.asarray(normal)**2)
                    worst = float(gap[np.asarray(active)].max(initial=0.))
                    name = type(body).__name__
                    proxy_stats[name] = max(proxy_stats.get(name, 0.), worst)
                off = xp.where(fill[None] == 0, cross[(slice(None),) + view],
                               contrast[None]*weights)
                fresh = arithmetic
            held = _paste(xp, held, local, fresh)
            cross = _paste(xp, cross, local, off)
        out[term.name] = cross
    return out


def enclosure(diagonal, cross, xp, field='E'):
    """Gershgorin bounds: diagonal plus/minus the absolute off-diagonal row sum."""
    radius = cross_action(xp.ones_like(diagonal), xp.abs(cross), xp, field, bounded=True)
    return xp.min(diagonal-radius), xp.max(diagonal+radius)


def validate(scene, plan):
    """Refuse unsupported memory regions and uncertified operators once, at plan time."""
    if not plan.constitutive:
        return
    from .._runtime import runtime
    from .._kernel.observe import InteriorEnergy
    background, structures = materials_of(scene)
    for term in plan.constitutive:
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
        coefficients, cross = build(scene, plan, rt), lower(scene, plan, rt)
    except (TypeError, ValueError, RuntimeError) as err:
        raise ValueError('fd.plan requires concrete geometry and materials to certify '
                         'constitutive coupling before numerical lowering. Build the plan '
                         f'from concrete values ({type(err).__name__}: {err}).') from err
    upper = {'E': max(1/float(v) for v in background.eps),
             'H': max(1/float(v) for v in background.mu_value)}
    for s in structures:
        for field, values in (('E', s.material.eps), ('H', s.material.mu_value)):
            upper[field] = max(upper[field], *(1/float(v) for v in values))
    for term in plan.constitutive:
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
                                     'there needs a mirror-weighted adjoint. Leave its '
                                     'surface one cell clear of the mirror.')
        key = 'be' if term.field == 'E' else 'bh'
        diagonal = np.broadcast_to(InteriorEnergy._cut(coefficients[key], slice(None),
                                                       term.box)/term.ratio, c.shape)
        lo, hi = (float(v) for v in enclosure(diagonal, c, np, term.field))
        if not (math.isfinite(lo) and math.isfinite(hi) and lo > 0):
            raise ValueError(f'{term.name} inverse-material spectrum is enclosed by '
                             f'[{lo:.17g}, {hi:.17g}]. The positive-energy certificate '
                             'requires a strictly positive lower bound. A failed bound '
                             'is inconclusive, not evidence of a negative eigenvalue; '
                             'refine the grid or reduce the material contrast.')
        upper[term.field] = max(upper[term.field], hi)
    from .checks import cfl_axis_sum
    rate = pole_rate(scene, rt)
    bound = 1 / math.sqrt(cfl_axis_sum(scene, plan.dx)*upper['E']*upper['H'] + float(rate)/4)
    if not plan.dt < bound:
        raise ValueError(f'Constitutive CFL bound dt < {bound:.17g} is not met by '
                         f'dt={plan.dt:.17g}; upper inverse-material eigenvalue bounds '
                         f'are E={upper["E"]:.17g}, H={upper["H"]:.17g}. '
                         f'Lower Run(courant=...) below {bound*math.sqrt(3)/plan.dx:.17g}.')


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


def guard(scene, plan, params, rt):
    """Whole-operator certificate for prepared plans lowered under differentiation.

    Planning reports concrete failures by name. A transformed leaf has no
    Python truth value; the existing stable=NaN contract makes a failed
    certificate visible in results, without a per-cell exception or warning.
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
    valid = rt.array(True)
    for term in plan.constitutive:
        key = 'be' if term.field == 'E' else 'bh'
        cross = params['constitutive'][term.name]
        diagonal = xp.broadcast_to(InteriorEnergy._cut(params[key], slice(None),
                                                       term.box)/term.ratio, cross.shape)
        lo, hi = enclosure(diagonal, cross, xp, term.field)
        valid = valid & (lo > 0) & xp.isfinite(hi)
        upper[term.field] = xp.maximum(upper[term.field], hi)
    valid = valid & (plan.dt**2 * (cfl_axis_sum(scene, plan.dx)*upper['E']*upper['H']
                                 + pole_rate(scene, rt)/4) < 1)
    return rt.constant(xp.where(valid, rt.array(1., 'float64'), rt.array(float('nan'), 'float64')))
