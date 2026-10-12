"""One concrete, axis-aligned conductive film in a lossless isotropic host.

This is a bounded C8 slice, not conductive smoothing for arbitrary bodies. A single
unturned public Box must span two grid axes; its separated faces and support stay
outside absorbing layers. No traced values, initial fields, or material junctions
are admitted. In this geometry the interface tensor is diagonal, so the existing
lossless constitutive cross-coupling operator is neither used nor extended.

For conductor fraction f, g=1-f, conductor (a, sigma), host b, and s=-i*omega,
the tangential response is f*a+g*b+f*sigma/s. The normal response is
    1/(f/(a+sigma/s)+g/b) = b*(a*s+sigma)/(A*s+g*sigma), A=f*b+g*a.
For 0<f<1 this is eps_inf=a*b/A plus a passive Debye pole with
delta=f*b*b/(g*A), tau=A/(g*sigma), and no residual DC conductivity.
This is a direct partial-fraction expansion of the normal harmonic/tangential
arithmetic tensor in Deinega and Valuev, Optics Letters 32, 3429 (2007), Eqs. 1-2:
https://doi.org/10.1364/OL.32.003429 . Equal real eps does not remove this pole.

The split memory is P_tilde=P-c*E, as in materials.Debye. With
D=2*A+dt*g*sigma, its gain=(2*A-dt*g*sigma)/D,
c=f*b*b*sigma*dt/(A*D), and drive=4*f*b*b*sigma*dt/D**2. These
finite forms avoid constructing divergent delta and tau near f=1. Exact empty,
full, and zero-sigma cells have no pole; a full cell uses ordinary conductivity.
The update permittivity includes c. Energy needs inverse_delta and c explicitly,
because reconstructing delta from 1-gain**2 loses accuracy and fails at endpoints.

Only this supported scene classification creates a term. Every other scene returns
None/empty and retains the existing lowering and refusal paths, bit for bit.
"""
import dataclasses
import math

import numpy as np

from ..boundaries import Absorbing, FACES
from ..geometry import Box
from ..materials import Conductor, Material, as_material, concrete
from .._kernel.grid import OFFSETS
from .._kernel.masks import Region, Slab
from .._kernel.poles import PoleTerm

__all__ = ['Film', 'InterfacePoleTerm', 'descriptor', 'terms', 'material_profiles',
           'pole_params', 'extra_rate', 'refuse_mismatch']


@dataclasses.dataclass(frozen=True)
class Film:
    """Host-readable identity and support of the sole admitted film."""

    axis: int
    low: float
    high: float
    eps: float
    sigma: float
    host_eps: float
    box: tuple
    signature: tuple


@dataclasses.dataclass(frozen=True)
class InterfacePoleTerm(PoleTerm):
    """One derived Debye recurrence, identified even when its drive is zero."""

    signature: tuple


def _isotropic(values, *, positive):
    numbers = tuple(concrete(v) for v in values)
    if any(v is None or not math.isfinite(v) for v in numbers):
        return None
    if numbers[0] != numbers[1] or numbers[0] != numbers[2]:
        return None
    value = numbers[0]
    return value if (value > 0 if positive else value >= 0) else None


def descriptor(scene, grid):
    """The exact supported scene, or None so existing doors handle every other case."""
    structures = tuple(getattr(scene, 'structures', ()) or ())
    if len(structures) != 1 or getattr(scene, 'initial', None) is not None:
        return None
    structure = structures[0]
    shape, material = structure.shape, structure.material
    if (type(shape) is not Box or shape.angle != 0 or shape.range is not None
            or structure.discretisation != 'voxel' or type(material) is not Conductor):
        return None
    host = as_material(getattr(scene, 'background', None))
    if type(host) is not Material:
        return None
    # concrete() alone accepts Torch forward-mode duals; this shared door does not.
    from .shape_ranges import materials_traced
    if materials_traced(scene):
        return None
    a = _isotropic(material.eps, positive=True)
    sigma = _isotropic(material.sigma, positive=False)
    b = _isotropic(host.eps, positive=True)
    mu_a = _isotropic(material.mu_value, positive=True)
    mu_b = _isotropic(host.mu_value, positive=True)
    if None in (a, sigma, b) or mu_a != 1.0 or mu_b != 1.0:
        return None
    uniform = tuple(shape.uniform_along(k, (0, n), grid)
                    for k, n in enumerate(grid.n))
    if sum(uniform) != 2:
        return None
    axis = uniform.index(False)
    if shape.size[axis] < grid.dx:
        return None                           # two surfaces may not cut one E voxel
    box = tuple(tuple(int(v) for v in pair) for pair in shape.support(grid))
    lo, hi = box[axis]
    if lo <= 0 or hi >= grid.n[axis] or lo >= hi:
        return None                           # also excludes normal wall half-voxels
    for name in FACES:
        face = scene.boundaries.face(name)
        if not isinstance(face, Absorbing):
            continue
        k = 'xyz'.index(name[0])
        start, end = box[k]
        if (start < face.cells if name[1] == '-' else end > grid.n[k]-face.cells):
            return None
    low = float(shape.center[axis] - shape.size[axis]/2)
    high = float(shape.center[axis] + shape.size[axis]/2)
    signature = ('conductive-film-v1', tuple(float(v) for v in shape.size),
                 tuple(float(v) for v in shape.center), float(shape.angle), shape.axis,
                 a, sigma, b, material.mu is not None, host.mu is not None,
                 structure.discretisation, structure.junction, tuple(grid.n),
                 float(grid.dx), tuple(float(v) for v in grid.low))
    return Film(axis, low, high, a, sigma, b, box, signature)


def terms(scene, grid):
    """One static term, including at sigma=0, or no change to the existing plan."""
    film = descriptor(scene, grid)
    if film is None:
        return ()
    region = Region(z=film.box[2], y=film.box[1], x=film.box[0])
    slabs = tuple(Slab(c=c, z=region.z, y=region.y, x=region.x) for c in range(3))
    return (InterfacePoleTerm(name='conductive[0]', region=region, slabs=slabs,
                              n_poles=1, signature=film.signature),)


def refuse_mismatch(scene, plan):
    """A prepared film plan cannot lose its memory or change its concrete inputs."""
    # Stack terms inherit InterfacePoleTerm. Delegate only after every real
    # member passes the stack descriptor; its guard checks all pole memories.
    from .conductive_stack import descriptor as stack_descriptor, refuse_mismatch as stack_guard
    if stack_descriptor(scene, plan.grid) is not None:
        stack_guard(scene, plan)
        return
    held = tuple(t for t in plan.terms
                 if isinstance(t, InterfacePoleTerm)
                 or getattr(t, 'name', '').startswith('conductive['))
    expected = terms(scene, plan.grid)
    if not expected and any(s.discretisation == 'voxel' and isinstance(s.material, Conductor)
                            and s.shape.body() is not None
                            for s in (getattr(scene, 'structures', ()) or ())):
        raise ValueError('conductive smoothing is limited to one concrete axis-aligned film '
                         'in a lossless isotropic host, with separated interfaces outside '
                         'absorbing layers and no initial fields or differentiation. '
                         'A prepared non-film plan cannot bypass this bounded C8 scope.')
    if held != expected:
        raise ValueError('the prepared conductive-film plan does not match this scene. '
                         'This bounded C8 film takes concrete geometry and material values; '
                         'traced or changed values and missing/extra interface memory need '
                         'a new supported fd.plan(scene).')


def _fill(film, grid, component):
    """Analytic 1-D overlap at this E component's own Yee position; no near-endpoint snap."""
    k = film.axis
    points = grid.low[k] + (np.arange(grid.n[k]) + .5*OFFSETS['E'][component][k])*grid.dx
    left, right = points-grid.dx/2, points+grid.dx/2
    overlap = np.minimum(right, film.high)-np.maximum(left, film.low)
    fill = np.clip(overlap/grid.dx, 0.0, 1.0)
    # A wholly contained voxel is exactly full, independent of cancellation in right-left.
    # These are geometric endpoint predicates, never an epsilon-wide snap of a fraction.
    fill = np.where((left >= film.low) & (right <= film.high), 1.0, fill)
    return np.where((right <= film.low) | (left >= film.high), 0.0, fill)


def _compact(values, axis):
    shape = [3, 1, 1, 1]
    shape[3-axis] = values.shape[1]
    return values.reshape(shape)


def _profiles(film, grid, dt):
    """Concrete compact profiles; fail before lowering if derived coefficients overflow."""
    n = grid.n[film.axis]
    eps = np.full((3, n), film.host_eps)
    sigma, gain, drive, shift, inverse, rates = (np.zeros((3, n)) for _ in range(6))
    a, b, conductivity = film.eps, film.host_eps, film.sigma
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            for component in range(3):
                f = _fill(film, grid, component)
                g = 1.0-f
                if component != film.axis:
                    eps[component] = f*a+g*b
                    sigma[component] = f*conductivity
                else:
                    A = f*b+g*a
                    eps[component] = a*b/A
                    sigma[component] = np.where(f == 1.0, conductivity, 0.0)
                    live = (f > 0) & (f < 1) & (conductivity > 0)
                    fl, gl, al = f[live], g[live], A[live]
                    D = 2*al+dt*gl*conductivity
                    gain[component, live] = (2*al-dt*gl*conductivity)/D
                    shift[component, live] = fl*b*b*conductivity*dt/(al*D)
                    drive[component, live] = 4*fl*b*b*conductivity*dt/(D*D)
                    inverse[component, live] = gl*al/(fl*b*b)
                    rates[component, live] = fl*gl*b*conductivity**2/(a*al*al)
                    eps[component] += shift[component]
                eps[component, f == 0.0] = b
                eps[component, f == 1.0] = a
    except (FloatingPointError, OverflowError) as error:
        raise ValueError('conductive-film derived coefficients exceed finite float64 range; '
                         'rescale the physical units or use less extreme material values.') from error
    arrays = (eps, sigma, gain, drive, shift, inverse, rates)
    if (not all(np.isfinite(v).all() for v in arrays) or not np.all(eps > 0)
            or any(np.any(v < 0) for v in (sigma, drive, shift, inverse, rates))
            or np.any(np.abs(gain) > 1)):
        raise ValueError('conductive-film derived coefficients are not finite and passive; '
                         'rescale the physical units or use less extreme material values.')
    return tuple(_compact(v, film.axis) for v in arrays)


def material_profiles(scene, plan, rt):
    """None, or compact (eps including the Debye shift, sigma) on the full grid."""
    film = descriptor(scene, plan.grid)
    if film is None:
        return None
    eps, sigma, *_ = _profiles(film, plan.grid, plan.dt)
    return rt.array(eps, 'float64'), rt.array(sigma, 'float64')


def pole_params(scene, plan, rt):
    """One tuple entry per pole; spatial arrays are compact and cropped to term support."""
    film = descriptor(scene, plan.grid)
    if film is None:
        return {}
    _, _, gain, drive, shift, inverse, _ = _profiles(film, plan.grid, plan.dt)
    cut = [slice(None)]*4
    cut[3-film.axis] = slice(*film.box[film.axis])
    cut = tuple(cut)
    array = lambda value: rt.array(value[cut], 'float64')
    return {'conductive[0]': {
        'step_gain': (array(gain),), 'hold': (rt.array(0.0, 'float64'),),
        'drive': (array(drive),), 'dx_over_dt': rt.array(plan.dx/plan.dt, 'float64'),
        'energy_inverse_delta': (array(inverse),), 'energy_instantaneous': (array(shift),),
    }}


def extra_rate(scene, plan):
    """Actual max (delta/tau**2)/eps_inf on live normal nodes, for the existing CFL policy.

    This conservative Debye policy is not a necessary relaxation-time restriction:
    the trapezoidal relaxation is passive. The same policy as a declared Debye pole
    is kept here, with max f*g*b*sigma**2/(a*A**2), bounded by sigma**2/(4*a**2).
    """
    film = descriptor(scene, plan.grid)
    if film is None:
        return 0.0
    *_, rate = _profiles(film, plan.grid, plan.dt)
    return float(np.max(rate))
