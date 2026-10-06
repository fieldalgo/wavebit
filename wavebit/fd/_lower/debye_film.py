"""One concrete flat film with one passive Debye pole in a plain dielectric.

For film fraction f, g=1-f, film eps=a+delta/(1+tau*s), host b, set
A=f*b+g*a and B=A+g*delta. The arithmetic tangential response has
eps_inf=f*a+g*b, delta_eff=f*delta, tau_eff=tau. The harmonic normal
response has eps_inf=a*b/A, delta_eff=f*b*b*delta/(A*B), tau_eff=A*tau/B.
These direct rational reductions use the diagonal interface rule in Deinega
and Valuev, Optics Letters 32, 3429 (2007), Eqs. 1-2:
https://doi.org/10.1364/OL.32.003429 . No static-minus-infinite subtraction,
pole fitting, conductivity approximation or tensor coupling is needed.

With u=2*tau_eff/(2*tau_eff+dt), v=dt/(2*tau_eff+dt), the existing split
memory P=Ptilde+c*E has gain=u-v, hold=0, c=delta_eff*v and drive=2*delta_eff*u*v.
The update permittivity includes c. Energy adds
|previous+c*E|**2/delta_eff-c*|E|**2 through the existing spatial Debye branch.

The repository's conservative Debye policy is delta/(eps_inf*tau**2), not a
necessary relaxation-time restriction. Normal mixing can increase this rate:
R_normal/R_bulk=q*(1+(delta/a)*(1-q)), q=f*b/A. We retain the larger of
the original bulk rate and every actual Yee-node rate, without double counting.
This policy does not extend the existing PML stability contract.
"""
import dataclasses
import math

import numpy as np

from ..boundaries import Absorbing, FACES
from ..geometry import Box
from ..materials import Debye, Dispersive, Material, as_material, concrete
from .._kernel.masks import Region, Slab
from .._kernel.poles import PoleTerm
from .conductive_film import _compact, _fill, _isotropic

__all__ = ['Film', 'DebyeFilmTerm', 'descriptor', 'terms', 'material_profiles',
           'pole_params', 'refuse_mismatch', 'policy_rate']


@dataclasses.dataclass(frozen=True)
class Film:
    """Concrete identity of the supported one-Debye film."""

    axis: int
    low: float
    high: float
    eps: float
    host_eps: float
    delta: float
    tau: float
    box: tuple
    signature: tuple


@dataclasses.dataclass(frozen=True)
class DebyeFilmTerm(PoleTerm):
    """One spatial split memory, reserved even when delta_eps is zero."""

    signature: tuple


def descriptor(scene, grid):
    """The bounded supported scene, or None for the existing refusal paths."""
    structures = tuple(getattr(scene, 'structures', ()) or ())
    if len(structures) != 1 or getattr(scene, 'initial', None) is not None:
        return None
    structure = structures[0]
    shape, material = structure.shape, structure.material
    if (type(shape) is not Box or shape.angle != 0 or shape.range is not None
            or structure.discretisation != 'voxel' or type(material) is not Dispersive
            or len(material.poles) != 1 or type(material.poles[0]) is not Debye):
        return None
    host = as_material(getattr(scene, 'background', None))
    if type(host) is not Material:
        return None
    from .shape_ranges import materials_traced, traced
    if materials_traced(scene) or traced(scene):
        return None
    a, b = (_isotropic(m.eps, positive=True) for m in (material, host))
    mu_a, mu_b = (_isotropic(m.mu_value, positive=True) for m in (material, host))
    pole = material.poles[0]
    delta, tau = concrete(pole.delta_eps), concrete(pole.tau)
    if (None in (a, b, delta, tau) or mu_a != 1.0 or mu_b != 1.0
            or not math.isfinite(delta) or delta < 0 or not math.isfinite(tau) or tau <= 0):
        return None
    uniform = tuple(shape.uniform_along(k, (0, n), grid) for k, n in enumerate(grid.n))
    if sum(uniform) != 2:
        return None
    axis = uniform.index(False)
    if shape.size[axis] < grid.dx:
        return None
    box = tuple(tuple(int(v) for v in pair) for pair in shape.support(grid))
    lo, hi = box[axis]
    if lo <= 0 or hi >= grid.n[axis] or lo >= hi:
        return None
    for name in FACES:
        face = scene.boundaries.face(name)
        if not isinstance(face, Absorbing):
            continue
        k = 'xyz'.index(name[0])
        start, end = box[k]
        if (start < face.cells if name[1] == '-' else end > grid.n[k]-face.cells):
            return None
    low = float(shape.center[axis]-shape.size[axis]/2)
    high = float(shape.center[axis]+shape.size[axis]/2)
    signature = ('debye-film-v1', tuple(float(v) for v in shape.size),
                 tuple(float(v) for v in shape.center), float(shape.angle), shape.axis,
                 a, b, delta, tau, material.mu is not None, host.mu is not None,
                 structure.discretisation, structure.junction, tuple(grid.n),
                 float(grid.dx), tuple(float(v) for v in grid.low))
    return Film(axis, low, high, a, b, delta, tau, box, signature)


def terms(scene, grid):
    """One spatial term replacing the declared film's ordinary PoleTerm."""
    film = descriptor(scene, grid)
    if film is None:
        return ()
    region = Region(z=film.box[2], y=film.box[1], x=film.box[0])
    slabs = tuple(Slab(c=c, z=region.z, y=region.y, x=region.x) for c in range(3))
    return (DebyeFilmTerm(name='debye_film[0]', region=region, slabs=slabs,
                          n_poles=1, signature=film.signature),)


def refuse_mismatch(scene, plan):
    """Reject changed film identity and missing or additional pole memory."""
    held = tuple(t for t in plan.terms if isinstance(t, DebyeFilmTerm)
                 or getattr(t, 'name', '').startswith('debye_film['))
    expected = terms(scene, plan.grid)
    all_poles = tuple(t for t in plan.terms if hasattr(t, 'n_poles'))
    if held != expected or (expected and all_poles != expected):
        raise ValueError('the prepared Debye-film plan does not match this scene. Changed '
                         'or traced geometry/material values and missing/extra interface '
                         'memory need a new supported fd.plan(scene).')


def _range_error():
    return ValueError('Debye-film derived values exceed finite positive float64 range; '
                      'rescale the physical units or use less extreme material values.')


def _effective(film, grid):
    """Direct eps_inf, delta_eff, tau_eff and live mask at every E component."""
    n = grid.n[film.axis]
    eps = np.full((3, n), film.host_eps)
    delta = np.zeros((3, n))
    tau = np.full((3, n), film.tau)
    live = np.zeros((3, n), dtype=bool)
    a, b, strength = film.eps, film.host_eps, film.delta
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            for component in range(3):
                f = _fill(film, grid, component)
                g = 1-f
                live[component] = (f > 0) & (strength > 0)
                if component != film.axis:
                    eps[component] = f*a+g*b
                    delta[component] = f*strength
                else:
                    mixed = (f > 0) & (f < 1)
                    fm, gm = f[mixed], g[mixed]
                    A = fm*b+gm*a
                    eps[component, mixed] = a*(b/A)
                    if strength > 0:
                        B = A+gm*strength
                        delta[component, mixed] = (fm*b/A)*(b/B)*strength
                        tau[component, mixed] = (A/B)*film.tau
                # Exact geometry endpoints, never tolerance-based snapping.
                eps[component, f == 0] = b
                eps[component, f == 1] = a
                delta[component, f == 0] = 0
                delta[component, f == 1] = strength
                tau[component, f == 1] = film.tau
    except (FloatingPointError, OverflowError) as error:
        raise _range_error() from error
    if (not all(np.isfinite(v).all() for v in (eps, delta, tau)) or np.any(eps <= 0)
            or np.any(delta[live] <= 0) or np.any(tau <= 0)):
        raise _range_error()
    return eps, delta, tau, live


def _profiles(film, grid, dt):
    """Compact split-memory and energy profiles with stable small drive coefficients."""
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('Debye-film coefficients need a finite positive time step.')
    eps, delta, tau, live = _effective(film, grid)
    sigma, gain, drive, shift, inverse = (np.zeros_like(eps) for _ in range(5))
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            scale = np.maximum(tau[live], dt/2)
            left, right = tau[live]/scale, (dt/2)/scale
            u, v = left/(left+right), right/(left+right)
            gain[live] = u-v
            shift[live] = delta[live]*v
            drive[live] = (2*u)*shift[live]  # not shift*(1+gain), which can cancel to zero
            inverse[live] = 1/delta[live]
            eps += shift
    except (FloatingPointError, OverflowError) as error:
        raise _range_error() from error
    arrays = (eps, sigma, gain, drive, shift, inverse)
    if (not all(np.isfinite(v).all() for v in arrays) or np.any(eps <= 0)
            or np.any(drive[live] <= 0) or np.any(shift[live] <= 0)
            or np.any(inverse[live] <= 0) or np.any(np.abs(gain) > 1)):
        raise _range_error()
    return tuple(_compact(v, film.axis) for v in arrays)


def material_profiles(scene, plan, rt):
    """None, or eps_inf plus the instantaneous split shift and zero conductivity."""
    film = descriptor(scene, plan.grid)
    if film is None:
        return None
    eps, sigma, *_ = _profiles(film, plan.grid, plan.dt)
    return rt.array(eps, 'float64'), rt.array(sigma, 'float64')


def pole_params(scene, plan, rt):
    """Spatial split-memory arrays, cropped to support, with existing Debye energy keys."""
    film = descriptor(scene, plan.grid)
    if film is None:
        return {}
    _, _, gain, drive, shift, inverse = _profiles(film, plan.grid, plan.dt)
    cut = [slice(None)]*4
    cut[3-film.axis] = slice(*film.box[film.axis])
    cut = tuple(cut)
    array = lambda value: rt.array(value[cut], 'float64')
    return {'debye_film[0]': {
        'step_gain': (array(gain),), 'hold': (rt.array(0.0, 'float64'),),
        'drive': (array(drive),), 'dx_over_dt': rt.array(plan.dx/plan.dt, 'float64'),
        'energy_inverse_delta': (array(inverse),), 'energy_instantaneous': (array(shift),),
    }}


def policy_rate(scene, plan):
    """None outside this scope; max of bulk and actual node Debye policy rates inside.

The caller must skip ordinary Debye.rate() for this classified scene. In
particular, delta=0 stays inactive for every finite positive tau, including
values whose square cannot be represented. No original rate is added twice.
"""
    film = descriptor(scene, plan.grid)
    if film is None:
        return None
    if film.delta == 0:
        return 0.0
    eps, delta, tau, live = _effective(film, plan.grid)
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            # Square last, so a representable rate need not form tau**2 or delta/eps.
            local = np.square(np.sqrt(delta[live])/np.sqrt(eps[live])/tau[live])
            bulk = (math.sqrt(film.delta)/math.sqrt(film.eps)/film.tau)**2
    except (FloatingPointError, OverflowError, ZeroDivisionError) as error:
        raise _range_error() from error
    if (not np.isfinite(local).all() or np.any(local <= 0)
            or not math.isfinite(bulk) or bulk <= 0):
        raise _range_error()
    return max(bulk, float(np.max(local)))
