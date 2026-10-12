"""Two passive Debye poles at one concrete, flat dielectric-film interface.

For film a+sum(w_i/(s+r_i)), r_i=1/tau_i, w_i=delta_i*r_i, host b,
film fraction f and g=1-f, let A=f*b+g*a, alpha=g/A, and u_i=sqrt(w_i).
The harmonic normal response is exactly
    a*b/A + (f*b*b/A**2) * u.T @ inv(s*I+J) @ u,
    J=diag(r_i)+alpha*u*u.T.
J is positive definite. Its two eigenvalues and nonnegative residues give
two passive Debye memories, without fitted poles or conductivity terms.
Tangential response is arithmetic. These are the diagonal interface rules
in Deinega and Valuev, Optics Letters 32, 3429 (2007), Eqs. 1-2:
https://doi.org/10.1364/OL.32.003429 .

The eigenproblem is evaluated in the basis (u/|u|, its perpendicular).
Its small spectral weight is computed from the off-diagonal square instead
of subtracting nearly equal numbers. Positive products use binary scaling
so a tiny intermediate weight need not erase a representable final pole.
Exact equal-tau and zero-strength cases collapse before the eigenproblem;
two memory slots remain reserved, ordered by increasing effective tau.

Each pole uses the existing split Debye memory and spatial energy weights.
The conservative timestep policy uses the sum of both rates at each actual
Yee node, then the maximum of that sum and the original bulk summed rate.
No individual-pole maxima are added and no PML stability claim is changed.
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

__all__ = ['Film', 'TwoDebyeFilmTerm', 'descriptor', 'terms', 'material_profiles',
           'pole_params', 'refuse_mismatch', 'policy_rate']


@dataclasses.dataclass(frozen=True)
class Film:
    """Concrete identity of the supported, exactly two-declared-pole film."""

    axis: int
    low: float
    high: float
    eps: float
    host_eps: float
    deltas: tuple
    taus: tuple
    box: tuple
    signature: tuple


@dataclasses.dataclass(frozen=True)
class TwoDebyeFilmTerm(PoleTerm):
    """Two spatial split memories, including exact inactive/collapsed slots."""

    signature: tuple


def descriptor(scene, grid):
    """The supported concrete scene, or None for the existing refusal paths."""
    structures = tuple(getattr(scene, 'structures', ()) or ())
    if len(structures) != 1 or getattr(scene, 'initial', None) is not None:
        return None
    structure = structures[0]
    shape, material = structure.shape, structure.material
    if (type(shape) is not Box or shape.angle != 0 or shape.range is not None
            or structure.discretisation != 'voxel' or type(material) is not Dispersive
            or len(material.poles) != 2 or any(type(p) is not Debye for p in material.poles)):
        return None
    host = as_material(getattr(scene, 'background', None))
    if type(host) is not Material:
        return None
    from .shape_ranges import materials_traced, traced
    if materials_traced(scene) or traced(scene):
        return None
    a, b = (_isotropic(m.eps, positive=True) for m in (material, host))
    mu_a, mu_b = (_isotropic(m.mu_value, positive=True) for m in (material, host))
    deltas = tuple(concrete(p.delta_eps) for p in material.poles)
    taus = tuple(concrete(p.tau) for p in material.poles)
    if (None in (a, b, *deltas, *taus) or mu_a != 1.0 or mu_b != 1.0
            or any(not math.isfinite(d) or d < 0 for d in deltas)
            or any(not math.isfinite(t) or t <= 0 for t in taus)):
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
    signature = ('two-debye-film-v1', tuple(float(v) for v in shape.size),
                 tuple(float(v) for v in shape.center), float(shape.angle), shape.axis,
                 a, b, deltas, taus, material.mu is not None, host.mu is not None,
                 structure.discretisation, structure.junction, tuple(grid.n),
                 float(grid.dx), tuple(float(v) for v in grid.low))
    return Film(axis, low, high, a, b, deltas, taus, box, signature)


def terms(scene, grid):
    """One two-slot term replacing the film's ordinary declared-pole term."""
    film = descriptor(scene, grid)
    if film is None:
        return ()
    region = Region(z=film.box[2], y=film.box[1], x=film.box[0])
    slabs = tuple(Slab(c=c, z=region.z, y=region.y, x=region.x) for c in range(3))
    return (TwoDebyeFilmTerm(name='two_debye_film[0]', region=region, slabs=slabs,
                            n_poles=2, signature=film.signature),)


def refuse_mismatch(scene, plan):
    """Reject changed identities and missing/additional pole-bearing memory."""
    held = tuple(t for t in plan.terms if isinstance(t, TwoDebyeFilmTerm)
                 or getattr(t, 'name', '').startswith('two_debye_film['))
    expected = terms(scene, plan.grid)
    all_poles = tuple(t for t in plan.terms if hasattr(t, 'n_poles'))
    if held != expected or (expected and all_poles != expected):
        raise ValueError('the prepared two-Debye-film plan does not match this scene. '
                         'Changed or traced geometry/material values and missing/extra '
                         'interface memory need a new supported fd.plan(scene).')


def _range_error():
    return ValueError('Two-Debye-film derived values exceed finite positive float64 '
                      'range; rescale the physical units or use less extreme material values.')


def _positive_product(numerators, denominators=(), *, allow_zero=False):
    """Positive product/quotient without overflowing intermediate powers.

Only internal eigenweight/determinant contributions may round to zero. A
live final pole, coefficient or energy weight must remain positive finite.
"""
    mantissa, exponent = 1.0, 0
    for values, divide in ((numerators, False), (denominators, True)):
        for value in values:
            if not math.isfinite(value) or value <= 0:
                raise _range_error()
            part, power = math.frexp(value)
            mantissa = mantissa/part if divide else mantissa*part
            exponent += -power if divide else power
            mantissa, power = math.frexp(mantissa)
            exponent += power
    try:
        result = math.ldexp(mantissa, exponent)
    except OverflowError as error:
        raise _range_error() from error
    if not math.isfinite(result) or (result == 0 and not allow_zero):
        raise _range_error()
    return result


def _active_poles(film):
    """Canonical active input poles; exact equality alone merges two taus."""
    active = sorted(((d, t) for d, t in zip(film.deltas, film.taus) if d > 0),
                    key=lambda pole: pole[1])
    if len(active) == 2 and active[0][1] == active[1][1]:
        try:
            strength = math.fsum(pole[0] for pole in active)
        except OverflowError as error:
            raise _range_error() from error
        active = [(strength, active[0][1])]
    return tuple(active)


def _normal_modes(film, fraction, A, active):
    """Exact passive normal poles, with the faster effective pole first."""
    f, g, b = fraction, 1-fraction, film.host_eps
    if len(active) == 1:
        delta, tau = active[0]
        B = math.fsum((A, g*delta))
        return ((_positive_product((f, b, b, delta), (A, B)),
                 _positive_product((A, tau), (B,))),)
    # Sort by increasing rate for the symmetric 2-by-2 formulas below.
    (delta2, tau2), (delta1, tau1) = active
    r1, r2 = (_positive_product((1.0,), (t,)) for t in (tau1, tau2))
    w1, w2 = (_positive_product((d,), (t,)) for d, t in ((delta1, tau1),
                                                                  (delta2, tau2)))
    W = math.fsum((w1, w2))
    p, q = (_positive_product((w,), (W,)) for w in (w1, w2))
    beta = _positive_product((g, W), (A,))
    # Unlike r2-r1, this remains positive if reciprocal rounding ties the rates.
    difference = _positive_product((tau1-tau2,), (tau1, tau2))
    scale = max(r2, beta, difference)
    ds, bs = difference/scale, beta/scale
    h = math.fsum((bs, ds*(q-p)))
    gap = math.hypot(h, (2*ds*math.sqrt(p))*math.sqrt(q))
    if not math.isfinite(gap) or gap <= 0:
        raise _range_error()
    high = _positive_product((scale, .5*(r1/scale+r2/scale+bs+gap)))
    low = math.fsum((_positive_product((r1, r2), (high,), allow_zero=True),
                     _positive_product((beta, p, r2), (high,), allow_zero=True),
                     _positive_product((beta, q, r1), (high,), allow_zero=True)))
    if not math.isfinite(low) or low <= 0 or low > high:
        raise _range_error()
    small_numerators = (2.0, p, q, ds, ds)
    small_denominators = (gap, gap, 1+abs(h/gap))
    small_weight = _positive_product(small_numerators, small_denominators,
                                     allow_zero=True)
    # Form the small final strength directly: a subnormal/zero intermediate
    # spectral weight must not discard a representable positive modal strength.
    small_rate, large_rate = (low, high) if h >= 0 else (high, low)
    small_delta = _positive_product((f, b, b, W, *small_numerators),
                                    (A, A, small_rate, *small_denominators))
    large_delta = _positive_product((f, b, b, W, 1-small_weight),
                                    (A, A, large_rate))
    strengths = (large_delta, small_delta) if h >= 0 else (small_delta, large_delta)
    return tuple((d, _positive_product((1.0,), (rate,)))
                 for d, rate in zip(strengths, (high, low)))


def _effective(film, grid):
    """eps_inf(3,n), delta/tau/live(2,3,n), with exact geometric endpoints."""
    n = grid.n[film.axis]
    eps = np.full((3, n), film.host_eps)
    delta = np.zeros((2, 3, n))
    tau = np.ones((2, 3, n))
    live = np.zeros((2, 3, n), dtype=bool)
    active = _active_poles(film)
    a, b = film.eps, film.host_eps
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            for component in range(3):
                fractions = _fill(film, grid, component)
                for cell, f in enumerate(fractions):
                    if f == 0:
                        continue
                    if f == 1:
                        eps[component, cell], modes = a, active
                    elif component != film.axis:
                        eps[component, cell] = f*a+(1-f)*b
                        modes = tuple((_positive_product((f, d)), t) for d, t in active)
                    else:
                        A = f*b+(1-f)*a
                        eps[component, cell] = _positive_product((a, b), (A,))
                        modes = _normal_modes(film, float(f), A, active) if active else ()
                    for slot, (strength, relaxation) in enumerate(modes):
                        delta[slot, component, cell] = strength
                        tau[slot, component, cell] = relaxation
                        live[slot, component, cell] = True
    except (FloatingPointError, OverflowError, ZeroDivisionError) as error:
        raise _range_error() from error
    if (not all(np.isfinite(v).all() for v in (eps, delta, tau)) or np.any(eps <= 0)
            or np.any(delta[live] <= 0) or np.any(tau <= 0)):
        raise _range_error()
    return eps, delta, tau, live


def _profiles(film, grid, dt):
    """Stable split coefficients; two independent existing Debye energy slots."""
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('Two-Debye-film coefficients need a finite positive time step.')
    eps, delta, tau, live = _effective(film, grid)
    sigma = np.zeros_like(eps)
    gain, drive, shift, inverse = (np.zeros_like(delta) for _ in range(4))
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            scale = np.maximum(tau[live], dt/2)
            left, right = tau[live]/scale, (dt/2)/scale
            u, v = left/(left+right), right/(left+right)
            gain[live] = u-v
            shift[live] = delta[live]*v
            drive[live] = (2*u)*shift[live]
            inverse[live] = 1/delta[live]
            eps += np.sum(shift, axis=0)
    except (FloatingPointError, OverflowError) as error:
        raise _range_error() from error
    arrays = (eps, sigma, gain, drive, shift, inverse)
    if (not all(np.isfinite(v).all() for v in arrays) or np.any(eps <= 0)
            or np.any(drive[live] <= 0) or np.any(shift[live] <= 0)
            or np.any(inverse[live] <= 0) or np.any(np.abs(gain) > 1)):
        raise _range_error()
    profiles = tuple(tuple(_compact(v[j], film.axis) for j in range(2))
                     for v in (gain, drive, shift, inverse))
    return _compact(eps, film.axis), _compact(sigma, film.axis), *profiles


def material_profiles(scene, plan, rt):
    """None, or eps_inf plus both split shifts and zero conductivity."""
    film = descriptor(scene, plan.grid)
    if film is None:
        return None
    eps, sigma, *_ = _profiles(film, plan.grid, plan.dt)
    return rt.array(eps, 'float64'), rt.array(sigma, 'float64')


def pole_params(scene, plan, rt):
    """The two cropped pole arrays and existing spatial Debye energy metadata."""
    film = descriptor(scene, plan.grid)
    if film is None:
        return {}
    _, _, gain, drive, shift, inverse = _profiles(film, plan.grid, plan.dt)
    cut = [slice(None)]*4
    cut[3-film.axis] = slice(*film.box[film.axis])
    cut = tuple(cut)
    arrays = lambda values: tuple(rt.array(value[cut], 'float64') for value in values)
    return {'two_debye_film[0]': {
        'step_gain': arrays(gain), 'hold': (rt.array(0.0, 'float64'),)*2,
        'drive': arrays(drive), 'dx_over_dt': rt.array(plan.dx/plan.dt, 'float64'),
        'energy_inverse_delta': arrays(inverse), 'energy_instantaneous': arrays(shift),
    }}


def policy_rate(scene, plan):
    """Max of original bulk summed rate and the summed rate at every Yee node.

The caller skips ordinary Debye.rate() only for this recognized scene.
Inactive poles never divide by tau, even if its square is unrepresentable.
"""
    film = descriptor(scene, plan.grid)
    if film is None:
        return None
    if not any(d > 0 for d in film.deltas):
        return 0.0
    eps, delta, tau, live = _effective(film, plan.grid)
    try:
        bulk = math.fsum(_positive_product((d,), (film.eps, t, t))
                         for d, t in zip(film.deltas, film.taus) if d > 0)
        local = np.zeros_like(eps)
        for component in range(3):
            for cell in range(eps.shape[1]):
                local[component, cell] = math.fsum(
                    _positive_product((delta[j, component, cell],),
                                      (eps[component, cell], tau[j, component, cell],
                                       tau[j, component, cell]))
                    for j in range(2) if live[j, component, cell])
    except (FloatingPointError, OverflowError, ZeroDivisionError) as error:
        raise _range_error() from error
    if not math.isfinite(bulk) or bulk <= 0 or not np.isfinite(local).all():
        raise _range_error()
    return max(bulk, float(np.max(local)))
