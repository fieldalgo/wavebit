"""Finite passive Debye sums at a single concrete flat-film interface.

Only declarations with at least three poles enter this path; the completed
one/two-pole paths retain their original arithmetic and admission rules.
For film a+chi(s), chi=sum(w_i/(s+r_i)), r_i=1/tau_i, w_i=delta_i*r_i,
host b, fraction f, g=1-f, A=f*b+g*a, alpha=g/A and kappa=f*b*b/A**2,
the normal response is a*b/A+kappa*chi/(1+alpha*chi). Equivalently it is
a*b/A+kappa*u.T*inv(s*I+diag(r)+alpha*u*u.T)*u, with u_i=sqrt(w_i).
This is the harmonic normal/arithmetic tangential interface rule in
Deinega and Valuev, Optics Letters 32, 3429 (2007), Eqs. 1-2:
https://doi.org/10.1364/OL.32.003429 .

Exact-zero strengths and exactly equal input taus are removed/coalesced.
For distinct positive rates, the positive normal rates strictly interlace
them. Exact Fraction arithmetic brackets each zero of
F(lambda)=1+alpha*sum(w_i/(r_i-lambda)). The normal modal increment is
kappa/(lambda*alpha**2*sum(w_i/(r_i-lambda)**2)). Endpoint distances bound
this whole expression and tau=1/lambda. A mode is accepted only when both
ends of each enclosure round to the same positive finite float64 value.
No eigensolver residue cutoff, approximate merging or pole fitting is used.

The 4096-bisection limit per mode is a computational safeguard, not an error
tolerance. Uncertified or unrepresentable inputs receive a named refusal.
This does not guarantee admission of every finite input. Immutable normal
mode results are cached for the repeated lowering of concrete face fills;
interior/tangential cells never require a secular solve. Memory still reserves
the declared pole count, with active slots in ascending tau and zero padding.
"""
import dataclasses
from fractions import Fraction
import functools
import math

import numpy as np

from wavebit.fdtd._legacy_boundaries import Absorbing
from wavebit.fdtd._legacy_boundaries import FACES
from wavebit.fdtd._legacy_geometry import Box
from wavebit.materials._time_domain import Debye
from wavebit.materials._time_domain import Dispersive
from wavebit.materials._time_domain import Material
from wavebit.materials._time_domain import as_material
from wavebit.materials._time_domain import concrete
from wavebit.fdtd._kernel.masks import Region
from wavebit.fdtd._kernel.masks import Slab
from wavebit.fdtd._kernel.poles import PoleTerm
from wavebit.fdtd._prepare.conductive_film import _compact
from wavebit.fdtd._prepare.conductive_film import _fill
from wavebit.fdtd._prepare.conductive_film import _isotropic

__all__ = ['Film', 'DebyeSumFilmTerm', 'descriptor', 'terms', 'material_profiles',
           'pole_params', 'refuse_mismatch', 'policy_rate']

_MAX_BISECTIONS = 4096


@dataclasses.dataclass(frozen=True)
class Film:
    """Concrete identity, retaining original declared parameter order."""

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
class DebyeSumFilmTerm(PoleTerm):
    """One spatial pole set with the fixed declared number of memory slots."""

    signature: tuple


def descriptor(scene, grid):
    """Concrete singleton flat film with >=3 exact Debye declarations, or None."""
    structures = tuple(getattr(scene, 'structures', ()) or ())
    if len(structures) != 1 or getattr(scene, 'initial', None) is not None:
        return None
    structure = structures[0]
    shape, material = structure.shape, structure.material
    if (type(shape) is not Box or shape.angle != 0 or shape.range is not None
            or structure.discretisation != 'voxel' or type(material) is not Dispersive
            or len(material.poles) < 3 or any(type(p) is not Debye for p in material.poles)):
        return None
    host = as_material(getattr(scene, 'background', None))
    if type(host) is not Material:
        return None
    from wavebit.fdtd._prepare.shape_ranges import materials_traced
    from wavebit.fdtd._prepare.shape_ranges import traced
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
    signature = ('debye-sum-film-v1', tuple(float(v) for v in shape.size),
                 tuple(float(v) for v in shape.center), float(shape.angle), shape.axis,
                 a, b, deltas, taus, material.mu is not None, host.mu is not None,
                 structure.discretisation, structure.junction, tuple(grid.n),
                 float(grid.dx), tuple(float(v) for v in grid.low))
    return Film(axis, low, high, a, b, deltas, taus, box, signature)


def terms(scene, grid):
    """Reserve all declared slots, including inactive and coalesced slots."""
    film = descriptor(scene, grid)
    if film is None:
        return ()
    region = Region(z=film.box[2], y=film.box[1], x=film.box[0])
    slabs = tuple(Slab(c=c, z=region.z, y=region.y, x=region.x) for c in range(3))
    return (DebyeSumFilmTerm(name='debye_sum_film[0]', region=region, slabs=slabs,
                            n_poles=len(film.deltas), signature=film.signature),)


def refuse_mismatch(scene, plan):
    """Unconditional identity/memory check, including prepared plans of other types."""
    held = tuple(t for t in plan.terms if isinstance(t, DebyeSumFilmTerm)
                 or getattr(t, 'name', '').startswith('debye_sum_film['))
    expected = terms(scene, plan.grid)
    all_poles = tuple(t for t in plan.terms if hasattr(t, 'n_poles'))
    if held != expected or (expected and all_poles != expected):
        raise ValueError('the prepared Debye-sum-film plan does not match this scene. '
                         'Changed or traced geometry/material values and missing/extra '
                         'interface memory need a new supported fd.plan(scene).')


def _range_error():
    return ValueError('Debye-sum-film derived values exceed finite positive float64 '
                      'range; rescale the physical units or use less extreme material values.')


def _positive(value):
    """Round an exact positive quantity once, refusing zero or nonfinite output."""
    try:
        result = float(value)
    except OverflowError as error:
        raise _range_error() from error
    if not math.isfinite(result) or result <= 0:
        raise _range_error()
    return result


def _rounded_interval(low, high):
    """A common endpoint rounding certifies the entire positive enclosure."""
    values = []
    for value in (low, high):
        try:
            values.append(float(value))
        except OverflowError:
            values.append(math.inf)
    if values[0] != values[1]:
        return None
    return _positive(values[0])


def _active_poles(film):
    """Exact coalescence, independent of declaration order and unused tau size."""
    merged = {}
    for delta, tau in zip(film.deltas, film.taus):
        if delta > 0:
            time = Fraction.from_float(tau)
            merged[time] = merged.get(time, Fraction(0))+Fraction.from_float(delta)
    return tuple((delta, tau) for tau, delta in sorted(merged.items()))


def _secular_mode(rates, weights, alpha, kappa, low, high):
    """Certified tau and increment for one strict interlacing interval."""
    for _ in range(_MAX_BISECTIONS):
        middle = (low+high)/2
        secular = 1+alpha*sum(w/(r-middle) for r, w in zip(rates, weights))
        if secular == 0:
            squared = sum(w/(r-middle)**2 for r, w in zip(rates, weights))
            return _positive(kappa/(middle*alpha**2*squared)), _positive(1/middle)
        if secular < 0:
            low = middle
        else:
            high = middle
        distances = [(min(abs(r-low), abs(r-high)), max(abs(r-low), abs(r-high)))
                     for r in rates]
        if any(small == 0 for small, _ in distances):
            continue                 # still touches an original pole, never divide by zero
        squared_low = sum(w/large**2 for w, (_, large) in zip(weights, distances))
        squared_high = sum(w/small**2 for w, (small, _) in zip(weights, distances))
        delta = _rounded_interval(kappa/(high*alpha**2*squared_high),
                                  kappa/(low*alpha**2*squared_low))
        tau = _rounded_interval(1/high, 1/low)
        if delta is not None and tau is not None:
            return delta, tau
    raise ValueError('Debye-sum-film secular enclosure did not certify positive float64 '
                     f'modes within {_MAX_BISECTIONS} bisections per mode. This is a '
                     'computational safeguard; no pole was clipped or silently discarded.')


@functools.lru_cache(maxsize=128)
def _normal_modes(a, b, fraction, active):
    """Immutable cached face reduction; no cell arrays or traced values are cached."""
    a, b, f = (Fraction.from_float(value) for value in (a, b, fraction))
    g, A = 1-f, f*b+(1-f)*a
    if len(active) == 1:
        delta, tau = active[0]
        B = A+g*delta
        return ((_positive(f*b*b*delta/(A*B)), _positive(A*tau/B)),)
    alpha, kappa = g/A, f*b*b/(A*A)
    # Rates ascend, opposite to the canonical input relaxation times.
    rates = tuple(1/tau for _, tau in reversed(active))
    weights = tuple(delta/tau for delta, tau in reversed(active))
    endpoints = rates+(rates[-1]+alpha*sum(weights),)
    modes = tuple(_secular_mode(rates, weights, alpha, kappa, left, right)
                  for left, right in zip(endpoints, endpoints[1:]))
    return modes[::-1]                 # ascending effective tau, then inactive padding


def _effective(film, grid):
    """eps_inf(3,n), delta/tau/live(N,3,n), with exact geometric endpoints."""
    n, count = grid.n[film.axis], len(film.deltas)
    eps = np.full((3, n), film.host_eps)
    delta = np.zeros((count, 3, n))
    tau = np.ones((count, 3, n))
    live = np.zeros((count, 3, n), dtype=bool)
    active = _active_poles(film)
    bulk = tuple((_positive(d), _positive(t)) for d, t in active)
    a, b = (Fraction.from_float(value) for value in (film.eps, film.host_eps))
    for component in range(3):
        for cell, value in enumerate(_fill(film, grid, component)):
            if value == 0:
                continue
            f = Fraction.from_float(float(value))
            if value == 1:
                eps[component, cell], modes = film.eps, bulk
            elif component != film.axis:
                eps[component, cell] = _positive(f*a+(1-f)*b)
                modes = tuple((_positive(f*d), _positive(t)) for d, t in active)
            else:
                A = f*b+(1-f)*a
                eps[component, cell] = _positive(a*b/A)
                modes = _normal_modes(film.eps, film.host_eps, float(value), active) if active else ()
            for slot, (strength, relaxation) in enumerate(modes):
                delta[slot, component, cell] = strength
                tau[slot, component, cell] = relaxation
                live[slot, component, cell] = True
    return eps, delta, tau, live


def _profiles(film, grid, dt):
    """Stable split coefficients and existing per-slot spatial Debye energy weights."""
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('Debye-sum-film coefficients need a finite positive time step.')
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
    if (not all(np.isfinite(value).all() for value in arrays) or np.any(eps <= 0)
            or np.any(drive[live] <= 0) or np.any(shift[live] <= 0)
            or np.any(inverse[live] <= 0) or np.any(np.abs(gain) > 1)):
        raise _range_error()
    profiles = tuple(tuple(_compact(value[j], film.axis) for j in range(len(film.deltas)))
                     for value in (gain, drive, shift, inverse))
    return _compact(eps, film.axis), _compact(sigma, film.axis), *profiles


def material_profiles(scene, plan, rt):
    """None, or eps_inf plus all instantaneous split shifts and zero conductivity."""
    film = descriptor(scene, plan.grid)
    if film is None:
        return None
    eps, sigma, *_ = _profiles(film, plan.grid, plan.dt)
    return rt.array(eps, 'float64'), rt.array(sigma, 'float64')


def pole_params(scene, plan, rt):
    """All declared slots, cropped to support, using unchanged Debye energy keys."""
    film = descriptor(scene, plan.grid)
    if film is None:
        return {}
    _, _, gain, drive, shift, inverse = _profiles(film, plan.grid, plan.dt)
    cut = [slice(None)]*4
    cut[3-film.axis] = slice(*film.box[film.axis])
    cut = tuple(cut)
    arrays = lambda values: tuple(rt.array(value[cut], 'float64') for value in values)
    return {'debye_sum_film[0]': {
        'step_gain': arrays(gain), 'hold': (rt.array(0.0, 'float64'),)*len(film.deltas),
        'drive': arrays(drive), 'dx_over_dt': rt.array(plan.dx/plan.dt, 'float64'),
        'energy_inverse_delta': arrays(inverse), 'energy_instantaneous': arrays(shift),
    }}


def policy_rate(scene, plan):
    """Maximum of bulk summed rate and every actual effective-node summed rate."""
    film = descriptor(scene, plan.grid)
    if film is None:
        return None
    if not any(delta > 0 for delta in film.deltas):
        return 0.0
    a = Fraction.from_float(film.eps)
    bulk = _positive(sum(Fraction.from_float(d)/(a*Fraction.from_float(t)**2)
                         for d, t in zip(film.deltas, film.taus) if d > 0))
    eps, delta, tau, live = _effective(film, plan.grid)
    maximum = bulk
    for component in range(3):
        for cell in range(eps.shape[1]):
            if not live[:, component, cell].any():
                continue
            local = sum(Fraction.from_float(delta[j, component, cell])
                        /(Fraction.from_float(eps[component, cell])
                          *Fraction.from_float(tau[j, component, cell])**2)
                        for j in range(len(film.deltas)) if live[j, component, cell])
            maximum = max(maximum, _positive(local))
    return maximum
