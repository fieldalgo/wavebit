"""A finite passive Drude/Lorentz sum at one concrete flat-film interface.

For f film, g=1-f host, film high-frequency permittivity a and host b, set
A=f*b+g*a, alpha=g/A and kappa=f*b*b/A**2. Writing S_j>=0, d_j=omega0_j**2,
the normal harmonic response is eps_n=a*b/A+kappa*chi/(1+alpha*chi), where
chi=sum S_j/(s*s+gamma_j*s+d_j). Keep physical polarizations p_j instead of
diagonalizing: p_j''+gamma_j*p_j'+d_j*p_j+alpha*S_j*sum(p)=kappa*S_j*E.
The tangential arithmetic response sets alpha=0, kappa=f and eps_t=f*a+g*b.
These are direct reductions of the diagonal interface rules in Deinega and
Valuev, Optics Letters 32, 3429 (2007), Eqs. 1-2:
https://doi.org/10.1364/OL.32.003429 . No fitting or equal-damping assumption.

Only the new term's advance changes: one sum couples N existing physical
polarizations, retaining the three arrays per declared slot and sum(delta)
Ampere update. The observer needs the ordinary unshifted d_j/(kappa*S_j)
potential plus exactly one (alpha/kappa)*Re(Ptotal_new*conj(Ptotal_old)).
Inactive slots must be masked before that sum. Damping is diagonal positive
semidefinite in the weighted coordinates, even with unequal gamma_j.

For u_j=sqrt(S_j), J=diag(d)+alpha*u*u.T and v=sqrt(kappa)*u,
J+v*v.T/eps_n=diag(d)+u*u.T/a exactly. The existing global curl bound plus
max(d)+sum(S)/a therefore suffices for positive leapfrog field/velocity
energy. Drude static null modes give semidefinite stored potential energy;
this is not a positive-definiteness claim for every polarization state.

All declared slots remain in their declared order, including inactive ones.
Geometry and materials are concrete; this is neither general-body smoothing
nor a differentiation path. Scalar lowering uses exact rational arithmetic
before finite float conversion, refusing unrepresentable active weights.
"""
import dataclasses
from fractions import Fraction
import math

import numpy as np

from wavebit.fdtd._legacy_boundaries import Absorbing
from wavebit.fdtd._legacy_boundaries import FACES
from wavebit.fdtd._legacy_geometry import Box
from wavebit.materials.time_domain import Dispersive
from wavebit.materials.time_domain import Drude
from wavebit.materials.time_domain import Lorentz
from wavebit.materials.time_domain import Material
from wavebit.materials.time_domain import as_material
from wavebit.materials.time_domain import concrete
from wavebit.fdtd._kernel.masks import Region
from wavebit.fdtd._kernel.masks import Slab
from wavebit.fdtd._kernel.poles import PoleTerm
from wavebit.fdtd._prepare.conductive_film import _compact
from wavebit.fdtd._prepare.conductive_film import _fill
from wavebit.fdtd._prepare.conductive_film import _isotropic

__all__ = ['Film', 'OscillatorSumFilmTerm', 'descriptor', 'terms',
           'material_profiles', 'pole_params', 'refuse_mismatch', 'policy_rate']

_SLOT_KEYS = ('step_gain', 'hold', 'drive', 'coupling', 'energy_inverse_strength',
              'energy_resonance_over_strength', 'energy_active')


@dataclasses.dataclass(frozen=True)
class Film:
    """Concrete >=2-declared-pole identity; all tuples retain declared order."""

    axis: int
    low: float
    high: float
    eps: float
    host_eps: float
    omega0_squared: tuple
    gammas: tuple
    strengths: tuple
    box: tuple
    signature: tuple


@dataclasses.dataclass(frozen=True)
class OscillatorSumFilmTerm(PoleTerm):
    """Physical polarizations with a rank-one, cell-local restoring coupling."""

    signature: tuple

    def advance(self, mem, E, H, n, params, rt):
        """Centered ADE with each pole's own damping; one old-state sum, O(N)."""
        coefficients = params['poles'][self.name]
        drive_field = E[(slice(None),) + self.region.slices]
        total = None
        for j in range(self.n_poles):
            value = coefficients['energy_active'][j] * mem['P'][j]
            total = value if total is None else total + value
        P, previous, delta = [], [], []
        for j in range(self.n_poles):
            nxt = (coefficients['step_gain'][j] * mem['P'][j]
                   - coefficients['hold'][j] * mem['previous'][j]
                   + coefficients['drive'][j] * drive_field
                   - coefficients['coupling'][j] * total)
            P.append(nxt)
            previous.append(mem['P'][j])
            delta.append(nxt - mem['P'][j])
        return {'P': tuple(P), 'previous': tuple(previous), 'delta': tuple(delta)}


def _range_error():
    return ValueError('oscillator-sum-film derived values exceed finite float64 '
                      'range; rescale the physical units or use less extreme values.')


def _number(value, *, positive=False):
    """Round a rational once; critical active positives may not vanish or overflow."""
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise _range_error() from error
    if not math.isfinite(result) or (positive and result <= 0):
        raise _range_error()
    return result


def descriptor(scene, grid):
    """The bounded supported scene, or None for the existing refusal paths."""
    structures = tuple(getattr(scene, 'structures', ()) or ())
    if len(structures) != 1 or getattr(scene, 'initial', None) is not None:
        return None
    structure = structures[0]
    shape, material = structure.shape, structure.material
    if (type(shape) is not Box or shape.angle != 0 or shape.range is not None
            or structure.discretisation != 'voxel' or type(material) is not Dispersive
            or len(material.poles) < 2
            or any(type(p) not in (Drude, Lorentz) for p in material.poles)):
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
    values = tuple(tuple(concrete(getattr(p, field.name)) for field in dataclasses.fields(p))
                   for p in material.poles)
    if (None in (a, b) or mu_a != 1.0 or mu_b != 1.0
            or any(v is None or not math.isfinite(v) or v < 0 for row in values for v in row)):
        return None
    strengths, gammas, resonances = [], [], []
    for pole in material.poles:
        gamma = concrete(pole.gamma)
        if type(pole) is Drude:
            frequency = Fraction(concrete(pole.omega_p))
            strength = _number(frequency*frequency, positive=True)
            resonance = 0.0
        else:
            strength = concrete(pole.strength)
            # A zero-strength declaration has no oscillator, whatever its finite
            # frequency/damping. Preserve the raw values in the signature but do
            # not square an unused extreme frequency or derive unused weights.
            frequency = Fraction(concrete(pole.omega_0))
            resonance = _number(frequency*frequency, positive=True) if strength else 0.0
        strengths.append(strength)
        gammas.append(gamma)
        resonances.append(resonance)
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
    signature = ('oscillator-sum-film-v1', tuple(float(v) for v in shape.size),
                 tuple(float(v) for v in shape.center), float(shape.angle), shape.axis,
                 a, b, tuple(type(p).__name__ for p in material.poles), values,
                 material.mu is not None, host.mu is not None, structure.discretisation,
                 structure.junction, tuple(grid.n), float(grid.dx),
                 tuple(float(v) for v in grid.low))
    return Film(axis, low, high, a, b, tuple(resonances), tuple(gammas),
                tuple(strengths), box, signature)


def terms(scene, grid):
    """One rank-one term with exactly the declared number of physical memories."""
    film = descriptor(scene, grid)
    if film is None:
        return ()
    region = Region(z=film.box[2], y=film.box[1], x=film.box[0])
    slabs = tuple(Slab(c=c, z=region.z, y=region.y, x=region.x) for c in range(3))
    return (OscillatorSumFilmTerm(name='oscillator_sum_film[0]', region=region, slabs=slabs,
                                 n_poles=len(film.strengths), signature=film.signature),)


def refuse_mismatch(scene, plan):
    """Reject changed identities and missing or additional pole-bearing terms."""
    held = tuple(t for t in plan.terms if isinstance(t, OscillatorSumFilmTerm)
                 or getattr(t, 'name', '').startswith('oscillator_sum_film['))
    expected = terms(scene, plan.grid)
    all_poles = tuple(t for t in plan.terms if hasattr(t, 'n_poles'))
    if held != expected or (expected and all_poles != expected):
        raise ValueError('the prepared oscillator-sum-film plan does not match this scene. '
                         'Changed or traced geometry/material values and missing/extra '
                         'interface memory need a new supported fd.plan(scene).')


def _node_profiles(film, fraction, normal, dt):
    """Exact scalar arithmetic until each final, certified-finite coefficient rounds."""
    f, a, b, h = map(Fraction, (fraction, film.eps, film.host_eps, dt))
    g = 1-f
    if f == 0:
        return film.host_eps, 0.0, ((0.0,)*len(_SLOT_KEYS),)*len(film.strengths)
    if f == 1:
        eps, alpha, kappa = a, Fraction(0), Fraction(1)
    elif normal:
        A = f*b+g*a
        eps, alpha, kappa = a*b/A, g/A, f*b*b/(A*A)
    else:
        eps, alpha, kappa = f*a+g*b, Fraction(0), f
    rank_one = (_number(alpha/kappa, positive=True)
                if alpha and any(film.strengths) else 0.0)
    slots = []
    for resonance, gamma, strength in zip(film.omega0_squared, film.gammas, film.strengths):
        if strength == 0:
            slots.append((0.0,)*len(_SLOT_KEYS))
            continue
        d, damping, S = map(Fraction, (resonance, gamma, strength))
        half = damping*h/2
        denominator = 1+half
        beta = kappa*S
        gain = _number((2-d*h*h)/denominator)
        hold = _number((1-half)/denominator)
        drive = _number(beta*h*h/denominator, positive=True)
        coupling = _number(alpha*S*h*h/denominator, positive=bool(alpha))
        inverse = _number(1/beta, positive=True)
        potential = _number(d/beta, positive=bool(d))
        slots.append((gain, hold, drive, coupling, inverse, potential, 1.0))
    return _number(eps, positive=True), rank_one, tuple(slots)


def _profiles(film, grid, dt):
    """Finite compact spatial arrays; reuse scalar work at repeated fill values."""
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('oscillator-sum-film coefficients need a finite positive time step.')
    n, count = grid.n[film.axis], len(film.strengths)
    eps = np.zeros((3, n))
    rank_one = np.zeros((3, n))
    slots = {key: np.zeros((count, 3, n)) for key in _SLOT_KEYS}
    cached = {}
    for component in range(3):
        fractions = _fill(film, grid, component)
        for index, fraction in enumerate(fractions):
            key = (float(fraction), component == film.axis)
            if key not in cached:
                cached[key] = _node_profiles(film, *key, dt)
            epsilon, rho, coefficients = cached[key]
            eps[component, index], rank_one[component, index] = epsilon, rho
            for j, row in enumerate(coefficients):
                for name, value in zip(_SLOT_KEYS, row):
                    slots[name][j, component, index] = value
    result = {name: tuple(_compact(value[j], film.axis) for j in range(count))
              for name, value in slots.items()}
    result.update(eps=_compact(eps, film.axis), sigma=_compact(np.zeros_like(eps), film.axis),
                  energy_rank_one=_compact(rank_one, film.axis))
    return result


def material_profiles(scene, plan, rt):
    """None, or compact high-frequency permittivity and zero conductivity."""
    film = descriptor(scene, plan.grid)
    if film is None:
        return None
    profiles = _profiles(film, plan.grid, plan.dt)
    return tuple(rt.array(profiles[key], 'float64') for key in ('eps', 'sigma'))


def pole_params(scene, plan, rt):
    """Spatial recurrence/energy arrays cropped to the single term's support."""
    film = descriptor(scene, plan.grid)
    if film is None:
        return {}
    profiles = _profiles(film, plan.grid, plan.dt)
    cut = [slice(None)]*4
    cut[3-film.axis] = slice(*film.box[film.axis])
    cut = tuple(cut)
    array = lambda value: rt.array(value[cut], 'float64')
    coefficients = {key: tuple(array(value) for value in profiles[key]) for key in _SLOT_KEYS}
    coefficients.update(energy_rank_one=array(profiles['energy_rank_one']),
                        dx_over_dt=rt.array(_number(Fraction(plan.dx)/Fraction(plan.dt),
                                                   positive=True), 'float64'))
    return {'oscillator_sum_film[0]': coefficients}


def policy_rate(scene, plan):
    """The existing oscillator policy, with active-only robust scoped arithmetic.

    No interface penalty: max_active(d)+sum_active(S)/a bounds the coupled
    normal material block as well as the tangent and bulk blocks. All inactive
    declarations give exactly zero without squaring their unused frequencies.
    """
    film = descriptor(scene, plan.grid)
    if film is None:
        return None
    live = [(d, S) for d, S in zip(film.omega0_squared, film.strengths) if S > 0]
    if not live:
        return 0.0
    rate = Fraction(max(d for d, _ in live)) + sum(Fraction(S) for _, S in live)/Fraction(film.eps)
    result = _number(rate, positive=True)
    if Fraction(result) < rate:
        result = math.nextafter(result, math.inf)
    if not math.isfinite(result):
        raise _range_error()
    return result
