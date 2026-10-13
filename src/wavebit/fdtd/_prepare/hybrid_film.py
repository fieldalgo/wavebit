"""One concrete film with declared Debye and Drude/Lorentz families.

Let chi be the declared susceptibility sum. Normal mixing has
eps=eta+kappa*chi/(1+alpha*chi), with A=f*b+(1-f)*a,
eta=a*b/A, alpha=(1-f)/A and kappa=f*b*b/A**2. Tangent mixing uses
eta=f*a+(1-f)*b, alpha=0 and kappa=f. These follow the diagonal interface
rules in Deinega and Valuev, Optics Letters 32, 3429 (2007), Eqs. 1-2:
https://doi.org/10.1364/OL.32.003429 . No external implementation was read.

Half-step Debye coordinates D have physical integer output m=mean(D).
Oscillator positions q use the existing central ADE. With rho=alpha/kappa,
Debye t_i=2*tau_i/(2*tau_i+h), bare_i=kappa*delta_i*h/(2*tau_i+h),
c_i=bare_i/(1+rho*sum(bare)), and T=sum(t_i*Dold_i),
m_i=t_i*Dold_i+c_i*(E-rho*(T+sum(q))). The stored Debye output is
y_i=m_i-c_i*E. Predict Dnew and qnext locally, then
ynext_i=t_i*Dnew_i-rho*c_i*(sum(t*Dnew)+sum(qnext)). Ampere uses
sum(delta_y)+sum(delta_q) with update permittivity eta+sum(c_i).

This preserves the bilinear Debye and central oscillator symbols in exact
arithmetic. It is not the all-midpoint oscillator alternative. The closed
Yee-system energy uses physical previous+y shift, centred oscillator
velocity, a rank-one potential once, and a negative force-squared term;
it needs its own observer branch before either existing energy branch.

All four memory tuples retain declared order: P, previous, delta, aux.
Debye aux is the half-step D; oscillator aux is the centred velocity at
the observer clock. This is four vector arrays per declared slot, not the
three-array PoleTerm report layout. Existing family routes stay separate.

Scalar coefficients use exact Fraction arithmetic until final conversion.
Critical active positive values may not disappear or overflow; an active
Debye gain rounded to either endpoint +/-1 is explicitly refused. These
are representability checks, not a float64 spectral/passivity certificate.
"""
import dataclasses
from fractions import Fraction
import math

import numpy as np

from wavebit.fdtd._legacy_boundaries import Absorbing
from wavebit.fdtd._legacy_boundaries import FACES
from wavebit.fdtd._legacy_geometry import Box
from wavebit.materials.time_domain import Debye
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

__all__ = ['Film', 'HybridFilmTerm', 'descriptor', 'terms', 'material_profiles',
           'pole_params', 'refuse_mismatch', 'policy_rate']

_SLOT_KEYS = (
    'step_gain', 'hold', 'drive', 'coupling',
    'debye_t', 'debye_gain', 'debye_c', 'debye_drive', 'debye_feedback',
    'energy_active', 'energy_debye_active', 'energy_oscillator_active',
    'energy_inverse_delta', 'energy_inverse_strength',
    'energy_resonance_over_strength', 'energy_kinetic', 'energy_force_weight',
)


@dataclasses.dataclass(frozen=True)
class Film:
    """Aligned declared-order tuples; unused family entries are zero placeholders."""

    axis: int
    low: float
    high: float
    eps: float
    host_eps: float
    kinds: tuple
    deltas: tuple
    taus: tuple
    omega0_squared: tuple
    gammas: tuple
    strengths: tuple
    box: tuple
    signature: tuple


@dataclasses.dataclass(frozen=True)
class HybridFilmTerm(PoleTerm):
    """Declared mixed-family term with four, rather than three, arrays per slot."""

    memory_arrays_per_pole = 4

    kinds: tuple
    signature: tuple

    def init_mem(self, plan, rt):
        """At rest; aux is Debye Dhalf or oscillator centred velocity, by kind."""
        zeros = [rt.zeros((3,)+self.region.shape, plan.dtype)
                 for _ in range(4*self.n_poles)]
        return {name: tuple(zeros[i*self.n_poles:(i+1)*self.n_poles])
                for i, name in enumerate(('P', 'previous', 'delta', 'aux'))}

    def advance(self, mem, E, H, n, params, rt):
        """O(N) half-Debye/central-oscillator predictor using the known E only.

        Input at E^n has P=(y^n,q^n), previous=(y^(n-1),q^(n-1)),
        and Debye aux=D^(n-1/2). Output predicts P at n+1, retains
        previous at n, and sets aux=(D^(n+1/2),vbar^n). The observer
        then reconstructs physical m^n=previous_D+c_D*E^n.
        """
        co = params['poles'][self.name]
        field = E[(slice(None),)+self.region.slices]
        old = tuple(co['energy_active'][j]*mem['P'][j] for j in range(self.n_poles))
        older = tuple(co['energy_active'][j]*mem['previous'][j]
                      for j in range(self.n_poles))
        history, oscillators = 0.0, 0.0
        for j, kind in enumerate(self.kinds):
            if kind == 'debye':
                auxiliary = co['energy_debye_active'][j]*mem['aux'][j]
                history = history + co['debye_t'][j]*auxiliary
            else:
                oscillators = oscillators + old[j]
        coupled_history = history + oscillators
        physical, half = [None]*self.n_poles, [None]*self.n_poles
        debye_total = 0.0
        for j, kind in enumerate(self.kinds):
            if kind != 'debye':
                continue
            auxiliary = co['energy_debye_active'][j]*mem['aux'][j]
            physical[j] = (co['debye_t'][j]*auxiliary + co['debye_c'][j]*field
                           - co['debye_feedback'][j]*coupled_history)
            half[j] = (co['debye_gain'][j]*auxiliary + co['debye_drive'][j]*field
                       - 2*co['debye_feedback'][j]*coupled_history)
            debye_total = debye_total + physical[j]
        total = debye_total + oscillators
        P, aux = [None]*self.n_poles, [None]*self.n_poles
        next_oscillators = 0.0
        for j, kind in enumerate(self.kinds):
            if kind == 'debye':
                continue
            nxt = (co['step_gain'][j]*old[j] - co['hold'][j]*older[j]
                   + co['drive'][j]*field - co['coupling'][j]*total)
            P[j] = nxt
            aux[j] = co['inverse_two_dt']*(nxt-older[j])
            next_oscillators = next_oscillators + nxt
        next_history = 0.0
        for j, kind in enumerate(self.kinds):
            if kind == 'debye':
                next_history = next_history + co['debye_t'][j]*half[j]
        next_total = next_history + next_oscillators
        for j, kind in enumerate(self.kinds):
            if kind == 'debye':
                P[j] = (co['debye_t'][j]*half[j]
                        - co['debye_feedback'][j]*next_total)
                aux[j] = half[j]
        return {'P': tuple(P), 'previous': old,
                'delta': tuple(value-prior for value, prior in zip(P, old)), 'aux': tuple(aux)}


def _range_error(reason='a critical active coefficient or energy weight is unrepresentable'):
    return ValueError(f'hybrid-film derived float64 values are not representable: {reason}. '
                      'Rescale the physical units or use less extreme values.')


def _number(value, *, positive=False):
    """One final conversion; no positive cutoff or silent active-value underflow."""
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise _range_error() from error
    if not math.isfinite(result) or (positive and result <= 0):
        raise _range_error()
    return result


def descriptor(scene, grid):
    """A declared mixed Debye/oscillator film, or None for existing refusal paths."""
    structures = tuple(getattr(scene, 'structures', ()) or ())
    if len(structures) != 1 or getattr(scene, 'initial', None) is not None:
        return None
    structure = structures[0]
    shape, material = structure.shape, structure.material
    if (type(shape) is not Box or shape.angle != 0 or shape.range is not None
            or structure.discretisation != 'voxel' or type(material) is not Dispersive
            or any(type(p) not in (Debye, Drude, Lorentz) for p in material.poles)
            or not any(type(p) is Debye for p in material.poles)
            or not any(type(p) in (Drude, Lorentz) for p in material.poles)):
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
    kinds, deltas, taus, resonances, gammas, strengths = [], [], [], [], [], []
    for pole in material.poles:
        kind = type(pole).__name__.lower()
        kinds.append(kind)
        if kind == 'debye':
            delta, tau = concrete(pole.delta_eps), concrete(pole.tau)
            if tau <= 0:
                return None
            d, gamma, strength = 0.0, 0.0, 0.0
        else:
            delta, tau, gamma = 0.0, 0.0, concrete(pole.gamma)
            if kind == 'drude':
                frequency = Fraction(concrete(pole.omega_p))
                strength = _number(frequency*frequency, positive=True)
                d = 0.0
            else:
                strength = concrete(pole.strength)
                frequency = Fraction(concrete(pole.omega_0))
                d = _number(frequency*frequency, positive=True) if strength else 0.0
        deltas.append(delta)
        taus.append(tau)
        resonances.append(d)
        gammas.append(gamma)
        strengths.append(strength)
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
    low, high = (float(shape.center[axis]+sign*shape.size[axis]/2) for sign in (-1, 1))
    signature = ('hybrid-film-v1', tuple(float(v) for v in shape.size),
                 tuple(float(v) for v in shape.center), float(shape.angle), shape.axis,
                 a, b, tuple(type(p).__name__ for p in material.poles), values,
                 material.mu is not None, host.mu is not None, structure.discretisation,
                 structure.junction, tuple(grid.n), float(grid.dx),
                 tuple(float(v) for v in grid.low))
    return Film(axis, low, high, a, b, tuple(kinds), tuple(deltas), tuple(taus),
                tuple(resonances), tuple(gammas), tuple(strengths), box, signature)


def terms(scene, grid):
    """One four-array mixed-family term; inactive declarations still reserve slots."""
    film = descriptor(scene, grid)
    if film is None:
        return ()
    region = Region(z=film.box[2], y=film.box[1], x=film.box[0])
    slabs = tuple(Slab(c=c, z=region.z, y=region.y, x=region.x) for c in range(3))
    return (HybridFilmTerm(name='hybrid_film[0]', region=region, slabs=slabs,
                           n_poles=len(film.kinds), kinds=film.kinds, signature=film.signature),)


def refuse_mismatch(scene, plan):
    """Reject missing, additional or changed material memory, including traced reuse."""
    held = tuple(t for t in plan.terms if isinstance(t, HybridFilmTerm)
                 or getattr(t, 'name', '').startswith('hybrid_film['))
    expected = terms(scene, plan.grid)
    all_poles = tuple(t for t in plan.terms if hasattr(t, 'n_poles'))
    if held != expected or (expected and all_poles != expected):
        raise ValueError('the prepared hybrid-film plan does not match this scene. '
                         'Changed or traced geometry/material values and missing/extra '
                         'interface memory need a new supported fd.plan(scene).')


def _mixture(film, fraction, normal):
    """Exact physical eta, alpha and kappa, with zero/full-fill endpoint branches."""
    f, a, b = map(Fraction, (fraction, film.eps, film.host_eps))
    if f == 0:
        return b, Fraction(0), Fraction(0)
    if f == 1:
        return a, Fraction(0), Fraction(1)
    if normal:
        A = f*b+(1-f)*a
        return a*b/A, (1-f)/A, f*b*b/(A*A)
    return f*a+(1-f)*b, Fraction(0), f


def _node_profiles(film, fraction, normal, dt):
    """Finite scalar recurrence/storage data, derived with exact rational arithmetic."""
    eta, alpha, kappa = _mixture(film, fraction, normal)
    empty = (0.0,)*len(_SLOT_KEYS)
    if kappa == 0:
        return _number(eta, positive=True), 0.0, 0.0, (empty,)*len(film.kinds)
    h = Fraction(dt)
    rho = alpha/kappa
    bare = [Fraction(0)]*len(film.kinds)
    t = [Fraction(0)]*len(film.kinds)
    for j, (kind, delta, tau) in enumerate(zip(film.kinds, film.deltas, film.taus)):
        if kind == 'debye' and delta > 0:
            D, T = Fraction(delta), Fraction(tau)
            denominator = 2*T+h
            t[j] = 2*T/denominator
            bare[j] = kappa*D*h/denominator
    screen = 1+rho*sum(bare)
    c = [value/screen for value in bare]
    shift = sum(c)
    active = any(film.deltas) or any(film.strengths)
    rank = _number(rho, positive=True) if rho and active else 0.0
    slots = []
    for j, kind in enumerate(film.kinds):
        row = dict.fromkeys(_SLOT_KEYS, 0.0)
        if kind == 'debye' and film.deltas[j] > 0:
            gain = _number(2*t[j]-1)
            if not -1.0 < gain < 1.0:
                raise _range_error('an active Debye gain rounds to +/-1, losing strict '
                                   'relaxation separation at the discrete endpoint')
            beta = kappa*Fraction(film.deltas[j])
            row.update(debye_t=_number(t[j], positive=True), debye_gain=gain,
                       debye_c=_number(c[j], positive=True),
                       debye_drive=_number(2*c[j], positive=True),
                       debye_feedback=_number(rho*c[j], positive=bool(rho)),
                       energy_active=1.0, energy_debye_active=1.0,
                       energy_inverse_delta=_number(1/beta, positive=True))
        elif kind != 'debye' and film.strengths[j] > 0:
            d, gamma, strength = map(Fraction, (film.omega0_squared[j], film.gammas[j],
                                                film.strengths[j]))
            beta = kappa*strength
            denominator = 1+gamma*h/2
            row.update(step_gain=_number((2-d*h*h)/denominator),
                       hold=_number((1-gamma*h/2)/denominator),
                       drive=_number(beta*h*h/denominator, positive=True),
                       coupling=_number(alpha*strength*h*h/denominator, positive=bool(alpha)),
                       energy_active=1.0, energy_oscillator_active=1.0,
                       energy_inverse_strength=_number(1/beta, positive=True),
                       energy_resonance_over_strength=_number(d/beta, positive=bool(d)),
                       energy_kinetic=_number((1+h*h*gamma*gamma/4)/beta, positive=True),
                       energy_force_weight=_number(h*h*beta/4, positive=True))
        slots.append(tuple(row[key] for key in _SLOT_KEYS))
    return (_number(eta+shift, positive=True), _number(shift, positive=bool(shift)),
            rank, tuple(slots))


def _profiles(film, grid, dt):
    """Compact spatial arrays; scalar work is cached per distinct fill and component kind."""
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('hybrid-film coefficients need a finite positive time step.')
    n, count = grid.n[film.axis], len(film.kinds)
    eps, shift, rank = (np.zeros((3, n)) for _ in range(3))
    slots = {key: np.zeros((count, 3, n)) for key in _SLOT_KEYS}
    cached = {}
    for component in range(3):
        for index, fraction in enumerate(_fill(film, grid, component)):
            key = (float(fraction), component == film.axis)
            if key not in cached:
                cached[key] = _node_profiles(film, *key, dt)
            epsilon, c, rho, coefficients = cached[key]
            eps[component, index], shift[component, index], rank[component, index] = epsilon, c, rho
            for j, row in enumerate(coefficients):
                for name, value in zip(_SLOT_KEYS, row):
                    slots[name][j, component, index] = value
    result = {name: tuple(_compact(value[j], film.axis) for j in range(count))
              for name, value in slots.items()}
    result.update(eps=_compact(eps, film.axis), sigma=_compact(np.zeros_like(eps), film.axis),
                  energy_hybrid_shift=_compact(shift, film.axis),
                  energy_rank_one=_compact(rank, film.axis))
    return result


def material_profiles(scene, plan, rt):
    """None, or compact update eta+sum(c) and zero conductivity."""
    film = descriptor(scene, plan.grid)
    if film is None:
        return None
    profiles = _profiles(film, plan.grid, plan.dt)
    return tuple(rt.array(profiles[key], 'float64') for key in ('eps', 'sigma'))


def pole_params(scene, plan, rt):
    """Four-memory recurrence and explicit hybrid energy metadata, in declared order."""
    film = descriptor(scene, plan.grid)
    if film is None:
        return {}
    profiles = _profiles(film, plan.grid, plan.dt)
    cut = [slice(None)]*4
    cut[3-film.axis] = slice(*film.box[film.axis])
    cut = tuple(cut)
    array = lambda value: rt.array(value[cut], 'float64')
    coefficients = {key: tuple(array(value) for value in profiles[key]) for key in _SLOT_KEYS}
    for key in ('energy_hybrid_shift', 'energy_rank_one'):
        coefficients[key] = array(profiles[key])
    h = Fraction(plan.dt)
    coefficients.update(
        inverse_two_dt=rt.array(_number(1/(2*h), positive=True) if any(film.strengths) else 0., 'float64'),
        dx_over_dt=rt.array(_number(Fraction(plan.dx)/h, positive=True), 'float64'))
    return {'hybrid_film[0]': coefficients}


def policy_rate(scene, plan):
    """A conservative scoped material rate, leaving the physical-eta curl bound intact.

    The hybrid proof needs only R_O=max(d)+sum(S)/a. This first bounded
    proposal additionally retains the existing Debye sum-moment policy:
    R_D=max(bulk sum(delta/tau**2)/a, actual-node
    kappa/eta*(sum(delta/tau**2)+alpha*sum(delta/tau)**2)). Return R_O+R_D,
    not a claimed necessary condition. The positive extra budget preserves
    the older pure-Debye policy in inactive-oscillator limits. Zero-strength
    declarations contribute zero without evaluating unused reciprocals.
    """
    film = descriptor(scene, plan.grid)
    if film is None:
        return None
    moment1, moment2 = Fraction(0), Fraction(0)
    for delta, tau in zip(film.deltas, film.taus):
        if delta > 0:
            D, T = Fraction(delta), Fraction(tau)
            moment1 += D/T
            moment2 += D/(T*T)
    relaxation = moment2/Fraction(film.eps)
    if moment2:
        for component in range(3):
            for fraction in _fill(film, plan.grid, component):
                eta, alpha, kappa = _mixture(film, float(fraction), component == film.axis)
                relaxation = max(relaxation, kappa/eta*(moment2+alpha*moment1*moment1))
    active = [(d, S) for d, S in zip(film.omega0_squared, film.strengths) if S > 0]
    oscillator = (Fraction(max(d for d, _ in active))
                  + sum(Fraction(S) for _, S in active)/Fraction(film.eps)) if active else Fraction(0)
    rate = oscillator+relaxation
    if rate == 0:
        return 0.0
    result = _number(rate, positive=True)
    if Fraction(result) < rate:
        result = math.nextafter(result, math.inf)
    if not math.isfinite(result):
        raise _range_error()
    return result
