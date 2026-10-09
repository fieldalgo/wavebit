"""Separated parallel films containing finite passive Debye sums.

Each exact Dispersive member is projected onto the existing one-, two-, or
many-Debye singleton path according to its declared pole count. Their padded
half-open grid supports must be disjoint along a common normal axis. No local
response, modal reduction, recurrence or energy coefficient is recomputed here.
Complete profiles are pasted into the common host; polarization memories add
on disjoint supports and the global electromagnetic energy is counted once.

For a member with a+chi(s), chi=sum(delta_i/(1+tau_i*s)), its normal response
at fill f is a*b/A+kappa*chi/(1+alpha*chi), with A=f*b+(1-f)*a,
alpha=(1-f)/A and kappa=f*b*b/A**2. Its tangential response is arithmetic.
The singleton modules document and realize these passive rational responses.
The retained Debye CFL policy is the maximum of each member's bulk and actual
local summed rates, then the maximum across members, never a spatial sum.

For x=omega**2, |eps|**2=a**2+sum(C_i/(1+tau_i**2*x)), where
C_i=delta_i**2+2*a*delta_i+sum_{j!=i}2*delta_i*delta_j*tau_i/(tau_i+tau_j).
All C_i are nonnegative. Therefore |k|**4=x**2*|eps|**2 increases for x>0.
The whole-band maximum is the upper endpoint. Here its real and imaginary
permittivity parts are formed as exact rationals, and two upward-verified
square roots bound |k|. The certificate refers to the public float parameters,
including band.sigmas and 2*pi, not exact transcendental values. It certifies
bulk mesh scales, not interface resolution, scattering accuracy or convergence.
"""
import dataclasses
from fractions import Fraction
import math

import numpy as np

from wavebit._core.conventions import boundary_kz

from ..diagnostics import CoarseMeshWarning
from ..materials import Debye, Dispersive, concrete
from .._kernel.masks import Region, Slab
from .._kernel.poles import PoleTerm
from . import debye_film, two_debye_film, debye_sum_film
from .shape_ranges import _carries, materials_traced

__all__ = ['Stack', 'DebyeStackTerm', 'descriptor', 'terms', 'material_profiles',
           'pole_params', 'policy_rate', 'refuse_mismatch', 'validate', 'notes', 'index_at']

_MODULES = {'one': debye_film, 'two': two_debye_film, 'sum': debye_sum_film}
_KEYS = {'one': 'debye_film[0]', 'two': 'two_debye_film[0]', 'sum': 'debye_sum_film[0]'}


@dataclasses.dataclass(frozen=True)
class Stack:
    """Singleton descriptors and the complete identity, in original scene order."""

    axis: int
    films: tuple
    kinds: tuple
    signature: tuple


@dataclasses.dataclass(frozen=True)
class DebyeStackTerm(PoleTerm):
    """One member's original declared memory slots, bound to the entire stack."""

    signature: tuple
    member_index: int


def _member_scene(scene, index):
    """Preserve host, faces, sources and requests while selecting one structure."""
    return dataclasses.replace(scene, structures=(scene.structures[index],))


def _boundary_record(value):
    """Concrete face declaration data, without an opaque object repr."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (tuple, list)):
        return tuple(_boundary_record(item) for item in value)
    if dataclasses.is_dataclass(value):
        return (type(value).__module__, type(value).__qualname__,
                tuple((field.name, _boundary_record(getattr(value, field.name)))
                      for field in dataclasses.fields(value)))
    scalar = concrete(value)
    if scalar is None or not math.isfinite(scalar):
        raise ValueError('DebyeStackConcrete: finite concrete boundary declarations required.')
    return scalar


def _candidate(scene, grid):
    """A complete singleton-admitted common-axis stack, before separation."""
    structures = tuple(getattr(scene, 'structures', ()) or ())
    if len(structures) < 2 or getattr(scene, 'initial', None) is not None:
        return None
    # Inspect original declarations before projection or host float conversion.
    # This includes Torch forward duals, geometry/orientation, source and band leaves.
    if materials_traced(scene) or _carries(scene):
        return None
    kinds, films = [], []
    for index, structure in enumerate(structures):
        material = structure.material
        if (type(material) is not Dispersive or not material.poles
                or any(type(pole) is not Debye for pole in material.poles)):
            return None
        count = len(material.poles)
        kind = 'one' if count == 1 else 'two' if count == 2 else 'sum'
        film = _MODULES[kind].descriptor(_member_scene(scene, index), grid)
        if film is None:
            return None
        kinds.append(kind)
        films.append(film)
    axis = films[0].axis
    if any(film.axis != axis for film in films):
        return None
    signature = ('debye-stack-v1', tuple((kind, film.signature)
                                       for kind, film in zip(kinds, films)),
                 tuple(_boundary_record(scene.boundaries.face(name))
                       for name in ('x-', 'x+', 'y-', 'y+', 'z-', 'z+')))
    return Stack(axis, tuple(films), tuple(kinds), signature)


def _overlap(stack):
    intervals = sorted((film.box[stack.axis], index)
                       for index, film in enumerate(stack.films))
    return next(((left, right) for left, right in zip(intervals, intervals[1:])
                 if left[0][1] > right[0][0]), None)


def descriptor(scene, grid):
    """K>=2 structurally admitted members, or None; never admit a partial subset.

    Decay remains visible structurally so validate/refuse_mismatch can issue its
    named refusal rather than falling through to an unrelated material route.
    """
    stack = _candidate(scene, grid)
    return stack if stack is not None and _overlap(stack) is None else None


def _refuse_overlap(scene, grid):
    stack = _candidate(scene, grid)
    if stack is None:
        return
    pair = _overlap(stack)
    if pair is not None:
        (left, i), (right, j) = pair
        raise ValueError(f'DebyeStackOverlap: structures[{i}] support {left} and '
                         f'structures[{j}] support {right} overlap along {"xyz"[stack.axis]}. '
                         'Actual padded half-open supports must be disjoint. Increase '
                         'the host gap or refine dx; physical non-contact alone is '
                         'insufficient, and no fixed gap tolerance replaces this check.')


def _refuse_decay(scene):
    if getattr(scene.run, 'decay', None) is not None:
        raise ValueError('DebyeStackDecayRefused: separated Debye films currently require '
                         'a fixed Run time, steps or periods without Decay. Their evolving '
                         'state has not been admitted to the stopping protocol.')


def terms(scene, grid):
    """Reserve one ordinary-layout term per member, including all inactive slots."""
    stack = descriptor(scene, grid)
    if stack is None:
        return ()
    out = []
    for index, film in enumerate(stack.films):
        region = Region(z=film.box[2], y=film.box[1], x=film.box[0])
        slabs = tuple(Slab(c=c, z=region.z, y=region.y, x=region.x) for c in range(3))
        out.append(DebyeStackTerm(name=f'debye_stack[{index}]', region=region,
                                 slabs=slabs, n_poles=len(scene.structures[index].material.poles),
                                 signature=stack.signature, member_index=index))
    return tuple(out)


def refuse_mismatch(scene, plan):
    """Protect direct lower callers, including missing/foreign source or memory terms."""
    held = tuple(term for term in plan.terms if isinstance(term, DebyeStackTerm)
                 or getattr(term, 'name', '').startswith('debye_stack['))
    expected = terms(scene, plan.grid)
    if held != expected:
        raise ValueError('DebyeStackPlan: the prepared ordered stack does not match this '
                         'scene. Every member, declared pole and memory must retain its '
                         'identity; changed, missing or duplicate members need a new plan.')
    if expected:
        _refuse_decay(scene)
        # This is called only after source construction, never during term planning.
        # Reconstructing static source descriptors may plan their source columns.
        # Those scenes contain no sources; no runtime lowering or field solve occurs.
        from .sources import source_terms
        complete = expected + source_terms(scene, plan)
        if (len({term.name for term in complete}) != len(complete)
                or tuple(plan.terms) != complete):
            raise ValueError('DebyeStackPlan: the complete ordered material/source term '
                             'tuple differs. Missing, extra, reordered or foreign memories '
                             'and sources cannot reuse this prepared plan.')
        return
    _refuse_overlap(scene, plan.grid)
    structures = tuple(getattr(scene, 'structures', ()) or ())
    if len(structures) >= 2 and any(
            s.discretisation == 'voxel' and type(s.material) is Dispersive
            and any(type(pole) is Debye for pole in s.material.poles)
            and s.shape.body() is not None for s in structures):
        raise ValueError('DebyeStackScope: smoothing multiple Debye films requires only '
                         'concrete parallel axis-aligned voxel Boxes, each with one or more '
                         'passive Debye poles, a plain lossless nonmagnetic host, and disjoint '
                         'padded supports outside absorbers. Initial fields, tracing, touching '
                         'films, other materials and pole families are outside this scope.')


def material_profiles(scene, plan, rt):
    """Paste unchanged singleton update profiles on disjoint supports in one axis."""
    stack = descriptor(scene, plan.grid)
    if stack is None:
        return None
    _refuse_decay(scene)
    shape = [3, 1, 1, 1]
    shape[3-stack.axis] = plan.grid.n[stack.axis]
    eps = np.full(shape, stack.films[0].host_eps)
    sigma = np.zeros(shape)
    for kind, film in zip(stack.kinds, stack.films):
        member_eps, member_sigma, *_ = _MODULES[kind]._profiles(film, plan.grid, plan.dt)
        cut = [slice(None)]*4
        cut[3-stack.axis] = slice(*film.box[stack.axis])
        cut = tuple(cut)
        eps[cut] = member_eps[cut]
        sigma[cut] = member_sigma[cut]
    return rt.array(eps, 'float64'), rt.array(sigma, 'float64')


def pole_params(scene, plan, rt):
    """Reuse each singleton's coefficients, crop and energy weights; rename its key."""
    stack = descriptor(scene, plan.grid)
    if stack is None:
        return {}
    _refuse_decay(scene)
    return {f'debye_stack[{index}]': _MODULES[kind].pole_params(
                _member_scene(scene, index), plan, rt)[_KEYS[kind]]
            for index, kind in enumerate(stack.kinds)}


def policy_rate(scene, plan):
    """None outside scope, else the maximum complete singleton Debye policy rate.

    The caller must skip ordinary Debye.rate() for this entire recognized stack.
    Every member supplies its bulk/local summed rates, including exact zero when
    all declared strengths are inactive; no inactive extreme tau is evaluated.
    """
    stack = descriptor(scene, plan.grid)
    if stack is None:
        return None
    _refuse_decay(scene)
    return max(_MODULES[kind].policy_rate(_member_scene(scene, index), plan)
               for index, kind in enumerate(stack.kinds))


def _range_error():
    return ValueError('DebyeStackRange: derived spectral values exceed finite positive '
                      'float64 range; rescale physical units or use less extreme values.')


def _finite(value, where):
    result = concrete(value)
    if _carries(value) or result is None or not math.isfinite(result) or result <= 0:
        raise ValueError(f'DebyeStackConcrete: {where} needs a concrete finite positive '
                         'value; traced bands and wavelengths are not supported.')
    return result


def _declared(film):
    return ((film.delta,), (film.tau,)) if hasattr(film, 'delta') else (film.deltas, film.taus)


def _epsilon_parts(eps, deltas, taus, omega):
    """Exact rational real/loss parts under the e^{-i omega t} convention."""
    omega = Fraction(omega)
    real, loss = Fraction(eps), Fraction(0)
    for delta, tau in zip(deltas, taus):
        if delta == 0:
            continue
        ratio = omega*Fraction(tau)
        denominator = 1+ratio*ratio
        real += Fraction(delta)/denominator
        loss += Fraction(delta)*ratio/denominator
    return real, loss


def _sqrt_upper(value):
    """Finite positive float whose square is certified above the exact rational."""
    try:
        rounded = float(value)
        if not 0 < rounded < math.inf:
            raise _range_error()
        root = math.sqrt(rounded)
        if Fraction(root)**2 < value:
            root = math.nextafter(root, math.inf)
        if not 0 < root < math.inf or Fraction(root)**2 < value:
            raise _range_error()
        return root
    except (OverflowError, ZeroDivisionError) as error:
        raise _range_error() from error


def _bulk_bound(eps, deltas, taus, high):
    """Upward-certified endpoint |k|, with no rounded complex arithmetic in the bound."""
    high = Fraction(high)
    real, loss = _epsilon_parts(eps, deltas, taus, high)
    return _sqrt_upper(Fraction(_sqrt_upper(high**4*(real*real+loss*loss))))


def _band_bound(film, low, high):
    try:
        if not (0 < low <= high and 0 < float(low) <= float(high) < math.inf):
            raise _range_error()
    except OverflowError as error:
        raise _range_error() from error
    return _bulk_bound(film.eps, *_declared(film), high)


def _bulk_wavenumber(eps, deltas, taus, wavelength):
    """Requested-point wavenumber using the shared passive-root convention."""
    try:
        omega = Fraction(2*math.pi)/Fraction(wavelength)
        if not 0 < float(omega) < math.inf:
            raise _range_error()
        real, loss = _epsilon_parts(eps, deltas, taus, omega)
        real, loss = float(omega*omega*real), float(omega*omega*loss)
        active = any(delta > 0 for delta in deltas)
        if not (0 < real < math.inf and 0 <= loss < math.inf) or (active and loss == 0):
            raise _range_error()
        k = complex(boundary_kz(np, complex(real, loss), 0.0))
    except (OverflowError, ZeroDivisionError) as error:
        raise _range_error() from error
    if (not math.isfinite(k.real) or not math.isfinite(k.imag) or k.real <= 0
            or k.imag < 0 or (active and k.imag == 0)):
        raise _range_error()
    return k


def index_at(film, wavelength):
    """Any singleton descriptor's bulk index, preserving the actual member identity."""
    w = _finite(wavelength, 'diagnostic wavelength')
    k = _bulk_wavenumber(film.eps, *_declared(film), w)
    index = (k.real/(2*math.pi))*w
    if not 0 < index < math.inf:
        raise _range_error()
    return index, 1/k.imag if k.imag > k.real else None


def _points(scene):
    points = {}
    for monitor in getattr(scene, 'monitors', ()) or ():
        for value in getattr(monitor, 'wavelengths', ()) or ():
            w = _finite(value, f'{type(monitor).__name__}({monitor.name!r}) wavelength')
            points.setdefault(w, f'{type(monitor).__name__}({monitor.name!r})')
    value = scene.resolution.wavelength
    if value is not None:
        points.setdefault(_finite(value, 'Resolution wavelength'), 'Resolution')
    for name in ('x-', 'x+', 'y-', 'y+', 'z-', 'z+'):
        value = getattr(scene.boundaries.face(name), 'wavelength', None)
        if value is not None:
            points.setdefault(_finite(value, f'{name} PML wavelength'), f'{name} PML')
    return points


def _check_point(stack, dx, wavelength, label):
    media = [('Scene(background=)', stack.films[0].host_eps, (), ())]
    media += [(f'structures[{index}]', film.eps, *_declared(film))
              for index, film in enumerate(stack.films)]
    for where, eps, deltas, taus in media:
        k = _bulk_wavenumber(eps, deltas, taus, wavelength)
        phase = (2*math.pi/k.real)/dx
        skin = (1/k.imag)/dx if k.imag > 0 else math.inf
        if phase < 8 or skin < 1:
            raise ValueError(f'DebyeStackSpectral: {label} asks for wavelength {wavelength:g}, '
                             f'where {where} has {phase:.3g} cells per phase wavelength and '
                             f'{skin:.3g} per skin depth; the required minima are 8 and 1. '
                             'Refine dx or remove the requested sample.')
        yield where, phase, skin


def validate(scene, grid):
    """Every source band and requested point for every member, or False outside scope."""
    stack = descriptor(scene, grid)
    if stack is None:
        _refuse_overlap(scene, grid)
        return False
    _refuse_decay(scene)
    for source in getattr(scene, 'sources', ()) or ():
        band = source.band
        wavelength = _finite(band.wavelength, f'source {source.name!r} band wavelength')
        df = _finite(band.df, f'source {source.name!r} band df')
        f0 = 1/Fraction(wavelength)
        width = Fraction(_finite(band.sigmas, 'source band sigmas'))*Fraction(df)
        if f0 <= width:
            raise ValueError('DebyeStackSpectral: a source band needs a finite positive '
                             'frequency interval; narrow df.')
        low = Fraction(2*math.pi)*(f0-width)
        high = Fraction(2*math.pi)*(f0+width)
        bounds = [(f'structures[{index}]', _band_bound(film, low, high))
                  for index, film in enumerate(stack.films)]
        bounds.insert(0, ('Scene(background=)',
                         _bulk_bound(stack.films[0].host_eps, (), (), high)))
        for where, bound in bounds:
            if Fraction(grid.dx)*Fraction(bound) > Fraction(2*math.pi)/8:
                raise ValueError(f'DebyeStackSpectral: source {source.name!r} band: {where} '
                                 'exceeds the conservative bulk spectral mesh bound '
                                 f'(dx*K={grid.dx*bound:.4g}, maximum 2*pi/8). Refine dx '
                                 'or narrow the source band. Every member is included; '
                                 'this sufficient bulk floor is not an accuracy bound.')
    for wavelength, label in sorted(_points(scene).items()):
        tuple(_check_point(stack, grid.dx, wavelength, label))
    return True


def notes(scene, plan, wavelength=None):
    """Scoped phase/skin diagnostics use the original structure index throughout."""
    stack = descriptor(scene, plan.grid)
    if stack is None:
        return None
    validate(scene, plan.grid)
    points = _points(scene)
    if wavelength is not None:
        points.setdefault(_finite(wavelength, 'diagnostic wavelength'), 'diagnostic')
    if not points and not scene.sources:
        return [(CoarseMeshWarning, 'The Debye stack has no source band or declared '
                 'wavelength to certify bulk resolution; validate observables under '
                 'refinement. No bulk mesh certificate was inferred.')]
    out = []
    for w, label in sorted(points.items()):
        for where, phase, skin in _check_point(stack, plan.dx, w, label):
            if phase < 12 or skin < 2:
                out.append((CoarseMeshWarning, f'{where}: {phase:.3g} cells per phase '
                            f'wavelength and {skin:.3g} per skin depth at wavelength {w:g}; '
                            'the Debye-stack bulk floor is not an accuracy bound.'))
    return out
