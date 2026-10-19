"""Separated parallel films with exactly one passive Drude or Lorentz pole each.

Cross-film Drude/Lorentz mixtures are explicitly admitted. Every exact Dispersive
member retains single_pole_film's descriptor, harmonic normal / arithmetic
transverse response, ADE coefficients and local energy weights without new
reduction arithmetic. Complete profiles are pasted on disjoint padded supports;
each member reserves one ordinary PoleTerm memory, including inactive Lorentz
slots. The observer counts the global electromagnetic energy once and adds the
local material storage. No cross-film constitutive coupling is introduced.

The singleton proof gives local Omega**2+S_eff/eps_inf <= d+S/a. Disjoint
supports make these local material blocks a direct sum, so their norm is bounded
by max_i(d_i+S_i/a_i), not a sum over films. The curl bound uses the smallest
physical high-frequency permittivity. As in the singleton, inactive Lorentz
frequency squares must remain finite and positive and still enter this
conservative time-step policy; no relaxed inactive-frequency range is claimed.

Every source band uses the existing single_pole_resolution._band_bound for
every member, including its possible interior resonance, plus the host. This is
the existing analytic triangle bound evaluated in ordinary finite floating
arithmetic, not a new outward-rounded spectral certificate. Requested points
retain the same phase/skin floors. These bulk checks do not certify interface
resolution, scattering error or spatial convergence. Decay and derivatives are
not admitted, and source construction retains its existing TFSF restrictions.
"""
import dataclasses
from fractions import Fraction
import math

import numpy as np

from wavebit.fdtd.diagnostics import CoarseMeshWarning
from wavebit.materials._time_domain import Drude
from wavebit.materials._time_domain import Lorentz
from wavebit.materials._time_domain import Dispersive
from wavebit.materials._time_domain import concrete
from wavebit.fdtd._kernel.masks import Region
from wavebit.fdtd._kernel.masks import Slab
from wavebit.fdtd._kernel.poles import PoleTerm
import wavebit.fdtd._prepare.single_pole_film as single
import wavebit.fdtd._prepare.single_pole_resolution as resolution
from wavebit.fdtd._prepare.shape_ranges import _carries
from wavebit.fdtd._prepare.shape_ranges import materials_traced
from ._identity import recorded_module

__all__ = ['Stack', 'OscillatorStackTerm', 'descriptor', 'terms', 'material_profiles',
           'pole_params', 'policy_rate', 'refuse_mismatch', 'validate', 'notes', 'index_at']


@dataclasses.dataclass(frozen=True)
class Stack:
    """Singleton descriptors and the complete identity, in original scene order."""

    axis: int
    films: tuple
    signature: tuple


@dataclasses.dataclass(frozen=True)
class OscillatorStackTerm(PoleTerm):
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
        return (recorded_module(type(value)), type(value).__qualname__,
                tuple((field.name, _boundary_record(getattr(value, field.name)))
                      for field in dataclasses.fields(value)))
    scalar = concrete(value)
    if scalar is None or not math.isfinite(scalar):
        raise ValueError('OscillatorStackConcrete: finite concrete boundary declarations required.')
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
    films = []
    for index, structure in enumerate(structures):
        material = structure.material
        if (type(material) is not Dispersive or len(material.poles) != 1
                or type(material.poles[0]) not in (Drude, Lorentz)):
            return None
        film = single.descriptor(_member_scene(scene, index), grid)
        if film is None:
            return None
        films.append(film)
    axis = films[0].axis
    if any(film.axis != axis for film in films):
        return None
    signature = ('oscillator-stack-v1', tuple(film.signature for film in films),
                 tuple(_boundary_record(scene.boundaries.face(name))
                       for name in ('x-', 'x+', 'y-', 'y+', 'z-', 'z+')))
    return Stack(axis, tuple(films), signature)


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
        raise ValueError(f'OscillatorStackOverlap: structures[{i}] support {left} and '
                         f'structures[{j}] support {right} overlap along {"xyz"[stack.axis]}. '
                         'Actual padded half-open supports must be disjoint. Increase '
                         'the host gap or refine dx; physical non-contact alone is '
                         'insufficient, and no fixed gap tolerance replaces this check.')


def _refuse_decay(scene, plan=None):
    if (getattr(scene.run, 'decay', None) is not None
            or getattr(plan, 'decay', None) is not None):
        raise ValueError('OscillatorStackDecayRefused: separated oscillator films require a '
                         'fixed Run time, steps or periods without a DecayStop. Their evolving '
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
        out.append(OscillatorStackTerm(name=f'oscillator_stack[{index}]', region=region,
                                       slabs=slabs, n_poles=1,
                                       signature=stack.signature, member_index=index))
    return tuple(out)


def refuse_mismatch(scene, plan):
    """Protect direct lower callers, including missing/foreign source or memory terms."""
    held = tuple(term for term in plan.terms if isinstance(term, OscillatorStackTerm)
                 or getattr(term, 'name', '').startswith('oscillator_stack['))
    expected = terms(scene, plan.grid)
    if held != expected:
        raise ValueError('OscillatorStackPlan: the prepared ordered stack does not match this '
                         'scene. Every member, declared pole and memory must retain its '
                         'identity; changed, missing or duplicate members need a new plan.')
    if expected:
        _refuse_decay(scene, plan)
        # This is called only after source construction, never during term planning.
        # Reconstructing static source descriptors may plan their source columns.
        # Those scenes contain no sources; no runtime lowering or field solve occurs.
        from wavebit.fdtd._prepare.sources import source_terms
        complete = expected + source_terms(scene, plan)
        if (len({term.name for term in complete}) != len(complete)
                or tuple(plan.terms) != complete):
            raise ValueError('OscillatorStackPlan: the complete ordered material/source term '
                             'tuple differs. Missing, extra, reordered or foreign memories '
                             'and sources cannot reuse this prepared plan.')
        return
    _refuse_overlap(scene, plan.grid)
    structures = tuple(getattr(scene, 'structures', ()) or ())
    if len(structures) >= 2 and any(
            s.discretisation == 'voxel' and type(s.material) is Dispersive
            and any(type(pole) in (Drude, Lorentz) for pole in s.material.poles)
            and s.shape.body() is not None for s in structures):
        raise ValueError('OscillatorStackScope: smoothing multiple oscillator films requires only '
                         'concrete parallel axis-aligned voxel Boxes, each with exactly one '
                         'passive Drude or Lorentz pole, a plain lossless nonmagnetic host, and '
                         'disjoint padded supports outside absorbers. Initial fields, tracing, '
                         'touching films, other materials, Debye poles and within-film sums '
                         'are outside this scope.')


def material_profiles(scene, plan, rt):
    """Paste unchanged singleton update profiles on disjoint supports in one axis."""
    stack = descriptor(scene, plan.grid)
    if stack is None:
        return None
    _refuse_decay(scene, plan)
    shape = [3, 1, 1, 1]
    shape[3-stack.axis] = plan.grid.n[stack.axis]
    eps = np.full(shape, stack.films[0].host_eps)
    sigma = np.zeros(shape)
    for film in stack.films:
        member_eps, member_sigma, *_ = single._profiles(film, plan.grid, plan.dt)
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
    _refuse_decay(scene, plan)
    return {f'oscillator_stack[{index}]': single.pole_params(
                _member_scene(scene, index), plan, rt)['dispersive_film[0]']
            for index in range(len(stack.films))}


def _range_error():
    return ValueError('OscillatorStackRange: derived values exceed finite float64 '
                      'range; rescale physical units or use less extreme values.')


def policy_rate(scene, plan):
    """None outside scope; otherwise the largest member's conservative d+S/a.

    Reserve and retain the singleton's inactive Lorentz frequency contribution.
    Exact rational formation and upward rounding avoid overflowing an unused
    intermediate or rounding the material-rate enclosure down.
    """
    stack = descriptor(scene, plan.grid)
    if stack is None:
        return None
    _refuse_decay(scene, plan)
    rate = max(Fraction(film.omega0_squared)
               + Fraction(film.strength)/Fraction(film.eps) for film in stack.films)
    try:
        result = float(rate)
    except OverflowError as error:
        raise _range_error() from error
    if not 0 < result < math.inf:
        raise _range_error()
    if Fraction(result) < rate:
        result = math.nextafter(result, math.inf)
    if not math.isfinite(result):
        raise _range_error()
    return result


def _finite(value, where):
    result = concrete(value)
    if _carries(value) or result is None or not math.isfinite(result) or result <= 0:
        raise ValueError(f'OscillatorStackConcrete: {where} needs a concrete finite '
                         'positive value; traced bands and wavelengths are not supported.')
    return result


def _band_bound(film, low, high):
    """Reuse the singleton interior-resonance-aware triangle bound unchanged."""
    return resolution._band_bound(film, low, high)


def _wavenumber(film, wavelength):
    """Preserve singleton response, refusing nonfinite active denominator factors."""
    omega = 2*math.pi/wavelength
    squared = omega*omega
    if not 0 < squared < math.inf:
        raise _range_error()
    if film.strength > 0:
        damping = film.gamma*omega
        if (not math.isfinite(damping) or not math.isfinite(film.omega0_squared-squared)
                or (film.gamma > 0 and damping == 0)):
            raise _range_error()
    return resolution._wavenumber(film, wavelength)


def index_at(film, wavelength):
    """Per-member bulk phase index and metallic skin depth, including inactive poles."""
    w = _finite(wavelength, 'diagnostic wavelength')
    k = _wavenumber(film, w)
    index = k.real*w/(2*math.pi)
    if not math.isfinite(index) or index < 0:
        raise _range_error()
    return index, 1/k.imag if k.imag > k.real else None


def _points(scene):
    return resolution._points(scene)


def _check_point(stack, dx, wavelength, label):
    host = dataclasses.replace(stack.films[0], eps=stack.films[0].host_eps, strength=0.0)
    media = [('Scene(background=)', host)]
    media += [(f'structures[{index}]', film) for index, film in enumerate(stack.films)]
    for where, film in media:
        k = _wavenumber(film, wavelength)
        phase = (2*math.pi/k.real)/dx if k.real > 0 else math.inf
        skin = (1/k.imag)/dx if k.imag > 0 else math.inf
        if phase < 8 or skin < 1:
            raise ValueError(f'OscillatorStackSpectral: {label} asks for wavelength '
                             f'{wavelength:g}, where {where} has {phase:.3g} cells per '
                             f'phase wavelength and {skin:.3g} per skin depth; the required '
                             'minima are 8 and 1. Refine dx or remove the requested sample.')
        yield where, phase, skin


def validate(scene, grid):
    """Every source band and requested point for every member, False outside scope."""
    stack = descriptor(scene, grid)
    if stack is None:
        _refuse_overlap(scene, grid)
        return False
    _refuse_decay(scene)
    for source in getattr(scene, 'sources', ()) or ():
        band = source.band
        wavelength = _finite(band.wavelength, f'source {source.name!r} band wavelength')
        df = _finite(band.df, f'source {source.name!r} band df')
        sigmas = _finite(band.sigmas, f'source {source.name!r} band sigmas')
        f0, width = 1/wavelength, sigmas*df
        if not math.isfinite(f0+width) or f0 <= width or width == 0:
            raise ValueError('OscillatorStackSpectral: a source band needs a finite '
                             'positive frequency interval; narrow df or rescale units.')
        low, high = 2*math.pi*(f0-width), 2*math.pi*(f0+width)
        bounds = [(f'structures[{index}]', _band_bound(film, low, high))
                  for index, film in enumerate(stack.films)]
        host = dataclasses.replace(stack.films[0], eps=stack.films[0].host_eps, strength=0.0)
        bounds.insert(0, ('Scene(background=)', _band_bound(host, low, high)))
        for where, bound in bounds:
            if grid.dx*bound > 2*math.pi/8:
                raise ValueError(f'OscillatorStackSpectral: source {source.name!r} band: '
                                 f'{where} exceeds the conservative bulk spectral mesh '
                                 f'bound (dx*K={grid.dx*bound:.4g}, maximum 2*pi/8). '
                                 'Refine dx or narrow the source band. Every member and '
                                 'interior resonance is included; this sufficient bulk '
                                 'floor is not an accuracy bound.')
    for wavelength, label in sorted(_points(scene).items()):
        tuple(_check_point(stack, grid.dx, wavelength, label))
    return True


def notes(scene, plan, wavelength=None):
    """All-member phase/skin notes retain original structure indices."""
    stack = descriptor(scene, plan.grid)
    if stack is None:
        return None
    validate(scene, plan.grid)
    points = _points(scene)
    if wavelength is not None:
        points.setdefault(_finite(wavelength, 'diagnostic wavelength'), 'diagnostic')
    if not points and not scene.sources:
        return [(CoarseMeshWarning, 'The oscillator stack has no source band or declared '
                 'wavelength to certify bulk resolution; validate observables under '
                 'refinement. No bulk mesh certificate was inferred.')]
    out = []
    for w, label in sorted(points.items()):
        for where, phase, skin in _check_point(stack, plan.dx, w, label):
            if phase < 12 or skin < 2:
                out.append((CoarseMeshWarning, f'{where}: {phase:.3g} cells per phase '
                            f'wavelength and {skin:.3g} per skin depth at wavelength {w:g}; '
                            'the oscillator-stack bulk floor is not an accuracy bound.'))
    return out
