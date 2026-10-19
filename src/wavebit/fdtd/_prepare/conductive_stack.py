"""Parallel conductive films, separated or an exactly touching pair.

Every member is a concrete Conductor (including sigma=0) on an unturned,
unranged voxel Box in one plain dielectric host. Each member separately passes
conductive_film.descriptor. They have the same normal axis and disjoint half-open
padded supports. The actual floor/ceil support decides admission, without a gap
tolerance. An unturned Box is cross-free, so there is no constitutive interpolation
halo beyond this support. H remains mu=1. Exactly two touching films instead use the local DC plus Debye
response in conductive_contact, with one memory covering their union.
Other material families,
a heterogeneous host and traced geometry/material values are outside this scope.

The singleton's analytic Yee fills, induced Debye recurrence and split energy
weights are reused without changing their arithmetic. Profiles are pasted into
the host, avoiding subtraction and readdition of a potentially much larger host
permittivity. Material storage adds on disjoint supports; the observer counts
the global electromagnetic field once. The retained conservative Debye CFL
policy takes the maximum local rate over members, not their sum across space.

For the bulk conductor k**2=a*omega**2+i*sigma*omega, hence
|k|**4=a**2*omega**4+sigma**2*omega**2 increases on every positive band.
The spectral guard checks its upper endpoint for every member and the host,
plus all monitor, Resolution and explicit PML wavelengths. This is a sufficient
bulk phase/skin certificate, not an observable-error or convergence bound.
"""
import dataclasses
from fractions import Fraction
import math

import numpy as np

from wavebit._common.conventions import boundary_kz

from wavebit.fdtd.diagnostics import CoarseMeshWarning
from wavebit.materials.time_domain import Conductor
from wavebit.materials.time_domain import concrete
from wavebit.fdtd._public_names import public_name
from wavebit.fdtd._kernel.masks import Region
from wavebit.fdtd._kernel.masks import Slab
import wavebit.fdtd._prepare.conductive_film as single
from wavebit.fdtd._prepare import conductive_contact as contact
from wavebit.fdtd._prepare.shape_ranges import _carries
from wavebit.fdtd._prepare.shape_ranges import materials_traced

__all__ = ['Stack', 'ConductiveStackTerm', 'descriptor', 'terms', 'material_profiles',
           'pole_params', 'extra_rate', 'refuse_mismatch', 'validate', 'notes', 'index_at']


@dataclasses.dataclass(frozen=True)
class Stack:
    """Members and identity stay in the original scene order."""

    axis: int
    films: tuple
    signature: tuple


@dataclasses.dataclass(frozen=True)
class ConductiveStackTerm(single.InterfacePoleTerm):
    """One member's memory, identified by the complete ordered stack."""

    member_index: int


def _member_scene(scene, index):
    """Project only structures; preserve the actual host, boundaries and requests."""
    return dataclasses.replace(scene, structures=(scene.structures[index],))


def _candidate(scene, grid):
    """All singleton-admitted members with a shared axis, before separation."""
    structures = tuple(getattr(scene, 'structures', ()) or ())
    if len(structures) < 2 or getattr(scene, 'initial', None) is not None:
        return None
    # Check all original leaves before projection/float conversion, including
    # Torch duals and shape orientation/range as well as sizes and centres.
    if materials_traced(scene) or any(_carries(s.shape) for s in structures):
        return None
    films = tuple(single.descriptor(_member_scene(scene, i), grid)
                  for i in range(len(structures)))
    if any(film is None for film in films):
        return None
    axis = films[0].axis
    if any(film.axis != axis for film in films):
        return None
    # Structure.junction defaults to 'split'. Disjoint supports preclude an
    # actual junction; the declaration still participates in the signature.
    signature = ('conductive-stack-v1', tuple(film.signature for film in films))
    return Stack(axis, films, signature)


def _overlap(stack):
    intervals = sorted((film.box[stack.axis], i) for i, film in enumerate(stack.films))
    return next(((left, right) for left, right in zip(intervals, intervals[1:])
                 if left[0][1] > right[0][0]), None)


def descriptor(scene, grid):
    """The fully admitted K>=2 stack, or None without admitting a partial subset."""
    stack = _candidate(scene, grid)
    return stack if stack is not None and (contact.touching(stack) or _overlap(stack) is None) else None


def _refuse_overlap(scene, grid):
    """Name an adjacent pair only when every other stack condition already passes."""
    stack = _candidate(scene, grid)
    if stack is None:
        return
    pair = _overlap(stack)
    if pair is not None and not contact.touching(stack):
        (left, i), (right, j) = pair
        raise ValueError(f'conductive-stack structures[{i}] support {left} and '
                         f'structures[{j}] support {right} overlap along {"xyz"[stack.axis]}. '
                         'The padded half-open grid supports must be disjoint; increase '
                         'the host gap or refine dx. Physical non-contact alone is not '
                         'sufficient, and no fixed physical-gap tolerance replaces this check.')


def terms(scene, grid):
    """One inherited Debye memory per member, including at sigma=0."""
    stack = descriptor(scene, grid)
    if stack is None:
        return ()
    if contact.touching(stack):
        box = contact.box(stack)
        region = Region(z=box[2], y=box[1], x=box[0])
        slabs = tuple(Slab(c=c, z=region.z, y=region.y, x=region.x) for c in range(3))
        return (ConductiveStackTerm(name='conductive_stack[contact]', region=region,
                slabs=slabs, n_poles=1, signature=stack.signature, member_index=-1),)
    out = []
    for index, film in enumerate(stack.films):
        region = Region(z=film.box[2], y=film.box[1], x=film.box[0])
        slabs = tuple(Slab(c=c, z=region.z, y=region.y, x=region.x) for c in range(3))
        out.append(ConductiveStackTerm(name=f'conductive_stack[{index}]', region=region,
                                       slabs=slabs, n_poles=1, signature=stack.signature,
                                       member_index=index))
    return tuple(out)


def refuse_mismatch(scene, plan):
    """Require all ordered memories, including through direct lower callers."""
    held = tuple(t for t in plan.terms
                 if isinstance(t, ConductiveStackTerm)
                 or getattr(t, 'name', '').startswith('conductive_stack['))
    expected = terms(scene, plan.grid)
    all_poles = tuple(t for t in plan.terms if hasattr(t, 'n_poles'))
    if held != expected or (expected and all_poles != expected):
        raise ValueError('the prepared conductive-stack plan does not match this scene. '
                         'Every member, its original order, and the complete interface '
                         'memory set must match; changed or traced values and missing, '
                         'extra or foreign memories require a new supported fd.plan(scene).')
    if not expected:
        _refuse_overlap(scene, plan.grid)
    structures = tuple(getattr(scene, 'structures', ()) or ())
    if not expected and len(structures) >= 2 and any(
            s.discretisation == 'voxel' and isinstance(s.material, Conductor)
            and s.shape.body() is not None for s in structures):
        raise ValueError('conductive-stack smoothing requires only concrete axis-aligned '
                         'Conductor films with one common normal, a plain lossless host, '
                         'and pairwise disjoint padded supports or exactly two touching films, outside absorbers. '
                         'Initial fields, tracing, gaps with overlapping support and mixed material '
                         'families are outside this scope; a prepared plan cannot bypass it.')


def material_profiles(scene, plan, rt):
    """Paste complete singleton profiles onto disjoint supports in one compact axis."""
    stack = descriptor(scene, plan.grid)
    if stack is None:
        return None
    if contact.touching(stack):
        eps, sigma, *_ = contact.profiles(stack, plan.grid, plan.dt)
        return rt.array(eps, 'float64'), rt.array(sigma, 'float64')
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
    """Keep singleton crop/energy arithmetic, changing only each dictionary key."""
    stack = descriptor(scene, plan.grid)
    if stack is None:
        return {}
    if contact.touching(stack):
        return contact.pole_params(stack, plan, rt)
    return {f'conductive_stack[{i}]':
            single.pole_params(_member_scene(scene, i), plan, rt)['conductive[0]']
            for i in range(len(stack.films))}


def extra_rate(scene, plan):
    """The existing induced-Debye policy's maximum across disjoint members."""
    stack = descriptor(scene, plan.grid)
    if stack is None:
        return 0.0
    if contact.touching(stack):
        return float(np.max(contact.profiles(stack, plan.grid, plan.dt)[-1]))
    return max(float(np.max(single._profiles(film, plan.grid, plan.dt)[-1]))
               for film in stack.films)


def _range_error():
    return ValueError('conductive-stack spectral values exceed finite float64 range; '
                      'rescale the physical units or use less extreme values.')


def _finite(value, where):
    result = concrete(value)
    if _carries(value) or result is None or not math.isfinite(result) or result <= 0:
        raise ValueError(f'conductive-stack {where} needs a concrete finite positive '
                         'value; its spectral certificate does not support traced bands.')
    return result


def _sqrt_upper(value):
    """An upward verified square root of a positive exact rational."""
    try:
        rounded = float(value)
        if not 0 < rounded < math.inf:
            raise _range_error()
        root = math.sqrt(rounded)
        if Fraction(root)**2 < value:
            root = math.nextafter(root, math.inf)
        if not 0 < root < math.inf or Fraction(root)**2 < value:
            raise _range_error()
    except (OverflowError, ZeroDivisionError) as error:
        raise _range_error() from error
    return root


def _bulk_bound(eps, sigma, high):
    """Upward rounded |k(high)|, with exact arithmetic preceding the roots."""
    A, S, W = Fraction(eps), Fraction(sigma), Fraction(high)
    fourth = A*A*W**4+S*S*W*W
    return _sqrt_upper(Fraction(_sqrt_upper(fourth)))


def _band_bound(film, low, high):
    try:
        if not (0 < float(low) <= float(high) < math.inf and 0 < low <= high):
            raise _range_error()
    except OverflowError as error:
        raise _range_error() from error
    return _bulk_bound(film.eps, film.sigma, high)


def _bulk_wavenumber(eps, sigma, wavelength):
    """Evaluate k squared before its standard passive root, avoiding sigma/omega."""
    try:
        W = Fraction(2*math.pi)/Fraction(wavelength)
        if not 0 < float(W) < math.inf:
            raise _range_error()
        real, loss = float(Fraction(eps)*W*W), float(Fraction(sigma)*W)
        if not (0 < real < math.inf and 0 <= loss < math.inf):
            raise _range_error()
        if sigma > 0 and loss == 0:
            raise _range_error()
        # The convention module owns the passive normal-incidence root.
        k = complex(boundary_kz(np, complex(real, loss), 0.0))
    except (OverflowError, ZeroDivisionError) as error:
        raise _range_error() from error
    if not (math.isfinite(k.real) and math.isfinite(k.imag) and k.real > 0 and k.imag >= 0):
        raise _range_error()
    if sigma > 0 and k.imag == 0:
        raise _range_error()
    return k


def index_at(film, wavelength):
    """This actual member's continuum index for its own proximity warning."""
    w = _finite(wavelength, 'diagnostic wavelength')
    k = _bulk_wavenumber(film.eps, film.sigma, w)
    skin = 1/k.imag if k.imag > k.real else None
    index = (k.real/(2*math.pi))*w
    if not 0 < index < math.inf:
        raise _range_error()
    return index, skin


def _points(scene):
    points = {}
    for monitor in getattr(scene, 'monitors', ()) or ():
        for value in getattr(monitor, 'wavelengths', ()) or ():
            w = _finite(value, f'{public_name(monitor)}({monitor.name!r}) wavelength')
            points.setdefault(w, f'{public_name(monitor)}({monitor.name!r})')
    value = scene.resolution.wavelength
    if value is not None:
        points.setdefault(_finite(value, 'Resolution wavelength'), 'Resolution')
    for name in ('x-', 'x+', 'y-', 'y+', 'z-', 'z+'):
        value = getattr(scene.boundaries.face(name), 'wavelength', None)
        if value is not None:
            points.setdefault(_finite(value, f'{name} PML wavelength'), f'{name} PML')
    return points


def _check_point(stack, dx, w, label):
    media = [('Scene(background=)', stack.films[0].host_eps, 0.0)]
    media += [(f'structures[{i}]', film.eps, film.sigma)
              for i, film in enumerate(stack.films)]
    for where, eps, sigma in media:
        k = _bulk_wavenumber(eps, sigma, w)
        phase = (2*math.pi/k.real)/dx
        skin = (1/k.imag)/dx if k.imag > 0 else math.inf
        if phase < 8 or skin < 1:
            raise ValueError(f'conductive-stack {label} asks for wavelength {w:g}, where '
                             f'{where} has {phase:.3g} cells per phase wavelength and '
                             f'{skin:.3g} per skin depth: the required minima are 8 and 1. '
                             'Refine dx or drop this requested sample.')
        yield where, phase, skin


def validate(scene, grid):
    """All source bands and requested points, or False outside the stack scope."""
    stack = descriptor(scene, grid)
    if stack is None:
        _refuse_overlap(scene, grid)
        return False
    for source in getattr(scene, 'sources', ()) or ():
        band = source.band
        wavelength = _finite(band.wavelength, f'source {source.name!r} band wavelength')
        df = _finite(band.df, f'source {source.name!r} band df')
        # Certify the public float sigmas and 2*pi convention without first
        # rounding the endpoint arithmetic inward. This does not claim exact
        # transcendental pi/log/sqrt values underlying those public constants.
        f0 = 1/Fraction(wavelength)
        width = Fraction(_finite(band.sigmas, 'source band sigmas'))*Fraction(df)
        if f0 <= width:
            raise ValueError('conductive-stack source band needs a finite positive '
                             'frequency interval; narrow df.')
        low, high = Fraction(2*math.pi)*(f0-width), Fraction(2*math.pi)*(f0+width)
        bounds = [(f'structures[{i}]', _band_bound(film, low, high))
                  for i, film in enumerate(stack.films)]
        bounds.insert(0, ('Scene(background=)', _bulk_bound(stack.films[0].host_eps, 0.0, high)))
        for where, k in bounds:
            if Fraction(grid.dx)*Fraction(k) > Fraction(2*math.pi)/8:
                raise ValueError(f'conductive-stack source {source.name!r} band: {where} '
                                 'exceeds the conservative bulk spectral mesh bound '
                                 f'(dx*K={grid.dx*k:.4g}, maximum 2*pi/8). Refine dx to '
                                 f'at most {(2*math.pi/8)/k:g}, or narrow the band. '
                                 'Every member is included; this modulus certificate is '
                                 'sufficient, not necessary.')
    for w, label in sorted(_points(scene).items()):
        tuple(_check_point(stack, grid.dx, w, label))
    return True


def notes(scene, plan, wavelength):
    """Resolution warnings preserve the original member index for every request."""
    stack = descriptor(scene, plan.grid)
    if stack is None:
        return None
    validate(scene, plan.grid)
    points = _points(scene)
    if wavelength is not None:
        points.setdefault(_finite(wavelength, 'diagnostic wavelength'), 'diagnostic')
    if not points and not scene.sources:
        return [(CoarseMeshWarning, 'The conductive stack has no source band or declared '
                 'wavelength to certify its bulk mesh resolution; validate observables '
                 'under refinement.')]
    out = []
    for w, label in sorted(points.items()):
        for where, phase, skin in _check_point(stack, plan.dx, w, label):
            if phase < 12 or skin < 2:
                out.append((CoarseMeshWarning, f'{where}: {phase:.3g} cells per phase '
                            f'wavelength and {skin:.3g} per skin depth at wavelength {w:g}; '
                            'the conductive-stack bulk mesh floor is not an accuracy bound.'))
    return out
