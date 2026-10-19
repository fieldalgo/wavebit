"""Conservative whole-band bulk certificate for a finite oscillator-sum film.

For x=omega**2, d_j=omega0_j**2 and L_j=d_j-x-i*gamma_j*sqrt(x),
|k|**2=|eps_inf*x+sum(S_j*x/L_j)| <= eps_inf*xmax+sum(S_j*max R_j),
R_j=x/sqrt((d_j-x)**2+gamma_j**2*x). Each R_j needs only band endpoints
and its possible interior peak 2*d_j**2/(2*d_j-gamma_j**2). Bounds for
distinct peaks are added conservatively, without assuming a common damping
or sampling interior frequencies. Exact rational candidates and outward
square-root rounding retain the mathematical upper-bound direction.

The certificate bounds bulk phase and skin scales, not every effective normal
interface resonance, oblique longitudinal scale or numerical observable error.
"""
import cmath
from fractions import Fraction
import math

from wavebit.fdtd.diagnostics import CoarseMeshWarning
from wavebit.materials._time_domain import concrete
from wavebit.fdtd._public_names import public_name
from wavebit.fdtd._prepare.shape_ranges import _carries
from wavebit.fdtd._prepare.oscillator_sum_film import descriptor


def _range_error():
    return ValueError('oscillator-sum-film spectral values exceed finite float64 range; '
                      'rescale the physical units or use less extreme values.')


def _finite(value, where):
    result = concrete(value)
    if _carries(value) or result is None or not math.isfinite(result) or result <= 0:
        raise ValueError(f'oscillator-sum-film {where} needs a concrete finite positive '
                         'value; its spectral certificate does not support traced bands.')
    return result


def _sqrt_upper(value):
    """A verified float upper bound on sqrt of a strictly positive rational."""
    try:
        rounded = float(value)
        if not 0 < rounded < math.inf:
            raise _range_error()
        root = math.sqrt(rounded)
        if Fraction(root)*Fraction(root) < value:
            root = math.nextafter(root, math.inf)
        if not 0 < root < math.inf or Fraction(root)*Fraction(root) < value:
            raise _range_error()
    except (OverflowError, ZeroDivisionError) as error:
        raise _range_error() from error
    return root


def _ratio_bound(resonance, gamma, xmin, xmax):
    """Analytic maximum of x/|L|, with exact band and critical-point comparisons."""
    d, damping = Fraction(resonance), Fraction(gamma)
    if damping == 0 and xmin <= d <= xmax:
        raise ValueError('oscillator-sum-film source band contains an active undamped '
                         'Lorentz resonance; no finite mesh certifies this band. '
                         'Narrow the band away from it or use physical damping.')
    candidates = [xmin, xmax]
    denominator = 2*d-damping*damping
    if denominator > 0:
        critical = 2*d*d/denominator
        if xmin < critical < xmax:
            candidates.append(critical)
    ratios_squared = []
    for x in candidates:
        denominator_squared = (d-x)*(d-x)+damping*damping*x
        if denominator_squared <= 0:
            raise _range_error()
        ratios_squared.append(x*x/denominator_squared)
    return _sqrt_upper(max(ratios_squared))


def _band_bound(film, low, high):
    """Sufficient bulk |k| maximum over the entire declared positive interval."""
    if not (math.isfinite(low) and math.isfinite(high) and 0 < low <= high):
        raise _range_error()
    xmin, xmax = Fraction(low)**2, Fraction(high)**2
    bound_squared = Fraction(film.eps)*xmax
    for d, gamma, strength in zip(film.omega0_squared, film.gammas, film.strengths):
        if strength == 0:
            continue
        upper = _ratio_bound(d, gamma, xmin, xmax)
        bound_squared += Fraction(strength)*Fraction(upper)
    return _sqrt_upper(bound_squared)


def _wavenumber(film, wavelength):
    """Actual sampled bulk index, retaining all active unequal-damping terms."""
    try:
        omega = 2*math.pi/wavelength
        x = omega*omega
        if not 0 < x < math.inf:
            raise _range_error()
        terms = [complex(film.eps)]
        for d, gamma, strength in zip(film.omega0_squared, film.gammas, film.strengths):
            if strength == 0:
                continue
            denominator = complex(d-x, -gamma*omega)
            if not (math.isfinite(denominator.real) and math.isfinite(denominator.imag)):
                raise _range_error()
            if denominator == 0:
                raise ValueError(f'oscillator-sum-film wavelength {wavelength:g} is an '
                                 'active undamped Lorentz resonance; no finite mesh '
                                 'resolves this sample.')
            value = strength/denominator
            if not (math.isfinite(value.real) and math.isfinite(value.imag)):
                raise _range_error()
            terms.append(value)
        eps = complex(math.fsum(t.real for t in terms), math.fsum(t.imag for t in terms))
        k = omega*cmath.sqrt(eps)
    except (OverflowError, ZeroDivisionError) as error:
        raise _range_error() from error
    if not math.isfinite(k.real) or not math.isfinite(k.imag):
        raise _range_error()
    return k


def index_at(film, wavelength):
    """Real index and metallic skin depth; inactive resonances are skipped exactly."""
    k = _wavenumber(film, wavelength)
    skin = 1/k.imag if k.imag > k.real else None
    return k.real*wavelength/(2*math.pi), skin


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


def _check_point(film, dx, w, label):
    for where, k in (('structures[0]', _wavenumber(film, w)),
                     ('Scene(background=)', complex(2*math.pi*math.sqrt(film.host_eps)/w))):
        if not (math.isfinite(k.real) and math.isfinite(k.imag)):
            raise _range_error()
        # Ordered divisions avoid a spurious zero denominator when k*dx
        # underflows for an extremely well-resolved finite positive sample.
        phase_cells = (2*math.pi/k.real)/dx if k.real > 0 else math.inf
        skin_cells = (1/k.imag)/dx if k.imag > 0 else math.inf
        if phase_cells < 8 or skin_cells < 1:
            raise ValueError(f'oscillator-sum-film {label} asks for wavelength {w:g}, where '
                             f'{where} has {phase_cells:.3g} cells per phase wavelength '
                             f'and {skin_cells:.3g} per skin depth: the required minima '
                             'are 8 and 1. Refine dx or drop this requested sample.')
        yield where, phase_cells, skin_cells


def validate(scene, grid):
    """Validate every declared source band and point, or False outside this scope."""
    film = descriptor(scene, grid)
    if film is None:
        return False
    for source in getattr(scene, 'sources', ()) or ():
        band = source.band
        wavelength = _finite(band.wavelength, f'source {source.name!r} band wavelength')
        df = _finite(band.df, f'source {source.name!r} band df')
        f0, width = 1/wavelength, band.sigmas*df
        if not math.isfinite(f0+width) or f0 <= width:
            raise ValueError('oscillator-sum-film source band needs a finite positive '
                             'frequency interval; narrow df.')
        low, high = 2*math.pi*(f0-width), 2*math.pi*(f0+width)
        bounds = (('structures[0]', _band_bound(film, low, high)),
                  ('Scene(background=)', _sqrt_upper(Fraction(film.host_eps)*Fraction(high)**2)))
        for where, k in bounds:
            if grid.dx*k > 2*math.pi/8:
                raise ValueError(f'oscillator-sum-film source {source.name!r} band: {where} '
                                 'exceeds the conservative bulk spectral mesh bound '
                                 f'(dx*K={grid.dx*k:.4g}, maximum 2*pi/8). Refine dx to '
                                 f'at most {2*math.pi/(8*k):g}, or narrow the band. '
                                 'The bound includes every active pole and interior '
                                 'resonance; it is sufficient, not a necessary threshold.')
    for w, label in sorted(_points(scene).items()):
        tuple(_check_point(film, grid.dx, w, label))
    return True


def notes(scene, plan, wavelength):
    """Scoped warning notes, preserving explicit zero-strength endpoint behavior."""
    film = descriptor(scene, plan.grid)
    if film is None:
        return None
    validate(scene, plan.grid)
    points = _points(scene)
    if wavelength is not None:
        points.setdefault(_finite(wavelength, 'diagnostic wavelength'), 'diagnostic')
    if not points and not scene.sources:
        return [(CoarseMeshWarning, 'The oscillator-sum film has no source band or '
                 'declared wavelength to certify its bulk mesh resolution; '
                 'validate observables under refinement.')]
    out = []
    for w, label in sorted(points.items()):
        for where, phase, skin in _check_point(film, plan.dx, w, label):
            if phase < 12 or skin < 2:
                out.append((CoarseMeshWarning, f'{where}: {phase:.3g} cells per phase '
                            f'wavelength and {skin:.3g} per skin depth at wavelength {w:g}; '
                            'the oscillator-sum bulk mesh floor is not an accuracy bound.'))
    return out
