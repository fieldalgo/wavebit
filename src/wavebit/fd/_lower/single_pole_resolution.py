"""Bulk spectral mesh certificate for the bounded single-pole film scope.

For x=omega**2 and L=omega0**2-x-i*gamma*sqrt(x),
|k|**2 = |eps_inf*x+S*x/L| <= eps_inf*x_max+S*max R(x),
R=x/sqrt((omega0**2-x)**2+gamma**2*x). Its only possible interior
maximum is x=2*omega0**4/(2*omega0**2-gamma**2). Endpoints plus this
candidate bound every frequency in the declared positive source interval.
This sufficient bound certifies bulk phase and skin scales, not every normal
interface resonance or oblique longitudinal scale. Refinement is still needed.
"""
import cmath
import math

from ..diagnostics import CoarseMeshWarning
from ..materials import concrete
from .shape_ranges import _carries
from .single_pole_film import descriptor


def _finite(value, where):
    result = concrete(value)
    if _carries(value) or result is None or not math.isfinite(result) or result <= 0:
        raise ValueError(f'single-pole-film {where} needs a concrete finite positive value; '
                         'its spectral mesh certificate does not support traced bands.')
    return result


def _band_bound(film, low, high):
    """Upper bound on the bulk wavenumber modulus over positive angular frequencies."""
    try:
        xmax, xmin = high*high, low*low
        if not 0 < xmin <= xmax < math.inf:
            raise ValueError('single-pole-film squared band frequencies exceed finite '
                             'float64 range; rescale the physical units.')
        if film.strength == 0:
            bound = math.sqrt(film.eps)*high
        else:
            w0, gamma = film.omega0_squared, film.gamma
            if gamma == 0 and xmin <= w0 <= xmax:
                raise ValueError('single-pole-film source band contains an undamped '
                                 'Lorentz resonance; no finite mesh certifies this band. '
                                 'Narrow the band away from it or use physical damping.')
            candidates = [xmin, xmax]
            # Divide before squaring: 2*w0*w0 can overflow even when the critical
            # frequency and every input are finite. A nonfinite denominator must
            # never silently erase the interior candidate.
            scaled_damping = (gamma/math.sqrt(w0))**2 if w0 else math.inf
            if scaled_damping < 2:
                critical = w0/(1-scaled_damping/2)
                if xmin < critical < xmax:
                    candidates.append(critical)
            denominators = [math.hypot(w0-x, gamma*math.sqrt(x)) for x in candidates]
            if not all(0 < d < math.inf for d in denominators):
                raise ValueError('single-pole-film spectral denominator exceeds finite '
                                 'float64 range; rescale the physical units.')
            ratios = [x/d for x, d in zip(candidates, denominators)]
            bound = math.sqrt(film.eps*xmax+film.strength*max(ratios))
    except (OverflowError, ZeroDivisionError) as error:
        raise ValueError('single-pole-film spectral bound exceeds finite float64 range; '
                         'rescale the physical units.') from error
    if not math.isfinite(bound) or bound <= 0:
        raise ValueError('single-pole-film spectral bound is not finite and positive; '
                         'rescale the physical units.')
    return bound


def _wavenumber(film, wavelength):
    omega = 2*math.pi/wavelength
    if not 0 < omega*omega < math.inf:
        raise ValueError('single-pole-film sampled frequency exceeds finite float64 range; '
                         'rescale the physical units.')
    eps = complex(film.eps)
    if film.strength != 0:
        denominator = complex(film.omega0_squared-omega*omega, -film.gamma*omega)
        if denominator == 0:
            raise ValueError(f'single-pole-film wavelength {wavelength:g} is an undamped '
                             'Lorentz resonance; no finite mesh resolves this sample.')
        eps += film.strength/denominator
    k = omega*cmath.sqrt(eps)
    if not math.isfinite(k.real) or not math.isfinite(k.imag):
        raise ValueError('single-pole-film sampled wavenumber exceeds finite float64 range; '
                         'rescale the physical units.')
    return k


def index_at(film, wavelength):
    """Real index and metallic skin depth, including the zero-strength resonance endpoint."""
    k = _wavenumber(film, wavelength)
    skin = 1/k.imag if k.imag > k.real else None
    return k.real*wavelength/(2*math.pi), skin


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


def _check_point(film, dx, w, label):
    for where, k in (('structures[0]', _wavenumber(film, w)),
                     ('Scene(background=)', complex(2*math.pi*math.sqrt(film.host_eps)/w))):
        phase_cells = 2*math.pi/(k.real*dx) if k.real > 0 else math.inf
        skin_cells = 1/(k.imag*dx) if k.imag > 0 else math.inf
        if phase_cells < 8 or skin_cells < 1:
            raise ValueError(f'single-pole-film {label} asks for wavelength {w:g}, where '
                             f'{where} has {phase_cells:.3g} cells per phase wavelength '
                             f'and {skin_cells:.3g} per skin depth: the required minima '
                             'are 8 and 1. Refine dx or drop this requested sample.')
        yield where, phase_cells, skin_cells


def validate(scene, grid):
    """Validate all declared bands and spectral samples; False outside this film scope."""
    film = descriptor(scene, grid)
    if film is None:
        return False
    for source in getattr(scene, 'sources', ()) or ():
        band = source.band
        wavelength = _finite(band.wavelength, f'source {source.name!r} band wavelength')
        df = _finite(band.df, f'source {source.name!r} band df')
        f0, width = 1/wavelength, band.sigmas*df
        if not math.isfinite(f0+width) or f0 <= width:
            raise ValueError('single-pole-film source band needs a finite positive '
                             'frequency interval; narrow df.')
        low, high = 2*math.pi*(f0-width), 2*math.pi*(f0+width)
        bounds = (('structures[0]', _band_bound(film, low, high)),
                  ('Scene(background=)', math.sqrt(film.host_eps)*high))
        for where, k in bounds:
            if grid.dx*k > 2*math.pi/8:
                raise ValueError(f'single-pole-film source {source.name!r} band: {where} '
                                 f'exceeds the conservative bulk spectral mesh bound '
                                 f'(dx*K={grid.dx*k:.4g}, maximum 2*pi/8). Refine dx to '
                                 f'at most {2*math.pi/(8*k):g}, or narrow the band. '
                                 'The bound includes interior resonances and is sufficient, '
                                 'not a necessary mesh threshold.')
    for w, label in sorted(_points(scene).items()):
        tuple(_check_point(film, grid.dx, w, label))
    return True


def notes(scene, plan, wavelength):
    """Scoped mesh notes, avoiding the ordinary scalar pole's inactive 0/0 endpoint."""
    film = descriptor(scene, plan.grid)
    if film is None:
        return None
    validate(scene, plan.grid)
    points = _points(scene)
    if wavelength is not None:
        points.setdefault(_finite(wavelength, 'diagnostic wavelength'), 'diagnostic')
    if not points and not scene.sources:
        return [(CoarseMeshWarning, 'The single-pole film has no source band or declared '
                 'wavelength to certify its bulk mesh resolution; validate observables '
                 'under refinement.')]
    out = []
    for w, label in sorted(points.items()):
        for where, phase, skin in _check_point(film, plan.dx, w, label):
            if phase < 12 or skin < 2:
                out.append((CoarseMeshWarning, f'{where}: {phase:.3g} cells per phase '
                            f'wavelength and {skin:.3g} per skin depth at wavelength {w:g}; '
                            'the single-pole bulk mesh floor is not an accuracy bound.'))
    return out
