"""Whole-band bulk mesh certificate for the bounded one-Debye film scope.

For x=omega**2, C=delta*(2*eps_inf+delta), passive Debye has
|k|**4=eps_inf**2*x**2+C*x**2/(1+tau**2*x). Its derivative is
2*eps_inf**2*x+C*x*(2+tau**2*x)/(1+tau**2*x)**2 > 0.
Thus the upper frequency endpoint gives the exact maximum bulk |k| over
every positive band. This is a mesh floor, not an accuracy bound or a
certificate for every oblique effective-interface length scale.
"""
import cmath
import math

from wavebit.fdtd.diagnostics import CoarseMeshWarning
from wavebit.materials.time_domain import concrete
from wavebit.fdtd._public_names import public_name
from wavebit.fdtd._prepare.debye_film import descriptor
from wavebit.fdtd._prepare.shape_ranges import _carries


def _finite(value, where):
    result = concrete(value)
    if _carries(value) or result is None or not math.isfinite(result) or result <= 0:
        raise ValueError(f'Debye-film {where} needs a concrete finite positive value; '
                         'its spectral mesh certificate does not support traced bands.')
    return result


def _frequency_k(film, omega):
    """Bulk k at positive angular frequency, without squaring large omega*tau."""
    if not math.isfinite(omega) or omega <= 0:
        raise ValueError('Debye-film frequency exceeds finite positive float64 range; '
                         'rescale the physical units.')
    eps = complex(film.eps)
    if film.delta != 0:
        ratio = omega*film.tau
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError('Debye-film spectral relaxation product exceeds finite '
                             'positive float64 range; rescale the physical units.')
        denominator = math.hypot(1.0, ratio)
        scaled = film.delta/denominator
        eps += complex(scaled/denominator, scaled*(ratio/denominator))
    if not math.isfinite(eps.real) or not math.isfinite(eps.imag) or eps.real <= 0:
        raise ValueError('Debye-film sampled permittivity exceeds finite positive '
                         'float64 range; rescale the physical units.')
    k = omega*cmath.sqrt(eps)
    if not math.isfinite(k.real) or not math.isfinite(k.imag) or k.real <= 0 or k.imag < 0:
        raise ValueError('Debye-film sampled wavenumber exceeds finite positive '
                         'float64 range; rescale the physical units.')
    return k


def _band_bound(film, low, high):
    """Exact bulk modulus maximum over the declared positive angular-frequency band."""
    if not 0 < low <= high < math.inf:
        raise ValueError('Debye-film band needs finite positive angular frequencies.')
    k = _frequency_k(film, high)
    bound = math.hypot(k.real, k.imag)
    if not math.isfinite(bound) or bound <= 0:
        raise ValueError('Debye-film spectral bound exceeds finite positive float64 range; '
                         'rescale the physical units.')
    return bound


def _wavenumber(film, wavelength):
    return _frequency_k(film, 2*math.pi/_finite(wavelength, 'sampled wavelength'))


def index_at(film, wavelength):
    """Real bulk index and metallic skin-depth convention used by existing diagnostics."""
    k = _wavenumber(film, wavelength)
    omega = 2*math.pi/wavelength
    return k.real/omega, 1/k.imag if k.imag > k.real else None


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


def _host_k(film, omega):
    k = math.sqrt(film.host_eps)*omega
    if not math.isfinite(k) or k <= 0:
        raise ValueError('Debye-film host wavenumber exceeds finite positive float64 range; '
                         'rescale the physical units.')
    return k


def _check_point(film, dx, w, label):
    for where, k in (('structures[0]', _wavenumber(film, w)),
                     ('Scene(background=)', complex(_host_k(film, 2*math.pi/w)))):
        # Divide in this order so a tiny, well-resolved k*dx cannot underflow to
        # zero and cause a spurious division error. Infinite cell counts are safe.
        phase_cells = (2*math.pi/k.real)/dx
        skin_cells = (1/k.imag)/dx if k.imag > 0 else math.inf
        if phase_cells < 8 or skin_cells < 1:
            raise ValueError(f'Debye-film {label} asks for wavelength {w:g}, where '
                             f'{where} has {phase_cells:.3g} cells per phase wavelength '
                             f'and {skin_cells:.3g} per skin depth: the required minima '
                             'are 8 and 1. Refine dx or drop this requested sample.')
        yield where, phase_cells, skin_cells


def validate(scene, grid):
    """Validate every declared band and sampled wavelength; False outside this scope."""
    film = descriptor(scene, grid)
    if film is None:
        return False
    for source in getattr(scene, 'sources', ()) or ():
        band = source.band
        wavelength = _finite(band.wavelength, f'source {source.name!r} band wavelength')
        df = _finite(band.df, f'source {source.name!r} band df')
        f0, width = 1/wavelength, band.sigmas*df
        if not math.isfinite(f0+width) or f0 <= width:
            raise ValueError('Debye-film source band needs a finite positive frequency '
                             'interval; narrow df.')
        low, high = 2*math.pi*(f0-width), 2*math.pi*(f0+width)
        bounds = (('structures[0]', _band_bound(film, low, high)),
                  ('Scene(background=)', _host_k(film, high)))
        for where, k in bounds:
            if grid.dx*k > 2*math.pi/8:
                raise ValueError(f'Debye-film source {source.name!r} band: {where} exceeds '
                                 f'the bulk spectral mesh bound (dx*K={grid.dx*k:.4g}, '
                                 f'maximum 2*pi/8). Refine dx to at most '
                                 f'{(2*math.pi/8)/k:g}, or narrow the band. The monotone '
                                 'bulk modulus is largest at the upper band endpoint; '
                                 'this sufficient mesh floor is not an accuracy bound.')
    for w, label in sorted(_points(scene).items()):
        tuple(_check_point(film, grid.dx, w, label))
    return True


def notes(scene, plan, wavelength=None):
    """Scoped phase/skin notes in the existing diagnostic warning format."""
    film = descriptor(scene, plan.grid)
    if film is None:
        return None
    validate(scene, plan.grid)
    points = _points(scene)
    if wavelength is not None:
        points.setdefault(_finite(wavelength, 'diagnostic wavelength'), 'diagnostic')
    if not points and not scene.sources:
        return [(CoarseMeshWarning, 'The Debye film has no source band or declared wavelength '
                 'to certify bulk mesh resolution; validate observables under refinement.')]
    out = []
    for w, label in sorted(points.items()):
        for where, phase, skin in _check_point(film, plan.dx, w, label):
            if phase < 12 or skin < 2:
                out.append((CoarseMeshWarning, f'{where}: {phase:.3g} cells per phase '
                            f'wavelength and {skin:.3g} per skin depth at wavelength {w:g}; '
                            'the Debye bulk mesh floor is not an accuracy bound.'))
    return out
