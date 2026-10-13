"""Sufficient whole-band bulk mesh certificate for a mixed-family film.

For x=omega**2, triangle inequality gives
|k|**2 <= a*xmax + sum_D delta*xmax/sqrt(1+tau**2*xmax)
                       + sum_O S*max_band x/sqrt((d-x)**2+gamma**2*x).
Each Debye summand increases with x. Each oscillator ratio needs both band
endpoints and its possible peak 2*d**2/(2*d-gamma**2). This does not assume
that the total mixed-family modulus is monotone, or that unequal oscillator
resonances have a common maximum. Exact rational candidate comparisons and
verified upward square-root conversion preserve the sufficient direction.

The certificate covers bulk phase/skin scales over every requested positive
source band and explicit point. It does not certify all effective normal
resonances, oblique longitudinal scales, or observable error.
"""
import cmath
from fractions import Fraction
import math

from wavebit.fdtd.diagnostics import CoarseMeshWarning
from wavebit.materials.time_domain import concrete
from wavebit.fdtd._prepare.shape_ranges import _carries
from wavebit.fdtd._prepare.hybrid_film import descriptor


def _range_error():
    return ValueError('hybrid-film spectral values exceed finite float64 range; '
                      'rescale the physical units or use less extreme values.')


def _finite(value, where):
    result = concrete(value)
    if _carries(value) or result is None or not math.isfinite(result) or result <= 0:
        raise ValueError(f'hybrid-film {where} needs a concrete finite positive '
                         'value; its spectral certificate does not support traced bands.')
    return result


def _sqrt_upper(value):
    """Verify a positive float upper bound on sqrt of a positive rational."""
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
    """Analytic oscillator maximum x/|L|, with exact interior-peak decisions."""
    d, damping = Fraction(resonance), Fraction(gamma)
    if damping == 0 and xmin <= d <= xmax:
        raise ValueError('hybrid-film source band contains an active undamped '
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
    """A conservative maximum of bulk |k| on the full positive frequency band."""
    if not (math.isfinite(low) and math.isfinite(high) and 0 < low <= high):
        raise _range_error()
    xmin, xmax = Fraction(low)**2, Fraction(high)**2
    bound_squared = Fraction(film.eps)*xmax
    for delta, tau in zip(film.deltas, film.taus):
        if delta == 0:
            continue
        D, T = Fraction(delta), Fraction(tau)
        # Include strength before sqrt so the bound need not represent an
        # enormous intermediate ratio that a small strength would cancel.
        upper = _sqrt_upper((D*xmax)**2/(1+T*T*xmax))
        bound_squared += Fraction(upper)
    for d, gamma, strength in zip(film.omega0_squared, film.gammas, film.strengths):
        if strength == 0:
            continue
        upper = _ratio_bound(d, gamma, xmin, xmax)
        bound_squared += Fraction(strength)*Fraction(upper)
    return _sqrt_upper(bound_squared)


def _wavenumber(film, wavelength):
    """Actual sampled continuum index; inactive families skip their unused values."""
    try:
        omega = 2*math.pi/wavelength
        if not 0 < omega < math.inf:
            raise _range_error()
        terms = [complex(film.eps)]
        # Fraction evaluation avoids a spurious omega*tau overflow before a
        # finite Debye quotient, and permits exact skipping of delta=0.
        W = Fraction(omega)
        for delta, tau in zip(film.deltas, film.taus):
            if delta == 0:
                continue
            D, product = Fraction(delta), W*Fraction(tau)
            denominator = 1+product*product
            value = complex(float(D/denominator), float(D*product/denominator))
            terms.append(value)
        for d, gamma, strength in zip(film.omega0_squared, film.gammas, film.strengths):
            if strength == 0:
                continue
            real = Fraction(d)-W*W
            loss = Fraction(gamma)*W
            denominator = real*real+loss*loss
            if denominator == 0:
                raise ValueError(f'hybrid-film wavelength {wavelength:g} is an active '
                                 'undamped Lorentz resonance; no finite mesh resolves '
                                 'this sample.')
            S = Fraction(strength)
            terms.append(complex(float(S*real/denominator), float(S*loss/denominator)))
        if any(not math.isfinite(t.real) or not math.isfinite(t.imag) for t in terms):
            raise _range_error()
        eps = complex(math.fsum(t.real for t in terms), math.fsum(t.imag for t in terms))
        k = omega*cmath.sqrt(eps)
    except (OverflowError, ZeroDivisionError) as error:
        raise _range_error() from error
    if not math.isfinite(k.real) or not math.isfinite(k.imag):
        raise _range_error()
    return k


def index_at(film, wavelength):
    """Real index and metallic skin depth for scoped proximity diagnostics."""
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
        if not (math.isfinite(k.real) and math.isfinite(k.imag)):
            raise _range_error()
        # Ordered divisions avoid underflow of k*dx to a zero denominator.
        phase_cells = (2*math.pi/k.real)/dx if k.real > 0 else math.inf
        skin_cells = (1/k.imag)/dx if k.imag > 0 else math.inf
        if phase_cells < 8 or skin_cells < 1:
            raise ValueError(f'hybrid-film {label} asks for wavelength {w:g}, where '
                             f'{where} has {phase_cells:.3g} cells per phase wavelength '
                             f'and {skin_cells:.3g} per skin depth: the required minima '
                             'are 8 and 1. Refine dx or drop this requested sample.')
        yield where, phase_cells, skin_cells


def validate(scene, grid):
    """Certify every source band and explicit point, or False outside the scope."""
    film = descriptor(scene, grid)
    if film is None:
        return False
    for source in getattr(scene, 'sources', ()) or ():
        band = source.band
        wavelength = _finite(band.wavelength, f'source {source.name!r} band wavelength')
        df = _finite(band.df, f'source {source.name!r} band df')
        f0, width = 1/wavelength, band.sigmas*df
        if not math.isfinite(f0+width) or f0 <= width:
            raise ValueError('hybrid-film source band needs a finite positive frequency '
                             'interval; narrow df.')
        low, high = 2*math.pi*(f0-width), 2*math.pi*(f0+width)
        bounds = (('structures[0]', _band_bound(film, low, high)),
                  ('Scene(background=)', _sqrt_upper(Fraction(film.host_eps)*Fraction(high)**2)))
        for where, k in bounds:
            if grid.dx*k > 2*math.pi/8:
                raise ValueError(f'hybrid-film source {source.name!r} band: {where} '
                                 'exceeds the conservative bulk spectral mesh bound '
                                 f'(dx*K={grid.dx*k:.4g}, maximum 2*pi/8). Refine dx to '
                                 f'at most {2*math.pi/(8*k):g}, or narrow the band. '
                                 'Every active Debye term and oscillator interior peak '
                                 'is included; the condition is sufficient, not necessary.')
    for w, label in sorted(_points(scene).items()):
        tuple(_check_point(film, grid.dx, w, label))
    return True


def notes(scene, plan, wavelength):
    """Scoped warning notes without imposing a mixed-family monotonicity assumption."""
    film = descriptor(scene, plan.grid)
    if film is None:
        return None
    validate(scene, plan.grid)
    points = _points(scene)
    if wavelength is not None:
        points.setdefault(_finite(wavelength, 'diagnostic wavelength'), 'diagnostic')
    if not points and not scene.sources:
        return [(CoarseMeshWarning, 'The hybrid film has no source band or declared '
                 'wavelength to certify its bulk mesh resolution; validate '
                 'observables under refinement.')]
    out = []
    for w, label in sorted(points.items()):
        for where, phase, skin in _check_point(film, plan.dx, w, label):
            if phase < 12 or skin < 2:
                out.append((CoarseMeshWarning, f'{where}: {phase:.3g} cells per phase '
                            f'wavelength and {skin:.3g} per skin depth at wavelength {w:g}; '
                            'the mixed-family bulk mesh floor is not an accuracy bound.'))
    return out
