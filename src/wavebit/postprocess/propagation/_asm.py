"""Scalar angular-spectrum propagation of calibrated, one-way sampled fields."""
from dataclasses import field
from operator import index
import math

from wavebit._common.backend import array_namespace, as_array, _concrete, eager_under_torch_compile
from wavebit._common.conventions import boundary_kz
from wavebit._common.declarations import Declaration, spec
from wavebit._common.execution import frozen_mapping
from wavebit._common.pytrees import register_dataclass
from wavebit.errors import ValidationError
from wavebit.observations import FieldSamples, RawFieldSpectrum, TimeSeries
from wavebit.units import Units, require_units
from .api import Plane, PropagationResult, _require


@spec
class ASM(Declaration):
    """Padding on each side in (y, x) cells; None uses ceil(N/2) for isolated data."""
    padding: object = None

    def __post_init__(self):
        if self.padding is not None:
            try:
                p = tuple(index(v) for v in self.padding)
            except (TypeError, ValueError) as exc:
                raise ValidationError('Expected two nonnegative integers',
                                      code='propagation.sampling', path='padding') from exc
            _require(len(p) == 2 and all(v >= 0 for v in p) and
                     not any(isinstance(v, bool) for v in self.padding),
                     'padding', 'Expected two nonnegative integers', 'sampling')
            object.__setattr__(self, 'padding', p)
        register_dataclass(type(self), static_fields=('padding',))


@spec
class PhasorPlane(Declaration):
    """One scalar electric component on a uniform Cartesian plane, in V/m."""
    values: object
    x: object
    y: object
    z: object
    wavelength: object
    units: Units
    medium_n: object
    component: str
    direction: str
    boundary: str
    carrier_kxy: tuple = (0., 0.)
    metadata: object = field(default_factory=dict)

    def __post_init__(self):
        require_units(self.units)
        object.__setattr__(self, 'metadata', frozen_mapping(self.metadata))
        register_dataclass(type(self), static_fields=('units', 'component', 'direction', 'boundary'))


def from_phasor(samples, *, component, medium_n, direction, boundary,
                wavelength_index=0, carrier_kxy=(0., 0.), assume_outgoing=False):
    """Select a calibrated component; explicitly assert a one-way exterior field.

    Accepts public RCWA/FDTD collocated phasors, never raw Fourier spectra or
    time series. Selecting a direction cannot separate incident/reflected fields.
    """
    _require(isinstance(samples, FieldSamples) and not isinstance(samples, (RawFieldSpectrum, TimeSeries)),
             'samples', 'Expected calibrated FieldSamples, not a raw spectrum', 'normalization')
    _require(assume_outgoing is True, 'assume_outgoing',
             'Explicitly assert that the selected field is one-way in the chosen exterior', 'direction')
    _require(samples.sampling in ('phasor', 'collocated phasor'),
             'samples.sampling', 'Expected collocated calibrated phasor', 'normalization')
    _require(component in ('Ex', 'Ey', 'Ez') and component in samples.components,
             'component', 'Expected an available electric component')
    _require(samples.units.get(component) == 'V/m',
             'samples.units', 'Expected physical electric field in V/m', 'normalization')
    if samples.sampling == 'collocated phasor':
        _require(bool(samples.metadata.get('normalization')), 'samples.metadata.normalization',
                 'Collocated FDTD phasors need source calibration provenance', 'normalization')
    _require('coordinates' in samples.units, 'samples.units.coordinates', 'Missing length unit')
    coords = samples.coordinates
    for key in ('x', 'y', 'z', 'wavelength'):
        _require(key in coords, 'samples.coordinates.'+key, 'Missing plane coordinate')
    v = samples.values
    _require(v.ndim == 4 and v.shape[-1] == len(samples.components),
             'samples.values', 'Expected (wavelength,y,x,component)')
    try:
        wi = index(wavelength_index)
    except TypeError as exc:
        raise ValidationError('Expected wavelength index', code='propagation.input_contract',
                              path='wavelength_index') from exc
    _require(0 <= wi < v.shape[0], 'wavelength_index', 'Index outside wavelength axis')
    xp = array_namespace(v)
    device = getattr(v, 'device', None)
    wavelengths = as_array(xp, coords['wavelength'], device=device)
    _require(wavelengths.shape == (v.shape[0],), 'samples.coordinates.wavelength',
             'Wavelength coordinates must match the field axis')
    z = as_array(xp, coords['z'], device=device)
    if z.ndim == 1:
        _require(z.shape in ((1,), (v.shape[0],)), 'samples.coordinates.z', 'Expected one plane per wavelength')
        z = z[0 if z.shape[0] == 1 else wi]
    plane = PhasorPlane(values=v[wi, :, :, samples.components.index(component)],
        x=coords['x'], y=coords['y'], z=z, wavelength=wavelengths[wi],
        units=Units(length=samples.units['coordinates']), medium_n=medium_n,
        component=component, direction=direction, boundary=boundary, carrier_kxy=carrier_kxy,
        metadata={'source_metadata': samples.metadata, 'assume_outgoing': True,
                  'continuum_approximation': True, 'wavelength_index': wi})
    # Validate the physical/grid contract at the adapter boundary.
    plan_asm(plane, numerics=ASM(padding=(0, 0)))
    return plane


def _array(xp, value, like, dtype=None):
    return as_array(xp, value, dtype=dtype, device=getattr(like, 'device', None))


def _scalar(xp, value, like, path, *, positive=False, nonnegative=False):
    v = _array(xp, value, like, like.dtype)
    _require(v.ndim == 0, path, 'Expected scalar')
    real = xp.real(v)
    valid = xp.isfinite(v) & (xp.imag(v) == 0)
    if positive:
        valid = valid & (real > 0)
    if nonnegative:
        valid = valid & (real >= 0)
    if _concrete(valid) is False:
        raise ValidationError('Invalid real scalar or physical range',
                              code='propagation.input_contract', path=path)
    return real, valid


def _pad(xp, v, py, px):
    if px:
        zero = xp.zeros((v.shape[0], px), dtype=v.dtype, device=getattr(v, 'device', None))
        v = xp.concat((zero, v, zero), axis=1)
    if py:
        zero = xp.zeros((py, v.shape[1]), dtype=v.dtype, device=getattr(v, 'device', None))
        v = xp.concat((zero, v, zero), axis=0)
    return v


@spec
class ASMPlan(Declaration):
    """Validated grid reused under JAX jit; numerical invalidity is explicit.

    Eager invalid parameters raise. Abstract invalid parameters produce NaNs and
    diagnostics['input_valid']=False, so JIT never silently skips admission.
    """
    plane: PhasorPlane
    transverse_squared: object
    carrier_phase: object
    padding: tuple

    def __post_init__(self):
        register_dataclass(type(self), static_fields=('padding',))

    @eager_under_torch_compile
    def __call__(self, distance, *, values=None, wavelength=None, medium_n=None):
        p = self.plane
        xp = array_namespace(p.values)
        u = p.values if values is None else values
        _require(getattr(u, 'shape', None) == p.values.shape, 'values', 'Shape must match the validated grid')
        _require(array_namespace(u).__name__ == xp.__name__, 'values', 'Backend must match the plan')
        _require(u.dtype == p.values.dtype and (getattr(u, 'device', None) is None or getattr(u, 'device', None) == getattr(p.values, 'device', None)),
                 'values', 'Dtype/device must match the plan')
        d, vd = _scalar(xp, distance, u, 'distance', nonnegative=True)
        wl, vw = _scalar(xp, p.wavelength if wavelength is None else wavelength, u, 'wavelength', positive=True)
        n, vn = _scalar(xp, p.medium_n if medium_n is None else medium_n, u, 'medium_n', positive=True)
        valid = vd & vw & vn & xp.all(xp.isfinite(u))
        # Safe operands also keep rejected abstract inputs from overflowing before masking.
        d = xp.where(vd, d, xp.zeros_like(d))
        wl = xp.where(vw, wl, xp.ones_like(wl))
        n = xp.where(vn, n, xp.ones_like(n))
        k0 = 2*math.pi/wl
        radicand = n*n-self.transverse_squared/(k0*k0)
        # Forward values at cutoff are defined, but wavelength/index derivatives are not.
        away = xp.all(xp.abs(radicand) > 32*xp.finfo(xp.real(u).dtype).eps)
        kz = k0*boundary_kz(xp, _array(xp, radicand, u, u.dtype), xp.zeros_like(radicand))
        py, px = self.padding
        envelope = u*xp.conj(self.carrier_phase)
        spectrum = xp.fft.fft2(_pad(xp, envelope, py, px))
        propagated = xp.fft.ifft2(spectrum*xp.exp(1j*kz*d))
        out = propagated[py:py+u.shape[0], px:px+u.shape[1]]*self.carrier_phase
        out = xp.where(valid, out, xp.full_like(out, float('nan')))
        sign = 1 if p.direction == '+z' else -1
        diagnostics = {'method': 'asm', 'input_valid': valid, 'parameter_gradient_defined': away, 'padding': self.padding,
            'boundary': p.boundary, 'carrier_kxy': p.carrier_kxy,
            'continuum_approximation': True, 'source_metadata': p.metadata,
            'power_flux_available': False, 'intensity_units': 'V^2/m^2'}
        field_samples = FieldSamples(values=out[None, ..., None], components=(p.component,),
            coordinates={'x': p.x, 'y': p.y, 'z': p.z+sign*d, 'wavelength': wl[None]},
            units={p.component: 'V/m', 'coordinates': p.units.length},
            sampling='phasor', metadata=diagnostics)
        return PropagationResult(field=field_samples, diagnostics=diagnostics)


def plan_asm(plane, *, numerics=None):
    """Validate a fixed grid eagerly and prepare unshifted FFT wavevectors."""
    _require(isinstance(plane, PhasorPlane), 'plane', 'Expected PhasorPlane')
    numerics = ASM() if numerics is None else numerics
    _require(isinstance(numerics, ASM), 'numerics', 'Expected ASM')
    p = plane
    _require(p.direction in ('+z', '-z'), 'direction', 'Expected +z or -z', 'direction')
    _require(p.boundary in ('isolated', 'periodic'), 'boundary', 'Expected isolated or periodic', 'sampling')
    _require(p.component in ('Ex', 'Ey', 'Ez'), 'component', 'Expected electric component')
    u = p.values
    _require(getattr(u, 'ndim', None) == 2, 'values', 'Expected (y,x) scalar plane')
    xp = array_namespace(u)
    _require(xp.isdtype(u.dtype, 'complex floating'), 'values', 'Expected complex phasor values')
    coordinates, spacings = [], []
    for name, size in (('x', u.shape[1]), ('y', u.shape[0])):
        v = _array(xp, getattr(p, name), u, u.dtype)
        _require(v.ndim == 1 and v.shape[0] == size and size > 0, name, 'Coordinate/field shape mismatch')
        _require(xp.all(xp.isfinite(v) & (xp.imag(v) == 0)), name, 'Expected finite real coordinates')
        v = xp.real(v)
        step = v[1]-v[0] if size > 1 else xp.ones_like(v[0])
        if size > 1:
            delta = v[1:]-v[:-1]
            tolerance = 64*xp.finfo(v.dtype).eps*xp.maximum(xp.abs(step), xp.max(xp.abs(v)))
            _require(xp.all((delta > 0) & (xp.abs(delta-step) <= tolerance)), name,
                     'ASM needs an increasing uniform grid', 'sampling')
        coordinates.append(v)
        spacings.append(step)
    for name in ('wavelength', 'medium_n', 'z'):
        _, valid = _scalar(xp, getattr(p, name), u, name, positive=name != 'z')
        _require(valid, name, 'Prepare parameters must be concrete and valid')
    carrier = _array(xp, p.carrier_kxy, u, u.dtype)
    _require(carrier.shape == (2,), 'carrier_kxy', 'Expected two transverse wavevectors')
    _require(xp.all(xp.isfinite(carrier) & (xp.imag(carrier) == 0)), 'carrier_kxy', 'Expected real carrier')
    carrier = xp.real(carrier)
    ny, nx = u.shape
    padding = numerics.padding
    if padding is None:
        padding = ((ny+1)//2 if ny > 1 else 0, (nx+1)//2 if nx > 1 else 0) if p.boundary == 'isolated' else (0, 0)
    _require(p.boundary != 'periodic' or padding == (0, 0), 'padding',
             'Zero padding changes a periodic field; use isolated explicitly', 'sampling')
    _require(not (ny == 1 and padding[0]) and not (nx == 1 and padding[1]), 'padding',
             'Do not pad an invariant singleton axis', 'sampling')
    py, px = padding
    dtype = xp.real(u).dtype
    # fftfreq uses integer indices; dx/dy remain backend scalars with no host conversion.
    def frequencies(count, spacing):
        j = xp.arange(count, dtype=dtype, device=getattr(u, 'device', None))
        j = xp.where(j < (count+1)//2, j, j-count)
        return 2*math.pi*j/(count*spacing)
    kx = frequencies(nx+2*px, spacings[0])+carrier[0]
    ky = frequencies(ny+2*py, spacings[1])+carrier[1]
    ksq = ky[:, None]**2+kx[None, :]**2
    x, y = coordinates
    phase = xp.exp(1j*(carrier[0]*x[None, :]+carrier[1]*y[:, None]))
    canonical = p.replace(x=x, y=y, z=_scalar(xp, p.z, u, 'z')[0])
    return ASMPlan(plane=canonical, transverse_squared=ksq, carrier_phase=phase, padding=padding)


def propagate_asm(plane, target, numerics):
    _require(isinstance(target, Plane), 'target', 'Expected Plane')
    plan = plan_asm(plane, numerics=numerics)
    xp, u = array_namespace(plane.values), plane.values
    factor = target.units.length_unit_meters/plane.units.length_unit_meters
    for name in ('x', 'y'):
        v = _array(xp, getattr(target, name), u, u.dtype)*factor
        expected = getattr(plan.plane, name)
        _require(v.shape == expected.shape, 'target.'+name, 'ASM keeps the source observation grid')
        _require(xp.all(xp.isfinite(v) & (xp.imag(v) == 0)), 'target.'+name,
                 'Expected finite real coordinates', 'sampling')
        _require(xp.all(xp.abs(v-expected) <= 64*xp.finfo(expected.dtype).eps*xp.maximum(xp.abs(expected), xp.ones_like(expected))),
                 'target.'+name, 'ASM does not interpolate/resample the observation grid', 'sampling')
    result = plan(target.distance*factor)
    c = {key: value/factor for key, value in result.field.coordinates.items()}
    return result.replace(field=result.field.replace(coordinates=c,
        units={plane.component: 'V/m', 'coordinates': target.units.length}))
