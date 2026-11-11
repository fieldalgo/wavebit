"""Outgoing plane-wave synthesis, using public observation contracts only."""
from dataclasses import field
from operator import index

from wavebit._common.backend import array_namespace, as_array, _concrete, eager_under_torch_compile
from wavebit._common.declarations import Declaration, spec
from wavebit._common.execution import frozen_mapping
from wavebit._common.pytrees import register_dataclass
from wavebit.errors import ValidationError, UnsupportedFeatureError
from wavebit.observations import DiffractionOrders, FieldSamples
from wavebit.units import Units, require_units


@spec
class Plane(Declaration):
    """Cartesian observation grid; distance is outward from the source port plane."""
    x: object
    y: object
    distance: object
    units: Units

    def __post_init__(self):
        require_units(self.units)
        register_dataclass(type(self), static_fields=('units',))


@spec
class PropagationResult(Declaration):
    """Electric phasor samples and the unchanged source provenance."""
    field: FieldSamples
    diagnostics: object = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, 'diagnostics', frozen_mapping(self.diagnostics))
        register_dataclass(type(self))


def _require(flag, path, message, code='input_contract'):
    value = _concrete(flag)
    if value is None:
        raise UnsupportedFeatureError('Abstract validation is unsupported; use eager evaluation/AD',
                                      code='propagation.convention', path=path)
    if not value:
        raise ValidationError(message, code='propagation.' + code, path=path)


@eager_under_torch_compile
def propagate(orders, *, target, method='orders', chunk_size=256):
    """Sum outgoing vector plane waves on ``target`` without a new Maxwell solve.

    The result has axes (wavelength, y, x, xyz), E in V/m. Only calibrated CW,
    z-normal continuum orders in a uniform lossless nonmagnetic medium are
    supported. Eager NumPy/JAX/Torch CPU and eager AD are supported; abstract
    JIT validation is refused. Invalid source amplitudes remain NaN.
    ``chunk_size`` limits forward observation-point intermediates, not AD storage.
    """
    if method != 'orders':
        raise UnsupportedFeatureError('Only direct orders propagation is implemented',
                                      code='propagation.method', path='method')
    if not isinstance(orders, DiffractionOrders) or not isinstance(target, Plane):
        raise ValidationError('Expected DiffractionOrders and Plane',
                              code='propagation.input_contract', path='orders/target')
    try:
        chunk = index(chunk_size)
    except TypeError as exc:
        raise ValidationError('Expected positive integer', code='propagation.sampling', path='chunk_size') from exc
    _require(chunk > 0 and not isinstance(chunk_size, bool), 'chunk_size', 'Expected positive integer', 'sampling')
    r, m = orders.reference, orders.metadata
    for key in ('plane', 'lateral_origin', 'outward_normal', 'epsilon', 'mu', 'length_units', 'wavevector_units'):
        _require(key in r and r[key] is not None, 'reference.' + key, 'Missing propagation reference')
    _require(r.get('wave') in ('reflected', 'transmitted'), 'reference.wave', 'Expected outgoing wave')
    _require(m.get('propagation') == 'continuum' and m.get('time_convention') == 'exp(-i*omega*t)',
             'metadata', 'Expected continuum exp(-i*omega*t)', 'convention')
    _require(r['wavevector_units'] == 'rad/' + r['length_units'], 'reference.wavevector_units', 'Inconsistent units')
    source_units = Units(length=r['length_units'])
    normal = tuple(r['outward_normal'])
    _require(normal in ((0., 0., 1.), (0., 0., -1.)), 'reference.outward_normal', 'Expected z-normal port', 'direction')
    e = orders.field_xyz
    _require(e is not None, 'orders.field_xyz', 'Cartesian CW field in V/m is required', 'normalization')
    _require(orders.k_parallel is not None and orders.k_normal is not None, 'orders', 'Missing wavevectors')
    xp = array_namespace(e, orders.k_parallel, orders.k_normal)
    device = getattr(e, 'device', None)
    _require(device is None or str(device).startswith('cpu') or 'CPU' in str(device).upper(),
             'orders', 'Only CPU is validated')

    def arr(value, dtype=None):
        if isinstance(value, (tuple, list)) and any(hasattr(v, 'dtype') for v in value):
            return xp.stack([arr(v, dtype) for v in value])
        return as_array(xp, value, dtype=dtype, device=device)

    real_dtype = xp.real(e).dtype
    complex_dtype = xp.complex128 if xp.finfo(real_dtype).bits > 32 else xp.complex64
    e = arr(e, complex_dtype)

    def real_array(value, path):
        v = arr(value, complex_dtype)
        _require(xp.all(xp.imag(v) == 0), path, 'Expected real coordinates', 'sampling')
        return xp.real(v)

    x, y, d = (real_array(v, 'target.'+name) for name, v in
               (('x', target.x), ('y', target.y), ('distance', target.distance)))
    for name, v in (('x', x), ('y', y)):
        _require(v.ndim == 1 and v.shape[0] > 0,
                 'target.' + name, 'Expected nonempty one-dimensional coordinates', 'sampling')
        _require(xp.all(xp.isfinite(v)), 'target.' + name, 'Coordinates must be finite', 'sampling')
    _require(d.ndim == 0, 'target.distance', 'Expected scalar distance', 'direction')
    _require(xp.isfinite(d) & (d >= 0), 'target.distance', 'Expected finite outward distance >= 0', 'direction')
    for key in ('epsilon', 'mu'):
        v = arr(r[key], e.dtype)
        _require(v.ndim == 0 or v.shape == (e.shape[0],), 'reference.'+key, 'Expected scalar or wavelength axis')
        _require(xp.all(xp.isfinite(v) & (xp.imag(v) == 0) & (xp.real(v) > 0)), 'reference.'+key, 'Expected lossless positive medium')
        if key == 'mu':
            _require(xp.all(v == 1), 'reference.mu', 'Magnetic media are unsupported')
    kp, kn = arr(orders.k_parallel, complex_dtype), arr(orders.k_normal, complex_dtype)
    _require(xp.all(xp.isfinite(kp) & (xp.imag(kp) == 0)), 'orders.k_parallel', 'Expected finite real transverse wavevectors')
    sign = normal[2]
    _require(xp.all(xp.isfinite(kn) & (xp.real(kn)*sign >= 0) & (xp.imag(kn)*sign >= 0)),
             'orders.k_normal', 'Expected outgoing/decaying wavevectors', 'direction')
    origin = real_array(r['lateral_origin'], 'reference.lateral_origin')
    plane = real_array(r['plane'], 'reference.plane')
    _require(origin.shape == (2,), 'reference.lateral_origin', 'Expected x,y origin')
    _require(plane.ndim == 0 or plane.shape == (e.shape[0],), 'reference.plane', 'Expected scalar or wavelength axis')
    _require(xp.all(xp.isfinite(origin)) & xp.all(xp.isfinite(plane)), 'reference', 'Reference coordinates must be finite')
    # Work in the source length unit; the ratio is dimensionless and preserves AD.
    factor = target.units.length_unit_meters / source_units.length_unit_meters
    xx, yy = xp.meshgrid(x*factor-origin[0], y*factor-origin[1], indexing='xy')
    xx, yy = xp.reshape(xx, (-1,)), xp.reshape(yy, (-1,))
    nw = e.shape[0]
    e = xp.reshape(e, (nw, -1, 3))
    kp, kn = xp.reshape(kp, (nw, -1, 2)), xp.reshape(kn, (nw, -1))
    blocks = []
    for start in range(0, xx.shape[0], chunk):
        phase = (kp[..., 0, None]*xx[None, None, start:start+chunk]
                 + kp[..., 1, None]*yy[None, None, start:start+chunk]
                 + kn[..., None]*(sign*d*factor))
        weights = xp.exp(1j*phase)
        blocks.append(xp.sum(weights[..., None]*e[:, :, None, :], axis=1))
    values = xp.reshape(xp.concat(blocks, axis=1), (nw, y.shape[0], x.shape[0], 3))
    diagnostics = dict(method='orders', source_reference=r, source_metadata=m, chunk_size=chunk,
                       intensity_units='V^2/m^2', power_flux_available=False)
    samples = FieldSamples(coordinates={'x': x, 'y': y, 'z': plane/factor+sign*d,
                                        'wavelength': arr(orders.wavelengths)/factor},
                           values=values, components=('Ex', 'Ey', 'Ez'),
                           units={'Ex': 'V/m', 'Ey': 'V/m', 'Ez': 'V/m', 'coordinates': target.units.length},
                           sampling='phasor', metadata=diagnostics)
    return PropagationResult(field=samples, diagnostics=diagnostics)
