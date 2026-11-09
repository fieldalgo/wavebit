"""Prescribed monochromatic sources, independent of the layer geometry."""
from dataclasses import dataclass
from dataclasses import replace

import numpy as np
from wavebit._common.backend import eager_under_torch_compile


@eager_under_torch_compile
class Source:
    """Common type for prescribed sources at the calculation's wavelength."""


@eager_under_torch_compile
@dataclass(frozen=True)
class _ExternalSource(Source):
    """Shared direction, polarization and boundary-side inputs."""
    theta: object = None
    phi: object = None
    kx: object = None
    ky: object = None
    polarization: object = None
    psi: object = None
    chi: object = None
    amplitude: object = 1.
    side: str | None = None

    def __post_init__(self):
        if self.side not in (None, 'top', 'bottom'):
            raise ValueError("side must be 'top' or 'bottom'")
        if (self.theta is not None or self.phi is not None) and (self.kx is not None or self.ky is not None):
            raise ValueError('A PlaneWave takes theta/phi or kx/ky, not both.')
        if self.polarization is not None and (self.psi is not None or self.chi is not None):
            raise ValueError('A PlaneWave takes polarization or psi/chi, not both.')


@eager_under_torch_compile
@dataclass(frozen=True)
class PlaneWave(_ExternalSource):
    """One incident plane wave, referenced at x=y=0 on its outside boundary.

    Angles are radians, kx/ky are inverse scene lengths, and polarization is an
    electric (s,p) pair. The complex amplitude includes its relative phase.
    An omitted side means top in a source list, or the containing dictionary key.
    """


@eager_under_torch_compile
@dataclass(frozen=True)
class OrderExcitation(_ExternalSource):
    """Plane waves specified by order weights around a reference direction.

    Each weight is its order's amplitude at x=y=0 on the outside boundary: the same
    weights the ordinary ``excitation=`` keyword takes, and the same light (since
    2026-09-27, when the ordinary path moved its phase reference to the cell centre).
    """
    orders: object = None


# Read old programs while keeping new plane-wave objects unambiguously singular.
Beam = OrderExcitation


@eager_under_torch_compile
@dataclass(frozen=True)
class GaussianBeam(_ExternalSource):
    """Forward propagating Gaussian angular spectrum in a lossless exterior.

    waist is the scalar envelope's 1/e amplitude radius, in scene lengths.
    focus uses global xyz coordinates. At the focus, amplitude is the norm of
    the complex electric vector (V/m when a physical field scale is assigned).
    sampling and spectral_extent control midpoint quadrature in the two
    directions perpendicular to the central propagation direction. The
    spectrum is exp(-waist**2 * kt**2/4), projected transverse to each k.
    Only propagating components directed into the stack are retained.
    """
    waist: float = 1.
    focus: tuple = (0., 0., 0.)
    sampling: tuple = (9, 9)
    spectral_extent: float = 4.

    def __post_init__(self):
        super().__post_init__()
        _positive(self.waist, 'waist')
        _positive(self.spectral_extent, 'spectral_extent')
        _vector(self.focus, 3, 'focus', real=True)
        _grid(self.sampling)


@eager_under_torch_compile
@dataclass(frozen=True)
class ElectricDipole(Source):
    """Prescribed electric dipole moment in C m, with exp(-i omega t).

    position is global xyz in scene lengths. A uniform host layer is required.
    repetition='periodic' uses bloch_k, in inverse scene lengths, for the phase
    between copies; bloch_k + G (G reciprocal) is the same array, and the solve
    reduces bloch_k into the reciprocal cell centred on the origin. 'isolated' uses Brillouin-zone midpoint quadrature; sampling
    sets its grid. Finite quadrature approximates an isolated source by a larger
    repeated supercell and must be converged. rcwa requires length_unit (metres
    per scene length) when a dipole is present.
    """
    position: tuple
    moment: tuple
    repetition: str = 'periodic'
    bloch_k: tuple | None = None
    sampling: tuple = (5, 5)

    def __post_init__(self):
        _vector(self.position, 3, 'position', real=True)
        _vector(self.moment, 3, 'moment')
        if self.repetition not in ('periodic', 'isolated'):
            raise ValueError("repetition must be 'periodic' or 'isolated'")
        if self.bloch_k is not None:
            _vector(self.bloch_k, 2, 'bloch_k', real=True)
            if self.repetition == 'isolated':
                raise ValueError('An isolated dipole integrates Bloch wavevectors; omit bloch_k.')
        _grid(self.sampling)


def _positive(value, name):
    # A 0-d array or tensor is one number; a grad tensor or an eager JAX tracer is read through
    # its value, and one under jit or vmap is judged by its shape and dtype alone.
    if _carries_derivative(value) or getattr(value, 'shape', None) == ():
        a = _readable(value)
        if a is None:
            if getattr(value, 'shape', None) == () and np.dtype(value.dtype).kind in 'iuf':
                return
            raise ValueError(f'{name} must be a positive finite real number')
        value = a[()] if getattr(a, 'shape', None) == () else a
    if not np.isscalar(value) or not np.isrealobj(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be a positive finite real number')


def _vector(value, size, name, real=False):
    message = f'{name} must contain {size} finite {"real " if real else ""}components'
    if _carries_derivative(value):
        # Validation must not detach the caller's graph: each leaf is read through its value
        # (a grad tensor detached, an eager JAX tracer through its primal). A value under jit or
        # vmap has none yet and is judged by its shape and dtype alone.
        leaves = list(value) if isinstance(value, (list, tuple)) else None
        if leaves is None and getattr(value, 'ndim', None) != 1:
            raise ValueError(message)
        read = [_readable(v) for v in (leaves if leaves is not None else [value])]
        if any(a is None for a in read):
            items = leaves if leaves is not None else [value]
            shapes = [tuple(getattr(v, 'shape', ())) for v in items]
            count = len(items) if leaves is not None else (shapes[0] or (0,))[0]
            if (count != size or any(len(s) != (0 if leaves is not None else 1) for s in shapes)
                    or real and any(_complex_dtype(v) for v in items)):
                raise ValueError(message)
            return
        a = np.asarray(read if leaves is not None else read[0])
    else:
        a = np.asarray(value)
    if a.shape != (size,) or not np.all(np.isfinite(a)) or (real and np.any(a.imag)):
        raise ValueError(message)


def _carries_derivative(value):
    """Whether ``value`` is, or holds, a JAX tracer or a PyTorch tensor that requires grad."""
    import sys
    jax = sys.modules.get('jax')
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, (list, tuple)):
            stack.extend(item)
        elif jax is not None and isinstance(item, jax.core.Tracer):
            return True
        elif bool(getattr(item, 'requires_grad', False)):
            return True
    return False


def _readable(value):
    """``value`` as NumPy for a check, without touching its graph; None under jit or vmap."""
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    import sys
    jax = sys.modules.get('jax')
    if jax is not None and isinstance(value, jax.core.Tracer):
        try:
            return np.asarray(jax.lax.stop_gradient(value))
        except (jax.errors.TracerArrayConversionError, jax.errors.ConcretizationTypeError):
            return None
    return np.asarray(value)


def _complex_dtype(value):
    dtype = getattr(value, 'dtype', None)
    return dtype is not None and 'complex' in str(dtype)


def _grid(value):
    a = np.asarray(value)
    if a.shape != (2,) or a.dtype.kind not in 'iu' or np.any(a < 1):
        raise ValueError('sampling must contain two positive integers')


def source_list(value):
    """Resolve list inputs and boundary dictionaries into typed source objects."""
    if isinstance(value, Source):
        value = [value]
    if isinstance(value, (list, tuple)):
        if not value or not all(isinstance(v, (PlaneWave, OrderExcitation, GaussianBeam, ElectricDipole)) for v in value):
            raise TypeError('excitation must be a nonempty list of source objects')
        return [replace(v, side=v.side or 'top') if isinstance(v, _ExternalSource) else v for v in value]
    if not isinstance(value, dict) or not value or set(value) - {'top', 'bottom'}:
        raise ValueError("excitation must be sources or a 'top'/'bottom' dictionary")
    result = []
    for side, entries in value.items():
        entries = entries if isinstance(entries, (list, tuple)) else [entries]
        if not entries:
            raise ValueError(f'excitation[{side!r}] is empty')
        for entry in entries:
            if isinstance(entry, dict):
                entry = (OrderExcitation(orders=entry) if entry and all(
                    isinstance(k, (int, np.integer, tuple)) for k in entry) else OrderExcitation(**entry))
            if not isinstance(entry, _ExternalSource):
                raise TypeError('Boundary dictionaries accept plane waves and Gaussian beams; '
                                'put internal sources in an excitation list.')
            if entry.side is not None and entry.side != side:
                raise ValueError('Source side conflicts with its excitation dictionary key')
            result.append(replace(entry, side=side))
    return result


def is_source_excitation(value):
    return (isinstance(value, Source) or
            isinstance(value, (list, tuple)) and any(isinstance(v, Source) for v in value) or
            isinstance(value, dict) and any(k in ('top', 'bottom') for k in value))
