"""Static dispersion tables evaluated as relative constitutive parameters.

Tables are immutable host-side data, not optimizer variables. Evaluation uses
Wavebit's array adapter so wavelength derivatives survive NumPy/JAX/Torch routing.
"""
from dataclasses import dataclass
import sys

import numpy as np

from wavebit._common.backend import _concrete
from wavebit._common.backend import array_namespace
from wavebit._common.backend import as_array
from wavebit.materials.response import Material
from wavebit.materials.response import eps_from_index
from wavebit._common.backend import eager_under_torch_compile

_PYTREE_REGISTERED = False


@eager_under_torch_compile
@dataclass(frozen=True, init=False, repr=False)
class TabulatedMaterial:
    """A wavelength-dependent material, assigned once to a region.

    ``TabulatedMaterial(wavelengths, epsilon, mu=None)`` linearly interpolates
    complex relative permittivity (and optional permeability). Wavelengths must
    be positive, finite, strictly increasing, and use the scene's length unit.
    At least two samples are required. Tables are copied into immutable tuples.
    Each sample may be a scalar, three diagonal values, or a 3x3 tensor;
    tensor entries are interpolated component by component in the same xyz frame.

    ``from_index(wavelengths, index, mu=...)`` instead interpolates isotropic
    index and optional permeability before computing ``epsilon = index**2/mu``.

    ``material.at(wavelength)`` returns a numerical :class:`Material`. RCWA does
    this automatically for shape materials, layer backgrounds and half-spaces,
    including subsequent ``solve(wavelength=...)`` calls and JobSpec sweeps.
    Raster arrays remain evaluated numerical fields.

    Out-of-range evaluation raises eagerly; under JAX tracing it returns NaN.
    Wavelength derivatives are piecewise derivatives of the interpolation; at
    knots the right-hand segment is used (left-hand at the final endpoint).
    Sample values themselves are static, not differentiable design variables.
    This is frequency-domain data, not an FDTD time-domain dispersion model.
    """

    wavelengths: tuple
    values: tuple
    mu: tuple | None
    _quantity: str

    def __init__(self, wavelengths, epsilon, mu=None):
        self._initialize(wavelengths, epsilon, mu, 'epsilon')

    def _initialize(self, wavelengths, values, mu, quantity):
        x = np.asarray(wavelengths)
        if x.ndim != 1 or len(x) < 2 or np.iscomplexobj(x):
            raise ValueError('wavelengths must be a real, one-dimensional table of at least two samples')
        x = np.asarray(x, dtype=float)
        if not np.all(np.isfinite(x)) or np.any(x <= 0) or np.any(np.diff(x) <= 0):
            raise ValueError('wavelengths must be finite, positive and strictly increasing')

        def samples(value, name):
            a = np.asarray(value, dtype=complex)
            shapes = (x.shape,) if quantity == 'index' else (x.shape, x.shape+(3,), x.shape+(3,3))
            if a.shape not in shapes or not np.all(np.isfinite(a)):
                detail = (' for isotropic index conversion' if quantity == 'index' else
                          ', or one three-value diagonal / 3x3 tensor')
                raise ValueError(f'{name} must contain one finite scalar per wavelength' + detail)
            if quantity == 'index' and name == 'mu' and np.any(a == 0):
                raise ValueError('index conversion requires nonzero relative permeability mu')
            def immutable(v):
                return complex(v) if np.ndim(v)==0 else tuple(immutable(e) for e in v)
            return immutable(a)

        if quantity == 'index' and mu is not None and np.ndim(mu) == 0:
            mu = np.full(x.shape, mu, dtype=complex)
        object.__setattr__(self, 'wavelengths', tuple(float(v) for v in x))
        object.__setattr__(self, 'values', samples(values, quantity))
        object.__setattr__(self, 'mu', None if mu is None else samples(mu, 'mu'))
        object.__setattr__(self, '_quantity', quantity)
        self._register_pytree()

    @classmethod
    def from_index(cls, wavelengths, index, *, mu=None):
        """Import an isotropic index table, optionally with relative permeability.

        ``index`` contains one complex scalar per wavelength. ``mu`` may be a
        constant scalar or one complex scalar per wavelength on the same grid.
        When omitted, permeability is one and the material is non-magnetic.
        Index and mu are interpolated separately, then epsilon is computed as
        ``index**2 / mu`` at the requested wavelength. This preserves the
        interpolation convention of published nk tables; interpolating already
        converted epsilon samples would be a different model between samples.

        Permeability samples must be finite and nonzero. If interpolated mu is
        zero, evaluation raises eagerly or produces NaN epsilon under JAX JIT.
        Tensor mu is not supported by this scalar-index conversion; specify
        epsilon and mu tensors directly with the regular constructor instead.
        The evaluated output always carries permittivity and any supplied mu.
        """
        obj = object.__new__(cls)
        obj._initialize(wavelengths, index, mu, 'index')
        return obj

    @property
    def is_magnetic(self):
        return self.mu is not None

    def rotated(self, angle, axis='z'):
        """Rotate every tensor sample by a fixed angle in radians; retain dispersion."""
        Material(1.).rotated(angle, axis)  # validate even when the table is isotropic
        if self._quantity == 'index':
            return self
        values, mus = [], []
        for i, e in enumerate(self.values):
            m = Material(e, None if self.mu is None else self.mu[i]).rotated(angle, axis)
            values.append(m.epsilon)
            if self.mu is not None: mus.append(m.mu)
        return TabulatedMaterial(self.wavelengths, values, None if self.mu is None else mus)

    @property
    def wavelength_range(self):
        """Inclusive lower and upper wavelengths, in the scene's length unit."""
        return self.wavelengths[0], self.wavelengths[-1]

    def __repr__(self):
        return (f'TabulatedMaterial(samples={len(self.wavelengths)}, '
                f'wavelength_range={self.wavelength_range}, '
                f'interpolation={self._quantity!r}, magnetic={self.is_magnetic})')

    def _register_pytree(self):
        global _PYTREE_REGISTERED
        if _PYTREE_REGISTERED:
            return
        jax = sys.modules.get('jax')
        tree = getattr(jax, 'tree_util', None)
        if tree is not None:
            try:
                # No array leaves: the immutable table is static; wavelength is dynamic.
                tree.register_pytree_node(type(self), lambda obj: ((), obj),
                                          lambda obj, children: obj)
            except ValueError:
                pass
            _PYTREE_REGISTERED = True

    def at(self, wavelength):
        """Evaluate a real wavelength or array as numerical ``Material(epsilon, mu)``.

        Output arrays follow the wavelength shape and backend. A vector is useful
        for plotting a spectrum; a solver's region slot still needs a scalar
        evaluated value or the unevaluated model itself.
        """
        try:
            xp = array_namespace(wavelength)
        except TypeError:
            xp = array_namespace(np.asarray(0.))
        dtype = xp.complex64 if str(getattr(wavelength, 'dtype', '')) in (
            'float32', 'torch.float32') else xp.complex128
        device = getattr(wavelength, 'device', None)
        return self._evaluate(wavelength, xp=xp, dtype=dtype, device=device)

    def _evaluate(self, wavelength, *, xp, dtype, device):
        self._register_pytree()
        w = as_array(xp, wavelength, device=device)
        if not xp.isdtype(w.dtype, ('real floating', 'integral')):
            raise ValueError('material evaluation requires real wavelengths')
        real_dtype = xp.float32 if dtype == xp.complex64 else xp.float64
        w = as_array(xp, w, dtype=real_dtype, device=device)
        x = as_array(xp, self.wavelengths, dtype=real_dtype, device=device)
        valid = xp.isfinite(w) & (w >= x[0]) & (w <= x[-1])
        if _concrete(xp.all(valid)) is False:
            raise ValueError(f'wavelength is outside material table range '
                             f'[{self.wavelengths[0]}, {self.wavelengths[-1]}]')
        # Clipped indexing prevents invalid memory access even when a traced input
        # is outside the table. The output is masked to NaN, never silently clamped.
        # Binary search also handles spectra without a wavelengths-by-table
        # comparison matrix. Interval selection is discrete; interpolation below
        # carries the piecewise wavelength derivative.
        index = xp.clip(xp.searchsorted(x, w, side='right') - 1, 0, len(x) - 2)
        fraction = (w - x[index]) / (x[index + 1] - x[index])

        def interpolate(values):
            y = as_array(xp, values, dtype=dtype, device=device)
            tail = (None,) * (y.ndim - 1)
            f = fraction[(...,) + tail]
            value = y[index] + f * (y[index + 1] - y[index])
            return xp.where(valid[(...,) + tail], value, as_array(xp, complex('nan'), dtype=dtype, device=device))

        mu = None if self.mu is None else interpolate(self.mu)
        epsilon = interpolate(self.values)
        if self._quantity == 'index':
            epsilon = eps_from_index(epsilon, mu=mu)
        return Material(epsilon, mu,
                        _wavelength_shape=tuple(w.shape) if w.ndim else None)


def evaluate_material(value, wavelength, *, xp, dtype, device):
    """Resolve only tables; preserve all existing numerical inputs by identity."""
    if isinstance(value, TabulatedMaterial):
        return value._evaluate(wavelength, xp=xp, dtype=dtype, device=device)
    return value


def evaluate_structure(structure, wavelength, *, xp, dtype, device):
    """Copy shape recipes only where a tabulated material must be evaluated."""
    if not isinstance(structure, list):
        return structure

    def entry(item, key):
        if not isinstance(item, dict):
            return item
        value = item.get(key)
        resolved = evaluate_material(value, wavelength, xp=xp, dtype=dtype, device=device)
        out = dict(item, **{key: resolved}) if resolved is not value else item
        if 'shapes' in item:
            shapes = [entry(shape, 'material') for shape in item['shapes']]
            if any(a is not b for a, b in zip(shapes, item['shapes'])):
                out = dict(out, shapes=shapes)
        return out

    layers = [entry(layer, 'background') for layer in structure]
    return layers if any(a is not b for a, b in zip(layers, structure)) else structure
