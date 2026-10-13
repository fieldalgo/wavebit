"""JAX differentiation instrument for the unaccepted harmonic-energy candidate.

This evaluates the same finite model as nvf_harmonic_energy, with implicit
derivatives of its cell solve. It is not a Wavebit backend or NVF dispatch.
Only positive real, disjoint periodic rectangle fixtures are supported. Shape
parameters are nanometers; order, inner and grid are fixed before tracing.
Enable JAX x64 before constructing a model. Nonconverged forward or adjoint
linear solves produce NaN, including under jit, rather than plausible data.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
import argparse
import json
import time
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.sparse.linalg import cg
from scipy.fft import next_fast_len

from studies.rcwa.nvf_harmonic_energy.nvf_harmonic_energy import PERIOD, powers


def _checked_solve(action, right, maxiter):
    """Check the true residual of each primal/transpose solve inside the JIT.

    JAX cg's info is currently None even when maxiter is exhausted. The outer
    custom_linear_solve uses this callback for both directions; gradients do not
    differentiate the iterations or the success predicate. A complex Hermitian
    operator need not equal its transpose, so symmetric=True must not be used.
    """
    value, _ = cg(action, right, tol=1e-12, atol=1e-14, maxiter=maxiter)
    residual = jnp.linalg.norm(action(value)-right)/jnp.maximum(jnp.linalg.norm(right), 1.)
    valid = jnp.isfinite(residual) & (residual < 1e-10)
    return jnp.where(valid, value, jnp.nan)


def _implicit_solve(action, right, maxiter):
    def solve(matvec, rhs):
        return _checked_solve(matvec, rhs, maxiter)
    return jax.lax.custom_linear_solve(action, right, solve=solve,
                                       transpose_solve=solve, symmetric=False)


def _modes(cap):
    return np.meshgrid(np.arange(-cap, cap+1), np.arange(-cap, cap+1), indexing='ij')


def material_coefficients(cap, shapes):
    """Exact material transform, retaining derivatives at sinc's zero argument."""
    n, m = _modes(cap)
    coefficients = jnp.asarray((m == 0) & (n == 0), dtype=jnp.complex128)
    for shape in shapes:
        width, height = shape['lx']/PERIOD, shape['ly']/PERIOD
        epsilon = shape['epsilon']
        angle = shape.get('angle', 0.)
        cx, cy = shape.get('cx', 0.), shape.get('cy', 0.)
        parameters = jnp.array([width, height, epsilon, angle, cx, cy])
        valid = jnp.all(jnp.isfinite(parameters)) & jnp.all(parameters[:3] > 0)
        u = m*jnp.cos(angle)+n*jnp.sin(angle)
        v = -m*jnp.sin(angle)+n*jnp.cos(angle)
        increment = ((epsilon-1)*width*height*jnp.sinc(width*u)*jnp.sinc(height*v)
                     *jnp.exp(-2j*jnp.pi*(m*cx+n*cy)/PERIOD))
        coefficients = coefficients+jnp.where(valid, increment, jnp.nan)
    return coefficients


def sample(coefficients, grid):
    cap = (coefficients.shape[-1]-1)//2
    n, m = _modes(cap)
    array = jnp.zeros((*coefficients.shape[:-2], grid, grid), dtype=jnp.complex128)
    array = array.at[..., n % grid, m % grid].set(
        coefficients*jnp.exp(2j*jnp.pi*(m+n)*(.5/grid-.5)))
    values = jnp.fft.ifft2(array)*grid**2
    valid = jnp.max(jnp.abs(values.imag)) < 1e-8
    return jnp.where(valid, values.real, jnp.nan)


class HarmonicEnergyModel:
    """Fixed finite discretization; shapes remain live JAX pytree parameters.

    Call jax.jit(model.transmission) or jit(value_and_grad(objective)). Geometry
    must stay in the documented nonoverlapping domain. A gap derivative at zero
    is interpreted from the positive-gap side, not by overlapping the images.
    This study has no painted-geometry, loss, magnetic or higher-derivative API.
    """

    def __init__(self, order, inner, grid=None, maxiter=300, *, polarization='p'):
        if not jax.config.x64_enabled:
            raise ValueError('the harmonic-energy study requires JAX x64')
        if not isinstance(order, int) or order < 0:
            raise ValueError('outer order must be a nonnegative integer')
        if not isinstance(inner, int) or inner < 1:
            raise ValueError('inner order must be a positive integer')
        if not isinstance(maxiter, int) or maxiter < 1:
            raise ValueError('maxiter must be a positive integer')
        if polarization not in ('s', 'p'):
            raise ValueError('polarization must be s or p')
        self.polarization = polarization
        self.order, self.inner, self.maxiter = order, inner, maxiter
        minimum = 4*(inner+order)
        self.grid = next_fast_len(minimum+1) if grid is None else grid
        if not isinstance(self.grid, int) or self.grid <= minimum:
            raise ValueError('grid must exceed 4*(inner+outer order) to prevent aliasing')
        self.fft_size = next_fast_len(6*inner+1)
        n, m = _modes(inner)
        norm = np.sqrt(m*m+n*n)
        norm[inner, inner] = 1
        self.gradient = jnp.asarray([1j*m/norm, 1j*n/norm])
        n, m = _modes(order)
        self.count = m.size
        self.dm = m.ravel()[:, None]-m.ravel()
        self.dn = n.ravel()[:, None]-n.ravel()

    def corrector(self, shapes):
        inner, size = self.inner, self.fft_size
        material = material_coefficients(2*inner, shapes)
        material_fft = jnp.fft.fft2(material, s=(size, size))
        middle = material[inner:3*inner+1, inner:3*inner+1]
        gradient = self.gradient

        def action(value):
            flux = jnp.fft.ifft2(material_fft*jnp.fft.fft2(
                gradient*value, s=(size, size)))[..., 2*inner:4*inner+1, 2*inner:4*inner+1]
            result = jnp.sum(jnp.conj(gradient)*flux, axis=0)
            return result.at[inner, inner].set(value[inner, inner])

        columns, residuals = [], []
        for j in range(2):
            right = gradient[j]*middle
            potential = _implicit_solve(action, right, self.maxiter)
            residuals.append(jnp.linalg.norm(action(potential)-right)
                             /jnp.maximum(jnp.linalg.norm(right), 1.))
            column = gradient*potential
            columns.append(column.at[j, inner, inner].add(1.))
        return jnp.stack(columns, axis=1), jnp.stack(residuals)

    def matrices(self, shapes):
        coefficients, residuals = self.corrector(shapes)
        grid, count = self.grid, self.count
        basis = sample(coefficients, grid)
        epsilon = sample(material_coefficients((grid-1)//2, shapes), grid)

        def convolution(value):
            transformed = jnp.fft.fft2(value)/grid**2
            return (transformed[self.dn % grid, self.dm % grid]
                    *jnp.exp(1j*jnp.pi*(self.dm+self.dn)*(1-1/grid)))

        change = jnp.block([[convolution(basis[i, j]) for j in range(2)] for i in range(2)])
        gram = jnp.block([[convolution(epsilon*jnp.sum(basis[:, i]*basis[:, j], axis=0))
                           for j in range(2)] for i in range(2)])
        inverse = jnp.linalg.solve(change, jnp.eye(2*count))
        effective = inverse.conj().T@gram@inverse
        hermiticity = jnp.max(jnp.abs(effective-effective.conj().T))
        effective = jnp.where(hermiticity < 1e-9, effective, jnp.nan)
        slots = (effective[:count, :count], effective[count:, count:],
                 jnp.linalg.inv(convolution(epsilon)),
                 -effective[:count, count:], -effective[count:, :count])
        return slots, dict(cell_residuals=residuals, hermiticity=hermiticity)

    def transmission(self, shapes):
        slots, metadata = self.matrices(shapes)
        t, r = powers(self.order, slots, jnp, polarization=self.polarization)
        return dict(T=t, R=r, energy_error=jnp.abs(t+r-1), **metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inners', type=int, nargs='+', default=[31, 63, 127])
    parser.add_argument('--order', type=int, default=7)
    parser.add_argument('--polarizations', choices=['s', 'p'], nargs='+', default=['p'])
    args = parser.parse_args()
    jax.config.update('jax_enable_x64', True)
    from studies.rcwa.nvf_harmonic_energy.nvf_harmonic_energy import transmission

    for polarization, inner in product(args.polarizations, args.inners):
        start = time.monotonic()
        model = HarmonicEnergyModel(args.order, inner, polarization=polarization)
        shapes = lambda gap: [dict(lx=350., ly=700.-gap, epsilon=12.25)]

        def objective(gap):
            result = model.transmission(shapes(gap))
            return result['T'], result

        (_, result), derivative = jax.jit(jax.value_and_grad(objective, has_aux=True))(0.)
        result = {key: np.asarray(value).tolist() for key, value in result.items()}
        derivative = float(derivative)
        closed = transmission(args.order, inner, shapes(0.), polarization=polarization)['T']
        steps = [1e-4, 1e-6]
        secants = [(transmission(args.order, inner, shapes(step), polarization=polarization)['T']-closed)/step
                   for step in steps]
        if not np.isfinite([result['T'], result['R'], derivative, *secants]).all():
            raise FloatingPointError('failed forward or adjoint solve is not comparison data')
        print(json.dumps(dict(kind='gap_autodiff', polarization=polarization,
                              order=args.order, inner=inner,
                              **result, gap_derivative=derivative, numpy_T=closed,
                              steps=steps, numpy_secants=secants,
                              seconds=time.monotonic()-start)), flush=True)
        jax.clear_caches()


if __name__ == '__main__':
    main()
