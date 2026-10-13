"""Material-defined energy factorization candidate; not a production NVF method.

Two periodic electrostatic correctors give B = I + grad(chi). The positive
electric-energy Gram matrix is expressed in the retained Fourier E coordinates:
    effective_epsilon = inv(C(B)).H @ C(B.H epsilon B) @ inv(C(B)).
This preserves lossless Hermiticity and depends on the material function rather
than primitive seams. Its inner corrector truncation is still consequential.
It has no production or interface-exactness acceptance contract. An opt-in JAX
study in nvf_harmonic_energy_jax checks derivatives of this finite model.

Only the documented, nonoverlapping rectangular fixtures are supported here.
All lengths supplied in shapes are nanometers, in centered cell coordinates.
"""
import argparse
import json
import time

import numpy as np
from scipy.fft import fft2, ifft2, next_fast_len
from scipy.linalg import solve
from scipy.sparse.linalg import LinearOperator, cg

from wavebit._core.solve_2d import layer_modes_2d
from wavebit._core.modal_gradient import exponential

from studies.nvf_fourier_orders import order_grid

PERIOD = 700.
WAVELENGTH = 900.
THICKNESS = 300.
THETA = .2


def tilted():
    return [dict(lx=640., ly=700./np.sqrt(5), angle=np.arctan2(1, 2), epsilon=12.25)]


def material_coefficients(cap, shapes):
    """Exact centered Fourier transform of disjoint periodic rectangles.

    The supplied fixtures touch but do not overlap their periodic images.
    This additive formula is not a general periodic painting implementation.
    """
    n, m = np.meshgrid(np.arange(-cap, cap+1), np.arange(-cap, cap+1), indexing='ij')
    coefficients = ((m == 0) & (n == 0)).astype(complex)
    for shape in shapes:
        width, height = shape['lx']/PERIOD, shape['ly']/PERIOD
        epsilon = shape['epsilon']
        if not np.isfinite([width, height, epsilon]).all() or min(width, height, epsilon) <= 0:
            raise ValueError('this study requires positive dimensions and real permittivity')
        angle = shape.get('angle', 0.)
        center = (m*shape.get('cx', 0.)+n*shape.get('cy', 0.))/PERIOD
        u = m*np.cos(angle)+n*np.sin(angle)
        v = -m*np.sin(angle)+n*np.cos(angle)
        coefficients += ((epsilon-1)*width*height*np.sinc(width*u)*np.sinc(height*v)
                         *np.exp(-2j*np.pi*center))
    return coefficients


def corrector(inner, shapes):
    """Fourier Galerkin solve of div(epsilon*(e_j+grad chi_j))=0.

    Products use zero-padded convolution of exact material coefficients. The
    inverse square root of the Laplacian preconditions both sides, and the
    constant potential is fixed to zero. No permittivity raster is solved.
    """
    if not isinstance(inner, int) or inner < 1:
        raise ValueError('inner order must be a positive integer')
    length = 2*inner+1
    n, m = np.meshgrid(np.arange(-inner, inner+1), np.arange(-inner, inner+1), indexing='ij')
    norm = np.sqrt(m*m+n*n)
    norm[inner, inner] = 1
    gradient = np.array([1j*m/norm, 1j*n/norm])
    material = material_coefficients(2*inner, shapes)
    middle = material[inner:3*inner+1, inner:3*inner+1]
    fft_size = next_fast_len(6*inner+1)
    material_fft = fft2(material, s=(fft_size, fft_size))
    def convolution(value):
        return ifft2(material_fft*fft2(value, s=(fft_size, fft_size)))[
            2*inner:4*inner+1, 2*inner:4*inner+1]
    def action(flat):
        value = flat.reshape(length, length)
        result = sum(np.conj(gradient[j])*convolution(gradient[j]*value) for j in range(2))
        result[inner, inner] = value[inner, inner]
        return result.ravel()
    operator = LinearOperator((length*length, length*length), matvec=action, dtype=complex)
    coefficients = np.zeros((2, 2, length, length), complex)
    coefficients[0, 0, inner, inner] = 1
    coefficients[1, 1, inner, inner] = 1
    residuals = []
    for j in range(2):
        right = (gradient[j]*middle).ravel()
        potential, info = cg(operator, right, rtol=1e-12, atol=1e-14, maxiter=300)
        if info:
            raise RuntimeError(f'electrostatic corrector did not converge: {info}')
        residual = np.linalg.norm(action(potential)-right)/max(np.linalg.norm(right), 1.)
        if not np.isfinite(residual) or residual > 1e-10:
            raise FloatingPointError(f'invalid electrostatic residual: {residual}')
        residuals.append(float(residual))
        coefficients[:, j] += gradient*potential.reshape(length, length)
    return coefficients, residuals


def sample(coefficients, grid):
    """Evaluate the centered series at midpoint samples, with no Nyquist mode."""
    cap = (coefficients.shape[-1]-1)//2
    n, m = np.meshgrid(np.arange(-cap, cap+1), np.arange(-cap, cap+1), indexing='ij')
    array = np.zeros((*coefficients.shape[:-2], grid, grid), complex)
    array[..., n % grid, m % grid] = coefficients*np.exp(2j*np.pi*(m+n)*(.5/grid-.5))
    values = np.fft.ifft2(array)*grid*grid
    if np.max(abs(values.imag)) > 1e-8:
        raise FloatingPointError('real material/corrector acquired an imaginary component')
    return values.real


def matrices(order, inner, shapes, grid=None):
    """Energy Gram closure with exact low-order product integrals.

    B has bandwidth inner. grid > 4*(inner+order) ensures that multiplying its
    Gram field by the projected material cannot alias onto retained differences.
    Although the projected material can ring pointwise, these integrals equal
    those against the exact positive piecewise material, not a clipped raster.
    """
    if not isinstance(order, int) or order < 0:
        raise ValueError('outer order must be a nonnegative integer')
    minimum = 4*(inner+order)
    grid = next_fast_len(minimum+1) if grid is None else grid
    if grid <= minimum:
        raise ValueError('grid must exceed 4*(inner+outer order) to prevent aliasing')
    coefficients, residuals = corrector(inner, shapes)
    basis = sample(coefficients, grid)
    epsilon = sample(material_coefficients((grid-1)//2, shapes), grid)
    n, m = np.meshgrid(np.arange(-order, order+1), np.arange(-order, order+1), indexing='ij')
    m, n = m.ravel(), n.ravel()
    dm, dn = m[:, None]-m, n[:, None]-n
    def convolution(value):
        transformed = np.fft.fft2(value)/grid**2
        return transformed[dn % grid, dm % grid]*np.exp(1j*np.pi*(dm+dn)*(1-1/grid))
    change = np.block([[convolution(basis[i, j]) for j in range(2)] for i in range(2)])
    gram = np.block([[convolution(epsilon*np.sum(basis[:, i]*basis[:, j], axis=0))
                      for j in range(2)] for i in range(2)])
    inverse = solve(change, np.eye(len(change)))
    effective = inverse.conj().T@gram@inverse
    longitudinal = convolution(epsilon)
    hermiticity = float(np.max(abs(effective-effective.conj().T)))
    if not np.isfinite(effective).all() or hermiticity > 1e-9:
        raise FloatingPointError('invalid energy closure')
    count = len(m)
    slots = (effective[:count, :count], effective[count:, count:],
             np.linalg.inv(longitudinal), -effective[:count, count:], -effective[count:, :count])
    return slots, dict(grid=grid, cell_residuals=residuals, hermiticity=hermiticity)


def polarized_powers(order, slots, xp=np, *, polarizations=('p', 's'), theta=THETA, azimuth=0.):
    """Share the layer eigensolve across incident polarizations; return vectors.

    order accepts a square half width or an explicit (x, y) pair. The physical
    period, layer and incident wave stay fixed when either half width changes.
    Optional incidence angles are fixed NumPy scalars, also when xp is JAX;
    live angular derivatives are supplied by the separate doubling study.
    """
    if isinstance(polarizations, str) or not polarizations or any(p not in ('s', 'p') for p in polarizations):
        raise ValueError('polarizations must be a nonempty sequence of s or p')
    if not np.isfinite([theta, azimuth]).all() or abs(theta) >= np.pi/2:
        raise ValueError('incidence angles must be finite with abs(theta) < pi/2')
    linear_solve = solve if xp is np else xp.linalg.solve
    n, m = order_grid(order)
    kx = np.sin(theta)*np.cos(azimuth)+WAVELENGTH/PERIOD*m.ravel()
    ky = np.sin(theta)*np.sin(azimuth)+WAVELENGTH/PERIOD*n.ravel()
    count = len(kx)
    identity = xp.eye(count, dtype=xp.complex128)
    pair_cap = max(n.shape)
    w, v, rates, coupled = layer_modes_2d(
        xp, xp.diag(xp.asarray(kx)), xp.diag(xp.asarray(ky)), identity, *slots,
        xp.complex128, pair_cap=pair_cap, with_coupling=True)
    def exterior(epsilon):
        phi = np.arctan2(ky, kx)
        phi = np.where((m.ravel() == 0) & (n.ravel() == 0), azimuth, phi)
        co, si = np.cos(phi), np.sin(phi)
        kz = np.sqrt((epsilon-kx*kx-ky*ky).astype(complex))
        index = np.sqrt(epsilon)
        e = xp.block([[-identity*si, identity*(co*kz/index)],
                      [identity*co, identity*(si*kz/index)]])
        h = 1j*xp.block([[-identity*(co*kz), -identity*(si*index)],
                         [-identity*(si*kz), identity*(co*index)]])
        return e, h, kz
    wt, vt, kt = exterior(1.)
    wb, vb, kb = exterior(2.25)
    if rates.ndim == 1:
        decay = xp.diag(xp.exp(-2*np.pi/WAVELENGTH*THICKNESS*rates))
    else:
        # Rebuilt pairs, and every abstract trace, carry a matrix generator.
        decay = exponential(xp, rates, -2*np.pi/WAVELENGTH*THICKNESS,
                            coupled=pair_cap if coupled else False)
    a, d = linear_solve(w, wb), linear_solve(v, vb)
    b, ai = (a-d)/2, xp.linalg.inv((a+d)/2)
    roundtrip = decay@b@ai@decay
    full_identity = xp.eye(2*count)
    ef = linear_solve(wt, w@(full_identity+roundtrip))
    hf = linear_solve(vt, v@(full_identity-roundtrip))
    incoming = np.zeros((2*count, len(polarizations)), complex)
    for column, polarization in enumerate(polarizations):
        incoming[(count if polarization == 'p' else 0)+count//2, column] = 1
    incoming = xp.asarray(incoming)
    forward = 2*linear_solve(ef+hf, incoming)
    transmitted, reflected = ai@decay@forward, ef@forward-incoming
    t = xp.sum(xp.asarray(np.tile(kb.real, 2))[:, None]*abs(transmitted)**2, axis=0)/np.cos(theta)
    r = xp.sum(xp.asarray(np.tile(kt.real, 2))[:, None]*abs(reflected)**2, axis=0)/np.cos(theta)
    return t, r


def powers(order, slots, xp=np, *, polarization='p'):
    """Shared NumPy/JAX scattering instrument, returning differentiable T and R."""
    t, r = polarized_powers(order, slots, xp, polarizations=(polarization,))
    return t[0], r[0]


def transmission(order, inner, shapes=None, grid=None, *, polarization='p'):
    """Use Wavebit's layer eigensolve with the candidate constitutive matrices."""
    shapes = tilted() if shapes is None else shapes
    slots, metadata = matrices(order, inner, shapes, grid)
    t, r = (float(value) for value in powers(order, slots, polarization=polarization))
    if not np.isfinite([t, r]).all():
        raise FloatingPointError('nonfinite candidate power')
    return dict(order=order, inner=inner, T=t, R=r, energy_error=abs(t+r-1), **metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--orders', type=int, nargs='+', default=[6, 10, 14])
    parser.add_argument('--inners', type=int, nargs='+', default=[31, 63, 127])
    parser.add_argument('--gates', action='store_true')
    args = parser.parse_args()
    for inner in args.inners:
        if args.gates:
            start = time.monotonic()
            merged = [dict(lx=600., ly=300., epsilon=4.)]
            split = [dict(lx=300., ly=300., cx=cx, epsilon=4.) for cx in (-150., 150.)]
            delta = transmission(3, inner, split)['T']-transmission(3, inner, merged)['T']
            values = [transmission(7, inner, [dict(lx=350., ly=700.-gap, epsilon=12.25)])['T']
                      for gap in (0., 1e-4, 1e-6)]
            print(json.dumps(dict(kind='gates', inner=inner, split_error=delta, gap_T=values,
                                  gap_slopes=[(t-values[0])/h for t, h in zip(values[1:], (1e-4, 1e-6))],
                                  seconds=time.monotonic()-start)), flush=True)
        for order in args.orders:
            start = time.monotonic()
            result = transmission(order, inner)
            print(json.dumps(dict(kind='resonance', **result, seconds=time.monotonic()-start)), flush=True)


if __name__ == '__main__':
    main()
