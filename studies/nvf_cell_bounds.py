"""Cell-energy enclosures and projected matrices for the NVF research candidate.

The upper/lower bounds concern a static scalar periodic cell problem. They are
not bounds on optical transmission. Integrals and solves use floating-point
arithmetic, not interval arithmetic. No production factorization is selected.
"""
import argparse
import json
import time

import numpy as np
from scipy.fft import fft2, ifft2, next_fast_len
from scipy.linalg import eigvalsh, solve

from studies.nvf_fourier_orders import order_grid, orders_xy


def energy_bounds(upper, reciprocal_upper):
    """2D primal/complementary variational enclosure for scalar positive epsilon.

    A curl-free trial of mean I supplies upper. Rotating the reciprocal-material
    trial gives a divergence-free flux of mean I. Its complementary energy
    supplies lower = R inv(reciprocal_upper) R.T.
    """
    matrices = []
    for value in (upper, reciprocal_upper):
        value = np.asarray(value)
        if value.shape != (2, 2) or not np.isfinite(value).all():
            raise ValueError('cell energy must be a finite 2-by-2 matrix')
        scale = max(float(np.max(abs(value))), 1.)
        if np.max(abs(value.imag)) > 1e-10*scale or np.max(abs(value-value.T)) > 1e-10*scale:
            raise ValueError('cell energy must be real symmetric')
        value = (value.real+value.real.T)/2
        if eigvalsh(value)[0] <= 0:
            raise ValueError('cell energy must be positive definite')
        matrices.append(value)
    upper, reciprocal = matrices
    rotation = np.array([[0., -1.], [1., 0.]])
    lower = rotation@solve(reciprocal, np.eye(2), assume_a='pos')@rotation.T
    gap = upper-lower
    gap_eigenvalues = eigvalsh(gap)
    if gap_eigenvalues[0] < -1e-10*max(float(np.max(abs(upper))), 1.):
        raise ValueError('primal/complementary cell bounds are inconsistent')
    return dict(upper=upper, lower=lower, gap_eigenvalues=gap_eigenvalues,
                relative_gap=float(eigvalsh(gap, lower)[-1]))


def fourier_energy(inner, shapes):
    # Lazy: the optional FEM instrument can import energy_bounds without Wavebit.
    from studies.nvf_harmonic_energy import corrector, material_coefficients
    basis, residuals = corrector(inner, shapes)
    size = next_fast_len(6*inner+1)
    material = material_coefficients(2*inner, shapes)
    flux = ifft2(fft2(material, s=(size, size))*fft2(basis, s=(size, size)))[
        ..., 2*inner:4*inner+1, 2*inner:4*inner+1]
    energy = np.einsum('kihw,kjhw->ij', basis.conj(), flux)
    return energy, residuals


def fourier_bounds(inner, shapes):
    upper, residuals = fourier_energy(inner, shapes)
    reciprocal = [dict(shape, epsilon=1/shape['epsilon']) for shape in shapes]
    dual, dual_residuals = fourier_energy(inner, reciprocal)
    return dict(**energy_bounds(upper, dual), inner=inner,
                residuals=residuals, reciprocal_residuals=dual_residuals)


def projected_matrices(coefficients, order):
    """Apply the same energy closure to independently integrated cell fields.

    Rows are Bxx, Bxy, Byx, Byy, (B.H eps B)xx/xy/yy, epsilon. Each row contains
    centered Fourier coefficients on [-cap,cap]^2. Real scalar epsilon and real
    correctors make the Gram field symmetric. Adequate numerical quadrature
    remains the exporter's responsibility; lossless energy alone cannot check it.
    order is a square half width or an explicit (x, y) pair. Each retained
    space gets its own closure; cropping an already inverted larger closure
    would define a different model.
    """
    c = np.asarray(coefficients)
    if c.ndim != 3 or c.shape[0] != 8 or c.shape[1] != c.shape[2] or c.shape[1] % 2 != 1:
        raise ValueError('expected eight square odd-sized coefficient arrays')
    cap = (c.shape[-1]-1)//2
    if 2*max(orders_xy(order)) > cap:
        raise ValueError('coefficient cap must cover all retained order differences')
    if not np.isfinite(c).all() or np.max(abs(c-c[:, ::-1, ::-1].conj())) > 1e-9:
        raise ValueError('invalid real-field Fourier coefficients')
    n, m = order_grid(order)
    dm, dn = m.ravel()[:, None]-m.ravel(), n.ravel()[:, None]-n.ravel()
    count = m.size
    def conv(row):
        return c[row, dn+cap, dm+cap]
    change = np.block([[conv(0), conv(1)], [conv(2), conv(3)]])
    gram = np.block([[conv(4), conv(5)], [conv(5), conv(6)]])
    inverse = solve(change, np.eye(2*count))
    effective = inverse.conj().T@gram@inverse
    if not np.isfinite(effective).all() or np.max(abs(effective-effective.conj().T)) > 1e-9:
        raise FloatingPointError('invalid projected energy closure')
    return (effective[:count, :count], effective[count:, count:],
            np.linalg.inv(conv(7)), -effective[:count, count:], -effective[count:, :count])


def serializable(result):
    return {key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in result.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inners', type=int, nargs='+', default=[31, 63, 127, 255])
    parser.add_argument('--fixture', choices=['bar', 'tilted'], default='tilted')
    parser.add_argument('--projection', help='NPZ exported by nvf_cell_corrector.py')
    parser.add_argument('--orders', type=int, nargs='+', default=[6, 10, 14])
    args = parser.parse_args()
    from studies.nvf_harmonic_energy import powers, tilted
    if args.projection:
        with np.load(args.projection, allow_pickle=False) as data:
            coefficients = data['coefficients']
            metadata = json.loads(str(data['metadata']))
        for order in args.orders:
            start = time.monotonic()
            t, r = (float(value) for value in powers(order, projected_matrices(coefficients, order)))
            if not np.isfinite([t, r]).all():
                raise FloatingPointError('nonfinite projected power')
            print(json.dumps(dict(kind='projected_cell', order=order, T=t, R=r,
                                  energy_error=abs(t+r-1), projection=metadata,
                                  seconds=time.monotonic()-start)), flush=True)
    else:
        shapes = tilted() if args.fixture == 'tilted' else [dict(lx=350., ly=700., epsilon=12.25)]
        for inner in args.inners:
            start = time.monotonic()
            result = serializable(fourier_bounds(inner, shapes))
            print(json.dumps(dict(kind='fourier_cell', fixture=args.fixture, **result,
                                  seconds=time.monotonic()-start)), flush=True)


if __name__ == '__main__':
    main()
