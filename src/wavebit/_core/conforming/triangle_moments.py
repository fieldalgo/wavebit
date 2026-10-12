"""Signed quadratic Bernstein triangle Fourier integrals and moving derivatives.

A degree-two Bernstein moment is twice an exponential divided difference
with five nodes, repeating vertex i alpha_i+1 times. Differentiating a
vertex phase appends that node with multiplicity alpha_i+1. Confluent and
near-confluent nodes use the convergent complete-homogeneous series, so no
geometric area or phase difference is divided by at collapse.

This internal NumPy instrument supports the experimental quadratic pair cell.
"""
from math import factorial
import numpy as np


QUADRATIC_MULTI_INDICES = ((2, 0, 0), (0, 2, 0), (0, 0, 2),
                           (1, 1, 0), (0, 1, 1), (1, 0, 1))


def _exp_divided_difference(nodes):
    """Evaluate exp[z0,...,zn] for imaginary nodes, including repeated nodes."""
    nodes = np.take_along_axis(nodes, np.argsort(nodes.imag, axis=-1), axis=-1)
    values = np.exp(nodes)
    for degree in range(1, nodes.shape[-1]):
        delta = nodes[..., degree:] - nodes[..., :-degree]
        close = np.abs(delta) < 2
        updated = (values[..., 1:] - values[..., :-1]) / np.where(close, 1, delta)
        for index in range(updated.shape[-1]):
            repeated = delta[..., index] == 0
            updated[..., index] = np.where(
                repeated, np.exp(nodes[..., index]) / factorial(degree), updated[..., index])
            mask = close[..., index] & ~repeated
            if not np.any(mask):
                continue
            subset = nodes[..., index:index+degree+1][mask]
            center = subset.mean(axis=-1)
            shifted = subset - center[..., None]
            # h_k are coefficients of product_j (1 - shifted_j*t)^(-1).
            # With phase span < 2 and at most six nodes, the omitted tail
            # after k=20 is below double precision's absolute roundoff scale.
            h = np.zeros((len(subset), 21), complex)
            h[:, 0] = 1
            for node in shifted.T:
                for k in range(1, 21):
                    h[:, k] += node * h[:, k-1]
            series = sum(h[:, k] / float(factorial(degree+k)) for k in range(21))
            updated[..., index][mask] = np.exp(center) * series
        values = updated
    return values[..., 0]


def triangle_phase_average(points, velocity, wave):
    """Mean plane wave on an affine triangle, including its collapsed limit.

    This is twice the integral on the reference simplex, not a physical
    integral divided numerically by area. Leading triangle batch axes are
    supported with one shared (modes, 2) wave array. Its moving-vertex derivative is
    consequently finite even when two physical vertices coincide. Multiplying
    by the signed physical area gives the ordinary indicator transform.
    """
    phase = -1j * np.einsum('md,...vd->...mv', wave, points)
    speed = -1j * np.einsum('md,...vd->...mv', wave, velocity)
    value = 2 * _exp_divided_difference(phase)
    derivative = sum(2 * speed[..., i] * _exp_divided_difference(
        phase[..., [0, 1, 2, i]]) for i in range(3))
    return value, derivative


def triangle_linear_phase_average(points, velocity, wave):
    """Reference means of lambda_i times a plane wave and their derivatives.

    The last two result axes are (3, modes). Their sum over the barycentric
    index is ``triangle_phase_average``. A repeated phase contributes twice
    to its derivative; this matters even at a geometrically collapsed face.
    """
    phase = -1j * np.einsum('md,...vd->...mv', wave, points)
    speed = -1j * np.einsum('md,...vd->...mv', wave, velocity)
    values, derivatives = [], []
    for i in range(3):
        indices = [0, 1, 2, i]
        values.append(2 * _exp_divided_difference(phase[..., indices]))
        derivatives.append(sum(2 * (1 + (i == j)) * speed[..., j]
                               * _exp_divided_difference(phase[..., indices+[j]])
                               for j in range(3)))
    return np.stack(values, -2), np.stack(derivatives, -2)


def quadratic_triangle_weights(points, velocity, wave):
    """Return six signed integrals and their moving-vertex derivatives.

    points/velocity have shape (3,2); wave has shape (modes,2). Outputs have
    shape (6,modes), in QUADRATIC_MULTI_INDICES order. No cell-area division
    is included. The velocity parameter uses the caller's derivative units.
    Points and wave vectors must be real; the caller validates their domain.
    """
    phase = -1j * (wave @ points.T)
    dphase = -1j * (wave @ velocity.T)
    j = np.column_stack([points[1]-points[0], points[2]-points[0]])
    dj = np.column_stack([velocity[1]-velocity[0], velocity[2]-velocity[0]])
    determinant = j[0, 0]*j[1, 1] - j[0, 1]*j[1, 0]
    slope = (dj[0, 0]*j[1, 1] + j[0, 0]*dj[1, 1]
             - dj[0, 1]*j[1, 0] - j[0, 1]*dj[1, 0])
    alpha = np.asarray(QUADRATIC_MULTI_INDICES)
    nodes = np.stack([phase[:, np.repeat(np.arange(3), a+1)] for a in alpha])
    base = _exp_divided_difference(nodes)
    derivative = np.zeros_like(base)
    if determinant != 0:
        for i in range(3):
            repeated = np.eye(3, dtype=int)[i]
            nodes = np.stack([phase[:, np.repeat(np.arange(3), a+1+repeated)] for a in alpha])
            derivative += (alpha[:, i]+1)[:, None] * _exp_divided_difference(nodes) * dphase[:, i]
    return 2*determinant*base, 2*(slope*base + determinant*derivative)
