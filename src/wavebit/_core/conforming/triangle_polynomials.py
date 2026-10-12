"""Real Bernstein polynomials and exact moving triangle Fourier means.

NumPy-only support for the explicit polynomial Galerkin family. Integer
derivative coefficients and factorial product integrals are evaluated directly;
no fitted basis, quadrature grid, external polynomial code or dependency is used.
"""
from functools import lru_cache
from math import factorial, prod
import numpy as np

from .triangle_moments import _exp_divided_difference


@lru_cache(None)
def multi_indices(degree):
    """Vertex-first Bernstein indices, followed by remaining lexicographic ones."""
    if degree == 0:
        return ((0, 0, 0),)
    vertices = ((degree, 0, 0), (0, degree, 0), (0, 0, degree))
    others = tuple((i, j, degree-i-j) for i in range(degree+1)
                   for j in range(degree-i+1) if (i, j, degree-i-j) not in vertices)
    return vertices+others


def gradient_coefficients(degree):
    """Coefficients in degree-(p-1) Bernstein polynomials, shape (terms,nodes,2)."""
    lower = {alpha: i for i, alpha in enumerate(multi_indices(degree-1))}
    indices = multi_indices(degree)
    gradient = np.zeros((len(lower), len(indices), 2), dtype=np.int64)
    for column, alpha in enumerate(indices):
        for coordinate, direction in enumerate(((-1, -1), (1, 0), (0, 1))):
            if alpha[coordinate]:
                reduced = list(alpha); reduced[coordinate] -= 1
                gradient[lower[tuple(reduced)], column] += degree*np.asarray(direction)
    return gradient


@lru_cache(None)
def product_integrals(degree):
    """Reference-simplex Gram of degree-p Bernstein polynomials."""
    indices = multi_indices(degree)
    gram = np.array([[factorial(degree)**2
                     *prod(factorial(a+b) for a,b in zip(alpha,beta))
                     /(prod(factorial(v) for v in alpha+beta)*factorial(2*degree+2))
                     for beta in indices] for alpha in indices])
    gram.flags.writeable = False
    return gram


def phase_average(points, velocity, waves, degree):
    """Twice the reference integral of every Bernstein polynomial times exp(-iGx).

    No physical area or phase difference is divided at contact. Each moving
    phase derivative appends the corresponding divided-difference node with
    its exact multiplicity, including repeated vertices.
    """
    phase = -1j*np.einsum('md,...vd->...mv', waves, points)
    speed = -1j*np.einsum('md,...vd->...mv', waves, velocity)
    values, changes = [], []
    factor = 2*factorial(degree)
    for alpha in multi_indices(degree):
        nodes = np.repeat(np.arange(3), np.asarray(alpha)+1)
        values.append(factor*_exp_divided_difference(phase[..., nodes]))
        changes.append(sum(factor*(alpha[j]+1)*speed[..., j]
                           *_exp_divided_difference(phase[..., np.r_[nodes,j]]) for j in range(3)))
    return np.stack(values, -2), np.stack(changes, -2)


def values_and_tangent(barycentric, tangent, degree):
    """Evaluate real Bernstein polynomials without dividing by a coordinate."""
    def evaluate(alpha):
        return np.prod(barycentric**np.asarray(alpha), axis=-1)
    values, changes = [], []
    for alpha in multi_indices(degree):
        scale = factorial(degree)/prod(factorial(i) for i in alpha)
        values.append(scale*evaluate(alpha))
        change = np.zeros(barycentric.shape[:-1])
        for i, power in enumerate(alpha):
            if power:
                lower = list(alpha); lower[i] -= 1
                change += scale*power*tangent[...,i]*evaluate(lower)
        changes.append(change)
    return np.stack(values, axis=-1), np.stack(changes, axis=-1)
