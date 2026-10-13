"""Analytic trace remainders against decimal arithmetic and cardinal fields."""
from decimal import Decimal, localcontext
from math import comb

import numpy as np
import pytest

from wavebit.rcwa._engine.conforming.polynomial_pair_trace import _frame, _raw_basis
from tests.rcwa.test_nvf_cartesian_pair_opening import _oracle_geometry
from tests.rcwa.test_nvf_pair_seam_galerkin import cell, GEOMETRY
from tests.rcwa.test_nvf_polynomial_galerkin import cardinal_gradient, indices
from tests.rcwa.test_nvf_pair_quadratic import duffy

pytestmark = pytest.mark.research


def decimal_even_remainder(order, data, coordinate):
    """Subtract the retained polynomial in 70 digits, using integer binomials.

    The oracle evaluates asin and its analytic derivatives independently.
    It does not use the candidate's tail recurrence or Clenshaw evaluation.
    """
    with localcontext() as context:
        context.prec = 70
        a, k, h, u = map(Decimal.from_float, (data['amplitude'], data['frequency'],
                                            data['width']/2, float(coordinate)))
        coefficients = [Decimal(0) for _ in range(2*order+1)]
        for n in range(1, 300):
            term = a**(2*n)/(k*k*n*n)
            coefficients[0] += term/2
            for j in range(1, min(order, n)+1):
                coefficients[2*j] += term*Decimal(comb(2*n, n-j))/Decimal(comb(2*n, n))
        coefficients[0] -= h*h
        values = [Decimal(1), u]
        first = [Decimal(0), Decimal(1)]
        second = [Decimal(0), Decimal(0)]
        for j in range(1, 2*order):
            values.append(2*u*values[-1]-values[-2])
            first.append(2*values[-2]+2*u*first[-1]-first[-2])
            second.append(4*first[-2]+2*u*second[-1]-second[-2])
        z = a*u
        term = arcsine = z
        for n in range(300):
            term *= z*z*Decimal((2*n+1)**2)/Decimal(2*(n+1)*(2*n+3))
            arcsine += term
        root = (1-z*z).sqrt()
        bubble = (arcsine*arcsine/(k*k)-h*h,
                  2*a*arcsine/(k*k*root),
                  2*a*a/(k*k)*(1/(root*root)+arcsine*z/root**3))
        scale = Decimal.from_float(data['scale'])
        return np.array([float((sum(c*v for c,v in zip(coefficients, polynomial))-b)/scale)
                         for polynomial,b in zip((values, first, second), bubble)])


@pytest.mark.parametrize('order', (2, 4, 8))
def test_small_even_remainder_and_spatial_derivatives_keep_decimal_accuracy(order):
    data = _frame(order, 3, 1., 3/7, 64)
    y = np.array([-.21, -.13, .001, .079, .19])
    k, a = data['frequency'], data['amplitude']
    u = np.sin(k*y)/a
    du, ddu = k*np.cos(k*y)/a, -k*k*u
    value, first, second = np.array([decimal_even_remainder(order, data, x) for x in u]).T
    expected = (value, first*du, second*du*du+first*ddu)
    for actual, reference in zip(_raw_basis(y, order, data), expected):
        np.testing.assert_allclose(actual[:,0], reference, atol=2e-11, rtol=3e-13)


def decimal_odd_remainder(order, data, coordinates):
    """Independent decimal product identity followed by direct subtraction."""
    with localcontext() as context:
        context.prec = 70
        a, k, h = map(Decimal.from_float, (data['amplitude'], data['frequency'], data['width']/2))
        count = 100
        c = [Decimal(0) for _ in range(count+1)]
        for n in range(1, count+1):
            term = a**(2*n)/(k*k*n*n)
            c[0] += term/2
            for j in range(1, n+1):
                c[j] += term*Decimal(comb(2*n, n-j))/Decimal(comb(2*n, n))
        c[0] -= h*h
        square = [Decimal(0) for _ in range(2*count+1)]
        for i, x in enumerate(c):
            for j, y in enumerate(c):
                square[i+j] += x*y/2
                square[abs(i-j)] += x*y/2
        coefficients = [k/a*sum(Decimal(l)*square[l] for l in range(j+1, len(square)))
                        for j in range(order)]
        results = []
        for coordinate in coordinates:
            u = Decimal.from_float(float(coordinate))
            values, first, second = [Decimal(1), u], [Decimal(0), Decimal(1)], [Decimal(0), Decimal(0)]
            for j in range(1, 2*order):
                values.append(2*u*values[-1]-values[-2])
                first.append(2*values[-2]+2*u*first[-1]-first[-2])
                second.append(4*first[-2]+2*u*second[-1]-second[-2])
            p, dp, ddp = [sum(c*v[2*j+1] for j,c in enumerate(coefficients))
                          for v in (values, first, second)]
            z = a*u
            term = arcsine = z
            for n in range(200):
                term *= z*z*Decimal((2*n+1)**2)/Decimal(2*(n+1)*(2*n+3))
                arcsine += term
            root = (1-z*z).sqrt()
            y, du, ddu = arcsine/k, k*root/a, -k*k*u
            value = root*p-y*(y*y-h*h)
            derivative = -k*z*p+root*dp*du-(3*y*y-h*h)
            curvature = -k*k*root*p-2*k*z*dp*du+root*(ddp*du*du+dp*ddu)-6*y
            results.append([float(v/Decimal.from_float(data['odd_scale']))
                            for v in (value, derivative, curvature)])
        return np.array(results).T


@pytest.mark.parametrize('order', (2, 4, 8))
def test_small_odd_remainder_and_spatial_derivatives_keep_decimal_accuracy(order):
    data = _frame(order, 4, 1., 3/7, 64)
    y = np.array([-.21, -.13, .001, .079, .19])
    u = np.sin(data['frequency']*y)/data['amplitude']
    expected = decimal_odd_remainder(order, data, u)
    for actual, reference in zip(_raw_basis(y, order, data), expected):
        np.testing.assert_allclose(actual[:,1], reference, atol=2e-11, rtol=3e-13)


@pytest.mark.parametrize('degree', (3, 4))
@pytest.mark.parametrize('refinement', (0, 1))
@pytest.mark.parametrize('gap', (7., .001))
def test_subtracted_bubbles_belong_to_every_original_cardinal_triangle(degree, refinement, gap):
    obj = cell(degree=degree, refinement=refinement)._base
    length = np.sqrt(np.linalg.det(GEOMETRY['basis']))
    geometry = {k: np.asarray(v)/length for k,v in GEOMETRY.items()}
    graph = _oracle_geometry(obj.data, geometry)
    a, b = geometry['widths']
    height = geometry['height']
    center = np.diag(geometry['basis'])/2+geometry['offset']+[(a-b)/2, 0.]
    s = gap/length
    points = np.asarray(graph['vertices0'], float)+s*np.asarray(graph['vertices1'], float)-center
    barycentric, _ = duffy(8)
    cardinal_points = np.asarray(indices(degree))/degree

    def analytic(xy, power):
        x, y = xy.T
        bubble, by = (y-height/2)*(y+height/2), 2*y
        if power == 3:
            bubble, by = y*bubble, 3*y*y-height*height/4
        profile = np.where(x < -s/2, -s*(x+a+s/2)/(2*a),
                           np.where(x < s/2, x, s*(b+s/2-x)/(2*b)))
        normal = np.where(x < -s/2, -s/(2*a), np.where(x < s/2, 1., -s/(2*b)))
        inside = ((x >= -a-s/2-1e-14) & (x <= b+s/2+1e-14)
                  & (np.abs(y) <= height/2+1e-14))
        return (np.where(inside, profile*bubble, 0.),
                np.where(inside[:,None], np.column_stack([normal*bubble, profile*by]), 0.))

    for triangle in graph['triangles']:
        vertex = points[triangle[:3]]
        jacobian = np.column_stack([vertex[1]-vertex[0], vertex[2]-vertex[0]])
        gradient = cardinal_gradient(barycentric, degree)@np.linalg.inv(jacobian)
        for power in range(2, degree):
            coefficients = analytic(cardinal_points@vertex, power)[0]
            represented = np.einsum('qna,n->qa', gradient, coefficients)
            expected = analytic(barycentric@vertex, power)[1]
            np.testing.assert_allclose(represented, expected, atol=5e-13, rtol=0.)
