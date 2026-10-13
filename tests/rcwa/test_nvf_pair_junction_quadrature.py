"""Analytic endpoint energies and exact piecewise polynomial transport."""
from fractions import Fraction
from math import fsum, gamma
import numpy as np
import pytest
from wavebit.rcwa._engine.conforming.cartesian_pair_opening import CartesianPairOpeningPlan
from wavebit.rcwa._engine.conforming.cartesian_pair_polynomial_galerkin import CartesianPairPolynomialGalerkinGapCell
from wavebit.rcwa._engine.conforming.cartesian_pair_seam_galerkin import _geometry
from wavebit.rcwa._engine.conforming.pair_corner_fields import PairCornerFields
from wavebit.rcwa._engine.conforming.pair_junction_partition import partitions, scalar, point, det
from wavebit.rcwa._engine.conforming.pair_junction_transport import samples

pytestmark = pytest.mark.research


@pytest.mark.parametrize('exponent', (.6, .8))
@pytest.mark.parametrize('order', (24, 32, 48))
def test_both_thin_endpoints_integrate_the_analytic_beta_energy(exponent, order):
    obj = CartesianPairPolynomialGalerkinGapCell(CartesianPairOpeningPlan(0, 0), 1, degree=3)
    fields = PairCornerFields(obj, junctions=True)
    triangle = next(tri for tri, thin in zip(obj.data['triangles'], obj.data['thin']) if thin)
    hats = fields.hats[triangle[:3]]
    corner = next(i for i in (5, 8) if np.sum(hats[:, i] == 1) == 1)
    beta = lambda a, b: gamma(a)*gamma(b)/gamma(a+b)
    # Integrate [d/dr r*(1-r)**alpha]**2 with cross-section 1-r.
    exact = (beta(1, 2*exponent+2)-2*exponent*beta(2, 2*exponent+1)
             +exponent**2*beta(3, 2*exponent))
    integrals = []
    for bary, weight in fields.quadrature(triangle, order):
        radial = bary@hats[:, corner]
        complement = bary@hats[:, np.arange(10) != corner].sum(axis=1)
        derivative = complement**exponent-exponent*radial*complement**(exponent-1)
        integrals.extend(weight*derivative**2)
    assert abs(fsum(integrals)-exact) < 3e-13


def profile_region(poly, vertices, chart):
    x = sum(point(vertex[0], vertices)[0] for vertex in poly)/len(poly)
    gap = scalar(chart[1])
    left, right = map(scalar, chart[5])
    return (0 if x < -left-gap/2 else 1 if x < -gap/2 else
            2 if x < gap/2 else 3 if x < right+gap/2 else 4)


def exact_polynomial_moments(poly, determinant, derivative):
    values = [Fraction(0)]*6
    changes = [Fraction(0)]*6
    for i in range(1, len(poly)-1):
        vertices, directions = zip(poly[0], poly[i], poly[i+1])
        vertices, directions = list(vertices), list(directions)
        area = det(vertices)/2
        darea = sum(det(vertices[:j]+[directions[j]]+vertices[j+1:]) for j in range(3))/2
        means = [Fraction(1)]+[sum(row[j] for row in vertices)/3 for j in range(3)]
        dmeans = [Fraction(0)]+[sum(row[j] for row in directions)/3 for j in range(3)]
        for j, k in ((0, 0), (0, 1)):
            a, b = (sum(row[index] for row in vertices) for index in (j, k))
            da, db = (sum(row[index] for row in directions) for index in (j, k))
            means.append((a*b+sum(row[j]*row[k] for row in vertices))/12)
            dmeans.append((da*b+a*db+sum(v[j]*dv[k]+v[k]*dv[j]
                                       for v, dv in zip(vertices, directions)))/12)
        for j, (mean, dmean) in enumerate(zip(means, dmeans)):
            values[j] += determinant*area*mean
            changes[j] += derivative*area*mean+determinant*(darea*mean+area*dmean)
    return values, changes


@pytest.mark.parametrize('refinement', (0, 1))
@pytest.mark.parametrize('gap', (.001, 1e-7, 1e-9))
def test_transport_preserves_exact_moving_profile_moments(refinement, gap):
    obj = CartesianPairPolynomialGalerkinGapCell(
        CartesianPairOpeningPlan(0, refinement), 1, degree=3)
    settings = dict(basis=np.diag([700., 700.]), widths=(250., 320.), height=300.,
                    offset=(21., -13.), epsilon=(4., 9.), background=1.,
                    epsilon_tangent=(0., 0.), background_tangent=0., gap_tangent=1., gap=gap)
    base = list(obj._assemble(**settings))
    base[-1] = dict(base[-1], corner_fields=PairCornerFields(obj, junctions=True))
    chart = _geometry(obj, settings)
    s, ds = scalar(chart[1]), scalar(base[-1]['ds'])
    exact, sampled, total, determinants = {}, {}, {}, {}
    for tri, local, vertices, speeds, poly in partitions(obj, base, settings, chart, True):
        key = tuple(tri), profile_region(poly, vertices, chart)
        determinant, derivative = ((scalar(local[10]), Fraction(0)) if local[11] else
                                  (scalar(local[9])+s*scalar(local[10]), ds*scalar(local[10])))
        piece = exact_polynomial_moments(poly, determinant, derivative)
        accumulated = exact.setdefault(key, [[Fraction(0)]*6, [Fraction(0)]*6])
        whole = total.setdefault(tuple(tri), [[Fraction(0)]*6, [Fraction(0)]*6])
        determinants[tuple(tri)] = determinant, derivative
        for group in range(2):
            for i in range(6):
                accumulated[group][i] += piece[group][i]
                whole[group][i] += piece[group][i]
    reference = [Fraction(1, n) for n in (2, 6, 6, 6, 12, 24)]
    assert len(total) == len(obj.data['triangles'])
    for key, groups in total.items():
        for group in range(2):
            assert groups[group] == [determinants[key][group]*v for v in reference]
    for tri, local, vertices, speeds, poly, bary, dbary, measure, linear in samples(
            obj, base, settings, chart, 24):
        weight, dweight = measure(local[11])
        assert np.all(weight >= 0)
        np.testing.assert_array_equal(dbary, 0.)
        key = tuple(tri), profile_region(poly, vertices, chart)
        polynomial = np.column_stack([np.ones(len(bary)), bary, bary[:, 0]**2, bary[:, 0]*bary[:, 1]])
        groups = sampled.setdefault(key, [[], []])
        groups[0].append(weight[:, None]*polynomial)
        groups[1].append(dweight[:, None]*polynomial)
    assert sampled.keys() == exact.keys()
    for key, groups in sampled.items():
        for group in range(2):
            actual = np.array([fsum(column) for column in np.concatenate(groups[group]).T])
            expected = np.asarray(exact[key][group], float)
            factor = 700 if group else 1
            np.testing.assert_allclose(factor*actual, factor*expected,
                                       atol=3e-10 if group else 3e-13, rtol=0)
