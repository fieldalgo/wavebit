"""Independent Decimal potential differences at moving transmission samples."""
from decimal import Decimal as Z, localcontext
from functools import lru_cache
from math import factorial, prod

import numpy as np
import pytest

from wavebit._core.conforming.cartesian_pair_opening import CartesianPairOpeningPlan
from wavebit._core.conforming.cartesian_pair_transmission_galerkin import CartesianPairTransmissionGalerkinGapCell
from wavebit._core.conforming.pair_transmission_fields import PairTransmissionFields
from wavebit._core.conforming.triangle_polynomials import multi_indices
from test_nvf_pair_transmission import GEOMETRY, REFERENCE
import test_nvf_transmission_common_precision as oracle

pytestmark = pytest.mark.research


def dec(value):
    return Z.from_float(float(value))


@pytest.mark.parametrize('degree,refinement', ((5, 0), (6, 0), (5, 1)))
def test_complete_moving_gradients_match_decimal_potential_differences(degree, refinement):
    check_moving_potential(degree, refinement)


def check_moving_potential(degree, refinement, cutoff_order=0, exact_points=False):
    obj = CartesianPairTransmissionGalerkinGapCell(
        CartesianPairOpeningPlan(0, refinement), 0, reference=REFERENCE,
        degree=degree, branches=2, angular_quadrature=32, cutoff_order=cutoff_order)._base
    fields = PairTransmissionFields(obj, GEOMETRY, REFERENCE, 2, 32, cutoff_order)
    data = obj.data
    nold = data['Cfull'].shape[1] - 1
    mapping = np.random.default_rng(71).normal(size=(nold + fields.count, 3))
    mapping *= [1., 32., 1/32]
    common = fields.common_coordinate_type(fields, mapping)
    bary = np.array([[.23, .31, .46], [.52, .17, .31]])
    if exact_points:
        from wavebit._core.conforming.pair_corner_precision import D
        from wavebit._core.conforming.pair_contact_quadrature import Exact
        bary = Exact(D(bary, np.array([[1., -2., 1.], [-2., 1., 1.]])*2.**-59))
    dbary = np.array([[.04, -.01, -.03], [-.03, .02, .01]])
    inverse = np.array([[1.25, .25], [.125, .875]])
    dinverse = np.array([[.03125, .0625], [-.015625, .03125]])

    verified = []
    with localcontext() as context:
        context.prec = 100
        pi = 16*oracle.atan(Z(1)/5) - 4*oracle.atan(Z(1)/239)
        oracle.PI = pi
        step = Z('1e-20')
        raw = np.full((fields.count, 3), Z(0), object)
        for ids, transform in zip(fields.groups, fields.transforms):
            for i, index in enumerate(ids):
                for column in range(3):
                    raw[index, column] = dec(fields.column_scale[index])*sum(
                        dec(transform[i, j])*dec(mapping[nold + k, column])
                        for j, k in enumerate(ids))

        for tri in data['triangles'][[0, 7, 16, -1]]:
            actual = common.sample(tri, bary, inverse, dinverse, dbary)
            # A supplied zero transport must preserve the previous static path.
            static = common.sample(tri, bary, inverse, dinverse)
            zero = common.sample(tri, bary, inverse, dinverse, np.zeros_like(bary))
            for a, b in zip(static, zero):
                np.testing.assert_array_equal(a.hi, b.hi)
                np.testing.assert_array_equal(a.lo, b.lo)
            local = fields.interpolant[data['pmap'][data['periodic'][tri]]]
            polynomial = np.empty((len(tri), 3), object)
            for row, node in enumerate(tri):
                for column in range(3):
                    polynomial[row, column] = sum(
                        dec(v)*dec(mapping[i, column]) for i, v in enumerate(
                            data['Cfull'][data['periodic'][node], 1:])) - sum(
                        dec(local[row, i])*raw[i, column] for i in range(fields.count))

            @lru_cache(None)
            def potential(b):
                bernstein = [Z(factorial(degree)//prod(factorial(int(a)) for a in alpha))
                             * prod(b[i]**int(a) for i, a in enumerate(alpha))
                             for alpha in multi_indices(degree)]
                result = [sum(v*polynomial[row, col] for row, v in enumerate(bernstein))
                          for col in range(3)]
                first = 0
                for (corner, center, mode, parts), power in zip(fields.modes, fields.cutoff_powers):
                    cutoff = fields.hats[tri[:3], corner]
                    if not np.any(cutoff):
                        first += len(parts)
                        continue
                    offsets = fields.zero[tri[:3]] - center
                    offsets[cutoff == 1] = 0.
                    x, y = (sum(b[i]*dec(offsets[i, axis]) for i in range(3)) for axis in range(2))
                    center_angle = (np.arctan2(*offsets.mean(axis=0)[::-1]) - mode.start) % (2*np.pi)
                    sector = min(np.searchsorted(np.cumsum(mode.angles), center_angle, side='right'), 2)
                    theta = oracle.phase(x, y)
                    shift = round((center_angle + mode.start - float(theta))/(2*np.pi))
                    quarter = round(mode.start/(np.pi/2))
                    angle = theta + (2*shift - Z(quarter + sum((1, 1, 2)[:sector]))/2)*pi
                    lam = oracle.scalar(mode.complete_exponent)
                    u, flux = map(oracle.scalar, mode.complete_states[sector])
                    q = oracle.div(flux, oracle.scalar(mode.complete_coefficients[sector]))
                    a = oracle.div(oracle.add(u, oracle.neg(oracle.mul((Z(0), Z(1)), q))), (Z(2), Z(0)))
                    bcoef = oracle.div(oracle.add(u, oracle.mul((Z(0), Z(1)), q)), (Z(2), Z(0)))
                    logr = (x*x + y*y).ln()/2
                    value = oracle.add(
                        oracle.mul(a, oracle.exponential(oracle.mul(lam, (logr, angle)))),
                        oracle.mul(bcoef, oracle.exponential(oracle.mul(lam, (logr, -angle)))))
                    hat = sum(b[i]*dec(cutoff[i]) for i in range(3))
                    complement = sum(b[i]*dec(1-cutoff[i]) for i in range(3))
                    for part in parts:
                        component = value[0 if part == 'real' else 1]*hat*complement**power
                        for col in range(3):
                            result[col] += component*raw[first, col]
                        first += 1
                return result

            def gradient(b, inv, axis):
                direction = [-inv[0][axis] - inv[1][axis], inv[0][axis], inv[1][axis]]
                left = potential(tuple(v-step*d for v, d in zip(b, direction)))
                right = potential(tuple(v+step*d for v, d in zip(b, direction)))
                return [(r-l)/(2*step) for l, r in zip(left, right)]

            inv = [[dec(v) for v in row] for row in inverse]
            dinv = [[dec(v) for v in row] for row in dinverse]
            for row, point in enumerate(bary):
                b, db = list(map(dec, point)), list(map(dec, dbary[row]))
                if exact_points:
                    b = [dec(bary.exact.hi[row, i])+dec(bary.exact.lo[row, i]) for i in range(3)]
                anchor = int(np.argmax(abs(point)))
                b[anchor] = 1-sum(v for i, v in enumerate(b) if i != anchor)
                db[anchor] = -sum(v for i, v in enumerate(db) if i != anchor)
                for axis in range(2):
                    expected = gradient(b, inv, axis)
                    perturbed = [gradient(
                        [v+sign*step*d for v, d in zip(b, db)],
                        [[v+sign*step*d for v, d in zip(a, da)] for a, da in zip(inv, dinv)],
                        axis) for sign in (-1, 1)]
                    changed = [(r-l)/(2*step) for l, r in zip(*perturbed)]
                    for sample, target in ((actual[axis], expected), (actual[axis+2], changed)):
                        for col, reference in enumerate(target):
                            observed = dec(sample.hi[row, col]) + dec(sample.lo[row, col])
                            error = abs(observed-reference)/max(Z(1), abs(reference))
                            assert error < Z('3e-28'), (degree, refinement, row, axis, col, error)
            potential.cache_clear()
            verified.append((tri.copy(), actual))
    return fields, mapping, bary, dbary, inverse, dinverse, verified


def test_wide_chunked_fields_retain_decimal_verified_columns_after_triangle_revisits():
    from wavebit._core.conforming.pair_corner_precision import D
    from wavebit._core.conforming.pair_contact_quadrature import Exact

    fields, mapping, bary, dbary, inverse, dinverse, verified = check_moving_potential(
        8, 1, exact_points=True)
    # Independently checked Decimal columns also define a wide bank by exact
    # binary scaling. Repeated samples exercise several bounded row blocks.
    scales = np.ldexp(np.ones(258), np.arange(258) % 9 - 4)
    wide_mapping = np.tile(mapping, (1, 86))*scales
    common = fields.common_coordinate_type(fields, wide_mapping)
    points = Exact(D(np.tile(bary.exact.hi, (257, 1)),
                     np.tile(bary.exact.lo, (257, 1))))
    directions = np.tile(dbary, (257, 1))
    for tri, reference in [*verified, verified[0]]:
        actual = common.sample(tri, points, inverse, dinverse, directions)
        for value, expected in zip(actual, reference):
            np.testing.assert_array_equal(value.hi, np.tile(expected.hi, (257, 86))*scales)
            np.testing.assert_array_equal(value.lo, np.tile(expected.lo, (257, 86))*scales)
