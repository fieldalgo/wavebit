"""P2 finite equations against nodal polynomials and independent Duffy integrals.

No candidate gradient, bilinear integral, Fourier moment or rescaled matrix is
used by the reference. The collapse graph and rational geometric map are shared
with independently qualified P1 topology; a six-node Vandermonde defines P2.
The old coefficient 3e-10, normalized tangent 3e-9, residual 1e-10 guards remain.
These are finite algebra checks, not assertions of optical convergence.
"""
from functools import lru_cache
from itertools import permutations
from math import factorial

import numpy as np
import pytest

from wavebit.rcwa._engine.conforming.cartesian_pair_quadratic import CartesianPairQuadraticGapCell
from wavebit.rcwa._engine.conforming.cartesian_pair_opening import CartesianPairOpeningPlan
from wavebit.rcwa._engine.conforming.triangle_moments import quadratic_triangle_weights
from tests.rcwa.test_nvf_cartesian_pair_opening import _oracle_geometry, _oracle_differentiated_native_solve_reference
from tests.rcwa.test_nvf_cartesian_pair_gap import analytic_material

pytestmark = pytest.mark.research


ALPHA = ((2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (0, 1, 1), (1, 0, 1))


def duffy(order):
    x, w = np.polynomial.legendre.leggauss(order)
    x, w = (x+1)/2, w/2
    r, t = np.meshgrid(x, x, indexing='ij')
    return (np.column_stack([(1-r).ravel(), (r*(1-t)).ravel(), (r*t).ravel()]),
            (np.outer(w, w)*r).ravel())


@pytest.mark.parametrize('gap', [0., 1e-12, .37])
@pytest.mark.parametrize('permutation', list(permutations(range(3))))
def test_quadratic_triangle_integrals_including_collapse_and_repeated_phases(gap, permutation):
    p = np.array([[.2, .1], [.2+gap, .1], [.4, .9]])[list(permutation)]
    v = np.array([[-.5, 0.], [.5, 0.], [.5, 0.]])[list(permutation)]
    random = np.random.default_rng(47)
    # Exact repeated phases, near-confluent phases, the series boundary and
    # oscillatory modes all enter the same predetermined absolute guards.
    wave = np.vstack([np.zeros((1, 2)), np.array([[0., 1.], [2., 0.], [1e-13, 1e-13]]),
                      random.normal(size=(37, 2))*25])
    actual, derivative = quadratic_triangle_weights(p, v, wave)
    b, w = duffy(64)
    polynomial = np.array([2*np.prod(b**a, axis=1)/np.prod([factorial(i) for i in a]) for a in ALPHA])
    j = np.column_stack([p[1]-p[0], p[2]-p[0]])
    dj = np.column_stack([v[1]-v[0], v[2]-v[0]])
    det = np.linalg.det(j)
    slope = j[0, 0]*dj[1, 1]+dj[0, 0]*j[1, 1]-j[0, 1]*dj[1, 0]-dj[0, 1]*j[1, 0]
    phase = np.exp(-1j*b@p@wave.T)
    expected = (polynomial*w)@(det*phase)
    change = (polynomial*w)@(phase*(slope-1j*det*b@v@wave.T))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=3e-12)
    np.testing.assert_allclose(derivative, change, rtol=0, atol=3e-12)
    np.testing.assert_allclose(actual[:, 0], det/12, rtol=0, atol=1e-15)
    np.testing.assert_allclose(derivative[:, 0], slope/12, rtol=0, atol=1e-15)


def nodal_gradients(barycentric):
    nodes = np.array([[0., 0.], [1., 0.], [0., 1.], [.5, 0.], [.5, .5], [0., .5]])
    x, y = nodes.T
    vandermonde = np.column_stack([np.ones(6), x, y, x*x, x*y, y*y])
    coefficients = np.linalg.inv(vandermonde)
    x, y = barycentric[:, 1:].T
    zero, one = np.zeros_like(x), np.ones_like(x)
    dx = np.column_stack([zero, one, zero, 2*x, y, zero])@coefficients
    dy = np.column_stack([zero, zero, one, zero, x, 2*y])@coefficients
    return np.stack([dx, dy], axis=-1)


def refined_solve(matrix, rhs):
    """Correct the original solve defect, using wider arithmetic where available."""
    value = np.linalg.solve(np.asarray(matrix, complex), np.asarray(rhs, complex)).astype(np.clongdouble)
    for _ in range(2):
        residual = rhs-matrix@value
        value += np.linalg.solve(np.asarray(matrix, complex), np.asarray(residual, complex))
    return value


def reference(cell, settings, gap, quadrature):
    geometry = _oracle_geometry(cell.data, settings)
    length = np.sqrt(np.linalg.det(settings['basis']))
    p0 = np.asarray(geometry['vertices0'], float)/length
    speed = np.asarray(geometry['vertices1'], float)
    s = gap/length
    material = np.r_[settings['background'], settings['epsilon']].astype(complex)
    size = len(cell.data['pmap'])
    k = np.zeros((3, size, size), np.clongdouble)
    f = np.zeros((2, size, 2), np.clongdouble)
    b, weights = duffy(3)
    gradients = nodal_gradients(b)
    local = []
    for triangle, tag, thin in zip(cell.data['triangles'], cell.data['tags'], cell.data['thin']):
        vertices, velocity = p0[triangle[:3]], speed[triangle[:3]]
        j0, j1 = (np.column_stack([v[1]-v[0], v[2]-v[0]]) for v in (vertices, velocity))
        # The 2x2 adjugate is linear. Extract it from original invertible
        # Jacobians at two fixed positive gaps, not from candidate blocks.
        ja, jb = j0+j1, j0+2*j1
        aa, ab = np.linalg.det(ja)*np.linalg.inv(ja), np.linalg.det(jb)*np.linalg.inv(jb)
        a0, a1 = 2*aa-ab, ab-aa
        d0 = 0. if thin else np.linalg.det(j0)
        d1 = np.linalg.det(ja)-d0
        n0, n1 = gradients@a0, gradients@a1
        e = material[tag]
        integral = lambda a, b: e*np.einsum('q,qia,qja->ij', weights, a, b)
        load = lambda a: -e*np.einsum('q,qia->ia', weights, a)
        if gap == 0:
            h0, h1 = integral(n0, n0), integral(n0, n1)+integral(n1, n0)
            if thin:
                blocks = [h0/d1, h1/d1, integral(n1, n1)/d1]
            else:
                blocks = [np.zeros((6, 6)), h0/d0, h1/d0-h0*d1/d0**2]
            loads = [load(n0), load(n1)]
        else:
            j = j0+s*j1
            inverse = np.linalg.inv(j)
            grad = gradients@inverse
            dgrad = -grad@j1@inverse
            d = np.linalg.det(j)
            blocks = [np.zeros((6, 6)), d*integral(grad, grad),
                      d1*integral(grad, grad)+d*(integral(dgrad, grad)+integral(grad, dgrad))]
            loads = [d*load(grad), d1*load(grad)+d*load(dgrad)]
        ids = cell.data['periodic'][triangle]
        for i, row in enumerate(ids):
            f[:, row] += np.asarray(loads)[:, i]
            for j, column in enumerate(ids):
                k[:, row, column] += np.asarray(blocks)[:, i, j]
        local.append((d0, d1, a0, a1, vertices, velocity))
    if gap == 0:
        # Successive original Laurent equations determine w0 and w1. No
        # finite-gap rescaled solve, candidate solution or candidate fields.
        c, z = cell.data['Cfull'][:, 1:], cell.data['Zfull']
        a = c.T@k[1]@c
        w0 = c@refined_solve(a, c.T@f[0])
        difference = z@refined_solve(z.T@k[0]@z, z.T@(f[0]-k[1]@w0))
        w1 = c@refined_solve(a, c.T@(f[1]-k[2]@w0-k[1]@difference))+difference
        residuals = [np.linalg.norm(k[0]@w0)/max(np.linalg.norm(f[0]), 1.),
                     np.linalg.norm(k[0]@w1+k[1]@w0-f[0])/max(np.linalg.norm(f[0]), 1.),
                     np.linalg.norm(c.T@(k[1]@w1+k[2]@w0-f[1]))/max(np.linalg.norm(f[1]), 1.)]
    else:
        w0 = np.vstack([np.zeros((1, 2)), refined_solve(k[1, 1:, 1:], f[0, 1:])])
        w1 = np.vstack([np.zeros((1, 2)), refined_solve(k[1, 1:, 1:], (f[1]-k[2]@w0)[1:])])
        residuals = [np.linalg.norm(k[1]@w0-f[0])/max(np.linalg.norm(f[0]), 1.),
                     np.linalg.norm(k[1]@w1+k[2]@w0-f[1])/max(np.linalg.norm(f[1]), 1.)]
    w0, w1 = np.asarray(w0, complex), np.asarray(w1, complex)
    b, weights = duffy(quadrature)
    gradients = nodal_gradients(b)
    ox, oy = cell.order
    n, m = np.meshgrid(np.arange(-2*oy, 2*oy+1), np.arange(-2*ox, 2*ox+1), indexing='ij')
    basis = np.asarray(settings['basis'])/length
    wave = 2*np.pi*np.column_stack([m.ravel(), n.ravel()])@np.linalg.inv(basis)
    value, derivative = np.zeros((2, 9, len(wave)), complex)
    for triangle, tag, thin, (d0, d1, a0, a1, vertices, velocity) in zip(
            cell.data['triangles'], cell.data['tags'], cell.data['thin'], local):
        ids = cell.data['periodic'][triangle]
        n0, n1 = gradients@a0, gradients@a1
        if gap == 0 and thin:
            field = np.eye(2)+(n1.swapaxes(-1, -2)@w0[ids]+n0.swapaxes(-1, -2)@w1[ids])/d1
            change = np.zeros_like(field)
        else:
            d = d0+s*d1
            grad = (n0+s*n1)/d
            dgrad = n1/d-(n0+s*n1)*d1/d**2
            field = np.eye(2)+grad.swapaxes(-1, -2)@w0[ids]
            change = dgrad.swapaxes(-1, -2)@w0[ids]+grad.swapaxes(-1, -2)@w1[ids]
        e = material[tag]
        gram = e*field.conj().swapaxes(-1, -2)@field
        dgram = e*(change.conj().swapaxes(-1, -2)@field+field.conj().swapaxes(-1, -2)@change)
        values = np.array([field[:, 0, 0], field[:, 0, 1], field[:, 1, 0], field[:, 1, 1],
                           gram[:, 0, 0], gram[:, 0, 1], gram[:, 1, 1], np.full(len(b), e), gram[:, 1, 0]])
        changes = np.array([change[:, 0, 0], change[:, 0, 1], change[:, 1, 0], change[:, 1, 1],
                            dgram[:, 0, 0], dgram[:, 0, 1], dgram[:, 1, 1], np.zeros(len(b)), dgram[:, 1, 0]])
        phase = np.exp(-1j*b@(vertices+s*velocity)@wave.T)
        d = d0+s*d1
        value += (values*weights)@(d*phase)
        derivative += (changes*weights)@(d*phase)+(values*weights)@(phase*(d1-1j*d*b@velocity@wave.T))
    shape = (9, 4*oy+1, 4*ox+1)
    return value.reshape(shape), derivative.reshape(shape)/length, residuals


def scene(name):
    r, shift, metal, skew = {
        'square': (0, 0, False, False), 'metal': (0, 0, True, False),
        'skew': (1, 1, False, True), 'skew-metal': (1, 2, True, True),
        'fine-metal': (2, 0, True, False),
    }[name]
    settings = dict(basis=np.array([[700., 140.], [0., 630.]]) if skew else np.diag([700., 700.]),
                    widths=(180., 210.) if skew else (300., 300.),
                    height=210. if skew else 300., offset=(14., -7.) if skew else (0., 0.),
                    epsilon=(12.25, -10.+1j) if metal else (4.+.8j, 9.+.3j), background=1.)
    plan = CartesianPairOpeningPlan.from_geometry(**{k: v for k, v in settings.items()
                                                    if k not in ('epsilon', 'background')}, refinement=r)
    return (CartesianPairQuadraticGapCell(plan, (1, 2),
                                          coercivity_angle=1.53 if metal else 0.), settings)


@lru_cache(None)
def qualified(name, gap):
    cell, settings = scene(name)
    a, da, ra = reference(cell, settings, gap, 24)
    b, db, rb = reference(cell, settings, gap, 32)
    length = np.sqrt(np.linalg.det(settings['basis']))
    np.testing.assert_allclose(a, b, rtol=0, atol=3e-10)
    np.testing.assert_allclose(length*da, length*db, rtol=0, atol=3e-9)
    assert max(*ra, *rb) < 1e-10
    return cell, settings, length, b, db


@pytest.mark.parametrize('name', ['square', 'metal', 'skew', 'skew-metal', 'fine-metal'])
@pytest.mark.parametrize('gap', [0., 7.])
def test_quadratic_finite_gap_against_original_equations(name, gap):
    cell, settings, length, expected, change = qualified(name, gap)
    value, derivative, meta, _ = cell.coefficients(gap=gap, **settings)
    np.testing.assert_allclose(value, expected, rtol=0, atol=3e-10)
    np.testing.assert_allclose(length*derivative, length*change, rtol=0, atol=3e-9)
    slots, dslots, cm, _ = cell.closure(gap=gap, **settings)
    ref, dref = _oracle_differentiated_native_solve_reference(expected, change, cell.order)
    np.testing.assert_allclose(slots, ref, rtol=0, atol=3e-10)
    np.testing.assert_allclose(length*np.stack(dslots), length*dref, rtol=0, atol=3e-9)
    assert max(meta['original_flux_residual'], meta['original_flux_derivative_residual']) < 1e-10
    assert cm['minimum_dissipation'] > -1e-9
    ox, oy = cell.order
    np.testing.assert_allclose(value[:4, 2*oy, 2*ox], [1, 0, 0, 1], rtol=0, atol=3e-12)
    np.testing.assert_allclose(length*derivative[:4, 2*oy, 2*ox], 0, rtol=0, atol=3e-10)
    material, dmaterial = analytic_material(settings, cell.order, gap)
    np.testing.assert_allclose(value[7], material, rtol=0, atol=3e-12)
    np.testing.assert_allclose(length*derivative[7], length*dmaterial, rtol=0, atol=3e-12)


@pytest.mark.parametrize('refinement', range(3))
@pytest.mark.parametrize('shift', range(4))
def test_quadratic_graph_counts_and_immutable_polynomial_collapse(refinement, shift):
    cell = CartesianPairQuadraticGapCell(CartesianPairOpeningPlan(shift, refinement))
    assert cell.data['triangles'].shape == (18*4**refinement, 6)
    assert cell.data['C'].shape[1] == 28*4**refinement-1
    assert cell.data['Z'].shape[1] == 8*4**refinement
    assert len(cell.data['pmap'])-1 == 36*4**refinement-1
    assert all(not v.flags.writeable for v in cell.data.values() if isinstance(v, np.ndarray))


@pytest.mark.parametrize('epsilon,angle', [(3., 0.), (3.+.2j, 0.), (-10.+1j, 1.53)])
@pytest.mark.parametrize('gap', [0., 7., 1e-9])
def test_uniform_material_has_identity_field_and_no_gap_response(epsilon, angle, gap):
    cell = CartesianPairQuadraticGapCell(CartesianPairOpeningPlan(0, 1), (2, 1), coercivity_angle=angle)
    value, derivative, _, _ = cell.coefficients(gap=gap, basis=np.diag([700., 700.]),
                                               widths=(300., 300.), height=300.,
                                               epsilon=(epsilon, epsilon), background=epsilon)
    expected = np.zeros_like(value)
    expected[:, 2, 4] = [1, 0, 0, 1, epsilon, 0, epsilon, epsilon, 0]
    np.testing.assert_allclose(value, expected, rtol=0, atol=3e-11)
    np.testing.assert_allclose(700*derivative, 0, rtol=0, atol=3e-10)


def test_refuses_unsupported_quadratic_refinement():
    with pytest.raises(ValueError, match='quadratic geometric refinement'):
        CartesianPairQuadraticGapCell(CartesianPairOpeningPlan(0, 3))


@pytest.mark.parametrize('gap', [.0005, .001, .002, 1e-9])
def test_fine_metal_small_gaps_preserve_original_flux_residual(gap):
    cell, settings = scene('fine-metal')
    _, _, meta, details = cell.coefficients(gap=gap, **settings)
    assert meta['original_flux_derivative_residual'] < 1e-10
    # The metadata cannot hide a projection that discards the original rows.
    assert np.linalg.norm(details['original_flux_derivative_ds']) < 1e-10


@pytest.mark.parametrize('scale', [1e-4, 1e4])
def test_physical_units_preserve_coefficients_and_scale_gap_derivatives(scale):
    cell, settings, _, expected, derivative = qualified('skew-metal', 7.)
    changed = dict(settings)
    for key in ('basis', 'widths', 'height', 'offset'):
        changed[key] = np.asarray(settings[key])*scale
    actual, tangent, _, _ = cell.coefficients(gap=7.*scale, **changed)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=3e-10)
    np.testing.assert_allclose(700*scale*tangent, 700*derivative, rtol=0, atol=3e-9)


@pytest.mark.parametrize('change', [dict(gap=-1.), dict(epsilon=(4., -10.)),
                                    dict(epsilon=(4., -10.-1j)), dict(height=0.)])
def test_invalid_quadratic_replays_do_not_change_material_or_geometry(change):
    cell, settings = scene('metal')
    settings = dict(settings, gap=0.)
    settings.update(change)
    with pytest.raises(ValueError):
        cell.coefficients(**settings)


def test_coarse_metal_high_fourier_order_against_resolved_independent_quadrature():
    base, settings = scene('metal')
    cell = CartesianPairQuadraticGapCell(base.plan, 8, coercivity_angle=1.53)
    # Coarse triangles at order 8 need a separately resolved Fourier integral;
    # successful low-order comparisons do not qualify these oscillatory modes.
    a, da, ra = reference(cell, settings, 0., 48)
    b, db, rb = reference(cell, settings, 0., 64)
    np.testing.assert_allclose(a, b, rtol=0, atol=3e-10)
    np.testing.assert_allclose(700*da, 700*db, rtol=0, atol=3e-9)
    assert max(*ra, *rb) < 1e-10
    actual, derivative, _, _ = cell.coefficients(gap=0., **settings)
    np.testing.assert_allclose(actual, b, rtol=0, atol=3e-10)
    np.testing.assert_allclose(700*derivative, 700*db, rtol=0, atol=3e-9)
