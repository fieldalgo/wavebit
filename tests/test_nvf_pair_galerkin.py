"""Independent original nodal equations for constrained real Galerkin cells."""
from functools import lru_cache
import numpy as np
from numpy.polynomial.legendre import leggauss
import pytest

from wavebit._core.conforming.cartesian_pair_opening import CartesianPairOpeningPlan
from wavebit._core.conforming.cartesian_pair_galerkin import CartesianPairGalerkinGapCell
from wavebit._core.conforming import cartesian_pair_galerkin as module
from wavebit._core.conforming.triangle_moments import triangle_phase_average, triangle_linear_phase_average
from test_nvf_cartesian_pair_opening import (
    _oracle_geometry, _oracle_original_limit, _oracle_polynomial_vandermonde, POWERS,
)

GEOMETRY = dict(basis=np.array([[700., 65.], [0., 560.]]), widths=(170., 230.), height=280., offset=(21., -14.))
MATERIALS = ((4., 4.), (4.+.8j, 9.+.3j), (12.25, -10.+1j))


def cell(refinement=0, order=(1, 2), shift=0, degree=1):
    return CartesianPairGalerkinGapCell(CartesianPairOpeningPlan(shift, refinement), order,
                                        degree=degree, coercivity_angle=1.53)


def tensor(slots):
    return np.block([[slots[0][0], -slots[3][0]], [-slots[4][0], slots[1][0]]])


def duffy(q):
    u, w = leggauss(q)
    u, w = (u+1)/2, w/2
    r, s = np.meshgrid(u, u, indexing='ij')
    return np.stack([1-r, r*(1-s), r*s], -1).reshape(-1, 3), (w[:, None]*w[None, :]*r).ravel()


@lru_cache(None)
def reference(epsilon, gap, refinement=0, order=(1, 2), quadrature=24, shift=0):
    """Original unscaled nodal/Laurent equations and numerical simplex integrals.

    At contact, solve successive Laurent equations for w0 and w1. The test
    never constructs the candidate C/sZ matrix or its condensed field solve.
    Positive-gap values use an independently projected Schur complement.
    """
    candidate = cell(refinement, order, shift)
    length = np.sqrt(np.linalg.det(GEOMETRY['basis']))
    settings = {k: np.asarray(v)/length for k, v in GEOMETRY.items()}
    if shift:
        settings['basis'] = settings['basis'] @ POWERS[shift]
    g = _oracle_geometry(candidate.data, settings)
    materials = np.array([1., *epsilon], complex)
    bary, weights = duffy(quadrature)
    vertices = np.asarray(g['vertices0'], float)+gap/length*np.asarray(g['vertices1'], float)
    velocity = np.asarray(g['vertices1'], float)
    ox, oy = order
    n, m = np.meshgrid(np.arange(-oy, oy+1), np.arange(-ox, ox+1), indexing='ij')
    n, m = n.ravel(), m.ravel()
    N, dofs = len(n), len(g['pmap'])-1
    wave = 2*np.pi*np.column_stack([m, n])@np.linalg.inv(settings['basis'])
    area_cell = np.linalg.det(settings['basis'])
    q, dq, r, dr = (np.zeros((2*N, dofs), complex) for _ in range(4))
    a, da = np.zeros((N, N), complex), np.zeros((N, N), complex)
    k = np.zeros((dofs, dofs), complex)
    if gap == 0:
        _, _, gradients, kt, _, _ = _oracle_original_limit(g, materials)
        km, k0, k1 = np.tensordot(materials, np.asarray(kt, float), axes=(0, 0))[:, 1:, 1:]/area_cell
    for ie, (triangle, tag) in enumerate(zip(g['triangles'], g['tags'])):
        ids = g['periodic'][triangle]-1
        points, speed = bary@vertices[triangle], bary@velocity[triangle]
        phase = np.exp(-1j*points@wave.T)
        dphase = -1j*(speed@wave.T)*phase
        moment, dmoment = weights@phase, weights@dphase
        if gap == 0:
            d0, d1, grad0, grad1 = gradients[ie]
            if d0:
                product, dproduct = d0*grad0, d1*grad0+d0*grad1
            else:
                product, dproduct = d1*grad0, d1*grad1
            area2, darea2 = d0, d1
        else:
            v = np.column_stack([np.ones(3), vertices[triangle]])
            vi = np.linalg.inv(v)
            grad = vi[1:].T
            area2 = np.linalg.det(v)
            product = area2*grad
            dproduct = np.zeros_like(product)
            darea2 = 0.
            local = materials[tag]*area2/2*(grad@grad.T)/area_cell
            for i in range(3):
                for j in range(3):
                    if min(ids[i], ids[j]) >= 0:
                        k[ids[i], ids[j]] += local[i, j]
        for ii, index in enumerate(ids):
            if index < 0:
                continue
            mixed = (product[ii, :, None]*moment[None, :]/area_cell).ravel()
            dmixed = ((dproduct[ii, :, None]*moment[None, :]
                       +product[ii, :, None]*dmoment[None, :])/area_cell).ravel()
            q[:, index] += mixed; dq[:, index] += dmixed
            r[:, index] += materials[tag]*mixed; dr[:, index] += materials[tag]*dmixed
        gram = phase.T@(weights[:, None]*phase.conj())
        dgram = dphase.T@(weights[:, None]*phase.conj())+phase.T@(weights[:, None]*dphase.conj())
        a += materials[tag]*area2*gram/area_cell
        da += materials[tag]*(darea2*gram+area2*dgram)/area_cell
    reverse = np.r_[np.arange(N-1, -1, -1), np.arange(2*N-1, N-1, -1)]
    source, dsource = r[reverse].T, dr[reverse].T
    A, dA = np.kron(np.eye(2), a), np.kron(np.eye(2), da)
    I = np.eye(2*N)
    if gap == 0:
        C, Z = g['C'], g['Z']
        nc = C.shape[1]
        # Unknowns are independent retained field, original contact potential,
        # and retained physical flux. No candidate block inverse is reused.
        matrix = np.block([[I, q@C, np.zeros_like(I)],
                           [A, r@C, -I],
                           [C.T@source, C.T@k0@C, -C.T@q.conj().T]])
        rhs = np.vstack([I, np.zeros_like(I), np.zeros((nc, 2*N))])
        solved = np.linalg.solve(matrix, rhs)
        x, u, H = solved[:2*N], solved[2*N:2*N+nc], solved[-2*N:]
        w0 = C@u
        v = np.linalg.solve(Z.T@km@Z, Z.T@(q.conj().T@H-k0@w0-source@x))
        particular = Z@v
        drhs = np.vstack([-dq@w0-q@particular,
                          -dA@x-dr@w0-r@particular,
                          -C.T@(k0@particular+k1@w0+dsource@x-dq.conj().T@H)])
        dH = np.linalg.solve(matrix, drhs)[-2*N:]/length
    else:
        Qh = q.conj().T
        B = r-A@q
        D = k-source@q-Qh@r+Qh@A@q
        rhs = source-Qh@A
        potential = -np.linalg.solve(D, rhs)
        x = I-q@potential
        H = A+B@potential
        # Independently integrate the physical corrected-field energy.
        energy = (x.conj().T@A@x+x.conj().T@r@potential
                  +potential.conj().T@source@x+potential.conj().T@k@potential)
        np.testing.assert_allclose(energy, H, atol=3e-10, rtol=0)
        dH = None
    return H, dH, np.linalg.inv(a)


@lru_cache(None)
def quadratic_reference(epsilon, gap, refinement=0, order=(1, 2), quadrature=24, shift=0):
    """Six-node Vandermonde, numerical quadrature, original Laurent equations.

    No production gradient, moment, integral or rescaled weak block is used.
    The shared geometry graph supplies node coordinates and periodic identity.
    """
    candidate = cell(refinement, order, shift, degree=2)
    length = np.sqrt(np.linalg.det(GEOMETRY['basis']))
    settings = {k: np.asarray(v)/length for k, v in GEOMETRY.items()}
    settings['basis'] = settings['basis'] @ POWERS[shift]
    g = _oracle_geometry(candidate.data, settings)
    return nodal_reference(g, epsilon, gap, order, quadrature, length, settings['basis'], degree=2)


def nodal_reference(g, epsilon, gap, order, quadrature, length, basis, *, degree=2, background=1.,
                    gradient_evaluator=None):
    """Original nodal/Laurent calculation for an independently supplied chart."""
    from test_nvf_pair_quadratic import nodal_gradients
    p0, speed = np.asarray(g['vertices0'], float), np.asarray(g['vertices1'], float)
    s = gap/length
    material = np.array([background, *epsilon], complex)
    bary, weights = duffy(quadrature)
    reference_grad = (gradient_evaluator(bary) if gradient_evaluator is not None else
                      nodal_gradients(bary) if degree == 2 else
                      np.broadcast_to(np.array([[-1., -1.], [1., 0.], [0., 1.]]), (len(bary), 3, 2)))
    ox, oy = order
    n, m = np.meshgrid(np.arange(-oy, oy+1), np.arange(-ox, ox+1), indexing='ij')
    n, m = n.ravel(), m.ravel()
    N, dofs = len(n), len(g['pmap'])-1
    wave = 2*np.pi*np.column_stack([m, n])@np.linalg.inv(basis)
    q, dq, r, dr = (np.zeros((2*N, dofs), complex) for _ in range(4))
    a, da = np.zeros((N, N), complex), np.zeros((N, N), complex)
    km, k0, k1 = (np.zeros((dofs, dofs), complex) for _ in range(3))
    for triangle, tag, thin in zip(g['triangles'], g['tags'], g['thin']):
        p, v = p0[triangle[:3]], speed[triangle[:3]]
        j0, j1 = (np.column_stack([t[1]-t[0], t[2]-t[0]]) for t in (p, v))
        # Rational Vandermonde cofactors avoid subtracting rounded inverse
        # Jacobians to infer the first Laurent coefficient on fine triangles.
        det, numerator = _oracle_polynomial_vandermonde(
            g['vertices0'][triangle[:3]], g['vertices1'][triangle[:3]])
        a0, a1 = np.asarray(numerator[:, 1:, :], float)
        n0, n1 = reference_grad@a0, reference_grad@a1
        d0, d1 = map(float, det)
        integral = lambda u, v: material[tag]*np.einsum('q,qia,qja->ij', weights, u, v)
        if gap == 0:
            cross = integral(n0, n1)+integral(n1, n0)
            blocks = ([integral(n0, n0)/d1, cross/d1, integral(n1, n1)/d1] if thin else
                      [np.zeros((len(triangle), len(triangle))), integral(n0, n0)/d0,
                       cross/d0-integral(n0, n0)*d1/d0**2])
            product, change = n0, n1
        else:
            jac = j0+s*j1
            grad = reference_grad@np.linalg.inv(jac)
            blocks = [np.zeros((len(triangle), len(triangle))), np.linalg.det(jac)*integral(grad, grad), np.zeros((len(triangle), len(triangle)))]
            product, change = np.linalg.det(jac)*grad, np.zeros_like(grad)
        phase = np.exp(-1j*(bary@(p+s*v))@wave.T)
        dphase = -1j*((bary@v)@wave.T)*phase
        mixed = np.einsum('q,qia,qm->ami', weights, product, phase).reshape(2*N, len(triangle))
        dmixed = (np.einsum('q,qia,qm->ami', weights, change, phase)
                  +np.einsum('q,qia,qm->ami', weights, product, dphase)).reshape(2*N, len(triangle))
        ids = g['periodic'][triangle]-1
        for i, row in enumerate(ids):
            if row < 0:
                continue
            q[:, row] += mixed[:, i]; dq[:, row] += dmixed[:, i]
            r[:, row] += material[tag]*mixed[:, i]; dr[:, row] += material[tag]*dmixed[:, i]
            for j, column in enumerate(ids):
                if column >= 0:
                    for target, block in zip((km, k0, k1), blocks):
                        target[row, column] += block[i, j]
        gram = phase.T@(weights[:, None]*phase.conj())
        dgram = dphase.T@(weights[:, None]*phase.conj())+phase.T@(weights[:, None]*dphase.conj())
        a += material[tag]*(d0+s*d1)*gram
        da += material[tag]*(d1*gram+(d0+s*d1)*dgram)
    reverse = np.r_[np.arange(N-1, -1, -1), np.arange(2*N-1, N-1, -1)]
    source, dsource = r[reverse].T, dr[reverse].T
    A, dA, I = np.kron(np.eye(2), a), np.kron(np.eye(2), da), np.eye(2*N)
    if gap == 0:
        C, Z = g['C'], g['Z']
        nc = C.shape[1]
        original = np.block([[I, q@C, np.zeros_like(I)], [A, r@C, -I],
                             [C.T@source, C.T@k0@C, -C.T@q.conj().T]])
        rhs = np.vstack([I, np.zeros_like(I), np.zeros((nc, 2*N))])
        solved = np.linalg.solve(original, rhs)
        x, u, H = solved[:2*N], solved[2*N:2*N+nc], solved[-2*N:]
        w0 = C@u
        particular = Z@np.linalg.solve(Z.T@km@Z, Z.T@(q.conj().T@H-k0@w0-source@x))
        drhs = np.vstack([-dq@w0-q@particular, -dA@x-dr@w0-r@particular,
                          -C.T@(k0@particular+k1@w0+dsource@x-dq.conj().T@H)])
        dH = np.linalg.solve(original, drhs)[-2*N:]/length
    else:
        # A coupled retained/flux system uses the unprojected nodal inverse,
        # independently of the production potential Schur complement.
        u, v = np.split(np.linalg.solve(k0, np.hstack([q.conj().T, source])), 2, axis=1)
        original = np.block([[I-q@v, q@u], [-A+r@v, I-r@u]])
        x, H = np.split(np.linalg.solve(original, np.vstack([I, np.zeros_like(I)])), 2, axis=0)
        potential = u@H-v@x
        energy = (x.conj().T@A@x+x.conj().T@r@potential
                  +potential.conj().T@source@x+potential.conj().T@k0@potential)
        np.testing.assert_allclose(energy, H, atol=3e-10, rtol=0)
        dH = None
    return H, dH, np.linalg.inv(a)


@pytest.mark.parametrize('degree', (1, 2))
@pytest.mark.parametrize('epsilon', MATERIALS)
@pytest.mark.parametrize('gap', (0., 7.))
@pytest.mark.parametrize('refinement', (0, 1))
def test_original_nodal_and_contact_laurent_reference(epsilon, gap, refinement, degree):
    oracle = reference if degree == 1 else quadratic_reference
    H16, dH16, z16 = oracle(epsilon, gap, refinement, quadrature=16)
    H24, dH24, z24 = oracle(epsilon, gap, refinement)
    np.testing.assert_allclose(H16, H24, atol=3e-10, rtol=0)
    np.testing.assert_allclose(z16, z24, atol=3e-10, rtol=0)
    slots, derivative, meta = cell(refinement, degree=degree).closure(gap=gap, epsilon=epsilon, **GEOMETRY)
    np.testing.assert_allclose(tensor(slots), H24, atol=3e-10, rtol=0)
    np.testing.assert_allclose(slots.epz_i[0], z24, atol=3e-10, rtol=0)
    if gap == 0:
        L = np.sqrt(np.linalg.det(GEOMETRY['basis']))
        np.testing.assert_allclose(L*dH16, L*dH24, atol=3e-9, rtol=0)
        np.testing.assert_allclose(L*tensor(derivative), L*dH24, atol=3e-9, rtol=0)
    assert max(v for k, v in meta.items() if k.endswith('residual')) < 1e-10
    assert meta['relative_reciprocity_defect'] < 1e-10
    assert meta['minimum_dissipation'] > -1e-9


@pytest.mark.parametrize('shift', (1, 2, 3))
@pytest.mark.parametrize('degree', (1, 2))
def test_other_frozen_charts_and_directional_orders(shift, degree):
    order, epsilon = (2, 1), MATERIALS[-1]
    expected, change, _ = (reference if degree == 1 else quadratic_reference)(epsilon, 0., order=order, shift=shift)
    geometry = dict(GEOMETRY, basis=GEOMETRY['basis'] @ POWERS[shift])
    slots, derivative, _ = cell(order=order, shift=shift, degree=degree).closure(gap=0., epsilon=epsilon, **geometry)
    L = np.sqrt(np.linalg.det(GEOMETRY['basis']))
    np.testing.assert_allclose(tensor(slots), expected, rtol=0, atol=3e-10)
    np.testing.assert_allclose(L*tensor(derivative), L*change, rtol=0, atol=3e-9)


@pytest.mark.parametrize('gap', (0., 7., 1e-9))
@pytest.mark.parametrize('epsilon', (4., 4.+.3j, -10.+1j))
@pytest.mark.parametrize('degree', (1, 2))
def test_uniform_medium_and_its_first_material_variation(gap, epsilon, degree):
    obj = cell(degree=degree)
    slots, derivative, _ = obj.closure(gap=gap, epsilon=(epsilon, epsilon), background=epsilon,
                                       gap_tangent=0., epsilon_tangent=(1., 1.), background_tangent=1., **GEOMETRY)
    I = np.eye(15)
    expected = np.array([epsilon*I, epsilon*I, I/epsilon, 0*I, 0*I])[:, None]
    dexpected = np.array([I, I, -I/epsilon**2, 0*I, 0*I])[:, None]
    np.testing.assert_allclose(slots, expected, atol=3e-10, rtol=0)
    np.testing.assert_allclose(derivative, dexpected, atol=3e-10, rtol=0)


@pytest.mark.parametrize('gap', (0., 7., 1e-9))
@pytest.mark.parametrize('degree', (1, 2))
def test_material_jvp_is_complex_linear_and_agrees_with_two_secants(gap, degree):
    obj, epsilon = cell(degree=degree), (12.25+.2j, -10.+1j)
    direction = np.array([.3+.2j, -.4+.1j])
    _, tangent, _ = obj.closure(gap=gap, epsilon=epsilon, background=1.+.1j, gap_tangent=0.,
                               epsilon_tangent=direction, background_tangent=.2j, **GEOMETRY)
    _, rotated, _ = obj.closure(gap=gap, epsilon=epsilon, background=1.+.1j, gap_tangent=0.,
                               epsilon_tangent=1j*direction, background_tangent=-.2, **GEOMETRY)
    np.testing.assert_allclose(rotated, 1j*np.asarray(tangent), atol=3e-10, rtol=0)
    # Fourth-order differences resolve the material curvature without making
    # the step so small that subtraction masks the derivative comparison.
    for step in (.002, .001):
        values = [np.asarray(obj.closure(gap=gap, epsilon=np.asarray(epsilon)+sign*step*direction,
                                         background=1+.1j+sign*step*.2j, gap_tangent=0., **GEOMETRY)[0])
                  for sign in (-2, -1, 1, 2)]
        secant = (values[0]-8*values[1]+8*values[2]-values[3])/(12*step)
        np.testing.assert_allclose(secant, tangent, atol=3e-9, rtol=0)


@pytest.mark.parametrize('degree', (1, 2))
def test_gap_and_material_direction_are_additive_and_match_positive_gap_secants(degree):
    obj, settings = cell(degree=degree), dict(gap=7., epsilon=MATERIALS[-1], **GEOMETRY)
    _, dg, _ = obj.closure(**settings)
    _, dm, _ = obj.closure(**settings, gap_tangent=0., epsilon_tangent=(.2, .3j))
    _, both, _ = obj.closure(**settings, epsilon_tangent=(.2, .3j))
    np.testing.assert_allclose(both, np.asarray(dg)+dm, atol=3e-10, rtol=0)
    for step in (.002, .001):
        values = [obj.closure(**dict(settings, gap=7+sign*step), gap_tangent=0.)[0] for sign in (-1, 1)]
        np.testing.assert_allclose((np.asarray(values[1])-values[0])/(2*step), dg, atol=3e-9, rtol=0)


@pytest.mark.parametrize('degree', (1, 2))
def test_subpixel_gap_approaches_contact_with_the_predicted_slope(degree):
    obj, epsilon = cell(refinement=2, degree=degree), MATERIALS[-1]
    base, slope, _ = obj.closure(gap=0., epsilon=epsilon, **GEOMETRY)
    errors = []
    for g in (.02, .01, .005):
        value, _, _ = obj.closure(gap=g, epsilon=epsilon, **GEOMETRY)
        errors.append(np.max(np.abs(np.asarray(value)-base-g*np.asarray(slope))))
    assert errors[1] < .3*errors[0]+3e-10
    assert errors[2] < .3*errors[1]+3e-10
    small, _, _ = obj.closure(gap=1e-9, epsilon=epsilon, **GEOMETRY)
    np.testing.assert_allclose(small, np.asarray(base)+1e-9*np.asarray(slope), atol=3e-10, rtol=0)


@pytest.mark.parametrize('scale', (1e-4, 1e4))
@pytest.mark.parametrize('degree', (1, 2))
def test_physical_unit_covariance(scale, degree):
    a, da, _ = cell(degree=degree).closure(gap=7., epsilon=MATERIALS[-1], **GEOMETRY)
    b, db, _ = cell(degree=degree).closure(gap=7.*scale, epsilon=MATERIALS[-1],
                             **{k: np.asarray(v)*scale for k, v in GEOMETRY.items()})
    np.testing.assert_allclose(a, b, atol=3e-10, rtol=0)
    np.testing.assert_allclose(da, np.asarray(db)*scale, atol=3e-10, rtol=0)


@pytest.mark.parametrize('gap', (0., 1e-12, .2))
def test_triangle_average_has_a_finite_collapsed_derivative(gap):
    p = np.array([[.1, .2], [.1+gap, .2], [.4, .9]])
    speed = np.array([[0., 0.], [1., 0.], [.2, -.1]])
    wave = np.array([[0., 0.], [0., 4.], [4., 0.], [8., 11.], [1e-12, 2e-12]])
    got, change = triangle_phase_average(p, speed, wave)
    b, w = duffy(32)
    phase = np.exp(-1j*(b@p)@wave.T)
    np.testing.assert_allclose(got, 2*w@phase, atol=3e-12, rtol=0)
    np.testing.assert_allclose(change, 2*w@(-1j*((b@speed)@wave.T)*phase), atol=3e-12, rtol=0)


def test_failed_original_solve_is_not_repaired_by_a_reciprocity_projection(monkeypatch):
    solve = module._solve
    def corrupt(matrix, rhs):
        result = solve(matrix, rhs)
        result[0, 0] += .01
        return result
    monkeypatch.setattr(module, '_solve', corrupt)
    with pytest.raises(ValueError, match='guards'):
        cell().closure(gap=0., epsilon=MATERIALS[-1], **GEOMETRY)


@pytest.mark.parametrize('update', ({'gap': -1.}, {'epsilon': (0., 4.)},
                                     {'epsilon': (4., -10.)}, {'epsilon': (4., 3.-.1j)},
                                     {'epsilon_tangent': (1.,)}, {'background_tangent': np.nan}))
def test_invalid_replay_is_refused(update):
    with pytest.raises(ValueError):
        cell().closure(**dict(dict(gap=0., epsilon=MATERIALS[-1], **GEOMETRY), **update))


@pytest.mark.parametrize('refinement', (2, 3))
def test_finer_contact_spaces_retain_the_original_laurent_derivative(refinement):
    order, epsilon = (1, 0), MATERIALS[-1]
    h16, d16, _ = reference(epsilon, 0., refinement, order, 16)
    h24, d24, _ = reference(epsilon, 0., refinement, order, 24)
    slots, derivative, _ = cell(refinement, order).closure(gap=0., epsilon=epsilon, **GEOMETRY)
    L = np.sqrt(np.linalg.det(GEOMETRY['basis']))
    np.testing.assert_allclose(h16, h24, atol=3e-10, rtol=0)
    np.testing.assert_allclose(tensor(slots), h24, atol=3e-10, rtol=0)
    np.testing.assert_allclose(L*d16, L*d24, atol=3e-9, rtol=0)
    np.testing.assert_allclose(L*tensor(derivative), L*d24, atol=3e-9, rtol=0)


@pytest.mark.parametrize('gap', (0., 7.))
@pytest.mark.parametrize('degree', (1, 2))
def test_nested_real_spaces_lower_the_lossless_energy_and_retain_scalar_bounds(gap, degree):
    previous = None
    for refinement in range(3):
        slots, _, meta = cell(refinement, degree=degree).closure(gap=gap, epsilon=(4., 9.), **GEOMETRY)
        current = tensor(slots)
        assert meta['minimum_rotated_storage'] >= np.cos(1.53)-1e-9
        if previous is not None:
            difference = previous-current
            np.testing.assert_allclose(difference, difference.conj().T, atol=3e-10, rtol=0)
            assert np.linalg.eigvalsh(difference)[0] > -1e-9
        previous = current


@pytest.mark.parametrize('degree', (1, 2))
def test_first_material_variation_about_uniformity_is_the_direct_physical_transform(degree):
    from test_nvf_cartesian_pair_gap import analytic_material
    epsilon, direction, bg_direction = 4.+.3j, (.3+.2j, -.4j), .2+.1j
    settings = dict(GEOMETRY, epsilon=direction, background=bg_direction)
    e, _ = analytic_material(settings, (1, 2), 7.)
    n, m = np.meshgrid(np.arange(-2, 3), np.arange(-1, 2), indexing='ij')
    dE = e[n.ravel()[:, None]-n.ravel()+4, m.ravel()[:, None]-m.ravel()+2]
    _, derivative, _ = cell(degree=degree).closure(gap=7., epsilon=(epsilon, epsilon), background=epsilon,
                                      gap_tangent=0., epsilon_tangent=direction,
                                      background_tangent=bg_direction, **GEOMETRY)
    expected = np.array([dE, dE, -dE/epsilon**2, 0*dE, 0*dE])[:, None]
    np.testing.assert_allclose(derivative, expected, atol=3e-10, rtol=0)


@pytest.mark.parametrize('degree', (1, 2))
def test_certificate_angle_changes_no_physical_equation(degree):
    first = cell(degree=degree).closure(gap=0., epsilon=MATERIALS[-1], **GEOMETRY)
    other = CartesianPairGalerkinGapCell(cell(degree=degree).plan, (1, 2), degree=degree, coercivity_angle=1.50)
    second = other.closure(gap=0., epsilon=MATERIALS[-1], **GEOMETRY)
    np.testing.assert_array_equal(first[:2], second[:2])


def test_batched_triangle_means_keep_collapsed_and_regular_moving_nodes():
    points = np.array([[[.1, .2], [.1, .2], [.4, .9]],
                       [[.4, .9], [.1, .2], [.3, .2]],
                       [[-.3, .1], [.8, -.1], [.2, .7]]])
    velocity = np.array([[[0., 0.], [1., 0.], [.2, -.1]],
                         [[.2, -.1], [0., 0.], [1., 0.]],
                         [[0., .2], [.3, -.1], [.2, 0.]]])
    wave = np.array([[0., 0.], [0., 4.], [4., 0.], [8., 11.], [1e-12, 2e-12]])
    got, change = triangle_phase_average(points, velocity, wave)
    b, w = duffy(32)
    for i, (p, v) in enumerate(zip(points, velocity)):
        phase = np.exp(-1j*(b@p)@wave.T)
        np.testing.assert_allclose(got[i], 2*w@phase, atol=3e-12, rtol=0)
        np.testing.assert_allclose(change[i], 2*w@(-1j*((b@v)@wave.T)*phase), atol=3e-12, rtol=0)


@pytest.mark.parametrize('gap', (0., 1e-12, .2))
def test_linear_triangle_means_and_repeated_node_derivatives(gap):
    p = np.array([[.1, .2], [.1+gap, .2], [.4, .9]])
    speed = np.array([[0., 0.], [1., 0.], [.2, -.1]])
    wave = np.array([[0., 0.], [0., 4.], [4., 0.], [8., 11.], [1e-12, 2e-12]])
    points = np.stack([p, p[[2, 0, 1]], p[[1, 0, 2]]])
    velocity = np.stack([speed, speed[[2, 0, 1]], speed[[1, 0, 2]]])
    got, change = triangle_linear_phase_average(points, velocity, wave)
    b, w = duffy(32)
    for i, (p, v) in enumerate(zip(points, velocity)):
        phase = np.exp(-1j*(b@p)@wave.T)
        np.testing.assert_allclose(got[i], (2*b.T*w)@phase, atol=3e-12, rtol=0)
        np.testing.assert_allclose(change[i], (2*b.T*w)@(-1j*((b@v)@wave.T)*phase), atol=3e-12, rtol=0)
    scalar, dscalar = triangle_phase_average(points, velocity, wave)
    np.testing.assert_allclose(got.sum(-2), scalar, atol=3e-12, rtol=0)
    np.testing.assert_allclose(change.sum(-2), dscalar, atol=3e-12, rtol=0)


@pytest.mark.parametrize('refinement', (2, 3))
def test_finest_quadratic_contact_retains_the_original_laurent_derivative(refinement):
    order, epsilon = (1, 0), MATERIALS[-1]
    h16, d16, _ = quadratic_reference(epsilon, 0., refinement, order, 16)
    h24, d24, _ = quadratic_reference(epsilon, 0., refinement, order, 24)
    slots, derivative, _ = cell(refinement, order, degree=2).closure(gap=0., epsilon=epsilon, **GEOMETRY)
    L = np.sqrt(np.linalg.det(GEOMETRY['basis']))
    np.testing.assert_allclose(h16, h24, atol=3e-10, rtol=0)
    np.testing.assert_allclose(tensor(slots), h24, atol=3e-10, rtol=0)
    np.testing.assert_allclose(L*d16, L*d24, atol=3e-9, rtol=0)
    np.testing.assert_allclose(L*tensor(derivative), L*d24, atol=3e-9, rtol=0)


@pytest.mark.parametrize('gap', (0., 7.))
def test_quadratic_enrichment_lowers_lossless_energy_on_the_same_geometry(gap):
    for r in range(3):
        linear = tensor(cell(r).closure(gap=gap, epsilon=(4., 9.), **GEOMETRY)[0])
        quadratic = tensor(cell(r, degree=2).closure(gap=gap, epsilon=(4., 9.), **GEOMETRY)[0])
        assert np.linalg.eigvalsh(linear-quadratic)[0] > -1e-9


@pytest.mark.parametrize('degree,refinement', ((0, 0), (3, 0), (1.5, 0), (True, 0), (2, 4)))
def test_unsupported_potential_spaces_are_refused(degree, refinement):
    with pytest.raises(ValueError):
        cell(refinement, degree=degree)


@pytest.mark.parametrize('degree', (1, 2))
def test_local_assembly_never_expands_a_triangle_to_global_nodal_width(monkeypatch, degree):
    original = module._inner
    widths = []
    def observed(left, right):
        widths.append(left.shape[-1])
        return original(left, right)
    monkeypatch.setattr(module, '_inner', observed)
    obj = cell(2, degree=degree)
    obj.closure(gap=0., epsilon=MATERIALS[-1], **GEOMETRY)
    assert max(widths) <= (3 if degree == 1 else 6)


@pytest.mark.parametrize('degree', (1, 2))
@pytest.mark.parametrize('gap', (0., 7., 1e-9))
def test_both_eliminations_preserve_the_original_material_and_gap_response(degree, gap):
    obj = cell(1, degree=degree)
    args = dict(gap=gap, epsilon=MATERIALS[-1], background=1., gap_tangent=.7,
                epsilon_tangent=(.2j, .3), background_tangent=.1j, **GEOMETRY)
    k, dk, q, dq, r, dr, qt, dqt, s, ds, e, de, info = obj._assemble(**args)
    a, da = np.kron(np.eye(2), e), np.kron(np.eye(2), de)
    responses = [module._response(k, dk, q, dq, r, dr, qt, dqt, s, ds, a, da, form)
                 for form in ('potential', 'retained')]
    np.testing.assert_allclose(responses[0][4], responses[1][4], atol=3e-10, rtol=0)
    np.testing.assert_allclose(info['length']*responses[0][5], info['length']*responses[1][5],
                               atol=3e-9, rtol=0)
    for p, dp, x, dx, h, dh, _ in responses:
        residual, derivative_residual = obj._original_residual(x, dx, h, dh, p, dp, info)
        assert max(residual, derivative_residual) < 1e-10
