"""Independent moving-triangle weak derivatives for the bounded P1 study.

The oracle shares only the frozen finite mesh definition. It separately
assembles and differentiates original nodal weak equations, then integrates
moving Fourier moments with Duffy quadrature instead of polygon transforms.
"""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.linalg import lu_factor, lu_solve

from studies.rcwa.nvf_cartesian_cell.nvf_cartesian_cell import CartesianRectangleCell, CartesianRectanglePlan

pytestmark = pytest.mark.research

LENGTH = 700.
PARAMETERS = np.array([1., .45, 0., .7, .5, 3/7, 0., .07, -.03, 4., .8, 1., .1])


def settings(p, xp=np):
    p = xp.asarray(p)
    return dict(basis=LENGTH*p[:4].reshape(2, 2), width=LENGTH*p[4], height=LENGTH*p[5],
                angle=p[6], offset=LENGTH*p[7:9], epsilon=p[9]+1j*p[10],
                background=p[11]+1j*p[12], xp=xp)


def geometry_plan(p):
    physical = settings(p)
    return CartesianRectanglePlan.from_geometry(**{
        key: physical[key] for key in ('basis', 'width', 'height', 'angle', 'offset')})


def rectangle_coefficients(p, order):
    """Independent Cartesian sinc product at physical reciprocal vectors."""
    ox, oy = order
    n, m = np.meshgrid(np.arange(-2*oy, 2*oy+1), np.arange(-2*ox, 2*ox+1), indexing='ij')
    basis = LENGTH*p[:4].reshape(2, 2)
    wave = 2*np.pi*np.stack([m, n], axis=-1)@np.linalg.inv(basis)
    center = basis.sum(axis=1)/2+LENGTH*p[7:9]
    angle = p[6]
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    local = wave@rotation
    width, height = LENGTH*p[4:6]
    indicator = (width*height/np.linalg.det(basis)*np.exp(-1j*(wave@center))
                 *np.sinc(local[..., 0]*width/(2*np.pi))*np.sinc(local[..., 1]*height/(2*np.pi)))
    return indicator, (m == 0) & (n == 0)


def matrix_blocks(c, order, cap):
    ox, oy = order
    n, m = np.meshgrid(np.arange(-oy, oy+1), np.arange(-ox, ox+1), indexing='ij')
    d = c[:, n.ravel()[:, None]-n.ravel()+cap[1], m.ravel()[:, None]-m.ravel()+cap[0]]
    return (np.block([[d[0], d[1]], [d[2], d[3]]]),
            np.block([[d[4], d[5]], [d[8], d[6]]]), d[7])


def geometry_variation(parameters, direction, fixed):
    """Independent explicit physical vertices and their directional derivatives."""
    p, dp = np.asarray(parameters, float), np.asarray(direction, float)
    a, da = LENGTH*p[:4].reshape(2, 2), LENGTH*dp[:4].reshape(2, 2)
    offset, doffset = LENGTH*p[7:9], LENGTH*dp[7:9]
    unit = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    outer, douter = unit@a.T+offset, unit@da.T+doffset
    signs = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])
    local, dlocal = signs*(LENGTH*p[4:6]), signs*(LENGTH*dp[4:6])
    angle = p[6]
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    drotation = dp[6]*np.array([[-np.sin(angle), -np.cos(angle)], [np.cos(angle), -np.sin(angle)]])
    inner = local@rotation.T+a.sum(axis=1)/2+offset
    dinner = dlocal@rotation.T+local@drotation.T+da.sum(axis=1)/2+doffset
    vertices = fixed._weights@np.concatenate([outer, inner])
    dvertices = fixed._weights@np.concatenate([douter, dinner])
    return a, da, vertices, dvertices


def oracle(parameters, direction, fixed, quadrature):
    """Differentiate original element assembly and use independent Duffy moments.

    No production assembly, balancing, polygon FT or autodiff is used here.
    Sharing the frozen mesh plan identifies the finite space being tested.
    """
    p, dp = np.asarray(parameters), np.asarray(direction)
    a, da, vertices, dvertices = geometry_variation(p, dp, fixed)
    area_cell = np.linalg.det(a)
    dlog_cell = np.trace(np.linalg.solve(a, da))
    material = np.where(fixed._inclusion, p[9]+1j*p[10], p[11]+1j*p[12])
    dmaterial = np.where(fixed._inclusion, dp[9]+1j*dp[10], dp[11]+1j*dp[12])
    count = (fixed.degrees_of_freedom+1)
    matrix, dmatrix = np.zeros((count, count), complex), np.zeros((count, count), complex)
    right, dright = np.zeros((count, 2), complex), np.zeros((count, 2), complex)
    areas, dareas, gradients, dgradients = [], [], [], []
    for cell, eps, deps in zip(fixed._cells, material, dmaterial):
        x, dx = vertices[cell], dvertices[cell]
        jac = np.column_stack([x[1]-x[0], x[2]-x[0]])
        djac = np.column_stack([dx[1]-dx[0], dx[2]-dx[0]])
        inverse = np.linalg.inv(jac)
        area = np.linalg.det(jac)/2
        if area <= 0:
            raise ValueError('oracle requires positive original triangles')
        darea = area*np.trace(inverse@djac)
        gradient = np.array([[-1., -1.], [1., 0.], [0., 1.]])@inverse
        dgradient = -gradient@djac@inverse
        weight, dweight = eps*area, deps*area+eps*darea
        local = weight*(gradient@gradient.T)
        dlocal = dweight*(gradient@gradient.T)+weight*(dgradient@gradient.T+gradient@dgradient.T)
        load, dload = -weight*gradient, -dweight*gradient-weight*dgradient
        ids = fixed._periodic[cell]
        # Separate original nodal accumulation from the production restriction.
        for i in range(3):
            right[ids[i]] += load[i]
            dright[ids[i]] += dload[i]
            for j in range(3):
                matrix[ids[i], ids[j]] += local[i, j]
                dmatrix[ids[i], ids[j]] += dlocal[i, j]
        areas.append(area)
        dareas.append(darea)
        gradients.append(gradient)
        dgradients.append(dgradient)
    factor = lu_factor(matrix[1:, 1:])
    solution, dsolution = np.zeros((count, 2), complex), np.zeros((count, 2), complex)
    solution[1:] = lu_solve(factor, right[1:])
    derivative_right = dright-dmatrix@solution
    dsolution[1:] = lu_solve(factor, derivative_right[1:])
    residual = np.linalg.norm(matrix@solution-right)/max(np.linalg.norm(right), 1.)
    derivative_residual = np.linalg.norm(matrix@dsolution-derivative_right)/max(np.linalg.norm(derivative_right), 1.)
    assert max(residual, derivative_residual) <= 1e-10
    gradient, dgradient = np.array(gradients), np.array(dgradients)
    local = solution[fixed._periodic[fixed._cells]]
    dlocal = dsolution[fixed._periodic[fixed._cells]]
    field = np.eye(2)+np.einsum('eia,eic->eac', gradient, local)
    dfield = (np.einsum('eia,eic->eac', dgradient, local)
              +np.einsum('eia,eic->eac', gradient, dlocal))
    dual, ddual = field.conj().transpose(0, 2, 1), dfield.conj().transpose(0, 2, 1)
    gram = material[:, None, None]*(dual@field)
    dgram = dmaterial[:, None, None]*(dual@field)+material[:, None, None]*(ddual@field+dual@dfield)
    def components(f, g, e):
        return np.stack([f[:, 0, 0], f[:, 0, 1], f[:, 1, 0], f[:, 1, 1],
                         g[:, 0, 0], g[:, 0, 1], g[:, 1, 1], e, g[:, 1, 0]])
    values, dvalues = components(field, gram, material), components(dfield, dgram, dmaterial)
    q, w = leggauss(quadrature)
    q, w = (q+1)/2, w/2
    r, s = np.meshgrid(q, q, indexing='ij')
    weights = (w[:, None]*w[None, :]*r).ravel()
    barycentric = np.stack([1-r, r*(1-s), r*s], axis=-1).reshape(-1, 3)
    n, m = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3), indexing='ij')
    wave = 2*np.pi*np.column_stack([m.ravel(), n.ravel()])@np.linalg.inv(a)
    dwave = -wave@da@np.linalg.inv(a)
    transforms, dtransforms = [], []
    # Small independent point batches bound memory during the concurrent FEM run.
    for first in range(0, len(fixed._cells), 8):
        selection = slice(first, first+8)
        cells = fixed._cells[selection]
        points = np.einsum('qi,eia->eqa', barycentric, vertices[cells])
        dpoints = np.einsum('qi,eia->eqa', barycentric, dvertices[cells])
        phase = np.exp(-1j*np.einsum('eqa,ka->eqk', points, wave))
        dphase = -1j*phase*(np.einsum('eqa,ka->eqk', dpoints, wave)
                            +np.einsum('eqa,ka->eqk', points, dwave))
        triangle_area = np.array(areas)[selection]
        value = 2*triangle_area[:, None]/area_cell*np.einsum('q,eqk->ek', weights, phase)
        derivative = value*(np.array(dareas)[selection]/triangle_area-dlog_cell)[:, None]
        derivative += 2*triangle_area[:, None]/area_cell*np.einsum('q,eqk->ek', weights, dphase)
        transforms.append(value)
        dtransforms.append(derivative)
    transforms, dtransforms = np.concatenate(transforms), np.concatenate(dtransforms)
    projected = values@transforms
    derivative = dvalues@transforms+values@dtransforms
    return projected.reshape(9, 5, 5), derivative.reshape(9, 5, 5), dict(
        linear_residual=float(residual), derivative_residual=float(derivative_residual),
        factorizations=1, inverse_applications=2)


@pytest.mark.parametrize('refinement', [0, 1, 2])
def test_physical_partition_periodicity_and_independent_variation(refinement):
    cell = CartesianRectangleCell(refinement=refinement)
    p = PARAMETERS
    direction = np.array([.05, -.09, .04, .03, .06, -.05, .11, .07, -.08, .2, -.1, .09, .03])
    actual, metadata = cell.coefficients(**settings(p))
    low, dlow, _ = oracle(p, direction, cell, 16)
    high, dhigh, _ = oracle(p, direction, cell, 24)
    np.testing.assert_allclose(high, low, rtol=0, atol=3e-10)
    np.testing.assert_allclose(dhigh, dlow, rtol=0, atol=3e-10)
    np.testing.assert_allclose(actual, high, rtol=0, atol=3e-10)
    assert metadata['degrees_of_freedom'] == (4, 19, 79)[refinement]
    assert metadata['elements'] == 10*4**refinement
    assert metadata['minimum_signed_area'] > 0
    np.testing.assert_allclose([metadata['area'], metadata['inclusion_area']],
                               [343000., 105000.], rtol=2e-12, atol=0)
    a, _, vertices, _ = geometry_variation(p, np.zeros(13), cell)
    fractional = (vertices-LENGTH*p[7:9])@np.linalg.inv(a).T
    for axis in range(2):
        lower = np.flatnonzero(abs(fractional[:, axis]) < 1e-12)
        assert len(lower) == 2**refinement+1
        for i in lower:
            distance = np.linalg.norm(vertices-vertices[i]-a[:, axis], axis=1)
            j = np.argmin(distance)
            assert distance[j] < 2e-12*LENGTH
            assert cell._periodic[i] == cell._periodic[j]
    # Rectangle edges remain Cartesian while the lattice is sheared.
    inner = geometry_variation(p, np.zeros(13), CartesianRectangleCell())[2][4:8]
    np.testing.assert_allclose(inner[1]-inner[0], [350., 0.], rtol=0, atol=1e-12)
    np.testing.assert_allclose(inner[2]-inner[1], [0., 300.], rtol=0, atol=1e-12)
    indicator, delta = rectangle_coefficients(p, (1, 1))
    eps, bg = p[9]+1j*p[10], p[11]+1j*p[12]
    np.testing.assert_allclose(actual[7], bg*delta+(eps-bg)*indicator, rtol=0, atol=2e-12)


def test_numpy_translation_scale_and_joint_rotation_covariance():
    cell = CartesianRectangleCell(refinement=1)
    p = PARAMETERS.copy()
    original, _ = cell.coefficients(**settings(p))
    translated = p.copy()
    translated[7:9] += [.13, -.07]
    actual, _ = cell.coefficients(**settings(translated))
    n, m = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3), indexing='ij')
    shift = np.linalg.solve(p[:4].reshape(2, 2), translated[7:9]-p[7:9])
    np.testing.assert_allclose(actual, original*np.exp(-2j*np.pi*(m*shift[0]+n*shift[1])), rtol=0, atol=2e-11)
    scaled = p.copy()
    scaled[:6] *= 1.73
    scaled[7:9] *= 1.73
    np.testing.assert_allclose(cell.coefficients(**settings(scaled))[0], original, rtol=0, atol=2e-11)
    angle = .37
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = p.copy()
    rotated[:4] = (rotation@p[:4].reshape(2, 2)).ravel()
    rotated[6] += angle
    rotated[7:9] = rotation@p[7:9]
    expected = original.copy()
    for ids in ((0, 1, 2, 3), (4, 5, 8, 6)):
        tensor = original[np.array(ids)].reshape(2, 2, 5, 5)
        tensor = np.einsum('ai,ijnm,bj->abnm', rotation, tensor, rotation)
        expected[np.array(ids)] = tensor.reshape(4, 5, 5)
    np.testing.assert_allclose(cell.coefficients(**settings(rotated))[0], expected, rtol=0, atol=2e-11)


@pytest.mark.parametrize('order', [(0, 0), (0, 1), (2, 0), (1, 2), (2, 1)])
@pytest.mark.parametrize('lossless', [False, True])
def test_exact_unequal_and_zero_order_selection_before_closure(order, lossless):
    cell = CartesianRectangleCell(order, refinement=1)
    parameters = PARAMETERS.copy()
    if lossless:
        parameters[[10, 12]] = 0.
    actual, _ = cell.coefficients(**settings(parameters))
    full, _ = CartesianRectangleCell((2, 2), refinement=1).coefficients(**settings(parameters))
    ox, oy = order
    np.testing.assert_allclose(actual, full[:, 4-2*oy:5+2*oy, 4-2*ox:5+2*ox], rtol=0, atol=2e-12)
    q, g, eps = matrix_blocks(full, order, (4, 4))
    inverse = np.linalg.inv(q)
    expected = inverse.conj().T@g@inverse
    count = eps.shape[0]
    slots = cell.conv_matrices(**settings(parameters))
    expected_slots = (expected[:count, :count], expected[count:, count:], np.linalg.inv(eps),
                      -expected[:count, count:], -expected[count:, :count])
    for value, reference in zip(slots, expected_slots):
        assert value.shape == (1, count, count)
        np.testing.assert_allclose(value[0], reference, rtol=0, atol=2e-10)
    storage = (expected+expected.conj().T)/2
    loss = (expected-expected.conj().T)/(2j)
    assert np.linalg.eigvalsh(storage)[0] > 0
    assert np.linalg.eigvalsh(loss)[0] > -1e-9
    if lossless:
        np.testing.assert_allclose(expected, expected.conj().T, rtol=0, atol=1e-9)


@pytest.mark.parametrize('delta_eps,both', [(1., False), (1j, False), (.3+.7j, True)])
def test_homogeneous_constitutive_material_derivative(delta_eps, both):
    cell = CartesianRectangleCell(refinement=1)
    p = PARAMETERS.copy()
    p[9:13] = [2., .2, 2., .2]
    direction = np.zeros(13)
    direction[9:11] = [np.real(delta_eps), np.imag(delta_eps)]
    if both:
        direction[11:13] = direction[9:11]
    value, derivative, _ = oracle(p, direction, cell, 24)
    q, g, _ = matrix_blocks(value, (1, 1), (2, 2))
    dq, dg, de = matrix_blocks(derivative, (1, 1), (2, 2))
    np.testing.assert_allclose(q, np.eye(18), rtol=0, atol=2e-12)
    np.testing.assert_allclose(g, (2.+.2j)*np.eye(18), rtol=0, atol=2e-12)
    indicator, delta = rectangle_coefficients(p, (1, 1))
    expected_coefficient = delta_eps*(delta if both else indicator)
    n, m = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2), indexing='ij')
    expected = expected_coefficient[n.ravel()[:, None]-n.ravel()+2, m.ravel()[:, None]-m.ravel()+2]
    zero = np.zeros_like(expected)
    np.testing.assert_allclose(dg-(2.+.2j)*(dq.conj().T+dq), np.block([[expected, zero], [zero, expected]]),
                               rtol=0, atol=3e-9)
    np.testing.assert_allclose(-de/(2.+.2j)**2, -expected/(2.+.2j)**2, rtol=0, atol=3e-9)
    slots = cell.conv_matrices(**settings(p))
    for actual, exact in zip(slots, ((2.+.2j)*np.eye(9), (2.+.2j)*np.eye(9),
                                     np.eye(9)/(2.+.2j), np.zeros((9, 9)), np.zeros((9, 9)))):
        np.testing.assert_allclose(actual[0], exact, rtol=0, atol=2e-12)


def test_invalid_options_and_physical_geometry():
    for order in (True, np.bool_(True), -1, 1.5, (1, -1), (1, True), (1,), '1'):
        with pytest.raises(ValueError, match='truncation_order'):
            CartesianRectangleCell(order)
    for refinement in (True, np.bool_(False), -1, 3, .5, '1'):
        with pytest.raises(ValueError, match='refinement'):
            CartesianRectangleCell(refinement=refinement)
    cell = CartesianRectangleCell()
    valid = settings(PARAMETERS)
    for update in (dict(width=0.), dict(height=-1.), dict(width=True), dict(angle=1j),
                   dict(width=[350.]), dict(offset=[np.nan, 0.]), dict(offset=[0.]),
                   dict(basis=[[700., 315.], [0., -490.]]), dict(basis=[[1., 1.], [1., 1.]]),
                   dict(basis=[[700., 315.], [1j, 490.]]), dict(width=1400.), dict(angle=np.pi),
                   dict(epsilon=0.), dict(epsilon=1-.1j), dict(background=1-.1j),
                   dict(background=np.inf), dict(epsilon=[4.]), dict(background=True)):
        with pytest.raises(ValueError):
            cell.coefficients(**dict(valid, **update))


@pytest.mark.parametrize('refinement', [0, 1, 2])
@pytest.mark.parametrize('planned', [False, True])
def test_jax_all_physical_directions_against_independent_oracle(refinement, planned):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    parameters = PARAMETERS.copy()
    if planned:
        parameters[6] = np.pi+.31
    cell = CartesianRectangleCell(refinement=refinement, plan=geometry_plan(parameters) if planned else None)
    evaluate = jax.jit(lambda p, d: jax.jvp(lambda p: cell.coefficients(**settings(p, jnp))[0], (p,), (d,)))
    generator = np.array([[0., -1.], [1., 0.]])
    rotation = np.zeros(13)
    rotation[:4], rotation[6] = (generator@parameters[:4].reshape(2, 2)).ravel(), 1.
    rotation[7:9] = generator@parameters[7:9]
    scale = np.zeros(13)
    scale[:6], scale[7:9] = parameters[:6], parameters[7:9]
    shift = np.zeros(13)
    shift[7:9] = [.2, -.1]
    directions = [np.array([.05, -.09, .04, .03, .06, -.05, .11, .07, -.08, .2, -.1, .09, .03]),
                  rotation, scale, shift]
    if refinement == 1:
        directions.extend(np.eye(13))
    for direction in directions:
        low, dlow, _ = oracle(parameters, direction, cell, 16)
        expected, derivative, _ = oracle(parameters, direction, cell, 24)
        np.testing.assert_allclose(expected, low, rtol=0, atol=3e-10)
        np.testing.assert_allclose(derivative, dlow, rtol=0, atol=3e-10)
        actual, tangent = evaluate(jnp.asarray(parameters), jnp.asarray(direction))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=3e-10)
        np.testing.assert_allclose(tangent, derivative, rtol=0, atol=3e-9)


@pytest.mark.parametrize('refinement', [0, 1, 2])
def test_planned_geometry_equivalent_charts_and_independent_variation(refinement):
    p = PARAMETERS.copy()
    p[6] = np.pi+.31
    cell = CartesianRectangleCell(refinement=refinement, plan=geometry_plan(p))
    direction = np.array([.05, -.09, .04, .03, .06, -.05, .11, .07, -.08, .2, -.1, .09, .03])
    actual, metadata = cell.coefficients(**settings(p))
    low, dlow, _ = oracle(p, direction, cell, 16)
    high, dhigh, _ = oracle(p, direction, cell, 24)
    np.testing.assert_allclose(high, low, rtol=0, atol=3e-10)
    np.testing.assert_allclose(dhigh, dlow, rtol=0, atol=3e-10)
    np.testing.assert_allclose(actual, high, rtol=0, atol=3e-10)
    assert metadata['inner_permutation'] == cell.plan.permutation
    indicator, delta = rectangle_coefficients(p, (1, 1))
    np.testing.assert_allclose(actual[7], (p[11]+1j*p[12])*delta
                               +(p[9]+1j*p[10]-p[11]-1j*p[12])*indicator, rtol=0, atol=2e-12)
    original_points = geometry_variation(p, np.zeros(13), cell)[2]
    for swap in (False, True):
        equivalent = p.copy()
        equivalent[6] += np.pi/2 if swap else np.pi
        if swap:
            equivalent[4:6] = equivalent[4:6][::-1]
        other = CartesianRectangleCell(refinement=refinement, plan=geometry_plan(equivalent))
        np.testing.assert_allclose(geometry_variation(equivalent, np.zeros(13), other)[2],
                                   original_points, rtol=0, atol=2e-12*LENGTH)
        np.testing.assert_allclose(other.coefficients(**settings(equivalent))[0], actual, rtol=0, atol=2e-11)
        for value, reference in zip(other.conv_matrices(**settings(equivalent)), cell.conv_matrices(**settings(p))):
            np.testing.assert_allclose(value, reference, rtol=0, atol=2e-10)


def test_concrete_plan_covariance_immutability_and_invalid_replay():
    p = PARAMETERS.copy()
    p[6] = np.pi+.31
    plan = geometry_plan(p)
    with pytest.raises(FrozenInstanceError):
        plan.permutation = (0, 1, 2, 3)
    assert isinstance(plan.permutation, tuple)
    assert hash(plan) == hash(CartesianRectanglePlan(plan.permutation))
    rotation_angle = .37
    rotation = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                         [np.sin(rotation_angle), np.cos(rotation_angle)]])
    rotated = p.copy()
    rotated[:4], rotated[6] = (rotation@p[:4].reshape(2, 2)).ravel(), p[6]+rotation_angle
    rotated[7:9] = rotation@p[7:9]
    scaled = p.copy()
    scaled[:6], scaled[7:9] = p[:6]*1.73, p[7:9]*1.73
    shifted = p.copy()
    shifted[7:9] += [.13, -.07]
    assert geometry_plan(rotated) == geometry_plan(scaled) == geometry_plan(shifted) == plan
    cell = CartesianRectangleCell(plan=plan)
    stale = p.copy()
    stale[6] += np.pi
    with pytest.raises(ValueError, match='nonpositive triangles'):
        cell.coefficients(**settings(stale))
    # The new physical chart is valid only after explicit concrete replanning.
    CartesianRectangleCell(plan=geometry_plan(stale)).coefficients(**settings(stale))
    for bad in (True, (0, 1, 2, 3), 'auto'):
        with pytest.raises(ValueError, match='plan'):
            CartesianRectangleCell(plan=bad)
    for bad in ([0, 1, 2, 3], (0, 2, 1, 3), (True, 1, 2, 3), (0, 1, 2), (0., 1., 2., 3.)):
        with pytest.raises(ValueError, match='permutation'):
            CartesianRectanglePlan(bad)
    physical = settings(p)
    geometry = {key: physical[key] for key in ('basis', 'width', 'height', 'angle', 'offset')}
    for update in (dict(basis=[[1., 1.], [1., 1.]]), dict(basis=[[1., 0.], [0., -1.]]),
                   dict(width=0.), dict(width=1400.), dict(height=np.nan), dict(width=True),
                   dict(angle=1j), dict(offset=[np.nan, 0.]), dict(offset=[0.])):
        with pytest.raises(ValueError):
            CartesianRectanglePlan.from_geometry(**dict(geometry, **update))


@pytest.mark.parametrize('skew,lattice_height,aspect,degrees,fill', [
    (.45, .7, 7/6, 0., .6), (.45, .7, 7/6, 45., .95), (.45, .7, .1, 87., .98),
    (-3., .2, .1, 0., .2), (3., .2, 10., 115., .98), (-1.5, 2., 3., 35., .7),
    (0., 1., 1., 45., .98), (0., 1., 1., 90., .98)])
def test_concrete_plan_contained_orientation_samples(skew, lattice_height, aspect, degrees, fill):
    basis = np.array([[1., skew], [0., lattice_height]])
    angle = np.deg2rad(degrees)
    width, height = np.sqrt(aspect), 1/np.sqrt(aspect)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    signs = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])
    extent = abs((signs*[width, height])@rotation.T@np.linalg.inv(basis).T).max()
    width, height = width*.5*fill/extent, height*.5*fill/extent
    physical = dict(basis=basis, width=width, height=height, angle=angle)
    plan = CartesianRectanglePlan.from_geometry(**physical)
    cell = CartesianRectangleCell(0, plan=plan)
    _, metadata = cell.coefficients(**physical, epsilon=4.+.8j, background=1.+.1j)
    assert metadata['minimum_signed_area'] > 0
    np.testing.assert_allclose(metadata['area'], lattice_height, rtol=2e-12, atol=0)
    np.testing.assert_allclose(metadata['inclusion_area'], width*height, rtol=2e-12, atol=0)
    for swap in (False, True):
        equivalent = dict(physical, width=height if swap else width, height=width if swap else height,
                          angle=angle+(np.pi/2 if swap else np.pi))
        other = CartesianRectanglePlan.from_geometry(**equivalent)
        # Equivalent raw vertex arrays are cyclically shifted by one or two.
        shift = 1 if swap else 2
        assert tuple((i+shift) % 4 for i in other.permutation) == plan.permutation


def test_jax_frozen_plan_stale_chart_and_planning_inside_trace_are_refused():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    p = PARAMETERS.copy()
    p[6] = np.pi+.31
    cell = CartesianRectangleCell(plan=geometry_plan(p))
    evaluate = jax.jit(lambda p: cell.coefficients(**settings(p, jnp))[0])
    assert np.isfinite(np.asarray(evaluate(jnp.asarray(p)))).all()
    for index, invalid in ((6, p[6]+np.pi), (4, 2.)):
        invalid_p = p.copy()
        invalid_p[index] = invalid
        assert np.isnan(np.asarray(evaluate(jnp.asarray(invalid_p)))).all()
    with pytest.raises(ValueError):
        jax.jit(lambda angle: CartesianRectanglePlan.from_geometry(
            basis=LENGTH*p[:4].reshape(2, 2), width=LENGTH*p[4], height=LENGTH*p[5],
            angle=angle).permutation[0])(p[6])


def test_jax_homogeneous_closure_derivative_and_invalid_live_updates():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    cell = CartesianRectangleCell(refinement=1)
    evaluate = jax.jit(lambda p, d: jax.jvp(lambda p: cell.conv_matrices(**settings(p, jnp)), (p,), (d,)))
    p = PARAMETERS.copy()
    p[9:13] = [2., .2, 2., .2]
    indicator, delta = rectangle_coefficients(p, (1, 1))
    n, m = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2), indexing='ij')
    for delta_eps, both in ((1., False), (1j, False), (.3+.7j, True)):
        direction = np.zeros(13)
        direction[9:11] = [np.real(delta_eps), np.imag(delta_eps)]
        if both:
            direction[11:13] = direction[9:11]
        _, derivative = evaluate(jnp.asarray(p), jnp.asarray(direction))
        reference = delta_eps*(delta if both else indicator)
        exact = reference[n.ravel()[:, None]-n.ravel()+2, m.ravel()[:, None]-m.ravel()+2]
        for actual, target in zip(derivative, (exact, exact, -exact/(2.+.2j)**2,
                                               np.zeros((9, 9)), np.zeros((9, 9)))):
            np.testing.assert_allclose(actual[0], target, rtol=0, atol=3e-9)
    values = jax.jit(lambda p: cell.coefficients(**settings(p, jnp))[0])
    for index, invalid in ((3, -.7), (4, 2.), (6, np.pi), (9, 0.), (12, -.1), (7, np.nan)):
        p = PARAMETERS.copy()
        p[index] = invalid
        assert np.isnan(np.asarray(values(jnp.asarray(p)))).all()
