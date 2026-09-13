"""Independent original-nodal derivatives and triangle quadrature for the pair.

Reference integration shares the declared frozen geometry only. The optional
shift1/contact/r0/(2,1) case uses q24/q32 because its original q16/q24 derivative
comparison failed the same3e-10 guard; its finite space is unchanged.
"""
from dataclasses import FrozenInstanceError
from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss
import pytest

from studies.nvf_cartesian_pair import CartesianPairCell, CartesianPairPlan

LENGTH = 700.
PARAMETERS = np.array([1., .45, 0., .7, .2, 193/700, .3, 17/700, .07, -.03,
                       4., .8, 9., .3, 1., .1])
CASES = {
    'gap-r0': (False, 0, .45, (1, 1), (16, 24), 0),
    'gap-r1': (False, 1, .45, (1, 1), (16, 24), 0),
    'contact-r0': (True, 0, .45, (1, 1), (16, 24), 0),
    'contact-r1': (True, 1, .45, (1, 1), (16, 24), 0),
    'shift1-gap-r1-xy12': (False, 1, -1.1, (1, 2), (16, 24), 1),
    'shift1-contact-r0-xy21': (True, 0, -1.1, (2, 1), (24, 32), 1),
}



def settings(p, xp=np):
    p = xp.asarray(p)
    return dict(basis=700*p[:4].reshape(2, 2), widths=700*p[4:6], height=700*p[6],
                gap=700*p[7], offset=700*p[8:10], epsilon=xp.stack([p[10]+1j*p[11], p[12]+1j*p[13]]),
                background=p[14]+1j*p[15])


def geometry_settings(p):
    return {key: value for key, value in settings(p).items() if key not in ('epsilon', 'background')}


def moving_vertices(p, dp, cell):
    """Explicit Cartesian vertex tangents, independent of production geometry."""
    a, da = LENGTH*p[:4].reshape(2, 2), LENGTH*dp[:4].reshape(2, 2)
    offset, doffset = LENGTH*p[8:10], LENGTH*dp[8:10]
    w1, w2, height, gap = LENGTH*p[4:8]
    dw1, dw2, dheight, dgap = LENGTH*dp[4:8]
    if cell.plan.contact and (gap != 0 or dgap != 0):
        raise ValueError('the contact oracle requires gap=dgap=0')
    x = [-(w1+w2+gap)/2, (w1-w2-gap)/2]
    dx = [-(dw1+dw2+dgap)/2, (dw1-dw2-dgap)/2]
    if not cell.plan.contact:
        x.append((w1-w2+gap)/2)
        dx.append((dw1-dw2+dgap)/2)
    x.append((w1+w2+gap)/2)
    dx.append((dw1+dw2+dgap)/2)
    center, dcenter = a.sum(axis=1)/2+offset, da.sum(axis=1)/2+doffset
    inner = np.array([[v, -height/2] for v in x]+[[v, height/2] for v in x[::-1]])+center
    dinner = np.array([[v, -dheight/2] for v in dx]+[[v, dheight/2] for v in dx[::-1]])+dcenter
    unit = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    control = np.vstack([unit@a.T+offset, inner])
    dcontrol = np.vstack([unit@da.T+doffset, dinner])
    return a, da, cell._weights@control, cell._weights@dcontrol


def oracle(p, dp, cell, quadrature):
    """Differentiate the original weak equations and independently integrate FT.

    Only frozen geometry/topology is shared. No production geometry, assembly,
    balancing, polygon FT, closure or finite difference occurs in this oracle.
    """
    a, da, vertices, dvertices = moving_vertices(p, dp, cell)
    cell_area = np.linalg.det(a)
    dlog_cell = np.trace(np.linalg.solve(a, da))
    material = np.array([p[14]+1j*p[15], p[10]+1j*p[11], p[12]+1j*p[13]])[cell._tags]
    dmaterial = np.array([dp[14]+1j*dp[15], dp[10]+1j*dp[11], dp[12]+1j*dp[13]])[cell._tags]
    count = cell.degrees_of_freedom+1
    matrix, dmatrix = np.zeros((count, count), complex), np.zeros((count, count), complex)
    right, dright = np.zeros((count, 2), complex), np.zeros((count, 2), complex)
    areas, dareas, gradients, dgradients = [], [], [], []
    for triangle, epsilon, depsilon in zip(cell._cells, material, dmaterial):
        r, dr = vertices[triangle], dvertices[triangle]
        jac = np.column_stack([r[1]-r[0], r[2]-r[0]])
        djac = np.column_stack([dr[1]-dr[0], dr[2]-dr[0]])
        inverse = np.linalg.inv(jac)
        area = np.linalg.det(jac)/2
        if not np.isfinite(area) or area <= 0:
            raise ValueError('oracle requires positive finite triangles')
        darea = area*np.trace(inverse@djac)
        gradient = np.array([[-1., -1.], [1., 0.], [0., 1.]])@inverse
        dgradient = -gradient@djac@inverse
        weight, dweight = epsilon*area, depsilon*area+epsilon*darea
        local = weight*(gradient@gradient.T)
        dlocal = dweight*(gradient@gradient.T)+weight*(dgradient@gradient.T+gradient@dgradient.T)
        load, dload = -weight*gradient, -dweight*gradient-weight*dgradient
        ids = cell._periodic[triangle]
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
    solution, dsolution = np.zeros((count, 2), complex), np.zeros((count, 2), complex)
    solution[1:] = np.linalg.solve(matrix[1:, 1:], right[1:])
    derivative_right = dright-dmatrix@solution
    dsolution[1:] = np.linalg.solve(matrix[1:, 1:], derivative_right[1:])
    residual = np.linalg.norm(matrix@solution-right)/max(np.linalg.norm(right), 1.)
    derivative_residual = np.linalg.norm(matrix@dsolution-derivative_right)/max(np.linalg.norm(derivative_right), 1.)
    gradient, dgradient = np.asarray(gradients), np.asarray(dgradients)
    local, dlocal = solution[cell._periodic[cell._cells]], dsolution[cell._periodic[cell._cells]]
    field = np.eye(2)+np.einsum('eia,eic->eac', gradient, local)
    dfield = np.einsum('eia,eic->eac', dgradient, local)+np.einsum('eia,eic->eac', gradient, dlocal)
    adjoint, dadjoint = field.conj().transpose(0, 2, 1), dfield.conj().transpose(0, 2, 1)
    gram = material[:, None, None]*(adjoint@field)
    dgram = dmaterial[:, None, None]*(adjoint@field)+material[:, None, None]*(dadjoint@field+adjoint@dfield)
    def components(b, g, epsilon):
        return np.stack([b[:, 0, 0], b[:, 0, 1], b[:, 1, 0], b[:, 1, 1],
                         g[:, 0, 0], g[:, 0, 1], g[:, 1, 1], epsilon, g[:, 1, 0]])
    values, dvalues = components(field, gram, material), components(dfield, dgram, dmaterial)
    q, w = leggauss(quadrature)
    q, w = (q+1)/2, w/2
    r, s = np.meshgrid(q, q, indexing='ij')
    weights = (w[:, None]*w[None, :]*r).ravel()
    barycentric = np.stack([1-r, r*(1-s), r*s], axis=-1).reshape(-1, 3)
    ox, oy = cell.order
    n, m = np.meshgrid(np.arange(-2*oy, 2*oy+1), np.arange(-2*ox, 2*ox+1), indexing='ij')
    wave = 2*np.pi*np.column_stack([m.ravel(), n.ravel()])@np.linalg.inv(a)
    dwave = -wave@da@np.linalg.inv(a)
    transforms, dtransforms = [], []
    for first in range(0, len(cell._cells), 8):
        selection = slice(first, first+8)
        triangles = cell._cells[selection]
        points = np.einsum('qi,eia->eqa', barycentric, vertices[triangles])
        dpoints = np.einsum('qi,eia->eqa', barycentric, dvertices[triangles])
        phase = np.exp(-1j*np.einsum('eqa,ka->eqk', points, wave))
        dphase = -1j*phase*(np.einsum('eqa,ka->eqk', dpoints, wave)+np.einsum('eqa,ka->eqk', points, dwave))
        area = np.array(areas)[selection]
        value = 2*area[:, None]/cell_area*np.einsum('q,eqk->ek', weights, phase)
        derivative = value*(np.array(dareas)[selection]/area-dlog_cell)[:, None]
        derivative += 2*area[:, None]/cell_area*np.einsum('q,eqk->ek', weights, dphase)
        transforms.append(value)
        dtransforms.append(derivative)
    transforms, dtransforms = np.concatenate(transforms), np.concatenate(dtransforms)
    shape = (9, 4*oy+1, 4*ox+1)
    return (values@transforms).reshape(shape), (dvalues@transforms+values@dtransforms).reshape(shape), dict(
        original_residual=float(residual), derivative_residual=float(derivative_residual))


def directions(p, contact):
    result = [(f'coordinate_{i}', np.eye(16)[i]) for i in range(16) if not (contact and i == 7)]
    mixed = np.array([.1, -.07, .02, .04, -.03, .06, .02, .01, .03, -.02, .13, .02, -.09, .03, .05, -.01])
    if contact:
        mixed[7] = 0
    translation = np.zeros(16)
    translation[8:10] = [.17, -.11]
    scale = np.r_[p[:10], np.zeros(6)]
    return result+[('mixed', mixed), ('whole_cell_translation', translation), ('common_scale', scale)]


def finite_difference(p, direction, cell, step):
    values = {i: cell.coefficients(**settings(p+i*step*direction))[0] for i in (-2, -1, 1, 2)}
    return (-values[2]+8*values[1]-8*values[-1]+values[-2])/(12*step)


def analytic_indicators(geometry, order=(3, 2)):
    basis = np.asarray(geometry['basis'])
    area = np.linalg.det(basis)
    width1, width2 = geometry['widths']
    height, gap = geometry['height'], geometry['gap']
    center = basis.sum(axis=1)/2+geometry.get('offset', (0., 0.))
    n, m = np.meshgrid(np.arange(-order[1], order[1]+1), np.arange(-order[0], order[0]+1), indexing='ij')
    wave = 2*np.pi*np.stack([m, n], axis=-1)@np.linalg.inv(basis)
    result = []
    for width, dx in ((width1, -(width2+gap)/2), (width2, (width1+gap)/2)):
        local_center = center+np.array([dx, 0.])
        result.append(width*height/area*np.exp(-1j*(wave@local_center))
                      *np.sinc(wave[..., 0]*width/(2*np.pi))*np.sinc(wave[..., 1]*height/(2*np.pi)))
    delta = (m == 0) & (n == 0)
    return np.stack([delta-result[0]-result[1], *result]), n, m


def overlap_area(triangle, clip):
    """Independent convex clipping for nonoverlap checks, no mesh construction."""
    polygon = list(triangle)
    cross = lambda a, b: a[0]*b[1]-a[1]*b[0]
    for a, b in zip(clip, np.roll(clip, -1, axis=0)):
        old, polygon = polygon, []
        if not old:
            return 0.
        for p, q in zip(old, old[1:]+old[:1]):
            dp, dq = cross(b-a, p-a), cross(b-a, q-a)
            if dp >= 0:
                polygon.append(p)
            if (dp >= 0) != (dq >= 0):
                polygon.append(p+dp/(dp-dq)*(q-p))
    if len(polygon) < 3:
        return 0.
    polygon = np.asarray(polygon)
    return abs(np.sum(polygon[:, 0]*np.roll(polygon[:, 1], -1)-polygon[:, 1]*np.roll(polygon[:, 0], -1)))/2


def closure_oracle(c, dc, order):
    """Explicit inverse derivative independent of production closure solves."""
    ox, oy = order
    n, m = np.meshgrid(np.arange(-oy, oy+1), np.arange(-ox, ox+1), indexing='ij')
    rows = n.ravel()[:, None]-n.ravel()+2*oy
    columns = m.ravel()[:, None]-m.ravel()+2*ox
    selected, derivative = c[:, rows, columns], dc[:, rows, columns]
    def blocks(values):
        return (np.block([[values[0], values[1]], [values[2], values[3]]]),
                np.block([[values[4], values[5]], [values[8], values[6]]]), values[7])
    q, g, e = blocks(selected)
    dq, dg, de = blocks(derivative)
    qi, z = np.linalg.inv(q), np.linalg.inv(e)
    tensor = qi.conj().T@g@qi
    dtensor = qi.conj().T@dg@qi-qi.conj().T@dq.conj().T@tensor-tensor@dq@qi
    dz = -z@de@z
    count = len(e)
    def slots(h, longitudinal):
        return np.stack([h[:count, :count], h[count:, count:], longitudinal,
                         -h[:count, count:], -h[count:, :count]])[:, None]
    return slots(tensor, z), slots(dtensor, dz)


def case_cell(case, order=None):
    contact, refinement, skew, declared_order, _, shift = CASES[case]
    p = PARAMETERS.copy()
    p[1] = skew
    if contact:
        p[7] = 0
    plan = CartesianPairPlan.from_geometry(refinement=refinement, **geometry_settings(p))
    assert plan.shift == shift
    return CartesianPairCell(plan, declared_order if order is None else order), p


@lru_cache(maxsize=None)
def independent_case(case):
    cell, p = case_cell(case)
    low_order, high_order = CASES[case][4]
    named = directions(p, cell.plan.contact)
    expected, derivative = [], []
    for _, tangent in named:
        low, dlow, low_meta = oracle(p, tangent, cell, low_order)
        high, dhigh, high_meta = oracle(p, tangent, cell, high_order)
        np.testing.assert_allclose(low, high, rtol=0, atol=3e-10)
        np.testing.assert_allclose(dlow, dhigh, rtol=0, atol=3e-10)
        assert max(*low_meta.values(), *high_meta.values()) <= 1e-10
        expected.append(high)
        derivative.append(dhigh)
    return np.array(expected), np.array(derivative), np.array([tangent for _, tangent in named])


@pytest.mark.parametrize('case', tuple(CASES))
def test_geometry_exact_partition_material_and_periodicity(case):
    cell, p = case_cell(case)
    physical = geometry_settings(p)
    vertices, triangles, tags, periodic, weights = cell.plan.replay(**physical)
    points = vertices[triangles]
    areas = np.linalg.det(np.stack([points[:, 1]-points[:, 0], points[:, 2]-points[:, 0]], axis=-1))/2
    assert np.all(np.isfinite(areas) & (areas > 0))
    refinement = cell.plan.refinement
    assert len(triangles) == (14 if cell.plan.contact else 18)*4**refinement
    assert cell.degrees_of_freedom == ((6, 27) if cell.plan.contact else (8, 35))[refinement]
    edges = {}
    for number, triangle in enumerate(triangles):
        for a, b in zip(triangle, np.roll(triangle, -1)):
            edges.setdefault(tuple(sorted((int(a), int(b)))), []).append((number, int(a), int(b)))
    assert len(vertices)-len(edges)+len(triangles) == 1
    assert all(len(adjacent) in (1, 2) for adjacent in edges.values())
    assert all(adjacent[0][1:] == adjacent[1][1:][::-1] for adjacent in edges.values() if len(adjacent) == 2)
    boundary = [edge for edge, adjacent in edges.items() if len(adjacent) == 1]
    assert len(boundary) == 4*2**refinement
    assert all(not np.any(weights[list(edge), 4:]) for edge in boundary)
    unit = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    coordinates = weights[:, :4]@unit
    boundary_nodes = ~np.any(weights[:, 4:], axis=1)
    for axis in range(2):
        low = np.where(boundary_nodes & (coordinates[:, axis] == 0))[0]
        high = np.where(boundary_nodes & (coordinates[:, axis] == 1))[0]
        low = low[np.argsort(coordinates[low, 1-axis])]
        high = high[np.argsort(coordinates[high, 1-axis])]
        assert len(low) == 2**refinement+1
        np.testing.assert_array_equal(periodic[low], periodic[high])
        np.testing.assert_allclose(vertices[high]-vertices[low],
                                   np.broadcast_to(physical['basis'][:, axis], (len(low), 2)), rtol=0, atol=3e-12)
    shared = [edge for edge, adjacent in edges.items() if len(adjacent) == 2
              and {tags[adjacent[0][0]], tags[adjacent[1][0]]} == {1, 2}]
    assert len(shared) == (2**refinement if cell.plan.contact else 0)
    shared_length = sum(np.linalg.norm(vertices[b]-vertices[a]) for a, b in shared)
    np.testing.assert_allclose(shared_length, physical['height'] if cell.plan.contact else 0., rtol=0, atol=3e-12)
    fractional = (points-physical['offset'])@np.linalg.inv(physical['basis']).T
    assert max(overlap_area(fractional[i], fractional[j]) for i in range(len(points)) for j in range(i)) <= 3e-12
    expected_areas = np.r_[np.linalg.det(physical['basis'])-sum(physical['widths'])*physical['height'],
                           physical['widths']*physical['height']]
    np.testing.assert_allclose(np.array([areas[tags == tag].sum() for tag in range(3)]),
                               expected_areas, rtol=3e-12, atol=3e-12)
    coefficients, _ = cell.coefficients(**settings(p))
    indicators, _, _ = analytic_indicators(physical, order=tuple(2*n for n in cell.order))
    material = np.r_[settings(p)['background'], settings(p)['epsilon']]
    np.testing.assert_allclose(coefficients[7], np.einsum('e,enm->nm', material, indicators), rtol=0, atol=3e-12)
    shifted = dict(settings(p), offset=physical['offset']+np.array([37., -23.]))
    moved, _ = cell.coefficients(**shifted)
    wave = 2*np.pi*np.stack([cell._m, cell._n], axis=-1)@np.linalg.inv(physical['basis'])
    np.testing.assert_allclose(moved, coefficients*np.exp(-1j*(wave@np.array([37., -23.]))), rtol=0, atol=3e-12)


@pytest.mark.parametrize('case', tuple(CASES))
def test_numpy_original_weak_coefficients_and_derivatives(case):
    cell, p = case_cell(case)
    expected, derivative, tangents = independent_case(case)
    actual, metadata = cell.coefficients(**settings(p))
    assert metadata['cell_residual'] <= 1e-10
    for reference in expected:
        np.testing.assert_allclose(actual, reference, rtol=0, atol=3e-10)
    # Preserve the original 74 finite-difference controls. Both additional
    # skewed cases use the independent analytical oracle in the JAX controls.
    if CASES[case][-1] == 0:
        for tangent, dvalue in zip(tangents, derivative):
            for step in (2e-4, 1e-4):
                np.testing.assert_allclose(finite_difference(p, tangent, cell, step), dvalue, rtol=0, atol=3e-9)
    wave = 2*np.pi*np.stack([cell._m, cell._n], axis=-1)@np.linalg.inv(settings(p)['basis'])
    np.testing.assert_allclose(derivative[-2], -1j*(wave@(LENGTH*np.array([.17, -.11])))*actual,
                               rtol=0, atol=3e-9)
    np.testing.assert_allclose(derivative[-1], 0., rtol=0, atol=3e-9)


@pytest.mark.parametrize('case', ('gap-r0', 'gap-r1', 'contact-r0', 'contact-r1'))
def test_homogeneous_and_selected_closure_passivity(case):
    for order in ((0, 0), (1, 0), (0, 1), (1, 1)):
        cell, p = case_cell(case, order)
        for state in ('lossy', 'lossless', 'homogeneous'):
            physical = settings(p)
            if state == 'lossless':
                physical.update(epsilon=np.real(physical['epsilon']), background=np.real(physical['background']))
            elif state == 'homogeneous':
                physical['epsilon'] = np.repeat(physical['background'], 2)
            slots, metadata = cell.conv_matrices(**physical)
            count = (2*order[0]+1)*(2*order[1]+1)
            assert slots.epx.shape == (1, count, count)
            assert metadata['minimum_storage_eigenvalue'] > 0
            assert metadata['minimum_dissipation_eigenvalue'] > -1e-9
            if state == 'lossless':
                assert metadata['hermitian_error'] < 1e-9
            if state == 'homogeneous':
                values, _ = cell.coefficients(**physical)
                expected = np.zeros_like(values)
                expected[[0, 3], 2*order[1], 2*order[0]] = 1.
                expected[[4, 6, 7], 2*order[1], 2*order[0]] = physical['background']
                np.testing.assert_allclose(values, expected, rtol=0, atol=3e-10)
                for slot in (slots.epx, slots.epy):
                    np.testing.assert_allclose(slot, physical['background']*np.eye(count)[None], rtol=0, atol=3e-10)


def test_frozen_plan_and_invalid_concrete_admission():
    cell, p = case_cell('gap-r0')
    physical = settings(p)
    geometry = geometry_settings(p)
    with pytest.raises(FrozenInstanceError):
        cell.plan.shift = 1
    for key in ('basis', 'widths', 'height', 'gap', 'offset'):
        with pytest.raises(ValueError):
            CartesianPairPlan.from_geometry(**dict(geometry, **{key: np.asarray(geometry[key], complex)}))
        with pytest.raises(ValueError):
            cell.coefficients(**dict(physical, **{key: np.asarray(physical[key], complex)}))
    for bad in (dict(physical, epsilon=(4-.1j, 9.)), dict(physical, background=0.),
                dict(physical, background=np.nan), dict(physical, basis=np.diag([1e200, 1e200])),
                dict(physical, height=1000.), dict(physical, gap=0.), dict(physical, offset=(np.nan, 0.))):
        with np.errstate(over='ignore'), pytest.raises(ValueError):
            cell.coefficients(**bad)
    for order in (-1, True, (1, -1), (1, 2, 3), 1.5):
        with pytest.raises(ValueError):
            CartesianPairCell(cell.plan, order)
    for args in ((False, 4, 0), (False, 0, 4), (False, True, 0), (0, 0, 0)):
        with pytest.raises(ValueError):
            CartesianPairPlan(*args)
    with pytest.raises(ValueError):
        cell.coefficients(**physical, xp=object())
    contact, _, = case_cell('contact-r0')
    with pytest.raises(ValueError):
        contact.coefficients(**physical)
    refused = 0
    for shift in range(4):
        try:
            CartesianPairPlan(False, shift).replay(**geometry)
        except ValueError:
            refused += 1
    assert refused > 0


@pytest.mark.parametrize('case', tuple(CASES))
def test_jax_live_coefficient_and_independent_closure_derivatives(case):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    cell, p = case_cell(case)
    expected, derivative, tangents = independent_case(case)
    def coefficient_function(parameters):
        return cell.coefficients(**settings(parameters, jnp), xp=jnp)[0]
    def closure_function(parameters):
        return jnp.stack(cell.conv_matrices(**settings(parameters, jnp), xp=jnp)[0])
    coefficient_jvp = jax.jit(lambda parameters, tangent: jax.jvp(coefficient_function, (parameters,), (tangent,)))
    closure_jvp = jax.jit(lambda parameters, tangent: jax.jvp(closure_function, (parameters,), (tangent,)))
    for reference, dvalue, tangent in zip(expected, derivative, tangents):
        value, actual_derivative = coefficient_jvp(jnp.asarray(p), jnp.asarray(tangent))
        np.testing.assert_allclose(value, reference, rtol=0, atol=3e-10)
        np.testing.assert_allclose(actual_derivative, dvalue, rtol=0, atol=3e-9)
        expected_slots, dslots = closure_oracle(reference, dvalue, cell.order)
        slots, actual_dslots = closure_jvp(jnp.asarray(p), jnp.asarray(tangent))
        np.testing.assert_allclose(slots, expected_slots, rtol=0, atol=3e-10)
        np.testing.assert_allclose(actual_dslots, dslots, rtol=0, atol=3e-9)


@pytest.mark.parametrize('case', ('gap-r0', 'contact-r0'))
def test_jax_invalid_replay_and_concrete_planning(case):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    cell, p = case_cell(case)
    coefficients = jax.jit(lambda parameters: cell.coefficients(**settings(parameters, jnp), xp=jnp)[0])
    closure = jax.jit(lambda parameters: jnp.stack(cell.conv_matrices(**settings(parameters, jnp), xp=jnp)[0]))
    invalids = []
    for index, value in ((4, 0.), (6, 2.), (11, -.1), (14, -1.), (8, np.nan),
                         (7, .02 if cell.plan.contact else 0.)):
        bad = p.copy()
        bad[index] = value
        invalids.append(bad)
    bad = p.copy()
    bad[:4] = [1., 0., 0., 0.]
    invalids.append(bad)
    for bad in invalids:
        assert np.all(np.isnan(coefficients(jnp.asarray(bad))))
        assert np.all(np.isnan(closure(jnp.asarray(bad))))
        with pytest.raises(ValueError):
            cell.coefficients(**settings(jnp.asarray(bad), jnp), xp=jnp)
    geometry = geometry_settings(p)
    planner = jax.jit(lambda widths: CartesianPairPlan.from_geometry(**dict(geometry, widths=widths)))
    with pytest.raises(ValueError, match='concrete'):
        planner(jnp.asarray(geometry['widths']))
    jax.config.update('jax_enable_x64', False)
    try:
        with pytest.raises(ValueError, match='x64'):
            cell.coefficients(**settings(p), xp=jnp)
    finally:
        jax.config.update('jax_enable_x64', True)
