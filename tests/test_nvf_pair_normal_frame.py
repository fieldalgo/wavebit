"""Independent field products, physical limits and AD for the normal-frame cell.

Quadrature consistency checks the declared finite model. Optical accuracy is
measured separately; the analytic normal extension itself is shared with the
existing, independently tested parametric path.
"""
from functools import lru_cache
import numpy as np
import pytest

from wavebit._core import parametric as P
from wavebit._core.conforming.cartesian_pair_normal_frame import CartesianPairNormalFrame
from wavebit._core.conforming import cartesian_pair_normal_frame as normal_frame_module
from wavebit._core.lattice import as_lattice


GEOMETRY = dict(basis=np.diag([700., 560.]), widths=(170., 230.), height=280., offset=(21., -14.))
MATERIALS = {'equal': (4., 4.), 'lossy': (4.+.8j, 9.+.3j), 'metal': (12.25, -10.+1j)}


def make_cell(**kwargs):
    settings = dict(GEOMETRY, order=(1, 2), quadrature=16, coercivity_angle=1.53)
    settings.update(kwargs)
    return CartesianPairNormalFrame(**settings)


def reference(epsilon, gap, geometry=GEOMETRY, order=(1, 2), quadrature=16, background=1., return_fields=False):
    """Direct B and epsilon B^H B at independently constructed physical nodes."""
    px, py = np.diag(geometry['basis'])
    wa, wb = geometry['widths']
    h = geometry['height']
    ox, oy = geometry['offset']
    xedges = [0., (px-wa-wb-gap)/2, (px+wa-wb-gap)/2, (px+wa-wb+gap)/2, (px+wa+wb+gap)/2, px]
    yedges = [0., (py-h)/2, (py+h)/2, py]
    u, weight = np.polynomial.legendre.leggauss(quadrature)
    points, weights, materials = [], [], []
    for j, (bottom, top) in enumerate(zip(yedges[:-1], yedges[1:])):
        for i, (left, right) in enumerate(zip(xedges[:-1], xedges[1:])):
            if right == left or top == bottom:
                continue
            x = left+(right-left)*(u+1)/2+ox
            y = bottom+(top-bottom)*(u+1)/2+oy
            X, Y = np.meshgrid(x, y)
            points.append(np.column_stack([X.ravel(), Y.ravel()]))
            weights.append((np.outer(weight, weight)*(right-left)*(top-bottom)/4).ravel())
            e = epsilon[i//2] if j == 1 and i in (1, 3) else background
            materials.extend([e]*(quadrature**2))
    points, weights, materials = np.vstack(points), np.concatenate(weights), np.asarray(materials)
    shapes = [dict(type='rectangle', lx=wa, ly=h, cx=ox-(wb+gap)/2, cy=oy, material=epsilon[0]),
              dict(type='rectangle', lx=wb, ly=h, cx=ox+(wa+gap)/2, cy=oy, material=epsilon[1])]
    lattice = as_lattice(lattice=geometry['basis'])
    plan = P.plan_gamma(shapes, lattice)
    # No field correction, closure inverse, or separable Fourier product from
    # the candidate is read by this independent pointwise product integration.
    nx, ny = P._piece_field(np, points[:, 0], points[:, 1], shapes, plan, None, lattice, np.float64)
    normal = np.column_stack([nx, ny])
    projector = normal[:, :, None]*normal[:, None, :].conj()
    kappa = sum(epsilon)/2
    field = np.eye(2)+(kappa/materials-1)[:, None, None]*projector
    gram = materials[:, None, None]*(field.conj().swapaxes(-1, -2)@field)
    if return_fields:
        return points, weights/(px*py), materials, field
    n, m = np.meshgrid(np.arange(-2*order[1], 2*order[1]+1),
                       np.arange(-2*order[0], 2*order[0]+1), indexing='ij')
    wave = 2*np.pi*np.column_stack([m.ravel()/px, n.ravel()/py])
    phase = np.exp(-1j*points@wave.T)
    fields = np.array([field[:, 0, 0], field[:, 0, 1], field[:, 1, 0], field[:, 1, 1],
                       gram[:, 0, 0], gram[:, 0, 1], gram[:, 1, 1], materials, gram[:, 1, 0]])
    return ((fields*weights)@phase/(px*py)).reshape(9, 4*order[1]+1, 4*order[0]+1)


@pytest.mark.parametrize('name', MATERIALS)
@pytest.mark.parametrize('gap', [0., 7., 1e-7])
def test_fitted_normal_products_against_direct_field_integrals(name, gap):
    cell = make_cell()
    e = MATERIALS[name]
    value, slots, meta = cell.evaluate(gap=gap, epsilon=e)
    expected = reference(e, gap)
    np.testing.assert_allclose(value, expected, rtol=0, atol=3e-10)
    assert meta['minimum_dissipation'] > -1e-9
    assert meta['inverse_residual'] < 1e-10
    assert meta['frame_minimum_rotated_storage'] > 0
    assert meta['frame_storage_lower_bound'] > 0
    assert meta['frame_minimum_rotated_storage'] >= meta['frame_storage_lower_bound']-1e-12
    assert all(s.shape == (1, 15, 15) for s in slots)
    # This independent inverse closure does not use ConvMatrices.from_energy.
    modes = [(m, n) for n in range(-2, 3) for m in range(-1, 2)]
    coeff = np.array([[expected[:, a[1]-b[1]+4, a[0]-b[0]+2] for b in modes] for a in modes]).transpose(2, 0, 1)
    q = np.block([[coeff[0], coeff[1]], [coeff[2], coeff[3]]])
    g = np.block([[coeff[4], coeff[5]], [coeff[8], coeff[6]]])
    qi = np.linalg.inv(q)
    h = qi.conj().T@g@qi
    np.testing.assert_allclose(np.block([[slots.epx[0], -slots.epxy[0]], [-slots.epyx[0], slots.epy[0]]]), h,
                               rtol=0, atol=3e-10)


@pytest.mark.parametrize('gap', [0., 7., 1e-9])
@pytest.mark.parametrize('tile_rows', [7, 64])
def test_tiled_integrals_retain_material_regions_and_last_partial_tile(gap, tile_rows, monkeypatch):
    epsilon = MATERIALS['metal']
    expected = reference(epsilon, gap, quadrature=24)
    field = P._piece_field
    visited = []
    def observed(xp, x, y, *args, **kwargs):
        visited.append(x.shape)
        return field(xp, x, y, *args, **kwargs)
    monkeypatch.setattr(normal_frame_module, '_QUADRATURE_TILE_ROWS', tile_rows)
    monkeypatch.setattr(P, '_piece_field', observed)
    value, _, meta = make_cell(quadrature=24).evaluate(gap=gap, epsilon=epsilon)
    np.testing.assert_allclose(value, expected, rtol=0, atol=3e-10)
    assert sum(shape[0] for shape in visited) == 72
    assert all(shape[0] <= tile_rows and shape[1] == 120 for shape in visited)
    assert visited[-1][0] == 72 % tile_rows
    assert meta['maximum_field_samples'] == tile_rows*120


@pytest.mark.parametrize('e,angle', [(4., 0.), (4.+.2j, 0.), (-10.+1j, 1.53)])
@pytest.mark.parametrize('gap', [0., 7.])
def test_uniform_material_is_exact_without_a_finite_field_error(e, angle, gap):
    value, slots, meta = make_cell(coercivity_angle=angle).evaluate(gap=gap, epsilon=(e, e), background=e)
    expected = np.zeros_like(value)
    expected[:, 4, 2] = [1, 0, 0, 1, e, 0, e, e, 0]
    np.testing.assert_allclose(value, expected, rtol=0, atol=2e-13)
    identity = np.eye(15)
    np.testing.assert_allclose(slots, np.array([e*identity, e*identity, identity/e,
                                              0*identity, 0*identity])[:, None], rtol=0, atol=2e-13)
    assert meta['reciprocity_error'] < 2e-13


@pytest.mark.parametrize('gap', [0., 7.])
def test_reflection_and_material_listing_covariance(gap):
    cell = make_cell(offset=(0., 0.))
    epsilon = MATERIALS['metal']
    value = cell.coefficients(gap=gap, epsilon=epsilon)[0]
    reflected = make_cell(offset=(0., 0.), widths=GEOMETRY['widths'][::-1])
    result = reflected.coefficients(gap=gap, epsilon=epsilon[::-1])[0]
    signs = np.array([1, -1, -1, 1, 1, -1, 1, 1, -1])[:, None, None]
    np.testing.assert_allclose(result, signs*value[:, :, ::-1], rtol=0, atol=3e-11)


@pytest.mark.parametrize('scale', [1e-4, 1e4])
def test_physical_unit_covariance(scale):
    reference_value = make_cell().coefficients(gap=7., epsilon=MATERIALS['metal'])[0]
    geometry = {k: np.asarray(v)*scale for k, v in GEOMETRY.items()}
    scaled = make_cell(**geometry).coefficients(gap=7.*scale, epsilon=MATERIALS['metal'])[0]
    np.testing.assert_allclose(scaled, reference_value, rtol=0, atol=3e-10)


@pytest.mark.parametrize('gap', [0., 7.])
def test_admission_angle_does_not_rotate_physical_coefficients_or_slots(gap):
    first = make_cell(coercivity_angle=1.50).evaluate(gap=gap, epsilon=MATERIALS['metal'])
    second = make_cell(coercivity_angle=1.54).evaluate(gap=gap, epsilon=MATERIALS['metal'])
    for a, b in zip(first[:2], second[:2]):
        np.testing.assert_array_equal(a, b)


def test_admission_order_is_not_a_quadrature_accuracy_guarantee():
    cell = make_cell(height=560., order=(2, 2), quadrature=8)
    with pytest.raises(ValueError, match='base-integral'):
        cell.evaluate(gap=7., epsilon=(4., 9.))


@pytest.mark.parametrize('epsilon', [(4., 9.), (12.25, -10.+1j)])
@pytest.mark.parametrize('gap', [0., 7.])
def test_laminated_limit_recovers_exact_cartesian_inverse_and_direct_rules(epsilon, gap):
    cell = make_cell(height=560., offset=(0., 0.), order=(2, 1), quadrature=24)
    _, slots, meta = cell.evaluate(gap=gap, epsilon=epsilon)
    shapes = [dict(type='rectangle', lx=170., ly=560., cx=-(230.+gap)/2, material=epsilon[0]),
              dict(type='rectangle', lx=230., ly=560., cx=(170.+gap)/2, material=epsilon[1])]
    e, r = P.assemble_eps_coeffs(np, shapes, 1., 2, 1, as_lattice(lattice=GEOMETRY['basis']), np.complex128)
    e, r = P.conv_matrix_2d(np, e, 2, 1), P.conv_matrix_2d(np, r, 2, 1)
    expected = np.array([np.linalg.inv(r), e, np.linalg.inv(e), np.zeros_like(e), np.zeros_like(e)])[:, None]
    np.testing.assert_allclose(slots, expected, rtol=0, atol=3e-10)
    assert meta['reciprocity_error'] < 3e-10


@pytest.mark.parametrize('name', MATERIALS)
def test_reciprocity_diagnostic_matches_transpose_in_a_real_standing_wave_basis(name):
    _, slots, meta = make_cell().evaluate(gap=0., epsilon=MATERIALS[name])
    tensor = np.block([[slots.epx[0], -slots.epxy[0]], [-slots.epyx[0], slots.epy[0]]])
    modes = [(m, n) for n in range(-2, 3) for m in range(-1, 2)]
    index = {k: i for i, k in enumerate(modes)}
    transform = np.zeros((len(modes), len(modes)), complex)
    transform[index[0, 0], 0] = 1.
    used, column = {(0, 0)}, 1
    for m, n in modes:
        if (m, n) in used:
            continue
        a, b = index[m, n], index[-m, -n]
        transform[[a, b], column] = 1/np.sqrt(2)
        transform[a, column+1], transform[b, column+1] = -1j/np.sqrt(2), 1j/np.sqrt(2)
        used.update([(m, n), (-m, -n)])
        column += 2
    zero = np.zeros_like(transform)
    change = np.block([[transform, zero], [zero, transform]])
    real_basis = change.conj().T@tensor@change
    expected = np.linalg.norm(real_basis-real_basis.T)/max(np.linalg.norm(real_basis), 1.)
    np.testing.assert_allclose(meta['relative_reciprocity_defect'], expected, rtol=0, atol=3e-13)
    if name == 'equal':
        assert expected < 1e-12


@pytest.fixture(scope='module')
def jax_runtime():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    old = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    @lru_cache(None)
    def program(name):
        cell = make_cell(order=(1, 1))
        e = MATERIALS[name]
        def value(gap):
            coeff, slots, _ = cell.evaluate(gap=gap, epsilon=e, xp=jnp)
            return coeff, jnp.stack(slots)
        return cell, value, jax.jit(lambda g: jax.jvp(value, (g,), (jnp.asarray(1.),)))
    yield jax, jnp, program
    jax.config.update('jax_enable_x64', old)


@pytest.mark.parametrize('name', MATERIALS)
@pytest.mark.parametrize('gap', [0., 7.])
def test_gap_jit_derivative_includes_the_opening_strip(name, gap, jax_runtime):
    _, jnp, program = jax_runtime
    cell, _, derivative = program(name)
    actual, tangent = derivative(jnp.asarray(gap))
    def value(g):
        coeff, slots, _ = cell.evaluate(gap=g, epsilon=MATERIALS[name])
        return coeff, np.stack(slots)
    base = value(gap)
    for got, want in zip(actual, base):
        np.testing.assert_allclose(got, want, rtol=0, atol=3e-10)
    references = []
    # A weak material contrast makes the contact curvature large. These two
    # windows resolve the second-order secant before it is compared to AD.
    for step in [2e-5, 1e-5] if gap == 0 else [.002, .001]:
        if gap == 0:
            first, second = value(step), value(2*step)
            slope = tuple((-3*v+4*a-b)/(2*step) for v, a, b in zip(base, first, second))
        else:
            first, second = value(gap-step), value(gap+step)
            slope = tuple((b-a)/(2*step) for a, b in zip(first, second))
        references.append(slope)
    for i in range(2):
        np.testing.assert_allclose(tangent[i], references[1][i], rtol=3e-5, atol=2e-8)
        np.testing.assert_allclose(references[0][i], references[1][i], rtol=3e-5, atol=2e-8)


def rectangle_transforms(gap, order=(1, 1), height=280.):
    """Closed-form area integrals, with no candidate quadrature or field."""
    px, py = np.diag(GEOMETRY['basis'])
    wa, wb = GEOMETRY['widths']
    ox, oy = GEOMETRY['offset']
    n, m = np.meshgrid(np.arange(-2*order[1], 2*order[1]+1),
                       np.arange(-2*order[0], 2*order[0]+1), indexing='ij')
    result = []
    for width, center in [(wa, ox-(wb+gap)/2), (wb, ox+(wa+gap)/2)]:
        phase = np.exp(-2j*np.pi*(m*(.5+center/px)+n*(.5+oy/py)))
        result.append(width*height/(px*py)*np.sinc(m*width/px)*np.sinc(n*height/py)*phase)
    return np.array(result), m


def gather(coeff, order=(1, 1)):
    modes = [(m, n) for n in range(-order[1], order[1]+1) for m in range(-order[0], order[0]+1)]
    return np.array([[coeff[a[1]-b[1]+2*order[1], a[0]-b[0]+2*order[0]] for b in modes] for a in modes])


@pytest.mark.parametrize('gap', [0., 7.])
@pytest.mark.parametrize('epsilon', [4.+.2j, -10.+1j])
def test_uniform_material_variation_recovers_exact_material_convolution(gap, epsilon, jax_runtime):
    jax, jnp, _ = jax_runtime
    cell = make_cell(order=(1, 1))
    direction = np.array([.3+.1j, -.2+.05j, .4-.1j])
    def value(t):
        e = epsilon+t*jnp.asarray(direction)
        return jnp.stack(cell.evaluate(gap=gap, epsilon=e[1:], background=e[0], xp=jnp)[1])
    _, tangent = jax.jit(lambda t: jax.jvp(value, (t,), (jnp.asarray(1.),)))(jnp.asarray(0.))
    shape, _ = rectangle_transforms(gap)
    material = (direction[1]-direction[0])*shape[0]+(direction[2]-direction[0])*shape[1]
    material[2, 2] += direction[0]
    de = gather(material)
    expected = np.array([de, de, -de/epsilon**2, 0*de, 0*de])[:, None]
    np.testing.assert_allclose(tangent, expected, rtol=0, atol=3e-10)


@pytest.mark.parametrize('epsilon', [(4., 9.), (12.25, -10.+1j)])
def test_contact_opening_derivative_against_exact_laminate_integrals(epsilon, jax_runtime):
    jax, jnp, _ = jax_runtime
    order = (2, 1)
    cell = make_cell(height=560., order=order, quadrature=24)
    def value(g):
        return jnp.stack(cell.evaluate(gap=g, epsilon=epsilon, xp=jnp)[1])
    _, tangent = jax.jit(lambda g: jax.jvp(value, (g,), (jnp.asarray(1.),)))(jnp.asarray(0.))
    shape, m = rectangle_transforms(0., order=order, height=560.)
    derivative = shape*np.array([1, -1])[:, None, None]*(1j*np.pi*m/700.)
    def integrals(e):
        coeff = np.einsum('s,svu->vu', np.asarray(e)-1., shape)
        coeff[2, 4] += 1.
        diff = np.einsum('s,svu->vu', np.asarray(e)-1., derivative)
        return gather(coeff, order), gather(diff, order)
    e, de = integrals(epsilon)
    r, dr = integrals(1/np.asarray(epsilon))
    ei, ri = np.linalg.inv(e), np.linalg.inv(r)
    expected = np.array([-ri@dr@ri, de, -ei@de@ei, 0*de, 0*de])[:, None]
    np.testing.assert_allclose(tangent, expected, rtol=0, atol=3e-10)


@pytest.mark.parametrize('gap', [0., 7.])
def test_joint_gap_and_complex_material_jvp_matches_resolved_secants(gap, jax_runtime):
    jax, jnp, _ = jax_runtime
    cell = make_cell(order=(1, 1))
    base = np.array([1.+.2j, 4.+.8j, 9.+.3j])
    direction = np.array([.05+.02j, .1-.03j, -.2+.01j])
    # At contact keep the gap fixed; a two-sided material derivative is valid.
    opening = .3 if gap else 0.
    def value(t, xp):
        e = xp.asarray(base)+t*xp.asarray(direction)
        coeff, slots, _ = cell.evaluate(gap=gap+opening*t, epsilon=e[1:], background=e[0], xp=xp)
        return coeff, xp.stack(slots)
    _, tangent = jax.jit(lambda t: jax.jvp(lambda x: value(x, jnp), (t,), (jnp.asarray(1.),)))(jnp.asarray(0.))
    slopes = []
    for step in [2e-5, 1e-5]:
        low, high = value(-step, np), value(step, np)
        slopes.append(tuple((b-a)/(2*step) for a, b in zip(low, high)))
    for i in range(2):
        np.testing.assert_allclose(slopes[0][i], slopes[1][i], rtol=2e-5, atol=2e-8)
        np.testing.assert_allclose(tangent[i], slopes[1][i], rtol=2e-5, atol=2e-8)


@pytest.mark.parametrize('gap', [0., 7.])
def test_reverse_gap_derivative_matches_the_forward_linear_functional(gap, jax_runtime):
    jax, jnp, program = jax_runtime
    _, value, derivative = program('metal')
    def observable(g):
        _, slots = value(g)
        return jnp.real(jnp.sum(slots*jnp.arange(slots.size).reshape(slots.shape)/slots.size))
    reverse = jax.jit(jax.grad(observable))(jnp.asarray(gap))
    _, tangent = derivative(jnp.asarray(gap))
    slots = tangent[1]
    expected = np.real(np.sum(np.asarray(slots)*np.arange(slots.size).reshape(slots.shape)/slots.size))
    np.testing.assert_allclose(reverse, expected, rtol=1e-10, atol=1e-10)


def test_numpy_tiles_and_single_expression_jax_have_the_same_coefficients(jax_runtime):
    jax, jnp, _ = jax_runtime
    cell = make_cell(quadrature=32)
    value, slots, meta = cell.evaluate(gap=0., epsilon=MATERIALS['metal'])
    def replay(g):
        c, s, m = cell.evaluate(gap=g, epsilon=MATERIALS['metal'], xp=jnp)
        return c, jnp.stack(s), m['field_tile_rows']
    got, gs, rows = jax.jit(replay)(jnp.asarray(0.))
    np.testing.assert_allclose(got, value, rtol=0, atol=3e-10)
    np.testing.assert_allclose(gs, slots, rtol=0, atol=3e-10)
    assert meta['field_tile_rows'] == 64 and rows == 96


@pytest.mark.parametrize('change', [dict(gap=-1.), dict(gap=301.), dict(gap=np.nan),
                                    dict(epsilon=(4., -10.)), dict(epsilon=(4., 9.-1j))])
def test_invalid_eager_replay_is_refused(change):
    settings = dict(gap=0., epsilon=(4., 9.))
    settings.update(change)
    with pytest.raises(ValueError):
        make_cell().evaluate(**settings)


def test_invalid_jit_replay_poisons_every_output(jax_runtime):
    jax, jnp, _ = jax_runtime
    cell = make_cell()
    def value(gap, e):
        coeff, slots, meta = cell.evaluate(gap=gap, epsilon=e, xp=jnp)
        return coeff, jnp.stack(slots), meta['valid']
    fn = jax.jit(value)
    for gap, e in [(-1., (4., 9.)), (0., (4., -10.)), (np.nan, (4., 9.))]:
        coeff, slots, valid = fn(jnp.asarray(gap), jnp.asarray(e))
        assert not valid
        assert np.isnan(coeff).all() and np.isnan(slots).all()


@pytest.mark.parametrize('change', [dict(order=True), dict(order=(-1, 2)), dict(quadrature=7),
                                    dict(quadrature=True), dict(basis=[[700., 1.], [0., 560.]]),
                                    dict(widths=(400., 400.)), dict(height=561.)])
def test_invalid_static_geometry_or_orders_are_refused(change):
    with pytest.raises(ValueError):
        make_cell(**change)
