"""Physical and variational controls for the opt-in conforming-gap study."""
import numpy as np
import pytest

from studies.rcwa.nvf_harmonic_energy.nvf_harmonic_energy import material_coefficients, powers
from studies.rcwa.nvf_thin_gap_energy.nvf_thin_gap_energy import PERIOD, ThinGapEnergyModel, _periodic_line

pytestmark = pytest.mark.research


def _li(order, gap, epsilon=12.25, width=350.):
    from tests.fixtures.rcwa.contact_instruments import rectangular_li
    from wavebit.lattice import as_lattice

    return rectangular_li(np, [dict(type='rectangle', lx=width, ly=PERIOD-gap, material=epsilon)],
                          1., order, order, as_lattice((PERIOD, PERIOD)), np.complex128)


def test_closed_bar_recovers_independent_constitutive_matrices():
    model = ThinGapEnergyModel(3, rows=4)
    n, m = np.meshgrid(np.arange(-3, 4), np.arange(-3, 4), indexing='ij')
    phase = (-1.)**(n.ravel()+m.ravel())
    for epsilon, width in ((12.25, 350.), (6.25, 440.), (1., 270.)):
        actual, metadata = model.matrices(0., epsilon, width)
        for got, want in zip(actual, _li(3, 0., epsilon, width)):
            # Li uses a [0,1] cell; this study's Fourier origin is its center.
            want = phase[:, None]*want*phase[None]
            np.testing.assert_allclose(got, want, atol=2e-10, rtol=0)
        assert metadata['cell_residual'] < 1e-9


def test_projection_material_mean_field_and_energy_are_resolved():
    packs = []
    for q in (12, 16):
        model = ThinGapEnergyModel(3, rows=4, quadrature=q)
        coeffs, _ = model.coefficients(.3, 6.25, 440.)
        exact = material_coefficients(6, [dict(lx=440., ly=699.7, epsilon=6.25)])
        np.testing.assert_allclose(coeffs[7], exact, atol=2e-13, rtol=0)
        # Any periodic H1 corrector has average gradient zero, even in a gap.
        np.testing.assert_allclose(coeffs[:4, 6, 6].reshape(2, 2), np.eye(2), atol=1e-12, rtol=0)
        packs.append(coeffs)
        for pol in ('p', 's'):
            result = model.transmission(.3, 6.25, 440., polarization=pol)
            assert result['energy_error'] < 1e-11
            assert result['hermiticity'] < 1e-11
    np.testing.assert_allclose(*packs, atol=2e-12, rtol=0)


def _ordinary_q1_coefficients(model, gap, epsilon, width):
    """Independent unscaled periodic nodal solve at a strictly positive gap.

    Both ends of the body have distinct nodal unknowns. Physical dy enters the
    ordinary gradients as 1/dy, with no jump variable or divided test equation.
    This control checks that the contact extension represents a real Galerkin
    forward problem, rather than merely prescribing an endpoint derivative.
    """
    q = model.quadrature
    g, w = gap/PERIOD, width/PERIOD
    starts = np.array([-.5, -w/2, w/2])
    sizes = np.array([(1-w)/2, w, (1-w)/2])
    xe = np.r_[(starts[:, None]+sizes[:, None]*model.fractions[:-1]).ravel(), .5]
    ye = np.r_[(1-g)*(model.yedges-.5), (1-g)*(-.5)+1]
    point, weight, fx, dxref = _periodic_line(len(xe)-1, q)
    _, _, fy, dyref = _periodic_line(len(ye)-1, q)
    dx, dy = np.diff(xe), np.diff(ye)
    x = (xe[:-1, None]+dx[:, None]*point).ravel()
    y = (ye[:-1, None]+dy[:, None]*point).ravel()
    wx, wy = (dx[:, None]*weight).ravel(), (dy[:, None]*weight).ravel()
    dfx, dfy = dxref/np.repeat(dx, q)[:, None], dyref/np.repeat(dy, q)[:, None]
    epsx = np.where(abs(x) < w/2, epsilon, 1.)
    body = np.repeat(np.arange(len(dy)) < model.rows, q)
    nx, ny = len(dx), len(dy)
    matrix, right = np.zeros((nx*ny, nx*ny)), np.zeros((nx*ny, 2))
    for mask, ep in ((body, epsx), (~body, np.ones_like(x))):
        wy_part = wy*mask
        mx, kx = fx.T@((wx*ep)[:, None]*fx), dfx.T@((wx*ep)[:, None]*dfx)
        my, ky = fy.T@(wy_part[:, None]*fy), dfy.T@(wy_part[:, None]*dfy)
        matrix += np.kron(my, kx)+np.kron(ky, mx)
        right[:, 0] -= np.kron(fy.T@wy_part, dfx.T@(wx*ep))
        right[:, 1] -= np.kron(dfy.T@wy_part, fx.T@(wx*ep))
    potential = np.zeros((nx*ny, 2))
    potential[1:] = np.linalg.solve(matrix[1:, 1:], right[1:])
    potential = potential.reshape(ny, nx, 2)
    bx = np.einsum('yi,ijc,xj->yxc', fy, potential, dfx)
    by = np.einsum('yi,ijc,xj->yxc', dfy, potential, fx)
    basis = np.stack([bx, by], axis=2)+np.eye(2)
    material = np.where(body[:, None], epsx[None], 1.)
    fields = np.stack([basis[:, :, 0, 0], basis[:, :, 0, 1], basis[:, :, 1, 0], basis[:, :, 1, 1],
                       material*np.sum(basis[:, :, :, 0]**2, axis=2),
                       material*np.sum(basis[:, :, :, 0]*basis[:, :, :, 1], axis=2),
                       material*np.sum(basis[:, :, :, 1]**2, axis=2), material])
    ex = np.exp(-2j*np.pi*model.modes[:, None]*x[None])*wx
    ey = np.exp(-2j*np.pi*model.modes[:, None]*y[None])*wy
    return np.einsum('ny,ayx,mx->anm', ey, fields, ex)


@pytest.mark.parametrize('gap', [.7, 70.])
def test_scaled_contact_space_equals_ordinary_conforming_galerkin_at_positive_gap(gap):
    model = ThinGapEnergyModel(2, rows=4, edge_fraction=1e-4, quadrature=8, grade_y=False)
    expected = _ordinary_q1_coefficients(model, gap, 6.25, 440.)
    actual, _ = model.coefficients(gap, 6.25, 440.)
    np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=0)


@pytest.mark.parametrize('polarization', ['p', 's'])
def test_jit_contact_derivative_passes_independent_physics_gate(polarization):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    model = ThinGapEnergyModel(7, rows=8, edge_fraction=1e-5)
    value, derivative = jax.jit(jax.value_and_grad(
        lambda gap: model.transmission(gap, xp=jnp, polarization=polarization)['T']))(0.)
    closed = float(powers(7, _li(7, 0.), polarization=polarization)[0])
    slopes = [(float(powers(7, _li(7, h), polarization=polarization)[0])-closed)/h
              for h in (1e-3, 5e-4)]
    expected = 2*slopes[1]-slopes[0]
    assert float(value) == pytest.approx(closed, abs=2e-10)
    # Stricter than the original s failure's 1e-6 gate; no production test changes.
    assert float(derivative) == pytest.approx(expected, abs=1e-7, rel=0)
    jax.clear_caches()


@pytest.mark.parametrize('polarization', ['p', 's'])
def test_jit_live_gap_material_width_gradients_match_numpy(polarization):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    model = ThinGapEnergyModel(2, rows=4, edge_fraction=1e-4, quadrature=8)
    fun = lambda p: model.transmission(*p, xp=jnp, polarization=polarization)['T']
    parameters = np.array([17., 6.25, 440.])
    value, derivative = jax.jit(jax.value_and_grad(fun))(jnp.asarray(parameters))
    expected = model.transmission(*parameters, polarization=polarization)['T']
    assert float(value) == pytest.approx(expected, abs=2e-10)
    for i, step in enumerate((1e-3, 1e-5, 1e-3)):
        delta = np.eye(3)[i]*step
        plus = model.transmission(*(parameters+delta), polarization=polarization)['T']
        minus = model.transmission(*(parameters-delta), polarization=polarization)['T']
        assert float(derivative[i]) == pytest.approx((plus-minus)/(2*step), rel=3e-6, abs=2e-9)
    jax.clear_caches()


def test_normal_static_energy_derivative_converges_to_thin_laminate_limit():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    # Local series-laminate derivative, averaged over the inclusion's x width.
    expected = -12.25*(12.25-1)*.5/PERIOD
    errors = []
    for fraction in (1e-4, 1e-5):
        model = ThinGapEnergyModel(0, rows=4, edge_fraction=fraction)
        derivative = jax.jit(jax.grad(
            lambda gap: model.coefficients(gap, xp=jnp)[0][6, 0, 0].real))(0.)
        errors.append(abs(float(derivative)-expected))
    assert errors[1] < .12*errors[0]
    assert errors[1] < 2e-6
    jax.clear_caches()


def test_positive_gap_derivative_approaches_jit_contact_limit():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    model = ThinGapEnergyModel(3, rows=8, edge_fraction=1e-5)
    derivative = jax.jit(jax.grad(
        lambda gap: model.transmission(gap, xp=jnp, polarization='s')['T']))
    limit = float(derivative(0.))
    errors = [abs(float(derivative(g))-limit) for g in (1e-4, 1e-5, 1e-6)]
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 5e-8
    jax.clear_caches()


def test_zero_contrast_does_not_freeze_material_derivative():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    model = ThinGapEnergyModel(2, rows=4, edge_fraction=1e-4, quadrature=8)
    derivative = jax.jit(jax.grad(
        lambda epsilon: model.transmission(17., epsilon, 440., jnp, polarization='s')['T']))(1.)
    step = 1e-4
    plus = model.transmission(17., 1+step, 440., polarization='s')['T']
    minus = model.transmission(17., 1-step, 440., polarization='s')['T']
    assert float(derivative) == pytest.approx((plus-minus)/(2*step), rel=2e-5, abs=1e-9)
    jax.clear_caches()


def test_invalid_domain_is_rejected():
    model = ThinGapEnergyModel(1, rows=4)
    for parameters in ((-1., 4., 350.), (700., 4., 350.), (0., 0., 350.), (0., 4., 700.)):
        with pytest.raises(ValueError, match='requires'):
            model.coefficients(*parameters)
