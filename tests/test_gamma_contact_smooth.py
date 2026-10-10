"""A material seam's moving endpoint must not create a jump off the interface."""
import numpy as np
import pytest
from wavebit._core import parametric as P
from wavebit._core.lattice import as_lattice

CELL = as_lattice(period=(700., 700.))


def shapes(height):
    return [dict(type='rectangle', cx=-150., cy=0., lx=300., ly=300., angle=0., material=4.),
            dict(type='rectangle', cx=150., cy=0., lx=300., ly=height, angle=0., material=9.)]


def nvf_projector(height):
    s = shapes(height)
    plan = P.plan_gamma(s, CELL)
    nx, ny = P._piece_field(np, np.array([340.]), np.array([490.]), s, plan, None,
                            CELL, np.float64)
    return np.array([nx*nx, nx*ny, ny*ny])


def test_moving_a_run_end_has_no_off_interface_value_jump():
    large = np.max(np.abs(nvf_projector(280.+1e-4)-nvf_projector(280.-1e-4)))
    small = np.max(np.abs(nvf_projector(280.+1e-6)-nvf_projector(280.-1e-6)))
    assert small < 0.02*large
    assert small < 1e-6


def test_field_derivative_through_the_endpoint_matches_two_sided_differences():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    plan = P.plan_gamma(shapes(280.), CELL)
    def loss(height):
        nx, ny = P._piece_field(jnp, jnp.array([340.]), jnp.array([490.]), shapes(height),
                                plan, None, CELL, jnp.float64)
        return (nx*nx + .3*nx*ny)[0]
    compiled = jax.jit(jax.value_and_grad(loss))
    for height in (279.5, 280., 280.5):
        value, derivative = compiled(jnp.asarray(height))
        assert np.isfinite(value) and np.isfinite(derivative)
        # Squared tangential excess is C1 at the endpoint; its central
        # secant error is O(h), rather than O(h**2), right at that join.
        for h in (1e-5, 1e-6):
            fd = (float(loss(height+h))-float(loss(height-h)))/(2*h)
            assert float(derivative) == pytest.approx(fd, abs=2e-8, rel=1e-4)


def test_solve_derivative_matches_the_endpoint_crossing_secant():
    import wavebit
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    grid = 64
    rows = (np.arange(grid)+.5)*700/grid-350
    center = 2*rows[rows<150].max()
    def solver(height, backend):
        return wavebit.rcwa(backend=backend, period=(700.,700.), wavelength=900.,
                           top=1., bot=2.25, theta=.2, phi=.13, psi=.2, thickness=(300.,),
                           truncation_order=(2,2), shape_nvf_grid_size=grid,
                           structure=[dict(background=1., shapes=shapes(height))])
    frozen = solver(center, 'numpy').freeze_geometry()
    def loss(height):
        return jnp.sum(solver(height, 'jax').solve(geometry=frozen).T).real
    got = float(jax.jit(jax.grad(loss))(jnp.asarray(center)))
    assert np.isfinite(got) and abs(got) > 1e-6
    for h in (1e-3, 1e-4):
        plus = float(np.sum(solver(center+h,'numpy').solve().T))
        minus = float(np.sum(solver(center-h,'numpy').solve().T))
        assert got == pytest.approx((plus-minus)/(2*h), rel=2e-3, abs=1e-7)
