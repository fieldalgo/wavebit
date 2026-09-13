"""Derivative instrument checks, not acceptance of the candidate's physics."""
import numpy as np
import pytest

from studies import nvf_harmonic_energy as reference

pytest.importorskip('jax')
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from studies.nvf_harmonic_energy_jax import HarmonicEnergyModel, _implicit_solve  # noqa: E402

jax.config.update('jax_enable_x64', True)


@pytest.mark.parametrize('polarization,eigen_rule', [
    ('p', 'regularized'),
    ('s', 'regularized'),
    ('s', 'native'),
])
def test_jit_forward_and_geometry_material_gradient_against_numpy_secants(
        polarization, eigen_rule, monkeypatch):
    if eigen_rule == 'native':
        # This nondegenerate control isolates the cell derivative from the
        # existing production eigenvalue broadening. It is not a dispatch change
        # and cannot be used at the uniform/zero-contrast degeneracies below.
        import jax.lax.linalg as jll
        from wavebit._core import solve_2d

        def native(matrix):
            if isinstance(matrix, np.ndarray):
                return np.linalg.eig(matrix)
            out = jll.eig(matrix, compute_left_eigenvectors=False,
                          compute_right_eigenvectors=True, enable_eigvec_derivs=True)
            return out[0], out[-1]
        monkeypatch.setattr(solve_2d, 'eig', native)
    # Translation makes the cell matrix complex Hermitian, not real symmetric.
    model = HarmonicEnergyModel(2, 7, polarization=polarization)

    def shapes(p):
        return [dict(lx=p[0], ly=p[1], angle=p[2], epsilon=p[3], cx=17., cy=-21.)]

    objective = lambda p: model.transmission(shapes(p))['T']
    p = np.array([440., 270., .31, 6.25])
    value, gradient = jax.jit(jax.value_and_grad(objective))(jnp.asarray(p))
    expected = reference.transmission(2, 7, shapes(p), polarization=polarization)
    assert float(value) == pytest.approx(expected['T'], abs=2e-11)
    for i, step in enumerate([1e-3, 1e-3, 1e-6, 1e-5]):
        for scale in (1., .5):
            delta = np.eye(4)[i]*step*scale
            secant = (reference.transmission(2, 7, shapes(p+delta), polarization=polarization)['T']
                       -reference.transmission(2, 7, shapes(p-delta), polarization=polarization)['T'])/(2*step*scale)
            assert float(gradient[i]) == pytest.approx(secant, rel=3e-6, abs=2e-9)


def test_exact_contact_gradient_matches_positive_gap_secants():
    model = HarmonicEnergyModel(3, 15)
    shapes = lambda gap: [dict(lx=350., ly=700.-gap, epsilon=12.25)]
    objective = lambda gap: model.transmission(shapes(gap))['T']
    value, gradient = jax.jit(jax.value_and_grad(objective))(0.)
    closed = reference.transmission(3, 15, shapes(0.))['T']
    assert float(value) == pytest.approx(closed, abs=2e-11)
    # No negative gaps: overlapping periodic images are outside this instrument.
    for step in (1e-4, 1e-5):
        secant = (reference.transmission(3, 15, shapes(step))['T']-closed)/step
        assert float(gradient) == pytest.approx(secant, rel=1e-5, abs=2e-9)


def test_split_preserves_values_and_gradients_including_movable_artificial_seam():
    model = HarmonicEnergyModel(2, 7)

    def objective(p, split):
        width, height, epsilon, seam = p
        if split:
            shapes = [dict(lx=seam+width/2, ly=height, epsilon=epsilon,
                           cx=(seam-width/2)/2),
                      dict(lx=width/2-seam, ly=height, epsilon=epsilon,
                           cx=(seam+width/2)/2)]
        else:
            shapes = [dict(lx=width, ly=height, epsilon=epsilon)]
        return model.transmission(shapes)['T']

    p = jnp.array([600., 300., 4., 31.])
    merged = jax.jit(jax.value_and_grad(lambda p: objective(p, False)))(p)
    split = jax.jit(jax.value_and_grad(lambda p: objective(p, True)))(p)
    np.testing.assert_allclose(split[0], merged[0], atol=1e-10, rtol=0)
    np.testing.assert_allclose(split[1], merged[1], atol=2e-10, rtol=0)
    assert abs(float(split[1][-1])) < 1e-12


def test_zero_contrast_keeps_the_material_derivative_live():
    model = HarmonicEnergyModel(2, 7)
    shapes = lambda epsilon: [dict(lx=440., ly=270., angle=.31, epsilon=epsilon)]
    value, gradient = jax.jit(jax.value_and_grad(
        lambda epsilon: model.transmission(shapes(epsilon))['T']))(1.)
    step = 1e-4
    secant = (reference.transmission(2, 7, shapes(1.+step))['T']
               -reference.transmission(2, 7, shapes(1.-step))['T'])/(2*step)
    assert np.isfinite(float(value))
    assert float(gradient) == pytest.approx(secant, rel=2e-5, abs=1e-9)


def test_jit_rejects_failed_primal_and_failed_adjoint_solves():
    failed = HarmonicEnergyModel(1, 7, maxiter=1)
    coefficients, residuals = jax.jit(failed.corrector)(reference.tilted())
    assert not np.isfinite(np.asarray(coefficients)).all()
    assert not np.isfinite(np.asarray(residuals)).all()

    # The forward RHS is an eigenvector, so it converges in one iteration. The
    # adjoint RHS is not: returning a finite gradient here would hide its failure.
    def objective(p):
        matrix = jnp.diag(jnp.array([2.+p, 4.+p]))
        solution = _implicit_solve(lambda x: matrix@x, jnp.array([1., 0.]), 1)
        return jnp.sum(solution)
    value, gradient = jax.jit(jax.value_and_grad(objective))(.3)
    assert float(value) == pytest.approx(1/2.3)
    assert not np.isfinite(float(gradient))


def test_jax_matrices_match_numpy_with_exact_product_grid():
    shapes = [dict(lx=440., ly=270., angle=.31, epsilon=6.25, cx=17., cy=-21.)]
    expected, _ = reference.matrices(2, 7, shapes)
    for grid in (40, 64):
        slots, metadata = jax.jit(HarmonicEnergyModel(2, 7, grid).matrices)(shapes)
        for actual, want in zip(slots, expected):
            np.testing.assert_allclose(actual, want, atol=2e-11, rtol=0)
        assert np.max(np.asarray(metadata['cell_residuals'])) < 1e-10
        assert float(metadata['hermiticity']) < 1e-10
    with pytest.raises(ValueError, match='prevent aliasing'):
        HarmonicEnergyModel(2, 7, 36)


@pytest.mark.xfail(strict=True, reason='finite Fourier corrector misses the normal-field thin-gap variation')
def test_s_contact_derivative_matches_independent_li_instrument():
    # An explicit unresolved physics gate. Finite-model AD/FD agreement alone
    # cannot detect this error, nor can the earlier p-polarized contact fixture.
    from contact_instruments import rectangular_li
    from wavebit._core.lattice import as_lattice

    def independent(gap):
        shapes = [dict(type='rectangle', lx=350., ly=700.-gap, material=12.25)]
        slots = rectangular_li(np, shapes, 1., 7, 7, as_lattice((700., 700.)), np.complex128)
        return float(reference.powers(7, slots, polarization='s')[0])
    model = HarmonicEnergyModel(7, 31, polarization='s')
    derivative = jax.jit(jax.grad(lambda gap: model.transmission(
        [dict(lx=350., ly=700.-gap, epsilon=12.25)])['T']))(0.)
    expected = (independent(1e-4)-independent(0.))/1e-4
    assert float(derivative) == pytest.approx(expected, abs=1e-6)
