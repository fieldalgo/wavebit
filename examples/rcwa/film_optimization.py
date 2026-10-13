"""Optimize one antireflection film with the current RCWA API and JAX.

Run after installing wavebit[jax]. Lengths are micrometres; dR/dd is per um.
The quarter-wave answer and an independent Airy formula check this small example.
"""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from wavebit import Material, Units, rcwa

# Explicit application choice, before creating any JAX arrays.
jax.config.update('jax_enable_x64', True)
WAVELENGTH = .8
N_TOP, N_FILM, N_BOTTOM = 1., np.sqrt(1.5), 1.5


def airy_reflectance(thickness):
    """Normal-incidence three-medium formula, independent of the RCWA engine."""
    r01 = (N_TOP - N_FILM) / (N_TOP + N_FILM)
    r12 = (N_FILM - N_BOTTOM) / (N_FILM + N_BOTTOM)
    phase = np.exp(4j * np.pi * N_FILM * thickness / WAVELENGTH)
    return abs((r01 + r12 * phase) / (1 + r01 * r12 * phase)) ** 2


def main():
    initial = .08
    problem = rcwa.Problem(units=Units(length='um'),
        stack=rcwa.Stack(period=(1., 1.), top=Material(epsilon=N_TOP**2),
            bottom=Material(epsilon=N_BOTTOM**2),
            layers=(rcwa.Layer(thickness=initial, background=Material(epsilon=N_FILM**2)),)),
        excitation=rcwa.Excitation(wavelength=WAVELENGTH,
            sources=(rcwa.PlaneWave(polarization='s'),)))
    numerics = rcwa.Numerics(orders=(0, 0))  # Uniform film: only the zeroth order.
    prepared = rcwa.prepare(problem, numerics=numerics, backend='jax')
    prepared.report.raise_if_invalid()

    def live_problem(thickness):
        layer = problem.stack.layers[0].replace(thickness=thickness)
        return problem.replace(stack=problem.stack.replace(layers=(layer,)))

    def reflectance(thickness):
        return prepared.run(live_problem(thickness)).R.sum()

    value_and_grad = jax.jit(jax.value_and_grad(reflectance))
    before, derivative = value_and_grad(jnp.asarray(initial))
    step = 1e-5
    finite_difference = (airy_reflectance(initial + step) - airy_reflectance(initial - step)) / (2 * step)
    np.testing.assert_allclose(float(before), airy_reflectance(initial), atol=2e-14)
    np.testing.assert_allclose(float(derivative), finite_difference, rtol=2e-7, atol=1e-10)

    def objective(x):
        value, gradient = value_and_grad(jnp.asarray(x[0]))
        return float(value), np.array([float(gradient)])

    optimum = minimize(objective, [initial], jac=True, method='L-BFGS-B',
        bounds=[(.02, .30)], options={'maxiter': 15, 'gtol': 1e-10, 'ftol': 1e-14})
    if not optimum.success:
        raise RuntimeError(optimum.message)
    thickness = float(optimum.x[0])
    expected = WAVELENGTH / (4 * N_FILM)
    final = rcwa.run(live_problem(thickness), numerics=numerics, backend='numpy')
    np.testing.assert_allclose(thickness, expected, atol=2e-7, rtol=0)
    np.testing.assert_allclose(float(final.R.sum()), airy_reflectance(thickness), atol=2e-14)
    np.testing.assert_allclose(float(final.R.sum() + final.T.sum()), 1., atol=2e-13)
    assert float(final.R.sum()) < 1e-10
    print(f'dR/dd: JAX {float(derivative):.9f}, Airy difference {finite_difference:.9f} per um')
    print(f'thickness: {initial*1000:.3f} -> {thickness*1000:.6f} nm; quarter-wave {expected*1000:.6f} nm')
    print(f'reflectance: {float(before):.8f} -> {float(final.R.sum()):.3e}; iterations: {optimum.nit}')


if __name__ == '__main__':
    main()
