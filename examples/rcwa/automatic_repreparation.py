"""Fit a rectangle's position across a topology boundary with automatic preparation."""
import jax
import numpy as np
from scipy.optimize import least_squares

from wavebit import Material, Units, rcwa


def problem(parameters):
    offset = parameters[0]
    return rcwa.Problem(units=Units(length='nm'),
        stack=rcwa.Stack(period=(700., 700.), bottom=Material(2.25), layers=(
            rcwa.Layer(thickness=180., regions=(
                rcwa.Region(geometry=rcwa.Rectangle(size=(250., 200.), center=(-50., 0.)),
                            material=Material(4.)),
                rcwa.Region(geometry=rcwa.Rectangle(size=(200., 240.), center=(offset, 30.)),
                            material=Material(5.)))),)),
        excitation=rcwa.Excitation(wavelength=900., sources=(
            rcwa.PlaneWave(theta=.2, phi=.13, polarization='p'),)))


def main():
    jax.config.update('jax_enable_x64', True)
    # A small Laurent calculation isolates preparation recovery from normal-grid
    # discretization. This is a lifecycle example, not a convergence study.
    numerics = rcwa.Numerics(orders=(1, 0), factorization='laurent')
    truth, initial = 200., 160.
    target = float(rcwa.run(problem([truth]), numerics=numerics).R.sum())
    residual = rcwa.compile(problem, lambda result: result.R.sum()/target-1., numerics=numerics)
    fitted = least_squares(lambda p: np.atleast_1d(residual(p)), np.asarray([initial]),
                           jac=lambda p: np.asarray(residual.jacobian(p)).reshape(1, 1),
                           bounds=([140.], [240.]), xtol=1e-12, ftol=1e-12, gtol=1e-12,
                           max_nfev=30)
    assert fitted.success, fitted.message
    np.testing.assert_allclose(fitted.x, [truth], atol=1e-5, rtol=0)
    assert np.linalg.norm(fitted.fun) < 1e-9
    assert residual.preparation_count >= 2
    print(f'Initial center: {initial:.6f} nm (overlapping)')
    print(f'Fitted center:  {fitted.x[0]:.6f} nm (separated); target {truth:.6f} nm')
    print(f'Residual: {np.linalg.norm(fitted.fun):.3e}')
    print(f'Preparations: {residual.preparation_count}; evaluations: {fitted.nfev}')


if __name__ == '__main__':
    main()
