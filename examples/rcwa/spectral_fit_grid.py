"""Refit a spectrum on finer normal-field grids before accepting fitted dimensions.

Run with the base Wavebit installation (NumPy and SciPy). ``fit_grid_ladder`` is
application code: copy it with your model and choose tolerances in your own units.
It checks normal-field discretization at fixed Fourier order, not Maxwell accuracy.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from wavebit import Material, Units, rcwa


@dataclass
class FitStage:
    grid: int
    parameters: np.ndarray
    noise_std: np.ndarray
    residual_norm: float
    parameter_change: float | None
    spectrum_change: float | None
    optimizer_success: bool


@dataclass
class GridFit:
    stages: list[FitStage]
    converged: bool
    reason: str

    @property
    def parameters(self):
        return self.stages[-1].parameters.copy()


def _positive_vector(value, shape, name):
    if np.iscomplexobj(value):
        raise ValueError(f'{name} must be real')
    array = np.broadcast_to(np.asarray(value, dtype=float), shape).copy()
    if not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise ValueError(f'{name} must contain finite positive numbers')
    return array


def fit_grid_ladder(model_factory, initial, data, *, grids, noise_sigma,
                    parameter_tolerance, spectrum_tolerance, max_nfev=40, on_stage=None):
    """Warm-start least-squares fits; require two consecutive refinement checks.

    ``model_factory(grid)(parameters)`` returns (spectrum, Jacobian), with a
    checked finite real spectrum of shape (m,) and Jacobian of shape (m, n).
    The model must reject invalid solves and handle any geometry re-preparation.
    ``noise_sigma`` is the known independent observation noise, not a grid error.
    Tolerances are absolute, per parameter / observation, in their physical units.
    A change passes only if *every* component is within its tolerance. Spectra
    are compared on adjacent grids at the SAME finer-grid fitted parameters.
    A failed optimizer, deficient Jacobian rank or exhausted grid list is not
    convergence. A failed coarse fit is retried at the next grid, with its status
    retained; it cannot count toward a pass. Two passes are evidence on this
    ladder, not an error bound. ``on_stage``, when given, receives each completed
    FitStage for progress reporting or checkpointing.
    """
    if np.iscomplexobj(initial) or np.iscomplexobj(data):
        raise ValueError('initial and data must be real')
    x = np.asarray(initial, dtype=float).copy()
    data = np.asarray(data, dtype=float)
    if x.ndim != 1 or data.ndim != 1 or not x.size or data.size < x.size:
        raise ValueError('initial and data must be vectors with m >= n > 0')
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(data)):
        raise ValueError('initial and data must be finite')
    grids = tuple(grids)
    if (len(grids) < 3 or any(isinstance(g, (bool, np.bool_)) or
                            not isinstance(g, (int, np.integer)) or g <= 0 for g in grids)
            or any(b <= a for a, b in zip(grids, grids[1:]))):
        raise ValueError('provide at least three strictly increasing positive integer grids')
    sigma = _positive_vector(noise_sigma, data.shape, 'noise_sigma')
    ptol = _positive_vector(parameter_tolerance, x.shape, 'parameter_tolerance')
    ytol = _positive_vector(spectrum_tolerance, data.shape, 'spectrum_tolerance')
    stages, previous_model, consecutive = [], None, 0
    previous_fit_ok = False
    last_failure = None

    def checked(model, p):
        y, jac = (np.asarray(a) for a in model(p))
        if (y.shape != data.shape or jac.shape != (data.size, x.size)
                or np.iscomplexobj(y) or np.iscomplexobj(jac)
                or not np.all(np.isfinite(y)) or not np.all(np.isfinite(jac))):
            raise ValueError('model returned invalid spectrum or Jacobian')
        return y, jac

    for grid in grids:
        model = model_factory(grid)
        # SciPy requests fun and jac at the same point: share that evaluation.
        cached_x, cached_pair, accepted = None, None, None

        def evaluate(p):
            nonlocal cached_x, cached_pair
            if cached_x is None or not np.array_equal(p, cached_x):
                cached_pair = checked(model, p)
                cached_x = p.copy()
            return cached_pair

        def jacobian(p):
            nonlocal accepted
            # SciPy asks for the Jacobian at accepted iterates. Retain that
            # pair when a later trial is rejected before termination, so the
            # final report does not repeat an expensive solve at the accepted x.
            accepted = (p.copy(), tuple(a.copy() for a in evaluate(p)))
            return accepted[1][1] / sigma[:, None]

        fit = least_squares(lambda p: (evaluate(p)[0] - data) / sigma, x,
                            jac=jacobian,
                            x_scale='jac', ftol=1e-10, xtol=1e-10, gtol=1e-10,
                            max_nfev=max_nfev)
        y, jac = (accepted[1] if accepted is not None and np.array_equal(accepted[0], fit.x)
                  else evaluate(fit.x))
        _, singular, vh = np.linalg.svd(jac / sigma[:, None], full_matrices=False)
        full_rank = singular[-1] > (np.finfo(float).eps * max(jac.shape) * singular[0])
        noise_std = (np.sqrt(np.sum((vh.T / singular)**2, axis=1)) if full_rank
                     else np.full(x.shape, np.inf))
        dp = dy = None
        if previous_model is not None:
            dp = float(np.max(np.abs(fit.x - x) / ptol))
            # Refitting can hide model error in shifted parameters. Check the
            # old and new forward models at the finer fit as a separate test.
            old_y, _ = checked(previous_model, fit.x)
            dy = float(np.max(np.abs(y - old_y) / ytol))
        stages.append(FitStage(grid, fit.x.copy(), noise_std,
                               float(np.linalg.norm((y - data) / sigma)),
                               dp, dy, bool(fit.success)))
        if on_stage is not None:
            on_stage(stages[-1])
        fit_ok = bool(fit.success and full_rank)
        last_failure = (f'optimizer failed at grid {grid}: {fit.message}' if not fit.success
                        else f'Jacobian is rank deficient at grid {grid}' if not full_rank else None)
        passes = fit_ok and previous_fit_ok and dp <= 1 and dy <= 1
        consecutive = consecutive + 1 if passes else 0
        if consecutive >= 2:
            return GridFit(stages, True, 'two consecutive grid refinements meet both tolerances')
        previous_model, x = model, fit.x.copy()
        previous_fit_ok = fit_ok
    return GridFit(stages, False, last_failure or 'grid budget exhausted before two consecutive passes')


def rectangle_spectrum(width, grid):
    """A small width-only fit; finite differences keep this example NumPy-only."""
    problem = rcwa.Problem(units=Units(length='nm'),
        stack=rcwa.Stack(period=(700., 700.), top=Material(epsilon=1.),
            bottom=Material(epsilon=2.25), layers=(rcwa.Layer(thickness=180.,
                background=Material(epsilon=1.), regions=(rcwa.Region(
                    geometry=rcwa.Rectangle(size=(width, 210.), angle=.3),
                    material=Material(epsilon=4.)),)),)),
        excitation=rcwa.Excitation(wavelength=np.linspace(800., 1100., 5),
            sources=(rcwa.PlaneWave(theta=.2, phi=.3, polarization='p'),)))
    result = rcwa.run(problem, numerics=rcwa.Numerics(
        orders=(1, 1), shape_nvf_grid_size=grid), backend='numpy')
    if not all(np.all(np.asarray(v)) for v in result.validity.checks.values()):
        raise RuntimeError('invalid optical solve')
    return np.asarray(result.reflection.order(0, 0).efficiency).real


def rectangle_model(grid):
    def model(p):
        h = 1e-3  # nm; verify derivative steps for a different model.
        y = rectangle_spectrum(float(p[0]), grid)
        derivative = (rectangle_spectrum(float(p[0]) + h, grid)
                      - rectangle_spectrum(float(p[0]) - h, grid)) / (2*h)
        return y, derivative[:, None]
    return model


def main():
    true_width = 420.
    data = rectangle_spectrum(true_width, 256)
    report = fit_grid_ladder(rectangle_model, [425.], data,
                            grids=(16, 32, 64, 128, 256, 512), noise_sigma=1e-4,
                            parameter_tolerance=.05, spectrum_tolerance=1e-5)
    for stage in report.stages:
        print(f'grid={stage.grid:4d}, width={stage.parameters[0]:.6f} nm, '
              f'noise-only std={stage.noise_std[0]:.6f} nm, '
              f'weighted residual norm={stage.residual_norm:.3f}, '
              f'parameter/tolerance={stage.parameter_change}, '
              f'spectrum/tolerance={stage.spectrum_change}')
    print(f'grid converged={report.converged}: {report.reason}')
    if not report.converged:
        raise RuntimeError(report.reason)
    # The same-family synthetic target checks this workflow, not optical accuracy.
    np.testing.assert_allclose(report.parameters, [true_width], rtol=0, atol=.05)


if __name__ == '__main__':
    main()
