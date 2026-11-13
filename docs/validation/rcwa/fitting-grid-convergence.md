# Normal-field resolution can move a spectral fit

The [automatic refinement example](../../guide/rcwa/fitting-grid-convergence.md)
checks the fitted parameters as well as the forward spectrum. This page records
what its numerical controls establish; it does not certify one normal-field grid
for all geometries.

The measurements on solver `18bbcbfd` below describe the earlier normal-field
construction. The integral construction landed in `2056a39e`; its fitting and
grid convergence must be measured separately. Historical failures here are not
measurements of that newer implementation.

## A complete small fit

The example declares a 700 nm square period, air above glass (`epsilon=2.25`),
and a 180 nm layer containing a rotated rectangle (`epsilon=4`, height 210 nm,
angle 0.3 rad). It fits the rectangle width to five zeroth-order p reflectances
at wavelengths 800–1100 nm, with incidence angles 0.2 and 0.3 rad. Fourier orders
are `(1, 1)`. The target is generated at width 420 nm and normal grid 256.

On solver commit `18bbcbfd`, NumPy 2.5.1, CPU double precision:

| Normal grid | Fitted width (nm) | Maximum parameter change / 0.05 nm | Maximum same-point spectrum change / 1e-5 |
| --- | ---: | ---: | ---: |
| 16 | 424.795633 | — | — |
| 32 | 420.235932 | 91.194 | 53.522 |
| 64 | 420.160407 | 1.511 | 1.219 |
| 128 | 420.041553 | 2.377 | 1.246 |
| 256 | 420.000000 | 0.831 | 0.443 |
| 512 | 419.998461 | 0.031 | 0.021 |

All optimizers succeeded. The ladder ending at 256 **failed** the predeclared
requirement of two consecutive acceptable refinements. Extending it to 512,
without changing either tolerance, passed. Notice that the 64→128 change is
larger than 32→64: finer sampling need not give monotone parameter changes.
The noise-only width standard deviation for observation noise `1e-4` is about
0.60 nm, reported separately. The example verifies the recovered width within
0.05 nm of this same-family synthetic target; its small Fourier order is not an
optical accuracy qualification.

## Four fitted parameters and thirty observations

The study
[`nvf_fitting_grid.py`](https://github.com/fieldalgo/wavebit/blob/main/studies/rcwa/nvf_fitting_grid/nvf_fitting_grid.py)
declares a rectangle painted under an ellipse in a 700 nm square cell. It fits
rectangle width and angle, ellipse center x, and layer thickness using 15 s and
15 p zeroth reflected orders at wavelengths 800–1200 nm, fixed orders `(4, 4)`.
The target parameters are `(420 nm, 0.3 rad, 90 nm, 300 nm)`; all other geometry,
materials, incidence and the fixed grid-128 target are declared in the script.

```shell
python studies/rcwa/nvf_fitting_grid/nvf_fitting_grid.py --output artifacts/nvf-fitting-grid.json
```

It checks parameter changes against `(0.02 nm, 4e-5 rad, 0.015 nm, 0.004 nm)` and
same-point reflectance changes against `1e-5`, with independent noise sigma
`1e-4`. It writes the target, its Jacobian, fit stages and outcome to a new JSON
file; exit code 2 means the supplied ladder did not converge. A finer model
fitted to a coarse synthetic target need not recover the generating parameters.
Refinement stability and recovery of known truth are different experiments.

### Reproduction of the coarse-grid shift

With the fixed grid-128 target, a grid-256 fit on solver `18bbcbfd` gave:

| Parameter | Generating value | Grid-256 fit | Shift | Grid-128 noise-only standard deviation |
| --- | ---: | ---: | ---: | ---: |
| Rectangle width (nm) | 420 | 420.341686 | +0.341686 | 0.086855 |
| Rectangle angle (rad) | 0.3 | 0.299912115 | −0.000087885 | 0.000187304 |
| Ellipse center x (nm) | 90 | 89.987132 | −0.012868 | 0.068026 |
| Thickness (nm) | 300 | 299.730583 | −0.269417 | 0.017574 |

The width and thickness shifts are about 3.9 and 15.3 times their noise-only
standard deviations. The residual norm after refitting is `0.00195635`; at
noise sigma `1e-4` its squared weighted norm is about 382.7 for 30 observations
and four parameters. Refining the model does not make that coarse synthetic
target a fine-grid truth.

This CPU double-precision reproduction used JAX 0.12.0.1, NumPy 2.5.4 and SciPy
1.18.1, all 30 observations in one batch, and the original LM settings
(`x_scale='jac'`, `xtol=1e-10`, `ftol=1e-13`, `gtol=3e-16`, `max_nfev=40`).
It took seven evaluations. A separate NumPy grid-128 spectrum on NumPy 2.5.1
agreed with the retained target to `2.8e-13` maximum absolute difference.
These reproduce the grid effect; they do not establish convergence at 1024.

### Automatic refinement through the default resolution

The full automatic ladder used the same fixed target and solver `18bbcbfd`,
JAX 0.12.0.1, NumPy 2.5.4, SciPy 1.18.1 and 30 lanes per batch. Unlike the LM
reproduction above, the example uses TRF with `x_scale='jac'`,
`ftol=xtol=gtol=1e-10`, `max_nfev=40` and the preceding fit as its next initial
point. The [reviewed measurement](https://github.com/fieldalgo/wavebit/blob/main/studies/rcwa/nvf_fitting_grid/measurement.json) preserves the target, target
Jacobian, fitted parameters, noise-only standard deviations and settings.

| Normal grid | Width (nm) | Angle (rad) | Center x (nm) | Thickness (nm) |
| --- | ---: | ---: | ---: | ---: |
| 128 | 420.000000 | 0.300000000 | 90.000000 | 300.000000 |
| 256 | 420.341686 | 0.299912115 | 89.987132 | 299.730583 |
| 512 | 420.393242 | 0.299577816 | 90.128351 | 299.547011 |
| 1024 | 420.352219 | 0.299411942 | 90.218438 | 299.475121 |

| Normal grid | Maximum parameter change / tolerance | Maximum same-point spectrum change / tolerance | Weighted residual norm |
| --- | ---: | ---: | ---: |
| 128 | — | — | 0.000 |
| 256 | 67.354 | 236.497 | 19.564 |
| 512 | 45.893 | 127.014 | 32.117 |
| 1024 | 17.972 | 49.105 | 37.641 |

A change ratio must be at most one to pass. **All four optimizers succeeded,
but every adjacent refinement failed both grid checks.** The completed ladder
therefore returned `converged=false` and exit code 2, with unchanged tolerances.
The default 1024 did not establish convergence for this target and requested
accuracy. The noise-only thickness standard deviation remains about 0.018 nm;
it does not cover the displacement caused by changing the grid. Neither the
coarse synthetic target nor this fixed-order study supplies an independent
optical reference.

### Derivative check on the same baseline

At the generating parameters on grid 128, compare the fresh JAX Jacobian with
Richardson-extrapolated central differences, using steps
`(0.01 nm, 0.0001 rad, 0.01 nm, 0.01 nm)` and their halves. The maximum absolute
difference in each column, divided by that column's maximum absolute AD entry,
was `(0.0021014, 0.0041754, 0.0015886, 2.6e-11)`. The first three columns **fail**
the predeclared `1e-3` criterion; thickness passes. All evaluations were finite.

This is a separate limitation of the baseline painted normal field. The
refinement wrapper does not repair its derivative artifacts; see the
[normal-field limits](normal-field-known-limits.md). A stable fit and a derivative
check are separate requirements, and the measured finite-order extension must
be requalified when the normal-field construction changes.

## Controls and scope

`tests/rcwa/test_spectral_fit_grid_example.py` uses exact linear inverse problems
to check bias reduction and noise covariance, and negative controls where every
optimizer succeeds while grid convergence fails. It checks an accidental
plateau, unchanged parameters hiding forward drift, rank deficiency, nonfinite
values/Jacobians and exhausted optimizer/grid budgets. These controls qualify
the stopping logic; the optical example exercises the actual public solver.

The default normal grid remains 1024 (or the Fourier bandwidth minimum if
larger). Its value alone supplies no error estimate. No solver tolerance or
reference value was changed to pass these checks. Independent Maxwell accuracy,
Fourier-order convergence and the ellipse's geometric approximation remain
separate validation questions.
