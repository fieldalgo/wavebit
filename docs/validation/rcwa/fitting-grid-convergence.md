# Normal-field resolution can move a spectral fit

The [automatic refinement example](../../guide/rcwa/fitting-grid-convergence.md)
checks the fitted parameters as well as the forward spectrum. This page records
what its numerical controls establish; it does not certify one normal-field grid
for all geometries.

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
