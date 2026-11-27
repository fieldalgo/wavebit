# Normal-field resolution can move a spectral fit

The [automatic refinement example](../../guide/rcwa/fitting-grid-convergence.md)
checks the fitted parameters as well as the forward spectrum. This page records
what its numerical controls establish; it does not certify one normal-field grid
for all geometries.

The measurements labelled solver `18bbcbfd` describe the earlier normal-field
construction. The integral construction landed in `2056a39e`; its fixed-point
and refitting measurements are reported separately below. Historical failures here are not
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

## Fixed-point requalification of the integral construction

On solver `2056a39e`, repeat the thirty-observation case at the generating
parameters, with the same Fourier orders and CPU double precision. The
[reviewed fixed-point record](https://github.com/fieldalgo/wavebit/blob/main/studies/rcwa/nvf_fitting_grid/integral-fixed-point.json)
retains all values, all four Jacobian columns, finite differences, versions and
provenance. The study uses bounded groups of five scalar JIT calls; it preserves
all observations while avoiding the memory expansion of an outer spectral
`vmap`. Two-wavelength, two-polarization comparisons against independent scalar
calls and the public wavelength-array route agree.

At grid 128, the original Richardson steps give relative column errors
`(6.68e-9, 7.58e-9, 1.38e-9, 9.15e-12)`. **All four pass** the unchanged `1e-3`
AD/FD criterion. This checks differentiation of the sampled model, separately
from convergence when its grid changes.

For adjacent grids, the declared spectrum criterion is maximum absolute change
`1e-5`. Each Jacobian column's maximum difference is divided by that column's
maximum magnitude on the finer grid and must be at most `1e-3`.

| Grid change | Maximum spectrum difference | Width Jacobian | Angle Jacobian | Center-x Jacobian | Thickness Jacobian |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 → 256 | 2.393e-5 | 9.329e-4 | 1.509e-2 | 7.576e-4 | 2.718e-5 |
| 256 → 512 | 8.917e-6 | 5.978e-4 | 5.599e-3 | 2.193e-4 | 1.335e-5 |
| 512 → 1024 | 2.621e-6 | 2.468e-4 | 3.714e-3 | 2.888e-4 | 8.106e-6 |

The last two spectrum comparisons pass. The angle Jacobian still changes by
0.371% at the final transition, exceeding the 0.1% criterion; the other three
columns pass. **Neither AD/FD agreement nor stable spectra establish Jacobian
grid convergence.** These fixed-point results also do not establish stability
of fitted parameters. Reproduce them with `check_grid.py` as described in the
[study README](https://github.com/fieldalgo/wavebit/blob/main/studies/rcwa/nvf_fitting_grid/README.md).

## Refitting with the integral construction

The [completed refitting record](https://github.com/fieldalgo/wavebit/blob/main/studies/rcwa/nvf_fitting_grid/integral-fitting.json) uses solver `2056a39e`,
JAX 0.12.0.1, NumPy 2.5.4, SciPy 1.18.1 and batches of five scalar calls.
The run generated a new grid-128 target with this solver and held it fixed across
all four fits. The earlier solver's target is retained separately; do not mix the
change of normal-field construction with the effect of grid refinement.
The parameter tolerances, spectrum tolerance, noise sigma, TRF settings and
two-consecutive-pass rule are unchanged from the historical ladder above.

| Normal grid | Width (nm) | Angle (rad) | Center x (nm) | Thickness (nm) |
| --- | ---: | ---: | ---: | ---: |
| 128 | 420.000000 | 0.300000000 | 90.000000 | 300.000000 |
| 256 | 420.014209 | 0.300018938 | 89.989685 | 300.000675 |
| 512 | 420.005667 | 0.300005112 | 89.995820 | 299.999960 |
| 1024 | 420.005790 | 0.300005446 | 89.996013 | 299.999854 |

| Normal grid | Maximum parameter change / tolerance | Maximum same-point spectrum change / tolerance | Weighted residual norm |
| --- | ---: | ---: | ---: |
| 128 | — | — | 0.000 |
| 256 | 0.710 | 2.398 | 0.515 |
| 512 | 0.427 | 0.892 | 0.434 |
| 1024 | 0.026 | 0.264 | 0.409 |

All four optimizers succeeded. The 128→256 transition fails the spectrum check;
the next two transitions pass both checks without changing either tolerance.
At 512→1024, thickness changes by `0.000105642 nm`, against `0.004 nm` allowed,
and the maximum same-point reflectance change is `2.63504e-6`, against `1e-5`.
The target and its Jacobian remain bit-for-bit equal to the grid-128 fixed-point
record. The execution sources were also unchanged throughout the measurement.

The completed ladder returns `converged=true` for the declared parameter
and same-point spectrum criteria. A ratio must be at most one to pass. This
fit decision is separate from the fixed-point angle-Jacobian check above,
which still fails the `1e-3` grid-stability criterion. Noise-only standard
deviations in the record exclude discretization bias. This fixed-order,
same-family target does not establish independent optical accuracy.

## Finer grids for sensitivity stability

The derivative refinement uses solver `98353d8e`, the same N5 generating
parameters, all thirty observations, Fourier orders `(4, 4)` and CPU double
precision. Its scalar grid-128 values and all 120 Jacobian entries are bitwise
equal to the retained `2056a39e` record, and all four original Richardson checks
pass. This is a fixed-point sensitivity study; the fitted-parameter ladder above
remains its own measurement.
The [reviewed continuation record](https://github.com/fieldalgo/wavebit/blob/main/studies/rcwa/nvf_fitting_grid/jacobian-refinement.json)
preserves all values, derivatives, controls and execution sources.

| Grid change | Maximum spectrum difference | Width Jacobian | Angle Jacobian | Center-x Jacobian | Thickness Jacobian |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1024 → 2048 | 4.980e-7 | 7.594e-5 | 1.859e-3 | 1.493e-4 | 1.195e-6 |
| 2048 → 4096 | 1.996e-7 | 6.253e-5 | 5.903e-4 | 7.989e-5 | 4.967e-7 |
| 4096 → 6144 | 6.010e-8 | 2.109e-5 | 7.447e-5 | 1.933e-5 | 1.639e-7 |

The angle column still fails at 1024→2048. Every column and the spectrum pass
the next two transitions. The angle changes are 0.0590% and 0.00745%, against
0.1% allowed, satisfying the two-consecutive-refinement criterion on this ladder.
Grid 6144 was selected before execution for the memory budget and uses sequential
JVP; the last refinement ratio is 1.5. This is evidence on the tested ladder,
not an 8192 measurement or a bound on the remaining derivative error.

At grid 4096, the original Richardson steps give relative AD/FD errors
`(9.36e-9, 1.98e-6, 2.81e-7, 2.03e-11)` for width, angle, center x and thickness.
All four pass `1e-3`. A same-grid execution control also compares the sequential
angle JVP with the full forward Jacobian: all thirty values are bitwise equal
and the angle column's relative difference is `5.68e-14`, within the original
execution-control tolerances. Agreement on one grid and stability between grids
remain separate tests.

## Controls and scope

For the fixed nondispersive N5 case, the study's `--shared-matrices` execution
path builds the layer convolution matrices once within the differentiated
spectral evaluation. All geometry derivatives pass through that build; only
the optical solves are batched across wavelengths and polarizations. This reuse
depends on the declared constant scalar, nonmagnetic materials and is not a
general dispersive cache. `--sequential-jvp` computes the four columns in turn
to bound live tangent arrays, repeating primal work. The independent scalar
`Prepared.run` controls compare both paths at two wavelengths and polarizations,
including a change in all four parameters and a value-only evaluation. They use
unchanged tolerances: values `rtol=1e-10, atol=1e-12`, derivatives
`rtol=1e-8, atol=1e-11`.

`tests/rcwa/test_spectral_fit_grid_example.py` uses exact linear inverse problems
to check bias reduction and noise covariance, and negative controls where every
optimizer succeeds while grid convergence fails. It checks an accidental
plateau, unchanged parameters hiding forward drift, rank deficiency, nonfinite
values/Jacobians and exhausted optimizer/grid budgets. Reused-buffer controls
check that the observed target and accepted evaluations stay fixed after a
rejected trial; reporting does not repeat that accepted solve. These controls qualify
the stopping logic; the optical example exercises the actual public solver.
Additional analytic controls hold the optimum and spectrum fixed while changing
the model slope. They verify that the optional Jacobian criterion prevents a
false convergence report, compares at the same finer-grid fit and scales columns
separately. This criterion is a refinement test, not an AD/FD consistency test.

The default normal grid remains 1024 (or the Fourier bandwidth minimum if
larger). Its value alone supplies no error estimate. No solver tolerance or
reference value was changed to pass these checks. Independent Maxwell accuracy,
Fourier-order convergence and the ellipse's geometric approximation remain
separate validation questions.

## Spectral batch memory

The integral builder selects candidate interfaces using geometry alone. Material
contrast is applied when integrating those candidates. Previously contrast also
entered the candidate predicate, so a wavelength `vmap` turned the sparse/dense
`cond` into a `select` and evaluated the large dense branch even for sparse lanes.
For batches whose geometry really varies, dense integration now scans smaller
point groups, targeting the sparse branch's point-by-interface working size
(with a minimum of one point). Every point still sees all its intervals, including
facing corrections, before its nearest-distance scaling and ordered reductions.
No interface, observation, derivative direction or grid sample is dropped.

The [measurement record](https://github.com/fieldalgo/wavebit/blob/main/studies/rcwa/nvf_fitting_grid/batch-memory.json)
compares base `ed01b317` with the recorded integral-source hash on Linux CPU.
The N5 parameters, grid 128, orders (4, 4), double precision and host topology
checks are fixed. These batches use **all s polarization**, with 8 or 30 equally
spaced wavelengths from 800 to 1200 nm; they are not the 15s+15p fitting data.

| Lanes / JAX | Before: compiled temporaries | After: compiled temporaries | Before: peak RSS | After: peak RSS |
| --- | ---: | ---: | ---: | ---: |
| 8 / 0.11.0 | 4.29 GiB | 0.63 GiB | 6.75 GiB | 2.50 GiB |
| 30 / 0.12.0.1 | 14.75 GiB | 2.22 GiB | Memory-guard interruption | 4.82 GiB |

The 30-lane baseline was stopped by a 12 GiB shared-cgroup guard; this is not a
completed run or a measured 16 GiB OOM. The corrected run completed all 120
Jacobian entries. Their maximum absolute difference from independent scalar
calls on the baseline was `5.91e-12`, within `rtol=1e-8, atol=1e-11`; all scalar
values and validity checks passed. The 8-lane before/after Jacobian difference
was `1.32e-13`. Timings in the record shared CPU with some local checks and are
not an isolated speed benchmark. Two wavelengths with both s and p polarization
also agree in their batched Jacobians with baseline scalar calls (maximum
absolute difference `3.81e-14`); their scalar values and validity checks pass.

Regression tests retain conditional execution under material-only batching,
check a zero-contrast boundary against NumPy, and compare batched geometry's
forward/reverse derivatives with independent finite differences. A dense-fallback
control limits compiled temporary memory to twice its one-point execution;
the baseline exceeded that control by 10.68 times. These checks qualify execution,
not grid convergence. Total memory still grows with batch size, geometry,
derivative directions and optical matrix dimensions; this is not a universal
memory ceiling or a claim of the historical 2 GiB footprint on every platform.

## Grouped JIT execution and JAX versions

A separate CPU liveness failure occurs in JAX/jaxlib 0.11.0. On solver base
`91badd36`, six five-lane `vmap` calls concatenated inside one JIT can stop after
compilation, even though the geometry and materials are valid. This measurement
uses the N5 **15 s + 15 p** observations, grid 128, orders (4, 4), double precision,
and all four parameters. The value-only calls prepare concrete geometry at
ellipse centre x = 90.1 and 89.9 nm (the original `10h` central-difference pair);
the Jacobian calls use the preparation at the generating parameters with host
topology checks and return the values and validity flags alongside all 120 entries.

On Linux, the 0.11.0 joined value call compiled in 207.4 s, then stopped making
CPU progress. Native stacks showed all five worker threads waiting in
`ParallelBatchMap` / `BlockingCounter::Wait`: three in `lapack_zgesdd_ffi` (SVD),
two in `lapack_zgetrf_ffi` (LU). This is a thread-pool starvation deadlock, not a
physical rejection or the preceding memory regression. Separate five-lane calls
also stalled after four groups on that runtime; splitting calls is not a general
cure for the affected library.

The macOS 26.7.2 arm64 runner also stalled in all six 0.11.0 cases (joined and
separate centre± values and origin Jacobians). Its centre+ joined native sample
shows two SVD workers and one LU worker waiting in the same nested pool path.
The process guard interrupted these calls after 120 s without CPU progress;
none is counted as a completed spectrum.

The public JAX 0.11.1 wheels disable this nested LAPACK thread-pool path
([upstream change](https://github.com/jax-ml/jax/commit/859992407d38c09dfb0cb97d4da00e9fdd9e7830),
[release source](https://github.com/jax-ml/jax/blob/jax-v0.11.1/jaxlib/cpu/lapack_kernels.cc)).
The dependency floor and `Execution` admission check now require both JAX and
jaxlib >= 0.11.1, before device initialization or graph tracing. An older loaded
runtime returns `execution.jax_version` with upgrade/restart guidance. The solver's
linear algebra, derivatives, numerical tolerances and global thread settings are
unchanged.

Completed measurements use the same 30 observations and 120 Jacobian entries:

| Runtime / platform | Completed cases | Maximum joined/separate value difference | Maximum Jacobian difference |
|---|---|---|---|
| 0.11.1 / macOS arm64, CI threads | 6 / 6 | 4.33e-15 | 3.93e-14 |
| 0.12.0.1 / Linux x86_64, CI threads | 6 / 6 | 0 | 0 |
| 0.12.0.1 / macOS arm64, CI threads | 6 / 6 | 2.89e-15 | 5.52e-14 |
| 0.12.0.1 / macOS arm64, default XLA | 6 / 6 | 4.67e-15 | 5.52e-14 |

CI threads means BLAS/OpenMP thread counts of 1 and
`XLA_FLAGS=--xla_cpu_multi_thread_eigen=false`. The default-XLA measurement keeps
the BLAS/OpenMP limits and leaves XLA flags empty. Every completed case has
finite values/derivatives and true validity flags. Linux uses Python 3.12.14;
macOS uses 3.12.10. These are measurements on the stated runners, not the original
reporter's exact hardware.

The [retained record](https://github.com/fieldalgo/wavebit/blob/main/studies/rcwa/nvf_fitting_grid/grouped-execution.json)
contains the completed values/Jacobians, comparison tolerances and interrupted
runs. The [reproducer](https://github.com/fieldalgo/wavebit/blob/main/studies/rcwa/nvf_fitting_grid/grouped_execution.py)
measures one execution strategy per fresh process; choose `--execution joined`
or `separate`, `--mode value` with `--offset 0.1` or `-0.1`, or `--mode jacobian
--offset 0`. Each command requires a new `--output` path. Compare values with
`rtol=1e-10, atol=1e-12` and Jacobians with `rtol=1e-8, atol=1e-11`; every validity
flag must be true. Measuring an affected historical checkout requires an external
process timeout and must not be reported as completed when interrupted.

This checks CPU completion and agreement between execution strategies at the same
normal grid. It does not establish grid convergence, full numerical support on
all operating systems, or a guarantee that arbitrary future JAX versions cannot
have other concurrency defects.
