# Check the normal-field grid when fitting spectra

A successful optimizer finds a fit to the **chosen discretized model**. It does
not establish that changing `shape_nvf_grid_size` leaves the fitted dimensions
unchanged. A finite Jacobian, agreement with finite differences on one grid, and
small noise-only covariance also do not establish this.

The normal-field default is `max(1024, 4*half_width+1)`. This is a starting
resolution, not a tolerance guarantee for a spectrum, derivative or fitted
parameter. Choose tolerances for the quantities your application needs and
check refinement. Keep Fourier order fixed during this check; verify order and
curve approximation separately afterwards.

## Automatic refinement and refitting

The source-checkout example
[`spectral_fit_grid.py`](https://github.com/fieldalgo/wavebit/blob/main/examples/rcwa/spectral_fit_grid.py)
contains a reusable application function, `fit_grid_ladder`. It is example code,
not a new `rcwa.run` option or an installed fitting API. Run its complete small
NumPy/SciPy demonstration with:

```shell
python examples/rcwa/spectral_fit_grid.py
```

For your model, supply a factory which accepts a grid size and returns a callable
`model(parameters) -> (spectrum, Jacobian)`. The spectrum has shape `(m,)` and
the Jacobian `(m, n)`, with real finite entries. Validate every solve and handle
geometry preparation within that callable. The host owns the refinement and
optimizer loops; the individual value/Jacobian calculation may be compiled.

Pass these explicit controls to `fit_grid_ladder`:

| Argument | Meaning |
| --- | --- |
| `initial`, `data` | Initial parameter vector and fixed observed spectrum. |
| `grids` | At least three strictly increasing positive grid sizes. |
| `noise_sigma` | Known independent noise standard deviation per observation, scalar or vector. |
| `parameter_tolerance` | Maximum acceptable change per fitted parameter, in that parameter's units. |
| `spectrum_tolerance` | Maximum acceptable forward-model change per observation. |
| `max_nfev` | Optimizer evaluation budget at each grid. |
| `on_stage` | Optional callback receiving each completed stage, for progress records. |

At each grid the function fits the same data, starting from the previous fitted
parameters. It checks both the parameter change and the difference between the
old and new spectra **at the same finer-grid fitted parameters**. Comparing only
the two fitted residuals would let a parameter shift conceal the model change.
Every component must meet its tolerance on two consecutive refinements. A failed
coarse optimizer is retried on the next grid and cannot count toward a pass.
Invalid evaluations raise; rank-deficient fits cannot pass.
The last accepted value/Jacobian pair is retained for reporting, so rejecting a
final trial does not trigger another solve at the already evaluated optimum.
Observed data is copied on entry so a model's reused output buffer cannot move
the target during fitting.

The returned `GridFit` has `parameters`, `stages`, `converged` and `reason`.
Each stage records optimizer success, the noise-only standard deviations, the
noise-weighted residual norm and both maximum change/tolerance ratios. A ratio
of 1 is the limit. Exhausting the grid list returns `converged=False`; the example
raises on that outcome. Retain the report and extend the grid budget, rather than
presenting the last optimizer success as resolution convergence.

Two consecutive passes protect against one accidental plateau. They are evidence
on the tested ladder, **not a bound on the remaining error**, global-optimum
proof or physical accuracy guarantee. Check additional levels when the sequence
is irregular. Noise-only standard deviations use the singular-value decomposition
of the weighted Jacobian, assuming a locally identifiable least-squares model;
they do not include discretization bias or model mismatch. Check the weighted
residuals separately: grid convergence does not establish agreement with the
measurements within their noise.

## What to verify for a new problem

Choose dimensional tolerances before the sweep. For example, a 0.02 nm allowed
thickness change and a `1e-5` allowed reflectance change are separate requirements;
a small change in reflectance can still move a poorly identified dimension.
Check AD against finite differences with multiple step sizes at the chosen grid.
Use tighter optimizer tolerances than the requested grid stability and retain
failed fits. Then check Fourier order, polygonization and the experimental model.

The [measured example and four-parameter study](../../validation/rcwa/fitting-grid-convergence.md)
show how fitted dimensions, noise uncertainty and grid changes differ. The small
example uses central differences for its one parameter and a synthetic target;
the larger study uses forward-mode JAX differentiation. Neither target is an
independent Maxwell reference.
