# Pole arithmetic with mixed concrete and differentiated parameters

This regression record concerns numerical admission, not optical convergence or support
for every extreme material parameter. The public problem is a 4-by-4-by-4 periodic grid,
a cell-snapped Region, one Lorentz or Drude pole, and two time steps.

A finite SI frequency is chosen so that its normalized value is `1e200`. A Lorentz damping,
Lorentz strength, or Drude damping remains differentiated. Before the repair, skipping the
entire pole's eager check when any parameter was traced allowed a Python `OverflowError`
from squaring the concrete frequency to escape. The 30 combinations of these three cases,
JAX `jit` or Torch reverse-mode carriers, and `check`, `prepare`, `run`, `Prepared.validate`
or `Prepared.run` all reproduced it.

The selected bulk recurrence is now evaluated inside the arithmetic exception boundary
also for mixed parameters. Concrete host overflow or division by zero receives the existing
`PoleArithmeticRefused` diagnostic. Finite-value judgment of a traced recurrence remains
with the existing runtime guard. Specialized spatial-film recurrences keep their own
admission rules. Six ordinary-size controls check the recurrence drive and its analytic
parameter derivative; no field convergence tolerance is introduced. Two additional public
`Prepared.run` controls carry the Drude frequency itself: normalized frequency `2` retains
a finite nonzero field and valid status, while `1e200` produces an invalid runtime status
and an all-NaN field on JAX and Torch without an eager planning exception.

A separate baseline failure on ARM macOS showed that a direct integer zero test of a
bitcast strength could change its result when combined with the compiled drive bound.
The repaired check folds the magnitude's upper and lower integer words before the zero
comparison. Tests distinguish positive and negative zero from the smallest positive
float64, `1e-320`, and half the smallest normal float64. Each nonzero subnormal remains
an active, unrepresentable drive and must produce runtime invalidity rather than zero drive.
No XLA flags, numerical tolerances, golden arrays or skip conditions were changed.

## Focused verification

On macOS ARM64, Python 3.12.9, NumPy 2.5.1, SciPy 1.18.0, JAX 0.11.0 and Torch 2.14.1,
the four files below passed 123 cases after both repairs. This focused run is not a claim
that the complete suite or another operating system passed.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python -m pytest -q tests/fdtd/test_mixed_pole_arithmetic.py   tests/fdtd/test_selected_pole_arithmetic.py   tests/fdtd/test_material_finite_guard.py tests/fdtd/test_large_damping.py
```

Actual product mutations were tested in fresh processes and restored byte for byte:

| Mutation | Expected detection |
|---|---|
| Restore the old whole-pole trace skip | All 30 public refusal cases fail assertions |
| Double the ordinary recurrence drive | All six analytic value/tangent controls fail assertions |
| Bypass the live drive guard | All six subnormal-strength cases fail assertions |
| Judge traced coefficient finiteness eagerly | Both public live-frequency cases fail assertions |

The small case's independent derivative is obtained from
`drive = strength * dt**2 / (1 + gamma * dt / 2)`; the Drude strength here is
`omega_p**2`. It does not use the implementation's derivative as the expected value.
