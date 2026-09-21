# Sphere clipping across a JIT boundary

Written: 2026-09-21.

Targets: `wavebit/fd/_shapes/_exact.py`, `tests/test_fd_sphere_clip_jit.py`,
`tests/test_fd_shapes.py`.

Factoring squared radial distances keeps their endpoint signs stable across fused
computations. It prevents a guarded square root's inactive value from entering the
physical volume; the 32-origin, four-backend gate measures the remaining fill
differences below 2.30e-13 without changing the integration method.

## Reproducer and cause

Measured on 2026-09-21 on the Windows CPU backend, with float64 throughout:
Python 3.12.14, NumPy 2.5.1, JAX/jaxlib 0.11.0, and Torch 2.13.0+cpu.
The defect in the geometry branch at `5fd02a4` occurs at ordinary resolution:
`Sphere(0.3)`, voxel side `0.05`, and
`Box((0.2, 0.2, 0.9), angle=0.5)`. A 16-cubed grid starts at -0.4, with
centres at `-0.4 + (arange(16) + 0.5) * 0.05`. Sweep 32 origins from
`default_rng(73).uniform(-0.5, 0.5, (32, 3)) * 0.05`, replacing the first by zero.

The failing voxel at origin 26, index `(8, 7, 1)`, is centred at
`(0.00161135212954675, -0.02089849705058622, -0.3005275396021332)`.
The box contains this whole voxel, so the three CSG answers are known from the
sphere's own fill without approximating the intersection of two curved surfaces.

| Quantity | NumPy / eager JAX | Original jitted CSG |
|---|---:|---:|
| Child sphere fill | 0.460806344307509 | -200.940388564916 |
| Union with the box | 1 | -199.940388564916 |
| Intersection with the box | 0.460806344307509 | -200.940388564916 |

Origin 13, index `(11, 7, 2)`, independently catches the positive excursion:
Difference reads 142.652258838284 against 0.010556720698161. Compiling the sphere
alone or returning additional diagnostic intermediates can conceal the defect.
The test must compile the complete occupancy call as its actual caller does.

The first bad formula is the old line in `_wedge_integral`:

```python
rest = _root(k2 - t * t, xp)  # t = min(z, sqrt(k2))
```

At the upper endpoint this subtracts independently rounded squares. The radicand
is mathematically zero but its evaluated sign can change with contraction and
recomputation. `_root` must protect an inactive square root from an infinite
derivative, and previously evaluated

```python
live = v > 0
where(live, sqrt(where(live, v, 1)), 0)
```

XLA can place the protected square root and its outer selection in different
fused computations. Re-evaluating the unstable radicand there can give the two
selections different predicates, allowing the inactive value 1 into the result.
This is not a float32 path, an eight-corner rounding residual, or NaN propagation.
The following causal diagnostic changes only an algebraically unobservable value:

| Inactive radicand in `_root` | Child sphere fill in original jitted CSG |
|---:|---:|
| 0.25 | -203.436302 |
| 1 | -200.940389 |
| 4 | -186.097088 |
| 0, diagnostic only | 0.460806344308 |

Zero would bring back the non-finite reverse derivative that the guard prevents;
it is not the repair. Disabling fast math alone did not repair this example.
Evaluating the optimized HLO arithmetic with NumPy returned 0.460806344308;
diagnostic compilation with backend optimization level zero or fusion disabled
also returned that value. No XLA_FLAGS were set, and no compiler settings or
optimization barriers are needed by the repaired implementation.

## Exact algebraic repair

Use the same computed endpoint `k` for the integration limit and the radicand:

```python
k2 = (radius - a) * (radius + a)
k = _root(k2, xp)
t = minimum(z, k)
rest = _root((k - t) * (k + t), xp)
```

Since `t <= k`, the two factors are non-negative, and `k - t` is exactly zero
when the limit equals `k`. This removes the unstable zero comparison rather than
clamping the resulting volume or changing the integration method. The same
factorization is used in `_arc_integral` and the corner's radial and rectangle
transition heights. For the latter, `R^2 - a^2 - b^2 = (ka - b)(ka + b)` uses
the same `ka` that its antiderivatives see. All backends execute these same formulas.
The positive inactive guard remains, preserving finite automatic derivatives.
There is no new tolerance, floor, per-voxel exception, or backend-specific path.

This repair addresses the unstable radicands. The separate rounding budget of an
eight-corner cumulative volume remains approximately `eps * (R/h)^3`, as recorded
by the existing sphere-volume test. It is not a promise of uniform relative
accuracy for arbitrarily large `R/h`; the earlier probe at `R/h = 262144` was a
different cancellation regime and did not reproduce this ordinary-resolution bug.

## Measured gate and mutants

The new gate evaluates all 32 grid origins for Sphere, Union, Intersection, and
Difference under NumPy, eager JAX, jax.jit, and Torch: 512 grid evaluations, each
with 4096 voxels. Every result is checked for float64, finiteness, fill bounds,
backend agreement, normal-moment bounds and trace, and exact CSG containment laws.
The sphere additionally must reproduce `4*pi*R^3/3` at every origin on every backend.
The existing independent adaptive-disc quadrature and gradient tests remain active.

| Repaired formula, entire sweep | Measured worst | Test tolerance |
|---|---:|---:|
| Absolute fill difference from NumPy | 2.30e-13 | 1e-12 |
| Fill excursion outside [0, 1] | 2.78e-14 | 1e-12 |
| Absolute normal-moment difference | 2.71e-14 | 1e-13 |
| Relative total-sphere-volume error | 1.67e-15 | 1e-14 |

These tolerances were set after the sweep; their measured margins are also beside
the constants in the test. A second test varies the positive inactive radicand
over 0.25, 1, and 4 and compiles a fresh full CSG call for each one.

The source mutations were applied, graded by exit code, and restored:

1. Restore the original square-subtraction formulas in `_exact.py`, then run
   `python -m pytest tests/test_fd_sphere_clip_jit.py -k "jax.jit or inactive_root" -q`.
   Both selected tests failed normally: exit 1, 2 failed, 3 deselected. The sentinel
   assertion reported -203.43630164659726 against 0.46080634430750905.
2. Multiply the sphere's returned volume by 0.99, then run
   `python -m pytest tests/test_fd_sphere_clip_jit.py -k across_grid_origins -q`.
   All four backend cases failed at the analytic-volume assertion: exit 1, 4 failed,
   1 deselected. Bounds and agreement with the equally mutated NumPy reference still
   passed. Agreement with a shared implementation alone is not an oracle.

An additional incorrect-antiderivative mutant (`z**3 / 2` for `z**3 / 3`) also
failed all four backend cases at the fill bounds. Restoring only the wedge's
unfactored `k2 - t*t` after factoring the other transition heights did not reproduce
the original defect on these samples; the old formula was therefore restored as a
whole for the regression mutant. Conversely, factoring just the wedge endpoint in
the original formula was already sufficient to repair the isolated failing voxel.

The mutations are restored before the full suite and frozen-reference checks.
