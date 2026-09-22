# Smoothed occupancy and the tilted-interface measurement

Written: 2026-09-22.

Targets: `wavebit/fd/_lower/occupancy.py`, `wavebit/fd/_lower/coefficients.py`,
`wavebit/fd/_lower/checks.py`, `wavebit/fd/geometry.py`,
`studies/fdtd_subpixel_tilted_tensor.py`, `tests/test_fd_occupancy.py`,
`tests/test_fd_occupancy_integration.py`, `tests/test_fd_subpixel_study.py`.

`fd.Sphere` now supplies fractional material coefficients at each field component's
own location. Unsupported mixtures are refused during planning. Approximate CSG
volumes produce one measured scene warning before compiled lowering. The diagonal
mixing formula remains one part of the inverse material operator; the later
[constitutive implementation](fd-mie-validation.md#production-constitutive-coupling-and-energy)
adds adjoint cross terms and compact D/B memory. The tilted-interface study
does not establish second-order sphere scattering or settle every possible
off-diagonal scheme.

## The coefficient rule

For a fraction `f` of material `a` in material `b`, and the component's squared-normal
moment `w`, the inverse effective permittivity is

```text
1/eps_c = w * (f/eps_a + (1-f)/eps_b)
          + (1-w) / (f*eps_a + (1-f)*eps_b).
```

The normal component is harmonic and the tangential component arithmetic. These are
the diagonal entries of the effective inverse tensor described in
[Farjadpour et al. (2006)](https://doi.org/10.1364/OL.31.002972) and the
[Meep documentation](https://meep.readthedocs.io/en/latest/Subpixel_Smoothing/).
Initially anisotropic phases require the more general construction in
[Kottke et al. (2008)](https://doi.org/10.1103/PhysRevE.77.036611).
Only documentation and papers were consulted; no Meep implementation was read.

`occupancy.py` builds coordinates from each component's Yee offset. Electric
permittivity uses E points; magnetic permeability uses H points. The two are half a
cell apart. The mixing rule receives the fill and the corresponding normal-moment
component at that same point. Non-magnetic scenes avoid unused H evaluations.
Cell-snapped Slab/Box painters keep their old `where` operation, with no reciprocal
round trip. The geometry engine's JIT endpoint repair remains in place; see
[the clipping analysis](fd-sphere-clip-jit.md).

The planar analytic-fill, transfer-matrix, translation and derivative tests remain
the independent gates. Differentiation uses a plan built from concrete geometry,
then differentiates only numerical lowering. Public radius/centre tracing awaits a
declared support range. Existing shape subclasses must answer `body()` explicitly;
`None` selects snapped cells. `Sphere.min_extent(grid)` follows the current protocol,
and non-finite radii and coordinates are refused by name.

## Bounds are reported while planning

The geometry collector is `Approximations.record(node, counted, widest)`. Numerical
lowering does not convert its arrays to Python counts: that would run only while
tracing JIT and disappear on cached executions. Planning and `fd.check` evaluate the
static geometry on NumPy and report the same note, including when a report receives
already-built params.

For two spheres of radius 0.2 centred at x = -0.07 and +0.07, with `dx=0.05` in a
1.2-cubed domain, each of Union, Intersection and Difference reports **284 component
voxel evaluations**, widest bracket **0.05377 of a voxel**. Counts sum over evaluated
components and nodes; they are not counts of unique spatial cells. Union and Difference
take upper volume bounds; Intersection takes a lower bound. A volume bound does not
bound the electromagnetic error.

| Complete coefficient lowering, eps = 3.7 | Worst measured relative difference from NumPy |
| --- | ---: |
| JAX eager | 6.16e-14 |
| JAX JIT | 4.66e-14 |
| Torch | 3.53e-14 |

The regression budget is `1e-12`, set after this measurement (16 times the worst).
The test checks the actual CSG producer, first and cached JIT calls, the named node,
the width, and a report with prebuilt params. The synthetic reporting body separately
checks aggregation: 21 records become 42 for two bodies while the width remains 0.0625.

## The preparation-time limits

The C2 implementation refuses conductive/dispersive or anisotropic media touching
a smoothed support, wrapped or invariant axes a sphere reaches, domain-face losses,
and two separately painted smoothed surfaces in one voxel. These are deliberately
conservative limits until the respective geometry/material rules exist. Touching
surfaces inside one composed body use the geometry kernel and its bound report.
Conductivity and poles are not silently left unsmoothed in an accepted mixture.

The time-step bound also needs the two component locations. Air and a material with
`eps=4, mu=0.25` each have index 1. Before the correction, a radius-0.25 sphere at
`dx=0.05`, Courant 0.5, reported a stability margin of 2. The sampled arrays contained
`min sqrt(eps*mu) = 0.562900`. A safe global bound uses independent minima:

```text
n_bound = sqrt(min(eps) * min(mu)) = sqrt(1 * 0.25) = 0.5.
```

Every positive two-phase mix lies between its material endpoints. Using independent
extrema therefore bounds both staggered coefficient families without assuming that
their minima belong to the same phase or position. It is conservative, not the exact
instability threshold of this finite sphere. Planning refuses a time step at or above
that bound, names `Run(courant=...)`, and never changes `dt`. A tighter scene-specific
bound remains separate work. Scenes without smoothed bodies retain their previous bound.

## Correcting the S1 instrument

The corrected [study](../studies/fdtd_subpixel_tilted_tensor.py) measures a half-filled
unit-period lamellar medium, `k_t p=0.6`, contrasts 1.2/2.25/12, and rational tilts
0/14.036/26.565/45 degrees. The 14.036-degree row is `tan(theta)=1/4`, not 15 degrees.
It uses 16/32/48 nominal cells per period and fits against the actual rounded grid.

Three details change what the old instrument could establish:

1. `sigma=0, which='LM'` finds modes near zero, not the most negative eigenvalue.
   On `diag(-100, 1, ..., 100)` the old minimum/maximum ratio was -0.01 instead of -1.
   The study now uses algebraic smallest/largest queries for Hermitian operators.
2. The naive cross term is non-Hermitian. `eigsh` returns real parts for complex
   input, so the old call could conceal a complex mode. The general `eigs` call keeps
   complex eigenvalues, and the study reports their equation residuals. The solver
   distinction is documented by [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html).
3. Fill coordinates are folded before clipping, and image coverage follows the voxel
   extent. No fixed five-image window or final fill clamp remains. Translating 200
   points by up to 100 periods changed fill by at most 2.57e-13.

A homogeneous anisotropic control is a dispersion measurement in a different medium,
not a rigorous lower bound on the layered medium's error. The study reports both;
it neither subtracts one from the other nor uses a chosen multiplier to erase rows.

## S1 results and what they decide

Measured on Windows, AMD64 Family 25 Model 97 Stepping 2, Python 3.12.14,
NumPy 2.5.1 and SciPy 1.18.0, with two numerical-library threads.
The [recorded run](../studies/results/fdtd_subpixel_tilted_tensor.log) contains 108 series,
324 solves, in 285.7 seconds including overlapping occupancy probes; this elapsed time
is not an isolated performance benchmark. Homogeneous-control orders are 1.999966 to 1.999992;
the arithmetic TM fill control is 1.99900 to 2.06171. The largest selected-eigenpair
relative residual is 5.42e-12. The self-adjoint operator's non-Hermiticity is at most
1.52e-17, versus 0.02485 for the naive writing.

At contrast 12, signed finest-grid error and the three-point fit are:

| Tilt | Diagonal error | Diagonal order / scatter | Self-adjoint tensor error | Tensor order / scatter |
| --- | ---: | ---: | ---: | ---: |
| 14.036 degrees | -1.9374e-3 | 0.999 / 0.015% | -1.8726e-4 | 0.929 / 0.60% |
| 26.565 degrees | -4.3386e-3 | 1.012 / 0.15% | -2.1383e-4 | 0.387 / 107% |
| 45 degrees | -7.0242e-3 | 0.864 / 7.7% | -2.5129e-3 | -2.452 / 196% |

Scatter is `exp(max absolute log-fit residual) - 1`. The last two tensor rows are
non-monotone. Their fitted powers are descriptive numbers, not convergence claims.
The algebraic spectral minimum is positive at every sampled self-adjoint grid;
the smallest minimum/maximum ratio is 1.0303e-5. This tests these matrices only.

The naive 26.565-degree, contrast-12, 48-cell mode has relative imaginary frequency
**2.188e-5**, which the old solver discarded. Reporting its real-frequency accuracy
alone would hide that the comparator does not conserve the Hermitian structure.
At low contrast the tensor's total errors can be close to the homogeneous control;
these samples do not isolate the interface contribution or its asymptotic order.

The diagonal rule is an explicit approximation with a measured tilted-interface
cost. The sampled tensor alternatives do not establish a robust general improvement
in order, and they do not prove that no better off-diagonal discretisation exists.
The 3-D Mie test and the 2-D cylinder study remain necessary before a sphere-scattering
convergence claim. The separate extreme `R/dx` cumulative-volume precision limitation
described in the clipping analysis also remains.

## Regression mutants

Source mutations are run individually and restored before ordinary validation.
Each selected mutant must exit 1 with test failures; collection errors and
internal pytest errors do not count as a killed mutant. The tests cover missing plan
notes, the old phase-index stability bound, non-finite geometry, the old extent
signature, near-zero spectral queries, discarded imaginary eigenvalues, fixed image
coverage, concealed fit scatter, shared component coordinates, swapped mixing rules,
and rounded fills. Tolerances carry the measured values beside their definitions.

The 2026-09-22 run killed all 11 source mutants below with **exit 1**, no collection
or internal errors. All mutations were restored. The selected tests are those named
by `-k` in the corresponding occupancy or study test file.

| Mutation | Test failures |
| --- | ---: |
| Remove the plan's geometry note | 14 (all 12 CSG/backend rows and both traced-plan doors) |
| Use only declared phase indices for stability | 1 |
| Accept non-finite sphere geometry | 6 |
| Restore `min_extent(self)` without the grid argument | 1 |
| Query spectrum near zero and use largest magnitude as denominator | 1 |
| Return only the real part of an eigenvalue | 1 |
| Restore the fixed image window without folding coordinates | 1 |
| Report zero fit scatter | 1 |
| Sample every component at the cell centre | 2 |
| Swap normal and tangential mixing | 2 |
| Round fills to zero or one | 3 |

## Integrated validation

The complete 2026-09-22 run used three pytest workers and two threads per numerical
library, with `PYTHONUTF8=1` and without setting `XLA_FLAGS`.

| Gate | Result | Exit code |
| --- | --- | ---: |
| All `test_fd_*.py` and `test_docs_contract.py`, excluding slow | 1801 passed, 6 skipped | 0 |
| The same files with `--run-slow -m slow` | 65 passed | 0 |
| Frozen stage 0 / 1 / 2 / 3 arrays | 143 / 288 / 140 / 570 bit-identical | 0 each |

The six skips are the existing NumPy automatic-differentiation comparisons: NumPy
supplies their reference values. Collection, XML testcase counts and unique test names
agree at 1807 fast and 65 slow cases. No `INTERNALERROR` occurred. All four reference
archives retain their original SHA-256 hashes; no rebaseline was performed.
