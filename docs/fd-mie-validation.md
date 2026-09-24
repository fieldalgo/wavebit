# Mie reference and the curved-interface accuracy gate

Written: 2026-09-24

Targets: `tests/references/mie.py`, `studies/fdtd_subpixel_mie_cylinder.py`,
`tests/test_fd_mie_reference.py`, `tests/test_fd_mie_study.py`,
`tests/test_fd_monitor_structure.py`, `wavebit/fd/_lower/monitors.py`,
`wavebit/fd/monitors.py`, `wavebit/fd/diagnostics.py`, `wavebit/fd/_lower/checks.py`,
`tests/test_fd_occupancy_integration.py`, `tests/test_fd_decay.py`,
`studies/fdtd_subpixel_mie_sphere.py`, `tests/test_fd_mie_external.py`,
`tests/references/interface/measure_fd_stage4_meep.py`,
`tests/references/interface/measure_fd_stage4_fdtdx.py`,
`studies/fdtd_subpixel_constitutive.py`, `tests/test_fd_constitutive_study.py`,
`wavebit/fd/_kernel/constitutive.py`, `wavebit/fd/_lower/constitutive.py`,
`wavebit/fd/_kernel/observe.py`, `tests/test_fd_constitutive.py`,
`studies/fdtd_subpixel_coupled_cylinder.py`,
`studies/fdtd_subpixel_interface_consistency.py`,
`studies/fdtd_subpixel_triplet_cylinder.py`,
`studies/fdtd_subpixel_triplet_sphere.py`, `tests/test_fd_triplet_sphere.py`,
`studies/fdtd_subpixel_circle_moments.py`,
`tests/test_fd_triplet_research.py`, `tests/test_fd_mie_physics.py`,
`tests/references/mie_fields.py`, `studies/fdtd_subpixel_mie_transform.py`,
`tests/test_fd_mie_transform.py`.

## TL;DR

Latest radiation result: the separately derived lattice map,
`FarField(propagation='yee')`, reduces the weak-sphere optical residuals to
**+.0264028%/-.0433575%/-.00357156%** on the unchanged 20/28/40-cell
records. A fresh public 20-cell run gives **+.0267422%**. Independent
reciprocity, Fourier Green and source-current tests validate the map.
The historical continuum results below remain intact; the production
cylinder accuracy failure still blocks C3. See
[the derivation and current validation](fd-lattice-radiation.md).

The analytic reference and the closed-contour instrument have independent checks.
The earlier diagonal material rule gives near-second-order fits in the cylinder's axial-electric warm-up,
but the high-contrast axial-magnetic case does **not** establish second-order
convergence. C3 is not complete. Exact geometric volume alone does not establish
electromagnetic accuracy; a successful regression suite must not be reported as
a successful sphere convergence gate.

The cylinder study preceded the three-dimensional sphere measurements. The
public glass sphere with that diagonal rule misses the required order: three placements at
20/28/40 cells per wavelength give **1.714**, below the required 1.8, with 1.38%
fit scatter. A measured failure is retained with its placement spread and fit
residual. No tolerance is chosen to relabel it as a second-order result.

A controlled constitutive comparator now identifies the discarded cross terms
as the cause of the glass fixture's lost order. Exact surface moments alone
leave the order near 1.72. Restoring shared-node cross terms gives **2.02267**
over the same three placements, with **0.9744%** scatter. This comparator is
research code. The production engine now includes the same adjoint cross
coupling and consistent compact D/B memory. Its validation below is separate
from this comparator; C3 remains incomplete until all its gates pass.

The production sphere sweeps now have all three resolutions and three
placements at epsilon 1.1, 2.25 and 12. Their fitted orders are **2.07490,
2.02267 and 1.82685**, respectively. The high-index sweep finishes with a
6.18% mean scattering error at 80 cells per wavelength; a fitted order is
not a claim of small absolute error. The coupled cylinder revalidation now
fails: all eight placements at 40/56/80 cells give mean/RMS orders
**1.60337/1.58655**, below 1.8. The seven new public physics cases pass and
eight actual source mutants are caught; that does not waive the cylinder gate.
The weak-scattering optical
theorem limitation below is retained.

A separate symmetric edge/face research comparator now meets the finite
cylinder comparison: all eight placements at 40/56/80 cells give mean/RMS
orders **1.82481/1.82899**, with **4.5563%/4.5894%** fit scatter. Its finest
mean error is **1.00864%**. This is a different research operator; the
production engine remains unchanged. The result does not combine with the
earlier production sphere measurements to certify C3. Its separate 3D
research extension now gives glass/weak-sphere mean orders **2.03778/2.04781**
over the original three resolutions and placements, with independent
geometry, backend and local radius-derivative checks. The weak-scattering
optical residual remains **12.90%** at 40 cells. High-index convergence with
the new operator, full physical derivatives and production integration
remain unvalidated. The candidate also misses two existing production
physical controls: glass peak-normalized angular error **1.164% > 1%**, and
small high-index sphere scattering error **3.771% > 3%**. It is not ready
to replace the production operator; neither limit is relaxed.

An exact-near-field diagnostic now separates continuum sampling errors from
the errors involving computed fields. At the standard radiation surface,
weak-sphere optical closure is **1.8475%/0.9370%/0.4577%** at 20/28/40 cells
when exact Mie fields replace the solver output, versus **52.0475%/26.3798%/
12.9138%** in production. Surface dependence remains even with exact data.
This rules out continuum sampling alone as an explanation of the standard
fixture's large residual; it does not distinguish grid dispersion, source
normalization, material error, or a mismatch between discrete fields and a
continuum radiation formula. No production correction follows from this test.

## Independent analytic reference

The reference imports NumPy and SciPy, and no FDTD code. It uses the spherical
coefficients and angular recurrence in Bohren and Huffman (1983), chapters 4 and 8,
with `exp(-i omega t)`, positive imaginary refractive index for loss, and outgoing
`j_n + i y_n`. The internal logarithmic derivative is computed by downward
recurrence; the external real-argument Bessel functions come from SciPy.

The complex coefficient pins are the numerical examples in
[Scott Prahl's algorithm documentation](https://miepython.readthedocs.io/en/3.0.5/07_algorithm.html).
Only documented equations and examples were consulted. No Meep source was read.

| Check | Measurement |
| --- | ---: |
| Three published complex `a1, b1` pairs | maximum absolute gap 7.58e-13 |
| Downward recurrence versus direct interior Bessel derivatives, 72 cases | maximum finite coefficient gap 1.42e-14 |
| Rayleigh scattering, `m=1.5`, `x=0.001` | relative finite-size correction 7.06e-8 |
| Lossless glass, `m=1.5`, `x=pi/2` | `Qsca = 0.863559745639449` |
| Absorbing sphere, `m=1.5+0.1i`, `x=10` | `Qext = 2.45979053`, `Qsca = 1.23514421` |
| Extinction paradox, `m=1.5`, `x=5000` | `Qext = 2.008649849173719`, approaching 2 |
| Extend the series by 16 terms | largest efficiency change 6.69e-13 |
| Angular integral, 64 Gauss-Legendre points | relative energy closure 1.34e-15 |
| Cylinder radial wave equation integrated independently | largest selected coefficient gap 5.76e-12 |

Padding the downward recursion by 64 terms at `x=5000` misses extinction by
2.39e-8. Padding by 128, 256 and 512 agrees to printed double precision. The
implementation uses 256 and tests against 512. Absorption is the actual difference
`Qext-Qsca`; it is never set to zero by a lossless branch or positivity clamp.
The Rayleigh check also pins complex phase and both polarizations. The angular
integral and optical theorem independently check the amplitude normalization.

For the infinite cylinder, the reference matches the interior radial solution to
outgoing Hankel functions. TM means axial electric field, TE axial magnetic field.
The boundary derivative is weighted by `1/eps` only for TE. An independent radial
ODE integration checks that distinction without using the interior Bessel formula.
At radius 0.15 and wavelength 1, doubling the truncation from 32 to 64 orders gives
the same printed cross-sections:

| Polarization | epsilon | Cross-section |
| --- | ---: | ---: |
| TM | 1.2 | 0.00845929167318637 |
| TM | 12 | 1.016762944607247 |
| TE | 1.2 | 0.0035611157245021728 |
| TE | 12 | 0.909413120174101 |

## What the cylinder experiment measures

The study uses the actual component positions, geometry occupancy and coefficient
mixing functions. Until `fd.Cylinder` is public, it substitutes those coefficients
after preparing an empty scene on a one-cell invariant x axis. This is a measuring
instrument, not validation of public scene preparation. In particular, substituting
the material bypasses the declared-material resolution refusal; low resolutions
of epsilon 12 in this study must not be presented as accepted public scenes.

The y axis is wide and periodic. A finite time gate ends before neighbouring
cylinders can contribute. The z ends have 12-cell PML layers. The source is a plane
wave; a paired empty run is subtracted at the complex-field level before measuring
outward scattered power. Controls vary gate, period, source position and distance
to the absorbing layer. The contour half-width is 0.55 wavelengths, and additional
contours move it by four cells in each direction.

The endpoint weights are essential: node values use a trapezoid and half-cell
values use midpoint weights. A manufactured radial field with power density
`1/(2r)` must integrate to pi on every surrounding contour. At 20/40/80 cells per
wavelength the measured orders are 1.99845 (TM) and 1.99904 (TE), with finest relative
errors 6.060e-4 and 6.061e-4. Deliberately opening the endpoints instead gives
orders 0.762 and 0.722. This independently tests the measurement rather than using
Mie agreement to assume that the measuring contour works.

Every resolution retains eight fixed fractional-cell placements. A log fit reports
both signed mean and RMS error, along with `exp(max absolute log residual)-1`.
Opposite placement errors can cancel in the mean; a manufactured `+h,-h,+h,-h`
test makes sure RMS still reports order one. All 70 four-placement subsets are
reported as a sensitivity check. Their range is not a confidence interval for the
regression and cannot be used to exclude an asymptotic order.

## Measured cylinder results

The [recorded run](../studies/results/fdtd_subpixel_mie_cylinder.log) retains all
128 JAX cases (two polarizations, two contrasts, four resolutions, eight placements),
48 repeated NumPy cases, a Torch repeat and the independent controls. It was measured on Windows,
AMD64 Family 25 Model 97 Stepping 2, Python 3.12.14, NumPy 2.5.1 and SciPy 1.18.0,
with two numerical-library threads. The JAX sweep took 1386.5 seconds while other
checks overlapped it; this is not an isolated speed benchmark. The 48 matched NumPy
and JAX cross-sections differ by at most 8.89e-16 absolute.
The 20-cell, origin-placement, epsilon-12 TE cross-section on Torch differs from
NumPy by 2.23e-16 and is identical to the printed JAX value, 0.9159581743334625.

Signed mean relative errors at 20/28/40/56 vacuum cells per wavelength:

| Case | 20 | 28 | 40 | 56 | Mean fit: order / scatter | RMS fit: order / scatter |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TM, epsilon 1.2 | +0.9091% | +0.4119% | +0.2096% | +0.1070% | 2.058 / 6.09% | 2.097 / 4.37% |
| TM, epsilon 12 | -6.3039% | -2.9228% | -1.3911% | -0.7079% | 2.119 / 3.04% | 2.124 / 2.35% |
| TE, epsilon 1.2 | +1.3166% | +0.6585% | +0.3530% | +0.1988% | 1.827 / 4.46% | 1.838 / 4.30% |
| TE, epsilon 12 | -1.2080% | -3.3102% | -3.5780% | -3.2730% | -0.888 / 48.25% | -0.555 / 32.37% |

The last row is non-monotone. Its negative fitted powers are descriptive failed
fits, not an assertion that the continuum limit diverges. The four-placement
subset orders range from 1.926 to 2.170, 2.042 to 2.192, 1.704 to 1.938, and
-2.942 to -0.279, respectively. The low-contrast TE result is also placement
sensitive; a single selected placement would overstate what has been established.

The high-contrast TE controls at 40 cells isolate the approximately 2.90% error
at the origin placement. Values below are relative changes in cross-section from
the baseline, not absolute errors against Mie:

| Control | Change |
| --- | ---: |
| Gate and transverse period multiplied by 1.4 | +7.03e-7 |
| Transverse period alone multiplied by 1.4 | +1.76e-14 |
| Gate and transverse period multiplied by 0.55 | -2.17e-6 |
| Source moved by +0.2 wavelength | +3.75e-7 |
| Absorbing-layer inner face moved out by 1 wavelength | +1.85e-6 |
| Measuring contour moved by four cells, baseline | at most 1.14e-6 |

All long-gate controls end below 7.64e-10 monitor-energy decay; the shortened
gate ends at 2.75e-7. The largest contour shift in the shortened control is
6.68e-6. These measured sensitivities are far below the scattering error.
The final study module reproduces the original 40-cell baseline cross-section
`0.8830476795866435` exactly after the probe was refactored into the committed
instrument. The measured issue is not a NumPy-versus-JAX discrepancy or explained
by these source, periodic-neighbour, time-tail, PML-distance or contour controls.

The shortened schedule was also measured at the origin on 56/80/112 cells. Relative
errors are -3.0891%, -2.8494% and -2.2148%, giving fitted order 0.477 and scatter
6.21%. The 56-cell short-versus-long schedule changes the cross-section by 7.57e-6
relative; the fine-grid contour sensitivity is below 3.91e-6. These are single
placement measurements and do not replace the eight-placement sweep. They show
the finest sampled case still far from the requested near-second-order behavior;
they do not determine the eventual asymptotic order.

The data reject a general second-order curved-scattering claim for the current
diagonal approximation. They do not prove that every possible tensor
discretisation fails. The corrected [tilted-interface experiment](fd-subpixel-occupancy.md)
already makes that distinction. Choosing and validating a different interface
update is unresolved; changing an empirical coefficient to fit one cylinder would
not resolve it. The full 3-D sphere gate and subsequent stages remain incomplete.

Reproduce the main sweep and controls with the configured Python environment:

```text
python studies/fdtd_subpixel_mie_cylinder.py cylinder-sweep.json
python studies/fdtd_subpixel_mie_cylinder.py cylinder-controls.json --controls --cells 40 --polarizations TE --eps 12
python studies/fdtd_subpixel_mie_cylinder.py cylinder-fine.json --cells 56,80,112 --polarizations TE --eps 12 --placements 1 --period-scale .55 --gate-scale .55
```

The output path is chosen by the caller. Use two threads for OMP, OpenBLAS and
MKL, set `PYTHONUTF8=1`, and do not set `XLA_FLAGS`.

## Closed power surfaces and material interiors

At plan construction, every closed `Flux` box must enclose or exclude the entire
support of each structure with one cell of clearance. The margin follows the H
half-plane stencil, not a fitted numerical tolerance. A cut or touching support
is refused with the monitor name, structure index, shape, cell ranges and remedy.
For a curved body this support check is conservative and the refusal says so.

`Field(box=...)` samples a volume, including useful interior electric and magnetic
fields. It is allowed inside a material. It is not a closed power surface.
`FarField` already requires every structure strictly inside its surface; its
existing refusal is separately pinned. None of these checks depends on per-cell
exceptions or on whether numerical lowering is compiled.

## Source mutation checks

Each mutation below was applied to actual source and restored afterward. Every
selected run exited 1 with test failures, without collection errors or an
internal pytest error. A surviving test within a selected group is not hidden;
the table reports the number that failed.

| Source mutation | Failed tests |
| --- | ---: |
| Incoming instead of outgoing spherical Hankel function | 4 |
| Remove size normalization from scattering | 5 |
| Start downward recursion with only eight extra terms | 1 |
| Discard absorption | 1 |
| Break angular recurrence | 2 |
| Reverse cylinder scattering sign | 8 |
| Swap cylinder polarization derivative weights | 4 |
| Remove the closed-Flux structure check | 7 |
| Refuse wholly enclosed or excluded structures | 2 |
| Treat a Field volume as a closed power surface | 1 |
| Remove FarField containment | 1 |
| Replace trapezoid endpoint weights by full weights | 3 |
| Omit the complex conjugate in power | 6 |
| Square placement errors only after averaging | 1 |
| Remove explicit host conversion before the NumPy reduction | 2 |
| Remove the static curved-accuracy warning | 4 |
| Emit the curved-accuracy warning for a cell-snapped Box | 1 |
| Judge the internal-layer echo as ended with the source envelope | 1 |

`SubpixelAccuracyWarning` is emitted once during planning whenever a smoothed
structure is present, and appears in `check` even when parameters are already
built. It quotes the measured 3.09% origin-placement cylinder error, explicitly
states that this is not an error bound for the caller's scene, and says the Mie
gate is incomplete. Lowering on NumPy, JAX, cached JAX JIT and Torch emits no
warning. A cell-snapped Box does not inherit this warning.

The manufactured contour runs on NumPy, JAX and Torch. An end-to-end Torch probe
initially raised `TypeError` because NumPy's reduction dispatched its `axis/out`
keywords to a Torch tensor. The measuring contour now converts only its sampled
vectors to host NumPy arrays before reducing them. The solver is unchanged; the
removed-conversion source mutant fails both Torch polarization tests.

The pre-existing internal-layer echo test used a closed Flux box intersected by
its transverse slab, although it only asserted source end times and energy-growth
diagnostics. That fixture now uses a Field volume when the internal layer exists.
Its physical assertions are retained. The old fixture was refused by the new
whole-structure door; the replacement passes, and an actual source edit that ends
the source at its waveform envelope still fails this test.

## Integrated validation of the C3 preparation

The final runs use three pytest workers, two threads per numerical library and
`PYTHONUTF8=1`, without setting `XLA_FLAGS`. Exit codes, collection counts and XML
test records were checked independently of the printed progress bar.

| Gate | Result | Exit code |
| --- | --- | ---: |
| Fast suite, including documentation contract | 1,845 passed, 6 intentional skips; 1,851 collected and reported | 0 |
| Slow suite | 65 passed; 65 collected and reported | 0 |
| Stage 0 reference | 143 of 143 arrays reproduce bit for bit | 0 |
| Stage 1 reference | 288 of 288 arrays reproduce bit for bit | 0 |
| Stage 2 reference | 140 of 140 arrays reproduce bit for bit | 0 |
| Stage 3 reference | 570 of 570 arrays reproduce bit for bit | 0 |
| Source mutants | 18 of 18 detected by their selected tests | 1 each, as required |

The six skips are the existing NumPy differentiation cases: NumPy supplies the
finite-difference reference for the other backends. The full collection is 1,916
tests, exactly 1,851 fast plus 65 slow. Final suite logs contain no `INTERNALERROR`;
XML reports zero failures and zero errors. The fast run took 917.25 seconds and
the slow run 353.38 seconds. All four archive hashes were checked before and
after their reproduction commands; no reference was rewritten.

An initial attempt exited 3 during pytest startup because its default temporary
directory was inaccessible; a fresh external base directory resolved that setup
problem. The first complete fast run exited 1 on the old internal-layer Flux
fixture described above. Neither attempt is counted as a pass. After correcting
the fixture and adding the Torch contour regression, the complete fast suite was
rerun successfully. The numerical cylinder convergence limitation remains: these
regression passes do not complete C3 or certify sphere convergence.

## Public sphere: the required order is not reached

The public `fd.Sphere` path, without substituting coefficients, was run for
radius 0.25, epsilon 2.25 and wavelength 1. The three centre offsets, measured
in fractions of a voxel, are `(0,0,0)`, `(1/2,1/3,1/5)` and `(1/4,2/3,2/5)`.
All nine results are retained in
[`fdtd_subpixel_mie_external.log`](../studies/results/fdtd_subpixel_mie_external.log).
These are research measurements, not replacement goldens.

| Vacuum cells per wavelength | Mean signed scattering error | Placement range of signed error |
| --- | ---: | ---: |
| 20 | +2.13260% | 0.01518 percentage points |
| 28 | +1.17334% | 0.01769 percentage points |
| 40 | +0.64978% | 0.00290 percentage points |

The signed-mean fit is **p=1.71401**, with 1.3807% relative fit scatter. The
placement RMS gives **p=1.71402**, with 1.3796% scatter. Thus the small placement
spread does not hide the failure of the specified `p >= 1.8` target. No
first-order fallback is substituted for that sphere requirement. The weak- and
high-contrast sphere sweeps have not been used to claim a completed gate.

The baseline uses 24 periods and 12-cell absorbing layers. The nested half-widths
are 0.4 for absorption, 0.55 for TFSF, 0.7 for scattered Flux, and 0.75 for
FarField, followed by a 0.2 air gap to the layer. At 20 cells this leaves three
cells from the sphere to absorption, three to the source boundary, and three
to scattering. The FarField surface is one further cell out. These dimensions
define the measuring fixture; they are not fitted accuracy tolerances.

Controls at 40 cells and the origin placement compare to its baseline
`Qsca=0.8691566659595557`:

| Change | Relative change in Qsca |
| --- | ---: |
| 24 to 36 periods | -7.705e-8 |
| 12 to 24 absorbing cells | +2.779e-5 |
| Air gap 0.2 to 0.5 | +4.958e-5 |

The largest measured control change is more than 130 times smaller than the
baseline Mie error, 0.00648122. This supports a resolved error in the current
discretisation rather than a dominant time-tail or absorbing-layer error on this
fixture. It does not uniquely identify the omitted tensor components as its
cause. Geometry moments, component coupling and other discretisation choices
remain distinguishable hypotheses.

Scattered Flux and the larger FarField surface agree to about 2.1e-7 relative
at 20 cells, and 7.8e-8 at 40 cells. The angular integral's difference from
scattered Flux decreases from about 7.3e-4 to 1.8e-4. The forward optical theorem,
with the incident amplitude explicitly referred to `plane=0`, differs from
`Qsca+Qabs` by 4.924% at 20 cells and 1.236% at 40. These finite-grid closure
errors are recorded, not silently set to zero. Lossless absorption at the
baseline origin placements is between -1.16e-6 and +2.57e-7 in efficiency.
Doubling the angular integration from 24/32 to 48/64 samples at the 20-cell
glass fixture changes the integral by only 7.78e-15 relative; the stated closure
residual is not an unresolved angular-quadrature error on that fixture.

The 20-cell origin case was repeated through the public run on NumPy and Torch.
Against JAX, the largest Qsca difference is 8.89e-16; the largest difference
among scattering, absorption, FarField flux, angular integral and optical
efficiencies is 1.45e-15. The two angular cuts differ by at most 1.12e-16 in
differential cross-section. This is a backend repeat of this case, not a claim
that every unmeasured scene agrees.

## External runs: Meep and the released FDTDX package

Meep **1.34.0** was run through its documented public `Simulation`, `Sphere`,
`Cylinder` and flux APIs. Its
[Mie tutorial](https://meep.readthedocs.io/en/latest/Python_Tutorials/Basics/#mie-scattering-of-a-lossless-dielectric-sphere)
describes the paired empty-cell subtraction used here. No Meep implementation
was opened, inspected or copied. The independent Mie series above remains the
accuracy reference; neither numerical solver is declared exact.

The sphere has the same radius, epsilon, wavelength and linear polarization.
Meep uses a plane source and subtracts the saved incident fields at all six
faces, whereas wavebit uses TFSF. Their pulse-width definitions, source end
times, cell rounding and flux interpolation differ. Meep's retained final
time is 124 units for this source and duration. At resolution 28 its requested
cell is rounded differently from wavebit's even-sized cell, so matching a label
such as "origin" does not establish identical sampling lattices.

| Vacuum cells per wavelength | Wavebit origin Qsca | Meep Qsca | Meep error against Mie |
| --- | ---: | ---: | ---: |
| 20 | 0.881963681476 | 0.873713585151 | +1.17581% |
| 28 | 0.873608033227 | 0.868270025941 | +0.54545% |
| 40 | 0.869156665960 | 0.866110759456 | +0.29541% |

Meep's one-placement sphere fit is **p=1.99015**, with 6.7178% scatter. This is
descriptive evidence, not an equally complete placement study or a proof of
which wavebit term causes the missing order. Meep's measured lossless Qabs is
-0.0087343, -0.0040824 and -0.0021492 on these grids. Its total-field power
interpolation therefore also has a visible finite-grid residual. The expanded
Meep controls below are for the cylinder; they are not a measured 3-D sphere
uncertainty budget.

FDTDX **0.6.2**, installed from PyPI, was run with its public `Cylinder` and
`UniformPlaneSource`, a paired empty run and complex-field subtraction. Its
native cylinder is a **binary voxel mask**; no subpixel projection was invoked
or inspected. It is a staircase control, not a smoothed reference. The current
online documentation describes `TFSFPlaneSourceRegion`, `UniformGrid` and a
closed phasor-flux detector that are absent from this installed release. The
interface uses its actual API rather than silently substituting wavebit's
coefficients. A three-dimensional FDTDX sphere has not been measured here.

The released FDTDX plane source initially rejected the one-cell invariant axis
because it squeezed its transverse amplitude array to one dimension. The run
instead uses two identical periodic slices along that axis. Its public phasor
detector collocates E and H in space and averages H in time; the interface then
integrates the four faces with trapezoidal weights, without interpolating again.
The periodic transverse span is 50 wavelengths and the duration 45 periods,
matching the wavebit high-contrast cylinder fixture. No installed package was
patched to obtain these results.

The following origin-placement comparison uses the same continuum cylinder,
`R/lambda=0.15`, epsilon 12 and axial magnetic polarization. Entries are signed
errors relative to the analytic cross-section 0.909413120174101:

| Vacuum cells per wavelength | Wavebit diagonal smoothing | Meep averaging enabled | FDTDX native binary mask |
| --- | ---: | ---: | ---: |
| 20 | +0.71970% | +24.94370% | +48.45261% |
| 28 | -0.87808% | +11.14344% | -14.41871% |
| 40 | -2.89917% | +6.71731% | -5.28040% |
| 56 | -3.08985% | +2.40261% | -3.01004% |

A single row would give a misleading ranking. In particular, the smallest
coarse-grid error is not evidence of the correct convergence order. Meep was
also run at two shifted cylinder placements on every grid, at 80/112 cells at
the origin, and with averaging disabled at 40 cells. All results, including
their placement variation, are retained in the log. Meep's three-placement mean
errors at 20/28/40/56 are +29.5315%, +9.8602%, +6.5540%, +2.5982%; the large
coarse error is not hidden by selecting the origin.

At 40 cells, doubling Meep's after-source duration changes the cylinder result
by -4.02e-9 relative; doubling the layer thickness changes it by +5.35e-6;
increasing the air gap and moving the source changes it by -4.04e-6. Disabling
averaging gives cross-section 0.800199826007 instead of 0.970501227658. The
axial-electric control gives 1.001437616650 against its Mie 1.016762944607.
For FDTDX the three measuring contours differ by about 1.1e-3 relative at 20
cells and 3.1e-5 at 56. Its time/period/PML controls have not been expanded into
an uncertainty budget. These limits prevent treating it as an exact oracle.

Both external solvers ran on CPU in an isolated Ubuntu/WSL environment:
Python 3.12.14, NumPy 2.4.6, SciPy 1.18.1, JAX/JAXlib 0.11.2. Wavebit uses the
Windows environment recorded above. OMP, OpenBLAS and MKL each use two threads;
no XLA flags were set. Elapsed times are retained but are not a performance
benchmark: some runs overlapped and the environments differ.

### Reproduce the external and sphere measurements

Choose fresh output filenames. Each interpreter containing the named external
solver is called directly; external packages are not wavebit dependencies.

```shell
python studies/fdtd_subpixel_mie_sphere.py sphere20.json --cpw 20
python studies/fdtd_subpixel_mie_sphere.py sphere28.json --cpw 28
python studies/fdtd_subpixel_mie_sphere.py sphere40.json --cpw 40
python studies/fdtd_subpixel_mie_sphere.py shifted40.json --cpw 40 --offset 0.5,0.3333333333333333,0.2
python studies/fdtd_subpixel_mie_sphere.py long40.json --cpw 40 --periods 36
python studies/fdtd_subpixel_mie_sphere.py thick40.json --cpw 40 --pml 24
python studies/fdtd_subpixel_mie_sphere.py gap40.json --cpw 40 --gap 0.5
"$MEEP_PYTHON" tests/references/interface/measure_fd_stage4_meep.py meep-sphere.json --dimension 3 --eps 2.25 --cells 20,28,40
"$MEEP_PYTHON" tests/references/interface/measure_fd_stage4_meep.py meep-cylinder.json
"$MEEP_PYTHON" tests/references/interface/measure_fd_stage4_meep.py meep-long.json --cells 40 --after 48
"$FDTDX_PYTHON" tests/references/interface/measure_fd_stage4_fdtdx.py fdtdx-cylinder.json
```

### Tests of the new measuring code

A radiating electric dipole with an independent analytic polarizability pins
the sphere instrument's power normalization, two angular cuts and forward
phase. Its radiation-reaction correction makes the optical theorem exact.
The angular integral closes within 6.7e-16 relative; finite-size Mie differs
from its dipole pattern by 5.4282e-7 of the peak. A manufactured radial field
independently pins the collocated FDTDX contour: its exact power is pi and
20/40/80-cell errors are 4.3845e-4, 1.0962e-4 and 2.7403e-5. Isotropic outward
radiation independently pins Meep's signed-face and incident-area normalization.

The six new tests pass. Seven actual source edits were applied one at a time,
detected and restored: leave the incident phase at the entry plane (1 failure),
swap the angular polarizations (1), reverse absorption sign (1), add unsigned
face powers (2), use the wrong incident-area dimension (2), open the trapezoid
endpoints (2), and replace complex H conjugation by its real part (2). Every
mutation run exited 1 through test failures, with no collection or internal
pytest error.

The combined analytic-reference, measuring-instrument and documentation run
reports **97 passed**, exit 0; its XML contains exactly 97 test cases, zero
errors, failures or skips. A fresh execution of the committed Meep interface
reproduces its 40-cell cylinder cross-section exactly at printed precision.
This follow-up changes studies, external interfaces, their tests and this
record. Production solver files and all four frozen archives are unchanged;
the earlier full-suite and bit-identity results are identified separately above.
The C3 sphere order gate remains failed.

## Isolating the cause: surface moment versus constitutive cross terms

The controlled experiment is
[`fdtd_subpixel_constitutive.py`](../studies/fdtd_subpixel_constitutive.py).
Its measurements are in
[`fdtd_subpixel_constitutive.log`](../studies/results/fdtd_subpixel_constitutive.log).
It keeps the public sphere, exact fills, incident pulse, grid, time step,
absorbing layers and measuring surfaces fixed. The substitutions are explicit
research arms, not an added public smoothing option.

### An incorrect geometric argument, corrected

The geometry docstring previously claimed that a spherical surface's diagonal
normal moment requires volume second moments. It does not. Let coordinates be
relative to the sphere centre, let `V` be the clipped volume, and let `A_i-`
and `A_i+` be the clipped disc areas on the voxel's lower and upper faces.
The divergence theorem applied to `x_i e_i` gives

```text
div(x_i e_i) = 1
x_i n_i = R n_i^2 on the sphere
M_i = integral_surface(n_i^2 dA) = [V - (u_i A_i+ - l_i A_i-)] / R
surface area = M_x + M_y + M_z
```

The research implementation uses this identity and an exact geometric support
test. It does not infer the presence of a surface from a cancellation residue.
Independent integration in `(z, azimuth)`, splitting at rectangle-edge
intersections, measured maximum `|dM|/h^2` of
`4.44e-15 / 1.62e-14 / 3.08e-14 / 1.33e-13` at `R/h=3/5/10/20`
(60 seeded random cut voxels per resolution). Closed spherical-cap integrals
also check the expression on NumPy, JAX, compiled JAX and Torch.

Changing only this moment does **not** repair the glass sphere's order:

| Vacuum cells/wavelength | Current origin Qsca | Exact-moment origin Qsca | Relative change |
| --- | --- | --- | --- |
| 20 | 0.881963681476089 | 0.881997458256976 | +3.8297e-5 |
| 28 | 0.873608033226615 | 0.873623517790378 | +1.7725e-5 |
| 40 | 0.869156665959556 | 0.869161576771374 | +5.6501e-6 |

The exact-moment origin fit is **1.71793**, with **1.8270%** scatter. The
production tangent-plane moment therefore remains an identifiable geometric
approximation, but it does not explain this fixture's failed order.
Applying the exact moment together with the cross terms at 40 cells gives
`Qsca=0.867399790510`, only `+6.1890e-6` relative to the cross-only result.

### The discarded term has a local electromagnetic defect

Write `K=epsilon_eff^-1`, `a=<1/epsilon>` and `b=1/<epsilon>`.
For a flat interface with unit normal `n`, the complete local relation is

```text
K_ij = b delta_ij + (a-b) n_i n_j
E_i = K_ii D_i + sum_(j != i) K_ij D_j
```

`coefficients.mix` retains only `K_ii`, and the kernel's electric update
multiplies each curl component by its own scalar `be`. The omitted expression
is **`(a-b) n_i n_j D_j` for `j != i`**. No more accurate fill or diagonal
normal moment can supply that coupling.

This has an independent, local check: at a half-filled 45-degree interface
between epsilon 2.25 and 1, continuity of normal `D` and tangential `E` gives
normal inverse epsilon `a=13/18` and tangential inverse epsilon `b=8/13`.
The diagonal truncation instead gives their mean in both directions. Its local
relative errors are **7.3964% normal** and **8.6806% tangential**, despite the
normal being exact. The complete tensor satisfies both conditions. This local
calculation is not a bound on the whole sphere's scattering error.

### Restoring that term changes the measured order

The comparator uses a common-node cross coefficient and adjoint interpolation:
average component `j` to the node, multiply by `K_ij`, then average to component
`i`. This is the discretization described in
[Oskooi, Kottke and Johnson (2009), Fig. 1](https://math.mit.edu/~stevenj/papers/OskooiKo09.pdf).
The diagonal coefficients and production tangent-plane moments are retained.
No Meep implementation source is used.

The same offsets `(0,0,0)`, `(1/2,1/3,1/5)` and `(1/4,2/3,2/5)` in voxel
units are retained at every resolution:

| Cells/wavelength | Diagonal mean relative error | With cross terms | Placement spread with cross terms |
| --- | --- | --- | --- |
| 20 | 0.0213260 | 0.0181319 | 1.8535e-4 |
| 28 | 0.0117334 | 0.00904683 | 2.9648e-4 |
| 40 | 0.00649781 | 0.00446104 | 4.5208e-5 |
| Fitted order | 1.71401 | **2.02267** | |
| Fit scatter | 1.38068% | **0.974414%** | |

The placement-RMS fit is **2.02268**, with **0.96736%** scatter. This isolates
the discarded cross terms as the cause of the observed order loss in this
glass fixture. It does not prove second order for every contrast or interface.
The final-script origin runs reproduce the independently constructed in-memory
prototype's three Qsca values to all reported digits.

For the origin glass fixtures, absolute row sums enclose the symmetric
constitutive matrix's eigenvalues in `[0.4314,1.0192]`, `[0.4270,1.0232]`
and `[0.4259,1.0255]`. Positive lower endpoints establish positive definiteness
for these sampled operators. At the fixture's Courant factor 0.5, the upper
endpoints also stay within the homogeneous curl-norm time-step bound. These
are not proofs for arbitrary shapes or for the absorbing-layer update.
The measured field-decay ratios remain near `1e-11`.
Across all nine glass operators the eigenvalue enclosures stay inside
`[0.4250,1.0281]`. The original diagonal energy diagnostic is excluded from
this stability evidence.

### Why this is still a comparator

At the time of this study, the public solver used the diagonal relation. Its interior-energy
diagnostic assumes that relation, so its energy output is **not an oracle for
the research tensor arm** and is not reported as one. With a full tensor,
recovering `D_i` by dividing `E_i` by `K_ii` is incorrect. A production change
must implement the constitutive update and its energy accounting consistently,
then repeat backend agreement, stability, contrast, pattern and phase gates.

The new fast checks use independent cap integrals, the staggered Fourier
symbol, an inhomogeneous adjoint identity, the electric-only runtime hook,
and flat-interface boundary conditions. Six actual source mutations are
detected: omitted face flux, tangent-plane substitution, reversed adjoint
shift, applying electric coupling to H as well, dropped cross terms, and
omitted spectral radius. Each source is restored after its failing run.
The lower-radius omission initially survived the two-dimensional-normal
fixture because its uncoupled z row retained the same lower endpoint. A
body-diagonal normal removes that accidental protection and detects the
mutation. The corrected normal/tangential enclosure is checked analytically.

```bash
python studies/fdtd_subpixel_constitutive.py tensor-origin.json --arm tensor
python studies/fdtd_subpixel_constitutive.py tensor-shift1.json --arm tensor --offset 0.5,0.3333333333333333,0.2
python studies/fdtd_subpixel_constitutive.py tensor-shift2.json --arm tensor --offset 0.25,0.6666666666666666,0.4
python studies/fdtd_subpixel_constitutive.py exact-moment.json --arm moment
python -m pytest tests/test_fd_constitutive_study.py -q -p no:cacheprovider
```

These are CPU float64 measurements with the numerical thread limits stated
above. No accuracy threshold was changed, no reference archive was rebaselined,
and this study does not complete C3.

Validation of this research addition: **109 selected fast tests**, including
the 12 new cases and the documentation contract; all reported cases match the
collected count, with exit code 0 and no INTERNALERROR. All six final source
mutations fail their selected tests with exit code 1 and are restored. The
production edit only corrects docstrings: after docstring removal its parsed
syntax tree is unchanged. The full fast/slow suites and four reference freezes
were not rerun for this research-only addition; their earlier results above
are not presented as new runs.

## Production constitutive coupling and energy

The kernel now integrates electric displacement D and magnetic induction B
in compact lossless regions around smoothed geometry. It reads each
**source-corrected** curl, advances D/B, and adds the off-diagonal field
increment after the existing diagonal update. For fixed material and zero
initial field this maintains `E=K_e D`, `H=K_m B` up to accumulation roundoff.
Dividing E by `K_ii` does not recover D.

Cross coefficients are sampled at common E nodes and common H cell centres.
For E, the component-to-node shift is +1 and the return shift -1; H uses the
opposite pair. Both averages have weight 1/2. They are adjoints, including
finite ends with zero extension. PEC/PMC field masks also project the flux
increments. Intersecting support boxes, including one interpolation-cell halo,
merge before memory allocation; disjoint regions write disjoint cells. Scene
painting order applies to cross coefficients too.

Within these regions, the interior-energy observer replaces the diagonal
contribution by the same conserved leapfrog quadratic form:

```text
W^n = Re(E^n* . D^n) + Re(H^(n-1/2)* . B^(n+1/2)).
```

NumPy reads before its destructive H update, after B memory advances. JAX
and Torch read the magnetic half after H updates and the electric half at
the end of the previous step. This diagnostic carries no gradient.

### Independent instruments

The [kernel measurement record](../studies/results/fdtd_subpixel_coupled_kernel.log)
retains the numerical probes, actual source mutants and whole-suite validation.

CPU float64, with the numerical thread limits stated above. The closed PEC
cavity is 16 cubed at dx=.05, with a radius-.15 sphere centred at
(.012,.007,-.009), epsilon=2.25, mu=1.5, courant=.5, 80 steps and arbitrary
consistent D/B data from seed 224. The initial state is an internal fixture;
it does not claim that a public E/H initialization inverse solve exists.

| Instrument | Measurement |
| --- | ---: |
| Relative conserved-energy span, NumPy | 2.86e-15 |
| Same, eager JAX / compiled JAX | 5.05e-16 / 5.05e-16 |
| Same, Torch | 1.69e-16 |
| Maximum backend field difference after 80 steps | 3.31e-14 |
| Energy span with sphere tangent to low PEC face | 2.86e-15 |
| Energy span with sphere one cell from high PEC face | 3.70e-15 |
| JAX / Torch material derivative, eight steps | -103.80604302068576 / -103.80604302068565 |
| Central-difference relative errors at steps 1e-2 / 1e-3 / 1e-4 / 1e-5 | 3.09e-5 / 3.09e-7 / 3.03e-9 / 1.18e-9 |
| First public 20-cell glass-sphere Qsca | 0.8791374695867283 |
| Difference from controlled tensor comparator | 1.11e-16 absolute |

The derivative objective is `sum(E**2 + .3*H**2)`; .3 makes both fields
observable. The selected finite-difference step 1e-4 precedes the subtraction
noise plateau. Tests cite these measurements beside their tolerances. Fourier
symbols cover both staggerings on all four execution paths. A dense finite
operator built from independent neighbour indices checks adjoints, end
treatment and both spectral bounds against every eigenvalue. A unit electric
or magnetic dipole sample checks flux memory and its coupled field response,
independently of pulse timing.

### Preparation-time certificates and limits

The diagonal plus/minus the absolute off-diagonal row sum gives a Gershgorin
spectral enclosure. A strictly positive lower endpoint certifies positive
energy. The upper E/H endpoints bound the curl time step, combined with the
existing pole-rate bound for remote pole regions. A negative lower endpoint
is **inconclusive**, not proof of a negative eigenvalue: planning refuses it
with the measured enclosure. This lossless curl certificate is not a PML
stability proof. All nine epsilon=12 plans at 40/56/80 cells and the three
stated placements pass the certificate.

A prepared plan lowered under differentiation evaluates the same whole-operator
certificate on its backend. Failure sets the existing `stable` diagnostic and
update arrays to NaN. No per-cell exception or warning can disappear under JIT.

The compact region and its interpolation halo must avoid conductivity,
dispersion and absorbing layers. Cross coupling on a symmetry plane needs a
mirror-weighted adjoint and is refused. Initial E, and initial H for a magnetic
smoothed scene, need an inverse constitutive solve not yet provided; planning
names the missing solve and recommends source excitation. These are explicit
limits, without diagonal fallbacks.

The overlap check includes the common points. A diagonal dimer with a 1.7-cell
surface gap has **zero** shared voxels at the six Yee locations but **one** at
the common E node, where each sphere fills 3.55e-6. Omitting that point admits
a three-medium mixture. The duplicate-sphere fixture now counts 2422 shared
voxels, versus 1815 at Yee points alone; the axial .95-cell dimer counts 9,
versus 8.

CSG approximation reports include these additional points. The two-sphere
fixture counts 348 bounded voxels with maximum width .06124, versus 284 and
.05377 at the three E components alone. Custom bodies using a nearest-surface
cross-moment proxy additionally report the measured discrepancy between that
normal's squared components and their supplied diagonal surface moments.
The fixture's measured maxima are .628253 for Union and .821168 for both
Intersection and Difference; all twelve operation/backend cases assert the
reported values, and omitting the measurement fails all twelve.
This is a geometry-proxy measurement, not an electromagnetic error bound.
Creases and CSG cross moments remain separate C4 validation work.

```bash
python -m pytest tests/test_fd_constitutive.py -q -p no:cacheprovider
python studies/fdtd_subpixel_mie_sphere.py --cpw 20 --backend jax public-tensor20.json
```

### Actual source mutations

All 26 mutations below were applied to production source, executed in separate
pytest processes, observed to fail, and restored byte for byte. Every final
run exits 1 with assertion failures, no collection errors and no INTERNALERROR.
The merge mutation initially stopped at the independent internal-support
certificate; the test was strengthened to assert the merged support count
before validation, then the mutation was rerun and failed that assertion.

| Source mutation | Failing cases |
| --- | ---: |
| Reverse the return average's adjoint shift | 10 |
| Use periodic wrapping at finite ends | 2 |
| Omit the cross field increment | 8 |
| Omit the current from D/B memory | 8 |
| Reverse the magnetic increment sign | 4 |
| Omit wall projection of flux increments | 1 |
| Stop the cross-coefficient gradient | 2 |
| Retain diagonal electric energy | 4 |
| Retain diagonal magnetic energy | 4 |
| Sample H cross terms at the wrong common point | 4 |
| Omit the initial-field inverse-solve door | 2 |
| Omit the positive-energy door | 1 |
| Omit the coupled CFL door | 1 |
| Omit the traced positive-energy certificate | 1 |
| Keep overlapping memory regions separate | 1 |
| Omit the interpolation halo's absorbing-layer door | 1 |
| Omit its conductivity/pole door | 1 |
| Omit the mirror-weighted-adjoint door | 1 |
| Skip common points in the shared-surface check | 1 |
| Substitute vacuum for the held background phase | 1 |
| Use a corner's nearest normal for axis-aligned box cross moments | 1 |
| Omit the lower enclosure's row radius | 2 |
| Omit D/B bytes from the state report | 1 |
| Omit common points from the approximation report | 12 |
| Omit the incomplete-accuracy warning | 4 |
| Omit the measured custom-body normal-proxy discrepancy | 12 |

### Whole-engine regression result

Both complete test tiers ran with three workers, the stated numerical thread
limits, and cache writing disabled. Collected and reported counts agree:

| Verification | Result | Exit |
| --- | --- | ---: |
| Fast tier | 1,901 passed, 6 expected NumPy differentiation skips; 1,907 collected/reported | 0 |
| Slow tier | 65 passed; 65 collected/reported | 0 |
| Stage 0 reference | 143 arrays reproduce bit for bit | 0 |
| Stage 1 reference | 288 arrays reproduce bit for bit | 0 |
| Stage 2 reference | 140 arrays reproduce bit for bit | 0 |
| Stage 3 reference | 570 arrays reproduce bit for bit | 0 |

There are no INTERNALERROR or collection errors. No reference archive was
rewritten. The first fast run found two historical energy-mutant signatures
that needed the new optional memory argument and one unresolved docstring
path. An isolated rerun also exposed an order-dependent cached reference in
the bare-curl mutant; its healthy reference now precedes mutation. Both
mutants pass alone, and the full fast tier above was rerun after these fixes.
The measurement record retains the failed run and its resolution.

These passes verify the coupled kernel and preserve the earlier engine.
They do not replace the full production sphere convergence gate and its
electromagnetic source mutations. C3 remains open.

## Public sphere measurements with the coupled kernel

The [production measurement record](../studies/results/fdtd_subpixel_production_sphere.log)
uses kernel commit `c2c8d98`. All runs use the public scene and `fd.run`, JAX
float64, radius .25, wavelength 1, the three stated placements and 24 periods
unless the time column says otherwise. Every completed process exits 0.

| Epsilon | Mean errors at 20 / 28 / 40 cells | Mean order | Log-fit scatter | RMS order |
| ---: | --- | ---: | ---: | ---: |
| 2.25 | .01813193 / .00904683 / .00446104 | 2.02267353 | .974414% | 2.02267636 |
| 1.1 | .00997457 / .00493858 / .00236727 | 2.07489568 | .318473% | 2.07494199 |

Both families meet the fitted-order and scatter criteria, and their finest
errors exceed ten times the stated 9.7e-5 open-scene floor. The largest
late-monitor energy fraction for epsilon 1.1 is 5.28e-9. These eighteen runs
establish these two families; the epsilon-12 family and the production
electromagnetic mutations remain outstanding.

### High-index time truncation

The first epsilon-12 runs are visibly unresolved. They are excluded from any
spatial-order fit. The origin's Mie efficiency is 4.315415843395232.

| Periods, at 40 cells | Qsca | Apparent Qabs | Largest late-monitor energy fraction |
| ---: | ---: | ---: | ---: |
| 24 | 3.36067685425 | .31368604163 | .0375576860 |
| 48 | 3.43571859257 | .14522379924 | .0027553423 |
| 288 | 3.41818846015 | -.00067742618 | 5.8496611e-7 |
| 456 | 3.41722828553 | .00030935229 | 2.8138152e-8 |
| 864 | 3.41728005839 | -6.5655589e-11 | 7.0893296e-10 |

The 24-period batch was stopped after its first two high-index cases; no
56- or 80-cell result from that unresolved time window is presented. Source
hashes still matched the saved sweep when it stopped. The public diagnostics
issue UnsettledFieldWarning for these cases.

The correct interior-energy history over times 36..48 fits an exponential
with energy time constant 28.6225 periods and maximum log residual .00414.
An independent frequency scan of the analytic Mie coefficients finds the
magnetic n=3 resonance near frequency 1.03717 with power FWHM .0053392,
corresponding to an estimated energy time constant 29.81 periods. The scan
has frequency spacing 1.86e-4; these are measured estimates, not exact poles.
The source's spectral amplitude there is .9333 of its carrier amplitude.
This long-lived resonance explains why the glass run length is insufficient.

The completed 288-period trial was chosen from
`48 + 29.81*log(.0027553423/1e-6)`, rounded up to a multiple of the original
24-period window. Here 1e-6 is one tenth of the existing 1e-5 unsettled-energy
line. This extrapolation chooses an experiment; its actual decay, Qabs and
time-extension change must be measured before selecting a production test
duration or accepting a high-index order. It exits 0, without an
UnsettledFieldWarning. Its Qsca still differs from the 48-period value by
-.51023%, and the apparent negative absorption above is retained rather than
clipped. Absence of the warning does not establish time convergence.

At 288 periods, the final interior energy divided by its peak is 8.84262e-6.
This whole-domain measure differs from the late-monitor energy fraction in
the table. Fitting times 240..288 gives an energy time constant of 34.0645
periods and maximum log residual .01110; the late decay is slower than the
earlier fit. The completed extension to 456 periods was chosen from
`ceil((288 + 34.0645*log(100))/24)*24`: two further decades of energy decay
would reduce a residual field amplitude by ten. This remains a trial, not a
certified duration.

The 456-period process exits 0. Relative to 288 periods, Qsca changes by
-2.80902e-4, Qsca+Qabs by +7.78457e-6, and the complex forward field by
7.77038e-4. The largest change in either angular cut, divided by that cut's
peak intensity, is .00130910. Thus nearly unchanged total extinction alone
would miss remaining differences in the separate observables.

The final interior energy is 4.011996e-7 of its peak. Fits over successive
48-period windows give energy time constants 39.18 (288..336), 51.33
(336..384) and 72.32 (384..432); the final window 408..456 gives 83.96035,
with maximum log residual .01342. Extrapolating the earlier 34-period time
constant was too optimistic. The completed 864-period trial was chosen from
`ceil((456 + 83.96035*log(100))/24)*24`, again targeting two further energy
decades before measuring the actual change.

The 864-period process exits 0. Relative to 456 periods, Qsca changes by
+1.51505e-5, Qsca+Qabs by -7.53699e-5, and the complex forward field by
9.12316e-5. The largest peak-normalized change in the angular cuts is
1.77691e-4. The Qsca change is below the previously measured 9.7e-5
open-scene floor; apparent absorption is now -6.56556e-11. The final interior
energy is 1.10498e-8 of its peak. Its final 48-period fit has time constant
116.68585 and maximum log residual 9.84217e-5, consistent with the slower
tail seen in preceding windows.

These controls justify proceeding with 864 periods for the spatial study;
they do not prove the same temporal error at every resolution or placement.
Each new run retains its decay, absorption and energy history, and any
unresolved case requires its own extension. The 40-cell origin still differs
from Mie by -20.8123%, far above the measured time-extension change. The
full 40/56/80-cell, three-placement sweep is now complete. Its physical source
mutations remain incomplete.

### High-index spatial refinement

The following origin placements use the same 864-period duration. Every
completed run exits 0, and the kernel source hashes still match the recorded
coupled-kernel sweep.

| Cells per wavelength | Qsca | Relative error against Mie | Apparent Qabs | Largest late-monitor energy fraction |
| ---: | ---: | ---: | ---: | ---: |
| 40 | 3.41728005839 | -20.8122651% | -6.56556e-11 | 7.08933e-10 |
| 56 | 3.80237882305 | -11.8884724% | -6.57791e-6 | 5.42638e-10 |
| 80 | 4.05061866133 | -6.1360757% | 6.30685e-6 | 4.12368e-10 |

The 56-cell run takes 6351.5 seconds on the stated CPU setup. Its final
interior energy is 1.08325e-8 of its peak; the 816..864 fit gives an energy
time constant of 116.78486 periods and maximum log residual 7.42419e-5.
These residuals permitted proceeding to the 80-cell measurement; this is not
an independent time-extension bound at 56 cells.

At 56 cells, the FarField surface flux differs from the scattered Flux by
7.81749e-8 relative, the angular intensity integral by -3.91514e-4, and the
optical-theorem result from Qsca+Qabs by -.209252%. Each angular cut has
15.2871% peak-normalized error against Mie. The descriptive two-resolution
Qsca order is 1.66425, below 1.8; two resolutions at one placement cannot
replace the required three-resolution, three-placement fit.

All nine runs at offsets `(0,0,0)`, `(.5,1/3,.2)` and `(.25,2/3,.4)` have
now completed with exit code 0. The offsets are fractions of a grid cell.
The finite batch also exits 0 and verifies unchanged production source
hashes before and after each run. No unresolved 24- or 48-period result
enters this fit.

| Cells per wavelength | Mean signed relative error | Placement RMS error | Placement range |
| ---: | ---: | ---: | ---: |
| 40 | -21.9223315% | 21.9363882% | 1.6892727 percentage points |
| 56 | -11.6311266% | 11.6326012% | .4282868 percentage points |
| 80 | -6.1771641% | 6.1772496% | .0794428 percentage points |

The mean fit gives **order 1.8268534**, with **1.27144%** log-fit scatter;
the RMS fit gives **1.8277523**, with **1.28560%** scatter. Both meet the
predeclared order 1.8 and scatter 10% criteria. The finest mean error is
636.82 times the previously measured 9.7e-5 open-scene floor. Cancellation
between placements does not explain the fitted order.

Across the nine runs, the largest absolute apparent Qabs is 6.57791e-6,
the largest late-monitor energy fraction is 7.41356e-10, and the final
interior energy divided by its peak ranges from 9.30745e-9 to 1.17275e-8.
These per-case controls accompany the origin's time-extension measurement;
they are not an independent temporal-error bound for every placement.
At 80 cells the largest peak-normalized angular-cut error is 7.98238%,
and the largest optical-theorem closure error is .0972985%.

The completed measurements establish the stated high-index scattering
convergence criterion. They do not finish C3: the coupled cylinder
revalidation below fails its predeclared order criterion.

### Coupled cylinder revalidation

The old cylinder study prepares an empty scene and replaces only diagonal
coefficients. It has no constitutive terms and cannot test the coupled kernel
by simply rerunning its command. The new
`studies/fdtd_subpixel_coupled_cylinder.py` retains the same grid, pulse,
paired empty run and closed measuring contour, while using the production
material lowerer, cross terms, compact flux memory and time stepper.

This remains a research instrument. It attaches the invariant cylinder slice
to the historical empty plan and applies the production whole-operator
positive-energy and CFL checks. Public periodic-face and declared-material
resolution checks are not exercised. In particular, the 40-cell epsilon-12
scene has only 5.99 cells per shortest pulse wavelength and is correctly
refused by public preparation. A first probe reached that refusal when
requesting its final public report and exited 1. The comparator's final
report now explicitly uses the historical empty-scene context. No production
refusal was weakened, and public `fd.Cylinder` remains C4.

At 40 cells, the prepared diagonal coefficients differ from the historical
substitution by at most 1.11023e-15. The coupled plan adds one constitutive
term with 117 nonzero yz cross coefficients, maximum magnitude .20286891.
The xy and xz pairs vanish. Its inverse-material enclosure is
`[.03471344, 1.10143445]`.

The [coupled-cylinder record](../studies/results/fdtd_subpixel_coupled_cylinder.log)
retains paired diagonal/coupled measurements at the origin. These use the
full transverse period 50 and time gate 45 from the historical experiment;
the earlier 56/80/112-cell fine study used their .55-scaled counterparts.
All four new comparisons exit 0 with unchanged engine source hashes.

| Cells per wavelength | Diagonal relative error | Coupled relative error | Coupled cross-section | Coupled late-monitor energy fraction |
| ---: | ---: | ---: | ---: | ---: |
| 40 | -2.89917% | +6.84118% | .971627672959 | 9.44687e-9 |
| 56 | -3.08985% | +3.72573% | .943295410203 | 3.54991e-9 |
| 80 | -2.85086% | +1.30613% | .921291240609 | 1.30968e-9 |
| 112 | -2.21760% | +.90022% | .917599801212 | 1.03914e-9 |

The coupled error decreases substantially under refinement. An origin-only
fit over 40/56/80 gives order 2.39444 but 13.9635% scatter; 56/80/112 gives
2.05807 but 23.5500% scatter. Neither meets both predeclared criteria. These
selected origin measurements do not establish a repaired cylinder gate.

The complete 40/56/80-cell sweep at all eight historical placements now
exits 0 with unchanged source hashes. It retains the full period 50 and
gate 45 and all per-case decay, absorption and contour controls:

| Cells per wavelength | Mean signed relative error | Placement RMS error | Placement range |
| ---: | ---: | ---: | ---: |
| 40 | +6.0280152% | 6.0707757% | 2.0865973 percentage points |
| 56 | +3.3089053% | 3.3416264% | 1.3761035 percentage points |
| 80 | +1.9815693% | 2.0189031% | 1.2187415 percentage points |

The mean fit has **order 1.6033652** and **4.06250%** scatter; the RMS fit
has **order 1.5865547** and **4.26088%** scatter. Both fail the required
order of 1.8. All 70 four-placement subsets span orders 1.37696 to 1.84300;
this descriptive spread is not a confidence interval and does not authorize
selecting a passing subset. Cross coupling repaired the specified glass
sphere fixture, but did not establish the cylinder convergence gate.

The coupled time and instrument controls have now completed at 40 cells:

| Control | Relative cross-section change |
| --- | ---: |
| Period and gate multiplied by 1.4 | -3.38427e-7 |
| Period alone multiplied by 1.4 | +1.35447e-14 |
| Period and gate multiplied by .55 | -8.13153e-5 |
| Gate alone multiplied by .55 | -8.13153e-5 |
| Source moved by +.2 wavelength | -3.70667e-6 |
| Absorbing-layer inner face moved out by 1 wavelength | +1.73273e-6 |

The full-duration baseline is reproduced exactly by the committed study
module. Its contour displacement changes the cross-section by at most
1.13752e-6 relative. The shortened schedules at 40/56/80/112 cells differ
from the full runs by -8.13153e-5, -1.22107e-4, -2.68518e-5 and
-1.20518e-5. The 56-cell difference exceeds the 9.7e-5 open-scene floor,
so the shortened schedule is not selected for the placement sweep. The
full period 50 and gate 45 are retained with per-case decay, contour and
apparent-absorption measurements. The global interior energy still contains
outgoing waves in the wide periodic cell; it is recorded and is not treated
as the local monitor's settling measure.

The 40-cell origin has also been rerun with NumPy and Torch, using the same
full period/gate and paired empty reference as the completed JAX run.
All three use unchanged production source. The comparison exits 0:

| Backend | Cross-section | Relative difference from JAX | Absolute apparent-absorption difference from JAX |
| --- | ---: | ---: | ---: |
| JAX, compiled executor | .9716276729593055 | reference | reference |
| NumPy | .9716276729593055 | 0 | -4.37391e-17 |
| Torch | .9716276729593067 | 1.33227e-15 | -2.84307e-16 |

This controls a backend-dependent explanation at this origin; it is not
an additional eight-placement fit or a public-cylinder validation.

The discretization needs an accuracy audit beyond the surface-volume
formula. [Oskooi, Kottke and Johnson (2009), Fig. 1](https://math.mit.edu/~stevenj/papers/OskooiKo09.pdf)
describes diagonal inverse-material entries at component locations and
off-diagonal entries at a common node, connected by adjoint averages, as in
the production operator. [Werner, Bauer and Cary (2013), sections 5 and 8](https://arxiv.org/html/1212.4857v2)
reports eventual first-order convergence for sharp interfaces with this
family of schemes. That paper also distinguishes accuracy from a positive
operator certificate. The engine already checks the assembled enclosure at
plan time. This literature comparison is a reason to investigate interface
consistency; it does not prove the cause of this scattering result or relax
the predeclared gate.

A separate static planar-interface diagnostic isolates the field mapping
from curved geometry and time integration. It uses a unit tangential E or
unit normal D, integrates E exactly along each Yee edge and D across its
dual face, then applies the production diagonal mixing and `cross_action`.
The interface has normal `(0,.6,.8)` and offset `.23*dx`; these are fixed
experiment coordinates, not optimized parameters. The sampling square has
fixed physical extent and excludes the interpolation halo.

| Epsilon | Excitation | Largest component error at 16/32/64 cells per unit length | Mean absolute error, 16 / 32 / 64 |
| ---: | --- | ---: | --- |
| 2.25 | unit tangential E | .08417033 at each level | .00180625 / .00092324 / .00046680 |
| 2.25 | unit normal D | .05887511 at each level | .00131555 / .00066994 / .00033809 |
| 12 | unit tangential E | .64111408 at each level | .01814667 / .00925197 / .00467191 |
| 12 | unit normal D | .17383109 at each level | .00424308 / .00215755 / .00108800 |

The exact input edge/face integrals satisfy discrete curl(E)=0 and div(D)=0
to at most 6.026e-12, including the `1/dx` factor. At epsilon 12 the mapped
field instead has maximum raw circulation 1.74682514 for tangential E and
.07709699 for normal D, independent of these three spacings. Axis-aligned
and zero-contrast controls have component error at most 1.111e-16.
Thus this operator does not exactly preserve the integrated static solution
at an inclined interface. These are absolute local errors in the stated
unit fields, not scattering-efficiency errors. Their support narrows as dx
decreases. This diagnoses a consistency limitation of the discretized map;
it does not by itself establish the global scattering order or identify a
stable replacement. No production formula or acceptance threshold is
changed by the diagnostic.

Exactness of the local static map also does not certify a time-domain
replacement. [Bauer, Werner and Cary (2011)](https://www.sciencedirect.com/science/article/pii/S0021999110006698)
derives an accurate edge/face map for a frequency-domain method; its
accuracy result does not provide the stability guarantee needed here.
A replacement must address interface consistency together with the
assembled operator's symmetry, positive energy and time-step bound. The
current measurements authorize neither an asymmetric update nor a claim
of universal second-order convergence from the three successful sphere fits.

The diagnostic is reproducible without a time-domain solve:

```bash
python studies/fdtd_subpixel_interface_consistency.py interface-consistency.json
```

```bash
python studies/fdtd_subpixel_coupled_cylinder.py cylinder-origins.json --cells 40,56,80,112 --placements 1 --diagonal-control
python studies/fdtd_subpixel_coupled_cylinder.py cylinder-placements.json --cells 40,56,80 --placements 8
python studies/fdtd_subpixel_coupled_cylinder.py cylinder-short.json --cells 40,56,80,112 --placements 1 --period-scale .55 --gate-scale .55
python studies/fdtd_subpixel_coupled_cylinder.py cylinder-controls.json --cells 40 --placements 1 --controls
```

### Exact circle moments and a separate edge/face comparator

The exact arc-moment control has completed at the worst placement of the
existing 80-cell sweep, `(1/8,4/9)` cell offsets. It changes both the
component diagonal moments and the common-node yz moment, keeping the
production interpolation and time stepper. The full period 50 and gate 45
are retained. The control exits 0 with unchanged engine source hashes:

| Moment rule | Cross-section | Relative error | Apparent absorption cross-section |
| --- | ---: | ---: | ---: |
| Nearest tangent normal | .932374635914496 | 2.52487184% | 3.83684e-7 |
| Exact clipped-arc moments | .932237121241328 | 2.50975058% | 1.82820e-7 |

The exact circle is split at its intersections with the four voxel sides.
On each retained arc of width d and midpoint m, the diagonal integrals use
`d*(1 +/- cos(2*m)*sinc(d/pi))/2` and the yz integral uses
`d*sin(2*m)*sinc(d/pi)/2`, multiplied by the radius. An independent
divergence identity using the clipped disc area and straight face integrals
differs by at most 1.45717e-14 after division by dx in this displaced case.
The earlier 40-cell origin is less well conditioned at grazing cuts: its
largest unweighted identity discrepancy is 1.04292e-8. That discrepancy
must not be described as machine-precision agreement. The 40-cell scattering
error changes from 6.84118% to 6.39030% with exact moments. Neither bounded
control establishes a repaired convergence gate.
The arithmetic is retained in `studies/fdtd_subpixel_circle_moments.py`,
with a direct quadrant-integral test for length and diagonal/cross moments.

The separate `fdtd_subpixel_triplet_cylinder.py` research comparator evaluates
the symmetric edge/dual-face construction of
[Werner, Bauer and Cary (2013), sections 4 and 7](https://arxiv.org/html/1212.4857v2).
It uses exact circle/segment intersections for E-edge length fractions and
D-dual-face fractions. In this invariant-x TE problem, each node has four
oriented yz pairs. The local map is `Gamma @ inv(Pi)`, with each row of
Gamma weighted by its edge fraction and each row of Pi by its face fraction.
Only its symmetric part is used in time stepping. The asymmetric map is
retained solely as a static consistency control; it is not advanced in time.
The paper does not promise universal second-order accuracy for this
symmetric construction, and this experiment makes no such assumption.

All local symmetric blocks are checked before a solve. Their four
permutations have weight 1/4 and cover every yz degree of freedom once;
their average is therefore positive definite when every block is positive
definite. The minimum and maximum local eigenvalues enclose the assembled
operator. Exact vacuum blocks at the compact region's ends complete this
argument for zero extension. The uncoupled x diagonal and the existing curl
norm bound enter the time-step check. Failed scope, halo, symmetry, positive
energy or CFL checks refuse the whole prepared experiment by name. This
certificate is derived for this lossless, invariant-x fixture; it is not a
general public-geometry certificate.

At the 40-cell origin the independent indexed matrix agrees with the shift
implementation to 1.66534e-16, is exactly symmetric, and has minimum
eigenvalue .08124190. The local lower bound is .06431980. Before the spatial
sweep, all 24 preparations have positive lower bounds (minimum .05429109)
and `dt^2 * rate < 1` (maximum .27589319). These are stability evidence,
not a scattering-accuracy result.

The full spatial comparison now exits 0 with 24 of 24 rows and unchanged
source hashes. The same eight placements, period 50, gate 45 and three
resolutions are used; neither a favorable subset nor a new resolution range
is selected after seeing these results. The complete measurements and
energy histories are in the
[research comparison record](../studies/results/fdtd_subpixel_triplet_cylinder.log).

| Cells per wavelength | Mean relative error | Placement RMS error | Placement range |
| ---: | ---: | ---: | ---: |
| 40 | 3.57784224% | 3.59012574% | .91980010 percentage points |
| 56 | 1.80991510% | 1.81270518% | .31959441 percentage points |
| 80 | 1.00863719% | 1.00915962% | .09981245 percentage points |

The mean fit gives **order 1.82480894**, with **4.55629%** scatter; the RMS
fit gives **1.82899317**, with **4.58937%** scatter. Both meet the predeclared
finite-comparison requirements of order at least 1.8 and scatter below 10%.
All 70 four-placement subsets span orders 1.72505 to 1.91950 (standard
deviation .04477); this is descriptive subset sensitivity, not a confidence
interval and not permission to choose a subset. No claim about an eventual
second-order asymptote follows from this finite range.

Across all 24 cases the largest late-monitor energy fraction is 6.01283e-9,
the largest apparent absorption cross-section magnitude is 7.90565e-6, and
moving the measuring contour by four cells changes the cross-section by at
most 1.50287e-6 relative. No runtime warning is emitted. The largest final
global interior-energy fraction is .00026933; the wide cell still holds
outgoing radiation, so this is not substituted for the local decay measure.

**This does not repair the production cylinder or complete C3.** Its existing
operator still has the failed order 1.60337. This cylinder instrument uses
the earlier empty-scene preparation limits. Public-cylinder support remains
C4. The separate three-dimensional extension below has its own edge/face
geometry, stability certificate, backend and local derivative checks, and
fresh sphere measurements. The old production sphere fits cannot be reused
to validate a changed operator.

For the same static tilted interface as above, the asymmetric local map
reproduces both exact excitations with maximum component error 1.11023e-15
across all three contrasts, both normals and three spacings. Symmetrization
leaves a residual: at epsilon 12 its tangential maximum is .22594325 and
normal maximum .09082507, compared with .64111408 and .17383109 for the
production map. Its tangential raw circulation is .62080315, but its normal
raw circulation is .09200501, larger than the production value .07709699.
These mixed local results require the scattering comparison; improvement of
one norm is not proof of a higher convergence order.

The initial nine research tests pass, including NumPy, JAX eager,
JAX compiled and Torch incremental actions. Eight actual study-source
mutations produce eleven assertion failures with matching collected and
reported counts, no incidental errors, and exit 1 for each mutant. They
replace edge fractions by area fractions, choose the wrong transverse chord,
change a 1/4 weight to 1/3, reverse the adjoint shift, omit symmetrization,
reverse the cross increment, permit negative local energy, and remove the
CFL check. Every mutation is restored byte for byte.

Three further tests cover exact quadrant moments, 90-degree rotation
covariance of the isotropic circle, and a fresh paired FDTD/Mie physical
control. The physical control keeps the measured 40-cell origin's 4.09268%
error below 5%, with its late-monitor fraction below 1e-7. This one-resolution
limit is a control of the instrument; it is not a substitute for the fit.

The mutation audit exposed a blind spot: a frozen circle normal survived
that scalar-power control (pytest exit 0, mutation-driver exit 1). This
unsuccessful witness is retained. The added rotation-covariance test catches
the same actual source defect: its discrepancy reaches .79469953, against
the healthy measurement 3.33067e-16 and assertion limit 1e-14. Doubling the
arc cross moment is caught by the analytic quadrant integral; omitting the
material from the local blocks is caught by the physical control (scattering
zero, hence relative error one). In total, **11 distinct source defects**
have **14 assertion-failure witnesses**, covering all twelve new test cases.
The successful witnesses have matching collected/reported counts, no
incidental errors, and exit 1. All mutated files are restored byte for byte.
These tests validate the research instrument; its convergence evidence is
the separately retained 24-case experiment.

Final targeted validation exits 0: **76 collected and 76 reported**, all
passed (the twelve research cases plus 64 documentation contracts), with no
skips, errors or `INTERNALERROR`. The portable exact-moment study also
exits 0 and reproduces the earlier 40-cell tangent/exact cross-sections.
Fresh stage 0/1/2/3 reference checks each exit 0: **143/288/140/570 arrays
reproduce bit for bit**, 1,141 in total. No reference is regenerated.
Engine and measured study/test source hashes remain unchanged during these
checks. The full fast and slow regression counts above belong to the
previous production validation; they are not relabeled as a new full-suite
run for these research-only additions.

```bash
python studies/fdtd_subpixel_triplet_cylinder.py triplet-static.json --mode static
python studies/fdtd_subpixel_triplet_cylinder.py triplet-preflight.json --mode preflight --cells 40 --placements 0
python studies/fdtd_subpixel_triplet_cylinder.py triplet-scattering.json --cells 40,56,80 --placements 0,1,2,3,4,5,6,7
python studies/fdtd_subpixel_circle_moments.py circle-moments.json --cells 80 --placement 4
```

### Three-dimensional edge/face sphere extension

The research construction now has eight oriented xyz triplets. It uses the
exact intersection of a sphere with each E component's line segment and
each D component's transverse dual square, at that component's own edge
midpoint. The normal is radial at the common node. The center's arbitrary
unit normal affects only homogeneous blocks and cancels from the result.
No production material operator or public API is replaced.

For edge fractions `ell`, face fractions `a`, material epsilon `e` and a
unit normal `n`, the local map is `Gamma @ inv(Pi)`. Define
`d_i = 1 + (e-1)*a_i`, `beta_i = (e-1)*a_i/d_i`, and
`denom = 1-sum(beta_i*n_i^2)`. The rank-one inverse gives

```
K_ij = delta_ij/d_i
     + (beta_i + (1/e-1)*ell_i)*n_i*n_j/(denom*d_j).
```

The time step uses only `(K+K.T)/2`. The nonsymmetric map is an exact-static
diagnostic, not a time-domain operator. Eight oriented permutations each
cover every electric degree of freedom once; averaging their symmetric
blocks gives a symmetric assembled map. Before execution, the preparation
checks finite fractions in [0,1], exact vacuum blocks at all six patch ends,
symmetry, strictly positive local eigenvalues, and the full curl-norm CFL
bound. The whole scene is refused by name if any check fails. No exception
or dynamic acceptance branch is hidden in a compiled cell update.

Unlike the historical cylinder fixture, this study first calls the ordinary
public Sphere scene planner. It retains material-resolution, source,
monitor, layer, loss and initial-field checks. It then substitutes the
research electric term in a vacuum background and certifies the expanded
patch. Magnetic, anisotropic, multiple-body and non-vacuum scenes are outside
this instrument's scope. The normal public accuracy warning is retained in
the raw records; its old production fit does not certify this substituted
operator.

A provisional unguarded disc-area evaluation gave a full face fraction
`1+2.29e-14`. The extension handles geometrically full/empty squares by
comparing the circle radius with the nearest/farthest square points before
evaluating a partial cut. This sets exact full/empty values from geometry,
without clipping a computed fraction to one. Inactive roots and inverse
trigonometric inputs are made regular before evaluation. Nine preliminary
grid/placement combinations give edge and face ranges exactly [0,1].
Independent adaptive chord integration over 30 placements and all three
face directions, 5,400 samples in total, differs by at most **9.41425e-12**
in dimensionless area. This is a measured finite sample, not a universal
error bound for every grazing cut.

Independent matrix indexing on a 7-cubed compact grid differs from the
shift implementation by **2.49801e-16**. The diagonal/cross split differs by
**2.84495e-16**. The indexed matrix is exactly symmetric and its minimum
eigenvalue is **.08191882**. Dense inversion independently checks the
rank-one formula within **1.77636e-15**; exact static line/face fields agree
within **1.22125e-15** in the initial audit. NumPy, eager JAX, compiled JAX
and Torch small-grid maps differ by at most **5.55112e-15**, and their
assembled actions by at most **4.21885e-15**. The tests additionally compare
30 shifted point clouds and all four execution modes at the independently
measured area-oracle scale.

A local-map radius derivative is nonzero: JAX gives **311.149506244259**,
Torch **311.149506244267**. A finite-difference step sweep finds topology
changes at the larger steps; at radius step **3e-7**, the slope is
**311.149502422540**, differing by **1.23e-8** relative. The test retains the
predeclared `1e-7*abs(g_fd)+1e-10*abs(observable)` comparison. This checks a
local radius derivative away from geometry events; a complete physical
adjoint and all geometric parameter derivatives remain unvalidated.

The glass sphere uses the unchanged physical fixture: epsilon 2.25,
radius/wavelength .25, 24 periods, 20/28/40 cells per wavelength, and all
three original placements. All nine rows finish with exit 0 and unchanged
measured source hashes:

| Cells per wavelength | Mean relative error | Placement RMS error | Placement range |
| ---: | ---: | ---: | ---: |
| 20 | 2.76349106% | 2.76355915% | .04399814 percentage points |
| 28 | 1.38223130% | 1.38223674% | .00895205 percentage points |
| 40 | .67292166% | .67292201% | .00162337 percentage points |

Mean/RMS fitted orders are **2.03778331/2.03781785**, with
**.472412%/.473014%** fit scatter. This meets the same finite-comparison
requirements, including separation from the 9.7e-5 instrument floor.
It does not improve every observable: the finest glass mean error is
**.67292%**, larger than the unchanged production operator's **.44610%**.
The candidate's motivation remains the cylinder improvement. A fit that
passes must not be described as uniformly smaller error.

The weak sphere repeats the same nine conditions with epsilon 1.1. Every
case exits 0; mean/RMS orders are **2.04780880/2.04801434** with
**.381547%/.375413%** scatter. Its errors also exceed the existing
production weak-sphere measurements:

| Cells per wavelength | Mean relative error | Placement RMS error | Placement range |
| ---: | ---: | ---: | ---: |
| 20 | 1.70753179% | 1.70785607% | .08074341 percentage points |
| 28 | .86224652% | .86227082% | .01474500 percentage points |
| 40 | .41301434% | .41303319% | .00950983 percentage points |

The weak forward optical-theorem residual at the origin is still
**51.8192% / 26.3226% / 12.9000%** at 20/28/40 cells. The unchanged
production measurements remain **52.0475% / 26.3798% / 12.9138%**. This
compares extinction inferred from the complex forward field with
`Qsca+Qabs`; it is not a claim that 13% of the time-domain energy is lost.
No underived wave-number substitution is applied to either result.

Across the glass/weak nine-case groups respectively, maximum apparent
absorption magnitudes are **1.17370e-6 / 1.22721e-6**, maximum late-monitor
energy fractions **1.56150e-11 / 5.22515e-9**, minimum local energy
eigenvalues **.4378716 / .9074754**, and maximum `dt^2*rate`
**.2559408 / .2505515**. These distinguish settled, certified calculations
from accurate scattering predictions; they do not waive the physical gates.

At 40 cells, glass peak-normalized angular errors reach **1.1638569%**,
exceeding the existing public physics test's **1%** limit. The origin's
optical closure is 1.234959%, the far-field/flux-surface gap 7.60333e-8,
and the angular integral gap 1.68941e-4. The side-scattering relative error
at theta=90 degrees remains below 1.368%. Thus it is specifically the
peak-normalized angular control, not all radiation controls, that fails
for this candidate. The limit stays unchanged.

The additional small high-index control retains radius/wavelength .1,
epsilon 12, 40 cells, origin placement and 48 periods. It exits 0 with
`Qsca=.399374827329122`, relative Mie error **3.7708391%**,
`Qabs=2.89558e-6`, and largest late-monitor energy fraction **1.57245e-11**.
Its scattering error exceeds the existing **3%** production control,
which measured 2.0576774%. This is another measured accuracy failure of
the candidate. The settled small-sphere run is not a time-convergence
certificate for the unmeasured radius-.25 high-index problem. No new long
864-period sweep has been started and no old high-index fit is credited
to this changed operator.

All twelve new research tests pass. Fourteen actual study-source defects
produce eighteen assertion-failure witnesses, including every parameterized
test case. They change the edge chord, face slice, rank-one sign, symmetry,
radial normal, orientation weight, cross increment, positive-energy check,
CFL check, vacuum halo, fraction bounds, background scope, radius derivative
and physical material inclusion. Every selected mutant exits 1, collection
and report counts agree, and there are no incidental errors or
`INTERNALERROR`. All source bytes are restored. These passes validate the
research instrument; they do not turn its two failed production controls
or weak forward-phase discrepancy into C3 passes.

The complete 19 physical rows, energy histories, source hashes, initial
geometry/operator measurements and mutation witnesses are retained in the
[3D research record](../studies/results/fdtd_subpixel_triplet_sphere.log).
The glass measurement and mutation audit precede an additive CLI radius
option; their exact source hashes are preserved. The material, geometry,
assembly and certificate arithmetic is unchanged in the weak and small
high-index runs. All measurement batches exit 0 with unchanged watched
source during execution. The production `wavebit/fd/` tree is unchanged.

Final targeted validation exits 0: **86 fast plus two slow cases**, with
matching collected/reported counts, no skips, errors, failures or
`INTERNALERROR`. This includes both research instruments and the document
contracts. The slow physical controls run sequentially. Fresh stage 0/1/2/3
reference checks each exit 0, reproducing **143/288/140/570 arrays bit for
bit** (1,141 total); no reference is regenerated. Measured study, test and
engine sources remain unchanged throughout those checks. The earlier full
production regression counts are not relabeled as a new full-suite run.

```bash
python studies/fdtd_subpixel_triplet_sphere.py glass-triplet.json
python studies/fdtd_subpixel_triplet_sphere.py weak-triplet.json --eps 1.1
python studies/fdtd_subpixel_triplet_sphere.py small-high-triplet.json --eps 12 --radius .1 --cells 40 --placements 0 --periods 48
```

### Short high-contrast mutation control

The radius-.25 high-index sweep is retained as the convergence measurement.
An additional epsilon-12 sphere of radius .1 supplies a shorter physical
control for arithmetic/harmonic source mutations. It uses the same public
scene, radiation surfaces and 40-cell vacuum resolution.

| Duration | Qsca | Relative Mie error | Apparent Qabs | Largest late-monitor energy fraction |
| ---: | ---: | ---: | ---: | ---: |
| 24 periods | .392783093650 | +2.0580878% | -9.20641e-6 | 2.12093e-9 |
| 48 periods | .392781513935 | +2.0576774% | 2.70344e-6 | 1.48130e-11 |

Both measurements exit 0 with unchanged source hashes. The relative Qsca
change is -4.02185e-6, compared with the 9.7e-5 open-scene floor. The test
uses 48 periods, rounds the measured Mie error upward to 3%, and rounds
twice the absorption residual upward to 6e-6. This is a material-rule
regression control, not a substitute for a fitted convergence order.
Its optical closure is 3.73589%; the main glass fixture remains the optical
phase gate. The full raw results are in the production-sphere log.

### Fresh public physics regression

`tests/test_fd_mie_physics.py` performs new public `fd.run` solves. The
recorded measurement log is not its oracle. Seven cases cover glass and
weak-scattering convergence (three resolutions and three placements each),
lossless absorption, independent radiation power, optical phase origin,
angular cuts and the short high-contrast material-rule control. Shared
in-process results are only cached to avoid repeating the same fresh solve.

The complete slow tier, including these seven cases, collects and reports
72 cases: **72 passed, zero skips, failures or errors**, exit 0, in 949.99 s.
No `INTERNALERROR` occurred, and the production/test source hashes were
unchanged during the run. This is a regression result, not a C3 pass; the
failed cylinder fit above remains blocking.

All eight actual source mutations below are now detected by assertions.
Their selected test collections and reports both total nine cases, with
nine assertion failures, no incidental errors, and exit 1 for each mutated
pytest invocation. The mutation driver itself exits 0. Every changed file
is restored byte for byte after its run. Source fragments, original hashes,
test names and assertion traces are retained in the production-sphere log.

| Actual source mutation | Observed assertion failure |
| --- | --- |
| Swap the diagonal arithmetic/harmonic weights | Short high-contrast sphere error 3.16600%, above the premeasured 3% limit |
| Freeze the sphere normal to z | Angular peak error 3.60681%, above 1% |
| Double the radiation amplitude gain | Angular-integral relative mismatch 3.000675, above 2e-4 |
| Read absorption on the outer scattered-field box | Apparent Qabs = -.8673944, instead of magnitude below 6e-7 |
| Refer the incident amplitude to the entry plane | Optical closure = -2.760918, instead of magnitude below .016 |
| Turn smoothing off | Glass mean order 1.73505; weak-scattering fit scatter 5.11196, above .1 despite apparent order 2.82345 |
| Fix diagonal normal weights to 1/3 | Glass mean order 1.65901, below 1.8 |
| Remove production cross coefficients | Glass mean order 1.71401, below 1.8 |

The last mutation reproduces the earlier diagonal-only failure through the
public production solver. All seven new test cases have a corresponding
actual source defect that fails their physical assertion. The diagonal
weight mutations change that part of the rule; the cross coefficients are
independently exercised by the final row.

The final fast tier also exits 0: 1,907 collected and reported cases,
**1,901 passed and six expected NumPy differentiation skips**, with no
failures, errors or `INTERNALERROR` (762.27 s). Stage 0/1/2/3 reference
reproduction each exits 0: **143/288/140/570 arrays reproduce bit for bit**,
1,141 arrays in total. No reference is regenerated. All production source
hashes remain unchanged through these checks.

### Weak-scattering forward phase

The optical theorem remains sensitive even when Qsca converges. At the
origin, its relative difference from Qsca+Qabs is:

| Epsilon | 20 cells | 28 cells | 40 cells |
| ---: | ---: | ---: | ---: |
| 2.25 | 4.92250% | 2.51814% | 1.23656% |
| 1.1 | 52.0475% | 26.3798% | 12.9138% |

For epsilon 1.1, Mie's normalized complex forward field is
`.020388555018 + .000731186485 i`. Its magnitude divided by its imaginary
part is 27.90, so a small phase defect strongly affects the extinction
inferred from that imaginary part. The study now retains both quadratures
and the ratio of grid incident flux to `|E0|**2/2`; the latter includes the
normal-incidence Yee H-plane average. Intensity alone loses the phase sign.

A separate 20-cell run moves only the radiation surfaces. The direct complex
field measurements, normalized at the same incident phase origin, are:

| Radiation half-width | Forward phase error | Complex-field relative error | Phase error after diagnostic k substitution |
| ---: | ---: | ---: | ---: |
| .65 | .74519 degrees | 1.58826% | -.35961 degrees |
| .75 | 1.06432 degrees | 2.20620% | -.20916 degrees |
| .85 | 1.46300 degrees | 2.72914% | .05130 degrees |

The substitution uses the exact normal-incidence Yee wavenumber
`2/dx*asin(dx/dt*sin(pi*dt)) = 6.30713500398`, instead of `2*pi`, in the
continuum radiation transform. Its remaining surface dependence shows that
changing that number alone is insufficient. The production transform stays
unchanged; a discrete transform has not been derived by this experiment.
This measured weak-scattering limitation must accompany any optical-theorem
claim. The glass closure meets its stated 1.6% criterion at 40 cells.

The manufactured Rayleigh instrument additionally checks complex phase:
the finite-size amplitude discrepancy from Mie is 2.71406e-7. An actual source
mutation `f -> -conj(f)` preserves Im(f) and intensity but fails the new
phase assertion. A second mutation omitting the intensity factor 1/2 fails
the flux-ratio assertion. Both exit 1 with one assertion failure, no errors,
and are restored byte for byte; the six healthy instrument cases pass.

```bash
python studies/fdtd_subpixel_mie_sphere.py --eps 12 --cpw 40 --periods 48 --energy tail48.json
python studies/fdtd_subpixel_mie_sphere.py --eps 12 --cpw 40 --periods 288 --energy tail288.json
python studies/fdtd_subpixel_mie_sphere.py --eps 12 --cpw 40 --periods 456 --energy tail456.json
python studies/fdtd_subpixel_mie_sphere.py --eps 12 --cpw 40 --periods 864 --energy tail864.json
python studies/fdtd_subpixel_mie_sphere.py --eps 12 --cpw 56 --periods 864 --energy high56.json
python studies/fdtd_subpixel_mie_sphere.py --eps 12 --cpw 80 --periods 864 --energy high80.json
python studies/fdtd_subpixel_mie_sphere.py --eps 1.1 --cpw 20 --phase-control phase20.json
```

## Exact Mie near fields through the radiation transform

### Question and independent reference

Does the large weak-scattering optical residual persist when the production
transform receives exact continuum near fields instead of a computed field?
`studies/fdtd_subpixel_mie_transform.py` measures that question without a time
step, material lowering, incident-source solve, or fitted wave number.

The experimental separation follows the approach described in
[Zhai et al. (2004)](https://doi.org/10.1364/AO.43.003738).
The exterior component series is implemented independently from equations
2.40--2.45, printed page 22 of
[Zhai's public 2006 dissertation](https://oaktrust.library.tamu.edu/server/api/core/bitstreams/4aa05d10-5f44-4037-ab10-7a42dd8154d8/content).
The accessible text was checked against Maxwell identities; the PDF image
fetch was unavailable. No external solver implementation was read.

`tests/references/mie_fields.py` returns Cartesian scattered E and Z0 H for
unit x-polarized incidence along +z, vacuum wavelength one, and a nonmagnetic
dielectric sphere. It reuses the separately validated Mie coefficients, with
outgoing spherical Hankel functions. The recurrence for angular functions
does not divide by sin(theta), including the forward and backward axes.
Moving the sphere multiplies the local incident field by exp(i k center_z);
the corresponding far field also has the outgoing translation phase.

The usual far-field series truncation is insufficient very near the sphere:
at radii .250001/.3/.625/1.5, epsilon 1.1/2.25/12 and 100 directions,
adding 0/2/4/8 terms differs from a +16-term control by at most
2.58603e-7/1.39518e-9/4.65756e-12/7.14100e-17 of peak field. Eight extra
terms are therefore used, with that provenance beside the constant. This
measurement covers radius .25 and these dielectric contrasts; it is not a
general near-field error bound for every size or complex refractive index.

Independent controls establish that the injected fields themselves are sound:

| Control | Measured result |
| --- | --- |
| Both vacuum Maxwell curls and divergences, Cartesian fourth-order differences | Maximum normalized residual 1.46423e-12 at difference step 1e-4; the recorded sweep covers 1e-2 to 1e-5 |
| Outward E cross H power through spheres of radius .3/.65/1.1 | Relative difference from Mie scattering at most 4.56e-15; angular quadrature orders 16/32/64 agree |
| Far-field limit, extrapolating radii 16/32/64/128 in 1/r | Complex peak discrepancy at most 2.20e-10, consistent with the remaining fourth-order term |
| Independent Cartesian surface integral on a cube, with Gaussian quadrature | At most 7.49e-15 at order 24 and 1.02e-14 at order 48, versus 7.25e-7 at order 12 |
| Translation and scalar/vectorized Yee sample checks | Maximum absolute discrepancies 7.85e-17 and 2.80e-17 |
| NumPy, JAX eager, JAX compiled, Torch radiation outputs at three placements | Maximum absolute backend difference 1.073e-17 |

The independent surface integral uses collocated vectors, explicit Cartesian
cross products and Gaussian weights. It does not call the production layout,
half-plane average, cyclic component assembly or surface-transform helper.
It recovers both complex polarizations at forward, backward and oblique angles,
including a translated sphere. Agreement of only intensity would not suffice.

### The 162 measurements

The raw record is
[`fdtd_subpixel_mie_transform.log`](https://github.com/fieldalgo/wavebit/blob/fdtd-general/studies/results/fdtd_subpixel_mie_transform.log).
All 162 cases complete with exit 0 and unchanged production/study/reference
hashes in 207.07 s of summed measurement time. Inputs are epsilon 1.1/2.25/12,
20/28/40 cells per wavelength, the original three sphere placements, radiation
half-widths .65/.75/.85, and two sampling arms. The record retains actual snapped
face coordinates; at 28 cells a requested .65 or .85 does not land exactly on
a node. No time convergence claim is involved in this diagnostic.

The first arm samples E and H at every actual Yee location read by the existing
monitor and passes the arrays through the unchanged production transform.
The second replaces each H pair by the exact H on its E plane. This is a
diagnostic with information unavailable in a normal run, not an interpolation
fix or a proposed production algorithm.

At the origin and standard half-width .75, weak-scattering results are:

| Cells per wavelength | Exact-input optical closure | Existing production optical closure | Exact-input angular intensity peak error |
| ---: | ---: | ---: | ---: |
| 20 | 1.84746% | 52.0475% | .91228% |
| 28 | .93701% | 26.3798% | .46638% |
| 40 | .45769% | 12.9138% | .22877% |

Here exact-input closure is Qext inferred from Im(forward field), divided by
the surface-integrated Qsca, minus one. Production closure uses its separately
computed Qsca+Qabs. These ratios have different finite-grid denominators and
must not be subtracted as an exact additive error budget. Neither residual is
a fraction of energy spuriously absorbed by the sphere.

At 40 cells, the exact-input forward field is
`.020375906578 + .000733015647 i`, versus exact far field
`.020388555018 + .000731186485 i`: .062642% complex relative error and
.00641073 degrees of phase error. The inferred Qext is .250164% above exact
Mie, while surface power is .206583% below it. With exact collocated H,
forward complex error falls to .003079%, phase to -.00159908 degrees, and
closure to -.0730533%. This measures the effect of H averaging in this arm;
it does not make computed Yee fields satisfy continuum Maxwell equations.

Surface dependence is still material for phase-sensitive weak scattering:

| Half-width, 40 cells, origin | Staggered-input phase error | Staggered optical closure | Exact-collocated-H closure |
| ---: | ---: | ---: | ---: |
| .65 | -.0216519 degrees | -.98198% | -.34090% |
| .75 | .0064107 degrees | .45769% | -.07305% |
| .85 | .0600854 degrees | 3.02685% | .45612% |

The two displaced spheres also matter: at standard half-width and 40 cells,
weak closure ranges from .45769% to .69496%. Mean angular peak errors over all
three placements, for 20/28/40 cells, are .91122/.46610/.22869% (epsilon 1.1),
.81947/.41598/.20315% (2.25), and 2.30130/1.18763/.58598% (12). Thus there is
measurable radiation sampling error even with exact input. Choosing a surface
where one error happens to cancel is not an accuracy certificate.

### What this establishes, and what remains

For a fixed linear radiation map L_h and incident-normalized computed field
F_h, the complex amplitude error splits exactly as

    L_h(F_h) - f_exact = [L_h(F_exact) - f_exact] + L_h(F_h - F_exact).

The new experiment measures the first bracket. The existing weak-scattering
standard-surface residual is much larger. Further investigation must therefore
include the computed field and its incident phase reference. The second term
includes material discretization, finite-grid propagation, source normalization,
and their interaction with the continuum radiation map. This experiment does
not assign that term to one update equation, prove a source bug, or derive a
discrete Green function. A wave-number substitution remains unjustified.

The next discriminating control is complex incident/reference propagation and
sampled scattered E/H on the same surfaces, separating vacuum propagation from
material-boundary error. Repeating a long high-index spatial sweep would not
answer that question. The existing inclined-interface consistency failure and
the cylinder/candidate accuracy failures remain open. C3 has not passed, and
the production solver is unchanged.

### Regression witnesses and reproduction

The eleven new cases pass. Eleven actual source defects produce fourteen
assertion failures: wrong electric azimuth sign, inadequate near-field series,
doubled reference amplitude, omitted incident translation phase, reversed
magnetic surface-current sign, omitted H staggering, doubled production
radiation gain (all four execution modes), halved power normalization, and
omitted surface/point/finite-input refusals. Every mutation exits 1 with matching
collection/report counts, no incidental errors or INTERNALERROR. The driver
exits 0 and restores all three touched source files byte for byte.

Final targeted verification (the Mie coefficient, measurement and exact-field
tests plus the documentation contract) exits 0 with **101 collected/reported
passes**, no skips, failures, errors or INTERNALERROR. Stage 0/1/2/3 reference
checks each exit 0: **143/288/140/570 arrays reproduce bit for bit**, 1,141 in
total. Source hashes remain unchanged throughout. The complete engine fast
and slow tiers are not rerun for this research-only addition; earlier full
suite counts above are historical, not results from this diagnostic.

```bash
python studies/fdtd_subpixel_mie_transform.py exact-transform.json
python -m pytest tests/test_fd_mie_transform.py -p no:cacheprovider -q
```

## Incident phase and discrete interference controls

This follows the exact-field injection experiment above. The production solver,
source normalization, radiation transform and C3 acceptance criteria are unchanged.
The new study is `studies/fdtd_subpixel_phase_control.py`; its record is
`studies/results/fdtd_subpixel_phase_control.log`. All five paired measurements
complete with actual exit 0 and unchanged source hashes during each measurement.
Each pair contains an empty scene and the public epsilon-1.1, radius-.25 sphere.
The original source, three diagnostic half-widths .65/.75/.85 and Courant .5 are
retained. The new central Ex/Hy line remains inside the total-field source box.

### Independent phase reference

For unit vacuum wave speed, angular frequency omega, time step dt and grid step h,
the normal-incidence Yee update equations give

    Omega = 2 sin(omega dt/2) / dt,
    2 sin(kappa h/2) / h = Omega.

The study transforms the raw source E and H series on their respective clocks,
including the finite-record endpoint term. It independently separates forward
and backward waves using their actual plane coordinates, then carries the forward
amplitude to z=0 with kappa. It does not call the production DFT or direction-split
helper. The resulting incident amplitude agrees with the public API within
2.159e-15 relative over the five pairs. Synthetic records with known complex modes,
nonzero backward waves and nonzero final samples also recover amplitude and power
on NumPy, JAX eager, JAX compiled and Torch: maximum absolute discrepancies
1.692e-15 and 4.441e-16 in the preliminary three-grid probe.

The independently derived unit-forward vacuum fields are

    E_x(z) = exp(i kappa z) + B exp(-i kappa z),
    H_y(z) = exp(i kappa z) - B exp(-i kappa z).

B is measured from the raw incident column, not fitted to the 3D monitor. At each
actual Yee location these fields match the empty 3D line as follows. All entries
below are relative to the incident forward amplitude; the continuum comparison
uses exp(i 2 pi z) without a backward wave.

| Cells/wavelength, 24 periods | kappa | Measured backward amplitude | Max two-wave field discrepancy | Max continuum forward-wave discrepancy |
| ---: | ---: | ---: | ---: | ---: |
| 20 | 6.307135004 | 9.8984e-5 | 3.888e-7 | .0108523 |
| 28 | 6.295337993 | 4.9008e-5 | 2.345e-7 | .00546187 |
| 40 | 6.289122988 | 2.8488e-5 | 2.064e-7 | .00265580 |

The measured intensity divided by |A0|^2/2 also equals cos(kappa h/2), with
maximum absolute discrepancy 4.2e-15. This cosine follows from averaging the two
H half-planes onto an E plane. The empty field outside the total-field source box
is below 8.65e-17 of incident amplitude. These controls do not support a misplaced
incident reference phase or appreciable source leakage in this fixture.

### Extinction from the discrete surface power

In continuum electromagnetism, expanding the Poynting product of incident plus
scattered fields separates incident power, scattering and their interference;
see [Striebel et al., Scientific Reports 7, 15420 (2017), section "Absorption,
Scattering and Extinction"](https://pmc.ncbi.nlm.nih.gov/articles/PMC5684246/).
That paper supplies the physical interpretation, not a discrete-grid theorem.
For the actual wavebit monitor define its real bilinear flux functional

    Phi_h(E,H) = (1/2) Re sum_faces sum_samples
                 [E cross conjugate(H_mean)] dot n * weight * h^2.

Use the existing staggered sample positions, the same two-plane H average and
the same face weights in every term. Direct expansion then gives the exact
algebraic identity

    Phi_h(Ei+Es, Hi+Hs) = Phi_h(Ei,Hi) + Phi_h(Es,Hs)
                          + Phi_h(Es,Hi) + Phi_h(Ei,Hs).

Define Pext_h as the negative sum of the last two terms. For a settled solution
with zero net incident power, Pext_h should equal scattered plus absorbed power.
Equality additionally needs consistent physical fields and a source-free vacuum
region between the measuring surfaces; the algebra alone cannot establish those
conditions. Here absorption is independently read from the inner total-field
surface. The outer scattered field is paired with a separate empty run, and each
run is normalized by its own incident amplitude. The large incident power is
never numerically subtracted from total power to form the interference terms.

An independent analytic check on each z face uses arbitrary counterpropagating
mode amplitudes Ai/Bi and As/Bs. The exact mixed power is

    sign * area * cos(kappa h/2) * Re(As conjugate(Ai) - Bs conjugate(Bi)).

The diagnostic agrees within 5.90e-17 absolute. This checks both interference
terms, the backward magnetic sign, staggering, normalization and orientation.
Replacing kappa by 2 pi changes the incident field to one that does not satisfy
the discrete update. It is a deliberately inconsistent control, not a candidate
correction. None of these experiments changes k in the production far-field map.

### Measured closures and controls

Interference closure is Pext_h/(Psca_h+Pabs_h)-1 on each diagnostic surface.
The standard surface has requested half-width .75. The first column retains the
original forward-amplitude optical-theorem observable and its original denominator;
the other columns measure surface interference. They are different finite-grid
observables and are not interchangeable error budgets.

| Cells/wavelength, 24 periods | Original optical closure | Continuum incident interference, .75 | Grid two-wave interference, .75 | Largest absolute grid two-wave closure across three surfaces |
| ---: | ---: | ---: | ---: | ---: |
| 20 | 52.04750% | 62.57161% | .0144693% | .0166438% |
| 28 | 26.37983% | 31.69403% | -.0036056% | .0059680% |
| 40 | 12.91384% | 15.50382% | -.0053926% | .0053926% |

Two additional paired controls bound only the changes actually measured:

* At 20 cells, extending 24 to 48 periods changes the original optical closure
  from 52.04750% to 52.01519%. The three grid two-wave closures become
  -.0079563/-.0053396/-.0078311%. Their largest change is .0246001 percentage
  points. The largest late monitor energy ratio falls from 5.202e-9 to 6.651e-10.
  Duration doubling is an experiment, not a certificate of the infinite-time
  value. The remaining small surface dependence and finite-record uncertainty
  must not be described as roundoff or exact energy closure.
* At 20 cells and 24 periods, lowering the PML reflection *target* from 1e-8 to
  1e-12 increases the measured backward amplitude from 9.8984e-5 to 3.1827e-4.
  A smaller target is not evidence of less reflection on a fixed discrete layer.
  The original optical closure stays at 51.96115%; the largest absolute grid
  two-wave closure is .0162657%. The measured backward wave must be retained.

Direct comparison of scattered E with exact Mie E adds spatial information.
At 20 cells, its descriptive complex least-squares gain has phase
.59289/.68650/.77859 degrees on the three surfaces; at 40 cells this becomes
.14758/.17002/.19266 degrees. The corresponding relative field differences at
40 cells are .33075/.36296/.39694%. These gains are never applied to a field or
used to improve a reported result. The imaginary part of the forward amplitude
is small for this weak sphere, so small complex-field phase errors can produce
large relative optical-theorem residuals.

The evidence locates an important inconsistency in comparing computed grid
fields with the continuum forward-scattering identity: grid propagation and
interference give a nearly closed energy balance while the existing continuum
observable remains far from closed. It does not isolate the entire error to a
single radiation-code line, derive a lattice Green function or validate a new
far-field amplitude. It also does not remove material-interface error. The
inclined-interface consistency failure, production cylinder refinement failure,
and two failed 3D research-candidate accuracy controls remain. C3 has not passed.

### Instrument regression and reproduction

The test budgets were selected after the measurements: 2e-14 for the analytic
source-mode check (measured at most 1.70e-15), 1e-15 for analytic mixed power
(5.90e-17), 1e-6 for vacuum two-wave fields (3.89e-7), 2e-15 for empty leakage
(8.65e-17), and 5e-4 for interference closure (twice the measured 2.46e-4 temporal
change). These are diagnostic regression budgets. The 5e-4 value is not a new
C3 gate and is not a bound on continuum scattering accuracy.

Validation results are recorded with collection/report counts, actual process
exits, mutation details and source restoration in the accompanying measurement log.
The six new cases pass. Nine actual source defects cause 25 assertion failures:
wrong reference-carrier sign and H clock in production normalization; omitted
endpoint correction, continuum carrier and wrong backward sign in the independent
source audit; an omitted interference term; a wrong backward H sign; a continuum
incident carrier on the diagnostic surfaces; and a wrong vacuum-line H sign.
The surface-carrier defect fails both analytic power and fresh physical energy
checks. Every selected mutant case exits 1 by assertion with matching collection
and report counts, no incidental errors or INTERNALERROR. The mutation driver
exits 0 and restores both touched source files byte for byte.

Final targeted checks exit 0 with **106 fast cases and one fresh physical case**,
all collected/reported counts matching, no skips, failures, errors or INTERNALERROR.
This research addition does not rerun the complete engine fast/slow tiers; the
earlier full-suite counts in this document remain historical.
Stage 0/1/2/3 reference checks each exit 0 with **143/288/140/570 arrays reproduced
bit for bit**, 1,141 in total. All monitored source hashes are unchanged throughout
the final verification. No baseline is rewritten.

```bash
python studies/fdtd_subpixel_phase_control.py phase20.json --cpw 20
python studies/fdtd_subpixel_phase_control.py phase28.json --cpw 28
python studies/fdtd_subpixel_phase_control.py phase40.json --cpw 40
python studies/fdtd_subpixel_phase_control.py phase20-target.json --cpw 20 --reflection 1e-12
python studies/fdtd_subpixel_phase_control.py phase20-time.json --cpw 20 --periods 48
python -m pytest tests/test_fd_phase_control.py -p no:cacheprovider --run-slow -q
```

Use new output names: the study refuses to overwrite an existing JSON/array pair.
The JSON retains source hashes, warnings, complex amplitudes, line fields, three
surface observables and residual energies; a companion compressed array file
preserves raw source records and sampled monitor arrays with a SHA256 digest.

## Derived lattice radiation

The [Yee radiation derivation and independent validation](fd-lattice-radiation.md)
now supplies the missing complex far-field map. It includes the reciprocal
reference wave's H-plane average, the outgoing dispersion curvature and the
full Cartesian flux weight. Independent Fourier Green and electric/magnetic
point-current controls validate its amplitude and phase.

On the unchanged weak-sphere records, central optical residuals become
.0264028%/-.0433575%/-.00357156% at 20/28/40 cells. Historical continuum
residuals remain 52.04750%/26.37983%/12.91384%. A fresh public 20-cell run
with `FarField(propagation='yee')` gives .0267422%. No material coefficient,
recorded phase or acceptance gate is fitted to obtain these values.

This resolves the identified radiation inconsistency. The production cylinder
refinement failure and the small high-index error of the research 3D material
candidate remain. C3 has not passed. The new radiation API preserves the
original continuum path and its frozen reference arrays.

## Symmetric extended interface stencil experiment

The [extended-stencil derivation and controls](fd-symmetric-stencil.md)
address the remaining material-interface defect. A symmetric matrix is
constructed from exact edge/dual-face interface constraints, with a
two-cell neighbour stencil and a measured whole-operator energy bound.
It remains a research study rather than the public `fd.run` material map.

With the same operator on every case, the historical cylinder's full
40/56/80 by eight-placement study now has mean/RMS orders
**2.137819/2.137121**, scatter **2.128691%/2.093839%**, and finest mean
error **0.740709%**. All 24 calculations exited normally as one finite
batch (exit zero), with unchanged numerical sources. The small epsilon-12
sphere control has error **2.468123%**, below its unchanged 3% limit.

The cylinder control passes. C3 still requires new glass, weak and large
high-index sphere measurements with this same operator, followed by the
public preparation/integration review. The earlier public sphere fits
belong to a different operator and cannot be combined with this cylinder
result to claim a C3 pass. C4 remains pending.
