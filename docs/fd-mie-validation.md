# Mie reference and the curved-interface accuracy gate

Written: 2026-09-22

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
`wavebit/fd/_kernel/observe.py`, `tests/test_fd_constitutive.py`.

## TL;DR

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

The next trial is 288 periods. Its provenance is
`48 + 29.81*log(.0027553423/1e-6)`, rounded up to a multiple of the original
24-period window. Here 1e-6 is one tenth of the existing 1e-5 unsettled-energy
line. This extrapolation chooses an experiment; its actual decay, Qabs and
time-extension change must be measured before selecting a production test
duration or accepting a high-index order.

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
python studies/fdtd_subpixel_mie_sphere.py --eps 1.1 --cpw 20 --phase-control phase20.json
```
