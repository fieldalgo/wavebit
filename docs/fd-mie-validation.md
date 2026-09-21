# Mie reference and the curved-interface accuracy gate

Written: 2026-09-22

Targets: `tests/references/mie.py`, `studies/fdtd_subpixel_mie_cylinder.py`,
`tests/test_fd_mie_reference.py`, `tests/test_fd_mie_study.py`,
`tests/test_fd_monitor_structure.py`, `wavebit/fd/_lower/monitors.py`,
`wavebit/fd/monitors.py`, `wavebit/fd/diagnostics.py`, `wavebit/fd/_lower/checks.py`,
`tests/test_fd_occupancy_integration.py`, `tests/test_fd_decay.py`.

## TL;DR

The analytic reference and the closed-contour instrument have independent checks.
The current diagonal material rule gives near-second-order fits in the cylinder's axial-electric warm-up,
but the high-contrast axial-magnetic case does **not** establish second-order
convergence. C3 is not complete. Exact geometric volume alone does not establish
electromagnetic accuracy; a successful regression suite must not be reported as
a successful sphere convergence gate.

This study is run before the full three-dimensional sphere sweep. A measured
failure is retained with its placement spread and fit residual. No tolerance is
chosen to relabel it as a second-order result.

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

## Integrated validation

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
