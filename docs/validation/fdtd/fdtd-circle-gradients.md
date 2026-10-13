# Circular FDTD shapes and diameter derivatives

Written: 2026-09-09

Targets:
- `src/wavebit/fdtd/_periodic/scene.py`
- `src/wavebit/compat/legacy.py`
- `src/wavebit/manifest.py`
- `tests/fdtd/test_fdtd_circles.py`
- `tests/fdtd/test_fdtd_circle_optics.py`
- `studies/cross_solver/circle_gradients/fdtd_circle_gradients.py`
- `tests/references/interface/meep_rotation.py`
- `tests/references/test_meep_circles.py`

TL;DR: the common area route now accepts shared `circle`/`disk` recipes on
NumPy, JAX and Torch. Centre, diameter and scalar material derivatives remain
live on a fixed mesh. Sharp-cell areas are analytic. Translation averaging
uses stable local arc integration, including very narrow kernels. Independent
chord integration checks geometry; Bessel-series RCWA and public-API Meep runs
check a matching graded optical target. Exact cell geometry does not replace
full effective-tensor averaging or establish general sharp-interface gradient
accuracy. GPU and general second derivatives are outside this CPU validation.

## Shared geometry and analytic sharp cells

The [usage contract](../../guide/fdtd/time-domain.md#circular-shape-derivatives) preserves
the common shape reader: `lx` or `l` is the diameter, centres are relative to the
period centre, and a circle has no `ly`. A finite angle has no geometric effect.
Diameters fit inside both rectangular periods. One area shape per layer keeps
periodic images disjoint; different layers can contain different shape types.

For radius `r`, integrating the vertical chord `sqrt(r*r-x*x)` gives the
circular-segment primitive. Signed quadrant integrals at four box corners give
the disk/cell area without a polygon or intermediate raster. The signed branch
at zero retains centre derivatives when a cell face passes through the centre.
Near `x=r`, the elementary arcsine primitive subtracts two divergent slopes.
An endpoint series for the cap integral removes that cancellation. Adjacent
sharp cells also reuse identical stored faces: otherwise independent rounding
gaps can produce chord-slope changes proportional to the square root of the gap.
First derivatives remain finite at tangency; second derivatives need not.

Geometry tests integrate physical vertical chords independently with adaptive
one-dimensional quadrature. They cover periodic seams, staggered sampling,
maximum diameter, axes, tangencies, exact total area `pi*d*d/4`, and its gradient
`(0, 0, pi*d/2)`. JAX/Torch cell Jacobians also agree with two independent chord
finite-difference steps. Full bottom-incidence conical Jones solves check joint
centre, diameter, epsilon, mu and layer-thickness derivatives while mesh, source,
time step and extraction coordinates stay fixed. Invalid traced shapes produce
NaN scattering and `stable=False` without retracing.

## Stable translation averaging

The same static physical `(x,y)` box widths apply to circles and rectangles.
A cell indicator convolved with that box is piecewise bilinear. A first
candidate integrated four global disk moments, then converted their differences
to local box coordinates. That is algebraically correct but numerically poor:
at width `1e-9` on a period-one cell, cancellation changed some fractions by
nearly one. It is not the shipped positive-width implementation.

The shipped method splits the disk into quadrant strips. Flat strips integrate
exactly. On a curved strip, `x=r*sin(theta)` turns the integrand into a
trigonometric polynomial of degree at most four, over a span no larger than
`pi/2`. A fixed 16-point Gauss rule evaluates that smooth angular integral.
Weights are evaluated in the local window, avoiding differences of large global
moments divided by tiny widths. This is numerical arc integration, distinct
from the exact polygon moments used by rectangles. There is no user quadrature
setting, shape raster or polygonal circle boundary.

A positive knot interval can round to zero after adding its cell centre.
Such a window has zero measure and uses safe inactive denominators. Regression
cases include width `1e-9`, widths immediately above/below a cell width, an exact
cell width with rounded-away middle pieces, and full-period widths. The narrow
kernel approaches sharp fractions to about `2.2e-15` in the recorded period-one
probe; the period-integrated area differs by at most `2.8e-17`. Full-period
averaging gives the constant disk fill and its exact area derivative on all
three backends.

## Independent optical target

The graded benchmark has period `(0.6,0.6)`, a thickness-0.2 layer containing a
circle of diameter 0.3 at centre `(0.071,-0.047)`, background epsilon 1, circle
epsilon 2.25 and substrate epsilon 1.21. All mu values are one. Incidence is
normal, wavelength 1; all eight complex s/p reflection/transmission channels
and their diameter derivatives are compared at the physical layer faces.
Smoothing width 0.05 is held fixed in both transverse directions.

The reference uses the independent radial Fourier transform
`2*pi*r*J1(k*r)/k`, with its DC limit, multiplied by translation-box sinc
factors. Its diameter derivative is `pi*r*J0(k*r)`. These follow from the
[Bessel integral identity](https://dlmf.nist.gov/10.22.E1). A cell-centred inverse
Fourier series supplies a scalar graded RCWA raster; clipping its small ringing
to `[0,1]` also permits the same profile in Meep MaterialGrid. Profile-grid and
perturbation controls include this clipping approximation. FDTD geometry
functions are never used to construct the reference.

The primary reference uses profile grid 256 and RCWA order `(8,8)`, with
centred differences `1e-5`. Doubling the grid changes amplitudes by `6.01e-7`
and the diameter derivative by `3.12e-4`. Raising the order to `(10,10)` changes
them by `1.37e-6` and `4.03e-5`; halving the difference step changes the derivative
by `3.06e-7`. The profile derivative is therefore the limiting reference control.

Native FDTD uses physical PML thickness 1, pulse width 2.5, duration 60 optical
periods, aligned layer boundaries and directional extraction. Resolution is in
cells per wavelength; the solver's static mesh rule determines the transverse
counts. All figures use CPU float64.

| Solver / resolution | Max complex amplitude error | Max complex diameter-derivative error |
| --- | ---: | ---: |
| Wavebit / 20 | 0.008602 | 0.02352 |
| Wavebit / 30 | 0.003874 | 0.01453 |
| Wavebit / 40 | 0.002220 | 0.006521 |
| Meep / 20 | 0.02239 | 0.2208 |
| Meep / 40 | 0.008994 | 0.1147 |

These are absolute errors over all eight complex channels against the same
profile-grid-256 RCWA reference. Native energy residuals are 0.002656, 0.001163
and 0.000651 at the three resolutions. Extending native duration 60 to 80
changes amplitudes by `1.77e-11` and diameter derivatives by `3.94e-10` at
resolution 40. The sharp circle differs from the graded reference by 0.005933
in amplitude and 0.02000 in the diameter derivative (sharp RCWA order 10,
normal grid 512). The graded comparison therefore does not validate the sharp
circle's continuum derivative.

Meep 1.34.0 meets the preselected runtime, perturbation and profile controls:
maximum duration changes are `1.96e-11` for amplitudes and `4.04e-10` for
derivatives; the largest difference-step change is `1.25e-4`. Doubling its
profile grid at resolution 20 changes amplitudes by `1.07e-5` and derivatives
by `0.004527`. Profile placement probes differ by at most `8.89e-16`.

The Meep diameter error still reaches **0.115** at resolution 40 despite these
controls. Its improvement over resolution 20 establishes a refinement trend,
not a precise continuum-gradient oracle or a known asymptotic convergence
order. No tolerance was relaxed to erase that discrepancy. The smaller native
errors above are established against independent Fourier optics. They do not
establish blanket agreement with Meep or identify the cause of its residual.


## External solver boundary and reproducibility

Meep is executed as a separate reference tool. Only its
[documented public API](https://meep.readthedocs.io/en/latest/Python_User_Interface/)
is consulted; its implementation is never opened, copied or imported as a
Wavebit dependency. The existing reference bridge now also takes `--shape circle`.
Sharp geometry uses public `Cylinder`; positive smoothing uses the independently
constructed Fourier MaterialGrid with internal averaging disabled. The recorded
circle goldens below use the graded MaterialGrid target, not a sharp cylinder.
Diameter sensitivities are central finite differences, not Meep adjoints.

The reference domain is `(0.6,0.6,8)`, periodic transversely, with z PML thickness
1 and Courant factor 0.4. A sine-Gaussian pulse at z=-2 has width 2.5, peak 12.5
and end time 25. Every-step DFT monitors at z=-1 and 1.2 use public integration
weights. Empty-cell calibration removes the pulse; discrete propagation phases
refer the amplitudes to the physical faces. Both incident polarizations are run.
Two durations (80/100), two diameter steps (`1e-4`/`5e-5`), two resolutions (20/40)
and an independent profile refinement (256/512 at resolution 20) are retained.
Public epsilon-grid probes check profile placement before scattering is recorded.

Run the studies with `python -m studies.fdtd_circle_gradients --output circle.json`
and add `--smoothing 0` to study the sharp target separately. Regenerate external
candidates only in an environment providing Meep, NumPy and SciPy, using
`python tests/references/interface/meep_rotation.py --shape circle --smoothing .05
--resolution 20 --output candidate.npz`; use resolution 40 for the fine grid and
`--profile-grid 512` for the independent profile control. A candidate must pass
its controls before replacing a golden; regeneration is not a response to a
failing assertion. The guide contains an executable JAX diameter example.
