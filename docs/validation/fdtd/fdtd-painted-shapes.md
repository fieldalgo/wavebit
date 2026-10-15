# Periodic painting of multiple rotated FDTD rectangles

Written: 2026-09-09

Targets:
- `src/wavebit/fdtd/_periodic/scene.py`
- `src/wavebit/fdtd/_periodic/api.py`
- `src/wavebit/manifest.py`
- `tests/fdtd/test_fdtd_painted_shapes.py`
- `tests/fdtd/test_fdtd_painted_optics.py`
- `studies/cross_solver/painted_shapes/fdtd_painted_shapes.py`
- `tests/references/interface/meep_rotation.py`
- `tests/references/test_meep_painted_shapes.py`

TL;DR: the common area route accepts multiple rotated rectangles/squares in a
layer with later-shape priority, periodic overlaps and live shape/material
parameters on NumPy/JAX/Torch. A boundary-event representation permits changing
overlap topology without a frozen host plan. Independent Cartesian integration
checks geometry and an inclusion-exclusion Fourier profile checks graded
optics. This is not mixed curved-shape painting, a general interface-effective
tensor, unique derivatives at contacts or a large-layout performance guarantee.

## The material jump representation

The [guide](../../guide/fdtd/time-domain.md#multiple-rotated-rectangles) uses the existing
shape list and shared RCWA shape reader and rectangle vertices. This extension
does not change the default separable route or the single-shape polygon/disk/
ellipse integrators. Multiple area shapes currently require rectangles/squares.
Every primitive's projected x/y extents must fit the rectangular periods.

Simply summing single-shape fill fractions counts overlaps twice. Instead, each
oriented rectangle edge is split where it intersects other periodic rectangles.
Half-plane segment intersections and sorting keep the number of stored events
static; all event coordinates remain in the array graph. Infinitesimal left and
right membership selects the last-painted epsilon and mu on either side. Their
difference is the material jump on that segment. A hidden boundary has zero
jump. Coincident boundaries belong to the last shape that owns them.

Physical edge pieces are folded into the fundamental cell. The cell-border
loop is also split, and carries the inside material minus the background.
This loop is essential: a fully filled periodic cell has no physical material
boundary but still has a nonzero constant material contrast. Omitting the cell
loop would lose that mean. Physical edges coincident with a cell seam are
replaced by the cell-border representation to avoid counting them twice.

Signed distances and parallel slopes within 32 machine-epsilon units times the
maximum period use their limiting half-plane classification. This addresses
roundoff at shared boundaries, not geometrically finite overlap. Exact contacts
and coincident boundaries retain only piecewise first derivatives; no unique
slope or general second derivative is claimed there. A positive translation
width averages the painted pattern as a whole, not the relative contact motion
of its shapes.

## Cell and translation integrals

[Green's theorem](https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/pages/3.-double-integrals-and-line-integrals-in-the-plane/part-c-greens-theorem/)
converts a two-dimensional weighted area into an oriented boundary integral.
For a separable cell kernel `w_x(x)*w_y(y)`, choose
`Q(x,y)=CDF_x(x)*w_y(y)`, with `d(CDF_x)/dx=w_x`. The material correction is
`sum integral(jump * Q dy)` over the split boundary segments, divided by the
cell area. Add the scalar background to recover each epsilon/mu coefficient.

Translation windows are piecewise linear. Splitting an edge at all x and y
window knots makes `CDF_x*w_y` a polynomial of degree at most three along that
edge. Two Gaussian nodes therefore integrate each piece exactly in arithmetic.
The computation still incurs floating-point roundoff. Coordinates local to each
window avoid global moment subtraction divided by narrow smoothing widths;
rounded zero-width windows have safe inactive denominators. Zero, very narrow,
cell-width and full-period kernels use the same route. Wide kernels include a
second ring of periodic images.

The method has no tunable sampling grid or quadrature order, frozen host clip
plan, or custom backward rule. Stored segments and cell sampling grow
quadratically with shape count; classifying all segments against all shapes
adds a higher preparation cost. There is no hard shape-count limit, but small
examples do not demonstrate acceptable time or memory for large layouts.
Transverse material coefficients remain arithmetic averages.

## Independent geometry and complete graphs

The geometric oracle uses vertical Cartesian chords and last-painted y
intervals, followed by independent adaptive x integration. It does not use
boundary jumps, Green kernels or the production clipping events. Tests cover
reversed paint order, seam overlaps, staggered cells, three shapes, completely
hidden shapes and whole-cell coverage. An exact full-cell difference checks
last-shape priority independently of the overlap size of a selected example.

Fourteen position/size/angle/epsilon/mu derivatives agree with two centred
perturbation steps on JAX and Torch. Parallel boundaries, disjoint shapes with
equal projected coordinates, quarter turns, narrow kernels and hidden-shape
sensitivities receive separate checks. The complete bottom-incidence conical
Jones graph includes both shapes and one layer thickness in eight joint
parameters. Source, time step, mesh and PML coordinates remain fixed.

One quarter-turn edge lands exactly on a translation-kernel knot. The cell
integral is C1 there, but its one-sided second derivatives differ. Consequently
a centred difference `D(h)` has a leading `O(h)` bias. An initial `h=2e-6` check
differed from AD by `1.375e-4`; halving the step halved that gap. The two
first-order Richardson estimates `2*D(h/2)-D(h)` differed from AD by at most
`4.21e-9`. The test retains its original `5e-7` absolute budget and also checks
the affected centre/size/angle slopes using the independent Cartesian oracle.
The production derivative was not altered to fit a finite difference at a knot.

A separate horizontal-edge check did expose production derivative errors even
though the material values agreed. Adjacent y windows counted their shared
knot twice, and sorting fixed integration endpoints together with live cuts
could let a tied cut move the integration domain under differentiation. Window
membership now uses half-open coordinate intervals. Both the boundary-event and
cell-integral stages sort internal cuts separately and retain explicit fixed
zero/one endpoints. JAX and Torch angle derivatives at the affected alignments
pass the independent Cartesian checks with the original error budget.

One JAX trace accepts overlap creation/removal and periodic translations; it
matches NumPy after the topology changes. Invalid live sizes, angles or
materials return NaN scattering and `stable=False`. This validates replay of
the event representation, not smoothness at an actual physical contact.

## Independent graded optical target

The optical case uses period `(0.6,0.6)`, layer depth 0.2, top epsilon 1,
substrate epsilon 1.21 and all mu values 1 at wavelength 1, normal incidence.
Two overlapping rectangles share a rotation angle 0.37. In their local frame,
the first has centre `(-0.04,-0.015)`, size `(0.3,0.18)` and epsilon 1.8. The
second has centre `(0.055,0.025)`, size `(0.22,0.22)` and epsilon 2.25 and is
painted last. Both rotate about the global centre `(0.025,-0.025)`.

Their intersection is an explicit rectangle in the common local frame. An
independent sinc Fourier transform forms the material contrast as
`0.8*indicator(first) + 1.25*indicator(second) - 0.8*indicator(intersection)`.
The native boundary algorithm is not used to build this profile. Box sinc
factors apply a fixed physical translation width 0.05 along laboratory x/y.
Cell-centred Fourier sampling and small-ringing clipping supply one identical
scalar profile for RCWA and Meep MaterialGrid. Derivatives are with respect to
the shared angle, the second rectangle's local x position and its width.

The primary profile grid is 256, RCWA order `(8,8)` and centred difference step
`1e-5`. Independent controls double the grid, raise order to `(10,10)` and halve
the difference step. Native FDTD uses physical PML thickness 1, duration 60,
pulse width 2.5, aligned layer faces and directional extraction. All eight
complex s/p reflection/transmission channels are compared at the layer faces.
All recorded calculations use CPU float64.

The maximum absolute errors against the independent Fourier-profile RCWA
reference are:

| Solver / resolution | Complex amplitude | Angle derivative | Relative-position derivative | Width derivative |
| --- | ---: | ---: | ---: | ---: |
| Native / 20 | 7.946e-3 | 1.586e-3 | 7.253e-3 | 9.653e-3 |
| Native / 30 | 3.643e-3 | 7.192e-4 | 2.910e-3 | 4.816e-3 |
| Native / 40 | 2.080e-3 | 3.459e-4 | 1.954e-3 | 2.862e-3 |
| Meep / 20 | 2.087e-2 | 3.248e-3 | 4.117e-2 | 9.437e-2 |
| Meep / 40 | 8.664e-3 | 2.024e-3 | 2.416e-2 | 4.178e-2 |

All four errors decrease under each solver's refinement. Native energy-balance
residuals at resolutions 20/30/40 are `2.657e-3`, `1.163e-3` and `6.509e-4`.
An audit of all four Yee offsets at those resolutions confirms that the
endpoint corrections leave this optical case's material fields unchanged;
their three-parameter derivatives change by at most `1.777e-15`.
These are graded-profile results with a fixed positive smoothing width, not
sharp-interface convergence or a proof that one solver is generally more
accurate. Meep retains a larger position/width derivative difference, beyond
the controls below. This experiment does not isolate its cause or establish
Meep's finite-grid sensitivities as an exact gradient oracle.

Independent reference and runtime controls record the following maximum
absolute changes. Derivative columns have the same order as above.

| Control | Amplitude | Angle derivative | Relative-position derivative | Width derivative |
| --- | ---: | ---: | ---: | ---: |
| RCWA profile grid 256 to 512 | 8.542e-7 | 6.209e-6 | 5.391e-5 | 9.944e-6 |
| RCWA order 8 to 10 | 1.710e-6 | 2.473e-5 | 9.222e-5 | 1.183e-4 |
| RCWA difference step halved | — | 2.261e-8 | 1.573e-6 | 8.015e-8 |
| Native / 40 duration 60 to 80 | 1.574e-11 | 1.716e-11 | 9.684e-11 | 2.013e-10 |
| Meep / 20 profile grid 256 to 512 | 1.174e-5 | 1.609e-4 | 9.937e-4 | 3.600e-4 |

Across all three Meep archives, increasing duration 80 to 100 changes amplitudes
by at most `1.784e-11` and derivatives by `2.123e-10`. Halving the shape
perturbation changes derivatives by at most `8.543e-6`. Public epsilon probes
agree with the intended profile to `4.441e-16`. The raw coarse/fine/profile
candidates passed the preselected runtime, perturbation, profile and refinement
criteria before their bytes were frozen; those budgets were not relaxed.

## Public Meep reference and reproduction

The existing bridge's `--shape painted_rectangles` mode requires positive
smoothing. It passes the independent inclusion-exclusion profile to the public
[MaterialGrid API](https://meep.readthedocs.io/en/latest/Python_User_Interface/#materialgrid)
with internal averaging disabled and checks its placement with public epsilon
probes. No Meep implementation is opened, copied or used as a Wavebit dependency.
The shape sensitivities in these references are central differences, not Meep
adjoints.

The domain is `(0.6,0.6,8)`, transversely periodic, with z PML thickness 1 and
Courant factor 0.4. A sine-Gaussian source at z=-2 has width 2.5, peak 12.5 and
end 25. Every-step DFT monitors at z=-1 and 1.2 use public integration weights,
empty-cell calibration and discrete phase transport to the layer faces. Both
polarizations, durations 80/100, difference steps `1e-4`/`5e-5`, FDTD resolutions
20/40 and profile grids 256/512 are preserved in the archives.

Run `python -m studies.cross_solver.painted_shapes.fdtd_painted_shapes --output painted.json`
to repeat the native study. Generate an external candidate with
`python tests/references/interface/meep_rotation.py --shape painted_rectangles
--smoothing .05 --resolution 20 --output candidate.npz`. Repeat at resolution 40
and separately with `--profile-grid 512` at resolution 20. Preserve raw
candidates and require their original controls to pass before promoting them;
regeneration is not a substitute for diagnosing a failing assertion.
