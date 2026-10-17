# FDTD research-alpha scope

> **Historical API:** snippets and descriptions using `wavebit.compat` on this page require a historical checkout.
> The namespace is absent from current installations; the pre-retirement baseline is
> [revision `3c8a50f8`](https://github.com/fieldalgo/wavebit/tree/3c8a50f8a6506240169538957e99648a4b18ca3a).
> Reproduce recorded experiments with their own cited commit and environment; their measurements have not been rerun.


:::{note}
This page records the measured scope of the earlier periodic factory, kept as `wavebit.compat.legacy.fdtd`, and of its native grid `fdtd_grid`; the evidence was measured there and its limits are unchanged. In the current `wavebit.fdtd` module a periodic stack is an `fdtd.Problem` with `Periodic` or `Bloch` faces, a `PlaneWave` and `Diffraction` monitors (see [the time-domain guide](time-domain.md)), and its scope and admission rules are in [how the solver models a problem](model.md).
:::

The earlier periodic factory, `wavebit.compat.legacy.fdtd` (formerly `wavebit.fdtd(...)`), is
experimental. Its intended use is research on periodic isotropic stacks, with diffraction
results comparable to RCWA. The separate [general FDTD API](../../migration/general-fdtd-v1.md)
uses `wavebit.compat.legacy.fd.Scene`; this page covers the periodic Stack/Incidence routes. It
is not yet a general-purpose replacement for an established FDTD package. This page states what
has evidence, what remains uncertain, and how to reproduce the checks.

## Supported interface versus validation coverage

An accepted input is not a guarantee of accuracy for every combination in this table.
This table describes the legacy scene path selected by omitting `backend`;
the following sections describe the shared engine and explicit-backend scene path.
How far the two engines agree on the same scenes, with grid, time step and materials
matched, is measured in [route parity](../../validation/fdtd/fdtd-route-parity.md).

| Area | Available interface | Evidence and limits |
| --- | --- | --- |
| Raster geometry | `(layer, y, x)` arrays; y-invariant and rectangular crossed cells | Analytic planar checks, 1D gratings, and selected asymmetric crossed gratings. The crossed external reference covers two dielectric scenes, not arbitrary 3D geometry. |
| Incidence | From either side through the public adapter; finite real scalar angles, including signed theta; TE/TM at zero azimuth; `psi`/`chi` or complex Jones pairs at general azimuth | Invalid complex/nonfinite angles, malformed Jones vectors and conflicting settings are refused before dispatch. Circular slab amplitudes are compared with RCWA on scalar and coupled paths; patterned ellipticity, overall phase and interference are checked. Near-Rayleigh and grazing incidence require separate refinement. |
| Materials | Positive real isotropic epsilon/mu; electric Drude/Lorentz poles | Closed-form dielectric, magnetic and Drude slabs, selected 1D patterned cases. The installation matrix covers planar magnetic/Drude media through both raster paths. Three moderate-contrast patterned cases and one damped negative-epsilon metal have independent Meep references (below); low-loss and strong-resonance regimes remain unvalidated. |
| Spectra | Multiple wavelengths in one pulse run | At oblique incidence the in-plane wavevector is fixed, so angle varies across the band. Check the reported angles and incident spectrum. |
| Outputs | `Result.incident`, R/T and complex s/p amplitudes, labelled diffraction orders; no Cartesian `r_xyz`/`t_xyz` | Frozen external-reference tests include complex propagating amplitudes. Evanescent channels are not covered by the convergence table below; deeply evanescent amplitudes can be masked. |
| Time records | Ex/Ey/Ez/Hx/Hy/Hz snapshots on request, measured axes and times, plotting and animation | Component staggering, coherent polarization, recording invariance and plotting contracts are tested. Circular vacuum records check equal component energy, temporal quadrature and both rotation directions. Six-component recording preserves native Yee staggering, including the H half-step time; Point sampling and instantaneous Poynting vectors interpolate the native axes; nonzero-Bloch time records propagate a positive-frequency carrier so transverse energy flows in the intended direction; manufactured-field checks and vacuum flux/refinement tests cover this interpolation. Streaming six-component frequency planes and integrated spectral power are available; offline DFT equality, coherent Jones mixing, and slab R/T ratios are tested. Whole-volume harmonic reconstruction remains unavailable. |
| Execution | NumPy on CPU; optional Matplotlib plotting | Clean wheel installation and calculation checked on macOS arm64/Python 3.12. Other operating systems and Python versions need release CI evidence. |

The legacy Stack/Incidence FDTD path does not provide autodiff or JAX/PyTorch execution.
The explicit-backend common scene adapter now provides both, with fixed meshes
and incidence-angle/material/pole and layer-thickness gradients. Total depth
can vary with explicit `variable_depth=True` and a fixed nominal mesh.
Common-backend time histories and user z monitors
are also available, with native component staggering and centre-calibrated
causal time records; see the [guide](time-domain.md#common-scene-adapter).
Both public periodic routes support bottom incidence and return fields in the
original laboratory coordinates: bottom-to-top layers, bottom face `z=0`, +z
upward. The private legacy kernel remains one-way. All public periodic order
amplitudes use the cell centre and each outgoing order's own s/p frame. The
backend selector takes names or `wavebit.Backend` members, never integer indices.
Explicit backends require `0 <= theta < pi/2`; the backend-omitted path also
accepts signed scalar theta. Both use radians and refuse Cartesian `kx`/`ky`
incidence. See the [shared periodic conventions](time-domain.md#shared-periodic-conventions).

Diagnostic field normalization remains route-specific: legacy frequency planes
use the soft-source spectrum, explicit-backend planes use unit incident E at
each wavelength, and explicit time records use a constant carrier calibration.
H is `Z0 H`, and monitor `power_z` is signed along +z. These fields are not the
Cartesian order amplitudes or reconstruction state returned by RCWA. The
[integration checkpoint](../../development/solver-integration.md) separates these current
contracts from the planned material and namespace redesign.
Neither scene path provides MPI execution,
skewed lattices, arbitrary curved parametric geometry, anisotropic tensors,
or complex-angle incidence. Complex Jones states support diffraction and transverse time records through
separately propagated sine/quadrature sources. Use the appropriate RCWA capability where available; do not
infer FDTD support from RCWA examples. Unsupported inputs are described in [the solver
notes](fdtd.md) and [API reference](../../reference/api.md).

## Shared native-grid engine

The separate `wavebit.compat.legacy.fdtd_grid` interface provides NumPy, compiled JAX and checkpointed Torch Yee
updates, streaming field/power planes, and differentiable material/source inputs.
It accepts native grids with uniform x/y and optional nonuniform z widths, diagonal real coefficients, periodic
boundaries, optional graded z damping, homogeneous-end z CPML, and electric
Drude/Lorentz ADE poles. An explicit-backend scene adapter now adds rasterization,
reference-calibrated plane-wave sources and diffraction Results. Static nonuniform
z widths and scene grading now share this engine, including CPML, records,
monitors and material gradients. CPU eigenmode,
NumPy/JAX parity, finite-difference material sensitivities, pulse sensitivities,
and absorber energy-removal checks pass. Live external checks now also compare a
three-wavelength slab against Meep at three resolutions, and nonuniform diagonal
material updates/gradients against FDTDX. Meep is accessed only through its public
API. Both solvers' slab answers approach the independent Airy solution at about
second order; the fine-grid complex error is 0.0072. Frozen references retain
these CPU results for regression (see `tests/references/README.md` in the checkout).
Additional live CPU checks now cover native CPML and dielectric/Lorentz/Drude/
mixed slabs at two resolutions, plus FDTDX two-pole 3D updates and strength
gradients. Direct Meep comparisons match its documented point sampling of the
dispersive part; separate tangential half-cell weights converge against Airy.
PML width/runtime controls and oblique pulse removal are tested. Arbitrary spatially varying or dispersive PML backgrounds remain outside this evidence.
The common scene engine now also has bounded [low-loss mixed-metal](../../validation/fdtd/fdtd-mixed-metal.md)
and [oblique-pulse](../../validation/fdtd/fdtd-oblique-pulse.md) checks. Its static `pulse_width` control
narrows source bandwidth to reduce glancing-wave tails. Planar TE/TM amplitudes
refine against Airy, and a selected TE metal grating refines against RCWA. This
does not validate arbitrary low-loss patterns: the corresponding sharp-corner
TM RCWA reference remains unconverged, and normal-component dispersive
interface averaging remains approximate.
Optional [directional E/H extraction](../../validation/fdtd/fdtd-directional.md) separates outgoing
and returning components using the discrete Maxwell admittance. Exact-mode,
magnetic/conical, bottom-record and backend-gradient checks cover the extraction.
A nearby long-lived TM oscillation remains a separate source-bandwidth and
temporal-convergence concern; direction separation alone does not validate
near-Rayleigh scattering.
Static [CPML strength](../../validation/fdtd/fdtd-pml-strength.md) is independently configurable
on the shared grid and on both scene paths. Its nominal amplitude
target is not a measured-reflection bound; width, strength, mesh and temporal
checks remain separate, and the original default profile is preserved.
Offline [ringdown fitting](../../validation/fdtd/fdtd-ringdown.md) now extracts damped frequencies
from point traces and is checked against exact/native/external field records.
The remaining near-Rayleigh signal has a cutoff-related, window-dependent tail;
a small multimode residual alone does not establish a physical Q or time convergence.
Optional [source spectral zeros](../../validation/fdtd/fdtd-pulse-notches.md) suppress one or two
selected frequencies before the four scene runs. Uniform material-mode suppression
and complex records match fresh public-API Meep results on all three CPU backends;
calibrated Airy optics and material gradients are preserved. In the cutoff study,
smaller returning components do not reliably reduce both polarizations' total
amplitude errors. The Gaussian remains the default.
The [independent source-position control](../../validation/fdtd/fdtd-source-clearance.md) now allows
fixed illumination and measurement planes when padding changes. In the tested
near-cutoff case, separate source and clearance changes are much smaller than
the remaining RCWA discrepancy; they do not settle its time or grid error.
An [independent harmonic lattice reference](../../validation/fdtd/fdtd-lattice-reference.md) now
separates these contributions in that uniform-grid dielectric case. Its exact
outgoing-boundary solution refines approximately at second order against RCWA.
A separate finite-CPML solve exposes boundary errors hidden by cancellation
with lattice and finite-time errors. These are bounded study instruments,
sharing the native rasterization, rather than general frequency-domain APIs.
Static [independent transverse resolution](../../validation/fdtd/fdtd-transverse-resolution.md)
now permits x/y refinement without moving z nodes, sources or measurement
planes. The time step still follows all actual spacings. Transverse normal
harmonic averaging now precedes tangential arithmetic averaging consistently
for epsilon and mu, fixing an x/y asymmetry in crossed raster corner cells.
This can change crossed-raster results even at the default mesh; it does not
replace the diagonal interface model with general tensor homogenization.
Fixed-grid [incidence-angle sensitivities](../../validation/fdtd/fdtd-scene-angles.md) now preserve
the `theta` graph through illumination, Bloch boundaries, diffraction and
reference normalization. Array angles require an explicit order window;
the mesh remains static. CPU checks cover
exact uniform-medium phase derivatives, differentiated magnetic Airy refinement,
crossed material/pole derivatives, bottom incidence and calibrated records.
These checks do not establish derivative convergence near diffraction thresholds
or sharp dispersive corners.
The [azimuth extension](../../validation/fdtd/fdtd-scene-azimuth.md) now differentiates phi jointly
with theta and material parameters. Array phi keeps full y sampling across
zero and requires Jones/psi polarization. Exact discrete phase derivatives,
normal-incidence field rotations, axis exchange and two finite-difference
steps check this path. Unequal transverse cells introduce a small planar
azimuth sensitivity that decreases under refinement toward continuum Airy.
The [half-space extension](../../validation/fdtd/fdtd-halfspace-gradients.md) also retains exterior
epsilon/mu derivatives through CPML, illumination and reference calibration.
An explicit pair of nominal mesh indices fixes padding/grading. CPU checks
include analytic CPML derivatives, magnetic Airy refinement and new public-API
Meep finite differences with duration and perturbation-size controls.

The [layer-thickness extension](../../validation/fdtd/fdtd-layer-gradients.md) moves internal
interfaces through a fixed nominal mesh while preserving total depth. CPU
JAX/Torch checks include typed scenes, bottom illumination, dispersive masks,
field records and user monitors. Planar Airy refinement and independent Meep
thickness perturbations expose position-dependent continuum derivative error,
even when the two FDTD derivatives closely agree. A coating-optimization example
includes a finer-grid physical check. Optional [internal-interface averaging](../../validation/fdtd/fdtd-layer-smoothing.md)
provides C1 cell fractions at a fixed positive `layer_smoothing` width. The
width changes the material profile; sharp and graded continuum references
separate that bias from finite-grid error. The [total-depth extension](../../validation/fdtd/fdtd-variable-depth.md)
also permits a moving exterior face with fixed laboratory coordinates and
live reference-phase derivatives. It supports both incidence sides and
optionally averages the moving face. Independent magnetic-film optics and
public-API Meep perturbations bound one selected case; arbitrary combinations
still need refinement. General higher-order geometry derivatives remain unsupported; circles and
ellipses have the bounded area route described below. Opt-in rectangle rotation is described below.

The explicit route now accepts [axis-aligned rectangle and square recipes](../../validation/fdtd/fdtd-rectangle-gradients.md)
using the same geometry reader as RCWA. Centres, lengths and scalar Material
epsilon/mu retain gradients on a fixed mesh, including periodic wrapping and
later-shape painting. Optional transverse translation averaging smooths cell
crossings; actual shape contacts can still have topology kinks. CPU checks
include exact area/cell integrals, coincident projected edges, bottom joint
material/thickness gradients, RCWA refinement and public-API Meep perturbations.
The default separable path does not accept rotation; neither recipe method
accepts dispersive shape materials.

The explicit common route also provides opt-in `shape_integration='area'` for
one rotated rectangle or square per layer. It retains angle derivatives in
radians through exact periodic cell areas, with arithmetic transverse material
averages and a full static y-period mesh. This is a bounded interface
approximation, not full effective-tensor smoothing. The
[rotation contract](time-domain.md#rotated-rectangle-area-integration)
and [independent measurements](../../validation/fdtd/fdtd-rotation-gradients.md) describe its limits.
Positive `shape_smoothing` also works on this area route through exact
translated-polygon moments. Independent geometry and full JAX/Torch graph
checks cover its continuous cell-crossing derivatives. Matching graded RCWA
references separate the changed physical profile from finite-grid error;
[the measurements](../../validation/fdtd/fdtd-rotation-smoothing.md) do not establish general
sharp-interface gradient accuracy. A separate [adjoint comparison](../../validation/fdtd/fdtd-adjoint-validation.md)
checks Meep adjoint against its own forward differences for two complex
transmissions. The existence of an adjoint gradient does not establish its
finite-grid or continuum accuracy.


The shared `layer_thicknesses` helper now supplies positive, fixed-total
lengths with individual minima for NumPy, JAX and Torch. A
[six-variable coating example](../../validation/fdtd/fdtd-multilayer-optimization.md) jointly
optimizes three lengths and three dielectric permittivities, with finer-grid,
sharp/graded continuum and public-API Meep derivative checks. An independent
81-wavelength sharp-stack check confirms improvement beyond the three training
wavelengths. The example stops at an iteration limit; these measurements do
not establish global optimization convergence or arbitrary-material design.

The common engine reports physical and discrete lower-cutoff classifications
and supports independent diffraction-plane distances. A binary dielectric
experiment at wavelengths 0.2% either side of a Rayleigh threshold exposes a
false-open channel on a coarse grid and persistent TM amplitude errors after the
classification agrees. See the [near-Rayleigh measurements](../../validation/fdtd/fdtd-rayleigh.md);
this regime remains an accuracy limit, even when R+T is close to one.

The common engine also supports [layer-aligned z meshes](../../validation/fdtd/fdtd-layer-alignment.md):
`align_layers=True` fits every nominal layer face without changing its thickness;
with `mesh_thickness`, actual internal faces can move through the fixed mesh.
Capped-metal TE/TM tests refine against an independent characteristic matrix,
including bottom incidence. This reduces a selected normal-interface error;
it does not resolve the lateral metal-corner limitation above.
Selected [lossy ENZ film](../../validation/fdtd/fdtd-enz.md) checks now cover complex TE/TM scattering,
the enhanced internal normal field, bottom illumination and material sensitivity
refinement. Per-column and Jones-input convergence diagnostics distinguish late
TM transients from a settled TE column. Time-step refinement remains necessary
near the material zero crossing; the exactly lossless ENZ limit is not validated.
Native three-axis Bloch phases and their gradients are also implemented, with
exact discrete-mode checks and a live FDTDX complex-field/material-gradient
comparison. Transverse phases may coexist with z CPML; longitudinal phase must
be zero. The native core now follows RCWA's Array API architecture: one physics
implementation, common graph-preserving conversion, and separate NumPy/JAX/Torch
time-loop policies. Torch additionally passes the existing FDTDX field/gradient
goldens for non-dispersive, ADE and Bloch cases. Omitting backend in `wavebit.compat.legacy.fdtd`
preserves legacy Stack/Incidence dispatch; explicitly selecting one invokes the common scene
adapter. These historical external comparisons used CPU. Current public APIs
have separate [native and periodic-slab CUDA evidence](../../validation/fdtd/native-cuda.md)
and [bounded general-scene CUDA evidence](../../validation/fdtd/general-cuda.md).
See the [native-grid guide](time-domain.md) for the earlier interface.

## Reproduce the installation check

The distribution runner builds a source archive, builds its wheel from that archive,
checks metadata with `twine check --strict`, and installs into a fresh virtual environment.
It verifies package origin, bundled material tables, a NumPy RCWA solve, the 36-case
FDTD matrix and the current-API `examples/fdtd/pulse_slab.py`. It then installs the same
wheel's `plot` extra and runs the compatibility-API `examples/fdtd/fdtd_record.py` scene to
save a field image. Python isolated mode prevents the checkout or `PYTHONPATH` from
shadowing the installed package. No artifacts are published.

```sh
python -m pip install build twine
# Choose a new output directory. The runner works on Windows, macOS and Linux.
python tools/run_distribution_smoke.py --output artifacts/wavebit-distribution
```

The configured CI runs this check on Linux, macOS and Windows with Python 3.12.
Configuration is not evidence of a successful remote run; those results remain a release
gate. The output directory retains both distributions, `fdtd-install.json` and
`fdtd-field.png`; temporary build sources and virtual environments are removed afterward.

The underlying `tests/distribution/fdtd_install_smoke.py` can also run directly with an installed
package's Python from outside the checkout.

The script calls the installed package's compatibility API, `wavebit.compat.legacy`, and does
not add the checkout to the import path. Its JSON prints the imported package path, dependency
versions, and every measured case. It checks 36 planar combinations: two raster paths, three
materials, three incidence directions, and two linear polarizations. It compares complex
amplitudes with RCWA, requires a settled field, and checks that JAX, PyTorch and Matplotlib
were not imported by the base calculation.
On 2026-09-07, Python 3.12 with NumPy 2.5.3, SciPy 1.18.1 and array-api-compat
1.15.0 passed all 36 cases: maximum amplitude difference 0.006079 and maximum residual
4.35e-9. The per-case measurements are in the run record `fdtd_alpha_install.json`
(2026-09-07), archived outside the repository; `tests/distribution/fdtd_install_smoke.py`
prints a report of the same form for a fresh installation.

The optional plot extra was also installed (Matplotlib 3.11.1); a public `Stack`/`Incidence`
conical solve recorded Ey/Ex and rendered an Ex slice from the installed package.

The 0.02 amplitude bound is an installation smoke-test criterion, not a promised solver
accuracy. Both solvers belong to Wavebit; independent evidence comes from the existing
external-reference suite (`tests/references/README.md` in the checkout).

## Crossed dielectric refinement

Run from the checkout root:

```sh
python studies/cross_solver/alpha_convergence/fdtd_alpha_convergence.py --output studies/cross_solver/alpha_convergence/records/fdtd_alpha.json
```

The checked-in JSON records the source digest, dependency versions, actual grid sizes,
PML thicknesses, runtime, residuals and warnings. The case definitions are shared with
`tests/references/_meep_crossed_cases.py`; no external solver is imported or regenerated.
Lengths here are micrometres. Both s and p incident basis solutions are compared on
explicit order labels. Only channels with `n² - |k_parallel|² > 0.05` are included.
Amplitude errors are absolute complex differences, maximized across r_s/r_p/t_s/t_p and
both sources; power errors are absolute per-order R/T differences, not percentages.

The reference is RCWA at order `(13, 13)`, with `(9, 9)` also evaluated to expose its
sensitivity. A small reference change is evidence of stability over that interval, not
an exact solution. FDTD uses 120 optical cycles, 12 PML cells, Courant fraction 0.99,
subpixel averaging, uniform z spacing, and padding scale 1.0. Holding PML *cell count*
fixed changes physical absorber thickness as the grid refines; the sweep therefore
measures the combined numerical setup, not a pure spatial truncation error.

| Scene | Grid (nx × ny × nz) | Cell size | Max amplitude error | Max power error | Residual |
| --- | --- | --- | --- | --- | --- |
| square_18deg | 14 × 14 × 50 | 0.06000 | 0.07492 | 0.02870 | 1.15e-03 |
| square_18deg | 21 × 21 × 54 | 0.04000 | 0.02538 | 0.01348 | 5.99e-04 |
| square_18deg | 28 × 28 × 61 | 0.03000 | 0.01441 | 0.00740 | 5.23e-04 |
| rect_28deg | 16 × 24 × 53 | 0.05000 | 0.06504 | 0.04774 | 5.91e-03 |
| rect_28deg | 24 × 36 × 63 | 0.03333 | 0.02489 | 0.01697 | 4.56e-03 |
| rect_28deg | 32 × 48 × 76 | 0.02500 | 0.01450 | 0.00917 | 3.99e-03 |

RCWA `(9,9)` → `(13,13)` maximum amplitude changes were 0.000215 (square_18deg), 0.000457 (rect_28deg).


These are two selected lossless dielectric scenes. They establish neither a universal
second-order rate nor an error bound for magnetic, lossy, resonant, grazing or near-Rayleigh
structures. Monitor residual is a settling indicator, not an error estimate. A new device
needs separate increases in runtime, padding/PML and spatial resolution until the requested
observable stabilizes. Symmetry and R+T alone cannot establish amplitude or phase accuracy.

## Runtime and boundary controls

The companion study holds the square scene at resolution 25 and the rectangular scene
at resolution 30. These are the middle grids above, not the finest grids. It changes
runtime from 120 to 180 to 240 optical cycles; at 240 cycles it independently increases
PML depth from 12 to 20 cells and padding scale from 1.0 to 1.5. Each boundary control
changes one parameter from the 240-cycle baseline. This separates their sensitivity
from the spatial sweep without claiming the same sensitivity on finer grids.

```sh
python studies/cross_solver/alpha_controls/fdtd_alpha_controls.py --output studies/cross_solver/alpha_controls/records/fdtd_alpha_controls.json
```

The JSON records complete settings, actual grids, residuals, warnings, reference errors,
and pairwise differences across the same propagating channels. The table reports the
maximum complex amplitude change between each pair, rather than a total R+T change.

| Control | Square amplitude change | Rectangular amplitude change |
| --- | --- | --- |
| 120 → 180 cycles | 0.000145 | 0.001778 |
| 180 → 240 cycles | 0.000040 | 0.000403 |
| PML 12 → 20 cells | 0.000551 | 0.001804 |
| Padding 1.0 → 1.5 | 0.005509 | 0.002415 |

The time sensitivity falls from 120→180 to 180→240 cycles in both scenes. This supports
using longer runs to reduce the remaining transient, but does not make the spatial error
vanish. PML and padding controls must still be inspected separately. A larger padding
region need not monotonically improve the complex coefficient at fixed resolution.
Residuals differ under the boundary controls: these are finite-runtime sensitivities,
not isolated asymptotic absorber errors. Tight error targets require further runtime
refinement for each boundary setting.

An independent analytic check isolates that last point without RCWA: in vacuum,
`t_s = t_p = exp(i k0 cos(theta) d)`. `tests/fdtd/test_fdtd_boundary_controls.py` uses wavelength
0.8, thickness 0.2, period (0.16, 0.16), theta 18 degrees and phi 40 degrees, with 120 cycles
and 12 PML cells. For padding scales 1.0/1.5, complex transmission errors are respectively
0.006717/0.008513 at resolution 25, and 0.001716/0.002157 at resolution 50 (both source
polarizations agree at these digits). The error and padding sensitivity decrease with
refinement. This is evidence for a numerical phase contribution even in vacuum; it does
not identify every source of padding sensitivity in a patterned device. The test checks
both the analytic phase and the decrease, rather than relying on transmitted power alone.

## Selected patterned 3D materials

`tests/references/golden_meep_crossed_materials.npz` adds black-box Meep references for
three asymmetric crossed rasters: patterned permeability, electric Drude dispersion and
electric Lorentz dispersion. Meep is used only through its public API in an external
environment; it is never a Wavebit installation dependency. The generator checks Meep's
actual epsilon and mu at three wavelengths against a closed-form material evaluation.
Its frequency conversion follows the [Meep susceptibility API](https://meep.readthedocs.io/en/master/Python_User_Interface/#lorentziansusceptibility).

All three scenes use period (0.6, 0.8), thickness 0.2, wavelength 0.9 micrometres,
theta 20 degrees and phi 35 degrees. Both half-spaces have mu=1; their indices are
1.0 and 1.4. The active material occupies the ones in this y-by-x raster:

```text
1 1 0
1 0 0
0 0 0
0 1 1
```

Background epsilon/mu are 1. The active material has epsilon_infinity=2.25. For the
magnetic case mu=1.5; for both dispersive cases mu=1. With `omega_c=2*pi/0.9`, the
Drude pole is `(0, 0.2*omega_c, 0.9*omega_c**2)` and the Lorentz pole is
`(1.5*omega_c, 0.2*omega_c, 1.2*omega_c**2)`. At the test wavelength their respective
active permittivities are 1.384615+0.173077i and 3.186037+0.149766i. These are moderate
contrast, positive-real-epsilon cases; they do not validate metallic negative-epsilon
regimes, arbitrary resonances, or magnetic dispersion.

Meep 1.34.0 runs at resolutions 20/40, PML thickness 0.5, padding 0.6, monitor gap 0.1,
source fractional bandwidth 0.2 and post-source duration 120, inheriting that version's
remaining defaults. The runtime control doubles the post-source duration at resolution
40 and leaves the frozen reference untouched. Wavebit uses resolutions 20/40, 160
optical cycles, 12 PML cells, padding scale 1, Courant fraction 0.99, subpixel averaging
and uniform z spacing. The solvers' time parameters have different meanings and are
not treated as equal runtimes.

Comparisons cover both incident polarizations and all complex s/p reflection/transmission
coefficients in propagating orders with `n²-|k_parallel|² > 0.05`, on explicit order labels.
RCWA orders 7 and 11 provide an additional internal refinement check; the frozen values
come exclusively from Meep.

| Material | FDTD / Meep | FDTD / RCWA 11 | Meep / RCWA 11 | Material omitted / Meep |
| --- | --- | --- | --- | --- |
| magnetic | 0.032199 | 0.009064 | 0.031502 | 0.169766 |
| drude | 0.026571 | 0.003859 | 0.025702 | 0.186435 |
| lorentz | 0.045729 | 0.011593 | 0.044459 | 0.171626 |

Values are maximum absolute complex-amplitude differences, not percentages. The last
column uses RCWA with mu contrast or the poles removed. The drude and lorentz FDTD columns
were remeasured after the legacy path (`backend` omitted) moved Ey's pole fill to Ey's own
y half-step, where Ey's permittivity was already sampled. Before that fix the drude row
read 0.024954 (FDTD / Meep) and 0.007410 (FDTD / RCWA 11), and the lorentz row 0.048092
and 0.011859: those rows were produced with Ey's pole strength sampled half a cell off in
y. At resolution 40, the table's grid, the fix moves FDTD toward RCWA order 11 on both
rows; at resolution 20 both move slightly away (drude 0.014384 → 0.016455, lorentz
0.035711 → 0.037368). On the drude row it also moves FDTD away from Meep. But Meep
resolution 40 is itself 0.025702 (drude) and 0.044459 (lorentz) from RCWA order 11, about
as far as FDTD is from Meep, so the FDTD / Meep column mostly measures Meep's own grid
error; the FDTD / RCWA 11 column is the one that shows the fix. The magnetic row has no
poles and reproduces to the printed digits. The largest Meep doubled-runtime
change across all six material/source combinations was 6.82e-8, below the 0.001 control
criterion and far below these spatial discrepancies.

Both solvers' finer grids are closer to RCWA order 11 than their coarser grids. Removing
mu contrast or removing the electric poles produces a discrepancy distinguishable from
the correct-material reference gap; the regression test checks that separation. This is
evidence for these material paths and scenes, not a universal 0.05 accuracy guarantee.
The runtime control and its reference-file digest are recorded separately in
`tests/references/golden_meep_crossed_materials_runtime.json`.

Reproduce the external half in the dedicated Meep environment, choosing new output paths:

```sh
python tests/references/gen_goldens_meep_crossed_materials.py --output artifacts/new-materials.npz
python tests/references/check_meep_crossed_material_runtime.py --output artifacts/new-runtime.json
```

The second command checks the committed reference, not the first command's output.
Both tools refuse to overwrite an existing output. CI reads the frozen artifacts with
NumPy and never imports Meep. Broader cases need their own external and convergence checks.

## Damped negative-epsilon metal

A separate subwavelength crossed case uses a Drude material with epsilon_infinity=1,
mu=1 and pole `(0, 0.5*omega_c, 4*omega_c**2)`, where `omega_c=2*pi/0.9`.
Its active permittivity is **-2.2+1.6i** at wavelength 0.9 micrometres. The rectangular
period is (0.2, 0.3), thickness 0.1, and the active y-by-x raster is:

```text
1 0
1 1
0 0
```

The background and half-spaces are nondispersive; incidence and half-space indices
match the material cases above. Only order (0,0) propagates in either half-space.
This is a 3D patterned, strongly damped metal case, not a low-loss plasmonic resonance
or a higher-order metallic diffraction benchmark.

The original Meep grids 30/60 are preserved in `golden_meep_crossed_metal.npz`, including
a doubled-runtime control at resolution 60. That control changes amplitudes by at most
2.10e-9. Nevertheless, comparing Wavebit resolution 60 with Meep resolution 60 **failed**
the initial 0.06 complex-amplitude and 0.05 absorption criteria, and still fails them after
the Ey pole-fill fix described below: amplitude difference 0.07865, and absorption
differences 0.08801 (s) and 0.07017 (p). Increasing runtime alone
cannot establish spatial accuracy. These discrepant reference arrays are retained.

A separate resolution-120 tier is reproduced by `refine_meep_crossed_metal.py`; it does
not replace the coarser arrays. All tiers use the same physical geometry and public Meep
material API. The finer tier uses post-source duration 120; the doubled-runtime evidence
belongs specifically to resolution 60.

The original comparison bounds are unchanged. With the independent Meep-120 tier,
Wavebit resolution 60 at 240 cycles gives maximum complex-amplitude difference **0.04308**
(bound 0.06). Absorption is 0.42457/0.44894 (s/p), versus Meep's 0.47043/0.48626;
differences 0.04586/0.03732 are below the 0.05 absolute criterion (five percentage points).
This is a structural regression check at modest accuracy, not a precision guarantee.

| Meep resolution | Max amplitude difference from RCWA order 11 | s absorption | p absorption |
| --- | --- | --- | --- |
| 30 | 0.103694 | 0.509381 | 0.533899 |
| 60 | 0.067957 | 0.512580 | 0.519110 |
| 120 | 0.031885 | 0.470433 | 0.486264 |

Wavebit's amplitude difference from RCWA order 11 decreases from 0.03053 at resolution
30 to 0.01379 at resolution 60. The separate Meep tiers also approach that internal
reference. The near-constant s absorption on Meep's first two grids was not evidence
of spatial convergence. The new regression retains those grids and checks the decrease
in complex-amplitude discrepancy across all three external tiers.

The Wavebit numbers in this section were remeasured after the Ey pole-fill fix described
under the material table above. Before it, the same runs gave 0.08069 and 0.09168/0.07004
(s/p) against Meep 60, 0.05160 and absorption 0.42090/0.44907 (s/p) against Meep 120, and
0.06273 → 0.03433 from RCWA order 11 at resolutions 30 → 60. At resolution 60 the fix
moves s absorption 0.0037 toward both Meep tiers and p 0.0001 away from them, and it more
than halves the amplitude difference from RCWA order 11 at both resolutions. The largest
single R or T difference from Meep 120 grows, from 0.031 to 0.040 (s) and from 0.021 to
0.024 (p), still inside the test's 0.05. The Meep-60 comparison still fails its initial
criteria.


Wavebit's opt-in `studies/fdtd/metal_validation/fdtd_metal_validation.py` records grids 30/60 and separate
resolution-60 controls for 160→240 optical cycles, PML 12→20 cells and padding 1→1.5.
It compares against the preserved Meep-60 reference, making that discrepancy reproducible.
The control changes are respectively 3.19e-12, 5.49e-5 and 3.36e-4 in maximum complex
amplitude; monitor residuals are below 4.3e-11. These small changes do not erase spatial
error. RCWA orders 7→11 change amplitudes by 0.001061, another finite-truncation check
rather than an exact solution.

```sh
# In the separate Meep environment; choose fresh output paths.
python tests/references/gen_goldens_meep_crossed_metal.py --output artifacts/metal-base.npz
python tests/references/refine_meep_crossed_metal.py --output artifacts/metal-fine.npz
# In the Wavebit development environment, with committed references available.
python studies/fdtd/metal_validation/fdtd_metal_validation.py --output studies/fdtd/metal_validation/records/fdtd_metal_validation.json
```

## Remaining release gates

Since 2026-10-15 Wavebit is published as a tagged alpha, and the configured CI, including
the clean installs on the claimed platforms, runs on `develop`; no remote CI result is
asserted by this page. Select and publish the supported scope with each release, and
retain warnings and known limitations in examples. Before claiming a broader 3D material
capability, add more independent material cases, especially low-loss metals and strong
resonances, and repeat runtime/absorber refinement for those scenes. The selected cases above do not
cover the whole input space.
GPU evidence is limited to the [current native cases and periodic slab](../../validation/fdtd/native-cuda.md)
and [restricted general-scene route](../../validation/fdtd/general-cuda.md); the historical CPU
comparisons do not extend those scopes. Autodiff has since landed on the
explicit-backend route above, and general domains, sources and field monitors in the
[current FDTD module](general-fdtd.md).

The common area route also accepts `circle`/`disk` with a live centre and diameter.
[Disk integration and validation](../../validation/fdtd/fdtd-circle-gradients.md) cover exact sharp-cell
areas, stable translation averaging, joint shape/material/thickness derivatives,
and independent Fourier and public-API Meep comparisons. One shape per layer,
positive scalar materials and diameters within both periods remain required.
CPU validation does not establish GPU or general higher-order derivative support.

Rotated `ellipse` recipes now retain both principal-axis lengths, centres and
angle on the same common area route. [Ellipse integration and validation](../../validation/fdtd/fdtd-ellipse-gradients.md)
cover independent implicit-conic chords, circle/axis-swap limits, full JAX/Torch
graphs and controlled graded optical comparisons. Angular integration uses a
fixed 16-point rule; one shape per layer and projected extents within the two
periods remain required for curved shapes. Recipe dispersion remains outside
this route.

Multiple rotated rectangles/squares now use [live boundary painting](../../validation/fdtd/fdtd-painted-shapes.md)
with last-shape priority, periodic overlaps and shape/material derivatives.
Overlap topology may change inside one trace; exact contacts still have only
piecewise shape sensitivities. Curved shapes remain limited to single-shape
layers. Boundary-event costs grow with shape count; this is not a demonstrated
large-layout performance result.
