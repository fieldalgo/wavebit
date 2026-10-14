# Reference cases

Validation against solvers that are not ours. The expected answer is known, so these are tests,
not studies. What makes them their own directory is that *producing* the reference needs
software this project does not ship.

**How closely "known" means differs by row, and the difference is a fact about the method
rather than about the harness.** Where the oracle is another RCWA and neither side truncates,
agreement is ~1e-13 and a golden is a drift detector as well as a correctness one. Where the
oracle discretises — the Meep rows, which are an FDTD — both sides carry a grid error and
the agreement is ~1e-2 on the crossed cells and ~1e-3 on the 1D one, whose single planar
interface per period both grids hold better than a corner. Those rows are oracles for
*structure*: they separate a right answer from a wrong sign, a swapped axis or a mislabelled
order, and they say nothing about the last digits. Each row below states which kind it is.

The arrangement is the same for every oracle: the expensive half runs once, by hand, on a
machine that has the tools; what CI checks on every push is the committed numbers.

**Regenerating a golden is not a way to make a failing test pass.** These numbers did not come
from this codebase, which is exactly what makes them worth having.

## What the permittivity flip did to this directory (2026-08-20)

Wavebit took a refractive index at its front door until 2026-08-20 and takes a relative
permittivity now. The oracles did not all move the same way, and it is worth reading the
directions off rather than assuming a single rule:

| oracle | what it takes | what changed here |
| --- | --- | --- |
| **S4** (`Epsilon`) | permittivity, always did | the generators' `** 2` is **deleted**. Nothing was added |
| **fmmax** (`eigensolve_*`) | permittivity, always did | same: `gen_goldens_fmmax_field.py` lost its `** 2`, and `_fmmax_cases.py`, `_fmmax_mu_ambient_cases.py` and `_fmmax_patterned_mu_cases.py` were **already** stating tensors as `eps` and were not touched at all |
| **RETICOLO** (textures) | refractive **index** | the only place that gained a conversion. `interface/Reticolo.py::_index` takes the principal root on the way into Octave, in one function, for both half-spaces and every layer |

Two goldens store a material and could not follow: `golden_reticolo_conical_sweep.npz` (a
2024-09-09 study whose `.mat` inputs are not in this repository) and `golden_s4_field.npz`'s
`__stack` arrays. The RETICOLO sweep holds a refractive index still, converted **at the read**,
and says so where the conversion sits. The S4 field golden did the same until it was
regenerated on 2026-10-14; its `__stack` arrays hold permittivity now and are read as such.

The RETICOLO change is the one that could have moved numbers, so it was checked the way this
directory's own note on a generator that could not reproduce its own golden prescribes:
regenerating `golden_reticolo.npz` (72 arrays), `golden_reticolo_field.npz` (33) and
`golden_reticolo_conical_field.npz` (33) against V10 after the flip reproduces **every array
bit for bit**. Nothing in the frozen data moved, so the scenes are the ones they always were.
`gen_goldens_s4_field.py` was rerun too: 26 of its 28 arrays are bit-identical and the two that
differ are exactly the `__stack` pair above. The 2026-10-14 regeneration, with the same code and
the same `s4` env, also moved the 24 field arrays, by at most 1.1e-15 (7.2e-16 of their scale);
the generator's docstring says where that difference comes from.

## The oracles

[`lattice.py`](lattice.py) supplies pointwise discrete reciprocity, an independent
Fourier-pole Green integral, and radiation predicted from the actual source's
analytic cubic stamp and raw drive. It does not call the production surface
transform. [Derivation and witnesses](../../docs/validation/fdtd/fd-lattice-radiation.md).

The analytic sphere and cylinder reference in [`mie.py`](mie.py) requires only
NumPy and SciPy. Its published coefficient pins, independent radial wave equation,
energy and angular checks are described in [Mie validation](../../docs/validation/fdtd/fd-mie-validation.md).

[`mie_fields.py`](mie_fields.py) adds exact exterior E/H for the diagnostic
radiation study. Both Maxwell curls, concentric-sphere power, complex far limits,
translation and an independent surface integral validate its component series.
Its measured truncation scope and source-mutant witnesses are documented in
[the exact-near-field section](../../docs/validation/fdtd/fd-mie-validation.md#exact-mie-near-fields-through-the-radiation-transform).
It does not import the FDTD engine or depend on a captured FDTD answer.

[`gen_star_lift_oracle.py`](gen_star_lift_oracle.py) freezes the FDTD star lift's diagonal
change, with its first and second derivatives along two directions, on four small regions into
`star_lift_oracle.npz`. It is written from the definitions in `src/wavebit/fdtd/_prepare/stars.py`'s
docstring, imports nothing from wavebit, and works at 50 digits: explicit 6x6 stars, bisection to
`2**-150`, an eigensolver certificate and central differences. It needs mpmath, which wavebit does
not depend on; `test_fd_star_lift.py` compares JAX, `jax.jit` and Torch's nested derivatives with
the committed numbers. [The star lift](../../docs/validation/fdtd/fd-subpixel-occupancy.md#the-star-lift-the-smoothed-operator-positive-by-construction-2026-09-27).

| Oracle | Reaches | Generator needs | Golden |
| --- | --- | --- | --- |
| **RETICOLO** (Li/Lalanne lineage) | diffraction *orders*; uniform layers with a complex 3x3 tensor — **but only up to one off-diagonal pair, at classical mounting**, see below | Octave + a RETICOLO V10 tree from its Zenodo record, CC BY 4.0 ([terms below](#what-reticolos-records-and-packages-say-about-its-licence)) (`RETICOLO_PATH`) | `golden_reticolo.npz`, `golden_reticolo_anisotropic.npz` |

| **Meep**, **transverse rectangle sensitivities** | three TE diffraction orders, rectangle width/centre derivatives, two grids, averaged-profile controls and independent RCWA/translation checks | documented 2D Blocks or global material function; two durations, perturbation sizes and averaging tolerances | `golden_meep_rectangle.npz`, `golden_meep_rectangle_fine.npz`, `golden_meep_rectangle_smoothed.npz`, `golden_meep_rectangle_smoothed_refined.npz`, `test_meep_rectangle.py`, `interface/meep_rectangle.py`; [measurement](../../docs/validation/fdtd/fdtd-rectangle-gradients.md) |
| **Meep**, **rotated block sensitivities** | eight complex channels and angle/width derivatives, independent grid refinement with explicit finite-grid gradient limits | documented 3D rotated Blocks; two grids, durations and perturbation sizes | `golden_meep_rotation.npz`, `golden_meep_rotation_coarse.npz`, `test_meep_rotation.py`, `interface/meep_rotation.py`; [measurement](../../docs/validation/fdtd/fdtd-rotation-gradients.md) |
| **Meep**, **graded painted rectangles** | Eight complex channels and angle/relative-position/width derivatives | Independent inclusion-exclusion sinc profile, two FDTD grids, durations/steps and a profile refinement | `golden_meep_painted_shapes.npz`, `golden_meep_painted_shapes_coarse.npz`, `golden_meep_painted_shapes_profile.npz`, `test_meep_painted_shapes.py`, `interface/meep_rotation.py`; [measurement](../../docs/validation/fdtd/fdtd-painted-shapes.md) |
| **Meep**, **graded ellipse sensitivities** | Eight complex channels and angle/axis derivatives | Independent affine Bessel profile, two FDTD grids, durations/steps and a profile refinement | `golden_meep_ellipse.npz`, `golden_meep_ellipse_coarse.npz`, `golden_meep_ellipse_profile.npz`, `test_meep_ellipses.py`, `interface/meep_rotation.py`; [measurement](../../docs/validation/fdtd/fdtd-ellipse-gradients.md) |
| **Meep**, **graded disk sensitivities** | Eight complex channels and a diameter derivative for the translation-averaged disk | Independent Bessel profile, two FDTD resolutions, two durations/steps and a profile-grid control | `golden_meep_circle.npz`, `golden_meep_circle_coarse.npz`, `golden_meep_circle_profile.npz`, `test_meep_circles.py`, `interface/meep_rotation.py`; [measurement](../../docs/validation/fdtd/fdtd-circle-gradients.md) |
| **Meep**, **graded rotated block sensitivities** | Eight complex channels and two shape derivatives for a fixed translation-averaged profile | Public MaterialGrid, independent sinc profile, two FDTD grids and profile grids, duration/step controls | `golden_meep_rotation_smoothing.npz`, `golden_meep_rotation_smoothing_coarse.npz`, `golden_meep_rotation_smoothing_profile.npz`, `test_meep_rotation_smoothing.py`, `interface/meep_rotation.py`; [measurement](../../docs/validation/fdtd/fdtd-rotation-smoothing.md) |
| **Meep**, **adjoint transmission sensitivities** | Two complex co-polarized transmissions and angle/width sensitivities, with explicit finite-grid adjoint differences | Public adjoint API, analytic sinc-profile chain, matching forward differences and time controls | `golden_meep_adjoint_rotation.npz`, `golden_meep_adjoint_rotation_coarse.npz`, `test_meep_adjoint_rotation.py`, `interface/meep_adjoint_rotation.py`; [measurement](../../docs/validation/fdtd/fdtd-adjoint-validation.md) |
| **Meep**, **constrained multilayer design** | three coating weights and three permittivities: complex r/t, six derivatives and separate continuum/grid-refinement checks | documented global material function; two durations, perturbation sizes and averaging tolerances | `golden_meep_multilayer.npz`, `golden_meep_multilayer_refined.npz`, `test_meep_multilayer.py`, `interface/meep_multilayer.py`; [measurement](../../docs/validation/fdtd/fdtd-multilayer-optimization.md) |
| **Meep**, **magnetic half-space sensitivities** | normal-incidence slab complex r/t and four exterior epsilon/mu derivatives; two perturbation sizes and two durations | separately installed Meep, documented API only | `golden_meep_halfspace.npz`, `test_meep_halfspace.py`, `interface/meep_halfspace.py` |
| **Meep**, **internal layer sensitivities** | magnetic film complex r/t and a thickness derivative with fixed output plane; two perturbations and durations, with explicit continuum-error limits | separately installed Meep, documented API only; generator `--parameter thickness` | `golden_meep_layer_thickness.npz`, `test_meep_layer_thickness.py`, `interface/meep_halfspace.py` |
| **Meep**, **moving total depth** | magnetic film complex r/t and depth derivatives referenced to the moving exit; two grids, durations, steps and averaging controls | public global material function; generator `--parameter depth --smoothing .025` | `golden_meep_variable_depth.npz`, `golden_meep_variable_depth_coarse.npz`, `golden_meep_variable_depth_refined.npz`, `test_meep_variable_depth.py`, `interface/meep_halfspace.py`; [measurement](../../docs/validation/fdtd/fdtd-variable-depth.md) |
| **Meep**, **graded internal-interface sensitivities** | magnetic linear epsilon/mu ramp, complex r/t and its translation derivative; perturbation, duration and averaging-tolerance controls | documented global material function in a periodic 3D cell; generator `--parameter thickness --smoothing .0125`, plus `--subpixel-tol 1e-9` for the refined archive | `golden_meep_layer_smoothing.npz`, `golden_meep_layer_smoothing_refined.npz`, `test_meep_layer_smoothing.py`, `interface/meep_halfspace.py` |
| **RETICOLO** (Li/Lalanne lineage) | diffraction *orders*; uniform layers with a complex 3x3 tensor — **but only up to one off-diagonal pair, at classical mounting**, see below | Octave + a licensed RETICOLO V10 tree (`RETICOLO_PATH`) | `golden_reticolo.npz`, `golden_reticolo_anisotropic.npz` |
| **RETICOLO**, swept over `(kx, ky)` | conical incidence across the light line, **including complex (evanescent) angles** | nothing — frozen from a 2024-09-09 study, see its generator | `golden_reticolo_conical_sweep.npz` |
| **RETICOLO**, at a **complex `kx`** | the formal complex-angle path (`kx/k0 = n*sin(theta)`, `Im(kx) != 0`) on interfaces and gratings. Both solvers take Maystre's cut (this one since 2026-10-02). The lossless-top grating cases sit where the earlier rule took the other root, which the test asserts. `R`, `T` per order and `r_0`, 1e-16 .. 4e-15 | Octave + `RETICOLO_PATH` | `golden_reticolo_complex_kx.npz` |
| **fmmax** (Liu & Fan lineage, MIT) | uniform complex tensors, **patterned** tensors, **permeability** | the `fwref` env — see below | `golden_fmmax.npz`, `golden_fmmax_magnetic.npz` |
| **S4** (same lineage, C++) | real tensors, patterned | the S4 extension, already in the project env | `golden_s4.npz` |
| **S4**, the in-cell **field** | the six field components on *uniform* stacks — every reconstruction path, all angles and azimuths, exact at ~1e-15 | same | `golden_s4_field.npz` |
| **fmmax**, the in-cell **field** | uniform stacks exactly (7.8e-16), **x-patterned layers to ~1e-4**, and an **x/y-patterned** one to ~3e-2 on both formulations — the only outside check of a field whose `ky` orders come from the structure | the `fwref` env | `golden_fmmax_field.npz` |
| **RETICOLO**, the in-cell **field** | the *pure TE/TM 1D* path (`field_1d`), which the two rows above never reach — **1e-14 on gratings**, five components of six | Octave + `RETICOLO_PATH` | `golden_reticolo_field.npz` |
| **RETICOLO**, the **conical** field | `field_1d_conical` on a *patterned* layer — an azimuth and a discontinuity at once, which nothing else here reached | same | `golden_reticolo_conical_field.npz` |
| **fmmax**, `Formulation.FFT` | `factorization='laurent'`, the same operator: **identity** on patterned layers -- 6.1e-13 to 3.2e-11 in total and per order, on a rectangular cell, a lossy metal and a hexagonal one -- once fmmax's sampled coefficients are extrapolated to the exact ones | the `fwref` env | `golden_fmmax_laurent.npz` |
| **fmmax**, **double negatives** | gratings on a lossy double-negative substrate (1D and 2D, `\|mu\| != 1`), double-negative pixels inside a grating (patterned `eps` and `mu`), and a grating lit **from** a lossy double negative: every order's unpolarized `R` and `T`, ETM and SMM, through the `theta` door and the direct-`k` door -- **identity**, 2.7e-14 to 6.2e-12, once the quadrature is extrapolated away as for the row above | the `fwref` env | `golden_fmmax_dng.npz` |
| **PyMoosh** 4.0.1 (MIT), **lossless double negatives** | flat multilayers with a lossless double negative as the substrate, as the lit superstrate, as an inner layer, and all three: `r`, `t`, `R`, `T`, TE and TM, four in-plane wavevectors each, on the TE/TM, 2D and first-order paths, ETM and SMM -- 8.8e-15 at worst. The case fmmax cannot arbitrate | a throwaway venv with PyMoosh 4.0.1 -- see its generator | `golden_pymoosh_dng.npz` |
| **RETICOLO**, a **crossed** grating in both of Li's nestings | the 2D Li build, `li_nesting='fixed'` against `parm.res1.li = 1` and `'symmetric'` against `li = 0` — the one case here that can tell the two apart (a 1D grating cannot) | Octave + RETICOLO V10, with `li` set by hand: the bridge does not expose it | inline in `test_reticolo_li_nesting.py`, totals to ten decimals (2026-09) |

| **Meep** (FDTD, GPL) | a **crossed grating in the time domain**: per-order complex amplitudes on a 2D order grid at a conical mount, both incident polarisations. Not RCWA at all, so it is the only entry that can arbitrate `src/wavebit/fdtd/_periodic/legacy.py` — and the only external check `_extract_crossed` has ever had | the `meep` env; **never an install dependency**, and its source is not read — see below | `golden_meep_crossed.npz` |
| **Meep**, **patterned materials in 3D** | selected asymmetric crossed rasters with permeability contrast, electric Drude and electric Lorentz poles; both source polarizations, complex propagating amplitudes at two grids, and a separate doubled-runtime check | `meep` env, public API only; never an install dependency | `golden_meep_crossed_materials.npz`, `golden_meep_crossed_materials_runtime.json` |
| **Meep**, **damped negative-epsilon metal** | a subwavelength rectangular 3D Drude patch; only zeroth order radiates. Preserved grids 30/60, separate grid 120, runtime control at 60; coarse-reference discrepancy is documented | same public-API-only Meep environment | `golden_meep_crossed_metal.npz`, `golden_meep_crossed_metal_fine.npz` |
| **Meep**, **source-free ringdown traces** | uniform Lorentz oscillations and a conductive periodic cavity mode; offline fitted frequencies/decay agree with independently derived discrete poles | same public-API-only Meep environment; ordinary tests read the archive | `golden_meep_ringdown.npz`, `test_meep_ringdown.py`, `interface/meep_ringdown.py` |
| **Meep**, **source spectral zero** | unit-peak delayed-difference source suppresses an undamped Lorentz mode; complex native records match at identical current/field times on all three CPU backends | same public-API-only Meep environment, using `CustomSource` | `golden_meep_source_notch.npz`, `test_meep_source_notch.py`, `interface/meep_source_notch.py` |
| **Meep**, the **1D benchmark** | the grating `docs/guide/fdtd/fdtd.md` §2 publishes and `README.md` leans on — pure TE/TM, normal incidence, both polarisations at two resolutions, plus a **planar control against a closed form**. Not new physics: it *is* that table's Meep column, which until 2026-09-06 was quoted from elsewhere. Grade **~1e-3**, structure not drift | same | `golden_meep_1d.npz` |
| **Meep** + **FDTDX**, **native-grid Yee updates** | slab complex Ex/Hy transfer and power at resolutions 20/40/80 with a runtime control; FDTDX 3D periodic E/H updates and the epsilon derivative | separate Meep and FDTDX interpreters, public API or installed update functions only; frozen by `interface/freeze_grid_references.py` | `golden_native_grid_external.npz`, `test_native_grid_external.py`, `test_native_torch_external.py`, `interface/validate_grid_meep.py`, `interface/validate_grid_fdtdx.py` |
| **Meep** + **FDTDX**, **CPML and electric ADE** | four dispersive slab models at two resolutions, a low-loss mixed metal, and FDTDX two-pole updates with the oscillator-strength derivative | the same interpreters; frozen by `interface/freeze_grid_material_references.py` | `golden_native_material_external.npz`, `golden_low_loss_metal_meep.json`, `test_native_material_external.py`, `interface/validate_grid_dispersive.py`, `interface/validate_grid_fdtdx_ade.py` |
| **FDTDX**, **complex Bloch boundaries** | three-axis phase advance, real and imaginary E/H, and the material derivative | FDTDX interpreter; `interface/freeze_grid_references.py --kind bloch` | `golden_native_bloch_external.npz`, `test_native_bloch_external.py`, `interface/validate_grid_fdtdx_bloch.py` |
| **Meep**, **bottom-incident scene** | two-layer complex r/t with the source physically below the stack | Meep interpreter, public API only | `golden_scene_bottom_meep.json`, `test_scene_bottom_meep.py`, `interface/validate_scene_bottom_meep.py` |

One more oracle lives outside this directory: the Berreman 4x4 in
`tests/rcwa/test_anisotropic_berreman.py` is an independent *formulation* — not RCWA at all — but our
own implementation, so it sits a tier below the rows above. Meep is the only entry that is both
at once: a different formulation *and* somebody else's code.

### One file here is not an oracle: `golden_fdtd_table.npz`

Every row above freezes somebody else's answer. `golden_fdtd_table.npz` freezes **ours** —
`docs/guide/fdtd/fdtd.md` §2's two tables, all three columns, with the conditions they were taken at
stored beside them — and it exists because a published table is a claim with no owner. §2's
"this module" column was taken at `pad_scale=2.0`, `PAD_WAVELENGTHS` was then halved, the
sentence "at the shipped defaults" stayed true-looking, and for three weeks the document
printed a configuration the library no longer had. Nothing caught it because **nothing in the
suite read a number out of a document**: `tests/api/test_docs_contract.py` checks paths, symbols,
URLs and the toctree, never a value.

So `test_fdtd_table_golden.py` parses §2's markdown tables cell by cell and holds every value
to half a unit in its last printed place, re-solves the two columns this environment can, and
— the one that would have caught the original drift on the day it happened — asserts that the
conditions written out **by value** in `_meep_1d_cases.py` are still the ones
`src/wavebit/fdtd/_periodic/legacy.py` ships. That case file is the record now, for both sides of the
comparison: it carries what the harness passes, what it *inherits* (the four largest
determinants of the Meep column are meep defaults the generator never passes, pinned only by
the version string), the derived grid in lengths and counts, and a measured sensitivity table
saying what a deviation costs. A record that names values can be checked against the module;
a record that says "the defaults" is true by construction and can be checked against nothing.

`gen_goldens_fdtd_table.py` writes it, in the project env, and copies the Meep column across
from `golden_meep_1d.npz` with that file's digest so a stale copy is a failing test.

### A second file that is not an oracle: `fd_golden_stage0.npz` (2026-09-18)

The general FDTD kernel at `src/wavebit/fd/` runs one body of code on NumPy, JAX and Torch, and
its fifth gate (`tests/fdtd/test_fd_backends.py`) holds the three to each other on the scenes of
gates 1 to 4 — a PEC box, a PMC box, a periodic and a Bloch box each carrying an exact Yee
eigenmode, and the four half boxes of the symmetry gate — at 1e-12 of each array's peak,
whole volumes and every recorded series. That agreement is **necessary and nowhere near
sufficient**: a defect in the plan moves all three backends together, and three backends that
agree say nothing about a dropped wall mask or a flipped pad sign. So the same test pins
**each backend to what it itself produced** when this archive was frozen — the probe series,
the final E and H at eight fixed cells per component, and one whole-volume energy per scene —
at the same 1e-12. Per backend rather than one shared number, so a red test names the
backend that moved. The gate's mutant is the argument run: the PEC box's plan with its masks
dropped passes the agreement half on all three backends and fails the pin on all three by
O(1) (4e-2 on the energy, 3 to 13 on the series).

What it is not is an oracle for the kernel's correctness. Every number came from `wavebit.fdtd._legacy`,
so it cannot say the kernel is right — gates 1 to 4 do that against closed forms. What it
adds is a drift detector with two properties a bare self-capture lacks: the frozen eigenmode
series are held, in the archive, to the three-term recurrence those gates hold their live
series to, so what was frozen is known to have been a right answer; and the scene behind
every key is shown to be the one the test builds — by the plan's skeleton hash for the static
content and by the sampled initial field for the excitation — so a moved helper reads as a
moved scene rather than as drift.

`interface/freeze_fd_stage0.py` writes it, in the project env with all three backends, from
this checkout and no other (it refuses an imported `wavebit` from elsewhere). The archive
carries the Python, wavebit, numpy, jax and torch versions, the platform, LF-normalised
digests of the freezer and of the case list, and a SHA-256 over its own numeric content that
the test recomputes. The freezer validates that digest before it lets it arbitrate anything:
it freezes **twice in one process and refuses to write unless every array comes back byte
for byte** — the check that a digest depending on the process (an `id()`, an address in a
`repr`, an unseeded draw) fails and a digest that is a function of the code passes. Measured
on the freezing machine: Torch bit-identical to NumPy on every real scene and 2e-15 off on
the complex one, JAX 1e-15 to 4e-14 off both, so the 1e-12 budget is 25x above the worst of
that and 1e3 below anything a wrong sign, pad or mask does. Without `--rebaseline` the
freezer never writes over the committed file: it reproduces the archive and prints, array by
array, what moved and whether the move is inside the budget. **Regenerating is not how a
failing test gets fixed** here any more than it is for the rows above.

### Its stage-1 twin: `fd_golden_stage1.npz` (2026-09-18)

Stage 1 added everything in `src/wavebit/fd/` that is not a wall — a stretched CPML slab with
four psi arrays per face, graded `Absorber` damping, update coefficients built from a
material, and an ADE pole recurrence carrying its own memory — and an eigenmode in a PEC box
exercises none of it. So a second archive pins six scenes of its own, one per feature: a 20-cubed
vacuum box with a PML on all six faces, a 30-degree travelling packet under a Bloch phase (the same
stretch, complex), a dielectric slab between two PMLs, a Drude + Lorentz half-space running
out through an `Absorber` face, a conductive slab at 20 degrees, and an `Absorber` box with no PML
at all. `tests/fdtd/test_fd_stage1_backends.py` holds the three backends to each other and each to
its own frozen answer at the same 1e-12, and its two mutants are the argument: a plan with its
CPML faces dropped, and a pole term whose `drive` is zeroed, both keep all three backends
agreeing and put every one of them O(1) from its golden.

Two things are new beside the fields and the series. The **lowered numbers are frozen once per
scene** — the four update coefficient arrays, every CPML profile (`b`, `a`, `1/kappa`) and
every pole's recurrence coefficients — because they are backend-independent by construction,
so a failure that touches only those rows says the layer was regraded and one that touches
only the per-backend rows says the step changed under an unchanged layer. And the scenes are
built so that **each one empties itself**: the leapfrog conserves the discrete divergence of
E exactly, so an initial field with a longitudinal part leaves a static charge sitting in the
box that no absorber could ever remove (measured: 25% of the energy, on a plain Gaussian). The
cube is driven by the curl of a scalar blob, the oblique lines by an explicit superposition of
travelling eigenmodes, and the test then holds the frozen energy against the same box with
PEC in place of every absorbing face — 2e-3 of it in the cube, 5e-2 at 30 degrees where the packet's
own band reaches grazing and gate 1's table says a 12-cell layer reflects 3e-2.

`interface/freeze_fd_stage1.py` writes it under exactly the stage-0 rules above: this checkout
only, twice in one process, LF-normalised source digests (its own, its case list's, and the
stage-0 case list it imports its capture from), a content digest the test recomputes, and
`--rebaseline` as the only way to overwrite. It adds one line its predecessor has no need of:
an array whose **shape** moved is reported as a changed scene rather than compared, because
comparing it raises instead of reporting.

### Its stage-2 twin: `fd_golden_stage2.npz` (2026-09-19)

Stage 2 added the first things in `src/wavebit/fd/` that *drive* the grid rather than start from an
initial field -- a waveform table built inside the traced function, a dipole's stamp added into
the curl, and an incident column carried in a term's memory that a TFSF box's six faces and a
PlaneWave sheet read their corrections from -- so a third archive pins four scenes of its own:
an electric (B-spline) and a magnetic (tent) dipole in a PML box, a 20-degree p `PlaneWave` in a
Bloch cell over a glass layer (complex), a `+x` TFSF box in vacuum, and a `-z` TFSF box on a
glass half-space that crosses it. `tests/fdtd/test_fd_stage2_backends.py` holds the three backends to
each other and each to its own frozen answer at the same 1e-12 (measured: Torch bit-identical
to NumPy on both TFSF scenes and 3e-15 off on the other two, JAX 8e-16 to 4e-15), and its two
mutants are the argument: a stamp normalised by its sum of squares and one TFSF face with its
sign flipped both keep all three backends agreeing and put every one of them O(1) (1.9 and 1.2)
from its golden.

Two things are new. **Every source's reference series is frozen**, per backend, beside the
fields -- the incident E and H a sheet or a box records, the drive, work and stamp-weighted field
of a dipole -- because stage 3 normalises by them, and a moved reference is a moved answer even
when the field is unchanged. And **the samples are not the stage-1 cells**: those sit at the
ends of the long axis and inside the absorbing layers, which in a TFSF scene is all scattered
region, exactly 0.0 on NumPy and Torch and 1e-16 on JAX. The first freeze of this archive
pinned that round-off on all eight final-field arrays of both TFSF scenes (a harmless
reordering would have moved them by O(1) of their own peak); the stage-2 picks cluster on the
domain centre, inside every box and next to every dipole. What the frozen numbers are known to
be right about is asserted against closed forms rather than left to the digest: the drive is
`Band.waveform` at the slot's half step on every backend, the stamps are partitions of unity
centred on the dipole, the Bloch pattern is `exp(i k_par . r)` with the `k_par` of
`Bloch.from_source`, and the TFSF runs behind the pins leave the scattered region at round-off.

**`tfsf_substrate` was re-frozen once, on purpose (2026-09-19).** Its first freeze pinned a
known defect: the layered TFSF column was driven by a hard source, which reflects the
substrate's echo back down the column (`r = -1`), and the glass top sits one cell inside the
6-cell box, so every array of that scene carried the echo -- and a self-captured golden agrees
with whatever produced it. The fix (a uniform driver column feeding the layered column through
a one-sided sheet, `src/wavebit/fdtd/_kernel/sources.py`) moved that scene's arrays by 13-29% of
their peaks and nothing else: the other three scenes reproduced bit for bit, their skeletons
changing only because `ColumnTerm` and `Correction` gained fields. What says the new numbers
are right is `tests/fdtd/test_fd_tfsf.py`: the box interior over a substrate is the
`PlaneWave`-driven physical field to round-off at every step, and the discrete Airy field.

**`plane_wave` was re-frozen once, on purpose (2026-09-19).** Its first freeze pinned a launch
defect: the PlaneWave column's hard source wrote the continuum polarisation into the stored
`E_x` and `E_y`, without the half-cell stagger phase `exp(i k_c dx/2)` each carries along its
own axis and not transverse to the grid's discrete `K_par`, so this conical (`phi = 30`) p wave
was launched with an s part first order in `dx` (an empty conical cell at 20 cells per
wavelength transmitted `|t_p| = 9.1e-3` for an s wave). The drive now writes the grid's own s
or p wave (`src/wavebit/fdtd/_prepare/sources.py::plane_wave_drive` derives it). That moved this
scene's fields, `Record` series and reference series by 2.7e-2 to 4.7e-2 of their peaks and its
energy by 7.9e-6, and nothing else: the lowered pattern and waveform rows and the other three
scenes (116 of 140 arrays) reproduced bit for bit, and the stage-0 and stage-1 archives did too
(143 of 143, 288 of 288). The skeleton changed because the drive weights are now complex
(a static plan now encodes a Python `complex`). What says the new numbers are right is
`tests/fdtd/test_fd_planewave.py` (the forward field is the closed form with the grid's own
polarisation, and the incident field carries no trace of the other one at `phi = 35`) and the
leak tests of `tests/fdtd/test_fd_axis1d.py` and `tests/fdtd/test_fd_diffraction.py` (1e-16 where they read
1e-3 to 1e-2).

`interface/freeze_fd_stage2.py` writes it under exactly the stage-0 and stage-1 rules: this
checkout only, twice in one process, LF-normalised source digests (its own, its case list's,
and the stage-0 case list it imports its capture from), a content digest the test recomputes,
a changed shape reported as a moved scene, and `--rebaseline` as the only way to overwrite.

### Its stage-3 twin: `fd_golden_stage3.npz` (2026-09-19)

Stage 3 added one kernel observer -- a running DFT of named components over a region, carried
in the scan (`DFTRegion`) -- and made everything else post-processing of its accumulators after
the run: flux over a plane or a box's six faces, `Field` phasors, the near-to-far transform,
the `Diffraction` orders (index-DFT, direction split, Yee kz de-embedding, s/p frame), and the
sources' spectra (`power`, `incident_amplitude`, `intensity`). It also added the cell-snapped
box (`fd.Region` since stage 4 C4, `fd.Box` before) as a structure and the `Axis1D` face. So a fourth archive pins five scenes of its own: two dipoles in
a PML box with a `Flux` box, a `Flux` plane and a `Field` plane; a 25-degree p `PlaneWave`
through a glass layer in a Bloch cell with a reflect and a transmit `Diffraction` plane; a
lamellar `Region` grating at normal incidence with three propagating orders; a magnetic and an
electric dipole in a `FarField` box; and a conical `Region` grating whose invariant axis is an
`Axis1D`. `tests/fdtd/test_fd_stage3_backends.py` holds the three backends to each other and each to
its own frozen answer at the same 1e-12 (measured: JAX and Torch 1e-15 to 3e-14 off NumPy, the
worst a dipole's `power()`), and `tests/fdtd/test_fd_stage3_grad.py` holds `jax.grad` and
`torch.autograd` to central differences through each new route.

What is frozen is the pipeline end to end, per backend: **every DFT accumulator whole** and its
energy series (the raw observer output), then **every result built from them** (fluxes and face
shares, phasors, the ten `Result` arrays of each `Diffraction` monitor and its `theta_at`, the
far-field intensity and fields at fixed directions), then the sources' spectra. A failure in the
result rows alone says the post-processing moved under unchanged accumulators, and the gate's
mutants show that this is not hypothetical: the continuum kz in the de-embedding and the far
field's magnetic current with its sign flipped both leave every accumulator bit-identical and
move only the complex amplitudes (1e-2 of their peak) or the far field (O(1)). Not frozen: the
decay ratios, whose relative round-off on a settled run is 1e-7 of themselves (a difference of
nearly equal energies), and the stage-2 reference series, which `fd_golden_stage2.npz` pins on
its own scenes and which would be 60% of this archive; the gate holds the backends to each
other on those live.

Two traps met on the way are asserted rather than remembered. **A lone `Ez` dipole radiates no
`Hz`** (its field is TM to z), so the first draft froze four box-face `Hz` accumulators of pure
round-off -- JAX and Torch 1.0 to 1.8 of their own peaks off NumPy, the stage-2 trap exactly; each
dipole scene now carries a second dipole, and the gate asserts every frozen array is either an
exact 0.0 or above 1e-10. **The lamellar grating is TE by symmetry**, so its p arrays and its
`Ex`/`Hy` accumulators *are* exact zeros on every backend: frozen, and asserted exact, because a
pin on an exact zero is one any leak of p breaks. The runs are short (300 to 500 steps), so every
frequency-domain monitor warns `UnsettledFieldWarning`; the gate asserts that it does, naming
each monitor, rather than silencing it unseen.

The two dipole scenes' boxes are 0.4 on a side, not the first freeze's 0.2: a flux box must
enclose a dipole's stamp (two cells either side of its centre) with a cell to spare, which
`src/wavebit/fdtd/_prepare/monitors.py` now refuses otherwise, and the 0.2 boxes cut both scenes'
stamps -- the first freeze pinned a flux that missed part of the power. They were re-frozen with
`--rebaseline` for that reason alone (only those two scenes moved; 582 of 582 arrays reproduce
since).

**`bloch_stack` and `conical` were re-frozen (2026-09-19)** for the PlaneWave launch fix that
re-froze the stage-2 `plane_wave` scene above: both are conical p waves, and both had been
frozen with the launch's first-order s part in them. Moved, as a fraction of each array's
peak and the same on every backend: `bloch_stack`'s accumulators 2.8e-2 to 5.9e-2, sampled
fields 5.1e-2 / 5.7e-2, `incident_amplitude` 5.1e-2 (the stagger phases), and its energies,
fluxes, `R`, `T`, `R_p`, `T_p` and `intensity` 8e-6 to 9.5e-5, while its complex `r_p` and
`t_p` moved 4e-16 (the normalisation by the column divides the launch out); `conical`'s
accumulators 5.9e-4 to 2.2e-2, fields 7.4e-3 / 1.0e-2, complex amplitudes 1.9e-3 to 3.9e-2,
efficiencies 1.8e-5 to 2.6e-3, `incident_amplitude` 5.1e-3. The three other scenes reproduce bit
for bit (414 arrays). **Four arrays left the archive:** the stack's `r_s`, `R_s`, `t_s` and
`T_s` were the defect itself (`|r_s| = 4.3e-3`, `|t_s| = 8.8e-3`, `R_s = 1.8e-5` -- the
smallest array the round-off line was calibrated on) and are round-off now, because a layered
cell couples none of the grid's own s into its own p. A pin on round-off is the trap this
archive's round-off line exists to refuse, so `_fd_stage3_cases.cross_polarised` leaves them
out of the capture of a `Region`-free cell and `tests/fdtd/test_fd_stage3_backends.py` holds them live
under 1e-13 of the co-polarised amplitude on every backend instead; 570 arrays are frozen.

`interface/freeze_fd_stage3.py` writes it under exactly the earlier stages' rules: this checkout
only, twice in one process, LF-normalised source digests (its own, its case list's, the stage-0
case list its digest and gap come from, and the stage-2 one its sampling rule comes from), a
content digest the test recomputes, a changed shape reported as a moved scene, and
`--rebaseline` as the only way to overwrite.

**`PML(stabilize=)` moved nothing in any of the four fd archives (2026-09-19).** The PML's
stabilizing loss -- a matched loss on the component normal to the face, which stops a bound mode
near the light line from growing under a PEC-backed stretch (`tests/fdtd/test_fd_pml_stability.py`)
-- is opt-in, and a pure-stretch face lowers to the same `CpmlFace`, the same skeleton and the
same arithmetic as before it existed (the kernel skips the loss on the face's static type, and
`params['cpml_loss']` is empty). All four freezers in compare mode, after the change: stage 0
143 of 143, stage 1 288 of 288, stage 2 140 of 140, stage 3 570 of 570 arrays bit for bit, no
rebaseline. Measured with the loss on by default instead, every PML scene of stages 1-3 moved
(`final__E` by 2e-2 to 0.34 of its peak on the stage-1 PML scenes) -- and so did oracle gates
the goldens do not arbitrate, which is why it is not the default
(`src/wavebit/fdtd/_prepare/cpml.py::stabilizing_rate`).

### Why the field row exists, and what it cost to not have it

Until 2026-08-14 every oracle here checked efficiencies or amplitudes, and **field
reconstruction had none at all**. Its only reference was a golden captured from this project's
own earlier implementation — a change detector wearing an oracle's name. What it froze was a
wrong sign convention (`Ez`, `Hx`, `Hy` in the frame whose `+z` points *out* of the stack, on
samples that ran *into* it; the frame declared then was `+z` into the stack, and the one
declared since 2026-09-21 is `+z` out of it with the samples running bottom to top — the
defect was the mismatch, not either direction), and
the suite stayed green for as long as the file existed, because a self-captured golden agrees
with whatever produced it. It is gone; `golden_s4_field.npz` replaces the claim it was making,
and `tests/fixtures/legacy_rcwa/golden_field.npz` keeps the drift-detection job under its own honest name.

S4 stops at **uniform** stacks. Conventions and frames are decided there and neither solver
truncates, so that comparison is exact rather than converged — but on a patterned layer the two
disagree at 1e-3 to 1e-2, and up to 7.6e-1 on the discontinuous `Ez`.

**fmmax reaches the patterned case, and the reason is that its two relevant knobs are
reachable.** Both were found by measurement, on 2026-08-15, and with either one wrong the
grating cases sit at 1e-1 and read as a solver disagreement:

| what had to match | wrong | right |
| --- | --- | --- |
| the **order set** — fmmax's parallelogramic 5x5 against our `truncation_order` | `truncation_order=(8,0)`, 17 orders: **1.2e-1** | `truncation_order=(2,2)`, the same 25: **1.9e-5** |
| the **factorization** — fmmax's `Formulation` | `FFT`, the plain Laurent rule: **2.7e-1** on p | `JONES_DIRECT`: **7.2e-5** |

That leaves ~1e-4 rather than 1e-13, because the two vector factorizations still differ in
detail. It is an oracle for *structure*, not for drift: every sign, transpose and
component error this repository has actually shipped moved fields by 1e-1 or more. Slow
numerical drift is `tests/fixtures/legacy_rcwa/golden_field.npz`'s job.

One further thing fell out of the calibration and is worth having as a statement rather than a
constant: fmmax's unit **mode** amplitude relates to our unit **flux** incidence by exactly
`-cos(theta)` for s and `-1` for p — times `exp(-1j k0 sin(theta) P/2)` since 2026-09-27, the
phase between the two codes' unit points: fmmax's incident wave is unit at its cell corner, ours
at the cell centre. The test asserts that dependence, phase included, so a case added at a new
angle is checked rather than accommodated.

**The grating above varies along x only, and that turned out to matter** (2026-08-15). Its
reconstructed field is y-constant to 2.5e-15, so no `ky != 0` order carries amplitude and the
`Ky` terms of `Ez` / `Hz` multiply nothing. The same golden therefore also carries an
x/y-asymmetric **crossed patch** whose field varies in y, at two truncations (49 and 361 orders)
so that its floor can be shown not to shrink, and both formulations are compared against it:

| | agreement | reaches |
| --- | --- | --- |
| `grating` (x only) | 1.9e-5 s, 7.2e-5 p | the transverse assignment; a wholesale `Kx`/`Ky` exchange |
| `patch` (x and y) | 2.2e-2 to 3.0e-2, the same at 49 and 361 orders and on both formulations | also the `Ky` terms on orders the structure creates, and which end of the raster is `-y` |

3e-2 is a much weaker oracle than 1e-4, and the test says so: it is the two vector
factorizations disagreeing near the rectangle's corners, which is why more orders do not help.
What it buys was measured on 2026-10-14 by mutating the field assembly. Reflecting the `Ky`
entries of the structure's orders about the incident one, in either formulation's copy, fails
the patch cases and no other outside golden (S4's field cases are uniform stacks, RETICOLO's
conical cases have one `ky` row); only the internal comparisons in `tests/rcwa/` also see it. A
`Ky`-only sign flip, by contrast, is caught by the S4 and RETICOLO conical field goldens as well,
because the incident `ky` is not zero there.

Two conventions had to be measured rather than assumed, and both are recorded in
`gen_goldens_s4_field.py`: S4's p amplitude is a **global** `-1` against ours (not a frame
parity — it moves all six components together), and the comparison must sample layer
**interiors**, because `Ez` is discontinuous at an interface and the two codes may legitimately
report different one-sided limits there.

### The 1D field row: the pure TE/TM path, and the strongest of the three

Added 2026-08-15. S4 and fmmax both reach field reconstruction through the *coupled*
6-component routes. Neither exercises `src/wavebit/rcwa/_engine/fields.py::field_1d` (`grating_type == 0`), which
returns 3 components in its own layout — TE `(Ey, Hx, Hz)`, TM `(Hy, Ex, Ez)` — and applies its
own signs. RETICOLO returns exactly those 3 for a 1D grating, so nothing is padded or discarded.

It is the best of the three, **including on gratings**:

| | uniform | patterned |
| --- | --- | --- |
| S4 | ~1e-15 | not reachable (1e-3, and 7.6e-1 on `Ez`) |
| fmmax | 7.8e-16 | ~1e-4 |
| RETICOLO | ~1e-15 | **6.3e-14**, on five components of six |

That is not luck, and the reason bounds the claim. RETICOLO and this project share the 1D
formulation exactly; the bridge hands RETICOLO an analytic piecewise-constant texture while the
solver is given `fourier_coefficients='integrated'`, so both assemble the *same* convolution
matrices. What 1e-14 then establishes is the implementation and every convention around it —
frame, component layout, z direction, amplitude normalisation, order-index sign — which is what
a field oracle is for. It does **not** independently confirm the formulation choice; nothing
sharing a formulation can. Read it beside the fmmax row rather than instead of it.

**The sixth component is a real exception.** TM's `Ex` is discontinuous across a grating's
vertical interfaces, and the two codes resolve it differently at any finite truncation:
7.0e-2 to 1.0e-1 across the committed cases at `truncation_order=8`. Measured on the `grating` case at 24
degrees, it *decreases* with truncation but **not monotonically** — 3.1e-1, 2.3e-1, 9.7e-2,
1.1e-1, 3.8e-2, 4.4e-2 at `truncation_order = 2, 4, 8, 16, 32, 64` — about 8x over a 32x increase in orders,
and nowhere near machine precision. That shape is Gibbs ringing rather than convergence
failure: the gap is set by the samples nearest a discontinuity (10 nm from one here), and the
local error there oscillates with the number of retained terms. Two things it is *not*: a
rescaling (a per-component complex fit does not remove it), and a disagreement about which side
of the interface to divide by — forming `Dx = eps*Ex` leaves a gap of the same order
(1.7e-1, 1.6e-1, 8.1e-2, 4.8e-2 at `truncation_order = 4, 8, 16, 32`), neither better nor worse.

The test excludes that one slot on patterned stacks and bounds it **from both sides**, so the
exception can neither grow nor quietly disappear. Whether either code's `Ex` is closer to the
true field is *not* established here — settling that needs a reference finer than either, and
nothing in this directory provides one.

Three things had to be measured, each worth O(1) when guessed wrong:

| what | wrong | right |
| --- | --- | --- |
| **which rows are the layer** — `res3` emits `npts` rows per profile entry *including* the zero-thickness ambients, so a one-slab cell has 8 rows at `z == 0` | align on the nonzero `z` values, which is what `z` alone invites: **O(1)** | entry `k` owns rows `k*npts:(k+1)*npts`; `res3`'s third output names them: **1e-14** |
| **which way z runs** — RETICOLO's `+z` points back *out* of the stack | reverse each layer in place: **1.3e0** | reverse the whole cell, layer order included: **2.8e-14** |
| **the component parity** the reversal carries | none, or a hand-applied `.conj()` | `Ez, Hx, Hy` flip / `Et, Hz` do not — the rule `src/wavebit/rcwa/_engine/fields.py` already states |

Those were measured against this library's frame of the day, `+z` down into the stack. Since
2026-09-21 the public frame is RETICOLO's — `+z` up from `z = 0` at the bottom interface, bottom
layer first — so RETICOLO's raw cell and the public cell agree entry for entry (after the
conical transpose) on every slot held exactly: `Ex` on patterned stacks is excepted, TM's on
the pure TE/TM path and both polarizations' on the conical one. The tests still compare in the
top-to-bottom depth frame their case lists are written in, reversing both cells with the same
parity, which cancels.

**What unblocked all of this was that the wrapper was throwing away two of `res3`'s three
outputs.** `[e,z,o]=res3(...)` and only `e` reached Python, which is very likely why this
comparison had defeated earlier attempts: `z` cannot locate a layer on its own, and `o` — the
refractive index at every sample point — can. `interface/reticolo_res3.m` now forwards both
(appended, so callers asking for the first 9 are unaffected). `o` earns its keep four times
over: it settles the **x origin** (RETICOLO's docs describe a texture as running
`-period/2 .. +period/2`, and a correspondingly dead `grid_x -= period/2` sat in the bridge —
but `o` reproduces our own structure on `linspace(0, period, sample_x)` entry for entry, so the two
origins coincide as written and there is no shift; that line is now gone, with the measurement
in its place), it is what the generator asserts the row rule against, on the `lossy` case it
comes back as `2+0.05j` — RETICOLO spelling loss the same way this project does — and since the
permittivity flip it is also what checks the **quantity**: `o` is an index and our stacks are
`eps`, so `test_the_patterned_stacks_are_actually_patterned` closes only if the bridge really
took the root, and it fails by O(1) before any field is compared if it did not.

With the frame transform applied there is **no free amplitude at all**: not a fitted scale as
the fmmax golden needs, but exactly 1, to 2.7e-14 — after one stated lateral phase between the
two codes' unit points since 2026-09-27, `exp(-1j k0 sin(theta) P/2)`: RETICOLO's incident wave
is unit at its `x = 0`, the cell corner (the `o` measurement above), and ours at the cell centre.
The test asserts that rather than fitting it, because a scale free to absorb a factor absorbs a
defect just as happily. For TM that holds because every case is lit from air: RETICOLO's 1D TM
input (`einc`, `PlaneWave_H(2)` in its examples) is the incident `H_y`, while this project's pure
TM solve is lit by unit incident **E** since 2026-09-27, like every other path. A case lit
through glass needs `x n_inc/mu_inc` on RETICOLO's side, and that factor is the conversion, not a
regression.

The case list is in `_reticolo_field_cases.py` and covers uniform, grating, two-layer and
absorbing stacks at 0/24/41 degrees in both polarisations. The geometry deliberately keeps
every x sample **off** a permittivity jump: `Ex` is genuinely two-valued there, so a sample
sitting on one is a harness artefact worth 4e-1 that would drown the real story above.

### The conical field row: the same calibration, and that is the result

Added 2026-08-15, one grating type along from the row above. `field_1d_conical` is a 1D grating
at a **non-zero azimuth**: s and p no longer decouple, and one coupled mode set produces all six
components. S4 runs that path but only on uniform stacks; fmmax reaches a patterned layer only
through the 2D route. So a patterned layer *on the conical path* — an azimuth and a
discontinuity at once, where the s/p mixing and the factorization have to be right together —
was checked by nothing outside this repository.

It lands in the same two tiers, measured over 24 cases (4 stacks x 3 angle pairs x s/p):

| | agreement |
| --- | --- |
| every component both codes resolve | **9.8e-14** |
| fitted amplitude, departure from 1 | **4.2e-14** |
| `Ex` on a patterned layer | 8.6e-2 to 1.2e-1 — the exception, bounded from both sides |

**The finding is that nothing conical-specific had to be discovered.** The transform established
on the pure TE/TM path — whole-cell z reversal plus the declared frame's z-parity, no
conjugation, no scale — transfers unchanged, and so does the one stated lateral phase between
the two codes' unit points, `exp(-1j k0 sin(theta) (cos(phi) + sin(phi)) P/2)` on the `(P, P)`
cell at an azimuth. A full search over all 720 component permutations x
z-flip x conjugation confirms the identity permutation as the unique best fit, which is what
says there is no hidden frame quirk rather than merely no *observed* one. `Ex` is the slow slot
again, for the same reason: it is normal to the grating's vertical interfaces and therefore the
discontinuous component.

The one genuinely new fact is the array shape, and it was measured: `res3`'s conical branch
returns `(z, x, y, 6)` — **x before y**, the transpose of this project's `(z, y, x, 6)` — with
`y` of length 1 because the `.m` sets `y = 0` for `grating_type == 1`.

Two structural facts came free with the oracle and are pinned:

* **At `phi = 0` the conical solve decouples exactly.** s leaves `(Ey, Hx, Hz)` and p leaves
  `(Ex, Ez, Hy)`; the other three are not small, they are *identically* zero — ours returns
  `0.0` and RETICOLO agrees. That is the one azimuth where the answer is independently known,
  so it is where the s/p rotation is actually checked rather than merely reproduced.
* **The two RETICOLO goldens agree with each other at `phi = 0`**, to 1e-12. Two different
  RETICOLO code paths (`grating_type` 1 and 0) on the same structure, compared without the
  solver involved at all — which checks both harnesses, both row rules and both array shapes
  against something that is not this project.

### fmmax's FFT formulation: identity, after two corrections

Added 2026-09-27 with `factorization='laurent'`. Every other patterned comparison here is one of
convergence, because the two solvers close the products differently. This one is not: fmmax's
`Formulation.FFT` puts `[[eps]]` in both transverse slots and solves against `[[eps]]` for the z
slot, which is exactly what `'laurent'` builds. Two things still separate the numbers, both found
by measurement and both removed rather than tolerated:

| what | size | how it is removed |
| --- | --- | --- |
| **fmmax's DFT is the midpoint rule** for the coefficient integral Wavebit evaluates exactly on each pixel. Per axis the ratio is `y / sin(y)`, `y = pi k / (N R)`, even in the samples per pixel `R` | 1e-4 at `R = 16`, falling fourfold per doubling | the generator freezes `R = 16, 32, 64, 128` and the test extrapolates (Richardson, through `1/R^4`), after checking the fourfold fall at every level |
| **fmmax 1.7.1 forms its pixel-centring phase in single precision** (`fmmax.fft._fft2` asks `fftfreq` for float32), which turns an exact translation into an order-dependent one | 1e-8 to 1e-7 in the efficiencies; 3.8e-8 in `[[eps]]` entries of size 4 | a second run in which fmmax's `fft` module sees a float64 `fftfreq` -- nothing of fmmax's is rewritten -- and the stock run is still held, within twice the gap between the two |

What is left is round-off: 6.1e-13, 3.2e-11 and 4.8e-13 in total, and the same over every order of
both polarizations, where the mirrored order index misses by 1.3e-2 to 7.4e-2 and Li or NVF by
4e-2 to 2e-1. The order set is handed to fmmax as an explicit `Expansion`: its own
parallelogramic truncation balances the sides in k-space, which on the 700 x 560 cell keeps
17 x 13 orders where `truncation_order=(7, 7)` keeps 15 x 15.

### Double negatives: what fmmax can arbitrate, and what it cannot (2026-10-04)

`golden_fmmax_dng.npz` is the outside check on negative-index media (`docs/theory/rcwa/branch-cuts.md` §9:
the mirror of Maystre's cut, the signed index, the meaning of `theta` on a lit negative-index
side). fmmax makes none of this solver's choices. It keeps each layer's modes with
`Im(kz) > 0`, which is the physical root for **every lossy medium, of either sign of index** --
limiting absorption needs no sign of `Re(n)` once there is loss -- and the wrong one, or a
round-off one, for a lossless double negative. So every negative-index medium in the golden is
lossy and has `|mu| != 1`; the lossless limit is pinned elsewhere (closed forms and thick lossy
slabs, `tests/rcwa/test_negative_index_half_space.py`).

Two things make the comparison one of identity, and both were inherited rather than invented:
the generator freezes four samplings per pixel and two flavours of the centring phase exactly as
the FFT row above does, and the quantity frozen is each order's **unpolarized** efficiency,
`trace(H_in^-1 H_out)/2`, so that the arbitrary polarization mix in which fmmax's general
eigensolver returns a lossy double negative's two order-0 modes does not have to be matched to
this solver's s and p. Measured at freezing (fmmax 1.7.1, jax 0.10.2, numpy 2.4.6): the raw gap
falls by 4.00 per doubling of the samples; the extrapolated gap, worst order per case, is 2.7e-14
to 6.2e-12 against tolerances (the extrapolation's spread plus 1e-12 of round-off) of 2.7e-12 to
3.9e-10; the stock run sits 2.3e-9 to 3.9e-9 away (its float32 phase);
Li's rules miss by 8e-3 to 4e-2. The mirror cut switched off fails every case, an unsigned
`k_par` fails the lit case through the `theta` door, and develop fails 32 of 35 tests. Rerunning
the generator reproduces all 73 arrays byte for byte.

What it cannot arbitrate is ETM's *labelling* of a double-negative layer's modes. On these lossy
scenes either label rule gives the SMM answer to 1e-15 -- a label moves conditioning, not the
answer -- and the scenes where the labels decide a `LinAlgError` are lossless, where fmmax's own
choice has no loss to read. Those are held to SMM and closed forms in
`tests/rcwa/test_negative_index_layers.py`.

### PyMoosh: the lossless double negatives fmmax cannot read (2026-10-04)

`golden_pymoosh_dng.npz` is the outside check for **lossless** double negatives in flat stacks.
fmmax keeps every mode with `Im(kz) > 0`, which has nothing to read on a lossless order;
PyMoosh (D. Langevin et al., *J. Opt. Soc. Am. B* **41**, A67 (2024)) writes its choice down in
`coefficient_S`: the principal `kz`, negated in a lit or last medium with `Re(eps) < 0` and
`Re(mu) < 0` (the last one only where `Re(kz) != 0`), and inner layers flipped to
`Im(kz) >= 0`. On every lossless medium that is this solver's root -- Maystre's cut, or its
mirror for a double negative -- and the two codes share nothing else (2x2 scalar scattering
matrices against an RCWA stack at `truncation_order = 0`).

Two of its conventions decide how the comparison is set up. **Its angle is the phase's**:
`alpha = sqrt(eps*mu)*k0*sin(theta)` with the principal root, so from a double negative its `kx`
has the sign opposite to the one this solver's `theta` (the light's direction) gives. The golden
holds PyMoosh's `kx`, which this solver takes through its `kx` door; a flat stack answers both
signs alike, so the sign of a lit double negative's `k_par` is the fmmax golden's to check, not
this one's. **Its TM amplitudes are `H_y` ratios**: this solver's `r_p` is that ratio and its
`t_p` the `E` ratio, `t_H (n1/mu1)/(n2/mu2)` with the signed index, which the test undoes with an
index it writes out itself. With those, every row agrees to 8.8e-15 (numpy 2.5.1 here, 2.5.3
in PyMoosh's environment); a double negative beyond the lit side replaced by its positive mirror
misses by 1.8 to 2.9. A lossless lit double negative has exactly its mirror's admittances, so
no flat stack can tell those two apart; what that case pins is the incident wave's root and the
sign of the index in `t_p` (reverting the mirror cut or the signed index fails the test; an
unsigned `k_par` and the layer labels do not, as said).

**Where PyMoosh is not an oracle**, found while choosing the cases, both from its inner-layer
rule: a lossless double-negative layer keeps the principal root, whose power runs against the
label, and wherever that admittance cancels a neighbour's the interface matrix divides by zero.
A `(-3, -0.5)` layer on a `(-3, -0.5)` substrate returns `NaN` at every angle (the stack this
solver's own label rule was fixed for, `tests/rcwa/test_negative_index_layers.py`); a `(-6, -1.5)`
layer beside `(4, 1)`, which have the same impedance, returns `R + T = 3.10` at normal incidence
and `1 + 2.6e-10` at 1e-3 rad. The cases avoid both, and the generator refuses any row with
`|R + T - 1| > 1e-12`. Rerunning the generator reproduces the file byte for byte.

### The Meep row: the first oracle here that is not RCWA, and what it took

Added 2026-09-06. Every other row truncates a Fourier series and inherits whatever
factorization that forces, which is the formulation `src/wavebit/rcwa/_engine/engine.py` uses — so none of them
can arbitrate `src/wavebit/fdtd/_periodic/legacy.py`, whose errors are numerical dispersion, a staircased boundary and
an imperfect absorber instead. Meep is an FDTD. Nothing is shared but Maxwell's equations.

Two gaps close at once, and the second one is wider than it looks:

* **a crossed grating solved in the time domain.** `_extract_crossed` is a line-for-line copy
  of `_extract_conical`, and every per-order sign in it — the three `sigma` legs, the
  `s_hat`/`p_hat` frame, the `n / kz_m` secant — is invisible to `R`, `T` and `R + T`. Its only
  arbiter was `tests/fdtd/test_fdtd.py`, against **wavebit's own** RCWA path, which shares those
  conventions with it by construction: a convention wrong in both moves nothing.
* **per-order complex amplitudes on a two-dimensional order grid, for either solver.**
  `docs/theory/rcwa/amplitude-sp-frame.md` §2 names what every golden here compares — efficiencies, or
  fields, or 1D amplitudes — and concludes that the per-order s/p half-turn "could not have
  been reached" by any of them. Both cells below have propagating orders on both sides of
  `kx = 0`, which is what makes that reachable, and `test_the_half_turn_half_plane_is_covered`
  is what keeps it so as cases are added.

**Nothing in the map is fitted.** The axis map is the *identity* — a 3D meep cell has three
real axes, so unlike the 2D bridge nothing is permuted. Meep's DFT phasor is `exp(+i k z)` for
a wave travelling `+z`, which is wavebit's `e^{-iωt}`, so there is no conjugation. The frame is
`s_hat = z_hat × k_hat_par` and `p_hat = k_hat × s_hat`, and `E · k_hat = 0` collapses the p
projection to `∓ (n / kz)(c Ex + s Ey)` with the upper sign downgoing — which *is*
`_extract_crossed`'s `sigma = (-1, +1, -1)`, so those three signs are geometry rather than a
convention either side chose. `test_the_leg_signs_are_the_ones_the_generator_derived` pins the
two spellings together.

**The one free factor is an origin, and it is measured rather than assumed.** wavebit places a
raster from the cell **centre** (`wavebit.meshgrid_xy` on a 7-column cell of period 0.84 returns
`-0.36 … +0.36`, and the meep blocks are built on the same rule) but references each order's
amplitude phase from the cell **corner** — its demodulation kernel runs over cells `0 … n-1`,
in both solvers. Normalised by the incident order the two differ by
`exp(-i(k_m - k_inc)P/2)` on each axis, which is `(-1)^m (-1)^l`. Applied, not fitted, and
scored against four alternatives on the square cell at a deliberately coarse resolution 25:

| rule | worst amplitude gap against wavebit's RCWA |
| --- | --- |
| identity | 7.2e-1 |
| `(-1)^m` alone | 4.6e-1 |
| `(-1)^l` alone | 7.2e-1 |
| **`(-1)^(m+l)`** | **3.9e-2** |
| `exp(-i k_m P_x/2) exp(-i k_l P_y/2)` (the whole field, not the envelope) | 8.2e-1 |

That is a 12x separation at a grid where the FDTD's own energy defect is 5 %, so it is not a
close call. **It is also a finding for the maintainer rather than a fact about meep**:
`docs/reference/common/conventions.md` says every coordinate this library takes or hands back is measured from
the centre of the cell, and does not carve out the per-order amplitude phase. Nothing here says
which of the two should move — a translation of the structure and a translation of the phase
origin are the same operation on an amplitude, so no amplitude comparison can separate them,
and it is `meshgrid_xy` (which involves no amplitude at all) that says the structure is centred.

#### Five harness faults, each returning a plausible energy-conserving wrong number first

Three were inherited from the 2D bridge that established the pure TE/TM map; two are new to 3D
and to this file. Every one of them is still live for anyone extending the generator.

| what | what it cost, measured |
| --- | --- |
| meep replicates geometry periodically along the **PML axis** too, so a substrate block overhanging the far cell edge wraps into the superstrate | `T = 0.412` against a true 0.684, with `R + T` **exactly 1** |
| DFT monitor samples are **weighted**, not uniform: a plane spanning one period overhangs it by half a cell at each end | a plain mean put 9.9e-3 of a plane wave into orders a plane wave does not have |
| an oblique source sheet must carry the **Bloch phase itself** (`amp_func`) | `\|E\|` varied 1.66x along a monitor where a plane wave is 1.00; leak 2.99 |
| the **z pixel count** decides where meep anchors its grid, and an odd count puts a grid point exactly *on* the structure's interfaces | resolutions 40 and 80 (122 and 244 pixels) give 4.5e-3 and 1.2e-3 on `r`; **60** (183 pixels) gives **4.3e-2** — while `\|r\|` and `R + T` stay exactly on the second-order curve |
| a **layer thickness that is not an integer number of pixels** gets rounded, which moves the plane the transmitted phase is referenced to | a 0.24 µm layer at resolution 60 is 14.4 pixels; snapping it to 14 put the specular amplitude **4.6e-2** from the converged RCWA answer with `R + T = 0.9954` and every per-order efficiency inside 1.6e-2 |

The last one is the one worth reading twice, because it is this directory's recurring shape in
a new costume. It survived the **two-tier error bar** as well: resolution 30 rounds 7.2 pixels
to the same `14/60`, so the coarse and fine runs carried the identical offset and agreed with
each other to 7e-3 while both were 4.6e-2 wrong. A convergence study cannot see an error that
does not depend on the grid. What saw it was comparing *amplitudes* against a reference that
had not made the same choice — which is the same instrument `docs/guide/fdtd/fdtd.md` §5 records finding
four defects with. The generator now refuses such a thickness by name.

The two instruments that caught the rest are cheap and are both asserted rather than printed:
decompose the **incident-only** run into orders and demand that every order a plane wave does
not have comes back at round-off (1.2e-16 to 2.1e-16 on the committed cases), and decompose the
incident wave into s and p and demand that the leg not driven is empty (2.4e-5 to 3.3e-4). An
electric current sheet along `s_hat` drives only the s wave and a magnetic one along `s_hat`
only the p wave, because the two waves' in-plane fields are orthogonal — so both legs are
driven by purely transverse sheets and neither needs a z-directed source.

#### What it is worth, and what it is not

**This comparison is bounded by two grids rather than one**, which is the price of the oracle
being a different method rather than a different implementation of the same one. So the golden
is frozen at **two resolutions** per case, and `test_the_golden_states_its_own_error_bar`
asserts the claim rather than an estimate: with a converged RCWA answer in hand, the finer tier
has to be *nearer* it than the coarser one, by an implied order in a band a crossed grating can
be in.

**The order is not 2, and that turned out to be the interesting part.** Measured against RCWA
at `truncation_order` 21, which is converged: over 7, 9, 13, 17, 21 the gap below moves by 5e-5
on the square cell and 7e-4 on the rectangular one, against gaps of 1.2e-2 to 3.7e-2:

| cell | resolutions (dz) | worst propagating amplitude gap | implied order |
| --- | --- | --- | --- |
| square, s | 25 → 50 (40 → 20 nm) | 3.9e-2 → 1.6e-2 | 1.26 |
| square, p | 25 → 50 | 3.8e-2 → 1.2e-2 | 1.63 |
| rectangular, s | 30 → 60 (33 → 17 nm) | 6.6e-2 → 3.7e-2 | 0.83 |
| rectangular, p | 30 → 60 | 5.6e-2 → 3.5e-2 | 0.69 |

with the rectangular cell's s leg measured at four resolutions rather than two — 6.6e-2 /
5.0e-2 / 4.3e-2 / 3.7e-2 at 30 / 40 / 50 / 60, orders 0.99, 0.65, 0.80 — so the sub-second order
is a trend and not a pair of points. `R + T` on the same runs converges cleanly at second order (defect 1.90e-2 / 9.96e-3 /
6.14e-3 / 3.95e-3), which is the usual shape: **the energy check is blind to what is actually
limiting the answer.**

That is worth having as more than a caveat. `docs/guide/fdtd/fdtd.md` §8g measures wavebit's own crossed
convergence at 1.83 and 1.50, with a control that stalled at 0.67, and §9 lists as *unmeasured*
whether the corner cells where a vertical and a horizontal interface meet cost first order or
second. Meep's subpixel scheme is a different implementation of the same idea, written by
somebody else, and it degrades the same way on the same class of geometry — which is as close to
an independent answer as this directory can give, and it says the ceiling is the geometry rather
than either code.

Against that, a defect of the kind this row exists to catch moves an amplitude by 1e-1 to 1.2 —
a single flipped leg sign is `2|t_p|`, which is 1.28 and 1.19 on the two cells — and
`test_a_leg_sign_would_still_be_visible` asserts that rather than assuming it. So the resolving
power is 13x to 28x, not the 1e13 an RCWA-against-RCWA golden gets. **It is an oracle for
structure, exactly as the fmmax field row is**; slow drift is the job of the in-process
comparisons in `tests/fdtd/test_fdtd.py`. Five mutants, applied and reverted, are what says the
bounds are doing work rather than merely being satisfied:

| mutant | killed by |
| --- | --- |
| the transmitted p leg sign flipped (`sigma = (-1, +1, +1)`) | 3 tests |
| the y order ladder written on `Px` instead of `Py` | the **rectangular** case alone — which is that case's entire reason for existing |
| one demodulation kernel for both transverse components instead of four | the **square** case |
| the per-order frame half-turn dropped from `order_azimuth` | 7 tests, on *both* solver paths |
| the raster's y rows reversed | 8 tests |

One consequence decides how the time-domain test is run, and it is a measurement rather than a
budget: **refining wavebit's own grid past the golden's error buys nothing.** wavebit's crossed
run converges at a clean second order (2.6e-2 → 6.9e-3 over the square's 25 → 50; 6.9e-2 →
2.5e-2 → 1.4e-2 over the rectangular one's 20 → 30 → 40), so the golden is the binding
constraint: taking the rectangular run from 30 to 40 costs 2.7x the wall time and moves the gap
6.0e-2 → 4.8e-2 against a 3.7e-2 floor that is Meep's alone. The test therefore runs at the
coarse end on purpose. Whole file: 18 tests in 37 s, of which 36 s is the two time-domain
runs and the rest is four RCWA solves and arithmetic.

### The Meep 1D row: the published table, put in the repository

Added 2026-09-06, and it is the row with the least new physics in it and the most to say about
where evidence lives. `docs/guide/fdtd/fdtd.md` §2 prints a three-way agreement — the RCWA path, this
project's FDTD, and Meep — on the benchmark binary grating, and `README.md` quotes its worst
per-order gaps as the FDTD path's headline validation. **Those Meep and gyptis columns were
computed in a sibling lab repository on 2026-08-25 and quoted, not checked.** Until this pair,
`grep meep` over this repository returned only the crossed files, so the project's strongest
published evidence for its second solver was not in the project.

**Grade: an oracle for structure, at ~1e-3, on the pure TE/TM path.** Both codes discretise, so
this row is bounded by two grids the way the crossed Meep row is, and it belongs in that tier
rather than the ~1e-13 RCWA-against-RCWA one — with one exception that is genuinely tighter,
the planar calibration below (1.7e-3 against a closed form, and **9.0e-8** on the p leg's sign).

**The grade is a floor set by configuration, not by the grid, and that is the unusual part of
this row.** Refining the mesh does not get past ~1e-3 here, because this column is *bimodal*
in `(dpml, pad)`: at the frozen resolution 200, `dpml` 0.6 → 0.9 moves TE `R0` from 0.204410
to 0.205985 — **1.6e-3**, on the far side of the RCWA reference, with `Σ` unchanged at 0.99957
to five decimals. Eleven candidate mechanisms were tested on 2026-09-07 and every one refuted;
settling it would need meep's source, which this project does not open, so it is recorded as
unexplained in `_meep_1d_cases.py::MEEP_UNEXPLAINED`. **So no bound here may be tightened
below 1.6e-3 on the strength of a finer grid** — a run that agrees better than that agrees
better than the oracle's own spread, and `MEEP_RCWA_BOUND`, `FDTD_MEEP_BOUND` and `LAB_BOUND`
are all sized above it deliberately. Measured, on `tests/fdtd/test_fdtd.py::_grating()` at
meep 1.34.0:

| what is compared | agreement |
| --- | --- |
| this Meep column against a converged RCWA answer | **4.9e-4** (TE) / **7.4e-4** (TM) |
| wavebit's FDTD against this Meep column, same 5 nm cell | **1.1e-3** (TE) / **2.5e-4** (TM) |
| specular amplitudes, Meep against RCWA | 2.4e-3 / 4.6e-3 (TE `r`/`t`), 1.8e-3 / 3.1e-4 (TM) |
| the planar calibration against the closed form | 1.7e-3 at 5 nm, 6.9e-3 at 10 nm — order 2.00 |
| `r_p = -r_s`, two separately driven meep runs | 9.0e-8 |
| a swapped polarisation label, for scale | 1.9e-1 |

**It reproduces the published table, and re-deriving it moved two of the three columns.** The
RCWA column comes back to **4.5e-7** — every printed digit, unchanged. The other two did not,
for two unrelated reasons, and `docs/guide/fdtd/fdtd.md` §2 now prints the re-derived values with a
subsection saying what moved:

| | RCWA(80) | this Meep, res 200 | the 2026-08-25 lab column | gap |
| --- | --- | --- | --- | --- |
| TE `R0` | 0.204898 | 0.204410 | 0.204162 | 2.5e-4 |
| TE `T0` | 0.352474 | 0.352379 | 0.352583 | 2.0e-4 |
| TE `T±1` | 0.221314 | 0.221391 | 0.221396 | 5e-6 |
| TM `R0` | 0.282393 | 0.283135 | 0.282607 | **5.3e-4** |
| TM `T0` | 0.163805 | 0.164044 | 0.164495 | 4.5e-4 |
| TM `T±1` | 0.276901 | 0.276526 / 0.276413 | 0.276318 / 0.276231 | 2.1e-4 |

**5.3e-4 is not machine agreement and cannot be.** §2 recorded the resolution and nothing else
about how the lab column was produced — not the run length, the pad, the PML depth, the monitor
distance or the source. `test_meep_1d_golden.py::test_the_2026_08_25_lab_column_is_reproduced`
holds 1e-3 for that reason rather than for a tolerance chosen to pass, and a pass means "no
structural disagreement" rather than "the same number".

**That bound's justification changed on 2026-09-07 and the bound did not**, which is worth
stating because the two are easy to confuse. It used to rest on §2's claim that Meep's two
readings of a transmitted order differ by 2.5e-4 (TE) and 3.4e-4 (TM). Re-measured here, that
ambiguity is **≤ 7.2e-5 per order** (2.1e-4 on the total) — the old figure was a statement
about the *total*, and 2.9x too large on TM — so it cannot carry a 5.3e-4 gap and never could.
What can, and what the bound now rests on, is the `(dpml, pad)` sensitivity above: 1.6e-3 at
this very resolution, neither value recorded of the old run. The number the bound is justified
by moved by a factor of four; the number itself was already the right size for the reason
nobody had written down.

The headline pair moves with the column: **4.9e-4 (TE) / 7.4e-4 (TM)** against the quoted
7.4e-4 / 6.9e-4 — better on TE, the same on TM.

**The third column is the one worth reading, because nothing outside this repository was
involved in it going wrong.** wavebit's own "this module, 5 nm" column reproduced *exactly* —
all six values and both `Σ` — at `pad_scale=2.0`, and at the shipped default it is 6.6e-4 /
6.9e-4 rather than the printed 7.4e-4 / 6.2e-4. §2's own following paragraph records halving
that clearance and taking the dielectric from 7.37e-4 to 6.57e-4, so the table and the
paragraph beneath it were taken on either side of one change and only the paragraph was
re-derived; the section disagreed with itself by up to 8e-5 an order, for as long as no test
re-solved it. **An outside oracle was not what this needed** — a solve at the documented
settings was, and there had never been one. All three columns are now re-read on every push,
by `test_fdtd_table_golden.py::test_the_document_prints_the_frozen_three_way_table` (the
document against the golden, every cell at half a unit in its own last printed place, with
the row set asserted by name so a deleted row fails saying which),
`test_fdtd_table_golden.py::test_the_frozen_rcwa_column_is_still_what_the_solver_returns`
and `test_fdtd_table_golden.py::test_the_frozen_wavebit_column_is_still_what_the_solver_returns`
(the two solver columns re-solved, the second passing a resolution and *nothing else* so
that what is under test is the call a reader would type), and
`test_fdtd_table_golden.py::test_the_meep_column_is_the_committed_meep_golden` (the third
column, which is a copy, held against `golden_meep_1d.npz` with that file's digest).

Those name the file and the definition together, with a `::` between, rather than giving the
bare test name on purpose: that is the spelling
`tests/api/test_docs_contract.py::test_every_symbol_a_document_names_exists` checks, and a bare
backticked name is checked by nothing. All three names this paragraph carried until
2026-09-07 were stale — renamed while the sentence stayed put, which is the defect this whole
row is about, one level up. The same rewrite was made to the three that `docs/guide/fdtd/fdtd.md` §2
names.

#### What this row adds that the crossed one does not

The crossed row reaches further into the solver — a 2D order grid, a conical mount, per-order
complex amplitudes on both sides of `kx = 0`. What it does not reach is the *scene a reader is
shown*: a 1D grating at normal incidence in the pure TE/TM formulation, which is a different
code path (`_extract` rather than `_extract_crossed`) and the one the headline claim, the
tutorials and most of `tests/fdtd/test_fdtd.py` run on. Three things follow from the scene being 1D
and normal rather than crossed and conical:

* **the two legs decouple exactly**, so cross-polarisation purity is 0 rather than the crossed
  cases' 2.4e-5 to 3.3e-4 — which makes a *swapped polarisation label* the trap this scene is
  most exposed to, since both legs conserve energy and no theorem separates them. The harness
  never uses meep's polarisation vocabulary: it drives a named **component** (an electric sheet
  along `ŷ` for TE, a magnetic one for TM), so the reversal between meep's 2D naming and the
  grating convention never has to be translated. `test_the_two_legs_are_not_interchangeable`
  asserts the two answers are 1.9e-1 apart against a 1.2e-3 bound;
* **both codes keep second order** — measured 1.93 / 1.93 / 2.17 / 1.73 over the four rows —
  where the crossed cells' corners cost it (0.69 to 1.63 there);
* **a planar control is available**, because a uniform layer has a closed form.

#### The calibration, which runs before anything is frozen

`slab_uniform` is the same 0.30 µm layer with the patterning removed.
`_meep_1d_cases.py::airy_rt` writes the planar answer out from the boundary conditions — a
transfer-matrix chain, no solver of any kind — and `gen_goldens_meep_1d.py::_calibrate` refuses
to write a golden unless the frozen amplitudes come back within `120 / resolution**2`, a bound
the grid can explain rather than one chosen to pass. Measured **1.72e-3** at 5 nm cells and
**6.88e-3** at 10 nm: order **2.00**, the same coefficient 68.8 at both tiers.

It reaches three things no efficiency comparison in this directory can, and all three are
invisible to `R + T`:

| what | why nothing else here sees it |
| --- | --- |
| the **absolute** normalisation | `\|r\|` and `\|t\|` are pinned to a known number, not to each other: a stray `cos(theta)`, `n` or 2 divides out of every efficiency ratio |
| the **phase** of both | where a misplaced reference plane lives — the fault that cost the crossed golden 4.6e-2 while `R + T` stayed at 0.9954 |
| the **p leg's sign** | `r_p = -r_s` and `t_p = +t_s` at normal incidence. Two separate meep runs driven by different current sheets, so it is not an identity the harness can arrange, and it comes back to **9.0e-8** against a 1.7e-3 grid error |

#### A sixth harness fault: a block face is an interface whether or not the materials differ

The crossed generator's five faults all carry over. A sixth appeared here, in two costumes, and
both were caught by a symmetry the answer must have and the harness cannot arrange:

| the symmetry | what a block per raster run gave | what it must be |
| --- | --- | --- |
| a planar layer has no grating harmonic, so `T(±1) = 0` — and both first orders propagate in the substrate, so it is a channel with somewhere to go | 4.4e-5 each | 0 |
| a 50 % duty profile is mirror-symmetric, so `T(+1) = T(-1)` | split 6.2e-3 at resolution 40, **1.7e-3 at the frozen 200** — larger than the 1.3e-3 gap the row exists to measure | 0 |

The grating's split is the instructive one, because three explanations were eliminated before
the right one. It is **not the extraction**: the same monitor samples through a plain
uniform-weight quadrature over exactly one period give the same split to six digits. It is
**not the sample positions**: the monitor abscissae map onto themselves exactly under the
mirror. It is **not a missing periodic image**: adding image copies of every block changes
nothing. The monitor *field itself* is asymmetric by 1.9e-2, and the cause is the **air block**
— the half of the grating whose index equals the ambient. Its face lands on the cell edge with
`default_material` (the same air) on the other side, and meep smooths across it anyway.

| geometry | `T(-1)` / `T(+1)` at resolution 40 |
| --- | --- |
| tooth block + air block | 0.198521 / 0.204668 |
| **tooth block alone, ambient fills the rest** | **0.203965 / 0.203965** |
| the same tooth centred elsewhere in the cell | 0.203965 / 0.203965 — translation-invariant, as a periodic scene must be |

That is `memory: self-coincident faces carry no normal` — a face with the same material on both
sides is not an interface — arrived at independently in another code. `_layer_blocks` now emits
a full-period background plus only the runs that differ from it, and **which run is the
background is not free**: meep does not wrap the geometry when it smooths at the low cell edge,
so the background has to be the material periodicity puts just outside it, which is the *last*
column's. Backgrounding the first column instead makes the split **worse** (1.3e-2), and that
asymmetry between the two choices is the whole content of the fix. Meep's own object precedence
was measured rather than assumed: `[background, tooth]` reproduces the tooth-alone answer to
every digit while `[tooth, background]` returns a uniform layer, so the **later** object wins.

Two residues survive and both are honest: TE's split is exactly zero (8.3e-17), TM's is not,
because TM's `Ex` is the component normal to the tooth's vertical faces and therefore the
discontinuous one — 1.13e-4 fine, 1.92e-4 coarse, well under the gaps being measured.
`test_the_two_first_orders_are_equal` pins both.

#### One finding the efficiencies could not have produced

The frozen **amplitudes** say something the table does not. The specular ones agree with the
RCWA path to 2.4e-3 (TE `r`) and 3.1e-4 (TM `t`) — the grid. The **first orders** do not, and
the reason is that wavebit does not hold the per-order `ŝ` frame in one place:

| spelling | `r_s(-1)` against `r_s(+1)` |
| --- | --- |
| the **pure TE/TM** path (`_extract`), which applies no per-order frame at all | `-r_s(+1)` |
| the **conical** path at the same scene (`psi = 90`, `phi = 0`), which applies `order_azimuth`'s half-turn for `Re(kx) < 0` | `+r_s(+1)` |
| **Meep**, which holds no wavebit convention at all, read through the crossed generator's frame | `+r_s(+1)`, to 2.5e-3 |

So the two spellings of the same physical answer differ by a sign on every `m < 0` amplitude,
and an outside solver lands on the conical one. This is a statement about a frame and not a
verdict — `docs/theory/rcwa/amplitude-sp-frame.md` is where that question lives, and nothing here settles
it. `test_the_amplitudes_show_which_frame_each_solver_path_is_in` asserts all three legs of it,
so a change to either spelling is a failing test rather than a silent disagreement.

#### Cost

Whole file: 22 tests in 17 s, of which 15 s is the two time-domain runs and six RCWA solves.
The time-domain pair runs at the shipped defaults through one module-scoped fixture — two
tests read it, and sharing the solve is what stops the number a test measures from drifting
away from the number the document prints.
A full regeneration is **30 minutes** on one core — 6 runs at resolution 200 (250 to 600 s each)
and 6 at 100, with the incident-only run memoised across the two scenes because they share a
cell. `until_after_sources = 120` is a measurement rather than a budget: over 60 / 120 / 240 /
480 the answer moves 3.5e-9, 7.1e-10, 3.2e-10, and `R + T`, `T0` and `T(+1)` are identical to
eight digits.

### A generator that could not reproduce its own golden

Found 2026-08-15 while building the row above, by regenerating `golden_reticolo.npz` to check
that extending the shared bridge had not disturbed it. It had not — but the regeneration did
not match the committed file either: **20 arrays moved by up to 4.6e-1, and 17 tests failed
against freshly generated "reference" numbers.**

`gen_goldens_reticolo.py` picked the polarisation with `0 if kw.get('psi', 0.0) > 0.1 else 1`.
That reads the *conical* cases correctly, which carry `psi`. The pure cases declare `tetm='te'`
/ `'tm'` and carry no `psi` at all, so the expression returned TM for all ten of them —
including the five `te_*`. The committed golden holds genuine TE numbers, so this broke at some
point after the file was frozen and sat dormant, because a committed golden is only reproduced
when someone deliberately reruns the generator, which its own docstring invites.

`R + T` stayed exactly 1 on every case, before and after. **Solving the wrong polarisation of a
lossless stack is still a perfectly lossless answer**, so the generator's energy guard was blind
to it — the same shape as the inkstone failure and the second-order path's Voigt failure recorded
below. What caught it was regeneration-vs-committed, which is worth doing whenever the bridge is
touched, and is the only check that would have.

The polarisation is now read from whichever spelling the case uses, `tetm` first. Verified the
only way this can be verified: regenerating produces all 72 arrays byte-identical to the
committed file. Separately, the `sys.path` line pointed one level *above* the repo root, so the
invocation in the docstring failed with `ModuleNotFoundError` before reaching any of this.

### RETICOLO's two 1D frames, and which one this solver quotes (2026-09-27)

RETICOLO's own 1D path (`res0(+-1)`) and its conical path at `delta0 = 0` report amplitudes
that differ by exactly `sign(kx_m) * sign(kx_inc)` per order: the 1D one quotes the Cartesian
`E_y` / `H_y`, the conical one each order's own s/p frame, `s_hat = sign(kx) * y_hat` on that
row. Measured on V10 in the 2026-09-27 investigation, whose raw record is not public, TE and
TM, `r` and `t`, at `theta = +-0.3`, to 3e-14; after dividing out one common phase and
RETICOLO's power normalisation of `amplitude`, this solver matched RETICOLO path for path.

Since 2026-09-27 this solver's pure 1D `tetm` path quotes the per-order frame too
(`docs/theory/rcwa/amplitude-sp-frame.md` §6), so it matches RETICOLO's **conical** amplitudes and differs
from RETICOLO's 1D ones by that factor. Since §7 of the same record, the same day, it differs
from both by `(-1)**m` as well: RETICOLO's origin is the cell corner and this solver references
every per-order phase at the cell centre. That factor is derived from the field rows' measured
phase, not from a second per-order run. Any per-order comparison against RETICOLO applies both
relations at the comparison, as `test_golden_legacy_parity.py`'s `_frame_sign` and
`_centre_sign` do for the legacy golden. None needs them today: the only committed 1D
amplitude, `golden_reticolo_complex_kx.npz`'s `r0`, is the specular order, where both factors
are 1. The field rows are frame-free; they carry the lateral phase
`exp(-1j k0 sin(theta) P/2)` between the two codes' unit points, and the TM input normalisation
noted in the 1D field row above. The frame-free per-order quantity RETICOLO also returns -- each
order's Cartesian E and H from `res2` -- is what this solver's `r_xyz` / `t_xyz` correspond to,
and is the natural next per-order golden: it needs no frame relation on either of RETICOLO's
paths, only the lateral `(-1)**m`.

### Where RETICOLO stops being an oracle

Measured 2026-08-11, not taken from the manual. Feed a **real symmetric** tensor — Hermitian,
therefore lossless, therefore `R + T = 1` by theorem — through `interface/reti_aniso.m`:

| tensor | `delta0` | RETICOLO's own `R + T` |
|---|---|---|
| diagonal | 0 and 40 | 1 to 1e-15 |
| one off-diagonal pair | 0 | 1 to 1e-15 |
| one off-diagonal pair | 1 / 40 | off by 5.8e-6 / 6.3e-3 |
| two or three off-diagonal pairs | 0 | off by 2.5e-3 to 1.2e-2 |

The deviation is equal and opposite between TE and TM and does not move when `nn` goes from 0 to
8, so it is converged rather than truncated. It is not a convention difference on our side
either: a real symmetric matrix stays real symmetric under transposition and under any
permutation of axes, so nothing this bridge could misread makes the medium lossy. RETICOLO's own
`exemple_V9_0D_anisotrope.m` uses a single `xz` pair at `delta0 = 0`, inside the region that
holds.

So a **tilted magnetisation — the ordinary state of a real magnetised film — has no external
oracle here**, and `tests/rcwa/test_anisotropic_berreman.py` is still the only judge of it. The cases
that would have covered it were written, measured, and removed rather than frozen; the
generator's energy guard is what caught them. This says nothing about RETICOLO's internals,
which have not been read — only about the numbers it returns through this bridge.

### The other three, and why none of them fills that gap

Checked 2026-08-11, because "no oracle exists" is a claim that has to be earned rather than
assumed:

Each was run, at its current release, against two checks that need no agreement with anything
else: does a lossless tensor come back with `R + T = 1`, and does zeroing `xz/zx/yz/zy` change
the answer.

| solver | version | z-coupling | how it was established |
|---|---|---|---|
| fmmax | 1.7.1 | none | `eigensolve_general_anisotropic_media` takes `xx/xy/yx/yy/zz` for each of eps and mu, with no `xz/yz/zx/zy` — "general" there means *magnetic*. API signature. |
| inkstone | 0.3.15 | **accepted, silently dropped** | `AddMaterial` takes a `(3,3)` array; output is bit-identical to the same tensor with the z row zeroed |
| EMpy | 2.2.3 | **read, but unusable** | `AnisotropicTransferMatrix` does use the z row, and returns `R + T = 1.6e13` for a plain Faraday tensor at normal incidence |
| S4 | — | none | same five-slot shape; not re-run here |

inkstone deserves the detail because it is the trap. It has the only full-3×3 API of the eight,
it conserves energy to 1.3e-15 across thirty lossless configurations, and it reproduces our
first-order path to 1e-15 on every z-*decoupled* case including conical incidence — so the harness
is sound and nothing looks wrong. On z-coupled tensors it is 4e-4 to 3e-2 out, unmoved by `num_g`
from 1 to 21, and on the four cases inside RETICOLO's validated region **RETICOLO and our
first-order path agree to 1e-15 while inkstone alone departs**. Energy conservation cannot catch
this: the tensor it actually solves is a valid lossless one, which is the same reason our own
second-order path returns `R + T = 1` on a Voigt tensor while being wrong
(`test_anisotropic_berreman.py`).

EMpy fails the opposite way, and its normal-incidence blowup is what makes that readable: at
`theta = 0` there is no angle convention to get wrong, and the same harness returns the exact
Airy answer for isotropic and diagonal tensors, so `1.6e13` is the solver rather than the
bridge. It is a pure T-matrix, which its own documentation calls unstable.

**The pattern worth carrying forward: "supports a full 3×3 tensor" was wrong or badly incomplete
for three of the four.** An API that accepts an argument is not an API that uses it correctly.

inkstone is AGPL, so its implementation has not been read; every statement above is black-box —
public API signatures and returned numbers. EMpy and fmmax are MIT.

Two things to weigh when reading agreement between them:

* **fmmax generalises the S4 paper**, so those two share a formulation lineage. Agreement between
  *them* is weaker evidence than agreement with RETICOLO or Berreman.
* **RETICOLO validates individual orders**; the others are compared on totals per incident
  polarisation. For anisotropy that is deliberate: S4 is silently wrong when polarisation decomposition
  and anisotropy are combined, so totals are the only sound comparison against it.

GRCWA and TORCWA bridges used to sit in `interface/`. They predated all of this, never had a
frozen case, supported no anisotropy, and nothing referenced them — so they were the only
top-level imports of a third-party solver in the tree while buying nothing. Deleted 2026-08-15,
together with `reti_2d.m` / `reti_2D.py`: that pair was dead the same way, and the `.m` carried a
`FractureGeom` helper of unrecorded provenance, which is not something to keep in a repository
about to go out. `docs/theory/rcwa/fourier-truncation-shape.md` still discusses GRCWA's *algorithm*, which is
unaffected — the code is what went.

`reti_1D.py` / `reti_1Dc.py` went the same day and for the same reason. They compared field
cells against RETICOLO and printed norms, asserted nothing, and were not collected. They were
briefly kept as "the only record of how a RETICOLO field cell lines up with ours" — a bad
reason twice over: git history keeps that record perfectly well, and the hand-applied
`.conj()` they carried was an **unaudited** convention patch, which is the precise shape of
the thing that hid the sign defect fixed in `9824f0d`. A calibration worth trusting is
measured, not inherited: the S4 field oracle beside them was built by measuring the two
solvers against each other until every component agreed to 1e-15, and read no prior script to
do it.

`golden_reticolo_field.npz` was built the same way the next day, and the measurement settled
what those scripts had been doing. Their `np.flip(z)` was right in substance but applied per
layer, which only a multi-layer stack can tell from the whole-cell reversal that is actually
correct; their `.conj()` was **wrong outright** — the calibration search finds no conjugation
at any residual near machine precision, and what the flip really needs beside it is the
component parity. Deleting them and re-measuring cost an afternoon and produced a tighter
oracle than either script had ever asserted.

### Two S4 traps that only show on a *patterned* isotropic layer

Both found on 2026-08-16, cross-checking the half-pixel cell-centre phase
(`_half_pixel_phase`) against all three outside solvers on a chiral L cut into two disjoint
rectangles. Neither is visible from the anisotropic uniform-slab work above, which is the only
S4 use this repository had before.

**The `x`/`y` exchange is a tensor-indexing convention, not a coordinate one.**
`gen_goldens_s4.py` records that S4 indexes the permittivity tensor with `x` and `y` swapped
against this repo, and `SWAP @ eps @ SWAP` is what its uniform slabs need. It does *not* follow
that geometry needs the same treatment, and it does not: `SetRegionRectangle` takes `Center` and
`Halfwidths` in the frame this repo already uses. Swapping them there put a grating that varies
in `x` alone 1.5e-1 (s) and 4.1e-1 (p) away from the 1D path; leaving them alone brought it to
4.3e-3 and 2.9e-3, which is the truncation gap between two different order sets. Established on
a one-axis structure precisely because a 2D one cannot tell the two readings apart.

**`PolarizationDecomposition` does not converge on a corner-rich layer.** S4's vector
formulation is the natural counterpart to NVF and is the obvious thing to reach for. On this L
it wanders instead, while the plain direct rule settles:

| NumBasis | direct rule, `R_s` | `PolarizationDecomposition=True`, `R_s` |
|---|---|---|
| 121 | 0.171476 | 0.089167 |
| 241 | 0.153653 | 0.102906 |
| 441 | 0.147913 | 0.141349 |
| 761 | 0.146175 | 0.171986 |

The direct rule's steps shrink by roughly a factor of three per refinement; the decomposed one
moves 8e-2 across the same span with no sign of settling. So S4 arbitrates a patterned layer
through its direct rule only, and a comparison built on the vector formulation would be reading
its own truncation. Same shape of caveat as the anisotropy one above, reached independently.

For the record, since it is the only place this repository has compared all three outside
solvers on one patterned 2D case: on that L at 16x16, `R_s` came out 0.1426 (RETICOLO, truncation_order 14),
0.1462 (S4 direct, NumBasis 761) and 0.1396 (fmmax FFT, truncation_order 22), against this solver's 0.1419 --
a 7e-3 spread with ours inside it and RETICOLO nearest. All four are still converging on a
staircase with this many corners, which is the honest reading of that spread rather than a
tolerance anybody should inherit.

## Attribution, and what each oracle's terms require

These solvers are not vendored, and none of them is a dependency. Apart from one function
ported from fmmax, and the earlier Li builders this repository's history keeps, which descend
through meent from code written with reference to RETICOLO's source (both attributed in
`LICENSE`), nothing in this repository contains their implementations, and the committed
goldens let the suite run without any of them installed. What *is* here is our own adapter
code plus the numbers they produced. Their terms still ask for credit, and RETICOLO's ask is
explicit.

**RETICOLO.** Its documentation states: *"In publications and reports, acknowledgments have to
be provided by referencing to"* —

> J.P. Hugonin and P. Lalanne, *RETICOLO software for grating analysis*, Institut d'Optique,
> Orsay, France (2005). [arXiv:2101.00901](https://arxiv.org/abs/2101.00901)

and it names one further reference as **required when near-field distributions are shown**.
Since 2026-08-15 that obligation is live rather than anticipatory: `golden_reticolo_field.npz`
holds near-field distributions, so this reference is owed alongside the one above:

> P. Lalanne and M.P. Jurek, *Computation of the near-field pattern with the coupled-wave
> method for TM polarization*, J. Mod. Opt. **45**, 1357–1374 (1998).

Two more are asked for "fairly" rather than required, and both are already cited in
`docs/` for the formulation itself: Moharam et al. (1995) for TE efficiencies, and Lalanne &
Morris (1996) with Granet & Guizal (1996) for TM.

Of the bridge scripts, `interface/reti_aniso.m` lays out its calls to RETICOLO (`res0`, `res1`,
`res2`) after RETICOLO's own example `exemple_V9_0D_anisotrope.m`, as its header says: a
texture holding a full 3x3 permittivity tensor, and `nn = 0`.

**S4.**

> V. Liu and S. Fan, *S4: A free electromagnetic solver for layered periodic structures*,
> Comput. Phys. Commun. **183**, 2233–2244 (2012).
> [doi:10.1016/j.cpc.2012.04.026](https://doi.org/10.1016/j.cpc.2012.04.026)

**fmmax** (MIT licence).

> M.F. Schubert and A.M. Hammond, *Fourier modal method for inverse design of
> metasurface-enhanced micro-LEDs* (2023). [arXiv:2308.08573](https://arxiv.org/abs/2308.08573)

**PyMoosh** (MIT licence: its `LICENSE.txt` and package metadata; the header of its
`__init__.py` carries GPL-3 boilerplate that ends by pointing to the MIT licence, checked in
4.0.1).

> D. Langevin, P. Bennet, A. Khaireh-Walieh, P. Wiecha, O. Teytaud and A. Moreau, *PyMoosh: a
> comprehensive numerical toolkit for computing the optical properties of multilayered
> structures*, J. Opt. Soc. Am. B **41**, A67 (2024).
> [doi:10.1364/JOSAB.506175](https://doi.org/10.1364/JOSAB.506175)

### What RETICOLO's records and packages say about its licence

**Meep** (GPL-2.0-or-later).

> A.F. Oskooi, D. Roundy, M. Ibanescu, P. Bermel, J.D. Joannopoulos and S.G. Johnson,
> *Meep: A flexible free-software package for electromagnetic simulations by the FDTD method*,
> Comput. Phys. Commun. **181**, 687–702 (2010).
> [doi:10.1016/j.cpc.2009.11.008](https://doi.org/10.1016/j.cpc.2009.11.008)

### Meep is GPL, and the whole arrangement here is what keeps that a non-issue

wavebit is Apache-2.0, so a GPL dependency is not one of the options. The policy is the same
one this directory already applies to inkstone, stated up front rather than reached later:

* **no file under the meep package is opened, read, grepped or copied.** Meep is used by
  *calling* its public Python API and reading the numbers that come back, so every statement
  this directory makes about its conventions is a black-box measurement. The generator sets
  `sys.tracebacklimit = 0` so that an uncaught exception cannot print lines of another
  package's source into the terminal;
* **nothing in the shipped package imports it**, and it is not a dependency of anything —
  including the tests. `gen_goldens_meep_crossed.py` and `gen_goldens_meep_1d.py` run by hand
  in their own environment, exactly as the RETICOLO generator needs Octave and the fmmax ones
  need `fwref`;
* what is committed is **our own adapter code plus the numbers meep produced**, which is the
  same footing as every other row.

The `meep` environment, for the record:

```bash
conda create -y -n meep -c conda-forge pymeep
```

A full regeneration of the **crossed** golden is about 25 minutes on ten cores: 154 s and 210 s
for the square cell's two legs at resolution 50, 498 s and 517 s for the rectangular one at 60,
plus a tenth of that again for the two coarse control tiers. The **1D** golden is another 30
minutes on one core: six runs at resolution 200 (250 to 600 s each) and six at 100, with the
incident-only run memoised across the two scenes because they share a cell.

### What RETICOLO's licence actually says, since the answer is not tidy

Checked 2026-08-15, because this file previously asserted "not redistributable" without a
source. The declarations do not agree with each other:

| where | what it says |
|---|---|
| the V10 package's own documentation PDF | *"Reticolo is a **freeware** operating under Matlab"*, `copyright © 2005 Institut d'Optique/CNRS`. **No** redistribution clause either way |
| the V9 and V9a documentation PDFs | `copyright © 2005 Institut d'Optique/CNRS`, and no "freeware" sentence |
| [Zenodo, V10](https://zenodo.org/records/14631951) — the official download | **CC BY 4.0**, which *does* permit redistribution with attribution |
| [Zenodo, V9](https://zenodo.org/records/4419063), [V9a](https://zenodo.org/records/5905381) and [V8](https://zenodo.org/record/3610175/export/json) | no licence declared; the `rights` field is empty |

Re-checked 2026-09-25, with the packages opened as well as the records read. None of V9, V9a
or V10 carries a licence file or licence text in its source. The record this table used to
call V9 is V9a (`version` `9a`, created 2022-01-26); V9 itself is 4419063 (2021-01-05). V10's
record reads `publication_date` 2020-01-16, which is V8's date carried over; it was created
2025-01-11, and its documentation says V10 was launched in 01/2025. The reason for the second
look is `LICENSE`'s RETICOLO notice: meent's Li-rule factorization was written with reference to
V9's MATLAB source, and the files it followed -- `retcouche.m` (its `ret2li`), `retf.m` and
`rettoeplitz.m`, with `res0.m` for the `li = 1` default -- are byte-identical in V9, V9a and V10.
Between V9 and V10, `res1.m` (one line) and `retu.m` (a texture addition) changed, and so did the
documentation. The builders were rewritten from Li's rules the same day, so that notice now covers
the earlier versions this repository's history still holds. Recorded as `CONTRIBUTING.md`'s
licensing section asks: the licence relied on is the one the V10 record states (CC BY 4.0); that
record names itself a new version of V9 (`isNewVersionOf` 10.5281/zenodo.4419063), and its copies
of the files the earlier versions followed are byte-identical to V9's; the use is the modified,
earlier Li builders and the RETICOLO-marked meent code kept in this repository's history, with
no RETICOLO file in the repository (`_circulant_index`, which came with the 2026-08-03 port of
meent's `dfs2d` and which the rewritten builders still call, was itself rewritten from its
definition on 2026-09-26); and the obligation is the attribution `LICENSE` gives.

The GitHub repository the Zenodo entries are named after (`Light-in-complex-nanostructures/RETICOLO`)
returned 404 when checked on 2026-08-15 and 2026-09-26, so no licence could be read there. Nothing
beyond the records and packages above is relied on here.

**The policy does not change: RETICOLO stays outside this repository.** Vendoring a 184-file
MATLAB tree would buy nothing — version pinning is what `RETICOLO_PATH` is for. What changed
is the *reason*: it is a deliberate choice, not a prohibition we had verified. If redistribution
ever becomes useful, the V10 record's CC BY 4.0 is the thing to re-read.

## The `fwref` environment

fmmax carries its own JAX floor and the project env sits well ahead of it, so installing fmmax
alongside would drag JAX back and break the backend the suite tests. It gets its own environment,
used by nothing but the generators here:

```bash
conda create -y -n fwref python=3.11
conda run -n fwref pip install fmmax
```

Two consequences worth knowing before touching those generators:

* **The case list cannot live in the test module** the way the RETICOLO pair's does — neither env
  can import the other's module, since `fwref` has no `wavebit` and the project env has no
  `fmmax`. It lives in `_fmmax_cases.py`, which imports numpy and nothing else.
* **JAX defaults to float32.** Without `jax.config.update('jax_enable_x64', True)` an fmmax
  generator agrees with an analytic slab only to ~3e-8, which reads as a mediocre solver and is
  really a mediocre configuration.
* **x64 is not enough on a patterned layer.** fmmax 1.7.1 builds its pixel-centring phase from a
  float32 frequency grid whatever the x64 setting, which costs 1e-8 to 1e-7 in the efficiencies
  of a high-contrast patterned layer. Invisible at the 1e-4 the vector-formulation comparisons
  here work to; decisive for an identity comparison -- see the fmmax FFT section above.

## Which file does what

| Piece | Needs |
| --- | --- |
| `interface/` — the RETICOLO bridge (`Reticolo.py` + our `.m` wrappers) | Octave or MATLAB, and RETICOLO |
| `gen_goldens_*.py` — freeze the reference numbers | the oracle; run by hand, rarely |
| `golden_*.npz` — the frozen numbers, committed | nothing |
| `test_*_golden.py` — the assertions | nothing; **run in CI everywhere** |
| `interface/freeze_fd_stage0.py`, `_fd_stage0_cases.py` — freeze the `src/wavebit/fd/` kernel's own answers, per backend | the project env with NumPy, JAX and Torch; run by hand |
| `fd_golden_stage0.npz` — those answers, committed; `tests/fdtd/test_fd_backends.py` holds them | nothing; **run in CI everywhere** |
| `interface/freeze_fd_stage1.py`, `_fd_stage1_cases.py` — the same for stage 1's absorbers and materials, plus the lowered coefficients and profiles | the project env with NumPy, JAX and Torch; run by hand |
| `fd_golden_stage1.npz` — those answers, committed; `tests/fdtd/test_fd_stage1_backends.py` holds them | nothing; **run in CI everywhere** |
| `interface/freeze_fd_stage2.py`, `_fd_stage2_cases.py` — the same for stage 2's sources, with every source's reference series and the lowered stamps, pattern and waveform table | the project env with NumPy, JAX and Torch; run by hand |
| `fd_golden_stage2.npz` — those answers, committed; `tests/fdtd/test_fd_stage2_backends.py` holds them | nothing; **run in CI everywhere** |
| `interface/freeze_fd_stage3.py`, `_fd_stage3_cases.py` — the same for stage 3's monitors, with every DFT accumulator whole, every result built from it and the sources' spectra, plus the lowered DFT frequencies and Axis1D factor | the project env with NumPy, JAX and Torch; run by hand |
| `fd_golden_stage3.npz` — those answers, committed; `tests/fdtd/test_fd_stage3_backends.py` holds them | nothing; **run in CI everywhere** |
## A golden frozen ahead of its consumer

`golden_fmmax_magnetic.npz` holds rows nothing in this solver can consume — permeability has no
slot anywhere (`docs/guide/rcwa/anisotropic-materials.md` §5). It was frozen anyway, on purpose: **an oracle
chosen after the work is an oracle chosen to agree.**

What keeps that honest is that every case is generated twice, once with its permeability and once
at `mu = 1`, and the `mu = 1` half *is* held to the solver today. If the harness, units or
conventions were wrong, that subset would fail now, while the answer is independently knowable.
The magnetic half carries a physical guard instead of a comparative one: every case is Hermitian
in both tensors, hence lossless, so `R + T = 1` is enforced on the frozen numbers.

### Crossed material reference scope

The three material scenes are specified in `_meep_crossed_material_cases.py` and documented
in `docs/guide/fdtd/fdtd-alpha.md`. Their dispersive active permittivity has positive real part at the
test wavelength; these are not a validation of negative-epsilon metals. The generator
checks the external material API at three wavelengths, reuses the existing crossed
monitor/frame bridge, and freezes a new file. `check_meep_crossed_material_runtime.py`
checks that file with twice the runtime and records its digest, without replacing it.
The regression suite compares complex amplitudes, checks spatial refinement against RCWA,
and verifies that omitting the material contrast is distinguishable from discretization.
No Meep implementation source is used by these tools.

### Damped metal and reference resolution

The negative-epsilon case is separate from the positive-epsilon material file. The original
Meep-60 comparison failed the selected amplitude/absorption bounds. Those data remain in
`golden_meep_crossed_metal.npz`; the separately measured Meep-120 tier is tied to it by a
digest in `golden_meep_crossed_metal_fine.npz`. `refine_meep_crossed_metal.py` reproduces
the finer tier. Both helpers and case definitions are included in its physics-harness
digest. The original measurement was made as separate s/p public-API runs, combined without
changing their values. `docs/guide/fdtd/fdtd-alpha.md` records the discrepancy and convergence;
a reference with insufficient resolution is not grounds to loosen the comparison bound.


## Native-grid external validation (2026-09-07)

`golden_native_grid_external.npz` stores independently executed Meep 1.34.0 and
FDTDX 0.6.2 outputs for the new `wavebit.fdtd_grid` engine. It contains no Wavebit
outputs. Generator/input hashes and the installed FDTDX update-module digest are
embedded as provenance. Meep implementation source was not opened or searched;
only its [documented Python API](https://meep.readthedocs.io/en/master/Python_User_Interface/)
was used. FDTDX update functions are called from the installed external package,
with no implementation copied into Wavebit.

The Meep case is a vacuum-surrounded epsilon=2.25 slab of thickness 0.5 at normal
incidence, wavelengths 0.6/0.7/0.8, and resolution 20/40/80. Both implementations
use a periodic domain of length 32, Courant number 0.4, a prescribed current
pulse, and a time window of 18 (before wraparound). The monitor is outside the
slab. Divide slab Ex/Hy Fourier fields and power by a matching empty run, preserving
complex phase. Meep's interface averaging and native tangential half-cell weights
represent the same plane. No PML/absorber is involved in this comparison.

| Resolution | Maximum complex transfer difference | Maximum power difference | Meep complex transfer error against Airy |
| --- | --- | --- | --- |
| 20 | 1.59e-7 | 1.43e-7 | 0.12984 |
| 40 | 2.54e-13 | 2.30e-13 | 0.02947 |
| 80 | 8.26e-14 | 5.23e-14 | 0.00720 |

The Airy errors decrease at orders 2.14 and 2.03. Agreement between two Yee
implementations does not remove their shared discretization error. Extending the
resolution-40 window from 18 to 20 changes either solver's complex transfer by
1.18e-10. An initial shorter window missed internal echoes; that pilot also used
unmatched interface sampling and is not the frozen reference. The final setup
matches the physical averaging and captures the decayed transmitted pulse.

The FDTDX case calls its native `update_H` then `update_E` for 37 steps on a
9×7×5 periodic grid, with deterministic nonuniform diagonal epsilon and mu and
initial six-component fields. Arrays are transposed between (z,y,x) and (x,y,z).
Its Courant number is 0.2, matching the c=1 Wavebit time step/spacing. No source,
PML, geometry placement or detector is being validated by this test. Maximum E/H
errors are 1.67e-16 / 2.64e-16. The local field objective's epsilon derivative is
-0.4196157538227005 (FDTDX) and -0.41961575382270005 (Wavebit); both agree with
central finite differences within 1e-9 absolute error.

These are **CPU checks**. They do not establish GPU execution, general scattering
accuracy, CPML/ADE parity, or FDTDX source/detector compatibility. Ordinary tests
consume the frozen external outputs, so passing them is not a fresh external run.

To reproduce live references, run from the repository root with separate Python
environments. `$MEEP_PYTHON`, `$FDTDX_PYTHON` and `$WAVEBIT_PYTHON` below are paths
to their interpreters; choose a new output directory first:

```sh
mkdir -p artifacts/native-grid-reference
for resolution in 20 40 80; do
  "$MEEP_PYTHON" tests/references/interface/validate_grid_meep.py --backend meep --resolution "$resolution" --output "artifacts/native-grid-reference/meep-grid-$resolution.json"
  PYTHONPATH=. "$WAVEBIT_PYTHON" tests/references/interface/validate_grid_meep.py --backend wavebit --resolution "$resolution" --output "artifacts/native-grid-reference/wavebit-grid-$resolution.json"
done
"$MEEP_PYTHON" tests/references/interface/validate_grid_meep.py --backend meep --resolution 40 --duration 20 --output artifacts/native-grid-reference/meep-grid-runtime.json
"$FDTDX_PYTHON" tests/references/interface/validate_grid_fdtdx.py --backend fdtdx --output artifacts/native-grid-reference/fdtdx-grid.json
PYTHONPATH=. "$WAVEBIT_PYTHON" tests/references/interface/validate_grid_fdtdx.py --backend wavebit --output artifacts/native-grid-reference/wavebit-fdtdx-grid.json
"$WAVEBIT_PYTHON" tests/references/interface/freeze_grid_references.py --input artifacts/native-grid-reference --output artifacts/native-grid-reference/external.npz
"$WAVEBIT_PYTHON" -m pytest tests/references/test_native_grid_external.py
```

Freezing refuses an existing destination. Review a newly generated archive before
replacing committed external evidence; never regenerate it merely to pass a test.

## Native CPML and dispersive materials (2026-09-08)

The additional `golden_low_loss_metal_meep.json` reference covers a low-loss
mixed metal with epsilon-infinity 1.5 and poles (0, 0.08, 160), (10, 0.12, 40).
Its wavelengths are 0.9/1/1.1 and duration is 160; the remaining geometry,
Courant factor and field normalization follow the worker below. The sine
carrier period and Gaussian width are 1, centered at time 5. Meep 1.34.0 and
Wavebit with point-sampled strengths agree within 1.2e-7 for complex Ex/Hy and
2.7e-9 for power. The test bound is 1e-6 absolute. See the
[measurement](../../docs/validation/fdtd/fdtd-mixed-metal.md) for the distinct scene, Airy and
gradient comparisons. Earlier goldens were not regenerated.

Reproduce the new external runs using a Meep environment; only the public API
is used, and tracebacks are suppressed to avoid displaying implementation frames:

```sh
for res in 40 80; do
  "$MEEP_PYTHON" tests/references/interface/validate_grid_dispersive.py --backend meep --model low-loss-mixed --resolution "$res" --duration 160 --output "artifacts/low-loss-meep-$res.json"
done
```

To freeze a separate candidate file, preserving the committed reference:

```python
import hashlib
import json
from datetime import date
from pathlib import Path

worker = Path('tests/references/interface/validate_grid_dispersive.py')
payload = dict(
    description='Low-loss mixed Drude/Lorentz slab; Meep public API execution, no implementation inspection.',
    generated=date.today().isoformat(), worker=str(worker),
    worker_sha256=hashlib.sha256(worker.read_bytes()).hexdigest(),
    epsilon_inf=1.5, poles=[[0., .08, 160.], [10., .12, 40.]],
    runs=[json.loads(Path(f'artifacts/low-loss-meep-{r}.json').read_text()) for r in (40, 80)])
with Path('artifacts/golden_low_loss_metal_meep.json').open('x') as output:
    output.write(json.dumps(payload, indent=2)+'\n')
```

`golden_native_material_external.npz` contains additional live Meep 1.34.0
outputs and a two-pole FDTDX 0.6.2 update/gradient reference. External input and
worker digests are stored in the archive. Meep source was never read; its
[material API](https://meep.readthedocs.io/en/master/Python_User_Interface/) and
[subpixel smoothing documentation](https://meep.readthedocs.io/en/latest/Subpixel_Smoothing/)
specify the conventions used. No external solver implementation was copied.

`tests/references/interface/validate_grid_dispersive.py` uses a length-12 cell, vacuum-end PML of
width 1, source at z=3, slab at 7..7.5 and measurement at 9. The three wavelengths
are 0.6/0.7/0.8, resolutions 40/80, cdt/dx=0.4 and duration 24. Meep's own PML
and Wavebit's CPML need not have the same implementation. The electric current
has a Gaussian envelope of width 0.7 centered at 3.5 and a period-0.7 sine carrier;
it has zero DC by symmetry. Both fields and power are divided by an empty run.

| Model | epsilon-infinity | (omega0, gamma, strength), angular c=1 units |
| --- | --- | --- |
| dielectric | 2.25 | none |
| Lorentz | 1.5 | (12, 1.5, 60) |
| Drude | 1 | (0, 1.2, 100) |
| mixed | 1.5 | (0, 1.2, 40), (12, 1.5, 60) |

Meep averages epsilon-infinity at interfaces but **does not average the
frequency-dependent part**. The `--pole-sampling point` Wavebit worker explicitly
uses the same point sampling of dispersive strength. All four models at both
resolutions pass a 3e-6 absolute bound for complex Ex/Hy transfer and power.
This is an actual direct comparison, not a loose allowance for different
interface discretizations.

The independent `--pole-sampling average` worker uses tangential half-cell
strength weights. It is checked against the Airy solution, not against Meep's
point-sampled result. Maximum complex transfer errors are:

| Model | resolution 40 | resolution 80 |
| --- | --- | --- |
| dielectric | 0.029475 | 0.007204 |
| Lorentz | 0.021089 | 0.005210 |
| Drude | 0.006561 | 0.001631 |
| mixed | 0.014133 | 0.003494 |

All four converge at approximately second order on this tangential planar case.
The initial comparison mixed these two discretizations and differed by up to
0.065 at resolution 40; the difference halved with refinement. Matching Meep's
documented strength sampling removed that discrepancy without changing the
solver or relaxing a test tolerance.

CPML controls extend the dielectric runtime to 36 or double PML width to 2;
complex transfer changes remain below 3e-6. A broader pilot pulse with nonzero
DC left slowly decaying CPML tails: extending duration from 24 to 48 reduced the
dielectric Meep difference from 6.7e-4 to 1.2e-4, while width alone did not fix
it. This behavior is retained as a convergence limitation, not hidden by the
zero-DC reference pulse. Separate tests require late normal/oblique pulse energy
to drop below 1e-5/1e-4 of a periodic control. These bounds are not universal
reflection specifications or grazing-angle validation.

`tests/references/interface/validate_grid_fdtdx_ade.py` independently asks FDTDX's coefficient factory
for a Drude and a Lorentz pole, then calls its native H/E updates for 53 steps
on a 9×7×5 periodic grid with nonuniform diagonal epsilon/mu and oscillator
strength. Frequencies are converted from c=1 to rad/s; no waveform/geometry
adapter is involved. E/H differences are 9.54e-16 / 7.08e-16. The oscillator
strength derivative is -0.25818464174803096 (FDTDX) and -0.25818464174803113
(Wavebit); both match central finite differences to within 1e-10 absolute.

Run each worker with its own environment, as above. Example commands:

```sh
"$MEEP_PYTHON" tests/references/interface/validate_grid_dispersive.py --backend meep --model lorentz --resolution 40 --output artifacts/meep-lorentz-40-band.json
PYTHONPATH=. "$WAVEBIT_PYTHON" tests/references/interface/validate_grid_dispersive.py --backend wavebit --model lorentz --resolution 40 --pole-sampling point --output artifacts/wavebit-lorentz-40-point.json
"$FDTDX_PYTHON" tests/references/interface/validate_grid_fdtdx_ade.py --backend fdtdx --output artifacts/fdtdx-ade.json
PYTHONPATH=. "$WAVEBIT_PYTHON" tests/references/interface/validate_grid_fdtdx_ade.py --backend wavebit --output artifacts/wavebit-fdtdx-ade.json
"$WAVEBIT_PYTHON" -m pytest tests/fdtd/test_fdtd_grid_materials.py tests/references/test_native_material_external.py
```

To regenerate a separate archive, produce every `meep-{model}-{40,80}-band.json`
for the four models and `fdtdx-ade.json` in one directory, then run
`tests/references/interface/freeze_grid_material_references.py --input DIR --output NEW.npz`.
It reads only external outputs and refuses to overwrite an existing archive.
These are CPU checks of native electric ADE and vacuum-end z CPML. Magnetic
poles, arbitrary PML backgrounds, full Stack/Incidence dispatch, and actual GPU
execution remain unsupported or unverified.

## Native Bloch boundaries (2026-09-08)

`tests/references/interface/validate_grid_fdtdx_bloch.py` repeats the live FDTDX comparison with
complex initial E/H and phase advances `(0.3,-0.2,0.4)` in `(z,y,x)` order.
These phases enter FDTDX as physical `(kx,ky,kz)` radians per meter, divided by
the matching cell lengths. The grid is 9×7×5, 37 steps, nonuniform diagonal
epsilon/mu, Courant number 0.2. No source or CPML is involved in this direct
boundary check. Maximum real/imaginary field differences are below 2.23e-16;
the material derivative is 0.43721809199586215 (FDTDX) and
0.4372180919958624 (Wavebit), with both checked against finite differences.

`golden_native_bloch_external.npz` retains only the external real/imaginary E/H,
objective/derivative and provenance. Regenerate into a new archive with:

```sh
"$FDTDX_PYTHON" tests/references/interface/validate_grid_fdtdx_bloch.py --backend fdtdx --output artifacts/fdtdx-bloch.json
PYTHONPATH=. "$WAVEBIT_PYTHON" tests/references/interface/validate_grid_fdtdx_bloch.py --backend wavebit --output artifacts/wavebit-fdtdx-bloch.json
"$WAVEBIT_PYTHON" tests/references/interface/freeze_grid_references.py --kind bloch --input /tmp --output artifacts/external-bloch.npz
"$WAVEBIT_PYTHON" -m pytest tests/fdtd/test_fdtd_grid_bloch.py tests/references/test_native_bloch_external.py
```

Independent tests check arbitrary-phase discrete Maxwell propagation in all
three axes, collocation at a wrapped monitor, derivatives with respect to phase,
and explicit complex64 scans with x64 enabled. The oblique example couples a
transverse Bloch phase to the source's spatial phase and differentiates both
through a CPML solve. These remain CPU checks, not GPU validation or a full
Stack/Incidence adapter.


## Shared Torch FDTD execution (2026-09-08)

After moving device/scan/checkpoint policies into `src/wavebit/fdtd/_periodic/runtime.py`, the
same native Array API core also runs on Torch. `test_native_torch_external.py`
reproduces the non-dispersive, two-pole and complex-Bloch problem inputs above
and compares full E/H fields, objective values and material/strength derivatives
with the existing FDTDX archives. All three pass with 1e-11 relative / 1e-12
absolute field bounds. No goldens were regenerated for the refactor. The Meep
CPML/material checks continue through JAX, while direct Torch/NumPy parity tests
cover the CPML/ADE/Bloch combination and derivatives. This distinguishes frozen
external regressions from fresh external runs. GPU execution remains unverified.


## Bottom-incident common scene (2026-09-08)

`golden_scene_bottom_meep.json` is a fresh Meep 1.34.0 CPU public-API run from
`tests/references/interface/validate_scene_bottom_meep.py`. Two layers (epsilon 2/3, thickness
.13/.21) sit at original z=0..0.34 in vacuum. A physical source at z=3 illuminates
toward -z; field planes at z=2/-2 isolate reflected/transmitted amplitudes using
an empty reference run. No coordinate reflection is used in this oracle.
Native vacuum Yee propagation de-embeds r to z=.34 and t to z=0.

Resolution80, Courant.4, duration30, wavelengths.9/1/1.1. A duration40 control
changed complex r/t components by at most6.1e-15. The common Wavebit scene adapter
uses its own Courant fraction and sampling; complex r gap<6e-4, t gap<2e-4.
`test_scene_bottom_meep.py` holds both to1.5e-3. This checks phases/signs, not
only power. New reference only; no existing goldens were regenerated. The worker
suppresses implementation tracebacks and reads no Meep source.
