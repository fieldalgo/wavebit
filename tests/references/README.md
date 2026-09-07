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
`__stack` arrays. Both hold a refractive index still, both are converted **at the read**, and
both say so where the conversion sits.

The RETICOLO change is the one that could have moved numbers, so it was checked the way this
directory's own note on a generator that could not reproduce its own golden prescribes:
regenerating `golden_reticolo.npz` (72 arrays), `golden_reticolo_field.npz` (33) and
`golden_reticolo_conical_field.npz` (33) against V10 after the flip reproduces **every array
bit for bit**. Nothing in the frozen data moved, so the scenes are the ones they always were.
`gen_goldens_s4_field.py` was rerun too: 26 of its 28 arrays are bit-identical and the two that
differ are exactly the `__stack` pair above.

## The oracles

| Oracle | Reaches | Generator needs | Golden |
| --- | --- | --- | --- |
| **RETICOLO** (Li/Lalanne lineage) | diffraction *orders*; uniform layers with a complex 3x3 tensor — **but only up to one off-diagonal pair, at classical mounting**, see below | Octave + a licensed RETICOLO V10 tree (`RETICOLO_PATH`) | `golden_reticolo.npz`, `golden_reticolo_anisotropic.npz` |
| **RETICOLO**, swept over `(kx, ky)` | conical incidence across the light line, **including complex (evanescent) angles** | nothing — frozen from a 2024-09-09 study, see its generator | `golden_reticolo_conical_sweep.npz` |
| **fmmax** (Liu & Fan lineage, MIT) | uniform complex tensors, **patterned** tensors, **permeability** | the `fwref` env — see below | `golden_fmmax.npz`, `golden_fmmax_magnetic.npz` |
| **S4** (same lineage, C++) | real tensors, patterned | the S4 extension, already in the project env | `golden_s4.npz` |
| **S4**, the in-cell **field** | the six field components on *uniform* stacks — every reconstruction path, all angles and azimuths, exact at ~1e-15 | same | `golden_s4_field.npz` |
| **fmmax**, the in-cell **field** | uniform stacks exactly (7.8e-16) **and patterned layers to ~1e-4**, through the coupled 2D path | the `fwref` env | `golden_fmmax_field.npz` |
| **RETICOLO**, the in-cell **field** | the *pure TE/TM 1D* path (`field_1d`), which the two rows above never reach — **1e-14 on gratings**, five components of six | Octave + `RETICOLO_PATH` | `golden_reticolo_field.npz` |
| **RETICOLO**, the **conical** field | `field_1d_conical` on a *patterned* layer — an azimuth and a discontinuity at once, which nothing else here reached | same | `golden_reticolo_conical_field.npz` |
| **Meep** (FDTD, GPL) | a **crossed grating in the time domain**: per-order complex amplitudes on a 2D order grid at a conical mount, both incident polarisations. Not RCWA at all, so it is the only entry that can arbitrate `_core/fdtd.py` — and the only external check `_extract_crossed` has ever had | the `meep` env; **never an install dependency**, and its source is not read — see below | `golden_meep_crossed.npz` |
| **Meep**, **patterned materials in 3D** | selected asymmetric crossed rasters with permeability contrast, electric Drude and electric Lorentz poles; both source polarizations, complex propagating amplitudes at two grids, and a separate doubled-runtime check | `meep` env, public API only; never an install dependency | `golden_meep_crossed_materials.npz`, `golden_meep_crossed_materials_runtime.json` |
| **Meep**, **damped negative-epsilon metal** | a subwavelength rectangular 3D Drude patch; only zeroth order radiates. Preserved grids 30/60, separate grid 120, runtime control at 60; coarse-reference discrepancy is documented | same public-API-only Meep environment | `golden_meep_crossed_metal.npz`, `golden_meep_crossed_metal_fine.npz` |
| **Meep**, the **1D benchmark** | the grating `docs/fdtd.md` §2 publishes and `README.md` leans on — pure TE/TM, normal incidence, both polarisations at two resolutions, plus a **planar control against a closed form**. Not new physics: it *is* that table's Meep column, which until 2026-09-06 was quoted from elsewhere. Grade **~1e-3**, structure not drift | same | `golden_meep_1d.npz` |

One more oracle lives outside this directory: the Berreman 4x4 in
`tests/test_anisotropic_berreman.py` is an independent *formulation* — not RCWA at all — but our
own implementation, so it sits a tier below the rows above. Meep is the only entry that is both
at once: a different formulation *and* somebody else's code.

### One file here is not an oracle: `golden_fdtd_table.npz`

Every row above freezes somebody else's answer. `golden_fdtd_table.npz` freezes **ours** —
`docs/fdtd.md` §2's two tables, all three columns, with the conditions they were taken at
stored beside them — and it exists because a published table is a claim with no owner. §2's
"this module" column was taken at `pad_scale=2.0`, `PAD_WAVELENGTHS` was then halved, the
sentence "at the shipped defaults" stayed true-looking, and for three weeks the document
printed a configuration the library no longer had. Nothing caught it because **nothing in the
suite read a number out of a document**: `tests/test_docs_contract.py` checks paths, symbols,
URLs and the toctree, never a value.

So `test_fdtd_table_golden.py` parses §2's markdown tables cell by cell and holds every value
to half a unit in its last printed place, re-solves the two columns this environment can, and
— the one that would have caught the original drift on the day it happened — asserts that the
conditions written out **by value** in `_meep_1d_cases.py` are still the ones
`wavebit/_core/fdtd.py` ships. That case file is the record now, for both sides of the
comparison: it carries what the harness passes, what it *inherits* (the four largest
determinants of the Meep column are meep defaults the generator never passes, pinned only by
the version string), the derived grid in lengths and counts, and a measured sensitivity table
saying what a deviation costs. A record that names values can be checked against the module;
a record that says "the defaults" is true by construction and can be checked against nothing.

`gen_goldens_fdtd_table.py` writes it, in the project env, and copies the Meep column across
from `golden_meep_1d.npz` with that file's digest so a stale copy is a failing test.

### Why the field row exists, and what it cost to not have it

Until 2026-08-14 every oracle here checked efficiencies or amplitudes, and **field
reconstruction had none at all**. Its only reference was a golden captured from this project's
own earlier implementation — a change detector wearing an oracle's name. What it froze was a
wrong sign convention (`Ez`, `Hx`, `Hy` in the frame whose `+z` points *out* of the stack), and
the suite stayed green for as long as the file existed, because a self-captured golden agrees
with whatever produced it. It is gone; `golden_s4_field.npz` replaces the claim it was making,
and `tests/data/golden_field.npz` keeps the drift-detection job under its own honest name.

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
numerical drift is `tests/data/golden_field.npz`'s job.

One further thing fell out of the calibration and is worth having as a statement rather than a
constant: fmmax's unit **mode** amplitude relates to our unit **flux** incidence by exactly
`-cos(theta)` for s and `-1` for p. The test asserts that dependence, so a case added at a new
angle is checked rather than accommodated.

Two conventions had to be measured rather than assumed, and both are recorded in
`gen_goldens_s4_field.py`: S4's p amplitude is a **global** `-1` against ours (not a frame
parity — it moves all six components together), and the comparison must sample layer
**interiors**, because `Ez` is discontinuous at an interface and the two codes may legitimately
report different one-sided limits there.

### The 1D field row: the pure TE/TM path, and the strongest of the three

Added 2026-08-15. S4 and fmmax both reach field reconstruction through the *coupled*
6-component routes. Neither exercises `_core/fields.py::field_1d` (`grating_type == 0`), which
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
| **the component parity** the reversal carries | none, or a hand-applied `.conj()` | `Ez, Hx, Hy` flip / `Et, Hz` do not — the rule `_core/fields.py` already states |

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
the fmmax golden needs, but exactly 1, to 2.7e-14. The test asserts that rather than fitting it,
because a scale free to absorb a factor absorbs a defect just as happily.

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
conjugation, no scale — transfers unchanged. A full search over all 720 component permutations x
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

### The Meep row: the first oracle here that is not RCWA, and what it took

Added 2026-09-06. Every other row truncates a Fourier series and inherits whatever
factorization that forces, which is the formulation `_core/engine.py` uses — so none of them
can arbitrate `_core/fdtd.py`, whose errors are numerical dispersion, a staircased boundary and
an imperfect absorber instead. Meep is an FDTD. Nothing is shared but Maxwell's equations.

Two gaps close at once, and the second one is wider than it looks:

* **a crossed grating solved in the time domain.** `_extract_crossed` is a line-for-line copy
  of `_extract_conical`, and every per-order sign in it — the three `sigma` legs, the
  `s_hat`/`p_hat` frame, the `n / kz_m` secant — is invisible to `R`, `T` and `R + T`. Its only
  arbiter was `tests/test_fdtd.py`, against **wavebit's own** RCWA path, which shares those
  conventions with it by construction: a convention wrong in both moves nothing.
* **per-order complex amplitudes on a two-dimensional order grid, for either solver.**
  `docs/amplitude-sp-frame.md` §2 names what every golden here compares — efficiencies, or
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
`docs/conventions.md` says every coordinate this library takes or hands back is measured from
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
had not made the same choice — which is the same instrument `docs/fdtd.md` §5 records finding
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

That is worth having as more than a caveat. `docs/fdtd.md` §8g measures wavebit's own crossed
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
comparisons in `tests/test_fdtd.py`. Five mutants, applied and reverted, are what says the
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
where evidence lives. `docs/fdtd.md` §2 prints a three-way agreement — the RCWA path, this
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
are all sized above it deliberately. Measured, on `tests/test_fdtd.py::_grating()` at
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
for two unrelated reasons, and `docs/fdtd.md` §2 now prints the re-derived values with a
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
`tests/test_docs_contract.py::test_every_symbol_a_document_names_exists` checks, and a bare
backticked name is checked by nothing. All three names this paragraph carried until
2026-09-07 were stale — renamed while the sentence stayed put, which is the defect this whole
row is about, one level up. The same rewrite was made to the three that `docs/fdtd.md` §2
names.

#### What this row adds that the crossed one does not

The crossed row reaches further into the solver — a 2D order grid, a conical mount, per-order
complex amplitudes on both sides of `kx = 0`. What it does not reach is the *scene a reader is
shown*: a 1D grating at normal incidence in the pure TE/TM formulation, which is a different
code path (`_extract` rather than `_extract_crossed`) and the one the headline claim, the
tutorials and most of `tests/test_fdtd.py` run on. Three things follow from the scene being 1D
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
verdict — `docs/amplitude-sp-frame.md` is where that question lives, and nothing here settles
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
to it — the same shape as the inkstone and 2ff-Voigt failures recorded below. What caught it was
regeneration-vs-committed, which is worth doing whenever the bridge is touched, and is the only
check that would have.

The polarisation is now read from whichever spelling the case uses, `tetm` first. Verified the
only way this can be verified: regenerating produces all 72 arrays byte-identical to the
committed file. Separately, the `sys.path` line pointed one level *above* the repo root, so the
invocation in the docstring failed with `ModuleNotFoundError` before reaching any of this.

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
oracle here**, and `tests/test_anisotropic_berreman.py` is still the only judge of it. The cases
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
it conserves energy to 1.3e-15 across thirty lossless configurations, and it reproduces our 4ff
to 1e-15 on every z-*decoupled* case including conical incidence — so the harness is sound and
nothing looks wrong. On z-coupled tensors it is 4e-4 to 3e-2 out, unmoved by `num_g` from 1 to
21, and on the four cases inside RETICOLO's validated region **RETICOLO and our 4ff agree to
1e-15 while inkstone alone departs**. Energy conservation cannot catch this: the tensor it
actually solves is a valid lossless one, which is the same reason our own 2ff path returns
`R + T = 1` on a Voigt tensor while being wrong (`test_anisotropic_berreman.py`).

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
about to go out. `docs/fourier-truncation-shape.md` still discusses GRCWA's *algorithm*, which is
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

These solvers are not vendored — nothing in this repository contains their implementations,
none of them is a dependency, and the committed goldens let the suite run without any of them
installed. What *is* here is our own adapter code plus the numbers they produced. Their terms
still ask for credit, and RETICOLO's ask is explicit.

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

**S4.**

> V. Liu and S. Fan, *S4: A free electromagnetic solver for layered periodic structures*,
> Comput. Phys. Commun. **183**, 2233–2244 (2012).
> [doi:10.1016/j.cpc.2012.04.026](https://doi.org/10.1016/j.cpc.2012.04.026)

**fmmax** (MIT licence).

> M.F. Schubert and A.M. Hammond, *Fourier modal method for inverse design of
> metasurface-enhanced micro-LEDs* (2023). [arXiv:2308.08573](https://arxiv.org/abs/2308.08573)

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
| [Zenodo, V10](https://zenodo.org/records/14631951) — the official download | **CC BY 4.0**, which *does* permit redistribution with attribution |
| [Zenodo, V9](https://zenodo.org/records/5905381) and [V8](https://zenodo.org/record/3610175/export/json) | no licence declared; the `rights` field is empty |

A GPL-2.0+ attribution circulates in search summaries. It is **not** in any record above, and
the GitHub repository the Zenodo entries are named after (`Light-in-complex-nanostructures/RETICOLO`)
returns 404, so it cannot be checked. It is not relied on here.

**The policy does not change: RETICOLO stays outside this repository.** That is safe under
every reading, and vendoring a 184-file MATLAB tree would buy nothing — version pinning is
what `RETICOLO_PATH` is for. What changed is the *reason*: it is a deliberate choice, not a
prohibition we had verified. If redistribution ever becomes useful, CC BY 4.0 on the V10
record is the thing to re-read — noting that Creative Commons itself advises against applying
CC licences to software.

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

## Which file does what

| Piece | Needs |
| --- | --- |
| `interface/` — the RETICOLO bridge (`Reticolo.py` + our `.m` wrappers) | Octave or MATLAB, and RETICOLO |
| `gen_goldens_*.py` — freeze the reference numbers | the oracle; run by hand, rarely |
| `golden_*.npz` — the frozen numbers, committed | nothing |
| `test_*_golden.py` — the assertions | nothing; **run in CI everywhere** |
## A golden frozen ahead of its consumer

`golden_fmmax_magnetic.npz` holds rows nothing in this solver can consume — permeability has no
slot anywhere (`docs/anisotropic-materials.md` §5). It was frozen anyway, on purpose: **an oracle
chosen after the work is an oracle chosen to agree.**

What keeps that honest is that every case is generated twice, once with its permeability and once
at `mu = 1`, and the `mu = 1` half *is* held to the solver today. If the harness, units or
conventions were wrong, that subset would fail now, while the answer is independently knowable.
The magnetic half carries a physical guard instead of a comparative one: every case is Hermitian
in both tensors, hence lossless, so `R + T = 1` is enforced on the frozen numbers.

### Crossed material reference scope

The three material scenes are specified in `_meep_crossed_material_cases.py` and documented
in `docs/fdtd-alpha.md`. Their dispersive active permittivity has positive real part at the
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
changing their values. `docs/fdtd-alpha.md` records the discrepancy and convergence;
a reference with insufficient resolution is not grounds to loosen the comparison bound.
