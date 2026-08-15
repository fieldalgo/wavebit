# Reference cases

Validation against solvers that are not ours. The expected answer is known — agreement to
~1e-13 — so these are tests, not studies. What makes them their own directory is that
*producing* the reference needs software this project does not ship.

The arrangement is the same for every oracle: the expensive half runs once, by hand, on a
machine that has the tools; what CI checks on every push is the committed numbers.

**Regenerating a golden is not a way to make a failing test pass.** These numbers did not come
from this codebase, which is exactly what makes them worth having.

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

A fourth oracle lives outside this directory: the Berreman 4x4 in
`tests/test_anisotropic_berreman.py` is an independent *formulation* — not RCWA at all — but our
own implementation, so it sits a tier below these three.

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
| the **order set** — fmmax's parallelogramic 5x5 against our `fto` | `fto=(8,0)`, 17 orders: **1.2e-1** | `fto=(2,2)`, the same 25: **1.9e-5** |
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
7.0e-2 to 1.0e-1 across the committed cases at `fto=8`. Measured on the `grating` case at 24
degrees, it *decreases* with truncation but **not monotonically** — 3.1e-1, 2.3e-1, 9.7e-2,
1.1e-1, 3.8e-2, 4.4e-2 at `fto = 2, 4, 8, 16, 32, 64` — about 8x over a 32x increase in orders,
and nowhere near machine precision. That shape is Gibbs ringing rather than convergence
failure: the gap is set by the samples nearest a discontinuity (10 nm from one here), and the
local error there oscillates with the number of retained terms. Two things it is *not*: a
rescaling (a per-component complex fit does not remove it), and a disagreement about which side
of the interface to divide by — forming `Dx = eps*Ex` leaves a gap of the same order
(1.7e-1, 1.6e-1, 8.1e-2, 4.8e-2 at `fto = 4, 8, 16, 32`), neither better nor worse.

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
(appended, so callers asking for the first 9 are unaffected). `o` earns its keep three more
times: it settles the **x origin** (RETICOLO's docs describe a texture as running
`-period/2 .. +period/2`, and a correspondingly dead `grid_x -= period/2` sat in the bridge —
but `o` reproduces our own `ucell` on `linspace(0, period, res_x)` entry for entry, so the two
origins coincide as written and there is no shift; that line is now gone, with the measurement
in its place), it is what the generator asserts the row rule against, and on the `lossy` case it
comes back as `2+0.05j` — RETICOLO spelling loss the same way this project does.

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
