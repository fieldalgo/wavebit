# Connecting algorithms — ETM, SMM, and why a plain transfer is not on the list

Written: 2026-08-11. Measured on an Apple M4, NumPy 2.5.1 against Accelerate, float64, single
thread as configured. Times are per solve, averaged over 3–8 runs after two warm-up calls; the
warm-up matters, and leaving it out reported SMM as **twice as fast** as ETM on the smallest case
before the first-call overhead had been paid.

Targets:
- `wavebit/_core/engine.py` — `_SOLVES`, where `layer_algorithm` becomes a function
- `wavebit/_core/solve_2d.py`, `solve_1d.py` — the ETM recursion (`big_A_i`, `big_T`)
- `wavebit/_core/scattering.py` — the SMM cascade
- `wavebit/_core/solve_2d_4ff.py` — all three, on the first-order path
- `wavebit/_core/conventions.py` — `etm_mode_q` / `layer_root`, the one place the two algorithms
  label a layer's modes differently (§10), and `layer_label_rule` / `power_down_labels`, how a
  double-negative layer is labelled (§11)

**The keyword is `layer_algorithm`** (it was `connecting_algo` until 2026-08-20; the values
`'ETM'` / `'SMM'` / `'TMM'` did not move). "Connecting algorithm" stays the name of the *idea*
throughout this document, because that is what the literature calls it; the keyword says what
it selects — how the per-layer solutions are chained into a stack — without an abbreviation.

**This document is about the connecting algorithm, not about anisotropy.** The choice is the same
question for an isotropic grating as for a magneto-optic film — it changes no physics, only how
per-layer modes are chained. Anisotropy appears here for one reason: the first-order (4ff) path
is the only one carrying **all three** algorithms, so it is the only place they can be measured
side by side. The shipped 2ff solver does not offer a plain transfer at all. Treat 4ff below as
the test bed, not the subject; what that path *is* and what it can do is
[`first-order-4ff.md`](first-order-4ff.md).

**What each section was run on**, because the numbers below do not all have the same scope and
the difference matters more than any of them:

| § | solve path | structure | patterned? | varies | held |
|---|---|---|---|---|---|
| 3 | 2ff | square pillar array, `n = 2.6` in air | **yes** | truncation_order 3–7, 1–12 layers | 17°/31°, 200 nm |
| 4 | 4ff | uniform slab, `n = 1.5 + 0.1j` | no | thickness 1 µm – 10 mm | 35°, one order |
| 5 | 4ff | uniform slab, `n = 1.5` | no | truncation_order 0–4 | 35°/40°, 220 nm |
| 5 | 2ff | pillar array **and** uniform slabs | **both** | truncation_order 1–5 | 35°/40°, 220 nm |
| 6 | 4ff | Voigt + rotated-uniaxial stack | no | 2–160 layers | 35°, one order |

The 4ff rows are unpatterned throughout, and not by choice: **a patterned anisotropic layer has no
factorization yet**, so the 4ff path cannot be handed one. Where a conclusion drawn from a uniform
case has to carry over to a grating — §5 — it is checked on the 2ff path, which does have
gratings, rather than assumed.

---

## TL;DR

| | 2ff (shipped) | 4ff | stability | cost |
|---|---|---|---|---|
| **`'ETM'`** | default | default | unbounded | 1.00x |
| `'SMM'` | yes | yes | unbounded | 1.25–1.57x |
| `'TMM'` | **not offered** | diagnostic only | refuses past modal exponent 9 | 0.94x |

1. ETM and SMM are **the same answer**, to 8e-16 on a real grating. Choosing between them is a
   cost and structure decision, never an accuracy one. At a complex in-plane wavevector that
   took ETM labelling its layers' modes by the half-space's rule (§10); before 2026-10-02 it
   lost up to 12 digits there. A double-negative layer is labelled by where its modes' power
   goes (§11); before 2026-10-04 ETM raised `LinAlgError` on a quarter of the stacks that rest
   one on its own medium. The one stack ETM cannot write is Pendry's perfect-lens pair, which
   it names (§11).
2. ETM is cheaper everywhere measured, so it is the default on both paths.
3. A plain transfer matrix is not a third option you would ever pick. It cannot run a grating at
   all — not because layers are thick, but because **diffraction orders are evanescent**, and it
   is already refusing at 25 orders on a 220 nm layer.

## 1. What is being chosen

Every RCWA solve has two halves. First each layer's eigenmodes, which all three algorithms share
— they are the same `eig` on the same operator. Then those per-layer modes are **joined** into a
stack response, and that is the connecting algorithm. It changes no physics, only the arithmetic
by which the layers are chained.

The reason it is a choice at all is that the obvious chaining does not work.

A note on the names, since the literature is looser than this codebase can afford to be: the
enhanced transmittance recursion is a transfer-matrix method too, so calling it `'ETM'` and
reserving `'TMM'` for the plain, unstabilised transfer is a distinction made here on purpose. The
public 2ff and 4ff paths offer `'ETM'` and `'SMM'`, including field reconstruction.
`'TMM'` exists only in the internal 4ff diagnostic helper; neither public path accepts it.

## 2. Why the obvious one fails

Inside a layer some modes decay along `+z` and others along `-z`. Write the layer as a transfer
matrix — field at the bottom from field at the top — and both live in one matrix:

```
        ┌ exp(-x)      ·    ┐          x = k0 * |Im(lam)| * d,  the modal exponent
        │    ·      exp(+x) │          per layer and per Fourier order
        └───────────────────┘
```

Inverting that loses relative precision like `eps * exp(2x)`. Measured against the exact Airy
slab, with a bound deliberately placed where accuracy fails rather than where the arithmetic
overflows:

| modal exponent | predicted `eps·exp(2x)` | plain transfer, measured |
|---|---|---|
| 5.7 | 2e-11 | 7.5e-13 |
| 9.1 | 1.8e-8 | 1.2e-8 |
| 15.9 | 1.5e-2 | 8.5e-4 |
| 34.0 | — | 5.1e+25 |

The prediction and the measurement agree, which is what makes the bound derivable rather than
tuned. `wavebit/_core/solve_2d_4ff.py::_MAX_EXPONENT` is set from it.

**ETM removes the growing term instead of cancelling it.** Sweeping upward with the state below a
layer written as `psi = G t`, decomposing in that layer's modes gives `a_f = X_f⁻¹ P t` — and
`X_f⁻¹` is precisely the growing factor. Absorbing it into the unknown leaves

```
    G_above = Psi_f + Psi_b · X_b · (Q P⁻¹) · X_f
```

in which `X_f` and `X_b` appear only as the bounded, decaying factors they are. **SMM gets there
differently**: writing outgoing waves in terms of incoming ones means every exponential is a
decay by construction, and layers compose by the Redheffer star product.

Both rest on the same physical fact — a mode carrying energy towards `+z` cannot gain amplitude
on the way, so `Im(lam_f) >= 0` — which is why neither is stable in a **gain** medium. There the
growth is the physics rather than an artefact, and nothing guards it. A complex in-plane
wavevector breaks the same fact without any gain, and §10 is what that costs ETM.

## 3. They agree

A square pillar array, `top = 1.0`, `bot = 2.1025` (vacuum over an `n = 1.45` substrate), 17°
at azimuth 31°, four 200 nm layers unless noted:

| truncation_order | orders | layers | \|ETM − SMM\| | R + T | ETM | SMM | SMM/ETM |
|---|---|---|---|---|---|---|---|
| 3 | 49 | 4 | 7.8e-15 | 1.000000000 | 50 ms | 69 ms | 1.37 |
| 5 | 121 | 4 | 3.8e-15 | 1.000000000 | 232 ms | 364 ms | 1.57 |
| 7 | 225 | 4 | 7.2e-15 | 1.000000000 | 1265 ms | 1974 ms | 1.56 |
| 5 | 121 | 1 | 1.3e-15 | 1.000000000 | 97 ms | 121 ms | 1.25 |
| 5 | 121 | 12 | 7.8e-16 | 1.000000000 | 818 ms | 1161 ms | 1.42 |

`scattering.py` records this ratio as ~1.1–1.4x. Re-measured here it is 1.25–1.57x — the same
picture, a little dearer, on different hardware and a different structure. Nothing in the choice
turns on the difference.

On the 4ff path, over five tensors × three incidences × both polarisations: **ETM vs SMM 3.3e-15,
ETM vs plain transfer 1.8e-15** wherever the plain transfer is willing to answer at all.

## 4. Thickness

4ff, `n = 1.5 + 0.1j` at 35°, against the exact Airy slab. The third column is the **answer** —
the transmittance the closed form gives — and the last three are each algorithm's **relative
error** against it:

| d (nm) | exponent | *T* (exact) | ETM err | SMM err | plain err |
|---|---|---|---|---|---|
| 1 000 | 1.1 | 0.0900 | 1.4e-15 | 3.1e-15 | 2.9e-15 |
| 5 000 | 5.7 | 1.1e-05 | 1.1e-14 | 1.3e-14 | 7.5e-13 |
| 10 000 | 11.3 | 1.3e-10 | 2.3e-14 | 2.5e-14 | refuses |
| 30 000 | 34.0 | 2.6e-30 | 8.4e-14 | 8.5e-14 | refuses |
| 100 000 | 113.3 | 3.5e-99 | 2.5e-13 | 2.6e-13 | refuses |
| 10 000 000 | 11 328 | 0 | exact 0 | exact 0 | refuses |

The medium absorbs, so *T* collapses with thickness: at 1 µm the exponent is 1.1, the single pass
keeps `exp(-2x) = 0.10` of the power, and after the interfaces 9% emerges (the other 83% is
absorbed, 8% reflected). By 30 µm the answer is 1e-30 and by 100 µm it is 1e-99 — numbers no
transfer matrix can reach by cancellation, and the reason the two stable algorithms are still
tracking a closed form there at 1e-13.

Past an exponent of about 700 the transmission underflows to zero, which is the right answer
rather than a failure — and the reflectance there is the closed-form single-interface Fresnel
value, reproduced to 4e-16.

## 5. Fourier orders, which is the real argument

Diffraction orders above the light line are evanescent, and their modal exponent grows with the
order index **whatever the layer thickness**. On a 220 nm slab of `n = 1.5`:

| truncation_order | orders | max exponent | ETM | SMM | plain | ETM | SMM |
|---|---|---|---|---|---|---|---|
| (0,0) | 1 | 0.0 | — | 2e-16 | 4e-16 | 0.2 ms | 0.2 ms |
| (1,1) | 9 | 7.0 | 4e-15 | 4e-15 | 4e-15 | 0.8 ms | 1.0 ms |
| (2,2) | 25 | 13.9 | 6e-15 | 6e-15 | **refuses** | 12 ms | 13 ms |
| (3,3) | 49 | 20.6 | 2e-14 | 2e-14 | **refuses** | 35 ms | 38 ms |
| (4,4) | 81 | 27.2 | 2e-14 | 2e-14 | **refuses** | 118 ms | 128 ms |

(Deviation columns are against the single-order answer, which a uniform layer must reproduce
exactly since it cannot diffract.)

**That table is a uniform slab, because the 4ff path cannot express a patterned anisotropic layer
at all** — so read on its own it extrapolates from an unpatterned case to a real grating. The 2ff
path does have gratings, and its modes go as `exp(-k0 q d)`, so the same exponent is directly
readable there. Measured on a square pillar array (`n = 2.6` in air, 220 nm) beside uniform slabs
of each index:

| truncation_order | orders | uniform n=1.5 | uniform n=2.6 | pillar array |
|---|---|---|---|---|
| 1 | 9 | 7.0 | 5.0 | 7.3 |
| 2 | 25 | 13.9 | 13.0 | **14.1** |
| 3 | 49 | 20.6 | 20.0 | **20.7** |
| 5 | 121 | 33.7 | 33.4 | **33.8** |

Patterning barely moves it, and neither does nearly doubling the index. The order index dominates:
a high order's decay rate is set by its `kx`, which the period and wavelength fix and the material
does not. So the uniform measurement does carry over — that is now measured rather than assumed.

`truncation_order = (2, 2)` is a *small* calculation. So the plain transfer is not "weak on thick stacks" — it
cannot run a real problem, and that is the historical reason RCWA needed the enhanced form
(Moharam et al. 1995) rather than a preference between two equivalent recursions.

## 6. Cost with stack depth

4ff, alternating Voigt and rotated-uniaxial layers, single order:

| layers | ETM | SMM | plain | SMM/ETM | plain/ETM |
|---|---|---|---|---|---|
| 2 | 0.22 ms | 0.28 ms | 0.20 ms | 1.27 | 0.92 |
| 10 | 0.78 ms | 1.04 ms | 0.73 ms | 1.33 | 0.93 |
| 40 | 2.74 ms | 3.75 ms | 2.60 ms | 1.37 | 0.95 |
| 160 | 10.95 ms | 14.46 ms | 10.26 ms | 1.32 | 0.94 |

All three are linear in depth — the per-layer `eig` dominates — and the ratios are flat, so the
choice does not change with stack size.

## 7. So when would you pick SMM

Not for accuracy, and not for stability. Two structural reasons:

- **A layer's scattering matrix depends only on that layer** and the shared reference, and the
  star product is associative. Layer responses can be built once and then cascaded, cached or
  reordered without touching their neighbours. ETM's recursion carries a state that already
  contains everything below the current layer, so it has no such piece to reuse.
- **It is the formulation most published results are stated in**, which makes it the natural
  thing to compare against a paper.

And one practical one: two independent joinings agreeing is a real check on both. Every 4ff
stability test in `tests/test_solve_2d_4ff.py` runs against ETM *and* SMM for that reason —
parametrising it is what caught SMM's block sizing being unexercised at more than one order once
ETM became the default.

## 8. The scattering matrix the cascade keeps

An SMM solve builds the stack's whole `S` — both sides, every order — in order to answer at all
(`R` and `T` are two of its blocks applied to the incident vector) and used to throw it away.
`solver.scattering` is what survives now:

```python
solver = wavebit.rcwa(layer_algorithm='SMM', answers='amplitude', ...)
solver.solve()
S = solver.scattering               # a ScatteringMatrix, or None
S.S11, S.S12, S.S21, S.S22          # half-space to half-space, incidence-relative
S.blocks                            # the four together
S.entry_port, S.exit_port           # the basis each end is expressed in
```

Three things bound it, and each is a decision rather than an implementation detail:

- **It rides `answers='amplitude'`, not a rung of its own.** An `S` *is* amplitude information,
  and the rung below said only power was wanted. It costs no arithmetic and four matrices of
  memory whatever the stack's depth — the cascade already built it.
- **Absent after an ETM solve, and it says so.** Reading `solver.scattering` when there is none
  raises, naming which of three reasons applies: the algorithm forms none, the rung keeps power
  only, or no solve has run. Two of those are declarations, where "solve again" is false advice,
  and one is a state, where it is the whole advice — the same split
  `_require_field_declaration` already makes for `reconstruct_field`. A bare `None` said all
  three at once and then surfaced as `'NoneType' object has no attribute 'S11'`. The ETM case is
  not a knob going unread: it is the shape of the algorithm the caller chose, and it is the
  *only* thing `'SMM'` buys. Everything else the two algorithms produce — amplitudes, per-layer
  state, the reconstructed field — is identical, and the ETM is cheaper at all of it.
- **It goes on the solver, not into `Result`.** `Result` is a registered pytree with fixed
  children, so carrying matrices there would change its structure and drag them across every
  `jit` and `grad` boundary.

**Its ports are the two half-spaces, not the cascade's gap.** The internal gap basis (`eps_gap`
chosen per order so `kz_gap == 1`) is a gauge, and it cancels between the entry and exit
interface factors. Re-run a solve with `eps_gap = 2.7 + k_par^2` and these blocks move 6e-15 to
1.8e-14 while a gap-embedded per-layer block moves 0.64 against its own magnitude. That is why
these numbers are directly readable and the per-layer ones are deliberately not exposed: they
would need `_ab_matrices`, the gap → half-space conversion, shipped beside them to mean
anything. The note beside `_ANSWERS_LADDER` in `_core/enums.py` carries that measurement.

**Orientation is the hazard, not the numbers.** These are orientation-blind matrices, so
handing them over without a stated frame is the same shape of defect as a backside field cell
in a mirrored frame (`conventions.md`): every number plausible, every number mirrored. They are
therefore *incidence-relative* — `S11` reflects back into the incidence medium whichever side
is lit — matching what the field cell and the `R`/`T` reads already do.

The tests pin the contract rather than the values: `S11` and `S21` times the incident column
*are* the reported `r_s` and `t_s` on either side -- and on the p block `-r_p` and
`t_p * (n/mu)_exit / (n/mu)_lit`, because its port variable is the p wave's magnetic field
(`docs/conventions.md` states it once for every path; the pure 1D matrix is the s-s or p-p
block of the conical one since 2026-09-27); `S22` and `S12` answer the *other* side's
solve, which is the point of keeping the whole matrix and the half of it the lit side never
reads; and the backside matrix matches the scene a caller would build by hand — flipped stack,
swapped half-spaces, lit from the top. The last two run at normal incidence on purpose:
`theta` is incidence-relative, so at an angle the two sides do not share `kx` and their
cascades are built on different order grids. `theta = 0` is where one matrix really does hold
both answers.

## 9. Cascading stacks

An `S` is what lets a stack be used as a *component*. `ScatteringMatrix.cascade` joins two of
them into the matrix their joined stack would have:

```python
# layers run bottom to top, so the upper piece -- the one the light reaches first -- is the tail
first  = solve(layers[2:], top=1.0,  bot=joint, theta=theta).scattering
second = solve(layers[:2], top=joint, bot=1.5,  theta=theta_in_joint).scattering
whole  = first.cascade(second)

whole.S11 @ delta                          # the ordinary answer again: r_s (-r_p on a p port)
first.cascade(second).cascade(third)       # chains, for a stack cut into more pieces
```

The two halves are solved as **independent problems** that know nothing of each other, and the
composite reproduces the stack solved whole to 1e-15 in all four blocks. The Redheffer star
product's `inv(I - second.S11 @ first.S22)` is the infinite bounce at the cut — which is why it
needs `first.S22`, the response to light arriving at the first piece's far face *from outside*,
a thing no single one-sided solve contains.

**The plane where they meet is a basis, not a layer.** A half-space has no thickness and nothing
propagates in it, so nominating one at the cut changes nothing: air, `eps = 2.25`, `eps = 12` all
give the same composite to 1e-15, and inserting a real zero-thickness air layer into the whole
stack moves it by 3.7e-16.

**It is incidence-relative, like every other orientation here.** The argument is the piece the
light reaches *next*, not the piece geometrically below. Lit from the top those are the same
thing; lit from the bottom they are opposite, and the next piece sits geometrically above.
Getting it backwards is refused rather than answered.

### Why it is worth the trouble

Two cases, and neither is exotic:

- **A thick substrate.** A grating on a 0.5–1 mm wafer cannot go in the stack — the phase across
  the wafer is meaningless at that length and coherence is lost anyway. The standard treatment
  is the grating's `S`, the back surface's Fresnel, and an *incoherent* sum of intensities over
  the bounces. That needs the grating's response from below, i.e. `S22` and `S12`. Simulating a
  sample somebody actually measured usually lands here.
- **Reuse across a sweep.** Twenty-five points moving only the last of six layers, 41 orders:
  0.243 s re-solving the whole stack each point against 0.063 s solving the fixed part once and
  cascading — **3.85x**, agreeing to 3e-15. This is the "built once and then cascaded, cached or
  reordered" of section 7, finally reachable.

### The trap, and why `cascade` refuses

Both pieces must sit on the **same order grid**. This API derives `kx` from
`n_inc * sin(theta)`, so a joint medium of a different index shifts the whole grid — the same
`theta` then means a different `k_par`. Joining across that mismatch is silently wrong:

| | reflected power | transmitted power |
| --- | --- | --- |
| `theta` converted | 0.696952 | 0.508050 |
| same `theta` (the mistake) | 0.764553 | 0.460943 |
| the stack solved whole | 0.696952 | 0.508050 |

Ten percent out, with no NaN, no negative power, and nothing to notice. It is the same
"`theta` is measured in the incidence medium" trap as backside illumination, in a new place.
So `ScatteringMatrix` carries its `Port` at each end — the retained `kx`/`ky`, the
half-space and the polarization channel — and `cascade` refuses a mismatch by name, with the
conversion in the message:

```
sin(theta_next) = sin(theta_this) * n_this / n_joint
```

The channel is the mismatch no order grid can show. `Port.channel` is `'te'` or `'tm'` on the
pure 1D path, whose ports count one polarization, and `'sp'` on the coupled ones (conical, 2D and
first-order alike, which is why a 2ff piece and a 4ff piece join). A TE and a TM matrix of the
same cut share every other field — the same orders, the same media, the same size — so until
2026-09-27 `cascade` joined them, and the result was the stack's matrix for neither: off by 0.28
to 2.3 of each block's largest entry against either polarization's whole-stack matrix, measured
on the cut `tests/test_scattering_core.py` takes apart. A pure piece against a coupled one was
refused already, but by its `ky` (`None` on the pure path), with advice to convert `theta` that
cannot help; the channel is now asked first. To join across polarizations, solve every piece on
the coupled path — `polarization` or `psi` at `phi = 0` on a 1D grating, instead of `tetm`.

Multiplying two matrices by hand with `redheffer_star` skips that check. It is internal for
that reason: it operates on gap-basis per-layer blocks, which carry no ports at all.

## 10. ETM's labels at a complex in-plane wavevector

Written 2026-10-02. NumPy, float64 unless noted.

ETM and SMM consume the same modes, but not the same **labels**. A layer mode comes as a `±q`
pair, and which root is called "down" (`+`) is a choice. In exact arithmetic it is free, and SMM
keeps it free in practice: flipping a random half of the roots in the scene below moved SMM's
`r0` by at most 1.3e-10. ETM is not free to choose.

**Why.** The recursion writes everything below an interface in terms of the layer's `+`
amplitudes, through `a = (W^-1 F + V^-1 G)/2`, and inverts it (`big_A_i`). So `a` is the inverse of
a *transmission*, from the layer's `+` modes into the medium below, and it is well conditioned only
when those modes are the waves that medium receives: the root `boundary_kz` gives the same plane
wave. A `+` mode that travels *up* turns that entry of `a` into a reflection, and an interface
that does not reflect the mode makes `a` singular. SMM never meets this, because it joins every
layer through its gap medium, whose `kz = 1` no layer mode resembles.

**When the labels part.** `mode_q` takes the decaying root, `Re(q) >= 0`. At every real `k_par` in
a passive layer that is the half-space's root too, bit for bit (4e5 random media). A complex
`k_par` (a complex `theta`) or a gain layer puts `q**2` above the real axis, and there a
propagating mode's decaying root travels up.

**The case that showed it.** A 1D grating of permittivities `[1, 25, 1, 25, 25, 25]`, period
2000 nm, 100 nm thick, vacuum above, `eps = 25` below, `truncation_order = 30`, 900 nm, conical at
`phi = 0.3`, `theta = 0.5 + 0.3j`. Its wide `eps = 25` block rests on the `eps = 25` substrate, so
the modes confined to it cross into the substrate with nothing to reflect them:

| | real `theta = 0.5` | `theta = 0.5 + 0.3j`, decaying labels | `0.5 + 0.3j`, ETM labels |
|---|---|---|---|
| cond(`a`) at the substrate | 1.2e+02 | **2.6e+15** | 1.4e+02 |
| `r0` moved by a 1e-14 nudge of `theta`, ETM | 1e-15 | **2.1e-04** | 8.1e-16 |
| same, SMM | 1e-15 | 2.0e-15 | 2.0e-15 |

The smallest singular value of `a` was 5.1e-15, carried by modes such as `q = 0.002348+4.978114j`:
a real part of 0.0023 is enough for the decaying rule to pick the upward root. A nudge of 1e-10
or 1e-6 moved ETM by as much (7.0e-05, 6.6e-05), so this was a floor, not conditioning. It needs
both a mislabelled mode and an interface it crosses unreflected. A 1 nm `eps = 1` film between
grating and substrate makes the interface reflect, and the floor went (1.9e-15, decaying labels
kept). It needs high contrast too: the same grating with permittivities 1 and 2.25 moved by
4.2e-15. A uniform layer never shows it, because its orders do not mix, so a near-zero diagonal
entry of `a` only rescales its own column.

**The rule.** `etm_mode_q` takes the half-space's root, `Re(q) > Im(q)` (Maystre's cut in `q`'s
variables, `q = -i*kz`). The exception is a root that grows across the layer by more than
`G = eps**-0.25` (8192 at double precision); there it keeps the decaying root. A growing root
costs the recursion about `eps*G**2`, because the reflection state carries the round trip. The
decaying root costs `eps/(r*G**2)`, with `r` the mode's reflection, which is at worst `1/G**2`.
The two cross at `eps**-0.25`, where either loses at most `sqrt(eps)`, about 1.5e-8. Measured on
two thick layers, as `|ETM - SMM|` against the cap:

| cap | 2 | e | 4 | 1e2 | **8192** | 1e6 | 1e8 | none |
|---|---|---|---|---|---|---|---|---|
| 50 um, `theta = 0.5+0.3j` | 6.8e-05 | 6.1e-14 | 2.0e-14 | 2.2e-14 | **2.2e-14** | 2.2e-14 | 2.2e-14 | 2.7 |
| 200 um, `theta = 1.2+0.5j` | 2.6e-06 | 2.6e-06 | 2.6e-06 | 2.6e-06 | **1.9e-12** | 2.7e-12 | 4.9e-11 | 1.6 |

Uncapped, near-cutoff modes grew by up to `e**167` across the 50 um layer. SMM's cascade keeps
`mode_q`'s labels: it needs none of this, and handed the uncapped labels it broke as well. The
field state SMM recovers for reconstruction does need it -- see the end of this section.

**What it changed.** `|ETM - SMM|` in `r0`, with the decaying labels and with `etm_mode_q`:

| scene (100 nm unless noted) | before | after |
|---|---|---|
| 1D TE, `theta = 0.5+0.3j` | 2.5e-06 | 1.2e-16 |
| 1D TM, `theta = 0.5+0.3j` | 3.8e-10 | 1.1e-15 |
| conical, `theta = 0.5+0.3j` | 5.1e-05 | 3.1e-16 |
| conical, `theta = 0.5+0.03j` | 9.1e-06 | 2.0e-15 |
| conical, `theta = 1.2+0.5j` | 5.5e-06 | 8.5e-16 |
| conical, `theta = 0.5+0.3j`, 50 um | 6.8e-05 | 2.2e-14 |
| 2D, 6x6 cell, `truncation_order = (6, 6)` | 5.1e-06 | 1.1e-14 |
| evanescent, `Re(theta) = pi/2 - 7.5e-8` | 1.0e-05 | 3.8e-15 |
| evanescent, `Re(theta) = pi/2 - 1e-4` | 3.6e-05 | 7.8e-15 |
| evanescent, `Re(theta) = pi/2 - 1e-2` | 9.8e-06 | 8.2e-15 |

The evanescent rows are `theta = Re + i*arccosh(sqrt(1.6**2 + 1.2**2 + 1e-4))` at
`phi = arctan(1.2/1.6001)`. The first offset is what the engine's float32 clamp produces, and the
same point at complex64 went from 5.1e-02 to 1.9e-06 off double precision (largest over all
orders; SMM 3.4e-06), with no branch cutoff to override: the new label reads the sign of `Re(q) - Im(q)`,
which round-off on `Re(q)` cannot flip. Every real-angle solve is unchanged to the bit (24 checked,
TE, TM, conical and real-`kx` evanescent incidence), since the two rules then pick the same root and
nothing grows. `tests/test_etm_layer_labels.py` pins the table, each case with a control that puts
the old rule back and must break it.

**What it needs from the half-space rule.** A layer can follow the half-space's root only if that
root does not grow in an evanescent order, or the layer's exponentials would. Maystre's cut keeps
every evanescent root decaying. The continuation rule it replaced on 2026-10-02 did not
(`docs/branch-cuts.md`), and under it the labels above left ETM 5.8e-06 off SMM at 100 nm. At
1 um neither algorithm was stable: SMM gave `r0 = 4.1191+3.5004j` and moved 3.2e-02 under the
nudge, where Maystre's cut gives `-0.6412-0.2681j` and 5.0e-15.

**SMM's field state takes the ETM's labels.** Added 2026-10-09. The state SMM hands to field
reconstruction (`scattering.py::_field_state`) is the ETM's: one vector per layer, its `+`
amplitudes at the top, carried down through `M`, the map from the gap below the layer onto them.
Inverting `M` presumes those modes are the waves the region below receives, which is the same
presumption `a` makes, and the gap does not protect this step: `M` maps out of the gap into the
layer, not across it. On the decaying labels a vacuum layer between vacuum half-spaces at
`theta = 0.5+0.3j` had every `+` amplitude zero and its whole field in the `-` amplitudes, which
the state has no slot for. `A_i` came out at 1e15 and `T1` at zero, and the s-polarized field had
`|E_z|/|E| = 6e-2` where it is zero, while `R` and `T` were exact. 2ff ETM, and 4ff under either
algorithm, were right.

So where the two rules may part (a complex `k_par`, a gain layer, a layer that may hold a double
negative) SMM rebuilds the layer on the ETM's labels for its field state, one more `eig`
(`scattering.py::_field_labels_differ`): on a two-layer 2D raster at `truncation_order = (6, 6)`
an SMM field solve cost 1.66x the ETM's at a real angle, where nothing is rebuilt, and 2.46x at
`theta = 0.5+0.3j`. The reflections below each split plane are label-free
and are still read off the cascade. The cascade itself keeps `mode_q`, and that is measured
rather than inherited. Handed the ETM's labels, SMM's `r0` at 200 um and `theta = 1.2+0.5j` sat
4.7e-10 off the ETM and moved 2.8e-9 under the 1e-14 nudge, against 1.9e-12 and 7.7e-12 on its
own labels. Keeping them also leaves every amplitude of a field solve identical to the bit to
the same solve declared below `'field'`.

Worst residual against a closed-form plane wave inside a reflectionless uniform layer (the layer
on its own medium under vacuum, `truncation_order = 0`, s and p, all six components), and SMM's
field against the ETM's on two-layer gratings at `theta = 0.5+0.3j` (`[1, 25, 1, 25, 25, 25]` over
2000 nm, a 6x6 raster for 2D, on `eps = 25`):

| SMM 2ff, worst over s and p | before | after |
|---|---|---|
| vacuum on vacuum, `0.5+0.3j`: conical / 2D / TE / TM | 0.75 / 3.5 / 0.80 / 0.24 | <= 1.6e-15 |
| `eps = 25` on its own substrate, `0.5+0.3j`: conical / 2D / TE / TM | 12 / 230 / 11 / 23 | <= 3.0e-15 |
| gain, `eps = 25-0.5i` on itself, real `theta = 0.5`: conical / 2D / TE / TM | 1.1e4 / 1.5e6 / 11 / 0.57 | <= 3.6e-15 |
| `eps = 4`, `mu = 2.25` on itself, `0.5+0.3j`, 2D | 4.2 | 1.1e-15 |
| grating vs ETM: conical / TE / TM / 2D | 1.5e5 / 1.7e-3 / 5.1e-7 / 5.8e-8 | <= 4.7e-14 |
| double negative `(-4, -1)` on itself, `theta = 0.5-0.3j`, 2D, vs ETM | 2.0 | 5e-16 |

The last row is why a layer that may hold a double negative is rebuilt without asking: there
`q**2` lies below the real axis, the plain rule's check calls the labels equal, and the mirror
flips them. At a real `k_par` nothing is rebuilt -- the two rules pick the same root, to the bit
-- and 134 arrays (field cells, `T1`, every layer record and amplitude over TE, TM, conical and
2D, both incidence sides, a magnetic raster and a uniform double negative) came out
byte-identical to the code before. `tests/test_smm_field_labels.py` pins the table.

**What is left.** The cap's worst case, `sqrt(eps)`, needs a mode that grows by about 8192 across a
layer *and* crosses an interface with a rounding-level reflection; no scene measured reached it.
Far thicker layers are out of reach for both algorithms: at 1 mm and `theta = 0.5+0.3j`, SMM itself
moved 3.9e-02 under the nudge. The 4ff path splits its modes in `split_forward_backward` by the
same partition (attenuation where decay dominates, energy flow otherwise), without a cap.

## 11. Double-negative layers: the label follows the power

Written 2026-10-04. NumPy, float64 unless noted.

§10 gave every ETM layer the root a half-space of the same medium takes, Maystre's cut. A
half-space whose permittivity and permeability both have negative real parts takes the *mirror*
of that cut ([`branch-cuts.md`](branch-cuts.md) §9), because such a medium carries its power
against its phase (V. G. Veselago, *Sov. Phys. Usp.* **10**, 509 (1968)). Its layers kept
Maystre's cut until this section, and that was wrong for the same reason. The rest of the
subject -- the half-space root, the signed index, the incidence angles, refusals and evidence --
is [`negative-index-media.md`](negative-index-media.md).

**Words used below.** A *double negative* (DNG) is a medium with `Re(epsilon) < 0` and
`Re(mu) < 0` -- for a tensor, every diagonal entry of each in the lab frame (the principal
values when the tensor is diagonal). A layer is *uniform* when its material is the same
everywhere in the cell (its convolution matrices are `c*I`), *patterned* otherwise. The *flux* of a mode is the time-averaged Poynting `z` component it carries on its own.

### What went wrong

At a real `k_par`, a passive double negative puts `q**2 = k_par**2 - eps*mu` in the closed
*upper* half plane, since `Im(eps*mu) = Re(eps) Im(mu) + Im(eps) Re(mu) < 0`. There Maystre's root
(`Re(q) > Im(q)`) is the one that advances its phase downward -- and in a double negative that
wave carries its power *up*. So a double-negative layer's `+` modes were the waves leaving the
region below, not the ones it receives, and `a = (W^-1 F + V^-1 G)/2` (§10) was the projection of
the received field onto waves going the other way. A layer resting on a medium that receives its
waves without reflection -- the same medium -- made `a` exactly singular.

| uniform DNG layer on its own DNG half-space, 720 scenes | ETM `LinAlgError` | SMM `LinAlgError` |
|---|---|---|
| `(-2, -1)`, lossless | 63 | 12 |
| `(-6, -1.5)`, lossless | 105 | 0 |
| `(-2+0.1i, -1+0.1i)` | 3 | 0 |
| `(-3+0.1i, -0.5+0.05i)` | 0 | 0 |
| after this section, all four | **0** (worst ETM-SMM 8.7e-16) | **0** |

The scan ran the TE, TM, conical, 2D and 4ff paths, vacuum and glass superstrates, three
(period, wavelength) pairs, 37/150/900 nm, real and complex angles. By path the ETM failures were
63 TE, 66 TM, 24 conical, 18 2D, and none on 4ff, which has always split its modes by flux
(`solve_2d_4ff.py::split_forward_backward`). Where the zero did not come out exact the answers
were still right -- a uniform layer's orders do not mix, so a tiny diagonal entry of `a` only
rescales its own column -- which is why this stayed hidden behind the cases that raised.

**SMM met it too, at its gap medium.** SMM's labels are free in exact arithmetic (§10), and
`mode_q` spends the freedom on keeping `exp(-k0*q*d)` bounded: `Re(q) >= 0`, ties broken by
`Im(q) <= 0`. That tie is the phase-down root, so a lossless double negative's propagating modes
were labelled against their power. SMM joins every layer through a gap of `eps = 1 + kx**2`,
`mu = 1`, whose `kz` is 1 on every order; for `eps = -2`, `mu = -1` at `kx = 1` the layer's mode
has `kz = 1` as well, the layer is the gap's exact mirror, and the gap-to-layer factor `A` was
exactly singular. Period 1000 nm at 500 nm puts orders +-2 there at normal incidence: the 12 SMM
failures above. A mode labelled by its power can never make `A` singular -- the gap's downward
wave carries power down, and no combination of upward-power modes can match it.

**A patterned layer holding a double negative** has modes of both kinds and modes between them.
On 360 random patterned scenes (TE, TM, conical, 2D; double-negative backgrounds with ordinary
inclusions and the reverse, fill 0.03 to 0.97, one or two layers, on double-negative, glass or
vacuum half-spaces, either side lit), ETM against SMM:

| | scenes | before: `LinAlgError` | before: worst finite | after: `LinAlgError` | after: worst |
|---|---|---|---|---|---|
| oblique, real or complex angle | 274 | 15 | 2.7e-09 | **0** | **6.1e-13** |
| exactly normal incidence | 86 | 9 | 1.4e-03 | **0** | 1.4e-03 (see below) |

And on §10's own scene with its `eps = 25` block made the double negative `(-25, -1)` --
permittivities `[1, X, 1, X, X, X]` over 2000 nm, 100 nm thick, on a half-space of `X`,
`truncation_order = 30`, 900 nm -- where the wide block rests on its own medium:

| `|ETM - SMM|`, all amplitudes | TE | TM | conical, `phi = 0.3` |
|---|---|---|---|
| `theta = 0.5`, before | 6.6e-07 | 1.0e-05 | 1.4e-05 |
| `theta = 0.5`, after | 1.2e-14 | 5.5e-15 | 6.4e-14 |
| `theta = 0.5+0.3i`, before | 7.5e-07 | 2.7e-04 | 1.2e-04 |
| `theta = 0.5+0.3i`, after | 4.0e-14 | 1.2e-14 | 1.5e-13 |

### The rule

| layer | ETM's `+` root | SMM's `+` root |
|---|---|---|
| no double-negative region (every layer before this section) | Maystre's cut, capped (§10) | decaying; tie `Im(q) <= 0` |
| uniform double negative | the **mirror**, `Re(q) + Im(q) > 0`, capped | decaying; tie `Im(q) >= 0` |
| patterned, holding a double-negative region | **per mode**: propagation-dominated (`|Im q| > |Re q|`) with resolvable flux, the root whose flux is positive; otherwise the decaying root; capped | decaying; tie broken by the flux |

SMM's column is its cascade's. Its field state takes the ETM's column (§10, the end).

* **Uniform: the half-space's own rule.** `etm_mode_q(..., mirrored)` is
  `conj(etm_mode_q(conj(q**2)))` before the cap, to the bit, and equals `-i*kz` of the
  half-space's mirrored root (`boundary_kz` with the medium) on every order, real or complex
  `k_par`. So a layer of the half-space's own medium has `a = I` again, as an ordinary layer on
  its own substrate does. `mode_q(..., mirrored)` mirrors only the tie: a lossless propagating
  mode is labelled `Im(q) >= 0`, by its power.
* **Patterned: the flux.** `power_down_labels` reads each mode's flux,
  `-Im(sum(X * conj(Y)))/2` with `(X, Y) = (H_s, E_s)` for an s wave and `(F_p, H_p)` for a p
  wave -- the boundary variables a substrate seed is written in, so a downward wave of a lossless
  half-space comes out at `Re(kz/mu)/2` and `Re(kz/eps)/2`, positive (`solve_1d.py::mode_flux`,
  and `solve_2d.py::mode_flux_2d` for the unrotated 2D pair). "Resolvable" is the 4ff split's own
  threshold, `|flux| > 1e-12 * sum(|psi|**2)`. Where decay dominates the flux is not read: a
  truncated operator can give an evanescent mode a small flux of either sign, which is the reason
  the 4ff split reads attenuation there, and following it would hand ETM a growing root. ETM then
  applies §10's cap; SMM never takes a growing root and lets the flux break only `mode_q`'s tie.
* **Everything else is untouched, structurally.** The engine reads each layer's materials
  (`engine.py::CoreFactorizationMixin._layer_double_negative`: does the layer hold a point
  where both real parts are negative, and is it one medium) and hands the kernels a
  `DoubleNegativeLayer(holds, uniform)` per layer. A layer that holds none takes the very
  arithmetic it had: the builders get `(mirrored, patterned) = (None, None)` from
  `layer_label_rule` and make the call they always made.

**Why per mode, and why the flux.** A patterned layer has no one medium to mirror. Each single
cut fails one limit: Maystre's sends a backward mode up, which is everything above, and the
mirror sends a forward one up -- a glass layer with one pixel of double negative, labelled by
the mirror, resting on glass. Labelling by decay, which is what SMM does, is right for every
lossy mode at a real `k_par` (a mode's flux decays in the direction it travels) but leaves the
lossless tie open and, at a complex `k_par`, sends propagating modes up: §10's failure. Measured
on a 1D harness that swapped only the labels of double-negative layers (400 scenes: blocks of
double negative on their own medium, ordinary blocks with double-negative inclusions on theirs,
near-uniform layers of both kinds, lossless mixtures, real and complex angles, 150 nm and
1.5 um), `|ETM - SMM|`:

| labels for a double-negative layer | `LinAlgError` | worst otherwise | `(-25, -1)` block, `0.5+0.3i`, TE / TM |
|---|---|---|---|
| Maystre's cut (before) | 14 | 1e-10 | 7e-07 / 3e-04 |
| decaying, `Im(q) <= 0` tie | 12 | 1e-10 | 1e-10 / 4e-11 |
| decaying, tie by flux | 4 (uniform, complex angle) | 9e-12 | 1e-10 / 4e-11 |
| mirror cut for any layer holding one | 0 | 9e-12 | 3e-14 / 1e-14 |
| **mirror if uniform, else the flux per mode** (shipped) | **0** | 2e-10 | **4e-14 / 1e-14** |

The mirror for every layer measured well, but it is the wrong cut for the ordinary modes of a
layer that is mostly ordinary, and is saved there only by the cap; the flux has no such limit.
Its one cost is the cap's: on lossless double-negative/glass mixtures at `theta = 0.3+0.05i` it
takes power-down roots that grow, up to `eps**-0.25`, and lost 2e-10 where decaying labels lost
2e-12 -- the trade §10 made for ordinary layers, made the same way. (At the real `theta = 0.5`
the block's lossless modes also show why the tie matters: decaying labels with `mode_q`'s tie
were 2e-06 / 1e-05 off.)

### What it costs, and what stays the same

* **Every scene without a double-negative layer is byte-identical** to develop (`0dfc4ce`):
  108,640 NumPy arrays of magnetic-layer scenes (ordinary magnetic, lossy, gain, magnetic metal,
  mu-negative, eps-negative beside mu-negative, the `Re = 0` seams; TE, TM, conical, 2D, 4ff and
  an oblique lattice; ETM and SMM; real, normal, complex and grazing angles; both sides; fields,
  scattering blocks and `solve_conv` relays), 38,652 eager JAX and 38,652 eager PyTorch, and 48
  `jax.jit` arrays: 24 with the raster itself traced, 24 with `theta` traced over a concrete
  raster. The engine reads materials under the JAX backend's compile-time evaluation, so a
  concrete raster gives a Python `False` even inside a trace and no `where` is staged for an
  ordinary layer.
* A traced double negative (`jax.jit` over the raster) is selected with `where`, never branched
  on: ETM through the trace matches NumPy to 6.1e-14 on uniform and patterned layers.
* Cost: a patterned double-negative layer pays for its modes' flux, `O(n**2)` (plus two matrix
  products on the conical path, whose partner blocks are formed later); a uniform one pays
  nothing; no other layer pays anything.
* `solve_conv` takes matrices, not materials. It reads the scene's own materials when the layer
  counts match, so `solve_conv(*eps_conv_matrices(), mu=mu_conv_matrices(), ...)` is `solve()` to
  the bit for a double-negative scene too; a layer the scene calls ordinary says nothing there, and
  a uniform double negative is then read off its matrices (`c*I`). A label moves no exact answer,
  only conditioning, so a matrix that is not the scene's costs digits at worst, never the answer.

### Pendry's pair: a double negative beside its exact negation

A uniform layer of `(eps, mu)` next to a medium of `(-eps, -mu)` -- the textbook perfect lens,
`(-1, -1)` in vacuum, is one (J. B. Pendry, *Phys. Rev. Lett.* **85**, 3966 (2000)). On every
evanescent order the two decaying roots are equal and the admittances `kz/mu` and `kz/(-mu)` are
each other's negatives.

**Is it singular?** The *single interface* is: `r = (Y1 - Y2)/(Y1 + Y2)` has a pole at every
evanescent `k_par`, a surface mode on each. A *layer* of it is not. A finite layer on its mirror
carries only the mode that grows toward the mirror, and transmits an evanescent order as

```
t = 2 ka/(ka + kn) * exp(kn * k0 * d),    ka, kn = sqrt(k_par**2 - eps_above*mu_above), sqrt(k_par**2 - eps*mu)
```

finite at any thickness -- Pendry's amplification. SMM's scattering matrix has it order by order
(`(-2, -1)` on `(2, 1)`, wavelength 600 nm, period 700 nm, vacuum above): every evanescent order
to 2e-15 relative at 30 and 100 nm and to 5e-10 at 300 nm (amplification 4.3e4, round-off growing
with `exp(2*kn*k0*d)`); at 600 nm the highest orders fall short of the closed form (7.4e8 against
1.9e9), where `exp(2*kn*k0*d)` passes `1/eps`. The plane wave on uniform layers excites none of
those orders, and its answer is finite, exact, and the limit of the lossy one: loss `delta` in
both parameters moves the amplitudes linearly, by 0.27, 0.92 and 2.8 times `delta` at 30, 100
and 300 nm. What is *not* finite is the limit over orders. The amplification grows without
bound with `k_par`, so wherever a patterned layer excites evanescent orders the lossless answer
does not converge in `truncation_order` -- specular `R` of a grating on a 100 nm lens went
0.1011, 0.0921, 0.0873, 0.0856 at truncation 5, 10, 20, 40 -- and loss is what bounds it: at
`delta = 1e-3` the same sequence settles at 0.0861. Each loss cuts the amplification off where
`delta * exp(2*kn*k0*d)` reaches 1 (D. R. Smith et al., *Appl. Phys. Lett.* **82**, 1506 (2003)).
So the pair is physical with loss, a lossless answer is a fixed-truncation answer, and SMM gives
that answer.

**What ETM does.** Its `a` for the layer is `(1 + Y_below/Y_layer)/2`, and that is zero on every
evanescent order under the decaying label, the mirror's included. The growing label would make
it 1, but it grows by `exp(kn*k0*d)` -- the amplification itself -- which passes §10's cap at
high orders, where the recursion would then lose `eps*G**2`. Floating point usually leaves `a` a
round-off away from zero, and then the answer is still SMM's: to 1e-13 on every configuration
measured (1D and 2D, conical, lit from either side, the pair at the exit or inside the stack, with
and without a grating, 30 nm to 2.5 um). Where the zero comes out exact -- uniform stacks with
the mirror at the exit on the pure TE/TM path, the lens with its mirror on both sides, the 2D
path at some truncations -- the inversion fails, and the solve now says why instead of retrying
the wavelength (the zero does not move with it):

```
ValueError: layer_algorithm='ETM' could not solve this stack: layer 0 is a uniform double
negative (eps = -2+0j, mu = -1+0j) beside the bottom half-space, made of its exact negation --
the perfect-lens pair ... layer_algorithm='SMM' solves it ...
```

raised from the `LinAlgError` (NumPy, PyTorch) or the non-finite result (JAX) it explains. The
double negative may be the substrate instead: an ordinary layer of `(2, 1)` on `(-2, -1)` is the
same pair, ETM's `a` for the layer is zero in the same way, and the message then says "the bottom
half-space is a uniform double negative ... beside layer 0" (until 2026-10 it looked only for a
double-negative *layer*, and this stack raised a bare `LinAlgError` on the TE, TM and 2D paths).
SMM's answer there is a closed form: the layer cancels an equal thickness of the substrate (J. B.
Pendry and S. A. Ramakrishna, *J. Phys.: Condens. Matter* **15**, 6345 (2003)), so the stack
reflects exactly as vacuum onto `(-2, -1)`, which is the `R` of vacuum onto `(2, 1)` and of the lens
the other way round. It is read off the convolution matrices and the half-spaces, so `solve_conv`
gets it too; under
`jax.jit` there is nothing to read and ETM returns what the inversion gave. A pair with any loss
is not exact and is solved (1e-9 of loss: ETM to SMM 1e-13).

**Both members semi-infinite is another problem, and it is refused where it has to be.** Two
lossless half-spaces that are each other's negation, with nothing between them but uniform layers of
the two media, have no scattering matrix at all: the single interface's pole at every evanescent
`k_par` is the whole stack's (B. Gralak and D. Maystre, *C. R. Physique* **13**, 786 (2012); M.
Cassier and P. Joly, arXiv:2504.00235 (2025)), and a layer of either medium only moves it (a layer
of `(-eps, -mu)` cancels an equal thickness of `(eps, mu)`). Neither algorithm answered it -- NumPy
raised `LinAlgError` without a layer, and with layers SMM returned evanescent amplitudes that were
round-off -- so the pair is refused before either runs, with the reason and its sources, wherever
the truncation retains an evanescent order. A truncation of propagating orders only never meets the
pole: uniform layers couple no two orders, every retained order is matched, and both algorithms
solve it, `r = 0` and `t = 1`. [`negative-index-media.md`](negative-index-media.md) §9 has the
measurements.

### What it does not cover

* **At and near normal incidence, a lossless double negative beside an ordinary region can be
  numerically singular on both algorithms.** Three of the 86 normal-incidence scenes above
  disagree by 2e-07 to 1.4e-03, identically before and after this section. Two of them fall below
  1e-9 at `theta = 1e-9` while the amplitudes themselves jump (0.97 to 0.73, 3.5 to 0.77): a
  nearly defective pair of modes at `kx = 0`. The third, a lossless `(-2, -1)` layer patterned
  with a metal, stays 6.9e-05 apart up to 1e-6 rad off the normal and is 5.6e-10 apart at 1e-4,
  also identically before and after. Not a labelling question, and not caught by `auto_detune`,
  since nothing raises.
* **Lossless half-and-half mixtures on the conical path** stay 1e-11 to 1.5e-10 apart (`(-2, -1)`
  beside `(2, 1.5)`, fill 0.5), before and after: the inverse rule's `[[1/eps]]` averages to nearly
  zero there and puts eigenvalues near `1e9` in the second mode family. The amplitudes swing by a
  factor of three with `theta` at 1e-9; 1e-6 of loss brings the two algorithms to 1e-13.
* **Anisotropic double negatives** count only when every lab-frame diagonal entry of both
  tensors is negative; a hyperbolic layer keeps Maystre's labels -- unless it is rotated so that
  its lab-frame diagonal is all negative, which the test reads as a double negative. So do its backward modes, if it has any,
  and so does any ordinary layer -- a plasmonic grating, say -- whose modes run backward: the
  per-mode rule is only switched on where a double negative was named.
* **An active double-negative layer** is labelled like a passive one. Unlike a half-space it is
  not refused: a finite layer keeps both roots, and only the conditioning depends on the choice.

## References

- M. G. Moharam, D. A. Pommet, E. B. Grann and T. K. Gaylord, "Stable implementation of the
  rigorous coupled-wave analysis for surface-relief gratings: enhanced transmittance matrix
  approach", *J. Opt. Soc. Am. A* **12**(5), 1077 (1995).
- L. Li, "Formulation and comparison of two recursive matrix algorithms for modeling layered
  diffraction gratings", *J. Opt. Soc. Am. A* **13**(5), 1024 (1996).
- R. Redheffer, "Difference equations and functional equations in transmission-line theory",
  in *Modern Mathematics for the Engineer*, McGraw-Hill (1961).
- V. G. Veselago, "The electrodynamics of substances with simultaneously negative values of
  epsilon and mu", *Sov. Phys. Usp.* **10**(4), 509 (1968).
- J. B. Pendry, "Negative refraction makes a perfect lens", *Phys. Rev. Lett.* **85**(18), 3966
  (2000).
- D. R. Smith, D. Schurig, M. Rosenbluth, S. Schultz, S. A. Ramakrishna and J. B. Pendry,
  "Limitations on subdiffraction imaging with a negative refractive index slab", *Appl. Phys.
  Lett.* **82**(10), 1506 (2003).
- J. B. Pendry and S. A. Ramakrishna, "Focusing light using negative refraction", *J. Phys.:
  Condens. Matter* **15**, 6345 (2003).
- B. Gralak and D. Maystre, "Negative index materials and time-harmonic electromagnetic field",
  *C. R. Physique* **13**, 786 (2012).
- M. Cassier and P. Joly, "An operator approach to the analysis of electromagnetic wave
  propagation in dispersive media. Part 2: transmission problems", arXiv:2504.00235 (2025).
