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
| 3 | 2ff | square pillar array, `n = 2.6` in air | **yes** | max_order 3–7, 1–12 layers | 17°/31°, 200 nm |
| 4 | 4ff | uniform slab, `n = 1.5 + 0.1j` | no | thickness 1 µm – 10 mm | 35°, one order |
| 5 | 4ff | uniform slab, `n = 1.5` | no | max_order 0–4 | 35°/40°, 220 nm |
| 5 | 2ff | pillar array **and** uniform slabs | **both** | max_order 1–5 | 35°/40°, 220 nm |
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
   cost and structure decision, never an accuracy one.
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
2ff path offers only `'ETM'` and `'SMM'`; `'TMM'` exists on the 4ff path alone, as a diagnostic,
and passing it to the 2ff solve raises rather than falling through to a default.

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
growth is the physics rather than an artefact, and nothing guards it.

## 3. They agree

A square pillar array, `top = 1.0`, `bot = 2.1025` (vacuum over an `n = 1.45` substrate), 17°
at azimuth 31°, four 200 nm layers unless noted:

| max_order | orders | layers | \|ETM − SMM\| | R + T | ETM | SMM | SMM/ETM |
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

| max_order | orders | max exponent | ETM | SMM | plain | ETM | SMM |
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

| max_order | orders | uniform n=1.5 | uniform n=2.6 | pillar array |
|---|---|---|---|---|
| 1 | 9 | 7.0 | 5.0 | 7.3 |
| 2 | 25 | 13.9 | 13.0 | **14.1** |
| 3 | 49 | 20.6 | 20.0 | **20.7** |
| 5 | 121 | 33.7 | 33.4 | **33.8** |

Patterning barely moves it, and neither does nearly doubling the index. The order index dominates:
a high order's decay rate is set by its `kx`, which the period and wavelength fix and the material
does not. So the uniform measurement does carry over — that is now measured rather than assumed.

`max_order = (2, 2)` is a *small* calculation. So the plain transfer is not "weak on thick stacks" — it
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
*are* the reported `r_s` and `t_s` on either side; `S22` and `S12` answer the *other* side's
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
first  = solve(layers[:2], top=1.0,  bot=joint, theta=theta).scattering
second = solve(layers[2:], top=joint, bot=1.5,  theta=theta_in_joint).scattering
whole  = first.cascade(second)

whole.S11 @ delta                          # the ordinary answer, back again
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
So `ScatteringMatrix` carries its `Port` at each end — the retained `kx`/`ky` and the
half-space — and `cascade` refuses a mismatch by name, with the conversion in the message:

```
sin(theta_next) = sin(theta_this) * n_this / n_joint
```

Multiplying two matrices by hand with `redheffer_star` skips that check. It is internal for
that reason: it operates on gap-basis per-layer blocks, which carry no ports at all.

## References

- M. G. Moharam, D. A. Pommet, E. B. Grann and T. K. Gaylord, "Stable implementation of the
  rigorous coupled-wave analysis for surface-relief gratings: enhanced transmittance matrix
  approach", *J. Opt. Soc. Am. A* **12**(5), 1077 (1995).
- L. Li, "Formulation and comparison of two recursive matrix algorithms for modeling layered
  diffraction gratings", *J. Opt. Soc. Am. A* **13**(5), 1024 (1996).
- R. Redheffer, "Difference equations and functional equations in transmission-line theory",
  in *Modern Mathematics for the Engineer*, McGraw-Hill (1961).
