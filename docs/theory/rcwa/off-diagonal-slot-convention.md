# The off-diagonal permittivity slots — what they hold, and why an asymmetric tensor came out wrong

> **Historical API:** snippets and descriptions using `wavebit.compat` on this page require a historical checkout.
> The namespace is absent from current installations; the pre-retirement baseline is
> [revision `3c8a50f8`](https://github.com/fieldalgo/wavebit/tree/3c8a50f8a6506240169538957e99648a4b18ca3a).
> Reproduce recorded experiments with their own cited commit and environment; their measurements have not been rerun.


Written: 2026-08-10. Measured on CPU, NumPy backend, float64. The ordering defect in §4 was
repaired the same day; the measurements are kept as taken, with it in place.

Targets:
- `src/wavebit/rcwa/_engine/solve_2d.py::layer_modes_2d` — the two blocks, and why they must order the pair oppositely
- `src/wavebit/rcwa/_engine/factorization.py::_nvf_assemble` — where the sign convention is set
- `src/wavebit/rcwa/_engine/engine.py::solve_conv` — the only way to reach the slots by hand
- `src/wavebit/rcwa/_engine/solve_2d_first_order.py::_conv_layers` — where the negation is undone for the z row
- `tests/rcwa/test_uniform_anisotropic.py`, `tests/rcwa/test_anisotropic_berreman.py`,
  `tests/references/test_reticolo_anisotropic_golden.py` — every number below

Companion to [`anisotropic-materials.md`](../../guide/rcwa/anisotropic-materials.md), which answers the scoping
question ("can I give the solver an anisotropic material, and what does each gap cost?"). This one
answers a narrower and more practical question: **if I fill the slots myself — five for an
in-plane tensor, nine for a full one — what exactly am I supposed to put in them, and where does
that go wrong?**

---

## TL;DR

1. The slot named `epxy` holds **`-eps_xy`**, not `eps_xy`. Same for `epyx` — and, since the z row
   became reachable on 2026-08-13, for `epxz`, `epyz`, `epzx` and `epzy` too: **every off-diagonal
   slot holds the negative of its component**, one rule across all six. Nothing in the public
   signature says so.
2. Getting that sign backwards is **invisible to every power efficiency** — R, T, co-pol, cross-pol
   and total are all bit-identical. Only a complex amplitude changes.
3. More seriously — and this is **now fixed** — the two blocks inside `layer_modes_2d` used to
   carry the off-diagonal pair in the *same* order, so an *asymmetric* tensor
   (`eps_xy != eps_yx`, i.e. gyrotropy) took its eigenvalues from one medium and its magnetic
   field from that medium's transpose. Measured error 3.6e-2, first order in the gyrotropy
   strength. The repair exchanges the pair inside `Omega_R` only.
4. **No shipped code path could reach point 3, and none is affected by the repair.** Every tensor
   the library builds is symmetric, and for a symmetric pair the ordering is a no-op — verified
   bit-identical on real gratings. It was a latent defect blocking a future capability, never a
   wrong number in anything released.

Everything below is written as it was measured, with the defect in place. That is deliberate: the
numbers are why the repair is the right one, and they are the only record of what the failure
looked like.

---

## 1. What the five slots are

Throughout this document **block** means a sub-matrix — what `solve_2d.py::_block` assembles — not
the `modeling='block'` path. Nothing here is about modelers.

The 2D mode operator reads five permittivity blocks. They correspond to a tensor like this:

```
                       E_x         E_y         E_z
                  ┌───────────┬───────────┬───────────┐
          D_x     │    epx    │   epxy    │     ·     │
                  ├───────────┼───────────┼───────────┤
          D_y     │   epyx    │    epy    │     ·     │
                  ├───────────┼───────────┼───────────┤
          D_z     │     ·     │     ·     │  epz_i⁻¹  │
                  └───────────┴───────────┴───────────┘

     named cell  = a block layer_modes_2d actually reads
     ·           = assumed zero; there is no slot for it
```

and they arrive through the public entry point as operators about the cell centre:

```python
res = mee.solve_conv(epx_conv_all, epy_conv_all, epz_conv_i_all,
                     epxy_conv_all, epyx_conv_all)
```

The docstring calls the last two "the optional off-diagonal permittivity blocks". That phrasing is
what this document exists to correct: they are not the off-diagonal permittivity. They are its
negative.

## 2. Deriving the sign, rather than trusting a name

You do not have to take anyone's word for it — the convention falls out of the code at normal
incidence, where `Kx = Ky = 0` and the whole operator collapses.

Start from what `layer_modes_2d` builds:

```python
B  = Kx @ epz_conv_i @ Kx - I          #  ->  -I     when Kx = 0
D  = Ky @ epz_conv_i @ Ky - I          #  ->  -I     when Ky = 0
Cc = Kx @ epz_conv_i @ Ky              #  ->   0
E  = Ky @ epz_conv_i @ Kx              #  ->   0

Omega2_LR = [[Ky Ky + B epx - Cc epyx,   Cc epy - Kx Ky - B epxy],
             [E epx - Ky Kx - D epyx,    Kx Kx + D epy - E epxy]]
```

Substituting `B = D = -I` and `Cc = E = Kx = Ky = 0`, every term with a `K` in it drops:

```
                    ┌                  ┐
     Omega2_LR  =   │  -epx     epxy   │
                    │   epyx   -epy    │
                    └                  ┘
```

Now bring in the physics. A mode varies as `exp(-k0 q z)`, so `d²E/dz²` contributes `k0² q²`, and
the wave equation for propagation along `z` is `d²E/dz² = -k0² eps E`. Therefore

```
     q² = eigenvalue of Omega2_LR         and         Omega2_LR = -eps_tensor
```

Read the tensor straight off:

```
                 ┌                  ┐
     eps_t  =    │   epx    -epxy   │            eps_xx = epx     eps_xy = -epxy
                 │  -epyx    epy    │            eps_yy = epy     eps_yx = -epyx
                 └                  ┘
```

**The slot holds the negative.** This agrees with `_nvf_assemble`, which returns
`(eps_xx, eps_yy, inv(⟦eps⟧), -eps_xy, -eps_xy)` — the negation is applied there, on the way in,
and every consumer downstream has been written against it ever since.

(That tuple was ordered `(eps_xx, eps_yy, -eps_xy, -eps_xy, inv(⟦eps⟧))` when this document was
written, and was reordered on 2026-08-18 so that builders and `solve_conv` agree; see §7.1.
Nothing about the *sign* convention changed, which is what this document is about.)

So the correct hand-filled call for a uniform layer is:

```python
I = np.eye(n_orders, dtype=np.complex128)

epx   = np.stack([eps_xx * I])
epy   = np.stack([eps_yy * I])
epxy  = np.stack([-eps_xy * I])        # <-- negate
epyx  = np.stack([-eps_yx * I])        # <-- negate
epz_i = np.stack([np.linalg.inv(eps_zz * I)])
```

### 2.1 And the z row, which the same rule covers

`layer_modes_2d` has no z-row slots to derive a sign from — a populated z row selects a different
operator entirely (`solve_2d_first_order.py::layer_matrix_first_order`, which reads the tensor as
written). So for those four the convention was *chosen* rather than forced, and it was chosen to
match:

```python
epxz  = np.stack([-eps_xz * I])        # <-- negate, same as the pair above
epyz  = np.stack([-eps_yz * I])
epzx  = np.stack([-eps_zx * I])
epzy  = np.stack([-eps_zy * I])
```

The alternative — plain components for the z row, negated ones for the in-plane pair — would put
two conventions in one call, which is a worse trap than one surprising convention. `epz_i` stays
the odd one out either way: it is an *inverse*, not a negation, because that is the form the
elimination of `E_z` needs.

Unlike the in-plane pair, a wrong sign here is **not** invisible to a power efficiency (§3 is
about the pair specifically). `tests/rcwa/test_anisotropic_berreman.py` judges these four against an
oracle that takes the plain tensor, so a slot filled the other way up fails there.

## 3. Why a wrong sign will not announce itself

Suppose you forget the negation and pass `+eps_xy`. What you have actually described is the same
material with its optic axis **mirrored about the x axis**:

```
     what you meant                    what you described

        y                                  y
        │    ╱ axis at +30°                │
        │   ╱                              │
        │  ╱                               │  ╲
        │ ╱                                │   ╲
        └──────── x                        └────╲─── x
                                               axis at -30°
```

That mirrored structure is a perfectly good physical structure, and at normal incidence it has
**exactly the same power efficiencies**. Measured, on a 30° waveplate:

| quantity | `-eps_xy` (correct) | `+eps_xy` (wrong) | difference |
|---|---|---|---|
| `de_ti_s` (co-pol) | 0.715686772101 | 0.715686772101 | 0 |
| `de_ti_p` (cross-pol) | 0.182714043398 | 0.182714043398 | 6.9e-17 |
| `de_ri_s`, `de_ri_p` | — | — | 0 |
| `T_s` (amplitude) | -0.718451-0.446672j | -0.718451-0.446672j | 0 |
| **`T_p` (amplitude)** | **+0.203278+0.376021j** | **-0.203278-0.376021j** | **sign flipped** |

Every efficiency agrees to machine zero. Only the complex `T_p` moves, and it moves by exactly a
sign — because `|T_p|²` cannot see a mirror, but `T_p` can. (The table uses the names of
2026-08-10. Since 2026-08-12 the complex amplitudes are the lowercase `t_s` / `t_p`, and
`T_s` / `T_p` name the per-order efficiencies that `de_ti_s` / `de_ti_p` were.)

> **If you are checking a hand-filled tensor against anything, check an amplitude.** A validation
> built from R and T numbers is structurally incapable of catching this.

This is pinned by `test_off_diagonal_sign_shows_up_in_the_amplitude_only`.

## 4. The deeper problem: the two blocks disagreed

Everything above concerns *your* sign. This section concerns the code's.

`layer_modes_2d` builds **two** blocks and uses them for different things:

```
     Omega2_LR   ──eig──►   q²  and  W        the modal constants and the E-field modes
     Omega_R     ────────►  V = (Omega_R @ W) / q        the H-field
```

That is the standard `P·Q` structure of RCWA: the second-order equation is `d²E/dz² = P Q E`, the
code writes the product `P·Q` out in expanded form as `Omega2_LR`, and keeps the right-hand factor
separately as `Omega_R` so it can recover `H`. For that to describe **one** medium, the two must
genuinely factor:

```
     Omega2_LR  =  Omega_L @ Omega_R          with Omega_L built from geometry ALONE
                                              (Kx, Ky, epz_i — never the in-plane eps)
```

The requirement on `Omega_L` is the whole test. `Omega_L` is the part of the wave operator that
knows about propagation, not about the material; if it turns out to depend on `epx` or `epxy`, the
two blocks are not two factors of one thing.

Now look at how each block carries the off-diagonal pair:

```
     Omega2_LR = [[ ... - Cc @ epyx ,   ... - B  @ epxy ],
                  [ ... - D  @ epyx ,   ... - E  @ epxy ]]
                            ▲                     ▲
                          epyx                  epxy

     Omega_R   = [[ -Kx Ky + epxy   ,   Kx Kx - epy    ],
                  [  epx - Ky Ky    ,   Ky Kx - epyx   ]]
                            ▲                     ▲
                          epxy                  epyx      ◄── the other way round
```

Compute `Omega_L = Omega2_LR @ inv(Omega_R)` symbolically and the mismatch shows up immediately.
At normal incidence, with scalar entries:

```
                                 ┌                                             ┐
                             1   │ epx (epyx - epxy)      -epx epy + epxy²     │
     Omega_L =  ─────────────── ·│                                             │
                epx epy - epxy epyx  │ epx epy - epyx²      epy (epyx - epxy)  │
                                 └                                             ┘
```

For `Omega_L` to be geometry-only, its diagonal must vanish, which requires

```
     epx · (epyx - epxy) = 0        ⟺        epxy = epyx
```

**The factorization is valid if and only if the off-diagonal pair is symmetric.** For an
asymmetric pair, `Omega_L` acquires a dependence on the permittivity itself — which a factorization
of the wave equation cannot have.

### The repair, and that it is general

Exchanging `epxy` and `epyx` **inside `Omega_R` only**:

```python
Omega_R = _block(xp, [
    [-Kx @ Ky + epyx_conv, Kx @ Kx - epy_conv],     # epyx here
    [epx_conv - Ky @ Ky,   Ky @ Kx - epxy_conv],    # epxy here
])
```

turns `Omega_L` into pure geometry, symbolically, for an **arbitrary** tensor and **any** incidence
angle (not just the normal-incidence special case above):

```
                    ┌                                        ┐
     Omega_L  =     │  -Kx Ky epz_i        Kx² epz_i - 1     │
                    │  -Ky² epz_i + 1      Kx Ky epz_i       │
                    └                                        ┘
```

No `epx`, no `epy`, no `epxy`, no `epyx`. That is what a factor is supposed to look like.

## 5. What it costs, measured

The physical test is a gyrotropic slab — the simplest asymmetric tensor there is:

```
              ┌                     ┐
     eps  =   │  eo     i·g     0   │        eo = 2.25,  g = 0.45
              │ -i·g    eo      0   │        thickness 220 nm, lambda 600 nm
              │  0       0     eo   │        normal incidence, vacuum both sides
              └                     ┘
```

Its exact answer needs no solver. The eigenstates are **circular**, and the interface preserves
them, so each circular component is an independent scalar Fabry–Pérot:

```
     v₊ = (1, -i)/√2   sees   n₊ = √(eo + g)  ─┐
                                               ├─►  T = t₊ P₊ + t₋ P₋
     v₋ = (1, +i)/√2   sees   n₋ = √(eo - g)  ─┘
```

Comparing the full transmitted Jones matrix:

| | max abs error vs the exact answer |
|---|---|
| **as shipped** | **3.581e-2** |
| with the pair exchanged in `Omega_R` | **6.474e-16** |

and the error is first order in `g`, so it does not wash out at weak gyrotropy:

| `g` | 0.01 | 0.05 | 0.10 | 0.20 | 0.45 | 0.80 |
|---|---|---|---|---|---|---|
| error | 1.1e-3 | 5.6e-3 | 1.1e-2 | 2.0e-2 | 3.6e-2 | 1.2e-1 |

### Why this is hard to notice

The shipped answer keeps every property you would think to check:

```
     diagonal degenerate     J[0,0] - J[1,1]  =  1e-16     ✓ still gyrotropic in form
     off-diagonal antisym.   J[0,1] + J[1,0]  =  4e-16     ✓
     energy conserved        R + T            =  1 ± 1e-12 ✓
     rotation sense          matches +g, not -g            ✓
     magnitude                                             ✗  ← only this is wrong
```

## 5.1 Measured again against an outside oracle

Everything above rests on closed forms that exist only because the layer is uniform and the tensor
is simple, plus an identity internal to the code. Neither is an outside check, so the argument was
one formulation talking to itself.

Berreman's 4x4 method is the outside check. It eliminates `E_z` and `H_z` and integrates the
transverse fields as a first-order system, `d/dz (Ex, Ey, Hx, Hy) = i k0 M (Ex, Ey, Hx, Hy)`, so a
uniform layer is a matrix exponential and a stack is their product. No Fourier orders, no
eigenmodes, no s/p rotation, no transfer recursion — it shares no algebra with RCWA.

Validated first on tensors a shipped path can produce (isotropic, absorbing, diagonal, biaxial,
rotated-uniaxial, three-layer), normal and oblique: **max 7.8e-16**. Only then applied to the
asymmetric cases:

| configuration | as shipped | pair exchanged in `Omega_R` |
|---|---|---|
| gyrotropic slab, normal | 5.2e-2 | 3.6e-16 |
| gyrotropic slab, 15 / 30 / 45 / 60 / 75 deg | 7.8e-3 … 1.9e-1 | ≤ 1.1e-15 |
| general asymmetric, complex, 0 and 40 deg | 2.6e-2 … 8.4e-1 | ≤ 2.0e-15 |
| multilayer (gyro / isotropic / gyro, and a 4-layer) | 8.5e-3 … 1.8e-1 | ≤ 1.6e-15 |
| thickness 40 … 1500 nm at 28 deg | 1.5e-2 … 1.3e-1 | ≤ 1.6e-15 |

Every asymmetric configuration tried, at every angle, lands on the oracle once the pair is
exchanged. The complex Jones matrix matches too, to ~1e-15, once wavebit's per-channel s/p phase
(`diag(1, -i)` on output, `diag(1, -1)` on input, fitted on a symmetric case) is taken out.

Two corrections this pass forced:

* **Power efficiencies do discriminate here.** §3 shows the *sign* error hiding from R and T, and
  that remains true — but it is a statement about a mirrored optic axis, not about the ordering
  defect. For an asymmetric tensor the efficiencies are simply wrong, which makes the discriminator
  convention-free and much easier to check than the amplitude argument in §6.
* **Onsager reciprocity does not discriminate.** `T(-g) = T(g)^T` holds to 5.7e-16 for the shipped
  code as well as the exchanged one. A plausible-looking symmetry check passes on the wrong answer.

Pinned by `tests/rcwa/test_anisotropic_berreman.py`.

## 6. It is not a time-convention artefact

A reasonable objection: this repo runs `e^{-iwt}` with `Im(n) > 0` as loss
([`conventions.py`](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/_common/conventions.py)), and handedness conventions decide which
circular state is which. Could the reference above simply have the wrong handedness?

No, for two independent reasons.

**First, a handedness error can only relabel `g -> -g`.** So scan `g` over a continuum and ask
whether the shipped output is the exact gyrotropic answer for *any* `g`:

| | best-fitting `g` anywhere in [-1, 1] | residual there |
|---|---|---|
| as shipped | +0.4264 | 3.176e-2 — never fits |
| exchanged | +0.45000 | 6.474e-16 — exact |

The shipped output is not a correct gyrotropic answer under *any* relabeling. The exchanged one
lands precisely on the tensor it was handed.

**Second, the `Omega_L` argument in §4 contains no physics at all.** It is an algebraic identity
about the code's own two matrices. There is no time convention in it, no handedness, no field —
just the requirement that a product of two factors be a product of two factors.

### What the convention *does* still leave open

Which physical magnetization direction `[[eo, i·g], [-i·g, eo]]` denotes under `e^{-iwt}` is **not**
settled by any of this. What is verified is that the solver reproduces the tensor it is handed.
Mapping that tensor onto a real material's Verdet constant and field direction is a separate
question, and anyone building a Faraday isolator will have to answer it.

## 7. Nothing shipped was affected

When this was written, every producer in the library returned the off-diagonal value **twice**.
One function builds the scalar NVF tensor now, and every route reaches it:

```
     factorization.py::_nvf_assemble       return eps_xx, eps_yy, inv(ep_dir), -eps_xy, -eps_yx
     factorization.py::nvf_conv_matrices   (raster: calls _nvf_assemble; so does
                                            nvf_conv_matrices_integrated)
     parametric.py::nvf_from_coeffs        (shapes: delegates to _nvf_assemble)
     boundary.py                           (painted overlaps: funnels into nvf_from_coeffs)
                                                                         └────┬─────┘
                                                        one object unless the projector is
                                                        asymmetric, so the ordering cannot matter
```

and the Li path passes `None`, which `solve_2d` turns into zeros. The symmetry is not a
coincidence: it comes from the ½-symmetrization inside `_nvf_assemble`, which exists because the
un-symmetrized product is not Hermitian. (The shape path's module was `analytic.py` when this was
written; it became `parametric.py` on 2026-09-03, and its `nvf_from_coeffs` carried its own copy of
the closing until the inverse-rule floor made it delegate, see
[`nvf-metal-inverse-rule.md`](nvf-metal-inverse-rule.md).)

Two routes added since do put two different values in the pair, and both depend on the §4 repair.
`nvf_projector='circular'` (2026-09-02) hands `_nvf_assemble` a `C_yx = C_xy^H` — the
convolution matrix of `conj(p_xy)`, the conjugate *transpose* of `C_xy` — so `eps_yx` is built
separately. A tensor material closes through the Popov–Nevière route, whose
`eps_xy` and `eps_yx` are two blocks of the tensor's own.

Verified rather than argued — a real 2D crossed grating (Si pillar, 400 nm pitch, 300 nm tall,
lambda 900 nm, `theta = 12°`, `phi = 33°`, `psi = 21°`, `truncation_order = (6, 6)`, 169 retained orders),
solved with the shipped and exchanged operators:

| factorization | max Δ`de_ri` | max Δ`de_ti` | bit-identical |
|---|---|---|---|
| `li` | 0.000e+00 | 0.000e+00 | yes |
| `nvf` | 0.000e+00 | 0.000e+00 | yes |

Not "agrees to 1e-15" — identical. The full verification suite (1066 tests) is likewise unmoved.

**So: when this was written it was reachable only by hand, through `solve_conv`, with a
deliberately asymmetric tensor;** the two routes above reach it now, after the §4 repair. The
first person to reach it by hand will be someone trying gyrotropy — and if their
magnetisation lies in the plane rather than along the stack normal, they will be filling the four
z-row slots of §2.1 at the same time.

## 7.1 The slot *order* changed on 2026-08-18; the sign convention did not

Everything above is about which **value** goes in a slot. A separate defect concerned which
**position** — and it is worth stating here because §1's `solve_conv` call is the entry point
both are reached through.

The builders returned five slots as `(epx, epy, epxy, epyx, epz_i)`, grouping the in-plane 2x2
block and putting the inverted zz matrix last. `solve_conv` takes
`(epx, epy, epz_i, epxy, epyx)`, and its order is forced: `epz_i` is required on every solve
because it eliminates `E_z`, while the off-diagonal pair defaults to `None`, and Python will not
accept a required parameter after a defaulted one. So

```python
res = mee._solve_conv(*to_conv_mat_raster_pn_eps(tensor, truncation_order, truncation_order))
```

put `epz_i` in the `epxy` slot. (The recipe above uses the private door for the private
builder's corner-frame output; this history concerns the unchanged slot order.) Nothing
raised. On a lossless passive pillar it returned a full
set of plausible efficiencies summing to `R + T = 1.61`, against `1.0000000000` for the same
structure solved properly — only a generic result warning fired, and that warning has since been
removed ([`result-guards.md`](../../reference/common/result-guards.md)); `tests/rcwa/test_slot_order.py` asserts the energy
directly instead. The same idiom was correct for the three-slot isotropic builders throughout,
which is what made it a trap rather than a mistake.

It had already been found once, inside `factorization.py`: `pn_mu_conv_matrices` existed solely
to swap the two groups on the way to `layer_modes_2d`, its docstring describing the identical
failure for `mu_zz^-1`. A local wrapper could not reach `solve_conv`, so the fix was to make
it **one order everywhere** — every builder now returns `solve_conv`'s parameter order, and
`pn_mu_conv_matrices` rearranges nothing.

`tests/rcwa/test_slot_order.py` pins it for every builder, two ways: energy on a lossless passive
layer, and the structural identity that slot 2 is `inv(⟦eps⟧)` (the only slot held as an inverse,
so it is identifiable without knowing the answer — and it still discriminates on a lossy
structure, where the energy assertion says nothing). Reverting either producer to the old order
fails eight of its cases and leaves the three-slot builders passing, which is the discrimination
the test is for.

Nothing about the sign convention in §2 changed, and no shipped path moved: every consumer inside
the library unpacks by name, so the reorder is invisible to `solve`.

## 8. State, and what is still open

| | |
|---|---|
| Sign convention (`slot = -eps`) | derived, measured, pinned by a test |
| Slot *order* (builders vs `solve_conv`) | mismatched until 2026-08-18, silent and worth `R + T = 1.61`; **fixed** by making it one order everywhere (§7.1), pinned by `tests/rcwa/test_slot_order.py` |
| Asymmetric-tensor ordering | defect confirmed symbolically, numerically, and against an outside oracle; **fixed** |
| The repair (exchange the pair in `Omega_R`) | applied; lands on both oracles at ~1e-15 across every configuration tried |
| Effect on released behaviour | none — verified bit-identical on real gratings |
| Oblique incidence, stacks, complex asymmetric tensors | covered by `tests/rcwa/test_anisotropic_berreman.py` |
| Which magnetization `+i·g` means | **open** |
| A *patterned* anisotropic layer | **out of scope here**; the factorization it needed landed on 2026-08-14 ([`patterned-anisotropic-factorization.md`](patterned-anisotropic-factorization.md)) |
| The z-row slots (`epxz`, `epyz`, `epzx`, `epzy`) | same sign rule, chosen not forced (§2.1); judged against Berreman, where a wrong sign *does* move a power efficiency |

### What the oracle is, and is not

Two limits are worth stating plainly, because the evidence above is easy to over-read.

**Berreman's method is a standard; the fifty lines of it here are not.** The 4x4 formulation
(Berreman 1972) is the reference technique for stratified anisotropic media, and it was built for
exactly this case — the paper's own abstract names Faraday rotation and optical activity as what
the 6x6 starting point is meant to carry. Nothing about the method is improvised.

What is local is the *implementation*: ~50 lines in `tests/rcwa/test_anisotropic_berreman.py`, written
by the same hand as the thing it judges and validated against this solver's own known-good cases.
That is a statement about provenance, not about quality. On the oracle tiers in `tests/README.md`
it lands in **physics** rather than **an outside solver**, for the narrow reason that a
misconception shared by both implementations would survive — not because the reference is weak.

**RETICOLO closes that gap** — for the tensors in this document, which is narrower than it sounded
when this was written. Hugonin and Lalanne's code has carried a toolbox for arbitrarily anisotropic
multilayer stacks since V9; V10 takes a texture as a full complex 3x3 permittivity tensor, and
ships `exemple_V9_{0D,1D,2D,conique}_anisotrope.m` to demonstrate it. This repo trusts it as its
strongest oracle.

Since measured (2026-08-11): that trust holds for a diagonal tensor at any mounting and for **one**
off-diagonal pair at classical mounting, which covers every case in this document. Past that
RETICOLO's own `R + T` leaves 1 by up to 1.2e-2 on a provably lossless tensor. See
[`five-slot-limit.md`](five-slot-limit.md) §4 and `tests/references/README.md`. Nothing below
changes; the cases here all have a single pair and `delta0 = 0`. Run on the same stacks, with the same convention gate applied first — isotropic
and symmetric tensors, where it agrees with Berreman to 2.1e-15, no convention fitting needed:

| configuration | shipped, vs RETICOLO | pair exchanged | Berreman, vs RETICOLO |
|---|---|---|---|
| real antisymmetric, normal | 5.3e-2 | **8.9e-16** | 4.4e-16 |
| real antisymmetric, 35 deg | 5.5e-2 | **7.8e-16** | 1.0e-15 |
| gyrotropic, normal | 4.2e-2 | **7.8e-16** | 5.6e-16 |
| gyrotropic, 45 deg | 5.6e-2 | **1.0e-15** | 3.3e-16 |

So a third party's code, in a different language, by a different group, reaches the same answer as
the exchanged operator and disagrees with the shipped one by the same margin measured everywhere
else. It also corroborates the Berreman implementation independently, which retroactively
strengthens every number in §5.1.

**S4 works too, and the reason it appeared not to was mine.** It was set aside once on the grounds
that its numbers disagreed on symmetric tensors — which they did, but the cause was neither the
angle convention nor anything physical. S4 indexes the permittivity tensor with **x and y
exchanged** relative to this repo. Conjugating by the swap matrix, `SWAP @ eps @ SWAP`, brings it
onto Berreman at **5.6e-16** for symmetric tensors and **6.7e-16 / 2.2e-16** for a real
antisymmetric one, at normal and oblique incidence alike. The angle argument is `(polar,
azimuth)` as written, verified against an analytic Airy slab at 0, 20 and 45 degrees.

Two limits, both worth knowing before reaching for it. This build **rejects complex tensors**, so
gyrotropy proper is out of reach; a real antisymmetric tensor is an active medium rather than a
physical one, but it discriminates the ordering just as well. A separate survey of the
available solvers records that S4 gives **silently wrong results when polarisation decomposition and anisotropy are
used together** — so read total power flux per incident polarisation, as the measurements above do,
and not order-resolved s/p amplitudes.

What S4 adds that RETICOLO cannot is the **patterned** case: a tensor material assigns to a
patterned region and runs (checked at 49 orders). That is the oracle the factorization work in
`anisotropic-materials.md` §3 will need, and it exists today.

**Every case is a uniform film.** Not one of them has structure in the plane. Concretely, that
means all of this runs at `truncation_order=(0, 0)` — a single Fourier order — with convolution matrices
hand-filled as scaled identities. So the machinery that turns a *pattern* into those matrices is
never touched: not the `dfs2d_*` builders, not Li's rules, not NVF, not the analytic or boundary
builders, not
truncation.

What is therefore validated is the layer mode operator, the s/p rotation, the boundary match and
the layer joining — which is exactly where the defect lives, so the finding stands. What is not
validated, at all, is any anisotropic behaviour of a *patterned* layer.

That gap will not close by adding cases here: a patterned anisotropic layer needs a factorization
that does not exist yet (`anisotropic-materials.md` §3), so there is nothing to test, and no
uniform-layer oracle could reach it if there were. (It has existed since 2026-08-14, the
Popov–Nevière factorization of [`patterned-anisotropic-factorization.md`](patterned-anisotropic-factorization.md),
judged against fmmax there; tensor Li joined it on 2026-09-24,
[`tensor-li-factorization.md`](tensor-li-factorization.md).)

It is, however, reachable when the time comes — though not by the route this document first claimed.
RETICOLO draws the line between patterned and uniform, and its own V10 manual states it on page 1:

> Diagonal anisotropy (`eps_xx != eps_yy != eps_zz`) can be incorporated in **structured grating
> layers** and gratings with **uniform layers** having **arbitrary anisotropy** (`eps_xy != 0` …)
> can also be handled

So a patterned layer gets a diagonal tensor and nothing off-diagonal; a full 3x3 is accepted only
where the layer is uniform. Two further constraints come with it, neither obvious from the examples
and both worth knowing before building a case around them:

* **The substrate and superstrate cannot be anisotropic** (manual, page 1). The goldens here have
  vacuum on both sides, so it does not bite — but it rules out, say, a magneto-optic substrate.
* The absorption-loss integral in `res3` is **not valid for uniform layers with non-diagonal
  anisotropy** (manual, page 13). Efficiencies are unaffected; only that loss route is.

The patterned off-diagonal oracle is S4, which takes a tensor material on a patterned region
directly — real tensors only, which is enough. For a patterned tensor that is also *complex*,
two candidates carry full complex `eps` **and** `mu` tensors: `inkstone` and `fmmax`. Neither
is installed here.

So the oracles exist for every stage of that work; what does not exist is the factorization. The
hard part will be Popov–Nevière, not finding something to check it against — which was the right
conclusion reached for the wrong reason the first time.

Tests: `tests/rcwa/test_uniform_anisotropic.py`

- `test_gyrotropic_layer_has_the_right_shape` — what does hold today (passes)
- `test_gyrotropic_layer_matches_faraday` — the exact answer (was xfail until the repair landed)

`tests/rcwa/test_anisotropic_berreman.py`

- `test_oracle_agrees_where_the_solver_is_known_good` — the convention gate (passes)
- `test_asymmetric_tensor_matches_the_oracle` — oblique, stacks, complex tensors

`tests/references/test_reticolo_anisotropic_golden.py`

- `test_symmetric_tensors_match_reticolo` — the convention gate against the outside solver
- `test_asymmetric_tensors_match_reticolo` — the same four cases, from a frozen golden

## References

Every number in §5.1 and §8 rests on one of these two. Both were checked against the publishers
rather than quoted from memory.

- D. W. Berreman, *Optics in Stratified and Anisotropic Media: 4×4-Matrix Formulation*,
  J. Opt. Soc. Am. **62**(4), 502 (1972).
  [doi:10.1364/JOSA.62.000502](https://doi.org/10.1364/JOSA.62.000502) — the formulation
  `tests/rcwa/test_anisotropic_berreman.py` implements. It starts from a 6×6 statement of Maxwell's
  equations chosen so that Faraday rotation and optical activity are expressible, which is why it
  is the natural reference for the gyrotropic case rather than a workaround for it.
- J. P. Hugonin and P. Lalanne, *RETICOLO software for grating analysis*, Institut d'Optique,
  Orsay (2005); [arXiv:2101.00901](https://arxiv.org/abs/2101.00901) — an RCWA implementation for
  1D, conical and 2D crossed gratings. From V9 it carries a toolbox for arbitrarily anisotropic
  multilayer stacks, which is what made the outside check in §8 possible. Already this repo's
  strongest oracle for isotropic gratings (`tests/references/golden_reticolo.npz`).

Also cited in `anisotropic-materials.md`, for the factorization a patterned anisotropic layer would
need: Li (2003) for the first-order formulation, Popov and Nevière (2001) for the rearrangement,
Schuster et al. (2007) for the NVF this repo implements.
