# The off-diagonal permittivity slots — what they hold, and why an asymmetric tensor comes out wrong

Written: 2026-08-10. Measured on CPU, NumPy backend, float64.

Targets:
- `fieldwave/_core/solve_2d.py::layer_modes_2d` — the two blocks, and the ordering mismatch between them
- `fieldwave/_core/factorization.py::_nvf_assemble` — where the sign convention is set
- `fieldwave/on_numpy/emsolver/rcwa.py::solve_for_conv` — the only way to reach the slots by hand
- `tests_verification/test_uniform_anisotropic.py` — every number below

Companion to [`anisotropic-materials.md`](anisotropic-materials.md), which answers the scoping
question ("can I give the solver an anisotropic material, and what does each gap cost?"). This one
answers a narrower and more practical question: **if I fill the five slots myself, what exactly am
I supposed to put in them, and where does that go wrong?**

---

## TL;DR

1. The slot named `epxy` holds **`-eps_xy`**, not `eps_xy`. Same for `epyx`. Nothing in the public
   signature says so.
2. Getting that sign backwards is **invisible to every power efficiency** — R, T, co-pol, cross-pol
   and total are all bit-identical. Only a complex amplitude changes.
3. More seriously: the two blocks inside `layer_modes_2d` carry the off-diagonal pair in **opposite
   orders**, so an *asymmetric* tensor (`eps_xy != eps_yx`, i.e. gyrotropy) gets its eigenvalues
   from one medium and its magnetic field from that medium's transpose. Measured error 3.6e-2,
   first order in the gyrotropy strength.
4. **No shipped code path can reach point 3.** Every tensor the library builds is symmetric, and
   for a symmetric pair the ordering is a no-op — verified bit-identical on real gratings. This is
   a latent defect that blocks a future capability, not a wrong number in anything released.

---

## 1. What the five slots are

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

and they arrive through the public entry point:

```python
res = mee.solve_for_conv(wavelength,
                         epx_conv_all, epy_conv_all, epz_conv_i_all,
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
`(eps_xx, eps_yy, -eps_xy, -eps_xy, inv(⟦eps⟧))` — the negation is applied there, on the way in,
and every consumer downstream has been written against it ever since.

So the correct hand-filled call for a uniform layer is:

```python
I = np.eye(ff_xy, dtype=np.complex128)

epx   = np.stack([eps_xx * I])
epy   = np.stack([eps_yy * I])
epxy  = np.stack([-eps_xy * I])        # <-- negate
epyx  = np.stack([-eps_yx * I])        # <-- negate
epz_i = np.stack([np.linalg.inv(eps_zz * I)])
```

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
sign — because `|T_p|²` cannot see a mirror, but `T_p` can.

> **If you are checking a hand-filled tensor against anything, check an amplitude.** A validation
> built from R and T numbers is structurally incapable of catching this.

This is pinned by `test_off_diagonal_sign_shows_up_in_the_amplitude_only`.

## 4. The deeper problem: the two blocks disagree

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

## 6. It is not a time-convention artefact

A reasonable objection: this repo runs `e^{-iwt}` with `Im(n) > 0` as loss
([`conventions.py`](../fieldwave/_core/conventions.py)), and handedness conventions decide which
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

## 7. Nothing shipped is affected

Every producer in the library returns the off-diagonal value **twice**:

```
     factorization.py:590   return eps_xx, eps_yy, -eps_xy, -eps_xy, inv(ep_dir)
     boundary.py:268        return eps_xx, eps_yy, -eps_xy, -eps_xy, inv(ep_dir)
     analytic.py:563        return eps_xx, eps_yy, -eps_xy, -eps_xy, inv(ep_dir)
                                                   └────┬────┘
                                              the same value, so the
                                              ordering cannot matter
```

and the Li path passes `None`, which `solve_2d` turns into zeros. The symmetry is not a
coincidence: it comes from the ½-symmetrization inside `_nvf_assemble`, which exists because the
un-symmetrized product is not Hermitian.

Verified rather than argued — a real 2D crossed grating (Si pillar, 400 nm pitch, 300 nm tall,
lambda 900 nm, `theta = 12°`, `phi = 33°`, `psi = 21°`, `fto = (6, 6)`, 169 retained orders),
solved with the shipped and exchanged operators:

| factorization | max Δ`de_ri` | max Δ`de_ti` | bit-identical |
|---|---|---|---|
| `li` | 0.000e+00 | 0.000e+00 | yes |
| `nvf` | 0.000e+00 | 0.000e+00 | yes |

Not "agrees to 1e-15" — identical. The full verification suite (1066 tests) is likewise unmoved.

**So: this is reachable only by hand, through `solve_for_conv`, with a deliberately asymmetric
tensor.** The first person to do that will be someone trying gyrotropy.

## 8. State, and what is still open

| | |
|---|---|
| Sign convention (`slot = -eps`) | derived, measured, pinned by a test |
| Asymmetric-tensor ordering | defect confirmed symbolically and numerically; **not fixed** |
| Effect on released behaviour | none — verified bit-identical |
| Which magnetization `+i·g` means | **open** |
| Asymmetric tensor at oblique incidence, vs an independent oracle | **not covered** |

The last row matters if the repair is applied. The evidence for it is a normal-incidence
gyrotropic slab plus an algebraic identity that holds at general `K`; the identity is internal
consistency, not an outside check. A published crossed magneto-optic grating, or a second solver,
would be the honest confirmation.

Tests: `tests_verification/test_uniform_anisotropic.py`

- `test_gyrotropic_layer_has_the_right_shape` — what does hold today (passes)
- `test_gyrotropic_layer_matches_faraday` — the exact answer (xfail, strict; flips to XPASS the
  moment the ordering is repaired)
