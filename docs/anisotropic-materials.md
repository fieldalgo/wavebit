# Anisotropic and magnetic materials — what the solver already takes, and what each gap costs

**Current input API (2026-09-22):** assign `Material(epsilon=[exx, eyy, ezz])`
or a 3×3 tensor directly to a layer background or shape and call `solve()`.
Tensor shapes now default to [continuous geometry integration](tensor-shape-integration.md),
with analytic primitive indicators and a sampled normal extension.
Material-axis rotation, tensor tables and tensor rasters are described in the
[material guide](guide/materials.md#anisotropic-materials). The discussion below
is the historical derivation and development record, not the current input guide.


Written: 2026-08-09, and it was scoping only then. Since 2026-08-11 the z-coupling row is
implemented, and since 2026-08-13 it is *reachable* — `solve_conv` takes nine slots and the
solver picks the formulation (§8, no longer conditional). See
[`first-order-formulation.md`](first-order-formulation.md) for what that path does and how far it is
checked. This document keeps the scoping question: what each *remaining* gap costs.

Targets:
- `src/wavebit/_core/solve_2d.py::layer_modes_2d` — the five permittivity slots the mode operator reads
- `src/wavebit/_core/factorization.py` — what `epx` / `epy` / `epz_i` actually are (not what they look like)
- `src/wavebit/_core/conventions.py::boundary_kz` — where `mu` would have to land besides the layer
- `src/wavebit/_core/engine.py::solve_conv` — the injection point that bypasses `structure`, and where the formulation is chosen
- `src/wavebit/_core/solve_2d_first_order.py::solve_2d_first_order` — the nine-slot twin of the 2D solve

---

## TL;DR

| case | status | what it costs |
|---|---|---|
| uniform (unpatterned) anisotropic layer, in-plane tensor | **works today** | a thin wrapper |
| `eps_xz` / `eps_yz` (z-coupling), uniform | **works today** | done — a different solver, [`first-order-formulation.md`](first-order-formulation.md), dispatched to automatically at ~8x |
| patterned anisotropic layer, in-plane tensor | **works today** | done — Popov–Nevière, [`patterned-anisotropic-factorization.md`](patterned-anisotropic-factorization.md); both solvers unchanged, as scoped |
| patterned **and** z-coupled | supported | full Li/NVF closure and first-order for epsilon and mu; [verification](full-tensor-patterns.md) |
| `mu != 1`, layers and half-spaces | **works today** | done — [`magnetic-materials.md`](magnetic-materials.md); the layers are the E/H dual of the five slots, the half-spaces one pair of admittances |

Nothing on this list is impossible. Exactly one of them changed the *shape* of the code rather
than its content — the z-coupling row, which is why it needed a second solver and got one. The
rest are content: new matrices for the same machinery.

The patterned row landed as §3 predicted — a factorization, both solvers untouched. The row
below it is the combination, and it is held back by evidence rather than by difficulty: fmmax was
the only solver that reached the patterned case and it has no z-row arguments, so there is
nothing to check the answer against.

## 1. What the mode operator already accepts

`layer_modes_2d` takes five convolution matrices, not three:

```python
def layer_modes_2d(xp, Kx, Ky, I, epx_conv, epy_conv, epz_conv_i, epxy_conv, epyx_conv, C):
    B  = Kx @ epz_conv_i @ Kx - I
    D  = Ky @ epz_conv_i @ Ky - I
    Cc = Kx @ epz_conv_i @ Ky
    E  = Ky @ epz_conv_i @ Kx

    Omega2_LR = _block(xp, [
        [Ky @ Ky + B @ epx_conv - Cc @ epyx_conv,  Cc @ epy_conv - Kx @ Ky - B @ epxy_conv],
        [E @ epx_conv - Ky @ Kx - D @ epyx_conv,   Kx @ Kx + D @ epy_conv - E @ epxy_conv],
    ])
```

Those five slots map onto a permittivity tensor exactly like this:

```
                    E_x        E_y        E_z
               ┌──────────┬──────────┬──────────┐
        D_x    │   epx    │   epxy   │    ·     │
               ├──────────┼──────────┼──────────┤
        D_y    │   epyx   │   epy    │    ·     │
               ├──────────┼──────────┼──────────┤
        D_z    │    ·     │    ·     │  epz_i⁻¹ │
               └──────────┴──────────┴──────────┘

   named cell = a slot the mode operator already reads
   ·          = assumed zero, no slot exists
```

So the solved tensor class is

```
        [ exx  exy   0  ]
  eps = [ eyx  eyy   0  ]
        [  0    0   ezz ]
```

with `epxy` and `epyx` as separate arguments — but **only a symmetric pair is actually solved
correctly**; see §6.1. The `ezz` entry enters only as its
inverse because `E_z` is eliminated algebraically as `E_z = ezz⁻¹ D_z`, which is also why field
reconstruction reads nothing but `epz_conv_i` (`fields.py::214`).

The slots reach the user through `solve_conv`, on every backend. Patterned hand-built
slots use Fourier coordinates measured from the cell centre, like `eps_conv_matrices()`:

```python
res = mee.solve_conv(epx_conv_all, epy_conv_all, epz_conv_i_all,
                     epxy_conv_all, epyx_conv_all)
```

The 1D and 1D-conical paths take three, not five (`engine.py::_solve_1d`, `::_solve_1d_conical`), so
in-plane off-diagonal terms are a 2D-path capability only.

## 2. The trap: `epx` / `epy` / `epz_i` are not tensor components

They are **one scalar permittivity** put through three different factorization rules:

```
             structure   (one relative permittivity per pixel)
                                 │
                        eps = structure          (no squaring since 2026-08-20)
                                 │
          ┌──────────────────────┼──────────────────────┐
  ..._piecewise_constant  ..._piecewise_constant  ..._piecewise_constant
      (eps, 0, 1)            (eps, 1, 0)            (eps, 1, 1)
  inverse rule along x  inverse rule along y      direct rule
          │                      │                      │
       epx_conv               epy_conv               epz_conv ──inv──> epz_conv_i
```

(`inverse_rule` names the cell direction whose discontinuity the inverse rule closes, or `None`
for the direct rule both ways.) NVF is the same story: `ep_dir = ⟦eps⟧` and
`ep_inv = ⟦1/eps⟧⁻¹` both come from that one scalar, and the tensor `_nvf_assemble` returns is the
*effective* tensor of the interface projector — not a material tensor.

**Therefore "put `exx` in the `epx` slot" is not a valid substitution.** The inverse rule is not a
property of the slot, it is a consequence of which product is being factorized.

## 3. Why the substitution fails, precisely

Li's rule follows from continuity at an interface:

```
      material A  │  material B
                  │
     E_x  ────────┤ ┌──────      normal to the interface -> jumps
                  │ │
     D_x  ─────────────────      normal component of D  -> continuous
                  │
              x ──┴──>           x-normal interface
```

A continuous quantity written as a product of two discontinuous ones is what forces the inverse
rule on `D_x = eps · E_x`. With a tensor, the continuous quantity is a **sum of three products**:

```
  D_x = exx·E_x  +  exy·E_y  +  exz·E_z
```

and there is no per-term rule to apply. The standard fix (Popov–Nevière) is to rearrange the
constitutive relation so that the continuous quantities — tangential `E`, normal `D` — sit on one
side, invert that block **pointwise in real space**, and then apply the direct rule throughout.
The resulting `eps̃` matrices are fully populated.

**Done, 2026-08-14** — `factorization.py::pn_eps_conv_matrices`, with
`to_conv_mat_raster_pn_eps` for a stack. Full write-up in
[`patterned-anisotropic-factorization.md`](patterned-anisotropic-factorization.md); two things
from it are worth carrying back here, because both contradict what this section assumed:

* "invert that block pointwise in real space" is doing more work than it looks. The projector
  algebra has to finish *before* anything is transformed. Building `⟦T⟧`, `⟦N⟧` and the transfer
  `⟦M⟧` separately and multiplying the matrices is the same algebra pointwise and is wrong — the
  normal field is a unit field extended into the bulk, hence discontinuous itself, so that
  product is Laurent applied across a concurrent jump. It conserves energy, it is rotation
  covariant, it sits plausibly near fmmax, and it is **wrong by 0.5 on a uniform slab**.
* the full tensor uses all three components in the closing step, as detailed in
  [full tensor patterns](full-tensor-patterns.md).

## 4. Uniform anisotropic layers work today

An unpatterned layer has no interface, so the factorization problem does not arise: every
convolution matrix is a scaled identity. The code states this itself —

> Uniform layers cannot expose it at all — their convolution matrix is `eps * I`.
> (`factorization.py::_circulant_index`)

So a birefringent film, a uniaxial substrate or an LC layer can be solved on `develop` right now by
filling the five slots directly. This is checked, not asserted:
`tests/test_uniform_anisotropic.py` drives the call below against an analytic Airy
slab per eigen-polarisation and a Jones waveplate, and matches to ~1e-16.

```python
import numpy as np

n_orders = (2 * half_width_u + 1) * (2 * half_width_v + 1)
I = np.eye(n_orders, dtype=np.complex128)

# eps = [[exx, exy, 0], [eyx, eyy, 0], [0, 0, ezz]] for a single uniform layer
epx   = np.stack([exx * I])
epy   = np.stack([eyy * I])
epxy  = np.stack([-exy * I])      # slot convention is (-exy, -eyx); see _nvf_assemble
epyx  = np.stack([-eyx * I])
epz_i = np.stack([np.linalg.inv(ezz * I)])

res = mee.solve_conv(epx, epy, epz_i, epxy, epyx)
```

A full tensor adds four more, under the **same** sign rule — one rule for every off-diagonal
slot, rather than two rules split down the middle of one call:

```python
epxz, epyz = np.stack([-exz * I]), np.stack([-eyz * I])
epzx, epzy = np.stack([-ezx * I]), np.stack([-ezy * I])

res = mee.solve_conv(epx, epy, epz_i, epxy, epyx,
                     epxz, epyz, epzx, epzy)   # -> formulation_assigned == 'first_order'
```

Two things to watch:

- The **slot sign convention** is `(epx, epy, epz_i, epxy, epyx) = (exx, eyy, ⟦eps⟧⁻¹, -exy, -eyx)`,
  and the z row extends it unchanged: `(epxz, epyz, epzx, epzy) = (-exz, -eyz, -ezx, -ezy)`.
  It is set by `_nvf_assemble`, not by the mode operator, and getting it backwards is close to
  undetectable: `+eps_xy` mirrors the optic axis about `x`, and **every power efficiency stays
  bit-identical** — co-polarised, cross-polarised and total alike. Only the complex amplitude
  moves, `T_p` flipping sign while `T_s` does not. Measured, not argued; see
  `test_uniform_anisotropic.py::test_off_diagonal_sign_shows_up_in_the_amplitude_only`. If you
  are checking a hand-filled tensor against anything, check an amplitude.
- The **2D path must be selected**. `engine.py::_assign_grating_type` picks the path from the `structure` shape, so
  a pure multilayer stack with `exy != 0` has to be routed to the 2D solve — the 1D and conical
  paths have no off-diagonal slots to fill. A z row does this for itself (§8); the in-plane pair
  does not, which is the asymmetry to remember.

## 5. Magnetic materials: `mu` on the layers and the half-spaces

**Layers: done, 2026-08-14.** `layer_modes_2d` takes an optional `mu` — the same five slots as
the permittivity, in the same convention — and `solve_2d` / `smm_2d` carry it per layer.
[`magnetic-materials.md`](magnetic-materials.md) has the derivation and what checks it.

This section used to point at the two `- I` terms in

```python
B = Kx @ epz_conv_i @ Kx - I        #  <- this I is mu_conv
D = Ky @ epz_conv_i @ Ky - I        #  <- and this one
```

which is right about the location and wrong about the content in three ways, all of which the
implementation had to correct:

* the pairing is **crossed** — `B` takes `mu_yy` and `D` takes `mu_xx`. Following the sketch
  would have been exactly right for an isotropic `mu` and silently transposed a uniaxial one.
* two more slots exist. `Cc` and `E` take the off-diagonal `mu`, and `mu_zz` enters `Omega_R`
  between the `K` factors, mirroring where `epz_conv_i` sits in `Omega_L`.
* `Omega2_LR`'s hand-expanded form **cannot be reused**. Its four bare `K @ K` terms look like
  geometry that survived the multiplication, and are really what two other terms collapse to
  when the `K` factors commute past a `muz_conv_i` of one. Reusing it conserved energy for scalar
  `mu` and broke it outright for gyromagnetic `mu` (`R + T = 1.035`).

**Half-spaces: done too**, and this is the part the original warning was about. They used to be
isotropic and non-magnetic by construction:

```python
kz_top = boundary_kz(xp, n_top ** 2, k_par_sq)          # takes eps alone
big_F  = _block(xp, [[I, O], [O, -1j * Kz_bot / eps_bot]])   # p-channel admittance
```

and the `/ eps_bot` in `big_F` is exactly the admittance where `mu` belongs. The instruction
here used to be "change it with a golden in hand", and that is what happened:
`golden_fmmax_mu_ambient.npz` was frozen first, while there was still nowhere to put a `mu`,
and it carries a **closed form** on top — the magnetic layer golden could not serve, because
every case in it sits between vacuum, which is exactly what let it validate the layer operator
on its own.

`boundary_kz` needed no change at all, as it turns out: it already took `n^2` under the name
`eps`, and `n_top` / `n_bot` stayed the refractive index. They still are — since 2026-08-20 they
are *derived* from the `top` / `bot` materials (`n = sqrt(eps*mu)`) and read-only, which is why
the flip to permittivity at the front door left this half of the code untouched
([`conventions.md`](conventions.md)). What moved is one pair of
admittances, `kz/mu` on s and `kz/eps` on p — see
[`magnetic-materials.md`](magnetic-materials.md) §5.

## 6. z-coupling: the one gap that changes the solver

### What it is

The tensor entries carrying a `z` index: `exz`, `ezx`, `eyz`, `ezy`. `exz` says "`E_z` contributes to
`D_x`"; `ezx` says "`E_x` contributes to `D_z`". These are the `·` cells in the figure in §1.

They appear whenever the **optic axis tilts out of the xy plane**:

```
     axis ∥ z                    axis ∥ x                  axis tilted by a
        z                           z                          z
        │ ↑                         │                          │    ↗
        │ │                         │                          │  ↗
        └─┴──── x                   └───→──── x                └────────── x

  diag(eo, eo, ee)            diag(ee, eo, eo)         R_y(a)·diag(eo,eo,ee)·R_y(a)ᵀ
  no z-coupling               no z-coupling            exz = ezx = (ee-eo)·sin a·cos a
                              (in-plane rotation                    ≠ 0
                               only makes exy)
```

In practice this is not an exotic case: an LC director with pretilt is tilted almost by definition,
as is a birefringent crystal cut off-axis, or a magneto-optic film magnetized obliquely.

### Why it breaks the second-order form

Eliminating `E_z` and `H_z` leaves a first-order system in `z` for the transverse fields:

```
   d psi / dz = M psi,        psi = [E_x, E_y, H_x, H_y]      (4N x 4N for N orders)
```

Without z-coupling, `M` is block-antidiagonal:

```
   no z-coupling                            with z-coupling

   M = [  0    P  ]                         M = [ M11    P  ]
       [  Q    0  ]                             [  Q    M22 ]

   dEt/dz depends on H only                  dEt/dz depends on E too
   dHt/dz depends on E only

   -> d²Et/dz² = P·Q·Et    (second-order)   -> a first-derivative term survives;
      no first-derivative term                  no clean second-order reduction
```

`P·Q` written out in expanded form *is* `Omega2_LR`; the right factor is kept separately as
`Omega_R` so that `V = (Omega_R @ W) / q` recovers `H`. And because the equation is purely second
order, `exp(+q z)` and `exp(-q z)` solve it with the **same eigenvector** — the `±` pair is free:

```
   second-order eigenproblem            first-order eigenproblem
   ─────────────────────────            ────────────────────────
   eig(Omega2_LR) -> (W, q²)            eig(M) -> 4N eigenpairs
   down: W·exp(-k0 q d)                 forward and backward are genuinely
   up:   W·exp(+k0 q d)   same W        different mode sets; sort by decay
                                        and energy flux, not by sign
```

That `±` pairing is baked into every consumer: the `big_W (I + Y)` / `big_V (I - Y)` construction
(`solve_2d.py::183`), the single `[epz_conv_i, W, V, q, d, big_A_i, big_B]` stored per layer, the
four-quadrant split in `sp_modes_2d`, the branch rule in `conventions.py::mode_q`, and the field
reconstruction in `fields.py`. All of it assumes the second-order shape.

Cost is not only structural: `eig` is `O(N³)`, so second-order → first-order is roughly **8x** on
the layer solve.

**Part of this now exists.** `src/wavebit/_core/solve_2d_first_order.py` carries the first-order
operator `M`, the forward/backward split, and a stack solve, pinned by
`tests/test_solve_2d_first_order.py` against both the second-order path (where that path is valid,
eigenvalues agree at `lam = i q` to 1e-15) and the Berreman oracle (entry for entry, to 1e-16, on
tensors the second-order path cannot express at all), and by
`tests/references/test_first_order_external_goldens.py` against RETICOLO and fmmax (24 cases, worst
2.4e-15, no convention fitting). Three of those are Voigt tensors generated for this — **the first
time an outside solver has judged the geometry this module exists for.**

That external reach has a hard edge, measured rather than assumed. RETICOLO is exact for a
diagonal tensor at any mounting and for a *single* off-diagonal pair at classical mounting;
beyond that it returns `R + T` up to 1.2e-2 from one on a provably lossless tensor, converged
and truncation-independent. A tilted magnetisation — xy and yz populated together, which is what
a real magnetised film has — is past that edge, so it still rests on our own Berreman. The
measurements and the reasoning are in `tests/references/README.md`.

Nothing dispatches to it. It is, however, stable for any thickness: layers are joined by an
enhanced-transmittance recursion (`layer_algorithm='ETM'`, the same string and the same
technique as the second-order path) or by scattering matrices (`'SMM'`), and both hold ~1e-13
relative accuracy out to a modal exponent of 2500, where they reproduce the closed-form
single-interface Fresnel reflectance to 4e-16. An unstabilised transfer is kept as `'TMM'` for
cross-checking and refuses past the point where it is accurate — which on a 220 nm layer means from
25 Fourier orders upward, since diffraction orders above the light line are strongly evanescent
whatever the thickness. That is the historical reason RCWA needed the enhanced form, not a
preference between two recursions. The block layout is checked at 9 and 25 orders; what is still
missing is order *coupling*, which needs the patterned factorization of §3. Field reconstruction
landed 2026-08-14 (`docs/first-order-formulation.md` §4b) — its `E_z` is the one formula the
second-order path could not lend, because a populated z row leaves two more terms in the
elimination.

Two findings from building it, both invisible until the case existed. The `±` pairing above does
not merely become unnecessary in the first-order form — it becomes *false*: at normal incidence the
backward set still matches the negated forward set to 1e-16, but for a tilted magnetisation at 35°
they part by 9e-2. z-coupling and oblique incidence have to combine before the assumption breaks,
which is why nothing noticed. And direction has to be decided per mode against its own norm: scaling
the flux threshold by the largest flux in the set works on every propagating case and then, once
every mode is evanescent — which is every diffraction order above the light line — promotes 1e-17 of
noise to real energy flow.

## 6.1 Gyrotropy: the two blocks disagree on the order of the pair

*Summarised here; the full derivation, the figures and every measurement are in
[`off-diagonal-slot-convention.md`](off-diagonal-slot-convention.md).*

In-plane antisymmetry (`exy = -eyx`, i.e. Faraday-type magneto-optics) does **not** break the
second-order form — what breaks that is the z-coupling, not the in-plane asymmetry. Gyrotropy
therefore needs no new formulation: the derivation below produces a geometry-only `Omega_L` for an
arbitrary in-plane tensor, asymmetric included. What it runs into is an **ordering error in the
implementation** of that form — one block carries the off-diagonal pair the other way round.

`layer_modes_2d` builds two blocks, and they carry the off-diagonal pair in **opposite orders**:

```
   Omega2_LR   ...  - Cc @ epyx   ... - B @ epxy    (eigenvalues q², and W)
   Omega_R     ...  + epxy        ... - epyx        (the H-field, V = Omega_R W / q)
```

A factorization `Omega2_LR = Omega_L @ Omega_R` is only meaningful if `Omega_L` is built from the
geometry alone. Symbolically it is not — with the shipped ordering `Omega_L` comes out depending
on `epx`, `epy`, `epxy` and `epyx`. Exchanging the pair inside `Omega_R` **only** reduces it to

```
   Omega_L = [[-Kx Ky epz_i,      Kx² epz_i - 1],
              [-Ky² epz_i + 1,    Kx Ky epz_i  ]]
```

for an arbitrary tensor and any incidence angle. Before the repair the two blocks carried the pair
in the same order, so an asymmetric tensor got its eigenvalues from one medium and its magnetic
field from that medium's transpose.

Measured on a gyrotropic slab (`eps = [[eo, i·g], [-i·g, eo]]`, g = 0.45) against the
circular-eigenstate answer — each circular component is an independent Fabry–Pérot with
`n± = sqrt(eo ± g)`:

| | max abs error in the transmitted Jones matrix |
|---|---|
| before the repair | 3.6e-2 — first order in `g`, so it did not wash out |
| with the pair exchanged in `Omega_R` | 6.5e-16 |

The result still *looks* right: the Jones matrix stays degenerate on the diagonal and
antisymmetric off it, energy conserves to 1e-12, and the rotation sense is correct. Only the
magnitude is wrong.

This is not an artefact of the reference's time convention. A handedness or `e^{-iwt}` sign error
in the analytic side could only relabel `g -> -g`, so scanning `g` over a continuum settles it:
the shipped output's best fit *anywhere* is `g = +0.4264` at 3.18e-2 — it is not the exact
gyrotropic answer for any `g` — while the exchanged version lands on `g = +0.45000` at 6.47e-16.
The `Omega_L` argument above is convention-free outright: it is an algebraic identity of the
code, with no physics in it.

Still open, and *not* settled by any of this: which magnetization direction `[[eo, i·g], [-i·g,
eo]]` denotes under this repo's `e^{-iwt}`, `Im(n) > 0` convention. What is verified is that the
solver reproduces the tensor it is handed — not what that tensor means physically.

**Nothing shipped could reach this when it was found.** `_nvf_assemble` returned `-eps_xy`
*twice*, and so did both routes beside it — `boundary.py`, and `nvf_from_coeffs`, then in
`analytic.py` with its own copy of the closing (now `parametric.py`, delegating to
`_nvf_assemble`) — so every tensor the library built was symmetric, and for a symmetric pair the
exchange is a no-op — bit-identical. The defect was reachable only by hand through
`solve_conv`, which is exactly what a gyrotropy experiment would do first. Two routes added since
do build an asymmetric pair, `nvf_projector='circular'` and tensor materials, and they depend
on the fix below.

**Fixed.** `Omega_R` now takes `epyx` where it took `epxy` and vice versa; every symmetric tensor
is bit-identical either way, so nothing the library builds moved. Pinned by
`test_uniform_anisotropic.py`, `test_anisotropic_berreman.py` (a Berreman 4x4 oracle) and
`tests/references/test_reticolo_anisotropic_golden.py` (RETICOLO, frozen golden).

## 7. Choosing the construction

Uniform material assignments use exact blocks. Patterned material assignments
use Cartesian Li or linear NVF, depending on the geometry and chosen option.
A z coupling in epsilon or mu selects the full first-order solver automatically.
The [material guide](guide/materials.md) describes the input forms; [full tensor
patterns](full-tensor-patterns.md) gives the supported geometry, validation and
convergence limits.

## 8. How the solver chooses

Everything above says what is missing. This says how the pieces that *do* exist are selected,
because that decision is not the user's to make. It was written as a design before the dispatch
existed; it is kept in that order — principle, cost, override, requirement — with what shipped
on 2026-08-13 noted against each part, because the reasoning is what a future axis (`mu`) will
need and the code alone does not carry it.

**The principle is already in the codebase.** `engine.py::_assign_grating_type` reads the `structure`
shape, `tetm` and `phi` and picks the cheapest formulation that can express the problem — its
docstring says so in as many words: *"This decides the efficient formulation for given case."*
1D TE/TM splits into scalars, 1D conical does not, 2D couples everything. Anisotropy adds axes to
that same decision rather than a new kind of decision.

| axis | read from | branches |
|---|---|---|
| grating dimensionality *(exists)* | `structure` shape, `tetm`, `phi` | 1D TE/TM, 1D conical, 2D |
| **z row of the tensor** *(exists)* | whether the four z-row slots were given | second-order, first-order |
| **permeability** | `mu != 1` | existing operator, magnetic operator |
| **patterned or uniform** | layer has in-plane structure | factorization, or scaled identities |

One thing about the z-row row came out differently from the sketch. It reads whether the slots
were **given**, not whether they are nonzero. The branch decides control flow, so it has to be
answerable on a JAX tracer, where there is nothing to read; and "passed all-zero matrices" is a
claim about the caller rather than about the tensor. Passing four zero matrices therefore costs
the 8x, and the cheap path is one omitted argument away.

The z-row axis reaches back into the dimensionality one. The 1D operators have no z-row slots,
so a z row — the permittivity's `ep{xz,yz,zx,zy}` or the permeability's `mu[5:]` — moves even
a one-row raster onto the 2D layout the first-order path is written on, and
`formulation='first_order'` does the same without one. That layout carries both polarizations, so
`tetm` beside either is refused by name rather than rerouted: the path it would be rerouted to reads
an incident polarization `tetm` declared absent. Until 2026-09-27 it was rerouted: a `tetm` given at
construction surfaced as a `TypeError` from deep in the solve, and one assigned afterwards
answered for s light whichever of the two it named. The message names the pair that describes
the same light, `polarization=(1, 0)` / `(0, 1)` at `phi = 0`, and
`tests/test_tetm_off_the_pure_path.py` checks that it does: beside an all-zero z row, and under
a forced `'first_order'`, the pair reproduces the pure `tetm` solve.

### Why it has to be automatic

The cost spread is large enough that a wrong default is a real tax, and small enough in the
common case that nobody would notice paying it. Measured on RETICOLO at 441 orders, against an
isotropic layer of the same size:

| layer | cost |
|---|---|
| diagonal anisotropic | 2.8x |
| in-plane tensor, empty z row (Faraday) | 3.3x |
| **z-coupled (Voigt)** | **9.5x** |

The 9.5x is the first-order jump — the eigenproblem doubles and `eig` is cubic. A user with a
Faraday stack has no reason to know that a populated `eps_yz` is what would cost them eight times
more, and no way to check whether their tensor has one without reading this document. That is what
an `auto` default is for.

### And why the override still matters

```python
class Formulation(str, Enum):
    AUTO         = 'auto'           # read the slots and decide           (default)
    SECOND_ORDER = 'second_order'   # force second-order; refuse z-coupling
    FIRST_ORDER  = 'first_order'    # force first-order; always possible, ~8x
```

That is `wavebit._core.enums.Formulation`, and it is reached as `solver.formulation` — an
attribute rather than a constructor keyword, because the JAX solver passes its pytree children
positionally and a new keyword shifts them.

Forcing `'first_order'` on a problem that does not need it is how the first-order solver gets
**validated**: a z-decoupled case run both ways must agree, and that is an internal cross-check no
external oracle provides. It is `tests/test_formulation_dispatch.py`, and the case that carries it
is a *patterned* grating at conical incidence — the two formulations then eig completely different
matrices and agree to 4e-14, which a uniform layer's scaled identities could not have shown. It is the same reason `layer_algorithm` is selectable — and the same reason
`test_anisotropic_berreman.py` drives SMM as well as ETM, since "the correction reaches the other
path for free" is exactly the sort of claim that turns out to be false.

Forcing `'second_order'` on a z-coupled tensor must **raise**, not quietly drop the z row. That
failure mode is already measured: dropping it moves the efficiencies by 2e-2 to 6e-2 while `R + T`
stays at 1 to machine precision, so every check a caller is likely to run still passes. The guard
used to live at the tensor adapter in `test_anisotropic_berreman.py`; it now lives in the solver,
which is where this paragraph said it belonged once the solver could be handed a tensor directly.

### One requirement on the dispatch itself

**Which path ran has to be visible in the result.** `grating_type` is already exposed for
exactly this reason. A silent 8x is indistinguishable from a slow machine, and a benchmark that
cannot say which formulation it measured is not a benchmark. The attribute is
`solver.formulation_assigned`, `'second_order'` or `'first_order'`, set by every 2D solve and `None`
before the first one.

## References

- L. Li, *Fourier modal method for crossed anisotropic gratings with arbitrary permittivity and
  permeability tensors*, J. Opt. A: Pure Appl. Opt. **5**(4), 345 (2003).
  [doi:10.1088/1464-4258/5/4/307](https://doi.org/10.1088/1464-4258/5/4/307) — the first-order
  formulation that covers §5 and §6 together, in one paper.
- E. Popov and M. Nevière, *Maxwell equations in Fourier space: fast-converging formulation for
  diffraction by arbitrary shaped, periodic, anisotropic media*, J. Opt. Soc. Am. A **18**(11),
  2886 (2001). [doi:10.1364/JOSAA.18.002886](https://doi.org/10.1364/JOSAA.18.002886) — the
  rearrangement described in §3.
- T. Schuster, J. Ruoff, N. Kerwien, S. Rafler and W. Osten, *Normal vector method for convergence
  improvement using the RCWA for crossed gratings*, J. Opt. Soc. Am. A **24**(9), 2880 (2007).
  [doi:10.1364/JOSAA.24.002880](https://doi.org/10.1364/JOSAA.24.002880) — the isotropic NVF this
  repo implements.
- P. Götz, T. Schuster, K. Frenner, S. Rafler and W. Osten, *Normal vector method for the RCWA with
  automated vector field generation*, Opt. Express **16**(22), 17295 (2008).
  [doi:10.1364/OE.16.017295](https://doi.org/10.1364/OE.16.017295) — the follow-up on generating
  the field automatically; a distinct paper from the one above.
