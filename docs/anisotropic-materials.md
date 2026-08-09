# Anisotropic and magnetic materials — what the solver already takes, and what each gap costs

Written: 2026-08-09. Scoping only; nothing here is implemented.

Targets:
- `fieldwave/_core/solve_2d.py::layer_modes_2d` — the five permittivity slots the mode operator reads
- `fieldwave/_core/factorization.py` — what `epx` / `epy` / `epz_i` actually are (not what they look like)
- `fieldwave/_core/conventions.py::boundary_kz` — where `mu` would have to land besides the layer
- `fieldwave/on_numpy/emsolver/rcwa.py::solve_for_conv` — the injection point that bypasses `ucell`

---

## TL;DR

| case | status | what it costs |
|---|---|---|
| uniform (unpatterned) anisotropic layer, in-plane tensor | **works today** | a thin wrapper |
| patterned anisotropic layer | not supported | a new factorization; **solver unchanged** |
| `mu != 1` | not supported | layer operator **and** the half-space boundary |
| `eps_xz` / `eps_yz` (z-coupling) | not supported | **a different solver** — 4ff first-order, not 2ff second-order |

Nothing on this list is impossible. Exactly one of them changes the *shape* of the code rather
than its content, and it is the last row.

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

with `epxy` and `epyx` as **independent** arguments — the algebra never assumes `exy == eyx`, so a
gyrotropic (`exy = -eyx`) in-plane block fits the same slots. The `ezz` entry enters only as its
inverse because `E_z` is eliminated algebraically as `E_z = ezz⁻¹ D_z`, which is also why field
reconstruction reads nothing but `epz_conv_i` (`fields.py::214`).

The slots reach the user through `solve_for_conv`, on every backend:

```python
res = mee.solve_for_conv(wavelength,
                         epx_conv_all, epy_conv_all, epz_conv_i_all,
                         epxy_conv_all, epyx_conv_all)
```

The 1D and 1D-conical paths take three, not five (`engine.py::solve_1d`, `::solve_1d_conical`), so
in-plane off-diagonal terms are a 2D-path capability only.

## 2. The trap: `epx` / `epy` / `epz_i` are not tensor components

They are **one scalar permittivity** put through three different factorization rules:

```
                 ucell   (one refractive index per pixel)
                                 │
                          eps = ucell ** 2
                                 │
          ┌──────────────────────┼──────────────────────┐
   dfs2d(eps, 0, 1)      dfs2d(eps, 1, 0)      dfs2d(eps, 1, 1)
  inverse rule along x  inverse rule along y      direct rule
          │                      │                      │
       epx_conv               epy_conv               epz_conv ──inv──> epz_conv_i
```

(`factorization.py::249-251`; `conti_x` / `conti_y` flag a direction as continuous, and a
discontinuous one takes the inverse rule.) NVF is the same story: `ep_dir = ⟦eps⟧` and
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
The resulting `eps̃` matrices are fully populated. That rearrangement is the actual work item, and
an anisotropic extension of NVF sits on top of it.

## 4. Uniform anisotropic layers work today

An unpatterned layer has no interface, so the factorization problem does not arise: every
convolution matrix is a scaled identity. The code states this itself —

> Uniform layers cannot expose it at all — their convolution matrix is `eps * I`.
> (`factorization.py::_circulant_index`)

So a birefringent film, a uniaxial substrate or an LC layer can be solved on `develop` right now by
filling the five slots directly:

```python
import numpy as np

ff_xy = (2 * fto_x + 1) * (2 * fto_y + 1)
I = np.eye(ff_xy, dtype=np.complex128)

# eps = [[exx, exy, 0], [eyx, eyy, 0], [0, 0, ezz]] for a single uniform layer
epx   = np.stack([exx * I])
epy   = np.stack([eyy * I])
epxy  = np.stack([-exy * I])      # slot convention is (-exy, -eyx); see _nvf_assemble
epyx  = np.stack([-eyx * I])
epz_i = np.stack([np.linalg.inv(ezz * I)])

res = mee.solve_for_conv(wavelength, epx, epy, epz_i, epxy, epyx)
```

Two things to watch:

- The **slot sign convention** is `(epx, epy, epxy, epyx, epz_i) = (exx, eyy, -exy, -eyx, ⟦eps⟧⁻¹)`.
  It is set by `_nvf_assemble`, not by the mode operator, and it is easy to get backwards.
- The **2D path must be selected**. `rcwa.py::_assign_grating_type` picks the path from the `ucell` shape, so
  a pure multilayer stack with `exy != 0` has to be routed to the 2D solve — the 1D and conical
  paths have no off-diagonal slots to fill.

## 5. Magnetic materials: `mu` has no slot anywhere

There is no permeability in `develop` at all. The old `meent` code at least carried a dead
`P_Q_kz(..., mu_conv)` in the scattering utilities; the rewrite dropped even that.

Where it would have to go:

```python
B = Kx @ epz_conv_i @ Kx - I        #  <- this I is mu_conv
D = Ky @ epz_conv_i @ Ky - I        #  <- and this one
```

Structurally this is benign — `mu` enters as the `E`/`H` dual of `eps`, and a magnetic layer needs
the same five-slot treatment on the `mu` side. What makes it more than a one-line edit is that the
**half-spaces** are isotropic and non-magnetic by construction too:

```python
kz_top = boundary_kz(xp, n_top ** 2, k_par_sq)          # takes eps alone
big_F  = _block(xp, [[I, O], [O, -1j * Kz_bot / eps_bot]])   # p-channel admittance
```

`boundary_kz` has one material argument, and the `/ eps_bot` in `big_F` is exactly the admittance
where `mu` belongs. So a magnetic substrate or superstrate means touching the boundary conventions
— and `conventions.py` is the module that exists because a sign seam there once turned absorbing
media into gain media. Change it with a golden in hand.

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
   d psi / dz = M psi,        psi = [E_x, E_y, H_x, H_y]      (4ff x 4ff)
```

Without z-coupling, `M` is block-antidiagonal:

```
   no z-coupling                        with z-coupling

   M = [  0    P  ]                     M = [ M11    P  ]
       [  Q    0  ]                         [  Q    M22 ]

   dEt/dz depends on H only              dEt/dz depends on E too
   dHt/dz depends on E only

   -> d²Et/dz² = P·Q·Et    (2ff)        -> a first-derivative term survives;
      no first-derivative term              no clean second-order reduction
```

`P·Q` written out in expanded form *is* `Omega2_LR`; the right factor is kept separately as
`Omega_R` so that `V = (Omega_R @ W) / q` recovers `H`. And because the equation is purely second
order, `exp(+q z)` and `exp(-q z)` solve it with the **same eigenvector** — the `±` pair is free:

```
   2ff eigenproblem                     4ff eigenproblem
   ────────────────                     ────────────────
   eig(Omega2_LR) -> (W, q²)            eig(M) -> 4ff eigenpairs
   down: W·exp(-k0 q d)                 forward and backward are genuinely
   up:   W·exp(+k0 q d)   same W        different mode sets; sort by decay
                                        and energy flux, not by sign
```

That `±` pairing is baked into every consumer: the `big_W (I + Y)` / `big_V (I - Y)` construction
(`solve_2d.py::183`), the single `[epz_conv_i, W, V, q, d, big_A_i, big_B]` stored per layer, the
four-quadrant split in `sp_modes_2d`, the branch rule in `conventions.py::mode_q`, and the field
reconstruction in `fields.py`. All of it assumes the 2ff shape.

Cost is not only structural: `eig` is `O(N³)`, so 2ff → 4ff is roughly **8x** on the layer solve.

### Note on gyrotropy

In-plane antisymmetry (`exy = -eyx`, i.e. Faraday-type magneto-optics) does **not** break the
second-order form — what breaks it is the z-coupling, not the in-plane asymmetry. The slots are
already independent, so gyrotropy is a validation problem (match a known Faraday rotation angle),
not a formulation problem.

## 7. Where to start

```
  Is the layer patterned in xy?
    │
    ├── no ──> uniform: fill the five slots with scaled identities        WORKS TODAY (§4)
    │
    └── yes ─> Does the optic axis lie in the xy plane?
                 │
                 ├── yes ──> new factorization (Popov–Nevière + NVF)      solver unchanged (§3)
                 │
                 └── no ───> z-coupling: 4ff first-order solver           solver replaced (§6)
```

The first row is a wrapper. The second is real work but reuses the mode operator unchanged. The
third is a second solver living beside the first.

One caveat that applies to rows two and three: **the existing goldens cannot arbitrate any of
this.** Every NVF golden is 1D or square-pixel, so an anisotropic factorization would need its own
reference — an analytic uniaxial slab for the uniform case, and a published crossed-anisotropic
grating for the patterned one.

## References

- L. Li, *Fourier modal method for crossed anisotropic gratings with arbitrary permittivity and
  permeability tensors*, J. Opt. A: Pure Appl. Opt. **5**(4), 345 (2003).
  [doi:10.1088/1464-4258/5/4/307](https://doi.org/10.1088/1464-4258/5/4/307) — the 4ff formulation
  that covers §5 and §6 together, in one paper.
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
