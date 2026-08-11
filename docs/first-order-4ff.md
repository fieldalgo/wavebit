# The first-order (4ff) path — what it is judged against, when no oracle will do it

Written: 2026-08-11. Measured on an Apple M4, NumPy 2.5.1 against Accelerate, float64.

Targets:
- `fieldwave/_core/solve_2d_4ff.py` — the whole path
- `tests/test_solve_2d_4ff.py`, `tests/references/test_4ff_external_goldens.py` — every number here
- `tests/test_anisotropic_berreman.py` — the Berreman oracle it is judged against

Companions. [`anisotropic-materials.md`](anisotropic-materials.md) scopes the whole anisotropy
question and says what each gap costs; [`five-slot-limit.md`](five-slot-limit.md) explains why the
2ff path has five permittivity slots and what other solvers do at that edge;
[`connecting-algorithms.md`](connecting-algorithms.md) covers `connecting_algo`, a general
question this path happens to be the best test bed for.

**This one answers a narrower question: how do you validate a solver when no other solver can
check it?** Three of the four decisions below exist only because the usual answer — compare
against someone else's code — was unavailable. Where the path currently stands, and what it
cannot yet do, lives beside the code in `solve_2d_4ff.py`; that goes stale and this should not.

---

## TL;DR — four decisions, and what forced each

| decision | forced by |
|---|---|
| Judge the tilted-magnetisation geometry by a **theorem** (rotation covariance about `z`), not by an oracle | all four surveyed solvers are out, each for a measured reason (§2) |
| Decide mode direction **per mode against its own norm**, not against the set | the set-relative rule reads 1e-17 of dust as energy flow once every mode is evanescent — which is every order above the light line |
| Accept `exx = epx`, `eyy = epy`, `ezz_i = epz_i` as the isotropic-grating mapping | it is the same continuity argument that puts the inverse rule in `epx`, not a fit (§3) |
| Treat a patterned *anisotropic* layer as a **factorization** problem, not a solver problem | this module takes convolution matrices as input; fed real ones it already works (§3) |

## 1. Why it exists

The 2ff path reduces a layer to `d²E_t/dz² = Omega2 E_t`. That reduction exists only when the
permittivity has an empty z row and column: eliminating `E_z` otherwise leaves a first-derivative
term and there is no second-order operator to write. Physically that case is the **Voigt
geometry** — a magnetisation lying *in* the plane, which is what an ordinary magnetised film has
unless perpendicular anisotropy or an applied field holds it out.

The first-order form keeps the whole transverse state instead:

```
    d/dz (Ex, Ey, Hx, Hy) = i k0 M (Ex, Ey, Hx, Hy)          M is 4ff x 4ff
```

## 2. What it is checked against

Four oracles, in decreasing order of independence:

| oracle | reaches | agreement |
|---|---|---|
| RETICOLO (external) | uniform tensors, **one** off-diagonal pair at `delta0 = 0` | 1e-15, 12 cases |
| fmmax (external) | uniform tensors, no z row | 1e-15, 12 cases |
| the 2ff path (ours) | z-decoupled tensors; isotropic gratings | `lam = i q` to 1e-15; gratings to 3e-14 |
| Berreman 4×4 (ours) | uniform tensors including z-coupling | entry-for-entry to 1e-16 |

**No outside solver reaches a tilted magnetisation.** All four surveyed are out, each for a
measured reason — the table is in [`../tests/references/README.md`](../tests/references/README.md).
So that geometry is judged by a *theorem* instead: turning the tensor and the incidence azimuth
together about the stack normal must leave the efficiencies unchanged, which holds to 3e-15 over
eight tensors, three angles and four rotations. That test is also the only thing in the repo
exercising `ky != 0`; five mutations that are the identity at `ky = 0` die there and nowhere else.

## 3. What it solves, measured

| | evidence |
|---|---|
| uniform z-coupled tensors | Berreman, entry-for-entry, plus end-to-end R/T |
| any thickness | closed-form Airy slab to ~1e-13 at modal exponent 113; single-interface Fresnel to 4e-16 at 2500 |
| any order count | a uniform layer cannot diffract, so 9 and 25 orders must reproduce 1 order — they do, to 1e-14 |
| **isotropic gratings** | fed the 2ff path's own convolution matrices, reproduces it to **3e-14** at up to 49 orders, normal and conical |
| lossless ⇒ `R + T = 1` | every Hermitian case, to 1e-10 |

The grating row is the one that exercises **order coupling**: those convolution matrices are full,
not scaled identities, so orders genuinely mix. It works because an isotropic grating needs no new
factorization — `exx = epx`, `eyy = epy`, `ezz_i = epz_i`, everything else zero — and that mapping
is principled rather than fitted: `epx` carries the inverse rule along `x` because `D_x = eps E_x`
is a continuous quantity written as a product of discontinuous ones, and `exx` sits in exactly
that product in the first-order form too.

## 4. The one thing it cannot do

**A patterned anisotropic layer.** Not a solver limitation — this module takes convolution
matrices as *input* and needs no change. The matrices do not exist: with a tensor, the continuous
quantity is a sum of three products,

```
    D_x = exx·E_x + exy·E_y + exz·E_z
```

and Li's rule has no per-term form to apply. The standard fix is the Popov–Nevière rearrangement
([`anisotropic-materials.md`](anisotropic-materials.md) §3), and an anisotropic NVF sits on top of
it. Until that exists, the geometry this path was built for cannot be patterned — which is worth
stating plainly, because "4ff solves z-coupling" and "4ff solves magneto-optic gratings" are not
the same claim.

## 5. Two things that were not obvious

**The `±` mode pairing becomes false, not merely unnecessary.** In 2ff, `exp(+q z)` and
`exp(-q z)` solve the same second-order equation with the *same* eigenvector, so the pair is free
and only a root has to be chosen. A first-order system has `4ff` eigenvalues that are not `±`
pairs. Measured: at normal incidence the backward set still matches the negated forward set to
1e-16, and for a tilted magnetisation at 35° they part by 9e-2. z-coupling and oblique incidence
have to combine before the assumption breaks, which is why nothing noticed.

**Direction has to be decided per mode, against its own norm.** Scaling the "carries no flux"
threshold by the largest flux in the set — the obvious thing — works on every propagating case and
then, once every mode is evanescent, promotes 1e-17 of numerical dust to real energy flow. Every
diffraction order above the light line is that case, so the set-relative version would have broken
on the first grating.
