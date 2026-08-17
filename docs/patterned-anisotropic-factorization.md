# Patterned anisotropic layers — the Popov–Nevière factorization

Landed 2026-08-14. [`anisotropic-materials.md`](anisotropic-materials.md) §3 scoped this as "the
actual work item"; this is what it turned into.

Target: `wavebit/_core/factorization.py` — `_pn_fields`, `pn_conv_matrices`,
`to_conv_mat_raster_pn`.

**Scope.** The **in-plane** tensor — the five slots `epx, epy, epz_i, epxy, epyx`. A patterned
layer that also fills the z row (`eps_xz`, `eps_yz`) is *not* implemented, and §6 below says why
that is a deliberate stop rather than an unfinished edge.

---

## 1. Why the scalar factorization does not extend

`_nvf_assemble`'s entire input is `⟦eps⟧` and `⟦1/eps⟧`. A tensor has no `1/eps` to take — not
in the sense the rule needs. Li's rule applies to a product of **one** discontinuous factor with
**one** continuous one. With a tensor the continuous quantity is a sum of three such products:

```
  D_x = exx·E_x  +  exy·E_y  +  exz·E_z
```

and there is no per-term rule. Applying the scalar recipe component-by-component is not a
degraded approximation of the right answer; it is an answer to a different question.

## 2. The rearrangement

At an interface with unit normal `n`, tangential `E` and normal `D` are continuous; normal `E`
and tangential `D` jump. Write `N = n nᵀ` and `T = I − N` for the pointwise projectors, and
eliminate the discontinuous `n·E` from `D = eps E`. With

```
  a = n·eps·n        p = eps n        q = epsᵀ n
```

the elimination gives `(n·E) = [ (n·D) − q·e_t ] / a` with `e_t = T E`, and substituting back:

```
  D = (eps − p qᵀ / a) e_t  +  (p / a) (n·D)
  E = (I   − n qᵀ / a) e_t  +  (n / a) (n·D)
```

Every right-hand side is now **one discontinuous coefficient times one continuous quantity** —
the case Laurent's rule converges on. Collecting the continuous vector `c = e_t + n (n·D)`:

```
  A = (eps − p qᵀ / a) T + (p / a) nᵀ        D = A c
  B = (I   − n qᵀ / a) T + (n / a) nᵀ        E = B c

  eps̃ = ⟦A⟧ ⟦B⟧⁻¹
```

One matrix inverse, no matrix products between transformed fields. That last part is the whole
design, and §3 is about why.

## 3. The arrangement that looks equivalent and is not

There is a tidier-looking route. Define the pointwise transfer `M = (T eps + N)(T + N eps)⁻¹`
between the continuous vector and the discontinuous one, note that `A = N + T M` and
`B = T + N M`, build `⟦T⟧`, `⟦N⟧` and `⟦M⟧` as three convolution matrices, and multiply. The
pointwise algebra is identical — checked to 1e-14 — and it is wrong.

The normal field is a **unit** field extended into the bulk (`normal_vector_field` — that is what
makes `C_xx + C_yy = I`), so it is discontinuous *itself*. `⟦T⟧ @ ⟦M⟧` is therefore Laurent
applied to two factors that jump in the same place: the concurrent jump Li's rule exists to
forbid. Doing the projector algebra **pointwise** and transforming once avoids it.

What makes this worth a section is how the wrong version behaved:

| check | the wrong arrangement | the shipped one |
|---|---|---|
| lossless tensor, `R + T = 1` | holds | holds |
| rotation covariance about z | holds | holds |
| distance to fmmax | plausible | plausible |
| non-hermiticity of `eps̃` | 4.4e-2, **plateaus** with fto | truncation only |
| **uniform layer** | **wrong by 0.5** | exact to 1e-15 |

Three of those five pass on a formulation that is wrong by 50% on a slab of glass. The one that
caught it is the cheapest test in the file and needs no oracle at all: a layer with no interface
has nothing to factorize, so `eps̃` must come back as the tensor it was given. For `eps = e I`
the shipped form satisfies `⟦A⟧ = e ⟦B⟧` *identically*, for any normal field whatsoever —
exact, not close. `tests/test_pn_factorization.py::test_a_uniform_layer_comes_back_as_the_tensor_it_was_given`.

## 4. Where the normal field vanishes

`normal_vector_field` returns `(0, 0)` on the skeleton of its distance transform — the ridge
where the gradient dies. The rearrangement divides by `a = n·eps·n`, which is zero there, and the
first working version produced NaN across the entire matrix from those few pixels.

The fix is not a patch. With `n = 0` the numerators `p`, `q` and `n nᵀ` all vanish and `T = I`,
so substituting `a = 1` makes the formulas return `A = eps` and `B = I` on their own — the direct
rule, which is exactly what a zero projector does in `_nvf_assemble`.

The guard is on **the field vanishing, not on `a` vanishing**, and the distinction is load-bearing.
A zero normal is geometry: expected, harmless, with an exact limit. An `a` that vanishes under a
*nonzero* normal is a null direction of the tensor itself — a real singularity of the method — and
substituting there would replace a diagnosable failure with a plausible number. It comes through
as non-finite instead.

## 5. What checks it

Nothing in the repo could arbitrate a patterned tensor before this. Of the four reference solvers
surveyed for this project, RETICOLO takes an anisotropic layer only as diagonal, S4 only as real,
and inkstone accepts a full 3×3 and returns results bit-identical to the z-row-zeroed tensor — it
drops what it is handed. **fmmax is the only one that both accepts a
patterned complex off-diagonal and demonstrably uses it**, gated before any of this was written:
a patterned `eps_xy` moves its eigenvalues by 8.9e-2, and patterning the gyrotropy differs from a
uniform one by 8.2e-2.

It arbitrates **by convergence, not by identity**. Three formulations are in play — this one,
fmmax's `FFT` (the plain direct rule) and fmmax's `JONES_DIRECT` (a vector formulation of a
different construction) — and they agree only in the limit. So the golden freezes *both* fmmax
answers and the tolerance is the spread between them. No claim can be sharper than the distance
between two of the oracle's own results.

At `fto=14`, gyrotropic pillar in vacuum:

| case | fmmax FFT vs JONES | this solver, to the nearer |
|---|---|---|
| `n=1.15, g=0.10i` | 4.8e-6 | **3.0e-7** |
| `n=1.30, g=0.20i` | 3.0e-5 | **1.4e-7** |
| `n=1.60, g=0.30i` | 1.7e-4 | **2.0e-5** |

Inside the band every time, and on the `JONES_DIRECT` side of it — which is where a vector
formulation belongs, and is asserted separately. A factorization that had the vector machinery in
the code and not in the answer would sit nearer `FFT`, and no energy check could see that.

`tests/references/test_fmmax_patterned_golden.py`, golden by `gen_goldens_fmmax_patterned.py`
(needs the `fwref` env).

### The transpose

**This project stores a layer as `A[y, x]`; fmmax's array axis 0 runs along its first lattice
vector, here X.** The same physical layer is `A.T` to fmmax — a relabelling of which axis means
x, not a mirror, so the tensor *components* are untouched.

Recorded because it cost a false diagnosis. Without it, both this project's factorizations — the
new one *and* the shipped isotropic NVF that predates it — disagreed with fmmax with the s/p split
**reversed**, which reads exactly like an off-diagonal sign error. What settled it was running the
shipped path as a control: a long-validated path failing the same way is not a bug in the new one.
Transposing moved fmmax onto both, to within its own formulation spread, in all three contrasts.

The pillar in the cases is deliberately **not square**, for the same reason: a square one is
invariant under the axis swap and would have passed while wrong.

### What holds without any oracle

- **`R + T = 1`** — the tensor is Hermitian at every pixel, so the layer is lossless however it is
  patterned. Residual 1.7e-5 / 1.7e-6 / 5.8e-7 at fto 4 / 8 / 12; the *trend* is asserted too,
  because an early arrangement conserved energy to ~1e-2 at every order and a fixed tolerance
  would have called that a pass.
- **Rotation covariance about z** at normal incidence, to 1e-12. At oblique incidence the raster
  rotation has to be matched against the `phi` convention, and the shipped NVF path shows the same
  residual there — so that combination measures the harness, not this code.
- **Agreement with the scalar NVF** where they overlap. Not by converging in fto — measured, that
  gap sits near 1.3e-5 and does not fall, which is fine: two vector formulations differ at any
  finite truncation, and 1.3e-5 is an order tighter than fmmax's own 1.8e-4 on the same structure.
  What is forced is the *contrast* limit, and there the gap falls 17x faster than the reflectance
  it lives on.

## 6. What is not implemented

**Patterned z-coupling.** The derivation in `_pn_fields` is written for a full 3×3 and stays valid
for one. What stops at the in-plane block is the closing step: with `nz = 0` the projectors are
block diagonal, so `A_zz = eps_zz` and `B_zz = 1` exactly and the z slot is the plain direct rule
inverted — but only while the z row is empty. A patterned layer that fills it needs the 3×3 solved
together.

It is left unwritten rather than written blind, and the reason is §5: **fmmax has no z-row
arguments at all**, and it was the only solver that reached the patterned case to begin with. There
is no oracle. Given that this session's central lesson was a formulation that passed energy,
covariance and an external comparison while being wrong by 50%, shipping the harder case with
strictly less evidence than that one had is not a trade worth taking.

The uniform z-coupled case is unaffected and works — it goes to the 4ff solver,
[`first-order-4ff.md`](first-order-4ff.md).

## 7. The z slot reads the raster as a staircase

`epz_i` is built with `fourier2d_piecewise_constant`, the exact closed form for a staircase — not
with `fourier2d_pointwise`, which the in-plane block `A` and `B` are stuck with. It was the other
way round until 2026-08-17, and the switch is recorded here because the argument that put the DFT
there was reading a bug, and because nothing cheap can arbitrate the replacement.

**The old argument.** `eps_zz` alone is a staircase, so the closed form is exact for it and was
always the obvious choice. It was rejected on consistency: the closed form reads the raster as a
continuum staircase and the DFT as point samples, `A` and `B` cannot have the closed form (they mix
eps with the sampled normal field), and mixing two readings of one layer appeared to cost a symmetry
the true answer has — rotation covariance about z fell from 1.9e-14 to 5.2e-8.

**Why it no longer holds.** That gap was the grid origin, not the description. The sampled transform
was placing a pixel value at its left edge while the closed form placed it at the cell centre;
`da4762a` added `_half_pixel_phase` and took the closed-form arm to 1.5e-14. Both arms hold the
quarter turn now.

**And the consistency it appealed to was never the operative one.** With `nz = 0` the projectors are
block diagonal, so `A_zz = eps_zz` and `B_zz = 1` and the z slot is never multiplied against `A` or
`B` — it decouples, and has no reason to match how they read the grid. What it does have to match is
the package: a permittivity raster names the material *occupying* each pixel, so it is a staircase
wherever wavebit reads one. `nvf_conv_matrices` mixes these same two transforms the same way, closed
form for `ep_dir` / `ep_inv` and sampled for the projector, for this same reason. The PN z slot was
the one place that did not.

### What could not decide it

Rotation covariance sits at ~1e-14 on either arm, so it separates nothing. Energy conservation never
could: the tensor is Hermitian at every pixel, so `R + T = 1` holds however the layer is factorized —
that is precisely what makes it the check §5 values, and precisely why it cannot rank two arms that
are both correct. No external solver can rank them either: what separates the two transforms is the
`sinc` factor, which is a coarse-grid term, so —
exactly as in `da4762a` — the signal vanishes where an oracle would be sharp. **And an ensemble
median cannot**, which is the trap this measurement nearly fell into: over 24 incidences at fto 6 and
10 the median error ratio is 0.98, apparently favouring the DFT. That median is the shallow-angle
regime, where the slot barely contributes at all.

### Against the transform's own exact limit

The model is `da4762a`'s: compare with the exact coefficients rather than with a rival formulation.
`fourier2d_integrated` supplies them, integrating over the rectangle's real edges. On a rectangle
whose edges are grid lines the raster **is** the layer, so the closed form has nothing to
approximate and any residue in the other arm is the `sinc` alone. Max relative coefficient error,
fto=6:

| N | closed form | DFT | DFT order |
|---|---|---|---|
| 16 | 1.8e-16 | 3.0e-2 | |
| 32 | 1.8e-16 | 3.5e-3 | 3.08 |
| 64 | 1.8e-16 | 7.7e-4 | 2.19 |
| 128 | 1.8e-16 | 1.9e-4 | 2.04 |
| 256 | 1.8e-16 | 4.6e-5 | 2.01 |
| 512 | 1.8e-16 | 1.2e-5 | 2.00 |

Exact at every N against `1/N**2` — the midpoint rule's own order, the same one the half-pixel phase
bought. Move the edges off the grid and pixelation contributes a shared `O(1/N)` term to both arms:
the closed form still leads at coarse N (2.5e-2 against 4.3e-2 at N=16) and the two merge from N≈64.

### At the solve, it reaches the answer through the angle

`epz_i` enters only through `Kx epz_i Kx` and its partners, so its weight grows with incidence. The
measurement is a raster sweep at **fixed fto against a fine-grid reference**, on a pillar whose edges
are grid lines at every N — the layer is then literally the same function throughout and only the
transform's grid moves. The reference is N=768, where the two arms agree to ≤5e-7 — some 60x below
the smallest error being compared — so it favours neither; doubling it to N=1536 moves the ratios in
the third digit.

L2 error over all orders, fto=6, N=64, gyrotropic pillar, `n_bot=1.45`:

| theta | closed form | DFT | DFT / closed |
|---|---|---|---|
| 30 | 3.83e-5 | 3.58e-5 | 0.94 |
| 39 | 3.77e-5 | 3.87e-5 | 1.03 |
| 45 | 3.58e-5 | 4.85e-5 | 1.35 |
| 48 | 3.38e-5 | 5.79e-5 | 1.72 |
| 51 | 3.05e-5 | 7.31e-5 | 2.39 |
| 54 | 2.78e-5 | 1.03e-4 | **3.69** |
| 60 | 5.96e-5 | 9.58e-5 | 1.61 |
| 75 | 7.26e-5 | 8.79e-5 | 1.21 |

Smooth in theta and reproduced at other settings, so it is a trend and not a Rayleigh spike: at
wl=655 / psi=30 the crossover sits near 48° and the peak is 1.94 at 66°, and at fto=10 the peak is
2.84 at 54°. The closed-form error is also far flatter in angle — the DFT arm's grows with incidence
because that is what `Kx epz_i Kx` does to it.

Below the crossover near 38° the DFT does lead, by about 5%. That is the honest half of the median
above, and it is worth 2.5e-6 absolute against the 7.5e-5 the closed form is worth at 54° — a factor
of 30, in the regime the slot was built for.

Raster convergence at theta=54°, same pillar:

| N | closed form | DFT | DFT / closed |
|---|---|---|---|
| 32 | 1.27e-4 | 4.40e-4 | 3.46 |
| 48 | 5.59e-5 | 1.91e-4 | 3.42 |
| 64 | 2.78e-5 | 1.03e-4 | 3.69 |
| 96 | 8.78e-6 | 4.03e-5 | 4.59 |
| 128 | 4.11e-6 | 1.98e-5 | 4.82 |
| 192 | 3.06e-6 | 6.57e-6 | 2.14 |
| 256 | 2.70e-6 | 2.73e-6 | 1.01 |

The two converge to the same answer — this is a coarse-grid term throughout, as it must be — and the
closed form gets there roughly a raster doubling sooner.

**On a curved boundary neither arm is the limit.** The same sweep on a disc is dominated by the
shape's own pixelation, which both arms carry equally: the error is non-monotone in N and the ratio
wanders between 0.6 and 1.6 with no trend. That is not evidence against the switch, it is the
statement that the z slot is not what limits a curved layer — the staircase is
([`analytic-modeling.md`](analytic-modeling.md), the path where it is gone).

`tests/test_pn_factorization.py::test_the_z_slot_reads_the_raster_as_the_staircase_it_is` pins the
choice, by the invariance that *is* exactness: pixel-repeating a raster respells the same staircase,
so the closed form cannot move and the DFT does. The angle measurement above is too slow to live in
that file.

## 8. It serves `mu` too

Unchanged, by the E/H duality: the rearrangement is about which quantities are continuous at an
interface, and *tangential H with normal B* is *tangential E with normal D* under a rename. See
[`magnetic-materials.md`](magnetic-materials.md) §7 for the wrapper and the two traps that are
specific to the permeability side — the inverted slot order, and a layer whose two tensors have
different boundaries and therefore need two normal fields.

## 9. Reaching it

`to_conv_mat_raster_pn(ucell_tensor, fto_x, fto_y)` takes `(n_layers, Ny, Nx, 3, 3)` and returns
the five slots for the stack. Note it takes **eps, not a refractive index**: `n ** 2` is a scalar
operation and there is no square root of a tensor field worth guessing at.

The public `ucell` is still a scalar refractive index, so a patterned tensor reaches the solver
the same way a uniform one does — through `solve_for_conv` with the slots filled directly, exactly
as [`anisotropic-materials.md`](anisotropic-materials.md) §4 describes for the uniform case.
Wiring a tensor `ucell` through the front door is a separate piece of work and is not done.

```python
import numpy as np
import wavebit
from wavebit._core.factorization import to_conv_mat_raster_pn

eps = np.where(mask, 2.56, 1.0).astype(complex)          # (Ny, Nx)
g   = np.where(mask, 0.30j, 0.0).astype(complex)

t = np.zeros(eps.shape + (3, 3), dtype=complex)
t[..., 0, 0] = t[..., 1, 1] = t[..., 2, 2] = eps
t[..., 0, 1], t[..., 1, 0] = g, -g                       # the physical tensor

epx, epy, epxy, epyx, epz_i = to_conv_mat_raster_pn(t[None], fto, fto)

mee = wavebit.rcwa(backend=0, fto=(fto, fto), period=(p, p), wavelength=wl,
                   thickness=[d], ucell=np.sqrt(eps)[None])
res = mee.solve_for_conv(wl, epx, epy, epz_i, epxy, epyx)   # epz_i is THIRD
```

Two ways to get this wrong, both silent: the slot order above is `(epx, epy, epz_i, epxy, epyx)`
— `epz_i` third, not last — and the off-diagonal **slots hold the negation**, which
`to_conv_mat_raster_pn` has already applied. See
[`off-diagonal-slot-convention.md`](off-diagonal-slot-convention.md).

## 10. References

- M. Nevière and E. Popov, *Light Propagation in Periodic Media* (2003) — the rearrangement.
- T. Schuster et al., "Normal vector method for convergence improvement using the RCWA for crossed
  gratings", JOSA A **24**, 2880 (2007),
  [doi:10.1364/JOSAA.24.002880](https://doi.org/10.1364/JOSAA.24.002880) — the isotropic NVF this
  reduces to, and the normal field it shares.
- L. Li, "Use of Fourier series in the analysis of discontinuous periodic structures", JOSA A
  **13**, 1870 (1996) — the factorization rules and the concurrent-jump condition §3 turns on.
