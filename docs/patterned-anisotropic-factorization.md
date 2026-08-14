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

## 7. It serves `mu` too

Unchanged, by the E/H duality: the rearrangement is about which quantities are continuous at an
interface, and *tangential H with normal B* is *tangential E with normal D* under a rename. See
[`magnetic-materials.md`](magnetic-materials.md) §7 for the wrapper and the two traps that are
specific to the permeability side — the inverted slot order, and a layer whose two tensors have
different boundaries and therefore need two normal fields.

## 8. Reaching it

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

## 9. References

- M. Nevière and E. Popov, *Light Propagation in Periodic Media* (2003) — the rearrangement.
- T. Schuster et al., "Normal vector method for convergence improvement using the RCWA for crossed
  gratings", JOSA A **24**, 2880 (2007),
  [doi:10.1364/JOSAA.24.002880](https://doi.org/10.1364/JOSAA.24.002880) — the isotropic NVF this
  reduces to, and the normal field it shares.
- L. Li, "Use of Fourier series in the analysis of discontinuous periodic structures", JOSA A
  **13**, 1870 (1996) — the factorization rules and the concurrent-jump condition §3 turns on.
