# Analytic (staircase-free) modeling — approach and results

Written: 2026-08
Targets: `fieldwave/on_{numpy,jax,torch}/emsolver/fourier_analytic.py`,
      `emsolver/nvf.py::get_normal_vector_field_analytical`,
      `emsolver/transfer_method.py::transfer_2d_2` (off-diagonal slots),
      `emsolver/rcwa.py` (modeling type 2 dispatch)

Related note: the NVF factorization derivation/verification is in [`NVF_FINDINGS.md`](NVF_FINDINGS.md).

---

## TL;DR

The existing vector/raster modelers approximate tilted or curved boundaries with a
**staircase of axis-aligned rectangles**. That staircase is pure geometric error, so it
only shrinks with high Fourier order / fine grids, and there is no **true smooth
boundary/normal** for the NVF (normal vector field) factorization to exploit.

This work adds a modeling path that computes the **exact analytic 2D Fourier
coefficients** of smooth primitives (rotated rectangle, circle/disk, ellipse) and feeds
them directly to the solver without staircase decomposition. `⟦ε⟧` and `⟦1/ε⟧` are formed
analytically, and the normal projectors are built from an **analytic normal field**, then
factorized with NVF.

Result: it **converges at far lower order than the staircase path for both tilted
(dielectric) and curved (metallic) boundaries**, satisfies energy conservation to machine
precision for lossless structures, agrees across numpy/jax/torch, and is differentiable
(compatible with the optimization pipeline).

---

## 1. Analytic Fourier coefficients

The coefficient convention matches `fourier_analysis.dfs2d` (checked to machine precision):

```
c(ky, kx) = 1/(Px Py) ∫∫_cell f(x,y) exp(-i 2π (ky y/Py + kx x/Px)) dx dy
```

With `u = 2π kx/Px`, `v = 2π ky/Py` this is the continuous FT of the shape indicator
evaluated on the reciprocal lattice. For rotation angle `θ` and center `(cx,cy)` we use the
rotated frequency coordinate `κ = Rot(θ)^T k` and the center phase `e^{-i k·c}`.

| primitive | `Ŝ(k) / (Px Py)` |
|---|---|
| rotated rectangle `lx,ly` | `e^{-i k·c} · lx·sinc(κx lx/2) · ly·sinc(κy ly/2)` |
| circle/disk diameter `lx` | `e^{-i k·c} · (π R²)·jinc(R ρ)`,  `R=lx/2`, `ρ=√(u²+v²)` |
| ellipse semi-axes `a=lx/2,b=ly/2` | `e^{-i k·c} · (π a b)·jinc(√((a κx)²+(b κy)²))` |

Here `sinc(z)=sin z/z`, `jinc(z)=2 J1(z)/z` (both removable at 0, value 1). The circle is the
`a=b=R` special case of the ellipse. All three formulas are verified against a fine
midpoint DFT (`tests_verification/test_analytic.py`).

A layer's coefficients are assembled as background + sum of (non-overlapping) shapes:

```
eps_c(k)     = eps_bg·δ_k0 + Σ_s (conj(n_s)² − conj(n_bg)²)·Ŝ_s(k)/(Px Py)
inv_eps_c(k) = (1/eps_bg)·δ_k0 + Σ_s (1/conj(n_s)² − 1/conj(n_bg)²)·Ŝ_s(k)/(Px Py)
```

`conj(n)²` matches fieldwave's `e^{-iωt}` convention (Im n>0 = loss), i.e. the same as
`to_conv_mat_raster_nvf`. `laurent_conv_2d` builds the block-Toeplitz matrices `⟦ε⟧`,
`⟦1/ε⟧` from this `(4fto+1)²` coefficient grid in the same ordering as `dfs2d`.

## 2. Coupling with the NVF factorization

Key observation: the five matrices the NVF consumer (`transfer_2d_2`) needs —
`(εxx, εyy, −εxy, −εyx, inv⟦ε⟧)` — and their building blocks (`⟦ε⟧`, `⟦1/ε⟧`, and the normal
projectors `⟦nx²⟧,⟦nx ny⟧,⟦ny²⟧`) are **all Laurent (direct-rule) convolutions**. No
axis-aligned 1D Li factorization is required. So the analytic coefficients plug straight
into the NVF path:

```
E = ⟦ε⟧(analytic),  A = inv(⟦1/ε⟧(analytic)),  Δ = E − A
εxx = E − ½(Δ Cxx + Cxx Δ),  εyy = E − ½(Δ Cyy + Cyy Δ),  εxy = εyx = −½(Δ Cxy + Cxy Δ)
epz_i = inv(⟦ε⟧)
```

The projectors `Cxx=⟦nx²⟧` etc. are obtained by sampling the **analytic unit normal field**
(`get_normal_vector_field_analytical`, a level-set gradient) on a fine grid and running
`dfs2d`. Unlike the SDF normals of a staircased boundary, these are the true-boundary
normals, so the full NVF benefit is realized. For why the `½`-symmetrization is needed, see
[`NVF_FINDINGS.md`](NVF_FINDINGS.md).

## 3. Usage (new modeling type)

Passing `ucell` as a **list of layer dicts** dispatches to the analytic path (modeling type
2, grating type 2). The existing raster (ndarray) and vector (nested list) paths keep
working.

```python
ucell = [dict(n_base=1.0, shapes=[
             dict(type='circle', cx=0.5, cy=0.5, lx=0.6, ly=0.6, n_index=3.5)]),
         dict(n_base=1.5, shapes=[
             dict(type='rectangle', cx=0.4, cy=0.5, lx=0.4, ly=0.5,
                  angle=np.deg2rad(25), n_index=1.0),
             dict(type='ellipse', cx=0.8, cy=0.9, lx=0.3, ly=0.2,
                  angle=0.6, n_index=3.0)])]
mee = fieldwave.call_mee(backend=0, ucell=ucell, fto=(11, 11), ...)  # analytic_normal_grid=256 (default)
res = mee.conv_solve()
```

`lx/ly/angle/cx/cy/n_index` may be tensors (torch/jax) and are differentiable w.r.t. them.

## 4. Results (`benchmarks/benchmark_analytic.py`)

**[1] Dielectric tilted rectangle** (30°, n=3.5, TM, subwavelength). Reference = analytic
fto=16. Same rectangle represented analytically vs. as a vector staircase (n_split); `t0`
error:

| fto | analytic | staircase n=8 | staircase n=24 |
|---:|---:|---:|---:|
| 3 | 3.6e-2 | 7.4e-2 | 7.5e-2 |
| 5 | 1.0e-2 | 5.0e-2 | 4.9e-2 |
| 7 | 3.7e-3 | 4.0e-2 | 3.8e-2 |
| 9 | 2.4e-3 | 3.4e-2 | 3.1e-2 |
| 11 | **6.2e-4** | 3.0e-2 | 2.6e-2 |

Analytic at fto=5 (1e-2) already beats the staircase at fto=11 (n=24, 2.6e-2), and is ~40×
more accurate at fto=11. The staircase saturates at its geometric-error floor (~3e-2) no
matter the order.

**[2] Metallic circle** (n=0.05+3.5j, R=0.3, TM). Reference = analytic fto=16. `t0` error:

| fto | analytic-NVF | raster-NVF (SDF normals) | raster-DFS |
|---:|---:|---:|---:|
| 3 | 1.6e-2 | 1.5e-2 | 4.2e-2 |
| 7 | 1.5e-2 | 1.6e-2 | 1.2e-2 |
| 9 | **1.5e-3** | 1.1e-2 | 2.6e-2 |
| 11 | **1.4e-3** | 5.5e-3 | 1.8e-2 |

At high order the accuracy ranks **analytic-NVF < raster-NVF < DFS** (as expected). Metal
oscillates at low order but the high-order trend is clear. Confirms that the NVF benefit
requires **true geometry/normals**.

**[3] Lossless multi-shape / multi-layer** (ellipse + rectangle + circle, conical
incidence, non-square period): `|R+T−1|` = 6.7e-16 (fto5), 1.1e-14 (fto8) — machine-
precision energy conservation. Backend agreement: `|np−jax|`~1e-14, `|np−torch|`~5e-10.

## 5. Backend consistency and differentiability

- numpy: `scipy.special.j1`. jax: a self-contained hybrid `_j1` (ascending series +
  asymptotic expansion, `<9e-13` on `[0,95]`, differentiable) — jax's `bessel_jn` returns
  NaN for small arguments and is unusable. torch: `torch.special.bessel_j1` (accurate but
  **has no backward**) wrapped in a custom autograd providing the analytic derivative
  `J1' = J0 − J1/x`.
- The ellipse coefficient's `κ=√(...)` has a NaN sqrt(0) derivative at the DC point, guarded
  with a safe-sqrt.
- Verification: `dT/dlx` w.r.t. circle/ellipse semi-axes matches finite differences
  (torch·jax, `test_analytic.py`).

## 6. Regression tests

`tests_verification/test_analytic.py`: (1) laurent==dfs2d convention, (2) the three shape
formulas vs a fine DFT, (3) axis-aligned analytic==raster-NVF, (4) lossless energy
conservation, (5) backend agreement, (6) torch/jax differentiability, (7) analytic ≥5× more
accurate than staircase for a tilted dielectric. Existing `test_nvf*` and `test_fixes` all
keep passing.

## 7. Limitations and next steps

- Shapes are assumed **non-overlapping** (indicator addition). Overlap support needs
  inclusion–exclusion handling.
- The normal field for multiple shapes is a superposition of level-set gradients
  (heuristic). A single shape is exact.
- The normal projectors still depend on `analytic_normal_grid` sampling (default 256) — the
  coefficients themselves are analytic, but the projectors retain grid error. Increase the
  grid if needed.
- `fourier_type` is ignored on the analytic path (always the NVF 5-matrix form). For a
  non-NVF pure-Laurent comparison, call `fourier_analytic.to_conv_mat_analytic` directly.

### References
- L. Li, JOSA A 13, 1870 (1996). — Fourier factorization / inverse rule.
- P. Götz, T. Schuster et al., JOSA A 24, 2880 (2007). — normal vector method.
