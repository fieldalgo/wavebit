# Scalar shape NVF on a short grid — study

Written: 2026-11. Study only; nothing in the default arm changes. Branch
`claude/scalar-nvf-grid-study`. Targets `src/wavebit/rcwa/_engine/parametric.py::_sampled_coeffs`,
`factorization.check_normal_grid`, `_lifecycle._refuse_aliasing_grid`.

Environment (the CI-pinned set): Python 3.12.3, NumPy 2.5.4, SciPy 1.18.1, JAX 0.12.0.1
(`enable_x64`), PyTorch 2.14.1+cpu, contourpy 1.4.0. Scripts and raw results:
`studies/rcwa/scalar_nvf_short_grid/` (`cases.py`, `measure.py`, `results/*.json`).

## The question

Scalar (isotropic) and tensor shape NVF layers both read the normal field off a
`shape_nvf_grid_size` grid. Below `4*max(orders)+1` the scalar builder refuses the grid
(`numerics.grid`, since #171) and the tensor builder computes it. Is the difference needed,
or only how the two builders were written? Could the scalar builder read the grid the way the
tensor builder does?

## Answer

**Only how the builders were written.** The scalar builder (Schuster et al. 2007 NVF) needs
the projector's Fourier coefficients for `|k| <= 2M` and nothing else; it multiplies truncated
matrices. It took them as wrapped DFT bins (`k % n`), and that fold is what made a short grid
wrong in kind — a constant field picks up off-diagonals of 1.0 — and so had to be refused. The
tensor builder (geometry-integrated Popov–Nevière) integrates the field *between* samples
against each region's indicator, so it had to build the field's trigonometric interpolant, and
an interpolant never folds. Nothing in the scalar formulation requires the fold: taking the
same `|k| <= 2M` coefficients from the interpolant (zero past Nyquist, the Nyquist mode split
between its two signs, the half-pixel phase as today) is a drop-in replacement.

The two *formulations* do differ (truncate-then-multiply versus integrate-then-truncate; see
the follow-up below), but that difference is not why one refuses short grids and the other
does not.

## The prototype

A private switch, `factorization._NVF_SHORT_GRID_INTERPOLANT` (default `False`). When on:

- `parametric._sampled_coeffs` — the one transform both the parametric and the boundary
  (painted/overlapping) routes use — multiplies the wrapped bins by real per-order weights
  `w(k, n) = 1` if `2|k| < n`, `1/2` if `2|k| = n`, `0` otherwise (`factorization.interpolant_weights`).
  At `n >= 4M+1` every weight is exactly `1.0`, and `x * 1.0` is exact in IEEE arithmetic on
  every backend, so the default arm and the prototype must agree to the bit there.
- `check_normal_grid` and `_refuse_aliasing_grid` stop refusing a positive short grid.
- the projector cache key carries the switch.

No branch on the grid size: the weights are applied at every grid, so (a) below measures
bit-identity rather than asserting it. The raster path is not touched (see (d)).

## (a) Bit-identity at and above the floor

Default arm versus prototype, R and T (every order) and field samples (`E`, `H` at two points),
compared with `np.array_equal`.

| orders | grids | cases | backends | rows | identical |
|---|---|---|---|---|---|
| (3, 3) | 13 (4M+1), 14 (4M+2), 64, 128, default (1024) | rectangle, tilted rectangle, circle, ellipse, pentagon, overlapping circles (boundary route), hexagonal-cell circle, oblique-cell ellipse, lossy (4+1j), metallic (−20+1.5j) | NumPy, JAX, PyTorch | 150 | **150 / 150, max diff 0.0** |
| (4, 2) | 17, 18, 64, 128, default | circle, tilted rectangle, overlap, oblique | NumPy, JAX, PyTorch | 60 | **60 / 60, max diff 0.0** |

The overlapping-circles case at the default grid was compared on NumPy at (3, 3) only; on JAX,
PyTorch and at (4, 2) its top grid is 256 (one boundary-route solve at 1024 took over an hour on
the loaded study machine).

## (b) Below the floor

Max per-order `max(|dR|, |dT|)` against the same case at the default grid (identical on both
arms). Cells below the floor read *prototype / wrapped*, where *wrapped* is today's wrapped
bins with the refusal lifted — what the floor guards against. At and above the floor the arms
are identical. Last row: today's tensor builder on a comparable layer (circle, `diag(4, 3, 2.5)`),
which has no floor.

**Orders (3, 3)** (floor 13):

| case | 1 | 2 | 3 | 4 | 5 | 8 | 11 | 12 | 13 | 16 | 25 | 64 | 256 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rectangle | 1.2e-02 / 2.7e-02 | 2.6e-03 / 1.5e-02 | 4.9e-03 / 3.4e-02 | 1.1e-02 / 1.5e-02 | 1.2e-04 / 9.2e-03 | 1.9e-03 / 1.6e-03 | 8.1e-04 / 7.8e-04 | 8.7e-04 / 8.6e-04 | 3.9e-04 | 8.4e-04 | 8.5e-05 | 6.7e-06 | 4.6e-06 |
| rect-tilted | 1.1e-02 / 4.4e-02 | 2.4e-03 / 1.8e-02 | 4.0e-03 / 2.8e-02 | 1.1e-03 / 2.1e-03 | 3.3e-04 / 7.3e-03 | 3.2e-03 / 3.2e-03 | 8.2e-05 / 1.3e-04 | 1.6e-04 / 1.5e-04 | 1.1e-03 | 9.0e-05 | 1.3e-04 | 1.4e-05 | 1.9e-07 |
| circle | 1.6e-02 / 1.6e-02 | 4.9e-02 / 4.7e-01 | 2.4e-02 / 3.5e-01 | 1.9e-02 / 4.2e-02 | 1.9e-02 / 4.1e-02 | 8.1e-03 / 1.9e-02 | 5.1e-03 / 4.3e-03 | 4.1e-03 / 4.3e-03 | 2.3e-03 | 1.4e-03 | 2.1e-04 | 4.3e-06 | 1.2e-07 |
| ellipse | 5.9e-02 / 5.9e-02 | 9.2e-03 / 3.0e-01 | 3.2e-02 / 1.1e-01 | 5.1e-03 / 9.1e-03 | 3.1e-03 / 2.6e-02 | 2.2e-03 / 2.2e-03 | 2.5e-03 / 2.8e-03 | 7.2e-04 / 6.5e-04 | 2.0e-03 | 3.4e-04 | 5.2e-04 | 2.0e-05 | 1.2e-06 |
| polygon | 5.9e-03 / 5.8e-02 | 5.9e-03 / 1.1e-02 | 4.7e-03 / 2.8e-03 | 3.6e-03 / 3.5e-03 | 4.0e-03 / 5.5e-03 | 3.6e-03 / 3.5e-03 | 2.3e-03 / 2.3e-03 | 2.0e-03 / 2.0e-03 | 2.0e-03 | 1.3e-03 | 6.8e-04 | 1.9e-04 | 1.9e-05 |
| overlap | 4.0e-02 / 2.7e-01 | 4.1e-02 / 4.4e-02 | 3.6e-02 / 7.0e-02 | 3.5e-02 / 7.7e-02 | 4.0e-02 / 4.8e-02 | 5.2e-02 / 5.2e-02 | 5.2e-02 / 5.2e-02 | 5.0e-02 / 5.0e-02 | 4.9e-02 | 4.6e-02 | 3.8e-02 | 1.9e-02 | — |
| hexagonal | 1.1e-02 / 1.1e-02 | 8.0e-03 / 2.3e-01 | 4.4e-03 / 1.5e-02 | 3.7e-04 / 3.1e-03 | 3.4e-03 / 9.3e-04 | 5.4e-04 / 3.2e-04 | 1.4e-04 / 1.1e-04 | 1.5e-05 / 1.4e-05 | 7.3e-05 | 2.5e-05 | 1.9e-05 | 2.2e-06 | 3.2e-07 |
| oblique | 2.7e-02 / 2.7e-02 | 9.6e-03 / 3.0e-02 | 1.4e-02 / 4.3e-02 | 3.2e-03 / 9.0e-03 | 4.4e-03 / 1.0e-02 | 3.2e-04 / 4.3e-04 | 1.3e-03 / 1.2e-03 | 3.1e-04 / 3.1e-04 | 9.1e-04 | 1.8e-04 | 2.5e-04 | 1.1e-05 | 4.2e-07 |
| lossy | 2.1e-02 / 2.1e-02 | 6.2e-03 / 6.0e-02 | 1.7e-03 / 1.1e-01 | 6.4e-04 / 8.0e-04 | 5.1e-03 / 2.2e-03 | 1.6e-04 / 1.3e-04 | 4.3e-04 / 5.8e-04 | 2.6e-05 / 2.4e-05 | 4.2e-04 | 1.1e-05 | 1.1e-04 | 9.1e-07 | 5.4e-08 |
| metallic | 1.7e-01 / 1.7e-01 | 2.6e-01 / 4.3e-01 | 2.5e-02 / 1.1e-01 | 5.2e-02 / 4.2e-01 | 6.9e-02 / 1.8e-01 | 9.7e-03 / 1.9e-02 | 5.7e-03 / 2.5e-03 | 4.9e-03 / 4.9e-03 | 1.5e-03 | 2.1e-03 | 3.5e-04 | 7.0e-05 | 3.9e-06 |
| tensor circle (today) | 5.0e-03 | 1.7e-03 | 5.5e-04 | 2.9e-03 | 6.7e-03 | 5.9e-04 | 1.7e-03 | 8.5e-05 | 2.3e-04 | 2.2e-05 | 1.3e-04 | 3.0e-06 | 1.6e-07 |

**Orders (5, 5)** (floor 21):

| case | 1 | 2 | 3 | 5 | 9 | 19 | 20 | 21 | 25 | 64 | 256 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rectangle | 7.0e-03 / 1.7e-02 | 6.7e-04 / 1.0e-02 | 1.9e-03 / 2.2e-02 | 1.2e-03 / 3.0e-02 | 1.7e-04 / 5.2e-04 | 4.1e-05 / 4.1e-05 | 3.8e-04 / 3.8e-04 | 5.3e-04 | 1.8e-04 | 6.1e-05 | 8.2e-07 |
| rect-tilted | 5.7e-03 / 1.9e-02 | 7.1e-04 / 7.1e-03 | 1.3e-03 / 2.1e-02 | 1.2e-03 / 3.1e-02 | 5.7e-04 / 1.4e-04 | 2.8e-04 / 2.7e-04 | 1.4e-04 / 1.5e-04 | 3.2e-04 | 2.4e-04 | 1.9e-05 | 4.1e-07 |
| circle | 1.3e-02 / 1.3e-02 | 3.2e-02 / 5.6e-02 | 2.1e-02 / 1.8e-01 | 3.2e-03 / 2.4e-01 | 4.6e-03 / 6.6e-03 | 1.1e-03 / 1.1e-03 | 5.4e-04 / 5.3e-04 | 8.7e-04 | 5.4e-04 | 4.3e-06 | 2.1e-07 |
| ellipse | 3.1e-02 / 3.1e-02 | 2.5e-03 / 8.5e-02 | 1.5e-02 / 8.0e-02 | 2.9e-03 / 3.6e-01 | 2.4e-03 / 2.1e-03 | 5.5e-04 / 5.1e-04 | 2.1e-04 / 2.1e-04 | 4.3e-04 | 3.2e-04 | 2.4e-05 | 1.4e-06 |
| polygon | 3.8e-03 / 7.7e-02 | 3.8e-03 / 1.2e-02 | 3.0e-03 / 4.6e-03 | 2.5e-03 / 6.1e-03 | 1.7e-03 / 1.9e-03 | 7.0e-04 / 7.1e-04 | 7.2e-04 / 7.2e-04 | 7.4e-04 | 4.9e-04 | 1.7e-04 | 2.1e-05 |
| overlap | 2.5e-02 / 1.2e-01 | 2.6e-02 / 1.4e-01 | 2.4e-02 / 1.7e-01 | 2.4e-02 / 3.6e-02 | 3.5e-02 / 3.9e-02 | 2.9e-02 / 2.9e-02 | 2.9e-02 / 2.9e-02 | 2.8e-02 | 2.6e-02 | 1.4e-02 | — |
| hexagonal | 8.0e-03 / 8.0e-03 | 5.8e-03 / 2.5e-02 | 3.5e-03 / 1.8e-02 | 1.6e-03 / 1.5e-02 | 6.2e-04 / 1.2e-03 | 4.5e-05 / 3.4e-05 | 1.4e-05 / 1.0e-05 | 2.9e-05 | 2.2e-05 | 6.4e-07 | 8.8e-08 |
| oblique | 1.4e-02 / 1.4e-02 | 4.6e-03 / 1.5e-02 | 6.5e-03 / 3.1e-02 | 1.5e-03 / 1.5e-02 | 1.6e-03 / 1.1e-03 | 9.4e-05 / 7.0e-05 | 6.8e-06 / 6.5e-06 | 4.4e-05 | 1.5e-05 | 3.8e-06 | 1.9e-07 |
| lossy | 1.4e-02 / 1.4e-02 | 4.7e-03 / 9.0e-02 | 2.3e-03 / 1.2e-01 | 2.5e-03 / 1.6e-02 | 1.3e-03 / 3.1e-03 | 2.2e-04 / 2.1e-04 | 1.5e-05 / 1.5e-05 | 1.7e-04 | 1.2e-04 | 1.9e-07 | 2.2e-08 |
| metallic | 2.1e-01 / 2.1e-01 | 1.6e-01 / 5.0e-01 | 1.2e-01 / 2.0e-01 | 3.5e-02 / 2.3e-01 | 3.7e-02 / 1.6e-01 | 4.1e-03 / 1.4e-03 | 2.7e-03 / 3.8e-03 | 9.9e-04 | 6.5e-04 | 6.2e-05 | 3.3e-06 |
| tensor circle (today) | 4.8e-03 | 2.8e-03 | 1.3e-03 | 2.7e-03 | 2.3e-03 | 2.3e-04 | 2.4e-05 | 1.8e-04 | 2.6e-05 | 3.8e-07 | 5.3e-08 |

Overlap stops at grid 64 (the boundary route's cost). No solve raised an error on either arm.

- **No cliff at the floor.** Accuracy degrades smoothly from the default grid down through
  `4M+1` and below, the same shape the tensor builder shows. The floor was never an accuracy
  threshold: at `4M+1` the error is already 7e-5–5e-2 (mostly 4e-4–2e-3), and the default
  grid is what buys 1e-6.
- **Energy:** lossless cases conserve `R + T` to 1.6e-14 (orders 3) and 1.7e-13 (orders 5) at
  every sub-floor grid on the prototype.
- **Constant field → identity:** with the interpolant, a constant field transforms to exactly
  the identity at grids 1, 2, 3, 4, 11, 12 (`max |offdiag| = 0`, `C_xx + C_yy - I` ≤ 6e-17);
  wrapped bins give off-diagonals of exactly 1.0 at every one of those grids. At `n >= 4M+1` the
  two coincide (≤ 4e-17).
- Wrapped bins are 2–20x worse than the prototype at grids 2–5 in most cases (circle at orders
  3, grid 2: 4.7e-1 versus 4.9e-2). There are a few exceptions where the fold happens to land
  closer (pentagon at grid 3, hexagonal cell at 5 and 8, lossy at 5, all at orders 3). The two
  arms are close at `2M+1 <= n < 4M+1`, where only the few orders past Nyquist differ.
- **Grid parity matters more than the floor.** The floor `4M+1` is odd, and an odd grid puts a
  sample on a centred shape's axis. At orders 3, grid 13 is worse than grid 12 in 5 of 10
  cases. Examples: tilted rectangle 1.1e-3 against 1.6e-4, lossy 4.2e-4 against 2.6e-5,
  hexagonal cell 7.3e-5 against 1.5e-5. Grid 25 is worse than 16 in 4 of 10. So a grid one
  step below the floor can beat the floor. The follow-up measures the same odd-grid penalty on
  both builders.
- Grid 1 is the same on both arms: its single sample is the shape centre, whose normal is zero,
  so both give `P = 0`.
- A short grid is still a poor grid — 1e-2–2.6e-1 at grids 1–3 for the metal. What changes is
  that it is wrong by degree rather than in kind.
- A solve with a region of the background's own material equals the unpatterned slab exactly on
  both arms at every grid; that check is blind to the projector (`Δ = 0`) and does not
  discriminate.

## (c) Gradients below the floor (JAX, prototype)

`d(sum T)/dx` by eager `jax.grad`, orders (3, 3), against the default-grid gradient and against a
NumPy central difference (`h = 1e-5`) at the same grid.

Each cell: relative difference from the default-grid gradient, then (in brackets) the relative
AD-versus-central-difference gap at that grid. The last column gives the default-grid gradient itself.

| case | 1 | 2 | 3 | 11 | 12 | 13 (floor) | 64 | default |
|---|---|---|---|---|---|---|---|---|
| circle (12.25) | 7.5e-01 (2e-06) | 2.6e-01 (5e-07) | 1.0e+00 (2e-06) | 1.5e-01 (9e-07) | 1.8e-01 (3e-06) | 8.7e-02 (1e-06) | 3.5e-04 (2e-06) | 1.2209 (2e-06) |
| tilted rectangle | 6.3e-02 (2e-09) | 2.7e-03 (8e-10) | 4.4e-02 (4e-11) | 4.4e-02 (2e-09) | 8.1e-02 (1e-08) | 8.5e-02 (2e-08) | 6.9e-03 (2e-08) | -0.0528 (4e-07) |
| ellipse | 1.2e+00 (1e-08) | 5.6e-01 (2e-08) | 3.0e-01 (1e-08) | 1.1e-01 (2e-08) | 4.1e-02 (6e-09) | 7.6e-02 (1e-08) | 4.8e-04 (8e-09) | 0.0471 (1e-08) |
| pentagon | 4.5e-02 (1e-09) | 3.3e-02 (9e-10) | 4.4e-02 (2e-09) | 2.0e-02 (7e-10) | 3.7e-01 (2e-04) | 7.3e-03 (6e-10) | 2.5e-03 (2e-08) | -0.1097 (5e-07) |
| hexagonal cell | 1.3e-02 (2e-08) | 5.9e-02 (2e-08) | 1.6e-02 (**5e-02**) | 4.6e-03 (2e-08) | 1.0e-03 (2e-04) | 5.3e-03 (**2e-03**) | 4.1e-05 (2e-08) | -0.3449 (2e-08) |
| lossy (4+1j) | 6.5e-02 (2e-09) | 4.6e-02 (1e-09) | 5.1e-02 (5e-10) | 1.4e-02 (5e-10) | 9.2e-04 (5e-10) | 9.5e-03 (4e-10) | 8.0e-06 (2e-10) | -1.2538 (5e-10) |

- AD agrees with the central difference to 1e-6–1e-9 below the floor, except where noted. The
  gradient is as differentiable as the default arm's; it is just as inaccurate as the short grid
  makes the forward value.
- **The two outliers are not the prototype's.** The hexagonal case disagrees at grid 13 — which
  is at the floor, where the prototype is bit-identical to today — and the central difference is
  smooth there at every step from 1e-3 to 1e-7. On unmodified code, eager `jax.grad` on the
  hexagonal cell differs from PyTorch backward (which equals the central difference) by
  1.9e-3 / 5.7e-4 / 1.8e-4 / 1.3e-3 / 7.6e-5 at grids 13 / 14 / 17 / 21 / 65, and agrees to 1e-13
  at 15, 16 and 128 (1e-8 at the default). This is an existing JAX gradient defect on the
  hexagonal lattice, reported separately. Circle, polygon and the oblique ellipse showed no such
  mismatch at 13 / 32 / 64.
- PyTorch backward runs below the floor (circle, grid 8: 0.8017).

## (d) What admitting short grids would touch

Code: `factorization.check_normal_grid` (refusal), `_lifecycle._refuse_aliasing_grid` (#171,
`numerics.grid` at check/prepare/run), `engine._stated_normal_grid_meets_its_floor` and the
`_resolve_normal_grid` docstring, `parametric._sampled_coeffs` (the weights), the parametric
caller's docstring (`normal_projector_coeffs`, "checked against the 4*truncation_order+1
floor"), both boundary callers (`boundary.py` lines calling `check_normal_grid`). The docstrings
of `fourier2d_pointwise`, `_normal_field_floor` and `_tensor_at_the_normal_field_floor` describe
the fold; they stay true for the raster path.

PASTE_D_TESTS

Docs (English and Korean together): `docs/reference/keywords.md` (`shape_nvf_grid_size` row:
the `numerics.grid` "below `max(4*Mu + 1, 4*Mv + 1)`" clause), `docs/guide/rcwa/boundary-modeling.md`
("The sampled normal field needs `4*m+1` points per axis"), and the Korean pages that mirror them.

**The raster path is not affected.** A raster's normal field is computed on the raster lifted by
repetition to the floor (`_normal_field_floor`, `_layer_eps_for_normal_field`,
`_tensor_at_the_normal_field_floor`) and transformed by `fourier2d_pointwise`, which the prototype
does not touch. Even if it did, a lifted grid is at or above `4M+1`, where every weight is 1.0.
Whether the lift itself could go (read the raster's own grid through the interpolant) is a
separate question this study did not measure.

`ShapeGridWarning` (#173) is not on `main` at the commit this study started from (21f9621), so
the tensor path's short-grid warning was not exercised here.

## (e) Cost at the default grid

PASTE_E_TABLE

## Recommendation

**Make the interpolant coefficients the scalar builder's only coefficients, and admit short grids
for scalar shape layers on the same terms as tensor layers.**

1. *Coefficients.* Apply the weights in `parametric._sampled_coeffs` unconditionally and delete
   the switch. At and above `4M+1` the result is bit-identical on all three backends ((a)), so no
   golden, hash or tolerance moves. The cost is one real elementwise multiply on a
   `(4M+1) x (4M+1)` array ((e)).
2. *Admission.* Stop refusing a positive `shape_nvf_grid_size` below the floor for shape layers:
   - remove the refusal from `check_normal_grid` for its two shape callers (`parametric`,
     `boundary`);
   - drop the scalar shape case from `_lifecycle._refuse_aliasing_grid` (`numerics.grid`);
   - let `engine._stated_normal_grid_meets_its_floor` stop distinguishing scalar from tensor
     shape layers, so one rule covers both.

   Give scalar layers the same short-grid warning tensor layers get (`ShapeGridWarning`, #173),
   so a short grid is a stated accuracy choice, not an error. Grid 0 and negative grids stay
   refused.
3. *Leave alone:* the default grid (1024), the raster path and its lift to the floor
   (`_normal_field_floor`), and `fourier2d_pointwise`. Lifting the raster floor is a separate
   question.
4. *Documentation.* Retire the "floor" wording rather than restating it as an accuracy bound.
   The floor never was one: grid `4M+1` is 1e-4–5e-2 off the default-grid answer ((b)), and
   being odd, it is often worse than `4M` (the parity effect in the follow-up). A
   recommendation, if one is wanted, is "an even grid, and converge it".

What changes with it: the tests listed in (d) that assert the refusal, which would assert the
warning and a finite, energy-conserving result instead. The docs rows listed in (d) change in
English and Korean. No other test changed outcome with the prototype on.

## Follow-up: isotropic layers through the integrated (Popov–Nevière) builder

The user-facing option is `factorization='nvf'` for both builders, so the natural follow-up was
whether isotropic shape layers should go through the integrated builder too, which would give
one route and no floor at all. A second private switch, `tensor_shapes._ISOTROPIC_INTEGRATED`
(default `False`), sends every patterned isotropic layer through `tensor_shapes.build`: plain
scalar stacks under `factorization='nvf'`, and the isotropic layers of a tensor stack. A
geometry frozen under one setting is refused under the other. With the switch off, a SHA-256
over R and T (9 cases × NumPy and PyTorch, plus a mixed tensor stack) matches the pre-change
tree, and the tensor-stack test files pass (52). Scripts and raw numbers:
`studies/rcwa/isotropic_integrated/` (`cases.py`, `orders_*`, `grid_*`, `gradients_*`,
`physics_*`, `review_*`, `adversarial_*`).

**Answer: keep the scalar builder for isotropic layers.** Do not split or re-route the option.

| measure | scalar NVF (today) | integrated (Popov–Nevière) |
|---|---|---|
| error per Fourier order, dielectric single shapes (geometric mean over `M >= 3`, joint reference) | 2–35x smaller; closer at most `M` in 9 of 10 case families, under either arm's reference | larger; oscillates in sign with `M` |
| lossless `1 - R - T` | ≤ 1.7e-13 at every order and grid | 3.7e-2 (high contrast, `M = 3`), 7e-3 (`M = 5`), 2e-3 (`M = 8`); 1e-5–3e-4 for simple shapes at `M = 8`; both signs; independent of grid |
| in-plane ε matrix | Hermitian to 1e-16 | anti-Hermitian part 6–28 %, gain eigenvalues in a lossless layer |
| reciprocity (specular, non-centrosymmetric) | 1e-13 (dielectric) | 3e-3 (pentagon), 3e-2 (painted overlap) |
| 1D lamellar against Li | ≤ 3e-13 dielectric; a metal lamella departs (up to 2.3e-3 in one configuration) where the inverse-rule floor binds | ≤ 1e-12, metal included |
| grid convergence, smooth convex shapes | grid^-2 to grid^-3 | steeper on even grids (grid^-3 to grid^-4), but an odd grid costs 10–300x |
| grid convergence, painted overlap | about grid^-1; still about 2e-3 at 512 | about grid^-2; 1e-4 by 128–192 (but `R + T - 1 = 1e-2` at `M = 2`) |
| `T(x)` and gradients, polygon and multi-shape layers | smooth; AD = FD | jagged (sample-crossing steps); AD differs from the local slope by 20–60 % |
| short grids | refused today (the subject of this note) | accepted |
| metal disk (−20+1.5j) | unconverged at `M = 16` on both arms; the floor bounds gain | unconverged; ahead on average but reference-dependent; `sum T = 1.08` at `M = 2` on a lossy disk |
| independent 3D FEM (two-piece affine profile) | p within 4e-5–1.1e-4 at every order | p off by 1e-2 at (4, 5), 1.2e-4 only at (16, 21) |

All numbers are NumPy at orders up to 16, s polarization unless noted. Where AD is exact, JAX and
PyTorch agree with each other to 1e-12 on both arms.

What the comparison does show:

- The formulations differ in what they promise. Truncate-then-multiply with a Hermitian projector
  keeps the scalar builder's effective tensor Hermitian, so a lossless layer conserves energy at
  every order. The integrated closure `[[A]][[B]]^-1` is not Hermitian at finite order. This is
  already stated for tensor layers in [tensor shape integration](../../guide/rcwa/tensor-shape-integration.md);
  for isotropic layers it would be a regression.
- Two places where the integrated arm is better point at separate questions about the scalar
  builder: the painted (boundary) route's slow, costly grid convergence (about grid^-1), and the
  metal inverse-rule floor binding on a lamella where Li's rule is exact. Neither is a reason to
  re-route isotropic layers.
- Part of the polygon and multi-shape roughness belongs to the integrated arm's hard
  nearest-candidate projector (`average=False`). Averaging it, as the scalar boundary route
  already does, removed the steps for polygon and a circle-plus-rectangle layer in a study-side
  patch, but not for painted overlap. A fair "integrated-averaged" arm was not run.
- Odd grids are consistently worse than the neighbouring even grid on both arms for centred
  shapes. This includes `4M + 1`, the floor itself, which is odd: in the (b) table, grid 12 beats
  grid 13 in half the cases.

Prototype defects found and fixed in passing: with the switch on, `factorization='li'` and
`'laurent'` on shape stacks were refused because their records carry no integrated plans
(fixed: only an NVF record is checked). Not fixed: `nvf_projector='circular'` is ignored on the
integrated arm.

## Not measured

- (a): the overlapping-circles case at the default grid ran only on NumPy at orders (3, 3); on
  JAX, PyTorch and at (4, 2) its top grid is 256. Only orders (3, 3) and (4, 2) were checked.
- (b), (c): one shape parameter per case; s polarization, one wavelength and angle; orders
  (3, 3) and (5, 5) only. Gradients only by eager `jax.grad` (not `jit`, `vmap` or forward mode);
  PyTorch backward at one point. The polygon grid-12 AD-versus-FD gap (2e-4) was not
  investigated.
- Lifting the raster path's floor through the same interpolant.
- `ShapeGridWarning` (#173) is not on the starting commit, so the warning behaviour the
  recommendation implies was not exercised.
- Tests: only the files that mention the normal grid (62 files, `-m "not slow"`), with the
  prototype switch on. The full suite and the slow markers were not run, nor CI.
- (e): NumPy, CPU, one machine shared with other work; no GPU, no JAX/PyTorch timings.
- Follow-up: wall-clock cost of the integrated arm (beyond the one row in (e)); orders past 16
  (the metal is unconverged on both arms); an "integrated-averaged" arm with the averaged
  projector; p polarization outside the FEM row; JAX/PyTorch in the physics checks.
