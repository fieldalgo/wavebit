# Scalar shape NVF on a short grid — study

Written: 2026-11. Study only; nothing in the default arm changes. Branch
`claude/scalar-nvf-grid-study`. Targets `src/wavebit/rcwa/_engine/parametric.py::_sampled_coeffs`,
`factorization.check_normal_grid`, `_lifecycle._refuse_aliasing_grid`.

Environment: the versions pinned in `.github/constraints/tests.txt` (NumPy 2.5.4, SciPy 1.18.1,
JAX 0.12.0.1 with `enable_x64`, PyTorch 2.14.1+cpu, contourpy 1.4.0) on Python 3.12.3 (CI pins
3.12), Linux x86_64, CPU only (`results/versions.json`). Scripts and raw results:
`studies/rcwa/scalar_nvf_short_grid/` (`cases.py` and `measure.py` for (a)–(c);
`lattice_gradients.py`, `subfloor_gradients.py` for (c); `section12_runs.py` and `proto_on.py`
for (d); `cost_e.py` for (e); `results/*.json`, `results/*.log`, `results/*.txt`).

## The question

Scalar (isotropic) and tensor shape NVF layers both read the normal field off a
`shape_nvf_grid_size` grid. Below `4*max(orders)+1` the scalar builder refuses the grid
(`numerics.grid`, since #171) and the tensor builder computes it. Is the difference needed,
or only how the two builders were written? Could the scalar builder read the grid the way the
tensor builder does?

## Answer

**Only how the builders were written.** The scalar builder (Schuster et al. 2007 NVF) needs
the projector's Fourier coefficients for `|k| <= 2M` and nothing else; it multiplies truncated
matrices. It took them as wrapped DFT bins (`k % n`). On a grid of `2M` points or fewer per
axis that fold is wrong in kind: a constant field picks up off-diagonals of 1.0. Between `2M+1`
and `4M` it only moves the few orders past Nyquist onto the wrong bin, which is a matter of
degree. The floor refused both bands. The tensor builder (geometry-integrated Popov–Nevière)
integrates the field *between* samples against each region's indicator, so it had to build the
field's trigonometric interpolant. The interpolant's coefficients stop at Nyquist, so an order
past it reads zero instead of a wrapped bin. Both builders still see the field only through its
`n` samples. Nothing in the scalar formulation requires the fold: taking the
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
  For `|k| <= 2M` and `n >= 4M+1`, `2|k| < n`, so every weight is exactly `1.0` on both axes,
  even grids included. The product therefore equals the wrapped bins in value on every
  backend. A complex-by-real product is evaluated as a complex multiply, so only the sign of an
  exactly-zero real or imaginary part can change.
- `check_normal_grid` and `_refuse_aliasing_grid` stop refusing a positive short grid.
- the projector cache key carries the switch.

No branch on the grid size: the weights are applied at every grid, so (a) below measures the
identity rather than asserting it. The raster path is not touched (see (d)).

## (a) Identity at and above the floor

Default arm versus prototype, compared with `np.array_equal`: R and T in every order, and field
samples. The field samples are all six `E`, `H` components on a 2 × 2 grid of points,
x ∈ {0.11, −0.3} and y ∈ {−0.07, 0.2}, at one z. `np.array_equal` treats `−0.0` and `+0.0` as
equal, so this measures equality of value.

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

- **No cliff at the floor.** Accuracy degrades from the default grid down through `4M+1` and
  below with no step at the floor. It is not monotone: there are parity swings between
  neighbouring grids, and the tensor builder has the same shape. The floor was never an
  accuracy threshold. At `4M+1` the error is already 3e-5–5e-2 (14 of 20 cells within
  3e-4–2.5e-3), and even grid 256 is still 2e-8–2e-5 off the default-grid answer.
- **Energy:** lossless cases conserve `R + T` to 1.6e-14 (orders 3) and 1.7e-13 (orders 5) at
  every sub-floor grid on the prototype. The tensor comparator (circle, `diag(4, 3, 2.5)`,
  lossless) does not: `|1 - R - T|` reaches 9.8e-4 (orders 3, grid 8) and 4.5e-4 (orders 5,
  grid 8), and is still 9.5e-5 and 7.7e-6 at grid 256 (`results/b.json`). That is the
  integrated closure's non-Hermitian tensor, not a short-grid effect.
- **Constant field → identity** (`M = 3`). With the interpolant, a constant field transforms to
  the identity at every grid tried (1, 2, 3, 4, 11, 12): off-diagonals ≤ 3e-17, and
  `C_xx + C_yy - I` ≤ 6.2e-17 for a random unit field. Wrapped bins give off-diagonals of
  exactly 1.0 at grids 1–4 (`n <= 2M`, where some `|k| <= 2M` folds onto `k = 0`). At 11 and 12
  they agree with the interpolant, and at `n >= 4M+1` the two coincide (≤ 4.2e-17).
- At grids 2–5 wrapped bins are worse than the prototype in most cases, typically by 2–30x.
  The worst cases are about 80x at orders 3 (rectangle, grid 5: 9.2e-3 versus 1.2e-4) and
  about 130x at orders 5 (ellipse, grid 5: 3.6e-1 versus 2.9e-3). At grids 2–5 the fold happens
  to land closer in a few cases: the pentagon at grid 3, and the hexagonal cell and the lossy
  circle at grid 5 (orders 3); also the rectangle at grid 4 (orders 5, not tabulated). At
  `2M+1 <= n < 4M+1` the two arms are within about 4x of each other (mostly within 1.5x), with
  either one ahead, since only the orders with `n/2 < |k| <= 2M` differ.
- **Grid parity matters more than the floor.** The floor `4M+1` is odd, and an odd grid puts a
  sample on a centred shape's axis. At orders 3, grid 13 is worse than grid 12 in 6 of 10
  cases, 5 of them by more than the table's rounding. Examples: tilted rectangle 1.1e-3 against
  1.6e-4, lossy 4.2e-4 against 2.6e-5, hexagonal cell 7.3e-5 against 1.5e-5. At orders 5,
  grid 21 is worse than grid 20 in 8 of 10. Grid 25 is worse than 16 in 4 of 10 at orders 3
  and 2 of 10 at orders 5. So a grid one step below the floor can beat the floor. The follow-up measures the same odd-grid penalty on
  both builders.
- At grid 1 the two arms agree for the six circle and ellipse cases. Their single sample has a
  zero normal, so both give `P = 0`. They differ for the rectangle, tilted rectangle, pentagon
  and overlapping circles, where wrapped bins are 2–10x worse.
- A short grid is still a poor grid — 2.5e-2–2.6e-1 at grids 1–3 for the metal. What changes is
  that it is wrong by degree rather than in kind.
- A solve with a region of the background's own material equals the unpatterned slab exactly on
  both arms at every grid tried (1, 2, 3, 11, 12, 13, 64; orders (3, 3)); that check is blind to the projector (`Δ = 0`) and does not
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
| pentagon | 4.5e-02 (1e-09) | 3.3e-02 (9e-10) | 4.4e-02 (2e-09) | 2.0e-02 (7e-10) | 3.7e-01 (**2e-04**) | 7.3e-03 (6e-10) | 2.5e-03 (2e-08) | -0.1097 (5e-07) |
| hexagonal cell | 1.3e-02 (2e-08) | 5.9e-02 (2e-08) | 1.6e-02 (**5e-02**) | 4.6e-03 (2e-08) | 1.0e-03 (**2e-04**) | 5.3e-03 (**2e-03**) | 4.1e-05 (2e-08) | -0.3449 (2e-08) |
| lossy (4+1j) | 6.5e-02 (2e-09) | 4.6e-02 (1e-09) | 5.1e-02 (5e-10) | 1.4e-02 (5e-10) | 9.2e-04 (5e-10) | 9.5e-03 (4e-10) | 8.0e-06 (2e-10) | -1.2538 (5e-10) |

- AD agrees with the central difference to 4e-11–3e-6 below the floor, except at the bold
  cells: hexagonal cell at grids 3 (5e-2) and 12 (2e-4), pentagon at grid 12 (2e-4). The
  gradient is as differentiable as the default arm's; it is just as inaccurate as the short grid
  makes the forward value.
- **The grid-13 outlier is not the prototype's.** The hexagonal case disagrees at grid 13, which
  is at the floor, where the prototype equals today's code. With the
  switch off (today's code, `lattice_gradients.py`, orders (3, 3)), eager `jax.grad` on the
  hexagonal cell disagrees with PyTorch backward at grids 13 (1.9e-3), 14 (5.7e-4), 17 (1.8e-4),
  21 (1.3e-3) and 65 (7.6e-5), and agrees to 5e-14–5e-13 at 15, 16, 32, 64, 128 and the default.
  PyTorch backward matches the central difference to 1.7e-8–1.9e-8 at every one of those grids.
  On the square-lattice circle and pentagon and on the oblique-lattice ellipse at grids 13, 32
  and 64, JAX and PyTorch agree to 4.3e-12 or better. So this is a JAX-backend gradient defect on the
  hexagonal lattice in today's code, present above the floor too. It is unrelated to the floor and
  to the prototype, and is left for a separate fix.
- **The sub-floor outliers are not the prototype's either.** With the prototype on
  (`subfloor_gradients.py`), PyTorch backward was run beside `jax.grad` at the three bold cells:
  - hexagonal cell, grids 3 and 12: PyTorch matches the central difference to 1.8e-8 and 1.7e-8;
    `jax.grad` is off by 5.0e-2 and 1.9e-4. This is the same hexagonal-lattice JAX defect.
  - pentagon, grid 12: `jax.grad` and PyTorch agree to 5e-13. The gap is the central difference's:
    it is 4.6e-1 at `h = 1e-3`, then falls as `h^2` (1.7e-2, 1.7e-4, 1.7e-6 at `h` = 1e-4 … 1e-6)
    to 9e-8 at `h = 1e-7`.
    `T(x)` is sharply curved at that grid and point.

  At grid 11, both cases agree three ways (AD, PyTorch, central difference) to 2e-8 or better.
  No gradient defect attributable to the prototype was found.
- PyTorch backward runs below the floor (circle, grid 8: 0.8017).

## (d) What admitting short grids would touch

Code: `factorization.check_normal_grid` (refusal), `_lifecycle._refuse_aliasing_grid` (#171,
`numerics.grid` at check/prepare/run), `engine._stated_normal_grid_meets_its_floor` and the
`_resolve_normal_grid` docstring, `parametric._sampled_coeffs` (the weights), the parametric
caller's docstring (`normal_projector_coeffs`, "checked against the 4*truncation_order+1
floor"), both boundary callers (`boundary.py` lines calling `check_normal_grid`). The docstrings
of `fourier2d_pointwise`, `_normal_field_floor` and `_tensor_at_the_normal_field_floor` describe
the fold; they stay true for the raster path.

Tests. With the prototype switch on, every test file that mentions the grid or the projector
(62 files matching `shape_nvf_grid_size|normal_grid|check_normal_grid|numerics.grid|_sampled_coeffs|normal_projector_coeffs`,
`-m "not slow"`, one pytest process per file, two at a time) gave 3042 passed and 24 failed
(`results/d_tests_on.txt`: per-file summary and every failing test). All 24 pass with the switch
off (`results/d_tests_off.txt`). They are exactly the tests that assert the refusal, plus one that
compares across the floor:

| test | cases | with the switch on |
|---|---|---|
| `tests/api/test_check_run_consistency.py::test_an_aliasing_shape_nvf_grid_is_refused_at_every_door` | 12 (4 stacks x 3 orders/grids) | `check` reports no issue; the test expects a refusal |
| `tests/api/test_check_run_consistency.py::test_an_aliasing_shape_nvf_grid_is_refused_alike_on_each_backend` | 6 (plane wave and beam x 3 backends) | same |
| `tests/rcwa/test_boundary.py::test_normal_grid_floor_is_enforced` | 3 (orders 3, 6, 12) | `check_normal_grid` does not raise |
| `tests/rcwa/test_boundary.py::test_both_shape_paths_draw_the_floor_in_the_same_place` | 2 (analytic, boundary) | neither path raises |
| `tests/rcwa/test_parametric.py::test_conv_matrix_matches_grid_convention` | 1 | raster and analytic spellings differ by 1.8e-2 (tolerance 1e-12) |

The first four are the floor's own tests and would be rewritten, not loosened: section 12 of
`test_check_run_consistency.py` becomes "a short scalar shape grid runs at every door, alike on
each backend", matching `test_a_short_grid_that_runs_today_still_runs` (the tensor arm, which
passes unchanged), and the two `test_boundary.py` tests keep only the positive-grid refusal. The
last one compares a 12x10 grid at orders (2, 3), below the floor on the v axis (4*3+1 = 13),
where the raster spelling (`fourier2d_pointwise` called directly on the unlifted 12x10 cell, a
plain wrapped fold) and the interpolant are meant to differ. It should either
move its grid to the floor or compare against the interpolant. No other test, tolerance, hash or
golden changed outcome. Not run: tests marked `slow`, tests in files outside that list, and CUDA
tests (no GPU; `test_general_cuda_dipole_radiation.py` deselects itself).

What the rewritten section 12 would assert was run directly (`section12_runs.py`,
`results/section12_runs.json`): its 18 problems with the prototype on, at check, prepare and run.
- All 18 are admitted by `check` with no issue. `prepare(...).run` and `run` agree exactly.
  Every output is finite.
- Its 12 plane-wave problems on scalar stacks conserve energy to 1.9e-15 (`region`,
  `region-free`, `no-layers`; three backends for `region`). The `region-free` and `no-layers`
  stacks give the floor-grid and default-grid answer exactly (they read no grid).
- `region` at the short grid is 3.9e-4–4.6e-4 off the default-grid T at orders (2, 2) and
  (3, 1), and 2.0e-2 at orders (1, 0), grid 4.
- `tensor-beside-region` shows `|1 - R - T|` of 1.7e-3–2.8e-3. That belongs to the tensor layer:
  the same stack gives 2.2e-3–5.2e-3 at the floor and default grids, on today's code and on the
  prototype alike.
- The beam problems' boundary powers are finite on all three backends and within 3.7e-3
  (relative) of the floor-grid answer; the plane-wave problems within 5.0e-3.

Docs: `docs/reference/keywords.md` (`shape_nvf_grid_size` row: the `numerics.grid` "below
`max(4*Mu + 1, 4*Mv + 1)`" clause) and `docs/guide/rcwa/boundary-modeling.md` ("The sampled
normal field needs `4*m+1` points per axis"). Neither page has a Korean counterpart under
`docs/locales/ko/pages`, and no Korean page states the floor, so there is no Korean text to
change.

**The raster path is not affected.** A raster's normal field is computed on the raster lifted by
repetition to the floor (`_normal_field_floor`, `_layer_eps_for_normal_field`,
`_tensor_at_the_normal_field_floor`) and transformed by `fourier2d_pointwise`, which the prototype
does not touch. Even if it did, a lifted grid is at or above `4M+1`, where every weight is 1.0.
Whether the lift itself could go (read the raster's own grid through the interpolant) is a
separate question this study did not measure.

`ShapeGridWarning` (#173) is not on `main` at the commit this study started from (21f9621), so
the tensor path's short-grid warning was not exercised here.

## (e) Cost at the default grid

NumPy, CPU, orders `(M, M)`, one solve with the projector cache cleared before it (so each solve
builds its projector), `cost_e.py`. Each (case, arm) ran in its own fresh process after one
untimed warm-up solve: 7 timed solves (5 at `M = 6`, 3 for overlap). The arms ran interleaved,
current, prototype, current, prototype, to spread drift; no other job of this study ran
meanwhile. Peak memory is the process's resident high-water mark, interpreter and NumPy included.
The overlapping-circles case (boundary route) is at grid 256 only: one solve at the default grid
did not finish within ten minutes here (and took over an hour in (a)), so the boundary route was
not timed at the default grid.

| case | M | grid | current s (median, two runs) | prototype s | prototype/current | peak RSS MiB current | prototype |
|---|---|---|---|---|---|---|---|
| rectangle | 3 | default | 0.629 / 0.62 | 0.587 / 0.602 | 0.95 | 149/149 | 149/149 |
| circle | 3 | default | 0.232 / 0.23 | 0.241 / 0.233 | 1.03 | 210/210 | 210/210 |
| ellipse | 3 | default | 0.226 / 0.221 | 0.23 / 0.221 | 1.01 | 210/210 | 210/210 |
| circle | 6 | default | 0.537 / 0.554 | 0.506 / 0.522 | 0.94 | 214/214 | 214/214 |
| overlap | 3 | 256 | 87.6 / 87.2 | 89.6 / 88.6 | 1.02 | 292/307 | 307/304 |
| _sampled_coeffs 1024x1024, M=3 |  |  | 27.7 ms / 29.1 ms | 28.3 ms / 33.2 ms | 1.08 | – | – |

- No cost is visible. The ratios run from 0.94 to 1.08, the prototype is faster in two of the six
  rows, and in every solve row the two arms' fastest solves (`min_s` in `results/e.json`, two per
  arm) overlap. Resident peaks agree to 1 MiB except overlap, where they span 292–307 MiB: the first process
  (current arm, 292 MiB) is 12–16 MiB below the other three (304–307 MiB).
- The added work is one complex-by-real multiply on each `(4Mv+1) x (4Mu+1)` coefficient grid
  (169 entries at `M = 3`) and two weight vectors. Against a 1024 x 1024 FFT in
  `_sampled_coeffs` it is below the timer's spread (last row).
- An earlier run with `tracemalloc` (`results/e_tracemalloc_partial.log`, stopped before the
  overlap case) gave equal traced peaks: 88.06 against 88.05 MiB on the rectangle, 137.04 against
  137.04 MiB on the circle.
- With the "skip the multiply when every weight is 1" form of the recommendation, the default grid
  runs exactly today's code, so its cost is zero by construction.

## Recommendation

**Make the interpolant coefficients the scalar builder's only coefficients, and admit short grids
for scalar shape layers on the same terms as tensor layers.**

1. *Coefficients.* Apply the weights in `parametric._sampled_coeffs` unconditionally and delete
   the switch. Skip the multiply when every weight is 1, which is always the case at and above
   `4M+1`. Then those grids are bit-exact by construction, not only equal in value as measured
   in (a), and no golden, hash or tolerance moves. The cost is two small weight vectors, their
   real outer product and one complex-by-real elementwise multiply on each `(4Mv+1) x (4Mu+1)`
   coefficient grid, three or four per projector build, and only below the floor. At and above
   it the cost is building and checking the two weight vectors; even the unskipped multiply
   measured in (e) was below the timer's spread. The `_sampled_coeffs` docstring
   claims it reproduces `fourier2d_pointwise`; that becomes true only at and above the floor.
2. *Admission.* Stop refusing a positive `shape_nvf_grid_size` below the floor for shape layers:
   - reduce `check_normal_grid` to its positive-grid check at all three call sites:
     `parametric.normal_projector_coeffs`, `boundary.polygon_projector_coeffs` and
     `boundary.normal_projector_coeffs`;
   - delete `_lifecycle._refuse_aliasing_grid` (`numerics.grid` at check, prepare and run) and
     `engine._stated_normal_grid_meets_its_floor`, which only `_refuse_aliasing_grid` consumes.
     The scalar case is their
     only refusal. This also admits scalar stacks whose layers have no regions; they read no
     grid but are refused today.

   Once `ShapeGridWarning` (#173, not in the tree this study ran on) lands, give scalar layers
   the same short-grid warning it gives tensor layers. A short grid is then a stated accuracy
   choice, not an error. Grid 0 and negative grids stay refused (`Numerics` refuses them at
   construction).
3. *Leave alone:* the default grid (1024), the raster path and its lift to the floor
   (`_normal_field_floor`), and `fourier2d_pointwise`. Lifting the raster floor is a separate
   question.
4. *Documentation.* Retire the "floor" wording rather than restating it as an accuracy bound.
   The floor never was one: grid `4M+1` is 3e-5–5e-2 off the default-grid answer ((b)). Being
   odd, it is often worse than `4M` (the parity effect in (b) and in the follow-up). A
   recommendation, if one is wanted, is "an even grid, and converge it".

What changes with it: the tests listed in (d).
- Section 12 of `test_check_run_consistency.py` would assert a finite result at every door, on
  each backend, energy-conserving for its scalar stacks, and the warning once #173 lands. That
  is what `section12_runs.py` measured.
- The two `test_boundary.py` floor tests keep only the positive-grid refusal.
- The test that equates the scalar transform with the raster transform would move to a grid at
  or above the floor, or compare against the interpolant.
- The docs rows listed in (d) change. They are English only.

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
| error per Fourier order, dielectric shapes (geometric mean over `M >= 3`, joint reference) | in \|Δ sum T\|, 2.4–35x smaller, and closer at most `M` in all 9 single-shape dielectric families under either arm's reference, but not on painted overlap; in max per-order \|ΔT\| only 1.1–4x smaller, and the rectangle goes to the integrated arm | larger; changes sign with `M` more often (2–6 sign flips against 0–4) |
| lossless `1 - R - T` | ≤ 2e-12 at every order and grid (≤ 8e-13 in the order sweep; 1.6e-12 in the FEM row at (16, 21)) | 3.7e-2 (high contrast, `M = 3`), 7e-3 (`M = 5`), 2e-3 (`M = 8`); 1e-5–3e-4 for simple shapes at `M = 8`; both signs; does not shrink with the grid (constant from about grid 64) |
| in-plane ε matrix | Hermitian to 1e-16 | anti-Hermitian part 6–28 %, gain eigenvalues in a lossless layer |
| reciprocity (specular, non-centrosymmetric) | ≤ 4e-13 (dielectric) | 3e-3 (pentagon), 3e-2 (painted overlap) |
| 1D lamellar against Li | ≤ 3e-13 dielectric; a metal lamella departs (up to 2.3e-3 in one configuration) where the inverse-rule floor binds | ≤ 1.4e-12, metal included (two metal lamellae) |
| grid convergence, single shapes | about grid^-2 (slopes −1.5 to −3.1); an odd grid costs up to 25–37x on centred circles | about grid^-2 on ellipses; grid^-3 to grid^-4 on centred circles on even grids; an odd grid costs 1–6x on ellipses and hexagonal cells and 10–1000x on centred circles |
| grid convergence, painted overlap | about grid^-1; still about 2e-3 at 256 (against its own grid-512 answer) | about grid^-2; 1e-4 by 128–192 (but `R + T - 1 = 1e-2` at `M = 2`) |
| `T(x)` and gradients, polygon and multi-shape layers | smooth; AD = FD | jagged (sample-crossing steps); on polygons AD follows the slope between steps and differs from a wide finite difference by 13–57 %; on the one two-shape layer differentiated, AD is within 0.5 % of FD |
| short grids | refused today (the subject of this note) | accepted |
| metal disk (−20+1.5j) | unconverged at `M = 16` on both arms; the floor bounds gain | unconverged, but closer at 9 of 10 `M` under either arm's reference (geometric-mean error about 1.8x smaller; 1.4–5x at r = 0.28); the `M = 16` reference itself is unconverged (tail spread 1.6e-2); `sum T = 1.08` at `M = 2` on the same metal disk (r = 0.28), i.e. gain |
| independent 3D FEM (two-piece affine profile) | p within 4e-5–1.1e-4 at every order; s 5e-4–2.4e-3 off | p off by 1e-2 at (4, 5), 1.2e-4 (within the FEM spread of about 1e-4) at (16, 21); s off by 1.5e-2 at (4, 5), but closer than the scalar arm at (12, 16) and (16, 21) (1.9e-4 against 5.0e-4) |
| cost: one NumPy solve (mean of two per-process medians), default grid (painted overlap at grid 64), `cost.py` | rectangle 0.61 s; circle 0.25 s (`M = 3`) and 0.52 s (`M = 6`); polygon 16.5 s; painted overlap 7.6 s. Peak resident memory 148–318 MiB | 4.0x slower on the rectangle, 8.8x and 5.1x on the circle; 0.68x on the polygon and 0.54x on the painted overlap. Peak resident memory 413–491 MiB (207 MiB on the painted overlap) |

All numbers are NumPy, at orders up to 16 for 2D shapes (lamellae to (40, 0), FEM to (16, 21)),
s polarization unless noted. Where AD is exact, JAX and PyTorch agree with each other to 6e-12
on both arms.

What the comparison does show:

- The formulations differ in what they promise. Truncate-then-multiply with a Hermitian projector
  keeps the scalar builder's effective tensor Hermitian, so a lossless layer conserves energy at
  every order. The integrated closure `[[A]][[B]]^-1` is not Hermitian at finite order. This is
  already stated for tensor layers in [tensor shape integration](../../guide/rcwa/tensor-shape-integration.md);
  for isotropic layers it would be a regression.
- The integrated arm is better in four places:
  - the painted (boundary) route's slow, costly grid convergence (about grid^-1);
  - the metal inverse-rule floor binding on a lamella where Li's rule is exact;
  - the metal disk, which neither arm converges by `M = 16` but where the integrated arm is
    closer at most orders;
  - s polarization against FEM at the two highest orders.

  The first two point at separate questions about the scalar builder. None is a reason to
  re-route isotropic layers.
- Part of the polygon and multi-shape roughness belongs to the integrated arm's hard
  nearest-candidate projector (`average=False`). Averaging it, as the scalar boundary route
  already does, removed the steps for polygon and a circle-plus-rectangle layer in a study-side
  patch, but not for painted overlap. A fair "integrated-averaged" arm was not run.
- For centred circles (dielectric, lossy, metal, high contrast), odd grids are consistently worse
  than the neighbouring even grid on both arms, by up to 37x on the scalar arm and about 1000x on
  the integrated arm. For rectangles and ellipses the effect is small or reversed. The floor
  `4M + 1` is odd: in the (b) table, grid 13 is worse than grid 12 in 6 of 10 cases.

Prototype defects found and fixed in passing: with the switch on, `factorization='li'` and
`'laurent'` on shape stacks were refused because their records carry no integrated plans
(fixed: only an NVF record is checked). `adversarial_c.json` predates the fix. After it,
`'laurent'` reaches the same refusals as on the scalar arm (`review_physics_other.json`). Not fixed: `nvf_projector='circular'` is ignored on the
integrated arm.

## Not measured

- (a): the overlapping-circles case at the default grid ran only on NumPy at orders (3, 3); on
  JAX, PyTorch and at (4, 2) its top grid is 256. Only orders (3, 3) and (4, 2) were checked.
- (b), (c): one shape parameter per case; s polarization, one wavelength and angle; orders
  (3, 3) and (5, 5) only for (b), (3, 3) only for (c). Gradients only by eager `jax.grad` (not `jit`, `vmap` or forward mode);
  PyTorch backward only at the points of `lattice_gradients.py` and `subfloor_gradients.py`
  and one circle point. The hexagonal-lattice JAX defect was located (today's code, not tied to
  the floor) but not diagnosed.
- Lifting the raster path's floor through the same interpolant.
- `ShapeGridWarning` (#173) is not on the starting commit, so the warning behaviour the
  recommendation implies was not exercised.
- Tests: only the files that mention the normal grid (62 files, `-m "not slow"`), with the
  prototype switch on. The full suite and the slow markers were not run, nor CI.
- (e): NumPy, CPU, one machine; no GPU, no JAX/PyTorch timings, no timing below the floor. The
  boundary route was timed at grid 256, not at the default grid (one solve there takes over ten
  minutes); its default-grid claim rests on the `_sampled_coeffs` row and on the
  skip-when-all-weights-are-1 form, which runs today's code unchanged.
- (d): section 12's problems at a single wavelength and angle; the beam's power window is not a
  closed surface, so its energy balance was not checked.
- Follow-up: cost only at the five rows of its table; orders past 16
  (the metal is unconverged on both arms); an "integrated-averaged" arm with the averaged
  projector; p polarization outside the FEM row; JAX/PyTorch in the physics checks.
