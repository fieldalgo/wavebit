# Laurent, direct, inverse, Toeplitz: which word names a rule and which names a matrix

Written: 2026-08-12. Structure claims below are measured, not asserted; the check is quoted in
§3 and reproduces in seconds. §6 rewritten 2026-09-25, when the raster Li builders were
rebuilt from Li's paper and gained the second nesting. §9 added 2026-09-27, when Laurent's rule
became a value of `factorization=`.

Targets:
- `src/wavebit/rcwa/_engine/parametric.py::conv_matrix_2d` — the builder the collision used to be in; it
  was called `laurent_conv_2d` until this document was written
- `src/wavebit/rcwa/_engine/parametric.py::_grid`, `src/wavebit/rcwa/_engine/parametric.py::assemble_eps_coeffs` —
  the doubled reciprocal grid the builder consumes, and why it is doubled
- `src/wavebit/rcwa/_engine/factorization.py::fourier2d_piecewise_constant` — the raster twin, and where
  `inverse_rule` chooses the rule
- `tests/rcwa/test_parametric.py` — `test_conv_matrix_matches_grid_convention`, which pins the ordering
  the two builders share
- `src/wavebit/rcwa/_engine/factorization.py::_li_slot` and `src/wavebit/rcwa/_engine/enums.py::Nesting` — the two
  nestings of Li's rules, and the keyword that chooses between them (§6)
- `src/wavebit/rcwa/_engine/factorization.py::laurent_conv_matrices` and
  `src/wavebit/rcwa/_engine/parametric.py::laurent_from_coeffs` — Laurent's rule as a factorization
  option, on the raster and on shapes (§9)

---

## TL;DR

Four words are in play and they belong to two different axes. **Laurent** and **direct** name
the same *factorization rule*; **inverse** (Li) names the other one. **Toeplitz** names a
*matrix shape* and no rule at all.

One function used to mix the axes. `conv_matrix_2d` was `laurent_conv_2d`, which put a rule
word on a builder that serves both rules; §5 records the rename and what stayed behind.

Two things are worth knowing before editing any of this:

1. `conv_matrix_2d` is **rule-neutral**. It is called on `eps` and on `1/eps` two lines apart,
   and on the three normal-field products. No rule has been chosen when it runs.
2. The matrix it returns is **not Toeplitz**. It is BTTB — block-Toeplitz with Toeplitz blocks
   — and the distinction is checkable, not pedantic (§3).

And one thing about the rules themselves: in 2D, Li's rules are applied one axis inside the
other, and there are two ways to nest them. Only Li's own treats x and y alike. The one
this path always built, and still builds by default, does not (§6).

Since 2026-09-27 the rule word is also a *value*: `factorization='laurent'` closes every product
with the direct rule, opt-in, for comparing with solvers that use it. What it builds, where it is
refused, and what it costs in accuracy are in §9.

## 1. The two rules

Multiplication in real space is convolution in Fourier space, so `D = eps E` becomes a matrix
acting on the coefficient vector of `E`. Which matrix depends on whether the two factors are
discontinuous at the same place.

| Rule | Also called | Matrix used | When it is the correct one |
| --- | --- | --- | --- |
| direct | Laurent, "plain Laurent" | `<eps>` | the factors are not concurrently discontinuous |
| inverse | Li's rule | `<1/eps>^-1` | they are — along that direction |

Both names for the first rule are in use in this repository and both are correct.
`docs/theory/rcwa/formulation.md` writes `Laurent (direct)`, `docs/guide/rcwa/parametric-modeling.md` writes
`Laurent (direct-rule)`, and `src/wavebit/rcwa/_engine/factorization.py` writes the pair as `direct` / `inverse`
in code (`ep_dir`, `ep_inv`). The gloss form — naming one and parenthesising the other — is the
one to copy in new prose, because a reader who knows only one name still lands.

`fourier2d_piecewise_constant`'s `inverse_rule` is the switch: `'u'` or `'v'` names the
direction whose discontinuity the rule closes, `None` asks for the direct rule both ways.
Those are the *cell's* two directions, not the lab's — they coincide only on a rectangular
cell, so `'x'` and `'y'` are refused by name rather than reinterpreted.
Nothing else selects it, and `fourier2d_pointwise` does not offer it at all — a field that is
continuous by construction has no discontinuity for the inverse rule to close.

## 2. What `conv_matrix_2d` actually is

It takes a grid of Fourier coefficients and lays them out as a convolution matrix:

```
M[(r,s),(c,d)] = coeff(ky = r - c, kx = s - d)
```

That is all it does. It never sees a permittivity, and it does not know which rule its caller
is building. In `to_conv_mat_parametric_nvf` it is called five times in a row — on `eps`, on
`1/eps`, and on the three normal-field products `nx^2`, `nx ny`, `ny^2`:

```python
ep_dir = conv_matrix_2d(xp, eps_c,     half_width_u, half_width_v)
ep_inv = conv_matrix_2d(xp, inv_eps_c, half_width_u, half_width_v)
...
delta_ep = ep_dir - xp.linalg.inv(ep_inv)
```

The inverse rule's *inversion* is the caller's, on the line after. The builder is shared.

This is what the old name got wrong: it carried a rule word into a place where no rule has been
chosen yet, so `ep_inv = laurent_conv_2d(...)` read as a contradiction without being one.

The index differences `r - c` and `s - d` each span `-2*truncation_order .. +2*truncation_order`, which is why the
coefficient grid arrives on a **doubled** reciprocal range — see `_grid`. A grid of only
`2*truncation_order+1` per axis cannot fill the corners of the matrix.

## 3. The matrix is BTTB, not Toeplitz

A Toeplitz matrix is constant along its diagonals: `T[i,j]` depends on `i - j` alone. The
matrix here does not satisfy that. Flattening `i = r*n_orders_u + s` and `j = c*n_orders_u + d`,

```
i - j = (r - c) * n_orders_u + (s - d)
```

and `s - d` ranges over `-(n_orders_u-1) .. (n_orders_u-1)`, which is wider than `n_orders_u`. So `(r-c, s-d)` is
not recoverable from `i - j`: the pairs `(k, m)` and `(k+1, m - n_orders_u)` land on the same
diagonal carrying different values.

Checked directly on `half_width_u, half_width_v = 2, 3` (a 35x35 matrix, random complex coefficients):

| Property | Holds |
| --- | --- |
| Toeplitz in the flattened index — `M[i,j]` depends on `i-j` alone | **no** |
| block-Toeplitz — block `B[r,c]` depends on `r-c` alone | yes |
| every block is itself Toeplitz | yes |

The last two together are the definition of **BTTB**, block-Toeplitz with Toeplitz blocks.
`block-Toeplitz` alone is the usual shorthand and appears in `src/wavebit/lattice.py`,
`docs/theory/rcwa/formulation.md`, `docs/guide/rcwa/parametric-modeling.md` and `docs/theory/rcwa/oblique-lattice.md`; it is
under-specified rather than wrong, since a block-Toeplitz matrix may have arbitrary blocks.
Harmless in prose, too weak to justify a function name.

## 4. Which word to use where

| Writing about | Use | Avoid |
| --- | --- | --- |
| the factorization choice | `direct` / `inverse`, or `Laurent (direct)` | `Laurent` bare, where a matrix could be meant |
| the matrix a builder returns | `convolution matrix` | `Toeplitz matrix` — it is not one |
| its structure, precisely | `BTTB` | — |
| its structure, in passing | `block-Toeplitz` | — |
| the builder itself | name what it does, not what shape comes out | a rule word |

## 5. The rename, and what did not move

`laurent_conv_2d` became `conv_matrix_2d`. Naming it for the structure was considered and
rejected: `toeplitz_` is false, `block_toeplitz_` under-specified, `bttb_` unreadable. What is
left is naming it for what it produces — a convolution matrix — which is the vocabulary the
rest of the code already uses (`nvf_conv_matrices`, `to_conv_mat_parametric_nvf`, `epx_conv`).

Done before the first release, while `_core` is private and nothing outside it imports the
name, so no alias was kept and none is owed.

| Moved | Count | Where |
| --- | --- | --- |
| `laurent_conv_2d` → `conv_matrix_2d` | 14 | `src/wavebit/rcwa/_engine/parametric.py`, `tests/rcwa/test_parametric.py`, `parametric-modeling.md` |
| `test_laurent_matches_dfs2d_convention` → `test_conv_matrix_matches_grid_convention` | 4 | `tests/rcwa/test_parametric.py`, `src/wavebit/rcwa/_engine/parametric.py`, `parametric-modeling.md` |
| "Laurent matrix" meaning the structure → "convolution matrix" | 2 | `src/wavebit/rcwa/_engine/parametric.py` |

**`Laurent` meaning the rule stayed, in all 15 places.** It is correct there, and one of them
is not prose at all: `modes=('nvf', 'laurent')` in
`studies/rcwa/benchmark_truncation_shape/benchmark_truncation_shape.py` is a value a user types. A rename that had swept the
word out of the repository would have broken that and fixed nothing.

## 6. The nested rule, and the two ways to nest it

Rewritten 2026-09-25. Until then this section said the `li` builder's x/y asymmetry "is intrinsic
to Li's crossed-grating scheme". It is not. It comes from using **one nesting order for both
field components**, and Li's own construction, which picks the order per component, does not
have it. Both are now built; `li_nesting=` chooses (`src/wavebit/rcwa/_engine/enums.py::Nesting`).

Applying the rules in 2D means nesting them (L. Li, JOSA A 14, 2758, 1997): one axis is
transformed inside, one line of cells at a time; the result is a matrix-valued function of the
other axis, which is transformed outside. With `T_u` / `T_v` the Toeplitz matrix along each
axis:

| slot | `li_nesting='fixed'` (default) | `li_nesting='symmetric'` |
| --- | --- | --- |
| `epx` (acts on `E_x`) | `( T_u( [T_v(eps)]^-1 ) )^-1` | `T_v( [T_u(1/eps)]^-1 )` |
| `epy` (acts on `E_y`) | `T_u( [T_v(1/eps)]^-1 )` | the same |
| `epz_i` | `( T_uv(eps) )^-1` | the same |

`'fixed'` keeps `v` inside for both slots. It is what the raster path built until this date,
the order the Cartesian shape Li and tensor Li builds were written in (tensor Li builds nothing
else), and RETICOLO's `parm.res1.li=1`. `'symmetric'` puts each slot's own inverse-rule axis inside —
Li's construction, RETICOLO's `li=0` — so `epx` is the x<->y mirror image of `epy`. On a staircase
both are Li's rules, exact for the structure they are given, and they converge to the same
answer. They differ in how they truncate. On a y-invariant layer they coincide, both `epx`
reducing to Li's 1D inverse rule (`tests/rcwa/test_li_nesting.py::test_a_y_invariant_layer_reduces_to_the_1d_rules_in_both_nestings`).

What *is* intrinsic to Li's rules, in either nesting, is that they are exact only where the
boundaries run along the lattice axes. A curved or tilted boundary is a staircase to them —
Liu & Fan give that as the reason S4 ships vector-field formulations — and that is why the NVF
path exists. The relabelling asymmetry is a separate defect, and it belongs to the fixed nesting.

**The symmetry.** Take a structure with `eps(x, y) = eps(y, x)` in a square cell at normal
incidence. Rotating the incident polarisation by 90 degrees maps the problem onto itself, so the
diffraction map must come back transposed. Measured on the L-shape of
`tests/rcwa/test_factorization_core.py` (transpose-symmetric, deliberately not 4-fold) at 96x96,
`max|T(psi=0) - T(psi=90)ᵀ|`:

| truncation_order | `li`, `'fixed'` | `li`, `'symmetric'` | `nvf` |
| --- | --- | --- | --- |
| 4 | 4.2e-3 | 2.8e-14 | 4.4e-14 |
| 8 | 2.9e-3 | 1.2e-13 | 4.0e-13 |
| 12 | 1.5e-3 | 9.7e-13 | 1.5e-13 |

The table this section carried before (5.3e-3, 1.4e-3, 5.9e-4 for `li`) predates the exact
sampled coefficients and the half-pixel phase, and is superseded. The NVF column is the control:
it takes the direct rule both ways, so no nesting is chosen at all.

The same holds with every moving part live: the chiral L of that file at oblique incidence, with
unequal periods and unequal truncation, turned a quarter (both spellings — transpose with
`psi -> -psi`, and a plain `np.rot90`). The symmetric nesting holds it to 1.2e-15 to 1.9e-14 at
`truncation_order` (5, 3), (8, 6), (11, 9); the fixed one misses by 1.1e-2, 1.4e-3 and 4.5e-4.

**Against RETICOLO**, which implements both nestings: an `eps = 6` rectangle, 200 x 150 nm in a
325 nm square cell, 600 nm thick, over `eps = 1.46**2` at 532 nm. Each nesting agrees with
RETICOLO V10's run of the same nesting to 4.2e-11 in total transmission — both polarizations,
`truncation_order` 3 and 5, both coefficient paths — while the two nestings disagree with each
other by 1.7e-5 (TE) to 1.7e-3 (TM). Turn the rectangle a quarter and the symmetric nesting's TE
is its old TM to round-off. The fixed nesting misses it by 1.4e-3 at truncation 3 and 7.1e-4
at 5 (`tests/references/test_reticolo_li_nesting.py`).

**Where it shows most: metals.** Measured in an earlier session (2026-09), and not pinned by a
test here. The scenes are given in full so that they can be re-run.

- *Au nanohole array.* Square period 500 nm, a 200 nm air hole in 150 nm of Au (Johnson–Christy
  at 700 nm, `eps ≈ -16.49 + 1.06i`), glass substrate `eps = 2.25`, normal incidence, raster
  N = 512. The structure is C4-symmetric, so the two polarizations must transmit the same.
  The fixed nesting gives T(x-pol) = 0.0172 and T(y-pol) = 0.0728 at `truncation_order` 5, and
  still 0.0239 against 0.0427 at 15. The symmetric nesting gives one number for both
  polarizations, which oscillates with truncation: 0.0428 at 13, 0.0488 at 15. For scale, fmmax's
  Jones-direct formulation gives 0.0432 (1605 terms), and wavebit's NVF on the parametric
  circle 0.0420 at 15.

  Re-run with the rewritten builders on 2026-09-25, with the hole centred and a pixel counted
  as air when its centre lies inside. The fixed nesting reproduces all four of its numbers
  to the digits quoted: 0.0172 and 0.0728 at 5, 0.023885 and 0.042708 at 15. The symmetric
  nesting gives 0.0725 for both polarizations at 5, 0.042744 at 13 and 0.048913 at 15.
  That is 5.6e-5 and 1.1e-4 from the earlier session's figures, whose construction was not
  recorded; the builder here matches RETICOLO's own `li=0` to 4.2e-11 on the rectangle
  above. Raster NVF gives 0.0597 for both at 5.
- *Au nanodisk array.* Period 450 nm, disk D = 140 nm, h = 50 nm, in an `n = 1.5` environment.
  At `truncation_order` 10 the fixed nesting puts the plasmon resonance about 25 nm apart for
  the two polarizations (near 795 nm and 770 nm). NVF puts it near 780 nm for both.

The symmetric nesting removes the polarization split, which is an artefact. It does not remove
the staircase: on these curved metal boundaries its answer still moves with truncation, and NVF
is the formulation built for them.

**What the default would cost to change.** The fixed nesting stays the default, so no golden moves.
Fixed against symmetric on the scenes the suite pins (2026-09-25, the larger of the per-order
`|ΔR|` and `|ΔT|`, and the change in total transmission; measured on the code of 2026-09-11,
before the layers were stacked bottom-to-top -- the `2d_2layer` rows are the same physical
scenes either way):

| scene | per order | total T |
| --- | --- | --- |
| `golden_legacy.npz` `2d_normal` (random 5x5, truncation 2) | 2.0e-1 | 1.5e-1 |
| `golden_legacy.npz` `2d_oblique` | 5.7e-2 | 2.9e-2 |
| `golden_legacy.npz` `2d_2layer` | 8.7e-2 | 2.0e-1 |
| `golden_legacy.npz` `2d_cfs` (integrated) | 5.1e-2 | 4.5e-2 |
| `golden_field.npz` `2d`, `2d_2layer`, `2d_first_order` | 5.7e-2 to 8.7e-2 | 2.5e-3 to 9.5e-2 |
| the L-shape at the test's own 48x48, truncation 4 and 6 | 4.8e-3 to 1.2e-2 | 4.8e-3 to 1.2e-2 |
| the chiral L, truncation (5, 3) to (11, 9) | 4.3e-4 to 1.6e-3 | 1.9e-4 to 2.1e-3 |
| `test_optimization_smoke.py`'s permittivity ramp | 2.5e-5 | 3.9e-5 |

Energy conserves to 6.8e-14 or better in both nestings on every one of them. The large numbers
are random rasters at truncation 2, where every factorization choice is a large one. They
converge: on `2d_normal` the gap in total T is 1.5e-1, 1.4e-2, 6.3e-4, 5.1e-4, 4.4e-4 and 2.6e-4 at
truncation 2, 4, 8, 12, 16 and 20. Flipping the default therefore moves every 2D `li` golden
in `golden_legacy.npz` and `golden_field.npz` far past their 1e-12, and the legacy file cannot
be re-frozen in place — `tests/rcwa/test_golden_legacy_parity.py` explains the retire-and-refreeze
route. Nothing in the 1D goldens moves: on the twelve 1D RETICOLO cases the two nestings return
the same efficiencies to the last bit. Three pins added since those
measurements are tied to the default too. `tests/rcwa/test_tensor_li.py::test_scalar_tensor_is_exactly_the_scalar_li_operator`
holds tensor Li on a scalar tensor to the scalar Li operator at 3e-12, and tensor Li builds the
fixed nesting only, so a flip breaks it unless tensor Li gains the symmetric one.
`tests/rcwa/test_rectilinear_li.py` compares the Cartesian shape Li with a painted raster through
the default on both sides, so it would still pass while every shape Li number moved -- among
them the explicit-Li reference (`li_T`) of `studies/rcwa/nvf_seam_contrast/nvf_seam_contrast.py` and
`studies/rcwa/nvf_seam_contrast/records/nvf_seam_contrast.log`.
`tests/references/test_tensor_li_golden.py` does not move: tensor Li does not read the nesting.
The case for flipping is the covariance, and agreement with RETICOLO's own `li=0`. The case
against is that every 2D `li` number measured so far was taken on the fixed nesting. That is the
maintainer's call.

Two things follow for anyone reading a `li` result:

- The residue is *not* energy error. Both nestings conserve to 1e-13, so no conservation check
  will surface it. It shows up as a difference in **total** transmitted power between two
  polarisations that ought to agree.
- The suite pins it in both directions. `test_nvf_does_not_tell_x_and_y_apart` and
  `test_li_does_not_tell_x_and_y_apart_in_the_symmetric_nesting` are the invariant.
  `test_li_tells_x_and_y_apart_and_this_is_expected` records the fixed nesting's cost, bounded
  above *and* below, so a regression that worsened it fails, and so does one that quietly fixed
  it and left this section stale. `test_li_is_covariant_when_the_whole_scene_is_turned_in_the_symmetric_nesting`
  holds the turned scene. All four are in `tests/rcwa/test_factorization_core.py`. The
  matrix-level statement — `epx` of the transposed layer is `epy` relabelled — is
  `tests/rcwa/test_li_nesting.py::test_the_symmetric_nesting_is_covariant_under_relabelling_x_and_y`,
  and the RETICOLO comparison is `tests/references/test_reticolo_li_nesting.py`.

**Where the code came from.** Until 2026-09-25 the raster builders here were ports of meent's
`dfs2d` and `cfs2d`, and meent's two-dimensional Li-rule code was written with reference to the
MATLAB source of RETICOLO V9 (J.P. Hugonin and P. Lalanne,
[arXiv:2101.00901](https://arxiv.org/abs/2101.00901)), following its `ret2li` in `retcouche.m`,
`retf.m` and `rettoeplitz.m`. The builders of this section were then rewritten from Li's papers
by an implementer who did not consult RETICOLO's or meent's source or the bodies they replaced,
and `compression_indices` / `compress_cell` were reimplemented from a written specification of
what they return, by an implementer who had not seen the earlier code (the specification's
author had seen the earlier functions and `ret2li`). `LICENSE` records how, and gives the
RETICOLO attribution (CC BY 4.0, from the V10 record) for the earlier versions this repository's
history still holds. The two Li builds added beside them
before the rewrite landed run on it too: the Cartesian shape Li (`rectilinear.py`) calls
`fourier2d_integrated`, and tensor Li's row transform (`tensor_li.py`), which called
`_fourier1d_integrated`, `_wrap` and `_circulant_gather` directly, was moved onto
`_raster_series` / `_segment_series` when the rewrite was rebased onto it. The fixed nesting stayed the default
because every golden here was computed with it, and it is also RETICOLO's default. The rewrite
reproduces the previous builders to round-off, and is exact on a uniform layer, where the old
sampled path left 1.3e-16 off the diagonal on grids that are not a power of two.

## 7. One more word that was doing two jobs: the grid builder

`dfs2d(..., piecewise_constant=False)` — later `_fourier2d_grid`, gone since the 2026-09-25
rewrite (§6) — carried the same defect as `laurent_conv_2d`, one level
down. The flag does not express a preference — it states **what the grid holds**, and there are
exactly two answers with exactly two consumers:

| Entry point | Grid holds | Callers |
| --- | --- | --- |
| `fourier2d_piecewise_constant` | values that hold across their pixel — `eps`, `1/eps` | 5 in `wavebit/` |
| `fourier2d_pointwise` | point samples of a continuous field — the NVF normal field | 2 in `wavebit/` |

The counts drift as sites are added and are re-checked rather than trusted; they were 5 and 3
when this was written, one site moved from pointwise to piecewise on 2026-08-17 (`ee715df`,
the PN z slot), they were 7 and 5 when last counted before 2026-09-25, when the Li builders
stopped going through the one-slot entry point (§6), and tensor Li's `zz` slot (2026-09-24) is
one of today's five. What does not drift is the split itself.

A value the caller can never be in doubt about should not be a parameter the caller restates,
and the default made it worse: at the time of the change `False` was the answer 3 of 8 sites
wanted, so the *majority* case was the one you got by forgetting. `_fourier2d_grid` was then made private with the flag keyword-only and
undefaulted, so even a direct call had to say which it meant. Since 2026-09-25 there is no flag
left to pass: each entry point hands `_raster_series` the correction its data needs —
`_pixel_correction` for a staircase, `_half_pixel_phase` for point samples — and that choice is
the whole difference between them.

`fourier2d_pointwise` also drops the rule argument rather than defaulting it. Every call in the
repository asked for the direct rule — not by accident: the inverse rule closes products of
*concurrently discontinuous* factors, and a unit-vector field that is continuous by
construction has no discontinuity for it to close. It was not a choice anyone had, only a
way to be wrong.

The same reasoning reached the rule argument itself on 2026-08-12. It was `conti_x` /
`conti_y`, a pair of integers flagging each direction *continuous* (`1`) or not (`0`) —
four states for three meanings, since `(0, 0)` is not part of Li's scheme as used here and
appeared nowhere, yet nothing refused it. The names also ran backwards: `conti_y=0` meant
"y takes the inverse rule", so the flag that was *set* said what was not true of the axis
it named. One `inverse_rule` of `None` / `'u'` / `'v'` says the same thing in the
direction the reader is thinking in, and makes the fourth state unrepresentable rather
than merely unused.

Why this was worth doing rather than documenting: the two mistakes are not symmetric, and the
dangerous one is silent. Feeding a permittivity raster to the sampled build costs 5.7e-4 per
order on a comfortable grid (N=64, truncation_order=4), 4.9e-3 on a tight one (N=24, truncation_order=10), and below
roughly `N < 2*truncation_order` the inverse-rule matrix goes singular and the solve raises `LinAlgError`
(measured: N=16, truncation_order=8). Energy conserves to 1e-14 through all of it, so nothing downstream
reports anything. The reverse mistake — the sampled field built as a staircase — is 1e-5 at
normal incidence into an index-matched substrate and 2e-4 to 2e-3 with contrast or at conical
incidence, and never structural.

`tests/rcwa/test_nvf_normal_field.py::test_the_projector_is_built_from_the_sampled_transform` pins
the choice at the one production site that makes it. It needs two halves, because the
projector identity `C_xx + C_yy = I` that the rest of that file checks holds under *either*
build: one assertion that the sampled build is what runs, and one that the two builds are far
enough apart (1.1e-2 here) for the first to be pinning anything.

The convention probe in `tests/rcwa/test_parametric.py` calls `fourier2d_pointwise` for a third reason: it
compares against a raw `np.fft.fft2` and wants no pixel correction in the way. The name still
fits — asking for the uncorrected build *is* declaring the input to be samples — but the intent
there is "show me the plain DFT", not "this is a measured field".

## 8. `sampled` and `integrated`, all the way down

`DFS` and `CFS` used to be the shorthand for the two ways a layer's Fourier coefficients are
obtained, and `sampled` / `integrated` the public spelling of the same choice. Two vocabularies
for one thing, and the internal one collided with the module's most load-bearing word:
**continuous**. Li's rules are decided by which field component is *continuous* across an
interface, so `CFS = continuous Fourier series` put that word on a method while the physics next
to it used it for a property of a field. `conti_x` / `conti_y` had already taken it for a third
meaning before they were removed.

The public words won, everywhere:

| was | is |
| --- | --- |
| `cfs2d`, `_cfs` | `fourier2d_integrated`, `_fourier1d_integrated` (since 2026-09-25 `_segment_series`) |
| `to_conv_mat_raster_discrete` | `to_conv_mat_raster_sampled` |
| `to_conv_mat_raster_continuous`, `..._continuous_nvf` | `..._integrated`, `..._integrated_nvf` |
| `nvf_conv_matrices_continuous` | `nvf_conv_matrices_integrated` |
| `DFS` / `CFS` in prose, tables and study mode names | `sampled` / `integrated` |

`NVF` stays. It abbreviates *normal vector field*, which is a thing rather than a method, and it
sits on the other axis: `sampled` and `integrated` are the two values of `fourier_coefficients`,
while `nvf` is one of the two values of `factorization`. The four combinations are all real
builders, which is why `to_conv_mat_raster_integrated_nvf` carries both words.

One caution for anyone repeating this. `DFS` and `CFS` were nouns; `sampled` and `integrated`
are adjectives. Substituting them one for one produces "the sampled/sampled path", "match the
legacy integrated", "Raster with sampled" — all of which a blind pass here produced before they
were rewritten as noun phrases. The rule that works is *the sampled path*, *integrated
coefficients*, *the integrated build*.

**One thing this vocabulary does not tell you, and should.** `fourier_coefficients` governs the
*permittivity* only — never the normal field. But the reason differs by path, and only one of
the two is the one you would guess.

On the **raster** path both values are accepted and the normal field is unaffected:
`_nvf_projector` calls `fourier2d_pointwise` from `nvf_conv_matrices` and from
`nvf_conv_matrices_integrated` alike, because a sampled field has no closed form to integrate.
So `('integrated', 'nvf')` integrates `eps` and `1/eps` exactly and still samples the
projector. Not a gap in the implementation — it is what the normal field is.

On the **shape** path (parametric layer dicts) the question never arises:
`fourier_coefficients='sampled'` is refused outright with `NotImplementedError` — parametric
geometry has no samples to take a discrete series of — so `'integrated'` is the only value
there.

The two paths also differ in where the field's grid comes from, which is easy to assume away.
Counting calls on a 32x32 raster and a circular shape layer:

| path | `lattice.sample_grid` calls |
| --- | --- |
| raster, `'sampled'` | 0 |
| raster, `'integrated'` | 0 |
| shapes (analytic dicts) | 1 |

`sample_grid` serves the non-raster builders only; the raster normal field is index-based and
goes through `factorization.normal_vector_field` without ever asking for cell coordinates.

## 9. Laurent's rule as an option

Added 2026-09-27. `factorization='laurent'` closes **every** product of the permittivity with a
field component with the direct rule -- the plain convolution matrix `[[eps]]`, whatever is
continuous across the interfaces. Nothing resolves to it: a raster still defaults to Li and a
shape to NVF. It is offered because other tools use it -- fmmax's `Formulation.FFT`, and the
formulation RCWA started from -- and a like-for-like comparison with them needs the same rule on
this side.

### The slots

With `[[f]]` the BTTB convolution matrix of §3, direct rule both ways:

| slot | `'laurent'` | `'li'`, `li_nesting='fixed'`, for comparison |
| --- | --- | --- |
| `epx` (acts on `E_x`) | `[[eps]]` | `( T_u( [T_v(eps)]^-1 ) )^-1` |
| `epy` (acts on `E_y`) | `[[eps]]` | `T_u( [T_v(1/eps)]^-1 )` |
| `epz_i` | `[[eps]]^-1` | `[[eps]]^-1`, the same |
| `epxy`, `epyx` | `None` | `None` |

A magnetic layer takes the same rule by the `(E, H, eps, mu) -> (H, -E, mu, eps)` duality:
`[[mu]]` in both in-plane slots, `[[mu]]^-1` in the z slot. Both in-plane slots are *one*
matrix, built once, so they are equal to the bit.

**The z slot is where the two rules agree, and it is `[[eps]]^-1`, not `[[1/eps]]`.** The solve
needs `E_z` from `D_z`. `E_z` is tangential to every vertical interface and continuous, so
`D_z = eps E_z` is a product the direct rule closes correctly -- Li's rules take it too -- and
inverting it gives `[[eps]]^-1`. `[[1/eps]]` would be the direct rule applied to
`E_z = (1/eps) D_z`, whose two factors jump together: measurably another matrix (more than 1e-2
apart on the crossed grating of `tests/rcwa/test_laurent_factorization.py`), and not the one any of
the sources below uses. Moharam and Gaylord's formulation inverts the Toeplitz matrix of the
permittivity there, and so does fmmax's eigenproblem, which solves against its
`z_permittivity_matrix`; its separate `[[1/eps]]` (`inverse_z_permittivity_matrix`) is kept
for field reconstruction and does not enter the eigenproblem or the scattering matrix.

What that leaves between the rules is the in-plane pair. On a grating that varies along x only,
TE reads `epy` alone, where both rules build `[[eps]]`: the answers are one to round-off at every
order. TM reads `epx`, where Li builds `[[1/eps]]^-1` and Laurent `[[eps]]` -- the slowly
converging case.

Sources: M. G. Moharam and T. K. Gaylord, JOSA **71**, 811 (1981), the formulation;
M. G. Moharam, E. B. Grann, D. A. Pommet and T. K. Gaylord, JOSA A **12**, 1068 (1995), which
states it for binary gratings in both polarizations; L. Li, JOSA A **13**, 1870 (1996), for why
it converges slowly -- where a field component that jumps at an interface meets a permittivity
that jumps at the same place, the product of the truncated series does not converge to the
series of the product. The code cites them at `laurent_conv_matrices`.

### Where it is built, and where it is refused

| input | `'laurent'` | why |
| --- | --- | --- |
| scalar raster, `'sampled'` | built | the exact staircase coefficients; decides nothing, so it traces without a frozen geometry |
| scalar raster, `'integrated'` | built | the same block compression Li's integrated build records |
| shapes, disjoint | built | the closed-form `[[eps]]` NVF starts from, with the normal field dropped |
| shapes, painted overlap | built | the resolved polygons' `[[eps]]`; curved primitives are polygonised at 256 vertices, as for NVF |
| a skewed cell | built | the rule names no axis -- one scalar convolution in both slots -- where Li is refused |
| 1D TE / TM, conical | built | TE equals Li to round-off; TM is Li (1996)'s slow case |
| `formulation='first_order'` | built | the same five slots; second-order and first-order agree to 9e-15 |
| tensor materials | **refused by name** | tensor Li and Popov-Neviere NVF are the tensor builders; a Laurent tensor build is not written and has no reference checked against it. A uniform tensor is the same under every rule, so leaving `factorization` unset loses nothing there |
| `li_nesting`, `raster_normal_method`, `nvf_projector`, `shape_nvf_grid_size` | **refused by name** | Laurent nests no rules and builds no normal field, so nothing reads them |

**A frozen geometry is stamped with its rule.** `StaticGeometry.factorization` records the rule a
record was frozen under, and a replay under another rule is refused by name, eagerly and under
`jax.jit`. A Laurent record of shapes plans the painting and no interface set, which the NVF
replay would read as "plan blind"; a Li record of an integrated raster holds no normal field, which
the NVF build would rebuild eagerly or, under a trace, could not build at all. Records built by
hand carry no stamp and replay on trust.

### How it converges

`studies/rcwa/laurent_convergence/laurent_convergence.py`, logged in `studies/rcwa/laurent_convergence/records/laurent_convergence.log`. Wavelength
900 nm, thickness 250 nm, top 1, bottom 2.25, eps 12.25 in air.

One-dimensional binary grating, 700 nm period, 50% duty, theta 10 degrees; `|T - T_ref|`,
the reference Li at order 400:

| order | TE, `li` | TE, `laurent` | TM, `li` | TM, `laurent` |
| --- | --- | --- | --- | --- |
| 5 | 1.15e-3 | 1.15e-3 | 3.43e-3 | 1.80e-2 |
| 10 | 3.11e-4 | 3.11e-4 | 4.90e-4 | 7.93e-3 |
| 20 | 4.11e-5 | 4.11e-5 | 7.57e-5 | 3.78e-3 |
| 40 | 5.22e-6 | 5.22e-6 | 8.26e-6 | 1.83e-3 |
| 80 | 6.52e-7 | 6.52e-7 | 4.93e-7 | 9.00e-4 |

TE is Li's answer at every order. TM under Laurent halves per doubling of the orders -- first
order in the truncation -- while Li falls sevenfold to seventeenfold per doubling. That is Li's
1996 result, reproduced.

A 350 x 245 nm rod in a 700 nm square cell, drawn on the pixels of a 40 x 40 raster so every
rule sees the same staircase, theta 10 and phi 20 degrees; `|T - T_ref|` per polarization, the
reference Li at order 18 (where Li is the exact rule for an axis-aligned rectangle):

| order | `li` s | `li` p | `nvf` s | `nvf` p | `laurent` s | `laurent` p |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 3.3e-4 | 1.0e-3 | 3.7e-4 | 3.6e-4 | 7.6e-4 | 1.1e-2 |
| 8 | 8.5e-5 | 2.4e-6 | 1.2e-4 | 6.4e-4 | 2.3e-4 | 6.6e-3 |
| 12 | 1.9e-5 | 1.4e-5 | 3.3e-5 | 2.9e-4 | 3.2e-4 | 4.7e-3 |
| 14 | 1.3e-5 | 2.7e-5 | 2.6e-5 | 2.7e-4 | 3.1e-4 | 4.1e-3 |

Laurent converges -- and conserves energy on every one of these solves, so the slow
convergence is not a loss (the log records the largest `|R + T - 1|`) -- but at order 14 it is
still 150 times further from the answer than Li in p. On a 400 nm disc on the shape route,
where there is no Li to compare with, Laurent sits below NVF by 7.4e-2 / 5.4e-2 (s / p) at order
4 and 2.0e-2 / 1.3e-2 at order 12, the gap roughly halving per doubling of the orders, as in the
TM column above.

**What it saves is nothing measurable.** At order 12 one solve of the rod takes 2.68 s under Li,
2.74 s under NVF and 2.72 s under Laurent, and of the disc 2.71 s under NVF and 2.70 s under
Laurent (medians of three, on a machine shared with other work): the eigensolve dominates, and
the normal field Laurent skips is under 1% of it. Laurent is a rule to compare with, not a
faster one.

### Against fmmax

`tests/references/test_fmmax_laurent_golden.py`. fmmax's `Formulation.FFT` assembles the same
operator, so the comparison is one of identity once the coefficients are made to agree: fmmax's
DFT is the midpoint rule for the Fourier integral Wavebit evaluates exactly on each pixel, and
the factor between them is `y / sin(y)` per axis, even in the samples per pixel `R`, so a
Richardson extrapolation over four samplings removes it. On a silicon-like rod in a 700 x 560
cell, a lossy metal film with a hole, and a cross on a hexagonal cell -- where Li is refused --
Wavebit's `'laurent'` agrees with the extrapolation to 6.1e-13, 3.2e-11 and 4.8e-13 in total,
and to the same order by order over four or five propagating orders; the mirrored order index
misses by 1.3e-2 to 7.4e-2, and Li or NVF by 4e-2 to 2e-1.

That is against fmmax with its pixel-centring phase in double precision. fmmax 1.7.1 computes
that phase from a single-precision frequency grid (`fmmax.fft._fft2`), which turns an exact
translation into an order-dependent one and leaves 1e-8 to 1e-7 on these efficiencies; the
generator freezes the stock run too, and the test holds Wavebit to it within twice the gap
between the two runs. A comparison with fmmax's FFT formulation that expects better than about
1e-7 on a patterned layer has to account for that phase first.
