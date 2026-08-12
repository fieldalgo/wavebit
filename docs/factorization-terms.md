# Laurent, direct, inverse, Toeplitz: which word names a rule and which names a matrix

Written: 2026-08-12. Structure claims below are measured, not asserted; the check is quoted in
§3 and reproduces in seconds.

Targets:
- `fieldwave/_core/analytic.py::laurent_conv_2d` — the builder whose name is the collision
- `fieldwave/_core/analytic.py::_grid`, `fieldwave/_core/analytic.py::assemble_eps_coeffs` —
  the doubled reciprocal grid the builder consumes, and why it is doubled
- `fieldwave/_core/factorization.py::dfs2d` — the raster twin, and where `conti_x` / `conti_y`
  choose the rule
- `tests/test_analytic.py` — `test_laurent_matches_dfs2d_convention`, which pins the ordering
  the two builders share

---

## TL;DR

Four words are in play and they belong to two different axes. **Laurent** and **direct** name
the same *factorization rule*; **inverse** (Li) names the other one. **Toeplitz** names a
*matrix shape* and no rule at all. The one function whose name mixes the axes is
`laurent_conv_2d`, which builds the matrix for both rules.

Two things are worth knowing before editing any of this:

1. `laurent_conv_2d` is **rule-neutral**. It is called on `eps` and on `1/eps` two lines apart.
   Reading its name as "the direct-rule builder" is wrong, and the name invites exactly that.
2. The matrix it returns is **not Toeplitz**. It is BTTB — block-Toeplitz with Toeplitz blocks
   — and the distinction is checkable, not pedantic (§3).

## 1. The two rules

Multiplication in real space is convolution in Fourier space, so `D = eps E` becomes a matrix
acting on the coefficient vector of `E`. Which matrix depends on whether the two factors are
discontinuous at the same place.

| Rule | Also called | Matrix used | When it is the correct one |
| --- | --- | --- | --- |
| direct | Laurent, "plain Laurent" | `<eps>` | the factors are not concurrently discontinuous |
| inverse | Li's rule | `<1/eps>^-1` | they are — along that direction |

Both names for the first rule are in use in this repository and both are correct.
`docs/formulation.md` writes `Laurent (direct)`, `docs/analytic-modeling.md` writes
`Laurent (direct-rule)`, and `_core/factorization.py` writes the pair as `direct` / `inverse`
in code (`ep_dir`, `ep_inv`). The gloss form — naming one and parenthesising the other — is the
one to copy in new prose, because a reader who knows only one name still lands.

`dfs2d`'s `conti_x` / `conti_y` are the switch: a direction flagged discontinuous (`0`) takes
the inverse rule. Nothing else selects it.

## 2. What `laurent_conv_2d` actually is

It takes a grid of Fourier coefficients and lays them out as a convolution matrix:

```
M[(r,s),(c,d)] = coeff(ky = r - c, kx = s - d)
```

That is all it does. It never sees a permittivity, and it does not know which rule its caller
is building. In `to_conv_mat_smooth_nvf` it is called five times in a row — on `eps`, on
`1/eps`, and on the three normal-field products `nx^2`, `nx ny`, `ny^2`:

```python
ep_dir = laurent_conv_2d(xp, eps_c,     fto_x, fto_y)
ep_inv = laurent_conv_2d(xp, inv_eps_c, fto_x, fto_y)
...
delta_ep = ep_dir - xp.linalg.inv(ep_inv)
```

The inverse rule's *inversion* is the caller's, on the line after. The builder is shared.

This is why the name misleads: it carries a rule word into a place where no rule has been
chosen yet. `ep_inv = laurent_conv_2d(...)` reads as a contradiction and is not one.

The index differences `r - c` and `s - d` each span `-2*fto .. +2*fto`, which is why the
coefficient grid arrives on a **doubled** reciprocal range — see `_grid`. A grid of only
`2*fto+1` per axis cannot fill the corners of the matrix.

## 3. The matrix is BTTB, not Toeplitz

A Toeplitz matrix is constant along its diagonals: `T[i,j]` depends on `i - j` alone. The
matrix here does not satisfy that. Flattening `i = r*ff_x + s` and `j = c*ff_x + d`,

```
i - j = (r - c) * ff_x + (s - d)
```

and `s - d` ranges over `-(ff_x-1) .. (ff_x-1)`, which is wider than `ff_x`. So `(r-c, s-d)` is
not recoverable from `i - j`: the pairs `(k, m)` and `(k+1, m - ff_x)` land on the same
diagonal carrying different values.

Checked directly on `fto_x, fto_y = 2, 3` (a 35x35 matrix, random complex coefficients):

| Property | Holds |
| --- | --- |
| Toeplitz in the flattened index — `M[i,j]` depends on `i-j` alone | **no** |
| block-Toeplitz — block `B[r,c]` depends on `r-c` alone | yes |
| every block is itself Toeplitz | yes |

The last two together are the definition of **BTTB**, block-Toeplitz with Toeplitz blocks.
`block-Toeplitz` alone is the usual shorthand and appears in `_core/lattice.py`,
`docs/formulation.md`, `docs/analytic-modeling.md` and `docs/oblique-lattice.md`; it is
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

## 5. The open item

`laurent_conv_2d` mixes the axes, and renaming it is the obvious repair. Naming it for the
structure does not work either: `toeplitz_` would be false, `block_toeplitz_` under-specified,
`bttb_` unreadable. What is left is naming it for what it produces — a convolution matrix —
which is also the vocabulary the rest of the code already uses (`nvf_conv_matrices`,
`to_conv_mat_smooth_nvf`, `epx_conv`).

Scope, if it is done: 14 occurrences of the identifier across `_core/analytic.py`,
`tests/test_analytic.py` and `docs/analytic-modeling.md`; 4 more for
`test_laurent_matches_dfs2d_convention`; 2 uses of "Laurent matrix" meaning the structure, in
`_core/analytic.py`. The 15 uses of `Laurent` meaning *the rule* stay — they are correct, and
one of them (`modes=('nvf', 'laurent')` in `studies/benchmark_truncation_shape.py`) is a value
a user types.

Not done here. This document exists so that the collision is legible whether or not the rename
happens.
