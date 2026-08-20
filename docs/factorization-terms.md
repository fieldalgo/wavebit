# Laurent, direct, inverse, Toeplitz: which word names a rule and which names a matrix

Written: 2026-08-12. Structure claims below are measured, not asserted; the check is quoted in
§3 and reproduces in seconds.

Targets:
- `wavebit/_core/analytic.py::conv_matrix_2d` — the builder the collision used to be in; it
  was called `laurent_conv_2d` until this document was written
- `wavebit/_core/analytic.py::_grid`, `wavebit/_core/analytic.py::assemble_eps_coeffs` —
  the doubled reciprocal grid the builder consumes, and why it is doubled
- `wavebit/_core/factorization.py::fourier2d_piecewise_constant` — the raster twin, and where
  `inverse_rule` chooses the rule
- `tests/test_analytic.py` — `test_conv_matrix_matches_grid_convention`, which pins the ordering
  the two builders share

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

`fourier2d_piecewise_constant`'s `inverse_rule` is the switch: `'x'` or `'y'` names the
direction whose discontinuity the rule closes, `None` asks for the direct rule both ways.
Nothing else selects it, and `fourier2d_pointwise` does not offer it at all — a field that is
continuous by construction has no discontinuity for the inverse rule to close.

## 2. What `conv_matrix_2d` actually is

It takes a grid of Fourier coefficients and lays them out as a convolution matrix:

```
M[(r,s),(c,d)] = coeff(ky = r - c, kx = s - d)
```

That is all it does. It never sees a permittivity, and it does not know which rule its caller
is building. In `to_conv_mat_smooth_nvf` it is called five times in a row — on `eps`, on
`1/eps`, and on the three normal-field products `nx^2`, `nx ny`, `ny^2`:

```python
ep_dir = conv_matrix_2d(xp, eps_c,     max_order_x, max_order_y)
ep_inv = conv_matrix_2d(xp, inv_eps_c, max_order_x, max_order_y)
...
delta_ep = ep_dir - xp.linalg.inv(ep_inv)
```

The inverse rule's *inversion* is the caller's, on the line after. The builder is shared.

This is what the old name got wrong: it carried a rule word into a place where no rule has been
chosen yet, so `ep_inv = laurent_conv_2d(...)` read as a contradiction without being one.

The index differences `r - c` and `s - d` each span `-2*max_order .. +2*max_order`, which is why the
coefficient grid arrives on a **doubled** reciprocal range — see `_grid`. A grid of only
`2*max_order+1` per axis cannot fill the corners of the matrix.

## 3. The matrix is BTTB, not Toeplitz

A Toeplitz matrix is constant along its diagonals: `T[i,j]` depends on `i - j` alone. The
matrix here does not satisfy that. Flattening `i = r*n_orders_x + s` and `j = c*n_orders_x + d`,

```
i - j = (r - c) * n_orders_x + (s - d)
```

and `s - d` ranges over `-(n_orders_x-1) .. (n_orders_x-1)`, which is wider than `n_orders_x`. So `(r-c, s-d)` is
not recoverable from `i - j`: the pairs `(k, m)` and `(k+1, m - n_orders_x)` land on the same
diagonal carrying different values.

Checked directly on `max_order_x, max_order_y = 2, 3` (a 35x35 matrix, random complex coefficients):

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

## 5. The rename, and what did not move

`laurent_conv_2d` became `conv_matrix_2d`. Naming it for the structure was considered and
rejected: `toeplitz_` is false, `block_toeplitz_` under-specified, `bttb_` unreadable. What is
left is naming it for what it produces — a convolution matrix — which is the vocabulary the
rest of the code already uses (`nvf_conv_matrices`, `to_conv_mat_smooth_nvf`, `epx_conv`).

Done before the first release, while `_core` is private and nothing outside it imports the
name, so no alias was kept and none is owed.

| Moved | Count | Where |
| --- | --- | --- |
| `laurent_conv_2d` → `conv_matrix_2d` | 14 | `_core/analytic.py`, `tests/test_analytic.py`, `analytic-modeling.md` |
| `test_laurent_matches_dfs2d_convention` → `test_conv_matrix_matches_grid_convention` | 4 | `tests/test_analytic.py`, `_core/analytic.py`, `analytic-modeling.md` |
| "Laurent matrix" meaning the structure → "convolution matrix" | 2 | `_core/analytic.py` |

**`Laurent` meaning the rule stayed, in all 15 places.** It is correct there, and one of them
is not prose at all: `modes=('nvf', 'laurent')` in
`studies/benchmark_truncation_shape.py` is a value a user types. A rename that had swept the
word out of the repository would have broken that and fixed nothing.

## 6. The nested rule is direction-dependent, and nothing tests it

Applying the rules in 2D means nesting them: one axis is transformed inside, the other
outside. The shared core fixes that order — inner y, outer x — which is Li's own construction and is
correct. But it means the builder is **not invariant under relabelling x and y**, and that is
observable in the answer.

Take a structure with `eps(x, y) = eps(y, x)` in a square cell at normal incidence. Rotating
the incident polarisation by 90 degrees maps the problem onto itself, so the diffraction map
must come back transposed, exactly. Measured on an L-shape (transpose-symmetric, deliberately
not 4-fold), 96x96 raster, `max|T(psi=0) - T(psi=90)ᵀ|`:

| max_order | `factorization='li'` | `factorization='nvf'` |
| --- | --- | --- |
| 4 | 5.3e-3 | 2.2e-14 |
| 8 | 1.4e-3 | 3.5e-13 |
| 12 | 5.9e-4 | 5.1e-13 |

The NVF column is the control: it asks for the direct rule both ways, so no nesting order is
chosen and
the symmetry holds to machine precision. On a high-contrast random raster the `li` residue
reaches 1e-1.

**This is not a defect in the implementation.** It is intrinsic to Li's crossed-grating
scheme, and Liu & Fan name it in the S4 paper as the reason they ship vector-field
formulations instead — an effective staircasing that comes from taking 1D transforms along the
principal lattice directions. It is why the NVF path exists.

Two things follow for anyone reading a `li` result:

- The residue is *not* energy error. Total power conserves to 1e-14 in both arms, so no
  conservation check will surface it. In the table above the whole residue shows up as a
  difference in **total** transmitted power between two polarisations that ought to agree.
- The suite pins it since 2026-08-12, in both directions.
  `test_nvf_does_not_tell_x_and_y_apart` is the invariant — NVF chooses no nesting order, so
  the symmetry is exact — and `test_li_tells_x_and_y_apart_and_this_is_expected` records the
  cost of not having it, bounded above *and* below: a regression that worsened the nesting
  would otherwise pass, and so would one that accidentally fixed it and left this section
  stale. Both are in `tests/test_factorization_core.py` and cost about a second.

## 7. One more word that was doing two jobs: the grid builder

`dfs2d(..., piecewise_constant=False)` — now `_fourier2d_grid` — carried the same defect
as `laurent_conv_2d`, one level
down. The flag does not express a preference — it states **what the grid holds**, and there are
exactly two answers with exactly two consumers:

| Entry point | Grid holds | Callers |
| --- | --- | --- |
| `fourier2d_piecewise_constant` | values that hold across their pixel — `eps`, `1/eps` | 7 in `wavebit/` |
| `fourier2d_pointwise` | point samples of a continuous field — the NVF normal field | 5 in `wavebit/` |

The counts drift as sites are added and are re-checked rather than trusted; they were 5 and 3
when this was written, and one site moved from pointwise to piecewise on 2026-08-17 (`ee715df`,
the PN z slot). What does not drift is the split itself.

A value the caller can never be in doubt about should not be a parameter the caller restates,
and the default made it worse: at the time of the change `False` was the answer 3 of 8 sites
wanted, so the *majority* case was the one you got by forgetting. `_fourier2d_grid` is now private with the flag keyword-only and
undefaulted, so even a direct call has to say which it means.

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
it named. One `inverse_rule` of `None` / `'x'` / `'y'` says the same thing in the
direction the reader is thinking in, and makes the fourth state unrepresentable rather
than merely unused.

Why this was worth doing rather than documenting: the two mistakes are not symmetric, and the
dangerous one is silent. Feeding a permittivity raster to the sampled build costs 5.7e-4 per
order on a comfortable grid (N=64, max_order=4), 4.9e-3 on a tight one (N=24, max_order=10), and below
roughly `N < 2*max_order` the inverse-rule matrix goes singular and the solve raises `LinAlgError`
(measured: N=16, max_order=8). Energy conserves to 1e-14 through all of it, so nothing downstream
reports anything. The reverse mistake — the sampled field built as a staircase — is 1e-5 at
normal incidence into an index-matched substrate and 2e-4 to 2e-3 with contrast or at conical
incidence, and never structural.

`tests/test_nvf_normal_field.py::test_the_projector_is_built_from_the_sampled_transform` pins
the choice at the one production site that makes it. It needs two halves, because the
projector identity `C_xx + C_yy = I` that the rest of that file checks holds under *either*
build: one assertion that the sampled build is what runs, and one that the two builds are far
enough apart (1.1e-2 here) for the first to be pinning anything.

The convention probe in `tests/test_analytic.py` calls `fourier2d_pointwise` for a third reason: it
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
| `cfs2d`, `_cfs` | `fourier2d_integrated`, `_fourier1d_integrated` |
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

On the **shape** path (analytic layer dicts) the question never arises:
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
