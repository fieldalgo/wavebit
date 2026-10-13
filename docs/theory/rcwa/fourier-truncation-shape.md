# Rectangular vs circular Fourier-order truncation — measurements

Written: 2026-08-09. Measured on CPU, NumPy backend, float64.

Targets:
- `src/wavebit/rcwa/_engine/solve_2d.py::solve_2d` — the tensor-product `(n_orders_v, n_orders_u)` order layout
- `src/wavebit/rcwa/_engine/factorization.py` — the truncation the convolution builds assume
- `studies/rcwa/benchmark_truncation_shape/benchmark_truncation_shape.py` — the harness this document reports

---

## TL;DR

S4 truncates the Fourier order set on a **disc** in reciprocal space; Wavebit keeps the full
**rectangle**. At the same maximum order the disc holds about `π/4` of the harmonics, and cost
scales as `N³`, so the disc looks like a free 2x.

It is not. Measured on the production NVF factorization, at equal accuracy the disc costs
**1.20x on a round pillar, 0.78x on an axis-aligned square, 1.01x on a 30°-rotated square** —
structure-dependent, within ±20%, and negative on a common case.

This agrees with the primary literature. Circular truncation was *introduced* by Li (1997),
whose abstract concludes the parallelogramic scheme "is superior". S4 and GRCWA default to
circular anyway.

**Decision: not implemented.** It would need a second indexing scheme through the solver, field
reconstruction and three backends, for an effect that is as often negative as positive.

## What was measured

One layer, pillar array on a square lattice, air above and below.

| | |
|---|---|
| λ / period / thickness | 900 nm / 1200×1200 nm / 300 nm |
| incidence | θ=20°, φ=30°, TM (psi=0) |
| pillars | circle, axis-aligned square, square rotated 30° — all at fill factor 0.4 |
| index | n=3.48 (and n=1.5 as a low-contrast control) |
| raster | 512×512 |
| propagating orders | 5: (0,0), (−1,0), (0,−1), (0,1), (−1,−1) |

Fill factor is held across the three shapes so that only the *spectrum* changes, not the
material volume. Error metric is the max deviation of a propagating order's diffraction
efficiency from a rectangular `truncation_order=20` (N=1681) reference.

## Why NVF is the only factorization this is measurable on

Everything `_nvf_assemble` consumes — `<eps>`, `<1/eps>`, `<nx²>`, `<nx·ny>`, `<ny²>` — is a
plain direct-rule matrix, i.e. an `f_hat[G_i − G_j]` lookup. So NVF assembles **natively on any
order set**. The circular results below are a real circular NVF.

The nested Li rule cannot do this: its 1D-then-2D inversion needs a complete rectangle. Taking
a sub-set of the box-built matrices leaks corner-order information the true restricted problem
would not have, and it flatters whatever sub-set is tested — measured that way, circular Li
looked *equal to* rectangular Li at 70% of the harmonics, which is the leak, not a result. The
harness refuses circular Li rather than reporting it.

Plain Laurent is also an exact G-difference build, so it runs as a control.

## Validation

The harness builds its own convolution matrices and its own flat-order-set solver, so both are
pinned against the production code before anything is read off them:

| check | residual |
|---|---|
| G-difference build vs `fourier2d_pointwise` | 1.7e-16 |
| G-difference build vs `fourier2d_piecewise_constant` | 4.5e-16 |
| `solve_flat` on a rectangular set vs `_core.solve_2d` (NVF) | 4.0e-15 |

Reference drift (`truncation_order=18` against `truncation_order=20`) is 4e-4 to 9e-4 under NVF, five to eight times
below the smallest error compared.

## Result

Cost ratio is `(N_circ / N_rect at equal error)³`, from a power-law fit over N ≥ 140. Below 1
means the disc wins.

| structure | NVF (production) | Laurent (control) |
|---|---|---|
| circle pillar | **1.20** | 1.49 |
| square pillar | **0.78** | 0.53 |
| square, rotated 30° | **1.01** | 1.04 |

The direction agrees between the two factorizations, so it is a property of the truncation and
not of the factorization.

**Match the cut to the spectrum, not to the feature.** An axis-aligned rectangular pillar has a
separable spectrum, so its corner coefficients are products of two small factors and really are
negligible — cut them. A round pillar's spectrum is isotropic, and there the box's corners are
useful, because they extend the cutoff along the diagonals. "Round cut for round features" is
the wrong intuition, and it is the intuition the `π/4` argument smuggles in.

**The effect collapses under NVF** because NVF converges far faster: fitted slopes −1.6 to −2.4
against Laurent's −0.65 to −1.2. A steeper slope compresses any fixed advantage in effective
cutoff into a smaller harmonic count. What survives is ~20% saved on the friendliest structure
and ~20% lost on a common one.

The circular curves are also visibly **non-monotone** — the circle pillar's circular fit reaches
only r²=0.66. Adding a shell can make a given order's efficiency worse. A box never does that,
and that is worth something to a user sweeping `truncation_order` for convergence.

### Memory

Memory scales as `N²` where compute scales as `N³`, so the same N ratio is *weaker* on the
memory axis, not stronger:

| structure | N ratio | memory (N²) | compute (N³) |
|---|---|---|---|
| circle pillar | 1.064 | 1.13 | 1.20 |
| square pillar | 0.922 | 0.85 | 0.78 |
| square, rotated 30° | 1.002 | 1.00 | 1.01 |

At a *fixed maximum order* the disc does save 44% (N=1257 against 1681, 0.81 GB against 1.45 GB
for the `final_A` pair). But maximum order is not what is bought — accuracy is. Matched on
memory instead, on the circle pillar: `rect truncation_order=16` is 0.61 GB at 2.14e-3, `circ radius=18` is
0.52 GB at 1.97e-3. Accuracy per byte is the same, because both axes are functions of the same
error-vs-N curve.

## What the literature says

Circular truncation is not an S4 invention. Li introduced it — and rejected it — in the paper
that also introduced the non-orthogonal formulation:

> "In addition, two matrix truncation schemes, the parallelogramic truncation and a new
> circular truncation, are considered. Numerical experiments show that the former is superior."
>
> — Li, *J. Opt. Soc. Am. A* **14**(10), 2758 (1997)

S4 ships that paper's convergence benchmark as
`examples/convergence/Li_JOSA_14_2758_1997/`, and defaults to circular regardless
(`S4.cpp:219`, and `lua_api.rst`: "Circular … This is the default"). GRCWA inherits the same
default. Neither documents a justification. The likely reason is API shape rather than
accuracy: only the circular scheme can hit an arbitrary requested harmonic count, because the
parallelogramic one is stuck on odd perfect squares — which S4's own documentation admits.

Note this also **decouples the question from lattice geometry**. Li introduced circular
truncation *in* the non-orthogonal setting and still found parallelogramic better, so an
oblique-lattice implementation does not drag circular truncation in with it — see
[oblique-lattice.md](oblique-lattice.md), which landed since.

## How S4 and GRCWA choose the set

Worth recording, because it is the part that is genuinely automatic — and the part that is not.
GRCWA's `kbloch.py` is a direct port ("From Liu's gsel.c"), so there is one algorithm:

1. **count → radius**: `R = sqrt(nG·|u×v|/π) + |u| + |v|`. Each lattice point owns area
   `|u×v|`; the padding covers the Gauss-circle boundary term.
2. **candidate box**: `u_extent = 1 + ⌊R/(|u|·sinθ)⌋`. Exact, not heuristic — a point
   `m·u + n·v` sits at perpendicular distance `|m|·|u|·sinθ` from `span(v)`.
3. **sort** by the reciprocal metric `|G|² = m²|u|² + 2mn(u·v) + n²|v|²`.
4. **close the shell**: walk back from `nG` while the boundary ties, so the returned count is
   ≤ the requested one.

What is automatic is only **count → set**. The criterion is `|G|` ordering, fixed a priori and
**structure-blind** — no permittivity, no spectrum, no error estimate, no adaptivity. `|G|`
ordering is right for the diagonal part of the operator (evanescence goes as `|k∥+G|`), but
truncation error is set by the coupling through `eps_hat[G_i − G_j]`, which the criterion never
looks at. That is exactly why the answer above came out structure-dependent.

Two implementation notes, if we ever compare against these codes order-for-order:

- **GRCWA drops a shell that S4 keeps** when the requested count lands exactly on a closed-shell
  count: request 441 and S4 returns 441, GRCWA returns 437. Its loop compares `(i, i−1)` where
  S4 compares `(i, i+1)`. So "same nG" is not the same set.
- S4's tie test compares a difference of *squared* lengths against a *length*
  (`d < 2*DBL_EPSILON*maxlen`). Harmless on an orthogonal lattice, where ties fall out exactly;
  on an oblique one it can miss a tie and return a broken shell. GRCWA fixed this to
  `1e-10*max(u², v²)`.

Comparing against either code at equal harmonic count is therefore apples-to-oranges twice
over. Match on maximum order, or on converged values. The GRCWA bridge this repository used to
carry passed `Init_Setup(Gmethod=1)` — parallelogramic — so that benchmark was matched to ours;
the bridge was deleted on 2026-08-15 as dead code, and anyone rebuilding one needs that argument
again.

## Reproducing

```bash
conda run -n wavebit python studies/rcwa/benchmark_truncation_shape/benchmark_truncation_shape.py
```

Run it from the repo root. `main()` takes `max_order_ref`, the structure list, the factorization
modes and `plot=<path>`; the low-contrast control is the same call with `n_ridge=1.5` and
`modes=('laurent',)`. The full sweep above is about 25 minutes and peaks near 2.5 GB at
`max_order_ref=20`.
