# Choosing the numerics

Four keywords decide how the permittivity becomes convolution matrices and how the layers are
joined. They change how fast the result converges with `truncation_order`, and how much each
step costs -- not what it converges to. Defaults, and the values each accepts, are in
[Keyword reference](../reference/keywords.md#the-numerics).

## How the coefficients are obtained, and how the products are closed

Two independent keywords, which is what makes all four pairings reachable:

| `fourier_coefficients` | |
| --- | --- |
| `'sampled'` | discrete series straight off the pixel grid, with the raster's coefficients in closed form |
| `'integrated'` | closed-form transform of the raster's rectangles. The more accurate route, and the one RETICOLO uses |

| `factorization` | |
| --- | --- |
| `'li'` | Li's rules, applied along x and y |
| `'nvf'` | normal vector field -- factorizes along the material interfaces. Made for curved and tilted boundaries, where a staircase converges slowly |

**`fourier_coefficients` and `factorization` resolve per modeling path**, and reading them back
reports the *effective* value rather than what you asked. On a raster they resolve to
`('sampled', 'li')`; on either shape path to `('integrated', 'nvf')`, which is the only build
those paths implement. Naming a value the path does not implement raises `NotImplementedError`
rather than quietly running the one it does. All four pairings work on the raster path,
including `('integrated', 'nvf')` — a staircase-free series *and* a rule that handles
boundaries not aligned with x or y, which is what a fit of a tilted sidewall wants.

**`normal_method` and `projector` are read exactly where a raster normal field is built**, that
is `factorization='nvf'` on a raster. Set explicitly anywhere else they are *refused* rather
than ignored, because a knob whose reader does not exist is a silent non-choice. They compose:
`projector` is a second axis over all four constructions, not a fifth construction.

`'distance'` is the default and the least accurate of the four — on a 60-degree cell it carries
18.5 degrees of mean normal error against `'contour'`'s 0.42. `'circular'` is off by default
because it is a trade rather than a fix: it removes the medial axis's arbitrariness and costs
accuracy in every cell measured, along with exact mirror symmetry. See
[`../nvf-normal-methods.md`](../nvf-normal-methods.md).

Both connecting algorithms share the same eigenmodes and boundary conventions and agree to
~1e-14. SMM costs ~1.1-1.4x, cascades layer responses that can be reused independently, and is
the only one that forms a `solver.scattering` matrix. See
[`../connecting-algorithms.md`](../connecting-algorithms.md).

## Sampling the normal field

`normal_grid` is the grid the normal field is read off, per axis, and it is the one number here
that can dominate a differentiated step: at `truncation_order=3` the default spends more per
step building the field than the solve itself costs. It defaults to
`max(1024, 4*half_width+1)`, and 1024 is not over-specification -- 256 was tried and reverted,
because it is a rectangle's *angle* that sets the grid and 256 sits 9.7% from the converged
gradient there.

Lowering it is bounded by accuracy rather than by taste. See
[`../nvf-upsampling.md`](../nvf-upsampling.md) and [`../backend-choice.md`](../backend-choice.md).
