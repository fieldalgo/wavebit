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
`('sampled', 'li')`; shapes default to `('integrated', 'nvf')`. Explicit shape
`factorization='li'` supports axis-aligned rectangles wholly inside a positive
axis-aligned rectangular cell. General shapes use NVF. Naming a value the path does not implement raises `NotImplementedError`
rather than quietly running the one it does. All four pairings work on the raster path,
including `('integrated', 'nvf')` — a staircase-free series *and* a rule that handles
boundaries not aligned with x or y, which is what a fit of a tilted sidewall wants.

**`normal_method` selects a raster normal field** under `factorization='nvf'`.
On that path, `projector` composes with all four field constructions. Shape NVF
also accepts `projector='linear'` or `'circular'`: the circular shape construction
supports rectangles, circles, ellipses and their polygon-resolved painting.
Projector selections require NVF; `normal_method` remains a raster option.
See [shape projector details](../parametric-modeling.md#complex-nvf-projection-for-rectangular-contacts).

`'distance'` is the default and the least accurate of the four — on a 60-degree cell it carries
18.5 degrees of mean normal error against `'contour'`'s 0.42. Raster `'circular'` is off by default
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

## Supplying an NVF energy closure

`wavebit.ConvMatrices.from_energy(change, gram, longitudinal)` converts a
supplied field basis and material integrals into the five matrices consumed
by `solve_conv()`. With H retained Fourier orders, supply Q = C(B) and
G = C(B.H epsilon B) with shape `(layers, 2*H, 2*H)`, and C(epsilon_zz)
with shape `(layers, H, H)`. All inputs must use the solver's Cartesian
frame and Fourier order layout, with all x-component orders followed by all
y-component orders in Q and G. Unbatched inputs are also accepted, but
`solve_conv()` expects a layer axis.
This five-slot closure assumes no xz, yz, zx or zy material coupling.

The builder computes `Q^(-H) G Q^(-1)` by solves, inverts the longitudinal
matrix and supplies the off-diagonal signs. It preserves complex material
loss and NumPy/JAX/PyTorch array precision and derivatives. Shape mismatches
and nonfinite eager inputs are refused; invalid traced solves produce NaNs.
Supplying B and resolving its approximation remain the caller's responsibility.

This complete example uses a uniform dielectric layer with B equal to the identity:

```python
import numpy as np
import wavebit

order = 1
count = (2*order+1)**2
change = np.eye(2*count, dtype=complex)[None]
gram = 4.0*change
longitudinal = 4.0*np.eye(count, dtype=complex)[None]
conv = wavebit.ConvMatrices.from_energy(change, gram, longitudinal)
solver = wavebit.rcwa(
    period=(700., 700.), wavelength=830., thickness=[177.],
    theta=.23, phi=.37, top=1., bot=1.44,
    truncation_order=(order, order),
)
result = solver.solve_conv(*conv)
print(result.p_inc.T.sum(), result.s_inc.T.sum())
```

[`wavebit.ConformingNVF`](conforming-nvf.md) constructs this standard layer
stack directly from ordinary rectangle layer dictionaries. It supports
contiguous equal-height tilings in square cells, with live materials,
translations, physical period, and common lattice rotation. The general
energy closure API also accepts externally constructed fields. Arbitrary
shape correctors remain outside the rectangular builder's scope.
