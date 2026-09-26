# Conforming NVF with live internal background intervals

Written: 2026-09-09

Targets:
- `wavebit/_core/conforming/__init__.py`
- `wavebit/_core/conforming/partitioned_gap.py`
- `wavebit/_core/conforming/material_grading.py`
- `tests/test_conforming_separated_rectangles.py`
- `tests/test_nvf_interface_resolution.py`

`ConformingNVF` supports separated rectangles along either layer axis.
They retain a common perpendicular extent and center; each internal gap
contains the layer background. A frozen plan can open or close these gaps
without changing the input shape count, ordering, selected axis or cell
unknown count. The [manual](guide/conforming-nvf.md) includes a mixed stack
with internal background intervals in both directions.

## Keep both endpoints when a gap closes

For ordered rectangles with endpoints `(left_i, right_i)` and relative
permittivities `epsilon_i/background`, the material profile is

```text
seams  = [right_0, left_1, right_1, left_2, ...]
values = [epsilon_0/background, 1, epsilon_1/background, 1, ...]
```

At contact, two neighboring seams coincide. Their background interval has
zero measure but keeps both independent endpoints. Averaging them into one
seam would discard the derivative for opening that interval. The shared
material integrals already accumulate telescoping suffix jumps, which
continue through zero-width intervals without dividing by their width.
The profile validator now admits nondecreasing seams, including the existing
32-machine-epsilon contact roundoff band. Larger overlaps are refused.

The background interval contributes to the actual material and energy
integrals. It is not a geometric mask or an added optical correction.
Explicitly inserting a rectangle made of the background gives the same
matrices as leaving that interval empty, within the existing 4e-12 control.

## A density that merges coincident interfaces

The former material-grading density added the square of each interface
contrast independently. For a profile `[4, background, 4]`, the two opposite
jumps retained positive weights even when their positions coincided. The
material integrals described a single rectangle, but the mesh still depended
on an interval with zero width. This changed its finite trial space.

Let normalized seam positions be `s_i` and positive relative materials be
`e_i`. The current construction uses

```text
j_i = (log(e_(i+1)) - log(e_i))/2
f(x) = sum_i j_i exp(-(x-s_i)^2/(2 radius^2))
rho(x) = 1 + strength/(sqrt(pi) radius) f(x)^2
C(x) = integral_0^x rho(t) dt
```

The density is positive. At coincident seams, log jumps add to the jump
between the surviving endpoint materials. Opposite jumps cancel, and an
equal-material seam has zero weight. Reciprocal materials reverse f and
therefore preserve the density in exact arithmetic. All values and seam
positions remain live; there is no equality-based merge or stopped gradient.

Products of two Gaussian factors have analytic primitives. With
`mid_ij=(s_i+s_j)/2`,

```text
C(x) = x + strength/2 * sum_(i,j) j_i j_j
       * exp(-(s_i-s_j)^2/(4 radius^2))
       * [erf((x-mid_ij)/radius) - erf(-mid_ij/radius)]
```

The implementation sums the diagonal and twice the upper triangle. It
accumulates the correction before adding x to reduce cancellation when
opposite jumps coincide. Density evaluation uses the signed sum directly,
so its positivity does not rely on expanded cross terms. CDF setup and
evaluation are quadratic in the number of seams; density evaluation is
linear. This does not add cell unknowns as pieces are inserted.

The same eighteen bracket steps and four Newton steps solve
`C(node)=target*C(1)`. Endpoints stay fixed and interior nodes stay ordered.
The production normalized-CDF guard remains 2e-12, the cell residual guard
remains 1e-8 and the energy Hermitian guard remains 1e-9.

## Independent controls and numerical precision

The mesh reference integrates the sampled positive density by adaptive
quadrature and solves its roots with Brent's method. It does not use the
Gaussian-product primitive. Integration in radius units avoids cancellation
when sampling a narrow feature in physical coordinates. The independent
root comparison retains its 8e-15 absolute criterion; JAX derivatives retain
the existing secant criteria.

At radius .0001, one tested float64 root has normalized-CDF residual
3.0087e-14. Its neighboring floats have residuals -4.2411e-14 and 1.0270e-13,
so a blanket 2e-14 residual assertion is impossible for that exported root.
The test now includes the half-ULP root rounding contribution, bounded using
the local density and `C(1)>=1`. This is a test of representable-root
precision, not a change to the production 2e-12 guard or optical accuracy.

Other controls compare every scalar material convolution against independent
analytic shape integrals, all five slots against independent inverse rules
at perpendicular-period contact, and both polarizations through NumPy, JAX
and Torch engines. Twelve live geometry/material/period/pose parameters are
checked against two centered secant steps. Internal-gap opening derivatives
at contact use two one-sided step windows for equal and unequal neighboring
materials in both layer directions. Representation checks retain the same
layer frame: a freshly planned single rectangle can prefer a different
finite trial space, as documented in the [axis contract](nvf-layer-axes.md).

Commands and measured results are recorded in
`studies/results/nvf_separated_rectangles_validation.log`.
The selected suites pass 397 checks with one historical fixed-P1 expected
failure. Both manual examples, strict Sphinx, lint and whitespace checks pass.

For the earlier static control (period 700, width 350, perpendicular gap 17,
seam 31, materials 4/9), the continuous-flux cell with rows 8, x/y
subdivisions 2, gap degree 4 and edge fraction 1e-4 gives:

| Density | Scalar unknowns | Relative static energy enclosure |
| --- | ---: | ---: |
| Fixed mesh, strength 0 | 2591 | 0.001024637861 |
| Signed log-jump density, strength 8, radius .01 | 2591 | 0.000353366564 |

The enclosure is about 65.5% narrower at the same dimension. The independent
fitted-cell reference remains inside both bounds under the existing test.
This measures this static control; it does not establish a general rate or
bound the optical truncation error.

## Effect on earlier results and remaining scope

The signed density replaces independent-jump grading in the shared cell
classes, including their study wrappers. It changes finite-resolution
coefficients and optical readouts when grading is enabled. Earlier tables
remain historical measurements of their original density, not measurements
of this revision. The former CDF is retained as
`independent_graded_fractions` for direct comparisons; complete historical
cell reproduction uses revision `532489b` (landed as `908f60b`) or the revision
named by its log.
Strength zero still selects the unchanged fixed mesh.

These controls qualify construction, derivatives and integration. They do
not certify optical convergence or turn the provisional 1e-5 order increment
into a product requirement. Unequal perpendicular extents or centers,
L-shaped unions, oblique or curved interfaces, and sheared cells remain
unsupported by this builder. The
[common-loss extension](nvf-common-loss.md) permits a shared passive factor
for every material in a patterned layer, including its internal background.

The later [complex-material extension](nvf-complex-material.md) supports
independent material loss. Its positive grading density uses the squared
magnitude of the smoothed complex log jumps, retaining interface merging
and including phase contrast. Real-material grading is unchanged.
