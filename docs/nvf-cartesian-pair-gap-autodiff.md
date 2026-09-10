# Automatic differentiation through the finite pair gap

Written: 2026-09-10

Targets:
- `wavebit/_core/conforming/cartesian_pair_gap_autodiff.py`
- `tests/test_nvf_cartesian_pair_gap_autodiff.py`
- `studies/results/nvf_cartesian_pair_gap_autodiff.log`

The separate experimental `CartesianPairGapEvaluator` evaluates coefficients
and their selected energy closure through the exact rescaled finite-gap
family using NumPy or JAX x64. Ordinary differentiation of this value function
supplies the physical gap response. It adds no custom derivative, stopped
valid gradient, fitted derivative or finite-gap extrapolation. The independently
qualified [NumPy analytical family](nvf-cartesian-pair-gap.md) and
[endpoint response](nvf-cartesian-pair-opening.md) remain unchanged. Usage is in
the [manual](guide/cartesian-pair-gap-autodiff.md).

## Finite equations and contact semantics

The same immutable collapse topology supplies C and Z at refinement 0 or 1.
Normalize physical lengths by L = √det(A), set s = g/L, and use the physical
potential `w = L*(C*u + s*Z*v)`. The analytically removed thin poles give

```text
M(s) x(s) = f(s),  x = [u; v]
M = [[H00, s*H01], [H01^T, H11]].
```

The local equations and field expressions are those of the
[exact finite family](nvf-cartesian-pair-gap.md). Neither its thin fields nor
its solve divides by gap. Exact gathered repeated endpoints and the frozen
dyadic graph determine which triangles collapse; floating small areas never
select the topology. The existing polygon contour transform is finite at
repeated vertices, retaining spatial phase in every collapsing triangle.

The implementation evaluates values only. JAX differentiates the finite
matrix solve, moving field and polygon operations, energy conjugations and
selected closure. The implicit relation `M*x' = f' - M'*x` and the full original
weak flux derivative are checked independently. At g=0, a positive tangent
has physical right-derivative meaning. Negative tangent linearity tests only
the formal extension and do not admit negative-gap primals.

Invalid traced inputs use safe geometry and finite replacement blocks before
final all-NaN coefficient/closure outputs. These masks leave valid arithmetic
live. JAX x64 is required, and planning stays concrete outside tracing.
The intentionally nonsymmetric M is not tested by the minimum eigenvalue of
its raw Hermitian part. The contact/difference diagonal blocks, original weak
residuals, closure storage/loss and inverse identities have separate guards.

## Fixed independent qualification

The predeclared 80 observations use two physical fixtures, the existing four
signed-column representations, refinements 0/1 and gaps `s=0, 1/64, 1/16`.
There are 30 primary, 12 common-unit, six zero/unequal-order, 30 homogeneous
and two explicit lossless points. The finite weak spaces contain 8 or 35 free
potentials. The lossless points are the main chart 0, refinement 0 fixture at
zero and `s=1/64`, with material imaginary parts set exactly to zero.

The references independently assemble original rational nodal Vandermonde
systems at positive gap and original Laurent endpoint systems at contact.
Moving-simplex Duffy q16/q24 integration checks coefficients and the analytical
physical-gap derivative. Independent differentiated closure solves select
exact integer Fourier labels before inversion. The existing NumPy analytical
family is an additional reference, not the only expected result.

The fixed value/quadrature guard is `3e-10`, including the q16/q24 derivative
difference multiplied by L. Actual gap JVP/reference discrepancies multiplied
by L require `3e-9`. Projected, original, differentiated and inverse residuals
require `1e-10`. Storage must remain positive; signed dissipation must exceed
`-1e-9`, with explicit Hermitian/zero-loss controls for lossless materials.
No cases, quadrature levels or tolerances changed after the numerical freeze.

| Frozen scratch measurement | Maximum discrepancy |
| --- | ---: |
| Coefficient value versus independent q24 | 4.458e-15 |
| Closure value versus independent reference | 6.695e-15 |
| L times coefficient gap JVP versus independent reference | 1.191e-13 |
| L times closure gap JVP versus independent reference | 3.072e-14 |
| q16/q24 value / normalized derivative | 1.219e-12 / 5.399e-12 |
| Projected / original unprojected primal residual | 6.840e-16 / 9.013e-16 |
| Differentiated projected / unprojected residual | 2.398e-14 / 2.575e-14 |
| All four closure product identities | 1.906e-15 |
| Signed-basis normalized derivative | 2.845e-14 |
| Common-unit normalized derivative | 2.472e-14 |
| Scalar reverse derivative versus independent projection | 1.336e-13 |
| Reverse versus actual forward JVP projection | 2.642e-14 |

All 18 signed-basis comparisons map coefficient labels and both closure axes.
The 12 unit controls scale every physical length and gap by .01; values remain
invariant and per-unit derivatives scale by 100. Eight endpoint zero/negative
tangent comparisons give zero linearity discrepancy. Four scalar reverse
checks use independently declared complex seeds. Thirty eager JVPs agree with
the compiled results. One compiled chart replays both genuinely different
physical fixtures, checking live input values and gap derivatives rather than
arbitrary geometry/material gradients.

Every triangle weight and its derivative, including exterior collapsing fan
triangles, agrees with the independent reference. Homogeneous derivatives
vanish within roundoff. The two lossless closure Hermitian errors are at most
5.579e-16, with normalized derivative Hermitian error at most 3.553e-15.
Across all 80 cases, M's measured condition ranges 23.8044–489.018. Minimum
closure storage is approximately 1 and the minimum signed dissipation is
-5.506e-16, retained as lossless roundoff without clamping.

## Provenance, audit and limits

Both original stages closed with exit 0 and unchanged 17-source pins.
NumPy reference parent/audit elapsed times were 5.349595/5.116957 s with an
observed process peak of 90,079,232 bytes. JAX parent/audit times were
19.583773/19.008142 s with a peak of 1,989,066,752 bytes. The exclusive 180 s
parent reaped each child. The 8 GiB guard observes cumulative process peaks
after checkpoints/native returns; it is neither a continuous RSS sampler nor
a hard allocation cap.

A separate saved-array audit verified 182 NPZ hashes: 80 references, 80
observations, eight tangent, four reverse and ten invalid controls. It
recomputed all coefficient/closure differences, selected closure product
identities, saved projected matrices and original flux/load residuals,
transform/unit comparisons and invalid all-NaN primals without importing a
solver, evaluator, reference implementation or JAX. Its additive v2 verifies
exact declarations and the finite guards on reported-only residuals; passed
v1 remains preserved.

The saved endpoint regular-field derivative mask was not retained, so that
selected metric is source/report verified rather than inferred from tiny
areas. Every endpoint weight and derivative is independently compared.
Original reference weak matrices/loads were not saved by the scratch stage;
their reported residuals are guarded and pinned, not reassembled by the
saved-array audit. Repository tests recompute the independent equations and
know the structural mask directly.

The dedicated repository suite passes **231 tests in 22.54 s**, recomputing
its independent references from package source. A fresh process blocking JAX
passes 96 NumPy/static tests and skips 135 optional JAX tests in 5.24 s.
The exact guide example also passes against a copied package outside the
repository, with tests, studies, test helpers and PyTorch imports blocked.
Its process peak is 425,345,024 bytes. The full suite's observed peak is
1,899,331,584 bytes; the no-JAX process peaks at 108,707,840 bytes.

The source was promoted mechanically: four imports became relative and
formatting changed, with the remaining numerical abstract syntax exact.
Full-precision frozen measurements, source pins, original guard-only draft
history and repository validation are retained in the
[result log](https://github.com/fieldalgo/wavebit/blob/develop/studies/results/nvf_cartesian_pair_gap_autodiff.log).
Intentional invalid-input NaN values in committed strict JSON use an explicit
`{"invalid_float": "NaN"}` encoding; valid outputs remain subject to finite
guards. External source/checkpoint paths are sanitized without changing their
recorded content hashes.

This is a separate fixed-chart, finite-space gap derivative qualification.
It does not qualify arbitrary geometry/material gradients, mixed or higher
derivatives, a uniform conditioning radius, continuum error or patterned
optical convergence. Ordinary contact-preserving adapter replay/JVP contracts
remain unchanged.
