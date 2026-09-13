# Passive metals in the finite contact cell

Written: 2026-09-13.

Targets:
- `wavebit/_core/conforming/material_sector.py`
- `wavebit/_core/conforming/cartesian_pair_opening.py`
- `wavebit/_core/conforming/cartesian_pair_gap.py`
- `wavebit/_core/conforming/cartesian_pair_gap_autodiff.py`
- `tests/test_nvf_pair_material_sector.py`
- `studies/nvf_pair_material_sector.py`
- `studies/results/nvf_pair_material_sector.log`

The experimental finite pair family now admits passive negative-real materials
when one explicitly supplied complex half-plane certifies the weak equations.
This extends the existing exact contact/opening algebra to metals, including
NumPy analytical and JAX gap derivatives. It does **not** repair the ordinary
distance-blended normal field or qualify the finite cell's optical accuracy.
The optical comparison below rejects using these coarse cells as its replacement.

## A certificate, with unchanged physical equations

For the original scalar weak form, let `z=exp(-i*theta)` and require

```text
Im(epsilon_j) >= 0,
Re(z*epsilon_j) > 0 for every material, including background.
```

For any nonconstant finite potential u, the real part of the rotated energy is
`sum_j Re(z*epsilon_j) integral_j |grad u|^2`, which is strictly positive after
fixing the constant gauge. This supplies coercivity without requiring positive
unrotated real permittivity. The contact and collapsing-difference diagonal
blocks inherit the same property. At positive gap the correctly rescaled
congruence is the physical energy; the raw nonsymmetric gap matrix's Hermitian
part is still not a valid test of it.

The phase enters **only admission and diagnostics**. The actual weak matrix,
right-hand side, fields, energy coefficients, longitudinal permittivity and
derivatives retain the supplied physical materials. There is no added loss,
material clipping, conjugation of the material, geometry snapping or fitted
optical parameter. Changing between valid certificate angles produces
bit-identical NumPy coefficients, closures and analytical derivatives.

For `B`, the cell's electric-field change of variables, the physical energy is
the Fourier compression of `epsilon B^H B`. Its congruence by the inverse
compressed B preserves nonnegative physical dissipation. It also preserves
positive **rotated** storage. More precisely, orthogonal Fourier projection
gives `integral |B u|^2 >= |Fourier(B) u|^2`. Consequently the rotated closure
is bounded below by `min_j Re(z*epsilon_j) I`, and its dissipative part by
`min_j Im(epsilon_j) I`, whenever the compressed change is invertible.
Independent tests check both lower bounds. Negative unrotated storage is
reported, not relabeled as positive or confused with optical gain.

## Explicit use and failure behavior

`CartesianPairOpeningCell`, `CartesianPairGapCell` and
`CartesianPairGapEvaluator` accept the static constructor keyword
`coercivity_angle` in radians. Its default 0 retains the positive-real passive
domain. A value in `[0, pi]` can certify other passive sectors. For example:

```python
import numpy as np
from wavebit._core.conforming.cartesian_pair_opening import CartesianPairOpeningPlan
from wavebit._core.conforming.cartesian_pair_gap import CartesianPairGapCell

geometry = dict(basis=np.diag([700.0, 700.0]), widths=(300.0, 300.0), height=300.0)
plan = CartesianPairOpeningPlan.from_geometry(**geometry, refinement=1)
cell = CartesianPairGapCell(plan, order=4, coercivity_angle=1.53)
slots, derivative, metadata, _ = cell.closure(
    gap=0.0, epsilon=(12.25, -10.0 + 1j), background=1.0, **geometry
)
assert metadata["minimum_rotated_storage"] > 0
assert metadata["minimum_dissipation"] > -1e-9
```

The same constructor keyword applies to the JAX value evaluator in the
[gap differentiation guide](guide/cartesian-pair-gap-autodiff.md). The angle
is fixed before tracing. Material values remain live; each evaluation checks
their certificate. A sweep leaving the chosen half-plane is refused rather
than automatically changing the angle. This increment qualifies the gap
tangent; arbitrary geometry/material tangents and higher derivatives are
outside its new independent audit.

For a nonzero angle, the normalized material projection must exceed
`64 * float64.eps`. This is a roundoff admission margin, not a smoothing
parameter in the equations. It prevents trigonometric roundoff from admitting
opposed lossless signs at `theta=pi/2`. The method does not admit a mixed
positive/negative lossless spectrum. All-negative lossless material, including
a uniform negative medium, can be certified by `theta=pi`. Gain, zero material,
nonfinite values and invalid angles are refused. Invalid material values raise
eagerly and produce all-NaN coefficients/slots under JIT after safe intermediate
calculations. The original residual and inverse tolerances are unchanged.

Metadata preserves `minimum_storage` and existing unrotated block readings,
and adds `minimum_rotated_storage`,
`contact_minimum_rotated_eigenvalue`,
`difference_minimum_rotated_eigenvalue` and `coercivity_angle`.
The ordinary `CartesianRectanglePairNVF` adapter, its material domain, ordinary
NVF dispatch and geometry admission remain outside this extension.

## Independent finite-family verification

The 40 new tests include 12 metal cases on four declared charts at normalized
gaps `0, 1/64, 1/16`, with refinements 0/1, two different skewed geometries and
signed-basis/unequal-order controls. They reuse the original independent
rational nodal/Laurent equations and moving-simplex Duffy integrals, not the
candidate rescaled weak matrix. The q16/q24 references must agree before
comparison. Coefficient and closure values retain the `3e-10` absolute guard;
gap derivatives multiplied by the cell normalization length retain `3e-9`.
Weak residuals retain `1e-10`. The separate endpoint implementation and actual
JIT gap tangents agree with those references at contact.

Additional tests cover uniform lossy and lossless negative media, scalar lower
bounds, certificate-angle invariance, gain/boundary rejection and invalid JIT
outputs. The 347 existing endpoint/gap/autodiff tests also pass. No external
code, dependency or reference arrays were added; the extension uses the
repository's existing NumPy/JAX implementations and analytical weak equations.

## Optical comparison and decision

The square cell is 700, with two 300-square boxes centered at x offsets
-150/+150, layer thickness 300, wavelength 900, top/bottom epsilon 1/2.25,
theta .2 and phi 0. Positive gap moves the rectangles outward symmetrically.
The finite cell uses `coercivity_angle=1.53`, no sampled normal grid, and the
unchanged harmonic energy closure. Cartesian Li independently integrates the
same physical painting and uses the same Maxwell solver.

At contact with materials `12.25 | -10+i`:

| order | cell refinement | p T | Li p T | s T | Li s T |
| --- | --- | ---: | ---: | ---: | ---: |
| 4 | 0 | .429988475 | .483946395 | .471708640 | .480212354 |
| 4 | 1 | .248701662 | .483946395 | .367536644 | .480212354 |
| 8 | 0 | .303281789 | .480976241 | .417832833 | .402781608 |
| 8 | 1 | .405985788 | .480976241 | .356434074 | .402781608 |

These cells are not optically converged. Refining the cell can worsen its
distance to finite-order Li, whose own order changes are also substantial.
The recorded matrix and gap-derivative agreement cannot establish either
continuum optical value as an accurate reference.

The pre-existing finite construction already removes the same-material
distance-field cusp. For equal materials 4, order 4/refinement 1, p
`T(1e-6)-T(0)=1.60e-11`. The cell's resolved second-order
contact slope is about `1.60e-5` per length unit, while same-order Li gives
`6.57e-6`. Its s slope is `-5.83e-5` versus `-1.58e-4`. Thus small contact
residual alone again fails to certify the physical derivative. Metal contact
values are now finite and differentiable too, but their optical slopes differ
strongly between refinements and from Li, sometimes in sign.

The result log retains both polarizations, all gaps, two second-order one-sided
secant windows, energy/weak diagnostics and source hashes. The `1e-9` values
can reach propagation roundoff; they do not define the reported slope.
This implementation supplies a qualified metal-capable finite contact family
for further work. It is excluded from default integration pending a cell and
optical convergence result that resolves the discrepancies above.

```sh
PYTHONPATH=. python -m pytest -q tests/test_nvf_pair_material_sector.py
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python studies/nvf_pair_material_sector.py --orders 4 8
```


The subsequent [refinement extension](nvf-pair-refinement.md) admits levels
2/3 and contracts the value/JAX weak blocks with less storage. Independent
finite derivatives pass on those spaces too, but the enlarged metal optical
comparison still fails to establish convergence. The 0/1 table above remains
the original material-admission evidence.
