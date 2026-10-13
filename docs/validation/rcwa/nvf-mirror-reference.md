# Independent auxiliary FEM cell with mirror boundary spaces

`studies/rcwa/nvf_mirror_reference/nvf_mirror_reference.py` solves the rectangular material profile on
the upper half of its periodic cell. The finite-element spaces implement
the exact y parities of the two scalar correctors. This permits the existing
mirror Maxwell instrument to accept independently fitted FEM fields under
its original reflection tolerance. The previously saved full-cell FEM export
still fails that strict symmetry check.

This is an independent discretization of the static auxiliary cell, using
NGSolve/Netgen fitted triangles. It supplies a control for the candidate's
auxiliary-field error. It is not an independent three-dimensional Maxwell
solution or an optical truncation-error certificate.

## Half-cell boundary problem

The material profile is invariant under y reflection. For a corrector
`B_j = e_j + grad(u_j)`, the full-cell weak problem is

```text
integral epsilon grad(u_j) . grad(v) = -integral epsilon e_j . grad(v).
```

The x corrector is even in y; the y corrector is odd. Periodicity makes
`y=0` and `y=period/2` reflection planes. On the upper half cell:

- `u_x` has natural homogeneous Neumann conditions at both planes. A scalar
  multiplier sets its mean to zero.
- `u_y` has zero Dirichlet values at both planes. Its odd periodic extension
  already has zero full-cell mean, so no multiplier is added.
- Both spaces remain periodic in x. The generated periodic point pairs,
  material areas and half-cell area are checked before solving.

The code solves these spaces separately for epsilon and its reciprocal.
Residuals are evaluated on free degrees of freedom; nonzero Dirichlet
reactions are excluded. Energy is normalized by the full cell area, with
twice the half-cell diagonal integrals. The cross energy integrates to zero
because it is odd. The primal/reciprocal energy enclosure uses the existing
two-dimensional duality construction.

## Fourier integration

The eight field/Gram rows have y parities
`+,-,-,+,+,-,+,+`. Each fitted triangle in the upper half contributes

```text
integral_half f(x,y) exp(-i kx x)
              [exp(-i ky y) + parity(f) exp(+i ky y)].
```

This integrates the two halves of the field defined by the boundary spaces.
It does not average an asymmetric full-cell coefficient export. The field
itself is conforming under its even or odd extension. Duffy-mapped Gauss
quadrature and point batches integrate the fitted triangles without a raster.
The zero Fourier mode must match both the independently integrated energy
and the identity mean field. High Fourier caps additionally require an
explicit quadrature-refinement check.

The integration now uses the shared `nvf_fem_quadrature.py` iterator. Each
Duffy point is transformed through its known triangle and retains that
element during field/material evaluation. The full-cell exporter uses the
same iterator with full-cell normalization. No global coordinate search is
needed to rediscover the element. New exports record
`quadrature_mapping='element_local'`.

The physical metadata retains `fixture='partitioned_gap'` for the existing
projection loader, with `cell_method='fitted_half_cell'`. `matrix_dimensions`
records the two assembled matrix sizes, including constrained Dirichlet
slots; the inverse acts only on free degrees of freedom. `ndof` is the sum
of those sizes, rather than the size of one factorization. No moving-mesh derivative or asymmetric-geometry contract
is supplied by this reference instrument.

## Independent controls

The initial four tests run in the optional NGSolve environment:

- At a closed gap, all retained Fourier coefficients of the laminate basis
  and Gram field match analytic material-interval integrals. Both static
  energy bounds recover the harmonic/arithmetic tensor.
- At a positive gap, Fourier quadrature refinement, real-field conjugacy,
  exact y parity, identity mean field and the static enclosure are checked.
- The half-cell and independently meshed full-cell energy enclosures are
  mutually consistent in the matrix ordering.
- Invalid geometry, insufficient quadrature and incomplete material maps
  are refused.

Initial validation on 2026-09-09: **4 tests passed** in NGSolve 6.2.2606. The record is
`studies/rcwa/nvf_mirror_reference/records/nvf_mirror_reference_validation.log`. The ordinary solver
environment skips these optional tests when NGSolve is unavailable.

## Resolved coefficient export and optical check

For the unchanged 17-nm-gap, 350-nm-wide, 4/9 profile at seam 31 nm, the
`h=.03` (30 nm), degree-8 half mesh has 1318 triangles. Its two assembled
matrix sizes are 42633 and 42632. The relative static enclosure is 4.19e-7. The
reported y-reflection coefficient defect is zero; it therefore passes the
original 1e-10 mirror guard without changing that guard or modifying the
older full-cell export.

At cap 96, raising the triangle quadrature from 28 to 32 changes the eight
coefficient arrays by at most 7.55e-15. Mean energy quadrature error is below
1.6e-13. The two projection runs peak below .89 GB in the separate FEM
process; their 71/95-second times overlap a different optical calculation
and are not isolated performance measurements. Raw projection and quadrature
records are in `studies/rcwa/nvf_mirror_reference/records/nvf_mirror_reference_projection.log`.

The order-16 p/s transmission is 0.907165904921 / 0.941480620939. It differs
from the recorded full-cell fitted FEM control by -2.22e-9 / -6.82e-9, with
power-balance residual 1.73e-12. The two meshes and parity spaces are distinct,
so exact equality of their finite approximations is not expected. This optical
check supplements the analytic laminate and static-energy controls.
Records are in `studies/rcwa/nvf_mirror_reference/records/nvf_mirror_reference_powers.log`.

The check also reaches the preceding full-cell reference's highest recorded
order, 24. The half-cell p/s transmission is 0.906794051046 /
0.941568376098, differing by -2.22e-9 / -6.76e-9 from that independent
full-cell calculation. The mesh-to-mesh discrepancy therefore stays below
7e-9 across both of these comparisons.

The new independent auxiliary-field comparison reaches orders 40 and 48:

| order | Fitted FEM p T | Fitted FEM s T | Candidate minus FEM p | Candidate minus FEM s |
| --- | ---: | ---: | ---: | ---: |
| 40 | 0.906643800454 | 0.941524029790 | +4.80152614e-7 | +9.57903912e-7 |
| 48 | 0.906625295961 | 0.941536667439 | +4.47042615e-7 | +8.39880476e-7 |

The fitted FEM changes by -1.85044931e-5 / +1.26376491e-5 from order 40 to
48. The candidate changes by -1.85376031e-5 / +1.25196257e-5. Thus both
auxiliary discretizations retain order-step changes above 1e-5; their
same-order differences stay below 9.58e-7. For this fixture, the observed
remaining change is dominated by outer Fourier truncation rather than the
choice between these two cell discretizations. This comparison
does not bound either calculation's error against a 3D Maxwell solution.

The 40/48 FEM optical process peaks at 6.95 GB, with lossless residuals below
2.1e-11. Its Maxwell propagation is shared with the candidate. The cell
discretization and its variational controls are independent. Summary records
are in `studies/rcwa/nvf_mirror_reference/records/nvf_mirror_reference_comparison.log`; all powers,
progress stages and peak memory are in `nvf_mirror_reference_powers.log`.

Set `NVF_CACHE` to an existing scratch directory for generated projection files.

Set `NVF_FEM_PYTHON` to the Python executable with the optional NGSolve dependencies.

```bash
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$NVF_FEM_PYTHON" -m unittest discover -s tests -p test_nvf_mirror_reference.py -v
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$NVF_FEM_PYTHON" -m studies.nvf_mirror_reference --cap 96 --h .03 --degree 8 --quadrature 32 --output "$NVF_CACHE/wavebit-mirror-fem-cap96.npz"
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_mirror_convergence --projection "$NVF_CACHE/wavebit-mirror-fem-cap96.npz" --orders 16 24 40 48 --progress
```

## Element ownership at high quadrature

Extending the same mesh to cap 112 exposed a small point-location artifact.
The old global coordinate lookup gives a q32-to-q36 maximum coefficient
change of 3.3862e-11. The independent material Fourier row alone changes by
2.8451e-11. A separate evaluation of 1,708,128 q36 points finds **four points
assigned to a different material** by global lookup. For example,
`x=0.03099999107330665` micrometers lies to the left of the `.031` seam:
its owning element has epsilon 4, while global lookup supplies epsilon 9.
The full-cell absolute weighted material discrepancy is 2.85594e-11.
This isolates the lookup problem independently of either corrector solve.

The shared element-transformation iterator removes this rediscovery step
for both half-cell and full-cell exports. It preserves the same physical
Duffy nodes and normalization. It also avoids materializing all quadrature
coordinates at once. The half/full regression integrates selected Fourier
orders through 112 and compares the material coefficients to exact rectangle
interval integrals with absolute tolerance 5e-13.

After sharing this fix, **13 optional FEM tests pass in 3.604 s**: five
half-cell tests (including both domains in the new Fourier control), five
full-cell corrector tests and three partitioned-gap reference tests. Existing
laminate, tilted-cell, energy-enclosure and invalid-input criteria are
unchanged. These are separate from the 76 modal/closure tests.

The element-local q32 cap-112 coefficients differ from the previous global
q32 export by at most 7.55e-15. Thus the earlier q32-based optical comparisons
are unaffected at coefficient precision. This observation is not an optical
error bound or a new full 3D reference. Diagnostic and validation records:
`nvf_mirror_reference_element_lookup.log`, `nvf_fem_element_validation.log`
and `nvf_mirror_reference_element_projection.log` under `studies/results/`.

After element-local evaluation, q32-to-q36 changes the cap-112 coefficients
by only **3.1086e-15**, compared with 3.3862e-11 before the fix. The q36 mean
energy error falls from 1.66e-11 to 1.51e-13. The static cell, mesh and degree
are identical. This confirms that the resolved difference was point lookup,
not a need to refine the auxiliary trial space. Both runs fit below .96 GB
of peak process memory; timings overlapped the directional optical sweep.
The validated higher-cap reference is saved as
`wavebit-mirror-fem-cap112-element-q36.npz` in this workspace
session. It supports later outer-order comparisons, but only completed
optical measurements should be reported as such.

A same-propagator order-16 check compares the saved global-q32 cap-96 field
with the corrected element-local q36 cap-112 field. Their p/s T differences
are +1.04e-13 / -4.87e-12, and their power-balance residuals remain below
3.54e-12. Record: `studies/rcwa/nvf_mirror_reference/records/nvf_mirror_reference_element_optical.log`.
The corrected reference passes the original mirror guard with zero measured
reflection defect.
