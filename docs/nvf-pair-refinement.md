# Refined finite contact cells with bounded assembly storage

Written: 2026-09-13.

Targets:
- `wavebit/_core/conforming/cartesian_pair.py`
- `wavebit/_core/conforming/cartesian_pair_gap_autodiff.py`
- `tests/test_nvf_pair_refinement.py`
- `studies/nvf_pair_material_sector.py`
- `studies/nvf_pair_assembly_cost.py`
- `studies/results/nvf_pair_refinement.log`
- `studies/results/nvf_pair_assembly_cost.log`

The experimental Cartesian pair plans now accept refinements 2 and 3, in
addition to 0 and 1. Exact contact, positive-gap evaluation and their qualified
derivatives remain available on these finer finite spaces. The value/JAX
evaluator contracts the weak matrices without first allocating a quadratic
matrix for every element. This enables finer numerical work; it does not
establish metal optical convergence or fix the ordinary NVF contact field.

## Geometry and the contact limit are unchanged

Each refinement divides every triangle into four using shared edge midpoints.
All vertices retain dyadic barycentric coordinates in the original physical
control vertices. Material boundaries, periodic node identifications and the
physical gap remain exact. No material or geometry is replaced by a raster.
The existing integer collapse map identifies the open nodes with their
contact nodes before evaluating geometry. Surviving oriented triangles and
tags must agree with the contact mesh; every thin triangle has exactly two
distinct contact endpoints, and its pole graph must connect exactly the
declared node groups. All certificates remain enforced at the new refinements.

| refinement | contact unknowns | open unknowns | difference unknowns | open triangles |
| --- | ---: | ---: | ---: | ---: |
| 0 | 6 | 8 | 2 | 18 |
| 1 | 27 | 35 | 8 | 72 |
| 2 | 111 | 143 | 32 | 288 |
| 3 | 447 | 575 | 128 | 1152 |

The plan's default remains 0. Refinement outside the integer range 0 through 3
is rejected. This bound is a qualified resource/geometry scope, not a statement
that three refinements suffice for optical accuracy. Geometry, material,
coercivity, original-equation residual and closure guards are unchanged.
The exact-gap and endpoint cells, the ordinary pair cell and its explicit
`CartesianRectanglePairNVF` adapter share the enlarged plan admission. Their
different derivative and material domains remain as documented in the
[pair](guide/cartesian-rectangle-pair-nvf.md),
[gap](guide/cartesian-pair-gap.md) and
[metal-sector](nvf-pair-material-sector.md) guides.

## Contract before constructing global matrices

The exact gap algebra uses element arrays X, X1 and Y with shapes
`(elements, 2, contact_unknowns)` or
`(elements, 2, difference_unknowns)`. The old value evaluator formed
`left[e].T @ right[e]` for every element, multiplied by its material weight,
selected the regular or collapsed contribution, and finally summed elements.
The intermediate storage was `O(elements * unknowns^2)`.

The new implementation chooses the correct regular/collapsed vector and
weight first. Each block is the same bilinear contraction

```text
flatten(left)^T @ flatten(weight * right).
```

The element and two Cartesian dimensions are flattened into the contraction
axis. There is no conjugation of the left factor: these are the original
complex symmetric weak equations, not a Hermitian surrogate. The thin H00
weight retains its explicit factor of physical normalized gap, including its
derivative at zero. No square root of a vanishing weight or division by gap
is introduced. H01 and H11 retain their original separate scalings.

Storage for this assembly is `O(elements * unknowns + unknowns^2)`. The formal
operation count and dense linear solve have not been removed. The final
matrices, their checks, fields and energy closure are unchanged apart from
ordinary summation roundoff. The actual JAX value-plus-gap-JVP graph has a
test that forbids any intermediate larger than the element/nodal restriction
tensor at refinement 3. This is an array-shape contract, not a bound on total
runtime or compiler RSS.

At refinement 3 the former single H00 element tensor would contain
`1152 * 447^2` entries: about 3.68 GB after complex weighting, before its other
copies. It was not executed at that size for the cost comparison. The final
H00 block contains only `447^2` complex entries, about 3.20 MB. Other inputs,
blocks, derivative buffers and runtime overhead still use memory.

## Independent controls and measured component cost

The 31 new tests cover refined topology on all four chart indices, 12
coefficient/closure cases at contact or gap 7, actual JIT gap tangents,
ordinary native replay with mixed contact-preserving geometry/material
tangents, resource shapes and invalid refinement values. References use
original rational nodal/Laurent equations and separately integrated q16/q24
Duffy moments. They do not assemble the candidate rescaled weak blocks.
The value guard remains `3e-10`; normalized gap derivatives retain `3e-9`,
and original weak residuals retain `1e-10`. No optical golden was changed.

The cost instrument captures actual finite-gap input arrays, then executes the
former expression and the production contraction in fresh NumPy processes.
The fixture uses square cell 700, widths/height 300, gap 7 and materials
`12.25 | -10+i`. Each child executes five assemblies with one-thread BLAS
settings. The table reports the median duration and peak process RSS from the
final run on this host:

| refinement | old median | new median | old RSS | new RSS |
| --- | ---: | ---: | ---: | ---: |
| 1 | .457 ms | .0342 ms | 62.7 MB | 58.9 MB |
| 2 | 37.44 ms | .472 ms | 369.1 MB | 63.5 MB |

At refinement 2 this component is about 79 times faster, with about 83% less
peak process RSS. Normalized maximum block differences are below `2e-16`,
against a predetermined `3e-13` comparison guard. RSS includes runtime and
input overhead. These measurements exclude planning, the cell solve,
Fourier integration, closure construction, compilation and Maxwell propagation;
they are not whole-solver speedup or memory ratios. The result log retains
all five timings and both source hashes.

## Finer optical measurements still reject a general convergence claim

The optical fixture and two second-order one-sided slope windows are unchanged
from the [metal admission study](nvf-pair-material-sector.md). Refinement is
varied independently of the retained Fourier order. At contact:

| materials | order | refinement | p T | Li p T | s T | Li s T |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 4, 4 | 4 | 2 | .997910174 | .997691658 | .984480867 | .988411850 |
| 4, 4 | 4 | 3 | .997778528 | .997691658 | .986545853 | .988411850 |
| 4, 4 | 8 | 2 | .997867938 | .997682393 | .984408760 | .988163075 |
| 4, 4 | 8 | 3 | .997770187 | .997682393 | .986340705 | .988163075 |
| 12.25, -10+i | 4 | 2 | .520814973 | .483946395 | .640217485 | .480212354 |
| 12.25, -10+i | 4 | 3 | .486649484 | .483946395 | .449130307 | .480212354 |
| 12.25, -10+i | 8 | 2 | .609365704 | .480976241 | .549681099 | .402781608 |
| 12.25, -10+i | 8 | 3 | .618325309 | .480976241 | .518567397 | .402781608 |

For equal dielectric materials, the finer value comparisons improve and the
slope discrepancy generally decreases. At order 8/refinement 3, p/s contact
slopes are about `6.31e-6 / -1.421e-4` per length unit; finite-order Li gives
`8.09e-6 / -1.583e-4`. They are still distinct finite models, without a verified
continuum derivative bound.

The metal p value at order 4/refinement 3 lies close to finite-order Li, but
raising the outer order to 8 moves it from .48665 to .61833. Its p/s slopes at
order 8/refinement 3 are about `.01089 / -.01266`, versus Li's
`-.003232 / .01155`, with opposite signs. Thus selecting the attractive
low-order value would conceal the unresolved problem. Cell refinement and
outer truncation must both be assessed. Neither these finite Li values nor
the finite auxiliary cells are certified optical references.

The enlargement and contraction are retained because their finite equations
and derivatives are independently verified and enable further investigation.
The ordinary NVF dispatch remains unchanged. Metal optical convergence and
the ordinary closing-gap contact residual are still open.

```sh
PYTHONPATH=. python -m pytest -q tests/test_nvf_pair_refinement.py
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python studies/nvf_pair_assembly_cost.py
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python studies/nvf_pair_material_sector.py --refinements 2 3 --orders 4 8 --out studies/results/nvf_pair_refinement.log
```
