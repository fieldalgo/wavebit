# Partition invariance and the thin-gap field in one model

`studies/rcwa/nvf_partitioned_gap_energy/nvf_partitioned_gap_energy.py` extends the conforming-gap study to
horizontal material partitions and translated rectangular shape lists. The
same finite model now passes both the equal-material partition checks and
the p/s contact derivative checks. This closes the earlier gap between two
separate instruments: the Fourier model had partition invariance but a wrong
s contact derivative; the initial conforming model had the contact field but
no partition representation.

This remains a research implementation. It supports a rectangular envelope
partitioned along x, with a periodic gap along y. Tilted/curved boundaries,
general painting and production NVF integration remain open.

The subsequent [material-harmonic coordinate study](nvf-harmonic-coordinates.md)
now resolves the closed mixed-bar field jump and static seam derivative while
preserving these partition/contact gates. It still has a smaller positive-gap
mesh-node sensitivity discrepancy. The fixed-P1 errors below remain recorded
and tested for this original model.

## Material jumps on a fixed space

The cell mesh depends on the envelope and explicit resolution settings. It
does not depend on the number or equality of the material pieces. With ordered
seams s and material values e, the material function is expressed as

```
epsilon(x) = 1 + (e[0]-1) I[envelope]
               + sum_j (e[j]-e[j-1]) I[s[j-1], envelope_right]
```

Each term after the envelope fills a suffix. This is a telescoping identity
for a piecewise-constant material, including unequal materials. An artificial
equal-material seam has exactly zero weight. Its motion therefore contributes
zero to the matrices and their physical-parameter derivatives. Independent
material derivatives remain live at equality: the derivative of `e[j]-e[j-1]`
is not zero when just one material value changes.

There is no equality predicate, conditional geometry merge or removal of a
material parameter. The same identity also cancels redundant seams inserted
inside an otherwise nonuniform material profile. All eight projected
coefficient arrays are bit-identical in the tested direct NumPy spellings.

The actual corrector space and its analytically scaled gap equation are the
ones derived in [the conforming-gap study](nvf-thin-gap-energy.md). A small
shared-helper refactor reuses that solve and the energy closure; production
factorization routines were not changed.

## Exact integration of a partitioned material

On each fixed P1 element, integrate the material against the local polynomial
basis and Fourier waves using moments

```
integral r^k exp(-2*pi*i*m*x) dx,    k = 0, 1, 2.
```

These moments supply the cell mass/stiffness/load contributions and the
Fourier projection of B, `B.T*epsilon*B` and epsilon. Sinc and its first two
derivatives give the closed forms. Convergent Taylor series near zero avoid
cancellation. Zero-length intersections remain in the formula, including the
live derivative at exact contact. No permittivity raster or Fourier projection
quadrature is used by the new model.

For numerical stability, the projected energy is evaluated from local
potential differences. The first implementation multiplied potentials by an
assembled stiffness matrix; on tiny interface elements, cancellation lost
about 1e-11 in projected coefficients. Forming local differences before their
products reduces disagreement with independent conforming quadrature to
about 2e-14 on the measured controls. The exact interval moments also agree
with independent 80-point Gauss integration, including near-zero frequency,
negative frequencies and zero-width intervals.

## Shapes, derivatives and measurements

The existing scalar interface remains available:

```python
model = PartitionedGapEnergyModel(order=7, rows=8, edge_fraction=1e-6)
profile = dict(seams=[31.], values=[12.25, 12.25])
result = model.transmission(gap=0., epsilon=profile, width=350.)
```

The shape adapter accepts study dictionaries with `lx`, `ly`, `epsilon`, and
optional `cx`, `cy`, `angle`. They must form an ordered horizontal tiling, with
the same height/y center and zero angle. It verifies that geometry, extracts
the common envelope and material seams, and applies the exact Fourier phase
for the overall translation. It does not fill gaps, unite unrelated shapes
or provide a general painting operation. Invalid traced updates produce
nonfinite data; invalid ordinary calls raise an error.

```python
shapes = [dict(lx=300., ly=300., epsilon=4., cx=-150.),
          dict(lx=300., ly=300., epsilon=4., cx=150.)]
result = model.shape_transmission(shapes)
```

The tests compare this 600x300 spelling with one rectangle, including a moving
artificial seam, translation, both polarizations and exact periodic contact at
height 700. Value differences pass the original 1e-10 partition criterion;
physical gradient differences pass 2e-10. Shifting the shapes across periodic
cell boundaries leaves powers invariant. Independent material derivatives at
equal materials, at zero contrast and at unequal materials agree with NumPy
finite differences, as does seam motion when it represents a real contrast.

At outer order 7, width 350, epsilon 12.25 and x edge fraction 1e-6, the
translated shape adapter gives identical merged/split results:

| polarization | closed T | dT/dgap |
| --- | ---: | ---: |
| p | 0.626061531896588 | 0.001891546100056 |
| s | 0.878242289574451 | -0.000778566984054 |

The artificial-seam derivative is zero in both runs. Translation derivatives
are below 4e-11 in this finest recorded case. The s contact discrepancy remains
approximately 0.001% against the independent reference. Both contact gates
also pass at the cheaper x fraction 1e-5, with the same 1e-7 absolute criterion
used by the preceding study. Cartesian Li is only an independent test
instrument; it is not called by the new model or its shape adapter.

## Representation invariance does not remove cell error

An unequal material seam may cut through a fixed mesh element. Integration of
that material is exact, but the finite cell space may not resolve its field
jump. A closed mixed bar with its seam on a mesh node reproduces the independent
inverse-rule matrices. An off-node seam retains a measured approximation error.

For width 350, seam x=31 and materials 4/9, the exact static normal energy is
1.676646706586826. Refining each original x interval independently of the
material partition gives:

| x subdivisions | computed normal energy | excess over exact |
| --- | ---: | ---: |
| 1 | 1.684405196727 | 0.007758490140 |
| 2 | 1.681276361636 | 0.004629655049 |
| 4 | 1.678794667052 | 0.002147960465 |

Use `x_subdivisions` to reproduce this nested refinement. Partition equality
is preserved at every resolution. This table is a static cell convergence
check, not an optical transmission error bound or a claim of converged mixed
material accuracy. General interface conformity, general primitive
partitions, tilted optical convergence and the independent 3D reference
uncertainty still need work before production adoption.

An independent moving-interface check reveals a stronger remaining limitation.
For the same 4/9 bar with its real seam initially on the x=0 mesh node, the exact
static derivative with respect to seam position is -0.000569451310. The computed
left/right secants at step 1e-4 are -0.001281263686 and -0.000253089616. The
closed energy is exact, but moving the material jump through a fixed P1 element
does not move the field jump with it. Agreement of this model's AD with its own
secants away from mesh nodes therefore cannot establish physical seam accuracy.
This is recorded as a new strict expected-failure gate. It does not affect the
proved cancellation of an equal-material seam.

The follow-up [harmonic-coordinate formulation](nvf-harmonic-coordinates.md)
uses the cumulative integral of `1/epsilon`, normalized by the whole-element
integral. It carries the moving field jump without changing the mesh or
branching at material equality. The implementation also changes the affine
lift, variational forcing and forward energy. Its independent closed static
and optical gates now pass; a positive-gap derivative discrepancy remains.

## Reproduction and validation

```bash
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 python -m studies.nvf_partitioned_gap_energy --order 7 --edge-fraction 0.000001 --shape-gates
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 python -m studies.nvf_partitioned_gap_energy --order 3 --width 600 --epsilon 4 --gap 400 --shape-gates
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/rcwa/test_nvf_partitioned_gap_energy.py tests/rcwa/test_nvf_thin_gap_energy.py
```

Recorded comparisons are in `studies/rcwa/nvf_partitioned_gap_energy/records/nvf_partitioned_gap_energy.log`.
The CLI uses JAX x64; the Python model also runs with NumPy alone.

Initial validation on 2026-09-08: **19 new tests pass**, with the newly exposed real-seam
physics failure retained as a strict expected failure. The combined relevant
selection gives **66 passed, 5 skipped, 4 xfailed**. The optional NGSolve skips
and three old strict expected failures are retained; those are the original
production split failure, finite Fourier s-contact failure and separate small
regularized eigen-gradient failure. Targeted Ruff and `git diff --check` pass.
No physics tolerance was relaxed.
