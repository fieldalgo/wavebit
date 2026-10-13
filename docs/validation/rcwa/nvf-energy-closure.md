# Lower-memory NVF energy closure

The NumPy mirror research path now evaluates the same finite energy closure
with one LU factorization and two solves. Tiled Fourier assembly and direct
Fortran-order block storage lower measured order-32 closure peak process
memory from 2.24 GB to .99 GB, a reduction of about 56%. The action on four fixed complex vectors agrees
with the preceding explicit-inverse implementation within 2.39e-13.
This changes storage and algebraic evaluation, not the NVF field or its
finite-dimensional factorization.

## Two solves

For the change-of-field matrix Q and energy Gram matrix G, the closure is

```text
D = Q^(-H) G Q^(-1).
```

`studies/rcwa/nvf_energy_closure/nvf_energy_closure.py` factors Q once. It first solves
`Q.H A = G`, then solves `D Q = A` on the right. For the LU factorization
`P Q = L U`, the right solve is `D = A U^(-1) L^(-1) P`. Two BLAS
right-side triangular solves reuse A's storage, followed by column swaps
in the reverse order of the LU pivots. These are ordinary, not adjoint,
triangular solves; complex Fourier coefficients and noncommuting pivot
swaps require both distinctions. No inverse array, dense transpose copy
or two dense matrix products are needed.

The default helper preserves its inputs. The mirror caller owns Q and G and
passes `overwrite=True` with Fortran-contiguous arrays, allowing LAPACK to
reuse their storage. Overlapping inputs are separated before factorization.
Singularity and reciprocal-condition diagnostics remain enabled. The helper
does not assume positive definiteness or replace G or D by a Hermitian
projection: a non-Hermitian defect remains observable by the existing guard.

The NumPy parity convolution builds 128 rows at a time, bounding the temporary
Fourier-index and weighting arrays. Four quadrants are copied individually
into each final Fortran-order Q or G, avoiding simultaneous storage of all
quadrants and an additional layout-conversion copy. The NumPy Hermiticity
guard also checks 128 rows at a time with its original absolute 1e-9
threshold, avoiding full conjugate and difference copies. A defect beyond
the first tile and nonfinite data remain detectable. The JAX path retains its
previous differentiable algebra and is checked against NumPy by the existing
mirror/doubling value and derivative controls.

## Maxwell assembly

The y wavevector maps each nonzero-y wave to exactly one wave of opposite
parity. The NumPy modal and exterior-admittance paths now construct that map
as a sparse matrix in linear storage. Multiplications such as `y.T @ z @ y`
use this exact map rather than multiplying dense arrays filled mostly with
zeros. The JAX first-order factors retain their existing dense representation.

The modal solve also constructs its right Maxwell factor after the
eigensolve and divides the resulting magnetic modes in place. Cross blocks
of the constitutive tensor are views instead of stored negated copies.
The bottom-interface elimination also avoids an unnecessary solve. With
`B = V^(-1) Ybottom W` and diagonal propagation D, the original outgoing
system `(I+B) O = D` is evaluated as `(V+Ybottom W) O = V D`. The roundtrip
then follows from the exact identity `(I-B) O = 2 O-D`. This removes one
dense solve, one dense product and the stored identity matrix; it does not
discard evanescent modes or approximate the propagation factors.

Magnetic modes are released before the final electric field is formed. The
entrance solve reuses the no-longer-needed magnetic field array for E+H,
avoiding simultaneous storage of H, E+H and a separate LAPACK copy.

These changes reduce simultaneous dense storage without altering the mode
branch, interface equations or cutoff tests. The closure memory measurements
below exclude this later Maxwell stage; they are not total-solve measurements.

## Measurements and controls

The benchmark reuses the saved cap-80 projection of the 7559-unknown cell in
[the mirror convergence study](nvf-mirror-convergence.md), order 32 and p
polarization, with `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` set.
Build inspection identifies Apple Accelerate for both NumPy and SciPy;
those environment settings do not establish the actual Accelerate thread
count. Each row comes from a fresh process;
peak resident memory includes process imports. GB denotes 10^9 bytes.

| Closure evaluation | Peak process memory | Time | Maximum action difference |
| --- | ---: | ---: | ---: |
| Previous explicit inverse and untiled assembly | 2.237 GB | 8.73 s | reference |
| Two solves, untiled assembly | 1.767 GB | 8.06 s | 2.39e-13 |
| Two solves and tiled Fortran assembly | 1.164 GB | 11.45 s | 2.39e-13 |
| Right triangular solves, tiled assembly and tiled guard | .988 GB | 7.11 s | 2.37e-13 |

The 11.45-second measurement overlapped a separate high-order projection.
The final .988-GB measurement ran after that job completed, with no other
task calculation in progress. Each timing is a single observation; the
memory comparison is the primary result, not a general speed claim.
Raw records are in `studies/rcwa/nvf_energy_closure/records/nvf_closure_memory.log`; build/runtime
details are in `studies/rcwa/nvf_energy_closure/records/nvf_closure_runtime.log`.

The larger cap-96 projection uses local flux coordinates with the same
trial space, cell settings and grading radius .01. Its overlap with the old
cap-80 coefficients differs by at most 3.57e-11. At outer order 16, its p/s
transmission differs from the preceding mirror record by +6.28e-12 / -5.27e-12.
This checks the coordinate change; it is not a cell-refinement measurement.
The original equation residual is 2.21e-11 and reflection defect 7.01e-12.
The projection and optical comparison are recorded in
`studies/rcwa/nvf_energy_closure/records/nvf_closure_projection_check.log`.

On that same cap-96 field, the later sparse modal assembly and reduced
interface elimination change order-16 p/s transmission by +2.83e-13 /
-1.68e-12 relative to the preceding dense modal implementation.

The tests compare both sides of the congruence against an explicit inverse,
including complex and non-Hermitian matrices, input ownership, overlapping
storage, mixed dtypes, singularity and conditioning warnings. Independent
orthonormal Fourier projection checks cross the convolution tile boundary.
The sparse y map is checked against full Fourier multiplication, including
zero half widths. Independent Airy slabs cover zero, 300 and 2400 nm
thicknesses, exercising the cancellation limit and strongly decaying modes.
Existing unreduced closure, Cartesian, contact and JAX mirror/doubling
controls exercise the complete path.

Validation on 2026-09-09: **59 tests passed**, including noncommuting LU
pivots and defects beyond the first guard tile. Ruff
passes for the changed Python files. The validation record is
`studies/rcwa/nvf_energy_closure/records/nvf_energy_closure_validation.log`.

Set `NVF_CACHE` to an existing scratch directory for generated projection files.

```bash
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests/rcwa/test_nvf_energy_closure.py tests/rcwa/test_nvf_mirror_convergence.py tests/rcwa/test_nvf_mirror_cartesian.py tests/rcwa/test_nvf_mirror_doubling.py
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_mirror_convergence --orders 48 --flux-basis local --save-projection "$NVF_CACHE/wavebit-mirror-local-cap96.npz" --progress
```

The mirror symmetry, real-field conjugacy, effective Hermiticity and power
balance thresholds are unchanged. This remains an explicit research path.
Reduced memory does not establish optical convergence, extend the mirror
method to asymmetric geometry or integrate the candidate into production NVF.

## Full-space projected fields

`studies/rcwa/nvf_cell_bounds/nvf_cell_bounds.py::projected_matrices` now uses the same two-sided
solve for independently projected real fields, including the asymmetric
[affine profile](nvf-affine-convergence.md). It retains every requested
Fourier mode without a symmetry split. Each quadrant of Q and G is filled
directly in its final Fortran storage, in tiles of at most 128 rows; a full
convolution temporary is unnecessary. The effective Hermiticity guard keeps
its absolute `1e-9` threshold. The material inverse is evaluated by solving
against the identity after the Q storage is released.

On the fixed degree-8 FEM affine projection at retained order `(24,24)`, two
fresh processes give the following closure-only observations. The exports,
coefficient cap 64 and quadrature 40 are identical. Peak RSS includes imports;
GB denotes `10^9` bytes. The old evaluation is the parent implementation
`e19fd82` (landed as `eba4465`); the new evaluation changes only assembly/storage and the algebraic
evaluation of the same closure.

| Evaluation | Peak process memory | Time |
| --- | ---: | ---: |
| Explicit inverse, full index grids and block copies | 3.430 GB | 9.45 s |
| Two-sided solve and direct tiled assembly | .963 GB | 7.58 s |

This is about 72% less peak process memory for this closure stage. The
maximum difference across every 97th column of all five returned matrices
is `7.19e-14`. This sampled comparison is a continuity check, not an optical
accuracy bound or a full-matrix error norm. The timings are single
observations, with a short regression run overlapping the new measurement;
they do not establish a general speed improvement or the memory of a full
Maxwell solve. NumPy/SciPy use Apple Accelerate; setting
`OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` does not establish its thread
count. Raw records are appended to `studies/rcwa/nvf_energy_closure/records/nvf_closure_memory.log`.

The independent controls construct a nonsymmetric real change of field and
its Gram field pointwise. They check the defining identity `Q.H D Q = G`,
the longitudinal inverse, Hermiticity and input preservation for square,
rectangular and zero orders. Both rectangular orientations cross the
128-row assembly boundary and exercise the final short tile. Singular
change-of-field and material matrices still raise. These controls do not
require mirror symmetry or a positive-definiteness assumption in the solve.

The complete order-24 optical rerun changes p/s transmission by only
`+1.15e-12` / `+1.86e-12` relative to the preceding explicit-inverse record.
That older optical record used quadrature 32 and the rerun uses 40; their
coefficient difference is below `4.89e-15`. Thus this is an end-to-end
continuity observation that includes the quadrature change. The separate
matrix-column comparison above holds the quadrature-40 export fixed.
Validation passes 214 distinct checks across projected-cell algebra,
directional and affine exports, contact controls, mirror/doubling,
the optional fitted FEM instrument and documentation contracts. The strict
documentation build and Ruff also pass. The appended validation details are
in `studies/rcwa/nvf_energy_closure/records/nvf_energy_closure_validation.log`.
