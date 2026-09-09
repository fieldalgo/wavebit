# Bounded temporary storage for mirror modal propagation

This NumPy research change retains the same finite NVF energy closure,
Fourier orders, modal roots and two-interface matching equations. It improves
outer-refinement capacity by removing dense temporary matrices. No mode or
small nonzero propagation factor is dropped, and no physical acceptance
threshold changes.

## Maxwell assembly

`sector_operator` fills the final Fortran-order LR matrix by quadrants. Dense
`np.block` assembly and its subsequent Fortran copy are unnecessary. Sparse
ky exchange contributes only indexed entries, not dense geometry matrices.
The cancellation-free LR expressions used before this change are retained.

`sector_magnetic_modes` computes the constitutive part of R W directly into
its final output, with the magnetic component permutation and signs. It adds
the diagonal and sparse-ky terms in panels of 128 modal columns. The dense R
factor is never allocated. Division by the original modal rates is in place.
The eigensolve still finds every retained mode; its cubic cost is unchanged.

## Interface matching

Let W and V contain electric and magnetic modes, D the diagonal decay matrix,
and Yt/Yb the exterior admittances. The bottom solve remains

```
O = (V + Yb W)^-1 V D
Rround = D (2 O - D)
```

Both bottom-solve operands are built in their final Fortran storage. Sparse
exterior products use 128-column panels. The entrance system is

```
Afront = Yt^-1 V (I - Rround) + W (I + Rround)
Afront f = 2 Ein
```

Rather than storing Rround, magnetic row panels form
`V (I + D^2) - 2 (V D) O`. Sparse Yt inversion is applied in column panels.
Electric row panels then add `W (I - D^2) + 2 (W D) O` directly into Afront.
This removes the stored round-trip matrix and the separate Efront matrix.
The final fields require only vectors:

```
b = O f
Etransmitted = 2 W b
Ereflected = W [f + D (2 b - D f)] - Ein
```

Every column of W, V and O participates. In particular, these identities do
not approximate weakly transmitted evanescent modes or their interface
coupling. The dense eigensolve and LU solves still limit large-order cost.

## Validation

The new test file compares direct LR storage and R W actions with explicit
Maxwell factors for both parity sectors, including empty component spaces,
unequal x/y orders, complex non-Hermitian test matrices and dimensions above
the panel size. Input arrays must be preserved.

A separate check solves the coupled forward/backward equations at both
interfaces as a 2H-by-2H system. Both polarizations agree with panel matching
for a patterned layer at thicknesses 0, 300 and 2400 nm. Reversing every
propagating mode label also preserves the finite-layer powers.

An initial proposed benchmark of V W^-1 was unsuitable: tiny eigensolve
roundoff can reverse the selected sign of nearly imaginary modal roots.
That changes V W^-1 even when the full slab scattering is unchanged. The
implementation keeps the existing root selection; final T/R and direct
Maxwell equations are the acceptance controls. The rejected diagnostic is
recorded in `studies/results/nvf_mirror_modes_memory.log`.

This is still the lossless, y-mirror-symmetric rectangular research path.
It does not implement general geometry, production integration, an optical
error bound or a near-grazing cluster treatment.

## Measured propagation memory

Fresh processes used the same saved cap-96 local-basis coefficients, outer
order 32 and p polarization. Each measurement includes closure, every modal
solve, both interfaces and final powers:

| Implementation | Peak process RSS | Elapsed | p T |
| --- | ---: | ---: | ---: |
| Previous dense modal/intermediate blocks | 2,122,334,208 bytes | 89.456 s | 0.9066971830116604 |
| Direct modal storage and interface panels | 1,429,438,464 bytes | 65.169 s | 0.9066971830119874 |

Peak process RSS falls by 32.65%. The T difference is 3.27e-13 and the R
difference is 5.58e-14. Both retain the original 1e-9 power-balance gate;
the panel result's residual is 2.50e-13. These measurements are for the
specified entire single-sector calculation, unlike the earlier closure-only
memory table. They do not predict the peak of every larger order.

Timing is a single observation per method, with small subsecond tests
running during each measurement. NumPy/SciPy use Apple Accelerate; setting
OPENBLAS_NUM_THREADS and OMP_NUM_THREADS does not establish its actual
thread count. This is not a controlled throughput or universal speed claim.
Raw records: `studies/results/nvf_mirror_propagation_memory.log`.

## Selective directional checks

`mirror_powers(..., polarizations=('p',))` and CLI `--polarizations p` can
explicitly request just one expensive sector; the default remains both.
Results omit uncomputed polarization keys, and every computed sector still
must satisfy the unchanged power-balance guard. An empty, repeated or invalid
selection is rejected. Single-polarization output must not be presented as a
combined p/s convergence check.

Final relevant validation after the modal, interface and sector-selection
changes: **76 passed in 82.41 s**. The five files are
`test_nvf_energy_closure.py`, `test_nvf_mirror_convergence.py`,
`test_nvf_mirror_cartesian.py`, `test_nvf_mirror_doubling.py` and
`test_nvf_mirror_modes.py`; the latter contributes 17 tests.
The suite includes the existing independent full-Fourier, Cartesian interval,
Airy, contact and JAX/doubling checks. Log:
`studies/results/nvf_mirror_modes_validation.log`.

The exact pre-change modal/matching comparison script is retained as
`studies/results/nvf_mirror_propagation_check.py.txt`. Copy it to a `.py`
file and run with arguments `original 32 p` or `panels 32 p` after generating
the specified cap-96 projection. The benchmark runtime and source hashes are
recorded in `studies/results/nvf_modal_memory_manifest.json`. This records the
dirty-worktree implementation rather than implying a committed release.
