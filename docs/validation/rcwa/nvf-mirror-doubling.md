# Differentiable scattering doubling in NVF mirror sectors

The [mirror closure](nvf-mirror-convergence.md) and
[eigenvector-free propagator](nvf-scattering-doubling.md) now work together in
NumPy and JAX. The finite NVF field and Gram closure are unchanged. The new
path handles an exactly grazing layer mode, retains the contact and material
derivative checks, and reduces the cost of the same doubling calculation.

On the recorded CPU fixture, the order-16 NumPy calculation decreases from
47.35 seconds without parity reduction to 12.90 seconds with it. A JIT value
plus one material-scale derivative at order 8 decreases from a warm median
1.922 seconds to 0.513 seconds. These are measured reductions within the
doubling method. The existing modal mirror solver is still faster for the
tested NumPy forward solve, and remains the default. Production NVF dispatch
is unchanged.

## Two distinct symmetry reductions

The original y-reflection restriction has electric components in spaces
`[Ex_a, Ey_b]` and magnetic components in `[Hx_b, Hy_a]`. Their component
dimensions differ because the zero y order appears only in the even space.
The new code retains this distinction when forming L, R, LR and RL.
Both products use expanded expressions to avoid cancellation of fourth-order
wavenumber terms. Tests compare all four factors with explicit orthonormal
projection of the unreduced operators, including empty odd spaces.

Convert the magnetic components to electric ordering with

```text
Hc = [-i Hy_a, i Hx_b].
```

This is a unitary permutation and phase change. It maps the canonical
admittance to identity and the flux to `Re(E.H Hc)`. Applying the same
coordinate change to R, L and RL is necessary; the electric LR product is
already in the correct space. No dense change-of-basis multiplication is
needed for this permutation.

The second reduction comes from a layer that is uniform in z. Its Maxwell
generator is off-diagonal in `(E,Hc)` and is invariant under reversing z and
the sign of Hc. With identical canonical reference ports, its scattering
matrix therefore has equal reflection blocks r and equal transmission blocks
t. This is a spatial symmetry, not an assumption that matrix factors commute.
It applies before matching the unequal physical exterior media.

The shared doubling kernel now stores and computes only r and t. One step is

```text
C = solve(I - r r, t)
r_new = r + (t r) C
t_new = t C.
```

The initial short-slice solve shares a factorization for its reflection and
transmission right-hand sides. The Taylor series, conservative local-tail
criterion and lossless power criterion retain their previous tolerances.
An independent full matrix exponential checks **all four complex scattering
blocks**, including their phases, for noncommuting complex constitutive
matrices and three slice counts. The reduction is not assumed for an
arbitrary stack of dissimilar layers; the general four-block `star` function
remains separate.

## Exterior matching at the sector dimension

The physical top and bottom admittances are mapped into the same flux
coordinates. For either port define `P=(I+Y)/2`, `M=(I-Y)/2`. The NumPy path
keeps these matrices sparse. The matching calculation eliminates the bottom
port first:

```text
O = solve(P_bottom - r M_bottom, t)
R_effective = r + t M_bottom O
reflected = solve(P_top - R_effective M_top,
                  (R_effective P_top - M_top) incoming)
transmitted = O (P_top incoming + M_top reflected).
```

Each dense solve stays at the single-sector dimension H, avoiding a 2H
boundary matrix. JAX uses dense constants for the exterior maps and ordinary
matrix derivatives. The live field-change inverse, energy Gram matrix,
material inverse, propagation and matching all remain in the differentiated
calculation.

## Scope and checks

The coefficient field must remain y-reflection symmetric as its parameters
move. The original reflection tolerance 1e-10, real-field conjugacy tolerance
1e-9, Hermiticity tolerance 1e-9 and lossless power tolerance 1e-9 are retained.
NumPy refuses invalid inputs; JIT returns NaNs for asymmetric or nonreal
fields rather than symmetrizing them. JAX requires float64. This does not
admit the previously rejected unstructured FEM projections.

An exactly grazing **layer** mode now works when doubling is explicitly
selected. Exactly grazing exterior orders remain outside the current port
matching contract. General geometry, nonzero y Bloch incidence, absorbing
layers and production integration are not supplied by this implementation.
The rectangular cell adapter's general-geometry limitations also remain.

Tests cover both JIT contact derivatives against an independently factorized
Cartesian limit, the final 7559-unknown cell contact criterion, live gap/seam/
material derivatives against modal NumPy secants at two step sizes, equal
materials, zero contrast and the exact layer-graze derivative. They also
compare directional and zero-half-width NVF/Cartesian calculations against
their unreduced counterparts. No old production expected failure is removed.

## Measurements

The saved cell is the same mixed rectangle used in the preceding order-40
study: gap 17 nm, width 350 nm, seam 31 nm, relative permittivities 4/9, 7559
cell unknowns, and the unchanged optical fixture. The time comparison includes
the finite optical closure and propagation, but reuses the saved cell field:

| order 16 path | seconds | p T | s T |
| --- | ---: | ---: | ---: |
| modal mirror | 8.559 | 0.907166395579 | 0.941482300436 |
| doubling mirror | 12.898 | 0.907166395579 | 0.941482300434 |
| doubling full | 47.348 | 0.907166395579 | 0.941482300433 |

The measured doubling ratio is 3.67. The timing is not a universal speedup or
a claim that doubling replaces the modal solver for fast forward sweeps.

At order 24, six doublings and twelve Taylor terms give p/s T of
0.906794640678716 / 0.941569618540630. These differ from the preceding modal
result by about +1.36e-12 / -1.93e-13. This checks the propagator at that
retained order; it does not resolve the known outer truncation error.

Increasing from six to seven doublings, while retaining twelve Taylor terms,
gives 0.906794640677999 / 0.941569618539877. The p/s differences are
-7.17e-13 / -7.53e-13. The largest power imbalance across both slice counts
is 1.76e-12. Thus the local-series estimate is accompanied by an explicit
slice-refinement check at the highest newly measured order.

The JIT comparison evaluates both p/s powers and their derivative under
uniform scaling of **all layer permittivities, including the background**.
Uniform material scaling leaves the auxiliary cell potential unchanged, so
its B rows stay fixed while its Gram and material rows scale. This isolates
optical closure/propagation cost and excludes a fresh cell solve. Seven warm
samples follow compilation, with matching value and derivative controls:

| order | full median, seconds | mirror median, seconds | ratio | full compile + first, seconds | mirror compile + first, seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.05045 | 0.02191 | 2.30 | 1.180 | 2.080 |
| 8 | 1.92245 | 0.51270 | 3.75 | 3.001 | 4.873 |

At order 8, the largest full/mirror value difference is about 2.0e-13 and the
largest derivative difference about 1.3e-12. Both derivatives also satisfy
the unchanged independent modal-secant criterion. This one-scalar forward-mode
benchmark does not estimate end-to-end optimization speed with a newly solved
cell or a large geometry Jacobian. Separate tests differentiate the live
rectangular cell in reverse mode.

## Reproduction

Use the cap-80 projection from the preceding mirror study, or create it using
that study's documented command. Forward selection is explicit:

Set `NVF_CACHE` to an existing scratch directory for generated projection files.

```bash
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_mirror_convergence --projection "$NVF_CACHE/wavebit-mirror-cap80.npz" --orders 16 24 --propagation doubling --progress
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_mirror_convergence --projection "$NVF_CACHE/wavebit-mirror-cap80.npz" --orders 24 --propagation doubling --doublings 7 --progress
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_scattering_doubling --mirror --jax
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_scattering_doubling --mirror --jax --gap 0
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_mirror_doubling --projection "$NVF_CACHE/wavebit-mirror-cap80.npz" --orders 4 8 --repeats 7
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests/rcwa/test_nvf_scattering_doubling.py tests/rcwa/test_nvf_mirror_convergence.py tests/rcwa/test_nvf_mirror_cartesian.py tests/rcwa/test_nvf_mirror_doubling.py
```

For direct JIT use, obtain live coefficients with
`PolynomialBodyGapModel.coefficients(..., xp=jax.numpy)` and pass them to
`mirror_doubling_powers(coefficients, order, jax.numpy)`. It returns p/s T,
R and power-balance diagnostics. No production backend switch is required.
The separate Cartesian CLI also accepts `--propagation doubling`; it retains
its independent material factorization.

Raw records are `studies/rcwa/nvf_mirror_doubling/records/nvf_mirror_doubling.log`,
`nvf_mirror_doubling_timing.log`, `nvf_mirror_doubling_jax.log` and
`nvf_mirror_doubling_gap.log`. The last record also verifies the new CLI's
live JIT geometry/material derivatives at mixed-material contact.

Validation on 2026-09-09: **54 passed** across the four test files above,
including 13 new combined-path tests and the preceding 41 controls. Ruff and
whitespace checks also pass.
