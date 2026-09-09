# Moving material jumps in a fixed gap mesh

`studies/nvf_harmonic_coordinates.py` carries a moving material field jump
inside each fixed x element. The closed mixed bar now matches the exact static
energy, seam/material/width derivatives and independent Cartesian constitutive
matrices even when the material interface cuts an element. The translated
partition and p/s contact checks from the preceding study continue to pass.

This is an opt-in research model for horizontal material partitions in a
rectangular envelope. A positive gap still leaves a small one-sided optical
derivative discrepancy as a real seam crosses a mesh node. That discrepancy,
general geometry, tilted optical convergence and production integration remain
open. No production NVF dispatch or eigenvalue derivative was changed.

The subsequent [integrated continuous-flux model](nvf-continuous-flux.md)
now passes that positive-gap seam gate by making the body displacement
continuous across artificial mesh nodes. The local model's failure and
measurements below remain as controls; general optical convergence is open.

## Material-harmonic coordinate and affine lift

On a fixed x element `[x_i, x_i+h_i]`, define

```
rho(x) = 1/epsilon(x)
R_i    = integral_element rho(x) dx
s_i(x) = integral_{x_i}^x rho(u) du / R_i
s_i'(x) = rho(x) / R_i
```

Replace the usual linear nodal coordinate with `s_i`. The potential remains
continuous, but its derivative follows each material jump inside the element.
The mesh depends on the envelope and explicit resolution settings, not the
number or equality of material pieces. Positive real permittivities are
required, as in the preceding gap study.

Changing the periodic corrector basis alone is insufficient. The physical
affine potential x is generally outside the harmonic element space. The x
column therefore uses a quasiperiodic affine lift as well:

```
ell(x) = x_i + h_i*s_i(x)
L(x)   = ell'(x) = h_i*s_i'(x)
phi_x  = ell(x) + chi_x(x,y)
phi_y  = y      + chi_y(x,y)
```

The lift is continuous, advances by one over the normalized period, and
`ell-x` is periodic. Consequently the mean of the total field B is still I.
The body/gap potential construction and analytic division of the gap test
equation are inherited from [the conforming-gap study](nvf-thin-gap-energy.md).

The lift changes the forward variational forcing and projected energy. In
particular, the x forcing pairs `L` with each nodal derivative in both the
body and the gap. The xx energy includes the weighted `L^2` term and the
`L*grad(chi)` cross terms. Only changing the corrector, while retaining the old
unit affine forcing, gave an incorrect static energy; the implemented model
changes all of these terms together. There is no optical-gradient multiplier.

## Exact integration and partition cancellation

Within each material piece, `s_i` is affine in the physical local coordinate.
The existing exact Fourier moments through polynomial degree two therefore
integrate the weighted basis products, derivative products and lift terms.
The required packs contain

```
integral w * s_i**k * Fourier_wave dx,       k = 0, 1, 2
integral w * s_i' * Fourier_wave dx
integral w * (s_i')**2 * Fourier_wave dx
integral w * L * Fourier_wave dx
integral w * L**2 * Fourier_wave dx
integral w * L*s_i' * Fourier_wave dx
```

Here w is the body permittivity or the unit gap permittivity. Piece changes
are integrated as signed polynomial-coefficient jumps on suffix intervals,
including the reciprocal material moments that define the coordinate. An
equal-material seam changes none of those coefficients; its contribution
cancels without an equality predicate. Independent derivatives of its two
material values remain live. The tested redundant partitions, including ones
inside a nonuniform profile, remain bit-identical in the direct NumPy arrays.

The shared strip projection now accepts the derivative and lift packs. Its
default path preserves the original linear-coordinate model as a control.
For stability on small elements, the new assembly adds differences from a
baseline with identical arithmetic grouping. An unpartitioned element thus
has exactly zero correction, avoiding unnecessary cancellation in its original
mass and stiffness matrices. Energy projection still uses local potential
differences before products.

Independent 48-point piecewise Gauss integration checks every pack, including
the lift terms, at absolute tolerance 2e-12 and relative tolerance 1e-13.

## Closed mixed-bar checks

The fixture has a 700 nm period, 350 nm envelope width, material values 4 and 9
and internal seam x=s. With the periodic gap closed, its exact static normal
energy and seam derivative are

```
Uxx = 1 / (0.5 + (175+s)/700/4 + (175-s)/700/9)
dUxx/ds = -Uxx**2 * (1/4 - 1/9) / 700
```

At seam zero, the old fixed P1 model's left/right secants were
-0.001281263686 and -0.000253089616, against the exact -0.000569451310.
The harmonic-coordinate model gives:

| seam, nm | computed static energy | JIT seam derivative | exact derivative |
| ---: | ---: | ---: | ---: |
| 0 | 1.6941176470588233 | -0.0005694513099357396 | -0.0005694513099357390 |
| 31 | 1.6766467065868258 | -0.0005577667021246928 | -0.0005577667021246925 |

Independent analytic tests also cover seam -31 and derivatives with respect
to both materials and the envelope width. They use absolute tolerances 2e-12
for energy and 2e-11 for derivatives. The old strict static seam-failure test
is parameterized over both models: it remains an expected failure for the
old model and passes without a changed tolerance for the new model.

At outer order 3, closed constitutive matrices for seams 0, 31 and the two-seam
profile `[-73,19]` with materials `[4,9,6.25]` match the independent Cartesian
inverse-rule matrices to 2e-10, accounting for the Fourier origin phase. At
outer order 2, closed p/s transmission and seam/material derivatives pass
independent Cartesian secants at relative tolerance 3e-6 and absolute tolerance
2e-9. Cartesian Li is only the test instrument; the new model never calls it.
These comparisons establish the tested closed laminate behavior at each
stated outer order, not general optical convergence.

## Remaining positive-gap mesh-node sensitivity

At a positive gap, the auxiliary field varies in x and y. The local harmonic
coordinate carries the material jump inside an element, but a finite conforming
Galerkin field enforces flux continuity across coarse element boundaries only
weakly. A material seam passing through such a boundary can still expose a
small discrepancy between the left and right optical derivatives.

For gap 17 nm, seam zero, materials 4/9, outer order 2, s polarization,
`rows=4`, `edge_fraction=1e-4` and a seam step of 1e-4 nm:

| x subdivisions | T | left seam secant, /nm | right seam secant, /nm | absolute difference |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.952153068848 | -0.001382052641 | -0.001383810998 | 1.75836e-6 |
| 2 | 0.951742849497 | -0.001388636899 | -0.001390193894 | 1.55700e-6 |
| 4 | 0.951611423083 | -0.001390800389 | -0.001391687297 | 8.86907e-7 |

Increasing the base body rows from 4 to 8 at four x subdivisions leaves the
difference about 8.86273e-7. Reducing the seam step does not remove the default
mesh discrepancy. Refinement reduces it, but the tested meshes do not meet
the 1e-7 derivative-continuity gate. That gate is a new strict expected failure;
a separate passing test records the measured refinement trend. The order-3
CLI log independently exposes the same issue, with a difference about 1.72e-6.

Unit mean field, lossless power conservation, Hermiticity and a small true
cell residual all pass at a positive gap. None certifies the remaining shape
derivative or establishes a converged optical value. In particular, the JAX
derivative at the mesh node lies between the unequal one-sided secants; its
existence does not make that finite model differentiable there. A subsequent
[integrated flux basis](nvf-continuous-flux.md) resolves this tested discrepancy
in the forward model while retaining the preceding gates.

## Reproduction and validation

Run from the repository root with NumPy, SciPy and JAX installed:

```bash
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_harmonic_coordinates
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_harmonic_coordinates --static
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_harmonic_coordinates --order 2 --gaps 17 --seams 0 --subdivisions 4
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests/test_nvf_harmonic_coordinates.py tests/test_nvf_partitioned_gap_energy.py tests/test_nvf_thin_gap_energy.py tests/test_nvf_cell_bounds.py tests/test_nvf_cell_corrector.py tests/test_nvf_harmonic_energy.py tests/test_nvf_harmonic_energy_jax.py tests/test_resonant_nvf_probe.py tests/test_contact_jones_probe.py::test_equal_material_split_is_equivalent_to_one_rectangle
```

`HarmonicCoordinateGapModel` inherits the preceding model's scalar profile,
translated shape-list API and domain guards. The CLI enables JAX x64 and logs
NumPy/JAX values, seam secants and live gap/seam/material derivatives. The
default optical and static runs are recorded in
`studies/results/nvf_harmonic_coordinates.log`.

Initial validation on 2026-09-08: **94 passed, 5 skipped, 5 xfailed** in the full selection
above; **11 passed, 1 xfailed** in the new dedicated test module. The skips are
the optional NGSolve controls. The five strict expected failures are production
split/merge dependence, the old finite-Fourier s contact derivative, the
separate regularized eigen-gradient bias, the old fixed-P1 static seam error
and the new positive-gap seam discrepancy. No existing tolerance was relaxed.
