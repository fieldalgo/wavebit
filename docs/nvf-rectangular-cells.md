# Conforming NVF on rectangular lattices

Written: 2026-09-09

Targets:
- `wavebit/_core/conforming/__init__.py`
- `wavebit/_core/conforming/continuous_flux.py`
- `wavebit/_core/conforming/polynomial_body.py`
- `wavebit/_core/conforming/partitioned_gap.py`
- `wavebit/_core/conforming/thin_gap.py`
- `tests/test_conforming_rectangular_lattice.py`
- `tests/test_nvf_polynomial_body.py`
- `tests/test_nvf_gap_convergence.py`

`ConformingNVF` supports unequal orthogonal periods by including the physical
aspect metric in the cell solve and energy projection. Merely stretching
the square-cell coordinates produces a conforming field, but generally
does not minimize the physical cell energy. Independent stationarity and
quadrature checks distinguish those two constructions. The implementation
also preserves contact derivatives when height normalization rounds to a
tiny negative gap.

## Pulling the physical energy back to the normalized cell

Let the local lattice periods be Lx and Ly, with normalized coordinates
xi = x/Lx and eta = y/Ly. The existing integrated-flux trial functions and
material partitions remain in this normalized unit cell. Write

```text
a = Lx/Ly
M = diag(1, a^2)
S = diag(1, 1/a)
F = grad_(xi,eta) [ell(xi) + chi_x, eta + chi_y]
B = diag(1, a) F S
```

F contains normalized-coordinate gradients. B contains physical electric
fields whose columns have unit Cartesian mean. The two columns of chi are
periodic potentials. Multiplying the physical energy by the common Lx
normalization yields the metric M, so the corrector equations minimize
the integral of `F.T M epsilon F`. The physical potential-column scale S
does not change the minimizer of either independently solved column.

For each Fourier mode, the projected physical energy is

```text
G_hat = integral_unit_cell F.T M epsilon F exp(-i k.x) dxi deta
G = S.T G_hat S
```

The determinant of the physical coordinate change cancels with the cell
area in the normalized average. Multiplying G by an extra aspect or area
factor would therefore be incorrect. The scalar material coefficients
still integrate epsilon without a metric factor. Common rotation and the
cell-center phase shift are applied after this local construction, before
the standard energy closure and scattering solve.

| Contribution | Rectangular-cell change |
| --- | --- |
| x derivatives, x forcing and x affine lift | Keep their normalized coefficients. |
| y derivative energy and y forcing | Multiply mass terms and y loads by `a^2`. |
| Body and gap energy projection | Apply `a^2` to y quadratic, cross and affine-lift terms. |
| Projected gradient field | Scale rows and columns as B above. |
| Projected energy | Apply S on both potential-column axes. |
| Longitudinal scalar material convolution | No metric factor. |

The same rules cover affine and enriched body/gap potentials. Aspect is a
live argument of the shared continuous-flux cell, not cached in the mesh.
The public shape plan records input ordering and a layer's
[tiling axis](nvf-layer-axes.md), so either period may change on replay while
the geometry remains a valid tiling in that frame. A square
cell has `a=1` and recovers the previous equations. Sheared lattices remain
refused because a full off-diagonal metric requires further assembly work.

## Contact roundoff and its derivative

The former adapter computed `700 - height*(700/Ly)`. At height=Ly=430,
rounding can make the normalized height slightly exceed 700. A conditional
replacement of the resulting negative gap with zero preserved the forward
value but made its JAX gap derivative exactly zero. The independent
one-sided secant in the tested rectangular fixture was about -0.2354.

The adapter now subtracts the physical height before normalizing:
`gap = (Ly-height)*(700/Ly)`. A rotated lattice norm can itself round below
the declared height, so this ordering alone is not sufficient. The existing
geometry allowance of 32 machine epsilons is also carried into the cell
validation. Within that tiny negative neighborhood, the implementation
evaluates the same analytically continued contact equations and signed
Fourier integrals. The scaled test equations are nonsingular at zero.
There is no value clamp, stopped gradient, custom derivative, or derivative
that differs from the implemented forward formula.

This does not admit a physical overlap beyond floating-point roundoff.
A height overrun outside the tolerance still raises eagerly and returns
NaNs when traced. The residual guard remains 1e-8 and the Hermitian guard
remains 1e-9; no accuracy or convergence threshold was relaxed.

## Independent controls and limits

| Control | What it checks |
| --- | --- |
| Tensor Gauss integration at y metrics .36, 1 and 3.24 | Body and gap energies, including affine lifts, agree with direct sampled-field integrals. |
| Physical-energy stationarity at aspects .6 and 1.8 | Independently varying the projected quadratic energy gives zero directional derivative for the solved potentials. The stretched square-cell control fails this check. |
| Fourier curl and mean | Both projected physical columns are curl-free and have mean I. |
| Analytic material integrals and full-height inverse-rule controls at aspects .25, .8, 1.7 and 4 | Correct Fourier phase, background normalization, tensor frame and all five contact slots. |
| Actual-engine rotation, translation and uniform length scaling | Every returned diffraction power agrees for both polarizations, including unequal retained orders and zero-order axes. |
| JAX replay and two finite-difference steps | Independent period, geometry, material, background and rotation derivatives remain live. |
| Contact at zero and one-ULP height overrun, with and without rotation | The full one-sided gap derivative survives roundoff; a larger overrun is refused. |

The stationarity test varies a real quadratic polynomial using complex
steps to avoid cancellation; it is not a lossy-material corrector. Its
energy projection has independent Gauss controls and does not read the
assembled cell matrix when evaluating the variation.

Commands and results are in `studies/results/nvf_rectangular_cells_validation.log`.
The construction and metric suite passes 117 checks, the shared-kernel and
engine regression passes 205 with one historical expected failure, and the
documentation contract passes 30. Both manual examples and the strict
Sphinx build pass. A comparison with the
prior square-cell implementation at `7b52efd` gives bitwise identical
results for the original 700-unit fixtures. Across 36 square-cell cases,
the largest slot difference is 2.83e-10 in a degree-4/6 cell with period 2.1.
The reordered gap calculation moves its normalized input by 2.84e-14;
using identical normalized gaps makes the old and new coefficient arrays
bitwise identical. High-degree cell conditioning can amplify geometry
roundoff, so arbitrary rescalings are not promised to be bitwise identical.

These tests qualify the finite cell construction and engine integration.
They do not establish optical convergence for arbitrary aspect ratios or
geometries. The default cell resolution remains a starting point requiring
observable-specific refinement, and the provisional 1e-5 optical increment
is not a release gate. Curved interfaces, object edges not aligned with the lattice,
sheared cells and lossy corrector construction remain outside this builder.
