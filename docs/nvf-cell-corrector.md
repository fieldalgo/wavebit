# Separating cell error from optical and contact error

The harmonic-energy candidate now has two independent discretizations of its
static auxiliary cell problem: Fourier Galerkin and conforming two-dimensional
FEM. This investigation substantially reduces the **cell** uncertainty while
also finding a new, serious **s-polarized contact derivative failure**. It does
not qualify the candidate for production NVF.

The later [conforming thin-gap formulation](nvf-thin-gap-energy.md) now passes
the rectangular contact derivative checks in both polarizations. It addresses
the missing moving-layer term with a different cell space; the fixed-geometry
FEM export and finite Fourier failure documented here are unchanged.

The instruments are `studies/nvf_cell_bounds.py` and
`studies/nvf_cell_corrector.py`. The latter uses optional NGSolve and exports
Fourier coefficients; the former applies the same energy closure and Wavebit
scattering routine to that export. No three-dimensional Maxwell FEM refinement
is needed for this auxiliary calculation.

## A measurable enclosure for the static cell

For a real positive scalar material, periodic curl-free fields B of mean I give
the primal energy tensor `U = average(B.T epsilon B)`. Solve the analogous cell
problem for reciprocal material `rho=1/epsilon`, obtaining energy tensor V.
Rotating its curl-free trial field by 90 degrees gives an admissible
divergence-free flux. With rotation R, the complementary bound is
`L = R inv(V) R.T`. The exact static homogenized tensor lies between L and U in
the positive-semidefinite ordering.

This primal/complementary construction and exact-integration Fourier Galerkin
framework are described by [Vondřejc, 2014](https://arxiv.org/abs/1412.2033).
The reported enclosure is evaluated in floating-point arithmetic; it is not an
interval-arithmetic certificate. More importantly, it is **not an enclosure for
optical T**, nor a bound on every entry of the finite-order constitutive matrix.
It diagnoses the auxiliary field separately from its small linear residual.

The relative width below is the largest generalized eigenvalue of `(U-L, L)`
for the tilted contact fixture:

| cell discretization | relative energy enclosure width |
| --- | ---: |
| Fourier H=31 | 0.05491477823 |
| Fourier H=63 | 0.02625956204 |
| Fourier H=127 | 0.01286227328 |
| Fourier H=255 | 0.00638098156 |
| Fourier H=511 | 0.00318067546 |
| FEM h=0.06, degree=6, vertex refinement=20 | 0.00004083213 |
| FEM h=0.06, degree=8, vertex refinement=80 | 0.00000291024 |

Thus the last FEM enclosure is about 1,100 times narrower than the H=511
Fourier enclosure. This is an improvement in a **static error diagnostic**, not
a 1,100-fold optical accuracy or speed claim. The highest FEM cell calculation
has 1,396 triangles and 44,673 unknowns per scalar problem including its
mean-zero constraint. This is much smaller than the earlier 3D Maxwell solve.

The FEM geometry uses clipped and united periodic rectangles, explicit
identification of both lateral axes, and material-area checks. In two dimensions
OCC face names supply material labels; using a volume material tag left the
initial control labeled `default`, and the area guard rejected that attempt.
The implemented version uses face names and passes both uniform and bar controls.
NGSolve documents its [periodic scalar spaces](https://ngsolve.org/ngsolve/docs/i-tutorials/unit-2.12-periodicity/periodicity.html).

## Fourier projection without a permittivity raster

On each conforming triangle, Duffy-mapped Gauss quadrature integrates the four
B components, three symmetric energy-field components, and epsilon against
Fourier waves. Those eight coefficient arrays supply Q, G, and the longitudinal
slot of the unchanged energy closure. Point batches limit temporary memory.

The unmodulated energy integral agrees with a separate NGSolve integral to
1.83e-13 in the final export. That control alone does not resolve the Fourier
oscillations: at coefficient cap 44, increasing quadrature from 20 to 24 points
per Duffy coordinate changes some coefficients by 4.38e-9. Increasing from 24 to
28 changes them by 3.37e-13. At 28 points, the projected epsilon coefficients
agree with the analytic rotated-rectangle transform to 1.78e-15.

At the degree-8/refinement-80 cell setting and quadrature 28, the tilted optical
results are approximately:

| Maxwell outer order | T |
| --- | ---: |
| 6 | 0.855485746686 |
| 10 | 0.857473156196 |
| 14 | 0.860355871708 |
| 18 | 0.861820093322 |
| 22 | 0.862617122442 |

At outer orders 6/10/14, changing the cell from degree 6/refinement 20 to
degree 8/refinement 80 changes T by about 1.4e-5. The remaining outer-order
change from 18 to 22 is much larger, 0.000797029. The n=22 value differs from
the earlier H=511 Fourier-corrector value by 0.00140120 and from the preliminary
3D FEM volume value by 0.00343878. Neither the finite outer sequence nor that
independent 3D reference is declared converged.

For the axis-aligned closed bar, this conforming-field energy closure gives
T=0.627676664411 at n=6 and reproduces the analytic inverse-rule constitutive
matrix. This removes the Fourier corrector's finite-H closed-value bias on this
control. All three fixtures use the same cell solve and energy construction;
the inverse rule is an independent check, not a rectangular dispatch branch.

## The s-polarized contact failure changes the acceptance decision

The previous approximately 0.54% contact-derivative discrepancy was measured
only for **p polarization**, with electric field tangential to the vanishing
horizontal gap. That statement must not be generalized to s polarization.

At n=7, the s-polarized closed-bar T is 0.878242289574 in both the current Fourier
candidate and the independent integrated Cartesian Li instrument. Nevertheless:

| contact derivative measurement | dT/dgap |
| --- | ---: |
| candidate JAX, H=31 | -0.00006355713415 |
| candidate JAX, H=127 | -0.00006355713415 |
| candidate JAX, H=255 | -0.00006355713415 |
| independent Li positive secant, step 1e-4 | -0.00077830402567 |
| independent Li positive secant, step 1e-5 | -0.00077854585001 |

The candidate's derivative is about 12.25 times too small, despite matching the
closed value and its own finite differences. Refining H does not remove this
failure at exact contact. It is retained as a strict expected-failure research
gate; production's original split/merge expected failure also remains.

The ratio has a physical explanation. For a thin planar gap in a host, the
tangential electric field is continuous but the **normal displacement** is
continuous. If host and gap permittivities are `eh` and `eg`, differentiating a
series laminate at zero gap gives the local normal constitutive variation
`eh*(eg-eh)/eg`, while a smooth finite-field material perturbation gives only
`eg-eh`. Here `eh/eg=12.25`. A shrinking gap carries an O(1) normal-field change
on an O(gap) region; its squared field contribution remains first order in gap.
A fixed smooth Fourier basis cannot capture that contribution by taking the
gap derivative before refining the basis. This is a noncommuting limit, not a
CG stopping-tolerance issue.

A correction must enter the forward formulation through the thin-layer field
or an equivalent interface contribution. Multiplying the reported gradient by
12.25 would not fix the forward model or its autodiff consistency. The new FEM
instrument supplies fixed-geometry fields and does **not** implement a moving
contact derivative, so its improved cell convergence does not close this gate.

There is also a much smaller, separate derivative limitation: on the translated
440x270/epsilon-6.25 test at n=2/H=7, Wavebit's existing regularized eigenvalue
gradient gives s-polarized dT/depsilon=0.00619276168 versus the secant's
0.0061927238, a bias of about 6 ppm. The native nondegenerate eigenvalue derivative
removes that bias and passes the same strict test. Both results are retained;
the production eigenvalue regularization and the test tolerance were not changed.
This numerical issue does not explain the roughly 92% contact-derivative error.

The later [scattering-doubling implementation](nvf-scattering-doubling.md)
passes that same derivative gate without using any eigenvector derivative.
It also handles exact layer graze and retains both independent contact
checks when paired with the conforming polynomial-body cell. It is an
opt-in research path; the old Fourier contact failure and production
regularization documented above remain unchanged.

## Reproduction

From the NVF worktree, use the ordinary Wavebit environment for Fourier cell
bounds and the optional NGSolve environment for the conforming export:

Set `NVF_CACHE` to an existing scratch directory for generated projection files.

Set `NVF_FEM_PYTHON` to the Python executable with the optional NGSolve dependencies.

```bash
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 python -m studies.nvf_cell_bounds --inners 31 63 127
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 "$NVF_FEM_PYTHON" -m studies.nvf_cell_corrector --cap 44 --degree 8 --quadrature 28 --refine 80 --output "$NVF_CACHE/nvf_cell.npz"
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 python -m studies.nvf_cell_bounds --projection "$NVF_CACHE/nvf_cell.npz" --orders 6 10 14 18 22
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 python -m studies.nvf_harmonic_energy_jax --inners 31 127 255 --polarizations s
PYTHONPATH=$PWD python -m pytest -q tests/test_nvf_cell_bounds.py tests/test_nvf_harmonic_energy.py tests/test_nvf_harmonic_energy_jax.py
PYTHONPATH=$PWD "$NVF_FEM_PYTHON" -m unittest discover -s tests -p test_nvf_cell_corrector.py
```

Selected outputs are in `studies/results/nvf_cell_bounds.log`,
`studies/results/nvf_cell_projection.log`, and
`studies/results/nvf_harmonic_energy_s.log`. The H=255/511 enclosure rows and
initial FEM refinement comparison came from the preceding local prototypes;
the final degree-8/quadrature-28 projection uses the packaged instrument.
