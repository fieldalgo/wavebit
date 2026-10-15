# A conforming forward formulation for the vanishing gap

`studies/rcwa/nvf_thin_gap_energy/nvf_thin_gap_energy.py` resolves the missing normal-field contribution
on the centered rectangular contact control. It evaluates an actual periodic
Galerkin cell problem and the existing energy Gram closure. The resulting
contact derivative passes the independent p and s comparison gates under JAX
JIT. It is an **opt-in study for one axis-aligned rectangle**, not a production
NVF method or a solution for arbitrary primitive partitions and tilted contacts.

The subsequent [partitioned-gap extension](nvf-partitioned-gap-energy.md) now
combines this contact field with equal-material partition invariance, live
internal material/seam derivatives and translated horizontal rectangle tilings.
It retains this cell space and leaves the general-geometry acceptance work open.
The further [harmonic-coordinate study](nvf-harmonic-coordinates.md) changes
the x basis and affine lift to resolve closed real-interface motion on a fixed
mesh. It retains the contact field but still has a positive-gap seam error.

The previous Fourier-corrector implementation and its strict s expected-failure
test remain unchanged. Its fixed smooth space still misses the thin-layer term;
this experiment uses a different, conforming cell space.

## Retaining a field on a region of vanishing measure

Normalize the period to one and write `g = gap / 700`. The material body has
height `1-g`, with permittivity depending on x only. The periodic gap has
permittivity one and straddles the top/bottom boundary. In body coordinate
`t = y/(1-g)`, use the potential

```
chi_body(x,t) = v(x,t) - g*t*a(x),       -1/2 <= t <= 1/2
chi_gap(x,s)  = v(x,-1/2) + g*s*a(x),   -1/2 <= s <= 1/2
```

Here v is a periodic tensor-product Q1 function on the body, and a is a
continuous piecewise-linear function on x. The two potentials match at both
interfaces. At every positive g this is an ordinary conforming periodic H1
space; the potential jump across the gap is `g*a`. Its normal gradient in the
gap is a, which remains finite as g vanishes. Its electric energy therefore
contains a first-order term `g * integral(a^2)`. That term was absent when the
old finite Fourier space was differentiated at exact contact.

Let M and K denote weighted one-dimensional mass and stiffness matrices. The
enrichment stiffness is

```
A_aa = g*M_bg + g^2*(M_eps/(1-g) + (1-g)*K_eps/12) + g^3*K_bg/12
A_va = g*C
```

Divide the enrichment **test equation** by g algebraically before evaluating
the linear system. The implemented system is

```
[ A_vv    g*C       ] [ v ] = [ b_v   ]
[ C.T     A_aa/g    ] [ a ]   [ b_a/g ]
```

Every displayed quotient is expanded analytically in the implementation. There
is no numerical division by zero, small-gap cutoff, stopped geometry gradient,
or prescribed optical derivative. One ordinary potential DOF fixes the gauge.
The gap quadrature retains its field at zero width; its integration weight
vanishes continuously and contributes to the derivative.

At exact contact under a uniform normal forcing, the second equation becomes
`M_bg*a = load(eps-1)`. The gap field `1+a` thus approaches the host/gap
permittivity ratio as the x mesh resolves the material step. The contrast
factor follows from the variational solve. It is not fitted to transmission.

Cell stiffness, mass and load integrals are assembled analytically on material
intervals. Gauss quadrature is used only to project B, `B.T*epsilon*B`, and
epsilon onto Fourier coefficients. The transverse closure remains
`inv(C(B)).H @ C(B.T*epsilon*B) @ inv(C(B))`; longitudinal factorization and the
shared Maxwell scattering instrument are unchanged. No Cartesian Li function
is called by this model.

## Measurements and numerical controls

For the existing 700 nm period, 350 nm width, epsilon 12.25 control at outer
order 7, the closed transmissions are p = 0.626061531897 and
s = 0.878242289574. The conforming closure recovers both independent Cartesian
constitutive matrices, accounting for the different Fourier coordinate origins.

The independent positive-gap secants, extrapolated to zero, give approximately
`dT_p/dgap = 0.001891546` and `dT_s/dgap = -0.000778575`, in inverse nm.
The exact-contact JAX derivatives from the conforming study are:

| x interface refinement fraction | p derivative | s derivative |
| --- | ---: | ---: |
| 1e-4 | 0.001891546100 | -0.000777803964 |
| 1e-5 | 0.001891546100 | -0.000778495983 |
| 1e-6 | 0.001891546100 | -0.000778566984 |

The last s error is about **0.001%**, compared with roughly 92% for the old
finite Fourier contact derivative. This is a finite-order, single-fixture
comparison, not an error bound for arbitrary optical configurations.

The x nodes are graded near material interfaces. Body y nodes also need
refinement near the gap: with only eight uniform body intervals, the exact
endpoint derivative improves but positive-gap secants approach it slowly.
Adding the graded y nodes addresses that resolution problem. At x fraction
1e-6, the recorded run has 16 body intervals and 611 scalar unknowns. The
s JAX derivative at gaps 1e-4 and 1e-5 nm is respectively -0.000776638606 and
-0.000778356234, approaching the endpoint value. These finite-gap derivatives
are not claimed equal to the independent Li model at every positive gap.

The mean static normal-energy derivative has a separate analytic check:
`-epsilon*(epsilon-1)*(width/period)/period`. Its absolute errors decrease from
1.96e-6 to 2.00e-7 to 2.01e-8 under the same x refinements. This verifies the
normal-field mechanism without relying on the optical eigensolve.

The tests also compare the scaled formulation against an independently
assembled, unscaled periodic nodal Q1 solve at positive gaps. They agree in all
eight projected coefficient fields. JIT derivatives of gap, material and width
agree with NumPy finite differences, including a live material derivative at
zero contrast. A positive-gap derivative ladder approaches the exact-contact
JIT derivative. Both polarizations conserve lossless power.

The smallest x elements increase conditioning costs: the recorded finest
scaled linear residual is about 1.3e-10. The study rejects nonfinite residuals
or residuals above 1e-8; that threshold is not an estimate of optical error.
Fourier quadrature must be checked separately for new orders and geometries.
An initial 8-point quadrature control was insufficient and was not accepted;
the tests compare 12 and 16 points without relaxing their accuracy criterion.
For the recorded order-7/fraction-1e-6 setup, all projected coefficients change
by at most 1.40e-13 from 12 to 16 points, at both zero and 1e-4 nm gap.

## Scope still open

This model supports a positive real scalar rectangle with live width, height
through gap, and permittivity on fixed conforming topology. It has no primitive
painting/split adapter, tilted or curved interfaces, magnetic/lossy materials,
or production backend integration. Therefore the earlier material-defined
Fourier model's split invariance cannot be attributed to this new mesh model.
Combining interface conformity with finite-order representation invariance is
still required. Tilted optical convergence and the independent 3D reference
uncertainty are also unchanged. The separate production eigenvalue-gradient
regularization limitation remains recorded in the earlier tests.

## Reproduction

```bash
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 python -m studies.rcwa.nvf_thin_gap_energy.nvf_thin_gap_energy --rows 8 --edge-fraction 0.000001 --gaps 0 0.0001 0.00001 --autodiff
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 python -m studies.rcwa.nvf_thin_gap_energy.nvf_thin_gap_energy --rows 8 16 --edge-fraction 0.00001 --gaps 0 0.00001 --autodiff
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/rcwa/test_nvf_thin_gap_energy.py
```

The executable supports NumPy alone; `--autodiff` opts into JAX x64. Use
`--uniform-y` to reproduce the coarser body control. Recorded outputs are in
`studies/rcwa/nvf_thin_gap_energy/records/nvf_thin_gap_energy.log`.

Validation on 2026-09-08: **12 new study tests pass**. The combined cell,
harmonic-energy, resonant-instrument and original split diagnostic selection
gives **47 passed, 5 skipped, 3 xfailed**. The skips require optional NGSolve;
the existing production split failure, Fourier s contact failure, and small
regularized eigen-gradient failure remain strict expected failures. Targeted
Ruff and `git diff --check` pass. No production solver or tolerance was changed.
