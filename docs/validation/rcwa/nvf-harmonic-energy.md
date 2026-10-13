# Material-defined energy factorization study

`studies/rcwa/nvf_harmonic_energy/nvf_harmonic_energy.py` is a new, unaccepted research candidate for the
representation and convergence problems. It has **not** been installed in the
production NVF dispatch. It does not select Cartesian Li for rectangles.

**The expanded contact check now fails for s polarization.** Its derivative is
about 12.25 times too small and does not improve with inner order. The earlier
0.54% discrepancy applies only to p polarization. A new conforming 2D cell
instrument reduces the auxiliary-field uncertainty substantially, but does not
solve that moving-contact limit. See the
[cell and polarization follow-up](nvf-cell-corrector.md).

A subsequent [conforming thin-gap study](nvf-thin-gap-energy.md) retains the
missing normal-field energy and passes both contact polarization gates on the
rectangular control. It uses a different cell space with live JIT derivatives;
the Fourier candidate below and its expected failure remain unchanged. The
new study has not yet established arbitrary primitive-partition invariance.

The next [material-jump extension](nvf-partitioned-gap-energy.md) combines the
conforming contact field and partition invariance for horizontal rectangle
tilings, including live internal contrasts. Arbitrary geometry is still open.

## Construction

Instead of retaining a primitive's artificial seams, two electrostatic cell
problems are determined by the exact material Fourier series:

```
div(epsilon (e_j + grad chi_j)) = 0
B_j = e_j + grad chi_j
```

The constant potential is fixed; the remaining scalar Fourier modes are solved
by conjugate gradients with a Laplacian preconditioner. Exact rectangle
coefficients and zero-padded convolution form this operator. A subpixel gap
therefore changes the material coefficients without a pixel threshold.

The transverse electric energy is expressed in the retained physical Fourier
coordinates using the Gram matrix:

```
Q = C(B)
G = C(B.H epsilon B)
epsilon_effective = inv(Q).H G inv(Q)
```

For positive real epsilon, G is Hermitian positive. This preserves lossless
Hermiticity in the constitutive closure. The longitudinal slot remains the
inverse direct material matrix. The study then calls Wavebit's layer eigensolve
and matches the usual homogeneous exterior modes. This is a proposed
constitutive closure, not a proof that two electrostatic correctors give the
best dynamic Maxwell basis at every finite truncation.

The related use of exact material integration and trigonometric Galerkin cell
problems is described by [Vondrejc, 2014](https://arxiv.org/abs/1412.2033). That
paper's scalar homogenization bounds are **not** bounds on the optical
transmission of this new closure. No external solver implementation was copied.

There are two real truncations: outer Maxwell order n and inner corrector order
H. The FFT evaluation grid is a third parameter only until it exceeds
`4*(H+n)`. Above that threshold, the retained Gram integrals are exact for the
chosen trigonometric corrector and exact piecewise material. Increasing H
changes the approximation; increasing an already adequate evaluation grid does
not. No positivity clipping or normal-field smoothing is applied.

## Measurements and limitations

The 700-square, 900-wavelength, 300-thick fixtures use theta=0.2 and p
polarization. The split fixture is two abutting epsilon-4 300x300 rectangles;
the merged fixture is one 600x300 rectangle. The gap fixture is the epsilon-12.25
350-by-(700-gap) bar, at outer order 7. The independently checked Li instrument
has gap slope approximately 0.00189156 and closed-bar T=0.626061531897 at that
same order.

| inner H | split-minus-merged T, n=3 | closed-bar T, n=7 | gap slope, step 1e-6 | resonant T, n=10 |
| --- | ---: | ---: | ---: | ---: |
| 15 | -5.21e-14 | 0.595114603959 | 0.002040114744 | 0.917691942820 |
| 31 | 5.59e-13 | 0.612500546747 | 0.001969603813 | 0.885300604895 |
| 63 | -5.57e-13 | 0.619488391427 | 0.001931713567 | 0.870597941843 |
| 127 | -1.31e-14 | 0.622779529439 | 0.001912010994 | 0.863796029117 |
| 255 | -1.72e-14 | 0.624419298425 | 0.001901788615 | 0.860559945999 |

The positive result is simultaneous representation equality, lossless energy,
and a p-polarized gap derivative approaching the independent reference. But the closed-bar
forward value is not exact at finite H, and the resonant inner-order variation
is still material. **Neither original NVF requirement is closed by these rows.**

At H=127, the resonant T values at outer orders 6/10/14 are
0.860575347 / 0.863796029 / 0.866562975. The last is close to the preliminary
Rayleigh FEM volume result 0.866055904, but H=255 already moves the n=10 result
by 0.00323608. A single close pair is not convergence. The independent FEM
status and the rejected short-PML values are recorded in
[`nvf-independent-fem.md`](nvf-independent-fem.md).

At H=255, outer orders 6/10/14/18/22 give T=
0.858008157 / 0.860559946 / 0.863397318 / 0.864755168 / 0.865461380.
The n=22 value differs from the preliminary h=0.18, p=4 FEM volume readout by
0.000594524. **The higher inner-order check does move it:** at n=22, H=511 gives
T=0.864018318441, a change of 0.00144306203. Thus the initial close pair cannot be
called convergence. Relative to the preliminary FEM value, the new result's
difference is about 0.00204, compared with about 0.01230 for circular NVF at n=22.
This is an improvement on this fixture, with remaining numerical error; it is
not a completed replacement for the current NVF implementation.

The field is material-defined and differentiable as a finite linear system.
The opt-in JAX study now tests first derivatives of that finite model (below).
This does not establish a production backend contract or a Torch implementation.
The band-limited corrector also does not enforce exact pointwise interface jumps.
Its inner convergence, contact limit, dynamic convergence, painted geometry,
lossy/magnetic media, and production backend contracts must be resolved before
adoption.

An earlier full H(curl) Galerkin prototype using these corrected electric basis
functions, with the corresponding variational exterior coupling, was also
tested. It conserved energy but at n=6 changed T from 0.847516652 to 0.825436908
when H changed 31 to 63. Its contact slopes did not approach the Li comparison
in the tested range. That full-basis variant was not adopted either; the retained
study is the energy constitutive closure above.

## Implicit differentiation of the finite candidate

`studies/rcwa/nvf_harmonic_energy_jax/nvf_harmonic_energy_jax.py` keeps rectangle dimensions, angle, position,
and real permittivity live while the three discretization sizes remain fixed.
The cell solution is differentiated implicitly using
[`jax.lax.custom_linear_solve`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_linear_solve.html).
Each forward and transpose solve checks its actual residual; failure yields
NaN under JIT and the measurement command refuses nonfinite data. The complex
Hermitian operator is **not** declared transpose-symmetric. This matters for
translated rectangles, whose material coefficients are complex. Scattering
uses the same study routine for NumPy and JAX and Wavebit's existing eigenmode
derivative. No cell-iteration history is differentiated.

The material FFT is now computed once per cell problem and reused in every
matrix-vector product. This changes the cost, not the discretization. The
original uniform-slab, split, exact-product and failed-solve checks still pass.

At outer order 7, the exact-contact derivatives from JAX are:

| inner H | JAX T at gap=0 | JAX dT/dgap at gap=0 | independent NumPy positive secant, step 1e-4 |
| --- | ---: | ---: | ---: |
| 31 | 0.612500546747 | 0.001969613266 | 0.001969620802 |
| 63 | 0.619488391427 | 0.001931692175 | 0.001931700341 |
| 127 | 0.622779529439 | 0.001911989676 | 0.001911997762 |
| 255 | 0.624419298425 | 0.001901820104 | 0.001901829385 |

NumPy/JAX T differs by at most 1.10e-13 in these rows. At H=255 the derivative
differs from the independent Li contact instrument's approximately 0.00189156
by 0.54%; that is **remaining discretization error**, not a JAX derivative error
and not an accepted contact tolerance. Gap=0 is the right derivative of the
documented nonoverlapping geometry; negative gaps are not used as a test.

A translated 440x270 rectangle with angle=0.31, epsilon=6.25, at n=2/H=7 gives
T=0.988388160136 and derivatives with respect to `(lx, ly, angle, epsilon)`:
`(0.000217965944, 0.000317737805, -0.032707833248, 0.008461345282)`.
Each agrees with independent NumPy central differences at two step sizes.
Other tests check that moving an artificial equal-material seam changes neither
the value nor its physical parameter derivatives, and that the permittivity
derivative remains live at zero contrast. A separate failure test has a
converged forward solve and an unconverged adjoint, and requires a nonfinite
gradient. The original six JAX tests and four candidate tests passed. The
expanded suite now records strict expected failures for the s contact physics
and the much smaller s gradient bias from existing eigenvalue regularization;
the native nondegenerate eigenvalue control passes the unchanged gradient test.

These checks cover first reverse derivatives on a fixed discretization. They
do not claim higher derivatives, production frozen geometry, lossy media, or
correct derivatives of the yet-unconverged Maxwell limit. The original
production split/merge diagnostic remains unresolved.

## Reciprocal-material flux variant: rejected contact derivative

A dual trial was checked to see whether representing electric displacement
would reduce the inner truncation error. It solves the same two correctors in
`rho=1/epsilon`, rotates their curl-free field by 90 degrees to obtain a
divergence-free flux basis F, and uses

```
Q = C(rho F)
G = C(rho F.H F)
epsilon_effective = inv(Q).H G inv(Q)
```

Constant column rotations and normalizations cancel in this congruence. The
longitudinal slot still uses the original direct epsilon matrix. Exact Fourier
product integration is used, with the same adequate grid criterion.

This variant gives the closed-bar n=7 value 0.626061531897 already at H=15,
matching the independent Li instrument. But its positive 1e-6 gap slopes at
H=15/31/63/127 are 0.0140110158 / 0.0140109736 / 0.0140110495 /
0.0140110449, about 7.4 times the independently checked 0.00189156. The exact
closed value therefore does not make its contact derivative correct. Its
resonant n=10 T also moves strongly, through 0.712083547 / 0.802017271 /
0.838896268 / 0.848265979. This trial is rejected; it is not installed in the
candidate or production dispatch, and it supplies no reference value.

## High-mode electrostatic condensation: exploratory only

A separate trial minimizes electric energy over gradient modes outside the
retained Maxwell box. With D the high-mode gradient and C the exact material
matrix, the retained transverse matrix is the Schur complement
`C_low - C_low,high D inv(D.H C_high D) D.H C_high,low`. High scalar modes are
Laplacian-preconditioned; low scalar modes are excluded rather than allowed to
change the retained electric coordinates. Each retained electric column needs
its own cell solve, so this is substantially more expensive than two correctors.

The n=3 closed-bar T values at H=7/15/31 are 0.602171643 / 0.635706685 /
0.652768616, versus the independent n=3 value 0.669377786. Tilted n=6 T is
0.934219637 / 0.898950973 / 0.878052788. True cell residuals are below 1e-12 and
energy errors below 2e-13 in these tilted runs. These few inner orders do not
establish a converged or faster alternative. The trial is not installed in the
candidate or production dispatch and has no optical error-bound claim.

## Reproduction

Run in the existing Wavebit environment, from the NVF worktree:

```bash
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 python studies/rcwa/nvf_harmonic_energy/nvf_harmonic_energy.py --inners 31 63 127 --orders 6 10 14 --gates
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 python -m studies.nvf_harmonic_energy_jax --inners 31 63 127 255
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/rcwa/test_nvf_harmonic_energy.py tests/rcwa/test_nvf_harmonic_energy_jax.py
```

Instrument tests check the candidate matrices and scattering against the exact
uniform slab, split-material transforms and transmission, invariance to an
adequate integration grid, rejection of an aliasing grid, and rejection of a
failed corrector solve. They do not turn the unresolved convergence findings
into passing physics gates. Selected packaged output is retained in
[`studies/rcwa/nvf_harmonic_energy/records/nvf_harmonic_energy.log`](../../../studies/rcwa/nvf_harmonic_energy/records/nvf_harmonic_energy.log).
The H=15/63 rows above come from the local prototype; the H=127/255 rows use the
packaged study. The latter uses a smaller adequate evaluation grid and Wavebit's
layer eigensolve. Forward values match the prototype to about 1e-12; the 1e-6
secants amplify the remaining solver roundoff.
The newer implicit-derivative measurements and independent NumPy secants are
in [`nvf_harmonic_energy_jax.log`](../../../studies/rcwa/nvf_harmonic_energy_jax/records/nvf_harmonic_energy_jax.log).
