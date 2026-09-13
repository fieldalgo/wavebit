# Passive scalar NVF closure — experimental

Written: 2026-09-13.

Targets:
- `wavebit/_core/passive_nvf.py`
- `tests/test_passive_nvf.py`
- `studies/nvf_passive_closure.py`
- `studies/results/nvf_passive_closure.log`

The existing symmetrized NVF closure can have **negative constitutive dissipation**
even when every physical material is passive. The experimental closure below
removes that defect by construction and improves the recorded metal comparison.
It is **not selected by ordinary solves**: the same-material contact control gets
worse, and low-order dielectric accuracy is not uniformly improved. It is a usable
matrix-level candidate, not an accepted replacement or a completed convergence fix.

## Constitutive dissipation is stronger than one power-balance check

For a transverse constitutive matrix H, passivity requires
`Im_H(H) = (H-H^H)/(2i)` to be positive semidefinite. A negative eigenvalue
means some retained electric field has negative dissipated power. The incident
wave used in a particular solve need not excite that direction: an apparently
reasonable R and T do not establish the matrix property.

The existing closure uses the anticommutator of the compressed normal projector
G and the difference between the direct and inverse material rules. That map is
Hermitian for real materials but is not a positive map on dissipative matrices.
The tests include a two-mode counterexample with both material-rule losses at
least I. This is separate from the material-equality and eigenmode-derivative
fixes in [known limits](normal-field-known-limits.md).

## A normalized energy form

Let E be the direct rule, A the inverse rule, and G the 2N-by-2N Fourier
compression of the unit normal projector. Write Q=I-G and

```text
K = Q^H Q + G^H G
Z = K^(-1/2)
H = Z^H [Q^H diag(E,E) Q + G^H diag(A,A) G] Z.
```

Fourier compression of the Hermitian unit projector gives `0 <= G <= I`, hence
`I/2 <= K <= I`. The dissipative part of H is the sum of two congruences of
the input dissipative parts. Passive E and A therefore give passive H, including
materials with negative real permittivity. Positive input storage is preserved
by the same argument. Fourier compression preserves the sign of the imaginary
part of `1/epsilon`, and inversion reverses that sign, so the inverse rule A is
passive when the physical scalar material is passive and the inverse exists.

For uniform epsilon, E=A=epsilon*I and the middle expression is epsilon*K,
so H=epsilon*I. A constant normal makes G a projector and K=I, recovering the
rotated one-dimensional Li rule exactly. In the untruncated pointwise limit,
G is also a projector; this consistency statement is not an optical convergence
rate or permission to exchange a closing-gap limit with Fourier truncation.

Seven coupled Newton-Schulz steps evaluate Z on the bounded spectrum of K.
They require no eigenvectors or derivatives of square roots at zero eigenvalues
of G. The tests compare this iteration with an independent spectral inverse
square root and check JAX/Torch derivatives at both compressed and exact
constant-normal projectors. Additional checks cover complex64/complex128
uniform controls, positive storage, passive loss, and a rotated 1D rule.

The closure uses additional dense 2N-by-2N blocks and seven matrix iterations.
No runtime or memory advantage over the original closure is claimed.

## Finite optical comparison

CPU NumPy 2.5.1, float64/complex128, normal grid 512. The square cell is 700;
two 300-square boxes at x=-150,+150 touch at x=0. The layer is 300 thick at
wavelength 900, with top/bottom permittivities 1/2.25 and incidence
`theta=.2, phi=psi=0`. The normal field and material Fourier coefficients are
identical on both closures. Cartesian Li is a separate factorization of the
same exact rectangular painting, using the same Maxwell solver.

| materials | order | ordinary T | passive T | Li T |
| --- | ---: | ---: | ---: | ---: |
| 12.25, -10+i | 4 | 0.308557374 | 0.408313865 | 0.483946395 |
| 12.25, -10+i | 8 | 0.512798546 | 0.493954699 | 0.480976241 |
| 12.25, -10+i | 12 | 0.417550248 | 0.487007220 | 0.489085870 |
| 4, 9 | 4 | 0.922018277 | 0.927549629 | 0.921911793 |
| 4, 9 | 8 | 0.920840934 | 0.920659293 | 0.920259468 |
| 4, 9 | 12 | 0.919853005 | 0.919909990 | 0.919793951 |
| 4, 4+0.5i | 4 | 0.785156016 | 0.787750614 | 0.785943741 |
| 4, 4+0.5i | 8 | 0.785471956 | 0.785643240 | 0.785769408 |
| 4, 4+0.5i | 12 | 0.785791618 | 0.785772089 | 0.785757952 |

For the metal rows, the ordinary minimum dissipative eigenvalues are
-6.633, -4.603 and -1.226. The new values are nonnegative within 1e-14
roundoff. Both closures still give R+T below one for these particular waves;
that scalar test alone missed the old constitutive defect. The order-12
metal distance to Li decreases from about 0.07154 to 0.002079, but the finite
Li sequence is itself unconverged. This is not a validated optical error bar.

## Contact control and integration decision

Use the same geometry with both box permittivities 4, order 4, and open a
positive gap symmetrically. At normal grid 512:

| gap | ordinary T(g)-T(0) | passive T(g)-T(0) | Li difference |
| --- | ---: | ---: | ---: |
| 1e-5 | +4.48416e-6 | -7.89008e-5 | +6.57991e-11 |
| 1e-7 | +1.94328e-6 | -3.04579e-5 | +7.18314e-13 |
| 1e-9 | +8.37835e-7 | -1.14411e-5 | +7.16094e-14 |
| 1e-10 | +2.04281e-14 | +2.88658e-14 | +6.99441e-14 |

The last rung falls below the existing face-pair tolerance and is treated as
contact by the shared normal-field builder. It is not evidence that the
positive-gap derivative was resolved. The 128 grid also retains the contact
residual. The energy closure changes neither that geometry classification nor
the field's closing-gap behavior; on the resolved positive-gap rungs it worsens
this control. Therefore ordinary dispatch remains unchanged. A replacement
still needs a contact-consistent field/closure and independent optical
convergence, including dielectric and metallic held-out geometries.

Reproduce the separate controls from the repository root:

```sh
PYTHONPATH=. python studies/nvf_passive_closure.py --sections optical --orders 4 8 12
PYTHONPATH=. python studies/nvf_passive_closure.py --sections contact --orders 4
```

The result log contains source hashes for the kernel, field and instrument.
The explicit `matrices` helper returns single-layer `ConvMatrices`; its fixed
cell/background and nonoverlapping analytic-shape domain are the study's.
`passive_nvf_assemble` is an internal experimental matrix kernel, not a new
solver option or a public geometry adapter.

A separate [metal-capable finite contact family](nvf-pair-material-sector.md)
extends the existing exact gap cell using a rotated coercivity certificate.
It keeps passive energy and finite contact derivatives, but coarse optical
values and slopes still fail the comparison controls. It is a distinct field
construction, not a contact repair to this normalized-projector closure.
