# The off-diagonal constitutive slot (stage 4, C11): the registered gate's outcome

Written: 2026-10-27

Targets: `src/wavebit/fdtd/_prepare/coefficients.py`, `src/wavebit/fdtd/_prepare/constitutive.py`,
`src/wavebit/fdtd/_prepare/checks.py`, `studies/fdtd/subpixel_tilted_tensor/fdtd_subpixel_tilted_tensor.py`,
`studies/fdtd/subpixel_mie_cylinder/fdtd_subpixel_mie_cylinder.py`,
`studies/fdtd/subpixel_coupled_cylinder/fdtd_subpixel_coupled_cylinder.py`,
`studies/fdtd/subpixel_constitutive/fdtd_subpixel_constitutive.py`.

## TL;DR

**C11 is not written, and it is not closed by its registered rule either: the rule's
premise is not met, so the decision is left to the owner.** No code changes with this record.

The rule, registered before studies S1 and S2 were run: the off-diagonal constitutive slot
(C11, scoped to stage 5) **is never written if S1 and S2 show that the off-diagonal terms buy
no order at a silicon-to-air contrast**, and the measured diagonal-projection price then goes
into the docstring. S1 says that on a tilted **flat** interface. S2, the 2-D
Mie cylinder, says the opposite on a **curved** one: the diagonal rule misses the order in the
TE arm at contrast 12, and on the glass sphere restoring the cross terms bought it back
(1.714 to 2.023). Those cross terms are already in the production engine, added during C3's
work rather than under a C11 commit. What C11 would still add, and whether it should be
written, is unmeasured; the questions are listed below.

## The evidence the rule reads

**S1, the tilted flat interface** (`studies/fdtd/subpixel_tilted_tensor/`,
[record](fd-subpixel-occupancy.md#s1-results-and-what-they-decide)). At contrast 12, with
16/32/48 cells per period, the diagonal rule's fitted orders are 0.999, 1.012 and 0.864 at
14.036, 26.565 and 45 degrees. The self-adjoint full-tensor comparator reads 0.929 at the first
tilt and is non-monotone at the other two (descriptive fits of 0.387 and -2.452). The record's
verdict: "the sampled tensor alternatives do not establish a robust general improvement in
order". **S1 says no.**

**S2, the 2-D Mie cylinder, TM then TE** (`studies/fdtd/subpixel_mie_cylinder/`,
[record](fd-mie-validation.md#measured-cylinder-results)). With the diagonal rule, eight
placements at 20/28/40/56 cells per wavelength:

| Case | Mean fit: order / scatter |
| --- | ---: |
| TM, epsilon 1.2 | 2.058 / 6.09% |
| TM, epsilon 12 | 2.119 / 3.04% |
| TE, epsilon 1.2 | 1.827 / 4.46% |
| TE, epsilon 12 | -0.888 / 48.25% (non-monotone, a failed fit) |

TM, which needs no interface normal, reads second order, as its role as the warm-up required;
TE at contrast 12 does not converge in the sampled range. **S2 says the diagonal rule misses
the order on a curved, high-contrast interface** -- TE failing where TM converges is the
pattern the design attributed in advance to the missing off-diagonal terms.

**The cross terms on curved interfaces** ([record](fd-mie-validation.md)):

| Instrument | Diagonal only | With the cross terms |
| --- | ---: | ---: |
| Glass sphere, controlled comparator, 20/28/40 cells per wavelength | 1.714 | **2.023** |
| Production spheres at epsilon 1.1 / 2.25 / 12 | -- | 2.075 / 2.023 / 1.827 |
| Production cylinder, TE, epsilon 12, eight placements, 40/56/80 | (fails, above) | **1.603** (fails 1.8) |
| Symmetric edge/face research comparator, same cylinder | -- | 1.825 |
| Extended stencil (`discretisation='stencil'`), same cylinder | -- | 2.138 |

So the off-diagonal terms buy an order on the glass sphere. On the high-contrast cylinder the
production writing of them improves the diagonal rule's failed fit but stays under 1.8; only
the extended stencil passes there.

## What the rule decides, and what it does not

The rule's "never written" branch requires S1 **and** S2 to say no. S2 does not, so **C11 is
not closed by the registered rule**. The opposite branch registered with it (open the slot in
stage 5, with a self-adjoint writing and a non-Hermiticity gate) was partly realised early:
`src/wavebit/fdtd/_prepare/constitutive.py` and the kernel's constitutive term integrate D and B
in compact regions around smoothed bodies and add adjoint cross coefficients at common nodes
(see [production constitutive coupling](fd-mie-validation.md#production-constitutive-coupling-and-energy)).
That realisation differs from the C11 sketch in carrying a D/B state field, which the sketch
avoided, and it covers **isotropic** phases only.

Under the owner's instruction for this outcome, nothing is built here. The diagonal rule's
measured price is already in `coefficients.mix`'s docstring (the S1 orders above).

## What is missing for the owner's decision

1. **Scope.** Is C11 (a) closed as realised by the C3-era cross coupling, or (b) re-scoped to
   what is still refused, an **anisotropic material** inside a smoothed cell (door 4 of
   `checks.check_smoothed`)? No study has measured (b): S1 and S2 both used isotropic phases.
2. **The high-contrast cylinder.** The voxel path with production cross terms still fails the
   1.8 line on the TE epsilon-12 cylinder (1.603); whether another off-diagonal writing would
   pass it is unmeasured. The extended stencil already passes it.
3. **The gate registered for C11.** A non-Hermiticity assertion below 1e-12 was C11's
   own gate. The production coupling is self-adjoint by construction (adjoint averages) and is
   checked by an inhomogeneous adjoint identity and a dense finite-operator test; whether that
   counts as C11's gate is the owner's call.
