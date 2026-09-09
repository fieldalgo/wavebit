# Conforming NVF with a common passive loss ratio

Written: 2026-09-09

Targets:
- `wavebit/_core/conforming/__init__.py`
- `wavebit/_core/conforming/partitioned_gap.py`
- `tests/test_conforming_common_loss.py`
- `tests/test_conforming_nvf.py`

The implementation measured on this page is the common-factor extension at
revision `6b0a60b`. The later [complex-material extension](nvf-complex-material.md)
removes the common-ratio restriction. The derivation below remains an exact
control for that general construction.

At that revision, `ConformingNVF` accepts passive complex permittivities with positive real
parts when all materials in a patterned layer share
`eta = Im(epsilon)/Re(epsilon)`. This includes the background, both outside
and between rectangles. Different layers may use different ratios. Uniform
layers can have independent positive-real-part passive permittivities.
The [manual](guide/conforming-nvf.md) gives the current input contract and
a JAX example with independently varying inclusion loss.

## Factoring the cell equation and energy closure

Write the material throughout one patterned layer as

```text
epsilon(x) = alpha * epsilon_storage(x)
alpha = 1 + i*eta, eta >= 0
epsilon_storage(x) = Re(epsilon(x)) > 0
```

Because alpha is constant and nonzero, it cancels from
`div(epsilon grad(phi))=0`, including continuity of normal displacement.
The corrector is therefore the existing real storage-material corrector.
The same aspect metric, gap enrichment, material grading, translation and
layer-axis mapping apply. No complex basis or alternate factorization is
substituted for this case.

For the projected field Q, energy G and longitudinal material Z,

```text
Q = Q_storage
G = alpha * G_storage
Z = alpha * Z_storage
E = Q^(-H) G Q^(-1) = alpha * E_storage
Z^(-1) = Z_storage^(-1) / alpha
```

Thus `epx`, `epy`, `epxy` and `epyx` multiply by alpha; `epz_i` divides by
alpha. The adapter closes the physical storage matrices and applies these
factors afterward. Its 1e-9 Hermitian guard still checks the storage
transverse matrices, since the physical lossy matrices need not be
Hermitian. It then checks that the scaled slots are finite.

For Hermitian positive storage E, the dissipative part is
`(E_lossy-E_lossy.H)/(2i) = eta*E_storage`, which is positive semidefinite.
The independent tests check this matrix property as well as actual-engine
absorption. Scaling the inverse longitudinal slot in the wrong direction
would violate the material and contact controls.

## Live parameters and refusal rules

The adapter constructs relative cell materials from real-part ratios,
`Re(epsilon_i)/Re(background)`. This avoids spurious cell derivatives from
cancelling a common factor in complex division. The loss-ratio comparison
remains live and accepts only rounding differences within
`128*machine_epsilon*(1+abs(eta_background))`. Gain, nonfinite values,
nonpositive real parts and unequal patterned-layer ratios are refused.

Coordinates, real material values and the common ratio may vary under one
frozen geometry plan. Differentiation is supported along that constraint,
including one-sided loss derivatives at eta=0. Independently varying one
material's loss while keeping another fixed leaves the supported material
family. Invalid traced input produces NaN matrices rather than a valid
looking real-material result.

The underlying polynomial-body cell remains a positive-real-material
corrector. Its profile validator now explicitly checks that the imaginary
parts vanish before real mean-matrix assembly. Direct calls with unfactored
loss raise eagerly and produce NaN coefficients under tracing. General
complex correctors need consistent complex trial/test functions and energy
projection; discarding imaginary mean matrices cannot supply that extension.

## Independent controls and scope

The added tests cover:

- Common-factor scaling and positive-semidefinite dissipation in both layer
  axes, including the lossless boundary and loss ratios through 1.3.
- Independent analytic material integrals with unequal retained orders and
  zero-order axes.
- Mixed horizontal, uniform and vertical absorbing layers against analytic
  contact matrices through NumPy, JAX and Torch engines.
- Absorbing uniform slabs against independent Airy formulas for both
  incident polarizations on all three backends.
- Thirteen live geometry, storage-material, period, pose and loss parameters
  against two finite-difference steps, with a separate exact loss derivative.
- One-sided internal-gap, perpendicular-contact and lossless-boundary
  derivatives, material wrappers and invalid eager/traced inputs.

Commands and results are in
`studies/results/nvf_common_loss_validation.log`: 346 distinct cases passed,
including 30 new common-loss cases, with one historical expected failure
in the fixed-P1 moving-interface control. The manual examples and strict
documentation build also passed. These qualify the finite construction
and integration; they do not certify optical convergence.

A lossy inclusion in a lossless background, or two materials with different
loss ratios in one patterned layer, was outside this common-factor extension
and is supported by the later complex-material construction. Curved
interfaces, sheared cells, nonpositive real permittivities and automatic
projector selection also remain separate work. The common-factor case does
not complete general lossy-corrector development.
