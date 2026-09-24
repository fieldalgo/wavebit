# Conforming NVF energy matrices in the Wavebit engine

Written: 2026-09-09

Targets:
- `wavebit/_core/conforming/__init__.py`
- `wavebit/_core/conforming/polynomial_body.py`
- `wavebit/_core/factorization.py`
- `tests/test_conforming_nvf.py`
- `tests/test_nvf_energy_builder.py`

`wavebit.ConformingNVF` builds conforming rectangular cells from ordinary
shape-layer dictionaries and feeds the existing `solve_conv()` pipeline.
The cell kernels now live in the library; the study classes reuse them and
keep only their fixed-fixture optical readouts in `studies/`. The shared
closure is `ConvMatrices.from_energy()`.

This removes the need to reconstruct the five-slot layout, invert the zz
matrix or guess the off-diagonal signs in an adapter. The consuming solver
owns wavelength, incidence, thickness, exterior media and layer scattering.
The previously hard-coded research propagator is no longer the only way to
use the cell's constitutive matrices.

## Closure and derivative contract

Given Q = C(B), G = C(B.H epsilon B) and Z = C(epsilon_zz), the builder forms

```text
E = Q^(-H) G Q^(-1)
ConvMatrices(E_xx, E_yy, Z^(-1), -E_xy, -E_yx).
```

Both sides of the congruence use linear solves. The builder accepts single
layers and stacks, preserves input arrays and their autodiff graphs, and
does not symmetrize or regularize G. Complex loss therefore survives the
closure. Every matrix must describe the same retained Fourier order set
and Cartesian frame; the builder checks dimensions, not that physical
correspondence. Finite input and output checks are separate from an accuracy
estimate. An ill-conditioned but finite solve is not certified as accurate.

The real-dielectric cell retains its absolute 1e-9
Hermitian-defect guard. The closure integration itself did not change cell
construction or contact equations; subsequent grading changes are described
below. The specialized high-order mirror path
retains its existing memory-conscious LU implementation.

The new tests recover a prescribed non-Hermitian constitutive tensor under
a complex non-orthogonal change of basis on NumPy, JAX and PyTorch, in both
complex precisions and with/without a layer axis. JAX JIT and Torch autograd
also check a basis change whose derivative must cancel while its material
derivative remains nonzero. These qualify the closure derivatives. The
existing production propagator retains its own derivative behavior; this
integration does not replace it with the eigenvector-free study propagator.

## End-to-end controls

The actual Wavebit engine is checked against independent lossless and
absorbing uniform-slab formulas at different wavelengths and thicknesses.
For the conforming mixed-material gap, both incident polarizations agree
with the separate scattering-doubling instrument on all three backends
under an absolute 2e-10 criterion. Splitting one conforming layer into two
identical adjacent layers preserves all returned order efficiencies under
2e-11 at a different wavelength and substrate. These are finite-model
integration controls, not a claim of optical convergence.

The original closure integration record has 29 builder/integration checks and 62 contact,
polynomial-body, localized-flux, pose and slot-order checks pass. The 28
documentation-contract checks also pass. These are 119 distinct checks,
not a run of the entire repository suite. The recorded command summaries
are in `studies/results/nvf_energy_engine_validation.log`.

## Shape construction and Fourier origin

The public constructor accepts a nonoverlapping rectangle sequence along either
lattice axis in each layer, uniform layers, positive scalar backgrounds,
independent orthogonal periods,
translations, common lattice/shape rotations and rectangular retained-order
sets. It normalizes physical lengths and material ratios before the shared
cell solve. It then applies translation and Cartesian tensor rotation to
the field and its energy, selects the requested Fourier difference indices,
and closes the projected matrices. Cropping an already-closed square matrix
would be a different finite model and is not used.

An independent analytic material integral exposed a half-cell phase mismatch
during integration. The cell kernels use a centered Fourier origin; the
ordinary shape API measures center offsets from `(a1+a2)/2` while its
Fourier coefficients use the cell corner. The adapter applies this shift
to **all** coefficient fields. The former mismatch changed odd-order signs
but preserved single-layer total powers, so power conservation alone could
not have found it. The material-convolution check now covers physical
scaling, background contrast, large translations, rotation and unequal
retained orders. An independent rotated Cartesian inverse-rule control
checks every slot at full-height contact.

Further public-API controls cover both incident polarizations and every
returned diffraction power under rigid rotation and translation; physical
scaling of the full stack; NumPy, JAX and Torch forward solves; equal-material
subdivision; live JIT geometry, material, background, period and pose
derivatives; one-sided contact derivatives; and invalid eager/traced replay.
The package-only subprocess smoke test constructs a patterned cell without
any `studies` package. Commands and results are recorded in
`studies/results/nvf_conforming_shapes_validation.log`.

The initial shape integration validation recorded 48 passing public-API checks, 204
passing shared-kernel and engine regressions, and 29 passing documentation
contract checks. One pre-existing strict expected failure records the old
fixed-P1 model's moving-interface limitation; the polynomial-body model
used by the public builder passes that control. The two manual examples,
strict Sphinx build, lint and diff whitespace checks also pass.

The contact derivative test scans finite-difference steps. Steps of 1e-4
in the 700-unit fixture do not resolve the short scale near contact; two
windows at 1e-6 and 1e-7 agree with the JAX derivative under the same 3e-6
absolute criterion. This is a derivative control of a finite cell, not an
optical convergence threshold.

## Reproduction and remaining scope

The [public manual](guide/conforming-nvf.md) contains standalone solve and
JAX examples, resolution defaults, material semantics and explicit refusal
rules. NumPy and JAX x64 construct the cell; a NumPy export can be
transferred to the Torch engine. The lower-level energy closure itself
accepts all three array frameworks and complex material data, while the
rectangular constructor supports positive-real-part passive materials with
independent loss ratios for every material and background.

The [rectangular-cell extension](nvf-rectangular-cells.md) supplies the live
physical aspect metric, with separate energy-stationarity, quadrature and
period-derivative controls. It also removes a zero-gradient contact clamp
that could be triggered by roundoff when normalizing the height.

The [layer-axis extension](nvf-layer-axes.md) permits first- and second-axis
tilings in the same stack. It maps every layer back to the solver's original
Cartesian frame and retained-order layout before closure. Plans freeze both
ordering and axis; equivalent quarter-turn rectangle spellings preserve
the geometry and all five matrices.

The [separated-rectangle extension](nvf-separated-rectangles.md) inserts live
background intervals between shapes. It retains zero-width intervals and
their opening derivatives, and uses signed log-jump grading so coincident
boundaries merge or cancel. This changes the finite approximation used by
graded cells; historical measurements retain their original density.

The [common-loss extension](nvf-common-loss.md) factors a shared passive
complex scalar out of each patterned layer's cell equations. It preserves
the real storage-cell construction and applies the factor to the energy
closure, including its inverse longitudinal slot. Uniform layers can have
independent passive permittivities. The underlying real cell explicitly
refuses unfactored imaginary material values.

The [complex-material extension](nvf-complex-material.md) removes the
common-ratio restriction by using complex correctors, conjugate test fields
and independent cross-energy terms. The original real kernels remain
available for reproducing their studies.

Arbitrary shapes, object edges not aligned with the lattice, sheared
corrector cells,
automatic selection through the ordinary
solver's `nvf_projector` setting, and accuracy qualification across use cases
remain separate work. The existing circular shape projector remains
available. This integration does not make the provisional 1e-5 research
increment a product accuracy requirement.

```bash
PYTHONPATH=$PWD python -m pytest -q tests/test_nvf_energy_builder.py tests/test_nvf_thin_gap_energy.py tests/test_nvf_polynomial_body.py tests/test_nvf_local_flux.py tests/test_nvf_posed_gap.py tests/test_slot_order.py
PYTHONPATH=$PWD python -m pytest -q tests/test_conforming_nvf.py
```

The [lattice-rebasing extension](nvf-lattice-rebasing.md) accepts skew basis
vectors that describe the same orthogonal point lattice. It preserves
physical rectangles, the supplied cell-center convention and the exact
original Fourier set. Genuinely nonorthogonal point lattices still need a
different cell construction.
