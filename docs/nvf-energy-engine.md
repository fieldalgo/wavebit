# Conforming NVF energy matrices in the Wavebit engine

The conforming rectangular cell can now feed the existing `solve_conv()`
pipeline. `ThinGapEnergyModel.conv_matrices()` and its polynomial-body and
material-graded subclasses return a one-layer `wavebit.ConvMatrices` stack.
The shared closure is implemented by `ConvMatrices.from_energy()` in the
library, without importing study modules into production code.

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

The existing real-dielectric research cell retains its absolute 1e-9
Hermitian-defect guard. Its coefficient construction, material grading and
contact equations are unchanged. The specialized high-order mirror path
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

Validation: 29 new builder/integration checks and 62 existing contact,
polynomial-body, localized-flux, pose and slot-order checks pass. The 28
documentation-contract checks also pass. These are 119 distinct checks,
not a run of the entire repository suite. The recorded command summaries
are in `studies/results/nvf_energy_engine_validation.log`.

## Reproduction and remaining scope

From this repository, the following sends the polynomial-body cell through
the normal engine and its result objects:

```python
import wavebit
from studies.nvf_polynomial_body import PolynomialBodyGapModel

order = 1
model = PolynomialBodyGapModel(order, rows=2, body_degree=2, gap_degree=2)
conv = model.conv_matrices(17., dict(seams=[31.], values=[4., 9.]))
solver = wavebit.rcwa(
    period=(700., 700.), wavelength=900., thickness=[300.],
    theta=.2, phi=.37, top=1., bot=2.25,
    truncation_order=(order, order),
)
result = solver.solve_conv(*conv)
print(result.p_inc.T.sum(), result.s_inc.T.sum())
```

The coarse cell here keeps the example inexpensive. It is not an accuracy
recommendation. `conv_matrices()` retains the model's square 700-nm cell,
centered horizontal geometry, positive real permittivities, fixed background
and square retained order. NumPy and JAX construct the research cell; a
NumPy export can be transferred to the Torch engine. The library closure
itself accepts all three array frameworks, including complex material data.

Arbitrary shapes, independently rotated objects, lossy corrector construction,
automatic shape dispatch and accuracy qualification across use cases remain
separate work. The existing circular shape projector remains available.

```bash
PYTHONPATH=$PWD python -m pytest -q tests/test_nvf_energy_builder.py tests/test_nvf_thin_gap_energy.py tests/test_nvf_polynomial_body.py tests/test_nvf_local_flux.py tests/test_nvf_posed_gap.py tests/test_slot_order.py
```
