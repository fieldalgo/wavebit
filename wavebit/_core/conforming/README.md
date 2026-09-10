# Conforming NVF cells

| Entry | Question it answers |
| --- | --- |
| `__init__.py` | How do rebased rectangle layers preserve physical geometry and the original Cartesian frame and Fourier order set? |
| `thin_gap.py` | How is the one-sided contact limit retained in a periodic conforming cell? |
| `partitioned_gap.py` | How do material intervals retain their opening derivatives at zero width without adding interface constraints? |
| `continuous_flux.py` | How can body displacement remain continuous across auxiliary mesh nodes? |
| `material_grading.py` | How can mesh resolution follow magnitude and phase material jumps and merge coincident interfaces while parameters stay differentiable? |
| `polynomial_body.py` | How do real and complex polynomial-body cells retain their contact traces and full energy projection? |
| `affine_cell.py` | How does an affine reference profile retain the full oblique metric, exact laminated contact and stable opening derivative, physical field frame and exact directional Fourier set? |
| `affine_profile.py` | How do explicit fractional profiles on nonorthogonal lattices preserve physical materials, offsets, contact and the caller's Fourier modes? |
| `cartesian_rectangle.py` | How does a separated physical rectangle retain its Cartesian edges and finite derivatives on an oblique lattice with an explicit frozen orientation plan? |
| `cartesian_pair.py` | How do two horizontal equal-height rectangles retain physical edges, exact contact state, input identity and finite derivatives through an explicitly planned public adapter? |
| `cartesian_pair_opening.py` | How does the finite pair retain its analytical right-opening response through exact collapsed topology, Laurent weak operators and moving Fourier weights? |
| `cartesian_pair_gap.py` | How can the finite pair cross exact contact and positive gaps without dividing its weak solve or thin fields by gap? |

These kernels build constitutive data. Optical study fixtures and their
fixed-wavelength propagators remain in `studies/`. The public entry points are
`wavebit.ConformingNVF` for physical rectangle layers and
`wavebit.AffineProfileNVF` for explicit fractional profiles. The separate
experimental `wavebit.CartesianRectangleNVF` admits one separated physical
rectangle per patterned layer on its planned mesh. The explicit experimental
`wavebit.CartesianRectanglePairNVF` admits two horizontal equal-height
rectangles with frozen input indices and separated/contact charts. Neither
builder certifies optical convergence. Strip-profile kernel
inputs retain the historical reference length; the Cartesian rectangle
and pair kernels use physical lengths. Direct kernel inputs are implementation details.

The separate NumPy-only `CartesianPairOpeningPlan` and
`CartesianPairOpeningCell` expose an explicit finite right-opening response
from their submodule. They do not change ordinary contact replay or JAX
tangents. See the [manual](../../../docs/guide/cartesian-pair-opening.md).
