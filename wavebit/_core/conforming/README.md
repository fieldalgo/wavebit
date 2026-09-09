# Conforming NVF cells

| Entry | Question it answers |
| --- | --- |
| `__init__.py` | How do rebased rectangle layers preserve physical geometry and the original Cartesian frame and Fourier order set? |
| `thin_gap.py` | How is the one-sided contact limit retained in a periodic conforming cell? |
| `partitioned_gap.py` | How do material intervals retain their opening derivatives at zero width without adding interface constraints? |
| `continuous_flux.py` | How can body displacement remain continuous across auxiliary mesh nodes? |
| `material_grading.py` | How can mesh resolution follow magnitude and phase material jumps and merge coincident interfaces while parameters stay differentiable? |
| `polynomial_body.py` | How do real and complex polynomial-body cells retain their contact traces and full energy projection? |
| `affine_cell.py` | How does an affine reference profile retain the full oblique metric, laminated contact lift, physical field frame and exact directional Fourier set? |

These kernels build constitutive data. Optical study fixtures and their
fixed-wavelength propagators remain in `studies/`. The public entry point is
`wavebit.ConformingNVF`; direct kernel inputs retain the historical reference
length and are implementation details.
