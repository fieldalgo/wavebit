# Conforming NVF cells

| Entry | Question it answers |
| --- | --- |
| `__init__.py` | How do ordinary rectangle layers become a conforming energy closure in the solver's Cartesian frame? |
| `thin_gap.py` | How is the one-sided contact limit retained in a periodic conforming cell? |
| `partitioned_gap.py` | How are material partitions integrated without inserting artificial interface constraints? |
| `continuous_flux.py` | How can body displacement remain continuous across auxiliary mesh nodes? |
| `material_grading.py` | How can mesh resolution follow real material jumps while parameters stay differentiable? |
| `polynomial_body.py` | How do higher-degree body potentials improve the cell without changing its contact traces? |

These kernels build constitutive data. Optical study fixtures and their
fixed-wavelength propagators remain in `studies/`. The public entry point is
`wavebit.ConformingNVF`; direct kernel inputs retain the historical reference
length and are implementation details.
