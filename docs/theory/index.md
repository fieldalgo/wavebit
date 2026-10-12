# Theory & validation

Derivations, numerical decisions and validation records state what was measured,
which inputs were covered and which limits remain.

| Topic | Starting point |
| --- | --- |
| RCWA derivation | [Maxwell equations to scattering amplitudes](../formulation.md). |
| Numerical decisions | [Method and implementation records](../decisions/index.md). |
| Solver conventions | [Coordinates, fields, materials and normalization](../conventions.md). |
| Periodic FDTD | [Scope and validation](../fdtd-alpha.md). |
| General FDTD | [Geometry validation](../fd-mie-validation.md) and [periodic metasurfaces](../fd-metasurface-validation.md). |
| Independent references | [Reference generators and frozen data](https://github.com/fieldalgo/wavebit/tree/develop/tests/references). |

A passing implementation test, a converged discretization and agreement with an
independent physical reference answer different questions. Each record specifies
which claims its evidence supports.
