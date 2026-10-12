# Independent source position and boundary clearance

Written: 2026-09-09

Targets:
- `src/wavebit/_core/scene_fdtd.py`
- `src/wavebit/main.py`
- `tests/test_fdtd_source_distance.py`
- `studies/fdtd_source_clearance.py`
- `studies/fdtd_rayleigh.py`

TL;DR: `source_distance` moves the current sheet without moving the mesh or
measurement planes. It also permits a fixed source while padding is enlarged.
Both controls preserve calibrated slab optics and backend material gradients.
In the tested near-cutoff grating, these changes move complex r/t much less than
the remaining discrepancy from RCWA. They do not resolve its time or grid error.

## Why separate these positions

Previously, increasing `pad_scale` also moved the source sheet farther from the
stack. That changed both boundary clearance and the illumination path. The
[ringdown comparison](fdtd-ringdown.md) explicitly retained that ambiguity.
The new option selects the nearest incident-side z node at a requested positive
interface distance, leaving the mesh, layers, PML and extraction planes intact.
It requires the source to stay outside the PML and upstream of the reflection
plane. Both incidence directions use the original scene coordinates in diagnostics.
The [guide](guide/time-domain.md#source-position-independent-of-padding)
defines snapping, clearance and validation rules.

A position change affects finite-time transients. It does not change the
interface phase reference of the requested scattering amplitudes: each run
still calibrates the source against its uniform incident-medium reference.
The option is static; it is not a geometry derivative or an independent new
Maxwell update. No default source position changes.

## Independent checks

Tests verify bit-identical results when the original source position is selected
explicitly, on both incidence sides. Moving the source leaves z nodes, cell
widths, PML faces and measurement positions identical. Conversely, changing
padding leaves the selected source and near measurement planes fixed on a
uniform mesh. A graded-grid test checks the nearest actual node in all four
basis/reference runs and refuses a user monitor that touches the current sheet.

A moved-source oblique TM slab refines to the independent Airy solution at
second order. An asymmetric magnetic, conical, bottom-incidence slab agrees
with RCWA, while changing the source position preserves complex amplitudes
within 1e-7. Combined source notches, grading, bottom coherent records and
material/mu/pole-strength sensitivities agree between NumPy, JAX and Torch;
the differentiated backends also match central finite differences.
Malformed and unplaceable positions are refused before time stepping.

## Near-cutoff controls

The physical grating, RCWA order-32 reference and measured orders -1/0 are the
[CPML-strength case](fdtd-pml-strength.md). All three new rows use resolution
40, 800 periods, Gaussian width 60 without notches, PML length 0.6 and nominal
target 1e-40. Layer alignment and directional extraction remain enabled.
The reflection/transmission planes stay at z=-0.2/0.4, and dt is
0.0115470053838. The study records full complex amplitudes so position
sensitivity is computed from the fields, not by subtracting two error norms.

| Control | Source z | Inner PML faces | Grid shape (z,y,x) | TE error | TM error |
| --- | ---: | --- | --- | ---: | ---: |
| Fixed source, padding 1 | -1.0 | -2.0, 2.2 | 216, 2, 32 | 0.024958 | 0.048172 |
| Same source, padding 2 | -1.0 | -3.9, 4.1 | 368, 2, 32 | 0.024807 | 0.048189 |
| Moved source, padding 1 | -1.5 | -2.0, 2.2 | 216, 2, 32 | 0.025005 | 0.048178 |

Error is the maximum absolute complex r/t discrepancy from RCWA over the two
requested wavelengths and the same two diffraction orders. Comparing full
amplitudes with the first row gives:

| Independent change | Maximum TE amplitude change | Maximum TM amplitude change |
| --- | ---: | ---: |
| Increase boundary clearance, source fixed | 1.51047e-4 | 3.73425e-5 |
| Move source by 0.5, boundaries fixed | 5.11239e-5 | 1.62142e-5 |

Transmission-side incoming order -1 magnitudes are respectively TE/TM
0.009442/0.003612, 0.009476/0.003591, and 0.009456/0.003617. These are
interface-referenced components of finite spectra, not a direct isolated
boundary-reflection measurement. Greater clearance does not remove them.

All three rows still fail state decay and scattering convergence. The latter
compares only 200 updates (2.3094 time units): maximum TE/TM changes are
2.31495e-4/7.33966e-5, 2.35485e-4/7.26717e-5, and
2.31912e-4/7.35155e-5. A small difference between two source/boundary settings
cannot certify an infinite-time answer or bound boundary error shared by both.

Within these tested choices, source placement and boundary clearance have a
much smaller effect than the remaining 0.025/0.048 discrepancy. They do not
explain away that discrepancy, and they do not by themselves prove it comes
entirely from the grid. The nonexponential cutoff tail and strong sensitivity
to discrete dispersion still require independent time and spatial checks.

## Reproduce

```shell
python -m studies.fdtd_source_clearance
```

The default is double-precision JAX on CPU; `--backend numpy` uses the same
engine. `--case fixed-near`, `fixed-far`, or `source-only` selects one row.
Each JSON row retains real/imaginary amplitudes, requested and actual source
coordinates, mesh, PML faces and convergence diagnostics. Running all cases
also prints direct complex-amplitude differences from `fixed-near`.
These controls do not add external solver or GPU validation beyond the cited
RCWA comparison and the existing shared-engine references.
