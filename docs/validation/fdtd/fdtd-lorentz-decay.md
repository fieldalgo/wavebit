# Residual Lorentz motion and dispersive settling

Written: 2026-09-08

Targets:
- `src/wavebit/fdtd/_periodic/grid.py`
- `src/wavebit/fdtd/_periodic/scene.py`
- `studies/fdtd/lorentz_decay/fdtd_lorentz_decay.py`
- `tests/fdtd/test_fdtd_material_decay.py`

TL;DR: weakly coupled material oscillations can survive a small electric field.
Inspect per-pole activity as well as fields; neither condition certifies spatial
or spectral accuracy on its own.

## A field-only blind spot

`PYTHONPATH=. python studies/fdtd/lorentz_decay/fdtd_lorentz_decay.py` first runs a uniform periodic
medium with epsilon-infinity 1, conductivity 10, and one Lorentz pole
(omega0, gamma, strength) = (1, 0.01, 0.01). The initial Ex is 1 and all other
fields and polarizations start at zero. Grid shape is (4,2,2), spacing 1,
dt=0.02, duration 40, with the last 100 updates as the diagnostic window.

| Indicator | Value |
| --- | --- |
| Field amplitude decay relative to its peak | 0.0000876375 |
| Material activity amplitude decay relative to its peak | 0.814188 |
| `field_decayed` at tolerance 1e-4 | True |
| `material_decayed` | False |
| `state_decayed` | False |

Conductivity damps the electric field while the weakly coupled oscillator keeps
moving. This motivated a separate material criterion. The existing field flag
still means field decay; it has not been silently redefined. The new combined
flag requires both fields and excited poles to decay. It is not used to stop
or resize the fixed-step calculation.

The material indicator uses positive activity proportional to
(omega0² abs(P)² + abs((P-P_previous)/dt)²)/strength, integrated with each native
E-component control volume. It ignores zero-strength cells. This includes both
polarization and its discrete velocity; a static Drude polarization with omega0=0
is not counted as ongoing motion. Different time levels prevent interpreting
this quantity as an exactly conserved discrete energy. It excludes CPML memory.

## Strongly dispersive slab against Airy

The same script next runs a thickness-0.2 slab in vacuum with epsilon-infinity
1.5, omega0=2 pi, gamma=0.05 and strength=16. Normal TE incidence and wavelengths
0.9, 1.0 and 1.1 span the material resonance. The analytic reference is the passive
Airy slab formula using epsilon(omega)=1.5+16/(omega0²-omega²-i gamma omega).
At wavelength 1 its imaginary permittivity is about 50.93, so this is a strong
material response, not a small perturbation of a dielectric.

Measurements used NumPy 2.5.1, complex128 arrays on arm64 CPU, period 0.05,
uniform grids and physical PML width 0.3 at both resolutions. Diagnostic windows
are approximately two optical periods. Errors are maximum absolute complex r/t
errors across the three wavelengths; decay is the maximum across both source
bases and all applicable actual/reference runs.

| Resolution | Periods | Max r error | Max t error | Field decay | Material decay | Max amplitude change | Combined state decayed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 40 | 40 | 0.0499350 | 0.0444946 | 0.0163800 | 0.0884314 | 0.0101721 | False |
| 40 | 160 | 0.0421528 | 0.00729038 | 0.000312329 | 0.00276566 | 0.000171126 | False |
| 80 | 160 | 0.0114101 | 0.00186209 | 0.000292324 | 0.00263100 | 0.000155945 | False |
| 40 | 320 | 0.0413564 | 0.00726711 | 0.00000301883 | 0.0000403128 | 0.00000206017 | True |
| 80 | 320 | 0.0105624 | 0.00181447 | 0.00000276694 | 0.0000372271 | 0.00000195752 | True |

Material motion outlives the apparent field transient. Increasing runtime reduces
the temporal indicators; the coarse-grid r error persists after they pass.
At 320 periods both resolutions also pass the amplitude-change criterion.
Doubling resolution reduces the maximum complex r/t errors by approximately four
times. Resolution and material time discretization still need refinement; this
two-grid check does not certify an asymptotic order or general material accuracy.
These cases do not validate arbitrary low-loss metals, overlapping poles,
anisotropic dispersion or dispersive media inside PML.

## Verification and source boundary

The tests derive a closed-form discrete uniform Lorentz oscillator from D=E+P=1,
then compare peak/tail activity and its strength derivative with the implementation
on NumPy, JAX and Torch. A longitudinal nonuniform-grid case verifies component
volumes and zero-strength masks. Further cases distinguish residual Lorentz motion
from static Drude polarization and check aggregation through the scene API.
Existing frozen material references continue to test the field solver separately.

The oscillator equation and frequency convention can be compared with Meep's
public [material-dispersion documentation](https://meep.readthedocs.io/en/latest/Materials/#material-dispersion).
Wavebit uses angular frequencies and a direct driving strength s; parameter names
and conductivity conventions must not be copied numerically between solvers.
Only public documentation was read. No Meep implementation or new external solver
execution was used for this change.
