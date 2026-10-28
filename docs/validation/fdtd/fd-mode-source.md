# A one-way waveguide mode source

Written: 2026-10-27

Targets: `src/wavebit/fdtd/_mode_source.py`, `src/wavebit/fdtd/_kernel/mode_sheet.py`, the
`ModeSource` declaration in `src/wavebit/fdtd/_sources.py`, the `source=` option of
`ModeMonitor`, `tests/fdtd/test_fd_mode_source.py`.

`fdtd.ModeSource` launches a mode of `fdtd.solve_modes` from a plane across a guide, one way.
The plane is a one-sided total-field/scattered-field (TFSF) sheet: the kernel's plane-wave
construction with the mode for its incident field. Its E pattern lives on the sheet's node
plane and its H pattern on the half plane upstream, each with its own waveform row. The H row
is the E row advanced by the mode's phase over half a cell, `beta dx / 2`, and by its group
delay over half a cell. At the pulse's centre frequency the pair is then exactly the discrete
mode, and the sheet launches it one way only. At other frequencies the profile and the
half-cell phase are the centre's, so the sheet also launches a little of the mode the other
way. The exact unconjugated orthogonality of the discrete modes gives both parts in closed form
from the actual waveform rows (the module docstring of `_mode_source.py` derives it). A
`ModeMonitor` declared with `source=` reports them as `injected` and `leakage`.

The TFSF construction is the standard one (Taflove and Hagness, *Computational
Electrodynamics*, 3rd ed., ch. 5). A mode as the incident field of a TFSF plane, with the H
plane's waveform advanced by half a cell, follows the same idea; nothing here is copied from
another solver.

## Registered acceptance criteria (2026-10-27, before computing)

**Setup S.** The slab of D1 in the through-PML record (`fd-waveguide-through-pml.md`): a
1.5 µm core of `epsilon = 4` in cladding of `epsilon = 2.085136`, y `InvariantAxis`, x `Open` at
±4.5 µm, z `Open` at ±4 µm, the core a `SnappedSlab` continued through both z faces, a 24-cell
`PML` at 1.55 µm. `ModeSource('src')` at z = -2 µm, direction `+z`, mode 0 (TE0), amplitude 1
(square-root watts per metre), pulse centred at 1.55 µm with `frequency_sigma_hz = 1.5e13`
and `cutoff = 1e-12`. `ModeMonitor`s with `source='src'`, `modes=1`, wavelengths 1.45, 1.55
and 1.65 µm: `p1` at z = +1 µm, `p2` at z = +2 µm, `q` at z = -3 µm. Spacings 0.05 and 0.025 µm.
Each run lasts until no `UnsettledFieldWarning` and no `GrowingFieldWarning` is raised.
`P(d)` below is the mode's propagation factor over a distance `d` with its own grid `beta`
at that wavelength (from `solve_modes`), in the run's Fourier convention.

* **S1, injection as predicted.** At every wavelength and both spacings,
  `|forward(p2) - injected * P(4 µm)| <= 1e-5 |injected|`. The same with the mirrored run (the
  source at z = +2 µm, direction `-z`, the monitor at z = -2 µm, its backward amplitude) at
  0.05 µm.
* **S2, leakage as predicted and stated.** At every wavelength and both spacings,
  `|backward(q) - (leakage * P(1 µm) + backward(p2) * P(5 µm))| <= 1e-6 |injected|`. The
  predicted leakage power `|leakage / injected|^2` at 1.55 µm is at most 1e-12 (the launch is
  one-way at the centre); at 1.45 and 1.65 µm it is reported.
* **S3, absolute power.** A continuous wave of amplitude `A` carries `A^2` watts per metre, so
  the launched mode's instantaneous power is `2 A^2 w(t)^2` for the unit waveform `w`, and its
  two-sided spectral energy is `2 A^2 |W(f)|^2`, `W` the continuous Fourier transform of `w`.
  At 1.55 µm and both spacings, `forward_energy(p2)` matches that to 2e-5 (relative).
* **S4, convergence with resolution.** At 1.55 µm the run's own effective index,
  `n_run = arg(forward(p2) / forward(p1)) / (k0 * 1 µm)` on the branch nearest the solver's
  index, matches the solver's grid index at the same spacing to 1e-9, and its error against the
  exact TE0 index of the slab (1.9562181938, the mode-solver record's M3) falls from 0.05 to
  0.025 µm with an observed order of at least 1.8.
* **S5, butt joint against mode matching.** Setup S, but the core is two `SnappedBox`es: 1.5 µm
  wide for z < 0 (continued through z-), 1.0 µm wide for z > 0 (continued through z+), with
  `p2` reading the TE0 of the narrow core and `q` moved to z = -1 µm. The TE0 power
  transmission `T = forward_energy(p2) / injected_energy` and reflection
  `R = backward_energy(q) / injected_energy` at 1.55 µm, against continuum mode matching at
  the same joint (written in the test: TE modes of both slabs in a perfectly conducting box,
  from a fine finite-difference solve of the one-dimensional continuum operator, coupled
  across z = 0 by continuity of Ey and Hx, converged in the number of modes and in the box
  width): `|T - T_mm| <= 5e-3` at 0.05 µm and `<= 2e-3` at 0.025 µm, smaller at the finer
  spacing; `|R - R_mm| <= 0.2 R_mm` at 0.025 µm, smaller than at 0.05 µm.
* **S6, backends.** Setup S at 0.05 µm on NumPy, JAX and Torch: `injected`, `leakage`,
  `forward(p2)` and `backward(q)` agree to 1e-12 (relative to `|injected|`).
* **S7, derivatives.** On JAX and Torch, at 0.05 µm: the derivatives of
  `forward_energy(p2)` at 1.55 µm with respect to the source's `amplitude`, of
  `Re forward(p2)` with respect to its `phase` and its `delay_seconds`, and of
  `forward_energy(p2)` with respect to the permittivity of a traced `Box` scatterer between
  the source and `p1` that reaches neither plane, each match a central difference (two steps,
  Richardson-extrapolated) to 1e-6 relative.
* **S8, refusals by code.** A direction not along the plane's normal (`source.direction`); a
  negative or non-integer mode (`source.mode`); a mode index beyond the confined modes found
  (`mode.count`); a traced structure reaching the source plane (`mode.traced`); a plane outside
  the physical domain (`source.outside_domain`); a monitor within a cell of the sheet
  (`monitor.source_sheet`); `source=` naming no `ModeSource` (`monitor.source`); a traced
  pulse wavelength (`mode.traced`); the engine placeholder `fd.ModeSource`
  (`NotImplementedError` naming the public source).

Amendment to S7, before any derivative was computed: as registered (Setup S, about 12k
steps) the reverse-mode runs keep every step and need about 8 MB a step on JAX here. Two
full attempts stalled without returning, and a 15-period probe of the same setup needed 13 GB
and was killed for memory; no derivative was computed. `Numerics(checkpoint_steps=)` is
admitted only on Torch CUDA. As for MM4b in the mode-solver record, the derivatives and
the central differences are taken of the same truncated run instead: a 5 µm by 4 µm
physical domain (x and z `Open`, 12-cell `PML`) holding the same slab, the source at
z = -1 µm, `p2` at z = +0.5 µm reading 1.55 µm, the pulse of Setup S with `cutoff = 1e-3`
and `frequency_sigma_hz = 2e13`, `Duration(steps=1200)`, and the scatterer, of permittivity
3.0, centred at (1.2, 0, -0.25) µm. The four derivatives, the steps and the 1e-6 limit are as
registered.

## Results

Measured with `tests/fdtd/test_fd_mode_source.py` on NumPy (JAX and Torch where named),
double precision. Wavelengths are listed as 1.45 / 1.55 / 1.65 µm.

| Criterion | Limit | 0.05 µm | 0.025 µm | Verdict |
| --- | --- | --- | --- | --- |
| S1 `forward(p2)` against the injected amplitude carried 4 µm | 1e-5 of `|injected|` | 2.7e-11 / 1.3e-14 / 2.3e-11 | 1.1e-11 / 7.2e-13 / 1.2e-11 | passed |
| S1 mirrored source (`-z`) | 1e-5 | 2.7e-11 / 1.3e-14 / 2.3e-11 | | passed |
| S2 `backward(q)` against the leakage and the transmitted reflection | 1e-6 of `|injected|` | 7.7e-14 / 2.8e-15 / 1.5e-12 | 4.8e-14 / 6.5e-15 / 1.2e-12 | passed |
| S2 predicted leakage power at 1.55 µm | 1e-12 | 6.8e-29 | 1.2e-29 | passed |
| S2 predicted leakage power at 1.45 / 1.65 µm | reported | 1.4e-6 / 1.5e-6 | 1.3e-6 / 1.4e-6 | |
| S3 `forward_energy(p2)` against `2 A^2 |W(f0)|^2` | 2e-5 | -1.2e-13 | -1.2e-13 | passed |
| S4 run index against the solver's grid index | 1e-9 | 1.969150425958 both | 1.959408135484 both | passed |
| S4 error against the exact TE0 index, observed order | at least 1.8 | 1.29e-2 | 3.19e-3 | passed (order 2.02) |
| S5 TE0 transmission, mode matching `T_mm = 0.955057` | 5e-3 at 0.05, 2e-3 at 0.025, smaller | 0.954677 (3.8e-4) | 0.954959 (9.8e-5) | passed |
| S5 TE0 reflection, mode matching `R_mm = 3.3287e-4` | 0.2 `R_mm` at 0.025, smaller than at 0.05 | 3.5366e-4 (6.2 %) | 3.3777e-4 (1.5 %) | passed |
| S6 JAX / Torch against NumPy | 1e-12 | 5.9e-16 / 1.3e-15 (worst) | | passed |
| S7 derivatives against extrapolated central differences | 1e-6 | pending | | |
| S8 refusals by code | as listed | all eleven raised | | passed |

The mode-matching oracle in S5 is converged in its own discretization before it is used:
halving the number of modes, and widening the box from 12 to 18 µm with more modes, each
moves `T_mm` by less than 2e-4 and `R_mm` by less than 2 %, the test's own check.

Added after registration, as further tests at registered limits (no limit changed). Two
deliberate breaks were not caught by the registered tests:
- a `-z` source whose H patterns were not signed for its direction. The exact prediction
  follows whatever the sheet holds, so S1 alone cannot see it;
- the sign of the corrections fed by Ex and Hy. TE0 has neither component.

The mirrored source is now also held to S2's one-way limit at the centre wavelength
(6.8e-29) and to S3's power limit (-1.2e-13). A TM0 launch (mode 1) is held to S1, S2 and S3
(3.5e-11, 6.2e-29, -1.3e-13). Both breaks, and the sign of the Ex-fed correction, then
failed. In all, sixteen deliberate breaks of the source were caught: the H row's half-cell
phase, the 1 W scaling, the H patterns' direction sign, each prediction formula and the
leakage's sign, each correction's sign, the H plane's side, the kernel's row offsets, the
monitor's `source=` check, the sheet clearance, the direction check, and the mode window
re-entering its own source. Each time the source was restored byte for byte. One parameter is
bounded by no limit: the H row's half-cell group delay changes only the off-centre leakage,
which S2 reports rather than limits, and the exact prediction follows it.
