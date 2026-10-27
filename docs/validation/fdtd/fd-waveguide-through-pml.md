# A cell-snapped waveguide through an absorbing layer

Written: 2026-10-27

Targets: `src/wavebit/fdtd/_lower_public.py` (`_continue_snapped` and the slab crossing
rule), `tests/fdtd/test_fd_waveguide_pml.py`.

An open-ended waveguide needs its core to run into the absorbing layer at each end.
Until this change the public lowering refused any body narrower than the cross-section
there. The engine has always accepted a structure that runs straight through a layer,
because no interface then lies inside it. The public rule was the stricter one.

The door opened here is narrow. A `SnappedBox` may declare
`Structure(continuation=('z-', 'z+'))` (any one axis) when it reaches those physical
faces, and it is then extended through their padding with its cross-section unchanged.
A `SnappedSlab` may name the faces of the axes it runs along. Every cell of the core is
then the same in every layer of the PML: the medium is uniform along the absorbing axis,
with a core the layer is not graded for. A smoothed body narrower than the cross-section
(a `Box`, `Cylinder` or `Prism` core) stays refused. That separate question was studied
and not admitted, and nothing here changes it. A body that touches an `Open` face it
does not name is refused as before.

## Registered acceptance criteria (2026-10-27, before computing)

The guided mode's reflection is measured with `ModeMonitor`, between a source in the
core and the far layer, as the backward over forward power of mode 0 at the wavelength
read.

D1, slab waveguide. The M3 slab of the mode-solver record (1.5 µm core, `epsilon = 4`,
cladding `epsilon = 2.085136`, `InvariantAxis` y), x `Open` at ±4.5 µm, z `Open` at
±4 µm with the core continued through both z faces, spacing 0.05 µm, the default `PML` (12 cells),
an `Ey` `CurrentDipole` in the core at z = -2 µm with a pulse centred at 1.55 µm, and a
`ModeMonitor` at z = +2 µm reading 1.55 µm:

* the TE0 power reflection `|b0/a0|^2` at most 1e-5 with 12 cells, and smaller with 24
  cells than with 12;
* no `GrowingFieldWarning` in a run of 400 periods.

D2, rectangular core. A 1 µm by 0.5 µm `SnappedBox` core of `epsilon = 4` in vacuum,
x and y `Open` (a 3 µm by 2.5 µm physical cross-section, so the mode's tail is below
the monitor's 1e-3 confinement limit at its edge), z `Open` at ±2.5 µm with the core
continued through both z faces, spacing 0.05 µm, the default `PML`, an `Ex` dipole in
the core at z = -1 µm with a pulse centred at 1 µm, and a `ModeMonitor` at z = +1 µm
reading 1 µm: the fundamental mode's power reflection at most 1e-5 with 12 cells, and
smaller with 24 cells than with 12.

D3, refusals by code (`fdtd.padding_continuation`): a smoothed `Box` core narrower than
the cross-section declaring continuation; a continued `SnappedBox` that touches an
`Open` face of another axis; continuation naming a face without padding; a
continued `SnappedBox` that does not reach the named face; a `SnappedSlab` naming a
face of its own axis.

Amendment to D1, before it was first run: a `CurrentDipole` is refused on an
`InvariantAxis` (its stamp is not uniform along the invariant axis), so D1's source is
instead an initial field. Its E is the slab's TE0 mode at 1.55 µm, as `solve_modes`
returns it, times `exp(-(z + 2)^2 / (2 (0.6 µm)^2)) cos(beta0 (z + 2))` with `beta0` that
mode's propagation constant, and its H is zero. It launches TE0 both ways from z = -2 µm
with a spectrum centred near 1.55 µm. Everything else in D1 is as registered.

## Results

Measured with `tests/fdtd/test_fd_waveguide_pml.py` on NumPy, double precision.

| Criterion | Limit | 12 cells | 24 cells | Verdict |
| --- | --- | --- | --- | --- |
| D1 TE0 power reflection, slab | at most 1e-5 with 12 cells, smaller with 24 | 4.3e-8 | 3.5e-16 | passed |
| D1 no `GrowingFieldWarning` in 400 periods | none | none raised | none raised | passed |
| D2 fundamental-mode power reflection, rectangular core | at most 1e-5 with 12 cells, smaller with 24 | 1.3e-6 | 3.5e-12 | passed |
| D3 refusals by code | `fdtd.padding_continuation` | all five raised | | passed |

Two details of the runs, neither a change to a limit. D1 has no source to infer the PML
wavelength from, so its `PML` names 1.55 µm, and its 400 periods are written as the
equivalent duration in seconds. D2's pulse width was not registered; the first width tried
(`frequency_sigma_hz = 3e13`) was refused by the resolution check before any step (its
shortest wavelength is sampled at fewer than 8 cells in the core), and the runs use
`1.5e13`. D2's core is sampled at about 10 cells per wavelength at 1 µm, so its runs carry the
`CoarseMeshWarning`; the reflection it measures is of the mode on that grid.

Added after registration, as further refusal and admission tests (no limit changed): a
continued core reaching only one of the two named faces (each way), continuation naming
faces of two axes, the registered D1 and D2 scenes passing `check` and `prepare`, and the
lowered core spanning the allocated domain with its cross-section unchanged. Each new rule
was broken on purpose and the tests failed: dropping the one-axis rule, either reach check,
the padding check, the extension through either face, the slab's own-axis rule or the
crossing-rule exemption, and letting a smoothed `Box` continue; the source was then restored
byte for byte.
