# Waveguide modes of the discrete Yee system

Written: 2026-10-27

Targets: `src/wavebit/fdtd/_modes.py`, `tests/fdtd/test_fd_waveguide_modes.py`.

This record covers the stage-7 mode solver, `fdtd.solve_modes`. The acceptance
criteria below were registered before the solver was run on any case. They are
not changed after the results are known; a criterion that fails stays failed in
this record.

## What is solved

`fdtd.solve_modes` prepares the problem exactly as `fdtd.prepare` does and reads
the update coefficients the time stepper will multiply by. On a plane across one
axis `w` it takes the diagonal permittivity and permeability of each Yee
component (`eps_c = (dt/dx)/be_c`, `mu_c = (dt/dx)/bh_c`) and solves the
eigenproblem of the *discrete* Maxwell equations for fields `exp(i(beta w - omega t))`.

On the Yee lattice with the kernel's update, a time-harmonic field satisfies the
continuum curl equations with two exact replacements:

* `omega -> Omega = (2/dt) sin(omega dt/2)`, from the leapfrog in time, and
* `d/dw -> i q` with `q = (2/dx) sin(beta dx/2)`, from the staggered difference
  along the propagation axis, when each component is referred to its own Yee
  position.

The transverse derivatives stay the kernel's forward and backward differences
with the faces' own neighbour rules. Eliminating the longitudinal components
gives `q^2 e = Q P e` for the transverse electric field `e`, where `P` and `Q`
are real symmetric for lossless diagonal media and zero, periodic or wall
neighbours. The solver returns `beta = (2/dx) asin(q dx/2)` and the effective
index `n_eff = beta c/omega` at the physical angular frequency. Because the
operator is the kernel's own, an injected mode is a mode of the discrete system,
not of the continuum: its residual in the time stepper is round-off, while its
distance from the continuum mode is the grid's discretization error.

Symmetry of `P` and `Q` gives the exact discrete orthogonality used for power
normalization and mode decomposition: for two modes with `q_m^2 != q_n^2`,
`sum (E_u^m H_v^n - E_v^m H_u^n) = 0` over the plane, without complex
conjugation. Modes are normalized to unit time-averaged power through the plane
(1 W, or 1 W/m with one invariant axis).

## Registered acceptance criteria (2026-10-27, before computing)

M1, homogeneous control. Transverse faces `Periodic`, medium `epsilon = 2.25`,
spacing 0.05 µm, wavelength 1 µm, with a 2-D cross-section and with one
`InvariantAxis`. The fundamental mode must have `q/Omega = 1.5` and `n_eff`
equal to the closed-form Yee dispersion
`n_eff = (2/(dx k0)) asin(1.5 Omega dx/2)`, both to a relative 1e-12.

M2, hollow metal guide. `PEC` on both transverse axes, a = 2 µm by b = 1 µm,
filled with `epsilon = 2.25`, wavelength 1 µm, spacings 0.05, 0.025 and
0.0125 µm. The first five modes must match the discrete closed form
`q^2 = Omega^2 eps - (2/dx)^2 [sin^2(m pi dx/(2a)) + sin^2(n pi dx/(2b))]`
(TE10, TE20, TE01 and the degenerate TE11/TM11 pair) to a relative 1e-10 in
`q^2`. Against the continuum `n_eff`, the observed order between successive
spacings must be at least 1.8 for each of these modes.

M3, symmetric slab against the exact dispersion relation. Core thickness 1.5 µm,
`epsilon = 4` (n = 2), cladding `epsilon = 2.085136` (n = 1.444), wavelength
1.55 µm, one `InvariantAxis` and `Open` faces across the slab at 4.5 µm from its
centre (the solver window is the physical domain). Spacings 0.05, 0.025 and
0.0125 µm, all aligned with the core faces. TE0, TE1, TE2 and TM0, TM1, TM2
against the roots of the TE and TM transcendental equations:

* observed order at least 1.8 for both refinements and all six modes;
* `|n_eff - exact| <= 1e-3` at 0.0125 µm for all six modes;
* the order-2 Richardson extrapolation of the two finest spacings within 1e-4
  of the exact value for all six modes;
* at every spacing, the cross-product of two distinct modes at most 1e-10 of
  the product of their norms, and the normalized power 1 to a relative 1e-12.

M4, published dielectric-corner benchmark. The four "analytically solvable"
corner problems of G. R. Hadley, J. Lightwave Technol. 20, 1210 (2002)
(stated error below 1e-8): a unit square at wavelength 1.5, a square core
`[0, 0.5]^2` in one corner, `PEC` on the listed sides and `PMC` on the others,
reference effective indices 1.27627404 (core 2.25, cladding 1, PEC left and
right), 2.65679692 (8, 1, PEC left and right), 1.387926425 (1, 2.25, PEC left
and top) and 2.761465320 (1, 8, PEC left and top). The walls are placed where
the kernel puts them (a PMC wall half a cell off the face) so that each wall
lies on the square's edge; the core is a smoothed `Box`. Three spacings near
1/40, 1/80 and 1/160. The largest-index mode must reach the published value
with an error that decreases monotonically over the three spacings and is at
most 1e-3 at the finest. The observed order is reported, not gated: a
dielectric corner is a field singularity, which is the subject of that paper.

M5, refusals by name: a conductive or dispersive medium in the window; a medium
that is not uniform along the axis at the plane; a `Bloch` or `MirrorSymmetry`
transverse face; an `InvariantAxis` with nonzero `k`; coupled smoothed
(constitutive) or extended-stencil material in the window; a traced or
gradient-carrying input (the eigensolve is a host computation and does not
differentiate); a plane outside the physical domain; a request for more confined
modes than exist. Across an `Open` face a mode is admitted only when its largest
transverse electric field on the window's outermost cells is at most 1e-3 of
its peak.

M6, backends. The arrays returned for `backend='numpy'`, `'jax'` and `'torch'`
are equal to the bit: the eigensolve runs once on the host and the result is
converted.

## Registered addition (2026-10-27, before computing)

Mutation testing of the first tests showed that M1-M4 cannot see two defects that
matter: a mode built with a permittivity sampled at the wrong Yee point, and a
propagation constant read from `q` without the `asin`. Both keep a symmetric
slab second order. The direct check is that a solved mode is an eigenmode of the
time stepper itself, so this criterion is added before it is first run:

M7, the mode is a mode of the run. A 1 µm by 0.5 µm `SnappedBox` core of
`epsilon = 4` in a 3 µm by 3 µm `PEC` box (spacing 0.05 µm, wavelength 1 µm), and
the M3 slab with `PEC` instead of `Open` faces at 0.05 µm, each with a `Bloch`
propagation axis whose `k` is the mode's `beta` over eight cells. The run starts
from the mode (E at t = 0, H at -dt/2, each component at its own Yee point) and
takes 100 steps. Every component of the final E and H must equal the mode advanced
by `exp(-i omega t)` to within 1e-10 of the largest value of that field, for the
two largest-index modes of each guide.

M1 and M2 compute `Omega` from the time step `fdtd.check` reports, not from the
solver's own output.

A further M7 case, registered after the mutation run that showed why and before it
was first run: for a cell-snapped core each component's permittivity array is the
same index for index, so reading one component's coefficient in place of another's
changes nothing there. The same M7 criterion therefore also holds for a smoothed
`Box` core of 1.03 µm by 0.51 µm, whose faces lie between cells and whose
per-component averages differ.

## Results (2026-10-27)

Measured with `tests/fdtd/test_fd_waveguide_modes.py` on Linux CPU, Python 3.12,
NumPy 2.5.4, SciPy 1.18.1. Every registered criterion passed.

| Criterion | Limit | Measured |
| --- | --- | --- |
| M1 `q/Omega`, `n_eff` vs closed form (2-D; invariant) | 1e-12 | 4.6e-15, 4.8e-15; 4.4e-16, 5.9e-16 |
| M2 `q^2` vs discrete closed form, five modes, three spacings | 1e-10 | 7e-14, 7e-14, 1.1e-12 (worst per spacing) |
| M2 observed order vs continuum | >= 1.8 | 2.024 to 2.027, then 2.006 to 2.007 |
| M3 observed order, six modes | >= 1.8 | 2.014 to 2.019, then 2.003 to 2.005 |
| M3 `|n_eff - exact|` at 0.0125 µm | <= 1e-3 | 5.2e-4 (TM2) to 7.9e-4 (TE0) |
| M3 Richardson extrapolation | <= 1e-4 | 1.6e-6 to 3.5e-6 |
| M3 orthogonality / power | 1e-10 / 1e-12 | passed at every spacing |
| M4 Hadley corners, error at the three spacings | monotone, <= 1e-3 | see below |
| M5 refusals | by code | each refusal raised with its code |
| M6 backends | bitwise | equal |
| M7 mode advanced by the run, 100 steps | 1e-10 | at most 6.4e-14 (slab), 2.1e-14 (snapped box), 1.3e-14 (smoothed box) |

The exact slab indices (TE0, TE1, TE2, TM0, TM1, TM2) are 1.9562181938,
1.8225308386, 1.5975660835, 1.9470205470, 1.7879894882 and 1.5490399425. At 0.05,
0.025 and 0.0125 µm the solver is above each by 1.29e-2 to 8.4e-3, 3.2e-3 to 2.1e-3,
and 7.9e-4 to 5.2e-4. The third TM mode's largest transverse E on the window edge is
4e-4 of its peak, inside the 1e-3 admission limit; it is the least confined mode
here.

Hadley's corners, signed error at 1/40, 1/80 and 1/160 (near those values for the
PEC-left/top cases, whose PMC walls set the spacing as described in the test):

| Core, cladding, PEC sides | Published | Error | Observed order |
| --- | --- | --- | --- |
| 2.25, 1, left/right | 1.27627404 | +1.10e-3, +2.74e-4, +6.84e-5 | 2.00, 2.00 |
| 8, 1, left/right | 2.65679692 | +8.86e-3, +2.20e-3, +5.49e-4 | 2.01, 2.00 |
| 1, 2.25, left/top | 1.387926425 | +1.17e-3, +2.89e-4, +7.17e-5 | 2.02, 2.01 |
| 1, 8, left/top | 2.761465320 | +9.69e-3, +2.37e-3, +5.85e-4 | 2.03, 2.02 |

Second order holds at these corners on this grid with a smoothed core: the field
singularity of a dielectric corner did not show at these spacings. This is an
observation at three spacings, not a statement about finer grids.

What this does not establish: accuracy for lossy, dispersive, anisotropic or leaky
guides, all refused; and any run's accuracy, which still needs its own spacing
convergence. The effective index is the grid's, by construction.

Mutation checks (each product change below was made, the named test failed, and the
source was restored byte for byte): leaving `Omega` as `omega` (M1, M2); dropping the
low-wall tangential-E mask (M2); dropping the cosine factor of the power (M3); dropping
the PMC high-face masks (M4); reading `eps_u` for `eps_w`, swapping `eps_u` and `eps_v`,
taking `beta = q`, flipping the sign of `E_w`, returning `Z0 H` for `H` (M7); disabling
each refusal (M5); and returning NumPy arrays for `backend='jax'` (M6).

## ModeMonitor: registered acceptance criteria (2026-10-27, before computing)

`fdtd.ModeMonitor(name, plane=..., wavelengths=..., modes=n)` projects the run's
DFT fields on its plane (the node-plane E and the mean of the two H half planes, as
a `FluxMonitor` pairs them) onto the first `n` modes the solver returns at each
wavelength. With `X = sum (E_u h_v - E_v h_u) dA` and `Y = sum (e_u H_v - e_v H_u) dA`
for a mode `(e, h)` normalized to unit kernel flux (`N = sum e . h dA = 2/cos(beta dx/2)`),
orthogonality gives the forward and backward amplitudes
`a = (X/N + Y/(N cos))/2` and `b = (X/N - Y/(N cos))/2` exactly for a field made of
modes. `|a|^2` and `|b|^2`, scaled as a `FluxMonitor` scales its flux, are the
two-sided pulse spectral energies (J/Hz) carried forward and backward by the mode.

MM1, exact direction separation through the public path. In the three M7 guides
(snapped box, slab, smoothed box) with a `Bloch` axis of eight cells whose `k` is
`+beta` (or `-beta`), a run of 200 steps starting from mode 0 or mode 1 travelling
`+z` (or `-z`), and a `ModeMonitor` with `modes=2` at the mode's own wavelength:
the opposite direction and the other mode each at most 1e-10 of the launched
amplitude, and the launched mode's spectral energy equal to a `FluxMonitor`'s on
the same plane to a relative 1e-10.

MM2, completeness against the flux. A hollow `PEC` guide 1 µm by 0.5 µm filled with
`epsilon = 2.25`, spacing 0.05 µm, `Open` along z with the default PML, a
`CurrentDipole` (`Ey`, off-centre so that it excites the guide's modes of both
parities) and monitors 3 µm on each side at wavelength 1.05 µm, where seven modes
propagate. For every monitor, the sum over the seven modes of forward minus
backward spectral energy equals the `FluxMonitor` spectral energy on the same plane
to a relative 1e-6. The run is long enough for its decay diagnostic to report the
fields settled.

MM3, backends. NumPy, JAX and Torch give the same MM2 amplitudes to a relative
1e-12 of the largest.

MM4, derivatives. In the MM2 guide with a `SnappedBox` block of `epsilon = 3`
between the dipole and the +z monitor (not reaching either monitor plane), the
derivative of mode 0's forward spectral energy with respect to that permittivity,
by `jax.grad` and by Torch autograd, agrees with the central difference at step
`1e-4 * epsilon` to a relative 1e-6.

MM5, refusals by name: a monitor plane where the medium changes along the axis
(`mode.nonuniform`); a traced or gradient-carrying structure that reaches the
monitor plane (`mode.traced`); more modes than are confined (`mode.count`); a
non-positive `modes` (`monitor.modes`); a plane outside the physical domain.

### MM2 outcome and a registered replacement (2026-10-27)

MM2 as registered **failed**. Its precondition was not met: after 150 source periods
the decay diagnostic reported 2.0e-3 of the peak energy still in the monitor regions
(the line is 1e-5), and the modal sum differed from the flux by 4.5e-5 (limit 1e-6).
Longer runs (300 and 600 periods) still did not settle. The cause is physical and
lies in the configuration: a closed metal guide traps the spectral content at each
mode's cutoff, where the group velocity is zero, so a broadband pulse whose spectrum
crosses cutoffs never leaves the guide.

The unsettled field explains the mismatch. The run's own `FluxMonitor` values differed
along the lossless guide by 3% (λ = 1.05 µm) to 14% (0.9 µm) between z = 1.5 and 3.5 µm,
while each well-propagating mode's net modal energy stayed constant along z and only
the near-cutoff TE21 mode's changed. The Fourier fields of a run that has not decayed
are not a solution of both discrete curl equations at once. The end correction
(`src/wavebit/fdtd/_post.py::end_correction`) makes each field's own update exact against the
other field's *uncorrected* transform, so the corrected pair leaves a residue at the
final field's level. A box of corrected phasors gave relative curl residuals of 0.7%
(Faraday) and 5-7% (Ampère); the same curls applied to the time-domain fields close to
4.7e-15. This is the engine's documented behavior for a run that has not decayed (its
`UnsettledFieldWarning`); it is recorded here, not changed.

MM2b, registered after the MM2 failure and before it was run: the MM2 guide and
monitors, with a narrowband pulse kept inside the band where exactly seven modes
propagate on this grid. The discrete cutoffs of the next mode above and the last mode
below are at 1.0087 µm and 1.0644 µm. The pulse is centred at 1.036 µm, with a
frequency standard deviation of 1.3e12 Hz (both band edges more than six standard
deviations away) and `cutoff=1e-9`, and the monitors read 1.036 µm. The run must
report the fields settled, and the seven-mode sum must equal the flux to a relative
1e-6 at both monitors.

### MM4 outcome and a registered replacement (2026-10-27)

MM4 as registered **failed**: run for 800 steps (enough for the pulse's leading part to
cross the +z monitor; a reverse-mode derivative keeps every step), `jax.grad` gave
8.794522e-20 J/Hz per unit permittivity, and the central difference at the registered
step `3e-4` gave a value 9.6e-6 higher (limit 1e-6). The registered step was too coarse.
The central difference against the same `jax.grad` value at five steps:

| Step | Relative difference |
| --- | --- |
| 3e-3 | 9.57e-4 |
| 1e-3 | 1.06e-4 |
| 3e-4 | 9.57e-6 |
| 1e-4 | 1.06e-6 |
| 3e-5 | 1.26e-7 |

The difference falls as the square of the step: the central difference's own truncation
error, converging on the AD value. Torch was not evaluated before this diagnosis.

MM4b, registered after this diagnosis and before it was run (JAX's plain central
differences above were already seen; the combination below was not computed): the MM4
derivative, by `jax.grad` and by Torch autograd, against the fourth-order combination
`(4 D(h/2) - D(h))/3` of central differences at `h = 3e-4` and `h/2`, to a relative 1e-7.

MM6, registered after a mutation run showed that MM1 cannot see how a finite span is
cut out of the plane (every MM1 window spans its whole cross-section), and before it
was run: the MM1 slab with its `PEC` faces at x = ±4.5 µm, launched as in MM1 (modes 0
and 1, both directions), read by a `ModeMonitor` and a `FluxMonitor` whose planes span
only x = (-3, 3) µm. The window's walls then truncate the run's mode where its field is
small but not zero, so the limits are looser than MM1's: the opposite direction and the
other mode each at most 1e-8 of the launched amplitude, and the launched mode's
spectral energy equal to the flux on the same span to a relative 1e-8.

### ModeMonitor results (2026-10-27)

Measured with `tests/fdtd/test_fd_mode_monitor.py` (same environment as above).

| Criterion | Limit | Measured | Verdict |
| --- | --- | --- | --- |
| MM1 opposite direction / other mode / energy vs flux, 12 launches | 1e-10 | 4.7e-15 / 6.8e-16 / 4.8e-16 (worst) | passed |
| MM2 seven-mode sum vs flux, broadband pulse | 1e-6, settled | 4.5e-5, not settled | **failed** (kept as an expected failure) |
| MM2b seven-mode sum vs flux, narrowband pulse | 1e-6, settled | 1.7e-10 at both monitors; 5.0e-15 of the peak energy left | passed |
| MM3 backends, MM2 amplitudes | 1e-12 | 6.2e-16 (JAX), 3.4e-16 (Torch) | passed |
| MM4 AD vs central difference at step 3e-4 | 1e-6 | 9.6e-6 | **failed** (kept as an expected failure) |
| MM4b AD vs fourth-order extrapolated difference | 1e-7 | 1.27e-8 (JAX and Torch) | passed |
| MM5 refusals | by code | raised by `check` and by `run` (the mode count at `run`) | passed |
| MM6 finite span: opposite direction / other mode / energy vs flux | 1e-8 | 3.1e-13 / 0 / 7.2e-13 | passed |

In MM2b the +z monitor reads backward energy only in the two slowest modes, 2.8e-5
(TE21) and 1.5e-6 (TM21) of the forward total. That is the end PML returning the
near-cutoff modes, which it absorbs worst. It is measured here, not judged.

Mutation checks (each made, the test failed, the source restored byte for byte):
swapping the forward and backward combinations, dropping the cosine factor of the H
overlap, reading one H half plane instead of their mean, dropping the spectral scale
(MM1); shifting the window crop of a finite span (MM6); removing the early check of the
window (MM5); solving the window with every structure instead of those that reach it
(MM4b); summing six modes instead of seven (MM2b).

### The window across an Open face (2026-10-27)

A `ModeMonitor` across an `Open` face was first refused at run time. Its flux plane read
the window's edge node, which on the physical edge is the absorbing layer's first node.
The window now stops one cell inside the physical edge on each `Open` side, for the
monitor and for `solve_modes`, and a regression test holds the two to the same modes.
The M3 slab, whose faces are `Open`, was re-run with the moved walls: every quoted
digit above is unchanged (the field there is below 1e-9 of its peak).

