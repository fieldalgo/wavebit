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

## Results

Pending. Results are added below, beside the criteria, once computed.
