# Separating lattice, boundary and finite-time scattering errors

Written: 2026-09-09

Targets:
- `wavebit/_core/conventions.py`
- `studies/fdtd_lattice_reference.py`
- `studies/fdtd_lattice_time_check.py`
- `studies/fdtd_rayleigh.py`
- `tests/test_fdtd_lattice_reference.py`

TL;DR: an independently assembled harmonic Yee equation separates spatial
discretization from finite CPML and finite-duration errors in the near-Rayleigh
dielectric case. With exact outgoing grid boundaries, complex amplitudes
refine approximately at second order against RCWA. The original fine-grid
TE discrepancy contains a substantial boundary contribution. Increasing PML
width and strength together removes that contribution in the measured band;
strength alone can worsen the coarse-grid result. These measurements do not
change defaults or establish general near-cutoff accuracy. A longer run with
the shaped pulse reaches the matched harmonic limit within 8.5e-6 TE and 2.4e-6 TM
on the coarse grid, leaving the independently measured lattice discrepancy.

## Three comparisons, with different meanings

The [source-clearance controls](fdtd-source-clearance.md) moved the source and
boundary positions separately. Both perturbations changed the answer only
slightly, while its discrepancy from RCWA remained appreciable. Position
insensitivity cannot bound an error shared by the compared runs.

The new study solves the same spatially discretized Maxwell equations directly
at each requested frequency. This avoids estimating the answer from a truncated
pulse record. Direct frequency-domain linear solving is also described in the
[public Meep introduction](https://meep.readthedocs.io/en/latest/Introduction/).
No external implementation was read or copied for this reference.

| Difference of complex amplitude arrays | What it measures here |
| --- | --- |
| Outgoing-boundary harmonic lattice minus RCWA | Yee dispersion and material rasterization at this resolution |
| Finite-CPML harmonic lattice minus outgoing-boundary harmonic lattice | Error from replacing the exact discrete exterior with finite CPML |
| Finite-time run minus finite-CPML harmonic lattice | Remaining time truncation with the same source location, mesh and boundaries |

Each reported scalar is a maximum absolute difference over TE or TM complex
r/t, the two requested wavelengths and orders -1/0. The arrays telescope;
their maximum norms do **not** add. Cancellation can make a less accurate
boundary or a shorter run look closer to RCWA.

## Independent harmonic construction

For the native leapfrog step, the harmonic derivative frequency is
`Omega = 2 sin(omega dt/2) / dt`. The transverse Yee symbol is
`Kx = 2 sin(kx dx/2) / dx`. In the vacuum exterior, let
`q = sqrt(1 - (Kx/Omega)^2)` on the passive branch. The longitudinal phase is
`phi = 2 asin(q Omega dz/2)`, with the outgoing continuation for both evanescent
and lattice-gap modes. The cell multiplier `lambda = exp(i phi)` satisfies
`lambda + 1/lambda = 2 - (q Omega dz)^2` and has magnitude at most one.
That branch choice lives in the common conventions module.

The scalar sparse operators use Ey for TE and Hy for TM, with their native
staggered material components. Every transverse Fourier harmonic participates
in the outgoing boundary map; only the reported diffraction amplitudes are
restricted to orders -1 and 0. Exterior ghost cells apply the exact lattice
multiplier. An incident Fourier mode supplies the top boundary forcing.
The interface phase reference and the physical s/p amplitude normalization
are retained, including the TM conversion from discrete Hy ratios.

The second reference keeps the entire finite CPML grid and its periodic outer
wrap. For a memory recurrence `psi_new = b psi_old + a derivative_new`, the
harmonic derivative stretch is
`S = 1/kappa + a / (1 - b exp(i omega dt))`.
Electric and magnetic stretches enter their respective staggered derivatives.
The actual and homogeneous-reference current sheets are solved separately;
independent E/H direction separation then calibrates their r/t amplitudes.
The current phase includes the half-time offset of the native electric update.

Both references share the scene's material rasterization. The finite-CPML
reference also shares the coefficient generator. They independently assemble
and solve the frequency equations, reconstruct fields and extract amplitudes;
they do not independently validate the rasterizer or profile formula.
They are study instruments for a y-invariant, uniform, nonmagnetic,
nondispersive dielectric grid with vacuum leads, thickness 0.2 and top
illumination. They are not public solvers for arbitrary materials or geometry.

## Checks before interpreting the grating

The validation suite uses criteria that do not depend on the grating errors:

- Vacuum gives zero reflection/diffraction and the exact discrete propagation
  phase. A uniform slab matches a separately derived two-interface recursion
  to 1e-11, using discrete TE admittance `sin(kz dz)` and TM admittance
  `tan(kz dz/2)/epsilon`. Its amplitudes refine toward continuous Airy optics.
- Moving exact outgoing truncation planes preserves r/t to 1e-11.
- The staggered Poynting flux balances between the two vacuum sides of the
  lossless binary grating to relative tolerance 1e-11, for both polarizations
  and wavelengths. This uses the discrete E/H flux, which differs from
  applying continuum power weights to lattice amplitudes near cutoff.
- Reconstructed E/H phasors obey one native Maxwell time update in the
  interior to 1e-11 on NumPy, JAX and Torch CPU.
- With manufactured harmonic CPML memory states, reconstructed fields obey
  the **entire** native update, including the source and absorbing cells,
  to 1e-10 on all three CPU backends. The memory initialization is a test hook,
  not a new public restart facility.
- A settled slab pulse matches the outgoing reference to 1e-6 and the matched
  finite-CPML reference to 1e-8. This checks source phase, direction separation
  and normalization as well as the sparse field solve.

These checks distinguish exact recurrence satisfaction from continuum
accuracy. Agreement with native one-step updates alone would not prove the
physical boundary problem is correct; the independent slab checks matter.

## Intrinsic grid error near the Rayleigh threshold

The geometry is the binary dielectric grating in the
[CPML-strength study](fdtd-pml-strength.md): permittivities 2.25 and 1,
period 0.8, thickness 0.2, fixed transverse wavevector
`kx = 2 pi (0.2/0.96)` and wavelengths 0.95808/0.96192. RCWA order 32 is the
reference, previously checked against orders 16 and 64. The same layer-aligned
rasterization and directional amplitude convention are used here.
The harmonic sparse solves use NumPy/SciPy double precision on CPU.

| Resolution | Outgoing lattice vs RCWA, TE | Outgoing lattice vs RCWA, TM |
| ---: | ---: | ---: |
| 40 | 0.0278171 | 0.0507211 |
| 80 | 0.00658717 | 0.0115905 |
| 160 | 0.00163124 | 0.00298847 |
| 320 | 0.000408289 | 0.000762524 |

Relative sparse residuals range from about 1e-14 to 3e-13. The roughly
fourfold error decrease per doubling separates this lattice/rasterization
error from the long pulse tail. It does not extend second-order convergence
to metallic corners or dispersive interfaces.

For the original resolution-40, 800-period Gaussian source control, the
finite-time answer differs from this same-grid outgoing reference by
0.0121813 TE and 0.00340953 TM. Enlarging padding with the source fixed gives
0.0121723/0.00340951; moving only the source gives 0.0121999/0.00341445.
The shared discrepancy survives both position controls.

## Finite CPML error is not monotone in strength or width

All rows below fix the source at z=-1, padding at 1 and extraction planes
at z=-0.2/0.4. No time integration enters this table. The target is the
nominal CPML amplitude parameter, not measured reflection power or an error
bound. Table entries compare the **finite-CPML harmonic amplitudes with the
exact outgoing harmonic amplitudes**, at identical resolution.

| PML width | Nominal target | Resolution 40, TE / TM | Resolution 80, TE / TM |
| ---: | ---: | --- | --- |
| 0.6 | 1e-40 | 2.89946e-3 / 6.15473e-4 | 1.06107e-2 / 2.56181e-3 |
| 1.2 | 1e-40 | 1.28141e-2 / 2.81499e-3 | 1.04091e-2 / 2.59448e-3 |
| 0.6 | 1e-80 | 4.79515e-5 / 3.58790e-5 | 3.32824e-6 / 1.02690e-6 |
| 1.2 | 1e-80 | 3.95354e-5 / 8.23800e-6 | 6.95674e-4 / 1.68098e-4 |
| 0.6 | 1e-100 | 2.84110e-4 / 1.88751e-4 | 2.84423e-9 / 8.10003e-10 |
| 0.6 | 1e-160 | 5.12073e-3 / 2.79864e-3 | 6.29352e-9 / 3.14283e-9 |
| 1.2 | 1e-160 | 5.19938e-9 / 4.10381e-9 | 7.31056e-11 / 2.49844e-11 |

At resolution 80 the original width-0.6, target-1e-40 TE boundary discrepancy
exceeds its intrinsic lattice discrepancy. At resolution 40 that boundary
moves the TE answer toward RCWA: the finite-CPML error against RCWA is
0.0249919, smaller than the exact-exterior lattice error 0.0278171.
This is a concrete cancellation, not improved lattice accuracy.

The original 800-period resolution-40 Gaussian result differs from the
matched finite-CPML limit by 0.0114135 TE and 0.00371478 TM. Time truncation
therefore remains measurable even after distinguishing the boundary term.

The width-1.2, target-1e-160 choice makes the finite-boundary difference very
small in these two-frequency solves. A frequency solve alone does not show
that a time update is stable or that a finite pulse record has settled.
It also does not establish a useful universal PML setting: the same strength
with width 0.6 substantially worsens the coarse grid. Defaults stay unchanged.

## Actual time evolution with the improved boundary

The matched time-domain runs use width 1.2, target 1e-160, source z=-1,
Gaussian width 60 and two repeated spectral zeros at the discrete vacuum
cutoff. The zeros are 1.04075226066 at resolution 40 and 1.04143811785 at
resolution 80; they remove a tail-driving frequency, not either requested
measurement frequency. No harmonic result is fed into the time update.

| Resolution | Periods | Time error TE / TM | Error against RCWA TE / TM |
| ---: | ---: | --- | --- |
| 40 | 800 | 3.26394e-3 / 9.03322e-4 | 0.0292407 / 0.0512007 |
| 40 | 1600 | 1.34779e-4 / 3.74078e-5 | 0.0277921 / 0.0507285 |
| 40 | 3200 | 8.46480e-6 / 2.32742e-6 | 0.0278088 / 0.0507192 |
| 80 | 800 | 8.00971e-4 / 2.68474e-4 | 0.00720676 / 0.0117110 |

The resolution-40 run contains 133022 updates with dt=0.0115470053838 and
grid shape (264, 2, 32). Its complete 800-period prefix has time error
0.00326394 TE and 0.000903322 TM. Direct prefix-to-final amplitude changes
are 0.00313771/0.000868317. Both state decay and scattering convergence
remain false over that long comparison window, despite the much smaller
final discrepancy from the independent harmonic limit. The returning
transmission-side order -1 components are 1.04949e-4/3.95202e-5.
An independent 800-period run reproduces that long run's prefix amplitudes
within 3.7e-14, checking the prefix reconstruction with an actual separate
time evolution. It also fails both convergence flags.

Extending resolution 40 to 3200 periods uses 266044 updates. Its 1600-period
prefix agrees with the separately run 1600-period amplitudes within 1e-12.
The state-decay flag now passes. The full second-half amplitude changes are
1.31622e-4 TE and 3.66187e-5 TM, so the combined scattering flag still fails
its 1e-4 threshold even though the final independently measured time errors
are below 1e-5. The last-interval difference bounds neither the exact error
nor its decay rate by definition; the harmonic comparison supplies separate
evidence here. Returning transmission order -1 is
6.59390e-6/2.52672e-6.

The resolution-80 run has 133022 updates, dt=0.00577350269190 and grid
shape (520, 2, 64). Its final 16628 updates span 100 periods. The prefix's
time errors are 0.00172352/0.000575744, and its changes to the final answer
are 0.00162151/0.000541330. Both convergence flags remain false.
Returning transmission-side order -1 components are
0.000589434/0.000290335, still much larger than the steady finite-boundary
error in the harmonic solve. A returning component in a truncated spectrum
is therefore not an isolated steady boundary-reflection coefficient.

The remaining RCWA discrepancy is now close to the independently measured
lattice error. This is evidence about a particular pulse, duration and mesh,
not permission to skip time convergence or to infer continuum accuracy from
a small boundary error.

## Reproduce

```shell
python -m studies.fdtd_lattice_reference
python -m studies.fdtd_lattice_reference --resolution 40 80 --cpml
python -m studies.fdtd_lattice_reference --resolution 40 80 --cpml --pml-width 1.2 --pml-reflection 1e-160
python -m studies.fdtd_lattice_time_check --resolution 40 --periods 1600
python -m studies.fdtd_lattice_time_check --resolution 40 --periods 3200
python -m studies.fdtd_lattice_time_check --resolution 80 --periods 800
```

The time study uses double-precision JAX CPU by default (`--backend numpy`
selects the common NumPy engine). Its default source has two repeated zeros
at the discrete vacuum cutoff, Gaussian width 60, PML width 1.2 and target
1e-160. `--zeros 0` or `--zeros 1` selects the corresponding control.
Long runs compare their full second half; the 800-period run
compares the final 100 periods. JSON output retains final and prefix complex
amplitudes and their direct differences from both harmonic limits.
Optional `--time-records` arguments to the harmonic study accept saved JSON
rows from the source-clearance study. Finite-CPML comparisons require matching
source, padding, profile, time step and resolution.
