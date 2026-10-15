# Rotated elliptical FDTD shapes and derivatives

Written: 2026-09-09

Targets:
- `src/wavebit/fdtd/_periodic/scene.py`
- `src/wavebit/compat/legacy.py`
- `src/wavebit/manifest.py`
- `tests/fdtd/test_fdtd_ellipses.py`
- `tests/fdtd/test_fdtd_ellipse_optics.py`
- `studies/cross_solver/ellipse_gradients/fdtd_ellipse_gradients.py`
- `tests/references/interface/meep_rotation.py`
- `tests/references/test_meep_ellipses.py`

TL;DR: the common area route accepts rotated ellipses with live centre, two
principal-axis lengths, angle and scalar material properties on NumPy, JAX and
Torch. Angular strip integration retains the analytic boundary on a fixed
mesh. Independent Cartesian chord quadrature checks cell geometry; independent
Bessel-series optics and public-API Meep runs check a matching graded target.
Neither accurate cell integration nor gradient repeatability establishes full
curved-interface effective tensors, general sharp-interface optical sensitivity,
GPU execution or general second derivatives.

## Shape and cell integration

The [guide](../../guide/fdtd/time-domain.md#elliptical-shape-derivatives) preserves the
shared RCWA reader: `lx`/`ly` are full principal-axis lengths, `angle` is radians,
and centres are measured from the period centre. One shape per layer keeps
periodic images disjoint. The rotated ellipse's projected x/y extents must fit
the two periods; a principal axis need not fit its corresponding unrotated
period. The full y mesh stays fixed even at zero rotation or equal axes.

Write the ellipse shape matrix as
`S = R(angle) diag(a*a,b*b) R(angle).T`, where `a=lx/2`, `b=ly/2`.
With `A=sqrt(Sxx)`, its vertical chords are
`y = cy + (Sxy/A)*sin(theta) +/- (a*b/A)*cos(theta)` at
`x = cx + A*sin(theta)`. Intersections with horizontal box faces and `x=cx`
split the integration into intervals of angular span at most `pi/2`.

Within each interval the lower and upper chord limits either follow an
ellipse arc or a box face. Multiplying by a bilinear translation kernel and
integrating over y leaves a trigonometric polynomial of degree at most four.
A fixed 16-point Gauss rule evaluates this smooth angular integral. This method
applies to both sharp and translation-averaged ellipse cells. It is numerical
angular integration, not polygon clipping, an intermediate raster or a custom
backward rule. All geometry coordinates stay in the common array graph.

Bounds and inactive tangent roots are guarded before square roots and inverse
sines. An exact projected contact can round a few units inside `abs(x/A)=1`,
where the inverse-sine slope amplifies angular integration roundoff. Contacts
within four machine-epsilon units use the limiting endpoint; the associated
area change scales as `eps**1.5`. A first candidate's period-integrated centre
slope was about `1e-9` at a nominal exact contact, instead of zero. The endpoint
guard restores the tested total-area derivative without changing a test budget.
Shared stored cell faces avoid introducing artificial gaps between adjacent cells.
General second derivatives and uniform derivative accuracy arbitrarily close
to every tangent contact are not claimed.

Positive smoothing averages translations along the laboratory x/y axes. Its
box remains fixed when the ellipse rotates. Local window coordinates avoid
subtracting global moments divided by narrow smoothing widths. Zero-measure
windows created by rounded static knots use safe inactive denominators.

## Independent geometry and complete graphs

The geometric oracle constructs the implicit quadratic form
`Q = R diag(1/a**2,1/b**2) R.T`, solves its Cartesian chord roots, and integrates
with independent adaptive quadrature. It does not use the native angular
parameterization or its Gaussian nodes. Tests cover four angles, periodic seams,
staggered cells, five shape derivatives at two difference steps, total area
`pi*lx*ly/4`, its exact derivative and projected tangent contacts.

Equal axes recover the previously verified disk solver and its shared diameter
derivative. Swapping the two axes while adding `pi/2` to the angle preserves
the physical shape. Narrow and full-period translation kernels, long axes that
fit only after rotation, and thin ellipses broaden the geometry checks.

A bottom-incidence conical Jones scene tests all eight joint parameters:
centre x/y, both axis lengths, angle, epsilon, mu and one layer thickness.
JAX/Torch complete-graph derivatives agree with two NumPy perturbation steps;
mesh nodes, time step, source and PML coordinates remain fixed. One JAX trace
accepts periodic translations and changing rotation, and masks invalid live
extents/materials with NaN scattering and `stable=False`.

## Independent graded optical target

The benchmark uses periods `(0.6,0.6)`, layer depth 0.2, background epsilon 1,
ellipse epsilon 2.25, substrate epsilon 1.21, all mu values 1, wavelength 1 and
normal incidence. The ellipse has angle 0.37, full axes `(0.34,0.2)` and centre
`(0.071,-0.047)`. A physical translation width 0.05 is held fixed along both
laboratory axes. All eight complex s/p reflection/transmission channels and
three derivatives `(angle,lx,ly)` are compared at the layer faces.

An independent affine disk transform supplies Fourier coefficients
`pi*a*b * 2*J1(q)/q`, where `q*q=(a*u)**2+(b*v)**2` and `(u,v)` are rotated
spatial frequencies. Box sinc factors account for the same translation average.
Analytic coefficient derivatives follow from the
[Bessel derivative identities](https://dlmf.nist.gov/10.6), including
`d[J1(q)/q]/dq=-J2(q)/q`. DC limits are explicit. In the circle limit, the angle
derivative vanishes and the sum of the two axis derivatives equals the disk's
diameter derivative.

The primary profile uses a cell-centred grid of 256 and clips small Fourier
ringing to `[0,1]`; RCWA order is `(8,8)`. This permits the same scalar graded
profile in Meep MaterialGrid. Doubling the profile grid changes complex
amplitudes by `1.11e-6` and the three derivatives by
`[1.01e-4,3.39e-4,7.52e-4]`. Raising RCWA order to `(10,10)` changes amplitudes
by `2.51e-6` and derivatives by `[1.10e-5,3.27e-5,1.17e-4]`. Halving the centred
difference step `1e-5` changes them by `[3.97e-8,1.18e-7,1.91e-6]`.
These controls include the finite profile and clipping approximation.

Native FDTD uses physical PML thickness 1, pulse width 2.5, duration 60 optical
periods, aligned layer boundaries and directional extraction. All recorded
calculations use CPU float64.

The table reports maximum absolute differences across the eight complex
channels against the independent graded RCWA target. Axis derivatives are per
unit length; the angle derivative is per radian. These are not relative errors.

| Solver / resolution | Amplitude | Angle derivative | `lx` derivative | `ly` derivative |
|---|---:|---:|---:|---:|
| Native / 20 | 8.15e-3 | 1.51e-3 | 1.56e-2 | 2.06e-2 |
| Native / 30 | 3.73e-3 | 4.23e-4 | 6.42e-3 | 8.76e-3 |
| Native / 40 | 2.13e-3 | 2.95e-4 | 3.34e-3 | 4.96e-3 |
| Meep / 20 | 1.59e-2 | 1.38e-2 | 7.81e-2 | 1.89e-1 |
| Meep / 40 | 7.67e-3 | 1.61e-3 | 4.05e-2 | 5.42e-2 |

All four error measures decrease with spatial refinement in each solver.
Native maximum energy-balance errors decrease from `2.67e-3` through `1.17e-3`
to `6.52e-4`. At resolution 40, extending the native duration from 60 to 80
changes amplitudes by `1.22e-11` and derivatives by
`[1.35e-11,1.53e-10,1.74e-10]`.

For Meep's resolution-40 run, extending duration from 80 to 100 changes
amplitudes by `1.41e-11` and derivatives by at most `1.84e-10`. Halving its
parameter step changes the three derivatives by at most
`[5.77e-7,1.65e-5,7.32e-6]`. Doubling the independent profile grid at resolution
20 changes amplitudes by `1.59e-5` and derivatives by
`[1.38e-4,1.16e-3,1.43e-3]`. All three public material probes have maximum
error `4.45e-16` or less. Thus the observed solver differences exceed the
measured duration, difference-step and profile-grid controls. Their cause is
not established by these controls; the remaining axis-derivative differences
mean Meep is a refinement cross-check here, not a high-precision gradient oracle.

Comparing the primary graded target with a sharp RCWA calculation at normal-field
grid 512 and order `(10,10)` changes amplitudes by `6.25e-3` and derivatives by
`[2.20e-3,1.44e-2,2.09e-2]`. This is a separate target-bias estimate, not a
validation of native sharp-interface optical derivatives. Fixing the physical
smoothing width during refinement is essential to keep the target unchanged.

## Public-API Meep reference boundary

The existing reference bridge accepts `--shape ellipse` for positive smoothing.
It requires a graded profile for this mode; it does not substitute a finite
three-dimensional ellipsoid for an extruded elliptical layer. The public
[MaterialGrid API](https://meep.readthedocs.io/en/latest/Python_User_Interface/#materialgrid)
receives the independent Bessel profile with internal averaging disabled.
Its placement is checked through public epsilon-grid probes before scattering
is measured. No Meep implementation is opened, copied or used as a Wavebit
dependency. Shape sensitivities here are central differences, not adjoints.

The reference domain is `(0.6,0.6,8)`, transversely periodic, with z PML thickness
1 and Courant factor 0.4. A sine-Gaussian source at z=-2 has width 2.5, peak 12.5
and end 25. Every-step DFT monitors at z=-1 and 1.2 use public integration
weights, empty-cell source calibration and discrete reference-face phases.
Both polarizations are run. Two durations (80/100), two parameter steps
(`1e-4`/`5e-5`), resolutions 20/40 and an independent profile-grid doubling
(256/512 at resolution 20) remain in the committed archives.

Run `python -m studies.cross_solver.ellipse_gradients.fdtd_ellipse_gradients --output ellipse.json`
for the native convergence and controls; `--smoothing 0` selects a separate sharp-target
study. Generate external candidates with
`python tests/references/interface/meep_rotation.py --shape ellipse --smoothing .05
--resolution 20 --output candidate.npz`. Repeat at resolution 40 and separately
with `--profile-grid 512` at resolution 20. A candidate must pass its controls
before it replaces a golden; regeneration does not repair a failing assertion.
