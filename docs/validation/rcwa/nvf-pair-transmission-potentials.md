# Fixed local transmission potentials for the pair cell

Written: 2026-09-16

Targets:

- `src/wavebit/rcwa/_engine/conforming/cartesian_pair_transmission_galerkin.py`
- `src/wavebit/rcwa/_engine/conforming/pair_transmission_fields.py`
- `src/wavebit/rcwa/_engine/conforming/pair_transmission_modes.py`
- `src/wavebit/rcwa/_engine/conforming/pair_transmission_mode_precision.py`
- `src/wavebit/rcwa/_engine/conforming/pair_transmission_common_precision.py`
- `src/wavebit/rcwa/_engine/conforming/pair_transmission_extended_precision.py`
- `src/wavebit/rcwa/_engine/conforming/pair_extended_precision.py`
- `tests/rcwa/test_nvf_pair_transmission.py`
- `tests/rcwa/test_nvf_transmission_common_precision.py`
- `tests/rcwa/test_nvf_transmission_mode_precision.py`
- `tests/rcwa/test_nvf_transmission_cutoff_precision.py`
- `tests/rcwa/test_nvf_extended_precision.py`
- `studies/rcwa/nvf_pair_angular_resolution/nvf_pair_angular_resolution.py`

The explicit NumPy transmission family supplies a separate finite trial space
whose corner directions satisfy a fixed reference transmission equation.
It preserves the original complete tensor and physical guards. Measurements
still depend on the number of modes, polynomial degree and mesh refinement;
this development does not establish optical convergence or replace the
existing fixed-power family.

| Choice | Meaning |
| --- | --- |
| Reference coefficients | Fixed left, right and background values defining the trial functions |
| Physical coefficients | Live materials passed to `closure`, independently of the reference |
| One through four branches | Successive noninteger growing transmission roots at all six corners |
| Cutoff order zero through two | At degrees 5 through 8, include every `h*(1-h)**m` multiplier through that order; zero retains the original space |
| Real directions | Both real and imaginary parts of complex modes; one direction for a real mode with real relative coefficients |
| Common coordinates | A continuous Pp interpolation subtraction and complete real QR within each corner group |

## Local equation and fixed basis

For a scalar reference coefficient `a`, a separated potential
`u(r, theta) = r**lambda * f(theta)` satisfies `div(a*grad(u)) = 0`
inside each constant sector when `f'' + lambda**2*f = 0`. Across an interface,
both `f` and `a*f'` are continuous. Propagating the state
`(f, a*f'/lambda)` through an angle `t` therefore uses

```text
M(a, t) = [[cos(lambda*t),       sin(lambda*t)/a],
           [-a*sin(lambda*t),   cos(lambda*t)  ]].
```

For quarter sectors `a`, `b` and a half sector `c`, the noninteger periodic
roots obey the characteristic obtained by multiplying these three matrices:

```text
A = a/b + b/a
B = (a+b)/c + c*(1/a + 1/b)
cos(pi*lambda) = -(4+B)/(2+A+B).
```

Near equal quarter-sector coefficients, subtracting nearly equal terms before
`acos` can round the characteristic to -1 and lose the noninteger separation.
When that distance is below the square root of machine epsilon, the same
principal root is evaluated from the factored half-angle identity

```text
q = ((a-b)/(a+b)) * ((a-b)/(a+c)) * (c/(b+c)) / 2
lambda = 1 - 2*asin(sqrt(q))/pi.
```

The ordinary characteristic chooses the existing root branch. The final root
is refined in complete high/low arithmetic using
`sin(pi*(1-lambda)/2) = sqrt(q)`. Solving for this separation retains weak
contrasts without differentiating or refitting the fixed reference space.
Independent 100-digit checks cover real contrasts from 1e-4 through 1e-12,
on both sides of equality and for all four branches, at a 3e-28 bound. The
rounded standard roots can change by their last floating-point bit; their
branch choices and defining equation remain the same.

The implementation uses the principal growing root and the successive roots
`2-lambda`, `2+lambda`, `4-lambda`. Forming the monodromy minus identity in
ordinary arithmetic loses the angular state when the contrast is weak.
Instead, the implementation forms `(M-I)/(a-b)` algebraically, using
`t = cos(pi*lambda/2) = (a-b)*h`, `h**2 = c/(2*(a+b)*(a+c)*(b+c))`.
A nonzero row of this complete two-by-two matrix determines its null vector.
The resulting potential and flux state propagates through all three sectors;
the original closure check remains unchanged. Both words of the roots,
normalized coefficients and states enter complete field assembly. Ordinary
sampling uses their rounded values.

Independent 100-digit characteristic and transfer equations verify eight real
and complex references, including 1e-12 contrasts, for every branch. The worst
measured transfer or full-turn discrepancy was 3.49e-31. Separate reflected
field tests align only the scalar normalization of two independently generated
modes and compare their entire values and Cartesian gradients at 3e-28. In a
metal junction diagnostic, ordinary root/state parameters gave a reflection
defect as large as 1.42e-14; complete parameters reduced it below 6.30e-31.
These checks concern the local functions. They do not certify conditioning
of an arbitrarily near-degenerate complete field bank or optical convergence.

A nonpositive real part is
rejected because the local gradient energy is not integrable. Equal quarter
coefficients give a straight interface with no noninteger corner direction.
At a material junction, a closed vertex becomes an edge of an opening
triangle. The squared gradient trace then contains
`integral(r**(2*Re(lambda)-2), dr)`, requiring `Re(lambda) > 1/2`.
The pair field factory rejects the entire reference bank if any junction
mode violates this opening condition. Outer-corner hats vanish on all
collapsing triangles, so their bulk condition remains `Re(lambda) > 0`.
No numerical rank or field-strength threshold chooses the surviving modes.
The sector coefficients are divided by the reference background before
constructing the modes. This common factor leaves the equation unchanged and
preserves the real trial bank under a purely imaginary multiple of real
reference materials, without introducing an identically zero direction.

Corner singularities for sign-changing coefficients have an established
literature, including [Bonnet-Ben Dhia, Carvalho, Chesnel and Ciarlet](https://arxiv.org/abs/1511.01932).
That paper's lossless scattering and corner radiation problem is distinct
from this lossy auxiliary-potential cell. Its critical-contrast conclusions
are not used to declare a divergent derivative here. The equations above
and their implementation were derived directly; no external solver source
or new dependency is integrated.

For reference `(12.25, -10+i, 1)`, the leading roots are approximately
`.720879888` at the dielectric outer corners,
`.582592739-.009794237i` at the metal outer corners and
`.914430391-.517977968i` at a material junction. Their positive real parts
permit finite local energy. These local numbers are not a global error bound
or an optical stopping rule.

## Common field and derivative

Each mode is multiplied by its closed corner hat. The same reference
function is transported with the fitted mesh as the gap opens. Its common
polynomial interpolant is subtracted before the complete real coordinate
change. The cell retains all selected directions in its weak equations,
normal traces, Fourier moments, response and analytical variation.

The real trial space is fixed when physical materials change. In particular,
the implementation does not recompute the reference exponent during a
material JVP. Geometry is fixed in that JVP, as in the original pair family.
Spatial gradients and Hessians differentiate the defining local function;
the Hessian combines homogeneity with the harmonic equation. Moving
barycentric coordinates and the inverse triangle map supply the transported
gradient variation.

At an interface, the triangle selects the limiting material sector.
Selecting a side only from a boundary point's rounded polar angle is wrong:
the potential is continuous but its normal gradient has different limits.
In an independent order-four cubic prototype, correcting this choice reduced
the maximum full derivative reflection defect from about `24.36` to
`2.92e-12`, without changing any response check.

Independent spatial differences verify all four local branches and complete
degree-3 through degree-6 field transport on the refined mesh. Whole contact controls
retain reflection, separate angular integration and length-unit identities
at absolute `3e-10` / `3e-9`. At positive gap `.02`, a two-branch order-zero
cell with physical materials `(4+.1i, 9+.2i, 1+.02i)` and the distinct fixed
reference `(12.25, -10+i, 1)` also passes an independent fourth-order joint
gap/material difference at `3e-9`. That test evaluates five complete physical
states and took 522 seconds on the stated runtime.

A separate metallic one-branch control evaluates exact contact and physical
gaps `.01`, `.0001`, `.000001` at period 700 and Fourier order zero. Every
state passes the original guards. Values approach the same contact response
and the complete derivative differences decrease across all three gaps. This
four-state test took 461 seconds; it assigns no universal rate or error bound.

## Integration and measured optical limits

The reference space requires angular integration of nonpolynomial functions,
including on thin exterior triangles away from the corner itself. At one
mesh refinement, angular Gauss orders 24 and 32 produced full value reflection
defects `3.59e-7` and `9.08e-9`; both fail the original `3e-10` criterion.
The error was localized to four narrow exterior triangles. Order 64 passed
with value and derivative defects below `8.67e-13` and `5.82e-12`.

Reflection alone also misses common integration errors. For the unrefined
one-branch space, increasing angular order from 48 to 64 changed the entire
value and derivative by `1.54e-9` and `1.12e-8`, exceeding the original
`3e-10` / `3e-9` criteria. The subsequent 64/80 comparison passed, with
differences `2.24e-12` and `1.53e-11`. Radial order was 80 throughout.

The following contact measurements use period 700, widths and height 300,
physical and reference materials `(12.25, -10+i, 1)`, Fourier order `(4,4)`,
wavelength 900, layer thickness 300 and incidence angle `.2`. The runtime
was NumPy 2.5.1 with one numerical-library thread. Slopes are per caller gap
unit, obtained by differentiating the optical solver along the analytical
constitutive JVP. They are not independent physical gap finite differences.

| Branches | Degree | Refinement | Angular Gauss order | p/s transmission | p/s optical slope |
| --- | --- | --- | --- | --- | --- |
| 1 | 3 | 0 | 80 | `.48581215` / `.46831714` | `-.04935045` / `.01254798` |
| 1 | 3 | 1 | 64 | `.48964782` / `.47673311` | `-.01293277` / `.03202701` |
| 2 | 3 | 0 | 64 | `.48483934` / `.46813181` | `-.02432054` / `.01957287` |
| 2 | 4 | 0 | 64 | `.48699404` / `.47283323` | `-.02736622` / `.01666854` |
| 2 | 4 | 1 | 64 | `.48911480` / `.47894903` | `-.04408189` / `.01018199` |
| 2 | 5 | 0 | 80 | `.48962262` / `.47820719` | `-.04258852` / `.02204977` |
| 2 | 6 | 0 | 80 | `.48933095` / `.47905421` | `-.04922588` / `.01600262` |
| 2 | 6 | 1 | 80 | `.48936842` / `.47952728` | `-.04493608` / `.01781915` |
| 4 | 3 | 0 | 64 | `.48475992` / `.47363551` | `-.00076895` / `.03447941` |
| 4 | 4 | 0 | 80 | `.48583765` / `.47515927` | `-.06005934` / `.02990420` |
| 4 | 5 | 0 | 80 | `.48946752` / `.47700685` | `-.04441011` / `.02749882` |

The 80-point quartic, four-branch row was evaluated by the actual library
and study entry point in 107 seconds. It passed every original physical and
full reflection check. After the known Fourier translation to the same
origin, its complete value and derivative differed from the independent
64-point prototype by `1.95e-12` and `1.25e-10`. The quintic two-branch
row also uses the actual library and study entry point, with radial order
100. It took 957 seconds and passed all original checks: energy residuals
were `1.04e-12` / `1.18e-12`, and full reflection defects were
`2.08e-12` / `1.96e-10`. A fresh sextic two-branch library run, with radial
order 120 and complete physical energy coordinates, took 1,585 seconds.
It passed every original guard, with energy residuals `1.11e-20` / `2.47e-12`
and full reflection defects `1.22e-11` / `1.48e-9`. Its complete value and
derivative differ from the preserved-system replay by only `2.42e-22` and
`1.39e-17`. The remaining rows are finite
prototype measurements, with complete physical and reflection checks. Raw
measurement records are not distributed. These changes across trial spaces
remain unresolved, even when individual finite equations are accurate.

The quintic four-branch row retains all 40 real transmission directions.
It re-solves freshly assembled complete weak matrices with the library
response and independent physical verifier, using radial order 100. That
replay took 970 seconds and passed all original guards, with energy residuals
`1.57e-17` / `8.92e-13` and full reflection defects `1.44e-11` / `3.11e-10`.
Increasing the corner bank still changes the optical slopes appreciably;
it does not close the approximation-convergence gate.

The refined sextic two-branch library control used radial order 120 and
angular order 80. It retained all 3,475 common coordinates and took 7,596
seconds. This measurement used the integrated geometric solve before the
complete vertex-measure and local transmission-parameter refinements. Its
original physical and full reflection checks passed: energy defects were
`1.55e-24` / `2.58e-12`, and full reflection defects were
`5.45e-12` / `7.19e-10`. The two optical slope stencils differ by at most
`2.51e-8`. The subsequent numerical paths require their own refined controls.
The change from the unrefined sextic space remains substantial, especially
in the optical slope, so this row does not establish approximation convergence.

The refined quartic two-branch prototype took 1,099 seconds and passed the
original guards, with full value and derivative reflection defects below
`3.76e-12` and `8.78e-11`. Its transmission and slope still change substantially
from the unrefined space. It does not close the approximation-convergence gate.

A fresh refined quintic two-branch library calculation uses the complete
normal Fourier means and independently reconstructed mean variation from
[the contact energy note](nvf-contact-energy-coordinates.md). At radial/angular
orders 100/80 and Fourier order `(4,4)`, it passed every original physical
and reflection guard in 6,311 seconds under concurrent workloads. Energy
and derivative-energy residuals were `1.61e-25` / `5.29e-22`; the physical
mean-variation residual was `3.59e-15`. Full-slot value and derivative
reflection defects were `7.24e-16` / `9.16e-16`.

Its p/s transmissions are `.489284990957` / `.479401083850`, with optical
directional slopes `-.044017967333` / `.017928929941`. The two slope stencils
differ by `3.39e-10` / `2.01e-8`. This fresh complete-state qualification
resolves the earlier refined derivative reflection failure; it does not
establish approximation or optical convergence.

A fresh sextic calculation on the same once-refined mesh and Fourier order
uses the same complete normal means, all 20 real transmission directions,
and radial/angular orders 120/80. Every original physical and full reflection
guard passes. Energy and derivative-energy residuals are `6.32e-25` /
`6.14e-20`; the independently reconstructed mean-variation residual is
`4.69e-15`. Maximum full-slot reflection defects are `7.24e-16` / `8.88e-16`.
Its p/s transmissions are `.489368420426` / `.479527278613`, and directional
optical slopes are `-.044936084696` / `.017819129094`. The two slope stencils
differ by `4.24e-9` / `6.08e-9`. As above, these slopes follow the constitutive
JVP through Maxwell propagation, not an independent finite-gap solve.

Relative to the fresh quintic calculation, transmission changes by
`+8.34e-5` / `+1.26e-4` and the slope by `-9.18e-4` / `-1.10e-4`.
The p-polarized slope changes by about 2.09% relative to the quintic value.
Thus the remaining trial-space dependence survives the corrected Fourier
means and independent physical verification; optical convergence is still
open. The earlier provisional saved-state sextic optics are superseded by
this complete qualification.

A fresh degree-seven calculation completes the same once-refined, two-branch
sequence at Fourier order `(4,4)`, with radial/angular integration orders
140/80. Its complete value and gap derivative pass every original physical
and reflection guard. Energy and derivative-energy residuals are `2.54e-22`
and `1.86e-18`; the independently reconstructed mean-variation residual is
`3.52e-15`. Full-slot reflection defects are `7.24e-16` / `4.44e-16`.

Its p/s transmissions are `.489376142234` / `.479557214869`, with optical
directional slopes `-.045131822981` / `.018022370310`. The two slope stencils
differ by `4.77e-9` / `4.58e-9`. Relative to degree six, transmission changes
by `+7.72e-6` / `+2.99e-5` and the slopes by `-1.96e-4` / `+2.03e-4`.
The relative p-slope change decreases from about 2.09% to 0.44%, but the
s-slope change increases from about 0.61% to 1.14% and reverses sign.
Thus the completed degree-five/six/seven sequence does not establish
convergence of both derivatives. These are resolution changes, not errors
against an independent optical golden value, and the Fourier orders have
not increased in this comparison. The calculation took 13,645 seconds,
including the original independent physical verification, with raw equations
saved after 4,564 seconds. It used the earlier arithmetic before row tiling;
this is not a timing comparison at matched optical accuracy.

The fresh degree-eight calculation extends that same once-refined, two-branch
sequence at Fourier order `(4,4)`, with radial/angular orders 160/80. It passes
every original physical and full reflection guard. Energy and derivative-energy
residuals are `2.31e-18` / `2.87e-17`; the independently integrated mean-variation
residual is `3.62e-15`. Maximum full-slot reflection defects are `7.24e-16` /
`8.88e-16`. Its p/s transmissions are `.489379581711` / `.479570175468`,
reflections `.283051190369` / `.171356821442`, and optical directional slopes
`-.045193763335` / `.018090695576`. The two slope stencils differ by
`2.51e-9` / `2.63e-10`.

From degree seven to eight, transmission changes by `+3.44e-6` / `+1.30e-5`,
reflection by `-1.51e-6` / `-1.05e-7`, and the slopes by `-6.19e-5` / `+6.83e-5`.
The relative p/s slope changes decrease to 0.137% / 0.379%, compared with
0.44% / 1.14% in the preceding degree step. Both latest changes therefore
shrink. These resolution differences do not bound the remaining error:
mesh and Fourier resolution remain fixed, the optical slopes still follow
the constitutive JVP rather than independently solved positive gaps, and
there is no certified independent optical reference for this fixture.

The complete run took 14,575 seconds, with raw equations saved after 3,910
seconds. It includes the original independent physical verifier and used
bounded three-word row tiling, before the fixed polynomial-gradient cache
described below. Different degrees and concurrent workloads prevent treating
this as a paired speed comparison with the degree-seven run. This finite
qualification does not close the production convergence or runtime gates.

A separate Fourier-resolution comparison holds the quintic once-refined,
two-branch space fixed, with radial/angular orders 100/80 and cutoff zero.
Increasing Fourier order from `(4,4)` to `(6,6)` passes every original physical
and full reflection guard. The latter calculation has energy and
energy-variation residuals `3.56e-22` / `9.41e-19`, independently reconstructed
mean-variation residual `4.20e-15`, and maximum full-slot value/derivative
reflection defects `6.13e-16` / `6.90e-16`.

| Fourier order | Transmission p / s | Reflection p / s | Optical directional slope p / s |
|---|---|---|---|
| `(4,4)` | `.489284990957` / `.479401083850` | `.283076728057` / `.171332352257` | `-.044017967333` / `.017928929941` |
| `(6,6)` | `.485471038870` / `.401024301210` | `.297075772571` / `.166846368605` | `-.035863765328` / `.019458755096` |

The p/s transmission changes are `-.00381395` / `-.07837678`, reflection
changes `+.01399904` / `-.00448598`, and slope changes `+.00815420` /
`+.00152983`. The magnitudes of the slope changes are 18.52% / 8.53% of
the preceding values. Thus shrinking polynomial-degree changes at fixed
Fourier order do not establish optical convergence. The `(6,6)` slope
stencils agree to `7.10e-9` / `3.25e-9`, but still propagate the constitutive
JVP rather than independently solving positive gaps. This comparison has
no certified independent optical reference.

The complete `(6,6)` run took 13,277 seconds, including the original
independent physical verifier, with raw equations saved after 1,783 seconds
and an observed peak resident memory of 3.87 GB. Concurrent workloads and
different retained Fourier sizes prevent treating these times as a paired
speed comparison at matched accuracy. Fourier convergence and practical
runtime remain open.

A subsequent quintic once-refined calculation at `(8,8)` keeps radial/angular
orders 100/80, two branches and cutoff zero. Every original physical and full
reflection guard passes. The p/s transmissions are `.494962537308` /
`.400355285583`, reflections `.297901145371` / `.153843914009`, and optical
directional slopes `-.031459516989` / `.014936181543`. The two slope stencils
differ by `4.76e-9` / `7.05e-9`.

Relative to the matched `(6,6)` calculation, transmission changes by
`+.00949150` / `-.00066902`, reflection by `+.00082537` / `-.01300245`, and
slopes by `+.00440425` / `-.00452257`. The magnitudes of the slope changes
are 12.28% / 23.24% of the preceding values. These resolution differences
are not errors against a known answer, and do not establish Fourier or
opening-derivative convergence. The slopes still propagate the constitutive
JVP rather than independently solving positive physical gaps.

The complete `(8,8)` run took 32,257 seconds, including the original physical
verifier, with an observed peak resident memory of 3.53 GB. Its energy and
energy-variation residuals are `2.17e-17` / `2.52e-17`, independently
reconstructed mean-variation residual `1.99e-14`, and maximum full-slot
value/derivative reflection defects `7.19e-15` / `5.30e-15`. Concurrent
workloads and different resolutions prevent a matched-accuracy speed claim.
This finite result does not close the convergence or practical runtime gates.

The corresponding once-refined sextic calculation at Fourier order `(6,6)`
uses radial/angular orders 120/80, two branches and cutoff zero. Every
original physical and full reflection guard passes. Energy and
energy-variation residuals are `8.15e-19` / `1.86e-17`; the independently
reconstructed mean-variation residual is `5.97e-15`. Full-slot value and
derivative reflection defects are `6.13e-16` / `4.16e-16`.

Its p/s transmissions are `.485491834179` / `.401176887420`, reflections
`.297067404219` / `.166832458235`, and optical directional slopes
`-.035890690295` / `.018412522101`. The two slope stencils differ by
`5.40e-10` / `1.73e-8`. Relative to degree five at the same Fourier order,
transmission changes by `+2.08e-5` / `+1.53e-4`, reflection by `-8.37e-6` /
`-1.39e-5`, and slopes by `-2.69e-5` / `-1.05e-3`. The relative p-slope
change is 0.0751%, but the s-slope change is 5.38%. Agreement in transmission
therefore does not establish derivative convergence.

Holding degree six and the once-refined mesh fixed while increasing Fourier
order from `(4,4)` to `(6,6)` changes transmission by `-.00387659` /
`-.07835039` and slopes by `+.00904539` / `+.00059339`. The slope changes
are 20.13% / 3.33% of the preceding magnitudes. Both Fourier and inner-space
sensitivity must still be resolved. These are finite resolution comparisons,
not errors against an independent optical reference or derivatives obtained
from separately solved positive gaps.

The complete sextic `(6,6)` run took 17,712 seconds, with raw equations
saved after 2,905 seconds and observed peak resident memory of 3.96 GB.
It includes the independent physical verifier and concurrent workloads;
it is not a runtime comparison at matched accuracy.

At Fourier order `(4,4)`, the fresh quintic and sextic runs took 6,311 and
9,912 seconds, respectively, under concurrent workloads. Their raw-equation
assembly checkpoints occurred at 1,230 and 2,971 seconds; those checkpoints precede the completed response
and independent physical verification. The verifier is part of this explicit
solver path, not merely an external optional test. These frozen runs predate
the bounded row tiling described below, and are not paired runtime benchmarks
against ordinary NVF. Ordinary NVF dispatch remains unchanged. Practical
runtime, along with optical convergence, remains an unresolved requirement
for a production replacement.

Reproduce the library measurement with

```shell
python -m studies.nvf_pair_angular_resolution --order 4 --degree 4 --refinement 0 --quadrature 80 --angular-quadrature 80 --transmission-branches 4
```

The study records source digests, full diagnostics and both polarizations.
Its optional tensor artifact preserves every value and derivative slot
before applying the unchanged strict reflection checks.

## Higher polynomial degrees and complete coordinate precision

The explicit transmission family admits common polynomial degrees 3 through
8. The existing dense bound `degree * 2**refinement <= 16` continues to
apply. The original fixed-power and polynomial family degree contracts are
unchanged. Increasing degree retains the complete Bernstein space, every
selected real transmission direction, and the full continuous Fourier trace.

At degrees 5 and 6, contact assembly combines the common polynomial
coefficients and every transmission remainder before rounding. The stored
reference exponent and angular states stay fixed. Compensated logarithms,
angles and complex powers evaluate those same functions, and their gradients
use the original triangle's one-sided material sector. No material direction
changes the reference coefficients or refits the field space.

The field-specific common-coordinate provider feeds the existing complete
contact equations. Their [free Fourier moments](nvf-contact-fourier-precision.md)
retain the same precision as K, Q and R. All original physical equations and
energy checks still apply. The established degree-3 and degree-4 evaluations
retain their original path. At positive gap, the higher degree supplies the
same complete polynomial and transported transmission field space.

Independent 100-digit Cartesian complex-power formulas verify all twelve
reference modes in three sectors and all four corner orientations at radii
0.2 and 1e-70. These check the
potential and both Cartesian gradient components. Complete potential finite
differences also verify arbitrary mixed polynomial/transmission coordinates
at degrees 5 and 6 and on a refined degree-5 mesh, retaining the original
3e-9 derivative bound. Separate moving-field differences cover all four
supported degrees, including collapsing triangles.

The compensated common-coordinate sampler also accepts a moving barycentric
sample. It differentiates each sector's harmonic homogeneous potential,
including the cutoff and polynomial interpolant, before applying the moving
physical inverse map. The inverse map and the moving sample contribute
separately to the same gradient variation. Refined chart offsets retain both
words when differentiated; rounding their difference first loses precision.
Independent 100-digit potential differences check complete mixed coordinates
at degrees 5 and 6 and refinement 1 against a relative `3e-28` bound. A zero
sample direction preserves the static path bit for bit. This primitive check
does not qualify a whole positive-gap response or optical convergence.

The degree-5 positive-gap control also evaluates five complete physical
states against the independent fourth-order joint gap/material difference
described above. It passes the unchanged 3e-9 bound and took 832 seconds.
Uniform-material contact controls cover degrees 5 and 6, including the
empty transmission bank and joint variation. The initial degree-6,
order-(4,4), two-branch study failed an energy-derivative guard. The
[energy-coordinate note](nvf-contact-energy-coordinates.md) isolates and
corrects coefficient handoff and collapsed-measure rounding without changing
that guard. These finite checks do not establish high-order optical convergence.

The initial sextic order-(4,0) positive-gap metal control at gap `.01`
was rejected by the original guards. It uses the flat auxiliary reference
`(2,2,2)`, physical materials `(12.25,-10+i,1)`, period 700, widths/height 300,
radial order 120 and angular order 24. Its weak residual is `1.98e-15`, while
its independent energy and derivative-energy residuals are `1.25e-6` and
`1.97e-7`. The [positive-coordinate note](nvf-positive-coordinate-precision.md)
isolates the amplified coordinate, moment and coefficient rounding and
records the complete correction. The actual public sextic control now passes
the unchanged physical and reflection thresholds. Its independent full-matrix
gap difference has error `5.72e-11` against the existing `3e-9` bound.
This finite precision correction is separate from unresolved optical
approximation changes.


## Complete geometric contact coordinates

For degrees 5 through 8, [complete retained elimination](nvf-retained-contact-solve.md)
uses the smaller block when its dimension is below the common potential count.
The complete common gradient space can also use its integrated
geometric metric instead of repeatedly factoring tall sampled field matrices.
Let `G = integral(grad(psi).T*grad(psi))` and let `Q` contain every retained
Fourier moment. The closed geometric metric is `G-Q.H*Q`. Its compensated
Cholesky factor defines a full invertible coordinate change before the same
material solve. Every polynomial and transmission direction remains present.
A nonpositive pivot is an error; there is no shifted metric, rank cutoff or
replacement field space. The physical JVP differentiates the original
material and geometry, not this computational coordinate change.

The full quintic, two-branch order-4 metal control has 619 common coordinates.
A prototype comparison with the independently sampled QR coordinates passes
all original physical and full reflection checks. The maximum changes over
the entire value and derivative are `1.05e-23` and `1.45e-22`. Energy defects
are `1.39e-17` and `1.18e-12`; full reflection defects are below `2.08e-12` and
`1.96e-10`. Reported optical transmission, reflection and directional slopes
are identical to the QR control at stored double precision.

The QR and integrated-metric runs took 1,095 and 1,050 seconds on the documented
single-BLAS-thread runtime under different concurrent workloads. These are
qualification timings, not a controlled speed comparison. The coordinate
change removes the repeated global sampled QR step; it does not reduce the
cost of independent physical energy integration or prove optical convergence.
The positive-gap metric and independent Decimal pivot controls are described
in [the coordinate precision note](nvf-positive-coordinate-precision.md).

## Nested cutoff spaces and complete cancellation

The optional `cutoff_order` extends each fixed transmission potential `F`
from `h*F` to the nested collection `h*(1-h)**m*F`, for every integer `m`
from zero through the requested order. Here `h` is the same piecewise affine
corner hat. No original mode, real component or common polynomial is
removed. Orders one and two are available at degrees five through eight; order
zero remains the default. With the documented two-branch metal reference,
the three choices contain 20, 40 and 60 real transmission directions.

The cutoff changes both the spatial and moving gradients. Writing
`w(h)=h*(1-h)**m`, the reference gradient is
`w*grad(F)+w'*F*grad(h)`. Its directional variation retains the full product
rule, including `w''*dh*F*grad(h)`. The same fixed potential bank and its
transport apply at contact and at positive gaps. Material differentiation
still leaves reference exponents and coefficients fixed.

A complete two-word evaluation alone was insufficient for this enlarged
space. Coordinate combinations of order `1e7` cancelled to fields of order
one; the independent potential test found normalized gradient errors above
its original `3e-28` criterion. Recomputing all high/low matrix cross products
did not resolve the failure. The nested provider therefore retains a third
arithmetic word through coefficient composition, Bernstein differentiation,
complex powers, cutoff multiplication and the final common combination.
Only then does it return the complete two-word physical field. Contact and
positive-gap weak assembly and physical verification keep their original
algorithms and tolerances.

Three-word products retain every cross term, and normalization carries a
surviving low remainder into the leading word after cancellation. Complex
powers use real range reduction and complete-coefficient exponential and
trigonometric polynomials. No third-party arithmetic implementation or new
runtime dependency is introduced. At degrees five and six the ordinary
cutoff-zero path continues to use its existing two-word provider. Degrees
seven and eight use the three-word provider even with an empty transmission
bank: their Bernstein columns exceed the older small-product kernel's
qualified dimension. Their disappearing positive-gap polynomial fields are
also differentiated completely before applying the physical inverse and its
variation. No existing small-product dimension guard is widened.

Independent 100-digit checks cover both nested orders at degrees five and
six, a refined degree-five mesh, ordinary points and points carrying a low
coordinate word. They differentiate the complete potential independently
in space and along moving coordinates, retaining the same `3e-28` bound.
Separate Decimal checks cover the arithmetic, cancellation, complete matrix
products and complex exponential range. These primitive controls do not
establish optical convergence of the enlarged trial spaces.

The same independent moving-potential oracle now covers all three cutoff
orders at degrees seven and eight, on both the original and once-refined
meshes, with complete low-word coordinates and the unchanged `3e-28` bound.
Independent normal-gradient and stored-trace checks extend through degree
eight as well. A separate 160-digit positive-potential oracle differentiates
the complete polynomial/common/normal combination at degrees six through
eight and physical gaps `.01`, `1e-9` and `1e-13`, at both ordinary and
complete low-word coordinates, again at `3e-28`.
Uniform-material contact controls at degrees five through eight preserve
the exact tensor and its joint material/opening variation under the original
physical and reflection bounds. These checks qualify finite field formulas;
they do not establish optical convergence at higher degree.

Complete metal contact controls use degree seven at Fourier order `(0,1)`
and degree eight at `(1,0)`, radial orders 140/160, angular order 48 and
joint gap/material/background directions. Both retain all 20 real
transmission modes and pass every original physical and reflection guard.
Their energy/derivative-energy residuals are `1.39e-25` / `2.44e-25` and
`3.77e-28` / `8.18e-27`; independently integrated mean-variation residuals
are `4.28e-30` and `1.19e-30`. The runs took 570 and 809 seconds with one
numerical-library thread under concurrent workloads. These are complete
finite contact checks, not an optical-convergence measurement.

A degree-seven zero-Fourier-order positive-gap control at gap `.02` uses
physical materials `(4+.1i, 9+.2i, 1+.02i)` and the distinct fixed reference
`(12.25, -10+i, 1)`. Its joint gap/material/background direction passes every
original physical and full reflection guard. Energy and derivative-energy
residuals are `1.68e-32` / `1.15e-31`; the independently integrated Fourier
mean defect is `1.11e-16`. Full reflection defects are `1.32e-20` / `2.01e-18`.
The run took 2,754 seconds under concurrent workloads, with a scoped
3,600-second test deadline and unchanged physical tolerances. This is one
complete positive state, not a new finite-difference or contact-limit check.

Large three-word field products now accumulate bounded row tiles. Every
entry keeps its original multiplication and addition order, including the
same global exact-zero decisions; all three words remain intact. Independent
100-digit dot checks and complete row-partition comparisons cover large,
empty and low-word-only products. A paired three-repeat degree-seven field
comparison on three triangles, with 512 points and 324 real output columns,
preserved every output bit and changed median evaluation time from 1.864 to
1.484 seconds.

A separate comparison reconstructs the actual complete cutoff-two quintic
order-`(4,4)` electric fields on one bulk and one collapsing triangle.
All four complete `162 x 162` energy, energy-variation, Fourier-mean and
mean-variation integrals remained bitwise identical. Those selected integrals
took 178 and 129 seconds before and after row tiling. Both measurements ran
under concurrent workloads; neither is a whole-solver speedup or a new
optical-convergence qualification.

Repeated samples of a fixed triangle reuse its two complete polynomial-gradient
coefficient matrices. Only the most recently sampled triangle remains in this
additional cache; revisiting another triangle recomputes the same contractions.
The arithmetic and coefficient bank are unchanged. At degree eight with 324
real output columns, the cached matrices occupy 559,872 bytes, independently
of the number of triangles sampled.

A 100-digit moving-potential control first checks three complete columns at
four triangles. Their exact binary-scaled copies then check 258 columns and
514 repeated points, including a return to the first triangle. Both words of
every gradient and variation match. Together with the existing moving and
nested-cutoff controls, all 28 targeted tests pass at the original `3e-28`
criterion. A separate paired comparison uses the stored degree-eight contact
candidate's complete coefficients, with 512 points and 324 columns. Every
output bit is preserved. Three-repeat median field times change from
`1.088 -> 1.079`, `2.106 -> 2.049`, and `1.083 -> 1.024` seconds under concurrent
workloads. These observations do not establish a whole-solver speedup or
qualify the candidate's optical convergence. A fresh complete degree-eight
order-`(1,0)` contact control with joint gap/material/background variation
also passes every original physical and full reflection guard in 686 seconds.
This finite control is not a paired whole-solver timing comparison.

To measure an enlarged space with the full original physical and reflection
checks, use the existing study entry point, for example:

```shell
python -m studies.nvf_pair_angular_resolution --order 4 --degree 5 --refinement 0 --quadrature 100 --angular-quadrature 80 --transmission-branches 2 --transmission-cutoff-order 1
```

A nested cutoff choice is an explicit trial-space choice, not an automatic
accuracy claim. Mesh, Fourier, polynomial, branch and cutoff changes still
require comparison of complete responses and optical observables.

The initial integrated degree-five order-`(1,0)` contact control, with
cutoff order one and a joint gap/material/background direction, passed
all original physical guards and the full reflection condition. It retained
all 40 real transmission directions and took 271 seconds. This is a finite
contact control, not an optical-convergence measurement. A degree-five
zero-Fourier-order positive-gap control at gap `.02`, with joint gap,
material and background directions, also passed every original physical
energy, retained-mean and reflection check. It took 1,699 seconds before
shared-mode sampling. This one integration test has a scoped 3,600-second
CI deadline; no physical tolerance changes. Its moving fields are checked
independently above, but this run is not a new finite-difference optical
or contact-limit measurement.

All cutoffs of one fixed mode share its Cartesian power. Sampling that
identical power once preserves every separate cutoff and its derivatives.
A paired three-repeat comparison at four triangles used 512 complete
quadrature points per triangle and cutoff order two. Median field times
changed from `0.581 -> 0.417` seconds at degree five, `0.617 -> 0.438` at
degree six, and `0.565 -> 0.392` on the refined degree-five mesh. Every
coordinate matrix and both words of every resulting field were identical.
These are field-evaluation timings, not whole-solver speedups.

The public cutoff-one order-`(4,4)` quintic contact study subsequently passed
all original physical and full reflection checks with radial/angular orders
100/80. Its maximum derivative reflection defect was `7.42e-10`, within the
unchanged `3e-9` bound. Relative energy, energy-variation and retained-mean
defects were `6.85e-18`, `1.98e-16` and `4.62e-18`. This run took 2,525 seconds
before shared-mode sampling and the normal-mean correction described in
[the contact-coordinate note](nvf-contact-energy-coordinates.md).

For the documented 900-wavelength, 300-thick contact metal layer, transmission
was `(0.489720305402, 0.478451080712)` for `(p,s)`, and optical directional
slopes were `(-0.045886578353, 0.019566672950)`. These slopes vary the Maxwell
calculation along the constitutive JVP; they are not independent finite-gap
derivatives. Their two stencil scales differed by `1.34e-9` and `5.79e-9`.
Changes relative to the unextended trial space remain appreciable, so this
qualified finite calculation does not establish optical convergence.

A fresh public cutoff-two calculation at the same degree, mesh, Fourier and
quadrature settings retains all 60 real transmission directions. It uses
the complete normal Fourier means and directly reconstructed mean variation
described in [the contact energy note](nvf-contact-energy-coordinates.md).
It passed every original physical and reflection guard in 2,868 seconds.
Energy and derivative-energy residuals were `5.76e-18` and `5.17e-18`;
the independently integrated mean-variation residual was `1.06e-14`.
Maximum full-slot reflection defects were `7.24e-16` and `2.28e-17`.
The resulting p/s transmissions were `.489660564455` / `.478724994095`,
with directional optical slopes `-.047058406141` / `.017519779557`.
The two slope stencils differed by `9.94e-9` / `1.41e-9`.

Relative to cutoff one, transmission still changes by approximately
`-5.97e-5` / `+2.74e-4` and slope by `-1.17e-3` / `-2.05e-3`.
The larger finite space is therefore qualified, but the cutoff sequence
has not met an optical convergence criterion. No field or tolerance was
removed to obtain this result.
