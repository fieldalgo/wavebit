# NVF contact follow-up — 2026-09-07

The JAX shape `solve_field` string-pytree defect is fixed; see
`normal-field-known-limits.md` §8d and `tests/test_parametric_field_jit.py`.
The contact blend and resonant truncation questions are not closed by that fix.

Latest research implementations: [mirror-sector convergence](nvf-mirror-convergence.md)
extends the unchanged finite NVF closure through directional order 40, and
[scattering doubling](nvf-scattering-doubling.md) removes the recorded modal
material-gradient bias in an opt-in research path. The latter retains both
independent contact derivative checks in the polynomial-body model. Neither
changes the production contact blend or establishes optical convergence.

## Usable Cartesian Li route

The validated Cartesian reference is now exposed through the production solver as
explicit `factorization='li'` for axis-aligned rectangular paintings. The builder
integrates a nonuniform partition at all rectangle edges, retaining zero-width
contact intervals; it does not use the experimental normal-field weighting.
Default shape factorization remains NVF. Supported geometry, frozen-plan scope,
and a runnable JIT gradient example are documented in
[parametric-modeling.md](parametric-modeling.md#cartesian-rectangles-with-explicit-li-factorization).
This supplies a working route for the rectangular contact cases investigated here;
it does not close the general NVF contact or resonant-convergence questions below.

## Separating quadrature from the contact derivative

CPU NumPy, float64. Square cell 700, rectangle 350 by `700-gap`, centered,
epsilon 12.25 on background 1, substrate epsilon 2.25, wavelength 900,
thickness 300, theta 0.2, phi 0, psi 0. Values below are central secants in
`ly`, with `h=gap/10`, at truncation `(3,3)`. Clear
`parametric._PROJECTOR_CACHE` before each solve.

| normal grid | gap 1e-2 | gap 1e-4 | gap 1e-6 | gap 1e-8 |
| --- | ---: | ---: | ---: | ---: |
| 64 | 0.0391197 | 3.21181 | -89.8799 | -21.7328 |
| 256 | 0.0318190 | 1.76922 | 97.6119 | 3142.38 |
| 1024 | 0.0316615 | 1.72342 | 72.9945 | 3120.41 |

The first gap is nearly quadrature-stable at grids 256/1024; the smaller gaps
are not. A large derivative at a single grid cannot be identified with the
continuous-field asymptotic exponent. A sign change under grid refinement is
also not evidence that the physical shape derivative changes sign.

## Diagnostic attenuation, not a production change

An isolated square-cell, axis-aligned-box experiment multiplied both opposed
face weights of axis `a` by

`clip((period_a - length_a) / length_a, 0, 1)**4`.

This is only a diagnostic for a self-facing, full-length periodic gap. It does
not specify a treatment for partial runs, multiple neighbors, unequal materials,
rotated cells, or small weights in float32. It is not installed in the solver.
The original distance blend remains the production behavior.

At grid 256, gap 1e-4, and the same central secants:

| Fourier truncation | original derivative | attenuated derivative |
| --- | ---: | ---: |
| (3,3) | 1.76922 | -0.0118299 |
| (5,5) | 0.731173 | -0.0135926 |
| (7,7) | 0.339046 | -0.0140121 |
| (9,9) | 0.292744 | -0.0141559 |

For the candidate at `(3,3)`, the derivative tends to about -0.0118276 at
`gap=1e-6` on all three sampled grids. Thus suppressing the large derivative
is possible in this fixture, but does not establish the physical derivative:
the candidate and original even have different signs. The independent Li
sequence below subsequently rejects the attenuated candidate for this fixture. Neither a smoother
curve nor a small transmission difference closes that question.

A production change needs an explicit near-interface construction for partial
and cross-material contacts, exact-interface checks including exposed ends,
traced replay and material semantics, and independent solve/derivative
convergence. Finite-order plateaus alone cannot prove non-convergence, nor can
these square-cell measurements settle the separate tilted resonant examples.


## Independent thin-gap reference

`tests/contact_instruments.py::rectangular_li` integrates the three-by-three
nonuniform material partition using Cartesian Li factorization. There is no
normal-field grid or thin-gap rasterization. The instrument is intentionally
limited to one centered, axis-aligned rectangle in a rectangular cell.
`tests/test_contact_reference.py` validates its matrices against independently
sampled, commensurate rasters at three heights, including a spanning bar, and
checks a subpixel-gap JIT derivative against two finite-difference steps.

For the same geometry at gap 1e-4, h=1e-5:

| truncation | Li derivative |
| --- | ---: |
| (3,3) | -0.00147645 |
| (5,5) | -0.00181769 |
| (7,7) | -0.00189156 |
| (9,9) | -0.00191639 |
| (11,11) | -0.00192694 |
| (13,13) | -0.00193213 |

This is a converging reference sequence for the stated fixture, not a universal
error bound or a reference for tilted boxes. It rejects the hypothesis that the
attenuated candidate's small, grid-stable derivative alone establishes its
accuracy. The derivative has the same sign as Li but remains about seven times
larger. A further diagnostic integrated that candidate on a nonuniform Gauss
mesh: the y intervals resolve gap/2 and geometrically increasing distances from
the gap, while x is split at the rectangle edges and center. At order 5, gap
1e-4, 8/16/32 points per interval gave -0.0135813 / -0.0135817 / -0.0135818.
Resolving its thin field region therefore does not remove the discrepancy.
The candidate remains excluded from production.

Correct normals on the material boundary are only one condition on a useful
finite-order extension. The primary NVF paper also examines how well the
truncated projector represents the boundary values; it does not supply a
near-contact shape-derivative guarantee. See [Normal vector method for the RCWA, 2008](https://doi.org/10.1364/OE.16.017295).

## Moving run endpoints — fixed

A separate, reproducible discontinuity was found in cross-material transfer.
At the point (340,490), for two 300-wide touching boxes centered at x offsets
-150 and +150, left height 300 and right height 280, changing the latter height
by +/-1e-4 or +/-1e-6 changed the old projector by approximately 0.5 in both
cases. The point is off the material seam; no geometry discontinuity justifies
that value jump there.

The transfer tail is now `n2/(n2+e**2)` off the run, one on it. Here n2 is
squared normal distance and e is tangential excess beyond the interval. It is
C1 in endpoint position away from the seam line and has the exact hard-run
values on that line. If a partner has tails t_i from several runs, its transfer
to run i is `t_i/(1+sum(j!=i, t_j))`. Each denominator is at least `sum(t_j)`,
so the total transferred fraction is at most one; the formula is symmetric
under record permutations. No smoothing length or material value is frozen.
Same-material transfer cancels in the vector sum as before.

The old transmission-step fixture now changes in proportion to endpoint
motion, at all three tested grids. Exact-interface, exposed-end, multi-material,
rotation, backend, and frozen-gradient tests pass. This is a fix for the
run-end jump, not a claim that all contacts are differentiable: nearest-image
switches and corners remain piecewise smooth, and the closing-gap cusp and
contact/material-equality steps still need separate work.


### Effect on the separate contact-value step

The endpoint fix changes the off-interface extension at contact; it does not
introduce a near-pair record across a positive gap. On two 300x300 boxes of
permittivities 4 and 9 separated along x, at N=512 and the same operating point,
the transmission difference `T(gap=1e-6)-T(contact)` is:

| truncation | before endpoint fix | after endpoint fix | nonuniform Li |
| --- | ---: | ---: | ---: |
| (4,4) | -2.63016e-3 | -2.79999e-3 | 1.36345e-9 |
| (8,8) | -1.76829e-4 | -1.96796e-4 | 1.54818e-9 |
| (12,12) | -1.56860e-4 | -1.75957e-4 | 1.52318e-9 |

The contact step is slightly larger on this fixture, while the endpoint jump
is removed. This tradeoff is recorded rather than counted as a fix to the
contact step. Li here integrates the exact nonuniform partition of the two
boxes, using the same Maxwell solve. At contact its transmission is
0.921911793 / 0.920259468 / 0.919793951 at those three truncations; neither its
low-order value nor agreement with it at a single order is an exact solution.


## Separating material and normal-projector sensitivity

`tests/contact_derivative_probe.py` reproduces the decomposition with the same
fixture and two central-secant steps. Run from the checkout with
`PYTHONPATH=$PWD python tests/contact_derivative_probe.py --orders 3 5`.
Each row reports full NVF, material-only, normal-only, and independent Li
sensitivities. The material-only builder freezes the normal coefficients at
the center height; the normal-only builder freezes epsilon and inverse-epsilon
coefficients there. Neither partial derivative is a replacement solver gradient.
Both partial models have exactly the full model's value at the center.

Grid 256, secant step `h=gap/100`:

| gap | order | full | material only | normal only | Li |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1e-2 | 3 | 0.0317128 | -0.00777758 | 0.0394905 | -0.00147945 |
| 1e-2 | 5 | 0.0103606 | -0.00569448 | 0.0160551 | -0.00181945 |
| 1e-4 | 3 | 1.76483 | -0.0102774 | 1.77511 | -0.00147644 |
| 1e-4 | 5 | 0.729281 | -0.0100748 | 0.739356 | -0.00181769 |
| 1e-6 | 3 | 97.3321 | -0.0112056 | 97.3433 | -0.00147587 |
| 1e-6 | 5 | 50.4032 | -0.0121434 | 50.4153 | -0.00181817 |

At the smallest gap, the Li secant at h=1e-8 shows cancellation error; the
wider h=1e-7 secants are -0.00147644 and -0.00181763. Do not infer a new
reference trend from the last digits of those narrower secants.

The changing projector supplies the large positive contribution. Removing its
derivative would leave a negative slope that is still several times the Li
reference; stopping that derivative therefore does not fix the forward model
or its gradient. This narrows the cause to the constitutive factorization and
its near-contact projector extension, rather than an AD-only defect.

`test_partial_derivatives_reconstruct_the_live_nvf_gradient` verifies equal
center values, JIT AD against both secants, and additivity of the AD partials
at gap 1e-4, order 3. Finite secants only add approximately because the forward
solve is nonlinear; the infinitesimal partials add by the chain rule. No
production gradient was frozen and no cusp assertion was relaxed. The old
test and case J wording have been corrected to describe the measured finite
grid defect rather than assert a physical divergence or a unique remedy.

## Material-contrast contact candidate

**Correction, 2026-09-13.** The production seam now uses the complex phase of the material
difference and a Hermitian projector. Ranking by material magnitude introduced a new jump
at distinct equal-modulus materials. The seam study's opening helper also moved the boxes
inward; its historical ladder and matrix contact columns measured overlaps. Corrected
positive-gap runs, modulus-crossing tests and current limitations are recorded in
[the seam contrast](normal-field-known-limits.md#8e-the-seam-contrast-2026-09-11).

> **Superseded 2026-09-11; instrument retired.** The seam contrast now in production
> (`_core/parametric.py`, `_piece_field`, step 3b; `normal-field-known-limits.md` §8e) passes
> the gates this candidate failed, by the mechanism this section names: the candidate's seam
> band was a quarter power of the contrast wide, because its residual seam vector was linear in
> `delta` against inverse-fourth-power distance weights, so the material-equality limit stepped
> on the grids that sampled the shrinking band. Production raises the contrast to the blend's
> own fourth power, `c = (|eps_a - eps_b| / (|eps_a| + |eps_b|))**4`, which makes the band
> linear in the contrast: `T(9 + 1e-7 | 9) - T(9 | 9)` reads +2.1e-9 to +2.4e-9 across orders
> 4 to 12 against Li's +2.2e-9 to +2.3e-9, the ramp from `delta = 1e-7` to 1e-2 tracks Li
> within 4% (12% at 1, the two rules' ordinary finite-order difference at full contrast), and
> the contact step of the same fixture is at rounding (2e-12 against -2.7e-3). Its four files
> -- the instruments contact_material_probe.py and contact_quadrature_probe.py and their tests
> test_contact_material_probe.py and test_contact_quadrature_probe.py, all under tests/ -- were
> deleted the same day; `studies/nvf_seam_contrast.py` reproduces the gates on production, and
> the record below is kept as it was written, as the reason the quarter-power width was the
> thing to fix.

The cross-material contact step had a promising experimental alternative in
contact_material_probe.py (under tests/, deleted 2026-09-11). This instrument changed only an
in-memory copy of `_piece_field`; the production strategy was unchanged. It weights each
piece by its complex material contrast to the background and transfers cross
weights with subtraction, avoiding the equal/different sign flip. If the
resulting complex vector is `(sx, sy)`, a real unit normal is the principal
axis of `Re(s s*)`, computed from
`atan2(2 Re(sx conj(sy)), |sx|²-|sy|²)/2`. This uses the imaginary contrast too:
materials with equal real parts still have a surviving seam. Background-equal
pieces are excluded before the nearest-distance weights are calculated.

Reproduced from the checkout, while the instrument existed, by running the probe from the
repo root with `PYTHONPATH=$PWD`: `--orders 4 8 12` for the ladder and `--orders 4 8 --lossy`
for the lossy row.

Two 300-square boxes at centers `x=±(150+gap/2)`, all other operating parameters
as above; grid 256. The tables give `T(gap=1e-6)-T(contact)`.

| materials | order | production NVF | contrast candidate | nonuniform Li |
| --- | ---: | ---: | ---: | ---: |
| 4, 9 | 4 | -2.79658e-3 | 1.27292e-9 | 1.36343e-9 |
| 4, 9 | 8 | -2.13940e-4 | 1.45265e-9 | 1.54812e-9 |
| 4, 9 | 12 | -1.80236e-4 | 1.47086e-9 | 1.52320e-9 |
| 4+0.2i, 4+0.7i | 4 | -1.00986e-3 | 4.05698e-9 | 4.10844e-9 |
| 4+0.2i, 4+0.7i | 8 | -6.56505e-4 | 4.16520e-9 | 4.23767e-9 |

At gap 1e-4 the candidate and Li offsets are about 100 times larger, consistent
with a finite local slope instead of the production value step. For the real
pair, the candidate/Li one-sided slopes estimated with gap 1e-4 differ by
6.64%, 6.18%, 3.47% at orders 4, 8, 12. This is a useful improvement over the
closing-gap attenuation experiment; it is still finite-order evidence.

Absolute contact transmission also needs convergence: for the real pair the
candidate minus Li is +4.59126e-4 / +2.80806e-4 / -1.47198e-4. At order 12,
production minus Li is -1.09482e-4, so the candidate is not uniformly closer
in value. For the lossy pair, candidate minus Li is -1.11684e-3 / -1.69983e-4
at orders 4 / 8. Removing the contact step does not by itself prove accuracy
of the full transmission or all derivatives.

`paired_rectangles_li` retains a zero-width middle interval at contact, keeping
its partition fixed. Its matrices match independently sampled commensurate
rasters for touching/separated, real/lossy pairs. Candidate tests also check
partial-seam/exposed-edge directions, shape-order and rigid-rotation invariance,
background-invisible geometry, and JIT geometry/material derivatives against
two finite-difference steps. These tests cover this candidate's measured
properties, not acceptance of a new default strategy.

Remaining integration requirements are concrete: thread the live background
through the normal-field pipeline; key cached coefficients on full complex
contrasts (the current production cache correctly needs only equality bits);
validate eigen-direction degeneracies, material-equality limits, multiple
neighbors, magnetic passes and all backends; and demonstrate solve/derivative
convergence beyond these fixtures. The probe clears the projector cache before
every solve and must not be installed directly into a cached application.
For a lone real-contrast box the weighting only rescales its vector, so it
cannot fix that box's closing-gap cusp. Resonant tilted convergence is also
still open.

### Acceptance decision after the extended contact gates

The material-contrast candidate is **not accepted for default integration**.
The additional three-neighbor seam and magnetic-slot/nonunit-background checks
pass, but the material-equality limit fails a necessary continuity gate.
For touching 300-square boxes with epsilon `4 | 4+delta`, order 3:

| grid | candidate T(1e-6)-T(0) | candidate T(1e-8)-T(0) |
| --- | ---: | ---: |
| 127 | 1.52362e-5 | 1.54264e-5 |
| 128 | -5.84362e-6 | -1.84858e-8 |
| 256 | -2.30397e-6 | -8.43944e-7 |

The nonuniform Li reference gives -6.46931e-10 and -6.47715e-12, respectively,
independent of normal grid. The odd grid samples the disappearing seam; merely
selecting an even grid does not establish a reliable material derivative, as
the grid-256 values show. Two strict expected-failure tests retain the required
continuity gate at grids 127 and 256. An unexpected pass will require reviewing
the gate and candidate; these are failures of the candidate's acceptance
criteria, not tests counted as successful validation of it.

Consequently the planned default integration/cache changes are withheld. The
single-box closing-gap issue also remains: multiplying every face by the same
nonzero real contrast preserves its projector, so this candidate cannot solve
that case. The baseline regression assertions have not been relaxed and no
production gradient has been stopped.


## Tilted resonant convergence: separating two numerical limits

The production piece field and the actual-interface polygon-distance field
were compared on the tilted fixture: square cell 700, angle `atan2(1,2)`,
`lx=640`, `ly=700*sin(angle)` at full precision, epsilon 12.25 on 1,
substrate 2.25, wavelength 900, thickness 300, theta 0.2, phi/psi zero.
The polygon field is another extension, not an exact Maxwell reference.
Both use exact rectangle material transforms and the same Maxwell solve;
at order 3 their inverse-z material matrices agree to 1.12e-15.

Reproduce with `PYTHONPATH=$PWD python tests/resonant_nvf_probe.py --orders 6 10 14 18`.

| order | pieces T, N=512 | polygon T, N=512 | difference | pieces grid change | polygon grid change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6 | 0.845759618 | 0.832775470 | 0.0129841 | 5.01406e-05 | 0.000158357 |
| 10 | 0.854479474 | 0.845657748 | 0.00882173 | 2.54383e-05 | 0.000247878 |
| 14 | 0.858743877 | 0.853279722 | 0.00546415 | 2.25003e-05 | 0.000261219 |
| 18 | 0.860788818 | 0.857897392 | 0.00289143 | 7.07705e-05 | 0.000261075 |

`grid change` means the absolute change from N=256 to N=512 at fixed
Fourier order. Across these measurements the difference between extensions
decreases with order and remains larger than either grid change. Increasing
normal-grid resolution alone therefore does not explain or remove it in this
range. Conversely, the order-18 difference of about 0.00289 is still too large
for a claim that the two methods have reached a common accurate value.
Conserved energy (maximum absolute R+T-1 below 9.4e-13) does not certify
accuracy of transmission. This extends the measured order range and rejects
a flat-plateau description of this sequence; it does not prove a convergence
rate, an asymptotic value, or a failure to converge.

## Half-filled raster translation defect — fixed

The broad regression run exposed an existing expected failure in the contour
method: at exactly 50% fill, rolling an asymmetric two-material raster could
complement the corner-based indicator. Contourpy then traversed the same
geometric boundary in the opposite direction, changing the offset resampling
points and the normal extension. Sorting material values would break the
geometry-only/material-relabeling contract and was not used.

The contour consumer now orients half-fill curves geometrically before
resampling. Contractible loops use signed area; percolating curves use the
first nonzero winding component, reversing that winding with their vertices.
This preserves the periodic adjacency and makes indicator complementation
irrelevant without changing how other fill fractions select their phase.

On the original asymmetric 64-square raster, 16 translations gave a transmission
spread of 2.0664255e-4 with the old traversal and 9.14824e-14 after the fix.
The original 1e-9 acceptance threshold is unchanged and its expected-failure
marker has been removed. New projector tests cover compact and percolating
half-fill geometries, square and sheared cells, translation, and complex-material
relabeling. The complete lattice-translation and normal-method suites plus the
new tests pass: **139 passed**.

## Verification and ordered-work status

Before the half-fill fix, the broad affected regression set completed with
**456 passed, 3 expected failures** in 513.52 seconds. Two expected failures
are the experimental contact candidate's equality-limit gates; the third was
the raster defect fixed above. After that production change, its affected
normal-method/translation suites were rerun as the 139-test set above. These
are overlapping runs, not additive test counts. The full repository suite was
not run.

Three wiring/material-transform tests had unnecessarily built million-pixel
normal fields. The inverse-z material-matrix comparison now uses the required
sampling floor because that matrix never reads the normal projector. Circle/
ellipse spelling equivalence and polygonization sanity use N=128. Their
assertions and tolerances are unchanged; dedicated quadrature, derivative and
convergence tests retain their own grids.

| planned step | current disposition |
| --- | --- |
| 1. Validate the contact candidate | Extended checks completed; equality-limit gate failed. Candidate not accepted. |
| 2. Integrate an accepted candidate | Withheld because step 1 did not pass; default material/geometry gradients remain unchanged. |
| 3. Solve the same-material closing-gap sensitivity | Open. This candidate cannot change the lone-box projector; the earlier attenuation candidate also failed Li comparison. |
| 4. Verify tilted resonant convergence | Extended to order 18 and two normal grids. Differences decrease, but a common accurate limit is not yet established. |
| 5. Regression and handoff | Broad affected tests completed, the discovered half-fill defect fixed and verified, and evidence recorded here. Scientific acceptance items above remain open. |

## The material-limit defect is not only a normal-grid artifact

`material_transition_width` in the retired contact_material_probe.py bisected the
pointwise candidate field, without a normal sampling grid or a Maxwell solve.
On the `4 | 4+delta` seam, sample at internal y=400 and move from x=350 into
the right box. The transition width is the distance where P_xx reaches 1/2.

| delta | measured width | width / delta^(1/4) |
| --- | ---: | ---: |
| 1e-2 | 20.7277587 | 65.54693 |
| 1e-4 | 6.49863755 | 64.98638 |
| 1e-6 | 2.05342211 | 64.93491 |
| 1e-8 | 0.649324508 | 64.93245 |

The residual seam vector is proportional to delta. In this local fixture,
competing exterior-face contributions have the inverse-fourth-distance ratio
weights, explaining the measured quarter-power width. This is a statement
about the candidate field near this seam, not a theorem that Maxwell's
transmission derivative diverges. A test checks the measured width ratios and
uniform length scaling. Grid refinement can better integrate this field but
cannot change its slowly shrinking transition region.

Changing the exponent alone also fails acceptance. The following experiments
use isolated in-memory copies with exponent 1 or 2; production stays at 4.
For grid 127, T(delta)-T(0) at deltas 1e-6 / 1e-8 is:

| exponent | delta 1e-6 | delta 1e-8 |
| --- | ---: | ---: |
| 1 | 1.71399e-5 | 1.71410e-5 |
| 2 | 1.61440e-5 | 1.61451e-5 |
| 4 | 1.52362e-5 | 1.54264e-5 |

On grid 256 exponent 1 instead gives -1.17803e-9 / -1.17730e-11, but its
small-delta one-sided slope is about -0.0011773, versus approximately
-0.0006469 from the nonuniform Li reference. A smoother value sequence again
does not establish a correct derivative. No exponent change was adopted.

Alternative vector formulations warrant comparison, but are not interchangeable
patches. The [FMMAX formulation documentation](https://invrs-io.github.io/fmmax/notebooks/formulations.html)
explicitly distinguishes normalizing a linear vector field (which introduces
discontinuities at its zeros) from complex elliptical Jones fields. Wavebit
already has a raster Jones projector helper; adapting a different formulation
to the shape path would still need its own interface, material, derivative,
cache and convergence validation. It is not installed as a silent replacement
for the current real-normal method.

The extended contact-candidate tests finish with **15 passed and 2 expected
failures**. Those two remain the unpassed equality-limit acceptance gates.

### Cartesian Li delivery checks

`tests/test_rectilinear_li.py`: 31 passed. The factorization dispatch, pytree, and
parametric field regression group passed 64 tests. The coefficients/frozen-geometry
group passed its existing behavior checks; its former assertion that Cartesian
paintings reject Li was replaced with explicit Li support plus sampled-coefficient
refusal, and that replacement passed. The contact example passes the repository's
example smoke runner. `git diff --check` is clean. This is focused validation, not
a complete repository test run. Single-precision JAX/Torch fixtures still emit the
existing unresolved square-root branch warning; their tested transmission agrees
with the double-precision reference within 2e-5. Contact derivative checks use x64.

## NVF contact investigation after the Cartesian alternative

The work below modifies NVF projectors in isolated test instruments. It neither
selects Li for the investigated solves nor changes the production default.

### Resolve the weak seam before accepting a material derivative

contact_quadrature_probe.py (under tests/; retired 2026-09-11 with the candidate it served)
replaced only the three sampled normal
projector integrals with tensor Gauss quadrature, graded geometrically around the
internal seam. The epsilon transforms and Maxwell/NVF assembly remain unchanged.
Plane-wave Fourier moments and a 1e-6-wide strip independently verify quadrature
normalization, phase, axis order, and subpixel measure.

The exponent-1 contrast candidate previously looked stable on even uniform grids.
At order 3 and delta=1e-6 the N=256 slope was -0.00117803. Resolving the seam with
Gauss degree 16 instead gives -0.00222388; degree 8 gives -0.00201428, so even that
new ladder still needs refinement. A uniform grid was missing the narrow region.
Its apparently finite slope was not an acceptance result.

The exponent-1/2 candidate is better behaved in this particular material ladder:

| order | degree 16 slope at delta=1e-6 | degree 32 slope | independent Li slope |
| --- | ---: | ---: | ---: |
| 3 | -0.00127037 | -0.00127241 | -0.000646916 |
| 7 | -0.000853343 | -0.000854699 | -0.000698713 |
| 11 | -0.000761428 | -0.000761532 | -0.000747898 |
| 15 | -0.000749194 | -0.000749109 | -0.000757490 |

These are one-sided transmission secants in the right permittivity at 4|4+delta,
not claims of infinite-order accuracy. In particular, matching a derivative on
one grid or at one order does not establish a correct general NVF field. Nothing
in this material experiment fixes the lone-box closing-gap geometry by itself.

Reproduced, while the instrument existed, by running the quadrature probe from the repo root
with `PYTHONPATH=.` and `--orders 3 7 11 15 --degrees 16 32 --exponent .5`.

### A non-cancelling complex NVF projector

`tests/contact_jones_probe.py` constructs a different NVF candidate. Given the
same nonnegative piece weights, form the normal dyad sum

`M = sum(w * n n^T) = [[a,b],[b,c]]`, then `J = M @ (1,i)` and `P = J J^H / |J|^2`.

This is a derived experimental construction, not a claim that it is identical to
the existing raster Jones implementation. Its denominator is
`a^2 + 2*b^2 + c^2 >= (a+c)^2/2`. At least one distance-ratio weight is one, so
oppositely oriented normals cannot produce a zero denominator. At a regular
interface all dominating normals are parallel: M has rank one along the normal,
and P is exactly the required normal projector. The experiment uses all four
complex projector components with the existing `_nvf_assemble` function.

Pointwise checks cover Hermiticity, unit trace, rank one, exact interface normals,
and rotation covariance. Frozen JAX JIT derivatives agree with two finite-
difference steps for the period gap, the unequal-material contact gap, and the
material-equality case. Grids 127, 128 and 256 all pass the no-value-step gate at
1e-4, 1e-6 and 1e-8. This includes the odd grid that defeated the contrast-vector
candidate at the seam itself.

For the original one-box gap at order 3, N=256, the new one-sided gap secants are
0.002078603 / 0.002078606 at gaps 1e-4 / 1e-6. At order 15, N=512 they are
0.002006872 / 0.002006511. This is the gap derivative; the earlier height
derivatives have the opposite sign. The smooth value ladder does not by itself
establish agreement with the infinite-order physical derivative.

**There is still a failed acceptance gate.** Opposite coincident dyads reinforce,
so an interface removed by material equality remains in the field extension.
One rectangle and the same rectangle split into two equal-material boxes have
identical epsilon and inverse-epsilon transforms, but their NVF projectors differ:

| order, N=128 | T(split) - T(merged) |
| --- | ---: |
| 3 | 1.48647e-5 |
| 7 | 1.02443e-5 |
| 11 | 8.23024e-6 |

The discrepancy decreases over this ladder but fails the newly introduced
finite-order representation-equivalence target. This was not a pre-existing
production guarantee; see the baseline comparison and integration update below. It is isolated in a strict expected-failure test
with a 1e-10 comparison target, not counted as a pass. The candidate
was initially left outside the production solver on that basis. The remaining
representation question is
removing artificial equal-material seams without restoring the contact singularity;
no accepted construction for that combination has been established here.

The field, derivative, and unresolved representation checks were reproduced with the
quadrature probe's test beside the Jones probe's; the former is retired since 2026-09-11 and
the latter stands:

```sh
PYTHONPATH=. python -m pytest tests/test_contact_jones_probe.py -q
PYTHONPATH=. python tests/contact_jones_probe.py --orders 3 7 11 15 --grids 127 128 256 512
```

Focused instrument validation: **12 passed, 1 expected failure**. The expected
failure is the unresolved split/merged representation gate above. Ruff and
`git diff --check` pass; production code was not changed by this investigation.

## Production NVF integration and correction of the representation premise

The previous stopping decision treated a newly added 1e-10 split/merged comparison
as a guarantee the old NVF already met. That premise was incorrect. On the same
N=128 fixture, the existing linear NVF has T(split)-T(merged) of 1.81637e-5,
6.61232e-7 and 8.13142e-8 at orders 3, 7 and 11. The circular method has a different
finite-order error ladder; neither this comparison nor its improvement establishes
infinite-order equivalence. The strict diagnostic remains an open accuracy target,
not a newly caused regression that prevents exposing the tested NVF construction.

`factorization='nvf', nvf_projector='circular'` is now wired into the production shape
solver for rectangles, including rotated/skew-cell geometry, painting, epsilon and
mu, and all three array backends. Material transforms remain exact. Boundary-resolved
material coefficients are now explicitly cast to the requested complex dtype;
otherwise double-precision polygon vertices promoted them and made the new
single-precision Torch projector multiply incompatible matrix types.

The circular field skips signed pair distance substitutions entirely. Its frozen
plan check retains image/reduction/shape requirements but does not reject a change
in contact/dead-face flags that the circular build does not read. This makes it
possible to freeze at contact, open the gap, and obtain the correct one-sided
JIT derivative of the implemented NVF solve. New overlap is still refused eagerly.

The default linear field and its known contact limitations are unchanged. Circular
projection is an explicit NVF option, not Li substitution and not a claim that the
curved-primitive contact or tilted resonant-convergence investigations are closed.
[The public usage section](parametric-modeling.md#complex-nvf-projection-for-rectangular-contacts)
and [runnable example](../examples/nvf_contact.py) describe the supported path.

Validation on 2026-09-08 (overlapping suites, not an additive total):

- Final production circular-projector suite: **24 passed**. Covers contact limits,
  contact-frozen one-sided JIT derivatives, independent coefficient comparison,
  tiled sampling, painting, rotated/skew cells, all three backends, single
  precision, Torch geometry/permeability gradients, uniform layers, and stale plans.
- Projector/factorization/Li regression run: **110 passed**, with two existing
  single-precision branch warnings in the Li fixture.
- Manifest/frozen geometry/JIT fields/polygon interfaces: **112 passed**.
- Gamma fields/frozen plans/contact/material probes/boundaries: **152 passed,
  2 expected failures** in the earlier rejected material-contrast construction.
- The public `examples/nvf_contact.py` smoke test passed. Targeted Ruff and
  `git diff --check` passed. These are focused checks, not a full repository run.

The separate strict split/merged representation diagnostic remains an expected
failure; it is not included among the production suite's passing checks.

## Disjoint curved NVF integration, 2026-09-08

Circular NVF now accepts disjoint circles, ellipses and mixtures with rectangles,
including tangent contacts. The square/disk aliases are supported too. The default
linear field's arithmetic is unchanged. Merging equal-material primitives only
at exact contact/equality would restore the value discontinuity, so that shortcut
has not replaced the open representation diagnostic.

For each curved periodic image, write f=(u/a)^2+(v/b)^2-1 and
g=min(a,b)*grad(f). Its positive dyad is tapered by
(1-r^2/(max(a,b)+h/2)^2)_+^4, with h the smaller reduced-cell height, and weighted
by the inverse fourth power of min(a,b)*abs(f). The unscaled extension tensor
also includes I/h^4. Positive common distance ratios evaluate the normalized
projector without evaluating any singular inverse power on the boundary. On a
regular boundary the pole is dominant and g is the exact normal; at the centre
g vanishes but the isotropic term remains. The compact image support is fully
inside the existing plan's ring, so image folding does not switch directions or
leave a truncated nonzero contribution behind. A larger ring gives the same field.

The common ratio includes the isotropic distance scale to bound its coefficient.
The positive matrix is scaled before squaring its norm, which prevents float32
underflow at the centres of very small inclusions. If all entries have underflowed,
the circular projector is the numerical continuation of the isotropic centre.

This is an analytic-curve construction. Curved paintings use polygon chords;
their normals differ, so both solve and freeze refuse circular projection of
overlapping curves. Rectangular painting remains supported. Frozen replay keeps
the full circular image ring and no longer relies on the linear arc's nearest-
image list; changing the required ring still requires refreezing.

Validation (overlapping suites, not an additive total):

- New curved suite: **27 passed**, including exact interface tensors, centre and
  periodic regularity, Hermitian rank-one projection, rotation covariance, ring
  independence, tiled/full equality, odd/even-grid contact limits, frozen-contact
  JIT derivatives, Torch geometry/material gradients, cross-backend single
  precision, JIT fields, painting refusal, aliases and very small circles.
- Related curved/rectangular projector, manifest, JIT field and frozen-geometry
  suite: **174 passed** before the final float32 normalization refinement. The
  final 27-case curved run includes that refinement.
- `examples/nvf_curved_contact.py`: **1/1 smoke passed**. Targeted Ruff and
  `git diff --check` passed. No full repository run is claimed.

The rectangular finite-order split/merged discrepancy, overlapping curved
circular construction and tilted resonant-convergence investigation remain open.
The new contact checks establish continuity and derivatives of this NVF solve;
they do not certify a converged Maxwell answer at every Fourier order.

## Curved painting support and local facet correction, 2026-09-08

`nvf_projector='circular'` now also accepts overlapping circles/ellipses, including
mixtures with rectangles. The material transform still uses the existing
polygon painting. The projector follows that same polygon geometry on every
regular visible interface, rather than applying an analytic ellipse normal to
a chord. Original polygon edges remain extension geometry when covered, which
keeps material equality from switching the field. The default linear field is
unchanged. The manifest now lists both projectors for the general shape path.

The polygon tensor sums positive edge-normal dyads with edge-length weights and
inverse fourth powers of periodic segment distance. Distances are minimized
per edge over a ring bounded by primitive size and reduced-cell height. An edge's
normal does not change between its images. Ratio weights avoid singular inverse
powers, and a positive scaling of the tensor protects its norm calculation.
The chunked JAX scan now also carries four complex projector components.

Replacing the whole analytic extension with that polygon tensor was rejected:
on the tangent-ellipse continuation at order 2, grid 64, the public solve changed
from T=0.9002281562 at contact to T=0.8837791323 at gap=-0.01. This was mainly a
normal-extension change, far larger than polygon material approximation.

The accepted build instead corrects the analytic positive tensor locally. Its
blend weight is the curved-edge fraction of the distance weights times
(ell^2/(ell^2+d_min^2))^2, where ell^2 is the weighted curved chord length squared.
The weight is one on a regular curved facet and zero on a rectangular face;
away from curves it tends to zero under polygon refinement. No selected nearest
edge sets the blend width. The positive real tensor of the analytic projector
is recovered, up to scale, from Re(P)+abs(Im(P_xy))*I before blending. The final
Jones lift preserves a Hermitian rank-one projector. With this correction the
gap=-0.01 solve gives T=0.9002392729.

Polygon approximation has not disappeared. Force the polygon route at exact
tangent contact and compare it with the analytic route at the same Fourier order:

| Polygon vertices | Normal grid | T(polygon) - T(analytic) |
| --- | --- | ---: |
| 64 | 64 | -4.25674e-5 |
| 128 | 64 | -8.94619e-6 |
| 256 | 64 | -2.64937e-7 |
| 512 | 64 | -6.64489e-7 |
| 256 | 128 | -1.48380e-6 |
| 256 | 256 | -9.84354e-7 |

The intermediate total errors are not monotonic: geometry and sampled-projector
errors can cancel. The new regression checks reduction over the polygon ladder
and its final absolute error, rather than assuming monotonic total transmission.
This is a fixed-Fourier-order approximation comparison, not a Maxwell oracle.
Crossing from disjoint analytic geometry into polygon painting still changes
the geometry approximation and requires a fresh clipping plan; exact continuity
across that representation switch is not claimed.

Validation:

- The new painting tests cover raw and corrected facet normals, periodicity,
  image-ring independence, chunking, material equality, public dispatch,
  single-precision lossy magnetic materials on JAX/Torch, JIT geometry/material
  gradients, Torch permeability gradients, rigid translations, stale clipping
  plans, polygon refinement, rotational covariance and JIT field reconstruction.
- The combined new/previous shape-projector and manifest run passed 107 cases.
  One new refinement assertion assumed monotonic total errors and a smaller
  final error than the ladder establishes. It was replaced with the convergence
  check described above; the corrected test passed with the boundary regression.
- Boundary and actual-interface regressions plus that corrected refinement
  test: **91 passed**. A Torch array-conversion advisory appeared in the combined
  run; its live derivatives passed the independent finite-difference checks.
- `examples/nvf_curved_painting.py`: **1/1 smoke passed**. Targeted Ruff and
  `git diff --check` passed. These suites overlap; no additive total or full
  repository run is claimed.

The remaining main investigations are the rectangular finite-order split/merged
representation discrepancy and tilted resonant convergence. Curved painting is
now implemented, with the polygon approximation and frozen-topology limits above.

## Independent tilted-fixture comparison, 2026-09-08

`tests/resonant_nvf_probe.py` now includes the public circular NVF and optional
S4 direct-rule solves. The S4 bridge uses the same rectangle, layer stack,
permittivities and p-polarized incidence as the fixture above. Its rectangle
and incidence angles are converted to degrees. `LatticeTruncation` is explicitly
`Parallelogramic`, and the **actual** `GetBasisSet()` entries must equal Wavebit's
square set `[-order, order]^2`; an equal number of circularly selected harmonics
would not be the same discretization.

A scoped `wavebit-direct` diagnostic keeps Wavebit's exact material coefficients
and replaces only the transverse NVF assembly with the direct rule. This is not
a new solver option or a fallback for rectangular shapes. At orders 6, 10, 14,
its transmission differs from S4 direct by 4.26e-14, 4.29e-14, 1.25e-12.
This checks the patterned structure and the Maxwell solve independently of
NVF: the large difference between S4 direct and circular NVF is not reproduced
when their constitutive factorization is matched. Neither finite-order
factorization is thereby established as an accurate Maxwell reference.

Reproduce the independent comparison with:

```bash
PYTHONPATH=$PWD python tests/resonant_nvf_probe.py --orders 6 10 14 18 --grids 256 512 --methods circular s4-direct
PYTHONPATH=$PWD python tests/resonant_nvf_probe.py --orders 6 10 14 --grids 512 --methods wavebit-direct s4-direct
PYTHONPATH=$PWD python tests/resonant_nvf_probe.py --orders 22 --grids 512 --methods circular pieces s4-direct
```

| Fourier order | circular T, grid 512 | S4 direct T | absolute difference | circular grid change, 256 to 512 |
| --- | ---: | ---: | ---: | ---: |
| 6 | 0.826543208 | 0.941709091 | 0.115165883 | 0.000095262 |
| 10 | 0.838581537 | 0.917495725 | 0.078914189 | 0.000156269 |
| 14 | 0.845478595 | 0.904278462 | 0.058799866 | 0.000190149 |
| 18 | 0.850166086 | 0.896188108 | 0.046022022 | 0.000193909 |
| 22 | 0.853755019 | 0.890772020 | 0.037017001 | not measured |

At order 22 the default linear piece field gives T=0.861988062, compared with
0.860788818 at order 18. Circular NVF changes by 0.003588934 over that step,
and S4 direct by -0.005416088. The order-22 circular/S4 difference is still
0.037017001 despite R+T-1 errors no larger than 4.28e-12 for these three solves.

These two sequences approach from different directions over the measured range;
their common limit and a usable error tolerance are still undetermined. The
small normal-grid changes do not close the much larger Fourier-order gap.
The periodic contact is only `640 - 700*cos(atan2(1,2)) = 13.900966` long.
That is a geometric scale to retain when examining refinement; it does not
by itself establish a necessary Fourier order or an error estimate.

Two local alternatives were rejected rather than integrated:

- Squaring the positive dyad matrix before its Jones lift (powers 2, 4, 8)
  preserves interface normals but does not materially improve the sequence.
  At order 14, grid 512, T is 0.845294732 / 0.845291881 / 0.845379556,
  compared with 0.845478595 for the existing power-one construction.
- Replacing the anticommutator with either `Delta @ C` or `C @ Delta`, keeping
  the same exact material matrices and circular projector, violates energy
  conservation. At order 14, grid 512, the absolute R+T-1 errors are
  0.003478986 and 0.003428438. The corresponding production error is
  2.65e-13. A different transmission value from this experiment is not an
  accepted accuracy improvement.

The comparison bridge's nine regression tests pass. They cover the oblique
uniform slab, rectangle rotation units, order-by-order direct-rule agreement
on the patterned contact fixture before and after a rigid translation, the
public circular option, restoration of NVF after the direct diagnostic, and exact
circular-projector normals on the tilted fixture's regular visible faces in
three periodic images. The normal test excludes the interior of the same-material
contact, which is not a physical interface, and the non-unique corner normals.
S4-dependent tests skip when the optional external solver is absent. The comparison CLI also completed all five methods at order 2, and targeted
Ruff and `git diff --check` passed. The production factorization and existing
numerical reference files are unchanged by this investigation. No full
repository test run is claimed.

## Further closure attempts: no accepted replacement, 2026-09-08

The representation and resonance investigations remain open. This follow-up
attempted changes to both the constitutive assembly and the source of the normal
field. A smaller split/merged difference alone is not an acceptance condition:
the contact derivative and independently checked physical inputs must survive too.

### A material-defined field removes the split error but fails the contact gate

`studies/nvf_material_projection.py` constructs the field from the exact complex
permittivity Fourier coefficients, so a different primitive partition of the same
material function supplies the same input. It Gaussian-filters the spectral
gradient, smooths its real covariance, and adds an isotropic regularization before
normalization. Three lifts were tested: the real covariance divided by its trace,
a Jones lift of its square root, and the existing Jones lift of the covariance.
These are research candidates; the smoothed field does not enforce exact interface
normals and has no production or autodiff contract.

The positive result is real: all twelve width/lift combinations at grid 256
reduce the order-3 split/merged transmission difference below 4e-14. However,
all twelve fail the independently measured thin-gap derivative by more than a
factor of four at order 7. Selected Jones-lift results, with two secant steps:

| spectral width | T(split)-T(merged), order 3 | gap slope, h=1e-4 | gap slope, h=1e-6 | resonant T, order 10 |
| --- | ---: | ---: | ---: | ---: |
| 8 | -3.22e-15 | 0.010099792 | 0.010109660 | 0.856963913 |
| 16 | 3.42e-14 | 0.008347705 | 0.008276916 | 0.862658251 |
| independent Cartesian Li | — | 0.001891554 | 0.001891559 | not applicable |

The reference uses the same full-period bar and its `700-gap` continuation as
before. Its order convergence is recorded earlier in this document. The changes
in the last two columns are not an accepted improvement of the complete model.
The widths were swept to test this construction, not fitted to a desired answer.
Reproduce the selected rows with:

```bash
PYTHONPATH=$PWD python studies/nvf_material_projection.py --widths 8 16 --lifts jones
```

The selected output is retained in `studies/results/nvf_material_projection.log`.
The original strict split/merged diagnostic is still unresolved; it has not been
relaxed or replaced by this candidate's successful representation-only check.

### Changing the constitutive assembly did not close the gap

A separate NumPy experiment constructs the full Jones basis F, takes the unitary
polar factor U of its convolution matrix, and uses
`U diag(inv(C(1/eps)), C(eps)) U.H`. Uniform-material behavior and lossless energy
are preserved by this construction, but the order-3 split difference grows to
-6.61606e-4 and the resonant order-14 transmission is 0.835551686. It was rejected.

Another experiment forms the scalar continuous-variable rearrangement pointwise,
`A = eps I + (1-eps) P`, `B = I + (1/eps-1) P`, integrating material/projector
products by Fourier convolution rather than sampling a subpixel material gap.
Closing it by `Ahat @ inv(Bhat)` fails lossless energy; symmetrizing that product
or using `Bhat**(-1/2) @ Ahat @ Bhat**(-1/2)` repairs energy but does not repair
representation equality. The latter gives a split difference of -4.59332e-5 at
order 3, and a resonant transmission of 0.843020829 at order 10. These experiments
do not establish a better general factorization and are not installed in the solver.

### S4 vector formulations are additional comparisons, not reference truth

The same checked square basis, geometry, materials and incidence were also run
with S4 polarization decomposition enabled. Only `PolarizationBasis` changes:

| S4 basis | T, order 6 | T, order 10 | T, order 14 | abs(R+T-1), order 14 |
| --- | ---: | ---: | ---: | ---: |
| Default | 0.857497621 | 0.850936456 | 0.844015664 | 2.04393e-4 |
| Normal | 0.776075440 | 0.810112826 | 0.816051794 | 2.80774e-4 |
| Jones | 0.737344213 | 0.795216503 | 0.808226490 | 3.73938e-3 |

This does not establish a common converged value. In particular, a closer value
from one of these finite-order sequences cannot certify the Wavebit NVF result.
The bridge now exposes these variants for reproducibility:

```bash
PYTHONPATH=$PWD python tests/resonant_nvf_probe.py --orders 6 10 14 --methods s4-default s4-normal s4-jones
```

Nonfinite fluxes or nonpositive incoming flux now raise instead of becoming
comparison data. The homogeneous-layer control is expressed as a homogeneous
S4 layer: asking this S4 build to construct a Default/Jones field for a full-cell
rectangle instead produces NaNs, although the physical medium is uniform.
Patterned fixtures continue to use their actual rectangle geometry.

An additional rotated-supercell prototype was investigated because this fixture
has slope 1:2. Its material matrix agrees with the exact primitive-cell matrix,
but its nested inverse-rule truncation is still unresolved: at outer order 18,
inner caps 54 and 108 give T=0.855694716 and 0.850057928. Matching the material
matrix is not enough to validate a new reference factorization. This prototype
is not used as an oracle or exposed as a production method.

These failed gates are the reason neither requested numerical property is marked
resolved. There is no accepted replacement from this follow-up, and no claim
that the tested failures prove a general impossibility of another construction.

Validation of the updated comparison instrument: **14 passed**, including all
four S4 settings on the homogeneous control and rejection of invalid external
flux. The three patterned S4 vector modes were also replayed through the CLI
at order 6. Targeted Ruff and `git diff --check` passed. These validate the
instrument, not acceptance of any rejected NVF candidate. No full repository
suite was rerun because this follow-up changes no production solver code.

## Independent spatial discretization and matched-coordinate study, 2026-09-08

The next investigation is a conforming three-dimensional H(curl) FEM reference,
implemented in `studies/nvf_fem_reference.py`. Its equations, geometry, PML,
periodic-pair safeguards, controls, and convergence status are documented in
[`nvf-independent-fem.md`](nvf-independent-fem.md). This is not a production
factorization change or a replacement for rectangular NVF.

Searching the other installed environments found NGSolve in `femtest`, Meep in
`meep`, and gyptis in `gyptis`. An earlier check of only the `fieldwave` Python
environment had missed them. No packages were installed. The gyptis interpreter
needed a sandbox exception for its local MPI initialization; that exception was
automatically allowed. The final NGSolve instrument runs in the ordinary sandbox.

The first FEM prototype exposed a concrete setup error: its outer-face selection
was tighter than OpenCascade's padded bounding boxes. Empty `Identify` requests
silently generated **zero** periodic pairs. The corrected instrument checks the
actual mesh pairs in both directions and its nonzero-source slab against the
analytic transfer-matrix solution. Discard all nonperiodic prototype values.
Tests of the optional instrument pass independently; those are instrument
checks, not newly passing NVF physics cases.

Other probes in this follow-up did not close either NVF investigation:

- A fixture-specific matched-axis coordinate map broadened the short contact
  interval while preserving both primitive translations. Both epsilon and mu
  were transformed, and exterior Rayleigh waves were pulled back into the same
  coordinates. The untransformed direct-rule control agreed with S4 at order 6
  to 7.6e-15. Two mapping strengths looked close at order 18 (T=0.869967037 and
  0.869750117), but separated again at order 22 (0.868311013 and 0.872560941).
  Thus that agreement was not evidence of convergence. Changing the retained
  Fourier set to a square aligned with the rectangle also remained unresolved.
  This exploratory matched-coordinate/Li reference was not substituted for NVF.
- Replacing the isotropic regularizer in the previous material-gradient study
  by its global anisotropic covariance preserves the exact one-dimensional
  projector in the closed-bar limit and gives split errors below 3e-14. It still
  gives a gap slope of approximately 0.014011, versus 0.00189156 for the
  independent order-7 Li instrument. The widths 4/8/16/32 all show that failure.
  Enforcing the one-dimensional forward limit alone does not fix its first
  variation as a horizontal interface appears.
- A signed logarithmic boundary-potential field was additive under collinear
  subdivision (sampled split differences below 3.2e-15). Its sampled order-7
  gap slopes were 0.0123 to 0.0138, still failing the derivative comparison.
  Resolving a singular boundary layer would require a separate quadrature
  analysis; these sampled values are not a validated continuum construction.
- Meep's oblique uniform-slab control approached the analytic result when
  resolution increased from 40 to 80 pixels per micrometer (T errors 0.0029681
  and 0.0007112). The actual tilted structure was not converged: after a further
  40 time units its T was about 0.95243 / 0.90744 at those resolutions, and a
  reflected-flux tail of about 4e-4 remained in the finer run. Its residual-based
  frequency-domain solve failed to converge at the 5000-iteration cap. None of
  these results was adopted as a reference. The separate gyptis 3D prototype
  also failed its uniform-slab accuracy control and was not used.

The open requirements remain finite-order representation equality without a
contact-derivative regression, and independently verified resonant convergence.
The FEM instrument provides a new route to the second requirement; it does not
by itself solve the first.

### Rejecting a false FEM plateau and testing a new constitutive closure

The same-order PML FEM sequence at h=0.18/0.14 initially differed by only
0.000120944 in T. A separate PML-strength check changed T by 0.005493334. The
near-grazing substrate (1,0) order is insufficiently absorbed by the short PML;
its round-trip amplitude attenuation is only 0.0896868 with stretch=2. Those
0.876 values are **not** adopted as an oracle. Neither energy conservation nor
mesh refinement alone revealed the boundary bias.

The reference now also has a Rayleigh/DtN boundary option: exact outgoing
admittance for retained Fourier traces, with local impedance only for omitted
evanescent orders. It checks that every propagating order is retained, applies
the boundary correction by a low-rank Woodbury solve, and reports both modal
and volume flux. Its preliminary h=0.18, p=4 volume T is 0.866055904. Mesh and
boundary truncation checks still apply; see the current tables in
[`nvf-independent-fem.md`](nvf-independent-fem.md).

A new material-defined candidate is preserved in `studies/nvf_harmonic_energy.py`.
Periodic electrostatic correctors form B, and a positive energy Gram matrix is
expressed in retained physical Fourier coordinates as
`inv(C(B)).H @ C(B.H epsilon B) @ inv(C(B))`. Exact material coefficients and
zero-padded products avoid a pixel cutoff. The split check is below 1e-12, and
the order-7 thin-gap slope approaches 0.00189156 as the inner corrector order
increases: 0.00196960 / 0.00193171 / 0.00191197 / 0.00190183 at inner orders
31 / 63 / 127 / 255. However, its closed-bar forward value and resonant result
still change with that inner order. This is a promising tested direction, not
an accepted NVF replacement. The construction, two separate truncations,
measured limitations, and reproduction are in
[`nvf-harmonic-energy.md`](nvf-harmonic-energy.md).

Validation: **18 passed, 12 skipped** in the ordinary Wavebit environment,
and **12 passed** in the optional NGSolve environment. These are instrument
checks, including exact-slab controls and condensation equivalence for both
FEM boundary formulations. The original strict NVF split/merge diagnostic,
production dispatch, and physics tolerances were not altered.

The original strict split/merged diagnostic was replayed separately and remains
**1 xfailed**. This is the production-path limitation, not a passing result of
the new candidate. The h=0.14, p=4 Rayleigh FEM refinement was terminated after
its measured peak memory footprint reached 23.6G and 15.8G was swapped out.
Smaller linear-solver controls did not qualify a better replacement. The current
independent FEM reference therefore retains a mesh-refinement uncertainty;
the resource-limited run produced no T/R value.

A further square-root-of-covariance variant of the earlier material-gradient
projector was tested to obtain an O(gap), rather than O(gap^2), transverse
projector contribution at a closed bar. It retained split equality but failed
the contact comparison: width=4 gave gap slope about 0.01469, width=8 about
0.00354, width=16 about
0.0302, and width=32 about -0.0370, versus 0.00189156. The dependence on smoothing
width and sign failure disqualify this variant; no width was fitted or adopted.

## Harmonic-energy implicit derivatives, 2026-09-08

The finite material-defined candidate now has an opt-in JAX instrument in
`studies/nvf_harmonic_energy_jax.py`. Its cell solve uses implicit differentiation,
with the actual residual checked separately for forward and transpose solves.
Both kinds of failure produce nonfinite data under JIT; a dedicated test verifies
that a converged forward solve cannot hide an unconverged adjoint. The existing
study scattering routine is shared by NumPy and JAX. A cached material FFT avoids
recomputing that transform during every cell iteration.

JIT derivatives agree with independent NumPy secants for width, height, angle,
permittivity, exact contact from the positive-gap side, and zero contrast. Equal
material split/merge values and physical parameter derivatives agree, including
when the artificial internal seam moves. At n=7, the exact-contact derivatives
at H=31/63/127/255 are 0.001969613266 / 0.001931692175 / 0.001911989676 /
0.001901820104. NumPy/JAX forward differences are below 1.10e-13 in this sweep.
The H=255 contact derivative is still about 0.54% above the independent reference;
correct finite-model differentiation does not remove the discretization error.

A reciprocal-material, divergence-free-flux version of the energy closure was
also tried. It gives the exact closed-bar value, but its contact derivative stays
near 0.01401105 at H=15 through 127, versus the independent 0.00189156. It is
rejected. The equations, measurements and current candidate limitations are in
[`nvf-harmonic-energy.md`](nvf-harmonic-energy.md).

Validation: **24 passed, 1 xfailed** for the original and JAX candidate tests,
the resonant instrument tests, and the original strict production split/merge
diagnostic. The xfail is the existing production limitation. Targeted Ruff and
`git diff --check` pass. No production NVF dispatch, physics tolerance, or FEM
solver was changed in this increment. The resource-heavy FEM refinement was
not restarted. Remaining work is inner/outer resonant convergence, an independently
converged reference, exact interface behavior, and production backend integration.

## Cell enclosures and expanded polarization check, 2026-09-08

The next investigation separated three errors instead of interpreting a small
CG residual as convergence. `studies/nvf_cell_bounds.py` now computes static
primal/complementary energy enclosures. `studies/nvf_cell_corrector.py` supplies
an optional conforming 2D scalar FEM corrector and Fourier projection on the
actual material triangles. This solves the same auxiliary problem as the
Fourier candidate; it is not a 3D Maxwell solve or a production backend.

For the tilted fixture, the relative static enclosure width decreases from
0.00318067546 at Fourier H=511 to 0.00000291024 with 1,396 triangles and 44,673
unknowns at FEM degree 8. Fourier quadrature is checked separately: cap-44
coefficients change by 3.37e-13 from quadrature 24 to 28. The projected material
matches its analytic transform to 1.78e-15. The same energy closure then gives
T=0.855485746686 / 0.857473156196 / 0.860355871708 / 0.861820093322 /
0.862617122442 at outer orders 6/10/14/18/22. The outer truncation and independent
3D reference still retain uncertainty. These static bounds must not be reported
as bounds on optical T. The closed-bar finite-H forward bias is removed by the
conforming cell control, which recovers the exact constitutive matrix.

The more important new failure is polarization-dependent. The previously
reported 0.54% discrepancy was for p polarization. Under s polarization, the
closed value still matches the independent Li instrument, but the candidate's
exact-contact derivative is -0.00006355713415 at H=31/63/127/255, while the
independent positive secants approach -0.00077857. The approximately 12.25 ratio
is the host/gap permittivity contrast: the finite smooth cell derivative omits
the first-order contribution of the thin normal-field boundary layer. Increasing
H after differentiating at contact does not recover it. This prevents adoption;
matching AD to the candidate's own secants cannot close this physics gate.

The expanded JAX test also exposed a separate roughly 6 ppm material-gradient
bias from the existing eigenvalue regularization on an s-polarized translated
rectangle. The native nondegenerate derivative passes the same strict tolerance.
Production eigenvalue regularization was not modified. Both this limitation and
the s contact failure have explicit strict expected-failure tests, alongside
production's original split/merge diagnostic; no original test was relaxed.

Validation: **35 passed, 5 skipped, 3 xfailed** in the ordinary environment, plus
**5 passed** in the optional NGSolve cell suite. The skips are those optional
tests. Ruff and `git diff --check` pass. The earlier 3D FEM resource failure was
not rerun, and no study process remains active. Full construction, derivation,
measurements and commands are in [`nvf-cell-corrector.md`](nvf-cell-corrector.md).
The next formulation work must represent the vanishing layer's normal field
in the forward energy or an equivalent interface term, while preserving
split/merge equality. A post-hoc gradient multiplier would not solve that task.

## Conforming thin-gap forward energy, 2026-09-08

The missing normal-field contribution now has a concrete variational
implementation in `studies/nvf_thin_gap_energy.py`. It represents the potential
jump across the gap as `gap * a(x)` in a periodic conforming Q1 cell space.
Dividing the enrichment test equation by gap analytically retains a finite
normal field and its first-order energy at zero gap. The optical derivative is
obtained from the actual forward energy closure, with no multiplier or cutoff.
At positive gaps the projected fields agree with an independently assembled
ordinary nodal Q1 solve. No Cartesian Li routine is called by the model.

The order-7 closed rectangular control gives p T=0.626061531897 and
s T=0.878242289574, matching the independent constitutive matrices. The p
contact derivative is 0.001891546100. With x interface fractions
1e-4 / 1e-5 / 1e-6, the s derivatives are -0.000777803964 /
-0.000778495983 / -0.000778566984, versus the independent approximately
-0.000778575. The last discrepancy is about 0.001%, compared with the old
Fourier model's roughly 92% error. The mean static normal-energy derivative
also converges to the analytic series-laminate variation independently of the
optical solve. Refining the body mesh near the gap is necessary for positive-gap
secants to approach the endpoint derivative promptly; a coarse uniform body
mesh alone is insufficient.

JIT checks cover gap, permittivity and width, positive-gap derivative continuity,
and a live material derivative at zero contrast. Cell integrals are analytic;
12-to-16-point Fourier projection changes the recorded order-7 coefficients by
at most 1.40e-13. The finest recorded solve has 611 scalar unknowns, with scaled
linear residual about 1.3e-10. Detailed derivation, reproducible measurements
and limits are in [`nvf-thin-gap-energy.md`](nvf-thin-gap-energy.md).

Validation: **12 new study tests pass**; the combined relevant selection gives
**47 passed, 5 skipped, 3 xfailed**. The three original strict expected failures
are retained. Targeted Ruff and `git diff --check` pass. This increment changes
no production solver, dispatch, or physics tolerance. The new conforming model
currently accepts one axis-aligned rectangle, so arbitrary primitive splits,
tilted/curved interfaces and production integration remain open. The prior
tilted optical and independent FEM convergence uncertainties also remain.

## Partitioned material integration on the conforming gap space, 2026-09-08

`studies/nvf_partitioned_gap_energy.py` now combines equal-material partition
invariance and the conforming gap field in one finite model. It keeps the
envelope mesh fixed and writes the material as a base envelope plus suffixes
weighted by the jumps in permittivity. Equal-material seams have exactly zero
weight without a material-equality predicate; independent material derivatives
remain live. Analytic polynomial/Fourier moments integrate the material and
the actual energy closure. A translated horizontal rectangle-tiling adapter
accepts one rectangle or its ordered partition, retaining live geometry under
JIT and rejecting invalid traced tilings.

Direct coefficient arrays are bit-identical for equal-material partitions,
including redundant seams in a nonuniform profile. Shape values and physical
gradients pass 1e-10 and 2e-10 comparison criteria, including movable artificial
seams, translation through periodic cell boundaries and both polarizations.
At n=7, edge fraction 1e-6, the translated merged and split contact values are
p T=0.626061531896588 and s T=0.878242289574451; their gap derivatives are
0.001891546100056 and -0.000778566984054. The earlier approximately 0.001% s
contact error is retained. Material derivatives at equality, at zero contrast
and at unequal values agree with NumPy secants. The model calls no Cartesian Li
routine; the latter is used only as an independent control.

Analytic projection initially lost about 1e-11 by contracting the potentials
with an assembled stiffness matrix on tiny elements. Local potential
differences before products reduce disagreement with independent conforming
quadrature to about 2e-14. The cell solve and energy closure are shared with
the preceding study. Explicit x subdivisions refine the basis independently
of material partitions, preserving representation invariance at each level.

This does not close real-interface accuracy. An off-node 4/9 seam retains
static normal-energy error that decreases from 0.00775849 to 0.00214796 with
1-to-4 x subdivisions. More seriously, moving that unequal seam through a
mesh node gives one-sided static derivatives -0.001281263686 and
-0.000253089616 instead of the independent -0.000569451310. The closed value
is exact but the fixed polynomial space does not carry the moving field jump.
This is a newly recorded strict expected-failure gate, separate from the
passing cancellation of equal-material seams. The next proposed basis uses
elementwise normalized integrals of reciprocal permittivity, so real field
jumps can move while the mesh and primitive-partition invariance stay fixed.
It must be checked in the forward energy before adoption.

Validation: **19 new tests pass**, and the new real-seam physics gate is
strictly xfailed. Together with the preceding selection this gives
**66 passed, 5 skipped, 4 xfailed**; the three earlier strict expected failures
are retained. Targeted Ruff and `git diff --check` pass. No production dispatch,
solver or physics tolerance was changed. Full derivation, measurements and
reproduction are in [`nvf-partitioned-gap-energy.md`](nvf-partitioned-gap-energy.md).

## Material-harmonic coordinates and the affine lift, 2026-09-08

`studies/nvf_harmonic_coordinates.py` now represents the real field jump inside
a fixed x element using its normalized cumulative integral of reciprocal
permittivity. The x forcing also uses the corresponding quasiperiodic affine
lift. Updating only the corrector basis gives the wrong static solution;
the implemented formulation changes the body/gap forcing and all lift and
cross terms in the actual projected forward energy. Exact piecewise Fourier
moments and algebraic material-jump cancellation retain partition invariance
and live independent material derivatives at equality.

The closed 4/9 laminate's seam-zero energy is 1.6941176470588233 and its JIT
seam derivative is -0.0005694513099357396, matching the independent analytic
values. The off-node seam at 31 nm gives 1.6766467065868258 and
-0.0005577667021246928, also matching the analytic values. The old model's
left/right seam-zero slopes were -0.001281263686 / -0.000253089616. That old
strict static-failure gate now passes unchanged for the new model while
remaining xfailed for the original model. Closed constitutive matrices and
p/s optical seam/material derivatives pass independent Cartesian controls;
the model itself never calls Cartesian Li. The inherited translated partition,
contact, live material and invalid-update guards continue to pass.

A positive gap still exposes a smaller finite-mesh limitation. At gap 17 nm,
outer order 2 and s polarization, a real seam crossing node zero gives slopes
-0.001382052641 and -0.001383810998 per nm. Their 1.75836e-6 discrepancy drops
to 8.86907e-7 with four x subdivisions but remains above the 1e-7 target.
This is a new strict expected-failure gate, separate from a passing refinement
trend test. A small cell residual, unit mean B and conserved lossless power
do not close that shape-sensitivity error. A globally smooth material-coordinate
basis is a possible next investigation; no post-hoc gradient correction was
introduced.

Validation: **94 passed, 5 skipped, 5 xfailed** in the combined relevant
selection, including **11 passed, 1 xfailed** in the new dedicated module.
The four preceding strict expected failures and all existing tolerances are
retained. Shared study helpers support the new lift but preserve the original
models' default paths. No production NVF dispatch or eigenvalue derivative
was changed. Full formulation, numerical tables and reproduction commands
are in [`nvf-harmonic-coordinates.md`](nvf-harmonic-coordinates.md); the default
optical and static CLI outputs are in `studies/results/nvf_harmonic_coordinates.log`.

## Integrated continuous flux closes the tested mesh-node seam defect, 2026-09-08

`studies/nvf_continuous_flux.py` integrates periodic P1 flux hats against
reciprocal permittivity to build the potential, with a consistent global
harmonic affine lift. Both the affine body flux and the periodic-correction
body flux are continuous through artificial mesh nodes. The real field jump
therefore follows a moving material seam without retaining the preceding
model's element-boundary flux jump. Analytic Fourier moments through degree
four integrate the quadratic potentials and their actual energy. Signed
material-polynomial jumps retain exact redundant-partition cancellation and
live material derivatives without an equality predicate.

The positive-gap order-2 s seam secants are now -0.001387386046 and
-0.001387388302 per nm, a difference 2.26e-9 compared with the preceding
1.76e-6. The existing 1e-7 continuity gate passes for the new model and stays
strictly xfailed for the old local-coordinate model. Additional p/s JIT
checks agree with both one-sided derivatives. Closed static and independent
Cartesian optical controls, translated partition equality, contact physics,
material derivatives and invalid-update guards also pass unchanged.

At order 7, uniform epsilon 12.25, rows 8 and edge fraction 1e-5, the new
closed p/s transmissions are 0.626061531896529 / 0.878242289574428. The gap
derivatives are 0.001891546100057 / -0.000778519822153, both within the same
1e-7 independent contact criterion. No Cartesian Li routine is called by
the model, and no production NVF or eigenvalue derivative was changed.

For a real seam at 31 nm and gap 17 nm, nested x subdivisions 1/2/4 decrease
static Uyy from 3.187630224161 to 3.186369469373 to 3.186197687187. At outer
order 2, s T changes from 0.906437006083 to 0.906236254717 to 0.906224852820.
These changes retain a cell/optical convergence question; the small residual
and corrected seam derivative are not an optical error bound. Tilted/curved
geometry, the independent tilted reference and production integration remain
open.

Validation: **122 passed, 5 skipped, 5 xfailed** in the combined selection.
The new model passes 23 shared checks and five dedicated integral/JIT checks.
All prior strict failure controls and original tolerances remain in place.
Full derivation and reproduction are in
[`nvf-continuous-flux.md`](nvf-continuous-flux.md), with numerical output in
`studies/results/nvf_continuous_flux.log`.

## Hierarchical finite-gap field and independent convergence controls, 2026-09-08

The integrated-flux model now adds interior gap potential bubbles scaled by
gap. Each bubble vanishes at both gap faces; dividing its test row by gap
analytically retains a nonsingular zero-gap limit. At contact the bubbles
vanish, preserving the previous first-order energy. Gap degree 4 is now the
study default, with degree 1 retained as the linear-gap control. Nested body
y subdivisions refine the graded near-gap intervals independently of material
partitions. The new forward terms pass full tensor Gauss projection, JIT
gap/seam/material, exact-contact and redundant-partition checks.

`studies/nvf_gap_convergence.py` now reuses a maximal projection across outer
orders and compares against independent Cartesian material intervals. A
separate `studies/nvf_gap_reference.py` fits periodic 2D FEM triangles to the
mixed material profile. Its primal/reciprocal static enclosure narrows from
3.23e-6 at degree 6 to 7.94e-7 at degree 8 with 43649 scalar unknowns. At the
fine mesh, cap-32 Fourier coefficients change by 1.69e-15 when projection
quadrature increases from 20 to 24. This is an auxiliary-field reference;
the heavy 3D tilted FEM refinement was not restarted.

At outer order 16, width 350, gap 17, seam 31 and materials 4/9, the fitted
FEM field gives p/s T=0.907165915611 / 0.941480649651 through the same energy
closure. The x4 linear-gap model gives 0.907203629098 / 0.941560169727; the
degree-4 gap gives 0.907170121909 / 0.941553172975. Further body y refinement
gives 0.907170049872 / 0.941540519833. The independent Cartesian values are
0.907193571579 / 0.941468107350. The model's static enclosure narrows from
0.00170539 to 0.000771781 across the corresponding gap/body refinements.

These improvements do not establish optical convergence. Even with the fitted
cell held fixed, outer orders 12 to 16 change p/s T by about 4.78e-4 / 2.99e-4.
This is larger than several of the same-order formulation differences. The
new instrument distinguishes these errors instead of treating a cell residual
or static enclosure as an optical bound.

The two polarizations now share one layer eigensolve. Polynomial gap projection
contracts the x basis before multiplying by y moments to avoid redundant work.
Independent slab and tensor quadrature controls retain the previous physics
criteria. Validation: **136 passed, 8 skipped, 5 xfailed** in the combined
selection, plus all **8 optional FEM tests passing** in their environment.
No existing tolerance, production NVF dispatch or eigenvalue regularization
was changed. Derivation, tables, commands and log locations are in
[`nvf-gap-convergence.md`](nvf-gap-convergence.md).

## Material-defined interface resolution, 2026-09-08

`studies/nvf_interface_resolution.py` redistributes the continuous-flux model's
fixed number of envelope nodes using a smooth positive density of material
jumps. Equal-material seams cancel algebraically, including in profiles with
multiple real interfaces. No equality branch, topology change or stopped
geometry gradient is introduced. NumPy/JAX share a bracketed/Newton root solve;
independent Brent roots and their secants check values and mesh Jacobians.
The existing partition, contact, material and real-seam derivative gates pass
for the new opt-in subclass.

At width 350 nm, gap 17 nm, seam 31 nm, materials 4/9, base rows 8, y subdivisions
2 and gap degree 4, grading strength 8 and radius .01 reduce the relative
static enclosure from 0.00102464 to 0.000367877 with 2591 unknowns. With 5183
unknowns it decreases from 0.000771781 to 0.000280985. At outer order 16, the
5183-unknown s transmission difference against the fitted FEM cell projection
decreases from 5.9870e-5 to 4.4462e-5, about 26%; p improves as well. The smaller
graded model is closer in both polarizations than the larger fixed-mesh control.
This reduces the tested cell approximation error, not a certified optical bound.

Independent Cartesian and fitted-cell optical comparisons now reach order 20.
The fitted-cell p/s transmissions are 0.906915509832 / 0.941563020477; Cartesian
values are 0.906934177474 / 0.941561099135. Despite their closer agreement,
fitted-cell transmission changes by about 2.50e-4 / 8.24e-5 from order 16 to 20,
so outer convergence remains open. The cap-40 FEM coefficients change by at
most 2.98e-15 under projection quadrature 24 to 28.

Validation: **168 passed, 8 skipped, 5 xfailed**, including seven dedicated
grading tests and 25 new shared parameterizations. Prior expected failures and
physics tolerances remain intact. Production NVF dispatch and eigenvalue
regularization are unchanged. Full derivation, settings, commands and logs are
in [`nvf-interface-resolution.md`](nvf-interface-resolution.md). General
geometry, optical convergence and production integration remain outstanding.

## Directional convergence and a contact-resolved cell setting, 2026-09-08

The shared NVF research instruments now accept explicit (x,y) Fourier half
widths in addition to square orders. They build each retained closure before
inversion and preserve the material profile, auxiliary solve and scattering
geometry. Independent slab formulas, physical axis swapping, the public
scattering path and transverse-invariant laminate controls validate the new
order handling. Production NVF dispatch is unchanged.

The fitted-cell four-corner comparison from orders 20 to 24 reveals opposing
s increments: x refinement contributes +1.9555e-5 and y refinement contributes
-1.4240e-5. Increasing both gives only +5.3840e-6. Thus apparent stability of the
square-order sequence partly reflects cancellation, not convergence. The p
square-order change remains -1.2145e-4. At order 24 the fitted-cell p/s T is
0.906794061748 / 0.941568404449, compared with Cartesian intervals at
0.906805437058 / 0.941572093232. No converged optical reference is accepted.

On the auxiliary cell, spending unknowns on y instead of further x refinement
is more effective for this fixture. Graded x2/y4 with gap degree 6 uses 5039
unknowns, fewer than the preceding x4/y2 model's 5183. The original finite-gap
sweep setting edge_fraction=1e-4 still misses the s contact derivative criterion
by about 2.1e-7. Resolving that edge with the fixed setting 1e-5 reduces the
error below the existing 1e-7 criterion, without adding unknowns or selecting
a different model at zero gap.

The fine edge exposes badly scaled linear equations. A live diagonal change
of coordinates now equilibrates the enriched continuous-flux system, adding
no regularization and retaining the original residual check. Conditioning
warnings are errors in the final refined contact test. The p/s derivative
discrepancies are 3.84e-9 / 2.07e-8 per nm. The final static enclosure is
0.000182637470, about 35% narrower than the preceding 5183-unknown model, and
the order-16 s difference against the fitted cell is about 1.78e-5 instead of
4.45e-5. This is about a further 60% reduction, with fewer unknowns.

Full settings, directional tables, the rejected coarse contact-edge setting,
equilibration formula and reproduction commands are in
[`nvf-directional-convergence.md`](nvf-directional-convergence.md). The
remaining limits are optical convergence, general geometry and production
integration; no tolerance or eigenvalue derivative regularization was changed.

Final validation after equilibration: **177 passed, 8 skipped, 5 xfailed**.
The refined contact check promotes conditioning warnings to errors. Existing
strict failures stay attached to their original models, and the shared JIT
and material-partition checks retain their physical tolerances.

## Conforming polynomial body implementation, 2026-09-08

`studies/nvf_polynomial_body.py` now enriches the body y potential with local
integrated Legendre bubbles. Their traces vanish at the interval endpoints,
so the periodic/contact coupling and x material-integrated flux remain the
same. The actual polynomial field and its energy are projected with analytic
Fourier moments. Local derivative couplings use Legendre orthogonality to
avoid cancellation on tiny contact cells. Body degree 1 is an exact control
for the preceding material-graded model; degrees 2 through 4 are opt-in.

At the same 5039 unknowns, body degree 4 narrows the relative static enclosure
from 0.000182637470 to 0.000115356596, about 37%. At outer order 16 its s
discrepancy against the preceding fitted-cell control decreases from 1.78458e-5
to 3.27400e-6, about 82%. A degree-3 model with 3887 unknowns also improves the
comparison. At outer order 24 the degree-4 p/s values are
0.906794978336 / 0.941570869123. These remain cell-control comparisons, not
certified optical errors.

The independent FEM reference was refined to h=.03 micrometers, degree 8 and
59969 scalar unknowns. Its relative static enclosure is 4.62166e-7, and cap-48
quadrature 24 to 28 changes coefficients by at most 2.67e-15. The reference
p/s transmissions move by only about 8.5e-9 / 2.2e-8 at both orders 16 and 24.
Thus the candidate improvement survives a finer independent cell comparison.
The final degree-4 p/s contact derivative errors are -2.55e-11 / 2.01e-8 per
nm, satisfying the existing 1e-7 criterion without conditioning warnings.

Validation: **213 passed, 8 skipped, 5 xfailed**, including ten independent
polynomial/JIT tests, 25 inherited physical/model checks and one final-resolution
contact check added to the preceding selection. Production dispatch and all
existing physical tolerances remain unchanged. This is research implementation;
outer optical convergence, general geometry and product integration remain
outstanding. Full derivation, settings, raw logs and reproduction commands are
in [`nvf-polynomial-body.md`](nvf-polynomial-body.md).

### Polynomial x flux and resolution allocation (2026-09-08)

The new `PolynomialFluxGapModel` implements positive interior Bernstein fluxes
and their reciprocal-material integrated potentials. Independent piecewise
Gauss integration, partition invariance and live JIT derivatives are checked.
The nodal partition of unity alone defines the affine lift, preventing bubble
normalizations from changing its physical mean.

Equal-dimension static comparisons favor refining the previous linear flux,
so the new degree-2/3 space is not selected as a replacement. Keeping linear
flux, body degree 4 and increasing x subdivisions from 2 to 3 narrows the
static enclosure by 57.7% and reduces the order-16 p/s discrepancy against the
finer fitted-cell control by 55.1% / 49.3%, using 7559 rather than 5039 unknowns.
The lower-cost degree-3 body option narrows static bounds but slightly worsens
the s optical comparison. No static-to-optical error bound is inferred.

Full implementation, comparisons and reproduction are recorded in
[`nvf-polynomial-flux.md`](nvf-polynomial-flux.md). This remains research code;
the outer optical and general-geometry questions, and production integration,
are still open.

The final checks cover **253 distinct passing tests, 8 skipped and 5 strict
expected failures** across targeted and complementary regression selections.
All four added final-resolution contact configurations retain the existing
independent p/s derivative criterion with conditioning warnings treated as
errors. The degree-3 direct JAX CLI also matches the closed-laminate static
energy and real-seam derivative to floating-point precision.

## Resonance-position convergence on the tilted fixture, 2026-09-11

The question left open above is which of the order-22 answers on the tilted
contact fixture is right: circular NVF `T=0.853755` or the direct rule
`T=0.890772`. Neither is, and the reason is that `T(900)` is the wrong
observable for this fixture. Instrument: `studies/nvf_resonant_spectrum.py`,
the probe's methods with a wavelength argument and every propagating order
recorded; NumPy backend, grid 512, this host (10 cores, 16 GB). Raw JSON lines
are in `studies/results/nvf_resonant_spectrum.log`.

### What the total hides

The substrate propagates `(0,0)`, `(+-1,0)` and `(0,+-1)`; air only `(0,0)`.
The `(1,0)` substrate order has its Rayleigh line at
`700 (1.5 - sin .2) = 910.93` nm, 11 nm from the operating wavelength, and at
900 nm it still leaves at 81.7 degrees (`kz/k0 = 0.216`) carrying a fifth of
the power. Circular minus direct at 900 nm, grid 512:

| order | T(0,0) | T(1,0) | T(-1,0) | T(0,1) | T(0,-1) | R(0,0) | sum(T) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | -0.2492 | +0.1233 | +0.0760 | -0.0426 | -0.0226 | +0.1152 | -0.1152 |
| 14 | -0.1160 | +0.0614 | +0.0333 | -0.0245 | -0.0129 | +0.0588 | -0.0588 |
| 22 | -0.0724 | +0.0382 | +0.0215 | -0.0155 | -0.0088 | +0.0370 | -0.0370 |
| 26 | -0.0593 | +0.0309 | +0.0179 | -0.0126 | -0.0072 | +0.0303 | -0.0303 |

The total difference is half the `(0,0)` difference; every channel moves
with the same sign pattern at every order, which is what a shifted
resonance looks like, not what a factorization disagreement about a smooth
quantity looks like.

### The spectrum

A 5 nm scan from 860 to 960 nm at orders 6/10/14 finds a reflection
resonance (`R(0,0)` up to 0.76) sitting on the Rayleigh line, with a second
one near 955 nm. 900 nm is on the blue flank of the first: the slope there is
about `-0.015` per nm for circular NVF and `-0.010` per nm for the direct rule,
so a 0.1 nm shift of the resonance is `1.5e-3` in `T(900)`, and the
0.037 gap corresponds to a 2-3 nm shift. At every wavelength the two order
ladders move toward each other (at 900 circular rises 0.8265 / 0.8386 / 0.8455
and direct falls 0.9417 / 0.9175 / 0.9043; at 920 direct rises
0.32 / 0.39 / 0.43 and circular falls 0.63 / 0.59 / 0.58).

A 1 nm scan from 905 to 925 nm then gives two markers per method and order:
the `R(0,0)` peak (parabola through the maximum and its two neighbours) and
the wavelength where `T` crosses 0.5 on the red side. The blue crossing sits
on the Rayleigh cusp itself and is not used.

| order | circular peak | circular red crossing | pieces peak | pieces red crossing | direct peak | direct red crossing |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 912.531 | 917.926 | 913.283 | 918.450 | 918.086 | 922.248 |
| 10 | 913.233 | 918.566 | 914.008 | 919.108 | 916.857 | 921.331 |
| 14 | 913.556 | 918.828 | 914.245 | 919.309 | 916.226 | 920.845 |
| 18 | 913.789 | 918.991 | — | — | 915.859 | 920.561 |
| 22 | 913.967 | 919.117 | — | — | — | — |

Circular NVF moves red with order, the direct rule moves blue, and the
increments decay at about `1/N` on both sides. Fitting orders 6-18 of both
methods to one common limit with one shared exponent gives:

| marker | common limit | exponent | max residual | last-pair `1/N`, circular | last-pair `1/N`, direct | order-22 circular: measured / predicted |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `R(0,0)` peak | 914.52 nm | 0.88 | 0.05 nm | 914.60 | 914.58 | 913.967 / 913.883 |
| red `T=0.5` crossing | 919.61 nm | 0.90 | 0.05 nm | 919.56 | 919.57 | 919.117 / 919.093 |

Dropping order 18 from the fit moves the limits by 0.01 and 0.02 nm. Two
independent factorizations approaching one wavelength from opposite sides and
agreeing to 0.03 nm after extrapolation is the evidence that the limit is
real; the default linear piece field lands about 0.25 nm redder on the same
extrapolation and was not run past order 14.

### `T(900)` to order 26, and the FEM row

| order | circular NVF | direct rule | circular increment | direct increment | seconds (circular / direct) | peak RSS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 0.826543208 | 0.941709091 | | | 0.2 / 0.2 | |
| 10 | 0.838581537 | 0.917495725 | +0.01204 | -0.02421 | 1.3 / 1.3 | |
| 14 | 0.845478595 | 0.904278462 | +0.00690 | -0.01322 | 6.3 / 6.1 | |
| 18 | 0.850166086 | 0.896188108 | +0.00469 | -0.00809 | 23 / 22 | |
| 22 | 0.853755019 | 0.890772020 | +0.00359 | -0.00542 | 71 / 66 | 3.6 GB (3.4 GB physical) |
| 26 | 0.856604151 | 0.886903561 | +0.00285 | -0.00387 | 178 / 171 | 5.0 GB (6.4 GB physical, 1.6 GB compressed) |

Order 30 would need about 8.5 GB and was not attempted. The common-limit fit
of `T(900)` gives 0.8665 (exponent 0.85, orders 6-26), 0.8660 (6-22), 0.8674
(10-26) and 0.8668 (10-22); the last-pair `1/N` extrapolations are 0.8723 /
0.8656 (22-26) and 0.8699 / 0.8664 (18-22). The direct rule's increment
ratios (1.83, 1.63, 1.49, 1.40) track `1/N` (2.33, 1.80, 1.57, 1.44); the
circular ladder's (1.75, 1.47, 1.31, 1.26) fall below it from order 18 on, so
its own limit is less certain than the joint one -- that is the reason the
last-pair number on the circular side drifts up.

The independent H(curl) FEM with the Rayleigh boundary
(`studies/nvf_fem_reference.py`, [details](nvf-independent-fem.md)) on the
same fixture: `h=.18`, degree 3, cut 2, pad .4 gives modal `T=0.863053`,
volume `T=0.862943` (bit-identical to the logged prototype value), energy
error `1.6e-4`, 81 s, peak RSS 4.19 GB (ru_maxrss 4.58 GB, physical footprint
3.59 GB). Degree 4 on the same mesh (177,625 unknowns), rerun with the footprint
sampled from outside: modal `T=0.866063`, volume `T=0.866056` (bit-identical
to the logged prototype value), energy error `2.7e-6`, modal/volume
disagreement `6.7e-6`, 362 s. Peak RSS 8.57 GB (ru_maxrss 9.01 GB) while the
physical footprint peaked at 14 GB with 11-12 GB of it compressed, swap grew
from 4.7 to 18 GB, and the memory-pressure level never passed 2 (warning)
on this 16 GB host. That is the size the RSS guard was hiding: a lone
process this big runs, slowly, because macOS compresses and swaps it; the
documented panics came from two such processes at once. The same degree
with pad .6 (174,645 unknowns after remeshing) gives modal `T=0.866027`,
volume `T=0.866020`, energy error `1.5e-5`, 437 s, RSS 8.26 GB, footprint
15.4 GB: the pad step that was `-1.7e-3` at degree 3 is `-3.6e-5` at
degree 4, so at degree 4 the outgoing boundary is settled and what remains
of the FEM's uncertainty is its own degree step.
Moving the Rayleigh planes from pad .4 to .6 at degree 3 (which also
regenerates the mesh: 3,421 elements, 84,860 unknowns) gives modal
`T=0.861317`, volume `T=0.861203`, energy error `7.8e-4`, 79 s, peak RSS
4.91 GB (physical 4.59 GB) -- a `-1.7e-3` change with the energy error up by
5x. The FEM's own degree step is `3.1e-3` and its pad/remesh step `1.7e-3`,
so the FEM band on this fixture is about `+-3e-3`, the same width as the
spread of the RCWA extrapolations.

The FEM can also be asked the better question. Setting the instrument's
module wavelength before each `solve_rayleigh` call (every reader of `k0`
takes it from the module at call time, and the mesh does not depend on it),
degree 3, cut 2, pad .4, 81-89 s per point:

| wavelength (nm) | 909 | 911 | 913 | 915 | 917 | 919 | 921 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| modal `R(0,0)` | 0.3619 | 0.5858 | 0.7225 | 0.7413 | 0.6589 | 0.5180 | 0.3787 |
| modal `T` | 0.6380 | 0.4141 | 0.2775 | 0.2588 | 0.3412 | 0.4819 | 0.6212 |

Its `R(0,0)` peak (parabola through 913/915/917) is at **914.37 nm**, peak
0.746, and its red `T=0.5` crossing at **919.26 nm** -- 0.15 and 0.35 nm
blue of the RCWA common limits (914.52, 919.61), between the order-22
circular markers (913.97, 919.12) and the order-18 direct ones (915.86,
920.56). A 0.35 nm blue offset on a `-0.012`/nm flank is `4e-3` in `T(900)`,
which is the degree-3 FEM's distance from the extrapolated 0.867; degree 4
moves it red by the `3.1e-3` step above. Three discretizations -- two
factorizations of one Fourier solver approaching from opposite sides, and a
conforming FEM -- put the resonance at 914.4-914.5 nm.

Degree 4 on the same mesh, five points, 347-383 s each, RSS 8.8-9.0 GB:

| wavelength (nm) | 913 | 915 | 917 | 919 | 921 |
| --- | ---: | ---: | ---: | ---: | ---: |
| modal `R(0,0)`, degree 4 | 0.7137 | 0.7437 | 0.6734 | 0.5380 | 0.3968 |
| modal `T`, degree 4 | 0.2863 | 0.2563 | 0.3266 | 0.4620 | 0.6032 |
| `T` change from degree 3 | +0.0088 | -0.0025 | -0.0146 | -0.0200 | -0.0181 |

Its `R(0,0)` peak is at **914.60 nm** (0.746) and its red `T=0.5` crossing at
**919.54 nm**. The degree-3 to degree-4 step moves both markers red by
0.23 / 0.28 nm, while the RCWA common limits sit at 914.52 / 919.61 and the
two last-pair `1/N` extrapolations at 914.60, 914.58 / 919.56, 919.57. The
independent FEM at degree 4 therefore lands within 0.1 nm of the value the
two Fourier ladders extrapolate to from opposite sides, and 0.6 / 1.3 nm
from the order-22 circular and order-18 direct markers. On the flank that
is `1e-3` in `T(900)`. The per-point `T` changes of up to 0.02 between FEM
degrees are the same resonance shift seen through a fixed wavelength, which
is why a single-wavelength FEM ladder reads as unconverged when the position
is already settled to a quarter of a nanometre.

Resource note for the scan: the process footprint reached 19.5 GB (swap,
not RAM), and one 5-second sample during the 921 nm point read pressure
level 4; the guard requires two consecutive critical samples and did not
fire. That is the edge of what a lone process should be asked to do here.

### What this decides

- **Neither order-22 answer is right.** Both ladders converge at roughly
  `1/N` from opposite sides of one limit: `T(900) = 0.867 +- 0.003` from the
  RCWA extrapolations, with the degree-4 FEM at 0.866 inside that band. At
  order 22 circular NVF is `-0.013` from it and the direct rule `+0.024`;
  at order 26, `-0.010` and `+0.020`.
- **The defensible reference is a resonance position, not a number at
  900 nm:** peak `914.5 +- 0.1` nm, red `T=0.5` crossing `919.6 +- 0.05` nm.
  A `T(900)` reference at `1e-3` would need the resonance to 0.07 nm, which no
  method here has reached.
- **On this fixture NVF buys a prefactor, not a rate.** The circular NVF
  prefactor is about half the direct rule's at the same `1/N` decay. The
  contact leaves two 270-degree re-entrant corners at the ends of the 13.9 nm
  run, and a corner singularity is not something any interface factorization
  removes; the fixture measures corner convergence with a resonance as the
  amplifier.
- The circular / pieces / polygon spread (0.003-0.008 at orders 18-22) is
  the extension's contribution and is small against the truncation error;
  none of the three is wrong in a way the others are not.

Where this stops being trustworthy: the common-limit fits assume one limit
and one shared power, which the pieces field's 0.25 nm offset shows is only
approximately true across extensions; the circular `T(900)` ladder is slowing
below `1/N` at orders 22-26; the FEM has a `3e-3` degree sensitivity and its
degree-4 footprint is unrecorded. A reference good to `1e-3` needs the
degree-4 FEM (or finer) with its footprint recorded, run on a host with more
free memory, plus a FEM wavelength scan at degree 3 to pin the resonance
independently. Reproduce with:

```bash
PYTHONPATH=$PWD python studies/nvf_resonant_spectrum.py --orders 6 10 14 --methods circular pieces wavebit-direct
PYTHONPATH=$PWD python studies/nvf_resonant_spectrum.py --orders 6 10 14 18 --methods circular wavebit-direct --wavelengths $(seq 905 925)
PYTHONPATH=$PWD python studies/nvf_resonant_spectrum.py --orders 22 26 --methods circular wavebit-direct --wavelengths 900
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 "$NVF_FEM_PYTHON" studies/nvf_fem_reference.py --hs .18 --orders 3 --boundary rayleigh --cut 2
```

## The J and K steps are not resonance-shift artefacts (2026-09-11)

The tilted fixture taught that a fixed-wavelength `T` near a resonance is
not a convergence observable. Every number in the case J and K rows is a
fixed-wavelength `T` difference, and both ladders are non-monotone in the
order, which is the signature a sliding spectral feature leaves. So the
same instrument was pointed at them: a wavelength scan over 864-936 nm
(`+-4%` of 900) at the documented order and grid, for each member of each
pair, asking whether the documented step equals the local slope times the
shift of the nearest spectral feature. Scripts under the session
scratchpad; each fixture was first reproduced to every printed digit of its
row (J: `-8.0007e-4 / -3.5678e-4 / -9.2235e-5` and secants
`0.0104054 / 0.7311734 / 50.64041`; K: the pre-fix row values reproduced on
`git archive` copies of the commits that wrote them, e2fdafd and c000c73).

The answer is no, for both. The only sharp feature in the window is the
substrate `(1, 0)` Rayleigh cutoff at `700 (1.5 - sin 0.2) = 910.93` nm,
which depends on the period, the angle and the substrate index alone and
does not move between pair members (`< 0.1` nm measured, zero analytically).
There is no structure resonance: J's spectrum has one very flat `sum(T)`
maximum at 890.7 nm (curvature `-4.3e-5` per nm squared) that shifts by
`-0.074` nm at gap 1e-4, whose slope times shift is at most 29% of the
`-3.57e-4` step; K's four spectra have no extremum other than the cusp.
The decisive test is the sign: a shift makes the difference proportional
to the local slope, vanishing at an extremum and changing sign with the
slope. J's difference from contact is negative at every one of the 41 + 81
+ 61 wavelengths measured, including at the 890.7 peak where the slope is
zero (`-2.88e-4`) and on both flanks where it changes sign. K's `-diff /
slope` runs from `+5.2` nm at 864 through zero near 888 to `-0.35` nm at
900 and `-0.80` nm at 936 for the contact pair, and from `+0.50` through
zero near 886 to `-0.38` and `-0.49` nm for the material pair: not a
constant, so not a shift.

What the scan adds is that the steps are wavelength-dependent and change
sign across the window. J's difference from contact at gap 1e-4 runs from
`-1.1e-4` at 864 to `-7.1e-4` at the cusp; K's contact step (order 8,
N = 512) is `+8.0e-4` at 864, zero near 888, `-2.0e-4` at 900 and `+5.6e-4`
at 936, and its material-equality step (N = 128) `+8.6e-4`, zero near 886,
`-5.9e-4` at 900, `-1.04e-3` at 929. The single-wavelength values in the
rows are point samples of smooth functions; the band across the window is
the honest size. Both fixtures sit 11 nm below a Wood anomaly with the
`(1, 0)` substrate order near grazing (`k_z / k_0 = 0.216`, carrying
`0.166` of the transmitted power in K), which is where a factorization
error is amplified, and is the same Rayleigh line the tilted fixture's
resonance sits 3.6 nm from.

The K row's ladders predate the 2026-09-07 run-end fix; on the current code
(c1c210d, landed as 4d34937) they read contact `-2.736e-3 / -8.737e-4 / -2.006e-4 / -7.183e-4 /
-3.444e-4` and material-equality `-5.003e-3 / -4.799e-4 / -5.894e-4 /
-1.393e-4 / -3.708e-5` at orders 4 / 6 / 8 / 10 / 12. Only `T_touching`
moved (`T_gap` agrees with the archived commit to 1e-15), consistent with
the fix's stated scope. The J fixture did not move at all.

So the resonance lens applies to the tilted fixture and not to J or K: the
contact steps are genuine finite-order steps of the factorization at 900 nm
and may be judged at the fixed wavelength, at the documented order. The
scan says nothing about their infinite-order limit.

Placement dependence, re-verified the same day on c1c210d (landed as 4d34937) with
`studies/nvf_placement_spread.py` (its rows in
`studies/results/nvf_placement_spread.log`): the closed
form's coefficient spread over translations of two disjoint boxes is
`1.074e-2 / 4.056e-3 / 1.592e-3` at N = 64 / 128 / 256 (fitted exponent
1.38) and the `sum(T)` spread `1.242e-3 / 6.395e-4 / 1.560e-4`, identical to
the 2026-09-07 numbers to every printed digit; a single shape is at
rounding (`2.8e-17`) on every route. The polygon route with its production
anchor (326ea41) is at rounding for the disjoint pair too, but production
routes a disjoint layer to the closed form (`painted=False`), so that fix
reaches only painted layers.

Fixed on 2026-09-11 by building the closed form's layer with its first shape at the cell
centre and carrying it by the phase (the area centroid the polygon route uses was built first
and rejected: it moves under naming a shape a period over and broke the layer's periodicity
across the cell edge): the closed rows are at 3e-17 and the `sum(T)` spread at 3e-13 on a rerun
of the same study. What remains, and why the unsampled Voronoi projector was measured and rejected
(a 0.086 value step at every same-material merge), is recorded in
`docs/normal-field-known-limits.md` beside the polygon route's anchor and in the
`normal_projector_coeffs` docstring.


## Passive metal admission of the exact finite pair — 2026-09-13

The [common material-sector certificate](nvf-pair-material-sector.md) now
allows the exact contact/opening family to evaluate passive metals while
retaining its original physical weak equations. Independent nodal/Laurent
references qualify the finite coefficients, closures and gap derivatives,
including JAX at contact. Both polarizations still show strong cell/order
changes and discrepancies from the separate Cartesian Li factorization.
The ordinary distance-field contact residual and final metallic optical
convergence remain open; no default dispatch changed.


## Finer finite contact cells — 2026-09-13

[Refinements 2/3 and contracted weak assembly](nvf-pair-refinement.md) now
support independent finite values and derivatives through contact with lower
assembly storage. The dielectric value and slope comparison improves, while
metal values and slopes retain large changes with both cell refinement and
outer Fourier order. These are qualified finite numerical tools, not a
resolution of the ordinary NVF contact field or metallic optical convergence.
