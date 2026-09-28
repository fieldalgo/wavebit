# Smoothed occupancy and the tilted-interface measurement

Written: 2026-09-22.

Targets: `wavebit/fd/_lower/occupancy.py`, `wavebit/fd/_lower/coefficients.py`,
`wavebit/fd/_lower/checks.py`, `wavebit/fd/geometry.py`,
`wavebit/fd/_shapes/_cut.py`, `wavebit/fd/_shapes/_bodies.py`,
`studies/fdtd_subpixel_tilted_tensor.py`, `tests/test_fd_occupancy.py`,
`tests/test_fd_occupancy_integration.py`, `tests/test_fd_subpixel_study.py`,
`tests/test_fd_shapes.py`, `wavebit/fd/_lower/junction.py`,
`tests/test_fd_junction.py`, `tests/test_fd_junction_physics.py`.

`fd.Sphere` now supplies fractional material coefficients at each field component's
own location. Unsupported mixtures are refused during planning. Approximate CSG
volumes produce one measured scene warning before compiled lowering. The diagonal
mixing formula remains one part of the inverse material operator; the later
[constitutive implementation](fd-mie-validation.md#production-constitutive-coupling-and-energy)
adds adjoint cross terms and compact D/B memory. The tilted-interface study
does not establish second-order sphere scattering or settle every possible
off-diagonal scheme.

## The coefficient rule

For a fraction `f` of material `a` in material `b`, and the component's squared-normal
moment `w`, the inverse effective permittivity is

```text
1/eps_c = w * (f/eps_a + (1-f)/eps_b)
          + (1-w) / (f*eps_a + (1-f)*eps_b).
```

The normal component is harmonic and the tangential component arithmetic. These are
the diagonal entries of the effective inverse tensor described in
[Farjadpour et al. (2006)](https://doi.org/10.1364/OL.31.002972) and the
[Meep documentation](https://meep.readthedocs.io/en/latest/Subpixel_Smoothing/).
Initially anisotropic phases require the more general construction in
[Kottke et al. (2008)](https://doi.org/10.1103/PhysRevE.77.036611).
Only documentation and papers were consulted; no Meep implementation was read.

`occupancy.py` builds coordinates from each component's Yee offset. Electric
permittivity uses E points; magnetic permeability uses H points. The two are half a
cell apart. The mixing rule receives the fill and the corresponding normal-moment
component at that same point. Non-magnetic scenes avoid unused H evaluations.
Cell-snapped Slab/Box painters keep their old `where` operation, with no reciprocal
round trip. The geometry engine's JIT endpoint repair remains in place; see
[the clipping analysis](fd-sphere-clip-jit.md).

The planar analytic-fill, transfer-matrix, translation and derivative tests remain
the independent gates. Differentiation uses a plan built from concrete geometry,
then differentiates only numerical lowering. Public radius/centre tracing awaits a
declared support range. Existing shape subclasses must answer `body()` explicitly;
`None` selects snapped cells. `Sphere.min_extent(grid)` follows the current protocol,
and non-finite radii and coordinates are refused by name.

## Bounds are reported while planning

The geometry collector is `Approximations.record(node, counted, widest)`. Numerical
lowering does not convert its arrays to Python counts: that would run only while
tracing JIT and disappear on cached executions. Planning and `fd.check` evaluate the
static geometry on NumPy and report the same note, including when a report receives
already-built params.

For two spheres of radius 0.2 centred at x = -0.07 and +0.07, with `dx=0.05` in a
1.2-cubed domain, each of Union, Intersection and Difference reports **284 component
voxel evaluations**, widest bracket **0.05377 of a voxel**. Counts sum over evaluated
components and nodes; they are not counts of unique spatial cells. Union and Difference
take upper volume bounds; Intersection takes a lower bound. A volume bound does not
bound the electromagnetic error.

| Complete coefficient lowering, eps = 3.7 | Worst measured relative difference from NumPy |
| --- | ---: |
| JAX eager | 6.16e-14 |
| JAX JIT | 4.66e-14 |
| Torch | 3.53e-14 |

The regression budget is `1e-12`, set after this measurement (16 times the worst).
The test checks the actual CSG producer, first and cached JIT calls, the named node,
the width, and a report with prebuilt params. The synthetic reporting body separately
checks aggregation: 21 records become 42 for two bodies while the width remains 0.0625.

## The preparation-time limits

The C2 implementation refuses conductive/dispersive or anisotropic media touching
a smoothed support, wrapped or invariant axes a sphere reaches, domain-face losses,
and two separately painted smoothed surfaces in one voxel. These are deliberately
conservative limits until the respective geometry/material rules exist. Touching
surfaces inside one composed body use the geometry kernel and its bound report.
Conductivity and poles are not silently left unsmoothed in an accepted mixture.

The time-step bound also needs the two component locations. Air and a material with
`eps=4, mu=0.25` each have index 1. Before the correction, a radius-0.25 sphere at
`dx=0.05`, Courant 0.5, reported a stability margin of 2. The sampled arrays contained
`min sqrt(eps*mu) = 0.562900`. A safe global bound uses independent minima:

```text
n_bound = sqrt(min(eps) * min(mu)) = sqrt(1 * 0.25) = 0.5.
```

Every positive two-phase mix lies between its material endpoints. Using independent
extrema therefore bounds both staggered coefficient families without assuming that
their minima belong to the same phase or position. It is conservative, not the exact
instability threshold of this finite sphere. Planning refuses a time step at or above
that bound, names `Run(courant=...)`, and never changes `dt`. A tighter scene-specific
bound remains separate work. Since C5 (46a1c33) every scene, snapped or smoothed, uses
`sqrt(min eps * min mu)`: a snapped magnetic pair ran to NaN under the per-phase bound. A scene
where one component of one phase holds both minima -- every scene with `mu = 1`, or with a
vacuum phase and `eps, mu >= 1` elsewhere -- keeps its old bound to the bit.

## Correcting the S1 instrument

The corrected [study](../studies/fdtd_subpixel_tilted_tensor.py) measures a half-filled
unit-period lamellar medium, `k_t p=0.6`, contrasts 1.2/2.25/12, and rational tilts
0/14.036/26.565/45 degrees. The 14.036-degree row is `tan(theta)=1/4`, not 15 degrees.
It uses 16/32/48 nominal cells per period and fits against the actual rounded grid.

Three details change what the old instrument could establish:

1. `sigma=0, which='LM'` finds modes near zero, not the most negative eigenvalue.
   On `diag(-100, 1, ..., 100)` the old minimum/maximum ratio was -0.01 instead of -1.
   The study now uses algebraic smallest/largest queries for Hermitian operators.
2. The naive cross term is non-Hermitian. `eigsh` returns real parts for complex
   input, so the old call could conceal a complex mode. The general `eigs` call keeps
   complex eigenvalues, and the study reports their equation residuals. The solver
   distinction is documented by [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html).
3. Fill coordinates are folded before clipping, and image coverage follows the voxel
   extent. No fixed five-image window or final fill clamp remains. Translating 200
   points by up to 100 periods changed fill by at most 2.57e-13.

A homogeneous anisotropic control is a dispersion measurement in a different medium,
not a rigorous lower bound on the layered medium's error. The study reports both;
it neither subtracts one from the other nor uses a chosen multiplier to erase rows.

## S1 results and what they decide

Measured on Windows, AMD64 Family 25 Model 97 Stepping 2, Python 3.12.14,
NumPy 2.5.1 and SciPy 1.18.0, with two numerical-library threads.
The [recorded run](../studies/results/fdtd_subpixel_tilted_tensor.log) contains 108 series,
324 solves, in 285.7 seconds including overlapping occupancy probes; this elapsed time
is not an isolated performance benchmark. Homogeneous-control orders are 1.999966 to 1.999992;
the arithmetic TM fill control is 1.99900 to 2.06171. The largest selected-eigenpair
relative residual is 5.42e-12. The self-adjoint operator's non-Hermiticity is at most
1.52e-17, versus 0.02485 for the naive writing.

At contrast 12, signed finest-grid error and the three-point fit are:

| Tilt | Diagonal error | Diagonal order / scatter | Self-adjoint tensor error | Tensor order / scatter |
| --- | ---: | ---: | ---: | ---: |
| 14.036 degrees | -1.9374e-3 | 0.999 / 0.015% | -1.8726e-4 | 0.929 / 0.60% |
| 26.565 degrees | -4.3386e-3 | 1.012 / 0.15% | -2.1383e-4 | 0.387 / 107% |
| 45 degrees | -7.0242e-3 | 0.864 / 7.7% | -2.5129e-3 | -2.452 / 196% |

Scatter is `exp(max absolute log-fit residual) - 1`. The last two tensor rows are
non-monotone. Their fitted powers are descriptive numbers, not convergence claims.
The algebraic spectral minimum is positive at every sampled self-adjoint grid;
the smallest minimum/maximum ratio is 1.0303e-5. This tests these matrices only.

The naive 26.565-degree, contrast-12, 48-cell mode has relative imaginary frequency
**2.188e-5**, which the old solver discarded. Reporting its real-frequency accuracy
alone would hide that the comparator does not conserve the Hermitian structure.
At low contrast the tensor's total errors can be close to the homogeneous control;
these samples do not isolate the interface contribution or its asymptotic order.

The diagonal rule is an explicit approximation with a measured tilted-interface
cost. The sampled tensor alternatives do not establish a robust general improvement
in order, and they do not prove that no better off-diagonal discretisation exists.
The 3-D Mie test and the 2-D cylinder study remain necessary before a sphere-scattering
convergence claim. The separate extreme `R/dx` cumulative-volume precision limitation
described in the clipping analysis also remains.

## The edge and dual-face cuts

Written: 2026-09-24. This section adds no physics: every measured number, plan skeleton
and frozen array is unchanged, and the checks below are what says so.

The material operator studied in `studies/fdtd_subpixel_symmetric_stencil.py` does not
constrain the fraction of a *voxel* a body fills. Following
[Werner, Bauer and Cary (2013)](https://doi.org/10.1016/j.jcp.2013.08.009), sections 4
and 7, it constrains the fields on the two objects the Yee grid integrates over: the
**edge** a component's line integral of E runs along, and the **dual face** its flux of
D crosses. Until this commit those two fractions existed only as closed forms inside
`studies/`, for a circle and a sphere, and one of them could not take a backend argument.

`wavebit/fd/_shapes/_cut.py` now carries the same cut one and two dimensions down:

```text
edge_cut(d, n, dx, axis)   the fraction of the dx-long segment along axis
face_cut(d, n, dx, axis)   the fraction of the dx-by-dx square normal to axis
```

both for the tangent half space `n.(x - centre) <= -d`, with `d` and `n` taken at the
**centre of that edge or face** -- the component's own Yee point, which the two share.
They are the same inclusion-exclusion sum `cut` already writes, with the corner count
falling from eight to four and to two: over the live axes, with `a_i = |n_i| dx`,
`A = sum a_i` and `c = clip(-d + A/2, 0, A)`,

```text
m = 3   [ <c>^3 - sum <c-a_i>^3 + sum <c-a_i-a_j>^3 - <c-A>^3 ] / (6 a_1 a_2 a_3)
m = 2   [ <c>^2 - <c-a_1>^2 - <c-a_2>^2 + <c-A>^2 ]            / (2 a_1 a_2)
m = 1   [ <c>   - <c-a_1> ]                                    / a_1
```

each written as a divided difference taken in a stable order, so the only division by a
step is by the largest one. One thing is new below three dimensions: `|n| = 1` no longer
forces the largest step to be large, because the edge can be exactly parallel to the
plane (`n_axis = 0`, every voxel of a slab or of an axis-aligned face) and the dual face
can be parallel to it too. There the cell is wholly inside or wholly outside, and the
measure of the closed half space -- `d <= 0` -- answers it. That is a branch on an exact
zero, not a tolerance.

The body behind `fd.Sphere` -- `wavebit/fd/_shapes/_bodies.py::Sphere`, which
`geometry.py` builds from the public shape -- carries **exact** overrides of both,
`exact_edge_fill` and `exact_face_fill`,
ported from the study forms with their autodiff guards; every other body returns `None`,
including every CSG node. There is deliberately no dispatcher pairing the two, for the
reason in the next paragraph.

### Scope: the tangent plane is not a drop-in for the exact form

C3 reaches only the exact `Sphere` overrides. The generic forms ship for the bodies that
have no closed form -- a CSG node above all, which has no exact clip at any dimension --
and they are a different answer, not a cheaper spelling of the same one. Over 4000 points
about a radius-0.25 sphere per row, one sub-cell jitter each, all three axes:

| R/h | 3 | 4 | 6 | 8 | 12 | 16 | 32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| worst \|edge gap\| | 5.8e-1 | 6.7e-1 | 4.5e-1 | 7.5e-1 | 3.3e-1 | 1.6e-1 | 1.7e-1 |
| worst \|face gap\| | 4.6e-1 | 1.6e-1 | 4.5e-1 | 2.7e-1 | 9.8e-2 | 8.9e-2 | 4.9e-2 |
| mean \|edge gap\| | 1.0e-2 | 7.7e-3 | 5.2e-3 | 3.9e-3 | 2.7e-3 | 1.8e-3 | 9.2e-4 |
| mean \|face gap\| | 2.0e-2 | 1.5e-2 | 9.7e-3 | 7.4e-3 | 4.9e-3 | 3.7e-3 | 1.8e-3 |

The mean falls like `h`, first order; the worst barely falls at all, because a cell whose
corner the plane misses is wrong by a finite fraction of the cell at any resolution. The
research operator's measured convergence orders -- 2.137819 on the cylinder, 2.063617 and
2.057240 on the glass and weak spheres, 2.468123 percent on the small high-index sphere --
were every one of them taken with the exact forms, so substituting the plane silently
would invalidate all of them while still returning a plausible number.

### Regularity, which is not the same one dimension down

The `m`-dimensional cut of a box is C^(m-1), so `face_cut` is C1 and `edge_cut` is only
C0: piecewise linear in the offset, with a corner where an end of the segment crosses the
plane, and a slope `1/a_1` that grows without bound as the plane turns parallel to the
edge. Measured on a slide of the plane through a cell, largest jump between consecutive
slopes, `dx = 0.1`:

| step h | `edge_cut` | `face_cut` |
| ---: | ---: | ---: |
| 1e-3 | 32.34 | 2.527e-2 |
| 5e-4 | 31.34 | 1.264e-2 |
| 2.5e-4 | 29.34 | 6.319e-3 |

`face_cut`'s halves with the step, which is a Lipschitz derivative; `edge_cut`'s stays at
the size of the slope itself (`max |f'| = 33.34 = 1/a_1`). The exact `Sphere` forms are
**both** C0: the edge's corner is an end of the segment leaving the ball, and the face's
is the cross-section disc appearing at all, where its area `pi (R^2 - v^2)/dx^2` starts
with a finite slope `2 pi R / dx^2 = 157.08` against the zero it leaves. So on this one
axis the exact form is less regular than the plane it replaces -- a trade for C4 to price,
not a defect of either.

### What was measured

| check | worst |
| --- | ---: |
| `edge_cut` against an exact Python clip, 5 families and 2 CSG nodes | 2.2e-16 |
| `face_cut` against Sutherland-Hodgman plus a shoelace area | 3.3e-16 |
| `face_cut` against a quadrature of its chord length | 3.3e-16 |
| `Sphere.exact_edge_fill` against the exact ball chord | 1.1e-16 |
| `Sphere.exact_face_fill` against adaptive integration of clipped disc chords | 3.7e-14 |
| `edge_cut`, `face_cut`: numpy vs jax, jax.jit, torch, scales 1e-6 to 1e+6 | 2.2e-16 |
| the two `Sphere` overrides, same sweep | 2.0e-13 |
| the same forms between coordinate scales, on one backend | 4.4e-16 |
| autodiff against a central difference on the overrides (jax, torch) | 1.31e-6 on gradients of 5.5e+3 |
| the three study helpers against the code they replaced | bit-identical |

The families are 400 random normals, 240 exactly axis-aligned ones, 400 exactly diagonal
ones (face and body diagonals), and cells wholly inside and wholly outside; the CSG rows
are 500 points on a `Union` and on a `Difference`, whose composed `(d, n)` these forms
take exactly as they take a leaf's. `studies/fdtd_subpixel_triplet_sphere.py`'s
`edge_fraction` and `face_fraction` and
`studies/fdtd_subpixel_triplet_cylinder.py`'s `circle_fraction` are now thin wrappers over
the public forms: the circle is the sphere's central plane, so the 2-D point is lifted to
a triple with x exactly zero, and `circle_fraction` gained the `xp` argument that lift
needs. (On 2026-09-26 `circle_fraction` itself moved into `wavebit/fd/_lower/triplet.py`
with the triplet maps that call it, unchanged; the cylinder study imports it from there.)

## Regression mutants of the edge and dual-face cuts

Ten source mutations, each applied alone, run, watched to fail, and restored
(2026-09-24). Two of them did not go red on the first sweep, and both are kept below
with what they changed, because a mutant that survives is a measurement of the gate.

| Mutation | Test failures |
| --- | ---: |
| Both `Sphere` interface overrides deleted | 30 |
| `_first_difference_square`'s `a = 0` piece written as the naive difference | 6 |
| Both degenerate branches dropped | 2 |
| The dual face's two steps not sorted | 2 |
| `_first_difference`'s safe denominator removed | 2 |
| The dual face's clip on `c` dropped | 2 |
| `_decided` reading `d < 0` for `d <= 0` | 1 |
| The `axis` door removed | 1 |
| `circle_fraction`'s `xp` argument dropped | 1 |
| `circle_fraction`'s lift built with `np.stack` again | 1 |

The safe denominator passed all twelve tests at first, because its caller pinned the
argument to the dead end of a `clip` whose transpose is a `select`, and a `select`
discards a NaN cotangent instead of multiplying by it. **That stopped being true one
paragraph later, and the sentence is corrected here rather than left standing**
(2026-09-24, re-run): the clip in question is the edge's, which the next paragraph
deleted, and deleting it took the mask with it. The mutant now drives the loss gradient of
`test_the_interface_cuts_differentiate_where_a_step_is_exactly_zero` to NaN on both JAX and
Torch and reddens both of its cases directly, before the helper check is reached. The
`where(a > 0, ...)` that remains at the end of `_edge_parts` does not mask it: a `where`
hands its unselected branch a zero cotangent, and zero times the infinity that `u**3 / 0`
differentiates to is NaN -- which is the reason the safe denominator is written before the
division rather than after it. The guard is still correct and
still there; what was added is a check of the invariant where it lives rather than a
second argument about the caller. Dropping the **edge's** clip also passed all twelve,
and that mutant was right: over 2000 normals at depths of 1 to 1e12 voxels, clipped and
unclipped agree to 0.0 exactly, because the one-dimensional form returns the constant 1
instead of differencing two large numbers. That clip was deleted; the dual face's, worth
1.9e-3 at the same depth, is the one the table above mutates.

### Two corrections to the forms above

Written: 2026-09-25, after review. No physics moves: `edge_cut` and `face_cut` are
reached by no scene, no plan and no golden -- nothing in `wavebit/` calls either -- and
the checks at the end of this section are what says the rest of the package is where it
was.

**A decided dual face is now exactly 0 or 1.** `face_cut` returned `1 +/- 2.220446e-16`
for a cell the interface does not reach: the clip on `c` pins it to `[0, total]`, but the
divided difference then evaluates `(2 a_1 + 2 a_2 - a_2 - a_2) / (2 a_1)`, which is one in
exact arithmetic and one plus or minus an ulp in three roundings. Over 2000 unit normals
at depths of 1, 10, 1e3, 1e6 and 1e12 voxels, at every one of those depths, 151 read one
ulp above and 151 one ulp below; the worst over a slide of the plane across the cell was
`1.0000000000000002`. A fraction outside `[0, 1]` is not a rounding detail to a rule that
divides by `f` and by `1 - f`, and this package had already refused the same thing once,
in `Sphere.exact_edge_fill`, whose world-coordinate spelling was rejected for returning
`1 + 8.88178e-16`. The two decided cases are now *selected* instead of computed:
`total/2` is the square's own support in the direction `n`, so the square lies wholly
inside the half space exactly when `-d >= total/2` and wholly outside exactly when
`-d <= -total/2`. That is an exact predicate on the same numbers, not a tolerance, and it
is the argument `_decided` already makes one dimension further down. `edge_cut` needed
nothing: its top piece is the literal constant and its middle piece divides `t <= a` by
`a`, so the same sweep measures its worst at exactly 1.0 and its least at exactly 0.0.

The three-dimensional `cut` is **deliberately** not changed, and the measurement is the
reason it is written down rather than quietly fixed. On the same 2000 normals at the same
five depths it reads between `0.99999999999999933` and `1.0000000000000007` for a voxel
the interface does not reach -- **519 of 2000 above one and 1050 of 2000 not exactly 0 or
1**, at every depth, and `1.0000000000000004` worst over the slide. That is three ulps
where the dual face had one, and it is a wider defect than the one this section fixes. But
`cut` is the routed form: every `occupancy` of every scene goes through it, and the same
select there moves coefficients that the frozen archives pin, which is a measurement and a
regeneration and not a minor. **Open, with its number, not fixed here.**

**`Sphere`'s two interface methods now door `dx`.** `edge_cut` and `face_cut` have doored
it since they landed; `Sphere.exact_edge_fill` and `Sphere.exact_face_fill` did not,
although they are public methods reached directly rather than through `occupancy`, which
is where `Sphere.exact_fill` gets its door. Everything inside them is written in units of
`dx`. Measured with the doors removed, three points about a radius-0.25 ball at
`|dx| = 0.05`: a negative `dx` divides the edge's two clips by it and swaps them, so the
two segments that read `1.0` at `+0.05` read `0.0` -- a wholly filled edge reported as
empty. The same input reads exactly the positive answer on the **face**, whose every
quantity is squared. A `dx` of zero reads `nan` on the face and `1.0` on the edge, through
a clip of an infinity. One invalid input, three different behaviours, no refusal anywhere:
the argument for the door rather than for a downstream comparison.

| Mutation | Failures in `tests/test_fd_shapes.py` | The case that went red |
| --- | ---: | --- |
| The decided-face select dropped, leaving the clip on `c` | 1 | `test_neither_interface_cut_ever_leaves_the_unit_interval` |
| `check_sizes` removed from `Sphere.exact_edge_fill` | 1 | `test_the_interface_cuts_refuse_a_typo_by_name` |
| `check_sizes` removed from `Sphere.exact_face_fill` | 1 | `test_the_interface_cuts_refuse_a_typo_by_name` |
| `_first_difference`'s safe denominator removed (re-run of the survivor above) | 2 | `test_the_interface_cuts_differentiate_where_a_step_is_exactly_zero[jax]` and `[torch]` |

Each was applied alone, run against the whole file with three workers, and restored from a
pristine copy; the restored run is 78 passed and 8 skipped with exit 0 every time. The last
row is the survivor above, re-run: it survives no longer, and the two cases fail on the
loss's own `assert np.isfinite(got)` rather than on the helper that was added for it --
which is what the corrected paragraph above says.

Unchanged and checked: every lowered array and plan skeleton of eleven scenes -- the six
stage-1 reference scenes plus two `Sphere` scenes and the three CSG nodes -- is byte
identical to a `git archive` of the parent commit, 281 quantities in all; and the four
frozen archives reproduce 143/288/140/570 arrays bit for bit.

## Regression mutants

Source mutations are run individually and restored before ordinary validation.
Each selected mutant must exit 1 with test failures; collection errors and
internal pytest errors do not count as a killed mutant. The tests cover missing plan
notes, the old phase-index stability bound, non-finite geometry, the old extent
signature, near-zero spectral queries, discarded imaginary eigenvalues, fixed image
coverage, concealed fit scatter, shared component coordinates, swapped mixing rules,
and rounded fills. Tolerances carry the measured values beside their definitions.

The 2026-09-22 run killed all 11 source mutants below with **exit 1**, no collection
or internal errors. All mutations were restored. The selected tests are those named
by `-k` in the corresponding occupancy or study test file.

| Mutation | Test failures |
| --- | ---: |
| Remove the plan's geometry note | 14 (all 12 CSG/backend rows and both traced-plan doors) |
| Use only declared phase indices for stability | 1 |
| Accept non-finite sphere geometry | 6 |
| Restore `min_extent(self)` without the grid argument | 1 |
| Query spectrum near zero and use largest magnitude as denominator | 1 |
| Return only the real part of an eigenvalue | 1 |
| Restore the fixed image window without folding coordinates | 1 |
| Report zero fit scatter | 1 |
| Sample every component at the cell centre | 2 |
| Swap normal and tangential mixing | 2 |
| Round fills to zero or one | 3 |

## Integrated validation

The complete 2026-09-22 run used three pytest workers and two threads per numerical
library, with `PYTHONUTF8=1` and without setting `XLA_FLAGS`.

| Gate | Result | Exit code |
| --- | --- | ---: |
| All `test_fd_*.py` and `test_docs_contract.py`, excluding slow | 1801 passed, 6 skipped | 0 |
| The same files with `--run-slow -m slow` | 65 passed | 0 |
| Frozen stage 0 / 1 / 2 / 3 arrays | 143 / 288 / 140 / 570 bit-identical | 0 each |

The six skips are the existing NumPy automatic-differentiation comparisons: NumPy
supplies their reference values. Collection, XML testcase counts and unique test names
agree at 1807 fast and 65 slow cases. No `INTERNALERROR` occurred. All four reference
archives retain their original SHA-256 hashes; no rebaseline was performed.

## S7: how often a voxel holds three media, or a crease (2026-09-26)

Run before the extruded bodies ship (stage 4, C4), because two populations had once been
conflated: a voxel holding **three or more media** has no answer from a two-phase rule, while a
voxel holding a **creased** two-media interface -- a box corner, a prism edge, a cylinder rim,
inside one body -- is still two phases, and the kernel's fill is exact there. Refusing the
second would refuse every prism edge and every turned box corner.
`studies/fdtd_subpixel_media_census.py` counts both, and flush joints (two structures'
surfaces in one voxel, two media), over every E-component voxel near a surface; its table is
`studies/results/fdtd_subpixel_media_census.log` (5.1 s). Percentages are of the cut voxels;
the bodies are about half a unit across, at 20 / 40 / 80 cells per unit.

| scene | creased | three or more media | flush joint |
| --- | ---: | ---: | ---: |
| sphere | 0 | 0 | 0 |
| cylinder (rims) | 14.3 / 7.2 / 3.6% | 0 | 0 |
| box, axis-aligned | 24.3 / 12.5 / 6.3% | 0 | 0 |
| box, turned 30 degrees | 25.3 / 12.0 / 6.0% | 0 | 0 |
| hexagonal prism | 26.0 / 13.8 / 6.6% | 0 | 0 |
| cylinder pillar flush on a substrate of another medium | 8.7 / 4.6 / 2.4% | 4.4 / 2.3 / 1.2% | 119 / 549 / 2362 voxels |
| sphere dimer, gap 0.5 D | 0 | 0 | 0 |
| dimer of two metals, gap 0.05 D | 0 | 1.15% at 20, then 0 | 0 |

What it decides:

* **Rotation and the creased bodies ship in C4.** Creased voxels are a line population -- each
  doubling of resolution halves their share of a surface -- and in them the fill is exact for
  every body and the diagonal moment exact for the box and the prism. Two prices stay and are
  named, not hidden: the cylinder rim's moment is the tangent plane's (worst 0.40-0.52 in
  `w`, `wavebit/fd/_shapes/_bodies.py`), and every body's off-diagonal cross terms use the
  nearest-surface normal as a stand-in, whose gap the plan reports
  (`wavebit/fd/_lower/constitutive.py`).
* **The three-media refusal is a usable stopping point for C4.** No single body and no
  canonical dimer reaches it. The scenes that do -- a pillar standing flush on a substrate
  spelled as a second body, and a near-touching dimer of two metals -- are refused today, by
  the two-surfaces-in-one-voxel door, before the three-media question is asked. So C10's
  polytope clip is not C4's. The flush joint is the population that matters next: it is how a
  metasurface pillar meets its substrate, it is two media wherever the materials match, and it
  is refused wholesale. How a pillar joins its substrate is a decision for before C7.

## C4: the extruded bodies (2026-09-26)

Stage 4's C4 puts three more bodies on the public door, smoothed exactly as the sphere is --
the same `coefficients.mix`, the same doors, the same `(fill, w)` at every component's own
Yee point:

```python
fd.Box(size=(0.3, 0.2, 0.1), center=(0, 0, 0), angle=30, axis='z')    # turned 30 degrees
fd.Cylinder(radius=0.1, height=0.2, axis='z', center=(0, 0, 0))
fd.Prism(vertices=((0, 0), (0.2, 0), (0.1, 0.17)), height=0.2, axis='z', center=(0, 0, 0))
```

`angle` is in degrees, as `PlaneWave(theta=)` is, and turns a body counter-clockwise about
`+axis` through its centre; `vertices` are `(u, v)` offsets in the right-handed pair across the
axis. A tilted axis is stage 5 and refused by name; a cylinder takes no `angle` (a circle turned
about its own axis is the same circle, and the kernel measured that key's gradient as exactly
zero); a prism's outline is checked -- counter-clockwise, simple, three vertices or more -- at
construction. Every value is static and finite, as the sphere's are; `range=` is C6.

**The rename came first.** The cell-snapped box a monitor or a `TFSF` takes, and that a
structure may still fill whole cells with, is `fd.Region` since the commit before this one; the
four archives reproduced 143 / 288 / 140 / 570 arrays bit for bit after it, because the class
name reaches no stored byte and its report rows keep the `box.*` labels the stage-3 archive
froze. The body box's rows are `cuboid.*`, so a reader can tell asked-and-got from snapped. A
monitor or TFSF given the body `fd.Box` is refused with the name that works.

**What an extrusion answers differently is `uniform_along`.** Along its own axis a body spans a
range of cell layers when its extent covers every voxel of them, a cell beyond each end; across
the axis only an unturned `Box` does, whose faces are planes of the grid's own axes. The
periodic and `Axis1D` doors used to refuse any smoothed shape that reached such a face; they now
read `uniform_along` first, so a pillar that runs the whole period, or the invariant
cross-section an `Axis1D` axis stands for, is carried -- every layer holds the same section, and
no image ring is needed. The fold itself, for a body that crosses a periodic face and changes
across it, is still not written, and that case is still refused.

Gates, in `tests/test_fd_bodies.py`: every component's lowered fill sums to the analytic
volume to 1e-12 for eight bodies (boxes, a turned one, cylinders on two axes, a turned hexagonal
prism, a non-convex L and a lopsided sliver); a turned body equals, voxel for voxel, the prism
whose vertices the test turned by hand; the `uniform_along` truth table; the wrapping-face doors
both ways; the refusals; three backends' coefficient arrays against NumPy's (1e-13). Mutants,
each alone, over that file and the five door and occupancy files beside it:

| Mutation | Fast cases failed |
| --- | ---: |
| support without its cell of margin | 8 (every volume row) |
| bounding box turned clockwise | 1 (the L-shaped row) |
| `fd.Box` hands the kernel degrees | 1 |
| `fd.Prism` hands the kernel degrees | 3 |
| uniform along a straight axis without asking whether it spans | 2 |
| a turned box counted uniform across its axis | 1 |
| spanning with no margin | 1 |
| spanning with half a cell of margin | 0 at this commit, recorded then as equivalent, which it is not: see "The review of C4's bodies" below |
| the periodic door ignoring `uniform_along` | 1 |
| the `Axis1D` door ignoring it | 1 |
| the monitor door's body-box sentence removed | 1 |
| `fd.Prism` without the outline check at construction | 1 |
| a NaN passing as finite | 1 |

**What C4 does not claim.** No body but the sphere has an electromagnetic convergence gate of
its own beyond G5 below -- the unturned box's planar faces. The cylinder rim's
moment is the tangent plane's, and every body's cross terms use the nearest-surface normal (S7
above prices both). CSG of the bodies, the periodic fold, and a body joining a cell-snapped
structure at a shared face are not in it.

### Door 6: a smoothed surface in a voxel a cell-snapped face cuts

Found while planning C4's gate, not asked for by the plan, and the reason it is a door and
not a note: it was silent. A cell-snapped shape gives whole cells, so the three components
whose Yee point sits on one of its faces have a voxel half on each side of it, painted all one
side or all the other -- on its own, the snapped shape's documented half-cell rule. A smoothed
body cutting the same voxel mixes its material into what the painter left, so one of the
voxel's other two media is lost: three media, and a rule of two. Written out by hand from the
geometry, at `dx = 0.05`, for a pillar `fd.Box` of `eps` 12 on an `fd.Slab` substrate of
`eps` 2.25, the tangential `eps` at the joint node:

| pillar base, from the substrate face | lowered | the voxel holds | door 6 |
| --- | ---: | ---: | --- |
| flush | 6.5000 | 7.1250 (8.8%) | refuses |
| 0.3 of a cell above | 3.2000 | 3.8250 (16%) | refuses |
| half a cell sunk in | 12.0000 under the pillar | 12.0000 | refuses: its rim holds three media (24 voxels) |
| half a cell above | 1.0000 | 1.6250 | admits: the body cuts no voxel the face cuts; that gap is the slab's own snapping |
| 1.2 cells above | -- | -- | admits |

The previous composition test put a ball's centre on a slab's face and called the result exact
against an oracle that was the snapped mask; it now places the ball a voxel clear, above and
inside, and the flush scene is one of the refusals. `_snapped_joint` finds the voxels: the
snapped cells' exact share of each (three 1-D overlaps; a side that reaches the domain's wall
is open, not a face) and the body's fill where that share is fractional and the surface is
within reach, both against `_CLIP_ROUNDOFF`. Three mutants -- the door never refusing, the
walls read as faces, a straddling voxel refused whether the body cuts it or not -- each fail
the new door's test alone. A stencil test that had put a snapped box half a cell from its
ball now places it one cell out, still inside the operator's region.

So **a pillar standing on a substrate cannot be said exactly yet**, snapped or smoothed: S7's
flush joint and three-media rim are exactly that scene. How a body joins a substrate is the
decision C7's metasurface needs.

### The review of C4's bodies (2026-09-26)

An adversarial review of the bodies commit found one major defect and four smaller ones, each
reproduced before it was fixed.

* **The medium at a plane read a smoothed body's support as its material.** `_post.medium_at`
  (and the grazing rule's `_layer_media`) decided "this structure fills the cell layer" from
  the shape's support and its uniformity across the plane -- exact for a snapped shape, whose
  support is its filled cells, and wrong for a smoothed one, whose support carries a cell of
  margin and its fractional face cell. C4 made it reachable, by letting an unturned `fd.Box`
  span a periodic cell. A diffraction plane 0.6 of a cell under a smoothed film planned with the
  film's `eps` 4 -- its orders de-embedded in glass that is not there, the +-1 orders counted as
  propagating -- and a plane 1.6 cells under it was refused as "an interface on the plane".
  A smoothed shape is now asked by its own exact fill at all six components' points in the
  layer (`_post.layer_share`): none is absent, all of it across the whole layer is the
  material, anything else is two media. The plane 0.6 of a cell under the film is refused (its
  `Ez` voxels there hold part of the film), the one 1.6 cells under reads air, one inside reads
  the film -- as the `fd.Region` film does, where it can.
* **Half a cell of spanning margin is not equivalent.** Recorded as equivalent above, and it
  is for the fill; but the moment and the cross terms read the *nearest* surface, and a rim
  voxel can sit up to about 0.71 of a cell inside the side, so a cap nearer than that gives the
  face layer the cap's normal. With half a cell, a rod whose caps sat 0.52 of a cell past a
  periodic face was carried and its first layer differed from its middle one by 1.6e-3 in `be`.
  The whole cell stays, and `test_a_rod_that_spans_a_period_lowers_every_layer_alike` now pins
  it (0.55 past is refused; 1.05 past lowers every layer alike to 1e-14).
* **"The plan gives an unturned box no cross term" was not yet true.** It is now:
  `constitutive.cross_free` names the one body whose cross coefficients are zero everywhere,
  `terms` plans it none -- no flux memory, and no refusal of an initial field over it -- and
  the lowering's zero-weight shortcut asks the same function.
* `min_extent` measured a body leaving through one face at its full length, against the
  protocol's "as it reaches into the box"; the `Axis1D` refusal's remedy left out the cell of
  margin the exemption asks; the three-backend test compared `be` but not the cross array,
  where the bodies' own normals enter; and messages, a degenerate outline told to "reverse
  it", and sentences still promising C4 were corrected.

Mutants of the fixes, each alone, over `test_fd_bodies.py`, `test_fd_constitutive.py`,
`test_fd_diffraction.py` and `test_fd_grazing.py` (fast cases):

| Mutation | Fast cases failed | Which |
| --- | ---: | --- |
| a smoothed shape read as filling wherever its support reaches the layer | 1 | the film-and-plane case |
| `cross_free` answering False | 2 | the no-cross-term case, and the constitutive test's box row |
| `min_extent` measuring the whole extent, not what is inside | 1 | the `min_extent` case |
| spanning with half a cell of margin | 1 | the spanning-rod case |

### G5: the Airy slab, spelled as a smoothed box

The stage-1 Airy gate (`tests/test_fd_airy.py`) runs a slab under a Gaussian packet in a
one-cell periodic column and compares `r` and `t` with two closed forms. Spelled as the snapped
`fd.Slab`, its faces sit half a cell low -- node `k0` takes the slab's `eps` over its whole dual
cell -- and the continuum formula meets it only with that half-cell reference-plane correction:
0.005 rad in `arg r` with it, 0.40 without. G5 spells the same slab as the smoothed `fd.Box`,
ten cells across the one-cell period, faces on the same node planes. An unturned box plans no
constitutive term (the review above), so the packet's initial field is not refused over it.

The oracle is new and independent of the painters: `airy_profile` marches the scheme's own
recurrence through any node profile of `eps`, and `box_profile` writes the box's profile by hand
-- each face gives one node the arithmetic mean over its dual cell, since `Ey` is tangential.
Predicted from them before the runs, and measured:

| arm | `|r|` | `arg r` (rad) | `|t|` | `arg t` (rad) |
| --- | ---: | ---: | ---: | ---: |
| slab, with the half-cell correction | 6.1616% | 0.00508 | 0.5970% | 0.00508 |
| slab, without it | -- | 0.40013 | -- | -- |
| **box, no correction** | 6.1931% | **0.00480** | 0.5723% | 0.00480 |
| box, the slab's correction applied (the gate's mutant) | -- | 0.39985 (83x) | -- | -- |

So the box's faces are where they were asked. Against the exact profile the box meets the
discrete gate's own tolerance on all five slabs of the file (worst 4.3e-7 absolute, 3.2e-6
relative) and at faces 0.1, 0.25, 0.3 and 0.7 of a cell off the nodes (1.3e-7), where the
continuum formula is off by up to 0.018 rad and is not the oracle. Its complex `r` converges to
the continuum at second order with no correction: 6.21e-2 / 1.47e-2 / 3.63e-3 at 20 / 40 / 80
cells per wavelength (ratios 4.22, 4.06). With its faces half a cell low the box fills every
node whole or not at all, and the run is the slab's to 7.0e-15 of the probe's peak -- not bit
for bit, because the exact clip reads a voxel that ends on a face as one to within its own
round-off. Three mutants -- the rule mixing a tangential component harmonically, the box handed
to the kernel 0.04 of a cell high, an unturned box planned a cross term -- fail 11, 12 and 12 of
the file's 31 cases.

## S2b: how few cells across a smoothed body may be (2026-09-26)

*The energy certificate this section measures was replaced on 2026-09-27 by the star lift
(the last section); its refusals are history. The door's numbers are the unlifted rule's, which
the star lift leaves unchanged on every `eps <= 12` sphere measured.*

Run before the cells-across door (stage 4, C5). The plan first meant to set that number from
the fill rule's volume error, and that question closed before the study ran: every public body's
fill is its exact clip, and the lowered volume matches the analytic one to 3.2e-15 relative on
all six components -- sphere, box turned 30 degrees, cylinder, hexagonal prism and an L-shaped
prism, 2 to 24 cells across, five sub-cell placements. What is left is electromagnetic: a body a
few cells across is a handful of mixed voxels, and the grid scatters from the effective
polarizability of those voxels, not from the particle's.

`studies/fdtd_subpixel_cells_across.py` measures it two ways; its rows are
`studies/results/fdtd_subpixel_cells_across.log`. The static instrument solves zero-frequency
electrostatics on the Yee node lattice with the production inverse-material operator (the
lowered diagonal plus the constitutive cross action) and reads the dipole moment; the sphere is
judged against Clausius-Mossotti, the other bodies against a second-order Richardson
extrapolation from 12 and 24 cells across, which the 16-cell value meets to 0.95% at worst. (At
one cell across the box had an odd cell count, so the origin was a cell centre and those rows'
`node` and `cell` labels are swapped; the ranges below do not depend on it, and the study now
makes the count even.) The dynamic instrument is five public `fd.run` solves of an
`eps = 12` sphere against Mie theory at 40 cells per wavelength. The energy certificate is held
open in the static runs only, so that the geometry's answer is what is measured.

The sphere, error in the polarizability, the range over five placements:

| cells across | eps 2.25 | eps 4 | eps 12 | eps 12, no cross terms | eps 12, arithmetic mean |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | -28.7 .. +30.1% | -40.3 .. +63.4% | -50.9 .. +149% | -52.0 .. +176% | +157 .. +237% |
| 2 | -0.9 .. +8.9% | +1.1 .. +19.4% | +9.6 .. +43.1% | +21.4 .. +56.0% | +89.9 .. +115% |
| 3 | -0.5 .. +2.1% | -0.1 .. +4.9% | +4.0 .. +15.7% | +16.0 .. +24.5% | +58.7 .. +71.5% |
| 4 | -0.3 .. +1.35% | -0.3 .. +2.9% | +1.2 .. +6.3% | +12.3 .. +17.3% | +42.2 .. +52.9% |
| 6 | 0.0 .. +0.44% | +0.27 .. +1.17% | +1.7 .. +4.0% | +10.0 .. +11.7% | +31.1 .. +35.2% |
| 8 | +0.07 .. +0.25% | +0.26 .. +0.59% | +0.6 .. +1.5% | +7.7 .. +8.2% | +22.5 .. +25.4% |
| 12 | +0.03 .. +0.12% | +0.11 .. +0.33% | +0.42 .. +0.78% | +5.5 .. +5.7% | +15.7 .. +16.9% |
| 24 | +0.00 .. +0.01% | +0.01 .. +0.05% | +0.10 .. +0.19% | +2.9 .. +2.9% | +7.8 .. +8.2% |

The two controls are there to say what the numbers are not. Without its cross terms the rule is
first order (+8% at 8 cells across, +5.6% at 12), and the plain arithmetic mean the design's
4.67x figure described is +190% at one cell and still +8% at 24. At one and two cells across the
error also depends on the field's direction, by up to 40 and 10 points; from three it is under
one.

The other bodies at `eps = 12`, worst over three placements and two field directions:

| cells across | 2 | 3 | 4 | 6 | 8 | 12 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| box turned 30 degrees | 14.9% | 7.4% | 3.6% | 2.5% | 2.0% | 0.97% |
| cylinder | 34.9% | 11.3% | 9.2% | 2.9% | 2.6% | 1.5% |
| hexagonal prism | 18.9% | 8.7% | 7.1% | 2.8% | 1.7% | 0.98% |

And the scattering itself, `eps = 12` sphere, 24 periods, against Mie:

| cells across | centred on a node | centred in a cell |
| ---: | ---: | ---: |
| 2 | +123% | -- |
| 4 | +13.2% | +1.28% |
| 8 | +1.94% | +0.99% |

The static probe predicted about +13% and +2.5% at four cells across and +3% at eight on a node,
so the placement lottery it shows is real and of the size it says; the remaining point or so is
not isolated (the probe leaves out grid dispersion and the magnetic dipole).

What it decides:

* **The cells-across door refuses a smoothed body under 4 cells across its smallest extent and
  warns under 8.** At two cells the answer is a different particle at any contrast (up to 9% at
  `eps = 2.25`, 43% at 12, and +123% scattered). At four every body is within 9.2% at
  `eps = 12`, and the sphere within 2.90% at `eps = 4`; at eight the worst is 2.63%, in line with
  the 3% the Mie
  gates allow a small sphere. A contrast-dependent number is defensible -- glass at three cells
  is within 2.1% -- and was not chosen, so that one number is what a reader has to remember. The
  refusal quotes these production numbers, not the arithmetic mean's.
* **The door reads `min_extent`, not a fill-volume budget.** No public body has a volume error
  to budget; only the CSG tangent-plane fill does, and CSG is not public.
* **The door runs before the energy certificate,** because the certificate already acts as a
  door, badly: at `eps = 12` it refuses the sphere at 2 to 4 of the 5 placements from 2 to 12
  cells across, and none at 24; above `eps` of about 12.5 it refuses at every resolution tried,
  and its advice to refine the grid is wrong there. Up to `eps = 20` the operator it judges was
  positive on every body measured -- the sphere's smallest eigenvalue about `0.8/eps` at 12 and
  `0.6/eps` at 20 -- so those refusals are the certificate's slack; but the C5 review measured
  the cylinder's at `0.031/eps` at 20 and negative at 25 (`-0.33/eps`), a real indefinite
  energy, so above about 20 a refusal is not only slack. A tighter certificate is an open
  decision.
* Not measured: the non-sphere bodies at `eps` 2.25 and 4; whether a non-convex prism's error
  follows its arm width or the hull width `min_extent` reads (2.83x apart for the L-shaped
  outline); CSG bodies; `discretisation='stencil'`.

## C5: what a structure may be at a Symmetry face (2026-09-26)

A `Symmetry` face stands for a scene that is its own mirror image: the run carries the domain
side and reflects the field, so it models every structure's domain part and that part's image.
The plane is the node plane on the wall at a `-` face and the half plane `dx/2` inside the wall
at a `+` face. Before C5 nothing compared a structure with that: a smoothed body crossing a
`-` mirror met door 3's message about material outside the box (untrue at a mirror), an
unturned `fd.Box` crossed silently whatever its placement, and a snapped `fd.Region` was
clipped silently. `checks.check_symmetry` now refuses the two readings that are silently wrong
and accepts the rest -- the symmetric body, and the dimer a mirror exists to halve:

* a structure that **crosses the plane and is not its own mirror image there** -- a smoothed
  body judged by its exact fill against the fill at the mirrored points, to `_CLIP_ROUNDOFF`,
  over a band as deep as it reaches past the plane; a snapped shape by its cells, with the
  edge the mirror makes within the snap's own half cell of the declared one;
* a smoothed body **within half a cell of the plane without crossing it, where the parity
  leaves the plane's components free** (`parity=+1` on a `-` face, `-1` on a `+` face). Its
  plane voxels hold its half and its image's, and read only its own. At the other parity the
  mirror zeroes those components and a flush half is exact, so it is not refused.

A cell-snapped face declared exactly on a `+` plane is refused too (the C5 review): that plane is
the centre of the last cell, so the snap rounds such a face up or down by the cell count's
parity and by round-off -- the plane's cell full, the body joined to its image, or empty, the two
a cell apart; at 13 cells a flush `fd.Region` left it empty and was accepted as exact. To the wall,
or half a cell short of the plane, is unambiguous. A snapped shape that also reaches the far
wall is judged against the image of that wall, and the clipping warning counts what lies past it
(a box `[-1, 1]` across a mirror at 0 with a PEC wall at 0.6 warned 20.0% where 40% is not run).

Door 3 no longer answers at a mirror face, and the constitutive plan's own mirror refusal (a
body's cross terms reaching the plane need a mirror-weighted adjoint) now says that the
unturned `fd.Box` is the one body that may cross a mirror.

`studies/fdtd_symmetry_mirror.py` measures what each refused case does with the door held
open: a PEC cavity with an `eps = 4` bar, the `Symmetry(parity=1)` half box against the full
box holding what was meant, by the fundamental frequency (the bar itself moves it by about
-10.8%). Rows in `studies/results/fdtd_symmetry_mirror.log`:

| case | shift in the resonance | `eps` on the plane, half / full |
| --- | ---: | ---: |
| `-` face, box symmetric about the plane (control) | +1.44e-15 | 4 / 4 |
| `-` face, box declared flush, `[0, 0.2]` | +5.50e-3 | 2.5 / 4 |
| `-` face, box shifted one cell, `[-0.15, 0.25]` | -8.74e-3 | 4 / 4 |
| `-` face, box declared `[-0.1, 0.2]`, against that box | -1.55e-2 | 4 / 4 |
| `-` face, Region symmetric about the plane (control) | +4.08e-3 | 4 / 4 |
| `-` face, Region declared `[-0.1, 0.2]`, against that Region | -9.09e-3 | 4 / 4 |
| `+` face, box symmetric about the plane (control) | +1.99e-15 | 4 / 4 |
| `+` face, box centred on the wall, not the plane | +6.91e-3 | 4 / 4 |
| `+` face, box declared flush, `parity=+1` | -3.43e-15 | 1.6 / 4 |

The last row is the parity rule measured: at `parity=+1` a `+` plane's components are zeroed,
so its plane voxel's `eps` of 1.6 is never used. The Region control is the snap's own half-cell
rule -- node components are painted at a Region's low face and not its high one -- so the
Region's asymmetric crossing costs about twice what snapping does; it is refused for its
geometry, a body a third longer than the one declared. `tests/test_fd_symmetry_door.py` pins
each case and runs the half box against the full box: 6.2e-16 of the peak for the symmetric
bar, 6.9e-2 for its flush half with the door held open (a bar 9.4 cells long since C5's
cells-across door; 6.6e-2 at 7.4). Mutants, each alone: the door removed
fails 9 cases, the parity rule inverted 4, the `+` plane put on the wall 3, the band one layer
deep 2, the snap's half-cell slack removed 1, door 3 answering at mirrors 1. A variant that
read the H points on the plane as well as E survived and was removed: the parity frees both
together and their voxels tile the same slab, so they always see the same material.

**Found on the way, not fixed here: a box face on a voxel boundary.** The half box and the
full box first disagreed by 4.5e-3 of the peak in 300 steps for the symmetric bar with faces at
`x = +-0.2`, which are voxel boundaries of `Ex`. The fills agree; the normal moment does not.
A face lying exactly on a voxel's boundary enters that voxel's moment or not by round-off --
at an `Ex` rim voxel `w` is `(0, 1, 0)` with the face at the boundary and `(1/3, 2/3, 0)` with
it 1e-12 inside -- and the two grids' coordinates differ by 1e-17, so the same geometry lowered
`eps` 2.105 against 2.5 on those voxels. It is a property of the C4 moment, not of the mirror,
and it breaks translation invariance for a box whose faces sit on node or half-node planes --
the commonest placement -- while the fill stays exact. The test's bar sits off every voxel
boundary. The convention a face on a boundary should follow, and whether it matters for C6's
differentiable edge, are open.

**A second tie of the same kind, found by the C5 review, also open.** The turned prism's
nearest-surface proxy normal (`_bodies._Polygonal._transverse_surface`) takes the nearest edge
with a strict `<`, so a Yee point on the section's medial axis -- equidistant from two faces, as
for a square turned 45 degrees centred on a node line -- takes whichever face round-off prefers.
`angle=45` and `angle=405` (vertices 5.6e-17 apart) flipped the proxy normal at 38 points and
differed by 2.45e-3 of the peak in 300 steps, and a mirrored half box and its full box by
7.8e-5, against 1e-15 at 30 degrees. The fill is exact either way; the cross coefficients are
not. A tie rule that does not depend on vertex order (the bisector, say) is the fix to decide
with the face-on-boundary one, before C6.

## C5: a structure past a face of the box (2026-09-26)

The box clips geometry, and before C5 the report did not say so. The audit measured, all
silent: an `fd.Region` of `(0.4, 0.4, 0.2)` at `x = 0.4` in a box 1.0 wide losing 25.0% at a
PEC face (printed only as `<- snapped`), an `fd.Slab` `x=(0.2, 0.9)` losing 57.1%, a smoothed
`fd.Box` at the Region's place and a cylinder out through a face along its axis losing 25%
each with rows claiming asked and got were the same, and a Region half across a periodic seam
losing 50% -- its wrapped half never drawn. What C5 does with each:

* **A structure past a PEC or PMC face warns**, `GeometryClippedWarning`, with the share of
  its declared volume lost and the faces that took it: the length past the wall over the
  declared length, multiplied over the axes -- exact, because what reaches a wall and plans
  has a constant section across it (a snapped box or layer, or a smoothed body uniform across
  the face; door 3 refuses any other). It is a warning because a guide ending on a wall is a
  real scene. Running out through an absorbing face does not warn (it is how a substrate is
  spelled), nor does spanning a periodic axis (a lamellar grating), and a mirror is
  `check_symmetry`'s.
* **A cell-snapped shape across one periodic seam without spanning the period is refused**
  (`checks.check_seams`), because it is clipped where it should have wrapped. Judged on its
  unclipped snapped cells: crossing is `k0 < 0` or `k1 > n`, spanning `k0 <= 0` and
  `k1 >= n`. One wider than the period but offset is refused too -- wrapped it would cover
  every cell, clipped it misses some.
* **The report's rows say what happened.** A smoothed body's row now prints the extent the box
  holds, and a row's note is `<- clipped` where the box's face cut the span, `<- snapped`
  where the cells moved it, or both -- compared to a billionth of a cell, not exactly, because
  the exact comparison printed `<- snapped` for a Region at `-0.2` whose snapped face came back
  `-0.20000000000000004` (a 0.6 box at `dx = 0.05`; `-0.19999999999999996` in a 1.0 one).

`tests/test_fd_clipping.py` pins each with the lengths by hand. Mutants, each alone: the note
removed fails 3 cases, every face treated as a wall 1, the seam door removed 2, an exact span
refused as crossing 1, the exact comparison restored 2, a body's rows claiming asked 1.

## C5: the cells-across door, and where the certificate refuses (2026-09-26)

*The certificate was replaced on 2026-09-27 by the star lift (the last section): the plan no
longer refuses on it, and its table below is history.*

S2b above set the door: a smoothed body under **4** cells across its smallest extent is refused
(`checks.check_cells_across`), 4 to 8 warns (`CoarseMeshWarning`), and the messages quote the
production numbers, not the arithmetic mean's. The door reads `Shape.min_extent` over the axes
a body is bounded along. A **layer** -- a body uniform along two axes, a film or a half space --
is exempt, because the rule is exact for a plane at any thickness (G5); a rod through the
domain is a two-dimensional particle and is judged by its section. It runs in `plan` before the
constitutive energy certificate, so a small high-contrast body hears the right reason.

The certificate's own message advised refining the grid. `studies/fdtd_subpixel_cells_across.py
certificate` swept where it refuses -- `fd.plan` alone, the placements counted that pass, in the
same log:

| eps | sphere (of 5) at 4 / 8 / 16 / 24 / 32 cells across | box turned 30 degrees, cylinder, hexagonal prism (of 3) |
| --- | --- | --- |
| 6 | 5 / 5 / 5 / 5 / 5 | all pass |
| 12 | 2 / 2 / 5 / 5 / 5 | the box fails once at 4, the cylinder once at 32 |
| 12.5 | 2 / 1 / 1 / 0 / 0 | box and cylinder fail at one placement at some resolutions; prism passes |
| 13 | 2 / 1 / 0 / 0 / 0 | box as at 12.5; the cylinder fails one placement at every resolution (two at 24); prism passes |
| 20 | 0 / 0 / 0 / 0 / 0 | box and prism fail everywhere; cylinder passes one placement at 4 and 8 |

From `eps` 12.5 a sphere fails *more* often as the grid is refined, so the advice was wrong
there; the message now gives this table's shape and says that refining helps up to about 12 and
not above it. On the sphere the operator the certificate judges stays positive (smallest
eigenvalue about `0.8/eps` at 12, `0.6/eps` at 20), so a tighter certificate would admit it; on
the cylinder it is `0.031/eps` at 20 and negative from 25 (the C5 review), so no certificate
should admit that one. Whether to build a tighter one is an open decision.

`tests/test_fd_cells_across.py` pins the door, the warning, the rod, the turned box, the layer,
the order against the certificate and the certificate's new sentence. Mutants, each alone:
refusing under 3 fails 1 case, warning under 4 fails 2, films not exempt 1, rods exempt 1, the
door removed 4, the old certificate advice 1. At a `Symmetry` face the body is measured on the
box grown to the image of the domain -- `n` cells past a `-` plane, `n - 1` past a `+` plane,
which sits half a cell inside the wall: measured on the box, the symmetric bar of the mirror
test, 7.4 cells long, read 3.7 and was refused. The C5 review found four more holes, each fixed
and tested: the thresholds compared exactly, so a box declared 4 cells across read
3.9999999999999996 at some centres and was refused (S2b's own 4-cell hexagonal prism too) -- they
carry a relative slack of 1e-9 now; a film flush against a wall or a mirror read as bounded,
because `uniform_along` asks a cell of margin, and was refused at its thickness -- a body
straight along an axis whose declared span reaches both ends (a wall, or a mirror plane) is a
layer's axis now; an unturned box flush against a mirror at the zeroing parity, which
`check_symmetry` accepts as exact, was measured as half the body -- it is measured with its
image now, and any other body declared on one side is told to declare it whole; and the S2b
study itself could no longer run, since the door refuses its small bodies -- it holds the door
open, as it holds the certificate. Mutants of the fixes, each alone: no slack fails 4 cases,
layers by `uniform_along` only 1, the flush half measured as declared 1, mirrors ignored 1. A
`+` extension of a whole `n` cells survives: with layers read from the declared span, the extra
cell is no longer observable, and `n - 1` is kept because it is the image. The refusal says its
numbers come from compact bodies (aspect at most 1.6): a plate, thin along one axis and wide
along two, was not measured, and is judged by its thickness. The `_shapes.Sphere` docstring's
claim that a sub-voxel sphere "is refused (C5)" is now true of the public sphere under four cells
across.

## C5: messages whose reason had changed, and the body contract (2026-09-26)

The audit listed messages that named a spelling that does not exist or a promise that had
lapsed. They now say what is true: the skin-depth refusal no longer suggests "fd.PEC-backed
geometry" (`fd.PEC` is a face, not a material); a self-crossing prism outline is told to be two
prisms, not an `fd.Union` (not public); the PlaneWave and TFSF structure messages name the
smoothed bodies as legal patterned structures; door 4 says study S1 found no order the tensor
form buys; door 5 names C10 as the owner of the three-media case S7 counted rare; the
absorbing-layer refusal names the bodies that can run through a layer; the voxel rule's
accuracy note says the default rule has not passed the stage-4 Mie gate that
`discretisation='stencil'` passed; and the per-cell message in `materials.py` no longer promises
shapes "in stage 4". (The `Scene(shapes=)` and `Scene(structure=)` messages still say the
keyword belongs to stage 4: a gate test requires each to name its stage, and it is a pointer,
not a promise.)

Two doors changed behaviour. **The stage-1 overlap door leaves pairs with a smoothed shape to
door 1**, whose reason is the true one there: it refused a Drude dimer on the diagonal, gap half
a diameter, by support box and advised disjoint slabs. And **the `eps <= 0` guard promised in
`coefficients._stack` is written** as the contract it rests on
(`checks._check_fill_contract`): a body whose fill or normal moment leaves `[0, 1]` by more
than `_CLIP_ROUNDOFF` is refused. Public bodies keep `[-2.6e-12, 1 + 3e-14]` (the audit), so it
fires on a body that breaks the `fd.Shape` contract; one answering 1.5 times its fill against
`eps = 12` would read the harmonic mean `-2.67` on the component its normal crosses.
`tests/test_fd_c5_texts.py` pins the dimer, a smoothed body across a PlaneWave sheet (refused;
a cell clear plans) and the guard; three mutants -- the guard removed, the overlap door seeing
smoothed pairs, the PEC advice restored -- each fail one case.

The C5 review tightened both. The guard refuses a NaN by name (every comparison passed it),
scales its tolerance with the grid's largest coordinate over `dx` (a correct sphere 3e5 from the
origin read fills to `1 + 2.8e-9` and was refused), prints how far the value is out rather than a
range rounded to `[0, 1]`, and runs from `coefficients.build` too, so it speaks before the
energy certificate, which read a contract-breaking turned box's real negative `1/eps` (-0.375)
as an inconclusive bound; every arm (negative fill, NaN, a moment above 1) is tested. Two
remedies named spellings that are refused: a figure-eight's lobes as two prisms meet in voxels
door 5 refuses, and an `fd.Cylinder` or `fd.Prism` along an absorbing face's axis is refused in
the layer by its cross terms until C9. The mixed overlap -- a smoothed body in a snapped Drude
Region -- is tested, and the advice to spell a layer as a smoothed `fd.Box` is limited to
lossless, non-dispersive layers (door 1 refuses the others until C8). Four mutants of these
fixes each fail one case.

## C5: what `fd.check` says about each smoothed structure (2026-09-26)

Before C5 a smoothed body appeared in the report only as three span rows claiming asked and got
were the same. Each now has a block, built from the fills the plan's notes take anyway -- no
fifth occupancy pass (the audit counted four):

```
body       structures[0] Sphere(radius=0.25, center=(0.013, -0.007, 0.021)), smoothed
  support  x 6..19  y 5..18  z 6..19  (2197 cells)
  across   10.00 cells at its narrowest (0.5)
  volume   0.0654498 asked, 0.0654498 in the box; held on Ex Ey Ez to -6.7e-16 of what their voxels cover, at worst
  cut      472 / 468 / 480 voxels (Ex Ey Ez)
  cross    constitutive[E0]: 0.08 MB params + 0.08 MB state
  normals  exact moment (the sphere)
body       structures[1] Box(size=(1.5, 1.5, 0.2), center=(0.0, 0.0, -0.45)), smoothed
  support  x 0..24  y 0..24  z 0..7  (4032 cells)
  across   a layer (uniform along two axes): no feature size to resolve
  volume   0.45 asked, 0.288 in the box; held on Ex Ey Ez to +2.7e-15 of what their voxels cover, at worst
  outside  36.0% past x-, x+, y-, y+
  cut      1152 / 1152 / 0 voxels (Ex Ey Ez)
  cross    none: its cross coefficients are zero (an unturned box)
```

`across` is the number the cells-across door reads; `volume` compares each lowered component's
fill with the body's analytic `volume()` (new on `fd.Sphere`, `fd.Box`, `fd.Cylinder`,
`fd.Prism`) inside that component's own voxels -- a node component's reach half a cell past a
minus face and stop half a cell short of a plus one -- which is where a support without its cell
of margin shows (3.79e-3 on some components and not others); `cut` counts fills strictly inside `_CLIP_ROUNDOFF` of 0 and 1;
`cross` names the constitutive terms with their bytes (shared where supports merge); `normals`
is the nearest-surface proxy's worst gap for that structure (the plan's warning keys it by body
type); and a `mirror` line names each Symmetry plane the scene is reflected in. The C5 review
corrected four of these lines: `volume` was read against the box, so a correct body leaving
through one face showed 2.9% and one across a mirror 12%; `outside` counted a mirror as a face
passed (it is measured against the modelled scene now, a wall's image across a mirror
included); a `discretisation='stencil'` body read "cross none ... (an unturned box)"; and a
merged term printed its full cost in each block without saying it was shared. A body without
`volume()` names the faces it passes without a share (it printed `nan%`). Mutants of these --
held against the box, the cover without its half-cell shift, a mirror counted as a wall, the
stencil and shared lines dropped -- each fail a case. Two design lines
were left out on the audit's evidence: "media per cut cell" can only print 2 in a planned scene
(doors 5 and 6), and a normal trace over cut voxels reads 1 by an invariant already gated.

The params line says `unknown` where the params could not be built on NumPy (a traced
material), not `0.00 MB`, which a threshold would read as small. S8 recommended **no**
params-to-state warning: the ratio moves in steps of 0.5 per full-volume coefficient array, so
a threshold only restates which arrays are full, and a smoothed body costs what a snapped Region
of the same extent costs. None is added.

`tests/test_fd_check_block.py` checks the block against closed forms: `4 pi r^3/3` held on every
component at two placements, the substrate's `2.4 * 2.4 * 0.3`, the cut count from the voxel
boundaries by hand, every body's `volume()`. Mutants, each alone: nothing counted outside,
whole voxels counted as cut, params always known, no terms named, the prism's volume without
its height, the mirror line dropped -- each fails one case. A printed plane position is rounded
to a billionth of a cell (`checks.shown`), because a grid's low corner carries round-off: the
mirror line first printed `x = -5.55112e-17` for a box centred at 0.3, and the Symmetry door's
messages read the same plane.

## The star lift: the smoothed operator positive by construction (2026-09-27)

*This replaces the energy certificate that S2b and the C5 sections above describe; they record
the state at their dates.* The default rule's inverse-material operator on a constitutive region,
`Xi = diag(e) + C` -- the edge diagonal `build` paints and the node cross coefficients `lower`
paints, `c/4` for each pair of edges that share a node -- is the operator Werner, Bauer and Cary
call wc07 (J. Comput. Phys. 247, 2013, Eq. 11). Nothing in the rule keeps it positive definite,
and at a high-contrast surface it is not: the 16^3 test cavity's `eps = 100` sphere reads
`lambda_min * eps = -1.55`, and a seeded run with that operator grows x1.9e7 in 300 steps while
its discrete energy stays flat -- leapfrog conserves the indefinite form exactly, so the energy
cannot see it until the cancelling parts are large enough for round-off to show (about step 160
here, at field norms near 1e9); `tests/test_fd_constitutive.py::test_an_unlifted_eps_100_sphere_grows`
judges the field norms. Until now the plan refused such scenes with a Gershgorin lower bound, which also
refused most positive ones: every sphere above `eps` about 12.5, at every resolution.

**The construction** (`wavebit/fd/_lower/stars.py`). `Xi` is the sum of one 6x6 matrix per node --
its star: the six edges that touch the node, each with half its diagonal, and every coupling
through the node at its full `c/4`. If every star is positive definite, so is `Xi`. A star has a
closed test: minimising its energy over how each axis's field splits between the axis's two
edges leaves `diag(H) + offdiag(c)`, `H_a` the harmonic mean of the two `a`-edges' diagonals (a
lone edge at a region's end reads `2 e_a`). The lift moves both `a`-edges of a star toward the
node's own Kottke tensor (Kottke, Farjadpour and Johnson, PRE 77, 2008) by the least `t` in
`[0, 1]` that puts the star above a floor `THETA * min(1/<eps>, its edges)`. At `t = 1` the test
matrix is that tensor -- positive definite, smallest eigenvalue at least `1/<eps>` -- and the
star's smallest eigenvalue is concave in `t`, so the least `t` always exists. Each edge's
diagonal is the sum of its two halves. It is Werner, Bauer and Cary's Sec. 4 argument and step-6
fallback with a larger piece, made minimal and continuous.

* **It never refuses** a lossless positive scene within float64's reach, for any number of
  bodies; a star with no cross coefficient never moves, so a region with nothing below the floor
  lowers to the same bits as before and `be` is not touched. Float64's reach: a fill's
  round-off (about `-7e-14`) times the contrast breaks the mix itself from about `1e12` -- the
  review of the star measured an edge diagonal of -15 at `1e13` -- which no lift repairs; the
  plan refuses it and says so. Measured on spheres, every contrast up to `1e11` certified,
  refusals began at `1e12` (cell-centred, 40 cells across), and a node-centred sphere 12 cells
  across held to `1e15`. The floor is clamped at zero, which a negative edge would otherwise
  make negative (the test then weaker than positivity).
* **It is continuous**: a star starts to move from `t = 0` as it reaches the floor (a trigger at
  zero with a lift to the floor, the first prototype, jumps).
* **The certificate is checked, not assumed**: every star at its `t` is tested on the numbers the
  kernel will use; `validate` refuses a failure by name and `guard` turns it into `stable = NaN`
  under a transform. The Gershgorin enclosure stays for its upper end, the time-step bound.
* **The test** is the sign of the three `L D L^T` pivots of the test matrix. Leading principal
  minors misjudged 331 pieces at `eps = 1e10` unscaled and, scaled, resolved `t*` only to about
  `1e-8` at a double root, where a determinant touches zero quadratically: a finite-difference
  gradient on a cell-centred sphere scattered by 6%.
* **Derivatives** of `t*` are the implicit function's, exact to second order (below). A
  determinant's derivative was 44% off at the double eigenvalues that a node- or cell-centred
  sphere has on its body diagonals.
* **Values and traces**: on the host only the stars below the floor are bisected; under
  `jax.jit` and the other value-less transforms every star goes through one scan. NumPy, JAX,
  `jax.jit` and Torch agree to `1e-12` relative; gradients agree with central differences to
  `4.1e-9` at three placements.

**The first form, and why the star replaced it.** The first form, in the history just before the
star, used WBC's own pieces, eight 3x3 blocks per node with one edge per axis, whose test binds on the
smaller of an axis's two edges. A design review (four agents) confirmed its proof and found no
scene whose lifted operator was indefinite, and found that the block test is much stronger
than the operator's positivity. `studies/fdtd_subpixel_lift.py` then measured what that costs
(its rows: `studies/results/fdtd_subpixel_lift_blocks.log`): an `eps = 12` sphere, positive
definite, moved 3.1% and 2.0% of its blocks at 8 and 16 cells across (the mean over five
placements, `THETA = 0.02`), and at `eps = 25` (placement `c`, `THETA = 0.05`) the lifted static
error read -0.20, -0.26 and -0.20% at 12, 16 and 24 cells across while the unlifted rule's fell
+0.96, +0.60, +0.36% -- an order-one move of a fixed share of the surface, which is an error of
first order.
At `eps = 12` it hid behind cancellation with the rule's own positive error, and the order at
`eps = 12`, which is all the study's first rule read, went *up*. The star's harmonic mean lies
between an axis's two edges.

**Choosing `THETA`** -- `studies/fdtd_subpixel_lift.py`, the rule written before the runs: the
smallest worst static error of the sphere (S2b's instrument) over `eps` 12, 20, 25, 40 and 100 at
8 and 16 cells across and five placements, among the values that move no star of an `eps <= 6`
body. No star of any of 126 such bodies moved at any value tried. The static error, the range
over five placements:

| eps | cells across | unlifted | `THETA` 0.02 | 0.1 | 0.3 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | 8 | +0.60 .. +1.47% | the same, to the bit | the same | the same |
| 25 | 8 | +0.89 .. +2.33% | +0.72 .. +2.32% | +0.67 .. +2.32% | +0.61 .. +2.35% |
| 25 | 16 | +0.56 .. +1.07% | +0.50 .. +0.98% | +0.48 .. +0.97% | +0.46 .. +0.95% |
| 40 | 16 | +0.92 .. +1.56% | +0.64 .. +1.20% | +0.63 .. +1.20% | +0.64 .. +1.22% |
| 100 | 8 | +1.02 .. +9.22% | +2.31 .. +5.19% | +2.34 .. +5.21% | +2.48 .. +5.28% |
| 100 | 16 | unsolvable at 3 of 5 | +1.02 .. +1.68% | +1.03 .. +1.68% | +1.07 .. +1.71% |

The worst errors were 5.189 / 5.196 / 5.207 / 5.239 / 5.279% at 0.02 / 0.05 / 0.1 / 0.2 / 0.3
(9.217% unlifted), so **`THETA = 0.02`**: accuracy hardly depends on it; the margin does (the
`eps = 100` sphere's smallest eigenvalue times `eps` 0.343 at 0.02, 0.395 at 0.1). Fitted static
orders over 8-24 cells across, mean over three placements: 1.808 at `eps = 12` (unlifted, to the
bit), 1.730 at 25 (1.637 unlifted), 1.770 at 40 (1.506) and 1.767 at 100 (unlifted unsolvable).
Stars moved at `THETA = 0.02`, the mean over placements: none on an `eps <= 12` sphere at 8 and
16 cells across; 6-17% of the stars with a cross coefficient at `eps` 20-25, 16-25% at 40, 24-38%
at 100; 7-14% of the turned box's, the cylinder's and the prism's at 25. Scene by scene the
ranges are wider: 0-23.5% at 20-25 (a node-centred `eps = 20` sphere 8 cells across moves
none), 15.1-27.2% at 40, 22.2-41.1% at 100, 1.2-16.3% for the other bodies.

**The C3 Mie evidence stands.** Over all 37 C3 scenes -- the `eps = 12`, glass and weak spheres
at their three resolutions and three placements, the small high-index control, and the TE
cylinder at 40/56/80 cells and three placements, built as
`studies/fdtd_subpixel_coupled_cylinder.py` builds it (its `CylinderSlice` through the
production coupling, 117-532 stars with a cross coefficient) -- no star moves, and `be`/`bh`
equal `build`'s bit for bit. The orders recorded in `docs/fd-mie-validation.md` are the lifted
rule's.

**The plan says so.** Where the lift moves stars, the plan warns (`SubpixelAccuracyWarning`) with
each term's count and the largest move, and `fd.check` prints a `lift` line under the body.

**Tests and mutants** (`tests/test_fd_star_lift.py`, and two tests each in
`test_fd_constitutive.py` and `test_fd_cells_across.py`): the star sum against `cross_action`'s
dense matrix for E and H; the least move against `eigvalsh` of each explicit star; the lifted
`eps = mu = 100` operator positive definite where the unlifted is not (x100: E -1.549 -> +0.518,
H -1.831 -> +0.391); no star moved on an `eps = 12` sphere or in the C3 spheres; bit identity on
glass; continuity; the node tensor from the definitions; every backend and `jax.jit`; the
gradient against finite differences at three placements and at stars with a ghost edge against
a wall (E and H); the floor never negative (`eps = 1e13` plans); the refusal past float64's reach
by name (`1e16`, and `1e200` where the eigenproblem fails); the certificate's wiring; the plan
note, the check line and its shared marker; one host lift per `fd.run`; no Torch warning. 37
mutants, each killed -- among them the star reading an axis's smaller edge as the blocks did, the
arithmetic mean in place of the harmonic (too weak), the determinant pivots, the implicit
gradient dropped, a lone edge's rate halved, and `THETA = 0.3`.

**Exact second derivatives** (2026-09-27, chosen by the owner after the star landed). The first
form held the null vector and the rate constant: first derivatives exact, `jax.hessian` of a
lowered sum 188% off its second differences at `eps = 40` (the review of the star). Now the star's
root function `F` -- the binding `L D L^T` pivot of its test matrix -- is evaluated once, at the
bisected `t*`, as a Taylor series in `t` on the live inputs, `F + F_1 s + F_2 s^2`, and `t` moves
by the series' reversion, `-g/F_1 - (F_2/F_1^3) g^2` with `g = F - sg F` (zero in value). At
`g = 0` a derivative of order `k` of `c g^2` reads `c`'s derivatives to order `k - 2` only, so
the second term carries `F_2/F_1` as a constant: the first and second derivatives are exact and
the third is not.
Everything differentiated is elementwise, so nested transforms and Torch's double backward work.

* *Against a 50-digit oracle.* `tests/references/gen_star_lift_oracle.py` was written from the
  module's docstring by a reviewer who was not given the design, and shares no arithmetic with
  it: explicit 6x6 stars, bisection to `2**-150`, an eigensolver certificate, central
  differences. On four regions -- random stars with and without ghost edges, a double root along
  directions that keep it, and a star nearly uncoupled along z -- the value, both first and all
  three second directional derivatives agree to 6.7e-16 on JAX, `jax.jit` and Torch.
* *On scenes.* `jax.hessian`, `jax.jit(jax.hessian)` and Torch's double backward of a lowered sum
  in `(eps, mu)`, against central differences of its gradient at `1e-4` of each: 3.8e-8 off a
  node, 4.6e-8 on a node, 2.6e-8 on a cell and 1.1e-8 against either wall, all three backends
  alike, and 100 times more at `1e-3` -- the step squared, so the residual is the differences'.
* *The axes are ordered per star*, the null vector's largest component last. Then the leading
  2x2 minor is `lambda_2 lambda_3 v_last^2`, so the middle pivot is at least `lambda_2 / 3`:
  small only near a double root. In the fixed `x, y, z` order a star nearly uncoupled along one
  axis left that pivot as small as `1e-17` of the scale, and the last pivot a ratio of two
  vanishing numbers; the oracle's nearly uncoupled star is there to catch it.
* *Double roots.* At a double smallest eigenvalue every 2x2 principal minor vanishes, and the
  root function is the middle pivot instead, which for the symmetric `a I + b J` of a body
  diagonal is smooth along directions that keep the symmetry -- a single body's `eps` or `mu`
  does; along others `t*` has one-sided slopes only (+1.107 and -1.601 measured). Which stars are
  double is read from the middle pivot, `|d2| / (|m_1| + |m_2| + |m_3|) < TAU = 1e-11`. Over
  every moved star of 14 scenes the ratio was at most 1.9e-13 (14 stars at the walls and on the
  `eps = 100` cell-centred sphere's diagonals: double in exact arithmetic, split by rounding) or
  at least 1.5e-4, nothing between. The two ways of being wrong are unlike: taking the middle
  pivot for a simple root misreads even the first derivative, 2 to 440 times over (the review,
  on synthetic stars); taking the last pivot for a split double loses about `1.6e-15` over the
  ratio. On the `eps = 100` sphere the Hessian in `eps` agrees with its differences to 1.2e-8 at
  `TAU = 1e-11` and 1.9e-6 at `1e-16`, where its eight split doubles read as simple.
* *What it cost, and the first attempt.* Two Newton steps from the root were exact too, to third
  order, but the second nests the first: `jax.jit` of the 16^3 cavity's Hessian ran 24.8 s a call
  against 14 ms before (compile 86 s against 18 s). The series runs it in 41 ms (compile 23 s),
  and `jax.jit(jax.grad)` at 32^3 in 16 ms against 8 ms. The refute review that measured this
  also found XLA fusing `a * b + c` into one rounding in one copy of an expression and not in
  another, so the value is taken as `t* + (t - sg t)`: `t*` to the bit whatever a compiler fuses.
* *Tests and mutants.* The oracle comparison (four regions, JAX, `jax.jit` and Torch) and the
  scene Hessians (five placements; JAX, and `jax.jit` and Torch off a node) in
  `tests/test_fd_star_lift.py`. 18 mutants, each killed: the `g^2` term dropped or its sign
  flipped, no reversion at all, the fixed axis order, the order read off the largest eigenvector,
  double roots never or always, `TAU` at `1e-16` and at `1e-2`, each term of the harmonic mean's
  series and the edges of its cross term, a lone edge's rate halved, the reciprocal's and the
  lower pivot's series, the series product and quotient, and `c13` dropped from the last pivot.
  The largest-eigenvector order survived the first run: the oracle's nearly uncoupled star then
  had its stiffest direction in the xy plane too, where the wrong vector happens to pick a safe
  axis. Its z axis is now the stiffest, and the mutant fails three cases.

**Open.** Exact third derivatives: `F_2` live and `F_3/F_1^5`'s term constant would give them
at 22 ms against 16 ms for `jax.jit(jax.grad)` at 32^3, measured, and they were not asked for.
Masked (PEC) edges could be treated as absent too -- the operator the kernel runs is the submatrix
without them -- which would move fewer stars at a wall. The lift does not touch the C3 cylinder
arm (voxel order 1.60): that is an accuracy question on a now-positive operator.


### Round-off fills preserve the preceding node tensor (2026-09-28)

The node painter now reads `abs(fill) <= checks._CLIP_ROUNDOFF` as empty,
the same `1e-9` clip round-off already used by the planning doors. It keeps
the preceding tensor, cross coefficients, and arithmetic medium there.
Previously `fill == 0` let a nominally absent later body replace the
preceding body's anisotropic node tensor by an almost isotropic one.

The regression is a cylinder lifted half a cell above a smoothed Box
substrate. Its base is placed `1e-12` of a cell inside the node voxels to
make the round-off deterministic. Replacing the new empty test by the
old exact-zero test fails on all 46 affected nodes: the normal inverse
permittivity is `0.6153846153846` instead of `0.7222222222222`, with maximum
absolute error `0.10683761`. The tangential entry and arithmetic inverse
are checked independently by the same regression.

The existing mirror refusal needed a corrected expectation, not a weaker
door. In its actual magnetic fixture (`eps=2.25`, `mu=1.5`, `dx=.05`,
x-minus mirror at `-.4`), the E node and H common point are distinct:

| Sphere centre x | Plane-layer E fill | Plane-layer H fill range | Maximum H cross before fix | Result after fix |
| --- | --- | --- | --- | --- |
| -.225 | 0 | 0 to .385815030367076 | .005227426745905 | refused |
| -.22 | 0 | 0 to .285815030367076 | .004563667272660 | refused |
| -.20 | 0 | -1.4521886630372944e-15 to 0 | 3.322000209562448e-17 | admitted |
| -.19 | 0 | 0 | 0 | admitted |

These are measurements of HEAD `18139a0` and the corrected painter on the
same fixture. The H point of the first layer is at x=-.375, half a cell
inside the E node at x=-.4; the -.22 refusal is real contact. Only the
-.20 refusal was a round-off artefact. `constitutive.validate`'s strict
cross-coefficient mirror door is unchanged.

Actual source-mutation witnesses: restoring exact-zero emptiness fails
both the substrate regression and the mirror test (pytest exit 1, two
failures); disabling the mirror door fails the reaching case (exit 1,
one failure, `DID NOT RAISE`). Each mutation was restored byte for byte.
The corrected constitutive module's 41 tests pass (exit 0).

Validation of this fix: the full fd/docs tier exited 0 with 2354 passed and
8 skipped (2362 collected); the separate slow tier exited 0 with all 80
tests passed. Neither run had an INTERNALERROR. The four freezer comparisons
exited 0 and reproduced 143/143, 288/288, 140/140, and 570/570 committed
arrays bit for bit; no golden was rewritten. The documentation contract
passed all 65 tests (exit 0).

## The junction: a body standing flush on a face

`fd.Structure(shape, material, junction='split')` chooses how a Cylinder or
unturned Box joins an axis-aligned face of a smoothed, unturned Box. The
choice belongs to the standing body, including one hanging below a face:

```python
glass = fd.Structure(fd.Box(size=(10., 10., 1.), center=(0., 0., -.5)), 2.25)
pillar = fd.Structure(fd.Cylinder(radius=.15, height=.4, center=(0., 0., .2)), 4.)
# Use these as the scene's structures. The pillar defaults to junction='split'.
```

The options are:

| Value | Interface rule | Measured limitation |
| --- | --- | --- |
| `split` (default) | Layer the substrate below the pillar/background mixture | Second order on the flat-film anchor; unresolved rim order on one pillar scene |
| `staircase` | Use the medium at each electric sampling point | First-order junction error and jumps when a point changes owner |
| `refuse` | Retain the shared-surface refusal | No numerical approximation is selected |

The FDTDX-style scene-order blend is not offered. Its junction term was
first order in the film study and depends on structure listing order.

### Scope

The standing body is exactly `fd.Cylinder` or an unturned `fd.Box`. Its
base must be flush, its straight walls normal to the supporting face, and
its rim clear of that Box's own edges. The same construction handles x,
y, or z faces, either orientation, and laterally uniform film stacks.
Every written voxel must contain only that face and the standing body's
walls: the substrate's other face and the standing body's far end must
be outside it. Two junction write sets cannot overlap.

Spheres, turned Boxes, Prisms, nonflush bases, and magnetic, anisotropic,
conductive or dispersive junction media remain refused. A named `mu`,
even `mu=1`, is the existing API's structural magnetic flag and is refused
here. A cell-snapped `fd.Slab` or `fd.Region` substrate also stays refused;
spell this substrate as a smoothed, unturned `fd.Box`.

An embedding layer can be listed before both pair members, covering the
junction voxels whole; it supplies the ground material. A partial earlier
surface would supply a tensor rather than the assumed isotropic ground and
is refused. Any unrelated later painter reaching a written voxel is refused.
The pair's own listing order does not change its junction values.

Non-default choices on shapes that cannot stand on a face are refused at
construction. A non-default choice on a body standing on no face in its
scene is refused at planning; this catches setting it on the substrate.
The existing single-structure restriction for `discretisation='stencil'`
also applies; this change does not extend the stencil's scope.

### Values and write set

Let `alpha` be the substrate fraction, `f` the standing body's transverse
fraction, and `es`, `ep`, `eg` the substrate, pillar and ground permittivities.
Write `a = f ep + (1-f) eg`, `eta = f/ep + (1-f)/eg`,
`A = (1-alpha) a + alpha es` and `B = (1-alpha)/eta + alpha es`.
The inverse entry normal to the face is always
`(1-alpha)/a + alpha/es`. For a cylinder's wall direction `g`, the transverse
tensor is `(I-P)/A + P/B`, with `P = g g^T` where the wall is active.
For an unturned Box its diagonal wall moment gives the corresponding
diagonal two-phase mixture, layered with the substrate.

Edges use the wall-only tangent-plane gate (Cylinder) or exact polygon edge
moment (Box). Common nodes use the nearest transverse wall direction, with
the Box cross entries kept zero, matching the existing node rule. In
particular, a Box sharing another body's constitutive region needs the node
rule too: using its exact edge moment at those nodes missed the existing
zero-substrate limit by `0.1005220005782` at four rim corners in the review
fixture (`dx=.05`, Box size `.28,.32,.4`, centre `.017,-.013,.2`). The corrected
limit agrees to `1.67e-16` there.

The primary write set is the existing shared-surface door's own marking:
both exact fills strictly between the clip round-off and one minus that
round-off. A base-side extension covers rim points whose old single-body
rule reads the glued base instead of the wall. It includes exact ties where
a Box base lands on a voxel face, respecting the original body's cap or
polygon-side inclusion convention. The draft's probe found errors up to
44.8% without this extension on the D' Box fixture. Round-off-empty nodes
need no extra write set: the preceding node-fill fix preserves their
substrate tensor.

The common-node tensor is supplied with arithmetic inverse floor `1/A`.
Its smallest eigenvalue is at least that floor, which is the condition the
existing star lift needs. The lift, its concrete plan certificate, and the
runtime guard continue to see the same coefficients used by the step.

### Static plans and differentiation

`Plan.junctions` stores geometry, masks, owner selection and the rule. Empty
records are omitted from skeleton encoding, both at the top level and in
source-column plans, preserving old plans byte for byte. The compiled executor
strips the digested mask arrays from its cache
key, as it already does for prepared stencil arrays.

Geometry and rule signatures are checked by `fd.lower` in both directions.
The signature includes all scene geometry, because moving an unrelated
third painter can invalidate the ground. Material values remain live backend
arrays; NumPy, JAX and Torch use the same expressions. Public geometry is
still static, while the internal value function carries geometry derivatives
for the future range-aware interface. A plan's discrete membership is fixed;
changing geometry requires replanning.

### Measurement provenance

The pre-existing study is `studies/fdtd_subpixel_junction.py`, arm `2` for
split and `1a` for staircase. Its retained logs use provenance
`cc2ccf8761b414ba`; the anchor report is `anchor|1790518977` and the pillar
report `report|1790530851`. The 16 film cases measured split orders
1.999-2.035 and staircase junction orders 0.980-1.024. Split was consistent
with RCWA on A, B, C', D', T' and R; its total error matched that of each
junction-free control. The rim order on B (eps 5.8) remains unresolved.
These measurements are not an error bound for an arbitrary scene.

The production tests compare public solves with copied, provenance-labelled
study rows and the exact film, and explicitly include node-plane and
half-plane ties. The corresponding production measurements follow below.

### Production review controls

The first full integration tier reported 2441 passed, 8 skipped and 9
failed out of 2458 collected, with no worker error. Three failures came
from a test helper that still included empty junction records; six frozen
skeleton checks exposed the same omission rule missing in nested source
plans. Applying the empty-record omission recursively restores the original
six skeletons. All thirteen focused checks then passed, and removing the
nested omission again fails the pre-existing plane-wave golden check.
The failed full log is retained in `fdtd_junction_review_controls/`.
No golden or numerical acceptance threshold was changed.

The exact-tie height sweep initially found a Box jump of
`0.0213585371974` at half a cell, against a median step of
`0.000344296183`; it failed the study's unchanged factor-50 gate. The
Box's exact moment counted its base cap, but a separately reconstructed
face distance rounded just above `dx/2` and excluded the same point from
the base-side write set. The exact moment now decides cap inclusion, and
the distance guard (which excludes the far cap) uses the already existing
flush-plane tolerance. Both shapes pass sweeps through zero and half-cell
ties; no acceptance threshold was fitted to the failure.

The draft's oblique-film pins were obsolete. The immutable report
`anchor|1790518977`, provenance `cc2ccf8761b414ba`, case FA / 45 degrees /
zeta .25, actually records errors `0.08316041684012954`,
`0.02005072445163558`, and `0.004970824005669929` at 20/40/80. Public
production reproduces these to the original relative tolerance `1e-8`.
The obsolete draft values differed by up to `3.9441e-7`; no research log,
reference computation, or golden was rewritten.

The high-contrast floor assertion applies to the junction's node write
set, as designed. An initial assertion over the entire constitutive term
also tested unchanged single-body nodes, where the eigensolver's largest
relative deficit was `1.5233e-13` at eleven nodes; the junction's deficit
was `2.22e-16`. Restricting that assertion to its intended write set keeps
the original `1e-13` bound. The complete operator is still advanced for
300 seeded steps: 101 of 294 stars moved, maximum lift fraction
`0.7045792972`, and the final/initial field-norm ratio was `0.9740758014`.

### Exact node-plane ties against RCWA

`studies/fdtd_junction_production.py` runs the public solver without the
study's lowering patches. At zeta zero, C' (Cylinder) and D' (Box), 40/80
cells per wavelength, both s and p: all eight solves were finite, their
monitor decay ratios were below `1e-5`, and the coupled C' solves certified
stability. All 24 complex reflection/transmission comparisons at wavelengths
.95, 1.0 and 1.05 passed the study's unchanged Richardson-segment rule
(orders 1.4 to 2, margin `3*u_ref + .1*|fine-coarse|`).

| Scene | RCWA uncertainty | Largest distance from the segment | Smallest allowed margin |
| --- | --- | --- | --- |
| C' | .0006994668797492 | .0004896032522472 | .0024089428278894 |
| D' | .0006507229325448 | .0002476898044381 | .0022404779973807 |

The complete amplitudes, decay ratios, individual comparisons, and source
SHA-256 (`2b3760f328919ca243d195da26de6261229648a7db7f5dc17141b3b631cfb549`)
are retained in `studies/results/fdtd_junction_production_ties.log`.
That source hash precedes the empty nested-plan serialization fix, which
does not change the numerical expressions. The final release-gate manifest
records the source hash used for the complete pytest and freezer checks.
This closes the draft's unmeasured node-plane tie; it does not resolve the
previously reported rim order on scene B or extend the supported geometry.

### Source-mutation witnesses

All 24 new test functions have actual source-mutation witnesses. The runner
`studies/fdtd_junction_mutations.py` writes each pytest failure log and an
exit-code/source-hash manifest under
`studies/results/fdtd_junction_mutations/`, restoring source bytes in a
`finally` block and checking the restoration. The witnesses cover the
public choices, scope refusals, independent laminate, listing order,
staircase ownership, stale plans, backend agreement, material and internal
radius derivatives, compiled cache, reporting, the node floor, exact-tie
continuity, forty coupled steps, film order and archived public solves.
There are 25 witnesses in total, including the nested-plan mutation checked
by the pre-existing frozen plane-wave test.

One negative control survived: swapping alpha with 1-alpha did not fail
the loose `.035` A-scene RCWA sanity gate proposed by the draft. Its passing
log is retained under `fdtd_junction_review_controls/`; the gate was not
tightened. A reciprocal-conversion error does fail that sanity test. Small
junction formula errors are tested by the independent `1e-13` laminate
oracle and `1e-10` archived complex-amplitude comparisons, not by treating
this loose sanity gate as a sensitivity certificate.

### Release validation

After the nested-plan fix, the full fd tier collected 2458 tests and exited 0:
2450 passed, 8 skipped, including all 96 new junction cases. The separate slow
tier collected 80 tests and exited 0 with all 80 passed. Neither run reported
an INTERNALERROR or a worker crash. The four freezers exited 0 and reproduced
143/143, 288/288, 140/140 and 570/570 committed arrays bit for bit, without
rewriting a golden. Seven additional no-junction scenes retained all 42
coefficient arrays and report/skeleton byte strings from the node-fill-fix
baseline `fe3ff20c`.

The source SHA-256 and collected/reported counts are recorded in
`studies/results/fdtd_junction_validation.json`. Successful full/slow logs and
the four freezer outputs are retained alongside the failed review controls.
