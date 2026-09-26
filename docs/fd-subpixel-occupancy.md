# Smoothed occupancy and the tilted-interface measurement

Written: 2026-09-22.

Targets: `wavebit/fd/_lower/occupancy.py`, `wavebit/fd/_lower/coefficients.py`,
`wavebit/fd/_lower/checks.py`, `wavebit/fd/geometry.py`,
`wavebit/fd/_shapes/_cut.py`, `wavebit/fd/_shapes/_bodies.py`,
`studies/fdtd_subpixel_tilted_tensor.py`, `tests/test_fd_occupancy.py`,
`tests/test_fd_occupancy_integration.py`, `tests/test_fd_subpixel_study.py`,
`tests/test_fd_shapes.py`.

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
bound remains separate work. Scenes without smoothed bodies retain their previous bound.

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
its own; the planar reference-plane gate for the box is the next commit. The cylinder rim's
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
