# Known limits of the normal field

Written: 2026-09-03. Measured on CPU, NumPy backend, float64, unless a row says otherwise.

This file collects the places where a normal-vector field this solver builds is **knowingly not
the field the geometry implies**. Each entry says what is wrong, how far wrong, how to tell
whether a given scene is affected, and what to do instead. It exists so that a number that moves
for one of these reasons can be recognised as such rather than chased.

The normal field is a **geometry** quantity: it says which direction is perpendicular to the
nearest material interface at each sample point, and the factorization uses it to decide how much
of Li's inverse rule to apply along which direction. It never enters the permittivity
coefficients. That is what makes every defect in this file quiet — `R + T` conserves, nothing
raises, and the only symptom is convergence that looks like a property of the modeling path.
See `parametric-modeling.md` and `nvf-normal-methods.md` for the constructions themselves.
§9 is the case table: every scene the piece field was designed on, lettered A to L, with its
status as of 2026-09-06 and where it is measured.


## 1. Polygon internal and periodic seams — FIXED (2026-09-07)

The resolved polygon route now extracts **actual interface intervals** before building its
nearest-boundary-point field. Previously `conv_mats` flattened every region's loops and lost
the material labels. A clockwise hole and the inclusion filling it were both distance
candidates even when their materials were equal. A face against its periodic image survived
for the same reason. Removing either seam only in a band around it would still leave the
wrong nearest boundary in the cell interior.

`interfaces.plan_interfaces` instead partitions each edge at the endpoints of every
collinear overlap, including translated edges. It sums oriented coverage **per region** before
assigning the left and right material. This also cancels zero-area out-and-back spurs in a
clipper contour: an opposed spur must not erase a third, genuine exterior edge. The frozen plan holds source-edge and endpoint
references, neighbor-region indices, the reduced-basis image offsets, and bounded image groups.
`replay_interfaces` recomputes the interval endpoints from traced vertices and the traced
lattice, then compares the values on the two sides. Equal values exclude the entire interval
from the distance minimum at **every** sample. Unequal values retain it. Uncovered intervals
compare against the background. The comparison includes the imaginary part of a lossy value.
The permeability pass uses the same geometry record with `mu_value` in the material slots;
`mu=None` consequently means 1, rather than a different physical permeability.

This is not a whole-edge mask. On the 700-high, 500-wide box in a cell with
`a1=(700,0), a2=(200,700)`, each horizontal face loses only its 300-unit shared run and keeps
its exposed 200-unit end. Multiple covering images partition the same edge before equality
is evaluated. An empty interface set returns `(1,0)`; a closed periodic set of parallel
interfaces has its exact line normal, including at the artificial segment joins of a grating.

The periodic distance search is bounded by the edges' fractional extents, rather than an
assumption that 3x3 images suffice for arbitrary segments. A second, conservative distance
bound drops impossible **segment/image combinations** so a long edge does not multiply every
small curved facet's cost. Sample folding, frozen basis-reduction replay, and the JAX
`scan`/`checkpoint` body remain in use.

There was a second representation dependence: one loop used centered Fourier quadrature,
while several loops used the original grid. Even after seam removal, an off-center inclusion
therefore changed the matrices merely by being painted before or after its host. The resolved
route now centers the **unwrapped painted union's area centroid** and applies the shift
phase for every partition count. One canonical, oriented representative of each unwrapped
exterior interval contributes to that centroid; internal cuts and spurs cancel structurally
before the floating-point sum. Periodic abutting faces still belong to this unwrapped area
integral even though they may be absent from the normal field. This is a common quadrature convention, not a change to the
permittivity transform or a claim that off-interface extensions are unique.

**Regression evidence.** `tests/test_polygon_interfaces.py` covers nested and partial overlap,
a hand-written union contour over the whole cell, different and lossy materials, x/y exchange,
rotation with the cell, partial periodic contact on positive and negative shear, long segments
against a much larger image set, multi-image face coverage, zero-area spurs, three-material
contacts, empty interfaces, the epsilon/mu split, frozen JIT and
geometry/material derivatives, and changed-contact staleness. It also disables the live mask
as a negative control. The original 400-square / rotated 200-square reproducer on a 700-square
cell retains its raw-loop control: the old `nnT` differs by 0.5 on more than 17% of the 128x128
samples; the actual-interface field matches the outer contour throughout the cell. Before the
fix, the two initial matrix regressions failed (maximum nested-order matrix difference
0.02271616, and cross-component magnitude 0.01149332 in the spanning case).

A 350-wide grating with an internal same-material cutter is additionally checked against
integrated raster Li factorization at orders 1, 3, and 5. At each order, `(n,0)` and `(n,n)`
transmission agree to the test's 2e-11 absolute tolerance, as does the Li result. This is an
exact 1D invariant and an order-for-order check, not a convergence claim for resonant 2D
geometries. The prior section's 1e-5 transverse-order violations described the old field.

The reduced-basis regression now checks its two-bar layer against a separately written
500x300 union contour instead of pinning the old internal-seam field's transmission.
The even respelling remains bit-identical. Odd respellings use a different Fourier quadrature
grid: the k=0 versus k=1 gap is 1.11589e-3 at grid 64 and 9.19375e-5 at grid 128, more than
an order smaller (1.51224e-5 at grid 256). Each spelling is also checked against its own independent union contour to
1e-12. The former 2e-4 allowance was measured on the defective field, not a basis-invariance
bound. This is quadrature convergence at a fixed RCWA order set.

**Still outside this fix.** Oversized/interpenetrating periodic shapes remain subject to the
existing geometry refusals: extracting normals does not implement a periodic painting union
for the material Fourier transform. Corners and medial-axis ties need not have a unique normal
or derivative. Frozen topology must be rechecked eagerly when geometry changes; JIT replay
assumes that topology still fits. Resonant truncation convergence and contact cusp/ramp
behavior in the parametric builder remain open. The separate `solve_field` string pytree
issue is fixed in §8d.


## 2. A face the fold cannot merge is refused, not built — FIXED (2026-09-06)

**Status: fixed.** Every scene below builds now, and the refusal is gone. The parametric field
is built from **pieces** — a box is its four faces, each a segment evaluated at every periodic
instance that can be its nearest, and a plan (`parametric.plan_gamma`, `GammaPlan`) records
which face lies on which image's face, opposed, over what run. Over the run the two pieces'
distances are bitwise one number and their normals cancel exactly; beyond it the neighbouring
image's face is simply one more candidate, which is exactly the face "no closed form over one
box" could reach. Measured against the exact interface oracle (`overlap.interfaces_of`) over a
band of 1.5 pixels, N = 512: the sheared cell at `lx = 500` reads **0.16 deg** mean with 0.11%
of samples past 45 deg, the tilted box at `lx = 640` **0.18 deg** / 0.15%, where the one-box
field read 90 deg along the whole run (1.4 to 17 deg averaged over the band).

Through the solve the picture is coarser, and it is worth saying exactly how coarse. The same
tilted box, the same operating point as the table below (its "shape path" row is reproduced
digit for digit by the fold's field injected into today's solve), against the exact-interface
field of the oracle injected in place of the built one -- a Voronoi extension, no staircase,
the same permittivity transform:

| truncation order | 6 | 9 | 12 |
| --- | ---: | ---: | ---: |
| pieces | 0.845760 | 0.850379 | 0.857004 |
| exact-interface field | 0.829832 | 0.838456 | 0.846136 |
| gap | 1.6e-2 | 1.2e-2 | 1.1e-2 |
| the fold's gap (same instrument) | 3.1e-2 | 3.0e-2 | 2.6e-2 |
| control: the same box at `lx = 500`, no contact | 8.8e-3 | 9.0e-4 | 1.3e-3 |

Half the fold's gap and falling, but not gone by order 12 -- and the control says why the
instrument cannot say more here: with no contact at all, where the fold was right on Gamma too,
the two fields still sit 1e-3 apart at orders 9-12, and pairwise the three extensions available
(the blend, the Voronoi field, a `'contour'` raster) disagree at 1e-2 on this box. A box of this
size at `eps = 12.25` is that sensitive to what a field does *off* the interface at these
orders. **The solve-level decay gate is not an instrument for B and L at this order, and the
field-level gates stand in for it.** Injecting an exact Voronoi field of the interface set into
the same shape-path solve at N = 512, the piece field sits +1.2e-2 / -8.1e-4 / +4.0e-3 / +2.1e-3
/ +2.1e-3 from it at orders 4 / 6 / 8 / 10 / 12 on the sheared box, +4.5e-3 / +1.43e-2 /
+1.04e-2 / +1.11e-2 / +9.2e-3 on the tilted one and +1.6e-2 / +7.2e-4 / -5.5e-4 / +2.2e-3 /
+1.4e-3 on the L box -- and so does every other exact-on-Gamma extension: the blend at exponent
2 or 8 plateaus at the same levels (tilted at order 12: +1.14e-2 / +7.2e-3), an image-aware
`'contour'` raster at N = 1024 sits +5.7e-3 / +1.0e-2 / +8.2e-3 / +6.0e-3 from the pieces at
orders 6..12 with its own staircase, while the one positive control, the one-box field that is
90 deg wrong along the run, reads -3.2e-2 / -3.4e-2 / -2.6e-2 / -2.5e-2. A box this large at
`eps = 12.25` is that sensitive to what a field does *off* Gamma below order 12: the piece and
Voronoi fields differ by more than 10 deg on a third to a half of the cell, mostly farther than
60 units from any interface, and the tilted box's near-Gamma disagreement is *smaller* than the
350 box's (0.48 against 0.72 deg mean within 5 units), whose own gap decays to 2e-4 by order 8.
So the decay of `sum(T)` cannot separate a wrong interface from a different extension on these
scenes under the order cap, and the gates that answer are on the field: exact on Gamma at
explicit points of every live segment (0.0 to 1.2e-6 deg, the oracle's own rounding) and the
interior band error, both in `tests/test_nvf_self_coincident_faces.py`. A solve-level gate,
if one is wanted, needs a scene the raster resolves exactly -- an axis-aligned box on a
rectangular cell with its edges on pixels -- or a positive-and-negative control pair as above.
The rest of this section is the record of what the fold could not do and why, kept because the
tables are what the fix was measured against; §8 says what moved when it landed.

The fold (§4) merged a face family when a lattice vector ran **along one of the shape's own
axes**. That is not the condition for a family to merge, only the condition the fold could act
on, and the true condition is strictly weaker.

A face is a *segment*. Its images under a lattice vector `t` land on a parallel line offset by
`t . v` (perpendicular) and slid along itself by `t . u` (tangential). Two opposite families
coincide over positive measure when some `t` has `|t . v|` equal to the face separation **and**
`|t . u|` short enough that the segments still overlap. The fold requires `t . u == 0`, which is
that condition's zero-shift special case. `overlapping_pairs` cannot see the rest: the boxes only
*touch*, so it reports a gap of zero and passes them — the right answer to the question it asks
(does the layer need a periodic image to be *built*) and the wrong one to this one.

Two families of scenes reach it, and both were accepted and silently wrong:

**A box on a sheared cell.** `a1 = (700, 0)`, `a2 = (200, 700)`, rectangle `ly = 700`, angle 0.
`a2 . v` is 700, which is `ly`, so the top face's images sit on the bottom face's line; `a2 . u` is
200, so they overlap whenever `lx > 200`.

**A rotated box on any cell, including a square one.** On a plain 700 x 700 cell, a rectangle at
`atan(1/2)` = 26.57 degrees with `ly = 700 sin(theta) = 313.0495` has `a1 . v = ly` and
`a1 . u = 626.10`, so any `lx` above that overlaps. Spell `ly` to full precision —
`700*np.sin(np.arctan2(1., 2.))`, not a rounded 313.05 — or the equality never fires and the case
looks inert. "Square cell" is not a safe category.

The boundary-band error against an exact nearest-interface oracle tracks the coincidence and is
zero at the onset in both families:

| shear, `ly = 700` | `lx` 100 | 200 | 300 | 500 | 690 |
| --- | ---: | ---: | ---: | ---: | ---: |
| mean angle, deg | 0.197 | 0.266 | 1.435 | 4.551 | 17.438 |

| tilted, square cell | `lx` 500 | 626 | 640 | 660 | 690 |
| --- | ---: | ---: | ---: | ---: | ---: |
| coincidence | 0 | 0 | 13.9 | 33.9 | 63.9 |
| mean angle, deg | 0.143 | 0.148 | 1.362 | 3.019 | 5.777 |

**Historical finite-order comparison of the earlier builder.** `sum(T)` for the tilted box at
`lx = 640` — 13.9 of a 640-long face, 2% of one face — against two independent instruments that
agree with each other to 1.2e-4: an exact nearest-interface field injected into the same
shape-path solve, and a raster of the 3x3 image union on the same lattice.

| truncation order | 6 | 9 | 12 | 15 | 18 |
| --- | ---: | ---: | ---: | ---: | ---: |
| shape path | 0.801261 | 0.811221 | 0.823352 | 0.832240 | 0.838137 |
| exact-interface field | 0.832096 | 0.841068 | 0.849004 | 0.853464 | 0.856906 |
| image-aware raster | 0.832010 | 0.840954 | 0.848972 | 0.853461 | 0.856791 |
| error | 3.1e-2 | 3.0e-2 | 2.6e-2 | 2.1e-2 | 1.9e-2 |

The gap falls from 3e-2 to 1.9e-2 over these orders. That finite sequence does not prove
non-convergence or separate truncation effects from the choice of off-interface extension.
The independently measured wrong direction on the interface establishes the earlier builder's
geometric defect. Energy is conserved and the material transform is shared, so this comparison
isolates the effect of the different normal fields at the listed finite orders.

**Why no closed form over one box reaches it.** Along the dead run the material continues into the
neighbouring image, so the nearest interface there belongs to *that image's* face — which a field
built from this box alone never looks at. Masking the run therefore removes the wrong normal
without supplying the right one. Measured, with the mask applied to the exact dead intervals:

| | shear `lx=300` | shear `lx=500` | shear `lx=690` | tilted `lx=690` |
| --- | ---: | ---: | ---: | ---: |
| unmasked | 1.435 | 4.551 | 17.438 | 5.777 |
| masked | 1.552 | 4.580 | 17.543 | 5.838 |

Face-level aliveness is the wrong knob rather than too coarse a knob, and there is a separate
measurement that says so outright: at `lx = 500` on the sheared cell the raster truth (0.8394)
sits **outside** the span of every aliveness answer — all-alive 0.8788, all-dead 0.8520,
geometrically-correct partial gate 0.8837. A better gate is not the fix; a working one would have
to blend face *segments* over several images, which is a different builder.

So the geometry *was* refused, by `overlap.partial_self_coincidences` from `check_layer_shapes`,
naming how much of the face was covered. That is the "different builder" the paragraph above
asks for: `overlap.face_pairs` is the sweep that finds every run (the runs and their union per
face are what the refusal computed), and `parametric._piece_field` blends the face *segments*
over their images. The refusal and its sweep were deleted with it; the interpenetration refusal
(larger than the period, overlap through an edge) is unchanged and still asked first.

### 2a. The union of images, not any single one

A per-image test gets this wrong in the direction of over-refusing, and the case is not exotic —
**Li's checkerboard cell reaches it with an ordinary rectangle.** On
`a1 = (1.25, 1.25), a2 = (1.25, -1.25)`, a 2.5 x 1.25 bar has each horizontal face covered by
`a1` and `-a2` taking **half each**. No single lattice vector covers the face, so a per-vector
test reports a partial coincidence and refuses; the face is in fact covered end to end and there
is no interface along it at all.

That shape tiles the plane — cell area `|det|` = 3.125, shape area 2.5 x 1.25 = 3.125, and the
3x3 image union covers the cell with no hole. And that is not a coincidence of the example: full
union coverage of a face pair forces the covering images to abut along that axis at exactly the
shape's own width, because anything closer interpenetrates and is refused upstream. **Full
coverage therefore implies the shape tiles**, its permittivity is uniform, and every convolution
matrix is `eps * I` whatever the normal field says. Refusing it would refuse an exact answer.

That the field is inert there was checked rather than argued, on the other session's tree:

| | `sum(T)` |
| --- | ---: |
| the tiling bar, 2.5 x 1.25 | 0.858800690221 |
| a uniform slab of `eps = 4` | 0.858800690221 |
| the tiling bar with its normal field replaced by the constant `(0.6, 0.8)` | 0.858800690221 |

An arbitrary constant field gives the same answer to the last digit, which is what "every conv
matrix is `eps * I`" means in practice.

**And it exposed a floor under the candidate-set argument.** `_axis_lattice_periods` (deleted
2026-09-06 with the fold, §8b) asked which lattice vectors ran *along* a shape axis, and matching
`overlap._IMAGES` fixed **which images exist** — it did not fix **how many it takes to cover a
face**. A face covered by two off-axis vectors taking half each was outside what a per-vector
search could express, which was the same shape of limitation as missing `t . u != 0`, one level
up. `test_a_rhombic_cell_spans_along_the_sum_of_its_basis_vectors` argued for the candidate set
more broadly than the set could deliver.

The planner accordingly unions the covered intervals per face (`overlap.covered_length`) and
calls a face dead only when the union covers it; a layer whose every face is dead is `empty`,
and its field is the constant `(1, 0)` — `tests/test_gamma_plan.py` pins that the tiling bar's
projector coefficients equal an empty layer's and its solve equals the slab's. Neither the fold
nor the first refusal had this: the fold read `merged_v = False` there and kept two faces that
are not interfaces, which is harmless for the reason above but is not what the geometry says.

## 3. Removing a face pair left a null normal on the survivor's midline — FIXED

**Status: fixed.** Recorded because of how it was found and because the first causal account of it
here was wrong.

Removing one face pair leaves the other two exactly opposed, so on their own centre line they carry
equal weight and cancel to `(0, 0)`. A blend of four faces has no term left to break that tie, and a
null normal is a null projection rather than a harmless one. On the Li checkerboard
`a1 = (1.25, 1.25)`, `a2 = (1.25, -1.25)` with `lx = 2.5` (`= |a1 + a2|`), `ly = 0.8`, angle 0, that
was 128 / 256 / 512 null normals at `N` = 128 / 256 / 512 — one line, not a shrinking set.

**A square cell cannot see it.** There the survivor's centre line is `x = cx`, and a centred
`(i + 0.5)/N` grid with even `N` never samples it; the same shape on a 700 square cell gives zero
null normals at every `N`. The checkerboard's skewed grid crosses the line once per row.

**How it was found, which is the reusable part.** A sibling session implemented the same
`t.u == 0` correction by a different mechanism — an explicit face-pair drop rather than this
distance fold — and asked which of the two avoided a degeneracy it had hit. Neither did, and the
counts were identical, which is what established that the cause is the *removal* rather than how
the removal is spelled.

**The first control run here was wrong and the conclusion drawn from it was too.** Forcing
`merged_u` / `merged_v` to False left the count unchanged, and this file briefly said "it is the
fold, not the equality test". That control disabled the *decision* but not the mechanism the
decision drives: with `2a == Pu` the fold still equalises the two u distances, so the pair was
still gone. Disabling the fold outright — which also clears the gate, since `merged_u` requires
`Pu > 0` — restores the pair and puts the count back to 0 with `max|n_x|` back to 1.000000. The
sibling's control did this correctly first.

**The fix breaks the tie instead of changing the removal.** Where the blend lands on zero,
substitute the axis of the pair that did *not* cancel there — read off the plan's pair records
by array equality of the two members' distances (since 2026-09-06; the fold's version read a
static "which pair merged" flag) — and `(1, 0)` when both cancelled or neither did. It needs no
sign, because the factorization reads `n n^T`. Array-only, so it survives `jit` and `grad`, and
the exact-zero comparison is safe because the tie is an exact symmetry rather than a near-miss.
On the sheared cell the two v runs overlap over a strip, and on the box's midline inside it the
tie reads the u axis — turned with the box, which is what kills a `(1, 0)`-always mutant. With
several shapes in the layer (§8a) the records are read across shapes: for each box in turn,
the axis whose records did not cancel where the other axis's did, and a cross pair whose
materials differ never counts as cancelled, since it did not.

Measured bit-identical on a small box, a box above half the repeat, an ellipse, the 1-D grating,
its 90-degree mirror, a cell-filling slab and a two-shape layer; the checkerboard scene goes
256 -> 0 null normals with `max|n_x|` still 0.000000, which is correct — that direction genuinely
has no interface. Pinned by
`tests/test_nvf_period_spanning_shape.py::test_a_merged_pair_leaves_no_null_normal_behind`.

**What the drop was worth even before this.** At those same points the pre-removal field was
`(±1, 0)` — pointing hard along the one axis that has no interface at all, so `n n^T` was `x x^T`
where the truth is `v v^T`. Removing the pair moved them from confidently wrong to null, which is
already an improvement for a factorization that reads the outer product; the tie-break finishes it.


## 4. The lattice-direction test is a threshold, not an exact question — REPLACED (2026-09-06)

**Status: replaced by a threshold one level down.** The fold this section describes is gone; the
piece field's one threshold is whether two face lines are *one line*, at 1e-12 of the longer
basis vector (`overlap.COLLINEAR_TOL_REL`, 7e-10 on a 700 cell), and whether two normals are
opposed at `|n_p + n_q|^2 <= 1e-24`. Both are rounding-scale for the reason below, and both
step: just above the collinearity threshold two opposed faces `delta` apart cancel only outside a
band of depth `(4 delta D^4)^(1/5)` — about two units for `D = 350` — and at it they cancel
identically. Through a solve that is a step in the value of 1.4e-5 on the square grating at
order 5 and 1.7e-4 on the sheared box (N = 256) -- not the ~1e-7 first estimated -- at a
rounding-scale location, and on the sheared cell it does not decay with order (1.8e-5 / 1.9e-5 /
5.2e-5 at orders 4 / 8 / 12, against 1.1e-5 / 4.1e-6 / 1.4e-6 on the square; at N = 512 and
order 5, 1.4e-5 and 4.9e-5), because the band's depth is independent of the grid and the number
of samples it holds grows with N. The rest of the ladder is monotone on both cells, and
`tests/test_gamma_field.py::test_the_ladder_through_the_pair_threshold_is_monotone` walks
`ly = 700 - gap` on the sheared cell down through rungs at 7.1e-10 and 6.9e-10 to 0 and requires
the approach to be monotone. The paragraphs below are the fold's record.

The parametric fold (§5) has to ask whether a lattice vector runs **along one of the shape's own
axes**. That question cannot be asked exactly, because `pi/2` is not a representable angle:
`cos(pi/2)` is 6.1e-17 rather than 0, so a shape turned a quarter turn misses an exact test by
4.3e-14 on a 700 cell. `parametric._LATTICE_DIR_TOL2` draws the line at 1e-12 relative — the same
place `lattice.is_axis_aligned` draws its diagonal test, for the same reason.

Being a threshold, it steps. A shape **more than half a repeat long** folds on one side of it and
does not on the other:

| rectangle, angle | field vs `angle = 0` |
|---|---|
| 1e-12 rad | 8.1e-11 |
| 1e-11 rad | 1.19e-1 |

Through a solve that is `sum(T)` 0.976583641 against 0.976535505 — 4.8e-5, the same size as the
correction the fold itself makes, with the **better** of the two values at `angle = 0`.

Three things bound it. 1e-11 rad is 5.7e-10 degrees. A shape at or below half a repeat does not step
at all, because the fold is arithmetically inert there. And the step is in the value, not in the
slope: `d(sum T)/d(angle)` at zero is -1.2e-14 by AD against 2.1e-11 by central differences at
`h = 1e-4`, both of them the zero symmetry requires — so nothing an optimiser descends walks into
it. A test pinned it while the fold existed and went with the fold; the threshold that replaced
it is pinned by `tests/test_gamma_field.py::test_the_ladder_through_the_pair_threshold_is_monotone`.


## 5. Which parametric scenes changed when the fold landed (2026-09-03)

**Status: fixed.** Recorded here because it moves published numbers, and the rule for *which*
numbers is not the obvious one.

Before this landed, a box as long as the cell repeat along one of its axes had that pair of faces
read as material interfaces. `ly = period` is the only way to write a 1-D grating on the
parametric path — above it is refused by name, below it is a 2-D structure — so the boundary value
was the one value in the family that was accepted and then mishandled. It gave `|n_y| = 1` over a
third of the cell where the raster builder for the same grating gives 0. The fix measures each
face distance to the nearest **image** of that face; at the span the two opposite face families
become the same set of lines, so their distances are equal and their normals cancel identically.

**What the fixed scene is worth, and it is not accuracy.** A normal field is a numerical device:
away from an interface any admissible continuation converges, so "closer to the reference at order
10" does not decide which is right. Measured against a converged reference the fix is 2.2x to 3.4x
closer on the factorization-sensitive polarization (9.4e-3 -> 4.3e-3, 2.7e-3 -> 9.4e-4,
1.2e-3 -> 3.4e-4 at order 6 / 10 / 14) and about 1.4x further on the other. That is not the case.

The case is that two spellings of one structure now give one answer. A rectangle with `ly = Py`
*is* a lamellar grating, whose exact field is the constant `(±1, 0)`, and under a constant field
the NVF projector reduces to Li's 1-D rule outright. So the parametric path and a raster of the
same grating factorized by `'li'` must be the same calculation:

| truncation order | 6 | 10 | 14 |
| --- | ---: | ---: | ---: |
| `\|sum(T) parametric − sum(T) raster+Li\|` | 7.4e-15 | 1.2e-13 | 6.1e-13 |

They were 5.6e-3 / 1.3e-3 / 4.6e-4 apart, which reads as an ordinary truncation residual and is
not one.

`examples/shapes_1d_verification.py` is the shipped scene that shows it — two `ly = 300` rods on a
300 period against a raster of the identical permittivity function, the rods landing on pixel
boundaries on purpose so the two paths describe the *same* geometry:

| | order 3 | 10 | 20 | 40 |
| --- | ---: | ---: | ---: | ---: |
| before, `norm(T_shapes − T_raster)` | 4.7e-3 | 3.7e-4 | 1.7e-4 | — |
| after | **3.6e-16** | 2.1e-15 | 6.9e-14 | 1.5e-13 |

That file carried a paragraph reading the old gap as a factorization difference "converging to the
same answer from different sides, so the gap is a truncation error rather than a disagreement" —
with the sweep that refutes it printed three lines below. 4.7e-3 -> 3.7e-4 -> 1.7e-4 across a
near-sevenfold rise in order is a plateau. The paragraph has been replaced.

**A scene's answer moves iff all three hold.** Spanning is the extreme of (3), not the condition —
checking for spanning shapes alone will miss cases.

1. the shape is a **box** type (`rectangle`, `square`). Round types are never affected, at any
   size.
2. one of the shape's **own axes runs along a lattice vector** — within the tolerance of §4, and
   counting `a1`, `a2`, `a1 + a2` and `a1 - a2`. A box turned off every lattice direction is never
   affected, at any size.
3. its length along that axis is **strictly greater than half** that lattice vector's length.

Measured against a control that keeps everything else and deletes only the distance fold, so the
rows isolate the fold rather than mixing it with the frame move that landed the same week.
Rectangle `lx = 350` at angle 0 on a 700 period, field compared at **N = 256**:

| `ly / P` | change |
|---|---|
| 0.40, 0.49, 0.499999, 0.50, 0.500001 | bit-identical |
| 0.51 | 6.5e-10 |
| 0.60 | 8.1e-5 |
| 0.90 | 1.2e-1 |
| 1.00 | bit-identical |
| the same shapes at 30 degrees, and ellipses at every size | bit-identical |

**The grid is part of those rows and has to be named.** Just above half, the fold's correction is
smaller than the spacing between samples, so whether it shows at all depends on N: `ly/P = 0.51`
is bit-identical at N = 64 and 96, and 5.0e-10 / 6.5e-10 / 7.8e-10 at N = 128, 256 and 1024. That
is also why `0.500001` reads as bit-identical here while rule (3) below counts it as affected —
the rule is about the arithmetic, the table is about what a finite grid can see.

**And `1.00` is bit-identical for a different reason**: at exactly the span the equality test
(`merged_v`) drops the pair in the control too, so there is nothing left for the fold to do. The
fold's whole job is the open interval between half and the span.

Through a solve, `truncation_order` `(6, 6)`, `|delta sum(T)|` against the same control: 4.9e-10 at
0.55, 2.0e-8 at 0.60, 4.8e-6 at 0.80, 1.2e-5 at 0.95, 0 exactly at 1.00, and 0 exactly at 0.90
turned 30 degrees. The largest movement is reached before spanning, not at it.

**The frame move that landed the same week moves nothing here, and it is worth being able to
say which change did what.** Shape centres and sample grids went from an internally corner-based
frame to the cell centre on 2026-09-04; the public spelling of a centre did not change, only the
plumbing. Compared against `git archive HEAD` on the same scene written the same way, the
parametric normal field is **bit-identical** on a centred box, a rotated box, a box near the cell
edge, two shapes, a large ellipse, and both of those on a sheared cell. The one scene that moves
is the spanning grating — by 2.0 over half the cell, which is this section's subject and not the
frame's doing. An audit of this change initially read that difference as a frame effect; the
control that separates them is any non-spanning shape, where the frame move is exactly zero.

**Condition (2) counts `a1 + a2`, and that is not decoration.** Writing it with the two basis
vectors alone was tried and is wrong. On the Li checkerboard basis `a1 = (1.25, 1.25)`,
`a2 = (1.25, -1.25)` the sum is `(2.5, 0)`, which runs along an unrotated box's x axis, and
`check_layer_shapes` accepts `lx` there right through the span — to 2.500001, since the overlap
check carries a `OVERLAP_TOL_REL` slack of 1e-6 relative; 2.5000025 is refused. The two-vector version left
`max|n_x| = 1.000`, with |n_x| above 0.01 over 46.9% of that cell. The candidate set matches `overlap._IMAGES`, the same
3x3 the layer check tests against, so the fold and the refusal cannot disagree about which images
exist; a shape able to reach across a longer direction is one whose images that check never looked
at either.

**A predicate**, reading the same values the fold read so it could not drift from it. It is
the fold's, kept as the record of what this section measured: `_LATTICE_DIR_TOL2` and the fold
went on 2026-09-06 (§8), and every box this predicate answered `None` for moved then anyway.
**A historical listing, not a runnable one**: the import below raises `ImportError` on any tree
since that day, and the fence says so rather than `python`.

```text
# HISTORICAL (2026-09-03 to 2026-09-06): `_LATTICE_DIR_TOL2` no longer exists; this does not import.
import numpy as np
from wavebit._core import overlap as _overlap
from wavebit._core.parametric import shape_geometry, _LATTICE_DIR_TOL2

def affected(shape, lattice):
    """None, or [(key, length, half_repeat)] for each axis whose answer moves."""
    stype, _, _, lx, ly, angle = shape_geometry(shape)
    if stype not in _overlap._BOX_TYPES:
        return None
    ca, sa = np.cos(float(angle)), np.sin(float(angle))
    a1x, a2x, a1y, a2y = (float(lattice[0][0]), float(lattice[0][1]),
                          float(lattice[1][0]), float(lattice[1][1]))
    Pu = Pv = 0.0
    for tx, ty in ((a1x, a1y), (a2x, a2y),
                   (a1x + a2x, a1y + a2y), (a1x - a2x, a1y - a2y)):
        length2 = tx * tx + ty * ty
        on_u, on_v = tx * ca + ty * sa, -tx * sa + ty * ca
        if on_v * on_v <= _LATTICE_DIR_TOL2 * length2 and (Pu == 0 or abs(on_u) < Pu):
            Pu = abs(on_u)
        if on_u * on_u <= _LATTICE_DIR_TOL2 * length2 and (Pv == 0 or abs(on_v) < Pv):
            Pv = abs(on_v)
    hits = [(n, l, P / 2) for n, l, P in (('lx', float(lx), Pu), ('ly', float(ly), Pv))
            if P > 0 and l > P / 2]
    return hits or None
```

**What the fold leaves behind at the span is a slow approach, not a step.** `sum(T)` is
continuous into `ly = P` but reaches it sub-linearly: measured downward from the span at
`truncation_order` `(5, 5)`, `|delta sum(T)|` is 6.7e-6 at `delta ly = 1e-3`, 2.9e-6 at 1e-5,
1.1e-6 at 1e-7, 4.1e-7 at 1e-9 and 1.2e-7 at 1e-11 — monotone, and about eight decades of
`delta ly` for under two decades of value. Below roughly 1e-11 it stops being monotone and sits
at the 1e-7 level: there the two folded distances agree to within a few ulp and their difference
has no significant digits left. So there is no cliff to climb, but an optimiser pushing `ly`
into the last picometre of the span is reading rounding. For scale, the systematic step
the fold *removes* is 4.3e-5 at the same operating point — measured by keeping the `merged_u` /
`merged_v` equality and deleting only the distance fold, which gives 0.976311243 at `ly = P`
against 0.976268339 one ulp below, where the equality no longer fires.

Measuring any of this requires clearing `parametric._PROJECTOR_CACHE` between builds.
`normal_projector_coeffs` reuses coefficients across solves, so a builder swapped in after the
same structure has been solved once is never called, and the two arms report identical numbers
with nothing wrong in either.

The **Li checkerboard itself does not move**: `lx = ly = 1.25` is exactly half of
`|a1 + a2| = 2.5`, and (3) requires strictly greater, so the fold is inert and the field is
bit-identical. Boxes above 1.25 on that cell do move — 2.2e-8 at 1.30, 7.8e-5 at 1.50, 3.4e-1 at
2.40.


## 8. Which scenes changed when the piece field landed (2026-09-06)

**Status: landed.** Recorded because it moves published numbers, and this time the rule for
which numbers is simple: **every lone box that is not a grating moves, by ~4e-4 in `sum(T)` at
order 6**, and nothing else does.

The corner-radial extension of the fold's box branch — outside a corner, the normal pointed
from the corner — gave way to fixed face normals: a face piece's direction is its outward
normal at every sample, inside or outside its tangential extent, and outside a corner the two
adjacent faces are equidistant and the blend reads their diagonal. That was measured before it
was chosen: the foot / corner-radial rule beyond a piece's extent breaks the 1-D grating
(`max|n_y|` 0.76 where it must be 0, because a covered face and its partner share endpoints and
their foot directions reinforce instead of cancelling) and puts a ~4e-4 non-decaying step at
every merge; fixed normals give the grating exactly, a zero step at every merge, and solve the
sheared and tilted scenes of §2 at 0.16–0.19 deg. The price is off-interface, and an admissible
continuation replaced by another -- and it has a size. A lone box that touches nothing reads a
higher band error than it did, by 0.06 to 0.34 deg in the mean, because every corner quadrant
now reads the diagonal of its two faces: at equal grid against the exact interface oracle the
350 box goes 0.358 -> 0.473 deg at N = 256 and 0.179 -> 0.237 at N = 512, the 120 x 150 rod
0.099 -> 0.279 and a 200 x 150 box at 25 deg 0.249 -> 0.587 (N = 512), with the fraction of the
band past 45 deg falling to zero on the 350 box and rising to 0.13% on the other two. The
`'contour'` raster instrument the goldens are booked against sits nearer the fold at every order
on every lone-box golden -- the rod, the 350 box, the two-shape pair of `test_lattice_pytree.py`
-- which is a resemblance of extensions (`'contour'` is distance-based and extends the way the
corner-radial rule did) and is recorded as what the measurement shows, not as a verdict; each
test carries its rows. Li's rule on a box with its edges on pixels does not say the same at every
order: on the 350 box it sits nearer the fold at orders 6 and 9 (6.3e-4 against 1.03e-3, 2.74e-3
against 3.02e-3) and nearer the *pieces* at order 12, by four times (1.1e-4 against 4.7e-4).

| scene, order 6, `normal_grid` 512 | fold (2026-09-03) | pieces | change |
|---|---|---|---|
| 350 x 350 box on a 700 square | 0.891888043 | 0.892284142 | +4.0e-4 |
| L box, 689 x 200 at atan2(117.9, 690) | 0.918790794 | 0.918264766 | -5.3e-4 |
| 600 x 650 at 10 deg | 0.545259776 | 0.519815227 | -2.5e-2 |
| 1-D grating, 350 x 700 | 0.627676664 | 0.627676664 | 0 |
| stripe along x, 700 x 200 | 0.969594227 | 0.969594227 | 0 |
| checkerboard span, 2.5 x 0.8 on Li's cell | 0.898529436 | 0.898529436 | 0 |
| the span golden, (5, 5), 550 nm | 0.9763112427108831 | 0.9763112427108831 | 0 |

The 600 x 650 row is not a free-region shift: its faces were 1.7 deg wrong on the fold with 1.9%
of the boundary band past 45 deg (the box leaves its Wigner-Seitz cell, and the nearest interface
for part of the band belonged to a neighbour), against 0.14 deg / 0.04% now. The gratings are
bit-identical because both mechanisms are exact on them, and round shapes inside their
Wigner-Seitz cells are untouched (the `'arc'` piece is the old round branch verbatim there; one
that leaves its cell is carried over a ring since the same day, case L). Multi-shape layers holding a box move by the
same class through their per-shape field; the cross-shape blend is the ratio form, which is
direction-identical to the product it replaced (a lone round shape bit-identical; on round-only
layers 3e-14 at N = 256 -- 2.5e-14 on two circles, 1.5e-14 on a circle and an ellipse on a hex
cell -- not the 1e-15 first written).

Two goldens outside the parametric files moved and were re-derived rather than re-baselined:
the 120 x 150 rod of `test_a_length_is_a_size_and_the_transform_carries_its_sign` and the
circle-plus-rectangle pair of `test_the_parametric_spelling_has_no_frame_to_declare`; each
test carries its reference rows.

### 8a. The second half: several shapes, one blend, and the materials decide the seam

Landed the same day. A layer of several shapes was two blends -- each box's four faces into
one `(distance, normal)`, then the shapes by the ratio of those distances -- and the second
stage switches extension at the moment two shapes come into contact, measured as a 1.1e-2 step
in `sum(T)` at order 6. There is one sum now, over the faces of every box and the arc of every
round shape (`parametric._piece_field`); a lone round shape is one piece and keeps its own
field.

Two boxes that abut share a face line over a run, inside the cell or against the other's
periodic image, and the plan records that pair too. Over the run the two faces' distances are
one number by the construction a lone box uses across its own images -- every instance is
named by its *absolute* image, fold plus offset, and the partner's normal coordinate is the
primary's expression at the coincident image, so the two shapes' folds cancel out of the
arithmetic -- and the **material slots decide the sign**, per pair and per run, as a window in
the tangential coordinate rebuilt from the traced geometry: equal materials cancel (the
material runs on), different ones co-orient (the seam's normal survives). Beyond the run the
two keep their opposed normals, which is right for both materials. The permeability pass reads
its own slots, so the interface set is per quantity. What it measured:

| scene | before | after |
|---|---|---|
| **D** two same-material bars, stacked / side by side, against the one bar they make | two-stage: a step at contact | `(n,0)` vs `(n,n)` at `psi = 0`, `(0,n)` vs `(n,n)` at `psi = pi/2`, and two-bar vs one-bar, all 4e-15..2.4e-14 at n = 3, 5, 7 (one-bar floor 4e-15) |
| **D** one 350 box against its two halves, `sum(T)`, orders 4..12 | -- | 1.09e-3, 5.11e-4, 4.73e-5, 1.51e-4, 8.05e-5: an order down, not monotone |
| **K** `eps 4 \| eps 9` bars spanning the cell | a null band on the seam, `\|n_y\| = 0.03` | `(+-1, 0)` exactly, no null; 1.4e-14 / 1.0e-13 from raster + Li at orders 6 / 10 |
| **K** two 300 boxes of different material abutting inside the cell, band vs the material-aware oracle | -- | 0.22 deg, 0.09% past 45 deg (0.33 deg against the oracle with the seam dropped) |
| the same two boxes of one material, band vs the oracle | -- | the 600 x 300 box's own numbers to 1e-7 |
| the two-rectangle position gradient, AD vs central differences at h = 1e-7 | 2.6e-6, 9.3e-5, 9.7e-5 | 1.9e-5, 3.8e-5, 8.7e-6 (the values moved with the extension: 6.90e-3 -> 8.69e-3 at d = 0.02) |

A weight's derivative moved with it, and this is worth a line because it was wrong before the
rewrite too: the ratio weight `where(d == d_min, 1, (d_min / d)^4)` gave every piece *tied*
with the nearest one a zero derivative, and such ties sit on lines a grid samples (`5 j - 7 i =
97` on the sheared cell's grid). The lone sheared box's `d(sum T)/dlx` at `lx = 350` read
5.2e-3 relative from every central difference down to `h = 1e-6`, constant in `h`; the honest
ratio (the winner's is `d / d`, exactly 1.0) leaves every value bitwise where it was and brings
it to 7e-7.

### 8b. What was deleted, and what stands in its place

Gone with the piece field, in the two commits that landed it: `_axis_lattice_periods`,
`_fold_by`, `_shorter_period`, `_reciprocal_or_zero`, `_LATTICE_DIR_TOL2`, the box branch of
`_shape_surface_normal` (its round half is `_round_field`), `_blend`, `_blend_tiled`,
`_refuse_partial_self_coincidence` and its call, `overlap.partial_self_coincidences`; and with
the second half, `_ratio_blend`, `_box_field` and `_folded_frame`. The cache key's scale-free
normalisation of a lone round shape went ahead of the round piece's ring. What stands in their
place: `overlap.face_pairs` (the sweep), `overlap.covered_length` (the union per face),
`overlap.interfaces_of` (the oracle, material-aware through `groups`), `parametric.plan_gamma`
/ `GammaPlan` / `Piece` / `Pair` (the record), `_Frame`, `_run_window` and `_piece_field` (the
replay), `_blind_plan` (a jitted solve with no record) and `check_gamma_plan_still_fits`.
`_fold_to_nearest_image` stays because the round piece reads it.

**The record is frozen.** `freeze_geometry` records every layer's plan beside its painting
plan (`StaticGeometry.gamma_plans`, in the content key), both quantities replay the one record,
a jitted solve with it builds the planned field where one without it gets the blind plan (every
piece on a ring read from the basis, no pairs -- and a basis that cannot be read, a solver built
inside the trace, is refused with the freeze message since 2026-09-06: the fixed 7x7 ring it fell
back to lacked 460 of the 479 planned instances on an unreduced basis and solved 6e-3 off; since
2026-09-06 that basis plans as its reduction, §8c, and a traced *ellipse* is refused there too,
its instance list being a size the trace cannot read), and
the eager replay re-derives the plan and refuses by name a pair that appeared or vanished, a
face gone dead or live, an image list that grew across a ring boundary, a basis whose reduction
changed. A material sweep is
never a staleness: the sign is an array op on the slots at build time. The projector cache
keys on the plan and on a cross pair's *equality bit*, so a sweep on a pair-free layer hits as
it always did and one that crosses equality on a seam rebuilds.

### 8c. Every image is counted on the reduced basis (2026-09-06)

**The fold was wrong on an accepted basis, and had been since it was written.** The round
piece finds the nearest image by rounding the fractional coordinates and sweeping one cell a
side (`parametric.py::_fold_to_nearest_image`), and its docstring said one cell a side was
enough for any basis the solver accepts. `as_lattice` accepts `a2 = k a1 + (50, 700)` for any
`k` -- the cell of `a2 = (50, 700)`, spelled with a longer second vector -- and on it the
Wigner-Seitz cell of the origin reaches `|s| ~ (1 + k) / 2` in fractional coordinates, past the
3x3 from `k = 3`. Measured on a lone 693 circle (0.99 of the period) against its own boundary
and level set, 4000 points: `k = 0 / 1 / 2` read 2.5e-7 deg mean, 1.2e-6 max; `k = 3` reads
**8.89 deg mean, 69.9 max**, 900 of 4000 points past 1 deg; `k = 4` 31.7 / 90.0; `k = 10` --
the basis the F1 refusal enshrined as accepted -- **43.8 / 90.0**, 3224 of 4000. Identical digits
on the fold tree (8bfc6aa): pre-existing, on every backend, on the eager path, with nothing
said. The box path was right on the same spellings (every face is carried over a size-only
ring, 441 and 479 instances there) and so was the ring ellipse (419 instances) -- at a price:
479 offsets per face for a structure whose reduced spelling needs 21, ring 14 against 3, and a
readable blind ring of 1559 under which a jitted lone box did not finish compiling.

**The fix is contained to the geometry.** `lattice.py::reduction_transform` is Gauss /
Lagrange reduction of the two vectors -- subtract the nearest multiple of the shorter from the
longer until `|a1 . a2| <= min(|a1|^2, |a2|^2) / 2`, within 1e-9 relative so that a hexagonal
cell reached through `cos(2 pi / 3)` stays itself -- returning the integer unimodular `T`;
`lattice.py::reduced_basis` applies it, and returns the *same object* when `T` is the
identity. Every enumeration of periodic images goes through it: the fold, `_Frame`'s
instance offsets, the ring and arc lists of the planner and the blind plan, `_run_window`,
`overlap.overlapping_pairs` (whose 3x3 on the basis as given accepted a 100 x 800 box on the
`10 a1` spelling that overlaps its `(50, 700)` image over a 50 x 100 patch), `face_pairs` and
the oracle `interfaces_of`. The plan records `T` (`GammaPlan.reduction`, in the content key),
so a jitted replay on a traced basis reduces with the recorded integers rather than reading
the tracer, and `dR / dA = T` exactly under `jax.grad`. Nothing on the Fourier side sees it:
`order_kxy`, the truncation window, `periods_of`, the sample grid and the origin a centre is
measured from are the basis as given, because a window on a different spelling is a different
set of retained orders. Why the 3x3 is enough on a reduced basis: with `|mu_i| <= 1/2` for both
projections, a point of the Wigner-Seitz cell has `|s1 + s2 mu1| <= 1/2` and `|s2 + s1 mu2|
<= 1/2`, hence `|s_i| <= 1`; the rounded residual sits within `|s_i| <= 1/2`, so its nearest
image is at most 3/2 cells away per index -- an integer in `{-1, 0, 1}`.

**Gates.** The circle ladder `k = 0..10` reads the `k = 0` value, 2.4e-7 deg mean and 1.2e-6
max, at every rung, and the 720 x 200 ellipse keeps its eleven instances and 1e-7 on every
spelling (`test_gamma_plan.py`); the 350 box, the 500 x 700 box, the circle and the ellipse
build one field at the same physical points on the `10 a1` and the reduced spellings to 1e-9
deg; the plan on the `10 a1` spelling is the reduced spelling's plan up to the recorded
transform (ring 3, 21 offsets, the pair at `(0, 1)`), the blind ring 25 from 1559; and on the
square, the rectangle, both hexagonal spellings, the `(200, 700)` shear and Li's checkerboard
-- reduced already -- 75 of 75 field and projector-coefficient arrays are byte-identical to the
tree before the change. The solve on the two spellings is deliberately *not* compared: the
retained order sets differ by the window, not by geometry, and at any affordable order that
difference dominates. One thing is outside this: the *tolerance scale* of the overlap check
(`OVERLAP_TOL_REL * max(|a1|, |a2|)`) is still read off the basis as given, so the `10 a1`
spelling tolerates touching ten times more loosely than its reduction -- a scale, not a
geometry.

**The boundary route carried the same fold, and the port took two folds, not one.**
`boundary.py::periodic_segments` -- the painted (overlapping) layers -- was a copy of the same
fold and 3x3 on the basis as given: two overlapping 300 x 300 boxes on the `10 a1` spelling
read `sum(T)` 0.810615440141 against 0.948626126969 on the reduction at the one window where
both spellings retain the same order set (`u = 0`, `v = 6`; `b2` is shared), the same at
`normal_grid` 64 / 128 / 256 -- geometry, not resolution -- while the disjoint control, the
closed form, read bitwise the same on both. For one revision that was *refused* by name on a
painted layer. The refusal over-reached: `test_oblique_lattice.py` pins that a painted overlap
survives the `k = 1` respelling `a2 = (400, 400)` of the 400 square, a genuine invariance the
sweep as given happens to satisfy for `k <= 2` (the Wigner-Seitz argument fails from `k = 3`),
and the refusal broke it. So the reduction is ported (2026-09-06): `periodic_segments` folds
and sweeps on `reduced_basis`, and the painting plan records the integer transform beside its
loops (`plan_layer` returns `(records, regions, reduction)`, in the content key), taken on the
concrete basis at the freeze so a jitted replay on a traced basis reduces with the recorded
integers, as `GammaPlan.reduction` does; a record taken on one spelling and replayed on
another is refused as stale (`check_plan_still_fits`, naming the basis). Folding the
*segments* alone moved the `10 a1` answer from 0.810615440141 to 0.985549487763 -- a different
wrong number, as unmoved by the grid as the first -- because the sample points are laid out on
the basis as given, and on that spelling that is a sliver reaching ten cells along `a1`, where
no image of a segment folded into the reduced cell exists. So the sample points are folded
into the reduced cell by the same rule (`fold_into_cell`); on a basis that is reduced to begin
with the fold is `x - 0.0 * a`, and the field is the field it was, to the bit. Pinned in
`test_boundary.py`: the nearest distance and normal at 4000 identical physical points agree on
`k = 0 / 1 / 3 / 10` to 1e-13 and 2e-6 deg; the solve on `k = 10` reads the `k = 0` value
**bitwise** (the structure moves by `5 a1`, a lattice vector, and the sample grid folds onto
the same points); `k = 1` and `k = 3` read one value, within 1.2e-4 of it (their sample grids
fold half a pixel off -- a sampling difference of the size of the route's own moves between
`normal_grid` 64 / 128 / 256, 3e-6 to 9e-5, and the same at every `k` of one parity, which is
what says it is the grid and not the fold); the jitted replay over a traced lattice equals the
eager one to 1e-12; and three mutants -- the reduction dropped, the point fold dropped, the
plan recording the identity -- are each killed by the bitwise pin.

### 8d. JAX shape `solve_field` — FIXED (2026-09-07)

The solver pytree now carries shape strings (such as `type`) as static declarations,
while shape dimensions and Material values remain dynamic leaves. The internal
`_ShapeStructure` view preserves the original list/dict/Material tree on reconstruction.
Changing a declaration changes the treedef; changing a numeric value does not freeze it
into a compilation key. Previously every parametric call failed before tracing because
JAX tried to accept the shape's `type` string as an array argument.

The supported recipe is:

```python
solver = wavebit.rcwa(..., answers='field')
geometry = solver.freeze_geometry()
result, field = solver.solve_field(4, 4, 3, geometry=geometry)
```

The frozen geometry requirement and eager staleness checks are unchanged. Regression
coverage in `tests/test_parametric_field_jit.py` compares the compiled field with eager
JAX and NumPy for lone and overlapping magnetic shapes on square and sheared cells,
checks static declarations versus dynamic values, and differentiates the reconstructed
field with respect to width and permittivity. The raster field and solver round-trip
regressions remain applicable too.


## 9. The cases, A to L, as of 2026-09-06

The letters are the design's. "Exact" means the field equals the geometry's on the interface
set; a band error is against `overlap.interfaces_of` over 1.5 pixels at N = 512, and the gate a
scene is held to is the scene's own -- the fixed-normal wedge beyond a run's end reads 0.22 deg
on the K seam, 0.27 on the A|B|C scene and 0.32 on two same-material boxes 300 and 400 tall
(identical whether the taller one is upright or turned a quarter turn, so the wedge, not a
defect), and a new contact scene is gated by comparison with its one-shape spelling or by its
separation from the mutants that must fail it (`tests/gamma_instruments.py::band_error`).

| case | scene | status | measured in |
|---|---|---|---|
| A | a 1-D grating written on a 2-D cell, `ly = P` | exact; `max\|n_y\| == 0.0`, raster + Li to 1e-11, the (5, 5) golden bit-identical to the fold | `test_nvf_self_coincident_faces.py`, `test_gamma_field.py` |
| B | a box on a sheared cell, a tilted box meeting its own image part way | built (refused until 2026-09-06): 0.16 deg / 0.11% and 0.18 deg / 0.15%; rotation with the frame an invariance to 5e-12 | `test_nvf_self_coincident_faces.py`, `test_gamma_field.py` |
| C | the bar that tiles Li's checkerboard cell | the plan is empty, the field the constant `(1, 0)`, the solve the slab's to 1e-12 | `test_gamma_plan.py` |
| D | two touching same-material bars; a box against its two halves | the one bar they are, to float noise (§8a); the 2-D residual an order down by order 8 | `test_gamma_layers.py` |
| E | a shape whose material equals the background | its faces are kept; a pair-free field is bit-identical across any material sweep, a seam's depends on the equality bit alone | `test_gamma_layers.py` |
| F | a shape nested in another | **fixed 2026-09-07** — actual polygon interface intervals; equal-material inner contours are absent throughout the cell (§1) | `test_polygon_interfaces.py` |
| G | a shape named a period over | exact: the plan is position-free and a lone shape rides a phase; a multi-shape layer is periodic to 1e-9 | `test_gamma_plan.py`, `test_shape_position_gradients.py` |
| H | a shape larger than the period | **refused**, as before (§1) | `test_parametric.py` |
| I | ties: a dead pair's line, a surviving pair's midline, a box's centre | the axis of the pair that did not cancel, across shapes; `(1, 0)` when both or neither did; zero null normals at N = 128 / 256 / 512 on the checkerboard | `test_nvf_self_coincident_faces.py`, `test_gamma_field.py` |
| J | merges: a face approaching its partner's line, two shapes coming into contact | **Closing-gap sensitivity remains unresolved.** On the square grating at order (5,5), normal grid 256, central secants with `h=gap/10` are +0.010405 / +0.73117 / +50.640 at gaps 1e-2 / 1e-4 / 1e-6. The corresponding transmission differences from contact are -8.0e-4 / -3.6e-4 / -9.2e-5. This finite ladder records a large slope in the current blend, not an established physical divergence or a universal power law. Smaller gaps require normal-grid refinement; that refinement changes the measured ladder. Separating the derivative at gap 1e-4 shows the large positive contribution comes from the changing normal projector. Freezing it also disagrees with the independent Cartesian Li reference, so that is not a gradient fix. An attenuated-field candidate likewise fails the reference comparison and remains excluded. Earlier claims that a tuned ramp is the only remedy, or that these measurements prove an unbounded physical slope, are withdrawn. Historical B-onset one-sided secants at `lx=200` differ relatively by +4.2e-4 / -1.1e-3 / -1.6e-3 / -1.3e-3 / -1.5e-3 at orders 4..12; this finite range also does not prove persistence at infinite order. See `nvf-contact-followup.md` for grids, reference convergence, and the reproducible derivative decomposition. | `test_gamma_field.py::test_closing_gap_has_large_slopes_on_the_documented_grid`, `test_contact_reference.py`, `test_gamma_frozen.py` |
| K | abutting boxes of different material | the seam is an interface: `(+-1, 0)` exactly and Li-exact where it spans, 0.22 deg / 0.09% where it does not, per quantity, and the sign is a window per record in the run's traced bounds (a face on two runs of different equality reads `0 \| 1 \| 1` in `\|n_x\|` along its line, `test_gamma_layers.py`). **Three steps, all recorded.** (1) *The contact step* (the design's open question 2; the following measurements precede the 2026-09-07 endpoint fix): the sign flips at exact contact, and a 1e-9 gap -- two opposed faces that cancel to rounding with no pair record -- reads `sum(T)` -2.6e-3 / -9.4e-4 / -1.8e-4 / -7.7e-4 / -3.3e-4 from touching at orders 4 / 6 / 8 / 10 / 12 (two 300 boxes, `eps 4 \| eps 9`, N = 512): an order down across the cap and not monotone, a residual by the decay rule, against 1e-7..1e-5 for the same-material control. **The mechanism is that a sub-threshold opposed pair is cancelled whatever its materials**: without a pair record the blend treats two opposed faces `delta` apart as a merging same-material pair, since a material is read only through a record's sign -- so the strip between two close boxes of *different* material reads the far faces' normal instead of the gap faces', which are two real interfaces. On the line 20 units from the seam across the strip `\|y\| < 140` the fraction reading `u` is 0.30 at gap 1e-2, 0.02 at 1e-4 .. 1e-9 and 0.93 at contact, where the exact field reads 0.93 at every gap above zero; off Gamma (both gap faces themselves read exactly) and decaying with order (gap 1e-2: -2.9e-3 at order 4 to +4.1e-4 at 12), but a value step at contact on top of J's cusp. (2) *The material-equality step*: the sign flips when the two slots become exactly equal, and `T(9 + 1e-7 \| 9) - T(9 \| 9)` is -4.77e-3 / -4.91e-4 / -5.76e-4 / -1.36e-4 / -4.26e-5 at orders 4 / 6 / 8 / 10 / 12 (N = 128; `9 + 1e-3` gives the same to leading order) -- the design's accepted one-time jump on the first step, two orders down across the cap and not monotone at 6 -> 8, a residual by the decay rule. A stale self-pair record replayed under `jit` is value-neutral (1.2e-14 either way), so only a stale cross-pair record with different materials could mislead a jitted replay. (3) *The run-end step* — **fixed 2026-09-07**. The old hard sign window jumped whenever an endpoint crossed a sample row (-5.23e-6 in transmission at N=128, unchanged as the displacement shrank). Transfer now uses the normal squared distance `n2` and tangential excess `e`: one on the run, `n2/(n2+e**2)` off it, and zero on the line beyond the run. Multiple tails sharing a partner are normalized symmetrically so their total never exceeds its weight. The exact line values and material decisions are preserved; only the off-interface extension of different-material cross pairs changes. The old N=128/256/512 fixture now changes by 1.13e-9 / 1.37e-9 / 1.42e-9 for a 2e-6 displacement, and that change scales with the displacement rather than staying a step. Field and solve JIT derivatives agree with local secants. This does not fix J's closing-gap cusp or K's separate contact/material-equality steps. See `tests/test_gamma_contact_smooth.py` and `docs/nvf-contact-followup.md`. | `test_gamma_layers.py`, `test_gamma_frozen.py` |
| L | a lone box or ellipse leaving its Wigner-Seitz cell without touching anything; a lone round shape on an unreduced spelling of its cell (§8c) | the box reads its neighbour's face: 0.19 deg / 0.05% (0.65 / 0.64% on the fold). The ellipse too, since the second commit of 2026-09-06: while its support stays inside the cell (`h(t) <= \|t\| / 2` for every lattice vector, `parametric._arc_offsets`) the nearest centre's level set *is* the nearest interface and the `'arc'` piece keeps its single instance and its field bit for bit; once it leaves, a neighbour's boundary is inside the cell and the piece is carried over a ring with the **exact** Euclidean distance to each instance (`parametric._ellipse_distance`, the nearest point's angle bisected on the sign of the tangency condition, then Newton-polished: 5e-13 against brute force, the gradient implicit). The 720 x 200 ellipse at 10 deg read 7.0 deg mean and 89.9 max on Gamma and 3.09 deg / 3.73% over the band on the fold; over its eleven instances it reads 2.5e-7 deg on Gamma and 0.194 deg / 0.00%, the same as the 690 control that stays inside. **What the threshold costs.** The fold switches instance on the cell's edge where the nearest boundary switches on the Voronoi curve between two tilted tips, so an ellipse *hugging* its edge is wrong in the sliver between the two: 0.47 deg / 0.42% at 0.9 short of the edge (`lx = 708`), 0.71 / 0.77% at 0.1 short, clean from 2.4 short out (the same shape forced over the ring reads 0.19 / 0.00% throughout). Through a solve at order 6 / 9 the fold and the ring differ by 2.6e-5 / 2.9e-5 on a 400 x 200 (the sliver is an extension there), 8.9e-4 / 3.3e-5 at 600, 1.6e-3 / 2.5e-4 at 690, 3.9e-3 / 1.9e-3 at 705 and 4.8e-3 / 2.7e-3 at 709.8 -- and that difference is the **step** `sum(T)` takes when the support crosses the threshold: +4.6e-3 from 709.8 to 710.0 at order 6, against -1.8e-4 per 0.2 of `lx` on either side. The fix that removes the step is the ring for *every* ellipse, whose price is every tilted ellipse's extension (2.6e-5 at order 6 on the 400 x 200) and eleven exact distances per build (6 ms -> 470 ms at N = 512, 1.9 s at 1024); not taken, so the step stands as a plan-level event, refused by name when a frozen record is replayed across it. A circle never leaves: `R > \|t\| / 2` is the overlap the layer check refuses -- and since 2026-09-06 the fold finds its nearest centre on any spelling of the basis, every image being counted on the reduced one (§8c: 8.89 deg mean on Gamma at `a2 = 3 a1 + (50, 700)` before, 2.4e-7 after) | `test_nvf_self_coincident_faces.py`, `test_gamma_plan.py`, `test_gamma_frozen.py` |

**The fixed-normal decision and its price, in one place.** A face piece's direction is its
outward normal at every sample, inside or outside its tangential extent (C3 of the design's
corrections). The foot rule beyond the extent -- the corner-radial direction the fold's box
branch used -- was measured first and rejected: it breaks A (`max|n_y|` 0.76 where it must be
0, because a covered face and its partner share endpoints and their foot directions reinforce
instead of cancelling) and puts a ~4e-4 non-decaying step at every merge. Fixed normals give A
exactly, a zero step at every merge, and B and L. What they cost: ~4e-4 in `sum(T)` off the
interface on every lone box golden at order 6 (§8's table), the two-spellings residual of D in
two dimensions (a cut face weighs twice near the cut, decaying), and J's slope kink at contact,
which is the same wedge near a run's end seen through a derivative. J's cusp is not on this
list: it is the blend's, older than the pieces, and the fold had it to four digits.

## 6. What is *not* an instrument for any of this

**Translation invariance does not detect a non-periodic normal field.** Translating a whole
structure changes only phases, so `sum(T)` must be invariant, and a field blind to periodic images
ought to break it. It does not. Measured on the pre-fix parametric builder, box at angle 0,
`cy = 0` against `cy = P/4`, `truncation_order` `(8, 8)`, at `ly/P` of 0.40 / 0.50 / 0.55 / 0.60 /
0.80 / 1.00: `|delta sum(T)|` between 3e-15 and 7e-14 at every size, and at 30 degrees too. The
field was translation-covariant while still being wrong.

**Comparing one angle against another at fixed length is not an instrument either** — those are
different structures, so the difference is physics.

**Energy conservation cannot see any of it**, since the permittivity coefficients are untouched.

What does work, in order of how little it asks you to trust:

1. **The y-order invariant.** A layer that does not vary in y must give the same answer at
   `truncation_order` `(n, 0)` and `(n, m)`, because orders along a direction with no variation can
   only carry zero. It needs no second builder and no reference value, and it states its own
   correctness criterion. It requires a genuinely invariant layer, so it reaches the spanning case
   and not the merely-large one.
2. **The other modeling path, order for order**, on a geometry both spell exactly — e.g. a bar
   whose edges land on the raster's pixel grid. Any gap is factorization rather than geometry.
3. **The raster field as an oracle**, which is independent but asks the reader to accept that the
   other builder is right.


## 7. What the mutation pass reached

The piece field carries its own harness: `tests/test_gamma_field.py::test_the_mutant_is_killed`
applies each mutant to the source (`gamma_instruments.mutated`, which says DID-NOT-APPLY and
NO-OP out loud and restores by md5), reloads, and asserts that the named instrument fails —
the flush substitution, the pair substitution, the tie and its axis, one face family's image
list (killed by the L box and its transpose through the oracle band), the coverage union, and
the blend exponent. The second half added its own: the seam sign forced opposed (killed by the
K scene turned with its cell, whose seam axis the tie's `(1, 0)` fallback is 17 deg from), the
winner's weight reverted to the synthesis's `d_min^4` (the 350 golden), fixed normals replaced
by the foot direction (the grating's exact zero), one face's image list emptied (the L box),
the pair sweep reading a fixed 3x3 instead of the ring (a bar along `2 a1 + a2` on the square,
as long as that vector, whose flush contact sits at `(m, n) = (2, 1)` -- it stood on an unreduced
spelling's `(-2, 1)` contact until §8c made that spelling plan as its reduction), and `mu = None`
read as unequal to 1.0 -- that one run in
a fresh interpreter, because `domain.py` defines `Material`, which `as_material` type-checks
against and `engine` imports by name, so a reload would leave two classes of that name in one
process. Fourteen then; the verifiers' round added eight more -- the self substitution keyed by
instance alone (the double-covered face), the arc ring dropped and the ring picking by the
algebraic distance (the ellipse that leaves its cell), the sign window always open, the sign
keyed on either piece and the run window built from the partner's half-length (the A|B|C
scene and the column above the K seam's end), and the cross-pair substitution removed (the
off-centre stacked bars, exact) -- twenty-two, all killed, none DID-NOT-APPLY. The reduced basis
(§8c) added two: the replay folding on the basis as given (killed by the 693 circle on the
`3 a1` spelling, 8.89 deg on Gamma) and the planner reducing nothing (killed by the `10 a1`
box planning on ring 14 with 479 offsets where its reduction has 3 and 21). What follows is
the record of the pass over
the fold it replaced.

Eighteen mutants over the fold, the refusal and the tie-break, each run against
`test_nvf_period_spanning_shape.py` and `test_nvf_self_coincident_faces.py`, with the source
restored and md5-checked afterwards. **Seventeen killed, one survivor, and the survivor is
explained rather than outstanding.**

Killed: fold disabled; `floor` for `round`; the reciprocal genuinely unguarded (`1/0 -> inf`);
`merged_u` forced True; `merged_v` forced False; `a1 ± a2` dropped from the candidates; `on_u` /
`on_v` swapped; the direction tolerance to 1e-2 and to 0; the fill constant `(0,1)` for `(1,0)`;
the blend exponent 4 -> 2; the x-face pair's fold deleted; the `near` reduction collapsed to a
plain minimum; the merged test turned into a 7e-4 window; the tie-break removed; the coverage
test taken per image instead of per union; and the refusal made never to fire.

**The survivor is `_shorter_period` ranking the longer candidate instead of the shorter.** Its
own docstring says the branch never has two live candidates to rank, and that is checkable rather
than assertable: align a shape axis with each of `a1`, `a2`, `a1 + a2`, `a1 - a2` in turn over
800,000 random non-degenerate cells and exactly one candidate is ever live, never two. The
algebra says the same — if two of those four lie along one axis then so does their difference,
every such difference is `2*a1`, `2*a2` or `a1 ± a2`, and that forces `a1 ∥ a2`, i.e. a degenerate
cell that never reaches here. So the branch is unreachable and the `where` is an
order-independence guard, not a ranking anybody uses. A test for it would have to build a cell
the solver refuses.

Two of the eighteen had to be re-stated before they meant anything, which is the reason to record
the pass rather than only its score. "Reciprocal without its guard" first replaced the zero branch
with a one — but `P * round(s * 1)` is still zero when `P` is zero, so the mutation was inert and
"survived" said nothing; the mutation that bites is the unguarded division itself. And "refusal
never fires" was first written with its before and after identical, which the harness reported as
a no-op rather than counting as a kill. A mutation that does not apply, or applies to nothing, is
indistinguishable from a passing test unless the harness says so out loud.

## Explicit Cartesian alternative for rectangular contacts

Axis-aligned rectangular paintings can now select `factorization='li'` to bypass
the NVF normal blend and integrate live edge intervals exactly, including zero-width
contact gaps. See [supported geometry and JIT usage](parametric-modeling.md#cartesian-rectangles-with-explicit-li-factorization).
This is an explicit alternative for those geometries, not a change to the open J/K
NVF findings above or a claim of Fourier-order convergence for every structure.

## Rectangular contacts with circular NVF

`factorization='nvf', projector='circular'` now provides the non-cancelling normal-
dyad construction for rectangular shapes. It passes the period-gap, unequal-
material contact, material-equality, and frozen-contact JIT derivative checks.
See [usage and accuracy limits](parametric-modeling.md#complex-nvf-projection-for-rectangular-contacts).
The earlier J/K measurements describe the default linear field. Circular NVF
retains an open finite-order split/merged representation discrepancy and does not
close the curved-shape or resonant-convergence investigations.

### Curved circular extension, 2026-09-08

Disjoint circles and ellipses now have a circular NVF construction with compact
periodic level-set dyads and an isotropic extension away from the boundaries.
Tangent circle/ellipse and mixed rectangle/circle fixtures pass contact and JIT
derivative checks. The centre and periodic-image direction defects are checked
directly on the projector. See [the supported curved path](parametric-modeling.md#disjoint-circles-and-ellipses).
Overlapping curves now also have a polygon-facet correction of this extension;
see [curved painting support](parametric-modeling.md#overlapping-circles-and-ellipses).
The rectangular split/merged representation and tilted resonant-convergence
questions remain open. Crossing from disjoint analytic curves to polygon-resolved
painting also retains a finite polygon approximation difference; the correction
does not make that geometry switch exact. Earlier restricted scopes above record
the prior integration stages.


### Independent check of tilted resonant convergence, 2026-09-08

The optional S4 bridge now checks the actual square reciprocal-order set and
matches Wavebit's direct-rule transmission on the tilted contact fixture to
1.25e-12 or better at orders 6, 10, 14. This isolates the remaining difference
from the direct-rule Maxwell solve and the physical input conventions. The
circular projector also matches the exact normals on the regular visible faces
of this fixture, including periodic images. These checks do not certify Fourier
convergence: at order 22, grid 512, circular NVF gives T=0.853755019 and S4 direct
gives T=0.890772020 (difference 0.037017001). Neither is a converged reference. See the
[comparison, rejected alternatives and reproducible commands](nvf-contact-followup.md#independent-tilted-fixture-comparison-2026-09-08).

The newer [independent FEM study](nvf-independent-fem.md) caught a separate
reference-boundary bias: a short PML appeared stable under mesh refinement but
reflected a near-grazing substrate diffraction order. Its 0.876 transmission
plateau is excluded as a reference. The study now also checks a Rayleigh/DtN
boundary with both modal and volume flux readouts.

A [material-defined energy closure](nvf-harmonic-energy.md) is being evaluated
separately. It removes the tested split/merged difference and approaches the
independent p-polarized contact derivative as its internal electrostatic order increases,
but retains inner-order and resonant convergence errors. Its opt-in JAX study
now verifies first derivatives under JIT against independent NumPy secants,
including zero gap, zero contrast and a movable equal-material seam. At inner
order 255 that p derivative still differs from the independent reference by
about 0.54%. The expanded s-polarized check fails much more strongly: its slope
is about 12.25 times too small at every tested inner order. A conforming 2D cell
calculation now narrows the static energy enclosure by about 1,100 times,
separating the auxiliary-field error from the missing thin-gap normal-field
variation. [Measurements and remaining limits](nvf-cell-corrector.md).
A subsequent [conforming thin-gap formulation](nvf-thin-gap-energy.md) passes
both polarization contact checks on the rectangular control: the s derivative
error decreases to about 0.001% with interface refinement. It retains the gap
field in the actual forward energy and supports JIT gap/material/width
derivatives. General primitive partitions and tilted interfaces are not yet
supported by that new study.
A further [material-jump extension](nvf-partitioned-gap-energy.md) now preserves
equal-material partitions in the same contact model for translated horizontal
rectangle tilings. Its material and seam derivatives remain live at equality,
and both polarization contact gates continue to pass. A real interior material
interface away from mesh nodes still has measured cell error; nested mesh
refinement reduces it while retaining representation invariance.
The new independent static check also finds inaccurate one-sided derivatives
when an unequal material seam moves through a mesh node. This has its own strict
expected-failure gate and prevents adopting the new model as a general method.
These research instruments do not change the production NVF dispatch or close
its open limits.

The latest [material-harmonic coordinate study](nvf-harmonic-coordinates.md)
resolves the closed mixed-bar static seam error by moving the field jump
inside each fixed element and changing the affine lift and forward energy
consistently. Closed static and optical seam/material derivative checks now
pass against independent controls, alongside the partition and p/s contact
gates. At a positive gap, a smaller mesh-node sensitivity discrepancy remains:
the tested s one-sided slopes differ by 1.76e-6 per nm at the default mesh and
8.87e-7 after four x subdivisions, above the 1e-7 target. This has a separate
strict expected-failure test. General geometry and production adoption remain
open; the previous model's failure is retained as a control.

The subsequent [integrated continuous-flux basis](nvf-continuous-flux.md)
also makes the body displacement continuous across artificial mesh nodes.
The same positive-gap seam gate now passes: at the stated order-2 fixture,
the one-sided derivative difference is 2.26e-9 per nm. JIT agrees with both
sides, and the partition, contact and independent closed-laminate checks
remain passing. The old local-coordinate failure is retained for that old
model. Nested refinement still changes positive-gap transmission, so optical
convergence, general geometry and production NVF integration remain open.

The [finite-gap convergence extension](nvf-gap-convergence.md) now resolves
interior gap variation with hierarchical bubbles, defaulting to degree 4,
and supports nested body refinement. Contact and partition checks remain
passing. At the measured order-16 mixed rectangle, its p transmission differs
from the independently fitted cell projection by about 4.2e-6 instead of the
linear-gap model's 3.8e-5. The s difference remains about 6e-5 after the recorded
body refinement. The fitted reference's static enclosure is below 8e-7, but
outer-order transmission changes are still larger. Static bounds and interface
derivative checks do not close that optical convergence question; production
NVF and general geometry remain separate outstanding work.

The opt-in [material-defined mesh extension](nvf-interface-resolution.md)
now narrows the tested static enclosure by about 64% with the same number of
unknowns while preserving redundant partitions and contact/material/seam
derivatives. At 5183 unknowns and outer order 16, the s discrepancy against
the fitted-cell control decreases from 5.99e-5 to 4.45e-5. Independent controls
now reach outer order 20, but the fitted-cell p/s transmission still changes
by 2.50e-4 / 8.24e-5 from order 16. Thus the improved cell resolution has not
closed optical convergence, general geometry or production integration.

The [directional follow-up](nvf-directional-convergence.md) now varies the x/y
Fourier half widths independently and reaches square order 24. It finds that
the small s change from 20 to 24 contains opposing x/y contributions, so the
square sequence alone can mask unresolved error. A better allocation of cell
unknowns reduces the order-16 s difference against the fitted cell by another
60%, using 5039 instead of 5183 unknowns. The contact-edge setting also needs
refinement: edge_fraction=1e-4 misses the existing s contact gate, while 1e-5
passes it. Diagonal equilibration of the same equations removes the resulting
conditioning warnings without changing the forward formulation or stopping
gradients. These improvements retain the open optical and general-geometry
questions; they do not complete production NVF development.

The [polynomial body implementation](nvf-polynomial-body.md) now improves the
research model's forward potential with conforming interior y bubbles. At the
same 5039 unknowns it narrows the static enclosure by another 37% and reduces
the order-16 s discrepancy against the fitted-cell control by about 82%.
Independent integration, partition, JIT and contact criteria remain passing.
A finer 59969-unknown FEM cell changes the comparison values by only about
1e-8, so the measured improvement persists. The implementation remains in
`studies/`; it does not update the production NVF dispatch or establish outer
optical convergence or general-geometry support.

The [x polynomial-flux comparison](nvf-polynomial-flux.md) adds a tested
experimental space but finds no equal-dimension static advantage over more
intervals in the original linear-flux model. A finer setting of that original
model reduces the order-16 p/s fitted-cell discrepancy by another 55% / 49%,
using 50% more unknowns. Another allocation halves the static enclosure while
slightly worsening the s transmission comparison, emphasizing that the static
interval is not an optical error bound. Production behavior and the unresolved
outer-order/general-geometry gates are unchanged.

The [mirror-sector implementation](nvf-mirror-convergence.md) now reduces the
cost of the same finite NVF closure and propagation, with a measured 3.47
speedup at outer order 16. Directional and square measurements extend through
order 40. The p square increment is still about 5.3e-5, and the smaller s
square increment hides opposing x/y contributions. A separate Cartesian
factorization control shows similar drift; neither sequence is declared
converged.

The [scattering-doubling implementation](nvf-scattering-doubling.md) separately
addresses the approximately 6 ppm material-gradient bias in the existing
regularized eigenvector derivative. A short-slice series and scattering
composition avoid eigenvectors entirely; the recorded s difference from the
first NumPy secant falls to about 0.028 ppm and the original two-step gate
passes. Its opt-in polynomial-body adapter also passes JIT gap/seam/material,
partition, translation and independent p/s contact controls. Exact layer
graze is checked against an independent Jordan-limit slab calculation.
This remains a lossless single-layer research path: production derivatives,
the old Fourier auxiliary field's separate contact failure and the general
geometry/outer-convergence limitations are unchanged.

The [combined mirror/doubling path](nvf-mirror-doubling.md) now restricts the
same live closure and propagation under JAX, with unchanged symmetry and
physical acceptance criteria. It matches the modal values through order 24
and the refined slice count, and retains contact, material, partition and
layer-graze controls. The measured order-8 warm value-plus-material-scale
derivative time falls from 1.922 to 0.513 seconds; this excludes a new cell
solve and is not an end-to-end optimization speed claim. The tested NumPy
forward sweep is still faster with modal mirror propagation, which remains
the default. Production NVF, general geometry and outer optical convergence
remain open.

The [rigid-pose extension](nvf-posed-gap.md) adds laboratory translation,
rotation of the pattern together with its lattice, live incidence angles and
complex diffraction amplitudes to the conforming research model. Independent
phase, tensor-rotation, slab and ten-parameter derivative checks pass. It does
not rotate independent objects within a fixed periodic cell. Extending the
contact control to mixed materials exposed an additional cell-resolution
error at real interior seams. A [localized basis](nvf-local-flux.md) preserves
the same space while allowing sharper grading; the stated mixed-material JIT
contact fixture now passes both polarizations under the original 1e-7 per nm
criterion. Default coarse grading, other contact geometries and high-order
optical convergence are not certified by this result. Production integration
remains open.

The localized model's [expanded contact sweep](nvf-local-flux.md) now passes
27 profile/angle/order combinations, including two real material seams, after
explicit x refinement. A fixed exterior basis pivot avoids a further
conditioning problem on stronger contrasts. Factored integral assembly lowers
the measured component's peak process memory from 3.77 GB to .37 GB and its
time from 6.55 s to .75 s; this is not an end-to-end solver speed claim.
Sharper grading still worsens the tested positive-gap s comparison even when
its static energy enclosure improves. No universal grading setting or
converged optical reference is inferred from the contact results.

The [lower-memory energy closure](nvf-energy-closure.md) now avoids an explicit
NumPy inverse and bounds temporary Fourier assembly storage. The measured
order-32 closure peak process memory falls from 2.24 GB to .99 GB with an
action difference below 2.39e-13. Sparse parity exchange and an equivalent
bottom-interface elimination also reduce dense Maxwell work. The 59 passing
controls include complex/non-Hermitian algebra, unchanged contact and JAX
checks, and independent zero/thick slab limits. These are computational
improvements to the same finite research model, not an optical convergence
certificate or a production integration.

Square measurements now reach order 48: the p/s changes from order 40 are
-1.85e-5 / +1.25e-5, both above 1e-5. The s sequence reverses direction
again. The four-corner check finds partial p cancellation and reinforcing s
increments; both fail the combined finite increment criterion. A new
[half-cell FEM reference](nvf-mirror-reference.md) removes the
previous symmetry obstruction to independent high-order auxiliary-field
controls by solving the exact even/odd boundary spaces. Its cap-96 export
passes the original guard, quadrature refinement changes coefficients by
less than 8e-15, and order-16 transmission agrees with the earlier full-cell
FEM calculation within 7e-9. This adds an independent cell discretization;
it does not supply a 3D Maxwell reference or close optical convergence.

That independent cell comparison now reaches orders 40 and 48. The
candidate/FEM p/s transmission differences stay below 9.58e-7, while both
discretizations retain 40-to-48 changes above 1e-5. The observed remaining
drift in this fixture is therefore dominated by outer Fourier truncation.
The FEM optical process peaks at 6.95 GB. These measurements identify the
next refinement priority; general geometry, production integration and a
certified optical error bound remain outstanding.

The subsequent [modal/interface storage change](nvf-mirror-modal-memory.md)
reduces measured order-32 full p-sector peak memory from 2.12 to 1.43 GB,
with T differing by 3.27e-13. It retains every finite mode while eliminating
dense modal and boundary intermediates. The 76 passing modal/closure tests
include direct coupled-interface systems and propagating-root sign changes.
Directional checks now reach 56: p-x at fixed y=40 changes by -7.15e-6 from
48, but s-y at fixed x=40 changes by +1.19e-5. The latter still fails the
1e-5 finite increment criterion. These selected checks are not a completed
square/four-corner convergence study.

A further [FEM integration correction](nvf-mirror-reference.md#element-ownership-at-high-quadrature)
removes incorrect neighboring-material lookup at four high-quadrature points.
Both full- and half-cell reference exporters now retain each point's known
triangle. At cap 112, q32-to-q36 coefficient drift falls from 3.39e-11 to
3.11e-15; the old q32-based optical controls remain consistent within 4.87e-12
at order 16. The 13 optional FEM tests pass, including exact material Fourier
controls through order 112 on both domains. There are 89 distinct passing
checks across these two validation runs. This corrects the auxiliary
reference integration; it does not certify the remaining optical truncation
or supply a 3D Maxwell solution.

The s-y continuation now reaches `(40,64)` with T=0.9415489501558668 and
lossless residual 4.92e-13. Its last increment falls to +2.93e-6, but the
48/56/64 window still spans 1.48e-5. The small last interval does not resolve
the earlier oscillation or complete a two-axis convergence check. The
combined coefficient/optical process peaks at 6.04 GB of observed RSS. A
wider s-y window, the p-x window and missing mixed-order corners are
conditional precision-study followups, as recorded in the
[directional continuation](nvf-mirror-convergence.md#the-next-s-y-interval-through-64).

The subsequent [criterion review](nvf-mirror-convergence.md#meaning-and-status-of-the-convergence-target)
clarifies that the reviewed records do not establish the 1e-5 transmission
increment as a product accuracy requirement. It remains a provisional
research target: neither exceeding it over a finite window nor passing it
on the latest step determines the true optical error. It must not by itself
block all NVF development or trigger unbounded order increases. Numerical
integrity checks retain their thresholds; accuracy qualification and
integration of the new conforming energy construction remain separate work.
The earlier circular shape NVF is already integrated in production.

The conforming energy construction now has an explicit
[engine integration](nvf-energy-engine.md): the public
`ConvMatrices.from_energy()` closure and
[`ConformingNVF`](guide/conforming-nvf.md) construction from ordinary
rectangle layers. The cell kernels live in the library and are shared with
the research instruments. Nonoverlapping rectangle sequences along either lattice axis,
uniform layers, live backgrounds, independent orthogonal periods, translations,
common lattice rotations and unequal retained-order counts are supported.
The exported stack runs through the actual Wavebit engine on NumPy, JAX
and Torch; the auxiliary cell supports NumPy/JAX x64. The general closure
preserves loss and autodiff graphs, while the patterned cell builder requires
positive real parts and independent passive loss in every material.
Independent material integrals, contact controls,
slab formulas, basis invariance and layer splitting qualify the integration.
The [rectangular-cell metric](nvf-rectangular-cells.md) is present in both
the variational equations and projected energy. Period derivatives remain
live, including contact roundoff, with independent quadrature and
stationarity controls. The [layer-axis mapping](nvf-layer-axes.md) supports
mixed tiling directions in one stack and equivalent quarter-turn rectangle
representations, with a fixed axis during JAX replay. The
[background-interval extension](nvf-separated-rectangles.md) supports separated
rectangles with equal perpendicular extent and center, including live gap
closure. Signed log-jump grading removes the artificial interface left by
the earlier independent-jump density at a collapsed interval. Automatic projector
selection, general corrector geometry and optical accuracy qualification
remain open. The
[common-loss construction](nvf-common-loss.md) supports absorbing patterned
layers with a shared ratio and independent absorbing uniform layers, with
lossless-boundary and contact derivatives, passivity, analytic contact
matrices and independent slab controls.

The [complex-material construction](nvf-complex-material.md) now permits
independent loss ratios within one patterned layer, including a lossy
inclusion in a lossless background. It uses the complex weak equations and
sesquilinear energy projection; it does not discard loss or substitute a
Li factorization. Nonpositive real permittivities remain unsupported.

The [lattice-rebasing extension](nvf-lattice-rebasing.md) accepts skew basis
vectors that describe the same orthogonal point lattice. It preserves
physical rectangles, the supplied cell-center convention and the exact
original Fourier set. The internal [full-metric affine cell](nvf-affine-cells.md)
now handles affine images of reference strip profiles on genuinely
nonorthogonal lattices, including their mixed weak terms and exact laminated
contact lift. Such a shear turns reference rectangles into parallelograms;
it is not applied to ordinary physical rectangle dictionaries. General
physical-rectangle input and oblique optical accuracy qualification remain
open.

The [independent affine comparison](nvf-affine-convergence.md) now separates
auxiliary-cell refinement from directional optical truncation. A fitted
static FEM reference reaches a relative primal/complementary enclosure of
1.43e-7 after degree-10 refinement on the tested profile. At optical order
(24,24), the refined candidate differs from that reference by about 4.3e-6
in p transmission and 1.1e-5 in s. The fixed degree-8 reference's 20-to-24
order changes remain larger: about 2.0e-5 and 5.9e-5, respectively. The
complete directional rectangles still identify y as the largest measured
s-polarized optical limitation. These are comparisons, not an independent
three-dimensional optical error bound or a release criterion.

The affine continuation through y half width 32 exposes a reversal in the
s transmission sequence. At x=20, the y=24-to-28 change is `+6.45e-6`,
followed by `+2.92e-5` from y=28 to 32; the preceding y=20-to-24 change
was negative. x=24 confirms the pattern, with changes of `+6.50e-6`
and `+2.93e-5`. Thus the small change near y=28 does not establish a
monotone tail or a stopping criterion. The auxiliary field remains fixed.
The study's full-space closure now uses tiled assembly and an exact
two-sided solve, reducing measured order-24 closure peak memory from
3.43 GB to .96 GB. The memory result concerns that stage, not the full
Maxwell solve or improved optical accuracy.

The affine cell now factors its vanishing body correction by the gap,
removing a source of contact-derivative roundoff. An
[independent contact variation](nvf-affine-cells.md#contact-fourier-variation-without-subtracting-nearby-solutions)
checks every projected component through coefficient cap 2 for four real
or complex profiles and mesh configurations. This verifies differentiation
of those finite cells; continuum derivative convergence and oblique optical
contact accuracy remain open.
