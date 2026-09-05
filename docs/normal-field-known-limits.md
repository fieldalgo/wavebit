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


## 1. The polygon route still reads an abutting face as an interface — OPEN

**Status: open.** The parametric route was partly fixed on 2026-09-03 (§5 below; §2 for what is refused instead and §3 for what it left). The polygon route was
not, and it is a different builder with a different mechanism, so it did not inherit the fix.

A shape as long as the cell repeat along one of its own axes abuts its own periodic images there:
the material runs on unbroken and there is no interface. `boundary.py`'s
`distance_gradient_normal_field` measures distance to polygon **edge segments**, replicated across
images by `periodic_segments`, and the abutting edge is still present as a segment. So the field
picks up a normal component perpendicular to an interface that is not there.

**Which scenes reach it.** The polygon route is taken only when a layer's shapes **overlap** —
`parametric.layer_needs_resolving` decides, and a non-overlapping layer stays on the closed form.
So a scene is affected when it has *both* an overlap *and* a shape spanning the cell.

**How far wrong.** A y-invariant layer cannot depend on how many y orders it is retained with:
orders along a direction the structure does not vary in can only carry zero. The scene, stated in
full so the rows can be re-run: two rectangles `lx = 300`, `ly = 700` (spanning), `cx = -80` and
`cx = +80` so they overlap, `angle = 0`, material 4.0 on background 1.0, one layer 300 thick,
wavelength 550, period 700 x 700, `psi = 0`, normal incidence, NumPy, float64. `sum(T)` at
`truncation_order` `(n, 0)` against `(n, n)`:

| `to_x` | `to_y = 0` | `to_y = to_x` | violation |
|---|---|---|---|
| 3 | 0.822541442 | 0.822493209 | 4.8e-5 |
| 5 | 0.810008546 | 0.810031937 | 2.3e-5 |
| 7 | 0.799804804 | 0.799811517 | 6.7e-6 |

The comparison is the same scene with the bars moved to `cx = -180` / `cx = +180`, which stops
them overlapping and so takes the parametric route instead — same shapes, same span, same
invariant: 4.9e-15, 7.0e-15, 7.3e-15. That route violated it by 1.1e-5 before its own fix.

**What to do instead.** State the layer as a raster structure, which carries the wrap, or arrange
the geometry so no spanning shape takes part in an overlap.


## 2. A face the fold cannot merge is refused, not built — REFUSED

**Status: refused.** The fold (§4) merges a face family when a lattice vector runs **along one of
the shape's own axes**. That is not the condition for a family to merge, only the condition the
fold can act on, and the true condition is strictly weaker.

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

**It does not converge away, which is what makes it a defect.** `sum(T)` for the tilted box at
`lx = 640` — 13.9 of a 640-long face, 2% of one face — against two independent instruments that
agree with each other to 1.2e-4: an exact nearest-interface field injected into the same
shape-path solve, and a raster of the 3x3 image union on the same lattice.

| truncation order | 6 | 9 | 12 | 15 | 18 |
| --- | ---: | ---: | ---: | ---: | ---: |
| shape path | 0.801261 | 0.811221 | 0.823352 | 0.832240 | 0.838137 |
| exact-interface field | 0.832096 | 0.841068 | 0.849004 | 0.853464 | 0.856906 |
| image-aware raster | 0.832010 | 0.840954 | 0.848972 | 0.853461 | 0.856791 |
| error | 3.1e-2 | 3.0e-2 | 2.6e-2 | 2.1e-2 | 1.9e-2 |

3e-2 falling to 1.9e-2 across a tripling of the order is an error, not a truncation. Energy is
conserved throughout and the permittivity is *right* — the boxes only touch, so the sum of their
transforms is the transform of the union. The normal field is the entire difference.

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

So the geometry is refused, by `overlap.partial_self_coincidences` from `check_layer_shapes`. The
refusal names how much of the face is covered, since that is what a caller acts on. State such a
layer as a raster structure, which carries the wrap; or keep the shape's axes on lattice
directions, where the fold fires; or keep every extent at or below half the corresponding repeat,
where no family can merge at all.

The refusal is asked **after** the interpenetration sweep, and that ordering is load-bearing: a rod
of `lx = 340` on a 300 period is partially coincident *as well as* larger than the period, and the
latter names its actual mistake.

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

**And it exposes a floor under the candidate-set argument.** `_axis_lattice_periods` asks which
lattice vectors run *along* a shape axis, and matching `overlap._IMAGES` fixes **which images
exist** — it does not fix **how many it takes to cover a face**. A face covered by two off-axis
vectors taking half each is outside what a per-vector search can express, which is the same shape
of limitation as missing `t . u != 0`, one level up. `test_a_rhombic_cell_spans_along_the_sum_of_
its_basis_vectors` argues for the candidate set more broadly than the set can deliver.

`partial_self_coincidences` accordingly unions the covered intervals per face family and reports
only what the union leaves uncovered. Neither the fold nor the first refusal had this: the fold
reads `merged_v = False` there and keeps two faces that are not interfaces, which is harmless for
the reason above but is not what the geometry says.

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

**The fix breaks the tie instead of changing the removal.** Where the blend lands on zero and
exactly one pair is gone, substitute the surviving axis — `(0, 1)` when the u pair went, `(1, 0)`
when the v pair did. It needs no sign, because the factorization reads `n n^T`. Array-only, so it
survives `jit` and `grad`, and the exact-zero comparison is safe because the tie is an exact
symmetry rather than a near-miss.

Measured bit-identical on a small box, a box above half the repeat, an ellipse, the 1-D grating,
its 90-degree mirror, a cell-filling slab and a two-shape layer; the checkerboard scene goes
256 -> 0 null normals with `max|n_x|` still 0.000000, which is correct — that direction genuinely
has no interface. Pinned by
`tests/test_nvf_period_spanning_shape.py::test_a_merged_pair_leaves_no_null_normal_behind`.

**What the drop was worth even before this.** At those same points the pre-removal field was
`(±1, 0)` — pointing hard along the one axis that has no interface at all, so `n n^T` was `x x^T`
where the truth is `v v^T`. Removing the pair moved them from confidently wrong to null, which is
already an improvement for a factorization that reads the outer product; the tie-break finishes it.


## 4. The lattice-direction test is a threshold, not an exact question

**Status: by construction, stated rather than removed.**

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
it. `tests/test_nvf_period_spanning_shape.py::test_the_lattice_direction_test_is_a_threshold`
pins it.


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

**A predicate**, reading the same values the fold reads so it cannot drift from it:

```python
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
