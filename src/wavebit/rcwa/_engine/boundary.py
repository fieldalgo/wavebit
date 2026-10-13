# -*- coding: utf-8 -*-
"""Convolution matrices for a layer whose overlap has been resolved into polygons.

The third modeling path, and the only one that gets both halves of the problem right at once:

===============  ==================  =====================
route            tilted edge         overlapping shapes
===============  ==================  =====================
raster           staircase           as drawn (painted)
parametric       **exact**           summed
**this one**     **exact**           **painted**
===============  ==================  =====================

A raster paints -- the later sample wins -- but staircases a tilted edge. The parametric path
transforms a tilted shape in closed form but *sums* the shapes, so an overlap would get both
contributions and end up as neither shape's material; it refuses one rather than answering.
Neither gap can be closed by mixing the two: the analytic normal field never asks which
boundary survived.

This path resolves the painting **once** into disjoint polygons (:mod:`.clipper`), transforms
those in closed form (:func:`.polygon.polygon_ft`), and builds the normal field from the same
polygons -- so the boundary the field follows is the boundary that survived.

Split in two halves, which is what makes it differentiable:

* **planning** -- which vertices a loop has, which edges cross. Discrete, NumPy, outside any
  trace (:mod:`.clipper`).
* **replay** -- everything here. Arithmetic only, no value-dependent control flow, no in-place
  writes, so it traces and differentiates on every backend.

Validated against S4 (an independent RCWA implementation): the permittivity coefficients agree
to 2.2e-16, and on a case with both tilt and overlap this path sat 4x closer to it than the
removed block path, while tying on axis-aligned overlap, where a staircase has nothing to
lose.
"""

from wavebit._common.geometry.periodic import fold_into_cell
import wavebit._common.geometry.clipper as _clipper
from wavebit._common.geometry.interfaces import plan_interfaces
from wavebit._common.geometry.interfaces import replay_interfaces
from wavebit._common.geometry.interfaces import interface_contrasts
from wavebit.rcwa._engine.parametric import nvf_from_coeffs
from wavebit._common.geometry.recipes import shape_geometry
from wavebit.rcwa._engine.parametric import _sampled_coeffs
from wavebit.materials.response import as_material
from wavebit._common.backend import checked_float
from wavebit.rcwa._engine.factorization import check_normal_grid
from wavebit.rcwa._engine.geometry import StaleGeometryError
from wavebit.lattice import reduced_basis
from wavebit.lattice import reduction_transform
from wavebit.lattice import sample_grid
from wavebit._common.geometry.polygon import ellipse_verts
from wavebit._common.geometry.polygon import polygon_ft
from wavebit._common.geometry.polygon import rect_verts


def _sum(xp, a, axis=-1):
    return xp.sum(a, axis=axis)


def segments_of(xp, polys):
    """Concatenated (start, edge vector, |edge|^2) for every loop's every edge."""
    A = xp.concat([V for V in polys], axis=0)
    B = xp.concat([xp.roll(V, -1, axis=0) for V in polys], axis=0)
    E = B - A
    L2 = _sum(xp, E * E)
    # a zero-length edge would divide by zero; it also cannot be anyone's nearest point
    L2 = xp.where(L2 > 0, L2, xp.ones_like(L2))
    return A, E, L2




def periodic_segments(xp, A, E, L2, lattice, transform=None, images=None, image_groups=None):
    """Fold segment starts and enumerate images on the reduced lattice.

    E and L2 are translation invariant. Folding the start rather than the
    point-to-start displacement lets the minimum select the image nearest the
    segment, including near its far endpoint. Samples must fold on the same
    reduced cell (distance_gradient_normal_field does this).

    Production passes images and image_groups from interfaces.plan_interfaces.
    The image ring is bounded by fractional edge extents; groups drop only
    segment/image combinations farther than a guaranteed nearest-endpoint upper
    bound. Arbitrarily long segments therefore do not rely on a fixed 3x3 ring.
    The default 3x3 remains for legacy unplanned field probes; callers with long
    segments must supply their image bound explicitly.

    transform is the integer basis reduction recorded while geometry was
    concrete. Replaying it preserves JIT over a traced lattice and avoids an
    unreduced spelling such as a2 = 10*a1 + (50, 700) changing which physical
    images are searched. Fourier coefficients retain the basis as supplied.
    """
    lattice = reduced_basis(lattice, transform)
    a1x, a2x = lattice[0][0], lattice[0][1]
    a1y, a2y = lattice[1][0], lattice[1][1]

    folded = fold_into_cell(xp, A, lattice)
    ax, ay = folded[:, 0], folded[:, 1]

    if image_groups is not None:
        indices = [xp.asarray(ids, dtype=xp.int64) for ids in image_groups]
        shifted = [folded[ids] + xp.stack([xp.asarray(u * a1x + v * a2x),
                                          xp.asarray(u * a1y + v * a2y)])
                   for (u, v), ids in zip(images, indices)]
        return (xp.concat(shifted, axis=0), xp.concat([E[ids] for ids in indices], axis=0),
                xp.concat([L2[ids] for ids in indices], axis=0))
    xs, ys = [], []
    if images is None:
        images = tuple((u, v) for u in (-1, 0, 1) for v in (-1, 0, 1))
    for du, dv in images:
        xs.append(ax + (du * a1x + dv * a2x))
        ys.append(ay + (du * a1y + dv * a2y))
    A9 = xp.stack([xp.concat(xs, axis=0), xp.concat(ys, axis=0)], axis=-1)
    E9 = xp.concat([E] * len(images), axis=0)
    L29 = xp.concat([L2] * len(images), axis=0)
    return A9, E9, L29


def _folded_sample_points(xp, grid_shape, lattice, dtype, transform):
    """The same reduced-cell quadrature for both nearest-interface extensions."""
    X, Y = sample_grid(xp, grid_shape, lattice, dtype)
    pts = xp.stack([xp.reshape(X, (-1,)), xp.reshape(Y, (-1,))], axis=-1)
    pts = fold_into_cell(xp, pts, reduced_basis(lattice, transform))
    return pts


def _seg_dist(xp, pts, A, E, L2):
    """Squared point-to-segment distance and the foot, for every (point, segment) pair.

    Elementwise across the segment axis on purpose: column ``k`` depends only on segment ``k``,
    never on how many segments were passed, which is what lets a pruned subset give bit-identical
    numbers.
    """
    d = pts[:, None, :] - A[None, :, :]
    t = _sum(xp, d * E[None, :, :]) / L2[None, :]
    t = xp.clip(t, 0.0, 1.0)
    foot = A[None, :, :] + t[:, :, None] * E[None, :, :]
    diff = pts[:, None, :] - foot
    return _sum(xp, diff * diff), foot


def _nearest(xp, pts, A, E, L2):
    dist2, foot = _seg_dist(xp, pts, A, E, L2)
    j = xp.argmin(dist2, axis=1)
    rows = xp.arange(pts.shape[0])
    return j, foot[rows, j]


def distance_gradient_normal_field(xp, polys, grid_shape, lattice, dtype, chunk=None,
                                   transform=None, segment_data=None, images=None,
                                   image_groups=None):
    """Unit normal from the direction to the closest boundary *point*.

    With segment_data, only live material-jump intervals compete in the
    distance minimum, everywhere in the cell. All-inactive sets return (1, 0).
    A closed periodic set of parallel interfaces uses its exact line normal.
    The optional image_groups carry the planner's conservative per-segment
    image pruning; the scan/checkpoint evaluation is shared with raw probes.

    The closest *edge* is not well defined -- beyond a corner both edges meeting there have the
    corner as their closest point, at exactly equal distance, so `argmin`'s tie-break decides on
    29-45% of the cell. The closest point is unique except on the medial axis (measure zero).

    Sign flips inside/outside, which does not matter: NVF only uses ``normal_x**2``, ``normal_x*normal_y``,
    ``normal_y**2``.

    The sample points come from :func:`wavebit.lattice.sample_grid`, shared with the
    parametric builder — this body was written by copying that one, and copying it is how it
    inherited a reversed ``period`` unpack that sampled over a transposed cell. It is also how
    it *failed* to inherit the periodic fold the analytic field was given on 2026-08-17: the
    copy carried a bug across and then missed the fix. :func:`periodic_segments` is that fix in
    this representation, and it is applied here rather than in :func:`_seg_dist` for the reason
    given there.

    ``chunk`` is how many pixels are handled at once. It bounds the ``pixels x segments``
    temporaries without changing any result -- the arithmetic per pixel is independent, so a
    chunked run is bit-identical to a flat one. ``None`` targets 4M entries per temporary;
    measured peak is about 5x that, since `_seg_dist` has several ``(chunk, M, 2)`` arrays
    alive at once. Peak then stops growing with the grid, which is the point.

    **That last sentence was false under ``jax.jit``, and the loop below is why.** A Python
    ``for`` does not survive tracing: it runs once per chunk while the program is recorded and
    appends its equations each time, so the recorded program holds one copy of the body per
    chunk and no loop at all. Counting ``argmin`` primitives in ``jax.make_jaxpr`` of the old
    path gives exactly ``ceil(npix / chunk)`` -- 24 copies at ``shape_nvf_grid_size=1024`` on the
    overlapping-pair scene of ``test_frozen_geometry``, where the fold leaves 90 segments and
    so ``chunk`` is 44444. Each copy holds several ``(44444, 90, 2)`` temporaries and the
    compiler cannot reuse one copy's buffers for the next, so the program grows with the grid
    **before any derivative is involved**: compiled forward-only, with no AD anywhere in it,
    the loop needs 0.31 / 0.91 / 3.55 GiB at grids 256 / 512 / 1024 against a flat 0.21 /
    0.22 / 0.24 for the scan. Reverse mode then multiplies that by about three, which is how
    3.55 becomes 10.77. An earlier version of this note blamed the backward pass alone; it is
    an aggravating factor and unrolling is the cause, and the forward-only column is what
    separates them. Measured with ``memory_analysis()``: 0.78 GiB at
    grid 256, 2.83 at 512, **10.77 at 1024** -- growing with the grid exactly as the chunking
    promised it would not. On GitHub's 7.9 GB runners that is
    `RESOURCE_EXHAUSTED: Out of memory allocating 11559862304 bytes`.

    Two dozen fat copies, then, not one flat table. An earlier version of this note said 361
    copies of a ``1024**2 x 1378`` table, and both numbers were arithmetic rather than
    measurement: 1378 is ``11559862304 / (1024**2 x 8)``, i.e. the runner's byte count divided
    by the pixel count. But that byte count *is* XLA's whole temp allocation -- the same
    quantity as the 10.77 GiB above -- so recovering a segment count from it is circular. The
    scene has 90 segments and the real table would be 0.70 GiB.

    So JAX takes :func:`lax.scan` with the body under :func:`jax.checkpoint`, and it needs
    both. ``scan`` puts a real loop in the recorded program -- one instruction instead of one
    copy per chunk -- which lets the compiler reuse the body's buffers; that alone gets 10.77
    GiB to 4.73, since scan's reverse still stacks each iteration's residuals. ``checkpoint``
    drops those and recomputes the chunk when the backward reaches it, which is cheap for one
    chunk and is not for two dozen: **0.57 GiB, and flat at 0.54 / 0.55 / 0.57 across grids
    256 / 512 / 1024.** Neither half is redundant -- `checkpoint` on the unrolled loop, with
    no `scan`, measures 12.19 GiB, worse than doing nothing. The promise the paragraph above
    makes is now kept under ``jit`` too.

    Same arithmetic, so same numbers -- **the compiled field, under ``jit``**, and that is the
    whole of what is promised. Bit-identical to the loop's across 426 audited cases: grids 64
    through 1024, six scenes, square and skewed lattices, chunk sizes that divide the grid and
    that do not. The 35 one-ULP disagreements in that sweep all belong to the *old* code --
    every one is `old@chunk != old@single-chunk` while `new@chunk == new@single-chunk`, and
    there is no case the other way round. ``scan`` gives every chunk the same shape, so it is
    less chunk-sensitive than the loop was, not more.

    **The gradient is not bit-identical, and neither was the loop's.** With one chunk it is,
    exactly. With more than one the cotangents are summed in a different order and the answers
    part company in proportion to the chunk count -- 5.4e-11 at grid 128, 6.5e-11 at 256,
    1.3e-09 at 1024. Those are all *smaller* than the spread the old code already had against
    itself when only ``chunk`` changed (1.9e-10, 5.9e-10, 3.7e-09 on the same scenes). So the
    defensible claim is agreement inside a band the loop already occupied, not agreement to
    every digit, and an earlier version of this note said the latter.

    **Eagerly the field moves too.** ``lax.scan`` compiles its body, so an untransformed call
    picks up XLA's rounding where the loop gets op-by-op dispatch: 5.1e-14 at grid 64 rising
    to 8.6e-11 at 1024. Isolated rather than assumed -- dropping ``checkpoint`` changes the
    forward bytes not at all, and eager-``scan`` equals *jit* of the Python loop exactly, so
    the new eager answer is the number the old code already produced once compiled. The
    axis-aligned overlapping pair reads bit-identical in both modes, which is exactly why the
    unqualified claim looked true: it is the one scene where it is.

    Nothing is wrong with those values. The difference scales as ``1/mag``, a fixed absolute
    error in ``v`` amplified by the division, with ``diff * mag`` clustering at two constants;
    no pixel goes near the ``mag <= 1e-13`` guard, unit norm holds to 3.3e-16, and end-to-end
    ``sum(T)`` moves by at most 1.7e-14 relative -- inside the band the three backends already
    occupy against each other. It is also not a new class of error: the loop already differed
    from *itself* by the same order between eager and compiled, 5.6e-13 on that scene, so this
    is the pre-existing eager-versus-jit fusion spread and not something ``scan`` introduced.
    Curved and tilted scenes select for it because they put more pixels at small ``mag``.

    Recomputing is not approximating; the same inputs give the same values, and what moves is
    which engine evaluates them.

    Two divergences that no public call can reach, recorded because they are real: with
    ``npix == 0`` the JAX branch returns empty arrays where every other path raises
    (``check_normal_grid`` rejects a zero grid first), and a non-positive ``chunk`` raises
    ``ZeroDivisionError`` or ``TypeError`` here against the loop's ``ValueError``. ``chunk`` is
    not a parameter of ``normal_projector_coeffs`` or ``conv_mats``, so neither is reachable
    except by calling this function directly.

    NumPy and PyTorch keep the loop. NumPy is immune because it never *traces*, not because
    `SolverNumpy.grad` raises -- the growth above appears under `jit` with no differentiation
    at all, so "it does not differentiate" would be the wrong reason for the right conclusion.
    Torch is a different matter: it does differentiate and it does keep the unrolled loop, so
    the same class of problem plausibly sits there unmeasured. Its answer would be
    `torch.utils.checkpoint`, a different call with its own caveats, and nobody has yet shown
    it is needed.

    (`cost_analysis()` cannot arbitrate this, and the way it fails is worth knowing because it
    fails *towards* the answer you want. It counts a scan body once rather than once per trip,
    so as the same work is split into more chunks the reported flops FALL -- measured at grid
    1024, 1.116e8 at 16 chunks against 1.091e7 at 256, a factor of ten in the direction of
    "the loop got better". The same sweep over the Python loop is flat, 1.08e8 both times,
    because there the work really is all in the trace. That contrast is the check; a single
    scan number means nothing on its own. Read `memory_analysis()` and the field itself.)

    ``transform`` is the recorded basis reduction, handed to :func:`periodic_segments`; the
    sample points are folded into the reduced cell by the same rule (:func:`fold_into_cell`),
    which is what makes the two agree on a spelling whose sample grid lies outside it.
    """
    Ny, Nx = grid_shape
    if not polys:
        return xp.ones(grid_shape, dtype=dtype), xp.zeros(grid_shape, dtype=dtype)
    if segment_data is None:
        A, E, L2 = segments_of(xp, polys)
        active = None
    else:
        A, E, L2, active = segment_data
    count = A.shape[0]
    A, E, L2 = periodic_segments(xp, A, E, L2, lattice=lattice, transform=transform,
                                images=images, image_groups=image_groups)
    M = A.shape[0]
    if active is not None:
        if image_groups is None:
            active = xp.concat([active] * (M // count))
        else:
            active = xp.concat([active[xp.asarray(ids, dtype=xp.int64)] for ids in image_groups])

    pts = _folded_sample_points(xp, grid_shape, lattice, dtype, transform)

    threshold2 = 1e-26
    if active is not None:
        scale = xp.max(xp.abs(xp.stack([xp.asarray(lattice[i][j], dtype=dtype)
                                       for i in (0, 1) for j in (0, 1)])))
        threshold2 = (64 * xp.finfo(dtype).eps * scale) ** 2
        reference = E[xp.argmax(xp.astype(active, xp.int32))]
        cross = E[:, 0] * reference[1] - E[:, 1] * reference[0]
        parallel = xp.all((cross == 0) | ~active)
        reference2 = xp.sum(reference * reference)
        line_normal = xp.stack([reference[1], -reference[0]]) / xp.sqrt(
            xp.where(reference2 > 0, reference2, xp.ones_like(reference2)))

    npix = Ny * Nx
    if chunk is None:
        chunk = max(1, min(npix, 4_000_000 // max(M, 1)))

    def _chunk_normals(p):
        """One chunk's unit normals, ``(len(p), 2)``. The loop body, named so JAX can reuse it."""
        if active is None:
            j, foot_j = _nearest(xp, p, A, E, L2)
        else:
            dist2, foot = _seg_dist(xp, p, A, E, L2)
            dist2 = xp.where(active[None, :], dist2, xp.asarray(float('inf'), dtype=dtype))
            j = xp.argmin(dist2, axis=1)
            foot_j = foot[xp.arange(p.shape[0]), j]

        v = p - foot_j

        # A pixel centre exactly on the boundary has no direction; grid-aligned designs hit it.
        # Fall back to that edge's own normal, which is what the limit approaches.
        #
        # The guard is on the SQUARED magnitude, and it has to be. Written the obvious way --
        # `mag = sqrt(sum(v*v))` and then `where(mag > 1e-13, mag, 1)` -- the value is right
        # and the derivative is not: `sqrt` is evaluated for every pixel before the `where`,
        # reverse mode sends a zero cotangent down the unselected branch, and `d/dx sqrt(x)`
        # at zero is infinite, so the chain rule computes `0 * inf` and the whole parameter
        # derivative comes back NaN. One poisoned pixel is enough, because the loss sums.
        #
        # It is reachable from a plain solve, not just a contrived call. `sample_grid` puts
        # centres at `(i + 0.5) * P / N`, so at `period=300, shape_nvf_grid_size=300` those are the
        # half-integers and a rectangle at `cx=0` with an ODD width lands its edges exactly on
        # a column of them. Measured through `rcwa(...).solve()` on two overlapping
        # rectangles: `lx=121` gives `sum(T)=0.9768138981` and `d/dlx = nan`, while `lx=120`
        # gives a perfectly ordinary `-7.085e-05`. An integer width, a correct transmission,
        # and a derivative that is not there.
        #
        # Guarding `s2` before the square root keeps `sqrt` off zero entirely, so nothing
        # infinite is ever differentiated. The comment above this block already said
        # "grid-aligned designs hit it"; it was describing the trigger for a bug in the
        # gradient, not only a special case in the value.
        s2 = _sum(xp, v * v)
        safe2 = xp.where(s2 > threshold2, s2, xp.ones_like(s2))
        mag = xp.sqrt(safe2)

        ej = E[j]
        n_edge = xp.stack([ej[:, 1], -ej[:, 0]], axis=-1)
        edge2 = _sum(xp, n_edge * n_edge)
        n_edge = n_edge / xp.sqrt(xp.where(edge2 > 0, edge2, xp.ones_like(edge2)))[:, None]

        n_far = v / mag[:, None]
        on = (s2 <= threshold2)[:, None]
        result = xp.where(on, n_edge, n_far)
        if active is not None:
            # A closed periodic interface set with parallel edges consists of
            # complete straight lines. Use its exact normal, including on segment
            # joins; no endpoint-radial extension or transverse roundoff belongs
            # to a genuinely one-dimensional layer.
            result = xp.where(parallel, line_normal[None, :], result)
            result = xp.where(xp.any(active), result, xp.asarray([1., 0.], dtype=dtype))
        return result

    if 'jax' in xp.__name__:
        n = _scanned_normals(xp, _chunk_normals, pts, npix, chunk)
    else:
        parts = [_chunk_normals(pts[lo:lo + chunk]) for lo in range(0, npix, chunk)]
        n = xp.concat(parts, axis=0)

    normal_x = xp.reshape(n[:, 0], (Ny, Nx))
    normal_y = xp.reshape(n[:, 1], (Ny, Nx))
    return normal_x, normal_y


def _scanned_normals(xp, body, pts, npix, chunk, width=2):
    """``body`` over ``pts`` in chunks, as a ``lax.scan`` whose body is rematerialised.

    The default output width is two real normal components; polygon Jones
    construction uses the same bounded scan for four complex components.

    Why not the plain loop: see :func:`distance_gradient_normal_field`. In one line, a Python
    loop is one copy of the body per chunk in the traced program -- 24 at the grid this was
    written for -- and this is one instruction.

    The tail is padded by repeating the last point rather than with zeros, and trimmed off
    before returning. ``scan`` needs a rectangular stack, the pixels are independent so a
    repeat changes nothing about the real ones, and repeating a point that is already in the
    set cannot reach a code path the real points do not.

    That last clause is the load-bearing one, and **what it protects is the gradient, not the
    field**. Padding with zeros, or with the first point, or with NaN, all give a forward field
    identical to this one -- the trim discards those rows before anything looks at them. The
    difference only appears under ``grad``: measured on a rectangle with a vertex exactly at
    the origin, zero-padding and NaN-padding both return a NaN gradient while repeating a real
    point does not, and the forward comparison between all four is 2.2e-16. So a ``.tobytes()``
    check on the field can never catch a wrong padding value. Anything that changes what the
    padding contains has to be checked under a derivative.

    ``chunk`` is clamped to ``npix`` on entry for a separate reason; see there.
    """
    import jax

    # `chunk` is documented as the bound on the temporaries, so a caller who wants no
    # chunking at all passes something enormous -- `tests/rcwa/test_boundary.py` uses `10 ** 9`
    # as exactly that idiom. The Python loop was indifferent to it: `range(0, npix, chunk)`
    # runs once over `npix` rows however large the step. `scan` is not. Without this clamp
    # the padded stack is `n_chunks * chunk` rows rather than `npix`, which is linear in a
    # number the caller chose to mean "no limit": measured at npix 256, the compiled
    # `jit(grad)` needs 0.0012 GiB at chunk 256 and 48.73 GiB at chunk 10**7, while the loop
    # stays at 0.0012 GiB across all of it. At `10 ** 9` it builds a billion-row array and
    # the process is killed. Clamping inverts nothing: a chunk at least as large as the grid
    # already means one pass.
    chunk = min(chunk, npix)

    n_chunks = -(-npix // chunk)
    pad = n_chunks * chunk - npix
    if pad:
        pts = xp.concat([pts, xp.repeat(pts[-1:], pad, axis=0)], axis=0)
    stacked = xp.reshape(pts, (n_chunks, chunk, 2))

    _, out = jax.lax.scan(lambda carry, p: (carry, jax.checkpoint(body)(p)), None, stacked)
    return xp.reshape(out, (n_chunks * chunk, width))[:npix]


def polygon_dyadic_projector(xp, pts, polys, lattice, dtype, transform, ring, chunk=None,
                             reference=None, curved_edges=None):
    """Complex projector of polygon edges at Cartesian points, ordered xx,xy,yy,yx.

    Each original polygon edge contributes its positive normal dyad, weighted
    by length and inverse fourth power of periodic segment distance. Clipping
    only shortens these same lines, so every regular painted interface has its
    exact polygon normal. Covered edges remain extension geometry, as primitive
    faces do in rectangular circular NVF; material equality changes no field.

    Fold point-to-midpoint displacements, then minimize over the image ring.
    An edge's normal is identical at every image, so a nearest-image tie does
    not switch directions. On a reduced basis a nearest lattice point lies in
    the 3x3 neighborhood of rounded fractional coordinates. The midpoint-to-
    foot displacement is bounded by R, the primitive bounding radius; therefore
    1+ceil(R/h) images per axis suffice. GammaPlan.ring=1+ceil(2R/h) covers them.

    With ``reference``, blend its positive tensor with the polygon tensor only
    near curved facets. The blend is exactly polygonal on a curved facet and
    tends to the analytic extension away from it and under polygon refinement.
    ``curved_edges`` flags the corresponding original edges; covered edges are
    retained, so neither the blend nor the reference reads material equalities.
    """
    if not polys:
        one = xp.ones((pts.shape[0],), dtype=dtype)
        return xp.stack([one/2, -.5j*one, one/2, .5j*one], axis=-1)
    polys = [xp.astype(v, dtype) for v in polys]
    A, E, _ = segments_of(xp, polys)
    length2 = _sum(xp, E*E)
    live = length2 > 0.
    safe2 = xp.where(live, length2, xp.ones_like(length2))
    length = xp.sqrt(safe2)
    tx, ty = E[:, 0]/length, E[:, 1]/length
    nx, ny = ty, -tx
    extent = xp.where(live, length/2, xp.zeros_like(length))
    mass = xp.where(live, length/xp.max(length), xp.zeros_like(length))
    curved = (xp.asarray(curved_edges, dtype=dtype) if reference is not None else None)
    mid = A + E/2
    reduced = reduced_basis(lattice, transform)
    ax, bx = reduced[0]
    ay, by = reduced[1]
    det = ax*by-bx*ay
    count = A.shape[0]
    if chunk is None:
        chunk = max(1, min(pts.shape[0], 4_000_000//(count*(2*ring+1)**2)))

    def evaluate(p):
        dx, dy = p[:, None, 0]-mid[None, :, 0], p[:, None, 1]-mid[None, :, 1]
        km = xp.round((by*dx-bx*dy)/det)
        kn = xp.round((ax*dy-ay*dx)/det)
        dx, dy = dx-km*ax-kn*bx, dy-km*ay-kn*by
        best = None
        for m in range(-ring, ring+1):
            for n in range(-ring, ring+1):
                x, y = dx-m*ax-n*bx, dy-m*ay-n*by
                along, normal = x*tx+y*ty, x*nx+y*ny
                excess = xp.maximum(xp.abs(along)-extent, xp.zeros_like(along))
                dist2 = normal*normal+excess*excess
                best = dist2 if best is None else xp.minimum(best, dist2)
        best = xp.where(live[None, :], best, xp.asarray(float('inf'), dtype=dtype))
        minimum = xp.min(best, axis=1, keepdims=True)
        # Empty/degenerate loops have the circular extension and zero gradients.
        minimum = xp.where(xp.any(live), minimum, xp.ones_like(minimum))
        on_gamma, on_edge = minimum == 0., best == 0.
        ratio = xp.where(on_gamma, xp.ones_like(minimum), minimum) / xp.where(
            on_edge, xp.ones_like(best), best)
        weight = xp.where(on_gamma, xp.astype(on_edge, dtype), ratio*ratio)*mass
        a = xp.sum(weight*(nx*nx), axis=1)
        b = xp.sum(weight*(nx*ny), axis=1)
        c = xp.sum(weight*(ny*ny), axis=1)
        peak = xp.maximum(a, c)
        empty = peak == 0.
        divisor = xp.where(empty, xp.ones_like(peak), peak)
        a = xp.where(empty, xp.ones_like(a), a/divisor)
        b = b/divisor
        c = xp.where(empty, xp.ones_like(c), c/divisor)
        if reference is not None:
            # Keep the analytic extension away from polygonized curves. Only
            # a chord-sized neighborhood needs its facet normal corrected.
            # All weights are continuous sums, not a selected nearest edge.
            total = xp.sum(weight, axis=1)
            curve_weight = weight*curved
            curve_mass = xp.sum(curve_weight, axis=1)
            band2 = xp.sum(curve_weight*length2, axis=1)/xp.where(
                curve_mass > 0., curve_mass, xp.ones_like(curve_mass))
            distance2 = minimum[:, 0]
            band_ratio = band2/xp.where(band2+distance2 > 0., band2+distance2,
                                       xp.ones_like(band2))
            alpha = (curve_mass/xp.where(total > 0., total, xp.ones_like(total)))*band_ratio**2
            base = reference(p)
            # Re(P)=M^2/tr(M^2). Its positive square root is proportional to
            # Re(P)+abs(Im(P_xy))*I, recovering the real dyad tensor up to scale.
            imaginary = xp.abs(xp.imag(base[:, 1]))
            aa, bb, cc = xp.real(base[:, 0])+imaginary, xp.real(base[:, 1]), xp.real(base[:, 2])+imaginary
            tr, base_tr = a+c, aa+cc
            a, b, c = (alpha*a/tr+(1-alpha)*aa/base_tr,
                       alpha*b/tr+(1-alpha)*bb/base_tr,
                       alpha*c/tr+(1-alpha)*cc/base_tr)
        norm2 = a*a+2*b*b+c*c
        xx, yy = (a*a+b*b)/norm2, (c*c+b*b)/norm2
        xy = ((a+c)*b+1j*(b*b-a*c))/norm2
        return xp.stack([xx, xy, yy, xp.conj(xy)], axis=-1)

    if 'jax' in xp.__name__:
        return _scanned_normals(xp, evaluate, pts, pts.shape[0], chunk, width=4)
    return xp.concat([evaluate(pts[lo:lo+chunk]) for lo in range(0, pts.shape[0], chunk)], axis=0)


def polygon_projector_coeffs(xp, polys, grid_shape, lattice, hu, hv, dtype, cdtype, plan,
                             shapes=None):
    """Sample the polygon Jones projector after removing a common translation."""
    check_normal_grid(grid_shape, hu, hv)
    polys = [xp.astype(v, dtype) for v in polys]
    # The first shape's centre, as on the parametric route since 2026-09-11 (the vertex mean
    # of an area-matched n-gon or a box is its centre to rounding). A phase restores a rigid
    # translation exactly instead of moving the quadrature grid across the painted boundary,
    # and anchoring both routes on the same point is what lets a polygon-count ladder
    # converge to the parametric solve of the same scene at the same grid
    # (``test_polygon_refinement_approaches_the_analytic_contact_solve``). It was the mean of
    # every vertex before, which is symmetric in the shape order but sits nowhere the
    # parametric route samples from.
    centre = xp.mean(polys[0], axis=0)
    ax, bx = lattice[0]
    ay, by = lattice[1]
    shift = centre-xp.stack([xp.asarray(.5*(ax+bx), dtype=dtype),
                            xp.asarray(.5*(ay+by), dtype=dtype)])
    built = [v-shift for v in polys]
    X, Y = sample_grid(xp, grid_shape, lattice, dtype)
    pts = xp.stack([xp.reshape(X, (-1,)), xp.reshape(Y, (-1,))], axis=-1)
    options = {}
    if shapes is not None:
        import wavebit.rcwa._engine.parametric as P
        moved = [dict(s, cx=s.get('cx', 0.)-shift[0], cy=s.get('cy', 0.)-shift[1]) for s in shapes]
        def reference(p):
            return xp.stack(P._piece_field(xp, p[:, 0], p[:, 1], moved, plan, None,
                                           lattice, dtype, dyadic=True), axis=-1)
        options = dict(reference=reference, curved_edges=[
            s.get('type', 'rectangle') in ('circle', 'disk', 'ellipse')
            for s, v in zip(shapes, polys) for _ in range(v.shape[0])])
    fields = polygon_dyadic_projector(xp, pts, built, lattice, dtype, plan.reduction, plan.ring,
                                      **options)
    KY, KX = _k_grids(xp, hu, hv, dtype)
    det = ax*by-bx*ay
    su, sv = (by*shift[0]-bx*shift[1])/det, (ax*shift[1]-ay*shift[0])/det
    phase = xp.exp(xp.astype(-2j*3.141592653589793*(KX*su+KY*sv), cdtype))
    return tuple(_sampled_coeffs(xp, xp.astype(xp.reshape(fields[:, i], grid_shape), cdtype),
                                 hu, hv)*phase for i in range(4))


# ============================================ convolution matrices for one layer
def _k_grids(xp, half_width_u, half_width_v, dtype):
    ky = xp.asarray([float(k) for k in range(-2 * half_width_v, 2 * half_width_v + 1)], dtype=dtype)
    kx = xp.asarray([float(k) for k in range(-2 * half_width_u, 2 * half_width_u + 1)], dtype=dtype)
    KY = ky[:, None] + 0.0 * kx[None, :]
    KX = 0.0 * ky[:, None] + kx[None, :]
    return KY, KX


def _centroid(xp, V):
    """Area centroid of a simple polygon (not the vertex mean)."""
    x, y = V[:, 0], V[:, 1]
    xn, yn = xp.roll(x, -1), xp.roll(y, -1)
    cr = x * yn - xn * y
    a = 0.5 * xp.sum(cr)
    return xp.stack([xp.sum((x + xn) * cr) / (6.0 * a),
                     xp.sum((y + yn) * cr) / (6.0 * a)])


def _paint_centroid(xp, segment_data, plan):
    """Centroid of the unwrapped painted union, with internal traces canceled first.

    The plan picks one oriented representative of each unwrapped exterior interval.
    This includes periodic abutting faces for the area integral, but excludes zero-area
    spurs and shared internal cuts. Their cancellation is combinatorial, so an invisible
    nested contour cannot perturb the quadrature origin through floating-point summation.
    """
    a, e, _, _ = segment_data
    ids = xp.asarray([i for i, p in enumerate(plan.pieces) if p.centroid_sign], dtype=xp.int64)
    sign = xp.asarray([p.centroid_sign for p in plan.pieces if p.centroid_sign], dtype=a.dtype)
    a, z = a[ids], (a + e)[ids]
    cross = (a[:, 0] * z[:, 1] - z[:, 0] * a[:, 1]) * sign
    return xp.stack([xp.sum((a[:, 0] + z[:, 0]) * cross),
                     xp.sum((a[:, 1] + z[:, 1]) * cross)]) / (3 * xp.sum(cross))


def _cell_scale(xp, lattice, dtype):
    """The largest basis entry's magnitude: the length :func:`material_projector_fields`
    measures its distances against."""
    return xp.max(xp.abs(xp.stack([xp.asarray(lattice[i][j], dtype=dtype)
                                   for i in (0, 1) for j in (0, 1)])))


def _on_interface_threshold2(xp, lattice, dtype):
    """The squared distance within which :func:`material_projector_fields` reads a sample as
    on an interface: 64 rounding units of the largest basis entry, squared.

    One definition, read by the blend and by the analytic curve candidates it is handed
    (:func:`~wavebit.rcwa._engine.tensor_shapes._curve_fields`), which switch to the interface
    normal inside it exactly as the segments do.
    """
    return (64*xp.finfo(dtype).eps*_cell_scale(xp, lattice, dtype))**2


def material_projector_fields(xp, grid_shape, lattice, dtype, segment_data, contrasts,
                               interface_plan, transform=None, chunk=None,
                               candidate_fields=None):
    """A contrast-weighted extension of the nearest-interface normal projector.

    Keep only the nearest point per unordered pair of region labels: all its intervals
    have the same contrast, and a farther copy cannot increase its inverse-distance
    influence. This also makes subdivision of a straight interval invisible.

    In increasing distance order, a candidate contributes only the positive increase
    over the largest preceding influence, with influence ``contrast**4 / distance**4``.
    Normalize those increases by their maximum. Equal nonzero contrasts recover the
    nearest-point extension exactly; a vanishing contrast loses its influence continuously
    off its own interface. On an interface its pole recovers the exact normal dyad.
    Equivalently, average the nearest-point vectors over all influence thresholds,
    then normalize. Shared endpoint candidates carry the same vector, so exchanging
    their order does not change that average. Material-dependent phases would break
    this property. The vectors and weights are real even for complex materials.

    This retains nearest-point spatial switches and the usual zero-vector tie;
    it does not promise global smoothness at medial axes or disappearing interfaces.

    ``candidate_fields`` optionally supplies analytic candidates for uncut curved
    interfaces, keyed by their unordered region pair. Each callable returns, at the
    supplied Cartesian points, the squared distance to its exact curve and a unit vector
    oriented as a segment candidate's: from the interface toward the sample on both sides,
    as ``(p - foot) / |p - foot|`` is, and on the interface itself -- a squared distance
    within :func:`_on_interface_threshold2` -- the normal a segment's ``n_edge`` carries,
    which leaves the region on the interval's left (for a shape on the background, its
    outward normal). Off the interface the vector need not be the nearest-point one: the
    tensor curve candidates carry the level-set direction
    (:func:`~wavebit.rcwa._engine.tensor_shapes._curve_direction`). Signed vectors are summed below
    before the dyad is formed, so a candidate that points another way changes the blend
    wherever a second candidate carries mass. Their exact curves replace the polygon's
    facets and corners without changing material contrasts or the actual-interface plan for
    contacts and painted regions.
    """
    from wavebit.rcwa._engine.parametric import _BLEND_EXPONENT
    A, E, L2, active = segment_data
    ids = [i for group in interface_plan.groups for i in group]
    groups = {}
    for image_index, i in enumerate(ids):
        piece = interface_plan.pieces[i]
        key = tuple(sorted((piece.left, piece.right)))
        if key[0] != key[1]:
            groups.setdefault(key, []).append(image_index)
    if not groups:
        one = xp.ones(grid_shape, dtype=dtype)
        zero = xp.zeros_like(one)
        return one, zero, zero
    A, E, L2 = periodic_segments(xp, A, E, L2, lattice, transform,
                                interface_plan.images, interface_plan.groups)
    extended = xp.asarray(ids, dtype=xp.int64)
    active, contrasts = active[extended], contrasts[extended]
    candidates = []
    for key, indices in groups.items():
        index = xp.asarray(indices, dtype=xp.int64)
        valid = xp.sum(E[index]*E[index], axis=1) > 0.
        field = None if candidate_fields is None else candidate_fields.get(key)
        candidates.append((A[index], E[index], L2[index], contrasts[indices[0]], valid, field))
    pts = _folded_sample_points(xp, grid_shape, lattice, dtype, transform)
    reference = E[xp.argmax(xp.astype(active, xp.int32))]
    cross = E[:, 0]*reference[1] - E[:, 1]*reference[0]
    parallel = xp.all((cross == 0.) | ~active)
    reference2 = xp.sum(reference*reference)
    normal = xp.stack([reference[1], -reference[0]]) / xp.sqrt(
        xp.where(reference2 > 0., reference2, xp.ones_like(reference2)))
    scale = _cell_scale(xp, lattice, dtype)
    threshold2 = _on_interface_threshold2(xp, lattice, dtype)
    # The ratio weights below are formed in units of the cell: the power of two nearest the
    # squared scale, which divides without rounding, so float64 is unchanged. JAX
    # differentiates minimum / d2 through 1 / d2**2, and on its CPU backend a float32 square
    # below the smallest normal number is flushed to zero: a sample nearer an interface than
    # about 3.3e-10, in whatever unit lengths are given, read an infinite cotangent (0.33 nm
    # with lengths in metres). In units of the cell every kept d2 exceeds half the
    # on-interface threshold's (64 eps)**2, and its square stays normal at any length unit.
    # Spelled as pow: JAX's exp2 is not exact at integers (127 of 160 powers off in float32).
    unit2 = xp.pow(xp.asarray(2., dtype=dtype), 2*xp.round(xp.log2(scale)))
    contrast = xp.stack([item[3] for item in candidates])
    strengths = contrast**_BLEND_EXPONENT*xp.stack([
        xp.astype(xp.any(item[4]), dtype) for item in candidates])
    count = len(candidates)
    if chunk is None:
        chunk = max(1, min(pts.shape[0], 4_000_000 // max(len(A), 1)))

    def evaluate(p):
        rows = xp.arange(p.shape[0])
        distances, normals = [], []
        for a, e, l2, _, valid, field in candidates:
            if field is not None:
                distance, direction = field(p)
                distances.append(distance)
                normals.append(direction)
                continue
            dist2, foot = _seg_dist(xp, p, a, e, l2)
            dist2 = xp.where(valid[None, :], dist2, xp.asarray(float('inf'), dtype=dtype))
            nearest = xp.argmin(dist2, axis=1)
            distance = dist2[rows, nearest]
            delta = p-foot[rows, nearest]
            edge = e[nearest]
            n_edge = xp.stack([edge[:, 1], -edge[:, 0]], axis=-1)
            edge2 = xp.sum(n_edge*n_edge, axis=1)
            n_edge = n_edge / xp.sqrt(xp.where(edge2 > 0., edge2,
                                               xp.ones_like(edge2)))[:, None]
            n_far = delta / xp.sqrt(xp.where(distance > threshold2, distance,
                                              xp.ones_like(distance)))[:, None]
            normals.append(xp.where((distance <= threshold2)[:, None], n_edge, n_far))
            distances.append(distance)
        d2 = xp.stack(distances, axis=1)
        direction = xp.stack(normals, axis=1)
        minimum = xp.min(xp.where(strengths[None, :] > 0., d2,
                                  xp.asarray(float('inf'), dtype=dtype)), axis=1)
        minimum = xp.where(xp.any(strengths > 0.), minimum, xp.ones_like(minimum))
        on = minimum <= threshold2
        # On an interface only the exact-interface weights below are used, and
        # a zero-strength candidate (an interval whose two materials agree in
        # this pass, as a magnetic-only shape's do for epsilon) is discarded
        # everywhere -- `on` never looks at it. Form neither's inverse distance:
        # its squared distance has no lower bound, in float32 a tiny one squares
        # to zero, and the discarded 0 * inf poisons the cotangent. The kept
        # ones are bounded below and formed in units of the cell (`unit2`).
        ratio = xp.where(on, xp.ones_like(minimum), minimum/unit2)[:, None] / xp.where(
            (d2 > 0.) & ~on[:, None] & (strengths[None, :] > 0.), d2/unit2, xp.ones_like(d2))
        influence = strengths[None, :] * xp.where(on[:, None], xp.astype(d2 <= threshold2, dtype),
                                                   ratio**(_BLEND_EXPONENT/2))
        order = xp.argsort(d2, axis=1, stable=True)
        sorted_influence = influence[rows[:, None], order]
        sorted_direction = direction[rows[:, None], order]
        # Parallel prefix maximum in log2(number of material pairs) Array-API steps.
        prefix = sorted_influence
        offset = 1
        while offset < count:
            shifted = xp.concat([xp.zeros_like(prefix[:, :offset]), prefix[:, :-offset]], axis=1)
            prefix = xp.maximum(prefix, shifted)
            offset *= 2
        previous = xp.concat([xp.zeros_like(prefix[:, :1]), prefix[:, :-1]], axis=1)
        mass = xp.maximum(sorted_influence-previous, xp.zeros_like(sorted_influence))
        peak = prefix[:, -1]
        mass = mass / xp.where(peak > 0., peak, xp.ones_like(peak))[:, None]
        sx = xp.sum(mass*sorted_direction[:, :, 0], axis=1)
        sy = xp.sum(mass*sorted_direction[:, :, 1], axis=1)
        tie = (sx == 0.) & (sy == 0.)
        reference_direction = sorted_direction[rows, xp.argmax(mass, axis=1)]
        sx, sy = (xp.where(tie, reference_direction[:, 0], sx),
                  xp.where(tie, reference_direction[:, 1], sy))
        sx, sy = xp.where(parallel, normal[0], sx), xp.where(parallel, normal[1], sy)
        sx, sy = xp.where(xp.any(active), sx, 1.), xp.where(xp.any(active), sy, 0.)
        norm2 = sx*sx+sy*sy
        return xp.stack([sx*sx/norm2, sx*sy/norm2, sy*sy/norm2], axis=-1)

    if 'jax' in xp.__name__:
        result = _scanned_normals(xp, evaluate, pts, pts.shape[0], chunk, width=3)
    else:
        result = xp.concat([evaluate(pts[lo:lo+chunk])
                            for lo in range(0, pts.shape[0], chunk)], axis=0)
    return tuple(xp.reshape(result[:, i], grid_shape) for i in range(3))


def normal_projector_coeffs(xp, loops, grid_shape, lattice, half_width_u, half_width_v,
                            dtype, cdtype, transform=None, segment_data=None, images=None,
                            image_groups=None, anchor=None, contrasts=None, interface_plan=None):
    """Coefficient grids of ``n_u**2``, ``n_u*n_v``, ``n_v**2``, built with the layer CENTRED.

    The resolved route supplies the area centroid of the unwrapped painted
    union as anchor. Center every partition count on it, then carry the sampled
    coefficients back by a Fourier phase. Centering only single loops made an
    off-center same-material inclusion change quadrature with painting order.

    Relative geometry is rebuilt on every call, including its derivative; the
    phase describes only the common shift. This is a sampling convention, not
    a claim that finite quadrature equals the continuous Fourier integral.
    Legacy raw-loop probes keep their single-loop centroid convention.
    """
    check_normal_grid(grid_shape, half_width_u, half_width_v)
    if len(loops) == 1 or anchor is not None:
        c = anchor if anchor is not None else _centroid(xp, loops[0])
        # The `lattice is None` arm that used to sit here computed the rectangle's answer
        # from `Px`, `Py`. The solver stopped reaching it once `_geometry_lattice_for`
        # became total (`cc016fc`): driving only the public API, the one function that
        # ever sees a `None` cell is `as_lattice` itself, where the conversion happens.
        # Deleted rather than re-keyed on `is_axis_aligned` -- production takes *this*
        # branch on a rectangle today, and the two differ in the last place (1.3e-16).
        a1x, a2x = lattice[0][0], lattice[0][1]
        a1y, a2y = lattice[1][0], lattice[1][1]
        mid_x, mid_y = 0.5 * (a1x + a2x), 0.5 * (a1y + a2y)
        shift = xp.stack([c[0] - mid_x, c[1] - mid_y])
        built = [v - shift[None, :] for v in loops]
        if segment_data is not None:
            a, e, l2, active = segment_data
            segment_data = (a - shift[None, :], e, l2, active)
        KY, KX = _k_grids(xp, half_width_u, half_width_v, dtype)
        # The displacement in *fractional lattice* coordinates, A^-1 @ shift. The phase is
        # already exp(-i G.d) written that way; it was only ever handed Px / Py fractions.
        det = a1x * a2y - a2x * a1y
        su = (a2y * shift[0] - a2x * shift[1]) / det
        sv = (-a1y * shift[0] + a1x * shift[1]) / det
        arg = -2.0 * 3.141592653589793 * (KY * sv + KX * su)
        phase = xp.astype(xp.cos(arg), cdtype) + 1j * xp.astype(xp.sin(arg), cdtype)
    else:
        built = loops
        phase = xp.asarray(1.0 + 0j, dtype=cdtype)

    if contrasts is not None and segment_data is not None:
        fields = material_projector_fields(xp, grid_shape, lattice, dtype, segment_data,
                                            contrasts, interface_plan, transform)
    else:
        normal_x, normal_y = distance_gradient_normal_field(xp, built, grid_shape, lattice, dtype,
                                                            transform=transform,
                                                            segment_data=segment_data, images=images,
                                                            image_groups=image_groups)
        fields = (normal_x*normal_x, normal_x*normal_y, normal_y*normal_y)
    out = []
    for f in fields:
        out.append(_sampled_coeffs(xp, xp.astype(f, cdtype), half_width_u, half_width_v) * phase)
    return out


def conv_mats(xp, regions, background, half_width_u, half_width_v, lattice, shape_nvf_grid_size,
              dtype, cdtype, transform=None, interface_plan=None, projector_coeffs=None,
              laurent=False):
    """``(epx, epy, epz_i, epxy, epyx)`` for one boundary-resolved layer.

    ``laurent=True`` stops once ``[[eps]]`` is known and returns Laurent's slots
    (:func:`~wavebit.rcwa._engine.parametric.laurent_from_coeffs`), off-diagonal pair ``None``.

    ``regions`` is ``[(material, [loop, ...]), ...]`` -- disjoint, painting already resolved by
    the clipper, each ``material`` a :class:`~wavebit._common.periodic.Material` or the bare
    permittivity that is shorthand for one. Hole loops arrive wound clockwise and need no
    special case here: the polygon FT is a signed contour integral, so a hole subtracts by
    itself.

    That last sentence was once in this docstring as an *assumption*, and it was wrong, because
    :func:`~.clipper.subtract_loop` was at the time normalising every loop to counter-clockwise
    and carrying the orientation in a separate ``sign`` that this function never read. It is
    true now because the clipper is written to make it true -- see
    :func:`~.clipper.subtract_loop`, which is where the convention is established and where the
    reason it is a convention rather than a coincidence is set out. Do not re-derive it from
    "the FT is signed": that says a *clockwise* loop subtracts, not that the clipper hands you
    one.

    Why it matters that the invariant lives in the data: nothing downstream can catch it being
    dropped. The matrices stay Hermitian and the solve conserves energy exactly, so a shape
    painted entirely inside another was solved as a different structure in silence -- measured
    ``<eps> = 2.4256`` against an exact 2.2371 for a disc of permittivity 9 inside a square of 4,
    at every ``truncation_order`` alike, and ``sum(R) = 0.1798`` against a converged raster's 0.4164.

    The material transform keeps these signed loops unchanged. The normal
    builder instead receives actual interface intervals: oriented coverage is
    summed per region, zero-area retraces cancel, and current material values
    decide whether the two sides differ. Winding matters to this extraction;
    only after extracting the interface set is the normal's overall sign free.

    The scalar-property view is shared with the permeability pass, which puts
    mu_value in the epsilon slots. The topology plan therefore holds no material
    equality and can be replayed under material gradients or either property.
    """
    KY, KX = _k_grids(xp, half_width_u, half_width_v, dtype)
    is_dc = (KY == 0) & (KX == 0)
    zero = xp.zeros(KY.shape, dtype=cdtype)

    # `eps`, not `conj(eps)`. Wavebit works in `e^{-iwt}` throughout (`Im(eps) > 0` = loss)
    # with no conjugation on the way in — the same sentence `parametric.assemble_eps_coeffs`
    # carries, and this builder is its later copy. It kept the conjugate that the parametric path
    # was moved off in 2026-08, which turned an absorbing painted layer into a gain medium:
    # `R + T` up to 1.11 and reflectance off by 0.16 on gold. Invisible to `test_boundary.py`,
    # which builds every shape with a real material, where the two spellings are identical.
    # See `docs/validation/rcwa/time-convention-audit.md` §6.
    #
    # No squaring either, on either line: both slots hold permittivity now, and the `** 2`
    # that used to sit here was the front door's index-to-eps conversion. Its removal is what
    # the convention flip *is* on this path (`docs/reference/common/conventions.md`).
    eps_bg = xp.astype(xp.asarray(as_material(background, 'layer background').epsilon,
                                  dtype=cdtype), cdtype)
    eps_c = xp.where(is_dc, eps_bg, zero)
    inv_c = xp.where(is_dc, 1.0 / eps_bg, zero)
    eps_bound = xp.abs(eps_bg)

    for material, polys in regions:
        eps_s = xp.astype(xp.asarray(as_material(material, 'shape material').epsilon,
                                     dtype=cdtype), cdtype)
        ft = zero
        for V in polys:
            ft = ft + xp.astype(polygon_ft(xp, V, KY, KX, lattice), cdtype)
        eps_c = eps_c + (eps_s - eps_bg) * ft
        inv_c = inv_c + (1.0 / eps_s - 1.0 / eps_bg) * ft
        eps_bound = xp.maximum(eps_bound, xp.abs(eps_s))

    if laurent:
        from wavebit.rcwa._engine.parametric import laurent_from_coeffs
        return laurent_from_coeffs(xp, eps_c, half_width_u, half_width_v)

    if projector_coeffs is not None:
        if len(projector_coeffs) != 4:
            raise ValueError('the complex shape projector needs xx, xy, yy, yx coefficients')
        return nvf_from_coeffs(xp, eps_c, inv_c, *projector_coeffs[:3],
                               half_width_u, half_width_v, c_yx=projector_coeffs[3], eps_bound=eps_bound)

    all_loops = [V for _, polys in regions for V in polys]
    if transform is None:
        transform = reduction_transform(lattice)
    if interface_plan is None:
        interface_plan = plan_interfaces(regions, lattice, transform)
    segment_data = replay_interfaces(xp, regions, background, lattice, transform, interface_plan)
    coeffs = normal_projector_coeffs(
        xp, all_loops, (shape_nvf_grid_size, shape_nvf_grid_size), lattice, half_width_u, half_width_v,
        dtype, cdtype, transform=transform, segment_data=segment_data,
        images=interface_plan.images, image_groups=interface_plan.groups,
        contrasts=interface_contrasts(xp, regions, background, interface_plan, dtype),
        interface_plan=interface_plan,
        anchor=_paint_centroid(xp, segment_data, interface_plan) if segment_data is not None else None)
    return nvf_from_coeffs(xp, eps_c, inv_c, *coeffs[:3], half_width_u, half_width_v,
                           c_yx=coeffs[3] if len(coeffs) == 4 else None,
                           eps_bound=eps_bound)


# ================================================== one painted layer, end to end

#: Vertices used to polygonise a curved primitive for this path. A polygonised ellipse
#: converges on the true one at O(1/n^2), and 256 puts that error far below the normal
#: field's, which is this path's floor. Straight-edged primitives are exact and ignore it.
POLYGON_VERTICES = 256


def shape_vertices(xp, shape, lattice, n_poly=POLYGON_VERTICES):
    """Counter-clockwise vertices of one parametric primitive.

    Same shape dictionaries the parametric path takes, and the same convention: ``lx`` / ``ly``
    are full lengths, so a circle is ``lx = ly = diameter``.

    The dict is read by :func:`~wavebit.rcwa._engine.parametric.shape_geometry` and nowhere else, for
    the reason that function was written: it used to be read here as well, directly, and the
    two readings disagreed. ``ly`` had no default here at all, so the natural spelling of a
    circle — give the diameter once — summed fine and raised ``KeyError`` the moment the same
    layer was painted; ``cx`` / ``cy`` had no default either, where the transform centres a
    shape in the cell; and ``square`` kept whatever ``ly`` it was handed instead of being
    square. ``lattice`` is taken for the same reason the transform takes it: ``cx`` / ``cy``
    are measured from ``0.5*(a1 + a2)``, so the basis is what puts a shape anywhere at all.
    A ``period`` pair is not a substitute -- it has the cell's angle already thrown away.
    """
    kind, cx, cy, lx, ly, angle = shape_geometry(shape, lattice)
    if kind in ('rectangle', 'square'):
        return rect_verts(xp, cx, cy, lx, ly, angle)
    if kind in ('circle', 'disk', 'ellipse'):
        return ellipse_verts(xp, cx, cy, lx, ly, angle, n_poly)
    raise NotImplementedError(
        f"The boundary route cannot draw a {kind!r} yet; it needs the shape as a polygon. "
        f"Supported: rectangle, square, circle, disk, ellipse.")


def _concrete_shape(shape):
    """Every numeric value of a shape dict as a plain Python float.

    :func:`plan_layer` is documented "concrete values only" and was relying on the caller to
    have honoured it. NumPy and JAX let that slide -- a JAX array satisfies ``__array__`` and
    converts on contact -- while a PyTorch tensor survives all the way into ``rect_verts`` and
    multiplies against a NumPy array, which raises. Enforcing the contract here is what makes
    the planning stage namespace-agnostic, which is what it always claimed to be.

    Values that are not numbers are passed through untouched: a shape's ``type``, and its
    ``material`` -- a ``Material``, or a complex permittivity, which ``float()`` would refuse
    and which the planner never reads anyway.

    The read is :func:`~wavebit._common.backend.checked_float`, which is what makes this
    namespace-agnostic *under a gradient* too. It used to spell the numeric half by hand --
    ``detach()`` then ``float()`` -- which reaches a PyTorch leaf and not a ``jax.grad``
    tracer, so planning worked under torch autograd and fell through to
    ``TracerArrayConversionError`` under JAX's. The plan is combinatorics taken from the
    primal; the vertices that carry the gradient are re-cut by it downstream, which is why
    reading the primal here is right rather than merely convenient. Under ``jax.jit`` there is
    no primal and the read raises, which is the answer a frozen record exists for.
    """
    out = {}
    for k, v in shape.items():
        try:
            out[k] = checked_float(v)
        except (TypeError, ValueError):
            out[k] = v
    return out


def plan_layer(shapes, lattice, n_poly=POLYGON_VERTICES):
    """Resolve one layer's painting order into disjoint loops. **Concrete values only.**

    This is the discrete half — which loop holds which vertex, which edges cross — so it runs
    on NumPy outside any trace. Freeze it once
    (:meth:`~wavebit.rcwa._engine.engine.RCWAEngine.freeze_geometry`) and every later step replays
    it against traced vertices.

    The third entry is the integer transform to the Lagrange-reduced basis
    (:func:`~wavebit.lattice.reduction_transform`), taken here on the concrete basis so
    that a replay on a traced one enumerates its periodic images on the same reduced cell
    (:func:`periodic_segments`). The fourth entry records the actual-interface
    intervals, unwrapped centroid representatives, and bounded image groups.
    """
    import numpy as np
    verts = _plan_vertices(shapes, lattice, n_poly)
    recs, regions = _clipper.resolve_painting(verts)
    transform = reduction_transform(lattice)
    vals = _clipper.eval_table(np, recs, verts)
    regs = [(None, [_clipper.loop_verts(np, lp, vals) for lp in region]) for region in regions]
    interfaces = plan_interfaces(regs, lattice, transform)
    return recs, regions, transform, interfaces


def _concrete_basis(lattice):
    """The cell basis as plain floats, read the way :func:`_concrete_shape` reads a centre.

    The basis became part of *where a shape is* on 2026-08-31, when ``cx`` / ``cy`` moved to
    the centre of the cell: :func:`~wavebit.rcwa._engine.parametric.shape_geometry` adds
    ``0.5*(a1 + a2)`` to every shape now, not only to a defaulted one. Without this read a plan
    cut from perfectly concrete shapes on a *differentiated cell* would fail the way the shape
    half used to -- ``TracerArrayConversionError`` under ``jax.grad`` -- and the staleness check
    would be skipped on exactly the loop it exists for. Same read, same reason: a plan is
    combinatorics taken from the primal.
    """
    return [[checked_float(lattice[i][j]) for j in (0, 1)] for i in (0, 1)]


def _plan_vertices(shapes, lattice, n_poly):
    """Every shape's vertices as plain float arrays -- the concrete half of building a plan.

    Split out so :func:`check_plan_still_fits` can tell the two ways a re-plan fails apart. This
    step is the one a tracer cannot survive, so its refusal means "skip the check"; anything
    ``resolve_painting`` then raises is about the layer and has to be heard.
    """
    import numpy as _np
    basis = _concrete_basis(lattice)
    return [_np.asarray(shape_vertices(_np, _concrete_shape(s), basis, n_poly), float)
            for s in shapes]


def _plan_key(plan):
    """A painting plan as nested lists, so a record compares by content whatever holds it."""
    recs, regions, reduction, interfaces = plan
    return ([tuple(r) for r in recs],
            [[list(loop) for loop in region] for region in regions],
            tuple(tuple(int(t) for t in row) for row in reduction), interfaces)


def check_plan_still_fits(shapes, lattice, plan, n_poly=POLYGON_VERTICES):
    """Refuse a frozen painting plan that the layer in front of it has outgrown.

    A plan records *combinatorics* -- which loop holds which vertex, which edges cross -- and
    deliberately holds no coordinates, which is exactly what lets a replay carry a gradient to
    vertices that have moved. Move them far enough and the combinatorics change with them: two
    shapes that did not overlap begin to, and the recorded loops describe a painting that is no
    longer the one being asked for. Nothing about the result says so -- on a two-rectangle layer
    a replay stays bit-identical while the overlap holds and then drifts to 3.3e-2 in total
    transmission once it does not, with no step in between to notice. The basis reduction the
    plan recorded is compared beside the loops: a plan taken on one spelling of the cell and
    replayed on another would enumerate its periodic images on the wrong cell
    (:func:`periodic_segments`), so a changed reduction is stale too.

    The comparison replays planning on concrete geometry, including interface
    intervals and image bounds; its cost scales with polygon complexity, not
    the retained Fourier order count. Under a trace it is skipped because no
    concrete geometry is available. The caller must recheck eagerly between
    traced steps when moving through topology or image-bound events.
    """
    try:
        _plan_vertices(shapes, lattice, n_poly)
    except (TypeError, ValueError):
        return                      # traced: nothing concrete to re-plan from
    fresh = plan_layer(shapes, lattice, n_poly)
    fresh_key, plan_key = _plan_key(fresh), _plan_key(plan)
    if fresh_key == plan_key:
        return

    raise StaleGeometryError(
        f'the frozen geometry no longer fits this structure: {_what_moved(fresh_key, plan_key)}. '
        f'The recorded plan resolves a painting this layer no longer has, so replaying it paints '
        f'a different structure. Call freeze_geometry() again, or drop geometry= to resolve '
        f'eagerly. (Moving a shape never trips this on its own; moving it far enough to change '
        f'which shapes cover which does.)')


def _what_moved(fresh_key, plan_key):
    """The one clause of the refusal that names what the re-plan found, in the layer's terms."""
    if fresh_key[2] != plan_key[2]:
        return ('the cell basis reduces by a different transform than the one the record '
                'enumerated its periodic images on')
    if fresh_key[:2] == plan_key[:2] and fresh_key[3] != plan_key[3]:
        return 'the polygon interface intervals or their periodic image bounds changed'
    fresh, held = fresh_key[1], plan_key[1]
    if len(fresh) != len(held):
        return (f'it was resolved from {len(held)} painted shapes and this layer has '
                f'{len(fresh)}')

    recut = [i for i, (a, b) in enumerate(zip(fresh, held)) if a != b]
    if not recut:
        return 'the same loops are now cut from a different table of crossings'

    parts = []
    for i in recut:
        now, was = len(fresh[i]), len(held[i])
        if now == was:
            parts.append(f"shape {i}'s {now} loop{'s' if now != 1 else ''} now run through "
                         f'different vertices')
        elif was == 0:
            parts.append(f'shape {i} was completely covered and now shows {now} '
                         f"loop{'s' if now != 1 else ''}")
        elif now == 0:
            parts.append(f'shape {i} is completely covered now and the record gives it '
                         f'{was} loop{"s" if was != 1 else ""}')
        else:
            parts.append(f'shape {i} now resolves into {now} loop{"s" if now != 1 else ""} '
                         f'where the record holds {was}')
    return '; '.join(parts)


def to_conv_mat_boundary_layer(xp, shapes, background, half_width_u, half_width_v, lattice,
                               dtype, shape_nvf_grid_size=256, n_poly=POLYGON_VERTICES, plan=None,
                               projector_coeffs=None, nvf_projector=None, projector_plan=None,
                               laurent=False):
    """The five convolution matrices for one layer whose shapes are *painted*.

    ``shapes`` is in painting order — a later shape covers an earlier one — which is what a
    raster of the same instruction agrees with. Pass
    ``plan`` from a frozen geometry to keep the call traceable; leave it ``None`` and the
    painting is resolved eagerly, which needs concrete shape parameters.

    ``nvf_projector='circular'`` corrects the analytic Jones extension near curved
    polygon facets. ``projector_plan`` carries the primitive image ring alongside
    the clipping plan. The default retains the actual-interface linear field.

    ``laurent=True`` closes every product with the painted layer's own ``[[eps]]`` instead
    (:func:`~wavebit.rcwa._engine.parametric.laurent_from_coeffs`): the same polygons and the same
    transform, and no normal field, so no projector option may come with it.
    """
    if nvf_projector not in (None, 'linear', 'circular'):
        raise ValueError("projector must be 'linear' or 'circular'")
    if laurent and (nvf_projector is not None or projector_coeffs is not None
                    or projector_plan is not None):
        raise ValueError("Laurent's rule builds no normal projector; pass no projector option")
    verts = [shape_vertices(xp, s, lattice, n_poly) for s in shapes]
    if plan is None:
        recs, regions, transform, interfaces = plan_layer(shapes, lattice, n_poly)
    else:
        check_plan_still_fits(shapes, lattice, plan, n_poly)
        recs, regions, transform, interfaces = plan
    vals = _clipper.eval_table(xp, recs, verts)
    regs = [(s['material'], [_clipper.loop_verts(xp, lp, vals) for lp in region])
            for s, region in zip(shapes, regions)]
    real_dtype = xp.float64 if dtype == xp.complex128 else xp.float32
    if nvf_projector == 'circular':
        import wavebit.rcwa._engine.parametric as P
        if projector_coeffs is not None:
            raise ValueError('choose a polygon projector or supplied coefficients, not both')
        if projector_plan is None:
            projector_plan = P.plan_gamma(shapes, lattice)
            if projector_plan is None:
                projector_plan = P._blind_plan(shapes, lattice)
        P.check_gamma_plan_still_fits(dict(shapes=shapes), lattice, projector_plan, 0, dyadic=True)
        projector_coeffs = polygon_projector_coeffs(
            xp, verts, (shape_nvf_grid_size, shape_nvf_grid_size), lattice, half_width_u, half_width_v,
            real_dtype, dtype, projector_plan, shapes=shapes)
    return conv_mats(xp, regs, background, half_width_u, half_width_v, lattice, shape_nvf_grid_size,
                     real_dtype, dtype, transform=transform, interface_plan=interfaces,
                     projector_coeffs=projector_coeffs, laurent=laurent)
