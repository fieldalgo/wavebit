"""The bodies: a signed distance, its gradient, and the closed forms each one happens to own.

A **body** is the smoothed, differentiable thing stage 4 adds, and it is not the same object as
a ``Shape`` in ``wavebit/fd/geometry.py``. A ``Slab`` or a ``Box`` there is cell-snapped and
answers questions about *cell indices* -- which cells it fills, which it may touch, whether it
is uniform across a layer. A body answers questions about a *point*: how far is your surface
from here, which way does it face, and -- where it can do better than a plane -- exactly how
much of this voxel do you fill. The two meet in C2, which is the only place that will know both.

Every body answers four things:

``surface(points) -> (d, n)``
    ``d`` negative inside, and ``n`` the **unit** outward normal, unit *everywhere* including
    at the points where the true gradient does not exist (a sphere's centre, a box's medial
    axis, a polygon's vertex). That is a contract ``_cut.cut_parts`` depends on: it divides by
    ``max_i |n_i| dx``, which is at least ``dx/sqrt(3)`` for a unit vector and zero for a zero
    one. Where the gradient is genuinely undefined the body names the direction it returns; the
    fill there is 0 or 1 whatever the direction is, so the choice cannot be observed.

``exact_fill(points, dx) -> fill | None``
    The exact filled fraction where a closed form is free, ``None`` otherwise -- and ``None``
    is the *common* case, which is why the plane exists.

``exact_moment(points, dx) -> (area, moment) | None``
    The exact interface area in the voxel and ``area * n_c^2`` over it. Separate from
    ``exact_fill`` because the two are not free together: a cylinder's fill is a closed form
    the repository already owns and its curved side's second moment is not.

``exact_edge_fill(points, dx, axis) -> fill | None``, ``exact_face_fill(...)``
    The same question on the component's own **edge** and its own **dual face** -- what the
    Werner/Bauer/Cary material operator constrains rather than the voxel. ``Sphere`` answers
    both; every other body, and every CSG node, returns ``None``. Nothing assembles these into
    ``parts``: the tangent-plane forms beside them (``_cut.edge_cut``, ``_cut.face_cut``) are a
    *different* answer, not a fallback, and ``_cut.py``'s section note and
    ``tests/test_fd_shapes.py::test_the_plane_is_not_a_drop_in_for_the_exact_edge_and_face_fills``
    carry how far apart they are.

``parts(points, dx) -> (fill, area, moment)``
    The three above, assembled: the exact forms where a body has them and the tangent plane
    where it does not. It is the one call :func:`_occupancy.occupancy` makes, and the one a
    :class:`_Combination` overrides -- a CSG node's answer is not its children's assembled
    separately, because where two children's surfaces touch the composed distance is wrong by
    half a voxel. See :class:`_Combination`.

**Where the exact forms are, and where they are not**

    body                fill                              moment
    Box, Prism          polygon-box clip x 1-D overlap    edge moments x overlap + cap areas
    Cylinder            disc-box clip x 1-D overlap       *tangent plane*
    Sphere              ball-box clip                     *tangent plane*
    Union/Intersection/ the children's own, composed per voxel: a boolean where no child's
    Difference          surface is in it, one child's own cut where one is, the exact two-plane
                        cut where two are (``_cut.two_plane``), and elsewhere an INTERVAL whose
                        safe end for that operation is what ships, counted and reported

A cylinder's exact moment would need the second moment of the arc of a circle clipped to a
rectangle -- the angular interval set, not the area -- and the repository owns no such form;
the arc-weighted *area* forms it does own (``_weighted_circle_area``) are a bilinear window for
a different geometry. A sphere's diagonal surface moment is available from its volume and
clipped face areas by the divergence theorem (see :class:`Sphere`); the previous claim that
volume second moments were needed was incorrect. The fallback remains the plane. Measured cost
of using the plane's moment on a cylinder instead of the truth, against 20001-point quadrature over the
true circumference
(``tests/test_fd_shapes.py::test_the_plane_is_the_cylinders_moment_and_this_is_what_that_costs``,
cut voxels only, worst ``|w - w_exact|`` then rms):

    R/h          away from the rim              on the rim
      4        1.31e-2   9.63e-3            3.98e-1   2.04e-1
      8        1.92e-2   1.02e-2            5.09e-1   2.35e-1
     16        7.19e-3   3.50e-3            5.18e-1   1.68e-1

Away from the rim the plane's normal *is* the true normal at the nearest point and the error is
the arc's own spread. It does **not** fall measurably over this sweep and the first writing of
this table said it did: the rms falls (9.63e-3, 1.02e-2, 3.50e-3) and the worst does not, and
an independent surface oracle in the C1 review read 7.6e-3 / 1.8e-2 / 6.9e-3 on the same three
resolutions -- non-monotone in both datasets. What the table shows is the worst.
**On the rim it does not converge at all.** A rim is a crease, and the plane sees exactly one
surface where the voxel holds two. The stage-4 design's section 3 says a creased two-media
interface is "handled exactly by the ``exact_fill``/moment forms"; for a cylinder's cap rim that
is not true as this table stands, and it is C2's to decide whether to write the clipped arc's
second moment or to record the price beside the rule.

**The crease is also where ``w`` jumps.** Sliding a body past one fixed voxel in steps of
0.0025 cell and keeping only the steps where *both* ends are cut voxels -- so that the allowed
``w -> 0`` step is excluded -- the largest single-step change is
(``tests/test_fd_shapes.py::test_a_crease_is_where_w_jumps_and_this_is_how_far``, 2026-09-21,
2400 steps per body):

    body                    steps   max |dw|   max |dfill|
    sphere                    668    0.0000      3.3e-03
    box, axis aligned         825    0.4455      3.3e-03
    box, rotated             1102    0.3028      3.0e-03
    hexagonal prism edge     1485    0.3222      1.6e-03
    cylinder cap rim         1485    1.0000      2.0e-03

so on a cap rim ``w`` swaps from the side wall's ``(0.5, 0.5, 0)`` to the cap's ``(0, 0, 1)``
inside one 0.0025-cell move. The design's G4 ("largest single-step change <= 1e-5") was measured
on a sphere and is out by five orders of magnitude there. An earlier writing of this paragraph
quoted a superseded dataset (an axis-aligned box at 0.0000) beside a test asserting the same
box jumps by more than 0.3; the table above is the test's own.

**The fill is smooth for every body with an exact fill, and -- since this round -- for a CSG
node too**: its worst single step over six node kinds on the same slide is **3.1e-03**, the same
size as a leaf's own, where three previous writings of that paragraph quoted 3.15e-02, 4.29e-02
and 4.48e-02. What is still discontinuous is a node's ``w``, by **4.91e-01** across a geometry
change of 2.8e-17, and that -- not the fill -- is now the whole of the reason a CSG node's
geometry may not be traced. See :class:`_Combination` for the mechanism, the cases the rule
decides between, the sign argument that makes a hole in solid material unwritable for each node
kind, what the rule costs measured on the shipped node, the price of the exact fix, and the one
refusal that is left -- which is at **plan time**, from the shapes, and not per voxel. The
per-voxel refusal the previous round shipped is gone: it fired on a legal T-junction at 16 of 16
random grid origins and could not fire at all under ``jax.jit``, where the same voxels read 0.75
against a true 1.0. What a node cannot answer exactly it now brackets, ships the safe end of, and
**counts** -- see :class:`Approximations`, which is the interface C2 and C5 warn from.
"""
from collections import namedtuple

import numpy as _np

from wavebit._core.backend import array_namespace

from ._cut import _axis_index, _parts, two_plane
from ._exact import (clipped_ball_volume, clipped_disc_area, clipped_polygon_area, overlap,
                     polygon_edge_moments)

#: The axis names a body may be extruded along, and the index each one is on a typed x-first
#: triple. Arrays in this package are ``[component, z, y, x]``; the triples here are x-first.
AXES = {'x': 0, 'y': 1, 'z': 2}

#: How many transform wrappers deep :func:`readable` looks. ``jax.vmap(jax.grad(...))`` is two
#: and ``torch.func.vmap(torch.func.grad(...))`` is two; the slack costs a ``getattr`` that fails.
_UNWRAP_DEPTH = 6

#: A turn is called convex when its sine clears this, and a sine is dimensionless -- which is
#: the whole point. The first writing compared the cross product against
#: ``1e-12 * max(|cross|, initial=1.0)``, an **absolute** floor below unit scale, so the same
#: star written in metres instead of micrometres was classified convex, took the convex clip
#: buffer and returned three different answers on three backends. The margin is one-sided on
#: purpose: a turn this file cannot call convex with confidence takes the larger buffer, which
#: is always correct and never faster.
_TURN_SINE = 1e-12

#: How close two tangent faces must be to count as **back to back**: ``1 + n_i . n_j`` below
#: this, and (for the interface) ``d_i + d_j`` below this many voxels. Both tests are
#: dimensionless -- ``1 + n_i . n_j`` is ``|n_i + n_j|^2 / 2`` between unit vectors, and
#: ``(d_i + d_j) / dx`` is a length over a length -- which is the same argument
#: :data:`_TURN_SINE` was rewritten for last round. A device's flush joint hits both
#: **exactly**: a box's top face at ``center + size/2`` and a cylinder's bottom at
#: ``center - height/2`` give ``n_i . n_j = -1`` and ``d_i + d_j = 0`` to the last bit when the
#: two were placed flush. The margin is there for the voxels a hair off that, where the last bit
#: decides; what it gives away is bounded in :func:`_no_common_volume`.
_SAME_FACE = 1e-12

#: A point is called **on the outline** when its distance to it is below this fraction of the
#: polygon's longest edge, and there the nearest edge's own outward normal is returned instead of
#: the offset vector's direction. The offset vector is ``q - a - e t``: at a point on the outline
#: it is the zero vector, and a hair off it, it is round-off whose *direction* is arbitrary.
#: Measured, a box face landing exactly on a voxel centre gave a distance of 5.9e-17 and a normal
#: of ``(-0.351, -0.936, 0)`` where the face's own is ``(-1, 0, 0)`` -- and that voxel is the one
#: a flush joint lands on. ``best > 0`` is an absolute predicate on a squared length, which is
#: the defect this file has now corrected three times.
_ON_THE_OUTLINE = 1e-12

#: Half the diagonal of a voxel, in voxels: the radius of the ball that contains it. A child
#: whose signed distance clears this is **wholly inside** the voxel, and that is a theorem and
#: not a margin -- ``|d|`` is at most the true distance to the surface for every child a CSG
#: node's rule admits, so no surface reaches the ball and none reaches the box inside it.
_CORNER = 3.0 ** 0.5 / 2

#: A child is called **flat** in a voxel when its own exact fill and its tangent half space's
#: agree to this fraction of the voxel. A convex body lies *inside* its own tangent half space,
#: so equality of the two volumes says the two sets differ by a null set: the body **is** the
#: half space in this voxel, and everything :func:`_cut.two_plane` proves about the half spaces
#: is then true of the bodies. The number is a fill difference, which is dimensionless -- the
#: same argument :data:`_TURN_SINE` and :data:`_cut._LIVE_AREA` were rewritten for. It is
#: one-sided in effect: a child this file cannot call flat simply has nothing subtracted from its
#: interface, which over-counts a joint and never invents one.
_FLAT = 1e-12

#: How wide a CSG node's own interval on a voxel's fill may be before the voxel is **counted**
#: as approximated and reported (:class:`Approximations`). It is a fraction of the voxel, so it
#: is dimensionless -- the same argument :data:`_TURN_SINE` and :data:`_FLAT` are written on.
#: The exact cases close the interval to **exactly** 0.0 (the two ends are the same expression
#: with the same operands), and the one near-exact case -- a pair of children proved flat -- to
#: at most ``2 _FLAT``, so this margin sits three decades above the only noise there is and
#: eighteen decades below anything a user would call an approximation.
_BOUNDED = 1e-9

#: What a CSG node knows about one child in one voxel. ``plane`` is the child's own tangent half
#: space's ``(fill, area, moment)`` and ``part`` is the child's own answer -- exact, because a
#: child with no exact fill is refused at construction. They are equal exactly where ``flat`` is
#: true. ``tangent`` is the static promise that the child lies **inside** its tangent half space,
#: which is what the ``outside`` test, the pair test and ``flat`` all rest on.
_Child = namedtuple('_Child', 'd n inside outside cross flat tangent plane part')

#: The one pair of crossing surfaces the exact two-plane cut is spent on, per voxel.
_Crease = namedtuple('_Crease', 'least most buried gate chosen')

#: One CSG node's own account of what it could not answer exactly in one call.
#: ``voxels`` and ``width`` are plain Python numbers, or ``None`` under a transform that carries
#: no number; ``counted`` and ``widest`` are the same two reductions as the backend's own 0-D
#: arrays, which is what a caller inside ``jax.jit`` returns from its jitted function to get them.
Approximation = namedtuple('Approximation', 'node voxels width counted widest')


def _number(value):
    """``value`` as a plain Python float, or ``None`` where there is no number to read.

    The scalar twin of :func:`readable` and it draws the line in the same place: under
    ``jax.jit`` a reduction is a tracer and there is nothing to read, exactly as
    :func:`_cut._unit` finds nothing to read there.
    """
    try:
        return float(value)
    except Exception:                   # noqa: BLE001 - a tracer has no number, and that is it
        return None


class Approximations:
    """Where a CSG node could not answer a voxel exactly, and by how much.

    A :class:`_Combination` answers **every** voxel -- there is no per-voxel refusal in this
    package any more -- but not every voxel exactly. Where the rule cannot close its interval on
    the true filled fraction it ships the end that is safe for its own operation (see
    :class:`_Combination`) and adds up two numbers here: how many voxels took a bound, and the
    widest interval it bracketed one with, in fractions of a voxel. Pass one to
    :func:`_occupancy.occupancy`::

        report = Approximations()
        fill, w = occupancy(node, points, dx, report=report)
        if report.voxels:
            warnings.warn(report.message())

    and it costs two reductions per node per image. Pass nothing -- the default -- and not even
    those are computed.

    **Why a collector and not a third return value.** ``occupancy`` answers ``(fill, w)`` and
    that pair is what C2 mixes; a third element would be in every caller's way for the benefit of
    the one caller that warns. A collector also holds a node's *own* name, which matters once a
    scene has several: ``entries`` is one row per node per periodic image, in the order they were
    evaluated, so a caller can say **which** body it could not resolve and not merely that
    something in the scene could not be.

    **Under ``jax.jit`` there is no number**, so ``voxels`` and ``width`` come back ``None`` --
    the same line :func:`readable` and :func:`_cut._unit` draw, and for the same reason. The two
    reductions are still on the entry, as ``counted`` and ``widest``, as the backend's own 0-D
    arrays; a jitted caller returns them from its own function and reads them outside. That is
    the difference between "cannot warn under jit" and "can", and it costs two scalars. Appending
    an entry is a Python side effect, so a **trace** appends one and a cached replay of that trace
    appends none: a collector handed to a jitted function is filled at trace time only, which is
    exactly when the arrays it must hold are created. Use a fresh collector per call if the
    accumulated list matters.
    """

    def __init__(self):
        self.entries = []

    def record(self, node, counted, widest):
        """One node's two reductions, named. Called by :class:`_Combination` alone."""
        voxels = _number(counted)
        self.entries.append(Approximation(node, None if voxels is None else int(round(voxels)),
                                          _number(widest), counted, widest))

    @property
    def voxels(self):
        """How many voxels took a bound, over every node and every image; ``None`` under jit."""
        if any(entry.voxels is None for entry in self.entries):
            return None
        return sum(entry.voxels for entry in self.entries)

    @property
    def width(self):
        """The widest interval any of them was bracketed with; ``None`` under jit, 0.0 if none."""
        if any(entry.width is None for entry in self.entries):
            return None
        return max((entry.width for entry in self.entries), default=0.0)

    def message(self):
        """One sentence a caller can warn with, or ``None`` when there is nothing to say."""
        if not self.voxels:
            return None
        worst = max((entry for entry in self.entries if entry.voxels), key=lambda e: e.width)
        return (f'{self.voxels} voxel(s) of this geometry were answered with a BOUND and not '
                f'exactly, the widest of them {self.width:.4g} of a voxel, on {worst.node}. A '
                'CSG node is exact where at most two of its children\'s surfaces share a voxel '
                'and both of those are flat in it; where three surfaces meet, or a curved wall '
                'crosses the same voxel as a joint, it brackets the filled fraction and ships '
                'the end that cannot remove material a Union or a Difference has, or add '
                'material an Intersection has not. Separating the third body by more than a '
                'voxel, or refining the grid, closes the interval.')


def _no_common_volume(d_a, n_a, d_b, n_b, dx, xp):
    """Do two tangent half spaces hold **no volume** of this voxel in common? Exactly.

    ``H = {n . (x - centre) + d <= 0}``. The pair holds volume in common iff
    ``min_x max(a(x), b(x)) < 0`` over the voxel, and by LP duality (both sets convex and
    compact, the inner function convex in ``x`` and linear in ``t``) that minimum equals

        max_{t in [0, 1]}  [ t d_a + (1-t) d_b - (dx/2) sum_k |t n_ak + (1-t) n_bk| ]

    which is concave and piecewise linear in ``t`` with at most three interior breaks, at
    ``t_k = n_bk / (n_bk - n_ak)``. So five evaluations answer it with no iteration, no tolerance
    and no branch on a value -- and the answer is **exact**, not conservative.

    ``>= 0`` and not ``> 0``: a common set of measure zero -- a shared edge, a corner touching
    the voxel wall, two coincident faces -- contributes no volume, and a pair that shares no
    volume is one a node can answer by adding its children. **No margin is carried on that
    boundary**, and one was: an earlier writing allowed ``-_SAME_FACE * dx``, because two boxes
    meeting on a plane read ``d_a + d_b`` as ``0.0`` down the middle of the joint and as
    ``-1e-18`` in 56 voxels off it. That was the fixed ``+u`` normal on the outline
    (:data:`_ON_THE_OUTLINE`) and not the arithmetic; with the nearest edge's own normal the two
    faces are exactly opposite and the pair reads exactly zero here. A margin no gate can make
    fail is a margin this package deletes.

    **The boundary is no longer observable, and that is a property of the round-6 rule and not a
    licence to move it.** Written ``> 0`` a flush pair stops being *proved* apart and becomes the
    pair the exact two-plane cut is spent on instead -- and :func:`_cut.two_plane`'s own parallel
    branch answers exactly what the split branch would have: shared volume ``max(F_a + F_b - 1,
    0)`` (zero for a joint) and both faces buried. Measured over 90 scene x grid-origin
    combinations of the device family, 14^3 voxels each: worst ``|dfill|`` **0.0** and worst
    ``|dw|`` **0.0**, bit for bit. Under the previous round's rule the same edit turned a
    T-junction of three boxes into a refusal, because a pair sharing a plane counted as a pair
    sharing the voxel; there is no refusal to trip now. ``>= 0`` stays because it is the true
    statement -- a common set of measure zero contributes no volume -- and because it is what
    keeps the cheap branch cheap: it spends no two-plane cut on a joint that has no volume in it.
    """
    zero, one = xp.zeros_like(d_a), xp.ones_like(d_a)
    knots = [zero, one]
    for k in range(3):
        den = n_b[..., k] - n_a[..., k]
        live = den != 0
        safe = xp.where(live, den, xp.ones_like(den))
        knots.append(xp.clip(xp.where(live, n_b[..., k] / safe, zero), zero, one))
    worst = None
    for t in knots:
        g = t * d_a + (1 - t) * d_b
        for k in range(3):
            g = g - (dx / 2) * xp.abs(t * n_a[..., k] + (1 - t) * n_b[..., k])
        worst = g if worst is None else xp.maximum(worst, g)
    return worst >= 0


def _module_of(value):
    """The top-level package a value's type comes from -- ``'jax'``, ``'torch'``, ``'numpy'``..."""
    return type(value).__module__.split('.')[0]


def _read_once(value):
    """One attempt to read ``value`` as a NumPy array; ``None`` when this layer holds no number.

    Four spellings, because no one of them reaches every wrapper: ``asarray`` for a plain value
    and a detached Torch tensor, ``cpu().numpy()`` for a Torch tensor on a device, ``item()``
    for a scalar a transform is watching (this is the one that reaches ``torch.func.grad`` and
    an un-jitted ``jax.grad``), and ``stop_gradient`` for a JAX *array* leaf under ``jax.grad``,
    where ``item()`` has no scalar to return.

    **The dtype is read, not imposed.** The first writing asked for ``dtype=float``, which for a
    complex array silently discards the imaginary part: ``Prism(HEX + 0.03j)`` passed every door
    on its real part and was then handed whole to the kernel by :func:`_as`, so the check and the
    computation saw two different bodies and the volume came back 10.4% wrong. What comes back
    here is whatever dtype the value has; :func:`readable` is where a dtype that is not a real
    number is refused.
    """
    detach = getattr(value, 'detach', None)
    if detach is not None:
        try:
            value = detach()
        except Exception:                   # noqa: BLE001 - a wrapper that will not detach
            pass
    for read in (_np.asarray,
                 lambda v: _np.asarray(v.cpu().numpy()),
                 lambda v: _np.asarray(v.item())):
        try:
            return read(value)
        except Exception:                   # noqa: BLE001 - try the next spelling
            continue
    if _module_of(value) == 'jax':
        import jax                          # local: jax is optional
        try:
            return _np.asarray(jax.lax.stop_gradient(value))
        except Exception:                   # noqa: BLE001 - a jit or batch tracer
            return None
    return None


def _unwrapped(value):
    """The value one transform wrapper down, or ``None`` when this is the bottom.

    ``primal`` is a JAX differentiation tracer's own name for the value it carries, ``val`` is a
    batch tracer's, and ``get_unwrapped`` is functorch's. They are internal names, read through
    ``getattr`` inside a ``try``, so a rename costs the door under that one transform and
    nothing else -- and the table in :func:`readable` is a test, so a rename goes red rather
    than quiet.
    """
    for name in ('primal', 'val'):
        inner = getattr(value, name, None)
        if inner is not None and _module_of(inner) in ('jax', 'jaxlib', 'numpy'):
            return inner
    if _module_of(value) == 'torch':
        try:
            import torch._C._functorch as functorch     # local: torch is optional
            return functorch.get_unwrapped(value)
        except Exception:                   # noqa: BLE001 - a plain tensor has nothing under it
            return None
    return None


def readable(value, where='a geometry value'):
    """``value`` as a plain NumPy float array **for a check**, or ``None`` when there is no value.

    The array twin of ``wavebit._core.backend.checked_float``. Reading a differentiated value to
    *check* geometry is the case that most needs checking -- the memory this repository carries
    is a rod grown by gradient ascent to three times its period with every refusal silently
    skipped, because the check had been spelled ``float(x)``.

    **Which transforms it holds under, measured and not asserted -- every row of this table is a
    row of** ``tests/test_fd_shapes.py::test_a_door_holds_under_every_transform_that_carries_a_value``:

        transform                               read?  how
        a float, a list, NumPy                   yes   asarray
        a concrete jax.Array, a Torch tensor     yes   asarray (detached first)
        torch autograd, ``requires_grad``        yes   detach, asarray
        jax.grad, scalar leaf                    yes   item
        jax.grad, array leaf                     yes   stop_gradient
        jax.vmap, jax.vmap(jax.grad)             yes   the batch tracer's value, one layer down
        torch.func.grad / jacrev / jacfwd        yes   item, or functorch's unwrapped tensor
        torch.func.vmap, torch.func.vmap(vmap)   yes   functorch's unwrapped tensor
        **jax.jit, jax.jit(jax.grad)**           NO    there is no number anywhere to read
        **torch.func.vmap(grad), jacrev(vmap)**  NO    the same, two functorch layers down

    Two of the rows are dead ends and the shipped table claimed only one. Under ``jax.jit`` a
    ``DynamicJaxprTracer`` carries a ``Var`` and not a value, and a *jitted* gradient is one too.
    Under ``torch.func.vmap`` **composed with a grad transform** the unwrapping reaches a third
    ``torch.Tensor`` that has no storage, and every spelling fails on it -- measured
    (2026-09-21): ``item()``, ``detach().item()``, ``float()`` and ``unwrap_if_dead().item()``
    all raise ``vmap: It looks like you're ... calling .item() on a Tensor``, and
    ``detach().cpu().numpy()`` raises ``Cannot access data pointer of Tensor that doesn't have
    storage``. ``torch.func.vmap(torch.func.vmap(...))`` is **not** a dead end -- the same three
    layers, and the third one reads -- so it is the *grad* layer under the batch layer that has
    no number, not the depth. The table said "yes" for ``vmap(grad)`` for two rounds;
    ``Sphere(radius=-0.3)`` comes back NaN there, which is what :func:`_loud_if_not_a_size` puts
    where a door cannot refuse, and not the refusal the table promised.

    The first writing of this function claimed ``jax.jit`` was the only dead end and was wrong on
    three the other way: ``torch.func.grad`` came back ``None`` and ``Sphere(radius=-0.3)`` was
    accepted there with a finite gradient of 135.00000000000014 -- this repository's own
    gradient-ascent scar, in the modern Torch functional API, which ``checked_float`` (the
    function this one calls itself the twin of) reads correctly. ``jax.vmap`` and
    ``torch.func.vmap`` were the same hole.

    **A value that is not a real number is refused here rather than cast.** ``_np.asarray(v,
    dtype=float)`` -- what this used to ask for -- silently drops a complex value's imaginary
    part, and :func:`_as` then hands the *original* to the kernel, so the door and the
    computation see two different numbers: measured, ``Prism(HEX + 0.03j)`` was accepted and read
    a volume 10.4% away from the real hexagon's, and ``Box(size=[0.34+0.2j, ...])`` returned a
    complex ``fill``. The same branch is what refuses a string or a dict by name instead of
    deferring it to ``array_namespace`` (memory: typos are refused on every door).
    """
    for _ in range(_UNWRAP_DEPTH):
        read = _read_once(value)
        if read is not None:
            if read.dtype.kind == 'c':
                raise ValueError(f'{where}: a real number, and this is complex; got '
                                 f'{_np.asarray(read).ravel()[0]!r}. A geometry value is a '
                                 'length, a coordinate or an angle -- all of them real. (This '
                                 'repository refuses a complex eps by name for the same reason: '
                                 'a check reads the real part and the kernel uses the whole '
                                 'number, so the two see different bodies.)')
            if read.dtype.kind == 'b':
                raise ValueError(f'{where}: a real number, and this is a bool; got '
                                 f'{_np.asarray(read).ravel()[0] != 0!r}. A bool passes every '
                                 'test a length has to pass -- it is finite and True is '
                                 'positive -- so without this branch Sphere(radius=True) is a '
                                 'unit sphere, Box(size=(True, True, True)) is a unit cube and '
                                 'occupancy(..., dx=True) silently solves at dx = 1.0 (measured: '
                                 'fill 0.03351032163829113 for Sphere(0.2)). A length is a '
                                 'number, and a flag is not one.')
            if read.dtype.kind not in 'iuf':
                if read.dtype.kind == 'O' and _module_of(value) in ('jax', 'jaxlib', 'torch'):
                    return None             # a tracer's internals (a jax `Var`), not a value
                raise ValueError(f'{where}: a real number, and this is not a number at all '
                                 f'({type(value).__name__}, read as dtype {read.dtype}).')
            return read.astype(float)
        inner = _unwrapped(value)
        if inner is None:
            return None
        value = inner
    return None


def _loud_if_not_a_size(value):
    """``value`` itself where it is a size, and ``value * nan`` where it is not.

    What a door does where it cannot raise. Under ``jax.jit`` there is no value to read, so the
    check becomes an array expression instead of a Python one: the factor is **exactly** 1.0
    whenever every element is positive and finite, and NaN otherwise. That the factor is
    exactly one is held bitwise in
    ``tests/test_fd_shapes.py::test_a_door_holds_under_every_transform_that_carries_a_value``
    and cannot be held by comparing a jitted result with an eager one, because XLA reassociates
    and those differ in the last bits whatever this function does.
    One scalar operation per body, not one per voxel. A door that cannot refuse must
    not hand back a plausible number instead: the measured alternative is ``fill = 16.0`` with
    ``sum_c w_c = 0`` from ``Cylinder(radius=-0.2)``, both free invariants broken at once.
    """
    xp = array_namespace(value)
    fine = xp.all(value > 0) & xp.all(xp.isfinite(value))
    one = xp.ones_like(value)
    return value * xp.where(fine, one, one * float('nan'))


def check_sizes(where, **named):
    """Refuse a non-positive length by name, and give back the value the body should keep.

    ``a length is a size``, and a size is positive -- the repository's own sentence and its own
    scar: a negative ``lx`` on the RCWA path *subtracted* the shape instead of refusing.
    Measured on this kernel before these doors, from three constructors whose docstrings declare
    a length: ``Cylinder(radius=-0.2)`` read ``fill = 16.0`` with ``sum_c w_c = 0`` -- both
    invariants broken at once, because ``clipped_disc_area`` with ``r < 0`` returns sixteen
    times the box it is clipped to while the plane reports no interface;
    ``Box(size=(-0.3, 0.2, 0.2))`` read 0.0000 and ``Sphere(radius=-0.2)`` read 0.0000. Three
    bodies, three different wrong answers, no refusal anywhere.

    The values come back because of the two transforms that carry no value at all: under
    ``jax.jit``, and under ``torch.func.vmap`` composed with a grad transform, the refusal cannot
    raise, so what is returned is :func:`_loud_if_not_a_size` of the length instead of the
    length. Everywhere else -- including under ``jax.grad``, ``jax.vmap``, Torch autograd and
    every *single* ``torch.func`` transform -- this raises, and the value comes back untouched.
    :func:`readable` holds the table of which is which.
    """
    out = []
    for what, value in named.items():
        if value is None:
            raise ValueError(f'{where}({what}=...): a length, and there is none; got None.')
        read = readable(value, where=f'{where}({what}=...)')
        if read is None:
            out.append(_loud_if_not_a_size(value))
            continue
        if not _np.all(_np.isfinite(read)):
            raise ValueError(f'{where}({what}=...): a finite length; got {read.tolist()}.')
        if _np.any(read <= 0):
            raise ValueError(f'{where}({what}=...): a length is a size and a size is positive; '
                             f'got {read.tolist()}. To place the body elsewhere use center=.')
        out.append(value)
    return tuple(out)


def check_reals(where, **named):
    """Refuse a geometry value that is not a real number by name, and give it back.

    :func:`check_sizes`' sibling for the values that are **not** sizes -- a centre may be
    negative and an angle may be anything -- but that are still real. Without it the complex door
    would cover only half the constructor: measured before this, ``Sphere(center=[0.01+0.5j, 0,
    0])`` was accepted and returned a ``fill`` of dtype ``complex128``, because ``_as`` hands a
    value that already is an array straight to the kernel. Nothing is checked under a transform
    that carries no value; :func:`readable` says which those are.
    """
    for what, value in named.items():
        if value is not None:
            readable(value, where=f'{where}({what}=...)')
    return tuple(named.values())


def traced(value):
    """Is ``value`` one a transform is watching -- a tracer, a functorch wrapper, an autograd leaf?

    Separate from :func:`readable`, and it has to be: a ``jax.grad`` tracer *is* readable and
    *is* traced, which is exactly the pair a door wants to tell apart. Used by
    :class:`_Combination` alone, which refuses a traced parameter outright; see its docstring
    for the measured reason.
    """
    if isinstance(value, (tuple, list)):
        return any(traced(v) for v in value)
    if any(base.__name__ == 'Tracer' for base in type(value).__mro__):
        return True
    if getattr(value, 'requires_grad', False):
        return True
    if _module_of(value) == 'torch':
        try:
            import torch._C._functorch as functorch     # local: torch is optional
            functorch.get_unwrapped(value)
        except Exception:                   # noqa: BLE001 - a plain tensor, and that is the answer
            return False
        return True
    return False


def _crossing(p, q, r, s):
    """Do the closed segments ``pq`` and ``rs`` meet? Orientation test, collinear cases included."""
    def side(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return 0 if v == 0 else (1 if v > 0 else -1)

    def between(a, b, c):
        return (min(a[0], b[0]) <= c[0] <= max(a[0], b[0])
                and min(a[1], b[1]) <= c[1] <= max(a[1], b[1]))

    d1, d2, d3, d4 = side(r, s, p), side(r, s, q), side(p, q, r), side(p, q, s)
    if d1 * d2 < 0 and d3 * d4 < 0:
        return True
    return ((d1 == 0 and between(r, s, p)) or (d2 == 0 and between(r, s, q))
            or (d3 == 0 and between(p, q, r)) or (d4 == 0 and between(p, q, s)))


def polygon_shape(where, vertices):
    """``(nv, convex)`` after refusing everything ``clipped_polygon_area`` is not written for.

    Three refusals, each measured on this kernel before it had them:

    * **fewer than three vertices** -- ``clipped_polygon_area``'s own docstring says ``nv >= 3``
      and a two-vertex ``Prism`` returned 0.0000 rather than saying so.
    * **clockwise** -- ``clipped_polygon_area`` ends with the source's rounding guard
      ``where(area < 0, 0, area)``, so a reversed vertex list reads area 0 in every cell while
      nothing else in the body is orientation-sensitive: the even-odd inside test and
      ``polygon_edge_moments`` (which returns ``n_c^2``) are both orientation-blind. Measured on
      ``HEX`` against ``HEX[::-1]`` at ``h = 0.05``: volume 0.030951 (true 0.030951) against
      **0.000000**, with an identical signed distance and 240 voxels still reporting
      ``sum_c w_c = 1``. A body that is not there, with a live interface.
    * **self-intersecting** -- the shoelace sum is a *signed* area, so a figure-eight returns
      the difference of its lobes and the even-odd test disagrees with it.

    ``convex`` is returned rather than refused: it is a static promise
    (:func:`_exact.clip_capacity`) that buys the smaller clip buffer, and a simple non-convex
    cross-section is a legal body -- ``fd.Union`` of convex prisms is not the only spelling.

    **Every test here is scale-free**, which the first writing's convexity test was not (see
    :data:`_TURN_SINE`). The orientation and crossing tests are sign tests on exact products, so
    they say the same thing at any scale by construction; the turn test is a *sine*, a ratio of
    two lengths squared, for the same reason. Measured over the six awkward outlines in the gate
    at 1e-9, 1e-6, 1e-3, 1, 1e+3 and 1e+6: every classification is identical at all six scales,
    where the shipped test read ``convex`` for all six below 1e-5
    (``tests/test_fd_shapes.py::test_every_geometric_predicate_is_free_of_the_coordinate_scale``).

    **What is checked when there is no value to read.** A ``jax.jit`` tracer has a *shape*, so
    the shape and the vertex count are still refused there; the orientation is turned into a NaN
    rather than a refusal (:func:`_loud_if_not_a_size`'s reason, in :class:`Prism`); and the
    self-intersection test, which is an O(n^2) walk over concrete points, is skipped. A vertex
    list that is not an array of numbers at all -- a ragged list, a string -- is **not** a
    tracer and is refused by name rather than deferred to an obscure failure at solve time
    (memory: typos are refused on every door).
    """
    read = readable(vertices)
    shape = tuple(read.shape) if read is not None else getattr(vertices, 'shape', None)
    if shape is None:
        raise ValueError(f'{where}(vertices=...): a sequence of (u, v) pairs, shape (nv, 2); '
                         f'this is not an array of numbers at all '
                         f'({type(vertices).__name__}). A ragged list is the usual cause.')
    shape = tuple(shape)
    if len(shape) != 2 or shape[-1] != 2:
        raise ValueError(f'{where}(vertices=...): a sequence of (u, v) pairs, shape (nv, 2); '
                         f'got shape {shape}.')
    nv = shape[0]
    if nv < 3:
        raise ValueError(f'{where}(vertices=...): a polygon has at least three vertices; '
                         f'got {nv}.')
    if read is None:
        return nv, False                    # a jit tracer: the shape is static, the values not
    if not _np.all(_np.isfinite(read)):
        raise ValueError(f'{where}(vertices=...): every vertex must be finite.')
    points = [tuple(v) for v in read]
    for i in range(nv):
        for j in range(i + 2, nv):
            if i == 0 and j == nv - 1:
                continue                    # adjacent round the ring
            if _crossing(points[i], points[(i + 1) % nv], points[j], points[(j + 1) % nv]):
                raise ValueError(f'{where}(vertices=...): a simple polygon -- edges {i} and {j} '
                                 'meet. A self-intersecting outline has no filled fraction to '
                                 'report (the shoelace sum returns the difference of its lobes '
                                 'and the even-odd inside test disagrees with it); build it as '
                                 'an fd.Union of simple pieces instead.')
    following = _np.roll(read, -1, axis=0)
    area = .5 * float(_np.sum(read[:, 0] * following[:, 1] - following[:, 0] * read[:, 1]))
    if area == 0:
        raise ValueError(f'{where}(vertices=...): the outline encloses no area -- its vertices '
                         f'are collinear or repeated -- so there is no body to fill.')
    if area < 0:
        raise ValueError(f'{where}(vertices=...): counter-clockwise, in the right-handed (u, v) '
                         f'pair across the extrusion axis; this list has signed area {area:.6g}. '
                         'Reverse it (vertices[::-1]) -- the clipped area of a clockwise list is '
                         'zero in every cell and the body silently disappears.')
    edge = following - read
    turn = _np.roll(edge, -1, axis=0)
    cross = edge[:, 0] * turn[:, 1] - edge[:, 1] * turn[:, 0]
    sine = _np.hypot(edge[:, 0], edge[:, 1]) * _np.hypot(turn[:, 0], turn[:, 1])
    return nv, bool(_np.all(cross > _TURN_SINE * _np.maximum(sine, _np.finfo(float).tiny)))


def _loud_if_not_counter_clockwise(vertices):
    """``1.0`` where the shoelace sum is positive, NaN where it is not -- a factor on the area.

    :func:`_loud_if_not_a_size`'s twin, for the one polygon refusal that is worth an array
    expression: a clockwise outline reads area 0 in every cell and the body **silently
    disappears** while its interface stays live, which is the quietest failure in this package.
    Reached only when there is no value to read -- a ``jax.jit`` tracer -- and exactly ``1.0``
    for a legal outline, so a jitted solve is bit-identical to an eager one. The vertex count
    and the array's shape are static and are refused properly even there; the O(n^2)
    self-intersection walk is over concrete points and is skipped, which
    :func:`polygon_shape`'s docstring says.

    **It multiplies the area and not the vertices**, which the first writing tried: NaN
    vertices come back as a silent **0.0**, not as a NaN, because the clip's every comparison
    (``da >= 0``) is ``False`` for a NaN, so the survivor count falls to zero and the shoelace
    sum is empty. A guard that a downstream comparison launders back into the failure it was
    guarding is not a guard; measured, ``Prism(HEX[::-1])`` under ``jax.jit`` read ``fill =
    0.0`` with the vertices poisoned and reads ``nan`` with the area poisoned.
    """
    xp = array_namespace(vertices)
    following = xp.roll(vertices, -1, axis=0)
    area = .5 * xp.sum(vertices[:, 0] * following[:, 1] - following[:, 0] * vertices[:, 1])
    one = xp.ones_like(area)
    return xp.where(area > 0, one, one * float('nan'))


def _as(xp, value, like):
    """``value`` as an array beside ``like`` -- and untouched if it already is one of ``like``'s.

    ``device=`` is read off ``like`` with a default because a JAX tracer has a ``dtype`` and no
    ``device``: inside ``jax.jit`` the placement is the trace's, not an attribute to copy.

    **The test is the namespace and not the attributes.** The first writing asked for ``dtype``
    and ``device``, which under numpy 2.5.1 a plain ``ndarray`` has *both* of -- so a NumPy-array
    parameter was never converted and then met a Torch tensor:
    ``Prism(np.asarray(HEX), ...)`` worked on NumPy and JAX and died on Torch with
    ``TypeError: unsupported operand type(s) for *: 'numpy.ndarray' and 'Tensor'``, a bare
    failure deep inside ``_vertices`` naming no door, and inconsistent inside one body (a NumPy
    ``center`` went through ``xp.asarray`` and worked). Comparing namespaces keeps a traced value
    untouched -- a tracer's namespace *is* the backend's -- and converts a foreign concrete one.
    """
    if hasattr(value, 'dtype') and hasattr(value, 'device'):
        try:
            if array_namespace(value) is xp:
                return value
        except Exception:                   # noqa: BLE001 - a NumPy scalar, say: convert it
            pass
    return xp.asarray(value, dtype=like.dtype, device=getattr(like, 'device', None))


def _transverse(axis):
    """The two axes across ``axis``, in the cyclic order that keeps ``(u, v, axis)`` right-handed.

    ``z -> (x, y)``, ``x -> (y, z)``, ``y -> (z, x)``. Right-handed is what makes
    "counter-clockwise" mean one thing for a prism whatever axis it is extruded along.
    """
    k = AXES[axis]
    return (k + 1) % 3, (k + 2) % 3


def _stack3(xp, parts, order, like):
    """A typed x-first triple from ``parts`` given for axes ``order``."""
    out = [None, None, None]
    for value, k in zip(parts, order):
        out[k] = value
    return xp.stack([p if p is not None else xp.zeros_like(like) for p in out], axis=-1)


class Body:
    """A shape that answers for a point, not for a cell. See the module docstring."""

    #: Does this body lie inside its own tangent half space at every surface point? It is a
    #: **static** property of the class (or, for a :class:`Prism`, of its outline), and the one
    #: thing :class:`_Combination`'s seam rule needs: ``H = {n.(x - p) <= -d}`` contains a convex
    #: body, so two convex children whose tangent half spaces hold no volume of a voxel in common
    #: cannot themselves overlap in it. For a non-convex body the containment fails and the rule
    #: is switched off rather than applied unsoundly.
    convex = False

    #: Does :meth:`exact_fill` return a closed form rather than ``None``? A CSG node reads it to
    #: decide whether a child can be proved **flat** in a voxel (:data:`_FLAT`): the proof is
    #: ``exact fill == half-space fill``, and a body with no exact fill answers *with* the half
    #: space, so the comparison would be true by construction and would prove nothing.
    has_exact_fill = False

    def leaves(self):
        """The geometry values this body was built from -- what a design loop would differentiate.

        Read by :class:`_Combination` alone, to refuse a traced one; see its docstring for the
        measured reason. Every concrete body here overrides it, and a body that does not returns
        an empty tuple, which means a new subclass that forgets to declare its parameters is
        admitted into a CSG node with them traced. That is the one hole in this refusal and it
        is inside this package, where a new body and its ``leaves`` are the same edit.
        """
        return ()

    def surface(self, points):
        raise NotImplementedError

    def exact_fill(self, points, dx):
        return None

    def exact_moment(self, points, dx):
        return None

    def exact_edge_fill(self, points, dx, axis):
        """The exact fraction of the component's own **edge** this body holds, or ``None``.

        The edge is the segment of length ``dx`` along ``axis`` centred on each point -- what a
        component's line integral of E runs along. ``None`` means "no closed form here", exactly
        as :meth:`exact_fill` does, and the caller must then decide, **in the open**, to use
        ``_cut.edge_cut``'s tangent plane instead.

        There is deliberately no dispatcher pairing the two. The plane's answer is not a
        drop-in for the exact one: it is the plane's, and on a curved body it differs at first
        order in the curvature (``tests/test_fd_shapes.py::
        test_the_plane_is_not_a_drop_in_for_the_exact_edge_and_face_fills`` measures how far).
        A ``parts``-style fallback would let that substitution happen silently, in a routine
        whose every measured convergence order was taken with the exact form.
        """
        return None

    def exact_face_fill(self, points, dx, axis):
        """The exact fraction of the component's own **dual face** this body holds, or ``None``.

        The dual face is the ``dx`` by ``dx`` square normal to ``axis`` centred on each point --
        what a component's flux of D crosses. The twin of :meth:`exact_edge_fill`, and its
        docstring is this one's.
        """
        return None

    def plane(self, points, dx):
        """``(fill, area, moment)`` from the tangent plane -- what every body falls back to.

        Goes to ``_cut._parts`` rather than the exported ``cut_parts``: the bodies here
        *guarantee* a unit normal, so the door on the public spelling would be a reduction and a
        host synchronisation per body per image, paid on every voxel of every solve to check a
        precondition this file establishes by construction.
        """
        d, n = self.surface(points)
        return _parts(d, n, dx)

    def parts(self, points, dx, surface=None, report=None):
        """``(fill, area, moment)``: the exact forms where they exist, the plane where they do not.

        The one call :func:`_occupancy.occupancy` makes per body per image, and the one place the
        table at the top of this module is realised. ``surface=`` is the already-computed
        ``(d, n)`` where the caller has it -- :class:`_Combination` needs its children's normals
        for the seam rule and would otherwise walk every child's distance twice. ``report=`` is
        an :class:`Approximations` a node adds its two reductions to; a leaf has nothing to add,
        because a leaf's answer is its own closed form and not a bound.
        """
        fill = self.exact_fill(points, dx)
        pair = self.exact_moment(points, dx)
        if fill is None or pair is None:
            d, n = self.surface(points) if surface is None else surface
            plane_fill, plane_area, plane_moment = _parts(d, n, dx)
            fill = plane_fill if fill is None else fill
            pair = (plane_area, plane_moment) if pair is None else pair
        return (fill,) + pair


class Sphere(Body):
    """``|x - centre| <= radius``, with an **exact** fill and the plane's moment.

    At the centre the gradient does not exist and the normal returned is ``+x``. Nothing can
    observe it: the fill there is 1 for any direction whenever ``radius >= dx sqrt(3)/2``, and a
    sphere smaller than a voxel is refused for other reasons (C5's "too few cells across").

    **Why the exact fill is written here after all.** Study S3(a) and the first draft of this
    package both decided the tangent plane was enough, on the grounds that its ``+0.250 (h/R)^2``
    over-fill is second order and one-signed, so the convergence study that is the gate fits it
    away. Two measurements taken by the C1 review overturned that. S2's companion claim -- order
    1.83 on a curved interface, the number the whole smoothing rule is sold on -- was itself
    measured with an *exact* fill, so it says nothing about a plane-filled sphere; and the
    review then measured the tangent plane's bias on top of it at 0.99x to 1.19x S2's own fitted
    errors at S2's own four resolutions, same sign. Together they read about twice the error at
    an order that still looks like two. The sphere is stage 4's pass mark (C3, Mie), so the one
    body the grade is taken on may not be the one body carrying a removable bias.

    The tangent plane stays for :class:`Union`, :class:`Intersection`, :class:`Difference` and
    anything else with no closed form; what it costs there is measured in ``_cut.py``.

    **The moment is still the plane's.** The previous explanation that an exact diagonal
    moment needs volume second moments was incorrect: in sphere-centred coordinates,
    ``div(x_i e_i)=1`` and ``x_i n_i=R n_i^2`` on the sphere, so
    ``R M_i=V-(u_i A_i+ - l_i A_i-)``. Only the volume and six clipped face areas are needed.
    ``studies/fdtd_subpixel_constitutive.py`` implements and independently checks this identity.
    Replacing only the moment changes the origin glass sphere's 40-cell scattering efficiency
    by 5.65e-6 relative and does not restore its convergence order. The research expression
    has not replaced this production fallback. What the plane's moment costs here is
    ``tests/test_fd_shapes.py::test_w_vanishes_only_where_the_fill_is_already_decided``'s
    measured sphere row.

    **It is also the one body that answers for a component's edge and dual face exactly**
    (:meth:`exact_edge_fill`, :meth:`exact_face_fill`), because a ball's chord along a line and
    its cross-section in a plane are the two closed forms this package already owns, one
    dimension down. Every research number the C3 material operator has measured was taken with
    those two, which is why the tangent-plane forms beside them are not wired in as a fallback.
    """

    #: A ball is convex, and its fill is a closed form.
    convex = has_exact_fill = True

    def __init__(self, radius, center=(0., 0., 0.)):
        self.radius, = check_sizes('Sphere', radius=radius)
        self.center, = check_reals('Sphere', center=center)

    def leaves(self):
        return (self.radius, self.center)

    def exact_fill(self, points, dx):
        xp = array_namespace(points)
        centre = _as(xp, self.center, points)
        half = dx / 2
        return clipped_ball_volume(
            centre[..., 0], centre[..., 1], centre[..., 2], _as(xp, self.radius, points),
            points[..., 0] - half, points[..., 0] + half,
            points[..., 1] - half, points[..., 1] + half,
            points[..., 2] - half, points[..., 2] + half, xp) / dx ** 3

    def exact_edge_fill(self, points, dx, axis):
        """The chord of the ball inside the segment, clipped to it -- exactly, in cell units.

        ``|x - centre| <= radius`` restricted to a line along ``axis`` is an interval of
        half-length ``sqrt(radius^2 - transverse^2)`` about the foot of the centre, so the
        answer is two clips and a subtraction. Ported from
        ``studies/fdtd_subpixel_triplet_sphere.py::edge_fraction``, which is now a wrapper over
        this, with both of its guards:

        * the root is taken of a value a :func:`xp.where` has already replaced by one where the
          line misses the ball, so an inactive ``sqrt`` cannot hand a NaN back through a
          gradient (this package's own rule, ``_cut._second_difference``'s docstring);
        * the two ends are clipped **in segment coordinates** -- ``+-1/2`` against
          ``(+-half - v)/dx`` -- and not in world coordinates. The study measured the world
          spelling returning ``1 + 8.88178e-16`` for a segment wholly inside a radius-one
          circle at ``dx = 0.1``; a fraction that exceeds one is not a rounding detail to a
          rule that divides by ``1 - f``.

        C0 and not C1, like the tangent-plane form it replaces: ``maximum(0, hi - lo)`` and the
        two clips are its three corners. That is the exact function and not an approximation of
        it (``_cut.py``'s section note on the two forms).

        ``dx`` is doored here, as it is on ``_cut.edge_cut`` beside it: this is a public method
        anyone may call with a length of their own, it is *not* reached through
        :func:`~wavebit.fd._shapes.occupancy`'s door the way :meth:`exact_fill` is, and the two
        clips below are written in units of it. Measured with the door removed, on three points
        about a radius-0.25 ball at ``|dx| = 0.05``: a **negative** ``dx`` divides both ends by
        it and swaps them, and the two segments that read ``1.0`` at ``+0.05`` read ``0.0`` --
        a wholly filled edge reported as empty, which is the failure mode ``check_sizes`` was
        written for (a length is a size). A ``dx`` of zero divides, and the edge then reads
        ``1.0`` through a clip of an infinity while the face reads ``nan``.
        """
        xp = array_namespace(points)
        dx, = check_sizes('Sphere.exact_edge_fill', dx=dx)
        axis = _axis_index('Sphere.exact_edge_fill', axis)
        v = points - _as(xp, self.center, points)
        radius = _as(xp, self.radius, points)
        transverse = sum(v[..., j] ** 2 for j in range(3) if j != axis)
        live = radius ** 2 > transverse
        half = xp.sqrt(xp.where(live, radius ** 2 - transverse, xp.ones_like(transverse)))
        lo = xp.maximum(xp.zeros_like(half) - .5, (-half - v[..., axis]) / dx)
        hi = xp.minimum(xp.zeros_like(half) + .5, (half - v[..., axis]) / dx)
        return xp.where(live, xp.maximum(xp.zeros_like(lo), hi - lo), xp.zeros_like(lo))

    def exact_face_fill(self, points, dx, axis):
        """The ball's cross-section clipped to the dual square -- the disc clip, one plane down.

        A plane ``x_axis = const`` cuts the ball in a disc of radius
        ``sqrt(radius^2 - offset^2)`` concentric with the centre's foot, so the exact area
        inside the square is :func:`_exact.clipped_disc_area` -- the form this package already
        owns and already gates. Ported from
        ``studies/fdtd_subpixel_triplet_sphere.py::face_fraction``, which is now a wrapper over
        this, with its guard: the **full** and **empty** cases are settled first, by the square's
        nearest and farthest corners, and the disc clip is then evaluated on inputs a
        :func:`xp.where` has already made describe a unit disc and a centred square wherever it
        is inactive. Without that, an inactive negative radius reaches a square root and an
        inactive ``asin`` reaches its branch point, and both are finite forwards and NaN
        backwards -- the defect family ``tests/test_fd_shapes.py`` records three rows of.

        Everything is done in cell units (``u``, ``v``, ``r2`` are all divided by ``dx`` first),
        so the answer is a fraction of the square and not an area, and full is exactly 1.

        ``dx`` is doored here for the reason :meth:`exact_edge_fill` gives, and the measurement
        is the other half of the argument for putting the check at the door rather than
        trusting the arithmetic. With the door removed, on the same three points: ``dx = 0``
        reads ``nan`` here (the edge reads ``1.0``), while a **negative** ``dx`` reads exactly
        what its positive twin reads, because every quantity below is squared -- so the same
        invalid input is silently wrong on the edge, silently right on the face, and a caller
        comparing the two would learn nothing. Only a refusal makes those two agree.
        """
        xp = array_namespace(points)
        dx, = check_sizes('Sphere.exact_face_fill', dx=dx)
        axis = _axis_index('Sphere.exact_face_fill', axis)
        v = points - _as(xp, self.center, points)
        radius = _as(xp, self.radius, points)
        others = tuple(j for j in range(3) if j != axis)
        u, w = (v[..., j] / dx for j in others)
        r2 = (radius / dx) ** 2 - (v[..., axis] / dx) ** 2
        nearest = (xp.maximum(abs(u) - .5, xp.zeros_like(u)) ** 2
                   + xp.maximum(abs(w) - .5, xp.zeros_like(w)) ** 2)
        farthest = (abs(u) + .5) ** 2 + (abs(w) + .5) ** 2
        empty, full = r2 <= nearest, r2 >= farthest
        partial = (~empty) & (~full)
        rad = xp.sqrt(xp.where(partial, r2, xp.ones_like(r2)))
        a = xp.where(partial, u, xp.zeros_like(u))
        b = xp.where(partial, w, xp.zeros_like(w))
        area = clipped_disc_area(0., 0., rad, a - .5, a + .5, b - .5, b + .5, xp)
        return xp.where(full, xp.ones_like(area), xp.where(empty, xp.zeros_like(area), area))

    def surface(self, points):
        xp = array_namespace(points)
        v = points - _as(xp, self.center, points)
        r2 = xp.sum(v * v, axis=-1)
        live = r2 > 0
        r = xp.where(live, xp.sqrt(xp.where(live, r2, xp.ones_like(r2))), xp.zeros_like(r2))
        safe = xp.where(live, r, xp.ones_like(r))[..., None]
        fallback = xp.stack([xp.ones_like(r), xp.zeros_like(r), xp.zeros_like(r)], axis=-1)
        return r - _as(xp, self.radius, points), xp.where(live[..., None], v / safe, fallback)


class _Extrusion(Body):
    """A plane figure swept along a grid axis: the half of a body that Box, Prism and Cylinder share.

    The exact signed distance of the intersection of an infinite prism with a slab, in the
    spelling that is exact rather than merely conservative (Quilez's elongation form; the same
    combination the boundary path's rounded joins use)::

        d = min(max(d_r, d_a), 0) + |max((d_r, d_a), 0)|

    -- outside both, the distance is the hypotenuse and the normal mixes the two; inside at
    least one, it is the larger of the two and the normal is that one's. ``angle`` rotates the
    plane figure about the extrusion axis only, so the cross-section does not change along the
    axis. That is exactly the property ``check_absorbing_layers`` reads, and it is why a tilted
    extrusion axis is stage 5 and refused by name.
    """

    #: Every extrusion in this file owns its fill in closed form: a clipped cross-section times
    #: a 1-D overlap. A CSG node reads it to decide whether the child can be proved flat.
    has_exact_fill = True

    def __init__(self, height, axis, center, angle):
        if axis not in AXES:
            raise ValueError(f"axis must be one of 'x', 'y', 'z'; got {axis!r}.")
        self.height, = check_sizes(type(self).__name__, height=height)
        center, angle = check_reals(type(self).__name__, center=center, angle=angle)
        self.axis, self.center, self.angle = axis, center, angle

    def leaves(self):
        return (self.height, self.center, self.angle)

    def _transverse_surface(self, q):
        """``(d, g)`` of the plane figure at ``q``, shape ``(..., 2)``; ``g`` a unit 2-vector.

        ``q`` is in **absolute** transverse coordinates, the same frame the clipped-area forms
        work in, so that one description of the figure serves both and a centre cannot be
        subtracted twice (it was, once, and only a direct test of the distance itself saw it:
        every volume and every AD check passed, because they were consistent with each other).
        """
        raise NotImplementedError

    def _split(self, points):
        xp = array_namespace(points)
        u, v = _transverse(self.axis)
        k = AXES[self.axis]
        centre = _as(xp, self.center, points)
        q = xp.stack([points[..., u], points[..., v]], axis=-1)
        s = points[..., k] - centre[..., k]
        return xp, (u, v, k), q, s

    def surface(self, points):
        xp, (u, v, k), q, s = self._split(points)
        dr, g = self._transverse_surface(q)
        half = _as(xp, self.height, points) / 2
        da = xp.abs(s) - half
        one = xp.ones_like(s)
        sign = xp.where(s >= 0, one, -one)
        out = (dr > 0) & (da > 0)
        hyp = xp.sqrt(xp.where(out, dr * dr + da * da, xp.ones_like(dr)))
        d = xp.where(out, hyp, xp.maximum(dr, da))
        radial = dr > da
        safe = xp.where(out, hyp, one)
        nu = xp.where(out, dr * g[..., 0] / safe, xp.where(radial, g[..., 0], xp.zeros_like(dr)))
        nv = xp.where(out, dr * g[..., 1] / safe, xp.where(radial, g[..., 1], xp.zeros_like(dr)))
        na = xp.where(out, da * sign / safe, xp.where(radial, xp.zeros_like(dr), sign))
        return d, _stack3(xp, (nu, nv, na), (u, v, k), s)

    def _boxes(self, points, dx):
        """The voxel's transverse rectangle and axial interval, and the shape's axial span."""
        xp, (u, v, k), _, _ = self._split(points)
        centre = _as(xp, self.center, points)
        half = _as(xp, self.height, points) / 2
        return (xp, (u, v, k),
                (points[..., u] - dx / 2, points[..., u] + dx / 2,
                 points[..., v] - dx / 2, points[..., v] + dx / 2),
                (centre[..., k] - half, centre[..., k] + half, points[..., k]))

    def _area(self, box, points):
        raise NotImplementedError

    def exact_fill(self, points, dx):
        xp, _, box, (lo, hi, s) = self._boxes(points, dx)
        return self._area(box, points) * overlap(lo, hi, s, dx, xp) / dx ** 3


class _Polygonal(_Extrusion):
    """An extrusion whose cross-section is a simple counter-clockwise polygon.

    ``self._convex`` is a **static** promise about that polygon, read once at the constructor
    from concrete vertices and never from a traced value -- it sizes an array, so it could not be
    otherwise. :func:`_exact.clip_capacity` says what it buys and what the safe answer is when
    there is no value to read.
    """

    #: Set by every concrete subclass' constructor; False is always safe (a bigger buffer).
    _convex = False

    #: Exactly ``1.0``, or NaN where an outline that could not be read is clockwise. A
    #: rectangle and a checked outline are both ``1.0``, so this multiply changes no number
    #: anywhere and costs one broadcast scalar op per voxel; see
    #: :func:`_loud_if_not_counter_clockwise` for the one case it is not 1.0 in.
    _turn = 1.0

    def _vertices(self, points):
        """The polygon in the grid's transverse frame, absolute, ``(nv, 2)``."""
        raise NotImplementedError

    def _transverse_surface(self, q):
        xp = array_namespace(q)
        vertices = self._vertices(q)
        nv = vertices.shape[-2]
        zero = xp.zeros_like(q[..., 0])
        one_ = xp.ones_like(zero)
        best = zero + float('inf')
        near = xp.stack([zero, zero], axis=-1)
        # The nearest EDGE's own outward normal, carried beside the offset vector, because on the
        # outline itself the offset is the zero vector and has no direction while the edge always
        # has one. The first writing returned a fixed ``+u`` there, and that was observable after
        # all: two boxes meeting on a plane that lands exactly on a voxel centre both read
        # ``+u``, so the pair looked like two copies of one half space instead of two faces back
        # to back, the seam rule did not fire and the joint kept its half-a-voxel error (measured:
        # 56 voxels, worst |dfill| 0.5000, at a seam on a voxel centre and nowhere else).
        edge_n = xp.stack([one_, zero], axis=-1)
        longest = zero
        inside = xp.zeros_like(zero, dtype=xp.bool)
        for i in range(nv):
            a, b = vertices[..., i, :], vertices[..., (i + 1) % nv, :]
            e = b - a
            w = q - a
            ee = xp.sum(e * e, axis=-1)
            live = ee > 0
            t = xp.clip(xp.sum(w * e, axis=-1) / xp.where(live, ee, xp.ones_like(ee)),
                        zero, xp.ones_like(zero))
            t = xp.where(live, t, zero)
            off = w - e * t[..., None]
            d2 = xp.sum(off * off, axis=-1)
            take = d2 < best
            best = xp.where(take, d2, best)
            near = xp.where(take[..., None], off, near)
            # outward for a counter-clockwise outline, which :func:`polygon_shape` guarantees
            length = xp.sqrt(xp.where(live, ee, one_))
            longest = xp.maximum(longest, xp.where(live, length, zero) + zero)
            out_n = xp.stack([e[..., 1] / length, -e[..., 0] / length], axis=-1)
            edge_n = xp.where((take & live)[..., None], out_n + zero[..., None], edge_n)
            # Even-odd crossing count, the same test the RCWA path's membership uses: a ray
            # along +u from q crosses this edge iff the edge straddles q's v and the crossing
            # is to the right. A horizontal edge straddles nothing and is skipped by the guard.
            dv = b[..., 1] - a[..., 1]
            straddle = (a[..., 1] > q[..., 1]) != (b[..., 1] > q[..., 1])
            cross = a[..., 0] + (e[..., 0] / xp.where(straddle, dv, xp.ones_like(dv))) * (q[..., 1] - a[..., 1])
            inside = inside ^ (straddle & (q[..., 0] < cross))
        one = xp.ones_like(zero)
        sign = xp.where(inside, -one, one)
        on_edge = _ON_THE_OUTLINE * longest
        live = best > on_edge * on_edge
        norm = xp.where(live, xp.sqrt(xp.where(live, best, one)), one)
        # On the outline itself the offset vector is zero and carries no direction, so the
        # nearest edge's own outward normal is what is returned -- which is the true outward
        # normal there, not a placeholder.
        g = xp.where(live[..., None], sign[..., None] * near / norm[..., None], edge_n)
        return sign * xp.sqrt(best), g

    def _clipped(self, box, points):
        """The clipped cross-section **without** the orientation factor."""
        xp = array_namespace(points)
        return clipped_polygon_area(self._vertices(points), *box, xp, convex=self._convex)

    def _area(self, box, points):
        return self._turn * self._clipped(box, points)

    def exact_moment(self, points, dx):
        xp, (u, v, k), box, (lo, hi, s) = self._boxes(points, dx)
        length, muu, mvv = polygon_edge_moments(self._vertices(points), *box, xp)
        along = overlap(lo, hi, s, dx, xp)
        # A cap contributes the whole clipped cross-section, or nothing: the face is a plane at
        # a fixed axial coordinate, so its area inside the voxel jumps as it leaves. That step
        # is the geometry's and not an artefact -- the surface really does leave the voxel --
        # and it is the reason C2 has a choice to make here; see the module docstring.
        #
        # The interval is HALF OPEN, ``[lo, hi)``, so a cap landing exactly on a voxel face
        # belongs to the voxel it enters and to that one only. Both of the other spellings are
        # wrong on that measure-zero set and neither was held: with two open ends (the first
        # writing) a hexagonal prism whose caps sit exactly on a face reported a total cap area
        # of **0.000000** where the truth is 0.020000 -- the interface vanished from the grid --
        # and with two closed ends it is counted twice, 0.040000. Measured in
        # ``tests/test_fd_shapes.py::test_the_exact_face_moments_are_the_surfaces_own``.
        half = dx / 2
        caps = xp.astype((lo >= s - half) & (lo < s + half), points.dtype) \
            + xp.astype((hi >= s - half) & (hi < s + half), points.dtype)
        #
        # The orientation factor multiplies the finished pair ONCE, and not each term. Applied
        # per term it is redundant -- the cap already carries it through ``_area`` and 0 * NaN is
        # NaN, so every spelling poisons the whole area and a mutant that drops it from the edge
        # moments changes no number. Applied once, dropping it is visible: the area comes back
        # finite, ``w`` comes back live, and a clockwise outline reports an interface on a body
        # that is not there (``tests/test_fd_shapes.py::
        # test_a_door_holds_under_every_transform_that_carries_a_value``).
        cap_area = caps * self._clipped(box, points)
        area = self._turn * (length * along + cap_area)
        moment = _stack3(xp, (muu * along, mvv * along, cap_area), (u, v, k), area)
        return area, self._turn * moment


class Prism(_Polygonal):
    """A polygon extruded along ``axis``: ``vertices`` are ``(u, v)`` offsets from ``center``.

    Counter-clockwise, simple, absolute lengths -- not fractions of anything (memory: analytic
    shape coordinates are absolute). ``(u, v)`` is the right-handed pair across ``axis``
    (:func:`_transverse`), so "counter-clockwise" means one thing on all three axes. All three
    of those words are refused by name when they are not true (:func:`polygon_shape`).

    **Non-convex is legal**, and gives the same number on all three backends; it costs the
    larger clip buffer (:func:`_exact.clip_capacity`) and nothing else.
    """

    def __init__(self, vertices, height, axis='z', center=(0., 0., 0.), angle=0.):
        super().__init__(height, axis, center, angle)
        _, self._convex = polygon_shape('Prism', vertices)
        self.convex = self._convex          # a non-convex outline extrudes to a non-convex body
        self.vertices = vertices
        if readable(vertices) is None:
            self._turn = _loud_if_not_counter_clockwise(vertices)

    def leaves(self):
        return super().leaves() + (self.vertices,)

    def _vertices(self, points):
        xp = array_namespace(points)
        u, v = _transverse(self.axis)
        local = _as(xp, self.vertices, points)
        angle = _as(xp, self.angle, points)
        co, si = xp.cos(angle), xp.sin(angle)
        centre = _as(xp, self.center, points)
        return xp.stack([local[..., 0] * co - local[..., 1] * si + centre[..., u],
                         local[..., 0] * si + local[..., 1] * co + centre[..., v]], axis=-1)


class Box(_Polygonal):
    """A rectangular body: ``size`` across all three axes, rotatable about ``axis`` by ``angle``.

    It is the four-vertex :class:`Prism` and shares every line of its machinery; it exists under
    its own name because ``size=(a, b, c)`` is the spelling a user writes for a rectangle and
    a vertex list is not. The public ``fd.Box`` (``wavebit/fd/geometry.py``) is this body; the
    cell-snapped box a monitor or a ``TFSF`` takes is ``fd.Region`` since stage 4's C4.
    """

    #: A rectangle, built here from ``size``, so convexity is a property of the constructor.
    _convex = convex = True

    def __init__(self, size, center=(0., 0., 0.), angle=0., axis='z'):
        if axis in AXES:
            size, = check_sizes('Box', size=size)
        super().__init__(size[AXES[axis]] if axis in AXES else None, axis, center, angle)
        self.size, self._uv = size, _transverse(axis)

    def leaves(self):
        return super().leaves() + (self.size,)

    def _vertices(self, points):
        xp = array_namespace(points)
        u, v = self._uv
        size = _as(xp, self.size, points)
        a, b = size[..., u] / 2, size[..., v] / 2
        zero = xp.zeros_like(a)
        local = xp.stack([xp.stack([-a, -b], axis=-1), xp.stack([a, -b], axis=-1),
                          xp.stack([a, b], axis=-1), xp.stack([-a, b], axis=-1)], axis=-2)
        angle = _as(xp, self.angle, points) + zero
        co, si = xp.cos(angle), xp.sin(angle)
        centre = _as(xp, self.center, points)
        return xp.stack([local[..., 0] * co - local[..., 1] * si + centre[..., u],
                         local[..., 0] * si + local[..., 1] * co + centre[..., v]], axis=-1)


class Cylinder(_Extrusion):
    """A disc extruded along ``axis``.

    ``angle`` is accepted here because the kernel's extrusions share one constructor, and it
    is a rotation of a circle: measured here, the gradient of a loss over 4096 voxels with
    respect to it is **exactly 0.0**
    (``tests/test_fd_shapes.py::test_a_circular_cylinders_angle_carries_no_gradient``) -- a
    key whose gradient cannot be anything but zero, the ``circle``/``ly`` lesson at
    ``wavebit/_core/parametric.py:348-355``. So the public ``fd.Cylinder`` does not take it
    (C4), and an elliptical cross-section, which would give it something to turn, is not
    built.
    """

    #: A disc extruded along an axis is convex.
    convex = True

    def __init__(self, radius, height, axis='z', center=(0., 0., 0.), angle=0.):
        super().__init__(height, axis, center, angle)
        self.radius, = check_sizes('Cylinder', radius=radius)

    def leaves(self):
        return super().leaves() + (self.radius,)

    def _transverse_surface(self, q):
        xp = array_namespace(q)
        u, v = _transverse(self.axis)
        centre = _as(xp, self.center, q)
        q = q - xp.stack([centre[..., u], centre[..., v]], axis=-1)
        r2 = xp.sum(q * q, axis=-1)
        live = r2 > 0
        r = xp.where(live, xp.sqrt(xp.where(live, r2, xp.ones_like(r2))), xp.zeros_like(r2))
        safe = xp.where(live, r, xp.ones_like(r))[..., None]
        zero = xp.zeros_like(r)
        fallback = xp.stack([xp.ones_like(r), zero], axis=-1)
        return r - _as(xp, self.radius, q), xp.where(live[..., None], q / safe, fallback)

    def _area(self, box, points):
        xp = array_namespace(points)
        u, v = _transverse(self.axis)
        centre = _as(xp, self.center, points)
        return clipped_disc_area(centre[..., u], centre[..., v], _as(xp, self.radius, points),
                                 *box, xp)


class _Combination(Body):
    """A CSG node. The rule is on the signed distance, resolved once, before any fill is taken.

    ``Union = min``, ``Intersection = max``, ``Difference(a, b, ...) = max(d_a, -d_b, ...)``.
    Painting order *is* this: structure ``k``'s effective body is the difference of it and
    everything after it, which on cell-snapped geometry reproduces the last-wins rule
    ``coefficients.painters`` writes, exactly, and on smoothed geometry makes the overlap
    exact and differentiable, with no second algebra competing with the first.

    The normal is the **selected** child's, so it is a unit vector by induction. No exact fill and
    no exact moment: an exact clip of a CSG node does not exist, which is the reason the tangent
    plane is the kernel's default and not its fallback.

    **This is the one class whose ``w`` rests entirely on this normal**, and the C1 review found
    that nothing held it: a mutant returning ``self.bodies[0]``'s normal from every node passed
    the whole gate while moving ``w`` by 0.88 in 616 voxels. It is now held by
    ``tests/test_fd_shapes.py::test_the_signed_distance_is_the_distance``, which reads each
    node's distance a second way (``min`` / ``max`` / ``max`` of the negation over the children's
    own closed forms) and its normal against a central difference of that distance. The
    *direction* of a normal is read nowhere else in this package -- ``_cut.cut_parts`` consumes
    ``n`` only as ``|n| dx`` and ``n n^T`` -- so that test is also the only thing holding
    :class:`Difference`'s reversal, and without it ``-other_n`` is a line with nothing but a
    docstring behind it.

    **THE COMPOSED DISTANCE IS WRONG BY HALF A VOXEL WHEREVER TWO CHILDREN TOUCH.** ``min`` of
    two signed distances is the union's distance *outside* it and not inside: where two children's
    surfaces are flush -- a rod standing on a substrate of the same material, a layer stack, any
    joint a device is built from -- both ``d`` are zero in the middle of solid material, so the
    composed plane cuts the voxel in half. Measured before the rule below (``h = 0.02``, the joint
    on a voxel centre, 40^3): 52 voxels well inside the rod's footprint read ``fill = 0.5`` where
    the truth is 1.0, with ``sum_c w_c = 1`` and ``w = (0, 0, 1)`` -- a full z-facing interface
    through solid metal. It is **first** order in ``h`` in a scheme sold at order two (worst
    ``|dfill|`` stays 0.5000 at ``h`` = 0.1, 0.05, 0.025) and both free invariants are satisfied
    by it. ``fd.Union`` is the design's own instruction to a user whose bodies share a material,
    so this sat on the recommended path. The composed plane is not used for a node's fill any
    more; ``surface`` still composes the distance, because that is what a node *is*.

    **WHAT SHIPS: AN INTERVAL, AND THE END THAT IS SAFE FOR THIS OPERATION.** Per voxel each
    child is put in exactly one of three boxes, by two predicates that are theorems and not
    tolerances:

        wholly inside    ``d_i <= -dx sqrt(3)/2``     the voxel is inside the ball that ``d_i``
                                                     proves empty of surface, so the child fills it
        wholly outside   ``d_i >= (dx/2) |n_i|_1``    the voxel misses the child's own tangent
                                                     half space, and a convex body is inside it
        crossing         neither                     the child's surface may reach this voxel

    ``|n_i|_1 dx / 2`` is the box's exact support along ``n_i``, so *outside* is exact for the
    half space and therefore for a **convex** body; for a child that is not convex -- the only
    one this package admits is a non-convex :class:`Prism` -- the same test is spelled with the
    circumradius ``dx sqrt(3)/2``, which needs nothing but ``|d_i|`` being at most the true
    distance and is what the *inside* test already uses.

    Then the node builds, per voxel, **an interval** ``[low, high]`` that brackets the true filled
    fraction, and ships the end that is safe for its own operation. The whole rule is four lines
    of set arithmetic over one number per child and two per pair:

        F_i                     the child's own **exact** filled fraction -- 1 where it is wholly
                                inside, 0 where it is wholly outside, its own closed form where
                                it crosses. A child with no exact fill is refused at construction
                                (below), so this number is never itself a bound.
        [lo_ij, hi_ij]          a bracket on ``|A_i n A_j n V| / dx^3``, the volume one pair of
                                children shares. Three cases, in this order: **0** where the pair
                                is proved to share no volume at all (:func:`_no_common_volume`,
                                which is exact); the **exact two-plane cut** where this is the one
                                pair the node spends it on; and otherwise the two sides any two
                                sets in one voxel obey, ``max(0, F_i + F_j - 1)`` and
                                ``min(F_i, F_j)``.

    and the arithmetic is in each subclass' ``_resolved``. The sign argument, per operation:

        Union            ``|u A_i| <= sum F_i - |A_p n A_q|`` for **any** single pair, so
                         subtracting a LOWER bound on one pair's shared volume can only leave the
                         answer at or above the truth. ``high = sum F_i - max_pairs lo``; it
                         **cannot remove material that is there**, and that is the end Union
                         ships. The other end is Bonferroni's, ``sum F_i - sum_pairs hi``.
        Intersection     ``|n A_i| >= |A_p n A_q| + sum_{i != p,q} F_i - (k - 2)``, so a LOWER
                         bound on the one pair gives a lower bound on the whole intersection.
                         ``low`` is that; it **cannot add material that is not there**, and that
                         is the end Intersection ships. The other end is ``min(min_i F_i,
                         min_pairs hi)``. This is the one place the previous writing had the sign
                         backwards -- it shipped the upper bound, which invents material.
        Difference       ``|A_0 \\ u_{j>0} A_j| = F_0 - |A_0 n u_{j>0} A_j|``, so what is
                         subtracted must be a LOWER bound on that union of intersections:
                         ``max(max_j lo_0j, sum_j lo_0j - sum_{0<j<m} hi_jm)``, the second term
                         Bonferroni's again, one dimension in. ``high = F_0 - that``; it
                         **cannot remove material the head has**, and that is the end Difference
                         ships. The C2 review measured this one removing **0.0206** of a voxel
                         over 72 voxels of a slab with two overlapping vias, one-signed, because
                         the shared volume of the two REMOVED children entered ``1 - F_1 - F_2 +
                         shared`` with the lower bound where the sign calls for the upper one.

    **There is no per-voxel refusal, and there was one.** The previous writing raised where two
    different pairs of crossing children both held volume of one voxel in common. The C2 review
    measured what that cost: a T-junction of three flush boxes whose union **is** one box -- three
    pairwise *disjoint* bodies, a case the rule answers exactly -- was refused at 16 of 16 random
    sub-cell grid origins and answered at the 1 the gate happened to use, and the same 21 voxels
    under ``jax.jit``, where there is no value to read and the refusal could not fire, fell back
    to the composed plane and read ``fill = 0.75`` against a true 1.0 with a full ``w = (1, 0, 0)``
    in solid material. A refusal that depends on where the grid lands, and that turns into a wrong
    answer under a transform, is not a door. What replaces it: every voxel gets the interval, and
    the node **reports** how many voxels took a bound and how wide the widest was
    (:class:`Approximations`), so a caller warns with a number instead of a body failing to solve.

    **The refusals that are left are at plan time, once, from the shapes.** A node's child must
    answer its own filled fraction **exactly** (:attr:`Body.has_exact_fill`). Two things fail
    that, and both fail it for one reason: a node's own answer is one end of an interval, and
    *which* end is safe depends on the operation above it -- a Union needs its children
    over-stated and an Intersection needs them under-stated, and one number cannot be both. So a
    node may not be a child of another node, and the constructor says so and names the flat
    spelling instead. Nesting used to be the remedy this class recommended for three surfaces in
    one voxel; that case is answered now, so the recommendation is gone and the refusal is free.
    Measured on the nesting that is now unwritable, against an exact oracle (a ball clipped by two
    axis-aligned boxes, so every region is a ``clipped_ball_volume``):
    ``Union(Box, Intersection(Sphere, Box))`` over five sub-cell grid origins read the union up to
    **0.1156** of a voxel above the truth, with the nested node itself up to 0.0152 above its own
    exact fill; and in the other direction, three nested families read up to **0.0265** of a voxel
    *removed* against a membership oracle -- a union falling below the truth, which is the one
    thing a union's whole sign argument says it cannot do. Both are errors with no bound on them
    and no test naming them, and the C2 review's own (unquoted) figure for this family was 0.2802.

    **What is exact, and what is a bound.** In the exact rows the interval closes to **exactly**
    0.0 -- both ends are the same expression over the same operands -- so the report's count is 0
    and not merely small:

        the voxel                                     exact?
        no child's surface crosses it                 exactly -- a boolean of the children's fills
        one child's surface crosses it                exactly, because that child's fill is exact
        two or more cross, every pair proved to       exactly -- disjoint children ADD, and this is
        share no volume                               the flush device joint and the T-junction
        exactly one pair shares it, both of that      exactly -- ``_cut.two_plane`` is the volume
        pair proved FLAT                              of a voxel cut by two half spaces
        anything else                                 a bound, counted and reported

    **The exact rows, measured** (2026-09-21, ``h`` = 0.05, NumPy/JAX/Torch):

        a three-box T-junction against the single box it is,     3.8e-15   over 36 grid placements
        the same rotated 90 degrees (seams on x and y)           2.4e-15   over 18
        the same under ``jax.jit``, against eager and truth      3.8e-15   (0.2500 before)
        two boxes whose union is one box, 3 seams x 7 origins    2.4e-15
        worst ``max_i F_i - fill`` (a hole in solid material)    1.3e-14   5 scenes x 3 joints x
        worst ``fill - min(1, sum F_i)``                         0.0e+00   6 grid origins
        flush children add exactly, worst departure              1.4e-15

    **The safe side, measured**, against 12^3 = 1728 stratified membership samples per voxel
    (standard error 1.2e-2; the oracle is the composed distance's SIGN and shares no line with the
    rule), 14^3 voxels at ``h`` = 0.05, three grid origins, 20 scenes. The worst violation of the
    safe side over **every** scene is **+0.0103**, i.e. inside the sampling error and consistent
    with the zero the sign argument proves. What being on that side costs -- the other tail -- is
    what is worth reading:

        operation      worst cost   on
        Union            0.3543     a non-convex star prism with a ball through it
                         0.2119     three pillars in a row, each sunk into a substrate
                         0.0735     one hexagonal pillar sunk into a substrate
        Intersection     0.3370     a ball clipped by two slabs
                         0.2601     two tilted boxes
                         0.1875     the corner of a cube spelled as three slabs
        Difference       0.1517     a slab with two overlapping vias
                         0.1406     an L-shaped groove (two overlapping boxes removed)

    and the widest interval the node reported over the same product is 0.5000, on that L-shaped
    groove. **The width is a bound on the interval and not an estimate of the error**: on the
    T-junction it reads 0.2500 where the shipped end is exact to 4e-15, because three bodies that
    are pairwise disjoint have tangent half spaces that are not, and the node can see only the
    half spaces. It over-warns and never under-warns, which is the direction a report should err
    in; a caller that wants the error itself has to refine the grid and watch the answer move.

    The one number this replaces is the C2 review's: the previous writing **removed** 0.0206 of a
    voxel over 72 voxels of a slab with two overlapping vias, one-signed, against a 128^3
    quadrature. The same probe now reads 0.0000 on both tails.

    **What "FLAT" is, and why it is a theorem and not a fudge.** A child is called flat in a voxel
    when its own exact fill and its tangent half space's agree to :data:`_FLAT` of the voxel. A
    convex body lies inside its own tangent half space, so ``f_i = F_i`` says the two sets differ
    by a volume of at most ``1e-12 dx^3`` -- they are the **same set** in this voxel, and every
    statement :func:`_cut.two_plane` makes about the half spaces is then a statement about the
    bodies. Where flatness holds for both children of the pair the two ends of the two-plane
    bracket coincide; where it does not, they differ by ``gap_i = f_i - F_i``, each child's own
    tangent-plane over-fill, and the containment ``H_a n H_b \\ (A n B) subset (H_a \\ A) u
    (H_b \\ B)`` is what makes that the right width. Flatness also gates every subtraction of
    *interface*: that is the second thing an earlier writing got wrong, deleting **both**
    children's whole face wherever a pair was back to back and flush, so the foot perimeter of a
    pillar -- where the substrate's top is genuinely exposed and the pillar's side wall is
    genuinely in the same voxel -- reported ``w = 0`` and ``sum_c w_c = 0``, no interface at all,
    in 108 of 5360 truly cut voxels of a 3x3 pillar array.

    **The interface, term by term.** Every child that crosses contributes its own area and its own
    moment; a pair subtracts only what it can prove is buried, and only where both children are
    flat:

        a pair back to back and on each other (``n_i . n_j = -1``, ``d_i + d_j <= 0``) buries both
        faces entirely -- that is the flush device joint, and it is why the rod on the substrate
        reads ``w = 0`` exactly;
        the one pair the two-plane cut is spent on buries ``area(dH_i n B n H_j)``, which
        :func:`_cut.two_plane` returns beside the volume, and **only for the two children of that
        pair** -- a third crossing child keeps its whole face, which an earlier writing did not
        do and which would have reported no interface where there is one.

    A :class:`Difference` does not fuse a flush pair between the body it keeps and a body it
    removes -- there the far side of the face is vacuum, and the face is still interface. Where
    nothing can be proved buried the areas are simply summed, which over-counts a joint and never
    reports no interface where there is one; ``sum_c w_c`` is 1 in every such voxel.

    **A CSG node's fill is continuous; ``w`` is not, and that is the whole of the reason its
    geometry may not be traced.** Three earlier writings of this class refused a traced parameter
    because the *fill* stepped, by 3.15e-02, then 4.29e-02, then 4.48e-02, across a geometry
    change of 5.5e-17. That step was the composed plane's: its fill reads the **selected** child's
    normal, and at a crease the selection swaps. A node answers from its children's own fills and
    one exact two-plane cut, and neither reads a selected normal. What is left is ``w``: at the
    worst live-to-live step of a ``box minus box`` slide, bisected to 2.8e-17, the fill is
    identical to ten digits and

        w  (0.0200, 0.4877, 0.4923)  ->  (0.0395, 0.9605, 0.0000)      |dw| = 4.91e-01

    with a central difference of ``w_y`` reading 233, 2361, 23639, 236424, 2364267 at steps 1e-3
    down to 1e-7: ``1/step``, the signature of a jump in the objective. ``occupancy`` returns the
    pair and C2's mixing rule reads both halves, so a loss through a node still steps. The refusal
    is on the *leaf* and not on tracing, so a forward solve that traces the points keeps working,
    and every body with an exact fill stays differentiable -- which is where a design variable
    belongs.

    **What it costs.** :func:`_cut.two_plane` runs **once per voxel**, not once per pair: the node
    spends it on the first pair of crossing children that is not proved disjoint, and every other
    pair takes the two-line bracket. What is O(k^2) is the pair test
    (:func:`_no_common_volume`, five evaluations of a piecewise-linear function) and the brackets.
    Measured on this machine (2026-09-21, ``OMP_NUM_THREADS=2``, float64, 66^3 = 287496 voxels at
    ``dx`` = 0.0152, the node's own support box, one component, best of three), on a substrate
    plus ``k - 1`` flush pillars:

        k   pairs  backend   NODE     children alone  composed plane  pair test  two_plane
        2     1    numpy      4.114 s     2.996 s        0.243 s       0.070 s    0.672 s
        2     1    jax        2.443 s     1.535 s        0.149 s       0.065 s    0.552 s
        2     1    torch      3.177 s     2.526 s        0.146 s       0.015 s    0.326 s
        4     6    numpy      5.418 s     3.671 s        0.291 s       0.391 s    0.614 s
        4     6    jax        3.150 s     1.847 s        0.179 s       0.371 s    0.547 s
        4     6    torch      3.384 s     2.552 s        0.169 s       0.079 s    0.291 s
       10    45    numpy     11.149 s     5.432 s        0.456 s       3.102 s    0.613 s
       10    45    jax        7.762 s     2.944 s        0.292 s       2.860 s    0.514 s
       10    45    torch      4.974 s     2.913 s        0.233 s       0.531 s    0.267 s

    Read it as a maintainer: on a 3x3 pillar array (10 children, 45 pairs) the **node** costs
    11.1 s on NumPy, of which 5.4 s is its children's own exact fills, 3.1 s is the O(k^2) pair
    test and 0.61 s is the one exact two-plane cut -- flat in ``k``, as it must be. The "composed
    plane" column is the path this rule replaced and is **not** computed any more (the node has
    no fallback to fall back to), so 0.456 s is what the node would cost if the answer were that
    plane, and 11.1 s is 24x it. Over six sub-cell grid origins the node varies between 10.94 s
    and 11.02 s, i.e. the number is a property of the scene and not of an alignment. The
    alternative -- Lasserre's recursion over all ``k`` planes (J. B. Lasserre, *An analytical
    expression and an algorithm for the volume of a convex polyhedron in R^n*, J. Optim. Theory
    Appl. **39**, 363 (1983)) -- was 440.4 s and 22.2 GiB on this scene, so it is still 40x this
    node on NumPy and out of reach on memory. :func:`_cut.two_plane` is that identity's ``k = 2``
    case with the recursion unrolled into arithmetic, and carries its derivation and its
    validation against an independent ConvexHull oracle. The pair test is the honest place to look
    for the next factor: 28% of the node at ``k`` = 10, and growing as ``k^2``.
    """

    #: Which end of its own interval this node kind ships: the high one where the operation must
    #: not remove material it has (``Union``, ``Difference``), the low one where it must not add
    #: material it has not (``Intersection``). See the sign argument in the class docstring.
    _ships_the_high_end = True

    def __init__(self, *bodies):
        if len(bodies) < 2:
            raise ValueError('a CSG node combines at least two bodies.')
        for body in bodies:
            if not isinstance(body, Body):
                raise ValueError(
                    f'{type(self).__name__}(...): every child is a Body -- Sphere, Cylinder, Box '
                    f'or Prism; got {type(body).__name__} ({body!r}). A CSG node asks each child '
                    'for its signed distance and its own cut, and a value that is not a body has '
                    'neither.')
            if not body.has_exact_fill:
                raise ValueError(
                    f'{type(self).__name__}(...): every child must answer its own filled fraction '
                    f'EXACTLY, and {type(body).__name__} does not. A node\'s own answer is one end '
                    'of an interval, and which end is safe depends on the operation above it: a '
                    'Union needs its children over-stated (so that it cannot remove material that '
                    'is there) and an Intersection needs them under-stated (so that it cannot add '
                    'material that is not), and one number cannot be both. So a node may not be a '
                    'child of another node. Write it FLAT -- Union(a, b, c), not '
                    'Union(Union(a, b), c) -- which this rule answers for any number of children, '
                    'including three or more surfaces meeting in one voxel. (Measured on the '
                    'nesting this refuses, against an exact oracle: Union(Box, '
                    'Intersection(Sphere, Box)) read up to 0.1156 of a voxel above the truth over '
                    'five sub-cell grid origins, with the nested node itself up to 0.0152 above '
                    'its own exact fill.) Sphere, Cylinder, Box and Prism all have an exact fill '
                    'and are all legal children.')
            for leaf in body.leaves():
                if traced(leaf):
                    raise ValueError(
                        f'{type(self).__name__}(...): a CSG node\'s geometry must be concrete, '
                        'and this child was built from a traced value. A CSG node\'s INTERFACE '
                        'is discontinuous where two of its surfaces meet: measured, w steps '
                        '4.91e-01 across a geometry change of 2.8e-17 in a voxel that is cut on '
                        'both sides, and a central difference of a loss on it DIVERGES as '
                        '1/step (233, 2361, 23639, 236424, 2364267 at 1e-3 down to 1e-7), which '
                        'is a jump in the objective and not a gradient disagreeing with a '
                        'difference. (The FILL is continuous -- worst step 3.1e-03 over six '
                        'node kinds, the same as a leaf\'s -- but occupancy returns the pair and '
                        'the mixing rule reads both halves.) A forward-only CSG node is honest '
                        'and a gradient through a jump is not, so the parameter is refused '
                        'rather than differentiated. Build the node outside the transform and '
                        'trace the POINTS, which is what a forward solve does; to differentiate '
                        'a dimension, use a body with an exact fill (Sphere, Cylinder, Box, '
                        'Prism), whose w is smooth away from its own creases.')
        self.bodies = bodies

    def leaves(self):
        return tuple(leaf for body in self.bodies for leaf in body.leaves())

    def _compose(self, d, n, other_d, other_n, xp):
        raise NotImplementedError

    def surface(self, points):
        xp = array_namespace(points)
        d, n = self.bodies[0].surface(points)
        for body in self.bodies[1:]:
            d, n = self._compose(d, n, *body.surface(points), xp)
        return d, n

    def _pairs(self):
        """Every pair of children. The pair test is what makes a node O(k^2); see the docstring."""
        k = len(self.bodies)
        return [(i, j) for i in range(k) for j in range(i + 1, k)]

    def _name(self):
        """``'Union(Box, Cylinder)'`` -- what a report names when a scene has several nodes."""
        return f'{type(self).__name__}({", ".join(type(b).__name__ for b in self.bodies)})'

    def _fused(self, i, j):
        """Does a flush back-to-back pair of these two children bury both faces?

        True wherever the material is on both sides of the joint. :class:`Difference` overrides
        it: between the body it keeps and a body it removes, the far side is vacuum.
        """
        return True

    def _state(self, faces, child, dx, xp):
        """Per child and per voxel: wholly inside, wholly outside, crossing, and provably flat."""
        out = []
        for body, (d, n), part in zip(self.bodies, faces, child):
            plane = _parts(d, n, dx)
            tangent = bool(body.convex)
            inside = d <= -_CORNER * dx
            # A convex body is inside its own tangent half space, so the box's exact support
            # along n answers `outside` exactly. Without that containment the same question is
            # answered by the circumradius, which needs only |d| <= the true distance -- the one
            # thing the `inside` test needs too, and true for every exact signed distance here.
            outside = (d >= (dx / 2) * xp.sum(xp.abs(n), axis=-1) if tangent
                       else d >= _CORNER * dx)
            # Equal volumes prove two sets equal only when one contains the other, so flatness is
            # a statement a non-convex body cannot make however close the two numbers are.
            flat = (xp.abs(plane[0] - part[0]) <= _FLAT if tangent
                    else xp.zeros_like(d) > xp.ones_like(d))
            out.append(_Child(d, n, inside, outside, ~(inside | outside), flat, tangent,
                              plane, part))
        return out

    def _pair_state(self, state, dx, xp):
        """Per pair and per voxel: proved to share no volume, flush and back to back, and which
        one pair (at most) the exact two-plane cut is spent on."""
        zero = xp.zeros_like(state[0].d)
        never = zero > xp.ones_like(zero)
        split, flush, first = {}, {}, {}
        seen = never
        for i, j in self._pairs():
            a, b = state[i], state[j]
            both = a.cross & b.cross
            # `_no_common_volume` is a question about the two tangent HALF SPACES, and only a
            # convex body is inside its own. For a pair this file cannot ask it of, nothing is
            # claimed -- which costs the pair its exact zero and never claims a false one.
            apart = (_no_common_volume(a.d, a.n, b.d, b.n, dx, xp)
                     if (a.tangent and b.tangent) else never)
            split[i, j] = both & apart
            on_each_other = ((1 + xp.sum(a.n * b.n, axis=-1) <= _SAME_FACE)
                             & (a.d + b.d <= _SAME_FACE * dx))
            flush[i, j] = (split[i, j] & on_each_other & a.flat & b.flat
                           if self._fused(i, j) else never)
            live = (both & ~apart) if (a.tangent and b.tangent) else never
            first[i, j] = live & ~seen
            seen = seen | live
        return split, flush, first

    def _crease(self, state, first, dx, xp):
        """The one pair of surfaces the exact two-plane cut is spent on, per voxel.

        :func:`_cut.two_plane` answers exactly for the two tangent **half spaces**, and a convex
        body is inside its own, so ``V`` is an upper bound on what the two bodies share and
        ``V - gap_a - gap_b`` a lower one, where ``gap_i = f_i - F_i`` is the child's own
        tangent-plane over-fill of this voxel::

            H_a n H_b  \\  (A n B)  is inside  (H_a \\ A) u (H_b \\ B)

        Both are then clamped into ``[max(0, F_a + F_b - 1), min(F_a, F_b)]``, which any two sets
        in one voxel obey. Where both children are **flat** the two gaps are zero and the two
        bounds are the same number, exactly ``V``.

        ``buried`` is the area of each child's tangent plane lying inside its partner's half
        space, ``gate`` says both children of the pair were proved flat -- which is what turns a
        statement about two half spaces into one about two bodies -- and ``chosen[i]`` says child
        ``i`` is one of the two, so that a third crossing child does not have its own face
        deleted by a pair it is not in.
        """
        zero = xp.zeros_like(state[0].d)
        one = xp.ones_like(zero)
        axis = xp.stack([one, zero, zero], axis=-1)
        # a pair that is not there is spelled as two coincident half spaces a voxel away, so the
        # cut runs on real numbers everywhere and no dead branch carries a nan into a cotangent
        d_a, d_b, n_a, n_b = dx * one, dx * one, axis, axis
        f_a, f_b, pf_a, pf_b, pa_a, pa_b, gap = (zero,) * 7
        gate = zero > one
        chosen = [zero > one] * len(state)
        for (i, j), take in first.items():
            a, b = state[i], state[j]
            wide = take[..., None]
            d_a, n_a = xp.where(take, a.d, d_a), xp.where(wide, a.n, n_a)
            d_b, n_b = xp.where(take, b.d, d_b), xp.where(wide, b.n, n_b)
            f_a, f_b = xp.where(take, a.part[0], f_a), xp.where(take, b.part[0], f_b)
            pf_a, pa_a = xp.where(take, a.plane[0], pf_a), xp.where(take, a.plane[1], pa_a)
            pf_b, pa_b = xp.where(take, b.plane[0], pf_b), xp.where(take, b.plane[1], pa_b)
            gap = xp.where(take, xp.maximum(a.plane[0] - a.part[0], zero)
                           + xp.maximum(b.plane[0] - b.part[0], zero), gap)
            gate = gate | (take & a.flat & b.flat)
            chosen[i], chosen[j] = chosen[i] | take, chosen[j] | take
        volume, cut_a, cut_b = two_plane(d_a, n_a, d_b, n_b, dx, (pf_a, pa_a), (pf_b, pa_b))
        floor = xp.maximum(f_a + f_b - one, zero)
        roof = xp.minimum(f_a, f_b)
        least = xp.clip(volume - gap, floor, roof)
        most = xp.clip(volume, floor, roof)
        buried = [zero] * len(state)
        taken = zero > one
        for (i, j), take in first.items():
            buried[i] = buried[i] + xp.where(take, cut_a, zero)
            buried[j] = buried[j] + xp.where(take, cut_b, zero)
            taken = taken | take
        return _Crease(xp.where(taken, least, zero), xp.where(taken, most, zero),
                       buried, gate, chosen)

    def _pair_bounds(self, state, split, first, crease, xp):
        """``(i, j, lo, hi)`` per pair: the bracket on ``|A_i n A_j n V| / dx^3``, one at a time.

        Three cases and no fourth: a pair proved to share no volume holds **exactly** zero; the
        one pair the node spends the exact two-plane cut on holds ``[least, most]``; every other
        pair holds the two sides any two sets in one voxel obey. Those last two are stated in the
        children's own **exact** fills, which is why a child without one is refused at
        construction -- they are what makes a child that is wholly inside the voxel (``F_i = 1``,
        so ``lo = hi = F_j``) and one that is wholly outside (``F_i = 0``, so ``lo = hi = 0``)
        come out exact with no case of their own.

        It is a **generator** and not a pair of dicts, because ``k (k - 1)`` full float arrays
        is not a thing to keep: at ``k = 10`` on a 66^3 grid that is 207 MiB of float64 that
        every ``_resolved`` reduces away the moment it sees it. Each is consumed once.
        """
        zero = xp.zeros_like(state[0].d)
        one = xp.ones_like(zero)
        for i, j in self._pairs():
            f_i, f_j = state[i].part[0], state[j].part[0]
            take = first[i, j]
            yield (i, j,
                   xp.where(split[i, j], zero,
                            xp.where(take, crease.least,
                                     xp.maximum(f_i + f_j - one, zero))),
                   xp.where(split[i, j], zero,
                            xp.where(take, crease.most, xp.minimum(f_i, f_j))))

    def _gone(self, i, state, flush, crease, xp):
        """How much of child ``i``'s interface this voxel can prove is buried in another child.

        ``crease.chosen[i]`` is **not** tested here, and that is not an oversight: ``buried[i]``
        is a sum over the pairs the cut was spent on, so it is exactly zero in a voxel where
        child ``i`` is in none of them, and ``where(gate & chosen_i, 0, 0)`` is ``where(gate, 0,
        0)``. Where it does matter is :meth:`_inside_face`, which *replaces* the child's whole
        area rather than subtracting from it -- a third crossing child would lose its face there.
        """
        zero = xp.zeros_like(state[0].d)
        gone = xp.where(crease.gate, crease.buried[i], zero)
        for pair, mask in flush.items():
            if i in pair:
                gone = gone + xp.where(mask, state[i].plane[1], zero)
        return gone

    @staticmethod
    def _kept(one, gone, xp):
        """A child's own area and moment, less the part proved buried.

        Subtracting ``gone * n n^T`` from the moment is exact and not an approximation of the
        direction: ``gone`` is non-zero only where the child was proved flat, and a flat child's
        exact moment **is** ``area * n n^T``.
        """
        nn = one.n * one.n
        return (xp.maximum(one.part[1] - gone, xp.zeros_like(gone)),
                xp.maximum(one.part[2] - gone[..., None] * nn, xp.zeros_like(one.part[2])))

    @staticmethod
    def _inside_face(one, buried, gate, xp):
        """The part of a child's interface that lies **inside** its partner -- an intersection's
        own boundary, and the wall of the hole a difference cuts.

        Where flatness could not be proved, and for a crossing child that is not one of the two
        the cut was spent on, the child's whole area is taken instead: that over-counts a joint,
        and the alternative is reporting no interface where there is one.
        """
        nn = one.n * one.n
        return (xp.where(gate, buried, one.part[1]),
                xp.where(gate[..., None], buried[..., None] * nn, one.part[2]))

    def _resolved(self, state, split, flush, first, crease, bounds, dx, xp):
        """``(low, high, area, moment)`` -- the interval and the interface, per node kind.

        ``bounds`` is :meth:`_pair_bounds`' generator and is walked **once**.
        """
        raise NotImplementedError

    def parts(self, points, dx, surface=None, report=None):
        xp = array_namespace(points)
        faces = [body.surface(points) for body in self.bodies]
        child = [body.parts(points, dx, surface=face)
                 for body, face in zip(self.bodies, faces)]
        state = self._state(faces, child, dx, xp)
        split, flush, first = self._pair_state(state, dx, xp)
        crease = self._crease(state, first, dx, xp)
        bounds = self._pair_bounds(state, split, first, crease, xp)
        low, high, area, moment = self._resolved(state, split, flush, first, crease, bounds,
                                                 dx, xp)
        if report is not None:
            width = xp.maximum(high - low, xp.zeros_like(low))
            one, zero = xp.ones_like(width), xp.zeros_like(width)
            report.record(self._name(), xp.sum(xp.where(width > _BOUNDED, one, zero)),
                          xp.max(width))
        return (high if self._ships_the_high_end else low), area, moment


class Union(_Combination):
    """Everything in any of them.

    A voxel any child fills reads 1; a voxel no child's surface crosses is a boolean; a voxel one
    child's surface crosses is that child's own answer; children proved to share no volume ADD,
    exactly; and where a pair does share the voxel, a lower bound on what it shares is subtracted,
    so the answer is provably at or above the truth. See :class:`_Combination` for the measurement
    that made each of those necessary.
    """

    def _compose(self, d, n, other_d, other_n, xp):
        take = other_d < d
        return xp.where(take, other_d, d), xp.where(take[..., None], other_n, n)

    def _resolved(self, state, split, flush, first, crease, bounds, dx, xp):
        zero = xp.zeros_like(state[0].d)
        one = xp.ones_like(zero)
        moment = xp.zeros_like(state[0].part[2])
        solid, total, biggest, area = None, zero, zero, zero
        for i, one_child in enumerate(state):
            solid = one_child.inside if solid is None else (solid | one_child.inside)
            total = total + one_child.part[0]
            biggest = xp.maximum(biggest, one_child.part[0])
            kept, kept_moment = self._kept(one_child, self._gone(i, state, flush, crease, xp), xp)
            area = area + xp.where(one_child.cross, kept, zero)
            moment = moment + xp.where(one_child.cross[..., None], kept_moment,
                                       xp.zeros_like(moment))
        # |u A_i| <= sum F_i - |A_p n A_q| for ANY single pair, so the LARGEST lower bound over
        # the pairs is the tightest thing that may be subtracted without falling below the truth;
        # and |u A_i| >= sum F_i - sum_pairs |A_i n A_j| (Bonferroni) gives the other end.
        shared_lo, shared_hi = zero, zero
        for _, _, pair_lo, pair_hi in bounds:
            shared_lo = xp.maximum(shared_lo, pair_lo)
            shared_hi = shared_hi + pair_hi
        roof = xp.minimum(one, total)                       # |u A_i| <= min(1, sum F_i)
        high = xp.minimum(roof, xp.maximum(total - shared_lo, biggest))
        low = xp.maximum(biggest, xp.minimum(total - shared_hi, roof))  # |u A_i| >= max_i F_i
        return (low, high, xp.where(solid, zero, area),
                xp.where(solid[..., None], xp.zeros_like(moment), moment))


class Intersection(_Combination):
    """Only what is in all of them.

    One child wholly outside the voxel, or one pair of crossing children holding no volume of it
    in common, empties the voxel whatever the others do. What is left is: nothing crossing (the
    voxel is inside all of them), one child crossing (its own answer), or two or more crossing,
    where the answer is the one pair's shared volume carried down by the rest through
    ``|n A_i| >= |A_p n A_q| + sum_{i != p,q} F_i - (k - 2)``.

    **It ships the LOW end of its interval**, which is the sign the previous writing had
    backwards: an intersection that over-states its fill invents material that is in none of its
    children, and at a corner where three faces meet that is the whole of the voxel's content.
    """

    def _compose(self, d, n, other_d, other_n, xp):
        take = other_d > d
        return xp.where(take, other_d, d), xp.where(take[..., None], other_n, n)

    #: An intersection must not add material that is not in every child.
    _ships_the_high_end = False

    def _resolved(self, state, split, flush, first, crease, bounds, dx, xp):
        zero = xp.zeros_like(state[0].d)
        one = xp.ones_like(zero)
        moment = xp.zeros_like(state[0].part[2])
        empty, crossing, total, smallest, area = None, zero, zero, one, zero
        for i, one_child in enumerate(state):
            empty = one_child.outside if empty is None else (empty | one_child.outside)
            crossing = crossing + xp.where(one_child.cross, one, zero)
            total = total + one_child.part[0]
            smallest = xp.minimum(smallest, one_child.part[0])
            face, face_moment = self._inside_face(one_child, crease.buried[i],
                                                  crease.gate & crease.chosen[i], xp)
            area = area + xp.where(one_child.cross, face, zero)
            moment = moment + xp.where(one_child.cross[..., None], face_moment,
                                       xp.zeros_like(moment))
        # |n A_i| >= |A_p n A_q| + sum_{i != p,q} F_i - (k - 2), for EVERY pair: everything else
        # in the node can take at most (1 - F_i) of the voxel away from the pair's shared volume.
        # The pair the exact cut was spent on is usually the tightest, and is not always -- a
        # pair of children that nearly fill the voxel beats a sharp cut between two thin ones --
        # so the max over pairs is taken, which costs three array operations per pair. With the
        # trivial ``lo`` it reduces to Bonferroni over all k children.
        floor, roof = None, smallest
        for i, j, pair_lo, pair_hi in bounds:
            one_pair = pair_lo - state[i].part[0] - state[j].part[0]
            floor = one_pair if floor is None else xp.maximum(floor, one_pair)
            roof = xp.minimum(roof, pair_hi)
        floor = floor + total - (len(state) - 2)
        for mask in split.values():
            empty = empty | mask
        low = xp.minimum(xp.maximum(floor, zero), roof)
        gone = empty | (crossing < 0.5)
        return (xp.where(empty, zero, low), xp.where(empty, zero, roof),
                xp.where(gone, zero, area),
                xp.where(gone[..., None], xp.zeros_like(moment), moment))


class Difference(_Combination):
    """The first, minus everything after it. The removed body's normal comes back reversed.

    A removed body that holds no volume of the voxel in common with the first removes nothing
    from it and contributes no interface either -- which is how a body resting *against* a cut
    face stays out of it. Where one does share the voxel, the two-plane cut says how much it
    takes and how much of each face survives: the kept body's face **outside** the removed one,
    and the removed body's face **inside** the kept one, which is the wall of the hole.

    A flush back-to-back pair is not fused across the cut. Two children of a union that meet flush
    have material on both sides and no interface between them; a removed body resting flush on the
    face of the body it is subtracted from leaves vacuum there, and the face is still interface.
    """

    def _compose(self, d, n, other_d, other_n, xp):
        take = -other_d > d
        return xp.where(take, -other_d, d), xp.where(take[..., None], -other_n, n)

    def _fused(self, i, j):
        return i != 0

    def _resolved(self, state, split, flush, first, crease, bounds, dx, xp):
        head = state[0]
        zero = xp.zeros_like(head.d)
        flat_moment = xp.zeros_like(head.part[2])
        kept, kept_moment = self._kept(head, self._gone(0, state, flush, crease, xp), xp)
        empty = head.outside
        wall, wall_moment = zero, flat_moment
        outer, outer_moment = zero, flat_moment
        # |A_0 n u_{j>0} A_j| -- what the difference takes out of the head. The two ends are
        # a lower bound (so the head keeps at least what it has) and an upper one.
        bite_lo, bite_sum, bite_hi = zero, zero, zero
        # Bonferroni one dimension in: |u_j (A_0 n A_j)| >= sum_j |A_0 n A_j| - sum_{j<m}
        # |A_j n A_m|, and the removed children's own pairwise upper bounds are what stands
        # in for those. Getting THAT term's sign wrong is what removed 0.0206 of a voxel.
        for i, j, pair_lo, pair_hi in bounds:
            if i == 0:
                bite_lo = xp.maximum(bite_lo, pair_lo)   # A_0 n A_j is in A_0 n u A_j
                bite_sum = bite_sum + pair_lo
                bite_hi = bite_hi + pair_hi
            else:
                bite_sum = bite_sum - pair_hi
        for j, one_child in enumerate(state[1:], start=1):
            empty = empty | one_child.inside
            met = first[0, j]
            face, face_moment = self._inside_face(one_child, crease.buried[j],
                                                  crease.gate & crease.chosen[j], xp)
            wall = wall + xp.where(met, face, zero)
            wall_moment = wall_moment + xp.where(met[..., None], face_moment, flat_moment)
            free, free_moment = self._kept(one_child, self._gone(j, state, flush, crease, xp), xp)
            outer = outer + xp.where(one_child.cross, free, zero)
            outer_moment = outer_moment + xp.where(one_child.cross[..., None], free_moment,
                                                   flat_moment)
        taken_hi = xp.minimum(head.part[0], bite_hi)
        taken_lo = xp.clip(xp.maximum(bite_lo, bite_sum), zero, taken_hi)
        low, high = head.part[0] - taken_hi, head.part[0] - taken_lo
        area = xp.where(head.cross, kept + wall, outer)
        moment = xp.where(head.cross[..., None], kept_moment + wall_moment, outer_moment)
        return (xp.where(empty, zero, low), xp.where(empty, zero, high),
                xp.where(empty, zero, area),
                xp.where(empty[..., None], flat_moment, moment))
