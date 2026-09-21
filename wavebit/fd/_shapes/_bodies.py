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
    Union/Intersection/ the children's own, by inclusion-exclusion, where the pair's shared
    Difference          volume is one this file can name; the *tangent plane* where it is not

A cylinder's exact moment would need the second moment of the arc of a circle clipped to a
rectangle -- the angular interval set, not the area -- and the repository owns no such form;
the arc-weighted *area* forms it does own (``_weighted_circle_area``) are a bilinear window for
a different geometry. A sphere's would need the second moment of a spherical patch, which by
the divergence theorem needs the *volume's* second moments and the clipped disc's; the patch's
plain area is free that way and the moment is not (see :class:`Sphere`). Measured cost of using
the plane's moment on a cylinder instead of the truth, against 20001-point quadrature over the
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

**The fill is smooth for every body with an exact fill, and it is not for a CSG node**, whose
worst single step over six node kinds on the same slide is **4.29e-02** in the fill and a full
**1.000** in ``w``. Two previous writings quoted 3.15e-02 and 0.566 from one node at one voxel.
See :class:`_Combination` for the mechanism, the rule that removes it where a pair's shared
volume can be named, what that rule costs where it cannot, the price of the exact fix, and the
refusal that follows from all of it.
"""
import warnings

import numpy as _np

from wavebit._core.backend import array_namespace

from ._cut import _parts
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
    the voxel wall -- contributes no volume. **No margin is carried on that boundary**, and one
    was: an earlier writing allowed ``-_SAME_FACE * dx``, because two boxes meeting on a plane
    read ``d_a + d_b`` as ``0.0`` down the middle of the joint and as ``-1e-18`` in 56 voxels off
    it. That was the fixed ``+u`` normal on the outline (:data:`_ON_THE_OUTLINE`) and not the
    arithmetic; with the nearest edge's own normal the two faces are exactly opposite and the
    pair is answered by the *slab* branch of :meth:`_Combination._pair_terms`, which does not
    read this boundary at all. A margin no gate can make fail is a margin this package deletes.
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
            if read.dtype.kind not in 'biuf':
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
    if area <= 0:
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

    def plane(self, points, dx):
        """``(fill, area, moment)`` from the tangent plane -- what every body falls back to.

        Goes to ``_cut._parts`` rather than the exported ``cut_parts``: the bodies here
        *guarantee* a unit normal, so the door on the public spelling would be a reduction and a
        host synchronisation per body per image, paid on every voxel of every solve to check a
        precondition this file establishes by construction.
        """
        d, n = self.surface(points)
        return _parts(d, n, dx)

    def parts(self, points, dx, surface=None):
        """``(fill, area, moment)``: the exact forms where they exist, the plane where they do not.

        The one call :func:`_occupancy.occupancy` makes per body per image, and the one place the
        table at the top of this module is realised. ``surface=`` is the already-computed
        ``(d, n)`` where the caller has it -- :class:`_Combination` needs its children's normals
        for the seam rule and would otherwise walk every child's distance twice.
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

    **The moment is still the plane's**, and that is not an oversight. By the divergence theorem
    the interface *area* would be free -- ``3V = R S + sum_faces d_f A_f`` with ``A_f`` a clipped
    disc -- but the second moment needs ``int (x_c - c_c)^2 dS`` over the spherical patch, which
    by the same theorem needs the volume's own second moments *and* the clipped disc's, neither
    of which this repository owns. What the plane's moment costs here is
    ``tests/test_fd_shapes.py::test_w_vanishes_only_where_the_fill_is_already_decided``'s
    measured sphere row.
    """

    #: A ball is convex.
    convex = True

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
    a vertex list is not. **Not** ``wavebit.fd.geometry.Box``, which is the cell-snapped region
    a monitor or a ``TFSF`` takes -- the two are different discretisations under one word today,
    and the plan's C4 is where that is resolved (``fd.Box`` becomes this, the snapped one
    becomes ``fd.Region``). Nothing imports both while the word means two things.
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

    ``angle`` is accepted because the stage-4 API names it, and it is a rotation of a circle:
    measured here, the gradient of a loss over 4096 voxels with respect to it is **exactly
    0.0** (``tests/test_fd_shapes.py::test_a_circular_cylinders_angle_carries_no_gradient``).
    That is the same defect the design
    refuses by name on ``Sphere(angle=)`` -- a key whose gradient cannot be anything but zero,
    the ``circle``/``ly`` lesson at ``wavebit/_core/parametric.py:348-355``. C5 should either
    refuse it on key *presence* as it does for a sphere, or the cross-section should become an
    ellipse, which would give the key something to turn.
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
    everything after it, which on cell-snapped geometry reproduces ``owner_map``'s last-wins
    exactly and on smoothed geometry makes the overlap exact and differentiable, with no
    second algebra competing with the first.

    The normal is the **selected** child's, so it is a unit vector by induction. The selection
    is C0 and not C1 at a tie -- the crease where two surfaces meet -- which is a real crease in
    the geometry and not an artefact of the spelling. No exact fill and no exact moment: an
    exact clip of a CSG node does not exist, which is the reason the tangent plane is the
    kernel's default and not its fallback.

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

    **A CSG node's FILL is discontinuous, and that is why its geometry may not be traced.**
    The plane's fill is a function of the *selected* normal, so at a crease the selection swaps
    between children and ``fill`` steps although ``d`` does not. Measured, sliding the node past
    one voxel and bisecting: ``fill`` goes 0.96063617 -> 0.99217239, a step of **3.15e-02**,
    across a slide of 5.5e-17, with ``d`` equal to twelve digits and ``w`` stepping by 0.566 at
    the same point. A central difference of a loss in that slide then reads

        step      1e-3      1e-4      1e-5      1e-6      1e-7
        plane    268.81   2685.85  26855.47 268551.57  2685512.50

    -- it **diverges as 1/step**, which is the signature of a jump in the objective and not of a
    gradient that merely disagrees with a difference. ``jax.grad`` reads -5.98 at the same point
    where the one-sided slope is +24.81: wrong in sign. No refinement of the step converges, so
    no finite-difference audit of a CSG design variable can be made to pass, and an optimiser's
    line search sees a real 3e-2 cliff in ``eps`` from a geometry change of order 1e-17.

    **A TOUCHING PAIR IS NOT A CREASE, AND THE COMPOSED PLANE IS WRONG THERE BY HALF A VOXEL.**
    ``min`` of two signed distances is the union's distance *outside* it and not inside: where
    two children's surfaces are flush -- a rod standing on a substrate of the same material, a
    layer stack, any joint a device is built from -- both ``d`` are zero in the middle of solid
    material, so the composed plane cuts the voxel in half. Measured before the rule below
    (``h = 0.02``, the joint on a voxel centre, 40^3): 52 voxels well inside the rod's footprint
    read ``fill = 0.5`` where the truth is 1.0, with ``sum_c w_c = 1`` and ``w = (0, 0, 1)`` --
    a full z-facing interface through solid metal. It is **first** order in ``h`` in a scheme
    sold at order two (worst ``|dfill|`` stays 0.5000 at ``h`` = 0.1, 0.05, 0.025), it closes
    only if the children are made to overlap by a cell, it vanishes when the joint happens to
    land on a voxel *face*, and both free invariants are satisfied by it. The design makes
    ``fd.Union`` the instruction for a user whose bodies share a material, so this sat on the
    recommended path.

    **What ships: inclusion-exclusion over the children, per voxel, where the pair's shared
    volume is one this file can name exactly.** Per voxel and per pair of children two questions
    are asked, both cheap and both exact:

    *Do the two tangent half spaces hold any volume of this voxel in common?*
    :func:`_no_common_volume` answers it by LP duality over the voxel -- five evaluations of a
    concave piecewise-linear function, no iteration -- and a convex body lies inside its own
    tangent half space, so *no common half-space volume* proves *no common body volume*. The
    pair's shared volume is then **zero**.

    *Are the two faces back to back?* If ``n_i . n_j = -1`` the pair of half spaces is a **slab**,
    whose volume in the voxel is exactly ``max(0, f_i + f_j - 1)`` in the two planes' own fills,
    because the fill of a half space is a monotone function of its offset along one direction.
    That is the case every device joint is: a rod on a substrate, a layer stack, two boxes meeting
    on a plane.

    Where every pair is one of those two, the node answers by inclusion-exclusion over its
    children's **own** routes -- so a union of a box and a cylinder that do not overlap in a voxel
    is exact to the last bit rather than a plane's approximation of one of them:

        Union          fill = sum_i fill_i - sum_pairs shared_ij
        Intersection   fill = 0, area = 0, moment = 0 as soon as ONE pair is apart
        Difference     fill, area, moment = the first child's, untouched

    Where a pair is neither -- two surfaces meeting at an angle *and* overlapping, which is a
    genuine crease -- the composed plane stands unchanged, so the rule can only improve a voxel
    and never spoil one. The interface follows the same two answers: two faces that are back to
    back and overlapping (or flush) are buried inside the material and carry no interface, so
    both areas come out of the sum, and the rod on the substrate reads ``fill = 1.0`` and
    ``w = 0`` in every one of those 60 voxels, on all three backends.

    **The slab term is what keeps the rule continuous at first contact, and where it does not
    reach, the rule has a step of its own.** Driving two children head on through each other past
    one voxel in 3e-4-cell steps, the largest single step in the fill is

        pair                              apart branch alone    with the slab term
        two boxes, faces parallel              4.998e-01            3.0e-04 (= the slide step)
        two spheres, head on                   -                    3.0e-04
        a cylinder standing on a box           -                    3.0e-04
        a sphere and a box tilted by 0.4 rad   -                    3.384e-01

    The last row is the honest cost. Where two surfaces meet at an **angle** and overlap, the
    pair is neither apart nor back to back, the node drops to the composed plane, and the drop is
    a step -- 3.384e-01 here, against 1.5e-04 for the same slide with the rule switched off. What
    the step is *between* is an exact value and an already-wrong one: at that placement the truth
    is 0.62, the sum of the children's own fills reads 0.62, and the composed plane reads 0.28 on
    both sides of the contact before the rule and on one side after it. So the accuracy improves
    and the smoothness in a *geometry parameter* gets worse, in the one quantity this class
    already declares discontinuous and already refuses as a design variable. Over the six-node
    slide family below the worst step did **not** grow (4.29e-02 with the rule against 4.48e-02
    without); 3.384e-01 is an adversarial placement built to find it. The cure that would remove
    it -- interpolating the shared volume between its three exact values at ``n_i . n_j = -1``,
    ``0`` and ``+1`` -- cannot be exact at first contact for an angled pair, because the shared
    volume of two half spaces in a cube is not a function of the two fills and the angle alone;
    only the multi-plane cut priced below is, and it is not affordable.

    **Where the rule is switched off, and what that costs.** It needs ``H_i`` to contain body
    ``i``, so a node with a non-convex child -- a nested CSG node, a non-convex :class:`Prism` --
    keeps the composed plane and keeps the seam error. The constructor says so with a warning
    that names the case, because a silent half-density sheet through a joint is what this rule
    exists to stop. Build such a union flat, over convex leaves, or overlap the children.

    **Why not the exact multi-plane cut, which would fix the seam, the crease and the refusal at
    once.** The cure is to cut the voxel by *all* the active children's planes together -- a box
    intersected with ``k`` half spaces, whose volume and whose per-plane face areas both come out
    of Lasserre's recursion (J. B. Lasserre, *J. Optim. Theory Appl.* **39**, 363 (1983)) -- and
    it was prototyped and measured rather than estimated (off-tree, in this commit's working
    notes; validated against this package's own one-plane closed form to 5.6e-16 in the fill and
    2.6e-18 in the area, and against 4e5-point Monte Carlo to 1.4e-3 at a 7.9e-4 standard error).
    It works: across the bisected crease above the fill steps **1.1e-16** and ``w`` **2.0e-15**,
    and the central difference becomes 24.74, 24.81, 24.81, 24.81, 24.81 -- convergent.

    The previous round rejected it on a *ratio*. Ratios are the wrong instrument, because
    ``occupancy`` runs once per component per solve and once more in a backward pass, not once
    per step; what matters is the absolute cost of one lowering. So it was re-measured in
    **seconds and GiB**, on realistic scenes, on the CSG node's own support box (2026-09-21, this
    machine, 6 cores, ``OMP_NUM_THREADS=2``, float64, peak resident memory over the call):

        scene                          voxels    backend   plane        exact multi-plane
        ring, Difference of two       220 000    numpy     0.102 s      33.5 s   0.34 GiB
        cylinders, h = 0.01                      jax jit   0.009 s       1.8 s   5.35 GiB
                                                 torch     0.047 s      25.5 s   0.08 GiB
        rod on a substrate, Union,    345 600    numpy     0.241 s      57.5 s   0.58 GiB
        h = 0.025                                jax jit   0.024 s       3.2 s   6.81 GiB
                                                 torch     0.139 s      40.8 s   0.16 GiB
        3x3 pillar array on a         288 000    numpy     0.483 s     440.4 s   0.48 GiB
        substrate, Union of ten,                 jax jit   0.054 s      24.7 s  22.22 GiB
        h = 0.025                                torch     0.376 s     330.7 s   0.12 GiB

    That is a *minute* per component on the two eager backends and **6.8 GiB** on the one that is
    fast, for the smallest realistic scene in the set; seven minutes and 22 GiB for a pillar
    array, which is an ordinary metasurface cell. Times three components, and again for a
    backward pass. It is not "seconds and a few hundred MiB" on any of the three, so it does not
    ship; the inclusion-exclusion rule above costs about forty scalar operations per pair per
    voxel instead. What a node now costs is its **children's** routes and not the rule: on the
    48^3 box of the gate's cost table, ``Difference(Sphere, Cylinder)`` reads 761.0 ms on NumPy
    against 786.6 ms for its ``Sphere`` child alone, so the pair test, the two half spaces and
    the composed plane together sit inside the noise of one exact ball clip.

    **A CSG node's FILL is still discontinuous where children genuinely overlap, and that is why
    its geometry may not be traced.** Where a pair does hold volume in common the plane's fill is
    a function of the *selected* normal, so at a crease the selection swaps between children and
    ``fill`` steps although ``d`` does not. Measured over six node kinds, sliding each past one
    voxel in 2400 steps of 2.9e-04 cell and keeping only the steps where both ends are cut
    voxels, the worst single-step change is

        node                     max |dfill|   max |dw|
        sphere minus a rod        3.46e-02      0.895
        sphere u box              4.12e-02      1.000
        two spheres, n            9.78e-03      8.19e-04
        box minus box             4.04e-02      1.000
        prism u cylinder          4.48e-02      1.000
        three-way (two cuts)      3.56e-02      1.000

    -- worst **4.48e-02** in the fill and a **full 1.000** in ``w``. The two previous writings of
    this paragraph quoted 3.15e-02 and 0.566, which is one voxel of one node; a worst over a
    family is what a caller can use. Bisected on the first of them, ``fill`` goes 0.96063617 ->
    0.99217239 across a slide of 5.5e-17 with ``d`` equal to twelve digits. A central difference
    of a loss in that slide then reads

        step      1e-3      1e-4      1e-5      1e-6      1e-7
        plane    268.81   2685.85  26855.47 268551.57  2685512.50

    -- it **diverges as 1/step**, the signature of a jump in the objective and not of a gradient
    that merely disagrees with a difference. ``jax.grad`` reads -5.98 at the same point where the
    one-sided slope is +24.81: wrong in sign. So the constructor refuses a traced parameter by
    name: a forward-only CSG node is honest and a gradient through a jump is not. The refusal is
    on the *leaf* and not on tracing, so a forward solve that traces the points keeps working,
    and every body with an exact fill stays differentiable -- which is where a design variable
    belongs. Making this a stage-5 object is what would lift the refusal.
    """

    #: Is a node of this kind convex when every child is? An intersection of convex sets is
    #: convex and its selected supporting plane contains it, so it may itself be a child of
    #: another node's seam rule; a union and a difference may not.
    _convex_when_children_are = False

    def __init__(self, *bodies):
        if len(bodies) < 2:
            raise ValueError('a CSG node combines at least two bodies.')
        for body in bodies:
            for leaf in body.leaves():
                if traced(leaf):
                    raise ValueError(
                        f'{type(self).__name__}(...): a CSG node\'s geometry must be concrete, '
                        'and this child was built from a traced value. A CSG node\'s FILL is '
                        'discontinuous where its children overlap -- measured over six node '
                        'kinds, a step of up to 4.48e-02 in the fill and a full 1.000 in w '
                        'across a geometry change of 5.5e-17 -- so a central difference of a '
                        'loss through one DIVERGES as 1/step (-775, -7845, -78549 at 1e-5, '
                        '1e-6, 1e-7 against a gradient of +21.76) and an optimiser sees a '
                        'cliff. A forward-only CSG node is honest and a gradient through a jump '
                        'is not, so the parameter is refused rather than differentiated. Build '
                        'the node outside the transform and trace the POINTS, which is what a '
                        'forward solve does; to differentiate a dimension, use a body with an '
                        'exact fill (Sphere, Cylinder, Box, Prism), whose fill is continuous.')
        self.bodies = bodies
        self._seam_rule = all(body.convex for body in bodies)
        self.convex = self._seam_rule and self._convex_when_children_are
        if not self._seam_rule:
            loose = ', '.join(sorted({type(b).__name__ for b in bodies if not b.convex}))
            warnings.warn(
                f'{type(self).__name__}(...): a child is not convex ({loose}), so the flush-seam '
                'rule is off for this node and two children whose surfaces TOUCH will read half '
                'the material along the joint (measured: 52 voxels at fill 0.5 where the truth '
                'is 1.0, first order in h, with a full interface normal in solid material). The '
                'rule needs a child to lie inside its own tangent half space, which a nested CSG '
                'node and a non-convex Prism do not. Build the node flat over convex leaves '
                '(Sphere, Cylinder, Box, convex Prism), or overlap the children by at least one '
                'cell.', stacklevel=2)

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
        """The child pairs whose overlap decides this node. Every pair, unless a node says less."""
        k = len(self.bodies)
        return [(i, j) for i in range(k) for j in range(i + 1, k)]

    def _pair_terms(self, faces, dx, xp):
        """Per pair: is it resolved, what volume do the two share, are their faces buried?"""
        half = [_parts(d, n, dx)[0] for d, n in faces]
        zero, one = xp.zeros_like(half[0]), xp.ones_like(half[0])
        resolved, shared, buried = None, {}, {}
        for i, j in self._pairs():
            (d_i, n_i), (d_j, n_j) = faces[i], faces[j]
            apart = _no_common_volume(d_i, n_i, d_j, n_j, dx, xp)
            opposed = 1 + xp.sum(n_i * n_j, axis=-1) <= _SAME_FACE
            slab = xp.maximum(half[i] + half[j] - one, zero)
            shared[i, j] = xp.where(apart, zero, slab)
            buried[i, j] = opposed & (d_i + d_j <= _SAME_FACE * dx)
            ok = apart | opposed
            resolved = ok if resolved is None else resolved & ok
        return resolved, shared, buried

    def _resolved(self, child, faces, dx, xp):
        """``(where this node has a rule, fill, area, moment)`` -- the rule's answer, per voxel.

        The first element is a per-voxel predicate, not a Python flag: whether the rule applies
        is a property of the placement and changes from voxel to voxel. The composed plane
        answers everywhere it is False.
        """
        raise NotImplementedError

    def parts(self, points, dx, surface=None):
        xp = array_namespace(points)
        faces = [body.surface(points) for body in self.bodies]
        d, n = faces[0]
        for other in faces[1:]:
            d, n = self._compose(d, n, *other, xp)
        plane = _parts(d, n, dx)
        if not self._seam_rule:
            return plane
        child = [body.parts(points, dx, surface=face)
                 for body, face in zip(self.bodies, faces)]
        ok, fill, area, moment = self._resolved(child, faces, dx, xp)
        return (xp.where(ok, fill, plane[0]), xp.where(ok, area, plane[1]),
                xp.where(ok[..., None], moment, plane[2]))


class Union(_Combination):
    """Everything in any of them.

    Where no two children hold volume of a voxel in common, the fill is the **sum** of theirs and
    the interface is the sum of theirs less any pair of faces lying back to back; see
    :class:`_Combination` for the measurement that made that necessary.
    """

    def _compose(self, d, n, other_d, other_n, xp):
        take = other_d < d
        return xp.where(take, other_d, d), xp.where(take[..., None], other_n, n)

    def _resolved(self, child, faces, dx, xp):
        ok, shared, buried = self._pair_terms(faces, dx, xp)
        fill = sum(one[0] for one in child) - sum(shared.values())
        area = sum(one[1] for one in child)
        moment = sum(one[2] for one in child)
        for (i, j), gone in buried.items():
            area = area - xp.where(gone, child[i][1] + child[j][1], xp.zeros_like(area))
            moment = moment - xp.where(gone[..., None], child[i][2] + child[j][2],
                                       xp.zeros_like(moment))
        return ok, fill, area, moment


class Intersection(_Combination):
    """Only what is in all of them.

    The one rule here is the empty one: as soon as a single pair of children holds no volume of
    the voxel in common, the intersection holds none either, whatever the other pairs do. The
    slab term :class:`Union` uses is deliberately not taken -- for two half spaces it would be an
    upper bound on the bodies' shared volume, which is the wrong side to be on for a body that
    *is* the shared volume.
    """

    #: An intersection of convex bodies is convex.
    _convex_when_children_are = True

    def _compose(self, d, n, other_d, other_n, xp):
        take = other_d > d
        return xp.where(take, other_d, d), xp.where(take[..., None], other_n, n)

    def _resolved(self, child, faces, dx, xp):
        empty = None
        for i, j in self._pairs():
            apart = _no_common_volume(*faces[i], *faces[j], dx, xp)
            empty = apart if empty is None else empty | apart
        zero = xp.zeros_like(child[0][0])
        return empty, zero, zero, xp.zeros_like(child[0][2])


class Difference(_Combination):
    """The first, minus everything after it. The removed body's normal comes back reversed.

    Only the first child against each of the others is asked: nothing else removes anything.
    Where a removed body holds no volume of the voxel in common with the first, the answer is the
    first child's own, untouched -- which is how a body resting *against* a cut face stays out of
    it. The slab term is not taken here either: it would remove an upper bound on what the second
    body actually removes.
    """

    def _compose(self, d, n, other_d, other_n, xp):
        take = -other_d > d
        return xp.where(take, -other_d, d), xp.where(take[..., None], -other_n, n)

    def _pairs(self):
        return [(0, j) for j in range(1, len(self.bodies))]

    def _resolved(self, child, faces, dx, xp):
        apart = None
        for i, j in self._pairs():
            one = _no_common_volume(*faces[i], *faces[j], dx, xp)
            apart = one if apart is None else apart & one
        return (apart,) + child[0]
