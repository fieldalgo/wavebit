"""The bodies: a signed distance, its gradient, and the closed forms each one happens to own.

A **body** is the smoothed, differentiable thing stage 4 adds, and it is not the same object as
a ``Shape`` in ``wavebit/fd/geometry.py``. A ``Slab`` or a ``Box`` there is cell-snapped and
answers questions about *cell indices* -- which cells it fills, which it may touch, whether it
is uniform across a layer. A body answers questions about a *point*: how far is your surface
from here, which way does it face, and -- where it can do better than a plane -- exactly how
much of this voxel do you fill. The two meet in C2, which is the only place that will know both.

Every body answers three things:

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

**Where the exact forms are, and where they are not**

    body                fill                              moment
    Box, Prism          polygon-box clip x 1-D overlap    edge moments x overlap + cap areas
    Cylinder            disc-box clip x 1-D overlap       *tangent plane*
    Sphere              ball-box clip                     *tangent plane*
    Union/Intersection/
    Difference          *tangent plane*                   *tangent plane*

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
    CSG (sphere minus rod)   2301    0.5660      3.1e-02

so on a cap rim ``w`` swaps from the side wall's ``(0.5, 0.5, 0)`` to the cap's ``(0, 0, 1)``
inside one 0.0025-cell move. The design's G4 ("largest single-step change <= 1e-5") was measured
on a sphere and is out by five orders of magnitude there. An earlier writing of this paragraph
quoted a superseded dataset (an axis-aligned box at 0.0000) beside a test asserting the same
box jumps by more than 0.3; the table above is the test's own.

**The fill is smooth for every body with an exact fill, and it is not for a CSG node.** The
last row is a jump in ``fill`` and not only in ``w``: bisected, 0.96063617 -> 0.99217239 across
a slide of 5.5e-17, a step of **3.15e-02**, while ``d`` holds to twelve digits. See
:class:`_Combination` for the mechanism, the price of the fix, and the refusal that follows
from it.
"""
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
    """
    detach = getattr(value, 'detach', None)
    if detach is not None:
        try:
            value = detach()
        except Exception:                   # noqa: BLE001 - a wrapper that will not detach
            pass
    for read in (lambda v: _np.asarray(v, dtype=float),
                 lambda v: _np.asarray(v.cpu().numpy(), dtype=float),
                 lambda v: _np.asarray(v.item(), dtype=float)):
        try:
            return read(value)
        except Exception:                   # noqa: BLE001 - try the next spelling
            continue
    if _module_of(value) == 'jax':
        import jax                          # local: jax is optional
        try:
            return _np.asarray(jax.lax.stop_gradient(value), dtype=float)
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


def readable(value):
    """``value`` as a plain NumPy array **for a check**, or ``None`` when there is no value.

    The array twin of ``wavebit._core.backend.checked_float``. Reading a differentiated value to
    *check* geometry is the case that most needs checking -- the memory this repository carries
    is a rod grown by gradient ascent to three times its period with every refusal silently
    skipped, because the check had been spelled ``float(x)``.

    **Which transforms it holds under, measured and not asserted**
    (``tests/test_fd_shapes.py::test_a_door_holds_under_every_transform_that_carries_a_value``):

        transform                              read?  how
        a float, a list, NumPy                  yes   asarray
        a concrete jax.Array, a Torch tensor    yes   asarray (detached first)
        torch autograd, ``requires_grad``       yes   detach, asarray
        jax.grad, scalar leaf                   yes   item
        jax.grad, array leaf                    yes   stop_gradient
        jax.vmap, jax.vmap(jax.grad)            yes   the batch tracer's value, one layer down
        torch.func.grad / jacrev                yes   item, or functorch's unwrapped tensor
        torch.func.vmap, torch.func.vmap(grad)  yes   functorch's unwrapped tensor
        **jax.jit, jax.jit(jax.grad)**          NO    there is no number anywhere to read

    The first writing of this function claimed the last row was the only one, and was wrong on
    three: ``torch.func.grad`` came back ``None`` and ``Sphere(radius=-0.3)`` was accepted there
    with a finite gradient of 135.00000000000014 -- this repository's own gradient-ascent scar,
    in the modern Torch functional API, which ``checked_float`` (the function this one calls
    itself the twin of) reads correctly. ``jax.vmap`` and ``torch.func.vmap`` were the same
    hole. Only ``jax.jit`` is a genuine dead end: a ``DynamicJaxprTracer`` carries a ``Var`` and
    not a value, and a *jitted* gradient is one too, so under it a door cannot be a refusal.
    :func:`check_sizes` puts there what can hold -- a NaN.
    """
    for _ in range(_UNWRAP_DEPTH):
        read = _read_once(value)
        if read is not None:
            return read
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

    The values come back because of the one transform that carries no value at all: under
    ``jax.jit`` the refusal cannot raise, so what is returned is :func:`_loud_if_not_a_size` of
    the length instead of the length. Everywhere else -- including under ``jax.grad``,
    ``jax.vmap``, Torch autograd and every ``torch.func`` transform (:func:`readable`) -- this
    raises, and the value comes back untouched.
    """
    out = []
    for what, value in named.items():
        if value is None:
            raise ValueError(f'{where}({what}=...): a length, and there is none; got None.')
        read = readable(value)
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
    """``value`` as an array beside ``like`` -- and untouched if it already is one (it may be traced).

    ``device=`` is read off ``like`` with a default because a JAX tracer has a ``dtype`` and no
    ``device``: inside ``jax.jit`` the placement is the trace's, not an attribute to copy.
    """
    if hasattr(value, 'dtype') and hasattr(value, 'device'):
        return value
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

    def __init__(self, radius, center=(0., 0., 0.)):
        self.radius, = check_sizes('Sphere', radius=radius)
        self.center = center

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
        best = zero + float('inf')
        near = xp.stack([zero, zero], axis=-1)
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
            # Even-odd crossing count, the same test the RCWA path's membership uses: a ray
            # along +u from q crosses this edge iff the edge straddles q's v and the crossing
            # is to the right. A horizontal edge straddles nothing and is skipped by the guard.
            dv = b[..., 1] - a[..., 1]
            straddle = (a[..., 1] > q[..., 1]) != (b[..., 1] > q[..., 1])
            cross = a[..., 0] + (e[..., 0] / xp.where(straddle, dv, xp.ones_like(dv))) * (q[..., 1] - a[..., 1])
            inside = inside ^ (straddle & (q[..., 0] < cross))
        one = xp.ones_like(zero)
        sign = xp.where(inside, -one, one)
        live = best > 0
        norm = xp.where(live, xp.sqrt(xp.where(live, best, one)), one)
        # On the boundary itself the direction is genuinely undefined; +u is returned and the
        # fill is the same for any of them, because the voxel is cut through its own centre.
        fallback = xp.stack([one, zero], axis=-1)
        g = xp.where(live[..., None], sign[..., None] * near / norm[..., None], fallback)
        return sign * xp.sqrt(best), g

    def _area(self, box, points):
        xp = array_namespace(points)
        return self._turn * clipped_polygon_area(self._vertices(points), *box, xp,
                                                 convex=self._convex)

    def exact_moment(self, points, dx):
        xp, (u, v, k), box, (lo, hi, s) = self._boxes(points, dx)
        length, muu, mvv = polygon_edge_moments(self._vertices(points), *box, xp)
        length, muu, mvv = self._turn * length, self._turn * muu, self._turn * mvv
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
        cap_area = caps * self._area(box, points)
        area = length * along + cap_area
        return area, _stack3(xp, (muu * along, mvv * along, cap_area), (u, v, k), area)


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
    _convex = True

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

    **The proper fix, priced.** The cure is to cut the voxel by *all* the active children's
    planes at once -- a box intersected with ``k`` half spaces, a convex polyhedron whose volume
    is ``(1/3) sum_f (p_f . n_f) A_f`` over its faces, each face a 2-D clip of a square by the
    other ``5 + k`` half planes -- which also gives the area-weighted moment for a crease for
    free, because each cut plane's own face area is one of the ``A_f``. It was prototyped and
    measured rather than estimated -- off-tree, in this commit's working notes, and not
    shipped -- and validated against
    Monte Carlo to 9.5e-04 on 400k points and against this package's own one-plane closed form
    to 6.7e-16 in the fill, 1.0e-17 in the area and 8.7e-18 in the moment). It works: across the
    same bisected crease the fill steps **1.1e-16** and ``w`` **2.0e-15**, and the central
    difference above becomes 24.74, 24.81, 24.81, 24.81, 24.81 -- convergent. Over the whole
    2400-step slide the largest single-step change falls from 3.12e-02 to 3.03e-03 in ``fill``
    and from 5.66e-01 to 2.08e-02 in ``w``.

    It is not taken, and the threshold was written down before the measurement: *prohibitive* is
    more than **10x** the tangent plane's time on the support box on any backend, or more than
    **3x** its peak memory. The exact ball's 15x was accepted because it is a setup cost on the
    one body stage 4 is graded on; a CSG node is the general case, in every component of every
    solve, so the same multiplier there would be the whole lowering stage. Measured on the same
    110592-voxel support box:

        backend      plane    polyhedron    ratio
        numpy       67.0 ms   11020.5 ms     164x
        jax (jit)    6.3 ms    1379.8 ms     220x
        torch       23.1 ms    8649.2 ms     375x
        numpy peak  23.3 MiB    331.3 MiB   14.2x

    16 to 37 times over the time threshold and 4.7 times over the memory one, for ``k = 2``; a
    three-child node is worse, and a *union* is worse again because a union of half spaces is
    not convex and needs an inclusion-exclusion over ``2^k`` intersections. So the plane ships,
    the jump is documented above with its measured size, and the constructor refuses a traced
    parameter by name: a forward-only CSG node is honest and a gradient through a jump is not.
    The refusal is on the *leaf* and not on tracing, so a forward solve that traces the points
    keeps working, and every body with an exact fill stays differentiable -- which is where a
    design variable belongs. Making this a stage-5 object is what would lift the refusal.
    """

    def __init__(self, *bodies):
        if len(bodies) < 2:
            raise ValueError('a CSG node combines at least two bodies.')
        for body in bodies:
            for leaf in body.leaves():
                if traced(leaf):
                    raise ValueError(
                        f'{type(self).__name__}(...): a CSG node\'s geometry must be concrete, '
                        'and this child was built from a traced value. A CSG node\'s FILL is '
                        'discontinuous at a crease -- measured here, a step of 3.15e-02 across '
                        'a geometry change of 5.5e-17 -- so a central difference of a loss '
                        'through one DIVERGES as 1/step (-775, -7845, -78549 at 1e-5, 1e-6, '
                        '1e-7 against a gradient of +21.76) and an optimiser sees a cliff. A '
                        'forward-only CSG node is honest and a gradient through a jump is not, '
                        'so the parameter is refused rather than differentiated. Build the node '
                        'outside the transform and trace the POINTS, which is what a forward '
                        'solve does; to differentiate a dimension, use a body with an exact '
                        'fill (Sphere, Cylinder, Box, Prism), whose fill is continuous.')
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


class Union(_Combination):
    """Everything in any of them."""

    def _compose(self, d, n, other_d, other_n, xp):
        take = other_d < d
        return xp.where(take, other_d, d), xp.where(take[..., None], other_n, n)


class Intersection(_Combination):
    """Only what is in all of them."""

    def _compose(self, d, n, other_d, other_n, xp):
        take = other_d > d
        return xp.where(take, other_d, d), xp.where(take[..., None], other_n, n)


class Difference(_Combination):
    """The first, minus everything after it. The removed body's normal comes back reversed."""

    def _compose(self, d, n, other_d, other_n, xp):
        take = -other_d > d
        return xp.where(take, -other_d, d), xp.where(take[..., None], -other_n, n)
