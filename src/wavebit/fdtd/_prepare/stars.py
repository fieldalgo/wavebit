"""Keep the smoothed inverse-material operator positive definite: the per-node star lift.

**The operator.** On one ``ConstitutiveTerm`` region the default rule's inverse-material operator
(inverse ``eps`` for E, inverse ``mu`` for H) is ``Xi = diag(e) + C``: ``e`` the edge diagonal
``coefficients.build`` paints (the voxel's own mix, ``coefficients.mix``) and ``C`` the node
cross coefficients ``constitutive.lower`` paints, applied by ``cross_action`` with ``c/4`` for
each pair of edges that share a node. It is the operator Werner, Bauer and Cary call wc07
(J. Comput. Phys. 247, 211, 2013, Eq. 11), after Oskooi, Kottke and Johnson (Opt. Lett. 34,
2778, 2009).

**Why it can fail.** ``Xi`` is positive definite only if the energy ``v . Xi v`` is positive for
every field, and nothing in the rule makes it so: at a high-contrast surface the node's cross
coefficient can outweigh the edges' diagonals. Then some field has negative energy and a run
grows without bound -- while the discrete energy, which leapfrog conserves exactly, stays flat
until round-off shows (``tests/fdtd/test_fd_constitutive.py::test_an_unlifted_eps_100_sphere_grows``).

**The stars.** ``Xi`` is a sum of one 6x6 matrix per node ``p`` -- its star: the six edges that
touch ``p``, two along each axis (the one whose array index is ``p``'s own, and the other: ``p -
e_a`` for E, whose node is an edge's low end, ``p + e_a`` for H, whose cell centre is a dual
edge's high end), with half of each edge's diagonal and every coupling through ``p`` at its
full ``c/4``::

    Xi = sum_p P_p^T S_p P_p

An edge touches two nodes, so its two halves make ``e``; a coupling lives at one node. **If every
star is positive definite, so is** ``Xi``. The star has a closed test: minimising its energy over
how each axis's field splits between the axis's two edges leaves ``(1/4)(diag(H) + offdiag(c))``,
``H_a`` the harmonic mean of the two ``a``-edges' diagonals (an edge outside the region counts as
infinite, ``H_a = 2 e_a``), so ``S_p`` is positive definite exactly when both halves are positive
and ``diag(H) + offdiag(c)`` is.

This is Werner, Bauer and Cary's Sec. 4 argument with a larger piece. Their pieces are the eight
3x3 blocks of one edge per axis at each node, whose test takes, for each axis, *one* of the two
edges -- so it binds on the smaller one. The star's harmonic mean lies between the two, and the
star fires far less. In the lift's study (``studies/fdtd/subpixel_lift/fdtd_subpixel_lift.py``) the eight blocks
moved 3.1% and 2.0% of an ``eps = 12`` sphere's blocks at 8 and 16 cells across (the mean over
five placements, ``THETA = 0.02``), though its operator was positive definite, and the error they
added did not shrink with the cell: at ``eps = 25``, placement ``c``, ``THETA = 0.05``, the static
error read -0.20, -0.26 and -0.20% at 12, 16 and 24 cells across while the unmoved rule's went
+0.96, +0.60 and +0.36%. The star moves none of the ``eps = 12`` sphere's nodes. The star is ours.

**The lift.** Each star moves both edges of each axis toward the node tensor's diagonal,
``e -> e + t (k - e)`` for ``t`` in ``[0, 1]``, where ``k`` is the diagonal of the node's Kottke
tensor ``K = (1/<eps>) I + (<1/eps> - 1/<eps>) n n^T`` (Kottke, Farjadpour and Johnson, Phys. Rev.
E 77, 036611, 2008), whose off-diagonal ``c`` already is. At ``t = 1`` both ``a``-edges read
``k_a``, the harmonic mean is ``k_a`` and the test matrix is ``K`` -- or larger on an axis with a
single edge, which reads ``2 k_a`` -- positive definite with smallest eigenvalue at least
``1/<eps>``, to round-off (a fill of ``-6e-14`` is kept by design, ``coefficients._Smoothed``).
``S_p(t)`` is affine in ``t``, so its smallest eigenvalue is concave in ``t`` and the ``t`` where
the star clears a floor form an interval ``[t*, 1]``. The lift takes
``t*``, the least move, and each edge's diagonal becomes the sum of its two halves:
``e + (1/2) sum_p t_p (k_p - e)`` over its two end nodes. That is Werner, Bauer and Cary's step-6
fallback (a failing piece replaced by the node tensor) made minimal and continuous.

**The floor** is ``S_p - (f/2) I`` positive definite, ``f = THETA * min(1/<eps>, the six edges'
diagonals)``: the star's physical scale, and never below zero -- a negative ``f``, which only a
value past float64's reach can give (below), would make the test weaker than positivity. Two
properties follow:

* a star with no cross coefficient has smallest eigenvalue ``min(e)/2 >= f/2`` and never moves,
  so a region with nothing below the floor lowers to the same bits it did before (the lift
  returns no change at all, and ``be`` is not touched);
* ``t*`` is continuous in every input: at ``t = 1`` the star clears the floor strictly, the
  crossing is from below, and a star that approaches the floor starts to move from ``t = 0``.
  A trigger at zero with a lift to the floor would jump (the understand workflow's correction).

**The certificate** is the construction, checked: every star at its ``t`` is tested positive
definite, on its two halves ``e + t (k - e)``, whose sum is the kernel's diagonal to round-off.
That test is at a floor of zero; the bisection's invariant adds the margin, so that ``Xi`` is
positive definite with smallest eigenvalue at least the smallest, over edges, of the mean floor of
the edge's two end nodes. The check does not assume the construction worked -- it is what
``constitutive.validate`` refuses on and what ``constitutive.guard`` turns into ``stable = NaN``
under a transform. A node one layer past a region's far end (the high end for E, the low end for
H) holds half of that end's edges and no coupling: it lies outside the cross array, which
``cross_action`` extends by zero -- a diagonal star that needs no test.

**Float64's reach.** A fill's round-off (up to ``-6e-14``, and ``-7e-14`` at a cell-centred sphere)
times the contrast breaks the mix itself from ``eps`` about ``1e12``: the design review measured an
edge diagonal of -15 and a node's ``1/<eps>`` of 3.9 in place of about 1 at ``1e13``. The lift
cannot repair a diagonal that is not a number of the right sign, the certificate fails, and the
plan refuses it by name. Measured on spheres: every contrast up to ``1e11`` certified; refusals
began at ``1e12`` for a cell-centred sphere 40 cells across, while a node-centred one 12 cells
across held to ``1e15``.

**The test** is the sign of the three pivots of an ``L D L^T`` (Cholesky) factorisation of the
3x3 test matrix, elementwise on every backend. Spellings with the leading principal minors were
measured on the first, eight-block form and dropped: unscaled they misjudged 331 blocks at
``eps = 1e10`` (the design review); scaled, the determinant touches zero like ``(t - t*)^2`` at a
double root, so its sign resolved ``t*`` only to about ``1e-8``, and a finite-difference gradient
on a cell-centred sphere scattered by 6% at a step of ``1e-6``. A pivot crosses zero linearly at
a double root as at a simple one.

**Ghost edges.** ``cross_action`` extends by zero past a region's end, which drops the edges
outside it: the region's operator is a principal submatrix, and a principal submatrix of a
positive definite matrix is positive definite. A star whose other ``a``-edge is outside has one
``a``-edge, which the harmonic mean reads as ``2 e_a``.

**Values and traces.** ``t*`` is found by bisection, on the host where the inputs can be read
(NumPy, a concrete or ``jax.grad`` JAX array, a Torch tensor under autograd) and only on the
stars below the floor; under ``jax.jit`` and the other transforms that carry no value, on every
star in the runtime's own arrays.

**Derivatives, to second order.** ``t*`` is an implicit function of the inputs: the ``t`` where a
root function ``F`` of the star's test matrix vanishes. ``F`` is an ``L D L^T`` pivot of that
matrix, the axes ordered per star by its null vector (:func:`_root`, :func:`_order`). Taken at the
bisected ``t*`` as a Taylor series in ``t`` on the live inputs, ``F + F_1 s + F_2 s^2``, its
reversion moves ``t`` by ``-g/F_1 - (F_2/F_1^3) g^2`` with ``g = F - sg F``, which is zero
(:func:`_assemble`): ``t`` is ``t*`` to the bit, and its first and second derivatives are exact on
every backend and under nested transforms, since nothing in the differentiated path is more than
elementwise arithmetic. The third derivative is not. Measured 2026-09-27: against a 50-digit
oracle of the definition (``tests/references/gen_star_lift_oracle.py``) the value, first and
second derivatives agree to 6.7e-16 on JAX, ``jax.jit`` and Torch; ``jax.hessian``,
``jax.jit(jax.hessian)`` and Torch's double backward of a lowered sum agree with central
differences of its gradient to 4.6e-8 at a step of ``1e-4`` of ``eps``, falling as the step
squared, on spheres off a node, on a node, on a cell and against the walls. The first form of the
derivative -- one step along a constant eigenvector -- was exact to first order only, 188% off in
the Hessian (the review of the star); two Newton steps from the root, exact too, ran ``jax.jit``
of a Hessian 600 times slower than this.

Where a star's smallest eigenvalue is double -- the stars of a sphere's body diagonals against a
wall, and of the ``eps = 100`` sphere centred in a cell -- ``t*`` has a derivative only along
directions that keep the degeneracy (a single body's ``eps`` or ``mu`` does), and none in general:
the review of the star measured one-sided slopes +1.107 and -1.601 along a random direction. There
``F`` is ``d2``, smooth along those directions. A determinant as ``F`` loses even the first
derivative at a double root: it touches zero quadratically there, and on the eight-block form its
derivative was 44% off (the design review).
"""
import collections

import numpy as np

#: The floor as a fraction of the star's physical scale; it must lie in ``(0, 1)``. Chosen by
#: ``studies/fdtd/subpixel_lift/fdtd_subpixel_lift.py``'s rule, written before its runs: the smallest worst static
#: error of the sphere over eps 12-100 at 8 and 16 cells across, among the values that move no
#: star of an eps <= 6 body. Every value tried moved none (0.02-0.3, 126 bodies); the worst
#: errors were 5.189 / 5.196 / 5.207 / 5.239 / 5.279% at 0.02 / 0.05 / 0.1 / 0.2 / 0.3 (9.217%
#: unlifted), so accuracy hardly depends on it. What it does set is the margin: the eps-100
#: sphere's smallest eigenvalue times eps is 0.343 at 0.02 against 0.395 at 0.1.
THETA = 0.02

#: Bisection steps: ``2**-52`` of the unit interval, the spacing of float64 in ``[0.5, 1]``.
STEPS = 52

#: Where ``|d2| / (|m_1| + |m_2| + |m_3|)`` at ``t*`` is below this, the star's root is taken to be
#: double and its root function is ``d2`` rather than ``d3`` (see :func:`_root`). Measured
#: 2026-09-27 over every moved star of 14 scenes (spheres at eps 40 on and off nodes, on cells and
#: 1e-9 to 1e-3 cells off a node, against PEC walls at eps 1000, eps 100 at 16 cells across; a
#: turned box and two cylinders at 25): with the axes in :func:`_order`'s order the ratio is either
#: at most 1.9e-13 (14 stars, all at walls or on the eps-100 sphere's diagonals: double roots in
#: exact arithmetic, split by rounding) or at least 1.5e-4 -- nothing in between. In the fixed
#: ``x, y, z`` order it ran continuously from 1e-17 to 3e-6, which is why the order is chosen per
#: star. The two ways of being wrong are not alike, measured on synthetic stars by the review:
#: ``d2`` for a simple root misreads even the first derivative, 2 to 440 times over; ``d3`` for a
#: split double root loses about ``1.6e-15`` over the ratio (2e-3 at 8e-13). ``1e-11`` takes the
#: largest double measured with a margin of 50, and misreads a simple root only where ``d3``
#: would itself have lost 1e-4.
TAU = 1e-11

#: The cross coefficients' order in a term's array: xy, xz, yz (``cross_action``'s).
PAIRS = ((0, 1), (0, 2), (1, 2))

#: What :func:`lift` returns. ``delta`` the three components' change to the inverse-material
#: diagonal over the region, or ``None`` when no star moves; ``certified`` whether every star
#: at its ``t`` is positive definite (a Python bool on the host, a runtime bool under a
#: transform); ``moved``, ``stars`` and ``largest`` the stars lifted, the stars with a cross
#: coefficient, and the largest ``t`` -- ``None`` under a transform, which has no value to count.
Lifted = collections.namedtuple('Lifted', 'delta certified moved stars largest')


def _shift(xp, value, step, axis):
    """``value`` moved one cell along array ``axis``: ``out[i] = value[i - step]``, zero entering.

    ``cross_action``'s bounded shift, out of place on every backend.
    """
    n = value.shape[axis]
    head = [slice(None)] * value.ndim
    head[axis] = slice(0, 1)
    zero = xp.zeros_like(value[tuple(head)])
    keep = [slice(None)] * value.ndim
    if step > 0:
        keep[axis] = slice(0, n - 1)
        return xp.concat([zero, value[tuple(keep)]], axis=axis)
    keep[axis] = slice(1, None)
    return xp.concat([value[tuple(keep)], zero], axis=axis)


def _ghost(shape, a, field):
    """Where the node's other ``a``-edge lies outside the region: the low end for E, the high
    end for H."""
    mask = np.zeros(shape[2 - a], dtype=bool)
    mask[0 if field == 'E' else -1] = True
    return mask.reshape(tuple(-1 if axis == 2 - a else 1 for axis in range(3)))


def _star(xp, field, diagonal, node, ghosts):
    """Each node's two ``a``-edge diagonals: its own-index edge and the other, which reads
    ``node[a]`` where it is a ghost (a placeholder: the ghost mask decides, not the value)."""
    step = 1 if field == 'E' else -1
    own = [diagonal[a] for a in range(3)]
    other = [xp.where(ghosts[a], node[a], _shift(xp, diagonal[a], step, 2 - a)) for a in range(3)]
    return own, other


def _floor(xp, theta, inverse, own, other, ghosts):
    """``THETA * min(1/<eps>, the star's edges)``, not below zero; a ghost's placeholder is
    ``k_a >= 1/<eps>``."""
    low = inverse
    for a in range(3):
        low = xp.minimum(low, xp.minimum(own[a], xp.where(ghosts[a], own[a], other[a])))
    return theta * xp.maximum(low, xp.zeros_like(low))


def _halves(own, other, node, t, floor):
    """Each axis's two edges at ``t``, floored: ``e + t (k - e) - f``."""
    return ([u + t * (k - u) - floor for u, k in zip(own, node)],
            [v + t * (k - v) - floor for v, k in zip(other, node)])


def _harmonic(xp, x, y, ghost, ok):
    one = xp.ones_like(x)
    return xp.where(ghost, 2 * x, 2 * x * y / xp.where(ok, x + y, one))


def _positive(xp, m, off):
    """The 3x3 matrix's ``L D L^T`` pivots are all positive (see the module): ``m_x``,
    ``m_y - c_xy^2 / m_x`` and
    ``m_z - c_xz^2 / m_x - (c_yz - c_xy c_xz / m_x)^2 / (m_y - c_xy^2 / m_x)``.
    """
    (mx, my, mz), (cxy, cxz, cyz) = m, off
    one = xp.ones_like(mx)
    ok = mx > 0
    first = xp.where(ok, mx, one)
    second = my - cxy * cxy / first
    ok = ok & (second > 0)
    lower = cyz - cxy * cxz / first
    return ok & (mz - cxz * cxz / first - lower * lower / xp.where(ok, second, one) > 0)


def _above(x, y, ghosts):
    """Every edge of the star above the floor: where the harmonic means are defined."""
    ok = None
    for a in range(3):
        here = (x[a] > 0) & (ghosts[a] | (y[a] > 0))
        ok = here if ok is None else ok & here
    return ok


def _test(xp, own, other, ghosts, node, off, t, floor):
    """Whether ``S_p(t) - (f/2) I`` is positive definite, and its 3x3 test matrix's diagonal."""
    x, y = _halves(own, other, node, t, floor)
    ok = _above(x, y, ghosts)
    h = [_harmonic(xp, x[a], y[a], ghosts[a], ok) for a in range(3)]
    return ok & _positive(xp, h, off), h


def _bisect(xp, scan, own, other, ghosts, node, off, floor):
    """The least ``t`` in ``[0, 1]`` with the star above its floor, to ``2**-52``.

    ``hi`` keeps the property throughout: at ``t = 1`` the star is the node tensor's, positive
    definite for finite positive inputs. Past float64's reach it may not be (see the module),
    and the certificate, not this function, reports it.
    """
    def body(carry, n):
        lo, hi = carry
        mid = 0.5 * (lo + hi)
        ok, _ = _test(xp, own, other, ghosts, node, off, mid, floor)
        return xp.where(ok, lo, mid), xp.where(ok, mid, hi)
    _, hi = scan(body, (xp.zeros_like(floor), xp.ones_like(floor)), STEPS)
    return hi


def _loop(body, carry, steps):
    for n in range(steps):
        carry = body(carry, n)
    return carry


def _lowest(xp, h, off):
    """The eigenvector of the smallest eigenvalue of ``diag(h) + offdiag(off)``: three arrays."""
    (cxy, cxz, cyz) = off
    matrix = xp.stack([xp.stack([h[0], cxy, cxz], axis=-1), xp.stack([cxy, h[1], cyz], axis=-1),
                       xp.stack([cxz, cyz, h[2]], axis=-1)], axis=-2)
    _, vectors = xp.linalg.eigh(matrix)
    return [vectors[..., a, 0] for a in range(3)]


def _pick(xp, values, index):
    """``values[index]`` elementwise, for a per-star axis index held as a float array."""
    return xp.where(index == 0, values[0], xp.where(index == 1, values[1], values[2]))


def _order(xp, v):
    """Each star's axes, ``(first, middle, last)``, from the null vector ``v`` of its test matrix:
    last the largest ``|v_a|``, the other two in axis order. Float arrays of 0, 1, 2.

    With the eigenvalues ``0 = l_1 <= l_2 <= l_3`` at the root, the leading 2x2 minor is
    ``d1 d2 = l_2 l_3 v_last^2`` (the adjugate's last diagonal entry), ``v_last^2 >= 1/3``, and
    ``d1 = m_first <= l_3``: so ``|d2| >= l_2 / 3``, small only where the root is nearly double.
    Which of the other two comes first changes ``d1`` alone, and ``d1 >= l_2 (1 - v_first^2) >=
    l_2 / 2`` either way.
    """
    a = [xp.abs(u) for u in v]
    zero = xp.zeros_like(a[0])
    last = xp.where((a[0] >= a[1]) & (a[0] >= a[2]), zero,
                    xp.where(a[1] >= a[2], zero + 1, zero + 2))
    first = xp.where(last == 0, zero + 1, zero)
    return first, 3 - first - last, last


def _times(a, b):
    """The product of two Taylor series in ``t``, truncated at their length."""
    return [sum(a[i] * b[k - i] for i in range(k + 1)) for k in range(len(a))]


def _over(a, b):
    """``a / b`` for two Taylor series in ``t``, truncated at their length; ``b[0]`` is not 0."""
    inverse = 1 / b[0]
    out = []
    for k in range(len(a)):
        out.append((a[k] - sum(out[i] * b[k - i] for i in range(k))) * inverse)
    return out


def _root(xp, own, other, ghosts, node, off, t, floor, double):
    """The star's root function at ``t + s`` to ``s^2``, ``[F, F_t, F_tt / 2]``: the binding
    ``L D L^T`` pivot of the test matrix ``diag(H) + offdiag(off)``, its axes already in the
    star's own order (:func:`_order`), ``off`` the pairs first-middle, first-last, middle-last.

    ``d3``, the last pivot, is ``det`` over the leading 2x2 minor: at a simple root it crosses zero
    linearly, and with the null vector's largest component last the leading block stays well away
    from singular -- in the fixed ``x, y, z`` order a star nearly uncoupled along one axis left
    ``d2`` as small as ``1e-17`` of the scale and ``d3`` a ratio of two vanishing numbers. At a
    double root every 2x2 principal minor vanishes; there (``double``) the root function is ``d2``,
    which for the symmetric ``a I + b J`` of a body diagonal is ``(a - b)(a + b)/a``: smooth, with
    a simple root along directions that keep the symmetry.

    ``H_a``, the harmonic mean of two edges linear in ``t`` (``x' = k - e``), has a closed series:
    ``2xy/s``, ``2(x' y^2 + y' x^2)/s^2``, ``-2(x' y - y' x)^2/s^3`` with ``s = x + y``; a lone
    edge's is ``2x``, ``2x'``, ``0``. The cross coefficients do not move with ``t``.
    """
    x, y = _halves(own, other, node, t, floor)
    ok = _above(x, y, ghosts)
    one, zero = xp.ones_like(t), xp.zeros_like(t)
    h = []
    for a in range(3):
        dx, dy = node[a] - own[a], node[a] - other[a]
        q = 1 / xp.where(ok, x[a] + y[a], one)
        bend = dx * y[a] - dy * x[a]
        pair = (2 * x[a] * y[a] * q, 2 * (dx * y[a] * y[a] + dy * x[a] * x[a]) * q * q,
                -2 * bend * bend * q * q * q)
        h.append([xp.where(ghosts[a], u, v) for u, v in zip((2 * x[a], 2 * dx, zero), pair)])
    m1, m2, m3 = h
    c12, c13, c23 = off
    m1 = [xp.where(m1[0] != 0, m1[0], one)] + m1[1:]
    reciprocal = _over([one, zero, zero], m1)
    d2 = [u - c12 * c12 * v for u, v in zip(m2, reciprocal)]
    lower = [c23 - c12 * c13 * reciprocal[0]] + [-c12 * c13 * v for v in reciprocal[1:]]
    safe = [xp.where(double | (d2[0] == 0), one, d2[0])] + d2[1:]
    d3 = [u - c13 * c13 * v - w
          for u, v, w in zip(m3, reciprocal, _over(_times(lower, lower), safe))]
    return [xp.where(double, u, v) for u, v in zip(d2, d3)]


def _choice(xp, h, off):
    """A star's axis order and whether its root is double, from its test matrix at ``t*``."""
    order = _order(xp, _lowest(xp, h, off))
    first, middle, _ = order
    m1, m2 = _pick(xp, h, first), _pick(xp, h, middle)
    c12 = _pick(xp, off, first + middle - 1)
    d2 = m2 - c12 * c12 / xp.where(m1 != 0, m1, xp.ones_like(m1))
    scale = xp.abs(h[0]) + xp.abs(h[1]) + xp.abs(h[2])
    return order, xp.abs(d2) < TAU * scale


def _read(rt, arrays):
    """The inputs as host NumPy arrays, or ``None`` under a transform that carries no value."""
    try:
        out = [np.asarray(rt.numpy(rt.constant(x)), dtype=float) for x in arrays]
    except Exception:                       # noqa: BLE001 - a jit or batch tracer
        return None
    if any(o.shape != tuple(x.shape) for o, x in zip(out, arrays)):
        return None
    return out


def lift(field, diagonal, cross, node, inverse, rt, theta=THETA):
    """The lift of one term's region: see the module docstring.

    ``diagonal`` ``(3, nz, ny, nx)`` the edge inverse-material diagonal (``be / ratio``),
    ``cross`` the term's cross coefficients, ``node`` ``(3, nz, ny, nx)`` the node tensor's
    diagonal and ``inverse`` ``(nz, ny, nx)`` its ``1/<eps>``, both from
    ``constitutive.lower(nodes=...)``; ``field`` ``'E'`` or ``'H'``.
    """
    if not 0 < theta < 1:
        raise ValueError(f'the lift floor THETA={theta!r} must lie in (0, 1): at 1 or above the '
                         'node tensor itself can sit on the floor and no t reaches it')
    shape = tuple(cross.shape[1:])
    host = _read(rt, (diagonal, cross, node, inverse))
    if host is None:
        return _traced(field, diagonal, cross, node, inverse, rt, theta, shape)
    hd, hc, hk, hi = (np.broadcast_to(a, (3,) + shape if a.ndim == 4 else shape) for a in host)
    ghosts = [np.broadcast_to(_ghost(shape, a, field), shape) for a in range(3)]
    own, other = _star(np, field, hd, hk, ghosts)
    floor = _floor(np, theta, hi, own, other, ghosts)
    zero = np.zeros(shape)
    fire = ~_test(np, own, other, ghosts, hk, hc, zero, floor)[0]
    stars = int(np.count_nonzero(np.any(hc != 0, axis=0)))
    still = ~fire
    certified = bool(np.all(_test(np, [u[still] for u in own], [v[still] for v in other],
                                  [g[still] for g in ghosts], [k[still] for k in hk],
                                  [c[still] for c in hc], zero[still], 0.0)[0]))
    if not fire.any():
        return Lifted(None, certified, 0, stars, 0.0)
    where = np.nonzero(fire)
    sub = lambda arrays: [np.broadcast_to(u, shape)[where] for u in arrays]
    so, sv, sg, sk, sc = sub(own), sub(other), sub(ghosts), sub(hk), sub(hc)
    sf = floor[where]
    t = _bisect(np, _loop, so, sv, sg, sk, sc, sf)
    certified = certified and bool(np.all(_test(np, so, sv, sg, sk, sc, t, 0.0)[0]))
    order, double = _choice(np, _test(np, so, sv, sg, sk, sc, t, sf)[1], sc)
    full_t, full_double = np.zeros(shape), np.zeros(shape, dtype=bool)
    full_order = [np.full(shape, float(a)) for a in range(3)]
    full_t[where], full_double[where] = t, double
    for a in range(3):
        full_order[a][where] = order[a]
    delta = _assemble(field, diagonal, cross, node, inverse, rt, theta, shape, rt.array(fire),
                      rt.array(full_t), [rt.array(o) for o in full_order],
                      rt.array(full_double))
    return Lifted(delta, certified, int(t.size), stars, float(t.max()))


def _traced(field, diagonal, cross, node, inverse, rt, theta, shape):
    """:func:`lift` on every star in the runtime's arrays, for a transform that has no value."""
    xp = rt.xp
    ghosts = [rt.array(_ghost(shape, a, field)) for a in range(3)]
    fd, fc, fk, fi = (rt.constant(x) for x in (diagonal, cross, node, inverse))
    fd = xp.broadcast_to(fd, (3,) + shape)
    fk = xp.broadcast_to(fk, (3,) + shape)
    own, other = _star(xp, field, fd, fk, ghosts)
    floor = _floor(xp, theta, fi, own, other, ghosts)
    zero = xp.zeros_like(floor)
    fire = ~_test(xp, own, other, ghosts, fk, fc, zero, floor)[0]
    t = _bisect(xp, rt.scan, own, other, ghosts, fk, fc, floor)
    t = xp.where(fire, t, zero)
    certified = xp.all(_test(xp, own, other, ghosts, fk, fc, t, 0.0)[0])
    order, double = _choice(xp, _test(xp, own, other, ghosts, fk, fc, t, floor)[1], fc)
    delta = _assemble(field, diagonal, cross, node, inverse, rt, theta, shape, fire, t,
                      [rt.constant(o) for o in order], rt.constant(double))
    return Lifted(delta, certified, None, None, None)


def _assemble(field, diagonal, cross, node, inverse, rt, theta, shape, fire, t, order, double):
    """``(1/2) sum_p t_p (k_p - e)`` per edge, on the live inputs, with ``t*``'s derivatives.

    ``fire``, ``t``, ``order`` (three arrays) and ``double`` are ``(nz, ny, nx)`` runtime arrays
    with no gradient (:func:`_choice`). The star's root function at ``t*`` is a series in the
    move ``s``, ``F + F_1 s + F_2 s^2`` (:func:`_root`), on the live inputs; ``g = F - sg F`` is
    zero in value and the inputs' pull in derivative, and the root moves by the series'
    reversion, ``s = -g/F_1 - (F_2/F_1^3) g^2 + O(g^3)``. At ``g = 0`` a derivative of order ``k``
    of ``c g^2`` reads ``c``'s derivatives to order ``k - 2`` only, so the second term may carry
    ``F_2/F_1`` as a constant, ``-sg(F_2/F_1) (g/F_1)^2``: the first and second derivatives are
    exact and the third is not. One evaluation at ``t*``, elementwise: two Newton steps from the
    root were exact too, but the second nests the first, and ``jax.jit`` of their Hessian ran 25 s
    against 41 ms here (16^3 cavity, measured 2026-09-27). The value is ``t* + (t - sg t)``,
    ``t*`` to the bit whatever a compiler fuses. The unselected branch divides by one, so it
    carries no NaN into a derivative.
    """
    xp = rt.xp
    step = 1 if field == 'E' else -1
    ghosts = [rt.array(_ghost(shape, a, field)) for a in range(3)]
    diagonal = xp.broadcast_to(diagonal, (3,) + shape)
    node = xp.broadcast_to(node, (3,) + shape)
    own, other = _star(xp, field, diagonal, node, ghosts)
    floor = _floor(xp, theta, inverse, own, other, ghosts)
    first, middle, last = order
    ordered = [[_pick(xp, arrays, i) for i in order] for arrays in (own, other, ghosts, node)]
    off = [_pick(xp, cross, i + j - 1) for i, j in ((first, middle), (first, last), (middle, last))]
    f = _root(xp, *ordered, off, t, floor, double)
    active = fire & (rt.constant(f[1]) != 0)
    slope = xp.where(active, f[1], xp.ones_like(t))
    pull = (f[0] - rt.constant(f[0])) / slope
    moved = t - xp.where(active, pull + rt.constant(f[2] / slope) * pull * pull, xp.zeros_like(t))
    t = t + (moved - rt.constant(moved))
    delta = []
    for a in range(3):
        mine = t * (node[a] - own[a]) / 2
        theirs = xp.where(ghosts[a], xp.zeros_like(mine), t * (node[a] - other[a]) / 2)
        delta.append(mine + _shift(xp, theirs, -step, 2 - a))
    return xp.stack(delta)
