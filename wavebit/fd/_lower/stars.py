"""Keep the smoothed inverse-material operator positive definite: the per-node star lift.

**The operator.** On one ``ConstitutiveTerm`` region the default rule's inverse-material operator
(inverse ``eps`` for E, inverse ``mu`` for H) is ``Xi = diag(e) + C``: ``e`` the edge diagonal
``coefficients.build`` paints (the voxel's own mix, ``coefficients._mix``) and ``C`` the node
cross coefficients ``constitutive.lower`` paints, applied by ``cross_action`` with ``c/4`` for
each pair of edges that share a node. It is the operator Werner, Bauer and Cary call wc07
(J. Comput. Phys. 247, 211, 2013, Eq. 11), after Oskooi, Kottke and Johnson (Opt. Lett. 34,
2778, 2009).

**Why it can fail.** ``Xi`` is positive definite only if the energy ``v . Xi v`` is positive for
every field, and nothing in the rule makes it so: at a high-contrast surface the node's cross
coefficient can outweigh the edges' diagonals. Then some field has negative energy and a run
grows without bound -- while the discrete energy, which leapfrog conserves exactly, stays flat
(``tests/test_fd_constitutive.py::test_an_unlifted_eps_100_sphere_grows``).

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
star fires far less: in the lift's study and the design review of 2026-09-27 the
eight blocks moved 1.3-3.3% of an ``eps = 12`` sphere's blocks, whose operator
was positive definite, and the error they added did not shrink with the cell (at ``eps = 25``
the static error settled near -0.2% from 12 to 24 cells across while the unmoved rule's went
+0.96% -> +0.36%); the star moves none of the ``eps = 12`` sphere's nodes. The star is ours.

**The lift.** Each star moves both edges of each axis toward the node tensor's diagonal,
``e -> e + t (k - e)`` for ``t`` in ``[0, 1]``, where ``k`` is the diagonal of the node's own Kottke
tensor ``K = (1/<eps>) I + (<1/eps> - 1/<eps>) n n^T`` (Kottke, Farjadpour and Johnson, Phys. Rev.
E 77, 036611, 2008), whose off-diagonal ``c`` already is. At ``t = 1`` both ``a``-edges read
``k_a``, the harmonic mean is ``k_a`` and the test matrix is ``K``: positive definite with smallest
eigenvalue at least ``1/<eps>`` (to round-off: a fill of ``-6e-14`` is kept by design,
``coefficients._mix``). ``S_p(t)`` is affine in ``t``, so its smallest eigenvalue is concave in
``t`` and the ``t`` where the star clears a floor form an interval ``[t*, 1]``. The lift takes
``t*``, the least move, and each edge's diagonal becomes the sum of its two halves:
``e + (1/2) sum_p t_p (k_p - e)`` over its two end nodes. That is Werner, Bauer and Cary's step-6
fallback (a failing piece replaced by the node tensor) made minimal and continuous.

**The floor** is ``S_p - (f/2) I`` positive definite, ``f = THETA * min(1/<eps>, the six edges'
diagonals)``: the star's physical scale. Two properties follow:

* a star with no cross coefficient has smallest eigenvalue ``min(e)/2 >= f/2`` and never moves,
  so a region with nothing below the floor lowers to the same bits it did before (the lift
  returns no change at all, and ``be`` is not touched);
* ``t*`` is continuous in every input: at ``t = 1`` the star clears the floor strictly, the
  crossing is from below, and a star that approaches the floor starts to move from ``t = 0``.
  A trigger at zero with a lift to the floor would jump (the understand workflow's correction).

**The certificate** is the construction, checked: every star at its ``t`` is tested on the
numbers the kernel will use, and ``Xi`` is then positive definite with smallest eigenvalue at
least the smallest, over edges, of the mean floor of the edge's two end nodes. The check does not
assume the construction worked -- it is what ``constitutive.validate`` refuses on and what
``constitutive.guard`` turns into ``stable = NaN`` under a transform. A node one layer past a
region's far end (the high end for E, the low end for H) holds half of that end's edges and no
coupling (``validate`` keeps its cross coefficients zero): a diagonal star that needs no test.

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
star in the runtime's own arrays. Its gradient is the implicit one: with ``M(t)`` the floored test
matrix, which is singular at ``t*``, ``v`` its (constant) null vector and ``q = v . M(t*) v``,
``t = t* - (q - sg q) / sg r`` with ``r = v . (dM/dt) v`` has the value ``t*`` exactly and the
derivative ``-dq/dp / r``. A determinant in place of ``q`` loses the derivative where the
smallest eigenvalue is double -- a sphere centred on a node or cell has such pieces on its body
diagonals, and the determinant's derivative was 44% off on the eight-block form (the design
review).
"""
import collections

import numpy as np

#: The floor as a fraction of the star's physical scale. PROVISIONAL (0.1, the understand
#: workflow's trial value): to be set by the block-lift study. It must lie in ``(0, 1)``.
THETA = 0.1

#: Bisection steps: ``2**-52`` of the unit interval, the spacing of float64 in ``[0.5, 1]``.
STEPS = 52

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
    """``THETA * min(1/<eps>, the star's edges)``; a ghost's placeholder is ``k_a >= 1/<eps>``."""
    low = inverse
    for a in range(3):
        low = xp.minimum(low, xp.minimum(own[a], xp.where(ghosts[a], own[a], other[a])))
    return theta * low


def _halves(own, other, node, t, floor):
    """Each axis's two edges at ``t``, floored: ``e + t (k - e) - f``."""
    return ([u + t * (k - u) - floor for u, k in zip(own, node)],
            [v + t * (k - v) - floor for v, k in zip(other, node)])


def _harmonic(xp, x, y, ghost, ok):
    one = xp.ones_like(x)
    return xp.where(ghost, 2 * x, 2 * x * y / xp.where(ok, x + y, one))


def _positive(xp, m, off):
    """The 3x3 matrix's ``L D L^T`` pivots are all positive (see the module): ``m_x``,
    ``m_y - c_xy^2 / m_x`` and ``m_z - c_xz^2 / m_x - (c_yz - c_xy c_xz / m_x)^2 / (m_y - c_xy^2 / m_x)``.
    """
    (mx, my, mz), (cxy, cxz, cyz) = m, off
    one = xp.ones_like(mx)
    ok = mx > 0
    first = xp.where(ok, mx, one)
    second = my - cxy * cxy / first
    ok = ok & (second > 0)
    lower = cyz - cxy * cxz / first
    return ok & (mz - cxz * cxz / first - lower * lower / xp.where(ok, second, one) > 0)


def _test(xp, own, other, ghosts, node, off, t, floor):
    """Whether ``S_p(t) - (f/2) I`` is positive definite, and its 3x3 test matrix's diagonal."""
    x, y = _halves(own, other, node, t, floor)
    ok = None
    for a in range(3):
        here = (x[a] > 0) & (ghosts[a] | (y[a] > 0))
        ok = here if ok is None else ok & here
    h = [_harmonic(xp, x[a], y[a], ghosts[a], ok) for a in range(3)]
    return ok & _positive(xp, h, off), h


def _bisect(xp, scan, own, other, ghosts, node, off, floor):
    """The least ``t`` in ``[0, 1]`` with the star above its floor, to ``2**-52``.

    ``hi`` keeps the property throughout: at ``t = 1`` the star is the node tensor's.
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


def _form(v, m, off):
    """``v . M v`` for the 3x3 matrix with diagonal ``m`` and off-diagonal ``off``."""
    (vx, vy, vz), (cxy, cxz, cyz) = v, off
    return (vx * vx * m[0] + vy * vy * m[1] + vz * vz * m[2]
            + 2 * (vx * vy * cxy + vx * vz * cxz + vy * vz * cyz))


def _rate(xp, own, other, ghosts, node, t, floor):
    """``dH_a/dt`` at ``t``: the harmonic mean's derivative, ``2 x'`` for a lone edge."""
    x, y = _halves(own, other, node, t, floor)
    out = []
    for a in range(3):
        dx, dy = node[a] - own[a], node[a] - other[a]
        total = xp.where(ghosts[a], xp.ones_like(x[a]), x[a] + y[a])
        pair = 2 * (dx * y[a] * y[a] + dy * x[a] * x[a]) / (total * total)
        out.append(xp.where(ghosts[a], 2 * dx, pair))
    return out


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
    vectors = _lowest(np, _test(np, so, sv, sg, sk, sc, t, sf)[1], sc)
    rate = _form(vectors, _rate(np, so, sv, sg, sk, t, sf), [np.zeros_like(c) for c in sc])
    full_t, full_r = np.zeros(shape), np.zeros(shape)
    full_v = [np.zeros(shape) for _ in range(3)]
    full_t[where], full_r[where] = t, rate
    for a in range(3):
        full_v[a][where] = vectors[a]
    delta = _assemble(field, diagonal, cross, node, inverse, rt, theta, shape, rt.array(fire),
                      rt.array(full_t), [rt.array(v) for v in full_v], rt.array(full_r))
    return Lifted(delta, certified, int(t.size), stars, float(t.max()))


def _traced(field, diagonal, cross, node, inverse, rt, theta, shape):
    """:func:`lift` on every star in the runtime's arrays, for a transform that has no value."""
    xp = rt.xp
    ghosts = [rt.array(np.broadcast_to(_ghost(shape, a, field), shape)) for a in range(3)]
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
    vectors = [rt.constant(v)
               for v in _lowest(xp, _test(xp, own, other, ghosts, fk, fc, t, floor)[1], fc)]
    rate = rt.constant(_form(vectors, _rate(xp, own, other, ghosts, fk, t, floor),
                             [xp.zeros_like(c) for c in fc]))
    delta = _assemble(field, diagonal, cross, node, inverse, rt, theta, shape, fire, t, vectors,
                      rate)
    return Lifted(delta, certified, None, None, None)


def _assemble(field, diagonal, cross, node, inverse, rt, theta, shape, fire, t, vectors, rate):
    """``(1/2) sum_p t_p (k_p - e)`` per edge, on the live inputs, with the implicit gradient.

    ``fire``, ``t``, ``vectors`` (three arrays) and ``rate`` are ``(nz, ny, nx)`` runtime arrays
    with no gradient; ``rate`` is ``r = v . (dM/dt) v``. The correction ``(q - sg q) / r`` is zero
    in value on every backend (``x - x`` is ``0`` for a finite ``x``), so the value is ``t*`` to the
    bit; the unselected branch divides by one, so it carries no NaN into a gradient.
    """
    xp = rt.xp
    step = 1 if field == 'E' else -1
    ghosts = [rt.array(np.broadcast_to(_ghost(shape, a, field), shape)) for a in range(3)]
    diagonal = xp.broadcast_to(diagonal, (3,) + shape)
    node = xp.broadcast_to(node, (3,) + shape)
    own, other = _star(xp, field, diagonal, node, ghosts)
    floor = _floor(xp, theta, inverse, own, other, ghosts)
    _, h = _test(xp, own, other, ghosts, node, cross, t, floor)
    q = _form(vectors, h, cross)
    active = fire & (rate > 0)
    t = t - xp.where(active, (q - rt.constant(q)) / xp.where(active, rate, xp.ones_like(rate)),
                     xp.zeros_like(q))
    delta = []
    for a in range(3):
        mine = t * (node[a] - own[a]) / 2
        theirs = xp.where(ghosts[a], xp.zeros_like(mine), t * (node[a] - other[a]) / 2)
        delta.append(mine + _shift(xp, theirs, -step, 2 - a))
    return xp.stack(delta)
