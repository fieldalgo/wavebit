"""Keep the smoothed inverse-material operator positive definite: the per-block lift.

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

**The blocks** (Werner, Bauer and Cary, Sec. 4). ``Xi`` is an average of 3x3 blocks, one per node
``p`` and orientation ``s`` in ``{0, 1}^3``: the three edges that touch ``p``, one along each
axis, with ``s_a`` choosing which of the two ``a``-edges (``0`` the one whose array index is
``p``'s own, ``1`` the other: ``p - e_a`` for E, whose node is an edge's low end, and ``p + e_a``
for H, whose cell centre is a dual edge's high end). A block holds its three edges' diagonals
and ``p``'s three cross coefficients::

    Xi = (1/8) sum_{p, s} P_{p,s}^T B_{p,s} P_{p,s}

An edge is in eight blocks (two end nodes, four orientations at each), so its diagonal is
``(1/8) 8 e = e``; a pair of edges that meet at ``p`` is in two, so its entry is
``(1/8) 2 c = c/4`` -- ``cross_action``'s. The nodes one layer past a region's far end (the high
end for E, the low end for H) hold the other four blocks of that end's edges; ``validate``
keeps their cross coefficients zero, so those blocks are ``diag(e)`` and never move. **If
every block is positive definite, so is** ``Xi``: ``v . Xi v = (1/8) sum (P v) . B (P v) > 0``.
An edge's eight blocks need not agree on its diagonal, only average to it -- which is what makes
a local repair possible.

**What a block test sees.** It is stronger than the operator's positivity: an ``eps = 12``
sphere or turned box whose ``Xi`` is positive definite still has blocks that are not (the
design review, 2026-09-27: 154 of 200 moved blocks of an off-node ``eps = 12`` sphere fail at
``t = 0`` with no floor at all). The lift moves those too. Redistributing an edge's diagonal
unequally among its eight blocks before lifting would move fewer; it is a coupled problem over
the whole region and is not done.

**The lift.** Each block moves its diagonal toward the node tensor's,
``B(t) = offdiag(c) + diag(e + t (k - e))`` for ``t`` in ``[0, 1]``, where ``k`` is the diagonal of
the node's own Kottke tensor ``K = (1/<eps>) I + (<1/eps> - 1/<eps>) n n^T`` (Kottke, Farjadpour
and Johnson, Phys. Rev. E 77, 036611, 2008), whose off-diagonal ``c`` already is. ``B(1) = K``
is positive definite, with smallest eigenvalue at least ``1/<eps>`` (to round-off: a fill of
``-6e-14`` is kept by design, ``coefficients._mix``); ``B(t)`` is affine in ``t``, so its smallest
eigenvalue is concave in ``t``, and the set of ``t`` where it clears a floor ``f < 1/<eps>`` is
an interval ``[t*, 1]``. The lift takes ``t*``, the least move that keeps the block above the
floor, and the edge's diagonal becomes the average of its eight blocks' entries:
``e + (1/8) sum_b t_b (k_b - e)``. That is Werner, Bauer and Cary's step-6 fallback (a failing
block replaced by the node tensor) made minimal and continuous; the continuous form is ours.

**The floor** is ``THETA * min(1/<eps>, e_x, e_y, e_z)`` -- the node's own ``1/<eps>`` and the
block's three edge diagonals, the physical scale of that block. Two properties follow:

* a block with no cross coefficient has smallest eigenvalue ``min(e) >= f`` and never moves, so
  a region with nothing below the floor lowers to the same bits it did before (the lift returns
  no change at all, and ``be`` is not touched);
* ``t*`` is continuous in every input: at ``t = 1`` the block clears the floor strictly, the
  crossing is from below, and a block that approaches the floor starts to move from ``t = 0``.
  A trigger at zero with a lift to the floor would jump (the understand workflow's correction).

**The certificate** is the construction, checked: every block at its ``t`` is tested positive
definite on the numbers the kernel will use, and the operator is then positive definite with
smallest eigenvalue at least ``min over edges q of (1/8) (sum_b f_b + (8 - n_q) e_q)``, ``n_q``
the blocks of ``q`` inside the region. The check does not assume the construction worked -- it
is what ``constitutive.validate`` refuses on and what ``constitutive.guard`` turns into
``stable = NaN`` under a transform.

**The test** is the sign of the three pivots of an ``L D L^T`` (Cholesky) factorisation of
``M = B(t) - f I``, elementwise on every backend. Two spellings with the leading principal minors
were measured and dropped. Unscaled, the minors misjudged 331 blocks at ``eps = 1e10`` (the design
review); the pivots, which a diagonal scaling does not change, certify every contrast tried up
to ``1e12``. Scaled, the minors were exact but for a double root, where the determinant touches
zero like ``(t - t*)^2``: its sign resolved ``t*`` only to about ``1e-8``, and a finite-difference
gradient on a cell-centred sphere scattered by 6% at a step of ``1e-6``. A pivot crosses zero
linearly at a double root as at a simple one.

**Ghost rows.** ``cross_action`` extends by zero past a region's end, which drops the edges
outside it: the region's operator is a principal submatrix, and a principal submatrix of a
positive definite matrix is positive definite. A block whose ``s_a = 1`` edge is outside keeps
the ``a`` row as ``k_a`` with no off-diagonal, which never constrains ``t``.

**Values and traces.** ``t*`` is found by bisection, on the host where the inputs can be read
(NumPy, a concrete or ``jax.grad`` JAX array, a Torch tensor under autograd) and only on the
blocks below the floor; under ``jax.jit`` and the other transforms that carry no value, on every
block of all eight orientations at once in the runtime's own arrays. Its gradient is the
implicit one, through the smallest eigenvalue: with ``v`` the (constant) eigenvector of
``B(t*)`` for it and ``q = v . (B(t*) - f I) v``, which is zero, ``t = t* - (q - sg q) / sg r`` with
``r = v . diag(k - e) v`` has the value ``t*`` exactly and the derivative ``-dq/dp / r``. ``r > 0``
by concavity. A determinant in place of ``q`` loses the derivative where the smallest eigenvalue
is double -- a sphere centred on a node or cell has such blocks on its body diagonals, and the
determinant's derivative was 44% off on them (the design review).
"""
import collections
import itertools

import numpy as np

#: The floor as a fraction of the block's physical scale. PROVISIONAL (0.1, the understand
#: workflow's trial value): to be set by the block-guard study. It must lie in ``(0, 1)``.
THETA = 0.1

#: Bisection steps: ``2**-52`` of the unit interval, the spacing of float64 in ``[0.5, 1]``.
STEPS = 52

#: ``(s_x, s_y, s_z)``: ``0`` the ``a``-edge with the node's own index, ``1`` the other.
ORIENTATIONS = tuple(itertools.product((0, 1), repeat=3))

#: The cross coefficients' order in a term's array: xy, xz, yz (``cross_action``'s).
PAIRS = ((0, 1), (0, 2), (1, 2))

#: What :func:`lift` returns. ``delta`` the three components' change to the inverse-material
#: diagonal over the region, or ``None`` when no block moves; ``certified`` whether every block
#: at its ``t`` is positive definite (a Python bool on the host, a runtime bool under a
#: transform); ``moved``, ``blocks`` and ``largest`` the blocks lifted, the blocks with a cross
#: coefficient, and the largest ``t`` -- ``None`` under a transform, which has no value to count.
Lifted = collections.namedtuple('Lifted', 'delta certified moved blocks largest')


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
    """Where the ``s_a = 1`` edge lies outside the region: the low end for E, the high for H."""
    mask = np.zeros(shape[2 - a], dtype=bool)
    mask[0 if field == 'E' else -1] = True
    return mask.reshape(tuple(-1 if axis == 2 - a else 1 for axis in range(3)))


def _orientation(xp, field, s, diagonal, node, cross, ghosts):
    """One orientation's blocks over the region: the diagonal at ``t = 0`` and the off-diagonal."""
    step = 1 if field == 'E' else -1
    start = []
    for a in range(3):
        if s[a]:
            start.append(xp.where(ghosts[a], node[a], _shift(xp, diagonal[a], step, 2 - a)))
        else:
            start.append(diagonal[a])
    off = []
    for k, (i, j) in enumerate(PAIRS):
        value = cross[k]
        for a in (i, j):
            if s[a]:
                value = xp.where(ghosts[a], xp.zeros_like(value), value)
        off.append(value)
    return start, off


def _stacked(xp, field, diagonal, node, cross, ghosts, shape):
    """All eight orientations along a new leading axis: ``start``, ``off`` as lists of three."""
    parts = [_orientation(xp, field, s, diagonal, node, cross, ghosts) for s in ORIENTATIONS]
    start = [xp.stack([xp.broadcast_to(p[0][a], shape) for p in parts]) for a in range(3)]
    off = [xp.stack([xp.broadcast_to(p[1][k], shape) for p in parts]) for k in range(3)]
    return start, off


def _floor(xp, theta, inverse, start):
    return theta * xp.minimum(xp.minimum(inverse, start[0]), xp.minimum(start[1], start[2]))


def _positive(xp, m, off):
    """The block's ``L D L^T`` pivots are all positive (see the module): ``m_x``,
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


def _moved(start, node, t, floor):
    """``B(t) - f I``'s diagonal."""
    return [u + t * (v - u) - floor for u, v in zip(start, node)]


def _bisect(xp, scan, start, node, off, floor):
    """The least ``t`` in ``[0, 1]`` with ``B(t) - f I`` positive definite, to ``2**-52``.

    ``hi`` keeps the property throughout: ``B(1) - f I`` has it by construction.
    """
    def body(carry, n):
        lo, hi = carry
        mid = 0.5 * (lo + hi)
        ok = _positive(xp, _moved(start, node, mid, floor), off)
        return xp.where(ok, lo, mid), xp.where(ok, mid, hi)
    _, hi = scan(body, (xp.zeros_like(floor), xp.ones_like(floor)), STEPS)
    return hi


def _loop(body, carry, steps):
    for n in range(steps):
        carry = body(carry, n)
    return carry


def _lowest(xp, m, off):
    """The eigenvector of the smallest eigenvalue of each block: three arrays."""
    (mx, my, mz), (cxy, cxz, cyz) = m, off
    matrix = xp.stack([xp.stack([mx, cxy, cxz], axis=-1), xp.stack([cxy, my, cyz], axis=-1),
                       xp.stack([cxz, cyz, mz], axis=-1)], axis=-2)
    _, vectors = xp.linalg.eigh(matrix)
    return [vectors[..., a, 0] for a in range(3)]


def _form(v, m, off):
    """``v . M v`` for the block with diagonal ``m`` and off-diagonal ``off``."""
    (vx, vy, vz), (cxy, cxz, cyz) = v, off
    return (vx * vx * m[0] + vy * vy * m[1] + vz * vz * m[2]
            + 2 * (vx * vy * cxy + vx * vz * cxz + vy * vz * cyz))


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
    ghosts = [_ghost(shape, a, field) for a in range(3)]
    start, off = _stacked(np, field, hd, hk, hc, ghosts, shape)
    node8 = [np.broadcast_to(k, (8,) + shape) for k in hk]
    floor = _floor(np, theta, hi, start)
    fire = ~_positive(np, [u - floor for u in start], off)
    blocks = 8 * int(np.count_nonzero(np.any(hc != 0, axis=0)))
    still = ~fire
    certified = bool(np.all(_positive(np, [u[still] for u in start], [c[still] for c in off])))
    if not fire.any():
        return Lifted(None, certified, 0, blocks, 0.0)
    where = np.nonzero(fire)
    s0, sk, so = ([u[where] for u in arrays] for arrays in (start, node8, off))
    sf = floor[where]
    t = _bisect(np, _loop, s0, sk, so, sf)
    certified = certified and bool(np.all(_positive(np, _moved(s0, sk, t, 0.0), so)))
    vectors = _lowest(np, _moved(s0, sk, t, sf), so)
    full_t, full_v = np.zeros(fire.shape), [np.zeros(fire.shape) for _ in range(3)]
    full_t[where] = t
    for a in range(3):
        full_v[a][where] = vectors[a]
    delta = _assemble(field, diagonal, cross, node, inverse, rt, theta, shape,
                      rt.array(fire), rt.array(full_t), [rt.array(v) for v in full_v])
    return Lifted(delta, certified, int(t.size), blocks, float(t.max()))


def _traced(field, diagonal, cross, node, inverse, rt, theta, shape):
    """:func:`lift` on every block in the runtime's arrays, for a transform that has no value."""
    xp = rt.xp
    ghosts = [rt.array(_ghost(shape, a, field)) for a in range(3)]
    fd, fc, fk, fi = (rt.constant(x) for x in (diagonal, cross, node, inverse))
    start, off = _stacked(xp, field, fd, fk, fc, ghosts, shape)
    node8 = [xp.broadcast_to(fk[a], (8,) + shape) for a in range(3)]
    floor = _floor(xp, theta, fi, start)
    fire = ~_positive(xp, [u - floor for u in start], off)
    t = _bisect(xp, rt.scan, start, node8, off, floor)
    t = xp.where(fire, t, xp.zeros_like(t))
    certified = xp.all(_positive(xp, _moved(start, node8, t, 0.0), off))
    vectors = [rt.constant(v) for v in _lowest(xp, _moved(start, node8, t, floor), off)]
    delta = _assemble(field, diagonal, cross, node, inverse, rt, theta, shape, fire, t, vectors)
    return Lifted(delta, certified, None, None, None)


def _assemble(field, diagonal, cross, node, inverse, rt, theta, shape, fire, t, vectors):
    """``(1/8) sum_b t_b (k_b - e)`` per edge, on the live inputs, with the implicit gradient.

    ``fire``, ``t`` and ``vectors`` are ``(8, nz, ny, nx)`` runtime arrays, in
    :data:`ORIENTATIONS` order, with no gradient. The correction ``(q - sg q) / r`` is zero in
    value on every backend (``x - x`` is ``0`` for a finite ``x``), so the value is ``t*`` to the
    bit; the unselected branch divides by one, so it carries no NaN into a gradient.
    """
    xp = rt.xp
    step = 1 if field == 'E' else -1
    ghosts = [rt.array(_ghost(shape, a, field)) for a in range(3)]
    diagonal = xp.broadcast_to(diagonal, (3,) + shape)
    node = xp.broadcast_to(node, (3,) + shape)
    start, off = _stacked(xp, field, diagonal, node, cross, ghosts, shape)
    node8 = [xp.broadcast_to(node[a], (8,) + shape) for a in range(3)]
    floor = _floor(xp, theta, inverse, start)
    q = _form(vectors, _moved(start, node8, t, floor), off)
    r = rt.constant(_form(vectors, [v - u for u, v in zip(start, node8)],
                          [xp.zeros_like(c) for c in off]))
    active = fire & (r > 0)
    t = t - xp.where(active, (q - rt.constant(q)) / xp.where(active, r, xp.ones_like(r)),
                     xp.zeros_like(q))
    delta = [xp.zeros_like(diagonal[a]) for a in range(3)]
    for i, s in enumerate(ORIENTATIONS):
        for a in range(3):
            share = t[i] * (node[a] - start[a][i]) / 8
            delta[a] = delta[a] + (_shift(xp, share, -step, 2 - a) if s[a] else share)
    return xp.stack(delta)
