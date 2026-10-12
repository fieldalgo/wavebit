"""Freeze a 50-digit oracle for the star lift into ``star_lift_oracle.npz``: the lifted diagonal
change and its first and second directional derivatives on four small regions.

Run once, by hand, where mpmath is installed (wavebit does not depend on it);
``tests/test_fd_star_lift.py`` then compares the lift's nested forward derivatives with the
committed numbers on every backend.

    python tests/references/gen_star_lift_oracle.py

Rerun it only to add a fixture, never to make a failing test pass.

**What it computes, and from what.** It is written from the definitions in
``src/wavebit/fd/_lower/stars.py``'s docstring and imports nothing from wavebit; ``THETA`` is restated.
It shares none of the module's arithmetic: the module tests a 3x3 matrix of harmonic means in
float64, finds ``t*`` by 52 halvings and differentiates by Newton steps on one ``L D L^T``
pivot; this builds each star's explicit matrix (up to 6x6), finds ``t*`` by 150 halvings at 50
digits, certifies it with an independent eigensolver and differentiates by central differences.
Per node ``p`` of the ``(nz, ny, nx)`` arrays:

* the star's edges: for each axis ``a`` (component ``a``, array axis ``2 - a``) the own-index
  edge ``p`` and the other one, ``p - e_a`` for E and ``p + e_a`` for H, kept only if inside the
  region (a ghost edge outside it is dropped: a principal submatrix);
* the explicit star at ``t``: diagonal ``(e_q + t (k_{p,a} - e_q)) / 2`` for edge ``q`` on axis
  ``a``, off-diagonal ``c_ab(p) / 4`` between every edge on axis ``a`` and every edge on axis
  ``b != a``, and 0 between the two edges of one axis;
* the floor ``f_p = THETA * max(min(1/<eps>(p), e_q over the star's edges), 0)``;
* ``t*_p`` the least ``t`` in ``[0, 1]`` with ``star(t) - (f_p / 2) I`` positive definite (``0``
  if it already is at ``t = 0``);
* ``delta[a][q] = (1/2) sum_{p : q in star(p)} t*_p (k_{p,a} - e_q)``.

**Its error.** ``t*`` is bisected on an ``L D L^T`` positivity test at 50 digits to
``2**-150 = 7e-46`` and certified per moved star with ``mpmath.eigsy``: the smallest eigenvalue
at ``t*`` must lie in ``[0, 1e-40]`` of the star's scale. Derivatives are central differences at
``h = 1e-10`` along ``inputs + s u``: the value noise ``~1e-45`` amplified by ``4 / h**2`` is
``~4e-25``, and the truncation, ``h**2 / 12`` of the fourth derivative, is measured as the
difference between the ``h`` and ``2h`` stencils and must be below ``1e-15``. The oracle refuses
a stencil point that changes which stars move or which candidate attains a floor's minimum (a
kink, where no derivative exists).

**Its provenance.** Built 2026-09-27 by an independent reviewer of the lift's derivatives, who was
given the docstring and not the design, and checked there three ways: its values against
``stars.lift``'s to 1.4e-16, its first derivatives against the lift's (then already exact)
``jax.jvp`` to 1.7e-16, and on the symmetric double root its second derivative of ``t*`` against
the closed form ``t* = (c + f - e) / (k - e)`` (the test matrix is ``(H - c) I + c J`` there) to
every digit printed. The lift's eigenvector correction of that day read its second derivatives up
to 1.6 off on these fixtures; the oracle saw it.

**The fixtures.** ``random``: E, 3x3x3 random Kottke nodes, six stars move, three with a ghost
edge. ``ghost``: H, 2x2x2, one star with one ghost axis and one with three. ``symmetric``: E,
2x2x2, one star whose test matrix is ``a I + b J``, a double root, along two directions that keep
the symmetry -- ``t*`` has no derivative along one that breaks it. ``decoupled``: E, 2x2x2, one
star whose normal is ``1e-5`` off the xy plane, so the matrix is nearly block-diagonal: in the
fixed ``x, y, z`` order its middle pivot vanishes with the xy block, which is why the lift orders
each star's axes by its null vector -- and z is its stiffest axis, so the largest eigenvector will
not do in the null vector's place.
"""
import itertools
import os

import mpmath
import numpy as np

DPS = 50
BISECT = 150
STEP = '1e-10'
#: ``stars.THETA``, restated so the oracle stands alone.
THETA = 0.02
PAIRS = {(0, 1): 0, (0, 2): 1, (1, 2): 2}


def _edges(field, p, shape):
    """``(axis, index)`` of each edge of ``p``'s star inside the region."""
    out = []
    for a in range(3):
        for step in (0, -1 if field == 'E' else 1):
            q = list(p)
            q[2 - a] += step
            if all(0 <= v < n for v, n in zip(q, shape)):
                out.append((a, tuple(q)))
    return out


def _star(edges, e, k, c, p, t, f):
    """The explicit star at ``t`` minus ``(f/2) I``, as a list of rows of ``mpf``."""
    n = len(edges)
    m = [[mpmath.mpf(0)] * n for _ in range(n)]
    for i, (a, q) in enumerate(edges):
        u = e[(a,) + q]
        m[i][i] = (u + t * (k[(a,) + p] - u)) / 2 - f / 2
    for i, j in itertools.combinations(range(n), 2):
        a, b = edges[i][0], edges[j][0]
        if a != b:
            m[i][j] = m[j][i] = c[(PAIRS[(min(a, b), max(a, b))],) + p] / 4
    return m


def _positive(m):
    """All ``L D L^T`` pivots of the symmetric matrix ``m`` positive (early exit)."""
    n = len(m)
    low = [[None] * n for _ in range(n)]
    piv = [None] * n
    for j in range(n):
        d = m[j][j] - mpmath.fsum(low[j][r] ** 2 * piv[r] for r in range(j))
        if d <= 0:
            return False
        piv[j] = d
        for i in range(j + 1, n):
            low[i][j] = (m[i][j] - mpmath.fsum(low[i][r] * low[j][r] * piv[r]
                                               for r in range(j))) / d
    return True


def _floor(edges, e, inverse, p, theta):
    """The floor, and which candidate attains its minimum (``-1`` the node's ``1/<eps>``, else
    the edge's position in ``edges``): a change of candidate is a kink."""
    cand = [(inverse[p], -1)] + [(e[(a,) + q], i) for i, (a, q) in enumerate(edges)]
    cand.sort(key=lambda v: v[0])
    return theta * max(cand[0][0], mpmath.mpf(0)), cand[0][1]


def _least(edges, e, k, c, p, f):
    """``t*`` for one star, by bisection to ``2**-BISECT``, and whether it moved."""
    test = lambda t: _positive(_star(edges, e, k, c, p, t, f))
    if test(mpmath.mpf(0)):
        return mpmath.mpf(0), False
    lo, hi = mpmath.mpf(0), mpmath.mpf(1)
    if not test(hi):
        raise ValueError(f'star {p} is not positive definite at t = 1: no t* exists')
    for _ in range(BISECT):
        mid = (lo + hi) / 2
        if test(mid):
            hi = mid
        else:
            lo = mid
    return hi, True


def _to_mp(x):
    return np.vectorize(lambda v: mpmath.mpf(float(v)), otypes=[object])(np.asarray(x, float))


def _floats(a):
    return np.vectorize(float, otypes=[float])(a)


def _lift(field, inputs, theta, certify=False):
    """``(delta, t, moved, argmins, info)`` at 50 digits for mp object arrays ``inputs``."""
    diagonal, cross, node, inverse = inputs
    shape = inverse.shape
    theta = mpmath.mpf(theta)
    delta = np.empty(diagonal.shape, dtype=object)
    delta[...] = mpmath.mpf(0)
    t_all = np.empty(shape, dtype=object)
    t_all[...] = mpmath.mpf(0)
    moved, argmins, info = {}, {}, {}
    for p in np.ndindex(*shape):
        edges = _edges(field, p, shape)
        f, argmins[p] = _floor(edges, diagonal, inverse, p, theta)
        t, moved[p] = _least(edges, diagonal, node, cross, p, f)
        t_all[p] = t
        if not moved[p]:
            continue
        for a, q in edges:
            delta[(a,) + q] += t * (node[(a,) + p] - diagonal[(a,) + q]) / 2
        if certify:
            m = mpmath.matrix(_star(edges, diagonal, node, cross, p, t, f))
            lam = sorted(mpmath.eigsy(m, eigvals_only=True))
            scale = max(abs(m[i, j]) for i in range(m.rows) for j in range(m.cols))
            if not -mpmath.mpf(10) ** -44 * scale <= lam[0] < mpmath.mpf(10) ** -40 * scale:
                raise AssertionError(f'star {p}: smallest eigenvalue {lam[0]} at t*')
            info[p] = dict(t=float(t), ghosts=6 - len(edges),
                           gap=float((lam[1] - lam[0]) / scale))
    return delta, t_all, moved, argmins, info


def reference(field, inputs, directions, theta=THETA, step=STEP):
    """The lift's delta and its first and second derivatives along two directions.

    ``inputs`` ``(diagonal, cross, node, inverse)`` as ``stars.lift`` takes them, full arrays;
    ``directions`` two 4-tuples shaped like them. Returns float64 ``delta``, ``first`` ``[D_u,
    D_w]``, ``second`` ``{(0, 0): D_uu, (1, 1): D_ww, (0, 1): D_uw}``, their truncation
    estimates, and per moved star its ``t*``, ghost count and relative eigenvalue gap.
    """
    with mpmath.workdps(DPS):
        base = [_to_mp(x) for x in inputs]
        dirs = [[_to_mp(u) for u in d] for d in directions]
        cache = {}

        def at(coeffs, certify=False):
            if coeffs not in cache:
                point = [b.copy() for b in base]
                for s, d in zip(coeffs, dirs):
                    if s != 0:
                        point = [x + s * u for x, u in zip(point, d)]
                cache[coeffs] = _lift(field, point, theta, certify)
            return cache[coeffs]

        d0, _, moved0, arg0, info = at((0, 0), certify=True)

        def derivatives(h):
            first, second = [], {}
            for i in range(2):
                up, down = (at(tuple(s if j == i else 0 for j in range(2)))[0] for s in (h, -h))
                first.append((up - down) / (2 * h))
                second[(i, i)] = (up - 2 * d0 + down) / h ** 2
            pp, pm, mp_, mm = (at((a * h, b * h))[0]
                               for a, b in ((1, 1), (1, -1), (-1, 1), (-1, -1)))
            second[(0, 1)] = (pp - pm - mp_ + mm) / (4 * h ** 2)
            return first, second

        h = mpmath.mpf(step)
        first, second = derivatives(h)
        first2, second2 = derivatives(2 * h)
        for key, (_, _, moved, arg, _) in cache.items():
            if moved != moved0:
                raise ValueError(f'the stencil point {key} changes which stars move: a kink')
            if any(arg[p] != arg0[p] for p in arg0 if moved0[p]):
                raise ValueError(f'the stencil point {key} changes a floor minimum: a kink')
        worst = lambda a, b: float(max(abs(v) for v in (a - b).ravel()))
        return dict(delta=_floats(d0), first=[_floats(x) for x in first],
                    second={k: _floats(v) for k, v in second.items()},
                    error_first=max(worst(a, b) for a, b in zip(first, first2)),
                    error_second=max(worst(second[k], second2[k]) for k in second),
                    stars=info)


def _kottke(rng, shape, contrast):
    """Random node Kottke tensors ``inv I + s n n^T`` and edge diagonals."""
    normal = rng.normal(size=(3,) + shape)
    normal /= np.linalg.norm(normal, axis=0)
    inverse = rng.uniform(.05, 1., size=shape)
    strength = rng.uniform(0., contrast, size=shape)
    node = inverse + strength * normal ** 2
    cross = np.stack([strength * normal[i] * normal[j] for i, j in PAIRS])
    diagonal = rng.uniform(.02, 1., size=(3,) + shape)
    return diagonal, cross, node, inverse


def _directions(rng, inputs, scale=.1):
    return [tuple(scale * rng.normal(size=x.shape) for x in inputs) for _ in range(2)]


def random_fixture():
    """E, 3x3x3 random Kottke nodes and edges: interior and ghost stars move."""
    rng = np.random.default_rng(11)
    inputs = _kottke(rng, (3, 3, 3), 1.5)
    return 'E', inputs, _directions(rng, inputs)


def ghost_fixture():
    """H, 2x2x2: cross coefficients only at node (0,0,1), whose x-edge is a ghost, and at node
    (1,1,1), where all three are; both stars move."""
    rng = np.random.default_rng(34)
    diagonal, cross, node, inverse = _kottke(rng, (2, 2, 2), 3.)
    diagonal = diagonal * .3
    keep = np.zeros((2, 2, 2), bool)
    keep[0, 0, 1] = keep[1, 1, 1] = True
    cross = np.where(keep, cross, 0.)
    inputs = (diagonal, cross, node, inverse)
    return 'H', inputs, _directions(rng, inputs)


def symmetric_fixture():
    """E, 2x2x2: node (1,1,1) with every edge 0.1 and its normal along (1,1,1) between eps 1 and
    100, so its test matrix is ``a I + b J``: a double root. Both directions keep the symmetry
    (equal on the star's six edges, on the three k's, on the three c's)."""
    shape = (2, 2, 2)
    arith, harm = (1 + 100) / 2, (1 + 1 / 100) / 2
    inv, contrast = 1 / arith, harm - 1 / arith
    diagonal = np.full((3,) + shape, .1)
    cross = np.zeros((3,) + shape)
    cross[:, 1, 1, 1] = contrast / 3
    node = np.full((3,) + shape, inv + contrast / 3)
    inverse = np.full(shape, inv)
    u = (np.full((3,) + shape, .3), np.zeros((3,) + shape), np.zeros((3,) + shape),
         np.full(shape, .05))
    u[1][:, 1, 1, 1] = .2
    u[2][:, 1, 1, 1] = -.1
    w = (np.zeros((3,) + shape), np.zeros((3,) + shape), np.zeros((3,) + shape),
         np.zeros(shape))
    w[1][:, 1, 1, 1] = .5
    w[2][:, 1, 1, 1] = .2
    w[0][0][1, 1, 1] = w[0][0][1, 1, 0] = -.15
    w[0][1][1, 1, 1] = w[0][1][1, 0, 1] = -.15
    w[0][2][1, 1, 1] = w[0][2][0, 1, 1] = -.15
    return 'E', (diagonal, cross, node, inverse), [u, w]


def decoupled_fixture():
    """E, 2x2x2: node (1,1,1) with its normal ``1e-5`` off the xy plane between eps 1 and 100, so
    the xy block is what goes singular at ``t*`` and z is coupled to it by about ``1e-5``. Its z
    axis is also the stiffest (edges 1, ``k_z`` 2), so the largest eigenvector points along z: an
    order read off it rather than off the null vector puts z last, as the fixed order does."""
    shape = (2, 2, 2)
    rng = np.random.default_rng(7)
    normal = np.array([1., 1.3, 1e-5])
    normal /= np.linalg.norm(normal)
    arith, harm = (1 + 100) / 2, (1 + 1 / 100) / 2
    inv, strength = 1 / arith, harm - 1 / arith
    diagonal = rng.uniform(.08, .12, size=(3,) + shape)
    cross = np.zeros((3,) + shape)
    for (i, j), k in PAIRS.items():
        cross[k, 1, 1, 1] = strength * normal[i] * normal[j]
    diagonal[2] = 1.
    node = np.full((3,) + shape, inv)
    node[:, 1, 1, 1] += strength * normal ** 2
    node[2, 1, 1, 1] = 2.
    inverse = np.full(shape, inv)
    inputs = (diagonal, cross, node, inverse)
    return 'E', inputs, _directions(rng, inputs, scale=.01)


FIXTURES = {'random': random_fixture, 'ghost': ghost_fixture, 'symmetric': symmetric_fixture,
            'decoupled': decoupled_fixture}


def main():
    out = {}
    for name, make in FIXTURES.items():
        field, inputs, dirs = make()
        ref = reference(field, inputs, dirs)
        if max(ref['error_first'], ref['error_second']) >= 1e-15:
            raise AssertionError(f'{name}: truncation estimate {ref["error_first"]:.1e} / '
                                 f'{ref["error_second"]:.1e}')
        print(f'{name} ({field}): moved', ', '.join(
            f'{p} t* {s["t"]:.6f} ghosts {s["ghosts"]} gap {s["gap"]:.2g}'
            for p, s in sorted(ref['stars'].items())),
            f'| truncation {ref["error_first"]:.1e} / {ref["error_second"]:.1e}')
        out[f'{name}_field'] = np.array(field)
        out[f'{name}_delta'] = ref['delta']
        for i, x in enumerate(inputs):
            out[f'{name}_input{i}'] = x
        for a, tag in enumerate('uw'):
            out[f'{name}_first_{tag}'] = ref['first'][a]
            for i, x in enumerate(dirs[a]):
                out[f'{name}_dir{tag}{i}'] = x
        for (i, j), value in ref['second'].items():
            out[f'{name}_second_{"uw"[i]}{"uw"[j]}'] = value
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'star_lift_oracle.npz')
    np.savez(path, **out)
    print('wrote', path)


if __name__ == '__main__':
    main()
