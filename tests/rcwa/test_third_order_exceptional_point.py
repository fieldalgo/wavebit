"""A third-order exceptional point of a genuinely 2D layer: three modes, one eigenvector.

**The points.** The PA616 pillar cut to a fraction of the y period
(``test_2d_exceptional_point``'s genuinely 2D layer) is lossless, so ``Omega2`` is a product of
two Hermitian matrices and its spectrum is closed under conjugation: two real ``q**2`` meet at an
exceptional point (EP) along a curve in two real parameters, and three meet at isolated points,
where the three modes share one eigenvector (a Jordan chain of length three, EP3). With the
wavelength and the pillar's permittivity root-found together -- Newton on the centred power sums
of the three eigenvalues in one x-parity block, which are analytic at the point where the
eigenvalues are not -- the two cut pillars below hold at least fourteen such points between
539 and 749 nm (2026-11-13). A design that tunes two parameters toward a higher-order EP tunes
exactly this way.

**What went wrong.** ``eig`` splits the triple eigenvalue by about ``u**(1/3)`` (1e-6 .. 1e-5 of
its scale here), and its three columns meet at sines of the same order. Off the point by a
relative ``d`` in the wavelength the three move apart like ``d**(1/3)``. From about 1e-9 to 1e-7
the pair gate took two of the three as an EP pair, and the cluster gate left the ball of three
to the pair rebuild -- exact for a pair beside an independent root, but the third mode here is
the chain's own: its column lay within 8e-5 of the pair's span, the rebuilt basis stayed nearly
dependent, and the three were too far apart for the unresolved screen's ball. Against a Cauchy
oracle the 2D path was silently up to 9.9e-12 off (eager) and 8.6e-12 (``jax.jit``); closer it
warned and was up to 9.3e-3 off (SMM at the point).

**Now** a ball of three holding one gated pair is taken jointly where its third column lies
within ``_EP_SINE`` of the pair's span (``solve_2d._defective_clusters``). Every answer near the
three points is within 1e-12 of the oracle or warns; the warnings are now on answers within
7.8e-13. The first-order split, whose gate reads before its pair gate, already took the three
jointly (quiet answers within 3.9e-14).

What this file pins:

1. the points are third order;
2. near them the 2D path is exact or loud, eagerly, with the gate taking the three jointly in
   the band where it did not;
3. under ``jax.jit`` the values are exact or NaN.
"""

import functools
import warnings

import numpy as np
import pytest

from wavebit._common.backend import ExceptionalPointWarning
from wavebit.rcwa._engine import solve_2d
from tests.rcwa.test_2d_exceptional_point import _genuine_solver
from tests.rcwa.test_conical_cutoff import _np

# CI may run each test of this file in a different shard (tools/run_verification.py --shard).
pytestmark = pytest.mark.shard_unit(per='test')

#: Third-order points of the cut pillar: ``(fy, wavelength, pillar permittivity)`` at
#: ``theta = 0.3``, ``phi = pi/2``, truncation ``(5, 2)``. The centred power sums of the three
#: eigenvalues there are 3e-17 .. 8e-17 of their scale.
EP3 = {
    'fy 0.5, 620.55 nm': (0.5, 620.5536992825428, 10.720356916283205),
    'fy 0.8, 664.35 nm': (0.8, 664.3456199320473, 13.62736000232683),
    'fy 0.5, 645.23 nm': (0.5, 645.2344987811947, 6.759992785437672),
}
FIRST = 'fy 0.5, 620.55 nm'


def _structure(fy, eps, ny=20):
    st = np.ones((1, ny, 20))
    st[0, :int(round(fy * ny)), 6:14] = eps
    return st


def _solver(name, rel=0.0, theta=0.3, algo='ETM', backend='numpy'):
    fy, wl, eps = EP3[name]
    return _genuine_solver(fy, wl * (1 + rel), theta, algo, backend=backend,
                           structure=_structure(fy, eps))


def _amplitudes(name, rel, theta, algo):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        r = _solver(name, rel, theta, algo).solve(auto_detune=False)
    loud = any(issubclass(w.category, ExceptionalPointWarning) for w in rec)
    return np.concatenate([_np(r.r_xyz).ravel(), _np(r.t_xyz).ravel()]), loud


@functools.lru_cache(maxsize=None)
def _oracle(name, rel):
    """The amplitudes' mean over ``theta = 0.3 + 1e-4 exp(2 pi i k / 16)``: Cauchy's formula on
    a circle (``test_2d_exceptional_point._circle``). The amplitudes are analytic in ``theta``
    and the circle passes off the point, where the product construction is accurate. Measured:
    it moves by at most 5.9e-15 to radius 2e-4 with 24 points, or to SMM."""
    unit = np.exp(2j * np.pi * np.arange(16) / 16)
    return np.mean([_amplitudes(name, rel, 0.3 + 1e-4 * u, 'ETM')[0] for u in unit], axis=0)


def _operator(monkeypatch, name, rel=0.0):
    seen = []
    eig = solve_2d._eig_or_diagonal

    def spy(xp, A):
        seen.append(np.array(A))
        return eig(xp, A)
    with monkeypatch.context() as m:
        m.setattr(solve_2d, '_eig_or_diagonal', spy)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _solver(name, rel).solve(auto_detune=False)
    return seen[0]


@pytest.mark.parametrize('name', list(EP3))
def test_the_points_are_third_order(monkeypatch, name):
    """Three eigenvalues of ``Omega2`` whose centred power sums ``sum d**2``, ``sum d**3``
    (``d = e - mean``) vanish to rounding -- a triple root -- and whose ``eig`` columns are pairwise
    nearly parallel, so the three share one eigenvector: a Jordan chain of length three, not a
    pair beside a root (whose root keeps a column of its own) nor a degenerate triple."""
    A = _operator(monkeypatch, name)
    w, W = np.linalg.eig(A)
    d = np.abs(w[:, None] - w[None, :])
    np.fill_diagonal(d, np.inf)
    seed = int(np.argmin(np.sort(d, axis=1)[:, 1]))
    tri = np.argsort(np.abs(w - w[seed]))[:3]
    e = w[tri]
    scale = max(1.0, abs(e.mean()))
    c = e - e.mean()
    assert abs(np.sum(c ** 2)) / scale ** 2 < 1e-14
    assert abs(np.sum(c ** 3)) / scale ** 3 < 1e-14
    Wn = W[:, tri] / np.linalg.norm(W[:, tri], axis=0)
    for i in range(3):
        for j in range(i + 1, 3):
            assert np.sqrt(max(0.0, 1 - abs(np.vdot(Wn[:, i], Wn[:, j])) ** 2)) < 1e-3
    gaps = np.sort(np.abs(w - e.mean()))
    assert gaps[3] / scale > solve_2d._EP_JOINT_RADIUS     # alone in the widest ball (7.2e-3 ..)


_NEAR = [0.0, 1e-9, -1e-9, 1e-8, -1e-8]
_FAR = [1e-12, -1e-12, 1e-10, -1e-10, 1e-7, -1e-7]


@pytest.mark.parametrize('name, rel', [(FIRST, r) for r in _NEAR]
                         + [pytest.param(n, r, marks=pytest.mark.slow)
                            for n in EP3 for r in (_NEAR + _FAR if n != FIRST else _FAR)])
def test_near_a_third_order_point_the_answer_is_exact_or_loud(name, rel):
    """ETM and SMM against the Cauchy oracle, within 1e-12 or with an ``ExceptionalPointWarning``.
    Measured (all three points, offsets 0 .. +-1e-7): every quiet answer within 2.4e-14 and every
    loud one within 7.8e-13; before, 1.2e-12 .. 9.9e-12 off and quiet at +-1e-8 (fy 0.5) and
    up to 2.2e-12 at +-1e-7 (fy 0.8). Catches the gate's dependency test removed or narrowed."""
    ref = _oracle(name, rel)
    for algo in ('ETM', 'SMM'):
        got, loud = _amplitudes(name, rel, 0.3, algo)
        err = np.max(np.abs(got - ref))
        assert loud or err < 1e-12, f'{name} {algo} rel {rel:g}: {err:.1e}, quiet'


@pytest.mark.parametrize('rel', [1e-8, -1e-8])
def test_a_pair_and_the_chains_third_mode_are_one_cluster(monkeypatch, rel):
    """At these offsets the pair gate takes two of the three and the third sits 1.6e-4 (relative)
    away with its column 7.2e-5 from the pair's span. The cluster gate takes the three as one
    cluster with that pair inside it. ``test_exceptional_point_clusters
    .test_a_pair_beside_an_unpaired_root`` pins the other side: a pair beside an independent root
    stays with the pair rebuild."""
    calls = {}
    for fn in ('_defective_pairs', '_defective_clusters'):
        orig = getattr(solve_2d, fn)

        def spy(*a, _orig=orig, _fn=fn, **k):
            out = _orig(*a, **k)
            calls[_fn] = out
            return out
        monkeypatch.setattr(solve_2d, fn, spy)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _solver(FIRST, rel).solve(auto_detune=False)
    pairs, clusters = calls['_defective_pairs'], calls['_defective_clusters']
    assert pairs is not None and clusters is not None
    a, b, mask = (np.asarray(x) for x in pairs[:3])
    members, valid = np.asarray(clusters[0]), np.asarray(clusters[1])
    assert valid.sum() == 3
    taken = set(members[valid].tolist())
    assert any(m and {int(i), int(j)} <= taken for i, j, m in zip(a, b, mask))


def test_the_third_column_is_read_against_the_pair_s_span():
    """``_sine_to_span`` reads the span of the pair's two orthonormal columns, not ``v``: a column
    inside the span is at sine 0 however far from ``v``. Catches the ``x`` term dropped: a chain's
    third column leaves ``v`` at about the square root of its sine to the span (7.1e-3 against
    7.2e-5 at the edge of the band), which leaves a margin of 1.4 below ``_EP_SINE``."""
    rng = np.random.default_rng(3)
    Q, _ = np.linalg.qr(rng.normal(size=(8, 3)) + 1j * rng.normal(size=(8, 3)))
    v, x, y = Q.T
    cols = np.stack([x, (v + x) / np.sqrt(2), np.cos(1e-3) * v + np.sin(1e-3) * y], axis=1)
    s = solve_2d._sine_to_span(np, cols, v[:, None], x[:, None])
    assert s[0] < 1e-15 and s[1] < 1e-15
    assert abs(s[2] - np.sin(1e-3)) < 1e-15


@pytest.mark.parametrize('seed', [0, 1, 3])
def test_the_third_column_is_read_against_its_own_pair(seed):
    """A synthetic length-3 Jordan chain at 0.3 (couplings 0.05) beside a separate Jordan pair at
    1.1, turned by a random unitary: the pair gate takes one pair from each (seeds 1 and 3 list
    the chain's second), and the cluster gate takes the chain's three modes as one cluster,
    reading the third column against the pair in the same ball. Catches that pair looked up in
    the wrong slot."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(12, 12)) + 1j * rng.normal(size=(12, 12)))
    J = np.diag([0.3, 0.3, 0.3, 0.7, 0.9, 1.1, 1.1, 1.6, 1.9, 2.3, 2.8, 3.4]).astype(complex)
    J[0, 1] = J[1, 2] = J[5, 6] = 0.05
    A = Q @ J @ Q.conj().T
    w, W = np.linalg.eig(A)
    q = np.sqrt(w)
    pairs = solve_2d._defective_pairs(np, w, W, q, 4)
    assert int(np.asarray(pairs[2]).sum()) == 2
    clusters = solve_2d._defective_clusters(np, w, W, q, pairs, lambda: A, 2)
    assert clusters is not None
    taken = np.asarray(clusters[0])[np.asarray(clusters[1])]
    assert len(taken) == 3 and np.all(np.abs(w[taken] - 0.3) < 1e-4)


@pytest.mark.parametrize('seed', [0, 1])
def test_a_chain_beside_a_coincident_cluster(seed):
    """A synthetic length-3 Jordan chain at 0.3 beside two coincident Jordan pairs at 1.5 (a ball
    of four modes the pair gate does not see), turned by a random unitary: both balls are read,
    the chain's three are taken with their pair, and the four jointly. The third member is found
    by position on every ball, whatever the ball holds; found as the sum of a ball's unpaired
    lanes it indexed past the layer on the ball of four (an ``IndexError`` here, and out of range
    under ``torch.func.vmap``). Catches that lookup."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(14, 14)) + 1j * rng.normal(size=(14, 14)))
    J = np.diag([0.3, 0.3, 0.3, 0.7, 0.9, 1.5, 1.5, 1.5, 1.5, 2.0, 2.3, 2.8, 3.1, 3.4]).astype(complex)
    J[0, 1] = J[1, 2] = J[5, 6] = 0.05
    J[7, 8] = 0.04
    A = Q @ J @ Q.conj().T
    w, W = np.linalg.eig(A)
    q = np.sqrt(w)
    pairs = solve_2d._defective_pairs(np, w, W, q, 4)
    clusters = solve_2d._defective_clusters(np, w, W, q, pairs, lambda: A, 2)
    assert pairs is not None and clusters is not None
    taken = sorted(sorted(np.round(np.abs(w[np.asarray(m)[np.asarray(v)]]), 4).tolist())
                   for m, v in zip(clusters[0], clusters[1]))
    assert taken == [[0.3] * 3, [1.5] * 4]


@pytest.mark.slow
@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_traced_values_near_a_third_order_point_are_exact_or_nan(algo):
    """``jax.jit`` of one wavelength at the fy 0.8 point, relative offsets +-1e-7 and +-1e-8: each
    NaN (the joint check's verdict under a trace, where it would warn eagerly) or within 1e-12 of
    the oracle, and not all NaN. Which offsets the check takes depends on the CPU's arithmetic
    path: measured on six OpenBLAS kernels, every value within 1.6e-14, but NaN at +-1e-8 on
    Sandybridge and at -1e-8 on a CI runner; +-1e-7 finite on all seven. Main was 4.0e-13 ..
    1.4e-12 off at +-1e-7, quiet, and NaN at +-1e-8."""
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    from wavebit._common.backend import _enable_x64
    _enable_x64()
    name = 'fy 0.8, 664.35 nm'
    _, wl, _ = EP3[name]
    s = _solver(name, algo=algo, backend='jax')

    def one(w):
        r = s.solve(wavelength=w, auto_detune=False)
        return jnp.concatenate([r.r_xyz.ravel(), r.t_xyz.ravel()])
    jf = jax.jit(one)
    finite = 0
    for rel in (1e-7, -1e-7, 1e-8, -1e-8):
        got = np.asarray(jf(jnp.asarray(wl * (1 + rel))))
        if np.all(np.isnan(got)):
            continue
        assert np.all(np.isfinite(got)), f'{algo} rel {rel:g}: partly NaN'
        finite += 1
        err = np.max(np.abs(got - _oracle(name, rel)))
        assert err < 1e-12, f'{algo} rel {rel:g}: {err:.1e}, finite'
    assert finite
