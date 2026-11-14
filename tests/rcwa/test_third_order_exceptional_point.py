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

**Then** a ball of three holding one gated pair was taken jointly where its third column lies
within ``_EP_SINE`` of the pair's span (``solve_2d._defective_clusters``). Every answer near the
three points was within 1e-12 of the oracle or warned; the warnings were on answers within
7.8e-13 -- at the point the joint rebuild's two-level split cannot represent a chain of length
three. The first-order split, whose gate reads before its pair gate, already took the three
jointly (quiet answers within 3.9e-14, loud ones within 1.4e-11).

**Now** (2026-11-13) a slot whose two-level split fails the check tries three levels
(``solve_2d._chain_split``): the coupling reaches two levels up (``N**2 != 0``, ``N**3 = 0``),
the root takes its second-order term, the inverse and the exponential theirs. Eagerly every
answer near the three points is quiet and within 3.9e-14 of the oracle, on both formulations
and every backend, fields within 3.7e-14. Since 2026-11-14 the derivative rules take the chain
too (the generator's and the first-order split's Sylvester solves, the inverse's tangent, SMM's
field state), and eager derivatives there are within 6.8e-13 of Cauchy derivative oracles,
quiet. A trace reads no chain: NaN at the point.

What this file pins:

1. the points are third order;
2. near them the 2D path is exact or loud, eagerly, with the gate taking the three jointly in
   the band where it did not;
3. under ``jax.jit`` the values are exact or NaN;
4. eagerly, quiet and exact at and near the points, values and fields, both formulations; the
   three-level split and the exponential's two-step paths on their own;
5. eager derivatives there, amplitudes and fields, against Cauchy derivative oracles; each
   derivative rule's chain term on its own; a trace answers NaN.
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


# ----------------------------------------------- 4. the chain in three levels, eagerly

def _quiet(fn):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        out = fn()
    assert not any(issubclass(w.category, ExceptionalPointWarning) for w in rec), \
        [str(w.message)[:80] for w in rec]
    return out


def _formulated(name, rel, algo, backend, first_order, theta=0.3):
    s = _solver(name, rel, theta, algo, backend)
    if first_order:
        s.formulation = 'first_order'
    return s


_CHAIN_CASES = (
    [(FIRST, r, fo, 'numpy') for r in (0.0, 1e-9, -1e-9) for fo in (False, True)]
    + [pytest.param(n, r, fo, be, marks=pytest.mark.slow)
       for n in EP3 for r in _NEAR + _FAR for fo in (False, True) for be in ('numpy', 'jax', 'torch')
       if not (n == FIRST and r in (0.0, 1e-9, -1e-9) and be == 'numpy')])


@pytest.mark.parametrize('name, rel, first_order, backend', _CHAIN_CASES)
def test_at_a_third_order_point_the_answer_is_exact_and_quiet(name, rel, first_order, backend):
    """Eagerly, with no derivative taken, ETM and SMM within 1e-12 of the Cauchy oracle and no
    ``ExceptionalPointWarning``, at the point and off it, on both formulations. Measured: 2D
    within 2.4e-14 and first-order within 3.9e-14 at all three points (offsets 0 .. +-1e-7),
    NumPy, JAX and PyTorch alike; before the three levels the 2D path warned on 36 of the 66
    NumPy solves within 1e-7 (up to 7.8e-13 off at the point) and first-order warned at the
    point (up to 1.4e-11). Catches the three levels, the root's second-order term, the inverse's
    or the exponential's dropped."""
    if backend == 'jax':
        pytest.importorskip('jax')
        from wavebit._common.backend import _enable_x64
        _enable_x64()
    elif backend == 'torch':
        pytest.importorskip('torch')
    ref = _oracle(name, rel)
    for algo in ('ETM', 'SMM'):
        r = _quiet(lambda: _formulated(name, rel, algo, backend, first_order).solve(auto_detune=False))
        err = np.max(np.abs(np.concatenate([_np(r.r_xyz).ravel(), _np(r.t_xyz).ravel()]) - ref))
        assert err < 1e-12, f'{name} {algo} rel {rel:g} first_order={first_order}: {err:.1e}'


_FIELD_AT = ([-200., 0., 130.], [0., 200.], [40., 150., 260.])


def _field(name, rel, algo, first_order, theta=0.3):
    r = _formulated(name, rel, algo, 'numpy', first_order, theta).solve(auto_detune=False,
                                                                          answers='field')
    return _np(r.reconstruct_field(*(np.asarray(v) for v in _FIELD_AT)))


@pytest.mark.parametrize('name, rel', [(FIRST, 0.0)] + [pytest.param(n, r, marks=pytest.mark.slow)
                                                        for n in EP3 for r in (0.0, 1e-9)
                                                        if (n, r) != (FIRST, 0.0)])
def test_fields_at_a_third_order_point_are_exact_and_quiet(name, rel):
    """The six components at three x, two y and three z inside the layer, against the same
    Cauchy circle of the field (it is analytic in ``theta``; no conjugate is taken), ETM and SMM,
    both formulations: within 1e-12 of ``max|F|``, quiet. Measured 3.7e-14 (2D) and 1.7e-14
    (first order) at the fy 0.5, 620.55 nm point."""
    unit = np.exp(2j * np.pi * np.arange(16) / 16)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ref = np.mean([_field(name, rel, 'ETM', False, 0.3 + 1e-4 * u) for u in unit], axis=0)
    scale = np.max(np.abs(ref))
    for first_order in (False, True):
        for algo in ('ETM', 'SMM'):
            got = _quiet(lambda: _field(name, rel, algo, first_order))
            err = np.max(np.abs(got - ref)) / scale
            assert err < 1e-12, f'{algo} first_order={first_order}: {err:.1e}'


def test_the_chain_split_is_a_schur_form():
    """A synthetic Jordan chain of length three (couplings 0.05 and 0.04, eigenvalue split by
    rounding) in a random unitary basis, padded to four lanes: no two-level split passes, and the
    three-level one puts one member on each level -- a Schur form, ``cond(X) = 1`` -- with
    ``X^-1 T X`` coupling only upward, the padded lane untouched."""
    rng = np.random.default_rng(5)
    U, _ = np.linalg.qr(rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))
    J = np.array([[0.3, 0.05, 0.0], [0.0, 0.3, 0.04], [0.0, 0.0, 0.3]], complex)
    T = np.zeros((1, 4, 4), complex)
    T[0, :3, :3] = U @ J @ U.conj().T
    T[0, 3, 3] = 40.0
    valid = np.array([[True, True, True, False]])
    X, level, cond = solve_2d._chain_split(np, T, valid, 40.0)
    assert sorted(np.asarray(level)[0, :3].tolist()) == [0, 1, 2]
    assert abs(float(cond[0]) - 1.0) < 1e-12
    Mx = np.linalg.solve(X[0], T[0] @ X[0])
    lv = np.asarray(level)[0]
    up = (lv[:, None] < lv[None, :]) | np.eye(4, dtype=bool)
    up[3, :3] = up[:3, 3] = False
    up[3, 3] = True
    assert np.max(np.abs(np.where(up, 0, Mx))) < 1e-15
    assert np.allclose(X[0][3], [0, 0, 0, 1]) and np.allclose(X[0][:, 3], [0, 0, 0, 1])


def _taylor_expm(A, terms=40):
    out, term = np.eye(A.shape[0], dtype=complex), np.eye(A.shape[0], dtype=complex)
    for n in range(1, terms):
        term = term @ A / n
        out = out + term
    return out


@pytest.mark.parametrize('split', [0.0, 1e-6, 1e-3])
@pytest.mark.parametrize('scale', [-5.0, -40.0, 2.0j])
def test_the_exponential_takes_two_coupling_steps(split, scale):
    """``modal_gradient.exponential`` of a chain's coupled ``q`` (couplings 0.05, 0.04 and 0.03,
    the eigenvalue split by ``split``) beside two plain modes, against ``exp(s lambda)`` times a
    float64 Taylor series of the chain block's small remainder: within 1e-14 relative. The two
    plain modes' entries are their phases, and a coupling on disjoint rows and columns takes no
    two-step term (``_chain_paths`` is ``None``). Catches the two-step paths dropped or their
    series' degree shifted."""
    from wavebit.rcwa._engine import modal_gradient
    lam = 0.3 + 0.01j
    d = np.array([lam, lam + split, lam - 2 * split, 1.1, 1.7 + 0.2j])
    q = np.diag(d)
    q[0, 1], q[1, 2], q[0, 2] = 0.05, 0.04, 0.03
    E = modal_gradient.exponential(np, q, scale)
    block = np.exp(scale * lam) * _taylor_expm(scale * (q[:3, :3] - lam * np.eye(3)))
    assert np.max(np.abs(E[:3, :3] - block)) / np.max(np.abs(block)) < 1e-14
    assert np.allclose(np.diag(E)[3:], np.exp(scale * d[3:]), rtol=1e-15)
    pairs = np.diag(d)
    pairs[0, 1], pairs[3, 4] = 0.05, 0.02
    assert modal_gradient._chain_paths(np, pairs - np.diag(d), d, np.asarray(scale)) is None


def test_the_exponential_takes_a_spread_chain():
    """The two-step paths' recurrence (points more than 0.5 apart): a chain at 0.3, 1.4 and 2.9
    (couplings 0.5, 0.4 and 0.3) against the eigendecomposition of the well-separated block:
    within 1e-14 relative."""
    from wavebit.rcwa._engine import modal_gradient
    d = np.array([0.3, 1.4, 2.9, 1.1, 1.7], complex)
    q = np.diag(d)
    q[0, 1], q[1, 2], q[0, 2] = 0.5, 0.4, 0.3
    s = -3.0
    E = modal_gradient.exponential(np, q, s)
    w, V = np.linalg.eig(s * q[:3, :3])
    block = V @ np.diag(np.exp(w)) @ np.linalg.inv(V)
    assert np.max(np.abs(E[:3, :3] - block)) / np.max(np.abs(block)) < 1e-14


# --------------------------------------- 5. derivatives there, eagerly; a trace NaN

def _circle(f, z0, radius, K, derivative=False):
    unit = np.exp(2j * np.pi * np.arange(K) / K)
    return np.mean([f(z0 + radius * u) / (radius * u if derivative else 1) for u in unit], axis=0)


@functools.lru_cache(maxsize=None)
def _derivative_oracle(name, rel):
    """``(a, da/dtheta, da/dwl, F, dF/dwl)``: the amplitudes and the field at ``_FIELD_AT`` and
    their derivatives from NumPy values on circles with the rebuild gates closed (no autodiff,
    no rebuild), as ``test_2d_exceptional_point.genuine_derivatives``: radius 3e-3 in theta and
    0.5 nm in the wavelength, 32 points; values on a 1e-4 circle of 16."""
    from wavebit.rcwa._engine import solve_2d_first_order as fo
    fy, wl0, eps = EP3[name]
    wl = wl0 * (1 + rel)

    def amps(theta, w):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = _solver(name, w / wl0 - 1, theta).solve(auto_detune=False)
        return np.concatenate([_np(r.r_xyz).ravel(), _np(r.t_xyz).ravel()])

    def field(theta, w):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = _solver(name, w / wl0 - 1, theta).solve(auto_detune=False, answers='field')
        return _np(r.reconstruct_field(*(np.asarray(v) for v in _FIELD_AT)))
    with pytest.MonkeyPatch.context() as m:
        m.setattr(solve_2d, '_EP_SINE', 0.0)
        m.setattr(fo, '_PARALLEL_SIN', 0.0)
        return (_circle(lambda z: amps(z, wl), 0.3, 1e-4, 16),
                _circle(lambda z: amps(z, wl), 0.3, 3e-3, 32, derivative=True),
                _circle(lambda z: amps(0.3, z), wl, 0.5, 32, derivative=True),
                _circle(lambda z: field(z, wl), 0.3, 1e-4, 16),
                _circle(lambda z: field(0.3, z), wl, 0.5, 32, derivative=True))


def _rel(got, ref):
    return float(np.max(np.abs(got - ref)) / np.max(np.abs(ref)))


@pytest.mark.parametrize('name, rel', [(FIRST, 0.0)] + [
    pytest.param(n, r, marks=pytest.mark.slow) for n, r in
    (('fy 0.8, 664.35 nm', 0.0), ('fy 0.5, 645.23 nm', 0.0), (FIRST, 1e-9),
     ('fy 0.8, 664.35 nm', -1e-8))])
@pytest.mark.parametrize('first_order', [False, True], ids=['second_order', 'first_order'])
def test_derivatives_at_a_third_order_point_match_cauchy_oracles(name, rel, first_order):
    """Eagerly at and beside a third-order point, quiet, against :func:`_derivative_oracle`, to
    1e-11 relative: ``jax.jacfwd`` of every amplitude in theta and the wavelength, ``jax.grad``,
    torch's backward pass and ``forward_ad`` of ``sum |a|**2`` in the wavelength (oracle
    ``2 Re(conj(a) . da)``), ETM and SMM. Measured 4.1e-13 .. 6.8e-13 at the fy 0.5, 620.55 nm
    point, the oracle's own spread; before the derivative rules took the chain, every one of them
    warned. Catches the chain's Sylvester terms, its inverse's tangent or the three levels under
    a derivative dropped."""
    jax = pytest.importorskip('jax')
    torch = pytest.importorskip('torch')
    import jax.numpy as jnp
    import torch.autograd.forward_ad as fwad
    from wavebit._common.backend import _enable_x64
    _enable_x64()
    a0, da_th, da_wl, _, _ = _derivative_oracle(name, rel)
    g_ref = 2 * np.real(np.sum(np.conj(a0) * da_wl))
    wl = EP3[name][1] * (1 + rel)
    for algo in ('ETM', 'SMM'):
        s = _formulated(name, rel, algo, 'jax', first_order)

        def A(x):
            r = s.solve(theta=x[0], wavelength=x[1], auto_detune=False)
            return jnp.concatenate([r.r_xyz.ravel(), r.t_xyz.ravel()])
        jac = _quiet(lambda: np.asarray(jax.jacfwd(A)(jnp.asarray([0.3, wl]))))
        assert _rel(jac[:, 0], da_th) < 1e-11, ('theta', algo, _rel(jac[:, 0], da_th))
        assert _rel(jac[:, 1], da_wl) < 1e-11, ('wl', algo, _rel(jac[:, 1], da_wl))
        g = _quiet(lambda: float(jax.grad(lambda w: jnp.sum(jnp.abs(
            A(jnp.stack([jnp.asarray(0.3), w]))) ** 2))(jnp.asarray(wl))))
        assert abs(g - g_ref) < 1e-11 * abs(g_ref), ('grad', algo, g, g_ref)
        st = _formulated(name, rel, algo, 'torch', first_order)
        w_t = torch.tensor(wl, dtype=torch.float64, requires_grad=True)

        def backward():
            r = st.solve(wavelength=w_t, auto_detune=False)
            torch.sum(torch.cat([r.r_xyz.ravel(), r.t_xyz.ravel()]).abs() ** 2).backward()
            return float(w_t.grad)
        tg = _quiet(backward)
        assert abs(tg - g_ref) < 1e-11 * abs(g_ref), ('torch backward', algo, tg, g_ref)

        def forward():
            with fwad.dual_level():
                w_d = fwad.make_dual(torch.tensor(wl, dtype=torch.float64),
                                     torch.tensor(1.0, dtype=torch.float64))
                r = st.solve(wavelength=w_d, auto_detune=False)
                v = torch.sum(torch.cat([r.r_xyz.ravel(), r.t_xyz.ravel()]).abs() ** 2)
                return float(fwad.unpack_dual(v).tangent)
        tf = _quiet(forward)
        assert abs(tf - g_ref) < 1e-11 * abs(g_ref), ('torch forward', algo, tf, g_ref)


@pytest.mark.parametrize('first_order', [False, True], ids=['second_order', 'first_order'])
def test_field_derivatives_at_a_third_order_point_match_a_cauchy_oracle(first_order):
    """``jax.jacfwd`` of the six field components at :data:`_FIELD_AT` in the wavelength,
    eagerly at the fy 0.5, 620.55 nm point, ETM and SMM, against :func:`_derivative_oracle`'s:
    within 1e-11 relative, quiet. Measured 5.4e-13 (2D) and 5.8e-13 (first order). SMM's field
    state reads the chain off its bottom face, and a kept mode beside it takes the chain's onward
    term (``modal_gradient.two_face``)."""
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    from wavebit._common.backend import _enable_x64
    _enable_x64()
    _, _, _, _, dF = _derivative_oracle(FIRST, 0.0)
    wl = EP3[FIRST][1]
    for algo in ('ETM', 'SMM'):
        s = _formulated(FIRST, 0.0, algo, 'jax', first_order)

        def F(w):
            r = s.solve(wavelength=w, auto_detune=False, answers='field')
            return r.reconstruct_field(*(jnp.asarray(v) for v in _FIELD_AT))
        got = _quiet(lambda: np.asarray(jax.jacfwd(F)(jnp.asarray(wl))))
        assert _rel(got, dF) < 1e-11, (algo, _rel(got, dF))


def _chain_q(seed=0, split=1e-7):
    """A coupled ``q`` with a chain of three (rows 0, 1 -> columns 1, 2, level by level, and
    0 -> 2) beside a Jordan pair (3 -> 4) and three plain modes, in the layout the rebuild
    places, with its entry list ``(a, b, mask)``."""
    rng = np.random.default_rng(seed)
    d = np.array([0.6 + 0.1j, 0.6 + 0.1j + split, 0.6 + 0.1j - split, 1.3 + 0.05j, 1.3 + 0.05j,
                  0.2 + 0.02j, 1.9 + 0.3j, 2.4 + 0.1j])
    q = np.diag(d)
    entries = [(0, 1), (1, 2), (0, 2), (3, 4)]
    for i, j in entries:
        q[i, j] = (rng.normal() + 1j * rng.normal()) * 0.1
    a, b = (np.array(x) for x in zip(*entries))
    return q, a, b, np.ones(len(entries), bool)


def test_the_chain_sylvester_is_exact():
    """``solve_2d._pair_sylvester`` with a chain in the coupling: ``T X + X T = S`` against a
    dense Kronecker solve, within 1e-14 relative. Catches the update taken too few times (three
    leaves the chain's fourth-order term out)."""
    q, a, b, mask = _chain_q()
    rng = np.random.default_rng(7)
    S = rng.normal(size=q.shape) + 1j * rng.normal(size=q.shape)
    X = solve_2d._pair_sylvester(np, q, S, a, b, mask)
    n = q.shape[0]
    K = np.kron(np.eye(n), q) + np.kron(q.T, np.eye(n))
    ref = np.linalg.solve(K, S.reshape(-1, order='F')).reshape(n, n, order='F')
    assert np.max(np.abs(X - ref)) / np.max(np.abs(ref)) < 1e-14


def test_the_chain_inverse_and_its_tangent_are_exact():
    """``solve_2d._pair_inverse`` of a coupled ``q`` with a chain: the value is ``q^-1`` and its
    ``jax.jvp`` tangent ``-q^-1 dq q^-1``, within 1e-14 relative. Catches the inverse's
    second-order term or the chain's own tangent dropped."""
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    from wavebit._common.backend import _enable_x64
    _enable_x64()
    q, a, b, mask = _chain_q()
    rng = np.random.default_rng(8)
    dq = rng.normal(size=q.shape) + 1j * rng.normal(size=q.shape)
    inv, tangent = jax.jvp(lambda x: solve_2d._pair_inverse(jnp, x, jnp.asarray(a), jnp.asarray(b),
                                                         jnp.asarray(mask)),
                           (jnp.asarray(q),), (jnp.asarray(dq),))
    ref = np.linalg.inv(q)
    assert np.max(np.abs(np.asarray(inv) - ref)) / np.max(np.abs(ref)) < 1e-14
    dref = -ref @ dq @ ref
    assert np.max(np.abs(np.asarray(tangent) - dref)) / np.max(np.abs(dref)) < 1e-14


def test_the_split_sylvester_takes_a_chain():
    """``modal_gradient._split_sylvester`` with a chain in each direction, as the first-order
    split holds one at an exceptional point (``+-q``): ``X (D + N) - (D + N) X = R`` on the
    entries across the two, against a dense solve restricted to them, within 1e-14 relative.
    ``N`` then reaches such an entry from both sides, and the series needs all five terms.
    Catches the update taken too few times (three, or the two that one chain alone needs)."""
    from wavebit.rcwa._engine import modal_gradient
    q, _, _, _ = _chain_q()
    back = np.diag(-np.diag(q)[:3])
    back[0, 1], back[1, 2], back[0, 2] = -0.07 + 0.02j, 0.05 - 0.04j, 0.03 + 0.06j
    q = np.block([[q, np.zeros((8, 3))], [np.zeros((3, 8)), back]])
    d = np.diag(q)
    N = q - np.diag(d)
    forward = np.array([True] * 8 + [False] * 3)
    separate = forward[:, None] != forward[None, :]
    rng = np.random.default_rng(9)
    R = rng.normal(size=q.shape) + 1j * rng.normal(size=q.shape)
    gap = np.where(separate, d[None, :] - d[:, None], 1.0)
    X = modal_gradient._split_sylvester(np, N, gap, R, separate)
    n = q.shape[0]
    K = np.kron(q.T, np.eye(n)) - np.kron(np.eye(n), q)
    sel = separate.reshape(-1, order='F')
    sol = np.linalg.solve(K[np.ix_(sel, sel)], R.reshape(-1, order='F')[sel])
    ref = np.zeros(n * n, complex)
    ref[sel] = sol
    ref = ref.reshape(n, n, order='F')
    assert np.max(np.abs(X - ref)) / np.max(np.abs(ref)) < 1e-14


def test_two_face_takes_the_chains_onward_term():
    """``modal_gradient.two_face`` with a chain: kept rows' first variation of
    ``exp(scale Q) top`` against ``jax.jvp`` of the dense product ``expm(scale Q) @ top``, the
    dropped rows read off ``bottom = expm(scale Q) top``, within 1e-12 relative. Catches the
    chain's middle without its onward term."""
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    import jax.scipy.linalg as jsl
    from wavebit.rcwa._engine import modal_gradient
    from wavebit._common.backend import _enable_x64
    _enable_x64()
    q, _, _, _ = _chain_q(split=1e-3)
    s = -1.5
    rng = np.random.default_rng(10)
    top = rng.normal(size=(q.shape[0], 3)) + 1j * rng.normal(size=(q.shape[0], 3))
    dq = rng.normal(size=q.shape) + 1j * rng.normal(size=q.shape)
    keep = np.ones(q.shape[0], bool)

    def exact(x):
        return jsl.expm(s * x) @ jnp.asarray(top)

    def face(x):
        bottom = jax.lax.stop_gradient(exact(x))
        return modal_gradient.two_face(jnp, x, s, jnp.asarray(top), bottom, jnp.asarray(keep),
                                       coupled='pairs')
    v, dv = jax.jvp(face, (jnp.asarray(q),), (jnp.asarray(dq),))
    ve, dve = jax.jvp(exact, (jnp.asarray(q),), (jnp.asarray(dq),))
    kept = np.array([False, False, False, False, False, True, True, True])
    assert np.max(np.abs(np.asarray(v) - np.asarray(ve))) < 1e-13
    err = np.max(np.abs(np.asarray(dv)[kept] - np.asarray(dve)[kept]))
    assert err / np.max(np.abs(np.asarray(dve)[kept])) < 1e-12, err


def test_a_trace_at_a_third_order_point_answers_nan():
    """Under ``jax.jit`` no chain is read (the three-level split is eager), so the joint rebuild's
    two-level check answers NaN at the point -- never a silent answer from a rule that drops the
    second step. Catches the three levels taken under a trace."""
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    from wavebit._common.backend import _enable_x64
    _enable_x64()
    _, wl, _ = EP3[FIRST]
    s = _solver(FIRST, backend='jax')
    jv = np.asarray(jax.jit(lambda w: s.solve(wavelength=w, auto_detune=False).r_xyz)(
        jnp.asarray(wl)))
    assert np.all(np.isnan(jv))
