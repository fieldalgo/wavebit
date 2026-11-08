"""Defective clusters: rebuilt jointly and exact on the 2D path; loud where no rebuild takes
them; quiet where they are degenerate.

Two order rows of a y-invariant raster meet the x-cutoff at one ``q`` when ``ky0 = Gy / 2``
(``sin(theta) = wavelength / (2 * Py)`` at ``phi = pi/2``): two coincident Jordan blocks. ``eig``
mixes them, so no two of its four vectors are nearly parallel and the pair gate sees nothing,
yet the four span about two dimensions (smallest singular value ~1e-7). Just off the
coincidence the two pairs sit a few 1e-8 .. 1e-6 apart, and one pair rebuild each cannot
separate them. Without the joint rebuild the 2D path is 4.6e-7 .. 1.0e-3 off there (the screens
change no arithmetic: before them it was silent, since then it warns), and first-order SMM up to
0.55. The joint rebuild (``solve_2d._rebuild_defective_clusters``) takes the four modes as one
cluster.
"""

from wavebit._common.backend import ExceptionalPointWarning as _legacy_wavebit_ExceptionalPointWarning

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa
import warnings

import numpy as np
import pytest

from wavebit.rcwa._engine import solve_2d
from tests.rcwa import test_conical_cutoff as _conical_cutoff
from tests.rcwa.test_conical_cutoff import PATTERNED, _np, _pillar, _solver, _vector

PL952 = dict(structure=_pillar(), wavelength=952.0673283430336, period=1000., thickness=[200.],
             top=1.0, bot=2.25, theta=0.3)
SCENES = {'PA616': PATTERNED['PA616'], 'PL952': PL952}


def _littrow(scene, dth):
    theta0 = float(np.arcsin(scene['wavelength'] / (2 * 700.0)))
    return dict(scene, theta=theta0 * (1 + dth))


def _row(result):
    R, T, r, t = (_np(getattr(result, k)) for k in ('R', 'T', 'r_xyz', 't_xyz'))
    row = list(_np(result.orders_v)).index(0)
    return np.concatenate([[np.real(R[row]).sum(), np.real(T[row]).sum()],
                           r[row].ravel(), t[row].ravel()])


def _solve_2d(scene, algo, first_order):
    s = _solver(scene, algo=algo, two_d=True)
    if first_order:
        s.formulation = 'first_order'
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        out = _row(s.solve(auto_detune=False))
    return out, [w.category for w in rec]


COINCIDENT = [0.0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 3e-4, 1e-3]


def _exact_and_quiet(scene, algo, label, solve=None):
    ref = _vector(_solver(scene).solve(auto_detune=False))
    got, warned = solve() if solve is not None else _solve_2d(scene, algo, False)
    err = np.max(np.abs(got[2:] - ref[2:]))
    assert _legacy_wavebit_ExceptionalPointWarning not in warned, f'{label}: warned ({err:.1e} off)'
    assert err < 1e-12, f'{label}: {err:.1e} off'


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('name', list(SCENES))
@pytest.mark.parametrize('dth', COINCIDENT)
def test_coincident_exceptional_points_are_exact(name, algo, dth):
    """Every theta offset from the coincidence, 2D path: within 1e-12 of the 1D rebuild, and
    quiet. The four modes are one defective cluster, rebuilt jointly out to ``dth`` 1e-4 (PA616)
    and 5e-5 (PL952). Measured <= 1.7e-14 at 39 offsets of either sign on seven CPU arithmetic
    paths; with the pair rebuild alone the 2D path was 4.6e-7 .. 1.0e-3 off (SMM at the
    coincidence) and loud for ``dth <= 1e-5``. Catches the joint rebuild removed or its gate
    closed (loud, or off)."""
    _exact_and_quiet(_littrow(SCENES[name], dth), algo, f'{name} {algo} dth {dth:g}')


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('name', list(SCENES))
@pytest.mark.parametrize('dth', COINCIDENT)
def test_coincident_exceptional_points_on_the_first_order_path_are_exact_or_loud(name, algo, dth):
    """The first-order formulation, which has no joint rebuild: within 1e-12 of the 1D rebuild,
    or an ``ExceptionalPointWarning``. Loud for ``dth <= 1e-4`` (its separation refusal), exact
    beyond. Catches the screen removed (first-order PL952 at 1e-8 silent at 0.39)."""
    scene = _littrow(SCENES[name], dth)
    ref = _vector(_solver(scene).solve(auto_detune=False))
    got, warned = _solve_2d(scene, algo, True)
    err = np.max(np.abs(got[2:] - ref[2:]))
    loud = _legacy_wavebit_ExceptionalPointWarning in warned
    assert loud or err < 1e-12, f'{name} {algo} dth {dth:g}: {err:.1e} off, silently'
    if dth >= 3e-4:
        assert not loud and err < 1e-12


#: A uniform layer at oblique incidence whose first-order split holds two exactly degenerate
#: pairs 4.5e-5 apart in a nearly parallel eig basis (sigma 3.8e-2): normal, so harmless. A
#: defect measure that reads the diagonal spread (``||T - mean I||`` = 9.1e-5) warned here.
UNIFORM_NEAR_DEGENERATE = dict(
    backend='numpy', structure=np.full((1, 8, 8), 1.7094811474576292),
    thickness=[56.52767275682794], period=[712.7651596910116, 712.7651596910116],
    truncation_order=[4, 4], wavelength=991.8737967602215, top=1.0, bot=2.25,
    theta=0.3117819521670157, phi=0.1965062629149125, psi=2.948417220805255)


@pytest.mark.parametrize('kind', ['near-degenerate', 'normal incidence'])
@pytest.mark.parametrize('first_order', [False, True], ids=['2d', 'first_order'])
@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_degenerate_clusters_stay_quiet(kind, first_order, algo, monkeypatch):
    """A uniform layer's degenerate clusters, in whatever basis ``eig`` chose, raise nothing and
    keep the eigenvector arithmetic: the same bits with the screen disabled, and no joint
    rebuild. The first-order prototype's refusal warned on 15 of 120 such scenes;
    ``||T - mean I||`` as the measure warns on the first."""
    kw = dict(UNIFORM_NEAR_DEGENERATE)
    if kind == 'normal incidence':
        kw.update(structure=np.full((1, 8, 8), 3.1), theta=0.0, phi=0.0, truncation_order=[5, 5],
                  wavelength=731.0)

    def run():
        s = _legacy_wavebit_rcwa(layer_algorithm=algo, **kw)
        if first_order:
            s.formulation = 'first_order'
        return s.solve(auto_detune=False)

    def refused(*args, **kwargs):
        raise AssertionError('a degenerate cluster was rebuilt jointly')
    monkeypatch.setattr(solve_2d, '_rebuild_defective_clusters', refused)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        quiet = run()
    saved = solve_2d._unresolved_defective_cluster
    solve_2d._unresolved_defective_cluster = lambda *a, **k: False
    try:
        closed = run()
    finally:
        solve_2d._unresolved_defective_cluster = saved
    for key in ('R', 'T', 'r_xyz', 't_xyz'):
        assert np.array_equal(_np(getattr(quiet, key)), _np(getattr(closed, key))), key


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_the_screen_reads_non_normality(backend):
    """The defect measure on synthetic clusters: a Jordan block beside a third root is
    defective; two exactly degenerate eigenvalues in a nearly parallel basis, beside a third
    root 1e-5 away, are not."""
    if backend == 'jax':
        jax = pytest.importorskip('jax')
        jax.config.update('jax_enable_x64', True)
        import jax.numpy as xp
    elif backend == 'torch':
        pytest.importorskip('torch')
        import array_api_compat.torch as xp
    else:
        xp = np
    rng = np.random.default_rng(3)
    Q, _ = np.linalg.qr(rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6)))
    jordan = np.diag([0.3, 0.3, 0.3 + 1e-6, 0.9, 1.4, 2.0]).astype(complex)
    jordan[0, 1] = 0.05
    A = Q @ jordan @ Q.conj().T
    w, W = np.linalg.eig(A)
    assert solve_2d._unresolved_defective_cluster(
        xp, xp.asarray(w), xp.asarray(W), lambda: xp.asarray(A), 1e-2)
    D = np.diag([0.3, 0.3, 0.3 + 1e-5, 0.9, 1.4, 2.0]).astype(complex)
    V = np.eye(6, dtype=complex)
    V[:, 1] = (V[:, 0] + 1e-4 * V[:, 1]) / np.linalg.norm(V[:, 0] + 1e-4 * V[:, 1])
    B = Q @ D @ Q.conj().T
    w = np.diag(D).copy()
    assert not solve_2d._unresolved_defective_cluster(
        xp, xp.asarray(w), xp.asarray(Q @ V), lambda: xp.asarray(B), 1e-2)


def _jordan_beside(delta, seed=0):
    """A 12 x 12 product operator: a Jordan pair at 0.3 (coupling 0.05) beside a plain root at
    ``0.3 + delta``, the rest spread out, turned by a random unitary; with ``eig``'s modes."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(12, 12)) + 1j * rng.normal(size=(12, 12)))
    J = np.diag([0.3, 0.3, 0.3 + delta, 0.7, 0.9, 1.1, 1.3, 1.6, 1.9, 2.3, 2.8, 3.4]).astype(complex)
    J[0, 1] = 0.05
    A = Q @ J @ Q.conj().T
    w, W = np.linalg.eig(A)
    return A, w, W


def _layer(A, cap=4, slots=2, xp=np):
    """The gates and the rebuild on a synthetic product operator, as ``layer_modes_2d`` runs
    them (``Omega_R = I``, so ``V = W q^-1``): ``(pairs, clusters, rebuilt, warnings)``."""
    w, W = np.linalg.eig(A)
    q = np.sqrt(w)
    w, W, q, A = (xp.asarray(x) for x in (w, W, q, A))
    pairs = solve_2d._defective_pairs(xp, w, W, q, cap)
    clusters = solve_2d._defective_clusters(xp, w, W, q, pairs, lambda: A, slots)
    rebuilt = None
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        if pairs is not None or clusters is not None:
            extra = {} if clusters is None else dict(clusters=clusters)
            rebuilt = solve_2d._rebuild_defective_pairs(
                xp, A, xp.asarray(np.eye(A.shape[0], dtype=complex)), W, W / q, q, w, pairs, **extra)
    return pairs, clusters, rebuilt, [r.category for r in rec]


def _backward(A, rebuilt):
    W, _, Q = (np.asarray(x) for x in rebuilt[:3])
    return np.linalg.norm(A @ W - W @ (Q @ Q)) / np.linalg.norm(A)


@pytest.mark.parametrize('delta', [0.0, 1e-10, 1e-8, 2e-8, 5e-8, 9e-8, 2e-7, 1e-6, 1e-4])
def test_a_pair_beside_an_unpaired_root(delta):
    """The 2D pair gate skips a pair with an unpaired root within ``_EP_SEPARATION`` (1e-7) and
    rebuilds it beyond, where the Newton step leaves at most 8.1e-14 (measured; 2.0e-12 at 1e-8,
    3.5e-13 at 2e-8 with the guard off). Within it the three modes are one defective cluster,
    rebuilt jointly. Measured on six bases, delta 0 .. 1e-7 on eight CPU arithmetic paths and
    9e-8, 1e-7 on three more: <= 2.8e-15 of ``||A||``, but where ``eig`` collapsed the basis
    (one path, one basis at 0: :func:`test_a_collapsed_start_is_loud`). Catches the guard
    removed or widened past 1e-7, a
    gated pair beside a far root taken jointly (it is exact on the pair path), and the joint
    rebuild of a three-mode cluster (its padded lane) broken."""
    A, w, W = _jordan_beside(delta)
    pairs, clusters, rebuilt, warned = _layer(A)
    if delta <= 1e-7:
        assert pairs is None and clusters is not None
        assert np.asarray(clusters[1]).sum() == 3
    else:
        assert pairs is not None and clusters is None
    assert not warned
    assert _backward(A, rebuilt) < 1e-13


def _first(A, near, count=3):
    """``eig``'s modes of ``A`` with the ``count`` nearest ``near`` moved to the front."""
    w, W = np.linalg.eig(A)
    front = np.argsort(np.abs(w - near))[:count].tolist()
    order = front + [i for i in range(len(w)) if i not in front]
    return w[order], W[:, order]


def test_a_three_mode_ball_holding_mode_zero():
    """A ball of three pads its fourth lane, and the padded pick is whatever index the search
    over infinities returns: index 0. Where mode 0 is a member that pick must stay padding. A
    Jordan pair beside a root, its modes first: three distinct members, exact. A normal ball of
    three (eigenvalues 2e-5 apart, orthonormal columns), modes first: no cluster at all -- read as
    four lanes with mode 0 twice, its columns looked dependent and it was rebuilt, failing the
    check (the micrometre R1 twin in ``tests/rcwa/test_conical_cutoff.py``, three evanescent y
    orders within 1e-4)."""
    A = _jordan_beside(1e-9)[0]
    w, W = _first(A, 0.3)
    q = np.sqrt(w)
    clusters = solve_2d._defective_clusters(np, w, W, q, None, lambda: A, 2)
    members, valid = (np.asarray(x) for x in clusters[:2])
    assert valid.tolist() == [[True, True, True, False]]
    assert sorted(members[valid].tolist()) == [0, 1, 2]
    rng = np.random.default_rng(4)
    U, _ = np.linalg.qr(rng.normal(size=(9, 9)) + 1j * rng.normal(size=(9, 9)))
    B = U @ np.diag([0.3, 0.3 + 2e-5, 0.3 + 4e-5, 0.8, 1.1, 1.5, 1.9, 2.4, 3.0]) @ U.conj().T
    w, W = _first(B, 0.3)
    assert solve_2d._defective_clusters(np, w, W, np.sqrt(w), None, lambda: B, 2) is None


def test_the_split_does_not_trust_eig_s_order(monkeypatch):
    """A three-mode cluster pads its fourth lane at a stand-in eigenvalue, and the position in
    which ``eig`` hands that eigenpair back is not promised (LAPACK's returns the decoupled one
    last; another solver need not). With every small ``eig`` of the split handing its pairs back
    reversed, the joint rebuild is the same: exact and quiet. Catches the padded pair placed by
    position."""
    original = solve_2d._eig_raw
    monkeypatch.setattr(solve_2d, '_eig_raw', lambda T: tuple(x[..., ::-1] for x in original(T)))
    for delta in (0.0, 1e-9, 5e-8):
        A = _jordan_beside(delta)[0]
        _, clusters, rebuilt, warned = _layer(A)
        assert np.asarray(clusters[1]).sum() == 3
        assert not warned and _backward(A, rebuilt) < 1e-13, delta


def test_a_cluster_takes_one_root_side():
    """A cluster's members may carry labels of opposite sides (a tie the flux breaks, a complex
    root on Maystre's cut); every root then takes the side of the most decaying member, so the
    coupling's divisors ``r_j + r_k`` stay about ``2r``, as a pair's do. A pair beside a root with
    one member's label flipped: exact, every root on one side. Catches each member kept on its own
    label's side (a divisor near zero)."""
    A, w, W = _jordan_beside(1e-9)
    q = np.sqrt(w)
    flip = np.argsort(np.abs(w - 0.3))[1]
    q[flip] = -q[flip]
    clusters = solve_2d._defective_clusters(np, w, W, q, None, lambda: A, 2)
    rebuilt = solve_2d._rebuild_defective_pairs(np, A, np.eye(12), W, W / q, q, w, None,
                                                clusters=clusters)
    assert _backward(A, rebuilt) < 1e-13
    members = np.asarray(clusters[0])[np.asarray(clusters[1])]
    roots = np.diag(np.asarray(rebuilt[2]))[members]
    assert np.all(np.real(roots) > 0), roots


def test_a_chain_is_not_taken():
    """A pair refused for a root within ``_EP_SEPARATION``, a mode 0.9e-4 beyond them and another
    0.9e-4 beyond that: the first four are a ball of the lowest member, but the third's ball
    holds five -- a chain, not a closed ball. The joint rebuild does not take it (a closed ball is
    what keeps every divisor of its Newton step at least the cluster radius); the unresolved screen
    reads it. Catches the closed-ball condition dropped."""
    rng = np.random.default_rng(0)
    U, _ = np.linalg.qr(rng.normal(size=(12, 12)) + 1j * rng.normal(size=(12, 12)))
    J = np.diag([0.3, 0.3, 0.3 + 5e-8, 0.3 + 0.9e-4, 0.3 + 1.8e-4, 0.7, 0.9, 1.3, 1.6, 1.9, 2.3,
                 2.8]).astype(complex)
    J[0, 1] = 0.05
    A = U @ J @ U.conj().T
    w, W = _first(A, 0.3)
    q = np.sqrt(w)
    pairs = solve_2d._defective_pairs(np, w, W, q, 4)
    assert pairs is None
    assert solve_2d._defective_clusters(np, w, W, q, pairs, lambda: A, 2) is None
    assert solve_2d._unresolved_defective_cluster(np, w, W, lambda: A, solve_2d._EP_SINE)


def test_a_degenerate_ball_in_a_parallel_basis_is_not_taken():
    """Two exactly degenerate pairs 1e-5 apart, each in a nearly parallel basis (``eig`` may
    choose one): four modes, a closed ball, unit columns nearly dependent -- and a normal
    restriction, so no defect. The joint rebuild does not take it; the pair rebuild, exact for a
    degenerate pair, does. Catches the defect measure dropped from the gate."""
    rng = np.random.default_rng(3)
    U, _ = np.linalg.qr(rng.normal(size=(10, 10)) + 1j * rng.normal(size=(10, 10)))
    w = np.array([0.3, 0.3, 0.3 + 1e-5, 0.3 + 1e-5, 0.9, 1.4, 2.0, 2.6, 3.1, 3.7], dtype=complex)
    A = U @ np.diag(w) @ U.conj().T
    V = np.eye(10, dtype=complex)
    for i in (0, 2):
        V[:, i + 1] = (V[:, i] + 1e-4 * V[:, i + 1]) / np.linalg.norm(V[:, i] + 1e-4 * V[:, i + 1])
    W = U @ V
    q = np.sqrt(w)
    pairs = solve_2d._defective_pairs(np, w, W, q, 4)
    assert pairs is not None and int(np.sum(pairs[2])) == 2
    assert solve_2d._defective_clusters(np, w, W, q, pairs, lambda: A, 2) is None


@pytest.mark.parametrize('name', list(_conical_cutoff.UNIFORM))
@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_uniform_twins_take_no_joint_rebuild(name, algo, monkeypatch):
    """Every uniform-spacer twin of the conical suite, 2D path: no joint rebuild and no
    ``ExceptionalPointWarning``. Their spacers hold degenerate (normal) balls of evanescent y
    orders; the micrometre one at mode 0."""
    def refused(*args, **kwargs):
        raise AssertionError('a uniform layer was rebuilt jointly')
    monkeypatch.setattr(solve_2d, '_rebuild_defective_clusters', refused)
    _, warned = _solve_2d(_conical_cutoff.UNIFORM[name], algo, False)
    assert _legacy_wavebit_ExceptionalPointWarning not in warned


@pytest.mark.parametrize('name', ['arange element (um)', 'R1 air spacer', 'theta 1.4'])
def test_a_traced_uniform_twin_is_the_eager_answer(name):
    """Under ``jax.jit`` every 2D layer runs the cluster slots; a uniform spacer's degenerate
    balls of evanescent y orders must take none (their unit columns are orthonormal), or a layer
    with more such balls than slots would come back NaN. Values within 1e-13 of the eager solve.
    Catches the unit columns' dependence dropped from the traced gate."""
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    jnp = jax.numpy
    scene = _conical_cutoff.UNIFORM[name]
    s = _solver(scene, backend='jax', two_d=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        eager = np.asarray(s.solve(auto_detune=False).r_xyz)
        traced = np.asarray(jax.jit(lambda w: s.solve(wavelength=w, auto_detune=False).r_xyz)(
            jnp.asarray(scene['wavelength'])))
    assert np.all(np.isfinite(traced)) and np.max(np.abs(traced - eager)) < 1e-13


def _collapsed(seed):
    """A Jordan pair and a plain root at one eigenvalue, 0.3, with the three columns ``eig`` may
    hand back for them on some CPU arithmetic paths (one basis of six on one path measured,
    smallest singular value 3.6e-15) built by hand: the pair's eigenvector, and two copies of it
    turned by 1e-15 into the cluster's other two directions. ``(A, w, W)``."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(12, 12)) + 1j * rng.normal(size=(12, 12)))
    J = np.diag([0.3, 0.3, 0.3, 0.7, 0.9, 1.1, 1.3, 1.6, 1.9, 2.3, 2.8, 3.4]).astype(complex)
    J[0, 1] = 0.05
    A = Q @ J @ Q.conj().T
    w, W = np.linalg.eig(A)
    near = np.argsort(np.abs(w - 0.3))[:3]
    for i, c in zip(near, (Q[:, 0], Q[:, 0] + 1e-15 * Q[:, 1], Q[:, 0] + 1e-15 * Q[:, 2])):
        W[:, i] = c / np.linalg.norm(c)
    w[near] = 0.3
    return A, w, W


@pytest.mark.parametrize('seed', [0, 1, 2, 3])
def test_a_collapsed_start_is_loud(seed):
    """Three nearly parallel columns for a three-mode cluster (:func:`_collapsed`, smallest
    singular value ~6e-16) leave the cluster's subspace undetermined beyond one Newton step:
    measured backward errors 3.7e-4 .. 6.1e-2 on four CPU arithmetic paths, and the joint
    rebuild's check reads them (its subspace residual against <= 7e-14 on every measured
    layer). Eagerly an ``ExceptionalPointWarning``; under ``jax.jit`` a NaN answer. Catches the
    check removed, or not applied to a traced layer."""
    A, w, W = _collapsed(seed)
    q = np.sqrt(w)
    pairs = solve_2d._defective_pairs(np, w, W, q, 4)
    clusters = solve_2d._defective_clusters(np, w, W, q, pairs, lambda: A, 2)
    assert clusters is not None
    with pytest.warns(_legacy_wavebit_ExceptionalPointWarning, match='joint rebuild failed'):
        rebuilt = solve_2d._rebuild_defective_pairs(np, A, np.eye(12), W, W / q, q, w, pairs,
                                                    clusters=clusters)
    assert _backward(A, rebuilt) > 1e-8
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    jnp = jax.numpy

    @jax.jit
    def traced(w, W, A):
        q = jnp.sqrt(w)
        pairs = solve_2d._defective_pairs(jnp, w, W, q, 4)
        clusters = solve_2d._defective_clusters(jnp, w, W, q, pairs, lambda: A, 2)
        return solve_2d._rebuild_defective_pairs(jnp, A, jnp.eye(12, dtype=complex), W, W / q, q,
                                                 w, pairs, clusters=clusters)[2]
    assert np.all(np.isnan(np.asarray(traced(jnp.asarray(w), jnp.asarray(W), jnp.asarray(A)))))


def _chain(seed=0):
    """A length-3 Jordan chain at 0.3 (couplings 0.05), the rest spread, turned by a unitary."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(12, 12)) + 1j * rng.normal(size=(12, 12)))
    J = np.diag([0.3, 0.3, 0.3, 0.7, 0.9, 1.1, 1.3, 1.6, 1.9, 2.3, 2.8, 3.4]).astype(complex)
    J[0, 1] = J[1, 2] = 0.05
    return Q @ J @ Q.conj().T


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_a_length_three_chain_is_loud(seed):
    """No basis puts a length-3 Jordan chain in the disjoint form, so no rebuild here is exact
    there. ``eig`` splits its eigenvalue by ~1e-6 and the pair gate takes two of the three, beside
    a third more than 1e-7 away: the pair rebuild, after which the cluster screen still sees
    three nearly dependent columns of a non-normal cluster (``cond(W)`` ~3e9) and warns. Handed
    to the joint rebuild instead, its check reads the split's dropped part times ``cond(X)`` and
    warns. Catches a three-mode ball holding one gated pair taken jointly without the check, and
    the check's split term removed."""
    A = _chain(seed)
    pairs, clusters, rebuilt, warned = _layer(A)
    assert pairs is not None and clusters is None and not warned
    w, W = np.linalg.eig(A)
    assert solve_2d._unresolved_defective_cluster(np, w, rebuilt[0], lambda: A, solve_2d._EP_SINE)
    q = np.sqrt(w)
    forced = solve_2d._defective_clusters(np, w, W, q, None, lambda: A, 2)
    assert forced is not None
    with pytest.warns(_legacy_wavebit_ExceptionalPointWarning, match='joint rebuild failed'):
        solve_2d._rebuild_defective_pairs(np, A, np.eye(12), W, W / q, q, w, None, clusters=forced)


def _two_clusters_and_a_pair(gap, seed=1, far=(2.4, 3.0)):
    """A 16 x 16 product operator in a basis of condition 10: two coincident Jordan pairs at
    0.3 and again at ``0.3 + gap`` (two defective clusters of four), a Jordan pair at
    ``0.3 - gap``, and six roots spread out, the last two at ``far``."""
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16)))
    Vh, _ = np.linalg.qr(rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16)))
    P = U @ np.diag(np.geomspace(1, 10, 16)) @ Vh
    J = np.diag([0.3] * 4 + [0.3 + gap] * 4 + [0.3 - gap] * 2
                + [0.8, 1.1, 1.5, 1.9, *far]).astype(complex)
    J[0, 1], J[2, 3], J[4, 5], J[6, 7], J[8, 9] = 0.05, 0.04, 0.03, 0.06, 0.05
    return P @ J @ np.linalg.inv(P)


@pytest.mark.parametrize('seed', [1, 2, 3])
@pytest.mark.parametrize('gap', [2e-4, 5e-4, 1e-3, 1e-2])
def test_two_clusters_and_a_pair_in_a_general_basis(gap, seed):
    """Two defective clusters and a pair, ``gap`` apart, in a basis that does not decouple them:
    every block's Newton step reads the others' rows -- a cluster's rows in another cluster's
    correction by the Kronecker solve, in a pair's by ``k x k`` solves. Measured on four CPU
    arithmetic paths (seeds 1 .. 5) and, at gaps 2e-4, 1e-3 and 1e-2, seven more (seeds 1 .. 3):
    the layer model ``A W = W Q^2`` to <= 2.5e-15 of ``||A||`` at gaps 5e-4 .. 1e-2, quiet; at
    2e-4 one Newton step leaves a remainder of second order over
    the small gap, up to 4.3e-13, and on one path the check read it as a miss (residual 3.24e-12
    against 3.00e-12) and warned -- still exact, so there the warning is allowed.
    ``Omega_R W = V Q``; ``Q``'s coupling squares to exactly zero; the entry list names exactly
    its nonzero entries. On the y-Littrow scenes the inter-block rows change no answer (the
    blocks live in orthogonal order rows) and only the check reads their absence
    (:func:`test_two_clusters_in_one_layer` then warns); here they change the answer. Catches the
    Kronecker or the pair-to-cluster rows dropped."""
    A = _two_clusters_and_a_pair(gap, seed)
    pairs, clusters, rebuilt, warned = _layer(A, cap=8, slots=4)
    assert gap < 5e-4 or not warned
    assert np.asarray(clusters[1]).sum(axis=1).tolist() == [4, 4]
    W, V, Q, a, b, mask = (np.asarray(x) for x in rebuilt)
    assert _backward(A, rebuilt) < 1e-12
    assert np.linalg.norm(W - V @ Q) < 1e-13 * np.linalg.norm(W)
    off = Q - np.diag(np.diag(Q))
    assert np.array_equal(off @ off, np.zeros_like(off))
    assert set(zip(a[mask].tolist(), b[mask].tolist())) == set(zip(*np.nonzero(off)))
    assert len(a) == np.count_nonzero(mask) == np.count_nonzero(off)
    assert not set(a.tolist()) & set(b.tolist())


@pytest.mark.parametrize('seed', [1, 2, 3])
def test_a_large_operator_norm_does_not_fail_the_check(seed):
    """The clusters of :func:`test_two_clusters_and_a_pair_in_a_general_basis` beside two roots at
    3e4 and 5e4, as evanescent orders put them at large truncations: ``||A||_F`` ~1.3e5, so the
    corrected subspace's rounding residual (0.5 .. 0.8 ``eps ||A||_F`` here, at most 0.26 on the
    y-Littrow layers at truncations 5 .. 20) is ~2e-11, above the cluster-scale threshold of
    3e-12. Quiet and exact to 1e-12 of ``||A||``. Catches the residual read against the cluster's
    scale alone."""
    A = _two_clusters_and_a_pair(1e-3, seed, far=(3e4, 5e4))
    pairs, clusters, rebuilt, warned = _layer(A, cap=8, slots=4)
    assert clusters is not None and not warned
    assert _backward(A, rebuilt) < 1e-12


@pytest.mark.parametrize('model', ['beside 1e-9', 'two clusters'])
def test_the_traced_cluster_gate_takes_the_eager_set(model):
    """Under ``jax.jit`` and ``torch.func.vmap`` the gate returns static slots with masks as data;
    the live slots must be the eager clusters, member for member. Catches the slots ranked or
    masked differently from the eager gate."""
    A = _jordan_beside(1e-9)[0] if model == 'beside 1e-9' else _two_clusters_and_a_pair(1e-3)
    w, W = np.linalg.eig(A)
    q = np.sqrt(w)
    eager = solve_2d._defective_clusters(np, w, W, q, solve_2d._defective_pairs(np, w, W, q, 8),
                                         lambda: A, 4)

    def sets(members, valid):
        members, valid = np.asarray(members), np.asarray(valid)
        return sorted(tuple(sorted(m[v].tolist())) for m, v in zip(members, valid) if v.any())
    want = sets(*eager[:2])
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    jnp = jax.numpy

    def gate(xp, w, W, q, A):
        out = solve_2d._defective_clusters(xp, w, W, q, solve_2d._defective_pairs(xp, w, W, q, 8),
                                           lambda: A, 4)
        return out[0], out[1], out[3]
    m, v, over = jax.jit(lambda *x: gate(jnp, *x))(*(jnp.asarray(x) for x in (w, W, q, A)))
    assert sets(m, v) == want and not bool(over)
    torch = pytest.importorskip('torch')
    import array_api_compat.torch as tx
    batch = [torch.asarray(x)[None] for x in (w, W, q, A)]
    m, v, over = torch.func.vmap(lambda *x: gate(tx, *x))(*batch)
    assert sets(m[0], v[0]) == want and not bool(over[0])


#: Offsets on both sides of a y-Littrow coincidence where, in one scene or the other, two pairs
#: sit a few 1e-8 .. 1e-7 apart and one pair rebuild each cannot separate them. Near there the
#: grid of ``test_coincident_exceptional_points_are_exact`` has only 1e-8, 1e-7 and 1e-6, and no
#: negative offsets.
NEAR_COINCIDENT = [5e-8, 7e-8, 1.5e-7, 2e-7, 4e-7, -5e-8, -1.5e-7]


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('name', list(SCENES))
@pytest.mark.parametrize('dth', NEAR_COINCIDENT)
def test_nearly_coincident_pairs_are_exact(name, algo, dth):
    """Within 1e-12 of the 1D rebuild, and quiet, where one Newton step per pair cannot
    separate two pairs: the four modes are rebuilt as one cluster. Measured <= 1.5e-14 on seven
    CPU arithmetic paths. With
    the pair rebuild alone the 2D path was 1.2e-12 .. 1.7e-10 off on one or more of seven CPU
    arithmetic paths, silent before ``_EP_BACKWARD`` and loud after it; the band's edge moved
    with the path. Catches the joint rebuild removed, and a gate that leaves a nearly coincident
    pair to the pair rebuild (PL952 7e-8 holds one gated pair beside two ungated modes)."""
    _exact_and_quiet(_littrow(SCENES[name], dth), algo, f'{name} {algo} dth {dth:g}')


#: The same at higher truncations, each scene at its own exceptional-point wavelength there (the
#: zero of the 1D family's cutoff root): offsets the pair rebuild alone got wrong on most of four
#: CPU arithmetic paths, 2.2e-11 .. 1.3e-10 off at worst.
HIGHER_TRUNCATION = [('PA616', 10, 638.8575175362382, 2e-7), ('PL952', 10, 956.9477951876295, -3e-7),
                     ('PA616', 15, 639.0691826791563, 3e-7), ('PL952', 15, 957.4006448059722, 5e-7),
                     ('PA616', 20, 639.1092327634085, 4e-7), ('PA616', 20, 639.1092327634085, 5.5e-7),
                     ('PL952', 20, 957.5144881024428, 6e-7), ('PA616', 15, 639.0691826791563, 0.0),
                     ('PL952', 20, 957.5144881024428, 0.0)]


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('name, truncation, wavelength, dth', HIGHER_TRUNCATION)
def test_nearly_coincident_pairs_at_higher_truncation_are_exact(name, truncation, wavelength,
                                                                dth, algo):
    """Within 1e-12 of the 1D rebuild, and quiet, at truncations 10 .. 20, where the evanescent
    orders grow ``||A||_F`` to about 2.6e3 and the joint rebuild's check reads its subspace
    residual against ``eps ||A||_F`` (:data:`~wavebit.rcwa._engine.solve_2d._EP_RESIDUAL`).
    Measured <= 7.1e-14 at 39 offsets on four CPU arithmetic paths; the residual at most 0.26 of
    ``eps ||A||_F`` over truncations 5 .. 20. Catches the joint rebuild removed."""
    scene = dict(_littrow(dict(SCENES[name], wavelength=wavelength), dth), truncation=truncation)
    _exact_and_quiet(scene, algo, f'{name} to {truncation} {algo} dth {dth:g}')


#: PA616 thickened to 10 um and 3 um, the y period set so the coincidence sits at ``ky0 = 0.025``:
#: the cluster's roots are ``|q| = 0.025``, so its propagation sensitivity ``k0 d / (2|q|)`` is
#: 2.0e3 (6.1e2 at 3 um). With the pair rebuild alone, -5.999e-6 was 3.3e-12 (10 um) and
#: 1.3e-12 (3 um) off; the backward-error screen, scaled by that sensitivity, warned there and,
#: read on a joint cluster, warned on exact answers within 1e-8 of the coincidence.
THICK = [(10000., -5.999e-6), (3000., -5.999e-6), (10000., 6e-6), (10000., 0.0),
         (10000., 1e-11), (10000., -1e-10), (10000., -1e-9), (10000., -2e-9), (10000., 3e-9),
         (10000., -3e-9), (10000., -1e-8), (3000., 0.0), (3000., -1e-9), (10000., 1e-3)]
THICK_WAVELENGTH = 616.5157327418266


def _thick(thickness, dth):
    return dict(SCENES['PA616'], thickness=[thickness], wavelength=THICK_WAVELENGTH,
                theta=float(np.arcsin(0.025)) * (1 + dth))


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('thickness, dth', THICK)
def test_nearly_coincident_pairs_in_a_thick_layer_are_exact(monkeypatch, thickness, dth, algo):
    """Within 1e-12 of the 1D rebuild, and quiet, where a thick layer and small roots turn a
    small miss into a large error. Measured <= 3.1e-13 at 17 offsets, 0 warnings in 204 solves on
    NumPy, JAX and PyTorch (the 1D rebuild's own floor here is ~2e-13). Catches a joint cluster
    handed to the sensitivity-scaled backward-error screen, which warned on these exact answers
    at 24 of 264 solves (NumPy, JAX and PyTorch, 22 offsets out to 1e-5; every one within 2e-8 of
    the coincidence, here at -2e-9 and -3e-9 on NumPy), and
    the joint rebuild removed; ``test_the_crowded_pair_screen_reads_the_backward_error`` pins the
    screen's scaling itself."""
    monkeypatch.setattr(_conical_cutoff, 'PERIOD_Y', THICK_WAVELENGTH / (2 * 0.025))
    _exact_and_quiet(_thick(thickness, dth), algo, f'{thickness:g} nm {algo} dth {dth:g}')


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_a_thick_layer_at_the_coincidence_is_exact_on_every_backend(monkeypatch, backend, algo):
    """The thick layer at and near its coincidence, eagerly on JAX and PyTorch: the same answer
    and silence as NumPy's (measured <= 2.6e-13). The points are where those backends' last bits
    made the backward-error screen, handed the joint clusters, warn (JAX at 1e-10, PyTorch at
    1e-10 and -2e-9)."""
    monkeypatch.setattr(_conical_cutoff, 'PERIOD_Y', THICK_WAVELENGTH / (2 * 0.025))
    if backend == 'jax':
        pytest.importorskip('jax').config.update('jax_enable_x64', True)
    else:
        pytest.importorskip('torch')
    for dth in (0.0, 1e-10, -2e-9):
        scene = _thick(10000., dth)

        def solve():
            s = _solver(scene, algo=algo, backend=backend, two_d=True)
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter('always')
                out = _row(s.solve(auto_detune=False))
            return out, [w.category for w in rec]
        _exact_and_quiet(scene, algo, f'{backend} {algo} dth {dth:g}', solve)


def _solver_v2(scene, algo='ETM', backend='numpy'):
    """``scene``'s 2D twin at truncation ``(5, 2)``: rows -1, 0 meet the x-cutoff at the
    coincidence and rows -2, 1 meet it too (``|ky| = 3 Gy / 2``), two clusters in one layer."""
    return _legacy_wavebit_rcwa(
        backend=backend, wavelength=scene['wavelength'], top=scene['top'], bot=scene['bot'],
        theta=scene['theta'], phi=_conical_cutoff.HALF_PI, psi=_conical_cutoff.HALF_PI,
        thickness=scene['thickness'], layer_algorithm=algo,
        period=[scene['period'], _conical_cutoff.PERIOD_Y], truncation_order=[5, 2],
        structure=np.repeat(scene['structure'], 2, axis=1))


@pytest.fixture
def joint_layers(monkeypatch):
    """Every jointly rebuilt layer: its operator, clusters and answer."""
    seen = []
    original = solve_2d._rebuild_defective_clusters

    def recorded(xp, A, Omega_R, W, V, q, w, pairs, clusters):
        out = original(xp, A, Omega_R, W, V, q, w, pairs, clusters)
        seen.append((np.asarray(A), [np.asarray(c) for c in clusters[:2]], out))
        return out
    monkeypatch.setattr(solve_2d, '_rebuild_defective_clusters', recorded)
    return seen


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('name', list(SCENES))
@pytest.mark.parametrize('dth', [0.0, 1e-8, 7e-8, -1.5e-7, 1e-6])
def test_two_clusters_in_one_layer(name, algo, dth, joint_layers):
    """At truncation ``(5, 2)`` the layer holds two four-mode clusters, so the Kronecker rows
    run on a physical layer: the ``v = 0`` row within 1e-12 of the 1D rebuild, quiet, and the
    layer model ``A W = W Q^2`` to 1e-13 of ``||A||`` for both clusters (the rows -2 and 1 carry
    no amplitude, so only the model sees the second). Measured <= 8.1e-15 and <= 4.8e-15; with
    the pair rebuild alone up to 2.5e-3 off."""
    scene = _littrow(SCENES[name], dth)

    def solve():
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            out = _row(_solver_v2(scene, algo).solve(auto_detune=False))
        return out, [w.category for w in rec]
    _exact_and_quiet(scene, algo, f'{name} {algo} dth {dth:g}', solve)
    assert joint_layers
    for A, (members, valid), out in joint_layers:
        assert valid.sum(axis=1).tolist() == [4, 4]
        assert _backward(A, out) < 1e-13


def test_more_clusters_than_the_trace_holds_is_loud(monkeypatch):
    """Under ``jax.jit`` the clusters are rebuilt up to a static count of slots; the ``(5, 2)``
    layer at the coincidence holds two, so one slot must give NaN, never the other cluster left
    to the pair rebuild; two slots are exact."""
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    jnp = jax.numpy
    scene = _littrow(SCENES['PL952'], 0.0)
    ref = _vector(_solver(scene).solve(auto_detune=False))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        row = list(_np(_solver_v2(scene).solve(auto_detune=False).orders_v)).index(0)
    original = solve_2d._defective_clusters
    for slots, finite in ((1, False), (2, True)):
        monkeypatch.setattr(solve_2d, '_defective_clusters',
                            lambda *a, slots=slots: original(*a[:-1], slots))
        s = _solver_v2(scene, backend='jax')
        R, T, r, t = (np.asarray(x) for x in jax.jit(lambda w: [
            getattr(s.solve(wavelength=w, auto_detune=False), k)
            for k in ('R', 'T', 'r_xyz', 't_xyz')])(jnp.asarray(scene['wavelength'])))
        if not finite:
            assert np.all(np.isnan(R)), slots
            continue
        got = np.concatenate([r[row].ravel(), t[row].ravel()])
        assert np.max(np.abs(got - ref[2:])) < 1e-12


@pytest.mark.slow
@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_a_trace_keeps_the_bits_where_no_cluster_is_live(algo, monkeypatch):
    """Under a trace every 2D layer runs the cluster slots with their masks as data; where no
    slot is live the layer must keep the pair rebuild's bits. ``jax.jit`` values (R, T, r, t),
    ``jit(grad)`` and ``torch.func.vmap`` values with the cluster gate open and closed (the pair
    rebuild alone) are identical on PA616 (three pairs) and the uniform R1 twin. Catches a dead
    slot's arithmetic reaching a live value."""
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    torch = pytest.importorskip('torch')
    jnp = jax.numpy
    from tests.rcwa.test_conical_cutoff import R1
    original = solve_2d._defective_clusters

    def run(scene):
        s = _solver(scene, algo=algo, backend='jax', two_d=True)
        wl = jnp.asarray(scene['wavelength'])
        values = jax.jit(lambda w: [getattr(s.solve(wavelength=w, auto_detune=False), k)
                                    for k in ('R', 'T', 'r_xyz', 't_xyz')])(wl)
        grad = jax.jit(jax.grad(lambda w: jnp.sum(jnp.real(s.solve(wavelength=w,
                                                                  auto_detune=False).R))))(wl)
        st = _solver(scene, algo=algo, backend='torch', two_d=True)
        lanes = torch.tensor([scene['wavelength'], scene['wavelength'] * (1 + 1e-3)],
                             dtype=torch.float64)
        tv = torch.func.vmap(lambda w: st.solve(wavelength=w, auto_detune=False).r_xyz)(lanes)
        return [np.asarray(v) for v in values] + [np.asarray(grad), tv.numpy()]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for label, scene in (('PA616', PATTERNED['PA616']), ('R1', R1)):
            live = run(scene)
            monkeypatch.setattr(solve_2d, '_defective_clusters', lambda *a: None)
            closed = run(scene)
            monkeypatch.setattr(solve_2d, '_defective_clusters', original)
            for k, (x, y) in enumerate(zip(live, closed)):
                assert x.tobytes() == y.tobytes(), f'{label} {algo} output {k}'


LITTROW_CASES = [('PL952', 0.0), ('PA616', 1e-8)]


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('name, dth', LITTROW_CASES + [
    pytest.param('PL952', 7e-8, marks=pytest.mark.slow)])
def test_derivatives_at_a_coincidence_are_the_1d_rebuild_s(name, dth, algo):
    """``d(sum R)`` in the raster, the wavelength, the thickness and ``theta`` at a jointly
    rebuilt cluster: ``jax.grad``, ``jax.jvp``, torch reverse and ``forward_ad`` against
    ``jax.grad`` of the 1D rebuild. Measured <= 1.8e-13; with the pair rebuild alone 3.3e-8 ..
    3.1e-3. Catches an entry list that leaves out a cluster's coupling (the generator and
    inverse rules then drop its terms while the values stay exact)."""
    from tests.rcwa.test_conical_cutoff import _rel
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    torch = pytest.importorskip('torch')
    from torch.autograd import forward_ad
    jnp = jax.numpy
    scene = _littrow(SCENES[name], dth)
    u0, t0, th0, wl0 = (np.asarray(scene['structure'], float), float(scene['thickness'][0]),
                        float(scene['theta']), float(scene['wavelength']))

    def make(backend, two_d):
        s = _solver(scene, algo=algo if two_d else 'ETM', backend=backend, two_d=two_d)
        if backend == 'jax':
            def f(u, w, d, th):
                st = jnp.repeat(u, 2, axis=1) if two_d else u
                return jnp.sum(jnp.real(s.solve(structure=st, wavelength=w, thickness=jnp.stack([d]),
                                                theta=th, auto_detune=False).R))
        else:
            def f(u, w, d, th):
                st = u.repeat_interleave(2, dim=1) if two_d else u
                return s.solve(structure=st, wavelength=w, thickness=torch.stack([d]), theta=th,
                               auto_detune=False).R.real.sum()
        return f
    args = tuple(jnp.asarray(x) for x in (u0, wl0, t0, th0))
    ref = jax.grad(make('jax', False), argnums=(0, 1, 2, 3))(*args)
    labels = ('raster', 'wavelength', 'thickness', 'theta')
    f = make('jax', True)
    with warnings.catch_warnings():
        warnings.simplefilter('error', _legacy_wavebit_ExceptionalPointWarning)
        got = jax.grad(f, argnums=(0, 1, 2, 3))(*args)
        for label, g, r in zip(labels, got, ref):
            assert _rel(g, r) < 1e-10, f'jax.grad {label}: {_rel(g, r):.1e}'
        jvp = jax.jvp(lambda w: f(args[0], w, args[2], args[3]), (args[1],), (jnp.asarray(1.0),))[1]
        assert _rel(jvp, ref[1]) < 1e-10
        ft = make('torch', True)
        leaves = [torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in (u0, wl0, t0, th0)]
        ft(*leaves).backward()
        for label, leaf, r in zip(labels, leaves, ref):
            assert _rel(leaf.grad.numpy(), r) < 1e-10, f'torch {label}: {_rel(leaf.grad.numpy(), r):.1e}'
        with forward_ad.dual_level():
            w = forward_ad.make_dual(torch.tensor(wl0, dtype=torch.float64),
                                     torch.tensor(1., dtype=torch.float64))
            tangent = forward_ad.unpack_dual(ft(torch.tensor(u0), w, torch.tensor(t0, dtype=torch.float64),
                                                torch.tensor(th0, dtype=torch.float64))).tangent
        assert _rel(float(tangent), ref[1]) < 1e-10


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('name, dth', LITTROW_CASES)
def test_a_coincidence_under_an_abstract_trace(name, dth, algo):
    """A jointly rebuilt cluster under ``jit``, ``jit(vmap)`` and ``torch.func.vmap`` (values), and
    ``jit(grad(sum vmap))``, ``jit(vmap(grad))``, torch backward through ``torch.func.vmap``
    (``dR/dwl`` per lane), lanes ``rel`` 0, +-1e-12 and 1e-2 (off the cluster): the eager 1D
    rebuild's. Measured <= 1.4e-14 (values) and <= 9.6e-14 (derivatives); with the pair rebuild
    alone, silent and up to 1.9e-3 (values) and 7.5e-3 (derivatives) off. Catches the traced
    gate or slots broken, and the check or the overflow poisoning a live layer."""
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    torch = pytest.importorskip('torch')
    jnp = jax.numpy
    scene = _littrow(SCENES[name], dth)
    wl0 = scene['wavelength']
    lanes = [wl0, wl0 * (1 - 1e-12), wl0 * (1 + 1e-12), wl0 * (1 + 1e-2)]
    ref = np.stack([np.concatenate([n['r'], n['t']]) for n in (
        _conical_cutoff._numbers(_solver(scene, wavelength=w).solve(auto_detune=False))
        for w in lanes)])
    s1 = _solver(scene, backend='jax')
    gref = np.array([float(jax.grad(lambda w: jnp.sum(jnp.real(
        s1.solve(wavelength=w, auto_detune=False).R)))(jnp.asarray(w))) for w in lanes])
    s = _solver(scene, algo=algo, backend='jax', two_d=True)
    row = 1                                      # orders_v = (-1, 0, 1)

    def one(w):
        r = s.solve(wavelength=w, auto_detune=False)
        return jnp.concatenate([r.r_xyz[row].ravel(), r.t_xyz[row].ravel()])
    jf = jax.jit(one)
    values = {'jit': np.stack([np.asarray(jf(jnp.asarray(w))) for w in lanes]),
              'jit(vmap)': np.asarray(jax.jit(jax.vmap(one))(jnp.asarray(lanes)))}
    st = _solver(scene, algo=algo, backend='torch', two_d=True)

    def tone(w):
        r = st.solve(wavelength=w, auto_detune=False)
        return torch.cat([r.r_xyz[row].ravel(), r.t_xyz[row].ravel()])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)          # torch's batching fallbacks
        values['torch.func.vmap'] = torch.func.vmap(tone)(torch.tensor(lanes, dtype=torch.float64)).numpy()
    for label, got in values.items():
        err = np.max(np.abs(got - ref))
        assert np.isfinite(err) and err < 1e-12, f'{label}: {err:.1e}'

    def F(w):
        return jnp.sum(jnp.real(s.solve(wavelength=w, auto_detune=False).R))
    grads = {'jit(grad(sum vmap))': jax.jit(jax.grad(lambda ws: jnp.sum(jax.vmap(F)(ws))))(jnp.asarray(lanes)),
             'jit(vmap(grad))': jax.jit(jax.vmap(jax.grad(F)))(jnp.asarray(lanes))}
    wt = torch.tensor(lanes, dtype=torch.float64, requires_grad=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        torch.func.vmap(lambda w: st.solve(wavelength=w, auto_detune=False).R.real.sum())(wt).sum().backward()
    grads['torch vmap + backward'] = wt.grad.numpy()
    for label, g in grads.items():
        err = np.max(np.abs(np.asarray(g) - gref) / np.abs(gref))
        assert np.isfinite(err) and err < 1e-11, f'{label}: {err:.1e}'


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('name, dth', LITTROW_CASES)
def test_a_coincidence_field_and_its_gradient_are_the_1d_rebuild_s(name, dth, algo):
    """Six components on a 12 x 1 x 8 grid against the 1D field (its global phase removed), and
    ``jax.grad`` of ``sum |E|**2`` at nine points in the raster and the wavelength, eagerly and
    under ``jax.jit``, at a jointly rebuilt cluster. Measured: fields <= 1.3e-13 of ``max|F|``,
    gradients <= 3.3e-12; with the pair rebuild alone up to 7.0e-3 and 5.9e-2."""
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    jnp = jax.numpy
    scene = _littrow(SCENES[name], dth)
    ref = _np(_solver(scene).solve(auto_detune=False, answers='field').reconstruct_field(
        sample_x=12, sample_y=1, sample_z=8))
    got = _np(_solver(scene, algo=algo, two_d=True).solve(auto_detune=False, answers='field')
              .reconstruct_field(sample_x=12, sample_y=1, sample_z=8)).reshape(ref.shape)
    phase = np.exp(1j * np.angle(np.vdot(got.ravel(), ref.ravel())))
    assert np.max(np.abs(got * phase - ref)) < 1e-12 * np.max(np.abs(ref))
    xs, zs = jnp.asarray([-200., 0., 130.]), jnp.asarray([0.13, 0.5, 0.87]) * sum(scene['thickness'])

    def objective(two_d, a):
        s = _solver(scene, algo=a, backend='jax', two_d=two_d)

        def F(u, w):
            st = jnp.repeat(u, 2, axis=1) if two_d else u
            f = s.solve(structure=st, wavelength=w, auto_detune=False, answers='field').reconstruct_field(
                sample_x=xs, sample_y=jnp.asarray([0.]), sample_z=zs)
            return jnp.sum(jnp.abs(f[..., :3]) ** 2)
        return F
    args = (jnp.asarray(scene['structure'], dtype=float), jnp.asarray(scene['wavelength']))
    gref = jax.grad(objective(False, 'ETM'), argnums=(0, 1))(*args)
    F = objective(True, algo)
    for label, g in (('eager', jax.grad(F, argnums=(0, 1))(*args)),
                     ('jit', jax.jit(jax.grad(F, argnums=(0, 1)))(*args))):
        raster = np.max(np.abs(np.asarray(g[0]) - np.asarray(gref[0]))) / np.max(np.abs(np.asarray(gref[0])))
        wavelength = abs(float(g[1]) - float(gref[1])) / abs(float(gref[1]))
        assert raster < 1e-10 and wavelength < 1e-10, f'{label}: raster {raster:.1e}, wl {wavelength:.1e}'


#: A square pillar on a square lattice at normal incidence, Laurent factorization (which keeps the
#: raster's C4v symmetry), truncation (4, 4): two exceptional points of one degenerate pair of
#: orders coincide by symmetry at this wavelength, in a raster that is not y-invariant.
GENUINE_C4V = 641.1137704156972


def _c4v_amplitudes(wavelength, eps=12.25, algo='ETM'):
    st = np.ones((1, 20, 20), dtype=complex if np.iscomplexobj(eps) else float)
    st[0, 6:14, 6:14] = eps
    s = _legacy_wavebit_rcwa(backend='numpy', wavelength=wavelength, period=[1000., 1000.],
                             truncation_order=[4, 4], structure=st, thickness=[300.], top=1.0,
                             bot=2.1025, theta=0.0, phi=0.0, psi=np.pi / 4, layer_algorithm=algo,
                             factorization='laurent')
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        r = s.solve(auto_detune=False)
    return (np.concatenate([_np(r.r_xyz).ravel(), _np(r.t_xyz).ravel()]),
            [w.category for w in rec])


@pytest.mark.parametrize('rel', [0.0, 1e-9, -1e-9])
def test_a_genuinely_2d_coincident_cluster_matches_a_cauchy_oracle(rel, joint_layers, monkeypatch):
    """A coincidence on a genuinely 2D raster (:data:`GENUINE_C4V`): the four modes are one
    cluster, rebuilt jointly, within 1e-12 of a Cauchy oracle and quiet, ETM and SMM. The oracle is
    the mean of the amplitudes over a circle in the pillar's permittivity (radius 1e-2, 16 points:
    the trapezoidal rule, L. N. Trefethen and J. A. C. Weideman, SIAM Rev. 56, 385, 2014, Sec. 12),
    on which the exceptional points move off, with every rebuild closed; it agreed with itself
    to 2.1e-15 across radii 1e-3 / 1e-2 and 16 / 32 points. Measured <= 4.3e-15; with the pair
    rebuild alone 3.3e-8 (ETM) and 1.1e-6 (SMM), loud. Further off the coincidence the two
    doublets leave the cluster radius while their columns stay nearly dependent: silent and a few
    1e-12 off with or without the joint rebuild (``rayleigh-points.md`` §8, limits)."""
    wavelength = GENUINE_C4V * (1 + rel)
    with monkeypatch.context() as m:
        m.setattr(solve_2d, '_EP_SINE', 0.0)
        from wavebit.rcwa._engine import solve_2d_first_order as fo
        m.setattr(fo, '_PARALLEL_SIN', 0.0)
        oracle = np.mean([_c4v_amplitudes(wavelength, 12.25 + 1e-2 * np.exp(2j * np.pi * k / 16))[0]
                          for k in range(16)], axis=0)
    for algo in ('ETM', 'SMM'):
        got, warned = _c4v_amplitudes(wavelength, algo=algo)
        err = np.max(np.abs(got - oracle))
        assert _legacy_wavebit_ExceptionalPointWarning not in warned, f'{algo} rel {rel:g}: warned'
        assert err < 1e-12, f'{algo} rel {rel:g}: {err:.1e} off'
    assert joint_layers and all(valid.sum(axis=1).tolist() == [4] for _, (_, valid), _ in joint_layers)


def test_smm_screens_with_the_thickness_etm_reads(monkeypatch):
    """SMM takes no labels from a layer's thickness, so it hands the screen of nearly coincident
    pairs the optical thickness ETM's labels read: each layer's run, here two identical 150 nm
    slices read as 300 nm. Read where the screen receives it, at a near-coincident offset where
    both slices hold two pairs (rebuilt jointly, so the screen is handed them and screens none,
    but the thickness still reaches it). Catches SMM screening with no thickness (it then keeps
    the calibrated threshold however thick the layer) or with a slice's own, whether
    ``scattering`` stops passing it or ``layer_modes_2d`` stops forwarding it."""
    seen = {'ETM': [], 'SMM': []}
    algo_now = []
    inner = solve_2d._crowded_pairs_imprecise

    def recorded(*args, **kwargs):
        seen[algo_now[0]].append(kwargs.get('k0_d'))
        return inner(*args, **kwargs)
    monkeypatch.setattr(solve_2d, '_crowded_pairs_imprecise', recorded)
    scene = dict(_littrow(SCENES['PA616'], 1e-7), thickness=[150., 150.],
                 structure=np.repeat(SCENES['PA616']['structure'], 2, axis=0))
    for algo in ('ETM', 'SMM'):
        algo_now[:] = [algo]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _solver(scene, algo=algo, two_d=True).solve(auto_detune=False)
    k0_run = 2 * np.pi / scene['wavelength'] * 300.
    assert len(seen['SMM']) == 2 and None not in seen['SMM'], seen
    np.testing.assert_allclose([float(v) for v in seen['SMM']], [k0_run, k0_run], rtol=1e-12)
    np.testing.assert_allclose([float(v) for v in seen['SMM']], [float(v) for v in seen['ETM']],
                               rtol=0, atol=0)


def _two_pair_model(gap, dtype=complex, seed=5):
    """A 10 x 10 layer model with two 2x2 upper-triangular pairs ``gap`` apart, laid out as
    ``_rebuild_defective_pairs`` returns them: ``A = W Q^2 W^-1`` with ``Q = diag(q) + tau``, and
    a unit-norm miss. Two roots far out (31, 47) make ``||A||_2`` (88) large next to the
    cluster's scale (1): a unit miss reads 1.53 through ``W_c W^-1[c, :]``, the exact model
    1.7e-14 (about ``eps ||A||_2``)."""
    rng = np.random.default_rng(seed)
    n2 = 10
    W = np.eye(n2) + 0.2 * (rng.normal(size=(n2, n2)) + 1j * rng.normal(size=(n2, n2)))
    w = np.array([0.3, 0.3, 0.3 + gap, 0.3 + gap, 0.9, 1.4, 2.0, 2.6, 31.0, 47.0], dtype=complex)
    Q = np.diag(np.sqrt(w))
    Q[0, 1], Q[2, 3] = 0.04, -0.03
    A = W @ (Q @ Q) @ np.linalg.inv(W)
    P = rng.normal(size=(n2, n2)) + 1j * rng.normal(size=(n2, n2))
    miss = P / np.linalg.norm(P, 2)
    return [x.astype(dtype) for x in (A, w, W, Q, miss)]


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_the_crowded_pair_screen_reads_the_backward_error(backend):
    """``_crowded_pairs_imprecise`` on a synthetic two-pair layer model. Exact (``A = W Q^2
    W^-1`` to rounding): quiet, however close the pairs. Missing ``A`` by 1e-12 (it reads 0.51
    of ``_EP_BACKWARD``): quiet; by 1e-11 (5.1 of it): loud. Quiet again with one pair masked
    out, or with the two pairs 0.5 apart (not one cluster): the screen is about pairs that share
    a cluster. In single precision the threshold scales with ``eps``, so the exact model stays
    quiet. Catches the screen off, one-pair clusters checked, the threshold moved more than
    about 2x down or 5x up, or the miss read relative to ``||A||_2`` instead of the cluster's
    scale (``||A||_2`` is 88 here, so a 3e-13 relative threshold, 2.6e-11, stays above the
    1.53e-11 read), or the layer's propagation sensitivity ignored (``k0_d``)."""
    if backend == 'jax':
        jax = pytest.importorskip('jax')
        jax.config.update('jax_enable_x64', True)
        import jax.numpy as xp
    elif backend == 'torch':
        pytest.importorskip('torch')
        import array_api_compat.torch as xp
    else:
        xp = np
    a, b = xp.asarray(np.array([0, 2])), xp.asarray(np.array([1, 3]))
    both, one = xp.asarray(np.array([True, True])), xp.asarray(np.array([True, False]))

    def screen(gap, missed, mask=both, dtype=complex, k0_d=None):
        A, w, W, Q, miss = _two_pair_model(gap, dtype)
        A = A + missed * miss
        return solve_2d._crowded_pairs_imprecise(
            xp, xp.asarray(w), xp.asarray(A), xp.asarray(W), xp.asarray(Q), a, b, mask,
            k0_d=k0_d)

    # The model must keep a 3e-13 read relative to ||A||_2 above the 1e-11 case's 1.53e-11.
    assert 3e-13 * np.linalg.norm(_two_pair_model(1e-7)[0], 2) > 2e-11
    assert screen(1e-7, 0.0) is False
    assert screen(1e-7, 1e-12) is False
    assert screen(1e-7, 1e-11) is True
    assert screen(1e-7, 1e-11, one) is False
    assert screen(0.5, 1e-11) is False
    # The cluster's roots are sqrt(0.3): k0 d = 1 gives S = 0.91, under _EP_SENSITIVITY (no
    # change); k0 d = 100 gives S = 91, dividing the threshold by 26 below the 1.53e-12 read.
    assert screen(1e-7, 1e-12, k0_d=1.0) is False
    assert screen(1e-7, 1e-12, k0_d=100.0) is True
    if backend == 'numpy':
        assert screen(1e-7, 0.0, dtype=np.complex64) is False


@pytest.mark.parametrize('rel', [1e-12, 1e-9])
@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_a_graze_on_the_first_order_path_is_not_an_exceptional_point(algo, rel):
    """Near a Rayleigh point an up- and a down-going mode of the first-order operator merge (the
    ``{1, z}`` regime): a non-normal cluster, but across the two directions, which the screen
    keeps apart. Uniform air spacer at ``wavelength = period``, normal incidence: quiet, and the
    2D path's answer. Catches the screen run over both directions at once (it warned here)."""
    kw = dict(backend='numpy', structure=np.full((2, 12, 12), 1.0), thickness=[300., 200.],
              period=[800., 800.], wavelength=800. * (1 + rel), theta=0.0, phi=0.0, top=1.0,
              bot=1.5, psi=0.3, truncation_order=[2, 2], layer_algorithm=algo)
    s = _legacy_wavebit_rcwa(**kw)
    s.formulation = 'first_order'
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        got = s.solve(auto_detune=False)
    ref = _legacy_wavebit_rcwa(**kw).solve(auto_detune=False)
    assert abs(float(np.sum(np.real(_np(got.R)))) - float(np.sum(np.real(_np(ref.R))))) < 1e-12
