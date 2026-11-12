"""Defective clusters: rebuilt jointly and exact on the 2D path; loud where no rebuild takes
them; quiet where they are degenerate.

Two order rows of a y-invariant raster meet the x-cutoff at one ``q`` when ``ky0 = Gy / 2``
(``sin(theta) = wavelength / (2 * Py)`` at ``phi = pi/2``): two coincident Jordan blocks. ``eig``
mixes them, so no two of its four vectors are nearly parallel and the pair gate sees nothing,
yet the four span about two dimensions (smallest singular value ~1e-7). Just off the
coincidence the two pairs sit a few 1e-8 .. 1e-6 apart, and one pair rebuild each cannot
separate them. Without the joint rebuild the 2D path is 4.6e-7 .. 1.0e-3 off there (the screens
change no arithmetic: before them it was silent, since then it warns), and first-order SMM up to
1.1. The joint rebuild (``solve_2d._rebuild_defective_clusters``; on the first-order split,
``solve_2d_first_order._regular_clusters``, in each direction) takes the four modes as one
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

#: PA616's grating with the pillar's permittivity and the wavelength root-found together so that
#: an even and an odd family-1 mode of the x-symmetric pillar sit at the x-cutoff at once (Newton
#: on the two parity blocks of the solver's own family-1 operator; its two smallest ``|wA|`` are
#: 6.1e-16 and 8.9e-16). Each order row then holds two coincident Jordan pairs, and at
#: ``ky0 = Gy / 2`` two rows coincide: eight modes at one eigenvalue. Two parameters tuned at once,
#: as a design aiming at a higher-order exceptional point would tune them.
DC612 = dict(PATTERNED['PA616'], wavelength=611.8949901188266,
             structure=_pillar(eps=11.549834654024636))
LITTROW_SCENES = dict(SCENES, DC612=DC612)


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
def test_coincident_exceptional_points_on_the_first_order_path_are_exact(name, algo, dth):
    """The first-order formulation: within 1e-12 of the 1D rebuild, and quiet. Its split holds
    the four modes in each direction, and takes them as one defective cluster, rebuilt jointly
    (``solve_2d_first_order._regular_clusters``). Measured <= 2.0e-14 at 19 offsets of either sign
    on NumPy, JAX and PyTorch; before the joint rebuild the split warned for ``dth <= 1e-4``
    (PA616; 1e-5 PL952) and was up to 1.1 off there (SMM). Catches the joint rebuild removed
    from the split or its gate closed (loud, or off)."""
    scene = _littrow(SCENES[name], dth)
    _exact_and_quiet(scene, algo, f'{name} {algo} dth {dth:g} first order',
                     solve=lambda: _solve_2d(scene, algo, True))


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_the_screen_never_reads_a_jointly_rebuilt_first_order_cluster(algo, monkeypatch):
    """A jointly rebuilt cluster answers to its own check, as on 2D: the screen after the split
    puts each of its members in a group of its own. Its rebuilt columns are a Jordan block's and
    may still be nearly dependent -- PL952's at the coincidence had a smallest singular value of
    0.10 on OpenBLAS's SkylakeX path (0.29 .. 0.64 on five others), and on one CI runner fell
    below the screen's 0.05, which then warned on an exact answer. Here the screen reads every
    ball whatever its columns (its sine at 1): quiet eagerly, and the exact answer under
    ``jax.jit``. Catches the rebuilt members left in the screen's groups."""
    eager, traced = solve_2d._unresolved_defective_cluster, solve_2d._unresolved_cluster_traced
    monkeypatch.setattr(solve_2d, '_unresolved_defective_cluster',
                        lambda xp, w, W, op, sine, group=None: eager(xp, w, W, op, 1.0, group))
    monkeypatch.setattr(solve_2d, '_unresolved_cluster_traced',
                        lambda xp, w, W, op, sine, group=None, slots=1:
                        traced(xp, w, W, op, 1.0, group, slots))
    scene = _littrow(SCENES['PL952'], 0.0)
    _exact_and_quiet(scene, algo, f'PL952 {algo} first order, screen at sine 1',
                     solve=lambda: _solve_2d(scene, algo, True))
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    ref = _vector(_solver(scene).solve(auto_detune=False))
    s = _solver(scene, algo=algo, backend='jax', two_d=True)
    s.formulation = 'first_order'
    rows, grad = _jit_rows(s, scene['wavelength'])
    assert np.isfinite(grad) and np.max(np.abs(rows[2:] - ref[2:])) < 1e-12


def test_the_double_x_cutoff_is_one_and_its_1d_rebuild_is_exact(monkeypatch):
    """The premises of the eight-mode tests below. :data:`DC612` is a double x-cutoff: the 1D
    layer's gate reads two ``|wA|`` under 1e-13. And the 1D rebuild, the oracle, answers there: at
    relative wavelength offsets 0, +-1e-12 and +-1e-9 it is within 1e-12 of a degree-8 fit through
    the family construction at ``|rel|`` in [1e-4, 1e-3], where that construction's rounding is
    1e-12 or less (as ``tests/rcwa/test_conical_cutoff.py::_fit``). Measured 1.2e-13 for ETM and
    SMM, the fit's own residual 4.0e-13; the family construction itself is 0.88 off at the point."""
    from wavebit.rcwa._engine import solve_1d
    from tests.rcwa.test_conical_cutoff import _FIT_OFFSETS, _family_construction
    seen = []
    gate = solve_1d._cutoff_cluster

    def spy(xp, wA, wB, *rest):
        seen.append(np.sort(np.abs(_np(wA)))[:2])
        return gate(xp, wA, wB, *rest)
    monkeypatch.setattr(solve_1d, '_cutoff_cluster', spy)
    wl0 = DC612['wavelength']
    with _family_construction(), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Y = np.array([_vector(_solver(DC612, wavelength=wl0 * (1 + h)).solve(auto_detune=False))
                      for h in _FIT_OFFSETS])
    coeffs = np.polyfit(1e3 * _FIT_OFFSETS, Y, 8)
    for rel in (0.0, 1e-12, -1e-12, 1e-9, -1e-9):
        del seen[:]
        truth = np.polyval(coeffs, 1e3 * rel)
        for algo in ('ETM', 'SMM'):
            got = _vector(_solver(DC612, algo=algo, wavelength=wl0 * (1 + rel))
                          .solve(auto_detune=False))
            assert np.max(np.abs(got - truth)) < 1e-12, (
                f'rel {rel:g} {algo}: {np.max(np.abs(got - truth)):.1e} from the fit')
        if rel == 0.0:
            assert max(seen[0]) < 1e-13, f'not a double cutoff: {seen[0]}'


#: Offsets from :data:`DC612`: of ``theta`` (relative, from ``ky0 = Gy / 2``), of the pillar's
#: permittivity and of the wavelength, one at a time.
DOUBLE_OFFSETS = ([('theta', x) for x in (0.0, 1e-9, -1e-9, 1e-7, 1e-5, 1e-4, 1e-3)]
                  + [('eps', x) for x in (1e-12, -1e-12, 1e-8, -1e-8, 1e-6, 1e-4)]
                  + [('wavelength', x) for x in (1e-12, 1e-8, -1e-8, 1e-6, -1e-5, 1e-4)])


def _double(kind, x):
    if kind == 'theta':
        return _littrow(DC612, x)
    if kind == 'eps':
        return _littrow(dict(DC612, structure=_pillar(eps=11.549834654024636 * (1 + x))), 0.0)
    return _littrow(dict(DC612, wavelength=DC612['wavelength'] * (1 + x)), 0.0)


@pytest.mark.parametrize('first_order', [False, True], ids=['2d', 'first_order'])
@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('kind, x', DOUBLE_OFFSETS)
def test_eight_modes_at_a_double_x_cutoff_are_exact(kind, x, algo, first_order):
    """At :data:`DC612` and around it, both formulations are within 1e-12 of the 1D rebuild, and
    quiet: the eight modes are one defective cluster, rebuilt jointly (with a cluster of four
    beside it, the third order row's two pairs; on the first-order split, in each direction).
    Measured <= 2.1e-14. With clusters of at most four modes (until 2026-11-12) every offset
    measured up to theta 1e-4, eps 1e-6 and wavelength 1e-5 was loud and off: the 2D path up to
    1.2e-5 (ETM) and 9.8e-4 (SMM), the first-order split up to 1.5e-5 (ETM) and 1.5 (SMM).
    Catches the cluster size cap lowered below eight (loud, and off)."""
    scene = _double(kind, x)
    _exact_and_quiet(scene, algo, f'DC612 {kind} {x:g} {algo}' + (' first order' * first_order),
                     solve=lambda: _solve_2d(scene, algo, first_order))


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
    from wavebit.rcwa._engine import solve_2d_first_order
    monkeypatch.setattr(solve_2d, '_rebuild_defective_clusters', refused)
    monkeypatch.setattr(solve_2d_first_order, '_regular_clusters', refused)
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
    assert valid.tolist() == [[True, True, True, False]]          # no ball of five: four lanes
    assert sorted(members[valid].tolist()) == [0, 1, 2]
    rng = np.random.default_rng(4)
    U, _ = np.linalg.qr(rng.normal(size=(9, 9)) + 1j * rng.normal(size=(9, 9)))
    B = U @ np.diag([0.3, 0.3 + 2e-5, 0.3 + 4e-5, 0.8, 1.1, 1.5, 1.9, 2.4, 3.0]) @ U.conj().T
    w, W = _first(B, 0.3)
    assert solve_2d._defective_clusters(np, w, W, np.sqrt(w), None, lambda: B, 2) is None


def test_the_split_does_not_trust_eig_s_order(monkeypatch):
    """A cluster smaller than its slot pads its last lanes at a stand-in eigenvalue, and the
    position in which ``eig`` hands those eigenpairs back is not promised (LAPACK's returns the
    decoupled ones last; another solver need not). With every small ``eig`` of the split handing
    its pairs back reversed, the joint rebuild is the same: exact and quiet -- a three-mode
    cluster in four lanes, and clusters of six and five modes in eight (two and three padded
    lanes). Catches the padded pairs placed by position, or only one of them put back."""
    original = solve_2d._eig_raw
    monkeypatch.setattr(solve_2d, '_eig_raw', lambda T: tuple(x[..., ::-1] for x in original(T)))
    models = [(f'beside {d:g}', _jordan_beside(d)[0], 3) for d in (0.0, 1e-9, 5e-8)]
    models += [('three pairs', _coincident_pairs(3), 6), ('short chain', _spaced([0.9e-4, 1.8e-4]), 5)]
    for label, A, size in models:
        _, clusters, rebuilt, warned = _layer(A)
        assert np.asarray(clusters[1]).sum() == size, label
        assert not warned and _backward(A, rebuilt) < 1e-13, label


def _collapse_the_split_eig(monkeypatch, sine):
    """Make the split's ``eig`` of each cluster restriction hand back, at every eigenvalue of
    geometric multiplicity two or more, eigenvectors all within ``sine`` of one direction: each
    one a true eigenvector (in the eigenspace, to rounding), as ``eig``'s collapsed vectors are.
    ``DC612`` at ``theta`` 1e-4 came out that way on OpenBLAS's Haswell, Zen and Prescott kernels
    (sines of 1.3e-3, best ``cond(X)`` 1833). The quotients' ``eig`` is left alone."""
    original = solve_2d._eig_raw

    def collapsed(T):
        lam, E = original(T)
        if T.ndim != 3:
            return lam, E
        lam, E, T = np.array(lam), np.array(E), np.asarray(T)
        for s in range(T.shape[0]):
            seen = np.zeros(lam.shape[1], bool)
            for i in range(lam.shape[1]):
                group = np.flatnonzero(~seen & (np.abs(lam[s] - lam[s, i])
                                                <= 1e-6 * max(1.0, abs(lam[s, i]))))
                seen[group] = True
                if group.size < 2:
                    continue
                shifted = T[s] - lam[s, group].mean() * np.eye(T.shape[1])
                _, sv, Vh = np.linalg.svd(shifted)
                null = Vh[sv <= 1e-12 * np.linalg.norm(T[s])].conj().T
                if null.shape[1] < 2:
                    continue
                for n, g in enumerate(group):
                    c = null[:, 0] + sine * (n + 1) * null[:, 1 + n % (null.shape[1] - 1)]
                    E[s, :, g] = c / np.linalg.norm(c)
        return lam, E
    monkeypatch.setattr(solve_2d, '_eig_raw', collapsed)


@pytest.mark.parametrize('model', ['two pairs', 'four pairs', 'DC612 2d', 'DC612 first order'])
def test_a_split_from_collapsed_eigenvectors_is_re_derived(model, monkeypatch):
    """With the split's ``eig`` handing back collapsed eigenvectors (:func:`_collapse_the_split_eig`,
    sines of 1e-6), every split from them has ``cond(X)`` near 1e6 and the check fails. The split
    is tried again with the collapsed vectors re-derived (``_spread_eigenvectors``): exact and
    quiet -- two and four coincident Jordan pairs, and ``DC612`` at ``theta`` 1e-4 on both
    formulations (the scene that warned on an exact answer on three OpenBLAS kernels). Catches the
    retry removed, its vectors not orthogonalised, or kept without the residual test."""
    _collapse_the_split_eig(monkeypatch, 1e-6)
    if model.startswith('DC612'):
        scene, first_order = _double('theta', 1e-4), model.endswith('first order')
        _exact_and_quiet(scene, 'ETM', f'collapsed split, {model}',
                         solve=lambda: _solve_2d(scene, 'ETM', first_order))
        return
    A = _coincident_pairs(2 if model == 'two pairs' else 4)
    _, clusters, rebuilt, warned = _layer(A)
    assert clusters is not None and not warned, model
    assert _backward(A, rebuilt) < 1e-13, model


def test_the_split_retry_keeps_the_split_the_check_passes(monkeypatch):
    """The second split replaces the first only where the check would read less of it: its
    ``cond(X)`` times its dropped part. Instrumented: collapsed eigenvectors at sines of 1.5e-2
    (best ``cond(X)`` above the retry's 32, a split the check still passes), and re-derived
    vectors replaced by an orthonormal basis of the same columns, each mixed with 1e-6 of the
    next -- well conditioned, but some no longer eigenvectors. Kept by ``cond(X)`` alone that
    split fails the check, loud; kept by the miss the first split stays, exact and quiet. Catches
    the choice made by ``cond(X)`` alone."""
    _collapse_the_split_eig(monkeypatch, 1.5e-2)
    split, retry, first = solve_2d._cluster_split, solve_2d._SPLIT_RETRY, []

    def recorded(xp, T, valid, pad):
        # The first split's cond(X), with the retry held off.
        monkeypatch.setattr(solve_2d, '_SPLIT_RETRY', float('inf'))
        first.append(float(np.max(np.asarray(split(xp, T, valid, pad)[2]))))
        monkeypatch.setattr(solve_2d, '_SPLIT_RETRY', retry)
        return split(xp, T, valid, pad)
    monkeypatch.setattr(solve_2d, '_cluster_split', recorded)

    def not_eigenvectors(xp, T, lam, E, live):
        Q = np.linalg.qr(np.asarray(E))[0]
        Q = Q + 1e-6 * np.roll(Q, 1, axis=-1)
        return Q / np.linalg.norm(Q, axis=-2, keepdims=True)
    monkeypatch.setattr(solve_2d, '_spread_eigenvectors', not_eigenvectors)
    A = _coincident_pairs(2)
    _, clusters, rebuilt, warned = _layer(A)
    assert first and min(first) > retry, first
    assert clusters is not None and not warned
    assert _backward(A, rebuilt) < 1e-13


def test_re_derived_eigenvectors_are_orthogonal_eigenvectors():
    """At a double eigenvalue that is not defective every vector of the eigenspace is an
    eigenvector. Given two of them within a sine of 1e-9, ``_spread_eigenvectors`` keeps the first
    and re-derives the second orthogonal to it, an eigenvector to rounding. A Jordan pair's second
    member has no eigenvector orthogonal to its first and keeps ``eig``'s, to the bit; so do the
    plain lanes. Catches the re-derived vector not orthogonalised against the earlier lane, or
    kept without the residual test."""
    rng = np.random.default_rng(3)
    U, _ = np.linalg.qr(rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6)))
    J = np.diag([0.3, 0.3, 1.4, 1.4, 2.0, 2.6]).astype(complex)
    J[2, 3] = 0.05
    T = U @ J @ U.conj().T
    lam, E = np.linalg.eig(T)
    order = np.argsort(lam.real)
    lam, E = lam[order], E[:, order] / np.linalg.norm(E[:, order], axis=0)
    lam[:2] = 0.3
    E[:, 0] = U[:, 0]
    E[:, 1] = (U[:, 0] + 1e-9 * U[:, 1]) / np.linalg.norm(U[:, 0] + 1e-9 * U[:, 1])
    live = np.ones((1, 6), bool)
    out = np.asarray(solve_2d._spread_eigenvectors(np, T[None], lam[None], E[None], live))[0]
    assert np.array_equal(out[:, 0], E[:, 0]) and np.array_equal(out[:, 2:], E[:, 2:])
    assert abs(np.vdot(out[:, 0], out[:, 1])) < 1e-12
    assert np.linalg.norm(T @ out[:, 1] - 0.3 * out[:, 1]) < 1e-14


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


def _spaced(steps, seed=0):
    """A 12 x 12 product operator: a Jordan pair at 0.3 (coupling 0.05) beside a root 5e-8 away --
    a pair the gate refuses -- and modes at ``0.3 + steps``, the rest spread out to 2.8, turned by
    a random unitary."""
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.normal(size=(12, 12)) + 1j * rng.normal(size=(12, 12)))
    d = [0.3, 0.3, 0.3 + 5e-8] + [0.3 + s for s in steps]
    J = np.diag(d + list(np.linspace(0.7, 2.8, 12 - len(d)))).astype(complex)
    J[0, 1] = 0.05
    return U @ J @ U.conj().T


def test_a_chain_is_not_taken():
    """The refused pair and its root, then seven modes 0.9e-4 apart beyond them: ten modes within
    6.3e-4. At :data:`~solve_2d._EP_JOINT_RADIUS` their ball holds ten, more than a cluster may;
    at :data:`~solve_2d._EP_CLUSTER` the first four are a ball of the lowest member, but the
    fourth's ball holds five -- a chain, not a closed ball. The joint rebuild does not take it (a
    closed ball is what keeps every divisor of its Newton step at least the cluster radius); the
    unresolved screen reads it. Catches the closed-ball condition dropped."""
    A = _spaced([0.9e-4 * j for j in range(1, 8)])
    w, W = _first(A, 0.3)
    q = np.sqrt(w)
    pairs = solve_2d._defective_pairs(np, w, W, q, 4)
    assert pairs is None
    assert solve_2d._defective_clusters(np, w, W, q, pairs, lambda: A, 2) is None
    assert solve_2d._unresolved_defective_cluster(np, w, W, lambda: A, solve_2d._EP_SINE)


@pytest.mark.parametrize('seed', range(4))
def test_a_short_chain_is_one_wide_cluster(seed):
    """The refused pair and its root with two modes 0.9e-4 and 1.8e-4 beyond: a chain at
    :data:`~solve_2d._EP_CLUSTER`, but one closed ball of five at the joint radius, which the joint
    rebuild takes whole -- exact (measured <= 1.7e-15 of ``||A||`` on four bases) and quiet. With
    at most four modes a cluster (until 2026-11-12) it was a chain left to the screen: loud.
    Catches the cluster size cap lowered below five."""
    A = _spaced([0.9e-4, 1.8e-4], seed)
    pairs, clusters, rebuilt, warned = _layer(A)
    assert pairs is None and np.asarray(clusters[1]).sum(axis=1).tolist() == [5]
    assert not warned and _backward(A, rebuilt) < 1e-13


@pytest.mark.parametrize('seed', range(3))
@pytest.mark.parametrize('gap', [1e-7, 1e-6, 1e-5])
def test_two_gated_pairs_beside_a_root_are_one_cluster(gap, seed):
    """Two Jordan pairs ``gap`` apart, both gated, beside a root 5e-5 away, in a basis of
    condition 10: a ball of five, which the joint rebuild takes whole -- exact (measured <= 1.9e-15
    of ``||A||``) and quiet. With at most four modes a cluster (until 2026-11-12) each pair was
    rebuilt alone and the backward-error screen warned. Catches the cluster size cap lowered
    below five."""
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.normal(size=(14, 14)) + 1j * rng.normal(size=(14, 14)))
    Vh, _ = np.linalg.qr(rng.normal(size=(14, 14)) + 1j * rng.normal(size=(14, 14)))
    P = U @ np.diag(np.geomspace(1, 10, 14)) @ Vh
    lead = [0.3, 0.3, 0.3 + gap, 0.3 + gap, 0.3 + 5e-5]
    J = np.diag(lead + list(np.linspace(0.7, 2.8, 9))).astype(complex)
    J[0, 1], J[2, 3] = 0.05, 0.04
    A = P @ J @ np.linalg.inv(P)
    pairs, clusters, rebuilt, warned = _layer(A)
    assert np.asarray(clusters[1]).sum(axis=1).tolist() == [5]
    assert not warned and _backward(A, rebuilt) < 1e-13


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
    measured backward errors 3.7e-4 .. 6.1e-2 on four CPU arithmetic paths (9.7e-6 .. 1.9e-2
    after the second step), and the joint
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


def _poor_start(err, seed, gap=1e-2, both=False):
    """:func:`_two_clusters_and_a_pair`'s operator with the first cluster's four ``eig`` columns
    (with ``both``, the second's too) replaced by an orthonormal basis of its invariant subspace
    mixed by a matrix of singular values ``1, 1, 1e-3, 1e-3``, plus a perturbation of
    ``1e-3 * err``: a start whose subspace is off by about ``err``, as ``eig``'s is where its
    columns are nearly dependent. ``(A, w, W)``."""
    A = _two_clusters_and_a_pair(gap, seed)
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16)))
    Vh, _ = np.linalg.qr(rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16)))
    P = U @ np.diag(np.geomspace(1, 10, 16)) @ Vh
    w, W = np.linalg.eig(A)
    W = W.copy()
    rng = np.random.default_rng(100 + seed)
    for centre, cols in ((0.3, slice(0, 4)), (0.3 + gap, slice(4, 8)))[:2 if both else 1]:
        near = np.sort(np.argsort(np.abs(w - centre))[:4])
        B = np.linalg.qr(P[:, cols])[0]
        X, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
        Y, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
        R = rng.normal(size=(16, 4)) + 1j * rng.normal(size=(16, 4))
        W[:, near] = (B @ X @ np.diag([1, 1, 1e-3, 1e-3]) @ Y
                      + 1e-3 * err * R / np.linalg.norm(R, axis=0))
        W[:, near] /= np.linalg.norm(W[:, near], axis=0)
    return A, w, W


@pytest.mark.parametrize('seed', [1, 2, 3])
@pytest.mark.parametrize('err, both', [(1e-6, False), (3e-6, False), (1e-4, False), (1e-6, True)],
                         ids=['1e-6', '3e-6', '1e-4', 'both 1e-6'])
def test_a_poor_start_takes_a_second_step(err, both, seed):
    """One Newton step leaves a remainder of second order in the start's error, which the check
    reads; a second step, the chord method's (Kelley 1995, ch. 5) with no new solve, leaves third
    order. A start off by 1e-6 .. 3e-6 (:func:`_poor_start`): one step measured 8.0e-14 ..
    1.5e-12 of ``||A||``, loud from 3e-6; two steps <= 1.2e-14 and quiet, on seven CPU arithmetic
    paths. Off by 1e-4, two steps leave 4.6e-11 .. 4.3e-10 and the check still warns. Both
    clusters off by 1e-6: two steps <= 1.5e-15, and 2.7e-13 with the residual expansion's rows
    between the clusters left out of the second step's residual. Catches the second step dropped,
    or its residual missing a block's rows."""
    A, w, W = _poor_start(err, seed, both=both)
    q = np.sqrt(w)
    pairs = solve_2d._defective_pairs(np, w, W, q, 8)
    clusters = solve_2d._defective_clusters(np, w, W, q, pairs, lambda: A, 4)
    assert np.asarray(clusters[1]).sum(axis=1).tolist() == [4, 4]
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        rebuilt = solve_2d._rebuild_defective_pairs(np, A, np.eye(16, dtype=complex), W, W / q, q,
                                                    w, pairs, clusters=clusters)
    loud = any(issubclass(r.category, _legacy_wavebit_ExceptionalPointWarning) for r in rec)
    if err >= 1e-4:
        assert loud
        return
    assert not loud
    assert _backward(A, rebuilt) < 3e-14


@pytest.mark.parametrize('name', list(SCENES))
@pytest.mark.parametrize('dth', [0.0, 1e-8, -1.5e-7, 1e-6])
def test_a_first_step_at_rounding_keeps_its_basis(name, dth, monkeypatch):
    """Where one step already reaches rounding (the y-Littrow clusters), the second moves the basis
    by rounding alone and is not kept (``_EP_STEP_GAIN``): the answer is the one-step answer
    (``_EP_STEP_GAIN = 0``), bit for bit, and not the two-step one (``_EP_STEP_GAIN = inf``),
    whose bits differ. At an exact coincidence ``eig(T)``'s eigenvectors are sensitive to rounding
    in ``T``, and on two CPU arithmetic paths such a basis turned the split's ``cond(X)`` from 1.9
    to 2.6e3 and from 3.4 to 156, each loud. Catches the second step kept always, or on any
    decrease."""
    scene = _littrow(SCENES[name], dth)

    def solve():
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            r = _solver(scene, algo='SMM', two_d=True).solve(auto_detune=False)
        return [_np(getattr(r, k)).tobytes() for k in ('R', 'T', 'r_xyz', 't_xyz')], rec
    got, rec = solve()
    assert not [w for w in rec if issubclass(w.category, _legacy_wavebit_ExceptionalPointWarning)]
    monkeypatch.setattr(solve_2d, '_EP_STEP_GAIN', 0.0)
    assert solve()[0] == got
    monkeypatch.setattr(solve_2d, '_EP_STEP_GAIN', float('inf'))
    assert solve()[0] != got


def test_the_step_choice_reads_the_split(monkeypatch):
    """The second step is kept by the check's miss, which reads the split as well as the residual.
    Instrumented: a start off by 1e-5 (:func:`_poor_start`), where one step fails the check
    (residual) and two pass it, with the two-step basis's split reporting a ``cond(X)`` 1e8 times
    its own -- a split the check would refuse. The one-step basis is kept, and the answer is
    loud; reading the residual alone would keep the two-step basis, quiet. Catches the choice made
    by the residual alone."""
    A, w, W = _poor_start(1e-5, 1)
    q = np.sqrt(w)
    pairs = solve_2d._defective_pairs(np, w, W, q, 8)
    clusters = solve_2d._defective_clusters(np, w, W, q, pairs, lambda: A, 4)
    original, calls = solve_2d._cluster_split, []

    def split(xp, T, valid, pad):
        X, J, cond = original(xp, T, valid, pad)
        calls.append(None)
        return X, J, cond * (1e8 if len(calls) == 2 else 1.0)
    monkeypatch.setattr(solve_2d, '_cluster_split', split)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        rebuilt = solve_2d._rebuild_defective_pairs(np, A, np.eye(16, dtype=complex), W, W / q, q,
                                                    w, pairs, clusters=clusters)
    assert len(calls) == 2
    assert any(issubclass(r.category, _legacy_wavebit_ExceptionalPointWarning) for r in rec)
    assert _backward(A, rebuilt) > 1e-13


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


@pytest.mark.parametrize('seed', [1, 2, 3, 4, 5])
@pytest.mark.parametrize('gap', [2e-4, 5e-4, 1e-3, 1e-2])
def test_two_clusters_and_a_pair_in_a_general_basis(gap, seed):
    """Two defective clusters and a pair, ``gap`` apart, in a basis that does not decouple them:
    every block's Newton step reads the others' rows -- a cluster's rows in another cluster's
    correction by the Kronecker solve, in a pair's by ``k x k`` solves. One step leaves a
    remainder of second order over the gap: at 2e-4 up to 4.3e-13 of ``||A||``, which the check
    on one path read as a miss (residual 3.24e-12 against 3.00e-12). The second step takes it
    out: <= 2.0e-15 on seven CPU arithmetic paths, quiet at every gap. ``Omega_R W = V Q``; ``Q``'s coupling squares to
    exactly zero; the entry list names exactly its nonzero entries. On the y-Littrow scenes the
    inter-block rows change no answer (the blocks live in orthogonal order rows) and only the
    check reads their absence (:func:`test_two_clusters_in_one_layer` then warns); here they
    change the answer. Catches the Kronecker or the pair-to-cluster rows dropped, and the second
    step's residual missing them."""
    A = _two_clusters_and_a_pair(gap, seed)
    pairs, clusters, rebuilt, warned = _layer(A, cap=8, slots=4)
    assert not warned
    assert np.asarray(clusters[1]).sum(axis=1).tolist() == [4, 4]
    W, V, Q, a, b, mask = (np.asarray(x) for x in rebuilt)
    assert _backward(A, rebuilt) < 1e-13
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


def _split_doublets(s, seed=1, crowd=False):
    """Two copies of a Jordan pair at 0.3 split to ``0.3 +- s`` (coupling 0.05 above the diagonal,
    ``s**2 / 0.05`` below): two degenerate doublets ``2 s`` apart, each column nearly parallel to
    its copy's other one (sine about ``2 s / 0.05``), as off a C4v coincidence. With ``crowd`` also
    :func:`_two_clusters_and_a_pair`'s two clusters and pair, 1e-3 apart, at 1.2. In a basis of
    condition 10."""
    rng = np.random.default_rng(seed)
    d = [0.3] * 4 + ([1.2] * 4 + [1.201] * 4 + [1.199] * 2 if crowd else []) + [
        0.8, 1.6, 1.9, 2.2, 2.6, 3.0, 3.4, 3.8]
    n = len(d)
    U, _ = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    Vh, _ = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    P = U @ np.diag(np.geomspace(1, 10, n)) @ Vh
    J = np.diag(d).astype(complex)
    for i in (0, 2):
        J[i, i + 1], J[i + 1, i] = 0.05, s * s / 0.05
    if crowd:
        J[4, 5], J[6, 7], J[8, 9], J[10, 11], J[12, 13] = 0.05, 0.04, 0.03, 0.06, 0.05
    return P @ J @ np.linalg.inv(P)


@pytest.mark.parametrize('seed', [1, 2, 3])
@pytest.mark.parametrize('s', [1e-6, 1e-4, 3e-4])
def test_split_doublets_are_one_cluster(s, seed):
    """Two doublets ``2 s`` apart whose columns stay nearly dependent (:func:`_split_doublets`):
    no pair is gated (each mode's nearest eigenvalue is its degenerate twin, at a sine above
    ``_EP_SINE``), and beyond ``_EP_CLUSTER`` the narrow ball holds a doublet only. The wide ball
    (``_EP_JOINT_RADIUS``) takes the four as one cluster while their columns are dependent to
    within ``_EP_SINE``: out to ``s = 3e-4`` on every path measured (at 1e-3, the sine's edge, on
    some, exact there too). Measured <= 2.4e-15 of ``||A||`` on seven paths, quiet. Catches the wide level removed or
    narrowed below ``2 s``."""
    A = _split_doublets(s, seed)
    pairs, clusters, rebuilt, warned = _layer(A, cap=8, slots=4)
    assert pairs is None and not warned
    assert np.asarray(clusters[1]).sum(axis=1).tolist() == [4]
    assert _backward(A, rebuilt) < 1e-13


@pytest.mark.parametrize('model', ['beside 1e-9', 'two clusters', 'split doublets',
                                   'split doublets beside two clusters'])
def test_the_traced_cluster_gate_takes_the_eager_set(model):
    """Under ``jax.jit`` and ``torch.func.vmap`` the gate returns static slots with masks as data;
    the live slots must be the eager clusters, member for member, from either radius. Beside the
    split doublets, :func:`_two_clusters_and_a_pair`'s clusters sit in one wide ball of ten (too
    many for one cluster), so the narrow level takes them: three clusters, one wide and two
    narrow. Catches the slots ranked or masked differently from the eager gate, and a level's
    seeds, or the narrow balls inside a taken wide one, masked differently under a trace."""
    A, want_sizes = {'beside 1e-9': (_jordan_beside(1e-9)[0], [3]),
                     'two clusters': (_two_clusters_and_a_pair(1e-3), [4, 4]),
                     'split doublets': (_split_doublets(1e-4), [4]),
                     'split doublets beside two clusters': (_split_doublets(1e-4, crowd=True),
                                                            [4, 4, 4])}[model]
    w, W = np.linalg.eig(A)
    q = np.sqrt(w)
    eager = solve_2d._defective_clusters(np, w, W, q, solve_2d._defective_pairs(np, w, W, q, 8),
                                         lambda: A, 4)

    def sets(members, valid):
        members, valid = np.asarray(members), np.asarray(valid)
        return sorted(tuple(sorted(m[v].tolist())) for m, v in zip(members, valid) if v.any())
    want = sets(*eager[:2])
    assert [len(c) for c in want] == want_sizes
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


@pytest.mark.parametrize('first_order', [False, True], ids=['2d', 'first_order'])
def test_more_clusters_than_the_trace_holds_is_loud(monkeypatch, first_order):
    """Under ``jax.jit`` the clusters are rebuilt up to a static count of slots; the ``(5, 2)``
    layer at the coincidence holds two (four in the first-order split, two in each direction),
    so one slot fewer must give NaN, never a cluster left to the pair rebuild; enough slots are
    exact."""
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    jnp = jax.numpy
    scene = _littrow(SCENES['PL952'], 0.0)
    ref = _vector(_solver(scene).solve(auto_detune=False))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        row = list(_np(_solver_v2(scene).solve(auto_detune=False).orders_v)).index(0)
    original = solve_2d._defective_clusters
    for slots, finite in ((3, False), (4, True)) if first_order else ((1, False), (2, True)):
        monkeypatch.setattr(solve_2d, '_defective_clusters',
                            lambda *a, slots=slots, **k: original(*a[:-1], slots, **k))
        s = _solver_v2(scene, backend='jax')
        if first_order:
            s.formulation = 'first_order'
        R, T, r, t = (np.asarray(x) for x in jax.jit(lambda w: [
            getattr(s.solve(wavelength=w, auto_detune=False), k)
            for k in ('R', 'T', 'r_xyz', 't_xyz')])(jnp.asarray(scene['wavelength'])))
        if not finite:
            assert np.all(np.isnan(R)), slots
            continue
        got = np.concatenate([r[row].ravel(), t[row].ravel()])
        assert np.max(np.abs(got - ref[2:])) < 1e-12


def _sum_R(s):
    import jax.numpy as jnp
    return lambda w: jnp.sum(jnp.real(s.solve(wavelength=w, auto_detune=False).R))


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_two_clusters_a_direction_on_the_first_order_path(algo):
    """The first-order split of the ``(5, 2)`` layer at the coincidence holds two four-mode
    clusters in each direction beside row 2's pair: five coupled rows a direction, more than the
    traced exponential gathers by default (``modal_gradient._COUPLED_CAP``, four). Eagerly and
    under ``jax.jit`` within 1e-12 of the 1D rebuild and quiet, and ``jit(grad)`` of ``sum R``
    within 1e-10 of the eager ``jax.grad``, relative. Measured TBD. Catches the split's coupled
    count not reaching the exponential's traced rule (its tangent NaN)."""
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    jnp = jax.numpy
    scene = _littrow(SCENES['PL952'], 0.0)
    ref = _vector(_solver(scene).solve(auto_detune=False))
    s = _solver_v2(scene, algo, backend='jax')
    s.formulation = 'first_order'
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        eager = _row(s.solve(auto_detune=False))
        g = float(jax.grad(_sum_R(s))(jnp.asarray(scene['wavelength'])))
    assert not any(issubclass(r.category, _legacy_wavebit_ExceptionalPointWarning) for r in rec)
    assert np.max(np.abs(eager[2:] - ref[2:])) < 1e-12
    rows, grad = _jit_rows(s, scene['wavelength'])
    assert np.max(np.abs(rows[2:] - ref[2:])) < 1e-12
    assert abs(grad - g) <= 1e-10 * max(1.0, abs(g)), (grad, g)
    # The field reconstruction reads the count off the layer record's length.
    xs, zs = jnp.asarray([-200., 130.]), jnp.asarray([0.3, 0.7]) * sum(scene['thickness'])

    def field(w):
        f = s.solve(wavelength=w, auto_detune=False, answers='field').reconstruct_field(
            sample_x=xs, sample_y=jnp.asarray([0.]), sample_z=zs)
        return jnp.sum(jnp.abs(f[..., :3]) ** 2)
    wl = jnp.asarray(scene['wavelength'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gf = float(jax.grad(field)(wl))
        gj = float(jax.jit(jax.grad(field))(wl))
    assert abs(gj - gf) <= 1e-10 * abs(gf), (gj, gf)


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


LITTROW_CASES = [('PL952', 0.0), ('PA616', 1e-8), ('DC612', 0.0)]


@pytest.mark.parametrize('first_order', [False, True], ids=['2d', 'first_order'])
@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('name, dth', LITTROW_CASES + [
    pytest.param('PL952', 7e-8, marks=pytest.mark.slow)])
def test_derivatives_at_a_coincidence_are_the_1d_rebuild_s(name, dth, algo, first_order):
    """``d(sum R)`` in the raster, the wavelength, the thickness and ``theta`` at a jointly
    rebuilt cluster: ``jax.grad``, ``jax.jvp``, torch reverse and ``forward_ad`` against
    ``jax.grad`` of the 1D rebuild. Measured <= 1.8e-13; with the pair rebuild alone 3.3e-8 ..
    3.1e-3. On the first-order split, whose cluster block of the generator is coupled the same
    way, <= 1.2e-13. At :data:`DC612`'s eight modes <= 2.9e-13 on both. Catches an entry list that leaves out a cluster's coupling (the generator
    and inverse rules then drop its terms while the values stay exact)."""
    from tests.rcwa.test_conical_cutoff import _rel
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    torch = pytest.importorskip('torch')
    from torch.autograd import forward_ad
    jnp = jax.numpy
    scene = _littrow(LITTROW_SCENES[name], dth)
    u0, t0, th0, wl0 = (np.asarray(scene['structure'], float), float(scene['thickness'][0]),
                        float(scene['theta']), float(scene['wavelength']))

    def make(backend, two_d):
        s = _solver(scene, algo=algo if two_d else 'ETM', backend=backend, two_d=two_d)
        if two_d and first_order:
            s.formulation = 'first_order'
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


@pytest.mark.parametrize('first_order', [False, True], ids=['2d', 'first_order'])
@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('name, dth', LITTROW_CASES)
def test_a_coincidence_under_an_abstract_trace(name, dth, algo, first_order):
    """A jointly rebuilt cluster under ``jit``, ``jit(vmap)`` and ``torch.func.vmap`` (values), and
    ``jit(grad(sum vmap))``, ``jit(vmap(grad))``, torch backward through ``torch.func.vmap``
    (``dR/dwl`` per lane), lanes ``rel`` 0, +-1e-12 and 1e-2 (off the cluster): the eager 1D
    rebuild's. Measured <= 1.4e-14 (values) and <= 9.6e-14 (derivatives); with the pair rebuild
    alone, silent and up to 1.9e-3 (values) and 7.5e-3 (derivatives) off. On the first-order
    split <= 1.6e-14 and <= 1.2e-13. At :data:`DC612`'s eight modes <= 1.9e-14 and <= 6.2e-14 on
    both; NaN with clusters of at most four modes. Catches the traced gate or slots broken, and the check or
    the overflow poisoning a live layer."""
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    torch = pytest.importorskip('torch')
    jnp = jax.numpy
    scene = _littrow(LITTROW_SCENES[name], dth)
    wl0 = scene['wavelength']
    lanes = [wl0, wl0 * (1 - 1e-12), wl0 * (1 + 1e-12), wl0 * (1 + 1e-2)]
    ref = np.stack([np.concatenate([n['r'], n['t']]) for n in (
        _conical_cutoff._numbers(_solver(scene, wavelength=w).solve(auto_detune=False))
        for w in lanes)])
    s1 = _solver(scene, backend='jax')
    gref = np.array([float(jax.grad(lambda w: jnp.sum(jnp.real(
        s1.solve(wavelength=w, auto_detune=False).R)))(jnp.asarray(w))) for w in lanes])
    s = _solver(scene, algo=algo, backend='jax', two_d=True)
    st = _solver(scene, algo=algo, backend='torch', two_d=True)
    if first_order:
        s.formulation = st.formulation = 'first_order'
    row = 1                                      # orders_v = (-1, 0, 1)

    def one(w):
        r = s.solve(wavelength=w, auto_detune=False)
        return jnp.concatenate([r.r_xyz[row].ravel(), r.t_xyz[row].ravel()])
    jf = jax.jit(one)
    values = {'jit': np.stack([np.asarray(jf(jnp.asarray(w))) for w in lanes]),
              'jit(vmap)': np.asarray(jax.jit(jax.vmap(one))(jnp.asarray(lanes)))}

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


@pytest.mark.parametrize('first_order', [False, True], ids=['2d', 'first_order'])
@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('name, dth', LITTROW_CASES)
def test_a_coincidence_field_and_its_gradient_are_the_1d_rebuild_s(name, dth, algo, first_order):
    """Six components on a 12 x 1 x 8 grid against the 1D field (its global phase removed), and
    ``jax.grad`` of ``sum |E|**2`` at nine points in the raster and the wavelength, eagerly and
    under ``jax.jit``, at a jointly rebuilt cluster. Measured: fields <= 1.3e-13 of ``max|F|``,
    gradients <= 3.3e-12; with the pair rebuild alone up to 7.0e-3 and 5.9e-2. On the
    first-order split <= 3.5e-14 and <= 1.0e-12."""
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    jnp = jax.numpy
    scene = _littrow(LITTROW_SCENES[name], dth)
    ref = _np(_solver(scene).solve(auto_detune=False, answers='field').reconstruct_field(
        sample_x=12, sample_y=1, sample_z=8))
    s2 = _solver(scene, algo=algo, two_d=True)
    if first_order:
        s2.formulation = 'first_order'
    got = _np(s2.solve(auto_detune=False, answers='field')
              .reconstruct_field(sample_x=12, sample_y=1, sample_z=8)).reshape(ref.shape)
    phase = np.exp(1j * np.angle(np.vdot(got.ravel(), ref.ravel())))
    assert np.max(np.abs(got * phase - ref)) < 1e-12 * np.max(np.abs(ref))
    xs, zs = jnp.asarray([-200., 0., 130.]), jnp.asarray([0.13, 0.5, 0.87]) * sum(scene['thickness'])

    def objective(two_d, a):
        s = _solver(scene, algo=a, backend='jax', two_d=two_d)
        if two_d and first_order:
            s.formulation = 'first_order'

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


def _c4v_solver(wavelength, eps=12.25, algo='ETM', factorization='laurent', first_order=False,
                backend='numpy'):
    st = np.ones((1, 20, 20), dtype=complex if np.iscomplexobj(eps) else float)
    st[0, 6:14, 6:14] = eps
    s = _legacy_wavebit_rcwa(backend=backend, wavelength=wavelength, period=[1000., 1000.],
                             truncation_order=[4, 4], structure=st, thickness=[300.], top=1.0,
                             bot=2.1025, theta=0.0, phi=0.0, psi=np.pi / 4, layer_algorithm=algo,
                             factorization=factorization)
    if first_order:
        s.formulation = 'first_order'
    return s


def _c4v_amplitudes(wavelength, eps=12.25, algo='ETM', factorization='laurent', first_order=False):
    s = _c4v_solver(wavelength, eps, algo, factorization, first_order)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        r = s.solve(auto_detune=False)
    return (np.concatenate([_np(r.r_xyz).ravel(), _np(r.t_xyz).ravel()]),
            [w.category for w in rec])


def _c4v_exact_and_quiet(wavelength, factorization, joint_layers, monkeypatch):
    """ETM and SMM within 1e-12 of the Cauchy oracle and quiet, every jointly rebuilt layer one
    cluster of four. The oracle is the mean of the amplitudes over a circle in the pillar's
    permittivity (radius 1e-2, 16 points: the trapezoidal rule, L. N. Trefethen and J. A. C.
    Weideman, SIAM Rev. 56, 385, 2014, Sec. 12), on which the exceptional points move off, with
    every rebuild closed; it agreed with itself to 2.1e-15 across radii 1e-3 / 1e-2 and 16 / 32
    points."""
    with monkeypatch.context() as m:
        m.setattr(solve_2d, '_EP_SINE', 0.0)
        from wavebit.rcwa._engine import solve_2d_first_order as fo
        m.setattr(fo, '_PARALLEL_SIN', 0.0)
        oracle = np.mean([_c4v_amplitudes(wavelength, 12.25 + 1e-2 * np.exp(2j * np.pi * k / 16),
                                          factorization=factorization)[0] for k in range(16)],
                         axis=0)
    joint_layers.clear()
    for algo in ('ETM', 'SMM'):
        got, warned = _c4v_amplitudes(wavelength, algo=algo, factorization=factorization)
        err = np.max(np.abs(got - oracle))
        assert _legacy_wavebit_ExceptionalPointWarning not in warned, f'{algo}: warned'
        assert err < 1e-12, f'{algo}: {err:.1e} off'
    assert joint_layers and all(valid.sum(axis=1).tolist() == [4] for _, (_, valid), _ in joint_layers)


@pytest.mark.parametrize('rel', [0.0, 1e-9, -1e-9])
def test_a_genuinely_2d_coincident_cluster_matches_a_cauchy_oracle(rel, joint_layers, monkeypatch):
    """A coincidence on a genuinely 2D raster (:data:`GENUINE_C4V`): the four modes are one
    cluster, rebuilt jointly, within 1e-12 of a Cauchy oracle (:func:`_c4v_exact_and_quiet`) and
    quiet, ETM and SMM. Measured <= 4.3e-15; with the pair rebuild alone 3.3e-8 (ETM) and 1.1e-6
    (SMM), loud. Off the coincidence: :func:`test_off_a_c4v_coincidence_the_doublets_are_one_cluster`."""
    _c4v_exact_and_quiet(GENUINE_C4V * (1 + rel), 'laurent', joint_layers, monkeypatch)


#: C4v coincidences of :func:`_c4v_amplitudes`'s pillar: :data:`GENUINE_C4V`, a second under the
#: Laurent factorization, and one under NVF.
C4V = [('laurent', GENUINE_C4V), ('laurent', 623.6028730062790), ('nvf', 603.4978930437392)]


@pytest.mark.parametrize('rel', [1e-8, -1e-8, -3e-7, 1e-6, -1e-6])
@pytest.mark.parametrize('factorization, coincidence', C4V)
def test_off_a_c4v_coincidence_the_doublets_are_one_cluster(factorization, coincidence, rel,
                                                             joint_layers, monkeypatch):
    """Off a C4v coincidence the two exceptional points split into degenerate doublets whose
    spread grows like the square root of the offset (1.6e-4 .. 2.6e-4 at relative offsets 1e-8 ..
    1e-7) while their four columns stay nearly dependent, so beyond ``_EP_CLUSTER`` only the wide
    ball (``_EP_JOINT_RADIUS``) holds them. Within 1e-12 of the Cauchy oracle
    (:func:`_c4v_exact_and_quiet`), quiet, ETM and SMM, the four rebuilt as one cluster.
    Measured <= 3.3e-14 on seven CPU arithmetic paths, five coincidences and offsets 1e-9 ..
    3e-6; with the narrow ball alone silent and up to 9.1e-12 off. Catches the wide level
    removed."""
    _c4v_exact_and_quiet(coincidence * (1 + rel), factorization, joint_layers, monkeypatch)


def _c4v_first_order_oracle(wavelength, factorization, monkeypatch):
    """:func:`_c4v_exact_and_quiet`'s Cauchy circle, on the first-order split."""
    with monkeypatch.context() as m:
        m.setattr(solve_2d, '_EP_SINE', 0.0)
        from wavebit.rcwa._engine import solve_2d_first_order as fo
        m.setattr(fo, '_PARALLEL_SIN', 0.0)
        return np.mean([_c4v_amplitudes(wavelength, 12.25 + 1e-2 * np.exp(2j * np.pi * k / 16),
                                        factorization=factorization, first_order=True)[0]
                        for k in range(16)], axis=0)


@pytest.mark.parametrize('rel', [0.0, 1e-7])
@pytest.mark.parametrize('factorization, coincidence', C4V)
def test_a_c4v_cluster_on_the_first_order_split(factorization, coincidence, rel, monkeypatch):
    """The C4v pillar's coincidences, and the doublets just off them, on the first-order split:
    one four-mode cluster a direction, rebuilt jointly, ETM and SMM within 1e-12 of the Cauchy
    oracle and quiet. Measured <= 2.0e-14 at relative offsets 0 and +-1e-9 at the three
    coincidences and +-1e-8 .. 1e-5 at two of them; before the joint rebuild loud and up to
    3.7e-4 off at the
    coincidence, and with its gate closed silent and up to 3.1e-13 off at 1e-7, where the
    screen's narrow balls hold two modes each. Catches the gate closed on the first-order split
    or its wide radius lost."""
    wavelength = coincidence * (1 + rel)
    oracle = _c4v_first_order_oracle(wavelength, factorization, monkeypatch)
    for algo in ('ETM', 'SMM'):
        got, warned = _c4v_amplitudes(wavelength, algo=algo, factorization=factorization,
                                      first_order=True)
        err = np.max(np.abs(got - oracle))
        assert _legacy_wavebit_ExceptionalPointWarning not in warned, f'{algo}: warned'
        assert err < 1e-12, f'{algo}: {err:.1e} off'


@pytest.mark.slow
def test_a_c4v_cluster_the_trace_leaves_over_is_loud(monkeypatch):
    """At a relative 1e-7 off the C4v coincidence a cluster the trace's slots leave over would go
    to nothing that reads it -- no pair is gated, and the screen's narrow balls hold two modes
    each -- and the first-order split was 2.2e-13 off with the joint gate closed, silently. So
    under ``jax.jit`` one slot, where the split holds a cluster in each direction, must answer
    NaN, and two are within 1e-12 of the Cauchy oracle. Catches the cluster gate's overflow
    dropped from the first-order split."""
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    jnp = jax.numpy
    wavelength = GENUINE_C4V * (1 + 1e-7)
    oracle = _c4v_first_order_oracle(wavelength, 'laurent', monkeypatch)
    original = solve_2d._defective_clusters
    for slots, finite in ((1, False), (2, True)):
        monkeypatch.setattr(solve_2d, '_defective_clusters',
                            lambda *a, slots=slots, **k: original(*a[:-1], slots, **k))
        s = _c4v_solver(wavelength, first_order=True, backend='jax')
        r, t = jax.jit(lambda w: [getattr(s.solve(wavelength=w, auto_detune=False), k)
                                  for k in ('r_xyz', 't_xyz')])(jnp.asarray(wavelength))
        got = np.concatenate([np.asarray(r).ravel(), np.asarray(t).ravel()])
        if not finite:
            assert np.all(np.isnan(got)), slots
            continue
        assert np.max(np.abs(got - oracle)) < 1e-12


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


# ------------------------------------------------------------------ the unresolved screen, traced

def _coincident_pairs(count, root=None, seed=0):
    """``count`` coincident Jordan pairs at 0.3 (couplings 0.05, 0.04, 0.03, 0.06, 0.07), a plain
    root at ``0.3 + root`` beside them if given, the rest spread out, in a basis of condition 10.
    Up to four pairs, or three and a root within ``_EP_CLUSTER``, are a defective cluster the joint
    rebuild takes (measured <= 1.8e-15 of ``||A||``); five pairs, or four and a root, hold more
    than eight modes, which no rebuild takes."""
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.normal(size=(14, 14)) + 1j * rng.normal(size=(14, 14)))
    Vh, _ = np.linalg.qr(rng.normal(size=(14, 14)) + 1j * rng.normal(size=(14, 14)))
    P = U @ np.diag(np.geomspace(1, 10, 14)) @ Vh
    lead = [0.3] * (2 * count) + ([0.3 + root] if root is not None else [])
    J = np.diag(lead + list(np.linspace(0.7, 2.8, 14 - len(lead)))).astype(complex)
    for p, c in zip(range(count), (0.05, 0.04, 0.03, 0.06, 0.07)):
        J[2 * p, 2 * p + 1] = c
    return P @ J @ np.linalg.inv(P)


def _a_chain():
    """:func:`test_a_chain_is_not_taken`'s operator: a refused pair, then seven modes 0.9e-4
    apart."""
    return _spaced([0.9e-4 * j for j in range(1, 8)])


def _degenerate_balls(centres, seed=3):
    """Two exactly degenerate pairs 1e-5 apart at each centre, each pair in a nearly parallel
    basis (:func:`test_a_degenerate_ball_in_a_parallel_basis_is_not_taken`): normal four-mode balls
    whose unit columns are nearly dependent. ``(A, w, W)``, the basis as given."""
    rng = np.random.default_rng(seed)
    w = []
    for c in centres:
        w += [c, c, c + 1e-5, c + 1e-5]
    w = np.array(w + list(np.linspace(3.3, 4.0, 10 - len(w))), dtype=complex)
    n = len(w)
    U, _ = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    V = np.eye(n, dtype=complex)
    for i in range(0, 4 * len(centres), 2):
        V[:, i + 1] = (V[:, i] + 1e-4 * V[:, i + 1]) / np.linalg.norm(V[:, i] + 1e-4 * V[:, i + 1])
    return U @ np.diag(w) @ U.conj().T, w, U @ V


def _kept(A):
    """The basis the 2D path keeps for ``A`` and the group its screen reads (jointly rebuilt
    members each apart): ``(A, w, W, group)``, ``w`` ``eig``'s eigenvalues."""
    pairs, clusters, rebuilt, _ = _layer(A)
    w, W = np.linalg.eig(A)
    if rebuilt is None:
        return A, w, W, None
    group = None
    if clusters is not None:
        ii = np.arange(len(w))
        joint = np.isin(ii, np.asarray(clusters[0])[np.asarray(clusters[1])])
        group = np.where(joint, ii + 1, 0)
    return A, w, np.asarray(rebuilt[0]), group


TRACED_SCREEN_CASES = {
    'a chain': (lambda: _kept(_a_chain()), True),
    'a length-3 Jordan chain': (lambda: _kept(_chain(0)), True),
    'a length-3 Jordan chain, every mode apart': (
        lambda: _kept(_chain(0))[:3] + (np.arange(12),), False),
    'five coincident pairs': (lambda: _kept(_coincident_pairs(5)), True),
    'four coincident pairs and a root': (lambda: _kept(_coincident_pairs(4, 5e-5)), True),
    'four coincident pairs, rebuilt jointly': (lambda: _kept(_coincident_pairs(4)), False),
    'two coincident pairs and a root, rebuilt jointly': (
        lambda: _kept(_coincident_pairs(2, 5e-5)), False),
    'two coincident pairs, rebuilt jointly': (lambda: _kept(_coincident_pairs(2)), False),
    'a Jordan pair beside a root, rebuilt': (lambda: _kept(_jordan_beside(1e-6)[0]), False),
    'a degenerate ball in a parallel basis': (lambda: _degenerate_balls([0.3]) + (None,), False),
    'a lone Jordan pair as eig left it': (
        lambda: (lambda A: (A,) + np.linalg.eig(A) + (None,))(_jordan_beside(1e-2)[0]), False),
}


def _traced_screen(backend, A, w, W, group, slots=2):
    """:func:`solve_2d._unresolved_cluster_traced` under ``jax.jit``, or under ``torch.func.vmap``
    over a batch of one."""
    if backend == 'jax':
        jax = pytest.importorskip('jax')
        jax.config.update('jax_enable_x64', True)
        jnp = jax.numpy
        g = None if group is None else jnp.asarray(group)
        return bool(jax.jit(lambda w_, W_, A_: solve_2d._unresolved_cluster_traced(
            jnp, w_, W_, lambda: A_, solve_2d._EP_SINE, g, slots))(
            jnp.asarray(w), jnp.asarray(W), jnp.asarray(A)))
    torch = pytest.importorskip('torch')
    import array_api_compat.torch as tx
    g = None if group is None else torch.as_tensor(group)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')            # functorch's batching-rule notices for conj
        out = torch.func.vmap(lambda w_, W_, A_: solve_2d._unresolved_cluster_traced(
            tx, w_, W_, lambda: A_, solve_2d._EP_SINE, g, slots))(
            *(torch.as_tensor(np.asarray(x))[None] for x in (w, W, A)))
    return bool(out[0])


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('name', list(TRACED_SCREEN_CASES))
def test_the_traced_screen_reads_what_the_eager_screen_reads(name, backend):
    """Under an abstract trace the unresolved screen cannot walk a cluster; it reads every mode's
    ball of at most eight modes instead, and must reach the eager verdict: loud on a chain, a
    length-3 Jordan chain after the pair rebuild, and defective clusters of nine and ten modes
    (none of which any rebuild takes); quiet on a cluster rebuilt jointly (its members apart), a
    pair rebuilt beside a root, a degenerate ball in a nearly parallel basis, the length-3 chain
    with every mode in a group of its own (as a jointly rebuilt member is, or the other direction
    of the first-order split), and a lone Jordan pair in ``eig``'s basis (a ball of two: the pair
    gate's business, not a cluster). Measured: the same verdict as the eager screen on 200 random
    clustered layers as well (129 loud). Catches the traced screen returning a constant, its
    singular-value or defect test dropped, the group ignored, and balls of two read."""
    build, loud = TRACED_SCREEN_CASES[name]
    A, w, W, group = build()
    eager = solve_2d._unresolved_defective_cluster(np, w, W, lambda: A, solve_2d._EP_SINE, group)
    assert eager is loud
    assert _traced_screen(backend, A, w, W, group) is loud



@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_the_traced_crowded_pair_screen_reads_what_the_eager_one_reads(backend):
    """:func:`test_the_crowded_pair_screen_reads_the_backward_error`'s verdicts, under ``jax.jit``
    (``k0_d`` a traced argument) and ``torch.func.vmap``: exact model quiet; a 1e-12 miss quiet, a
    1e-11 miss loud; one pair masked out, or the pairs 0.5 apart, quiet; ``k0_d`` 100 loud on the
    1e-12 miss, 1 quiet; single precision quiet. Catches the traced screen off, its one-pair or
    threshold or sensitivity terms lost, or ``k0_d`` read only when concrete."""
    a, b = np.array([0, 2]), np.array([1, 3])
    both, one = np.array([True, True]), np.array([True, False])
    if backend == 'jax':
        jax = pytest.importorskip('jax')
        jax.config.update('jax_enable_x64', True)
        jnp = jax.numpy

        def traced(A, w, W, Q, mask, k0_d):
            f = jax.jit(lambda w_, A_, W_, Q_, m_, k_: solve_2d._crowded_pairs_traced(
                jnp, w_, A_, W_, Q_, jnp.asarray(a), jnp.asarray(b), m_,
                k0_d=None if k0_d is None else k_, slots=2))
            return bool(f(*(jnp.asarray(x) for x in (w, A, W, Q, mask)),
                          jnp.asarray(0.0 if k0_d is None else k0_d)))
    else:
        torch = pytest.importorskip('torch')
        import array_api_compat.torch as tx

        def traced(A, w, W, Q, mask, k0_d):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')        # functorch's batching-rule notices
                out = torch.func.vmap(lambda w_, A_, W_, Q_, k_: solve_2d._crowded_pairs_traced(
                    tx, w_, A_, W_, Q_, torch.as_tensor(a), torch.as_tensor(b),
                    torch.as_tensor(mask), k0_d=None if k0_d is None else k_, slots=2))(
                    *(torch.as_tensor(np.asarray(x))[None] for x in (w, A, W, Q)),
                    torch.tensor([0.0 if k0_d is None else k0_d], dtype=torch.float64))
            return bool(out[0])

    def screen(gap, missed, mask=both, dtype=complex, k0_d=None):
        A, w, W, Q, miss = _two_pair_model(gap, dtype)
        return traced(A + missed * miss, w, W, Q, mask, k0_d)

    assert screen(1e-7, 0.0) is False
    assert screen(1e-7, 1e-12) is False
    assert screen(1e-7, 1e-11) is True
    assert screen(1e-7, 1e-11, one) is False
    assert screen(0.5, 1e-11) is False
    assert screen(1e-7, 1e-12, k0_d=1.0) is False
    assert screen(1e-7, 1e-12, k0_d=100.0) is True
    assert screen(1e-7, 0.0, dtype=np.complex64) is False


def test_more_crowded_balls_than_slots_are_loud():
    """Two clusters of two pairs each (at 0.3 and 0.9), an exact layer model: the traced
    backward-error screen is quiet with a slot for each, and loud with one -- a cluster it cannot
    read answers NaN, never silence. Catches its overflow term dropped."""
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    jnp = jax.numpy
    rng = np.random.default_rng(7)
    n2 = 12
    W = np.eye(n2) + 0.2 * (rng.normal(size=(n2, n2)) + 1j * rng.normal(size=(n2, n2)))
    w = np.array([0.3, 0.3, 0.3 + 1e-7, 0.3 + 1e-7, 0.9, 0.9, 0.9 + 1e-7, 0.9 + 1e-7,
                  1.4, 2.0, 2.6, 3.1], dtype=complex)
    Q = np.diag(np.sqrt(w))
    Q[0, 1], Q[2, 3], Q[4, 5], Q[6, 7] = 0.04, -0.03, 0.05, 0.02
    A = W @ (Q @ Q) @ np.linalg.inv(W)
    a, b, mask = np.array([0, 2, 4, 6]), np.array([1, 3, 5, 7]), np.ones(4, dtype=bool)
    assert solve_2d._crowded_pairs_imprecise(np, w, A, W, Q, a, b, mask) is False

    def traced(slots):
        return bool(jax.jit(lambda *x: solve_2d._crowded_pairs_traced(jnp, *x, slots=slots))(
            *(jnp.asarray(x) for x in (w, A, W, Q, a, b, mask))))
    assert traced(2) is False
    assert traced(1) is True


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_one_ball_takes_one_slot_and_more_balls_than_slots_are_loud(backend):
    """Every member of a closed ball reads the same ball; it takes one slot. A degenerate
    four-mode ball in a nearly parallel basis with one slot: quiet, as eagerly. Two such balls
    with one slot: more candidates than slots, so the trace answers NaN where the eager screen
    is quiet -- loud, never silent, as the cluster gate's overflow. Catches the one-seed-per-ball
    rule dropped (four seeds overflow one slot) and the overflow term dropped."""
    A, w, W = _degenerate_balls([0.3])
    assert not solve_2d._unresolved_defective_cluster(np, w, W, lambda: A, solve_2d._EP_SINE)
    assert _traced_screen(backend, A, w, W, None, slots=1) is False
    A, w, W = _degenerate_balls([0.3, 1.4])
    assert not solve_2d._unresolved_defective_cluster(np, w, W, lambda: A, solve_2d._EP_SINE)
    assert _traced_screen(backend, A, w, W, None, slots=1) is True
    assert _traced_screen(backend, A, w, W, None, slots=2) is False


def _jit_rows(s, wavelength):
    """``_row`` of ``s.solve`` under ``jax.jit`` in the wavelength, and its ``jit(grad)`` of
    ``sum R``. A trace cannot warn: its verdict is in the answer."""
    import types

    import jax
    jnp = jax.numpy
    keys = ('R', 'T', 'r_xyz', 't_xyz')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        orders_v = s.solve(auto_detune=False).orders_v
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        out = jax.jit(lambda w: [getattr(s.solve(wavelength=w, auto_detune=False), k)
                                 for k in keys])(jnp.asarray(wavelength))
        grad = jax.jit(jax.grad(lambda w: jnp.sum(jnp.real(
            s.solve(wavelength=w, auto_detune=False).R))))(jnp.asarray(wavelength))
    assert not any(issubclass(r.category, _legacy_wavebit_ExceptionalPointWarning) for r in rec)
    return _row(types.SimpleNamespace(orders_v=orders_v, **dict(zip(keys, out)))), float(grad)


@pytest.mark.parametrize('name,algo,dth', [
    ('PL952', 'SMM', 1e-8), ('PA616', 'ETM', 1e-7), ('PL952', 'ETM', 0.0),
    pytest.param('PA616', 'SMM', 3e-4, marks=pytest.mark.slow)])
def test_coincident_exceptional_points_on_the_first_order_path_are_exact_under_jit(
        name, algo, dth):
    """The first-order joint rebuild under ``jax.jit``, its clusters in traced slots: the answer
    within 1e-12 of the 1D rebuild, and ``jit(grad)`` of ``sum R`` within 1e-11 of the 1D path's
    ``jax.grad``, relative. Measured <= 1.7e-14 and <= 1.0e-13 (both scenes, ETM and SMM, five
    offsets 0 .. 1e-4). Before the joint rebuild the eager solve warned at every offset here but
    3e-4 and the trace answered NaN; before the traced screens it was up to 0.69 off, silently.
    Catches the traced cluster slots not wired into the first-order split."""
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    jnp = jax.numpy
    scene = _littrow(SCENES[name], dth)
    one = _solver(scene, backend='jax')
    ref = _vector(one.solve(auto_detune=False))
    g = float(jax.grad(_sum_R(one))(jnp.asarray(scene['wavelength'])))
    s = _solver(scene, algo=algo, backend='jax', two_d=True)
    s.formulation = 'first_order'
    rows, grad = _jit_rows(s, scene['wavelength'])
    assert np.max(np.abs(rows[2:] - ref[2:])) < 1e-12
    assert abs(grad - g) <= 1e-11 * abs(g), (grad, g)


@pytest.mark.parametrize('dth,says', [(0.0, 'no rebuild takes'), (3e-7, 'could not separate')],
                         ids=['unresolved', 'crowded'])
@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_what_the_2d_screens_warn_on_is_nan_under_jit(algo, dth, says, monkeypatch):
    """The 2D wiring of both traced screens: PL952 with the joint gate closed. At the coincidence
    its four modes stay in ``eig``'s nearly dependent basis (the unresolved screen's case, 2.6e-4
    .. 8.7e-4 off eagerly); at 3e-7 the pair gate takes the two pairs one at a time and they miss
    the operator (the backward-error screen's case, 2.8e-11 off). Eagerly that screen's warning;
    under ``jax.jit`` NaN in every answer and ``jit(grad)``, where the trace was silent. Catches
    either traced verdict not applied to the 2D layer's ``Q``."""
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    monkeypatch.setattr(solve_2d, '_defective_clusters', lambda *a, **k: None)
    scene = _littrow(SCENES['PL952'], dth)
    s = _solver(scene, algo=algo, two_d=True)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        s.solve(auto_detune=False)
    said = [str(r.message) for r in rec
            if issubclass(r.category, _legacy_wavebit_ExceptionalPointWarning)]
    assert len(said) == 1 and says in said[0], said
    rows, grad = _jit_rows(_solver(scene, algo=algo, backend='jax', two_d=True),
                           scene['wavelength'])
    assert np.all(np.isnan(rows)) and np.isnan(grad)


@pytest.mark.parametrize('kind', ['near-degenerate', 'normal incidence'])
@pytest.mark.parametrize('first_order', [False, True], ids=['2d', 'first_order'])
@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_degenerate_clusters_keep_their_bits_under_jit(kind, first_order, algo, monkeypatch):
    """:func:`test_degenerate_clusters_stay_quiet` under ``jax.jit``: the traced screen reads a
    uniform layer's degenerate clusters, in whatever basis ``eig`` chose, as quiet, and a quiet
    layer keeps its bits -- the same ``jit`` values and ``jit(grad)`` as with the traced screen
    forced off. At normal incidence the first-order split's degenerate clusters are sixteen modes
    in a basis that is not orthonormal: read as eight-mode balls once per lane set, or with the
    pair gate's four slots, they overflowed and the layer came back NaN. On the first-order split
    the joint rebuild's slots run dead there, and the bits are the same with its gate closed too.
    Catches a degenerate cluster read as defective, the slots or the one-seed-per-ball rule
    narrowed, and a dead slot that moves a bit."""
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    jnp = jax.numpy
    kw = dict(UNIFORM_NEAR_DEGENERATE, backend='jax', layer_algorithm=algo)
    if kind == 'normal incidence':
        kw.update(structure=np.full((1, 8, 8), 3.1), theta=0.0, phi=0.0, truncation_order=[5, 5],
                  wavelength=731.0)

    def run():
        s = _legacy_wavebit_rcwa(**kw)
        if first_order:
            s.formulation = 'first_order'
        wl = jnp.asarray(kw['wavelength'])
        vals = jax.jit(lambda w: [getattr(s.solve(wavelength=w, auto_detune=False), k)
                                  for k in ('R', 'T', 'r_xyz', 't_xyz')])(wl)
        grad = jax.jit(jax.grad(lambda w: jnp.sum(jnp.real(
            s.solve(wavelength=w, auto_detune=False).R))))(wl)
        return [np.asarray(v) for v in vals] + [np.asarray(grad)]
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        screened = run()
    assert all(np.all(np.isfinite(v)) for v in screened)
    monkeypatch.setattr(solve_2d, '_unresolved_cluster_traced',
                        lambda xp, w, *a, **k: xp.zeros((), dtype=xp.bool))
    unscreened = run()
    for got, ref in zip(screened, unscreened):
        assert got.dtype == ref.dtype and got.tobytes() == ref.tobytes()
    if first_order:
        # The first-order split's joint rebuild runs its traced slots dead here: the same bits
        # with its gate closed, the coupled exponential's count with it.
        monkeypatch.setattr(solve_2d, '_defective_clusters', lambda *a, **k: None)
        for got, ref in zip(run(), unscreened):
            assert got.dtype == ref.dtype and got.tobytes() == ref.tobytes()
