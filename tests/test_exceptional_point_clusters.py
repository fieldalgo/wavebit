"""clusters the pair rebuild cannot span are loud; degenerate ones are quiet.

Two order rows of a y-invariant raster meet the x-cutoff at one ``q`` when ``ky0 = Gy / 2``
(``sin(theta) = wavelength / (2 * Py)`` at ``phi = pi/2``): two coincident Jordan blocks. ``eig``
mixes them, so no two of its four vectors are nearly parallel and the pair gate sees nothing,
yet the four span about two dimensions (smallest singular value ~1e-7). Measured before the
screen: 2D up to 5.8e-4 and first-order SMM up to 0.55 in amplitudes, the 2D path silent.
"""
import warnings

import numpy as np
import pytest

import wavebit
from wavebit._core import solve_2d
from test_conical_cutoff import PATTERNED, _np, _pillar, _solver, _vector

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


@pytest.mark.parametrize('first_order', [False], ids=['2d'])
@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('name', list(SCENES))
@pytest.mark.parametrize('dth', [0.0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 3e-4, 1e-3])
def test_coincident_exceptional_points_are_exact_or_loud(name, algo, first_order, dth):
    """Every theta offset from the coincidence: within 1e-12 of the 1D rebuild, or an
    ``ExceptionalPointWarning``. Catches the screen removed (2D silent at 2.0e-6 .. 5.8e-4 and
    first-order PL952 at 1e-8 silent at 0.39). Measured with the screen: 2D loud for
    ``dth <= 1e-8`` (PA616) and ``<= 1e-7`` (PL952, 6.8e-9 under the warning), exact to 6.0e-13
    beyond; first order loud for ``dth <= 1e-4`` (its separation refusal), exact beyond."""
    scene = _littrow(SCENES[name], dth)
    ref = _vector(_solver(scene).solve(auto_detune=False))
    got, warned = _solve_2d(scene, algo, first_order)
    err = np.max(np.abs(got[2:] - ref[2:]))
    loud = wavebit.ExceptionalPointWarning in warned
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
@pytest.mark.parametrize('first_order', [False], ids=['2d'])
@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_degenerate_clusters_stay_quiet(kind, first_order, algo):
    """A uniform layer's degenerate clusters, in whatever basis ``eig`` chose, raise nothing and
    keep the eigenvector arithmetic: the same bits with the screen disabled. The first-order
    prototype's refusal warned on 15 of 120 such scenes; ``||T - mean I||`` as the measure
    warns on the first."""
    kw = dict(UNIFORM_NEAR_DEGENERATE)
    if kind == 'normal incidence':
        kw.update(structure=np.full((1, 8, 8), 3.1), theta=0.0, phi=0.0, truncation_order=[5, 5],
                  wavelength=731.0)

    def run():
        s = wavebit.rcwa(layer_algorithm=algo, **kw)
        if first_order:
            s.formulation = 'first_order'
        return s.solve(auto_detune=False)
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


@pytest.mark.parametrize('delta', [1e-8, 2e-8, 5e-8, 2e-7, 1e-6, 1e-4])
def test_a_pair_beside_an_unpaired_root(delta):
    """The 2D pair gate skips a pair with an unpaired root within ``_EP_SEPARATION`` (1e-7) and
    rebuilds it beyond, where the Newton step leaves at most 8.1e-14 (measured; 2.0e-12 at 1e-8,
    3.5e-13 at 2e-8 with the guard off). Catches the guard removed or widened past 1e-7."""
    A, w, W = _jordan_beside(delta)
    pairs = solve_2d._defective_pairs(np, w, W, np.sqrt(w), 4)
    if delta <= 5e-8:
        assert pairs is None
        return
    assert pairs is not None
    Wr, _, Qm, a, b, _ = solve_2d._rebuild_defective_pairs(np, A, np.eye(12), W, W / np.sqrt(w),
                                                           np.sqrt(w), w, pairs)
    assert np.linalg.norm(A @ Wr - Wr @ (Qm @ Qm)) < 1e-13 * np.linalg.norm(A)
