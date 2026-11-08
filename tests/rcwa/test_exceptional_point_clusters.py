"""clusters the pair rebuild cannot span are loud; degenerate ones are quiet.

Two order rows of a y-invariant raster meet the x-cutoff at one ``q`` when ``ky0 = Gy / 2``
(``sin(theta) = wavelength / (2 * Py)`` at ``phi = pi/2``): two coincident Jordan blocks. ``eig``
mixes them, so no two of its four vectors are nearly parallel and the pair gate sees nothing,
yet the four span about two dimensions (smallest singular value ~1e-7). Measured before the
screen: 2D up to 5.8e-4 and first-order SMM up to 0.55 in amplitudes, the 2D path silent.
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


@pytest.mark.parametrize('first_order', [False, True], ids=['2d', 'first_order'])
@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('name', list(SCENES))
@pytest.mark.parametrize('dth', [0.0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 3e-4, 1e-3])
def test_coincident_exceptional_points_are_exact_or_loud(name, algo, first_order, dth):
    """Every theta offset from the coincidence: within 1e-12 of the 1D rebuild, or an
    ``ExceptionalPointWarning``. Catches the screen removed (2D silent at 2.0e-6 .. 5.8e-4 and
    first-order PL952 at 1e-8 silent at 0.39). Measured with the screens on seven CPU
    arithmetic paths: 2D loud by the cluster screen for ``dth <= 1e-8`` (PA616) and ``<= 1e-7``
    (PL952), and by the backward-error screen (``_EP_BACKWARD``) at PA616 1e-7 (up to 2.0e-12
    off, silent before it) and, on most paths, at PA616 1e-6 and PL952 1e-6 and 1e-5, which were
    exact (the screen is conservative); every quiet point within 1.7e-14. First order loud for
    ``dth <= 1e-4`` (its separation refusal), exact beyond."""
    scene = _littrow(SCENES[name], dth)
    ref = _vector(_solver(scene).solve(auto_detune=False))
    got, warned = _solve_2d(scene, algo, first_order)
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
        s = _legacy_wavebit_rcwa(layer_algorithm=algo, **kw)
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


#: Offsets on both sides of a y-Littrow coincidence where, in one scene or the other, two rebuilt
#: pairs sit a few 1e-8 .. 1e-7 apart and one Newton step cannot separate them. Near there the
#: grid of ``test_coincident_exceptional_points_are_exact_or_loud`` has only 1e-8, 1e-7 and 1e-6,
#: and no negative offsets.
NEAR_COINCIDENT = [5e-8, 7e-8, 1.5e-7, 2e-7, 4e-7, -5e-8, -1.5e-7]


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('name', list(SCENES))
@pytest.mark.parametrize('dth', NEAR_COINCIDENT)
def test_nearly_coincident_pairs_are_exact_or_loud(name, algo, dth):
    """Within 1e-12 of the 1D rebuild, or an ``ExceptionalPointWarning``, where one Newton step
    cannot separate two rebuilt pairs. Before ``_EP_BACKWARD`` the 2D path was silent here and
    1.2e-12 .. 1.7e-10 off on one or more of seven CPU arithmetic paths (PL952 1.5e-7 .. 6e-7 and
    -1.5e-7 .. -5e-7, PA616 5e-8 .. 1.2e-7 and -5e-8 .. -1e-7); the band's edge moved with the
    path, and on one GitHub runner it reached the 1e-7 point of the test above. Catches the
    backward-error screen removed, or its threshold raised past the band, on a machine whose
    arithmetic leaves one of these points off."""
    scene = _littrow(SCENES[name], dth)
    ref = _vector(_solver(scene).solve(auto_detune=False))
    got, warned = _solve_2d(scene, algo, False)
    err = np.max(np.abs(got[2:] - ref[2:]))
    loud = _legacy_wavebit_ExceptionalPointWarning in warned
    assert loud or err < 1e-12, f'{name} {algo} dth {dth:g}: {err:.1e} off, silently'


#: The same at higher truncations, each scene at its own exceptional-point wavelength there (the
#: zero of the 1D family's cutoff root): offsets the 2D path without the backward-error screen
#: got silently wrong on most of four CPU arithmetic paths, 2.2e-11 .. 1.3e-10 off at worst
#: (PA616 at truncation 20, 5.5e-7: 2.7e-12). That point also stayed silent on two of them with
#: the miss read relative to ``||A||_2``.
HIGHER_TRUNCATION = [('PA616', 10, 638.8575175362382, 2e-7), ('PL952', 10, 956.9477951876295, -3e-7),
                     ('PA616', 15, 639.0691826791563, 3e-7), ('PL952', 15, 957.4006448059722, 5e-7),
                     ('PA616', 20, 639.1092327634085, 4e-7), ('PA616', 20, 639.1092327634085, 5.5e-7),
                     ('PL952', 20, 957.5144881024428, 6e-7)]


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('name, truncation, wavelength, dth', HIGHER_TRUNCATION)
def test_nearly_coincident_pairs_at_higher_truncation_are_exact_or_loud(name, truncation,
                                                                        wavelength, dth, algo):
    """Within 1e-12 of the 1D rebuild, or loud, at truncations 10 .. 20, where the band is wider
    (PL952 out to 1.5e-6) and the silent errors larger (up to 1.0e-9) than at truncation 5.
    Catches the backward-error screen removed, on any of the four paths measured."""
    scene = dict(_littrow(dict(SCENES[name], wavelength=wavelength), dth), truncation=truncation)
    ref = _vector(_solver(scene).solve(auto_detune=False))
    got, warned = _solve_2d(scene, algo, False)
    err = np.max(np.abs(got[2:] - ref[2:]))
    loud = _legacy_wavebit_ExceptionalPointWarning in warned
    assert loud or err < 1e-12, f'{name} to {truncation} {algo} dth {dth:g}: {err:.1e} off, silently'


#: PA616 thickened to 10 um and 3 um, the y period set so the coincidence sits at ``ky0 = 0.025``:
#: the cluster's roots are ``|q| = 0.025``, so its propagation sensitivity ``k0 d / (2|q|)`` is
#: 2.0e3 (6.1e2 at 3 um) against the 3.47 the threshold was calibrated at. Without
#: ``_EP_SENSITIVITY`` the screen read misses under 3e-12 there and stayed silent on answers up
#: to 1.1e-11 off. Which offsets those are moves with the last bits of the angle: on the machine
#: measured, -5.999e-6 was 3.3e-12 (10 um) and 1.3e-12 (3 um) off and silent.
THICK = [(10000., -5.999e-6), (3000., -5.999e-6), (10000., 6e-6)]
THICK_WAVELENGTH = 616.5157327418266


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('thickness, dth', THICK)
def test_nearly_coincident_pairs_in_a_thick_layer_are_exact_or_loud(monkeypatch, thickness, dth,
                                                                     algo):
    """Within 1e-12 of the 1D rebuild, or loud, where a thick layer and small roots turn a small
    miss into a large error. Catches the threshold left at its calibrated value whatever the
    layer's propagation sensitivity, on a machine whose arithmetic leaves one of these points
    off; ``test_the_crowded_pair_screen_reads_the_backward_error`` pins the scaling itself."""
    monkeypatch.setattr(_conical_cutoff, 'PERIOD_Y', THICK_WAVELENGTH / (2 * 0.025))
    scene = dict(SCENES['PA616'], thickness=[thickness], wavelength=THICK_WAVELENGTH,
                 theta=float(np.arcsin(0.025)) * (1 + dth))
    ref = _vector(_solver(scene).solve(auto_detune=False))
    got, warned = _solve_2d(scene, algo, False)
    err = np.max(np.abs(got[2:] - ref[2:]))
    loud = _legacy_wavebit_ExceptionalPointWarning in warned
    assert loud or err < 1e-12, f'{thickness:g} nm {algo} dth {dth:g}: {err:.1e} off, silently'


def test_smm_screens_with_the_thickness_etm_reads(monkeypatch):
    """SMM takes no labels from a layer's thickness, so it hands the screen of nearly coincident
    pairs the optical thickness ETM's labels read: each layer's run, here two identical 150 nm
    slices read as 300 nm. Read where the screen receives it, at a near-coincident offset where
    both slices hold two rebuilt pairs. Catches SMM screening with no thickness (it then keeps
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
