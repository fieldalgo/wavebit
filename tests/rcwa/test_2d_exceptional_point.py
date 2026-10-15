"""A 2D layer at an exceptional point of its operator solves exactly, and nothing else moves.

**The point.** The 2D path diagonalises the product ``Omega2 = Omega_L Omega_R`` of each layer.
Where two of its eigenvalues meet in a 2x2 Jordan block (an exceptional point, EP) no
eigenvector basis exists: ``eig`` hands back two columns that are each accurate to the unit
round-off ``u`` but meet at a sine ``s ~ 1e-8 .. 1e-6``, the mode matrix's condition number is
``1/s`` and the interface inverses amplify rounding by about ``1/s**2``. Two kinds were measured:

* a *patterned* layer at a conical x-cutoff (``docs/theory/rcwa/rayleigh-points.md`` section 7): a y-invariant
  raster at the 1D path's exceptional points, one EP per order row (``wA`` does not depend on
  ``ky``), at ``q = |ky_v|``. Before: amplitudes 4.2e-8 .. 5.4e-3 off (ETM) and 8.2e-7 .. 3.6e-2
  (SMM) against the landed 1D rebuild, ``R + T - 1`` to -2.1e-2, gradients 2e-8 .. 6e-2, with
  no warning;
* a *genuinely 2D* lossless layer, where a pair of real ``q**2`` turns complex (codimension one in
  the wavelength: five to eight per 6% of wavelength on the cut pillars below). Before:
  4e-11 .. 1.7e-6 (ETM) and 2.5e-9 .. 6.4e-6 (SMM). The first-order formulation of the same
  layer holds the same EP once in each direction; its eigenvector split was 1.0e-5 (ETM) and
  0.32 (SMM, ``R + T - 1 = +0.3``) off at fy 0.95, silently.

:func:`~wavebit.rcwa._engine.solve_2d._rebuild_defective_pairs` now rebuilds every such pair as an
orthonormal Schur pair with an upper-triangular ``q``, gated by
:func:`~wavebit.rcwa._engine.solve_2d._defective_pairs`; the first-order split's
:func:`~wavebit.rcwa._engine.solve_2d_first_order._regular_pair` does the same per direction.
What this file pins:

1. the y-invariant EP scenes at ``rel`` 0, +-1e-15, +-1e-12, +-1e-9 against the 1D rebuild;
2. a genuinely 2D EP against a Cauchy-circle oracle that never runs the rebuild, on both
   formulations;
3. the rebuilt layer's two first-order relations and its conditioning (y-invariant and genuinely
   2D, both formulations), and the coupled exponential against a closed form;
4. the subspace step on a synthetic operator with two interacting Jordan pairs;
5. derivatives (``jax.grad``, ``jvp``, torch reverse and forward mode) against the 1D rebuild's,
   and at a genuinely 2D EP against Cauchy oracles that run no autodiff, eager and traced;
6. values and derivatives under every abstract trace, and the overflow refusal;
7. fields and field gradients, eager and under ``jax.jit``; at a genuinely 2D EP against Cauchy
   oracles;
8. inertness off the band, and a uniform cutoff left as it was.

Sizes stay at ``truncation_order=(5, 1)`` (``(5, 2)`` for the 2D EPs).
"""

from wavebit._common.backend import _enable_x64 as _legacy_wavebit_enable_x64

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa
import warnings

import numpy as np
import pytest

from wavebit.rcwa._engine import modal_gradient, solve_2d
from wavebit.rcwa._engine import solve_2d_first_order as fo
from tests.rcwa.test_conical_cutoff import PATTERNED, R1, _np, _numbers, _rel, _solver, _strict

_RELS = (0.0, 1e-15, -1e-15, 1e-12, -1e-12, 1e-9, -1e-9)


@pytest.fixture
def rebuilt(monkeypatch):
    """How many layers the pair rebuild took, eager or traced."""
    calls = []
    original = solve_2d._rebuild_defective_pairs

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(solve_2d, '_rebuild_defective_pairs', counted)
    return calls


def _row(result):
    n = _numbers(result, two_d=True)
    return n


def _amp(got, ref):
    return max(np.max(np.abs(got['r'] - ref['r'])), np.max(np.abs(got['t'] - ref['t'])))


# --------------------------------------------------- 1. y-invariant rasters at the 1D EPs

@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('name', list(PATTERNED))
def test_a_patterned_ep_in_2d_is_the_1d_rebuild(name, algo, rebuilt):
    """The raster's ``v = 0`` row is the landed 1D rebuild's answer, silently, R + T = 1.

    Measured <= 4.8e-14 in amplitudes (MU614; the others <= 1.4e-14) at every offset, ``R + T - 1``
    <= 1.2e-14. Catches: the gate off (1e-8 .. 4e-2), the Newton step dropped (the eig span's
    ``u/s``: 8e-10), the coupling ``t`` dropped, ``V_b`` without ``- V_a t``, and only one pair
    per layer rebuilt (the v = +-1 rows' EPs left: 2e-7).
    """
    scene = PATTERNED[name]
    for rel in _RELS:
        wavelength = scene['wavelength'] * (1 + rel)
        ref = _numbers(_solver(scene, wavelength=wavelength).solve(auto_detune=False))
        del rebuilt[:]
        result = _strict(lambda: _solver(scene, algo=algo, two_d=True, wavelength=wavelength)
                         .solve(auto_detune=False))
        assert rebuilt, f'{name} rel {rel:g}: the gate did not engage'
        got = _row(result)
        energy = float(np.sum(np.real(_np(result.R))) + np.sum(np.real(_np(result.T))) - 1)
        assert _amp(got, ref) < 1e-12, f'{name} {algo} rel {rel:g}: {_amp(got, ref):.1e}'
        assert abs(energy) < 1e-12, f'{name} {algo} rel {rel:g}: R + T - 1 = {energy:.1e}'


# ------------------------------------------------------------ 2. a genuinely 2D EP

def _cut_pillar(fy, ny=20):
    st = np.ones((1, ny, 20))
    st[0, :int(round(fy * ny)), 6:14] = 12.25
    return st


#: Lossless EPs of the PA616 pillar cut to a fraction of the y period, found by minimising the
#: gate's own sine over the wavelength (the pair's ``q**2`` turns from real to complex there).
_GENUINE = {'fy 0.5, 630.56 nm': (0.5, 630.558820467248), 'fy 0.95, 600.01 nm': (0.95, 600.010223815108)}


def _genuine_solver(fy, wavelength, theta=0.3, algo='ETM', backend='numpy', first_order=False,
                    **over):
    """The cut pillar. ``first_order`` takes the first-order formulation, whose operator is built
    from the same convolution matrices (measured: ``M``'s blocks carry ``epx``, ``epy`` and
    ``Kx epz_i Kx`` to 4e-16, and its eigenvalues square to ``-q**2`` of the product's ``q`` to
    2e-14 relative to ``max(1, |q**2|)``, 1e-2 off the point), so the same wavelengths are its EPs
    -- one Jordan pair in each direction."""
    s = _legacy_wavebit_rcwa(**{**dict(
        backend=backend, wavelength=wavelength, period=[1000., 700.], truncation_order=[5, 2],
        structure=_cut_pillar(fy), thickness=[300.], top=1.0, bot=2.1025, theta=theta,
        phi=np.pi / 2, psi=np.pi / 2, layer_algorithm=algo), **over})
    if first_order:
        s.formulation = 'first_order'
    return s


def _genuine(fy, wavelength, theta, algo, first_order=False, **over):
    s = _genuine_solver(fy, wavelength, theta, algo, first_order=first_order, **over)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = s.solve(auto_detune=False)
    return np.concatenate([_np(r.r_xyz).ravel(), _np(r.t_xyz).ravel()])


def _close_the_gates(m):
    """Both rebuilds off: the product construction and the first-order eigenvector split."""
    m.setattr(solve_2d, '_EP_SINE', 0.0)
    m.setattr(fo, '_PARALLEL_SIN', 0.0)


def _circle(f, z0, radius, K, derivative=False):
    """Cauchy: the mean of ``f`` over ``z0 + radius exp(2 pi i k / K)`` is ``f(z0)``, and with the
    weight ``1 / (radius exp(2 pi i k / K))`` it is ``f'(z0)``, for ``f`` analytic on the disc: the
    K-point trapezoidal rule on the circle, geometrically convergent in K (L. N. Trefethen and
    J. A. C. Weideman, SIAM Rev. 56, 385, 2014, Sec. 12, eq. (12.3) and Theorem 12.1; numerical
    differentiation on a circle goes back to J. N. Lyness and C. B. Moler, SIAM J. Numer. Anal. 4,
    202, 1967)."""
    unit = np.exp(2j * np.pi * np.arange(K) / K)
    return np.mean([f(z0 + radius * u) / (radius * u if derivative else 1) for u in unit], axis=0)


@pytest.mark.parametrize('name', list(_GENUINE))
def test_a_genuinely_2d_ep_matches_a_cauchy_oracle(name, monkeypatch):
    """At a real-to-complex transition of a lossless layer: the circle oracle, to 1e-12.

    The amplitudes are analytic in ``theta``, so their value is the mean over
    ``theta* + r exp(2 pi i k / 16)``, ``r = 1e-4``; on the circle the pair is off its EP and the
    product construction (gate off) is accurate: the oracle reproduced direct solves off the EP to
    2e-14 and itself across radii and algorithms to 7e-14. Measured: the rebuild <= 1.1e-14 from
    it; the product construction 6.4e-6 (fy 0.5, SMM) and 2.0e-6 (fy 0.95).
    """
    fy, wavelength = _GENUINE[name]
    monkeypatch.setattr(solve_2d, '_EP_SINE', 0.0)
    oracle = np.mean([_genuine(fy, wavelength, 0.3 + 1e-4 * np.exp(2j * np.pi * k / 16), 'ETM')
                      for k in range(16)], axis=0)
    product = max(np.max(np.abs(_genuine(fy, wavelength, 0.3, a) - oracle)) for a in ('ETM', 'SMM'))
    monkeypatch.undo()
    for algo in ('ETM', 'SMM'):
        err = np.max(np.abs(_genuine(fy, wavelength, 0.3, algo) - oracle))
        assert err < 1e-12, f'{name} {algo}: {err:.1e} from the oracle'
    assert product > 1e-8, f'{name}: the product construction is {product:.1e} -- not an EP'


@pytest.mark.parametrize('name', list(_GENUINE))
def test_a_genuinely_2d_ep_on_the_first_order_path_matches_the_cauchy_oracle(name, monkeypatch):
    """The first-order path at the same EPs, at ``rel`` 0, 1e-12 and -1e-9: within 1e-12 of the
    product path's circle oracle -- another formulation of the same discretized layer -- with
    ``R + T = 1``, a pair rebuilt in each direction, and no warning.

    Measured (aadbc351): <= 3.9e-14, and a first-order circle oracle agreed with the product one
    to 5.4e-15 here (1.1e-13 over all six measured EPs). With the split's gate closed, fy 0.95: 1.0e-5 (ETM) and 0.32 (SMM, ``R + T - 1 =
    +0.3``, silent); fy 0.5: 6.3e-11 (ETM), SMM at its floor (1.1e-14).
    """
    fy, wl0 = _GENUINE[name]
    pairs = []
    original = fo._regular_pair

    def counted(*args, **kwargs):
        pairs.append(1)
        return original(*args, **kwargs)
    for rel in (0.0, 1e-12, -1e-9):
        wavelength = wl0 * (1 + rel)
        with monkeypatch.context() as m:
            _close_the_gates(m)
            oracle = _circle(lambda th: _genuine(fy, wavelength, th, 'ETM'), 0.3, 1e-4, 16)
        for algo in ('ETM', 'SMM'):
            with monkeypatch.context() as m:
                m.setattr(fo, '_regular_pair', counted)
                del pairs[:]
                r = _strict(lambda: _genuine_solver(fy, wavelength, algo=algo, first_order=True)
                            .solve(auto_detune=False))
            err = np.max(np.abs(np.concatenate([_np(r.r_xyz).ravel(), _np(r.t_xyz).ravel()]) - oracle))
            energy = float(np.sum(np.real(_np(r.R))) + np.sum(np.real(_np(r.T))) - 1)
            assert len(pairs) >= 2, f'{name} {algo} rel {rel:g}: {len(pairs)} pairs rebuilt'
            assert err < 1e-12, f'{name} {algo} rel {rel:g}: {err:.1e} from the oracle'
            assert abs(energy) < 1e-12, f'{name} {algo} rel {rel:g}: R + T - 1 = {energy:.1e}'


# ------------------------------------------------- 3. the rebuilt layer's two relations

def _omega_factors(Kx, Ky, epx, epy, epzi, epxy, epyx):
    Kx, Ky = _np(Kx), _np(Ky)
    kx, ky = np.diag(Kx), np.diag(Ky)
    epx, epy, epzi, epxy, epyx = (_np(a) for a in (epx, epy, epzi, epxy, epyx))
    I = np.eye(len(kx))
    B = (kx[:, None] * epzi) * kx[None, :] - I
    D = (ky[:, None] * epzi) * ky[None, :] - I
    Cc = (kx[:, None] * epzi) * ky[None, :]
    E = (ky[:, None] * epzi) * kx[None, :]
    OL = np.block([[-Cc, B], [-D, E]])
    OR = np.block([[-Kx @ Ky + epyx, Kx @ Kx - epy], [epx - Ky @ Ky, Ky @ Kx - epxy]])
    return OL, OR


@pytest.mark.parametrize('name', ['PA616', 'PD574'])
def test_the_rebuilt_layer_satisfies_both_first_order_relations(name, monkeypatch):
    """``Omega_R W = V q`` and ``Omega_L V = W q`` to 1e-13 relative, and ``W`` well conditioned.

    Measured 1e-16 .. 6e-16 and ``cond(W)`` 30 .. 80, where the product construction's ``W``
    reached 1e7 .. 1.6e8. Catches the subspace step's ``a`` rows without their pair's ``M_01``
    only through test 4; here, ``t`` and ``V_b`` (an O(1) residual).
    """
    seen = []
    original = solve_2d.layer_modes_2d

    def spy(xp, Kx, Ky, I, epx, epy, epzi, epxy, epyx, C, **kw):
        out = original(xp, Kx, Ky, I, epx, epy, epzi, epxy, epyx, C, **kw)
        seen.append((_omega_factors(Kx, Ky, epx, epy, epzi, epxy, epyx), out))
        return out
    monkeypatch.setattr(solve_2d, 'layer_modes_2d', spy)
    scene = PATTERNED[name]
    _strict(lambda: _solver(scene, two_d=True).solve(auto_detune=False))
    (OL, OR), (W, V, q, coupled) = seen[0]
    W, V, q = _np(W), _np(V), _np(q)
    assert coupled and q.ndim == 2 and np.count_nonzero(q - np.diag(np.diag(q))) == 3
    r1 = np.linalg.norm(OR @ W - V @ q) / (np.linalg.norm(OR) * np.linalg.norm(W))
    r2 = np.linalg.norm(OL @ V - W @ q) / (np.linalg.norm(OL) * np.linalg.norm(V))
    assert r1 < 1e-13 and r2 < 1e-13, (r1, r2)
    assert np.linalg.cond(W) < 1e4, np.linalg.cond(W)


def _couplings(q):
    rows, cols = np.nonzero(q - np.diag(np.diag(q)))
    return list(zip(rows.tolist(), cols.tolist()))


def _coupled_entry(q, a, b, scale):
    """``exp(scale q)[a, b]`` for one coupling in closed form, ``t s e^{s (alpha + beta)/2}
    sinh(z) / z``, ``z = s (alpha - beta) / 2`` -- no divided difference to share with the code
    (the 2 x 2 exponential in sinch form with its lower-left entry zero: A. H. Al-Mohy and
    N. J. Higham, SIAM J. Matrix Anal. Appl. 31, 970, 2009, eq. (2.2))."""
    alpha, beta, t = q[a, a], q[b, b], q[a, b]
    z = scale * (alpha - beta) / 2
    return t * scale * np.exp(scale * (alpha + beta) / 2) * np.sinc(1j * z / np.pi)


@pytest.mark.parametrize('first_order', [False, True], ids=['second_order', 'first_order'])
def test_a_genuinely_2d_rebuilt_layer(first_order, monkeypatch):
    """At fy 0.95, 600.01 nm, read off the solve's own call: one coupling (one per direction on
    the first-order path), the modal relations to 1e-14 relative (``Omega_R W = V q`` and
    ``Omega_L V = W q``; ``M psi = psi lam``), an orthonormal pair, a basis conditioned below 1e3,
    and the coupled exponential the propagation takes equal to the closed form of its 2 x 2 block.

    Measured (aadbc351): residuals <= 3.2e-16, condition 46 (2D) and 65 (``[psi_f, psi_b]``) where
    ``eig``'s basis had 1.3e7 and 4.0e7, the exponential's coupling within 3.7e-18 of a 50-digit
    one (SciPy's ``expm`` of the same generator: 1.7e-10 off).
    """
    fy, wavelength = _GENUINE['fy 0.95, 600.01 nm']
    k0d = 2 * np.pi / wavelength * 300.
    seen = []
    if first_order:
        original = fo._layer_split

        def spy(xp, M, n, k0_d=None, **kw):
            out = original(xp, M, n, k0_d, **kw)
            seen.append((_np(M), out))
            return out
        monkeypatch.setattr(fo, '_layer_split', spy)
    else:
        original = solve_2d.layer_modes_2d

        def spy(xp, Kx, Ky, I, epx, epy, epzi, epxy, epyx, C, **kw):
            out = original(xp, Kx, Ky, I, epx, epy, epzi, epxy, epyx, C, **kw)
            seen.append((_omega_factors(Kx, Ky, epx, epy, epzi, epxy, epyx), out))
            return out
        monkeypatch.setattr(solve_2d, 'layer_modes_2d', spy)
    _strict(lambda: _genuine_solver(fy, wavelength, first_order=first_order).solve(auto_detune=False))
    operator, out = seen[0]
    assert out[-1], 'the layer was not rebuilt'
    if first_order:
        lam_f, psi_f, lam_b, psi_b = (_np(x) for x in out[:4])
        blocks = [(lam_f, psi_f, 1j * k0d), (lam_b, psi_b, -1j * k0d)]
        basis = np.concatenate([psi_f, psi_b], axis=1)
        for lam, psi, _ in blocks:
            res = (np.linalg.norm(operator @ psi - psi @ lam)
                   / (np.linalg.norm(operator) * np.linalg.norm(psi)))
            assert res < 1e-14, f'M psi - psi lam: {res:.1e}'
    else:
        (OL, OR), (W, V, q) = operator, (_np(x) for x in out[:3])
        blocks = [(q, W, -k0d)]
        basis = W
        r1 = np.linalg.norm(OR @ W - V @ q) / (np.linalg.norm(OR) * np.linalg.norm(W))
        r2 = np.linalg.norm(OL @ V - W @ q) / (np.linalg.norm(OL) * np.linalg.norm(V))
        assert r1 < 1e-14 and r2 < 1e-14, (r1, r2)
    assert np.linalg.cond(basis) < 1e3, np.linalg.cond(basis)
    for generator, columns, scale in blocks:
        (a, b), = _couplings(generator)
        pair = columns[:, [a, b]]
        if first_order:
            assert np.max(np.abs(pair.conj().T @ pair - np.eye(2))) < 1e-14
        E = _np(modal_gradient.exponential(np, generator, scale, coupled=True))
        exact = _coupled_entry(generator, a, b, scale)
        assert abs(E[a, b] - exact) < 1e-13 * abs(exact), (E[a, b], exact)


# ---------------------------------------------- 4. two interacting pairs, synthetically

def test_two_interacting_jordan_pairs_are_both_rebuilt_exactly():
    """A random operator with two 2x2 Jordan blocks in a non-orthogonal basis: both pairs' rebuilt
    subspaces are invariant to 1e-13 and ``q**2`` is the restricted operator.

    In this basis the two pairs couple in the Newton step's coefficient space, which the
    y-invariant scenes never do (their order rows are decoupled). Catches the ``a`` rows of
    another pair taken without that pair's ``M_01`` (a residual of about 1e-10) and the step
    dropped (the eig span's ``u/s``).
    """
    rng = np.random.default_rng(3)
    n2 = 24
    J = np.diag(rng.uniform(1.0, 6.0, n2) + 0j)
    J[0, 0] = J[1, 1] = 2.5
    J[0, 1] = 0.3
    J[2, 2] = J[3, 3] = 4.1
    J[2, 3] = -0.2
    S = rng.standard_normal((n2, n2)) + 1j * rng.standard_normal((n2, n2))
    A = S @ J @ np.linalg.inv(S)
    OR = rng.standard_normal((n2, n2)) + 1j * rng.standard_normal((n2, n2)) + 4 * np.eye(n2)
    w, W = np.linalg.eig(A)
    q = np.sqrt(w)
    V = (OR @ W) / q
    pairs = solve_2d._defective_pairs(np, w, W, q, 4)
    assert pairs is not None and len(pairs[0]) == 2
    W2, V2, Q, a, b, mask = solve_2d._rebuild_defective_pairs(np, A, OR, W, V, q, w, pairs)
    for p in range(2):
        cols = [int(a[p]), int(b[p])]
        X = W2[:, cols]
        T = (Q @ Q)[np.ix_(cols, cols)]
        res = np.linalg.norm(A @ X - X @ T) / (np.linalg.norm(A) * np.linalg.norm(X))
        assert res < 1e-13, f'pair {p}: invariant-subspace residual {res:.1e}'
    assert np.linalg.norm(OR @ W2 - V2 @ Q) / (np.linalg.norm(OR) * np.linalg.norm(W2)) < 1e-13


def test_split_labels_are_put_on_one_side():
    """A pair whose labels disagree (a complex pair on Maystre's cut, a tie the flux breaks) is
    rebuilt with both roots on the more decaying member's side: ``alpha + beta`` stays away from
    zero and the ``+`` set takes the whole block."""
    rng = np.random.default_rng(4)
    n2 = 12
    J = np.diag(rng.uniform(-6.0, -1.0, n2) + 0j)
    J[0, 0] = J[1, 1] = -2.0
    J[0, 1] = 0.4
    S = rng.standard_normal((n2, n2)) + 1j * rng.standard_normal((n2, n2))
    A = S @ J @ np.linalg.inv(S)
    OR = np.eye(n2) * 3 + rng.standard_normal((n2, n2))
    w, W = np.linalg.eig(A)
    q = np.sqrt(w)
    i, j = np.argsort(np.abs(w + 2.0))[:2]
    if abs(q[i] - q[j]) < abs(q[i] + q[j]):
        q[j] = -q[j]                             # opposite labels on the pair
    assert abs(q[i] - q[j]) > abs(q[i] + q[j])
    V = (OR @ W) / q
    pairs = solve_2d._defective_pairs(np, w, W, q, 4)
    assert pairs is not None
    _, _, Q, a, b, _ = solve_2d._rebuild_defective_pairs(np, A, OR, W, V, q, w, pairs)
    alpha, beta = Q[a[0], a[0]], Q[b[0], b[0]]
    assert abs(alpha - beta) < abs(alpha + beta)


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_split_labels_at_an_imaginary_angle(algo, rebuilt, monkeypatch):
    """PA616's raster at ``theta = 0.8j``: every root is a tie (``Re(q) = 0``) that ``Im(q) <= 0``
    breaks, and SMM's labels split the pair at every offset. Against the landed 1D path (which
    takes the scene without a split): measured <= 5.6e-13; develop 2.9e-3 (ETM) and 1.7e-8 (SMM).
    Catches the pair kept on its own labels (SMM)."""
    scene = dict(PATTERNED['PA616'], theta=0.8j)
    split = []
    original = solve_2d._rebuild_defective_pairs

    def spy(xp, A, OR, W, V, q, w, pairs):
        qa, qb = np.asarray(q)[np.asarray(pairs[0])], np.asarray(q)[np.asarray(pairs[1])]
        split.append(bool(np.any(np.abs(qa - qb) >= np.abs(qa + qb))))
        return original(xp, A, OR, W, V, q, w, pairs)
    monkeypatch.setattr(solve_2d, '_rebuild_defective_pairs', spy)
    for rel in (0.0, 1e-12, -1e-9):
        wavelength = scene['wavelength'] * (1 + rel)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ref = _numbers(_solver(scene, wavelength=wavelength).solve(auto_detune=False))
            got = _row(_solver(scene, algo=algo, two_d=True, wavelength=wavelength).solve(auto_detune=False))
        assert _amp(got, ref) < 1e-11, f'{algo} rel {rel:g}: {_amp(got, ref):.1e}'
    assert split and (algo == 'ETM' or all(split))


# --------------------------------------------------------------------- 5. derivatives

@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('name', ['PA616', 'MU614'])
def test_derivatives_at_a_2d_ep_are_the_1d_rebuild_s(name, algo):
    """``d(sum R)`` in the raster, the wavelength, the thickness and ``theta``: ``jax.grad``,
    ``jax.jvp``, torch reverse and ``forward_ad`` against ``jax.grad`` of the 1D rebuild.

    Measured <= 1.1e-13 (PA616) and <= 6.0e-12 (MU614, the 1D rebuild's own consistency there);
    before, 2e-7 .. 6e-2. Catches the coupled exponential taking the formula's tangent, the exact
    inverse's coupling terms dropped, and the pair Sylvester's coupling passes dropped.
    """
    jax = pytest.importorskip('jax')
    _legacy_wavebit_enable_x64()
    torch = pytest.importorskip('torch')
    from torch.autograd import forward_ad
    jnp = jax.numpy
    scene = PATTERNED[name]
    u0, t0, th0, wl0 = (np.asarray(scene['structure'], float), float(scene['thickness'][0]),
                        float(scene['theta']), float(scene['wavelength']))

    def make(backend, two_d):
        s = _solver(scene, algo=algo if two_d else 'ETM', backend=backend, two_d=two_d)
        if backend == 'jax':
            def f(u, w, t, th):
                st = jnp.repeat(u, 2, axis=1) if two_d else u
                return jnp.sum(jnp.real(s.solve(structure=st, wavelength=w, thickness=jnp.stack([t]),
                                                theta=th, auto_detune=False).R))
        else:
            def f(u, w, t, th):
                st = u.repeat_interleave(2, dim=1) if two_d else u
                return s.solve(structure=st, wavelength=w, thickness=torch.stack([t]), theta=th,
                               auto_detune=False).R.real.sum()
        return f
    args = tuple(jnp.asarray(x) for x in (u0, wl0, t0, th0))
    ref = jax.grad(make('jax', False), argnums=(0, 1, 2, 3))(*args)
    f = make('jax', True)
    got = jax.grad(f, argnums=(0, 1, 2, 3))(*args)
    for label, g, r in zip(('raster', 'wavelength', 'thickness', 'theta'), got, ref):
        assert _rel(g, r) < 1e-10, f'jax.grad {label}: {_rel(g, r):.1e}'
    jvp = jax.jvp(lambda w: f(args[0], w, args[2], args[3]), (args[1],), (jnp.asarray(1.0),))[1]
    assert _rel(jvp, ref[1]) < 1e-10
    ft = make('torch', True)
    leaves = [torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in (u0, wl0, t0, th0)]
    ft(*leaves).backward()
    for label, leaf, r in zip(('raster', 'wavelength', 'thickness', 'theta'), leaves, ref):
        assert _rel(leaf.grad.numpy(), r) < 1e-10, f'torch {label}: {_rel(leaf.grad.numpy(), r):.1e}'
    with forward_ad.dual_level():
        w = forward_ad.make_dual(torch.tensor(wl0, dtype=torch.float64), torch.tensor(1., dtype=torch.float64))
        tangent = forward_ad.unpack_dual(ft(torch.tensor(u0), w, torch.tensor(t0, dtype=torch.float64),
                                            torch.tensor(th0, dtype=torch.float64))).tangent
    assert _rel(float(tangent), ref[1]) < 1e-10


#: The genuinely 2D EP the derivative and field oracles are built at; the circles below keep
#: clear of its nearest Rayleigh point (0.033 in theta, 10.9 nm) and of a singularity a few nm
#: away in the wavelength, eps_p and d that spoils larger circles (3 nm, 32 points: 1.9e-3).
_ORACLE_EP = _GENUINE['fy 0.95, 600.01 nm']
_ARGS = ('theta', 'wavelength', 'eps_p', 'thickness')


def _pillar_eps(fy, eps):
    return np.where(_cut_pillar(fy) > 1, eps, 1.0)


@pytest.fixture(scope='module')
def genuine_derivatives():
    """``a`` (every order's ``r_xyz``, ``t_xyz``) and ``da/d(theta, wavelength, eps_p, d)`` at
    ``_ORACLE_EP`` from NumPy values with both rebuilds closed: no autodiff, no rebuild.

    ``a`` is analytic in all four. Cauchy's derivative formula (:func:`_circle`) on circles of 32
    points, radius
    3e-3 in theta, 0.5 nm, and 0.02 in the pillar permittivity: each moves the EP, so the circle
    is off it. The EP does not move with ``d`` (and the solver refuses a complex thickness), so
    ``da/dd`` is the theta-circle mean (16 points) of the centre derivative of a 24-node Chebyshev
    interpolant on ``d +- 1.5`` nm. Measured spread over halved radii, doubled point counts and the
    first-order formulation: <= 1.6e-12 of each column's largest entry.
    """
    fy, wl = _ORACLE_EP

    def a(theta=0.3, wavelength=wl, eps=12.25, d=300.):
        return _genuine(fy, wavelength, theta, 'ETM', structure=_pillar_eps(fy, eps), thickness=[d])

    def d_thickness(theta):
        x = np.cos(np.pi * (np.arange(24) + 0.5) / 24)
        values = np.array([a(theta=theta, d=300. + 1.5 * xj) for xj in x])
        cheb = np.polynomial.chebyshev
        coef = np.linalg.solve(cheb.chebvander(x, 23), values)
        return cheb.chebval(0.0, cheb.chebder(coef)) / 1.5
    with pytest.MonkeyPatch.context() as m:
        _close_the_gates(m)
        value = _circle(lambda z: a(theta=z), 0.3, 1e-4, 16)
        da = np.stack([_circle(lambda z: a(theta=z), 0.3, 3e-3, 32, derivative=True),
                       _circle(lambda z: a(wavelength=z), wl, 0.5, 32, derivative=True),
                       _circle(lambda z: a(eps=z), 12.25, 0.02, 32, derivative=True),
                       _circle(d_thickness, 0.3, 3e-3, 16)], axis=1)
    return value, da


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('first_order', [False, True], ids=['second_order', 'first_order'])
def test_genuinely_2d_ep_derivatives_match_cauchy_oracles(first_order, algo, genuine_derivatives):
    """At a genuinely 2D EP, against :func:`genuine_derivatives`: ``jax.jacfwd`` of every amplitude
    in ``(theta, wavelength, eps_p, d)``, and the gradient of ``sum |a|**2`` (oracle
    ``2 Re(conj(a) . da)``, no circle over a power) by ``jax.grad``, ``jit(grad)``,
    ``jit(vmap(grad))`` and torch backward, to 1e-10; the vmap's other lanes (``rel`` 1e-12, 1e-3)
    to 1e-11 of eager ``jax.grad`` there.

    Measured (aadbc351): <= 7.3e-13 on all four paths; at fy 0.8 / 620.39 and fy 0.5 / 630.56,
    <= 3.2e-12. With the rebuilds closed, 1.4e-6 (first-order ETM, ``d``) .. 44 (theta).
    """
    jax = pytest.importorskip('jax')
    _legacy_wavebit_enable_x64()
    torch = pytest.importorskip('torch')
    jnp = jax.numpy
    fy, wl = _ORACLE_EP
    a0, da = genuine_derivatives
    g0 = 2 * np.real(np.conj(a0) @ da)
    s = _genuine_solver(fy, wl, algo=algo, backend='jax', first_order=first_order)
    mask = jnp.asarray(_cut_pillar(fy) > 1)

    def f(x):
        r = s.solve(theta=x[0], wavelength=x[1], structure=jnp.where(mask, x[2], 1.0),
                    thickness=jnp.stack([x[3]]), auto_detune=False)
        return jnp.concatenate([r.r_xyz.ravel(), r.t_xyz.ravel()])

    def J(x):
        return jnp.sum(jnp.abs(f(x)) ** 2)
    x0 = jnp.asarray([0.3, wl, 12.25, 300.])
    jac = np.asarray(jax.jacfwd(f)(x0))
    for k, label in enumerate(_ARGS):
        err = np.max(np.abs(jac[:, k] - da[:, k])) / np.max(np.abs(da[:, k]))
        assert err < 1e-10, f'jacfwd {label}: {err:.1e}'
    lanes = jnp.stack([x0.at[1].set(wl * (1 + rel)) for rel in (0.0, 1e-12, 1e-3)])
    swept = np.asarray(jax.jit(jax.vmap(jax.grad(J)))(lanes))
    grads = {'grad': jax.grad(J)(x0), 'jit(grad)': jax.jit(jax.grad(J))(x0), 'jit(vmap(grad))': swept[0]}
    st = _genuine_solver(fy, wl, algo=algo, backend='torch', first_order=first_order)
    tmask, one = torch.tensor(_cut_pillar(fy) > 1), torch.tensor(1.0, dtype=torch.float64)
    xt = torch.tensor(np.asarray(x0), requires_grad=True)
    rt = st.solve(theta=xt[0], wavelength=xt[1], structure=torch.where(tmask, xt[2], one),
                  thickness=torch.stack([xt[3]]), auto_detune=False)
    (torch.cat([rt.r_xyz.ravel(), rt.t_xyz.ravel()]).abs() ** 2).sum().backward()
    grads['torch'] = xt.grad.numpy()
    for label, g in grads.items():
        err = np.abs(np.asarray(g) - g0) / np.abs(g0)
        assert np.all(err < 1e-10), f'{label}: ' + ', '.join(f'{p} {e:.1e}' for p, e in zip(_ARGS, err))
    for lane in (1, 2):
        eager = np.asarray(jax.grad(J)(lanes[lane]))
        err = np.max(np.abs(swept[lane] - eager) / np.abs(eager))
        assert err < 1e-11, f'jit(vmap(grad)) lane {lane}: {err:.1e} from eager'


# ------------------------------------------------------------ 6. abstract traces

@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_a_2d_ep_under_an_abstract_trace(algo, rebuilt):
    """PA616's raster under ``jit``, ``jit(vmap)`` and ``torch.func.vmap`` (values), and
    ``jit(grad(sum vmap))``, ``jit(vmap(grad))``, torch backward through ``torch.func.vmap``
    (``dR/dwl`` per lane), lanes ``rel`` 0, +-1e-12 and 1e-2: the eager 1D rebuild's.

    Measured <= 1e-14 (values) and <= 8e-14 (derivatives); before 2e-8 .. 3e-3.
    """
    jax = pytest.importorskip('jax')
    _legacy_wavebit_enable_x64()
    torch = pytest.importorskip('torch')
    jnp = jax.numpy
    scene = PATTERNED['PA616']
    wl0 = scene['wavelength']
    lanes = [wl0, wl0 * (1 - 1e-12), wl0 * (1 + 1e-12), wl0 * (1 + 1e-2)]
    ref = np.stack([np.concatenate([n['r'], n['t']]) for n in (
        _numbers(_solver(scene, wavelength=w).solve(auto_detune=False)) for w in lanes)])
    s1 = _solver(scene, backend='jax')
    gref = np.array([float(jax.grad(lambda w: jnp.sum(jnp.real(
        s1.solve(wavelength=w, auto_detune=False).R)))(jnp.asarray(w))) for w in lanes])
    s = _solver(scene, algo=algo, backend='jax', two_d=True)
    row = 1                                      # orders_v = (-1, 0, 1)

    def one(w):
        r = s.solve(wavelength=w, auto_detune=False)
        return jnp.concatenate([r.r_xyz[row].ravel(), r.t_xyz[row].ravel()])
    del rebuilt[:]
    jf = jax.jit(one)
    values = {'jit': np.stack([np.asarray(jf(jnp.asarray(w))) for w in lanes]),
              'jit(vmap)': np.asarray(jax.jit(jax.vmap(one))(jnp.asarray(lanes)))}
    assert rebuilt
    st = _solver(scene, algo=algo, backend='torch', two_d=True)

    def tone(w):
        r = st.solve(wavelength=w, auto_detune=False)
        return torch.cat([r.r_xyz[row].ravel(), r.t_xyz[row].ravel()])
    values['torch.func.vmap'] = torch.func.vmap(tone)(torch.tensor(lanes, dtype=torch.float64)).numpy()
    for label, got in values.items():
        err = np.max(np.abs(got - ref))
        assert np.isfinite(err) and err < 1e-12, f'{label}: {err:.1e}'

    def F(w):
        return jnp.sum(jnp.real(s.solve(wavelength=w, auto_detune=False).R))
    grads = {'jit(grad(sum vmap))': jax.jit(jax.grad(lambda ws: jnp.sum(jax.vmap(F)(ws))))(jnp.asarray(lanes)),
             'jit(vmap(grad))': jax.jit(jax.vmap(jax.grad(F)))(jnp.asarray(lanes))}
    wt = torch.tensor(lanes, dtype=torch.float64, requires_grad=True)
    torch.func.vmap(lambda w: st.solve(wavelength=w, auto_detune=False).R.real.sum())(wt).sum().backward()
    grads['torch vmap + backward'] = wt.grad.numpy()
    for label, g in grads.items():
        err = np.max(np.abs(np.asarray(g) - gref) / np.abs(gref))
        assert np.isfinite(err) and err < 1e-11, f'{label}: {err:.1e}'


def test_more_pairs_than_the_trace_holds_is_loud(monkeypatch):
    """Under ``jax.jit`` the pairs are rebuilt up to a static count; the y-invariant raster has
    three at once, so a count of one must give NaN, never the two left silently (2e-7)."""
    jax = pytest.importorskip('jax')
    _legacy_wavebit_enable_x64()
    jnp = jax.numpy
    original = solve_2d.layer_modes_2d
    monkeypatch.setattr(solve_2d, 'layer_modes_2d',
                        lambda *a, **k: original(*a, **{**k, 'pair_cap': 1}))
    scene = PATTERNED['PA616']
    s = _solver(scene, backend='jax', two_d=True)
    R = np.asarray(jax.jit(lambda w: s.solve(wavelength=w, auto_detune=False).R)(
        jnp.asarray(scene['wavelength'])))
    assert np.all(np.isnan(R))


# ------------------------------------------------------------------------ 7. fields

@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('name', ['PA616', 'MU614'])
def test_a_2d_ep_field_and_its_gradient_are_the_1d_rebuild_s(name, algo):
    """Six components on a 12 x 1 x 8 grid against the 1D field (its global phase removed), and
    ``jax.grad`` of ``sum |E|**2`` at nine points in the raster and the wavelength, eagerly and
    under ``jax.jit``.

    Measured: fields <= 1.2e-13 of ``max|F|``, gradients <= 2.6e-13; before 2.5e-7 .. 5.1e-2 and
    2.3e-6 .. 7.3e-3. Catches the field state reading a pair with ``coupled=True`` (only its
    ``a`` row off the bottom face) and field reconstruction ignoring the record's tenth slot.
    """
    jax = pytest.importorskip('jax')
    _legacy_wavebit_enable_x64()
    jnp = jax.numpy
    scene = PATTERNED[name]
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


def test_a_genuinely_2d_ep_field_gradient_agrees_across_algorithms():
    """At a genuinely 2D EP (fy 0.8, 620.39 nm, ``|k0 d q| = 0.91`` for the pair, so SMM's field
    state would keep it on the top face): SMM's field gradient is the ETM's (measured 2.9e-14 in the
    raster, 1.4e-12 in the wavelength; read with only its ``a`` row off the bottom face, 1.6e-10 and
    1.1e-9: f2_genuine_fieldgrad). The pair's
    tangent ``d_ba`` is generic here -- not tied to its gap, as on a y-invariant raster -- so this
    is the scene that catches a pair read with only its ``a`` row off the bottom face."""
    jax = pytest.importorskip('jax')
    _legacy_wavebit_enable_x64()
    jnp = jax.numpy
    st0 = _cut_pillar(0.8)
    xs, zs = jnp.asarray([-200., 0., 130.]), jnp.asarray([40., 150., 260.])
    grads = {}
    for algo in ('ETM', 'SMM'):
        s = _legacy_wavebit_rcwa(backend='jax', wavelength=620.391376855451, period=[1000., 700.],
                         truncation_order=[5, 2], structure=st0, thickness=[300.], top=1.0,
                         bot=2.1025, theta=0.3, phi=np.pi / 2, psi=np.pi / 2, layer_algorithm=algo)

        def F(u, w):
            f = s.solve(structure=u, wavelength=w, auto_detune=False, answers='field').reconstruct_field(
                sample_x=xs, sample_y=jnp.asarray([0., 200.]), sample_z=zs)
            return jnp.sum(jnp.abs(f[..., :3]) ** 2)
        gu, gw = jax.grad(F, argnums=(0, 1))(jnp.asarray(st0), jnp.asarray(620.391376855451))
        grads[algo] = (np.asarray(gu), float(gw))
    (gu_e, gw_e), (gu_s, gw_s) = grads['ETM'], grads['SMM']
    raster = np.max(np.abs(gu_s - gu_e)) / np.max(np.abs(gu_e))
    wavelength = abs(gw_s - gw_e) / abs(gw_e)
    assert raster < 2e-11 and wavelength < 2e-11, (raster, wavelength)


_FIELD_AT = ([-200., 0., 130.], [0., 200.], [40., 150., 260.])


@pytest.fixture(scope='module')
def genuine_fields():
    """The six components at ``_FIELD_AT`` (x, y, z in nm; z inside the layer) at ``_ORACLE_EP``,
    and their theta and wavelength derivatives, from circles with both rebuilds closed: the field
    is analytic in both (no conjugate is taken). Radii and point counts as in
    :func:`genuine_derivatives`; measured spread over doubled counts, a doubled value radius and
    the first-order formulation: <= 9.5e-13 of the largest entry (values: 1.9e-14)."""
    fy, wl = _ORACLE_EP

    def field(theta, wavelength):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = _genuine_solver(fy, wavelength, theta).solve(auto_detune=False, answers='field')
            return _np(r.reconstruct_field(*(np.asarray(v) for v in _FIELD_AT)))
    with pytest.MonkeyPatch.context() as m:
        _close_the_gates(m)
        return (_circle(lambda z: field(z, wl), 0.3, 1e-4, 16),
                _circle(lambda z: field(z, wl), 0.3, 3e-3, 32, derivative=True),
                _circle(lambda z: field(0.3, z), wl, 0.5, 32, derivative=True))


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('first_order', [False, True], ids=['second_order', 'first_order'])
def test_a_genuinely_2d_ep_field_matches_cauchy_oracles(first_order, algo, genuine_fields):
    """The field at a genuinely 2D EP against :func:`genuine_fields`: NumPy (silent, the pair
    rebuilt) and ``jax.jit`` values to 1e-12 of ``max|F|``; ``jax.jacfwd`` in theta and the
    wavelength, eagerly and under ``jax.jit``, to 1e-10. The independent reference the
    cross-algorithm field-gradient test above does not have.

    Measured (aadbc351): values <= 4.5e-14, derivatives <= 4.7e-13 (fy 0.8 and fy 0.5:
    <= 8.1e-14 and <= 1.8e-12, within their oracles' spread of 2.4e-12). With the rebuilds closed the values were 3.0e-6 (2D SMM) ..
    0.88 (first-order SMM) off.
    """
    jax = pytest.importorskip('jax')
    _legacy_wavebit_enable_x64()
    jnp = jax.numpy
    fy, wl = _ORACLE_EP
    value, dtheta, dwl = genuine_fields
    scale = np.max(np.abs(value))
    rebuilt = []
    module, name = (fo, '_regular_pair') if first_order else (solve_2d, '_rebuild_defective_pairs')
    original = getattr(module, name)

    def counted(*args, **kwargs):
        rebuilt.append(1)
        return original(*args, **kwargs)
    with pytest.MonkeyPatch.context() as m:
        m.setattr(module, name, counted)
        r = _strict(lambda: _genuine_solver(fy, wl, algo=algo, first_order=first_order)
                    .solve(auto_detune=False, answers='field'))
    assert rebuilt
    eager = _np(r.reconstruct_field(*(np.asarray(v) for v in _FIELD_AT)))
    s = _genuine_solver(fy, wl, algo=algo, backend='jax', first_order=first_order)

    def F(x):
        return s.solve(theta=x[0], wavelength=x[1], auto_detune=False, answers='field').reconstruct_field(
            *(jnp.asarray(v) for v in _FIELD_AT))
    x0 = jnp.asarray([0.3, wl])
    for label, got in (('numpy', eager), ('jit', np.asarray(jax.jit(F)(x0)))):
        err = np.max(np.abs(got - value)) / scale
        assert err < 1e-12, f'{label} field: {err:.1e}'
    for label, jac in (('jacfwd', jax.jacfwd(F)(x0)), ('jit(jacfwd)', jax.jit(jax.jacfwd(F))(x0))):
        jac = np.asarray(jac)
        for k, ref in ((0, dtheta), (1, dwl)):
            err = np.max(np.abs(jac[..., k] - ref)) / np.max(np.abs(ref))
            assert err < 1e-10, f'{label} d/d{_ARGS[k]}: {err:.1e}'


# ----------------------------------------------------------- 8. off the band, unchanged

def _random_2d(rng):
    nx, ny = int(rng.choice([8, 10])), int(rng.choice([8, 10]))
    st = np.full((1, ny, nx), rng.uniform(1.0, 2.5))
    x0, x1 = sorted(rng.integers(0, nx, 2))
    y0, y1 = sorted(rng.integers(0, ny, 2))
    st[0, y0:y1 + 1, x0:x1 + 1] = rng.uniform(2.0, 12.0)
    return dict(structure=st, thickness=[float(rng.uniform(50, 400))],
                period=[float(rng.uniform(300, 900)), float(rng.uniform(300, 900))],
                truncation_order=[3, 2], wavelength=float(rng.uniform(400, 1500)), top=1.0,
                bot=2.25, theta=float(rng.uniform(0.05, 1.2)), phi=float(rng.uniform(0, np.pi)),
                psi=float(rng.uniform(0, np.pi)))


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_off_the_band_nothing_moves(algo, rebuilt, monkeypatch):
    """Twelve random 2D layers: the gate never engages and every bit equals the gate switched off
    (the product construction, develop's arithmetic). Over 400 random layers the gate engaged on
    one (its smallest sine 6.6e-3)."""
    rng = np.random.default_rng(12)
    for _ in range(12):
        kw = _random_2d(rng)
        on = _legacy_wavebit_rcwa(backend='numpy', layer_algorithm=algo, **kw).solve(auto_detune=False)
        with monkeypatch.context() as m:
            m.setattr(solve_2d, '_EP_SINE', 0.0)
            off = _legacy_wavebit_rcwa(backend='numpy', layer_algorithm=algo, **kw).solve(auto_detune=False)
        for key in ('R', 'T', 'r_xyz', 't_xyz'):
            assert np.array_equal(_np(getattr(on, key)), _np(getattr(off, key)))
    assert not rebuilt


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_a_uniform_cutoff_in_2d_is_unchanged(algo, monkeypatch):
    """R1's raster at its exact point -- a uniform layer's double root, diagonalizable, where the
    product construction was already exact: the rebuild, engaged or not, is that answer."""
    on = _numbers(_solver(R1, algo=algo, two_d=True).solve(auto_detune=False), two_d=True)
    monkeypatch.setattr(solve_2d, '_EP_SINE', 0.0)
    off = _numbers(_solver(R1, algo=algo, two_d=True).solve(auto_detune=False), two_d=True)
    assert _amp(on, off) < 1e-13
