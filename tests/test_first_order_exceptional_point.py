"""First-order layers at a patterned exceptional point: a defective pair in a regular basis.

A patterned conical x-cutoff is a 2x2 Jordan block of the first-order operator
(``docs/rayleigh-points.md`` section 7). ``eig`` returns its two eigenvectors at an angle of about
``sqrt(u)``; through that basis ETM rounds like ``C u / s**2`` and SMM like ``C u / s**3``. The
split rebuilds the pair in an orthonormal basis of its invariant subspace with a triangular
generator (``solve_2d_first_order._regular_pair``). The oracle is the landed 1D conical rebuild,
which the same structure must reproduce with ``formulation='first_order'``.
"""
import warnings

import numpy as np
import pytest

import wavebit
from wavebit._core import modal_gradient, solve_2d_first_order as fo
from test_conical_cutoff import (PATTERNED as _LANDED, UNIFORM, _np, _pillar, _solver,
                                 _strict, _vector)

#: The landed scenes plus two more exceptional points of the same family (root-found the same way).
PATTERNED = dict(_LANDED,
                 PB687=dict(structure=_pillar(), wavelength=687.239832514180, period=1000.,
                            thickness=[300.], top=1.0, bot=2.1025, theta=0.4, phi=1.0),
                 PL952=dict(structure=_pillar(), wavelength=952.0673283430336, period=1000.,
                            thickness=[200.], top=1.0, bot=2.25, theta=0.3))

_RELS = (0.0, 1e-15, -1e-15, 1e-12, -1e-12, 1e-9, -1e-9)


def _first_order(scene, algo='ETM', backend='numpy', **over):
    s = _solver(scene, algo=algo, backend=backend, **over)
    s.formulation = 'first_order'
    return s


@pytest.fixture
def rebuilt(monkeypatch):
    """Every pair the split rebuilt."""
    calls = []
    original = fo._regular_pair

    def counted(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)
    monkeypatch.setattr(fo, '_regular_pair', counted)
    return calls


def _rel(a, b):
    a, b = np.ravel(np.asarray(a)), np.ravel(np.asarray(b))
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# ------------------------------------------------------------------------ 1. values

@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('name', list(PATTERNED))
def test_a_patterned_exceptional_point_is_the_1d_rebuild(name, algo, rebuilt):
    """At ``rel`` 0, +-1e-15, +-1e-12, +-1e-9: the 1D rebuild's amplitudes, ``R + T = 1``, silently.

    Measured <= 1.1e-13 (MU614; the others <= 3.8e-14); develop 3.0e-2 .. 2.5 (SMM) and 2.1e-7 ..
    2.5e-4 (ETM), no warning. Catches the gate off, the pair orthonormalised without the Newton
    step (1.4e-11 .. 5.7e-10), the coupling dropped or misplaced, and the second Newton column
    without its coupling term.
    """
    scene = PATTERNED[name]
    for rel in _RELS:
        wavelength = scene['wavelength'] * (1 + rel)
        ref = _vector(_solver(scene, wavelength=wavelength).solve(auto_detune=False))
        del rebuilt[:]
        got = _vector(_strict(lambda: _first_order(scene, algo, wavelength=wavelength)
                              .solve(auto_detune=False)))
        assert rebuilt, f'{name} rel {rel:g}: the pair was not rebuilt'
        err = np.max(np.abs(got[2:] - ref[2:]))
        assert err < 1e-12, f'{name} {algo} rel {rel:g}: {err:.1e} from the 1D rebuild'
        if 'mu' not in scene:
            assert abs(np.real(got[0] + got[1]) - 1) < 1e-12


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('name', ['PA616', 'PL952'])
def test_offsets_into_the_gate_edge(name, algo):
    """Offsets 1e-6 .. 2e-3 either side, the gate's edge among them: <= 1e-12 from the 1D rebuild.

    The pair's sine grows like ``15 |rel|`` (PA616), so ``rel = 2e-3`` sits just outside the gate.
    Develop: 1.3e-5 .. 9.3e-5 (SMM) at 1e-6.
    """
    scene = PATTERNED[name]
    for rel in (1e-6, -1e-6, 1e-5, -1e-5, 1e-4, -1e-4, 1e-3, -1e-3, 2e-3, -2e-3, 5e-3):
        wavelength = scene['wavelength'] * (1 + rel)
        ref = _vector(_solver(scene, wavelength=wavelength).solve(auto_detune=False))
        got = _vector(_strict(lambda: _first_order(scene, algo, wavelength=wavelength)
                              .solve(auto_detune=False)))
        assert np.max(np.abs(got[2:] - ref[2:])) < 1e-12, f'{name} {algo} rel {rel:g}'


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('name, side', [('PA616', 'top'), ('PA616', 'bottom'), ('PL952', 'top'),
                                        ('PL952', 'bottom'), ('MU614', 'top')])
def test_a_y_invariant_raster_rebuilds_every_row(name, side, algo, rebuilt):
    """The structure as a y-invariant raster (``truncation_order = (M, 1)``) forced onto the
    first-order path: its ``v = 0`` row is the 1D rebuild's answer.

    The x-cutoff condition does not depend on ``ky``, so all three rows of orders reach it at the
    same wavelength: three defective pairs per direction, all rebuilt eagerly. (Lit from below,
    ``kx0 = n_bot sin(theta) cos(phi)`` moves the cutoff off this wavelength unless
    ``phi = pi/2``, so MU614 is lit from the top only.) Measured <= 1.6e-13; develop up to 7.9
    (SMM) and 9e-5 (ETM). Catches a gate that rebuilds one pair per direction (the other rows
    keep their O(1) error; that version also warned).
    """
    scene = dict(PATTERNED[name], incidence_side=side)
    ref = _vector(_solver(scene).solve(auto_detune=False))
    s = _solver(scene, algo=algo, two_d=True)
    s.formulation = 'first_order'
    del rebuilt[:]
    r = _strict(lambda: s.solve(auto_detune=False))
    row = list(_np(r.orders_v)).index(0)
    got = np.concatenate([[0, 0], _np(r.r_xyz)[row].ravel(), _np(r.t_xyz)[row].ravel()])
    assert len(rebuilt) >= 6, f'{len(rebuilt)} pairs rebuilt'
    assert np.max(np.abs(got[2:] - ref[2:])) < 1e-12, f'{name} {algo} {side}'


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('name', ['R1 air spacer', 'magnetic spacer', 'lossless double negative',
                                  'backside', 'theta 1.4'])
def test_a_uniform_cutoff_on_the_first_order_path(name, algo):
    """Uniform cutoffs were never the first-order paths' trouble; they stay at the 2D answer."""
    scene = UNIFORM[name]
    ref = _vector(_solver(scene).solve(auto_detune=False))
    got = _vector(_strict(lambda: _first_order(scene, algo).solve(auto_detune=False)))
    assert np.max(np.abs(got[2:] - ref[2:])) < 1e-12, name


# ------------------------------------------------------------------- 2. the regular pair

def _layer(name='PA616'):
    """The patterned layer's first-order operator at its exceptional point, and its ``n``."""
    scene = PATTERNED[name]
    seen = []
    original = fo._layer_split

    def spy(xp, M, n, k0_d=None, **kwargs):
        seen.append((np.asarray(M), n, k0_d))
        return original(xp, M, n, k0_d, **kwargs)
    fo._layer_split = spy
    try:
        _first_order(scene).solve(auto_detune=False)
    finally:
        fo._layer_split = original
    return seen[0]


@pytest.mark.parametrize('name', ['PA616', 'MU614'])
def test_the_rebuilt_pair_is_an_invariant_subspace(name):
    """``M B = B T`` to round-off, ``B`` orthonormal, ``T`` triangular with one coupling.

    Measured residual <= 4e-16 ``|M|``; without the Newton step up to 5e-10.
    """
    M, n, k0_d = _layer(name)
    lam_f, psi_f, lam_b, psi_b, coupled = fo._layer_split(np, M, n, k0_d)
    assert coupled
    for lam, psi in ((lam_f, psi_f), (lam_b, psi_b)):
        assert lam.ndim == 2
        coupling = lam - np.diag(np.diag(lam))
        rows, cols = np.nonzero(coupling)
        assert len(rows) == 1 and rows[0] != cols[0]
        assert np.linalg.norm(M @ psi - psi @ lam) < 1e-15 * np.linalg.norm(M) * psi.shape[1] ** 0.5
        pair = psi[:, [rows[0], cols[0]]]
        assert np.allclose(pair.conj().T @ pair, np.eye(2), atol=1e-14)


def test_the_split_sylvester_solves_its_equation():
    """``X (D + N) - (D + N) X = R`` on the cross-direction entries, ``0`` within a direction."""
    rng = np.random.default_rng(3)
    n = 8
    forward = np.array([True] * 4 + [False] * 4)
    D = rng.normal(size=n) + 1j * rng.normal(size=n)
    N = np.zeros((n, n), complex)
    N[1, 3] = 0.7 - 0.2j            # forward pair (1, 3)
    N[6, 4] = -0.4 + 0.9j           # backward pair (6, 4): rows and columns stay disjoint
    R = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    separate = forward[:, None] != forward[None, :]
    gap = np.where(separate, D[None, :] - D[:, None], 1.0)
    X = modal_gradient._split_sylvester(np, N, gap, R, separate)
    L = np.diag(D) + N
    assert np.all(X[~separate] == 0)
    assert np.max(np.abs((X @ L - L @ X - R)[separate])) < 1e-13


def test_a_pair_beside_a_coincident_root_is_refused_loudly():
    """Two identical decoupled Jordan blocks share their root: no pair basis spans them.

    The split keeps the eigenvector basis and says so (``ExceptionalPointWarning``).
    """
    J = np.array([[0.3j, 1.0], [0.0, 0.3j]])
    K = np.array([[-0.3j, 1.0], [0.0, -0.3j]])
    M = np.zeros((8, 8), complex)
    for k, block in enumerate((J, J, K, K)):
        M[2 * k:2 * k + 2, 2 * k:2 * k + 2] = block
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8)))
    with pytest.warns(wavebit.ExceptionalPointWarning):
        fo._layer_split(np, Q @ M @ Q.conj().T, 2)


def _jordan_beside(offset, seed=0):
    """A ``12 x 12`` first-order operator (``n_orders = 3``): a Jordan pair at ``0.3j`` beside a
    plain root at ``0.3j + offset``, with mirror roots below, turned by a random unitary."""
    blocks = [np.array([[0.3j, 1.0], [0.0, 0.3j]]), np.array([[0.3j + offset]]),
              np.array([[0.7j]]), np.array([[0.9j]]), np.array([[1.1j]]),
              np.array([[-0.3j, 1.0], [0.0, -0.3j]]), np.array([[-0.5j]]), np.array([[-0.8j]]),
              np.array([[-1.0j]]), np.array([[-1.2j]])]
    M = np.zeros((12, 12), complex)
    k = 0
    for block in blocks:
        m = block.shape[0]
        M[k:k + m, k:k + m] = block
        k += m
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(12, 12)) + 1j * rng.normal(size=(12, 12)))
    return Q @ M @ Q.conj().T


@pytest.mark.parametrize('offset', [0.0, 1e-9, 1e-5])
def test_a_pair_beside_a_root_it_nearly_shares_is_refused_loudly(offset):
    """A third root within ``_PAIR_SEPARATION`` of the pair: no subspace of the pair's own.

    The third mode is not parallel to the pair, so only the separation test sees it. Without
    the refusal the rebuilt pair's residual was 6.0e-11 at ``offset = 1e-5`` and worse nearer.
    """
    with pytest.warns(wavebit.ExceptionalPointWarning):
        fo._layer_split(np, _jordan_beside(offset), 3)


def test_a_pair_beyond_the_separation_floor_is_rebuilt_quietly():
    """At ``1e-3`` the pair is rebuilt, silently, its residual at most 1.2e-12 (measured)."""
    M = _jordan_beside(1e-3)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        lam_f, psi_f, _, _, coupled = fo._layer_split(np, M, 3)
    assert coupled
    assert np.linalg.norm(M @ psi_f - psi_f @ lam_f) < 5e-12 * np.linalg.norm(M)


# --------------------------------------------------------------------- 3. derivatives

@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('name', ['PA616', 'MU614', 'PL952'])
def test_derivatives_at_a_patterned_exceptional_point(name, algo):
    """``jax.grad`` (raster, wavelength, thickness, theta), ``jax.jvp``, torch reverse and
    ``forward_ad``: the 1D rebuild's ``jax.grad``.

    Measured <= 7.1e-12 relative; develop 1.4e-2 .. 8.3e3 (SMM), up to 5.7e-2 (ETM). Catches the coupled modal generator differentiated as a diagonal one, a
    Sylvester pass dropped, and the exponential's tangent taken from the formula.
    """
    jax = pytest.importorskip('jax')
    torch = pytest.importorskip('torch')
    from torch.autograd import forward_ad
    jnp = jax.numpy
    scene = PATTERNED[name]
    reference = _solver(scene, backend='jax')    # first: a JAX solver lifts x64 before any array
    args = (jnp.asarray(np.asarray(scene['structure'], float)), jnp.asarray(scene['wavelength']),
            jnp.asarray(float(scene['thickness'][0])), jnp.asarray(scene['theta']))

    def objective(s):
        def f(u, w, t, th):
            r = s.solve(structure=u, wavelength=w, thickness=jnp.stack([t]), theta=th,
                        auto_detune=False)
            return jnp.sum(jnp.real(r.R))
        return f
    ref = jax.grad(objective(reference), argnums=(0, 1, 2, 3))(*args)
    f = objective(_first_order(scene, algo, 'jax'))
    got = jax.grad(f, argnums=(0, 1, 2, 3))(*args)
    for label, a, b in zip(('raster', 'wavelength', 'thickness', 'theta'), got, ref):
        assert _rel(a, b) < 1e-10, f'{name} {algo} {label}: {_rel(a, b):.1e}'
    jw = jax.jvp(lambda w: f(args[0], w, args[2], args[3]), (args[1],), (jnp.asarray(1.0),))[1]
    assert _rel(jw, ref[1]) < 1e-10
    u = torch.tensor(np.asarray(scene['structure'], float), requires_grad=True)
    _first_order(scene, algo, 'torch', structure=u).solve(auto_detune=False).R.real.sum().backward()
    assert _rel(u.grad.numpy(), ref[0]) < 1e-10
    with forward_ad.dual_level():
        w = forward_ad.make_dual(torch.tensor(scene['wavelength'], dtype=torch.float64),
                                 torch.tensor(1., dtype=torch.float64))
        r = _first_order(scene, algo, 'torch', wavelength=w).solve(auto_detune=False)
        tangent = float(forward_ad.unpack_dual(r.R.real.sum()).tangent)
    assert _rel(tangent, ref[1]) < 1e-10


# ----------------------------------------------------------------------------- 4. fields

_GLASS = dict(PATTERNED['PA616'], thickness=[150., 300.], structure=np.concatenate(
    [np.full((1, 1, 20), 1.7), PATTERNED['PA616']['structure']], axis=0))
_FIELDS = {'PA616': PATTERNED['PA616'], 'MU614': PATTERNED['MU614'],
           'glass over PA616, top lit': _GLASS,
           'glass over PA616, bottom lit': dict(_GLASS, incidence_side='bottom')}


@pytest.mark.parametrize('name', list(_FIELDS))
def test_fields_and_field_gradients_at_a_patterned_exceptional_point(name):
    """Fields (eager and jitted) and field gradients (``jax.grad``, ``jit(grad)``, torch): the 1D
    rebuild's. Measured: fields <= 2.0e-13 of ``max|F|``, gradients <= 7.1e-13; SMM = ETM to
    3.3e-15. Develop: SMM fields 1.3e-2 .. 7.3, field gradients to 5.3e2. Catches a record that
    loses its ninth slot (the field's exponential then takes the formula's tangent).
    """
    jax = pytest.importorskip('jax')
    torch = pytest.importorskip('torch')
    jnp = jax.numpy
    scene = _FIELDS[name]
    reference = _solver(scene, backend='jax')    # first: a JAX solver lifts x64 before any array
    xs = jnp.asarray([-200., 0., 130.])
    zs = jnp.asarray(np.array([0.13, 0.5, 0.87]) * sum(scene['thickness']))
    u0 = jnp.asarray(np.asarray(scene['structure'], float))
    w0 = jnp.asarray(scene['wavelength'])

    def build(s):
        def field(u, w):
            r = s.solve(structure=u, wavelength=w, auto_detune=False, answers='field')
            return r.reconstruct_field(sample_x=xs, sample_y=jnp.asarray([0.]), sample_z=zs)
        return field, lambda u, w: jnp.sum(jnp.abs(field(u, w)[..., :3]) ** 2)
    ref_field, ref_F = build(reference)
    ref = np.asarray(ref_field(u0, w0))
    gref = jax.grad(ref_F, argnums=(0, 1))(u0, w0)
    fields = {}
    for algo in ('ETM', 'SMM'):
        field, F = build(_first_order(scene, algo, 'jax'))
        fields[algo] = np.asarray(field(u0, w0))
        scale = np.max(np.abs(ref))
        assert np.max(np.abs(fields[algo] - ref)) < 1e-12 * scale, f'{name} {algo} field'
        assert np.max(np.abs(np.asarray(jax.jit(field)(u0, w0)) - ref)) < 1e-12 * scale
        for g in (jax.grad(F, argnums=(0, 1))(u0, w0), jax.jit(jax.grad(F, argnums=(0, 1)))(u0, w0)):
            for label, a, b in zip(('raster', 'wavelength'), g, gref):
                err = np.max(np.abs(np.asarray(a) - np.asarray(b))) / np.max(np.abs(np.asarray(b)))
                assert err < 1e-11, f'{name} {algo} field gradient, {label}: {err:.1e}'
        u = torch.tensor(np.asarray(scene['structure'], float), requires_grad=True)
        r = _first_order(scene, algo, 'torch', structure=u).solve(auto_detune=False, answers='field')
        f = r.reconstruct_field(sample_x=torch.tensor(np.asarray(xs)), sample_y=torch.tensor([0.]),
                                sample_z=torch.tensor(np.asarray(zs)))
        (f[..., :3].abs() ** 2).sum().backward()
        g0 = np.asarray(gref[0])
        assert np.max(np.abs(u.grad.numpy() - g0)) < 1e-11 * np.max(np.abs(g0))


# ------------------------------------------------------------------ 5. under a trace

@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('name', ['PA616', 'MU614'])
def test_a_patterned_exceptional_point_under_an_abstract_trace(name, algo):
    """``jit``, ``jit(vmap)`` over a sweep through the point and off the band, ``jit(grad)`` and
    ``torch.func.vmap`` with autograd: the eager 1D rebuild. Every traced layer takes the
    regular-pair layout with its masks as data. Measured: values <= 9.7e-14, gradients
    <= 4.5e-12; develop 0.21 .. 5.2 (SMM values), to 5.8e3 (SMM gradients). Catches masks that
    never engage under a trace and the traced tangent taken from the formula.
    """
    jax = pytest.importorskip('jax')
    torch = pytest.importorskip('torch')
    jnp = jax.numpy
    scene = PATTERNED[name]
    wls = np.array([scene['wavelength'] * (1 + r) for r in (0.0, 1e-12, -1e-9, 1e-3, 1e-2)])

    def amps(r, i=None):
        parts = [np.asarray(r.r_xyz), np.asarray(r.t_xyz)]
        return np.concatenate([(p if i is None else p[i]).ravel() for p in parts])
    ref = [amps(_solver(scene, wavelength=w).solve(auto_detune=False)) for w in wls]
    s1 = _solver(scene, backend='jax')
    gref = [float(jax.grad(lambda w: jnp.sum(jnp.real(s1.solve(wavelength=w, auto_detune=False).R)))(
        jnp.asarray(w))) for w in wls]
    s = _first_order(scene, algo, 'jax')
    f = lambda w: s.solve(wavelength=w, auto_detune=False)       # noqa: E731
    assert max(np.max(np.abs(amps(jax.jit(f)(jnp.asarray(w))) - r)) for w, r in zip(wls, ref)) < 1e-12
    swept = jax.jit(jax.vmap(f))(jnp.asarray(wls))
    assert max(np.max(np.abs(amps(swept, i) - ref[i])) for i in range(len(wls))) < 1e-12
    jg = jax.jit(jax.grad(lambda w: jnp.sum(jnp.real(f(w).R))))
    assert max(abs(float(jg(jnp.asarray(w))) - g) / abs(g) for w, g in zip(wls, gref)) < 1e-10
    st = _first_order(scene, algo, 'torch')
    w = torch.tensor(wls, requires_grad=True)
    R = torch.func.vmap(lambda x: st.solve(wavelength=x, auto_detune=False).R.real.sum())(w)
    R.sum().backward()
    assert max(abs(float(a) - g) / abs(g) for a, g in zip(w.grad, gref)) < 1e-10


def test_a_traced_y_invariant_raster_matches_eager():
    """MU614's three simultaneous pairs per direction survive the SMM value trace.

    One slot left an amplitude error of 2.0 before overflow was made loud.
    The eager first-order answer is independently pinned to the 1D rebuild above.
    """
    jax = pytest.importorskip('jax')
    s = _first_order(PATTERNED['MU614'], 'SMM', 'jax', two_d=True)
    w = jax.numpy.asarray(PATTERNED['MU614']['wavelength'])

    def amplitudes(wavelength):
        result = s.solve(wavelength=wavelength, auto_detune=False)
        return jax.numpy.concatenate([result.r_xyz.ravel(), result.t_xyz.ravel()])

    eager = np.asarray(amplitudes(w))
    traced = np.asarray(jax.jit(amplitudes)(w))
    np.testing.assert_allclose(traced, eager, atol=1e-12, rtol=0)


def test_a_traced_y_invariant_raster_gradient_matches_eager():
    """MU614's SMM wavelength gradient uses every row's regular pair under jit.

    One slot left a relative derivative error of 2.4e2; three slots measured
    9.6e-13 against the 1D reference in the design experiment.
    """
    jax = pytest.importorskip('jax')
    s = _first_order(PATTERNED['MU614'], 'SMM', 'jax', two_d=True)
    w = jax.numpy.asarray(PATTERNED['MU614']['wavelength'])

    def reflected(wavelength):
        result = s.solve(wavelength=wavelength, auto_detune=False)
        return jax.numpy.sum(jax.numpy.real(result.R))

    derivative = jax.grad(reflected)
    eager = float(derivative(w))
    traced = float(jax.jit(derivative)(w))
    assert abs(traced - eager) < 1e-12 * abs(eager)


@pytest.mark.parametrize('slots', [1, 2])
def test_more_pairs_than_the_trace_holds_is_loud(slots):
    """A separated synthetic layer with slots + 1 Jordan pairs per direction returns NaN.

    Eager rebuilding ignores the slot limit; a trace with enough slots stays
    finite and satisfies the invariant-subspace relation. Thus a generic split
    failure cannot satisfy the overflow assertion.
    """
    jax = pytest.importorskip('jax')
    wavebit.enable_x64()
    count = slots + 1
    size = 4 * count
    M = np.zeros((size, size), complex)
    for direction in (0, 1):
        for pair in range(count):
            start = 2 * (direction * count + pair)
            root = (1 if direction == 0 else -1) * (0.3 + 0.4 * pair) * 1j
            M[start:start + 2, start:start + 2] = [[root, 0.2], [0., root]]
    rng = np.random.default_rng(19)
    Q, _ = np.linalg.qr(rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size)))
    M = Q @ M @ Q.conj().T

    def split(matrix, capacity):
        return fo._layer_split(jax.numpy, matrix, count, slots=capacity)[:4]

    matrix = jax.numpy.asarray(M)
    eager = split(matrix, slots)
    enough = jax.jit(lambda a: split(a, count))(matrix)
    for modes in (eager, enough):
        for generator, basis in ((modes[0], modes[1]), (modes[2], modes[3])):
            generator, basis = np.asarray(generator), np.asarray(basis)
            assert np.isfinite(generator).all()
            assert np.linalg.norm(M @ basis - basis @ generator) < 1e-11 * np.linalg.norm(M)
    overflow = jax.jit(lambda a: split(a, slots))(matrix)
    assert np.isnan(np.asarray(overflow[0])).all()
    assert np.isnan(np.asarray(overflow[2])).all()


# --------------------------------------------------------------------- 6. off the band

def _random_first_order(count, seed=20261012):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        n = int(rng.choice([16, 20, 24]))
        st = np.full((1, 1, n), rng.uniform(1.0, 3.0))
        a, b = sorted(rng.integers(0, n, 2))
        st[0, 0, a:b + 1] = rng.uniform(1.5, 14.0)
        out.append(dict(structure=st, wavelength=float(rng.uniform(350, 1700)),
                        period=float(rng.uniform(250, 1800)), thickness=[float(rng.uniform(30, 700))],
                        top=1.0, bot=float(rng.choice([1.0, 1.45, 3.48])),
                        theta=float(rng.uniform(0.02, 1.3)), phi=float(rng.uniform(0, np.pi)),
                        psi=float(rng.uniform(0, np.pi)), truncation=int(rng.integers(2, 10))))
    return out


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_off_the_band_nothing_moves(algo, rebuilt):
    """Random first-order solves the gate does not route are bit-for-bit the eigenvector split's,
    and quiet. The control is the same solve with the gate closed (``_PARALLEL_SIN = 0``).
    Catches a gate opened wide (its refusals warn on ordinary layers).
    """
    routed = 0
    for scene in _random_first_order(12):
        del rebuilt[:]
        open_ = _strict(lambda: _first_order(scene, algo).solve(auto_detune=False))
        if rebuilt:
            routed += 1
            continue
        saved, fo._PARALLEL_SIN = fo._PARALLEL_SIN, 0.0
        try:
            closed = _first_order(scene, algo).solve(auto_detune=False)
        finally:
            fo._PARALLEL_SIN = saved
        for key in ('R', 'T', 'r_xyz', 't_xyz'):
            assert np.array_equal(_np(getattr(open_, key)), _np(getattr(closed, key))), key
    assert routed < 3
