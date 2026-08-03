"""Single-source Fourier factorization (fieldwave._core.factorization) — Phase 4a/4b gate.

Beyond the usual parity/cross-backend checks, this closes the differentiability gap that
motivated the port: with the factorization inside the core, the gradient runs end-to-end from
the *design variable* — the ``ucell`` raster — through to diffraction efficiency, rather than
stopping at the convolution matrices.
"""
import pathlib

import numpy as np
import pytest

import fieldwave
from fieldwave._core.factorization import (
    to_conv_mat_raster_discrete as core_dfs,
    to_conv_mat_raster_nvf as core_nvf,
    to_conv_mat_raster_continuous as core_cfs,
    to_conv_mat_vector as core_vector,
    normal_field_for_ucell,
    compression_indices,
    compress_cell,
)
from fieldwave._core.solve_1d import solve_1d
from fieldwave._core.solve_2d import solve_2d


def _rand(shape, seed):
    return np.random.default_rng(seed).uniform(1.0, 4.0, size=shape)


# The legacy builders these were originally checked against are gone; their outputs live on
# as frozen reference data (tests_verification/gen_goldens_factorization.py).
GOLDEN = np.load(pathlib.Path(__file__).parent / 'golden_legacy_factorization.npz')


CASES = [
    # name, ucell, fto_x, fto_y
    ('1d_1layer', _rand((1, 1, 8), 0), 4, 0),
    ('2d_1layer', _rand((1, 5, 5), 1), 2, 2),
    ('2layer_rect', _rand((2, 4, 6), 2), 2, 1),
    ('3layer_1d', _rand((3, 1, 10), 3), 5, 0),
]


@pytest.mark.parametrize('case', CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize('enhanced_dfs', [True, False], ids=['enhanced', 'plain'])
def test_factorization_bit_parity_vs_legacy(case, enhanced_dfs):
    name, ucell, fto_x, fto_y = case
    tag = 'enhanced' if enhanced_dfs else 'plain'
    got = core_dfs(ucell, fto_x, fto_y, enhanced_dfs=enhanced_dfs)
    for slot, b in zip(('epx', 'epy', 'epz_i'), got):
        a = GOLDEN[f'dfs__{name}__{tag}__{slot}']
        assert np.array_equal(a, np.array(b)), (slot, np.max(np.abs(a - np.array(b))))


def _to(backend, x):
    if backend == 'jax':
        import jax.numpy as jnp
        return jnp.asarray(x)
    import torch
    return torch.as_tensor(np.asarray(x))


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('case', CASES, ids=[c[0] for c in CASES])
def test_factorization_cross_backend(case, backend):
    pytest.importorskip(backend)
    _, ucell, fto_x, fto_y = case
    ref = core_dfs(ucell, fto_x, fto_y)
    got = core_dfs(_to(backend, ucell), fto_x, fto_y)
    for name, a, b in zip(('epx', 'epy', 'epz_i'), ref, got):
        assert np.allclose(np.array(a), np.asarray(b), atol=1e-10, rtol=0), name


def test_factorization_feeds_solve_bit_exactly():
    """core factorization -> core solve reproduces the legacy end-to-end result exactly."""
    ucell = np.array([[[3., 3., 3., 1., 1., 1.]]])
    wl, n_top, n_bot, theta, period, fto = 0.777, 2.0, 1.0, 0.3, [0.77], [6, 0]
    thickness = [0.15]

    mee = fieldwave.call_mee(backend=0, n_top=n_top, n_bot=n_bot, theta=theta, phi=None, pol=0,
                          fto=fto, period=period, wavelength=wl, thickness=thickness, ucell=ucell)
    legacy = mee.conv_solve()
    kx, _ = mee.get_kx_ky_vector(wl)

    epx, epy, epz = core_dfs(ucell, fto[0], fto[1])
    ri, ti = solve_1d(0, wl, kx, n_top, n_bot, mee.theta, mee.thickness, epx, epy, epz)

    assert np.array_equal(np.array(legacy.de_ri), np.array(ri))
    assert np.array_equal(np.array(legacy.de_ti), np.array(ti))


# ------------------------------------------------- end-to-end differentiability (the point)

def _chain_1d(fac_kwargs, ucell, kx, theta):
    epx, epy, epz = core_dfs(ucell, **fac_kwargs)
    _, ti = solve_1d(0, 0.777, kx, 2.0, 1.0, theta, [0.15], epx, epy, epz,
                     same_material=[False])
    return ti


def test_end_to_end_gradient_wrt_ucell_1d():
    """d(sum de_ti)/d(ucell raster): jax == torch == finite-diff, through factorization."""
    jax = pytest.importorskip('jax')
    torch = pytest.importorskip('torch')
    import jax.numpy as jnp

    ucell = np.array([[[3., 3., 3., 1., 1., 1.]]])
    direction = np.ones_like(ucell)
    fto = [6, 0]
    mee = fieldwave.call_mee(backend=0, n_top=2.0, n_bot=1.0, theta=0.3, phi=None, pol=0,
                          fto=fto, period=[0.77], wavelength=0.777, thickness=[0.15], ucell=ucell)
    kx, _ = mee.get_kx_ky_vector(0.777)
    theta = complex(mee.theta)
    fk = dict(fto_x=fto[0], fto_y=fto[1])

    def f_np(s):
        return float(np.sum(np.array(_chain_1d(fk, ucell + s * direction, kx, theta))))
    h = 1e-6
    g_fd = (f_np(h) - f_np(-h)) / (2 * h)

    g_jax = float(jax.grad(lambda s: jnp.sum(_chain_1d(
        fk, jnp.asarray(ucell) + s * jnp.asarray(direction),
        jnp.asarray(kx), theta)).real)(0.0))

    s = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    torch.sum(_chain_1d(fk, torch.as_tensor(ucell) + s * torch.as_tensor(direction),
                        torch.as_tensor(kx), theta)).real.backward()
    g_torch = float(s.grad)

    assert abs(g_jax - g_torch) < 1e-8, (g_jax, g_torch)
    assert abs(g_jax - g_fd) < 1e-6, (g_jax, g_fd)


def test_end_to_end_gradient_wrt_ucell_2d():
    """Same end-to-end gradient through the 2D coupled block chain."""
    jax = pytest.importorskip('jax')
    torch = pytest.importorskip('torch')
    import jax.numpy as jnp

    ucell = _rand((1, 4, 4), 7)
    direction = np.ones_like(ucell)
    fto = [1, 1]
    wl, n_top, n_bot = 0.9, 1.0, 1.0
    mee = fieldwave.call_mee(backend=0, n_top=n_top, n_bot=n_bot, theta=0.2, phi=0.4, psi=0.9,
                          fto=fto, period=[1.0, 1.0], wavelength=wl, thickness=[0.15], ucell=ucell)
    kx, ky = mee.get_kx_ky_vector(wl)
    theta, psi = complex(mee.theta), complex(mee.psi)

    def chain(u, kx_, ky_, th_, ps_):
        epx, epy, epz = core_dfs(u, fto[0], fto[1])
        _, ti = solve_2d(wl, kx_, ky_, n_top, n_bot, th_, ps_, [0.15], epx, epy, epz,
                         same_material=[False])
        return ti

    def f_np(s):
        return float(np.sum(np.array(chain(ucell + s * direction, kx, ky, theta, psi))))
    h = 1e-6
    g_fd = (f_np(h) - f_np(-h)) / (2 * h)

    g_jax = float(jax.grad(lambda s: jnp.sum(chain(
        jnp.asarray(ucell) + s * jnp.asarray(direction), jnp.asarray(kx), jnp.asarray(ky), theta, psi)).real)(0.0))

    s = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    torch.sum(chain(torch.as_tensor(ucell) + s * torch.as_tensor(direction),
                    torch.as_tensor(kx), torch.as_tensor(ky), theta, psi)).real.backward()
    g_torch = float(s.grad)

    assert abs(g_jax - g_torch) < 1e-8, (g_jax, g_torch)
    assert abs(g_jax - g_fd) < 1e-6, (g_jax, g_fd)


# ================================================================== NVF (fourier_type=2)

NVF_CASES = [
    ('nvf_5x5', _rand((1, 5, 5), 10), 2, 2),
    ('nvf_2layer', _rand((2, 4, 6), 11), 2, 2),
    ('nvf_binary', np.where(_rand((1, 8, 8), 12) > 2.5, 3.5, 1.0), 2, 2),
]


@pytest.mark.parametrize('case', NVF_CASES, ids=[c[0] for c in NVF_CASES])
def test_nvf_bit_parity_vs_legacy(case):
    name, ucell, fto_x, fto_y = case
    got = core_nvf(ucell, fto_x, fto_y)
    for slot, b in zip(('epx', 'epy', 'epxy', 'epyx', 'epz_i'), got):
        a = GOLDEN[f'nvf__{name}__{slot}']
        assert np.array_equal(a, np.array(b)), (slot, np.max(np.abs(a - np.array(b))))


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('case', NVF_CASES, ids=[c[0] for c in NVF_CASES])
def test_nvf_runs_on_every_backend(case, backend):
    """NVF on jax / torch — a capability the legacy per-backend builders never had.

    ``call_mee(backend=1|2, fourier_type=2)`` raises ValueError on the legacy path; the
    single-source builder makes the same factorization available on all three.
    """
    pytest.importorskip(backend)
    _, ucell, fto_x, fto_y = case
    ref = core_nvf(ucell, fto_x, fto_y)
    got = core_nvf(_to(backend, ucell), fto_x, fto_y)
    for a, b in zip(ref, got):
        assert np.allclose(np.array(a), np.asarray(b), atol=1e-12, rtol=0)


def test_nvf_now_works_on_every_backend():
    """NVF used to be NumPy-only; conv_solve now routes every backend through the core.

    ``call_mee(backend=1|2, fourier_type=2)`` raised ValueError before the factorization was
    wired — the per-backend builders simply had no NVF branch.
    """
    ucell = _rand((1, 5, 5), 13)
    opts = dict(n_top=1.0, n_bot=1.0, theta=0.0, phi=0.0, psi=0.0, fto=[2, 2],
                period=[1.0, 1.0], wavelength=0.9, thickness=[0.2], ucell=ucell,
                fourier_type=2)
    ref = fieldwave.call_mee(backend=0, **opts).conv_solve()
    for backend in (1, 2):
        pytest.importorskip('jax' if backend == 1 else 'torch')
        got = fieldwave.call_mee(backend=backend, **opts).conv_solve()
        assert np.allclose(np.asarray(ref.res.de_ti), np.asarray(got.res.de_ti),
                           atol=1e-12, rtol=0), backend


def test_nvf_end_to_end_gradient_wrt_ucell():
    """d(sum de_ti)/d(ucell) through the NVF factorization: jax == torch == finite-diff.

    The normal vector field is static geometry (precomputed once), so gradients measure the
    permittivity dependence with the interface directions held fixed.
    """
    jax = pytest.importorskip('jax')
    torch = pytest.importorskip('torch')
    import jax.numpy as jnp

    ucell = np.where(_rand((1, 6, 6), 14) > 2.5, 3.5, 1.0)
    direction = np.ones_like(ucell)
    fto = [1, 1]
    wl, n_top, n_bot, d = 0.9, 1.0, 1.0, 0.2
    normal_field = normal_field_for_ucell(ucell, fto[0], fto[1])

    mee = fieldwave.call_mee(backend=0, n_top=n_top, n_bot=n_bot, theta=0.2, phi=0.4, psi=0.9,
                          fto=fto, period=[1.0, 1.0], wavelength=wl, thickness=[d], ucell=ucell)
    kx, ky = mee.get_kx_ky_vector(wl)
    theta, psi = complex(mee.theta), complex(mee.psi)

    def chain(u, kx_, ky_, th_, ps_):
        epx, epy, epxy, epyx, epz = core_nvf(u, fto[0], fto[1], normal_field=normal_field)
        _, ti = solve_2d(wl, kx_, ky_, n_top, n_bot, th_, ps_, [d], epx, epy, epz,
                         epxy_conv_all=epxy, epyx_conv_all=epyx, same_material=[False])
        return ti

    def f_np(s):
        return float(np.sum(np.array(chain(ucell + s * direction, kx, ky, theta, psi))))
    h = 1e-6
    g_fd = (f_np(h) - f_np(-h)) / (2 * h)

    g_jax = float(jax.grad(lambda s: jnp.sum(chain(
        jnp.asarray(ucell) + s * jnp.asarray(direction), jnp.asarray(kx), jnp.asarray(ky), theta, psi)).real)(0.0))

    s = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    torch.sum(chain(torch.as_tensor(ucell) + s * torch.as_tensor(direction),
                    torch.as_tensor(kx), torch.as_tensor(ky), theta, psi)).real.backward()
    g_torch = float(s.grad)

    assert abs(g_jax - g_torch) < 1e-8, (g_jax, g_torch)
    assert abs(g_jax - g_fd) < 1e-6, (g_jax, g_fd)


# ============================================ CFS / analytic + vector (fourier_type=1)

CFS_CASES = [
    ('cfs_1d', _rand((1, 1, 8), 20), 4, 0),
    ('cfs_2d', _rand((1, 5, 5), 21), 2, 2),
    ('cfs_binary', np.where(_rand((1, 6, 6), 22) > 2.5, 3.5, 1.0), 2, 2),
    ('cfs_2layer', _rand((2, 4, 6), 23), 2, 1),
]


@pytest.mark.parametrize('case', CFS_CASES, ids=[c[0] for c in CFS_CASES])
def test_cfs_bit_parity_vs_legacy(case):
    name, ucell, fto_x, fto_y = case
    got = core_cfs(ucell, fto_x, fto_y)
    for slot, b in zip(('epx', 'epy', 'epz_i'), got):
        a = GOLDEN[f'cfs__{name}__{slot}']
        assert np.array_equal(a, np.array(b)), (slot, np.max(np.abs(a - np.array(b))))


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('case', CFS_CASES, ids=[c[0] for c in CFS_CASES])
def test_cfs_cross_backend(case, backend):
    pytest.importorskip(backend)
    _, ucell, fto_x, fto_y = case
    ref = core_cfs(ucell, fto_x, fto_y)
    # Compression is static geometry, so it is shared rather than recomputed per backend.
    compression = [compression_indices(ucell[i]) for i in range(ucell.shape[0])]
    got = core_cfs(_to(backend, ucell), fto_x, fto_y, compression=compression)
    for name, a, b in zip(('epx', 'epy', 'epz_i'), ref, got):
        assert np.allclose(np.array(a), np.asarray(b), atol=1e-12, rtol=0), name


def test_vector_path_bit_parity_vs_legacy():
    """Vector-modelled layers (explicit segment edges, no compression) match the legacy CFS."""
    info = [[GOLDEN[f'vector__in{i}__cell'], GOLDEN[f'vector__in{i}__x'],
             GOLDEN[f'vector__in{i}__y']] for i in (0, 1)]
    got = core_vector(info, 2, 2)
    for slot, b in zip(('epx', 'epy', 'epz_i'), got):
        a = GOLDEN[f'vector__{slot}']
        assert np.array_equal(a, np.array(b)), (slot, np.max(np.abs(a - np.array(b))))


def test_compression_indices_match_legacy():
    """The extracted static geometry step reproduces the legacy compression exactly."""
    import array_api_compat as ac
    for seed in (26, 27):
        ucell = GOLDEN[f'compress__{seed}__ucell']
        row_idx, col_idx, x, y = compression_indices(ucell)
        got_cell = compress_cell(ac.array_namespace(ucell), ucell, row_idx, col_idx)
        assert np.array_equal(GOLDEN[f'compress__{seed}__cell'], np.array(got_cell))
        assert np.array_equal(GOLDEN[f'compress__{seed}__x'], x)
        assert np.array_equal(GOLDEN[f'compress__{seed}__y'], y)


def test_cfs_end_to_end_gradient_wrt_ucell():
    """d(sum de_ti)/d(ucell) through the analytic CFS path: jax == torch == finite-diff."""
    jax = pytest.importorskip('jax')
    torch = pytest.importorskip('torch')
    import jax.numpy as jnp

    ucell = np.array([[[3., 3., 3., 1., 1., 1.]]])
    direction = np.ones_like(ucell)
    fto = [6, 0]
    wl, n_top, n_bot, d = 0.777, 2.0, 1.0, 0.15
    compression = [compression_indices(ucell[i]) for i in range(ucell.shape[0])]

    mee = fieldwave.call_mee(backend=0, n_top=n_top, n_bot=n_bot, theta=0.3, phi=None, pol=0,
                          fto=fto, period=[0.77], wavelength=wl, thickness=[d], ucell=ucell)
    kx, _ = mee.get_kx_ky_vector(wl)
    theta = complex(mee.theta)

    def chain(u, kx_):
        epx, epy, epz = core_cfs(u, fto[0], fto[1], compression=compression)
        _, ti = solve_1d(0, wl, kx_, n_top, n_bot, theta, [d], epx, epy, epz,
                         same_material=[False])
        return ti

    def f_np(s):
        return float(np.sum(np.array(chain(ucell + s * direction, kx))))
    h = 1e-6
    g_fd = (f_np(h) - f_np(-h)) / (2 * h)

    g_jax = float(jax.grad(lambda s: jnp.sum(chain(
        jnp.asarray(ucell) + s * jnp.asarray(direction), jnp.asarray(kx))).real)(0.0))

    s = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    torch.sum(chain(torch.as_tensor(ucell) + s * torch.as_tensor(direction),
                    torch.as_tensor(kx))).real.backward()
    g_torch = float(s.grad)

    assert abs(g_jax - g_torch) < 1e-8, (g_jax, g_torch)
    assert abs(g_jax - g_fd) < 1e-6, (g_jax, g_fd)
