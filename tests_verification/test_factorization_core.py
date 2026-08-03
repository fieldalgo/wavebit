"""Single-source DFS factorization (fieldwave._core.factorization) — Phase 4a gate.

Beyond the usual parity/cross-backend checks, this closes the differentiability gap that
motivated the port: with the factorization inside the core, the gradient runs end-to-end from
the *design variable* — the ``ucell`` raster — through to diffraction efficiency, rather than
stopping at the convolution matrices.
"""
import numpy as np
import pytest

import fieldwave
from fieldwave.on_numpy.emsolver.convolution_matrix import to_conv_mat_raster_discrete as legacy_dfs
from fieldwave._core.factorization import to_conv_mat_raster_discrete as core_dfs
from fieldwave._core.solve_1d import solve_1d
from fieldwave._core.solve_2d import solve_2d


def _rand(shape, seed):
    return np.random.default_rng(seed).uniform(1.0, 4.0, size=shape)


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
    _, ucell, fto_x, fto_y = case
    want = legacy_dfs(ucell, fto_x, fto_y, enhanced_dfs=enhanced_dfs)
    got = core_dfs(ucell, fto_x, fto_y, enhanced_dfs=enhanced_dfs)
    for name, a, b in zip(('epx', 'epy', 'epz_i'), want, got):
        assert np.array_equal(a, np.array(b)), (name, np.max(np.abs(a - np.array(b))))


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
        jnp.asarray(kx), jnp.asarray(theta))).real)(0.0))

    s = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    torch.sum(_chain_1d(fk, torch.as_tensor(ucell) + s * torch.as_tensor(direction),
                        torch.as_tensor(kx), torch.as_tensor(theta))).real.backward()
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
        jnp.asarray(ucell) + s * jnp.asarray(direction), jnp.asarray(kx), jnp.asarray(ky),
        jnp.asarray(theta), jnp.asarray(psi))).real)(0.0))

    s = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    torch.sum(chain(torch.as_tensor(ucell) + s * torch.as_tensor(direction),
                    torch.as_tensor(kx), torch.as_tensor(ky),
                    torch.as_tensor(theta), torch.as_tensor(psi))).real.backward()
    g_torch = float(s.grad)

    assert abs(g_jax - g_torch) < 1e-8, (g_jax, g_torch)
    assert abs(g_jax - g_fd) < 1e-6, (g_jax, g_fd)
