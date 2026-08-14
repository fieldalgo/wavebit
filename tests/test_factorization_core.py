"""Single-source Fourier factorization (wavebit._core.factorization) — Phase 4a/4b gate.

Beyond the usual parity/cross-backend checks, this closes the differentiability gap that
motivated the port: with the factorization inside the core, the gradient runs end-to-end from
the *design variable* — the ``ucell`` raster — through to diffraction efficiency, rather than
stopping at the convolution matrices.
"""
import pathlib

import numpy as np
import pytest

import array_api_compat.numpy as xnp

import wavebit
from wavebit._core.factorization import fourier2d_piecewise_constant
from wavebit._core.factorization import (
    to_conv_mat_raster_sampled as core_dfs,
    to_conv_mat_raster_nvf as core_nvf,
    to_conv_mat_raster_integrated as core_cfs,
    to_conv_mat_block as core_vector,
    normal_field_for_ucell,
    compression_indices,
    compress_cell,
)
from wavebit._core.solve_1d import solve_1d
from wavebit._core.solve_2d import solve_2d

# Enables jax x64, which every jax case here needs: at float32 the cross-backend comparisons
# miss their 1e-10 tolerance and the gradient checks miss 1e-8. Guarded because jax is
# optional -- when it is absent the jax cases skip themselves via `pytest.importorskip`.
#
# At the top of the module rather than beside each `import jax.numpy`, which is the idiom
# elsewhere in this suite, because the jax paths here are scattered across `_to` and three
# gradient tests: a jax case added later would silently inherit float32. That is how this file
# came to depend on *another* test module importing `wavebit.on_jax` first -- the whole suite
# passed while `pytest tests/test_factorization_core.py` alone failed nine of them.
try:
    import wavebit.on_jax                                    # noqa: E402,F401
except ImportError:
    pass


def _rand(shape, seed):
    return np.random.default_rng(seed).uniform(1.0, 4.0, size=shape)


# The legacy builders these were originally checked against are gone; their outputs live on
# as frozen reference data (tests/data/gen_goldens_factorization.py).
GOLDEN = np.load(pathlib.Path(__file__).parent / 'data' / 'golden_legacy_factorization.npz')


CASES = [
    # name, ucell, fto_x, fto_y
    ('1d_1layer', _rand((1, 1, 8), 0), 4, 0),
    ('2d_1layer', _rand((1, 5, 5), 1), 2, 2),
    ('2layer_rect', _rand((2, 4, 6), 2), 2, 1),
    ('3layer_1d', _rand((3, 1, 10), 3), 5, 0),
]


def _exact_coeff(eps, ky, kx):
    """Fourier coefficient of a pixel raster, by integrating each pixel analytically.

    The definition, spelled out with no shortcut: ``eps`` is constant on
    ``[j/Ny, (j+1)/Ny) x [i/Nx, (i+1)/Nx)``, so the transform is a finite sum of elementary
    integrals. Deliberately not the closed form the builder uses — a reference that shared its
    derivation would only confirm the arithmetic.
    """
    Ny, Nx = eps.shape

    def axis(k, N, n):
        a, b = n / N, (n + 1) / N
        if k == 0:
            return b - a
        return (np.exp(-2j * np.pi * k * b) - np.exp(-2j * np.pi * k * a)) / (-2j * np.pi * k)

    return sum(eps[j, i] * axis(ky, Ny, j) * axis(kx, Nx, i)
               for j in range(Ny) for i in range(Nx))


@pytest.mark.parametrize('case', CASES, ids=[c[0] for c in CASES])
def test_dfs_coefficients_are_the_exact_ones(case):
    """``<eps>`` must be the raster's own Fourier coefficients, to round-off.

    This replaces a bit-parity check against the legacy builder, which is gone: that builder
    resampled the raster onto a grid resolving ``4*fto+1`` harmonics and transformed *that*,
    approaching these values as ``1/R`` and stopping ~1e-2 short on the coefficients (~1e-5
    on efficiency, measured against RETICOLO). The exact route made the frozen values wrong to
    hold, and ``gen_goldens_factorization.py`` cannot be re-run — it imports modules deleted in
    ``306c9ea``.

    So the oracle moved rather than being regenerated. It is stronger for it: this compares
    against the definition instead of against another implementation, and needs no golden.
    """
    name, ucell, fto_x, fto_y = case
    eps = np.asarray(ucell, dtype=float)[0] ** 2

    got = np.asarray(fourier2d_piecewise_constant(xnp, eps.astype(complex), fto_x, fto_y, inverse_rule=None))
    ff_x, ff_y = 2 * fto_x + 1, 2 * fto_y + 1
    want = np.empty_like(got)
    for r in range(ff_y * ff_x):
        for c in range(ff_y * ff_x):
            ky = r // ff_x - c // ff_x
            kx = r % ff_x - c % ff_x
            want[r, c] = _exact_coeff(eps, ky, kx)

    assert np.abs(got - want).max() < 1e-12 * max(1.0, np.abs(want).max()), (
        name, np.abs(got - want).max())


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

    solver = wavebit.rcwa(backend=0, n_top=n_top, n_bot=n_bot, theta=theta, phi=None, tetm='te',
                          fto=fto, period=period, wavelength=wl, thickness=thickness, ucell=ucell)
    legacy = solver.conv_solve()
    kx, _ = solver.get_kx_ky_vector(wl)

    epx, epy, epz = core_dfs(ucell, fto[0], fto[1])
    ri, ti = solve_1d(0, wl, kx, n_top, n_bot, solver.theta, solver.thickness, epx, epy, epz)

    assert np.array_equal(np.array(legacy.R), np.array(ri))
    assert np.array_equal(np.array(legacy.T), np.array(ti))


# ------------------------------------------------- end-to-end differentiability (the point)

def _chain_1d(fac_kwargs, ucell, kx, theta):
    epx, epy, epz = core_dfs(ucell, **fac_kwargs)
    _, ti = solve_1d(0, 0.777, kx, 2.0, 1.0, theta, [0.15], epx, epy, epz)
    return ti


def test_end_to_end_gradient_wrt_ucell_1d():
    """d(sum T)/d(ucell raster): jax == torch == finite-diff, through factorization."""
    jax = pytest.importorskip('jax')
    torch = pytest.importorskip('torch')
    import jax.numpy as jnp

    ucell = np.array([[[3., 3., 3., 1., 1., 1.]]])
    direction = np.ones_like(ucell)
    fto = [6, 0]
    solver = wavebit.rcwa(backend=0, n_top=2.0, n_bot=1.0, theta=0.3, phi=None, tetm='te',
                          fto=fto, period=[0.77], wavelength=0.777, thickness=[0.15], ucell=ucell)
    kx, _ = solver.get_kx_ky_vector(0.777)
    theta = complex(solver.theta)
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
    solver = wavebit.rcwa(backend=0, n_top=n_top, n_bot=n_bot, theta=0.2, phi=0.4, psi=0.9,
                          fto=fto, period=[1.0, 1.0], wavelength=wl, thickness=[0.15], ucell=ucell)
    kx, ky = solver.get_kx_ky_vector(wl)
    theta, psi = complex(solver.theta), complex(solver.psi)

    def chain(u, kx_, ky_, th_, ps_):
        epx, epy, epz = core_dfs(u, fto[0], fto[1])
        _, ti = solve_2d(wl, kx_, ky_, n_top, n_bot, th_, ps_, [0.15], epx, epy, epz)
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


# ============================================================== NVF (factorization='nvf')

NVF_CASES = [
    ('nvf_5x5', _rand((1, 5, 5), 10), 2, 2),
    ('nvf_2layer', _rand((2, 4, 6), 11), 2, 2),
    ('nvf_binary', np.where(_rand((1, 8, 8), 12) > 2.5, 3.5, 1.0), 2, 2),
]


@pytest.mark.parametrize('case', NVF_CASES, ids=[c[0] for c in NVF_CASES])
def test_nvf_permittivity_inputs_are_the_exact_ones(case):
    """NVF's ``<eps>`` and ``<1/eps>`` are exact too; its projectors are what still are not.

    The same retirement as the sampled check above, and for the same reason — but narrower,
    because only part of the NVF build changed. ``<eps>`` and ``<1/eps>`` read a raster, so
    they get the exact coefficients. ``C_xx``/``C_xy``/``C_yy`` read the normal field, which
    is a continuous vector field sampled on a grid: no closed form replaces it, refining the
    grid is the only thing that helps, and nothing about it moved.
    """
    name, ucell, fto_x, fto_y = case
    eps = np.asarray(ucell, dtype=float)[0] ** 2

    for label, cell in (('eps', eps), ('1/eps', 1.0 / eps)):
        got = np.asarray(fourier2d_piecewise_constant(xnp, cell.astype(complex), fto_x, fto_y, inverse_rule=None))
        ff_x, ff_y = 2 * fto_x + 1, 2 * fto_y + 1
        want = np.empty_like(got)
        for r in range(ff_y * ff_x):
            for c in range(ff_y * ff_x):
                want[r, c] = _exact_coeff(cell, r // ff_x - c // ff_x, r % ff_x - c % ff_x)
        assert np.abs(got - want).max() < 1e-12 * max(1.0, np.abs(want).max()), (name, label)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('case', NVF_CASES, ids=[c[0] for c in NVF_CASES])
def test_nvf_runs_on_every_backend(case, backend):
    """NVF on jax / torch — a capability the legacy per-backend builders never had.

    ``rcwa(backend=1|2, factorization='nvf')`` raises ValueError on the legacy path; the
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

    ``rcwa(backend=1|2, factorization='nvf')`` raised ValueError before the factorization was
    wired — the per-backend builders simply had no NVF branch.
    """
    ucell = _rand((1, 5, 5), 13)
    opts = dict(n_top=1.0, n_bot=1.0, theta=0.0, phi=0.0, psi=0.0, fto=[2, 2],
                period=[1.0, 1.0], wavelength=0.9, thickness=[0.2], ucell=ucell,
                factorization='nvf')
    ref = wavebit.rcwa(backend=0, **opts).conv_solve()
    for backend in (1, 2):
        pytest.importorskip('jax' if backend == 1 else 'torch')
        got = wavebit.rcwa(backend=backend, **opts).conv_solve()
        assert np.allclose(np.asarray(ref.res.T), np.asarray(got.res.T),
                           atol=1e-12, rtol=0), backend


def test_nvf_end_to_end_gradient_wrt_ucell():
    """d(sum T)/d(ucell) through the NVF factorization: jax == torch == finite-diff.

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

    solver = wavebit.rcwa(backend=0, n_top=n_top, n_bot=n_bot, theta=0.2, phi=0.4, psi=0.9,
                          fto=fto, period=[1.0, 1.0], wavelength=wl, thickness=[d], ucell=ucell)
    kx, ky = solver.get_kx_ky_vector(wl)
    theta, psi = complex(solver.theta), complex(solver.psi)

    def chain(u, kx_, ky_, th_, ps_):
        epx, epy, epxy, epyx, epz = core_nvf(u, fto[0], fto[1], normal_field=normal_field)
        _, ti = solve_2d(wl, kx_, ky_, n_top, n_bot, th_, ps_, [d], epx, epy, epz,
                         epxy_conv_all=epxy, epyx_conv_all=epyx)
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


# ============================ integrated + vector layers (fourier_coefficients='integrated')

INTEGRATED_CASES = [
    ('cfs_1d', _rand((1, 1, 8), 20), 4, 0),
    ('cfs_2d', _rand((1, 5, 5), 21), 2, 2),
    ('cfs_binary', np.where(_rand((1, 6, 6), 22) > 2.5, 3.5, 1.0), 2, 2),
    ('cfs_2layer', _rand((2, 4, 6), 23), 2, 1),
]


@pytest.mark.parametrize('case', INTEGRATED_CASES, ids=[c[0] for c in INTEGRATED_CASES])
def test_cfs_bit_parity_vs_legacy(case):
    name, ucell, fto_x, fto_y = case
    got = core_cfs(ucell, fto_x, fto_y)
    for slot, b in zip(('epx', 'epy', 'epz_i'), got):
        a = GOLDEN[f'cfs__{name}__{slot}']
        assert np.allclose(a, np.array(b), atol=1e-12, rtol=0), (
            slot, np.max(np.abs(a - np.array(b))))


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('case', INTEGRATED_CASES, ids=[c[0] for c in INTEGRATED_CASES])
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
    """Vector-modelled layers (explicit segment edges, no compression) match the legacy integrated build."""
    info = [[GOLDEN[f'vector__in{i}__cell'], GOLDEN[f'vector__in{i}__x'],
             GOLDEN[f'vector__in{i}__y']] for i in (0, 1)]
    got = core_vector(info, 2, 2)
    for slot, b in zip(('epx', 'epy', 'epz_i'), got):
        a = GOLDEN[f'vector__{slot}']
        assert np.allclose(a, np.array(b), atol=1e-12, rtol=0), (
            slot, np.max(np.abs(a - np.array(b))))


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
    """d(sum T)/d(ucell) through the analytic integrated path: jax == torch == finite-diff."""
    jax = pytest.importorskip('jax')
    torch = pytest.importorskip('torch')
    import jax.numpy as jnp

    ucell = np.array([[[3., 3., 3., 1., 1., 1.]]])
    direction = np.ones_like(ucell)
    fto = [6, 0]
    wl, n_top, n_bot, d = 0.777, 2.0, 1.0, 0.15
    compression = [compression_indices(ucell[i]) for i in range(ucell.shape[0])]

    solver = wavebit.rcwa(backend=0, n_top=n_top, n_bot=n_bot, theta=0.3, phi=None, psi=np.pi / 2,
                          fto=fto, period=[0.77], wavelength=wl, thickness=[d], ucell=ucell)
    kx, _ = solver.get_kx_ky_vector(wl)
    theta = complex(solver.theta)

    def chain(u, kx_):
        epx, epy, epz = core_cfs(u, fto[0], fto[1], compression=compression)
        _, ti = solve_1d(0, wl, kx_, n_top, n_bot, theta, [d], epx, epy, epz)
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


def _transpose_symmetric_L(N=48):
    """An L-shape with ``eps(x, y) == eps(y, x)``, deliberately not 4-fold symmetric."""
    u = np.ones((N, N))
    u[N // 8:N // 2, N // 8:N // 4] = 3.2
    u = np.maximum(u, u.T)
    assert np.array_equal(u, u.T)
    return u[None, :, :]


def _transmission_map(ucell, factorization, fto, psi):
    res = wavebit.rcwa(
        backend=0, factorization=factorization, fto=[fto, fto],
        n_top=1.0, n_bot=1.0, theta=0.0, phi=0.0, psi=psi,
        wavelength=1.3, period=[1.0, 1.0], thickness=[0.4], ucell=ucell).conv_solve()
    t = np.asarray(res.T)
    ff = int(round(t.size ** 0.5))
    return t.reshape(ff, ff)


def _relabelling_residue(factorization, fto):
    """``max|T(psi=0) - T(psi=90)^T|`` on a transpose-symmetric structure at normal incidence.

    Swapping x and y maps that problem onto itself with the polarisation exchanged, so the
    diffraction map has to come back transposed. Any residue is the builder telling x and y
    apart.
    """
    ucell = _transpose_symmetric_L()
    a = _transmission_map(ucell, factorization, fto, 0.0)
    b = _transmission_map(ucell, factorization, fto, np.pi / 2)
    return float(np.abs(a - b.T).max())


@pytest.mark.parametrize('fto', [4, 6])
def test_nvf_does_not_tell_x_and_y_apart(fto):
    """NVF chooses no nesting order, so the relabelling symmetry is exact.

    This is the invariant; the ``li`` test below is the record of what it costs not to have
    it. Both were unpinned until 2026-08-12 -- see ``docs/factorization-terms.md`` §6.
    """
    resid = _relabelling_residue('nvf', fto)
    assert resid < 1e-11, f'nvf broke the x<->y relabelling symmetry by {resid:.2e}'


@pytest.mark.parametrize('fto', [4, 6])
def test_li_tells_x_and_y_apart_and_this_is_expected(fto):
    """Li's nested rule fixes one axis inside the other, so it does *not* hold the symmetry.

    Intrinsic to the crossed-grating scheme rather than a defect here -- Liu & Fan name it in
    the S4 paper as the reason vector-field formulations exist, which is what the NVF path is.

    Bounded on both sides on purpose. Above, because a regression that worsened the nesting
    would otherwise pass unnoticed; below, because one that *fixed* it should also stop and
    make somebody update ``docs/factorization-terms.md`` §6 rather than quietly invalidating
    it. Energy conservation sees none of this: the whole residue appears as a difference in
    total transmitted power between two polarisations that ought to agree.
    """
    resid = _relabelling_residue('li', fto)
    assert 1e-4 < resid < 5e-2, (
        f'li relabelling residue {resid:.2e} left the band this test records; if the nesting '
        f'genuinely changed, docs/factorization-terms.md §6 needs the new numbers')
