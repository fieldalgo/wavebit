"""Single-source Fourier factorization (wavebit._core.factorization) — Phase 4a/4b gate.

Beyond the usual parity/cross-backend checks, this closes the differentiability gap that
motivated the port: with the factorization inside the core, the gradient runs end-to-end from
the *design variable* — the permittivity raster — through to diffraction efficiency, rather than
stopping at the convolution matrices.
"""
import numpy as np
import pytest

import array_api_compat.numpy as xnp

import wavebit
from wavebit._core.factorization import fourier2d_piecewise_constant, fourier2d_integrated
from wavebit._core.factorization import (
    to_conv_mat_raster_sampled as core_dfs,
    to_conv_mat_raster_nvf as core_nvf,
    to_conv_mat_raster_integrated as core_cfs,
    to_conv_mat_block as core_vector,
    normal_field_for_raster,
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
    """A random **permittivity** raster -- what ``structure`` holds since the material flip.

    Drawn uniform over an index range and squared, rather than drawn uniform in eps: every
    case in this file used to hand the solver an index in [1, 4], which it squared, so this
    is the same raster the pre-flip file measured. Drawing eps uniform on [1, 4] instead
    would have been tidier and would have quietly changed the contrast every tolerance here
    was set against.
    """
    return np.random.default_rng(seed).uniform(1.0, 4.0, size=shape) ** 2


CASES = [
    # name, structure, half_width_u, half_width_v
    ('1d_1layer', _rand((1, 1, 8), 0), 4, 0),
    ('2d_1layer', _rand((1, 5, 5), 1), 2, 2),
    ('2layer_rect', _rand((2, 4, 6), 2), 2, 1),
    ('3layer_1d', _rand((3, 1, 10), 3), 5, 0),
]


def _axis_int(k, a, b):
    """``integral over [a, b] of exp(-2 pi i k t) dt`` — the one elementary integral here."""
    if k == 0:
        return b - a
    return (np.exp(-2j * np.pi * k * b) - np.exp(-2j * np.pi * k * a)) / (-2j * np.pi * k)


def _exact_coeff_segments(eps, x, y, ky, kx):
    """Fourier coefficient of a piecewise-constant cell on **arbitrary** segment edges.

    ``x`` / ``y`` are right-hand edges normalized to end at 1, exactly as
    :func:`~wavebit._core.factorization.compression_indices` returns them, so ``eps[j, i]``
    covers ``[y[j-1], y[j]) x [x[i-1], x[i])`` with ``0`` implied before the first.

    The definition spelled out with no shortcut, and deliberately not the closed form any
    builder uses — a reference that shared its derivation would only confirm the arithmetic.
    One oracle serves the sampled path (uniform edges, :func:`_exact_coeff`), the integrated
    path and the vector path, which is the point: they claim to transform the *same* staircase
    and differ only in where its steps may sit.
    """
    xr = np.asarray(x, dtype=float).ravel()
    yr = np.asarray(y, dtype=float).ravel()
    xl = np.concatenate([[0.0], xr[:-1]])
    yl = np.concatenate([[0.0], yr[:-1]])

    return sum(eps[j, i] * _axis_int(ky, yl[j], yr[j]) * _axis_int(kx, xl[i], xr[i])
               for j in range(eps.shape[0]) for i in range(eps.shape[1]))


def _exact_coeff(eps, ky, kx):
    """:func:`_exact_coeff_segments` on the uniform pixel grid — ``eps`` as a raster."""
    Ny, Nx = eps.shape
    return _exact_coeff_segments(eps, np.arange(1, Nx + 1) / Nx, np.arange(1, Ny + 1) / Ny,
                                 ky, kx)


def _exact_matrix(eps, x, y, half_width_u, half_width_v):
    """The whole direct-rule convolution matrix from the definition, index by index."""
    n_orders_u, n_orders_v = 2 * half_width_u + 1, 2 * half_width_v + 1
    n = n_orders_v * n_orders_u
    want = np.empty((n, n), dtype=complex)
    for r in range(n):
        for c in range(n):
            want[r, c] = _exact_coeff_segments(
                eps, x, y, r // n_orders_u - c // n_orders_u, r % n_orders_u - c % n_orders_u)
    return want


@pytest.mark.parametrize('case', CASES, ids=[c[0] for c in CASES])
def test_dfs_coefficients_are_the_exact_ones(case):
    """``<eps>`` must be the raster's own Fourier coefficients, to round-off.

    This replaced a bit-parity check against the legacy builder, which is gone: that builder
    resampled the raster onto a grid resolving ``4*truncation_order+1`` harmonics and transformed *that*,
    approaching these values as ``1/R`` and stopping ~1e-2 short on the coefficients (~1e-5
    on efficiency, measured against RETICOLO). The exact route made the frozen values wrong to
    hold, and the generator could not be re-run — it imported modules deleted in ``306c9ea``.

    So the oracle moved rather than being regenerated. It is stronger for it: this compares
    against the definition instead of against another implementation, and needs no golden. The
    same move was made for the integrated, vector and compression checks on 2026-08-18, which
    is what retired the golden entirely — see the note further down this file.
    """
    name, structure, half_width_u, half_width_v = case
    eps = np.asarray(structure, dtype=float)[0]      # the raster *is* the permittivity now

    got = np.asarray(fourier2d_piecewise_constant(xnp, eps.astype(complex), half_width_u, half_width_v, inverse_rule=None))
    n_orders_u, n_orders_v = 2 * half_width_u + 1, 2 * half_width_v + 1
    want = np.empty_like(got)
    for r in range(n_orders_v * n_orders_u):
        for c in range(n_orders_v * n_orders_u):
            ky = r // n_orders_u - c // n_orders_u
            kx = r % n_orders_u - c % n_orders_u
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
    _, structure, half_width_u, half_width_v = case
    ref = core_dfs(structure, half_width_u, half_width_v)
    got = core_dfs(_to(backend, structure), half_width_u, half_width_v)
    for name, a, b in zip(('epx', 'epy', 'epz_i'), ref, got):
        assert np.allclose(np.array(a), np.asarray(b), atol=1e-10, rtol=0), name


def test_factorization_feeds_solve_bit_exactly():
    """core factorization -> core solve reproduces the legacy end-to-end result exactly."""
    structure = np.array([[[9., 9., 9., 1., 1., 1.]]])
    wl, n_top, n_bot, theta, period, truncation_order = 0.777, 2.0, 1.0, 0.3, [0.77], [6, 0]
    thickness = [0.15]

    # Two spellings of one pair of half-spaces, on purpose. `solve_1d` is the core, and there
    # the quantity genuinely is an amplitude ratio -- an index. The constructor is the public
    # boundary and takes a permittivity, converting once in the `top` / `bot` setters.
    eps_top, eps_bot = n_top ** 2, n_bot ** 2

    solver = wavebit.rcwa(backend=0, top=eps_top, bot=eps_bot, theta=theta, phi=None,
                          tetm='te',
                          truncation_order=truncation_order, period=period, wavelength=wl, thickness=thickness, structure=structure)
    legacy = solver.solve()
    kx, _ = solver.order_kxy()

    epx, epy, epz, *_ = core_dfs(structure, truncation_order[0], truncation_order[1])
    ri, ti = solve_1d(0, wl, kx, n_top, n_bot, solver.theta, solver.thickness, epx, epy, epz)

    assert np.array_equal(np.array(legacy.R), np.array(ri))
    assert np.array_equal(np.array(legacy.T), np.array(ti))


# ------------------------------------------------- end-to-end differentiability (the point)

def _chain_1d(fac_kwargs, structure, kx, theta):
    epx, epy, epz, *_ = core_dfs(structure, **fac_kwargs)
    _, ti = solve_1d(0, 0.777, kx, 2.0, 1.0, theta, [0.15], epx, epy, epz)
    return ti


def test_end_to_end_gradient_wrt_structure_1d():
    """d(sum T)/d(structure raster): jax == torch == finite-diff, through factorization."""
    jax = pytest.importorskip('jax')
    torch = pytest.importorskip('torch')
    import jax.numpy as jnp

    structure = np.array([[[9., 9., 9., 1., 1., 1.]]])
    direction = np.ones_like(structure)
    truncation_order = [6, 0]
    # top: eps 4.0 is the index 2.0 the `_chain_1d` core call below still takes.
    solver = wavebit.rcwa(backend=0, top=4.0, bot=1.0, theta=0.3, phi=None, tetm='te',
                          truncation_order=truncation_order, period=[0.77], wavelength=0.777, thickness=[0.15], structure=structure)
    kx, _ = solver.order_kxy()
    theta = complex(solver.theta)
    fk = dict(half_width_u=truncation_order[0], half_width_v=truncation_order[1])

    def f_np(s):
        return float(np.sum(np.array(_chain_1d(fk, structure + s * direction, kx, theta))))
    h = 1e-6
    g_fd = (f_np(h) - f_np(-h)) / (2 * h)

    g_jax = float(jax.grad(lambda s: jnp.sum(_chain_1d(
        fk, jnp.asarray(structure) + s * jnp.asarray(direction),
        jnp.asarray(kx), theta)).real)(0.0))

    s = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    torch.sum(_chain_1d(fk, torch.as_tensor(structure) + s * torch.as_tensor(direction),
                        torch.as_tensor(kx), theta)).real.backward()
    g_torch = float(s.grad)

    assert abs(g_jax - g_torch) < 1e-8, (g_jax, g_torch)
    assert abs(g_jax - g_fd) < 1e-6, (g_jax, g_fd)


def test_end_to_end_gradient_wrt_structure_2d():
    """Same end-to-end gradient through the 2D coupled block chain."""
    jax = pytest.importorskip('jax')
    torch = pytest.importorskip('torch')
    import jax.numpy as jnp

    structure = _rand((1, 4, 4), 7)
    direction = np.ones_like(structure)
    truncation_order = [1, 1]
    wl, n_top, n_bot = 0.9, 1.0, 1.0
    solver = wavebit.rcwa(backend=0, top=n_top, bot=n_bot, theta=0.2, phi=0.4, psi=0.9,
                          truncation_order=truncation_order, period=[1.0, 1.0], wavelength=wl, thickness=[0.15], structure=structure)
    kx, ky = solver.order_kxy()
    theta, inc = complex(solver.theta), solver._incidence_pair()

    def chain(u, kx_, ky_, th_, inc_):
        epx, epy, epz, *_ = core_dfs(u, truncation_order[0], truncation_order[1])
        _, ti = solve_2d(wl, kx_, ky_, n_top, n_bot, th_, inc_, [0.15], epx, epy, epz,
                         phi=solver.phi)
        return ti

    def f_np(s):
        return float(np.sum(np.array(chain(structure + s * direction, kx, ky, theta, inc))))
    h = 1e-6
    g_fd = (f_np(h) - f_np(-h)) / (2 * h)

    g_jax = float(jax.grad(lambda s: jnp.sum(chain(
        jnp.asarray(structure) + s * jnp.asarray(direction), jnp.asarray(kx), jnp.asarray(ky), theta, inc)).real)(0.0))

    s = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    torch.sum(chain(torch.as_tensor(structure) + s * torch.as_tensor(direction),
                    torch.as_tensor(kx), torch.as_tensor(ky), theta, inc)).real.backward()
    g_torch = float(s.grad)

    assert abs(g_jax - g_torch) < 1e-8, (g_jax, g_torch)
    assert abs(g_jax - g_fd) < 1e-6, (g_jax, g_fd)


# ============================================================== NVF (factorization='nvf')

NVF_CASES = [
    ('nvf_5x5', _rand((1, 5, 5), 10), 2, 2),
    ('nvf_2layer', _rand((2, 4, 6), 11), 2, 2),
    ('nvf_binary', np.where(_rand((1, 8, 8), 12) > 6.25, 12.25, 1.0), 2, 2),
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
    name, structure, half_width_u, half_width_v = case
    eps = np.asarray(structure, dtype=float)[0]

    for label, cell in (('eps', eps), ('1/eps', 1.0 / eps)):
        got = np.asarray(fourier2d_piecewise_constant(xnp, cell.astype(complex), half_width_u, half_width_v, inverse_rule=None))
        n_orders_u, n_orders_v = 2 * half_width_u + 1, 2 * half_width_v + 1
        want = np.empty_like(got)
        for r in range(n_orders_v * n_orders_u):
            for c in range(n_orders_v * n_orders_u):
                want[r, c] = _exact_coeff(cell, r // n_orders_u - c // n_orders_u, r % n_orders_u - c % n_orders_u)
        assert np.abs(got - want).max() < 1e-12 * max(1.0, np.abs(want).max()), (name, label)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('case', NVF_CASES, ids=[c[0] for c in NVF_CASES])
def test_nvf_runs_on_every_backend(case, backend):
    """NVF on jax / torch — a capability the legacy per-backend builders never had.

    ``rcwa(backend=1|2, factorization='nvf')`` raises ValueError on the legacy path; the
    single-source builder makes the same factorization available on all three.
    """
    pytest.importorskip(backend)
    _, structure, half_width_u, half_width_v = case
    ref = core_nvf(structure, half_width_u, half_width_v)
    got = core_nvf(_to(backend, structure), half_width_u, half_width_v)
    for a, b in zip(ref, got):
        assert np.allclose(np.array(a), np.asarray(b), atol=1e-12, rtol=0)


def test_nvf_now_works_on_every_backend():
    """NVF used to be NumPy-only; solve now routes every backend through the core.

    ``rcwa(backend=1|2, factorization='nvf')`` raised ValueError before the factorization was
    wired — the per-backend builders simply had no NVF branch.
    """
    structure = _rand((1, 5, 5), 13)
    opts = dict(top=1.0, bot=1.0, theta=0.0, phi=0.0, psi=0.0, truncation_order=[2, 2],
                period=[1.0, 1.0], wavelength=0.9, thickness=[0.2], structure=structure,
                factorization='nvf')
    ref = wavebit.rcwa(backend=0, **opts).solve()
    for backend in (1, 2):
        pytest.importorskip('jax' if backend == 1 else 'torch')
        got = wavebit.rcwa(backend=backend, **opts).solve()
        assert np.allclose(np.asarray(ref.given.T), np.asarray(got.given.T),
                           atol=1e-12, rtol=0), backend


def test_nvf_end_to_end_gradient_wrt_structure():
    """d(sum T)/d(structure) through the NVF factorization: jax == torch == finite-diff.

    The normal vector field is static geometry (precomputed once), so gradients measure the
    permittivity dependence with the interface directions held fixed.
    """
    jax = pytest.importorskip('jax')
    torch = pytest.importorskip('torch')
    import jax.numpy as jnp

    structure = np.where(_rand((1, 6, 6), 14) > 6.25, 12.25, 1.0)
    direction = np.ones_like(structure)
    truncation_order = [1, 1]
    wl, n_top, n_bot, d = 0.9, 1.0, 1.0, 0.2
    normal_field = normal_field_for_raster(structure, truncation_order[0], truncation_order[1])

    solver = wavebit.rcwa(backend=0, top=n_top, bot=n_bot, theta=0.2, phi=0.4, psi=0.9,
                          truncation_order=truncation_order, period=[1.0, 1.0], wavelength=wl, thickness=[d], structure=structure)
    kx, ky = solver.order_kxy()
    theta, inc = complex(solver.theta), solver._incidence_pair()

    def chain(u, kx_, ky_, th_, inc_):
        epx, epy, epxy, epyx, epz = core_nvf(u, truncation_order[0], truncation_order[1], normal_field=normal_field)
        _, ti = solve_2d(wl, kx_, ky_, n_top, n_bot, th_, inc_, [d], epx, epy, epz,
                         epxy_conv_all=epxy, epyx_conv_all=epyx, phi=solver.phi)
        return ti

    def f_np(s):
        return float(np.sum(np.array(chain(structure + s * direction, kx, ky, theta, inc))))
    h = 1e-6
    g_fd = (f_np(h) - f_np(-h)) / (2 * h)

    g_jax = float(jax.grad(lambda s: jnp.sum(chain(
        jnp.asarray(structure) + s * jnp.asarray(direction), jnp.asarray(kx), jnp.asarray(ky), theta, inc)).real)(0.0))

    s = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    torch.sum(chain(torch.as_tensor(structure) + s * torch.as_tensor(direction),
                    torch.as_tensor(kx), torch.as_tensor(ky), theta, inc)).real.backward()
    g_torch = float(s.grad)

    assert abs(g_jax - g_torch) < 1e-8, (g_jax, g_torch)
    assert abs(g_jax - g_fd) < 1e-6, (g_jax, g_fd)


# ============================ integrated + vector layers (fourier_coefficients='integrated')

INTEGRATED_CASES = [
    ('cfs_1d', _rand((1, 1, 8), 20), 4, 0),
    ('cfs_2d', _rand((1, 5, 5), 21), 2, 2),
    ('cfs_binary', np.where(_rand((1, 6, 6), 22) > 6.25, 12.25, 1.0), 2, 2),
    ('cfs_2layer', _rand((2, 4, 6), 23), 2, 1),
]


@pytest.mark.parametrize('case', INTEGRATED_CASES, ids=[c[0] for c in INTEGRATED_CASES])
def test_integrated_coefficients_are_the_exact_ones(case):
    """The integrated path's ``<eps>`` must be the staircase's own coefficients, to round-off.

    The integrated twin of :func:`test_dfs_coefficients_are_the_exact_ones`, and it replaced a
    bit-parity check against the legacy builder for the same reason: parity says "unchanged",
    the definition says "right". Asserted at both ends of the path — the coefficient engine
    directly, and the builder's ``epz_i``, which is that matrix inverted and so pins the slot
    the solve actually eliminates ``E_z`` with.
    """
    _, structure, half_width_u, half_width_v = case
    layer = np.asarray(structure)[0]
    row_idx, col_idx, x, y = compression_indices(layer)
    eps = np.asarray(compress_cell(xnp, layer, row_idx, col_idx)).astype(complex)

    want = _exact_matrix(eps, x, y, half_width_u, half_width_v)
    scale = max(1.0, np.abs(want).max())

    got = np.asarray(fourier2d_integrated(xnp, eps, x, y, half_width_u, half_width_v, inverse_rule=None))
    assert np.abs(got - want).max() < 1e-12 * scale, np.abs(got - want).max()

    inv_z = np.linalg.inv(np.asarray(core_cfs(structure, half_width_u, half_width_v).epz_i[0]))
    assert np.abs(inv_z - want).max() < 1e-11 * scale, np.abs(inv_z - want).max()


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('case', INTEGRATED_CASES, ids=[c[0] for c in INTEGRATED_CASES])
def test_cfs_cross_backend(case, backend):
    pytest.importorskip(backend)
    _, structure, half_width_u, half_width_v = case
    ref = core_cfs(structure, half_width_u, half_width_v)
    # Compression is static geometry, so it is shared rather than recomputed per backend.
    compression = [compression_indices(structure[i]) for i in range(structure.shape[0])]
    got = core_cfs(_to(backend, structure), half_width_u, half_width_v, compression=compression)
    for name, a, b in zip(('epx', 'epy', 'epz_i'), ref, got):
        assert np.allclose(np.array(a), np.asarray(b), atol=1e-12, rtol=0), name


def test_the_vector_path_integrates_its_segments_exactly():
    """Segment edges given directly, on a grid no raster could express.

    The vector path's whole reason to exist is that its steps need not sit at ``j/N``, so the
    case is built with deliberately uneven edges — a raster would have to snap them. Nothing is
    frozen: the cell and its edges are written here, and the answer comes from the definition.
    """
    # Permittivities, not indices: the block path's cell holds what the transform integrates.
    cell = np.array([[1.0, 12.25, 4.0],
                     [6.25, 1.0, 9.0]])
    # column vectors, the shape `compression_indices` hands the builder
    x = np.array([[0.17], [0.61], [1.0]])          # uneven on purpose
    y = np.array([[0.43], [1.0]])
    half_width_u = half_width_v = 2

    got = core_vector([(cell, x, y)], half_width_u, half_width_v)
    want = _exact_matrix(cell.astype(complex), x, y, half_width_u, half_width_v)
    scale = max(1.0, np.abs(want).max())

    inv_z = np.linalg.inv(np.asarray(got.epz_i[0]))
    assert np.abs(inv_z - want).max() < 1e-11 * scale, np.abs(inv_z - want).max()


@pytest.mark.parametrize('seed', [26, 27, 28])
def test_compression_is_lossless_and_minimal(seed):
    """Compression must lose nothing and merge everything it can — both, or it is not it.

    This replaced a check that the indices matched the legacy implementation's byte for byte,
    which could only say "unchanged". These two properties are what the step is *for*, and they
    are checkable without a reference: expanding the compressed cell back over its segment
    widths has to give the original raster (lossless), and no two neighbouring segments may be
    equal, or a merge was missed (minimal).

    The raster is built with duplicated rows and columns so there is something to merge; a
    fully random one compresses to itself and would assert nothing.
    """
    base = _rand((3, 4), seed)
    layer = np.repeat(np.repeat(base, 3, axis=0), 2, axis=1)      # 9 x 8, blocks of 3 x 2

    row_idx, col_idx, x, y = compression_indices(layer)
    cell = np.asarray(compress_cell(xnp, layer, row_idx, col_idx))
    xr = np.asarray(x, dtype=float).ravel()
    yr = np.asarray(y, dtype=float).ravel()

    # lossless: widths are whole pixel counts, so expanding by them rebuilds the raster
    ny, nx = layer.shape
    rows = np.rint(np.diff(np.concatenate([[0.0], yr])) * ny).astype(int)
    cols = np.rint(np.diff(np.concatenate([[0.0], xr])) * nx).astype(int)
    assert rows.sum() == ny and cols.sum() == nx, (rows, cols)
    rebuilt = np.repeat(np.repeat(cell, rows, axis=0), cols, axis=1)
    assert np.array_equal(rebuilt, layer), 'compression lost something'

    # minimal: neighbouring segments differ, or a merge was missed
    assert not any(np.array_equal(cell[j], cell[j + 1]) for j in range(cell.shape[0] - 1)), (
        'two neighbouring rows are equal -- they should have merged')
    assert not any(np.array_equal(cell[:, i], cell[:, i + 1]) for i in range(cell.shape[1] - 1)), (
        'two neighbouring columns are equal -- they should have merged')

    # and it actually compressed this one, or the assertions above are vacuous
    assert cell.shape == base.shape, (cell.shape, base.shape)


def test_cfs_end_to_end_gradient_wrt_structure():
    """d(sum T)/d(structure) through the analytic integrated path: jax == torch == finite-diff."""
    jax = pytest.importorskip('jax')
    torch = pytest.importorskip('torch')
    import jax.numpy as jnp

    structure = np.array([[[9., 9., 9., 1., 1., 1.]]])
    direction = np.ones_like(structure)
    truncation_order = [6, 0]
    wl, n_top, n_bot, d = 0.777, 2.0, 1.0, 0.15      # indices: `solve_1d` takes them below
    compression = [compression_indices(structure[i]) for i in range(structure.shape[0])]

    solver = wavebit.rcwa(backend=0, top=n_top ** 2, bot=n_bot ** 2, theta=0.3, phi=None,
                          psi=np.pi / 2,
                          truncation_order=truncation_order, period=[0.77], wavelength=wl, thickness=[d], structure=structure)
    kx, _ = solver.order_kxy()
    theta = complex(solver.theta)

    def chain(u, kx_):
        epx, epy, epz, *_ = core_cfs(u, truncation_order[0], truncation_order[1], compression=compression)
        _, ti = solve_1d(0, wl, kx_, n_top, n_bot, theta, [d], epx, epy, epz)
        return ti

    def f_np(s):
        return float(np.sum(np.array(chain(structure + s * direction, kx))))
    h = 1e-6
    g_fd = (f_np(h) - f_np(-h)) / (2 * h)

    g_jax = float(jax.grad(lambda s: jnp.sum(chain(
        jnp.asarray(structure) + s * jnp.asarray(direction), jnp.asarray(kx))).real)(0.0))

    s = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    torch.sum(chain(torch.as_tensor(structure) + s * torch.as_tensor(direction),
                    torch.as_tensor(kx))).real.backward()
    g_torch = float(s.grad)

    assert abs(g_jax - g_torch) < 1e-8, (g_jax, g_torch)
    assert abs(g_jax - g_fd) < 1e-6, (g_jax, g_fd)


def _transpose_symmetric_L(N=48):
    """An L-shape with ``eps(x, y) == eps(y, x)``, deliberately not 4-fold symmetric."""
    u = np.ones((N, N))
    u[N // 8:N // 2, N // 8:N // 4] = 10.240000000000002       # 3.2**2, to the bit
    u = np.maximum(u, u.T)
    assert np.array_equal(u, u.T)
    return u[None, :, :]


def _transmission_map(structure, factorization, truncation_order, psi):
    res = wavebit.rcwa(
        backend=0, factorization=factorization, truncation_order=[truncation_order, truncation_order],
        top=1.0, bot=1.0, theta=0.0, phi=0.0, psi=psi,
        wavelength=1.3, period=[1.0, 1.0], thickness=[0.4], structure=structure).solve()
    t = np.asarray(res.T)
    n_orders = int(round(t.size ** 0.5))
    return t.reshape(n_orders, n_orders)


def _relabelling_residue(factorization, truncation_order):
    """``max|T(psi=0) - T(psi=90)^T|`` on a transpose-symmetric structure at normal incidence.

    Swapping x and y maps that problem onto itself with the polarisation exchanged, so the
    diffraction map has to come back transposed. Any residue is the builder telling x and y
    apart.
    """
    structure = _transpose_symmetric_L()
    a = _transmission_map(structure, factorization, truncation_order, 0.0)
    b = _transmission_map(structure, factorization, truncation_order, np.pi / 2)
    return float(np.abs(a - b.T).max())


@pytest.mark.parametrize('truncation_order', [4, 6])
def test_nvf_does_not_tell_x_and_y_apart(truncation_order):
    """NVF chooses no nesting order, so the relabelling symmetry is exact.

    This is the invariant; the ``li`` test below is the record of what it costs not to have
    it. Both were unpinned until 2026-08-12 -- see ``docs/factorization-terms.md`` §6.
    """
    resid = _relabelling_residue('nvf', truncation_order)
    assert resid < 1e-11, f'nvf broke the x<->y relabelling symmetry by {resid:.2e}'


@pytest.mark.parametrize('truncation_order', [4, 6])
def test_li_tells_x_and_y_apart_and_this_is_expected(truncation_order):
    """Li's nested rule fixes one axis inside the other, so it does *not* hold the symmetry.

    Intrinsic to the crossed-grating scheme rather than a defect here -- Liu & Fan name it in
    the S4 paper as the reason vector-field formulations exist, which is what the NVF path is.

    Bounded on both sides on purpose. Above, because a regression that worsened the nesting
    would otherwise pass unnoticed; below, because one that *fixed* it should also stop and
    make somebody update ``docs/factorization-terms.md`` §6 rather than quietly invalidating
    it. Energy conservation sees none of this: the whole residue appears as a difference in
    total transmitted power between two polarisations that ought to agree.
    """
    resid = _relabelling_residue('li', truncation_order)
    assert 1e-4 < resid < 5e-2, (
        f'li relabelling residue {resid:.2e} left the band this test records; if the nesting '
        f'genuinely changed, docs/factorization-terms.md §6 needs the new numbers')


# ------------------------------------------------------ the same symmetry, on a scene that can see it
#
# `_relabelling_residue` above is the shipped path, but it is degenerate three times over for a
# turn of the whole scene: normal incidence, a SQUARE period, and a transpose-symmetric cell. No
# raster is ever rotated and no period pair is ever swapped, so most of what a quarter turn has
# to carry is never exercised. Its two neighbours are degenerate the same way --
# `test_pn_factorization.py::test_it_is_covariant_under_a_quarter_turn_about_z` does a real
# `np.rot90` but on the internal PN path at normal incidence with a square period, and
# `test_solve_2d_4ff.py::test_rotating_the_whole_problem_about_z_changes_nothing` turns uniform
# tensors with no raster and no period pair at all.
#
# What follows turns the scene where all four moving parts are live: oblique incidence, unequal
# periods, unequal truncation_order, and a chiral cell.

def _chiral_L(N=80, eps_hi=5.76):                   # 2.4**2
    """An L with unequal arms -- no mirror axis, and not equal to its own transpose.

    The chirality is the whole point. On a shape symmetric under a reflection the rotation and
    reflection spellings of a quarter turn collapse onto each other and the ``psi`` sign stops
    mattering, which is exactly how the three tests named above ended up unable to see any of
    this. The asserts are the premise of every test below, checked rather than assumed, so that
    an edit to the arms cannot quietly make them vacuous.
    """
    u = np.ones((N, N))
    u[N // 8:N // 2, N // 8:5 * N // 16] = eps_hi      # rows are y: a tall narrow arm
    u[N // 8:N // 5, N // 8:13 * N // 16] = eps_hi     # a long flat arm, different length *and* width
    assert not np.array_equal(u, u.T), 'cell is transpose-symmetric; the relabelling is vacuous'
    assert not np.array_equal(u, np.flipud(u)), 'cell is y-symmetric; the mirror test is vacuous'
    assert not np.array_equal(u, np.fliplr(u)), 'cell is x-symmetric'
    assert not np.array_equal(u, np.rot90(u)), 'cell is quarter-turn symmetric'
    return u[None, :, :]


#: Unequal periods and a wavelength short enough to put many orders above the light line. Both
#: matter: at 620 nm only 2 of 77 entries are nonzero, every order map agrees on a mostly-zero
#: array, and an earlier pass at this fooled itself that way. 250 nm leaves 15 propagating.
_TURN_PERIOD = [500.0, 800.0]
_TURN_MAX_ORDER = [5, 3]
_TURN_WAVELENGTH = 250.0
_TURN_THETA = np.deg2rad(25.0)
_TURN_PSI = np.deg2rad(37.0)


def _turn_scene(structure, period, truncation_order, phi, psi, factorization='nvf'):
    """One solve of the turnable scene. Returns ``(R, T, truncation_order)`` -- see :func:`_by_order`."""
    res = wavebit.rcwa(
        backend=0, factorization=factorization, truncation_order=[int(truncation_order[0]), int(truncation_order[1])],
        top=1.0, bot=1.0, theta=_TURN_THETA, phi=phi, psi=psi,
        wavelength=_TURN_WAVELENGTH, period=[period[0], period[1]], thickness=[120.0],
        structure=structure).solve()
    return np.asarray(res.R), np.asarray(res.T), (int(truncation_order[0]), int(truncation_order[1]))


def _by_order(scene):
    """``{(m, n): (R, T)}`` -- keyed by the diffraction order, not by array position.

    Two of the four moving parts below change the *shape* of the returned grids (swapping the
    ``truncation_order`` pair transposes it; relabelling the orders is the map onto it), so a comparison
    written as array arithmetic either cannot be expressed or silently compares the wrong
    entries. Keying by ``(m, n)`` states which physical order is being matched to which and
    leaves the shape out of it.

    ``R`` and ``T`` are ``(n_orders_v, n_orders_u)`` with the incident order at the centre -- the transpose
    of the ``truncation_order=[x, y]`` input order, which is itself one of the swaps under test.
    """
    R, T, (half_width_u, half_width_v) = scene
    return {(m, n): (R[half_width_v + n, half_width_u + m], T[half_width_v + n, half_width_u + m])
            for n in range(-half_width_v, half_width_v + 1) for m in range(-half_width_u, half_width_u + 1)}


def _residue(scene_a, scene_b, relabel):
    """``max |a(m, n) - b(relabel(m, n))|`` over the orders both solves retain."""
    a, b = _by_order(scene_a), _by_order(scene_b)
    worst = 0.0
    for order, (r_a, t_a) in a.items():
        pair = b.get(relabel(*order))
        if pair is not None:
            worst = max(worst, abs(r_a - pair[0]), abs(t_a - pair[1]))
    return float(worst)


_SWAP = lambda m, n: (n, m)          # noqa: E731 -- the x<->y relabelling, as an order map
_MIRROR_Y = lambda m, n: (m, -n)     # noqa: E731


def test_the_turned_scene_is_a_scene_that_can_actually_see_the_turn():
    """The premise of the two tests below: enough orders propagate to tell the maps apart.

    Without this the suite can go green on a degeneracy rather than on the physics. At 620 nm
    with these periods only 2 of 77 entries are nonzero, so the identity map, the transpose and
    the quarter-turn map all agree to roundoff on a grid that is almost entirely zero.
    """
    scene = _turn_scene(_chiral_L(), _TURN_PERIOD, _TURN_MAX_ORDER, np.deg2rad(90.0), _TURN_PSI)
    propagating = int((np.abs(scene[0]) > 1e-14).sum())
    assert propagating >= 10, (
        f'only {propagating} of {scene[0].size} reflected orders propagate; this scene is too '
        f'close to the degenerate one to distinguish the order maps')


@pytest.mark.parametrize('truncation_order', [[5, 3], [8, 6], [11, 9]])
def test_nvf_is_covariant_when_the_whole_scene_is_turned(truncation_order):
    """A layer at ``phi=90`` equals its turned counterpart at ``phi=0`` -- exactly, on nvf.

    A theorem-oracle: it needs no reference implementation, only the same physics set up twice.
    Four things have to move together, and :func:`test_every_part_of_the_turn_is_load_bearing`
    pins that each one is genuinely carrying weight rather than being along for the ride.

    **The spelling is a reflection, not a rotation, and that is a real choice.** Transposing the
    raster maps ``eps(x, y)`` to ``eps(y, x)``, which is a reflection across the diagonal, so it
    reverses orientation and the s direction flips with it: ``psi -> -psi``, i.e. ``a_s -> -a_s``
    (``chi`` likewise, when named). The rotation spelling ``np.rot90(U, 1)`` keeps ``psi`` instead,
    and since ``np.rot90(A, 1) == np.flipud(A.T)`` identically, the two spellings differ by exactly
    one y-mirror. Both are exact as of 2026-08-16; until then this one was 1e-15 and the rotation
    1.4e-03, because the mirror between them was broken -- see
    :func:`test_a_y_mirror_is_exact_on_both_factorizations` for the defect and the fix.

    This test is therefore the *decomposition*, not the whole theorem: it isolates the x<->y
    relabelling half. The theorem as a physicist would state it -- a plain ``np.rot90`` -- is
    :func:`test_a_plain_quarter_turn_of_the_whole_scene_is_exact`, and that is the one to reach for
    first. Neither substitutes for the other: they now agree on every fault they can both see, but
    a regression that moved only one of them would mean the decomposition itself needs re-deriving,
    and that is the case no single test can report.

    Measured on the shipped path over N in {40, 80, 160} x truncation_order in {4..12} x three
    (theta, psi) pairs: 8.6e-16 to 2.7e-13, so 1e-12 keeps roughly two decades of headroom at
    the configurations parametrized here and about 4x at the worst corner of that wider sweep.
    Exact on all three backends (numpy 1.4e-15, jax 1.8e-15, torch 3.3e-15) -- torch used to
    refuse ``np.rot90``'s negative strides, which ``_core.backend.forward_strided`` fixed.

    **What this is structurally blind to, measured rather than assumed.** A relabelling
    covariance test cannot see a bug that is *itself* a consistent relabelling of the two axes:
    the mutation gets conjugated by the very transformation under test, so both halves move
    together and the residue stays at roundoff. Swapping the distance transform's
    ``sampling=(step_j, step_i)`` pair in ``normal_vector_field`` is exactly that shape of bug --
    a live mutant, moving ``R`` by 2.0e-03 -- and it passes every one of the 67 tests in this
    file, this one included. The class is not unguarded, it is just guarded elsewhere: it dies
    in ``tests/test_nvf_non_square_grid.py::test_the_metric_is_what_makes_a_non_square_grid_point_the_right_way``,
    which pins the normal *direction* against the true geometry and so is absolute where this is
    relative. The two are complements, and neither substitutes for the other. What does die here
    is a bug with a hand to it: building ``ny`` from the x gradient kills all three
    parametrizations below.
    """
    structure = _chiral_L()
    upright = _turn_scene(structure, _TURN_PERIOD, truncation_order, np.deg2rad(90.0), _TURN_PSI)
    turned = _turn_scene(structure.transpose(0, 2, 1), _TURN_PERIOD[::-1], truncation_order[::-1],
                         0.0, -_TURN_PSI)
    resid = _residue(upright, turned, _SWAP)
    assert resid < 1e-12, f'nvf is not covariant under a turn of the whole scene: {resid:.2e}'


def test_every_part_of_the_turn_is_load_bearing():
    """Each of the four changes, omitted alone, must break the equivalence.

    This is what keeps the test above from passing for the wrong reason. A covariance check is
    only as strong as the transformation it applies, and three of these four are the kind of
    thing that is easy to leave out and easy not to notice: on a square period the period swap
    is a no-op, on an equal ``truncation_order`` pair so is that swap, and on a mirror-symmetric cell the
    ``psi`` sign is free. Every one of them is a no-op in the three tests this file's header
    block names, which is why none of them could see a turn go wrong.

    The measured cost of each omission, at ``truncation_order=[5, 3]``, N=80 -- all of them four to fourteen
    orders of magnitude above the 1e-12 the full recipe meets.
    """
    structure = _chiral_L()
    turned_cell = structure.transpose(0, 2, 1)
    upright = _turn_scene(structure, _TURN_PERIOD, _TURN_MAX_ORDER, np.deg2rad(90.0), _TURN_PSI)

    omissions = {
        # period pair left unswapped -- the turned cell solved in the upright cell's box
        'period': (_residue(upright, _turn_scene(
            turned_cell, _TURN_PERIOD, _TURN_MAX_ORDER[::-1], 0.0, -_TURN_PSI), _SWAP), 1.2e-1),
        # truncation_order pair left unswapped -- the orders retained along the wrong axis
        'truncation_order': (_residue(upright, _turn_scene(
            turned_cell, _TURN_PERIOD[::-1], _TURN_MAX_ORDER, 0.0, -_TURN_PSI), _SWAP), 2.8e-2),
        # psi left unnegated -- the reflection's s-flip dropped
        'psi': (_residue(upright, _turn_scene(
            turned_cell, _TURN_PERIOD[::-1], _TURN_MAX_ORDER[::-1], 0.0, +_TURN_PSI), _SWAP), 3.4e-2),
    }
    # ...and the order map itself, against the same correct solve the test above uses
    correct = _turn_scene(turned_cell, _TURN_PERIOD[::-1], _TURN_MAX_ORDER[::-1], 0.0, -_TURN_PSI)
    omissions['order map (identity)'] = (
        _residue(upright, correct, lambda m, n: (m, n)), 6.3e-2)
    omissions['order map (quarter turn)'] = (
        _residue(upright, correct, lambda m, n: (n, -m)), 3.8e-2)

    for name, (resid, measured) in omissions.items():
        assert resid > 1e-4, (
            f'omitting the {name} left a residue of only {resid:.2e} (it measured {measured:.1e} '
            f'when this was written); that part of the turn has stopped being load-bearing and '
            f'the covariance test above is correspondingly weaker')


@pytest.mark.parametrize('factorization', ['li', 'nvf'])
@pytest.mark.parametrize('truncation_order', [[4, 2], [6, 4], [8, 6]])
def test_a_y_mirror_is_exact_on_both_factorizations(truncation_order, factorization):
    """Flip the cell in y and the polarisation with it; at ``phi=0`` the orders must flip too.

    ``phi=0`` because the x-z plane of incidence is what a y-mirror leaves invariant. Note this
    is not the same symmetry as the x<->y relabelling higher up, and the two factorizations do
    not hold them for the same reasons: ``li`` fixes one axis inside the other, so it *breaks*
    the relabelling (``test_li_tells_x_and_y_apart_and_this_is_expected``) while holding a
    single-axis mirror exactly, because that axis' nesting is untouched by the flip.

    **``nvf`` used to break this, at 7.6e-04 to 8.4e-03, and the defect was real.** The normal
    field is a *pointwise* sample while the permittivity is *piecewise constant*, and only the
    latter was given the half-pixel phase that puts a value at its cell centre rather than at
    the cell's left edge -- so the normal field sat half a pixel off the permittivity it
    multiplies. A lateral shift of the whole structure cannot absorb that, because it is a
    *relative* offset between two grids read from the same raster. Mirror the layer and the
    offset has to change sign; it did not, the two landed a full pixel apart, and the result was
    an O(1/N) asymmetry. Fixed 2026-08-16 by giving the sampled path the same phase and not the
    ``sinc`` (:func:`~wavebit._core.factorization._half_pixel_phase`); ``nvf`` now runs
    1.8e-15 to 1.6e-14 here, and ``|nvf - li|`` at matched truncation improved in all six
    resolutions measured -- roughly 1.5x to 2x at ``truncation_order=10``.

    **Nothing the suite already had could see it**, which is why it survived the whole rewrite:
    energy conservation, the projector identity ``C_xx + C_yy = I``, the x<->y relabelling, and
    every golden are all even in the half-pixel offset. It took a mirror on a chiral cell.
    """
    structure = _chiral_L()
    upright = _turn_scene(structure, _TURN_PERIOD, truncation_order, 0.0, +_TURN_PSI,
                          factorization=factorization)
    mirrored = _turn_scene(np.flip(structure, axis=1), _TURN_PERIOD, truncation_order, 0.0, -_TURN_PSI,
                           factorization=factorization)
    resid = _residue(upright, mirrored, _MIRROR_Y)
    assert resid < 1e-12, f'{factorization} broke a y-mirror by {resid:.2e}'


@pytest.mark.parametrize('truncation_order', [[5, 3], [8, 6], [11, 9]])
def test_a_plain_quarter_turn_of_the_whole_scene_is_exact(truncation_order):
    """The theorem as it would be stated: turn the scene a quarter turn, the answer follows.

    This is the natural form -- an actual ``np.rot90``, with ``psi`` untouched because a rotation
    preserves handedness -- and it is the one to reach for first. The transpose spelling above is
    a *decomposition* of the same turn, kept because of what it isolates, not because it is the
    statement anybody means.

    The period and truncation_order pairs swap here exactly as they do there. That is not a property of either
    spelling: a quarter turn exchanges the axes, so a 500x800 box becomes 800x500 however the
    raster is written, and omitting the swap costs 1.2e-01 on both.

    **This did not hold until 2026-08-16, and the reason is worth keeping.** Since
    ``np.rot90(A, 1) == np.flipud(A.T)`` identically, this spelling is the transpose one composed
    with a y-mirror::

        quarter turn  =  x<->y relabelling  o  y-mirror

    The relabelling half was always exact; the mirror half was not, and this test inherited the
    whole of that defect -- 5.1e-04 to 4.4e-03, with the two turned solves compared directly
    against each other reproducing it to every digit, which is what said the gap was the mirror
    and contained nothing else. Fixing the half-pixel offset behind it (see
    :func:`test_a_y_mirror_is_exact_on_both_factorizations`) closed this at the same stroke.

    Keeping all three is not redundancy: the decomposition is what localises a regression. A
    handed error injected into the normal field at amplitude 1e-4, 1e-3 and 1e-2 moves the
    transpose spelling to 2.3e-05, 2.3e-04 and 2.3e-03 while this one -- when it was banded at
    the mirror's noise floor -- stayed inside its band and saw none of them. Now that both are
    bounded at 1e-12 they catch the same faults, but they still fail in distinguishable
    combinations: relabelling alone moves the transpose test, a mirror alone moves the mirror
    test, and only a broken *composition* moves this one while both halves stay green.

    Measured 1.6e-15 to 3.7e-14 across the truncations parametrized here.
    """
    structure = _chiral_L()
    upright = _turn_scene(structure, _TURN_PERIOD, truncation_order, np.deg2rad(90.0), _TURN_PSI)
    turned = _turn_scene(np.rot90(structure, 1, axes=(1, 2)), _TURN_PERIOD[::-1], truncation_order[::-1],
                         0.0, _TURN_PSI)                      # psi kept: a rotation, not a mirror
    resid = _residue(upright, turned, lambda m, n: (n, -m))
    assert resid < 1e-12, (
        f'a plain quarter turn of the whole scene is not covariant: {resid:.2e}')


# ------------------------------------------------- what a self-captured golden cannot claim
#
# This file had a companion golden, `golden_legacy_factorization.npz`, that pinned these
# builders against their own earlier output. Such a golden can say "unchanged" and nothing
# else, which is the exact condition under which a wrong sign convention lived in the field
# golden for the whole rewrite (tests/data/gen_goldens_field.py records it). The checks below
# and in the two sections above are the other kind: they compare against a closed form, so they
# would fail on a builder that had always been wrong.
#
# The golden is gone as of 2026-08-18, and every check it carried is now definitional:
# `test_dfs_coefficients_are_the_exact_ones` (done earlier, and the precedent for the rest),
# `test_integrated_coefficients_are_the_exact_ones`,
# `test_the_vector_path_integrates_its_segments_exactly` and
# `test_compression_is_lossless_and_minimal`. Nothing was dropped in the trade.
#
# It was worth finishing rather than leaving half-migrated: 39 of that file's 68 arrays were
# already read by nothing -- residue of the dfs migration above -- and the rewrite it was
# evidence for has long since landed, on a solver that has never been released. Legacy parity
# was protecting no user, and it cost two wrong turns in one session to keep straight.

#: 1.0, 1.5**2 and 2.7**2 -- the squares spelled to the bit, because the assertions below are
#: exact equalities and `7.29` is not `2.7 ** 2`.
@pytest.mark.parametrize('eps', [1.0, 2.25, 7.290000000000001])
@pytest.mark.parametrize('rule', ['dfs', 'cfs', 'nvf'])
def test_a_uniform_cell_factorizes_to_exactly_eps_times_identity(rule, eps):
    """No structure, so every rule must collapse to the same scalar — exactly, not nearly.

    A homogeneous layer has nothing to factorize: Li's rules, the normal-vector projector and
    the direct rule all reduce to multiplication by ``eps``. There is no truncation to hide
    behind either, so the target is ``0.0``, and the off-diagonal slots must be identically
    zero rather than small.

    This is the check that caught the patterned-anisotropic factorization in 2026-08-14: its
    first arrangement passed energy conservation, rotation covariance *and* an external
    comparison while being wrong by 0.5 on a uniform slab. Cheap structural checks with exact
    answers catch what plausibility does not.
    """
    builder = {'dfs': core_dfs, 'cfs': core_cfs, 'nvf': core_nvf}[rule]
    cell = np.full((1, 8, 8), eps, dtype=complex)

    out = builder(cell, 3, 3)
    epx, epy = np.asarray(out[0])[0], np.asarray(out[1])[0]
    epz_i = np.asarray(out[2])[0]
    n_orders = epx.shape[0]
    I = np.eye(n_orders)

    assert np.max(np.abs(epx - eps * I)) == 0.0, np.max(np.abs(epx - eps * I))
    assert np.max(np.abs(epy - eps * I)) == 0.0, np.max(np.abs(epy - eps * I))
    # `epz_i` is the inverse of the zz block, so its closed form is 1/eps
    assert np.max(np.abs(epz_i - I / eps)) == 0.0, np.max(np.abs(epz_i - I / eps))

    if rule == 'nvf':                      # (epx, epy, epz_i, epxy, epyx)
        for k in (3, 4):
            off = np.asarray(out[k])[0]
            assert np.max(np.abs(off)) == 0.0, f'slot {k} is not identically zero'


@pytest.mark.parametrize('axis', ['x', 'y'])
def test_the_normal_field_of_a_straight_interface_is_the_axis_normal(axis):
    """One geometry where the normal field *does* have a closed form, so use it.

    ``test_nvf_permittivity_inputs_are_the_exact_ones`` notes that the normal field "is a
    continuous vector field sampled on a grid: no closed form replaces it". True in general —
    and false for the cases that matter most cheaply. A grating whose only interfaces are
    straight lines parallel to an axis has a normal pointing along the other axis *everywhere*,
    so one component must vanish identically. That pins the orientation, which is the half of
    the field a golden captured from the same builder can never arbitrate.

    The x/y pair getting swapped is a live failure mode in this codebase --
    ``wavebit/_core/lattice.py`` records that unpacking a period as ``(Py, Px)`` is "a mistake
    nothing reacts to" -- and this is the cheapest statement of it that needs no solve.
    """
    cell = np.ones((1, 16, 16), dtype=complex)
    if axis == 'x':                        # stripes varying along x -> normal is +-x_hat
        cell[0, :, 6:11] = 6.25
        vanishing, present = 1, 0
    else:                                  # stripes varying along y -> normal is +-y_hat
        cell[0, 6:11, :] = 6.25
        vanishing, present = 0, 1

    nf = np.asarray(normal_field_for_raster(cell, 3, 3))[0]
    assert np.max(np.abs(nf[vanishing])) == 0.0, (
        f'the {"y" if vanishing else "x"} component must vanish for a {axis}-varying grating, '
        f'got {np.max(np.abs(nf[vanishing])):.3e}')
    assert np.max(np.abs(nf[present])) == pytest.approx(1.0, abs=1e-12)
