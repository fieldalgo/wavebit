"""Single-source 2D solve core (fieldwave._core.solve_2d).

Cross-backend agreement, cross-backend-consistent gradients, and a check that driving the core
through ``conv_solve`` matches calling it directly. The physics oracle is the frozen legacy
reference in test_golden_legacy_parity.py.
"""
import numpy as np
import pytest

import fieldwave
from fieldwave._core.factorization import to_conv_mat_raster_discrete
from fieldwave._core.solve_2d import solve_2d


def _run_and_inputs(ucell, thickness, n_top, n_bot, wl, period, fto, theta, phi, psi):
    solver = fieldwave.rcwa(backend=0, n_top=n_top, n_bot=n_bot, theta=theta, phi=phi, psi=psi,
                          fto=fto, period=period, wavelength=wl, thickness=thickness, ucell=ucell)
    via_api = solver.conv_solve()
    assert solver.grating_type_assigned == 2, solver.grating_type_assigned    # ensure 2D path
    epx, epy, epz = to_conv_mat_raster_discrete(
        solver.ucell, solver.fto[0], solver.fto[1],
        enhanced_dfs=solver.enhanced_dfs)
    kx, ky = solver.get_kx_ky_vector(wl)
    inputs = dict(wavelength=wl, kx=kx, ky=ky, n_top=n_top, n_bot=n_bot,
                  theta=solver.theta, psi=solver.psi, thickness=solver.thickness,
                  epx_conv_all=epx, epy_conv_all=epy, epz_conv_i_all=epz)
    return (np.array(via_api.de_ri), np.array(via_api.de_ti)), inputs


def _rand(shape, seed):
    return np.random.default_rng(seed).uniform(1.0, 4.0, size=shape)


CASES = [
    # name, ucell, thickness, n_top, n_bot, wl, period, fto, theta, phi, psi
    ('normal', _rand((1, 5, 5), 0), [0.2], 1.0, 1.0, 0.9, [1.0, 1.0], [2, 2], 0.0, 0.0, 0.0),
    ('oblique_psi', _rand((1, 6, 6), 1), [0.3], 1.0, 1.5, 0.8, [1.0, 1.2], [2, 2], 0.3, 0.4, 0.9),
    ('two_layer', _rand((2, 4, 4), 2), [0.15, 0.22], 1.2, 1.0, 0.7, [0.9, 0.9], [2, 2], 0.2, 0.5, 1.1),
]


@pytest.mark.parametrize('case', CASES, ids=[c[0] for c in CASES])
def test_2d_wiring_matches_direct_core_call(case):
    _, ucell, thickness, n_top, n_bot, wl, period, fto, theta, phi, psi = case
    (lri, lti), inp = _run_and_inputs(
        ucell, thickness, n_top, n_bot, wl, period, fto, theta, phi, psi)
    cri, cti = solve_2d(**inp)
    assert np.array_equal(lri, np.array(cri)), np.max(np.abs(lri - np.array(cri)))
    assert np.array_equal(lti, np.array(cti)), np.max(np.abs(lti - np.array(cti)))


def _to(backend, x):
    if backend == 'jax':
        import jax.numpy as jnp
        return jnp.asarray(x)
    import torch
    return torch.as_tensor(np.asarray(x))


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('case', CASES, ids=[c[0] for c in CASES])
def test_2d_cross_backend(case, backend):
    pytest.importorskip(backend)
    _, ucell, thickness, n_top, n_bot, wl, period, fto, theta, phi, psi = case
    (_, _), inp = _run_and_inputs(
        ucell, thickness, n_top, n_bot, wl, period, fto, theta, phi, psi)
    ri0, ti0 = solve_2d(**inp)                               # numpy reference

    bk = dict(inp)
    for k in ('kx', 'ky', 'epx_conv_all', 'epy_conv_all', 'epz_conv_i_all'):
        bk[k] = _to(backend, inp[k])
    bk['theta'] = _to(backend, complex(inp['theta']))
    bk['psi'] = _to(backend, complex(inp['psi']))
    ri, ti = solve_2d(**bk)

    assert np.allclose(np.array(ri0), np.asarray(ri), atol=1e-10, rtol=0)
    assert np.allclose(np.array(ti0), np.asarray(ti), atol=1e-10, rtol=0)


def test_2d_gradient_jax_torch_fd():
    """d(sum de_ti)/ds through the coupled 2D block chain: jax == torch == finite-diff."""
    jax = pytest.importorskip('jax')
    torch = pytest.importorskip('torch')
    import jax.numpy as jnp

    ucell = _rand((1, 4, 4), 3)
    (_, _), inp = _run_and_inputs(ucell, [0.15], 1.0, 1.0, 0.9, [1.0, 1.0], [1, 1], 0.2, 0.4, 0.9)
    key = 'epx_conv_all'
    base = np.array(inp[key])
    direction = np.real(base).copy()
    conv_keys = ('kx', 'ky', 'epx_conv_all', 'epy_conv_all', 'epz_conv_i_all')

    def out(d):
        _, ti = solve_2d(**d)
        return ti

    def f_np(s):
        d = dict(inp); d[key] = base + s * direction
        return float(np.sum(np.array(out(d))))
    h = 1e-6
    g_fd = (f_np(h) - f_np(-h)) / (2 * h)

    def f_jax(s):
        d = dict(inp)
        for k in conv_keys:
            d[k] = jnp.asarray(inp[k])
        d[key] = jnp.asarray(base) + s * jnp.asarray(direction)
        d['theta'] = jnp.asarray(complex(inp['theta']))
        d['psi'] = jnp.asarray(complex(inp['psi']))
        return jnp.sum(out(d)).real
    g_jax = float(jax.grad(lambda s: f_jax(s))(0.0))

    def f_torch(s):
        d = dict(inp)
        for k in conv_keys:
            d[k] = torch.as_tensor(np.asarray(inp[k]))
        d[key] = torch.as_tensor(base) + s * torch.as_tensor(direction)
        d['theta'] = torch.as_tensor(complex(inp['theta']))
        d['psi'] = torch.as_tensor(complex(inp['psi']))
        return torch.sum(out(d)).real
    s = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    val = f_torch(s)
    val.backward()
    g_torch = float(s.grad)

    assert abs(g_jax - g_torch) < 1e-6, (g_jax, g_torch)
    assert abs(g_jax - g_fd) < 1e-4, (g_jax, g_fd)


def test_python_scalar_incidence_keeps_double_precision_on_torch():
    """A Python-complex theta/psi must not inherit torch's default float32 dtype.

    ``torch.as_tensor(complex(...))`` silently yields complex64 when the default dtype is
    float32, which costs ~8 digits in the efficiency denominators and shows up as a
    cross-backend gradient mismatch. The core builds incidence scalars at its own working
    dtype instead, so passing them through raw is precision-safe either way.
    """
    torch = pytest.importorskip('torch')
    _, ucell, thickness, n_top, n_bot, wl, period, fto, theta, phi, psi = CASES[1]
    (_, _), inp = _run_and_inputs(
        ucell, thickness, n_top, n_bot, wl, period, fto, theta, phi, psi)
    ri0, ti0 = solve_2d(**inp)                                   # numpy, double

    default_before = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)                       # the hostile setting
    try:
        bk = dict(inp)
        for k in ('kx', 'ky', 'epx_conv_all', 'epy_conv_all', 'epz_conv_i_all'):
            bk[k] = torch.as_tensor(np.asarray(inp[k]))
        bk['theta'] = complex(inp['theta'])                      # raw Python scalars
        bk['psi'] = complex(inp['psi'])
        ri, ti = solve_2d(**bk)
    finally:
        torch.set_default_dtype(default_before)

    assert np.allclose(np.array(ri0), np.asarray(ri), atol=1e-12, rtol=0)
    assert np.allclose(np.array(ti0), np.asarray(ti), atol=1e-12, rtol=0)
