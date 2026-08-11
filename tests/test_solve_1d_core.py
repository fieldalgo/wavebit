"""Single-source 1D solve core (fieldwave._core.solve_1d).

Three properties, one implementation:
  * wiring        — driving the core through ``rcwa``/``conv_solve`` gives the same numbers
                    as calling it directly (the mixin passes the right arguments);
  * cross-backend — the same core on numpy / jax / torch agrees to ~1e-10;
  * gradients     — d(sum T)/ds through the complex chain agrees across jax / torch / FD.

The physics oracle lives in test_golden_legacy_parity.py: reference values frozen from the
legacy transfer-matrix chains before they were replaced. Comparing against ``conv_solve``
in-process would prove nothing now that it dispatches into the core.
"""
import numpy as np
import pytest

import fieldwave
from fieldwave._core.factorization import to_conv_mat_raster_discrete
from fieldwave._core.solve_1d import solve_1d, solve_1d_conical


def _run_and_inputs(pol, ucell, thickness, n_top, n_bot, wl, period, fto, theta):
    """Solve via the public API and return (its R/T, the same inputs)."""
    solver = fieldwave.rcwa(backend=0, n_top=n_top, n_bot=n_bot, theta=theta, phi=None,
                          pol=pol, fto=fto, period=period, wavelength=wl,
                          thickness=thickness, ucell=ucell)
    via_api = solver.conv_solve()
    epx, epy, epz = to_conv_mat_raster_discrete(
        solver.ucell, solver.fto[0], solver.fto[1])
    kx, _ = solver.get_kx_ky_vector(wl)
    inputs = dict(pol=pol, wavelength=wl, kx=kx, n_top=n_top, n_bot=n_bot,
                  theta=solver.theta, thickness=solver.thickness,
                  epx_conv_all=epx, epy_conv_all=epy, epz_conv_i_all=epz)
    return (np.array(via_api.R), np.array(via_api.T)), inputs


def _assert_same_answer(via_api, direct):
    """The API route and the direct core call agree — to a tolerance, not bit for bit.

    The property under test is that the mixin passes the right arguments, and every way of
    getting that wrong moves the efficiencies by far more than 1e-12. Exact equality would
    additionally assert that two independent runs of the same expressions land on the same last
    bit, which is a claim about the BLAS rather than about this code. Its SMM twin has failed
    that way intermittently on GitHub's runners, on the conical case this file also carries; see
    the note on that test in test_scattering_core.py for what it looked like and for which tests
    do own bit-identity.
    """
    for a, b in zip(via_api, direct):
        np.testing.assert_allclose(a, np.array(b), rtol=1e-12, atol=1e-15)


CASES = [
    # name, pol, ucell, thickness, n_top, n_bot, wl, period, fto, theta
    ('te_1layer', 0, [[[3., 3., 3., 1., 1., 1.]]], [0.15], 2.0, 1.0, 0.777, [0.77], [6, 0], 0.0),
    ('tm_1layer', 1, [[[3., 3., 3., 1., 1., 1.]]], [0.15], 2.0, 1.0, 0.777, [0.77], [6, 0], 0.0),
    ('te_2layer', 0, [[[3., 3., 1., 1., 3., 1.]], [[2., 2., 2., 1., 1., 1.]]], [0.15, 0.22],
     1.0, 1.0, 0.9, [0.7], [5, 0], 0.0),
    ('tm_2layer', 1, [[[3., 3., 1., 1., 3., 1.]], [[2., 2., 2., 1., 1., 1.]]], [0.15, 0.22],
     1.0, 1.0, 0.9, [0.7], [5, 0], 0.0),
    ('te_oblique_lossy', 0, [[[3., 3., 3., 1., 1., 1.]]], [0.2], 1.0, 1.5, 0.777, [0.77], [6, 0], 0.3),
    ('tm_oblique_lossy', 1, [[[3., 3., 3., 1., 1., 1.]]], [0.2], 1.0, 1.5, 0.777, [0.77], [6, 0], 0.3),
    # bottom layer uniform and equal to the substrate (n_bot=1): a matched interface
    ('te_same_material', 0, [[[3., 3., 1., 1., 3., 1.]], [[1., 1., 1., 1., 1., 1.]]], [0.2, 0.1],
     1.0, 1.0, 0.9, [0.7], [5, 0], 0.0),
    ('tm_same_material', 1, [[[3., 3., 1., 1., 3., 1.]], [[1., 1., 1., 1., 1., 1.]]], [0.2, 0.1],
     1.0, 1.0, 0.9, [0.7], [5, 0], 0.0),
]


@pytest.mark.parametrize('case', CASES, ids=[c[0] for c in CASES])
def test_wiring_matches_direct_core_call(case):
    _, pol, ucell, thickness, n_top, n_bot, wl, period, fto, theta = case
    (lri, lti), inputs = _run_and_inputs(
        pol, np.array(ucell), thickness, n_top, n_bot, wl, period, fto, theta)
    cri, cti = solve_1d(**inputs)
    _assert_same_answer((lri, lti), (cri, cti))


def _to(backend, x):
    if backend == 'jax':
        import jax.numpy as jnp
        return jnp.asarray(x)
    import torch
    return torch.as_tensor(np.asarray(x))


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('case', CASES, ids=[c[0] for c in CASES])
def test_core_cross_backend(case, backend):
    pytest.importorskip(backend)
    _, pol, ucell, thickness, n_top, n_bot, wl, period, fto, theta = case
    (_, _), inp = _run_and_inputs(
        pol, np.array(ucell), thickness, n_top, n_bot, wl, period, fto, theta)
    ri0, ti0 = solve_1d(**inp)                              # numpy reference

    bk = dict(inp)
    for k in ('kx', 'epx_conv_all', 'epy_conv_all', 'epz_conv_i_all'):
        bk[k] = _to(backend, inp[k])
    bk['theta'] = _to(backend, complex(inp['theta']))
    ri, ti = solve_1d(**bk)

    assert np.allclose(np.array(ri0), np.asarray(ri), atol=1e-10, rtol=0)
    assert np.allclose(np.array(ti0), np.asarray(ti), atol=1e-10, rtol=0)


@pytest.mark.parametrize('pol', [0, 1], ids=['te', 'tm'])
def test_core_gradient_jax_torch_fd(pol):
    """d(sum T)/ds where s scales a real perturbation of the layer eps-conv matrix."""
    jax = pytest.importorskip('jax')
    torch = pytest.importorskip('torch')
    import jax.numpy as jnp

    ucell = np.array([[[3., 3., 3., 1., 1., 1.]]])
    (_, _), inp = _run_and_inputs(pol, ucell, [0.15], 2.0, 1.0, 0.777, [0.77], [6, 0], 0.3)
    # perturb epx (TM couples through it; TE is unaffected -> use epy for TE)
    key = 'epy_conv_all' if pol == 0 else 'epx_conv_all'
    base = np.array(inp[key])
    direction = np.real(base).copy()

    def out(sub_inputs):
        _, ti = solve_1d(**sub_inputs)
        return ti

    def f_np(s):
        d = dict(inp); d[key] = base + s * direction
        return float(np.sum(np.array(out(d))))

    h = 1e-6
    g_fd = (f_np(h) - f_np(-h)) / (2 * h)

    conv_keys = ('kx', 'epx_conv_all', 'epy_conv_all', 'epz_conv_i_all')

    def f_jax(s):
        d = dict(inp)
        for k in conv_keys:
            d[k] = jnp.asarray(inp[k])
        d[key] = jnp.asarray(base) + s * jnp.asarray(direction)   # perturbed leaf
        d['theta'] = jnp.asarray(complex(inp['theta']))
        return jnp.sum(out(d)).real
    g_jax = float(jax.grad(lambda s: f_jax(s))(0.0))

    def f_torch(s):
        d = dict(inp)
        for k in conv_keys:
            d[k] = torch.as_tensor(np.asarray(inp[k]))
        d[key] = torch.as_tensor(base) + s * torch.as_tensor(direction)   # perturbed leaf
        d['theta'] = torch.as_tensor(complex(inp['theta']))
        return torch.sum(out(d)).real
    s = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    val = f_torch(s)
    val.backward()
    g_torch = float(s.grad)

    assert abs(g_jax - g_torch) < 1e-6, (g_jax, g_torch)
    assert abs(g_jax - g_fd) < 1e-4, (g_jax, g_fd)


# =========================================================================== conical

def _run_and_inputs_conical(ucell, thickness, n_top, n_bot, wl, period, fto, theta, phi, psi):
    """Same, forced onto the conical path."""
    solver = fieldwave.rcwa(backend=0, n_top=n_top, n_bot=n_bot, theta=theta, phi=phi, psi=psi,
                          fto=fto, period=period, wavelength=wl, thickness=thickness, ucell=ucell)
    via_api = solver.conv_solve()
    assert solver.grating_type_assigned == 1, solver.grating_type_assigned   # ensure conical path
    epx, epy, epz = to_conv_mat_raster_discrete(
        solver.ucell, solver.fto[0], solver.fto[1])
    kx, ky = solver.get_kx_ky_vector(wl)
    inputs = dict(wavelength=wl, kx=kx, ky=ky, n_top=n_top, n_bot=n_bot,
                  theta=solver.theta, psi=solver.psi, thickness=solver.thickness,
                  epx_conv_all=epx, epy_conv_all=epy, epz_conv_i_all=epz)
    return (np.array(via_api.R), np.array(via_api.T)), inputs


CONICAL_CASES = [
    # name, ucell, thickness, n_top, n_bot, wl, period, fto, theta, phi, psi
    ('con_1layer', [[[3., 3., 3., 1., 1., 1.]]], [0.15], 2.0, 1.0, 0.777, [0.77], [6, 0], 0.2, 0.3, 0.4),
    ('con_2layer_lossy', [[[3., 3., 1., 1., 3., 1.]], [[2., 2., 2., 1., 1., 1.]]], [0.15, 0.22],
     1.0, 1.5, 0.9, [0.7], [5, 0], 0.3, 0.5, 1.1),
    ('con_phi0', [[[4., 4., 1., 1., 1., 1.]]], [0.3], 1.5, 1.0, 0.6, [0.5], [4, 0], 0.25, 0.0, 0.7),
]


@pytest.mark.parametrize('case', CONICAL_CASES, ids=[c[0] for c in CONICAL_CASES])
def test_conical_wiring_matches_direct_core_call(case):
    _, ucell, thickness, n_top, n_bot, wl, period, fto, theta, phi, psi = case
    (lri, lti), inp = _run_and_inputs_conical(
        np.array(ucell), thickness, n_top, n_bot, wl, period, fto, theta, phi, psi)
    cri, cti = solve_1d_conical(**inp)
    _assert_same_answer((lri, lti), (cri, cti))


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('case', CONICAL_CASES, ids=[c[0] for c in CONICAL_CASES])
def test_conical_cross_backend(case, backend):
    pytest.importorskip(backend)
    _, ucell, thickness, n_top, n_bot, wl, period, fto, theta, phi, psi = case
    (_, _), inp = _run_and_inputs_conical(
        np.array(ucell), thickness, n_top, n_bot, wl, period, fto, theta, phi, psi)
    ri0, ti0 = solve_1d_conical(**inp)                       # numpy reference

    bk = dict(inp)
    for k in ('kx', 'ky', 'epx_conv_all', 'epy_conv_all', 'epz_conv_i_all'):
        bk[k] = _to(backend, inp[k])
    bk['theta'] = _to(backend, complex(inp['theta']))
    bk['psi'] = _to(backend, complex(inp['psi']))
    ri, ti = solve_1d_conical(**bk)

    assert np.allclose(np.array(ri0), np.asarray(ri), atol=1e-10, rtol=0)
    assert np.allclose(np.array(ti0), np.asarray(ti), atol=1e-10, rtol=0)


def test_conical_gradient_jax_torch_fd():
    """d(sum T)/ds through the coupled s/p conical chain: jax == torch == finite-diff."""
    jax = pytest.importorskip('jax')
    torch = pytest.importorskip('torch')
    import jax.numpy as jnp

    ucell = np.array([[[3., 3., 3., 1., 1., 1.]]])
    (_, _), inp = _run_and_inputs_conical(
        ucell, [0.15], 2.0, 1.0, 0.777, [0.77], [6, 0], 0.2, 0.3, 0.4)
    key = 'epx_conv_all'
    base = np.array(inp[key])
    direction = np.real(base).copy()
    conv_keys = ('kx', 'ky', 'epx_conv_all', 'epy_conv_all', 'epz_conv_i_all')

    def out(d):
        _, ti = solve_1d_conical(**d)
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
