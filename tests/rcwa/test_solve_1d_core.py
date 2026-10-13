"""Single-source 1D solve core (wavebit.rcwa._engine.solve_1d).

Three properties, one implementation:
  * wiring        — driving the core through ``rcwa``/``solve`` gives the same numbers
                    as calling it directly (the mixin passes the right arguments);
  * cross-backend — the same core on numpy / jax / torch agrees to ~1e-10;
  * gradients     — d(sum T)/ds through the complex chain agrees across jax / torch / FD.

The physics oracle lives in test_golden_legacy_parity.py: reference values frozen from the
legacy transfer-matrix chains before they were replaced. Comparing against ``solve``
in-process would prove nothing now that it dispatches into the core.
"""

from wavebit.compat import legacy as _legacy_wavebit
import numpy as np
import pytest

from wavebit.rcwa._engine.factorization import to_conv_mat_raster_sampled
from wavebit.rcwa._engine.solve_1d import solve_1d, solve_1d_conical

# Enables jax x64, which every jax case here needs: at float32 the cross-backend comparisons
# miss their 1e-10 tolerance and the gradient checks miss 1e-6. Guarded because jax is optional
# -- when it is absent the jax cases skip themselves via `pytest.importorskip`.
#
# Without it this file passes only when some *other* test module has already lifted the cap,
# because x64 is a cap set at import time and cannot be raised afterwards.
# The whole suite was green while `pytest tests/rcwa/test_solve_1d_core.py` alone failed 13 of these
# -- every `[*-jax]` cross-backend case plus two gradient checks -- and a failure that looks
# like a real cross-backend physics bug is exactly what that hides.
#
# At the top of the module rather than beside each `import jax.numpy`, so a jax case added
# later cannot silently inherit float32. `test_factorization_core.py` carries the same guard
# for the same reason, found the same way.
try:
    _legacy_wavebit.enable_x64()
except ImportError:
    pass


def _run_and_inputs(tetm, structure, thickness, n_top, n_bot, wl, period, truncation_order, theta):
    """Solve via the public API and return (its R/T, the same inputs)."""
    # Two quantities, one pair of half-spaces. The constructor takes permittivity; `solve_1d`
    # below takes the index, which is what the amplitude ratios in it actually are.
    solver = _legacy_wavebit.rcwa(backend='numpy', top=n_top ** 2, bot=n_bot ** 2, theta=theta, phi=None,
                          tetm=tetm, truncation_order=truncation_order, period=period, wavelength=wl,
                          thickness=thickness[::-1], structure=structure[::-1])
    via_api = solver.solve()
    epx, epy, epz, *_ = to_conv_mat_raster_sampled(
        solver.structure, solver.truncation_order[0], solver.truncation_order[1])
    kx, _ = solver.order_kxy()
    inputs = dict(tetm_index=0 if tetm == 'te' else 1, wavelength=wl, kx=kx, n_top=n_bot, n_bot=n_top, incidence_side=solver._kernel_incidence_side,
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
    # name, tetm, structure, thickness, n_top, n_bot, wl, period, truncation_order, theta.
    # The rasters are PERMITTIVITY (9 and 4 where the indices 3 and 2 used to be); the
    # half-spaces stay indices and are squared at the constructor -- see _run_and_inputs.
    # `structure` / `thickness` list the layers TOP TO BOTTOM, the order these cases were
    # written in; `_run_and_inputs` reverses them into the public bottom-to-top order.
    ('te_1layer', 'te', [[[9., 9., 9., 1., 1., 1.]]], [0.15], 2.0, 1.0, 0.777, [0.77], [6, 0], 0.0),
    ('tm_1layer', 'tm', [[[9., 9., 9., 1., 1., 1.]]], [0.15], 2.0, 1.0, 0.777, [0.77], [6, 0], 0.0),
    ('te_2layer', 'te', [[[9., 9., 1., 1., 9., 1.]], [[4., 4., 4., 1., 1., 1.]]], [0.15, 0.22],
     1.0, 1.0, 0.9, [0.7], [5, 0], 0.0),
    ('tm_2layer', 'tm', [[[9., 9., 1., 1., 9., 1.]], [[4., 4., 4., 1., 1., 1.]]], [0.15, 0.22],
     1.0, 1.0, 0.9, [0.7], [5, 0], 0.0),
    ('te_oblique_lossy', 'te', [[[9., 9., 9., 1., 1., 1.]]], [0.2], 1.0, 1.5, 0.777, [0.77], [6, 0], 0.3),
    ('tm_oblique_lossy', 'tm', [[[9., 9., 9., 1., 1., 1.]]], [0.2], 1.0, 1.5, 0.777, [0.77], [6, 0], 0.3),
    # bottom layer -- listed last here, structure[0] once reversed -- uniform and equal to the
    # substrate (bot=1): a matched interface
    ('te_same_material', 'te', [[[9., 9., 1., 1., 9., 1.]], [[1., 1., 1., 1., 1., 1.]]], [0.2, 0.1],
     1.0, 1.0, 0.9, [0.7], [5, 0], 0.0),
    ('tm_same_material', 'tm', [[[9., 9., 1., 1., 9., 1.]], [[1., 1., 1., 1., 1., 1.]]], [0.2, 0.1],
     1.0, 1.0, 0.9, [0.7], [5, 0], 0.0),
]


@pytest.mark.parametrize('case', CASES, ids=[c[0] for c in CASES])
def test_wiring_matches_direct_core_call(case):
    _, tetm, structure, thickness, n_top, n_bot, wl, period, truncation_order, theta = case
    (lri, lti), inputs = _run_and_inputs(
        tetm, np.array(structure), thickness, n_top, n_bot, wl, period, truncation_order, theta)
    cri, cti, _ = solve_1d(**inputs)
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
    _, tetm, structure, thickness, n_top, n_bot, wl, period, truncation_order, theta = case
    (_, _), inp = _run_and_inputs(
        tetm, np.array(structure), thickness, n_top, n_bot, wl, period, truncation_order, theta)
    ri0, ti0, _ = solve_1d(**inp)                              # numpy reference

    bk = dict(inp)
    for k in ('kx', 'epx_conv_all', 'epy_conv_all', 'epz_conv_i_all'):
        bk[k] = _to(backend, inp[k])
    bk['theta'] = _to(backend, complex(inp['theta']))
    ri, ti, _ = solve_1d(**bk)

    assert np.allclose(np.array(ri0), np.asarray(ri), atol=1e-10, rtol=0)
    assert np.allclose(np.array(ti0), np.asarray(ti), atol=1e-10, rtol=0)


@pytest.mark.parametrize('tetm', ['te', 'tm'])
def test_core_gradient_jax_torch_fd(tetm):
    """d(sum T)/ds where s scales a real perturbation of the layer eps-conv matrix."""
    jax = pytest.importorskip('jax')
    torch = pytest.importorskip('torch')
    import jax.numpy as jnp

    structure = np.array([[[9., 9., 9., 1., 1., 1.]]])
    (_, _), inp = _run_and_inputs(tetm, structure, [0.15], 2.0, 1.0, 0.777, [0.77], [6, 0], 0.3)
    # perturb epx (TM couples through it; TE is unaffected -> use epy for TE)
    key = 'epy_conv_all' if tetm == 'te' else 'epx_conv_all'
    base = np.array(inp[key])
    direction = np.real(base).copy()

    def out(sub_inputs):
        _, ti, _ = solve_1d(**sub_inputs)
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

def _run_and_inputs_conical(structure, thickness, n_top, n_bot, wl, period, truncation_order, theta, phi, psi):
    """Same, forced onto the conical path."""
    solver = _legacy_wavebit.rcwa(backend='numpy', top=n_top ** 2, bot=n_bot ** 2, theta=theta, phi=phi,
                          psi=psi, truncation_order=truncation_order, period=period, wavelength=wl,
                          thickness=thickness[::-1], structure=structure[::-1])
    via_api = solver.solve()
    assert solver.grating_type == 1, solver.grating_type   # ensure conical path
    epx, epy, epz, *_ = to_conv_mat_raster_sampled(
        solver.structure, solver.truncation_order[0], solver.truncation_order[1])
    kx, ky = solver.order_kxy()
    inputs = dict(wavelength=wl, kx=kx, ky=ky, n_top=n_bot, n_bot=n_top, incidence_side=solver._kernel_incidence_side,
                  theta=solver.theta, phi=solver.phi,
                  incidence=solver._incidence_pair(), thickness=solver.thickness,
                  epx_conv_all=epx, epy_conv_all=epy, epz_conv_i_all=epz)
    return (np.array(via_api.R), np.array(via_api.T)), inputs


CONICAL_CASES = [
    # name, structure, thickness, n_top, n_bot, wl, period, truncation_order, theta, phi, psi
    # (layers top to bottom, as in CASES; `_run_and_inputs_conical` reverses them)
    ('con_1layer', [[[9., 9., 9., 1., 1., 1.]]], [0.15], 2.0, 1.0, 0.777, [0.77], [6, 0], 0.2, 0.3, 0.4),
    ('con_2layer_lossy', [[[9., 9., 1., 1., 9., 1.]], [[4., 4., 4., 1., 1., 1.]]], [0.15, 0.22],
     1.0, 1.5, 0.9, [0.7], [5, 0], 0.3, 0.5, 1.1),
    ('con_phi0', [[[16., 16., 1., 1., 1., 1.]]], [0.3], 1.5, 1.0, 0.6, [0.5], [4, 0], 0.25, 0.0, 0.7),
]


@pytest.mark.parametrize('case', CONICAL_CASES, ids=[c[0] for c in CONICAL_CASES])
def test_conical_wiring_matches_direct_core_call(case):
    _, structure, thickness, n_top, n_bot, wl, period, truncation_order, theta, phi, psi = case
    (lri, lti), inp = _run_and_inputs_conical(
        np.array(structure), thickness, n_top, n_bot, wl, period, truncation_order, theta, phi, psi)
    cri, cti, _ = solve_1d_conical(**inp)
    _assert_same_answer((lri, lti), (cri, cti))


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('case', CONICAL_CASES, ids=[c[0] for c in CONICAL_CASES])
def test_conical_cross_backend(case, backend):
    pytest.importorskip(backend)
    _, structure, thickness, n_top, n_bot, wl, period, truncation_order, theta, phi, psi = case
    (_, _), inp = _run_and_inputs_conical(
        np.array(structure), thickness, n_top, n_bot, wl, period, truncation_order, theta, phi, psi)
    ri0, ti0, _ = solve_1d_conical(**inp)                       # numpy reference

    bk = dict(inp)
    for k in ('kx', 'ky', 'epx_conv_all', 'epy_conv_all', 'epz_conv_i_all'):
        bk[k] = _to(backend, inp[k])
    bk['theta'] = _to(backend, complex(inp['theta']))
    bk['incidence'] = tuple(_to(backend, complex(v)) for v in inp['incidence'])
    ri, ti, _ = solve_1d_conical(**bk)

    assert np.allclose(np.array(ri0), np.asarray(ri), atol=1e-10, rtol=0)
    assert np.allclose(np.array(ti0), np.asarray(ti), atol=1e-10, rtol=0)


def test_conical_gradient_jax_torch_fd():
    """d(sum T)/ds through the coupled s/p conical chain: jax == torch == finite-diff."""
    jax = pytest.importorskip('jax')
    torch = pytest.importorskip('torch')
    import jax.numpy as jnp

    structure = np.array([[[9., 9., 9., 1., 1., 1.]]])
    (_, _), inp = _run_and_inputs_conical(
        structure, [0.15], 2.0, 1.0, 0.777, [0.77], [6, 0], 0.2, 0.3, 0.4)
    key = 'epx_conv_all'
    base = np.array(inp[key])
    direction = np.real(base).copy()
    conv_keys = ('kx', 'ky', 'epx_conv_all', 'epy_conv_all', 'epz_conv_i_all')

    def out(d):
        _, ti, _ = solve_1d_conical(**d)
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
        d['incidence'] = tuple(jnp.asarray(complex(v)) for v in inp['incidence'])
        return jnp.sum(out(d)).real
    g_jax = float(jax.grad(lambda s: f_jax(s))(0.0))

    def f_torch(s):
        d = dict(inp)
        for k in conv_keys:
            d[k] = torch.as_tensor(np.asarray(inp[k]))
        d[key] = torch.as_tensor(base) + s * torch.as_tensor(direction)
        d['theta'] = torch.as_tensor(complex(inp['theta']))
        d['incidence'] = tuple(torch.as_tensor(complex(v)) for v in inp['incidence'])
        return torch.sum(out(d)).real
    s = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    val = f_torch(s)
    val.backward()
    g_torch = float(s.grad)

    assert abs(g_jax - g_torch) < 1e-6, (g_jax, g_torch)
    assert abs(g_jax - g_fd) < 1e-4, (g_jax, g_fd)
