"""Single-source 2D solve core (wavebit._core.solve_2d).

Cross-backend agreement, cross-backend-consistent gradients, and a check that driving the core
through ``conv_solve`` matches calling it directly. The physics oracle is the frozen legacy
reference in test_golden_legacy_parity.py.
"""
import numpy as np
import pytest

import wavebit
from wavebit._core.factorization import to_conv_mat_raster_sampled
from wavebit._core.solve_2d import solve_2d

# Enables jax x64, which every jax case here needs: at float32 the cross-backend comparisons
# miss their 1e-10 tolerance and the gradient checks miss 1e-6. Guarded because jax is optional
# -- when it is absent the jax cases skip themselves via `pytest.importorskip`.
#
# Without it this file passes only when some *other* test module has already imported
# `wavebit.on_jax`, because x64 is a cap set at import time and cannot be raised afterwards.
# The whole suite was green while `pytest tests/test_solve_2d_core.py` alone failed 4 of these
# -- three `[*-jax]` cross-backend cases and the gradient check -- and a failure that looks
# like a real cross-backend physics bug is exactly what that hides.
#
# At the top of the module rather than beside each `import jax.numpy`, so a jax case added
# later cannot silently inherit float32. `test_factorization_core.py` carries the same guard
# for the same reason, found the same way.
try:
    import wavebit.on_jax                                    # noqa: E402,F401
except ImportError:
    pass


def _run_and_inputs(structure, thickness, n_top, n_bot, wl, period, fto, theta, phi, psi):
    # The constructor takes permittivity, `solve_2d` the index -- one pair of half-spaces
    # spelled in the quantity each side actually works in.
    solver = wavebit.rcwa(backend=0, top=n_top ** 2, bot=n_bot ** 2, theta=theta, phi=phi,
                          psi=psi, fto=fto, period=period, wavelength=wl,
                          thickness=thickness, structure=structure)
    via_api = solver.conv_solve()
    assert solver.grating_type_assigned == 2, solver.grating_type_assigned    # ensure 2D path
    epx, epy, epz = to_conv_mat_raster_sampled(
        solver.structure, solver.fto[0], solver.fto[1])
    kx, ky = solver.get_kx_ky_vector(wl)
    inputs = dict(wavelength=wl, kx=kx, ky=ky, n_top=n_top, n_bot=n_bot,
                  theta=solver.theta, phi=solver.phi,
                  incidence=solver._incidence_pair(), thickness=solver.thickness,
                  epx_conv_all=epx, epy_conv_all=epy, epz_conv_i_all=epz)
    return (np.array(via_api.R), np.array(via_api.T)), inputs


def _assert_same_answer(via_api, direct):
    """The API route and the direct core call agree — to a tolerance, not bit for bit.

    The property under test is that the mixin passes the right arguments, and every way of
    getting that wrong moves the efficiencies by far more than 1e-12. Exact equality would
    additionally assert that two independent runs of the same expressions land on the same last
    bit, which is a claim about the BLAS rather than about this code. Its SMM twin
    (test_scattering_core.py, where the full note lives) has failed that way intermittently on
    GitHub's runners, and only on the larger mode matrices — of which these 2D cases have the
    largest in the suite, so they were next in line.
    """
    for a, b in zip(via_api, direct):
        np.testing.assert_allclose(a, np.array(b), rtol=1e-12, atol=1e-15)


def _rand(shape, seed):
    """A random PERMITTIVITY raster -- an index draw, squared, so the layers below are the
    same ones this file measured before the material convention flipped."""
    return np.random.default_rng(seed).uniform(1.0, 4.0, size=shape) ** 2


CASES = [
    # name, structure, thickness, n_top, n_bot, wl, period, fto, theta, phi, psi
    # (half-spaces as indices; squared where they enter the constructor)
    ('normal', _rand((1, 5, 5), 0), [0.2], 1.0, 1.0, 0.9, [1.0, 1.0], [2, 2], 0.0, 0.0, 0.0),
    ('oblique_psi', _rand((1, 6, 6), 1), [0.3], 1.0, 1.5, 0.8, [1.0, 1.2], [2, 2], 0.3, 0.4, 0.9),
    ('two_layer', _rand((2, 4, 4), 2), [0.15, 0.22], 1.2, 1.0, 0.7, [0.9, 0.9], [2, 2], 0.2, 0.5, 1.1),
]


@pytest.mark.parametrize('case', CASES, ids=[c[0] for c in CASES])
def test_2d_wiring_matches_direct_core_call(case):
    _, structure, thickness, n_top, n_bot, wl, period, fto, theta, phi, psi = case
    (lri, lti), inp = _run_and_inputs(
        structure, thickness, n_top, n_bot, wl, period, fto, theta, phi, psi)
    cri, cti = solve_2d(**inp)
    _assert_same_answer((lri, lti), (cri, cti))


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
    _, structure, thickness, n_top, n_bot, wl, period, fto, theta, phi, psi = case
    (_, _), inp = _run_and_inputs(
        structure, thickness, n_top, n_bot, wl, period, fto, theta, phi, psi)
    ri0, ti0 = solve_2d(**inp)                               # numpy reference

    bk = dict(inp)
    for k in ('kx', 'ky', 'epx_conv_all', 'epy_conv_all', 'epz_conv_i_all'):
        bk[k] = _to(backend, inp[k])
    bk['theta'] = _to(backend, complex(inp['theta']))
    bk['incidence'] = tuple(_to(backend, complex(v)) for v in inp['incidence'])
    ri, ti = solve_2d(**bk)

    assert np.allclose(np.array(ri0), np.asarray(ri), atol=1e-10, rtol=0)
    assert np.allclose(np.array(ti0), np.asarray(ti), atol=1e-10, rtol=0)


def test_2d_gradient_jax_torch_fd():
    """d(sum T)/ds through the coupled 2D block chain: jax == torch == finite-diff."""
    jax = pytest.importorskip('jax')
    torch = pytest.importorskip('torch')
    import jax.numpy as jnp

    structure = _rand((1, 4, 4), 3)
    (_, _), inp = _run_and_inputs(structure, [0.15], 1.0, 1.0, 0.9, [1.0, 1.0], [1, 1], 0.2, 0.4, 0.9)
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


def test_python_scalar_incidence_keeps_double_precision_on_torch():
    """A Python-complex theta/psi must not inherit torch's default float32 dtype.

    ``torch.as_tensor(complex(...))`` silently yields complex64 when the default dtype is
    float32, which costs ~8 digits in the efficiency denominators and shows up as a
    cross-backend gradient mismatch. The core builds incidence scalars at its own working
    dtype instead, so passing them through raw is precision-safe either way.
    """
    torch = pytest.importorskip('torch')
    _, structure, thickness, n_top, n_bot, wl, period, fto, theta, phi, psi = CASES[1]
    (_, _), inp = _run_and_inputs(
        structure, thickness, n_top, n_bot, wl, period, fto, theta, phi, psi)
    ri0, ti0 = solve_2d(**inp)                                   # numpy, double

    default_before = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)                       # the hostile setting
    try:
        bk = dict(inp)
        for k in ('kx', 'ky', 'epx_conv_all', 'epy_conv_all', 'epz_conv_i_all'):
            bk[k] = torch.as_tensor(np.asarray(inp[k]))
        bk['theta'] = complex(inp['theta'])                      # raw Python scalars
        bk['incidence'] = tuple(complex(v) for v in inp['incidence'])
        ri, ti = solve_2d(**bk)
    finally:
        torch.set_default_dtype(default_before)

    assert np.allclose(np.array(ri0), np.asarray(ri), atol=1e-12, rtol=0)
    assert np.allclose(np.array(ti0), np.asarray(ti), atol=1e-12, rtol=0)
