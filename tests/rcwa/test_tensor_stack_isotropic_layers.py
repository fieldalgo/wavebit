"""An isotropic patterned layer keeps its scalar rule beside a tensor layer.

The factorization rule is chosen per layer: a tensor material in one layer must not move
another layer's answer at a fixed truncation order. Each case is checked against the same
layer in a stack with no tensor anywhere (the scalar path, which this change does not touch)
or against energy conservation, never against the tensor stack's own output.
"""

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa
import numpy as np
import pytest

import wavebit
from wavebit.rcwa._engine.geometry import StaleGeometryError


ANISOTROPIC = wavebit.Material([2.1, 2.5, 3.0])

SHAPES = {
    'circle': [dict(type='circle', lx=300., material=6.25)],
    'rectangle': [dict(type='rectangle', lx=300., ly=250., material=6.25)],
    'disjoint': [dict(type='rectangle', lx=200., ly=250., cx=-120., material=6.25),
                 dict(type='circle', lx=180., cx=150., material=4.)],
    'painted': [dict(type='rectangle', lx=250., ly=250., material=6.25),
                dict(type='rectangle', lx=200., ly=150., cx=100., cy=40., material=4.)],
}


def host(a):
    return np.asarray(a.detach().cpu() if hasattr(a, 'detach') else a)


def stack(second, shapes='circle', backend='numpy', order=(2, 2), first_background=1., **kw):
    """A patterned layer over a uniform one, lossless, at a fixed order."""
    layers = [dict(background=first_background, shapes=SHAPES[shapes]
                   if isinstance(shapes, str) else shapes),
              dict(background=second)]
    return _legacy_wavebit_rcwa(backend=backend, wavelength=730., period=(600., 600.),
                                thickness=[173., 90.], truncation_order=order,
                                theta=.23, phi=.19, structure=layers,
                                **{'shape_nvf_grid_size': 32, **kw})


def amplitudes(result):
    return [host(getattr(inc, name)) for inc in (result.s_inc, result.p_inc)
            for name in ('r_s', 'r_p', 't_s', 't_p')]


@pytest.mark.parametrize('shapes', list(SHAPES))
def test_the_patterned_layer_is_built_as_in_a_stack_without_a_tensor(shapes):
    # Same layer, same rule: its slots beside an anisotropic layer are those beside a scalar
    # one. On the tensor rule they differed by up to 0.19 (circle, orders 2 x 2).
    expected = stack(2.1, shapes).eps_conv_matrices()
    actual = stack(ANISOTROPIC, shapes).eps_conv_matrices()
    for a, b in zip(actual[:5], expected[:5]):
        np.testing.assert_allclose(host(a)[0], host(b)[0], rtol=0, atol=1e-14)


@pytest.mark.parametrize('second', [wavebit.Material([2.1] * 3), wavebit.Material(2.1 * np.eye(3))],
                         ids=['principal', 'matrix'])
@pytest.mark.parametrize('shapes', ['circle', 'painted'])
def test_a_scalar_multiple_of_the_identity_solves_as_the_scalar(second, shapes):
    # c*I is the scalar c: the whole stack must give the scalar stack's amplitudes. On the
    # tensor rule the circle's R moved from 0.6196 to 0.6677 at orders 3 x 3.
    expected = amplitudes(stack(2.1, shapes, order=(3, 3)).solve(auto_detune=False))
    actual = amplitudes(stack(second, shapes, order=(3, 3)).solve(auto_detune=False))
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)


def test_a_scalar_multiple_of_the_identity_in_the_patterned_layer_is_the_scalar():
    shapes = [dict(type='circle', lx=300., material=wavebit.Material(6.25 * np.eye(3)))]
    expected = amplitudes(stack(2.1, 'circle').solve(auto_detune=False))
    actual = amplitudes(stack(2.1, shapes, first_background=wavebit.Material([1.] * 3))
                        .solve(auto_detune=False))
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)


@pytest.mark.parametrize('shapes', list(SHAPES))
def test_a_lossless_stack_with_a_tensor_layer_conserves_energy(shapes):
    # Every material is real, so R + T = 1 at any order. The tensor rule on the isotropic
    # layer missed it by up to 1.7e-2 (rectangle, orders 3 x 3).
    result = stack(ANISOTROPIC, shapes, order=(3, 3)).solve(auto_detune=False)
    for inc in (result.s_inc, result.p_inc):
        assert float(np.sum(host(inc.R)) + np.sum(host(inc.T))) == pytest.approx(1., abs=1e-11)


def test_an_anisotropic_patterned_layer_keeps_the_tensor_rule():
    # The anisotropic layer is built as when it stands alone, by the tensor builder.
    crystal = wavebit.Material([2.25, 4., 2.8]).rotated(.31)
    shapes = [dict(type='circle', lx=300., material=crystal)]
    alone = _legacy_wavebit_rcwa(backend='numpy', wavelength=730., period=(600., 600.),
                                 thickness=[173.], truncation_order=(2, 2), shape_nvf_grid_size=32,
                                 structure=[dict(background=1., shapes=shapes)])
    beside = stack(2.1, shapes).eps_conv_matrices()
    for a, b in zip(beside, alone.eps_conv_matrices()):
        np.testing.assert_array_equal(host(a)[0], host(b)[0])


def test_the_permeability_of_an_isotropic_layer_is_built_as_in_a_scalar_stack():
    shapes = [dict(type='circle', lx=300., material=wavebit.Material(6.25, mu=1.3))]
    expected = stack(2.1, shapes).mu_conv_matrices()
    actual = stack(ANISOTROPIC, shapes).mu_conv_matrices()
    for a, b in zip(actual[:5], expected[:5]):
        np.testing.assert_allclose(host(a)[0], host(b)[0], rtol=0, atol=1e-14)


@pytest.mark.parametrize('shapes', list(SHAPES))
def test_a_frozen_record_replays_the_scalar_layer(shapes):
    solver = stack(ANISOTROPIC, shapes)
    geometry = solver.freeze_geometry()
    expected = amplitudes(stack(ANISOTROPIC, shapes).solve(auto_detune=False))
    for a, b in zip(amplitudes(solver.solve(geometry=geometry, auto_detune=False)), expected):
        np.testing.assert_array_equal(a, b)


def test_a_record_of_an_isotropic_layer_refuses_an_anisotropic_one():
    geometry = stack(ANISOTROPIC).freeze_geometry()
    shapes = [dict(type='circle', lx=300., material=wavebit.Material([6.25, 6.3, 6.25]))]
    with pytest.raises(StaleGeometryError, match='materials changed'):
        stack(ANISOTROPIC, shapes).eps_conv_matrices(geometry=geometry)
    geometry = stack(ANISOTROPIC, shapes).freeze_geometry()
    with pytest.raises(StaleGeometryError, match='materials changed'):
        stack(ANISOTROPIC).eps_conv_matrices(geometry=geometry)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_backends_agree_with_the_scalar_stack(backend):
    if backend == 'jax':
        pytest.importorskip('jax')
    else:
        pytest.importorskip('torch')
    expected = amplitudes(stack(wavebit.Material([2.1] * 3), 'circle').solve(auto_detune=False))
    actual = amplitudes(stack(wavebit.Material([2.1] * 3), 'circle', backend=backend)
                        .solve(auto_detune=False))
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-11)


def test_jit_replay_and_width_gradient_of_the_isotropic_layer():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp

    def shapes(lx):
        return [dict(type='circle', lx=lx, material=6.25)]
    geometry = stack(ANISOTROPIC, shapes(300.), backend='jax').freeze_geometry()

    def reflectance(lx):
        result = stack(ANISOTROPIC, shapes(lx), backend='jax').solve(
            geometry=geometry, auto_detune=False)
        return jnp.sum(result.s_inc.R)

    # Against the NumPy eager build of the same stack, and a central difference of it.
    eager = float(np.sum(stack(ANISOTROPIC, shapes(300.)).solve(auto_detune=False).s_inc.R))
    assert float(jax.jit(reflectance)(300.)) == pytest.approx(eager, abs=1e-12)
    h = 1e-3
    central = (float(np.sum(stack(ANISOTROPIC, shapes(300. + h)).solve(auto_detune=False).s_inc.R))
               - float(np.sum(stack(ANISOTROPIC, shapes(300. - h)).solve(auto_detune=False).s_inc.R))) / (2 * h)
    assert float(jax.jit(jax.grad(reflectance))(300.)) == pytest.approx(central, rel=1e-5, abs=1e-9)
