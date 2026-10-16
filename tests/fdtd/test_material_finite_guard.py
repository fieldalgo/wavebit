"""Traced material range doors retain constructor finiteness requirements."""
import numpy as np
import pytest

from wavebit.fdtd import _legacy as fd
from wavebit.fdtd._prepare.params import _material_door_guard
from wavebit.fdtd._runtime import runtime
from tests.fdtd.test_material_resolution_guard import backend_call, quiet, scene


KINDS = ('eps', 'mu', 'sigma', 'drude_frequency', 'drude_gamma',
         'lorentz_frequency', 'inactive_gamma', 'lorentz_strength', 'debye_tau',
         'debye_strength', 'diagonal_eps', 'diagonal_mu')
NONNEGATIVE = {'sigma', 'drude_gamma', 'inactive_gamma', 'lorentz_strength', 'debye_strength'}


def material(kind, x):
    if kind == 'eps':
        return fd.Material(eps=x)
    if kind == 'mu':
        return fd.Material(eps=1., mu=x)
    if kind == 'diagonal_eps':
        return fd.Material(eps=(1., 1., x))
    if kind == 'diagonal_mu':
        return fd.Material(eps=1., mu=(1., x, 1.))
    if kind == 'sigma':
        return fd.Conductor(eps=1., sigma=x)
    if kind.startswith('drude'):
        return fd.Dispersive(eps_inf=1., poles=(fd.Drude(
            omega_p=x if kind == 'drude_frequency' else 1.,
            gamma=x if kind == 'drude_gamma' else 0.),))
    if kind.startswith('debye'):
        return fd.Dispersive(eps_inf=1., poles=(fd.Debye(
            tau=x if kind == 'debye_tau' else 1.,
            delta_eps=x if kind == 'debye_strength' else 1.),))
    return fd.Dispersive(eps_inf=1., poles=(fd.Lorentz(
        omega_0=x if kind == 'lorentz_frequency' else 1.,
        gamma=x if kind == 'inactive_gamma' else 0.,
        strength=x if kind == 'lorentz_strength' else 0.),))


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('kind', KINDS)
def test_runtime_range_door_matches_eager_finiteness_and_zero_boundary(backend, kind):
    rt = runtime(backend)
    evaluate = backend_call(backend, lambda x: _material_door_guard(
        scene(material(kind, x), wavelength=None), rt, None))
    # Huge but finite values remain in the constructor's declared range. A CFL
    # or arithmetic representability check is separate; do not clip this door.
    for x in (float('nan'), float('inf'), -float('inf'), -1., -0., 0., 1e-300, 1., 1e308):
        expected = np.isfinite(x) and (x >= 0 if kind in NONNEGATIVE else x > 0)
        if expected:
            material(kind, x)
        else:
            with pytest.raises(ValueError, match='finite'):
                material(kind, x)
        assert bool(np.isfinite(evaluate(x))) == expected, (backend, kind, x)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('kind', ['sigma', 'inactive_gamma'])
def test_nonfinite_loss_is_refused_by_actual_lowering_guard_without_a_wavelength(backend, kind):
    seed = quiet(fd.plan, scene(material(kind, .1), wavelength=None))
    evaluate = backend_call(backend, lambda x: quiet(fd.lower,
        scene(material(kind, x), wavelength=None), seed, backend)['stable'])
    assert float(evaluate(.1)) == 1.
    assert np.isnan(evaluate(float('inf')))


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_admitted_conductivity_coefficients_and_derivative_are_unchanged(backend):
    seed = quiet(fd.plan, scene(material('sigma', .1), wavelength=None))
    def objective(x, engine):
        params = quiet(fd.lower, scene(material('sigma', x), wavelength=None), seed, engine)
        return params['be'].sum()
    value = .3
    expected = float(objective(value, 'numpy'))
    h = 1e-5
    derivative = (float(objective(value+h, 'numpy'))-float(objective(value-h, 'numpy')))/(2*h)
    if backend == 'jax':
        import jax
        runtime('jax')
        got, grad = jax.jit(jax.value_and_grad(lambda x: objective(x, 'jax')))(value)
    else:
        import torch
        runtime('torch')
        x = torch.tensor(value, dtype=torch.float64, requires_grad=True)
        got = objective(x, 'torch')
        grad, = torch.autograd.grad(got, x)
        got, grad = got.detach(), grad.detach()
    assert float(got) == pytest.approx(expected, rel=1e-12)
    assert derivative != 0
    assert float(grad) == pytest.approx(derivative, rel=2e-6)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('kind', ['eps', 'sigma', 'inactive_gamma'])
def test_overflowed_derived_parameter_does_not_pass_a_positive_range_check(backend, kind):
    rt = runtime(backend)
    evaluate = backend_call(backend, lambda x: _material_door_guard(
        scene(material(kind, x*x), wavelength=None), rt, None))
    with np.errstate(over='ignore'):
        finite = float(np.float64(1e150)*np.float64(1e150))
        overflowed = float(np.float64(1e155)*np.float64(1e155))
    material(kind, finite)
    with pytest.raises(ValueError, match='finite'):
        material(kind, overflowed)
    assert float(evaluate(1e150)) == 1.
    assert np.isnan(evaluate(1e155))
