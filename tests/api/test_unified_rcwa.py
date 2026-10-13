"""Independent physical and lifecycle contracts of the public immutable RCWA API."""
import os
from pathlib import Path
import subprocess
import tempfile
import sys

import numpy as np
import pytest

from wavebit import Material, Units, rcwa
from wavebit.errors import IncompatibleProblemError, UnsupportedFeatureError, ValidationError
from wavebit.materials import ConductiveMaterial
from wavebit.units import EPSILON0, C0, Z0


def slab(*, side='top', polarization='s', wavelength=(.65, .8, 1.), amplitude=1.,
         material=None, fields=False, units='um', scale=1.):
    return rcwa.Problem(
        units=Units(length=units),
        stack=rcwa.Stack(period=(.7*scale, .8*scale), top=Material(2.25, mu=1.4),
                         bottom=Material(3.24, mu=1.1),
                         layers=(rcwa.Layer(thickness=.18*scale,
                                           background=material or Material(4.2, mu=1.25)),)),
        excitation=rcwa.Excitation(wavelength=np.asarray(wavelength)*scale,
                                  sources=(rcwa.PlaneWave(side=side, polarization=polarization,
                                                         theta=.29, phi=.4, amplitude=amplitude),)),
        outputs=rcwa.Outputs(fields=fields))


def film_oracle(wavelengths, epsilon, mu, thickness, theta, polarization):
    """Tangential-field admittance Fresnel sum, independent of Wavebit engines."""
    transverse_squared = epsilon[0]*mu[0]*np.sin(theta)**2
    q = np.sqrt(np.asarray(epsilon)*np.asarray(mu)-transverse_squared+0j)
    admittance = q/np.asarray(mu) if polarization == 's' else np.asarray(epsilon)/q
    a, b, c = admittance
    r01, r12 = (a-b)/(a+b), (b-c)/(b+c)
    phase = np.exp(2j*np.pi*q[1]*thickness/np.asarray(wavelengths))
    denominator = 1+r01*r12*phase**2
    r = (r01+r12*phase**2)/denominator
    t = (2*a/(a+b))*(2*b/(b+c))*phase/denominator
    return abs(r)**2, (c.real/a.real)*abs(t)**2


@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('polarization', ['s', 'p'])
def test_magnetic_conical_slab_matches_independent_fresnel(side, polarization):
    problem = slab(side=side, polarization=polarization)
    result = rcwa.run(problem, numerics=rcwa.Numerics())
    eps, mu = ([2.25, 4.2, 3.24], [1.4, 1.25, 1.1])
    if side == 'bottom':
        eps, mu = eps[::-1], mu[::-1]
    expected_r, expected_t = film_oracle(problem.excitation.wavelength, eps, mu, .18, .29, polarization)
    np.testing.assert_allclose(result.R[:, 0, 0], expected_r, rtol=2e-12, atol=2e-14)
    np.testing.assert_allclose(result.T[:, 0, 0], expected_t, rtol=2e-12, atol=2e-14)
    np.testing.assert_allclose(result.absorption, 0., atol=3e-14)
    sign = 1 if side == 'top' else -1
    assert np.all(result.reflection.kz.real*sign > 0)
    assert np.all(result.transmission.kz.real*sign < 0)
    incident = .5/Z0*np.sqrt(eps[0]/mu[0])*np.cos(.29)*(.7*.8*1e-12)
    np.testing.assert_allclose(result.reflection.reference['incident_power_w'], incident, rtol=2e-13)
    np.testing.assert_allclose(result.reflection.power[:, 0, 0], incident*expected_r, rtol=3e-12)
    selected = result.transmission.at_wavelength(1)
    assert selected.reference['epsilon'] == (eps[2],)
    assert selected.k_parallel.shape == (1, 1, 1, 2)
    assert selected.reference['outward_normal'] == (0., 0., -float(sign))


def test_physical_amplitude_scales_power_and_si_fields_but_not_efficiency():
    one = slab(wavelength=.8, fields=True)
    scaled = one.replace(excitation=one.excitation.replace(sources=(one.excitation.sources[0].replace(amplitude=2j),)))
    a = rcwa.run(one, numerics=rcwa.Numerics())
    b = rcwa.run(scaled, numerics=rcwa.Numerics())
    np.testing.assert_allclose(b.R, a.R)
    np.testing.assert_allclose(b.reflection.power, 4*a.reflection.power, rtol=1e-12)
    fa = a.boundary_fields(side='top', x=[0.], y=[0.], part='incoming')
    fb = b.boundary_fields(side='top', x=[0.], y=[0.], part='incoming')
    np.testing.assert_allclose(fb.values, 2j*fa.values, rtol=2e-12, atol=1e-15)
    e, h = fa.values[..., :3], fa.values[..., 3:]
    np.testing.assert_allclose(np.linalg.norm(e, axis=-1), 1., rtol=2e-12)
    np.testing.assert_allclose(np.linalg.norm(h, axis=-1), np.sqrt(2.25/1.4)/Z0, rtol=2e-12)
    assert fa.units['H'] == 'A/m'


@pytest.mark.parametrize('unit,scale', [('nm', 1000.), ('m', 1e-6)])
def test_si_conductivity_and_physical_power_are_independent_of_length_unit(unit, scale):
    conductivity = 1800.
    epsilon = 2.5+1j*conductivity/(EPSILON0*(2*np.pi*C0/(.8e-6)))
    direct = rcwa.run(slab(wavelength=.8, material=Material(epsilon)), numerics=rcwa.Numerics())
    physical = rcwa.run(slab(wavelength=.8, material=ConductiveMaterial(2.5, conductivity), units=unit, scale=scale),
                        numerics=rcwa.Numerics())
    np.testing.assert_allclose(physical.R, direct.R, rtol=2e-12)
    np.testing.assert_allclose(physical.T, direct.T, rtol=2e-12)
    np.testing.assert_allclose(physical.transmission.power, direct.transmission.power, rtol=2e-12, atol=1e-30)
    assert physical.absorption[0] > 0


def test_prepared_live_coefficients_match_fresh_run_and_earlier_fields_stay_owned():
    original = slab(wavelength=.8, fields=True)
    prepared = rcwa.prepare(original, numerics=rcwa.Numerics())
    old_result = prepared.run(original)
    old_fields = old_result.field(x=[0.], y=[0.], z=[.09]).values.copy()
    newer = slab(wavelength=.9, material=Material(5.1, mu=1.25), fields=True)
    actual = prepared.run(newer)
    fresh = rcwa.run(newer, numerics=rcwa.Numerics())
    np.testing.assert_allclose(actual.T, fresh.T, atol=2e-14)
    np.testing.assert_array_equal(old_result.field(x=[0.], y=[0.], z=[.09]).values, old_fields)
    with pytest.raises(IncompatibleProblemError):
        prepared.run(newer.replace(outputs=rcwa.Outputs(fields=False)))


def test_validation_error_family_is_preserved_at_the_engine_boundary():
    problem = slab(wavelength=.8)
    invalid = problem.replace(stack=problem.stack.replace(top=Material([2., 3., 4.])))
    report = rcwa.check(invalid, numerics=rcwa.Numerics())
    assert not report.ok and report.issues
    with pytest.raises(ValidationError) as failure:
        rcwa.prepare(invalid, numerics=rcwa.Numerics())
    assert failure.value.code and failure.value.path


def test_check_does_not_run_scattering(monkeypatch):
    from wavebit.rcwa._backends.numpy_backend import RCWANumpy
    monkeypatch.setattr(RCWANumpy, 'solve', lambda *a, **k: pytest.fail('check executed the solver'))
    report = rcwa.check(slab(wavelength=.8), numerics=rcwa.Numerics())
    assert report.ok
    assert report.estimates['retained_spatial_modes'] == 1
    assert report.capabilities['physical_convergence'] is not True


def test_coherent_sources_require_explicit_boundary_power_output():
    problem = slab(wavelength=.8)
    source = problem.excitation.sources[0]
    problem = problem.replace(excitation=problem.excitation.replace(sources=(source, source.replace(side='bottom'))))
    with pytest.raises(UnsupportedFeatureError, match='diffraction'):
        rcwa.prepare(problem, numerics=rcwa.Numerics())
    result = rcwa.run(problem.replace(outputs=rcwa.Outputs(diffraction=False, power_region='cell')), numerics=rcwa.Numerics())
    assert set(result.outputs) >= {'top', 'bottom'}
    assert result.outputs['top'].units == 'W'
    with pytest.raises(UnsupportedFeatureError):
        _ = result.R


def test_jit_result_retains_spectral_calibration_and_marks_invalid_live_wavelength():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    with jax.enable_x64(True):
        problem = slab(wavelength=.8)
        prepared = rcwa.prepare(problem, numerics=rcwa.Numerics(), backend='jax')
        def evaluate(wavelength):
            return prepared.run(problem.replace(excitation=problem.excitation.replace(wavelength=wavelength)))
        result = jax.jit(evaluate)(jnp.asarray(.8))
        assert result.R.shape == (1, 1, 1)
        assert result.reflection.reference['incident_power_w'].shape == (1,)
        assert bool(result.validity.checks['positive_wavelength'])
        invalid = jax.jit(evaluate)(jnp.asarray(-.8))
        assert not bool(invalid.validity.checks['positive_wavelength'])
        assert np.isnan(invalid.R).all()
        derivative = jax.grad(lambda w: evaluate(w).R.sum())(jnp.asarray(.8))
        step = 1e-5
        finite_difference = (rcwa.run(slab(wavelength=.8+step), numerics=rcwa.Numerics()).R.sum()
                             - rcwa.run(slab(wavelength=.8-step), numerics=rcwa.Numerics()).R.sum())/(2*step)
        assert float(derivative) == pytest.approx(finite_difference, rel=3e-7)


def test_new_jax_single_solve_never_enables_x64_or_imports_torch():
    pytest.importorskip('jax')
    code = '''
import sys, jax
jax.config.update('jax_enable_x64', False)
from wavebit import Material, rcwa
p=rcwa.Problem(stack=rcwa.Stack(layers=(rcwa.Layer(thickness=.2,background=Material(2.25)),)),excitation=rcwa.Excitation(wavelength=1.,sources=(rcwa.PlaneWave(),)))
r=rcwa.run(p,numerics=rcwa.Numerics(),backend='jax',precision='single')
assert not jax.config.x64_enabled
assert str(r.R.dtype) == 'float32'
assert 'torch' not in sys.modules
'''
    import wavebit
    env = dict(os.environ, PYTHONPATH=str(Path(wavebit.__file__).parents[1]))
    subprocess.run([sys.executable, '-c', code], check=True, env=env, cwd=tempfile.gettempdir())


@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_empty_stack_is_the_single_interface_with_physical_port_labels(side):
    top, bottom = Material(2.25), Material(3.24)
    problem = rcwa.Problem(stack=rcwa.Stack(top=top, bottom=bottom),
                           excitation=rcwa.Excitation(wavelength=.8, sources=(rcwa.PlaneWave(side=side),)))
    result = rcwa.run(problem, numerics=rcwa.Numerics())
    expected = ((1.5-1.8)/(1.5+1.8))**2
    np.testing.assert_allclose(result.R, expected, atol=2e-14)
    np.testing.assert_allclose(result.T, 1-expected, atol=2e-14)
    assert result.reflection.reference['port'] == side
    assert result.reflection.reference['z'] == (0.,)


def test_negative_rectangular_period_is_refused_before_solving():
    problem = slab(wavelength=.8)
    invalid = problem.replace(stack=problem.stack.replace(period=(-1., 1.)))
    report = rcwa.check(invalid, numerics=rcwa.Numerics())
    assert not report.ok
    with pytest.raises(ValidationError):
        rcwa.prepare(invalid, numerics=rcwa.Numerics())


def test_explicit_magnetic_operators_match_direct_solver_and_refuse_stale_stack():
    from wavebit.rcwa import operators
    problem = slab(wavelength=.8)
    matrices = operators.assemble(problem, numerics=rcwa.Numerics())
    assert matrices.mu[0] is not None
    direct, assembled = rcwa.run(problem, numerics=rcwa.Numerics()), operators.solve(problem, matrices)
    np.testing.assert_allclose(assembled.R, direct.R, atol=2e-14)
    np.testing.assert_allclose(assembled.transmission.amplitude_s, direct.transmission.amplitude_s, atol=2e-14)
    assert assembled.provenance['constitutive_operators'] == 'explicit cell-centred operators'
    changed = problem.replace(stack=problem.stack.replace(layers=(problem.stack.layers[0].replace(thickness=.2),)))
    with pytest.raises(IncompatibleProblemError):
        operators.solve(changed, matrices)


def test_torch_compiled_public_lifecycle_preserves_eager_value_and_gradient():
    torch = pytest.importorskip('torch')
    problem = slab(wavelength=.8)
    prepared = rcwa.prepare(problem, numerics=rcwa.Numerics(), backend='torch')
    def loss(epsilon):
        layer = problem.stack.layers[0].replace(background=Material(epsilon, mu=1.25))
        live = problem.replace(stack=problem.stack.replace(layers=(layer,)))
        return prepared.run(live).R.sum()
    eager_x = torch.tensor(4.2, dtype=torch.float64, requires_grad=True)
    compiled_x = torch.tensor(4.2, dtype=torch.float64, requires_grad=True)
    expected, actual = loss(eager_x), torch.compile(loss)(compiled_x)
    eg, = torch.autograd.grad(expected, eager_x)
    cg, = torch.autograd.grad(actual, compiled_x)
    np.testing.assert_allclose(actual.detach().numpy(), expected.detach().numpy(), atol=2e-14)
    np.testing.assert_allclose(cg.detach().numpy(), eg.detach().numpy(), atol=2e-14)
    assert abs(float(eg)) > 1e-4


def test_assembled_operators_refuse_a_material_table_with_changed_wavelength_unit():
    from wavebit.materials import TabulatedMaterial
    from wavebit.rcwa import operators
    material = TabulatedMaterial((.5, 1.5), (2., 3.), wavelength_unit='nm')
    problem = rcwa.Problem(units=Units(length='nm'),
        stack=rcwa.Stack(period=(.8, .9), layers=(rcwa.Layer(thickness=.2, background=material),)),
        excitation=rcwa.Excitation(wavelength=1., sources=(rcwa.PlaneWave(polarization='s'),)))
    matrices = operators.assemble(problem, numerics=rcwa.Numerics())
    changed = TabulatedMaterial((.5, 1.5), (2., 3.), wavelength_unit='um')
    live = problem.replace(stack=problem.stack.replace(layers=(problem.stack.layers[0].replace(background=changed),)))
    with pytest.raises(IncompatibleProblemError):
        operators.solve(live, matrices)


def test_traced_invalid_thickness_masks_retained_fields_and_scattering_too():
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    problem = slab(wavelength=.8, fields=True)
    problem = problem.replace(outputs=rcwa.Outputs(fields=True, scattering=True))
    prepared = rcwa.prepare(problem, numerics=rcwa.Numerics(algorithm='smm'), backend='jax')
    def evaluate(thickness):
        live = problem.replace(stack=problem.stack.replace(layers=(problem.stack.layers[0].replace(thickness=thickness),)))
        result = prepared.run(live)
        return (result.validity.checks['nonnegative_thickness'], result.R,
                result.field(x=[0.], y=[0.], z=[.09]).values,
                result.boundary_fields(side='top', x=[0.], y=[0.]).values,
                tuple(getattr(result.scattering[0], key) for key in ('S11', 'S12', 'S21', 'S22')))
    calculate = jax.jit(evaluate)
    good = calculate(jnp.asarray(.18))
    bad = calculate(jnp.asarray(-.18))
    assert bool(good[0]) and not bool(bad[0])
    for value in (good[1], good[2], good[3], *good[4]):
        assert np.isfinite(np.asarray(value)).all()
    for value in (bad[1], bad[2], bad[3], *bad[4]):
        assert np.isnan(np.asarray(value)).all()


def test_prepared_refuses_wrong_problem_type_through_the_public_error_contract():
    prepared = rcwa.prepare(slab(wavelength=.8), numerics=rcwa.Numerics())
    report = prepared.validate(None)
    assert not report.ok
    assert report.issues[0].code == 'problem.type'
    with pytest.raises(ValidationError) as error:
        prepared.run(None)
    assert error.value.issue.code == 'problem.type'
