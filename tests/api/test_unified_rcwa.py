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


@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_each_port_names_its_side_the_lit_side_and_its_medium(side):
    """R and T are incidence-relative; the references say which medium each one is in.

    On a stack whose half-spaces differ (an OLED: air one side, glass the other), "what reached
    the air" is R lit from one side and T from the other. The answer has to be readable from
    the result alone, without the problem that made it.
    """
    result = rcwa.run(slab(side=side), numerics=rcwa.Numerics())
    other = 'bottom' if side == 'top' else 'top'
    media = {'top': (2.25, 1.4), 'bottom': (3.24, 1.1)}          # slab()'s half-spaces
    for observation, port in ((result.reflection, side), (result.transmission, other)):
        reference = observation.reference
        assert reference['port'] == port
        assert reference['incident_port'] == side
        assert reference['outward_normal'] == (0., 0., 1. if port == 'top' else -1.)
        epsilon, mu = media[port]
        assert len(reference['epsilon']) == len(result.wavelengths) == 3
        np.testing.assert_array_equal(np.asarray(reference['epsilon'], dtype=complex), epsilon)
        np.testing.assert_array_equal(np.asarray(reference['mu'], dtype=complex), mu)


@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_amplitudes_are_linear_in_the_normalized_jones_pair(side):
    """Two runs, 's' and 'p', give every polarization's amplitudes (docs/guide/rcwa/light.md).

    A patterned layer at an oblique azimuth couples s and p, so each port's s and p amplitudes
    both carry a cross term the recipe has to reproduce. The source amplitude (V/m) does not
    enter the dimensionless amplitudes; summed efficiencies are not the pair's efficiency.
    """
    eps = np.array([[4.0, 1.0, 2.25, 1.0], [1.0, 9.0, 1.0, 4.0]])

    def run(polarization, amplitude=1.):
        problem = rcwa.Problem(
            stack=rcwa.Stack(period=(700., 700.), top=Material(1.), bottom=Material(2.25),
                             layers=(rcwa.Layer(thickness=200., epsilon=eps),
                                     rcwa.Layer(thickness=100., epsilon=eps))),
            excitation=rcwa.Excitation(wavelength=900., sources=(rcwa.PlaneWave(
                side=side, polarization=polarization, theta=.3, phi=.4, amplitude=amplitude),)))
        return rcwa.run(problem, numerics=rcwa.Numerics(orders=(2, 1)))

    pair = (0.7 - 0.2j, 0.4 - 1.3j)
    a_s, a_p = np.asarray(pair) / np.linalg.norm(pair)
    s, p, both, brighter = run('s'), run('p'), run(pair), run(pair, amplitude=3.)
    for port in ('reflection', 'transmission'):
        for name in ('amplitude_s', 'amplitude_p', 'cartesian'):
            got = np.asarray(getattr(getattr(both, port), name))
            np.testing.assert_allclose(got, a_s*np.asarray(getattr(getattr(s, port), name))
                                       + a_p*np.asarray(getattr(getattr(p, port), name)),
                                       atol=1e-14, rtol=0, err_msg=f'{port}.{name}')
            np.testing.assert_array_equal(np.asarray(getattr(getattr(brighter, port), name)), got)
    summed = abs(a_s)**2*np.asarray(s.R) + abs(a_p)**2*np.asarray(p.R)
    assert np.abs(summed - np.asarray(both.R)).max() > 1e-3


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


def patterned(geometry, period=(600., 600.)):
    """One patterned layer in nanometres, wavelength off every Rayleigh anomaly."""
    regions = tuple(rcwa.Region(geometry=g, material=Material(12.25))
                    for g in (geometry if isinstance(geometry, tuple) else (geometry,)))
    return rcwa.Problem(units=Units(length='nm'),
        stack=rcwa.Stack(period=period, bottom=Material(2.25),
                         layers=(rcwa.Layer(thickness=200., background=Material(1.), regions=regions),)),
        excitation=rcwa.Excitation(wavelength=850., sources=(rcwa.PlaneWave(polarization='s'),)))


PATTERNED_NUMERICS = rcwa.Numerics(orders=(2, 2), shape_nvf_grid_size=256)

# The lifecycle's stated scope (docs/design/unified-api.md section 7, docs/guide/common/autodiff.md):
# its topology guard reads concrete values. An eager jax.grad tracer carries one, as a PyTorch
# tensor does; under jit or vmap there is none, so that geometry is refused at trace
# construction rather than replayed against a record it may no longer fit.
TRACED_GEOMETRY = {
    'circle radius': (lambda v: patterned(rcwa.Circle(radius=v)), 160.,
                      'stack.layers[0].regions[0].geometry.radius'),
    'rectangle size': (lambda v: patterned(rcwa.Rectangle(size=(v, 200.))), 260.,
                       'stack.layers[0].regions[0].geometry.size'),
    'rectangle center': (lambda v: patterned(rcwa.Rectangle(size=(260., 200.), center=(v, 0.))), 10.,
                         'stack.layers[0].regions[0].geometry.center'),
    'rectangle angle': (lambda v: patterned(rcwa.Rectangle(size=(260., 200.), angle=v)), .1,
                        'stack.layers[0].regions[0].geometry.angle'),
    'period': (lambda v: patterned(rcwa.Circle(radius=150.), period=(v, 600.)), 600., 'stack.cell'),
}


@pytest.mark.parametrize('transform', ['jit', 'grad under jit', 'vmap'])
@pytest.mark.parametrize('case', sorted(TRACED_GEOMETRY))
def test_jax_traced_geometry_is_refused_by_a_prepared_run(case, transform):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    build, value, path = TRACED_GEOMETRY[case]
    with jax.enable_x64(True):
        prepared = rcwa.prepare(build(value), numerics=PATTERNED_NUMERICS, backend='jax')
        loss = lambda v: jnp.sum(prepared.run(build(v)).T)
        apply = {'jit': jax.jit(loss), 'grad under jit': jax.jit(jax.grad(loss)),
                 'vmap': jax.vmap(loss)}[transform]
        with pytest.raises(UnsupportedFeatureError) as error:
            apply(jnp.asarray([value, value]) if transform == 'vmap' else jnp.asarray(value))
    assert error.value.issue.code == 'prepared.host_topology'
    assert error.value.issue.path == path


@pytest.mark.parametrize('case', sorted(TRACED_GEOMETRY))
def test_jax_grad_differentiates_geometry_as_pytorch_does(case):
    # Outside jit the tracer's primal is read for the topology check; the run keeps the tangent.
    jax = pytest.importorskip('jax')
    torch = pytest.importorskip('torch')
    import jax.numpy as jnp
    build, value, _ = TRACED_GEOMETRY[case]
    at = 1.02*value
    with jax.enable_x64(True):
        prepared = rcwa.prepare(build(value), numerics=PATTERNED_NUMERICS, backend='jax')
        derivative = float(jax.grad(lambda v: jnp.sum(prepared.run(build(v)).T))(jnp.asarray(at)))
    prepared = rcwa.prepare(build(torch.tensor(value, dtype=torch.float64)),
                            numerics=PATTERNED_NUMERICS, backend='torch')
    live = torch.tensor(at, dtype=torch.float64, requires_grad=True)
    prepared.run(build(live)).T.sum().backward()
    # measured 4e-14 to 3e-13 relative; a lone rectangle's centre derivative is zero by symmetry
    assert derivative == pytest.approx(float(live.grad), rel=1e-10, abs=1e-14)
    if case in ('circle radius', 'rectangle size', 'rectangle angle'):
        concrete = rcwa.prepare(build(value), numerics=PATTERNED_NUMERICS)
        step = 1e-6 if case == 'rectangle angle' else 1e-3
        transmitted = lambda v: float(np.sum(concrete.run(build(v)).T))
        finite_difference = (transmitted(at+step) - transmitted(at-step))/(2*step)
        assert abs(finite_difference) > 1e-4
        # measured 8e-10 (radius), 4e-8 (width) and 8e-8 (angle)
        assert derivative == pytest.approx(finite_difference, rel=1e-6)


def test_torch_differentiates_a_shape_dimension_through_a_prepared_run():
    torch = pytest.importorskip('torch')
    build = lambda r: patterned(rcwa.Circle(radius=r))
    prepared = rcwa.prepare(build(torch.tensor(150., dtype=torch.float64)),
                            numerics=PATTERNED_NUMERICS, backend='torch')
    radius = torch.tensor(160., dtype=torch.float64, requires_grad=True)
    prepared.run(build(radius)).T.sum().backward()
    transmitted = lambda r: float(prepared.run(build(torch.tensor(r, dtype=torch.float64))).T.sum())
    step = 1e-3
    finite_difference = (transmitted(160.+step) - transmitted(160.-step))/(2*step)
    assert abs(finite_difference) > 1e-3
    assert float(radius.grad) == pytest.approx(finite_difference, rel=1e-7)


def test_torch_differentiates_a_tuple_period_like_a_stacked_one():
    # Validation reads a tuple's leaves one by one; np.asarray on the tuple itself
    # raised a bare RuntimeError on the grad-tracking component.
    torch = pytest.importorskip('torch')
    f64 = dict(dtype=torch.float64)
    as_tuple = lambda p: patterned(rcwa.Circle(radius=150.), period=(p, 600.))
    as_tensor = lambda p: patterned(rcwa.Circle(radius=150.), period=torch.stack([p, torch.tensor(600., **f64)]))
    gradients = []
    for build in (as_tuple, as_tensor):
        prepared = rcwa.prepare(build(torch.tensor(600., **f64)), numerics=PATTERNED_NUMERICS, backend='torch')
        period = torch.tensor(610., **f64, requires_grad=True)
        prepared.run(build(period)).T.sum().backward()
        gradients.append(float(period.grad))
    transmitted = lambda p: float(prepared.run(as_tensor(torch.tensor(p, **f64))).T.sum())
    step = 1e-3
    finite_difference = (transmitted(610.+step) - transmitted(610.-step))/(2*step)
    assert abs(finite_difference) > 1e-5
    assert gradients[0] == pytest.approx(gradients[1], rel=1e-12)
    assert gradients[0] == pytest.approx(finite_difference, rel=1e-7)


def test_torch_differentiates_a_jones_pair_tuple_like_a_stacked_one():
    # The same sequence read as the period, then a lowering that stacks the
    # components, since torch.asarray on the tuple would sever the graph.
    torch = pytest.importorskip('torch')
    import dataclasses
    f64 = dict(dtype=torch.float64)
    with_jones = lambda pair: dataclasses.replace(patterned(rcwa.Rectangle(size=(300., 150.))),
        excitation=rcwa.Excitation(wavelength=850., sources=(rcwa.PlaneWave(polarization=pair),)))
    as_tuple = lambda a: with_jones((1., a))
    as_tensor = lambda a: with_jones(torch.stack([torch.tensor(1., **f64), a]))
    gradients = []
    for build in (as_tuple, as_tensor):
        prepared = rcwa.prepare(build(torch.tensor(.3, **f64)), numerics=PATTERNED_NUMERICS, backend='torch')
        component = torch.tensor(.3, **f64, requires_grad=True)
        prepared.run(build(component)).T.sum().backward()
        gradients.append(float(component.grad))
    transmitted = lambda a: float(prepared.run(as_tensor(torch.tensor(a, **f64))).T.sum())
    step = 1e-4
    finite_difference = (transmitted(.3+step) - transmitted(.3-step))/(2*step)
    assert abs(finite_difference) > 1e-3
    assert gradients[0] == pytest.approx(gradients[1], rel=1e-12)
    assert gradients[0] == pytest.approx(finite_difference, rel=1e-6)


def test_torch_shape_step_across_a_topology_boundary_asks_for_a_new_preparation():
    torch = pytest.importorskip('torch')
    pair = lambda w: patterned(tuple(rcwa.Rectangle(size=(w, 200.), center=(c, 0.)) for c in (-60., 60.)))
    prepared = rcwa.prepare(pair(100.), numerics=PATTERNED_NUMERICS, backend='torch')
    width = torch.tensor(110., dtype=torch.float64, requires_grad=True)
    prepared.run(pair(width)).T.sum().backward()
    assert torch.isfinite(width.grad)
    with pytest.raises(IncompatibleProblemError) as error:
        prepared.run(pair(torch.tensor(130., dtype=torch.float64, requires_grad=True)))
    assert error.value.issue.code == 'prepared.topology'


def test_jax_grad_shape_step_across_a_topology_boundary_asks_for_a_new_preparation():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    pair = lambda w: patterned(tuple(rcwa.Rectangle(size=(w, 200.), center=(c, 0.)) for c in (-60., 60.)))
    with jax.enable_x64(True):
        prepared = rcwa.prepare(pair(100.), numerics=PATTERNED_NUMERICS, backend='jax')
        derivative = jax.grad(lambda w: jnp.sum(prepared.run(pair(w)).T))
        assert np.isfinite(float(derivative(jnp.asarray(110.))))
        with pytest.raises(IncompatibleProblemError) as error:
            derivative(jnp.asarray(130.))
    assert error.value.issue.code == 'prepared.topology'


#: One declaration of every region geometry the public namespace exports.
GEOMETRIES = {'Rectangle': dict(size=(.3, .2), angle=.3),
              'Circle': dict(radius=.15),
              'Ellipse': dict(radii=(.2, .1), angle=.4),
              'Polygon': dict(vertices=((-.2, -.2), (.2, -.2), (0., .2)))}


def test_every_exported_region_geometry_is_listed_here():
    from wavebit.rcwa import problem as declarations
    shapes = {name for name in rcwa.__all__
              if getattr(getattr(rcwa, name), '__module__', None) == declarations.__name__
              and name not in ('Problem', 'Stack', 'Layer', 'Region', 'Excitation', 'Outputs')}
    assert shapes == set(GEOMETRIES)


#: Exported geometries refused at their declaration, with the ``(code, path)`` they raise. Empty:
#: ``rcwa.Polygon`` was the one entry, refused from 0cd24a45 until a polygon region was built.
REFUSED_AT_DECLARATION = {}


@pytest.mark.parametrize('name', sorted(GEOMETRIES))
def test_a_region_geometry_is_built_or_refused_at_its_declaration(name):
    """No geometry is accepted when declared and refused only when the engine reaches it.

    ``rcwa.Polygon`` was: it lowered to a shape type no builder takes, ``check`` reported
    nothing, and ``run`` refused it from inside the engine. It was then refused at its
    declaration, and now it is built: ``check`` passes it and ``run`` solves it.
    """
    try:
        geometry = getattr(rcwa, name)(**GEOMETRIES[name])
    except UnsupportedFeatureError as error:
        assert REFUSED_AT_DECLARATION.get(name) == (error.issue.code, error.issue.path)
        return
    assert name not in REFUSED_AT_DECLARATION
    layer = rcwa.Layer(thickness=.2, background=Material(1.),
                       regions=(rcwa.Region(geometry=geometry, material=Material(4.)),))
    problem = rcwa.Problem(stack=rcwa.Stack(period=(1., 1.), layers=(layer,)),
                           excitation=rcwa.Excitation(wavelength=1.3, sources=(rcwa.PlaneWave(),)))
    numerics = rcwa.Numerics(orders=(2, 2))
    assert rcwa.check(problem, numerics=numerics).ok
    result = rcwa.run(problem, numerics=numerics)
    np.testing.assert_allclose(np.sum(result.R) + np.sum(result.T), 1., rtol=0, atol=1e-10)


# A layer whose regions cannot be built is refused while its geometry is frozen, on concrete
# values. `check` freezes nothing, so until 2026-10-15 it passed every case below, for every
# shape type, and `run` refused it: the shape of 0cd24a45, one level down. Each test asks
# `check` and `prepare` for the same issue, so a refusal reaching only one of them fails.

TENSOR = Material(np.diag([4., 4.5, 5.]))
#: Material and numerics of each build a region layer can take. Laurent shares the scalar NVF
#: layer checks; Li and the sampled tensor masks refuse with checks of their own.
BUILDS = {'scalar': (Material(4.), rcwa.Numerics(orders=(2, 2))),
          'tensor': (TENSOR, rcwa.Numerics(orders=(2, 2))),
          'scalar li': (Material(4.), rcwa.Numerics(orders=(2, 2), factorization='li')),
          'tensor li': (TENSOR, rcwa.Numerics(orders=(2, 2), factorization='li')),
          'tensor sampled': (TENSOR, rcwa.Numerics(orders=(2, 2), fourier_coefficients='sampled'))}


def one_layer(*geometries, build='scalar'):
    """A problem of ``geometries`` painted in order on one layer of a unit cell, and the
    numerics to check it with: both as ``build`` says."""
    material, numerics = BUILDS[build]
    regions = tuple(rcwa.Region(geometry=g, material=material) for g in geometries)
    layer = rcwa.Layer(thickness=.2, background=Material(1.), regions=regions)
    return rcwa.Problem(stack=rcwa.Stack(period=(1., 1.), layers=(layer,)),
                        excitation=rcwa.Excitation(wavelength=1.3, sources=(rcwa.PlaneWave(),))), numerics


def the_issue_check_and_prepare_report(problem, numerics):
    report = rcwa.check(problem, numerics=numerics)
    assert report.ok is False
    with pytest.raises(ValidationError) as refused:
        rcwa.prepare(problem, numerics=numerics)
    assert report.issues == (refused.value.issue,)
    return report.issues[0]


@pytest.mark.parametrize('geometry,build,says', [
    (rcwa.Rectangle(size=(1.3, .4)), 'scalar', 'is larger than the period and runs into its own image'),
    (rcwa.Circle(radius=.6), 'scalar', 'is larger than the period and runs into its own image'),
    (rcwa.Ellipse(radii=(.6, .2), angle=.1), 'scalar', 'is larger than the period and runs into its own image'),
    (rcwa.Rectangle(size=(1.3, .4)), 'tensor', 'is larger than the period and runs into its own image'),
    (rcwa.Circle(radius=.6), 'tensor', 'is larger than the period and runs into its own image'),
    (rcwa.Ellipse(radii=(.6, .2), angle=.1), 'tensor', 'is larger than the period and runs into its own image'),
    (rcwa.Rectangle(size=(1.3, .4)), 'scalar li', 'requires rectangles inside one cell'),
    (rcwa.Rectangle(size=(1.3, .4)), 'tensor li', 'requires rectangles inside one cell'),
], ids=['rectangle', 'circle', 'ellipse', 'tensor-rectangle', 'tensor-circle', 'tensor-ellipse',
        'li-rectangle', 'tensor-li-rectangle'])
def test_check_refuses_a_region_larger_than_the_period(geometry, build, says):
    issue = the_issue_check_and_prepare_report(*one_layer(geometry, build=build))
    assert issue.code == 'rcwa.invalid'
    assert says in issue.message


def test_check_refuses_a_polygon_reaching_into_its_own_image():
    bar = rcwa.Polygon(vertices=((-.65, -.1), (.65, -.1), (.65, .1), (-.65, .1)))
    issue = the_issue_check_and_prepare_report(*one_layer(bar))
    assert issue.code == 'rcwa.invalid'
    assert 'shape 0 (polygon) runs into its own periodic image' in issue.message


# Each pair overlaps only through the x edge: the first reaches past x = 0.5 into the second's
# image. A painted mix of types, since the sweep pairs primitives and outlines differently.
EDGE_PAIRS = {'rectangles': (rcwa.Rectangle(size=(.2, .2), center=(.45, 0.)),
                             rcwa.Rectangle(size=(.2, .2), center=(-.45, 0.))),
              'circle-ellipse': (rcwa.Circle(radius=.1, center=(.45, 0.)),
                                 rcwa.Ellipse(radii=(.1, .05), center=(-.45, 0.))),
              'polygon-rectangle': (rcwa.Polygon(vertices=((.35, -.1), (.55, -.1), (.55, .1), (.35, .1))),
                                    rcwa.Rectangle(size=(.2, .2), center=(-.45, 0.)))}


@pytest.mark.parametrize('pair,build', [('rectangles', 'scalar'), ('circle-ellipse', 'scalar'),
                                        ('polygon-rectangle', 'scalar'), ('rectangles', 'tensor'),
                                        ('circle-ellipse', 'tensor')])
def test_check_refuses_regions_that_overlap_only_across_a_cell_edge(pair, build):
    issue = the_issue_check_and_prepare_report(*one_layer(*EDGE_PAIRS[pair], build=build))
    assert issue.code == 'rcwa.invalid'
    assert 'overlap only across the x period boundary' in issue.message


@pytest.mark.parametrize('geometry,build,says', [
    (rcwa.Rectangle(size=(-.3, .2)), 'scalar', 'has lx = -0.3. A length here is a size'),
    (rcwa.Circle(radius=0.), 'scalar', 'has lx = 0. A length here is a size'),
    (rcwa.Ellipse(radii=(.2, -.1)), 'scalar', 'has ly = -0.2. A length here is a size'),
    (rcwa.Rectangle(size=(-.3, .2)), 'tensor', 'has lx = -0.3. A length here is a size'),
    (rcwa.Rectangle(size=(-.3, .2)), 'scalar li', 'requires positive finite sizes'),
    (rcwa.Rectangle(size=(-.3, .2)), 'tensor li', 'requires positive finite sizes'),
    (rcwa.Circle(radius=-.1), 'tensor sampled', 'shape lengths must be positive'),
], ids=['rectangle', 'circle', 'ellipse', 'tensor', 'li', 'tensor-li', 'tensor-sampled'])
def test_check_refuses_a_region_whose_size_is_not_positive(geometry, build, says):
    issue = the_issue_check_and_prepare_report(*one_layer(geometry, build=build))
    assert issue.code == 'rcwa.invalid'
    assert says in issue.message


def test_check_passes_what_only_touches_its_images():
    # The other side of each refusal above: the checks do not refuse a legal layer.
    legal = [one_layer(rcwa.Rectangle(size=(1., .4))),                    # fills the period
             one_layer(rcwa.Rectangle(size=(.2, .2), center=(.4, 0.)),    # abut across the edge
                       rcwa.Rectangle(size=(.2, .2), center=(-.4, 0.))),
             one_layer(rcwa.Polygon(vertices=((.3, -.1), (.7, -.1), (.7, .1), (.3, .1)))),  # past the edge
             one_layer(rcwa.Rectangle(size=(1.3, .4)), build='tensor sampled')]  # masks wrap
    for problem, numerics in legal:
        assert rcwa.check(problem, numerics=numerics).ok
        rcwa.prepare(problem, numerics=numerics)
