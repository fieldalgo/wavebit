"""Public mixed traced/concrete poles retain explicit arithmetic refusals."""
import numpy as np
import pytest

from wavebit import Units, fdtd
from wavebit.errors import ValidationError
from wavebit.materials import DispersiveMaterial, Drude, Lorentz
from wavebit.fdtd import _legacy as fd
from wavebit.fdtd._lower_public import lower_problem
from wavebit.fdtd._runtime import runtime
from tests.fdtd.test_material_resolution_guard import quiet


KINDS = ('lorentz-gamma', 'lorentz-strength', 'drude-gamma')


def problem(kind, value, frequency=2.):
    units = Units(length='m')
    scale = units.time_seconds
    if kind == 'drude-gamma':
        pole = Drude(omega_p=frequency/scale, gamma=value/scale)
    else:
        pole = Lorentz(omega_0=frequency/scale,
                       gamma=(value if kind == 'lorentz-gamma' else .5)/scale,
                       strength=(value if kind == 'lorentz-strength' else 1.)/scale**2)
    return fdtd.Problem(units=units, scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.4,)*3),
        boundaries=fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.Periodic(),
                                  z=fdtd.Periodic()),
        structures=(fdtd.Structure('block', fdtd.Region(size=(.2,)*3),
                                  DispersiveMaterial(1., (pole,))),)))


def numerics():
    return fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.1),
                         duration=fdtd.Duration(steps=2), courant=.2)


def transformed(backend, fn):
    if backend == 'jax':
        import jax
        return jax.jit(fn)(.5)
    import torch
    return fn(torch.tensor(.5, dtype=torch.float64, requires_grad=True))


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('door', ['check', 'prepare', 'run', 'validate', 'prepared-run'])
def test_public_mixed_poles_refuse_host_overflow_by_name(backend, kind, door):
    runtime(backend)
    num = numerics()
    prepared = (quiet(fdtd.prepare, problem(kind, .5), numerics=num, backend=backend)
                if door in ('validate', 'prepared-run') else None)

    def probe(value):
        bad = problem(kind, value, frequency=1e200)
        if door in ('check', 'validate'):
            report = (quiet(fdtd.check, bad, numerics=num, backend=backend)
                      if door == 'check' else quiet(prepared.validate, bad))
            assert not report.ok
            assert 'PoleArithmeticRefused' in str(report.issues)
        else:
            with pytest.raises(ValidationError, match='PoleArithmeticRefused'):
                if door == 'prepared-run':
                    quiet(prepared.run, bad)
                else:
                    quiet(getattr(fdtd, door), bad, numerics=num, backend=backend)
        return value*0

    failure = None
    try:
        transformed(backend, probe)
    except Exception as exc:
        failure = exc
    # An actual product regression must be reported as an assertion failure.
    assert failure is None, repr(failure)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('kind', KINDS)
def test_public_mixed_poles_keep_valid_coefficient_tangents(backend, kind):
    runtime(backend)
    num = numerics()
    prepared = quiet(fdtd.prepare, problem(kind, .5), numerics=num, backend=backend)
    dt = prepared._plan.dt
    gamma = .5
    strength = .5 if kind == 'lorentz-strength' else 4. if kind == 'drude-gamma' else 1.
    denominator = 1 + .5*gamma*dt
    expected = strength*dt**2/denominator
    derivative = (dt**2/denominator if kind == 'lorentz-strength'
                  else -.5*strength*dt**3/denominator**2)

    def drive(value):
        live = problem(kind, value)
        assert quiet(prepared.validate, live).ok
        scene, _ = lower_problem(live, num)
        return quiet(fd.lower, scene, prepared._plan, backend)['poles']['poles[0]']['drive'][0]

    if backend == 'jax':
        import jax
        value, tangent = jax.jit(jax.value_and_grad(drive))(.5)
        value, tangent = float(value), float(tangent)
    else:
        import torch
        x = torch.tensor(.5, dtype=torch.float64, requires_grad=True)
        result = drive(x)
        value = float(result.detach())
        tangent = float(torch.autograd.grad(result, x)[0])
    assert np.isfinite((value, tangent)).all()
    assert value == pytest.approx(expected, rel=1e-12, abs=0.)
    assert derivative != 0
    assert tangent == pytest.approx(derivative, rel=1e-12, abs=0.)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_public_traced_frequency_retains_runtime_invalidity(backend):
    from dataclasses import replace
    runtime(backend)

    def initial(x, y, z):
        return 1+0*x

    def live(frequency):
        return replace(problem('drude-gamma', .5, frequency=frequency),
                       initial=fdtd.InitialField(E=(None, initial, None)),
                       outputs=fdtd.Outputs(monitors=(
                           fdtd.Record('field', at=(0.,)*3, component='Ey'),)))

    prepared = quiet(fdtd.prepare, live(2.), numerics=numerics(), backend=backend)

    def run(frequency):
        result = quiet(prepared.run, live(frequency))
        return result['field'].values, result.validity.checks['runtime_stability']

    for frequency, expected_valid in ((2., True), (1e200, False)):
        failure = None
        try:
            if backend == 'jax':
                import jax
                values, valid = jax.jit(run)(frequency)
                values = np.asarray(values)
            else:
                import torch
                values, valid = run(torch.tensor(frequency, dtype=torch.float64,
                                                requires_grad=True))
                values = values.detach().numpy()
        except Exception as exc:
            failure = exc
        assert failure is None, repr(failure)
        assert bool(valid) == expected_valid
        if expected_valid:
            assert np.isfinite(values).all() and np.any(values != 0)
        else:
            assert np.isnan(values).all()
