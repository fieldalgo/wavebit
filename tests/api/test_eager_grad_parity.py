"""Eager ``jax.grad`` takes every branch eager PyTorch takes through the RCWA lifecycle.

Both hold concrete values in eager autodiff: a PyTorch tensor carries its own, and a
``jax.grad`` tracer carries its primal, which the lifecycle reads through
``jax.lax.stop_gradient``. Each case below ran on PyTorch and was refused, or reported
differently, under ``jax.grad`` until 2026-10-16. Under ``jax.jit`` and ``jax.vmap`` a tracer
carries no value, so the same cases stay refused there, with the codes pinned here.

Problems are small (orders (2, 2)) and off every Rayleigh anomaly unless the case is about one.
"""
import dataclasses

import numpy as np
import pytest

from wavebit import Material, Units, rcwa
from wavebit.errors import (IncompatibleProblemError, NumericalError, UnsupportedFeatureError,
                            ValidationError)

jax = pytest.importorskip('jax')
torch = pytest.importorskip('torch')
import jax.numpy as jnp  # noqa: E402

NUMERICS = rcwa.Numerics(orders=(2, 2))
TWO_WAVES = (rcwa.PlaneWave(theta=.23, amplitude=1.), rcwa.PlaneWave(theta=-.23, amplitude=1j))
BOUNDARY_POWER = rcwa.Outputs(diffraction=False, power_region='cell')


def patterned(eps=12.25, radius=150., thickness=200., wavelength=850., sources=None, outputs=None):
    """A circle in a 600 nm cell on n = 1.5; 850 nm is off its Rayleigh anomalies, 900 nm on one."""
    extra = {} if outputs is None else dict(outputs=outputs)
    region = rcwa.Region(geometry=rcwa.Circle(radius=radius), material=Material(eps))
    return rcwa.Problem(units=Units(length='nm'),
        stack=rcwa.Stack(period=(600., 600.), bottom=Material(2.25),
                         layers=(rcwa.Layer(thickness=thickness, background=Material(1.), regions=(region,)),)),
        excitation=rcwa.Excitation(wavelength=wavelength,
                                   sources=sources or (rcwa.PlaneWave(polarization='s'),)), **extra)


def raster(scale, pattern):
    pattern = torch.as_tensor(pattern) if isinstance(scale, torch.Tensor) else pattern
    return rcwa.Problem(units=Units(length='nm'),
        stack=rcwa.Stack(period=(600., 600.), bottom=Material(2.25),
                         layers=(rcwa.Layer(thickness=200., epsilon=pattern*scale),)),
        excitation=rcwa.Excitation(wavelength=850., sources=(rcwa.PlaneWave(polarization='s'),)))


def derivatives(build, prepared_at, at, loss, numerics, step):
    """The JAX ``jax.grad`` derivative, the PyTorch one, and a NumPy central difference."""
    with jax.enable_x64(True):
        on_jax = rcwa.prepare(build(prepared_at), numerics=numerics, backend='jax')
        through_jax = float(jax.grad(lambda v: loss(on_jax.run(build(v))))(jnp.asarray(at)))
    on_torch = rcwa.prepare(build(torch.tensor(prepared_at, dtype=torch.float64)),
                            numerics=numerics, backend='torch')
    live = torch.tensor(at, dtype=torch.float64, requires_grad=True)
    loss(on_torch.run(build(live))).backward()
    concrete = rcwa.prepare(build(prepared_at), numerics=numerics)
    value = lambda v: float(loss(concrete.run(build(v))))
    return through_jax, float(live.grad), (value(at+step) - value(at-step))/(2*step)


def refused_under_jit_and_vmap(build, prepared_at, at, loss, numerics):
    with jax.enable_x64(True):
        prepared = rcwa.prepare(build(prepared_at), numerics=numerics, backend='jax')
        objective = lambda v: loss(prepared.run(build(v)))
        codes = []
        for transformed, argument in ((jax.jit(jax.grad(objective)), jnp.asarray(at)),
                                      (jax.vmap(objective), jnp.asarray([at, at]))):
            with pytest.raises(UnsupportedFeatureError) as error:
                transformed(argument)
            codes.append(error.value.issue.code)
    return codes


# ------------------------------------------------------------------ coherent sources

def outgoing(result):
    return result.outputs['bottom'].outgoing.sum()


COHERENT = {
    'region permittivity': (lambda v: patterned(eps=v, sources=TWO_WAVES, outputs=BOUNDARY_POWER), 12.25, 12.5),
    'thickness': (lambda v: patterned(thickness=v, sources=TWO_WAVES, outputs=BOUNDARY_POWER), 200., 204.),
    'radius': (lambda v: patterned(radius=v, sources=TWO_WAVES, outputs=BOUNDARY_POWER), 150., 153.),
}


@pytest.mark.parametrize('variable', sorted(COHERENT))
def test_coherent_sources_differentiate_under_eager_jax_grad_as_on_pytorch(variable):
    build, prepared_at, at = COHERENT[variable]
    through_jax, through_torch, finite_difference = derivatives(build, prepared_at, at, outgoing,
                                                                NUMERICS, step=1e-4)
    assert abs(finite_difference) > 1e-20    # watts over one cell: small, and not zero
    # measured 5.6e-9 (permittivity), 1.6e-8 (thickness), 4.8e-8 (radius): the difference's own error
    assert through_jax == pytest.approx(finite_difference, rel=2e-7)
    # measured 1e-12 for thickness and radius. For the region permittivity PyTorch's derivative
    # sits 5.6e-8 from a Richardson extrapolation where JAX's sits 1.6e-10 (2026-10-16).
    assert through_jax == pytest.approx(through_torch, rel=1e-7)


def test_coherent_sources_stay_refused_under_jit_and_vmap():
    build, prepared_at, at = COHERENT['thickness']
    assert refused_under_jit_and_vmap(build, prepared_at, at, outgoing, NUMERICS) == ['rcwa.coherent_transform'] * 2


# ------------------------------------------------------------------ adaptive detuning

DETUNED = rcwa.Numerics(orders=(2, 2), detuning=1e-6)


def on_a_rayleigh_point(v):
    return patterned(thickness=v, wavelength=900.)


def test_adaptive_detuning_differentiates_under_eager_jax_grad_as_on_pytorch():
    loss = lambda result: result.T.sum()
    through_jax, through_torch, finite_difference = derivatives(on_a_rayleigh_point, 200., 204., loss,
                                                                DETUNED, step=1e-3)
    assert abs(finite_difference) > 1e-3
    assert through_jax == pytest.approx(through_torch, rel=1e-12)   # measured identical
    assert through_jax == pytest.approx(finite_difference, rel=1e-6)  # measured 2.0e-10


def test_adaptive_detuning_stays_refused_under_jit_and_vmap():
    codes = refused_under_jit_and_vmap(on_a_rayleigh_point, 200., 204., lambda r: r.T.sum(), DETUNED)
    assert codes == ['prepared.detuning_transform'] * 2


# ------------------------------------------------------------------ rasters with many phases

MANY_PHASES = 2. + np.random.default_rng(1).random((20, 20))     # 400 distinct values
RECORDING = {'integrated li': rcwa.Numerics(orders=(2, 2), fourier_coefficients='integrated', factorization='li'),
             'sampled nvf': rcwa.Numerics(orders=(2, 2), fourier_coefficients='sampled', factorization='nvf')}


@pytest.mark.parametrize('path', sorted(RECORDING))
def test_a_raster_with_more_than_256_phases_differentiates_under_eager_jax_grad(path):
    loss = lambda result: result.T.sum()
    through_jax, through_torch, finite_difference = derivatives(lambda s: raster(s, MANY_PHASES), 1., 1.01,
                                                                loss, RECORDING[path], step=1e-5)
    assert abs(finite_difference) > 1e-2
    assert through_jax == pytest.approx(through_torch, rel=1e-12)   # measured identical
    assert through_jax == pytest.approx(finite_difference, rel=1e-7)  # measured 1.5e-9 and 6e-10


def test_a_raster_with_more_than_256_phases_stays_refused_under_jit_and_vmap():
    codes = refused_under_jit_and_vmap(lambda s: raster(s, MANY_PHASES), 1., 1.01, lambda r: r.T.sum(),
                                       RECORDING['integrated li'])
    assert codes == ['rcwa.raster_partition_transform'] * 2


# ------------------------------------------------------------------ failures raise alike

def failure(backend, build, prepared_at, at, numerics, error_type, run_build=None):
    """The error an eager derivative step raises on ``backend``, for a failing live value.

    Prepared on ``build(prepared_at)``; the step runs ``run_build`` (default ``build``) at ``at``.
    """
    run_build = run_build or build
    with jax.enable_x64(True):
        if backend == 'jax':
            prepared = rcwa.prepare(build(prepared_at), numerics=numerics, backend='jax')
            with pytest.raises(error_type) as error:
                jax.grad(lambda v: prepared.run(run_build(v)).T.sum())(jnp.asarray(at))
        else:
            prepared = rcwa.prepare(build(torch.tensor(prepared_at, dtype=torch.float64)),
                                    numerics=numerics, backend='torch')
            with pytest.raises(error_type) as error:
                live = torch.tensor(at, dtype=torch.float64, requires_grad=True)
                prepared.run(run_build(live)).T.sum().backward()
    return error.value.issue.code, error.value.issue.path


def test_a_negative_thickness_raises_the_same_validation_error_on_both_backends():
    seen = {b: failure(b, lambda v: patterned(thickness=v), 200., -5., NUMERICS, ValidationError)
            for b in ('jax', 'torch')}
    assert seen['jax'] == seen['torch'] == ('layer.thickness', 'stack.layers[0].thickness')


def test_a_moved_raster_partition_raises_the_same_guard_on_both_backends():
    rng = np.random.default_rng(2)
    prepared_pattern = np.where(rng.random((8, 8)) > .5, 4., 1.)
    moved = prepared_pattern.copy()
    moved[0, 0] = 5. - moved[0, 0]
    seen = {b: failure(b, lambda s: raster(s, prepared_pattern), 1., 1.25, RECORDING['integrated li'],
                       IncompatibleProblemError, run_build=lambda s: raster(s, moved))
            for b in ('jax', 'torch')}
    assert seen['jax'] == seen['torch'] == ('prepared.raster_partition', 'problem')


def test_nonfinite_observables_raise_on_both_backends(monkeypatch):
    """No admitted problem is known to produce one, so the result's finiteness flag is poisoned."""
    from wavebit.rcwa import results
    original = results.pack

    def poisoned(*args, **kwargs):
        result = original(*args, **kwargs)
        checks = dict(result.validity.checks)
        checks['numerical_finite'] = checks['numerical_finite'] & (result.T.sum()*0 != 0)
        return dataclasses.replace(result, validity=dataclasses.replace(result.validity, checks=checks))

    monkeypatch.setattr(results, 'pack', poisoned)
    seen = {b: failure(b, lambda v: patterned(thickness=v), 200., 204., NUMERICS, NumericalError)
            for b in ('jax', 'torch')}
    assert seen['jax'] == seen['torch'] == ('rcwa.nonfinite', 'result')


# ------------------------------------------------------------------ the eager doors

def test_run_inside_eager_jax_grad_prepares_on_the_primal_as_pytorch_does():
    for keyword, at in (('thickness', 203.), ('radius', 153.)):
        with jax.enable_x64(True):
            through_jax = float(jax.grad(lambda v: rcwa.run(patterned(**{keyword: v}), numerics=NUMERICS,
                                                            backend='jax').T.sum())(jnp.asarray(at)))
            with pytest.raises(UnsupportedFeatureError) as refused:
                jax.jit(jax.grad(lambda v: rcwa.run(patterned(**{keyword: v}), numerics=NUMERICS,
                                                    backend='jax').T.sum()))(jnp.asarray(at))
        assert refused.value.issue.code == 'prepare.concrete'
        live = torch.tensor(at, dtype=torch.float64, requires_grad=True)
        rcwa.run(patterned(**{keyword: live}), numerics=NUMERICS, backend='torch').T.sum().backward()
        # measured identical for the thickness and 4e-15 for the radius
        assert through_jax == pytest.approx(float(live.grad), rel=1e-12)


def test_check_and_validate_report_inside_eager_jax_grad_and_refuse_under_jit():
    with jax.enable_x64(True):
        prepared = rcwa.prepare(patterned(), numerics=NUMERICS, backend='jax')
        reports = {}

        def objective(v):
            reports['check'] = rcwa.check(patterned(thickness=v), numerics=NUMERICS, backend='jax')
            reports['validate'] = prepared.validate(patterned(thickness=v))
            return v*v

        jax.grad(objective)(jnp.asarray(203.))
        assert reports['check'].ok and reports['validate'].ok
        jax.jit(objective)(jnp.asarray(203.))
    assert [i.code for i in reports['check'].issues] == ['check.concrete']
    assert [i.code for i in reports['validate'].issues] == ['validate.concrete']
