"""A traced ranged body (a Sphere radius) under ``checkpoint_steps``, eager autograd.

The prepared plan carries the checkpoint block as a static ``schedule``, which enters its
skeleton; the traced-geometry preflight (``shape_ranges.validate``) rebuilt the concrete
plan without it and refused the run as "a different scene". The checkpointed derivative
must agree with the plain scan's to rounding, and with NumPy central differences.
"""
import warnings

import numpy as np
import pytest

from wavebit import Material, Units, fdtd
from wavebit.fdtd.diagnostics import CoarseMeshWarning
from wavebit.units import C0

RADIUS = .12


def problem(radius=RADIUS):
    return fdtd.Problem(units=Units(length='um'), scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.8, .8, 1.2)),
        boundaries=fdtd.Boundaries.periodic(z=fdtd.Open()),
        structures=(fdtd.Structure('ball', shape=fdtd.Sphere(radius=radius, center=(.011, -.007, 0.)),
                                   material=Material(2.25)),)),
        sources=(fdtd.PlaneWave('pw', plane=fdtd.Plane(z=.42),
                                pulse=fdtd.GaussianPulse(wavelength=1., frequency_sigma_hz=.15 * C0 / 1e-6)),),
        outputs=fdtd.Outputs(monitors=(fdtd.DiffractionMonitor(
            'T', plane=fdtd.Plane(z=-.42), source='pw', wave='transmitted',
            wavelengths=(1.,), orders=(0, 0)),)))


def numerics(block):
    return fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.05), duration=fdtd.Duration(steps=120),
                         absorber=fdtd.PML(cells=4, wavelength=1., allow_thin=True),
                         checkpoint_steps=block,
                         geometry_ranges={'ball': ((-.16, .16), (-.16, .16), (-.16, .16))})


def prepared(block, backend):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', CoarseMeshWarning)
        return fdtd.prepare(problem(), numerics=numerics(block), backend=backend)


def value_and_gradient(block, backend):
    if backend == 'jax':
        jax = pytest.importorskip('jax')
        jax.config.update('jax_enable_x64', True)
    else:
        torch = pytest.importorskip('torch')
    runner = prepared(block, backend)

    def transmission(radius):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', CoarseMeshWarning)
            return runner.run(problem(radius))['T'].efficiency[0, 0, 0]

    if backend == 'jax':
        value, gradient = jax.value_and_grad(transmission)(jax.numpy.asarray(RADIUS))
        return float(value), float(gradient)
    radius = torch.tensor(RADIUS, dtype=torch.float64, requires_grad=True)
    value = transmission(radius)
    gradient, = torch.autograd.grad(value, radius)
    return float(value), float(gradient)


@pytest.fixture(scope='module')
def central_difference():
    runner, h = prepared(None, 'numpy'), 1e-4

    def transmission(radius):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', CoarseMeshWarning)
            return float(np.asarray(runner.run(problem(radius))['T'].efficiency[0, 0, 0]))

    return transmission(RADIUS), (transmission(RADIUS + h) - transmission(RADIUS - h)) / (2 * h)


@pytest.mark.parametrize('backend', ['torch', 'jax'])
def test_checkpointed_ranged_radius_derivative_matches_plain_scan(backend, central_difference):
    plain_value, plain = value_and_gradient(None, backend)
    reference_value, reference = central_difference
    assert plain_value == pytest.approx(reference_value, rel=1e-10)
    assert plain == pytest.approx(reference, rel=1e-4)
    for block in (7, 1000):
        value, gradient = value_and_gradient(block, backend)
        assert value == pytest.approx(plain_value, rel=1e-12)
        assert gradient == pytest.approx(plain, rel=1e-12)


@pytest.mark.parametrize('block', [None, 7])
def test_traced_preflight_keeps_refusing_a_plan_for_a_different_scene(block):
    # The rebuilt plan takes only the schedule from the caller's plan: any other static
    # difference -- here one more step -- is still "a different scene".
    torch = pytest.importorskip('torch')
    from wavebit.fdtd._prepare.shape_ranges import validate
    runner = prepared(block, 'torch')
    radius = torch.tensor(RADIUS, dtype=torch.float64, requires_grad=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', CoarseMeshWarning)
        scene, _ = runner._lower(problem(radius))
        plan = runner._plan
        assert plan.schedule == block
        validate(scene, plan)
        with pytest.raises(ValueError, match='plan belongs to a different scene'):
            validate(scene, plan.with_static(steps=plan.steps + 1))
