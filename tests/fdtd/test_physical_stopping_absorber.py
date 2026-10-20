"""SI stopping windows and outside graded padding preserve the concrete Yee solve."""
import math

import numpy as np
import pytest

from wavebit import ConductiveMaterial, Job, Units, fdtd
from wavebit.fdtd._lower_public import lower_problem
from wavebit.units import C0, EPSILON0, Z0


def scalar_case(length='um', scale=1.):
    problem = fdtd.Problem(units=Units(length=length),
        scene=fdtd.Scene(domain=fdtd.Domain(size=(.4*scale,)*3),
            boundaries=fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.Periodic()),
            background=ConductiveMaterial(epsilon=2., conductivity=4*EPSILON0*C0/1e-6)),
        initial=fdtd.InitialField(E=(1., None, None)),
        outputs=fdtd.Outputs(final_fields=True, monitors=(fdtd.TimeSeriesMonitor('E', at=(0., 0., 0.), component='Ex'),)))
    numerics = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.1*scale),
        duration=fdtd.Duration(steps=173), stopping=fdtd.DecayStop(threshold=.02, window_seconds=.19e-6/C0))
    return problem, numerics


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_physical_stopping_matches_an_independent_scalar_recurrence_and_prefix(backend):
    if backend != 'numpy':
        module = pytest.importorskip(backend)
        if backend == 'jax': module.config.update('jax_enable_x64', True)
    p, n = scalar_case()
    result = fdtd.run(p, numerics=n, backend=backend)
    dt = result.provenance['time_step_seconds']
    # The uniform periodic field has zero curl: one scalar CN conductor update.
    ratio = p.scene.background.conductivity * dt / (2*EPSILON0*2.)
    a = (1-ratio)/(1+ratio)
    width = math.ceil(n.stopping.window_seconds/dt)
    expected = next(end for end in range(width, 174, width)
                    if a**(end-width+1) <= n.stopping.threshold)
    assert result.provenance['actual_steps'] == expected < 173
    np.testing.assert_allclose(result.final_fields.values['E'][0], a**expected, rtol=3e-14)
    assert result.final_fields.metadata['E_time_seconds'] == pytest.approx(expected*dt)
    assert result.final_fields.metadata['H_time_seconds'] == pytest.approx((expected-.5)*dt)
    stopping = result.diagnostics['stopping']
    assert stopping['reason'] == 'decay' and stopping['reached']
    assert stopping['window_seconds'] == pytest.approx(n.stopping.window_seconds)
    fixed = fdtd.run(p, numerics=n.replace(duration=fdtd.Duration(steps=expected), stopping=None), backend=backend)
    for component in ('E', 'H'):
        np.testing.assert_array_equal(result.final_fields.values[component], fixed.final_fields.values[component])
    np.testing.assert_array_equal(result['E'].values, fixed['E'].values)
    np.testing.assert_array_equal(result['E'].times_seconds, fixed['E'].times_seconds)


def test_stopping_is_invariant_under_geometry_unit_changes_and_portable_jobs():
    p, n = scalar_case()
    a = fdtd.run(p, numerics=n)
    other, other_n = scalar_case('nm', 1000.)
    job = Job(solver='fdtd', problem=other, numerics=other_n)
    restored = Job.from_json(job.to_json())
    assert restored.key == job.key and restored.numerics.stopping == other_n.stopping
    b = fdtd.run(restored.problem, numerics=restored.numerics)
    assert a.provenance['actual_steps'] == b.provenance['actual_steps']
    np.testing.assert_allclose(a.final_fields.values['E'], b.final_fields.values['E'], rtol=5e-14)
    assert a.final_fields.metadata['E_time_seconds'] == pytest.approx(b.final_fields.metadata['E_time_seconds'], rel=1e-15)
    graded = job.replace(numerics=other_n.replace(absorber=fdtd.ConductiveAbsorber(cells=4)))
    assert Job.from_json(graded.to_json()).key == graded.key


def test_decay_keeps_the_cap_when_no_field_has_been_excited():
    p, n = scalar_case()
    result = fdtd.run(p.replace(initial=None), numerics=n)
    assert result.provenance['actual_steps'] == 173
    assert result.diagnostics['stopping']['reason'] == 'cap'


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_decay_never_detaches_an_initial_amplitude_derivative(backend):
    module = pytest.importorskip(backend)
    p, n = scalar_case()
    if backend == 'jax': module.config.update('jax_enable_x64', True)
    prepared = fdtd.prepare(p, numerics=n, backend=backend)
    def loss(amplitude):
        return prepared.run(p.replace(initial=p.initial.replace(amplitude=amplitude))).final_fields.values['E'].sum()
    with pytest.raises((ValueError, RuntimeError), match='DecayConcreteOnly'):
        if backend == 'jax': module.jit(module.grad(loss))(1.)
        else: loss(module.tensor(1., dtype=module.float64, requires_grad=True))


def test_graded_absorber_padding_matches_an_independently_declared_lossy_scene():
    from wavebit.fdtd import _legacy as old
    p = fdtd.Problem(units=Units(length='um'), scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.4, .4, 1.)), boundaries=fdtd.Boundaries.periodic(z=fdtd.Open()),
        background=ConductiveMaterial(epsilon=2., conductivity=.4*EPSILON0*C0/1e-6)),
        initial=fdtd.InitialField(E=(lambda x,y,z: np.exp(-(z/.2)**2), None, None)),
        outputs=fdtd.Outputs(final_fields=True))
    n = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.1), duration=fdtd.Duration(steps=30),
                      absorber=fdtd.ConductiveAbsorber(cells=4))
    manual = old.Scene(domain=old.Domain(size=(.4, .4, 1.8)),
        boundaries=old.Boundaries.periodic(z=old.Absorber(cells=4)),
        resolution=old.Resolution(dx=.1), run=old.Run(steps=30),
        background=old.Conductor(eps=2., sigma=.4),
        initial=old.InitialField(E=(lambda x,y,z: np.exp(-(z/.2)**2), None, None)))
    lowered, meta = lower_problem(p, n)
    assert lowered.domain == manual.domain
    assert meta['padding_cells']['z-'] == meta['padding_cells']['z+'] == 4
    a, b = fdtd.run(p, numerics=n), old.run(manual)
    np.testing.assert_allclose(a.final_fields.values['E'], b.final.E, rtol=2e-14, atol=1e-15)
    np.testing.assert_allclose(a.final_fields.values['H'], b.final.H/Z0, rtol=2e-14, atol=1e-17)
    # The same conductive boundary still cannot use the default PML.
    assert not fdtd.check(p, numerics=n.replace(absorber=fdtd.PML(cells=8, allow_thin=True))).ok
