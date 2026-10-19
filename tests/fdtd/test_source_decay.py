"""Incident-column stopping: independent state replay and public prefix results.

The pulse/mesh is a fixed-grid time test, not a spatial accuracy certificate.
Each memory group retains its own peak, including round-off scattered fields.
Consequently the exactly layered TFSF example may legitimately reach its cap.
"""
import dataclasses
from fractions import Fraction
import math
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

from wavebit import Debye, DispersiveMaterial, Material, Units, fdtd
from wavebit.fdtd import _decay, _legacy as old
from wavebit.fdtd._kernel.sources import ColumnTerm, waveform_table
from wavebit.fdtd._kernel.step import initial_state, step
from wavebit.fdtd._lower_public import lower_problem
from wavebit.fdtd._runtime import runtime
from wavebit.units import C0


def column_case(kind='plane', *, steps=1800, decay=True):
    pulse = fdtd.GaussianPulse(wavelength=1., frequency_sigma_hz=.05*C0/1e-6)
    lateral = .05 if kind == 'plane' else .3
    source = (fdtd.PlaneWave('s', plane=fdtd.Plane(z=.6), pulse=pulse, polarization='p')
              if kind == 'plane' else
              fdtd.TFSF('s', box=fdtd.SnappedBox(size=(.1, .1, 1.2)), pulse=pulse))
    structures = ()
    if kind in ('layered', 'dispersive'):
        material = (Material(epsilon=2.25) if kind == 'layered' else
                    DispersiveMaterial(epsilon_inf=2., poles=(Debye(.3e-6/C0, .1),)))
        structures = (fdtd.Structure('film', shape=fdtd.SnappedSlab(z=(-.4, 0.)), material=material),)
    problem = fdtd.Problem(units=Units(length='um'), scene=fdtd.Scene(
        domain=fdtd.Domain(size=(lateral, lateral, 2.)),
        boundaries=fdtd.Boundaries.periodic(z=fdtd.Open()), structures=structures),
        sources=(source,), outputs=fdtd.Outputs(final_fields=True, monitors=(
            fdtd.TimeSeriesMonitor('probe', at=(0., 0., .2), component='Ex'),
            fdtd.FieldMonitor('spectrum', plane=fdtd.Plane(z=.2), wavelengths=(.9, 1., 1.1),
                       components=('Ex',)))))
    numerics = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.05),
        duration=fdtd.Duration(steps=steps),
        stopping=fdtd.DecayStop(threshold=.02, window_seconds=1e-6/C0) if decay else None)
    return problem, numerics


def quiet(call, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', old.CoarseMeshWarning)
        warnings.simplefilter('ignore', old.UnsettledFieldWarning)
        return call(*args, **kwargs)


def to_numpy(value):
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def near(actual, expected, *, scale=None):
    a, b = to_numpy(actual), to_numpy(expected)
    assert a.shape == b.shape
    # Inherited 128-epsilon prefix budget; explicitly scale tiny SI transforms.
    scale = max(1., float(np.max(np.abs(b), initial=0.))) if scale is None else scale
    assert float(np.max(np.abs(a-b), initial=0.)) <= 128*np.finfo(float).eps*scale


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
@pytest.mark.parametrize('kind', ['plane', 'tfsf', 'layered'])
@pytest.mark.parametrize('padding', ['pml', 'absorber'])
def test_public_column_decay_returns_the_executed_prefix(kind, backend, padding):
    if backend != 'numpy':
        module = pytest.importorskip(backend)
        if backend == 'jax':
            module.config.update('jax_enable_x64', True)
    p, n = column_case(kind)
    if padding == 'absorber':
        n = n.replace(absorber=fdtd.ConductiveAbsorber(cells=12))
    prepared = quiet(fdtd.prepare, p, numerics=n, backend=backend)
    result = quiet(prepared.run, p)
    stop = result.diagnostics['stopping']
    count = result.provenance['actual_steps']
    if padding == 'absorber':
        assert stop['reached'] and count < 1800
    if kind == 'layered':
        assert any(':drv:' in key for key in stop['groups'])
    if stop['reached']:
        assert all(Fraction(stop['window_max'][k]) <= Fraction(stop['threshold'])*
                   Fraction(stop['peaks'][k]) for k in stop['groups'])
    else:
        assert stop['reason'] == 'cap' and count == 1800
    fixed = quiet(fdtd.run, p, numerics=n.replace(decay=None,
        duration=fdtd.Duration(steps=count)), backend=backend)
    assert 'mem:s:col:E' in stop['groups'] and 'mem:s:col:H' in stop['groups']
    for component in ('E', 'H'):
        near(result.final_fields.values[component], fixed.final_fields.values[component])
    near(result['probe'].values, fixed['probe'].values)
    np.testing.assert_array_equal(result['probe'].time, fixed['probe'].time)
    for key in ('time', 'magnetic_time', 'incident', 'incident_H'):
        near(result.sources['s'][key], fixed.sources['s'][key])
        assert result.sources['s'][key].shape[0] == count
    near(result['spectrum'].values['Ex'], fixed['spectrum'].values['Ex'],
         scale=1e-6/C0)
    assert result['spectrum'].metadata == fixed['spectrum'].metadata
    assert result.final_fields.metadata == fixed.final_fields.metadata
    # Reuse keeps the requested cap; executing a prefix must not mutate the preparation.
    assert prepared.numerics.duration.steps == 1800


def prepared_state(kind='layered'):
    p, n = column_case(kind)
    scene, _ = lower_problem(p, n)
    plan = quiet(old.plan, scene)
    rt = runtime('numpy')
    params = quiet(old.lower, scene, plan, rt)
    params = dict(params, waveform=waveform_table(plan, params, rt))
    return scene, plan, params, rt, initial_state(plan, params, rt)


def independent_groups(state):
    """Read actual arrays directly, without production term/group enumeration."""
    arrays = {}

    def visit(node, prefix=''):
        for key in ('E', 'H'):
            arrays[prefix+key] = node[key]
        for name, value in node['psi'].items():
            arrays[prefix+'psi:'+name] = value
        for name, memory in node['mem'].items():
            stem = prefix+'mem:'+name+':'
            if 'col' in memory:
                visit(memory['col'], stem+'col:')
                if 'drv' in memory:
                    visit(memory['drv'], stem+'drv:')
            else:
                assert set(memory) == {'P', 'previous', 'delta'}
                for key, values in memory.items():
                    for i, value in enumerate(values):
                        arrays[f'{stem}{key}[{i}]'] = value
    visit(state)
    return arrays


def host_amplitudes(arrays):
    return {key: max(float(np.max(np.abs(value.real), initial=0.)),
                     float(np.max(np.abs(value.imag), initial=0.)))
            for key, value in arrays.items()}


def replay(kind):
    scene, plan, params, rt, state = prepared_state(kind)
    peaks = host_amplitudes(independent_groups(state))
    width_ratio = Fraction(scene.run.decay.window)/Fraction(plan.dt)
    width = (width_ratio.numerator+width_ratio.denominator-1)//width_ratio.denominator
    source = scene.sources[0]
    tau = 1/(2*math.pi*float(source.band.df))
    gate = math.nextafter(math.fsum((float(source.delay), float(source.band.sigmas)*tau,
        math.sqrt(-2*math.log(min(source.band.cutoff, scene.run.decay.threshold)))*tau)), math.inf)
    samples = []
    for index in range(plan.steps):
        state = step(plan, params, state, index, rt)
        row = host_amplitudes(independent_groups(state))
        samples.append(row)
        for key in peaks:
            peaks[key] = max(peaks[key], row[key])
        if (index+1) % width:
            continue
        if Fraction(index+1-width)*Fraction(plan.dt) < Fraction(gate):
            continue
        if not (peaks['E'] > 0 or peaks['H'] > 0):
            continue
        maxima = {k: max(r[k] for r in samples[-width:]) for k in peaks}
        if all(Fraction(maxima[k]) <= Fraction(scene.run.decay.threshold)*Fraction(peaks[k])
               for k in peaks):
            return index+1, peaks, samples
    return plan.steps, peaks, samples


@pytest.mark.parametrize('kind', ['plane', 'tfsf', 'layered'])
def test_column_stop_agrees_with_independent_state_replay(kind):
    expected, peaks, samples = replay(kind)
    p, n = column_case(kind)
    result = quiet(fdtd.run, p, numerics=n)
    stop = result.diagnostics['stopping']
    assert result.provenance['actual_steps'] == expected
    assert set(stop['groups']) == set(peaks)
    for key in peaks:
        assert stop['peaks'][key] == peaks[key]
    assert len(samples) == expected


@pytest.mark.parametrize('kind', ['plane', 'tfsf', 'layered'])
@pytest.mark.parametrize('padding', ['pml', 'absorber'])
def test_stopped_observables_agree_with_long_fixed_runs(kind, padding):
    p, n = column_case(kind)
    if padding == 'absorber':
        n = n.replace(absorber=fdtd.ConductiveAbsorber(cells=12))
    short = quiet(fdtd.run, p, numerics=n)
    long = quiet(fdtd.run, p, numerics=n.replace(decay=None,
        duration=fdtd.Duration(steps=5400)))
    count = short.provenance['actual_steps']
    trace = np.asarray(long['probe'].values)
    # The pre-registered 0.02 temporal observable budget is independent of the
    # state threshold's meaning. It is checked at each of the three wavelengths.
    assert np.max(np.abs(trace[count:])) <= .02*np.max(np.abs(trace))
    a, b = (np.asarray(r['spectrum'].values['Ex']) for r in (short, long))
    axes = tuple(range(1, b.ndim))
    assert np.all(np.max(np.abs(a-b), axis=axes) <= .02*np.max(np.abs(b), axis=axes))


@pytest.mark.parametrize('target', ['col:E', 'col:H', 'col:psi', 'drv:E', 'drv:H',
                                   'drv:psi', 'col:P', 'col:previous', 'col:delta'])
def test_every_incident_memory_can_hold_off_stopping(target):
    _, plan, _, _, state = prepared_state('dispersive')
    arrays = independent_groups(state)
    groups = _decay.state_groups(plan, state)
    assert set(groups) == set(arrays)
    assert all(groups[key] is value for key, value in arrays.items())
    # Reference histories and observation accumulators are not evolving drive state.
    state['mem']['s']['E'][:] = state['mem']['s']['H'][:] = 1e100
    role, kind = target.split(':')
    prefix = 'mem:s:'+role+':'
    if kind in ('E', 'H'):
        name = prefix+kind
    elif kind == 'psi':
        name = next(k for k in arrays if k.startswith(prefix+'psi:'))
    else:
        name = next(k for k in arrays if k.startswith(prefix+'mem:') and k.endswith(kind+'[0]'))
    initial = {key: 1. for key in groups}
    instrument = SimpleNamespace(dt=1., steps=10, skeleton='instrument',
                                decay=old.Decay(threshold=.1, window=1.))
    for value in arrays.values():
        value[:] = 0.
    arrays[name].flat[0] = .2
    tracker = _decay.Criterion(instrument, initial, 0.)
    assert not tracker.observe(host_amplitudes(_decay.state_groups(plan, state)), 1, 1)
    arrays[name].flat[0] = 0.
    assert tracker.observe(host_amplitudes(_decay.state_groups(plan, state)), 2, 1)


@pytest.mark.parametrize('role', ['col', 'drv'])
@pytest.mark.parametrize('fault', ['unknown-term', 'duplicate-term', 'extra-state',
                                  'missing-psi', 'extra-history'])
def test_unknown_column_schemas_are_refused(role, fault):
    scene, plan, _, _, state = prepared_state('dispersive')
    term = next(t for t in plan.terms if type(t) is ColumnTerm)
    sub = term.sub if role == 'col' else term.driver
    if fault in ('unknown-term', 'duplicate-term'):
        from wavebit.fdtd._kernel.poles import PoleTerm
        known = next(t for t in term.sub.terms if type(t) is PoleTerm)
        if fault == 'unknown-term':
            @dataclasses.dataclass(frozen=True)
            class Unknown(PoleTerm):
                pass
            terms = (Unknown(**vars(known)),)
        else:
            terms = (known, known)
        changed = dataclasses.replace(sub, terms=terms)
        term = dataclasses.replace(term, **{('sub' if role == 'col' else 'driver'): changed})
        bad = dataclasses.replace(plan, terms=tuple(
            term if type(t) is ColumnTerm else t for t in plan.terms))
        with pytest.raises(ValueError, match='DecayMemoryRefused'):
            _decay.validate_plan(scene, bad)
    else:
        mem = state['mem']['s']
        if fault == 'extra-state':
            mem[role]['unknown'] = np.zeros(1)
        elif fault == 'missing-psi':
            mem[role]['psi'].pop(next(iter(mem[role]['psi'])))
        else:
            mem['unknown'] = np.zeros(1)
        with pytest.raises(ValueError, match='DecayMemoryRefused'):
            _decay.state_groups(plan, state)


@pytest.mark.parametrize('kind', ['plane', 'tfsf'])
@pytest.mark.parametrize('mode', ['delay', 'jax', 'torch'])
def test_column_source_gate_and_concrete_execution_are_preserved(kind, mode):
    p, n = column_case(kind, steps=4)
    if mode == 'delay':
        delay = 100e-6/C0
        p = p.replace(sources=(p.sources[0].replace(delay=delay),))
        result = quiet(fdtd.run, p, numerics=n)
        stop = result.diagnostics['stopping']
        assert stop['eligible_time_seconds'] > delay
        # The Gaussian is not truncated: even this remote leading tail can be nonzero.
        assert stop['reason'] == 'cap' and not stop['reached']
        return
    module = pytest.importorskip(mode)
    if mode == 'jax':
        module.config.update('jax_enable_x64', True)
    prepared = quiet(fdtd.prepare, p, numerics=n, backend=mode)

    def evaluate(amplitude):
        live = p.replace(sources=(p.sources[0].replace(amplitude=amplitude),))
        return prepared.run(live).final_fields.values['E'].sum()

    with pytest.raises((ValueError, RuntimeError), match='DecayConcreteOnly'):
        if mode == 'jax':
            module.jit(module.grad(evaluate))(1.)
        else:
            evaluate(module.tensor(1., dtype=module.float64, requires_grad=True))
