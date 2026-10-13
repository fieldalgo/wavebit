"""Bounded automatic stopping: independent decisions and returned-prefix clocks.

Thresholds in these tests are explicit instruments, not optical error ceilings.
The scalar decay oracle uses the declared conductive update, and the window
oracle uses exact rational comparisons of supplied host amplitudes. The end-clock
instrument deliberately requests a fixed early boundary; it does not claim that
that driven scene has settled. Every new/changed test needs an actual production
mutation assertion witness before this feature can be accepted.
"""
import dataclasses
from fractions import Fraction
import math
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

import wavebit.fdtd._legacy as fd
from tests.fixtures.fdtd.fd_helpers import backends, to_numpy
from wavebit.fdtd import _decay
from wavebit.fdtd._kernel.poles import PoleTerm
from wavebit.fdtd._kernel.sources import DipoleTerm, waveform_table
from wavebit.fdtd._kernel.step import initial_state, step
from wavebit.fdtd._runtime import runtime


# Accepted after the six-row stopping calibration: exact prefix arrays and
# a worst closed-form relative error of 6.581e-15 support the inherited128eps
# round-off allowance. This scales an absolute allowance by max(1,
# peak(expected)); no relative claim is made for tiny arrays.
CANDIDATE_ROUND_OFF = 128 * np.finfo(float).eps
ROUND_OFF = CANDIDATE_ROUND_OFF


def roundoff_budget():
    assert ROUND_OFF is not None, 'Disclosed prefix/analytic calibration is required.'
    return ROUND_OFF


def quiet(call, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', fd.CoarseMeshWarning)
        warnings.simplefilter('ignore', fd.UnsettledFieldWarning)
        return call(*args, **kwargs)


def scalar_scene(*, steps=173, threshold=.02, window=.19, amplitude=1.,
                 sigma=4., complex_fields=False, decay=True, sources=(), monitors=()):
    periodic = fd.Axis1D(k=0.) if complex_fields else fd.Periodic()
    size = (.1, .1, .8) if complex_fields else (.4, .4, .4)
    criterion = fd.Decay(threshold=threshold, window=window) if decay else None
    return fd.Scene(domain=fd.Domain(size=size),
        boundaries=fd.Boundaries(x=periodic, y=periodic, z=fd.Periodic()),
        resolution=fd.Resolution(dx=.1),
        run=fd.Run(steps=steps, courant=.5, decay=criterion),
        background=fd.Conductor(eps=2., sigma=sigma),
        initial=fd.InitialField(E=(1., None, None), amplitude=amplitude),
        sources=sources, monitors=monitors)


def scalar_norm(value):
    """Independent componentwise amplitude; never square or complex abs."""
    value = complex(value)
    return max(abs(value.real), abs(value.imag))


def oracle_windows(initial, samples, *, width, threshold, dt=1., eligible=0.):
    """Return first eligible complete boundary, from supplied scalar histories.

    This deliberately does not use the production tracker, its window reduction,
    or its group selector. Fractions avoid a threshold-times-subnormal product.
    The dictionary names identify independent unit systems, not summable energy.
    """
    peaks = dict(initial)
    finite = all(math.isfinite(x) for x in peaks.values())
    seen = any(peaks[k] > 0 for k in ('E', 'H'))
    for start in range(0, len(samples), width):
        part = samples[start:start+width]
        maxima = {key: max(row[key] for row in part) for key in peaks}
        for row in part:
            finite &= all(math.isfinite(x) for x in row.values())
            seen |= any(row[k] > 0 for k in ('E', 'H'))
            for key in peaks:
                peaks[key] = max(peaks[key], row[key])
        if len(part) != width or start*dt < eligible or not finite or not seen:
            continue
        if all(peaks[key] == 0 or Fraction(maxima[key])/Fraction(peaks[key])
               <= Fraction(threshold) for key in peaks):
            return start+width
    return len(samples)


def near(actual, expected, *, budget=None):
    budget = roundoff_budget() if budget is None else budget
    actual, expected = to_numpy(actual), to_numpy(expected)
    assert actual.shape == expected.shape
    scale = max(float(np.max(np.abs(expected), initial=0.)), 1.)
    assert np.max(np.abs(actual-expected), initial=0.) <= budget*scale


SPEC_CASES = (
    ('threshold-zero', {'threshold': 0., 'window': 1.}, ValueError, 'threshold'),
    ('threshold-one', {'threshold': 1., 'window': 1.}, ValueError, 'threshold'),
    ('threshold-negative', {'threshold': -.1, 'window': 1.}, ValueError, 'threshold'),
    ('threshold-nan', {'threshold': math.nan, 'window': 1.}, ValueError, 'threshold'),
    ('threshold-inf', {'threshold': math.inf, 'window': 1.}, ValueError, 'threshold'),
    ('threshold-bool', {'threshold': True, 'window': 1.}, ValueError, 'threshold'),
    ('window-zero', {'threshold': .1, 'window': 0.}, ValueError, 'window'),
    ('window-negative', {'threshold': .1, 'window': -1.}, ValueError, 'window'),
    ('window-nan', {'threshold': .1, 'window': math.nan}, ValueError, 'window'),
    ('window-inf', {'threshold': .1, 'window': math.inf}, ValueError, 'window'),
    ('window-bool', {'threshold': .1, 'window': True}, ValueError, 'window'),
    ('missing-window', {'threshold': .1}, TypeError, 'window'),
    ('missing-threshold', {'window': 1.}, TypeError, 'threshold'),
    ('valid', {'threshold': .125, 'window': .3}, None, ''),
    ('valid-numpy', {'threshold': np.float64(.125), 'window': np.float64(.3)}, None, ''),
)


@pytest.mark.parametrize('case, values, exception, message', SPEC_CASES,
                         ids=[row[0] for row in SPEC_CASES])
def test_decay_requires_explicit_static_finite_amplitude_ratio_and_window(
        case, values, exception, message):
    if exception is not None:
        with pytest.raises(exception, match=message):
            fd.Decay(**values)
        return
    criterion = fd.Decay(**values)
    assert criterion.threshold == .125 and criterion.window == .3
    assert isinstance(criterion.threshold, float) and isinstance(criterion.window, float)
    assert criterion.replace(window=.5).window == .5
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        criterion.threshold = .2
    assert fd.Run(steps=5, decay=criterion).decay == criterion
    # A decay declaration supplements one hard cap, rather than replacing it.
    with pytest.raises(ValueError, match='length'):
        fd.Run(decay=criterion)


AMPLITUDE_CASES = (
    ('zero', (0., 0.), 0.),
    ('signs', (-3., 2.), 3.),
    ('complex-components', (3.+4.j, -5.+2.j), 5.),
    ('subnormal', (math.ulp(0.), -2*math.ulp(0.)), 2*math.ulp(0.)),
    ('small-normal', (1e-250, -2e-250), 2e-250),
    ('large-complex', (complex(1.7e308, 1.7e308),), 1.7e308),
    ('nan', (1., math.nan), math.nan),
    ('infinite', (1., math.inf), math.inf),
)


@pytest.mark.parametrize('backend', backends())
@pytest.mark.parametrize('mode', ('eager', 'compiled-reduction', 'actual-window'))
@pytest.mark.parametrize('case, values, expected', AMPLITUDE_CASES,
                         ids=[row[0] for row in AMPLITUDE_CASES])
def test_component_amplitude_keeps_subnormals_and_extreme_complex_values(
        backend, mode, case, values, expected, monkeypatch):
    # A private runtime prevents this synthetic kernel's compiled block from
    # entering the shared public executor cache.
    rt = type(runtime(backend))()
    dtype = 'complex128' if any(isinstance(x, complex) for x in values) else 'float64'
    array = rt.array(np.array(values), dtype)
    if mode == 'eager':
        got = float(to_numpy(_decay.amplitude(array, rt)))
    elif mode == 'compiled-reduction':
        def reduce_window(value):
            first = _decay.amplitude(value, rt)
            def body(maximum, n):
                sample = rt.xp.where(rt.xp.asarray(n) == 0, first, rt.array(0., 'float64'))
                return rt.positive_maximum(maximum, sample)
            return rt.scan(body, rt.array(0., 'float64'), 2, None)
        got = float(to_numpy(rt.jit(reduce_window)(array)))
    else:
        import importlib
        kernel = importlib.import_module('wavebit.fdtd._kernel.step')
        scene = scalar_scene(steps=2, threshold=.3141592653589793,
                             window=.04, complex_fields=dtype == 'complex128')
        plan = quiet(fd.plan, scene)
        assert _decay.window_steps(scene.run.decay.window, plan.dt) == 2
        params = quiet(fd.lower, scene, plan, rt)
        payload = np.zeros(plan.shape, dtype=dtype)
        payload.reshape(-1)[:len(values)] = values
        params = dict(params, metric_probe=rt.array(payload, dtype))
        def synthetic_step(plan, params, state, n, rt):
            value = rt.xp.where(rt.xp.asarray(n) == 0, params['metric_probe'],
                                rt.xp.zeros_like(params['metric_probe']))
            return dict(state, E=value)
        # This executes the production window aggregation and host receipt,
        # while prescribing states rather than advancing electromagnetic fields.
        monkeypatch.setattr(kernel, 'step', synthetic_step)
        actual, state, report = _decay.execute(scene, plan, params, rt)
        assert actual.steps == 2 and report['complete_windows'] == 1
        got = report['window_max']['E']
    if math.isnan(expected):
        assert not math.isfinite(got)
    else:
        assert got == expected


WINDOW_CASES = (
    'first-qualifying', 'initial-peak', 'running-peak', 'threshold-equality',
    'weak-memory', 'late-source', 'whole-window', 'partial-cap', 'all-zero',
    'constant-rotating-amplitude', 'nonfinite-initial', 'nonfinite-middle',
    'subnormal-ratio', 'rounded-ratio-boundary', 'zero-group',
)


def synthetic_history(case):
    initial = {'E': 1., 'H': 0., 'memory': 1e-200}
    e = [.4, .3, .08, .07, .006, .005, .004, .003]
    memory = [4e-201, 3e-201, 8e-202, 7e-202, 6e-203, 5e-203, 4e-203, 3e-203]
    width, threshold, eligible = 2, .1, 0.
    if case == 'initial-peak':
        e = [.08]*8
        memory = [0.]*8
    elif case == 'running-peak':
        e = [10., 8., .8, .7, .6, .5, .4, .3]
        memory = [0.]*8
    elif case == 'threshold-equality':
        e = [.1]*8
        memory = [0.]*8
    elif case == 'weak-memory':
        e = [.001]*8
        memory = [1e-200]*4 + [1e-203]*4
    elif case == 'late-source':
        eligible = 4.
        e, memory = [.001]*8, [0.]*8
    elif case == 'whole-window':
        eligible = 2.5
        e, memory = [.001]*8, [0.]*8
    elif case == 'partial-cap':
        e, memory = [1., 1., 1., 1., .001], [0.]*5
    elif case == 'all-zero':
        initial = {key: 0. for key in initial}
        e, memory = [0.]*8, [0.]*8
    elif case == 'constant-rotating-amplitude':
        e, memory = [1.]*8, [0.]*8
    elif case == 'nonfinite-initial':
        initial['memory'] = math.inf
        e, memory = [.001]*8, [0.]*8
    elif case == 'nonfinite-middle':
        e = [1., math.inf, .001, .001, .001, .001, .001, .001]
        memory = [0.]*8
    elif case == 'subnormal-ratio':
        initial['E'] = 8*math.ulp(0.)
        initial['memory'] = 0.
        e = [math.ulp(0.)]*8
        memory = [0.]*8
        threshold = .125
    elif case == 'rounded-ratio-boundary':
        initial = {'E': 3., 'H': 0., 'memory': 0.}
        threshold = math.ulp(0.)
        e, memory = [4*math.ulp(0.)]*8, [0.]*8
    elif case == 'zero-group':
        initial['memory'] = 0.
        memory = [0.]*8
    samples = [dict(E=a, H=0., memory=b) for a, b in zip(e, memory)]
    return initial, samples, width, threshold, eligible


@pytest.mark.parametrize('case', WINDOW_CASES)
def test_complete_window_tracker_matches_independent_scalar_history(case):
    initial, samples, width, threshold, eligible = synthetic_history(case)
    criterion = fd.Decay(threshold=threshold, window=float(width))
    plan = SimpleNamespace(dt=1., steps=len(samples), decay=criterion,
                           skeleton='synthetic-cap')
    tracker = _decay.Criterion(plan, initial, eligible)
    expected = oracle_windows(initial, samples, width=width, threshold=threshold,
                              eligible=eligible)
    actual, reached = len(samples), False
    for start in range(0, len(samples), width):
        part = samples[start:start+width]
        maxima = {key: max(row[key] for row in part) for key in initial}
        hit = tracker.observe(maxima, start_step=start, count=len(part))
        if hit:
            actual, reached = start+len(part), True
            break
    assert actual == expected
    report = tracker.report(actual, reached)
    assert report['steps'] == actual and report['cap_steps'] == len(samples)
    assert report['window_steps'] == width
    assert report['reached'] is reached
    if case in ('partial-cap', 'all-zero', 'constant-rotating-amplitude',
                'nonfinite-initial', 'nonfinite-middle', 'rounded-ratio-boundary'):
        assert reached is False
    if case.startswith('nonfinite-'):
        assert report['nonfinite'] is True
    if case == 'all-zero':
        assert report['field_seen'] is False
    assert _decay.window_steps(2.01, 1.) == 3
    assert _decay.window_steps(2., 1.) == 2


MEMORY_CASES = (
    'vacuum-source', 'cpml', 'voxel-two-regions', 'conductive-film',
    'conductive-stack', 'lorentz-film', 'debye-film', 'two-debye-film',
    'debye-sum-film', 'oscillator-sum-film', 'hybrid-film', 'constitutive', 'stencil-layout',
)


def memory_scene(case):
    debye = fd.Debye(tau=.2, delta_eps=2.)
    drude = fd.Drude(omega_p=2., gamma=.5)
    lorentz = fd.Lorentz(omega_0=3., gamma=.7, strength=2.)
    inactive = fd.Lorentz(omega_0=4., gamma=1.2, strength=0.)
    normal = fd.PML(cells=12, wavelength=4., stabilize=0.) if case == 'cpml' else fd.Periodic()
    lengths = (.8, .8, 4. if case == 'cpml' else 2.)
    structures, sources = (), ()
    if case == 'vacuum-source':
        sources = (fd.Dipole('source', center=(0., 0., 0.), component='Ex',
                            band=fd.Band(wavelength=4., df=.025)),)
    elif case == 'voxel-two-regions':
        material = fd.Dispersive(eps_inf=4., poles=(debye, drude, inactive))
        structures = tuple(fd.Structure(fd.Region(size=(.3, .4, .3), center=(0., 0., z)),
                                        material) for z in (-.4, .4))
    elif case == 'constitutive':
        lengths = (1.4, 1.4, 2.)
        structures = (fd.Structure(fd.Box(size=(.43, .47, .54),
                                          center=(.01, -.01, .03), angle=25.),
                                   fd.Material(eps=2.25)),)
    elif case == 'conductive-stack':
        structures = tuple(fd.Structure(fd.Box(size=(1.2, 1.2, .32), center=(0., 0., z)),
                                        fd.Conductor(eps=3., sigma=sigma))
                           for z, sigma in ((-.5, 2.), (.5, 0.)))
    elif case.endswith('-film'):
        materials = {
            'conductive-film': fd.Conductor(eps=4., sigma=2.),
            'lorentz-film': fd.Dispersive(eps_inf=4., poles=(lorentz,)),
            'debye-film': fd.Dispersive(eps_inf=4., poles=(debye,)),
            'two-debye-film': fd.Dispersive(eps_inf=4., poles=(debye, fd.Debye(tau=.2, delta_eps=0.))),
            'debye-sum-film': fd.Dispersive(eps_inf=4., poles=(debye, debye, fd.Debye(tau=.2, delta_eps=0.))),
            'oscillator-sum-film': fd.Dispersive(eps_inf=4., poles=(drude, lorentz, inactive)),
            'hybrid-film': fd.Dispersive(eps_inf=4., poles=(debye, drude, inactive)),
        }
        structures = (fd.Structure(fd.Box(size=(1.2, 1.2, .54), center=(0., 0., .03)),
                                   materials[case]),)
    return fd.Scene(domain=fd.Domain(size=lengths),
        boundaries=fd.Boundaries(x=fd.Periodic(), y=fd.Periodic(), z=normal),
        resolution=fd.Resolution(dx=.1), run=fd.Run(steps=3, courant=.2,
            decay=fd.Decay(threshold=.1, window=.1)), structures=structures, sources=sources)


@pytest.mark.parametrize('case', MEMORY_CASES)
def test_each_evolving_array_is_a_separate_group(case):
    scene = memory_scene('constitutive' if case == 'stencil-layout' else case)
    plan = quiet(fd.plan, scene)
    rt = runtime('numpy')
    params = quiet(fd.lower, scene, plan, rt)
    state = initial_state(plan, params, rt)
    if case == 'stencil-layout':
        # The stencil inherits the compact constitutive flux layout. Its
        # operator need not be assembled to test that existing layout's door.
        from wavebit.fdtd._kernel.stencil import StencilTerm
        plan = dataclasses.replace(plan, terms=tuple(StencilTerm(**{
            f.name: getattr(t, f.name) for f in dataclasses.fields(t)})
            for t in plan.terms))
    expected = []

    def mark(array):
        marker = float(len(expected)+1)
        answer = np.full(array.shape, marker, dtype=array.dtype)
        expected.append(marker)
        return answer

    state['E'], state['H'] = mark(state['E']), mark(state['H'])
    for name, array in state['psi'].items():
        state['psi'][name] = mark(array)
    for term in plan.terms:
        if isinstance(term, DipoleTerm):
            state['mem'][term.name] = {key: np.full_like(value, 1e100)
                                      for key, value in state['mem'][term.name].items()}
        elif isinstance(term, PoleTerm):
            memory = state['mem'][term.name]
            required = ('P', 'previous', 'delta') + (('aux',) if 'aux' in memory else ())
            assert set(memory) == set(required)
            for key in required:
                memory[key] = tuple(mark(array) for array in memory[key])
        else:
            # These fixtures' only remaining material terms are actual compact
            # constitutive D/B arrays; no size formula infers their payload.
            assert term in plan.constitutive
            state['mem'][term.name] = mark(state['mem'][term.name])
    state['energy'][:] = 1e100
    groups = _decay.state_groups(plan, state)
    assert list(groups)[:2] == ['E', 'H']
    got = sorted(float(np.max(np.real(value))) for value in groups.values())
    assert got == expected
    assert len(groups) == len(set(groups))
    if case == 'hybrid-film':
        assert len(groups) == 2 + 4*3
    if case == 'conductive-stack':
        assert len(groups) == 2 + 3*2, 'inactive reserved slots are still evolving arrays'
    if case == 'cpml':
        assert state['psi'] and len(groups) == 2+len(state['psi'])

@pytest.mark.parametrize('backend', backends())
@pytest.mark.parametrize('case', ('real', 'complex', 'small-amplitude', 'large-amplitude',
                                  'zero-field', 'cap-remainder', 'lossless'))
def test_uniform_conductor_stops_at_first_independently_predicted_window(backend, case):
    amplitude = {'small-amplitude': 1e-140, 'large-amplitude': 1e140,
                 'complex': 1.+.75j, 'zero-field': 0.}.get(case, 1.)
    scene = scalar_scene(amplitude=amplitude, complex_fields=case == 'complex',
                         sigma=0. if case == 'lossless' else 4.)
    plan = quiet(fd.plan, scene)
    ratio = Fraction(scene.run.decay.window)/Fraction(plan.dt)
    width = (ratio.numerator+ratio.denominator-1)//ratio.denominator
    # A spatially uniform E in a periodic uniform conductor has zero curl,
    # H=0, and this scalar trapezoidal update at every Yee component.
    sigma = 0. if case == 'lossless' else 4.
    a = (2.-.5*sigma*plan.dt)/(2.+.5*sigma*plan.dt)
    values = []
    current = amplitude
    for _ in range(plan.steps):
        current *= a
        values.append(dict(E=scalar_norm(current), H=0.))
    expected = oracle_windows(dict(E=scalar_norm(amplitude), H=0.), values,
                              width=width, threshold=scene.run.decay.threshold)
    if case == 'cap-remainder':
        # The next full window would first pass; merely one quiet final sample
        # is insufficient. The cap is explicit and is never rounded upward.
        assert expected < plan.steps and width > 1
        cap = expected-1
        scene = scene.replace(run=scene.run.replace(steps=cap))
        plan = quiet(fd.plan, scene)
        expected = cap
    got = quiet(fd.run, scene, backend)
    stopping = got.diagnostics['stopping']
    assert got.plan.steps == expected
    assert stopping['steps'] == expected and stopping['cap_steps'] == plan.steps
    assert stopping['reason'] == ('cap' if case in ('zero-field', 'cap-remainder', 'lossless') else 'decay')
    assert stopping['reached'] is (case not in ('zero-field', 'cap-remainder', 'lossless'))
    assert stopping['cap_skeleton'] == plan.skeleton
    assert got.report.steps == expected and got.report.duration == expected*plan.dt
    assert stopping['window_steps'] == width
    final = to_numpy(got.final.E)
    scale = scalar_norm(amplitude)
    if scale:
        # Divide before comparison, so tiny fields cannot pass via absolute atol.
        assert np.max(np.abs(final[0]/amplitude-a**expected)) <= roundoff_budget()
    else:
        assert np.count_nonzero(final) == 0
    assert np.count_nonzero(to_numpy(got.final.H)) == 0
    times, energy = got.diagnostics['interior_energy']
    assert np.array_equal(times, np.arange(1, expected)*plan.dt)
    assert energy.shape == (max(expected-1, 0),)
    if scale:
        # Independent stored field energy at the observer's completed integer
        # clock; loss does not change eps=2 into the conductive update factor.
        want = 2.*plan.cells*abs(complex(amplitude)/scale)**2*a**(2*np.arange(1, expected))
        assert np.max(np.abs(energy/scale/scale-want), initial=0.) <= roundoff_budget()*max(1., np.max(want, initial=0.))


def prefix_scene():
    band = fd.Band(wavelength=4., df=.025)
    return fd.Scene(domain=fd.Domain(size=(.8, .8, .8)),
        boundaries=fd.Boundaries(x=fd.Periodic(), y=fd.Periodic(), z=fd.Periodic()),
        resolution=fd.Resolution(dx=.1),
        run=fd.Run(steps=53, courant=.5, decay=fd.Decay(threshold=.1, window=.14)),
        background=fd.Conductor(eps=2., sigma=1.),
        initial=fd.InitialField(E=(lambda x, y, z: np.cos(2*math.pi*z/.8), None, None)),
        sources=(fd.Dipole('electric', center=(-.15, -.15, -.15), component='Ex', band=band),
                 fd.Dipole('magnetic', center=(-.15, .15, -.15), component='Hy', kind='magnetic', band=band)),
        monitors=(fd.Record('er', at=(.1, 0., .1), component='Ex'),
                  fd.Record('hr', at=(.1, 0., .1), component='Hy'),
                  fd.Field('field', plane=fd.Plane(z=.2), components=('Ex', 'Hy'), wavelengths=(4., 4.5)),
                  fd.Flux('flux', plane=fd.Plane(z=.2), wavelengths=(4., 4.5))))


@pytest.mark.parametrize('backend', backends())
@pytest.mark.parametrize('windows', (1, 3))
def test_prefix_outputs_use_actual_count_and_complete_energy_clock(backend, windows, monkeypatch):
    scene = prefix_scene()
    planned = quiet(fd.plan, scene)
    width = _decay.window_steps(scene.run.decay.window, planned.dt)
    count = windows*width
    assert count < planned.steps and count > 1
    # Isolate output-prefix bookkeeping from the convergence decision. The
    # separate scalar/public gate exercises the unmodified real criterion.
    def prescribed_stop(self, window_metrics, start_step, count):
        self.window_max = dict(window_metrics)
        self._incorporate(window_metrics)
        self.complete_windows += 1
        return start_step+count == windows*self.window_steps
    monkeypatch.setattr(_decay.Criterion, 'observe', prescribed_stop)
    got = quiet(fd.run, scene, backend)
    fixed_scene = scene.replace(run=fd.Run(steps=count, courant=scene.run.courant))
    fixed = quiet(fd.run, fixed_scene, backend)
    assert got.plan.steps == count and got.diagnostics['stopping']['reached'] is True
    for component in ('E', 'H'):
        near(getattr(got.final, component), getattr(fixed.final, component))
    for name in ('er', 'hr'):
        assert np.array_equal(got[name].t, fixed[name].t)
        assert got[name].values.shape == (count,)
        near(got[name].values, fixed[name].values)
    for name in ('electric', 'magnetic'):
        a, b = got.sources[name], fixed.sources[name]
        assert a.context.plan.steps == count
        for key in ('t', 't_field'):
            assert np.array_equal(getattr(a, key), getattr(b, key))
        for key in ('drive', 'field', 'work'):
            assert getattr(a, key).shape == (count,)
            near(getattr(a, key), getattr(b, key))
        near(a.power((4., 4.5)), b.power((4., 4.5)))
    for component in ('Ex', 'Hy'):
        near(got['field'].fields[component], fixed['field'].fields[component])
    near(got['flux'].flux, fixed['flux'].flux)

    # Independent whole-domain energy from the two H clocks, using actual
    # production steps but no energy observer output as the oracle.
    rt = runtime(backend)
    p = quiet(fd.plan, fixed_scene)
    params = quiet(fd.lower, fixed_scene, p, rt)
    params = dict(params, waveform=waveform_table(p, params, rt))
    state, expected = initial_state(p, params, rt), []
    for n in range(count):
        old_e, old_h = to_numpy(state['E']).copy(), to_numpy(state['H']).copy()
        state = step(p, params, state, n, rt)
        if n:
            new_h = to_numpy(state['H'])
            expected.append(2.*np.sum(np.abs(old_e)**2)
                            + np.sum(np.real(np.conj(old_h)*new_h)))
    times, energy = got.diagnostics['interior_energy']
    assert np.array_equal(times, np.arange(1, count)*p.dt)
    near(energy, np.asarray(expected))
    assert energy.shape == (count-1,), 'pending electric-only T^N must be discarded'


@pytest.mark.parametrize('threshold', (.1, .001, 1e-7))
@pytest.mark.parametrize('component', ('Ex', 'Hy'))
def test_source_eligibility_uses_every_gaussian_tail_and_conservative_clock(threshold, component):
    sources = (
        fd.Dipole('early', center=(0., 0., 0.), component=component,
                  kind='magnetic' if component == 'Hy' else 'electric',
                  band=fd.Band(wavelength=4., df=.025, cutoff=.001), delay=0.),
        fd.Dipole('late', center=(0., 0., 0.), component=component,
                  kind='magnetic' if component == 'Hy' else 'electric',
                  band=fd.Band(wavelength=4., df=.04, cutoff=.01), delay=39.),
    )
    scene = scalar_scene(threshold=threshold, sources=sources)
    individual = []
    for source in sources:
        band = source.band
        tau = 1./(2.*math.pi*band.df)
        center = source.delay+math.sqrt(-2.*math.log(band.cutoff))*tau
        tail = math.sqrt(-2.*math.log(min(band.cutoff, threshold)))*tau
        individual.append(center+tail)
    expected = max(individual)
    got = _decay.source_gate(scene, threshold)
    assert got >= expected-4*math.ulp(expected)
    assert abs(got-expected) <= 8*math.ulp(expected)
    assert individual[1] > individual[0], 'the late-source fixture must set the bound'
    assert _decay.source_gate(scene.replace(sources=()), threshold) == 0.
    # The Gaussian remains unchanged and merely becomes eligible; no hard
    # truncation or mutation of its documented Band.cutoff is introduced.
    for source in sources:
        tau = 1./(2.*math.pi*source.band.df)
        center = source.delay+source.band.sigmas*tau
        envelope = math.exp(-.5*((got-center)/tau)**2)
        assert 0. < envelope <= min(source.band.cutoff, threshold)*(1.+32*np.finfo(float).eps)
    width, dt = 3, .5
    first_start = math.ceil(got/(width*dt))*width
    plan = SimpleNamespace(dt=dt, steps=first_start+2*width,
                           decay=fd.Decay(threshold=threshold, window=width*dt),
                           skeleton='synthetic-source-cap')
    tracker = _decay.Criterion(plan, {'E': 1., 'H': 0.}, got)
    for start in range(0, first_start, width):
        assert not tracker.observe({'E': 0., 'H': 0.}, start, width)
    assert tracker.observe({'E': 0., 'H': 0.}, first_start, width)

REFUSAL_CASES = (
    'plane-wave', 'tfsf', 'unknown-runtime', 'torch-device', 'unknown-term',
    'missing-memory', 'extra-pole-slot', 'unknown-observer', 'private-fixed-execute',
    'torch-reverse', 'torch-forward', 'torch-jvp', 'torch-vmap', 'torch-callable-dual',
    'jax-grad', 'jax-jvp', 'jax-jit', 'jax-vmap', 'jax-jit-captured', 'jax-callable',
)


@pytest.mark.parametrize('case', REFUSAL_CASES)
def test_decay_refuses_uncovered_sources_memory_backends_and_transforms(case):
    scene = scalar_scene(steps=2)
    if case in ('plane-wave', 'tfsf'):
        band = fd.Band(wavelength=4., df=.025)
        if case == 'plane-wave':
            source = fd.PlaneWave('pw', z=0., band=band)
        else:
            source = fd.TFSF('tf', box=fd.Region(size=(.2, .2, .2)), band=band)
        with pytest.raises(ValueError, match='DecaySourceRefused'):
            fd.plan(scene.replace(sources=(source,)))
        return
    if case == 'unknown-runtime':
        with pytest.raises(ValueError, match='DecayBackendRefused'):
            _decay.validate_runtime(SimpleNamespace(name='numpy'))
        return
    if case == 'torch-device':
        pytest.importorskip('torch')
        rt = runtime('torch')
        # Inspect the device door without allocating on a GPU or needing one.
        other = object.__new__(type(rt))
        other.device = SimpleNamespace(type='cuda')
        with pytest.raises(ValueError, match='DecayDeviceRefused'):
            _decay.validate_runtime(other)
        return
    if case in ('unknown-term', 'missing-memory', 'extra-pole-slot', 'unknown-observer'):
        scene = memory_scene('voxel-two-regions')
        plan = quiet(fd.plan, scene)
        rt = runtime('numpy')
        params = quiet(fd.lower, scene, plan, rt)
        state = initial_state(plan, params, rt)
        if case == 'unknown-term':
            @dataclasses.dataclass(frozen=True)
            class UnknownPole(PoleTerm):
                pass
            term = plan.terms[0]
            bad = UnknownPole(term.name, term.region, term.slabs, term.n_poles)
            plan = dataclasses.replace(plan, terms=(bad,)+plan.terms[1:])
            with pytest.raises(ValueError, match='DecayMemoryRefused'):
                _decay.validate_plan(scene, plan)
        elif case == 'unknown-observer':
            from wavebit.fdtd._kernel.observe import PointRecord
            @dataclasses.dataclass(frozen=True)
            class UnknownRecord(PointRecord):
                pass
            plan = dataclasses.replace(plan, observers=(UnknownRecord('bad', 'E', 0, 0, 0, 0),))
            with pytest.raises(ValueError, match='DecayObserverRefused'):
                _decay.validate_plan(scene, plan)
        else:
            name = plan.terms[0].name
            if case == 'missing-memory':
                del state['mem'][name]
            else:
                memory = state['mem'][name]
                memory['P'] += (memory['P'][0],)
            with pytest.raises(ValueError, match='DecayMemoryRefused'):
                _decay.state_groups(plan, state)
        return
    if case == 'private-fixed-execute':
        from wavebit.fdtd._execution import execute
        plan = quiet(fd.plan, scene)
        rt = runtime('numpy')
        params = quiet(fd.lower, scene, plan, rt)
        with pytest.raises(ValueError, match='DecayExecutionRefused'):
            execute(plan, params, rt)
        return
    if case.startswith('torch-'):
        torch = pytest.importorskip('torch')
        x = torch.tensor(1., dtype=torch.float64)
        def evaluate(value):
            initial = (fd.InitialField(E=(lambda x, y, z: value, None, None))
                       if case == 'torch-callable-dual' else scene.initial.replace(amplitude=value))
            return fd.run(scene.replace(initial=initial),
                          'numpy' if case == 'torch-callable-dual' else 'torch').final.E.real.sum()
        with pytest.raises(ValueError, match='DecayConcreteOnly'):
            if case == 'torch-reverse':
                evaluate(x.requires_grad_())
            elif case in ('torch-forward', 'torch-callable-dual'):
                from torch.autograd import forward_ad
                with forward_ad.dual_level():
                    evaluate(forward_ad.make_dual(x, torch.ones_like(x)))
            elif case == 'torch-jvp':
                torch.func.jvp(evaluate, (x,), (torch.ones_like(x),))
            else:
                torch.func.vmap(evaluate)(torch.ones(2, dtype=torch.float64))
        return
    jax = pytest.importorskip('jax')
    with jax.enable_x64(True):
        x = jax.numpy.asarray(1., dtype=jax.numpy.float64)
        def evaluate(value):
            initial = (fd.InitialField(E=(lambda x, y, z: value, None, None))
                       if case == 'jax-callable' else scene.initial.replace(amplitude=value))
            return fd.run(scene.replace(initial=initial), 'jax').final.E.real.sum()
        with pytest.raises(ValueError, match='DecayConcreteOnly'):
            if case in ('jax-grad', 'jax-callable'):
                jax.grad(evaluate)(x)
            elif case == 'jax-jvp':
                jax.jvp(evaluate, (x,), (jax.numpy.ones_like(x),))
            elif case == 'jax-jit':
                jax.jit(evaluate)(x)
            elif case == 'jax-vmap':
                jax.vmap(evaluate)(jax.numpy.ones(2, dtype=jax.numpy.float64))
            else:
                jax.jit(lambda: fd.run(scene, 'jax').final.E)()


@pytest.mark.parametrize('cap_kind', ('steps', 'time', 'periods'))
def test_prepared_plan_and_report_bind_cap_criterion_and_effective_prefix(cap_kind, capsys):
    scene = scalar_scene()
    if cap_kind == 'time':
        scene = scene.replace(run=fd.Run(time=5., courant=.5, decay=scene.run.decay))
    elif cap_kind == 'periods':
        source = fd.Dipole('late', center=(0., 0., 0.), component='Ex',
                          band=fd.Band(wavelength=4., df=.025), amplitude=0.)
        scene = scene.replace(sources=(source,),
            run=fd.Run(periods=3., courant=.5, decay=scene.run.decay))
    planned = quiet(fd.plan, scene)
    public = quiet(fd.check, scene)
    assert public.steps == planned.steps
    count = _decay.window_steps(scene.run.decay.window, planned.dt)
    assert public.stopping == (planned.steps, scene.run.decay.threshold, scene.run.decay.window, count)
    assert 'hard cap' in capsys.readouterr().out
    for altered in (scene.run.decay.replace(threshold=.03),
                    scene.run.decay.replace(window=.23)):
        changed = scene.replace(run=scene.run.replace(decay=altered))
        other = quiet(fd.plan, changed)
        assert other.skeleton != planned.skeleton
        with pytest.raises(ValueError, match='different scene'):
            quiet(fd.run, changed, plan=planned)
        with pytest.raises(ValueError, match='DecayPlanMismatch'):
            quiet(fd.lower, changed, planned, runtime('numpy'))
    fixed = scene.replace(run=scene.run.replace(decay=None))
    fixed_plan = quiet(fd.plan, fixed)
    assert fixed_plan.skeleton != planned.skeleton
    with pytest.raises(ValueError, match='DecayPlanMismatch'):
        quiet(fd.lower, fixed, planned, runtime('numpy'))
    with pytest.raises(ValueError, match='DecayPlanMismatch'):
        quiet(fd.lower, scene, fixed_plan, runtime('numpy'))
    got = quiet(fd.run, scene, plan=planned)
    actual = got.plan.steps
    assert got.report.steps == actual and got.report.duration == actual*planned.dt
    assert got.report.stopping[0] == planned.steps
    assert got.diagnostics['stopping']['cap_skeleton'] == planned.skeleton
    assert got.report.bytes_per_state == public.bytes_per_state
    if actual < planned.steps:
        assert got.plan.skeleton != planned.skeleton
        with pytest.raises(ValueError, match='different scene'):
            quiet(fd.run, scene, plan=got.plan)


@pytest.mark.parametrize('backend', backends())
def test_fixed_runs_keep_their_energy_cadence_and_amplitude_derivative(backend):
    scene = scalar_scene(steps=20, decay=False)
    plan = quiet(fd.plan, scene)
    assert plan.decay is None and plan.energy.stride == 16
    result = quiet(fd.run, scene, backend)
    assert 'stopping' not in result.diagnostics
    assert result.report.stopping is None
    assert 'stopping' not in str(result.report)
    times, _ = result.diagnostics['interior_energy']
    assert np.array_equal(times, np.array([1., 17., 19.])*plan.dt)
    a = (2.-2.*plan.dt)/(2.+2.*plan.dt)
    expected = plan.cells*a**plan.steps
    if backend == 'numpy':
        doubled = quiet(fd.run, scene.replace(initial=scene.initial.replace(amplitude=2.)), backend)
        assert np.array_equal(doubled.final.E, 2.*result.final.E)
        # In this linear scalar recurrence dE/damplitude is its unit response.
        derivative = result.final.E.sum()
        assert abs(derivative-expected) <= roundoff_budget()*max(1., abs(expected))
    elif backend == 'jax':
        jax = pytest.importorskip('jax')
        with jax.enable_x64(True):
            def evaluate(value):
                request = scene.replace(initial=scene.initial.replace(amplitude=value))
                return fd.run(request, 'jax').final.E.sum()
            derivative = jax.grad(evaluate)(jax.numpy.asarray(1.))
            assert abs(float(derivative)-expected) <= roundoff_budget()*max(1., abs(expected))
    else:
        torch = pytest.importorskip('torch')
        value = torch.tensor(1., dtype=torch.float64, requires_grad=True)
        request = scene.replace(initial=scene.initial.replace(amplitude=value))
        total = fd.run(request, 'torch').final.E.sum()
        derivative, = torch.autograd.grad(total, value)
        assert abs(float(derivative)-expected) <= roundoff_budget()*max(1., abs(expected))
