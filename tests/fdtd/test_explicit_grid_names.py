"""``fdtd.native`` and ``fdtd.periodic`` under their public names.

The explicit-grid modules record ``Snapshots`` of components spelled 'Ex'..'Hz', diagnose decay
with a ``DecayCheck`` whose window is in seconds, and take a periodic pulse by its spectral
standard deviation. The old names fail and name the replacement; the engine underneath keeps its
own lowercase keywords, and only the public doors reword its refusals.
"""
import math
import traceback
from collections.abc import Mapping

import numpy as np
import pytest

from wavebit import Units, Material
from wavebit.errors import ValidationError
from wavebit.fdtd import native, periodic
from wavebit.fdtd._periodic import adapter, scene
from wavebit.fdtd._periodic.grid import window_steps
from wavebit.units import C0

COMPONENTS = ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')

#: Engine keywords a public refusal must not name: the caller never wrote them.
ENGINE_KEYWORDS = ('pulse_width', 'pulse_notches', 'n_periods', 'record_', 'mesh_indices',
                   'mesh_thickness', 'truncation_order', 'convergence_window')

#: The decay window of the periodic run below, in time steps.
PERIODIC_WINDOW = 40


def _shown(error):
    """What a reader sees of ``error``, Python's "Did you mean" included."""
    return ''.join(traceback.format_exception_only(error))


def _no_engine_keyword(text):
    named = [k for k in ENGINE_KEYWORDS if k in text]
    assert not named, f'{named} in {text!r}'


def _found(tree, key):
    """Every value stored under ``key`` anywhere in nested mappings, tuples and lists."""
    if isinstance(tree, Mapping):
        return [v for k, v in tree.items() if k == key] + [
            x for k, v in tree.items() if k != key for x in _found(v, key)]
    if isinstance(tree, (tuple, list)):
        return [x for v in tree for x in _found(v, key)]
    return []


def native_case(length='um', scale=1., window_steps_=7):
    """The small native problem of ``tests/fdtd/test_physical_native.py::problem``."""
    shape = (16, 2, 2)
    dt = .02e-6/C0
    steps = 24
    current = np.zeros((3,)+shape); current[0, 4] = 1e7
    q = native.Problem(units=Units(length=length), medium=native.Medium(epsilon=np.ones(shape)*2.),
        current=native.Current(density=current, samples=np.sin(np.arange(steps)*.1)),
        outputs=native.Outputs(spectrum=native.Spectrum(wavelengths=[.8*scale], z_indices=(10,)),
            snapshots=native.Snapshots(every=3, components=('Ex', 'Hy'),
                                       slices=((2, 13, 2), 0, (None, None, None)))))
    nums = native.Numerics(grid=native.Grid(spacing=(.05*scale,)*3,
        z_spacing=np.linspace(.045, .055, 16)*scale), time_step_seconds=dt, steps=steps,
        decay_check=native.DecayCheck(window_seconds=window_steps_*dt))
    return q, nums


def periodic_case(length='um', scale=1., wavelength=(.85, .9, .95), duration_periods=55, **problem):
    """The small periodic slab of ``tests/fdtd/test_physical_periodic.py::case``."""
    q = periodic.Problem(units=Units(length=length),
        stack=periodic.Stack(period=(.2*scale,), top=Material(1.), bottom=Material(1.),
            layers=(periodic.Layer(thickness=.1*scale, epsilon=np.full((1, 2), 2.25)),)),
        excitation=periodic.Excitation(wavelength=tuple(w*scale for w in wavelength),
                                       sources=(periodic.PlaneWave(),)),
        outputs=periodic.Outputs(planes={'outside': -.2*scale}), **problem)
    n = periodic.Numerics(spacing=scale/30, duration_seconds=duration_periods*.9e-6/C0, orders=0)
    return q, n


@pytest.fixture(scope='module')
def native_run():
    q, nums = native_case()
    return q, nums, native.run(q, numerics=nums)


@pytest.fixture(scope='module')
def periodic_run():
    """One periodic run with a pulse given by its sigma, a decay check and two snapshot components.

    The sigma is written as the expression of a 3-time-unit envelope width, so the engine gets
    the width it was always given.
    """
    q, n = periodic_case()
    width = 3*q.units.time_unit_seconds
    sigma = 1/(np.sqrt(2)*np.pi*3*q.units.time_unit_seconds)
    dt = periodic.check(q, numerics=n).resolved['time_step_seconds']
    q = q.replace(pulse=periodic.Pulse(frequency_sigma_hz=sigma),
                  outputs=q.outputs.replace(snapshots=periodic.Snapshots(
                      every=17, components=('Ex', 'Hy'))))
    n = n.replace(decay_check=periodic.DecayCheck(window_seconds=PERIODIC_WINDOW*dt,
                                                  threshold=3e-4))
    return dict(problem=q, numerics=n, result=periodic.run(q, numerics=n), sigma=sigma,
                width=width, dt=dt)


# 1. Removed names are tombstones that name their replacement.

REMOVED_NAMES = [(native, 'Initial', 'InitialField'), (native, 'Record', 'Snapshots'),
                 (native, 'Tail', 'DecayCheck'), (periodic, 'Record', 'Snapshots')]


@pytest.mark.parametrize('module, old, new', REMOVED_NAMES,
                         ids=[f'{m.__name__.rsplit(".", 1)[1]}.{old}' for m, old, _ in REMOVED_NAMES])
def test_a_removed_module_name_raises_attribute_error_naming_its_replacement(module, old, new):
    assert not hasattr(module, old)
    assert old not in module.__all__ and new in module.__all__
    with pytest.raises(AttributeError) as caught:
        getattr(module, old)
    message = str(caught.value)
    assert f'{module.__name__}.{old} is now {module.__name__}.{new}' in message, message
    assert caught.value.name == old
    shown = _shown(caught.value)
    assert 'public-names.md' in shown and 'Did you mean' not in shown, shown
    assert isinstance(getattr(module, new), type)


@pytest.mark.parametrize('module', [native, periodic], ids=['native', 'periodic'])
def test_a_missing_module_name_is_still_a_plain_attribute_error(module):
    with pytest.raises(AttributeError) as caught:
        module.NoSuchDeclaration
    assert str(caught.value) == f"module {module.__name__!r} has no attribute 'NoSuchDeclaration'"
    assert 'public-names.md' not in _shown(caught.value)
    # A typo of a public name still gets Python's own suggestion.
    with pytest.raises(AttributeError) as caught:
        module.Snapshot
    assert "Did you mean: 'Snapshots'?" in _shown(caught.value)
    sentinel = object()
    assert getattr(module, 'NoSuchDeclaration', sentinel) is sentinel


RESULT_RENAMED = [('native', 'fields', 'snapshots'),
                  ('native', 'polarization', 'polarization_density'),
                  ('native', 'previous_polarization', 'previous_polarization_density'),
                  ('periodic', 'fields', 'snapshots')]


def _result(request, which):
    run = request.getfixturevalue(f'{which}_run')
    return run[2] if which == 'native' else run['result']


@pytest.mark.parametrize('which, old, new', RESULT_RENAMED,
                         ids=[f'{w}.Result.{old}' for w, old, _ in RESULT_RENAMED])
def test_a_removed_result_attribute_raises_attribute_error_naming_its_replacement(
        request, which, old, new):
    result = _result(request, which)
    assert not hasattr(result, old)
    with pytest.raises(AttributeError) as caught:
        getattr(result, old)
    message = str(caught.value)
    assert f'{which}.Result.{old} is now {which}.Result.{new}' in message, message
    shown = _shown(caught.value)
    assert 'public-names.md' in shown and 'Did you mean' not in shown, shown
    assert getattr(result, new) is not None


@pytest.mark.parametrize('which', ['native', 'periodic'])
def test_a_missing_result_attribute_is_still_a_plain_attribute_error(request, which):
    result = _result(request, which)
    with pytest.raises(AttributeError) as caught:
        result.unrelated
    assert str(caught.value) == "'Result' object has no attribute 'unrelated'"
    assert 'public-names.md' not in _shown(caught.value)
    with pytest.raises(AttributeError) as caught:
        result.snapshot
    assert "Did you mean: 'snapshots'?" in _shown(caught.value)
    assert getattr(result, 'unrelated', None) is None
    # The renamed fields are ordinary fields, never answered by the tombstone hook.
    assert set(result.snapshots) == {'Ex', 'Hy'}


# 2. Components are 'Ex'..'Hz'; the snapshot interval and native slices are checked.

MODULES = pytest.mark.parametrize('module', [native, periodic], ids=['native', 'periodic'])


@MODULES
@pytest.mark.parametrize('components, public, given', [
    (('ex',), 'Ex', 'ex'), ('hz', 'Hz', 'hz'), (('Ex', 'hy'), 'Hy', 'hy'), (('EX',), 'Ex', 'EX'),
    (list(c.lower() for c in COMPONENTS), 'Ex', 'ex')])
def test_a_component_not_spelled_ex_to_hz_is_refused_naming_the_public_spelling(
        module, components, public, given):
    with pytest.raises(ValidationError) as caught:
        module.Snapshots(components=components)
    assert caught.value.path == 'outputs.snapshots.components'
    assert f'{public!r}, not {given!r}' in caught.value.message, caught.value.message
    assert repr(list(COMPONENTS)) in caught.value.message


@MODULES
@pytest.mark.parametrize('components', [('Ex', 'Ex'), (), ('Bx',), ('Ex', 3), 5, 'Exx'])
def test_unknown_duplicate_or_missing_components_are_refused(module, components):
    with pytest.raises(ValidationError) as caught:
        module.Snapshots(components=components)
    assert caught.value.path == 'outputs.snapshots.components'


@MODULES
def test_all_a_single_name_and_the_default_are_accepted(module):
    assert module.Snapshots().components == COMPONENTS
    assert module.Snapshots(components='all').components == COMPONENTS
    assert module.Snapshots(components='Ex').components == ('Ex',)
    assert module.Snapshots(components=['Hz', 'Ex']).components == ('Hz', 'Ex')


@MODULES
@pytest.mark.parametrize('every', [0, -1, 1.5, 2., np.float64(2.), True, np.True_, '2', None])
def test_the_snapshot_interval_must_be_a_positive_whole_number_of_steps(module, every):
    with pytest.raises(ValidationError) as caught:
        module.Snapshots(every=every)
    assert caught.value.path == 'outputs.snapshots.every'


@MODULES
def test_a_numpy_integer_interval_is_kept_as_a_python_int(module):
    snapshots = module.Snapshots(every=np.int64(4))
    assert snapshots.every == 4 and type(snapshots.every) is int


@pytest.mark.parametrize('slices', [(0, 0), (0, 0, 0, 0), 'zyx', b'zyx', 3, ()])
def test_native_slices_need_three_zyx_selectors(slices):
    with pytest.raises(ValidationError) as caught:
        native.Snapshots(slices=slices)
    assert caught.value.path == 'outputs.snapshots.slices'
    assert 'three (z,y,x) selectors' in caught.value.message


def test_native_slices_take_an_index_or_a_start_stop_step_tuple_per_axis():
    assert native.Snapshots(slices=(0, 1, (0, 2))).slices == (0, 1, (0, 2))
    assert native.Snapshots(slices=[(2, 13, 2), 0, (None, None, None)]).slices is not None
    assert native.Snapshots().slices is None


def test_native_snapshots_are_keyed_by_the_public_component_names(native_run):
    _, _, result = native_run
    assert tuple(result.snapshots) == ('Ex', 'Hy')
    for name, series in result.snapshots.items():
        assert series.components == (name,)
        assert series.units['field'] == ('V/m' if name.startswith('E') else 'A/m')
    assert len(result.polarization_density) == len(result.previous_polarization_density) == 0


def test_periodic_snapshots_are_keyed_by_the_public_component_names(periodic_run):
    result = periodic_run['result']
    assert tuple(result.snapshots) == ('Ex', 'Hy')
    for name, series in result.snapshots.items():
        assert series.components == (name,)
        assert series.units['field'] == ('V/m' if name.startswith('E') else 'A/m')


# 3. DecayCheck: a ratio in (0, 1) and a positive window in seconds.

@MODULES
@pytest.mark.parametrize('threshold', [0, 0., 1, 1., -1e-4, 1.5, np.nan, np.inf, True, np.True_,
                                       np.array([1e-3]), 1e-3+0j, 'small', None])
def test_a_decay_threshold_must_be_a_ratio_between_zero_and_one(module, threshold):
    with pytest.raises(ValidationError) as caught:
        module.DecayCheck(window_seconds=1e-15, threshold=threshold)
    assert caught.value.path == 'numerics.decay_check.threshold'


@MODULES
@pytest.mark.parametrize('window', [0, 0., -1e-15, np.nan, np.inf, -np.inf, True,
                                    np.array([1e-15]), 1e-15+0j, 'long', None])
def test_a_decay_window_must_be_a_finite_positive_duration(module, window):
    with pytest.raises(ValidationError) as caught:
        module.DecayCheck(window_seconds=window)
    assert caught.value.path == 'numerics.decay_check.window_seconds'


@MODULES
def test_a_decay_check_needs_its_window_and_defaults_its_threshold(module):
    with pytest.raises(TypeError, match='window_seconds'):
        module.DecayCheck()
    check = module.DecayCheck(window_seconds=np.float64(2e-15))
    assert check.threshold == 1e-4
    assert type(check.window_seconds) is float and check.window_seconds == 2e-15
    assert type(module.DecayCheck(window_seconds=2e-15, threshold=np.float32(.5)).threshold) is float


def test_each_module_takes_only_its_own_decay_check():
    with pytest.raises(ValidationError, match='expected native.DecayCheck') as caught:
        native.Numerics(grid=native.Grid(spacing=(.1,)*3), time_step_seconds=1e-16, steps=4,
                        decay_check=periodic.DecayCheck(window_seconds=1e-16))
    assert caught.value.path == 'numerics.decay_check'
    with pytest.raises(ValidationError, match='expected periodic.DecayCheck') as caught:
        periodic.Numerics(spacing=.02, duration_seconds=1e-14,
                          decay_check=native.DecayCheck(window_seconds=1e-16))
    assert caught.value.path == 'numerics.decay_check'


@pytest.mark.parametrize('window_in_steps', [24, 23.5, 23.0001, 30])
def test_native_refuses_a_decay_window_not_shorter_than_the_run(window_in_steps):
    q, nums = native_case()
    dt = nums.time_step_seconds
    nums = nums.replace(decay_check=native.DecayCheck(window_seconds=window_in_steps*dt))
    report = native.check(q, numerics=nums)
    assert not report.ok
    (issue,) = report.issues
    assert issue.path == 'numerics.decay_check.window_seconds'
    assert f'covers {math.ceil(window_in_steps)} steps' in issue.message, issue.message
    assert 'Numerics(steps=24)' in issue.message
    with pytest.raises(ValidationError) as caught:
        native.run(q, numerics=nums)
    assert caught.value.path == 'numerics.decay_check.window_seconds'


@pytest.mark.parametrize('window_in_steps', [23, 22.5, 1, .25])
def test_native_admits_a_decay_window_shorter_than_the_run(window_in_steps):
    q, nums = native_case()
    dt = nums.time_step_seconds
    assert native.check(q, numerics=nums.replace(
        decay_check=native.DecayCheck(window_seconds=window_in_steps*dt))).ok


# 4. The rounding rule both modules share: N*dt is N steps, anything longer rounds up.

#: A time unit's seconds, for the units a length can be given in (c=1 time).
TIME_UNITS = {'m': 1/C0, 'mm': 1e-3/C0, 'um': 1e-6/C0, 'nm': 1e-9/C0}
#: Time steps in the engine's c=1 unit: the Courant steps of common grids and arbitrary ones.
STEPS_C1 = (.02, .1, 1/3, .8/30, .8/np.sqrt(3)/30, .8/np.sqrt(2)/40, .5/7, .0123456789, .999)
COUNTS = tuple(range(1, 257)) + (1000, 3215, 4096, 12345, 10**6)


@pytest.mark.parametrize('unit', list(TIME_UNITS))
def test_a_window_of_n_time_steps_is_n_steps_in_any_time_unit(unit):
    ts = TIME_UNITS[unit]
    rng = np.random.default_rng(57)
    for dt in STEPS_C1 + tuple(rng.uniform(1e-3, 1., 8)):
        dt_seconds = dt*ts
        for n in COUNTS:
            # The engine's own unit, SI seconds (native), and the periodic round trip: the
            # window in seconds from check(...)'s time step, divided back by the time unit.
            for window, step in ((n*dt, dt), (n*dt_seconds, dt_seconds),
                                 ((n*dt_seconds)/ts, dt)):
                got = window_steps(window, step)
                assert got == n and type(got) is int, (unit, dt, n, window, step, got)


@pytest.mark.parametrize('dt', [.02, 1/3, 5.1355552041239214e-17, 1e-3])
def test_a_window_between_whole_steps_rounds_up(dt):
    for n in (1, 2, 7, 40, 1000):
        for fraction in (1e-6, .25, .5, .999):
            assert window_steps((n+fraction)*dt, dt) == n+1, (n, fraction)
        # Far more than rounding error above N*dt is a further step.
        assert window_steps(n*dt*(1+1e-12), dt) == n+1
        assert window_steps(n*dt*(1-1e-12), dt) == n
    assert window_steps(.3*dt, dt) == 1
    assert window_steps(1e-9*dt, dt) == 1


@pytest.mark.parametrize('length, scale', [('um', 1.), ('nm', 1000.)])
def test_a_native_window_of_n_time_steps_reports_n_steps(length, scale):
    for n in (1, 7, 23):
        q, nums = native_case(length, scale, window_steps_=n)
        tail = native.run(q, numerics=nums).diagnostics['tail']
        assert tail['window_steps'] == n
        assert tail['tolerance'] == 1e-4


# 5. The periodic pulse is given by its spectral standard deviation.

def _engine_options(monkeypatch, q, n):
    """The options ``periodic.check`` hands the engine."""
    seen = []
    engine = adapter.solve_periodic_fdtd

    def spy(options, *args, **kwargs):
        seen.append(dict(options))
        return engine(options, *args, **kwargs)
    monkeypatch.setattr(adapter, 'solve_periodic_fdtd', spy)
    report = periodic.check(q, numerics=n)
    monkeypatch.undo()
    assert report.ok, report.issues
    (options,) = seen
    return options


@pytest.mark.parametrize('width_units', [3, .9, 2.5])
def test_a_periodic_sigma_reaches_the_engine_as_the_width_it_names(monkeypatch, width_units):
    q, n = periodic_case()
    ts = q.units.time_unit_seconds
    sigma = 1/(np.sqrt(2)*np.pi*width_units*ts)
    options = _engine_options(monkeypatch, q.replace(pulse=periodic.Pulse(frequency_sigma_hz=sigma)), n)
    np.testing.assert_allclose(options['pulse_width'], width_units, rtol=4e-16, atol=0)
    assert _engine_options(monkeypatch, q, n)['pulse_width'] is None


def test_a_periodic_decay_check_reaches_the_engine_in_seconds(monkeypatch):
    q, n = periodic_case()
    ts = q.units.time_unit_seconds
    n = n.replace(decay_check=periodic.DecayCheck(window_seconds=40*ts, threshold=2e-3))
    options = _engine_options(monkeypatch, q, n)
    assert options['convergence_window_time'] == 40*ts/ts and options['convergence_tol'] == 2e-3
    assert options.get('convergence_window') is None
    options = _engine_options(monkeypatch, q, n.replace(decay_check=None))
    assert options['convergence_window_time'] is None and options['convergence_tol'] == 1e-4


def test_the_periodic_run_reports_the_width_and_spectrum_its_sigma_names(periodic_run):
    pulse = periodic_run['result'].diagnostics['pulse']
    sigma = periodic_run['sigma']
    np.testing.assert_allclose(pulse['width_seconds'], periodic_run['width'], rtol=1e-15, atol=0)
    np.testing.assert_allclose(pulse['width_seconds'], 1/(np.sqrt(2)*np.pi*sigma), rtol=1e-15,
                               atol=0)
    # Independently: the amplitude spectrum is exp(-(f-f0)**2/(2*sigma**2)) about the centre.
    frequency = C0/(np.array([.85, .9, .95])*1e-6)
    np.testing.assert_allclose(pulse['frequency_hz'], C0/.9e-6, rtol=1e-14)
    np.testing.assert_allclose(pulse['relative_spectrum'],
                               np.exp(-(frequency-C0/.9e-6)**2/(2*sigma**2)), rtol=1e-12)


@pytest.mark.parametrize('sigma', [0, 0., -1., -3e13, np.nan, np.inf, True, np.True_,
                                   np.array([3e13]), np.full((2,), 3e13), 3e13+0j, [3e13]])
def test_a_periodic_sigma_must_be_a_finite_positive_scalar(sigma):
    with pytest.raises(ValidationError) as caught:
        periodic.Pulse(frequency_sigma_hz=sigma)
    assert caught.value.path == 'pulse.frequency_sigma_hz'


def test_a_periodic_sigma_may_be_any_real_positive_number():
    assert periodic.Pulse().frequency_sigma_hz is None
    for sigma in (3e13, np.float64(3e13), 30000000000000):
        assert periodic.Pulse(frequency_sigma_hz=sigma).frequency_sigma_hz == 3e13


def test_the_removed_pulse_width_states_how_to_convert_it():
    with pytest.raises(TypeError) as caught:
        periodic.Pulse(width_seconds=1e-14)
    message = str(caught.value)
    assert 'Pulse(width_seconds=...) is now Pulse(frequency_sigma_hz=...)' in message, message
    assert 'frequency_sigma_hz = 1/(sqrt(2)*pi*width_seconds)' in message, message


# 6. The periodic door spells engine refusals as the declarations the caller wrote.

def _refused(q, n, phrase):
    report = periodic.check(q, numerics=n)
    assert not report.ok
    (issue,) = report.issues
    assert phrase in issue.message, issue.message
    _no_engine_keyword(issue.message)
    return issue


@pytest.mark.parametrize('make, phrase', [
    (lambda: periodic_case(wavelength=(.4, 2.)),
     'source spectrum is numerically zero at a requested wavelength: raise '
     'Pulse(frequency_sigma_hz=...)'),
    (lambda: periodic_case(duration_periods=2),
     'Numerics(duration_seconds=...) is too short for the pulse'),
], ids=['zero-source-spectrum', 'too-short'])
def test_a_periodic_engine_refusal_names_the_public_declaration_in_check_and_run(make, phrase):
    q, n = make()
    issue = _refused(q, n, phrase)
    with pytest.raises(ValidationError) as caught:
        periodic.run(q, numerics=n)
    assert caught.value.message == issue.message
    assert caught.value.path == issue.path
    _no_engine_keyword(str(caught.value))
    # The engine keeps its own words for its own callers; the public refusal is raised from them.
    assert isinstance(caught.value.__cause__, ValueError)
    assert any(k in str(caught.value.__cause__) for k in ENGINE_KEYWORDS)


@pytest.mark.parametrize('change, phrase', [
    (lambda q, n: (q.replace(pulse=periodic.Pulse(frequency_sigma_hz=1e20)), n),
     'Pulse(frequency_sigma_hz=...) is too broad for the time step'),
    (lambda q, n: (q, n.replace(decay_check=periodic.DecayCheck(window_seconds=2*n.duration_seconds))),
     'DecayCheck(window_seconds=...) must be shorter than the run, Numerics(duration_seconds=...)'),
    (lambda q, n: (q.replace(outputs=periodic.Outputs(snapshots=periodic.Snapshots(
        bounds={'w': (0, 1)}))), n), 'Snapshots(bounds=...)'),
    (lambda q, n: (q.replace(outputs=periodic.Outputs(snapshots=periodic.Snapshots(
        bounds={'z': (5, 6)}))), n), "no native samples for 'Ex'"),
    (lambda q, n: (q, n.replace(orders=-1)), 'Numerics(orders=...)'),
    (lambda q, n: (q, n.replace(nominal_indices=(-1.,))), 'Numerics(nominal_indices=...)'),
    (lambda q, n: (q, n.replace(nominal_layer_thickness=(-1.,))),
     'Numerics(nominal_layer_thickness=...)'),
    (lambda q, n: (q.replace(pulse=periodic.Pulse(notch_frequencies_hz=[-1.])), n),
     'Pulse(notch_frequencies_hz=...)'),
], ids=['sigma-too-broad', 'window-too-long', 'bounds-axis', 'bounds-empty', 'orders',
        'nominal-indices', 'nominal-thickness', 'notches'])
def test_other_periodic_engine_refusals_name_the_public_declaration(change, phrase):
    q, n = change(*periodic_case())
    assert _refused(q, n, phrase).path == 'problem'


def test_a_periodic_decay_window_of_the_whole_run_is_refused_and_one_step_less_is_not():
    q, n = periodic_case()
    resolved = periodic.check(q, numerics=n).resolved
    dt, steps = resolved['time_step_seconds'], resolved['steps']
    _refused(q, n.replace(decay_check=periodic.DecayCheck(window_seconds=steps*dt)),
             'DecayCheck(window_seconds=...) must be shorter than the run')
    assert periodic.check(q, numerics=n.replace(
        decay_check=periodic.DecayCheck(window_seconds=(steps-1)*dt))).ok


def test_sampling_a_periodic_result_without_all_six_components_names_snapshots(periodic_run):
    with pytest.raises(ValidationError) as caught:
        periodic_run['result'].sample(z=.05)
    assert 'Result.sample needs all six Snapshots(components=...)' in caught.value.message
    _no_engine_keyword(str(caught.value))


# 7. A periodic window of N time steps is N steps, whichever unit the problem uses.

def test_a_periodic_decay_window_of_n_time_steps_reports_n_steps(periodic_run):
    convergence = periodic_run['result'].diagnostics['convergence']
    counts = _found(convergence, 'window_steps')
    assert counts and all(c == PERIODIC_WINDOW for c in counts), counts
    tolerances = _found(convergence, 'tolerance')
    assert tolerances and all(t == 3e-4 for t in tolerances), tolerances


@pytest.mark.parametrize('length, scale', [('um', 1.), ('nm', 1000.), ('mm', 1e-3)])
def test_periodic_check_converts_a_window_of_n_time_steps_to_n_steps(monkeypatch, length, scale):
    q, n = periodic_case(length, scale)
    resolved = periodic.check(q, numerics=n).resolved
    dt, steps = resolved['time_step_seconds'], resolved['steps']
    counts = []

    def spy(window, step):
        counts.append(window_steps(window, step))
        return counts[-1]
    monkeypatch.setattr(scene, 'window_steps', spy)
    wanted = (1, 7, PERIODIC_WINDOW, 333, steps-1)
    for count in wanted:
        assert periodic.check(q, numerics=n.replace(
            decay_check=periodic.DecayCheck(window_seconds=count*dt))).ok
    assert tuple(counts) == wanted
