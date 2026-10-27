"""A one-way waveguide mode source.

Criteria S1-S8 of ``docs/validation/fdtd/fd-mode-source.md``, registered before the source was
first run; every limit here is quoted from there.
"""
import cmath
import math
import warnings

import numpy as np
import pytest

from wavebit import Material, Units, fdtd
from wavebit.errors import WavebitError

EPS_CORE, EPS_CLAD = 4.0, 2.085136
LAMBDAS = (1.45, 1.55, 1.65)
CENTRE = 1                              # LAMBDAS[CENTRE] is the pulse's centre
SIGMA_HZ = 1.5e13
N_EXACT_TE0 = 1.9562181938              # the mode-solver record's M3, TE0 at 1.55 um
PERIODS = 120


def _quiet():
    warnings.simplefilter('ignore')


def _pulse(wavelength=1.55):
    return fdtd.GaussianPulse(wavelength=wavelength, frequency_sigma_hz=SIGMA_HZ, cutoff=1e-12)


def _core(kind, eps=EPS_CORE):
    if kind == 'slab':
        return (fdtd.Structure('core', shape=fdtd.SnappedSlab(x=(-.75, .75)),
                               material=Material(epsilon=eps), continuation=('z-', 'z+')),)
    # S5: a butt joint at z = 0, 1.5 um wide below and 1.0 um wide above.
    return (fdtd.Structure('wide', shape=fdtd.SnappedBox(size=(1.5, 1., 4.), center=(0., 0., -2.)),
                           material=Material(epsilon=eps), continuation=('z-',)),
            fdtd.Structure('narrow', shape=fdtd.SnappedBox(size=(1., 1., 4.), center=(0., 0., 2.)),
                           material=Material(epsilon=eps), continuation=('z+',)))


def _setup(dx, *, mirrored=False, kind='slab', amplitude=1., phase=0., delay=0., scatterer=None,
           core_eps=EPS_CORE, pulse=None, monitors=None, source_kw=None, periods=PERIODS,
           duration=None):
    """Setup S (or its mirror, or the S5 joint) at spacing ``dx``."""
    structures = _core(kind, core_eps)
    if scatterer is not None:
        # S7: a traced block in the cladding beside the core, between the source and p1.
        structures += (fdtd.Structure('block', shape=fdtd.SnappedBox(size=(.4, 1., .4),
                                                                     center=(1.2, 0., -.5)),
                                      material=Material(epsilon=scatterer)),)
    z_source, direction = (2., '-z') if mirrored else (-2., '+z')
    source = fdtd.ModeSource('src', plane=fdtd.Plane(z=z_source), pulse=pulse or _pulse(),
                             direction=direction, mode=0, amplitude=amplitude, phase=phase,
                             delay_seconds=delay, **(source_kw or {}))
    if monitors is None:
        if mirrored:
            places = (('m', -2.),)
        else:
            places = (('p1', 1.), ('p2', 2.), ('q', -1. if kind == 'joint' else -3.))
        monitors = tuple(fdtd.ModeMonitor(name, plane=fdtd.Plane(z=z), wavelengths=LAMBDAS,
                                          modes=1, source='src') for name, z in places)
    problem = fdtd.Problem(
        units=Units(length='um'),
        scene=fdtd.Scene(
            domain=fdtd.Domain(size=(9., dx, 8.)), background=Material(epsilon=EPS_CLAD),
            boundaries=fdtd.Boundaries(x=fdtd.Open(), y=fdtd.InvariantAxis(), z=fdtd.Open()),
            structures=structures),
        sources=(source,), outputs=fdtd.Outputs(monitors=monitors))
    numerics = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=dx),
                             duration=duration or fdtd.Duration(periods=periods, source='src'),
                             absorber=fdtd.PML(cells=24, wavelength=1.55))
    return problem, numerics


def _run(problem, numerics, backend='numpy'):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = fdtd.run(problem, numerics=numerics, backend=backend)
    names = [type(w.message).__name__ for w in caught]
    assert 'UnsettledFieldWarning' not in names and 'GrowingFieldWarning' not in names, names
    return result


def _np(value):
    if hasattr(value, 'detach'):
        value = value.detach()
    return np.asarray(value)


def _P(spectrum, distance):
    """The mode's grid propagation factor over ``distance`` um at each wavelength (the run's
    Fourier convention is exp(+i omega t), so a wave advancing by d gains exp(+i beta d))."""
    beta = 2 * np.pi * _np(spectrum.effective_index)[:, 0] / np.asarray(LAMBDAS)
    return np.exp(1j * beta * distance)


class _Straight(dict):
    """Setup S per spacing, run once on first use (a test at one spacing runs only that one)."""

    def __missing__(self, dx):
        _quiet()
        self[dx] = _run(*_setup(dx))
        return self[dx]


@pytest.fixture(scope='module')
def straight():
    return _Straight()


# ---------------------------------------------------------------------------------------------
# S1, S2: what the sheet injects and leaks, as predicted
# ---------------------------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize('dx', [.05, .025])
def test_s1_the_injected_mode_arrives_as_predicted(straight, dx):
    p2 = straight[dx]['p2']
    injected = _np(p2.injected)
    predicted = injected * _P(p2, 4.)
    error = np.abs(_np(p2.forward)[:, 0] - predicted) / np.abs(injected)
    print(f'S1 dx={dx}: |forward - predicted| / |injected| =', error)
    assert np.all(error <= 1e-5), error


@pytest.mark.slow
def test_s1_the_mirrored_source_injects_as_predicted():
    _quiet()
    m = _run(*_setup(.05, mirrored=True))['m']
    injected = _np(m.injected)
    error = np.abs(_np(m.backward)[:, 0] - injected * _P(m, 4.)) / np.abs(injected)
    print('S1 mirrored: |backward - predicted| / |injected| =', error)
    assert np.all(error <= 1e-5), error


@pytest.mark.slow
@pytest.mark.parametrize('dx', [.05, .025])
def test_s2_the_leakage_is_as_predicted(straight, dx):
    q, p2 = straight[dx]['q'], straight[dx]['p2']
    injected, leakage = _np(q.injected), _np(q.leakage)
    predicted = leakage * _P(q, 1.) + _np(p2.backward)[:, 0] * _P(q, 5.)
    error = np.abs(_np(q.backward)[:, 0] - predicted) / np.abs(injected)
    power = np.abs(leakage / injected) ** 2
    print(f'S2 dx={dx}: |backward - predicted| / |injected| =', error,
          '; leakage power / injected =', power)
    assert np.all(error <= 1e-6), error
    assert power[CENTRE] <= 1e-12, power


# ---------------------------------------------------------------------------------------------
# S3: absolute power
# ---------------------------------------------------------------------------------------------

def _unit_spectrum(f):
    """``|W(f)|`` of ``w(t) = exp(-s^2/(2 tau^2)) sin(2 pi f0 s)``, both Gaussian sidebands."""
    tau = 1.0 / (2 * math.pi * SIGMA_HZ)
    f0 = 299792458.0 / 1.55e-6
    g = lambda x: math.exp(-0.5 * (2 * math.pi * x * tau) ** 2)
    # W(f) = tau sqrt(2 pi) / (2i) [g(f - f0) - g(f + f0)] up to a phase of s's origin.
    return tau * math.sqrt(2 * math.pi) / 2 * abs(g(f - f0) - g(f + f0))


@pytest.mark.slow
@pytest.mark.parametrize('dx', [.05, .025])
def test_s3_a_unit_amplitude_carries_one_watt_per_metre(straight, dx):
    p2 = straight[dx]['p2']
    f0 = float(_np(p2.frequencies_hz)[CENTRE])
    expected = 2 * 1.0 ** 2 * _unit_spectrum(f0) ** 2
    got = float(_np(p2.forward_energy)[CENTRE, 0])
    print(f'S3 dx={dx}: forward_energy / expected - 1 =', got / expected - 1, p2.units)
    assert abs(got / expected - 1) <= 2e-5, (got, expected)


# ---------------------------------------------------------------------------------------------
# S4: convergence with resolution
# ---------------------------------------------------------------------------------------------

def _n_run(result):
    p1, p2 = result['p1'], result['p2']
    k0 = 2 * math.pi / 1.55
    phase = cmath.phase(complex(_np(p2.forward)[CENTRE, 0] / _np(p1.forward)[CENTRE, 0]))
    n_solver = float(_np(p2.effective_index)[CENTRE, 0])
    branch = round((n_solver * k0 * 1. - phase) / (2 * math.pi))
    return (phase + 2 * math.pi * branch) / k0, n_solver


@pytest.mark.slow
def test_s4_the_run_propagates_the_grid_mode_and_converges(straight):
    errors = {}
    for dx in (.05, .025):
        n_run, n_solver = _n_run(straight[dx])
        print(f'S4 dx={dx}: n_run={n_run:.12f}, n_solver={n_solver:.12f}, '
              f'n_run - n_exact={n_run - N_EXACT_TE0:.3e}')
        assert abs(n_run - n_solver) <= 1e-9, (n_run, n_solver)
        errors[dx] = abs(n_run - N_EXACT_TE0)
    order = math.log2(errors[.05] / errors[.025])
    print('S4 observed order', order)
    assert order >= 1.8, (errors, order)


# ---------------------------------------------------------------------------------------------
# S5: butt joint against continuum mode matching
# ---------------------------------------------------------------------------------------------

def _te_modes(half_width, W, h, k0, count):
    """TE modes (Ey) of a symmetric slab in a perfectly conducting box ``|x| < W``: the
    continuum operator ``d^2/dx^2 + k0^2 eps(x)`` by second-order finite differences on nodes
    ``h`` apart (the core edges on nodes, eps averaged there), Ey = 0 at the walls. Returns
    ``(beta^2, modes)`` with ``sum(mode^2) h = 1``, largest ``beta^2`` first."""
    from scipy.linalg import eigh_tridiagonal
    n = int(round(2 * W / h)) - 1
    x = -W + h * np.arange(1, n + 1)
    eps = np.where(np.abs(x) < half_width, EPS_CORE, EPS_CLAD)
    eps[np.isclose(np.abs(x), half_width, atol=h * 1e-6)] = (EPS_CORE + EPS_CLAD) / 2
    diagonal = -2.0 / h ** 2 + k0 ** 2 * eps
    off = np.full(n - 1, 1.0 / h ** 2)
    values, vectors = eigh_tridiagonal(diagonal, off, select='i', select_range=(n - count, n - 1))
    order = np.argsort(-values)
    return values[order], vectors[:, order] / math.sqrt(h)


def _mode_matching(W, h, count, wavelength=1.55):
    """TE0 power transmission and reflection at the joint 1.5 um -> 1.0 um at z = 0.

    Fields go as exp(-i beta z) forward; an evanescent mode has beta = -i |beta|. Ey continuity
    is tested with the right-hand modes and Hx continuity with the left-hand ones (both sets
    orthonormal): ``c = O (e0 + b)`` and ``diag(bL) (e0 - b) = O^T diag(bR) c``."""
    k0 = 2 * math.pi / wavelength
    b2L, eL = _te_modes(.75, W, h, k0, count)
    b2R, eR = _te_modes(.5, W, h, k0, count)
    beta = lambda b2: np.where(b2 > 0, np.sqrt(np.abs(b2)), -1j * np.sqrt(np.abs(b2)))
    bL, bR = beta(b2L), beta(b2R)
    O = eR.T @ eL * h
    e0 = np.zeros(count)
    e0[0] = 1.0
    M = O.T @ (bR[:, None] * O)
    b = np.linalg.solve(np.diag(bL) + M, bL * e0 - M @ e0)
    c = O @ (e0 + b)
    return abs(c[0]) ** 2 * bR[0].real / bL[0].real, abs(b[0]) ** 2


def _oracle():
    T, R = _mode_matching(W=12., h=.0025, count=800)
    # Converged in the number of modes and in the box width at the scale of the criteria.
    for other in (_mode_matching(W=12., h=.0025, count=400), _mode_matching(W=18., h=.0025, count=1200)):
        assert abs(other[0] - T) <= 2e-4 and abs(other[1] - R) <= .02 * R, (T, R, other)
    return T, R


@pytest.mark.slow
def test_s5_a_butt_joint_matches_mode_matching():
    _quiet()
    T_mm, R_mm = _oracle()
    errors = {}
    for dx in (.05, .025):
        result = _run(*_setup(dx, kind='joint'))
        p2, q = result['p2'], result['q']
        T = float(_np(p2.forward_energy)[CENTRE, 0] / _np(p2.injected_energy)[CENTRE])
        R = float(_np(q.backward_energy)[CENTRE, 0] / _np(q.injected_energy)[CENTRE])
        errors[dx] = (abs(T - T_mm), abs(R - R_mm))
        print(f'S5 dx={dx}: T={T:.6f} (mode matching {T_mm:.6f}), R={R:.4e} (mode matching '
              f'{R_mm:.4e})')
    assert errors[.05][0] <= 5e-3 and errors[.025][0] <= 2e-3, errors
    assert errors[.025][0] < errors[.05][0], errors
    assert errors[.025][1] <= .2 * R_mm and errors[.025][1] < errors[.05][1], errors


# ---------------------------------------------------------------------------------------------
# S6: backends
# ---------------------------------------------------------------------------------------------

def _readings(result):
    p2, q = result['p2'], result['q']
    return {'injected': _np(p2.injected), 'leakage': _np(p2.leakage),
            'forward': _np(p2.forward)[:, 0], 'backward': _np(q.backward)[:, 0]}


@pytest.mark.slow
def test_s6_backends_agree(straight):
    import jax
    jax.config.update('jax_enable_x64', True)
    _quiet()
    reference = _readings(straight[.05])
    scale = np.max(np.abs(reference['injected']))
    for backend in ('jax', 'torch'):
        got = _readings(_run(*_setup(.05), backend=backend))
        for key, value in reference.items():
            error = float(np.max(np.abs(got[key] - value))) / scale
            print(f'S6 {backend} {key}: {error:.2e}')
            assert error <= 1e-12, (backend, key, error)


# ---------------------------------------------------------------------------------------------
# S7: derivatives
# ---------------------------------------------------------------------------------------------

S7_POINT = {'amplitude': 1.3, 'phase': .4, 'delay': 2e-15, 'scatterer': 3.0}
S7_STEPS = {'amplitude': 1e-3, 'phase': 1e-3, 'delay': 1e-17, 'scatterer': 3e-4}


def _s7_outputs(point, backend):
    problem, numerics = _setup(.05, amplitude=point['amplitude'], phase=point['phase'],
                               delay=point['delay'], scatterer=point['scatterer'],
                               monitors=(fdtd.ModeMonitor('p2', plane=fdtd.Plane(z=2.),
                                                          wavelengths=(1.55,), source='src'),))
    p2 = fdtd.run(problem, numerics=numerics, backend=backend)['p2']
    return p2.forward_energy[0, 0], p2.forward[0, 0].real


def _central(name, which):
    def at(step):
        hi, lo = dict(S7_POINT), dict(S7_POINT)
        hi[name] += step
        lo[name] -= step
        return (float(_s7_outputs(hi, 'numpy')[which])
                - float(_s7_outputs(lo, 'numpy')[which])) / (2 * step)
    h = S7_STEPS[name]
    return (4 * at(h / 2) - at(h)) / 3


#: (parameter, output): 0 is forward_energy, 1 is Re forward.
S7_CASES = (('amplitude', 0), ('phase', 1), ('delay', 1), ('scatterer', 0))


@pytest.mark.slow
def test_s7_derivatives_match_central_differences():
    _quiet()
    import jax
    import torch
    jax.config.update('jax_enable_x64', True)
    names = [name for name, _ in S7_CASES]
    for name, which in S7_CASES:
        reference = _central(name, which)

        def f(x, backend, name=name, which=which):
            point = dict(S7_POINT)
            point[name] = x
            return _s7_outputs(point, backend)[which]
        by_jax = float(jax.grad(lambda x: f(x, 'jax'))(S7_POINT[name]))
        x = torch.tensor(S7_POINT[name], dtype=torch.float64, requires_grad=True)
        f(x, 'torch').backward()
        by_torch = float(x.grad)
        print(f'S7 {name}: jax {by_jax:.10e}, torch {by_torch:.10e}, central {reference:.10e}')
        for got in (by_jax, by_torch):
            assert abs(got - reference) <= 1e-6 * abs(reference), (name, got, reference)
    assert names == ['amplitude', 'phase', 'delay', 'scatterer']


# ---------------------------------------------------------------------------------------------
# S8: refusals
# ---------------------------------------------------------------------------------------------

def _refused(code, problem, numerics, backend='numpy'):
    _quiet()
    report = fdtd.check(problem, numerics=numerics, backend=backend)
    assert [issue.code for issue in report.issues] == [code], report.issues
    with pytest.raises(WavebitError) as info:
        fdtd.run(problem, numerics=numerics, backend=backend)
    assert info.value.code == code, (info.value.code, str(info.value))


def test_s8_a_direction_off_the_normal_is_refused():
    _refused('source.direction', *_direction('+x'))


def _direction(direction):
    problem, numerics = _setup(.05, periods=1)
    source = problem.sources[0].replace(direction=direction)
    return problem.replace(sources=(source,)), numerics


@pytest.mark.parametrize('mode', [-1, 1.5, True])
def test_s8_a_mode_is_a_nonnegative_integer(mode):
    with pytest.raises(WavebitError) as info:
        fdtd.ModeSource('src', plane=fdtd.Plane(z=-2.), pulse=_pulse(), mode=mode)
    assert info.value.code == 'source.mode'


def test_s8_a_mode_beyond_the_confined_ones_is_refused():
    problem, numerics = _setup(.05, periods=1)
    problem = problem.replace(sources=(problem.sources[0].replace(mode=40),))
    _refused('mode.count', problem, numerics)


def test_s8_a_traced_structure_on_the_source_plane_is_refused():
    import torch
    eps = torch.tensor(EPS_CORE, dtype=torch.float64, requires_grad=True)
    _refused('mode.traced', *_setup(.05, periods=1, core_eps=eps), backend='torch')


def test_s8_a_plane_outside_the_domain_is_refused():
    problem, numerics = _setup(.05, periods=1)
    problem = problem.replace(sources=(problem.sources[0].replace(plane=fdtd.Plane(z=5.)),))
    _refused('source.outside_domain', problem, numerics)


def test_s8_a_monitor_on_the_sheet_is_refused():
    monitors = (fdtd.ModeMonitor('m', plane=fdtd.Plane(z=-1.95), wavelengths=(1.55,)),)
    _refused('monitor.source_sheet', *_setup(.05, periods=1, monitors=monitors))


def test_s8_source_names_a_mode_source():
    monitors = (fdtd.ModeMonitor('m', plane=fdtd.Plane(z=1.), wavelengths=(1.55,), source='nope'),)
    _refused('monitor.source', *_setup(.05, periods=1, monitors=monitors))
    with pytest.raises(WavebitError) as info:
        fdtd.ModeMonitor('m', plane=fdtd.Plane(z=1.), wavelengths=(1.55,), source='')
    assert info.value.code == 'monitor.source'


def test_s8_a_traced_pulse_wavelength_is_refused():
    import torch
    lam = torch.tensor(1.55, dtype=torch.float64, requires_grad=True)
    # A run length in periods of the source would read the wavelength first (and refuse it as a
    # static input of preparation); in seconds, the mode source itself refuses it.
    _refused('mode.traced', *_setup(.05, pulse=_pulse(lam), duration=fdtd.Duration(seconds=1e-15)),
             backend='torch')


def test_s8_the_engine_placeholder_names_the_public_source():
    import wavebit.fdtd._legacy as fd
    with pytest.raises(NotImplementedError, match='wavebit.fdtd.ModeSource'):
        fd.ModeSource()
