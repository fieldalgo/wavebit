"""Evanescent incidence carries no incident power, so its efficiencies are NaN (#84).

``theta = pi/2 + 1j*alpha`` from a lossless medium has a real ``kx = n*cosh(alpha)`` beyond the
light line: the complex-incidence test passes it, and no ``ComplexIncidenceWarning`` fires.
Its ``kz`` is imaginary, so the incident wave carries no normal power. Until #84 the
efficiencies were divided by the roundoff ``cos(pi/2) = 6.1e-17`` leaves in ``Re(kz)``. Air
onto glass returned ``R = 1`` and ``T = -+1.72458e16`` with ``power_defined`` true, the sign of
``T`` set by the sign of ``alpha``. The amplitudes were right and must stay so: on a bare
interface they are Fresnel's, in closed form.

``alpha = +-1e-6`` sits just past the light line, where ``|kz| = sinh(alpha)`` is 1e-6 while
the roundoff in ``Re(kz)`` stays near 6e-17. A tolerance relative to ``|kz|`` lets that point
through (it returned ``T = -4.2e4``), so these two values pin the floor's absolute part.
"""
import warnings

import numpy as np
import pytest

from wavebit import Material, Units, rcwa
from wavebit._common.backend import ComplexIncidenceWarning
from wavebit.rcwa._factory import rcwa as legacy_rcwa

ALPHAS = [np.arccosh(1.2), -np.arccosh(1.2), 1e-6, -1e-6]
EPS_TOP, EPS_BOTTOM = 1.0, 2.25


def _numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def _fresnel(theta, polarization):
    """The bare interface's ``(r, t)``, with the incident root taken decaying, as the solve does."""
    kx = np.sin(theta).real                     # n_top = 1; the roundoff imaginary part dropped
    k1 = 1j * np.sqrt(kx**2 - EPS_TOP)          # evanescent in air: Im(kz) > 0
    k2 = np.sqrt(EPS_BOTTOM - kx**2 + 0j)       # propagating in glass
    if polarization == 's':
        return (k1 - k2) / (k1 + k2), 2 * k1 / (k1 + k2)
    # E-field ratios, in the convention where normal incidence gives r_p = -r_s.
    denominator = EPS_BOTTOM * k1 + EPS_TOP * k2
    return ((EPS_BOTTOM * k1 - EPS_TOP * k2) / denominator,
            2 * np.sqrt(EPS_TOP * EPS_BOTTOM) * k1 / denominator)


def _problem(theta, polarization='s'):
    return rcwa.Problem(
        units=Units(length='nm'),
        stack=rcwa.Stack(period=(1000., 1000.), top=Material(epsilon=EPS_TOP),
                         bottom=Material(epsilon=EPS_BOTTOM), layers=()),
        excitation=rcwa.Excitation(wavelength=1000., sources=(
            rcwa.PlaneWave(theta=theta, polarization=polarization),)))


def _solve_quietly(run):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        result = run()
    return result, [w for w in record if issubclass(w.category, UserWarning)]


@pytest.mark.parametrize('polarization', ['s', 'p'])
@pytest.mark.parametrize('alpha', ALPHAS, ids=lambda a: f'{a:+.4g}')
def test_evanescent_efficiencies_are_undefined_and_amplitudes_are_kept(alpha, polarization):
    """The issue's repro, on ``rcwa.run``: marked as a complex ``kx`` is, with no warning.

    Everything divided by the incident power is NaN and ``power_defined`` is false, the port
    powers included, since they share that flag. The amplitudes are Fresnel's to roundoff, as
    they were before the fix. No warning: the NaN is the mark.
    """
    theta = np.pi / 2 + 1j * alpha
    result, warned = _solve_quietly(
        lambda: rcwa.run(_problem(theta, polarization), numerics=rcwa.Numerics(orders=(0, 0))))
    assert not warned, [str(w.message) for w in warned]

    assert not np.asarray(result.reflection.metadata['power_defined']).any()
    for name, value in (('R', result.R), ('T', result.T),
                        ('power_entering', result.outputs['power_entering']),
                        ('absorption', result.outputs['absorption']),
                        ('reflected power', result.reflection.power),
                        ('transmitted power', result.transmission.power),
                        ('incident power', result.reflection.reference['incident_power'])):
        assert np.isnan(np.asarray(value)).all(), (name, value)

    want_r, want_t = _fresnel(theta, polarization)
    got_r = complex(np.ravel(getattr(result.reflection, 'amplitude_' + polarization))[0])
    got_t = complex(np.ravel(getattr(result.transmission, 'amplitude_' + polarization))[0])
    assert abs(got_r - want_r) < 1e-14, (got_r, want_r)
    assert abs(got_t - want_t) < 1e-14, (got_t, want_t)


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
@pytest.mark.parametrize('algorithm', ['ETM', 'SMM'])
@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('alpha', ALPHAS[:2], ids=lambda a: f'{a:+.4g}')
def test_the_solver_marks_every_ratio_on_a_grating(alpha, side, algorithm, backend):
    """The engine result itself, on a grating, lit from either side, on every backend.

    ``R``, ``T``, their s/p parts and the pure-s / pure-p variants, ``power_entering`` and so
    ``absorption``: every ratio to the incident power is NaN, and no amplitude is. The two
    algorithms solve the same scene, so their amplitudes must still agree.
    """
    if backend == 'jax':
        pytest.importorskip('jax')
        from wavebit._common.backend import _enable_x64
        _enable_x64()
    elif backend == 'torch':
        pytest.importorskip('torch')

    def solve(layer_algorithm):
        solver = legacy_rcwa(backend=backend, top=EPS_TOP, bot=EPS_BOTTOM,
                             theta=np.pi / 2 + 1j * alpha, phi=0.3, psi=np.pi / 3,
                             wavelength=1000., period=[800.], truncation_order=[3],
                             thickness=[150.], structure=np.array([[[1.0] * 4 + [6.25] * 4]]),
                             layer_algorithm=layer_algorithm)
        solver.incidence_side = side
        return _solve_quietly(solver.solve)

    result, warned = solve(algorithm)
    assert not warned, [str(w.message) for w in warned]
    for name in ('incident', 's_inc', 'p_inc'):
        sub = getattr(result, name)
        for ratio in ('R', 'R_s', 'R_p', 'T', 'T_s', 'T_p', 'power_entering', 'absorption'):
            assert np.isnan(_numpy(getattr(sub, ratio))).all(), (name, ratio)
        for amplitude in ('r_s', 'r_p', 't_s', 't_p'):
            assert np.isfinite(_numpy(getattr(sub, amplitude))).all(), (name, amplitude)
    other, _ = solve('SMM' if algorithm == 'ETM' else 'ETM')
    for amplitude in ('r_s', 'r_p', 't_s', 't_p'):
        np.testing.assert_allclose(_numpy(getattr(result.incident, amplitude)),
                                   _numpy(getattr(other.incident, amplitude)), atol=1e-11, rtol=0)


@pytest.mark.parametrize('alpha', ALPHAS, ids=lambda a: f'{a:+.4g}')
def test_an_excitation_lighting_an_evanescent_order_is_refused_for_either_sign(alpha):
    """The explicit-excitation door refuses it, as before, but now for both signs of ``alpha``.

    It tested ``weight > 0``, and the roundoff weight is positive for one sign: ``alpha < 0``
    went through and returned ``T = 1.09e16`` on this scene.
    """
    solver = legacy_rcwa(backend='numpy', top=EPS_TOP, bot=EPS_BOTTOM,
                         theta=np.pi / 2 + 1j * alpha, phi=0., psi=np.pi / 2,
                         wavelength=1000., period=[1000.], truncation_order=[0],
                         thickness=[100.], structure=np.full((1, 1, 8), 1.5),
                         excitation={0: 1.0})
    with pytest.raises(ValueError, match='no positive incident power'):
        solver.solve()


@pytest.mark.parametrize('theta', [0.4, np.pi / 2 - 1e-7], ids=['oblique', 'grazing-1e-7'])
def test_a_propagating_wave_keeps_its_efficiencies(theta):
    """The control: a real angle, even 1e-7 from grazing, is still normalized and conserves.

    ``kz = 1e-7`` there, ten million times the roundoff floor, so a floor drawn too wide would
    show up here as NaN.
    """
    result, warned = _solve_quietly(
        lambda: rcwa.run(_problem(theta), numerics=rcwa.Numerics(orders=(0, 0))))
    assert not warned, [str(w.message) for w in warned]
    assert np.asarray(result.reflection.metadata['power_defined']).all()
    total = float(np.sum(result.R) + np.sum(result.T))
    assert abs(total - 1) < 1e-12, total


def test_a_complex_transverse_wavevector_keeps_its_formal_ratios():
    """The other unpowered regime is unchanged: the solver keeps formal ``R`` and ``T`` and warns.

    ``theta = 0.5 + 0.3j`` makes ``kx`` complex; its incident ``kz`` has a real part well
    above roundoff, so it is no evanescent incidence and nothing new marks it.
    """
    solver = legacy_rcwa(backend='numpy', top=EPS_TOP, bot=EPS_BOTTOM, theta=0.5 + 0.3j, phi=0.,
                         psi=np.pi / 2, wavelength=1000., period=[1000.],
                         truncation_order=[0], thickness=[100.],
                         structure=np.full((1, 1, 8), 1.5))
    with pytest.warns(ComplexIncidenceWarning):
        result = solver.solve()
    assert np.isfinite(_numpy(result.incident.R)).all()
    assert np.isfinite(_numpy(result.incident.T)).all()
    assert np.isnan(_numpy(result.incident.power_entering)).all()


def test_a_jitted_solve_marks_the_same_values():
    """Under ``jax.jit`` nothing is read: the flag is traced, and the same values come back NaN.

    The amplitudes agree with the eager solve's, and a propagating angle through the same
    compiled function is still normalized, so the mark follows the value, not the trace.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    with jax.enable_x64(True):
        prepared = rcwa.prepare(_problem(np.pi / 2 + 1j * ALPHAS[0]),
                                numerics=rcwa.Numerics(orders=(0, 0)), backend='jax')

        def evaluate(theta):
            result = prepared.run(_problem(theta))
            return (result.R, result.T, result.reflection.metadata['power_defined'],
                    result.transmission.amplitude_s)

        jitted = jax.jit(evaluate)
        for alpha in ALPHAS[:2]:
            theta = jnp.asarray(np.pi / 2 + 1j * alpha)
            R, T, defined, t_s = jitted(theta)
            assert np.isnan(np.asarray(R)).all() and np.isnan(np.asarray(T)).all()
            assert not np.asarray(defined).any()
            eager_t_s = evaluate(theta)[3]
            np.testing.assert_allclose(np.asarray(t_s), np.asarray(eager_t_s), atol=1e-14, rtol=0)
        R, T, defined, _ = jitted(jnp.asarray(0.4 + 0j))
        assert np.asarray(defined).all()
        assert abs(float(np.sum(R) + np.sum(T)) - 1) < 1e-12
