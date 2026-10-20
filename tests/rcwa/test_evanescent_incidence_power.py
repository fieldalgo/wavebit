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

Only the ratios go. The port powers in W stay: the light tunnels into the glass, through the
interference of the incident and reflected waves, and that power is physical. So ``rcwa.run``
keeps two flags: ``power_defined`` (a cell power integral exists; false for a complex ``kx``)
and ``efficiency_defined`` (an incident power exists too; false here as well).

Under ``jax.jit`` the mark is a traced ``where``, and a ``where`` does not keep a NaN in the
branch it discards out of a gradient. Where the incident power is exactly 0 (a direct ``kx``
beyond the light line, an evanescent lit order, ``theta = pi/2``), the kernels therefore divide
by 1 instead when the power is traced, and only then.
"""
import warnings

import numpy as np
import pytest

from wavebit import Material, Units, rcwa
from wavebit._common.backend import ComplexIncidenceWarning
from wavebit.observations.power import net_power
from wavebit.rcwa._factory import rcwa as legacy_rcwa
from wavebit.units import Z0

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


def _problem(theta, polarization='s', outputs=None):
    return rcwa.Problem(
        units=Units(length='nm'),
        stack=rcwa.Stack(period=(1000., 1000.), top=Material(epsilon=EPS_TOP),
                         bottom=Material(epsilon=EPS_BOTTOM), layers=()),
        excitation=rcwa.Excitation(wavelength=1000., sources=(
            rcwa.PlaneWave(theta=theta, polarization=polarization),)),
        **({} if outputs is None else {'outputs': outputs}))


def _first(value):
    return np.ravel(np.asarray(value))[0]


def _solve_quietly(run):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        result = run()
    return result, [w for w in record if issubclass(w.category, UserWarning)]


@pytest.mark.parametrize('polarization', ['s', 'p'])
@pytest.mark.parametrize('alpha', ALPHAS, ids=lambda a: f'{a:+.4g}')
def test_evanescent_efficiencies_are_undefined_and_amplitudes_are_kept(alpha, polarization):
    """The issue's repro, on ``rcwa.run``: every ratio NaN, every amplitude kept, no warning.

    ``efficiency_defined`` is false and everything divided by the incident power is NaN, with
    ``incident_power`` itself. ``power_defined`` stays true: the cell power integral exists.
    The amplitudes are Fresnel's to roundoff, as they were before the fix. No warning: the NaN
    is the mark.
    """
    theta = np.pi / 2 + 1j * alpha
    result, warned = _solve_quietly(
        lambda: rcwa.run(_problem(theta, polarization), numerics=rcwa.Numerics(orders=(0, 0))))
    assert not warned, [str(w.message) for w in warned]

    assert np.asarray(result.reflection.metadata['power_defined']).all()
    assert not np.asarray(result.reflection.metadata['efficiency_defined']).any()
    for name, value in (('R', result.R), ('T', result.T),
                        ('power_entering', result.outputs['power_entering']),
                        ('absorption', result.outputs['absorption']),
                        ('incident power', result.reflection.reference['incident_power'])):
        assert np.isnan(np.asarray(value)).all(), (name, value)

    want_r, want_t = _fresnel(theta, polarization)
    got_r = complex(np.ravel(getattr(result.reflection, 'amplitude_' + polarization))[0])
    got_t = complex(np.ravel(getattr(result.transmission, 'amplitude_' + polarization))[0])
    assert abs(got_r - want_r) < 1e-14, (got_r, want_r)
    assert abs(got_t - want_t) < 1e-14, (got_t, want_t)


@pytest.mark.parametrize('polarization', ['s', 'p'])
@pytest.mark.parametrize('alpha', ALPHAS, ids=lambda a: f'{a:+.4g}')
def test_the_tunnelled_power_is_kept_in_watts(alpha, polarization):
    """The port powers stay, because the light that tunnels into the glass carries real power.

    The incident wave alone carries no normal power, and neither does the specular reflected
    wave. Their interference does: on a lossless interface the net flux through the lit face is
    what the glass receives. So the transmitted power in W equals ``|t|**2 * Re(kz)`` and the
    interference term, in the cell's watts. The specular reflected power is roundoff of zero,
    bounded on the cell's absolute scale (at ``alpha = 1e-6`` the transmitted power is itself
    only 5e-27 W). The per-wave budget ``incident - reflected - transmitted`` cannot see the
    interference and would read the tunnelled power as a negative absorption, so it comes back
    NaN with ``incident_power``.
    """
    theta = np.pi / 2 + 1j * alpha
    result = rcwa.run(_problem(theta, polarization), numerics=rcwa.Numerics(orders=(0, 0)))
    reference = result.reflection.reference
    scale = (float(_first(reference['cell_measure'][0]))
             * abs(complex(_first(reference['incident_amplitude'])))**2 / (2 * Z0))
    r = complex(_first(getattr(result.reflection, 'amplitude_' + polarization)))
    t = complex(_first(getattr(result.transmission, 'amplitude_' + polarization)))
    kx = np.sin(theta).real
    k1, k2 = 1j * np.sqrt(kx**2 - EPS_TOP), np.sqrt(EPS_BOTTOM - kx**2)

    transmitted = float(_first(result.transmission.power))
    assert transmitted > 0
    assert transmitted == pytest.approx(scale * abs(t)**2 * k2, rel=1e-12)
    # The interference term: (1 + r) against (1 - r) with r near -1 near the light line.
    admittance = k1 if polarization == 's' else k1 / EPS_TOP
    assert transmitted == pytest.approx(scale * net_power(np, 1.0, r, admittance), rel=1e-8)
    assert abs(float(_first(result.reflection.power))) < 1e-14 * scale

    budget = (np.asarray(reference['incident_power']) - np.sum(np.asarray(result.reflection.power))
              - np.sum(np.asarray(result.transmission.power)))
    assert np.isnan(budget).all()


@pytest.mark.parametrize('diffraction', [True, False])
def test_each_flag_marks_exactly_its_own_values(diffraction):
    """Three regimes, two flags, one meaning each.

    ``(power_defined, efficiency_defined)`` is ``(T, T)`` at a real angle, ``(T, F)`` under
    evanescent incidence and ``(F, F)`` for a complex ``kx``. The port powers are finite exactly
    where ``power_defined`` holds; the efficiencies, ``incident_power``, ``power_entering``
    and ``absorption`` exactly where ``efficiency_defined`` does. With no diffraction output the
    diagnostics carry both flags, so a NaN ``power_entering`` is still explained.
    """
    cases = [(0.4, True, True), (np.pi / 2 + 1j * ALPHAS[0], True, False), (0.5 + 0.3j, False, False)]
    for theta, power_ok, ratio_ok in cases:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ComplexIncidenceWarning)
            result = rcwa.run(_problem(theta, outputs=rcwa.Outputs(diffraction=diffraction)),
                              numerics=rcwa.Numerics(orders=(0, 0)))
        where = f'theta={theta}'
        assert bool(_first(result.diagnostics['port_power_defined'])) == power_ok, where
        assert bool(_first(result.diagnostics['efficiency_defined'])) == ratio_ok, where
        for name in ('power_entering', 'absorption'):
            assert np.isfinite(np.asarray(result.outputs[name])).all() == ratio_ok, (where, name)
        if not diffraction:
            continue
        for port in (result.reflection, result.transmission):
            assert bool(_first(port.metadata['power_defined'])) == power_ok, where
            assert bool(_first(port.metadata['efficiency_defined'])) == ratio_ok, where
            assert np.isfinite(np.asarray(port.power)).all() == power_ok, where
            assert np.isfinite(np.asarray(port.efficiency)).all() == ratio_ok, where
            assert np.isfinite(np.asarray(port.reference['incident_power'])).all() == ratio_ok, where


def test_selecting_a_wavelength_slices_the_new_flag_with_the_rest():
    """``efficiency_defined`` is a per-wavelength entry, so a selection slices it like its twin.

    Undeclared in ``metadata_wavelength_fields``, ``at_wavelength_index`` would hand back the whole
    two-wavelength flag beside a one-wavelength ``power_defined``.
    """
    problem = _problem(np.pi / 2 + 1j * ALPHAS[0])
    problem = problem.replace(excitation=problem.excitation.replace(wavelength=(1000., 1100.)))
    result = rcwa.run(problem, numerics=rcwa.Numerics(orders=(0, 0)))
    for port in (result.reflection, result.transmission):
        assert np.shape(port.metadata['efficiency_defined']) == (2,)
        selected = port.at_wavelength_index(1).metadata
        assert np.shape(selected['efficiency_defined']) == np.shape(selected['power_defined']) == (1,)
        assert not np.asarray(selected['efficiency_defined']).any()


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
    assert np.asarray(result.reflection.metadata['efficiency_defined']).all()
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
    """Under ``jax.jit`` nothing is read: the flags are traced, and the same values come back.

    The ratios are NaN, the transmitted watts and the amplitudes agree with the eager solve's,
    and a propagating angle through the same compiled function is still normalized, so the
    mark follows the value, not the trace.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    with jax.enable_x64(True):
        prepared = rcwa.prepare(_problem(np.pi / 2 + 1j * ALPHAS[0]),
                                numerics=rcwa.Numerics(orders=(0, 0)), backend='jax')

        def evaluate(theta):
            result = prepared.run(_problem(theta))
            metadata = result.reflection.metadata
            return (result.R, result.T, metadata['power_defined'], metadata['efficiency_defined'],
                    result.transmission.amplitude_s, result.transmission.power)

        jitted = jax.jit(evaluate)
        for alpha in ALPHAS[:2]:
            theta = jnp.asarray(np.pi / 2 + 1j * alpha)
            R, T, power_ok, ratio_ok, t_s, watts = jitted(theta)
            assert np.isnan(np.asarray(R)).all() and np.isnan(np.asarray(T)).all()
            assert np.asarray(power_ok).all() and not np.asarray(ratio_ok).any()
            eager = evaluate(theta)
            np.testing.assert_allclose(np.asarray(t_s), np.asarray(eager[4]), atol=1e-14, rtol=0)
            np.testing.assert_allclose(np.asarray(watts), np.asarray(eager[5]), rtol=1e-12)
            assert np.isfinite(np.asarray(watts)).all()
        R, T, power_ok, ratio_ok, _, _ = jitted(jnp.asarray(0.4 + 0j))
        assert np.asarray(power_ok).all() and np.asarray(ratio_ok).all()
        assert abs(float(np.sum(R) + np.sum(T)) - 1) < 1e-12


def _film_problem(eps, kx, polarization):
    """A 100 nm film of ``eps`` between air and glass, lit at a direct ``kx``."""
    return rcwa.Problem(
        units=Units(length='nm'),
        stack=rcwa.Stack(period=(1000., 1000.), top=Material(epsilon=EPS_TOP),
                         bottom=Material(epsilon=EPS_BOTTOM),
                         layers=(rcwa.Layer(thickness=100., background=Material(epsilon=eps)),)),
        excitation=rcwa.Excitation(wavelength=1000., sources=(
            rcwa.PlaneWave(kx=kx, ky=0., polarization=polarization),)))


@pytest.mark.parametrize('polarization', ['s', 'p'])
@pytest.mark.parametrize('algorithm', ['smm', 'etm'])
def test_a_jitted_gradient_through_the_mark_is_zero_at_the_direct_kx_door(algorithm, polarization):
    """A loss masking ``T`` with ``efficiency_defined`` differentiates either side of the light line.

    A real ``kx`` beyond the lit medium's light line makes the incident ``kz`` exactly imaginary
    and the incident power exactly 0. The eager door refuses it; under ``jax.jit`` the refusal
    cannot run, and the kernels divided by that 0. The mask kept the value right (0), but the
    ``inf`` and ``0/0`` in the branch it discarded reached the gradient: ``jit(grad)`` returned NaN
    here, where the same wave written as ``theta = pi/2 + 1j*acosh(1.2)``, whose divisor is the
    roundoff 6e-17, returned 0. The kernels now divide by 1 there.

    The mask is ``efficiency_defined``, the flag of the ratios. ``power_defined`` stays true here
    (the cell power integral exists, and the tunnelled watts are kept), so masking ``T`` with it
    would hand the NaN ``T`` through.

    One compiled function serves both doors, ``kx`` traced. Beyond the light line the value and
    gradient are 0. At a propagating ``kx`` they are the eager solve's, which never reaches the
    substitute 1, and the gradient is no masked zero.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    k0 = 2 * np.pi / 1000.
    with jax.enable_x64(True):
        prepared = rcwa.prepare(_film_problem(4.0, 0.3 * k0, polarization),
                                numerics=rcwa.Numerics(orders=(0, 0), algorithm=algorithm),
                                backend='jax')

        def loss(eps, kx):
            result = prepared.run(_film_problem(eps, kx, polarization))
            defined = result.reflection.metadata['efficiency_defined'][0]
            return jnp.where(defined, result.T.sum(), 0.)

        jitted = jax.jit(jax.value_and_grad(loss))
        value, gradient = jitted(4.0, 1.2 * k0)
        assert float(value) == 0. and float(gradient) == 0., (value, gradient)

        value, gradient = jitted(4.0, 0.3 * k0)
        eager_value, eager_gradient = jax.value_and_grad(loss)(4.0, 0.3 * k0)
        assert abs(float(eager_gradient)) > 1e-3, eager_gradient
        np.testing.assert_allclose(float(value), float(eager_value), rtol=1e-13, atol=0)
        np.testing.assert_allclose(float(gradient), float(eager_gradient), rtol=1e-12, atol=0)


@pytest.mark.parametrize('algorithm', ['ETM', 'SMM'])
@pytest.mark.parametrize('layout', [{'tetm': 'te'}, {'tetm': 'tm'}, {'phi': 0.3, 'psi': np.pi / 3}],
                         ids=['pure-TE', 'pure-TM', 'conical'])
def test_every_kernel_divides_by_a_finite_power_under_jit(layout, algorithm):
    """Each kernel division, reached under ``jax.jit`` with an incident power of exactly 0.

    The pure TE/TM kernels (``solve_1d``, ``smm_1d``) divide by ``incident_power``; the coupled
    ones (``final_matching``, ``_sp_scattering_result``) by ``sp_incident_power``. Two inputs
    make that power exactly 0: an excitation lighting order 1, evanescent in air on this 800 nm
    grating (refused eagerly), and ``theta = pi/2``, whose incident ``kz`` is exactly 0. Both
    gradients were NaN on every kernel. A propagating angle through the same kernels keeps the
    eager gradient.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    with jax.enable_x64(True):
        def loss(excitation):
            def value(eps, theta):
                solver = legacy_rcwa(backend='jax', top=EPS_TOP, bot=EPS_BOTTOM, theta=theta,
                                     wavelength=1000., period=[800.], truncation_order=[3],
                                     thickness=[150.],
                                     structure=jnp.array([[[1.0] * 4 + [eps] * 4]]),
                                     layer_algorithm=algorithm, excitation=excitation, **layout)
                result = solver.solve()
                s, p = solver._incident_power_defined()
                return jnp.where(s & p, jnp.sum(result.incident.T), 0.)
            return value

        gradient = jax.jit(jax.grad(loss({1: 1.0})))(6.25, 0.3)
        assert float(gradient) == 0., gradient

        unlit = jax.jit(jax.grad(loss(None)))
        assert float(unlit(6.25, np.pi / 2)) == 0., unlit(6.25, np.pi / 2)
        eager = jax.grad(loss(None))(6.25, 0.3)
        assert abs(float(eager)) > 1e-3, eager
        np.testing.assert_allclose(float(unlit(6.25, 0.3)), float(eager), rtol=1e-12, atol=0)


@pytest.mark.filterwarnings('ignore:invalid value encountered:RuntimeWarning')
@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_an_exact_graze_is_still_refused_eagerly(backend):
    """``theta = pi/2`` gives an incident power of exactly 0, and ``check_finite`` still refuses it.

    The kernels put 1 in place of that power only where nothing can be read, under ``jax.jit``,
    where ``check_finite`` cannot run either. Doing it everywhere would turn this refusal into a
    NaN-marked result, which changes an eager answer rather than fixing a gradient.
    """
    if backend == 'jax':
        pytest.importorskip('jax')
        from wavebit._common.backend import _enable_x64
        _enable_x64()
    elif backend == 'torch':
        pytest.importorskip('torch')
    solver = legacy_rcwa(backend=backend, top=EPS_TOP, bot=EPS_BOTTOM, theta=np.pi / 2,
                         phi=0.3, psi=np.pi / 3, wavelength=1000., period=[800.],
                         truncation_order=[3], thickness=[150.],
                         structure=np.array([[[1.0] * 4 + [6.25] * 4]]))
    with pytest.raises(FloatingPointError, match='non-finite value in de_ri'):
        solver.solve()
