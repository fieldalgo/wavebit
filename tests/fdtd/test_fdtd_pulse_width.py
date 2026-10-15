"""Scene pulse width controls the spectrum without changing its calibration contract."""

from wavebit.fdtd._periodic.api import fdtd as _legacy_wavebit_fdtd
import numpy as np
import pytest



def options(**changes):
    return dict(dict(structure=np.full((1, 1, 1), 2.25), thickness=[.2], top=1., bot=1.,
                     period=.1, wavelength=[.9, 1., 1.1], backend='numpy', resolution=10,
                     n_periods=45, n_pml=4, truncation_order=0, tetm='te', return_diagnostics=True),
                **changes)


def test_default_pulse_is_unchanged_and_metadata_uses_requested_band():
    implicit, a = _legacy_wavebit_fdtd(**options(record_field=11))
    explicit, b = _legacy_wavebit_fdtd(**options(pulse_width=1., record_field=11))
    for first, second in zip(implicit, explicit):
        for name in ('r_s', 't_s', 'R', 'T'):
            np.testing.assert_array_equal(getattr(first.incident, name), getattr(second.incident, name))
    np.testing.assert_array_equal(a['fields']['ey']['values'], b['fields']['ey']['values'])
    assert a['pulse']['width'] == 1.
    assert a['pulse']['peak_time'] == 5.
    assert a['pulse']['frequency'] == 1.
    expected = np.exp(-(np.pi*(1/np.array([.9, 1., 1.1])-1))**2)
    np.testing.assert_allclose(a['pulse']['relative_spectrum'], expected, atol=1e-15)


def test_pulse_width_uses_physical_c1_time_and_preserves_scale_invariance():
    first, a = _legacy_wavebit_fdtd(**options(pulse_width=2.))
    second, b = _legacy_wavebit_fdtd(**options(pulse_width=4., thickness=[.4], period=.2,
                                      wavelength=[1.8, 2., 2.2], resolution=5))
    for before, after in zip(first, second):
        for name in ('r_s', 't_s', 'R', 'T'):
            np.testing.assert_allclose(getattr(before.incident, name), getattr(after.incident, name), atol=1e-11)
    assert b['dt'] == pytest.approx(2*a['dt'])
    assert b['pulse']['peak_time'] == 2*a['pulse']['peak_time']
    np.testing.assert_allclose(a['pulse']['relative_spectrum'], b['pulse']['relative_spectrum'], atol=1e-15)


@pytest.mark.parametrize('value', [0., -1., np.nan, np.inf, True, 1j, [1.], '1'])
def test_invalid_static_width_is_rejected(value):
    with pytest.raises(ValueError, match='pulse_width'):
        _legacy_wavebit_fdtd(**options(pulse_width=value))


@pytest.mark.parametrize('changes,match', [
    ({'pulse_width': .001}, 'time step'),
    ({'pulse_width': 20.}, 'numerical source tail'),
    ({'wavelength': [.1, 1., 2.]}, 'numerical source tail'),
    ({'pulse_width': 5., 'wavelength': 1.}, 'too short'),
])
def test_unresolved_source_or_incomplete_run_is_rejected(changes, match):
    with pytest.raises(ValueError, match=match):
        _legacy_wavebit_fdtd(**options(**changes))


def test_source_tail_guard_refuses_past_the_double_precision_limit(monkeypatch):
    """The guard refuses a pulse whose tail the source precision cannot represent.

    The guard reads the backend's source precision, and that precision is double on every
    backend now, JAX included -- so which backend it read is no longer observable from a scene
    call, and a guard hard-coding float64 would pass this too: ``JaxTimeLoop.__init__``
    lifts ``jax_enable_x64`` when the run builds its time loop, which is upstream of the
    ``finfo`` this guard reads. So the float32 limit this test used to straddle -- with
    ``pulse_width=10.``, between the float64 floor and 100 times float32 epsilon -- is no
    longer reachable from a scene call: setting the cap back on below is overridden from
    inside ``wavebit.fdtd``, and that is asserted rather than worked around.

    The limit is therefore the double one, ``sqrt(-log(100*eps))/(pi*span)`` with
    ``span = max|1/wavelength - 1/centre|``, computed here from the same two quantities the
    guard uses rather than pasted in (16.0629 for this band, against 9.6460 at float32). It is
    straddled from both sides: just past it must refuse, just short of it must reach the time
    loop -- which is stubbed out, because what is under test is the guard, not a 16-unit pulse
    and the 180 periods a run of one would need. The cap is switched off again before each arm,
    so each one shows on its own that the call lifts it before the guard reads ``finfo``.
    """
    jax = pytest.importorskip('jax')
    import wavebit.fdtd._periodic.scene as scene

    band = np.array([.9, 1., 1.1])
    centre = (band.min()+band.max())/2
    span = np.max(np.abs(1/band-1/centre))
    limit = np.sqrt(-np.log(100*np.finfo(np.float64).eps))/(np.pi*span)
    assert limit == pytest.approx(16.0629, abs=1e-4)

    class Stepped(Exception):
        """Raised in place of the time loop: the guard let this width through."""

    def stub(*args, **kwargs):
        raise Stepped
    monkeypatch.setattr(scene, 'fdtd_grid', stub)

    previous = jax.config.jax_enable_x64
    try:
        jax.config.update('jax_enable_x64', False)
        with pytest.raises(ValueError, match='numerical source tail'):
            _legacy_wavebit_fdtd(**options(backend='jax', pulse_width=1.01*limit, n_periods=180))
        assert jax.config.jax_enable_x64, 'the run must lift the cap before its first array'
        jax.config.update('jax_enable_x64', False)
        with pytest.raises(Stepped):
            _legacy_wavebit_fdtd(**options(backend='jax', pulse_width=.99*limit, n_periods=180))
        assert jax.config.jax_enable_x64, 'the run must lift the cap before its first array'
    finally:
        jax.config.update('jax_enable_x64', previous)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_narrow_pulse_preserves_bottom_graded_band_material_gradient(backend):
    def run(value, which):
        results, _ = _legacy_wavebit_fdtd(**options(
            structure=value.reshape(1, 1, 1), backend=which, pulse_width=2., grading=True,
            incidence_side='bottom', dft_subsample=2, convergence_window=30))
        return sum((v.incident.r_s+.4j*v.incident.t_s).real.sum() for v in results)
    expected = run(np.array(2.25), 'numpy')
    delta = 1e-4
    finite = (run(np.array(2.25+delta), 'numpy')-run(np.array(2.25-delta), 'numpy'))/(2*delta)
    if backend == 'jax':
        jax = pytest.importorskip('jax')
        jax.config.update('jax_enable_x64', True)
        value, derivative = jax.jit(jax.value_and_grad(lambda v: run(v, 'jax')))(2.25)
        convert = np.asarray
    else:
        torch = pytest.importorskip('torch')
        parameter = torch.tensor(2.25, dtype=torch.float64, requires_grad=True)
        value = run(parameter, 'torch')
        derivative, = torch.autograd.grad(value, parameter)
        convert = lambda v: v.detach().numpy()
    np.testing.assert_allclose(convert(value), expected, atol=1e-11)
    np.testing.assert_allclose(convert(derivative), finite, rtol=2e-5, atol=1e-8)
    assert abs(finite) > .01


def test_narrow_oblique_metal_pulse_settles_and_refines_to_fresnel_airy():
    band = np.array([.9, 1., 1.1])
    omega = 2*np.pi/band
    epsilon = 1.5+160/(-omega**2-1j*.08*omega)+40/(100-omega**2-1j*.12*omega)
    transverse = np.sin(.45)*band  # One fixed transverse wavevector throughout the band.
    q0 = np.sqrt(1-transverse**2)
    q1 = np.sqrt(epsilon-transverse**2)  # Passive upper-half-plane material in this test.
    phase = np.exp(1j*omega*q1*.2)
    expected = {}
    for component, interface in [('s', (q0-q1)/(q0+q1)),
                                  ('p', (epsilon*q0-q1)/(epsilon*q0+q1))]:
        denominator = 1-interface**2*phase**2
        expected[component] = np.array([interface*(1-phase**2)/denominator,
                                        (1-interface**2)*phase/denominator])
    errors = []
    for resolution, width in [(40, 1.), (40, 3.), (80, 3.)]:
        result, diagnostic = _legacy_wavebit_fdtd(**options(
            structure=np.full((1, 1, 1), 1.5), period=.05, theta=.45,
            resolution=resolution, pulse_width=width, n_periods=80,
            n_pml=round(.3*resolution), convergence_window=200,
            dispersion=np.ones((1, 1, 1), int), poles={1: [(0., .08, 160.), (10., .12, 40.)]}))
        c = diagnostic['convergence']
        if width == 1:
            assert not c['field_decayed']
            assert not c['amplitudes_settled']
        else:
            assert c['state_decayed']
            assert c['amplitudes_settled']
            error = 0.
            for component, basis in [('s', 's_inc'), ('p', 'p_inc')]:
                actual = np.array([[getattr(getattr(v, basis), name+'_'+component).item()
                                    for v in result] for name in ('r', 't')])
                error = max(error, np.max(abs(actual-expected[component])))
            errors.append(error)
    assert errors[1] < .003
    assert errors[1] < .4*errors[0]
