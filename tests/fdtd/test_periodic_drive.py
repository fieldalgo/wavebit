"""fdtd.periodic's diffraction amplitude ratios describe the cell; the drive moves power and fields.

fieldalgo/wavebit#54: the ``DiffractionOrders`` amplitudes from ``fdtd.periodic.run`` were the
engine's unit-drive ratios multiplied by the source amplitude ``A`` and labelled ``'V/m'``,
while ``rcwa.run`` and ``fdtd.run`` published the ratio itself under the same name. They are
the ratio now in every solver, ``amplitude_ratio_s/_p`` (dimensionless, like a Fresnel
coefficient), and the drive is carried by what has units: ``reference['incident_field']``
(V/m), the fields ``field_s/_p`` (the ratio times that drive, V/m), the new ``power`` and
``reference['incident_power']`` (W, or W/m across an invariant y), the planes, the snapshots
and ``plane_power_z``.
"""
import numpy as np
import pytest
from wavebit import Units, Material
from wavebit.fdtd import periodic as p
from wavebit.units import C0, Z0


def slab(algorithm='shared', amplitude=1., snapshots=False):
    q = p.Problem(units=Units(length='um'),
        stack=p.Stack(period=(.2,), top=Material(1.), bottom=Material(1.),
            layers=(p.Layer(thickness=.1, epsilon=np.full((1, 2), 2.25)),)),
        excitation=p.Excitation(wavelength=(.85, .9, .95),
                                sources=(p.PlaneWave(amplitude=amplitude, theta=.2),)),
        outputs=p.Outputs(planes={'outside': -.2},
                          snapshots=p.Snapshots(every=17) if snapshots else None))
    n = p.Numerics(algorithm=algorithm, spacing=1/30, duration_seconds=55*.9e-6/C0, orders=(0, 0))
    return q, n


@pytest.mark.parametrize('algorithm', ['shared', 'reference'])
def test_the_drive_moves_power_and_fields_and_leaves_the_ratios(algorithm):
    """Doubling the drive (to 2j) quadruples every power, multiplies every field by 2j, and
    leaves every amplitude ratio and efficiency bit for bit as it was."""
    q, n = slab(algorithm, snapshots=True)
    one = p.run(q, numerics=n)
    two = p.run(slab(algorithm, amplitude=2j, snapshots=True)[0], numerics=n)
    pairs = [(one.reflection, two.reflection), (one.transmission, two.transmission)]
    # Pinned, not only equal: two empty bases would leave the legs below unchecked.
    assert set(one.basis) == set(two.basis) == {'s_inc', 'p_inc'}
    pairs += [(a, b) for leg in one.basis for a, b in zip(one.basis[leg], two.basis[leg])]
    for a, b in pairs:
        assert a is not None and b is not None
        # assert_array_equal(None, None) passes, so every compared array is checked present.
        for name in ('amplitude_ratio_s', 'amplitude_ratio_p', 'efficiency', 'power'):
            assert getattr(a, name) is not None and getattr(b, name) is not None
        for name in ('amplitude_ratio_s', 'amplitude_ratio_p', 'efficiency'):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
        for orders in (a, b):
            assert orders.reference['amplitude_ratio_units'] == '1'
            assert orders.reference['incident_field_units'] == 'V/m'
        assert a.reference['incident_field'] == 1. and b.reference['incident_field'] == 2j
        for name in ('s', 'p'):
            field_a, field_b = getattr(a, 'field_'+name), getattr(b, 'field_'+name)
            assert field_a is not None and field_b is not None
            # The field is the ratio times the drive, 2j here, phase included ...
            np.testing.assert_allclose(field_b, 2j*getattr(b, 'amplitude_ratio_'+name),
                                       rtol=1e-15, atol=0)
            # ... so it moves by 2j with the drive while the ratio stays.
            np.testing.assert_allclose(field_b, 2j*field_a, rtol=1e-15, atol=0)
        np.testing.assert_allclose(b.order(0, 0).field_s, b.field_s[:, 0, 0], rtol=1e-15, atol=0)
        np.testing.assert_allclose(b.power, 4*a.power, rtol=1e-14)
        np.testing.assert_allclose(b.reference['incident_power'], 4*a.reference['incident_power'],
                                   rtol=1e-14)
    np.testing.assert_allclose(two.plane_power_z['outside'], 4*one.plane_power_z['outside'],
                               rtol=1e-14)
    np.testing.assert_allclose(two.planes['outside'].values, 2j*one.planes['outside'].values,
                               rtol=1e-14)
    for name in one.snapshots:
        np.testing.assert_allclose(two.snapshots[name].values, 2j*one.snapshots[name].values,
                                   rtol=1e-14, atol=0)
    # Result.sample interpolates the unit-drive records and multiplies by the drive itself.
    sampled = one.sample(z=-.2).values
    assert np.max(np.abs(sampled)) > 0
    np.testing.assert_allclose(two.sample(z=-.2).values, 2j*sampled, rtol=1e-14, atol=0)


@pytest.mark.parametrize('algorithm', ['shared', 'reference'])
def test_a_zero_drive_is_admitted_and_divides_nothing(algorithm):
    """A = 0 runs: nothing divides by the drive. The ratios and efficiencies are the A = 1
    run's bit for bit; the fields, the powers and the incident power are zero, and both flags
    stay True, because a zero power is a defined power, not a missing one."""
    q, n = slab(algorithm)
    one = p.run(q, numerics=n)
    zero = p.run(slab(algorithm, amplitude=0.)[0], numerics=n)
    pairs = [(one.reflection, zero.reflection), (one.transmission, zero.transmission)]
    assert set(one.basis) == set(zero.basis) == {'s_inc', 'p_inc'}
    pairs += [(a, b) for leg in one.basis for a, b in zip(one.basis[leg], zero.basis[leg])]
    for a, b in pairs:
        assert a is not None and b is not None
        for name in ('amplitude_ratio_s', 'amplitude_ratio_p', 'efficiency'):
            assert getattr(a, name) is not None and getattr(b, name) is not None
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
        # A finite efficiency, so the zero power below is a product and not a masked entry.
        assert np.isfinite(b.efficiency).all()
        assert b.reference['incident_field'] == 0
        for name in ('field_s', 'field_p', 'power'):
            assert getattr(b, name) is not None
            # assert_array_equal against 0 fails on a NaN, which a division by A would leave.
            np.testing.assert_array_equal(getattr(b, name), 0.)
        np.testing.assert_array_equal(b.reference['incident_power'], [0., 0., 0.])
        for flag in ('power_defined', 'efficiency_defined'):
            np.testing.assert_array_equal(b.metadata[flag], (True, True, True))


@pytest.mark.parametrize('algorithm', ['shared', 'reference'])
def test_incident_power_follows_each_wavelengths_angle_on_both_algorithms(algorithm):
    """The per-wavelength closed form on both routes, through one period of a 1D cell (W/m).

    Both engines fix k_par at the band's centre, so wavelength lambda meets the cell at
    sin(theta) * lambda / 0.9 um: 0.189, 0.200 and 0.211 rad here. The flux comparison below
    is shared-only; this pins the closed form on the reference route too, whose angles are its
    own (``diag['angles']``) and not the declared 0.2 rad.
    """
    q, n = slab(algorithm)
    r = p.run(q, numerics=n)
    sines = np.sin(.2)*np.array([.85, .9, .95])/.9
    for orders in (r.reflection, r.transmission):
        reference = orders.reference
        assert reference['invariant_axes'] == ('y',)
        assert reference['cell_measure_units'] == 'm'
        np.testing.assert_allclose(reference['cell_measure'], .2e-6, rtol=1e-15)
        assert reference['incident_power_units'] == reference['power_units'] == 'W/m'
        np.testing.assert_allclose(orders.metadata['theta_radians'], np.arcsin(sines), rtol=1e-14)
        np.testing.assert_allclose(reference['incident_power'],
                                   .5/Z0*np.sqrt(1-sines**2)*.2e-6, rtol=1e-14)
        np.testing.assert_allclose(orders.power,
                                   orders.efficiency*reference['incident_power'][:, None, None],
                                   rtol=1e-15)


def matched(spacing):
    """A vacuum cell lit at 0.3 rad, crossed: every wave is the incident one."""
    q = p.Problem(units=Units(length='um'),
        stack=p.Stack(period=(.2, .2), top=Material(1.), bottom=Material(1.),
            layers=(p.Layer(thickness=.1, epsilon=np.full((2, 2), 1.)),)),
        excitation=p.Excitation(wavelength=(.85, .9, .95), sources=(p.PlaneWave(theta=.3),)),
        outputs=p.Outputs(planes={'below': -.2}))
    n = p.Numerics(spacing=spacing, duration_seconds=55*.9e-6/C0, orders=(0, 0))
    return p.run(q, numerics=n)


def test_incident_power_is_the_flux_through_a_matched_cell():
    """The closed form against the field flux through a plane below an empty cell.

    The band fixes k_par at its centre, so each wavelength meets the cell at its own angle
    (0.283, 0.300 and 0.317 rad here): ``incident_power`` must follow it. The declared angle at
    every wavelength would be 5e-3 off at the band's ends, which the flux's own discretization
    error (common to all three wavelengths) cannot hide in the ratio across the band.
    """
    coarse, fine = matched(1/30), matched(1/60)
    t = coarse.transmission
    incident = t.reference['incident_power']
    assert t.reference['incident_power_units'] == t.reference['power_units'] == 'W'
    sines = np.sin(.3)*np.array([.85, .9, .95])/.9
    np.testing.assert_allclose(incident, .5/Z0*np.sqrt(1-sines**2)*(.2e-6)**2, rtol=1e-14)
    gaps, across_gaps = [], []
    for r in (coarse, fine):
        flux = -r.plane_power_z['below']
        gaps.append(np.max(np.abs(flux/incident-1)))
        across = flux/flux[1]
        across_gaps.append(np.max(np.abs(across/(incident/incident[1])-1)))
        assert np.max(np.abs(across-1)) > 3e-3        # the declared angle would give 1 here
    # Measured: 7.2e-3 and 2.0e-3 against the closed form, 1.6e-3 and 5.6e-4 across the band.
    assert gaps[0] < 1e-2 and gaps[1] < gaps[0]/2
    assert across_gaps[0] < 2e-3 and across_gaps[1] < across_gaps[0]
    np.testing.assert_allclose(t.power, t.efficiency*incident[:, None, None], rtol=1e-15)


@pytest.mark.parametrize('algorithm', ['shared', 'reference'])
def test_a_band_past_the_light_line_has_no_incident_power(algorithm):
    """At 0.8 rad a band of 0.5 to 1.5 um has no propagating incident wave at 1.5 um.

    The shared route refuses the band; the reference route runs it, and there the incident
    power, the power and the efficiency are NaN with both flags False -- never a zero, and
    never an efficiency divided by one. The amplitude ratios are left as the engine computes them.
    """
    q = p.Problem(units=Units(length='um'),
        stack=p.Stack(period=(.2,), top=Material(1.), bottom=Material(1.),
            layers=(p.Layer(thickness=.1, epsilon=np.full((1, 2), 2.25)),)),
        excitation=p.Excitation(wavelength=(.5, .9, 1.5), sources=(p.PlaneWave(theta=.8),)))
    n = p.Numerics(algorithm=algorithm, spacing=1/30, duration_seconds=55*.9e-6/C0, orders=(0, 0))
    report = p.check(q, numerics=n)
    if algorithm == 'shared':
        assert not report.ok
        assert any('must propagate throughout the wavelength band' in i.message for i in report.issues)
        return
    assert report.ok, report.issues
    with pytest.warns(Warning):
        r = p.run(q, numerics=n)
    for orders in (r.reflection, r.transmission):
        for flag in ('power_defined', 'efficiency_defined'):
            np.testing.assert_array_equal(orders.metadata[flag], (True, True, False))
        assert np.isnan(orders.reference['incident_power'][2])
        assert np.isnan(orders.power[2]).all() and np.isnan(orders.efficiency[2]).all()
        assert np.isfinite(orders.reference['incident_power'][:2]).all()
        assert np.isfinite(orders.power[:2]).all() and np.isfinite(orders.efficiency[:2]).all()
        assert orders.amplitude_ratio_s is not None
        assert np.isfinite(orders.amplitude_ratio_s).all()


def test_jit_and_grad_through_a_traced_drive():
    """Under jax.jit with a traced real drive ``a``: the ratios do not move with ``a`` (a zero
    derivative), the fields move as ``a`` and the power as ``a**2``."""
    jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    q, n = slab()
    held = p.prepare(q, numerics=n, backend='jax')

    def forward(a):
        return held.run(q.replace(excitation=q.excitation.replace(
            sources=(q.excitation.sources[0].replace(amplitude=a),))))
    eager, jitted = forward(1.3), jax.jit(forward)(1.3)
    for name in ('amplitude_ratio_s', 'field_s', 'power'):
        assert getattr(jitted.transmission, name) is not None
        np.testing.assert_allclose(getattr(jitted.transmission, name),
                                   getattr(eager.transmission, name), rtol=5e-12)
    unit = forward(1.).transmission
    grad = jax.grad(lambda a: jnp.sum(forward(a).transmission.power))(1.3)
    np.testing.assert_allclose(grad, 2*1.3*np.sum(unit.power), rtol=1e-10)
    # field_s is the ratio times the traced drive: its derivative is the ratio itself.
    field = jax.grad(lambda a: jnp.real(jnp.sum(forward(a).transmission.field_s)))(1.3)
    np.testing.assert_allclose(field, np.real(np.sum(unit.amplitude_ratio_s)), rtol=1e-10)
    ratio = jax.grad(lambda a: jnp.real(jnp.sum(forward(a).transmission.amplitude_ratio_s)))(1.3)
    assert ratio == 0.


def test_torch_carries_the_drive_on_its_own_arrays():
    torch = pytest.importorskip('torch')
    q, n = slab(snapshots=True)
    a = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
    live = q.replace(excitation=q.excitation.replace(
        sources=(q.excitation.sources[0].replace(amplitude=a),)))
    r = p.prepare(q, numerics=n, backend='torch').run(live)
    t = r.transmission
    assert t.reference['incident_field_units'] == 'V/m'
    for value in (t.power, t.reference['incident_power'], t.reference['incident_field'], t.field_s):
        assert isinstance(value, torch.Tensor)
    t.power.sum().backward()
    numpy_run = p.run(q, numerics=n)
    unit = numpy_run.transmission.power
    np.testing.assert_allclose(a.grad.item(), 2*1.3*np.sum(unit), rtol=1e-10)
    sample = r.sample(z=-.2)
    assert isinstance(sample.values, np.ndarray)
    # The sample carries the Torch drive: 1.3 times the NumPy unit-drive sample. The two
    # backends' unit-drive snapshots agree to about 2e-8 of their peak, not to round-off, so
    # the comparison allows 1e-6 of the peak; a dropped drive would be 30% off.
    expected = 1.3*numpy_run.sample(z=-.2).values
    assert np.max(np.abs(expected)) > 0
    np.testing.assert_allclose(sample.values, expected, rtol=1e-6,
                               atol=1e-6*np.max(np.abs(expected)))
