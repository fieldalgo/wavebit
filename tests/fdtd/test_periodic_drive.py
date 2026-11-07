"""fdtd.periodic's diffraction amplitudes describe the cell; the drive moves power and fields.

fieldalgo/wavebit#54: ``DiffractionOrders.amplitude_s/_p`` from ``fdtd.periodic.run`` were the
engine's unit-drive ratios multiplied by the source amplitude ``A`` and labelled ``'V/m'``,
while ``rcwa.run`` and ``fdtd.run`` publish the ratio itself. They are the ratio now, like a
Fresnel coefficient, and the drive is carried by what has units: ``incident_amplitude`` (V/m),
the new ``power`` and ``reference['incident_power']`` (W, or W/m across an invariant y), the
planes, the snapshots and ``plane_power_z``.
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
    assert set(one.basis) == set(two.basis)
    pairs += [(a, b) for leg in one.basis for a, b in zip(one.basis[leg], two.basis[leg])]
    for a, b in pairs:
        for name in ('amplitude_s', 'amplitude_p', 'efficiency'):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
        assert a.reference['amplitude_units'] == '1'
        np.testing.assert_allclose(b.power, 4*a.power, rtol=1e-14)
        np.testing.assert_allclose(b.reference['incident_power'], 4*a.reference['incident_power'],
                                   rtol=1e-14)
        assert b.reference['incident_amplitude'] == 2j
    np.testing.assert_allclose(two.plane_power_z['outside'], 4*one.plane_power_z['outside'],
                               rtol=1e-14)
    np.testing.assert_allclose(two.planes['outside'].values, 2j*one.planes['outside'].values,
                               rtol=1e-14)
    for name in one.snapshots:
        np.testing.assert_allclose(two.snapshots[name].values, 2j*one.snapshots[name].values,
                                   rtol=1e-14, atol=0)


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
    never an efficiency divided by one. The amplitudes are left as the engine computes them.
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
        assert np.isfinite(orders.amplitude_s).all()


def test_jit_and_grad_through_a_traced_drive():
    """Under jax.jit with a traced real drive ``a``: the ratios do not move with ``a`` (a zero
    derivative), the power moves as ``a**2``."""
    jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    q, n = slab()
    held = p.prepare(q, numerics=n, backend='jax')

    def forward(a):
        return held.run(q.replace(excitation=q.excitation.replace(
            sources=(q.excitation.sources[0].replace(amplitude=a),))))
    eager, jitted = forward(1.3), jax.jit(forward)(1.3)
    for name in ('amplitude_s', 'power'):
        np.testing.assert_allclose(getattr(jitted.transmission, name),
                                   getattr(eager.transmission, name), rtol=5e-12)
    unit = forward(1.).transmission.power
    grad = jax.grad(lambda a: jnp.sum(forward(a).transmission.power))(1.3)
    np.testing.assert_allclose(grad, 2*1.3*np.sum(unit), rtol=1e-10)
    ratio = jax.grad(lambda a: jnp.real(jnp.sum(forward(a).transmission.amplitude_s)))(1.3)
    assert ratio == 0.


def test_torch_carries_the_drive_on_its_own_arrays():
    torch = pytest.importorskip('torch')
    q, n = slab(snapshots=True)
    a = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
    live = q.replace(excitation=q.excitation.replace(
        sources=(q.excitation.sources[0].replace(amplitude=a),)))
    r = p.prepare(q, numerics=n, backend='torch').run(live)
    t = r.transmission
    for value in (t.power, t.reference['incident_power'], t.reference['incident_amplitude']):
        assert isinstance(value, torch.Tensor)
    t.power.sum().backward()
    unit = p.run(q, numerics=n).transmission.power
    np.testing.assert_allclose(a.grad.item(), 2*1.3*np.sum(unit), rtol=1e-10)
    sample = r.sample(z=-.2)
    assert isinstance(sample.values, np.ndarray)
