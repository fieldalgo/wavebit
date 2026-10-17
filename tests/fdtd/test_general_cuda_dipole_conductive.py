"""Electric Dipole deposition in a conductor against an independent SI update.

From zero initial fields, E1=-dt*J/(epsilon0*epsilon+sigma*dt/2).
Recover the actual deposited current moment and its centroid from E1 without
reading implementation stamp weights, coefficients, or coordinate metadata.
The public conductivity is scalar; epsilon and mu may be diagonal.
"""
import numpy as np
import pytest

from wavebit import Units, fdtd
from wavebit.materials import ConductiveMaterial
from wavebit.units import EPSILON0


UNITS = Units(length='um')
DX = .05
CENTER = (.013, -.021, .007)
MOMENT = 1.7e-16


@pytest.fixture(params=['cpu', 'cuda:0'])
def torch_device(request):
    torch = pytest.importorskip('torch')
    if request.param.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('requires a CUDA-enabled Torch installation and CUDA device')
    return torch, request.param


@pytest.mark.parametrize('axis', 'xyz')
@pytest.mark.parametrize('stamp', ['bspline', 'tent'])
@pytest.mark.parametrize('epsilon,mu,conductivity', [
    pytest.param(2.25, 1.4, 600., id='scalar'),
    pytest.param((2., 2.3, 2.6), (1.2, 1.4, 1.6), 2400., id='diagonal-epsilon-mu'),
])
def test_conductive_dipole_one_step_si_moment_centroid_and_work(
        torch_device, axis, stamp, epsilon, mu, conductivity):
    torch, device = torch_device
    # The fixed Courant policy sets this timestep independently of material.
    dt = .5 * DX * UNITS.time_seconds / np.sqrt(3.)
    pulse = fdtd.GaussianPulse(wavelength=2., frequency_sigma_hz=.08/UNITS.time_seconds)
    peak = np.sqrt(-2*np.log(pulse.cutoff))/(2*np.pi*pulse.frequency_sigma_hz)
    physical = fdtd.Problem(units=UNITS, scene=fdtd.Scene(
        domain=fdtd.Domain(size=(16*DX,)*3),
        boundaries=fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.Periodic()),
        background=ConductiveMaterial(epsilon=epsilon, mu=mu, conductivity=conductivity)),
        sources=(fdtd.Dipole('d', center=CENTER, component='E'+axis, kind='electric',
            current_moment=MOMENT, stamp=stamp, pulse=pulse,
            phase=np.pi/2, delay=dt/2-peak),), outputs=fdtd.Outputs(final_fields=True))
    numerics = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=DX),
        duration=fdtd.Duration(steps=1), courant=.5)
    with torch.no_grad():
        result = fdtd.run(physical, numerics=numerics, backend='torch', device=device)
    np.testing.assert_allclose(result.provenance['dt_seconds'], dt, rtol=2e-15, atol=0.)
    assert bool(result.validity.checks['runtime_stability'])
    for name in ('E', 'H'):
        value = result.final_fields.values[name]
        assert value.device == torch.device(device) and value.dtype == torch.float64
        assert bool(torch.isfinite(value).all())
    electric = result.final_fields.values['E'].detach().cpu().numpy()
    magnetic = result.final_fields.values['H'].detach().cpu().numpy()
    component = 'xyz'.index(axis)
    eps = epsilon[component] if isinstance(epsilon, tuple) else epsilon
    coefficient = EPSILON0*eps + conductivity*dt/2
    cell_volume = (DX*UNITS.length_metres)**3
    deposited = -electric[component]*coefficient*cell_volume/dt
    moment = deposited.sum()
    np.testing.assert_allclose(moment, MOMENT, rtol=1e-13, atol=0.)
    assert np.max(np.abs(magnetic)) == 0.
    assert np.max(np.abs(electric[[c for c in range(3) if c != component]])) == 0.
    # The chosen loss is large enough that omitting its half-step denominator
    # would fail this oracle, rather than disappearing beneath its roundoff budget.
    lossless_moment = (-electric[component]*EPSILON0*eps*cell_volume/dt).sum()
    assert abs(lossless_moment-MOMENT) > 1e-5*MOMENT

    offsets = tuple(.5 if a == component else 0. for a in range(3))
    axes = [-8*DX + (np.arange(16)+offset)*DX for offset in offsets]
    z, y, x = np.meshgrid(axes[2], axes[1], axes[0], indexing='ij')
    centroid = [(deposited*coordinate).sum()/moment for coordinate in (x, y, z)]
    np.testing.assert_allclose(centroid, CENTER, rtol=0., atol=1e-13)

    reference = result.sources['d']
    assert reference['drive_units'] == 'A*m' and reference['field_units'] == 'V/m'
    assert reference['power_units'] == 'W' and reference['time_unit'] == 's'
    for name in ('drive', 'field', 'delivered_power'):
        value = reference[name]
        assert value.device == torch.device(device) and value.dtype == torch.float64
        assert bool(torch.isfinite(value).all())
    np.testing.assert_allclose(reference['time'], [dt/2], rtol=2e-15, atol=0.)
    np.testing.assert_allclose(reference['field_time'], [dt], rtol=2e-15, atol=0.)
    # The normalized deposited moment supplies independent spatial weights.
    # Source work uses the mean of zero E0 and this E1, not E1 alone.
    sampled_field = float(((deposited/MOMENT)*electric[component]).sum())
    power = -MOMENT*sampled_field/2
    assert sampled_field < 0 and power > 0
    np.testing.assert_allclose(reference['drive'].detach().cpu().numpy(), [MOMENT], rtol=1e-14, atol=0.)
    np.testing.assert_allclose(reference['field'].detach().cpu().numpy(), [sampled_field], rtol=1e-13, atol=0.)
    np.testing.assert_allclose(reference['delivered_power'].detach().cpu().numpy(), [power], rtol=1e-13, atol=0.)
