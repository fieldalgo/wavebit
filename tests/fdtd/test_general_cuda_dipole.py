"""Fixed-center public Dipoles: independent SI moments and bounded CUDA AD.

The one-step moment/centroid oracle does not use the source stamping code.
Small PML/slab trajectories exercise device ownership and checkpoint replay;
their thin padding and unfinished pulses are not radiation-accuracy evidence.
Position derivatives, wrapping stamps and sources in PML remain unsupported.
"""
import numpy as np
import pytest

from wavebit import Material, Units, fdtd
from wavebit.errors import IncompatibleProblemError
from wavebit.units import C0, EPSILON0, Z0


UNITS = Units(length='um')
DX = .05
MU0 = 1 / (EPSILON0 * C0 ** 2)
CENTER = (.013, -.021, .007)
AD_CENTER = (-.031, .017, -.06)
STEPS = 41


@pytest.fixture(params=['cpu', 'cuda:0'])
def torch_device(request):
    torch = pytest.importorskip('torch')
    if request.param.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('requires a CUDA-enabled Torch installation and CUDA device')
    return torch, request.param


def host(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def parity(actual, expected):
    actual, expected = host(actual), host(expected)
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    peak = np.max(np.abs(expected), initial=0.)
    error = np.max(np.abs(actual - expected), initial=0.)
    assert error == 0 if peak == 0 else error / peak < 5e-13


def pulse():
    return fdtd.GaussianPulse(wavelength=2., frequency_sigma_hz=.08 / UNITS.time_unit_seconds)


def pulse_peak_seconds():
    p = pulse()
    return np.sqrt(2 * np.log(1 / p.cutoff)) / (2 * np.pi * p.frequency_sigma_hz)


def periodic_problem(kind='electric', component='Ey', stamp='bspline', center=CENTER):
    return fdtd.Problem(units=UNITS, scene=fdtd.Scene(
        domain=fdtd.Domain(size=(16 * DX,) * 3),
        background=Material(epsilon=2.25, mu=1.4),
        boundaries=fdtd.Boundaries.periodic(z=fdtd.Periodic())),
        sources=(fdtd.CurrentDipole('d', center=center, component=component, kind=kind, stamp=stamp,
                            current_moment=1.7e-16, phase=np.pi / 2, pulse=pulse()),),
        outputs=fdtd.Outputs(final_fields=True))


def numerics(steps=STEPS, checkpoint=None):
    return fdtd.Numerics(grid=fdtd.UniformGrid(spacing=DX),
        duration=fdtd.Duration(steps=steps), checkpoint_steps=checkpoint,
        absorber=fdtd.PML(cells=4, wavelength=2., allow_thin=True))


@pytest.mark.parametrize('kind', ['electric', 'magnetic'])
@pytest.mark.parametrize('axis', 'xyz')
@pytest.mark.parametrize('stamp', ['bspline', 'tent'])
def test_public_dipole_one_step_si_moment_and_yee_centroid(torch_device, kind, axis, stamp):
    torch, device = torch_device
    field = 'E' if kind == 'electric' else 'H'
    problem = periodic_problem(kind, field + axis, stamp)
    prepared = fdtd.prepare(problem, numerics=numerics(1), backend='torch', device=device)
    dt = prepared.contract['resolved_time_step_seconds']
    first = .5 * dt if kind == 'electric' else 0.
    source = problem.sources[0].replace(delay_seconds=first - pulse_peak_seconds())
    problem = problem.replace(sources=(source,))
    with torch.no_grad():
        result = prepared.run(problem)
    value = result.final_fields.values[field]
    assert value.device == torch.device(device) and value.dtype == torch.float64
    assert bool(result.validity.checks['runtime_stability'])
    component = 'xyz'.index(axis)
    density = host(value[component])
    coefficient = EPSILON0 * 2.25 if field == 'E' else MU0 * 1.4
    # E1=-dt*J/(epsilon0*epsilon), H1/2=-dt*M/(mu0*mu).
    moment_per_node = -density * coefficient * (DX * UNITS.length_unit_meters) ** 3 / dt
    moment = moment_per_node.sum()
    np.testing.assert_allclose(moment, source.current_moment, rtol=1e-13, atol=0.)
    np.testing.assert_allclose(host(result.source_references['d']['drive'])[0], source.current_moment,
                               rtol=1e-14, atol=0.)
    assert np.max(np.abs(host(value)[[c for c in range(3) if c != component]])) == 0
    # Independent Yee geometry: E_c is half a cell along c, H_c along the
    # other two axes. Do not read stamp weights or solver coordinate metadata.
    offsets = tuple(.5 * ((a == component) if field == 'E' else (a != component))
                    for a in range(3))
    axes = [-8 * DX + (np.arange(16) + offset) * DX for offset in offsets]
    z, y, x = np.meshgrid(axes[2], axes[1], axes[0], indexing='ij')
    centroid = [(moment_per_node * coordinate).sum() / moment for coordinate in (x, y, z)]
    np.testing.assert_allclose(centroid, CENTER, rtol=0., atol=1e-13)


def slab_problem(kind='electric', stamp='bspline', moment=1.3e-16, phase=.4,
                 delay=None, epsilon=2.25):
    if delay is None:
        delay = -pulse_peak_seconds() + .12 * UNITS.time_unit_seconds
    component = 'Ey' if kind == 'electric' else 'Hy'
    return fdtd.Problem(units=UNITS, scene=fdtd.Scene(
        domain=fdtd.Domain(size=(8 * DX, 8 * DX, 12 * DX)),
        background=Material(epsilon=1., mu=1.2),
        boundaries=fdtd.Boundaries.periodic(z=fdtd.Open()),
        structures=(fdtd.Structure('film', shape=fdtd.SnappedSlab(z=(.1, .25)),
                                   material=Material(epsilon=epsilon, mu=1.2)),)),
        sources=(fdtd.CurrentDipole('d', center=AD_CENTER, component=component, kind=kind,
                            stamp=stamp, pulse=pulse(), current_moment=moment,
                            phase=phase, delay_seconds=delay),),
        outputs=fdtd.Outputs(monitors=(
            fdtd.TimeSeriesMonitor('probe', at=(.05, -.03, -.025), component=component),
            fdtd.FieldMonitor('near', plane=fdtd.Plane(z=0.), wavelengths=(1.8, 2.),
                       components=('Ey', 'Hy')),
            fdtd.FluxMonitor('flux', plane=fdtd.Plane(z=.15), wavelengths=(1.8, 2.))), final_fields=True))


def assert_source_contract(result, problem):
    source, history = problem.sources[0], result.source_references['d']
    magnetic = source.kind == 'magnetic'
    dt = result.provenance['time_step_seconds']
    times = (np.arange(STEPS) + (0. if magnetic else .5)) * dt
    field_times = (np.arange(STEPS) + (.5 if magnetic else 1.)) * dt
    np.testing.assert_allclose(host(history['time']), times, rtol=2e-14, atol=0.)
    np.testing.assert_allclose(host(history['field_time']), field_times, rtol=2e-14, atol=0.)
    tau = 1 / (2 * np.pi * source.pulse.frequency_sigma_hz)
    shifted = times - pulse_peak_seconds() - float(host(source.delay_seconds))
    expected = (float(host(source.current_moment)) * np.exp(-.5 * (shifted / tau) ** 2)
                * np.sin(2 * np.pi * C0 / (source.pulse.wavelength * UNITS.length_unit_meters)
                         * shifted + float(host(source.phase))))
    np.testing.assert_allclose(host(history['drive']), expected, rtol=1e-13, atol=0.)
    assert history['drive_units'] == ('V*m' if magnetic else 'A*m')
    assert history['field_units'] == ('A/m' if magnetic else 'V/m')
    assert history['power_units'] == 'W' and history['time_unit'] == 's'
    field = host(history['field'])
    previous = np.concatenate((np.zeros(1, dtype=field.dtype), field[:-1]))
    expected_power = -host(history['drive']) * (field + previous) / 2
    np.testing.assert_allclose(host(history['delivered_power']), expected_power,
                               rtol=1e-13, atol=0.)


def compare_results(result, reference, device):
    assert bool(result.validity.checks['runtime_stability'])
    arrays = [(result.final_fields.values[f], reference.final_fields.values[f]) for f in ('E', 'H')]
    arrays += [(result['probe'].values, reference['probe'].values),
               (result['flux'].spectral_energy, reference['flux'].spectral_energy)]
    arrays += [(result['near'].values[f], reference['near'].values[f]) for f in ('Ey', 'Hy')]
    arrays += [(result.source_references['d'][key], reference.source_references['d'][key])
               for key in ('drive', 'field', 'delivered_power')]
    for actual, expected in arrays:
        assert actual.device.type == device.split(':')[0]
        parity(actual, expected)


def objective(result, kind):
    fields = result.final_fields.values
    scale = Z0 if kind == 'magnetic' else 1.
    # A quadratic of final fields AND the live source-field history exercises
    # both routes, including a history carried across checkpoint boundaries.
    return ((fields['E'] ** 2).sum() + Z0 ** 2 * (fields['H'] ** 2).sum()
            + scale ** 2 * (result.source_references['d']['field'] ** 2).sum())


def live_solve(torch, device, prepared, kind, stamp, values):
    # Moment and delay start on CPU; phase/material on the execution device.
    leaves = tuple(torch.tensor(v, dtype=torch.float64, device=where, requires_grad=True)
                   for v, where in zip(values, ('cpu', device, 'cpu', device)))
    snapshots = tuple(leaf.detach().clone() for leaf in leaves)
    problem = slab_problem(kind, stamp, *leaves)
    result = prepared.run(problem)
    loss = objective(result, kind)
    derivatives = torch.autograd.grad(loss, leaves)
    for leaf, derivative, snapshot in zip(leaves, derivatives, snapshots):
        assert leaf.is_leaf and leaf.grad is None
        assert torch.equal(leaf, snapshot)
        assert derivative.device == leaf.device and derivative.dtype == leaf.dtype
    return problem, result, float(host(loss)), np.asarray([float(host(g)) for g in derivatives])


@pytest.mark.parametrize('kind', ['electric', 'magnetic'])
@pytest.mark.parametrize('stamp', ['bspline', 'tent'])
def test_public_dipole_device_fields_histories_and_mixed_leaf_gradients(torch_device, kind, stamp):
    torch, device = torch_device
    base = slab_problem(kind, stamp)
    values = np.asarray([1.3e-16, .4, base.sources[0].delay_seconds, 2.25])
    prepared = fdtd.prepare(base, numerics=numerics(), backend='torch', device=device)
    problem, result, loss, gradient = live_solve(torch, device, prepared, kind, stamp, values)
    assert_source_contract(result, problem)
    reference = fdtd.run(base, numerics=numerics())
    compare_results(result, reference, device)
    assert np.isfinite(loss) and loss > 0 and np.isfinite(gradient).all()
    np.testing.assert_allclose(gradient[0], 2 * loss / values[0], rtol=1e-12, atol=0.)
    # Work in dimensionless perturbation coordinates for the SI moment/delay.
    # Preserve the existing stage-2 budget: 1e-7*|g_fd| + 1e-10*|loss|.
    scales = np.asarray([values[0], 1., UNITS.time_unit_seconds, 1.])
    assert np.all(np.abs(gradient * scales) > 1e-5 * loss)
    for fraction in (1e-5, 5e-6):
        secants = []
        for index, scale in enumerate(scales):
            offset = np.zeros(4)
            offset[index] = fraction * scale
            plus = fdtd.run(slab_problem(kind, stamp, *(values + offset)), numerics=numerics())
            minus = fdtd.run(slab_problem(kind, stamp, *(values - offset)), numerics=numerics())
            secants.append((float(objective(plus, kind)) - float(objective(minus, kind)))
                           / (2 * fraction))
        np.testing.assert_allclose(gradient * scales, secants, rtol=1e-7, atol=1e-10 * loss)


@pytest.mark.parametrize('kind', ['electric', 'magnetic'])
@pytest.mark.parametrize('stamp', ['bspline', 'tent'])
def test_cuda_dipole_checkpoint_tail_and_fresh_reuse_own_source_histories(kind, stamp):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('requires a CUDA-enabled Torch installation and CUDA device')
    device = 'cuda:0'
    base = slab_problem(kind, stamp)
    assert divmod(STEPS, 7) == (5, 6)
    plain = fdtd.prepare(base, numerics=numerics(), backend='torch', device=device)
    scheduled = fdtd.prepare(base, numerics=numerics(checkpoint=7), backend='torch', device=device)
    saved = None
    for moment, phase, epsilon in ((1.3e-16, .4, 2.25), (.9e-16, .55, 2.4)):
        values = (moment, phase, base.sources[0].delay_seconds, epsilon)
        p, result, loss, gradient = live_solve(torch, device, scheduled, kind, stamp, values)
        _, reference, expected_loss, expected_gradient = live_solve(
            torch, device, plain, kind, stamp, values)
        assert result.provenance['checkpoint_steps'] == 7
        assert_source_contract(result, p)
        compare_results(result, reference, device)
        np.testing.assert_allclose(loss, expected_loss, rtol=5e-13, atol=0.)
        np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-12, atol=0.)
        fresh = fdtd.run(slab_problem(kind, stamp, *values), numerics=numerics(checkpoint=7),
                         backend='torch', device=device)
        compare_results(result, fresh, device)
        if saved is None:
            saved = (result, {key: result.source_references['d'][key].detach().clone()
                             for key in ('drive', 'field', 'delivered_power')},
                     result.final_fields.values['E'].detach().clone())
        else:
            for key, value in saved[1].items():
                assert torch.equal(saved[0].source_references['d'][key], value)
            assert torch.equal(saved[0].final_fields.values['E'], saved[2])
            assert not torch.equal(result.source_references['d']['drive'], saved[1]['drive'])


@pytest.mark.parametrize('invalid, pattern', [
    ('center_leaf', 'center'), ('stamp', 'stamp'), ('component', 'component'),
    ('outside', 'physical domain'), ('seam', 'periodic seam'), ('pml_support', 'PML')])
def test_public_dipole_invalid_static_or_support_inputs_refuse(torch_device, invalid, pattern):
    torch, device = torch_device
    problem = periodic_problem()
    source = problem.sources[0]
    if invalid == 'center_leaf':
        source = source.replace(center=(torch.tensor(.013, dtype=torch.float64, requires_grad=True),
                                        CENTER[1], CENTER[2]))
    elif invalid == 'stamp':
        source = source.replace(stamp='trilinear')
    elif invalid == 'component':
        source = source.replace(component='Hy')
    elif invalid == 'outside':
        source = source.replace(center=(.5, 0., 0.))
    elif invalid == 'seam':
        source = source.replace(center=(.39, 0., 0.))
    else:
        source = source.replace(center=(0., 0., .39))
        problem = problem.replace(scene=problem.scene.replace(
            boundaries=fdtd.Boundaries.periodic(z=fdtd.Open())))
    problem = problem.replace(sources=(source,))
    assert not fdtd.check(problem, numerics=numerics(1), backend='torch', device=device).ok
    with pytest.raises(ValueError, match=pattern):
        fdtd.prepare(problem, numerics=numerics(1), backend='torch', device=device)


def test_public_dipole_prepared_position_and_stamp_are_static(torch_device):
    _, device = torch_device
    problem = periodic_problem()
    prepared = fdtd.prepare(problem, numerics=numerics(1), backend='torch', device=device)
    for source in (problem.sources[0].replace(center=(.023, -.021, .007)),
                   problem.sources[0].replace(stamp='tent')):
        changed = problem.replace(sources=(source,))
        assert not prepared.validate(changed).ok
        with pytest.raises(IncompatibleProblemError):
            prepared.run(changed)
