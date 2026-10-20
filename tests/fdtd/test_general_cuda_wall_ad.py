"""Public PEC/PMC projection, live derivatives and prepared ownership.

The projection oracle explicitly marks physical Yee node layers; it does not
read implementation masks. Radiation and cavity-mode accuracy belong to their
independent gates. CUDA cases require actual public admission and hardware.
"""
import numpy as np
import pytest

from wavebit import Material, Units, fdtd
from wavebit.errors import IncompatibleProblemError
from wavebit.units import Z0


UNITS = Units(length='um')
DX = .05
STEPS = 41
SHAPE = (3, 8, 8, 8)


@pytest.fixture(params=['cpu', 'cuda:0'])
def torch_device(request):
    torch = pytest.importorskip('torch')
    if request.param.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('requires CUDA-enabled Torch and an NVIDIA GPU')
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


def walls(case):
    if case == 'pec':
        return fdtd.Boundaries(x=fdtd.PEC(), y=fdtd.PEC(), z=fdtd.PEC())
    if case == 'pmc':
        return fdtd.Boundaries(x=fdtd.PMC(), y=fdtd.PMC(), z=fdtd.PMC())
    pair = (fdtd.PMC(), fdtd.PEC()) if case == 'ghost' else (fdtd.PEC(), fdtd.PMC())
    return fdtd.Boundaries(**{axis: pair if case in ('mixed', 'ghost', axis)
                             else fdtd.Periodic() for axis in 'xyz'})


def patterns():
    rng = np.random.default_rng(2417)
    # Every raw sample is nonzero, including the physical wall layers.
    return rng.uniform(.2, 1., SHAPE), rng.uniform(.2, 1., SHAPE) / Z0


def projected(electric, magnetic, boundary):
    e, h = electric.copy(), magnetic.copy()
    for axis, name in enumerate('xyz'):
        # On this Yee layout PEC-low and PMC-high are stored node layers.
        # PEC-high and PMC-low act through exterior ghosts, not extra masks.
        for side, kind, index in (('-', fdtd.PEC, 0), ('+', fdtd.PMC, -1)):
            if not isinstance(boundary.face(name + side), kind):
                continue
            for component in range(3):
                selector = [component, slice(None), slice(None), slice(None)]
                selector[3 - axis] = index
                target = e if (component != axis) == (side == '-') else h
                target[tuple(selector)] = 0.
    return e, h


def numerics(block=None, steps=STEPS, open_padding=False):
    return fdtd.Numerics(grid=fdtd.UniformGrid(spacing=DX),
        duration=fdtd.Duration(steps=steps), checkpoint_steps=block,
        absorber=fdtd.PML(cells=4, wavelength=2., allow_thin=True)
        if open_padding else fdtd.PML())


def initial_problem(case='mixed', amplitude=1.):
    e, h = patterns()
    return fdtd.Problem(units=UNITS, scene=fdtd.Scene(
        domain=fdtd.Domain(size=(8 * DX,) * 3),
        background=Material(epsilon=2.25, mu=1.4), boundaries=walls(case)),
        initial=fdtd.InitialField(E=e, H=h, amplitude=amplitude),
        outputs=fdtd.Outputs(monitors=(fdtd.TimeSeriesMonitor('probe', at=(.04, .03, -.02),
                                                 component='Ey'),), final_fields=True))


@pytest.mark.parametrize('case', ['pec', 'pmc', 'mixed', 'ghost', 'x', 'y', 'z'])
def test_public_wall_projects_initial_host_fields_before_first_curl(torch_device, monkeypatch, case):
    from wavebit.fdtd import _execution

    torch, device = torch_device
    problem = initial_problem(case)
    raw = (problem.initial.E.copy(), problem.initial.H.copy())
    expected = projected(*raw, problem.scene.boundaries)
    observed = []
    original = _execution.initial_state

    def capture(plan, params, runtime):
        state = original(plan, params, runtime)
        # Read-only observation: use the public run, never replace its state.
        assert state['E'].device == state['H'].device == torch.device(device)
        np.testing.assert_array_equal(host(state['E']), expected[0])
        # The kernel stores Z0*H; compare in its units to avoid an extra division.
        np.testing.assert_array_equal(host(state['H']), expected[1] * Z0)
        for got, want in zip((host(state['E']), host(state['H'])), (raw[0], raw[1] * Z0)):
            np.testing.assert_array_equal(got[:, 1:-1, 1:-1, 1:-1],
                                          want[:, 1:-1, 1:-1, 1:-1])
            assert np.min(np.abs(got[:, 1:-1, 1:-1, 1:-1])) > 0
        observed.append(True)
        return state

    monkeypatch.setattr(_execution, 'initial_state', capture)
    with torch.no_grad():
        result = fdtd.run(problem, numerics=numerics(steps=1), backend='torch', device=device)
    assert observed == [True]
    assert bool(result.validity.checks['runtime_stability'])
    for got, want in zip((problem.initial.E, problem.initial.H), raw):
        np.testing.assert_array_equal(got, want)


def field_loss(result):
    return ((result.final_fields.values['E'] ** 2).sum()
            + Z0 ** 2 * (result.final_fields.values['H'] ** 2).sum()
            + (result['probe'].values ** 2).sum())


def compare_fields(result, expected, device):
    assert bool(result.validity.checks['runtime_stability'])
    for got, want in [(result.final_fields.values[name], expected.final_fields.values[name])
                      for name in ('E', 'H')] + [(result['probe'].values, expected['probe'].values)]:
        assert got.device.type == device.split(':')[0] and got.dtype.is_floating_point
        parity(got, want)


@pytest.mark.parametrize('case', ['pec', 'pmc', 'mixed'])
@pytest.mark.parametrize('device,leaf_device', [('cpu', 'cpu'), ('cuda:0', 'cpu'), ('cuda:0', 'cuda:0')],
                         ids=['cpu', 'cuda-hostleaf', 'cuda-deviceleaf'])
def test_wall_initial_amplitude_gradients_checkpoints_and_prepared_ownership(case, device, leaf_device):
    torch = pytest.importorskip('torch')
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('requires CUDA-enabled Torch and an NVIDIA GPU')
    assert divmod(STEPS, 7) == (5, 6)
    base = initial_problem(case)
    host_copies = (base.initial.E.copy(), base.initial.H.copy())
    schedules = (None, 7) if device.startswith('cuda') else (None,)
    for block in schedules:
        prepared = fdtd.prepare(base, numerics=numerics(block), backend='torch', device=device)
        saved = None
        for amplitude in (1.3, .7):
            leaf = torch.tensor(amplitude, dtype=torch.float64, device=leaf_device, requires_grad=True)
            live = base.replace(initial=base.initial.replace(amplitude=leaf))
            result = prepared.run(live)
            loss = field_loss(result)
            derivative, = torch.autograd.grad(loss, leaf)
            value = float(host(loss))
            assert np.isfinite(value) and value > 1.
            assert derivative.device == leaf.device and leaf.grad is None
            assert float(host(leaf)) == amplitude and leaf.requires_grad
            np.testing.assert_allclose(float(host(derivative)), 2 * value / amplitude,
                                       rtol=2e-12, atol=0.)
            reference = fdtd.run(initial_problem(case, amplitude), numerics=numerics())
            compare_fields(result, reference, device)
            for step in (1e-5, 5e-6):
                plus = field_loss(fdtd.run(initial_problem(case, amplitude + step), numerics=numerics()))
                minus = field_loss(fdtd.run(initial_problem(case, amplitude - step), numerics=numerics()))
                np.testing.assert_allclose(float(host(derivative)), (plus - minus) / (2 * step),
                                           rtol=1e-7, atol=0.)
            fresh = fdtd.run(initial_problem(case, amplitude), numerics=numerics(block),
                             backend='torch', device=device)
            compare_fields(result, fresh, device)
            if saved is None:
                saved = (result, result.final_fields.values['E'].detach().clone(),
                         result['probe'].values.detach().clone())
            else:
                assert torch.equal(saved[0].final_fields.values['E'], saved[1])
                assert torch.equal(saved[0]['probe'].values, saved[2])
                assert not torch.equal(result['probe'].values, saved[2])
        changed = base.replace(scene=base.scene.replace(boundaries=walls('pmc' if case != 'pmc' else 'pec')))
        assert not prepared.validate(changed).ok
        with pytest.raises(IncompatibleProblemError):
            prepared.run(changed)
    for got, want in zip((base.initial.E, base.initial.H), host_copies):
        np.testing.assert_array_equal(got, want)


def dipole_problem(kind, axis, moment=1.3e-16, phase=.4, open_padding=False):
    pulse = fdtd.GaussianPulse(wavelength=2., frequency_sigma_hz=.08 / UNITS.time_unit_seconds)
    peak = np.sqrt(2 * np.log(1 / pulse.cutoff)) / (2 * np.pi * pulse.frequency_sigma_hz)
    component = ('E' if kind == 'electric' else 'H') + axis
    return fdtd.Problem(units=UNITS, scene=fdtd.Scene(
        domain=fdtd.Domain(size=(8 * DX,) * 3), background=Material(epsilon=2.25, mu=1.4),
        boundaries=fdtd.Boundaries(x=fdtd.PEC(), y=fdtd.PMC(), z=fdtd.Open())
        if open_padding else walls('mixed')),
        sources=(fdtd.CurrentDipole('d', center=(.007, -.011, .013), kind=kind, component=component,
                            pulse=pulse, current_moment=moment, phase=phase,
                            delay_seconds=-peak + .12 * UNITS.time_unit_seconds),),
        outputs=fdtd.Outputs(monitors=(fdtd.TimeSeriesMonitor('probe', at=(.04, .03, -.02),
                                                 component=component),), final_fields=True))


def dipole_loss(result, kind):
    scale = Z0 if kind == 'magnetic' else 1.
    return field_loss(result) + scale ** 2 * (result.source_references['d']['field'] ** 2).sum()


@pytest.mark.parametrize('kind,axis,open_padding', [
    pytest.param(kind, axis, False, id=f'{kind}-{axis}')
    for kind in ('electric', 'magnetic') for axis in 'xyz'
] + [pytest.param(kind, 'y', True, id=f'open_pml-{kind}-y')
     for kind in ('electric', 'magnetic')])
def test_wall_interior_dipole_fields_source_work_and_live_gradients(
        torch_device, axis, kind, open_padding):
    torch, device = torch_device

    def physical(moment=1.3e-16, phase=.4):
        return dipole_problem(kind, axis, moment, phase, open_padding)

    def settings(block=None):
        return numerics(block, open_padding=open_padding)

    schedules = ((7,) if open_padding else (None, 7)) if device.startswith('cuda') else (None,)
    for block in schedules:
        prepared = fdtd.prepare(physical(), numerics=settings(block),
                                backend='torch', device=device)
        saved = None
        for moment, phase in ((1.3e-16, .4), (.9e-16, .6)):
            m = torch.tensor(moment, dtype=torch.float64, device='cpu', requires_grad=True)
            p = torch.tensor(phase, dtype=torch.float64, device=device, requires_grad=True)
            live = physical(m, p)
            result = prepared.run(live)
            loss = dipole_loss(result, kind)
            gradients = torch.autograd.grad(loss, (m, p))
            value = float(host(loss))
            assert np.isfinite(value) and value > 0
            for gradient, leaf, original in zip(gradients, (m, p), (moment, phase)):
                assert gradient.device == leaf.device and leaf.grad is None and leaf.requires_grad
                assert float(host(leaf)) == original
            np.testing.assert_allclose(float(host(gradients[0])), 2 * value / moment,
                                       rtol=1e-12, atol=0.)
            actual = np.asarray([float(host(gradients[0])) * moment, float(host(gradients[1]))])
            assert np.all(np.abs(actual) > 1e-5 * value)
            for fraction in (1e-5, 5e-6):
                expected = []
                for dm, dp in ((moment * fraction, 0.), (0., fraction)):
                    plus = fdtd.run(physical(moment + dm, phase + dp), numerics=settings())
                    minus = fdtd.run(physical(moment - dm, phase - dp), numerics=settings())
                    expected.append((dipole_loss(plus, kind) - dipole_loss(minus, kind)) / (2 * fraction))
                np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-10 * value)
            reference = fdtd.run(physical(moment, phase), numerics=settings())
            compare_fields(result, reference, device)
            history = result.source_references['d']
            for key in ('drive', 'field', 'delivered_power'):
                parity(history[key], reference.source_references['d'][key])
                assert history[key].device == torch.device(device)
            fresh = fdtd.run(physical(moment, phase), numerics=settings(block),
                             backend='torch', device=device)
            compare_fields(result, fresh, device)
            for key in ('drive', 'field', 'delivered_power'):
                parity(history[key], fresh.source_references['d'][key])
            field = host(history['field'])
            previous = np.concatenate((np.zeros(1), field[:-1]))
            np.testing.assert_allclose(host(history['delivered_power']),
                -host(history['drive']) * (field + previous) / 2, rtol=1e-13, atol=0.)
            if saved is None:
                saved = (result, {key: history[key].detach().clone()
                                  for key in ('drive', 'field', 'delivered_power')},
                         result.final_fields.values['E'].detach().clone())
            else:
                for key, copy in saved[1].items():
                    assert torch.equal(saved[0].source_references['d'][key], copy)
                assert torch.equal(saved[0].final_fields.values['E'], saved[2])
                assert not torch.equal(history['drive'], saved[1]['drive'])
