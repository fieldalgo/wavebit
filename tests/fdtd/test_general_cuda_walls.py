"""Public PEC/PMC walls against independent discrete cavity solutions.

The 16^3, 300-step cavity budgets are unchanged from test_fd_pec/test_fd_pmc:
1e-11 relative recurrence and 1e-10 relative phase/amplitude. Small mixed-wall
cavities isolate both faces of each Cartesian axis without a symmetry boundary.
All coordinates and wall positions in the oracle come from the Yee geometry,
not the prepared plan. CUDA admission is never bypassed by these tests.
"""
import math

import numpy as np
import pytest

from wavebit import Units, fdtd
from wavebit.units import Z0
from tests.fixtures.fdtd.fd_helpers import CavityMode, discrete_omega, recurrence_residual


pytestmark = pytest.mark.slow
UNITS = Units(length='um')
DX = 1 / 16
DT = .5 * DX / math.sqrt(3)
STEPS = 300
RECURRENCE_TOL = 1e-11
FIELD_TOL = 1e-10
MUTANT_FLOOR = 1e-4


@pytest.fixture(params=['cpu', 'cuda:0'])
def device(request):
    torch = pytest.importorskip('torch')
    if request.param.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('requires a CUDA-enabled Torch installation and CUDA device')
    return torch, request.param


def host(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def yee_axes(shape, field, component):
    # shape is xyz; field arrays are component,z,y,x.
    return tuple((-n / 2 + np.arange(n) + .5 * (
        axis == component if field == 'E' else axis != component)) * DX
        for axis, n in enumerate(shape))


def mesh(shape, field, component):
    x, y, z = yee_axes(shape, field, component)
    z, y, x = np.meshgrid(z, y, x, indexing='ij')
    return x, y, z


def probe_at(shape, field, component):
    axes = yee_axes(shape, field, component)
    indices = tuple(n // 2 + (1 if axis == 0 else -1 if axis == 2 else 0)
                    for axis, n in enumerate(shape))
    return tuple(a[index] for a, index in zip(axes, indices))


def public_problem(shape, boundaries, electric, magnetic, components):
    return fdtd.Problem(units=UNITS, scene=fdtd.Scene(
        domain=fdtd.Domain(size=tuple(n * DX for n in shape)), boundaries=boundaries),
        initial=fdtd.InitialField(E=electric, H=magnetic / Z0),
        outputs=fdtd.Outputs(final_fields=True, monitors=tuple(
            fdtd.TimeSeriesMonitor(field + 'xyz'[c], component=field + 'xyz'[c],
                        at=probe_at(shape, field, c)) for field, c in components)))


def run(device, problem, steps=STEPS):
    torch, name = device
    print(dict(device=name, boundaries=repr(problem.scene.boundaries), steps=steps,
               spacing=DX, courant=.5), flush=True)
    numerics = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=DX),
                             duration=fdtd.Duration(steps=steps), courant=.5)
    with torch.no_grad():
        result = fdtd.run(problem, numerics=numerics, backend='torch', device=name)
    assert bool(result.validity.checks['runtime_stability'])
    assert result.provenance['time_step_seconds'] == pytest.approx(
        DT * UNITS.time_unit_seconds, rel=2e-15, abs=0.)
    for field in ('E', 'H'):
        value = result.final_fields.values[field]
        assert value.device == torch.device(name) and value.dtype == torch.float64
        assert np.isfinite(host(value)).all()
    return result


def assert_trace(result, field, c, expected, omega, wrong_omega, device):
    torch, name = device
    series = result[field + 'xyz'[c]]
    assert series.values.device == torch.device(name) and series.values.dtype == torch.float64
    values = host(series.values) * (Z0 if field == 'H' else 1.)
    times = (np.arange(STEPS) + (1. if field == 'E' else .5)) * DT
    np.testing.assert_allclose(host(series.times_seconds), times * UNITS.time_unit_seconds,
                               rtol=2e-14, atol=0.)
    peak = np.max(np.abs(expected))
    assert peak > 1e-3 and np.max(np.abs(values)) > 1e-3
    assert np.ptp(values) > .1 * np.max(np.abs(values))
    assert recurrence_residual(values, omega, DT) < RECURRENCE_TOL
    assert np.max(np.abs(values - expected)) < FIELD_TOL * peak
    # A wrong wall location changes the frequency: reject it by the original
    # wall-mutant floor, not by comparing only two backend implementations.
    assert recurrence_residual(values, wrong_omega, DT) > MUTANT_FLOOR


@pytest.mark.parametrize('kind', ['pec', 'pmc'])
def test_public_closed_cavity_has_exact_discrete_phase_and_amplitude(device, kind):
    shape = (16,) * 3
    wall = -.5 if kind == 'pec' else -.5 - DX / 2
    mode = CavityMode((wall,) * 3, (1.,) * 3, (1, 1, 1), DX)
    omega = discrete_omega(mode.k, DX, DT)
    factor = DT / (2 * math.sin(omega * DT / 2))
    arrays = {}
    for field in ('E', 'H'):
        primary = field == ('E' if kind == 'pec' else 'H')
        arrays[field] = np.stack([
            (mode.component(c) if primary else mode.curl_component(c))(*mesh(shape, field, c))
            for c in range(3)])
    if kind == 'pec':
        initial_e, initial_h = arrays['E'], DT / 2 * arrays['H']
        expected_e = arrays['E'] * math.cos(omega * STEPS * DT)
        expected_h = -factor * arrays['H'] * math.sin(omega * (STEPS - .5) * DT)
    else:
        initial_e = np.zeros_like(arrays['E'])
        initial_h = arrays['H'] * math.cos(omega * DT / 2)
        expected_e = factor * arrays['E'] * math.sin(omega * STEPS * DT)
        expected_h = arrays['H'] * math.cos(omega * (STEPS - .5) * DT)
    face = fdtd.PEC() if kind == 'pec' else fdtd.PMC()
    problem = public_problem(shape, fdtd.Boundaries(x=face, y=face, z=face),
                             initial_e, initial_h,
                             tuple((field, c) for field in ('E', 'H') for c in range(3)))
    result = run(device, problem)
    wrong_omega = discrete_omega((math.pi / (1 + DX), math.pi, math.pi), DX, DT)
    for field in ('E', 'H'):
        times = (np.arange(STEPS) + (1. if field == 'E' else .5)) * DT
        for c in range(3):
            at = probe_at(shape, field, c)
            primary = field == ('E' if kind == 'pec' else 'H')
            expected = (mode.value(c, at) * np.cos(omega * times) if primary else
                (1 if kind == 'pmc' else -1) * factor * mode.curl_value(c, at)
                * np.sin(omega * times))
            assert_trace(result, field, c, expected, omega, wrong_omega, device)
        actual = host(result.final_fields.values[field]) * (Z0 if field == 'H' else 1.)
        expected = expected_e if field == 'E' else expected_h
        assert np.max(np.abs(actual - expected)) < FIELD_TOL * np.max(np.abs(expected))


@pytest.mark.parametrize('axis', range(3), ids=list('xyz'))
@pytest.mark.parametrize('low_kind', ['pec', 'pmc'])
def test_public_mixed_walls_resolve_half_cell_length_on_every_axis(device, axis, low_kind):
    shape = tuple(16 if a == axis else 4 for a in range(3))
    e_component, h_component = (axis + 1) % 3, (axis + 2) % 3
    # A tangential E has a node at the PEC wall and zero derivative at the PMC
    # wall. The two actual wall positions differ by L +/- dx/2 on this layout.
    length = 1 - DX / 2 if low_kind == 'pec' else 1 + DX / 2
    origin = -.5 if low_kind == 'pec' else -.5 - DX / 2
    k = 1.5 * math.pi / length
    discrete_k = 2 * math.sin(k * DX / 2) / DX
    omega = discrete_omega((k, 0., 0.), DX, DT)
    factor = DT / (2 * math.sin(omega * DT / 2))

    def pattern(position):
        argument = k * (position - origin)
        return np.sin(argument) if low_kind == 'pec' else np.cos(argument)

    def derivative(position):
        argument = k * (position - origin)
        return discrete_k * (np.cos(argument) if low_kind == 'pec' else -np.sin(argument))

    electric = np.zeros((3,) + shape[::-1])
    magnetic = np.zeros_like(electric)
    electric[e_component] = pattern(mesh(shape, 'E', e_component)[axis])
    curl_pattern = derivative(mesh(shape, 'H', h_component)[axis])
    magnetic[h_component] = DT / 2 * curl_pattern
    faces = {a: fdtd.Periodic() for a in 'xyz'}
    faces['xyz'[axis]] = ((fdtd.PEC(), fdtd.PMC()) if low_kind == 'pec' else
                          (fdtd.PMC(), fdtd.PEC()))
    problem = public_problem(shape, fdtd.Boundaries(**faces), electric, magnetic,
                             (('E', e_component), ('H', h_component)))
    result = run(device, problem)
    # Keeping a physical-domain length of L, ignoring half-cell placement,
    # must fail. No measured/fitted phase or amplitude enters this oracle.
    wrong_omega = discrete_omega((1.5 * math.pi, 0., 0.), DX, DT)
    for field, c in (('E', e_component), ('H', h_component)):
        times = (np.arange(STEPS) + (1. if field == 'E' else .5)) * DT
        position = probe_at(shape, field, c)[axis]
        expected = (pattern(position) * np.cos(omega * times) if field == 'E' else
                    -factor * derivative(position) * np.sin(omega * times))
        assert_trace(result, field, c, expected, omega, wrong_omega, device)
        full_expected = np.zeros_like(electric)
        full_expected[c] = (electric[c] * math.cos(omega * STEPS * DT) if field == 'E' else
            -factor * curl_pattern * math.sin(omega * (STEPS - .5) * DT))
        actual = host(result.final_fields.values[field]) * (Z0 if field == 'H' else 1.)
        assert np.max(np.abs(actual - full_expected)) < FIELD_TOL * np.max(np.abs(full_expected))


@pytest.mark.parametrize('kind', ['pec', 'pmc'])
@pytest.mark.parametrize('axis', range(3), ids=list('xyz'))
@pytest.mark.parametrize('side', [-1, 1], ids=['low', 'high'])
def test_public_dipole_stamp_cannot_touch_a_conducting_wall(device, kind, axis, side):
    _, name = device
    face = fdtd.PEC() if kind == 'pec' else fdtd.PMC()
    center = [0., 0., 0.]
    center[axis] = side * .49
    problem = fdtd.Problem(units=UNITS, scene=fdtd.Scene(
        domain=fdtd.Domain(size=(1.,) * 3), boundaries=fdtd.Boundaries(x=face, y=face, z=face)),
        sources=(fdtd.CurrentDipole('d', center=tuple(center), component='E'+'xyz'[(axis+1) % 3],
            pulse=fdtd.GaussianPulse(wavelength=1., frequency_sigma_hz=.15/UNITS.time_unit_seconds)),))
    numerics = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=DX), duration=fdtd.Duration(steps=1))
    assert not fdtd.check(problem, numerics=numerics, backend='torch', device=name).ok
    with pytest.raises(ValueError, match=kind.upper() + ' wall'):
        fdtd.prepare(problem, numerics=numerics, backend='torch', device=name)
