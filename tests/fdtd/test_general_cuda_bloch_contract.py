"""Static Bloch declarations and lifecycle guards on the public Torch route.

These short checks establish device and preparation contracts. Independent
propagation and derivative oracles are in the separate Bloch gates.
"""
import pytest

from wavebit import Execution, Units, fdtd
from wavebit.errors import IncompatibleProblemError, UnsupportedFeatureError, ValidationError
from tests.fixtures.fdtd.fd_helpers import torch_forward_paths


UNITS = Units(length='um')


@pytest.fixture(params=['cpu', 'cuda:0'])
def device(request):
    torch = pytest.importorskip('torch')
    if request.param != 'cpu' and not torch.cuda.is_available():
        pytest.skip('CUDA-enabled Torch and a device required')
    return torch, request.param


def numerics(steps=5):
    return fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.05),
                         duration=fdtd.Duration(steps=steps))


def initial_problem(face=None):
    return fdtd.Problem(units=UNITS, scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.4,) * 3),
        boundaries=fdtd.Boundaries(x=fdtd.Bloch(phase=.3) if face is None else face,
                                   y=fdtd.Periodic(), z=fdtd.Periodic())),
        initial=fdtd.InitialField(E=(0., 1. + .2j, 0.), H=(0., 0., 0.)),
        outputs=fdtd.Outputs(final_fields=True))


def plane_problem():
    source = fdtd.PlaneWave('pw', plane=fdtd.Plane(z=.2), theta=.2,
        pulse=fdtd.GaussianPulse(wavelength=.8, frequency_sigma_hz=2e13))
    return fdtd.Problem(units=UNITS, scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.2, .2, 1.2)),
        boundaries=fdtd.Boundaries(**fdtd.Bloch.from_source(source, units=UNITS),
                                   z=fdtd.Open())), sources=(source,))


@pytest.mark.parametrize('storage', ['python', 'host-tensor'])
def test_static_phase_and_wavevector_give_the_same_complex_run(device, storage):
    torch, name = device
    def scalar(value):
        return value if storage == 'python' else torch.tensor(value, dtype=torch.float64)
    phase = initial_problem(fdtd.Bloch(phase=scalar(.3)))
    wavevector = initial_problem(fdtd.Bloch(k=scalar(.3 / .4)))
    with torch.no_grad():
        a = fdtd.run(phase, numerics=numerics(), backend='torch', device=name)
        b = fdtd.run(wavevector, numerics=numerics(), backend='torch', device=name)
    for field in ('E', 'H'):
        actual, expected = a.final_fields.values[field], b.final_fields.values[field]
        assert actual.dtype == torch.complex128 and actual.device == torch.device(name)
        assert torch.isfinite(actual).all() and torch.any(actual.imag != 0)
        torch.testing.assert_close(actual, expected, rtol=1e-13, atol=0.)
    assert a.validity.checks['runtime_stability'] and b.validity.checks['runtime_stability']


@pytest.mark.parametrize('change', ['phase', 'representation', 'initial'])
def test_bloch_prepared_static_changes_refuse_before_execution(device, monkeypatch, change):
    import wavebit.fdtd._api as api
    _, name = device
    problem = initial_problem()
    prepared = fdtd.prepare(problem, numerics=numerics(), backend='torch', device=name)
    if change == 'initial':
        altered = problem.replace(initial=problem.initial.replace(E=(0., 1. + .3j, 0.)))
        path = 'initial.E'
    else:
        face = fdtd.Bloch(phase=.31) if change == 'phase' else fdtd.Bloch(k=.3 / .4)
        altered = problem.replace(scene=problem.scene.replace(
            boundaries=problem.scene.boundaries.replace(x=face)))
        path = 'scene.boundaries'
    monkeypatch.setattr(api, 'execute', lambda *args, **kwargs: pytest.fail('static change executed'))
    report = prepared.validate(altered)
    assert not report.ok and report.issues[0].code == 'prepared.structure'
    with pytest.raises(IncompatibleProblemError) as caught:
        prepared.run(altered)
    assert caught.value.code == 'prepared.structure' and caught.value.path == path


@pytest.mark.parametrize('field', ['phase', 'k'])
def test_bloch_phase_and_wavevector_are_static_not_live_parameters(device, field):
    torch, name = device
    value = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
    problem = initial_problem(fdtd.Bloch(**{field: value}))
    with pytest.raises(IncompatibleProblemError) as caught:
        fdtd.prepare(problem, numerics=numerics(), backend='torch', device=name)
    assert caught.value.code == 'prepared.static_input'
    assert caught.value.path.startswith('scene.boundaries.x')
    assert value.grad is None and value.device.type == 'cpu'


@pytest.mark.parametrize('case,match', [
    ('zero', 'Periodic'), ('unequal', 'same declaration'),
    ('unpaired', 'same declaration'), ('real', 'complex'),
    ('source_k', 'Bloch'), ('one_cell', 'one cell')])
def test_bloch_scientific_guards_remain_active(device, case, match):
    _, name = device
    problem = initial_problem()
    if case == 'zero':
        problem = initial_problem(fdtd.Bloch(phase=0.))
    elif case == 'unequal':
        problem = initial_problem((fdtd.Bloch(phase=.3), fdtd.Bloch(phase=.4)))
    elif case == 'unpaired':
        problem = initial_problem((fdtd.Bloch(phase=.3), fdtd.Periodic()))
    elif case == 'real':
        problem = initial_problem(fdtd.Periodic())
    else:
        problem = plane_problem()
        if case == 'source_k':
            problem = problem.replace(scene=problem.scene.replace(
                boundaries=problem.scene.boundaries.replace(x=fdtd.Bloch(k=.1))))
        else:
            problem = problem.replace(scene=problem.scene.replace(
                domain=fdtd.Domain(size=(.05, .2, 1.2))))
    with pytest.raises(ValueError, match=match):
        # Initial-field dtype is checked when state arrays are materialized.
        operation = fdtd.run if case == 'real' else fdtd.prepare
        operation(problem, numerics=numerics(), backend='torch', device=name)


def test_cuda_bloch_dipole_refuses_without_constructing_a_runtime(monkeypatch):
    import wavebit.fdtd._api as api
    monkeypatch.setattr(Execution, 'validate', lambda self: self)
    monkeypatch.setattr(api, '_planned', lambda *args, **kwargs: pytest.fail('unsupported source planned'))
    problem = initial_problem().replace(initial=None, sources=(fdtd.CurrentDipole(
        'd', center=(0., 0., 0.), component='Ey', current_moment=1e-16,
        pulse=fdtd.GaussianPulse(wavelength=.8, frequency_sigma_hz=2e13)),))
    with pytest.raises(UnsupportedFeatureError) as caught:
        fdtd.prepare(problem, numerics=numerics(), backend='torch', device='cuda:0')
    assert caught.value.code == 'execution.cuda_scope' and caught.value.path == 'sources.d'


@pytest.mark.parametrize('wall', [fdtd.PEC, fdtd.PMC])
def test_cuda_bloch_wall_combination_remains_outside_qualified_scope(monkeypatch, wall):
    import wavebit.fdtd._api as api
    monkeypatch.setattr(Execution, 'validate', lambda self: self)
    monkeypatch.setattr(api, '_planned', lambda *args, **kwargs: pytest.fail('unsupported faces planned'))
    problem = initial_problem()
    problem = problem.replace(scene=problem.scene.replace(
        boundaries=problem.scene.boundaries.replace(z=wall())))
    with pytest.raises(UnsupportedFeatureError) as caught:
        fdtd.prepare(problem, numerics=numerics(), backend='torch', device='cuda:0')
    assert caught.value.code == 'execution.cuda_scope' and caught.value.path == 'scene.boundaries.z-'


@pytest.mark.parametrize('field', ['phase', 'k'])
@pytest.mark.parametrize('transform', ['torch forward_ad', 'torch.func.jvp',
                                      'torch.func.jacfwd', 'torch.func.vmap'])
def test_cuda_bloch_transformed_static_parameters_refuse_before_planning(monkeypatch, field, transform):
    torch = pytest.importorskip('torch')
    import wavebit.fdtd._api as api
    monkeypatch.setattr(Execution, 'validate', lambda self: self)
    monkeypatch.setattr(api, '_planned', lambda *args, **kwargs: pytest.fail('live boundary planned'))

    def prepare(value):
        problem = initial_problem(fdtd.Bloch(**{field: value}))
        fdtd.prepare(problem, numerics=numerics(), backend='torch', device='cuda:0')
        return value

    with pytest.raises(ValidationError) as caught:
        torch_forward_paths(torch)[transform](prepare, .3)
    if transform == 'torch.func.vmap':
        # Existing scalar lowering refuses batches before CUDA admission.
        assert caught.value.code == 'fdtd.invalid' and caught.value.path == 'problem'
        assert 'Bloch(' + field in str(caught.value)
    else:
        assert isinstance(caught.value, IncompatibleProblemError)
        assert caught.value.code == 'prepared.static_input'
        assert caught.value.path == 'scene.boundaries.x-.' + field


@pytest.mark.parametrize('field', ['phase', 'k'])
def test_cuda_bloch_reuse_cannot_introduce_a_forward_tangent(monkeypatch, field):
    torch = pytest.importorskip('torch')
    import wavebit.fdtd._api as api
    monkeypatch.setattr(Execution, 'validate', lambda self: self)
    value = torch.tensor(.3, dtype=torch.float64)
    problem = initial_problem(fdtd.Bloch(**{field: value}))
    prepared = fdtd.prepare(problem, numerics=numerics(), backend='torch', device='cuda:0')
    monkeypatch.setattr(api, '_planned', lambda *args, **kwargs: pytest.fail('live boundary replanned'))
    monkeypatch.setattr(api, 'execute', lambda *args, **kwargs: pytest.fail('live boundary executed'))
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(value, torch.ones_like(value))
        changed = initial_problem(fdtd.Bloch(**{field: dual}))
        report = prepared.validate(changed)
        assert not report.ok and report.issues[0].code == 'prepared.static_input'
        with pytest.raises(IncompatibleProblemError) as caught:
            prepared.run(changed)
        assert caught.value.code == 'prepared.static_input'
        assert caught.value.path == 'scene.boundaries.x-.' + field
