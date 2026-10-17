"""Homogeneous ADE CUDA admission without allocating a CUDA runtime.

Numerical GPU validation lives in separate independent-reference gates. These
tests isolate public check/prepare/reuse, actual rounded-cell coverage, and the
restricted differentiation contract. CPU material behavior remains unchanged.
"""
import math

import numpy as np
import pytest

from wavebit import Units, fdtd
from wavebit._common.execution import Execution
from wavebit.errors import IncompatibleProblemError, UnsupportedFeatureError, ValidationError
from wavebit.materials import Debye, DispersiveMaterial, Drude, Lorentz


UNITS = Units(length='um')
SCALE = UNITS.time_seconds


def problem(kind='lorentz', **parameters):
    material = DispersiveMaterial(
        epsilon_inf=parameters.get('epsilon_inf', 2.25), mu=parameters.get('mu', 1.3),
        poles=((Drude(omega_p=parameters.get('omega_p', math.sqrt(12)/SCALE),
                      gamma=parameters.get('gamma', .7/SCALE)) if kind == 'drude' else
                Lorentz(omega_0=parameters.get('omega_0', 5/SCALE),
                        gamma=parameters.get('gamma', .7/SCALE),
                        strength=parameters.get('strength', 12/SCALE**2))),))
    electric = np.zeros((3, 8, 8, 8))
    electric[1] = 1.
    return fdtd.Problem(units=UNITS, scene=fdtd.Scene(
        domain=fdtd.Domain(size=(1.,)*3),
        boundaries=fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.Periodic()),
        structures=(fdtd.Structure('body', fdtd.Slab(z=(-.5, .5)), material),)),
        initial=fdtd.InitialField(E=electric, amplitude=parameters.get('amplitude', 1.)),
        outputs=fdtd.Outputs(final_fields=True, monitors=(
            fdtd.Record('electric', component='Ey', at=(0., 0., 0.)),)))


def numerics():
    return fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.125),
                         duration=fdtd.Duration(steps=17), courant=.5)


@pytest.fixture
def no_device(monkeypatch):
    import wavebit.fdtd._api as api
    monkeypatch.setattr(Execution, 'validate', lambda self: self)
    monkeypatch.setattr(api, 'execute', lambda *args, **kwargs: pytest.fail('CUDA runtime requested'))
    return api


@pytest.mark.parametrize('axis', 'xyz')
@pytest.mark.parametrize('unit,factor', [('um', 1.), ('nm', 1000.)])
def test_translated_rounded_full_cells_admit_without_runtime(no_device, axis, unit, factor):
    p = problem()
    # Upward cell counts are explicitly independent of the planner helper.
    cells = (13, 10, 15)
    size = np.array([.61, .49, .71])*factor
    center = np.array([.1, -.15, .2])*factor
    resolved = np.array(cells)*(.05*factor)
    index = 'xyz'.index(axis)
    span = (center[index]-resolved[index]/2, center[index]+resolved[index]/2)
    structure = p.scene.structures[0].replace(shape=fdtd.Slab(**{axis: span}))
    fields = np.zeros((3, cells[2], cells[1], cells[0]))
    fields[1] = 1.
    p = p.replace(units=Units(length=unit), scene=p.scene.replace(
        domain=fdtd.Domain(size=tuple(size), center=tuple(center)), structures=(structure,)),
        initial=p.initial.replace(E=fields), outputs=fdtd.Outputs(final_fields=True))
    n = numerics().replace(grid=fdtd.UniformGrid(spacing=.05*factor))
    report = fdtd.check(p, numerics=n, backend='torch', device='cuda:0')
    assert report.ok
    assert report.resolved['grid_shape'] == cells
    prepared = fdtd.prepare(p, numerics=n, backend='torch', device='cuda:0')
    assert prepared.validate(p).ok
    assert prepared.contract['resolved_dt_seconds'] == pytest.approx(
        .5*.05/math.sqrt(3)*SCALE, rel=2e-15, abs=0.)
    assert 'oscillator weight' in prepared.contract['differentiation']


@pytest.mark.parametrize('case,path', [
    ('partial', 'scene.structures.body.shape'), ('two', 'scene.structures'),
    ('background', 'scene.background'), ('multipole', 'scene.structures.body.material.poles'),
    ('debye', 'scene.structures.body.material.poles'),
    ('anisotropy', 'scene.structures.body.material.epsilon_inf'),
    ('wall', 'scene.boundaries.z-'), ('bloch', 'scene.boundaries.z-'),
    ('source', 'sources'), ('dft', 'outputs.field'), ('no_initial', 'initial'),
    ('empty_initial', 'problem'), ('complex', 'initial.E'), ('pattern_shape', 'initial.E'),
    ('callable', 'initial.E'), ('components', 'initial.E'), ('pattern_nan', 'initial.E'),
    ('complex_amplitude', 'initial.amplitude'), ('array_amplitude', 'initial.amplitude'),
    ('bool_pattern', 'initial.E'), ('masked_pattern', 'initial.E'),
    ('bool_amplitude', 'initial.amplitude'),
])
def test_unsupported_ade_refuses_before_planning(no_device, monkeypatch, case, path):
    monkeypatch.setattr(no_device, '_planned', lambda *args, **kwargs: pytest.fail('unsupported ADE planned'))
    p, n = problem(), numerics()
    s = p.scene.structures[0]
    if case == 'partial':
        p = p.replace(scene=p.scene.replace(structures=(s.replace(shape=fdtd.Slab(z=(-.375, .5))),)))
    elif case == 'two':
        p = p.replace(scene=p.scene.replace(structures=(s, s.replace(name='other'))))
    elif case == 'background':
        p = p.replace(scene=p.scene.replace(background=s.material))
    elif case in ('multipole', 'debye', 'anisotropy'):
        material = (s.material.replace(poles=s.material.poles*2) if case == 'multipole' else
                    s.material.replace(poles=(Debye(tau=SCALE, delta_epsilon=.5),)) if case == 'debye' else
                    s.material.replace(epsilon_inf=(2.25,)*3))
        p = p.replace(scene=p.scene.replace(structures=(s.replace(material=material),)))
    elif case in ('wall', 'bloch'):
        p = p.replace(scene=p.scene.replace(boundaries=p.scene.boundaries.replace(
            z=fdtd.PEC() if case == 'wall' else fdtd.Bloch(phase=.3))))
    elif case == 'source':
        p = p.replace(sources=(fdtd.Dipole('drive', center=(0., 0., 0.),
            pulse=fdtd.GaussianPulse(wavelength=1., frequency_sigma_hz=2e13),
            current_moment=1e-10),))
    elif case == 'dft':
        p = p.replace(outputs=fdtd.Outputs(monitors=(fdtd.Field('field', plane=fdtd.Plane(z=0.), wavelengths=(1.,)),)))
    elif case == 'no_initial':
        p = p.replace(initial=None)
    elif case == 'empty_initial':
        p = p.replace(initial=fdtd.InitialField())
    elif case == 'complex_amplitude':
        p = p.replace(initial=p.initial.replace(amplitude=1+.1j))
    elif case == 'array_amplitude':
        p = p.replace(initial=p.initial.replace(amplitude=np.ones(2)))
    elif case == 'bool_amplitude':
        p = p.replace(initial=p.initial.replace(amplitude=True))
    else:
        patterns = {'complex': p.initial.E.astype(complex),
                    'pattern_shape': np.ones((3, 7, 8, 8)),
                    'callable': lambda x, y, z: np.ones_like(x),
                    'components': (None, 1., None),
                    'bool_pattern': p.initial.E.astype(bool),
                    'masked_pattern': np.ma.array(p.initial.E),
                    'pattern_nan': np.full((3, 8, 8, 8), np.nan)}
        p = p.replace(initial=p.initial.replace(E=patterns[case]))
    report = fdtd.check(p, numerics=n, backend='torch', device='cuda:0')
    # Empty InitialField is refused by the established common lowerer first.
    code = 'fdtd_input' if case == 'empty_initial' else 'cuda_scope'
    error = ValidationError if case == 'empty_initial' else UnsupportedFeatureError
    assert not report.ok and report.issues[0].code == code
    assert report.issues[0].path == path
    with pytest.raises(error) as caught:
        fdtd.prepare(p, numerics=n, backend='torch', device='cuda:0')
    assert caught.value.code == code and caught.value.path == path


@pytest.mark.parametrize('axis', 'xyz')
@pytest.mark.parametrize('fraction,admitted', [(.49, True), (.51, False)])
def test_cell_coverage_uses_snapped_cells_not_declared_continuum_span(no_device, axis, fraction, admitted):
    p = problem()
    structure = p.scene.structures[0].replace(shape=fdtd.Slab(**{
        axis: (-.5+fraction*.125, .5)}))
    p = p.replace(scene=p.scene.replace(structures=(structure,)))
    report = fdtd.check(p, numerics=numerics(), backend='torch', device='cuda:0')
    assert report.ok is admitted
    if admitted:
        assert any(issue.code == 'geometry.cell_snapped' for issue in report.issues)
    else:
        assert report.issues[0].code == 'cuda_scope'
        assert report.issues[0].path == 'scene.structures.body.shape'


FIXED = [('epsilon_inf', 2.25), ('mu', 1.3), ('gamma', .7/SCALE), ('omega_0', 5/SCALE)]


@pytest.mark.parametrize('field,value', FIXED)
@pytest.mark.parametrize('transform', ['reverse', 'forward', 'jvp', 'jacfwd'])
def test_static_material_transform_refuses_before_planning(no_device, monkeypatch, field, value, transform):
    torch = pytest.importorskip('torch')
    monkeypatch.setattr(no_device, '_planned', lambda *args, **kwargs: pytest.fail('static material planned'))

    def prepare(x):
        fdtd.prepare(problem(**{field: x}), numerics=numerics(), backend='torch', device='cuda:0')
        return x

    x = torch.tensor(value, dtype=torch.float64)
    with pytest.raises(IncompatibleProblemError) as caught:
        if transform == 'reverse':
            prepare(x.requires_grad_())
        elif transform == 'forward':
            with torch.autograd.forward_ad.dual_level():
                prepare(torch.autograd.forward_ad.make_dual(x, torch.ones_like(x)))
        elif transform == 'jvp':
            torch.func.jvp(prepare, (x,), (torch.ones_like(x),))
        else:
            torch.func.jacfwd(prepare)(x)
    suffix = 'poles[0].'+field if field in ('gamma', 'omega_0') else field
    assert caught.value.code == 'static_input'
    assert caught.value.path == 'scene.structures.body.material.'+suffix


@pytest.mark.parametrize('field,value', FIXED)
def test_reuse_cannot_introduce_static_parameter_tangents(no_device, monkeypatch, field, value):
    torch = pytest.importorskip('torch')
    x = torch.tensor(value, dtype=torch.float64)
    p = problem(**{field: x})
    prepared = fdtd.prepare(p, numerics=numerics(), backend='torch', device='cuda:0')
    monkeypatch.setattr(no_device, '_planned', lambda *args, **kwargs: pytest.fail('live static material replanned'))
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(x, torch.ones_like(x))
        changed = problem(**{field: dual})
        report = prepared.validate(changed)
        assert not report.ok and report.issues[0].code == 'static_input'
        with pytest.raises(IncompatibleProblemError, match='static scalars'):
            prepared.run(changed)


@pytest.mark.parametrize('kind,field', [('lorentz', 'strength'), ('drude', 'omega_p')])
def test_weight_amplitude_reuse_and_static_scalars_remain_accepted(no_device, kind, field):
    torch = pytest.importorskip('torch')
    p = problem(kind)
    prepared = fdtd.prepare(p, numerics=numerics(), backend='torch', device='cuda:0')
    value = 10/SCALE**2 if field == 'strength' else 3/SCALE
    live = problem(kind, **{field: torch.tensor(value, dtype=torch.float64, requires_grad=True),
                           'amplitude': torch.tensor(.7, dtype=torch.float64, requires_grad=True),
                           'epsilon_inf': torch.tensor(1.9, dtype=torch.float64),
                           'mu': torch.tensor(1.1, dtype=torch.float64)})
    assert prepared.validate(live).ok
    fresh = fdtd.prepare(live, numerics=numerics(), backend='torch', device='cuda:0')
    assert fresh.contract == prepared.contract
    changed = live.replace(initial=live.initial.replace(E=2*live.initial.E))
    assert not prepared.validate(changed).ok
    with pytest.raises(IncompatibleProblemError) as caught:
        prepared.run(changed)
    assert caught.value.code == 'prepared_structure' and caught.value.path == 'initial'


def test_cpu_material_derivative_and_contract_remain_unchanged():
    torch = pytest.importorskip('torch')
    epsilon = torch.tensor(2.25, dtype=torch.float64, requires_grad=True)
    p = problem(epsilon_inf=epsilon)
    prepared = fdtd.prepare(p, numerics=numerics(), backend='torch', device='cpu')
    assert prepared.contract['differentiation'] == (
        'existing Yee runtime guards apply; invalid transformed states return NaN and failed validity')
    result = prepared.run(p)
    derivative, = torch.autograd.grad(result.final_fields.values['E'].square().sum(), epsilon)
    assert torch.isfinite(derivative) and abs(float(derivative)) > .01


@pytest.mark.parametrize('field', ['E', 'H'])
def test_default_mu_and_one_real_initial_array_are_admitted(no_device, field):
    p = problem(mu=None)
    if field == 'H':
        p = p.replace(initial=p.initial.replace(E=None, H=p.initial.E))
    prepared = fdtd.prepare(p, numerics=numerics(), backend='torch', device='cuda:0')
    assert prepared.validate(p).ok
    assert 'oscillator weight' in prepared.contract['live'][0]


@pytest.mark.parametrize('dtype', [np.float32, np.int64])
@pytest.mark.parametrize('device', ['cpu', 'cuda:0'], ids=['cpu', 'cuda'])
def test_real_host_input_conversion_keeps_double_working_precision(dtype, device):
    torch = pytest.importorskip('torch')
    if device != 'cpu' and not torch.cuda.is_available():
        pytest.skip('requires an actual CUDA device')
    p = problem()
    p = p.replace(initial=p.initial.replace(E=p.initial.E.astype(dtype)))
    before = p.initial.E.copy()
    result = fdtd.run(p, numerics=numerics(), backend='torch', device=device)
    reference = fdtd.run(problem(), numerics=numerics(), backend='torch', device=device)
    assert result.final_fields.values['E'].dtype == torch.float64
    assert result.final_fields.values['E'].device == torch.device(device)
    assert torch.equal(result.final_fields.values['E'], reference.final_fields.values['E'])
    np.testing.assert_array_equal(p.initial.E, before)


@pytest.mark.parametrize('kind', ['lorentz', 'drude'])
@pytest.mark.parametrize('invalid', ['negative', 'unstable'])
@pytest.mark.parametrize('device', ['cpu', 'cuda:0'], ids=['cpu', 'cuda'])
def test_invalid_live_weight_retains_existing_runtime_guard(kind, invalid, device):
    torch = pytest.importorskip('torch')
    if device != 'cpu' and not torch.cuda.is_available():
        pytest.skip('requires an actual CUDA device')
    prepared = fdtd.prepare(problem(kind), numerics=numerics(), backend='torch', device=device)
    field = 'strength' if kind == 'lorentz' else 'omega_p'
    value = (-12 if invalid == 'negative' else 1e6)/SCALE**2 if kind == 'lorentz' else (
        (-3 if invalid == 'negative' else 1e3)/SCALE)
    valid_value = 12/SCALE**2 if kind == 'lorentz' else math.sqrt(12)/SCALE
    leaf = torch.tensor(valid_value, dtype=torch.float64, device=device, requires_grad=True)
    live = problem(kind, **{field: leaf})
    # A real optimizer can leave the declared domain after construction. The
    # prepared run must recheck this live value, not reuse the valid seed.
    with torch.no_grad():
        leaf.fill_(value)
    saved = leaf.detach().clone()
    result = prepared.run(live)
    assert not bool(result.validity.checks['runtime_stability'])
    assert torch.isnan(result.final_fields.values['E']).any()
    assert result.final_fields.values['E'].device == leaf.device
    assert torch.equal(leaf, saved) and leaf.grad is None
