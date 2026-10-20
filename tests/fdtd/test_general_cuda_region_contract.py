"""Bounded interior SnappedBox admission; numerical Yee oracles live separately."""
import numpy as np
import pytest

from wavebit import Material, Units, fdtd
from wavebit._common.execution import Execution
from wavebit.errors import ValidationError
from wavebit.materials import ConductiveMaterial, DispersiveMaterial, Lorentz


def problem(*, epsilon=2.25, mu=1.3, sigma=600., amplitude=1.):
    e = np.zeros((3, 15, 10, 13))
    e[1] = 1.
    h = np.zeros_like(e)
    h[2] = .003
    return fdtd.Problem(units=Units(length='um'), scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.61, .49, .71), center=(.1, -.15, .2)),
        background=ConductiveMaterial(epsilon=1.2, mu=1.1, conductivity=100.),
        boundaries=fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.Periodic()),
        structures=(fdtd.Structure('body', shape=fdtd.SnappedBox(size=(.305, .225, .365),
            center=(.0375, -.2025, .1625)),
            material=ConductiveMaterial(epsilon=epsilon, mu=mu, conductivity=sigma)),)),
        initial=fdtd.InitialField(E=e, H=h, amplitude=amplitude),
        outputs=fdtd.Outputs(final_fields=True, monitors=(
            fdtd.TimeSeriesMonitor('e', component='Ey', at=(.1, -.15, .2)),)))


def numerics(block=None):
    return fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.05),
                         duration=fdtd.Duration(steps=17), courant=.5, checkpoint_steps=block)


@pytest.fixture
def no_device(monkeypatch):
    import wavebit.fdtd._api as api
    monkeypatch.setattr(Execution, 'validate', lambda self: self)
    monkeypatch.setattr(api, 'execute', lambda *a, **k: pytest.fail('CUDA execution requested'))
    return api


@pytest.mark.parametrize('block', [None, 7])
@pytest.mark.parametrize('unit,factor', [('um', 1.), ('nm', 1000.)])
def test_metadata_admits_translated_rounded_cells_and_schedule(no_device, block, unit, factor):
    p = problem()
    s = p.scene.structures[0]
    p = p.replace(units=Units(length=unit), scene=p.scene.replace(
        domain=fdtd.Domain(size=tuple(factor*np.array(p.scene.domain.size)),
                           center=tuple(factor*np.array(p.scene.domain.center))),
        structures=(s.replace(shape=fdtd.SnappedBox(size=tuple(factor*np.array(s.shape.size)),
                            center=tuple(factor*np.array(s.shape.center)))),)),
        outputs=fdtd.Outputs(final_fields=True))
    n = numerics(block).replace(grid=fdtd.UniformGrid(spacing=.05*factor))
    report = fdtd.check(p, numerics=n, backend='torch', device='cuda:0')
    assert report.ok and report.resolved['grid_shape'] == (13, 10, 15)
    assert any(issue.code == 'geometry.cell_snapped' for issue in report.issues)
    prepared = fdtd.prepare(p, numerics=n, backend='torch', device='cuda:0')
    assert prepared.validate(p).ok
    assert prepared.contract['resolved_checkpoint_steps'] == block
    assert 'strictly interior SnappedBox' in prepared.contract['differentiation']
    assert not any('source amplitude' in item for item in prepared.contract['live'])
    assert prepared.contract['resolved_time_step_seconds'] == pytest.approx(
        .5*.05/np.sqrt(3)*Units(length='um').time_unit_seconds, rel=2e-15)


def changed(p, case):
    s = p.scene.structures[0]
    if case == 'two':
        return p.replace(scene=p.scene.replace(structures=(s, s.replace(name='other'))))
    if case == 'slab_mix':
        return p.replace(scene=p.scene.replace(structures=(s, s.replace(
            name='other', shape=fdtd.SnappedSlab(z=(.05, .15))))))
    if case in ('box', 'shape_subclass', 'fullspan', 'seam', 'clipped', 'empty'):
        class RegionSubclass(fdtd.SnappedBox):
            pass
        shapes = dict(box=fdtd.Box(size=s.shape.size, center=s.shape.center),
            shape_subclass=RegionSubclass(size=s.shape.size, center=s.shape.center),
            fullspan=fdtd.SnappedBox(size=(.65, .225, .365), center=(.1, -.2025, .1625)),
            seam=fdtd.SnappedBox(size=(.1, .225, .365), center=(-.175, -.2025, .1625)),
            clipped=fdtd.SnappedBox(size=(.4, .225, .365), center=(-.2, -.2025, .1625)),
            empty=fdtd.SnappedBox(size=(.001, .225, .365), center=(.025, -.2025, .1625)))
        return p.replace(scene=p.scene.replace(structures=(s.replace(shape=shapes[case]),)))
    if case in ('open', 'pec', 'pmc', 'bloch', 'periodic_subclass'):
        class PeriodicSubclass(fdtd.Periodic):
            pass
        face = dict(open=fdtd.Open(), pec=fdtd.PEC(), pmc=fdtd.PMC(),
                    bloch=fdtd.Bloch(phase=.2), periodic_subclass=PeriodicSubclass())[case]
        return p.replace(scene=p.scene.replace(boundaries=fdtd.Boundaries(
            x=face, y=fdtd.Periodic(), z=fdtd.Periodic())))
    if case in ('anisotropy', 'complex_material', 'material_subclass', 'ade', 'background_ade'):
        class MaterialSubclass(Material):
            pass
        materials = dict(anisotropy=Material(epsilon=(2., 2.2, 2.4)),
            complex_material=Material(epsilon=2.+.1j), material_subclass=MaterialSubclass(2.25),
            ade=DispersiveMaterial(epsilon_inf=2., poles=(Lorentz(omega_0=1., gamma=0., strength=1.),)))
        if case == 'background_ade':
            return p.replace(scene=p.scene.replace(background=materials['ade']))
        return p.replace(scene=p.scene.replace(structures=(s.replace(material=materials[case]),)))
    if case == 'source':
        return p.replace(sources=(fdtd.CurrentDipole('d', center=(.1, -.15, .2), component='Ey',
            pulse=fdtd.GaussianPulse(wavelength=2., frequency_sigma_hz=1e13), current_moment=1e-16),))
    if case in ('field', 'flux', 'record_subclass'):
        class RecordSubclass(fdtd.TimeSeriesMonitor):
            pass
        monitor = dict(field=fdtd.FieldMonitor('f', plane=fdtd.Plane(z=.2), wavelengths=(2.,)),
            flux=fdtd.FluxMonitor('f', plane=fdtd.Plane(z=.2), wavelengths=(2.,)),
            record_subclass=RecordSubclass('r', component='Ey', at=(.1, -.15, .2)))[case]
        return p.replace(outputs=fdtd.Outputs(monitors=(monitor,)))
    if case == 'no_initial':
        return p.replace(initial=None)
    if case in ('complex_pattern', 'shape', 'nan_pattern', 'bool_pattern', 'masked', 'callable', 'components'):
        value = p.initial.E.copy()
        if case == 'complex_pattern': value = value.astype(complex)
        elif case == 'shape': value = value[:, :-1]
        elif case == 'nan_pattern': value[0, 0, 0, 0] = np.nan
        elif case == 'bool_pattern': value = value.astype(bool)
        elif case == 'masked': value = np.ma.array(value)
        elif case == 'callable': value = lambda x, y, z: (x*0, y*0+1, z*0)
        elif case == 'components': value = (0., 1., 0.)
        return p.replace(initial=p.initial.replace(E=value))
    if case in ('complex_amplitude', 'array_amplitude', 'bool_amplitude'):
        value = dict(complex_amplitude=1j, array_amplitude=np.ones(2), bool_amplitude=True)[case]
        return p.replace(initial=p.initial.replace(amplitude=value))
    raise AssertionError(case)


@pytest.mark.parametrize('case', ['two', 'slab_mix', 'box', 'shape_subclass', 'fullspan', 'seam',
    'clipped', 'empty', 'open', 'pec', 'pmc', 'bloch', 'periodic_subclass', 'anisotropy',
    'complex_material', 'material_subclass', 'ade', 'background_ade', 'source', 'field', 'flux',
    'record_subclass', 'no_initial', 'complex_pattern', 'shape', 'nan_pattern', 'bool_pattern',
    'masked', 'callable', 'components', 'complex_amplitude', 'array_amplitude', 'bool_amplitude'])
def test_metadata_refuses_outside_region_family(no_device, monkeypatch, case):
    p = changed(problem(), case)
    scope_paths = {'two': 'scene.structures', 'slab_mix': 'scene.structures',
        'fullspan': 'scene.structures.body.shape', 'anisotropy': 'scene.structures.body.material.epsilon',
        'source': 'sources', 'field': 'outputs.f', 'flux': 'outputs.f'}
    if case in scope_paths:
        monkeypatch.setattr(no_device, '_planned',
                            lambda *a, **k: pytest.fail('unsupported SnappedBox reached planning'))
    report = fdtd.check(p, numerics=numerics(), backend='torch', device='cuda:0')
    assert not report.ok, case
    if case in scope_paths:
        assert report.issues[0].code == 'execution.cuda_scope'
        assert report.issues[0].path == scope_paths[case]
    with pytest.raises(ValidationError) as exc:
        fdtd.prepare(p, numerics=numerics(), backend='torch', device='cuda:0')
    if case in scope_paths:
        assert exc.value.issue.code == 'execution.cuda_scope'
        assert exc.value.issue.path == scope_paths[case]


@pytest.mark.parametrize('field', ['E', 'H'])
def test_metadata_single_pattern_default_mu_and_record_only(no_device, field):
    p = problem(mu=None)
    p = p.replace(initial=p.initial.replace(**{('H' if field == 'E' else 'E'): None}),
                  outputs=fdtd.Outputs(monitors=p.outputs.monitors, final_fields=False))
    prepared = fdtd.prepare(p, numerics=numerics(), backend='torch', device='cuda:0')
    assert prepared.validate(p).ok


def test_metadata_live_material_reuse_and_static_pattern_refusal(no_device):
    torch = pytest.importorskip('torch')
    prepared = fdtd.prepare(problem(), numerics=numerics(7), backend='torch', device='cuda:0')
    leaves = [torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (1.8, 1.1, 900., .7)]
    live = problem(epsilon=leaves[0], mu=leaves[1], sigma=leaves[2], amplitude=leaves[3])
    background = ConductiveMaterial(epsilon=leaves[0], mu=leaves[1], conductivity=leaves[2])
    assert prepared.validate(live.replace(scene=live.scene.replace(background=background))).ok
    value = live.initial.E.copy(); value[1, 2, 3, 4] += .1
    assert not prepared.validate(live.replace(initial=live.initial.replace(E=value))).ok
    moved = live.scene.structures[0].replace(shape=fdtd.SnappedBox(size=(.31, .225, .365),
                                                             center=(.0375, -.2025, .1625)))
    report = prepared.validate(live.replace(scene=live.scene.replace(structures=(moved,))))
    assert not report.ok and report.issues[0].code == 'prepared.structure'
    assert all(leaf.is_leaf and leaf.grad is None for leaf in leaves)


@pytest.mark.parametrize('transform', ['reverse', 'dual', 'jvp', 'jacfwd'])
def test_region_geometry_is_static_under_torch_transforms(transform):
    torch = pytest.importorskip('torch')
    x = torch.tensor(.305, dtype=torch.float64, requires_grad=transform == 'reverse')
    def make(value):
        fdtd.SnappedBox(size=(value, .225, .365), center=(.0375, -.2025, .1625))
        return value*2
    with pytest.raises((ValueError, ValidationError), match='static'):
        if transform == 'reverse': make(x)
        elif transform == 'jvp': torch.func.jvp(make, (x,), (torch.ones_like(x),))
        elif transform == 'jacfwd': torch.func.jacfwd(make)(x)
        else:
            with torch.autograd.forward_ad.dual_level():
                make(torch.autograd.forward_ad.make_dual(x, torch.ones_like(x)))


@pytest.mark.parametrize('dtype', [np.float32, np.int64])
@pytest.mark.parametrize('device', ['cpu', 'cuda:0'], ids=['cpu', 'gpu'])
def test_host_pattern_conversion_preserves_double_and_ownership(dtype, device):
    torch = pytest.importorskip('torch')
    if device != 'cpu' and not torch.cuda.is_available():
        pytest.skip('actual CUDA device required')
    p = problem()
    p = p.replace(initial=p.initial.replace(E=p.initial.E.astype(dtype), H=None))
    saved = p.initial.E.copy()
    result = fdtd.run(p, numerics=numerics(), backend='torch', device=device)
    reference = fdtd.run(p.replace(initial=p.initial.replace(E=saved.astype(np.float64))),
                         numerics=numerics(), backend='torch', device=device)
    for field in ('E', 'H'):
        value = result.final_fields.values[field]
        assert value.dtype == torch.float64 and value.device == torch.device(device)
        torch.testing.assert_close(value, reference.final_fields.values[field], rtol=0, atol=0)
    np.testing.assert_array_equal(p.initial.E, saved)


def test_cpu_scope_is_unchanged_and_fullspan_region_remains_legal():
    import wavebit.fdtd._api as api
    assert api._cuda_scope(None, None, Execution(backend='numpy', device='cpu')) is None
    p = changed(problem(), 'fullspan')
    report = fdtd.check(p, numerics=numerics(), backend='numpy')
    assert report.ok
    result = fdtd.run(p, numerics=numerics())
    assert np.isfinite(result.final_fields.values['E']).all()


@pytest.mark.parametrize('parameter,bad', [('epsilon', 0.), ('mu', 0.), ('sigma', -1.),
                                        ('amplitude', float('nan'))])
@pytest.mark.parametrize('device', ['cpu', 'cuda:0'], ids=['cpu', 'gpu'])
def test_invalid_same_leaf_recovers_without_mutating_prior_outputs(device, parameter, bad):
    torch = pytest.importorskip('torch')
    if device != 'cpu' and not torch.cuda.is_available():
        pytest.skip('actual CUDA device required')
    values = dict(epsilon=2.25, mu=1.3, sigma=600., amplitude=1.2)
    leaves = {name: torch.tensor(value, dtype=torch.float64,
              device='cpu' if name == 'amplitude' else device, requires_grad=True)
              for name, value in values.items()}
    identities = {name: id(value) for name, value in leaves.items()}
    p = problem(**leaves)
    n = numerics().replace(duration=fdtd.Duration(steps=3))
    prepared = fdtd.prepare(problem(), numerics=n, backend='torch', device=device)
    first = prepared.run(p)
    assert bool(first.validity.checks['runtime_stability'])
    def outputs(result):
        return (*result.final_fields.values.values(), result['e'].values)
    held = tuple(value.detach().clone() for value in outputs(first))
    assert all(bool(torch.isfinite(value).all()) for value in held)
    assert all(bool((value != 0).any()) for value in held)
    with torch.no_grad():
        leaves[parameter].fill_(bad)
    try:
        invalid = prepared.run(p)
    except ValidationError as exc:
        # An unrelated structure/device failure must not qualify invalid-value safety.
        message = str(exc).lower()
        labels = {'epsilon': ('epsilon', 'eps'), 'mu': ('mu', 'permeability'),
                  'sigma': ('sigma', 'conductivity'), 'amplitude': ('amplitude', 'initial')}
        assert exc.issue.code == 'fdtd.invalid'
        assert any(label in message for label in labels[parameter])
        reasons = (('finite', 'nan') if parameter == 'amplitude' else
                   ('finite', 'positive', 'nonnegative', 'non-negative', 'stable', 'cfl'))
        assert any(reason in message for reason in reasons)
    else:
        assert not bool(invalid.validity.checks['runtime_stability'])
        assert bool(torch.isnan(invalid.final_fields.values['E']).any())
    with torch.no_grad():
        leaves[parameter].fill_(values[parameter])
    recovered = prepared.run(p)
    assert bool(recovered.validity.checks['runtime_stability'])
    for current, old, saved in zip(outputs(recovered), outputs(first), held):
        torch.testing.assert_close(current, saved, rtol=0, atol=0)
        torch.testing.assert_close(old, saved, rtol=0, atol=0)
    for name, leaf in leaves.items():
        assert id(leaf) == identities[name] and leaf.is_leaf and leaf.grad is None
        assert leaf.device == torch.device('cpu' if name == 'amplitude' else device)
        assert leaf.item() == values[name]
