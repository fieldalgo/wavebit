"""The public surface is a decision, so it is pinned here and changed on purpose.

Wavebit follows two rules from NumPy's 2.0 cleanup (NEP 52). A public module says what it
exports in ``__all__``, and every public object has one public path. ``PUBLIC`` pins each
public module's ``__all__``: a name is added or removed by editing it here, in the same change
as the code. ``KNOWN_DUPLICATES`` lists the objects still reachable from two public paths; it
may only shrink (fieldalgo/wavebit#57 removes them). The FDTD part of the rename
(``docs/migration/public-names.md``), ``fdtd.native`` and ``fdtd.periodic`` included, is checked
here too: the old names and keywords fail with an error that names their replacement.
"""
import dataclasses
import importlib
import pkgutil
import traceback

import pytest

import wavebit
from wavebit import fdtd
from wavebit.errors import ValidationError
from wavebit.fdtd import native, periodic

#: Each public module's ``__all__``, sorted.
PUBLIC = {
 'wavebit': ('Backend', 'CheckReport', 'ConductiveMaterial', 'Debye', 'DiffractionOrders',
             'DispersiveMaterial', 'Drude', 'Execution', 'FieldSamples', 'IncompatibleProblemError',
             'Issue', 'Job', 'Lattice', 'Lorentz', 'Material', 'MaterialResponse', 'NumericalError',
             'RawDiffractionSpectrum', 'SpectralFieldSamples', 'TabulatedMaterial', 'TimeSeries',
             'Units', 'UnsupportedFeatureError', 'ValidationError', 'Validity', 'WavebitError',
             'eps_from_refractive_index', 'fdtd', 'materials', 'rcwa', 'refractive_index_from_eps',
             'run'),
 'wavebit.fdtd': ('Bloch', 'Boundaries', 'Box', 'ConductiveAbsorber', 'CurrentDipole', 'Cylinder',
                  'DecayStop', 'DiffractionMonitor', 'Domain', 'Duration', 'FarFieldMonitor',
                  'FarFieldSpectrum', 'FieldMonitor', 'FluxMonitor', 'GaussianPulse',
                  'InitialField', 'InvariantAxis', 'MirrorSymmetry', 'Numerics', 'Open', 'Outputs',
                  'PEC', 'PMC', 'PML', 'Periodic', 'Plane', 'PlaneWave', 'Prepared', 'Prism',
                  'Problem', 'Result', 'Scene', 'Shape', 'SnappedBox', 'SnappedSlab',
                  'SpectralFlux', 'Sphere', 'Structure', 'TFSF', 'TimeSeriesMonitor', 'UniformGrid',
                  'check', 'native', 'periodic', 'prepare', 'run'),
 'wavebit.fdtd.diagnostics': ('AbsorberProximityWarning', 'BoundedVoxelWarning',
                              'CoarseMeshWarning', 'GeometryClippedWarning', 'GrowingFieldWarning',
                              'PMLStabilityWarning', 'RadiationApproximationWarning',
                              'SubpixelAccuracyWarning', 'UnsettledFieldWarning'),
 'wavebit.fdtd.native': ('Current', 'DecayCheck', 'Grid', 'InitialField', 'Medium', 'Numerics',
                         'Oscillator', 'Outputs', 'PML', 'Prepared', 'Problem', 'Result',
                         'Snapshots', 'Spectrum', 'check', 'prepare', 'run'),
 'wavebit.fdtd.periodic': ('Circle', 'DecayCheck', 'Ellipse', 'Excitation', 'Layer', 'Numerics',
                           'Outputs', 'PlaneWave', 'Polygon', 'Prepared', 'Problem', 'Pulse',
                           'RasterDispersion', 'Rectangle', 'Region', 'Result', 'Snapshots',
                           'Stack', 'check', 'prepare', 'run'),
 'wavebit.lattice': ('DrawnOn', 'Lattice', 'as_lattice', 'is_axis_aligned', 'jacobian',
                     'meshgrid_xy', 'order_kxy', 'periods_of', 'reduced_basis',
                     'reduction_transform', 'sample_grid'),
 'wavebit.materials': ('ConductiveMaterial', 'Debye', 'DispersiveMaterial', 'Drude', 'Lorentz',
                       'Material', 'MaterialResponse', 'TabulatedMaterial', 'available',
                       'eps_from_refractive_index', 'load', 'refractive_index_from_eps'),
 'wavebit.materials.catalog': ('available', 'load'),
 'wavebit.materials.time_domain': ('Conductor', 'Debye', 'Dispersive', 'Drude', 'Lorentz',
                                   'Material', 'MaterialBase', 'Pole', 'as_material', 'concrete',
                                   'eps_at'),
 'wavebit.migration': ('migrate_legacy_jobs',),
 'wavebit.observations': ('DiffractionOrder', 'DiffractionOrders', 'FieldSamples', 'ModeFit',
                          'RawDiffractionSpectrum', 'SpectralFieldSamples', 'TimeSeries',
                          'Validity', 'fit_modes'),
 'wavebit.observations.modes': ('ModeFit', 'fit_modes'),
 'wavebit.rcwa': ('BoundaryPower', 'Circle', 'ElectricDipole', 'Ellipse', 'Excitation',
                  'GaussianBeam', 'Layer', 'Numerics', 'OrderExcitation', 'Outputs', 'PlaneWave',
                  'Polygon', 'Prepared', 'Problem', 'Rectangle', 'Region', 'Result', 'Stack',
                  'check', 'layer_thicknesses', 'prepare', 'run', 'sample_structure'),
 'wavebit.rcwa.experimental': ('AffineProfileNVF', 'CartesianPairOpeningPlan',
                               'CartesianPairTransmissionGalerkinGapCell', 'CartesianRectangleNVF',
                               'CartesianRectanglePairNVF', 'ConformingNVF'),
 'wavebit.rcwa.operators': ('ConvMatrices', 'Operators', 'Port', 'ScatteringMatrix',
                            'TensorConvMatrices', 'assemble', 'solve'),
 'wavebit.records': ('RecordedFields', 'RunRecord', 'load_run', 'run_recorded', 'save_run')
}

#: Objects reachable from more than one public path today. This list may only shrink.
KNOWN_DUPLICATES = (('wavebit.ConductiveMaterial', 'wavebit.materials.ConductiveMaterial'),
 ('wavebit.Debye', 'wavebit.materials.Debye'),
 ('wavebit.DiffractionOrders', 'wavebit.observations.DiffractionOrders'),
 ('wavebit.DispersiveMaterial', 'wavebit.materials.DispersiveMaterial'),
 ('wavebit.Drude', 'wavebit.materials.Drude'),
 ('wavebit.FieldSamples', 'wavebit.observations.FieldSamples'),
 ('wavebit.Lattice', 'wavebit.lattice.Lattice'), ('wavebit.Lorentz', 'wavebit.materials.Lorentz'),
 ('wavebit.Material', 'wavebit.materials.Material'),
 ('wavebit.MaterialResponse', 'wavebit.materials.MaterialResponse'),
 ('wavebit.RawDiffractionSpectrum', 'wavebit.observations.RawDiffractionSpectrum'),
 ('wavebit.SpectralFieldSamples', 'wavebit.observations.SpectralFieldSamples'),
 ('wavebit.TabulatedMaterial', 'wavebit.materials.TabulatedMaterial'),
 ('wavebit.TimeSeries', 'wavebit.observations.TimeSeries'),
 ('wavebit.Validity', 'wavebit.observations.Validity'),
 ('wavebit.eps_from_refractive_index', 'wavebit.materials.eps_from_refractive_index'),
 ('wavebit.fdtd.periodic.Circle', 'wavebit.rcwa.Circle'),
 ('wavebit.fdtd.periodic.Ellipse', 'wavebit.rcwa.Ellipse'),
 ('wavebit.fdtd.periodic.Excitation', 'wavebit.rcwa.Excitation'),
 ('wavebit.fdtd.periodic.Layer', 'wavebit.rcwa.Layer'),
 ('wavebit.fdtd.periodic.PlaneWave', 'wavebit.rcwa.PlaneWave'),
 ('wavebit.fdtd.periodic.Polygon', 'wavebit.rcwa.Polygon'),
 ('wavebit.fdtd.periodic.Rectangle', 'wavebit.rcwa.Rectangle'),
 ('wavebit.fdtd.periodic.Region', 'wavebit.rcwa.Region'),
 ('wavebit.fdtd.periodic.Stack', 'wavebit.rcwa.Stack'),
 ('wavebit.materials.available', 'wavebit.materials.catalog.available'),
 ('wavebit.materials.catalog.load', 'wavebit.materials.load'),
 ('wavebit.materials.refractive_index_from_eps', 'wavebit.refractive_index_from_eps'),
 ('wavebit.observations.ModeFit', 'wavebit.observations.modes.ModeFit'),
 ('wavebit.observations.fit_modes', 'wavebit.observations.modes.fit_modes'))


def _public_modules():
    names = ['wavebit']
    for info in pkgutil.walk_packages(wavebit.__path__, 'wavebit.'):
        if not any(part.startswith('_') for part in info.name.split('.')[1:]):
            names.append(info.name)
    return names


def test_every_public_module_exports_what_is_pinned():
    found = {}
    for name in _public_modules():
        exported = getattr(importlib.import_module(name), '__all__', None)
        if exported is not None:
            found[name] = tuple(sorted(exported))
    assert found == PUBLIC


def test_no_public_object_gains_a_second_public_path():
    paths = {}
    for name in _public_modules():
        module = importlib.import_module(name)
        for attr in getattr(module, '__all__', ()):
            obj = getattr(module, attr)
            if isinstance(obj, type) or callable(obj):
                paths.setdefault(id(obj), []).append(f'{name}.{attr}')
    duplicates = tuple(sorted(tuple(sorted(p)) for p in paths.values() if len(p) > 1))
    assert set(duplicates) <= set(KNOWN_DUPLICATES), sorted(set(duplicates) - set(KNOWN_DUPLICATES))
    assert set(duplicates) == set(KNOWN_DUPLICATES), (
        'a duplicate was removed: delete it from KNOWN_DUPLICATES',
        sorted(set(KNOWN_DUPLICATES) - set(duplicates)))


def test_fdtd_has_no_public_implementation_modules():
    """Every FDTD declaration has one public path, ``wavebit.fdtd.<Name>``."""
    public = sorted(info.name for info in pkgutil.iter_modules(fdtd.__path__)
                    if not info.name.startswith('_'))
    assert public == ['diagnostics', 'native', 'periodic']


#: Old public FDTD name -> its replacement.
RENAMED = {'Record': 'TimeSeriesMonitor', 'Field': 'FieldMonitor', 'Flux': 'FluxMonitor',
           'Diffraction': 'DiffractionMonitor', 'FarField': 'FarFieldMonitor',
           'Region': 'SnappedBox', 'Slab': 'SnappedSlab', 'Dipole': 'CurrentDipole',
           'Axis1D': 'InvariantAxis', 'Symmetry': 'MirrorSymmetry',
           'Absorber': 'ConductiveAbsorber', 'Decay': 'DecayStop'}


@pytest.mark.parametrize('old, new', sorted(RENAMED.items()))
def test_a_removed_fdtd_name_fails_naming_its_replacement(old, new):
    assert not hasattr(fdtd, old)
    with pytest.raises(AttributeError, match=rf'wavebit\.fdtd\.{old} is now wavebit\.fdtd\.{new}\b'):
        getattr(fdtd, old)
    assert new in fdtd.__all__


@pytest.mark.parametrize('module', ['problem', 'geometry', 'boundaries', 'sources', 'monitors',
                                    'numerics', 'result', 'results'])
def test_a_former_fdtd_module_is_not_a_public_path(module):
    with pytest.raises(AttributeError, match='no longer a public module'):
        getattr(fdtd, module)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f'wavebit.fdtd.{module}')


def _pulse():
    return fdtd.GaussianPulse(wavelength=1., frequency_sigma_hz=1e13)


#: (constructor, the keywords it needs, the removed keyword, its replacement).
RENAMED_KEYWORDS = [
    (lambda **kw: fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.1), duration=fdtd.Duration(steps=1), **kw),
     'discretisation', 'discretization', 'voxel'),
    (lambda **kw: fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.1), duration=fdtd.Duration(steps=1), **kw),
     'decay', 'stopping', None),
    (lambda **kw: fdtd.PML(**kw), 'stabilize', 'stabilizing_loss', 1e-3),
    (lambda **kw: fdtd.PlaneWave('p', plane=fdtd.Plane(z=0.), pulse=_pulse(), **kw),
     'delay', 'delay_seconds', 0.),
    (lambda **kw: fdtd.CurrentDipole('d', center=(0., 0., 0.), pulse=_pulse(), **kw),
     'delay', 'delay_seconds', 0.),
    (lambda **kw: fdtd.TFSF('t', box=fdtd.SnappedBox(size=(.2, .2, .2)), pulse=_pulse(), **kw),
     'delay', 'delay_seconds', 0.),
    (lambda **kw: fdtd.DiffractionMonitor('r', plane=fdtd.Plane(z=0.), wavelengths=(1.,),
                                          source='p', **kw), 'side', 'wave', 'reflected'),
]


@pytest.mark.parametrize('make, old, new, value', RENAMED_KEYWORDS,
                         ids=[f'{old}->{new}' for _, old, new, _ in RENAMED_KEYWORDS])
def test_a_removed_fdtd_keyword_fails_naming_its_replacement(make, old, new, value):
    with pytest.raises(TypeError, match=rf'\({old}=\.\.\.\) is now \w+\({new}=\.\.\.\)'):
        make(**{old: value})
    make(**{new: value})


def test_bloch_from_source_names_the_invariant_axes_keyword():
    source = fdtd.PlaneWave('p', plane=fdtd.Plane(z=0.), pulse=_pulse())
    with pytest.raises(TypeError, match=r'invariant_axes=\.\.\.'):
        fdtd.Bloch.from_source(source, units=wavebit.Units(length='um'), axis1d=('y',))


def test_a_diffraction_monitor_names_the_new_wave_values():
    with pytest.raises(ValueError, match="'reflect' is now 'reflected'"):
        fdtd.DiffractionMonitor('r', plane=fdtd.Plane(z=0.), wavelengths=(1.,), source='p',
                                wave='reflect')


#: Every FDTD declaration whose fields follow a name takes them by keyword only.
NAMED = ('TimeSeriesMonitor', 'FieldMonitor', 'FluxMonitor', 'DiffractionMonitor',
         'FarFieldMonitor', 'PlaneWave', 'CurrentDipole', 'TFSF', 'Structure')


@pytest.mark.parametrize('name', NAMED)
def test_an_fdtd_declaration_takes_only_its_name_positionally(name):
    with pytest.raises(TypeError, match='positionally and everything else by keyword'):
        getattr(fdtd, name)('n', object())


def test_an_unknown_fdtd_keyword_is_refused_with_a_near_match():
    with pytest.raises(TypeError, match="did you mean 'wavelengths'"):
        fdtd.FluxMonitor('f', wavelength=(1.,), plane=fdtd.Plane(z=0.))


def test_the_fdtd_reference_documents_every_public_name():
    """``docs/reference/fdtd/index.md`` names its objects by hand, so a rename must reach it."""
    import pathlib
    import re
    page = pathlib.Path(__file__).parents[2] / 'docs' / 'reference' / 'fdtd' / 'index.md'
    documented = set(re.findall(r'^\.\. auto(?:class|function|module):: wavebit\.fdtd\.(\w+)',
                                page.read_text(encoding='utf-8'), flags=re.M))
    assert documented == set(fdtd.__all__)


def test_dir_of_fdtd_is_its_public_surface():
    """``dir(fdtd)`` is what Python's "Did you mean" reads; it named the private ``_monitors``."""
    listed = [name for name in dir(fdtd) if not name.startswith('__')]
    assert sorted(listed) == sorted(set(fdtd.__all__) | {'diagnostics'})


@pytest.mark.parametrize('name', ['Record', 'Region', 'monitors', 'problem', 'results'])
def test_a_tombstone_is_not_followed_by_a_guess(name):
    with pytest.raises(AttributeError) as caught:
        getattr(fdtd, name)
    shown = ''.join(traceback.format_exception_only(caught.value))
    assert 'public-names.md' in shown and 'Did you mean' not in shown, shown


def _shown(error):
    """What a reader sees of ``error``, Python's "Did you mean" included."""
    return ''.join(traceback.format_exception_only(error))


#: Old name in an explicit-grid module -> its replacement in the same module.
GRID_RENAMED = [('native', 'Initial', 'InitialField'), ('native', 'Record', 'Snapshots'),
                ('native', 'Tail', 'DecayCheck'), ('periodic', 'Record', 'Snapshots')]


@pytest.mark.parametrize('module, old, new', GRID_RENAMED,
                         ids=[f'{module}.{old}' for module, old, _ in GRID_RENAMED])
def test_a_removed_native_or_periodic_name_fails_naming_its_replacement(module, old, new):
    namespace = getattr(fdtd, module)
    assert not hasattr(namespace, old)
    with pytest.raises(AttributeError, match=rf'wavebit\.fdtd\.{module}\.{old} is now '
                                             rf'wavebit\.fdtd\.{module}\.{new}\b') as caught:
        getattr(namespace, old)
    shown = _shown(caught.value)
    assert 'public-names.md' in shown and 'Did you mean' not in shown, shown
    assert new in namespace.__all__


def test_the_native_tail_tombstone_says_where_its_replacement_goes():
    """``DecayCheck`` is no drop-in: it goes in ``Numerics(decay_check=)``, with ``threshold=``
    for ``tolerance=``."""
    with pytest.raises(AttributeError, match=r'Numerics\(decay_check=\.\.\.\), with tolerance= '
                                             r'now threshold='):
        native.Tail


def _native_numerics(**kw):
    return native.Numerics(grid=native.Grid(spacing=(.1, .1, .1)), time_step_seconds=1e-16,
                           steps=4, **kw)


def _periodic_numerics(**kw):
    return periodic.Numerics(spacing=.02, duration_seconds=1e-14, **kw)


#: (module, constructor with the keywords it needs, removed keyword, a value it took, its
#: replacement, a value that takes). A converted quantity keeps its old value on the old side.
GRID_RENAMED_KEYWORDS = [
    ('native', lambda **kw: native.Snapshots(**kw),
     'bounds', (0, 0, (0, 2)), 'slices', (0, 0, (0, 2))),
    ('native', lambda **kw: native.Spectrum(wavelengths=(1.,), **kw),
     'planes', (0,), 'z_indices', (0,)),
    ('native', lambda **kw: native.Outputs(**kw),
     'record', native.Snapshots(), 'snapshots', native.Snapshots()),
    ('native', _native_numerics,
     'tail', native.DecayCheck(window_seconds=1e-16), 'decay_check',
     native.DecayCheck(window_seconds=1e-16)),
    ('periodic', lambda **kw: periodic.Outputs(**kw),
     'record', periodic.Snapshots(), 'snapshots', periodic.Snapshots()),
    ('periodic', _periodic_numerics,
     'tail_steps', 40, 'decay_check', periodic.DecayCheck(window_seconds=1e-15)),
    ('periodic', _periodic_numerics,
     'tail_tolerance', 1e-3, 'decay_check',
     periodic.DecayCheck(window_seconds=1e-15, threshold=1e-3)),
    ('periodic', lambda **kw: periodic.RasterDispersion(labels=[[[1]]], **kw),
     'models', {1: ()}, 'poles', {1: ()}),
    ('periodic', lambda **kw: periodic.Pulse(**kw),
     'width_seconds', 1e-14, 'frequency_sigma_hz', 3e13),
]


@pytest.mark.parametrize('module, make, old, old_value, new, new_value', GRID_RENAMED_KEYWORDS,
                         ids=[f'{m}:{old}->{new}'
                              for m, _, old, _, new, _ in GRID_RENAMED_KEYWORDS])
def test_a_removed_native_or_periodic_keyword_fails_naming_its_replacement(
        module, make, old, old_value, new, new_value):
    with pytest.raises(TypeError, match=rf'\({old}=\.\.\.\) is now \w+\({new}=\.\.\.\)') as caught:
        make(**{old: old_value})
    shown = _shown(caught.value)
    assert 'public-names.md' in shown and 'did you mean' not in shown.lower(), shown
    assert type(make(**{new: new_value})).__module__ == f'wavebit.fdtd.{module}'


#: Keywords whose value is not carried over as it is: the refusal says how to convert it.
CONVERTED_KEYWORDS = [
    (lambda: periodic.Pulse(width_seconds=1e-14),
     r'frequency_sigma_hz = 1/\(sqrt\(2\)\*pi\*width_seconds\)'),
    (lambda: _periodic_numerics(tail_steps=40), r'window_seconds = tail_steps\*time_step_seconds'),
    (lambda: _periodic_numerics(tail_tolerance=1e-3), r'threshold= is the old tail_tolerance='),
]


@pytest.mark.parametrize('make, conversion', CONVERTED_KEYWORDS,
                         ids=['width_seconds', 'tail_steps', 'tail_tolerance'])
def test_a_converted_periodic_keyword_says_how_to_convert_its_value(make, conversion):
    with pytest.raises(TypeError, match=conversion):
        make()


def test_both_periodic_tail_keywords_are_named_with_one_conversion_note():
    with pytest.raises(TypeError) as caught:
        _periodic_numerics(tail_steps=40, tail_tolerance=1e-3)
    message = str(caught.value)
    assert 'Numerics(tail_steps=...) is now Numerics(decay_check=...)' in message, message
    assert 'Numerics(tail_tolerance=...) is now Numerics(decay_check=...)' in message, message
    assert message.count('takes the window in seconds') == 1, message


#: (module, removed Result attribute, its replacement).
GRID_RESULT_RENAMED = [('native', 'fields', 'snapshots'),
                       ('native', 'polarization', 'polarization_density'),
                       ('native', 'previous_polarization', 'previous_polarization_density'),
                       ('periodic', 'fields', 'snapshots')]


def _empty_result(module):
    """A ``Result`` of ``module`` with every public field empty: only its attributes are read."""
    cls = getattr(fdtd, module).Result
    return cls(**{f.name: {} for f in dataclasses.fields(cls)
                  if f.init and not f.name.startswith('_')})


@pytest.mark.parametrize('module, old, new', GRID_RESULT_RENAMED,
                         ids=[f'{m}.Result.{old}' for m, old, _ in GRID_RESULT_RENAMED])
def test_a_removed_native_or_periodic_result_attribute_fails_naming_its_replacement(
        module, old, new):
    result = _empty_result(module)
    assert not hasattr(result, old)
    with pytest.raises(AttributeError, match=rf'{module}\.Result\.{old} is now '
                                             rf'{module}\.Result\.{new}\b') as caught:
        getattr(result, old)
    shown = _shown(caught.value)
    assert 'public-names.md' in shown and 'Did you mean' not in shown, shown
    assert new in {f.name for f in dataclasses.fields(type(result))}
    with pytest.raises(AttributeError, match="has no attribute 'unrelated'"):
        result.unrelated


@pytest.mark.parametrize('module', ['native', 'periodic'])
def test_a_lowercase_snapshot_component_is_refused_naming_its_public_spelling(module):
    """The engine spells components 'ex'..'hz'; the public declaration takes only 'Ex'..'Hz'."""
    snapshots = getattr(fdtd, module).Snapshots
    with pytest.raises(ValidationError, match=r"'Ey', not 'ey'"):
        snapshots(components=('Ex', 'ey'))
    assert snapshots(components=('Ex', 'Ey')).components == ('Ex', 'Ey')
    assert snapshots().components == ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')


def test_a_numerical_failure_is_a_wavebit_error_and_not_a_validation_error():
    """``except ValidationError`` meant for a bad input does not swallow a failed calculation."""
    from wavebit.errors import WavebitError, ValidationError, NumericalError, IncompatibleProblemError
    error = NumericalError('nonfinite', code='rcwa.nonfinite', path='result')
    assert isinstance(error, WavebitError) and not isinstance(error, ValidationError)
    assert not isinstance(error, ValueError)
    assert (error.code, error.path, error.issue.code) == ('rcwa.nonfinite', 'result', 'rcwa.nonfinite')
    assert issubclass(ValidationError, WavebitError) and issubclass(ValidationError, ValueError)
    assert issubclass(IncompatibleProblemError, ValidationError)
