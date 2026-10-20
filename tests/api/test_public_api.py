"""The public surface is a decision, so it is pinned here and changed on purpose.

Wavebit follows two rules from NumPy's 2.0 cleanup (NEP 52). A public module says what it
exports in ``__all__``, and every public object has one public path. ``PUBLIC`` pins each
public module's ``__all__``: a name is added or removed by editing it here, in the same change
as the code. ``KNOWN_DUPLICATES`` lists the objects still reachable from two public paths; it
may only shrink (fieldalgo/wavebit#57 removes them). The FDTD part of the rename
(``docs/migration/public-names.md``), ``fdtd.native`` and ``fdtd.periodic`` included, is checked
here too, as are the shared, observation and RCWA renames: the old names and keywords fail with
an error that names their replacement.
"""
import dataclasses
import importlib
import pkgutil
import re
import traceback

import numpy as np

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
             'RawDiffractionSpectrum', 'RawFieldSpectrum', 'TabulatedMaterial', 'TimeSeries',
             'Units', 'UnsupportedFeatureError', 'ValidationError', 'Validity', 'WavebitError',
             'eps_from_refractive_index', 'fdtd', 'layer_thicknesses', 'materials', 'rcwa',
             'refractive_index_from_eps', 'run'),
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
 'wavebit.lattice': ('DrawnOn', 'Lattice', 'as_lattice', 'meshgrid_xy'),
 'wavebit.materials': ('ConductiveMaterial', 'Debye', 'DispersiveMaterial', 'Drude', 'Lorentz',
                       'Material', 'MaterialResponse', 'TabulatedMaterial', 'available',
                       'eps_from_refractive_index', 'load', 'refractive_index_from_eps'),
 'wavebit.materials.catalog': ('available', 'load'),
 'wavebit.migration': ('migrate_legacy_jobs',),
 'wavebit.observations': ('DiffractionOrder', 'DiffractionOrders', 'FieldSamples', 'ModeFit',
                          'RawDiffractionSpectrum', 'RawFieldSpectrum', 'TimeSeries',
                          'Validity', 'fit_modes'),
 'wavebit.observations.modes': ('ModeFit', 'fit_modes'),
 'wavebit.rcwa': ('BoundaryPower', 'Circle', 'ElectricDipole', 'Ellipse', 'Excitation',
                  'GaussianBeam', 'IncidentOrders', 'Layer', 'Numerics', 'Outputs', 'PlaneWave',
                  'Polygon', 'Prepared', 'Problem', 'Rectangle', 'Region', 'Result', 'Stack',
                  'check', 'prepare', 'run', 'sample_structure'),
 'wavebit.rcwa.diagnostics': ('AutoDetuneWarning', 'ComplexIncidenceWarning',
                              'ExceptionalPointWarning', 'GainCutoffWarning',
                              'RayleighPointWarning'),
 'wavebit.rcwa.experimental': ('AffineProfileNVF', 'CartesianPairOpeningPlan',
                               'CartesianPairTransmissionGalerkinGapCell', 'CartesianRectangleNVF',
                               'CartesianRectanglePairNVF', 'ConformingNVF'),
 'wavebit.rcwa.operators': ('ConvMatrices', 'Operators', 'Port', 'ScatteringMatrix',
                            'TensorConvMatrices', 'assemble', 'solve'),
 'wavebit.records': ('RecordedObject', 'RunRecord', 'load_run', 'run_recorded', 'save_run')
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
 ('wavebit.RawFieldSpectrum', 'wavebit.observations.RawFieldSpectrum'),
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


# ---------------------------------------------------------------------------------------------
# Shared modules (#57 section 4): units, lattice, materials, execution
# ---------------------------------------------------------------------------------------------

SHARED_RENAMED = (
    (lambda: wavebit.Units(length='um'), 'length_metres', 'Units.length_unit_meters'),
    (lambda: wavebit.Units(length='um'), 'time_seconds', 'Units.time_unit_seconds'),
    (lambda: wavebit.Units(length='um'), 'time_to_internal', 'time_unit_seconds'),
    (lambda: wavebit.Material(2.25), 'is_anisotropic', 'Material.is_tensor'),
    (lambda: wavebit.TabulatedMaterial([1., 2.], [2.25, 2.25], wavelength_unit='um'), 'interpolation',
     'TabulatedMaterial.interpolated_quantity'),
    (lambda: wavebit.Execution(), 'scope', 'private'),
)


@pytest.mark.parametrize('build, old, replacement', SHARED_RENAMED,
                         ids=[row[1] for row in SHARED_RENAMED])
def test_a_renamed_shared_member_fails_naming_its_replacement(build, old, replacement):
    value = build()
    with pytest.raises(AttributeError) as caught:
        getattr(value, old)
    message = str(caught.value)
    assert replacement in message and 'public-names.md' in message, message
    assert 'Did you mean' not in ''.join(traceback.format_exception(caught.value))
    with pytest.raises(AttributeError, match='has no attribute'):
        value.no_such_member
    assert not hasattr(value, 'dtype')


def test_the_renamed_shared_members_answer_what_they_did():
    units = wavebit.Units(length='um')
    assert units.length_unit_meters == 1e-6
    assert units.time_unit_seconds == 1e-6/wavebit.units.C0
    assert wavebit.Material((2., 2., 2.)).is_tensor and not wavebit.Material(2.).is_tensor
    table = wavebit.TabulatedMaterial([1., 2.], [2.25, 2.25], wavelength_unit='um')
    assert table.interpolated_quantity == 'epsilon'
    assert 'interpolated_quantity=' in repr(table) and 'interpolation=' not in repr(table)


def test_oblique_takes_the_lengths_of_the_two_lattice_vectors():
    import inspect
    assert list(inspect.signature(wavebit.Lattice.oblique).parameters) == [
        'length_u', 'length_v', 'angle']
    a = wavebit.Lattice.oblique(1., 2., 1.)
    b = wavebit.Lattice.oblique(length_u=1., length_v=2., angle=1.)
    assert np.array_equal(a.u, b.u) and np.array_equal(a.v, b.v)
    with pytest.raises(TypeError, match=r'Lattice\.oblique\(lx=\.\.\.\) is now '
                                        r'Lattice\.oblique\(length_u=\.\.\.\)') as caught:
        wavebit.Lattice.oblique(lx=1., ly=2., angle=1.)
    assert 'length_v' in str(caught.value) and 'public-names.md' in str(caught.value)


@pytest.mark.parametrize('name', ['time_domain', 'tabulated'])
def test_a_former_materials_module_is_not_a_public_path(name):
    import importlib
    import wavebit.materials as materials
    with pytest.raises(AttributeError, match='no longer a public module') as caught:
        getattr(materials, name)
    assert 'Did you mean' not in ''.join(traceback.format_exception(caught.value))
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f'wavebit.materials.{name}')


# ---------------------------------------------------------------------------------------------
# Observation and result objects (#57 section 4)
# ---------------------------------------------------------------------------------------------

#: (module, removed name, its replacement in the same module).
OBSERVATION_MODULE_RENAMED = [('wavebit', 'SpectralFieldSamples', 'RawFieldSpectrum'),
                              ('wavebit.observations', 'SpectralFieldSamples', 'RawFieldSpectrum'),
                              ('wavebit.records', 'RecordedFields', 'RecordedObject')]


@pytest.mark.parametrize('module, old, new', OBSERVATION_MODULE_RENAMED,
                         ids=[f'{module}.{old}' for module, old, _ in OBSERVATION_MODULE_RENAMED])
def test_a_removed_observation_name_fails_naming_its_replacement(module, old, new):
    namespace = importlib.import_module(module)
    assert not hasattr(namespace, old)
    with pytest.raises(AttributeError, match=rf'{module}\.{old} is now {module}\.{new}\b') as caught:
        getattr(namespace, old)
    shown = ''.join(traceback.format_exception(caught.value))
    assert 'public-names.md' in shown and 'Did you mean' not in shown, shown
    assert new in namespace.__all__
    with pytest.raises(AttributeError, match="has no attribute 'unrelated'"):
        getattr(namespace, 'unrelated')


def _time_series(**kw):
    from wavebit.observations import TimeSeries
    return TimeSeries(coordinates={}, values=np.zeros(2), components=('Ex',), units='V/m',
                      sampling='raw', **kw)


def _field_spectrum(**kw):
    from wavebit.observations import RawFieldSpectrum
    return RawFieldSpectrum(coordinates={}, values=np.zeros(1), components=('Ex',), units='V*s/m',
                            sampling='raw', wavelengths=(1.,), **kw)


def _orders(**kw):
    from wavebit.observations import DiffractionOrders
    return DiffractionOrders(**{'wavelengths': (1.,), 'orders_u': (0,), 'orders_v': (0,), **kw})


def _order(**kw):
    from wavebit.observations import DiffractionOrder
    return DiffractionOrder(wavelengths=(1.,), order_u=0, order_v=0, **kw)


def _raw_orders(**kw):
    from wavebit.observations import RawDiffractionSpectrum
    return RawDiffractionSpectrum(wavelengths=(1.,), orders_u=(0,), orders_v=(0,), **kw)


def _flux(**kw):
    kw = {'frequencies_hz': np.ones(1), 'spectral_energy': np.ones(1), **kw}
    return fdtd.SpectralFlux(wavelengths=(1.,), units='J/Hz', **kw)


def _fdtd_result(**kw):
    from wavebit.observations import Validity
    return fdtd.Result(outputs={}, validity=Validity(), diagnostics={}, provenance={}, **kw)


#: (builder, removed attribute, the replacement its message names).
OBSERVATION_RENAMED = [
    (_time_series, 'times', 'TimeSeries.times_seconds'),
    (_time_series, 'time', 'TimeSeries.times_seconds'),
    (_field_spectrum, 'frequencies', 'RawFieldSpectrum.frequencies_hz'),
    (_field_spectrum, 'spectrum', 'RawFieldSpectrum.sidedness'),
    (_orders, 'cartesian', 'DiffractionOrders.amplitude_xyz'),
    (_orders, 'kz', 'DiffractionOrders.k_normal'),
    (_orders, 'at_wavelength', 'DiffractionOrders.at_wavelength_index'),
    (_order, 'cartesian', 'DiffractionOrder.amplitude_xyz'),
    (_order, 'kz', 'DiffractionOrder.k_normal'),
    (_raw_orders, 'cartesian', 'RawDiffractionSpectrum.amplitude_xyz'),
    (_raw_orders, 'kz', 'RawDiffractionSpectrum.k_normal'),
    (_raw_orders, 'spectrum', 'RawDiffractionSpectrum.sidedness'),
    (_raw_orders, 'at_wavelength', 'RawDiffractionSpectrum.at_wavelength_index'),
    (_flux, 'frequencies', 'fdtd.SpectralFlux.frequencies_hz'),
    (_flux, 'energy_density', 'fdtd.SpectralFlux.spectral_energy'),
    (_fdtd_result, 'sources', 'fdtd.Result.source_references'),
]


@pytest.mark.parametrize('build, old, replacement', OBSERVATION_RENAMED,
                         ids=[f'{row[2].rsplit(".", 1)[0]}.{row[1]}' for row in OBSERVATION_RENAMED])
def test_a_renamed_observation_member_fails_naming_its_replacement(build, old, replacement):
    value = build()
    assert not hasattr(value, old)
    with pytest.raises(AttributeError, match=rf'\.{old} is now {replacement}\b') as caught:
        getattr(value, old)
    shown = ''.join(traceback.format_exception(caught.value))
    assert 'public-names.md' in shown and 'Did you mean' not in shown, shown
    assert hasattr(value, replacement.rsplit('.', 1)[1])
    with pytest.raises(AttributeError) as caught:
        value.unrelated
    assert str(caught.value) == f"'{type(value).__name__}' object has no attribute 'unrelated'"


#: (builder, removed keyword, a value it took, its replacement). Each class of a hierarchy is
#: listed on its own: a subclass regenerates its constructor, so each must refuse by itself.
OBSERVATION_RENAMED_KEYWORDS = [
    (_time_series, 'times', np.arange(2.), 'times_seconds'),
    (_field_spectrum, 'frequencies', np.ones(1), 'frequencies_hz'),
    (_field_spectrum, 'spectrum', 'two-sided', 'sidedness'),
    (_orders, 'cartesian', np.zeros((1, 1, 1, 3)), 'amplitude_xyz'),
    (_orders, 'kz', np.ones((1, 1, 1)), 'k_normal'),
    (_order, 'cartesian', np.zeros((1, 3)), 'amplitude_xyz'),
    (_order, 'kz', np.ones(1), 'k_normal'),
    (_raw_orders, 'cartesian', np.zeros((1, 1, 1, 3)), 'amplitude_xyz'),
    (_raw_orders, 'kz', np.ones((1, 1, 1)), 'k_normal'),
    (_raw_orders, 'spectrum', 'two-sided', 'sidedness'),
    (_flux, 'frequencies', np.ones(1), 'frequencies_hz'),
    (_flux, 'energy_density', np.ones(1), 'spectral_energy'),
    (_fdtd_result, 'sources', {}, 'source_references'),
]


@pytest.mark.parametrize('build, old, value, new', OBSERVATION_RENAMED_KEYWORDS,
                         ids=[f'{row[0].__name__}:{row[1]}->{row[3]}'
                              for row in OBSERVATION_RENAMED_KEYWORDS])
def test_a_renamed_observation_keyword_fails_naming_its_replacement(build, old, value, new):
    owner = type(build()).__name__
    with pytest.raises(TypeError, match=rf'{owner}\({old}=\.\.\.\) is now {owner}\({new}=\.\.\.\)') as caught:
        build(**{old: value})
    shown = _shown(caught.value)
    assert 'public-names.md' in shown and 'did you mean' not in shown.lower(), shown
    assert getattr(build(**{new: value}), new) is not None


def test_periodic_sample_names_its_seconds_keyword():
    result = _empty_result('periodic')
    with pytest.raises(TypeError, match=r'periodic\.Result\.sample\(times=\.\.\.\) is now '
                                        r'periodic\.Result\.sample\(times_seconds=\.\.\.\)') as caught:
        result.sample(z=0., times=(0.,))
    assert 'public-names.md' in _shown(caught.value)
    import inspect
    assert 'times_seconds' in inspect.signature(periodic.Result.sample).parameters


def test_the_renamed_observation_members_answer_what_they_did():
    orders = _orders(wavelengths=(1., 2.), k_normal=np.arange(2.).reshape(2, 1, 1),
                     amplitude_xyz=np.arange(6.).reshape(2, 1, 1, 3))
    selected = orders.at_wavelength_index(1)
    assert selected.wavelengths == (2.,) and selected.k_normal.tolist() == [[[1.]]]
    order = orders.order(0)
    assert order.k_normal.tolist() == [0., 1.] and order.amplitude_xyz.shape == (2, 3)
    assert _raw_orders().sidedness == _field_spectrum().sidedness == 'two-sided'
    assert _time_series(times_seconds=np.arange(2.)).times_seconds.tolist() == [0., 1.]
    assert 'times_seconds=' in repr(_time_series()) and 'times=' not in repr(_time_series())


@pytest.mark.parametrize('build', [_time_series, _field_spectrum, _orders, _order, _raw_orders,
                                   _flux, _fdtd_result], ids=lambda build: build.__name__)
def test_a_tombstoned_observation_survives_a_jit_round_trip(build):
    """The class ``__getattr__`` reads no instance state, so flatten/unflatten (which build an
    object field by field) and ``jit`` see the fields only, and the tombstones still answer."""
    jax = pytest.importorskip('jax')
    value = build()                   # built after JAX is imported, so it is a registered pytree
    leaves, tree = jax.tree_util.tree_flatten(value)
    assert type(jax.tree_util.tree_unflatten(tree, leaves)) is type(value)
    out = jax.jit(lambda v: v)(value)
    assert type(out) is type(value)
    assert [f.name for f in dataclasses.fields(out)] == [f.name for f in dataclasses.fields(value)]
    old = next(old for b, old, _ in OBSERVATION_RENAMED if b is build)
    with pytest.raises(AttributeError, match='is now'):
        getattr(out, old)


# ---------------------------------------------------------------------------------------------
# RCWA (#57 sections 4 and 6)
# ---------------------------------------------------------------------------------------------

from wavebit import rcwa  # noqa: E402
from wavebit.rcwa import experimental, operators  # noqa: E402

#: (removed name in wavebit.rcwa, the full path that replaced it).
RCWA_RENAMED = [('OrderExcitation', 'wavebit.rcwa.IncidentOrders'),
                ('layer_thicknesses', 'wavebit.layer_thicknesses')]


@pytest.mark.parametrize('old, new', RCWA_RENAMED, ids=[old for old, _ in RCWA_RENAMED])
def test_a_removed_rcwa_name_fails_naming_its_replacement(old, new):
    assert not hasattr(rcwa, old)
    with pytest.raises(AttributeError, match=rf'wavebit\.rcwa\.{old} is now {re.escape(new)}\b') as caught:
        getattr(rcwa, old)
    shown = _shown(caught.value)
    assert 'public-names.md' in shown and 'Did you mean' not in shown, shown
    module, name = new.rsplit('.', 1)
    assert name in importlib.import_module(module).__all__
    with pytest.raises(AttributeError, match="has no attribute 'unrelated'"):
        getattr(rcwa, 'unrelated')


def test_rcwa_plotting_is_no_longer_a_public_path():
    with pytest.raises(AttributeError, match='no longer public') as caught:
        getattr(rcwa, 'plotting')
    shown = _shown(caught.value)
    assert 'public-names.md' in shown and 'Did you mean' not in shown, shown
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module('wavebit.rcwa.plotting')


def test_dir_of_rcwa_is_its_public_surface():
    listed = [name for name in dir(rcwa) if not name.startswith('__')]
    assert sorted(listed) == sorted(set(rcwa.__all__) | {'diagnostics', 'experimental', 'operators'})


def test_the_rcwa_warnings_have_one_public_home():
    """The engine warns with these very classes, so a filter on the public name catches them."""
    from wavebit._common import backend
    from wavebit.rcwa import diagnostics
    for name in diagnostics.__all__:
        assert getattr(diagnostics, name) is getattr(backend, name)
        text = getattr(diagnostics, name).__doc__
        for engine in ('auto_detune', 'truncation_order', 'OrderExcitation', '_engine'):
            assert engine not in text, (name, engine)
        assert f'wavebit.rcwa.diagnostics.{name}' in text


def test_a_complex_angle_warns_with_the_public_category():
    from wavebit import Material, Units
    from wavebit.rcwa import diagnostics
    problem = rcwa.Problem(units=Units(length='um'), stack=rcwa.Stack(period=(.8, .9), layers=(
        rcwa.Layer(thickness=.2, background=Material(2.25)),)),
        excitation=rcwa.Excitation(wavelength=.6, sources=(rcwa.PlaneWave(theta=.2+.1j),)))
    with pytest.warns(diagnostics.ComplexIncidenceWarning):
        rcwa.run(problem, numerics=rcwa.Numerics())


def _gap_cell(**kw):
    return experimental.CartesianPairTransmissionGalerkinGapCell(
        experimental.CartesianPairOpeningPlan(0), reference=(12.25, -10.+1j, 1.), **kw)


#: (constructor with the keywords it needs, removed keyword, a value it took, its replacement).
RCWA_RENAMED_KEYWORDS = [
    (lambda **kw: rcwa.IncidentOrders(**kw), 'orders', (((0, 0), 1.),), 'weights'),
    (lambda **kw: rcwa.GaussianBeam(waist=1., **kw), 'spectral_extent', 3., 'angular_spectrum_extent'),
    (lambda **kw: experimental.ConformingNVF(**kw), 'truncation_order', (1, 1), 'orders'),
    (lambda **kw: experimental.AffineProfileNVF(**kw), 'truncation_order', (1, 1), 'orders'),
    (lambda **kw: experimental.CartesianRectangleNVF(**kw), 'truncation_order', (1, 1), 'orders'),
    (lambda **kw: experimental.CartesianRectanglePairNVF(**kw), 'truncation_order', (1, 1), 'orders'),
    (_gap_cell, 'order', (1, 1), 'orders'),
]


@pytest.mark.parametrize('make, old, value, new', RCWA_RENAMED_KEYWORDS,
                         ids=[f'{i}:{old}->{new}' for i, (_, old, _, new)
                              in enumerate(RCWA_RENAMED_KEYWORDS)])
def test_a_removed_rcwa_keyword_fails_naming_its_replacement(make, old, value, new):
    with pytest.raises(TypeError, match=rf'\({old}=\.\.\.\) is now \w+\({new}=\.\.\.\)') as caught:
        make(**{old: value})
    shown = _shown(caught.value)
    assert 'public-names.md' in shown and 'did you mean' not in shown.lower(), shown
    made = make(**{new: value})
    assert getattr(made, new) == value


def test_sample_structure_names_its_samples_keyword():
    import inspect
    from wavebit import Material, Units
    problem = rcwa.Problem(units=Units(length='um'), stack=rcwa.Stack(period=(.8, .9), layers=(
        rcwa.Layer(thickness=.2, background=Material(2.25)),)),
        excitation=rcwa.Excitation(wavelength=.6, sources=(rcwa.PlaneWave(),)))
    with pytest.raises(TypeError, match=r'sample_structure\(n=\.\.\.\) is now '
                                        r'sample_structure\(samples=\.\.\.\)') as caught:
        rcwa.sample_structure(problem, n=8)
    assert 'public-names.md' in _shown(caught.value)
    with pytest.raises(TypeError, match=r"does not take \['size'\]"):
        rcwa.sample_structure(problem, size=8)
    with pytest.raises(ValidationError, match='samples must be') as caught:
        rcwa.sample_structure(problem, samples=0)
    assert caught.value.path == 'samples'
    assert rcwa.sample_structure(problem, samples=8).shape == (8, 8)
    assert 'samples' in inspect.signature(rcwa.sample_structure).parameters


def _port():
    return operators.Port(kx_over_k0=np.zeros(1), ky_over_k0=None, refractive_index=1.5, mu=1.,
                          channel='sp')


#: (builder, removed attribute, the replacement its message names).
RCWA_RENAMED_ATTRIBUTES = [
    (lambda: rcwa.BoundaryPower(incoming=0., outgoing=0., units='W', power_region='cell'),
     'integration_measure', 'rcwa.BoundaryPower.power_region'),
    (_port, 'kx', 'rcwa.operators.Port.kx_over_k0'),
    (_port, 'ky', 'rcwa.operators.Port.ky_over_k0'),
    (_port, 'index', 'rcwa.operators.Port.refractive_index'),
    (lambda: experimental.ConformingNVF((1, 1)), 'truncation_order',
     'rcwa.experimental.ConformingNVF.orders'),
    (lambda: experimental.AffineProfileNVF((1, 1)), 'truncation_order',
     'rcwa.experimental.AffineProfileNVF.orders'),
    (lambda: experimental.CartesianRectangleNVF((1, 1)), 'truncation_order',
     'rcwa.experimental.CartesianRectangleNVF.orders'),
    (lambda: experimental.CartesianRectanglePairNVF((1, 1)), 'truncation_order',
     'rcwa.experimental.CartesianRectanglePairNVF.orders'),
    (_gap_cell, 'order', 'rcwa.experimental.CartesianPairTransmissionGalerkinGapCell.orders'),
    (lambda: rcwa.IncidentOrders(weights=(((0, 0), 1.),)), 'orders', 'rcwa.IncidentOrders.weights'),
    (lambda: rcwa.GaussianBeam(waist=1.), 'spectral_extent',
     'rcwa.GaussianBeam.angular_spectrum_extent'),
]


def test_the_transmission_gap_cell_still_reads_its_settings():
    # Before the wrapper the public class was the engine cell; only ``order`` was renamed.
    cell = _gap_cell()
    names = ('reference', 'branches', 'quadrature', 'angular_quadrature', 'angular_order',
             'cutoff_order', 'root_order', 'exponents', 'junctions', 'corner_controls', 'mesh')
    for name in names:
        assert repr(getattr(cell, name)) == repr(getattr(cell._cell, name)), name


@pytest.mark.parametrize('build, old, replacement', RCWA_RENAMED_ATTRIBUTES,
                         ids=[row[2].rsplit('.', 2)[-2] + '.' + row[1]
                              for row in RCWA_RENAMED_ATTRIBUTES])
def test_a_renamed_rcwa_attribute_fails_naming_its_replacement(build, old, replacement):
    value = build()
    assert not hasattr(value, old)
    with pytest.raises(AttributeError, match=rf'\.{old} is now {re.escape(replacement)}\b') as caught:
        getattr(value, old)
    shown = _shown(caught.value)
    assert 'public-names.md' in shown and 'Did you mean' not in shown, shown
    assert hasattr(value, replacement.rsplit('.', 1)[1])
    with pytest.raises(AttributeError) as caught:
        value.unrelated
    assert str(caught.value) == f"'{type(value).__name__}' object has no attribute 'unrelated'"


def test_the_renamed_rcwa_members_answer_what_they_did():
    power = rcwa.BoundaryPower(incoming=1., outgoing=0., units='W/m^2',
                               power_region='infinite-area mean')
    assert power.power_region == 'infinite-area mean'
    assert 'power_region=' in repr(power) and 'integration_measure' not in repr(power)
    with pytest.raises(TypeError, match=r'BoundaryPower\(integration_measure=\.\.\.\) is now '
                                        r'BoundaryPower\(power_region=\.\.\.\)'):
        rcwa.BoundaryPower(incoming=1., outgoing=0., units='W', integration_measure='cell')
    port = _port()
    assert port._fields == ('kx_over_k0', 'ky_over_k0', 'refractive_index', 'mu', 'channel')
    assert port.refractive_index == 1.5 and port.ky_over_k0 is None
    assert experimental.CartesianRectanglePairNVF((1, 2)).orders == (1, 2)
    assert _gap_cell(orders=(1, 2)).orders == (1, 2)
    with pytest.raises(ValidationError, match=r'ConformingNVF\(orders=\.\.\.\) must be'):
        experimental.ConformingNVF(-1)


def test_a_port_named_tuple_constructor_cannot_name_its_replacement():
    """``typing.NamedTuple`` refuses a custom ``__new__``, so the old keywords get Python's own
    error; the migration guide names the replacement. Only attribute reads are tombstoned."""
    with pytest.raises(TypeError, match="unexpected keyword argument 'kx'"):
        operators.Port(kx=0., ky=None, index=1., mu=1., channel='sp')
    with pytest.raises(ValueError, match="unexpected field names: .*'kx'"):
        _port()._replace(kx=0.)


def test_a_tombstoned_port_survives_a_jit_round_trip():
    jax = pytest.importorskip('jax')
    from wavebit.rcwa._engine import scattering
    port = _port()
    scattering._register_pytree_if_jax_loaded()
    out = jax.jit(lambda p: p)(port)
    assert type(out) is operators.Port and out.channel == 'sp'
    with pytest.raises(AttributeError, match='is now'):
        out.index


def _fields_result():
    from wavebit import Material, Units
    problem = rcwa.Problem(units=Units(length='um'), stack=rcwa.Stack(period=(.8, .9), layers=(
        rcwa.Layer(thickness=.2, background=Material(2.25)),)),
        excitation=rcwa.Excitation(wavelength=.6, sources=(rcwa.PlaneWave(),)),
        outputs=rcwa.Outputs(fields=True))
    return rcwa.run(problem, numerics=rcwa.Numerics())


@pytest.mark.parametrize('coordinates, axis', [
    ({'x': np.zeros((2, 2))}, 'x'), ({'x': 'a'}, 'x'), ({'y': []}, 'y'), ({'y': 0}, 'y'),
    ({'z': [np.nan]}, 'z'), ({'z': 1.5}, 'z'), ({'x': [1j]}, 'x'), ({'z': [5.]}, 'z')])
def test_field_coordinates_are_refused_under_their_public_names(coordinates, axis):
    """The sampler names ``sample_x``; the caller wrote ``x``."""
    result = _fields_result()
    with pytest.raises(ValidationError) as caught:
        result.field(**{'x': [0.], 'y': [0.], 'z': [.1], **coordinates})
    assert caught.value.code == 'field.coordinates' and caught.value.path == axis
    assert 'sample_' not in str(caught.value)
    if axis != 'z':
        with pytest.raises(ValidationError) as caught:
            result.boundary_fields(side='top', **{'x': [0.], 'y': [0.],
                                                  **{k: v for k, v in coordinates.items()}})
        assert caught.value.path == axis and 'sample_' not in str(caught.value)
    assert result.field(x=3, y=[0.], z=[.1]).values.shape == (1, 1, 1, 3, 6)


def test_the_public_door_rewords_the_engine_detuning_advice():
    """The engine's ``solve`` retries by default; a public run only with Numerics(detuning=)."""
    from wavebit._common.backend import check_finite
    from wavebit.rcwa._lifecycle import _public
    with pytest.raises(FloatingPointError) as caught:
        check_finite(np, r=np.array([np.nan]))
    engine = str(caught.value)
    assert 'solve(auto_detune=True), the default' in engine
    public = _public(engine)
    assert 'auto_detune' not in public and 'truncation_order' not in public, public
    assert 'rcwa.Numerics(detuning=...) makes that retry' in public
    assert 'off by default (detuning=None)' in public
