"""The public surface is a decision, so it is pinned here and changed on purpose.

Wavebit follows two rules from NumPy's 2.0 cleanup (NEP 52). A public module says what it
exports in ``__all__``, and every public object has one public path. ``PUBLIC`` pins each
public module's ``__all__``: a name is added or removed by editing it here, in the same change
as the code. ``KNOWN_DUPLICATES`` lists the objects still reachable from two public paths; it
may only shrink (fieldalgo/wavebit#57 removes them). The FDTD part of the rename
(``docs/migration/public-names.md``) is checked here too: the old names and keywords fail with
an error that names their replacement.
"""
import importlib
import pkgutil

import pytest

import wavebit
from wavebit import fdtd

#: Each public module's ``__all__``, sorted.
PUBLIC = {
 'wavebit': ('Backend', 'CheckReport', 'ConductiveMaterial', 'Debye', 'DiffractionOrders',
             'DispersiveMaterial', 'Drude', 'Execution', 'FieldSamples', 'IncompatibleProblemError',
             'Issue', 'Job', 'Lattice', 'Lorentz', 'Material', 'MaterialResponse', 'NumericalError',
             'RawDiffractionSpectrum', 'SpectralFieldSamples', 'TabulatedMaterial', 'TimeSeries',
             'Units', 'UnsupportedFeatureError', 'ValidationError', 'Validity',
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
 'wavebit.fdtd.native': ('Current', 'Grid', 'Initial', 'Medium', 'Numerics', 'Oscillator',
                         'Outputs', 'PML', 'Prepared', 'Problem', 'Record', 'Result', 'Spectrum',
                         'Tail', 'check', 'prepare', 'run'),
 'wavebit.fdtd.periodic': ('Circle', 'Ellipse', 'Excitation', 'Layer', 'Numerics', 'Outputs',
                           'PlaneWave', 'Polygon', 'Prepared', 'Problem', 'Pulse',
                           'RasterDispersion', 'Record', 'Rectangle', 'Region', 'Result', 'Stack',
                           'check', 'prepare', 'run'),
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
