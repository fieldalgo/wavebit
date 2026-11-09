"""Ordinary bad input to a public constructor is a structured refusal, never a raw Python error.

Each case names the class, the ``code`` and the ``path`` a caller can match on
(``docs/reference/keywords.md``, Refusal codes). Before these were structured, each one escaped
as a bare ``TypeError`` or ``ValueError`` (a NumPy ``isfinite`` or ``complex()`` message among
them) or silently built a material whose permittivity was NaN.
"""
import json

import numpy as np
import pytest

from wavebit import Job, Material, Units, rcwa
from wavebit import materials as M
from wavebit.errors import ValidationError


def _refused(build, cls, code, path):
    with pytest.raises(cls) as caught:
        build()
    error = caught.value
    assert type(error) is cls, type(error)
    assert (error.code, error.path) == (code, path), (error.code, error.path, error.message)
    assert error.issue.code == code and error.issue.path == path
    return error


# -- Materials -------------------------------------------------------------------------------------

UM = Units(length='um')

MATERIALS = {
    'Drude omega_p None': (lambda: M.Drude(None, 1.), 'material.scalar', 'material.poles.omega_p'),
    'Drude gamma object': (lambda: M.Drude(1., object()), 'material.scalar', 'material.poles.gamma'),
    'Lorentz omega_0 None': (lambda: M.Lorentz(None, 1., 1.), 'material.scalar',
                             'material.poles.omega_0'),
    'Lorentz strength None': (lambda: M.Lorentz(1., 1., None), 'material.scalar',
                              'material.poles.strength'),
    'Debye tau object': (lambda: M.Debye(object(), 1.), 'material.scalar', 'material.poles.tau'),
    'Debye delta None': (lambda: M.Debye(1e-14, None), 'material.scalar',
                         'material.poles.delta_epsilon'),
    'Conductive conductivity None': (lambda: M.ConductiveMaterial(2., None), 'material.scalar',
                                     'material.conductivity'),
    'Conductive epsilon None': (lambda: M.ConductiveMaterial(None, 1.), 'material.shape',
                                'material.epsilon'),
    'Conductive epsilon str': (lambda: M.ConductiveMaterial('2.25', 1.), 'material.shape',
                               'material.epsilon'),
    'Conductive epsilon pair': (lambda: M.ConductiveMaterial([1., 2.], 1.), 'material.shape',
                                'material.epsilon'),
    'Conductive epsilon ragged': (lambda: M.ConductiveMaterial([[1, 0, 0], [0, 1]], 1.),
                                  'material.shape', 'material.epsilon'),
    'Conductive mu str': (lambda: M.ConductiveMaterial(2., 1., mu='x'), 'material.shape',
                          'material.mu'),
    'Dispersive epsilon_inf None': (lambda: M.DispersiveMaterial(None, (M.Drude(1., 1.),)),
                                    'material.shape', 'material.epsilon_inf'),
    'Dispersive mu shape': (lambda: M.DispersiveMaterial(2., (M.Drude(1., 1.),), mu=np.ones(2)),
                            'material.shape', 'material.mu'),
    'Dispersive poles None': (lambda: M.DispersiveMaterial(2., None), 'material.poles',
                              'material.poles'),
    'Dispersive one bare pole': (lambda: M.DispersiveMaterial(2., M.Drude(1., 1.)),
                                 'material.poles', 'material.poles'),
    'Table wavelengths None': (lambda: M.TabulatedMaterial(None, [1., 2.], wavelength_unit='nm'),
                               'wavelength.shape', 'material.wavelengths'),
    'Table wavelengths text': (lambda: M.TabulatedMaterial(['a', 'b'], [1., 2.],
                                                           wavelength_unit='nm'),
                               'wavelength.shape', 'material.wavelengths'),
    'Table wavelengths decreasing': (lambda: M.TabulatedMaterial([2., 1.], [1., 2.],
                                                                 wavelength_unit='nm'),
                                     'wavelength.value', 'material.wavelengths'),
    'Table epsilon None': (lambda: M.TabulatedMaterial([1., 2.], None, wavelength_unit='nm'),
                           'material.shape', 'material.epsilon'),
    'Table epsilon text': (lambda: M.TabulatedMaterial([1., 2.], ['a', 'b'], wavelength_unit='nm'),
                           'material.shape', 'material.epsilon'),
    'Table epsilon length': (lambda: M.TabulatedMaterial([1., 2.], [1., 2., 3.],
                                                         wavelength_unit='nm'),
                             'material.shape', 'material.epsilon'),
    'Table mu length': (lambda: M.TabulatedMaterial([1., 2.], [1., 2.], mu=[1.],
                                                    wavelength_unit='nm'),
                        'material.shape', 'material.mu'),
    'Index table tensor': (lambda: M.TabulatedMaterial.from_refractive_index(
        [1., 2.], np.ones((2, 3)), wavelength_unit='nm'), 'material.shape', 'material.index'),
    'Index table zero mu': (lambda: M.TabulatedMaterial.from_refractive_index(
        [1., 2.], [1., 2.], mu=0., wavelength_unit='nm'), 'material.shape', 'material.mu'),
}


@pytest.mark.parametrize('case', list(MATERIALS))
def test_a_material_constructor_refuses_bad_input_with_a_code_and_path(case):
    build, code, path = MATERIALS[case]
    error = _refused(build, ValidationError, code, path)
    assert path.rsplit('.', 1)[-1] in error.message


def test_a_material_refusal_is_still_the_value_error_it_was():
    """``except ValueError`` around a material keeps working: ``ValidationError`` is one."""
    with pytest.raises(ValueError, match='strictly increasing'):
        M.TabulatedMaterial([2., 1.], [1., 2.], wavelength_unit='nm')


def test_valid_material_spellings_are_still_accepted():
    """The structural check reads shapes only: tensors, NumPy scalars and lists all pass."""
    M.ConductiveMaterial(np.float64(2.), 1e3)
    M.ConductiveMaterial([2., 3., 4.], 1e3, mu=np.eye(3))
    M.ConductiveMaterial(2. + .1j, np.float32(1.))
    M.DispersiveMaterial([[2, 0, 0], [0, 2, 0], [0, 0, 3]], [M.Drude(1e15, 1e13)])
    M.TabulatedMaterial(np.array([1, 2]), [[1., 1., 1.], [2., 2., 2.]], wavelength_unit='um')
    assert M.ConductiveMaterial(2.25, 1e3).at(.5, units=UM).epsilon is not None


# -- Saved jobs ------------------------------------------------------------------------------------

def _table_job():
    table = M.TabulatedMaterial([1., 2.], [2., 3.], wavelength_unit='um')
    problem = rcwa.Problem(units=UM, stack=rcwa.Stack(top=table),
                           excitation=rcwa.Excitation(wavelength=1.5, sources=(rcwa.PlaneWave(),)))
    return json.loads(Job(solver='rcwa', problem=problem, numerics=rcwa.Numerics()).to_json())


def _with_table(key, value):
    body = _table_job()
    body['problem']['fields']['stack']['fields']['top'][key] = value
    return lambda: Job.from_json(json.dumps(body))


@pytest.mark.parametrize('key,value', [
    ('wavelengths', {'type': 'tuple', 'items': [2., 1.]}),
    ('wavelengths', None),
    ('values', {'type': 'tuple', 'items': [{'type': 'complex', 'real': 1., 'imag': 0.}]}),
    ('values', 'abc'),
    ('wavelength_unit', 'km'),
], ids=['decreasing', 'no wavelengths', 'short values', 'text values', 'unit'])
def test_a_malformed_saved_table_is_a_job_encoding_refusal(key, value):
    _refused(_with_table(key, value), ValidationError, 'job.encoding', 'problem.stack.top')


def test_a_saved_slab_with_two_axes_is_a_job_encoding_refusal():
    from wavebit import fdtd
    from wavebit.jobs import _decode, _encode, _registry
    registry = _registry()
    encoded = _encode(fdtd.SnappedSlab(z=(-.1, .1)), registry, 'shape')
    encoded['fields']['x'] = {'type': 'tuple', 'items': [-.1, .1]}
    path = 'problem.scene.structures[0].shape'
    error = _refused(lambda: _decode(encoded, registry, path), ValidationError, 'job.encoding', path)
    assert 'SnappedSlab' in error.message and 'ONE axis' in error.message


def test_a_valid_saved_table_still_round_trips():
    body = _table_job()
    restored = Job.from_json(json.dumps(body))
    assert restored.problem.stack.top == M.TabulatedMaterial([1., 2.], [2., 3.], wavelength_unit='um')
    assert json.loads(restored.to_json()) == body


# -- FDTD constructors -----------------------------------------------------------------------------

def _fdtd():
    from wavebit import fdtd
    return fdtd


FDTD = {
    'Domain negative size': (lambda f: f.Domain(size=(-1., 1., 1.)), 'fdtd.invalid',
                             'scene.domain.size'),
    'Domain size None': (lambda f: f.Domain(size=None), 'fdtd.invalid', 'scene.domain.size'),
    'Domain centre text': (lambda f: f.Domain(size=(1., 1., 1.), center='a'), 'fdtd.invalid',
                           'scene.domain.center'),
    'SnappedSlab no axis': (lambda f: f.SnappedSlab(), 'shape.dimension', 'shape'),
    'SnappedSlab scalar span': (lambda f: f.SnappedSlab(z=1.), 'shape.dimension', 'shape.z'),
    'SnappedSlab reversed': (lambda f: f.SnappedSlab(z=(1., 0.)), 'shape.dimension', 'shape.z'),
    'SnappedSlab text bound': (lambda f: f.SnappedSlab(z=('a', 0.)), 'shape.dimension', 'shape.z'),
    'SnappedBox size None': (lambda f: f.SnappedBox(size=None), 'shape.dimension', 'shape.size'),
    'SnappedBox negative': (lambda f: f.SnappedBox(size=(-1., 1., 1.)), 'shape.dimension',
                            'shape.size'),
    'Plane no normal': (lambda f: f.Plane(), 'fdtd.invalid', 'plane'),
    'Plane two normals': (lambda f: f.Plane(x=0., z=0.), 'fdtd.invalid', 'plane'),
    'Plane long span': (lambda f: f.Plane(z=0., x=(0., 1., 2.)), 'fdtd.invalid', 'plane.x'),
    'Plane reversed span': (lambda f: f.Plane(z=0., x=(1., 0.)), 'fdtd.invalid', 'plane.x'),
    'Plane text position': (lambda f: f.Plane(z='a'), 'fdtd.invalid', 'plane.z'),
    'TimeSeriesMonitor name None': (lambda f: f.TimeSeriesMonitor(None, at=(0, 0, 0)),
                                    'fdtd.invalid', 'outputs.monitors.name'),
    'TimeSeriesMonitor at scalar': (lambda f: f.TimeSeriesMonitor('p', at=1.),
                                    'monitor.geometry', 'outputs.p.at'),
    'TimeSeriesMonitor at None': (lambda f: f.TimeSeriesMonitor('p', at=None),
                                  'monitor.geometry', 'outputs.p.at'),
    'TimeSeriesMonitor component': (lambda f: f.TimeSeriesMonitor('p', at=(0, 0, 0),
                                                                  component='Q'),
                                    'fdtd.invalid', 'outputs.p.component'),
    'MirrorSymmetry parity 2': (lambda f: f.MirrorSymmetry(parity=2), 'boundaries.face',
                                'MirrorSymmetry.parity'),
    'MirrorSymmetry parity None': (lambda f: f.MirrorSymmetry(parity=None), 'boundaries.face',
                                   'MirrorSymmetry.parity'),
    'InvariantAxis k text': (lambda f: f.InvariantAxis(k='a'), 'boundaries.face',
                             'InvariantAxis.k'),
    'InvariantAxis k above k_max': (lambda f: f.InvariantAxis(k=1., k_max=.5), 'boundaries.face',
                                    'InvariantAxis'),
    'PML reference_media int': (lambda f: f.PML(reference_media=3), 'numerics.absorber',
                                'numerics.absorber.reference_media'),
    'PML reference_media None': (lambda f: f.PML(reference_media=None), 'numerics.absorber',
                                 'numerics.absorber.reference_media'),
    'PML reference_media str': (lambda f: f.PML(reference_media='z-'), 'numerics.absorber',
                                'numerics.absorber.reference_media'),
    'PML reference_media list': (lambda f: f.PML(reference_media=['z-']), 'numerics.absorber',
                                 'numerics.absorber.reference_media'),
}


@pytest.mark.parametrize('case', list(FDTD))
def test_an_fdtd_constructor_refuses_bad_input_with_a_code_and_path(case):
    build, code, path = FDTD[case]
    fdtd = _fdtd()
    error = _refused(lambda: build(fdtd), ValidationError, code, path)
    # The engine's own explanation is kept, in the public vocabulary.
    assert 'fd.' not in error.message, error.message


def test_a_traced_fdtd_length_is_a_static_input_refusal():
    jax = pytest.importorskip('jax')
    fdtd = _fdtd()
    with pytest.raises(ValidationError) as caught:
        jax.grad(lambda v: (fdtd.Domain(size=(v, 1., 1.)), v)[1])(.3)
    assert (caught.value.code, caught.value.path) == ('prepared.static_input', 'scene.domain.size')
    assert 'is static' in caught.value.message


def test_an_unknown_fdtd_keyword_is_still_a_type_error():
    """A keyword the declaration does not have is Python's TypeError, as on every declaration."""
    fdtd = _fdtd()
    with pytest.raises(TypeError):
        fdtd.Domain(sizes=(1., 1., 1.))
    with pytest.raises(TypeError):
        fdtd.TimeSeriesMonitor('p', point=(0, 0, 0))


def test_valid_fdtd_constructions_are_unchanged():
    fdtd = _fdtd()
    assert fdtd.SnappedSlab(x=[np.float64(-1), 1]).x == (-1., 1.)
    assert fdtd.Plane(y=.1, x=(-.3, .3)).x == (-.3, .3)
    assert fdtd.MirrorSymmetry(parity=-1.).parity == -1
    assert fdtd.TimeSeriesMonitor('p', at=(0, 0, np.float64(.1))).at == (0., 0., .1)
    assert dict(fdtd.PML(reference_media={'z-': Material(2.25)}).reference_media) == {
        'z-': Material(2.25)}
    assert dict(fdtd.PML().reference_media) == {}


# -- RCWA sampling ---------------------------------------------------------------------------------

def _rcwa_problem(background=2., fields=False):
    layer = rcwa.Layer(thickness=.5, background=Material(background), regions=(
        rcwa.Region(geometry=rcwa.Circle(radius=.2), material=Material(4.)),))
    return rcwa.Problem(units=UM, stack=rcwa.Stack(period=(1., 1.), layers=(layer,)),
                        excitation=rcwa.Excitation(wavelength=1.5, sources=(rcwa.PlaneWave(),)),
                        outputs=rcwa.Outputs(fields=fields))


@pytest.mark.parametrize('background,component', [
    (2., 'xy'), (2., 'qq'), (2., 3), (2., ('x', 'y')), (2., np.array(['xx'])),
    ([2., 3., 4.], None), ([2., 3., 4.], 'xyz'), ([2., 3., 4.], 1),
], ids=['scalar layer, a component', 'unknown pair', 'integer', 'tuple', 'array',
        'tensor layer, no component', 'triple', 'tensor, integer'])
def test_sample_structure_refuses_a_component_it_cannot_sample(background, component):
    error = _refused(lambda: rcwa.sample_structure(_rcwa_problem(background), samples=8,
                                                   component=component),
                     ValidationError, 'structure.component', 'component')
    assert 'component' in error.message


def test_sample_structure_still_samples_valid_components():
    tensor = _rcwa_problem([2., 3., 4.])
    assert rcwa.sample_structure(tensor, samples=8, component='zz').shape == (8, 8)
    assert rcwa.sample_structure(_rcwa_problem(), samples=8).shape == (8, 8)


@pytest.fixture(scope='module')
def field_result():
    return rcwa.run(_rcwa_problem(fields=True), numerics=rcwa.Numerics(orders=(1, 1)))


@pytest.mark.parametrize('side,part,path', [
    ('left', 'total', 'side'), (None, 'total', 'side'), (np.array(['top']), 'total', 'side'),
    ('top', 'foo', 'part'), ('top', None, 'part'),
])
def test_boundary_fields_refuses_an_unknown_side_or_part(field_result, side, part, path):
    _refused(lambda: field_result.boundary_fields(side=side, x=2, y=2, part=part),
             ValidationError, f'field.{path}', path)


def test_boundary_fields_still_samples_every_side_and_part(field_result):
    for side in ('top', 'bottom'):
        for part in ('incoming', 'outgoing', 'total'):
            assert field_result.boundary_fields(side=side, x=2, y=2, part=part).values.shape[-1] == 6
