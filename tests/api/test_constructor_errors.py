"""Ordinary bad input to a public constructor is a structured refusal, never a raw Python error.

Each case names the class, the ``code`` and the ``path`` a caller can match on
(``docs/reference/keywords.md``, Refusal codes). Before these were structured, each one escaped
as a bare ``TypeError`` or ``ValueError`` (a NumPy ``isfinite`` or ``complex()`` message among
them) or silently built a material whose permittivity was NaN.
"""
import json

import numpy as np
import pytest

from wavebit import Job, Units, rcwa
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
