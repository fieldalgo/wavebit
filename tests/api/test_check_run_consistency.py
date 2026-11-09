"""``check`` refuses what ``run`` would refuse, with a structured code and the field's path.

Each case here once passed ``check`` (and usually ``prepare``) and then failed at ``run``, or
failed with a message that named no field, a private engine name, or advice that led to
another refusal.
"""
import numpy as np
import pytest

from wavebit import Material, rcwa
from wavebit.errors import IncompatibleProblemError, ValidationError


def _rcwa(*, layer=None, top=None, theta=.2):
    layer = layer or rcwa.Layer(thickness=.2, background=Material(1.5), regions=(
        rcwa.Region(geometry=rcwa.Rectangle(size=(.3, .4)), material=Material(4.)),))
    return rcwa.Problem(stack=rcwa.Stack(period=(1., 1.), layers=(layer,), top=top or Material(1.)),
                        excitation=rcwa.Excitation(wavelength=1.3, sources=(rcwa.PlaneWave(theta=theta),)))


NUMERICS = rcwa.Numerics(orders=(2, 2))


def _refused_alike(problem, numerics=NUMERICS, backend='numpy'):
    """The one issue ``check`` reports, after asserting ``prepare`` and ``run`` raise it."""
    report = rcwa.check(problem, numerics=numerics, backend=backend)
    assert not report.ok
    (issue,) = report.issues
    for door in (rcwa.prepare, rcwa.run):
        with pytest.raises(ValidationError) as caught:
            door(problem, numerics=numerics, backend=backend)
        assert (caught.value.code, caught.value.path) == (issue.code, issue.path)
    return issue


# -- 1. A NaN or an infinity is refused at check, at the field that holds it ------------------

def _region_material(value):
    return _rcwa(layer=rcwa.Layer(thickness=.2, background=Material(1.5), regions=(
        rcwa.Region(geometry=rcwa.Rectangle(size=(.3, .4)), material=Material(value)),)))


NONFINITE = {
    'region-material': (_region_material, 'stack.layers[0].regions[0].material.epsilon'),
    'background': (lambda v: _rcwa(layer=rcwa.Layer(thickness=.2, background=Material(v))),
                   'stack.layers[0].background.epsilon'),
    'half-space': (lambda v: _rcwa(top=Material(v)), 'stack.top.epsilon'),
    'raster-epsilon': (lambda v: _rcwa(layer=rcwa.Layer(
        thickness=.2, epsilon=np.where(np.eye(8) > 0, v, 2.25))), 'stack.layers[0].epsilon'),
    'raster-mu': (lambda v: _rcwa(layer=rcwa.Layer(
        thickness=.2, epsilon=np.full((8, 8), 2.25), mu=np.where(np.eye(8) > 0, v, 1.))),
        'stack.layers[0].mu'),
}


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf, complex(2., np.nan)],
                         ids=['nan', 'inf', 'minus-inf', 'complex-nan'])
@pytest.mark.parametrize('case', sorted(NONFINITE))
def test_a_nonfinite_input_is_refused_at_check_with_its_path(case, value):
    build, path = NONFINITE[case]
    issue = _refused_alike(build(value))
    assert (issue.code, issue.path) == ('problem.nonfinite', path)
    assert 'non-finite' in issue.message


@pytest.mark.parametrize('backend', ['torch', 'jax'])
def test_a_nonfinite_input_is_refused_at_check_on_every_backend(backend):
    pytest.importorskip(backend)
    import contextlib
    x64 = __import__('jax').enable_x64(True) if backend == 'jax' else contextlib.nullcontext()
    with x64:
        issue = _refused_alike(_region_material(np.nan), backend=backend)
    assert (issue.code, issue.path) == ('problem.nonfinite', 'stack.layers[0].regions[0].material.epsilon')


def test_a_prepared_run_names_the_nonfinite_field_its_guard_refuses():
    prepared = rcwa.prepare(_region_material(4.), numerics=NUMERICS)
    live = _region_material(np.nan)
    path = 'stack.layers[0].regions[0].material.epsilon'
    (issue,) = prepared.validate(live).issues
    assert (issue.code, issue.path) == ('prepared.finite_inputs', path)
    with pytest.raises(IncompatibleProblemError) as caught:
        prepared.run(live)
    assert (caught.value.code, caught.value.path) == ('prepared.finite_inputs', path)

