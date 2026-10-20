"""One ``area.item`` grammar for every public refusal code, and one ``orders=`` rule.

Every structured failure a public caller can meet carries a lowercase dotted code
(``numerics.orders``, ``prepared.structure``, ``native.invalid``); none is the bare
``invalid_input`` an uncoded construction used to get. Every public ``orders=`` takes an
explicit ``(Mu, Mv)`` pair of nonnegative integers and refuses a bare integer and a bool.
"""
import ast
import pathlib
import re

import numpy as np
import pytest

import wavebit
from wavebit import Execution, Material, Units, fdtd, rcwa
from wavebit.errors import (IncompatibleProblemError, Issue, NumericalError,
                            UnsupportedFeatureError, ValidationError, WavebitError)
from wavebit.fdtd import native, periodic
from wavebit.materials import Drude
from wavebit.rcwa import experimental
from wavebit.units import C0

#: Two or more lowercase dotted parts, each a snake-case word.
GRAMMAR = re.compile(r'[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+')
ERRORS = {'ValidationError', 'UnsupportedFeatureError', 'IncompatibleProblemError',
          'NumericalError', 'WavebitError', 'Issue'}


def conforms(code):
    return isinstance(code, str) and GRAMMAR.fullmatch(code) is not None


def _rendered(node):
    """Each code a ``code=`` expression can produce, a computed part written as ``x``.

    An expression that only copies another error's code (``exc.code``) renders as nothing.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.JoinedStr):
        return [''.join(v.value if isinstance(v, ast.Constant) else 'x' for v in node.values)]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return [a + b for a in _rendered(node.left) or ['x'] for b in _rendered(node.right) or ['x']]
    if isinstance(node, ast.IfExp):
        return _rendered(node.body) + _rendered(node.orelse)
    if isinstance(node, ast.BoolOp):
        return [c for v in node.values for c in _rendered(v)]
    return []


def _sites():
    root = pathlib.Path(wavebit.__file__).parent
    for path in sorted(root.rglob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, 'id', getattr(node.func, 'attr', None))
            keywords = {k.arg: k.value for k in node.keywords if k.arg}
            if name in ERRORS or 'code' in keywords:
                yield path.relative_to(root), node.lineno, name, keywords.get('code')


def test_every_code_written_in_the_source_has_the_grammar():
    bad, uncoded = [], []
    for path, line, name, code in _sites():
        if code is None:
            if name in ERRORS:
                uncoded.append(f'{path}:{line}')
            continue
        bad += [f'{path}:{line} {value!r}' for value in _rendered(code) if not conforms(value)]
    assert not bad, bad
    # Every construction names its code; the class defaults are a safety net, not a vocabulary.
    assert not uncoded, uncoded


def test_each_failure_class_defaults_to_a_dotted_code():
    assert ValidationError('x').code == 'input.invalid'
    assert UnsupportedFeatureError('x').code == 'feature.unsupported'
    assert IncompatibleProblemError('x').code == 'problem.incompatible'
    assert NumericalError('x').code == 'numerical.failure'
    assert WavebitError('x').code == 'input.invalid'
    error = ValidationError('x', code='numerics.orders', path='numerics.orders')
    assert error.issue == Issue(code='numerics.orders', path='numerics.orders', message='x')


# -- Refusals a public caller meets, in each solver ------------------------------------------------

def _rcwa_problem(layers=1):
    return rcwa.Problem(units=Units(length='um'), stack=rcwa.Stack(period=(.8, .9), layers=(
        rcwa.Layer(thickness=.2, background=Material(2.25)),)*layers),
        excitation=rcwa.Excitation(wavelength=.6, sources=(rcwa.PlaneWave(),)))


def _fdtd_problem(z=.5):
    return fdtd.Problem(units=Units(length='um'), scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.1, .1, 1.6)),
        boundaries=fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.Open()),
        structures=(fdtd.Structure('film', shape=fdtd.SnappedSlab(z=(-.4, -.2)),
                                   material=Material(epsilon=2.25)),)),
        sources=(fdtd.PlaneWave('pw', plane=fdtd.Plane(z=z), pulse=fdtd.GaussianPulse(
            wavelength=.8, frequency_sigma_hz=4e13)),),
        outputs=fdtd.Outputs(monitors=(fdtd.TimeSeriesMonitor('e', at=(0, 0, 0), component='Ey'),)))


def _fdtd_numerics(**changes):
    return fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.025), duration=fdtd.Duration(steps=20),
                         absorber=fdtd.PML(cells=12), **changes)


def _native_case(dt=.02e-6/C0):
    shape = (16, 2, 2)
    current = np.zeros((3,)+shape)
    current[0, 4] = 1e7
    q = native.Problem(units=Units(length='um'), medium=native.Medium(epsilon=np.ones(shape)*2.),
        current=native.Current(density=current, samples=np.sin(np.arange(24)*.1)))
    n = native.Numerics(grid=native.Grid(spacing=(.05,)*3), time_step_seconds=dt, steps=24)
    return q, n


def _periodic_case(orders=(0, 0), wavelength=(.85, .9, .95)):
    q = periodic.Problem(units=Units(length='um'),
        stack=periodic.Stack(period=(.2,), top=Material(1.), bottom=Material(1.),
            layers=(periodic.Layer(thickness=.1, epsilon=np.full((1, 2), 2.25)),)),
        excitation=periodic.Excitation(wavelength=wavelength, sources=(periodic.PlaneWave(),)))
    n = periodic.Numerics(spacing=1/30, duration_seconds=55*.9e-6/C0, orders=orders)
    return q, n


def _code(action):
    """The code ``action`` refuses with: raised, or the first error of a returned report."""
    try:
        out = action()
    except WavebitError as error:
        assert error.issue.code == error.code
        return error.code
    errors = [i for i in out.issues if i.severity == 'error']
    assert errors, 'nothing was refused'
    return errors[0].code


REFUSALS = {
    'units.length': lambda: Units(length='km'),
    'units.type': lambda: fdtd.Problem(scene=None, units='nm'),
    'execution.backend': lambda: Execution(backend='JAX').validate(),
    'execution.precision': lambda: Execution(precision='quad').validate(),
    'material.scalar': lambda: Drude(omega_p='x', gamma=1.),
    'rcwa.numerics.orders': lambda: rcwa.Numerics(orders=2),
    'rcwa.numerics.choice': lambda: rcwa.Numerics(algorithm='ETM'),
    'rcwa.problem.type': lambda: rcwa.check(None, numerics=rcwa.Numerics()),
    'rcwa.stack.period': lambda: rcwa.check(rcwa.Problem(
        units=Units(length='um'), stack=rcwa.Stack(period=(-.8, .9), layers=()),
        excitation=rcwa.Excitation(wavelength=.6, sources=(rcwa.PlaneWave(),))),
        numerics=rcwa.Numerics()),
    'experimental.orders': lambda: experimental.ConformingNVF(3),
    'fdtd.numerics.choice': lambda: fdtd.check(_fdtd_problem(),
                                               numerics=_fdtd_numerics(discretization='bogus')),
    'fdtd.numerics.duration': lambda: fdtd.Duration(),
    'fdtd.boundaries.face': lambda: fdtd.Boundaries(x='open', y=fdtd.Periodic(), z=fdtd.Open()),
    'fdtd.monitor.orders': lambda: fdtd.DiffractionMonitor(
        't', plane=fdtd.Plane(z=-.3), wavelengths=(.8,), source='pw', wave='transmitted', orders=1),
    'fdtd.execution.device': lambda: fdtd.check(_fdtd_problem(), numerics=_fdtd_numerics(),
                                                device='gpu'),
    'native.medium.poles': lambda: native.Medium(epsilon=1., poles=(1.,)),
    'native.numerics.type': lambda: native.Numerics(grid=None, time_step_seconds=1e-17, steps=1),
    'native.native.invalid': lambda: native.check(*_native_case(dt=100*.02e-6/C0)[:1],
                                                  numerics=_native_case(dt=100*.02e-6/C0)[1]),
    'periodic.numerics.choice': lambda: periodic.Numerics(spacing=.1, duration_seconds=1e-15,
                                                          algorithm='bogus'),
    'periodic.numerics.orders': lambda: periodic.Numerics(spacing=.1, duration_seconds=1e-15,
                                                          orders=0),
    'periodic.numerics.spacing': lambda: periodic.Numerics(spacing=-.1, duration_seconds=1e-15),
    'periodic.periodic.invalid': lambda: periodic.check(*_periodic_case(orders=(1, 1))[:1],
                                                        numerics=_periodic_case(orders=(1, 1))[1]),
}


@pytest.mark.parametrize('expected', list(REFUSALS))
def test_a_public_refusal_carries_its_dotted_code(expected):
    code = _code(REFUSALS[expected])
    assert conforms(code) and code != 'invalid_input', code
    # The key names the solver first where the code alone would not say which one refused.
    assert expected.endswith(code), (expected, code)


# -- prepared.structure, naming what changed -------------------------------------------------------

def test_rcwa_prepared_structure_names_the_changed_declaration():
    prepared = rcwa.prepare(_rcwa_problem(), numerics=rcwa.Numerics())
    changed = _rcwa_problem(layers=2)
    (issue,) = prepared.validate(changed).issues
    assert (issue.code, issue.path) == ('prepared.structure', 'stack.layers')
    with pytest.raises(IncompatibleProblemError) as caught:
        prepared.run(changed)
    assert (caught.value.code, caught.value.path) == ('prepared.structure', 'stack.layers')


@pytest.mark.parametrize('change, path', [
    (lambda p: p.replace(scene=p.scene.replace(domain=fdtd.Domain(size=(.1, .1, 1.7)))),
     'scene.domain'),
    (lambda p: p.replace(sources=(p.sources[0].replace(plane=fdtd.Plane(z=.45)),)), 'sources.pw'),
    (lambda p: p.replace(units=Units(length='nm')), 'units.length'),
])
def test_fdtd_prepared_structure_names_the_changed_declaration(change, path):
    prepared = fdtd.prepare(_fdtd_problem(), numerics=_fdtd_numerics())
    changed = change(_fdtd_problem())
    (issue,) = prepared.validate(changed).issues
    assert (issue.code, issue.path) == ('prepared.structure', path)
    with pytest.raises(IncompatibleProblemError) as caught:
        prepared.run(changed)
    assert (caught.value.code, caught.value.path) == ('prepared.structure', path)


def test_native_prepared_structure_names_the_changed_declaration():
    q, n = _native_case()
    prepared = native.prepare(q, numerics=n)
    changed = q.replace(medium=native.Medium(epsilon=np.ones((16, 2, 3))*2.))
    (issue,) = prepared.validate(changed).issues
    assert (issue.code, issue.path) == ('prepared.structure', 'medium.epsilon')
    with pytest.raises(IncompatibleProblemError) as caught:
        prepared.run(changed)
    assert (caught.value.code, caught.value.path) == ('prepared.structure', 'medium.epsilon')


@pytest.mark.parametrize('change, path', [
    (lambda q: q.replace(excitation=q.excitation.replace(wavelength=(.85, .9))),
     'excitation.wavelength'),
    (lambda q: q.replace(stack=q.stack.replace(layers=q.stack.layers*2)), 'stack.layers'),
])
def test_periodic_prepared_structure_names_the_changed_declaration(change, path):
    q, n = _periodic_case()
    prepared = periodic.prepare(q, numerics=n)
    changed = change(q)
    (issue,) = prepared.validate(changed).issues
    assert (issue.code, issue.path) == ('prepared.structure', path)
    with pytest.raises(IncompatibleProblemError) as caught:
        prepared.run(changed)
    assert (caught.value.code, caught.value.path) == ('prepared.structure', path)


# -- An engine refusal reaches prepare and run as the check's coded issue --------------------------

@pytest.mark.parametrize('module, case, code', [
    (native, lambda: _native_case(dt=100*.02e-6/C0), 'native.invalid'),
    (periodic, lambda: _periodic_case(orders=(1, 1)), 'periodic.invalid'),
])
def test_an_engine_refusal_is_the_same_validation_error_at_every_door(module, case, code):
    q, n = case()
    (issue,) = module.check(q, numerics=n).issues
    assert issue.code == code
    for door in (module.prepare, module.run):
        with pytest.raises(ValidationError) as caught:
            door(q, numerics=n)
        assert (caught.value.code, caught.value.path) == (code, issue.path)
        assert caught.value.message == issue.message
        assert type(caught.value.__cause__) is ValueError


@pytest.mark.parametrize('raised, kind, code', [
    (ValueError('an engine refusal'), ValidationError, 'fdtd.invalid'),
    (TypeError('an engine refusal'), ValidationError, 'fdtd.invalid'),
    (NotImplementedError('an engine refusal'), UnsupportedFeatureError, 'fdtd.unsupported'),
])
def test_an_fdtd_execution_refusal_carries_the_catch_all_code(monkeypatch, raised, kind, code):
    import wavebit.fdtd._api as api
    prepared = fdtd.prepare(_fdtd_problem(), numerics=_fdtd_numerics())

    def execute(*args, **kwargs):
        raise raised
    monkeypatch.setattr(api, 'execute', execute)
    with pytest.raises(kind) as caught:
        prepared.run(_fdtd_problem())
    assert (caught.value.code, caught.value.path, caught.value.message) == (
        code, 'problem', 'an engine refusal')
    assert caught.value.__cause__ is raised


def test_an_fdtd_runtime_error_is_not_reworded_as_a_refusal(monkeypatch):
    import wavebit.fdtd._api as api
    prepared = fdtd.prepare(_fdtd_problem(), numerics=_fdtd_numerics())

    def execute(*args, **kwargs):
        raise RuntimeError('oom')
    monkeypatch.setattr(api, 'execute', execute)
    with pytest.raises(RuntimeError, match='oom'):
        prepared.run(_fdtd_problem())


# -- orders: an explicit (Mu, Mv) pair everywhere --------------------------------------------------

ORDERS = {
    'rcwa.Numerics': (lambda v: rcwa.Numerics(orders=v).orders, 'numerics.orders',
                      'numerics.orders'),
    'fdtd.periodic.Numerics': (lambda v: periodic.Numerics(
        spacing=.1, duration_seconds=1e-15, orders=v).orders, 'numerics.orders', 'numerics.orders'),
    'fdtd.DiffractionMonitor': (lambda v: fdtd.DiffractionMonitor(
        't', plane=fdtd.Plane(z=-.3), wavelengths=(.8,), source='pw', wave='transmitted',
        orders=v).orders, 'monitor.orders', 'outputs.t.orders'),
    'rcwa.experimental': (lambda v: experimental.CartesianRectangleNVF(v).orders,
                          'experimental.orders', 'orders'),
}


@pytest.mark.parametrize('where', list(ORDERS))
@pytest.mark.parametrize('value', [2, 0, np.int64(2), True, (True, 1), (1, np.bool_(True)), (-1, 0),
                                   (2,), (1, 2, 3), (2., 2), {'u': 2, 'v': 1}, {1, 3}, range(2, 4),
                                   np.array([2, 2]), '22'],
                         ids=repr)
def test_orders_refuses_anything_but_an_explicit_pair(where, value):
    make, code, path = ORDERS[where]
    with pytest.raises(ValidationError) as caught:
        make(value)
    assert (caught.value.code, caught.value.path) == (code, path)
    assert '(orders=...) must be an explicit pair (Mu, Mv)' in caught.value.message


@pytest.mark.parametrize('where', list(ORDERS))
def test_orders_takes_a_pair_of_nonnegative_integers_numpy_ones_included(where):
    make = ORDERS[where][0]
    for value in ((2, 1), [2, 1], (np.int64(2), np.int32(1)), (np.uint8(2), 1)):
        orders = make(value)
        assert orders == (2, 1) and all(type(m) is int for m in orders), (value, orders)


def test_a_bare_order_says_which_pair_to_write():
    with pytest.raises(ValidationError, match=r'write \(3, 3\) for the same half-width along both '
                                              r'lattice vectors, or \(3, 0\) for the first only'):
        fdtd.DiffractionMonitor('t', plane=fdtd.Plane(z=-.3), wavelengths=(.8,), source='pw',
                                wave='transmitted', orders=3)


def test_the_experimental_builders_default_to_pairs():
    assert experimental.ConformingNVF().orders == (3, 3)
    assert experimental.AffineProfileNVF().orders == (3, 3)
    assert experimental.CartesianRectangleNVF().orders == (1, 1)
    assert experimental.CartesianRectanglePairNVF().orders == (1, 1)


def test_the_transmission_gap_cell_reports_its_window_as_orders():
    # The engine cell names the window ``order`` (its own tests read that); the public closure
    # returns it as ``orders``, in place. A stand-in engine keeps this test from solving one.
    class Engine:
        order = (1, 2)

        def closure(self, **given):
            return 'value', 'tangent', {'closure': given['gap'], 'order': self.order,
                                        'angular_order': 0, 'transmission_cutoff_order': 0}

    cell = object.__new__(experimental.CartesianPairTransmissionGalerkinGapCell)
    cell._cell = Engine()
    value, tangent, metadata = cell.closure(gap=0., basis=np.eye(2), widths=(.3, .4), height=.45,
                                            epsilon=(3., 3.))
    assert (value, tangent, cell.orders) == ('value', 'tangent', (1, 2))
    assert list(metadata) == ['closure', 'orders', 'angular_order', 'transmission_cutoff_order']
    assert metadata['orders'] == (1, 2)
