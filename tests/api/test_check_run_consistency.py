"""``check`` refuses what ``run`` would refuse, with a structured code and the field's path.

Each case here once passed ``check`` (and usually ``prepare``) and then failed at ``run``, or
failed with a message that named no field, a private engine name, or advice that led to
another refusal.
"""
import numpy as np
import pytest

from wavebit import Material, Units, fdtd, rcwa
from wavebit.errors import IncompatibleProblemError, UnsupportedFeatureError, ValidationError
from wavebit.fdtd import native, periodic
from wavebit.units import C0


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



# -- 2. A raster of the wrong shape is refused with its expected and actual shape -------------

def _raster(epsilon, mu=None):
    return _rcwa(layer=rcwa.Layer(thickness=.2, epsilon=epsilon, mu=mu))


@pytest.mark.parametrize('epsilon', [np.full(8, 2.), np.full((4, 5, 2), 2.), np.full((8, 8, 3), 2.),
                                     np.full((8, 8, 2, 2), 2.), np.full((0, 8), 2.), 2.],
                         ids=['1d', '3d', 'diagonal', 'tensor-2x2', 'empty', 'scalar'])
def test_a_wrongly_shaped_raster_is_refused_with_its_shape(epsilon):
    issue = _refused_alike(_raster(epsilon))
    assert (issue.code, issue.path) == ('layer.raster_shape', 'stack.layers[0].epsilon')
    assert issue.actual == np.shape(epsilon)
    assert issue.expected == '(Ny, Nx) or (Ny, Nx, 3, 3)'
    assert f'has shape {np.shape(epsilon)}' in issue.message


def test_a_wrongly_shaped_permeability_raster_is_refused_at_its_own_path():
    issue = _refused_alike(_raster(np.full((8, 8), 2.), mu=np.ones((8, 8, 3))))
    assert (issue.code, issue.path, issue.actual) == ('layer.raster_shape', 'stack.layers[0].mu', (8, 8, 3))


# -- 3. A complex or string shape size is refused when the shape is written -------------------

@pytest.mark.parametrize('build, path', [
    (lambda v: rcwa.Rectangle(size=(v, .3)), 'geometry.size'),
    (lambda v: rcwa.Rectangle(size=(.3, .3), center=(v, 0.)), 'geometry.center'),
    (lambda v: rcwa.Rectangle(size=(.3, .3), angle=v), 'geometry.angle'),
    (lambda v: rcwa.Circle(radius=v), 'geometry.radius'),
    (lambda v: rcwa.Ellipse(radii=(.2, v)), 'geometry.radii'),
], ids=['rectangle-size', 'rectangle-center', 'rectangle-angle', 'circle-radius', 'ellipse-radii'])
@pytest.mark.parametrize('value', [.3j, np.complex128(.3), '.3', np.asarray(.3 + 0j)],
                         ids=['complex', 'numpy-complex', 'string', 'complex-array'])
def test_a_complex_or_string_shape_entry_is_refused_at_construction(build, path, value):
    with pytest.raises(ValidationError) as caught:
        build(value)
    assert (caught.value.code, caught.value.path) == ('shape.value', path)
    assert 'is a real number' in caught.value.message
    assert 'tracer' not in caught.value.message


# -- 4. An optional dependency the numerics need is reported by check -------------------------

def test_contour_normals_without_contourpy_are_refused_at_check(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'contourpy', None)      # import contourpy -> ImportError
    numerics = rcwa.Numerics(orders=(2, 2), factorization='nvf', raster_normal_method='contour')
    problem = _raster(np.where(np.eye(16) > 0, 4., 2.25))
    issue = _refused_alike(problem, numerics=numerics)
    assert (issue.code, issue.path) == ('numerics.unavailable', 'numerics.raster_normal_method')
    assert 'contourpy' in issue.message and "raster_normal_method='spectral'" in issue.message
    # The advice: the spectral construction needs no optional dependency.
    assert rcwa.check(problem, numerics=numerics.replace(raster_normal_method='spectral')).ok


# -- 5. A tensor PyTorch differentiates, given to a NumPy or JAX run, is a check issue ---------

@pytest.mark.parametrize('backend', ['numpy', 'jax'])
@pytest.mark.parametrize('where', ['theta', 'epsilon'])
def test_a_differentiated_torch_tensor_on_another_backend_is_a_check_issue(backend, where):
    torch = pytest.importorskip('torch')
    jax = pytest.importorskip('jax')
    leaf = torch.tensor(.2 if where == 'theta' else 4., dtype=torch.float64, requires_grad=True)
    if where == 'theta':
        problem, path = _rcwa(theta=leaf), 'excitation.sources[0].theta'
    else:
        problem, path = _region_material(leaf), 'stack.layers[0].regions[0].material.epsilon'
    with jax.enable_x64(True):
        report = rcwa.check(problem, numerics=NUMERICS, backend=backend)    # a report, never a raise
        (issue,) = report.issues
        assert (issue.code, issue.path) == ('execution.backend_autodiff', path)
        for door in (rcwa.prepare, rcwa.run):
            with pytest.raises(ValidationError) as caught:
                door(problem, numerics=NUMERICS, backend=backend)
            assert (caught.value.code, caught.value.path) == (issue.code, issue.path)
        # The advice: a detached tensor runs on this backend, and the tensor itself runs on torch.
        detached = _rcwa(theta=leaf.detach()) if where == 'theta' else _region_material(leaf.detach())
        assert rcwa.check(detached, numerics=NUMERICS, backend=backend).ok
    assert rcwa.check(problem, numerics=NUMERICS, backend='torch').ok


# -- 7. A refusal is one exception class whichever path raises it -----------------------------

def _fdtd_case():
    problem = fdtd.Problem(units=Units(length='um'), scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.1, .1, 1.6)),
        boundaries=fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.Open())),
        sources=(fdtd.PlaneWave('pw', plane=fdtd.Plane(z=.5), pulse=fdtd.GaussianPulse(
            wavelength=.8, frequency_sigma_hz=4e13)),),
        outputs=fdtd.Outputs(monitors=(fdtd.TimeSeriesMonitor('e', at=(0, 0, 0), component='Ey'),)))
    numerics = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.05), duration=fdtd.Duration(steps=4),
                             absorber=fdtd.PML(cells=12))
    changed = problem.replace(scene=problem.scene.replace(domain=fdtd.Domain(size=(.1, .1, 1.7))))
    return fdtd.prepare(problem, numerics=numerics), changed


def _periodic_case():
    q = periodic.Problem(units=Units(length='um'),
        stack=periodic.Stack(period=(.2,), top=Material(1.), bottom=Material(1.),
            layers=(periodic.Layer(thickness=.1, epsilon=np.full((1, 2), 2.25)),)),
        excitation=periodic.Excitation(wavelength=(.85, .9, .95), sources=(periodic.PlaneWave(),)))
    n = periodic.Numerics(spacing=1/30, duration_seconds=55*.9e-6/C0, orders=(0, 0))
    return periodic.prepare(q, numerics=n), q.replace(stack=q.stack.replace(period=(.3,)))


def _native_case():
    shape = (16, 2, 2)
    current = np.zeros((3,)+shape)
    current[0, 4] = 1e7
    q = native.Problem(units=Units(length='um'), medium=native.Medium(epsilon=np.ones(shape)*2.),
        current=native.Current(density=current, samples=np.sin(np.arange(24)*.1)))
    n = native.Numerics(grid=native.Grid(spacing=(.05,)*3), time_step_seconds=.02e-6/C0, steps=24)
    return (native.prepare(q, numerics=n),
            q.replace(medium=native.Medium(epsilon=np.ones((16, 2, 3))*2.)))


def _rcwa_case():
    problem = _rcwa()
    layers = problem.stack.layers*2
    return (rcwa.prepare(problem, numerics=NUMERICS),
            problem.replace(stack=problem.stack.replace(layers=layers)))


@pytest.mark.parametrize('case', [_rcwa_case, _fdtd_case, _periodic_case, _native_case],
                         ids=['rcwa', 'fdtd', 'periodic', 'native'])
def test_a_changed_prepared_structure_raises_one_class_on_validate_and_run(case):
    prepared, changed = case()
    with pytest.raises(ValidationError) as validated:
        prepared.validate(changed).raise_if_invalid()
    with pytest.raises(ValidationError) as ran:
        prepared.run(changed)
    assert type(validated.value) is type(ran.value) is IncompatibleProblemError
    assert (validated.value.code, validated.value.path) == (ran.value.code, ran.value.path)
    assert validated.value.code == 'prepared.structure'


def test_a_check_issue_is_raised_as_the_class_prepare_raises():
    problem, numerics = _rcwa(), rcwa.Numerics(orders=(2, 2), factorization='li', raster_normal_method='spectral')
    with pytest.raises(ValidationError) as checked:
        rcwa.check(problem, numerics=numerics).raise_if_invalid()
    with pytest.raises(ValidationError) as prepared:
        rcwa.prepare(problem, numerics=numerics)
    assert type(checked.value) is type(prepared.value) is UnsupportedFeatureError
    # An explicit class still names one class for every issue, as before.
    with pytest.raises(ValidationError) as named:
        rcwa.check(problem, numerics=numerics).raise_if_invalid(ValidationError)
    assert type(named.value) is ValidationError


# -- 6. A ConductiveAbsorber refusal speaks public names, and its advice runs -----------------

_PULSE = fdtd.GaussianPulse(wavelength=2., frequency_sigma_hz=1e13)


def _absorbed(*, clear=1., spacing=.1, material=None, source=None, xopen=False, initial=None):
    """A cylinder whose ends stand ``clear`` grid cells inside the two Open z faces."""
    top = .8 - clear*spacing
    problem = fdtd.Problem(units=Units(length='um'), scene=fdtd.Scene(
        domain=fdtd.Domain(size=(1., 1., 1.6)),
        boundaries=fdtd.Boundaries(x=fdtd.Open() if xopen else fdtd.PEC(), y=fdtd.PEC(), z=fdtd.Open()),
        structures=(fdtd.Structure('c', shape=fdtd.Cylinder(radius=.22, height=2*top),
                                   material=material or Material(2.25)),)),
        sources=(source or fdtd.CurrentDipole('d', center=(0., 0., 0.), component='Ey', pulse=_PULSE),),
        outputs=fdtd.Outputs(monitors=(fdtd.TimeSeriesMonitor('e', at=(0, 0, .1), component='Ey'),)),
        initial=initial)
    numerics = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=spacing), duration=fdtd.Duration(steps=6),
                             absorber=fdtd.ConductiveAbsorber(cells=3))
    return problem, numerics


ABSORBER_CASES = {
    'magnetic-material': (dict(material=Material(2.25, mu=1.1)), 'no mu (mu = 1)'),
    'magnetic-source': (dict(source=fdtd.CurrentDipole('m', center=(0., 0., 0.), component='Hx',
                                                       kind='magnetic', pulse=_PULSE)),
                        'every source must be an electric fdtd.CurrentDipole'),
    'two-axes': (dict(xopen=True), 'the Open faces must all lie on one axis'),
    'stopping': ('stopping', 'Remove stopping and run the fixed Numerics(duration=...)'),
}


def _absorber_case(name, **where):
    changes, says = ABSORBER_CASES[name]
    if changes == 'stopping':
        problem, numerics = _absorbed(**where)
        return problem, numerics.replace(stopping=fdtd.DecayStop(threshold=1e-5, window_seconds=1e-15)), says
    problem, numerics = _absorbed(**changes, **where)
    return problem, numerics, says


def _public_absorber_words(message):
    assert 'ConstitutiveAbsorber' not in message and 'first' not in message and 'new term' not in message
    assert 'ConductiveAbsorber' in message


@pytest.mark.parametrize('spacing', [.1, .05, .04])
@pytest.mark.parametrize('name', sorted(ABSORBER_CASES))
def test_a_conductive_absorber_refusal_names_public_declarations_and_advice_that_runs(name, spacing):
    problem, numerics, says = _absorber_case(name, spacing=spacing)
    (issue,) = fdtd.check(problem, numerics=numerics).issues
    _public_absorber_words(issue.message)
    assert says in issue.message
    assert 'at least three grid cells, 3*UniformGrid(spacing), clear of every Open face' in issue.message
    with pytest.raises(ValidationError) as caught:
        fdtd.run(problem, numerics=numerics)
    assert caught.value.message == issue.message
    # The advice: the same structure three grid cells clear of the Open faces passes.
    problem, numerics, _ = _absorber_case(name, clear=3., spacing=spacing)
    assert fdtd.check(problem, numerics=numerics).ok


def test_the_other_absorber_advice_runs():
    # Removing stopping, as advised, runs the same structure where it stands.
    problem, numerics, _ = _absorber_case('stopping')
    assert fdtd.check(problem, numerics=numerics.replace(stopping=None)).ok
    # An InitialField is refused there; a source in its place, as advised, runs.
    problem, numerics = _absorbed(initial=fdtd.InitialField(E=(None, lambda x, y, z: x*0 + 1, None)))
    (issue,) = fdtd.check(problem, numerics=numerics).issues
    _public_absorber_words(issue.message)
    assert 'Excite the field with a source instead of an fdtd.InitialField' in issue.message
    assert fdtd.check(problem.replace(initial=None), numerics=numerics).ok


def test_a_differentiated_material_beside_the_absorber_runs_once_moved_clear():
    torch = pytest.importorskip('torch')
    leaf = torch.tensor(2.25, dtype=torch.float64, requires_grad=True)
    problem, numerics = _absorbed()
    prepared = fdtd.prepare(problem, numerics=numerics, backend='torch')
    with pytest.raises(ValidationError) as caught:
        prepared.run(_absorbed(material=Material(leaf))[0])
    _public_absorber_words(caught.value.message)
    assert 'To differentiate them, move the smoothed structures' in caught.value.message
    assert 'run the traced problem with Prepared.run' in caught.value.message
    problem, numerics = _absorbed(clear=3.)
    prepared = fdtd.prepare(problem, numerics=numerics, backend='torch')
    prepared.run(_absorbed(material=Material(leaf), clear=3.)[0])['e'].values.real.sum().backward()
    assert leaf.grad is not None and bool(torch.isfinite(leaf.grad))


def test_every_absorber_coupling_refusal_the_engine_holds_has_public_words():
    import ast
    import pathlib
    import re
    from wavebit.fdtd._spelling import to_public
    root = pathlib.Path(fdtd.__file__).parent
    said = 0
    for path in (root/'_prepare'/'constitutive_absorber.py', root/'_kernel'/'constitutive_absorber.py'):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        inside = {id(v) for node in ast.walk(tree) if isinstance(node, ast.JoinedStr) for v in node.values}
        for node in ast.walk(tree):
            if id(node) in inside:
                continue
            if isinstance(node, ast.JoinedStr):
                text = ''.join(v.value if isinstance(v, ast.Constant) else '{' + ast.unparse(v.value) + '}'
                               for v in node.values)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                text = node.value
            else:
                continue
            if not text.startswith('ConstitutiveAbsorber'):
                continue
            sample = re.sub(r'\{[^}]*\}', 'x', text.replace('{node}', '(1, 2, 3)'))
            out = to_public(sample)
            said += 1
            assert 'ConstitutiveAbsorber' not in out, (path.name, text)
            for name in re.findall(r'(?<![\w.])fdtd\.(\w+)', out):
                assert hasattr(fdtd, name), name
    assert said >= 15, said


# -- 8. A differentiated CurrentDipole centre is refused with advice the public API can take ---

def test_a_traced_dipole_centre_names_what_differentiates_instead():
    torch = pytest.importorskip('torch')

    def problem(**dipole):
        return fdtd.Problem(units=Units(length='um'), scene=fdtd.Scene(
            domain=fdtd.Domain(size=(1., 1., 1.)),
            boundaries=fdtd.Boundaries(x=fdtd.PEC(), y=fdtd.PEC(), z=fdtd.Open())),
            sources=(fdtd.CurrentDipole('d', component='Ey', pulse=_PULSE, **dipole),),
            outputs=fdtd.Outputs(monitors=(fdtd.TimeSeriesMonitor('e', at=(0, 0, .2), component='Ey'),)))
    numerics = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.1), duration=fdtd.Duration(steps=10))
    x = torch.tensor(.01, dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValidationError) as caught:
        fdtd.run(problem(center=(x, 0., 0.)), numerics=numerics, backend='torch')
    message = caught.value.message
    assert 'range=' not in message                   # public CurrentDipole has no range field
    assert 'CurrentDipole(center=...)' in message and 'current_moment' in message
    import inspect
    assert 'range' not in inspect.signature(fdtd.CurrentDipole).parameters
    # The advice: a plain centre, with the source's own terms differentiated.
    moment = torch.tensor(1e-9, dtype=torch.float64, requires_grad=True)
    fdtd.run(problem(center=(.01, 0., 0.), current_moment=moment), numerics=numerics,
             backend='torch')['e'].values.real.sum().backward()
    assert moment.grad is not None and float(moment.grad) != 0.


# -- 9. periodic Result.sample names the public field component ------------------------------

def test_periodic_sample_outside_its_record_names_the_public_component():
    q = periodic.Problem(units=Units(length='um'),
        stack=periodic.Stack(period=(.2,), top=Material(1.), bottom=Material(1.),
            layers=(periodic.Layer(thickness=.1, epsilon=np.full((1, 2), 2.25)),)),
        excitation=periodic.Excitation(wavelength=(.85, .9, .95), sources=(periodic.PlaneWave(),)),
        outputs=periodic.Outputs(snapshots=periodic.Snapshots(every=17)))
    n = periodic.Numerics(spacing=1/30, duration_seconds=55*.9e-6/C0, orders=(0, 0))
    result = periodic.run(q, numerics=n)
    with pytest.raises(ValidationError) as caught:
        result.sample(z=50.)
    message = caught.value.message
    assert 'ex.' not in message and "'Ex'" in message
    assert 'Result.sample(z=50.0) is outside the interval' in message
    assert result.sample(z=.05).components == ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')
