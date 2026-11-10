"""``check`` refuses what ``run`` would refuse, with a structured code and the field's path.

Each case here once passed ``check`` (and usually ``prepare``) and then failed at ``run``, or
failed with a message that named no field, a private engine name, or advice that led to
another refusal.
"""
import dataclasses

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


@pytest.mark.parametrize('epsilon, mu', [(np.full((8, 8), 2.), np.ones((8, 7))),
                                         (np.full((1, 8), 2.), np.ones((8, 8))),
                                         (np.full((8, 8, 3, 3), 2.), np.ones((4, 8, 3, 3)))],
                         ids=['columns', 'rows', 'tensor'])
def test_a_permeability_raster_on_another_pixel_grid_is_refused_at_its_path(epsilon, mu):
    issue = _refused_alike(_raster(epsilon, mu=mu))
    assert (issue.code, issue.path, issue.actual) == ('layer.raster_shape', 'stack.layers[0].mu', mu.shape)
    pixels = epsilon.shape[:2]
    assert issue.expected == f'{pixels} or {pixels + (3, 3)}'
    assert f'sampled on {pixels[0]} x {pixels[1]} pixels' in issue.message
    assert 'structure is' not in issue.message
    # The advice: the same pixel grid runs, with a scalar or a tensor per pixel either side.
    for same in (np.ones(pixels), np.ones(pixels + (3, 3))):
        assert rcwa.check(_raster(epsilon, mu=same), numerics=NUMERICS).ok


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


def _absorber_refused_alike(problem, numerics, code, path):
    """check's issue, and what prepare and run raise: one class, ``code`` at ``path``."""
    (issue,) = fdtd.check(problem, numerics=numerics).issues
    assert (issue.code, issue.path) == (code, path)
    raised = []
    for door in (fdtd.prepare, fdtd.run):
        with pytest.raises(ValidationError) as caught:
            door(problem, numerics=numerics)
        raised.append(caught.value)
    assert type(raised[0]) is type(raised[1]) is ValidationError
    for exc in raised:
        assert (exc.code, exc.path, exc.message) == (code, path, issue.message)
    return issue


@pytest.mark.parametrize('name', sorted(ABSORBER_CASES) + ['initial'])
def test_a_conductive_absorber_coupling_refusal_is_numerics_absorber_at_every_door(name):
    # The owner's decision of 2026-11-10: the absorber's own code, at the absorber, not the
    # catch-all fdtd.invalid at problem. What is refused, and its words, are unchanged.
    if name == 'initial':
        problem, numerics = _absorbed(initial=fdtd.InitialField(E=(None, lambda x, y, z: x*0 + 1, None)))
    else:
        problem, numerics, _ = _absorber_case(name)
    issue = _absorber_refused_alike(problem, numerics, 'numerics.absorber', 'numerics.absorber')
    _public_absorber_words(issue.message)


def test_a_traced_material_beside_the_absorber_is_numerics_absorber_on_validate_and_run():
    torch = pytest.importorskip('torch')
    leaf = torch.tensor(2.25, dtype=torch.float64, requires_grad=True)
    problem, numerics = _absorbed()
    prepared = fdtd.prepare(problem, numerics=numerics, backend='torch')
    traced = _absorbed(material=Material(leaf))[0]
    (issue,) = prepared.validate(traced).issues
    with pytest.raises(ValidationError) as ran:
        prepared.run(traced)
    assert type(ran.value) is ValidationError
    assert ((issue.code, issue.path) == (ran.value.code, ran.value.path)
            == ('numerics.absorber', 'numerics.absorber'))
    _public_absorber_words(issue.message)


@pytest.mark.parametrize('cells, says', [(1, 'two is the minimum that grades at all'),
                                         (2.5, 'an integer is needed'),
                                         (True, 'an integer is needed, not a bool')])
def test_a_conductive_absorber_cell_count_is_refused_at_its_field(cells, says):
    problem, numerics = _absorbed(clear=3.)
    numerics = numerics.replace(absorber=fdtd.ConductiveAbsorber(cells=cells))
    issue = _absorber_refused_alike(problem, numerics, 'numerics.absorber', 'numerics.absorber.cells')
    assert issue.message.startswith(f'ConductiveAbsorber(cells={cells!r}): ') and says in issue.message


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


# -- 10. FDTD and periodic name the field that holds a NaN or an infinity ---------------------

def _fdtd_nonfinite(where, value):
    problem, numerics = _absorbed()
    if where == 'structure-material':
        problem, numerics = _absorbed(material=Material(value))
    elif where == 'structure-mu':
        problem, numerics = _absorbed(material=Material(2.25, mu=value))
    elif where == 'background':
        problem = problem.replace(scene=problem.scene.replace(background=Material(value)))
    elif where == 'shape':
        (structure,) = problem.scene.structures
        problem = problem.replace(scene=problem.scene.replace(structures=(
            structure.replace(shape=fdtd.Cylinder(radius=value, height=1.4)),)))
    elif where == 'pulse':
        problem, numerics = _absorbed(source=fdtd.CurrentDipole(
            'd', center=(0., 0., 0.), component='Ey',
            pulse=fdtd.GaussianPulse(wavelength=value, frequency_sigma_hz=1e13)))
    elif where == 'moment':
        problem, numerics = _absorbed(source=fdtd.CurrentDipole(
            'd', center=(0., 0., 0.), component='Ey', pulse=_PULSE, current_moment=value))
    return problem, numerics


FDTD_NONFINITE = {'structure-material': 'scene.structures.c.material.epsilon',
                  'structure-mu': 'scene.structures.c.material.mu',
                  'background': 'scene.background.epsilon',
                  'shape': 'scene.structures.c.shape.radius',
                  'pulse': 'sources.d.pulse.wavelength',
                  'moment': 'sources.d.current_moment'}


@pytest.mark.parametrize('value', [np.nan, np.inf], ids=['nan', 'inf'])
@pytest.mark.parametrize('where', sorted(FDTD_NONFINITE))
def test_fdtd_names_the_field_holding_a_nonfinite_value(where, value):
    problem, numerics = _fdtd_nonfinite(where, value)
    (issue,) = fdtd.check(problem, numerics=numerics).issues
    assert (issue.code, issue.path) == ('problem.nonfinite', FDTD_NONFINITE[where])
    assert 'non-finite' in issue.message
    for door in (fdtd.prepare, fdtd.run):
        with pytest.raises(ValidationError) as caught:
            door(problem, numerics=numerics)
        assert (caught.value.code, caught.value.path) == (issue.code, issue.path)


def _placed(where, value):
    """A source or monitor coordinate set to ``value``, with the path its refusal names."""
    problem, numerics = _absorbed()
    if where == 'dipole':
        return (_absorbed(source=fdtd.CurrentDipole('d', center=(0., value, 0.), component='Ey',
                                                    pulse=_PULSE))[0], numerics,
                'source.outside_domain', 'sources.d.center')
    if where == 'monitor':
        return (problem.replace(outputs=fdtd.Outputs(monitors=(
                    fdtd.TimeSeriesMonitor('e', at=(0, 0, value), component='Ey'),))), numerics,
                'monitor.outside_domain', 'outputs.e')
    if where == 'gaussian-beam':
        source = fdtd.GaussianBeam('s', plane=fdtd.Plane(z=value), pulse=_PULSE, waist=.3)
    else:
        source = (fdtd.PlaneWave if where == 'plane-wave' else fdtd.ModeSource)(
            's', plane=fdtd.Plane(z=value), pulse=_PULSE)
    return problem.replace(sources=(source,)), numerics, 'source.outside_domain', 'sources.s.plane'


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf, 5.], ids=['nan', 'inf', '-inf', 'outside'])
@pytest.mark.parametrize('where', ['dipole', 'monitor', 'plane-wave', 'mode-source', 'gaussian-beam'])
def test_a_nonfinite_position_is_not_reported_as_outside_the_domain(where, value):
    problem, numerics, outside, path = _placed(where, value)
    code = outside if np.isfinite(value) else 'problem.nonfinite'
    (issue,) = fdtd.check(problem, numerics=numerics).issues
    assert (issue.code, issue.path) == (code, path)
    assert ('outside' in issue.message) == bool(np.isfinite(value))
    for door in (fdtd.prepare, fdtd.run):
        with pytest.raises(ValidationError) as caught:
            door(problem, numerics=numerics)
        assert (caught.value.code, caught.value.path) == (code, path)



def _plane_wave_angles(theta, phi):
    problem = fdtd.Problem(units=Units(length='um'), scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.1, .1, 1.6)),
        boundaries=fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.Open())),
        sources=(fdtd.PlaneWave('pw', plane=fdtd.Plane(z=.5), theta=theta, phi=phi, pulse=fdtd.GaussianPulse(
            wavelength=.8, frequency_sigma_hz=4e13)),),
        outputs=fdtd.Outputs(monitors=tuple(fdtd.TimeSeriesMonitor(c.lower(), at=(0, 0, 0), component=c)
                                            for c in ('Ex', 'Ey'))))
    return problem, fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.05), duration=fdtd.Duration(steps=200),
                                  absorber=fdtd.PML(cells=12))


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf], ids=['nan', 'inf', '-inf'])
@pytest.mark.parametrize('theta', [0., .2], ids=['normal', 'oblique'])
def test_a_nonfinite_plane_wave_phi_is_refused_even_at_normal_incidence(theta, value):
    # phi sets the polarization frame at theta = 0 too (0 drives Ey, pi/2 Ex), so it is read
    # there. A NaN passed check and ran to NaN monitors; an infinity was refused only because
    # math.cos raises on it.
    problem, numerics = _plane_wave_angles(theta, value)
    (issue,) = fdtd.check(problem, numerics=numerics).issues
    assert (issue.code, issue.path) == ('problem.nonfinite', 'sources.pw.phi')
    for door in (fdtd.prepare, fdtd.run):
        with pytest.raises(ValidationError) as caught:
            door(problem, numerics=numerics)
        assert (caught.value.code, caught.value.path) == (issue.code, issue.path)
    # A finite phi at normal incidence still turns the field: Ey at 0, Ex at pi/2.
    for phi, lit in ((0., 'ey'), (np.pi/2, 'ex')):
        result = fdtd.run(*_plane_wave_angles(0., phi)[:1], numerics=numerics)
        values = {c: np.asarray(result[c].values) for c in ('ex', 'ey')}
        assert all(np.isfinite(v).all() for v in values.values())
        assert np.abs(values[lit]).max() > 0 and np.abs(values['ey' if lit == 'ex' else 'ex']).max() == 0


@pytest.mark.parametrize('value', [np.nan, np.inf], ids=['nan', 'inf'])
def test_periodic_refuses_a_nonfinite_phi_at_normal_incidence(value):
    problem = _periodic_nonfinite('raster', 2.25).replace(excitation=periodic.Excitation(
        wavelength=(.85, .9, .95), sources=(periodic.PlaneWave(theta=0., phi=value),)))
    numerics = periodic.Numerics(spacing=1/30, duration_seconds=55*.9e-6/C0, orders=(0, 0))
    (issue,) = periodic.check(problem, numerics=numerics).issues
    assert (issue.code, issue.path) == ('problem.nonfinite', 'excitation.sources[0].phi')
    for door in (periodic.prepare, periodic.run):
        with pytest.raises(ValidationError) as caught:
            door(problem, numerics=numerics)
        assert (caught.value.code, caught.value.path) == (issue.code, issue.path)

def test_a_prepared_fdtd_run_names_the_nonfinite_live_material():
    problem, numerics = _absorbed()
    prepared = fdtd.prepare(problem, numerics=numerics)
    live = _absorbed(material=Material(np.nan))[0]
    (issue,) = prepared.validate(live).issues
    assert (issue.code, issue.path) == ('problem.nonfinite', 'scene.structures.c.material.epsilon')
    with pytest.raises(ValidationError) as caught:
        prepared.run(live)
    assert (caught.value.code, caught.value.path) == (issue.code, issue.path)


def _periodic_nonfinite(where, value):
    return periodic.Problem(units=Units(length='um'),
        stack=periodic.Stack(period=(value,) if where == 'period' else (.2,),
            top=Material(value) if where == 'top' else Material(1.), bottom=Material(1.),
            layers=(periodic.Layer(thickness=value if where == 'thickness' else .1,
                                   epsilon=np.where(np.arange(2) > 0, value, 2.25)[None]
                                   if where == 'raster' else np.full((1, 2), 2.25)),)),
        excitation=periodic.Excitation(wavelength=(.85, .9, .95), sources=(periodic.PlaneWave(),)))


PERIODIC_NONFINITE = {'raster': 'stack.layers[0].epsilon', 'top': 'stack.top.epsilon',
                      'thickness': 'stack.layers[0].thickness', 'period': 'stack.period[0]'}


@pytest.mark.parametrize('value', [np.nan, np.inf], ids=['nan', 'inf'])
@pytest.mark.parametrize('where', sorted(PERIODIC_NONFINITE))
def test_periodic_names_the_field_holding_a_nonfinite_value(where, value):
    problem = _periodic_nonfinite(where, value)
    numerics = periodic.Numerics(spacing=1/30, duration_seconds=55*.9e-6/C0, orders=(0, 0))
    (issue,) = periodic.check(problem, numerics=numerics).issues
    assert (issue.code, issue.path) == ('problem.nonfinite', PERIODIC_NONFINITE[where])
    for door in (periodic.prepare, periodic.run):
        with pytest.raises(ValidationError) as caught:
            door(problem, numerics=numerics)
        assert (caught.value.code, caught.value.path) == (issue.code, issue.path)


# -- 11. periodic Result.sample names its times as times_seconds, in seconds ------------------

def test_periodic_sample_times_are_refused_in_seconds_under_their_public_name():
    q = periodic.Problem(units=Units(length='um'),
        stack=periodic.Stack(period=(.2,), top=Material(1.), bottom=Material(1.),
            layers=(periodic.Layer(thickness=.1, epsilon=np.full((1, 2), 2.25)),)),
        excitation=periodic.Excitation(wavelength=(.85, .9, .95), sources=(periodic.PlaneWave(),)),
        outputs=periodic.Outputs(snapshots=periodic.Snapshots(every=17)))
    n = periodic.Numerics(spacing=1/30, duration_seconds=55*.9e-6/C0, orders=(0, 0))
    result = periodic.run(q, numerics=n)
    recorded = result.sample(z=.05).times_seconds
    with pytest.raises(ValidationError) as caught:
        result.sample(z=.05, times_seconds=[2*recorded[-1]])
    error = caught.value
    assert (error.code, error.path) == ('periodic.invalid', 'times_seconds')
    assert error.message.startswith('Result.sample(times_seconds=...) is outside the interval')
    assert 'common record interval' not in error.message and 't is' not in error.message
    low, high = error.expected
    # The interval is quoted in seconds: it holds every time the default sample returns.
    assert low <= recorded[0] < recorded[-1] <= high < 2*recorded[-1]
    assert f'[{low!r}, {high!r}] seconds' in error.message
    # The advice: inside it, or with times_seconds omitted, the sample runs.
    assert result.sample(z=.05, times_seconds=[(low + high)/2]).values.shape == (1, 6)
    for bad, says in (([1j*recorded[1]], 'must hold real times'),
                      ([recorded[2], recorded[1]], 'strictly increasing times in seconds')):
        with pytest.raises(ValidationError) as caught:
            result.sample(z=.05, times_seconds=bad)
        assert caught.value.path == 'times_seconds' and says in caught.value.message


# -- 12. A shape NVF grid below 4*max(orders)+1 is refused at check, not first at run ----------

_REGION = (rcwa.Region(geometry=rcwa.Rectangle(size=(.3, .4)), material=Material(4.)),)
_TENSOR = (rcwa.Region(geometry=rcwa.Rectangle(size=(.3, .4)), material=Material(np.diag([4., 3., 2.]))),)
_BEAM = dict(sources=(rcwa.GaussianBeam(waist=1., angular_spectrum_extent=3., sampling=(3, 3)),))
_BEAM_OUT = rcwa.Outputs(diffraction=False, power_region=((-.4, .4), (-.45, .45)))
GRID_STACKS = {
    'region': (rcwa.Layer(thickness=.2, background=Material(1.5), regions=_REGION),),
    'region-free': (rcwa.Layer(thickness=.2, background=Material(1.5)),),
    'no-layers': (),
    'tensor-beside-region': (rcwa.Layer(thickness=.2, background=Material(1.5), regions=_TENSOR),
                             rcwa.Layer(thickness=.2, background=Material(1.5), regions=_REGION)),
}


def _grid_problem(layers, beam=False):
    excitation = rcwa.Excitation(wavelength=1.3, **(_BEAM if beam else dict(sources=(rcwa.PlaneWave(theta=.2),))))
    return rcwa.Problem(stack=rcwa.Stack(period=(1., 1.), layers=layers), excitation=excitation,
                        **(dict(outputs=_BEAM_OUT) if beam else {}))


@pytest.mark.parametrize('orders,grid', [((2, 2), 8), ((3, 1), 12), ((1, 0), 4)])
@pytest.mark.parametrize('stack', sorted(GRID_STACKS))
def test_an_aliasing_shape_nvf_grid_is_refused_at_every_door(stack, orders, grid):
    problem = _grid_problem(GRID_STACKS[stack])
    issue = _refused_alike(problem, rcwa.Numerics(orders=orders, shape_nvf_grid_size=grid))
    assert (issue.code, issue.path) == ('numerics.grid', 'numerics.shape_nvf_grid_size')
    need = 4*max(orders) + 1
    assert f'at least 4*max(orders)+1 = {need}' in issue.message
    assert 'truncation_order' not in issue.message
    # The advice: at the floor, the same problem runs.
    result = rcwa.run(problem, numerics=rcwa.Numerics(orders=orders, shape_nvf_grid_size=need))
    assert np.isfinite(result.T).all()


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
@pytest.mark.parametrize('beam', [False, True], ids=['plane-wave', 'beam'])
def test_an_aliasing_shape_nvf_grid_is_refused_alike_on_each_backend(backend, beam):
    pytest.importorskip(backend)
    problem = _grid_problem(GRID_STACKS['region'], beam=beam)
    x64 = (__import__('jax').enable_x64(True) if backend == 'jax' else __import__('contextlib').nullcontext())
    with x64:
        issue = _refused_alike(problem, rcwa.Numerics(orders=(2, 2), shape_nvf_grid_size=8), backend=backend)
    assert (issue.code, issue.path) == ('numerics.grid', 'numerics.shape_nvf_grid_size')


@pytest.mark.parametrize('layers', [
    (rcwa.Layer(thickness=.2, background=Material(1.5), regions=_TENSOR),),
    (rcwa.Layer(thickness=.2, background=Material(1.5), regions=_TENSOR), rcwa.Layer(thickness=.2, background=Material(1.5))),
    (rcwa.Layer(thickness=.2, background=Material(np.diag([2., 2.5, 3.]))),),
], ids=['tensor-region', 'tensor-beside-region-free', 'tensor-background'])
def test_a_short_grid_that_runs_today_still_runs(layers):
    # A tensor layer reads the grid through its own interpolation, with no aliasing floor, and a
    # tensor stack's region-free layer reads no grid: these ran below the floor and still do.
    import warnings
    problem, numerics = _grid_problem(layers), rcwa.Numerics(orders=(2, 2), shape_nvf_grid_size=8)
    assert rcwa.check(problem, numerics=numerics).ok
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', rcwa.diagnostics.ShapeGridWarning)
        assert np.isfinite(rcwa.run(problem, numerics=numerics).T).all()


# -- 12b. ... and a tensor region layer that reads such a grid warns, at every door ------------

_TENSOR_GRID_WARNS = {
    'tensor-region': ((rcwa.Layer(thickness=.2, background=Material(1.5), regions=_TENSOR),), [0]),
    'tensor-beside-region-free': ((rcwa.Layer(thickness=.2, background=Material(1.5)),
                                   rcwa.Layer(thickness=.2, background=Material(1.5), regions=_TENSOR)), [1]),
}


@pytest.mark.parametrize('coefficients', [None, 'sampled'])
@pytest.mark.parametrize('orders,grid', [((2, 2), 8), ((3, 1), 1)])
@pytest.mark.parametrize('case', sorted(_TENSOR_GRID_WARNS))
def test_a_tensor_layer_below_the_grid_floor_warns_and_runs(case, orders, grid, coefficients):
    layers, warned = _TENSOR_GRID_WARNS[case]
    problem = _grid_problem(layers)
    numerics = rcwa.Numerics(orders=orders, shape_nvf_grid_size=grid, fourier_coefficients=coefficients)
    need = 4*max(orders) + 1
    report = rcwa.check(problem, numerics=numerics)
    assert report.ok
    (issue,) = report.issues
    assert (issue.code, issue.path, issue.severity) == ('numerics.grid', 'numerics.shape_nvf_grid_size', 'warning')
    assert (issue.expected, issue.actual) == (f'>= {need}', grid)
    assert f'4*max(orders)+1 = {need}' in issue.message
    assert ', '.join(f'stack.layers[{i}]' for i in warned) in issue.message
    with pytest.warns(rcwa.diagnostics.ShapeGridWarning, match=f'shape_nvf_grid_size={grid} is below') as caught:
        prepared = rcwa.prepare(problem, numerics=numerics)
    assert caught[0].filename == __file__        # the caller's line, not the package's
    assert prepared.report.issues == report.issues
    with pytest.warns(rcwa.diagnostics.ShapeGridWarning):
        assert np.isfinite(rcwa.run(problem, numerics=numerics).T).all()


@pytest.mark.parametrize('grid', [9, None])
@pytest.mark.parametrize('layers', [
    (rcwa.Layer(thickness=.2, background=Material(1.5), regions=_TENSOR),),
    (rcwa.Layer(thickness=.2, background=Material(np.diag([2., 2.5, 3.]))),),
], ids=['tensor-region', 'tensor-background'])
def test_a_tensor_stack_at_the_floor_or_without_regions_does_not_warn(layers, grid):
    import warnings
    problem = _grid_problem(layers)
    numerics = rcwa.Numerics(orders=(2, 2), shape_nvf_grid_size=grid)
    assert rcwa.check(problem, numerics=numerics).issues == ()
    with warnings.catch_warnings():
        warnings.simplefilter('error', rcwa.diagnostics.ShapeGridWarning)
        rcwa.run(problem, numerics=numerics)
    if grid == 9:
        # A tensor background has no region to sample: grid 8 reads nothing, and says nothing.
        short = rcwa.Numerics(orders=(2, 2), shape_nvf_grid_size=8)
        if not any(layer.regions for layer in layers):
            assert rcwa.check(problem, numerics=short).issues == ()


# -- 13. A localized source without a finite power_region is refused at check, not first at run -

_ISOLATED = rcwa.ElectricDipole(position=(0., 0., .1), moment=(1e-30, 0., 0.), repetition='isolated',
                                sampling=(2, 2))
_PERIODIC = rcwa.ElectricDipole(position=(0., 0., .1), moment=(1e-30, 0., 0.))
_BOX = ((-.4, .4), (-.45, .45))
LOCALIZED = {
    'beam': ((_BEAM['sources'][0],), 0, 'GaussianBeam'),
    'isolated-dipole': ((_ISOLATED,), 0, "ElectricDipole(repetition='isolated')"),
    'plane-then-beam': ((rcwa.PlaneWave(theta=.2), _BEAM['sources'][0]), 1, 'GaussianBeam'),
    'periodic-then-isolated': ((_PERIODIC, _ISOLATED), 1, "ElectricDipole(repetition='isolated')"),
}


def _localized_problem(sources, region):
    return rcwa.Problem(units=Units(length='um'), stack=rcwa.Stack(period=(.8, .9), layers=(
        rcwa.Layer(thickness=.2, background=Material(2.25)),)),
        excitation=rcwa.Excitation(wavelength=.6, sources=sources),
        outputs=rcwa.Outputs(diffraction=False, power_region=region))


@pytest.mark.parametrize('region', [None, 'cell', 'infinite', 'bogus'])
@pytest.mark.parametrize('case', sorted(LOCALIZED))
def test_a_localized_source_without_a_finite_power_region_is_refused_at_every_door(case, region):
    sources, index, name = LOCALIZED[case]
    issue = _refused_alike(_localized_problem(sources, region))
    assert (issue.code, issue.path) == ('outputs.power_region', 'outputs.power_region')
    assert f'excitation.sources[{index}] is a {name}' in issue.message
    assert (issue.expected, issue.actual) == ('((xmin, xmax), (ymin, ymax))', region)
    # The advice: the same sources with a finite rectangle run.
    assert np.isfinite(rcwa.run(_localized_problem(sources, _BOX), numerics=NUMERICS).outputs['top'].outgoing).all()


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_a_localized_source_without_a_finite_power_region_is_refused_alike_on_each_backend(backend):
    pytest.importorskip(backend)
    x64 = (__import__('jax').enable_x64(True) if backend == 'jax' else __import__('contextlib').nullcontext())
    with x64:
        issue = _refused_alike(_localized_problem(_BEAM['sources'], 'cell'), backend=backend)
    assert (issue.code, issue.path) == ('outputs.power_region', 'outputs.power_region')


@pytest.mark.parametrize('region', [None, 'cell', 'infinite'])
@pytest.mark.parametrize('sources', [(_PERIODIC,), (rcwa.IncidentOrders(weights=(((0, 0), 1.), ((1, 0), .5j))),)],
                         ids=['periodic-dipole', 'incident-orders'])
def test_a_periodic_source_still_runs_without_a_finite_power_region(sources, region):
    problem = _localized_problem(sources, region)
    assert rcwa.check(problem, numerics=NUMERICS).ok
    assert np.isfinite(rcwa.run(problem, numerics=NUMERICS).outputs['top'].outgoing).all()


# -- 14. A power_region a coherent-source run cannot read is refused at check, not first at run --

_ORDERS = rcwa.IncidentOrders(weights=(((0, 0), 1.), ((1, 0), .5j)))
COHERENT = {
    'periodic-dipole': (_PERIODIC,),
    'incident-orders': (_ORDERS,),
    'two-planes-one-sector': (rcwa.PlaneWave(theta=.2), rcwa.PlaneWave(theta=.2, polarization='p')),
    'beam': _BEAM['sources'],
}
BAD_RECTANGLES = {'inverted': ((.4, -.4), (-.45, .45)), 'zero-width': ((0., 0.), (-.45, .45)),
                  'three-columns': ((-.4, .4, 0.), (-.45, .45, 0.)), 'one-row': ((-.4, .4),),
                  'ragged': ((-.4, .4), (0.,)), 'complex': ((-.4 + 1j, .4), (-.45, .45)),
                  'number': 1., 'bool': True}


@pytest.mark.parametrize('region', sorted(BAD_RECTANGLES))
@pytest.mark.parametrize('case', sorted(COHERENT))
def test_a_power_region_that_is_not_a_rectangle_is_refused_at_every_door(case, region):
    value = BAD_RECTANGLES[region]
    issue = _refused_alike(_localized_problem(COHERENT[case], value))
    assert (issue.code, issue.path) == ('outputs.power_region', 'outputs.power_region')
    assert 'is not a rectangle' in issue.message and 'xmin < xmax' in issue.message
    assert issue.actual == rcwa.Outputs(power_region=value).power_region


@pytest.mark.parametrize('region', ['bogus', '', 'CELL', 'Infinite'])
@pytest.mark.parametrize('case', ['periodic-dipole', 'incident-orders', 'two-planes-one-sector'])
def test_a_power_region_string_other_than_cell_or_infinite_is_refused_at_every_door(case, region):
    issue = _refused_alike(_localized_problem(COHERENT[case], region))
    assert (issue.code, issue.path) == ('outputs.power_region', 'outputs.power_region')
    assert f'power_region={region!r} names no area' in issue.message
    assert (issue.expected, issue.actual) == ("'cell', 'infinite' or ((xmin, xmax), (ymin, ymax))", region)


SECTORS = {
    'two-planes': ((rcwa.PlaneWave(theta=.2), rcwa.PlaneWave(theta=.5, phi=.3)), 2),
    'dipole-and-plane': ((_PERIODIC, rcwa.PlaneWave(theta=.2)), 2),
    'two-dipoles': ((_PERIODIC, rcwa.ElectricDipole(position=(0., 0., .1), moment=(1e-30, 0., 0.),
                                                     bloch_k=(1., 0.))), 2),
}


@pytest.mark.parametrize('case', sorted(SECTORS))
def test_no_power_region_across_several_bloch_sectors_is_refused_at_every_door(case):
    sources, count = SECTORS[case]
    issue = _refused_alike(_localized_problem(sources, None))
    assert (issue.code, issue.path) == ('outputs.power_region', 'outputs.power_region')
    assert f'fall into {count} Bloch sectors' in issue.message and 'at wavelength 0.6' in issue.message
    assert issue.actual is None
    # The advice: any named region runs.
    for region in ('cell', 'infinite', _BOX):
        problem = _localized_problem(sources, region)
        assert rcwa.check(problem, numerics=NUMERICS).ok
        assert np.isfinite(rcwa.run(problem, numerics=NUMERICS).outputs['top'].outgoing).all()


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('sources,region', [(SECTORS['two-planes'][0], None), ((_ORDERS,), ((.4, -.4), (-.45, .45))),
                                            ((_PERIODIC,), 'bogus')], ids=['sectors', 'rectangle', 'string'])
def test_an_unreadable_power_region_is_refused_alike_on_each_backend(backend, sources, region):
    pytest.importorskip(backend)
    x64 = (__import__('jax').enable_x64(True) if backend == 'jax' else __import__('contextlib').nullcontext())
    with x64:
        issue = _refused_alike(_localized_problem(sources, region), backend=backend)
    assert (issue.code, issue.path) == ('outputs.power_region', 'outputs.power_region')


@pytest.mark.parametrize('region', [None, 'cell', 'infinite', _BOX, [[-.4, .4], [-.45, .45]]],
                         ids=['none', 'cell', 'infinite', 'box', 'list-box'])
@pytest.mark.parametrize('case', ['periodic-dipole', 'incident-orders', 'two-planes-one-sector'])
def test_a_readable_power_region_still_runs(case, region):
    problem = _localized_problem(COHERENT[case], region)
    assert rcwa.check(problem, numerics=NUMERICS).ok
    assert np.isfinite(rcwa.run(problem, numerics=NUMERICS).outputs['top'].outgoing).all()


@pytest.mark.parametrize('region', ['bogus', ((.4, -.4), (-.45, .45)), 1.], ids=['string', 'inverted', 'number'])
def test_a_single_plane_wave_still_runs_beside_a_power_region_it_does_not_read(region):
    problem = _localized_problem((rcwa.PlaneWave(theta=.2),), region)
    problem = dataclasses.replace(problem, outputs=rcwa.Outputs(power_region=region))
    assert rcwa.check(problem, numerics=NUMERICS).ok
    assert np.isfinite(rcwa.run(problem, numerics=NUMERICS).T).all()


# -- 15. An ElectricDipole in a layer that is not one material is refused at check, not at run --

_PATTERNED = rcwa.Layer(thickness=.2, background=Material(2.25), regions=(
    rcwa.Region(geometry=rcwa.Rectangle(size=(.3, .4)), material=Material(4.)),))
_FILM = rcwa.Layer(thickness=.2, background=Material(2.25))
HOSTS = {
    'region': ((_PATTERNED,), 0),
    'raster': ((rcwa.Layer(thickness=.2, epsilon=np.where(np.eye(8) > 0, 4., 2.25)),), 0),
    'second-of-two': ((_FILM, _PATTERNED), 1),
}


def _dipole_problem(layers, sources, z, region=_BOX):
    sources = tuple(dataclasses.replace(s, position=(0., 0., z)) if isinstance(s, rcwa.ElectricDipole) else s
                    for s in sources)
    return rcwa.Problem(units=Units(length='um'), stack=rcwa.Stack(period=(.8, .9), layers=layers),
                        excitation=rcwa.Excitation(wavelength=.6, sources=sources),
                        outputs=rcwa.Outputs(diffraction=False, power_region=region))


@pytest.mark.parametrize('sources,index', [((_PERIODIC,), 0), ((_ISOLATED,), 0),
                                           ((rcwa.PlaneWave(theta=.2), _PERIODIC), 1)],
                         ids=['periodic', 'isolated', 'after-a-plane-wave'])
@pytest.mark.parametrize('host', sorted(HOSTS))
def test_a_dipole_in_a_patterned_layer_is_refused_at_every_door(host, sources, index):
    layers, layer = HOSTS[host]
    issue = _refused_alike(_dipole_problem(layers, sources, z=.2*layer + .1))
    assert (issue.code, issue.path) == ('source.host_layer', f'excitation.sources[{index}]')
    assert f'excitation.sources[{index}] is an ElectricDipole' in issue.message
    assert f'inside stack.layers[{layer}], which is not one material' in issue.message
    assert (issue.expected, issue.actual) == ('a uniform host layer', f'stack.layers[{layer}]')
    # The advice: the same dipole in a uniform layer runs (a raster stack keeps rasters throughout).
    uniform = rcwa.Layer(thickness=.2, epsilon=np.full((8, 8), 2.25)) if host == 'raster' else _FILM
    problem = _dipole_problem(layers + (uniform,), sources, z=.2*len(layers) + .1)
    assert np.isfinite(rcwa.run(problem, numerics=NUMERICS).outputs['top'].outgoing).all()


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_a_dipole_in_a_patterned_layer_is_refused_alike_on_each_backend(backend):
    pytest.importorskip(backend)
    x64 = (__import__('jax').enable_x64(True) if backend == 'jax' else __import__('contextlib').nullcontext())
    with x64:
        issue = _refused_alike(_dipole_problem((_PATTERNED,), (_PERIODIC,), z=.1), backend=backend)
    assert (issue.code, issue.path) == ('source.host_layer', 'excitation.sources[0]')


@pytest.mark.parametrize('layers,z', [((_PATTERNED, _FILM), .3), ((_FILM, _PATTERNED), .1),
                                      ((rcwa.Layer(thickness=.2, epsilon=np.full((8, 8), 2.25)),), .1)],
                         ids=['above-a-patterned-layer', 'below-a-patterned-layer', 'raster-of-one-value'])
def test_a_dipole_in_a_uniform_layer_still_runs(layers, z):
    problem = _dipole_problem(layers, (_PERIODIC,), z)
    assert rcwa.check(problem, numerics=NUMERICS).ok
    assert np.isfinite(rcwa.run(problem, numerics=NUMERICS).outputs['top'].outgoing).all()



# -- 16. A NaN or infinite PlaneWave kx or ky is refused at its field, as every non-finite input --

def _direct(**k):
    return rcwa.Problem(stack=rcwa.Stack(period=(1., 1.), layers=(_FILM,)),
                        excitation=rcwa.Excitation(wavelength=1.3, sources=(rcwa.PlaneWave(**k),)))


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf, complex(np.nan, 0.)],
                         ids=['nan', 'inf', 'minus-inf', 'complex-nan'])
@pytest.mark.parametrize('name,other', [('kx', None), ('ky', None), ('kx', 'ky'), ('ky', 'kx')],
                         ids=['kx', 'ky', 'kx-beside-ky', 'ky-beside-kx'])
def test_a_nonfinite_plane_wave_wavevector_is_refused_at_its_field(name, other, value):
    issue = _refused_alike(_direct(**{name: value, **({other: .5} if other else {})}))
    assert (issue.code, issue.path) == ('problem.nonfinite', f'excitation.sources[0].{name}')
    assert 'non-finite' in issue.message


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_a_nonfinite_plane_wave_wavevector_is_refused_at_its_field_on_each_backend(backend):
    pytest.importorskip(backend)
    x64 = (__import__('jax').enable_x64(True) if backend == 'jax' else __import__('contextlib').nullcontext())
    with x64:
        issue = _refused_alike(_direct(kx=np.nan), backend=backend)
    assert (issue.code, issue.path) == ('problem.nonfinite', 'excitation.sources[0].kx')


def test_a_finite_plane_wave_wavevector_still_runs():
    problem = _direct(kx=1., ky=.5)
    assert rcwa.check(problem, numerics=NUMERICS).ok
    assert np.isfinite(rcwa.run(problem, numerics=NUMERICS).T).all()


# -- 17. A NaN or infinite RCWA incidence angle is problem.nonfinite on every backend ----------

@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
@pytest.mark.parametrize('name,value', [('theta', np.nan), ('phi', -np.inf)])
def test_a_nonfinite_rcwa_angle_is_refused_alike_on_each_backend(backend, name, value):
    pytest.importorskip(backend)
    angles = dict(theta=.2, phi=.1)
    angles[name] = value
    problem = _rcwa().replace(excitation=rcwa.Excitation(wavelength=1.3, sources=(rcwa.PlaneWave(**angles),)))
    x64 = (__import__('jax').enable_x64(True) if backend == 'jax' else __import__('contextlib').nullcontext())
    with x64:
        issue = _refused_alike(problem, backend=backend)
    assert (issue.code, issue.path) == ('problem.nonfinite', f'excitation.sources[0].{name}')


@pytest.mark.parametrize('theta', [2., [.1, .2]], ids=['out-of-range', 'sequence'])
def test_a_finite_bad_rcwa_angle_keeps_source_direction(theta):
    problem = _rcwa().replace(excitation=rcwa.Excitation(wavelength=1.3, sources=(rcwa.PlaneWave(theta=theta),)))
    assert _refused_alike(problem).code == 'source.direction'
