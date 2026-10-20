"""What the public FDTD module says to its caller is spelled in its own vocabulary.

``wavebit.fdtd`` runs on the private normalized general-scene engine, whose refusals and
warnings name compat calls: a public caller with an oblique plane wave was told to "Use
fd.Bloch.from_source(pw, ...)", a call with another signature in a namespace they never
imported. ``wavebit.fdtd._spelling`` translates what leaves the public entry points; these tests
hold every message the engine can raise to it, the reported one end to end, and the compat
text unchanged.
"""
import ast
import pathlib
import re
import warnings

import numpy as np
import pytest

import wavebit
import wavebit.fdtd._legacy as fd
from wavebit import Material, Units, fdtd
from wavebit.errors import ValidationError
from wavebit.fdtd._lower_public import lower_problem
from wavebit.fdtd._spelling import public, spoken, to_public

ENGINE = pathlib.Path(wavebit.fdtd.__file__).parent

#: A compat name left in a message, as a public caller would read it.
COMPAT = re.compile(r'(?<![\w.])fd\.\w')


def _docstrings(tree):
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            first = node.body[0] if node.body else None
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                out.add(id(first.value))
    return out


def engine_strings():
    """``(where, text)`` for every non-docstring string the general engine holds.

    An f-string is read whole, with ``{}`` where a value goes, so a phrase split around a value
    is read as the reader sees it. The explicit-grid engine, ``_periodic``, is not read: what it
    says leaves through ``fdtd.native`` and ``fdtd.periodic``, which run on it and spell its
    keywords as their own declarations at their own doors, not through ``_spelling``.
    """
    for path in sorted(ENGINE.rglob('*.py')):
        if '_periodic' in path.parts or path.name == '_spelling.py':
            continue
        tree = ast.parse(path.read_text(encoding='utf-8'))
        skip = _docstrings(tree)
        inside = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.JoinedStr):
                parts = []
                for value in node.values:
                    inside.add(id(value))
                    if isinstance(value, ast.Constant):
                        parts.append(value.value)
                    else:
                        parts.append('{' + ast.unparse(value.value) + '}')
                yield f'{path.relative_to(ENGINE)}:{node.lineno}', ''.join(parts)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and id(node) not in skip and id(node) not in inside):
                yield f'{path.relative_to(ENGINE)}:{node.lineno}', node.value


#: The one place a compat name is formatted in: ``fd.{type(material).__name__}``, which is a
#: Material, a Conductor or a Dispersive at run time.
DYNAMIC = re.compile(r'(?<![\w.])fd\.\{type\(\w+\)\.__name__\}')


def test_every_compat_name_an_engine_message_holds_has_a_public_spelling():
    """No string in the engine leaves a compat name behind once spelled for a public caller."""
    read, left = 0, []
    for where, text in engine_strings():
        if 'fd.' not in text:
            continue
        read += 1
        out = to_public(DYNAMIC.sub('fd.Conductor', text))
        if COMPAT.search(out):
            left.append(f'{where}: {COMPAT.findall(out)}')
    assert read > 100, read          # the scan reads the messages (121 on 2026-10-15), not nothing
    assert not left, 'compat names with no public spelling:\n  ' + '\n  '.join(left)


def test_every_name_a_public_message_offers_exists_where_it_says():
    """A translated name is one the caller can reach: ``fdtd.X``, a shared material, or compat."""
    offered = set()
    for _, text in engine_strings():
        if 'fd.' in text:
            offered |= set(re.findall(r'(?<![\w.])fdtd\.(\w+)', to_public(text)))
            assert 'wavebit.compat' not in to_public(text)
    for name in sorted(offered):
        assert hasattr(fdtd, name), f'fdtd.{name}'
    for kind in ('Material', 'Conductor', 'Dispersive'):
        out = to_public(f'is an fd.{kind}.')
        assert re.fullmatch(r'is a (\w+)\.', out) and hasattr(wavebit.materials,
                                                             out[5:-1]), out


def test_every_module_path_a_message_names_exists():
    """Messages cite the engine's files for their measurements; each citation must resolve.

    Eleven cited ``src/wavebit/fd/_lower/...`` and ``src/wavebit/fd/...``, the layout the src
    move renamed, wrong for a compat caller and a public one alike.
    """
    root = ENGINE.parents[2]
    missing = sorted({f'{where}: {path}' for where, text in engine_strings()
                      for path in re.findall(r'src/wavebit/[\w/]+\.py', text)
                      if not (root / path).is_file()})
    assert not missing, '\n'.join(missing)


#: A path the engine's text cites: a source file or folder, a page, a test or a study record.
CITED = re.compile(r'(?<![\w/.])((?:src/wavebit|docs|tests|studies)/[\w/.*-]*[\w*/])')


def test_every_path_the_engine_cites_anywhere_exists():
    """Docstrings and comments cite files too, and so do messages outside ``src/``.

    The src move left ``src/wavebit/fd/_shapes`` and ``_lower.checks.check_seams`` in fourteen
    docstrings; the docs reorganisation left twelve ``fd-*.md`` pages cited at the root of
    ``docs/`` (now under ``docs/validation/fdtd/`` and ``docs/design/``) and the branch-cut page
    at its old place in the messages a reader is told to follow, and a moved study log's old
    path twice. Every cited path resolves, a glob matches, a function cited in a file is defined
    there, and no dotted ``_lower.`` module is named.
    """
    root = ENGINE.parents[2]
    missing = []
    for path in sorted(ENGINE.rglob('*.py')):
        where = path.relative_to(ENGINE)
        text = path.read_text(encoding='utf-8')
        for cited in CITED.findall(text):
            cited = cited.rstrip('.')
            if not (any(root.glob(cited)) if '*' in cited else (root / cited).exists()):
                missing.append(f'{where}: {cited}')
        for cited, name in re.findall(r'(src/wavebit/[\w/]+\.py)::(\w+)', text):
            source = root / cited
            if source.is_file() and not re.search(rf'^\s*(?:def|class) {name}\b|^{name} =',
                                                  source.read_text(encoding='utf-8'), re.M):
                missing.append(f'{where}: {cited}::{name}')
        missing += [f'{where}: {dotted}' for dotted in re.findall(r'(?<![\w.])_lower\.\w+', text)]
    assert not missing, '\n'.join(missing)


def oblique_on_periodic_faces():
    """A plane wave at 0.3 rad in a cell that is Periodic across it: a refusal names Bloch."""
    return fdtd.Problem(units=Units(length='um'), scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.1, .1, 1.6)),
        boundaries=fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.Open())),
        sources=(fdtd.PlaneWave('pw', plane=fdtd.Plane(z=.5), theta=.3,
                                pulse=fdtd.GaussianPulse(wavelength=.8, frequency_sigma_hz=4e13)),),
        outputs=fdtd.Outputs(monitors=(fdtd.TimeSeriesMonitor('e', at=(0, 0, 0), component='Ey'),)))


NUMERICS = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.025), duration=fdtd.Duration(steps=20),
                         absorber=fdtd.PML(cells=12))


def test_the_reported_bloch_refusal_names_the_public_call():
    report = fdtd.check(oblique_on_periodic_faces(), numerics=NUMERICS)
    (issue,) = report.issues
    assert 'fdtd.Bloch.from_source(source, units=problem.units' in issue.message
    assert not COMPAT.search(issue.message), issue.message
    for door in (fdtd.prepare, fdtd.run):
        with pytest.raises(ValidationError) as error:
            door(oblique_on_periodic_faces(), numerics=NUMERICS)
        assert 'fdtd.Bloch.from_source(' in str(error.value)
        assert not COMPAT.search(str(error.value))
    # The call it names is the one that fixes it.
    problem = oblique_on_periodic_faces()
    faces = fdtd.Bloch.from_source(problem.sources[0], units=problem.units)
    fixed = problem.replace(scene=problem.scene.replace(boundaries=fdtd.Boundaries(**faces, z=fdtd.Open())))
    assert fdtd.check(fixed, numerics=NUMERICS).ok


def test_a_compat_caller_still_reads_the_compat_spelling():
    scene, _ = lower_problem(oblique_on_periodic_faces(), NUMERICS)
    with pytest.raises(ValueError, match=re.escape('fd.Boundaries(**fd.Bloch.from_source(pw')):
        fd.plan(scene)
    assert spoken('fd.plan(scene)') == 'fd.plan(scene)'
    with public():
        assert spoken('fd.plan(scene)') == 'fdtd.prepare(problem, numerics=...)'
    assert spoken('fd.plan(scene)') == 'fd.plan(scene)'


def test_a_planning_warning_is_spelled_for_the_public_caller():
    """A Region past a PEC wall warns; the remedy is an Open face, not a compat absorber.

    With no source and no Open face the scene has no wavelength for the mesh check, which said
    so naming ``Resolution(wavelength=)``, a call with no public counterpart.
    """
    problem = fdtd.Problem(units=Units(length='um'), scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.4, .4, .4)),
        boundaries=fdtd.Boundaries(x=fdtd.PEC(), y=fdtd.PEC(), z=fdtd.PEC()),
        structures=(fdtd.Structure('block', shape=fdtd.SnappedBox(size=(.3, .3, .6), center=(0., 0., 0.)),
                                   material=Material(epsilon=2.25)),)),
        initial=fdtd.InitialField(E=(None, lambda x, y, z: np.cos(np.pi * x / .4), None)),
        outputs=fdtd.Outputs(monitors=(fdtd.TimeSeriesMonitor('e', at=(0, 0, 0), component='Ey'),)))
    numerics = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.05), duration=fdtd.Duration(steps=4))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fdtd.run(problem, numerics=numerics)
    clipped = [str(w.message) for w in caught if type(w.message).__name__ == 'GeometryClippedWarning']
    assert clipped and all('fdtd.Open()' in m and not COMPAT.search(m) for m in clipped), clipped
    coarse = [str(w.message) for w in caught if type(w.message).__name__ == 'CoarseMeshWarning']
    assert coarse and all('no source, and no fdtd.PML(wavelength=...)' in m
                          and not UNPREFIXED.search(m) for m in coarse), coarse
    report = fdtd.check(problem, numerics=numerics)
    notes = [w for w in report.capabilities['warnings'] if 'loses' in w]
    assert notes and not any(COMPAT.search(w) for w in notes), notes


#: The engine's vocabulary where the public module has its own: a compat name, a compat-only
#: constructor, the old source tree, the compat result's variable.
VOCABULARY = re.compile(r'(?<![\w.])fd\.\w|(?<![\w.])(?:Run|Resolution|Band|Decay|Absorber)\('
                        r'|src/wavebit/fd/|(?<![\w.])res\.diagnostics')


@pytest.mark.parametrize('build', [
    lambda: fdtd.SnappedSlab(x=(0., 1.), z=(0., 1.)),
    lambda: fdtd.SnappedSlab(size=1.),
    lambda: fdtd.SnappedSlab(radius=1.),
], ids=['two axes', 'size', 'radius'])
def test_a_shared_constructor_refuses_a_public_caller_in_the_public_vocabulary(build):
    """``fdtd.SnappedSlab`` refuses before any entry point runs, so the public name translates itself.

    It said "a rectangular rod or block is fd.Structure(fd.Region(size=..., center=...),
    material)", a call with another signature, from a namespace the caller never imported.
    """
    with pytest.raises(NotImplementedError) as error:
        build()
    message = str(error.value)
    assert 'fdtd.SnappedBox(size=..., center=...)' in message and 'fdtd.Box(' in message, message
    assert not VOCABULARY.search(message), message


def test_an_engine_shape_names_a_renamed_declaration_by_its_public_name():
    """The engine spells a renamed declaration publicly where it raises; nothing translates it."""
    with pytest.raises(NotImplementedError, match=re.escape(
            'fd.Structure(fdtd.SnappedBox(size=..., center=...), material)')):
        fd.Slab(x=(0., 1.), z=(0., 1.))
    with pytest.raises(NotImplementedError, match=re.escape('fdtd.SnappedBox(size=..., center=...)')):
        fd.Slab(size=1.)


@pytest.mark.parametrize('build', [
    lambda v: fdtd.SnappedSlab(z=(v, 1.)),
    lambda v: fdtd.SnappedBox(size=(v, 1., 1.)),
    lambda v: fdtd.Plane(z=v),
    lambda v: fdtd.Domain(size=(v, 1., 1.)),
    lambda v: fdtd.TimeSeriesMonitor('probe', at=(v, 0., 0.)),
], ids=['Slab', 'Region', 'Plane', 'Domain', 'Record'])
def test_a_traced_length_is_refused_with_the_public_leaves(build):
    """The engine's list named a Dipole's ``range=``, the Bloch phase and ``Band(df=...)``.

    A public ``Prepared`` holds the boundaries, a Dipole's centre and the initial fields static,
    so its leaves are the contract's ``live`` ones, spelled in its own vocabulary.
    """
    jax = pytest.importorskip('jax')
    with pytest.raises(ValueError) as error:
        jax.grad(lambda v: (build(v), v)[1])(0.3)
    message = str(error.value)
    assert 'is static' in message and 'Numerics(geometry_ranges=...)' in message, message
    assert 'GaussianPulse(' in message and not VOCABULARY.search(message), message
    assert 'Bloch' not in message and 'range=' not in message, message


def test_the_public_leaves_are_reviewed_when_the_engine_gains_one():
    """``PUBLIC_LEAVES`` is written, not derived: a twelfth compat leaf was the last reviewed."""
    from wavebit._common.declarations import TRACED_LEAVES
    assert len(TRACED_LEAVES) == 12, ('the engine has a new traced leaf; say whether the public '
                                      'module has it in wavebit.fdtd._spelling.PUBLIC_LEAVES')


def _snapped():
    """A cell-snapped film and rod in a cell closed by PEC across z."""
    return fdtd.Problem(units=Units(length='um'), scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.4, .4, .4)),
        boundaries=fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.PEC()),
        structures=(fdtd.Structure('film', shape=fdtd.SnappedSlab(z=(-.1, .05)), material=Material(epsilon=2.25)),
                    fdtd.Structure('rod', shape=fdtd.SnappedBox(size=(.1, .1, .1), center=(0., 0., .1)),
                                   material=Material(epsilon=4.)))),
        initial=fdtd.InitialField(E=(None, lambda x, y, z: np.cos(np.pi * z / .4), None)),
        outputs=fdtd.Outputs(monitors=(fdtd.TimeSeriesMonitor('e', at=(0, 0, 0), component='Ey'),)))


def _flush():
    """A pillar standing flush on a smoothed substrate -- a junction -- under a cell-snapped film."""
    glass = Material(epsilon=2.25)
    return fdtd.Problem(units=Units(length='um'), scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.8, .8, 1.2)),
        boundaries=fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.PEC()),
        structures=(
            fdtd.Structure('substrate', shape=fdtd.Box(size=(2., 2., 1.2125), center=(0., 0., -.59375)),
                           material=glass),
            fdtd.Structure('pillar', shape=fdtd.Cylinder(radius=.15, height=.4,
                                                   center=(.017, -.013, .2125)), material=Material(epsilon=4.)),
            fdtd.Structure('film', shape=fdtd.SnappedSlab(z=(.5, .55)), material=glass))),
        sources=(fdtd.CurrentDipole('d', center=(.3, .3, .3),
                             pulse=fdtd.GaussianPulse(wavelength=1.6, frequency_sigma_hz=2e13)),),
        outputs=fdtd.Outputs(monitors=(fdtd.TimeSeriesMonitor('e', at=(0, 0, 0), component='Ey'),)))


@pytest.mark.parametrize('problem, junctions', [(_snapped, 0), (_flush, 1)],
                         ids=['snapped', 'junction'])
def test_a_public_name_changes_no_plan(problem, junctions):
    """The public names add no field: a scene lowered from them plans as its compat twin does.

    A flush pair's record hashes every structure's class, and named the public one's module
    until ``junction._signature`` read the engine's.
    """
    numerics = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.05), duration=fdtd.Duration(steps=2))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        scene, _ = lower_problem(problem(), numerics)
        compat = {fdtd.SnappedSlab: fd.Slab, fdtd.SnappedBox: fd.Region}
        twin = scene.replace(
            structures=tuple(s.replace(shape=compat[type(s.shape)](**vars(s.shape)))
                             if type(s.shape) in compat else s for s in scene.structures))
        assert not any(type(s.shape) in compat for s in twin.structures)
        # A public monitor lowers to the engine's own class, so it has no twin to build.
        assert {type(m) for m in scene.monitors} == {fd.Record}
        plan = fd.plan(scene)
        assert len(plan.junctions) == junctions
        assert plan.skeleton == fd.plan(twin).skeleton


def test_the_public_reference_speaks_the_public_vocabulary():
    """``docs/reference/fdtd`` renders each public name with ``:members:``; none names a compat call.

    ``fdtd.Region`` was documented as ``fd.Region(size=..., center=...)``, "a structure,
    ``fd.Structure(fd.Region(...), material)``", and ``fdtd.Axis1D`` as
    ``fd.Boundaries(x=fd.Bloch(k=kx), ..., z=fd.PML(...))`` read off ``res.diagnostics``: the
    engine's own docstrings, rendered for a public reader. The warnings the public guide says
    come from ``wavebit.fdtd.diagnostics`` are held to it too: they cited ``fd._assemble``,
    ``fd.check`` and ``fd.PML``.
    """
    import inspect
    from wavebit.fdtd import diagnostics
    rendered = []
    for module, name in ([(fdtd, n) for n in fdtd.__all__]
                         + [(diagnostics, n) for n in diagnostics.__all__]):
        obj = getattr(module, name)
        rendered.append((name, inspect.getdoc(obj)))
        if isinstance(obj, type):
            rendered += [(f'{name}.{member}', inspect.getdoc(getattr(obj, member)))
                         for member in vars(obj) if not member.startswith('_')]
    left = [f'{where}: {VOCABULARY.findall(doc)}' for where, doc in rendered
            if doc and VOCABULARY.search(doc)]
    assert len(rendered) > 60 and not left, '\n'.join(left)


#: The engine's calls written without the prefix, where the public module spells them otherwise.
UNPREFIXED = re.compile(r'(?<![\w.])(?:(?:Run|Resolution|Band|Decay|Absorber)\('
                        r'|Structure\((?:junction|discretisation|memory_bound)='
                        r'|(?:Sphere|Cylinder|Box|Prism)\(range=|plan=)')


def test_every_unprefixed_compat_call_an_engine_message_holds_has_a_public_spelling():
    """``Run(steps=)`` is ``Duration(steps=)``, ``Structure(junction=)`` is ``Numerics(junction=)``.

    The ``fd.`` scan reads past a call the engine writes bare, and 47 strings do: a public caller
    with ``Duration(steps=0)`` was told "Run(steps=0): steps is a positive integer".
    """
    read, left = 0, []
    for where, text in engine_strings():
        if UNPREFIXED.search(text):
            read += 1
            out = to_public(text)
            if UNPREFIXED.search(out):
                left.append(f'{where}: {UNPREFIXED.findall(out)}')
    assert read > 30, read          # 39 on 2026-10-19, once renamed calls were spelled at the source
    assert not left, 'unprefixed compat calls with no public spelling:\n  ' + '\n  '.join(left)


def _film(pulse=fdtd.GaussianPulse(wavelength=.8, frequency_sigma_hz=4e13)):
    """A plane wave onto a glass film in a periodic cell."""
    return fdtd.Problem(units=Units(length='um'), scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.1, .1, 1.6)),
        boundaries=fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.Open()),
        structures=(fdtd.Structure('film', shape=fdtd.SnappedSlab(z=(-.4, -.2)), material=Material(epsilon=2.25)),)),
        sources=(fdtd.PlaneWave('pw', plane=fdtd.Plane(z=.5), pulse=pulse),),
        outputs=fdtd.Outputs(monitors=(fdtd.TimeSeriesMonitor('e', at=(0, 0, 0), component='Ey'),)))


@pytest.mark.parametrize('pulse, changes, said', [
    (None, dict(duration=fdtd.Duration(steps=0)), 'Duration(steps=0): steps is a positive integer.'),
    (None, dict(duration=fdtd.Duration(seconds=-1e-15)),
     'Duration(seconds=...) or Duration(periods=..., source=...): a positive length of time.'),
    (None, dict(courant=1.5), 'Numerics(courant=1.5): dt = courant'),
    (None, dict(junction='blend'), "Numerics(junction='blend'): choose split"),
    (None, dict(discretization='bogus'), "Numerics(discretization='bogus'): the two spellings"),
    (None, dict(grid=fdtd.UniformGrid(spacing=0.)),
     'UniformGrid(spacing=0.0): the spacing is a positive length.'),
    (fdtd.GaussianPulse(wavelength=.8, frequency_sigma_hz=1e15), {},
     'GaussianPulse(wavelength=0.8, frequency_sigma_hz=...): the spectrum reaches zero frequency'),
    (fdtd.GaussianPulse(wavelength=.8, frequency_sigma_hz=-1.), {},
     'GaussianPulse(frequency_sigma_hz=...): a positive number is needed.'),
], ids=['steps', 'seconds', 'courant', 'junction', 'discretization', 'spacing', 'wide pulse',
        'negative width'])
def test_a_refusal_the_engine_spells_bare_reaches_the_public_caller_in_its_spelling(pulse, changes,
                                                                                    said):
    """Each is raised by an engine constructor the lowering calls, and reported by ``check``.

    A bandwidth and an end time are held in the engine's units, so their public spelling says
    which keyword without the engine's number.
    """
    numerics = dict(grid=fdtd.UniformGrid(spacing=.025), duration=fdtd.Duration(steps=20),
                    absorber=fdtd.PML(cells=12))
    numerics.update(changes)
    report = fdtd.check(_film(*([pulse] if pulse else [])), numerics=fdtd.Numerics(**numerics))
    (issue,) = report.issues
    assert said in issue.message, issue.message
    assert not UNPREFIXED.search(issue.message) and not VOCABULARY.search(issue.message), issue.message


def _oblique(theta, **faces):
    """A plane wave at ``theta`` and phi 0.2 rad across the transverse ``faces``."""
    return fdtd.Problem(units=Units(length='um'), scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.1, .1, 1.6)), boundaries=fdtd.Boundaries(**faces, z=fdtd.Open())),
        sources=(fdtd.PlaneWave('pw', plane=fdtd.Plane(z=.5), theta=theta, phi=.2,
                                pulse=fdtd.GaussianPulse(wavelength=.8, frequency_sigma_hz=4e13)),),
        outputs=fdtd.Outputs(monitors=(fdtd.TimeSeriesMonitor('e', at=(0, 0, 0), component='Ey'),)))


@pytest.mark.parametrize('entry', ['check', 'prepare'])
@pytest.mark.parametrize('field, value, choices', [
    ('junction', 'blend', 'choose split, staircase, or refuse'),
    ('junction', None, 'choose split, staircase, or refuse'),
    ('discretization', 'bogus', "the two spellings are 'voxel' and 'stencil'"),
    ('discretization', None, "the two spellings are 'voxel' and 'stencil'"),
])
def test_public_numerics_refusal_preserves_message_and_issue_fields(entry, field, value, choices):
    numerics = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.025),
                            duration=fdtd.Duration(steps=20), absorber=fdtd.PML(cells=12),
                            **{field: value})
    if entry == 'check':
        (issue,) = fdtd.check(_film(), numerics=numerics).issues
    else:
        with pytest.raises(ValidationError) as error:
            fdtd.prepare(_film(), numerics=numerics)
        issue = error.value.issue
    assert issue.message == f'Numerics({field}={value!r}): {choices}.'
    assert issue.code == 'numerics.choice'
    assert issue.path == f'numerics.{field}'
    assert issue.actual == value
    assert issue.expected is None
    assert issue.severity == 'error'


@pytest.mark.parametrize('theta, faces, said', [
    (.3, dict(x=fdtd.Periodic(), y=fdtd.Periodic()), "PlaneWave('pw', theta=0.3, phi=0.2) has"),
    (.3, dict(x=fdtd.Bloch(k=1.), y=fdtd.Bloch(k=1.)), 'wavelength 0.8, theta 0.3, phi 0.2)'),
    (2., dict(x=fdtd.Periodic(), y=fdtd.Periodic()),
     'PlaneWave(theta=2): the polar angle from the propagation axis, in radians, in [0, 1.5708).'),
], ids=['periodic', 'bloch', 'range'])
def test_a_plane_wave_angle_is_quoted_in_the_radians_the_caller_gave(theta, faces, said):
    """The lowering hands the engine degrees, and its refusals quoted them: 0.3 read 17.1887."""
    (issue,) = fdtd.check(_oblique(theta, **faces), numerics=NUMERICS).issues
    assert said in issue.message, issue.message


def test_a_compat_plane_wave_angle_is_still_quoted_in_degrees():
    with pytest.raises(ValueError, match=re.escape(
            'PlaneWave(theta=95): the polar angle from the propagation axis, in degrees, '
            'in [0, 90).')):
        fd.PlaneWave('pw', band=fd.Band(wavelength=.8, df=.1), z=.5, theta=95)
