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
    is read as the reader sees it. The periodic engine is compat-only and is not read.
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


def oblique_on_periodic_faces():
    """A plane wave at 0.3 rad in a cell that is Periodic across it: a refusal names Bloch."""
    return fdtd.Problem(units=Units(length='um'), scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.1, .1, 1.6)),
        boundaries=fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.Open())),
        sources=(fdtd.PlaneWave('pw', plane=fdtd.Plane(z=.5), theta=.3,
                                pulse=fdtd.GaussianPulse(wavelength=.8, frequency_sigma_hz=4e13)),),
        outputs=fdtd.Outputs(monitors=(fdtd.Record('e', at=(0, 0, 0), component='Ey'),)))


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
    """A Region past a PEC wall warns; the remedy is an Open face, not a compat absorber."""
    problem = fdtd.Problem(units=Units(length='um'), scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.4, .4, .4)),
        boundaries=fdtd.Boundaries(x=fdtd.PEC(), y=fdtd.PEC(), z=fdtd.PEC()),
        structures=(fdtd.Structure('block', fdtd.Region(size=(.3, .3, .6), center=(0., 0., 0.)),
                                   Material(epsilon=2.25)),)),
        initial=fdtd.InitialField(E=(None, lambda x, y, z: np.cos(np.pi * x / .4), None)),
        outputs=fdtd.Outputs(monitors=(fdtd.Record('e', at=(0, 0, 0), component='Ey'),)))
    numerics = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.05), duration=fdtd.Duration(steps=4))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fdtd.run(problem, numerics=numerics)
    clipped = [str(w.message) for w in caught if type(w.message).__name__ == 'GeometryClippedWarning']
    assert clipped and all('fdtd.Open()' in m and not COMPAT.search(m) for m in clipped), clipped
    report = fdtd.check(problem, numerics=numerics)
    notes = [w for w in report.capabilities['warnings'] if 'loses' in w]
    assert notes and not any(COMPAT.search(w) for w in notes), notes
