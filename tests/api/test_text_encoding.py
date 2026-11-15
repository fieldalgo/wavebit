"""
A gate, not a measurement: every text file and process pipe the repository's own code reads or
writes names its encoding.

``open()``, ``Path.open()``, ``Path.read_text()`` and ``Path.write_text()`` without
``encoding=`` use the locale's codec. That is UTF-8 on macOS and on most Linux setups, cp949 on a
Korean-locale Windows machine and cp1252 on an English one. The sources and documents are
UTF-8 by decision, so under cp949 a read fails at the first em-dash
(``UnicodeDecodeError: 'cp949' codec can't decode byte 0xe2``). Under cp1252 most UTF-8 decodes
without an error, into the wrong characters, so a runner with an English locale rarely sees
the defect at all. Under a Korean locale it stopped 26 of the 27 source-mutation tests, the
``linear_solve`` scan over ``src/wavebit`` and the three documentation checks of
``test_fdtd_table_golden.py``.

The scan covers ``src``, ``tests``, ``tools``, ``examples``, ``tutorials``, ``docs`` and
``studies``. It leaves out only the files the ``immutable`` ledger of
``tools/relocations.json`` pins byte for byte: those are preserved records, which
``test_every_preserved_reference_and_record_keeps_its_original_bytes`` forbids editing.

What it reads as a file call missing its encoding:

* builtin ``open(...)`` unless its mode is a literal containing ``b``; a mode held in a
  variable is flagged, since the scan cannot tell text from binary;
* ``x.read_text()`` with no positional argument (a positional one is the encoding itself,
  or ``importlib.metadata.Distribution.read_text(filename)``);
* ``x.write_text(data)`` with only the data;
* ``x.open(mode)`` when the mode is a literal of mode letters without ``b``, and ``x.open()``
  with no argument, except on ``Boundaries`` (``fdtd.Boundaries.open()`` builds an open
  boundary, not a file).

A process pipe decodes with the locale's codec as well. What it reads as a pipe missing its
encoding:

* ``run``, ``check_output``, ``Popen``, ``call``, ``check_call``, ``getoutput`` and
  ``getstatusoutput`` of ``subprocess``, under any name an import binds to them, that ask for
  text (``text=``, ``universal_newlines=`` or ``errors=``) and pass no ``encoding=``;
* ``os.popen(...)``, which has no ``encoding=`` and always decodes with the locale's codec.

The parent's encoding is only half of a pipe: the child writes the other half. A Python child
writes in the locale's codec too, so on a Korean-locale machine a child that prints an
em-dash, which cp949 cannot encode, dies with ``UnicodeEncodeError``. The repository makes its
Python children write UTF-8. ``tests/conftest.py`` sets ``PYTHONIOENCODING=utf-8`` for every
child of the suite, and ``test_a_child_of_the_suite_writes_utf8`` checks that it takes effect.
The tools and studies set the same variable for each child they start, and a child run with
``-I``, which ignores ``PYTHON*`` variables, sets its own streams. A child that is another
program writes its own encoding: ``git`` writes UTF-8, and ``tasklist`` is read in the
console's ``oem`` code page.

It does not see text mode passed through ``**kwargs``, a ``.open`` whose mode is a variable,
or a ``tempfile``/``io.TextIOWrapper`` text mode; there is none of these in the scanned trees
today. It does not check the child's side of a pipe.
"""
import ast
import json
import pathlib
import subprocess
import sys
import warnings

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCANNED = ('src', 'tests', 'tools', 'examples', 'tutorials', 'docs', 'studies')
_MODE_LETTERS = set('rwxabt+')
_PROCESS_CALLS = {'run', 'check_output', 'Popen', 'call', 'check_call', 'getoutput', 'getstatusoutput'}
_TEXT_MODE = ('text', 'universal_newlines', 'errors')


def _preserved():
    """Repository paths whose bytes the relocation ledger pins: records, not maintained code."""
    ledger = json.loads((ROOT / 'tools/relocations.json').read_text(encoding='utf-8'))['immutable']
    return {row['target'] for row in ledger.values()}


def _mode(call, index):
    """The literal mode of ``call``, ``'?'`` when it is not a literal, ``None`` when absent."""
    for keyword in call.keywords:
        if keyword.arg == 'mode':
            value = keyword.value
            return value.value if isinstance(value, ast.Constant) and isinstance(value.value, str) else '?'
    if len(call.args) > index:
        value = call.args[index]
        return value.value if isinstance(value, ast.Constant) and isinstance(value.value, str) else '?'
    return None


def _subprocess_names(tree):
    """Names that reach ``subprocess`` in ``tree``: module aliases, and imported functions."""
    modules, functions = {'subprocess'}, {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {alias.asname or alias.name for alias in node.names if alias.name == 'subprocess'}
        elif isinstance(node, ast.ImportFrom) and node.module == 'subprocess':
            functions |= {alias.asname or alias.name: alias.name for alias in node.names
                          if alias.name in _PROCESS_CALLS}
    return modules, functions


def _asks_for_text(call):
    return any(k.arg in _TEXT_MODE and not (isinstance(k.value, ast.Constant) and k.value.value in (False, None))
               for k in call.keywords)


def _unencoded(tree):
    """``(line, form)`` of every text-mode file call or pipe in ``tree`` that passes no ``encoding=``."""
    modules, functions = _subprocess_names(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or any(k.arg == 'encoding' for k in node.keywords):
            continue
        func = node.func
        process = None
        if isinstance(func, ast.Name) and func.id in functions:
            process = functions[func.id]
        elif (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
              and func.value.id in modules and func.attr in _PROCESS_CALLS):
            process = func.attr
        if process is not None:
            if process in ('getoutput', 'getstatusoutput') or _asks_for_text(node):
                yield node.lineno, f'subprocess.{process}'
            continue
        if (isinstance(func, ast.Attribute) and func.attr == 'popen'
                and isinstance(func.value, ast.Name) and func.value.id == 'os'):
            yield node.lineno, 'os.popen'
        elif isinstance(func, ast.Name) and func.id == 'open':
            mode = _mode(node, 1)
            if mode is None or mode == '?' or 'b' not in mode:
                yield node.lineno, 'open'
        elif isinstance(func, ast.Attribute):
            if func.attr == 'read_text' and not node.args:
                yield node.lineno, 'read_text'
            elif func.attr == 'write_text' and len(node.args) == 1:
                yield node.lineno, 'write_text'
            elif func.attr == 'open':
                if not node.args and not node.keywords:
                    receiver = func.value
                    name = receiver.attr if isinstance(receiver, ast.Attribute) else getattr(receiver, 'id', None)
                    if name != 'Boundaries':
                        yield node.lineno, '.open'
                    continue
                mode = _mode(node, 0)
                if mode not in (None, '?') and set(mode) <= _MODE_LETTERS and 'b' not in mode:
                    yield node.lineno, '.open'


def _parse(source, name):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', SyntaxWarning)      # an invalid escape is not this gate's
        return ast.parse(source, name)


def test_every_text_file_and_pipe_the_repository_opens_names_its_encoding():
    files = [path for top in SCANNED for path in sorted((ROOT / top).rglob('*.py'))
             if '__pycache__' not in path.parts]
    assert len(files) > 500, len(files)                     # a wrong root must not pass empty
    preserved = _preserved()
    offenders = []
    for path in files:
        name = path.relative_to(ROOT).as_posix()
        if name in preserved:
            continue
        tree = _parse(path.read_bytes().decode('utf-8'), name)
        offenders += [f'{name}:{line}: {form} without encoding=' for line, form in _unencoded(tree)]
    assert not offenders, '\n'.join(offenders)


@pytest.mark.parametrize('call, flagged', [
    ("open(p)", True),
    ("open(p, 'w')", True),
    ("open(p, mode)", True),
    ("open(p, 'rb')", False),
    ("open(p, mode='wb')", False),
    ("open(p, encoding='utf-8')", False),
    ("p.read_text()", True),
    ("p.read_text(encoding='utf-8')", False),
    ("p.read_text('utf-8')", False),
    ("distribution.read_text('direct_url.json')", False),
    ("p.write_text(s)", True),
    ("p.write_text(s, encoding='utf-8')", False),
    ("p.write_text(s, 'utf-8')", False),
    ("p.open()", True),
    ("p.open('a')", True),
    ("p.open(mode='x')", True),
    ("p.open('xb')", False),
    ("p.open('w', encoding='utf-8')", False),
    ("fdtd.Boundaries.open()", False),
    ("fdtd.Boundaries.open(pml)", False),
    ("tarfile.open(sdist)", False),
    ("archive.open('manifest.json')", False),
    ("subprocess.run(c, text=True)", True),
    ("subprocess.run(c, capture_output=True, text=True, encoding='utf-8')", False),
    ("subprocess.run(c, capture_output=True)", False),
    ("subprocess.run(c, text=False)", False),
    ("subprocess.check_output(c, universal_newlines=True)", True),
    ("subprocess.Popen(c, stdout=PIPE, errors='replace')", True),
    ("subprocess.Popen(c, encoding='oem', errors='replace')", False),
    ("subprocess.getoutput('ls')", True),
    ("import subprocess as sp\nsp.call(c, text=True)", True),
    ("from subprocess import check_output as out\nout(c, text=True)", True),
    ("from subprocess import run\nrun(c, capture_output=True)", False),
    ("os.popen('ls')", True),
    ("runner.run(c, text=True)", False),
])
def test_the_scan_sees_each_form_it_claims(call, flagged):
    """The gate's own instrument, on each spelling its docstring names, both ways round."""
    assert bool(list(_unencoded(_parse(call, '<case>')))) is flagged


def test_a_child_of_the_suite_writes_utf8():
    """The child half of a pipe: a Python child the suite starts prints what cp949 cannot encode.

    Without ``PYTHONIOENCODING=utf-8`` from ``tests/conftest.py``, a Korean-locale Windows child
    dies on the em-dash. Where the locale is already UTF-8 (macOS, Linux CI) this passes either way.
    """
    text = '— ✓ θ'
    child = subprocess.run([sys.executable, '-c', f'print({text!r})'], capture_output=True,
                           text=True, encoding='utf-8')
    assert child.returncode == 0, child.stderr
    assert child.stdout.strip() == text
