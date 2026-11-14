"""
A gate, not a measurement: every text file the repository's own code opens names its encoding.

``open()``, ``Path.open()``, ``Path.read_text()`` and ``Path.write_text()`` without
``encoding=`` use the locale's codec. That is UTF-8 on macOS and on most Linux setups, cp949 on a
Korean-locale Windows machine and cp1252 on an English one. The sources and documents are
UTF-8 by decision, so under cp949 a read fails at the first em-dash
(``UnicodeDecodeError: 'cp949' codec can't decode byte 0xe2``). Under cp1252 most UTF-8 decodes
without an error, into the wrong characters, so a runner with an English locale rarely sees
the defect at all. Under a Korean locale it stopped 26 of the 27 source-mutation tests, the
``linear_solve`` scan over ``src/wavebit`` and the three documentation checks of
``test_fdtd_table_golden.py``.

The scan covers ``src``, ``tests``, ``tools``, ``examples``, ``tutorials`` and ``docs``. It
leaves out ``studies``: its drivers are research records, and many of them are pinned byte
for byte by ``tools/relocations.json``.

What it reads as a file call missing its encoding:

* builtin ``open(...)`` unless its mode is a literal containing ``b``; a mode held in a
  variable is flagged, since the scan cannot tell text from binary;
* ``x.read_text()`` with no positional argument (a positional one is the encoding itself,
  or ``importlib.metadata.Distribution.read_text(filename)``);
* ``x.write_text(data)`` with only the data;
* ``x.open(mode)`` when the mode is a literal of mode letters without ``b``, and ``x.open()``
  with no argument, except on ``Boundaries`` (``fdtd.Boundaries.open()`` builds an open
  boundary, not a file).

It does not see a process pipe (``subprocess.run(..., text=True)``): those bytes are in the
child's encoding, which no repository decision fixes. Nor does it see a ``.open`` whose mode
is a variable, or a ``tempfile``/``io.TextIOWrapper`` text mode; there is none of either in
the scanned trees today.
"""
import ast
import pathlib
import warnings

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCANNED = ('src', 'tests', 'tools', 'examples', 'tutorials', 'docs')
_MODE_LETTERS = set('rwxabt+')


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


def _unencoded(tree):
    """``(line, form)`` of every text-mode file call in ``tree`` that passes no ``encoding=``."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or any(k.arg == 'encoding' for k in node.keywords):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == 'open':
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


def test_every_text_file_the_repository_opens_names_its_encoding():
    files = [path for top in SCANNED for path in sorted((ROOT / top).rglob('*.py'))
             if '__pycache__' not in path.parts]
    assert len(files) > 500, len(files)                     # a wrong root must not pass empty
    offenders = []
    for path in files:
        name = path.relative_to(ROOT).as_posix()
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
])
def test_the_scan_sees_each_form_it_claims(call, flagged):
    """The gate's own instrument, on each spelling its docstring names, both ways round."""
    assert bool(list(_unencoded(_parse(call, '<case>')))) is flagged
