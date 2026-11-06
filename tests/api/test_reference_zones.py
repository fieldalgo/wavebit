"""Where a reference tool may be loaded, checked by path.

Every git-tracked ``.py`` file, and every code cell of a tracked tutorial notebook, falls in
one zone, decided by its path. A reference tool whose license is not permissive (GPL, AGPL,
LGPL and other copyleft, read from its ``external_tools`` entry in
``tools/third_party_licenses.json``) may be loaded only where its zone allows:

- shipped code (``src/wavebit/``), examples and tutorial notebooks, and the always-loaded
  files (``tools/``, ``docs/**/*.py``, every ``conftest.py``, every other test helper):
  never;
- test modules (``tests/**/test_*.py``): only inside a function, fixture or test. A
  module-level import, also inside a module-level ``try`` or ``if``, and a module-level
  ``importorskip`` fail. One exception: a module marked ``research`` as a whole, which must
  also run under ``python -m unittest`` alone, may guard a module-level import with
  ``try``/``except ImportError`` or an ``if`` -- never with ``importorskip``;
- generators and workers (``tests/references/{gen,refine,check}_*.py``,
  ``tests/references/interface/``) and studies (``studies/``): at any level.

A file that loads such a tool must match a glob in that tool's ``reference.imported_by``,
and a test or helper may import a generator, worker or study module at module level only if
that module has no module-level load of such a tool (the body of
``if __name__ == '__main__':`` does not count). The Mode cells of
``docs/development/third-party.md`` and its Korean page must equal each tool's
``reference.mode``.

The check reads source only. ``tools/check_licenses.py`` keeps its own scan of what the
package imports; this file does not change it.
"""

import ast
import fnmatch
import json
import os
import pathlib
import re
import subprocess

import pytest

pytest.importorskip('packaging')

from tools import check_licenses as checker  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
INVENTORY = json.loads(checker.INVENTORY.read_text(encoding='utf-8'))
MODES = ('live', 'stored', 'lane', 'local', 'past', 'lab')
FORBIDDEN_ZONES = {'shipped': 'shipped code', 'examples': 'an example or tutorial',
                   'always-loaded': 'an always-loaded file'}
DYNAMIC = {'importorskip', 'import_module', '__import__'}


def zone(path):
    """The zone of a repository-relative POSIX path, or None when it is in none."""
    parts = path.split('/')
    name = parts[-1]
    if path.endswith('.ipynb'):
        return 'examples' if parts[0] == 'tutorials' else None
    if not path.endswith('.py'):
        return None
    if parts[:2] == ['src', 'wavebit']:
        return 'shipped'
    if parts[0] == 'examples':
        return 'examples'
    if name == 'conftest.py' or parts[0] == 'tools' or parts[0] == 'docs':
        return 'always-loaded'
    if parts[0] == 'studies':
        return 'study'
    if parts[0] == 'tests':
        if name.startswith('test_'):
            return 'test module'
        if parts[:2] == ['tests', 'references'] and (
                parts[2:3] == ['interface'] or
                (len(parts) == 3 and re.match(r'(gen|refine|check)_', name))):
            return 'generator'
        return 'always-loaded'
    return None


def notebook_source(text):
    """A notebook's code cells as one Python source; IPython magics and shell lines blanked."""
    cells = json.loads(text).get('cells', [])
    lines = []
    for cell in cells:
        if cell.get('cell_type') != 'code':
            continue
        source = cell.get('source', '')
        source = ''.join(source) if isinstance(source, list) else source
        for line in source.splitlines():
            lines.append('' if line.startswith(('%', '!')) else line)
        lines.append('')
    return '\n'.join(lines)


def _is_main_guard(node):
    test = node.test
    return (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) and
            test.left.id == '__name__' and len(test.comparators) == 1 and
            isinstance(test.comparators[0], ast.Constant) and test.comparators[0].value == '__main__')


def _catches_import_error(node):
    for handler in node.handlers:
        names = []
        if handler.type is None:
            return True
        for item in (handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]):
            names.append(item.attr if isinstance(item, ast.Attribute) else getattr(item, 'id', ''))
        if {'ImportError', 'ModuleNotFoundError', 'Exception', 'BaseException'} & set(names):
            return True
    return False


def _call_name(func):
    return func.attr if isinstance(func, ast.Attribute) else getattr(func, 'id', None)


class _Loads(ast.NodeVisitor):
    """Every import of a module: its dotted name, line, level and how it was written.

    level is 'module' (runs on import of the file, module-level try and if included),
    'main' (inside ``if __name__ == '__main__':``) or 'lazy' (inside a function or lambda).
    guarded is true at module level inside a try that catches ImportError, or inside an if.
    """

    def __init__(self):
        self.loads = []
        self.level, self.guarded = 'module', False
        self.research = False

    def _push(self, level=None, guarded=None):
        saved = self.level, self.guarded
        if level is not None:
            self.level = level
        if guarded is not None:
            self.guarded = guarded
        return saved

    def _visit_body(self, nodes, level=None, guarded=None):
        saved = self._push(level, guarded)
        for node in nodes:
            self.visit(node)
        self.level, self.guarded = saved

    def visit_FunctionDef(self, node):
        for item in node.decorator_list + node.args.defaults + node.args.kw_defaults:
            if item is not None:
                self.visit(item)
        self._visit_body(node.body, level='lazy' if self.level == 'module' else self.level)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        self._visit_body([node.body], level='lazy' if self.level == 'module' else self.level)

    def visit_If(self, node):
        self.visit(node.test)
        if self.level == 'module' and _is_main_guard(node):
            self._visit_body(node.body, level='main')
            self._visit_body(node.orelse)
            return
        guarded = True if self.level == 'module' else None
        self._visit_body(node.body, guarded=guarded)
        self._visit_body(node.orelse, guarded=guarded)

    def visit_Try(self, node):
        guarded = True if self.level == 'module' and _catches_import_error(node) else None
        self._visit_body(node.body, guarded=guarded)
        for handler in node.handlers:
            self._visit_body(handler.body)
        self._visit_body(node.orelse)
        self._visit_body(node.finalbody)

    visit_TryStar = visit_Try

    def visit_Assign(self, node):
        if (self.level == 'module' and any(getattr(t, 'id', None) == 'pytestmark' for t in node.targets)
                and 'research' in ast.dump(node.value)):
            self.research = True
        self.generic_visit(node)

    def _add(self, name, node, how, relative=0, names=()):
        self.loads.append(dict(name=name, line=node.lineno, level=self.level, guarded=self.guarded,
                               how=how, relative=relative, names=tuple(names)))

    def visit_Import(self, node):
        for alias in node.names:
            self._add(alias.name, node, 'import')

    def visit_ImportFrom(self, node):
        self._add(node.module or '', node, 'import', node.level, [a.name for a in node.names])

    def visit_Call(self, node):
        name = _call_name(node.func)
        if (name in DYNAMIC and node.args and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            self._add(node.args[0].value, node, name)
        self.generic_visit(node)


def loads(source):
    """(loads, research) for one Python source."""
    visitor = _Loads()
    visitor.visit(ast.parse(source))
    return visitor.loads, visitor.research


def tools(inventory):
    """{import name: tool} for every external tool whose license is not permissive."""
    permissive = set(inventory['policy']['permissive'])
    found = {}
    for tool, entry in inventory['external_tools'].items():
        if checker.satisfies(checker.parse_expression(entry['spdx']), permissive):
            continue
        for module in entry.get('forbidden_imports', []):
            found[module] = tool
    return found


def _resolve(name, relative, names, path, files):
    """Repository files a module-level import may load: from the root or the file's folder."""
    folder = pathlib.PurePosixPath(path).parent
    bases = []
    if relative:
        base = folder
        for _ in range(relative - 1):
            base = base.parent
        bases.append(base)
    else:
        bases += [pathlib.PurePosixPath('.'), folder]
    found = []
    for base in bases:
        stem = base.joinpath(*name.split('.')) if name else base
        candidates = [f'{stem}.py', f'{stem}/__init__.py']
        candidates += [f'{stem}/{n}.py' for n in names]
        found += [c.removeprefix('./') for c in candidates if c.removeprefix('./') in files]
    return found


def zone_errors(sources, inventory):
    """Every zone rule broken by {path: source}; a source that is not Python is a notebook."""
    modules = tools(inventory)
    entries = inventory['external_tools']
    errors, scanned, loaders = [], {}, {}
    for path, text in sorted(sources.items()):
        where = zone(path)
        if where is None:
            if path.endswith('.py'):
                errors.append(f'{path}: in no zone; add the path to a zone in tests/api/test_reference_zones.py')
            continue
        source = notebook_source(text) if path.endswith('.ipynb') else text
        try:
            found, research = loads(source)
        except SyntaxError as error:
            errors.append(f'{path}: cannot be parsed ({error.msg}, line {error.lineno})')
            continue
        scanned[path] = (where, found, research)
        for load in found:
            top = load['name'].split('.')[0]
            if load['relative'] or top not in modules:
                continue
            tool = modules[top]
            loaders.setdefault(tool, set()).add(path)
            at = f"{path}:{load['line']}: {tool} ({top})"
            if where in FORBIDDEN_ZONES:
                errors.append(f'{at} is loaded in {FORBIDDEN_ZONES[where]}; run it as a separate program, or move '
                              'the code to a test module, generator or study')
                continue
            if load['level'] == 'module' and load['how'] == 'importorskip':
                errors.append(f'{at}: a module-level importorskip skips the whole file; '
                              'call it inside the test')
            elif where == 'test module' and load['level'] == 'module':
                if not (research and load['guarded'] and load['how'] == 'import'):
                    errors.append(f'{at} is loaded at module level; load it inside the test or fixture')
            listed = entries[tool].get('reference', {}).get('imported_by', [])
            if not any(fnmatch.fnmatchcase(path, glob) for glob in listed):
                errors.append(f'{at}: the file is not in reference.imported_by of {tool}')
    for path, (where, found, research) in sorted(scanned.items()):
        if (where not in FORBIDDEN_ZONES and where != 'test module') or research:
            continue
        for load in found:
            if load['level'] != 'module' or load['how'] != 'import':
                continue
            for target in _resolve(load['name'], load['relative'], load['names'], path, scanned):
                target_zone, target_loads, _ = scanned[target]
                if target_zone not in ('generator', 'study'):
                    continue
                for inner in target_loads:
                    top = inner['name'].split('.')[0]
                    if inner['level'] == 'module' and not inner['relative'] and top in modules:
                        errors.append(f"{path}:{load['line']} imports {target} at module level, which loads "
                                      f"{modules[top]} ({top}) at line {inner['line']}; move that import "
                                      "into a function or under if __name__ == '__main__':")
    for tool, entry in entries.items():
        reference = entry.get('reference')
        if not isinstance(reference, dict):
            errors.append(f'{tool}: no reference block in the inventory')
            continue
        mode = reference.get('mode')
        if not mode or not isinstance(mode, list) or set(mode) - set(MODES) or len(set(mode)) != len(mode):
            errors.append(f'{tool}: reference.mode must be a list of distinct modes from {MODES}')
        for glob in reference.get('imported_by', []):
            if not any(fnmatch.fnmatchcase(p, glob) for p in loaders.get(tool, ())):
                errors.append(f'{tool}: reference.imported_by {glob!r} matches no file that loads it')
        if mode == ['past'] and loaders.get(tool):
            errors.append(f'{tool}: mode past, but {sorted(loaders[tool])} load it')
    return errors


def documented_modes(text, inventory):
    """{tool: [set of modes per row]} from the reference-tool tables of one docs page."""
    names = set(inventory['external_tools'])
    found, mode_column, past = {}, None, False
    for line in text.splitlines():
        if not line.startswith('|'):
            mode_column, past = None, False
            continue
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if cells[0] in ('Tool', '도구'):
            mode_column = next((i for i, c in enumerate(cells) if c in ('Mode', '방식')), None)
            past = cells[1:2] in (['Version'], ['버전'])
            continue
        name = re.fullmatch(r'`([^`]+)`', cells[0])
        if not name or name.group(1) not in names or (mode_column is None and not past):
            continue
        modes = {'past'} if past else set(re.findall(r'\b(' + '|'.join(MODES) + r')\b', cells[mode_column]))
        found.setdefault(name.group(1), []).append(modes)
    return found


def mode_errors(docs, inventory):
    errors = []
    for doc, text in docs.items():
        found = documented_modes(text, inventory)
        for tool, entry in inventory['external_tools'].items():
            want = set(entry.get('reference', {}).get('mode', []))
            rows = found.get(tool, [])
            if len(rows) != 1:
                errors.append(f'{doc}: {tool} has {len(rows)} rows in the reference-tool tables, not one')
            elif rows[0] != want:
                errors.append(f'{doc}: the Mode cell of {tool} says {sorted(rows[0])}, the inventory '
                              f'{sorted(want)}')
    return errors


def _tracked(root):
    listed = subprocess.run(['git', 'ls-files', '-z'], cwd=root, capture_output=True, check=True,
                            env={k: v for k, v in os.environ.items() if not k.startswith('GIT_')}).stdout
    return [n for n in os.fsdecode(listed).split('\0')
            if n and n.endswith(('.py', '.ipynb')) and (root / n).is_file()]


def _repository_sources(root):
    try:
        top = subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=root, capture_output=True, text=True)
    except OSError:
        pytest.skip('git is not available, so the tracked files are unknown')
    if top.returncode != 0 or not pathlib.Path(top.stdout.rstrip('\n')).samefile(root):
        pytest.skip('not a git checkout, so the tracked files are unknown')
    return {p: (root / p).read_text(encoding='utf-8') for p in _tracked(root)}


def test_the_repository_loads_reference_tools_only_where_its_zone_allows():
    errors = zone_errors(_repository_sources(ROOT), INVENTORY)
    assert errors == [], '\n'.join(errors)


def test_the_docs_mode_cells_match_the_inventory():
    docs = {doc.relative_to(ROOT).as_posix(): doc.read_text(encoding='utf-8') for doc in checker.DOCS}
    errors = mode_errors(docs, INVENTORY)
    assert errors == [], '\n'.join(errors)


# Each case below changes one file of a small tree and names what the check must say.

GPL = {'_comment': '', 'policy': {'permissive': ['MIT']}, 'external_tools': {
    'gpltool': {'spdx': 'GPL-2.0-or-later', 'forbidden_imports': ['gpltool'],
                'reference': {'mode': ['stored', 'local'],
                              'imported_by': ['tests/x/test_listed.py', 'tests/references/gen_*.py',
                                              'studies/**', 'tests/rcwa/test_research*.py']}},
    'mittool': {'spdx': 'MIT', 'forbidden_imports': ['mittool'], 'reference': {'mode': ['stored']}}}}
LAZY = 'def test_it():\n    import gpltool\n'
GUARDED = 'try:\n    import gpltool\nexcept ImportError:\n    gpltool = None\n'
RESEARCH = ('import pytest\ntry:\n    import pytest\n    pytestmark = pytest.mark.research\n'
            'except ImportError:\n    pass\n')
BASE = {'tests/x/test_listed.py': LAZY, 'tests/references/gen_goldens_a.py': 'import gpltool\n',
        'studies/s/s.py': 'import gpltool\n', 'tests/rcwa/test_research.py': RESEARCH + GUARDED}


def _errors(changes=None):
    return zone_errors(dict(BASE, **(changes or {})), GPL)


def _one(errors, phrase):
    assert len(errors) == 1 and phrase in errors[0], errors


def test_a_small_tree_passes():
    assert _errors() == []


@pytest.mark.parametrize('path', [
    'src/wavebit/m.py',                    # shipped
    'examples/e.py',                       # examples
    'tools/t.py',                          # always loaded
    'docs/conf.py',
    'tests/fdtd/conftest.py',
    'tests/references/_cases.py',          # a test helper
])
def test_a_tool_load_in_a_never_zone_fails_also_when_lazy(path):
    for text in ('import gpltool\n', LAZY, 'def f():\n    __import__("gpltool")\n'):
        _one(zone_errors(dict(BASE, **{path: text}), GPL), 'is loaded in ')


def test_a_tool_load_in_a_tutorial_notebook_cell_fails():
    cell = {'cells': [{'cell_type': 'markdown', 'source': ['import gpltool']},
                      {'cell_type': 'code', 'source': ['%matplotlib inline\n', 'import gpltool.sub\n']}]}
    _one(zone_errors(dict(BASE, **{'tutorials/rcwa/n.ipynb': json.dumps(cell)}), GPL), 'an example or tutorial')


@pytest.mark.parametrize('text', [
    'import gpltool\n',
    GUARDED,
    'if True:\n    from gpltool import x\n',
    'class T:\n    import gpltool\n',
])
def test_a_module_level_load_in_a_test_module_fails(text):
    _one(_errors({'tests/x/test_listed.py': text}), 'at module level')


def test_a_module_level_importorskip_fails():
    _one(_errors({'tests/x/test_listed.py': 'import pytest\ng = pytest.importorskip("gpltool")\n'}),
         'module-level importorskip')


@pytest.mark.parametrize('text', [LAZY, 'import importlib\ndef f():\n    importlib.import_module("gpltool")\n'])
def test_a_lazy_load_in_an_unlisted_file_fails(text):
    _one(_errors({'tests/y/test_unlisted.py': text}), 'not in reference.imported_by')


def test_a_lazy_load_in_a_listed_file_passes():
    assert _errors({'tests/x/test_listed.py': 'def test_it(pytest):\n    pytest.importorskip("gpltool")\n'}) == []


def test_a_test_importing_a_generator_with_a_module_level_tool_load_fails():
    errors = _errors({'tests/references/test_g.py': 'from gen_goldens_a import x\n'})
    _one(errors, 'imports tests/references/gen_goldens_a.py at module level')
    errors = _errors({'tests/a/test_s.py': 'from studies.s.s import f\n'})
    _one(errors, 'imports studies/s/s.py at module level')


def test_a_main_block_load_in_an_imported_generator_passes():
    gen = 'def f():\n    pass\nif __name__ == "__main__":\n    import gpltool\n'
    assert _errors({'tests/references/gen_goldens_a.py': gen,
                    'tests/references/test_g.py': 'from gen_goldens_a import f\n'}) == []


def test_an_unclassified_python_file_fails():
    _one(_errors({'setup.py': 'x = 1\n'}), 'in no zone')


def test_a_permissive_tool_may_load_anywhere():
    assert _errors({'src/wavebit/m.py': 'import mittool\n'}) == []


def test_a_research_file_may_guard_a_module_level_import():
    assert _errors({'tests/rcwa/test_research.py': RESEARCH + GUARDED}) == []


def test_a_research_file_still_may_not_importorskip_at_module_level():
    text = RESEARCH + 'gpltool = pytest.importorskip("gpltool")\n'
    _one(_errors({'tests/rcwa/test_research.py': text}), 'module-level importorskip')


def test_the_guard_outside_a_research_file_fails():
    _one(_errors({'tests/rcwa/test_research.py': GUARDED}), 'at module level')


def test_a_stale_listing_and_a_past_tool_that_is_loaded_fail():
    inventory = json.loads(json.dumps(GPL))
    inventory['external_tools']['gpltool']['reference']['imported_by'].append('tests/gone/*.py')
    _one(zone_errors(BASE, inventory), "'tests/gone/*.py' matches no file")
    inventory = json.loads(json.dumps(GPL))
    inventory['external_tools']['gpltool']['reference']['mode'] = ['past']
    assert any('mode past' in e for e in zone_errors(BASE, inventory))
    inventory['external_tools']['gpltool']['reference']['mode'] = ['someday']
    assert any('reference.mode must be' in e for e in zone_errors(BASE, inventory))


def test_untracked_files_are_not_scanned(tmp_path):
    def git(*args):
        subprocess.run(['git', *args], cwd=tmp_path, check=True, capture_output=True,
                       env={k: v for k, v in os.environ.items() if not k.startswith('GIT_')})
    git('init', '-q')
    (tmp_path/'tests'/'x').mkdir(parents=True)
    (tmp_path/'tests'/'x'/'test_listed.py').write_text(LAZY, encoding='utf-8')
    git('add', '.')
    (tmp_path/'tests'/'.venv'/'lib').mkdir(parents=True)
    (tmp_path/'tests'/'.venv'/'lib'/'gpl.py').write_text('import gpltool\n', encoding='utf-8')
    assert _tracked(tmp_path) == ['tests/x/test_listed.py']


def test_the_mode_cell_is_compared():
    table = ('| Tool | SPDX | Mode | How it is used |\n| --- | --- | --- | --- |\n'
             '| `gpltool` | GPL-2.0-or-later | stored (3 goldens); local (11 nodes) | x |\n\n'
             '| Tool | Version | SPDX | How it was used |\n| --- | --- | --- | --- |\n'
             '| `mittool` | 1.0 | MIT | x |\n')
    inventory = json.loads(json.dumps(GPL))
    inventory['external_tools']['mittool']['reference']['mode'] = ['past']
    assert mode_errors({'d.md': table}, inventory) == []
    assert mode_errors({'d.md': table.replace('local (11 nodes)', 'lab')}, inventory) == [
        "d.md: the Mode cell of gpltool says ['lab', 'stored'], the inventory ['local', 'stored']"]
    inventory['external_tools']['mittool']['reference']['mode'] = ['stored']
    _one(mode_errors({'d.md': table}, inventory), 'mittool says')
