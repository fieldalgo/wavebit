"""Relocation preserves evidence identities and pytest rerun selection."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.conftest import _relocate_cached_nodeids
from tools.repository_paths import (RepositoryPath, relocate_nodeid, repository_path,
                                    repository_root)

ROOT = Path(__file__).resolve().parents[2]


def test_every_preserved_reference_and_record_keeps_its_original_bytes():
    evidence = json.loads((ROOT / 'tools/relocations.json').read_text())['immutable']
    # 7336 until 2026-10-14, when the 6,943 study records no test or live study input reads
    # were archived outside the repository and their ledger rows removed with them; 322 since the
    # same day, when the 71 tests/references rows left: goldens and oracles are regenerated and
    # extended on purpose (bc48a2ad added arrays to golden_fmmax_field.npz), so a byte pin on
    # them fails at every legitimate change. The ledger keeps study records, branding masters,
    # fixtures and tutorial inputs, which are preserved rather than regenerated.
    assert len(evidence) == 322
    for old, row in evidence.items():
        path = ROOT / row['target']
        assert path.is_file(), (old, path)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], old
        assert bool(path.stat().st_mode & 0o100) == (row['git_mode'] == '100755'), old


def test_imported_learning_records_keep_their_original_bytes():
    receipt = json.loads((ROOT / 'docs/development/learning-provenance/import-receipt.json').read_text())
    records = [r for r in receipt['files']
               if r['decision'] == 'historical-evidence-preserved-not-revalidated']
    assert len(records) == 17
    for row in records:
        assert hashlib.sha256((ROOT / row['target']).read_bytes()).hexdigest() == row['sha256']


def test_historical_joins_resolve_after_the_filename_is_known():
    path = repository_root() / 'tests' / 'test_solver_integration.py'
    assert path == ROOT / 'tests/cross_solver/test_solver_integration.py'
    assert path.relative_to(ROOT).as_posix() == 'tests/cross_solver/test_solver_integration.py'
    assert path.parent / 'sibling.py' == ROOT / 'tests/cross_solver/sibling.py'
    assert repository_path('studies/results') / 'new-unrecorded-run.json' == (
        ROOT / 'artifacts/studies/new-unrecorded-run.json')


def test_only_joins_resolve_and_a_derived_path_names_one_file(monkeypatch):
    """A path pathlib derives (``with_name``, ``relative_to``'s argument) keeps its string.

    On Python 3.12 such a path used to resolve underneath while pathlib overwrote its string,
    so ``str(p)`` and ``Path(p)`` named different files, and ``relative_to`` resolved a plain
    old location and passed for a path that is not under it.
    """
    old, new = 'examples/anisotropic_material.py', 'examples/rcwa/anisotropic_material.py'
    assert json.loads((ROOT / 'tools/relocations.json').read_text())['paths'][old] == new
    derived = (repository_root() / 'examples' / 'foo.py').with_name('anisotropic_material.py')
    assert str(Path(derived)) == str(derived) == str(ROOT / old)
    # Joins resolve the same name: `/`, `key / path` and absolute(), which joins the cwd.
    assert repository_root() / 'examples' / 'anisotropic_material.py' == ROOT / new
    assert str(ROOT) / RepositoryPath(old) == ROOT / new
    monkeypatch.chdir(ROOT)
    assert str(RepositoryPath(old).absolute()) == str(ROOT / new)

    moved = repository_path('studies/results') / 'new-unrecorded-run.json'
    assert moved.relative_to(ROOT / 'artifacts/studies') == Path('new-unrecorded-run.json')
    for location in (ROOT / 'studies/results', str(ROOT / 'studies/results')):
        assert not moved.is_relative_to(location)
        with pytest.raises(ValueError):
            moved.relative_to(location)


def test_archived_globs_keep_the_historical_roster_without_duplicate_paths():
    ledger = json.loads((ROOT / 'tools/relocations.json').read_text())['paths']
    old = [p for p in ledger if p.startswith('tests/test_fd_') and p.endswith('.py')]
    assert old
    paths = list((repository_root() / 'tests').glob('test_fd_*.py'))
    assert len(paths) == len(set(paths))
    assert set(paths) == {ROOT / ledger[p] for p in old}


def test_relocation_preserves_parameterized_node_suffixes():
    assert relocate_nodeid('tests/test_solver_integration.py::test_case[top-numpy]') == (
        'tests/cross_solver/test_solver_integration.py::test_case[top-numpy]')


def test_last_failed_and_stepwise_cache_follow_the_moved_test():
    before = 'tests/test_solver_integration.py::test_case[top-numpy]'
    after = relocate_nodeid(before)
    values = {'cache/lastfailed': {before: True}, 'cache/stepwise': before}
    cache = SimpleNamespace(get=lambda key, default: values.get(key, default),
                            set=lambda key, value: values.__setitem__(key, value))
    _relocate_cached_nodeids(SimpleNamespace(cache=cache))
    assert values == {'cache/lastfailed': {after: True}, 'cache/stepwise': after}


def test_maintained_probe_identity_hashes_the_relocated_source_tree():
    """A nonexistent old package directory must not hash as an empty implementation."""
    import importlib
    package = ROOT / 'src/wavebit/fdtd'
    files = sorted(package.rglob('*.py'))
    assert len(files) > 20
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(package).as_posix().encode() + b'\0')
        digest.update(path.read_bytes().replace(b'\r\n', b'\n') + b'\0')
    for family in ('conductive_film', 'single_pole_film', 'debye_film', 'two_debye_film',
                   'debye_sum_film', 'oscillator_sum_film', 'hybrid_film',
                   'conductive_stack', 'debye_stack', 'oscillator_stack'):
        probe = importlib.import_module(f'studies.fdtd.{family}.fdtd_{family}_probe')
        identity = probe.source_identity()
        assert identity['fd_source_sha256'] == digest.hexdigest(), family
        for path, expected in identity.get('reference_dependencies', {}).items():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected


def test_recursive_archived_inventory_preserves_every_old_filename():
    ledger = json.loads((ROOT / 'tools/relocations.json').read_text())['paths']
    expected = {ROOT / new: Path(old).relative_to('studies/results')
                for old, new in ledger.items() if old.startswith('studies/results/')}
    # The 276 records still tracked plus 37 relocated instruments (7256 before the
    # 2026-10-14 archive); an exact count, so a row lost from the roster fails here.
    assert len(expected) == 313
    historical = repository_path('studies/results')
    paths = [p for p in historical.rglob('*') if p.is_file()]
    assert len(paths) == len(set(paths))
    # Untracked new runs may coexist in artifacts; the complete preserved
    # roster must still be present, with its original names for classification.
    found = {p: p.relative_to(historical) for p in paths}
    assert {p: found[p] for p in expected} == expected
    sample = next(p for p in paths if p in expected)
    assert sample.relative_to(repository_root()) == Path(ledger[
        'studies/results/' + expected[sample].as_posix()])
    import pytest
    with pytest.raises(ValueError):
        sample.relative_to(Path(historical))
    with pytest.raises(ValueError):
        sample.relative_to(repository_path('unrelated-directory'))


def test_maintained_support_inventories_include_sources_tests_and_prior_evidence():
    import importlib
    ledger = json.loads((ROOT / 'tools/relocations.json').read_text())['paths']
    expected_tests = {ROOT / new for old, new in ledger.items()
                      if old.startswith('tests/test_fd_') and old.endswith('.py')}
    expected_goldens = set((ROOT / 'tests/references').glob('fd_golden_stage*.npz'))
    assert len(expected_goldens) == 4
    package = ROOT / 'src/wavebit/fdtd'
    digest = hashlib.sha256()
    for path in sorted(package.rglob('*.py')):
        digest.update(path.relative_to(package).as_posix().encode() + b'\0')
        digest.update(path.read_bytes().replace(b'\r\n', b'\n') + b'\0')
    for family in ('state_bytes', 'inactive_lorentz', 'conductive_stack', 'decay_stop',
                   'debye_stack', 'oscillator_stack'):
        support = importlib.import_module(f'studies.fdtd.{family}.fdtd_{family}_support')
        inventory = support.inventories()
        assert set(inventory['tests']) == expected_tests, family
        assert set(inventory['goldens']) == expected_goldens, family
        expected_prior = {ROOT / new for old, new in ledger.items()
                          if old.startswith('studies/results/')
                          and Path(old).relative_to('studies/results').parts[0].startswith(support.PRIOR)}
        assert expected_prior and set(inventory['previous_film_evidence']) == expected_prior, family
        assert support.identity()['fd'] == digest.hexdigest(), family
        helpers = support.helpers()
        assert helpers and all(p.is_file() for p in helpers), family


def test_maintained_watchers_scan_the_current_fdtd_package(monkeypatch):
    """Evaluate the real source selectors without invoking research solvers."""
    import ast
    monkeypatch.chdir(ROOT)
    expected = set((ROOT / 'src/wavebit/fdtd').rglob('*.py'))
    assert len(expected) > 20
    checked = []
    for path in (ROOT / 'studies/fdtd').glob('*/*.py'):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'rglob'):
                continue
            directories = [n.value for n in ast.walk(node.func.value)
                           if isinstance(n, ast.Constant) and isinstance(n.value, str)
                           and n.value.startswith('src/wavebit/fd')]
            if not directories:
                continue
            call = compile(ast.Expression(node), str(path), 'eval')
            selected = {p.resolve() for p in eval(call, {'ROOT': repository_root(), 'Path': Path})}
            assert selected == expected, (path, directories)
            checked.append(path)
    assert len(checked) >= 10


def test_maintained_shape_mutation_paths_resolve_without_executing_mutations():
    import importlib
    for family in ('sphere_range', 'sphere_grad'):
        module = importlib.import_module(f'studies.fdtd.{family}.fdtd_{family}_mutations')
        for name, relative, _before, _after, _test in module.MUTATIONS:
            path = repository_path('src/wavebit/fd', relative)
            assert path.is_file(), (family, name, path)



def test_historical_source_paths_select_the_preserved_engine_not_new_declarations():
    expected = {
        'geometry.py': 'src/wavebit/fdtd/_legacy_geometry.py',
        'sources.py': 'src/wavebit/fdtd/_legacy_sources.py',
        'boundaries.py': 'src/wavebit/fdtd/_legacy_boundaries.py',
        'monitors.py': 'src/wavebit/fdtd/_legacy_monitors.py',
        'materials.py': 'src/wavebit/materials/time_domain.py',
        '_check.py': 'src/wavebit/fdtd/_prepare/report.py',
        '_lower/plan.py': 'src/wavebit/fdtd/_prepare/plan.py',
        '_run.py': 'src/wavebit/fdtd/_execution.py',
    }
    for old, new in expected.items():
        assert repository_path('src/wavebit/fd', old) == ROOT / new
        assert repository_path('wavebit/fd', old) == ROOT / new
    import ast
    path = ROOT / 'studies/fdtd/subpixel_junction/fdtd_subpixel_junction.py'
    declarations = [node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.Assign) and any(
                        isinstance(target, ast.Name) and target.id == 'FILES'
                        for target in node.targets)]
    assert len(declarations) == 1
    for old in ast.literal_eval(declarations[0].value):
        assert repository_path('src/wavebit/fd', old).is_file(), old



def test_historical_directory_globs_keep_nested_campaign_evidence():
    ledger = json.loads((ROOT / 'tools/relocations.json').read_text())
    historical = repository_path('studies/results')
    prefix = 'studies/results/fdtd_conductive_stack_mutations_attempt'
    expected = {ROOT / new for old, new in ledger['directory_prefixes'].items()
                if old.startswith(prefix) and len(Path(old).parts) == 3}
    assert expected
    found = [p for p in historical.glob('fdtd_conductive_stack_mutations_attempt*')
             if p.is_dir()]
    assert set(found) == expected
    for directory in found:
        name = directory.relative_to(historical).as_posix()
        old_prefix = 'studies/results/' + name + '/'
        expected_files = {ROOT / new for old, new in ledger['paths'].items()
                          if old.startswith(old_prefix)}
        assert expected_files and set(p for p in directory.rglob('*') if p.is_file()) == expected_files
