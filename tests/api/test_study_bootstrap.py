"""Maintained studies select their checkout before importing legacy solver APIs."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
STUDIES = ('negative_index_media', 'smm_gap_waves', 'scene_hash', 'benchmark_backends',
           'smm_field_state')


def _study(name):
    return ROOT / 'studies' / 'rcwa' / name / (name + '.py')


def _stub_tree(root, *, layout='flat', compatibility=False):
    package = root / ('src' if layout == 'src' else '') / 'wavebit'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('MARKER = "selected package"\neps_from_refractive_index = lambda n: n**2\n')
    enums = ('from enum import IntEnum\nclass Backend(IntEnum):\n'
             '    NUMPY=0\n    JAX=1\n    TORCH=2\n'
             'class Formulation:\n    pass\n')
    core = package / '_core'
    core.mkdir()
    (core / '__init__.py').write_text('')
    (core / 'enums.py').write_text(enums)
    if compatibility:
        compat = package / 'compat'
        compat.mkdir()
        (compat / '__init__.py').write_text('')
        (compat / 'legacy.py').write_text('MARKER = "compatibility API"\neps_from_refractive_index = lambda n: n**2\n')
    return package


def _run(name, tmp_path, *, args=(), prefix=None, preload=False, inspect=''):
    # A fresh subprocess is essential: switching sys.path cannot replace an
    # already imported wavebit. Run outside either checkout, with a deliberately
    # competing package on PYTHONPATH when requested.
    code = '''import json, pathlib, runpy, sys
script, arguments, preload, inspect = sys.argv[1:]
sys.argv = [script, *json.loads(arguments)]
if preload == "yes":
    import wavebit
ns = runpy.run_path(script, run_name="_study_bootstrap_test")
exec(inspect, ns)
print(json.dumps({
    "origin": str(pathlib.Path(ns["_selected_wavebit"].__file__).resolve()),
    "legacy": ns["wavebit"] is ns["_legacy_wavebit"],
    "same_package": ns["_legacy_wavebit"] is ns["_selected_wavebit"],
    "marker": getattr(ns["_legacy_wavebit"], "MARKER", None),
    "results": ns.get("RESULTS"),
}))
'''
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH', None)
    if prefix is not None:
        env['PYTHONPATH'] = str(prefix)
    return subprocess.run(
        [sys.executable, '-c', code, str(_study(name)), json.dumps(list(map(str, args))),
         'yes' if preload else 'no', inspect], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=45)


@pytest.mark.parametrize('name', STUDIES)
def test_default_checkout_wins_over_another_package_on_pythonpath(name, tmp_path):
    other = tmp_path / 'unrelated'
    _stub_tree(other)
    result = _run(name, tmp_path, prefix=other)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['origin'] == str(ROOT / 'src/wavebit/__init__.py')
    assert report['legacy'] and not report['same_package']
    if name == 'benchmark_backends':
        assert report['results'] == str(ROOT / 'artifacts/studies/backends.json')


@pytest.mark.parametrize('name', STUDIES)
@pytest.mark.parametrize('layout,compatibility', [('flat', False), ('src', False), ('src', True)])
def test_root_selects_archived_or_current_api_before_import(name, layout, compatibility, tmp_path):
    selected = tmp_path / 'selected'
    package = _stub_tree(selected, layout=layout, compatibility=compatibility)
    # Exercise both argparse spellings used by manual and comparison subprocesses.
    args = ('--root', selected) if layout == 'flat' else (f'--root={selected}',)
    inspect = ''
    if not compatibility and name == 'negative_index_media':
        inspect = 'assert backend_arg("jax") == 1\nassert formulation("first_order") == "4ff"'
    elif not compatibility and name == 'smm_field_state':
        inspect = 'assert formulation("first_order") == "4ff"'
    elif not compatibility and name in ('scene_hash', 'benchmark_backends'):
        inspect = 'assert _backend_arg("torch") == 2'
    result = _run(name, tmp_path, args=args, prefix=ROOT / 'src', inspect=inspect)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['origin'] == str(package / '__init__.py')
    assert report['legacy']
    assert report['same_package'] is not compatibility
    assert report['marker'] == ('compatibility API' if compatibility else 'selected package')


@pytest.mark.parametrize('name', STUDIES)
def test_already_imported_wrong_checkout_is_refused(name, tmp_path):
    other = tmp_path / 'unrelated'
    _stub_tree(other)
    result = _run(name, tmp_path, prefix=other, preload=True)
    assert result.returncode != 0
    assert 'wavebit was already imported from' in result.stderr
    assert 'Run this study in a fresh Python process' in result.stderr


@pytest.mark.parametrize('name', STUDIES)
def test_missing_selected_package_does_not_fall_back_to_an_installed_copy(name, tmp_path):
    result = _run(name, tmp_path, args=('--root', tmp_path / 'absent'), prefix=ROOT / 'src')
    assert result.returncode != 0
    assert 'No wavebit source package in selected root' in result.stderr


def test_negative_index_uses_the_selected_reference_files_without_importing_tests(tmp_path):
    inspect = '''
from pathlib import Path
import sys
import types
# A preloaded unrelated tests package must not select another checkout's oracle.
sys.modules['tests'] = types.ModuleType('tests')
for name in ('_fmmax_dng_cases', '_fmmax_laurent_cases', '_pymoosh_dng_cases'):
    helper = _reference_module(name)
    assert Path(helper.__file__) == _SELECTED_ROOT / 'tests' / 'references' / (name + '.py')
for name in ('golden_fmmax_dng.npz', 'golden_pymoosh_dng.npz'):
    path = _reference_path(name)
    assert path == _SELECTED_ROOT / 'tests' / 'references' / name
    with np.load(path) as values:
        assert values.files
result = solver(path='te').solve(auto_detune=False)
assert np.isfinite(result.R).all()
assert np.isfinite(result.T).all()
'''
    result = _run('negative_index_media', tmp_path, inspect=inspect)
    assert result.returncode == 0, result.stderr


def test_historical_reference_helpers_are_read_from_the_selected_tree(tmp_path):
    selected = tmp_path / 'archive'
    _stub_tree(selected)
    tests = selected / 'tests'
    tests.mkdir()
    (tests / '_oracle.py').write_text('MARKER = "archived oracle"\n')
    result = _run('negative_index_media', tmp_path, args=('--root', selected), inspect='''
assert _reference_module('_oracle').MARKER == 'archived oracle'
assert _reference_path('_oracle.py') == _SELECTED_ROOT / 'tests' / '_oracle.py'
''')
    assert result.returncode == 0, result.stderr


HISTORICAL_REVISION = 'b9957c8bde0f60d3ca47126996cad257fc6e20a6'


def _require_historical_revision():
    available = subprocess.run(
        ['git', '-C', str(ROOT), 'cat-file', '-e', HISTORICAL_REVISION + '^{commit}'],
        capture_output=True).returncode == 0
    if not available:
        pytest.skip('The historical comparison commit is unavailable in this checkout')


@pytest.mark.parametrize('revision', [HISTORICAL_REVISION, 'HEAD'], ids=['historical-flat', 'head-src'])
def test_field_state_comparison_runs_an_actual_archived_package(revision, tmp_path):
    if revision == HISTORICAL_REVISION:
        _require_historical_revision()
    # Run the real comparison child, not a fake package. HEAD can move while the
    # historical commit stays flat; package-path discovery must support both.
    inspect = f"""
rows = other_tree({revision!r}, sections=(6,))
assert list(rows) == ['6 uniform double negative at 0.5-0.3j']
values = list(next(iter(rows.values())).values())
assert len(values) == 2 and np.isfinite(values).all(), rows
"""
    result = _run('smm_field_state', tmp_path, inspect=inspect)
    assert result.returncode == 0, result.stderr


@pytest.fixture
def historical_package(tmp_path):
    _require_historical_revision()
    selected = tmp_path / 'archived'
    selected.mkdir()
    archive = subprocess.run(
        ['git', '-C', str(ROOT), 'archive', HISTORICAL_REVISION, 'wavebit'],
        check=True, capture_output=True).stdout
    subprocess.run(['tar', '-x', '-C', str(selected)], input=archive, check=True)
    return selected


def test_field_state_keeps_the_historical_defect_visible(historical_package, tmp_path):
    # Section 6 is a cheap physical discriminant: the old inverse-based state is
    # wrong for this uniform double negative; selecting the current solver in both
    # columns would falsely erase precisely the defect the study records.
    inspect = """
rows = section_below_axis()
assert np.isfinite(list(rows.values())).all(), rows
if _legacy_wavebit is _selected_wavebit:
    assert rows['SMM vs ETM'] > 1
else:
    assert rows['SMM vs ETM'] < 1e-12
assert rows['ETM vs first-order'] < 1e-12
"""
    for args in ((), ('--root', historical_package)):
        result = _run('smm_field_state', tmp_path, args=args, prefix=ROOT / 'src', inspect=inspect)
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('archived', [False, True], ids=['current-src', 'historical-flat'])
def test_seam_comparison_keeps_the_local_oracle_with_the_selected_solver(
        archived, request, tmp_path):
    selected = request.getfixturevalue('historical_package') if archived else ROOT
    inspect = """
import pathlib
assert pathlib.Path(P.__file__).is_relative_to(_PACKAGE_PARENT)
assert pathlib.Path(O.__file__).is_relative_to(_PACKAGE_PARENT)
assert pathlib.Path(_lattice.__file__).is_relative_to(_PACKAGE_PARENT)
assert pathlib.Path(band_error.__code__.co_filename) == _INSTRUMENT_PATH
assert pathlib.Path(DEFAULT_OUT) == pathlib.Path(ROOT) / 'artifacts/studies/nvf_seam_contrast.log'
transmission = sum_T(K, order=0, shape_nvf_grid_size=32)
stats, _ = band(K, n=32)
assert 0 < transmission < 1 and np.isfinite(stats).all()
assert stats[2] > 0
"""
    result = _run('nvf_seam_contrast', tmp_path, args=('--root', selected),
                  prefix=ROOT / 'src', inspect=inspect)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    expected = selected / ('wavebit' if archived else 'src/wavebit') / '__init__.py'
    assert report['origin'] == str(expected)
