"""Installed third-party licenses match the reviewed inventory.

``tools/check_licenses.py`` holds the policy; ``docs/development/third-party.md``
states it. Here it runs on the test environment's own interpreter, restricted to
what Wavebit's requirements reach, and on small synthetic environments that each
break one rule.
"""

import json

import pytest

pytest.importorskip('packaging')

from tools import check_licenses as checker  # noqa: E402

INVENTORY = json.loads(checker.INVENTORY.read_text(encoding='utf-8'))


def test_this_environment_passes_the_reachable_check():
    errors, report = checker.check(checker.environment_paths(None), INVENTORY, reachable_only=True)
    assert errors == []
    assert report['wavebit']['use'] == 'first-party'
    assert {'numpy', 'scipy', 'array-api-compat'} <= {name for name, r in report.items() if r['use'] == 'runtime'}


def _distribution(root, name, license_expression, requires=(), extras=(), files=()):
    info = root/f'{name.replace("-", "_")}-1.0.dist-info'
    info.mkdir()
    lines = ['Metadata-Version: 2.4', f'Name: {name}', 'Version: 1.0',
             f'License-Expression: {license_expression}']
    lines += [f'Provides-Extra: {extra}' for extra in extras]
    lines += [f'Requires-Dist: {requirement}' for requirement in requires]
    (info/'METADATA').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    # importlib.metadata lists only RECORD entries that exist on disk, so create them.
    for path in files:
        (root/path).parent.mkdir(parents=True, exist_ok=True)
        (root/path).touch()
    record = [f'{path},,' for path in files] + [f'{info.name}/METADATA,,', f'{info.name}/RECORD,,']
    (info/'RECORD').write_text('\n'.join(record)+'\n', encoding='utf-8')


def _inventory(**distributions):
    return dict(INVENTORY, distributions=distributions)


def test_a_gpl_runtime_requirement_fails(tmp_path):
    _distribution(tmp_path, 'wavebit', 'Apache-2.0 AND MIT', requires=['solverlib'])
    _distribution(tmp_path, 'solverlib', 'GPL-3.0-only')
    errors, _ = checker.check([str(tmp_path)], _inventory(solverlib={'spdx': 'GPL-3.0-only'}), check_docs=False)
    assert any('not allowed for runtime use' in error for error in errors)


def test_an_sspl_tool_fails_even_when_reviewed(tmp_path):
    _distribution(tmp_path, 'wavebit', 'Apache-2.0 AND MIT')
    _distribution(tmp_path, 'server', 'SSPL-1.0')
    errors, _ = checker.check([str(tmp_path)], _inventory(server={'spdx': 'SSPL-1.0'}), check_docs=False)
    assert any('never allowed' in error for error in errors)


def test_an_agpl_tool_passes_only_with_its_review(tmp_path):
    _distribution(tmp_path, 'wavebit', 'Apache-2.0 AND MIT')
    _distribution(tmp_path, 'rcwatool', 'AGPL-3.0-or-later')
    errors, _ = checker.check([str(tmp_path)], _inventory(rcwatool={'spdx': 'AGPL-3.0-or-later'}), check_docs=False)
    assert any('needs an agpl_review note' in error for error in errors)
    reviewed = {'spdx': 'AGPL-3.0-or-later', 'agpl_review': 'unmodified upstream, unshipped tests only'}
    errors, report = checker.check([str(tmp_path)], _inventory(rcwatool=reviewed), check_docs=False)
    assert errors == [] and report['rcwatool']['use'] == 'tool'


@pytest.mark.parametrize('requires', ['rcwatool', 'rcwatool; extra == "fast"'])
def test_agpl_fails_as_a_requirement_even_when_reviewed(tmp_path, requires):
    _distribution(tmp_path, 'wavebit', 'Apache-2.0 AND MIT', extras=['fast'], requires=[requires])
    _distribution(tmp_path, 'rcwatool', 'AGPL-3.0-or-later')
    reviewed = {'spdx': 'AGPL-3.0-or-later', 'agpl_review': 'x', 'external_process': {'forbidden_imports': []}}
    errors, _ = checker.check([str(tmp_path)], _inventory(rcwatool=reviewed), check_docs=False)
    assert any('never allowed for' in error and 'also as a separate process' in error for error in errors)


def test_weak_copyleft_in_an_extra_needs_a_review(tmp_path):
    _distribution(tmp_path, 'wavebit', 'Apache-2.0 AND MIT', extras=['jax'],
                  requires=['progress; extra == "jax"'])
    _distribution(tmp_path, 'progress', 'MPL-2.0 AND MIT')
    unreviewed = _inventory(progress={'spdx': 'MPL-2.0 AND MIT'})
    errors, report = checker.check([str(tmp_path)], unreviewed, check_docs=False)
    assert report['progress']['use'] == 'optional'
    assert any('not allowed for optional use' in error for error in errors)
    reviewed = _inventory(progress={'spdx': 'MPL-2.0 AND MIT', 'weak_copyleft_review': 'separate, unmodified'})
    assert checker.check([str(tmp_path)], reviewed, check_docs=False)[0] == []


def test_weak_copyleft_in_the_base_requirements_fails_even_when_reviewed(tmp_path):
    _distribution(tmp_path, 'wavebit', 'Apache-2.0 AND MIT', requires=['progress'])
    _distribution(tmp_path, 'progress', 'MPL-2.0')
    inventory = _inventory(progress={'spdx': 'MPL-2.0', 'weak_copyleft_review': 'separate, unmodified'})
    errors, _ = checker.check([str(tmp_path)], inventory, check_docs=False)
    assert any('not allowed for runtime use' in error for error in errors)


def test_gpl_test_tools_installed_in_a_test_environment_pass(tmp_path):
    # Unshipped tests may run GPL tools in the Wavebit process; only SSPL-like licenses and requirements fail.
    _distribution(tmp_path, 'wavebit', 'Apache-2.0 AND MIT')
    _distribution(tmp_path, 's4', 'GPL-2.0-or-later')  # recorded under external_tools as S4
    _distribution(tmp_path, 'gpltool', 'GPL-3.0-only')
    errors, report = checker.check([str(tmp_path)], _inventory(gpltool={'spdx': 'GPL-3.0-only'}), check_docs=False)
    assert errors == []
    assert report['s4']['use'] == 'external-tool' and report['gpltool']['use'] == 'tool'


def test_an_external_tool_must_not_become_a_requirement(tmp_path):
    _distribution(tmp_path, 'wavebit', 'Apache-2.0 AND MIT', extras=['fast'], requires=['s4; extra == "fast"'])
    _distribution(tmp_path, 's4', 'GPL-2.0-or-later')
    errors, _ = checker.check([str(tmp_path)], _inventory(), check_docs=False)
    assert any('recorded only as an external tool' in error for error in errors)


def test_unreviewed_and_changed_licenses_fail(tmp_path):
    _distribution(tmp_path, 'wavebit', 'Apache-2.0 AND MIT')
    _distribution(tmp_path, 'newtool', 'MIT')
    _distribution(tmp_path, 'oldtool', 'Apache-2.0')
    errors, _ = checker.check([str(tmp_path)], _inventory(oldtool={'spdx': 'MIT'}), check_docs=False)
    assert any(error.startswith('newtool') and 'no reviewed entry' in error for error in errors)
    assert any(error.startswith('oldtool') and 'review the change' in error for error in errors)
    assert checker.check([str(tmp_path)], _inventory(), check_docs=False, reachable_only=True)[0] == []


def test_wavebit_does_not_import_an_external_tool():
    for tool, entry in INVENTORY['external_tools'].items():
        assert checker.first_party_imports(set(entry['forbidden_imports'])) == [], tool
    assert checker.first_party_imports({'numpy'})


def test_an_agpl_external_tool_needs_its_review():
    tools = dict(INVENTORY['external_tools'])
    tools['inkstone'] = {k: v for k, v in tools['inkstone'].items() if k != 'agpl_review'}
    errors, _ = checker.check([], dict(INVENTORY, external_tools=tools), check_docs=False)
    assert any(error.startswith('inkstone') and 'agpl_review' in error for error in errors)
    assert checker.check([], INVENTORY, check_docs=False)[0] == []


def test_an_sspl_external_tool_must_be_marked_lab_only():
    tools = dict(INVENTORY['external_tools'], server={'spdx': 'SSPL-1.0', 'forbidden_imports': []})
    errors, _ = checker.check([], dict(INVENTORY, external_tools=tools), check_docs=False)
    assert any(error.startswith('server') and 'lab_only' in error for error in errors)


def test_notice_names_every_bundled_source():
    notice = (checker.ROOT/'NOTICE').read_text(encoding='utf-8')
    for name, entry in INVENTORY['bundled'].items():
        assert entry['url'] in notice, name
    for history in ('https://doi.org/10.5281/zenodo.14631951', 'https://github.com/invrs-io/fmmax',
                    'https://github.com/zhaonat/Rigorous-Coupled-Wave-Analysis'):
        assert history in notice, history


def test_both_language_pages_name_every_entry():
    errors, _ = checker.check([], INVENTORY)
    assert errors == []


BLAS = 'numlib.libs/libopenblas-abc123.so'
EXTENSIONS = ['numlib/_core.cpython-313-x86_64-linux-gnu.so', 'numlib/_win.cp313-win_amd64.pyd']


def _bundled(**changes):
    return dict({'glob': 'numlib.libs/libopenblas-*.so', 'spdx': 'BSD-3-Clause', 'linkage': 'dynamic',
                 'platforms': ['linux']}, **changes)


def _bundled_check(tmp_path, entry, platform='linux', files=(BLAS,)):
    _distribution(tmp_path, 'wavebit', 'Apache-2.0 AND MIT', requires=['numlib'])
    _distribution(tmp_path, 'numlib', 'MIT', files=['numlib/__init__.py', *files])
    return checker.check([str(tmp_path)], _inventory(numlib=entry), check_docs=False, platform=platform)


def test_a_recorded_bundled_library_passes_and_is_reported(tmp_path):
    errors, report = _bundled_check(tmp_path, {'spdx': 'MIT', 'bundled_files': [_bundled()]})
    assert errors == []
    assert report['numlib']['bundled_files'] == [{'file': BLAS, 'glob': 'numlib.libs/libopenblas-*.so',
                                                  'spdx': 'BSD-3-Clause'}]


def test_an_unrecorded_bundled_library_fails(tmp_path):
    errors, _ = _bundled_check(tmp_path, {'spdx': 'MIT'})
    assert any('unrecorded bundled native library numlib.libs/libopenblas-abc123.so' in e for e in errors)


def test_extension_modules_are_not_bundled_libraries(tmp_path):
    errors, report = _bundled_check(tmp_path, {'spdx': 'MIT'}, files=EXTENSIONS)
    assert errors == [] and 'bundled_files' not in report['numlib']


def test_a_bundled_library_must_be_recorded_for_this_platform(tmp_path):
    errors, _ = _bundled_check(tmp_path, {'spdx': 'MIT', 'bundled_files': [_bundled()]}, platform='windows')
    assert any("recorded for ['linux'], but is installed on windows" in e for e in errors)


@pytest.mark.parametrize('spdx', ['GPL-3.0-or-later', 'GPL-2.0-only'])
def test_plain_gpl_bundled_in_process_fails_even_when_reviewed(tmp_path, spdx):
    errors, _ = _bundled_check(tmp_path, {'spdx': 'MIT', 'bundled_files': [_bundled(spdx=spdx, review='x')]})
    assert any('loads into the Wavebit process' in e for e in errors)


@pytest.mark.parametrize('spdx', ['GPL-3.0-or-later WITH GCC-exception-3.1', 'FTL OR GPL-2.0-or-later'])
def test_gpl_with_a_runtime_exception_or_another_option_passes_when_reviewed(tmp_path, spdx):
    (tmp_path/'unreviewed').mkdir()
    (tmp_path/'reviewed').mkdir()
    unreviewed = {'spdx': 'MIT', 'bundled_files': [_bundled(spdx=spdx)]}
    errors, _ = _bundled_check(tmp_path/'unreviewed', unreviewed)
    assert any('needs a review' in e for e in errors)
    reviewed = {'spdx': 'MIT', 'bundled_files': [_bundled(spdx=spdx, review='reviewed')]}
    assert _bundled_check(tmp_path/'reviewed', reviewed)[0] == []


def test_agpl_is_never_allowed_bundled(tmp_path):
    errors, _ = _bundled_check(tmp_path, {'spdx': 'MIT', 'bundled_files': [_bundled(spdx='AGPL-3.0-only', review='x')]})
    assert any('never allowed, also when bundled' in e for e in errors)


def test_bundled_files_records_are_validated():
    entry = {'spdx': 'MIT', 'bundled_files': [{'glob': 'a/*.so'}, _bundled(glob='b/*.so', linkage='static'),
                                              _bundled(glob='c/*.so', platforms=['solaris'])]}
    errors = checker.bundled_entry_errors('numlib', entry, INVENTORY['policy'])
    assert any('missing' in e for e in errors)
    assert any('describe static code' in e for e in errors)
    assert any('platforms must be' in e for e in errors)


def test_wavebit_itself_must_not_ship_native_files(tmp_path):
    _distribution(tmp_path, 'wavebit', 'Apache-2.0 AND MIT',
                  files=['wavebit/__init__.py', 'wavebit/_fast.cpython-313-x86_64-linux-gnu.so'])
    errors, _ = checker.check([str(tmp_path)], _inventory(), check_docs=False)
    assert any('first-party distribution contains native files' in e for e in errors)


def test_every_bundled_record_in_the_inventory_is_well_formed():
    for name, entry in INVENTORY['distributions'].items():
        assert checker.bundled_entry_errors(name, entry, INVENTORY['policy']) == [], name
