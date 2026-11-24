"""The recorded CI dependency versions, and the workflows' use of them.

Required CI installs exact versions from `.github/constraints/` through
`tools/ci_constraints.py` (docs/development/verification.md, Dependency versions in CI).
These tests hold the parts that would otherwise fail quietly: a workflow that goes back to a
bare `pip install` keeps passing on today's releases and stops being pinned, and a malformed
line in a recorded file is a constraint pip never applies.
"""

import re

import pytest

from tools import ci_constraints

WORKFLOWS = ci_constraints.ROOT / '.github' / 'workflows'
LINE = re.compile(r'^([a-z0-9]+(?:-[a-z0-9]+)*)==([0-9][0-9A-Za-z.!]*(?:\+[0-9A-Za-z.]+)?)$')


def recorded_text(text):
    return [line for line in text.splitlines() if line and not line.startswith('#')]


def recorded(path):
    return recorded_text(path.read_text(encoding='utf-8'))


@pytest.mark.parametrize('name', sorted({target for target, _ in ci_constraints.ENVIRONMENTS.values()}))
def test_each_recorded_file_is_sorted_exact_canonical_pins(name):
    lines = recorded(ci_constraints.CONSTRAINTS / f'{name}.txt')
    assert lines, f'{name}.txt records nothing'
    names = []
    for line in lines:
        match = LINE.match(line)
        assert match, f'{name}.txt: {line!r} is not `canonical-name==version`'
        names.append(match.group(1))
    assert names == sorted(set(names)), f'{name}.txt is not sorted, or names a distribution twice'
    assert not set(names) & ci_constraints.UNRECORDED


def test_the_numerical_stack_is_recorded_where_the_suite_runs():
    tests = dict(line.split('==') for line in recorded(ci_constraints.CONSTRAINTS / 'tests.txt'))
    assert {'numpy', 'scipy', 'array-api-compat', 'jax', 'jaxlib', 'torch', 'setuptools'} <= set(tests)
    # The local label is what keeps PyPI's CUDA build of the same version from matching.
    assert tests['torch'].endswith('+cpu')


def test_every_set_records_the_build_backend():
    for name in {target for target, _ in ci_constraints.ENVIRONMENTS.values()}:
        names = {line.split('==')[0] for line in recorded(ci_constraints.CONSTRAINTS / f'{name}.txt')}
        assert 'setuptools' in names, f'{name}.txt leaves the isolated build unpinned'


def test_required_workflows_install_only_through_the_tool():
    """Every pip install in the required and deploy workflows goes through the tool,
    except ruff, which its step pins itself."""
    for workflow in ('tests.yml', 'docs-deploy.yml', 'python-publish.yml'):
        text = (WORKFLOWS / workflow).read_text(encoding='utf-8')
        bare = [line.strip() for line in text.splitlines()
                if re.search(r'\bpip install\b', line) and not line.lstrip().startswith('#')]
        expected = ['run: pip install ruff==0.16.2'] if workflow == 'tests.yml' else []
        assert bare == expected, (workflow, bare)
        names = re.findall(r'ci_constraints\.py (?:install|env) ([a-z-]+)', text)
        assert names and set(names) <= set(ci_constraints.ENVIRONMENTS), (workflow, names)
        assert '--latest' not in text, f'{workflow} installs unrecorded versions'


def test_the_canary_installs_every_recorded_set_at_the_newest_releases():
    text = (WORKFLOWS / 'latest-deps.yml').read_text(encoding='utf-8')
    for name in ('tests', 'docs', 'distribution'):
        assert f'ci_constraints.py install {name} --latest' in text
        assert f'ci_constraints.py freeze {name} --output' in text


def test_environment_variables_reach_isolated_builds_and_subprocesses():
    env = ci_constraints.pip_environment('distribution-tools')
    assert env['PIP_CONSTRAINT'] == env['PIP_BUILD_CONSTRAINT'] == str(ci_constraints.CONSTRAINTS / 'distribution.txt')


def test_installed_drops_local_labels_and_unrecorded_names(monkeypatch):
    class Dist:
        def __init__(self, name, version):
            self.metadata = {'Name': name}
            self.version = version

    dists = [Dist('torch', '2.14.1+cpu'), Dist('MarkupSafe', '3.0.3'), Dist('jaraco.classes', '3.4.0'),
             Dist('pip', '26.2.1'), Dist('wavebit', '0.0.4')]
    monkeypatch.setattr(ci_constraints.metadata, 'distributions', lambda: dists)
    assert ci_constraints.installed() == {'torch': '2.14.1+cpu', 'markupsafe': '3.0.3', 'jaraco-classes': '3.4.0'}
    text = ci_constraints.render('tests', ci_constraints.installed())
    assert recorded_text(text) == ['jaraco-classes==3.4.0', 'markupsafe==3.0.3', 'torch==2.14.1+cpu']


def test_installed_keeps_the_first_copy_and_skips_nameless_metadata(monkeypatch):
    class Dist:
        def __init__(self, name, version):
            self.metadata = {'Name': name}
            self.version = version

    dists = [Dist(None, '1'), Dist('numpy', '2.5.4'), Dist('numpy', '2.4.0')]
    monkeypatch.setattr(ci_constraints.metadata, 'distributions', lambda: dists)
    assert ci_constraints.installed() == {'numpy': '2.5.4'}


def test_merge_takes_the_union_and_refuses_a_disagreement(tmp_path):
    linux, windows = tmp_path / 'distribution-Linux.txt', tmp_path / 'distribution-Windows.txt'
    linux.write_text('# header\nbuild==1.6.1\nsecretstorage==3.5.0\n', encoding='utf-8')
    windows.write_text('build==1.6.1\ncolorama==0.4.6\n', encoding='utf-8')
    out = tmp_path / 'distribution.txt'
    ci_constraints.merge('distribution', [linux, windows], out)
    assert recorded(out) == ['build==1.6.1', 'colorama==0.4.6', 'secretstorage==3.5.0']
    windows.write_text('build==1.7.0\n', encoding='utf-8')
    with pytest.raises(SystemExit, match='build'):
        ci_constraints.merge('distribution', [linux, windows], out)


def test_a_latest_distribution_install_upgrades_what_the_runner_image_ships():
    """The runners' own Pythons ship some distributions (certifi on macOS, packaging on
    Windows); a plain install keeps them and the three freezes stop agreeing."""
    for name in ('distribution', 'distribution-tools'):
        for arguments in ci_constraints.pip_arguments(name, latest=True):
            assert arguments[:3] == ['--upgrade', '--upgrade-strategy', 'eager'], (name, arguments)
        for arguments in ci_constraints.pip_arguments(name, latest=False):
            assert '--upgrade' not in arguments, (name, arguments)


def test_a_latest_tests_install_leaves_the_cpu_torch_alone():
    latest = ci_constraints.pip_arguments('tests', latest=True)
    assert latest[0] == ci_constraints.CPU_TORCH
    assert not any('--upgrade' in arguments or '--upgrade-strategy' in arguments for arguments in latest)
    recorded_ = ci_constraints.pip_arguments('tests', latest=False)
    assert recorded_[0] == [*ci_constraints.CPU_TORCH, '--extra-index-url', ci_constraints.PYPI]


def test_a_failed_refresh_opens_its_own_issue_which_only_a_refresh_closes():
    text = (WORKFLOWS / 'latest-deps.yml').read_text(encoding='utf-8')
    assert re.search(r'^  REFRESH_FAILED_TITLE: \S', text, re.MULTILINE)
    refresh = text[text.index('\n  refresh:\n'):]
    step = refresh[refresh.index('name: Open, comment on or close the refresh-failure issue'):]
    assert 'if: ${{ !cancelled() }}' in step and 'STATUS: ${{ job.status }}' in step
    assert 'gh issue create --title "$REFRESH_FAILED_TITLE"' in step
    assert 'gh issue close "$existing"' in step
    # The report job, which a daily run also reaches, never touches it.
    report = text[text.index('\n  report:\n'):text.index('\n  refresh:\n')]
    assert 'REFRESH_FAILED_TITLE' not in report
