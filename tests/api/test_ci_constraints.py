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
LINE = re.compile(r'^([a-z0-9]+(?:-[a-z0-9]+)*)==([0-9][0-9A-Za-z.!]*)$')


def recorded(path):
    return [line for line in path.read_text(encoding='utf-8').splitlines() if line and not line.startswith('#')]


@pytest.mark.parametrize('name', sorted({target for target, _ in ci_constraints.ENVIRONMENTS.values()}))
def test_each_recorded_file_is_sorted_exact_canonical_pins(name):
    lines = recorded(ci_constraints.CONSTRAINTS / f'{name}.txt')
    assert lines, f'{name}.txt records nothing'
    names = []
    for line in lines:
        match = LINE.match(line)
        assert match, f'{name}.txt: {line!r} is not `canonical-name==version` without a local label'
        names.append(match.group(1))
    assert names == sorted(set(names)), f'{name}.txt is not sorted, or names a distribution twice'
    assert not set(names) & ci_constraints.UNRECORDED


def test_the_numerical_stack_is_recorded_where_the_suite_runs():
    tests = {line.split('==')[0] for line in recorded(ci_constraints.CONSTRAINTS / 'tests.txt')}
    assert {'numpy', 'scipy', 'array-api-compat', 'jax', 'jaxlib', 'torch', 'setuptools'} <= tests


def test_every_set_records_the_build_backend():
    for name in {target for target, _ in ci_constraints.ENVIRONMENTS.values()}:
        names = {line.split('==')[0] for line in recorded(ci_constraints.CONSTRAINTS / f'{name}.txt')}
        assert 'setuptools' in names, f'{name}.txt leaves the isolated build unpinned'


def test_required_workflows_install_only_through_the_tool():
    """Every pip install in the required and deploy workflows goes through the tool,
    except ruff, which its step pins itself."""
    for workflow in ('tests.yml', 'docs-deploy.yml'):
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
    assert ci_constraints.installed() == {'torch': '2.14.1', 'markupsafe': '3.0.3', 'jaraco-classes': '3.4.0'}
    text = ci_constraints.render('tests', ci_constraints.installed())
    assert recorded_text(text) == ['jaraco-classes==3.4.0', 'markupsafe==3.0.3', 'torch==2.14.1']


def recorded_text(text):
    return [line for line in text.splitlines() if line and not line.startswith('#')]
