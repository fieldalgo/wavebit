"""Fast installation metadata/data checks; wheel isolation has its own smoke tool."""
import importlib.metadata
import importlib.resources
import re
from pathlib import Path
import tomllib

import wavebit


def test_installed_metadata_matches_the_checkout_distribution_contract():
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / 'pyproject.toml').read_text())['project']
    distribution = importlib.metadata.distribution('wavebit')
    assert distribution.version == project['version']
    assert distribution.metadata['Name'] == project['name']
    assert distribution.metadata['License-Expression'] == project['license']
    requires = {value.replace(' ', '') for value in distribution.requires or ()}
    assert {value.replace(' ', '') for value in project['dependencies']} <= requires
    package = importlib.resources.files('wavebit')
    assert package.joinpath('data/filmetrics/Si.txt').is_file()


def test_every_version_the_package_states_is_the_pyproject_one():
    root = Path(__file__).resolve().parents[2]
    version = tomllib.loads((root / 'pyproject.toml').read_text())['project']['version']
    assert wavebit.__version__ == version
    # The citation files name the release a citation points at; a release that bumps the
    # package and not them would have Zenodo archive it under the previous version.
    assert re.findall(r'^version: (\S+)$', (root / 'CITATION.cff').read_text(encoding='utf-8'),
                      re.MULTILINE) == [version]
    for name in ('CITATION.bib', 'README.md'):
        text = (root / name).read_text(encoding='utf-8')
        assert re.findall(r'^\s*version\s*=\s*\{([^}]*)\}', text, re.MULTILINE) == [version], name



def test_retired_namespace_is_not_distributed_or_importable():
    import importlib.util
    assert importlib.util.find_spec('wavebit.compat') is None
    assert not importlib.resources.files('wavebit').joinpath('compat').is_dir()
