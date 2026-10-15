"""Fast installation metadata/data checks; wheel isolation has its own smoke tool."""
import importlib.metadata
import importlib.resources
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



def test_retired_namespace_is_not_distributed_or_importable():
    import importlib.util
    assert importlib.util.find_spec('wavebit.compat') is None
    assert not importlib.resources.files('wavebit').joinpath('compat').is_dir()
