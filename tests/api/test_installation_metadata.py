"""Fast installation metadata/data checks; wheel isolation has its own smoke tool."""
import importlib.metadata
import importlib.resources
import json
from pathlib import Path
import tomllib
from urllib.parse import urlparse
from urllib.request import url2pathname

import pytest

import wavebit


def _built_from(distribution):
    """Where an installed distribution's metadata was built: the directory a ``pip install``
    from a directory names in PEP 610's ``direct_url.json``, otherwise the ``sys.path`` entry it
    sits on (inside the checkout for an ``.egg-info`` built in place)."""
    direct = distribution.read_text('direct_url.json')
    if direct:
        record = json.loads(direct)
        if 'dir_info' in record and record['url'].startswith('file:'):
            return Path(url2pathname(urlparse(record['url']).path)).resolve()
    return Path(distribution.locate_file('')).resolve()


def test_installed_metadata_matches_the_checkout_distribution_contract():
    root = Path(__file__).resolve().parents[2]
    package = importlib.resources.files('wavebit')
    assert package.joinpath('data/refractiveindex/Si.txt').is_file()
    distribution = importlib.metadata.distribution('wavebit')
    built = _built_from(distribution)
    if not built.is_relative_to(root):
        # PYTHONPATH runs (tests/conftest.py accepts them) import this checkout, but the metadata
        # importlib finds then belongs to whatever else is installed; comparing it with this
        # pyproject would check that other copy.
        pytest.skip(f'the installed wavebit metadata was built from {built}, not this checkout')
    project = tomllib.loads((root / 'pyproject.toml').read_text(encoding='utf-8'))['project']
    assert distribution.version == project['version']
    assert distribution.metadata['Name'] == project['name']
    assert distribution.metadata['License-Expression'] == project['license']
    requires = {value.replace(' ', '') for value in distribution.requires or ()}
    assert {value.replace(' ', '') for value in project['dependencies']} <= requires


def test_every_version_the_package_states_is_the_pyproject_one():
    root = Path(__file__).resolve().parents[2]
    version = tomllib.loads((root / 'pyproject.toml').read_text(encoding='utf-8'))['project']['version']
    assert wavebit.__version__ == version



def test_retired_namespace_is_not_distributed_or_importable():
    import importlib.util
    assert importlib.util.find_spec('wavebit.compat') is None
    assert not importlib.resources.files('wavebit').joinpath('compat').is_dir()
