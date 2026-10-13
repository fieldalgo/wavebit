"""Fast installation metadata/data checks; wheel isolation has its own smoke tool."""
import importlib.metadata
import importlib.resources
from pathlib import Path
import tomllib


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
    for asset in ('data/filmetrics/Si.txt', 'data/matlab/p_Si.mat'):
        assert package.joinpath(asset).is_file(), asset
