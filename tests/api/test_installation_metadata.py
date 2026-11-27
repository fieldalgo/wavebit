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
    from packaging.requirements import Requirement
    jax_requirements = {req.name: req for value in distribution.requires or ()
                        if (req := Requirement(value)).name in ('jax', 'jaxlib')}
    assert set(jax_requirements) == {'jax', 'jaxlib'}
    for requirement in jax_requirements.values():
        assert '0.11.0' not in requirement.specifier
        assert '0.11.1' in requirement.specifier


def test_every_version_the_package_states_is_the_pyproject_one():
    root = Path(__file__).resolve().parents[2]
    version = tomllib.loads((root / 'pyproject.toml').read_text(encoding='utf-8'))['project']['version']
    assert wavebit.__version__ == version



def test_retired_namespace_is_not_distributed_or_importable():
    import importlib.util
    assert importlib.util.find_spec('wavebit.compat') is None
    assert not importlib.resources.files('wavebit').joinpath('compat').is_dir()


@pytest.mark.parametrize('package', ['jax', 'jaxlib'])
@pytest.mark.parametrize('version', ['0.10.2', '0.11.0', '0.11.0+local',
                                   '0.11.1rc1', '0.11.1.dev20261001'])
def test_an_unsafe_jax_runtime_is_reported_before_preparation_or_device_startup(
        monkeypatch, package, version):
    """An existing/source environment must not reach the native deadlock through prepare."""
    jax = pytest.importorskip('jax')
    jaxlib = pytest.importorskip('jaxlib')
    from wavebit import Execution, Material, Units, rcwa
    from wavebit.errors import UnsupportedFeatureError

    monkeypatch.setattr({'jax': jax, 'jaxlib': jaxlib}[package], '__version__', version)

    def unexpected_devices(*args, **kwargs):
        pytest.fail('the unsupported runtime reached device initialization')

    monkeypatch.setattr(jax, 'devices', unexpected_devices)
    problem = rcwa.Problem(units=Units(length='um'),
        stack=rcwa.Stack(period=(.7, .7), top=Material(1.), bottom=Material(2.25), layers=()),
        excitation=rcwa.Excitation(wavelength=.8, sources=(rcwa.PlaneWave(),)))
    report = rcwa.check(problem, numerics=rcwa.Numerics(), backend='jax')
    assert not report.ok
    assert report.issues[0].code == 'execution.jax_version'
    for action in (report.raise_if_invalid,
                   lambda: rcwa.prepare(problem, numerics=rcwa.Numerics(), backend='jax')):
        with pytest.raises(UnsupportedFeatureError, match='Upgrade both') as caught:
            action()
        assert caught.value.code == 'execution.jax_version'
        assert caught.value.issue.actual[package] == version
    # A stale optional dependency must not disable another backend.
    assert Execution(backend='numpy').validate().backend == 'numpy'
