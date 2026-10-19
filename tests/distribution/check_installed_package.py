"""Base-install checks; invoked with isolated Python by run_distribution_smoke.py."""

import importlib.metadata
import importlib.util
from pathlib import Path
import sys

import numpy as np
import wavebit

package = Path(wavebit.__file__).resolve()
expected = Path(importlib.metadata.distribution('wavebit').locate_file(
    'wavebit/__init__.py')).resolve()
assert package == expected and package.is_relative_to(Path(sys.prefix).resolve()), package
for optional in ('jax', 'torch', 'matplotlib'):
    assert importlib.util.find_spec(optional) is None, f'{optional} present in base environment'

asset = 'wavebit/data/refractiveindex/Si.txt'
assert importlib.metadata.distribution('wavebit').locate_file(asset).is_file(), asset
names = wavebit.materials.catalog.available()
assert set(names) == {'Green2008-Li1980/Si', 'Malitson1965/SiO2',
                      'Luke2015/Si3N4', 'Malitson1962/Al2O3'}
for name in names:
    assert np.isfinite(wavebit.materials.catalog.load(name).at(900.).epsilon), name
problem = wavebit.rcwa.Problem(
    units=wavebit.Units(length='nm'),
    stack=wavebit.rcwa.Stack(period=(1000.,1000.), bottom=wavebit.Material(epsilon=2.25),
        layers=(wavebit.rcwa.Layer(thickness=200.,background=wavebit.Material(epsilon=14.44)),)),
    excitation=wavebit.rcwa.Excitation(wavelength=632.8,sources=(wavebit.rcwa.PlaneWave(polarization='p'),)))
result = wavebit.rcwa.run(problem,numerics=wavebit.rcwa.Numerics(orders=(2,0)))
r, t = float(result.R.sum()), float(result.T.sum())
assert 0 <= r <= 1 and 0 <= t <= 1 and abs(r + t - 1) < 1e-9, (r, t)
assert not {'jax', 'torch', 'matplotlib'} & sys.modules.keys()
print(f'Installed package: {package}; RCWA R={r:.6f}, T={t:.6f}; material tables OK')
