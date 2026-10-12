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

names = wavebit.materials.available()
assert set(names) == {'filmetrics/Si', 'filmetrics/SiO2',
                      'filmetrics/Si3N4', 'filmetrics/Al2O3'}
for name in names:
    assert np.isfinite(wavebit.materials.load(name).at(900.).epsilon), name
result = wavebit.rcwa(
    backend='numpy', tetm='tm', top=1., bot=2.25, theta=0., phi=0.,
    truncation_order=2, wavelength=632.8, period=[1000.],
    structure=np.array([[[14.44, 14.44]]]), thickness=[200.],
).solve()
r, t = float(result.R.sum()), float(result.T.sum())
assert 0 <= r <= 1 and 0 <= t <= 1 and abs(r + t - 1) < 1e-9, (r, t)
assert not {'jax', 'torch', 'matplotlib'} & sys.modules.keys()
print(f'Installed package: {package}; RCWA R={r:.6f}, T={t:.6f}; material tables OK')
