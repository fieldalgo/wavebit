"""Verify the installed distribution outside the source checkout, without extras.

Run with the environment's Python and ``-I`` so the checkout, user site and
PYTHONPATH cannot satisfy missing wheel files. This is an installation check;
it does not certify the experimental auxiliary solver's optical accuracy.
"""

from importlib import metadata
from pathlib import Path
import sys

import numpy as np
import wavebit


def main():
    checkout = Path(__file__).resolve().parents[2]
    installed = Path(wavebit.__file__).resolve()
    assert not installed.is_relative_to(checkout), installed
    distribution = metadata.distribution('wavebit')
    assert distribution.version == wavebit.__version__, (
        distribution.version, wavebit.__version__)
    assert distribution.metadata['License-Expression'] == 'Apache-2.0 AND MIT'
    shipped = distribution.files
    assert shipped is not None, 'distribution has no installed file record'
    for name in ('LICENSE', 'LICENSE.meent', 'LICENSE.fmmax', 'NOTICE'):
        matches = [p for p in shipped if p.name == name and '.dist-info/' in str(p)]
        assert len(matches) == 1, (name, matches)
        installed_text = distribution.locate_file(matches[0]).read_bytes()
        assert installed_text == (checkout / name).read_bytes(), f'installed {name} differs from the checkout'
    declared = set(distribution.metadata.get_all('License-File') or ())
    assert {'LICENSE', 'LICENSE.meent', 'LICENSE.fmmax', 'NOTICE'} <= declared, declared

    asset = 'wavebit/data/refractiveindex/Si.txt'
    assert distribution.locate_file(asset).is_file(), asset
    names = wavebit.materials.catalog.available()
    assert set(names) == {'Green2008-Li1980/Si', 'Malitson1965/SiO2',
                          'Luke2015/Si3N4', 'Malitson1962/Al2O3'}
    for name in names:
        material = wavebit.materials.catalog.load(name)
        assert isinstance(material, wavebit.TabulatedMaterial)
        assert np.isfinite(material.at(900.).epsilon), name

    problem = wavebit.rcwa.Problem(
        units=wavebit.Units(length='nm'),
        stack=wavebit.rcwa.Stack(period=(1000.,1000.), bottom=wavebit.Material(epsilon=2.25),
            layers=(wavebit.rcwa.Layer(thickness=200.,background=wavebit.Material(epsilon=14.44)),)),
        excitation=wavebit.rcwa.Excitation(wavelength=632.8,sources=(wavebit.rcwa.PlaneWave(polarization='p'),)))
    result = wavebit.rcwa.run(problem,numerics=wavebit.rcwa.Numerics(orders=(2,0)))
    reflection, transmission = float(result.R.sum()), float(result.T.sum())
    assert 0 <= reflection <= 1 and 0 <= transmission <= 1, (reflection, transmission)
    assert abs(reflection + transmission - 1) < 1e-9, reflection + transmission
    eager = {m.split('.')[0] for m in sys.modules} & {'jax', 'torch'}
    assert not eager, f'optional backend imported on the numpy path: {eager}'
    print(f'Installed Wavebit {wavebit.__version__}: metadata, licenses, material tables '
          f'and NumPy-only solve passed (R={reflection:.6f}, T={transmission:.6f}).')


if __name__ == '__main__':
    main()
