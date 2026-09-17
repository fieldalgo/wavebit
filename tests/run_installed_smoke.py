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
    checkout = Path(__file__).resolve().parents[1]
    installed = Path(wavebit.__file__).resolve()
    assert not installed.is_relative_to(checkout), installed
    distribution = metadata.distribution('wavebit')
    assert distribution.version == wavebit.__version__, (
        distribution.version, wavebit.__version__)
    assert distribution.metadata['License-Expression'] == 'Apache-2.0'
    shipped = distribution.files
    assert shipped is not None, 'distribution has no installed file record'
    for name in ('LICENSE', 'LICENSE.meent', 'LICENSE.fmmax'):
        matches = [p for p in shipped if p.name == name and '.dist-info/' in str(p)]
        assert len(matches) == 1, (name, matches)
        assert distribution.locate_file(matches[0]).stat().st_size > 0, name

    tables = wavebit.read_material_table()
    assert {'SI', 'SIO2', 'SI3N4', 'AL2O3', 'P_SI'} <= set(tables), sorted(tables)
    for name, values in tables.items():
        assert values.ndim == 2 and values.shape[1] == 3, (name, values.shape)
        assert np.isfinite(values).all(), name

    result = wavebit.rcwa(
        backend=0, tetm='tm', top=1.0, bot=2.25, theta=0.0, phi=0.0,
        truncation_order=2, wavelength=632.8, period=[1000],
        structure=np.array([[[14.44, 14.44]]]), thickness=[200],
    ).solve()
    reflection, transmission = float(result.R.sum()), float(result.T.sum())
    assert 0 <= reflection <= 1 and 0 <= transmission <= 1, (reflection, transmission)
    assert abs(reflection + transmission - 1) < 1e-9, reflection + transmission
    eager = {m.split('.')[0] for m in sys.modules} & {'jax', 'torch'}
    assert not eager, f'optional backend imported on the numpy path: {eager}'
    print(f'Installed Wavebit {wavebit.__version__}: metadata, licenses, material tables '
          f'and NumPy-only solve passed (R={reflection:.6f}, T={transmission:.6f}).')


if __name__ == '__main__':
    main()
