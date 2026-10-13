"""Separate source-sheet movement from boundary clearance near a diffraction cutoff."""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
import argparse
import json

import numpy as np

from studies.fdtd.rayleigh.fdtd_rayleigh import reference, run


CASES = {
    'fixed-near': dict(source_distance=1., pad_scale=1.),
    'fixed-far': dict(source_distance=1., pad_scale=2.),
    'source-only': dict(source_distance=1.5, pad_scale=1.),
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['all', *CASES], default='all')
    parser.add_argument('--backend', choices=['jax', 'numpy'], default='jax')
    args = parser.parse_args()
    if args.backend == 'jax':
        import jax
        jax.config.update('jax_enable_x64', True)
    expected = reference(32)
    measured = {}
    for name, settings in CASES.items():
        if args.case not in ('all', name):
            continue
        result = run(expected, args.backend, resolution=40, periods=800, width=60.,
                     distance=.2, pml_width=.6, pml_reflection=1e-40, method='directional', **settings)
        print(json.dumps(dict(case=name, **result), allow_nan=False), flush=True)
        measured[name] = {pol: np.array(result['amplitude_real'][pol])+1j*np.array(result['amplitude_imag'][pol])
                          for pol in expected}
    if args.case == 'all':
        for name in ('fixed-far', 'source-only'):
            print(json.dumps(dict(case=name, compared_with='fixed-near',
                                  amplitude_change={pol: float(np.max(abs(measured[name][pol]-measured['fixed-near'][pol])))
                                                    for pol in expected}), allow_nan=False), flush=True)
