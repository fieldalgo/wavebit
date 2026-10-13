"""Separate CPML strength, thickness and mesh error after narrow-band ringdown."""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
import argparse
import json

from studies.fdtd.rayleigh.fdtd_rayleigh import reference, run

CASES = {
    'baseline': dict(resolution=40, pml_width=.3, pml_reflection=1e-8),
    'width': dict(resolution=40, pml_width=.6, pml_reflection=1e-8),
    'strength': dict(resolution=40, pml_width=.3, pml_reflection=1e-40),
    'combined': dict(resolution=40, pml_width=.6, pml_reflection=1e-40),
    'long': dict(resolution=40, pml_width=.6, pml_reflection=1e-40, periods=1600),
    # Same final solve as long, but compare its entire second half against the
    # 800-period prefix (66511 steps at the measured Courant/grid settings).
    'long-window': dict(resolution=40, pml_width=.6, pml_reflection=1e-40,
                        periods=1600, convergence_window=66511),
    'fine': dict(resolution=80, pml_width=.6, pml_reflection=1e-40),
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
    for name, case in CASES.items():
        if args.case in ('all', name):
            options = dict(method='directional', distance=.2, width=60., periods=800)
            options.update(case)
            print(json.dumps(dict(case=name, **run(expected, args.backend, **options))), flush=True)
