"""Check improved CPML and a shaped pulse against the exact finite-grid limit."""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
import argparse
import json

import numpy as np

from studies.fdtd.lattice_reference.fdtd_lattice_reference import capture_grid, solve_harmonic, solve_cpml_harmonic
from studies.fdtd.pulse_notches.fdtd_pulse_notches import notch_frequency
from studies.fdtd.rayleigh.fdtd_rayleigh import BAND, reference, run


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resolution', type=int, choices=[40, 80], default=40)
    parser.add_argument('--periods', type=int, choices=[800, 1600, 3200], default=800)
    parser.add_argument('--zeros', type=int, choices=[0, 1, 2], default=2)
    parser.add_argument('--backend', choices=['jax', 'numpy'], default='jax')
    args = parser.parse_args()
    if args.backend == 'jax':
        import jax
        jax.config.update('jax_enable_x64', True)
    grid = capture_grid(args.resolution, pml_width=1.2, pml_reflection=1e-160)
    exact, finite = {}, {}
    for pol in ('te', 'tm'):
        exact[pol] = np.array([solve_harmonic(grid, w, pol)['amplitudes'] for w in BAND])
        finite[pol] = np.array([solve_cpml_harmonic(grid, w, pol)['amplitudes'] for w in BAND])
    zeros = [notch_frequency(args.resolution)]*args.zeros if args.zeros else None
    # Long runs retain their complete half-duration prefix. The shorter run
    # compares its final 100 periods, after the pulse is already inactive.
    window = int(np.ceil((args.periods/2 if args.periods >= 1600 else 100)*.96/grid['dt']))
    measured = run(reference(32), args.backend, resolution=args.resolution, periods=args.periods,
                   width=60., distance=.2, pml_width=1.2, pml_reflection=1e-160,
                   method='directional', source_distance=1., pulse_notches=zeros,
                   convergence_window=window)
    errors, prefixes = {}, {}
    for pol in ('te', 'tm'):
        values = np.array(measured['amplitude_real'][pol])+1j*np.array(measured['amplitude_imag'][pol])
        errors[pol] = dict(time=float(np.max(abs(values-finite[pol]))),
                           boundary=float(np.max(abs(finite[pol]-exact[pol]))),
                           time_and_boundary=float(np.max(abs(values-exact[pol]))))
        if measured['prefix_amplitude_real'][pol] is not None:
            earlier = (np.array(measured['prefix_amplitude_real'][pol])
                       +1j*np.array(measured['prefix_amplitude_imag'][pol]))
            prefixes[pol] = float(np.max(abs(earlier-finite[pol])))
    print(json.dumps(dict(**measured, separated_error=errors, prefix_time_error=prefixes), allow_nan=False), flush=True)
