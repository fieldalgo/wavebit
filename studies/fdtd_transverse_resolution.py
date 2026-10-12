"""Which axis limits near-cutoff accuracy, and what does selective refinement cost?"""
import argparse
import json
import time

import numpy as np

from studies.fdtd_lattice_reference import capture_grid, solve_harmonic, solve_cpml_harmonic
from studies.fdtd_rayleigh import BAND, PERIOD, reference, run


# (z resolution, x resolution, y resolution), all in cells per unit length.
CASES = dict(isotropic40=(40, 40, 40), x80=(40, 80, 40), x160=(40, 160, 40),
             x320=(40, 320, 40), z80=(80, 40, 40), z160=(160, 40, 40),
             xz80=(80, 80, 40), isotropic80=(80, 80, 80))


def harmonic_case(name, expected, cpml=False, fixed_dt=None):
    rz, rx, ry = CASES[name]
    grid = capture_grid(rz, transverse_resolution=(rx, ry), pml_width=1.2, pml_reflection=1e-160)
    if fixed_dt is not None:
        if not np.isfinite(fixed_dt) or not 0 < fixed_dt <= grid['dt']:
            raise ValueError('fixed-dt must be positive and no larger than every selected automatic step.')
        grid = dict(grid, dt=fixed_dt, steps=int(np.ceil(800*.96/fixed_dt)))
    exact, finite, errors, boundary = {}, {}, {}, {}
    for pol in ('te', 'tm'):
        exact[pol] = np.array([solve_harmonic(grid, w, pol)['amplitudes'] for w in BAND])
        errors[pol] = float(np.max(abs(exact[pol]-expected[pol])))
        if cpml:
            finite[pol] = np.array([solve_cpml_harmonic(grid, w, pol)['amplitudes'] for w in BAND])
            boundary[pol] = float(np.max(abs(finite[pol]-exact[pol])))
    shape = grid['epsilon'].shape[1:]
    row = dict(case=name, resolution=rz, transverse_resolution=[rx, ry],
               duration_periods=800,
               shape=shape, spacing=grid['spacing'], dt=grid['dt'], steps=grid['steps'],
               four_run_cell_updates=int(4*np.prod(shape)*grid['steps']),
               error_rcwa=errors, boundary_error=boundary)
    return row, grid, exact, finite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['all', *CASES], default='all')
    parser.add_argument('--cpml', action='store_true', help='Also measure finite-boundary error.')
    parser.add_argument('--fixed-dt', type=float, help='Harmonic-only control: hold the time symbol fixed across meshes.')
    parser.add_argument('--time', action='store_true', help='Run actual pulses for the selected case; cannot select all.')
    parser.add_argument('--periods', type=int, choices=[800, 1600], default=800)
    parser.add_argument('--backend', choices=['jax', 'numpy'], default='jax')
    args = parser.parse_args()
    if args.time and args.case == 'all':
        parser.error('--time requires one explicit --case')
    if args.time and args.fixed_dt is not None:
        parser.error('--fixed-dt is a harmonic-only control')
    if args.time and args.backend == 'jax':
        import jax
        jax.config.update('jax_enable_x64', True)
    expected = reference(32)
    for name in CASES:
        if args.case not in ('all', name):
            continue
        row, grid, exact, finite = harmonic_case(name, expected, args.cpml or args.time, args.fixed_dt)
        if args.time:
            # Independent transverse symbol gives the vacuum cutoff frequency
            # for this case's actual time step and x spacing.
            dt, dx = grid['dt'], grid['spacing'][2]
            kx = 2*np.pi*(.2/.96-1/PERIOD)
            cutoff = np.arcsin(dt*abs(np.sin(kx*dx/2))/dx)/(np.pi*dt)
            rz, rx, ry = CASES[name]
            window = int(np.ceil((args.periods/2 if args.periods >= 1600 else 100)*.96/dt))
            start = time.perf_counter()
            measured = run(expected, args.backend, resolution=rz, transverse_resolution=(rx, ry),
                           periods=args.periods, width=60., distance=.2, pml_width=1.2,
                           pml_reflection=1e-160, source_distance=1., method='directional',
                           pulse_notches=[cutoff, cutoff], convergence_window=window)
            seconds = time.perf_counter()-start
            time_error, prefix_error = {}, {}
            for pol in ('te', 'tm'):
                values = np.array(measured['amplitude_real'][pol])+1j*np.array(measured['amplitude_imag'][pol])
                earlier = (np.array(measured['prefix_amplitude_real'][pol])
                           +1j*np.array(measured['prefix_amplitude_imag'][pol]))
                time_error[pol] = float(np.max(abs(values-finite[pol])))
                prefix_error[pol] = float(np.max(abs(earlier-finite[pol])))
            row.update(time=measured, time_error=time_error, prefix_time_error=prefix_error,
                       elapsed_seconds=seconds, duration_periods=args.periods, steps=measured['steps'],
                       four_run_cell_updates=int(4*np.prod(measured['shape'])*measured['steps']))
        print(json.dumps(row, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
