"""Measure directional extraction separately from near-Rayleigh boundary errors."""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
import argparse
import json

import numpy as np

from studies.fdtd.rayleigh.fdtd_rayleigh import reference, run

CASES = {
    'far': dict(resolution=40),
    'near': dict(resolution=40, distance=.2),
    'long': dict(resolution=40, distance=.2, periods=600),
    'padding': dict(resolution=40, distance=.2, pad_scale=2., periods=600),
    'pml': dict(resolution=40, distance=.2, pml_width=.6, periods=600),
    'narrow': dict(resolution=40, distance=.2, width=60., periods=800),
    'fine': dict(resolution=80, distance=.2, width=60., periods=800),
}


from studies._support._fdtd_legacy import legacy_fdtd, legacy_rcwa  # noqa: E402


def ringdown(backend):
    """Fit a late-time point signal; the fit does not identify a mode by itself."""
    from studies.fdtd.rayleigh.fdtd_rayleigh import BAND, STRUCTURE, THETA, PERIOD
    _, d = legacy_fdtd(
        structure=STRUCTURE, thickness=[.2], top=1., bot=1., period=PERIOD, wavelength=BAND,
        theta=THETA, tetm='tm', truncation_order=1, backend=backend, resolution=40,
        n_pml=12, n_periods=300, pulse_width=3., align_layers=True, diffraction_distance=.2,
        diffraction_method='directional', return_diagnostics=True, record_field=10,
        record_components=['ez'], record_bounds={'x': (-.201, -.199), 'z': (.1, .13)})
    field = d['fields']['ez']
    time, values = np.asarray(field['t']), np.asarray(field['values'])[:, 0, 0, 0]
    rows = []
    for start in (60, 120, 180):
        signal, times = values[time > start], time[time > start]
        dt = times[1]-times[0]
        coefficient = np.vdot(signal[:-1], signal[1:])/np.vdot(signal[:-1], signal[:-1])
        rows.append(dict(start=start, frequency=float(-np.angle(coefficient)/(2*np.pi*dt)),
                         decay_rate=float(-np.log(abs(coefficient))/dt),
                         relative_residual=float(np.linalg.norm(signal[1:]-coefficient*signal[:-1])/np.linalg.norm(signal[1:]))))
    return dict(case='ringdown', coordinates={axis: np.asarray(field[axis]).tolist() for axis in ('x', 'y', 'z')}, fits=rows)


def resonance_reference():
    """Locate a nearby RCWA reflection maximum without assuming identical poles."""
    from studies.fdtd.rayleigh.fdtd_rayleigh import STRUCTURE, PERIOD
    rows = []
    for order in (16, 32):
        def power(frequency):
            result = legacy_rcwa(structure=STRUCTURE, thickness=[.2], top=1., bot=1., period=PERIOD,
                                   wavelength=1/frequency, theta=np.arcsin(.2/(.96*frequency)),
                                   tetm='tm', truncation_order=order).solve()
            return float(result.R.sum())
        frequencies = np.linspace(1., 1.035, 501)
        peak = frequencies[np.argmax([power(f) for f in frequencies])]
        fine = np.linspace(peak-.0001, peak+.0001, 201)
        values = np.array([power(f) for f in fine])
        rows.append(dict(order=order, frequency=float(fine[np.argmax(values)]), reflectance=float(values.max())))
    return dict(case='rcwa-resonance', peaks=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['all', 'ringdown', 'rcwa-resonance', *CASES], default='all')
    parser.add_argument('--backend', choices=['jax', 'numpy'], default='jax')
    args = parser.parse_args()
    if args.backend == 'jax':
        import jax
        jax.config.update('jax_enable_x64', True)
    expected = reference(32)
    for name, case in CASES.items():
        if args.case in ('all', name):
            print(json.dumps(dict(case=name, **run(expected, args.backend, method='directional', **case))), flush=True)
    if args.case in ('all', 'ringdown'):
        print(json.dumps(ringdown(args.backend)), flush=True)
    if args.case in ('all', 'rcwa-resonance'):
        print(json.dumps(resonance_reference()), flush=True)
