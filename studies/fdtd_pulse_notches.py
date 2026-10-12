"""Does a source zero at a diffraction cutoff reduce accumulated time error?"""
import argparse
import json

import numpy as np

from studies.fdtd_rayleigh import PERIOD, reference, run
from wavebit._core.conventions import time_delay_phase


CASES = {
    'single40': dict(resolution=40, periods=800, multiplicity=1),
    'double40': dict(resolution=40, periods=800, multiplicity=2),
    'long40': dict(resolution=40, periods=1600, multiplicity=1, convergence_window=66511),
    'double-long40': dict(resolution=40, periods=1600, multiplicity=2, convergence_window=66511),
    'single80': dict(resolution=80, periods=800, multiplicity=1),
    'double80': dict(resolution=80, periods=800, multiplicity=2),
    'physical40': dict(resolution=40, periods=800, multiplicity=1, physical=True),
}


def notch_frequency(resolution, physical=False):
    """Independent transverse Yee cutoff for this study's known uniform spacing."""
    kx = 2*np.pi*(.2/.96-1/PERIOD)
    dx = PERIOD/np.ceil(PERIOD*resolution)
    dt = .8/np.sqrt(2*resolution**2+dx**-2)
    return abs(kx)/(2*np.pi) if physical else np.arcsin(dt*abs(np.sin(kx*dx/2))/dx)/(np.pi*dt)


def source_edge_control(resolution):
    """Measure edge amplification, including the unsupported three-zero control."""
    dt = .8/np.sqrt(3*resolution**2)
    t = np.arange(int(np.ceil(800*.96/dt)))*dt
    base = np.exp(-((t-300)/60)**2)*np.exp(-2j*np.pi*t/.96)
    rows = []
    for count in (1, 2, 3):
        pulse = base[:-count]
        for _ in range(count):
            pulse = np.convolve(pulse, [1., -time_delay_phase(np, notch_frequency(resolution), dt)])
            pulse /= np.max(abs(pulse))
        spectrum = abs(np.fft.fft(pulse))
        frequencies = np.fft.fftfreq(len(pulse), dt)
        outside = abs(frequencies+1/.96) > .2
        rows.append(dict(multiplicity=count, sampled_peak_time=float(t[np.argmax(abs(pulse))]),
                         out_of_band_peak_ratio=float(np.max(spectrum[outside])/np.max(spectrum))))
    return dict(resolution=resolution, source_edge_control=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['all', *CASES], default='all')
    parser.add_argument('--backend', choices=['jax', 'numpy'], default='jax')
    parser.add_argument('--source-only', action='store_true', help='Measure source-edge amplification without a field solve.')
    args = parser.parse_args()
    if args.source_only:
        for resolution in (40, 80):
            print(json.dumps(source_edge_control(resolution), allow_nan=False), flush=True)
        return
    if args.backend == 'jax':
        import jax
        jax.config.update('jax_enable_x64', True)
    expected = reference(32)
    for name, case in CASES.items():
        if args.case not in ('all', name):
            continue
        options = dict(case)
        frequency = notch_frequency(options['resolution'], options.pop('physical', False))
        notches = [frequency]*options.pop('multiplicity')
        result = run(expected, args.backend, distance=.2, width=60., pml_width=.6,
                     method='directional', pml_reflection=1e-40, pulse_notches=notches, **options)
        print(json.dumps(dict(case=name, **result), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
