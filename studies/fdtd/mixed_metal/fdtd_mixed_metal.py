"""Low-loss Drude/Lorentz slab: temporal, spatial and sensitivity errors."""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
import argparse
import json

import numpy as np


BAND = np.array([.9, 1., 1.1])
PARAMETERS = np.array([1.5, .08, 160., 10., .12, 40.])


from studies._support._fdtd_legacy import legacy_fdtd  # noqa: E402


def airy(parameters=PARAMETERS):
    epsilon_inf, drude_gamma, drude_strength, omega0, gamma, strength = parameters
    omega = 2*np.pi/BAND
    epsilon = epsilon_inf+drude_strength/(-omega**2-1j*drude_gamma*omega)
    epsilon += strength/(omega0**2-omega**2-1j*gamma*omega)
    # Positive loss fixes the passive square root for this particular model.
    index = np.sqrt(epsilon+0j)
    interface = (1-index)/(1+index)
    phase = omega*index*.2
    denominator = 1-interface**2*np.exp(2j*phase)
    return (interface*(1-np.exp(2j*phase))/denominator,
            (1-interface**2)*np.exp(1j*phase)/denominator)


def solve(resolution, periods, parameters=PARAMETERS, *, courant=.8, pad_scale=1., pml_width=.3):
    epsilon_inf, drude_gamma, drude_strength, omega0, gamma, strength = parameters
    result, diagnostic = legacy_fdtd(
        structure=np.full((1, 1, 1), epsilon_inf), thickness=[.2], top=1., bot=1.,
        period=.05, wavelength=BAND, backend='numpy', resolution=resolution, n_periods=periods,
        n_pml=round(pml_width*resolution), courant=courant, pad_scale=pad_scale,
        tetm='te', truncation_order=0, dispersion=np.ones((1, 1, 1), int),
        poles={1: [(0., drude_gamma, drude_strength), (omega0, gamma, strength)]},
        return_diagnostics=True, convergence_window=round(2*resolution*np.sqrt(3)/courant))
    return (np.array([v.incident.r_s.item() for v in result]),
            np.array([v.incident.t_s.item() for v in result]), diagnostic)


def measure(resolution, periods, **options):
    r, t, diagnostic = solve(resolution, periods, **options)
    expected_r, expected_t = airy()
    c = diagnostic['convergence']
    row = dict(resolution=resolution, periods=periods, **options,
               dt=float(diagnostic['dt']), max_r_error=float(max(abs(r-expected_r))),
               max_t_error=float(max(abs(t-expected_t))),
               absorptance=(1-abs(r)**2-abs(t)**2).tolist(),
               field_decay=max(float(b[k]['field_decay']) for b in c['bases'] for k in ('actual', 'reference')),
               material_decay=np.max([b['actual']['material_decay'] for b in c['bases']], axis=0).tolist(),
               amplitude_change=max(float(abs(v).max()) for v in c['amplitude_change'].values()),
               field_decayed=bool(c['field_decayed']), material_decayed=bool(c['material_decayed']),
               state_decayed=bool(c['state_decayed']), amplitudes_settled=bool(c['amplitudes_settled']))
    print(json.dumps(row), flush=True)
    return r, t


def absorption(r, t):
    return np.mean(1-abs(r)**2-abs(t)**2)


def sensitivity(resolution, periods):
    # Relative perturbations give derivatives with respect to multiplicative
    # parameter factors, avoiding incomparable frequency/strength units.
    delta = 1e-3
    for index, name in [(1, 'drude_gamma'), (4, 'lorentz_gamma')]:
        upper, lower = PARAMETERS.copy(), PARAMETERS.copy()
        upper[index] *= 1+delta
        lower[index] *= 1-delta
        expected = (absorption(*airy(upper))-absorption(*airy(lower)))/(2*delta)
        high = solve(resolution, periods, upper)
        low = solve(resolution, periods, lower)
        actual = (absorption(*high[:2])-absorption(*low[:2]))/(2*delta)
        print(json.dumps(dict(resolution=resolution, periods=periods, parameter=name,
                              absorption_derivative=float(actual), airy_derivative=float(expected),
                              relative_error=float(abs(actual/expected-1)),
                              amplitudes_settled=all(bool(v[2]['convergence']['amplitudes_settled'])
                                                     for v in (high, low)))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controls', action='store_true', help='Vary dt, PML and padding at fixed resolution.')
    parser.add_argument('--gradients', action='store_true', help='Compare damping sensitivities with Airy.')
    args = parser.parse_args()
    runs = {(res, periods): measure(res, periods)
            for res, periods in [(40, 40), (40, 160), (40, 320), (80, 320)]}
    if args.controls:
        base_r, base_t = runs[(40, 320)]
        for options in [dict(courant=.4), dict(pml_width=.6), dict(pad_scale=1.5)]:
            r, t = measure(40, 320, **options)
            print(json.dumps(dict(control=options, max_amplitude_shift=float(max(
                max(abs(r-base_r)), max(abs(t-base_t)))))), flush=True)
    if args.gradients:
        for res in (40, 80):
            sensitivity(res, 320)
