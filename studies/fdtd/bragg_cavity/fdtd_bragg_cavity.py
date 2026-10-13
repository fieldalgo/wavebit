"""Separate long-lived Bragg cavity transients from spatial detuning."""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
import argparse

import numpy as np


from studies._support._fdtd_legacy import legacy_fdtd  # noqa: E402


def cavity():
    indices = np.r_[np.tile([2., 1.], 3), 1., np.tile([1., 2.], 3)]
    thickness = np.r_[np.tile([.125, .25], 3), .5, np.tile([.25, .125], 3)]
    return indices, thickness


def analytic_transmission(frequencies):
    """Normal-incidence lossless characteristic matrices, vacuum on both sides."""
    indices, thickness = cavity()
    frequencies = np.asarray(frequencies)
    matrix = np.broadcast_to(np.eye(2, dtype=complex), frequencies.shape+(2, 2)).copy()
    for index, depth in zip(indices, thickness):
        phase = 2*np.pi*frequencies*index*depth
        layer = np.empty_like(matrix)
        layer[..., 0, 0] = layer[..., 1, 1] = np.cos(phase)
        layer[..., 0, 1] = -1j*np.sin(phase)/index
        layer[..., 1, 0] = -1j*np.sin(phase)*index
        matrix = matrix @ layer
    return 2/matrix.sum(axis=(-1, -2))


def discrete_scattering(resolution, dt, frequency=1.):
    """Solve the uniform 1D Yee Helmholtz recurrence with exact outgoing leads.

    Tangential epsilon is independently averaged over E-node control volumes.
    No PML, time stepping, source calibration or Wavebit material sampler is used.
    """
    indices, thickness = cavity()
    spacing = 1/resolution
    depth = thickness.sum()
    z = np.arange(-1, round(depth/spacing)+2)*spacing
    cuts = np.r_[0., np.cumsum(thickness)]
    overlap = np.maximum(0., np.minimum((z+spacing/2)[:, None], cuts[1:])
                         - np.maximum((z-spacing/2)[:, None], cuts[:-1]))
    epsilon = 1+overlap@(indices**2-1)/spacing
    phase = 2*np.arcsin(spacing*np.sin(np.pi*frequency*dt)/dt)
    operator = np.diag((4*spacing**2*np.sin(np.pi*frequency*dt)**2/dt**2*epsilon-2).astype(complex))
    operator += np.diag(np.ones(len(z)-1), 1)+np.diag(np.ones(len(z)-1), -1)
    operator[0, 0] += np.exp(1j*phase)
    operator[-1, -1] += np.exp(1j*phase)
    rhs = np.zeros(len(z), complex)
    rhs[0] = (np.exp(1j*phase)-np.exp(-1j*phase))*np.exp(1j*phase*z[0]/spacing)
    field = np.linalg.solve(operator, rhs)
    r = (field[0]-np.exp(1j*phase*z[0]/spacing))*np.exp(1j*phase*z[0]/spacing)
    t = field[-1]*np.exp(-1j*phase*(z[-1]-depth)/spacing)
    return r, t


def measure(resolution, periods, *, n_pml=12, pad_scale=1.):
    indices, thickness = cavity()
    # 200 steps at resolution 40; preserve a comparable window on finer grids.
    window = 200 if periods < 400 else int(5*resolution*np.sqrt(3)/.8)
    result, diagnostic = legacy_fdtd(
        structure=(indices**2)[:, None, None], thickness=thickness, top=1., bot=1.,
        period=.05, wavelength=1., backend='numpy', resolution=resolution,
        n_periods=periods, n_pml=n_pml, pad_scale=pad_scale, truncation_order=0, tetm='te', return_diagnostics=True,
        convergence_window=window)
    c = diagnostic['convergence']
    discrete_r, discrete_t = discrete_scattering(resolution, diagnostic['dt'])
    return dict(resolution=resolution, periods=periods, window=window,
                n_pml=n_pml, pad_scale=pad_scale, pml_width=n_pml/resolution,
                dt=diagnostic['dt'], shape=diagnostic['shape'],
                r_real=result.incident.r_s.item().real, r_imag=result.incident.r_s.item().imag,
                t_real=result.incident.t_s.item().real, t_imag=result.incident.t_s.item().imag,
                discrete_error=max(abs(result.incident.r_s.item()-discrete_r),
                                   abs(result.incident.t_s.item()-discrete_t)),
                power_error=abs(result.R.item()+result.T.item()-1),
                T=result.T.item(), error=abs(result.incident.t_s.item()-analytic_transmission(1.).item()),
                decay=max(float(v[k]['field_decay']) for v in c['bases'] for k in ('actual', 'reference')),
                change=float(abs(c['amplitude_change']['t_s']).max()),
                settled=bool(c['amplitudes_settled']), field_decayed=bool(c['field_decayed']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controls', action='store_true', help='Vary PML and padding independently at 400 periods.')
    args = parser.parse_args()
    frequencies = np.linspace(.98, 1.02, 40001)
    power = abs(analytic_transmission(frequencies))**2
    middle = len(frequencies)//2
    low = np.interp(.5, power[:middle+1], frequencies[:middle+1])
    high = np.interp(.5, power[middle:][::-1], frequencies[middle:][::-1])
    print(f'Continuum linewidth Q: {1/(high-low):.6g}; t(f=1)={analytic_transmission(1.)}')
    if args.controls:
        from scipy.optimize import minimize_scalar
        for resolution in (40, 80):
            dt = .8/(resolution*np.sqrt(3))
            peak = minimize_scalar(lambda f: -abs(discrete_scattering(resolution, dt, f)[1])**2,
                                   bounds=(.99, 1.01), method='bounded', options={'xatol': 1e-12})
            print(dict(resolution=resolution, discrete_peak_frequency=peak.x,
                       discrete_peak_T=-peak.fun), flush=True)
        for resolution, n_pml, pad_scale in [(40, 12, 1.), (40, 24, 1.), (40, 12, 1.5), (80, 24, 1.)]:
            print(measure(resolution, 400, n_pml=n_pml, pad_scale=pad_scale), flush=True)
    else:
        for resolution, periods in [(40, 25), (40, 80), (40, 200), (40, 400), (80, 400)]:
            print(measure(resolution, periods), flush=True)


if __name__ == '__main__':
    main()
