"""Compare finite-exponential and algebraic fits of a near-Rayleigh tail."""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
import argparse
import json

import numpy as np
from wavebit.fdtd._periodic.probes import fit_fdtd_modes as wavebit_fit_fdtd_modes

from studies.fdtd.rayleigh.fdtd_rayleigh import BAND, STRUCTURE, THETA, PERIOD


from studies._support._fdtd_legacy import legacy_fdtd  # noqa: E402


def record(pad_scale, backend):
    _, d = legacy_fdtd(
        structure=STRUCTURE, thickness=[.2], top=1., bot=1., period=PERIOD,
        wavelength=BAND, theta=THETA, polarization=(1., 1.), truncation_order=1,
        backend=backend, resolution=40, n_pml=24, n_periods=800, pulse_width=60.,
        align_layers=True, diffraction_distance=.2, diffraction_method='directional',
        pml_reflection=1e-40, pad_scale=pad_scale, return_diagnostics=True,
        record_field=10, record_components=['ey', 'ez'],
        record_bounds={'x': (-.201, -.199), 'z': (.09, .13)})
    result = dict(pad_scale=pad_scale, dt=d['dt'], dx=d['spacing'][2], pml_bounds=d['pml_bounds'])
    for name, field in d['fields'].items():
        result.update({name+'_'+key: np.asarray(field[key]) for key in ('t', 'values', 'x', 'y', 'z')})
    return result


def power_law(t, y):
    """Descriptive envelope fit with the known pulse peak; not a pole model."""
    x = np.log((t-300)/(t[0]-300))
    slope, intercept = np.linalg.lstsq(np.c_[x, np.ones_like(x)], np.log(abs(y)), rcond=None)[0]
    phase_slope, phase = np.linalg.lstsq(
        np.c_[t-t[0], np.ones_like(t)], np.unwrap(np.angle(y)), rcond=None)[0]
    fitted = np.exp(intercept+slope*x+1j*(phase+phase_slope*(t-t[0])))
    return dict(exponent=float(-slope), frequency=float(-phase_slope/(2*np.pi)),
                relative_residual=float(np.linalg.norm(y-fitted)/np.linalg.norm(y)))


def analyze(data):
    dx, dt = float(data['dx']), float(data['dt'])
    kx = 2*np.pi*(.2/.96-1/PERIOD)
    cutoff = np.arcsin(dt*abs(np.sin(kx*dx/2))/dx)/(np.pi*dt)
    rows = []
    for name in ('ey', 'ez'):
        time, values = data[name+'_t'], data[name+'_values'][:, 0, 0, 0]
        for start in (540., 600., 660.):
            select = time >= start
            t, signal = time[select], values[select]
            fits = []
            for modes in (1, 2, 3):
                f = wavebit_fit_fdtd_modes(t, signal, modes=modes, pencil=128)
                fits.append(dict(modes=modes, frequency=f['frequency'].tolist(),
                                 decay_rate=f['decay_rate'].tolist(), amplitude_abs=abs(f['amplitude']).tolist(),
                                 relative_residual=f['relative_residual'], basis_condition=f['basis_condition'],
                                 singular_values=f['singular_values'][:5].tolist()))
            rows.append(dict(component=name, start=start, stop=float(t[-1]),
                             coordinates={axis: float(data[name+'_'+axis][0]) for axis in ('x', 'y', 'z')},
                             exponentials=fits, power_law=power_law(t, signal)))
    return dict(pad_scale=float(data['pad_scale']), discrete_cutoff_frequency=float(cutoff),
                physical_cutoff_frequency=float(abs(kx)/(2*np.pi)),
                pml_bounds=np.asarray(data['pml_bounds']).tolist(), fits=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pad-scale', type=float, choices=[1., 2.], default=1.)
    parser.add_argument('--backend', choices=['jax', 'numpy'], default='jax')
    parser.add_argument('--load-records', help='Analyze an existing archive from --save-records.')
    parser.add_argument('--save-records', help='Optional npz output for later window/model comparisons.')
    args = parser.parse_args()
    if args.load_records:
        with np.load(args.load_records, allow_pickle=False) as archive:
            data = dict(archive)
    else:
        if args.backend == 'jax':
            import jax
            jax.config.update('jax_enable_x64', True)
        data = record(args.pad_scale, args.backend)
    if args.save_records:
        np.savez_compressed(args.save_records, **data)
    print(json.dumps(analyze(data), allow_nan=False))
