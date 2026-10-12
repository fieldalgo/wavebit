"""Do live exterior reference planes preserve total-depth optical derivatives?"""
import argparse
import json
from pathlib import Path

import numpy as np

import wavebit
from wavebit.backends.time_domain import time_loop_backend
from .fdtd_layer_gradients import BAND, BACKGROUND, FILM, DEPTH
from .fdtd_layer_smoothing import graded_film


from studies._fdtd_legacy import legacy_fdtd  # noqa: E402


def complex_amplitudes(vector):
    r, i, t, j, _, _ = np.asarray(vector).reshape(6, -1)
    return np.stack((r+1j*i, t+1j*j))


def reference(d, smoothing=.025, *, side='top', slices=512):
    """Independent graded transfer matrix, translated to the physical exit.

    The earlier reference includes a substrate spacer to fixed DEPTH. Remove
    its phase and differentiate that removal. Lossless reciprocity supplies
    the opposite incidence; a separate sharp-film check validates that rule.
    """
    value, gradient = graded_film(d, smoothing, slices=slices)
    r, t = complex_amplitudes(value)
    dr, dt = complex_amplitudes(gradient)
    et, mt, eb, mb = BACKGROUND
    y0, yb = np.sqrt(et/mt), np.sqrt(eb/mb)
    kb = 2*np.pi*np.sqrt(eb*mb)/BAND
    phase = np.exp(-1j*kb*(DEPTH-d))
    dt, t = (dt+1j*kb*t)*phase, t*phase
    if side == 'bottom':
        dr, r = (-dr.conj()*t/t.conj()-r.conj()*dt/t.conj()
                 +r.conj()*t*dt.conj()/t.conj()**2), -r.conj()*t/t.conj()
        dt, t = dt*yb/y0, t*yb/y0
        y0, yb = yb, y0
    elif side != 'top':
        raise ValueError('side must be top or bottom.')
    value = np.r_[r.real, r.imag, t.real, t.imag, abs(r)**2, yb/y0*abs(t)**2]
    gradient = np.r_[dr.real, dr.imag, dt.real, dt.imag,
                      2*np.real(r.conj()*dr), 2*yb/y0*np.real(t.conj()*dt)]
    return value, gradient


def response(d, resolution, *, backend='jax', side='top', smoothing=.025, duration=65.):
    xp = time_loop_backend(backend).xp
    et, mt, eb, mb = BACKGROUND
    ef, mf = FILM
    results = legacy_fdtd(structure=np.array([[[ef]]]), mu=np.array([[[mf]]]), thickness=[d],
        mesh_thickness=[.2], variable_depth=True, layer_smoothing=smoothing,
        top=wavebit.Material(et, mt), bot=wavebit.Material(eb, mb), period=.05,
        wavelength=BAND, backend=backend, resolution=resolution, n_pml=resolution,
        n_periods=duration, pulse_width=3., align_layers=True, incidence_side=side,
        diffraction_method='directional', truncation_order=0)
    r = xp.stack([a.s_inc.r_s.reshape(()) for a in results])
    t = xp.stack([a.s_inc.t_s.reshape(()) for a in results])
    return xp.concat((xp.real(r), xp.imag(r), xp.real(t), xp.imag(t),
                       xp.stack([a.s_inc.R.reshape(()) for a in results]),
                       xp.stack([a.s_inc.T.reshape(()) for a in results])))


def measure(resolutions=(20, 40, 80), d=.173, smoothing=.025):
    import jax
    jax.config.update('jax_enable_x64', True)
    rows = []
    for side in ('top', 'bottom'):
        target, derivative = reference(d, smoothing, side=side)
        for resolution in resolutions:
            f = lambda p: response(p, resolution, side=side, smoothing=smoothing)
            value, jac = map(np.asarray, jax.jit(lambda p: (f(p), jax.jacfwd(f)(p)))(np.array(d)))
            row = dict(side=side, resolution=resolution,
                amplitude_error=float(np.max(abs(complex_amplitudes(value)-complex_amplitudes(target)))),
                derivative_error=float(np.max(abs(complex_amplitudes(jac)-complex_amplitudes(derivative)))),
                energy_error=float(np.max(abs(value[-4:-2]+value[-2:]-1))),
                values=value.tolist(), derivatives=jac.tolist())
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k not in ('values', 'derivatives')}), flush=True)
    return dict(thickness=d, smoothing=smoothing, wavelengths=BAND.tolist(), rows=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(measure(), indent=2)+'\n')
