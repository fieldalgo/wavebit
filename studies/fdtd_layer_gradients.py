"""Do moving-interface derivatives converge to fixed-reference-plane film optics?

Run: python -m studies.fdtd_layer_gradients --output /tmp/layer-gradients.json
The film ends at d; a substrate spacer continues to the fixed output plane D.
Only the public scene API is differentiated. No external implementation is used.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import wavebit
from wavebit.backends.time_domain import time_loop_backend


BAND = np.array([.98, 1.02])
DEPTH = .4
BACKGROUND = (1.2, 1.1, 1.6, 1.15)
FILM = (2.25, 1.12)


def airy(d, *, band=BAND, depth=DEPTH, background=BACKGROUND, film=FILM):
    """Independent Fresnel series; transmission includes the remaining spacer."""
    et, mt, eb, mb = background
    epsilon, mu = film
    y0, y1, y2 = np.sqrt(et/mt), np.sqrt(epsilon/mu), np.sqrt(eb/mb)
    a, b = (y0-y1)/(y0+y1), (y1-y2)/(y1+y2)
    phase = np.exp(2j*np.pi*np.sqrt(epsilon*mu)*d/band)
    r = (a+b*phase**2)/(1+a*b*phase**2)
    t = (1+a)*(1+b)*phase/(1+a*b*phase**2)*np.exp(2j*np.pi*np.sqrt(eb*mb)*(depth-d)/band)
    return np.concatenate((r.real, r.imag, t.real, t.imag, abs(r)**2, y2/y0*abs(t)**2))


def scene_vector(d, resolution, *, backend=1, band=BAND, depth=DEPTH,
                 background=BACKGROUND, film=FILM):
    xp = time_loop_backend(backend).xp
    et, mt, eb, mb = background
    epsilon, mu = film
    results = wavebit.fdtd(structure=np.array([[[epsilon]], [[eb]]]),
                           mu=np.array([[[mu]], [[mb]]]), thickness=[d, depth-d],
                           mesh_thickness=[depth/2, depth/2], top=wavebit.Material(et, mt),
                           bot=wavebit.Material(eb, mb), mesh_indices=(np.sqrt(et*mt), np.sqrt(eb*mb)),
                           period=.05, wavelength=band, backend=backend, resolution=resolution,
                           n_pml=resolution, n_periods=65, pulse_width=3., align_layers=True,
                           diffraction_method='directional', truncation_order=0)
    r = xp.stack([s.s_inc.r_s.reshape(()) for s in results])
    t = xp.stack([s.s_inc.t_s.reshape(()) for s in results])
    return xp.concat((xp.real(r), xp.imag(r), xp.real(t), xp.imag(t),
                      xp.stack([s.s_inc.R.reshape(()) for s in results]),
                      xp.stack([s.s_inc.T.reshape(()) for s in results])))


def measure(resolutions=(20, 40, 80), positions=(.163, .173, .183)):
    import jax
    jax.config.update('jax_enable_x64', True)
    rows = []
    for resolution in resolutions:
        f = lambda d: scene_vector(d, resolution)
        evaluate = jax.jit(lambda d: (f(d), jax.jacfwd(f)(d)))
        for d in positions:
            values, derivative = (np.asarray(v) for v in evaluate(np.array(d)))
            exact = airy(d)
            delta = 1e-5
            target = (airy(d+delta)-airy(d-delta))/(2*delta)
            finer = (airy(d+delta/2)-airy(d-delta/2))/delta
            row = dict(resolution=resolution, thickness=d,
                       value_error=float(np.max(abs(values-exact))),
                       derivative_error=float(np.max(abs(derivative-target))),
                       oracle_difference=float(np.max(abs(target-finer))),
                       values=values.tolist(), derivatives=derivative.tolist(),
                       exact_values=exact.tolist(), exact_derivatives=target.tolist())
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if not isinstance(v, list)}), flush=True)
    return dict(wavelengths=BAND.tolist(), total_depth=DEPTH, background=BACKGROUND, film=FILM,
                pulse_width=3., n_periods=65, physical_pml=1., rows=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(measure(), indent=2)+'\n')
