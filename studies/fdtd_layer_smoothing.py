"""Does C1 interface averaging improve derivatives, and what bias remains?

Run: python -m studies.fdtd_layer_smoothing --output /tmp/layer-smoothing.json
Both the sharp-film and the graded-interface continuum references are independent
of the Yee solver. The latter integrates many thin homogeneous slices.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .fdtd_layer_gradients import BAND, DEPTH, BACKGROUND, FILM, airy, scene_vector


def graded_film(d, smoothing, *, slices=512, band=BAND, depth=DEPTH,
                background=BACKGROUND, film=FILM):
    """Transfer matrices and analytic translation derivative of a linear ramp.

    At normal incidence the tangent epsilon and mu are linearly interpolated.
    Translating the ramp only changes the constant-material lengths around it;
    its transfer matrix is independent of d. No AD or FDTD fractions are used.
    """
    et, mt, eb, mb = background
    ef, mf = film
    frequency = 2*np.pi/np.atleast_1d(band)
    y0, yb = np.sqrt(et/mt), np.sqrt(eb/mb)
    def propagator(epsilon, mu, length):
        y = np.sqrt(epsilon/mu)
        angle = frequency*np.sqrt(epsilon*mu)*length
        c, s = np.cos(angle), 1j*np.sin(angle)
        return np.stack((np.stack((c, s/y), axis=-1), np.stack((y*s, c), axis=-1)), axis=-2)
    ramp = np.broadcast_to(np.eye(2, dtype=complex), (len(frequency), 2, 2)).copy()
    if smoothing:
        for fraction in (np.arange(slices)+.5)/slices:
            ramp = propagator(ef+(eb-ef)*fraction, mf+(mb-mf)*fraction, smoothing/slices) @ ramp
    left = propagator(ef, mf, d-smoothing/2)
    right = propagator(eb, mb, depth-d-smoothing/2)
    gf = 1j*frequency[:, None, None]*np.array([[0., mf], [ef, 0.]])
    gb = 1j*frequency[:, None, None]*np.array([[0., mb], [eb, 0.]])
    matrix = right @ ramp @ left
    derivative = -(gb @ matrix) + right @ ramp @ gf @ left
    a = matrix[:, 1, 0]-yb*matrix[:, 0, 0]
    b = y0*(matrix[:, 1, 1]-yb*matrix[:, 0, 1])
    da = derivative[:, 1, 0]-yb*derivative[:, 0, 0]
    db = y0*(derivative[:, 1, 1]-yb*derivative[:, 0, 1])
    r = -(a+b)/(a-b)
    dr = -((da+db)*(a-b)-(a+b)*(da-db))/(a-b)**2
    t = matrix[:, 0, 0]*(1+r)+y0*matrix[:, 0, 1]*(1-r)
    dt = (derivative[:, 0, 0]*(1+r)+y0*derivative[:, 0, 1]*(1-r)
          +(matrix[:, 0, 0]-y0*matrix[:, 0, 1])*dr)
    value = np.concatenate((r.real, r.imag, t.real, t.imag, abs(r)**2, yb/y0*abs(t)**2))
    gradient = np.concatenate((dr.real, dr.imag, dt.real, dt.imag,
                               2*np.real(r.conj()*dr), 2*yb/y0*np.real(t.conj()*dt)))
    return value, gradient


def measure(resolutions=(20, 40, 80), smoothing_cells=(0., .5, 1., 2.), positions=(.163, .173, .183)):
    import jax
    jax.config.update('jax_enable_x64', True)
    rows = []
    for resolution in resolutions:
        for cells in smoothing_cells:
            smoothing = cells/resolution
            f = lambda d: scene_vector(d, resolution, smoothing=smoothing)
            evaluate = jax.jit(lambda d: (f(d), jax.jacfwd(f)(d)))
            for d in positions:
                value, gradient = (np.asarray(v) for v in evaluate(np.array(d)))
                exact, target = graded_film(d, 0.)
                graded, graded_target = graded_film(d, smoothing)
                refined, refined_target = graded_film(d, smoothing, slices=1024)
                row = dict(resolution=resolution, smoothing_cells=cells, smoothing=smoothing, thickness=d,
                           value_error=float(np.max(abs(value-exact))),
                           derivative_error=float(np.max(abs(gradient-target))),
                           graded_value_error=float(np.max(abs(value-graded))),
                           graded_derivative_error=float(np.max(abs(gradient-graded_target))),
                           smoothing_value_bias=float(np.max(abs(graded-exact))),
                           smoothing_derivative_bias=float(np.max(abs(graded_target-target))),
                           reference_change=float(max(np.max(abs(refined-graded)), np.max(abs(refined_target-graded_target)))),
                           values=value.tolist(), derivatives=gradient.tolist())
                np.testing.assert_allclose(exact, airy(d), atol=2e-14)
                rows.append(row)
            summary = {key: max(r[key] for r in rows[-len(positions):]) for key in rows[-1]
                       if key.endswith(('_error', '_bias', '_change'))}
            print(json.dumps(dict(resolution=resolution, smoothing_cells=cells, **summary)), flush=True)
    return dict(wavelengths=BAND.tolist(), depth=DEPTH, background=BACKGROUND, film=FILM,
                physical_pml=1., pulse_width=3., duration=65, rows=rows)


def crossing(resolution=20, position=.175, offsets=(1e-4, 1e-6)):
    """Measure both sides of a Yee cell face with and without smoothing."""
    import jax
    jax.config.update('jax_enable_x64', True)
    rows = []
    for smoothing in (0., 1/resolution):
        f = lambda d: scene_vector(d, resolution, smoothing=smoothing)
        derivative = jax.jit(jax.jacfwd(f))
        for offset in offsets:
            left = np.asarray(derivative(np.array(position-offset)))
            right = np.asarray(derivative(np.array(position+offset)))
            rows.append(dict(smoothing=smoothing, offset=offset,
                             derivative_gap=float(np.max(abs(right-left)))))
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(dict(**measure(), crossing=crossing()), indent=2)+'\n')
