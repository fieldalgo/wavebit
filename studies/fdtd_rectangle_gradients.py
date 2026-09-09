"""How do rectangle shape derivatives refine toward independent Fourier optics?

Run: python -m studies.fdtd_rectangle_gradients --output /tmp/rectangle-gradients.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

import wavebit

PERIOD = [1.3, .05]
THICKNESS = [.2]


def scene(p):
    return wavebit.Stack.from_structure([dict(background=1., shapes=[
        dict(type='rectangle', lx=p[0], ly=PERIOD[1], cx=p[1], material=2.25)])],
        THICKNESS, top=1., bot=1.21, period=PERIOD)


def amplitudes(result, xp=np, order=1):
    return xp.stack([getattr(column, name).reshape(-1)[order-1:order+2]
                     for column, name in ((result.s_inc, 'r_s'), (result.s_inc, 't_s'),
                                           (result.p_inc, 'r_p'), (result.p_inc, 't_p'))])


def response(p, resolution=40, cells=1., backend='jax', duration=65):
    from wavebit.backends.time_domain import time_loop_backend
    result = wavebit.fdtd(scene(p), wavebit.Incidence(wavelength=1.), backend=backend,
                          resolution=resolution, n_pml=resolution, n_periods=duration,
                          pulse_width=3., truncation_order=(1, 0), align_layers=True,
                          shape_smoothing=[cells/resolution, 0.], diffraction_method='directional')
    return amplitudes(result, time_loop_backend(backend).xp)


def rcwa_reference(p, order=32):
    result = wavebit.rcwa(stack=scene(p), incidence=wavebit.Incidence(wavelength=1.),
                          backend=0, truncation_order=(order, 0), normal_grid=256).solve()
    return amplitudes(result, order=order)


def measure():
    import jax
    jax.config.update('jax_enable_x64', True)
    rows = []
    references = {}
    for centre in (.071, .083, .095):
        p = np.array([.43, centre])
        target, fine = rcwa_reference(p), rcwa_reference(p, 48)
        def finite(step, order=32):
            return np.stack([(rcwa_reference(p+v, order)-rcwa_reference(p-v, order))/(2*step)
                             for v in np.eye(2)*step], axis=-1)
        derivative, refined = finite(1e-5), finite(5e-6)
        spectral_refined = finite(1e-5, 48)
        translation = -2j*np.pi*np.array([-1, 0, 1])/PERIOD[0]*target
        np.testing.assert_allclose(derivative[..., 1], translation, atol=3e-8)
        references[centre] = (target, derivative, translation,
                             float(np.max(abs(target-fine))), float(np.max(abs(derivative-spectral_refined))),
                             float(np.max(abs(derivative-refined))))
    for resolution in (20, 30, 40):
        for cells in (0., 1.):
            f = lambda p: response(p, resolution, cells)
            evaluate = jax.jit(lambda p: (f(p), jax.jacfwd(f)(p)))
            for centre in (.071, .083, .095):
                p = np.array([.43, centre])
                actual, derivative = map(np.asarray, evaluate(p))
                target, target_derivative, translation, order_change, derivative_change, step_change = references[centre]
                row = dict(resolution=resolution, smoothing_cells=cells, centre=centre,
                           amplitude_error=float(np.max(abs(actual-target))),
                           width_error=float(np.max(abs(derivative[..., 0]-target_derivative[..., 0]))),
                           position_error=float(np.max(abs(derivative[..., 1]-translation))),
                           reference_order_change=order_change,
                           reference_derivative_order_change=derivative_change,
                           reference_step_change=step_change)
                rows.append(row)
                print(json.dumps(row), flush=True)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(measure(), indent=2)+'\n')
