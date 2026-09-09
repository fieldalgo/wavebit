"""Independent inclusion-exclusion Fourier optics for two painted rectangles."""
import argparse
import json
from pathlib import Path
import time

import numpy as np


def local_rectangles(p):
    """Co-rotating primitives with an explicitly rectangular intersection."""
    _, separation, width = p
    return [(-.04, -.015, .3, .18), (separation, .025, width, .22)]


def density(p, smoothing=.05, grid=256, clip=True):
    """Independent sinc transforms of the two shapes minus their intersection.

    The later rectangle overwrites the first. The geometric overlap is a
    rectangle in their common local frame; no native painter builds this data.
    """
    angle = p[0]
    first, second = [np.array(v) for v in local_rectangles(p)]
    low = np.maximum(first[:2]-first[2:]/2, second[:2]-second[2:]/2)
    high = np.minimum(first[:2]+first[2:]/2, second[:2]+second[2:]/2)
    overlap = np.r_[(low+high)/2, np.maximum(high-low, 0.)]
    k = 2*np.pi*np.fft.fftfreq(grid, d=.6/grid)
    kx, ky = k[None, :], k[:, None]
    co, si = np.cos(angle), np.sin(angle)
    u, v = co*kx+si*ky, -si*kx+co*ky
    coeff = np.zeros((grid, grid), complex)
    for rectangle, contrast in [(first, .8), (second, 1.25), (overlap, -.8)]:
        cx, cy, width, height = rectangle
        coeff += contrast*width*height/.6**2*np.sinc(u*width/(2*np.pi))*np.sinc(v*height/(2*np.pi))*np.exp(-1j*(u*cx+v*cy))
    hx, hy = np.broadcast_to(smoothing, (2,))
    coeff *= (np.sinc(kx*hx/(2*np.pi))*np.sinc(ky*hy/(2*np.pi))
              *np.exp(-1j*(kx*(.025+.3)+ky*(-.025+.3)))
              *np.exp(1j*(kx+ky)*.6/(2*grid)))
    raw = np.fft.ifft2(coeff*grid**2).real/1.25
    return np.clip(raw, 0., 1.) if clip else raw


def scene(p, backend=0):
    import wavebit
    from wavebit.backends.time_domain import time_loop_backend
    xp = time_loop_backend(backend).xp
    angle = time_loop_backend(backend).array(p[0])
    co, si = xp.cos(angle), xp.sin(angle)
    shapes = []
    for (cx, cy, width, height), epsilon in zip(local_rectangles(p), [1.8, 2.25]):
        shapes.append(dict(lx=width, ly=height, cx=.025+co*cx-si*cy, cy=-.025+si*cx+co*cy,
                           angle=angle, material=epsilon))
    return wavebit.Stack.from_structure([dict(background=1., shapes=shapes)],
        [.2], top=1., bot=1.21, period=[.6, .6])


def response(p, resolution=20, backend='jax', duration=60, smoothing=.05):
    import wavebit
    from studies.fdtd_rotation_gradients import amplitudes
    from wavebit.backends.time_domain import time_loop_backend
    result = wavebit.fdtd(scene(p, backend), wavebit.Incidence(wavelength=1.), backend=backend,
        resolution=resolution, n_pml=resolution, n_periods=duration, pulse_width=2.5,
        truncation_order=0, align_layers=True, shape_integration='area', shape_smoothing=smoothing,
        diffraction_method='directional')
    return amplitudes(result, time_loop_backend(backend).xp)


def reference(p, smoothing=.05, grid=256, order=8):
    import wavebit
    from studies.fdtd_rotation_gradients import amplitudes
    stack = wavebit.Stack.from_structure((1+1.25*density(p, smoothing, grid))[None, ...],
        [.2], top=1., bot=1.21, period=[.6, .6])
    result = wavebit.rcwa(stack=stack, incidence=wavebit.Incidence(wavelength=1.), backend=0,
        truncation_order=(order, order), factorization='li').solve()
    return amplitudes(result, order=order)


def measure():
    import jax
    from studies.fdtd_rotation_gradients import finite_difference
    jax.config.update('jax_enable_x64', True)
    p = np.array([.37, .055, .22])
    value, jac = reference(p), finite_difference(reference, p, 1e-5)
    controls = {}
    for name, f in [('grid', lambda p: reference(p, grid=512)),
                    ('order', lambda p: reference(p, order=10))]:
        controls[name+'_amplitude_change'] = float(np.max(abs(f(p)-value)))
        controls[name+'_derivative_change'] = np.max(abs(finite_difference(f, p, 1e-5)-jac), axis=0).tolist()
    controls['step_change'] = np.max(abs(finite_difference(reference, p, 5e-6)-jac), axis=0).tolist()
    print(json.dumps(controls), flush=True)
    rows = []
    for resolution in (20, 30, 40):
        f = lambda p: response(p, resolution)
        start = time.perf_counter()
        a, j = map(np.asarray, jax.jit(lambda p: (f(p), jax.jacfwd(f)(p)))(p))
        if not (np.isfinite(a).all() and np.isfinite(j).all()):
            raise RuntimeError('Nonfinite scattering or derivatives invalidate this study.')
        row = dict(resolution=resolution, amplitude_error=float(np.max(abs(a-value))),
            derivative_error=np.max(abs(j-jac), axis=0).tolist(), elapsed_seconds=time.perf_counter()-start,
            energy_error=float(max(abs(sum(abs(a[[0, 1]])**2)+1.1*sum(abs(a[[2, 3]])**2)-1),
                                   abs(sum(abs(a[[4, 5]])**2)+1.1*sum(abs(a[[6, 7]])**2)-1))))
        rows.append(row)
        print(json.dumps(row), flush=True)
    f = lambda p: response(p, 40, duration=80)
    late_a, late_j = map(np.asarray, jax.jit(lambda p: (f(p), jax.jacfwd(f)(p)))(p))
    controls['duration_amplitude_change'] = float(np.max(abs(a-late_a)))
    controls['duration_derivative_change'] = np.max(abs(j-late_j), axis=0).tolist()
    print(json.dumps(controls), flush=True)
    return dict(parameters=p.tolist(), smoothing=.05, controls=controls, rows=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(measure(), indent=2, allow_nan=False)+'\n')
