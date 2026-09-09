"""Separate graded-profile bias from cell error in averaged rotated-rectangle derivatives."""
import argparse
import json
from pathlib import Path

import numpy as np


def density(parameters, smoothing=.05, grid=256, clip=True, return_jacobian=False):
    """Independent Fourier-series rectangle convolved with a translation box.

    This reference uses sinc transforms, not FDTD polygon clipping or moments.
    Rows are y, columns x; samples are cell-centred. Clipping bounds the series
    for Meep MaterialGrid; grid refinement controls that approximation.
    Optional analytic derivatives have final axis (angle, width). At clipping
    knots the chosen derivative is zero; finite differences crossing a knot
    need not match that local derivative.
    """
    angle, width = parameters
    period, height = .6, .18
    frequency = 2*np.pi*np.fft.fftfreq(grid, d=period/grid)
    kx, ky = frequency[None, :], frequency[:, None]
    u, v = np.cos(angle)*kx+np.sin(angle)*ky, -np.sin(angle)*kx+np.cos(angle)*ky
    hx, hy = np.broadcast_to(smoothing, (2,))
    coefficients = (width*height/period**2*np.sinc(u*width/(2*np.pi))*np.sinc(v*height/(2*np.pi))
                    *np.sinc(kx*hx/(2*np.pi))*np.sinc(ky*hy/(2*np.pi))
                    *np.exp(-1j*(kx*(.071+period/2)+ky*(-.047+period/2)))
                    *np.exp(1j*(kx+ky)*.5*period/grid))
    samples = np.fft.ifft2(coefficients*grid**2).real
    result = np.clip(samples, 0., 1.) if clip else samples
    if not return_jacobian:
        return result
    def dsinc(z):
        small = abs(z) < 1e-3
        safe = np.where(small, 1., z)
        return np.where(small, -z/3+z**3/30-z**5/840,
                        (safe*np.cos(safe)-np.sin(safe))/safe**2)
    z, t = u*width/2, v*height/2
    kernel = (np.sinc(kx*hx/(2*np.pi))*np.sinc(ky*hy/(2*np.pi))
              *np.exp(-1j*(kx*(.071+period/2)+ky*(-.047+period/2)))
              *np.exp(1j*(kx+ky)*.5*period/grid))
    angle_coeff = width*height/period**2*kernel*(dsinc(z)*v*width/2*np.sinc(t/np.pi)
                                              -np.sinc(z/np.pi)*dsinc(t)*u*height/2)
    # d/dwidth [width*sin(z)/z] = cos(z), with z=u*width/2; safe at u=0.
    width_coeff = height/period**2*kernel*np.cos(z)*np.sinc(t/np.pi)
    jacobian = np.stack([np.fft.ifft2(c*grid**2).real for c in (angle_coeff, width_coeff)], axis=-1)
    if clip:
        jacobian = np.where(((samples > 0) & (samples < 1))[..., None], jacobian, 0.)
    return result, jacobian


def graded_reference(parameters, smoothing=.05, grid=256, order=8):
    import wavebit
    from studies.fdtd_rotation_gradients import amplitudes
    raster = (1+1.25*density(parameters, smoothing, grid))[None, ...]
    result = wavebit.rcwa(structure=raster, thickness=[.2], top=1., bot=1.21, period=[.6, .6],
                          wavelength=1., backend=0, truncation_order=(order, order), factorization='li').solve()
    return amplitudes(result, order=order)


def measure(meep_reference=None):
    import jax
    from studies.fdtd_rotation_gradients import response, finite_difference, rcwa_reference
    jax.config.update('jax_enable_x64', True)
    p = np.array([.37, .3])
    width = .05
    ref = lambda p: graded_reference(p, width)
    value, jac = ref(p), finite_difference(ref, p, 1e-5)
    sharp = rcwa_reference(p, 10, 2048)
    sharp_jac = finite_difference(lambda p: rcwa_reference(p, 10, 2048), p, 1e-5)
    controls = dict(sharp_profile_amplitude_bias=float(np.max(abs(value-sharp))),
                    sharp_profile_derivative_bias=np.max(abs(jac-sharp_jac), axis=0).tolist())
    for label, f in (('grid', lambda p: graded_reference(p, width, 512)),
                     ('order', lambda p: graded_reference(p, width, order=10))):
        controls[label+'_amplitude_change'] = float(np.max(abs(f(p)-value)))
        controls[label+'_derivative_change'] = np.max(abs(finite_difference(f, p, 1e-5)-jac), axis=0).tolist()
    controls['step_change'] = np.max(abs(finite_difference(ref, p, 5e-6)-jac), axis=0).tolist()
    controls['profile_series'] = []
    for grid in (128, 256, 512):
        raw = density(p, width, grid, clip=False)
        bounded = np.clip(raw, 0., 1.)
        controls['profile_series'].append(dict(grid=grid, overshoot=float(np.max(abs(raw-bounded))),
                                               mean_fill_change=float(abs(raw.mean()-bounded.mean()))))
    print(json.dumps(controls), flush=True)
    external = None if meep_reference is None else np.load(meep_reference)
    if external is not None:
        np.testing.assert_array_equal(external['parameters'], p)
        assert external['shape_smoothing'] == width
    rows = []
    for resolution in (20, 30, 40):
        f = lambda p: response(p, resolution, smoothing=width)
        a, j = map(np.asarray, jax.jit(lambda p: (f(p), jax.jacfwd(f)(p)))(p))
        row = dict(resolution=resolution, amplitude_error=float(np.max(abs(a-value))),
                   derivative_error=np.max(abs(j-jac), axis=0).tolist())
        if external is not None:
            row['meep_amplitude_gap'] = float(np.max(abs(a-external['amplitudes'][-1])))
            row['meep_derivative_gap'] = np.max(abs(j-external['jacobians'][-1, -1]), axis=0).tolist()
        print(json.dumps(row), flush=True)
        rows.append(row)
    later = lambda p: response(p, 40, duration=80, smoothing=width)
    late_value, late_jac = map(np.asarray, jax.jit(lambda p: (later(p), jax.jacfwd(later)(p)))(p))
    controls['r40_duration_amplitude_change'] = float(np.max(abs(late_value-a)))
    controls['r40_duration_derivative_change'] = float(np.max(abs(late_jac-j)))
    print(json.dumps(controls), flush=True)
    return dict(parameters=p.tolist(), smoothing=width, controls=controls, rows=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--meep-reference', type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(measure(args.meep_reference), indent=2)+'\n')
