"""Rotated ellipse shape derivatives against an independent Bessel-series profile."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import j1, jv


from studies._fdtd_legacy import legacy_fdtd, legacy_rcwa  # noqa: E402


def density(parameters, smoothing=.05, grid=256, clip=True, return_jacobian=False):
    """Affine disk Fourier transform and analytic angle/width/height derivatives.

    The radial kernel is 2*J1(q)/q, q**2=(a*u)**2+(b*v)**2. Its derivative
    follows from d[J1(q)/q]/dq=-J2(q)/q, with the DC limits handled explicitly:
    https://dlmf.nist.gov/10.6 . No FDTD geometry functions build this profile.
    """
    angle, width, height = parameters
    a, b, period = width/2, height/2, .6
    k = 2*np.pi*np.fft.fftfreq(grid, d=period/grid)
    kx, ky = k[None, :], k[:, None]
    u, v = np.cos(angle)*kx+np.sin(angle)*ky, -np.sin(angle)*kx+np.cos(angle)*ky
    q = np.sqrt((a*u)**2+(b*v)**2)
    safe = np.where(q == 0, 1., q)
    radial = np.where(q == 0, 1., 2*j1(safe)/safe)
    hx, hy = np.broadcast_to(smoothing, (2,))
    kernel = (np.sinc(kx*hx/(2*np.pi))*np.sinc(ky*hy/(2*np.pi))
        *np.exp(-1j*(kx*(.071+period/2)+ky*(-.047+period/2)))
        *np.exp(1j*(kx+ky)*period/(2*grid)))
    raw = np.fft.ifft2(np.pi*a*b/period**2*radial*kernel*grid**2).real
    value = np.clip(raw, 0., 1.) if clip else raw
    if not return_jacobian:
        return value
    ratio = np.where(q == 0, 1/8, jv(2, safe)/safe**2)
    coeff = [(-2*np.pi*a*b*(a*a-b*b)*u*v*ratio),
             (np.pi*b/2*(radial-2*a*a*u*u*ratio)), (np.pi*a/2*(radial-2*b*b*v*v*ratio))]
    jac = np.stack([np.fft.ifft2(c*kernel*grid**2/period**2).real for c in coeff], axis=-1)
    if clip:
        jac = np.where(((raw > 0) & (raw < 1))[..., None], jac, 0.)
    return value, jac


def scene(p):
    import wavebit
    return wavebit.Stack.from_structure([dict(background=1., shapes=[
        dict(type='ellipse', angle=p[0], lx=p[1], ly=p[2], cx=.071, cy=-.047, material=2.25)])],
        [.2], top=1., bot=1.21, period=[.6, .6])


def response(p, resolution=30, backend='jax', duration=60, smoothing=.05):
    import wavebit
    from studies.fdtd_rotation_gradients import amplitudes
    from wavebit.backends.time_domain import time_loop_backend
    result = legacy_fdtd(scene(p), wavebit.Incidence(wavelength=1.), backend=backend,
        resolution=resolution, n_pml=resolution, n_periods=duration, pulse_width=2.5,
        truncation_order=0, align_layers=True, shape_integration='area', shape_smoothing=smoothing,
        diffraction_method='directional')
    return amplitudes(result, time_loop_backend(backend).xp)


def reference(p, smoothing=.05, grid=256, order=8):
    import wavebit
    from studies.fdtd_rotation_gradients import amplitudes
    if smoothing:
        stack = wavebit.Stack.from_structure((1+1.25*density(p, smoothing, grid))[None, ...],
            [.2], top=1., bot=1.21, period=[.6, .6])
        options = dict(factorization='li')
    else:
        stack, options = scene(p), dict(shape_nvf_grid_size=grid)
    result = legacy_rcwa(stack=stack, incidence=wavebit.Incidence(wavelength=1.), backend='numpy',
        truncation_order=(order, order), **options).solve()
    return amplitudes(result, order=order)


def measure(smoothing=.05):
    import jax
    from studies.fdtd_rotation_gradients import finite_difference
    jax.config.update('jax_enable_x64', True)
    p = np.array([.37, .34, .2])
    ref = lambda p: reference(p, smoothing)
    value, jac = ref(p), finite_difference(ref, p, 1e-5)
    controls = {}
    for name, f in [('grid', lambda p: reference(p, smoothing, grid=512)),
                    ('order', lambda p: reference(p, smoothing, order=10))]:
        controls[name+'_amplitude_change'] = float(np.max(abs(f(p)-value)))
        controls[name+'_derivative_change'] = np.max(abs(finite_difference(f, p, 1e-5)-jac), axis=0).tolist()
    controls['step_change'] = np.max(abs(finite_difference(ref, p, 5e-6)-jac), axis=0).tolist()
    print(json.dumps(controls), flush=True)
    rows = []
    for resolution in (20, 30, 40):
        f = lambda p: response(p, resolution, smoothing=smoothing)
        a, j = map(np.asarray, jax.jit(lambda p: (f(p), jax.jacfwd(f)(p)))(p))
        if not (np.isfinite(a).all() and np.isfinite(j).all()):
            raise RuntimeError('Nonfinite scattering or derivatives invalidate this study.')
        row = dict(resolution=resolution, amplitude_error=float(np.max(abs(a-value))),
            derivative_error=np.max(abs(j-jac), axis=0).tolist(),
            energy_error=float(max(abs(sum(abs(a[[0, 1]])**2)+1.1*sum(abs(a[[2, 3]])**2)-1),
                                   abs(sum(abs(a[[4, 5]])**2)+1.1*sum(abs(a[[6, 7]])**2)-1))))
        rows.append(row)
        print(json.dumps(row), flush=True)
    f = lambda p: response(p, 40, duration=80, smoothing=smoothing)
    late_a, late_j = map(np.asarray, jax.jit(lambda p: (f(p), jax.jacfwd(f)(p)))(p))
    controls['duration_amplitude_change'] = float(np.max(abs(a-late_a)))
    controls['duration_derivative_change'] = np.max(abs(j-late_j), axis=0).tolist()
    if smoothing:
        sharp = reference(p, 0., grid=512, order=10)
        sharp_j = finite_difference(lambda p: reference(p, 0., grid=512, order=10), p, 1e-5)
        controls['sharp_profile_amplitude_bias'] = float(np.max(abs(value-sharp)))
        controls['sharp_profile_derivative_bias'] = np.max(abs(jac-sharp_j), axis=0).tolist()
    print(json.dumps(controls), flush=True)
    return dict(parameters=p.tolist(), smoothing=smoothing, controls=controls, rows=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoothing', type=float, default=.05)
    args = parser.parse_args()
    args.output.write_text(json.dumps(measure(args.smoothing), indent=2, allow_nan=False)+'\n')
