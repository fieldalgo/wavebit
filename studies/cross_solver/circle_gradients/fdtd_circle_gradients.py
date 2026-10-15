"""Disk diameter derivatives: exact cell geometry versus independent Fourier optics."""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import j0, j1



from studies._support._fdtd_legacy import legacy_fdtd, legacy_rcwa  # noqa: E402


def density(parameters, smoothing=.05, grid=256, clip=True, return_jacobian=False):
    """Independent disk Fourier transform, optionally convolved with a box.

    Radial integration of J0 gives 2*pi*r*J1(k*r)/k; differentiating with
    respect to the diameter gives pi*r*J0(k*r). Bessel integral identity:
    https://dlmf.nist.gov/10.22.E1 . No FDTD clipping or moments are called.
    """
    radius, period = parameters[0]/2, .6
    k = 2*np.pi*np.fft.fftfreq(grid, d=period/grid)
    kx, ky = k[None, :], k[:, None]
    kr = radius*np.sqrt(kx*kx+ky*ky)
    safe = np.where(kr == 0, 1., kr)
    hx, hy = np.broadcast_to(smoothing, (2,))
    kernel = (np.sinc(kx*hx/(2*np.pi))*np.sinc(ky*hy/(2*np.pi))
        *np.exp(-1j*(kx*(.071+period/2)+ky*(-.047+period/2)))
        *np.exp(1j*(kx+ky)*period/(2*grid)))
    coeff = np.pi*radius**2/period**2*np.where(kr == 0, 1., 2*j1(safe)/safe)*kernel
    raw = np.fft.ifft2(coeff*grid**2).real
    value = np.clip(raw, 0., 1.) if clip else raw
    if not return_jacobian:
        return value
    jac = np.fft.ifft2(np.pi*radius/period**2*j0(kr)*kernel*grid**2).real[..., None]
    if clip:
        jac = np.where(((raw > 0) & (raw < 1))[..., None], jac, 0.)
    return value, jac


def scene(p):
    from wavebit._common.periodic import Stack as wavebit_Stack
    return wavebit_Stack.from_structure([dict(background=1., shapes=[
        dict(type='circle', lx=p[0], cx=.071, cy=-.047, material=2.25)])],
        [.2], top=1., bot=1.21, period=[.6, .6])


def response(p, resolution=30, backend='jax', duration=60, smoothing=.05):
    from wavebit._common.periodic import Incidence as wavebit_Incidence
    from studies.cross_solver.rotation_gradients.fdtd_rotation_gradients import amplitudes
    from wavebit.fdtd._periodic.runtime import time_loop_backend
    result = legacy_fdtd(scene(p), wavebit_Incidence(wavelength=1.), backend=backend,
        resolution=resolution, n_pml=resolution, n_periods=duration, pulse_width=2.5,
        truncation_order=0, align_layers=True, shape_integration='area', shape_smoothing=smoothing,
        diffraction_method='directional')
    return amplitudes(result, time_loop_backend(backend).xp)


def reference(p, smoothing=.05, grid=256, order=8):
    from wavebit._common.periodic import Incidence as wavebit_Incidence
    from wavebit._common.periodic import Stack as wavebit_Stack
    from studies.cross_solver.rotation_gradients.fdtd_rotation_gradients import amplitudes
    incidence = wavebit_Incidence(wavelength=1.)
    if smoothing:
        stack = wavebit_Stack.from_structure((1+1.25*density(p, smoothing, grid))[None, ...],
            [.2], top=1., bot=1.21, period=[.6, .6])
        options = dict(factorization='li')
    else:
        stack, options = scene(p), dict(shape_nvf_grid_size=grid)
    result = legacy_rcwa(stack=stack, incidence=incidence, backend='numpy',
        truncation_order=(order, order), **options).solve()
    return amplitudes(result, order=order)


def measure(smoothing=.05):
    import jax
    from studies.cross_solver.rotation_gradients.fdtd_rotation_gradients import finite_difference
    jax.config.update('jax_enable_x64', True)
    p = np.array([.3])
    ref = lambda p: reference(p, smoothing=smoothing)
    value, jac = ref(p), finite_difference(ref, p, 1e-5)
    controls = {}
    for name, f in [('grid', lambda p: reference(p, smoothing, grid=512)),
                    ('order', lambda p: reference(p, smoothing, order=10))]:
        controls[name+'_amplitude_change'] = float(np.max(abs(f(p)-value)))
        controls[name+'_derivative_change'] = float(np.max(abs(finite_difference(f, p, 1e-5)-jac)))
    controls['step_change'] = float(np.max(abs(finite_difference(ref, p, 5e-6)-jac)))
    print(json.dumps(controls), flush=True)
    rows = []
    for resolution in (20, 30, 40):
        f = lambda p: response(p, resolution, smoothing=smoothing)
        a, j = map(np.asarray, jax.jit(lambda p: (f(p), jax.jacfwd(f)(p)))(p))
        if not (np.isfinite(a).all() and np.isfinite(j).all()):
            raise RuntimeError('Nonfinite scattering or derivatives invalidate this study.')
        row = dict(resolution=resolution, amplitude_error=float(np.max(abs(a-value))),
            derivative_error=float(np.max(abs(j-jac))),
            energy_error=float(max(abs(sum(abs(a[[0, 1]])**2)+1.1*sum(abs(a[[2, 3]])**2)-1),
                                   abs(sum(abs(a[[4, 5]])**2)+1.1*sum(abs(a[[6, 7]])**2)-1))))
        rows.append(row)
        print(json.dumps(row), flush=True)
    f = lambda p: response(p, 40, duration=80, smoothing=smoothing)
    late_a, late_j = map(np.asarray, jax.jit(lambda p: (f(p), jax.jacfwd(f)(p)))(p))
    controls['duration_amplitude_change'] = float(np.max(abs(a-late_a)))
    controls['duration_derivative_change'] = float(np.max(abs(j-late_j)))
    if smoothing:
        sharp = reference(p, 0., grid=512, order=10)
        sharp_j = finite_difference(lambda p: reference(p, 0., grid=512, order=10), p, 1e-5)
        controls['sharp_profile_amplitude_bias'] = float(np.max(abs(value-sharp)))
        controls['sharp_profile_derivative_bias'] = float(np.max(abs(jac-sharp_j)))
    print(json.dumps(controls), flush=True)
    return dict(parameters=p.tolist(), smoothing=smoothing, controls=controls, rows=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoothing', type=float, default=.05)
    args = parser.parse_args()
    args.output.write_text(json.dumps(measure(args.smoothing), indent=2, allow_nan=False)+'\n')
