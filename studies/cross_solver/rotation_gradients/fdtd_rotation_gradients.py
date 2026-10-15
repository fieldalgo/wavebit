"""How do area-integrated FDTD rotation and width derivatives approach analytic RCWA?"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
import argparse
import json
from pathlib import Path

import numpy as np

from wavebit._common.periodic import Incidence as wavebit_Incidence

from wavebit._common.periodic import Stack as wavebit_Stack


from studies._support._fdtd_legacy import legacy_fdtd, legacy_rcwa  # noqa: E402


def scene(p):
    return wavebit_Stack.from_structure([dict(background=1., shapes=[
        dict(type='rectangle', angle=p[0], lx=p[1], ly=.18, cx=.071, cy=-.047, material=2.25)])],
        [.2], top=1., bot=1.21, period=[.6, .6])


def amplitudes(result, xp=np, order=0):
    return xp.stack([getattr(column, name)[order, order] for column in (result.s_inc, result.p_inc)
                     for name in ('r_s', 'r_p', 't_s', 't_p')])


def response(p, resolution=24, backend='jax', duration=60, smoothing=0.):
    from wavebit.fdtd._periodic.runtime import time_loop_backend
    result = legacy_fdtd(scene(p), wavebit_Incidence(wavelength=1.), backend=backend,
                          resolution=resolution, n_pml=resolution, n_periods=duration, pulse_width=2.5,
                          truncation_order=0, align_layers=True, shape_integration='area', shape_smoothing=smoothing,
                          diffraction_method='directional')
    return amplitudes(result, time_loop_backend(backend).xp)


def rcwa_reference(p, order=8, normal_grid=256):
    result = legacy_rcwa(stack=scene(p), incidence=wavebit_Incidence(wavelength=1.), backend='numpy',
                          truncation_order=(order, order), shape_nvf_grid_size=normal_grid).solve()
    return amplitudes(result, order=order)


def finite_difference(f, p, step):
    return np.stack([(f(p+d)-f(p-d))/(2*step) for d in np.eye(len(p))*step], axis=-1)


def measure():
    import jax
    jax.config.update('jax_enable_x64', True)
    p = np.array([.37, .3])
    target = rcwa_reference(p)
    derivative = finite_difference(rcwa_reference, p, 1e-5)
    fine = rcwa_reference(p, 10)
    fine_derivative = finite_difference(lambda p: rcwa_reference(p, 10), p, 1e-5)
    small_step = finite_difference(rcwa_reference, p, 5e-6)
    normal_fine = rcwa_reference(p, normal_grid=512)
    normal_derivative = finite_difference(lambda p: rcwa_reference(p, normal_grid=512), p, 1e-5)
    controls = dict(order_amplitude_change=float(np.max(abs(target-fine))),
                    order_derivative_change=np.max(abs(derivative-fine_derivative), axis=0).tolist(),
                    step_change=np.max(abs(derivative-small_step), axis=0).tolist(),
                    normal_amplitude_change=float(np.max(abs(target-normal_fine))),
                    normal_derivative_change=np.max(abs(derivative-normal_derivative), axis=0).tolist())
    print(json.dumps(controls), flush=True)
    rows = []
    for resolution in (16, 24, 32, 40):
        f = lambda p: response(p, resolution)
        value, jac = map(np.asarray, jax.jit(lambda p: (f(p), jax.jacfwd(f)(p)))(p))
        row = dict(resolution=resolution, amplitude_error=float(np.max(abs(value-target))),
                   derivative_error=np.max(abs(jac-derivative), axis=0).tolist(),
                   energy_error=float(max(abs(sum(abs(value[[0, 1]])**2)+1.1*sum(abs(value[[2, 3]])**2)-1),
                                          abs(sum(abs(value[[4, 5]])**2)+1.1*sum(abs(value[[6, 7]])**2)-1))))
        rows.append(row)
        print(json.dumps(row), flush=True)
    # Higher normal-field sampling separates the reference's angular sampling
    # error from the finite-grid FDTD error; it does not enforce monotonicity.
    high = []
    for grid in (1024, 2048):
        f = lambda p: rcwa_reference(p, 10, normal_grid=grid)
        high.append((f(p), finite_difference(f, p, 1e-5)))
    controls.update(high_normal_amplitude_change=float(np.max(abs(high[1][0]-high[0][0]))),
                    high_normal_derivative_change=np.max(abs(high[1][1]-high[0][1]), axis=0).tolist(),
                    high_reference_r40_amplitude_error=float(np.max(abs(value-high[1][0]))),
                    high_reference_r40_derivative_error=np.max(abs(jac-high[1][1]), axis=0).tolist())
    later = lambda p: response(p, 40, duration=80)
    late_value, late_jac = map(np.asarray, jax.jit(lambda p: (later(p), jax.jacfwd(later)(p)))(p))
    controls.update(r40_duration_amplitude_change=float(np.max(abs(late_value-value))),
                    r40_duration_derivative_change=float(np.max(abs(late_jac-jac))))
    print(json.dumps(controls), flush=True)
    return dict(parameters=p.tolist(), controls=controls, rows=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(measure(), indent=2)+'\n')
