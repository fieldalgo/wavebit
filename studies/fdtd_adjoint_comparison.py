"""Separate Meep adjoint repeatability, discrete derivatives and continuum accuracy."""
import argparse
import json
from pathlib import Path

import numpy as np


def reference_controls(data, previous):
    a, j, f = data['amplitudes'], data['adjoint_jacobians'], data['finite_jacobians']
    indices = data['original_channel_indices']
    return dict(amplitude_time_change=float(np.max(abs(a[1]-a[0]))),
                adjoint_time_change=float(np.max(abs(j[1]-j[0]))),
                finite_time_change=float(np.max(abs(f[1]-f[0]))),
                finite_step_change=np.max(abs(f[:, :, 1]-f[:, :, 0]), axis=(0, 1)).tolist(),
                mapping_change=np.max(abs(data['mapping_jacobians']-j[:, :, None, :]), axis=(0, 1, 2)).tolist(),
                previous_amplitude_change=float(np.max(abs(a-previous['amplitudes'][:, indices]))),
                previous_derivative_change=float(np.max(abs(f-previous['jacobians'][:, :, indices].transpose(0, 2, 1, 3)))),
                adjoint_finite_gap=np.max(abs(j[-1]-f[-1, :, -1]), axis=0).tolist(),
                adjoint_finite_gap_by_channel=abs(j[-1]-f[-1, :, -1]).tolist())


def measure():
    import jax
    from studies.fdtd_rotation_gradients import response, finite_difference
    from studies.fdtd_rotation_smoothing import graded_reference
    jax.config.update('jax_enable_x64', True)
    base = Path(__file__).resolve().parents[1]/'tests/references'
    p, indices = np.array([.37, .3]), np.array([2, 7])
    target = graded_reference(p)[indices]
    derivative = finite_difference(graded_reference, p, 1e-5)[indices]
    rows = []
    for resolution, suffix in ((20, '_coarse'), (40, '')):
        data = dict(np.load(base/f'golden_meep_adjoint_rotation{suffix}.npz'))
        previous = dict(np.load(base/f'golden_meep_rotation_smoothing{suffix}.npz'))
        controls = reference_controls(data, previous)
        f = lambda p: response(p, resolution=resolution, smoothing=.05)
        a, j = map(np.asarray, jax.jit(lambda p: (f(p), jax.jacfwd(f)(p)))(p))
        a, j = a[indices], j[indices]
        adjoint, finite = data['adjoint_jacobians'][-1], data['finite_jacobians'][-1, :, -1]
        row = dict(resolution=resolution, controls=controls,
                   wavebit_amplitude_error=float(np.max(abs(a-target))),
                   wavebit_derivative_error=np.max(abs(j-derivative), axis=0).tolist(),
                   meep_amplitude_error=float(np.max(abs(data['amplitudes'][-1]-target))),
                   meep_finite_error=np.max(abs(finite-derivative), axis=0).tolist(),
                   meep_adjoint_error=np.max(abs(adjoint-derivative), axis=0).tolist(),
                   wavebit_meep_finite_gap=np.max(abs(j-finite), axis=0).tolist(),
                   wavebit_meep_adjoint_gap=np.max(abs(j-adjoint), axis=0).tolist())
        rows.append(row)
        print(json.dumps(row), flush=True)
    return dict(parameters=p.tolist(), channels=['t_ss', 't_pp'], smoothing=.05, rows=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(measure(), indent=2)+'\n')
