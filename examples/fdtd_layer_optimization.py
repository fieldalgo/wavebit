"""Optimize a coating thickness with JAX on a fixed FDTD mesh, then refine.

Run from the repository root: python -m examples.fdtd_layer_optimization
All lengths are in wavelengths. The substrate spacer keeps the total depth
and reference planes fixed while the coating's internal boundary moves.
"""
import json

import jax
import jax.numpy as jnp
import numpy as np

from wavebit import Incidence, Stack, fdtd


def reflection(d, resolution):
    stack = Stack.from_structure(jnp.array([[[1.5]], [[2.25]]]), [d, .4-d],
                                  top=1., bot=2.25, period=.05)
    light = Incidence(wavelength=1., polarization=(1., 0.))
    return fdtd(stack, light, backend='jax', resolution=resolution, mesh_thickness=[.2, .2],
                 align_layers=True, n_pml=resolution, n_periods=60, pulse_width=3.,
                 diffraction_method='directional', truncation_order=0).s_inc.R.sum()


def optimize(initial=.12, resolution=40, iterations=20):
    """Bounded descent with backtracking; accept only a measured improvement."""
    jax.config.update('jax_enable_x64', True)
    evaluate = jax.jit(jax.value_and_grad(lambda d: reflection(d, resolution)))
    forward = jax.jit(lambda d: reflection(d, resolution))
    d = float(initial)
    history = []
    for iteration in range(iterations):
        value, gradient = (float(v) for v in evaluate(jnp.asarray(d)))
        if not np.isfinite([value, gradient]).all():
            raise RuntimeError('Unresolved FDTD objective or gradient; increase convergence controls.')
        history.append(dict(iteration=iteration, thickness=d, reflection=value, derivative=gradient))
        if value < 1e-9 or abs(gradient) < 1e-7:
            break
        step = .1
        for _ in range(14):
            candidate = float(np.clip(d-step*gradient, .03, .37))
            trial = float(forward(jnp.asarray(candidate)))
            if np.isfinite(trial) and trial < value-1e-4*gradient*(d-candidate):
                d = candidate
                break
            step *= .5
        else:
            break
    # A finite-grid optimum needs an independent physical check and refinement.
    final = float(forward(jnp.asarray(d)))
    refined = float(jax.jit(lambda d: reflection(d, 2*resolution))(jnp.asarray(d)))
    exact_d = 1/(4*np.sqrt(1.5))
    a = (1-np.sqrt(1.5))/(1+np.sqrt(1.5))
    b = (np.sqrt(1.5)-1.5)/(np.sqrt(1.5)+1.5)
    phase = np.exp(4j*np.pi*np.sqrt(1.5)*d)
    exact_r = abs((a+b*phase)/(1+a*b*phase))**2
    return dict(history=history, thickness=d, resolution=resolution, reflection=final,
                refined_resolution=2*resolution, refined_reflection=refined,
                continuum_reflection=float(exact_r), quarter_wave_thickness=float(exact_d))


if __name__ == '__main__':
    print(json.dumps(optimize(), indent=2))
