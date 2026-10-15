"""Optimize a coating thickness with JAX on a fixed FDTD mesh, then refine.

Run from the repository root: python -m examples.fdtd.fdtd_layer_optimization
All lengths are in wavelengths. The substrate spacer keeps the total depth
and reference planes fixed while the coating's internal boundary moves.
"""
import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np

from wavebit import Units, Material
from wavebit.fdtd import periodic as fdtd


def reflection(d, resolution, smoothing_cells=0.):
    # Public stacks run bottom to top: the spacer touches the substrate.
    units=Units(length='um')
    stack=fdtd.Stack(period=(.05,),bottom=Material(2.25),layers=(
        fdtd.Layer(thickness=.4-d,epsilon=jnp.array([[2.25]])),
        fdtd.Layer(thickness=d,epsilon=jnp.array([[1.5]]))))
    problem=fdtd.Problem(stack=stack,units=units,
        excitation=fdtd.Excitation(wavelength=1.,sources=(fdtd.PlaneWave(),)),
        pulse=fdtd.Pulse(width_seconds=3*units.time_seconds))
    numerics=fdtd.Numerics(spacing=1/resolution,duration_seconds=60*units.time_seconds,
        nominal_layer_thickness=(.2,.2),align_layers=True,layer_smoothing=smoothing_cells/resolution,
        pml_cells=resolution,extraction_method='directional',orders=0)
    return fdtd.run(problem,numerics=numerics,backend='jax').R.sum()


def optimize(initial=.12, resolution=40, iterations=20, smoothing_cells=0.):
    """Bounded descent with backtracking; accept only a measured improvement."""
    jax.config.update('jax_enable_x64', True)
    half_width = smoothing_cells/(2*resolution)
    lower, upper = max(.03, half_width), min(.37, .4-half_width)
    if (not np.isfinite(smoothing_cells) or smoothing_cells < 0
            or lower >= upper or not lower <= initial <= upper):
        raise ValueError('Smoothing must be nonnegative with a nonempty design range containing the initial thickness.')
    evaluate = jax.jit(jax.value_and_grad(lambda d: reflection(d, resolution, smoothing_cells)))
    forward = jax.jit(lambda d: reflection(d, resolution, smoothing_cells))
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
            candidate = float(np.clip(d-step*gradient, lower, upper))
            trial = float(forward(jnp.asarray(candidate)))
            if np.isfinite(trial) and trial < value-1e-4*gradient*(d-candidate):
                d = candidate
                break
            step *= .5
        else:
            break
    # A finite-grid optimum needs an independent physical check and refinement.
    final = float(forward(jnp.asarray(d)))
    # Shrink the averaging width with the mesh toward the sharp-interface limit.
    refined = float(jax.jit(lambda d: reflection(d, 2*resolution, smoothing_cells))(jnp.asarray(d)))
    sharp = float(jax.jit(lambda d: reflection(d, 2*resolution))(jnp.asarray(d))) if smoothing_cells else refined
    exact_d = 1/(4*np.sqrt(1.5))
    a = (1-np.sqrt(1.5))/(1+np.sqrt(1.5))
    b = (np.sqrt(1.5)-1.5)/(np.sqrt(1.5)+1.5)
    phase = np.exp(4j*np.pi*np.sqrt(1.5)*d)
    exact_r = abs((a+b*phase)/(1+a*b*phase))**2
    return dict(history=history, thickness=d, resolution=resolution, reflection=final,
                smoothing_cells=smoothing_cells, layer_smoothing=smoothing_cells/resolution,
                refined_resolution=2*resolution, refined_reflection=refined,
                refined_smoothing=smoothing_cells/(2*resolution), sharp_refined_reflection=sharp,
                continuum_reflection=float(exact_r), quarter_wave_thickness=float(exact_d))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoothing-cells', type=float, default=0., help='Static interface averaging width in training cells.')
    args = parser.parse_args()
    print(json.dumps(optimize(smoothing_cells=args.smoothing_cells), indent=2))
