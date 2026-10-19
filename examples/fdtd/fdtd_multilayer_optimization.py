"""Optimize three coating layers and check the final physical design.

Run from the repository root: python -m examples.fdtd.fdtd_multilayer_optimization
Continuous dielectric permittivities are design variables, not a material catalog.
Exported epsilon/thickness arrays follow the public bottom-to-top layer order.
"""
import argparse
import json

import numpy as np

from wavebit import Units, Material
from wavebit.fdtd import periodic as fdtd
from wavebit.rcwa import layer_thicknesses

DEPTH = .8
MINIMUM = .04
BAND = np.array([.8, 1., 1.2])
INITIAL = np.r_[np.log(np.array([.04, .06, .08])/.46), [2.7, 1.6, 3.2]]
BOUNDS = [(-4., 4.)]*3+[(1.1, 4.)]*3


def design(parameters):
    """Decode incident-first parameters into a public bottom-to-top Stack."""
    import jax.numpy as jnp
    thickness = layer_thicknesses(jnp.concatenate((parameters[:3], jnp.zeros(1))), DEPTH, minimum=MINIMUM)
    epsilon = jnp.concatenate((parameters[3:], jnp.array([4.])))
    # Parameters and the independent references list the coatings before the spacer.
    # The public layer axis starts at the substrate, so reverse both arrays here.
    return fdtd.Stack(period=(.05,),bottom=Material(4.),layers=tuple(
        fdtd.Layer(thickness=d,epsilon=e[None,None]) for d,e in zip(thickness[::-1],epsilon[::-1])))


def spectrum(parameters, resolution=40, smoothing_cells=1., band=BAND, duration=65):
    units=Units(length='um')
    problem=fdtd.Problem(stack=design(parameters),units=units,
        excitation=fdtd.Excitation(wavelength=band,sources=(fdtd.PlaneWave(),)),
        pulse=fdtd.Pulse(frequency_sigma_hz=1/(np.sqrt(2)*np.pi*3*units.time_seconds)))
    numerics=fdtd.Numerics(spacing=1/resolution,
        duration_seconds=duration*.5*(float(np.min(band))+float(np.max(band)))*units.time_seconds,
        nominal_layer_thickness=(DEPTH/4,)*4,layer_smoothing=smoothing_cells/resolution,
        align_layers=True,pml_cells=resolution,extraction_method='directional',orders=0)
    return fdtd.run(problem,numerics=numerics,backend='jax').R.sum(axis=(-2,-1))


def sharp_reflection(epsilon, thickness, band):
    """Independent Fresnel recursion; layers run from the top coating to the bottom spacer."""
    indices = np.sqrt(np.r_[1., epsilon, 4.])
    reflection = np.full_like(np.asarray(band), (indices[-2]-indices[-1])/(indices[-2]+indices[-1]), dtype=complex)
    for layer in range(len(thickness)-1, -1, -1):
        interface = (indices[layer]-indices[layer+1])/(indices[layer]+indices[layer+1])
        phase = np.exp(4j*np.pi*indices[layer+1]*thickness[layer]/band)
        reflection = (interface+reflection*phase)/(1+interface*reflection*phase)
    return abs(reflection)**2


def optimize(iterations=30, resolution=40):
    """Supply an exact FDTD gradient to bounded L-BFGS-B through the public API."""
    import jax
    import jax.numpy as jnp
    from scipy.optimize import minimize
    jax.config.update('jax_enable_x64', True)
    evaluate = jax.jit(jax.value_and_grad(lambda p: jnp.mean(spectrum(p, resolution))))
    history = []
    def objective(p):
        value, gradient = evaluate(jnp.asarray(p))
        value, gradient = float(value), np.asarray(gradient)
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            raise RuntimeError('Unresolved FDTD objective or gradient.')
        return value, gradient
    def accepted(p):
        value, gradient = objective(p)
        history.append(dict(parameters=p.tolist(), reflection=value, gradient_norm=float(np.linalg.norm(gradient))))
    accepted(INITIAL)
    result = minimize(objective, INITIAL, method='L-BFGS-B', jac=True, bounds=BOUNDS,
                       callback=accepted, options=dict(maxiter=iterations, ftol=1e-12, gtol=1e-7, maxls=20))
    p = result.x
    stack = design(jnp.asarray(p))
    training = np.asarray(jax.jit(lambda p: spectrum(p, resolution))(jnp.asarray(p)))
    refined = np.asarray(jax.jit(lambda p: spectrum(p, 2*resolution))(jnp.asarray(p)))
    sharp = np.asarray(jax.jit(lambda p: spectrum(p, 2*resolution, smoothing_cells=0.))(jnp.asarray(p)))
    dense_band = np.linspace(BAND.min(), BAND.max(), 81)
    initial_stack = design(jnp.asarray(INITIAL))
    # The oracle is incident-first; Stack arrays and the exported design are bottom-to-top.
    initial_exact = sharp_reflection(np.asarray([l.epsilon for l in initial_stack.layers]).ravel()[::-1],
                                     np.asarray([l.thickness for l in initial_stack.layers])[::-1], dense_band)
    exact = sharp_reflection(np.asarray([l.epsilon for l in stack.layers]).ravel()[::-1],
                             np.asarray([l.thickness for l in stack.layers])[::-1], dense_band)
    return dict(parameters=p.tolist(), layer_order='bottom_to_top',
                thickness=[float(l.thickness) for l in stack.layers],
                epsilon=np.asarray([l.epsilon for l in stack.layers]).ravel().tolist(), wavelengths=BAND.tolist(),
                resolution=resolution, training_reflection=training.tolist(),
                refined_resolution=2*resolution, refined_reflection=refined.tolist(),
                sharp_refined_reflection=sharp.tolist(), history=history,
                continuum_wavelengths=dense_band.tolist(), continuum_reflection=exact.tolist(),
                initial_continuum_mean=float(np.mean(initial_exact)), continuum_mean=float(np.mean(exact)),
                initial_continuum_worst=float(np.max(initial_exact)), continuum_worst=float(np.max(exact)),
                optimizer_success=bool(result.success), optimizer_message=str(result.message),
                iterations=int(result.nit), evaluations=int(result.nfev))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations', type=int, default=30)
    args = parser.parse_args()
    print(json.dumps(optimize(iterations=args.iterations), indent=2))
