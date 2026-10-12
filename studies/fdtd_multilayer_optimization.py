"""How do constrained multilayer gradients and optimized spectra refine?

Run: python -m studies.fdtd_multilayer_optimization --design artifacts/design.json --output artifacts/multilayer-check.json
The continuum reference independently propagates tangential E/H through layers.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def sharp_stack(epsilon, thickness, wavelengths, *, top=1., bottom=4.):
    """Independent normal-incidence characteristic matrices, exp(-i omega t).

    Return complex r/t and energy R/T at the two stack faces. Only positive
    real dielectric materials are used in this reference, so roots are positive.
    """
    epsilon, thickness = np.asarray(epsilon), np.asarray(thickness)
    wavelengths = np.atleast_1d(wavelengths)
    admittance_top, admittance_bottom = np.sqrt(top), np.sqrt(bottom)
    matrix = np.broadcast_to(np.eye(2, dtype=complex), (len(wavelengths), 2, 2)).copy()
    for eps, length in zip(epsilon, thickness):
        index = np.sqrt(eps)
        phase = 2*np.pi*index*length/wavelengths
        c, s = np.cos(phase), -1j*np.sin(phase)
        propagation = np.stack((np.stack((c, s/index), axis=-1),
                                np.stack((index*s, c), axis=-1)), axis=-2)
        matrix = matrix @ propagation
    electric = matrix[:, 0, 0]+admittance_bottom*matrix[:, 0, 1]
    magnetic = matrix[:, 1, 0]+admittance_bottom*matrix[:, 1, 1]
    denominator = admittance_top*electric+magnetic
    r = (admittance_top*electric-magnetic)/denominator
    t = 2*admittance_top/denominator
    return dict(r=r, t=t, R=abs(r)**2, T=admittance_bottom/admittance_top*abs(t)**2)


def decode_numpy(parameters, depth=.8, minimum=.04):
    """Independent normalized exponential coordinates for the study's design."""
    p = np.asarray(parameters)
    weights = np.exp(np.r_[p[:3], 0.])
    lengths = minimum+(depth-4*minimum)*weights/weights.sum()
    return np.r_[p[3:], 4.], lengths


def graded_stack(epsilon, thickness, wavelengths, width, *, slices=128):
    """Integrate the independent linear profile, including overlapping ramps."""
    if width == 0:
        return sharp_stack(epsilon, thickness, wavelengths)
    epsilon, thickness = np.asarray(epsilon), np.asarray(thickness)
    faces, depth = np.cumsum(thickness)[:-1], np.sum(thickness)
    if width <= 0 or np.any(faces-width/2 < 0) or np.any(faces+width/2 > depth):
        raise ValueError('The positive-width reference profile must stay inside the stack.')
    cuts = np.unique(np.r_[0., faces-width/2, faces+width/2, depth])
    materials, lengths = [], []
    for left, right in zip(cuts[:-1], cuts[1:]):
        active = np.any(abs((left+right)/2-faces) < width/2)
        count = slices if active else 1
        positions = left+(right-left)*(np.arange(count)+.5)/count
        fills = np.clip((positions[:, None]-faces[None, :])/width+.5, 0., 1.)
        materials.extend(epsilon[0]+fills @ np.diff(epsilon))
        lengths.extend(np.full(count, (right-left)/count))
    return sharp_stack(materials, lengths, wavelengths)


def continuum(parameters, band, smoothing=0., *, slices=128):
    epsilon, thickness = decode_numpy(parameters)
    return graded_stack(epsilon, thickness, band, smoothing, slices=slices)['R']


def measure(parameters):
    import jax
    import jax.numpy as jnp
    from examples.fdtd_multilayer_optimization import BAND, INITIAL, spectrum
    jax.config.update('jax_enable_x64', True)
    rows = []
    for resolution in (20, 40, 80):
        f = lambda p: spectrum(p, resolution)
        evaluate = jax.jit(lambda p: (f(p), jax.jacfwd(f)(p)))
        for label, p in (('initial', INITIAL), ('optimized', np.asarray(parameters))):
            values, gradient = map(np.asarray, evaluate(jnp.asarray(p)))
            target = continuum(p, BAND)
            graded = continuum(p, BAND, 1/resolution)
            def finite(step, width=0., slices=128):
                return np.stack([(continuum(p+v, BAND, width, slices=slices)-continuum(p-v, BAND, width, slices=slices))/(2*step)
                                 for v in np.eye(len(p))*step], axis=-1)
            derivative, refined = finite(1e-5), finite(5e-6)
            graded_derivative = finite(1e-5, 1/resolution)
            graded_refined = finite(1e-5, 1/resolution, slices=256)
            row = dict(design=label, resolution=resolution,
                       value_error=float(np.max(abs(values-target))),
                       derivative_error=float(np.max(abs(gradient-derivative))),
                       graded_value_error=float(np.max(abs(values-graded))),
                       graded_derivative_error=float(np.max(abs(gradient-graded_derivative))),
                       smoothing_derivative_bias=float(np.max(abs(graded_derivative-derivative))),
                       graded_reference_change=float(np.max(abs(graded_derivative-graded_refined))),
                       oracle_step_change=float(np.max(abs(derivative-refined))),
                       reflection=values.tolist(), continuum_reflection=target.tolist())
            rows.append(row)
            print(json.dumps(row), flush=True)
    # A dense independent band also checks wavelengths absent from the objective.
    dense = np.linspace(.8, 1.2, 81)
    initial = continuum(INITIAL, dense)
    final = continuum(parameters, dense)
    return dict(rows=rows, wavelengths=dense.tolist(), initial_reflection=initial.tolist(),
                optimized_reflection=final.tolist(), initial_mean=float(np.mean(initial)),
                optimized_mean=float(np.mean(final)), initial_worst=float(np.max(initial)),
                optimized_worst=float(np.max(final)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    parameters = json.loads(args.design.read_text())['parameters']
    args.output.write_text(json.dumps(measure(parameters), indent=2)+'\n')
