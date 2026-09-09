"""Magnetic half-space or film-thickness differences through Meep's public API.

No external implementation is inspected. Each perturbation repeats both the
incident-medium reference and the physical slab experiment on one fixed cell.
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

# External failures must not print implementation-source excerpts.
sys.tracebacklimit = 0
warnings.showwarning = lambda message, category, filename, lineno, file=None, line=None: print(
    f'{category.__name__}: {message}', file=sys.stderr)


def generate(output, resolution=80, duration=50., parameter='halfspace'):
    import meep as mp
    mp.verbosity(0)
    band = np.array([.98, 1.02])
    background = np.array([1.2, 1.1, 1.6, 1.15])
    parameters = background.copy() if parameter == 'halfspace' else np.array([.173])
    names = ['epsilon_top', 'mu_top', 'epsilon_bottom', 'mu_bottom'] if parameter == 'halfspace' else ['film_thickness']
    epsilon, mu, depth = 2.25, 1.12, .2 if parameter == 'halfspace' else .4
    dt = .4/resolution
    zr, zt = -1., 1.2
    def pulse(t):
        return np.exp(-((t-6.)/1.2)**2)*np.sin(2*np.pi*t)
    def fields(p, actual, until):
        coefficients = p if parameter == 'halfspace' else background
        film_depth = depth if parameter == 'halfspace' else p[0]
        top = mp.Medium(epsilon=coefficients[0], mu=coefficients[1])
        geometry = [] if not actual else [
            mp.Block(center=mp.Vector3(z=(film_depth+5)/2), size=mp.Vector3(mp.inf, mp.inf, 5-film_depth),
                     material=mp.Medium(epsilon=coefficients[2], mu=coefficients[3])),
            mp.Block(center=mp.Vector3(z=film_depth/2), size=mp.Vector3(mp.inf, mp.inf, film_depth),
                     material=mp.Medium(epsilon=epsilon, mu=mu))]
        sim = mp.Simulation(cell_size=mp.Vector3(z=10.), dimensions=1, resolution=resolution,
                            boundary_layers=[mp.PML(1.)], Courant=.4, default_material=top,
                            geometry=geometry, eps_averaging=True, k_point=mp.Vector3(),
                            sources=[mp.Source(mp.CustomSource(pulse, end_time=12.), component=mp.Ex,
                                               center=mp.Vector3(z=-3.))])
        monitors = [sim.add_dft_fields([mp.Ex], list(1/band), center=mp.Vector3(z=z),
                                       size=mp.Vector3(), decimation_factor=1) for z in (zr, zt)]
        sim.run(until=until)
        result = np.array([[np.asarray(sim.get_dft_array(m, mp.Ex, i)).item()
                            for i in range(len(band))] for m in monitors])
        sim.reset_meep()
        return result
    def amplitudes(p, until):
        reference, actual = fields(p, False, until), fields(p, True, until)
        # Independent homogeneous Yee dispersion de-embeds E-node phases.
        def kz(epsilon, mu):
            return 2*resolution*np.arcsin(np.sqrt(epsilon*mu)*np.sin(np.pi*dt/band)/(dt*resolution))
        coefficients = p if parameter == 'halfspace' else background
        kt, kb = kz(*coefficients[:2]), kz(*coefficients[2:])
        r = (actual[0]/reference[0]-1)*np.exp(2j*kt*zr)
        t = actual[1]/reference[0]*np.exp(1j*kt*zr-1j*kb*(zt-depth))
        return np.stack((r, t))
    values, jacobians = [], []
    deltas = np.array([1e-3, 5e-4])
    durations = np.array([duration, duration+20])
    for until in durations:
        values.append(amplitudes(parameters, until))
        jacobians.append(np.stack([
            np.stack([(amplitudes(parameters+v, until)-amplitudes(parameters-v, until))/(2*step)
                      for v in np.eye(len(parameters))*step], axis=-1) for step in deltas]))
    np.savez_compressed(output, version=mp.__version__, resolution=resolution,
                        durations=durations, deltas=deltas, parameters=parameters,
                        parameter_names=names, background=background, reference_depth=depth,
                        wavelengths=band, epsilon=epsilon, mu=mu,
                        thickness=depth if parameter == 'halfspace' else parameters[0],
                        source_z=-3., monitor_z=[zr, zt], pml_thickness=1., cell_length=10.,
                        courant=.4, pulse_width=1.2, pulse_peak=6., source_end=12.,
                        amplitudes=np.array(values), jacobians=np.array(jacobians))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resolution', type=int, default=80)
    parser.add_argument('--duration', type=float, default=50.)
    parser.add_argument('--parameter', choices=('halfspace', 'thickness'), default='halfspace')
    args = parser.parse_args()
    generate(args.output, args.resolution, args.duration, args.parameter)
