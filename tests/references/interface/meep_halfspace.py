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


def generate(output, resolution=80, duration=50., parameter='halfspace', smoothing=0.,
             subpixel_tol=1e-8, transverse_cells=1):
    if not np.isfinite(smoothing) or smoothing < 0 or (smoothing and parameter != 'thickness'):
        raise ValueError('Smoothing must be nonnegative and applies only to thickness perturbations.')
    if transverse_cells < 1 or not np.isfinite(subpixel_tol) or subpixel_tol <= 0:
        raise ValueError('Transverse cell count and averaging tolerance must be positive.')
    import meep as mp
    mp.verbosity(0)
    band = np.array([.98, 1.02])
    background = np.array([1.2, 1.1, 1.6, 1.15])
    parameters = background.copy() if parameter == 'halfspace' else np.array([.173])
    names = ['epsilon_top', 'mu_top', 'epsilon_bottom', 'mu_bottom'] if parameter == 'halfspace' else ['film_thickness']
    epsilon, mu, depth = 2.25, 1.12, .2 if parameter == 'halfspace' else .4
    dt = .4/resolution
    zr, zt = -1., 1.2
    # The public material-function averaging check uses a periodic 3D cell.
    # A uniform transverse plane wave needs only one cell in each transverse axis.
    side = transverse_cells/resolution if smoothing else 0.
    dimension = 3 if smoothing else 1
    boundaries = [mp.PML(1., direction=mp.Z)] if smoothing else [mp.PML(1.)]
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
        material_options = {}
        if actual and smoothing:
            if smoothing/2 >= min(film_depth, depth-film_depth):
                raise ValueError('The graded interface must stay inside the fixed stack.')
            def ramp(position):
                if position.z < 0:
                    return top
                fraction = np.clip((position.z-film_depth)/smoothing+.5, 0., 1.)
                return mp.Medium(epsilon=epsilon+(coefficients[2]-epsilon)*fraction,
                                 mu=mu+(coefficients[3]-mu)*fraction)
            # Global profile avoids buried sharp object faces inside the ramp.
            # These are documented public API calls, not implementation access.
            ramp.do_averaging = True
            geometry = []
            material_options = dict(material_function=ramp, subpixel_tol=subpixel_tol,
                                    extra_materials=[top, mp.Medium(epsilon=epsilon, mu=mu),
                                                     mp.Medium(epsilon=coefficients[2], mu=coefficients[3])])
        sim = mp.Simulation(cell_size=mp.Vector3(x=side, y=side, z=10.), dimensions=dimension, resolution=resolution,
                            boundary_layers=boundaries, Courant=.4, default_material=top,
                            geometry=geometry, eps_averaging=True, k_point=mp.Vector3(), **material_options,
                            sources=[mp.Source(mp.CustomSource(pulse, end_time=12.), component=mp.Ex,
                                               center=mp.Vector3(z=-3.), size=mp.Vector3(side, side, 0.))])
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
    metadata = dict(layer_smoothing=smoothing, subpixel_tol=subpixel_tol, dimensions=dimension,
                    transverse_cells=transverse_cells, transverse_period=side) if smoothing else {}
    np.savez_compressed(output, **metadata, version=mp.__version__, resolution=resolution,
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
    parser.add_argument('--smoothing', type=float, default=0., help='Full width of a linear epsilon/mu interface ramp.')
    parser.add_argument('--subpixel-tol', type=float, default=1e-8, help='Public material-function averaging tolerance.')
    parser.add_argument('--transverse-cells', type=int, default=1, help='Periodic cells along each transverse axis for a graded profile.')
    args = parser.parse_args()
    generate(args.output, args.resolution, args.duration, args.parameter, args.smoothing,
             args.subpixel_tol, args.transverse_cells)
