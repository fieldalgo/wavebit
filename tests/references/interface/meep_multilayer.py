"""Independent six-variable coating perturbations through Meep's public API."""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.tracebacklimit = 0
warnings.showwarning = lambda message, category, filename, lineno, file=None, line=None: print(
    f'{category.__name__}: {message}', file=sys.stderr)


def generate(parameters, output, resolution=80, duration=50., subpixel_tol=1e-8):
    import meep as mp
    mp.verbosity(0)
    parameters = np.asarray(parameters, float)
    if parameters.shape != (6,) or not np.isfinite(parameters).all():
        raise ValueError('This reference expects six finite real coating parameters.')
    band = np.array([.8, 1., 1.2])
    total, minimum, width = .8, .04, 1/resolution
    side, dt = 1/resolution, .4/resolution
    zr, zt = -1., 1.4
    def decode(p):
        weights = np.exp(np.r_[p[:3], 0.]-max(0., float(np.max(p[:3]))))
        lengths = minimum+(total-4*minimum)*weights/weights.sum()
        epsilon = np.r_[p[3:], 4.]
        if np.min(epsilon) < 1. or np.min(lengths) < width/2:
            raise ValueError('The dielectric profile or interface support is outside the reference domain.')
        return epsilon, lengths
    def pulse(t):
        return np.exp(-((t-6.)/1.2)**2)*np.sin(2*np.pi*t)
    def fields(p, until):
        settings = {}
        if p is not None:
            epsilon, lengths = decode(p)
            boundaries = np.cumsum(lengths)[:-1]
            def profile(position):
                if position.z < 0:
                    return mp.Medium(epsilon=1.)
                cumulative = np.clip((position.z-boundaries)/width+.5, 0., 1.)
                value = epsilon[0]+np.dot(np.diff(epsilon), cumulative)
                return mp.Medium(epsilon=float(value))
            profile.do_averaging = True
            settings = dict(material_function=profile, subpixel_tol=subpixel_tol)
        sim = mp.Simulation(cell_size=mp.Vector3(side, side, 10.), dimensions=3, resolution=resolution,
                            boundary_layers=[mp.PML(1., direction=mp.Z)], Courant=.4,
                            k_point=mp.Vector3(), default_material=mp.Medium(epsilon=1.),
                            sources=[mp.Source(mp.CustomSource(pulse, end_time=12.), component=mp.Ex,
                                               center=mp.Vector3(z=-3.), size=mp.Vector3(side, side, 0.))],
                            eps_averaging=True, **settings)
        monitors = [sim.add_dft_fields([mp.Ex], list(1/band), center=mp.Vector3(z=z),
                                       size=mp.Vector3(), decimation_factor=1) for z in (zr, zt)]
        sim.run(until=until)
        data = np.array([[np.asarray(sim.get_dft_array(m, mp.Ex, i)).item()
                          for i in range(len(band))] for m in monitors])
        sim.reset_meep()
        return data
    def amplitudes(p, until, reference):
        data = fields(p, until)
        def kz(index):
            return 2*resolution*np.arcsin(index*np.sin(np.pi*dt/band)/(dt*resolution))
        r = (data[0]/reference[0]-1)*np.exp(2j*kz(1.)*zr)
        t = data[1]/reference[0]*np.exp(1j*kz(1.)*zr-1j*kz(2.)*(zt-total))
        return np.stack((r, t))
    durations, deltas = np.array([duration, duration+20]), np.array([1e-3, 5e-4])
    values, derivatives = [], []
    for until in durations:
        # The reference is independent of all six design variables: its medium,
        # mesh and excitation are fixed, so the same measured fields can be reused.
        reference = fields(None, until)
        values.append(amplitudes(parameters, until, reference))
        derivatives.append(np.stack([
            np.stack([(amplitudes(parameters+v, until, reference)-amplitudes(parameters-v, until, reference))/(2*step)
                      for v in step*np.eye(len(parameters))], axis=-1) for step in deltas]))
    epsilon, thickness = decode(parameters)
    np.savez_compressed(output, version=mp.__version__, parameters=parameters,
                        parameter_names=['weight_1', 'weight_2', 'weight_3', 'epsilon_1', 'epsilon_2', 'epsilon_3'],
                        wavelengths=band, epsilon=epsilon, thickness=thickness, total=total, minimum=minimum,
                        layer_smoothing=width, resolution=resolution, dimensions=3, transverse_cells=1,
                        subpixel_tol=subpixel_tol, durations=durations, deltas=deltas,
                        top=1., bottom=4., source_z=-3., monitor_z=[zr, zt], pml_thickness=1.,
                        cell_length=10., courant=.4, pulse_width=1.2, pulse_peak=6., source_end=12.,
                        amplitudes=np.array(values), jacobians=np.array(derivatives))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--subpixel-tol', type=float, default=1e-8)
    args = parser.parse_args()
    generate(json.loads(args.design.read_text())['parameters'], args.output, subpixel_tol=args.subpixel_tol)
