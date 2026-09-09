"""Independent rectangle sensitivities, using only Meep's documented public API."""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

sys.tracebacklimit = 0
warnings.showwarning = lambda message, category, filename, lineno, file=None, line=None: print(
    f'{category.__name__}: {message}', file=sys.stderr)


def generate(output, resolution=40, duration=100., step=1e-4, smoothing=0., subpixel_tol=1e-8):
    import meep as mp
    mp.verbosity(0)
    period, depth, cell_length = 1.3, .2, 10.
    p = np.array([.43, .083])  # width, centre relative to cell centre
    orders, durations, deltas = np.array([-1, 0, 1]), np.array([duration, duration+40.]), np.array([step, step/2])
    qx = 2*np.pi*orders/period
    dt = .4/resolution
    def longitudinal(index):
        return 2*resolution*np.arcsin(np.sqrt((index*np.sin(np.pi*dt)/(.4))**2
                                              -np.sin(qx/(2*resolution))**2)+0j)
    kt, kb = longitudinal(1.), longitudinal(1.1)
    def fields(parameters, until):
        geometry = [] if parameters is None else [
            mp.Block(center=mp.Vector3(y=(depth+cell_length/2)/2),
                     size=mp.Vector3(mp.inf, cell_length/2-depth, mp.inf), material=mp.Medium(epsilon=1.21)),
            mp.Block(center=mp.Vector3(parameters[1], depth/2),
                     size=mp.Vector3(parameters[0], depth, mp.inf), material=mp.Medium(epsilon=2.25))]
        settings = {}
        if parameters is not None and smoothing:
            left, right = parameters[1]-parameters[0]/2, parameters[1]+parameters[0]/2
            def profile(position):
                if position.y < 0:
                    return mp.Medium(epsilon=1.)
                if position.y >= depth:
                    return mp.Medium(epsilon=1.21)
                fill = sum(np.clip((position.x-left+image)/smoothing+.5, 0., 1.)
                           -np.clip((position.x-right+image)/smoothing+.5, 0., 1.) for image in (-period, 0., period))
                return mp.Medium(epsilon=float(1.+1.25*fill))
            profile.do_averaging = True
            geometry = []
            settings = dict(material_function=profile, subpixel_tol=subpixel_tol)
        pulse = lambda t: np.exp(-((t-15.)/3.)**2)*np.sin(2*np.pi*t)
        sim = mp.Simulation(cell_size=mp.Vector3(period, cell_length), dimensions=2, resolution=resolution,
                            boundary_layers=[mp.PML(1., direction=mp.Y)], Courant=.4,
                            k_point=mp.Vector3(), geometry=geometry, eps_averaging=True,
                            sources=[mp.Source(mp.CustomSource(pulse, end_time=30.), component=mp.Ez,
                                               center=mp.Vector3(y=-3.), size=mp.Vector3(period, 0.))], **settings)
        positions = (-1., 1.4)
        monitors = [sim.add_dft_fields([mp.Ez], [1.], center=mp.Vector3(y=y),
                                       size=mp.Vector3(period, 0.), decimation_factor=1) for y in positions]
        sim.run(until=until)
        result = []
        for monitor in monitors:
            x, _, _, weights = sim.get_array_metadata(dft_cell=monitor)
            field = np.asarray(sim.get_dft_array(monitor, mp.Ez, 0)).ravel()
            x, weights = np.asarray(x).ravel(), np.asarray(weights).ravel()
            assert np.isclose(weights.sum(), period)
            result.append(np.sum(field[:, None]*weights[:, None]*np.exp(-1j*np.asarray(x)[:, None]*qx), axis=0)/period)
        sim.reset_meep()
        return np.array(result)
    def amplitudes(parameters, until, reference):
        actual = fields(parameters, until)
        incident = reference[0, 1]*np.exp(1j*kt[1])
        assert np.max(abs(reference[:, [0, 2]]))/abs(incident) < 1e-8
        r = (actual[0]-reference[0])/incident*np.exp(-1j*kt)
        t = actual[1]/incident*np.exp(-1j*kb*(1.4-depth))
        # The s frame follows transverse momentum: s_y=-1 for order -1.
        # The solver's corner-based Fourier origin contributes (-1)^order.
        frame = np.array([-1., 1., 1.])*(-1.)**orders
        return np.stack((r, t))*frame
    values, jacobians = [], []
    for until in durations:
        reference = fields(None, until)
        values.append(amplitudes(p, until, reference))
        jacobians.append(np.stack([np.stack([
            (amplitudes(p+v, until, reference)-amplitudes(p-v, until, reference))/(2*step)
            for v in np.eye(2)*step], axis=-1) for step in deltas]))
    np.savez_compressed(output, version=mp.__version__, parameters=p, parameter_names=['width', 'centre'],
                        orders=orders, wavelength=1., period=period, thickness=depth, top=1., bottom=1.21,
                        epsilon=2.25, resolution=resolution, durations=durations, deltas=deltas,
                        dimensions=2, cell_length=cell_length, pml_thickness=1., courant=.4,
                        shape_smoothing=smoothing, subpixel_tol=subpixel_tol if smoothing else 1e-4,
                        source_position=-3., monitor_positions=[-1., 1.4], pulse_width=3.,
                        pulse_peak=15., source_end=30., amplitudes=values, jacobians=jacobians)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resolution', type=int, default=40)
    parser.add_argument('--duration', type=float, default=100.)
    parser.add_argument('--step', type=float, default=1e-4)
    parser.add_argument('--smoothing', type=float, default=0.)
    parser.add_argument('--subpixel-tol', type=float, default=1e-8)
    args = parser.parse_args()
    generate(args.output, args.resolution, args.duration, args.step, args.smoothing, args.subpixel_tol)
