"""Independent shape-scattering derivatives through Meep's documented API.

Only public documentation and black-box execution are used. Cylinder contract:
https://meep.readthedocs.io/en/latest/Python_User_Interface/#cylinder
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

sys.tracebacklimit = 0
warnings.showwarning = lambda message, category, filename, lineno, file=None, line=None: print(
    f'{category.__name__}: {message}', file=sys.stderr)


def generate(output, resolution=40, duration=80., step=1e-4, subpixel_tol=1e-8, smoothing=0., profile_grid=256,
             shape='rectangle'):
    if shape not in ('rectangle', 'circle', 'ellipse', 'painted_rectangles'):
        raise ValueError('shape must be rectangle, circle, ellipse or painted_rectangles.')
    if shape in ('ellipse', 'painted_rectangles') and smoothing <= 0:
        raise ValueError('This ellipse/painted reference bridge requires positive smoothing.')
    import meep as mp
    mp.verbosity(0)
    if smoothing:
        # The reference profile is an independent Fourier sinc series. Import
        # only its NumPy helper; no FDTD geometry or solver builds these data.
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        if shape == 'circle':
            from studies.fdtd_circle_gradients import density
        elif shape == 'ellipse':
            from studies.fdtd_ellipse_gradients import density
        elif shape == 'painted_rectangles':
            from studies.fdtd_painted_shapes import density
        else:
            from studies.fdtd_rotation_smoothing import density
    if smoothing < 0 or smoothing > .6 or not np.isfinite(smoothing):
        raise ValueError('smoothing must be finite and between zero and the period.')
    if profile_grid < 8:
        raise ValueError('profile_grid must be at least eight.')
    profile_errors = []
    p = np.array({'circle': [.3], 'rectangle': [.37, .3], 'ellipse': [.37, .34, .2],
                  'painted_rectangles': [.37, .055, .22]}[shape])
    period, depth, length = .6, .2, 8.
    zr, zt, dt = -1., 1.2, .4/resolution
    if not np.isclose(period*resolution, round(period*resolution)):
        raise ValueError('The Meep period must contain an integer number of cells.')
    kz = lambda n: 2*resolution*np.arcsin(n*np.sin(np.pi*dt)/(.4))
    pulse = lambda t: np.exp(-((t-12.5)/2.5)**2)*np.sin(2*np.pi*t)
    def fields(parameters, until, component):
        geometry = []
        if parameters is not None:
            angle, width = (0., parameters[0]) if shape == 'circle' else parameters[:2]
            geometry = [mp.Block(center=mp.Vector3(z=(depth+length/2)/2),
                                 size=mp.Vector3(mp.inf, mp.inf, length/2-depth), material=mp.Medium(epsilon=1.21)),
                        mp.Block(center=mp.Vector3(.071, -.047, depth/2), size=mp.Vector3(width, .18, depth),
                                 e1=mp.Vector3(np.cos(angle), np.sin(angle), 0.),
                                 e2=mp.Vector3(-np.sin(angle), np.cos(angle), 0.),
                                 e3=mp.Vector3(0., 0., 1.), material=mp.Medium(epsilon=2.25))]
            if shape == 'circle':
                geometry[1] = mp.Cylinder(radius=width/2, height=depth,
                    center=mp.Vector3(.071, -.047, depth/2), axis=mp.Vector3(0., 0., 1.),
                    material=mp.Medium(epsilon=2.25))
            if smoothing:
                weights = density(parameters, smoothing, profile_grid).T
                material = mp.MaterialGrid(mp.Vector3(profile_grid, profile_grid, 0),
                                           mp.Medium(epsilon=1.), mp.Medium(epsilon=2.25),
                                           weights=weights, do_averaging=False, beta=0.)
                geometry[1] = mp.Block(center=mp.Vector3(z=depth/2),
                                       size=mp.Vector3(period, period, depth), material=material)
        sim = mp.Simulation(cell_size=mp.Vector3(period, period, length), dimensions=3, resolution=resolution,
                            boundary_layers=[mp.PML(1., direction=mp.Z)], Courant=.4, k_point=mp.Vector3(),
                            geometry=geometry, eps_averaging=True, subpixel_tol=subpixel_tol,
                            sources=[mp.Source(mp.CustomSource(pulse, end_time=25.), component=component,
                                               center=mp.Vector3(z=-2.), size=mp.Vector3(period, period, 0.))])
        if smoothing and parameters is not None:
            # Verify axes and sampling positions through the public geometry
            # API, independently of the Yee grid and the scattering result.
            sim.init_sim()
            ix = (np.array([.207, .5, .602, .742, .848])*profile_grid).astype(int)
            iy = (np.array([.152, .355, .422, .543])*profile_grid).astype(int)
            epsilon = sim.get_epsilon_grid((ix+.5)*period/profile_grid-period/2,
                                           (iy+.5)*period/profile_grid-period/2, np.array([depth/2]))
            error = np.max(abs(epsilon-(1+1.25*weights[np.ix_(ix, iy)])))
            assert error < 2e-12, 'MaterialGrid sampling differs from the prescribed Fourier profile.'
            profile_errors.append(error)
        monitors = [sim.add_dft_fields([mp.Ex, mp.Ey], [1.], center=mp.Vector3(z=z),
                                       size=mp.Vector3(period, period, 0.), decimation_factor=1) for z in (zr, zt)]
        sim.run(until=until)
        values = []
        for monitor in monitors:
            _, _, _, weights = sim.get_array_metadata(dft_cell=monitor)
            weights = np.asarray(weights)
            assert np.isclose(weights.sum(), period**2)
            values.append([np.sum(sim.get_dft_array(monitor, c, 0)*weights)/period**2 for c in (mp.Ex, mp.Ey)])
        sim.reset_meep()
        return np.array(values)
    def amplitude(parameters, until, refs):
        columns = []
        for component, ref in zip((mp.Ey, mp.Ex), refs):
            actual = fields(parameters, until, component)
            incident = ref[0, 1 if component == mp.Ey else 0]*(1 if component == mp.Ey else -1)*np.exp(-1j*kz(1.)*zr)
            r = (actual[0]-ref[0])/incident*np.exp(1j*kz(1.)*zr)
            t = actual[1]/incident*np.exp(-1j*kz(1.1)*(zt-depth))
            columns.extend((r[1], r[0], t[1], -t[0]))
        return np.array(columns)
    durations, deltas = np.array([duration, duration+20.]), np.array([step, step/2])
    values, jacobians = [], []
    for until in durations:
        refs = [fields(None, until, c) for c in (mp.Ey, mp.Ex)]
        values.append(amplitude(p, until, refs))
        jacobians.append(np.stack([np.stack([
            (amplitude(p+d, until, refs)-amplitude(p-d, until, refs))/(2*h) for d in np.eye(len(p))*h], axis=-1)
            for h in deltas]))
        print(f'Completed duration {until:g}', flush=True)
    np.savez_compressed(output, version=mp.__version__, parameters=p,
                        parameter_names=({'circle': ['diameter'], 'rectangle': ['angle', 'width'],
                                          'ellipse': ['angle', 'width', 'height'],
                                          'painted_rectangles': ['angle', 'separation', 'width']}[shape]), shape=shape,
                        wavelength=1., period=[period, period], thickness=depth, top=1., bottom=1.21,
                        epsilon=2.25, ly=p[-1] if shape == 'ellipse' else .18,
                        centre=[.025, -.025] if shape == 'painted_rectangles' else [.071, -.047], resolution=resolution,
                        durations=durations, deltas=deltas, dimensions=3, cell_length=length,
                        pml_thickness=1., courant=.4, subpixel_tol=subpixel_tol,
                        shape_smoothing=smoothing, profile_grid=profile_grid if smoothing else 0,
                        material_grid_averaging=False, profile_sampling='cell_centres' if smoothing else 'none',
                        profile_probe_error=max(profile_errors, default=0.),
                        source_position=-2., monitor_positions=[zr, zt], pulse_width=2.5,
                        pulse_peak=12.5, source_end=25., amplitudes=values, jacobians=jacobians,
                        **(dict(paint_rule='later_shape_overwrites', paint_epsilon=[1.8, 2.25],
                                local_rectangles=[[-.04, -.015, .3, .18], [p[1], .025, p[2], .22]],
                                global_centre=[.025, -.025]) if shape == 'painted_rectangles' else {}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resolution', type=int, default=40)
    parser.add_argument('--duration', type=float, default=80.)
    parser.add_argument('--step', type=float, default=1e-4)
    parser.add_argument('--subpixel-tol', type=float, default=1e-8)
    parser.add_argument('--smoothing', type=float, default=0.)
    parser.add_argument('--profile-grid', type=int, default=256)
    parser.add_argument('--shape', choices=['rectangle', 'circle', 'ellipse', 'painted_rectangles'], default='rectangle')
    args = parser.parse_args()
    generate(args.output, args.resolution, args.duration, args.step, args.subpixel_tol, args.smoothing, args.profile_grid, args.shape)
