"""Compare public Meep adjoint and central differences for two complex transmissions."""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
from tools.repository_paths import repository_root
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

sys.tracebacklimit = 0
warnings.showwarning = lambda message, category, filename, lineno, file=None, line=None: print(
    f'{category.__name__}: {message}', file=sys.stderr)


def generate(output, resolution=40, duration=120., profile_grid=256):
    import autograd.numpy as npa
    import meep as mp
    import meep.adjoint as mpa
    sys.path.insert(0, str(repository_root()))
    from studies.fdtd.rotation_smoothing.fdtd_rotation_smoothing import density
    mp.verbosity(0)
    p = np.array([.37, .3])
    period, depth, smoothing = .6, .2, .05
    if not np.isclose(period*resolution, round(period*resolution)):
        raise ValueError('The period must contain an integer number of Meep cells.')
    zr, zt = -1., 1.2
    volume = mp.Volume(center=mp.Vector3(z=zt), size=mp.Vector3(period, period, 0))
    region = mp.Volume(center=mp.Vector3(z=depth/2), size=mp.Vector3(period, period, depth))
    kz = lambda index: 2*resolution*np.arcsin(index*np.sin(np.pi*.4/resolution)/.4)
    rho, mapping = density(p, smoothing, profile_grid, return_jacobian=True)
    rho, mapping = rho.T, mapping.transpose(1, 0, 2).reshape(-1, 2)
    map_steps = np.array([1e-5, 5e-6])
    mapping_fd = [np.stack([(density(p+d, smoothing, profile_grid).T.reshape(-1)
                              -density(p-d, smoothing, profile_grid).T.reshape(-1))/(2*h)
                             for d in np.eye(2)*h], axis=-1) for h in map_steps]
    steps, minima = np.array([1e-4, 5e-5]), np.array([duration, duration+20.])
    values, adjoints, finite, map_checks, end_times, probe_errors = [], [], [], [], [], []
    for component in (mp.Ey, mp.Ex):
        options = dict(cell_size=mp.Vector3(period, period, 8), dimensions=3, resolution=resolution,
                       Courant=.4, k_point=mp.Vector3(), boundary_layers=[mp.PML(1., direction=mp.Z)],
                       sources=[mp.Source(mp.GaussianSource(frequency=1., width=2.5, cutoff=5), component=component,
                                          center=mp.Vector3(z=-2.), size=mp.Vector3(period, period, 0))])
        calibration = mp.Simulation(**options)
        monitor = calibration.add_dft_fields([component], [1.], center=mp.Vector3(z=zr),
                                             size=mp.Vector3(period, period, 0), decimation_factor=1)
        calibration.run(until=duration)
        _, _, _, weights = calibration.get_array_metadata(dft_cell=monitor)
        incident = np.sum(calibration.get_dft_array(monitor, component, 0)*weights)/period**2*np.exp(-1j*kz(1.)*zr)
        _, _, _, weights = calibration.get_array_metadata(vol=volume)
        assert np.isclose(weights.sum(), period**2)
        # Both source and transmitted p signs reverse, so they cancel here.
        scale = np.exp(-1j*kz(1.1)*(zt-depth))/incident/period**2
        calibration.reset_meep()
        component_values, component_adjoints, component_finite, component_checks, component_times = [], [], [], [], []
        for minimum in minima:
            material = mp.MaterialGrid(mp.Vector3(profile_grid, profile_grid, 0),
                                       mp.Medium(epsilon=1.), mp.Medium(epsilon=2.25),
                                       weights=rho, do_averaging=False, beta=0.)
            sim = mp.Simulation(geometry=[mp.Block(center=mp.Vector3(z=2.1), size=mp.Vector3(mp.inf, mp.inf, 3.8),
                                                   material=mp.Medium(epsilon=1.21)),
                                          mp.Block(center=region.center, size=region.size, material=material)], **options)
            observation = mpa.FourierFields(sim, volume=volume, component=component,
                                            yee_grid=False, decimation_factor=1)
            real = lambda field: npa.real(npa.sum(field*weights)*scale)
            imag = lambda field: npa.imag(npa.sum(field*weights)*scale)
            problem = mpa.OptimizationProblem(simulation=sim, objective_functions=[real, imag],
                                              objective_arguments=[observation], frequencies=[1.],
                                              design_regions=[mpa.DesignRegion(material, volume=region)],
                                              decay_by=1e-11, minimum_run_time=minimum, maximum_run_time=minimum)
            def profile_probe(profile):
                ix = (np.array([.207, .5, .602, .742, .848])*profile_grid).astype(int)
                iy = (np.array([.152, .355, .422, .543])*profile_grid).astype(int)
                actual = sim.get_epsilon_grid((ix+.5)*period/profile_grid-period/2,
                                              (iy+.5)*period/profile_grid-period/2, np.array([depth/2]))
                error = np.max(abs(actual-(1+1.25*profile[np.ix_(ix, iy)])))
                assert error < 2e-12, 'The queried Meep profile differs from the design weights.'
                probe_errors.append(error)
            value, gradients = problem([rho.reshape(-1)], need_gradient=True)
            profile_probe(rho)
            gradient = np.asarray(gradients)
            assert gradient.shape == (2, profile_grid**2) and np.isfinite(gradient).all()
            complex_gradient = gradient[0]+1j*gradient[1]
            component_values.append(value[0]+1j*value[1])
            component_adjoints.append(complex_gradient@mapping)
            component_checks.append([complex_gradient@m for m in mapping_fd])
            times = [sim.meep_time()]
            differences = []
            for step in steps:
                columns = []
                for delta in np.eye(2)*step:
                    samples = []
                    for parameters in (p+delta, p-delta):
                        profile = density(parameters, smoothing, profile_grid).T
                        value, _ = problem([profile.reshape(-1)], need_gradient=False)
                        profile_probe(profile)
                        samples.append(value[0]+1j*value[1])
                        times.append(sim.meep_time())
                    columns.append((samples[0]-samples[1])/(2*step))
                differences.append(columns)
            component_finite.append(differences)
            component_times.append(times)
            sim.reset_meep()
            print(f'Completed {mp.component_name(component)}, minimum run time {minimum:g}', flush=True)
        values.append(component_values); adjoints.append(component_adjoints)
        finite.append(component_finite); map_checks.append(component_checks); end_times.append(component_times)
    # First axes are minimum run time, then incident s/p basis.
    np.savez_compressed(output, version=mp.__version__, parameters=p, parameter_names=['angle', 'width'],
                        amplitudes=np.swapaxes(values, 0, 1), adjoint_jacobians=np.swapaxes(adjoints, 0, 1),
                        finite_jacobians=np.swapaxes(finite, 0, 1), mapping_jacobians=np.swapaxes(map_checks, 0, 1),
                        run_end_times=np.swapaxes(end_times, 0, 1), minimum_run_times=minima,
                        maximum_run_times=minima, termination='fixed_minimum_maximum',
                        decay_by=1e-11, deltas=steps, mapping_deltas=map_steps,
                        channels=['t_ss', 't_pp'], original_channel_indices=[2, 7], resolution=resolution,
                        wavelength=1., period=[period, period], thickness=depth, top=1., bottom=1.21, epsilon=2.25,
                        ly=.18, centre=[.071, -.047], shape_smoothing=smoothing, profile_grid=profile_grid,
                        profile_probe_error=max(probe_errors), material_grid_averaging=False, profile_sampling='cell_centres',
                        source_kind='GaussianSource', source_width=2.5, source_cutoff=5., calibration_duration=duration,
                        dimensions=3, cell_length=8., pml_thickness=1., courant=.4, source_position=-2.,
                        monitor_positions=[zr, zt], yee_grid=False, decimation_factor=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resolution', type=int, default=40)
    parser.add_argument('--duration', type=float, default=120.)
    parser.add_argument('--profile-grid', type=int, default=256)
    args = parser.parse_args()
    generate(args.output, args.resolution, args.duration, args.profile_grid)
