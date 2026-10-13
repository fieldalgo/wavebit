"""Discrete radiation: independent reciprocity, Fourier Green and current oracles."""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))
from tools.repository_paths import repository_path
import warnings

import numpy as np
import pytest

import wavebit.fdtd._legacy as fd
from wavebit.fdtd import _radiation as radiation
from wavebit.fdtd._kernel.grid import Grid
from wavebit.fdtd._prepare import plan
from wavebit.fdtd._prepare.monitors import box_faces
from wavebit.fdtd._runtime import runtime
from tests.references.lattice import direct_field, dipole_field, plane, reciprocity, spectral_imag
from tests.fixtures.fdtd.fd_helpers import farfield_dipole, sphere_integral, to_numpy


def fixture():
    grid = Grid.from_domain((3.1,)*3, (0.,)*3, .05)
    faces = box_faces('oracle', grid, fd.Region(size=(1.3,)*3))
    theta = np.array((0., 31., 70., 90., 133., 180.))[:, None]
    phi = np.array((0., 39., 90., 231.))[None, :]
    m = radiation.modes(grid.dx, .5*grid.dx/np.sqrt(3), (.7, 1.1, 1.4), 2.25, 1.3, theta, phi)
    template = plane(grid, faces, np.zeros(3), np.ones(3), 1.)
    rng = np.random.default_rng(240924)  # Reproducible arbitrary complex data.
    data = {f: {k: rng.normal(size=(3,)+v.shape[1:])+1j*rng.normal(size=(3,)+v.shape[1:])
                for k, v in part.items()} for f, part in template.items()}
    return grid, faces, m, data


def test_modes_satisfy_dispersion_ray_and_axis_solution():
    rng = np.random.default_rng(240924)
    directions = np.concatenate((np.eye(3), np.ones((1, 3)), rng.normal(size=(300, 3))))
    directions /= np.linalg.norm(directions, axis=-1)[:, None]
    theta, phi = np.degrees(np.arccos(directions[:, 2])), np.degrees(np.arctan2(directions[:, 1], directions[:, 0]))
    dx, dt, eps, mu = .05, .01, 2.25, 1.3
    for band in (1e-9, 1e-6, .01, .1, .314, .9, 1.3, 1.4):
        freq = np.arcsin(band*dt/(2*dx*np.sqrt(eps*mu)))/(np.pi*dt)
        m = radiation.modes(dx, dt, (freq,), eps, mu, theta, phi)
        K = 2*np.sin(m.k[0]*dx/2)/dx
        ray = np.sin(m.k[0]*dx)/dx
        ray /= np.linalg.norm(ray, axis=-1)[:, None]
        # 304 directions x eight bands: norm residual <=9.55e-15,
        # direction residual <=3.34e-16 before test construction.
        assert np.max(abs(np.sum((K*dx/band)**2, axis=-1)-1)) < 2e-13
        assert np.max(abs(ray-directions)) < 5e-15
        axis_k = 2*np.arcsin(band/2)/dx
        np.testing.assert_allclose(m.k[0, :3], axis_k*np.eye(3), rtol=0, atol=2e-12)
        np.testing.assert_allclose(m.green_gain[0, :3], 1., rtol=0, atol=2e-13)
        np.testing.assert_allclose(m.flux_weight[0, :3], np.sqrt(1-band**2/4)/np.sqrt(mu/eps), rtol=0, atol=2e-13)


def test_green_gain_matches_independent_fourier_pole_integral():
    dx, dt = .05, .5*.05/np.sqrt(3)
    radii = np.linspace(12., 25.3, 32)
    for direction in ((0., 0., 1.), (1., 1., 1.), (.2, .7, .3)):
        r = np.asarray(direction)/np.linalg.norm(direction)
        m = radiation.modes(dx, dt, (1.,), 1., 1., np.degrees(np.arccos(r[2])), np.degrees(np.arctan2(r[1], r[0])))
        phase = radii*(m.k[0, 0]@r)
        design = np.array([np.sin(phase)/radii**j for j in range(4)]
                          +[np.cos(phase)/radii**j for j in range(4)]).T
        values = 4*np.pi*radii*spectral_imag(radii, r, dx, dt, 256)
        coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
        # Fourier orders 128/256/512: asymptotic gain discrepancy <=3.37e-11,
        # cosine limit <=5.74e-12, fit residual <=4e-13. No fitted correction.
        assert abs(coefficients[0]/m.green_gain[0, 0]-1) < 2e-10
        assert abs(coefficients[4]) < 5e-11
        assert np.max(abs(design@coefficients-values)) < 3e-12


@pytest.mark.parametrize('backend', ('numpy', 'jax', 'jax.jit', 'torch'))
def test_complex_reciprocity_all_faces_frequencies_background_and_power(backend):
    grid, faces, m, data = fixture()
    expected = direct_field(grid, faces, data, m)
    rt = runtime(backend.split('.')[0])
    arrays = {f: {k: rt.array(v, 'complex128') for k, v in part.items()} for f, part in data.items()}
    def evaluate(a):
        return radiation.transform(rt, a, faces, grid, m)
    if backend == 'jax.jit':
        import jax
        evaluate = jax.jit(evaluate)
    et, ep, U, vector = map(to_numpy, evaluate(arrays))
    # Three frequencies, nonunit eps/mu, 24 directions and four backends:
    # independent pointwise reciprocity measured <=2.78e-16 absolute.
    np.testing.assert_allclose(vector, expected, rtol=0, atol=5e-15)
    shape = (3,)+m.shape
    np.testing.assert_allclose(et, np.sum(expected.reshape(m.k.shape)*m.theta_vector, axis=-1).reshape(shape), rtol=0, atol=5e-15)
    np.testing.assert_allclose(ep, np.sum(expected.reshape(m.k.shape)*m.phi_vector, axis=-1).reshape(shape), rtol=0, atol=5e-15)
    # Use the dispersion gradient directly, including the full Cartesian
    # field. Its radial component is physical on an anisotropic lattice.
    weight = np.linalg.norm(np.sin(m.k*grid.dx)/grid.dx, axis=-1)/(m.omega[:, None]*m.mu)
    expected_power = .5*weight*np.sum(abs(expected.reshape(m.k.shape))**2, axis=-1)
    np.testing.assert_allclose(U, expected_power.reshape(shape), rtol=0, atol=5e-15)


def test_reciprocity_annihilates_regular_discrete_plane_waves():
    for cpw in (20, 28, 40):
        grid = Grid.from_domain((3.1,)*3, (0.,)*3, 1/cpw)
        faces = box_faces('oracle', grid, fd.Region(size=(1.5,)*3))
        m = radiation.modes(grid.dx, .5*grid.dx/np.sqrt(3), (1.,), 1., 1., (0., 39.), (0., 74.))
        k, n = m.k[0], m.unit_K[0]
        E = np.cross(n[0], (1., 0., 0.))
        a = plane(grid, faces, k[0], E, m.omega[0])
        b = plane(grid, faces, -k[1], np.cross(n[1], (0., 1., 0.)), m.omega[0])
        # Original six arbitrary plane-pair controls gave |W|<=2.85e-16.
        assert abs(reciprocity(a, b, faces, grid.dx)) < 5e-15
        value = radiation.transform(runtime('numpy'), a, faces, grid, m)[3]
        assert np.max(abs(value)) < 5e-15


def test_preparation_refuses_invalid_inputs_and_unresolved_bands():
    base = dict(dx=.05, dt=.01, freqs=(1.,), eps=1., mu=1., theta=0., phi=0.)
    for change in (dict(eps=-1.), dict(mu=1j), dict(theta=np.nan), dict(freqs=()),
                   dict(dx=(.05, .05)), dict(theta=(0., 1.), phi=(0., 1., 2.))):
        with pytest.raises(ValueError, match='RadiationInputRefused'):
            radiation.modes(**(base | change))
    with pytest.raises(ValueError, match='RadiationTemporalAliasRefused'):
        radiation.modes(**(base | dict(freqs=(50.,))))
    with pytest.raises(ValueError, match='RadiationNonconvexBandRefused'):
        radiation.modes(**(base | dict(freqs=(5.,))))
    with pytest.raises(ValueError, match='FarFieldPropagationRefused'):
        fd.FarField('ff', box=fd.Region(size=(.5,)*3), wavelengths=(1.,), propagation='fitted')


def test_scene_refuses_all_nonconvex_frequencies_before_time_stepping():
    scene = farfield_dipole(wavelengths=(1., .2))
    scene = scene.replace(monitors=tuple(m.replace(propagation='yee') for m in scene.monitors))
    with pytest.raises(ValueError, match='RadiationNonconvexBandRefused'):
        plan(scene)


def test_preparation_preserves_scalar_values_and_names_traced_refusal():
    import torch
    scalars = tuple(np.float32(v) for v in (.05, .01, 2.25, 1.3))
    dx, dt, eps, mu = map(float, scalars)
    expected = radiation.modes(dx, dt, (1.,), eps, mu, 37., 41.)
    for values in (scalars, tuple(torch.tensor(v) for v in scalars)):
        actual = radiation.modes(values[0], values[1], (1.,), values[2], values[3], 37., 41.)
        # These contain exactly the same representable input values. Before
        # promotion, NumPy float32 scalars changed k by 1.40182e-7. Require
        # exact agreement, rather than hiding that drift in a tolerance.
        for name in ('k', 'unit_K', 'omega', 'impedance', 'green_gain', 'flux_weight'):
            np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    try:
        radiation.modes(dx, dt, (1.,), eps, mu, torch.tensor(37., requires_grad=True), 0.)
    except Exception as error:
        assert isinstance(error, ValueError), (type(error).__name__, str(error))
        assert str(error).startswith('RadiationInputRefused:')
    else:
        pytest.fail('Traced preparation angles must be refused, not detached.')


@pytest.mark.parametrize('propagation', ('continuum', 'yee'))
def test_phase_control_cli_keeps_one_propagation(propagation, monkeypatch, tmp_path):
    import runpy
    import sys
    source = repository_path('studies/fdtd/subpixel_mie_sphere/fdtd_subpixel_mie_sphere.py')
    scenes = []
    class SceneCaptured(Exception):
        pass
    def capture(scene, **kwargs):
        scenes.append(scene)
        raise SceneCaptured
    monkeypatch.setattr(fd, 'run', capture)
    monkeypatch.setattr(sys, 'argv', [str(source), '--phase-control', '--propagation',
                                    propagation, str(tmp_path/'unused.json')])
    with pytest.raises(SceneCaptured):
        runpy.run_path(str(source), run_name='__main__')
    monitors = [m for m in scenes[0].monitors if isinstance(m, fd.FarField)]
    assert {m.name for m in monitors} == {'ff', 'near', 'far'}
    assert all(m.propagation == propagation for m in monitors)


def test_continuum_approximation_is_reported_once_for_the_whole_scene():
    from wavebit.fdtd._prepare.report import report
    scene = farfield_dipole(wavelengths=(1.,))
    scene = scene.replace(monitors=scene.monitors+(scene.monitors[0].replace(name='second'),))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        p = plan(scene)
    notes = [str(w.message) for w in caught if issubclass(w.category, fd.RadiationApproximationWarning)]
    assert len(notes) == 1 and 'ff' in notes[0] and 'second' in notes[0]
    assert '52.05%/26.38%/12.91%' in notes[0] and 'not an error bound' in notes[0]
    assert report(scene, p).warnings.count(notes[0]) == 1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        plan(scene.replace(monitors=tuple(m.replace(propagation='yee') for m in scene.monitors)))
    assert not any(issubclass(w.category, fd.RadiationApproximationWarning) for w in caught)


@pytest.mark.parametrize('backend', ('jax', 'torch'))
def test_field_derivatives_survive_radiation_transform(backend):
    grid, faces, m, data = fixture()
    rt = runtime(backend)
    arrays = {f: {k: rt.array(v, 'complex128') for k, v in part.items()} for f, part in data.items()}
    def loss(scale):
        scaled = {f: {k: scale*v for k, v in part.items()} for f, part in arrays.items()}
        return rt.xp.sum(radiation.transform(rt, scaled, faces, grid, m)[2])
    if backend == 'jax':
        import jax
        value, derivative = jax.jit(jax.value_and_grad(loss))(.72)
    else:
        import torch
        scale = torch.tensor(.72, dtype=torch.float64, requires_grad=True)
        value = loss(scale)
        derivative, = torch.autograd.grad(value, scale)
        value, derivative = value.detach(), derivative.detach()
    # Power is quadratic in the prescribed field scale. Preliminary JIT/
    # Torch derivative residuals were 4.45e-16 / 2.23e-16 relative.
    assert abs(float(derivative/(2*value/.72)-1)) < 1e-14


@pytest.mark.slow
@pytest.mark.parametrize('component', ('Ez', 'Hx'))
def test_public_radiation_matches_actual_discrete_current_and_energy(component):
    scene = farfield_dipole(component=component, kind='electric' if component[0] == 'E' else 'magnetic',
                            wavelengths=(1.,), pml=24)
    scene = scene.replace(monitors=(scene.monitors[0].replace(propagation='yee'),
        fd.FarField('ff2', box=fd.Region(size=(.7,)*3), wavelengths=(1.,), propagation='yee')))
    result = fd.run(scene, backend='jax')
    theta, phi = np.array((30., 60., 90., 120., 150.))[:, None], np.array((0., 70., 200.))[None, :]
    m = radiation.modes(result.plan.dx, result.plan.dt, (1.,), 1., 1., theta, phi)
    expected = dipole_field(scene, result, m)
    for name in ('ff', 'ff2'):
        value = to_numpy(result[name].vector(theta, phi))
        # Independent pml=12/24 probes measured <=1.102e-7 complex peak.
        # With pml=24, angular integral/box flux discrepancy <=1.358e-8;
        # pml=12 had a 1.131e-5 boundary-reflection floor.
        assert np.max(abs(value-expected))/np.max(abs(expected)) < 5e-7
        assert abs(sphere_integral(result[name], 24, 32)[0]/to_numpy(result[name].flux)[0]-1) < 1e-7
        assert float(result.diagnostics['decay'][name]) < 1e-9  # Measured <=5.10e-11.


@pytest.mark.slow
def test_public_weak_sphere_discrete_optical_theorem():
    from studies.fdtd.subpixel_mie_sphere.fdtd_subpixel_mie_sphere import sphere_scene, read_result
    scene = sphere_scene(cpw=20, eps=1.1, propagation='yee')
    result = fd.run(scene, backend='jax')
    row = read_result(result, 1.1)
    # Original G6 optical criterion is 1.6%. The legacy continuum transform
    # on this grid measured 52.0475%; the derived lattice forward observable
    # on saved fields measured 0.026403%. No gate is relaxed for the new map.
    assert abs(row['optical_closure']) < .016, row
    # Fresh public execution measured 2.67422e-4 closure. The saved 20-cell
    # 24->48-period control moved it by 1.97434e-4. Allow 5e-4 as an
    # instrument regression budget, within the unchanged G6 criterion.
    assert abs(row['optical_closure']) < 5e-4, row
    # Lattice angular integral/box flux measured <=8.4e-6 across the saved
    # 20/28/40 cases. Retain the original public 2e-4 power closure gate.
    assert abs(row['farfield_integral']/row['qsca']-1) < 2e-4, row
