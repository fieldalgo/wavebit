"""Artificial layer faces must not halve the spatial derivative of a plane wave."""
import numpy as np
import pytest

import wavebit


Z = np.array([0., .2, .4, .7, 1.])
THETA, WAVELENGTH = .23, 1.3
WEIGHT = np.array([.3+.7j, -.4+.2j, .5-.3j, .6+.1j, -.2+.8j, .9-.4j])


def vacuum(backend, kind, side, algorithm):
    pure = kind in ('te', 'tm')
    options = dict(backend=backend, wavelength=WAVELENGTH, period=(1.7, 1.9),
                   structure=np.ones((2, 1, 2)), thickness=(.4, .6),
                   truncation_order=(1, 0), theta=THETA, incidence_side=side,
                   answers='field', layer_algorithm=algorithm)
    if pure:
        options['tetm'] = kind
    else:
        options.update(phi=.31, polarization=(.6, .8j))
    if kind in ('second_order', 'first_order'):
        options.update(structure=np.ones((2, 2, 2)), truncation_order=(1, 1))
    solver = wavebit.rcwa(**options)
    if kind == 'first_order':
        solver.formulation = 'first_order'
    return solver


def plane_wave(kind, side, z, total=1.):
    phi = 0. if kind in ('te', 'tm') else .31
    sign = -1 if side == 'top' else 1
    boundary = total if side == 'top' else 0.
    k = np.array([np.sin(THETA)*np.cos(phi), np.sin(THETA)*np.sin(phi),
                  sign*np.cos(THETA)])
    s = np.array([-np.sin(phi), np.cos(phi), 0.])
    p = np.array([-np.cos(THETA)*np.cos(phi), -np.cos(THETA)*np.sin(phi),
                  sign*np.sin(THETA)])
    electric = s if kind == 'te' else p if kind == 'tm' else .6*s+.8j*p
    amplitude = np.r_[electric, np.cross(k, electric)]
    phase = np.exp(2j*np.pi/WAVELENGTH*(k[0]*.03+k[1]*.02+k[2]*(z-boundary)))
    field = phase[:, None]*amplitude
    return field, 2j*np.pi/WAVELENGTH*k[2]*field


def differentiate(backend, loss, points):
    if backend in ('jax', 'jit'):
        jax = pytest.importorskip('jax')
        import jax.numpy as jnp
        wavebit.enable_x64()
        calculate = jax.value_and_grad(lambda z: loss(jnp, z))
        if backend == 'jit':
            calculate = jax.jit(calculate)
        value, gradient = calculate(jnp.asarray(points))
        return float(value), np.asarray(gradient)
    torch = pytest.importorskip('torch')
    z = torch.tensor(points, dtype=torch.float64, requires_grad=True)
    value = loss(torch, z)
    gradient, = torch.autograd.grad(value, z)
    return float(value.detach()), gradient.detach().numpy()


@pytest.mark.parametrize('backend', ['jax', 'jit', 'torch'])
@pytest.mark.parametrize('kind', ['te', 'tm', 'conical', 'second_order', 'first_order'])
@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('algorithm', ['ETM', 'SMM'])
def test_saved_field_coordinate_gradient_matches_vacuum_maxwell_wave(backend, kind, side, algorithm):
    pytest.importorskip('torch' if backend == 'torch' else 'jax')
    if backend != 'torch':
        wavebit.enable_x64()
    solver = vacuum(2 if backend == 'torch' else 1, kind, side, algorithm)
    result = solver.solve(auto_detune=False)
    expected, derivative = plane_wave(kind, side, Z)
    field = result.reconstruct_field([.03], [.02], Z)[:, 0, 0]
    field = field.detach().numpy() if backend == 'torch' else np.asarray(field)
    np.testing.assert_allclose(field, expected, rtol=3e-12, atol=3e-12)
    # A later solve must not change the retained field's spatial derivative.
    solver.solve(thickness=(.5, .7), wavelength=1.6, auto_detune=False)
    def loss(xp, z):
        f = result.reconstruct_field([.03], [.02], z)[:, 0, 0]
        return xp.real(xp.sum(f*xp.asarray(WEIGHT)))
    value, gradient = differentiate(backend, loss, Z)
    assert value == pytest.approx(np.real(np.sum(expected*WEIGHT)), abs=3e-12)
    np.testing.assert_allclose(gradient, np.real(np.sum(derivative*WEIGHT, axis=1)),
                               rtol=3e-11, atol=3e-11)


@pytest.mark.parametrize('backend', ['jax', 'jit', 'torch'])
@pytest.mark.parametrize('kind', ['te', 'tm', 'conical', 'second_order', 'first_order'])
@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_moving_an_artificial_interface_cannot_change_a_fixed_field(backend, kind, side):
    pytest.importorskip('torch' if backend == 'torch' else 'jax')
    if backend != 'torch':
        wavebit.enable_x64()
    solver = vacuum(2 if backend == 'torch' else 1, kind, side, 'SMM')
    geometry = solver.freeze_geometry() if backend == 'jit' else None
    expected, _ = plane_wave(kind, side, np.array([.4]))
    expected_value = np.real(np.sum(expected*WEIGHT))
    def loss(xp, split):
        d = xp.stack([split[0], 1.-split[0]])
        result = solver.solve(thickness=d, geometry=geometry, auto_detune=False)
        field = result.reconstruct_field([.03], [.02], [.4])[0, 0, 0]
        return xp.real(xp.sum(field*xp.asarray(WEIGHT)))
    # Before, on and after the same observation point: no physical interface moves.
    for split in (.39, .4, .41):
        value, gradient = differentiate(backend, loss, np.array([split]))
        assert value == pytest.approx(expected_value, abs=3e-12)
        np.testing.assert_allclose(gradient, 0., atol=3e-11)


@pytest.mark.parametrize('backend', ['jax', 'jit', 'torch'])
@pytest.mark.parametrize('source_kind', ['top', 'bottom', 'dipole'])
def test_eager_source_snapshot_has_the_correct_observation_height_derivative(backend, source_kind):
    pytest.importorskip('torch' if backend == 'torch' else 'jax')
    if backend != 'torch':
        wavebit.enable_x64()
    if source_kind == 'dipole':
        source = wavebit.ElectricDipole(position=(.12, -.08, .25),
            moment=(1e-29, .3e-29j, .2e-29), bloch_k=(.21, .13))
        # Independent homogeneous dyadic Green spectrum for the sole Fourier order.
        k0 = 2*np.pi/WAVELENGTH
        kt = np.array(source.bloch_k)
        kz = np.sqrt(k0*k0-kt@kt)
        k = np.tile(np.r_[kt, kz], (len(Z), 1))
        k[:, 2] *= np.where(Z > source.position[2], 1., -1.)
        moment = np.array(source.moment)
        electric = 1j*k0*k0/(2*8.8541878188e-12*1.7*1.9*1e-18*kz)*(
            moment-k*(k@moment)[:, None]/(k0*k0))
        magnetic = np.cross(k, electric)/k0
        phase = np.exp(1j*(kt@(np.array([.03, .02])-source.position[:2])
                           +kz*np.abs(Z-source.position[2])))
        expected = np.concatenate([electric, magnetic], axis=1)*phase[:, None]
        derivative = 1j*k[:, 2, None]*expected
    else:
        source = wavebit.PlaneWave(side=source_kind, theta=THETA, phi=.31,
                                  polarization=(.6, .8j))
        expected, derivative = plane_wave('first_order', source_kind, Z)
    solver = wavebit.rcwa(backend=2 if backend == 'torch' else 1, excitation=source,
                         length_unit=1e-6, wavelength=WAVELENGTH, period=(1.7, 1.9),
                         structure=np.ones((2, 1, 1)), thickness=(.4, .6),
                         truncation_order=0, answers='field')
    # Only differentiate the observation coordinates of an already solved snapshot;
    # differentiation of source grouping and segmentation is not a supported contract.
    result = solver.solve()
    field = result.reconstruct_field([.03], [.02], Z)[:, 0, 0]
    field = field.detach().numpy() if backend == 'torch' else np.asarray(field)
    np.testing.assert_allclose(field, expected, rtol=3e-12, atol=3e-12)
    def loss(xp, z):
        f = result.reconstruct_field([.03], [.02], z)[:, 0, 0]
        return xp.real(xp.sum(f*xp.asarray(WEIGHT)))
    value, gradient = differentiate(backend, loss, Z)
    assert value == pytest.approx(np.real(np.sum(expected*WEIGHT)), abs=3e-12)
    np.testing.assert_allclose(gradient, np.real(np.sum(derivative*WEIGHT, axis=1)),
                               rtol=3e-11, atol=3e-11)


@pytest.mark.parametrize('backend', ['jax', 'jit', 'torch'])
@pytest.mark.parametrize('kind', ['conical', 'first_order'])
@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_unused_evanescent_layers_still_cannot_poison_the_field_gradient(backend, kind, side):
    pytest.importorskip('torch' if backend == 'torch' else 'jax')
    if backend != 'torch':
        wavebit.enable_x64()
    solver = vacuum(2 if backend == 'torch' else 1, kind, side, 'SMM')
    solver.period = (.17, .19)
    solver.thickness = (400., 600.)
    points = np.array([0., 200., 400., 700., 1000.])
    result = solver.solve(auto_detune=False)
    expected, derivative = plane_wave(kind, side, points, total=1000.)
    # Nonzero Fourier orders decay by exponents over 10,000 per layer. Sampling
    # an unrelated layer without clipping would evaluate the growing solution.
    def loss(xp, z):
        f = result.reconstruct_field([.03], [.02], z)[:, 0, 0]
        return xp.real(xp.sum(f*xp.asarray(WEIGHT)))
    value, gradient = differentiate(backend, loss, points)
    assert value == pytest.approx(np.real(np.sum(expected*WEIGHT)), abs=3e-10)
    np.testing.assert_allclose(gradient, np.real(np.sum(derivative*WEIGHT, axis=1)),
                               rtol=3e-10, atol=3e-10)
