"""Physical checks of the bottom-origin, upward-positive public coordinate system."""
import numpy as np
import pytest
import wavebit

PATHS = [('te', 'ETM'), ('tm', 'ETM'), ('conical', 'ETM'), ('2ff', 'ETM'),
         ('4ff', 'ETM'), ('te', 'SMM'), ('tm', 'SMM'), ('conical', 'SMM'), ('2ff', 'SMM')]


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def six(cell, path):
    if cell.ndim == 5:
        return cell[0]
    out = np.zeros(cell.shape[:-1] + (6,), complex)
    slots = (1, 3, 5) if path == 'te' else (4, 0, 2)
    out[..., slots] = cell
    return out


@pytest.mark.parametrize('backend', [0, 1, 2], ids=['numpy', 'jax', 'torch'])
@pytest.mark.parametrize('path,algorithm', PATHS)
@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('theta', [0., .53])
def test_plane_wave_phase_maxwell_and_flux(backend, path, algorithm, side, theta):
    """Absolute k direction, curl signs and phase; no old solver output is the oracle."""
    if backend:
        pytest.importorskip('jax' if backend == 1 else 'torch')
    phi = 0. if path in ('te', 'tm') else .61
    structure = np.ones((1, 2, 2) if path in ('2ff', '4ff') else (1, 1, 2))
    kw = dict(backend=backend, top=1., bot=1., wavelength=1.3, period=(.8, .7),
              thickness=[.9], truncation_order=(0, 0), theta=theta, structure=structure,
              incidence_side=side, answers='field', layer_algorithm=algorithm)
    if path in ('te', 'tm'):
        kw['tetm'] = path
    else:
        kw.update(phi=phi, polarization=(.6, .8j))
    solver = wavebit.rcwa(**kw)
    if path in ('2ff', '4ff'):
        solver.formulation = path
    solver.solve()
    x, y, z = np.array([-.13, .07]), np.array([-.11, .04]), np.array([.23, .48])
    field = six(array(solver.reconstruct_field(sample_x=x, sample_y=y, sample_z=z)), path)
    electric, magnetic = field[..., :3], field[..., 3:]
    k = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi),
                  (-1 if side == 'top' else 1)*np.cos(theta)])
    np.testing.assert_allclose(np.cross(k, electric), magnetic, atol=2e-12, rtol=0)
    np.testing.assert_allclose(np.cross(k, magnetic), -electric, atol=2e-12, rtol=0)
    for axis, coordinates, component in [(0, z, 2), (1, y, 1), (2, x, 0)]:
        if field.shape[axis] == 1:
            continue
        phase = np.exp(2j*np.pi/1.3*k[component]*np.diff(coordinates)[0])
        np.testing.assert_allclose(np.take(field, 1, axis=axis),
                                   np.take(field, 0, axis=axis)*phase, atol=2e-12, rtol=0)
    flux = .5*np.real(np.cross(electric, np.conj(magnetic)))
    np.testing.assert_allclose(flux, np.broadcast_to(.5*k, flux.shape), atol=2e-12, rtol=0)


@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('algorithm', ['ETM', 'SMM'])
def test_interface_uses_upper_layer_on_both_incidence_sides(side, algorithm):
    """z=d0 belongs to layer 1, independently of the propagation traversal."""
    solver = wavebit.rcwa(wavelength=1.3, period=(.8, .7), thickness=[.2, .3],
                         structure=np.array([[[2.25]], [[4.]]]), tetm='tm', theta=.4,
                         truncation_order=0, incidence_side=side, answers='field',
                         layer_algorithm=algorithm)
    solver.solve()
    grid = array(solver.reconstruct_field(sample_x=[0.], sample_z=3))
    named = array(solver.reconstruct_field(sample_x=[0.], sample_z=[.2]))
    np.testing.assert_allclose(named[0], grid[3], atol=2e-12, rtol=0)
    assert abs(grid[2, 0, 0, 2] - grid[3, 0, 0, 2]) > .01
    below = array(solver.reconstruct_field(sample_x=[0.], sample_z=[.2 - 1e-10]))
    np.testing.assert_allclose(below[0], grid[2], atol=2e-9, rtol=0)


@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_exact_interfaces_survive_cumulative_sum_roundoff(side):
    thickness = [.1, .2, .4, .7]
    solver = wavebit.rcwa(wavelength=2.3, period=(1., 1.), thickness=thickness,
                         structure=np.array([[[2.]], [[3.]], [[4.]], [[5.]]]),
                         tetm='tm', theta=.4, truncation_order=0, incidence_side=side,
                         answers='field')
    solver.solve()
    grid = array(solver.reconstruct_field(sample_x=[0.], sample_z=3))
    edges = np.cumsum(thickness)[:-1]
    actual = array(solver.reconstruct_field(sample_x=[0.], sample_z=edges))
    np.testing.assert_allclose(actual, grid[[3, 6, 9]], atol=3e-12, rtol=0)


@pytest.mark.parametrize('backend', [0, 1, 2], ids=['numpy', 'jax', 'torch'])
@pytest.mark.parametrize('path,algorithm', PATHS)
@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_one_height_sample_is_each_layers_lower_face(backend, path, algorithm, side):
    if backend:
        pytest.importorskip('jax' if backend == 1 else 'torch')
    shape = (2, 2, 2) if path in ('2ff', '4ff') else (2, 1, 2)
    structure = np.ones(shape) * np.array([2.25, 4.])[:, None, None]
    kw = dict(backend=backend, wavelength=1.3, period=(.8, .7),
              thickness=[.2, .3], structure=structure, truncation_order=0,
              theta=.4, incidence_side=side, answers='field', layer_algorithm=algorithm)
    if path in ('te', 'tm'):
        kw['tetm'] = path
    else:
        kw.update(phi=.61, polarization=(.6, .8j))
    solver = wavebit.rcwa(**kw)
    if path in ('2ff', '4ff'):
        solver.formulation = path
    solver.solve()
    counted = array(solver.reconstruct_field(sample_x=[0.], sample_y=[0.], sample_z=1))
    explicit = array(solver.reconstruct_field(sample_x=[0.], sample_y=[0.], sample_z=[0., .2]))
    np.testing.assert_allclose(counted, explicit, atol=3e-12, rtol=0)


def test_old_serialized_layer_order_requires_explicit_migration():
    with pytest.raises(wavebit.SpecError, match='bottom-to-top.*Reverse the old layer list'):
        wavebit.JobSpec.from_dict({'version': '9'})


@pytest.mark.parametrize('backend', [1, 2], ids=['jax', 'torch'])
@pytest.mark.parametrize('patterned', [False, True], ids=['uniform', 'two-pixel'])
def test_single_height_keeps_thickness_derivatives(backend, patterned):
    pytest.importorskip('jax' if backend == 1 else 'torch')
    structure = np.array([[[2.25, 3.]], [[4., 5.]]] if patterned else [[[2.25]], [[4.]]])
    kwargs = dict(backend=backend, wavelength=1.3, period=(.8, .7),
                  structure=structure,
                  truncation_order=0, theta=.4, tetm='tm', answers='field')
    geometry = wavebit.rcwa(thickness=[.2, .3], **kwargs).freeze_geometry()

    def field_sum(thickness):
        solver = wavebit.rcwa(thickness=thickness, **kwargs)
        _, cell = solver.solve_field(sample_x=1, sample_y=1, sample_z=1, geometry=geometry)
        return cell.real.sum()

    if backend == 1:
        jax = pytest.importorskip('jax')
        import jax.numpy as jnp
        thickness = jnp.asarray([.2, .3])
        actual = np.asarray(jax.jit(jax.grad(field_sum))(thickness))
    else:
        torch = pytest.importorskip('torch')
        thickness = torch.tensor([.2, .3], dtype=torch.float64, requires_grad=True)
        actual = array(torch.autograd.grad(field_sum(thickness), thickness)[0])
    h = 1e-6
    expected = []
    for i in range(2):
        delta = np.eye(2)[i]*h
        expected.append((float(field_sum(np.array([.2, .3])+delta))
                         - float(field_sum(np.array([.2, .3])-delta)))/(2*h))
    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-6)
