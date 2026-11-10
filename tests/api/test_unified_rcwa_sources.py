"""Physical SI power, including cross-sector interference on a finite aperture."""
import numpy as np
import pytest

from wavebit import Material, Units, rcwa
from wavebit.units import Z0


@pytest.mark.parametrize('unit,scale', [('m', 1e-6), ('um', 1.), ('nm', 1e3)])
def test_two_bloch_sectors_match_the_analytic_finite_aperture_power(unit, scale):
    theta = .23
    # The s frames of the +kx and -kx waves point along +y and -y.
    # Thus Ey = exp(i*kx*x) - i*exp(-i*kx*x), with Hx = cos(theta)*Ey/Z0.
    # Integrating their interference over this off-centre aperture is not the
    # sum of the two separately averaged powers.
    problem = rcwa.Problem(
        units=Units(length=unit),
        stack=rcwa.Stack(period=(scale, scale), layers=(
            rcwa.Layer(thickness=.2*scale, background=Material(epsilon=1.)),)),
        excitation=rcwa.Excitation(wavelength=.8*scale, sources=(
            rcwa.PlaneWave(theta=theta, amplitude=1.),
            rcwa.PlaneWave(theta=-theta, amplitude=1j))),
        outputs=rcwa.Outputs(diffraction=False, fields=True,
            power_region=((-0.2*scale, .35*scale), (-.4*scale, .4*scale))))
    result = rcwa.run(problem, numerics=rcwa.Numerics(orders=(1, 0)))
    assert result.diagnostics['source_group_counts'] == (2,)
    kx_per_m = 2*np.pi/(.8e-6)*np.sin(theta)
    width_m, centre_m, height_m = .55e-6, .075e-6, .8e-6
    mean_electric_squared = 2 - 2*np.sin(2*kx_per_m*centre_m)*np.sinc(kx_per_m*width_m/np.pi)
    watts = .5*np.cos(theta)/Z0*mean_electric_squared*width_m*height_m
    np.testing.assert_allclose(result.outputs['top'].incoming, [watts], rtol=1e-12, atol=0)
    np.testing.assert_allclose(result.outputs['bottom'].outgoing, [watts], rtol=1e-12, atol=0)
    assert result.outputs['top'].units == 'W'
    # Independently evaluate the physical field at the aperture centre.
    fields = result.boundary_fields(side='top', x=[.075*scale], y=[0.], part='incoming').values
    ey = np.exp(1j*kx_per_m*centre_m)-1j*np.exp(-1j*kx_per_m*centre_m)
    np.testing.assert_allclose(fields[0, 0, 0, 1], ey, rtol=1e-12, atol=0)
    np.testing.assert_allclose(fields[0, 0, 0, 3], np.cos(theta)*ey/Z0, rtol=1e-12, atol=0)


def test_problem_owns_python_containers_but_keeps_backend_array_leaves():
    center = [.1, -.1]
    vertices = [[-.1, -.1], [.1, -.1], [0., .1]]
    polarization = [1., 1j]
    wavelengths = [.8, .9]
    raster = [[[2., 3.], [4., 5.]]]
    basis = [[1., 0.], [0., 1.]]
    rectangle = rcwa.Rectangle(size=(.2, .1), center=center)
    polygon = rcwa.Polygon(vertices=vertices)
    beam = rcwa.GaussianBeam(waist=1., polarization=polarization)
    orders = rcwa.IncidentOrders(weights=(((0, 0), 1.),), polarization=polarization)
    excitation = rcwa.Excitation(wavelength=wavelengths, sources=(beam, orders))
    layer = rcwa.Layer(thickness=.2, epsilon=raster)
    stack = rcwa.Stack(lattice=basis)
    center[0] = 10.
    vertices[0][0] = 10.
    polarization[0] = 0.
    wavelengths[0] = 100.
    raster[0][0][0] = 9.
    basis[0][0] = 7.
    assert rectangle.center == (.1, -.1)
    assert polygon.vertices[0][0] == -.1
    assert beam.polarization == orders.polarization == (1., 1j)
    assert excitation.wavelength == (.8, .9)
    assert layer.epsilon[0][0][0] == 2.
    assert stack.lattice[0][0] == 1.
    array = np.ones((2, 2))
    assert rcwa.Layer(thickness=.2, epsilon=array).epsilon is array


def _coherent(*sources):
    return rcwa.Problem(
        units=Units(length='um'),
        stack=rcwa.Stack(period=(.8, .9), layers=(rcwa.Layer(thickness=.2, background=Material(2.25)),)),
        excitation=rcwa.Excitation(wavelength=.6, sources=sources),
        outputs=rcwa.Outputs(diffraction=False, power_region=((-.4, .4), (-.45, .45))))


def test_a_prepared_coherent_run_refuses_a_changed_lit_order_label():
    """The lit labels fix the source grouping; only the weights are live.

    The prepared-reuse guard reads the labels from ``IncidentOrders.weights`` by type, so a
    renamed field cannot silently drop them from it (it read ``hasattr(s, 'orders')``).
    """
    from wavebit.errors import IncompatibleProblemError
    numerics = rcwa.Numerics(orders=(1, 1))
    prepared = rcwa.prepare(_coherent(rcwa.IncidentOrders(weights=(((0, 0), 1.), ((1, 0), .5j)))),
                            numerics=numerics)
    live = prepared.run(_coherent(rcwa.IncidentOrders(weights=(((0, 0), 2.), ((1, 0), -.5)))))
    assert live.diagnostics['source_group_counts'] == (1,)
    with pytest.raises(IncompatibleProblemError) as caught:
        prepared.run(_coherent(rcwa.IncidentOrders(weights=(((0, 0), 1.), ((-1, 0), .5j)))))
    assert caught.value.code == 'prepared.structure'


def test_a_prepared_coherent_run_refuses_a_changed_angular_spectrum_extent():
    """The extent sets the beam's quadrature directions, so a preparation fixes it.

    The replay guard names each source type's fields explicitly, so a renamed field fails
    there instead of silently dropping out of the guard (it was looked up by name).
    """
    from wavebit.errors import IncompatibleProblemError
    numerics = rcwa.Numerics(orders=(1, 1))
    prepared = rcwa.prepare(_coherent(rcwa.GaussianBeam(waist=1.)), numerics=numerics)
    prepared.run(_coherent(rcwa.GaussianBeam(waist=1., amplitude=2.)))
    with pytest.raises(IncompatibleProblemError) as caught:
        prepared.run(_coherent(rcwa.GaussianBeam(waist=1., angular_spectrum_extent=3.)))
    assert caught.value.code == 'prepared.source_grouping'


@pytest.mark.parametrize('extent', [0., -1., float('nan'), float('inf'), 1j])
def test_check_names_the_public_angular_spectrum_extent(extent):
    from wavebit.errors import ValidationError
    problem = _coherent(rcwa.GaussianBeam(waist=1., angular_spectrum_extent=extent))
    numerics = rcwa.Numerics(orders=(1, 1))
    (issue,) = rcwa.check(problem, numerics=numerics).issues
    assert issue.code == 'source.angular_spectrum_extent'
    assert issue.path == 'excitation.sources[0].angular_spectrum_extent'
    assert 'GaussianBeam(angular_spectrum_extent=...)' in issue.message
    with pytest.raises(ValidationError) as caught:
        rcwa.run(problem, numerics=numerics)
    assert 'spectral_extent' not in str(caught.value)


def test_a_lit_order_outside_the_window_is_refused_in_public_spelling():
    """The engine names ``truncation_order``; the caller wrote ``Numerics(orders=...)``."""
    from wavebit.errors import ValidationError
    problem = _coherent(rcwa.IncidentOrders(weights=(((5, 0), 1.),)))
    with pytest.raises(ValidationError, match=r'IncidentOrders\(weights=\.\.\.\) names order '
                                              r'\(5, 0\).*Numerics\(orders=\(2, 2\)\)') as caught:
        rcwa.run(problem, numerics=rcwa.Numerics(orders=(2, 2)))
    assert 'truncation_order' not in str(caught.value)


def _crossed_grating(source):
    """A 2D grating lit by one source; ``orders=(1, 1)`` keeps three orders per axis."""
    return rcwa.Problem(
        units=Units(length='nm'),
        stack=rcwa.Stack(period=(700., 700.), bottom=Material(epsilon=2.25), layers=(
            rcwa.Layer(thickness=300., epsilon=np.array([[1., 12.25], [12.25, 12.25]])),)),
        excitation=rcwa.Excitation(wavelength=900., sources=(source,)))


_DIRECTED_SOURCES = {
    'PlaneWave': lambda **angles: rcwa.PlaneWave(**angles),
    'IncidentOrders': lambda **angles: rcwa.IncidentOrders(weights=(((0, 0), 1.),), **angles),
    'GaussianBeam': lambda **angles: rcwa.GaussianBeam(waist=2000., **angles),
}


@pytest.mark.parametrize('kind', sorted(_DIRECTED_SOURCES))
@pytest.mark.parametrize('name,value', [
    # Three angles against three orders per axis: this length broadcast inside the engine, and
    # check and run then reported one clean answer for a mixture of incidences.
    ('theta', np.linspace(.05, .3, 3)),
    ('theta', [.05, .175, .3]),
    ('theta', np.array([.2])),
    ('phi', np.linspace(0., .3, 3)),
])
def test_an_incidence_angle_is_one_number(kind, name, value):
    from wavebit.errors import ValidationError
    angles = dict(theta=.2, phi=.1)
    angles[name] = value
    problem = _crossed_grating(_DIRECTED_SOURCES[kind](**angles))
    numerics = rcwa.Numerics(orders=(1, 1))
    (issue,) = rcwa.check(problem, numerics=numerics).issues
    assert issue.code == 'source.direction'
    assert issue.path == f'excitation.sources[0].{name}'
    assert f'{kind}({name}=...)' in issue.message
    with pytest.raises(ValidationError) as caught:
        rcwa.run(problem, numerics=numerics)
    assert caught.value.code == 'source.direction'


@pytest.mark.parametrize('kind', sorted(_DIRECTED_SOURCES))
@pytest.mark.parametrize('name,value', [
    ('theta', float('nan')),
    ('phi', float('inf')),
    ('theta', complex(1., float('nan'))),
    ('phi', np.array([.1, float('nan')])),
])
def test_a_nonfinite_incidence_angle_is_problem_nonfinite(kind, name, value):
    # A NaN or an infinity is refused as every other non-finite input is, even inside an array
    # that the one-angle rule would also refuse: the code says what is wrong with the value.
    from wavebit.errors import ValidationError
    angles = dict(theta=.2, phi=.1)
    angles[name] = value
    problem = _crossed_grating(_DIRECTED_SOURCES[kind](**angles))
    numerics = rcwa.Numerics(orders=(1, 1))
    (issue,) = rcwa.check(problem, numerics=numerics).issues
    assert (issue.code, issue.path) == ('problem.nonfinite', f'excitation.sources[0].{name}')
    assert 'non-finite' in issue.message
    for door in (rcwa.prepare, rcwa.run):
        with pytest.raises(ValidationError) as caught:
            door(problem, numerics=numerics)
        assert (caught.value.code, caught.value.path) == (issue.code, issue.path)


def test_scalar_angles_of_every_spelling_still_run_and_agree():
    """A Python float, a 0-d NumPy array and a 0-d tensor with a derivative are one angle."""
    torch = pytest.importorskip('torch')
    numerics = rcwa.Numerics(orders=(1, 1))
    reference = rcwa.run(_crossed_grating(rcwa.PlaneWave(theta=.2, phi=.1)), numerics=numerics)
    as_array = rcwa.run(_crossed_grating(rcwa.PlaneWave(theta=np.float64(.2), phi=np.asarray(.1))),
                        numerics=numerics)
    np.testing.assert_array_equal(np.asarray(as_array.R), np.asarray(reference.R))
    theta = torch.tensor(.2, dtype=torch.float64, requires_grad=True)
    differentiable = rcwa.run(_crossed_grating(rcwa.PlaneWave(theta=theta, phi=.1)),
                              numerics=numerics, backend='torch')
    np.testing.assert_allclose(differentiable.R.detach().numpy(), np.asarray(reference.R),
                               rtol=0, atol=1e-12)
    differentiable.R.sum().backward()
    assert theta.grad is not None and torch.isfinite(theta.grad)


def test_a_traced_scalar_angle_still_runs_under_jit():
    """Under ``jax.jit`` the check reads only the angle's shape, which the trace still knows."""
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    with jax.enable_x64(True):
        problem = _crossed_grating(rcwa.PlaneWave(theta=.2, phi=.1))
        numerics = rcwa.Numerics(orders=(1, 1))
        prepared = rcwa.prepare(problem, numerics=numerics, backend='jax')
        def reflected(theta):
            source = problem.excitation.sources[0].replace(theta=theta)
            return prepared.run(problem.replace(excitation=problem.excitation.replace(sources=(source,)))).R.sum()
        eager = float(rcwa.run(problem, numerics=numerics).R.sum())
        assert float(jax.jit(reflected)(jnp.float64(.2))) == pytest.approx(eager, abs=1e-12)
