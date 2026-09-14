"""Original continuous-field energy and the unrestricted contact trace."""
import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from wavebit._core.conforming.periodic_seam_trace import PeriodicSeamTrace


SETTINGS = dict(period=(700., 560.), width=310., offset=21.)


def contact_reference(order, *, period, width, offset, epsilon, background):
    modes = np.arange(-order, order+1)
    difference = modes[:, None]-modes
    indicator = (width/period[0]*np.sinc(difference*width/period[0])
                 *np.exp(-2j*np.pi*difference*(.5+offset/period[0])))
    value = background*np.eye(len(modes))+(epsilon-background)*indicator
    derivative = (epsilon-epsilon**2/background)*indicator/period[1]
    return value, derivative


def integrated_reference(order, gap, epsilon, background, settings=SETTINGS, quadrature=48):
    """Integrate raw standing-wave potentials in physical 2D coordinates.

    This does not use the candidate's polynomial frame, samples, analytical
    y integrals, or gap-scaled weak matrix. The reconstructed field energy
    checks the finite response separately from its Schur elimination.
    """
    px, py = settings['period']
    width, center = settings['width'], px/2+settings['offset']
    cap = min(width, px-width)/4
    raw = np.sqrt(gap*(py-gap)/12)
    delta = cap*raw/(cap+raw)
    a, b = center-width/2, center+width/2
    xknots = [center-px/2, a-delta, a, a+delta, b-delta, b, b+delta, center+px/2]
    yknots = [0., gap/2, py-gap/2, py]
    nodes, weights = leggauss(quadrature)
    nodes, weights = (nodes+1)/2, weights/2
    modes = np.arange(-order, order+1)
    frequency = 2*np.pi*np.arange(1, order+1)/px
    count = len(modes)
    k = np.zeros((count, count), complex)
    r = np.zeros_like(k)
    a_matrix = np.zeros_like(k)
    projection = np.zeros_like(k)
    cells = []
    for left, right in zip(xknots, xknots[1:]):
        x = left+(right-left)*nodes
        distance = np.abs(x-center)
        fraction = np.clip((width/2+delta-distance)/(2*delta), 0., 1.)
        trace = fraction**2*(3-2*fraction)
        derivative = np.where((distance > width/2-delta) & (distance < width/2+delta),
                              -np.sign(x-center)*3*fraction*(1-fraction)/delta, 0.)
        phase = (x-center)[:, None]*frequency
        basis = np.column_stack([np.ones(len(x)), np.cos(phase), np.sin(phase)])
        dbasis = np.column_stack([np.zeros(len(x)), -frequency*np.sin(phase), frequency*np.cos(phase)])
        tx = trace[:, None]*basis
        dtx = derivative[:, None]*basis+trace[:, None]*dbasis
        phi = np.exp(2j*np.pi*x[:, None]*modes/px)
        for bottom, top in zip(yknots, yknots[1:]):
            y = bottom+(top-bottom)*nodes
            inside = (y > gap/2) & (y < py-gap/2)
            f = np.where(inside, gap*(py/2-y)/(py-gap), np.where(y < gap/2, y, y-py))
            df = np.where(inside, -gap/(py-gap), 1.)
            g = np.stack([dtx[:, None, :]*f[None, :, None], tx[:, None, :]*df[None, :, None]], axis=2)
            epsilon_grid = np.where(((x > a) & (x < b))[:, None] & inside[None], epsilon, background)
            w = weights[:, None]*weights[None]*(right-left)*(top-bottom)/(px*py)
            k += np.einsum('xy,xy,xyai,xyaj->ij', w, epsilon_grid, g, g)
            r += np.einsum('xy,xy,xm,xyj->mj', w, epsilon_grid, phi.conj(), g[:, :, 1])
            projection += np.einsum('xy,xm,xyj->mj', w, phi.conj(), g[:, :, 1])
            a_matrix += np.einsum('xy,xy,xm,xn->mn', w, epsilon_grid, phi.conj(), phi)
            cells.append((g, phi, epsilon_grid, w))
    np.testing.assert_allclose(projection, 0., atol=3e-12, rtol=0)
    u = np.linalg.solve(k, r[::-1].T)
    result = a_matrix-r@u
    energy = np.zeros_like(k)
    for gradient, phi, epsilon_grid, weight in cells:
        field = -gradient@u
        field[:, :, 1] += phi[:, None, :]
        energy += np.einsum('xy,xy,xyam,xyan->mn', weight, epsilon_grid, field.conj(), field)
    return result, energy


@pytest.mark.parametrize('order', (0, 1, 3))
@pytest.mark.parametrize('epsilon,background', ((12.25, 1.), (-10.+1j, 1.+.1j)))
def test_positive_gap_matches_original_continuous_field_energy(order, epsilon, background):
    obj = PeriodicSeamTrace(order, coercivity_angle=1.53)
    expected, energy = integrated_reference(order, 7., epsilon, background)
    value, _, metadata = obj.normal_response(gap=7., epsilon=epsilon, background=background, **SETTINGS)
    np.testing.assert_allclose(value, expected, atol=3e-10, rtol=0)
    np.testing.assert_allclose(value, energy, atol=3e-10, rtol=0)
    assert metadata['weak_residual'] < 1e-10
    assert metadata['weak_derivative_residual'] < 1e-10


@pytest.mark.parametrize('epsilon,background', ((12.25, 1.), (-10.+1j, 1.+.1j)))
def test_gap_derivative_matches_independent_moving_physical_fields(epsilon, background):
    obj = PeriodicSeamTrace(1, coercivity_angle=1.53)
    _, derivative, _ = obj.normal_response(gap=7., epsilon=epsilon, background=background, **SETTINGS)
    # A high-order stencil permits steps above the roundoff floor of the
    # independently integrated metal resonance, without weakening the guard.
    for step in (.02, .01):
        values = [integrated_reference(1, 7+i*step, epsilon, background, quadrature=32)[0]
                  for i in (-4, -3, -2, -1, 1, 2, 3, 4)]
        expected = sum(c*v for c, v in zip((3, -32, 168, -672, 672, -168, 32, -3), values))/(840*step)
        np.testing.assert_allclose(560*derivative, 560*expected, atol=3e-9, rtol=0)


@pytest.mark.parametrize('order', (0, 3, 9))
@pytest.mark.parametrize('fraction', (.2, .5, .8))
@pytest.mark.parametrize('epsilon,background', ((12.25, 1.), (-10.+1j, 1.+.1j)))
def test_contact_supplies_the_unrestricted_normal_trace(order, fraction, epsilon, background):
    settings = dict(SETTINGS, width=700*fraction)
    obj = PeriodicSeamTrace(order, coercivity_angle=1.53)
    value, derivative, _ = obj.normal_response(gap=0., epsilon=epsilon, background=background, **settings)
    expected, tangent = contact_reference(order, epsilon=epsilon, background=background, **settings)
    np.testing.assert_allclose(value, expected, atol=3e-10, rtol=0)
    np.testing.assert_allclose(560*derivative, 560*tangent, atol=3e-9, rtol=0)


@pytest.mark.parametrize('order', (0, 3, 9))
def test_positive_gap_derivative_converges_to_the_same_contact_limit(order):
    obj = PeriodicSeamTrace(order)
    expected, tangent = contact_reference(order, epsilon=12.25, background=1., **SETTINGS)
    errors = []
    for gap in (7e-4, 7e-6, 7e-8):
        value, derivative, _ = obj.normal_response(gap=gap, **SETTINGS)
        errors.append(np.linalg.norm(derivative-tangent))
        # The actual forward remainder is O(gap**1.5), not a slope-only fix.
        assert np.linalg.norm(value-expected-gap*tangent) < 2*gap*errors[-1]+3e-10
    assert errors[1] < .15*errors[0]
    assert errors[2] < .15*errors[1]


@pytest.mark.parametrize('gap', (0., 7., 140.))
def test_combined_material_and_gap_derivative(gap):
    obj = PeriodicSeamTrace(3, coercivity_angle=1.53)
    settings = dict(SETTINGS, gap=gap, epsilon=-10+1j, background=1+.1j)
    direction, bgdirection = .3+.2j, -.2+.1j
    _, material, _ = obj.normal_response(**settings, gap_tangent=0., epsilon_tangent=direction,
                                          background_tangent=bgdirection)
    _, imaginary, _ = obj.normal_response(**settings, gap_tangent=0., epsilon_tangent=1j*direction,
                                           background_tangent=1j*bgdirection)
    np.testing.assert_allclose(imaginary, 1j*material, atol=3e-10, rtol=0)
    _, dg, _ = obj.normal_response(**settings)
    _, both, _ = obj.normal_response(**settings, epsilon_tangent=direction, background_tangent=bgdirection)
    np.testing.assert_allclose(both, dg+material, atol=3e-10, rtol=0)
    if gap:
        for step in (.002, .001):
            values = [obj.normal_response(**dict(settings, gap=gap+i*step,
                                                   epsilon=settings['epsilon']+i*step*direction,
                                                   background=settings['background']+i*step*bgdirection))[0]
                      for i in (-2, -1, 1, 2)]
            np.testing.assert_allclose((values[0]-8*values[1]+8*values[2]-values[3])/(12*step), both,
                                       atol=3e-9, rtol=0)


@pytest.mark.parametrize('scale', (1e-4, 1e4))
def test_units_and_subpixel_gap(scale):
    obj = PeriodicSeamTrace(3)
    for gap in (0., 1e-9, 7.):
        value, tangent, _ = obj.normal_response(gap=gap, **SETTINGS)
        settings = {k: np.asarray(v)*scale for k, v in SETTINGS.items()}
        other, derivative, _ = obj.normal_response(gap=gap*scale, **settings)
        np.testing.assert_allclose(other, value, atol=3e-10, rtol=0)
        np.testing.assert_allclose(scale*derivative, tangent, atol=3e-10, rtol=0)


@pytest.mark.parametrize('gap', (0., 1e-9, 7.))
def test_uniform_material_and_translation(gap):
    obj = PeriodicSeamTrace(3, coercivity_angle=1.53)
    material = -10+1j
    value, derivative, _ = obj.normal_response(gap=gap, epsilon=material, background=material,
                                                epsilon_tangent=1., background_tangent=1., **SETTINGS)
    np.testing.assert_allclose(value, material*np.eye(7), atol=3e-10, rtol=0)
    np.testing.assert_allclose(derivative, np.eye(7), atol=3e-10, rtol=0)
    base, tangent, _ = obj.normal_response(gap=gap, **SETTINGS)
    other, change, _ = obj.normal_response(gap=gap, **dict(SETTINGS, offset=SETTINGS['offset']+91.))
    phase = np.exp(-2j*np.pi*np.arange(-3, 4)*91/700)
    np.testing.assert_allclose(other, phase[:, None]*base*phase.conj(), atol=3e-10, rtol=0)
    np.testing.assert_allclose(change, phase[:, None]*tangent*phase.conj(), atol=3e-10, rtol=0)


def test_pointwise_potentials_are_periodic_continuous_and_have_the_reported_gradients():
    obj = PeriodicSeamTrace(3)
    points = np.array([[180., 1.], [370., 120.], [630., 559.]])
    potential, gradient = obj.sample(points, gap=7., **SETTINGS)
    for shift in ((700., 0.), (0., 560.)):
        p, g = obj.sample(points+shift, gap=7., **SETTINGS)
        np.testing.assert_allclose(p, potential, atol=3e-10, rtol=0)
        np.testing.assert_allclose(g, gradient, atol=3e-10, rtol=0)
    for axis in range(2):
        shift = np.eye(2)[axis]*1e-4
        p = [obj.sample(points+i*shift, gap=7., **SETTINGS)[0] for i in (-1, 1)]
        np.testing.assert_allclose((p[1]-p[0])/2e-4, gradient[:, axis], atol=3e-9, rtol=0)
    boundary = np.array([[370., 3.5], [370., 556.5], [370., 0.]])
    p = [obj.sample(boundary+(0., i*1e-8), gap=7., **SETTINGS)[0] for i in (-1, 1)]
    np.testing.assert_allclose(p[0], p[1], atol=1e-7, rtol=0)


@pytest.mark.parametrize('gap', (.01, 7.))
def test_moving_quadrature_fields_have_an_analytic_gap_jvp(gap):
    obj = PeriodicSeamTrace(3)
    px, py = SETTINGS['period']
    width = SETTINGS['width']
    center = px/2+SETTINGS['offset']
    raw = np.sqrt(gap*(py-gap)/12)
    cap = min(width, px-width)/4
    delta = cap*raw/(cap+raw)
    points = np.array([[center-width/2+.3*delta, gap/4],
                       [center+width/2-.2*delta, 120.], [center, -gap/4]])
    direction = np.array([[.2, .1], [-.1, .3], [.1, -.2]])
    _, _, dp, dg = obj.sample_with_jvp(points, gap=gap, points_tangent=direction, **SETTINGS)
    for step in (gap*.01, gap*.005):
        values = [obj.sample(points+i*step*direction, gap=gap+i*step, **SETTINGS)
                  for i in (-4, -3, -2, -1, 1, 2, 3, 4)]
        p, g = [sum(c*v[j] for c, v in zip((3, -32, 168, -672, 672, -168, 32, -3), values))/(840*step)
                for j in range(2)]
        np.testing.assert_allclose(dp, p, atol=3e-9, rtol=0)
        np.testing.assert_allclose(dg, g, atol=3e-9, rtol=0)


@pytest.mark.parametrize('gap', (1e-6, 1e-9, 1e-12))
def test_subpixel_seam_coordinates_retain_the_voltage(gap):
    obj = PeriodicSeamTrace(3)
    center = SETTINGS['period'][0]/2+SETTINGS['offset']
    points = np.array([[center, gap/4], [center, -gap/4]])
    potential, gradient, dp, dg = obj.sample_with_jvp(points, gap=gap, **SETTINGS)
    np.testing.assert_allclose(potential[0]/(gap/4), gradient[0, 1], atol=3e-12, rtol=0)
    np.testing.assert_allclose(potential[1]/(-gap/4), gradient[1, 1], atol=3e-12, rtol=0)
    np.testing.assert_allclose(dp, 0., atol=3e-12, rtol=0)
    np.testing.assert_allclose(dg, 0., atol=3e-12, rtol=0)


def test_higher_quadrature_preserves_positive_gap_values_and_derivatives():
    for gap in (1e-9, .001, 7.):
        values = [PeriodicSeamTrace(9, quadrature=q).normal_response(gap=gap, **SETTINGS) for q in (64, 96)]
        np.testing.assert_allclose(values[0][0], values[1][0], atol=3e-10, rtol=0)
        np.testing.assert_allclose(560*values[0][1], 560*values[1][1], atol=3e-9, rtol=0)


@pytest.mark.parametrize('update', ({'gap': -1.}, {'gap': 560.}, {'width': 0.},
                                     {'period': (0., 560.)}, {'epsilon': -10.},
                                     {'epsilon_tangent': np.nan}))
def test_invalid_response_inputs_are_refused(update):
    with pytest.raises(ValueError):
        PeriodicSeamTrace().normal_response(**dict(dict(SETTINGS, gap=0.), **update))


@pytest.mark.parametrize('arguments', ({'order': True}, {'order': -1}, {'quadrature': 8}))
def test_invalid_basis_inputs_are_refused(arguments):
    with pytest.raises(ValueError):
        PeriodicSeamTrace(**arguments)
