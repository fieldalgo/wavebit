"""Independent original nodal coupling, physical energy, and contact traces."""
import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from wavebit.rcwa._engine.conforming.periodic_seam_galerkin import PeriodicSeamGalerkinGapCell, _free_fourier_energy
from wavebit.rcwa._engine.conforming.periodic_rectangle_galerkin import PeriodicRectangleGalerkinGapCell
from tests.rcwa.test_nvf_periodic_galerkin import oracle_chart
from tests.rcwa.test_nvf_pair_quadratic import nodal_gradients
from tests.rcwa.test_nvf_pair_galerkin import tensor
from tests.rcwa.test_nvf_seam_trace import contact_reference

pytestmark = pytest.mark.research


SETTINGS = dict(period=(700., 560.), width=310., offset=(21., -14.))


def original_reference(gap, epsilon, quadrature, order=(1, 2)):
    """Raw free nodal unknowns plus raw standing waves; no C/Z scaling.

    Rational chart data supplies physical vertices. Shape gradients come from
    an independent six-node Vandermonde, and all material/retained integrals
    are evaluated together. No candidate coupling or sampled lift is used.
    """
    obj = PeriodicRectangleGalerkinGapCell(order, refinement=0)
    graph, length = oracle_chart(obj, SETTINGS)
    points = np.asarray(graph['vertices0'], float)*length+gap*np.asarray(graph['vertices1'], float)
    nodes = len(graph['pmap'])-1
    nx, ny = order
    modes = np.array([(m, n) for n in range(-ny, ny+1) for m in range(-nx, nx+1)])
    count, extra = len(modes), 2*nx+1
    size = nodes+extra
    pxy = np.asarray(SETTINGS['period'])
    width = SETTINGS['width']
    ox, oy = SETTINGS['offset']
    center = pxy[0]/2+ox
    a, b = center-width/2, center+width/2
    cap = min(width, pxy[0]-width)/4
    raw = np.sqrt(gap*(pxy[1]-gap)/12)
    delta = cap*raw/(cap+raw)
    knots = [a-delta, a, a+delta, b-delta, b, b+delta]
    t, w = leggauss(quadrature)
    t, w = (t+1)/2, w/2
    k = np.zeros((size, size), complex)
    q, r = np.zeros((2*count, size), complex), np.zeros((2*count, size), complex)
    conv = np.zeros((count, count), complex)
    for number, triangle in enumerate(graph['triangles']):
        vertex = points[triangle[:3]]
        jacobian = np.column_stack([vertex[1]-vertex[0], vertex[2]-vertex[0]])
        determinant = np.linalg.det(jacobian)
        xl, xr = vertex[:, 0].min(), vertex[:, 0].max()
        bottom, top = vertex[:, 1].min(), vertex[:, 1].max()
        cuts = [xl]+[x for x in knots if xl < x < xr]+[xr]
        for left, right in zip(cuts, cuts[1:]):
            x = left+(right-left)*t
            u = (x-xl)/(xr-xl)
            if number % 2 == 0:
                bary = np.stack([np.broadcast_to(1-u[:, None], (quadrature, quadrature)),
                                 u[:, None]*(1-t), u[:, None]*t], axis=-1)
                y = bottom+(top-bottom)*u[:, None]*t
                weights = determinant*(right-left)/(xr-xl)*u[:, None]*w[:, None]*w
            else:
                bary = np.stack([(1-u[:, None])*(1-t), np.broadcast_to(u[:, None], (quadrature, quadrature)),
                                 (1-u[:, None])*t], axis=-1)
                y = bottom+(top-bottom)*(u[:, None]+(1-u[:, None])*t)
                weights = determinant*(right-left)/(xr-xl)*(1-u[:, None])*w[:, None]*w
            bary = bary.reshape(-1, 3)
            x = np.broadcast_to(x[:, None], y.shape).ravel()
            y, weights = y.ravel(), weights.ravel()/np.prod(pxy)
            gradient = np.zeros((len(weights), 2, size))
            # Give the free nodal and seam potentials the same length unit.
            # This invertible scalar basis change avoids an artificial 1/area
            # conditioning penalty; it introduces no collapse transformation.
            raw_gradient = length*nodal_gradients(bary)@np.linalg.inv(jacobian)
            for local, index in enumerate(graph['periodic'][triangle]):
                if index:
                    gradient[:, :, index-1] += raw_gradient[:, local]
            phase = np.exp(2j*np.pi*np.column_stack([x, y])@(modes/pxy).T)
            distance = np.abs(x-center)
            fraction = np.clip((width/2+delta-distance)/(2*delta), 0., 1.)
            hat = fraction**2*(3-2*fraction)
            hx = np.where((distance > width/2-delta) & (distance < width/2+delta),
                          -np.sign(x-center)*3*fraction*(1-fraction)/delta, 0.)
            frequency = 2*np.pi*np.arange(1, nx+1)/pxy[0]
            angle = (x-center)[:, None]*frequency
            basis = np.column_stack([np.ones(len(x)), np.cos(angle), np.sin(angle)])
            bx = np.column_stack([np.zeros(len(x)), -frequency*np.sin(angle), frequency*np.cos(angle)])
            trace, trace_x = hat[:, None]*basis, hx[:, None]*basis+hat[:, None]*bx
            local_y = y-oy
            inside = (local_y > gap/2) & (local_y < pxy[1]-gap/2)
            f = np.where(inside, gap*(pxy[1]/2-local_y)/(pxy[1]-gap),
                         np.where(local_y < gap/2, local_y, local_y-pxy[1]))
            fy = np.where(inside, -gap/(pxy[1]-gap), 1.)
            gradient[:, 0, nodes:] = f[:, None]*trace_x
            gradient[:, 1, nodes:] = fy[:, None]*trace
            material = np.where((x > a) & (x < b) & inside, epsilon, 1.)
            k += np.einsum('q,q,qai,qaj->ij', weights, material, gradient, gradient)
            q += np.einsum('q,qm,qai->ami', weights, phase.conj(), gradient).reshape(2*count, size)
            r += np.einsum('q,q,qm,qai->ami', weights, material, phase.conj(), gradient).reshape(2*count, size)
            conv += np.einsum('q,q,qm,qn->mn', weights, material, phase.conj(), phase)
    reverse = np.r_[np.arange(count-1, -1, -1), np.arange(2*count-1, count-1, -1)]
    direct = np.kron(np.eye(2), conv)
    source = r[reverse].T
    left, right = r-direct@q, source-q.conj().T@direct
    weak = k-source@q-q.conj().T@left
    return direct-left@np.linalg.solve(weak, right)


@pytest.mark.parametrize('epsilon', (12.25, -10+1j))
def test_complete_tensor_matches_original_unscaled_nodal_equations(epsilon):
    references = [original_reference(7., epsilon, q) for q in (20, 28)]
    value, _, _ = PeriodicSeamGalerkinGapCell((1, 2), quadrature=24, coercivity_angle=1.53).closure(
        gap=7., epsilon=epsilon, **SETTINGS)
    np.testing.assert_allclose(references[0], references[1], atol=3e-10, rtol=0)
    np.testing.assert_allclose(tensor(value), references[1], atol=3e-10, rtol=0)


@pytest.mark.parametrize('epsilon', (12.25, -10+1j))
@pytest.mark.parametrize('refinement', (0, 1, 2))
def test_contact_keeps_the_closed_tensor_and_supplies_the_exact_normal_trace(epsilon, refinement):
    order = (2, 1)
    obj = PeriodicSeamGalerkinGapCell(order, refinement=refinement, quadrature=24, coercivity_angle=1.53)
    value, derivative, metadata = obj.closure(gap=0., epsilon=epsilon, **SETTINGS)
    old = PeriodicRectangleGalerkinGapCell(order, refinement=refinement, coercivity_angle=1.53).closure(
        gap=0., epsilon=epsilon, **SETTINGS)[0]
    np.testing.assert_allclose(value, old, atol=3e-10, rtol=0)
    _, expected = contact_reference(2, period=SETTINGS['period'], width=SETTINGS['width'],
                                     offset=SETTINGS['offset'][0], epsilon=epsilon, background=1.)
    count = 15
    indices = count+5+np.arange(5)
    np.testing.assert_allclose(560*tensor(derivative)[np.ix_(indices, indices)], 560*expected, atol=3e-9, rtol=0)
    assert metadata['energy_derivative_residual'] < 1e-10
    assert metadata['weak_derivative_residual'] is None


@pytest.mark.parametrize('epsilon', (12.25, -10+1j))
def test_positive_gap_approaches_the_same_full_contact_derivative(epsilon):
    obj = PeriodicSeamGalerkinGapCell((1, 2), quadrature=24, coercivity_angle=1.53)
    base, tangent, _ = obj.closure(gap=0., epsilon=epsilon, **SETTINGS)
    errors = []
    for gap in (1e-4, 1e-6, 1e-8):
        value, derivative, metadata = obj.closure(gap=gap, epsilon=epsilon, **SETTINGS)
        errors.append(np.linalg.norm(np.asarray(derivative)-tangent))
        assert np.linalg.norm(np.asarray(value)-base-gap*np.asarray(tangent)) < 2*gap*errors[-1]+3e-10
        assert max(v for k, v in metadata.items() if k.endswith('residual')) < 1e-10
    assert errors[1] < .15*errors[0]
    assert errors[2] < .15*errors[1]


@pytest.mark.parametrize('gap', (0., 7., 1e-9))
def test_uniform_material_and_holomorphic_simultaneous_direction(gap):
    obj = PeriodicSeamGalerkinGapCell((1, 0), quadrature=20, coercivity_angle=1.53)
    e = -10+1j
    value, derivative, _ = obj.closure(gap=gap, epsilon=e, background=e,
                                      epsilon_tangent=1., background_tangent=1., **SETTINGS)
    expected = np.array([e*np.eye(3), e*np.eye(3), np.eye(3)/e, np.zeros((3, 3)), np.zeros((3, 3))])[:, None]
    expected_derivative = np.array([np.eye(3), np.eye(3), -np.eye(3)/e**2, np.zeros((3, 3)), np.zeros((3, 3))])[:, None]
    np.testing.assert_allclose(value, expected, atol=3e-10, rtol=0)
    np.testing.assert_allclose(derivative, expected_derivative, atol=3e-10, rtol=0)
    settings = dict(SETTINGS, gap=gap, epsilon=e, background=1+.1j)
    direction = dict(epsilon_tangent=.3+.2j, background_tangent=-.1+.2j)
    _, dm, _ = obj.closure(**settings, gap_tangent=0., **direction)
    _, di, _ = obj.closure(**settings, gap_tangent=0., **{k: 1j*v for k, v in direction.items()})
    _, dg, _ = obj.closure(**settings)
    _, both, _ = obj.closure(**settings, **direction)
    np.testing.assert_allclose(di, 1j*np.asarray(dm), atol=3e-10, rtol=0)
    np.testing.assert_allclose(both, np.asarray(dm)+dg, atol=3e-10, rtol=0)


@pytest.mark.parametrize('quadrature', (28, 36))
def test_independent_moving_nodal_derivative(quadrature):
    obj = PeriodicSeamGalerkinGapCell((1, 2), quadrature=24)
    _, derivative, _ = obj.closure(gap=7., **SETTINGS)
    # Keep differencing above the raw nodal solve's roundoff floor. The
    # tenth-order stencil permits larger steps with the same 3e-9 guard.
    for h in (.2, .1):
        indices = (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)
        weights = (-2, 25, -150, 600, -2100, 2100, -600, 150, -25, 2)
        values = [original_reference(7+i*h, 12.25, quadrature) for i in indices]
        expected = sum(c*v for c, v in zip(weights, values))/(2520*h)
        np.testing.assert_allclose(560*tensor(derivative), 560*expected, atol=3e-9, rtol=0)


@pytest.mark.parametrize('gap', (0., 7., 1e-9))
def test_units_translation_and_quadrature_preserve_the_coupled_tensor(gap):
    obj = PeriodicSeamGalerkinGapCell((1, 0), quadrature=24, coercivity_angle=1.53)
    settings = dict(SETTINGS, gap=gap, epsilon=-10+1j)
    value, derivative, _ = obj.closure(**settings)
    other, change, _ = PeriodicSeamGalerkinGapCell((1, 0), quadrature=32, coercivity_angle=1.53).closure(**settings)
    np.testing.assert_allclose(other, value, atol=3e-10, rtol=0)
    np.testing.assert_allclose(560*np.asarray(change), 560*np.asarray(derivative), atol=3e-9, rtol=0)
    for scale in (1e-4, 1e4):
        updated = {k: np.asarray(v)*scale for k, v in SETTINGS.items()}
        other, change, _ = obj.closure(gap=gap*scale, epsilon=-10+1j, **updated)
        np.testing.assert_allclose(other, value, atol=3e-10, rtol=0)
        np.testing.assert_allclose(scale*np.asarray(change), derivative, atol=3e-10, rtol=0)
    other, change, _ = obj.closure(**dict(settings, offset=np.asarray(SETTINGS['offset'])+(91., -37.)))
    phase = np.exp(-2j*np.pi*np.arange(-1, 2)*91/700)
    expected = phase[:, None]*np.asarray(value)*phase.conj()
    expected_tangent = phase[:, None]*np.asarray(derivative)*phase.conj()
    np.testing.assert_allclose(other, expected, atol=3e-10, rtol=0)
    np.testing.assert_allclose(change, expected_tangent, atol=3e-10, rtol=0)


@pytest.mark.parametrize('epsilon', (12.25, -10+1j))
def test_double_retained_frequency_does_not_alias_the_energy_check(epsilon):
    values = [PeriodicSeamGalerkinGapCell((0, 7), quadrature=q, coercivity_angle=1.53).closure(
        gap=.001, epsilon=epsilon, **SETTINGS) for q in (32, 48)]
    np.testing.assert_allclose(values[0][0], values[1][0], atol=3e-10, rtol=0)
    np.testing.assert_allclose(560*np.asarray(values[0][1]), 560*np.asarray(values[1][1]), atol=3e-9, rtol=0)
    for _, _, metadata in values:
        assert metadata['energy_residual'] < 1e-10
        assert metadata['energy_derivative_residual'] < 1e-10
        assert metadata['direct_fourier_residual'] < 1e-10
        assert metadata['direct_fourier_derivative_residual'] < 1e-10


def test_exact_free_energy_matches_independent_rectangle_quadrature_and_variation():
    order = (2, 3)
    settings = dict(SETTINGS, gap=17., epsilon=-10+1j, background=1+.1j,
                    gap_tangent=.7, epsilon_tangent=.2+.3j, background_tangent=-.1+.2j)
    value, derivative = _free_fourier_energy(order, settings)
    modes = np.array([(m, n) for n in range(-3, 4) for m in range(-2, 3)])
    node, weight = leggauss(40)
    def reference(step):
        gap = settings['gap']+step*settings['gap_tangent']
        epsilon = settings['epsilon']+step*settings['epsilon_tangent']
        background = settings['background']+step*settings['background_tangent']
        px, py = settings['period']
        x = px/2+settings['offset'][0]+settings['width']*node/2
        y = py/2+settings['offset'][1]+(py-gap)*node/2
        xx, yy = np.meshgrid(x, y, indexing='ij')
        phase = np.exp(2j*np.pi*np.column_stack([xx.ravel()/px, yy.ravel()/py])@modes.T)
        weights = (weight[:, None]*weight[None]*settings['width']*(py-gap)/(4*px*py)).ravel()
        indicator = phase.conj().T@(weights[:, None]*phase)
        material = background*np.eye(len(modes))+(epsilon-background)*indicator
        return np.kron(np.eye(2), material)
    np.testing.assert_allclose(value, reference(0.), atol=3e-10, rtol=0)
    for h in (.002, .001):
        samples = [reference(i*h) for i in (-2, -1, 1, 2)]
        expected = (samples[0]-8*samples[1]+8*samples[2]-samples[3])/(12*h)
        np.testing.assert_allclose(derivative, expected, atol=3e-9, rtol=0)


@pytest.mark.parametrize('update', ({'gap': -1.}, {'gap': 560.}, {'width': 0.},
                                     {'epsilon': -10.}, {'epsilon_tangent': np.nan}))
def test_invalid_inputs_are_refused(update):
    with pytest.raises(ValueError):
        PeriodicSeamGalerkinGapCell().closure(**dict(dict(SETTINGS, gap=0.), **update))


@pytest.mark.parametrize('gap', (7., 1e-9))
@pytest.mark.parametrize('epsilon', (12.25, -10+1j))
def test_positive_eliminations_preserve_simultaneous_gap_and_material_direction(gap, epsilon):
    from wavebit.rcwa._engine.conforming.periodic_seam_galerkin import _quadrature
    from wavebit.rcwa._engine.conforming.real_trace_coupling import coupled_positive
    obj = PeriodicSeamGalerkinGapCell((1, 0), refinement=1, quadrature=32, coercivity_angle=1.53)
    settings = dict(SETTINGS, gap=gap, epsilon=epsilon, background=1., gap_tangent=.7,
                    epsilon_tangent=.3+.2j, background_tangent=.2+.3j)
    period = np.asarray(settings['period'])
    base = obj._base._assemble(gap=gap, basis=np.diag(period), widths=(settings['width']/2,)*2,
        height=period[1], epsilon=(epsilon,)*2, background=1., offset=settings['offset'],
        gap_tangent=.7, epsilon_tangent=(settings['epsilon_tangent'],)*2,
        background_tangent=settings['background_tangent'])
    quad = lambda split=True: _quadrature(obj._base, obj._trace, base, settings, 32, split)
    values = [coupled_positive(obj._base, base, quad, _free_fourier_energy(obj.order, settings),
                              obj.order[0], elimination=form) for form in ('potential', 'retained')]
    np.testing.assert_allclose(values[0][0], values[1][0], atol=3e-10, rtol=0)
    np.testing.assert_allclose(values[0][1], values[1][1], atol=3e-9, rtol=0)
    for _, _, meta in values:
        assert max(v for k,v in meta.items() if k.endswith('residual')) < 1e-10
    _, _, default = obj.closure(**settings)
    assert default['elimination'] == 'retained'
