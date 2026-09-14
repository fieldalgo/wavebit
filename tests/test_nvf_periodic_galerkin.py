"""Independent periodic collapse, original weak equations and contact tangents."""
from fractions import Fraction as Q
import numpy as np
import pytest

from wavebit._core.conforming.periodic_rectangle_galerkin import PeriodicRectangleGalerkinGapCell
from test_nvf_pair_galerkin import nodal_reference, tensor, duffy
from test_nvf_pair_quadratic import nodal_gradients
from test_nvf_cartesian_pair_opening import _oracle_polynomial_vandermonde


SETTINGS = dict(period=(700., 560.), width=310., offset=(21., -14.))


def oracle_chart(obj, settings):
    """Rational tensor-partition vertices and independent periodic quotient."""
    length = np.sqrt(np.prod(settings['period']))
    px, py = (Q(float(v/length)) for v in settings['period'])
    width = Q(settings['width']/length)
    offset = [Q(float(v/length)) for v in settings['offset']]
    q = obj.degree*2**obj.plan.refinement
    coordinates = [(x, y) for y in range(3*q+1) for x in range(3*q+1)]
    p0, p1 = [], []
    def interpolate(knots, index):
        segment = min(index//q, 2)
        t = Q(index-segment*q, q)
        return (1-t)*knots[segment]+t*knots[segment+1]
    for x, y in coordinates:
        p0.append([interpolate([Q(0), (px-width)/2, (px+width)/2, px], x)+offset[0],
                   interpolate([Q(0), Q(0), py, py], y)+offset[1]])
        p1.append([Q(0), interpolate([Q(0), Q(1, 2), Q(-1, 2), Q(0)], y)])
    p0, p1 = np.asarray(p0, object), np.asarray(p1, object)
    periodic, ids, collapsed, pmap = [], {}, {}, []
    for coordinate, point in zip(coordinates, p0):
        key = tuple(v % (3*q) for v in coordinate)
        target = ((point[0]-offset[0]) % px, (point[1]-offset[1]) % py)
        if target not in collapsed:
            collapsed[target] = len(collapsed)
        if key not in ids:
            ids[key] = len(ids)
            pmap.append(collapsed[target])
        else:
            assert pmap[ids[key]] == collapsed[target]
        periodic.append(ids[key])
    periodic, pmap = np.asarray(periodic), np.asarray(pmap)
    groups = [np.flatnonzero(pmap == i) for i in range(len(collapsed))]
    c = np.eye(len(collapsed))[pmap][1:, 1:]
    extras = [int(v) for group in groups for v in group[1:]]
    z = np.eye(len(pmap))[:, extras][1:]
    # Material and collapse labels come from physical triangle interiors and
    # areas, not from the candidate tags or its C/Z transformation.
    thin, tags = [], []
    opened = p0+Q(1, 10)*p1
    for triangle in obj.data['triangles']:
        p = p0[triangle[:3]]
        a, b = p[1]-p[0], p[2]-p[0]
        thin.append(a[0]*b[1]-a[1]*b[0] == 0)
        center = opened[triangle[:3]].sum(0)/3-np.asarray(offset, object)
        tags.append(int((px-width)/2 < center[0] < (px+width)/2
                        and Q(1, 20) < center[1] < py-Q(1, 20)))
    return dict(vertices0=p0, vertices1=p1, periodic=periodic, pmap=pmap,
                triangles=obj.data['triangles'], thin=np.asarray(thin), tags=np.asarray(tags), C=c, Z=z), length


@pytest.mark.parametrize('degree', (1, 2))
@pytest.mark.parametrize('refinement', (0, 1))
@pytest.mark.parametrize('gap', (0., 7.))
@pytest.mark.parametrize('epsilon', (12.25, -10.+1j))
def test_original_nodal_and_rational_contact_equations(degree, refinement, gap, epsilon):
    obj = PeriodicRectangleGalerkinGapCell((1, 2), degree=degree, refinement=refinement, coercivity_angle=1.53)
    graph, length = oracle_chart(obj, SETTINGS)
    basis = np.diag(SETTINGS['period'])/length
    outputs = [nodal_reference(graph, (epsilon,), gap, obj.order, q, length, basis, degree=degree)
               for q in (24, 32)]
    h24, d24, z24 = outputs[0]
    h32, d32, z32 = outputs[1]
    slots, derivative, meta = obj.closure(gap=gap, epsilon=epsilon, **SETTINGS)
    np.testing.assert_allclose(h24, h32, atol=3e-10, rtol=0)
    np.testing.assert_allclose(z24, z32, atol=3e-10, rtol=0)
    np.testing.assert_allclose(tensor(slots), h32, atol=3e-10, rtol=0)
    np.testing.assert_allclose(slots.epz_i[0], z32, atol=3e-10, rtol=0)
    if gap == 0:
        np.testing.assert_allclose(length*d24, length*d32, atol=3e-9, rtol=0)
        np.testing.assert_allclose(length*tensor(derivative), length*d32, atol=3e-9, rtol=0)
    assert max(v for k, v in meta.items() if k.endswith('residual')) < 1e-10


@pytest.mark.parametrize('degree', (1, 2))
def test_collapsing_gauge_group_keeps_its_physical_difference_potentials(degree):
    obj = PeriodicRectangleGalerkinGapCell((1, 2), degree=degree, refinement=1)
    graph, _ = oracle_chart(obj, SETTINGS)
    expected = np.column_stack([graph['C'], graph['Z']])
    assert np.linalg.matrix_rank(expected) == len(expected)
    assert len(np.flatnonzero(obj.data['pmap'] == 0)) > 1
    assert any(obj.data['pmap'][i] == 0 for i in obj.data['extras'])


@pytest.mark.parametrize('gap', (0., 7., 1e-9))
def test_uniform_medium_and_material_direction(gap):
    obj = PeriodicRectangleGalerkinGapCell((1, 2), refinement=1, coercivity_angle=1.53)
    epsilon = -10.+1j
    slots, derivative, _ = obj.closure(gap=gap, epsilon=epsilon, background=epsilon,
                                       gap_tangent=.7, epsilon_tangent=1., background_tangent=1., **SETTINGS)
    I = np.eye(15)
    np.testing.assert_allclose(slots, np.array([epsilon*I, epsilon*I, I/epsilon, 0*I, 0*I])[:, None],
                               atol=3e-10, rtol=0)
    np.testing.assert_allclose(derivative, np.array([I, I, -I/epsilon**2, 0*I, 0*I])[:, None],
                               atol=3e-10, rtol=0)


@pytest.mark.parametrize('refinement', (1, 2, 3))
def test_contact_taylor_limit_and_subpixel_gap(refinement):
    obj = PeriodicRectangleGalerkinGapCell((1, 0), refinement=refinement)
    base, slope, _ = obj.closure(gap=0., **SETTINGS)
    errors = []
    for gap in (.02, .01, .005):
        value, _, _ = obj.closure(gap=gap, **SETTINGS)
        errors.append(np.max(abs(np.asarray(value)-base-gap*np.asarray(slope))))
    assert errors[1] < .3*errors[0]+3e-10
    assert errors[2] < .3*errors[1]+3e-10
    small, _, _ = obj.closure(gap=1e-9, **SETTINGS)
    np.testing.assert_allclose(small, np.asarray(base)+1e-9*np.asarray(slope), atol=3e-10, rtol=0)


@pytest.mark.parametrize('gap', (0., 7.))
def test_holomorphic_material_jvp_and_simultaneous_gap_direction(gap):
    obj = PeriodicRectangleGalerkinGapCell((1, 2), refinement=1, coercivity_angle=1.53)
    settings = dict(SETTINGS, gap=gap, epsilon=-10+1j, background=1+.1j)
    direction, bgdirection = .3+.2j, -.2+.1j
    _, material, _ = obj.closure(**settings, gap_tangent=0., epsilon_tangent=direction,
                                 background_tangent=bgdirection)
    _, imaginary, _ = obj.closure(**settings, gap_tangent=0., epsilon_tangent=1j*direction,
                                  background_tangent=1j*bgdirection)
    np.testing.assert_allclose(imaginary, 1j*np.asarray(material), atol=3e-10, rtol=0)
    _, dg, _ = obj.closure(**settings)
    _, both, _ = obj.closure(**settings, epsilon_tangent=direction, background_tangent=bgdirection)
    np.testing.assert_allclose(both, np.asarray(dg)+material, atol=3e-10, rtol=0)
    for h in (.002, .001):
        values = [np.asarray(obj.closure(**dict(settings, epsilon=settings['epsilon']+i*h*direction,
                                               background=settings['background']+i*h*bgdirection))[0])
                  for i in (-2, -1, 1, 2)]
        np.testing.assert_allclose((values[0]-8*values[1]+8*values[2]-values[3])/(12*h), material,
                                   atol=3e-9, rtol=0)


@pytest.mark.parametrize('update', ({'gap': -1.}, {'gap': 560.}, {'width': 700.},
                                     {'width': 0.}, {'period': (0., 560.)}, {'epsilon': -10.}))
def test_invalid_geometry_and_material_are_refused(update):
    obj = PeriodicRectangleGalerkinGapCell()
    with pytest.raises(ValueError):
        obj.closure(**dict(dict(SETTINGS, gap=0.), **update))


def test_finest_contact_uses_the_original_laurent_response():
    obj = PeriodicRectangleGalerkinGapCell((1, 0), refinement=3)
    graph, length = oracle_chart(obj, SETTINGS)
    basis = np.diag(SETTINGS['period'])/length
    h24, d24, _ = nodal_reference(graph, (12.25,), 0., obj.order, 24, length, basis)
    h32, d32, _ = nodal_reference(graph, (12.25,), 0., obj.order, 32, length, basis)
    slots, derivative, _ = obj.closure(gap=0., **SETTINGS)
    np.testing.assert_allclose(h24, h32, atol=3e-10, rtol=0)
    np.testing.assert_allclose(tensor(slots), h32, atol=3e-10, rtol=0)
    np.testing.assert_allclose(length*d24, length*d32, atol=3e-9, rtol=0)
    np.testing.assert_allclose(length*tensor(derivative), length*d32, atol=3e-9, rtol=0)


@pytest.mark.parametrize('scale', (1e-4, 1e4))
def test_physical_units_preserve_the_tensor_and_rescale_its_gap_derivative(scale):
    obj = PeriodicRectangleGalerkinGapCell((1, 2), refinement=1)
    base, derivative, _ = obj.closure(gap=7., **SETTINGS)
    scaled = {k: np.asarray(v)*scale for k, v in SETTINGS.items()}
    value, tangent, _ = obj.closure(gap=7.*scale, **scaled)
    np.testing.assert_allclose(base, value, atol=3e-10, rtol=0)
    np.testing.assert_allclose(derivative, scale*np.asarray(tangent), atol=3e-10, rtol=0)


def test_nested_periodic_spaces_lower_the_lossless_constitutive_energy():
    previous = None
    for refinement in range(3):
        obj = PeriodicRectangleGalerkinGapCell((1, 2), refinement=refinement)
        slots, _, metadata = obj.closure(gap=0., **SETTINGS)
        current = tensor(slots)
        assert metadata['minimum_rotated_storage'] >= 1.-1e-9
        if previous is not None:
            assert np.linalg.eigvalsh(previous-current)[0] > -1e-9
        previous = current


def trace_projection_reference(obj, epsilon, background):
    """Independent leading gap energy in the k_y=0 normal-field sector.

    The finite trace projection and the unrestricted L2 interface response
    differ by a positive semidefinite Gram error. No optical result or Li
    factorizer supplies this reference.
    """
    graph, length = oracle_chart(obj, SETTINGS)
    z = np.vstack([np.zeros((1, graph['Z'].shape[1])), graph['Z']])
    count = z.shape[1]
    modes = np.arange(-obj.order[0], obj.order[0]+1)
    mass = np.zeros((count, count), complex)
    forcing = np.zeros((count, len(modes)), complex)
    bary, weights = duffy(24)
    gradient = (nodal_gradients(bary) if obj.degree == 2 else
                np.broadcast_to(np.array([[-1., -1.], [1., 0.], [0., 1.]]), (len(bary), 3, 2)))
    px = SETTINGS['period'][0]/length
    center = (SETTINGS['period'][0]/2+SETTINGS['offset'][0])/length
    width = SETTINGS['width']/length
    for triangle, thin in zip(graph['triangles'], graph['thin']):
        if not thin:
            continue
        det, numerator = _oracle_polynomial_vandermonde(
            graph['vertices0'][triangle[:3]], graph['vertices1'][triangle[:3]])
        points = np.asarray(graph['vertices0'][triangle[:3]], float)
        local_z = z[graph['periodic'][triangle]]
        columns = np.flatnonzero(np.any(local_z, axis=0))
        normal = np.einsum('qia,ij->qaj', gradient@np.asarray(numerator[0, 1:], float),
                           local_z[:, columns])
        mass[np.ix_(columns, columns)] += background*np.einsum(
            'q,qai,qaj->ij', weights, normal, normal)/float(det[1])
        if abs(points[:, 0].mean()-center) < width/2:
            phase = np.exp(2j*np.pi*(bary@points)[:, 0, None]*modes[None]/px)
            forcing[columns] += (background-epsilon)*np.einsum(
                'q,qi,qm->im', weights, normal[:, 1], phase)
    difference = modes[:, None]-modes
    fraction = SETTINGS['width']/SETTINGS['period'][0]
    indicator = (fraction*np.sinc(difference*fraction)
                 *np.exp(-2j*np.pi*difference*(.5+SETTINGS['offset'][0]/SETTINGS['period'][0])))
    direct = (background-epsilon)*indicator/SETTINGS['period'][1]
    finite = direct-forcing.conj().T@np.linalg.solve(mass, forcing)/length
    unrestricted = (epsilon-epsilon**2/background)*indicator/SETTINGS['period'][1]
    return finite, unrestricted


@pytest.mark.parametrize('degree', (1, 2))
@pytest.mark.parametrize('refinement', (0, 1, 2))
@pytest.mark.parametrize('epsilon,background', ((12.25, 1.), (4., 2.25)))
def test_normal_contact_derivative_is_the_finite_gap_trace_projection(degree, refinement, epsilon, background):
    obj = PeriodicRectangleGalerkinGapCell((2, 2), degree=degree, refinement=refinement)
    finite, unrestricted = trace_projection_reference(obj, epsilon, background)
    _, derivative, _ = obj.closure(gap=0., epsilon=epsilon, background=background, **SETTINGS)
    size = (2*obj.order[0]+1)*(2*obj.order[1]+1)
    indices = size+obj.order[1]*(2*obj.order[0]+1)+np.arange(2*obj.order[0]+1)
    actual = tensor(derivative)[np.ix_(indices, indices)]
    np.testing.assert_allclose(actual, finite, atol=3e-9/np.sqrt(np.prod(SETTINGS['period'])), rtol=0)
    error = finite-unrestricted
    np.testing.assert_allclose(error, error.conj().T, atol=3e-10, rtol=0)
    assert np.linalg.eigvalsh(error)[0] > -1e-10
