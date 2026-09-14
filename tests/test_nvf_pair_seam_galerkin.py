"""Independent nodal coupling and physical variations of real pair gap traces."""
from functools import lru_cache
import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from wavebit._core.conforming.cartesian_pair_seam_galerkin import CartesianPairSeamGalerkinGapCell
from wavebit._core.conforming.cartesian_pair_galerkin import CartesianPairGalerkinGapCell
from wavebit._core.conforming.cartesian_pair_opening import CartesianPairOpeningPlan
from test_nvf_cartesian_pair_opening import _oracle_geometry
from test_nvf_pair_quadratic import nodal_gradients
from test_nvf_pair_galerkin import tensor

GEOMETRY = dict(basis=np.diag([700., 560.]), widths=(170., 230.), height=280., offset=(21., -14.))


def cell(order=(1, 2), refinement=0, quadrature=24):
    return CartesianPairSeamGalerkinGapCell(CartesianPairOpeningPlan(0, refinement), order,
                                           quadrature=quadrature, coercivity_angle=1.53)


def clip(polygon, vertex, axis, bound, sign):
    """Independent convex half-plane clipping in original barycentric coordinates."""
    result = []
    for a, b in zip(polygon, polygon[1:]+polygon[:1]):
        da = sign*((a@vertex)[axis]-bound)
        db = sign*((b@vertex)[axis]-bound)
        if da >= 0:
            result.append(a)
        if (da < 0 < db) or (db < 0 < da):
            result.append(a+(b-a)*da/(da-db))
    return result


@lru_cache(None)
def reference(gap, epsilon, quadrature, background=1.):
    """Raw nodal variables, raw sine/cosine traces and clipped physical triangles.

    The reference uses a six-node Vandermonde, an independent polygon partition,
    and one unscaled constrained solve. No production trace basis, gradient,
    sweep, contact formula or coupled matrix is used.
    """
    obj = cell()._base
    length = np.sqrt(np.linalg.det(GEOMETRY['basis']))
    settings = {k: np.asarray(v)/length for k, v in GEOMETRY.items()}
    graph = _oracle_geometry(obj.data, settings)
    points = np.asarray(graph['vertices0'], float)+gap/length*np.asarray(graph['vertices1'], float)
    period = np.diag(settings['basis'])
    a, b = settings['widths']; height = settings['height']; s = gap/length
    center = period/2+settings['offset']+[(a-b)/2, 0.]
    points -= center
    nodes = len(graph['pmap'])-1
    order = (1, 2)
    modes = np.array([(m, n) for n in range(-2, 3) for m in range(-1, 2)])
    count, extra = len(modes), 5
    size = nodes+extra
    raw = np.sqrt(s*(period[0]-s)/12)
    cap = min(height, period[1]-height)/4
    delta = cap*raw/(cap+raw)
    ys = [-height/2-delta, -height/2, -height/2+delta, height/2-delta, height/2, height/2+delta]
    xs = [-a-s/2, -s/2, s/2, b+s/2]
    t, w = leggauss(quadrature); t, w = (t+1)/2, w/2
    qb = np.stack([np.broadcast_to(1-t[:, None], (quadrature, quadrature)),
                   t[:, None]*(1-t), t[:, None]*t], axis=-1).reshape(-1, 3)
    qw = (t[:, None]*w[:, None]*w).ravel()
    k = np.zeros((size, size), complex)
    q, r = np.zeros((2*count, size), complex), np.zeros((2*count, size), complex)
    conv = np.zeros((count, count), complex)
    for number, triangle in enumerate(graph['triangles']):
        vertex = points[triangle[:3]]
        jac = np.column_stack([vertex[1]-vertex[0], vertex[2]-vertex[0]])
        determinant = np.linalg.det(jac)
        xcuts = [vertex[:, 0].min()]+[x for x in xs if vertex[:, 0].min() < x < vertex[:, 0].max()]+[vertex[:, 0].max()]
        ycuts = [vertex[:, 1].min()]+[y for y in ys if vertex[:, 1].min() < y < vertex[:, 1].max()]+[vertex[:, 1].max()]
        for xl, xr in zip(xcuts, xcuts[1:]):
            for yl, yr in zip(ycuts, ycuts[1:]):
                poly = list(np.eye(3))
                for axis, bound, sign in ((0, xl, 1), (0, xr, -1), (1, yl, 1), (1, yr, -1)):
                    poly = clip(poly, vertex, axis, bound, sign)
                for i in range(1, len(poly)-1):
                    local = np.array([poly[0], poly[i], poly[i+1]])
                    area = abs(np.linalg.det(local))
                    if area == 0:
                        continue
                    bary = qb@local
                    xy = bary@vertex
                    x, y = xy.T
                    weights = qw*area*determinant
                    gradient = np.zeros((len(weights), 2, size))
                    raw_gradient = nodal_gradients(bary)@np.linalg.inv(jac)
                    for local_id, index in enumerate(graph['periodic'][triangle]):
                        if index:
                            gradient[:, :, index-1] += raw_gradient[:, local_id]
                    fraction = np.clip((height/2+delta-np.abs(y))/(2*delta), 0., 1.)
                    hat = fraction**2*(3-2*fraction)
                    hy = np.where((np.abs(y) > height/2-delta) & (np.abs(y) < height/2+delta),
                                  -np.sign(y)*3*fraction*(1-fraction)/delta, 0.)
                    frequency = 2*np.pi*np.arange(1, 3)/period[1]
                    angle = y[:, None]*frequency
                    basis = np.column_stack([np.ones(len(y)), np.cos(angle), np.sin(angle)])
                    by = np.column_stack([np.zeros(len(y)), -frequency*np.sin(angle), frequency*np.cos(angle)])
                    trace = hat[:, None]*basis
                    trace_y = hy[:, None]*basis+hat[:, None]*by
                    f, fx = np.zeros_like(x), np.zeros_like(x)
                    left = (x > -a-s/2) & (x < -s/2)
                    middle = (x > -s/2) & (x < s/2)
                    right = (x > s/2) & (x < b+s/2)
                    f[left], fx[left] = -s*(x[left]+a+s/2)/(2*a), -s/(2*a)
                    f[middle], fx[middle] = x[middle], 1.
                    f[right], fx[right] = s*(b+s/2-x[right])/(2*b), -s/(2*b)
                    gradient[:, 0, nodes:] = fx[:, None]*trace
                    gradient[:, 1, nodes:] = f[:, None]*trace_y
                    phase = np.exp(2j*np.pi*(xy+center)@(modes/period).T)
                    material = [background, *epsilon][graph['tags'][number]]
                    k += material*np.einsum('q,qai,qaj->ij', weights, gradient, gradient, optimize=True)
                    moment = np.einsum('q,qm,qai->ami', weights, phase.conj(), gradient, optimize=True).reshape(2*count, size)
                    q += moment; r += material*moment
                    conv += material*(phase.conj().T@(weights[:, None]*phase))
    reverse = np.r_[np.arange(count-1, -1, -1), np.arange(2*count-1, count-1, -1)]
    direct = np.kron(np.eye(2), conv)
    source = r[reverse].T
    weak = k-source@q-q.conj().T@(r-direct@q)
    return direct-(r-direct@q)@np.linalg.solve(weak, source-q.conj().T@direct)


@pytest.mark.parametrize('epsilon', ((4., 9.), (12.25, -10+1j)))
def test_full_tensor_matches_independent_original_nodal_and_trace_equations(epsilon):
    refs = [reference(7., epsilon, q) for q in (16, 24)]
    np.testing.assert_allclose(refs[0], refs[1], atol=3e-10, rtol=0)
    value, _, _ = cell().closure(gap=7., epsilon=epsilon, **GEOMETRY)
    np.testing.assert_allclose(tensor(value), refs[1], atol=3e-10, rtol=0)


def test_gap_jvp_matches_variations_of_the_independent_clipped_reference():
    epsilon = (4., 9.)
    _, direction, _ = cell().closure(gap=7., epsilon=epsilon, **GEOMETRY)
    for h in (.04, .02):
        samples = [reference(7.+i*h, epsilon, 24) for i in (-2, -1, 1, 2)]
        expected = (samples[0]-8*samples[1]+8*samples[2]-samples[3])/(12*h)
        np.testing.assert_allclose(560*tensor(direction), 560*expected, atol=3e-9, rtol=0)


@pytest.mark.parametrize('epsilon', ((4., 9.), (12.25, -10+1j)))
@pytest.mark.parametrize('refinement', (0, 1))
def test_exact_contact_preserves_the_closed_mesh_value_and_physical_guards(epsilon, refinement):
    obj = cell(refinement=refinement)
    value, tangent, metadata = obj.closure(gap=0., epsilon=epsilon, **GEOMETRY)
    base, _, _ = obj._base.closure(gap=0., epsilon=epsilon, **GEOMETRY)
    np.testing.assert_allclose(value, base, atol=3e-10, rtol=0)
    assert metadata['weak_derivative_residual'] is None
    assert metadata['contact_derivative_residual'] < 1e-10
    assert metadata['energy_derivative_residual'] < 1e-10
    assert np.isfinite(tangent).all()


@pytest.mark.parametrize('epsilon', ((4., 9.), (12.25, -10+1j)))
def test_positive_gap_jvp_approaches_the_exact_contact_with_sqrt_gap_remainder(epsilon):
    obj = cell()
    closed, derivative, _ = obj.closure(gap=0., epsilon=epsilon, **GEOMETRY)
    errors = []
    for gap in (.001, 1e-5, 1e-9):
        value, tangent, metadata = obj.closure(gap=gap, epsilon=epsilon, **GEOMETRY)
        errors.append(np.linalg.norm(np.asarray(tangent)-derivative))
        assert max(v for k, v in metadata.items() if k.endswith('residual')) < 1e-10
        assert np.linalg.norm(np.asarray(value)-closed) < 2*gap*np.linalg.norm(derivative)
    assert errors[1] < .2*errors[0]
    assert errors[2] < .02*errors[1]


@pytest.mark.parametrize('gap', (0., 7., 1e-9))
def test_uniform_material_has_no_auxiliary_correction(gap):
    epsilon = 3+.4j
    value, tangent, _ = cell().closure(gap=gap, epsilon=(epsilon, epsilon), background=epsilon, **GEOMETRY)
    count = len(value[0][0])
    np.testing.assert_allclose(tensor(value), epsilon*np.eye(2*count), atol=3e-10, rtol=0)
    np.testing.assert_allclose(tangent, 0., atol=3e-10, rtol=0)


@pytest.mark.parametrize('gap', (0., 7.))
def test_complex_material_jvp_is_holomorphic_and_additive(gap):
    obj = cell(order=(1, 1))
    epsilon = np.array([12.25+.2j, -10+1j]); de = np.array([.3+.2j, -.4+.1j]); db = .2+.3j
    settings = dict(GEOMETRY, gap=gap, epsilon=epsilon, background=1+.1j)
    _, real, _ = obj.closure(**settings, gap_tangent=0., epsilon_tangent=de, background_tangent=db)
    _, imaginary, _ = obj.closure(**settings, gap_tangent=0., epsilon_tangent=1j*de, background_tangent=1j*db)
    np.testing.assert_allclose(imaginary, 1j*np.asarray(real), atol=3e-9, rtol=0)
    _, dg, _ = obj.closure(**settings)
    _, combined, _ = obj.closure(**settings, gap_tangent=.7, epsilon_tangent=de, background_tangent=db)
    np.testing.assert_allclose(combined, .7*np.asarray(dg)+real, atol=3e-9, rtol=0)
    for h in (.002, .001):
        values = [obj.closure(**dict(settings, epsilon=epsilon+i*h*de, background=1+.1j+i*h*db), gap_tangent=0.)[0] for i in (-2,-1,1,2)]
        reference_jvp = (np.asarray(values[0])-8*np.asarray(values[1])+8*np.asarray(values[2])-np.asarray(values[3]))/(12*h)
        np.testing.assert_allclose(real, reference_jvp, atol=3e-9, rtol=0)


@pytest.mark.parametrize('gap', (0., 7.))
def test_units_translation_and_quadrature_preserve_the_tensor(gap):
    obj = cell(order=(1, 1))
    settings = dict(GEOMETRY, gap=gap, epsilon=(12.25, -10+1j))
    value, tangent, _ = obj.closure(**settings)
    scale = .001
    scaled = {k: scale*np.asarray(v) for k,v in GEOMETRY.items()}
    other, change, _ = obj.closure(**scaled, gap=scale*gap, epsilon=settings['epsilon'])
    np.testing.assert_allclose(other, value, atol=3e-10, rtol=0)
    np.testing.assert_allclose(scale*np.asarray(change), tangent, atol=3e-9, rtol=0)
    other, change, _ = cell(order=(1, 1), quadrature=32).closure(**settings)
    np.testing.assert_allclose(other, value, atol=3e-10, rtol=0)
    np.testing.assert_allclose(change, tangent, atol=3e-9, rtol=0)
    delta = np.array([31., -17.])
    other, change, _ = obj.closure(**dict(settings, offset=np.asarray(GEOMETRY['offset'])+delta))
    modes = np.array([(m,n) for n in (-1,0,1) for m in (-1,0,1)])
    phase = np.exp(-2j*np.pi*modes@(delta/np.diag(GEOMETRY['basis'])))
    expected = np.asarray(value)*phase[None,None,:,None]*phase.conj()[None,None,None,:]
    expected_d = np.asarray(tangent)*phase[None,None,:,None]*phase.conj()[None,None,None,:]
    np.testing.assert_allclose(other, expected, atol=3e-10, rtol=0)
    np.testing.assert_allclose(change, expected_d, atol=3e-9, rtol=0)


@pytest.mark.parametrize('update', ({'gap': -1.}, {'basis': ((700., 65.), (0., 560.))},
                                    {'epsilon': (12.25, -10.)}, {'epsilon_tangent': (np.nan, 0.)}))
def test_invalid_or_unqualified_inputs_are_refused(update):
    with pytest.raises(ValueError):
        cell().closure(**(dict(GEOMETRY, gap=7., epsilon=(12.25, -10+1j)) | update))


@pytest.mark.parametrize('order', ((0, 2), (2, 0)))
def test_directional_orders_and_refined_subpixel_gap_keep_original_guards(order):
    obj = cell(order=order, refinement=1, quadrature=32)
    closed, contact_jvp, _ = obj.closure(gap=0., epsilon=(12.25, -10+1j), **GEOMETRY)
    value, derivative, metadata = obj.closure(gap=1e-9, epsilon=(12.25, -10+1j), **GEOMETRY)
    assert value[0].shape[-1] == (2*order[0]+1)*(2*order[1]+1)
    assert np.isfinite(derivative).all()
    assert max(v for k,v in metadata.items() if k.endswith('residual')) < 1e-10
    np.testing.assert_allclose(value, np.asarray(closed)+1e-9*np.asarray(contact_jvp), atol=3e-10, rtol=0)


@pytest.mark.parametrize('epsilon', ((4., 9.), (12.25, -10+1j)))
def test_contact_eliminations_match_with_simultaneous_physical_forcing(epsilon):
    from wavebit._core.conforming.cartesian_pair_seam_galerkin import _contact_blocks
    from wavebit._core.conforming.real_trace_coupling import coupled_contact
    obj = cell()
    settings = dict(GEOMETRY, gap=0., epsilon=epsilon, background=1., gap_tangent=.7,
                    epsilon_tangent=(.3+.2j, -.4+.1j), background_tangent=.2+.3j)
    blocks = _contact_blocks(obj._base, settings, 32)
    values = [coupled_contact(obj._base, *blocks, elimination=form) for form in ('potential', 'retained')]
    np.testing.assert_allclose(values[0][0], values[1][0], atol=3e-10, rtol=0)
    np.testing.assert_allclose(values[0][1], values[1][1], atol=3e-9, rtol=0)
    for _, _, meta in values:
        assert max(v for k,v in meta.items() if k.endswith('residual') and v is not None) < 1e-10


def test_order_eight_finest_contact_preserves_the_original_energy_guard():
    geometry = dict(basis=np.diag([700.,700.]), widths=(300.,300.), height=300.)
    plan = CartesianPairOpeningPlan.from_geometry(**geometry, refinement=3)
    obj = CartesianPairSeamGalerkinGapCell(plan, 8, quadrature=48, coercivity_angle=1.53)
    value, derivative, meta = obj.closure(gap=0., epsilon=(12.25,-10+1j), **geometry)
    assert meta['closed_elimination'] == 'retained'
    assert max(v for k,v in meta.items() if k.endswith('residual') and v is not None) < 1e-10
    assert meta['derivative_reciprocity'] < 1e-9
    assert np.isfinite(value).all() and np.isfinite(derivative).all()


@pytest.mark.parametrize('gap', (7., 1e-9))
@pytest.mark.parametrize('epsilon', ((4., 9.), (12.25, -10+1j)))
def test_positive_eliminations_preserve_all_physical_variations(gap, epsilon):
    from wavebit._core.conforming.cartesian_pair_seam_galerkin import _quadrature, _free_fourier_energy
    from wavebit._core.conforming.real_trace_coupling import coupled_positive
    obj = cell(order=(1, 1), refinement=1, quadrature=32)
    settings = dict(GEOMETRY, gap=gap, epsilon=epsilon, background=1., gap_tangent=.7,
                    epsilon_tangent=(.3+.2j, -.4+.1j), background_tangent=.2+.3j)
    base = obj._base._assemble(**settings)
    quad = lambda split=True: _quadrature(obj._base, base, settings, 32, split)
    values = [coupled_positive(obj._base, base, quad, _free_fourier_energy(obj.order, settings),
                              obj.order[1], elimination=form) for form in ('potential', 'retained')]
    np.testing.assert_allclose(values[0][0], values[1][0], atol=3e-10, rtol=0)
    np.testing.assert_allclose(values[0][1], values[1][1], atol=3e-9, rtol=0)
    for _, _, meta in values:
        assert max(v for k,v in meta.items() if k.endswith('residual')) < 1e-10
    _, _, default = obj.closure(**settings)
    assert default['elimination'] == 'retained'


@pytest.mark.parametrize('gap', (.001, 1e-9))
def test_refined_sweep_preserves_each_rational_triangle_area_and_variation(gap):
    from wavebit._core.conforming.cartesian_pair_seam_galerkin import _quadrature
    geometry = dict(basis=np.diag([700., 700.]), widths=(300., 300.), height=300., offset=(0., 0.))
    obj = cell(order=(0, 0), refinement=2, quadrature=16)._base
    settings = dict(geometry, gap=gap, epsilon=(12.25, -10+1j), background=1., gap_tangent=1.,
                    epsilon_tangent=(0., 0.), background_tangent=0.)
    base = obj._assemble(**settings)
    length = np.sqrt(np.linalg.det(geometry['basis']))
    graph = _oracle_geometry(obj.data, {k: np.asarray(v)/length for k,v in geometry.items()})
    p0, velocity = np.asarray(graph['vertices0'], float), np.asarray(graph['vertices1'], float)
    triangles = graph['triangles']
    key = lambda ids: tuple(int(i) for i in ids)
    index = {key(ids): i for i,ids in enumerate(graph['periodic'][triangles])}
    assert len(index) == len(triangles)
    areas, changes = np.zeros(len(triangles)), np.zeros(len(triangles))
    for row in _quadrature(obj, base, settings, 16):
        i = index[key(row[0])]
        areas[i] += row[3].sum()
        changes[i] += row[4].sum()
    for i, triangle in enumerate(triangles):
        p, v = p0[triangle[:3]], velocity[triangle[:3]]
        a, b = p[1]-p[0], p[2]-p[0]
        da, db = v[1]-v[0], v[2]-v[0]
        cross = lambda x,y: x[0]*y[1]-x[1]*y[0]
        area0 = 0. if graph['thin'][i] else cross(a,b)/2
        change = (cross(da,b)+cross(a,db))/(2*length)
        area = area0+gap*change
        np.testing.assert_allclose(areas[i], area, atol=0., rtol=3e-12)
        np.testing.assert_allclose(changes[i], change, atol=3e-14, rtol=0)
