"""Painted rectangle cells against independent Cartesian vertical-strip integrals."""
import numpy as np
import pytest
from scipy.integrate import quad_vec

import wavebit
from wavebit._core.scene_fdtd import _shape_partition, _painted_cell_materials, _require_scene
from wavebit.backends.time_domain import time_loop_backend
from test_fdtd_scene_angles import value_and_jacobian
from test_fdtd_circles import setup


PERIODS = [.6, .4]
PARAMETERS = np.array([-.08, -.02, .26, .18, .31, 2.1, 1.12,
                       .035, .015, .23, .16, -.27, 3.2, 1.31])


def recipe(p):
    return [dict(background=wavebit.Material(1., 1.), shapes=[
        dict(type='rectangle', cx=p[j], cy=p[j+1], lx=p[j+2], ly=p[j+3], angle=p[j+4],
             material=wavebit.Material(p[j+5], p[j+6])) for j in range(0, len(p), 7)])]


def cells(p, backend=0, smoothing=(0., 0.), offset=(0., 0.)):
    rt = time_loop_backend(backend)
    _, _, layers = _shape_partition(recipe(p), PERIODS, xp=rt.xp, array=rt.array,
                                    check=_require_scene, integration='area')
    base, shape = layers[0]
    return _painted_cell_materials(shape['edges'], base, (np.arange(6)+offset[0])*.1,
        (np.arange(4)+offset[1])*.1, .1, .1, PERIODS, xp=rt.xp, array=rt.array, smoothing=smoothing)


def cartesian_oracle(p, smoothing=(0., 0.), offset=(0., 0.)):
    """Independent vertical chords, last-painted y intervals and adaptive x quadrature."""
    polygons, owners = [], []
    for owner, j in enumerate(range(0, len(p), 7)):
        co, si = np.cos(p[j+4]), np.sin(p[j+4])
        local = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])*p[j+2:j+4]/2
        polygon = local@np.array([[co, si], [-si, co]])+np.mod(p[j:j+2]+np.array(PERIODS)/2, PERIODS)
        for a in (-.6, 0., .6):
            for b in (-.4, 0., .4):
                polygons.append(polygon+[a, b]); owners.append(owner)
    polygons, owners = np.array(polygons), np.array(owners)
    first = polygons.reshape(-1, 2); last = np.roll(polygons, -1, axis=1).reshape(-1, 2)
    edges = last-first
    cross = lambda a, b: a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]
    denominator = cross(edges[:, None], edges[None])
    safe = np.where(abs(denominator) > 1e-14, denominator, 1.)
    delta = first[None]-first[:, None]
    u, v = cross(delta, edges[None])/safe, cross(delta, edges[:, None])/safe
    valid = (abs(denominator) > 1e-14) & (u > 0) & (u < 1) & (v > 0) & (v < 1)
    global_cuts = np.r_[first[:, 0], (first[:, None, 0]+u*edges[:, None, 0])[valid]]
    material = np.vstack(([1., 1.], np.array(p).reshape(-1, 7)[:, 5:7]))
    def knots(c, h):
        return np.unique(c+np.array([-(.1+h)/2, -abs(.1-h)/2, abs(.1-h)/2, (.1+h)/2]))
    def weight(t, c, h):
        if h == 0:
            return (abs(t-c) <= .05).astype(float) if isinstance(t, np.ndarray) else float(abs(t-c) <= .05)
        return np.maximum(0., np.minimum(c+.05, t+h/2)-np.maximum(c-.05, t-h/2))/h
    result = np.zeros((2, 4, 6))
    for iy, ix in np.ndindex(4, 6):
        x, y = (ix+offset[0])*.1, (iy+offset[1])*.1
        xs, ys = knots(x, smoothing[0]), knots(y, smoothing[1])
        extra = []
        for face in ys:
            t = (face-first[:, 1])/np.where(abs(edges[:, 1]) > 1e-14, edges[:, 1], 1.)
            use = (abs(edges[:, 1]) > 1e-14) & (t > 0) & (t < 1)
            extra.extend((first[:, 0]+t*edges[:, 0])[use])
        cuts = np.unique(np.clip(np.r_[xs, global_cuts, extra], xs[0], xs[-1]))
        def integrand(at):
            t = (at-first[:, 0])/np.where(abs(edges[:, 0]) > 1e-14, edges[:, 0], 1.)
            hits = ((abs(edges[:, 0]) > 1e-14) & (t >= 0) & (t <= 1)).reshape(-1, 4)
            chord = (first[:, 1]+t*edges[:, 1]).reshape(-1, 4)
            low = np.min(np.where(hits, chord, np.inf), axis=1)
            high = np.max(np.where(hits, chord, -np.inf), axis=1)
            valid = (low < high) & (high > ys[0]) & (low < ys[-1])
            low, high, who = low[valid], high[valid], owners[valid]
            yc = np.unique(np.clip(np.r_[ys, low, high], ys[0], ys[-1]))
            mids = (yc[1:]+yc[:-1])/2
            paint = np.max(np.where((mids[None] > low[:, None]) & (mids[None] < high[:, None]),
                                    who[:, None]+1, 0), axis=0, initial=0)
            return weight(at, x, smoothing[0])*np.sum((np.diff(yc)*weight(mids, y, smoothing[1]))[:, None]*material[paint], axis=0)
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b-a > 8*np.finfo(float).eps:
                result[:, iy, ix] += quad_vec(integrand, a, b, epsabs=2e-14, epsrel=2e-12)[0]/.01
    return result


@pytest.mark.parametrize('backend', [0, 1, 2])
@pytest.mark.parametrize('smoothing', [(0., 0.), (.04, .06), (.6, .4)])
@pytest.mark.parametrize('case', ['overlap', 'periodic', 'reverse'])
def test_painted_rotated_cells_match_independent_cartesian_strips(backend, smoothing, case):
    setup(backend)
    p = PARAMETERS.copy()
    if case == 'periodic':
        p[[0, 7]] += .27; p[[1, 8]] += .19
    if case == 'reverse':
        p = p.reshape(2, 7)[::-1].copy().reshape(-1)
    value = np.asarray(cells(p, backend, smoothing, (.5, 0.)))
    np.testing.assert_allclose(value, cartesian_oracle(p, smoothing, (.5, 0.)), atol=5e-12, rtol=0.)
    assert value.min() >= 1.-3e-12
    assert value[0].max() <= 3.2+3e-12 and value[1].max() <= 1.31+3e-12
    moved = p.copy(); moved[[0, 7]] += 1.2; moved[[1, 8]] -= .8
    np.testing.assert_allclose(cells(moved, backend, smoothing, (.5, 0.)), value, atol=8e-12, rtol=0.)


@pytest.mark.parametrize('backend', [1, 2])
@pytest.mark.parametrize('smoothing', [(0., 0.), (.04, .06)])
def test_joint_fourteen_shape_and_material_derivatives(backend, smoothing):
    setup(backend)
    value, jac = value_and_jacobian(lambda p: cells(p, backend, smoothing).reshape(-1), PARAMETERS, backend)
    np.testing.assert_allclose(value, cartesian_oracle(PARAMETERS, smoothing).reshape(-1), atol=5e-12)
    for h in (2e-6, 1e-6):
        finite = np.stack([(cells(PARAMETERS+d, smoothing=smoothing)-cells(PARAMETERS-d, smoothing=smoothing))/(2*h)
                            for d in np.eye(14)*h], axis=-1).reshape(48, 14)
        np.testing.assert_allclose(jac, finite, atol=5e-7, rtol=5e-6)


@pytest.mark.parametrize('backend', [0, 1, 2])
@pytest.mark.parametrize('smoothing', [(0., 0.), (.04, .06), (.6, .4)])
def test_coincident_full_cell_and_hidden_shapes_have_correct_last_material(backend, smoothing):
    setup(backend)
    p = np.array([0., 0., .6, .4, 0., 2., 1.2, 0., 0., .6, .4, 0., 3., 1.4])
    value = np.asarray(cells(p, backend, smoothing))
    np.testing.assert_allclose(value, np.broadcast_to(np.array([3., 1.4])[:, None, None], value.shape), atol=8e-12)
    p[:7] = [.1, -.1, .2, .1, .2, 2.5, 1.1]
    np.testing.assert_allclose(cells(p, backend, smoothing), value, atol=8e-12)


def test_paint_order_matters_and_three_shapes_are_supported():
    p = np.r_[PARAMETERS, -.03, .02, .1, .12, .53, 1.6, 1.08]
    np.testing.assert_allclose(cells(p, smoothing=(.04, .06)), cartesian_oracle(p, (.04, .06)), atol=5e-12)
    reversed_p = p.reshape(3, 7)[::-1].copy().reshape(-1)
    np.testing.assert_allclose(cells(reversed_p), cartesian_oracle(reversed_p), atol=5e-12)
    # A known full-cell difference establishes paint priority without assuming
    # how large the overlap happens to be in the rotated example above.
    full = np.tile([0., 0., .6, .4, 0., 2., 1.1], (3, 1))
    full[:, 5:7] = [[2., 1.1], [3., 1.2], [4., 1.3]]
    difference = cells(full.reshape(-1))-cells(full[::-1].copy().reshape(-1))
    np.testing.assert_allclose(difference, np.broadcast_to(np.array([2., .2])[:, None, None], difference.shape), atol=8e-12)


def test_mixed_multiple_curves_are_refused_explicitly():
    rt = time_loop_backend(0)
    structure = recipe(PARAMETERS)
    structure[0]['shapes'][1]['type'] = 'ellipse'
    with pytest.raises(ValueError, match='Multiple area shapes currently require rectangles'):
        _shape_partition(structure, PERIODS, xp=rt.xp, array=rt.array, check=_require_scene, integration='area')


def response(p, backend=0, diagnostics=False, side='bottom'):
    stack = wavebit.Stack.from_structure([
        dict(background=1.2, shapes=[
            dict(lx=p[1], ly=.1, cx=-.035, cy=-.012, angle=p[0], material=2.),
            dict(lx=p[4], ly=.12, cx=p[2], cy=.014, angle=p[3], material=wavebit.Material(p[5], p[6]))]),
        dict(background=1.4)], [p[7], .3-p[7]], top=1., bot=1.4, period=[.4, .3])
    result, out = wavebit.fdtd(stack, wavebit.Incidence(wavelength=.9, theta=.17, phi=.23,
        polarization=(1., .4j), incidence_side=side), backend=backend, resolution=9,
        transverse_resolution=[10., 10.], n_pml=6, n_periods=35, pulse_width=1.5,
        truncation_order=0, mesh_thickness=[.15, .15], layer_smoothing=.025, align_layers=True,
        shape_integration='area', shape_smoothing=(.035, .025), diffraction_method='directional', return_diagnostics=True)
    xp = time_loop_backend(backend).xp
    value = xp.concat([xp.real(result.given.r_s).reshape(-1), xp.imag(result.given.t_p).reshape(-1)])
    return (value, out) if diagnostics else value


@pytest.mark.parametrize('backend', [1, 2])
def test_bottom_conical_complete_shape_material_and_layer_graph(backend):
    setup(backend)
    p = np.array([.31, .17, .025, -.27, .15, 2.8, 1.13, .14])
    value, jac = value_and_jacobian(lambda p: response(p, backend), p, backend)
    expected, mesh = response(p, diagnostics=True)
    np.testing.assert_allclose(value, expected, atol=5e-11)
    for h in (2e-5, 1e-5):
        finite = np.stack([(response(p+d)-response(p-d))/(2*h) for d in np.eye(8)*h], axis=-1)
        np.testing.assert_allclose(jac, finite, atol=7e-8, rtol=7e-5)
    assert np.min(np.max(abs(jac), axis=0)) > 1e-5
    _, moved = response(p+[.01, .01, .01, -.01, .01, .1, .01, .01], diagnostics=True)
    for key in ('z_nodes', 'z_spacing', 'dt', 'steps', 'source_z', 'pml_bounds', 'mesh_thickness'):
        np.testing.assert_array_equal(mesh[key], moved[key])


def test_one_trace_accepts_overlap_topology_changes_and_masks_invalid_geometry():
    jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    traces = []
    def f(p):
        traces.append(None)
        value, out = response(p, 1, diagnostics=True, side='top')
        return value, out['stable']
    run = jax.jit(f)
    p = np.array([.31, .17, .025, -.27, .15, 2.8, 1.13, .14])
    value, stable = run(p)
    assert stable
    for centre in (.19, .32, .425):
        moved = p.copy(); moved[2] = centre
        actual, valid = run(moved)
        assert valid
        np.testing.assert_allclose(actual, response(moved, side='top'), atol=5e-12)
    translated = p.copy(); translated[2] += .8
    actual, valid = run(translated)
    assert valid
    np.testing.assert_allclose(actual, value, atol=5e-12)
    for index, bad in [(1, -.1), (4, .6), (3, np.nan), (5, -1.)]:
        moved = p.copy(); moved[index] = bad
        actual, valid = run(moved)
        assert not valid and np.isnan(actual).all()
    assert len(traces) == 1


@pytest.mark.parametrize('backend', [1, 2])
@pytest.mark.parametrize('smoothing', [(1e-9, 1e-9), (.1, .1)])
def test_narrow_and_cell_width_kernels_keep_finite_shape_derivatives(backend, smoothing):
    setup(backend)
    value, jac = value_and_jacobian(lambda p: cells(p, backend, smoothing).reshape(-1), PARAMETERS, backend)
    target = cells(PARAMETERS, smoothing=(0., 0.)) if smoothing[0] < 1e-8 else cartesian_oracle(PARAMETERS, smoothing)
    np.testing.assert_allclose(value, target.reshape(-1), atol=5e-10)
    assert np.isfinite(jac).all()
    h = 1e-6
    finite = np.stack([(cells(PARAMETERS+d, smoothing=smoothing)-cells(PARAMETERS-d, smoothing=smoothing))/(2*h)
                        for d in np.eye(14)*h], axis=-1).reshape(48, 14)
    np.testing.assert_allclose(jac, finite, atol=5e-7, rtol=5e-6)


@pytest.mark.parametrize('backend', [1, 2])
def test_fully_hidden_shape_has_zero_geometry_and_material_sensitivity(backend):
    setup(backend)
    p = np.array([0., 0., .09, .07, .31, 2.1, 1.12, 0., 0., .4, .3, -.1, 3.2, 1.31])
    _, jac = value_and_jacobian(lambda p: cells(p, backend, (.04, .06)).reshape(-1), p, backend)
    np.testing.assert_allclose(jac[:, :7], 0., atol=3e-12)
    assert np.max(abs(jac[:, 12:])) > .1


@pytest.mark.parametrize('backend', [1, 2])
@pytest.mark.parametrize('case', ['aligned_disjoint', 'quarter_turn'])
def test_parallel_edges_and_equal_projections_keep_independent_shape_derivatives(backend, case):
    setup(backend)
    p = PARAMETERS.copy()
    if case == 'aligned_disjoint':
        p[:5] = [0., -.11, .22, .07, 0.]
        p[7:12] = [0., .11, .22, .07, 0.]
    else:
        p[4] = np.pi/2; p[11] = 0.
    value, jac = value_and_jacobian(lambda p: cells(p, backend, (.04, .06)).reshape(-1), p, backend)
    np.testing.assert_allclose(value, cartesian_oracle(p, (.04, .06)).reshape(-1), atol=5e-12)
    differences = []
    for h in ((2e-6, 1e-6, 5e-7) if case == 'quarter_turn' else (2e-6, 1e-6)):
        differences.append(np.stack([(cells(p+d, smoothing=(.04, .06))-cells(p-d, smoothing=(.04, .06)))/(2*h)
                                      for d in np.eye(14)*h], axis=-1).reshape(48, 14))
    # The quarter-turn edge lies exactly on a translation-kernel knot. Its
    # integral is C1 but has different one-sided second derivatives, so D(h)
    # has O(h), not O(h**2), bias. Two independent Richardson pairs remove it.
    estimates = ([2*b-a for a, b in zip(differences[:-1], differences[1:])]
                 if case == 'quarter_turn' else differences)
    for finite in estimates:
        np.testing.assert_allclose(jac, finite, atol=5e-7, rtol=5e-6)
    if case == 'quarter_turn':
        independent = []
        for h in (1e-6, 5e-7):
            independent.append(np.stack([(cartesian_oracle(p+d, (.04, .06))-cartesian_oracle(p-d, (.04, .06)))/(2*h)
                                         for d in np.eye(14)[[0, 3, 4]]*h], axis=-1).reshape(48, 3))
        np.testing.assert_allclose(jac[:, [0, 3, 4]], 2*independent[1]-independent[0], atol=5e-7, rtol=5e-6)


@pytest.mark.parametrize('backend', [1, 2])
def test_background_and_both_shape_material_pairs_retain_linear_sensitivities(backend):
    setup(backend)
    def sample(p, backend):
        rt = time_loop_backend(backend)
        structure = recipe(PARAMETERS)
        structure[0]['background'] = wavebit.Material(p[0], p[1])
        for j, shape in enumerate(structure[0]['shapes']):
            shape['material'] = wavebit.Material(p[2+2*j], p[3+2*j])
        _, _, layers = _shape_partition(structure, PERIODS, xp=rt.xp, array=rt.array,
                                        check=_require_scene, integration='area')
        base, shape = layers[0]
        return _painted_cell_materials(shape['edges'], base, np.arange(6)*.1, np.arange(4)*.1,
            .1, .1, PERIODS, xp=rt.xp, array=rt.array, smoothing=(.04, .06)).reshape(-1)
    p = np.array([1.4, 1.07, 2.1, 1.13, 3.2, 1.31])
    value, jac = value_and_jacobian(lambda p: sample(p, backend), p, backend)
    for h in (2e-6, 1e-6):
        finite = np.stack([(sample(p+d, 0)-sample(p-d, 0))/(2*h) for d in np.eye(6)*h], axis=-1)
        np.testing.assert_allclose(jac, finite, atol=2e-8, rtol=2e-7)
    # The two material channels are independent and each is a partition of
    # unity across background and the two visible shape regions.
    np.testing.assert_allclose(jac[:24, 1::2], 0., atol=2e-13)
    np.testing.assert_allclose(jac[24:, ::2], 0., atol=2e-13)
    np.testing.assert_allclose(jac[:24, ::2].sum(axis=1), 1., atol=2e-13)
    np.testing.assert_allclose(jac[24:, 1::2].sum(axis=1), 1., atol=2e-13)
    np.testing.assert_allclose(jac@p, value, atol=2e-13)


@pytest.mark.parametrize('backend', [1, 2])
@pytest.mark.parametrize('smoothing,height', [((.04, .06), .16), ((.1, .1), .2)])
def test_horizontal_kernel_knots_count_once_in_angle_sensitivity(backend, smoothing, height):
    setup(backend)
    p = np.array([-.08, 0., .26, height, 0., 2.1, 1.12, .23, .12, .05, .04, .27, 3.2, 1.31])
    def f(angle):
        values = list(p); values[4] = angle[0]
        return cells(values, backend, smoothing).reshape(-1)
    value, jac = value_and_jacobian(f, np.array([0.]), backend)
    np.testing.assert_allclose(value, cartesian_oracle(p, smoothing).reshape(-1), atol=5e-12)
    independent = []
    for h in (2e-6, 1e-6, 5e-7):
        direction = np.eye(14)[4]*h
        independent.append(((cartesian_oracle(p+direction, smoothing)-cartesian_oracle(p-direction, smoothing))/(2*h)).reshape(-1))
    for a, b in zip(independent[:-1], independent[1:]):
        np.testing.assert_allclose(jac[:, 0], 2*b-a, atol=5e-7, rtol=5e-6)
