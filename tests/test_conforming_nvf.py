"""Public shape construction, physical frame covariance, and live JAX replay."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from _conv_frame import about_centre, slots_about_centre
import pytest

import wavebit
from wavebit._core.parametric import assemble_eps_coeffs, conv_matrix_2d


SETTINGS = dict(rows=2, body_degree=2, gap_degree=2, grade_y=False)
PARAMETERS = np.array([350., 17., 31., 4., 9., 1.7, 700., .31, 17., -21.])


def scene(p=PARAMETERS, xp=np):
    width, gap, seam, e1, e2, background, period, angle, cx, cy = p
    rotation = xp.stack([xp.stack([xp.cos(angle), -xp.sin(angle)]),
                         xp.stack([xp.sin(angle), xp.cos(angle)])])
    shapes = []
    for left, right, epsilon in ((-width/2, seam, e1), (seam, width/2, e2)):
        center = rotation@xp.stack([cx+(left+right)/2, cy])
        shapes.append(dict(type='rectangle', lx=right-left, ly=period-gap,
                           cx=center[0], cy=center[1], angle=angle, material=epsilon))
    # Deliberately reversed: public callers do not owe the cell a sorted list.
    return [dict(background=background, shapes=shapes[::-1])], period*rotation


def solve(conv, lattice, order, *, scale=1., angle=.31, backend='numpy', kernel_frame=False):
    solver = wavebit.rcwa(backend=backend, lattice=lattice, wavelength=900.*scale,
                          theta=.23, phi=.47+angle, thickness=[173.*scale]*len(conv.epx),
                          truncation_order=order, top=1., bot=2.25)
    # Public builders and private study models deliberately use different doors.
    operation = solver._solve_conv if kernel_frame else solver.solve_conv
    answer = operation(*(solver.xp.asarray(a, dtype=solver.xp.complex128) for a in conv))
    def read(a):
        return a.detach().cpu().numpy() if hasattr(a, 'detach') else np.asarray(a)
    return np.array([[read(getattr(getattr(answer, pol), key)) for key in ('T', 'R')]
                     for pol in ('p_inc', 's_inc')])


def test_public_shapes_preserve_the_existing_cell_and_sort_only_at_plan_time():
    from studies.nvf_polynomial_body import PolynomialBodyGapModel
    p = PARAMETERS.copy()
    p[5], p[7], p[8], p[9] = 1., 0., 0., 0.
    structure, lattice = scene(p)
    builder = wavebit.ConformingNVF(1, **SETTINGS)
    plan = builder.plan(structure, lattice)
    assert plan.layers == ((1, 0),)
    actual = builder.conv_matrices(structure, lattice, plan=plan)
    expected = PolynomialBodyGapModel(1, flux_basis='local', **SETTINGS).conv_matrices(
        p[1], dict(seams=[p[2]], values=p[3:5]), p[0])
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want, atol=2e-12, rtol=0)


@pytest.mark.research
@pytest.mark.parametrize('order', [(2, 1), (0, 1), (0, 0)])
def test_longitudinal_material_matches_independent_analytic_shape_integration(order):
    p = PARAMETERS.copy()
    p[:3] *= .7
    p[6] *= .7
    p[8:] = [421., -530.]
    structure, lattice = scene(p)
    conv = wavebit.ConformingNVF(order, **SETTINGS).conv_matrices(structure, lattice)
    layer = structure[0]
    c, _ = assemble_eps_coeffs(np, layer['shapes'], layer['background'], *order, lattice, np.complex128)
    expected = about_centre(conv_matrix_2d(np, c, *order), order)
    np.testing.assert_allclose(np.linalg.inv(conv.epz_i[0]), expected, atol=1e-12, rtol=0)
    assert conv.epx.dtype == np.complex128
    count = (2*order[0]+1)*(2*order[1]+1)
    assert conv.epx.shape == (1, count, count)


@pytest.mark.research
@pytest.mark.parametrize('order', [(2, 1), (1, 0), (0, 1), (0, 0)])
def test_rigid_rotation_and_translation_preserve_every_engine_diffraction_power(order):
    builder = wavebit.ConformingNVF(order, **SETTINGS)
    structure, lattice = scene()
    baseline = solve(builder.conv_matrices(structure, lattice), lattice, order)
    p = PARAMETERS.copy()
    p[7] += .83
    p[8:] += [781., -451.]
    structure, lattice = scene(p)
    actual = solve(builder.conv_matrices(structure, lattice), lattice, order, angle=p[7])
    np.testing.assert_allclose(actual, baseline, atol=2e-10, rtol=0)
    np.testing.assert_allclose(actual.sum(axis=(-1, -2)).sum(axis=1), 1., atol=2e-10, rtol=0)


@pytest.mark.research
def test_uniform_physical_scaling_and_uniform_layers_reach_the_engine():
    builder = wavebit.ConformingNVF(1, **SETTINGS)
    structure, lattice = scene()
    structure.insert(0, dict(background=2.3))
    baseline = builder.conv_matrices(structure, lattice)
    p = PARAMETERS.copy()
    p[[0, 1, 2, 6, 8, 9]] *= .003
    structure, lattice_scaled = scene(p)
    structure.insert(0, dict(background=wavebit.Material(epsilon=2.3), shapes=[]))
    actual = builder.conv_matrices(structure, lattice_scaled)
    for got, want in zip(actual, baseline):
        np.testing.assert_allclose(got, want, atol=2e-12, rtol=0)
    np.testing.assert_allclose(solve(actual, lattice_scaled, (1, 1), scale=.003),
                               solve(baseline, lattice, (1, 1)), atol=2e-10, rtol=0)
    count = 9
    for got, expected in zip(actual, (2.3, 2.3, 1/2.3, 0., 0.)):
        np.testing.assert_allclose(got[0], expected*np.eye(count), atol=1e-15, rtol=0)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_public_export_runs_in_jax_and_torch_engines(backend):
    if backend == 'jax':
        pytest.importorskip('jax')
        wavebit.enable_x64()
    else:
        pytest.importorskip('torch')
    structure, lattice = scene()
    conv = wavebit.ConformingNVF(1, **SETTINGS).conv_matrices(structure, lattice)
    np.testing.assert_allclose(solve(conv, lattice, (1, 1), backend=backend),
                               solve(conv, lattice, (1, 1)), atol=2e-10, rtol=0)


@pytest.mark.research
def test_equal_material_subdivision_does_not_add_an_interface_constraint():
    p = PARAMETERS.copy()
    p[4] = p[3]
    structure, lattice = scene(p)
    builder = wavebit.ConformingNVF(1, **SETTINGS)
    actual = builder.conv_matrices(structure, lattice)
    rectangle = dict(structure[0]['shapes'][0], lx=p[0])
    rectangle['cx'], rectangle['cy'] = lattice@p[8:]/p[6]
    structure[0]['shapes'] = [rectangle]
    expected = builder.conv_matrices(structure, lattice)
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want, atol=4e-12, rtol=0)


def _objective(conv, xp):
    # Keep the established random-entry functional in the kernels' frame.
    # Changing its weights would change the secant error the tolerances measure.
    conv = slots_about_centre(conv)
    slots = xp.stack(conv)
    weights = xp.asarray(np.random.default_rng(518).normal(size=slots.shape))
    return xp.sum(weights*(slots.real+.37*slots.imag))


@pytest.mark.research
def test_frozen_plan_keeps_geometry_material_background_period_and_pose_live_under_jit():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    builder = wavebit.ConformingNVF(1, **SETTINGS)
    plan = builder.plan(*scene())
    def objective(p, xp):
        return _objective(builder.conv_matrices(*scene(p, xp), plan=plan, xp=xp), xp)
    value, derivative = jax.jit(jax.value_and_grad(lambda p: objective(p, jnp)))(jnp.asarray(PARAMETERS))
    assert value == pytest.approx(objective(PARAMETERS, np), abs=3e-10)
    for i, step in enumerate([1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4, 1e-3, 1e-5, 1e-3, 1e-3]):
        for scale in (1., .5):
            delta = np.eye(10)[i]*step*scale
            expected = (objective(PARAMETERS+delta, np)-objective(PARAMETERS-delta, np))/(2*step*scale)
            np.testing.assert_allclose(derivative[i], expected, atol=2e-7, rtol=4e-6)
    jax.clear_caches()


@pytest.mark.research
def test_contact_has_a_full_one_sided_derivative_and_does_not_snap_positive_gaps():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    builder = wavebit.ConformingNVF(1, **SETTINGS)
    p = PARAMETERS.copy()
    p[1], p[7] = 0., 0.
    plan = builder.plan(*scene(p))
    def objective(gap, xp):
        live = xp.asarray(p)+xp.asarray(np.eye(10)[1])*gap
        return _objective(builder.conv_matrices(*scene(live, xp), plan=plan, xp=xp), xp)
    value, gradient = jax.jit(jax.value_and_grad(lambda gap: objective(gap, jnp)))(0.)
    reference = objective(0., np)
    assert value == pytest.approx(reference, abs=3e-10)
    # The graded contact cell has a short gap scale. A step scan resolves
    # the derivative at 1e-6 and 1e-7; 1e-4 is still outside that window.
    for step in (1e-6, 1e-7):
        expected = (2*(objective(step/2, np)-reference)/(step/2)
                    -(objective(step, np)-reference)/step)
        assert gradient == pytest.approx(expected, abs=3e-6)
    assert abs(gradient) > 1e-3
    tiny = (objective(1e-8, np)-reference)/1e-8
    assert tiny == pytest.approx(float(gradient), abs=1e-3)
    jax.clear_caches()


@pytest.mark.parametrize('order', [-1, True, 1.5, (1,), (1, -1), (1, 2.5)])
def test_invalid_order_is_refused(order):
    with pytest.raises(ValueError, match='truncation_order'):
        wavebit.ConformingNVF(order)


def test_numpy_integer_order_and_lattice_object_are_accepted():
    builder = wavebit.ConformingNVF(np.int64(0), **SETTINGS)
    basis = np.eye(2)*2
    lattice = wavebit.Lattice(u=basis[:, 0], v=basis[:, 1], axis_aligned=True)
    assert builder.conv_matrices([dict(background=3.)], lattice).epx.item() == 3.
    assert builder.conv_matrices([dict(background=3.)], lattice,
                                 xp=wavebit.rcwa().xp).epx.item() == 3.


def test_jit_requires_a_concrete_plan_and_x64_precision():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    builder = wavebit.ConformingNVF(0, **SETTINGS)
    with pytest.raises(ValueError, match='concrete geometry before JIT'):
        jax.jit(lambda p: builder.conv_matrices(*scene(p, jnp), xp=jnp))(jnp.asarray(PARAMETERS))
    plan = builder.plan(*scene())
    jax.config.update('jax_enable_x64', False)
    try:
        with pytest.raises(ValueError, match='64-bit'):
            builder.conv_matrices(*scene(), plan=plan, xp=jnp)
    finally:
        jax.config.update('jax_enable_x64', True)


@pytest.mark.research
def test_full_height_contact_matches_independent_rotated_cartesian_inverse_rule():
    p = PARAMETERS.copy()
    p[1] = 0.
    structure, lattice = scene(p)
    order = (2, 1)
    actual = wavebit.ConformingNVF(order, **SETTINGS).conv_matrices(structure, lattice)
    layer = structure[0]
    eps, inverse = assemble_eps_coeffs(np, layer['shapes'], layer['background'],
                                       *order, lattice, np.complex128)
    tangent = about_centre(conv_matrix_2d(np, eps, *order), order)
    normal = np.linalg.inv(about_centre(conv_matrix_2d(np, inverse, *order), order))
    c, s = np.cos(p[7]), np.sin(p[7])
    expected = (c*c*normal+s*s*tangent, s*s*normal+c*c*tangent,
                np.linalg.inv(tangent), c*s*(tangent-normal), c*s*(tangent-normal))
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got[0], want, atol=3e-12, rtol=0)


@pytest.mark.parametrize('lattice', [[[700., 1.], [0., 700.]],
                                    np.diag([700., -700.]), np.zeros((2, 2)),
                                    np.array([[np.inf, 0.], [0., 700.]]), [700., 700.]])
def test_unsupported_lattice_is_refused(lattice):
    with pytest.raises(ValueError, match='lattice'):
        wavebit.ConformingNVF(0, **SETTINGS).conv_matrices([dict(background=1.)], lattice)


@pytest.mark.parametrize('change', [dict(lx=-3.), dict(ly=701.), dict(ly=0.), dict(cx=900.),
                                    dict(cy=17.), dict(angle=.4), dict(material=-2.),
                                    dict(material=3.-.1j), dict(material=np.eye(3)),
                                    dict(material=wavebit.Material(epsilon=3., mu=1.)),
                                    dict(angel=.3)])
def test_unsupported_or_invalid_shape_is_refused(change):
    structure, lattice = scene()
    structure[0]['shapes'][0].update(change)
    with pytest.raises((ValueError, NotImplementedError)):
        wavebit.ConformingNVF(0, **SETTINGS).conv_matrices(structure, lattice)


@pytest.mark.parametrize('structure', [[], [None], [dict(shapes=None)], [dict(shapes=[None])],
                                      [dict(shapes=[dict(type='circle', lx=30., material=4.)])]])
def test_invalid_structure_or_unsupported_shape_kind_is_refused(structure):
    with pytest.raises((ValueError, NotImplementedError)):
        wavebit.ConformingNVF(0, **SETTINGS).conv_matrices(structure, np.eye(2)*700)


def test_stale_shape_count_order_tiling_and_layer_count_are_refused():
    builder = wavebit.ConformingNVF(0, **SETTINGS)
    plan = builder.plan(*scene())
    for change in ('count', 'order', 'overlap', 'layers'):
        structure, lattice = scene()
        if change == 'count':
            structure[0]['shapes'].pop()
        elif change == 'order':
            structure[0]['shapes'].reverse()
        elif change == 'overlap':
            structure[0]['shapes'][0]['lx'] += 1.
        else:
            structure.append(dict(background=1.))
        with pytest.raises(wavebit.StaleGeometryError):
            builder.conv_matrices(structure, lattice, plan=plan)


def test_invalid_traced_geometry_and_material_poison_all_slots_including_uniform_layers():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    builder = wavebit.ConformingNVF(0, **SETTINGS)
    plan = builder.plan(*scene())
    build = jax.jit(lambda p: builder.conv_matrices(*scene(p, jnp), plan=plan, xp=jnp))
    for index, value in ((0, -1.), (1, -1.), (2, 500.), (3, -2.), (5, 0.), (6, 0.), (7, np.nan)):
        p = PARAMETERS.copy()
        p[index] = value
        assert all(np.isnan(np.asarray(slot)).all() for slot in build(jnp.asarray(p)))
    uniform_plan = builder.plan([dict(background=1.)], np.eye(2)*700)
    uniform = jax.jit(lambda epsilon: builder.conv_matrices(
        [dict(background=epsilon)], jnp.eye(2)*700, plan=uniform_plan, xp=jnp))
    for value in (-1., 0., np.nan, 3.-.1j):
        assert all(np.isnan(np.asarray(slot)).all() for slot in uniform(value))
    jax.clear_caches()


def test_public_construction_does_not_need_a_studies_checkout(tmp_path):
    # The wheel contains wavebit*, not studies. Exercise precisely that boundary
    # in a fresh process so this test cannot reuse already-imported study code.
    shutil.copytree(Path(wavebit.__file__).parent, tmp_path/'wavebit',
                    ignore=shutil.ignore_patterns('__pycache__'))
    code = """
import sys
import numpy as np
import wavebit
conv = wavebit.ConformingNVF(0, rows=2, grade_y=False, gap_degree=2).conv_matrices(
    [dict(shapes=[dict(lx=1., ly=1.9, material=4.)])], np.eye(2)*2)
assert np.isfinite(conv.epx).all()
assert not any(name == 'studies' or name.startswith('studies.') for name in sys.modules)
"""
    env = dict(os.environ, PYTHONPATH=str(tmp_path))
    result = subprocess.run([sys.executable, '-c', code], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


@pytest.mark.research
@pytest.mark.parametrize('door', ['conforming', 'rebased', 'affine', 'rectangle', 'pair'])
def test_each_public_nvf_door_publishes_the_accessor_frame(door):
    """The longitudinal integral checks the caller's original order labels independently."""
    order = (1, 1)
    if door in ('conforming', 'rebased'):
        structure, lattice = scene()
        if door == 'rebased': lattice = lattice @ np.array([[1., 1.], [0., 1.]])
        builder = wavebit.ConformingNVF(order, **SETTINGS)
        conv = builder.conv_matrices(structure, lattice)
    elif door == 'affine':
        from test_conforming_affine_profile import layer
        profile = layer()
        lattice = np.diag([500., 700.])
        p = profile['profile']; edges = [-p['width']/2, *p['seams'], p['width']/2]
        shapes = [dict(type='rectangle', lx=500*(b-a), ly=700*p['height'],
                       cx=p['offset'][0]+500*(a+b)/2, cy=p['offset'][1], material=e)
                  for a, b, e in zip(edges, edges[1:], p['values'])]
        structure = [dict(background=profile['background'], shapes=shapes)]
        conv = wavebit.AffineProfileNVF(order, **SETTINGS).conv_matrices([profile], lattice)
    elif door == 'rectangle':
        from test_nvf_cartesian_rectangle_api import scene as rectangle_scene
        structure, lattice = rectangle_scene()
        builder = wavebit.CartesianRectangleNVF(order, refinement=1)
        conv = builder.conv_matrices(structure, lattice, plan=builder.plan(structure, lattice))
    else:
        from test_nvf_cartesian_rectangle_pair_api import scene as pair_scene, native_parameters, case_cell
        structure, lattice = pair_scene(native_parameters(case_cell('gap-r1')[1]))
        builder = wavebit.CartesianRectanglePairNVF(order, refinement=1)
        conv = builder.conv_matrices(structure, lattice, plan=builder.plan(structure, lattice))
    s = wavebit.rcwa(structure=structure, lattice=lattice, truncation_order=order,
                    thickness=[100.], factorization='laurent')
    np.testing.assert_allclose(np.linalg.inv(conv.epz_i[0]), s.eps_conv_matrices().epx[0],
                               atol=5e-12, rtol=0)
