"""Layer-local tiling frames, rectangle representations, and frozen axis replay."""
import numpy as np
from _conv_frame import about_centre
import pytest

import wavebit
from wavebit._core.parametric import assemble_eps_coeffs, conv_matrix_2d
from test_conforming_nvf import SETTINGS, _objective, solve
from test_conforming_rectangular_lattice import scene as horizontal_scene


PARAMETERS = np.array([350., 29., 31., 4., 9., 1.7, 700., 430., .31, 17., -21.])


def vertical_scene(p=PARAMETERS, xp=np):
    span, gap, seam, e1, e2, background, px, py, angle, cx, cy = p
    rotation = xp.stack([xp.stack([xp.cos(angle), -xp.sin(angle)]),
                         xp.stack([xp.sin(angle), xp.cos(angle)])])
    shapes = []
    for lower, upper, epsilon in ((-span/2, seam, e1), (seam, span/2, e2)):
        center = rotation@xp.stack([cx, cy+(lower+upper)/2])
        shapes.append(dict(lx=px-gap, ly=upper-lower, cx=center[0], cy=center[1],
                           angle=angle, material=epsilon))
    return [dict(background=background, shapes=shapes[::-1])], rotation*xp.stack([px, py])[None]


@pytest.mark.research
@pytest.mark.parametrize('order', [(2, 1), (1, 2), (1, 0), (0, 1), (0, 0)])
def test_vertical_plan_and_material_match_the_original_lattice_fourier_layout(order):
    structure, lattice = vertical_scene()
    builder = wavebit.ConformingNVF(order, **SETTINGS)
    plan = builder.plan(structure, lattice)
    assert plan.axes == (1,)
    assert plan.layers == ((1, 0),)
    actual = builder.conv_matrices(structure, lattice, plan=plan)
    layer = structure[0]
    eps, _ = assemble_eps_coeffs(np, layer['shapes'], layer['background'],
                                 *order, lattice, np.complex128)
    np.testing.assert_allclose(np.linalg.inv(actual.epz_i[0]), about_centre(conv_matrix_2d(np, eps, *order), order),
                               atol=2e-12, rtol=0)


@pytest.mark.research
@pytest.mark.parametrize('turns', [(1, 0), (0, -1), (2, 3)])
@pytest.mark.parametrize('factory', [horizontal_scene, vertical_scene])
def test_equivalent_quarter_turn_rectangle_spellings_preserve_every_matrix(factory, turns):
    structure, lattice = factory()
    builder = wavebit.ConformingNVF((2, 1), **SETTINGS)
    plan = builder.plan(structure, lattice)
    reference = builder.conv_matrices(structure, lattice, plan=plan)
    for shape, turn in zip(structure[0]['shapes'], turns):
        shape['angle'] += turn*np.pi/2
        if turn % 2:
            shape['lx'], shape['ly'] = shape['ly'], shape['lx']
    assert builder.plan(structure, lattice) == plan
    actual = builder.conv_matrices(structure, lattice, plan=plan)
    for got, want in zip(actual, reference):
        np.testing.assert_allclose(got, want, atol=3e-12, rtol=0)


@pytest.mark.parametrize('angle', [np.pi/2, np.pi])
@pytest.mark.parametrize('lengths', [(0., 350.), (350., 0.), (-1e-30, 350.)])
def test_rotation_roundoff_cannot_turn_nonpositive_lengths_into_valid_geometry(angle, lengths):
    builder = wavebit.ConformingNVF(0, **SETTINGS)
    shape = dict(lx=lengths[0], ly=lengths[1], angle=angle, material=4.)
    with pytest.raises(ValueError, match='lengths must be positive'):
        builder.plan([dict(shapes=[shape])], np.diag([700., 430.]))


def test_original_lengths_remain_guarded_in_quarter_turn_traced_replay():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    builder = wavebit.ConformingNVF(0, **SETTINGS)
    lattice = np.diag([700., 430.])
    def structure(length):
        return [dict(shapes=[dict(lx=length, ly=700., angle=np.pi/2, material=4.)])]
    plan = builder.plan(structure(350.), lattice)
    assert plan.axes == (1,)
    build = jax.jit(lambda length: builder.conv_matrices(
        structure(length), jnp.asarray(lattice), plan=plan, xp=jnp))
    assert all(np.isfinite(np.asarray(slot)).all() for slot in build(350.))
    for length in (0., -1e-30):
        assert all(np.isnan(np.asarray(slot)).all() for slot in build(length))
    jax.clear_caches()


def _contact_reference(layer, lattice, order, axis):
    # Independent exact scalar Fourier integrals, followed by the closed
    # one-dimensional inverse rule in the physical material-normal direction.
    eps, reciprocal = assemble_eps_coeffs(np, layer['shapes'], layer['background'],
                                          *order, lattice, np.complex128)
    tangent = about_centre(conv_matrix_2d(np, eps, *order), order)
    normal = np.linalg.inv(about_centre(conv_matrix_2d(np, reciprocal, *order), order))
    rotation = lattice/np.linalg.norm(lattice, axis=0)
    x, y = (normal, tangent) if axis == 0 else (tangent, normal)
    c, s = rotation[0, 0], rotation[1, 0]
    return (c*c*x+s*s*y, s*s*x+c*c*y, np.linalg.inv(tangent), c*s*(y-x), c*s*(y-x))


@pytest.mark.research
@pytest.mark.parametrize('backend', [0, 1, 2])
def test_mixed_axis_contact_stack_matches_independent_analytic_matrices_and_engine(backend):
    if backend == 1:
        pytest.importorskip('jax')
        wavebit.enable_x64()
    elif backend == 2:
        pytest.importorskip('torch')
    p = PARAMETERS.copy()
    p[1] = 0.
    horizontal, lattice = horizontal_scene(p)
    vertical, _ = vertical_scene(p)
    structure = horizontal+[dict(background=2.3)]+vertical
    order = (2, 1)
    builder = wavebit.ConformingNVF(order, **SETTINGS)
    plan = builder.plan(structure, lattice)
    assert plan.axes == (0, 0, 1)
    actual = builder.conv_matrices(structure, lattice, plan=plan)
    identity = np.eye(15, dtype=complex)
    references = [_contact_reference(horizontal[0], lattice, order, 0),
                  (2.3*identity, 2.3*identity, identity/2.3, 0*identity, 0*identity),
                  _contact_reference(vertical[0], lattice, order, 1)]
    expected = wavebit.ConvMatrices(*(np.stack([layer[i] for layer in references]) for i in range(5)))
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want, atol=4e-12, rtol=0)
    np.testing.assert_allclose(solve(actual, lattice, order, scale=.65, backend=backend),
                               solve(expected, lattice, order, scale=.65), atol=2e-10, rtol=0)


@pytest.mark.research
@pytest.mark.parametrize('order', [(2, 1), (1, 0), (0, 1)])
def test_second_axis_cell_matches_the_same_physical_scene_in_a_respelt_lattice(order):
    structure, lattice = vertical_scene()
    builder = wavebit.ConformingNVF(order, **SETTINGS)
    actual = solve(builder.conv_matrices(structure, lattice), lattice, order, scale=.65)
    # (a2, -a1) describes the same periodic cell; its center changes by -a1,
    # a lattice translation. Its order labels are (m_new,n_new)=(n_old,-m_old).
    respelt = lattice@np.array([[0., -1.], [1., 0.]])
    reference_builder = wavebit.ConformingNVF(order[::-1], **SETTINGS)
    assert reference_builder.plan(structure, respelt).axes == (0,)
    reference = solve(reference_builder.conv_matrices(structure, respelt), respelt,
                       order[::-1], scale=.65)
    reference = np.rot90(reference, k=-1, axes=(-2, -1))
    np.testing.assert_allclose(actual, reference, atol=2e-10, rtol=0)


@pytest.mark.research
def test_vertical_geometry_material_periods_and_pose_remain_live_under_jit():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    builder = wavebit.ConformingNVF(1, **SETTINGS)
    plan = builder.plan(*vertical_scene())
    def objective(p, xp):
        return _objective(builder.conv_matrices(*vertical_scene(p, xp), plan=plan, xp=xp), xp)
    value, gradient = jax.jit(jax.value_and_grad(lambda p: objective(p, jnp)))(jnp.asarray(PARAMETERS))
    assert value == pytest.approx(objective(PARAMETERS, np), abs=3e-10)
    for i, step in enumerate([1e-3]*3+[1e-4]*3+[1e-3]*2+[1e-5]+[1e-3]*2):
        for scale in (1., .5):
            delta = np.eye(11)[i]*step*scale
            secant = (objective(PARAMETERS+delta, np)-objective(PARAMETERS-delta, np))/(2*step*scale)
            assert gradient[i] == pytest.approx(secant, abs=3e-7, rel=5e-6)
    jax.clear_caches()


@pytest.mark.research
@pytest.mark.parametrize('roundoff', [0., -np.spacing(700.)])
def test_first_period_contact_has_a_full_one_sided_derivative(roundoff):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    p = PARAMETERS.copy()
    p[1] = roundoff
    builder = wavebit.ConformingNVF(1, **SETTINGS)
    plan = builder.plan(*vertical_scene(p))
    assert plan.axes == (1,)
    def objective(gap, xp):
        live = xp.asarray(p)+xp.asarray(np.eye(11)[1])*gap
        return _objective(builder.conv_matrices(*vertical_scene(live, xp), plan=plan, xp=xp), xp)
    value, gradient = jax.jit(jax.value_and_grad(lambda gap: objective(gap, jnp)))(0.)
    baseline = objective(0., np)
    assert value == pytest.approx(baseline, abs=3e-10)
    for h in (1e-6, 1e-7):
        secant = 2*(objective(h/2, np)-baseline)/(h/2)-(objective(h, np)-baseline)/h
        assert gradient == pytest.approx(secant, abs=3e-6)
    assert abs(gradient) > 1e-3
    jax.clear_caches()


def test_replay_keeps_its_axis_and_refuses_an_incompatible_new_tiling():
    builder = wavebit.ConformingNVF(0, **SETTINGS)
    vertical, lattice = vertical_scene()
    plan = builder.plan(vertical, lattice)
    horizontal, _ = horizontal_scene()
    assert builder.plan(horizontal, lattice).axes == (0,)
    with pytest.raises(wavebit.StaleGeometryError):
        builder.conv_matrices(horizontal, lattice, plan=plan)
    for axes in ((), (2,), ('v',)):
        with pytest.raises(wavebit.StaleGeometryError):
            builder.conv_matrices(vertical, lattice, plan=plan._replace(axes=axes))
    # A single full-x stripe needs the second-axis frame. If its width shrinks,
    # both frames become valid, but replay retains the original approximation.
    stripe = [dict(shapes=[dict(lx=700., ly=130., material=4.)])]
    basis = np.diag([700., 430.])
    frozen = builder.plan(stripe, basis)
    assert frozen.axes == (1,)
    stripe[0]['shapes'][0]['lx'] = 630.
    assert builder.plan(stripe, basis).axes == (0,)
    assert all(np.isfinite(slot).all() for slot in builder.conv_matrices(stripe, basis, plan=frozen))


def test_invalid_vertical_traced_replay_poisoning_covers_all_slots():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    builder = wavebit.ConformingNVF(0, **SETTINGS)
    plan = builder.plan(*vertical_scene())
    build = jax.jit(lambda p: builder.conv_matrices(*vertical_scene(p, jnp), plan=plan, xp=jnp))
    for index, value in ((0, 500.), (1, -1.), (2, 200.), (3, -2.)):
        p = PARAMETERS.copy()
        p[index] = value
        assert all(np.isnan(np.asarray(slot)).all() for slot in build(jnp.asarray(p)))
    jax.clear_caches()
