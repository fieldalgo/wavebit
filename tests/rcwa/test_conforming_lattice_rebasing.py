"""Exact point-lattice rebasing, original Fourier sets and live physical data."""

from wavebit.compat import legacy as _legacy_wavebit
import numpy as np
from tests.fixtures.common._conv_frame import about_centre, slots_about_centre
import pytest

from wavebit.rcwa._engine.parametric import assemble_eps_coeffs, conv_matrix_2d
from tests.rcwa.test_conforming_complex_material import complex_scene, PARAMETERS, SETTINGS, _objective, solve


TRANSFORMS = [((1, 3), (0, 1)), ((1, 0), (-2, 1)), ((2, 1), (1, 1)), ((-1, 2), (0, -1))]


def rebased_scene(p=PARAMETERS, transform=TRANSFORMS[0], losses=(0., .8, 2.1), xp=np, axis=0,
                  preserve_positions=True):
    structure, orthogonal = complex_scene(p, losses, xp, axis)
    lattice = orthogonal@xp.asarray(transform)
    if preserve_positions:
        shift = xp.sum(orthogonal-lattice, axis=1)/2
        for shape in structure[0]['shapes']:
            shape['cx'] += shift[0]
            shape['cy'] += shift[1]
    return structure, lattice


def contact_reference(layer, lattice, order, axis, angle):
    epsilon, reciprocal = assemble_eps_coeffs(np, layer['shapes'], layer['background'],
                                              *order, lattice, np.complex128)
    tangent = about_centre(conv_matrix_2d(np, epsilon, *order), order)
    normal = np.linalg.inv(about_centre(conv_matrix_2d(np, reciprocal, *order), order))
    theta = angle+axis*np.pi/2
    c, s = np.cos(theta), np.sin(theta)
    return (c*c*normal+s*s*tangent, s*s*normal+c*c*tangent, np.linalg.inv(tangent),
            c*s*(tangent-normal), c*s*(tangent-normal))


@pytest.mark.research
@pytest.mark.parametrize('transform', TRANSFORMS)
@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('order', [(2, 1), (1, 0), (0, 1)])
def test_original_cell_origin_and_retained_orders_match_analytic_material_integrals(transform, axis, order):
    structure, lattice = rebased_scene(transform=transform, axis=axis, preserve_positions=False)
    builder = _legacy_wavebit.ConformingNVF(order, **SETTINGS)
    plan = builder.plan(structure, lattice)
    t = np.asarray(plan.reduction)
    assert round(np.linalg.det(t)) == 1
    assert not np.array_equal(t, np.eye(2))
    reduced = lattice@t
    frame = reduced/np.linalg.norm(reduced, axis=0)
    np.testing.assert_allclose(frame.T@frame, np.eye(2), atol=2e-14, rtol=0)
    actual = builder.conv_matrices(structure, lattice, plan=plan)
    assert actual.epx.shape == (1, (2*order[0]+1)*(2*order[1]+1), (2*order[0]+1)*(2*order[1]+1))
    layer = structure[0]
    eps, _ = assemble_eps_coeffs(np, layer['shapes'], layer['background'], *order, lattice, np.complex128)
    np.testing.assert_allclose(np.linalg.inv(actual.epz_i[0]), about_centre(conv_matrix_2d(np, eps, *order), order),
                               atol=3e-12, rtol=0)


@pytest.mark.research
@pytest.mark.parametrize('transform', TRANSFORMS)
@pytest.mark.parametrize('axis', [0, 1])
def test_rebased_contact_matches_independent_inverse_rules(transform, axis):
    p = PARAMETERS.copy()
    p[1] = 0.
    structure, lattice = rebased_scene(p, transform, axis=axis, preserve_positions=False)
    order = (2, 1)
    actual = _legacy_wavebit.ConformingNVF(order, **SETTINGS).conv_matrices(structure, lattice)
    expected = contact_reference(structure[0], lattice, order, axis, p[9])
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got[0], want, atol=4e-12, rtol=0)


@pytest.mark.research
@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('transform', TRANSFORMS)
def test_nonrectangular_retained_set_is_selected_before_energy_closure(monkeypatch, axis, transform):
    structure, lattice = rebased_scene(transform=transform, axis=axis)
    order = (2, 1)
    actual = _legacy_wavebit.ConformingNVF(order, **SETTINGS).conv_matrices(structure, lattice)
    reference_structure, reference_lattice = complex_scene(axis=axis)
    n, m = np.meshgrid(np.arange(-order[1], order[1]+1), np.arange(-order[0], order[0]+1), indexing='ij')
    # Match physical reciprocal vectors; this does not use the implementation's
    # integer remapping or its chosen reduced basis.
    physical = np.linalg.solve(lattice.T, np.stack([m.ravel(), n.ravel()]))
    labels = reference_lattice.T@physical
    np.testing.assert_allclose(labels, np.rint(labels), atol=5e-14, rtol=0)
    labels = np.rint(labels).astype(int)
    envelope = tuple(np.max(np.abs(labels), axis=1))
    index = (labels[1]+envelope[1])*(2*envelope[0]+1)+labels[0]+envelope[0]
    captured = []
    close = _legacy_wavebit.ConvMatrices.from_energy
    def capture(cls, change, gram, longitudinal):
        captured.append((change, gram, longitudinal))
        return close(change, gram, longitudinal)
    monkeypatch.setattr(_legacy_wavebit.ConvMatrices, 'from_energy', classmethod(capture))
    _legacy_wavebit.ConformingNVF(envelope, **SETTINGS).conv_matrices(reference_structure, reference_lattice)
    q, g, z = captured[0]
    both = np.r_[index, index+len(z)]
    expected = slots_about_centre(
        close(q[np.ix_(both, both)], g[np.ix_(both, both)], z[np.ix_(index, index)]), order)
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got[0], want, atol=5e-12, rtol=0)


@pytest.mark.research
@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('transform,order', [(TRANSFORMS[0], (0, 1)), (TRANSFORMS[1], (1, 0))])
def test_identical_physical_mode_sets_preserve_positive_gap_engine_results(axis, transform, order):
    structure, lattice = rebased_scene(transform=transform, axis=axis)
    reference_structure, reference_lattice = complex_scene(axis=axis)
    builder = _legacy_wavebit.ConformingNVF(order, **SETTINGS)
    actual = builder.conv_matrices(structure, lattice)
    reference = builder.conv_matrices(reference_structure, reference_lattice)
    for got, want in zip(actual, reference):
        np.testing.assert_allclose(got, want, atol=4e-12, rtol=0)
    np.testing.assert_allclose(solve(actual, lattice, order, scale=.65),
                               solve(reference, reference_lattice, order, scale=.65), atol=2e-10, rtol=0)


@pytest.mark.research
@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_rebased_mixed_axis_contact_stack_matches_analytic_controls_on_three_engines(backend):
    if backend == 'jax':
        pytest.importorskip('jax')
        _legacy_wavebit.enable_x64()
    elif backend == 'torch':
        pytest.importorskip('torch')
    p = PARAMETERS.copy()
    p[1] = 0.
    horizontal, lattice = rebased_scene(p)
    vertical, _ = rebased_scene(p, losses=(.9, 0., .3), axis=1)
    uniform = 2.3+.7j
    order = (2, 1)
    actual = _legacy_wavebit.ConformingNVF(order, **SETTINGS).conv_matrices(
        horizontal+[dict(background=uniform)]+vertical, lattice)
    identity = np.eye(15, dtype=complex)
    layers = [contact_reference(horizontal[0], lattice, order, 0, p[9]),
              (uniform*identity, uniform*identity, identity/uniform, 0*identity, 0*identity),
              contact_reference(vertical[0], lattice, order, 1, p[9])]
    expected = _legacy_wavebit.ConvMatrices(*(np.stack([layer[i] for layer in layers]) for i in range(5)))
    np.testing.assert_allclose(solve(actual, lattice, order, scale=.65, backend=backend),
                               solve(expected, lattice, order, scale=.65), atol=2e-10, rtol=0)


@pytest.mark.research
@pytest.mark.parametrize('axis,transform', [(0, TRANSFORMS[0]), (1, TRANSFORMS[1])])
def test_rebased_geometry_storage_loss_and_period_derivatives_remain_live(axis, transform):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    builder = _legacy_wavebit.ConformingNVF(1, **SETTINGS)
    plan = builder.plan(*rebased_scene(transform=transform, axis=axis))
    parameters = np.r_[PARAMETERS, .3, .8, 2.1]
    def objective(p, xp):
        return _objective(builder.conv_matrices(*rebased_scene(p[:12], transform, p[12:], xp, axis),
                                                plan=plan, xp=xp), xp)
    value, derivative = jax.jit(jax.value_and_grad(lambda p: objective(p, jnp)))(jnp.asarray(parameters))
    assert value == pytest.approx(objective(parameters, np), abs=3e-10)
    steps = [1e-3]*4+[1e-4]*3+[1e-3]*2+[1e-5]+[1e-3]*2+[1e-5]*3
    for i, step in enumerate(steps):
        for scale in (1., .5):
            delta = np.eye(15)[i]*step*scale
            secant = (objective(parameters+delta, np)-objective(parameters-delta, np))/(2*step*scale)
            assert derivative[i] == pytest.approx(secant, abs=3e-7, rel=5e-6)
    jax.clear_caches()


@pytest.mark.research
@pytest.mark.parametrize('axis,transform,order', [(0, TRANSFORMS[0], (0, 1)), (1, TRANSFORMS[1], (1, 0))])
def test_rebasing_preserves_contact_opening_derivatives_for_identical_physical_modes(axis, transform, order):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    p = PARAMETERS.copy()
    p[[1, 3]] = 0.
    builder = _legacy_wavebit.ConformingNVF(order, **SETTINGS)
    original_plan = builder.plan(*complex_scene(p, axis=axis))
    rebased_plan = builder.plan(*rebased_scene(p, transform, axis=axis))
    directions = jnp.asarray(np.eye(12)[[1, 3]])
    def objectives(gaps):
        live = jnp.asarray(p)+directions.T@gaps
        original = builder.conv_matrices(*complex_scene(live, xp=jnp, axis=axis), plan=original_plan, xp=jnp)
        rebased = builder.conv_matrices(*rebased_scene(live, transform, xp=jnp, axis=axis), plan=rebased_plan, xp=jnp)
        return jnp.stack([_objective(original, jnp), _objective(rebased, jnp)])
    values, derivatives = jax.jit(lambda gaps: (objectives(gaps), jax.jacfwd(objectives)(gaps)))(jnp.zeros(2))
    np.testing.assert_allclose(values[0], values[1], atol=3e-10, rtol=0)
    np.testing.assert_allclose(derivatives[0], derivatives[1], atol=3e-9, rtol=0)
    assert np.min(np.abs(derivatives[1])) > 1e-4
    jax.clear_caches()


def test_replay_rejects_a_changed_reduction_or_genuinely_nonorthogonal_lattice():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    builder = _legacy_wavebit.ConformingNVF(0, **SETTINGS)
    structure, lattice = rebased_scene()
    plan = builder.plan(structure, lattice)
    for reduction in (((2, 0), (0, 1)), ((0, 1), (1, 0)), ((1., 0), (0, 1)), ((1, 0),), None):
        with pytest.raises(_legacy_wavebit.StaleGeometryError, match='reduction'):
            builder.conv_matrices(structure, lattice, plan=plan._replace(reduction=reduction))
    with pytest.raises(ValueError, match='orthogonal'):
        builder.conv_matrices(structure, lattice, plan=plan._replace(reduction=((1, 0), (0, 1))))
    changed = lattice.copy()
    changed[:, 1] += .01*changed[:, 0]
    with pytest.raises(ValueError, match='orthogonal'):
        builder.conv_matrices(structure, changed, plan=plan)
    with pytest.raises(ValueError, match='orthogonal'):
        builder.plan(structure, changed)
    replay = jax.jit(lambda basis: builder.conv_matrices(structure, basis, plan=plan, xp=jnp))
    assert all(np.isnan(np.asarray(a)).all() for a in replay(jnp.asarray(changed)))
    assert all(np.isfinite(np.asarray(a)).all() for a in replay(jnp.asarray(lattice)))
    jax.clear_caches()
