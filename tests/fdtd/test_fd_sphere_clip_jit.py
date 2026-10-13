"""Sphere clipping must survive composition, compilation, and a change of grid origin.

The original gate compiled a sphere alone and one CSG on one grid origin. Its guarded
square roots still returned their inactive sentinel inside a larger fused computation.
This gate covers 32 origins x four bodies x four execution paths, with full voxel grids.
The NumPy reference is also held by analytic total volume and exact CSG containment laws;
the independent disc-quadrature oracle remains in test_fd_shapes.py.

Mutants are real source edits, run and restored; the commands and measured failures live
in docs/validation/fdtd/fd-sphere-clip-jit.md. Restoring the old squared radicands kills the JIT rows and
the sentinel test. Scaling every sphere volume to 99 percent kills every backend
through the analytic-volume assertion, even if the common NumPy reference is also mutated.
"""
import numpy as np
import pytest

from wavebit.fdtd._shapes import Box, Difference, Intersection, Sphere, Union, occupancy


# Measured on this 32-origin sweep, R/h=6, float64, 2026-09-21: worst backend fill
# difference 2.30e-13 and bound excursion 2.78e-14. Allow 1e-12 (4.3x the difference).
FILL_ATOL = 1e-12
# Worst moment-component difference 2.71e-14; 1e-13 leaves a 3.6x margin.
MOMENT_ATOL = 1e-13
# Worst relative total-sphere-volume error 1.67e-15; 1e-14 leaves a 5.9x margin.
VOLUME_RTOL = 1e-14


@pytest.fixture(scope='module')
def sphere_grids():
    # The failure was reproduced at R=0.3, h=0.05; this is six cells per radius,
    # not the extreme R/h cancellation regime of a cumulative volume primitive.
    radius, h = 0.3, 0.05
    axis = -0.4 + (np.arange(16) + 0.5) * h
    base = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), axis=-1)
    # Seed 73 produced both failure signs. Keep the aligned grid as well as 31
    # off-axis origins; all grids contain the complete sphere with a cell to spare.
    offsets = np.random.default_rng(73).uniform(-0.5, 0.5, (32, 3)) * h
    offsets[0] = 0
    points = tuple(base + offset for offset in offsets)
    sphere, box = Sphere(radius), Box((0.2, 0.2, 0.9), angle=0.5)
    bodies = (sphere, Union(sphere, box), Intersection(sphere, box), Difference(sphere, box))
    reference = tuple(tuple(occupancy(body, p, h) for p in points) for body in bodies)
    # Exact box membership of the WHOLE voxel, not a tolerance on a computed fill.
    box_distances = tuple(box.surface(p)[0] for p in points)
    return radius, h, points, bodies, reference, box_distances


@pytest.mark.parametrize('backend', ('numpy', 'jax', 'jax.jit', 'torch'))
def test_sphere_and_csg_across_grid_origins(backend, sphere_grids):
    radius, h, points, bodies, reference, box_distances = sphere_grids
    convert, compile_run = np.asarray, lambda f: f
    if backend.startswith('jax'):
        jax = pytest.importorskip('jax')
        import jax.numpy as jnp
        jax.config.update('jax_enable_x64', True)
        convert = lambda p: jnp.asarray(p, dtype=jnp.float64)
        if backend == 'jax.jit':
            compile_run = jax.jit
    elif backend == 'torch':
        torch = pytest.importorskip('torch')
        convert = lambda p: torch.as_tensor(p, dtype=torch.float64)

    sphere_volume = 4 * np.pi * radius ** 3 / 3
    for kind, body in enumerate(bodies):
        # Compile one complete occupancy call, including its moment. Returning extra
        # diagnostic intermediates can change fusion and used to conceal this defect.
        run = compile_run(lambda p: occupancy(body, p, h))
        for origin, p in enumerate(points):
            fill, moment = (np.asarray(v) for v in run(convert(p)))
            label = (backend, type(body).__name__, origin)
            assert fill.dtype == np.float64 and moment.dtype == np.float64, label
            assert np.isfinite(fill).all() and np.isfinite(moment).all(), label
            assert fill.min() >= -FILL_ATOL and fill.max() <= 1 + FILL_ATOL, label
            np.testing.assert_allclose(fill, reference[kind][origin][0], rtol=0,
                                       atol=FILL_ATOL, err_msg=str(label))
            np.testing.assert_allclose(moment, reference[kind][origin][1], rtol=0,
                                       atol=MOMENT_ATOL, err_msg=str(label))
            assert moment.min() >= -MOMENT_ATOL and moment.max() <= 1 + MOMENT_ATOL, label
            trace = moment.sum(axis=-1)
            assert np.minimum(np.abs(trace), np.abs(trace - 1)).max() < MOMENT_ATOL, label
            if kind == 0:
                assert abs(fill.sum() * h ** 3 / sphere_volume - 1) < VOLUME_RTOL, label
                continue
            whole = box_distances[origin] < -np.sqrt(3) * h / 2
            empty = box_distances[origin] > np.sqrt(3) * h / 2
            sphere_fill = reference[0][origin][0]
            expected_whole = (np.ones_like(fill), sphere_fill, np.zeros_like(fill))[kind - 1]
            expected_empty = (sphere_fill, np.zeros_like(fill), sphere_fill)[kind - 1]
            np.testing.assert_allclose(fill[whole], expected_whole[whole], rtol=0,
                                       atol=FILL_ATOL, err_msg=str(label))
            np.testing.assert_allclose(fill[empty], expected_empty[empty], rtol=0,
                                       atol=FILL_ATOL, err_msg=str(label))


def test_inactive_root_value_cannot_change_a_compiled_sphere(monkeypatch, sphere_grids):
    """Positive guards 0.25, 1, 4 have square roots 0.5, 1, 2; none may escape a dead branch.

    In the old formula these changed the same sphere fill to -203.4363, -200.9404,
    -186.0971. The enclosing box contains the whole first voxel, so Intersection must
    equal the sphere's independent NumPy answer there. The other point caught a +142.65
    excursion from a true fill of 0.01055672. No extra intermediates are returned from JIT.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    from wavebit.fdtd._shapes import _exact
    jax.config.update('jax_enable_x64', True)
    _, h, points, bodies, reference, _ = sphere_grids
    p = np.stack((points[26][8, 7, 1], points[13][11, 7, 2]))
    node = bodies[2]
    expected = np.array([reference[2][26][0][8, 7, 1], reference[2][13][0][11, 7, 2]])
    for sentinel in (0.25, 1.0, 4.0):
        def guarded_root(v, xp):
            live = v > 0
            return xp.where(live, xp.sqrt(xp.where(live, v, xp.ones_like(v) * sentinel)),
                            xp.zeros_like(v))
        monkeypatch.setattr(_exact, '_root', guarded_root)
        # A fresh callable is required: reusing a JIT cache would silently reuse the
        # preceding sentinel and make this diagnostic vacuous.
        run = jax.jit(lambda p: occupancy(node, p, h))
        fill, moment = run(jnp.asarray(p, dtype=jnp.float64))
        np.testing.assert_allclose(np.asarray(fill), expected, rtol=0, atol=FILL_ATOL,
                                   err_msg=f'inactive sqrt radicand {sentinel}')
        assert np.isfinite(np.asarray(moment)).all()
