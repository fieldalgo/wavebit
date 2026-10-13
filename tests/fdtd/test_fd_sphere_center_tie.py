"""A sphere's derivatives where a voxel face passes through its centre plane (the centre tie).

``src/wavebit/fdtd/_shapes/_exact.py::clipped_ball_volume`` folds each corner with ``_fold`` and
``_ball_corner`` gives an empty regime a constant zero limit. Before, ``xp.abs`` dropped a
derivative term on Torch and ``min(0, c)`` split one on both backends whenever a face held the
centre. Measurements: ``studies/fdtd/sphere_center_tie/fdtd_sphere_center_tie_probe.py``, results
``studies/results/fdtd_sphere_center_tie_{fill,run}_{before,after}.json``.
"""
import numpy as np
import pytest

import wavebit.fdtd._legacy as fd
from wavebit.fdtd._runtime import runtime
from wavebit.fdtd._shapes import _exact
from studies.fdtd.sphere_center_tie.fdtd_sphere_center_tie_probe import CENTRES, OFF, RADIUS, corner_points, energy, fill_points, fills, scene
from tests.fdtd.test_fd_sphere_range import quiet


def _old_ball_corner(a, b, c, radius, xp):
    """The primitive as it was before the fix (``min(limit, c)`` for every regime), verbatim."""
    R = radius
    a, b, c = xp.minimum(a, R), xp.minimum(b, R), xp.minimum(c, R)
    ka = _exact._root((R - a) * (R + a), xp)
    kb = _exact._root((R - b) * (R + b), xp)
    rect = xp.minimum(_exact._root((ka - b) * (ka + b), xp), c)
    both = xp.minimum(xp.minimum(ka, kb), c)
    one = xp.minimum(xp.maximum(ka, kb), c)
    lesser, outer = xp.minimum(a, b), xp.maximum(ka, kb)

    def two_chords(z):
        return .5 * (a * _exact._arc_integral(z, ka, xp) + b * _exact._arc_integral(z, kb, xp)
                     + _exact._wedge_integral(z, a, R, xp) + _exact._wedge_integral(z, b, R, xp)
                     - (_exact._PI / 2) * _exact._cubic_integral(z, R, xp))

    def one_chord(z):
        return .5 * (lesser * _exact._arc_integral(z, outer, xp)
                     + _exact._wedge_integral(z, lesser, R, xp))

    return (a * b * rect + (two_chords(both) - two_chords(rect))
            + (one_chord(one) - one_chord(both))
            + (_exact._PI / 4) * (_exact._cubic_integral(c, R, xp)
                                  - _exact._cubic_integral(one, R, xp)))


def _old_volume(cx, cy, cz, radius, x0, x1, y0, y1, z0, z1, xp):
    """``clipped_ball_volume`` before the fix, folding with ``xp.abs``."""
    corners = ((x0 - cx, x1 - cx), (y0 - cy, y1 - cy), (z0 - cz, z1 - cz))
    total = None
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                a, b, c = corners[0][i], corners[1][j], corners[2][k]
                sign = 1. if (i + j + k) % 2 == 1 else -1.
                one = xp.ones_like(a)
                signs = (xp.where(a >= 0, one, -one) * xp.where(b >= 0, one, -one)
                         * xp.where(c >= 0, one, -one))
                term = sign * signs * _old_ball_corner(xp.abs(a), xp.abs(b), xp.abs(c),
                                                       radius, xp)
                total = term if total is None else total + term
    zero = xp.zeros_like(total)
    far = sum(xp.maximum(lo * lo, hi * hi) for lo, hi in corners)
    near = sum(xp.maximum(xp.maximum(lo, -hi), xp.zeros_like(lo)) ** 2 for lo, hi in corners)
    box = (x1 - x0) * (y1 - y0) * (z1 - z0) + zero
    return xp.where(near >= radius * radius, zero, xp.where(far <= radius * radius, box, total))


def _boxes():
    """300 random ball/box pairs and the ties: faces through the centre plane on each axis and on
    three at once, offsets of exactly the radius, equal offsets on two axes, and ``-0.0``."""
    rng = np.random.default_rng(20261001)
    centre = rng.uniform(-.2, .2, (300, 3))
    low = centre + rng.uniform(-.35, .25, (300, 3))
    rows = [np.concatenate([centre, low, low + rng.uniform(.02, .15, (300, 3))], axis=1)]
    c = np.array(OFF)
    for axis in range(3):
        for start in (0., -.1, -RADIUS, RADIUS - .1, -.05):
            for shift in (-.13, .06):
                lo = c + shift
                lo[axis] = c[axis] + start
                rows.append(np.concatenate([c, lo, lo + .1])[None])
    rows.append(np.concatenate([c, c, c + .1])[None])
    rows.append(np.concatenate([c, c - .1, c])[None])
    rows.append(np.array([[0., 0., 0., -0., -0., -0., .1, .1, .1]]))
    return np.concatenate(rows)


def _volume(fn, rows, rt, radius=RADIUS):
    a = lambda k: rt.array(rows[:, k], 'float64')
    x0, y0, z0, x1, y1, z1 = (a(k) for k in range(3, 9))
    return fn(a(0), a(1), a(2), rt.array(radius, 'float64'), x0, x1, y0, y1, z0, z1, rt.xp)


@pytest.mark.parametrize('backend', ('numpy', 'jax', 'torch'))
def test_the_ball_volume_is_unchanged_to_the_bit(backend):
    """The fix changes derivatives only: every volume, ties included, is the same bytes."""
    rt = runtime(backend)
    rows = _boxes()
    new = rt.numpy(_volume(_exact.clipped_ball_volume, rows, rt))
    old = rt.numpy(_volume(_old_volume, rows, rt))
    assert np.any(new > 0) and np.any((new > 0) & (new < np.prod(rows[:, 6:] - rows[:, 3:6], 1)))
    assert np.asarray(new).tobytes() == np.asarray(old).tobytes()


@pytest.mark.parametrize('backend', ('jax', 'torch'))
def test_the_fold_is_abs_to_the_bit_with_its_sign_as_the_derivative_at_zero(backend):
    rt = runtime(backend)
    values = np.array([-2.5, -0., 0., 1e-300, 3.])
    s, folded = _exact._fold(rt.array(values, 'float64'), rt.xp)
    assert rt.numpy(folded).tobytes() == np.abs(values).tobytes()
    assert list(rt.numpy(s)) == [-1., 1., 1., 1., 1.]
    if backend == 'torch':
        import torch
        v = torch.zeros((), dtype=torch.float64, requires_grad=True)
        derivative = torch.autograd.grad(_exact._fold(v, torch)[1], v)[0]
    else:
        import jax
        import jax.numpy as jnp
        derivative = jax.grad(lambda v: _exact._fold(v, jnp)[1])(jnp.zeros(()))
    assert float(derivative) == 1.


@pytest.mark.parametrize('backend', ('jax', 'torch'))
@pytest.mark.parametrize('case', ('x', 'y', 'z', 'xyz', 'corner'))
def test_a_face_through_the_centre_plane_keeps_every_derivative(case, backend):
    """Per-voxel fills of voxels whose faces hold the centre plane, weighted and summed: reverse
    mode against a NumPy central difference in the radius and each centre coordinate. Measured
    after the fix: at most 7.4e-10 relative; before it, up to 12.5 (Torch) and 4.1 (JAX).
    ``corner`` reaches the quarter-disc regime (both chords gone) the other cases never cut;
    a mutant tying that regime's limit survived the first four cases."""
    axis, all_axes = {'x': (0, False), 'y': (1, False), 'z': (2, False),
                      'xyz': (0, True), 'corner': (None, None)}[case]
    points = corner_points() if axis is None else fill_points(axis, all_axes)
    weights = np.linspace(1., 2., len(points))
    leaves = np.array((RADIUS, *OFF))

    def objective(v, name):
        rt = runtime(name)
        return (fills(name, v[0], (v[1], v[2], v[3]), points)
                * rt.array(weights, 'float64')).sum()
    h = 1e-6
    expected = np.array([(float(objective(leaves + d, 'numpy'))
                          - float(objective(leaves - d, 'numpy'))) / (2 * h)
                         for d in np.eye(4) * h])
    runtime(backend)
    if backend == 'torch':
        import torch
        v = torch.tensor(leaves, dtype=torch.float64, requires_grad=True)
        got = torch.autograd.grad(objective(v, 'torch'), v)[0].numpy()
    else:
        import jax
        import jax.numpy as jnp
        got = np.asarray(jax.grad(lambda x: objective(x, 'jax'))(jnp.asarray(leaves)))
    assert np.all(expected != 0)
    np.testing.assert_allclose(got, expected, rtol=2e-6, atol=0.)


@pytest.mark.parametrize('backend', ('jax', 'torch'))
@pytest.mark.parametrize('where', ('half', 'half_z'))
def test_a_sphere_centred_on_half_node_planes_differentiates_like_its_difference(where, backend):
    """A lone ranged sphere in a PEC box with its centre on half-node planes (all three axes, or
    z alone): radius and centre derivatives of the final electric energy in a fixed slab.
    Measured after the fix 3.5e-8 and 2.5e-10 relative; before it, 176 and 2.0 (Torch), 0.64
    and 0.40 (JAX)."""
    point = np.array((RADIUS, *CENTRES[where]))
    p = quiet(fd.plan, scene(point))
    h = 3e-6
    expected = np.array([(float(quiet(energy, point + d, 'numpy', p))
                          - float(quiet(energy, point - d, 'numpy', p))) / (2 * h)
                         for d in np.eye(4) * h])
    runtime(backend)
    if backend == 'torch':
        import torch
        v = torch.tensor(point, dtype=torch.float64, requires_grad=True)
        got = torch.autograd.grad(quiet(energy, v, 'torch', p), v)[0].numpy()
    else:
        import jax
        import jax.numpy as jnp
        got = np.asarray(quiet(jax.grad(lambda x: energy(x, 'jax', p)), jnp.asarray(point)))
    assert np.all(expected != 0)
    np.testing.assert_allclose(got, expected, rtol=2e-6, atol=0.)
