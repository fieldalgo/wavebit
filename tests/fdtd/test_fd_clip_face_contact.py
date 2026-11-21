"""An edge on a cell face moves one cell's area: the clip's tangent at an exact contact.

Both clips -- ``_shapes/_exact.py::clipped_polygon_area`` and its source,
``_periodic/scene.py::_clipped_rectangle_area`` -- admit a vertex on a face to both cells that share
it. Before 2026-11-21 the vertex carried its tangent in both, so an axis-aligned square on the
faces of a 0.125 grid moved the summed area at twice its side (0.75 against 0.375), and on a 0.1
grid, whose faces are a rounding off the edges, the doubling, a translation slope (-0.3 where a
translation moves no area) and the cell it landed in came and went per edge and per backend.
"""

import numpy as np
import pytest

from wavebit.fdtd._shapes._exact import clipped_polygon_area
from wavebit.fdtd._periodic.runtime import time_loop_backend
from wavebit.fdtd._periodic.scene import _require_scene, _shape_cell_fractions, _shape_partition

jax = pytest.importorskip('jax')
torch = pytest.importorskip('torch')
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp  # noqa: E402
from array_api_compat import torch as xtorch  # noqa: E402


def _voxel_areas(xp, edges, faces):
    """The square ``[x0, x1] x [y0, y1]`` clipped to every voxel of the ``faces`` grid."""
    x0, x1, y0, y1 = edges[0], edges[1], edges[2], edges[3]
    vertices = xp.stack([xp.stack([x0, y0]), xp.stack([x1, y0]),
                         xp.stack([x1, y1]), xp.stack([x0, y1])])
    lo, hi = xp.asarray(faces[:-1]), xp.asarray(faces[1:])
    zero = 0 * lo[:, None] + 0 * lo[None, :]
    return clipped_polygon_area(vertices, lo[None, :] + zero, hi[None, :] + zero,
                                lo[:, None] + zero, hi[:, None] + zero, xp,
                                convex=True).reshape(-1)


@pytest.mark.parametrize('step', [0.125, 0.1])
def test_a_polygon_edge_on_a_voxel_face_moves_the_area_once_on_every_backend(step):
    faces = np.arange(9) * step
    edges = np.array([1, 4, 2, 5]) * step               # every edge on a face
    side = 3 * step
    want = np.array([-side, side, -side, side])
    h = 1e-7
    central = np.array([(_voxel_areas(np, edges + h * e, faces).sum()
                         - _voxel_areas(np, edges - h * e, faces).sum()) / (2 * h)
                        for e in np.eye(4)])
    np.testing.assert_allclose(central, want, atol=1e-8)
    jacobians = {
        'jax': np.asarray(jax.jacrev(lambda q: _voxel_areas(jnp, q, faces))(jnp.asarray(edges))),
        'jit': np.asarray(jax.jit(jax.jacrev(lambda q: _voxel_areas(jnp, q, faces)))(
            jnp.asarray(edges))),
        'torch': torch.autograd.functional.jacobian(
            lambda q: _voxel_areas(xtorch, q, faces), torch.tensor(edges)).numpy(),
    }
    for name, jacobian in jacobians.items():
        np.testing.assert_allclose(jacobian.sum(axis=0), want, atol=1e-12, err_msg=name)
        np.testing.assert_allclose(jacobian, jacobians['torch'], atol=1e-13, err_msg=name)


def _fractions(backend, step, n):
    runtime = time_loop_backend(backend)

    def fractions(p):
        shape = dict(type='rectangle', cx=p[0], cy=p[1], lx=p[2], ly=p[3], angle=p[4])
        _, _, layers = _shape_partition([dict(shapes=[shape])], [n * step] * 2, xp=runtime.xp,
                                        array=runtime.array, check=_require_scene,
                                        integration='area')
        return _shape_cell_fractions(layers[0][1][0], np.arange(n) * step, np.arange(n) * step,
                                     step, step, [n * step] * 2, xp=runtime.xp,
                                     array=runtime.array).reshape(-1)
    return fractions


@pytest.mark.parametrize('step, n', [(0.125, 8), (0.1, 10)])
@pytest.mark.parametrize('angle', [0.0, np.pi / 2, np.pi])
@pytest.mark.parametrize('cx', [0.3, 0.0])
def test_a_rectangle_on_cell_faces_has_its_own_area_derivative_on_every_backend(step, n, angle, cx):
    """``cx = 0.3`` puts the right edge on the periodic seam, where the cell above the face is
    the first cell and the vertex is an image; ``cx = 0`` keeps both edges inside the cell."""
    side = 3 * step
    p = np.array([cx, 4 * step, side, side, angle])
    fj = _fractions('jax', step, n)
    ft = _fractions('torch', step, n)
    value = np.asarray(_fractions('numpy', step, n)(p))
    np.testing.assert_allclose(value.sum() * step * step, side * side, atol=1e-15)
    jacobians = {
        'jax': np.asarray(jax.jacrev(fj)(jnp.asarray(p))),
        'jit': np.asarray(jax.jit(jax.jacrev(fj))(jnp.asarray(p))),
        'torch': torch.autograd.functional.jacobian(
            ft, torch.tensor(p, dtype=torch.float64)).numpy(),
    }
    want = np.array([0., 0., side, side, 0.])          # d(area)/d(cx, cy, lx, ly, angle)
    for name, jacobian in jacobians.items():
        np.testing.assert_allclose(jacobian.sum(axis=0) * step * step, want, atol=1e-12,
                                   err_msg=name)
        np.testing.assert_allclose(jacobian, jacobians['torch'], atol=1e-12, err_msg=name)
