"""Geometric identities at polygon ties, and the fields they protect.

The initial repaired probe measured fill error <=5.5e-15, translated-weight
drift <=3e-16 and a full-turn normal-tensor gap of 9.1e-15. Dimensionless
checks reuse the existing mirror's 1e-12 bound; surface areas retain the
existing exact-face regression's 1e-15 absolute bound.
"""
import numpy as np
import pytest

import wavebit.fd as fd
from fd_helpers import to_numpy
from wavebit.fd._runtime import runtime
from wavebit.fd._shapes import Box, Prism, occupancy


def evaluate(function, backend, points):
    rt = runtime('jax' if backend == 'jit' else backend)
    if backend == 'jit':
        import jax
        function = jax.jit(function)
    return [to_numpy(value) for value in function(rt.array(points, 'float64'))]


@pytest.mark.parametrize('backend', ('numpy', 'jax', 'torch', 'jit'))
@pytest.mark.parametrize('axis', ('x', 'y', 'z'))
def test_boundary_faces_share_area_without_a_preferred_axis(backend, axis):
    # The positive x face has area .04. A .05-square column wholly in that
    # face sees .0025 total, divided equally between the two adjacent voxels.
    points = np.array([[.175, 0., 0.], [.225, 0., 0.]])
    body = Box((.4, .2, .2), axis=axis)
    area, moment = evaluate(lambda p: body.exact_moment(p, .05), backend, points)
    np.testing.assert_allclose(area, [.00125, .00125], rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(moment, [[.00125, 0., 0.]]*2, rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize('backend', ('numpy', 'jax', 'torch', 'jit'))
@pytest.mark.parametrize('scale', (1e-6, 1., 1e6))
def test_translated_boundary_moments_keep_the_same_geometric_answer(backend, scale):
    points = np.array([[.175, .1, 0.], [-.175, .1, 0.], [.1, .075, .1]])
    # Boundary face: half of .025*.05; interior face: .05*.05. Their
    # area ratio is 1:4, independently of which Box axis spells the cap.
    expected = [[.2, .8, 0.], [.2, .8, 0.], [0., .2, .8]]
    for shift in (0., .013, .137, .6, 3.):
        body = Box(tuple(scale*np.array([.4, .2, .2])), center=(scale*shift,)*3)
        fill, moment = evaluate(lambda p: occupancy(body, p, .05*scale),
                                backend, scale*(points+shift))
        np.testing.assert_allclose(fill, .5, rtol=0, atol=1e-12)
        np.testing.assert_allclose(moment, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize('backend', ('numpy', 'jax', 'torch', 'jit'))
@pytest.mark.parametrize('scale', (1e-6, 1., 1e6))
def test_equal_distance_normals_do_not_depend_on_the_first_vertex(backend, scale):
    vertices = scale*np.array([[-.15, -.15], [.15, -.15], [.15, .15], [-.15, .15]])
    points = scale*np.array([[0., .18, 0.], [0., .2, 0.], [0., -.2, 0.], [0., 0., 0.]])
    expected = np.array([[0., 1., 0.], [0., 1., 0.], [0., -1., 0.], [1., 0., 0.]])
    for first in range(4):
        body = Prism(np.roll(vertices, first, axis=0), .4*scale, angle=np.pi/4)
        _, normal = evaluate(body.surface, backend, points)
        np.testing.assert_allclose(normal, expected, rtol=0, atol=1e-12)
        # A point 1e-8 of the shape scale away from the tie has a unique
        # nearest face. Averaging all near faces must not blur this choice.
        _, away = evaluate(body.surface, backend, scale*np.array([[1e-8, .2, 0.]]))
        np.testing.assert_allclose(away, [[2**-.5, 2**-.5, 0.]], rtol=0, atol=1e-12)
        # CSG needs a containing plane through an actual nearest face. A
        # bisector at the nearest distance cuts into this diamond's corner.
        distance, support = evaluate(body.support_surface, backend, points)
        outline = vertices @ (np.array([[1., 1.], [-1., 1.]])/2**.5)
        excess = np.sum((outline[None, :, :]-points[:, None, :2])
                        *support[:, None, :2], axis=-1)+distance[:, None]
        assert np.max(excess) <= 1e-12*scale


@pytest.mark.parametrize('backend', ('numpy', 'jit', 'torch'))
def test_a_full_turn_keeps_the_same_coupled_fields(backend):
    from test_fd_constitutive import solve_cavity
    def scene(angle):
        return fd.Scene(domain=fd.Domain(size=(.8,)*3), resolution=fd.Resolution(dx=.05),
                        boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()),
                        run=fd.Run(steps=80),
                        structures=(fd.Structure(fd.Box(size=(.3, .3, .4), angle=angle), 4.),))
    before, after = (solve_cavity(backend, scene(angle)) for angle in (45., 405.))
    for key in before:
        # Reuse the existing half/full mirror's 1e-12 relative field bound.
        scale = max(float(np.max(np.abs(before[key]))), np.finfo(float).tiny)
        assert np.max(np.abs(before[key]-after[key])) <= 1e-12*scale, key


@pytest.mark.parametrize('backend', ('numpy', 'jax', 'torch'))
def test_boundary_aligned_mirror_reproduces_the_full_box(backend):
    from studies.fdtd_polygon_ties import mirror_series
    full, half = mirror_series(False, backend), mirror_series(True, backend)
    assert np.max(np.abs(full-half)) <= 1e-12*np.max(np.abs(full))
