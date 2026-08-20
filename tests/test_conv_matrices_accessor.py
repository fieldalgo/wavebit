"""
`RCWAEngine.conv_matrices` — the public way to see what a modeling path built.

`solve_conv` has always *taken* these five matrices; nothing returned them, so a caller who
wanted to compare two modeling paths, check a geometry, or build a path of their own had to reach
into `_core` and reassemble the quantity from the doubled reciprocal grid, reproducing its layout
and the order-index sign convention by hand. That fails silently when guessed — three separate
attempts produced three wrong numbers in one day, all of them from someone doing something
legitimate.

These tests pin the two properties that make it worth having: it is the same object the solve
consumes, and it is the same object on every path.
"""
import numpy as np
import pytest

import wavebit

P = 1.0
BASE = dict(period=[P, P], thickness=[0.4], wavelength=0.8, theta=0.0, phi=0.0,
            top=1.0, bot=2.25, fto=[4, 4])
CIRCLE = dict(type='circle', cx=0.5, cy=0.5, lx=0.6, material=9.0)   # eps 9 is the old n = 3


def disc_raster(N, r=0.30):
    j = (np.arange(N) + 0.5) / N * P
    X, Y = np.meshgrid(j, j, indexing='xy')
    return np.where(r - np.hypot(X - 0.5, Y - 0.5) > 0, 3.0, 1.0)[None, :, :]


def test_it_is_exactly_what_solve_conv_takes():
    """The round trip is the contract: feeding the result back reproduces solve()."""
    solver = wavebit.rcwa(structure=[dict(background=1.0, shapes=[CIRCLE])], **BASE)
    direct = solver.solve().res.T
    relayed = solver.solve_conv(*solver.conv_matrices()).res.T
    assert np.abs(np.asarray(direct) - np.asarray(relayed)).max() == 0.0


def test_every_modeling_path_answers_with_the_same_shaped_object():
    """The point of the accessor: one quantity, comparable across paths that share nothing."""
    ff = (2 * 4 + 1) ** 2
    paths = {
        'raster+nvf': wavebit.rcwa(structure=disc_raster(96), factorization='nvf', **BASE),
        'smooth': wavebit.rcwa(structure=[dict(background=1.0, shapes=[CIRCLE])], **BASE),
        'boundary': wavebit.rcwa(structure=[dict(background=1.0, combine='paint', shapes=[
            dict(type='rectangle', cx=0.40, cy=0.40, lx=0.34, ly=0.34, material=4.0),
            dict(type='rectangle', cx=0.56, cy=0.56, lx=0.34, ly=0.34, angle=0.3,
                 material=9.0)])], **BASE),
    }
    for name, solver in paths.items():
        epx, epy, epz_i, epxy, epyx = solver.conv_matrices()
        for tag, m in (('epx', epx), ('epy', epy), ('epz_i', epz_i)):
            assert np.asarray(m).shape == (1, ff, ff), (name, tag, np.asarray(m).shape)
            assert np.isfinite(np.asarray(m)).all(), (name, tag)
        assert epxy is not None and epyx is not None, name       # all three are NVF paths


def test_a_frozen_geometry_is_accepted_the_same_way_conv_solve_accepts_one():
    solver = wavebit.rcwa(structure=disc_raster(96), factorization='nvf', **BASE)
    eager = solver.conv_matrices()
    replayed = solver.conv_matrices(geometry=solver.freeze_geometry())
    for a, b in zip(eager, replayed):
        if a is None:
            assert b is None
        else:
            assert np.abs(np.asarray(a) - np.asarray(b)).max() < 1e-14


def test_it_does_not_solve():
    """Cheaper than solve by the whole eigenproblem — otherwise there is no reason to have
    it separately, and a sweep over geometries would pay for a solve it discards."""
    import time
    solver = wavebit.rcwa(structure=[dict(background=1.0, shapes=[CIRCLE])], **dict(BASE, fto=[8, 8]))
    solver.conv_matrices(), solver.solve()             # warm

    t = time.perf_counter()
    for _ in range(3):
        solver.conv_matrices()
    build = time.perf_counter() - t

    t = time.perf_counter()
    for _ in range(3):
        solver.solve()
    whole = time.perf_counter() - t

    assert build < 0.7 * whole, (build, whole)


def test_a_misspelled_layer_key_is_refused_rather_than_ignored():
    """The failure this guards is silent: no shapes and nothing else read is a legal *uniform*
    layer, so a typo comes back as a slab of `background` that solves and conserves energy to
    1e-14.

    Found the hard way, on a layer whose one meaningful key went to a build that did not read it.
    """
    with pytest.raises(ValueError, match='not read by any path'):
        wavebit.rcwa(structure=[dict(background=1.0, shapez=[CIRCLE])], **BASE).solve()
    with pytest.raises(ValueError, match='not read by any path'):
        wavebit.rcwa(structure=[dict(background=1.0, shapes=[CIRCLE], combyne='add')],
                       **BASE).solve()
