"""
`RCWAEngine.conv_matrices` — the public way to see what a modeling path built.

`solve_for_conv` has always *taken* these five matrices; nothing returned them, so a caller who
wanted to compare two modeling paths had to reach into `_core` and reassemble the quantity from
the doubled reciprocal grid, reproducing its layout and the order-index sign convention by hand.
That fails silently when guessed. These tests pin the two properties that make the accessor
worth having: it is the same object the solve consumes, and it is the same object on every path.
"""
import numpy as np
import pytest

import fieldwave

P = 1.0
BASE = dict(period=[P, P], thickness=[0.4], wavelength=0.8, theta=0.0, phi=0.0,
            n_top=1.0, n_bot=1.5, fto=[4, 4])
CIRCLE = dict(type='circle', cx=0.5, cy=0.5, lx=0.6, n_index=3.0)


def phi_circle(N, r=0.30):
    j = (np.arange(N) + 0.5) / N * P
    X, Y = np.meshgrid(j, j, indexing='xy')
    return r - np.hypot(X - 0.5, Y - 0.5)


def test_it_is_exactly_what_solve_for_conv_takes():
    """The round trip is the contract: feeding the result back reproduces conv_solve()."""
    solver = fieldwave.rcwa(ucell=[dict(n_base=1.0, shapes=[CIRCLE])], **BASE)
    direct = solver.conv_solve().res.de_ti
    relayed = solver.solve_for_conv(solver.wavelength, *solver.conv_matrices()).res.de_ti
    assert np.abs(np.asarray(direct) - np.asarray(relayed)).max() == 0.0


def test_every_modeling_path_answers_with_the_same_shaped_object():
    """The point of the accessor: one quantity, comparable across paths that share nothing."""
    ff = (2 * 4 + 1) ** 2
    raster = np.where(phi_circle(96) > 0, 3.0, 1.0)[None, :, :]
    paths = {
        'raster+nvf': fieldwave.rcwa(ucell=raster, factorization='nvf', **BASE),
        'smooth': fieldwave.rcwa(ucell=[dict(n_base=1.0, shapes=[CIRCLE])], **BASE),
        'levelset': fieldwave.rcwa(ucell=[dict(n_base=1.0, n_index=3.0, phi=phi_circle(96))],
                                   **BASE),
    }
    for name, solver in paths.items():
        epx, epy, epz_i, epxy, epyx = solver.conv_matrices()
        for tag, m in (('epx', epx), ('epy', epy), ('epz_i', epz_i)):
            assert np.asarray(m).shape == (1, ff, ff), (name, tag, np.asarray(m).shape)
            assert np.isfinite(np.asarray(m)).all(), (name, tag)
        assert epxy is not None and epyx is not None, name       # all three are NVF paths


def test_a_frozen_geometry_is_accepted_the_same_way_conv_solve_accepts_one():
    solver = fieldwave.rcwa(ucell=[dict(n_base=1.0, n_index=3.0, phi=phi_circle(96))], **BASE)
    eager = solver.conv_matrices()
    replayed = solver.conv_matrices(geometry=solver.freeze_geometry())
    for a, b in zip(eager, replayed):
        assert np.abs(np.asarray(a) - np.asarray(b)).max() < 1e-14


def test_it_does_not_solve():
    """Cheaper than conv_solve by the whole eigenproblem — otherwise there is no reason to
    have it separately, and a sweep over geometries would pay for a solve it discards."""
    import time
    solver = fieldwave.rcwa(ucell=[dict(n_base=1.0, shapes=[CIRCLE])], **dict(BASE, fto=[8, 8]))
    solver.conv_matrices(), solver.conv_solve()             # warm

    t = time.perf_counter()
    for _ in range(3):
        solver.conv_matrices()
    build = time.perf_counter() - t

    t = time.perf_counter()
    for _ in range(3):
        solver.conv_solve()
    whole = time.perf_counter() - t

    assert build < 0.7 * whole, (build, whole)
