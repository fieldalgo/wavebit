"""The external convergence comparison must describe the same physical problem.

The comparisons with S4 read frozen numbers: see
``tests/references/test_s4_resonant_nvf_golden.py``.
"""
import numpy as np
from studies.rcwa.nvf_resonant_spectrum.resonant_nvf_probe import fixture_shape, make_solver, solve_case


def test_direct_comparison_does_not_leave_nvf_disabled():
    before = solve_case(2, 64, 'circular')
    direct = solve_case(2, 64, 'wavebit-direct')
    after = solve_case(2, 64, 'circular')
    np.testing.assert_array_equal(before, after)
    assert abs(before[0]-direct[0]) > .01


def test_probe_builds_the_public_circular_nvf_option():
    solver = make_solver(1, 32, 'circular')
    assert solver.nvf_projector == 'circular' and solver.factorization == 'nvf'
    result = solver.solve()
    assert np.isfinite(result.T).all()


def test_circular_projector_has_exact_normals_on_the_tilted_visible_faces():
    from wavebit.rcwa._engine import parametric as P
    shape = fixture_shape()
    solver = make_solver(1, 32, 'circular')
    plan = P.plan_gamma([shape], solver.lattice)
    c, s = np.cos(shape['angle']), np.sin(shape['angle'])
    rotation = np.array([[c, -s], [s, c]])
    a, b = shape['lx']/2, shape['ly']/2
    contact = shape['lx']-700*c
    # The periodic contact covers the left end of the top face and the right
    # end of the bottom face. Its interior is not a material interface.
    long = np.linspace(-a+contact+.01, a-.01, 101)
    short = np.linspace(-b+.01, b-.01, 101)
    for points, normal in (
        (np.array([long, np.full_like(long, b)]), rotation[:, 1]),
        (np.array([-long, np.full_like(long, -b)]), rotation[:, 1]),
        (np.array([np.full_like(short, a), short]), rotation[:, 0]),
        (np.array([np.full_like(short, -a), short]), rotation[:, 0]),
    ):
        want = np.broadcast_to(np.outer(normal, normal)[..., None], (2, 2, 101))
        for image in ((0., 0.), (700., 0.), (-700., 700.)):
            # The private sampler uses the unit-cell corner as its origin.
            xy = 350.+rotation@points+np.array(image)[:, None]
            xx, xy_, yy, yx = P._piece_field(solver.xp, *xy, [shape], plan, None,
                                           solver.lattice, solver.float_dtype, dyadic=True)
            np.testing.assert_allclose(np.array([[xx, xy_], [yx, yy]]), want,
                                       atol=3e-14, rtol=0)
