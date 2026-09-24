"""The external convergence comparison must describe the same physical problem."""
import numpy as np
import pytest
from resonant_nvf_probe import fixture_shape, make_solver, s4_direct, s4_solve, solve_case


@pytest.mark.parametrize('basis', [None, 'Default', 'Normal', 'Jones'])
def test_s4_bridge_matches_the_oblique_uniform_slab(basis):
    pytest.importorskip('S4')
    shape = dict(type='rectangle', lx=700., ly=700., material=4.)
    reference = s4_solve(1, shape, basis=basis)
    result = make_solver(1, 32, shape=shape).solve()
    np.testing.assert_allclose(reference, [result.T.sum(), result.R.sum()], atol=3e-13, rtol=0)


def test_s4_rotation_units_preserve_the_same_rectangle():
    pytest.importorskip('S4')
    shape = fixture_shape()
    equivalent = dict(shape, lx=shape['ly'], ly=shape['lx'], angle=shape['angle']+np.pi/2)
    np.testing.assert_allclose(s4_direct(2, shape), s4_direct(2, equivalent), atol=3e-12, rtol=0)


@pytest.mark.parametrize('order', [2, 4])
@pytest.mark.parametrize('shift', [(0., 0.), (41., -27.)])
def test_same_direct_rule_matches_s4_on_the_patterned_contact_fixture(order, shift):
    pytest.importorskip('S4')
    shape = dict(fixture_shape(), cx=shift[0], cy=shift[1])
    np.testing.assert_allclose(solve_case(order, 64, 'wavebit-direct', shape),
                               s4_direct(order, shape), atol=3e-12, rtol=0)


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
    from wavebit._core import parametric as P
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


@pytest.mark.parametrize('incoming', [np.nan, 0.])
def test_invalid_external_flux_is_rejected_as_a_reference(monkeypatch, incoming):
    S4 = pytest.importorskip('S4')
    new = S4.New

    class InvalidFlux:
        def __init__(self, *args, **kwargs):
            self.simulation = new(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.simulation, name)

        def GetPowerFlux(self, **kwargs):
            return complex(incoming), 0j

    monkeypatch.setattr(S4, 'New', InvalidFlux)
    with pytest.raises(FloatingPointError, match='not a reference'):
        s4_direct(1)
