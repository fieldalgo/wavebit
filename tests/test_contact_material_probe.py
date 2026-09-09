"""Acceptance probes for the experimental field, not a default-NVF replacement."""
import numpy as np
import pytest
import wavebit
from wavebit._core import parametric as P
from wavebit._core import factorization as F
from wavebit._core.lattice import as_lattice
from contact_material_probe import material_contrast_field
from contact_instruments import paired_rectangles_li
import array_api_compat.numpy as xp

CELL = as_lattice(period=(700., 700.))


def boxes(gap, materials=(4., 9.)):
    return [dict(type='rectangle', cx=sign*(150.+gap/2), cy=0.,
                 lx=300., ly=300., angle=0., material=mat)
            for sign, mat in zip((-1, 1), materials)]


def solver(gap, materials=(4., 9.), backend=0):
    return wavebit.rcwa(
        backend=backend, period=(700., 700.), wavelength=900., top=1., bot=2.25,
        theta=.2, phi=0., psi=0., thickness=(300.,), truncation_order=(3, 3),
        normal_grid=128, structure=[dict(background=1., shapes=boxes(gap, materials))])


@pytest.mark.parametrize('gap', [0., 50.])
@pytest.mark.parametrize('materials', [(4., 9.), (4.+.2j, 4.+.7j)])
def test_pair_reference_matches_an_independent_commensurate_raster(gap, materials):
    x = (np.arange(28)+.5)*25.-350.
    X, Y = np.meshgrid(x, x)
    cell = np.ones(X.shape, dtype=complex)
    for shape in boxes(gap, materials):
        cell[(abs(X-shape['cx']) < 150.) & (abs(Y) < 150.)] = shape['material']
    want = F.to_conv_mat_raster_integrated(cell[None], 3, 3)
    got = paired_rectangles_li(xp, boxes(gap, materials), 1., 3, 3, CELL, xp.complex128)
    for a, b in zip(got[:3], want[:3]):
        np.testing.assert_allclose(a, b[0], rtol=0, atol=2e-12)


@pytest.mark.parametrize('materials', [(4., 9.), (4.+.2j, 4.+.7j)])
def test_candidate_contact_offset_shrinks_with_gap(monkeypatch, materials):
    monkeypatch.setattr(P, '_piece_field', material_contrast_field())
    def total(gap):
        P._PROJECTOR_CACHE.clear()
        return float(np.sum(solver(gap, materials).solve().T))
    try:
        contact = total(0.)
        large, small = total(1e-4)-contact, total(1e-6)-contact
        assert abs(small) < .02*abs(large)
        assert abs(small) < 2e-8
    finally:
        P._PROJECTOR_CACHE.clear()


@pytest.mark.parametrize('materials', [(4., 9.), (4.+.2j, 4.+.7j)])
def test_candidate_is_normal_on_seam_and_exposed_partial_edges(materials):
    shapes = boxes(0., materials)
    shapes[1]['ly'] = 280.
    # Internal coordinates: the seam is x=350; the shorter top is y=490.
    X = np.array([350., 350., 350., 350., 500., 200.])
    Y = np.array([250., 400., 485., 495., 490., 500.])
    nx, ny = material_contrast_field()(
        np, X, Y, shapes, P.plan_gamma(shapes, CELL), None, CELL, np.float64)
    np.testing.assert_allclose(nx*nx, [1., 1., 1., 1., 0., 0.], rtol=0, atol=2e-12)
    np.testing.assert_allclose(nx*ny, 0., rtol=0, atol=2e-12)


@pytest.mark.parametrize('lossy', [False, True])
def test_candidate_geometry_and_material_jit_gradients_match_secants(monkeypatch, lossy):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    monkeypatch.setattr(P, '_piece_field', material_contrast_field())
    gap = 1e-4
    material = 9. if not lossy else .7
    def materials(value):
        return (4., value) if not lossy else (4.+.2j, 4.+1j*value)
    frozen = solver(gap, materials(material)).freeze_geometry()
    def loss(g, m):
        return jnp.sum(solver(g, materials(m), backend=1).solve(geometry=frozen).T).real
    try:
        gradients = jax.jit(jax.grad(loss, argnums=(0, 1)))(jnp.asarray(gap), jnp.asarray(material))
        for index, steps in enumerate(((1e-5, 1e-6), (1e-4, 1e-5))):
            assert np.isfinite(gradients[index]) and abs(float(gradients[index])) > 1e-5
            for h in steps:
                args = [gap, material]
                args[index] += h
                P._PROJECTOR_CACHE.clear()
                plus = float(np.sum(solver(args[0], materials(args[1])).solve().T))
                args[index] -= 2*h
                P._PROJECTOR_CACHE.clear()
                minus = float(np.sum(solver(args[0], materials(args[1])).solve().T))
                assert float(gradients[index]) == pytest.approx(
                    (plus-minus)/(2*h), rel=3e-4, abs=5e-8)
    finally:
        P._PROJECTOR_CACHE.clear()


def test_background_identical_box_does_not_supply_a_phantom_interface():
    shapes = boxes(0., (1., 9.))
    # Include points on the invisible box's outer face, excluding corners/ties.
    X = np.array([50., 50., 100., 200., 270., 510.])
    Y = np.array([230., 470., 310., 275., 460., 360.])
    field = material_contrast_field()
    nx, ny = field(np, X, Y, shapes, P.plan_gamma(shapes, CELL), None, CELL, np.float64)
    active = shapes[1:]
    ax, ay = field(np, X, Y, active, P.plan_gamma(active, CELL), None, CELL, np.float64)
    np.testing.assert_allclose([nx*nx, nx*ny, ny*ny], [ax*ax, ax*ay, ay*ay],
                               rtol=0, atol=2e-12)
    empty = boxes(0., (1., 1.))
    nx, ny = field(np, X, Y, empty, P.plan_gamma(empty, CELL), None, CELL, np.float64)
    assert np.all(np.isfinite(nx)) and np.all(np.isfinite(ny))
    np.testing.assert_allclose(nx*nx+ny*ny, 1., rtol=0, atol=2e-12)


def test_candidate_projector_rotates_and_ignores_shape_order():
    materials = (4.+.2j, 4.+.7j)
    shapes = boxes(0., materials)
    shapes[1]['ly'] = 280.
    X, Y = np.meshgrid(np.linspace(39., 659., 13), np.linspace(43., 647., 11))
    field = material_contrast_field()
    def tensor(layer, lattice, x, y):
        nx, ny = field(np, x, y, layer, P.plan_gamma(layer, lattice), None,
                       lattice, np.float64)
        n = np.stack([nx, ny], axis=-1)
        return n[..., :, None]*n[..., None, :]
    baseline = tensor(shapes, CELL, X, Y)
    np.testing.assert_allclose(tensor(shapes[::-1], CELL, X, Y), baseline,
                               rtol=0, atol=2e-12)
    angle = .31
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    layer = []
    for shape in shapes:
        center = rotation @ [shape['cx'], shape['cy']]
        layer.append(dict(shape, cx=center[0], cy=center[1], angle=angle))
    lattice = as_lattice(lattice=rotation @ np.asarray(CELL))
    tx, ty = np.einsum('ij,j...->i...', rotation, np.stack([X, Y]))
    got = tensor(layer, lattice, tx, ty)
    want = np.einsum('ij,...jk,lk->...il', rotation, baseline, rotation)
    np.testing.assert_allclose(got, want, rtol=0, atol=2e-12)


@pytest.mark.parametrize('grid', [127, 256])
@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason='candidate fails the material-equality continuity gate')
def test_candidate_material_equality_limit_is_continuous(monkeypatch, grid):
    monkeypatch.setattr(P, '_piece_field', material_contrast_field())
    def total(delta):
        P._PROJECTOR_CACHE.clear()
        s = solver(0., (4., 4.+delta))
        s.normal_grid = grid
        return float(np.sum(s.solve().T))
    try:
        base = total(0.)
        large, small = total(1e-6)-base, total(1e-8)-base
        assert abs(small) < .05*abs(large)
    finally:
        P._PROJECTOR_CACHE.clear()


def test_candidate_three_neighbors_keep_each_material_seam_direction():
    from test_gamma_layers import ABC
    field = material_contrast_field()
    y = np.concatenate([np.linspace(11., 299., 41), np.linspace(-299., -11., 41),
                        np.linspace(-9., 9., 7)])
    for shapes in ABC.values():
        nx, ny = field(np, np.full_like(y, 350.), 350.+y, shapes,
                       P.plan_gamma(shapes, CELL), None, CELL, np.float64)
        np.testing.assert_allclose(nx[:41]**2, 0., rtol=0, atol=2e-12)
        np.testing.assert_allclose(nx[41:]**2, 1., rtol=0, atol=2e-12)
        np.testing.assert_allclose(nx*nx+ny*ny, 1., rtol=0, atol=2e-12)


def test_candidate_magnetic_slots_and_nonunit_background():
    from wavebit import Material
    layer = boxes(0., (Material(4., mu=2.), Material(4., mu=3.)))
    # The actual solver's mu view must feed the same field implementation.
    s = solver(0.)
    s.structure = [dict(background=Material(1., mu=2.), shapes=layer)]
    mu = s._mu_structure()[0]
    background = mu['background']
    background = getattr(background, 'epsilon', background)
    field = material_contrast_field(background)
    X, Y = np.array([350., 350., 650., 500.]), np.array([240., 460., 280., 500.])
    nx, ny = field(np, X, Y, mu['shapes'], P.plan_gamma(mu['shapes'], CELL),
                   None, CELL, np.float64)
    # Left box is invisible to mu; the right box retains its three tested faces.
    np.testing.assert_allclose(nx*nx, [1., 1., 1., 0.], rtol=0, atol=2e-12)
    np.testing.assert_allclose(nx*ny, 0., rtol=0, atol=2e-12)


def test_material_limit_has_a_slowly_shrinking_band_even_without_a_sampling_grid():
    """Keep the candidate's field-level failure distinct from quadrature error."""
    from contact_material_probe import material_transition_width
    deltas = np.array([1e-4, 1e-6, 1e-8])
    widths = np.array([material_transition_width(d) for d in deltas])
    # A 100x contrast reduction shrinks this band only about sqrt(10)-fold,
    # consistent with the inverse-fourth-distance weights. These are measured
    # field widths, not an asserted power law for the Maxwell transmission.
    np.testing.assert_allclose(widths[1:]/widths[:-1], .1**.5, rtol=2e-3, atol=0.)
    assert widths[-1] > .6
    assert material_transition_width(1e-6, scale=2.) == pytest.approx(
        2.*widths[1], rel=2e-11, abs=0.)
