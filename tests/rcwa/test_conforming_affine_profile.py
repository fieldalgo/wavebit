"""Fractional affine profiles: physical Fourier rules, finite closure and replay."""
import copy

import numpy as np
from tests.fixtures.common._conv_frame import slots_about_centre
import pytest

from wavebit.rcwa._engine.conforming.affine_profile import AffineProfileNVF

from wavebit.rcwa._engine.conforming import ConformingNVF
from wavebit.materials.response import Material
from wavebit.lattice import Lattice


SETTINGS = dict(rows=2, body_degree=2, gap_degree=2, grade_y=False)
BASIS = np.array([[700., 315.], [0., 490.]])


def layer(height=.91):
    return dict(background=1.7+.2j, profile=dict(width=.5, height=height,
                seams=[.04], values=[4.+.8j, 9.+.3j], offset=(17., -21.)))


def labels(order):
    return np.array([(m, n) for n in range(-order[1], order[1]+1)
                     for m in range(-order[0], order[0]+1)])


def physical_material(value):
    return value.epsilon if isinstance(value, Material) else value


def material_matrix(scene, basis, order, reciprocal=False):
    """Integrate each fractional rectangle directly, without a cell or projector."""
    p = scene['profile']
    modes = labels(order)
    dm, dn = (modes[:, None]-modes[None]).transpose(2, 0, 1)
    center = np.linalg.solve(basis, p.get('offset', (0., 0.)))
    background = physical_material(scene.get('background', 1.))
    background = 1/background if reciprocal else background
    result = background*np.eye(len(modes), dtype=complex)
    edges = [-p['width']/2, *p['seams'], p['width']/2]
    for left, right, value in zip(edges, edges[1:], p['values']):
        epsilon = physical_material(value)
        epsilon = 1/epsilon if reciprocal else epsilon
        span = right-left
        result += ((epsilon-background)*span*p['height']*np.sinc(dm*span)*np.sinc(dn*p['height'])
                   *np.exp(-2j*np.pi*(dm*(center[0]+(left+right)/2)+dn*center[1])))
    return result


def transverse(conv):
    return np.block([[conv.epx[0], -conv.epxy[0]], [-conv.epyx[0], conv.epy[0]]])


@pytest.mark.research
@pytest.mark.parametrize('height', [.91, 1.])
def test_diagonal_profile_matches_the_physical_rectangle_adapter(height):
    scene, basis, order = layer(height), np.diag([500., 700.]), (1, 1)
    p = scene['profile']
    edges = [-p['width']/2, *p['seams'], p['width']/2]
    shapes = [dict(type='rectangle', lx=basis[0, 0]*(right-left), ly=basis[1, 1]*height,
                   cx=p['offset'][0]+basis[0, 0]*(left+right)/2, cy=p['offset'][1], material=value)
              for left, right, value in zip(edges, edges[1:], p['values'])]
    expected = ConformingNVF(order, **SETTINGS).conv_matrices(
        [dict(background=scene['background'], shapes=shapes)], basis)
    actual = AffineProfileNVF(order, **SETTINGS).conv_matrices([scene], basis)
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want, atol=4e-10, rtol=0)


@pytest.mark.research
@pytest.mark.parametrize('order', [(2, 1), (1, 0), (0, 1), (0, 0)])
@pytest.mark.parametrize('basis', [BASIS, np.array([[700., 1400.], [0., 490.]])])
def test_original_labels_match_independent_physical_material_fourier(basis, order):
    scene = layer()
    scene['background'] = Material(scene['background'])
    scene['profile']['values'][0] = Material(scene['profile']['values'][0])
    conv = AffineProfileNVF(order, **SETTINGS).conv_matrices([scene], basis)
    expected = material_matrix(scene, basis, order)
    np.testing.assert_allclose(np.linalg.inv(conv.epz_i[0]), expected, atol=3e-12, rtol=0)
    count = len(labels(order))
    assert all(slot.shape == (1, count, count) and slot.dtype == np.complex128 for slot in conv)
    tensor = transverse(conv)
    assert np.linalg.eigvalsh((tensor+tensor.conj().T)/2)[0] > 0
    assert np.linalg.eigvalsh((tensor-tensor.conj().T)/(2j))[0] > -1e-11


@pytest.mark.research
@pytest.mark.parametrize('order', [(2, 1), (0, 1), (1, 0)])
def test_sheared_contact_obeys_reciprocal_normal_inverse_rule(order):
    scene = layer(1.)
    conv = AffineProfileNVF(order, **SETTINGS).conv_matrices([scene], BASIS)
    tangent = material_matrix(scene, BASIS, order)
    harmonic = np.linalg.inv(material_matrix(scene, BASIS, order, reciprocal=True))
    normal = np.linalg.solve(BASIS.T, [1., 0.])
    normal /= np.linalg.norm(normal)
    nn = np.outer(normal, normal)
    expected = np.kron(nn, harmonic)+np.kron(np.eye(2)-nn, tangent)
    np.testing.assert_allclose(transverse(conv), expected, atol=4e-11, rtol=0)
    np.testing.assert_allclose(conv.epz_i[0], np.linalg.inv(tangent), atol=3e-12, rtol=0)


@pytest.mark.research
@pytest.mark.parametrize('height', [1., np.nextafter(1., 2.)])
def test_single_material_lossless_contact_accepts_only_roundoff_extension(height):
    scene = dict(profile=dict(width=.5, height=height, seams=[], values=[4.]))
    order = (1, 0)
    conv = AffineProfileNVF(order, **SETTINGS).conv_matrices([scene], BASIS)
    np.testing.assert_allclose(np.linalg.inv(conv.epz_i[0]), material_matrix(scene, BASIS, order),
                               atol=3e-12, rtol=0)
    tensor = transverse(conv)
    np.testing.assert_allclose(tensor, tensor.conj().T, atol=3e-12, rtol=0)
    assert np.linalg.eigvalsh(tensor)[0] > 0


@pytest.mark.research
def test_unequal_orders_are_selected_before_the_energy_inverses(monkeypatch):
    from wavebit.rcwa._engine.factorization import ConvMatrices

    order, scene, captured = (1, 0), layer(), []
    close = ConvMatrices.from_energy

    def capture(cls, change, gram, longitudinal):
        captured.append((change, gram, longitudinal))
        return close(change, gram, longitudinal)

    monkeypatch.setattr(ConvMatrices, 'from_energy', classmethod(capture))
    AffineProfileNVF(1, **SETTINGS).conv_matrices([scene], BASIS)
    full = labels((1, 1))
    index = np.array([np.flatnonzero(np.all(full == k, axis=1))[0] for k in labels(order)])
    both = np.r_[index, index+len(full)]
    q, g, z = captured[0]
    expected = slots_about_centre(
        close(q[np.ix_(both, both)], g[np.ix_(both, both)], z[np.ix_(index, index)]), order)
    actual = AffineProfileNVF(order, **SETTINGS).conv_matrices([scene], BASIS)
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got[0], want, atol=3e-12, rtol=0)
    assert captured[-1][0].shape == (6, 6)


@pytest.mark.research
def test_translation_rotation_common_scale_and_uniform_layers():
    order, scene = (1, 1), layer()
    builder = AffineProfileNVF(order, **SETTINGS)
    baseline = builder.conv_matrices([scene], BASIS)
    shift = np.array([913., -487.])
    moved = copy.deepcopy(scene)
    moved['profile']['offset'] = np.asarray(scene['profile']['offset'])+shift
    actual = builder.conv_matrices([moved], BASIS)
    phase = np.exp(-2j*np.pi*(labels(order)@np.linalg.solve(BASIS, shift)))
    for got, want in zip(actual, baseline):
        np.testing.assert_allclose(got[0], phase[:, None]*want[0]*phase.conj()[None], atol=4e-11, rtol=0)
    angle = .37
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = copy.deepcopy(scene)
    rotated['profile']['offset'] = rotation@scene['profile']['offset']
    actual = builder.conv_matrices([rotated], rotation@BASIS)
    rotate = np.kron(rotation, np.eye(len(labels(order))))
    np.testing.assert_allclose(transverse(actual), rotate@transverse(baseline)@rotate.T, atol=5e-11, rtol=0)
    np.testing.assert_allclose(actual.epz_i, baseline.epz_i, atol=3e-12, rtol=0)
    scaled = copy.deepcopy(scene)
    scaled['profile']['offset'] = .004*np.asarray(scene['profile']['offset'])
    actual = builder.conv_matrices([scaled, dict(background=2.3+.4j), dict(profile=None)], .004*BASIS)
    for got, want in zip(actual, baseline):
        np.testing.assert_allclose(got[0], want[0], atol=4e-11, rtol=0)
    identity = np.eye(len(labels(order)))
    for index, epsilon in ((1, 2.3+.4j), (2, 1.)):
        for got, value in zip(actual, (epsilon, epsilon, 1/epsilon, 0., 0.)):
            np.testing.assert_allclose(got[index], value*identity, atol=1e-15, rtol=0)


@pytest.mark.research
def test_lattice_object_and_coincident_seams_preserve_the_full_field():
    scene, order = layer(), (1, 0)
    builder = AffineProfileNVF(order, **SETTINGS)
    expected = builder.conv_matrices([scene], BASIS)
    scene['profile']['seams'] = [.04, .04]
    scene['profile']['values'].insert(1, 2.+3j)
    actual = builder.conv_matrices([scene], Lattice(BASIS[:, 0], BASIS[:, 1], axis_aligned=False))
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want, atol=4e-11, rtol=0)


@pytest.mark.parametrize('order', [-1, True, 1.2, (1, -1), (True, 1), (1,), '1'])
def test_invalid_orders_are_refused(order):
    with pytest.raises(ValueError, match='truncation_order'):
        AffineProfileNVF(order)


@pytest.mark.parametrize('basis', [np.zeros((2, 2)), np.diag([1., -1.]), [[1., 0.], [0., np.nan]],
                                   [[1., 0.], [0., np.inf]], [[1., 1j], [0., 1.]], [1., 1.]])
def test_invalid_lattices_are_refused_even_for_uniform_layers(basis):
    with pytest.raises(ValueError, match='lattice'):
        AffineProfileNVF(0, **SETTINGS).conv_matrices([{}], basis)


@pytest.mark.parametrize('change', [dict(width=0.), dict(width=1.), dict(width=1j), dict(height=0.),
                                    dict(height=1.+1e-8), dict(seams=[.1, -.1], values=[4., 5., 9.]),
                                    dict(seams=[.3]), dict(seams=[[.04]]), dict(values=[4.]),
                                    dict(values=[4., -1.]), dict(values=[4., 9.-.1j]),
                                    dict(values=[4., np.inf]), dict(offset=[1.]), dict(offset=[0., np.nan]),
                                    dict(offset=[0., 1j]), dict(angle=0.)])
def test_invalid_profiles_are_refused_before_cell_solution(monkeypatch, change):
    builder, scene = AffineProfileNVF(0, **SETTINGS), layer()
    scene['profile'].update(change)
    monkeypatch.setattr(builder._cell, 'coefficients', lambda *a, **k: pytest.fail('invalid profile reached cell'))
    with pytest.raises(ValueError):
        builder.conv_matrices([scene], BASIS)


def test_unknown_or_missing_recipe_keys_and_nonphysical_materials_are_refused():
    builder = AffineProfileNVF(0, **SETTINGS)
    for structure in ([], {}, [None], [dict(shapes=[])], [dict(profile={})], [dict(profile=[])],
                      [dict(background=-1.)], [dict(background=1.-.1j)]):
        with pytest.raises(ValueError):
            builder.conv_matrices(structure, BASIS)
    with pytest.raises(NotImplementedError, match='permeability'):
        builder.conv_matrices([dict(background=Material(1., mu=1.))], BASIS)
    with pytest.raises(ValueError, match='scalar'):
        builder.conv_matrices([dict(background=np.eye(2))], BASIS)
    with pytest.raises(NotImplementedError, match='NumPy and JAX'):
        builder.conv_matrices([{}], BASIS, xp=object())


def live_scene(p, xp):
    basis = xp.reshape(p[5:9], (2, 2))
    scene = dict(background=p[13]+1j*p[14], profile=dict(width=p[0], height=p[1], seams=p[2:3],
                 offset=p[3:5], values=[p[9]+1j*p[10], p[11]+1j*p[12]]))
    return [scene], basis


@pytest.mark.research
def test_jit_preserves_live_geometry_material_and_affine_basis_values():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    builder = AffineProfileNVF((1, 0), **SETTINGS)
    parameters = np.array([.5, .91, .04, 17., -21., 700., 315., 0., 490., 4., .8, 9., .3, 1.7, .2])
    direction = np.array([.03, -.01, .02, 20., -8., 50., -13., 11., 17., .4, .1, .3, .2, .1, .03])
    weights = np.random.default_rng(890).normal(size=(5, 1, 3, 3))

    def objective(p, xp):
        c = xp.stack(builder.conv_matrices(*live_scene(p, xp), xp=xp))
        return xp.sum(xp.asarray(weights)*(c.real+.37*c.imag))

    value, tangent = jax.jit(lambda p, d: jax.jvp(lambda x: objective(x, jnp), (p,), (d,)))(
        jnp.asarray(parameters), jnp.asarray(direction))
    assert value == pytest.approx(objective(parameters, np), abs=4e-10)
    for step in (2e-4, 1e-4):
        difference = (objective(parameters+step*direction, np)-objective(parameters-step*direction, np))/(2*step)
        assert tangent == pytest.approx(difference, abs=3e-6, rel=3e-6)
    build = jax.jit(lambda p: builder.conv_matrices(*live_scene(p, jnp), xp=jnp))
    for index, bad in ((0, 0.), (1, 1.01), (2, .4), (3, np.inf), (5, 0.), (10, -.1), (13, 0.)):
        invalid = parameters.copy()
        invalid[index] = bad
        assert all(np.isnan(np.asarray(slot)).all() for slot in build(jnp.asarray(invalid)))
    uniform = jax.jit(lambda basis: builder.conv_matrices([{}, dict(background=2.)], basis, xp=jnp))
    assert all(np.isnan(np.asarray(slot)).all() for slot in uniform(jnp.zeros((2, 2))))
    jax.clear_caches()


@pytest.mark.research
def test_coincident_seams_keep_the_independent_material_opening_derivative():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    builder, scene, order = AffineProfileNVF((1, 1), **SETTINGS), layer(), (1, 1)
    p = scene['profile']
    middle = scene['background']

    def material(opening):
        profile = dict(p, seams=jnp.stack([.04-opening/2, .04+opening/2]),
                       values=[p['values'][0], middle, p['values'][1]])
        conv = builder.conv_matrices([dict(scene, profile=profile)], jnp.asarray(BASIS), xp=jnp)
        return jnp.linalg.inv(conv.epz_i[0])

    value, derivative = jax.jit(lambda g: jax.jvp(material, (g,), (jnp.ones_like(g),)))(jnp.asarray(0.))
    np.testing.assert_allclose(value, material_matrix(scene, BASIS, order), atol=3e-12, rtol=0)
    dm, dn = (labels(order)[:, None]-labels(order)[None]).transpose(2, 0, 1)
    center = np.linalg.solve(BASIS, p['offset'])
    expected = ((middle-(p['values'][0]+p['values'][1])/2)*p['height']*np.sinc(dn*p['height'])
                *np.exp(-2j*np.pi*(dm*(center[0]+.04)+dn*center[1])))
    np.testing.assert_allclose(derivative, expected, atol=2e-8, rtol=0)
    jax.clear_caches()
