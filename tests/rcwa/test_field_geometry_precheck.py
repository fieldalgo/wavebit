"""Validate concrete plans before solve_field hides geometry behind a JAX trace.

These are field-entry regressions: eager builders already rejected the stale plans.
The matching valid controls compare complete fields to independently prepared NumPy
solves, so rejecting every shape update cannot satisfy the contract.
"""

from wavebit.rcwa._engine.geometry import StaleGeometryError as _legacy_wavebit_StaleGeometryError

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa
import numpy as np
import pytest

import wavebit

jax = pytest.importorskip('jax')
import jax.numpy as jnp


CASES = [(False, 'li'), (False, 'nvf'), (False, 'laurent'),
         (True, 'li'), (True, 'nvf')]
CFG = dict(period=(300., 300.), thickness=(205.,), truncation_order=(1, 1),
           top=1., bot=1., theta=.15, phi=.35, polarization=(.8, .6),
           wavelength=900., answers='field')


def scene(width=1.18, tensor=False, epsilon=9.):
    def material(e):
        return wavebit.Material([e, e+.2, e+.4]) if tensor else e
    return [dict(background=material(12.25), shapes=[
        dict(type='rectangle', cx=-90., cy=0., lx=60., ly=100., material=material(4.)),
        dict(type='rectangle', cx=0., cy=0., lx=100.*width, ly=100., material=material(epsilon)),
        dict(type='rectangle', cx=90., cy=0., lx=60., ly=100., material=material(6.25))])]


def solver(tensor=False, rule='nvf', backend='jax', width=1.18, **kwargs):
    cfg = dict(CFG, factorization=rule, structure=scene(width, tensor))
    if rule == 'nvf':
        cfg['shape_nvf_grid_size'] = 16
    cfg.update(kwargs)
    return _legacy_wavebit_rcwa(backend=backend, **cfg)


def no_compiled_solve(*args, **kwargs):
    pytest.fail('stale concrete geometry reached the compiled solve')


@pytest.mark.parametrize('tensor,rule', CASES)
@pytest.mark.parametrize('change', ['overlap', 'shape_count'])
def test_shape_changes_are_refused_before_compilation(monkeypatch, tensor, rule, change):
    s = solver(tensor, rule, width=1.3 if change == 'shape_count' else 1.18)
    frozen = s.freeze_geometry()
    moved = scene(1.3, tensor)
    if change == 'shape_count':
        moved[0]['shapes'].pop()
    # A successful eager refusal is the contract the compiled entry must retain.
    with pytest.raises(_legacy_wavebit_StaleGeometryError):
        solver(tensor, rule, structure=moved).solve(geometry=frozen)
    monkeypatch.setattr(type(s), '_solve_field_compiled', no_compiled_solve)
    with pytest.raises(_legacy_wavebit_StaleGeometryError):
        s.solve_field(2, 2, 2, geometry=frozen, structure=moved)


@pytest.mark.parametrize('tensor,rule', CASES)
@pytest.mark.parametrize('change', ['width', 'material'])
def test_valid_shape_changes_match_fresh_numpy_fields(tensor, rule, change):
    s = solver(tensor, rule)
    frozen = s.freeze_geometry()
    moved = scene(1.19 if change == 'width' else 1.18, tensor,
                  epsilon=9.4 if change == 'material' else 9.)
    got, field = s.solve_field(3, 3, 2, geometry=frozen, structure=moved)
    want, reference = solver(tensor, rule, backend='numpy', structure=moved).solve_field(3, 3, 2)
    np.testing.assert_allclose(got.T, want.T, rtol=2e-9, atol=2e-11)
    np.testing.assert_allclose(field, reference, rtol=2e-8, atol=2e-9)


@pytest.mark.parametrize('tensor,rule', CASES)
def test_material_trace_does_not_hide_concrete_shape_topology(monkeypatch, tensor, rule):
    s = solver(tensor, rule)
    frozen = s.freeze_geometry()
    monkeypatch.setattr(type(s), '_solve_field_compiled', no_compiled_solve)
    def loss(epsilon):
        return s.solve_field(2, 2, 2, geometry=frozen,
                             structure=scene(1.3, tensor, epsilon))[0].T.sum().real
    with pytest.raises(_legacy_wavebit_StaleGeometryError):
        jax.grad(loss)(jnp.asarray(9.))
    assert not any(isinstance(v, jax.core.Tracer) for v in jax.tree_util.tree_leaves(s))


@pytest.mark.parametrize('rule', ['li', 'nvf'])
@pytest.mark.parametrize('parameter', ['width', 'material'])
def test_tensor_shape_field_gradients_match_fresh_numpy_differences(rule, parameter):
    s = solver(True, rule)
    frozen = s.freeze_geometry()
    def structure(value):
        return scene(value if parameter == 'width' else 1.18, True,
                     value if parameter == 'material' else 9.)
    def loss(value):
        _, field = s.solve_field(3, 3, 2, geometry=frozen, structure=structure(value))
        return jnp.sum(jnp.abs(field)**2).real
    def reference(value):
        _, field = solver(True, rule, backend='numpy', structure=structure(value)).solve_field(3, 3, 2)
        return np.sum(np.abs(field)**2).real
    value = 1.18 if parameter == 'width' else 9.
    derivative = jax.jit(jax.grad(loss))(jnp.asarray(value))
    h = 1e-5
    expected = (reference(value+h)-reference(value-h))/(2*h)
    assert abs(expected) > 1e-5
    np.testing.assert_allclose(derivative, expected, rtol=3e-5, atol=2e-6)
    assert not any(isinstance(v, jax.core.Tracer) for v in jax.tree_util.tree_leaves(s))


def tensor_raster():
    t = np.broadcast_to(np.diag([4., 4.2, 3.8]), (1, 4, 4, 3, 3)).copy()
    t[0, :, 2:] = np.diag([6., 6.1, 5.6])
    return t


@pytest.mark.parametrize('change', ['epsilon', 'mu', 'lattice', 'orders'])
@pytest.mark.parametrize('uniform', [False, True])
def test_tensor_raster_records_are_checked_before_compilation(monkeypatch, change, uniform):
    eps = tensor_raster()
    if uniform:
        eps[:] = eps[0, 0, 0]
    mu = eps * .1 + np.eye(3)
    s = _legacy_wavebit_rcwa(backend='jax', **CFG, structure=eps, mu=mu)
    frozen = s.freeze_geometry()
    moved = eps.copy()
    moved[0, 1, 1] = np.diag([7., 7.2, 6.8])
    options = {'epsilon': dict(structure=moved), 'mu': dict(mu=moved*.1+np.eye(3)),
               'lattice': dict(lattice=((300., 30.), (0., 300.))),
               'orders': dict(truncation_order=(2, 1))}[change]
    monkeypatch.setattr(type(s), '_solve_field_compiled', no_compiled_solve)
    with pytest.raises(_legacy_wavebit_StaleGeometryError):
        s.solve_field(2, 2, 2, geometry=frozen, **options)


@pytest.mark.parametrize('quantity', ['structure', 'mu'])
def test_tensor_raster_material_sweep_keeps_interfaces_and_matches_numpy(quantity):
    eps = tensor_raster()
    mu = eps * .1 + np.eye(3)
    s = _legacy_wavebit_rcwa(backend='jax', **CFG, structure=eps, mu=mu)
    frozen = s.freeze_geometry()
    # Swapping phase values also swaps np.unique labels, but not the interfaces.
    original = eps if quantity == 'structure' else mu
    moved = original[:, :, ::-1].copy()
    cfg = dict(CFG, structure=eps, mu=mu)
    cfg[quantity] = moved
    got, field = s.solve_field(3, 3, 2, geometry=frozen, **{quantity: moved})
    want, reference = _legacy_wavebit_rcwa(backend='numpy', **cfg).solve_field(3, 3, 2)
    np.testing.assert_allclose(got.T, want.T, rtol=2e-9, atol=2e-11)
    np.testing.assert_allclose(field, reference, rtol=2e-8, atol=2e-9)


def test_laurent_disjoint_shape_count_changes_need_no_new_plan():
    s = solver(rule='laurent')
    frozen = s.freeze_geometry()
    moved = scene()
    moved[0]['shapes'].pop()
    got, field = s.solve_field(3, 3, 2, geometry=frozen, structure=moved)
    want, reference = solver(rule='laurent', backend='numpy', structure=moved).solve_field(3, 3, 2)
    np.testing.assert_allclose(got.T, want.T, rtol=2e-9, atol=2e-11)
    np.testing.assert_allclose(field, reference, rtol=2e-8, atol=2e-9)


def test_a_refused_concrete_change_drops_the_previous_eager_state():
    s = solver()
    frozen = s.freeze_geometry()
    s.solve()
    with pytest.raises(_legacy_wavebit_StaleGeometryError):
        s.solve_field(2, 2, 2, geometry=frozen, structure=scene(1.3))
    with pytest.raises(RuntimeError, match='solve'):
        s.reconstruct_field(2, 2, 2)
    # The concrete keyword remains assigned, as for an eager solve failure.
    assert s.structure[0]['shapes'][1]['lx'] == 130.


@pytest.mark.parametrize('tensor,rule', CASES)
def test_precheck_does_not_assemble_fourier_operators(monkeypatch, tensor, rule):
    from wavebit.rcwa._engine import anisotropy, rectilinear, parametric
    s = solver(tensor, rule)
    frozen = s.freeze_geometry()
    def forbidden(*args, **kwargs):
        pytest.fail('validation assembled Fourier operators')
    monkeypatch.setattr(anisotropy, 'build', forbidden)
    monkeypatch.setattr(rectilinear, 'layer_matrices', forbidden)
    monkeypatch.setattr(parametric, 'to_conv_mat_parametric_layers', forbidden)
    token = object()
    monkeypatch.setattr(type(s), '_solve_field_compiled', lambda *a, **kw: token)
    assert s.solve_field(2, 2, 2, geometry=frozen) is token


@pytest.mark.parametrize('tensor', [False, True])
@pytest.mark.parametrize('moving', ['structure', 'mu'])
@pytest.mark.parametrize('transform', ['grad', 'jit'])
def test_a_traced_material_does_not_hide_the_other_concrete_interfaces(
        monkeypatch, tensor, moving, transform):
    eps = tensor_raster() if tensor else tensor_raster()[..., 0, 0]
    mu = eps * .1 + (np.eye(3) if tensor else 1.)
    s = _legacy_wavebit_rcwa(backend='jax', **CFG, structure=eps, mu=mu, factorization='nvf')
    frozen = s.freeze_geometry()
    base = eps if moving == 'structure' else mu
    moved = base.copy()
    moved[0, 1, 1] = base[0, 1, 2]
    setattr(s, moving, moved)
    differentiated = 'mu' if moving == 'structure' else 'structure'
    live = getattr(s, differentiated)
    monkeypatch.setattr(type(s), '_solve_field_compiled', no_compiled_solve)
    def loss(scale):
        result, _ = s.solve_field(2, 2, 2, geometry=frozen, **{differentiated: live*scale})
        return result.T.sum().real
    with pytest.raises(_legacy_wavebit_StaleGeometryError):
        getattr(jax, transform)(loss)(jnp.asarray(1.))


@pytest.mark.parametrize('tensor', [False, True])
@pytest.mark.parametrize('quantity', ['structure', 'mu'])
def test_valid_traced_raster_material_updates_still_have_correct_gradients(tensor, quantity):
    eps = tensor_raster() if tensor else tensor_raster()[..., 0, 0]
    mu = eps * .1 + (np.eye(3) if tensor else 1.)
    cfg = dict(CFG, structure=eps, mu=mu, factorization='nvf')
    s = _legacy_wavebit_rcwa(backend='jax', **cfg)
    frozen = s.freeze_geometry()
    live = getattr(s, quantity)
    def loss(scale):
        result, _ = s.solve_field(2, 2, 2, geometry=frozen, **{quantity: live*scale})
        return result.T.sum().real
    def reference(scale):
        return _legacy_wavebit_rcwa(backend='numpy', **(cfg | {quantity: np.asarray(live)*scale})).solve().T.sum().real
    got = jax.jit(jax.grad(loss))(jnp.asarray(1.))
    h = 1e-5
    expected = (reference(1.+h)-reference(1.-h))/(2*h)
    assert abs(expected) > 1e-6
    np.testing.assert_allclose(got, expected, rtol=2e-5, atol=2e-8)
