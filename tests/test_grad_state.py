"""A gradient query must preserve declarations and release expired JAX solve state."""
from functools import partial

import numpy as np
import pytest

import wavebit

jax = pytest.importorskip('jax')
import jax.numpy as jnp

CFG = dict(period=(600., 600.), thickness=(170.,), truncation_order=(1, 1),
           theta=.13, phi=.21, wavelength=900., answers='field')
RASTER = np.ones((1, 4, 4))
RASTER[0, 1:3, 1:3] = 4.


def solver(backend=1, **kwargs):
    return wavebit.rcwa(backend=backend, **(dict(CFG, structure=RASTER) | kwargs))


@pytest.mark.parametrize('door', ['solve', 'solve_field_eager', 'solve_field'])
@pytest.mark.parametrize('fails', [False, True])
def test_gradient_state_is_released_even_if_the_loss_raises(door, fails):
    from wavebit._core.engine import _holds_tracer
    s = solver()
    geometry = s.freeze_geometry()
    saved = s.solve()
    original = np.asarray(saved.reconstruct_field(2, 2, 2))
    forward = partial(getattr(s, door), geometry=geometry)
    if door != 'solve':
        forward = partial(forward, 2, 2, 2)
    def loss(result):
        if fails:
            raise RuntimeError('intentional loss failure')
        if door != 'solve':
            result, _ = result
        return result.T.sum().real
    if fails:
        with pytest.raises(RuntimeError, match='intentional loss failure'):
            s.grad(['wavelength'], forward, loss)
    else:
        got = s.grad(['wavelength'], forward, loss)['wavelength']
        h = 1e-3
        def reference(w):
            return solver(0, wavelength=w).solve().T.sum().real
        expected = (reference(900.+h)-reference(900.-h))/(2*h)
        assert abs(expected) > 1e-6
        np.testing.assert_allclose(got, expected, rtol=2e-6, atol=1e-9)
    assert not any(_holds_tracer(v) for v in vars(s).values())
    assert s.T1 is None and s._scattering is None and s.layer_info_list == []
    with pytest.raises(RuntimeError, match='solve'):
        s.reconstruct_field(2, 2, 2)
    np.testing.assert_allclose(saved.reconstruct_field(2, 2, 2), original)
    np.testing.assert_allclose(s.solve().T, saved.T, atol=1e-12)


def test_default_polarization_remains_default_after_its_gradient():
    s = solver()
    assert not s._psi_given
    g = s.freeze_geometry()
    s.grad(['psi'], partial(s.solve, geometry=g), lambda r: r.T.sum().real)
    assert not s._psi_given
    s.polarization = (.8, .6)
    np.testing.assert_allclose(s.solve().T, solver(0, polarization=(.8, .6)).solve().T,
                               atol=2e-11)


@pytest.mark.parametrize('slot', ['top', 'bot', 'structure'])
@pytest.mark.parametrize('fails', [False, True])
def test_gradient_restores_the_material_definition_beside_its_value(slot, fails):
    material = wavebit.TabulatedMaterial([800., 1000.], [2.25, 4.])
    value = [dict(background=material, shapes=[])] if slot == 'structure' else material
    s = solver(**{slot: value})
    g = s.freeze_geometry()
    def loss(result):
        if fails:
            raise RuntimeError('intentional loss failure')
        return result.T.sum().real
    if fails:
        with pytest.raises(RuntimeError, match='intentional loss failure'):
            s.grad([slot], partial(s.solve, geometry=g), loss)
    else:
        gradient = s.grad([slot], partial(s.solve, geometry=g), loss)
        assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree_util.tree_leaves(gradient))
    s.wavelength = 1000.
    evaluated = s.structure[0]['background'] if slot == 'structure' else getattr(s, slot)
    assert complex(evaluated.epsilon) == pytest.approx(4.)
    np.testing.assert_allclose(s.solve().T, solver(0, **{slot: value}, wavelength=1000.).solve().T,
                               rtol=2e-9, atol=2e-11)


def test_gradient_preserves_the_drawn_cell_declaration():
    cell = wavebit.Lattice.rectangular(600., 600.)
    s = solver(structure=cell.drawn(RASTER))
    g = s.freeze_geometry()
    s.grad(['structure'], partial(s.solve, geometry=g), lambda r: r.T.sum().real)
    assert s._structure_drawn_on is not None
    with pytest.raises(ValueError, match='drawn|cell|lattice'):
        s.solve(lattice=((610., 0.), (0., 600.)))


@pytest.mark.parametrize('custom', [False, True])
def test_state_is_released_before_jax_checks_for_escaped_tracers(custom):
    s = solver()
    g = s.freeze_geometry()
    if custom:
        def forward(**conditions):
            # A custom forward may assign directly rather than use solve's keyword scope.
            for name, value in conditions.items():
                setattr(s, name, value)
            return s.solve(geometry=g)
    else:
        forward = partial(s.solve, geometry=g)
    original = np.asarray(s.thickness).copy()
    with jax.checking_leaks():
        gradient = s.grad(['thickness'], forward, lambda r: r.T.sum().real)
    assert np.isfinite(np.asarray(gradient['thickness'])).all()
    np.testing.assert_array_equal(s.thickness, original)


def test_field_loss_can_read_the_solver_while_the_gradient_is_being_evaluated():
    s = solver()
    g = s.freeze_geometry()
    def loss(_):
        return jnp.sum(jnp.abs(s.reconstruct_field(2, 2, 2))**2).real
    got = s.grad(['wavelength'], partial(s.solve, geometry=g), loss)['wavelength']
    def reference(w):
        reference_solver = solver(0, wavelength=w)
        reference_solver.solve()
        return np.sum(np.abs(reference_solver.reconstruct_field(2, 2, 2))**2).real
    h = 1e-3
    expected = (reference(900.+h)-reference(900.-h))/(2*h)
    np.testing.assert_allclose(got, expected, rtol=2e-6, atol=1e-8)
