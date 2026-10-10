"""Compiled optimization steps release temporary solver inputs and field state."""
from functools import partial

import numpy as np
import pytest

import wavebit
from wavebit._core.engine import _holds_tracer

jax = pytest.importorskip('jax')
import jax.numpy as jnp
optax = pytest.importorskip('optax')


def solver(backend='jax', **kwargs):
    cfg = dict(period=(600., 750.), structure=np.full((1, 2, 2), 4.),
               wavelength=900., thickness=(170.,), truncation_order=(0, 0),
               theta=.13, phi=.21, answers='field', factorization='li')
    return wavebit.rcwa(backend=backend, **(cfg | kwargs))


@pytest.mark.timeout(30)
@pytest.mark.parametrize('door', ['solve', 'solve_field', 'solve_field_eager'])
def test_fit_releases_state_before_jax_checks_leaks(door):
    s = solver()
    forward = getattr(s, door)
    if door != 'solve':
        forward = partial(forward, 2, 2, 2)
    def loss(result):
        if door != 'solve':
            result, _ = result
        return result.T.sum().real
    with jax.checking_leaks():
        final = s.fit(['thickness'], forward, loss, optax.sgd(.01), iteration=1)
    assert not any(_holds_tracer(v) for v in vars(s).values())
    np.testing.assert_array_equal(s.thickness, final['thickness'])
    assert s.T1 is None and s.layer_info_list == []


@pytest.mark.parametrize('failure', ['forward', 'loss'])
def test_failed_custom_assignment_restores_the_prepared_scene(failure):
    s = solver()
    saved = s.solve()
    original = np.asarray(s.thickness).copy()
    def forward(thickness, geometry):
        s.thickness = thickness
        if failure == 'forward':
            raise RuntimeError('intentional custom failure')
        return s.solve(geometry=geometry)
    def loss(result):
        raise RuntimeError('intentional custom failure')
    with pytest.raises(RuntimeError, match='intentional custom failure'):
        s.fit(['thickness'], forward, loss, optax.sgd(.01),
              iteration=1, prepare_geometry=s.freeze_geometry)
    assert not any(_holds_tracer(v) for v in vars(s).values())
    np.testing.assert_array_equal(s.thickness, original)
    np.testing.assert_allclose(s.solve().T, saved.T, atol=1e-12)


def design(width):
    return [dict(background=1., shapes=[dict(type='rectangle', cx=0., cy=0.,
                                           lx=width, ly=100., angle=0., material=4.)])]


@pytest.mark.parametrize('failure', ['forward', 'loss', None])
def test_custom_design_writes_restore_the_complete_raster_declaration(failure):
    cell = wavebit.Lattice.rectangular(600., 750.)
    raster = np.full((1, 2, 2), 4.)
    s = solver(structure=cell.drawn(raster))
    def prepare(width):
        return solver(structure=design(width)).freeze_geometry()
    def forward(width, geometry):
        s.structure = design(width)
        if failure == 'forward':
            raise RuntimeError('intentional custom design failure')
        return s.solve(geometry=geometry)
    def loss(result):
        if failure == 'loss':
            raise RuntimeError('intentional custom design failure')
        return result.T.sum().real
    params = dict(width=jnp.asarray(100.))
    if failure:
        with pytest.raises(RuntimeError, match='intentional custom design failure'):
            s.fit(params, forward, loss, optax.sgd(10.), iteration=1, prepare_geometry=prepare)
    else:
        final = s.fit(params, forward, loss, optax.sgd(10.), iteration=1, prepare_geometry=prepare)
        h = 1e-3
        fd = (solver('numpy', structure=design(100.+h)).solve().T.sum().real
              - solver('numpy', structure=design(100.-h)).solve().T.sum().real)/(2*h)
        assert float(final['width']) == pytest.approx(100.-10.*fd, abs=1e-8)
    assert not any(_holds_tracer(v) for v in vars(s).values())
    assert not hasattr(s, 'width')
    np.testing.assert_array_equal(s.structure, raster)
    np.testing.assert_array_equal(s._structure_drawn_on, np.asarray(cell))
    np.testing.assert_allclose(s.solve().T, solver('numpy').solve().T, atol=1e-11)
    with pytest.raises(ValueError, match='drawn|cell|lattice'):
        s.solve(period=(610., 750.))


def test_in_place_shape_edits_only_touch_the_temporary_design():
    s = solver(structure=design(100.))
    original = s.structure
    def prepare(width):
        return solver(structure=design(width)).freeze_geometry()
    def forward(width, geometry):
        s.structure[0]['shapes'][0]['lx'] = width
        return s.solve(geometry=geometry)
    s.fit(dict(width=jnp.asarray(100.)), forward, lambda result: result.T.sum().real,
          optax.sgd(10.), iteration=1, prepare_geometry=prepare)
    assert not any(_holds_tracer(v) for v in vars(s).values())
    assert s.structure is original
    assert original[0]['shapes'][0]['lx'] == 100.
    np.testing.assert_allclose(s.solve().T, solver('numpy', structure=design(100.)).solve().T, atol=1e-11)
