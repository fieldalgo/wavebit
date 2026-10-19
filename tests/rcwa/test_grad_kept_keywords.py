"""Gradient cleanup preserves explicit concrete keywords without leaking custom writes."""

from wavebit.materials._tabulated import TabulatedMaterial as _legacy_wavebit_TabulatedMaterial

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa
from functools import partial

import numpy as np
import pytest

import wavebit
from wavebit.rcwa._engine.engine import _holds_tracer

jax = pytest.importorskip('jax')

CFG = dict(period=(600., 750.), structure=np.full((1, 2, 2), 4.),
           thickness=(170.,), wavelength=900., truncation_order=(0, 0),
           theta=.13, phi=.21, answers='field', factorization='li')


def solver(backend='jax', **kwargs):
    return _legacy_wavebit_rcwa(backend=backend, **(CFG | kwargs))


@pytest.mark.parametrize('door', ['solve', 'solve_field', 'solve_field_eager'])
@pytest.mark.parametrize('wrapper', [False, True])
@pytest.mark.parametrize('failure', [False, True])
def test_concrete_forward_keywords_survive_the_gradient_scope(door, wrapper, failure):
    s = solver()
    geometry = s.freeze_geometry()
    method = getattr(s, door)
    if door != 'solve':
        method = partial(method, 2, 2, 2)
    fixed = dict(wavelength=850., theta=.23)
    forward = partial(method, geometry=geometry, **fixed)
    if wrapper:
        def forward(**values):
            return method(geometry=geometry, **fixed, **values)
    def loss(result):
        if failure:
            raise RuntimeError('intentional kept-keyword loss failure')
        return (result if door == 'solve' else result[0]).T.sum().real
    with jax.checking_leaks():
        if failure:
            with pytest.raises(RuntimeError, match='intentional kept-keyword loss failure'):
                s.grad(['thickness'], forward, loss)
        else:
            got = float(s.grad(['thickness'], forward, loss)['thickness'][0])
    if not failure:
        h = 1e-3
        def reference(t):
            return float(solver('numpy', thickness=(t,), **fixed).solve().T.sum().real)
        expected = (reference(170.+h)-reference(170.-h))/(2*h)
        assert got == pytest.approx(expected, abs=1e-9, rel=2e-6)
    assert float(s.wavelength) == 850.
    assert complex(s.theta) == .23
    np.testing.assert_array_equal(s.thickness, [170.])
    assert not any(_holds_tracer(v) for v in vars(s).values())
    np.testing.assert_allclose(s.solve().T, solver('numpy', **fixed).solve().T, atol=1e-12)


def test_a_named_gradient_parameter_stays_temporary_even_if_overridden_concretely():
    s = solver()
    def forward(**unused):
        return s.solve(thickness=[190.], wavelength=850.)
    got = s.grad(['thickness'], forward, lambda result: result.T.sum().real)['thickness']
    np.testing.assert_array_equal(got, [0.])
    np.testing.assert_array_equal(s.thickness, [170.])
    assert float(s.wavelength) == 850.


def test_replayed_keywords_keep_the_accepted_numpy_value_after_caller_mutation():
    s = solver()
    raster = np.full((1, 2, 2), 9.)
    def forward(**values):
        result = s.solve(structure=raster, **values)
        raster[:] = 16.
        return result
    s.grad(['thickness'], forward, lambda result: result.T.sum().real)
    np.testing.assert_array_equal(s.structure, np.full((1, 2, 2), 9.))
    np.testing.assert_array_equal(raster, np.full((1, 2, 2), 16.))
    np.testing.assert_allclose(s.solve().T, solver('numpy', structure=np.full((1, 2, 2), 9.)).solve().T,
                               atol=1e-12)


def test_a_keyword_incompatible_with_the_restored_table_rolls_back_the_scene():
    table = _legacy_wavebit_TabulatedMaterial([800., 1000.], [2.25, 4.])
    s = solver(top=table)
    original = s.solve()
    def forward(**values):
        s.top = 2.25
        return s.solve(wavelength=1100., **values)
    with pytest.raises(ValueError, match='explicitly kept conditions.*original scene'):
        s.grad(['thickness'], forward, lambda result: result.T.sum().real)
    assert float(s.wavelength) == 900.
    assert s._top_dispersion is table
    np.testing.assert_array_equal(s.thickness, [170.])
    assert not any(_holds_tracer(v) for v in vars(s).values())
    np.testing.assert_allclose(s.solve().T, original.T, atol=1e-12)


def test_replayed_drawn_raster_keeps_the_accepted_cell_and_material_values():
    s = solver()
    cell = wavebit.Lattice.rectangular(600., 750.)
    raster = np.full((1, 2, 2), 9.)
    declared = cell.drawn(raster)
    def forward(**values):
        result = s.solve(structure=declared, **values)
        raster[:] = 16.
        cell.u[0] = 620.
        return result
    s.grad(['thickness'], forward, lambda result: result.T.sum().real)
    np.testing.assert_array_equal(s.structure, np.full((1, 2, 2), 9.))
    np.testing.assert_array_equal(s._structure_drawn_on, np.diag([600., 750.]))
    np.testing.assert_allclose(s.solve().T, solver('numpy', structure=np.full((1, 2, 2), 9.)).solve().T,
                               atol=1e-12)
