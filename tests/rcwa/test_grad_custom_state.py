"""A custom gradient query cannot leave derived conditions or edited recipes behind."""

from wavebit.compat import legacy as _legacy_wavebit
import numpy as np
import pytest

from wavebit.rcwa._engine.engine import _holds_tracer

jax = pytest.importorskip('jax')


def design(width):
    return [dict(background=1., shapes=[dict(type='rectangle', cx=0., cy=0., lx=width,
                                           ly=100., angle=0., material=4.)])]


def solver(backend='jax', **kwargs):
    cfg = dict(period=(600., 750.), structure=design(100.), thickness=(100.,),
               wavelength=900., truncation_order=(0, 0), theta=.13, phi=.21,
               factorization='li', answers='field')
    return _legacy_wavebit.rcwa(backend=backend, **(cfg | kwargs))


@pytest.mark.parametrize('write', ['structure', 'inplace', 'wavelength'])
@pytest.mark.parametrize('failure', [None, 'forward', 'loss'])
def test_derived_conditions_and_recipe_edits_are_temporary(write, failure):
    s = solver()
    original_recipe = s.structure
    geometry = s.freeze_geometry()
    saved = s.solve()
    saved_field = np.asarray(saved.reconstruct_field(2, 2, 2))
    def forward(thickness):
        if write == 'structure':
            s.structure = design(thickness[0])
        elif write == 'inplace':
            s.structure[0]['shapes'][0]['lx'] = thickness[0]
        else:
            s.wavelength = 800.+thickness[0]
        if failure == 'forward':
            raise RuntimeError('intentional custom gradient failure')
        return s.solve(geometry=geometry)
    def loss(result):
        if failure == 'loss':
            raise RuntimeError('intentional custom gradient failure')
        return result.T.sum().real
    with jax.checking_leaks():
        if failure:
            with pytest.raises(RuntimeError, match='intentional custom gradient failure'):
                s.grad(['thickness'], forward, loss)
        else:
            got = float(s.grad(['thickness'], forward, loss)['thickness'][0])
    if failure is None:
        def reference(t):
            changes = ({'structure': design(t)} if write != 'wavelength'
                       else {'wavelength': 800.+t})
            return float(solver('numpy', thickness=(t,), **changes).solve().T.sum().real)
        h = 1e-3
        expected = (reference(100.+h)-reference(100.-h))/(2*h)
        assert abs(expected) > 1e-7
        assert got == pytest.approx(expected, abs=1e-9, rel=2e-6)
    assert not any(_holds_tracer(value) for value in vars(s).values())
    assert s.structure is original_recipe
    assert original_recipe[0]['shapes'][0]['lx'] == 100.
    np.testing.assert_array_equal(s.thickness, [100.])
    assert float(s.wavelength) == 900.
    assert s.T1 is None and s.layer_info_list == []
    np.testing.assert_array_equal(saved.reconstruct_field(2, 2, 2), saved_field)
    np.testing.assert_allclose(s.solve().T, saved.T, atol=1e-12)


@pytest.mark.parametrize('failure', [False, True])
def test_the_loss_uses_the_same_temporary_state_scope(failure):
    s = solver()
    geometry = s.freeze_geometry()
    original = s.structure
    def loss(result):
        value = result.T.sum().real
        s.structure[0]['shapes'][0]['lx'] = 100.*value
        s.theta = value
        if failure:
            raise RuntimeError('intentional mutating loss failure')
        return value
    with jax.checking_leaks():
        if failure:
            with pytest.raises(RuntimeError, match='intentional mutating loss failure'):
                s.grad(['thickness'], lambda **values: s.solve(geometry=geometry, **values), loss)
        else:
            gradient = s.grad(['thickness'], lambda **values: s.solve(geometry=geometry, **values), loss)
            assert np.isfinite(np.asarray(gradient['thickness'])).all()
    assert s.structure is original
    assert original[0]['shapes'][0]['lx'] == 100.
    assert complex(s.theta) == .13
    assert not any(_holds_tracer(value) for value in vars(s).values())
    np.testing.assert_allclose(s.solve().T, solver('numpy').solve().T, atol=1e-11)


def test_concrete_custom_conditions_do_not_change_the_original_scene():
    s = solver()
    original = s.structure
    def forward(thickness):
        s.wavelength = 850.
        s.structure = design(120.)
        return s.solve()
    got = float(s.grad(['thickness'], forward, lambda r: r.T.sum().real)['thickness'][0])
    def reference(t):
        return float(solver('numpy', thickness=(t,), wavelength=850., structure=design(120.)).solve().T.sum().real)
    h = 1e-3
    assert got == pytest.approx((reference(100.+h)-reference(100.-h))/(2*h), abs=1e-9, rel=2e-6)
    assert s.structure is original
    assert float(s.wavelength) == 900.
    np.testing.assert_allclose(s.solve().T, solver('numpy').solve().T, atol=1e-11)
