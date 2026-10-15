"""Physical record boxes select native staggered samples in scene coordinates."""

from wavebit._common.periodic import Incidence as _legacy_wavebit_Incidence

from wavebit._common.periodic import Stack as _legacy_wavebit_Stack

from wavebit.fdtd._periodic.api import fdtd as _legacy_wavebit_fdtd
import numpy as np
import pytest


def settings(**changes):
    return dict(dict(structure=np.array([[[2., 2.2], [1.7, 2.1]]]), thickness=[.3],
                     top=1., bot=1., period=[.4, .4], wavelength=1., backend='numpy',
                     resolution=10, n_periods=25, truncation_order=0, tetm='te',
                     record_field=31, return_diagnostics=True), **changes)


@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('grading', [None, True])
def test_physical_box_equals_cropped_full_scene(side, grading):
    bounds = dict(x=(-.11, .16), y=(-.21, .12), z=(-.08, .28))
    kw = settings(incidence_side=side, grading=grading)
    expected, full = _legacy_wavebit_fdtd(**kw)
    actual, cropped = _legacy_wavebit_fdtd(**kw, record_bounds=bounds)
    np.testing.assert_array_equal(actual.incident.t_s, expected.incident.t_s)
    count, total = 0, 0
    for name, field in cropped['fields'].items():
        original = full['fields'][name]
        values = original['values']
        for dimension, axis in enumerate(('z', 'y', 'x'), 1):
            coordinates = original[axis]
            indices = np.flatnonzero((coordinates >= bounds[axis][0]) & (coordinates <= bounds[axis][1]))
            values = np.take(values, indices, axis=dimension)
            np.testing.assert_allclose(field[axis], coordinates[indices], atol=1e-14)
        np.testing.assert_array_equal(field['values'], values)
        np.testing.assert_array_equal(field['t'], original['t'])
        count += field['values'].size
        total += original['values'].size
    assert count < total/4
    primary = cropped['fields'][cropped['field_e_component']]
    np.testing.assert_array_equal(cropped['field_z'], primary['z'])


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_typed_bottom_region_gradient_and_invariant_y(backend):
    def objective(value, which):
        structure = np.full((1, 1, 1), value) if which == 'numpy' else value*ones
        stack = _legacy_wavebit_Stack.from_structure(structure, [.3], top=1., bot=1., period=.4)
        _, d = _legacy_wavebit_fdtd(stack, _legacy_wavebit_Incidence(wavelength=1., tetm='te', incidence_side='bottom'),
                            backend=which, resolution=10, n_periods=25, grading=True,
                            record_field=41, record_components=['ey', 'hx'],
                            record_bounds=dict(x=(-.05, .11), y=(0., 0.), z=(-.11, .41)),
                            return_diagnostics=True, truncation_order=0)
        assert d['fields']['ey']['values'].shape[2] == 1
        return (abs(d['fields']['ey']['values'])**2).sum()
    if backend == 'jax':
        jax = pytest.importorskip('jax')
        jax.config.update('jax_enable_x64', True)
        import jax.numpy as jnp
        ones = jnp.ones((1, 1, 1))
        actual = jax.jit(jax.grad(lambda v: objective(v, 'jax')))(2.)
    else:
        torch = pytest.importorskip('torch')
        ones = torch.ones((1, 1, 1), dtype=torch.float64)
        value = torch.tensor(2., dtype=torch.float64, requires_grad=True)
        actual, = torch.autograd.grad(objective(value, 'torch'), value)
    delta = 1e-4
    expected = (objective(2.+delta, 'numpy')-objective(2.-delta, 'numpy'))/(2*delta)
    assert float(actual) == pytest.approx(expected, rel=1e-5, abs=1e-8)


@pytest.mark.parametrize('bounds', [{}, {'bad': (0., 1.)}, {'z': (1., 0.)},
                                    {'z': (0., float('nan'))}, {'x': (0j, 1.)},
                                    {'x': (0.,)}, {'y': (9., 10.)}])
def test_invalid_or_empty_bounds(bounds):
    with pytest.raises(ValueError, match='record_bounds'):
        _legacy_wavebit_fdtd(**settings(), record_bounds=bounds)


def test_bounds_require_history():
    with pytest.raises(ValueError, match='record_bounds requires record_field'):
        _legacy_wavebit_fdtd(**settings(record_field=None), record_bounds={'z': (0., .2)})
