"""The NVF option names describe their scope at every public entry point."""
import inspect

import numpy as np
import pytest

import wavebit
from wavebit import Numerics, NormalMethod, Projector, SpecError
from wavebit.manifest import manifest


RENAMED = [
    ('normal_method', 'raster_normal_method', 'smooth'),
    ('normal_grid', 'shape_nvf_grid_size', 32),
    ('projector', 'nvf_projector', 'circular'),
]
RASTER = np.array([[[1., 1., 4., 4.], [1., 4., 4., 1.],
                    [1., 4., 4., 1.], [1., 1., 4., 4.]]])
SHAPE = [dict(background=1., shapes=[
    dict(type='circle', cx=0., cy=0., lx=0.4, material=4.)])]
KW = dict(period=(1., 1.), wavelength=1.3, thickness=(.2,),
          truncation_order=(1, 1), factorization='nvf')


@pytest.mark.parametrize('backend,module', [(0, 'numpy'), (1, 'jax'), (2, 'torch')])
def test_constructor_and_properties_publish_only_the_new_names(backend, module):
    pytest.importorskip(module)
    solver = wavebit.rcwa(backend=backend, raster_normal_method='smooth',
                          shape_nvf_grid_size=32, nvf_projector='circular')
    params = {name for cls in type(solver).__mro__ if '__init__' in cls.__dict__
              for name in inspect.signature(cls.__init__).parameters}
    for old, new, value in RENAMED:
        assert new in params
        assert old not in params
        assert getattr(solver, new) == value
        assert not hasattr(solver, old)
    solver.shape_nvf_grid_size = None
    assert solver.shape_nvf_grid_size == 1024


@pytest.mark.parametrize('old,new,value', RENAMED)
def test_old_constructor_names_name_the_replacement(old, new, value):
    with pytest.raises(TypeError, match=new):
        wavebit.rcwa(**{old: value})


@pytest.mark.parametrize('old,new,value', RENAMED)
@pytest.mark.parametrize('door', ['solve', 'solve_field', 'freeze_geometry',
                                 'eps_conv_matrices', 'mu_conv_matrices'])
def test_old_solve_names_fail_before_changing_any_condition(old, new, value, door):
    solver = wavebit.rcwa(structure=RASTER, answers='field', **KW)
    call = getattr(solver, door)
    args = (2, 2, 2) if door == 'solve_field' else ()
    with pytest.raises(TypeError, match=new):
        call(*args, wavelength=9., **{old: value})
    assert float(solver.wavelength) == 1.3
    with pytest.raises(AttributeError, match=new):
        setattr(solver, old, value)


def test_each_scoped_name_is_rejected_on_the_other_input_path():
    with pytest.raises(NotImplementedError, match='raster_normal_method'):
        wavebit.rcwa(structure=SHAPE, raster_normal_method='smooth', **KW).solve()
    with pytest.raises(NotImplementedError, match='shape_nvf_grid_size'):
        wavebit.rcwa(structure=RASTER, shape_nvf_grid_size=32, **KW).solve()
    with pytest.raises(NotImplementedError, match='nvf_projector'):
        wavebit.rcwa(structure=RASTER, nvf_projector='linear',
                     **{**KW, 'factorization': 'li'}).solve()


def test_new_solve_names_reach_the_raster_and_shape_builders():
    raster = wavebit.rcwa(structure=RASTER, **KW)
    raster.solve(raster_normal_method='smooth', nvf_projector='circular')
    assert raster.raster_normal_method == NormalMethod.SMOOTH
    assert raster.nvf_projector == Projector.CIRCULAR
    shape = wavebit.rcwa(structure=SHAPE, **KW)
    shape.solve(shape_nvf_grid_size=32, nvf_projector='circular')
    assert shape.shape_nvf_grid_size == 32
    assert shape.nvf_projector == Projector.CIRCULAR


def test_numerics_serializes_only_canonical_keys():
    values = dict(raster_normal_method='smooth', nvf_projector='circular',
                  shape_nvf_grid_size=32)
    numerics = Numerics.from_dict(values)
    assert Numerics.from_dict(numerics.to_dict()) == numerics
    for old, new, value in RENAMED:
        assert old not in numerics.to_dict()
        assert numerics.to_dict()[new] == value
        assert numerics.as_kwargs()[new] == value
        with pytest.raises(SpecError, match=new):
            Numerics.from_dict({old: value})
    assert 'raster_normal_method' not in Numerics().as_kwargs()
    assert 'nvf_projector' not in Numerics().as_kwargs()
    assert Numerics().as_kwargs()['shape_nvf_grid_size'] is None


def test_version_seven_specs_explain_the_key_only_migration():
    with pytest.raises(SpecError, match="version='10'.*shape_nvf_grid_size"):
        wavebit.JobSpec.from_dict({'version': '7'})
    assert manifest()['results']['jobspec']['spec_version'] == '10'
    assert int(manifest()['manifest_version']) >= 10   # '10' renamed the keys; later bumps keep them
    for path in manifest()['paths']:
        assert 'normal_method' not in path and 'projector' not in path
        assert 'raster_normal_method' in path and 'nvf_projector' in path


@pytest.mark.parametrize('shape', [False, True])
def test_jax_tree_roundtrip_preserves_explicit_and_unset_options(shape):
    jax = pytest.importorskip('jax')
    options = (dict(shape_nvf_grid_size=32) if shape
               else dict(raster_normal_method='smooth'))
    solver = wavebit.rcwa(backend=1, structure=SHAPE if shape else RASTER,
                          nvf_projector='circular', **options, **KW)
    leaves, tree = jax.tree_util.tree_flatten(solver)
    restored = jax.tree_util.tree_unflatten(tree, leaves)
    assert restored._normal_grid == (32 if shape else None)
    assert restored._normal_method == (None if shape else NormalMethod.SMOOTH)
    assert restored.nvf_projector == Projector.CIRCULAR
    geometry = restored.freeze_geometry()
    direct = solver.solve().T
    compiled = jax.jit(lambda s: s.solve(geometry=geometry).T)(restored)
    np.testing.assert_allclose(compiled, direct, atol=1e-12, rtol=1e-12)
