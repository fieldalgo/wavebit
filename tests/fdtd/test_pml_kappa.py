"""Explicit coordinate stretch: compatibility, per-face isolation and backend derivatives."""
import dataclasses
import json
import warnings

import numpy as np
import pytest

from wavebit import fdtd
from wavebit.errors import IncompatibleProblemError
from wavebit.fdtd import _legacy as old
from wavebit.fdtd._lower_public import lower_problem
from wavebit.fdtd._prepare import cpml, grazing
from wavebit.fdtd._prepare.params import _cpml
from wavebit.fdtd._prepare.sources import column_scene
from wavebit.fdtd._runtime import runtime
from tests.api.test_fdtd_checkpoint_jobs import job
from tests.fdtd.test_lifecycle import plane_problem, plane_numerics


@pytest.mark.parametrize('declare', [fdtd.PML, old.PML])
@pytest.mark.parametrize('value', [0., .99, -1., float('nan'), float('inf'), -float('inf'),
                                   True, False, '8', 2+0j, None])
def test_stretch_is_a_finite_static_real_at_least_one(declare, value):
    with pytest.raises(ValueError, match='kappa_max'):
        declare(kappa_max=value)


@pytest.mark.parametrize('declare', [fdtd.PML, old.PML])
@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_stretch_refuses_traced_values(declare, backend):
    runtime(backend)
    with pytest.raises(ValueError, match='kappa_max.*static'):
        if backend == 'jax':
            import jax
            jax.jit(lambda v: declare(kappa_max=v).kappa_max)(8.)
        else:
            import torch
            declare(kappa_max=torch.tensor(8., requires_grad=True))


def test_job_omits_default_and_roundtrips_explicit_stretch():
    original = job()
    assert original.key == '442ed90310fa35d400ca737757206482eb56c7df1835fd266233920900c4baf4'
    fields = json.loads(original.to_json())['numerics']['fields']['absorber']['fields']
    assert 'kappa_max' not in fields
    explicit = original.replace(numerics=original.numerics.replace(absorber=fdtd.PML(kappa_max=2)))
    assert original.to_json() == explicit.to_json()
    changed = original.replace(numerics=original.numerics.replace(absorber=fdtd.PML(kappa_max=8)))
    assert changed.key != original.key
    restored = type(changed).from_json(changed.to_json())
    assert restored.numerics.absorber.kappa_max == 8.
    assert restored.key == changed.key
    assert type(original).from_json(original.to_json()).numerics.absorber.kappa_max == 2.


def test_default_profiles_are_identical_and_faces_and_source_columns_are_independent():
    n = plane_numerics(steps=2)
    scene, _ = lower_problem(plane_problem(), n)
    p = old.plan(scene)
    assert p.skeleton == 'cde9b6c3836c93acab70525ae5a4baf07c711523b9657708fb72415e6421d804'
    base = _cpml(scene, p, runtime('numpy'))
    faces = scene.boundaries.z
    explicit = scene.replace(boundaries=scene.boundaries.replace(z=tuple(
        face.replace(kappa_max=2.) for face in faces)))
    assert old.plan(explicit).skeleton == p.skeleton
    changed = scene.replace(boundaries=scene.boundaries.replace(z=(faces[0].replace(kappa_max=8.), faces[1])))
    q = old.plan(changed)
    assert q.skeleton != p.skeleton
    assert q.cpml[0].kappa_max == 8.
    params = _cpml(changed, q, runtime('numpy'))
    for key, table in base.items():
        for coefficient, value in table.items():
            if key.startswith('z+'):
                np.testing.assert_array_equal(params[key][coefficient], value)
            elif coefficient == 'ik':
                assert np.max(np.abs(params[key][coefficient]-value)) > .1
    column = column_scene(changed, changed.sources[0], q)
    assert [column.boundaries.face('z'+s).kappa_max for s in '-+'] == [8., 2.]
    closed = changed.replace(boundaries=changed.boundaries.replace(z=(old.PEC(), faces[1])))
    sub = column_scene(closed, closed.sources[0], old.plan(closed))
    assert sub.boundaries.face('z-').kappa_max == 2.
    # Omitting the new parameter and explicitly choosing the old value are bit-identical.
    a = cpml.profiles(12, .05, .025, 1., 1., '-', 1e-8)
    b = cpml.profiles(12, .05, .025, 1., 1., '-', 1e-8, kappa_max=2.)
    for name in a:
        for x, y in zip(a[name], b[name]):
            np.testing.assert_array_equal(x, y)


def test_reflection_prediction_and_refusal_spelling_keep_the_requested_stretch():
    n = plane_numerics(steps=2).replace(absorber=fdtd.PML(kappa_max=8.))
    s, _ = lower_problem(plane_problem(), n)
    p = old.plan(s)
    f = s.boundaries.face('z-')
    got = grazing.face_return(s, p, 'z-', f, 1., .8, .8, 3.)
    from wavebit.fdtd._prepare.layer_return import layer_return
    want = layer_return(12, p.dx, p.dt, .8, 1., '-', 1e-8, 0., .8, 3., kappa_max=8.)
    np.testing.assert_array_equal(got, want)
    baseline = grazing.face_return(s, p, 'z-', f.replace(kappa_max=2.), 1., .8, .8, 3.)
    assert np.max(np.abs(np.array(got)-baseline)) > 1e-5
    assert 'kappa_max=8' in grazing._spell_face(f, 24, 'stabilize=0')


def quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(*args, **kwargs)


def array(value):
    return value.detach().numpy() if hasattr(value, 'detach') else np.asarray(value)


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_public_prepared_forward_and_material_derivative(backend):
    runtime(backend)
    n = plane_numerics(steps=180).replace(absorber=fdtd.PML(kappa_max=8.,
        reference_media={'z-': 1., 'z+': 1.}))
    def problem(e=1.):
        return plane_problem(epsilon=e)
    prepared = quiet(fdtd.prepare, problem(), numerics=n, backend=backend)
    def objective(e):
        r = quiet(prepared.run, problem(e))
        return (r['e'].values**2).sum()
    result = quiet(prepared.run, problem())
    direct = quiet(fdtd.run, problem(), numerics=n)
    for name in ('e', 'h'):
        np.testing.assert_allclose(array(result[name].values), direct[name].values, rtol=3e-12, atol=1e-14)
    if backend == 'numpy':
        altered = dataclasses.replace(prepared, numerics=n.replace(absorber=n.absorber.replace(kappa_max=2.)))
        with pytest.raises(IncompatibleProblemError):
            altered.run(problem())
        return
    if backend == 'jax':
        import jax
        value, derivative = jax.jit(jax.value_and_grad(objective))(1.)
    else:
        import torch
        e = torch.tensor(1., dtype=torch.float64, requires_grad=True)
        value = objective(e)
        derivative, = torch.autograd.grad(value, e)
    def fresh(e):
        out = quiet(fdtd.run, problem(e), numerics=n)
        return (out['e'].values**2).sum()
    h = 1e-5
    finite = (fresh(1.+h)-fresh(1.-h))/(2*h)
    assert abs(finite) > 1e-8
    np.testing.assert_allclose(array(value), fresh(1.), rtol=3e-12)
    np.testing.assert_allclose(array(derivative), finite, rtol=2e-5, atol=1e-9)


def test_nondefault_stretch_does_not_claim_the_default_loss_table():
    from wavebit.fdtd._prepare.checks import _stabilize_notes
    from wavebit.fdtd._assemble import _pml_cause
    n = plane_numerics(steps=2).replace(absorber=fdtd.PML(kappa_max=8., stabilizing_loss=.003))
    scene, _ = lower_problem(plane_problem(), n)
    p = quiet(old.plan, scene)
    notes = _stabilize_notes(scene, p)
    assert len(notes) == 2
    assert all('kappa_max=2 only' in message for _, message in notes)
    assert 'kappa_max=2 only' in _pml_cause(scene, p)
