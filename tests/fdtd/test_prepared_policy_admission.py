"""Combined policy reports distinguish static acceptance from live runtime validity."""
import numpy as np
import pytest

from wavebit import fdtd
from wavebit.fdtd._runtime import runtime
from tests.fdtd.test_grazing_material_ad import quiet
from tests.fdtd.test_prepared_admission_recovery import problem, settings, array


@pytest.mark.parametrize('backend', ['torch', 'jax'])
def test_policy_report_exposes_deferred_runtime_validation(backend):
    runtime(backend)
    prepared = quiet(fdtd.prepare, problem(), numerics=settings(), backend=backend)
    original = prepared.report.resolved['numerical_policies']
    assert prepared.report.capabilities['runtime_validation'] == 'concrete_inputs'
    def validate(e):
        report = quiet(prepared.validate, problem(e))
        assert report.ok  # no host error, not a verdict on a traced numeric value
        assert report.capabilities['runtime_validation'] == 'deferred'
        assert report.resolved['numerical_policies'] == original
        return e*0
    if backend == 'jax':
        import jax
        compiled = jax.jit(validate)
        for epsilon in (2.1, 2.55, 2.1):
            compiled(epsilon)
        assert compiled._cache_size() == 1
    else:
        import torch
        for epsilon in (2.1, 2.55, 2.1):
            validate(torch.tensor(epsilon, dtype=torch.float64, requires_grad=True))
    held = quiet(prepared.validate, problem())
    for epsilon, accepted in ((2.1, True), (2.55, False), (2.1, True)):
        concrete = quiet(prepared.validate, problem(epsilon))
        assert concrete.ok == accepted
        if accepted:
            assert concrete.capabilities['runtime_validation'] == 'concrete_inputs'
            assert concrete.resolved['numerical_policies'] == original
        else:
            assert not concrete.resolved
    assert held.ok and held.resolved['numerical_policies'] == original
    assert prepared.report.resolved['numerical_policies'] == original


@pytest.mark.parametrize('backend', ['torch', 'jax'])
def test_run_policy_provenance_survives_valid_invalid_valid_without_mutation(backend):
    runtime(backend)
    n = settings(steps=60)
    prepared = quiet(fdtd.prepare, problem(), numerics=n, backend=backend)
    expected = prepared.report.resolved['numerical_policies']
    # The reported split is actually applied to a standing body, not just requested.
    assert expected['junction']['pairs'][0]['rule'] == 'split'
    def objective(e):
        result = quiet(prepared.run, problem(e))
        policy = result.provenance['numerical_policies']
        assert policy == expected
        rows = [r for r in result.diagnostics['grazing'] if r['medium'] == 'structures[0]']
        assert rows and all(r['verdict'] == 'skipped' for r in rows)
        return ((result.final_fields.values['E']**2).sum(),
                (result.final_fields.values['E'], result.validity.checks['runtime_stability']))
    if backend == 'jax':
        import jax
        evaluate = jax.jit(jax.value_and_grad(objective, has_aux=True))
    else:
        import torch
        def evaluate(e):
            e = torch.tensor(e, dtype=torch.float64, requires_grad=True)
            loss, aux = objective(e)
            gradient, = torch.autograd.grad(loss, e)
            return (loss, aux), gradient
    first = None
    for epsilon in (2.1, 2.55, 2.1):
        (loss, (electric, stable)), gradient = evaluate(epsilon)
        assert bool(stable) == (epsilon == 2.1)
        if epsilon == 2.55:
            assert np.isnan(array(loss)) and np.isnan(array(gradient))
            continue
        direct = quiet(fdtd.run, problem(epsilon), numerics=n)
        assert direct.provenance['numerical_policies'] == expected
        np.testing.assert_allclose(array(electric), direct.final_fields.values['E'], rtol=3e-12, atol=1e-14)
        assert abs(float(array(gradient))) > 1e-10
        now = (array(loss).copy(), array(electric).copy(), array(gradient).copy())
        if first is None:
            first = now
            held = electric
        else:
            for actual, original in zip(now, first):
                np.testing.assert_array_equal(actual, original)
            np.testing.assert_array_equal(array(held), first[1])
    h = 1e-5
    def fresh(e):
        return (quiet(fdtd.run, problem(e), numerics=n).final_fields.values['E']**2).sum()
    np.testing.assert_allclose(first[2], (fresh(2.1+h)-fresh(2.1-h))/(2*h), rtol=2e-5, atol=1e-9)
    with pytest.raises(ValueError, match='Grazing order refused'):
        quiet(prepared.run, problem(2.55))
    after = quiet(prepared.run, problem())
    assert after.provenance['numerical_policies'] == expected
    with pytest.raises(TypeError):
        after.provenance['numerical_policies']['junction']['structures'][0]['rule'] = 'refuse'
    assert prepared.report.resolved['numerical_policies'] == expected
    if backend == 'jax':
        assert evaluate._cache_size() == 1


def test_nondefault_applied_policy_is_transferred_without_reinterpreting_request():
    from tests.fdtd.test_public_junction import example
    p, n = example(extra=False)
    n = n.replace(duration=fdtd.Duration(steps=2))
    prepared = quiet(fdtd.prepare, p, numerics=n)
    expected = prepared.report.resolved['numerical_policies']
    assert expected['junction']['pairs'][0]['rule'] == 'staircase'
    first = quiet(prepared.run, p)
    report = quiet(prepared.validate, p)
    second = quiet(prepared.run, p)
    assert first.provenance['numerical_policies'] == report.resolved['numerical_policies'] == expected
    assert second.provenance['numerical_policies'] == expected
    assert first.provenance['numerical_policies'] is not second.provenance['numerical_policies']
