"""Per-run admission and diagnostics through valid/invalid/valid Prepared reuse."""
import numpy as np
import pytest

from wavebit import Material, fdtd
from wavebit.fdtd._runtime import runtime
from tests.fdtd.test_substrate_continuation import pillar, numerics
from tests.fdtd.test_grazing_material_ad import quiet


def problem(epsilon=2.1):
    p = pillar(substrate=epsilon)
    return p.replace(outputs=p.outputs.replace(monitors=(
        p.outputs.monitors[0].replace(wavelengths=(.95,)),)))


def settings(steps=12, stabilize=0.):
    n = numerics(steps=steps)
    return n.replace(absorber=n.absorber.replace(stabilizing_loss=stabilize,
                     reference_media={'z-': Material(2.1)}))


def array(value):
    return value.detach().numpy() if hasattr(value, 'detach') else np.asarray(value)


@pytest.mark.parametrize('backend', ['torch', 'jax'])
def test_traced_grazing_diagnostics_do_not_publish_seed_acceptance(backend):
    runtime(backend)
    p = quiet(fdtd.prepare, problem(), numerics=settings(), backend=backend)
    def run(e):
        r = quiet(p.run, problem(e))
        rows = [row for row in r.diagnostics['grazing']
                if row['medium'] == 'structures[0]' and row['what'] == 'orders']
        assert rows and all(row['verdict'] == 'skipped' for row in rows)
        assert all('traced' in row['why'] for row in rows)
        assert all('cos' not in row for row in rows)
        return r.validity.checks['runtime_stability']
    if backend == 'jax':
        import jax
        assert not bool(jax.jit(run)(2.55))
    else:
        import torch
        assert not bool(run(torch.tensor(2.55, dtype=torch.float64, requires_grad=True)))


@pytest.mark.parametrize('backend', ['torch', 'jax'])
def test_prepared_valid_invalid_valid_preserves_results_and_material_derivatives(backend):
    runtime(backend)
    n = settings(steps=60)
    p = quiet(fdtd.prepare, problem(), numerics=n, backend=backend)
    original_signature, original_plan, original_report = p._signature, p._plan, p.report
    def objective(e):
        r = quiet(p.run, problem(e))
        return ((r.final_fields.values['E']**2).sum(),
                (r.final_fields.values['E'], r.final_fields.values['H'],
                 r.validity.checks['runtime_stability'], r.diagnostics['stable'],
                 r.validity.checks['admitted']))
    if backend == 'jax':
        import jax
        evaluate = jax.jit(jax.value_and_grad(objective, has_aux=True))
    else:
        import torch
        def evaluate(e):
            e = torch.tensor(e, dtype=torch.float64, requires_grad=True)
            value, aux = objective(e)
            derivative, = torch.autograd.grad(value, e)
            return (value, aux), derivative
    first = None
    for value in (2.1, 2.55, 2.2, 2.1):
        (loss, (electric, magnetic, stable, diagnostic, admitted)), gradient = evaluate(value)
        allowed = value != 2.55
        assert bool(stable) == bool(diagnostic) == allowed
        assert bool(admitted)  # static declaration admission is separate from runtime validity
        if not allowed:
            assert np.isnan(array(loss)) and np.isnan(array(gradient))
            assert np.isnan(array(electric)).all() and np.isnan(array(magnetic)).all()
            continue
        direct = quiet(fdtd.run, problem(value), numerics=n)
        for actual, expected in ((electric, direct.final_fields.values['E']),
                                 (magnetic, direct.final_fields.values['H'])):
            np.testing.assert_allclose(array(actual), expected, rtol=3e-12, atol=1e-14)
        assert np.isfinite(array(gradient)) and abs(float(array(gradient))) > 1e-10
        if value == 2.1:
            current = (array(loss).copy(), array(gradient).copy(), array(electric).copy())
            if first is None:
                first = current
                held = electric
            else:
                for actual, expected in zip(current, first):
                    np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(array(held), first[2])
        assert p._signature is original_signature and p._plan is original_plan and p.report is original_report
    h = 1e-5
    def fresh(e):
        return (quiet(fdtd.run, problem(e), numerics=n).final_fields.values['E']**2).sum()
    finite = (fresh(2.1+h)-fresh(2.1-h))/(2*h)
    np.testing.assert_allclose(first[1], finite, rtol=2e-5, atol=1e-9)
    # A concrete eager refusal must also leave the preparation reusable.
    with pytest.raises(ValueError, match='Grazing order refused'):
        quiet(p.run, problem(2.55))
    (after, (_, _, stable, _, _)), derivative = evaluate(2.1)
    assert bool(stable)
    np.testing.assert_array_equal(array(after), first[0])
    np.testing.assert_array_equal(array(derivative), first[1])
    if backend == 'jax':
        assert evaluate._cache_size() == 1


@pytest.mark.parametrize('backend', ['torch', 'jax'])
def test_stabilized_material_refusal_does_not_poison_prepared_reuse(backend):
    runtime(backend)
    n = settings(stabilize=.0001)
    p = quiet(fdtd.prepare, problem(), numerics=n, backend=backend)
    before = quiet(p.run, problem())
    def run(e):
        return quiet(p.run, problem(e)).final_fields.values['E']
    with pytest.raises(ValueError, match='traced material value and fills a stabilized PML layer'):
        if backend == 'jax':
            import jax
            jax.jit(run)(2.1)
        else:
            import torch
            run(torch.tensor(2.1, dtype=torch.float64, requires_grad=True))
    after = quiet(p.run, problem())
    assert bool(after.validity.checks['runtime_stability'])
    assert before.diagnostics['grazing'] == after.diagnostics['grazing']
    for field in ('E', 'H'):
        np.testing.assert_array_equal(array(before.final_fields.values[field]), array(after.final_fields.values[field]))
