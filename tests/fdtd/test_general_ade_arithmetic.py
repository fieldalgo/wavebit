"""Compact pole storage retains the ordinary recurrence arithmetic guard."""
import numpy as np
import pytest

from wavebit import fdtd
from wavebit.fdtd import _legacy as fd
from wavebit.fdtd._kernel.poles import CompactPoleTerm, PoleTerm
from wavebit.fdtd._lower_public import lower_problem
from wavebit.fdtd._prepare.compact import compact_plan
from tests.fdtd.test_general_ade_contract import SCALE, numerics, problem


def live_problem(kind, tiny, device='cpu'):
    torch = pytest.importorskip('torch')
    # Positive finite declarations whose nonzero second-order drive is below
    # float64's normal range. The existing arithmetic contract refuses it.
    value = ((1e-310 if tiny else 12.) / SCALE**2 if kind == 'lorentz'
             else (1e-155 if tiny else np.sqrt(12.)) / SCALE)
    leaf = torch.tensor(value, dtype=torch.float64, device=device, requires_grad=True)
    return problem(kind, **{('strength' if kind == 'lorentz' else 'omega_p'): leaf})


@pytest.mark.parametrize('kind', ['lorentz', 'drude'])
@pytest.mark.parametrize('compact', [False, True], ids=['ordinary', 'compact'])
def test_nonrepresentable_live_drive_marks_lowered_params_invalid(kind, compact):
    torch = pytest.importorskip('torch')
    seed = problem(kind)
    scene, _ = lower_problem(seed, numerics())
    plan = fd.plan(scene)
    if compact:
        plan = compact_plan(scene, plan)
    assert type(plan.terms[0]) is (CompactPoleTerm if compact else PoleTerm)
    for tiny in (False, True):
        live, _ = lower_problem(live_problem(kind, tiny), numerics())
        params = fd.lower(live, plan, 'torch')
        coefficient = params['poles']['poles[0]']['drive'][0]
        assert bool(torch.isfinite(coefficient)) is (not tiny)
        assert bool(torch.isfinite(params['stable'])) is (not tiny)
        assert bool(torch.isfinite(params['be']).all()) is (not tiny)


@pytest.mark.parametrize('kind', ['lorentz', 'drude'])
def test_gpu_nonrepresentable_live_drive_is_not_reported_stable(kind):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA device unavailable')
    prepared = fdtd.prepare(problem(kind), numerics=numerics(), backend='torch', device='cuda:0')
    assert type(prepared._plan.terms[0]) is CompactPoleTerm
    for tiny in (False, True, False):
        result = prepared.run(live_problem(kind, tiny, 'cuda:0'))
        values = result['electric'].values
        assert bool(result.validity.checks['runtime_stability']) is (not tiny)
        if tiny:
            assert bool(torch.isnan(values).all())
        else:
            assert bool(torch.isfinite(values).all()) and bool((values != 0).any())


@pytest.mark.parametrize('kind', ['lorentz', 'drude'])
@pytest.mark.parametrize('compact', [False, True], ids=['ordinary', 'compact'])
def test_concrete_nonrepresentable_drive_is_refused_with_either_state(kind, compact):
    from wavebit.fdtd._prepare.checks import check_stability
    scene, _ = lower_problem(problem(kind), numerics())
    plan = fd.plan(scene)
    if compact:
        plan = compact_plan(scene, plan)
    bad = problem(kind, **({'strength': 1e-310/SCALE**2} if kind == 'lorentz'
                           else {'omega_p': 1e-155/SCALE}))
    live, _ = lower_problem(bad, numerics())
    with pytest.raises(ValueError, match='PoleArithmeticRefused'):
        check_stability(live, plan)
