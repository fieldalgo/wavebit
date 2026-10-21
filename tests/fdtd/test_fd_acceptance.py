"""Counterexamples to using a moving error budget as an accuracy gate."""
import numpy as np
import pytest

from tests.fixtures.fdtd.acceptance import assess, overall


def rows(values, valid=True):
    return [dict(values=np.array([v], complex), valid=valid) for v in values]


def grade(times=(0.,) * 5, boundaries=(0., 0.), reference=0., check=0., valid=True):
    return assess(rows(times, valid), rows(boundaries), [reference], [check], [1.])


def test_a_stable_wrong_answer_fails_the_independent_reference():
    assert grade((2.,) * 5, (2., 2.))['status'] == 'FAIL'


def test_growth_cannot_buy_a_larger_accuracy_budget():
    result = grade((0., 1., 2., 4., 8.), (8., 8.))
    assert result['status'] == 'BLOCKED'
    assert 'time_uncertainty' in result['issues']
    assert result['accuracy_ratio'] == [8.]


def test_matching_endpoints_do_not_hide_a_middle_time_excursion():
    result = grade((0., 0., .3, 0., 0.))
    assert result['status'] == 'BLOCKED'


def test_both_boundary_controls_are_compared_with_each_other_too():
    assert grade(boundaries=(.09, -.09))['status'] == 'BLOCKED'


def test_complex_phase_changes_are_not_hidden_by_equal_power():
    assert grade((1j,) * 5, (1j, 1j), reference=1., check=1.)['status'] == 'FAIL'


@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_nonfinite_data_never_pass(value):
    assert grade((value,) * 5)['status'] == 'BLOCKED'


def test_missing_runs_and_missing_stability_are_not_passes():
    assert grade(times=(0.,) * 4)['status'] == 'BLOCKED'
    assert grade(valid=False)['status'] == 'BLOCKED'
    assert overall([]) == 'BLOCKED'
    assert overall(['PASS', 'BLOCKED']) == 'BLOCKED'
    assert overall(['BLOCKED', 'FAIL']) == 'FAIL'


def test_reference_uncertainty_has_its_own_cap():
    assert grade(check=.051)['status'] == 'BLOCKED'
    assert grade(check=.05)['status'] == 'PASS'


def test_qualified_results_keep_the_original_accuracy_limit():
    assert grade((.99,) * 5, (.99, .99))['status'] == 'PASS'
    assert grade((1.,) * 5, (1., 1.))['status'] == 'FAIL'


def test_each_observable_uses_its_own_budget():
    data = [dict(values=[0., .02], valid=True)] * 5
    result = assess(data, data[:2], [0., 0.], [0., 0.], [1., .01])
    assert result['status'] == 'FAIL'
    assert result['accuracy_ratio'] == [0., 2.]


def test_absent_row_fields_are_missing_evidence():
    assert assess([{}]*5, [{}]*2, [0.], [0.], [1.])['status'] == 'BLOCKED'
    data = [dict(values=[0.])]*5
    assert assess(data, data[:2], [0.], [0.], [1.])['status'] == 'BLOCKED'


def test_qualification_only_still_enforces_fixed_uncertainty():
    result = assess(rows([2.]*5), rows([2., 2.]), [0.], [0.], [1.], qualification_only=True)
    assert result['status'] == 'PASS' and result['qualification_only']
    result = assess(rows([0., 1., 2., 4., 8.]), rows([8., 8.]), [0.], [0.], [1.],
                    qualification_only=True)
    assert result['status'] == 'BLOCKED'


@pytest.mark.parametrize('kind,n', [('planar', 20), ('planar', 40), ('planar', 80),
                                  ('conical', 20), ('conical', 40)])
@pytest.mark.parametrize('protocol', ['original', 'stabilized', 'spectral'])
def test_boundary_controls_preserve_the_physical_experiment(kind, n, protocol):
    from tests.fdtd.check_planewave_acceptance import fixture

    base, _ = fixture(kind, n, 2., protocol=protocol)
    for variant in ('thick', 'far'):
        changed, _ = fixture(kind, n, 2., variant, protocol)
        assert changed.sources == base.sources
        assert changed.monitors == base.monitors
        assert changed.resolution == base.resolution
        assert changed.boundaries.x == base.boundaries.x
        assert changed.boundaries.y == base.boundaries.y
        assert changed.run == base.run
        if kind == 'planar':
            assert changed.structures[:-1] == base.structures[:-1]
            assert changed.structures[-1].shape.z[1] == base.structures[-1].shape.z[1]
            assert changed.structures[-1].material == base.structures[-1].material
        else:
            assert changed.structures == base.structures
        # Moving the boundary by integral cells preserves the material grid phase.
        low = lambda scene: scene.domain.center[2]-scene.domain.size[2]/2
        shift = (low(changed)-low(base))*n
        assert shift == pytest.approx(round(shift))


@pytest.mark.parametrize('control,valid,issue', [(.11, True, 'bandwidth_uncertainty'),
                                               (0., False, 'invalid_run_diagnostics')])
def test_bandwidth_control_cannot_borrow_a_good_time_or_boundary_result(control, valid, issue):
    result = assess(rows((0.,)*5), rows((0.,)*2), [0.], [0.], [1.],
                    bandwidth_row=rows((control,), valid)[0])
    assert result['status'] == 'BLOCKED' and issue in result['issues']


def test_bandwidth_control_is_also_subject_to_absolute_accuracy():
    result = assess(rows((.99,)*5), rows((.99,)*2), [0.], [0.], [1.],
                    bandwidth_row=rows((1.01,))[0])
    assert result['status'] == 'FAIL' and result['issues'] == ['accuracy']
