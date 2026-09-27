"""Exact congruence controls, including complex and non-Hermitian data."""
import numpy as np
import pytest
from scipy.linalg import LinAlgWarning

from studies.nvf_energy_closure import congruence_solve, hermitian_defect

pytestmark = pytest.mark.research


@pytest.mark.parametrize('complex_data', [False, True])
@pytest.mark.parametrize('hermitian', [False, True])
def test_two_solve_congruence_preserves_both_sides_and_defects(complex_data, hermitian):
    rng = np.random.default_rng(735)
    q, g = rng.normal(size=(2, 13, 13))
    if complex_data:
        q, g = q+1j*rng.normal(size=q.shape), g+1j*rng.normal(size=g.shape)
    q += 4*np.eye(13)
    if hermitian:
        g = g@g.conj().T+np.eye(13)
    original_q, original_g = q.copy(), g.copy()
    inverse = np.linalg.inv(q)
    expected = inverse.conj().T@g@inverse
    result = congruence_solve(q, g)
    np.testing.assert_allclose(result, expected, atol=3e-12, rtol=3e-13)
    np.testing.assert_allclose(q.conj().T@result@q, g, atol=3e-12, rtol=3e-13)
    np.testing.assert_array_equal(q, original_q)
    np.testing.assert_array_equal(g, original_g)
    if not hermitian:
        assert np.max(abs(result-result.conj().T)) > .01
    consumed = congruence_solve(np.asfortranarray(q), np.asfortranarray(g), overwrite=True)
    np.testing.assert_array_equal(consumed, result)


def test_overlapping_inputs_are_copied_before_lu_and_invalid_shapes_are_refused():
    q = np.asfortranarray([[3.+1j, .2-.4j], [-.1+1j, 4.-2j]])
    expected = np.linalg.inv(q).conj().T
    np.testing.assert_allclose(congruence_solve(q, q, overwrite=True), expected, atol=2e-15, rtol=0)
    for q, g in ((np.ones((2, 3)), np.eye(2)), (np.eye(2), np.eye(3)),
                 (np.array([[np.nan]]), np.ones((1, 1))), (np.empty((0, 0)), np.empty((0, 0)))):
        with pytest.raises(ValueError, match='energy congruence'):
            congruence_solve(q, g)


def test_conditioning_and_singularity_diagnostics_are_not_lost_when_reusing_lu():
    with pytest.warns(LinAlgWarning, match='rcond'):
        congruence_solve(np.diag([1., 1e-18]), np.eye(2))
    with pytest.raises(np.linalg.LinAlgError, match='singular'):
        congruence_solve(np.diag([1., 0.]), np.eye(2))
    q, g = np.array([[2., .1], [.3, 1.]]), np.array([[3., 1j], [-1j, 4.]])
    inv = np.linalg.inv(q)
    np.testing.assert_allclose(congruence_solve(q, g, overwrite=True), inv.T@g@inv, atol=2e-15, rtol=0)


@pytest.mark.parametrize('complex_data', [False, True])
def test_noncommuting_pivots_in_the_right_solve(complex_data):
    rng = np.random.default_rng(853)
    q = np.diag(np.arange(1., 12.))+.03*rng.normal(size=(11, 11))
    g = rng.normal(size=q.shape)
    if complex_data:
        q = q+.03j*rng.normal(size=q.shape)
        g = g+1j*rng.normal(size=q.shape)
    q = np.roll(q, 3, axis=0)
    inv = np.linalg.inv(q)
    actual = congruence_solve(q, g)
    np.testing.assert_allclose(actual, inv.conj().T@g@inv, atol=2e-14, rtol=0)
    np.testing.assert_allclose(q.conj().T@actual@q, g, atol=2e-14, rtol=0)


@pytest.mark.parametrize('hermitian', [False, True])
def test_tiled_guard_retains_the_full_absolute_hermitian_defect(hermitian):
    rng = np.random.default_rng(39)
    a = rng.normal(size=(2, 259, 259))
    a = a[0]+1j*a[1]
    if hermitian:
        a = a+a.conj().T
        a[-1, 3] += 2e-9j
    expected = float(np.max(abs(a-a.conj().T)))
    assert hermitian_defect(a) == expected
    assert hermitian_defect(a.T) == expected
    a[5, -1] = np.nan
    assert np.isnan(hermitian_defect(a))
