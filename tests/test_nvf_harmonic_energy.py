"""Instrument checks for an unaccepted material-defined factorization study."""
import numpy as np
import pytest

from studies import nvf_harmonic_energy as candidate

pytestmark = pytest.mark.research


def test_uniform_slab_uses_candidate_matrices_and_matches_analytic():
    shape = [dict(lx=700., ly=700., epsilon=4.)]
    slots, _ = candidate.matrices(2, 7, shape)
    identity = np.eye(25)
    for got, want in zip(slots, (4*identity, 4*identity, identity/4, 0*identity, 0*identity)):
        np.testing.assert_allclose(got, want, atol=2e-13, rtol=0)
    result = candidate.transmission(2, 7, shape)
    assert result['T'] == pytest.approx(.838052025532751, abs=2e-13)
    assert result['energy_error'] < 2e-13


def test_split_material_is_identical_in_coefficients_and_solve():
    split = [dict(lx=300., ly=300., cx=cx, epsilon=4.) for cx in (-150., 150.)]
    merged = [dict(lx=600., ly=300., epsilon=4.)]
    np.testing.assert_allclose(candidate.material_coefficients(14, split),
                               candidate.material_coefficients(14, merged), atol=2e-15, rtol=0)
    left, right = [candidate.transmission(3, 7, shape) for shape in (split, merged)]
    assert left['T'] == pytest.approx(right['T'], abs=1e-10, rel=0)
    assert max(left['energy_error'], right['energy_error']) < 2e-12


def test_exact_product_integration_is_independent_of_adequate_grid():
    first, _ = candidate.matrices(2, 7, candidate.tilted(), grid=64)
    second, _ = candidate.matrices(2, 7, candidate.tilted(), grid=96)
    for a, b in zip(first, second):
        np.testing.assert_allclose(a, b, atol=2e-12, rtol=0)
    with pytest.raises(ValueError, match='prevent aliasing'):
        candidate.matrices(2, 7, candidate.tilted(), grid=36)


def test_failed_electrostatic_solve_is_not_comparison_data(monkeypatch):
    def failed(operator, right, **kwargs):
        return np.zeros_like(right), 300
    monkeypatch.setattr(candidate, 'cg', failed)
    with pytest.raises(RuntimeError, match='did not converge'):
        candidate.corrector(7, candidate.tilted())


@pytest.mark.parametrize('polarization', ['s', 'p'])
def test_both_polarizations_match_independent_slab_transfer_formula(polarization):
    epsilon = np.array([1., 4., 2.25])
    kz = np.sqrt(epsilon-np.sin(candidate.THETA)**2)
    admittance = kz if polarization == 's' else epsilon/kz
    y0, y1, y2 = admittance
    r01, r12 = (y0-y1)/(y0+y1), (y1-y2)/(y1+y2)
    phase = np.exp(2j*np.pi/candidate.WAVELENGTH*candidate.THICKNESS*kz[1])
    amplitude = (2*y0/(y0+y1))*(2*y1/(y1+y2))*phase/(1+r01*r12*phase**2)
    expected = y2/y0*abs(amplitude)**2
    actual = candidate.transmission(2, 7, [dict(lx=700., ly=700., epsilon=4.)],
                                    polarization=polarization)
    assert actual['T'] == pytest.approx(expected, abs=2e-13)
    with pytest.raises(ValueError, match='polarization'):
        candidate.powers(0, None, polarization='invalid')
