"""Dense Maxwell-factor controls for memory-bounded modal assembly."""
import numpy as np
import pytest
from scipy.linalg import solve

from studies import nvf_mirror_convergence as mirror
from studies.nvf_harmonic_energy import THETA, WAVELENGTH
from studies.nvf_mirror_convergence import parity_basis, sector_geometry, sector_magnetic_modes, sector_operator
from studies.nvf_mirror_doubling import sector_factors
from studies.nvf_polynomial_body import PolynomialBodyGapModel

pytestmark = pytest.mark.research


@pytest.mark.parametrize('order', [0, (0, 3), (3, 0), (2, 4), (6, 5)])
@pytest.mark.parametrize('parity', [-1, 1])
def test_direct_storage_operator_and_factor_action_match_dense_maxwell(order, parity):
    a, b = parity_basis(order, parity), parity_basis(order, -parity)
    size, na = len(a['n'])+len(b['n']), len(a['n'])
    rng = np.random.default_rng(663)
    def complex_matrix(shape):
        return rng.normal(size=shape)+1j*rng.normal(size=shape)
    # Non-Hermitian inputs also detect an accidental adjoint or hidden
    # symmetrization. Real-field/energy validation belongs to mirror_sector.
    e, z, w = complex_matrix((size, size)), complex_matrix((na, na)), complex_matrix((size, size))
    rates = rng.uniform(.5, 2, size)+1j*rng.uniform(.1, .3, size)
    originals = [x.copy() for x in (e, z, w, rates)]
    sector = dict(a=a, b=b, effective=e, z_inverse=z)
    geometry = sector_geometry(sector, sparse=True)
    left, right, dense_lr, _ = sector_factors(sector)
    operator = sector_operator(e, z, geometry)
    assert operator.flags.f_contiguous
    np.testing.assert_allclose(operator, dense_lr, atol=2e-11, rtol=2e-14)
    np.testing.assert_allclose(operator, left@right, atol=2e-10, rtol=2e-13)
    v = sector_magnetic_modes(e, w, rates, geometry)
    assert v.flags.f_contiguous
    np.testing.assert_allclose(v, (right@w)/rates[None], atol=3e-12, rtol=2e-13)
    for actual, expected in zip((e, z, w, rates), originals):
        np.testing.assert_array_equal(actual, expected)


@pytest.fixture(scope='module')
def patterned_coefficients():
    return PolynomialBodyGapModel(6, rows=4).coefficients(17., dict(seams=[31.], values=[4., 9.]))[0]


@pytest.mark.parametrize('pol', ['p', 's'])
@pytest.mark.parametrize('thickness', [0., 300., 2400.])
def test_panel_matching_equals_coupled_two_interface_system(patterned_coefficients, pol, thickness, monkeypatch):
    monkeypatch.setattr(mirror, 'THICKNESS', thickness)
    sector = mirror.mirror_sector(patterned_coefficients, 6, pol)
    w, v, rates = mirror.sector_modes(sector)
    size, na = len(w), len(sector['a']['n'])
    assert size > 128  # Exercise the second row and column panels.
    yt, yb = mirror.exterior_admittance(sector, 1.), mirror.exterior_admittance(sector, 2.25)
    decay = np.exp(-2*np.pi/WAVELENGTH*thickness*rates)
    basis = sector['a'] if pol == 'p' else sector['b']
    origin = np.flatnonzero((basis['n'] == 0) & (basis['m'] == 0))[0]
    incoming = np.zeros(size, complex)
    incoming[origin if pol == 'p' else na+origin] = np.cos(THETA) if pol == 'p' else 1.
    # Independent continuity system for forward amplitudes at the top and
    # backward amplitudes at the bottom. It never constructs O or Rround.
    boundary = np.block([[v+yt@w, (yt@w-v)*decay[None]],
                         [(v-yb@w)*decay[None], -v-yb@w]])
    amplitudes = solve(boundary, np.r_[2*(yt@incoming), np.zeros(size)])
    forward, backward = amplitudes[:size], amplitudes[size:]
    transmitted = w@(decay*forward+backward)
    reflected = w@(forward+decay*backward)-incoming
    def power(field, admittance):
        h = admittance@field
        return float((np.vdot(field[:na], h[size-na:])-np.vdot(field[na:], h[:size-na])).imag/np.cos(THETA))
    expected = power(transmitted, yb), power(reflected, yt)
    # Isolate matching from the eigensolve, then reverse every propagating
    # mode label. Slab powers must be invariant even though V W^-1 is not.
    monkeypatch.setattr(mirror, 'sector_modes', lambda _: (w.copy(), v.copy(), rates.copy()))
    np.testing.assert_allclose(mirror.sector_power(sector), expected, atol=2e-10, rtol=0)
    signs = np.where(abs(rates.imag) > abs(rates.real), -1., 1.)
    assert np.any(signs == -1)
    monkeypatch.setattr(mirror, 'sector_modes', lambda _: (w.copy(), v*signs[None], rates*signs))
    np.testing.assert_allclose(mirror.sector_power(sector), expected, atol=2e-10, rtol=0)


def test_explicit_single_polarization_keeps_its_physical_guard_and_omits_uncomputed_sector(patterned_coefficients):
    full = mirror.mirror_powers(patterned_coefficients, 2)
    for pol in ('p', 's'):
        events = []
        result = mirror.mirror_powers(patterned_coefficients, 2, events.append, polarizations=[pol])
        assert set(result) == {pol+'_T', pol+'_R', 'energy_error'}
        assert all(event['polarization'] == pol for event in events)
        for key in (pol+'_T', pol+'_R'):
            assert result[key] == pytest.approx(full[key], abs=1e-13, rel=0)
    for invalid in ([], ['p', 'p'], ['s', 'x']):
        with pytest.raises(ValueError, match='polarizations'):
            mirror.mirror_powers(patterned_coefficients, 2, polarizations=invalid)
    with pytest.raises(FloatingPointError, match='power balance'):
        mirror.powers_from_sectors(lambda pol: {}, propagate=lambda *_: (.9, .2), polarizations=['s'])
