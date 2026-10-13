"""Pins for the analytic reference used to judge C3, independent of the FDTD solver."""
import math

import numpy as np
import pytest
from scipy import special

from tests.references.mie import cylinder_coefficients, cylinder_cross_sections, sphere_amplitudes, sphere_coefficients, sphere_efficiencies


@pytest.mark.parametrize('m,x,a1,b1', (
    (4/3, 50, .531105889295-.499031485631j, .791924475935-.405931152229j),
    (1.5+1j, 2, .546520203397-.152373857258j, .389714727888+.227896075256j),
    (1.1+25j, 2, .322406907480-.465063542971j, .575167279092+.492912495262j),
))
def test_published_complex_coefficient_pins(m, x, a1, b1):
    """Scott Prahl's documented examples, converted to positive Im(m) for loss.

    The cited 12-decimal values differ by at most 7.58e-13 in this run;
    2e-12 covers their printed precision, not an FDTD error allowance.
    """
    a, b = sphere_coefficients(m, x)
    np.testing.assert_allclose((a[0], b[0]), (a1, b1), rtol=0, atol=2e-12)


def test_downward_recurrence_against_direct_bessel_derivatives():
    # Measured max absolute discrepancy 1.42e-14 over these 72 cases.
    for x in np.geomspace(.001, 50, 12):
        for m in (1.01, 1.5, math.sqrt(12), 1.5+.1j, .1+3j, 1.1+25j):
            a, b = sphere_coefficients(m, x)
            assert np.isfinite(a).all() and np.isfinite(b).all()
            n = np.arange(1, len(a)+1)
            jm = special.spherical_jn(n, m*x)
            djm = special.spherical_jn(n, m*x, derivative=True)
            px = x * special.spherical_jn(n, x)
            pd = special.spherical_jn(n, x) + x*special.spherical_jn(n, x, derivative=True)
            xi = px + 1j*x*special.spherical_yn(n, x)
            xd = pd + 1j*(special.spherical_yn(n, x)
                           + x*special.spherical_yn(n, x, derivative=True))
            pm, pmd = m*x*jm, jm+m*x*djm
            with np.errstate(invalid='ignore'):
                aa = (m*pm*pd-px*pmd)/(m*pm*xd-xi*pmd)
                bb = (pm*pd-m*px*pmd)/(pm*xd-m*xi*pmd)
            # Direct Bessel values overflow for very large imaginary arguments;
            # the published strongly absorbing pin above covers that regime.
            finite = np.isfinite(aa) & np.isfinite(bb)
            if finite.any():
                np.testing.assert_allclose(a[finite], aa[finite], rtol=0, atol=1e-13)
                np.testing.assert_allclose(b[finite], bb[finite], rtol=0, atol=1e-13)


def test_rayleigh_size_phase_and_polarization():
    m, x = 1.5, .001
    q = sphere_efficiencies(m, x)
    contrast = (m*m-1)/(m*m+2)
    rayleigh = 8/3*x**4*contrast**2
    # Measured relative finite-size term 7.06e-8 at x=1e-3; budget 1e-7.
    assert abs(q[1]/rayleigh-1) < 1e-7
    theta = np.array((0., 30., 60., 90., 120., 180.))
    s1, s2 = sphere_amplitudes(m, x, theta)
    base = -1j*contrast*x**3
    # Measured finite-size correction is O(x**2); 1e-6 resolves the leading
    # Rayleigh term while discriminating its phase and the two polarizations.
    np.testing.assert_allclose(s1/base, np.ones_like(theta), rtol=0, atol=1e-6)
    np.testing.assert_allclose(s2/base, np.cos(np.deg2rad(theta)), rtol=0, atol=1e-6)


@pytest.mark.parametrize('x', (.001, math.pi/2, 10., 5000.))
def test_lossless_energy_and_series_stability(x):
    q = sphere_efficiencies(1.5, x)
    # Lossless absorption is <=1.12e-16 on these cases; no clamp in the oracle.
    assert abs(q[2]) <= 1e-14 * q[1]
    longer = sphere_efficiencies(1.5, x, extra_orders=16, padding=512)
    np.testing.assert_allclose(q[:2], longer[:2], rtol=1e-12, atol=0)


def test_absorbing_sphere_and_extinction_paradox():
    q = sphere_efficiencies(1.5+.1j, 10)
    # Eight-decimal benchmark values; the budget is half the last printed digit.
    np.testing.assert_allclose(q[:2], (2.45979053, 1.23514421), rtol=0, atol=5e-9)
    assert q[2] > 0
    np.testing.assert_allclose(q, sphere_efficiencies(1.5+.1j, 10, extra_orders=16),
                               rtol=1e-12, atol=0)  # measured tail <=6.69e-13 absolute
    large = sphere_efficiencies(1.5, 5000)[0]
    # The geometric-optics limit is 2; measured residual at x=5000 is .00864985.
    assert abs(large-2) < .01


@pytest.mark.parametrize('m', (1.5, 1.5+.1j))
def test_angular_integral_and_optical_theorem(m):
    x = math.pi/2
    mu, weights = np.polynomial.legendre.leggauss(64)
    s1, s2 = sphere_amplitudes(m, x, np.degrees(np.arccos(mu)))
    integral = np.dot(weights, abs(s1)**2 + abs(s2)**2)/x**2
    q = sphere_efficiencies(m, x)
    # 64-point Gauss-Legendre: measured relative closure 1.34e-15.
    np.testing.assert_allclose(integral, q[1], rtol=1e-13, atol=0)
    forward = sphere_amplitudes(m, x, 0.)[1]
    np.testing.assert_allclose(4*forward.real/x**2, q[0], rtol=1e-14, atol=0)


@pytest.mark.parametrize('pol', ('TM', 'TE'))
@pytest.mark.parametrize('eps', (1.2, 12.))
def test_cylinder_energy_and_truncation(pol, eps):
    a = cylinder_cross_sections(.15, eps, 1., pol, orders=32)
    b = cylinder_cross_sections(.15, eps, 1., pol, orders=64)
    # Measured truncation change 0; largest relative energy mismatch 4.02e-15.
    np.testing.assert_allclose(a, b, rtol=1e-13, atol=0)
    np.testing.assert_allclose(a[0], a[1], rtol=1e-13, atol=0)


@pytest.mark.parametrize('pol', ('TM', 'TE'))
@pytest.mark.parametrize('eps', (1.2, 12.))
def test_cylinder_coefficients_match_the_radial_wave_equation(pol, eps):
    """Integrate the interior radial ODE without using its Bessel closed form."""
    from scipy.integrate import solve_ivp
    radius, k = .15, 2*math.pi
    # At r0/R=1e-6, DOP853 rtol=atol=1e-12, the 16 selected coefficients
    # differ by at most 5.76e-12. The 1e-10 budget is 17 times that measurement.
    r0 = radius*1e-6
    coefficients = cylinder_coefficients(radius, eps, 1., pol)
    for n in (0, 1, 2, 4):
        def rhs(r, y):
            return (y[1], -y[1]/r-(eps*k*k-n*n/r**2)*y[0])
        sol = solve_ivp(rhs, (r0, radius), (1., n/r0), method='DOP853',
                        rtol=1e-12, atol=1e-12)
        assert sol.success, sol.message
        derivative = sol.y[1, -1]/sol.y[0, -1]
        weight = 1. if pol == 'TM' else 1/eps
        x = k*radius
        reference = -(k*special.jvp(n, x)-weight*derivative*special.jv(n, x)) / (
            k*special.h1vp(n, x)-weight*derivative*special.hankel1(n, x))
        np.testing.assert_allclose(coefficients[32+n], reference, rtol=0, atol=1e-10)
