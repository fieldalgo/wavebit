"""Independent nonmagnetic Mie reference, using the exp(-i omega t) convention.

Bohren & Huffman (1983), chapter 4: coefficients (4.88), efficiencies (4.61--62),
angular amplitudes (4.74), and the logarithmic-derivative downward recurrence in
Appendix A. Positive Im(m) means absorption; xi = x (j_n + i y_n) is outgoing.
The cylinder coefficients follow continuity of u and (1/eps) du/dr for TE,
or u and du/dr for TM (chapter 8). No FDTD geometry or update code is imported.

The external coefficient pins and conventions are documented at
https://miepython.readthedocs.io/en/3.0.5/07_algorithm.html . Only the equations
and documented numerical examples are used, not a copied implementation.
"""
import math

import numpy as np
from scipy import special


# Measured at m=1.5, x=5000: padding 64 misses Qext by 2.39e-8;
# 128, 256 and 512 agree to printed double precision. Use 256 and test against
# 512 plus an independent direct-Bessel calculation on 72 smaller cases.
DOWNWARD_PADDING = 256


def sphere_coefficients(m, x, *, extra_orders=0, padding=DOWNWARD_PADDING):
    """Return a_n, b_n, n=1..N; m is the sphere/background index ratio, x=kR.

    Wiscombe's truncation estimate is N=ceil(x+4*x**(1/3)+2). Measured extension
    by 16 terms changes Qext by at most 6.69e-13 on the reference cases. This is
    a finite series, not an assertion that the truncation is identically zero.
    """
    m, x = complex(m), float(x)
    if not (math.isfinite(x) and x > 0 and math.isfinite(m.real)
            and math.isfinite(m.imag) and m.real > 0 and m.imag >= 0):
        raise ValueError('Mie reference requires finite x > 0, Re(m) > 0, Im(m) >= 0 '
                         'for exp(-i omega t); gain and perfect conductors are not covered.')
    count = math.ceil(x + 4 * x ** (1 / 3) + 2) + extra_orders
    start = max(count, math.ceil(abs(m * x))) + padding
    D = np.zeros(start + 1, dtype=complex)
    for n in range(start, 0, -1):
        t = n / (m * x)
        D[n - 1] = t - 1 / (D[n] + t)
    n = np.arange(1, count + 1)
    psi = x * special.spherical_jn(n, x)
    previous = x * special.spherical_jn(n - 1, x)
    xi = psi + 1j * x * special.spherical_yn(n, x)
    xi_previous = previous + 1j * x * special.spherical_yn(n - 1, x)
    electric = D[1:count + 1] / m + n / x
    magnetic = m * D[1:count + 1] + n / x
    a = (electric * psi - previous) / (electric * xi - xi_previous)
    b = (magnetic * psi - previous) / (magnetic * xi - xi_previous)
    return a, b


def sphere_efficiencies(m, x, **kwargs):
    """Qext, Qsca, Qabs, normalized by pi R**2; no lossless or positivity clamp."""
    a, b = sphere_coefficients(m, x, **kwargs)
    n = np.arange(1, len(a) + 1)
    qsca = 2 / x**2 * np.sum((2 * n + 1) * (abs(a)**2 + abs(b)**2))
    qext = 2 / x**2 * np.sum((2 * n + 1) * (a + b).real)
    return np.array((qext, qsca, qext - qsca))


def sphere_amplitudes(m, x, theta):
    """S1, S2 for angles in degrees, with no albedo or solid-angle normalization.

    For x-polarized incidence along +z, the far-field amplitude f=r*exp(-ikr)*E/E0
    is f_theta=i*S2*cos(phi)/k, f_phi=-i*S1*sin(phi)/k. Thus d(sigma)/d(Omega)
    is (|S2*cos(phi)|**2 + |S1*sin(phi)|**2)/k**2. In the forward direction,
    sigma_ext=4*pi*Im(f_theta)/k=4*pi*Re(S2)/k**2 (the optical theorem).
    """
    a, b = sphere_coefficients(m, x)
    mu = np.cos(np.deg2rad(theta))
    previous, current = np.zeros_like(mu), np.ones_like(mu)
    S1, S2 = np.zeros_like(mu, dtype=complex), np.zeros_like(mu, dtype=complex)
    for n, (an, bn) in enumerate(zip(a, b), 1):
        tau = n * mu * current - (n + 1) * previous
        weight = (2 * n + 1) / (n * (n + 1))
        S1 += weight * (an * current + bn * tau)
        S2 += weight * (an * tau + bn * current)
        previous, current = current, ((2 * n + 1) * mu * current
                                     - (n + 1) * previous) / n
    return S1, S2


def cylinder_coefficients(radius, eps, wavelength, polarization, orders=32):
    """c_n for n=-orders..orders: TM has axial E, TE has axial H.

    The field is sum i**n [J_n(kr)+c_n H_n(kr)] exp(i n phi). The coefficient
    sign therefore differs from the sphere's a_n convention. The truncation
    is checked by doubling 32 to 64 for the measured radius=0.15, eps<=12 band.
    """
    if polarization not in ('TM', 'TE'):
        raise ValueError("cylinder polarization is 'TM' (axial E) or 'TE' (axial H)")
    k, m = 2 * math.pi / wavelength, math.sqrt(eps)
    x = k * radius
    n = np.arange(-orders, orders + 1)
    j, dj = special.jv(n, x), special.jvp(n, x)
    h, dh = special.hankel1(n, x), special.h1vp(n, x)
    interior, derivative = special.jv(n, m * x), special.jvp(n, m * x)
    scale = m if polarization == 'TM' else 1 / m
    return -(scale * derivative * j - interior * dj) / (scale * derivative * h - interior * dh)


def cylinder_cross_sections(radius, eps, wavelength, polarization, orders=32):
    """sigma_ext, sigma_sca for the infinite cylinder, in length units."""
    c = cylinder_coefficients(radius, eps, wavelength, polarization, orders)
    factor = 4 / (2 * math.pi / wavelength)
    return np.array((-factor * np.real(c.sum()), factor * np.sum(abs(c)**2)))
