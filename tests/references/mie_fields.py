"""Exact exterior Mie fields, independent of the FDTD solver and its geometry.

The exp(-i omega t) component series is written from Zhai's 2006 dissertation,
Eqs. 2.40--2.45 (printed p. 22), with En=i**n*(2n+1)/(n*(n+1)) and
h_n=j_n+i*y_n. The formulas are checked against both vacuum Maxwell curls,
flux through concentric spheres, and the independent far-amplitude series.
https://oaktrust.library.tamu.edu/server/api/core/bitstreams/4aa05d10-5f44-4037-ab10-7a42dd8154d8/content
No external solver source is used.
"""
import numpy as np
from scipy import special
from .mie import sphere_coefficients, sphere_amplitudes

# At epsilon 1.1/2.25/12 and r=.250001/.3/.625/1.5, extending the existing
# Wiscombe count by 0/2/4/8 terms differs from +16 by at most
# 2.58603e-7/1.39518e-9/4.65756e-12/7.140996e-17 of peak field.
NEAR_EXTRA_ORDERS = 8


def exterior_scattered(points, eps=1.1, radius=.25, center=(0.,0.,0.), extra_orders=NEAR_EXTRA_ORDERS):
    """Cartesian scattered (E, Z0 H) for unit x-polarized exp(i k z) incidence.

    Vacuum wavelength is one; the sphere is isotropic and nonmagnetic. All
    points must be strictly exterior. This host-only oracle is never called
    in the FDTD scan. Its tested scope is documented in fd-mie-validation.md.
    """
    points, center = np.asarray(points, dtype=float), np.asarray(center, dtype=float)
    if (points.ndim < 1 or points.shape[-1] != 3 or center.shape != (3,)
            or not np.isfinite(points).all() or not np.isfinite(center).all()
            or not np.isfinite(radius) or radius <= 0
            or not np.isfinite(eps) or eps <= 0
            or not isinstance(extra_orders, int) or extra_orders < 0):
        raise ValueError('ExteriorMieInputRefused: finite exterior points, positive '
                         'radius/epsilon, and nonnegative integer extra_orders required.')
    p = points - center
    r = np.linalg.norm(p, axis=-1)
    if np.any(r <= radius):
        raise ValueError('ExteriorMieSampleRefused: every point must be outside the sphere.')
    k = 2*np.pi
    rho = k*r
    ct = p[...,2]/r
    st = np.hypot(p[...,0], p[...,1])/r
    phi = np.arctan2(p[...,1], p[...,0])
    cp,sp = np.cos(phi),np.sin(phi)
    a,b = sphere_coefficients(np.sqrt(eps), k*radius, extra_orders=extra_orders)
    previous,current = np.zeros_like(r),np.ones_like(r)
    er,et,ep,hr,ht,hp = [np.zeros_like(r,dtype=complex) for _ in range(6)]
    for n,(an,bn) in enumerate(zip(a,b),1):
        tau = n*ct*current-(n+1)*previous
        hn = special.spherical_jn(n,rho)+1j*special.spherical_yn(n,rho)
        dh = (special.spherical_jn(n,rho,derivative=True)
              +1j*special.spherical_yn(n,rho,derivative=True)+hn/rho)
        en = 1j**n*(2*n+1)/(n*(n+1))
        er += en*1j*an*n*(n+1)*st*current*hn/rho
        et += en*(1j*an*tau*dh-bn*current*hn)
        ep += en*(-1j*an*current*dh+bn*tau*hn)
        hr += en*1j*bn*n*(n+1)*st*current*hn/rho
        ht += en*(1j*bn*tau*dh-an*current*hn)
        hp += en*(1j*bn*current*dh-an*tau*hn)
        previous,current = current,((2*n+1)*ct*current-(n+1)*previous)/n
    er,et,ep = er*cp,et*cp,ep*sp
    hr,ht,hp = hr*sp,ht*sp,hp*cp
    radial = np.stack((st*cp,st*sp,ct),axis=-1)
    polar = np.stack((ct*cp,ct*sp,-st),axis=-1)
    azimuth = np.stack((-sp,cp,0*cp),axis=-1)
    phase = np.exp(1j*k*center[2])
    return tuple(phase*(v[...,None]*radial+t[...,None]*polar+f[...,None]*azimuth)
                 for v,t,f in ((er,et,ep),(hr,ht,hp)))


def far_amplitude(theta, phi, eps=1.1, radius=.25, center=(0., 0., 0.)):
    """Exact (f_theta, f_phi), f = r exp(-ikr) E for global unit incidence."""
    t, p = np.broadcast_arrays(np.radians(theta), np.radians(phi))
    s1, s2 = sphere_amplitudes(np.sqrt(eps), 2*np.pi*radius, np.degrees(t))
    rhat = np.stack((np.sin(t)*np.cos(p), np.sin(t)*np.sin(p), np.cos(t)), axis=-1)
    # Incident phase at the translated center times outgoing translation phase.
    phase = np.exp(2j*np.pi*(center[2]-rhat@np.asarray(center)))
    return (phase*1j*s2*np.cos(p)/(2*np.pi), -phase*1j*s1*np.sin(p)/(2*np.pi))
