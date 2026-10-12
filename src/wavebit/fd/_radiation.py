"""Outgoing radiation of the infinite homogeneous Yee lattice.

The observer direction is the group-ray direction, parallel to sin(k*h),
whereas the electric field is transverse to K=2*sin(k*h/2)/h. Reciprocity
uses the reference wave's H-plane average as well as the recorded field's.
The scalar lattice Green function contributes the curvature of its dispersion
surface. These three factors cannot be represented by replacing continuum k.

Preparation is host-only for a concrete, positive isotropic background and
the convex first band. The array transform differentiates with respect to
all recorded fields. FarField(propagation='yee') selects this transform;
the original continuum transform and its reference arrays are preserved.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from ._post import _mean_planes, _vu


# 304 directions at Omega*h = 1e-9, 1e-6, .01, .1, .314, .9, 1.3, 1.4:
# max dispersion residual 9.55e-15 and ray residual 3.34e-16. Allow >20x
# that measured floor; near the convex-band boundary an unresolved root is
# refused at preparation, never silently used by the array transform.
MODE_RESIDUAL = 2e-13


@dataclass(frozen=True)
class Modes:
    """Prepared spectral geometry; arrays are (nf, nd, ...) unless noted."""

    shape: tuple
    k: object
    unit_K: object
    omega: object
    impedance: float
    mu: float
    green_gain: object
    flux_weight: object
    theta_vector: object
    phi_vector: object


def modes(dx, dt, freqs, eps, mu, theta, phi):
    """Prepare every frequency/direction, or refuse the complete request."""
    try:
        raw = [np.asarray(v) for v in (dx, dt, eps, mu, freqs, theta, phi)]
        if any(np.iscomplexobj(v) for v in raw):
            raise ValueError('complex preparation input')
        inputs = np.asarray(raw[:4], dtype=float)
        f = np.asarray(raw[4], dtype=float)
        th, ph = np.broadcast_arrays(np.asarray(raw[5], dtype=float), np.asarray(raw[6], dtype=float))
    except (ValueError, TypeError, RuntimeError) as error:
        raise ValueError('RadiationInputRefused: concrete real scalars, frequencies and broadcastable angles required') from error
    if (inputs.shape != (4,) or not np.all(np.isfinite(inputs) & (inputs > 0))
            or f.ndim != 1 or not f.size or not np.all(np.isfinite(f) & (f > 0))
            or not th.size or not np.all(np.isfinite(th) & np.isfinite(ph))):
        raise ValueError('RadiationInputRefused: finite positive isotropic medium, steps and frequencies required')
    # Using the original float32 scalars after validation caused a 1.402e-7
    # wavevector gap against the same representable values promoted to float64
    # (dx=.05, dt=.01, eps=2.25, mu=1.3, frequency=1, angles=37/41 degrees).
    dx, dt, eps, mu = map(float, inputs)
    omega = 2*np.pi*f
    if not np.all(omega*dt < np.pi):
        raise ValueError('RadiationTemporalAliasRefused: all frequencies must be below the time-grid Nyquist limit')
    Omega = 2*np.sin(omega*dt/2)/dt
    index = np.sqrt(eps*mu)
    s = Omega*index*dx
    # The first inflection of the lattice dispersion is k_j*h=pi/2.
    # Omega*n*h < sqrt(2) keeps its complete outgoing surface strictly convex.
    if not np.all(np.isfinite(s) & (s > 0) & (s < np.sqrt(2))):
        raise ValueError('RadiationNonconvexBandRefused: require 0 < Omega*n*dx < sqrt(2) at every frequency')
    t, p = np.radians(th.ravel()), np.radians(ph.ravel())
    ray = np.stack((np.sin(t)*np.cos(p), np.sin(t)*np.sin(p), np.cos(t)), axis=-1)
    that = np.stack((np.cos(t)*np.cos(p), np.cos(t)*np.sin(p), -np.sin(t)), axis=-1)
    phat = np.stack((-np.sin(p), np.cos(p), np.zeros_like(t)), axis=-1)
    wavevectors, directions, gains, flux = [], [], [], []
    for band in s:
        band_k, band_n, band_a, band_g = [], [], [], []
        for direction in ray:
            # sin(k*h) = band*u*ray. Scaling the unknown by band avoids
            # cancellation and underflow in the continuum (band -> 0) limit.
            upper = min(1., 1/(band*np.max(abs(direction))))
            def residual(u):
                x = band*u*direction
                # At the bracket endpoint only, 1-x^2 can round below zero.
                # The accepted root's domain is checked separately below.
                return np.sum(2*u*u*direction**2/(1+np.sqrt(np.maximum(0., 1-x*x))))-1
            end = residual(upper)
            if end < -MODE_RESIDUAL:
                raise ValueError(f'RadiationRootBracketRefused: endpoint residual {end:g}')
            # A tiny negative endpoint residual is possible when band is
            # below sqrt(machine epsilon). The final dispersion check still
            # judges this endpoint exactly like a root returned by Brent.
            u = upper if end < 0 else brentq(residual, 0., upper, xtol=np.finfo(float).tiny)
            x = band*u*direction
            if np.any(abs(x) > 1):
                raise ValueError('RadiationRootDomainRefused: outgoing phase lies outside the first branch')
            kh = np.arcsin(x)
            unit = u*direction/np.cos(kh/2)
            measured = abs(np.dot(unit, unit)-1)
            if not np.isfinite(measured) or measured > MODE_RESIDUAL:
                raise ValueError(f'RadiationRootResidualRefused: |K/(Omega*n)|^2-1 residual {measured:g}')
            d = np.cos(kh)
            curvature = (direction[0]**2*d[1]*d[2]+direction[1]**2*d[0]*d[2]
                         +direction[2]**2*d[0]*d[1])
            if not np.isfinite(curvature) or curvature <= 0:
                raise ValueError('RadiationCurvatureRefused: nonpositive outgoing surface curvature')
            band_k.append(kh/dx)
            band_n.append(unit)
            band_a.append(1/np.sqrt(curvature))
            # |K*cos(k*h/2)|/(Omega*mu) = |sin(k*h)|/(band*eta).
            # This is a flux weight, not the time-domain group velocity.
            band_g.append(u*np.linalg.norm(direction)/np.sqrt(mu/eps))
        wavevectors.append(band_k)
        directions.append(band_n)
        gains.append(band_a)
        flux.append(band_g)
    return Modes(th.shape, np.asarray(wavevectors), np.asarray(directions), Omega,
                 np.sqrt(mu/eps), float(mu), np.asarray(gains), np.asarray(flux), that, phat)


def _integral(rt, field, us, vs, wu, wv, k, u, v, P, position, dx):
    """Separable face integral, retaining each frequency's own wavevector."""
    xp = rt.xp
    a = rt.array(wu, 'float64')[None, None, :]*xp.exp(
        -1j*k[:, :, u, None]*rt.array(us, 'float64')[None, None, :])
    b = rt.array(wv, 'float64')[None, None, :]*xp.exp(
        -1j*k[:, :, v, None]*rt.array(vs, 'float64')[None, None, :])
    product = xp.matmul(a, xp.permute_dims(field, (0, 2, 1)))
    return xp.sum(product*b, axis=-1)*xp.exp(-1j*k[:, :, P]*position)*dx**2


def transform(rt, accs, faces, grid, prepared):
    """Return (E_theta, E_phi, U, E_cartesian) with prefix (nf, *angle_shape).

    E_cartesian is the coefficient of exp(i*k_star dot r)/|r|. Its radial
    component need not vanish: E is transverse to K, not to the group ray.
    U includes that component and the discrete flux weight. Accumulators
    must have the same end correction and homogeneous exterior as FarField.
    """
    xp = rt.xp
    k = rt.array(prepared.k, 'float64')
    n = rt.array(prepared.unit_K, 'float64')
    N, L = [0., 0., 0.], [0., 0., 0.]
    dx = grid.dx
    for name, sign, layout in faces:
        P, u, v = layout.P, layout.u, layout.v
        data = accs[name]
        Eu, Ev = (_vu(rt, data[label], P, u, v) for label in ('Eu', 'Ev'))
        Hu, Hv = (_vu(rt, _mean_planes(data[label], P), P, u, v) for label in ('Hu', 'Hv'))
        hu0, hu1 = layout.half[u]
        hv0, hv1 = layout.half[v]
        nu0, nu1 = layout.nodes[u]
        nv0, nv1 = layout.nodes[v]
        uh = grid.low[u]+(np.arange(hu0, hu1)+.5)*dx
        vh = grid.low[v]+(np.arange(hv0, hv1)+.5)*dx
        un = grid.low[u]+np.arange(nu0, nu1)*dx
        vn = grid.low[v]+np.arange(nv0, nv1)*dx
        wu = np.ones(len(un)) if layout.weights[u] is None else layout.weights[u]
        wv = np.ones(len(vn)) if layout.weights[v] is None else layout.weights[v]
        position = grid.low[P]+layout.k*dx
        def integrate(field, a, b, wa, wb):
            return _integral(rt, field, a, b, wa, wb, k, u, v, P, position, dx)
        N[u] = N[u]+integrate(-sign*Hv, uh, vn, np.ones(len(uh)), wv)
        N[v] = N[v]+integrate(sign*Hu, un, vh, wu, np.ones(len(vh)))
        cosine = xp.cos(k[:, :, P]*dx/2)
        L[u] = L[u]+cosine*integrate(sign*Ev, un, vh, wu, np.ones(len(vh)))
        L[v] = L[v]+cosine*integrate(-sign*Eu, uh, vn, np.ones(len(uh)), wv)
    N, L = xp.stack(N, axis=-1), xp.stack(L, axis=-1)
    cross = xp.stack((n[:, :, 1]*L[:, :, 2]-n[:, :, 2]*L[:, :, 1],
                      n[:, :, 2]*L[:, :, 0]-n[:, :, 0]*L[:, :, 2],
                      n[:, :, 0]*L[:, :, 1]-n[:, :, 1]*L[:, :, 0]), axis=-1)
    gain = (1j*rt.array(prepared.omega, 'float64')[:, None]*prepared.mu
            *rt.array(prepared.green_gain, 'float64')/(4*np.pi))
    field = gain[:, :, None]*(N-n*xp.sum(N*n, axis=-1)[:, :, None]-cross/prepared.impedance)
    theta = xp.sum(field*rt.array(prepared.theta_vector, 'float64')[None], axis=-1)
    phi = xp.sum(field*rt.array(prepared.phi_vector, 'float64')[None], axis=-1)
    power = .5*rt.array(prepared.flux_weight, 'float64')*xp.sum(xp.abs(field)**2, axis=-1)
    shape = (len(prepared.omega),)+prepared.shape
    return (xp.reshape(theta, shape), xp.reshape(phi, shape), xp.reshape(power, shape),
            xp.reshape(field, shape+(3,)))
