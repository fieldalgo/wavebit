"""Differentiable rectangular Stack adapter for the shared Yee time-domain core.

Geometry is static; raster epsilon, mu and electric pole values retain their
backend graph. A uniform incident-medium run calibrates each current-sheet basis.
No external implementation is used here.
"""
import math

import numpy as np

from .backend import _concrete
from .conventions import incidence_clashes, incident_sp_amplitudes, order_azimuth, parse_order_pair, parse_period_mapping
from .domain import as_material
from .grid_fdtd import fdtd_grid
from .result import Result, ResultSub
from ..backends.time_domain import time_loop_backend


def _weights(centres, width, edges, periodic=False, subpixel=True):
    """Static cell overlaps; material arithmetic is performed by the backend."""
    if not subpixel:
        x = np.mod(centres, edges[-1]) if periodic else centres
        return np.eye(len(edges)-1)[np.clip(np.searchsorted(edges, x, side='right')-1, 0, len(edges)-2)]
    left, right = centres[:, None]-width/2, centres[:, None]+width/2
    shifts = (-edges[-1], 0., edges[-1]) if periodic else (0.,)
    weights = sum(np.maximum(0., np.minimum(right, edges[None, 1:]+s)
                             - np.maximum(left, edges[None, :-1]+s)) for s in shifts)/width
    return weights/weights.sum(axis=1, keepdims=True)


def solve_scene_fdtd(structure, thickness, top, bot, period=None, wavelength=None, *,
                     backend=0, device=None, resolution=None, n_pml=12, n_periods=40,
                     courant=.8, subpixel=True, theta=0., phi=None, tetm=None,
                     psi=None, chi=None, polarization=None, mu=None,
                     truncation_order=None, dispersion=None, poles=None,
                     pad_scale=1., incidence_side='top', lattice=None,
                     checkpoint_steps=32, return_diagnostics=False):
    """Solve a static rectangular scene with NumPy/JAX/Torch material gradients.

    Explicit resolution is required: optimizing materials must not resize the
    grid. Half spaces, geometry, wavelength and incidence angles are static.
    Time records, nonuniform grading and bottom incidence use the legacy API;
    they are not silently executed by a different backend.
    """
    if lattice is not None or incidence_side != 'top':
        raise ValueError('The common scene engine requires rectangular period and top incidence.')
    if resolution is None or not np.isfinite(resolution) or resolution <= 0:
        raise ValueError('The common scene engine requires a positive explicit resolution (cells/length).')
    if not 0 < courant < 1 or not np.isfinite(n_periods) or n_periods <= 0:
        raise ValueError('Require 0 < courant < 1 and positive n_periods.')
    if int(n_pml) != n_pml or n_pml < 2 or not np.isfinite(pad_scale) or pad_scale <= 0:
        raise ValueError('Require integer n_pml >= 2 and positive pad_scale.')
    conflicts = incidence_clashes(tetm, psi, chi, polarization)
    if conflicts:
        raise ValueError('; '.join(conflicts))
    if polarization is not None and len(polarization) != 2:
        raise ValueError('polarization must contain exactly two scalar amplitudes.')
    if tetm not in (None, 'te', 'tm'):
        raise ValueError("tetm must be 'te' or 'tm'.")
    phi = 0. if phi is None else float(phi)
    theta = float(theta)
    if tetm is not None and phi != 0:
        raise ValueError('Use psi or polarization for conical incidence instead of tetm.')
    if not np.isfinite([theta, phi]).all() or not 0 <= theta < np.pi/2:
        raise ValueError('Require finite incidence angles, 0 <= theta < pi/2.')
    band = np.atleast_1d(np.asarray(wavelength, float))
    if band.ndim != 1 or not len(band) or not np.isfinite(band).all() or np.any(band <= 0):
        raise ValueError('wavelength must be a positive scalar or vector.')
    centre = float((band.min()+band.max())/2)
    runtime = time_loop_backend(backend, device=device, reference=structure)
    xp, array = runtime.xp, runtime.array
    raster = array(structure)
    if raster.ndim != 3 or min(raster.shape) < 1:
        raise ValueError('structure must have shape (layers, y, x).')
    def real_material(value):
        value = array(value)
        real = xp.real(value) if xp.isdtype(value.dtype, 'complex floating') else value
        imaginary_ok = xp.all(xp.imag(value) == 0) if xp.isdtype(value.dtype, 'complex floating') else True
        if _concrete(xp.all(xp.isfinite(value)) & imaginary_ok & xp.all(real > 0)) is False:
            raise ValueError('Use positive real epsilon-infinity/mu; loss is specified by electric poles.')
        return xp.astype(real, xp.result_type(real, array(1.)))
    raster = real_material(raster)
    muraster = xp.ones_like(raster) if mu is None else real_material(mu)
    if muraster.shape != raster.shape:
        raise ValueError('mu must have the same raster shape as structure.')
    background = []
    for value in (top, bot):
        material = as_material(value)
        pair = [complex(material.epsilon), complex(1. if material.mu is None else material.mu)]
        if any(v.imag != 0 or not np.isfinite(v.real) or v.real <= 0 for v in pair):
            raise ValueError('Half spaces must have positive real scalar epsilon and mu.')
        background.append(tuple(v.real for v in pair))
    et, mt = background[0]
    eb, mb = background[1]
    nt, nb = np.sqrt(et*mt), np.sqrt(eb*mb)
    thick = np.asarray(thickness, float)
    if thick.shape != (raster.shape[0],) or not np.isfinite(thick).all() or np.any(thick <= 0):
        raise ValueError('thickness must contain one positive static length per layer.')
    lengths = np.atleast_1d(np.asarray(parse_period_mapping(period), float))
    if lengths.ndim != 1 or len(lengths) not in (1, 2) or not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise ValueError('period must contain one or two positive lengths.')
    if raster.shape[1] > 1 and len(lengths) != 2:
        raise ValueError('A crossed raster requires both x and y periods.')
    lx = float(lengths[0])
    dz = 1/float(resolution)
    ly = float(lengths[1]) if len(lengths) == 2 else 2*dz
    nx = max(2, math.ceil(lx*resolution))
    ny = max(2, math.ceil(ly*resolution)) if raster.shape[1] > 1 or phi != 0 else 2
    dx, dy = lx/nx, ly/ny
    pad = max(6, math.ceil(pad_scale*centre/min(nt, nb)/dz))
    source_z, reflect_z = int(n_pml)+2, int(n_pml)+2+pad
    interface = reflect_z+pad
    transmit_z = interface+math.ceil(thick.sum()/dz)+pad
    nz = transmit_z+pad+int(n_pml)+2
    z = (np.arange(nz)-interface)*dz
    cuts = np.r_[-(nz+2)*dz, 0., np.cumsum(thick), (nz+2)*dz]
    shape = (nz, ny, nx)
    def sample(values, upper, lower, magnetic=False, harmonic=True):
        components = []
        for c in range(3):
            offsets = np.ones(3)*.5 if magnetic else np.zeros(3)
            offsets[c] = 0. if magnetic else .5  # physical x,y,z offsets
            v = values
            for axis, centres, width, edges, periodic in (
                (2, (np.arange(nx)+offsets[0])*dx, dx, np.linspace(0, lx, raster.shape[2]+1), True),
                (1, (np.arange(ny)+offsets[1])*dy, dy, np.linspace(0, ly, raster.shape[1]+1), True)):
                weights = array(_weights(centres, width, edges, periodic, subpixel), dtype=v.dtype)
                use_harmonic = harmonic and c == 2-axis
                v = 1/v if use_harmonic else v
                v = xp.moveaxis(xp.tensordot(weights, v, axes=((1,), (axis,))), 0, axis)
                v = 1/v if use_harmonic else v
            v = xp.concat((xp.ones_like(v[:1])*upper, v, xp.ones_like(v[:1])*lower), axis=0)
            weights = array(_weights(z+offsets[2]*dz, dz, cuts, subpixel=subpixel), dtype=v.dtype)
            v = 1/(xp.tensordot(weights, 1/v, axes=((1,), (0,)))) if harmonic and c == 2 else xp.tensordot(weights, v, axes=((1,), (0,)))
            # Exact half-space values also make CPML material validation exact.
            v = xp.where(array((z < -dz)[:, None, None]), upper, v)
            v = xp.where(array((z > thick.sum()+dz)[:, None, None]), lower, v)
            components.append(v)
        return xp.stack(components)
    epsilon = sample(raster, et, eb)
    permeability = sample(muraster, mt, mb, magnetic=True)
    native_poles = []
    if (dispersion is None) != (poles is None):
        raise ValueError('dispersion labels and poles must be provided together.')
    if dispersion is not None:
        labels = np.asarray(dispersion)
        if labels.shape != raster.shape:
            raise ValueError('dispersion labels must match structure.')
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError('dispersion labels must be integers; zero is non-dispersive.')
        if set(np.unique(labels))-{0}-set(poles):
            raise ValueError('Every nonzero dispersion label must name a pole model.')
        if 0 in poles:
            raise ValueError('Dispersion label zero is reserved for non-dispersive cells.')
        for label, terms in poles.items():
            fill = sample(array((labels == label).astype(float)), 0., 0., harmonic=False)
            for omega, gamma, strength in terms:
                native_poles.append((omega, gamma, array(strength)*fill))
    # A static vacuum-speed CFL leaves the optimization grid/time step unchanged.
    dt = courant/np.sqrt(dz**-2+dy**-2+dx**-2)
    if n_periods*centre < 10*centre+2*(transmit_z-source_z)*dz*max(nt, nb):
        raise ValueError('n_periods is too short for the pulse and a round trip across the scene.')
    steps = math.ceil(n_periods*centre/dt)
    time = np.arange(steps)*dt
    width = centre
    pulse = np.exp(-((time-5*width)/width)**2)*np.exp(-2j*np.pi*time/centre)
    kx0, ky0 = 2*np.pi/centre*nt*np.sin(theta)*np.array([np.cos(phi), np.sin(phi)])
    if np.any((band/centre)*np.sin(theta) >= 1):
        raise ValueError('The fixed Bloch wavevector must propagate throughout the wavelength band.')
    if truncation_order is None:
        halfx = math.floor(lx*(max(nt, nb)/band.min()+abs(kx0)/(2*np.pi)))
        halfy = math.floor(ly*(max(nt, nb)/band.min()+abs(ky0)/(2*np.pi))) if raster.shape[1] > 1 else 0
    else:
        (halfx, halfy), _ = parse_order_pair(truncation_order)
    if 2*halfx+1 > nx or 2*halfy+1 > ny:
        raise ValueError('Diffraction window exceeds the spatial Nyquist limit; increase resolution.')
    mx, my = np.meshgrid(np.arange(-halfx, halfx+1), np.arange(-halfy, halfy+1))
    kx, ky = kx0+2*np.pi*mx/lx, ky0+2*np.pi*my/ly
    # Declared scalar TE/TM uses the fixed x-z frame, like RCWA's scalar
    # solver. Coupled/Jones scenes use each outgoing order's own s/p frame.
    azimuth = (np.zeros_like(kx) if tetm is not None and raster.shape[1] == 1 and phi == 0
               else order_azimuth(np, ky.astype(complex), kx.astype(complex), phi))
    co, si = array(np.cos(azimuth)), array(np.sin(azimuth))
    # Raster coordinates start at zero, as in the RCWA Fourier coefficients.
    transform = array(np.exp(-1j*(kx[..., None, None]*np.arange(nx)[None, :]*dx
                                  + ky[..., None, None]*np.arange(ny)[:, None]*dy)))
    correction = np.stack((np.cos(kx*dx/2), np.cos(ky*dy/2)))
    if np.min(np.abs(correction)) < 1e-8:
        raise ValueError('A retained order lies at the Yee interpolation Nyquist zero.')
    correction = array(correction)
    def modes(spectrum):
        tangential = xp.sum(spectrum[:, :, :2, None, None]*transform[None, None, None], axis=(-2, -1))/(nx*ny)
        return tangential/correction[None, None]
    runs = []
    stable = array(True)
    for vector in ((-np.sin(phi), np.cos(phi)), (-np.cos(phi), -np.sin(phi))):
        source = np.zeros((3,)+shape, complex)
        for c in range(2):
            xx = (np.arange(nx)+(.5 if c == 0 else 0))*dx
            yy = (np.arange(ny)+(.5 if c == 1 else 0))*dy
            source[c, source_z] = vector[c]*np.exp(1j*(kx0*xx[None, :]+ky0*yy[:, None]))
        settings = dict(spacing=(dz, dy, dx), dt=dt, steps=steps, source=source, pulse=pulse,
                        wavelengths=band, monitor_z=[reflect_z, transmit_z], backend=backend,
                        device=device, checkpoint_steps=checkpoint_steps, pml_cells=int(n_pml),
                        pml_wavelength=centre, bloch_phase=(0., ky0*ly, kx0*lx))
        reference = fdtd_grid(xp.ones_like(epsilon)*et, mu=mt,
                              pml_background=(background[0], background[0]), **settings)
        actual = fdtd_grid(epsilon, mu=permeability, poles=native_poles,
                           pml_background=background, **settings)
        stable = stable & reference['stable'] & actual['stable']
        runs.append((modes(reference['electric_spectrum']), modes(actual['electric_spectrum'])))
    ResultType, SubType = Result, ResultSub
    if backend in (1, 'jax'):
        from ..backends.jax_backend import ResultJax, ResultSubJax
        ResultType, SubType = ResultJax, ResultSubJax
    psi = (np.pi/2 if tetm == 'te' else 0.) if tetm else (np.pi/4 if psi is None else psi)
    a_s, a_p = incident_sp_amplitudes(xp, runs[0][0].dtype, psi, chi, polarization)
    if _concrete(xp.isfinite(a_s) & xp.isfinite(a_p)) is False:
        raise ValueError('Incident polarization must be finite and nonzero.')
    results = []
    for f, wl in enumerate(band):
        k0 = 2*np.pi/wl
        qtop = np.sqrt((nt**2-(kx/k0)**2-(ky/k0)**2).astype(complex))
        qbot = np.sqrt((nb**2-(kx/k0)**2-(ky/k0)**2).astype(complex))
        qi = float(np.sqrt(nt**2-(kx0/k0)**2-(ky0/k0)**2))
        def propagation(index, kx_value, ky_value):
            transverse = (np.sin(kx_value*dx/2)/dx)**2+(np.sin(ky_value*dy/2)/dy)**2
            square = (index*np.sin(k0*dt/2)/dt)**2-transverse
            return 2/dz*np.arcsin(dz*np.sqrt(np.asarray(square, complex)))
        kz_top, kz_bot = array(propagation(nt, kx, ky)), array(propagation(nb, kx, ky))
        kz_inc = propagation(nt, kx0, ky0)
        if np.min(np.abs(qtop)) < 1e-8 or np.min(np.abs(qbot)) < 1e-8:
            raise ValueError('A retained diffraction order is exactly grazing; adjust wavelength/window.')
        qt, qb = array(qtop), array(qbot)
        incident_columns, outgoing = [], []
        for ref, act in runs:
            inc = ref[f, 0, :, halfy, halfx]*np.exp(-1j*kz_inc*z[reflect_z])
            incident_columns.append(xp.stack((-np.sin(phi)*inc[0]+np.cos(phi)*inc[1],
                                              -nt/qi*(np.cos(phi)*inc[0]+np.sin(phi)*inc[1]))))
            reflected = (act[f, 0]-ref[f, 0])*xp.exp(1j*kz_top*z[reflect_z])
            transmitted = act[f, 1]*xp.exp(-1j*kz_bot*(z[transmit_z]-thick.sum()))
            outgoing.append(xp.stack((-si*reflected[0]+co*reflected[1],
                                      nt/qt*(co*reflected[0]+si*reflected[1]),
                                      -si*transmitted[0]+co*transmitted[1],
                                      -nb/qb*(co*transmitted[0]+si*transmitted[1]))))
        calibration = xp.linalg.inv(xp.stack(incident_columns, axis=1))
        scattering = xp.tensordot(xp.stack(outgoing, axis=-1), calibration, axes=((-1,), (0,)))
        def pack(amplitudes):
            rs, rp, ts, tp = xp.where(stable, amplitudes, array(complex(float("nan"), 0.)))
            fr, ft = xp.real(qt)/qi, xp.real(qb/mb)/(qi/mt)
            r_s, r_p = fr*xp.abs(rs)**2, fr*xp.abs(rp)**2
            t_s, t_p = ft*xp.abs(ts)**2, ft*xp.abs(tp)**2
            return SubType(rs, rp, ts, tp, r_s+r_p, r_s, r_p, t_s+t_p, t_s, t_p)
        results.append(ResultType(pack(scattering[..., 0]*a_s+scattering[..., 1]*a_p),
                                  pack(scattering[..., 0]), pack(scattering[..., 1]),
                                  order_window=((0, halfx), (0, halfy))))
    result = results[0] if np.ndim(wavelength) == 0 else results
    if return_diagnostics:
        return result, dict(stable=stable, dt=dt, steps=steps, spacing=(dz, dy, dx), shape=shape,
                            angles=np.arcsin(band/centre*np.sin(theta)),
                            normalization='uniform-medium reference; two electric-current bases')
    return result
