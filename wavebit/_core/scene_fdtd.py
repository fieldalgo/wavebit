"""Differentiable rectangular Stack adapter for the shared Yee time-domain core.

Geometry is static; incidence theta/phi, raster/half-space epsilon, mu and electric pole values
retain their backend graph. A uniform incident-medium run calibrates each current-sheet basis.
No external implementation is used here.
"""
import math

import numpy as np

from .backend import _concrete
from .conventions import boundary_kz, incidence_clashes, incident_sp_amplitudes, parse_order_pair, parse_period_mapping, real_order_frame, time_delay_phase, yee_longitudinal_phase
from .domain import as_material
from .grid_fdtd import _validate_pml_reflection, fdtd_grid
from .result import Result, ResultSub
from ..backends.time_domain import time_loop_backend


def _notched_pulse(pulse, dt, frequencies, notches, epsilon, carrier, record=False):
    """Static causal FIR shaping; finite-sum spectra guard incident calibration.

    Each zero adds one delayed sample. Reserve those samples before filtering,
    rather than dropping a convolution tail and filling the intended zero.
    Host NumPy is intentional: scene source/mesh parameters are static.
    """
    values = np.asarray(notches)
    if (values.ndim > 1 or not values.size or values.dtype.kind not in 'fiu'
            or not np.isfinite(values).all() or np.any(values <= 0)):
        raise ValueError('pulse_notches must be a positive finite static frequency or nonempty frequency vector.')
    values = np.atleast_1d(values).astype(float)
    if len(values) > 2:
        raise ValueError('pulse_notches supports at most two zeros; repeated differences amplify pulse truncation and roundoff.')
    if not np.isfinite(values).all() or np.any(values >= .5/dt):
        raise ValueError('pulse_notches must lie below temporal Nyquist.')
    if len(values) >= len(pulse)-1:
        raise ValueError('pulse_notches requires more time samples; increase n_periods.')
    checked = np.r_[frequencies, carrier] if record else frequencies
    if np.any(abs(checked[:, None]-values) <= 32*np.finfo(float).eps*np.maximum(checked[:, None], values)):
        raise ValueError('pulse_notches must avoid requested frequencies and the time-record carrier.')
    result = pulse[:-len(values)].copy()
    for frequency in values:
        result = np.convolve(result, [1., -time_delay_phase(np, frequency, dt)])
        scale = np.max(abs(result))
        if not np.isfinite(scale) or scale == 0:
            raise ValueError('pulse_notches produced an unresolved pulse; reduce the number of notches.')
        result /= scale
    # Check the actual source precision, including rounding after cancellation.
    result = result.astype(np.complex64 if epsilon > 1e-10 else np.complex128)
    samples = result.astype(np.complex128)
    norm = np.sum(abs(samples))
    time = np.arange(len(samples))*dt
    spectrum = np.array([abs(np.sum(samples*np.exp(2j*np.pi*f*time)))/norm
                         for f in np.r_[frequencies, carrier]])
    required = spectrum if record else spectrum[:-1]
    if not np.isfinite(spectrum).all() or np.any(required <= 100*epsilon):
        raise ValueError('pulse_notches leaves a required frequency in the numerical source tail; move or remove notches.')
    return result, dict(notches=values, relative_spectrum=spectrum[:-1],
                        spectrum_kind='sampled_l1', carrier_fraction=spectrum[-1],
                        sampled_peak_time=float(time[np.argmax(abs(samples))]))


def _scattering_change_bound(xp, columns):
    """Maximum component change for unit Jones inputs; zero subgradient at zero."""
    squared = xp.sum(xp.abs(columns)**2, axis=-1)
    zero = squared == 0
    return xp.where(zero, xp.zeros_like(squared),
                    xp.sqrt(xp.where(zero, xp.ones_like(squared), squared)))


def _require_scene(valid, message):
    """Reject concrete invalid inputs; the scene accumulates abstract validity."""
    if _concrete(valid) is False:
        raise ValueError(message)


def _diffraction_channels(band, kx, ky, index, dt, dx, dy, *, xp=np, check=_require_scene):
    """Continuum and Yee transverse-cutoff data for a lossless half space.

    Positive Yee cutoff parameter is the lower edge of the longitudinal
    passband, not a guarantee against longitudinal Nyquist or mesh reflections.
    """
    k0 = 2*np.pi/band[:, None, None]
    transverse = (kx/k0)**2+(ky/k0)**2
    q2 = index**2-(kx/k0)**2-(ky/k0)**2
    grazing = ((abs(q2) < 1e-16)
               | (abs(q2) <= 32*np.finfo(float).eps*(index**2+transverse)))
    check(~xp.any(grazing), 'A retained diffraction order is exactly grazing; adjust wavelength/window.')
    q = boundary_kz(xp, index**2, transverse+0j)
    yee = (2/k0)**2*((index*xp.sin(k0*dt/2)/dt)**2
                       -(xp.sin(kx*dx/2)/dx)**2-(xp.sin(ky*dy/2)/dy)**2)
    squared = kx*kx+ky*ky
    # Avoid an unused sqrt(0) derivative at normal incidence.
    length = xp.sqrt(xp.where(squared == 0, xp.ones_like(squared), squared))
    cutoff = xp.where(squared == 0, xp.full_like(length, np.inf), 2*np.pi*index/length)
    return dict(q=q, propagating=q2 > 0, cutoff_wavelength=cutoff,
                yee_cutoff_parameter=yee, cutoff_resolved=(q2 > 0) == (yee > 0),
                p_projection_gain=index/abs(q))


def _directional_projection(band, kx, ky, epsilon, mu, dt, dx, dy, dz, *, xp=np, check=_require_scene):
    """Map co-located tangential H to E+ - E- on a uniform Yee cell pair.

    Transverse interpolation is already removed. H retains its z interpolation;
    this matrix removes that factor and uses discrete Maxwell admittance. Both
    the angle-dependent matrix and its application preserve backend graphs.
    """
    omega = 2*np.pi/band[:, None, None]
    check(~xp.any(omega*dt >= np.pi), 'Directional diffraction requires frequencies below temporal Nyquist.')
    frequency = 2*xp.sin(omega*dt/2)/dt
    x, y = 2*xp.sin(kx*dx/2)/dx, 2*xp.sin(ky*dy/2)/dy
    transverse = (x/frequency)**2+(y/frequency)**2
    square = epsilon*mu-transverse
    check(~xp.any(abs(square) <= 32*np.finfo(float).eps*(epsilon*mu+transverse)),
          'Directional diffraction is singular at a discrete cutoff; refine the grid or shift wavelength.')
    z = frequency*boundary_kz(xp, epsilon*mu, transverse+0j)
    check(~xp.any(dz*xp.real(z)/2 >= 1-32*np.finfo(float).eps),
          'Directional diffraction reaches longitudinal Nyquist; refine the grid.')
    interpolation = xp.cos(xp.asin(dz*z/2))
    scale = epsilon*frequency*z*interpolation
    return xp.stack((xp.stack((xp.broadcast_to(x*y, scale.shape), z*z+y*y), axis=1),
                     xp.stack((-(z*z+x*x), xp.broadcast_to(-x*y, scale.shape)), axis=1)), axis=1)/scale[:, None, None]


def _split_directions(xp, electric, magnetic, projection):
    """Tangential E travelling toward +z and -z, including evanescent branches."""
    derived = xp.stack(tuple(sum(projection[:, :, c, h]*magnetic[:, :, h]
                                  for h in range(2)) for c in range(2)), axis=2)
    return xp.stack(((electric+derived)/2, (electric-derived)/2), axis=2)


def _weights(centres, width, edges, periodic=False, subpixel=True):
    """Static cell overlaps; material arithmetic is performed by the backend."""
    if not subpixel:
        x = np.mod(centres, edges[-1]) if periodic else centres
        return np.eye(len(edges)-1)[np.clip(np.searchsorted(edges, x, side='right')-1, 0, len(edges)-2)]
    width = np.asarray(width)
    width = width[:, None] if width.ndim else width
    left, right = centres[:, None]-width/2, centres[:, None]+width/2
    shifts = (-edges[-1], 0., edges[-1]) if periodic else (0.,)
    weights = sum(np.maximum(0., np.minimum(right, edges[None, 1:]+s)
                             - np.maximum(left, edges[None, :-1]+s)) for s in shifts)/width
    return weights/weights.sum(axis=1, keepdims=True)


def _validate_record_bounds(bounds):
    if bounds is None:
        return None
    if not isinstance(bounds, dict) or not bounds or set(bounds)-{'x', 'y', 'z'}:
        raise ValueError('record_bounds must map x/y/z to finite (lower, upper) pairs.')
    result = {}
    for axis, pair in bounds.items():
        values = np.asarray(pair)
        if (values.shape != (2,) or values.dtype.kind not in 'fiu'
                or not np.isfinite(values).all() or values[0] > values[1]):
            raise ValueError('record_bounds requires finite real lower <= upper pairs.')
        result[axis] = tuple(float(v) for v in values)
    return result


def _bounded_indices(coordinates, bounds):
    if bounds is None:
        return np.arange(len(coordinates))
    lower, upper = bounds
    tolerance = 8*np.finfo(float).eps*max(1., abs(lower), abs(upper), np.max(np.abs(coordinates)))
    return np.flatnonzero((coordinates >= lower-tolerance) & (coordinates <= upper+tolerance))


def _solve_bottom_scene(options):
    """Reflect z about the stack midplane; E is polar and H is axial.

    Diffraction amplitudes are incidence-relative, as in RCWA, so their s/p
    coefficients do not change. Field coordinates and Cartesian components do.
    """
    runtime = time_loop_backend(options['backend'], device=options['device'],
                                reference=options['structure'])
    xp, array = runtime.xp, runtime.array
    thickness = np.asarray(options['thickness'], float)
    raster = array(options['structure'])
    if raster.ndim != 3 or min(raster.shape) < 1:
        raise ValueError('structure must have shape (layers, y, x).')
    if thickness.shape != (raster.shape[0],) or not np.isfinite(thickness).all() or np.any(thickness <= 0):
        raise ValueError('thickness must contain one positive static length per layer.')
    depth = float(thickness.sum())
    mirrored = dict(options, incidence_side='top', top=options['bot'], bot=options['top'],
                    thickness=thickness[::-1], structure=xp.flip(raster, axis=0))
    if options['mesh_indices'] is not None:
        mirrored['mesh_indices'] = _validate_mesh_indices(options['mesh_indices'])[::-1]
    if options['record_bounds'] is not None:
        mirrored['record_bounds'] = dict(options['record_bounds'])
        if 'z' in mirrored['record_bounds']:
            lower, upper = mirrored['record_bounds']['z']
            mirrored['record_bounds']['z'] = (depth-upper, depth-lower)
    if options['mu'] is not None:
        permeability = array(options['mu'])
        if permeability.shape != raster.shape:
            raise ValueError('mu must have the same raster shape as structure.')
        mirrored['mu'] = xp.flip(permeability, axis=0)
    if options['dispersion'] is not None:
        mirrored['dispersion'] = np.flip(np.asarray(options['dispersion']), axis=0)
    if options['field_monitors'] is not None:
        planes = options['field_monitors']
        if not isinstance(planes, dict) or not planes:
            raise ValueError('field_monitors must be a nonempty mapping of names to z coordinates.')
        for coordinate in planes.values():
            if np.ndim(coordinate) or np.iscomplexobj(coordinate) or not np.isfinite(coordinate):
                raise ValueError('Monitor z must be a finite real scalar.')
        mirrored['field_monitors'] = {name: depth-z for name, z in planes.items()}
    answer = solve_scene_fdtd(**mirrored)
    if not options['return_diagnostics']:
        return answer
    result, diagnostics = answer
    signs = dict(ex=1, ey=1, ez=-1, hx=-1, hy=-1, hz=1)
    for plane in diagnostics['monitors'].values():
        plane['z'] = depth-plane['z']
        plane['fields'] = {name: signs[name]*value for name, value in plane['fields'].items()}
        plane['flux_density'] = plane['flux_density']*array([1., 1., -1.])
        plane['power_z'] = -plane['power_z']
        plane['normalization'] = 'unit incident E at bottom interface'
    for name, field in diagnostics.get('fields', {}).items():
        field['values'] = signs[name]*xp.flip(field['values'], axis=1)
        field['z'] = depth-xp.flip(field['z'], axis=0)
    if 'fields' in diagnostics:
        first = diagnostics['fields'][diagnostics['field_e_component']]
        diagnostics.update(field_e=first['values'], field_z=first['z'])
    diagnostics['z_nodes'] = depth-xp.flip(diagnostics['z_nodes'], axis=0)
    # Reversed nodal grid: closing cell remains last; reverse the interior widths.
    diagnostics['z_spacing'] = xp.concat((xp.flip(diagnostics['z_spacing'][:-1], axis=0), diagnostics['z_spacing'][-1:]))
    diagnostics.update(source_z=depth-diagnostics['source_z'],
                       pml_bounds=tuple(depth-z for z in reversed(diagnostics['pml_bounds'])),
                       layer_planes=np.r_[0., np.cumsum(thickness)],
                       incidence_side='bottom', reflection_z=depth, transmission_z=0.)
    for key in ('reflection_monitor_z', 'transmission_monitor_z'):
        diagnostics['diffraction'][key] = depth-diagnostics['diffraction'][key]
    diagnostics['mesh_indices'] = diagnostics['mesh_indices'][::-1]
    diagnostics['background'] = xp.flip(diagnostics['background'], axis=0)
    return result, diagnostics


def _validate_mesh_indices(value):
    """Nominal top/bottom indices select geometry, never material coefficients."""
    message = 'mesh_indices must be two positive finite static refractive indices (top, bottom).'
    try:
        value = np.asarray(value)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(message) from error
    if (value.shape != (2,) or value.dtype.kind not in 'fiu'
            or not np.isfinite(value).all() or np.any(value <= 0)):
        raise ValueError(message)
    result = tuple(float(v) for v in value)
    if not np.isfinite(result).all():
        raise ValueError(message)
    return result


def solve_scene_fdtd(structure, thickness, top, bot, period=None, wavelength=None, *,
                     backend=0, device=None, resolution=None, n_pml=12, n_periods=40,
                     courant=.8, subpixel=True, theta=0., phi=None, tetm=None,
                     psi=None, chi=None, polarization=None, mu=None,
                     truncation_order=None, dispersion=None, poles=None,
                     pad_scale=1., incidence_side='top', lattice=None,
                     checkpoint_steps=32, return_diagnostics=False,
                     field_monitors=None, record_field=None, record_components=None, grading=None,
                     convergence_window=None, convergence_tol=1e-4, dft_subsample=1, record_bounds=None,
                     pulse_width=None, align_layers=False, diffraction_distance=None,
                     diffraction_method='electric', pml_reflection=1e-8, pulse_notches=None,
                     source_distance=None, transverse_resolution=None, mesh_indices=None):
    """Solve a rectangular scene with NumPy/JAX/Torch angle/material gradients.

    Explicit resolution is required: optimizing materials must not resize the
    grid. Geometry and wavelength are static. Half-space epsilon/mu can be live
    positive real scalars with explicit mesh_indices and truncation_order.
    mesh_indices=(n_top,n_bottom) fixes padding/grading using nominal static
    refractive indices; actual materials still determine all physics and CPML.
    Without it, host scalar half spaces retain automatic mesh selection.
    theta and phi can
    be backend real scalars; either array angle requires truncation_order to
    fix the diffraction window. Array phi always samples the full y period,
    including at zero, and requires Jones/psi polarization rather than tetm.
    Angle derivatives hold the mesh fixed and change the one Bloch wavevector
    shared by the whole wavelength band.
    Bottom incidence uses a z-reflected scene and returns fields in original
    coordinates. Optional grading builds a static nonuniform z mesh; material
    gradients do not resize it. pulse_width is the static Gaussian envelope's
    1/e amplitude half-width in c=1 time units; None uses the central wavelength.
    align_layers places every layer boundary on a z node using equal cells per
    layer, each no wider than 1/resolution. It can be combined with grading.
    diffraction_distance optionally places the extraction planes near their
    reference interfaces without resizing the padding, source or PML mesh.
    diffraction_method='directional' separates E/H into travelling directions
    using the discrete admittance; both planes need locally uniform z cells.
    pml_reflection sets a static nominal normal-incidence round-trip amplitude
    target in (0,1) on both ends; smaller means stronger absorption. It is not
    a finite-grid error bound and does not change the PML thickness.
    pulse_notches optionally suppresses selected positive frequencies using
    causal delayed copies of the Gaussian. Required frequencies must remain
    resolved, including the carrier when time records are requested.
    source_distance optionally selects a source sheet a fixed physical distance
    from the incident interface, independently of padding and monitor positions.
    transverse_resolution optionally sets static x/y cells per unit length,
    as a scalar or (x,y) pair. The z mesh continues to use resolution.
    """
    record_bounds = _validate_record_bounds(record_bounds)
    if record_bounds is not None and record_field is None:
        raise ValueError('record_bounds requires record_field.')
    if incidence_side == 'bottom':
        return _solve_bottom_scene(locals().copy())
    if (field_monitors is not None or record_field is not None) and not return_diagnostics:
        raise ValueError('field_monitors/record_field require return_diagnostics=True.')
    if convergence_window is not None and not return_diagnostics:
        raise ValueError('convergence_window requires return_diagnostics=True.')
    if field_monitors is not None and (not isinstance(field_monitors, dict) or not field_monitors):
        raise ValueError('field_monitors must be a nonempty mapping of names to z coordinates.')
    if record_components is not None and record_field is None:
        raise ValueError('record_components requires record_field.')
    if incidence_side != 'top':
        raise ValueError("incidence_side must be 'top' or 'bottom'.")
    if lattice is not None:
        raise ValueError('The common scene engine requires rectangular period.')
    if not isinstance(align_layers, (bool, np.bool_)):
        raise ValueError('align_layers must be a static boolean.')
    if diffraction_method not in ('electric', 'directional'):
        raise ValueError("diffraction_method must be 'electric' or 'directional'.")
    pml_reflection = _validate_pml_reflection(pml_reflection)
    if source_distance is not None:
        value = np.asarray(source_distance)
        if (value.shape != () or value.dtype.kind not in 'fiu'
                or not np.isfinite(value) or value <= 0):
            raise ValueError('source_distance must be a positive finite static scalar.')
        source_distance = float(value)
        if not np.isfinite(source_distance):
            raise ValueError('source_distance must be representable as a finite static scalar.')
    if diffraction_distance is not None:
        distances = np.asarray(diffraction_distance)
        if (distances.shape not in ((), (2,)) or distances.dtype.kind not in 'fiu'
                or not np.isfinite(distances).all() or np.any(distances <= 0)):
            raise ValueError('diffraction_distance must be a positive finite scalar or (reflection, transmission) pair.')
        distances = np.broadcast_to(distances, (2,))
    if grading is True:
        grading = 1.06
    elif grading is False:
        grading = None
    if grading is not None and (not np.isfinite(grading) or not 1 < grading < 2):
        raise ValueError('grading must be True or a static growth ratio between 1 and 2.')
    if resolution is None or not np.isfinite(resolution) or resolution <= 0:
        raise ValueError('The common scene engine requires a positive explicit resolution (cells/length).')
    transverse = np.asarray(resolution if transverse_resolution is None else transverse_resolution)
    if (transverse.shape not in ((), (2,)) or transverse.dtype.kind not in 'fiu'
            or not np.isfinite(transverse).all() or np.any(transverse <= 0)):
        raise ValueError('transverse_resolution must be a positive finite static scalar or (x,y) pair.')
    rx, ry = np.broadcast_to(transverse, (2,)).astype(float)
    if not np.isfinite([rx, ry]).all():
        raise ValueError('transverse_resolution must be representable as finite static values.')
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
    phi = 0. if phi is None else phi
    scalar_phi = isinstance(phi, (int, float, np.number))
    in_plane = scalar_phi and phi == 0
    # Inferred order counts may depend on a host scalar, never on a live array.
    if truncation_order is None and (not scalar_phi or not isinstance(theta, (int, float, np.number))):
        raise ValueError('Array theta or phi requires explicit truncation_order to keep the diffraction window fixed.')
    host_theta, host_phi = theta, phi
    if tetm is not None and not in_plane:
        raise ValueError('Use psi or polarization for conical incidence or array phi instead of tetm.')
    band = np.atleast_1d(np.asarray(wavelength, float))
    if band.ndim != 1 or not len(band) or not np.isfinite(band).all() or np.any(band <= 0):
        raise ValueError('wavelength must be a positive scalar or vector.')
    centre = float((band.min()+band.max())/2)
    if pulse_width is None:
        width = centre
    else:
        value = np.asarray(pulse_width)
        if value.shape != () or value.dtype.kind not in 'fiu' or not np.isfinite(value) or value <= 0:
            raise ValueError('pulse_width must be a positive finite static scalar in c=1 time units.')
        width = float(value)
    span = float(np.max(np.abs(1/band-1/centre)))
    runtime = time_loop_backend(backend, device=device, reference=structure)
    xp, array = runtime.xp, runtime.array
    theta = array(theta)
    if theta.shape != () or not xp.isdtype(theta.dtype, ('real floating', 'integral')):
        raise ValueError('theta must be a real scalar, 0 <= theta < pi/2.')
    theta = xp.astype(theta, xp.result_type(theta, array(1.)))
    phi = array(phi)
    if phi.shape != () or not xp.isdtype(phi.dtype, ('real floating', 'integral')):
        raise ValueError('phi must be a finite real scalar.')
    phi = xp.astype(phi, xp.result_type(phi, array(1.)))
    scene_valid = array(True)
    def check(valid, message):
        nonlocal scene_valid
        _require_scene(valid, message)
        scene_valid = scene_valid & valid
    check(xp.isfinite(theta) & (theta >= 0) & (theta < np.pi/2),
          'Require finite incidence angles, 0 <= theta < pi/2.')
    check(xp.isfinite(phi), 'phi must be a finite real scalar.')
    cos_phi, sin_phi = xp.cos(phi), xp.sin(phi)
    # A requested frequency in the Gaussian's numerical tail cannot provide
    # reliable incident calibration. This is a roundoff guard, not an accuracy
    # guarantee for weakly excited frequencies.
    source_epsilon = xp.finfo(xp.real(array(np.array(0j, dtype=np.complex128))).dtype).eps
    if span and width > np.sqrt(-np.log(100*source_epsilon))/(np.pi*span):
        raise ValueError('pulse_width leaves a requested wavelength in the numerical source tail; reduce pulse_width or narrow the wavelength band.')
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
    host_indices = []
    array_background = False
    for value in (top, bot):
        material = as_material(value)
        raw = (material.epsilon, 1. if material.mu is None else material.mu)
        host = all(isinstance(v, (int, float, complex, np.number)) for v in raw)
        array_background |= not host
        pair = []
        for v in raw:
            v = array(v)
            if v.shape != () or not xp.isdtype(v.dtype, ('real floating', 'complex floating', 'integral')):
                raise ValueError('Half spaces must have positive real scalar epsilon and mu.')
            real = xp.real(v) if xp.isdtype(v.dtype, 'complex floating') else v
            imaginary_ok = xp.imag(v) == 0 if xp.isdtype(v.dtype, 'complex floating') else True
            check(xp.isfinite(v) & imaginary_ok & (real > 0),
                  'Half spaces must have positive real scalar epsilon and mu.')
            pair.append(xp.astype(real, xp.result_type(real, array(1.))))
        if host:
            host_indices.append(np.sqrt(complex(raw[0]).real*complex(raw[1]).real))
        background.append(tuple(pair))
    if array_background and (mesh_indices is None or truncation_order is None):
        raise ValueError('Array half spaces require explicit mesh_indices and explicit truncation_order to keep the grid fixed.')
    mesh_indices = _validate_mesh_indices(host_indices if mesh_indices is None else mesh_indices)
    et, mt = background[0]
    eb, mb = background[1]
    nt, nb = xp.sqrt(et*mt), xp.sqrt(eb*mb)
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
    ly = float(lengths[1]) if len(lengths) == 2 else 2/ry
    nx = max(2, math.ceil(lx*rx))
    # Array azimuths must not collapse/expand the mesh while crossing zero.
    # Scalar zero retains the established two-cell y-invariant shortcut.
    ny = max(2, math.ceil(ly*ry)) if raster.shape[1] > 1 or not in_plane else 2
    dx, dy = lx/nx, ly/ny
    pad = max(6, math.ceil(pad_scale*centre/min(mesh_indices)/dz))
    source_z, reflect_z = int(n_pml)+2, int(n_pml)+2+pad
    interface = reflect_z+pad
    transmit_z = interface+math.ceil(thick.sum()/dz)+pad
    nz = transmit_z+pad+int(n_pml)+2
    z = (np.arange(nz)-interface)*dz
    widths = np.full(nz, dz)
    bottom_interface = interface+math.ceil(thick.sum()/dz)
    nonuniform = grading is not None or align_layers
    if nonuniform:
        if align_layers:
            counts = np.ceil(thick*resolution).astype(int)
            core = np.concatenate([np.full(count, t/count) for t, count in zip(thick, counts)])
        else:
            core = np.full(bottom_interface-interface, thick.sum()/(bottom_interface-interface))
        if grading is not None:
            from .fdtd import graded_cells
            coarse = lambda fine, index: max(fine, min(2*fine, band.min()/(20*index)))
            up = graded_cells(core[0], coarse(core[0], mesh_indices[0]), interface*dz, grading)[::-1]
            down = graded_cells(core[-1], coarse(core[-1], mesh_indices[1]), (nz-bottom_interface)*dz, grading)
        else:
            up = np.full(interface, dz)
            down = np.full(nz-bottom_interface, dz)
        widths = np.r_[up, core, down]
        interface, bottom_interface = len(up), len(up)+len(core)
        nz = len(widths)
        z = np.r_[0., np.cumsum(widths[:-1])]-up.sum()
        if align_layers:
            # Keep represented faces exact even after long padding accumulation.
            # Point sampling must not pick a different side due to cancellation.
            z[interface+np.r_[0, np.cumsum(counts)]] = np.r_[0., np.cumsum(thick)]
        source_z = int(n_pml)+2
        reflect_z = int(np.argmin(abs(z+pad*dz)))
        transmit_z = int(np.argmin(abs(z-(thick.sum()+pad*dz))))
        if not source_z+2 < reflect_z < interface or transmit_z >= nz-int(n_pml)-2:
            raise ValueError('Nonuniform padding cannot fit source/monitors/PML; increase pad_scale.')
    if source_distance is not None:
        source_z = int(np.argmin(abs(z+source_distance)))
        if source_z < int(n_pml)+2 or source_z >= interface:
            raise ValueError('source_distance must place the source in the incident half space outside the PML; increase padding or adjust distance.')
    if diffraction_distance is not None:
        reflect_z = int(np.argmin(abs(z+distances[0])))
        transmit_z = int(np.argmin(abs(z-(thick.sum()+distances[1]))))
    if source_distance is not None and not source_z+2 < reflect_z:
        raise ValueError('source_distance must leave at least three cells before the reflection plane; adjust source or diffraction_distance.')
    if diffraction_distance is not None:
        if (not source_z+2 < reflect_z or z[reflect_z]+widths[reflect_z]/2 > 0
                or z[transmit_z]-widths[transmit_z-1]/2 < thick.sum()
                or transmit_z >= nz-int(n_pml)-2):
            raise ValueError('diffraction_distance must select exterior planes between the source/stack and stack/PML; adjust distance, resolution or pad_scale.')
    if diffraction_method == 'directional':
        indices = np.array([reflect_z, transmit_z])
        if not np.allclose(widths[indices], widths[indices-1], rtol=1e-12, atol=0):
            raise ValueError('Directional diffraction requires locally uniform z cells at both planes; adjust diffraction_distance or grading.')
    dual = .5*(widths+np.roll(widths, 1))
    cuts = np.r_[z[0]-2*widths.max(), 0., np.cumsum(thick), z[-1]+2*widths.max()]
    shape = (nz, ny, nx)
    def sample(values, upper, lower, magnetic=False, harmonic=True):
        components = []
        for c in range(3):
            offsets = np.ones(3)*.5 if magnetic else np.zeros(3)
            offsets[c] = 0. if magnetic else .5  # physical x,y,z offsets
            v = values
            transverse_axes = [
                (2, (np.arange(nx)+offsets[0])*dx, dx, np.linspace(0, lx, raster.shape[2]+1), True),
                (1, (np.arange(ny)+offsets[1])*dy, dy, np.linspace(0, ly, raster.shape[1]+1), True)]
            # Normal harmonic averaging precedes tangential arithmetic averaging
            # in the transverse plane. A fixed x-then-y order treated Ex and Ey
            # differently at crossed raster corners and broke x/y covariance.
            if harmonic and c == 1:
                transverse_axes.reverse()
            for axis, centres, width, edges, periodic in transverse_axes:
                weights = array(_weights(centres, width, edges, periodic, subpixel), dtype=v.dtype)
                use_harmonic = harmonic and c == 2-axis
                v = 1/v if use_harmonic else v
                v = xp.moveaxis(xp.tensordot(weights, v, axes=((1,), (axis,))), 0, axis)
                v = 1/v if use_harmonic else v
            v = xp.concat((xp.ones_like(v[:1])*upper, v, xp.ones_like(v[:1])*lower), axis=0)
            if not nonuniform:
                centres, cell_widths = z+offsets[2]*dz, dz
            elif offsets[2]:
                centres, cell_widths = z+widths/2, widths
            else:
                centres = z+(widths-np.roll(widths, 1))/4
                cell_widths = dual
            if not subpixel and nonuniform:
                centres = z+offsets[2]*widths
            weights = array(_weights(centres, cell_widths, cuts, subpixel=subpixel), dtype=v.dtype)
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
    dt = courant/np.sqrt(widths.min()**-2+dy**-2+dx**-2)
    if width < dt:
        raise ValueError('pulse_width must span at least one time step; increase pulse_width or resolution.')
    required_duration = 10*width+2*(z[transmit_z]-z[source_z])*xp.maximum(nt, nb)
    check(n_periods*centre >= required_duration,
          'n_periods is too short for the pulse and a round trip across the scene.')
    steps = math.ceil(n_periods*centre/dt)
    time = np.arange(steps)*dt
    pulse = np.exp(-((time-5*width)/width)**2)*np.exp(-2j*np.pi*time/centre)
    pulse_info = dict(width=width, peak_time=5*width, frequency=1/centre,
                      notches=np.array([], dtype=float), spectrum_kind='ideal_gaussian',
                      relative_spectrum=np.exp(-(np.pi*width*(1/band-1/centre))**2))
    if pulse_notches is not None:
        pulse, shaped = _notched_pulse(pulse, dt, 1/band, pulse_notches, source_epsilon,
                                       1/centre, record=record_field is not None)
        check((steps-len(shaped['notches']))*dt >= required_duration,
              'n_periods is too short for pulse_notches and the scene round trip.')
        pulse_info.update(shaped)
    kx0, ky0 = 2*np.pi/centre*nt*xp.sin(theta)*xp.stack((cos_phi, sin_phi))
    check(xp.all((array(band)/centre)*xp.sin(theta) < 1),
          'The fixed Bloch wavevector must propagate throughout the wavelength band.')
    if truncation_order is None:
        host_kx, host_ky = 2*np.pi/centre*host_indices[0]*np.sin(host_theta)*np.array([np.cos(float(host_phi)), np.sin(float(host_phi))])
        halfx = math.floor(lx*(max(host_indices)/band.min()+abs(host_kx)/(2*np.pi)))
        halfy = math.floor(ly*(max(host_indices)/band.min()+abs(host_ky)/(2*np.pi))) if raster.shape[1] > 1 else 0
    else:
        (halfx, halfy), _ = parse_order_pair(truncation_order)
    if 2*halfx+1 > nx or 2*halfy+1 > ny:
        raise ValueError('Diffraction window exceeds the spatial Nyquist limit; increase resolution.')
    mx, my = np.meshgrid(np.arange(-halfx, halfx+1), np.arange(-halfy, halfy+1))
    kx, ky = kx0+array(2*np.pi*mx/lx), ky0+array(2*np.pi*my/ly)
    # Refuse singular retained channels before any of the four time-domain runs.
    reflection_channels = _diffraction_channels(array(band), kx, ky, nt, dt, dx, dy, xp=xp, check=check)
    transmission_channels = _diffraction_channels(array(band), kx, ky, nb, dt, dx, dy, xp=xp, check=check)
    # Declared scalar TE/TM uses the fixed x-z frame, like RCWA's scalar
    # solver. Coupled/Jones scenes use each outgoing order's own s/p frame.
    co, si = ((xp.ones_like(kx), xp.zeros_like(kx)) if tetm is not None and raster.shape[1] == 1 and in_plane
              else real_order_frame(xp, ky, kx, phi))
    # Raster coordinates start at zero, as in the RCWA Fourier coefficients.
    transform = xp.exp(-1j*(kx[..., None, None]*array(np.arange(nx)[None, :]*dx)
                            + ky[..., None, None]*array(np.arange(ny)[:, None]*dy)))
    correction = xp.stack((xp.cos(kx*dx/2), xp.cos(ky*dy/2)))
    check(xp.min(xp.abs(correction)) >= 1e-8,
          'A retained order lies at the Yee interpolation Nyquist zero.')
    def modes(spectrum, magnetic=False):
        tangential = xp.sum(spectrum[:, :, :2, None, None]*transform[None, None, None], axis=(-2, -1))/(nx*ny)
        divisor = xp.flip(correction, axis=0) if magnetic else correction
        return tangential/divisor[None, None]
    monitor_indices = [reflect_z, transmit_z]
    monitor_brackets = {}
    for name, coordinate in (field_monitors or {}).items():
        if not isinstance(name, str) or not name:
            raise ValueError('Monitor names must be nonempty strings.')
        if np.ndim(coordinate) or np.iscomplexobj(coordinate) or not np.isfinite(coordinate):
            raise ValueError('Monitor z must be a finite real scalar.')
        if coordinate < z[int(n_pml)+1] or coordinate > z[-int(n_pml)-2]:
            raise ValueError('Monitor must lie in the interior outside the PML.')
        hi = int(np.searchsorted(z, coordinate))
        lo = hi if z[hi] == coordinate else hi-1
        if lo <= source_z <= hi:
            raise ValueError('Monitor interpolation must not touch the current sheet.')
        for index in (lo, hi):
            if index not in monitor_indices:
                monitor_indices.append(index)
        monitor_brackets[name] = (monitor_indices.index(lo), monitor_indices.index(hi),
                                  0. if lo == hi else (coordinate-z[lo])/(z[hi]-z[lo]))
    # Capture the smallest common native box, then trim each staggered component.
    # The union differs from individual bounds by at most the staggering cells.
    record_region, record_trims = None, {}
    if record_bounds is not None:
        names = ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')
        selected = names if record_components is None or (isinstance(record_components, str) and record_components == 'all') else record_components
        selected = (selected,) if isinstance(selected, str) else tuple(selected)
        if not selected or len(set(selected)) != len(selected) or any(name not in names for name in selected):
            raise ValueError('record_components must name unique ex/ey/ez/hx/hy/hz components.')
        positions = {}
        for name in selected:
            offsets = np.ones(3)*.5 if name.startswith('h') else np.zeros(3)
            offsets[names.index(name)%3] = 0. if name.startswith('h') else .5
            coordinates = (z+offsets[2]*widths,
                           (np.arange(ny)+offsets[1])*dy-ly/2 if raster.shape[1] > 1 else np.array([0.]),
                           (np.arange(nx)+offsets[0])*dx-lx/2)
            positions[name] = tuple(_bounded_indices(v, record_bounds.get(axis))
                                    for axis, v in zip(('z', 'y', 'x'), coordinates))
            if any(len(v) == 0 for v in positions[name]):
                raise ValueError(f'record_bounds contains no native samples for {name}; widen bounds or refine resolution.')
        region_starts = [min(v[axis][0] for v in positions.values()) for axis in range(3)]
        record_region = tuple(slice(start, max(v[axis][-1] for v in positions.values())+1)
                              for axis, start in enumerate(region_starts))
        record_trims = {name: tuple(slice(v[0]-start, v[-1]-start+1)
                                   for v, start in zip(indices, region_starts))
                        for name, indices in positions.items()}
    # Include the carrier frequency for causal, centre-calibrated time records.
    run_band = np.r_[band, centre] if record_field is not None else band
    projections = {}
    if diffraction_method == 'directional':
        for kind, materials in (('reference', (background[0], background[0])), ('actual', background)):
            planes = []
            for index, (eps, mu_value) in zip((reflect_z, transmit_z), materials):
                matrix = _directional_projection(array(band), kx, ky, eps, mu_value, dt, dx, dy,
                                                  widths[index], xp=xp, check=check)
                if record_field is not None:
                    # Only the incident order at the extra carrier is used for
                    # time-record calibration. Unrequested orders may be grazing.
                    carrier = array(np.zeros((1, 2, 2)+kx.shape, complex))
                    if kind == 'reference' and index == reflect_z:
                        value = _directional_projection(array([centre]), kx[halfy:halfy+1, halfx:halfx+1],
                                                         ky[halfy:halfy+1, halfx:halfx+1], eps, mu_value,
                                                         dt, dx, dy, widths[index], xp=xp, check=check)
                        carrier = value*array((mx == 0) & (my == 0))
                    matrix = xp.concat((matrix, carrier))
                planes.append(matrix)
            projections[kind] = xp.stack(planes, axis=1)
    def run_modes(run, kind, prefix=False):
        data = run['convergence'] if prefix else run
        suffix = '_before_window' if prefix else ''
        electric = modes(data['electric_spectrum'+suffix])
        if diffraction_method == 'electric':
            return electric
        magnetic = modes(data['magnetic_spectrum'+suffix], magnetic=True)
        return _split_directions(xp, electric[:, :2], magnetic[:, :2], projections[kind])
    native_runs = []
    convergence_runs = []
    earlier_runs = []
    runs = []
    stable = scene_valid
    source_plane = array((np.arange(shape[0]) == source_z)[:, None, None])
    for vector in ((-sin_phi, cos_phi), (-cos_phi, -sin_phi)):
        components = []
        for c in range(2):
            xx = array((np.arange(nx)+(.5 if c == 0 else 0))*dx)
            yy = array((np.arange(ny)+(.5 if c == 1 else 0))*dy)
            components.append(source_plane*vector[c]*xp.exp(1j*(kx0*xx[None, :]+ky0*yy[:, None])))
        source = xp.stack((*components, xp.zeros_like(components[0])))
        settings = dict(spacing=(dz, dy, dx), dt=dt, steps=steps, source=source, pulse=pulse,
                        wavelengths=run_band, monitor_z=monitor_indices, backend=backend,
                        device=device, checkpoint_steps=checkpoint_steps, pml_cells=int(n_pml),
                        convergence_window=convergence_window, convergence_tol=convergence_tol, dft_subsample=dft_subsample,
                        pml_wavelength=centre, pml_reflection=pml_reflection,
                        z_spacing=widths if nonuniform else None, bloch_phase=xp.stack((array(0.), ky0*ly, kx0*lx)))
        reference = fdtd_grid(xp.ones_like(epsilon)*et, mu=mt,
                              pml_background=(background[0], background[0]), **settings)
        actual = fdtd_grid(epsilon, mu=permeability, poles=native_poles,
                           pml_background=background, record_field=record_field,
                           record_components=record_components, record_region=record_region, **settings)
        stable = stable & reference['stable'] & actual['stable']
        runs.append((run_modes(reference, 'reference'), run_modes(actual, 'actual')))
        native_runs.append(actual)
        if convergence_window is not None:
            convergence_runs.append(dict(reference=reference['convergence'], actual=actual['convergence']))
            earlier_runs.append(tuple(run_modes(run, kind, prefix=True)
                                      for run, kind in ((reference, 'reference'), (actual, 'actual'))))
    ResultType, SubType = Result, ResultSub
    if backend in (1, 'jax'):
        from ..backends.jax_backend import ResultJax, ResultSubJax
        ResultType, SubType = ResultJax, ResultSubJax
    psi = (np.pi/2 if tetm == 'te' else 0.) if tetm else (np.pi/4 if psi is None else psi)
    a_s, a_p = incident_sp_amplitudes(xp, runs[0][0].dtype, psi, chi, polarization)
    if _concrete(xp.isfinite(a_s) & xp.isfinite(a_p)) is False:
        raise ValueError('Incident polarization must be finite and nonzero.')
    results = []
    weights = []
    amplitude_changes, calibration_valid = [], []
    basis_changes, basis_calibration_valid = [], []
    incoming_columns = []
    for f, wl in enumerate(run_band):
        k0 = 2*np.pi/wl
        transverse = (kx/k0)**2+(ky/k0)**2
        qt = boundary_kz(xp, nt**2, transverse+0j)
        qb = boundary_kz(xp, nb**2, transverse+0j)
        qi = xp.real(boundary_kz(xp, nt**2, (kx0/k0)**2+(ky0/k0)**2+0j))
        frequency = 2*np.sin(k0*dt/2)/dt
        def propagation(index, kx_value, ky_value, spacing):
            transverse = (2*xp.sin(kx_value*dx/2)/(dx*frequency))**2+(2*xp.sin(ky_value*dy/2)/(dy*frequency))**2
            q = boundary_kz(xp, index**2, transverse+0j)
            return yee_longitudinal_phase(xp, q, frequency, spacing)
        kz_top, kz_bot = propagation(nt, kx, ky, dz)/dz, propagation(nb, kx, ky, dz)/dz
        kz_inc = propagation(nt, kx0, ky0, dz)/dz
        phase_ref, phase_trn = kz_top*z[reflect_z], kz_bot*(z[transmit_z]-thick.sum())
        phase_inc = kz_inc*z[reflect_z]
        if nonuniform:
            def path_phase(index, kx_value, ky_value, start, stop):
                segment = widths[min(start, stop):max(start, stop)]
                angles = propagation(index, kx_value, ky_value, array(segment.reshape((-1,)+(1,)*kx_value.ndim)))
                return np.sign(stop-start)*xp.sum(angles, axis=0)
            phase_ref = path_phase(nt, kx, ky, interface, reflect_z)
            phase_trn = path_phase(nb, kx, ky, bottom_interface, transmit_z)
            phase_inc = path_phase(nt, kx0, ky0, interface, reflect_z)
        def extract(run_values, reference_scale=None):
            incident_columns, outgoing, incoming = [], [], []
            for ref, act in run_values:
                reference = ref[f, 0, 0] if diffraction_method == 'directional' else ref[f, 0]
                inc = reference[:, halfy, halfx]*xp.exp(-1j*phase_inc)
                incident_columns.append(xp.stack((-sin_phi*inc[0]+cos_phi*inc[1],
                                                  -nt/qi*(cos_phi*inc[0]+sin_phi*inc[1]))))
                if f < len(band):
                    if diffraction_method == 'directional':
                        reflected = (act[f, 0, 1]-ref[f, 0, 1])*xp.exp(1j*phase_ref)
                        transmitted = act[f, 1, 0]*xp.exp(-1j*phase_trn)
                        returned_r = (act[f, 0, 0]-ref[f, 0, 0])*xp.exp(-1j*phase_ref)
                        returned_t = act[f, 1, 1]*xp.exp(1j*phase_trn)
                        incoming.append(xp.stack((-si*returned_r[0]+co*returned_r[1],
                                                  -nt/qt*(co*returned_r[0]+si*returned_r[1]),
                                                  -si*returned_t[0]+co*returned_t[1],
                                                  nb/qb*(co*returned_t[0]+si*returned_t[1]))))
                    else:
                        reflected = (act[f, 0]-ref[f, 0])*xp.exp(1j*phase_ref)
                        transmitted = act[f, 1]*xp.exp(-1j*phase_trn)
                    outgoing.append(xp.stack((-si*reflected[0]+co*reflected[1],
                                              nt/qt*(co*reflected[0]+si*reflected[1]),
                                              -si*transmitted[0]+co*transmitted[1],
                                              -nb/qb*(co*transmitted[0]+si*transmitted[1]))))
            incident = xp.stack(incident_columns, axis=1)
            valid = xp.all(xp.isfinite(incident)) & (xp.abs(xp.linalg.det(incident))
                     > 100*xp.finfo(xp.real(incident).dtype).eps*xp.sum(xp.abs(incident)**2))
            scale = xp.sum(xp.abs(incident)**2)
            if reference_scale is not None:
                valid = valid & (scale > (100*xp.finfo(xp.real(incident).dtype).eps)**2*reference_scale)
                incident = xp.where(valid, incident, array(np.eye(2), dtype=incident.dtype))
            calibration = xp.linalg.inv(incident)
            scattering = (xp.tensordot(xp.stack(outgoing, axis=-1), calibration, axes=((-1,), (0,)))
                          if f < len(band) else None)
            returned = (xp.tensordot(xp.stack(incoming, axis=-1), calibration, axes=((-1,), (0,)))
                        if incoming else None)
            return scattering, calibration, valid, scale, returned
        scattering, calibration, valid, scale, returned = extract(runs)
        weights.append(xp.where(stable, calibration @ xp.stack((a_s, a_p)), array(complex(float('nan'), 0.))))
        if f == len(band):
            continue
        if returned is not None:
            incoming_columns.append(xp.where(stable, returned, array(complex(float('nan'), 0.))))
        if convergence_window is not None:
            earlier, _, earlier_valid, _, _ = extract(earlier_runs, reference_scale=scale)
            difference = scattering-earlier
            basis_valid = valid & earlier_valid & xp.all(xp.isfinite(difference), axis=(0, 1, 2))
            basis_changes.append(xp.where(basis_valid[None, None, None, :], difference,
                                           array(complex(float('nan'), 0.))))
            basis_calibration_valid.append(basis_valid)
            change = difference[..., 0]*a_s+difference[..., 1]*a_p
            valid = valid & earlier_valid & xp.all(xp.isfinite(change))
            amplitude_changes.append(xp.where(valid, change, array(complex(float('nan'), 0.))))
            calibration_valid.append(valid)
        def pack(amplitudes):
            rs, rp, ts, tp = xp.where(stable, amplitudes, array(complex(float("nan"), 0.)))
            fr, ft = xp.real(qt)/qi, xp.real(qb/mb)/(qi/mt)
            r_s, r_p = fr*xp.abs(rs)**2, fr*xp.abs(rp)**2
            t_s, t_p = ft*xp.abs(ts)**2, ft*xp.abs(tp)**2
            return SubType(rs, rp, ts, tp, r_s+r_p, r_s, r_p, t_s+t_p, t_s, t_p)
        results.append(ResultType(pack(scattering[..., 0]*a_s+scattering[..., 1]*a_p),
                                  pack(scattering[..., 0]), pack(scattering[..., 1]),
                                  order_window=((0, halfx), (0, halfy))))
    results = results[:len(band)]
    result = results[0] if np.ndim(wavelength) == 0 else results
    if return_diagnostics:
        diagnostics = dict(incidence_side='top', reflection_z=0., transmission_z=float(thick.sum()),
                           stable=stable, dt=dt, steps=steps, n_steps=steps, dft_subsample=dft_subsample,
                           dft_samples=native_runs[0]['dft_samples'],
                           spacing=(float(widths.min()), dy, dx), shape=shape, wavelengths=array(band),
                           transverse_resolution=(float(rx), float(ry)),
                           mesh_indices=mesh_indices, background=array(background),
                           pulse={key: array(value) if isinstance(value, np.ndarray) else value
                                  for key, value in pulse_info.items()},
                           z_nodes=array(z), z_spacing=array(widths), grading=grading, align_layers=bool(align_layers),
                           n_pml=int(n_pml), pml_reflection=pml_reflection,
                           source_z=z[source_z], source_distance=source_distance,
                           layer_planes=np.r_[0., np.cumsum(thick)],
                           # The final closing cell also belongs to the lower PML.
                           pml_bounds=(z[0], z[int(n_pml)], z[-int(n_pml)], z[-1]+widths[-1]),
                           angles=xp.asin(array(band)/centre*xp.sin(theta)), azimuth=phi,
                           normalization='uniform-medium reference; two electric-current bases')
        weights = xp.stack(weights)
        diagnostics['diffraction'] = dict(
            method=diffraction_method,
            order_x=array(mx), order_y=array(my), kx=array(kx), ky=array(ky),
            reflection_monitor_z=float(z[reflect_z]), transmission_monitor_z=float(z[transmit_z]),
            reflection={key: array(value) for key, value in reflection_channels.items()},
            transmission={key: array(value) for key, value in transmission_channels.items()},
            cutoff_resolved=xp.all(reflection_channels['cutoff_resolved'])
                            & xp.all(transmission_channels['cutoff_resolved']))
        if incoming_columns:
            incoming = xp.stack(incoming_columns)
            diagnostics['diffraction']['incoming'] = {
                name: {key: value[:, i] for i, key in enumerate(('r_s', 'r_p', 't_s', 't_p'))}
                for name, value in (('given', incoming[..., 0]*a_s+incoming[..., 1]*a_p),
                                    ('s_inc', incoming[..., 0]), ('p_inc', incoming[..., 1]))}
        if convergence_window is not None:
            change = xp.stack(amplitude_changes)
            valid = xp.stack(calibration_valid)
            inactive = xp.all(xp.stack([run[kind]['source_inactive']
                                       for run in convergence_runs for kind in ('reference', 'actual')]))
            diagnostics['convergence'] = dict(
                bases=convergence_runs, calibration_valid=valid,
                comparison_time=(steps-convergence_window)*dt,
                amplitude_change={key: change[:, i] for i, key in enumerate(('r_s', 'r_p', 't_s', 't_p'))},
                amplitudes_settled=stable & inactive & xp.all(valid) & xp.all(xp.abs(change) <= convergence_tol),
                field_decayed=xp.all(xp.stack([run[kind]['field_decayed']
                                              for run in convergence_runs for kind in ('reference', 'actual')])))
            for key in ('material_decayed', 'state_decayed'):
                diagnostics['convergence'][key] = xp.all(xp.stack(
                    [run[kind][key] for run in convergence_runs for kind in ('reference', 'actual')]))
            columns = xp.stack(basis_changes)
            column_valid = xp.stack(basis_calibration_valid)
            for leg, name in enumerate(('s_inc', 'p_inc')):
                diagnostics['convergence'][name] = dict(
                    calibration_valid=column_valid[:, leg],
                    amplitude_change={key: columns[:, i, ..., leg]
                                      for i, key in enumerate(('r_s', 'r_p', 't_s', 't_p'))},
                    amplitudes_settled=stable & inactive & xp.all(column_valid[:, leg])
                    & xp.all(xp.abs(columns[..., leg]) <= convergence_tol))
            # A row's Euclidean norm bounds its change for every unit Jones
            # vector, including constructive interference between both bases.
            row_bound = _scattering_change_bound(xp, columns)
            diagnostics['convergence']['scattering_change_bound'] = {
                key: row_bound[:, i] for i, key in enumerate(('r_s', 'r_p', 't_s', 't_p'))}
            diagnostics['convergence']['scattering_settled'] = (
                stable & inactive & xp.all(column_valid) & xp.all(row_bound <= convergence_tol))
        monitors = {}
        for name, (lo, hi, fraction) in monitor_brackets.items():
            fields = {}
            for family, key in (('e', 'electric_spectrum'), ('h', 'magnetic_spectrum')):
                combined = sum(native[key][:len(band)]*weights[:len(band), leg, None, None, None, None]
                               for leg, native in enumerate(native_runs))
                plane = (1-fraction)*combined[:, lo]+fraction*combined[:, hi]
                if raster.shape[1] == 1:
                    plane = plane[:, :, :1]
                for component, axis in enumerate('xyz'):
                    fields[family+axis] = plane[:, component]
            electric = xp.stack([fields['e'+axis] for axis in 'xyz'], axis=-1)
            magnetic = xp.stack([fields['h'+axis] for axis in 'xyz'], axis=-1)
            density = .5*xp.real(xp.linalg.cross(electric, xp.conj(magnetic), axis=-1))
            area = lx*(ly if raster.shape[1] > 1 else 1.)
            monitors[name] = dict(fields=fields, z=float(field_monitors[name]),
                                  x=array(np.arange(nx)*dx-lx/2),
                                  y=array(np.arange(ny)*dy-ly/2 if raster.shape[1] > 1 else [0.]),
                                  wavelengths=array(band), flux_density=density,
                                  power_z=xp.mean(density[..., 2], axis=(-2, -1))*area,
                                  cell_area=area, per_unit_y=raster.shape[1] == 1,
                                  region='total', normalization='unit incident E at top interface')
        diagnostics['monitors'] = monitors
        if record_field is not None:
            # Constant centre-frequency coefficients preserve a causal waveform.
            # Frequency-by-frequency calibration cannot be applied to time samples.
            carrier_spectrum = array(np.sum(pulse*np.exp(2j*np.pi*(time+dt)/centre))*dt)
            coefficients = weights[-1]*carrier_spectrum
            fields = {}
            for name, first in native_runs[0]['fields'].items():
                values = sum(coefficients[leg]*native['fields'][name]['values']
                             for leg, native in enumerate(native_runs))
                if raster.shape[1] == 1:
                    values = values[:, :, :1]
                fields[name] = dict(first, values=values, x=first['x']-lx/2,
                                    y=first['y']-ly/2 if raster.shape[1] > 1 else array([0.]),
                                    z=first['z']+z[0])
            for name, (zs, ys, xs) in record_trims.items():
                field = fields[name]
                field.update(values=field['values'][:, zs, ys, xs],
                             z=field['z'][zs], y=field['y'][ys], x=field['x'][xs])
            primary = next((name for name in ('ey', 'ex') if name in fields), next(iter(fields)))
            first = fields[primary]
            diagnostics.update(field_e=first['values'], field_e_component=primary,
                               field_t=first['t'], field_z=first['z'], field_x=first['x'], field_y=first['y'])
            diagnostics.update(fields=fields, field_temporal_representation='positive_frequency_carrier',
                               field_normalization='unit incident E at carrier frequency',
                               field_carrier_wavelength=centre)
        return result, diagnostics
    return result
