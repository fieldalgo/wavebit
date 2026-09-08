"""Differentiable rectangular Stack adapter for the shared Yee time-domain core.

Geometry is static; raster epsilon, mu and electric pole values retain their
backend graph. A uniform incident-medium run calibrates each current-sheet basis.
No external implementation is used here.
"""
import math

import numpy as np

from .backend import _concrete
from .conventions import boundary_kz, incidence_clashes, incident_sp_amplitudes, order_azimuth, parse_order_pair, parse_period_mapping
from .domain import as_material
from .grid_fdtd import fdtd_grid
from .result import Result, ResultSub
from ..backends.time_domain import time_loop_backend


def _scattering_change_bound(xp, columns):
    """Maximum component change for unit Jones inputs; zero subgradient at zero."""
    squared = xp.sum(xp.abs(columns)**2, axis=-1)
    zero = squared == 0
    return xp.where(zero, xp.zeros_like(squared),
                    xp.sqrt(xp.where(zero, xp.ones_like(squared), squared)))


def _diffraction_channels(band, kx, ky, index, dt, dx, dy):
    """Static continuum and Yee transverse-cutoff data for a lossless half space.

    Positive Yee cutoff parameter is the lower edge of the longitudinal
    passband, not a guarantee against longitudinal Nyquist or mesh reflections.
    """
    k0 = 2*np.pi/band[:, None, None]
    transverse = (kx/k0)**2+(ky/k0)**2
    q2 = index**2-(kx/k0)**2-(ky/k0)**2
    grazing = ((abs(q2) < 1e-16)
               | (abs(q2) <= 32*np.finfo(float).eps*(index**2+transverse)))
    if np.any(grazing):
        raise ValueError('A retained diffraction order is exactly grazing; adjust wavelength/window.')
    q = boundary_kz(np, index**2, transverse.astype(complex))
    yee = (2/k0)**2*((index*np.sin(k0*dt/2)/dt)**2
                       -(np.sin(kx*dx/2)/dx)**2-(np.sin(ky*dy/2)/dy)**2)
    length = np.hypot(kx, ky)
    cutoff = np.divide(2*np.pi*index, length, out=np.full_like(length, np.inf), where=length > 0)
    return dict(q=q, propagating=q2 > 0, cutoff_wavelength=cutoff,
                yee_cutoff_parameter=yee, cutoff_resolved=(q2 > 0) == (yee > 0),
                p_projection_gain=index/abs(q))


def _directional_projection(band, kx, ky, epsilon, mu, dt, dx, dy, dz):
    """Map co-located tangential H to E+ - E- on a uniform Yee cell pair.

    Transverse interpolation is already removed. H retains its z interpolation;
    this matrix removes that factor and uses discrete Maxwell admittance. Static
    geometry/half spaces stay in NumPy; application preserves backend field graphs.
    """
    omega = 2*np.pi/band[:, None, None]
    if np.any(omega*dt >= np.pi):
        raise ValueError('Directional diffraction requires frequencies below temporal Nyquist.')
    frequency = 2*np.sin(omega*dt/2)/dt
    x, y = 2*np.sin(kx*dx/2)/dx, 2*np.sin(ky*dy/2)/dy
    transverse = (x/frequency)**2+(y/frequency)**2
    square = epsilon*mu-transverse
    if np.any(abs(square) <= 32*np.finfo(float).eps*(epsilon*mu+transverse)):
        raise ValueError('Directional diffraction is singular at a discrete cutoff; refine the grid or shift wavelength.')
    z = frequency*boundary_kz(np, epsilon*mu, transverse.astype(complex))
    if np.any(dz*z.real/2 >= 1-32*np.finfo(float).eps):
        raise ValueError('Directional diffraction reaches longitudinal Nyquist; refine the grid.')
    interpolation = np.cos(np.arcsin(dz*z/2))
    scale = epsilon*frequency*z*interpolation
    return np.stack((np.stack((np.broadcast_to(x*y, scale.shape), z*z+y*y), axis=1),
                     np.stack((-(z*z+x*x), np.broadcast_to(-x*y, scale.shape)), axis=1)), axis=1)/scale[:, None, None]


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
    return result, diagnostics


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
                     diffraction_method='electric'):
    """Solve a static rectangular scene with NumPy/JAX/Torch material gradients.

    Explicit resolution is required: optimizing materials must not resize the
    grid. Half spaces, geometry, wavelength and incidence angles are static.
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
            up = graded_cells(core[0], coarse(core[0], nt), interface*dz, grading)[::-1]
            down = graded_cells(core[-1], coarse(core[-1], nb), (nz-bottom_interface)*dz, grading)
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
    if diffraction_distance is not None:
        reflect_z = int(np.argmin(abs(z+distances[0])))
        transmit_z = int(np.argmin(abs(z-(thick.sum()+distances[1]))))
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
            for axis, centres, width, edges, periodic in (
                (2, (np.arange(nx)+offsets[0])*dx, dx, np.linspace(0, lx, raster.shape[2]+1), True),
                (1, (np.arange(ny)+offsets[1])*dy, dy, np.linspace(0, ly, raster.shape[1]+1), True)):
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
    if n_periods*centre < 10*width+2*(z[transmit_z]-z[source_z])*max(nt, nb):
        raise ValueError('n_periods is too short for the pulse and a round trip across the scene.')
    steps = math.ceil(n_periods*centre/dt)
    time = np.arange(steps)*dt
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
    # Refuse singular retained channels before any of the four time-domain runs.
    reflection_channels = _diffraction_channels(band, kx, ky, nt, dt, dx, dy)
    transmission_channels = _diffraction_channels(band, kx, ky, nb, dt, dx, dy)
    # Declared scalar TE/TM uses the fixed x-z frame, like RCWA's scalar
    # solver. Coupled/Jones scenes use each outgoing order's own s/p frame.
    azimuth = (np.zeros_like(kx) if tetm is not None and raster.shape[1] == 1 and phi == 0
               else order_azimuth(np, ky.astype(complex), kx.astype(complex), phi).reshape(kx.shape))
    co, si = array(np.cos(azimuth)), array(np.sin(azimuth))
    # Raster coordinates start at zero, as in the RCWA Fourier coefficients.
    transform = array(np.exp(-1j*(kx[..., None, None]*np.arange(nx)[None, :]*dx
                                  + ky[..., None, None]*np.arange(ny)[:, None]*dy)))
    correction = np.stack((np.cos(kx*dx/2), np.cos(ky*dy/2)))
    if np.min(np.abs(correction)) < 1e-8:
        raise ValueError('A retained order lies at the Yee interpolation Nyquist zero.')
    correction = array(correction)
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
                matrix = _directional_projection(band, kx, ky, eps, mu_value, dt, dx, dy, widths[index])
                if record_field is not None:
                    # Only the incident order at the extra carrier is used for
                    # time-record calibration. Unrequested orders may be grazing.
                    carrier = np.zeros((1, 2, 2)+kx.shape, complex)
                    if kind == 'reference' and index == reflect_z:
                        value = _directional_projection(np.array([centre]), kx[halfy:halfy+1, halfx:halfx+1],
                                                         ky[halfy:halfy+1, halfx:halfx+1], eps, mu_value,
                                                         dt, dx, dy, widths[index])
                        carrier[..., halfy:halfy+1, halfx:halfx+1] = value
                    matrix = np.concatenate((matrix, carrier))
                planes.append(matrix)
            projections[kind] = array(np.stack(planes, axis=1))
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
    stable = array(True)
    for vector in ((-np.sin(phi), np.cos(phi)), (-np.cos(phi), -np.sin(phi))):
        source = np.zeros((3,)+shape, complex)
        for c in range(2):
            xx = (np.arange(nx)+(.5 if c == 0 else 0))*dx
            yy = (np.arange(ny)+(.5 if c == 1 else 0))*dy
            source[c, source_z] = vector[c]*np.exp(1j*(kx0*xx[None, :]+ky0*yy[:, None]))
        settings = dict(spacing=(dz, dy, dx), dt=dt, steps=steps, source=source, pulse=pulse,
                        wavelengths=run_band, monitor_z=monitor_indices, backend=backend,
                        device=device, checkpoint_steps=checkpoint_steps, pml_cells=int(n_pml),
                        convergence_window=convergence_window, convergence_tol=convergence_tol, dft_subsample=dft_subsample,
                        pml_wavelength=centre, z_spacing=widths if nonuniform else None, bloch_phase=(0., ky0*ly, kx0*lx))
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
        qtop = np.sqrt((nt**2-(kx/k0)**2-(ky/k0)**2).astype(complex))
        qbot = np.sqrt((nb**2-(kx/k0)**2-(ky/k0)**2).astype(complex))
        qi = float(np.sqrt(nt**2-(kx0/k0)**2-(ky0/k0)**2))
        def propagation(index, kx_value, ky_value):
            transverse = (np.sin(kx_value*dx/2)/dx)**2+(np.sin(ky_value*dy/2)/dy)**2
            square = (index*np.sin(k0*dt/2)/dt)**2-transverse
            return 2/dz*np.arcsin(dz*np.sqrt(np.asarray(square, complex)))
        kz_top, kz_bot = array(propagation(nt, kx, ky)), array(propagation(nb, kx, ky))
        kz_inc = propagation(nt, kx0, ky0)
        phase_ref, phase_trn = kz_top*z[reflect_z], kz_bot*(z[transmit_z]-thick.sum())
        phase_inc = kz_inc*z[reflect_z]
        if nonuniform:
            def path_phase(index, kx_value, ky_value, start, stop):
                segment = widths[min(start, stop):max(start, stop)]
                transverse = (np.sin(kx_value*dx/2)/dx)**2+(np.sin(ky_value*dy/2)/dy)**2
                square = (index*np.sin(k0*dt/2)/dt)**2-transverse
                angles = 2*np.arcsin(segment.reshape((-1,)+(1,)*np.ndim(square))*np.sqrt(np.asarray(square, complex)))
                return np.sign(stop-start)*angles.sum(axis=0)
            phase_ref = array(path_phase(nt, kx, ky, interface, reflect_z))
            phase_trn = array(path_phase(nb, kx, ky, bottom_interface, transmit_z))
            phase_inc = path_phase(nt, kx0, ky0, interface, reflect_z)
        qt, qb = array(qtop), array(qbot)
        def extract(run_values, reference_scale=None):
            incident_columns, outgoing, incoming = [], [], []
            for ref, act in run_values:
                reference = ref[f, 0, 0] if diffraction_method == 'directional' else ref[f, 0]
                inc = reference[:, halfy, halfx]*np.exp(-1j*phase_inc)
                incident_columns.append(xp.stack((-np.sin(phi)*inc[0]+np.cos(phi)*inc[1],
                                                  -nt/qi*(np.cos(phi)*inc[0]+np.sin(phi)*inc[1]))))
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
        weights.append(calibration @ xp.stack((a_s, a_p)))
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
                           pulse=dict(width=width, peak_time=5*width, frequency=1/centre,
                                      relative_spectrum=array(np.exp(-(np.pi*width*(1/band-1/centre))**2))),
                           z_nodes=array(z), z_spacing=array(widths), grading=grading, align_layers=bool(align_layers),
                           n_pml=int(n_pml), source_z=z[source_z], layer_planes=np.r_[0., np.cumsum(thick)],
                           # The final closing cell also belongs to the lower PML.
                           pml_bounds=(z[0], z[int(n_pml)], z[-int(n_pml)], z[-1]+widths[-1]),
                           angles=np.arcsin(band/centre*np.sin(theta)),
                           normalization='uniform-medium reference; two electric-current bases')
        weights = xp.stack(weights)
        diagnostics['diffraction'] = dict(
            method=diffraction_method,
            order_x=array(mx), order_y=array(my), kx=array(kx), ky=array(ky),
            reflection_monitor_z=float(z[reflect_z]), transmission_monitor_z=float(z[transmit_z]),
            reflection={key: array(value) for key, value in reflection_channels.items()},
            transmission={key: array(value) for key, value in transmission_channels.items()},
            cutoff_resolved=array(np.all(reflection_channels['cutoff_resolved'])
                                  and np.all(transmission_channels['cutoff_resolved'])))
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
