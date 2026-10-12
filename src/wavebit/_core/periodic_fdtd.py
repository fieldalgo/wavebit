"""Public RCWA conventions at the boundary of the two periodic Yee engines.

Private Yee scene kernels retain their historical start/end port names. Public
layers and coordinates are bottom-to-top, with the cell centre as phase origin.
The grid-level and general ``wavebit.fd`` APIs do not pass through this adapter.
"""
import numpy as np

from .backend import _device_of, array_namespace, as_array
from .conventions import centre_parity, order_frame_sign, parse_period_mapping
from .domain import as_material


_SIGNS = dict(ex=1, ey=1, ez=-1, hx=-1, hy=-1, hz=1)
_AMPLITUDES = ('r_s', 'r_p', 't_s', 't_p')


def _legacy_inputs(options, side):
    """One-way NumPy engine: order the layers from the illuminated face."""
    options = dict(options, incidence_side='top')
    if side == 'bottom':
        options['top'], options['bot'] = options['bot'], options['top']
        return options
    for name in ('structure', 'thickness', 'mu', 'dispersion'):
        value = options.get(name)
        if value is not None:
            options[name] = value[::-1] if isinstance(value, (list, tuple)) else np.flip(value, axis=0)
    if options.get('field_monitors') is not None:
        depth = float(np.sum(options['thickness']))
        options['field_monitors'] = {name: depth-z for name, z in options['field_monitors'].items()}
    return options


def _legacy_diagnostics(diagnostics, options, side):
    """Publish coordinates and vector fields; keep the diagnostic grid private in meaning."""
    d = diagnostics
    g = d['grid']
    depth = float(np.sum(options['thickness']))
    reflect = side == 'top'
    coordinate = (lambda z: depth-z) if reflect else (lambda z: z)
    source = .5*(g.z_node[g.k_tfsf-1]+g.z_node[g.k_tfsf])
    d['source_z'] = coordinate(source)
    d['layer_planes'] = np.r_[0., np.cumsum(np.asarray(options['thickness'], dtype=float))]
    d['reflection_z'] = depth if reflect else 0.
    d['transmission_z'] = 0. if reflect else depth
    nodes = g.z_node
    bounds = (nodes[0], nodes[g.n_pml], nodes[-1-g.n_pml], nodes[-1])
    d['pml_bounds'] = tuple(coordinate(z) for z in (reversed(bounds) if reflect else bounds))
    d['z_nodes'] = depth-nodes[::-1] if reflect else nodes.copy()
    d['grid_coordinate_convention'] = 'private distance from incident interface'
    # The private engine reports degrees from kx alone; the public band uses
    # the polar angle in radians, including conical and magnetic incidence.
    band = np.atleast_1d(np.asarray(options['wavelength'], dtype=float))
    d['angles'] = np.arcsin(np.clip(np.sin(g.theta)*band/g.wavelength, -1., 1.))
    d['angles_deg'] = np.rad2deg(d['angles'])
    d['azimuth'] = g.phi
    if not reflect:
        return d
    for name, field in d.get('fields', {}).items():
        field['values'] = _SIGNS[name]*field['values'][:, ::-1]
        field['z'] = depth-field['z'][::-1]
    if 'field_z' in d:
        d['field_z'] = depth-d['field_z'][::-1]
        d['field_e'] = _SIGNS[d['field_e_component']]*d['field_e'][:, ::-1]
    if 'field_ex' in d:
        d['field_ex'] = d['field_ex'][:, ::-1]
    for plane in d.get('monitors', {}).values():
        plane['z'] = depth-plane['z']
        plane['fields'] = {name: _SIGNS[name]*value for name, value in plane['fields'].items()}
        plane['flux_density'] = plane['flux_density']*np.array([1., 1., -1.])
        plane['power_z'] = -plane['power_z']
    return d


def _public_result(result, options, legacy, invariant_y=False):
    """Centre-referenced Fresnel amplitudes, in each order's own s/p frame."""
    answers = result if isinstance(result, list) else [result]
    band = np.atleast_1d(np.asarray(options['wavelength'], dtype=float))
    centre = float((band.min()+band.max())/2)
    period = np.atleast_1d(parse_period_mapping(options.get('period')))
    sample = answers[0].R
    xp = array_namespace(sample)
    def array(value):
        return as_array(xp, value, device=_device_of(sample))
    side = options.get('incidence_side', 'top')
    medium = as_material(options['top' if side == 'top' else 'bot'])
    index = xp.sqrt(array(medium.epsilon)*array(1. if medium.mu is None else medium.mu))
    theta = array(0. if options.get('theta') is None else options['theta'])
    phi = array(0. if options.get('phi') is None else options['phi'])
    kx = 2*np.pi/centre*index*xp.sin(theta)*xp.cos(phi)
    ky = 2*np.pi/centre*index*xp.sin(theta)*xp.sin(phi)
    scalar = (legacy and options.get('tetm') is not None
              and np.shape(options['structure'])[1] == 1)
    # The legacy vector source uses phi itself, while the public incident s/p
    # frame follows k_parallel: a negative tilt reverses both basis vectors.
    incoming_frame = order_frame_sign(xp, xp.sin(theta)) if legacy else 1.
    for answer, wavelength in zip(answers, band):
        odd = array(centre_parity(*answer.order_window))
        frame = incoming_frame
        if scalar:
            labels = array(answer.orders_u)
            frame = order_frame_sign(xp, kx+2*np.pi*labels/period[0])*order_frame_sign(xp, kx)
        for sub in (answer.incident, answer.s_inc, answer.p_inc):
            if sub is None:
                continue
            for name in _AMPLITUDES:
                value = getattr(sub, name)
                if value is not None:
                    setattr(sub, name, xp.where(odd, -value, value)*frame)
            # Both engines require a positive, real incident half-space. There
            # is no incident/reflected interference contribution to net flux.
            sub.power_entering = 1-xp.sum(sub.R)
    # A plane's field (unlike an amplitude ratio) also carries the incident
    # phase. Existing arrays label x/y from the centre but excite from a corner.
    # An invariant-y record is the raw y=0 slice, published at y=0. Only
    # a resolved y axis is translated from the corner to the cell centre.
    offset_y = period[1]/2 if len(period) > 1 and not invariant_y else 0.
    phase = xp.exp(-1j*(kx*period[0]/2+ky*offset_y))
    if scalar:
        phase = phase*order_frame_sign(xp, kx)
        if options['tetm'] == 'tm':
            # Scalar TM drives +Ex; a unit incident p field has transverse
            # electric component -cos(theta), as in the Jones combiner.
            phase = phase*(-xp.cos(theta))
    else:
        phase = phase*incoming_frame
    return phase


def _centre_fields(diagnostics, phase):
    for field in diagnostics.get('fields', {}).values():
        field['values'] = field['values']*phase
    for name in ('field_e', 'field_ex'):
        if name in diagnostics:
            diagnostics[name] = diagnostics[name]*phase
    for plane in diagnostics.get('monitors', {}).values():
        plane['fields'] = {name: value*phase for name, value in plane['fields'].items()}
        scale = array_namespace(phase).abs(phase)**2
        plane['flux_density'] = plane['flux_density']*scale
        plane['power_z'] = plane['power_z']*scale



def _centre_amplitude_diagnostics(diagnostics, result):
    answer = result[0] if isinstance(result, list) else result
    sample = answer.R
    xp = array_namespace(sample)
    odd = as_array(xp, centre_parity(*answer.order_window), device=_device_of(sample))
    def convert(mapping):
        for key, value in mapping.items():
            if key in _AMPLITUDES:
                mapping[key] = xp.where(odd, -value, value)
    for mapping in diagnostics.get('diffraction', {}).get('incoming', {}).values():
        convert(mapping)
    convergence = diagnostics.get('convergence', {})
    convert(convergence.get('amplitude_change', {}))
    for leg in ('s_inc', 'p_inc'):
        convert(convergence.get(leg, {}).get('amplitude_change', {}))


def solve_periodic_fdtd(options, backend=None):
    """Adapt the public scene once; private time stepping remains unchanged."""
    options = dict(options)
    missing = [name for name in ('structure', 'thickness', 'top', 'bot') if name not in options]
    if missing:
        raise TypeError(f'fdtd: missing required scene arguments: {", ".join(missing)}')
    side = options.get('incidence_side', 'top')
    if side not in ('top', 'bottom'):
        raise ValueError("incidence_side must be 'top' or 'bottom'.")
    # Match RCWA's default pure-s illumination, including Incidence carriers.
    if all(options.get(name) is None for name in ('tetm', 'psi', 'chi', 'polarization')):
        options['psi'] = np.pi/2
    legacy = backend is None
    if legacy:
        from .fdtd import solve_fdtd
        raw = solve_fdtd(**_legacy_inputs(options, side))
    else:
        from .scene_fdtd import solve_scene_fdtd
        # Internal start/end ports run along increasing z. This is the same
        # binding as RCWA's kernels: start is the public lower half-space.
        internal = dict(options, top=options['bot'], bot=options['top'],
                        incidence_side='bottom' if side == 'top' else 'top', _public_order_frame=True)
        if options.get('mesh_indices') is not None:
            internal['mesh_indices'] = options['mesh_indices'][::-1]
        raw = solve_scene_fdtd(backend=backend, **internal)
    result, diagnostics = raw if options.get('return_diagnostics') else (raw, None)
    invariant_y = False
    if diagnostics is not None:
        invariant_y = diagnostics['grid'].ny == 1 if legacy else diagnostics['invariant_y']
    phase = _public_result(result, options, legacy, invariant_y=invariant_y)
    if diagnostics is None:
        return result
    if legacy:
        diagnostics = _legacy_diagnostics(diagnostics, options, side)
    else:
        if diagnostics.get('mesh_indices') is not None:
            diagnostics['mesh_indices'] = diagnostics['mesh_indices'][::-1]
        for plane in diagnostics.get('monitors', {}).values():
            plane['normalization'] = f'unit incident E at {side} interface'
    diagnostics['invariant_y'] = invariant_y
    diagnostics['incidence_side'] = side
    diagnostics['incident_direction'] = -1 if side == 'top' else 1
    diagnostics['coordinate_convention'] = 'right-handed, +z up, bottom interface z=0'
    diagnostics['lateral_origin'] = 'cell_centre'
    _centre_fields(diagnostics, phase)
    _centre_amplitude_diagnostics(diagnostics, result)
    return result, diagnostics
