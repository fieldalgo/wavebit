"""Single-source Yee engine with NumPy, JAX and Torch execution.

This low-level API takes native grid coefficients, not Stack/Incidence scenes.
"""
import operator

import numpy as np

from .backend import eager_under_torch_compile, _concrete
from ..backends.time_domain import time_loop_backend


def _validate_pml_reflection(value):
    """Keep CPML profile selection static and independent of material graphs."""
    value = np.asarray(value)
    message = 'pml_reflection must be a finite static scalar strictly between zero and one.'
    if value.shape != () or value.dtype.kind not in 'fiu':
        raise ValueError(message)
    value = float(value)
    if not np.isfinite(value) or not 0 < value < 1:
        raise ValueError(message)
    return value


@eager_under_torch_compile
def fdtd_grid(epsilon, *, spacing, dt, steps, mu=1., conductivity=0.,
              electric=None, magnetic=None, source=None, pulse=None,
              wavelengths=(), monitor_z=(), backend='numpy', device=None,
              checkpoint_steps=32, absorber_cells=0, poles=(),
              pml_cells=0, pml_wavelength=1., bloch_phase=(0., 0., 0.),
              pml_background=((1., 1.), (1., 1.)),
              record_field=None, record_components=None, z_spacing=None, record_region=None,
              convergence_window=None, convergence_tol=1e-4, dft_subsample=1,
              pml_reflection=1e-8):
    """Advance a 3D Yee grid and accumulate frequency-domain field planes.

    epsilon: positive scalar-per-cell (z,y,x) or diagonal/native (3,z,y,x)
    permittivity. mu and electric conductivity broadcast to that shape. E/H/source
    have shape (3,z,y,x). Source is an electric current pattern; pulse has `steps`
    samples, and the update uses curl(H)-source*pulse. E starts at t=0 and H at
    t=-dt/2. H is normalized as Z0 H. bloch_phase=(phase_z,phase_y,phase_x)
    sets F(r+L_axis)=exp(1j*phase_axis)*F(r), in radians; zeros give periodic
    boundaries. Real phases are differentiable and applied in field updates and
    monitor interpolation. With z CPML, phase_z must be zero.

    spacing is (dz,dy,dx); lengths and time use c=1. Optional static z_spacing
    overrides dz with one positive width per z index, including the last periodic
    closing cell. E nodes are cumulative widths; H z halves lie between nodes.
    Curls use primal/dual widths, monitors interpolate by physical distance, and
    z CPML uses physical end-layer thicknesses. Geometry is not differentiable.
    dt is explicit, so optimizing
    epsilon does not change the number of steps or simulation duration. Choose dt
    below the CFL bound over the whole optimization domain. Concrete inputs are
    validated; traced JAX inputs expose validity through returned `stable`.

    Optional absorber_cells adds a cubic, impedance-matched damping layer at both
    z ends. It is a graded conductivity absorber, NOT the scene solver's CPML.
    Refine its width to establish reflection accuracy. monitor_z contains integer
    z-node indices; streaming planes are collocated in space, with native E/H DFT
    timestamps. Fields are unnormalized integrals in source*time units.

    poles contains (omega0, gamma, strength) tuples, broadcastable to native E.
    They solve P'' + gamma P' + omega0**2 P = strength E, starting from zero
    polarization and its previous time level. Frequencies are angular in c=1
    units; omega0=0 gives Drude. epsilon is epsilon-infinity. Pole parameters are
    differentiable. Their oscillator frequencies tighten the returned CFL bound.

    pml_cells adds convolutional PML at both z ends, with pml_wavelength setting
    its frequency shift. pml_background gives differentiable real (epsilon, mu)
    pairs at the low/high z ends; the profile follows their refractive indices.
    End layers plus one interior cell must match these positive backgrounds, with
    no conductivity or nonzero pole strength. Cannot combine with absorber_cells.
    x/y stay periodic. Static pml_reflection in (0,1), default 1e-8, sets nominal
    round-trip amplitude attenuation at normal incidence before frequency
    shifting. Smaller values increase conductivity; this is not a measured
    reflection bound. PML accuracy needs strength, width and resolution checks.

    backend selects the shared Array API core: 0/'numpy', 1/'jax', or 2/'torch',
    matching RCWA's numeric IDs. JAX uses compiled scan; Torch uses an autograd
    loop with non-reentrant block checkpointing. Both differentiate materials,
    initial fields, sources, poles and Bloch phases. device selects hardware;
    Torch defaults to epsilon's device, or CPU for non-tensor epsilon. Unavailable
    hardware raises. Optional record_field=N keeps steps 0,N,2N,... after
    their updates. record_components selects ex/ey/ez/hx/hy/hz (default all).
    record_region optionally selects (z,y,x) integer indices or positive-step
    slices before storing each snapshot. Integer selections keep length-one
    axes; negative indices follow Python indexing. Empty regions are refused.
    Selection is static and requires record_field. It does not crop the solver.
    fields contains native values(time,z,y,x), staggered coordinates and E/H
    timestamps. Records retain gradients and allocate only requested snapshots. Torch compile is not a
    validated execution mode; JAX jit and Torch eager autograd are tested.

    dft_subsample=K evaluates DFT planes every K steps (default 1). Each
    sample represents min(K, remaining steps) updates. This saves monitor work,
    not Maxwell steps or accumulator storage; no anti-alias filter is applied.
    Refine K against 1 for transient and out-of-band signals.

    Optional convergence_window is the number of final steps inspected for pulse
    decay (positive and smaller than steps). convergence_tol is a relative
    amplitude threshold in (0,1). The returned convergence mapping reports a
    volume-weighted field norm and the final-window DFT contribution. It does
    not terminate early or certify spatial/PML convergence. Fields include the
    absorbing layers; the norm is not dispersive stored energy. Separate
    material_decay ratios track each excited ADE pole via weighted
    (omega0**2*abs(P)**2 + abs((P-P_previous)/dt)**2)/strength.
    material_decayed ignores unexcited/zero-strength poles; state_decayed
    requires field_decayed and material_decayed. CPML memory is not assessed.
    """
    from .enums import backend_of
    backend = backend_of(backend)
    runtime = time_loop_backend(backend, device=device, reference=epsilon)
    xp, array = runtime.xp, runtime.array
    names = ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')
    if record_field is not None:
        if isinstance(record_field, (bool, np.bool_)) or not isinstance(record_field, (int, np.integer)) or record_field <= 0:
            raise ValueError('record_field must be a positive integer interval.')
    if record_region is not None and record_field is None:
        raise ValueError('record_region requires record_field.')
    if record_components is not None and record_field is None:
        raise ValueError('record_components requires record_field.')
    selected = names if record_components is None or (isinstance(record_components, str) and record_components == 'all') else record_components
    selected = (selected,) if isinstance(selected, str) else tuple(selected)
    if not selected or len(set(selected)) != len(selected) or any(v not in names for v in selected):
        raise ValueError('record_components must name unique ex/ey/ez/hx/hy/hz components.')
    steps = operator.index(steps)
    if (isinstance(dft_subsample, (bool, np.bool_))
            or not isinstance(dft_subsample, (int, np.integer)) or not 1 <= dft_subsample <= steps):
        raise ValueError('dft_subsample must be a positive integer no larger than steps.')
    if convergence_window is not None:
        if (isinstance(convergence_window, (bool, np.bool_))
                or not isinstance(convergence_window, (int, np.integer))
                or not 0 < convergence_window < steps):
            raise ValueError('convergence_window must be a positive integer smaller than steps.')
    if not np.isfinite(convergence_tol) or not 0 < convergence_tol < 1:
        raise ValueError('convergence_tol must be finite and between zero and one.')
    checkpoint_steps = operator.index(checkpoint_steps)
    absorber_cells = operator.index(absorber_cells)
    pml_cells = operator.index(pml_cells)
    if pml_cells < 0 or (pml_cells and absorber_cells):
        raise ValueError("pml_cells must be nonnegative and cannot combine with absorber_cells.")
    if not np.isfinite(pml_wavelength) or pml_wavelength <= 0:
        raise ValueError("pml_wavelength must be positive and finite.")
    pml_reflection = _validate_pml_reflection(pml_reflection)
    if steps < 1 or checkpoint_steps < 1 or absorber_cells < 0:
        raise ValueError('steps/checkpoint_steps must be positive; absorber_cells nonnegative.')
    spacing = np.asarray(spacing, float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError('spacing must contain positive dz,dy,dx.')
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be positive and finite.')
    dt = float(dt)
    shape = tuple(np.shape(epsilon))
    if len(shape) == 3:
        shape = (3,)+shape
    if len(shape) != 4 or shape[0] != 3 or min(shape[1:]) < 2:
        raise ValueError('epsilon needs (z,y,x) or (3,z,y,x), each spatial dimension >=2.')
    region = (slice(None),)*3 if record_region is None else record_region
    if not isinstance(region, (tuple, list)) or len(region) != 3:
        raise ValueError('record_region must contain three z,y,x indices or slices.')
    record_slices = []
    for selection, size in zip(region, shape[1:]):
        if isinstance(selection, (int, np.integer)) and not isinstance(selection, (bool, np.bool_)):
            index = int(selection)+size if selection < 0 else int(selection)
            if not 0 <= index < size:
                raise ValueError('record_region index lies outside the grid.')
            selection = slice(index, index+1)
        if not isinstance(selection, slice):
            raise ValueError('record_region entries must be integer indices or slices.')
        if any(v is not None and (isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)))
               for v in (selection.start, selection.stop, selection.step)):
            raise ValueError('record_region slice bounds and steps must be integers.')
        if selection.step is not None and selection.step <= 0:
            raise ValueError('record_region slice steps must be positive.')
        start, stop, stride = selection.indices(size)
        if start >= stop:
            raise ValueError('record_region must be nonempty on every axis.')
        record_slices.append(slice(start, stop, stride))
    record_slices = tuple(record_slices)
    nonuniform = z_spacing is not None
    if z_spacing is not None and np.iscomplexobj(z_spacing):
        raise ValueError('z_spacing must contain real positive cell widths.')
    widths = np.full(shape[1], spacing[0]) if z_spacing is None else np.asarray(z_spacing, float)
    if widths.shape != (shape[1],) or not np.isfinite(widths).all() or np.any(widths <= 0):
        raise ValueError('z_spacing must contain one positive static cell width per z index.')
    nodes = np.r_[0., np.cumsum(widths)]
    dual = .5*(widths+np.roll(widths, 1))
    minimum_spacing = np.array([widths.min(), spacing[1], spacing[2]])
    if 2*max(absorber_cells, pml_cells) >= shape[1]:
        raise ValueError('Absorbing layers must leave a nonempty interior.')
    indices = np.asarray(monitor_z)
    if indices.size == 0:
        indices = np.array([], dtype=int)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError('monitor_z must be integer z-node indices.')
    if np.any(indices < 0) or np.any(indices >= shape[1]):
        raise ValueError('monitor_z lies outside the grid.')
    band = np.asarray(wavelengths, float)
    if band.ndim != 1 or not np.isfinite(band).all() or np.any(band <= 0):
        raise ValueError('wavelengths must be a positive finite vector.')
    if bool(len(indices)) != bool(len(band)):
        raise ValueError('monitor_z and wavelengths must be provided together.')
    if dft_subsample > 1 and len(band) and 2*dft_subsample*dt >= band.min():
        raise ValueError('dft_subsample must keep the requested band strictly below sampling Nyquist.')
    phase = array(bloch_phase)
    if phase.shape != (3,) or xp.isdtype(phase.dtype, 'complex floating'):
        raise ValueError('bloch_phase must contain three real radians in z,y,x order.')
    phase_valid = xp.all(xp.isfinite(phase))
    if pml_cells:
        phase_valid = phase_valid & (phase[0] == 0)
    factors = xp.exp(1j*phase)
    eps = xp.broadcast_to(array(epsilon), shape)
    permeability = xp.broadcast_to(array(mu), shape)
    sigma = xp.broadcast_to(array(conductivity), shape)
    if any(xp.isdtype(v.dtype, 'complex floating') for v in (eps, permeability, sigma)):
        raise ValueError('epsilon, mu and conductivity must be real.')
    bound = xp.sqrt(xp.min(eps)*xp.min(permeability))/np.sqrt(np.sum(1/minimum_spacing**2))
    stable = (phase_valid & xp.all(xp.isfinite(eps)) & xp.all(eps > 0) & xp.all(xp.isfinite(permeability))
              & xp.all(permeability > 0) & xp.all(xp.isfinite(sigma)) & xp.all(sigma >= 0)
              & (dt <= bound))
    pole_coefficients = []
    pole_strengths = []
    pole_frequencies_squared = []
    resonance_bound = array(0.)
    for pole in poles:
        if len(pole) != 3:
            raise ValueError('Each pole must be (omega0, gamma, strength).')
        omega0, gamma, strength = [xp.broadcast_to(array(v), shape) for v in pole]
        if any(xp.isdtype(v.dtype, 'complex floating') for v in (omega0, gamma, strength)):
            raise ValueError('Pole parameters must be real.')
        for v in (omega0, gamma, strength):
            stable = stable & xp.all(xp.isfinite(v)) & xp.all(v >= 0)
        denominator = 1+.5*gamma*dt
        pole_coefficients.append(((2-omega0**2*dt**2)/denominator,
                                  (1-.5*gamma*dt)/denominator, strength*dt**2/denominator))
        pole_strengths.append(strength)
        if convergence_window is not None:
            pole_frequencies_squared.append(omega0**2)
        resonance_bound = xp.maximum(resonance_bound, xp.max(omega0**2))
    if pole_coefficients:
        # Conservative coupled Maxwell/oscillator bound; uncoupled resonances
        # must still satisfy the centered oscillator time-step restriction.
        rate = resonance_bound+xp.max(sum(pole_strengths)/eps)
        bound = 1/xp.sqrt(1/bound**2+rate/4)
        stable = stable & (dt < bound)
    background = array(pml_background)
    if background.shape != (2, 2) or not xp.isdtype(background.dtype, ('real floating', 'integral')):
        raise ValueError("pml_background needs two positive finite (epsilon, mu) pairs.")
    background = xp.astype(background, xp.result_type(background, array(1.)))
    background_valid = xp.all(xp.isfinite(background)) & xp.all(background > 0)
    if _concrete(background_valid) is False:
        raise ValueError("pml_background needs two positive finite (epsilon, mu) pairs.")
    stable = stable & background_valid
    if pml_cells:
        edge = np.r_[0:pml_cells+1, shape[1]-pml_cells-1:shape[1]]
        for side, cells in enumerate((slice(0, pml_cells+1), slice(-pml_cells-1, None))):
            stable = stable & xp.all(eps[:, cells] == background[side, 0])
            stable = stable & xp.all(permeability[:, cells] == background[side, 1])
        stable = stable & xp.all(sigma[:, edge] == 0)
        for strength in pole_strengths:
            stable = stable & xp.all(strength[:, edge] == 0)
    if _concrete(stable) is False:
        raise ValueError('Invalid material/poles/PML background or dt exceeds the conservative CFL bound.')
    dtype = xp.result_type(eps, permeability, sigma, phase, array(1j),
                           *[v for coefficients in pole_coefficients for v in coefficients])
    real_dtype = xp.real(runtime.zeros((), dtype=dtype)).dtype
    dz_forward = array(widths[:, None, None], dtype=real_dtype)
    dz_backward = array(dual[:, None, None], dtype=real_dtype)
    plane_weight = array((np.roll(widths, 1)/(2*dual))[indices, None, None], dtype=real_dtype)
    factors = xp.astype(factors, dtype)
    e = runtime.zeros(shape, dtype=dtype) if electric is None else array(electric, dtype=dtype)
    h = runtime.zeros(shape, dtype=dtype) if magnetic is None else array(magnetic, dtype=dtype)
    current = runtime.zeros(shape, dtype=dtype) if source is None else array(source, dtype=dtype)
    drive = runtime.zeros(steps, dtype=dtype) if pulse is None else array(pulse, dtype=dtype)
    if e.shape != shape or h.shape != shape or current.shape != shape or drive.shape != (steps,):
        raise ValueError('E/H/source must match (3,z,y,x); pulse must have steps samples.')
    # Static damping geometry; epsilon/mu remain differentiable through updates.
    def damping_profile(offsets):
        # Evaluate the matched electric/magnetic rates at their native z
        # locations. Across both end layers the continuum amplitude decays
        # by exp(-3): two integrals of 6*u**3 over a unit-width layer.
        if nonuniform:
            z = nodes[:-1][None, :]+np.asarray(offsets)[:, None]*widths
            low, high = nodes[absorber_cells], nodes[-1]-nodes[-absorber_cells-1]
            on_low = low-z >= high-(nodes[-1]-z)
            distance = np.where(on_low, z, nodes[-1]-z)
            thickness = np.where(on_low, low, high)
            ramp = np.maximum(1-distance/thickness, 0)
            return array((6*ramp**3/thickness)[:, :, None, None], dtype=real_dtype)
        z = np.arange(shape[1])[None, :]+np.asarray(offsets)[:, None]
        distance = np.minimum(z, shape[1]-z)
        ramp = np.maximum(1-distance/absorber_cells, 0)
        return array((6*ramp**3/(absorber_cells*spacing[0]))[:, :, None, None], dtype=real_dtype)
    damping_e = damping_profile((0., 0., .5)) if absorber_cells else 0.
    damping_h = damping_profile((.5, .5, 0.)) if absorber_cells else 0.
    electric_rate = sigma/eps+damping_e
    ae = (1-.5*dt*electric_rate)/(1+.5*dt*electric_rate)
    be = (dt/eps)/(1+.5*dt*electric_rate)
    ah = (1-.5*dt*damping_h)/(1+.5*dt*damping_h)
    bh = (dt/permeability)/(1+.5*dt*damping_h)
    iz = array(indices)
    frequency = array(1/band, dtype=real_dtype)
    accum_shape = (len(band), len(indices), 3, shape[2], shape[3])
    accum = runtime.zeros(accum_shape, dtype=dtype)
    if convergence_window is not None:
        # Native component control volumes: E x/y and H z use dual widths.
        weights_e = array(np.stack((dual, dual, widths))[:, :, None, None], dtype=real_dtype)
        weights_h = array(np.stack((widths, widths, dual))[:, :, None, None], dtype=real_dtype)
        def field_norm_squared(e, h):
            return xp.sum(weights_e*xp.abs(e)**2+weights_h*xp.abs(h)**2)*spacing[1]*spacing[2]
        material_zero = runtime.zeros((len(pole_coefficients),), dtype=real_dtype)
        def material_activity(polarizations, previous):
            values = []
            for p, old, omega2, strength in zip(polarizations, previous, pole_frequencies_squared, pole_strengths):
                active = strength > 0
                # Positive oscillator activity; P and its backward-difference
                # velocity are staggered, so this is not a conserved Yee energy.
                density = (omega2*xp.abs(p)**2+xp.abs((p-old)/dt)**2)/xp.where(active, strength, 1.)
                values.append(xp.sum(weights_e*xp.where(active, density, 0.))*spacing[1]*spacing[2])
            return xp.stack(values) if values else material_zero

    def shifted(v, axis, forward, factor):
        interior = [slice(None)]*v.ndim
        edge = [slice(None)]*v.ndim
        interior[axis] = slice(1, None) if forward else slice(None, -1)
        edge[axis] = slice(0, 1) if forward else slice(-1, None)
        wrapped = v[tuple(edge)]*(factor if forward else 1/factor)
        pieces = (v[tuple(interior)], wrapped) if forward else (wrapped, v[tuple(interior)])
        return xp.concat(pieces, axis=axis)

    def derivative(v, axis, forward):
        neighbor = shifted(v, axis, forward, factors[axis])
        denominator = ((dz_forward if forward else dz_backward)
                       if nonuniform and axis == 0 else float(spacing[axis]))
        return (neighbor-v if forward else v-neighbor)/denominator

    # Reuse Wavebit's own documented CPML coefficients; no external source.
    if pml_cells:
        from .fdtd import cpml_coefficients
        def profile(offset):
            position = np.arange(shape[1])+offset
            depth = np.maximum(pml_cells-np.minimum(position, shape[1]-position), 0)*spacing[0]
            return tuple(array(v[:, None, None], dtype=real_dtype) for v in cpml_coefficients(
                depth, pml_cells*spacing[0], dt, pml_wavelength,
                xp.where(array(position < shape[1]/2), xp.prod(background[0]), xp.prod(background[1])),
                target_reflection=pml_reflection, xp=xp, array=array))
        if nonuniform:
            from .fdtd import cpml_profiles
            pe, ph = cpml_profiles(nodes, nodes[:-1]+widths/2, nodes[pml_cells],
                                   nodes[-1]-nodes[-pml_cells-1], dt, pml_wavelength,
                                   xp.prod(background[0]), xp.prod(background[1]),
                                   target_reflection=pml_reflection, xp=xp, array=array)
            pml_e = tuple(array(pe[key][:-1, None, None], dtype=real_dtype) for key in ('b', 'a', 'kappa'))
            pml_h = tuple(array(ph[key][:, None, None], dtype=real_dtype) for key in ('b', 'a', 'kappa'))
        else:
            pml_e, pml_h = profile(0.), profile(.5)
    auxiliary = runtime.zeros((2,)+shape[1:], dtype=dtype) if pml_cells else runtime.zeros((0,), dtype=dtype)

    def curl(v, forward, memory):
        ex, ey, ez = v[0], v[1], v[2]
        dzey, dzex = derivative(ey, 0, forward), derivative(ex, 0, forward)
        if pml_cells:
            b, a, kappa = pml_h if forward else pml_e
            memory = b*memory+a*xp.stack((dzey, dzex))
            dzey, dzex = dzey/kappa+memory[0], dzex/kappa+memory[1]
        return xp.stack((derivative(ez, 1, forward)-dzey,
                         dzex-derivative(ez, 2, forward),
                         derivative(ey, 2, forward)-derivative(ex, 1, forward))), memory

    def planes(v, magnetic_field):
        offsets = ((1, 1, 0), (1, 0, 1), (0, 1, 1)) if magnetic_field else ((0, 0, 1), (0, 1, 0), (1, 0, 0))
        values = []
        for c, (zh, yh, xh) in enumerate(offsets):
            face = v[c, iz]
            if zh:
                previous = v[c, (iz-1)%shape[1]]*xp.where(iz == 0, 1/factors[0], 1)[:, None, None]
                weight = plane_weight if nonuniform else .5
                face = weight*face+(1-weight)*previous
            if yh:
                face = .5*(face+shifted(face, 1, False, factors[1]))
            if xh:
                face = .5*(face+shifted(face, 2, False, factors[2]))
            values.append(face)
        return xp.stack(values, axis=1)

    def step(state, n):
        e, h, fe, fh, pe, ph, polarizations, previous = state[:8]
        rotation, ph = curl(e, True, ph)
        h = ah*h-bh*rotation
        rotation, pe = curl(h, False, pe)
        updated = tuple(gain*p-hold*old+drive_p*e for (gain, hold, drive_p), p, old
                        in zip(pole_coefficients, polarizations, previous))
        delta_p = sum((new-old for new, old in zip(updated, polarizations)), 0.)
        e = ae*e+be*(rotation-current*drive[n]-delta_p/dt)
        monitor_state = (fe, fh)
        if convergence_window is not None:
            monitor_state += state[10:12]
        if len(indices):
            def accumulate(values):
                fe, fh = values[:2]
                # Left sample represents up to stride steps; clip the last block.
                weight = xp.minimum(array(dft_subsample), array(steps-n))
                ek = xp.exp(2j*xp.pi*frequency*(n+1)*dt)*dt
                hk = xp.exp(2j*xp.pi*frequency*(n+.5)*dt)*dt
                de = ek[:, None, None, None, None]*planes(e, False)[None]
                dh = hk[:, None, None, None, None]*planes(h, True)[None]
                updated = (fe+weight*de, fh+weight*dh)
                if convergence_window is not None:
                    tail_weight = xp.maximum(array(0), n+weight-(steps-convergence_window))
                    tail_weight = xp.minimum(weight, tail_weight)
                    updated += (values[2]+tail_weight*de, values[3]+tail_weight*dh)
                return updated
            monitor_state = (accumulate(monitor_state) if dft_subsample == 1 else
                             runtime.conditional(n % dft_subsample == 0, accumulate,
                                                 lambda v: v, monitor_state))
        fe, fh = monitor_state[:2]
        result = (e, h, fe, fh, pe, ph, updated, polarizations)
        if convergence_window is not None:
            peak, tail_peak = state[8:10]
            norm = field_norm_squared(e, h)
            in_tail = array(n >= steps-convergence_window)
            peak = xp.maximum(peak, norm)
            tail_peak = xp.maximum(tail_peak, xp.where(in_tail, norm, 0.))
            activity = material_activity(updated, polarizations)
            material_peak = xp.maximum(state[12], activity)
            material_tail = xp.maximum(state[13], xp.where(in_tail, activity, 0.))
            result += (peak, tail_peak)+monitor_state[2:]+(material_peak, material_tail)
        return result, None

    polarization = tuple(runtime.zeros(shape, dtype=dtype) for _ in pole_coefficients)
    state = (e, h, accum, accum, auxiliary, auxiliary, polarization, polarization)
    if convergence_window is not None:
        state += (field_norm_squared(e, h), array(0., dtype=real_dtype), accum, accum, material_zero, material_zero)
    records = {}
    if record_field is None:
        state = runtime.advance(step, state, steps, checkpoint_steps)
    else:
        def capture(current):
            return xp.stack([current[names.index(name)//3][names.index(name)%3][record_slices] for name in selected])
        state, history = runtime.advance_recorded(step, state, steps, checkpoint_steps, record_field, capture)
        stamps = array((np.arange(0, steps, record_field)+1)*dt)
        for index, name in enumerate(selected):
            component = names.index(name)%3
            offsets = np.ones(3)*.5 if name.startswith('h') else np.zeros(3)
            offsets[component] = 0. if name.startswith('h') else .5
            records[name] = dict(values=history[:, index],
                                 t=stamps-(dt/2 if name.startswith('h') else 0.),
                                 x=array((np.arange(shape[3])+offsets[0])*spacing[2])[record_slices[2]],
                                 y=array((np.arange(shape[2])+offsets[1])*spacing[1])[record_slices[1]],
                                 z=array(nodes[:-1]+offsets[2]*widths)[record_slices[0]])
    e, h, fe, fh, _, _, polarization, previous = state[:8]
    convergence = None
    if convergence_window is not None:
        peak, tail_peak, tail_e, tail_h = state[8:12]
        material_peak, material_tail = state[12:14]
        material_signal = xp.isfinite(material_peak) & (material_peak > 0)
        material_ratio = xp.sqrt(material_tail/xp.where(material_signal, material_peak, 1.))
        material_decayed = stable & xp.all(xp.isfinite(material_peak) & xp.isfinite(material_tail)
                                          & ((material_peak == 0) | (material_ratio <= convergence_tol)))
        signal_present = xp.isfinite(peak) & (peak > 0)
        field_ratio = xp.sqrt(tail_peak/xp.where(signal_present, peak, 1.))
        drive_peak = xp.max(xp.abs(drive))
        source_inactive = ((xp.max(xp.abs(current)) == 0)
                           | (xp.max(xp.abs(drive[-convergence_window:]))
                              <= convergence_tol*drive_peak))
        spectrum_norm = xp.sum(xp.abs(fe)**2+xp.abs(fh)**2, axis=(2, 3, 4))
        spectral_signal = xp.isfinite(spectrum_norm) & (spectrum_norm > 0)
        spectral_ratio = xp.sqrt(xp.sum(xp.abs(tail_e)**2+xp.abs(tail_h)**2, axis=(2, 3, 4))
                                 /xp.where(spectral_signal, spectrum_norm, 1.))
        convergence = dict(window_steps=convergence_window, tolerance=convergence_tol,
                           signal_present=signal_present, source_inactive=source_inactive,
                           field_decay=xp.where(signal_present, field_ratio, array(float('nan'))),
                           field_decayed=stable & signal_present & source_inactive & (field_ratio <= convergence_tol),
                           material_signal_present=material_signal,
                           material_decay=xp.where(material_signal, material_ratio, array(float('nan'))),
                           material_activity_peak=material_peak, material_activity_tail=material_tail,
                           material_decayed=material_decayed,
                           electric_spectrum_before_window=fe-tail_e,
                           magnetic_spectrum_before_window=fh-tail_h,
                           spectral_signal_present=spectral_signal,
                           spectral_tail=xp.where(spectral_signal, spectral_ratio, array(float('nan'))))
        convergence['state_decayed'] = convergence['field_decayed'] & material_decayed
    flux = .5*xp.real(xp.linalg.cross(xp.moveaxis(fe, 2, -1), xp.conj(xp.moveaxis(fh, 2, -1)), axis=-1))
    power = xp.sum(flux[..., 2], axis=(-2, -1))*spacing[1]*spacing[2]
    return dict(E=e, H=h, fields=records, polarization=polarization, previous_polarization=previous,
                electric_spectrum=fe, magnetic_spectrum=fh,
                flux_density=flux, power_z=power, wavelengths=array(band), monitor_z=iz,
                stable=stable, convergence=convergence, cfl_dt=bound, time=steps*dt,
                dft_subsample=dft_subsample, dft_samples=(steps-1)//dft_subsample+1 if len(band) else 0,
                h_time=(steps-.5)*dt, spacing=array(spacing), bloch_phase=phase,
                z_nodes=array(nodes[:-1]), z_spacing=array(widths), pml_reflection=pml_reflection)
