"""Functional uniform-grid Yee engine with NumPy and compiled differentiable JAX loops.

This low-level API takes native grid coefficients, not Stack/Incidence scenes.
"""
import operator

import numpy as np


def fdtd_grid(epsilon, *, spacing, dt, steps, mu=1., conductivity=0.,
              electric=None, magnetic=None, source=None, pulse=None,
              wavelengths=(), monitor_z=(), backend='numpy', device=None,
              checkpoint_steps=32, absorber_cells=0):
    """Advance a 3D Yee grid and accumulate frequency-domain field planes.

    epsilon: positive scalar-per-cell (z,y,x) or diagonal/native (3,z,y,x)
    permittivity. mu and electric conductivity broadcast to that shape. E/H/source
    have shape (3,z,y,x). Source is an electric current pattern; pulse has `steps`
    samples, and the update uses curl(H)-source*pulse. E starts at t=0 and H at
    t=-dt/2. H is normalized as Z0 H. All axes are periodic.

    spacing is (dz,dy,dx); lengths and time use c=1. dt is explicit, so optimizing
    epsilon does not change the number of steps or simulation duration. Choose dt
    below the CFL bound over the whole optimization domain. Concrete inputs are
    validated; traced JAX inputs expose validity through returned `stable`.

    Optional absorber_cells adds a cubic, impedance-matched damping layer at both
    z ends. It is a graded conductivity absorber, NOT the scene solver's CPML.
    Refine its width to establish reflection accuracy. monitor_z contains integer
    z-node indices; streaming planes are collocated in space, with native E/H DFT
    timestamps. Fields are unnormalized integrals in source*time units.

    backend='jax' uses lax.scan on CPU/GPU, with block checkpointing for reverse
    differentiation; device may be a JAX device or 'cpu'/'gpu'. Selecting unavailable
    hardware raises. Inputs remain differentiable: jax.grad may differentiate
    material, initial field, source and pulse. No full time history is stored.
    """
    if backend not in ('numpy', 'jax'):
        raise ValueError("backend must be 'numpy' or 'jax'.")
    steps = operator.index(steps)
    checkpoint_steps = operator.index(checkpoint_steps)
    absorber_cells = operator.index(absorber_cells)
    if steps < 1 or checkpoint_steps < 1 or absorber_cells < 0:
        raise ValueError('steps/checkpoint_steps must be positive; absorber_cells nonnegative.')
    spacing = np.asarray(spacing, float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError('spacing must contain positive dz,dy,dx.')
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be positive and finite.')
    shape = tuple(np.shape(epsilon))
    if len(shape) == 3:
        shape = (3,)+shape
    if len(shape) != 4 or shape[0] != 3 or min(shape[1:]) < 2:
        raise ValueError('epsilon needs (z,y,x) or (3,z,y,x), each spatial dimension >=2.')
    if 2*absorber_cells >= shape[1]:
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
    if backend == 'jax':
        import jax
        import jax.numpy as xp
        if isinstance(device, str):
            available = jax.devices(device)
            if not available:
                raise ValueError(f'No JAX {device} device is available.')
            device = available[0]
        def array(value, dtype=None):
            result = xp.asarray(value, dtype=dtype)
            return jax.device_put(result, device) if device is not None else result
    else:
        xp = np
        if device is not None:
            raise ValueError('device selection requires backend=jax.')
        array = np.asarray
    eps = xp.broadcast_to(array(epsilon), shape)
    permeability = xp.broadcast_to(array(mu), shape)
    sigma = xp.broadcast_to(array(conductivity), shape)
    if any(xp.iscomplexobj(v) for v in (eps, permeability, sigma)):
        raise ValueError('epsilon, mu and conductivity must be real.')
    bound = xp.sqrt(xp.min(eps)*xp.min(permeability))/np.sqrt(np.sum(1/spacing**2))
    stable = (xp.all(xp.isfinite(eps)) & xp.all(eps > 0) & xp.all(xp.isfinite(permeability))
              & xp.all(permeability > 0) & xp.all(xp.isfinite(sigma)) & xp.all(sigma >= 0)
              & (dt <= bound))
    if backend == 'numpy' or not isinstance(stable, jax.core.Tracer):
        if not bool(stable):
            raise ValueError('Invalid material or dt exceeds the conservative CFL bound.')
    dtype = xp.result_type(eps, array(1j))
    e = xp.zeros(shape, dtype=dtype) if electric is None else array(electric, dtype=dtype)
    h = xp.zeros(shape, dtype=dtype) if magnetic is None else array(magnetic, dtype=dtype)
    current = xp.zeros(shape, dtype=dtype) if source is None else array(source, dtype=dtype)
    drive = xp.zeros(steps, dtype=dtype) if pulse is None else array(pulse, dtype=dtype)
    if e.shape != shape or h.shape != shape or current.shape != shape or drive.shape != (steps,):
        raise ValueError('E/H/source must match (3,z,y,x); pulse must have steps samples.')
    # Static damping geometry; epsilon/mu remain differentiable through updates.
    def damping_profile(offsets):
        # Evaluate the matched electric/magnetic rates at their native z
        # locations. Across both end layers the continuum amplitude decays
        # by exp(-3): two integrals of 6*u**3 over a unit-width layer.
        z = np.arange(shape[1])[None, :]+np.asarray(offsets)[:, None]
        distance = np.minimum(z, shape[1]-z)
        ramp = np.maximum(1-distance/absorber_cells, 0)
        return array((6*ramp**3/(absorber_cells*spacing[0]))[:, :, None, None])
    damping_e = damping_profile((0., 0., .5)) if absorber_cells else 0.
    damping_h = damping_profile((.5, .5, 0.)) if absorber_cells else 0.
    electric_rate = sigma/eps+damping_e
    ae = (1-.5*dt*electric_rate)/(1+.5*dt*electric_rate)
    be = (dt/eps)/(1+.5*dt*electric_rate)
    ah = (1-.5*dt*damping_h)/(1+.5*dt*damping_h)
    bh = (dt/permeability)/(1+.5*dt*damping_h)
    iz = array(indices)
    frequency = array(1/band)
    accum_shape = (len(band), len(indices), 3, shape[2], shape[3])
    accum = xp.zeros(accum_shape, dtype=dtype)

    def derivative(v, axis, forward):
        return ((xp.roll(v, -1, axis=axis)-v) if forward else
                (v-xp.roll(v, 1, axis=axis)))/spacing[axis]

    def curl(v, forward):
        ex, ey, ez = v[0], v[1], v[2]
        return xp.stack((derivative(ez, 1, forward)-derivative(ey, 0, forward),
                         derivative(ex, 0, forward)-derivative(ez, 2, forward),
                         derivative(ey, 2, forward)-derivative(ex, 1, forward)))

    def planes(v, magnetic_field):
        offsets = ((1, 1, 0), (1, 0, 1), (0, 1, 1)) if magnetic_field else ((0, 0, 1), (0, 1, 0), (1, 0, 0))
        values = []
        for c, (zh, yh, xh) in enumerate(offsets):
            face = v[c, iz]
            if zh:
                face = .5*(face+v[c, (iz-1)%shape[1]])
            if yh:
                face = .5*(face+xp.roll(face, 1, axis=1))
            if xh:
                face = .5*(face+xp.roll(face, 1, axis=2))
            values.append(face)
        return xp.stack(values, axis=1)

    def step(state, n):
        e, h, fe, fh = state
        h = ah*h-bh*curl(e, True)
        e = ae*e+be*(curl(h, False)-current*drive[n])
        if len(indices):
            ek = xp.exp(2j*xp.pi*frequency*(n+1)*dt)*dt
            hk = xp.exp(2j*xp.pi*frequency*(n+.5)*dt)*dt
            fe = fe+ek[:, None, None, None, None]*planes(e, False)[None]
            fh = fh+hk[:, None, None, None, None]*planes(h, True)[None]
        return (e, h, fe, fh), None

    state = (e, h, accum, accum)
    if backend == 'jax':
        if device is not None:
            state = jax.tree.map(lambda v: jax.device_put(v, device), state)
        chunks, tail = divmod(steps, checkpoint_steps)
        def block(state, ns):
            return jax.lax.scan(step, state, ns)[0], None
        if chunks:
            ns = array(np.arange(chunks*checkpoint_steps).reshape(chunks, checkpoint_steps))
            state = jax.lax.scan(jax.checkpoint(block), state, ns)[0]
        if tail:
            state = jax.lax.scan(step, state, array(np.arange(chunks*checkpoint_steps, steps)))[0]
    else:
        for n in range(steps):
            state, _ = step(state, n)
    e, h, fe, fh = state
    flux = .5*xp.real(xp.cross(xp.moveaxis(fe, 2, -1), xp.conj(xp.moveaxis(fh, 2, -1))))
    power = flux[..., 2].sum(axis=(-2, -1))*spacing[1]*spacing[2]
    return dict(E=e, H=h, electric_spectrum=fe, magnetic_spectrum=fh,
                flux_density=flux, power_z=power, wavelengths=array(band), monitor_z=iz,
                stable=stable, cfl_dt=bound, time=steps*dt,
                h_time=(steps-.5)*dt, spacing=array(spacing))
