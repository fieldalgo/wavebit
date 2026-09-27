"""Result-owned Cartesian fields, shared by ordinary and source-based solves.

Solvers supply spectral coefficients; this module owns sampling, spatial expansion,
component layout and the lifetime of the solved field. No query reads a live solver.
"""
import numpy as np

from .backend import array_namespace, as_array
from .fields import (LayerSamples, is_count, field_1d, field_1d_conical,
                     field_2d, field_2d_4ff)


def _host_value(value):
    """Concrete data for eager validation; tracing never chooses the field path."""
    try:
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        return np.asarray(value)
    except (TypeError, ValueError, RuntimeError):
        return None


def _copy(value):
    """Copy mutable containers/arrays without severing automatic differentiation."""
    if isinstance(value, dict):
        return {k: _copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_copy(v) for v in value)
    if hasattr(value, 'shape') and hasattr(value, 'dtype'):
        xp = array_namespace(value)
        return xp.asarray(value, copy=True)
    return value


def _array(xp, value, like):
    return as_array(xp, value, dtype=like.dtype, device=getattr(like, 'device', None))


def _coordinates(xp, value, like, name):
    arr = _array(xp, value, like)
    if arr.ndim != 1 or not arr.shape[0]:
        raise ValueError(f'{name} must be a positive integer count or a nonempty 1-D coordinate array')
    concrete = _host_value(arr)
    if concrete is not None:
        if not np.isfinite(concrete).all() or np.any(np.imag(concrete)):
            raise ValueError(f'{name} must contain finite real coordinates')
    return xp.real(arr)


def _count(value, name):
    if value < 1:
        raise ValueError(f'{name} must be a positive integer count')
    return int(value)


def xy_grid(xp, lattice, sample_x, sample_y):
    """Centre-origin Cartesian points; counts span the cell's lattice directions."""
    x_count, y_count = is_count(sample_x), is_count(sample_y)
    if x_count and y_count:
        u = xp.linspace(-.5, .5, _count(sample_x, 'sample_x'),
                        dtype=lattice.dtype, device=getattr(lattice, 'device', None))
        v = xp.linspace(-.5, .5, _count(sample_y, 'sample_y'),
                        dtype=lattice.dtype, device=getattr(lattice, 'device', None))
        return (u[None, :]*lattice[0, 0] + v[:, None]*lattice[0, 1],
                u[None, :]*lattice[1, 0] + v[:, None]*lattice[1, 1])
    if x_count or y_count:
        concrete = _host_value(lattice)
        if concrete is None or np.any(np.abs(np.asarray(concrete)[[0, 1], [1, 0]]) > 1e-12):
            raise ValueError('On a skewed or traced lattice, use counts for both x/y axes '
                             'or explicit Cartesian coordinates for both.')
    def axis(spec, size, name):
        if is_count(spec):
            return xp.linspace(-.5*size, .5*size, _count(spec, name),
                               dtype=lattice.dtype, device=getattr(lattice, 'device', None))
        return _coordinates(xp, spec, lattice, name)
    x, y = axis(sample_x, lattice[0, 0], 'sample_x'), axis(sample_y, lattice[1, 1], 'sample_y')
    return xp.broadcast_arrays(x[None, :], y[:, None])


def expand_spectrum(xp, kxy, coefficients, grid):
    """Expand (..., order, six components) on the common transverse grid."""
    x, y = grid
    phase = xp.exp(1j*(x[..., None]*kxy[:, 0] + y[..., None]*kxy[:, 1]))
    flat = xp.reshape(phase, (-1, kxy.shape[0])) @ coefficients
    return xp.reshape(flat, coefficients.shape[:-2] + x.shape + (6,))


class FieldState:
    """Numeric solve snapshot and a static adapter name; a JAX pytree when registered."""

    def __init__(self, data, kind, mirrored=False, names=('incident',)):
        self.data, self.kind, self.mirrored, self.names = data, kind, mirrored, tuple(names)

    def tree_flatten(self):
        return (self.data,), (self.kind, self.mirrored, self.names)

    @classmethod
    def tree_unflatten(cls, meta, children):
        return cls(children[0], *meta)

    def _incidence(self, name):
        if name not in self.names:
            raise ValueError(f'incidence={name!r} is unavailable; choose from {self.names}')
        return self.names.index(name)

    def boundary(self, side, sample_x, sample_y, part, incidence):
        self._incidence(incidence)
        if side not in ('top', 'bottom') or part not in ('incoming', 'outgoing', 'total'):
            raise ValueError("Use side='top'/'bottom', part='incoming'/'outgoing'/'total'.")
        spectra = self.data['boundary']
        if spectra is None:
            raise RuntimeError("Boundary fields require solve(answers='amplitude') or answers='field'.")
        xp = array_namespace(self.data['lattice'])
        grid = xy_grid(xp, self.data['lattice'], sample_x, sample_y)
        incoming, outgoing = spectra[incidence][side]
        kxy = self.data['boundary_kxy']
        coefficients = incoming if part == 'incoming' else outgoing
        if part == 'total':
            coefficients = incoming + outgoing
        return expand_spectrum(xp, kxy, coefficients, grid)

    def reconstruct(self, sample_x, sample_y, sample_z, incidence):
        index = self._incidence(incidence)
        data = self.data
        if not data['groups'] or any(g['layers'] is None for g in data['groups']):
            raise RuntimeError("Internal fields require solve(answers='field') and at least one layer.")
        xp = array_namespace(data['lattice'])
        grid = xy_grid(xp, data['lattice'], sample_x, sample_y)
        ds = xp.real(data['thickness'])
        if not ds.shape[0]:
            raise ValueError('There are no finite layers to sample')
        edges = xp.concat([xp.zeros_like(ds[:1]), xp.cumulative_sum(ds)])
        counted = is_count(sample_z)
        if counted:
            count = _count(sample_z, 'sample_z')
            fractions = xp.linspace(0., 1., count, dtype=ds.dtype,
                                    device=getattr(ds, 'device', None))
            z = xp.concat([edges[j] + fractions*ds[j] for j in range(ds.shape[0])])
            # Count samples include each layer's own limiting value at both faces.
            owner = xp.repeat(xp.arange(ds.shape[0], device=getattr(ds, 'device', None)), count)
        else:
            z = _coordinates(xp, sample_z, data['lattice'], 'sample_z')
            owner = xp.sum(xp.astype(z[:, None] >= edges[None, 1:], xp.int64), axis=1)
            owner = xp.minimum(owner, xp.full_like(owner, ds.shape[0]-1))
        actual, total = _host_value(z), _host_value(edges[-1])
        if actual is not None and total is not None:
            if np.any(actual < 0) or np.any(actual > total):
                raise ValueError('sample_z coordinates must lie inside the stack, 0 to sum(thickness)')
        result = None
        for g in data['groups']:
            if self.kind == 'source':
                coefficient = self._source_coefficients(xp, g, z, owner, edges)
            else:
                coefficient = self._ordinary_coefficients(xp, g, z, owner, edges, index)
            field = expand_spectrum(xp, g['kxy'], coefficient, grid)
            result = field if result is None else result + field
        return result

    def _ordinary_coefficients(self, xp, g, z, owner, edges, index):
        layers = g['layers']
        nlayer = len(layers)
        # Work at every requested height in each layer, clipping unused samples so
        # an unselected evanescent solution cannot overflow and poison a gradient.
        local = [xp.minimum(xp.maximum(z-edges[j], xp.zeros_like(z)), edges[j+1]-edges[j])
                 for j in range(nlayer)]
        masks = [owner == j for j in range(nlayer)]
        if self.mirrored:
            local = [edges[j+1]-edges[j]-local[j] for j in range(nlayer-1, -1, -1)]
            masks.reverse()
        kwargs = dict(sample_z=LayerSamples(local), incidences=(self.names[index],),
                      _coefficients=True)
        args = (self.data['wavelength'], g['kx'])
        tail = (g['T1'], layers, self.data['lattice'])
        if self.kind in ('te', 'tm'):
            cells = field_1d(*args, *tail, int(self.kind == 'tm'), **kwargs)
        else:
            fn = {'conical': field_1d_conical, '2ff': field_2d, '4ff': field_2d_4ff}[self.kind]
            cells = fn(*args, g['ky'], *tail, **kwargs)
        coefficients = sum((xp.where(mask[:, None, None], cell[:, 0], 0)
                            for mask, cell in zip(masks, cells)))
        if self.mirrored:
            coefficients = coefficients * _array(xp, [1, 1, -1, -1, -1, 1], coefficients)
        # Ordinary operators use the corner Fourier origin; public coordinates
        # use the cell centre. Preserve the solved source's phase convention.
        origin = xp.sum(self.data['lattice'], axis=1)/2
        return coefficients * xp.exp(1j*(g['kxy'] @ origin))[None, :, None]

    def _source_coefficients(self, xp, g, z, owner, edges):
        ge = g['edges']
        heights = g['source_heights']
        actual = _host_value(z)
        if actual is not None and any(np.any(np.isclose(actual, h, rtol=0,
                         atol=1e-12*max(1., float(np.asarray(_host_value(edges[-1])))))) for h in heights):
            raise ValueError('The point-source plane has a singular field; sample strictly above or below it.')
        coefficient = None
        for j, (eps, d, lf, pf, lb, pb, af, ab) in enumerate(g['layers']):
            # Dipole planes split a layer into concrete segment thicknesses.
            d = xp.real(_array(xp, d, self.data['thickness']))
            local = xp.minimum(xp.maximum(z-ge[j], xp.zeros_like(z)), d)
            state = (pf @ (xp.exp(1j*g['k0']*local[:, None]*lf)*af)[..., None]
                     + pb @ (xp.exp(1j*g['k0']*(local-d)[:, None]*lb)*ab)[..., None])[..., 0]
            n, kx, ky = g['n'], g['kx'], g['ky']
            ex, ey, hx, hy = [state[:, k*n:(k+1)*n] for k in range(4)]
            ez = (eps['ezz_i'] @ (ky*hx-kx*hy)[..., None])[..., 0]
            ez = ez - (eps['ezz_i'] @ (eps['ezx'] @ ex[..., None] + eps['ezy'] @ ey[..., None]))[..., 0]
            hz = kx*ey-ky*ex
            mu = eps.get('mu_blocks')
            if mu is not None:
                hz = (mu['ezz_i'] @ (hz[..., None]-mu['ezx'] @ hx[..., None]-mu['ezy'] @ hy[..., None]))[..., 0]
            field = xp.stack([ex, ey, ez, hx, hy, hz], axis=-1)
            # The segment's host layer is static metadata, not a traced choice.
            host = g['hosts'][j]
            mask = owner == host
            if heights:
                mask = mask & (z >= ge[j]) & (z <= ge[j+1])
            field = xp.where(mask[:, None, None], field, 0)
            coefficient = field if coefficient is None else coefficient + field
        return coefficient


class ResultFields:
    """Public field API shared by every solve result."""

    def _require_field_state(self):
        if self._field_state is None:
            raise RuntimeError("This result has no field state; solve with answers='amplitude' "
                               "for boundary fields or answers='field' for internal fields.")
        return self._field_state

    def reconstruct_field(self, sample_x=20, sample_y=20, sample_z=20, *, incidence='incident'):
        """Complex (Ex,Ey,Ez,Z0Hx,Z0Hy,Z0Hz), shape (nz,ny,nx,6).

        Counts or 1-D physical coordinates are accepted on every axis. z counts
        are per finite layer; explicit heights start at the stack bottom and
        choose the upper layer at an interface. No live solver state is read.
        ``incidence`` selects the solved input, or a retained pure 's'/'p' variant.
        """
        return self._require_field_state().reconstruct(sample_x, sample_y, sample_z, incidence)

    def boundary_fields(self, side, sample_x=(0.,), sample_y=(0.,), part='total', *, incidence='incident'):
        """Boundary half-space fields, shape (ny,nx,6), with the same sampling rules.

        side is 'top' or 'bottom'; part is 'incoming', 'outgoing' or 'total'.
        """
        return self._require_field_state().boundary(side, sample_x, sample_y, part, incidence)


def ordinary_spectra(solver, variants, *, eps_top=None, eps_bot=None, mu_top=None,
                     mu_bot=None):
    """Every retained order's six-component plane wave, per incidence, before the lateral phase.

    ``variants`` maps ``'incident'`` / ``'s'`` / ``'p'`` to the solve's result mappings (the
    ``r_s`` / ``r_p`` / ``t_s`` / ``t_p`` it packed) or ``None``. Returns ``None`` below
    ``answers='amplitude'``, where there are no amplitudes to expand.

    One arithmetic serves two readers, which is why it is its own function: the boundary spectra
    :func:`ordinary_state` hands to the field API (these times the phase that moves the corner
    Fourier origin to the public centre), and the per-order Cartesian vectors ``r_xyz`` /
    ``t_xyz`` (:func:`cartesian_amplitudes`, these as they are). A second spelling of the s/p ->
    xyz basis would be one more copy of the rule ``order_azimuth`` / ``order_frame_sign`` state.
    """
    from .conventions import boundary_kz, order_azimuth, order_frame_sign, incident_column
    from .solve_2d import order_grid_kxy

    incident = variants.get('incident')
    if incident is None or incident['r_s'] is None:
        return None
    xp, dtype = solver.xp, solver.complex_dtype
    kx, ky = solver.order_kxy()
    gx, gy = order_grid_kxy(xp, kx, ky)
    gx, gy = xp.reshape(gx, (-1,)), xp.reshape(gy, (-1,))
    k0 = 2*np.pi/solver.wavelength
    kxy = xp.stack([gx, gy], axis=-1)*k0
    lattice = _array(xp, solver.lattice.columns, kxy)
    if solver._grating_type == 0:
        kind = 'te' if solver.tetm_index == 0 else 'tm'
    else:
        kind = 'conical' if solver._grating_type == 1 else solver._formulation_assigned.value
    names = tuple(name for name, sub in variants.items() if sub is not None)
    (cx, hx), (cy, hy) = solver.order_window
    specular = (hy-cy)*(2*hx+1) + hx-cx
    delta = _array(xp, incident_column(xp, dtype, solver._lit(), gx.shape[0], specular), kxy)
    angle = order_azimuth(xp, gy, gx, solver.phi)
    c, s = xp.cos(angle), xp.sin(angle)
    if kind in ('te', 'tm'):
        # The pure 1D path's own per-order frame, exactly: `s_hat = sigma * y_hat` on the
        # ky == 0 row, spelled without the angle so no `sin(pi)` reaches the zero component.
        c, s = xp.astype(order_frame_sign(xp, gx), gx.dtype), xp.zeros_like(gx)
    materials = {'top': (solver.n_top, eps_top, mu_top),
                 'bottom': (solver.n_bot, eps_bot, mu_bot)}

    def basis(side, direction):
        """Six Cartesian field components for this port's two amplitude channels."""
        n, epsilon, mu = materials[side]
        mu = _array(xp, 1. if mu is None else mu, kxy)
        epsilon = n*n/mu if epsilon is None else epsilon
        kz = boundary_kz(xp, epsilon*mu, gx*gx+gy*gy)
        rho = gx*c + gy*s
        zero = xp.zeros_like(gx)
        es = xp.stack([-s, c, zero], axis=-1)
        ep = xp.stack([-kz*c/n, -kz*s/n, direction*rho/n], axis=-1)
        k = xp.stack([gx, gy, direction*kz], axis=-1)
        return xp.stack([xp.concat([es, xp.linalg.cross(k, es)/mu], axis=-1),
                         xp.concat([ep, xp.linalg.cross(k, ep)/mu], axis=-1)], axis=-1)

    side = solver.incidence_side
    other = 'bottom' if side == 'top' else 'top'
    direction = -1 if side == 'top' else 1
    incoming_basis = basis(side, direction)
    reflected_basis = basis(side, -direction)
    transmitted_basis = basis(other, direction)
    spectra = {}
    for name in names:
        sub = variants[name]
        if name == 'incident' and kind not in ('te', 'tm'):
            pair = solver._incidence_pair()
        else:
            pair = (1., 0.) if name == 's' or kind == 'te' else (0., 1.)
        # Every ordinary input is a unit electric amplitude, the pure 1D TM solve's included
        # since 2026-09-27 (its kernel rescales the state it hands over), so the boundary
        # spectra carry the amplitudes as they are.
        inc = xp.stack([delta*pair[0], delta*pair[1]], axis=-1)
        # Fresnel reflected p uses the opposite sign to the basis above.
        ref = xp.stack([xp.reshape(sub['r_s'], (-1,)),
                        -xp.reshape(sub['r_p'], (-1,))], axis=-1)
        trn = xp.stack([xp.reshape(sub['t_s'], (-1,)),
                       xp.reshape(sub['t_p'], (-1,))], axis=-1)
        spectra[name] = (xp.sum(incoming_basis*inc[:, None, :], axis=-1),
                         xp.sum(reflected_basis*ref[:, None, :], axis=-1),
                         xp.sum(transmitted_basis*trn[:, None, :], axis=-1))
    return dict(kind=kind, names=names, side=side, other=other, kx=kx, ky=ky, kxy=kxy,
                lattice=lattice, spectra=spectra,
                phase=xp.exp(1j*(kxy @ (xp.sum(lattice, axis=1)/2)))[:, None])


def cartesian_amplitudes(spectra, name, shape):
    """``r_xyz`` / ``t_xyz`` of one incidence: each outgoing order's E, ``(Ex, Ey, Ez)``.

    The E half of :func:`ordinary_spectra`'s reflected and transmitted families, put on the
    order grid with the component axis last, ``shape + (3,)``. Per unit incident amplitude,
    exactly as ``r_s`` is -- the same numbers in the fixed public frame, not a new measurement:
    ``r_xyz = r_s * s_hat + r_p * p_hat_ref`` and ``t_xyz = t_s * s_hat + t_p * p_hat_trn``.
    With no per-order frame in it, it is continuous where an order's ``kx`` crosses zero on the
    ``ky == 0`` row, where ``r_s`` turns over with ``s_hat``. Its phase reference is ``r_s``'s:
    in ``z`` the lit and far interfaces, and laterally the cell **corner** (the Fourier origin)
    -- it is taken before the phase that moves the boundary spectra to the public centre.

    Prior art: RETICOLO's ``res2`` returns each order's Cartesian ``E`` and ``H`` components at
    its origin beside the amplitudes -- J. P. Hugonin and P. Lalanne, "RETICOLO software for
    grating analysis", arXiv:2101.00901. ``H`` is not shipped here: it is
    ``(k/k0) x E / mu`` in Z0 units with ``k/k0 = (kx, ky, +-kz)``, and
    ``boundary_fields(part='outgoing')`` already returns it in space.
    """
    _, reflected, transmitted = spectra['spectra'][name]
    xp = array_namespace(reflected)
    return {'r_xyz': xp.reshape(reflected[:, :3], tuple(shape) + (3,)),
            't_xyz': xp.reshape(transmitted[:, :3], tuple(shape) + (3,))}


def ordinary_state(solver, spectra):
    """Adapt one ordinary solve without retaining a reference to the solver.

    ``spectra`` is :func:`ordinary_spectra`'s answer -- ``None`` below ``answers='amplitude'``,
    which has no field state -- and the boundary waves are those spectra moved to the public
    centre origin.
    """
    if spectra is None:
        return None
    xp = array_namespace(spectra['kxy'])
    side, other, phase = spectra['side'], spectra['other'], spectra['phase']
    boundary = {}
    for name in spectra['names']:
        incoming, reflected, transmitted = spectra['spectra'][name]
        transmitted = transmitted*phase
        boundary[name] = {side: (incoming*phase, reflected*phase),
                          other: (xp.zeros_like(transmitted), transmitted)}
    groups = ()
    if solver.T1 is not None and solver.layer_info_list:
        groups = (dict(kx=spectra['kx'], ky=spectra['ky'], kxy=spectra['kxy'], T1=solver.T1,
                       layers=solver.layer_info_list),)
    data = dict(lattice=spectra['lattice'], wavelength=solver.wavelength,
                thickness=solver.thickness, boundary_kxy=spectra['kxy'], boundary=boundary,
                groups=groups)
    return FieldState(_copy(data), spectra['kind'], solver._kernel_incidence_side == 'bottom',
                      spectra['names'])


def source_state(spectra, groups, lattice, wavelength, thickness):
    """Adapt coherent source groups to the same sample/expand implementation."""
    boundary = {'incident': {
        side: (spectra[side, 'incoming'].field, spectra[side, 'outgoing'].field)
        for side in ('top', 'bottom')
    }}
    edges = np.r_[0., np.cumsum(_host_value(thickness).real)]
    converted = []
    for g in groups:
        item = dict(g)
        item['layers'] = item.pop('internal')
        # Source solves group and segment eagerly; these are static host-layer indices.
        item['hosts'] = tuple(
            min(int(np.searchsorted(edges, (a+b)/2, side='right'))-1, len(edges)-2)
            for a, b in zip(g['edges'][:-1], g['edges'][1:]))
        converted.append(item)
    data = dict(lattice=lattice, wavelength=wavelength, thickness=thickness,
                boundary_kxy=spectra['top', 'incoming'].kxy, boundary=boundary,
                groups=converted)
    return FieldState(_copy(data), 'source')
