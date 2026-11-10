"""Convert prescribed sources into boundary waves and internal field jumps."""
import copy

import numpy as np

from wavebit._common.backend import as_array
from wavebit.rcwa._engine.sources import PlaneWave


EPSILON_0 = 8.8541878188e-12


def resolve_wave(solver, wave):
    from wavebit.rcwa._engine.illumination import _host
    scene = solver._scene
    xp = scene.xp
    probe = copy.copy(scene)
    probe.incidence_side = wave.side
    values = dict(solver._defaults)
    if wave.theta is not None or wave.phi is not None:
        values.update(kx=None, ky=None, theta=wave.theta, phi=wave.phi)
    else:
        values.update(theta=None, phi=None)
        if wave.kx is not None or wave.ky is not None:
            values.update(kx=wave.kx, ky=wave.ky)
    if wave.polarization is not None:
        values.update(polarization=wave.polarization, psi=None, chi=None)
    elif wave.psi is not None or wave.chi is not None:
        values.update(polarization=None, psi=wave.psi, chi=wave.chi)
    probe.kx = probe.ky = probe.theta = probe.phi = None
    probe.polarization = probe.psi = probe.chi = None
    for key, value in values.items():
        setattr(probe, key, value)
    q = xp.stack(probe._incident_kxy(scene.wavelength)) * (2*np.pi/scene.wavelength)
    if not np.all(np.isfinite(_host(q))) or np.any(np.abs(_host(q).imag) > 1e-12):
        raise ValueError('Coherent sources require finite real kx and ky.')
    # The frame azimuth at k_par = 0 is the ordinary solve's: `phi`, or `phi + pi` from a
    # negative-index side, the limit of theta -> 0+ (`CoreSolveMixin._frame_azimuth`).
    azimuth = probe._frame_azimuth()
    return q, probe._incidence_pair(), 0. if azimuth is None else azimuth


def _carries_derivative(value):
    """Whether ``value`` is, or holds, a JAX tracer or a PyTorch tensor that requires grad."""
    from wavebit.rcwa._engine.engine import _holds_tracer
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, (list, tuple)):
            stack.extend(item)
        elif _holds_tracer(item) or bool(getattr(item, 'requires_grad', False)):
            return True
    return False


def gaussian_waves(solver, source):
    """Midpoint angular spectrum, with Maxwell-transverse vector components."""
    from wavebit.rcwa._engine.illumination import _host
    from wavebit.rcwa._engine.illumination import _plane_basis
    scene = solver._scene
    q, pair, azimuth = resolve_wave(solver, source)
    material = scene.top if source.side == 'top' else scene.bot
    moving = [name for name, value in (
        ('direction (theta, phi, kx, ky)', (q, azimuth)), ('wavelength', scene.wavelength),
        (f'{source.side} half-space epsilon and mu', (material.epsilon, material.mu_value)),
        ('waist', source.waist))
        if _carries_derivative(value)]
    if moving:
        # The quadrature directions, weights and transverse projections are built on the host
        # from these values, so a derivative through them would be silently partial.
        raise NotImplementedError(
            'GaussianBeam builds its angular spectrum from concrete values; a derivative with '
            f'respect to its {" or ".join(moving)} is not carried. Derivatives with respect to '
            'its amplitude, polarization and focus and to the layers are. Differentiate by finite '
            'differences, or expand the beam into PlaneWave sources.')
    q = _host(q).real
    xp, C = scene.xp, scene.complex_dtype
    # Every component is linear in the unit Jones pair, which stays a backend value; the
    # directions, weights and transverse projections below are host constants.
    jones = xp.stack([as_array(xp, a, dtype=C, device=scene._array_device) for a in pair])
    er, mu = complex(_host(material.epsilon)), complex(_host(material.mu_value))
    if er.real < 0 and mu.real < 0:
        # A negative-index medium's beam would be built from waves whose phase recedes from the
        # structure while their power arrives; its angular spectrum, transversality and focus
        # phase are not written for that, so it is refused rather than built forward-phase.
        raise ValueError(
            'GaussianBeam requires a lossless positive-index isotropic incident medium; the '
            f'{source.side} half-space is a negative-index one (Re(eps) < 0 and Re(mu) < 0). '
            'Expand the beam into PlaneWave sources lit from that side, which are supported, '
            'or illuminate from the other side.')
    if er.imag or mu.imag or er.real <= 0 or mu.real <= 0:
        raise ValueError('GaussianBeam requires a lossless positive-index isotropic incident medium.')
    k0 = float(2*np.pi/_host(scene.wavelength))
    k = k0*np.sqrt(er.real*mu.real)
    if np.linalg.norm(q) >= k:
        raise ValueError('GaussianBeam central direction must propagate into the stack.')
    direction = -1 if source.side == 'top' else 1
    central_k = np.r_[q, direction*np.sqrt(k*k-q@q)]
    if np.linalg.norm(q) < 1e-14*k:
        if np.any(_host(azimuth).imag):
            raise ValueError('GaussianBeam azimuth must be real.')
        phi = float(_host(azimuth).real)
        s = np.array([-np.sin(phi), np.cos(phi), 0.])
        p = np.array([-np.cos(phi), -np.sin(phi), 0.])
    else:
        s = np.array([-q[1], q[0], 0.])/np.linalg.norm(q)
        p = -direction*np.cross(s, central_k/k)
    e0 = np.stack([s, p], axis=1)
    axis = central_k/k
    nx, ny = source.sampling
    # A 0-d array or tensor waist is read as the number it holds; it sets the quadrature.
    waist = source.waist if np.isscalar(source.waist) else float(_host(source.waist))
    extent = source.spectral_extent/waist
    du, dv = 2*extent/nx, 2*extent/ny
    components = []
    for i in range(nx):
        for j in range(ny):
            u, v = -extent+(i+.5)*du, -extent+(j+.5)*dv
            if u*u+v*v >= k*k:
                continue
            kv = u*p + v*s + np.sqrt(k*k-u*u-v*v)*axis
            if direction*kv[2] <= 1e-12*k:
                continue
            electric = e0 - np.outer(kv, kv @ e0)/(k*k)
            weight = np.exp(-waist**2*(u*u+v*v)/4)*du*dv*waist**2/(4*np.pi)
            components.append((kv, electric, weight))
    if not components:
        raise ValueError('GaussianBeam quadrature contains no incoming propagating components.')
    focal = as_array(xp, sum(e*w for _, e, w in components), dtype=C,
                     device=scene._array_device) @ jones
    norm = xp.sqrt(xp.sum(xp.real(focal*xp.conj(focal))))
    if not np.isfinite(_host(norm)) or _host(norm) == 0:
        raise ValueError('GaussianBeam focal electric amplitude is zero or non-finite.')
    # Each component is referenced at x=y=0 on its own face. The top face's height stays a
    # backend value: the focus is fixed in global z, so the phase a component gathers from the
    # focus to that face depends on every layer thickness, and so does its derivative.
    # A focus that carries a derivative stays a backend value; it enters only these phases.
    live = _carries_derivative(source.focus)
    if live:
        focus = [xp.real(as_array(xp, source.focus[i], dtype=C, device=scene._array_device))
                 for i in range(3)]
    else:
        focus = np.asarray(source.focus).real
    if source.side == 'top':
        height = xp.sum(xp.real(as_array(xp, scene.thickness, dtype=C,
                                         device=scene._array_device))) - focus[2]
    elif live:
        height = 0.-focus[2]
    else:
        height = xp.real(as_array(xp, 0.-focus[2], dtype=C, device=scene._array_device))
    waves = []
    for kv, electric, weight in components:
        kx, ky = [as_array(xp, v/k0, dtype=C, device=scene._array_device) for v in kv[:2]]
        eps_array = as_array(xp, er, dtype=C, device=scene._array_device)
        mu_array = as_array(xp, mu, dtype=C, device=scene._array_device)
        basis = _host(_plane_basis(xp, kx, ky, eps_array, mu_array, direction))[:3]
        pair = as_array(xp, np.conj(basis).T @ electric, dtype=C,
                        device=scene._array_device) @ jones
        size = xp.sqrt(xp.sum(xp.real(pair*xp.conj(pair))))
        if _host(size) == 0:
            continue
        pair = pair/as_array(xp, size, dtype=C, device=scene._array_device)
        if live:
            phase = (float(-kv[0])*focus[0] + float(-kv[1])*focus[1]) + float(kv[2])*height
        else:
            phase = float(kv[0]*-focus[0] + kv[1]*-focus[1]) + float(kv[2])*height
        amplitude = source.amplitude*(weight*size/norm)*xp.exp(1j*phase)
        waves.append(PlaneWave(side=source.side, kx=kv[0], ky=kv[1],
                               polarization=(pair[0], pair[1]), amplitude=amplitude))
    return waves


def _uniform_tolerance(a):
    """How close to uniform a host's raster or operator block has to be, at its own precision.

    Float64 keeps the tolerances it always had. A coarser array is held to its own resolution,
    relative to its magnitude: the solve rebases every block to the centre origin by
    ``phase_i * conj(phase_j)``, whose diagonal is ``|phase|**2``. That is exactly 1 while
    ``bloch_k`` is zero (the phases are +-1) and 1 +- 1e-8 in complex64 once it is not, so the
    float64 ``atol=1e-12`` called a uniform single-precision host patterned.
    """
    eps = float(np.finfo(np.asarray(a).real.dtype).eps)
    if eps <= np.finfo(np.float64).eps:
        return dict(atol=1e-12, rtol=1e-10)
    tol = 32*eps
    return dict(atol=tol*float(np.max(np.abs(a), initial=0.)), rtol=tol)


def dipole_host_layer(edges, z, tol):
    """The index of the layer that holds a dipole at height ``z``, strictly inside it."""
    if z <= 0 or z >= edges[-1] or np.any(np.abs(edges-z) <= tol):
        raise ValueError('ElectricDipole must be strictly inside a layer, away from its interfaces.')
    return int(np.searchsorted(edges, z)-1)


def uniform_host_layer(scene, li):
    """Whether layer ``li`` of ``scene`` is one material: no shapes, or a raster of one value."""
    from wavebit.rcwa._engine.illumination import _host
    if isinstance(scene.structure, list):
        return not scene.structure[li].get('shapes', [])
    uniform = True
    for raster in (scene.structure, scene.mu):
        if raster is not None:
            a = _host(raster[li])
            uniform = uniform and np.allclose(a, a[0, 0], **_uniform_tolerance(a))
    return uniform


def dipole_layers(solver, group, layers, k0, kxy):
    """Split source-bearing layers and integrate the electric-current delta in z.

    With h=Z0 H and exp(-i omega t), a moment p gives J=-i omega p delta.
    Elimination of Ez gives its singular part -eps_zz^-1 pz/(eps0 area).
    Integrating the four first-order Maxwell equations gives the jump below.
    No external implementation or emission matrix is used.
    """
    from wavebit.rcwa._engine.illumination import _host
    scene = solver._scene
    xp, C = scene.xp, scene.complex_dtype
    edges = np.r_[0., np.cumsum([float(_host(d).real) for _, d in layers])]
    if not group['dipoles']:
        return layers, {}, edges, []
    tol = 1e-12*max(1., edges[-1])
    by_height = {}
    for dipole, weight in group['dipoles']:
        # The host value of z chooses the host layer; a z that carries a derivative also
        # stays live in the segment thicknesses below. Choosing the layer is discrete.
        z = (float(_host(dipole.position[2]).real) if _carries_derivative(dipole.position)
             else float(np.asarray(dipole.position)[2].real))
        li = dipole_host_layer(edges, z, tol)
        if not uniform_host_layer(scene, li):
            raise ValueError('ElectricDipole requires a uniform host layer: shapes=[] or a '
                             'raster with the same material in every spatial cell.')
        material = layers[li][0]
        blocks = [v for v in material.values() if not isinstance(v, dict)]
        blocks += list(material.get('mu_blocks', {}).values())
        for block in blocks:
            a = _host(block)
            if not np.allclose(a, np.eye(len(a))*a[0, 0], **_uniform_tolerance(a)):
                raise ValueError('ElectricDipole currently requires a spatially uniform host layer; '
                                 'other layers may be patterned.')
        by_height.setdefault(z, (li, []))[1].append((dipole, weight))
    # A plane's position stays live when one of its sources' z carries a derivative. Two
    # dipoles on one plane would share that single position, so a derivative with respect to
    # the z of one of them is refused there rather than returned as the plane's.
    planes = {}
    for z, (_, entries) in by_height.items():
        live = [d.position[2] for d, _ in entries if _carries_derivative(d.position[2])]
        if live and len(entries) > 1:
            raise NotImplementedError(
                'Two ElectricDipole sources share the source plane z = '
                f'{z:g} and a derivative with respect to the z position of one of them is '
                'requested; one plane has one position. Offset them in z.')
        planes[z] = live[0] if live else None
    lattice = _host(scene.lattice.columns).real
    area = abs(np.linalg.det(lattice))
    # k0 is per scene length and p is SI: division by length_unit**3
    # converts the surface moment density and the z derivative together.
    scale = 1/(EPSILON_0*area*solver.length_unit**3)
    new_layers, jumps, new_edges = [], {}, [0.]
    # The host edges above decide where each source sits; the segment thicknesses stay backend
    # values. A layer without a source keeps its own thickness, and a host layer's segments
    # are measured from its live lower face to the source planes, which are fixed in global z:
    # every thickness at or below the host moves them, and its derivative has to see that.
    start = 0.
    for li, (material, d) in enumerate(layers):
        heights = sorted(z for z, (index, _) in by_height.items() if index == li)
        top = start + d
        last = start
        for z in heights:
            plane = z if planes[z] is None else xp.real(
                as_array(xp, planes[z], dtype=C, device=scene._array_device))
            new_layers.append((material, plane-last))
            new_edges.append(z)
            p = xp.zeros((kxy.shape[0], 3), dtype=C, device=scene._array_device)
            for dipole, weight in by_height[z][1]:
                if _carries_derivative(dipole.position):
                    xy = xp.stack([as_array(xp, dipole.position[i], dtype=C,
                                            device=scene._array_device) for i in range(2)])
                else:
                    xy = as_array(xp, dipole.position[:2], dtype=C, device=scene._array_device)
                if _carries_derivative(dipole.moment):
                    moment = xp.stack([as_array(xp, dipole.moment[i], dtype=C,
                                                device=scene._array_device) for i in range(3)])
                else:
                    moment = as_array(xp, dipole.moment, dtype=C, device=scene._array_device)
                p = p + weight*scale*xp.exp(-1j*(kxy@xy))[:, None]*moment[None, :]
            normal = material['ezz_i'] @ p[:, 2]
            jumps[len(new_layers)] = xp.concat([
                -1j*kxy[:, 0]*normal,
                -1j*kxy[:, 1]*normal,
                -1j*k0*(p[:, 1]-material['eyz']@normal),
                1j*k0*(p[:, 0]-material['exz']@normal),
            ])
            last = plane
        new_layers.append((material, top-last if heights else d))
        new_edges.append(edges[li+1])
        start = top
    return new_layers, jumps, np.asarray(new_edges), sorted(by_height)
