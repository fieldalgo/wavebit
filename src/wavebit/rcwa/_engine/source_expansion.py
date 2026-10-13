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


def gaussian_waves(solver, source):
    """Midpoint angular spectrum, with Maxwell-transverse vector components."""
    from wavebit.rcwa._engine.illumination import _host
    from wavebit.rcwa._engine.illumination import _plane_basis
    scene = solver._scene
    q, pair, azimuth = resolve_wave(solver, source)
    q = _host(q).real
    pair = np.array([complex(_host(p)) for p in pair])
    material = scene.top if source.side == 'top' else scene.bot
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
    e0 = s*pair[0] + p*pair[1]
    axis = central_k/k
    nx, ny = source.sampling
    extent = source.spectral_extent/source.waist
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
            electric = e0 - kv*np.dot(kv, e0)/(k*k)
            weight = np.exp(-source.waist**2*(u*u+v*v)/4)*du*dv*source.waist**2/(4*np.pi)
            components.append((kv, electric, weight))
    if not components:
        raise ValueError('GaussianBeam quadrature contains no incoming propagating components.')
    norm = np.linalg.norm(sum(e*w for _, e, w in components))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError('GaussianBeam focal electric amplitude is zero or non-finite.')
    boundary_z = float(np.sum(_host(scene.thickness).real)) if source.side == 'top' else 0.
    origin = np.array([0., 0., boundary_z]) - np.asarray(source.focus).real
    waves = []
    xp, C = scene.xp, scene.complex_dtype
    for kv, electric, weight in components:
        kx, ky = [as_array(xp, v/k0, dtype=C, device=scene._array_device) for v in kv[:2]]
        eps_array = as_array(xp, er, dtype=C, device=scene._array_device)
        mu_array = as_array(xp, mu, dtype=C, device=scene._array_device)
        basis = _host(_plane_basis(xp, kx, ky, eps_array, mu_array, direction))[:3]
        pair = np.conj(basis).T @ electric
        size = np.linalg.norm(pair)
        if size == 0:
            continue
        amplitude = source.amplitude*(weight*size/norm)*np.exp(1j*kv@origin)
        waves.append(PlaneWave(side=source.side, kx=kv[0], ky=kv[1],
                               polarization=tuple(pair/size), amplitude=amplitude))
    return waves


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
        z = float(np.asarray(dipole.position)[2].real)
        if z <= 0 or z >= edges[-1] or np.any(np.abs(edges-z) <= tol):
            raise ValueError('ElectricDipole must be strictly inside a layer, away from its interfaces.')
        li = int(np.searchsorted(edges, z)-1)
        if isinstance(scene.structure, list):
            uniform = not scene.structure[li].get('shapes', [])
        else:
            uniform = True
            for raster in (scene.structure, scene.mu):
                if raster is not None:
                    a = _host(raster[li])
                    uniform = uniform and np.allclose(a, a[0, 0], atol=1e-12, rtol=1e-10)
        if not uniform:
            raise ValueError('ElectricDipole requires a uniform host layer: shapes=[] or a '
                             'raster with the same material in every spatial cell.')
        material = layers[li][0]
        blocks = [v for v in material.values() if not isinstance(v, dict)]
        blocks += list(material.get('mu_blocks', {}).values())
        for block in blocks:
            a = _host(block)
            if not np.allclose(a, np.eye(len(a))*a[0, 0], atol=1e-12, rtol=1e-10):
                raise ValueError('ElectricDipole currently requires a spatially uniform host layer; '
                                 'other layers may be patterned.')
        by_height.setdefault(z, (li, []))[1].append((dipole, weight))
    lattice = _host(scene.lattice.columns).real
    area = abs(np.linalg.det(lattice))
    # k0 is per scene length and p is SI: division by length_unit**3
    # converts the surface moment density and the z derivative together.
    scale = 1/(EPSILON_0*area*solver.length_unit**3)
    new_layers, jumps, new_edges = [], {}, [0.]
    for li, (material, d) in enumerate(layers):
        heights = sorted(z for z, (index, _) in by_height.items() if index == li)
        last = edges[li]
        for z in heights + [edges[li+1]]:
            new_layers.append((material, z-last))
            new_edges.append(z)
            if z in by_height:
                p = xp.zeros((kxy.shape[0], 3), dtype=C, device=scene._array_device)
                for dipole, weight in by_height[z][1]:
                    xy = as_array(xp, dipole.position[:2], dtype=C, device=scene._array_device)
                    moment = as_array(xp, dipole.moment, dtype=C, device=scene._array_device)
                    p = p + weight*scale*xp.exp(-1j*(kxy@xy))[:, None]*moment[None, :]
                normal = material['ezz_i'] @ p[:, 2]
                jumps[len(new_layers)] = xp.concat([
                    -1j*kxy[:, 0]*normal,
                    -1j*kxy[:, 1]*normal,
                    -1j*k0*(p[:, 1]-material['eyz']@normal),
                    1j*k0*(p[:, 0]-material['exz']@normal),
                ])
            last = z
    return new_layers, jumps, np.asarray(new_edges), sorted(by_height)
