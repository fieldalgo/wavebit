"""Independent bilinear reciprocity and spectral Green oracles for Yee radiation.

This uses pointwise products on the original four-dimensional records, not
the production separable surface transform. The Green oracle integrates the
Fourier delta pole of 1/(sum K_j^2-Omega^2), without using surface fields.
"""
import numpy as np
from wavebit.fdtd._kernel.grid import OFFSETS


def plane(grid, faces, k, electric, Omega, mu=1.):
    magnetic = np.cross(2*np.sin(k*grid.dx/2)/grid.dx, electric)/(Omega*mu)
    acc = {}
    for name, _, layout in faces:
        acc[name] = {}
        for label, field, slab in layout.parts():
            axes = [grid.low[d]+(np.arange(*bounds)+.5*OFFSETS[field][slab.c][d])*grid.dx
                    for d, bounds in enumerate((slab.x, slab.y, slab.z))]
            z, y, x = np.meshgrid(axes[2], axes[1], axes[0], indexing='ij')
            phase = np.exp(1j*(k[0]*x+k[1]*y+k[2]*z))
            acc[name][label] = ((electric if field == 'E' else magnetic)[slab.c]*phase)[None]
    return acc


def reciprocity(data, reference, faces, dx):
    total = 0j
    for name, sign, layout in faces:
        a, b = data[name], reference[name]
        def mean(v):
            return np.mean(v, axis=3-layout.P, keepdims=True)
        U = a['Eu']*mean(b['Hv'])-b['Eu']*mean(a['Hv'])
        V = a['Ev']*mean(b['Hu'])-b['Ev']*mean(a['Hu'])
        for value, axis in ((U, layout.v), (V, layout.u)):
            if layout.weights[axis] is not None:
                shape = [1]*4
                shape[3-axis] = len(layout.weights[axis])
                value *= np.asarray(layout.weights[axis]).reshape(shape)
        total += sign*(np.sum(U)-np.sum(V))*dx**2
    return total


def direct_field(grid, faces, data, prepared):
    """Project three independent reciprocal plane waves onto Cartesian axes."""
    result = np.zeros(prepared.k.shape, complex)
    for f in range(len(prepared.omega)):
        one = {name: {label: value[f:f+1] for label, value in part.items()}
               for name, part in data.items()}
        for j, k in enumerate(prepared.k[f]):
            n = prepared.unit_K[f, j]
            for c in range(3):
                e = np.eye(3)[c]-n*n[c]
                ref = plane(grid, faces, -k, e, prepared.omega[f], prepared.mu)
                W = reciprocity(one, ref, faces, grid.dx)
                result[f, j, c] = 1j*prepared.omega[f]*prepared.mu/(4*np.pi)*prepared.green_gain[f, j]*W
    return result.reshape((len(prepared.omega),)+prepared.shape+(3,))


def spectral_imag(radius, direction, dx, dt, order):
    """Imaginary outgoing vacuum Green at frequency 1 from its Fourier pole."""
    mu, w = np.polynomial.legendre.leggauss(order)
    phi = np.pi*np.arange(2*order)/order
    n = np.stack(np.broadcast_arrays(np.sqrt(1-mu[:, None]**2)*np.cos(phi),
        np.sqrt(1-mu[:, None]**2)*np.sin(phi), mu[:, None]), axis=-1)
    Omega = 2*np.sin(np.pi*dt)/dt
    k = 2*np.arcsin(Omega*dx*n/2)/dx
    jac = np.prod(np.sqrt(1-(Omega*dx*n/2)**2), axis=-1)
    phase = k@direction
    return np.array([Omega/(16*np.pi**2)*np.sum(np.cos(r*phase)*w[:, None]/jac)*np.pi/order
                     for r in radius])


def dipole_field(scene, result, prepared):
    """Discrete electric/magnetic current moment, independent of monitor data.

    The source stamp is reconstructed from the cubic B-spline definition,
    not from the production stamp helper or its prepared parameter array.
    """
    grid = result.plan.grid
    source = scene.sources[0]
    component, center = source.component, source.center
    axis, field = 'xyz'.index(component[1]), component[0]
    series = result.sources[source.name]
    dt = result.plan.dt
    freqs = np.asarray(tuple(1./w for w in result['ff'].wavelengths))
    drive = dt*np.sum(np.asarray(series.drive)[None, :]*np.exp(
        2j*np.pi*freqs[:, None]*np.asarray(series.t)[None, :]), axis=-1)
    coords, weights = [], []
    for j in range(3):
        x = grid.low[j]+(np.arange(grid.n[j])+.5*OFFSETS[field][axis][j])*grid.dx
        u = abs((x-center[j])/grid.dx)
        b = np.where(u < 1, (4-6*u*u+3*u**3)/6, np.where(u < 2, (2-u)**3/6, 0.))
        coords.append(x[b > 0]); weights.append(b[b > 0])
    z, y, x = np.meshgrid(coords[2], coords[1], coords[0], indexing='ij')
    weight = weights[2][:, None, None]*weights[1][None, :, None]*weights[0][None, None, :]
    expected = np.empty(prepared.k.shape, complex)
    for f, Omega in enumerate(prepared.omega):
        for j, k in enumerate(prepared.k[f]):
            K = 2*np.sin(k*grid.dx/2)/grid.dx
            stamp = np.sum(weight*np.exp(-1j*(k[0]*x+k[1]*y+k[2]*z)))
            unit = np.eye(3)[axis]
            vector = unit-K*K[axis]/Omega**2 if field == 'E' else -np.cross(K, unit)/Omega
            expected[f, j] = 1j*Omega/(4*np.pi)*prepared.green_gain[f, j]*vector*stamp*drive[f]
    return expected.reshape((len(freqs),)+prepared.shape+(3,))
