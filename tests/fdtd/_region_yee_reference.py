"""Independent indexed SI Maxwell-Yee reference for one cell-snapped Region.

Only public objects and SI constants are imported. Grid indices, material masks,
finite differences and their tangent equations are written here independently.
This tests the discrete cell-snapping contract, not continuum interface accuracy.
"""
import numpy as np

from wavebit import Material, Units, fdtd
from wavebit.materials import ConductiveMaterial
from wavebit.units import C0, EPSILON0, Z0

DX = .05e-6
DT = .5*DX/(C0*np.sqrt(3.))
POINTS = ((2.25, 1.3, 600., 1.2), (1.8, 1.1, 900., .7))
LABELS = tuple(f+a for f in ('E', 'H') for a in 'xyz')
TOL = 2e-12


def geometry(rotation=0):
    def rotate(x):
        return np.roll(np.asarray(x), rotation)
    cells = rotate((13, 10, 15))
    center = rotate((.1, -.15, .2))*1e-6
    return dict(cells=cells, size=rotate((.61, .49, .71))*1e-6,
                center=center, low=center-cells*DX/2,
                region_size=rotate((.305, .225, .365))*1e-6,
                region_center=rotate((.0375, -.2025, .1625))*1e-6,
                lo=rotate((2, 2, 3)), hi=rotate((8, 6, 10)),
                probe=rotate((4, 3, 6)))


def mask(rotation=0, mutant=None):
    geo = geometry(rotation)
    lo, hi = geo['lo'].copy(), geo['hi'].copy()
    if mutant == 'shift':
        lo[rotation] += 1
        hi[rotation] += 1
    if mutant == 'floor':
        lo[(1+rotation) % 3] -= 1
    if mutant == 'wrong_axis':
        a, b = rotation, (rotation+1) % 3
        lo[[a, b]], hi[[a, b]] = lo[[b, a]], hi[[b, a]]
    result = np.zeros(tuple(geo['cells'][::-1]))
    result[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]] = 1.
    return result


def patterns(rotation=0):
    geo = geometry(rotation)
    n = geo['cells']
    modes = np.roll([1, 2, 1], rotation)
    result = []
    for field in ('E', 'H'):
        parts = []
        for c in range(3):
            axes = [(np.arange(n[a])+.5*(a == c if field == 'E' else a != c))/n[a]
                    for a in range(3)]
            z, y, x = np.meshgrid(axes[2], axes[1], axes[0], indexing='ij')
            phase = 2*np.pi*sum(m*v for m, v in zip(modes, (x, y, z)))
            local = (c-rotation) % 3
            if field == 'E':
                part = (.8+.17*local)*np.cos(phase+.31*local)+.23*np.sin(2*phase-.2*local)
            else:
                part = (.41+.13*local)*np.sin(phase+.47*local)+.19*np.cos(2*phase+.3*local)
            parts.append(part)
        result.append(np.stack(parts)/(Z0 if field == 'H' else 1.))
    return tuple(result)


def problem(point=POINTS[0], rotation=0, unit='um', background=(1., 1., 0.),
            lossless=False, arrays=None):
    units = Units(length=unit)
    scale = units.length_metres
    geo = geometry(rotation)
    electric, magnetic = patterns(rotation) if arrays is None else arrays
    eps, mu, sigma, amplitude = point
    def material(values):
        e, m, s = values
        return (Material(epsilon=e, mu=m) if lossless else
                ConductiveMaterial(epsilon=e, mu=m, conductivity=s))
    records = []
    for field in ('E', 'H'):
        for c in range(3):
            offset = np.array([.5*(a == c if field == 'E' else a != c) for a in range(3)])
            at = (geo['low']+(geo['probe']+offset)*DX)/scale
            label = field+'xyz'[c]
            records.append(fdtd.Record(label, component=label, at=tuple(at)))
    return fdtd.Problem(units=units, scene=fdtd.Scene(
        domain=fdtd.Domain(size=tuple(geo['size']/scale), center=tuple(geo['center']/scale)),
        boundaries=fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.Periodic()),
        background=material(background), structures=(fdtd.Structure('body',
            fdtd.Region(size=tuple(geo['region_size']/scale),
                        center=tuple(geo['region_center']/scale)), material((eps, mu, sigma))),)),
        initial=fdtd.InitialField(E=electric, H=magnetic, amplitude=amplitude),
        outputs=fdtd.Outputs(final_fields=True, monitors=tuple(records)))


def numerics(steps, block=None, unit='um'):
    return fdtd.Numerics(grid=fdtd.UniformGrid(spacing=DX/Units(length=unit).length_metres),
                         duration=fdtd.Duration(steps=steps), courant=.5, checkpoint_steps=block)


def difference(value, axis, forward, periodic=True):
    # Physical xyz maps to the last three storage dimensions in reverse order.
    dimension = value.ndim-1-axis
    count = value.shape[dimension]
    indices = (np.arange(count)+(1 if forward else -1)) % count
    adjacent = np.take(value, indices, axis=dimension)
    if not periodic:
        boundary = [slice(None)]*value.ndim
        boundary[dimension] = -1 if forward else 0
        adjacent[tuple(boundary)] = 0.
    return (adjacent-value if forward else value-adjacent)/DX


def curl(value, forward, periodic=True):
    # Component dimension is immediately before spatial z,y,x, including tangents.
    parts = [np.take(value, c, axis=-4) for c in range(3)]
    return np.stack((difference(parts[2], 1, forward, periodic)-difference(parts[1], 2, forward, periodic),
                     difference(parts[0], 2, forward, periodic)-difference(parts[2], 0, forward, periodic),
                     difference(parts[1], 0, forward, periodic)-difference(parts[0], 1, forward, periodic)), axis=-4)


def reference(point=POINTS[0], rotation=0, steps=17, background=(1., 1., 0.),
              differentiate='body', mutant=None):
    eps_b, mu_b, sigma_b, amplitude = point
    chi = mask(rotation, mutant)
    eps = background[0]+chi*(eps_b-background[0])
    mu = background[1]+chi*(mu_b-background[1])
    sigma = background[2]+chi*(sigma_b-background[2])
    r = sigma*DT/(2*EPSILON0)
    denominator = eps+r
    q, g, b = (eps-r)/denominator, C0*DT/denominator, C0*DT/mu
    if mutant == 'omit_mu':
        b = np.full_like(b, C0*DT)
    if mutant == 'omit_sigma_denominator':
        q, g = (eps-r)/eps, C0*DT/eps
    support = chi if differentiate == 'body' else 1-chi
    dq = np.zeros((4, 1)+chi.shape)
    dg, db = np.zeros_like(dq), np.zeros_like(dq)
    dq[0, 0] = 2*r*support/denominator**2
    dg[0, 0] = -C0*DT*support/denominator**2
    db[1, 0] = -C0*DT*support/mu**2
    dr = support*DT/(2*EPSILON0)
    dq[2, 0] = -2*eps*dr/denominator**2
    dg[2, 0] = -C0*DT*dr/denominator**2
    if mutant == 'omit_dq':
        dq[:] = 0
    e0, h0 = patterns(rotation)
    h0 = Z0*h0
    e, h = amplitude*e0, amplitude*h0
    de, dh = np.zeros((4,)+e.shape), np.zeros((4,)+h.shape)
    de[3], dh[3] = e0, h0
    records, tangents = [], []
    ix, iy, iz = geometry(rotation)['probe']
    for _ in range(steps):
        ce = curl(e, True, periodic=mutant != 'wrong_wrap')
        if mutant == 'wrong_curl':
            ce = -ce
        hn = h-b*ce
        dhn = dh-b*curl(de, True)-db*ce
        ch = curl(hn, False)
        en = q*e+g*ch
        den = q*de+g*curl(dhn, False)+dq*e+dg*ch
        e, h, de, dh = en, hn, den, dhn
        records.append(np.concatenate((e[:, iz, iy, ix], h[:, iz, iy, ix])))
        tangents.append(np.concatenate((de[:, :, iz, iy, ix], dh[:, :, iz, iy, ix]), axis=1))
    return dict(E=e, H=h, records=np.asarray(records).T,
                dE=de, dH=dh, drecords=np.asarray(tangents).transpose(1, 2, 0))


def weights(shape, steps):
    nz, ny, nx = shape[-3:]
    spatial = (1+.1*np.cos(2*np.pi*np.arange(nx)/nx)[None, None, :]
               +.07*np.sin(2*np.pi*np.arange(ny)/ny)[None, :, None])
    field = np.array([1., .7, 1.3, .6, 1.1, .9])[:, None, None, None]*spatial
    record = np.array([1., .7, 1.3, .6, 1.1, .9])[:, None]*(1+.2*np.arange(steps)/steps)
    return field, record


def objectives(expected):
    values = np.concatenate((expected['E'], expected['H']))
    tangents = np.concatenate((expected['dE'], expected['dH']), axis=1)
    fw, rw = weights(expected['E'].shape, expected['records'].shape[1])
    return ((float(np.mean(fw*values**2)), np.mean(2*fw*values*tangents, axis=(1, 2, 3, 4))),
            (float(np.mean(rw*expected['records']**2)),
             np.mean(2*rw*expected['records']*expected['drecords'], axis=(1, 2))))
