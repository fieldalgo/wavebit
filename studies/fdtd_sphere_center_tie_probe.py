"""A sphere's centre derivative where a voxel face passes through its centre plane.

``_exact.clipped_ball_volume`` sums an odd extension ``sgn(a) sgn(b) sgn(c) F(|a|, |b|, |c|)``
of the corner primitive over a box's eight corners, ``a`` the corner's offset from the centre.
Its derivative in ``a`` is ``F_a(|a|)``, even and continuous, so the fill is smooth in the
centre even where a box face contains the centre (``a = 0``). If reverse mode takes ``|a|``'s
derivative at zero as zero, that term drops there.

Modes:

``fill``   per-voxel fills of one ball at points whose voxel faces straddle the centre plane on
           each axis in turn and on all three at once, reverse mode on JAX and Torch against a
           NumPy central difference in each centre coordinate and the radius;
``run``    a lone ranged sphere in a 16-cubed PEC box: the centre and radius derivatives of the
           final electric energy in a fixed slab, at centres on node planes, on half-node planes
           and off both, on all three backends.

Results go to ``studies/results/fdtd_sphere_center_tie_<mode>[_<label>].json``.
"""
import json
from pathlib import Path
import sys
import warnings

import numpy as np

import wavebit.fd as fd
from wavebit.fd._runtime import runtime


RESULTS = Path(__file__).resolve().parent / 'results'
DX = .1
RADIUS = .22
#: Off-plane centre coordinates, so only the axis under test ties.
OFF = (.013, -.021, .017)


def _ball(rt, radius, center):
    from wavebit.fd._shapes._bodies import Sphere
    return Sphere(radius, center)


def fill_points(axis, all_axes=False):
    """Voxel centres half a cell either side of the centre plane on ``axis`` (all three axes when
    ``all_axes``), across a 5 x 5 patch of the other two: their voxel faces contain the plane.

    The other two axes are offset by .37 and .41 cells. A first choice of .3 and .4 put a face
    at -.22 from the centre, exactly the radius: the tangent plane is a true jump of the second
    derivative, and the central difference there converged to reverse mode only as ``h``."""
    base = np.array(OFF)
    offsets = np.arange(-2, 3) * DX
    pts = []
    for s in (-.5, .5):
        for u in offsets:
            for w in offsets:
                p = base.copy()
                others = [j for j in range(3) if j != axis]
                p[axis] += s * DX
                p[others[0]] += u + (s * DX if all_axes else .37 * DX)
                p[others[1]] += w + (s * DX if all_axes else .41 * DX)
                pts.append(p)
    return np.array(pts)


def corner_points():
    """Voxels with a face in the centre's z plane and a far corner past the radius on x and y
    at once, still cut: the regime of ``_ball_corner`` with both chords gone and only the
    quarter disc left, which none of :func:`fill_points` reaches (its such voxels are empty).
    Found by a surviving mutant of the one-chord limit."""
    base = np.array(OFF)
    pts = []
    for sz in (-.5, .5):
        for u in (-.18, .18):
            for w in (-.19, .19):
                pts.append(base + np.array((u, w, sz * DX)))
    return np.array(pts)


def fills(backend, radius, center, points):
    rt = runtime(backend)
    xp = rt.xp
    body = _ball(rt, radius, center)
    return body.exact_fill(rt.array(points, 'float64'), DX)


def fill_mode(label='current'):
    warnings.filterwarnings('ignore')
    out = {}
    for case, (axis, all_axes) in {'x': (0, False), 'y': (1, False), 'z': (2, False),
                                   'xyz': (0, True), 'corner': (None, None)}.items():
        points = corner_points() if axis is None else fill_points(axis, all_axes)
        weights = np.linspace(1., 2., len(points))
        leaves = np.array((RADIUS, *OFF))

        def objective(v, backend):
            f = fills(backend, v[0], (v[1], v[2], v[3]), points)
            rt = runtime(backend)
            return (f * rt.array(weights, 'float64')).sum()
        h = 1e-6
        reference = np.array([(float(objective(leaves + d, 'numpy'))
                               - float(objective(leaves - d, 'numpy'))) / (2 * h)
                              for d in np.eye(4) * h])
        row = dict(points=len(points), central=reference.tolist())
        import torch
        v = torch.tensor(leaves, dtype=torch.float64, requires_grad=True)
        row['torch'] = torch.autograd.grad(objective(v, 'torch'), v)[0].numpy().tolist()
        runtime('jax')
        import jax
        import jax.numpy as jnp
        row['jax'] = np.asarray(jax.grad(lambda x: objective(x, 'jax'))(
            jnp.asarray(leaves))).tolist()
        for backend in ('torch', 'jax'):
            got = np.array(row[backend])
            row[backend + '_relative'] = (abs(got - reference) / np.maximum(abs(reference),
                                                                            1e-300)).tolist()
        out[case] = row
        print(case, row, flush=True)
    (RESULTS / f'fdtd_sphere_center_tie_fill_{label}.json').write_text(
        json.dumps(out, indent=2) + '\n', encoding='utf-8')


def scene(v):
    ball = fd.Sphere(radius=v[0], center=(v[1], v[2], v[3]), range=((-.36, .36),) * 3)
    return fd.Scene(domain=fd.Domain(size=(1.6,) * 3), resolution=fd.Resolution(dx=DX),
                    boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()),
                    run=fd.Run(steps=32),
                    structures=(fd.Structure(ball, fd.Material(eps=2.25)),),
                    sources=(fd.Dipole('d', center=(.05, -.55, -.02), component='Ey',
                                       band=fd.Band(wavelength=3., df=.03), delay=-18.),))


def energy(v, backend, plan):
    return (fd.run(scene(v), backend, plan=plan).final.E[:, 5:11, 5:11, :] ** 2).sum()


#: Grid low corner -.8, so node planes are at multiples of .1 and half-node planes at .05 + .1k.
CENTRES = {'off': OFF, 'node': (0., 0., 0.), 'half': (.05, .05, .05),
           'half_z': (.013, -.021, .05), 'node_z': (.013, -.021, 0.)}


def run_mode(label='current'):
    warnings.filterwarnings('ignore')
    out = {}
    h = 3e-6
    for name, centre in CENTRES.items():
        point = np.array((RADIUS, *centre))
        p = fd.plan(scene(point))
        reference = np.array([(float(energy(point + d, 'numpy', p))
                               - float(energy(point - d, 'numpy', p))) / (2 * h)
                              for d in np.eye(4) * h])
        import torch
        v = torch.tensor(point, dtype=torch.float64, requires_grad=True)
        row = dict(centre=centre, central=reference.tolist(),
                   torch=torch.autograd.grad(energy(v, 'torch', p), v)[0].numpy().tolist())
        runtime('jax')
        import jax
        import jax.numpy as jnp
        row['jax'] = np.asarray(jax.grad(lambda x: energy(x, 'jax', p))(
            jnp.asarray(point))).tolist()
        for backend in ('torch', 'jax'):
            row[backend + '_relative'] = float(np.max(abs(np.array(row[backend]) - reference)
                                                      / abs(reference)))
        out[name] = row
        print(name, row, flush=True)
    (RESULTS / f'fdtd_sphere_center_tie_run_{label}.json').write_text(
        json.dumps(out, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'fill'
    label = sys.argv[2] if len(sys.argv) > 2 else 'current'
    {'fill': fill_mode, 'run': run_mode}[mode](label)
