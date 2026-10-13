"""Measure equivalent polygon placements before and after the boundary-tie repair.

The comparisons are geometric identities: translate body and samples together,
cycle a polygon's first vertex, or add one full turn. The public mirror pair
uses the same physical bar with its faces on voxel boundaries. No lowering
patch, golden rewrite, or fitted electromagnetic acceptance threshold is used.
Run from the repository root; records append to the study log.
"""
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np

import wavebit.fdtd._legacy as fd
from wavebit.fdtd._shapes import Box, Prism, occupancy


def quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(*args, **kwargs)


def mirror_series(half, backend='numpy'):
    dx, width = .05, .6
    shape = fd.Box(size=(.4, .2, .2))
    domain = fd.Domain(size=(width if half else 2*width, width, width),
                       center=(width/2 if half else 0., 0., 0.))
    x = (fd.Symmetry(parity=1), fd.PEC()) if half else fd.PEC()
    def mode(x, y, z):
        return np.cos(np.pi*x/(2*width))*np.cos(np.pi*y/width)*np.ones_like(z)
    scene = fd.Scene(domain=domain, resolution=fd.Resolution(dx=dx),
                     boundaries=fd.Boundaries(x=x, y=fd.PEC(), z=fd.PEC()),
                     run=fd.Run(steps=300, courant=.5),
                     structures=(fd.Structure(shape, 4.),),
                     initial=fd.InitialField(E=(None, None, mode)),
                     monitors=(fd.Record('probe', at=(.1125, .0125, 0.), component='Ez'),))
    return np.asarray(quiet(fd.run, scene, backend=backend)['probe'].values)


def measure():
    points = np.array([[.175, .1, 0.], [-.175, .1, 0.], [.1, .075, .1]])
    shifts = (0., .013, .137, .6, 3.)
    rows = []
    for shift in shifts:
        body = Box((.4, .2, .2), center=(shift, shift, shift))
        fill, moment = occupancy(body, points+shift, .05)
        rows.append(dict(shift=shift, fill=fill.tolist(), moment=moment.tolist()))
    axis = np.arange(-7, 8)*.025
    x, y = np.meshgrid(axis, axis, indexing='ij')
    samples = np.stack([x, y, np.zeros_like(x)], axis=-1)
    normals = []
    weights = []
    for angle in (45., 405.):
        body = Box((.3, .3, .4), angle=np.deg2rad(angle))
        normals.append(body.surface(samples)[1])
        weights.append(occupancy(body, samples, .05)[1])
    verts = np.array([[-.15, -.15], [.15, -.15], [.15, .15], [-.15, .15]])
    cyclic = [Prism(np.roll(verts, k, axis=0), .4, angle=np.pi/4).surface(samples)[1]
              for k in range(4)]
    # Opposite choices of a normal give the same tensor, so compare n n^T.
    tensor = lambda n: n[..., :, None]*n[..., None, :]
    full, half = mirror_series(False), mirror_series(True)
    return dict(translated_boundary=rows,
                turn_normal_gap=float(np.max(np.abs(tensor(normals[0])-tensor(normals[1])))),
                turn_moment_gap=float(np.max(np.abs(weights[0]-weights[1]))),
                cyclic_normal_gap=float(max(np.max(np.abs(tensor(n)-tensor(cyclic[0])))
                                            for n in cyclic)),
                mirror_field_relative_gap=float(np.max(np.abs(full-half))/np.max(np.abs(full))))


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else 'measurement'
    digest = hashlib.sha256()
    for path in sorted(Path('src/wavebit/fdtd').rglob('*.py')):
        digest.update(path.relative_to('src/wavebit/fdtd').as_posix().encode())
        digest.update(path.read_bytes().replace(b'\r\n', b'\n'))
    row = dict(label=label, source=digest.hexdigest(), **measure())
    output = Path('studies/fdtd/polygon_ties/records/fdtd_polygon_ties.log')
    with output.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, allow_nan=False)+'\n')
    print(json.dumps(row, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
