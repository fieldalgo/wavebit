"""How far 'plane' and 'split' sit from the exact moment, rim voxels and the rest, at R/h 4/8/16.

The exact moment is the study's divergence-theorem form (checked against arc quadrature by
check_exact_tensor.py). Cut voxels = 1e-12 < fill < 1 - 1e-12 over the six Yee staggerings,
plane-dead voxels excluded,
cylinder height 2R, centre (0.137, -0.071, 0.043) h. 'rim' voxels hold part of a cap plane.
Prints worst then rms of max_c |w_c - w_c(exact)|.

``python split_error.py offsets N``: the same, pooled over N axial centre offsets
(j + 1/2)/N h, j = 0..N-1, so caps land at every distance from a voxel face rather than
always 0.043 h from one -- what a cap blend of width eps is judged on.
"""
import sys

import numpy as np

import wavebit.fdtd._shapes._bodies as bodies
from wavebit.fdtd._shapes import Cylinder, occupancy

print('R/h | law | away from the rim: worst rms | on the rim: worst rms')
ZS = ([(j + .5) / int(sys.argv[2]) for j in range(int(sys.argv[2]))]
      if sys.argv[1:2] == ['offsets'] else [.043])
for rh in (4, 8, 16):
    h = 1.0 / rh
    R, H = 1.0, 2.0
    k = np.arange(-(rh + 2), rh + 3) * h
    kz = np.arange(-(2 * rh // 2 + 2), 2 * rh // 2 + 3) * h
    out = {}
    for mode in ('plane', 'split', 'exact'):
        bodies._CYLINDER_MOMENT = mode
        ws, rims, fr = [], [], []
        for z, off in [(z, off) for z in ZS for off in
                       [(.5, 0, 0), (0, .5, 0), (0, 0, .5), (0, .5, .5), (.5, 0, .5), (.5, .5, 0)]]:
            body = Cylinder(R, H, 'z', tuple(np.array([.137, -.071, z]) * h))
            p = np.stack(np.meshgrid(k + off[0] * h, k + off[1] * h, kz + off[2] * h,
                                     indexing='ij'), -1).reshape(-1, 3)
            f, w = occupancy(body, p, h)
            caps, *_ = body._exact_tensor(p, h)
            ws.append(w)
            rims.append(caps > 0)
            fr.append((f > 1e-12) & (f < 1 - 1e-12))
        out[mode] = np.concatenate(ws)
        rim, cut = np.concatenate(rims), np.concatenate(fr)
    # Exclude voxels where the plane misses the voxel the curved body clips (plane w = 0, a
    # known corner case of _cut._LIVE_AREA; fill there is a sliver). Counted separately.
    dead = out['plane'].sum(-1) < .5
    print(f'{rh} | (cut voxels {cut.sum()}, of which rim {(cut & rim).sum()}, plane-dead {(cut & dead).sum()})')
    cut = cut & ~dead
    for mode in ('plane', 'split'):
        e = np.abs(out[mode] - out['exact']).max(-1)
        a, b = e[cut & ~rim], e[cut & rim]
        print(f'{rh} | {mode} | {a.max():.3e} {np.sqrt((a**2).mean()):.3e} | '
              f'{b.max():.3e} {np.sqrt((b**2).mean()):.3e}')
bodies._CYLINDER_MOMENT = 'plane'
