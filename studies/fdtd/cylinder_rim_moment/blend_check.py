"""The cap blend in 'split': is w continuous, and is its derivative, across a cap tie?

1. ``switch``: the claim that lets 'split' hand over to the plane with no jump. For 200,000
   random voxels whose nearest cap plane lies beyond (sqrt 2 - 1)/2 dx of the voxel's axial
   span, the side wall's own plane (cap share 0) and the nearest-surface plane are the same
   (area, moment). Prints the worst difference.
2. ``sweep``: moves a cap through a voxel face (the tie) in steps of 1e-4 dx, for several
   radii and Yee staggerings, and prints, per mode, the worst first difference of w and the
   worst |second difference| / step -- the jump in dw/dH -- with the median for scale. Also
   radius sweeps across an aligned rim with the cap on the face.
3. ``ad``: dw/dH and dw/dR by JAX and Torch against a NumPy central difference at the exact
   tie and +/-1e-16, 1e-9, 1e-6 around it.

Usage: python blend_check.py switch|sweep|ad
"""
import sys

import numpy as np

import wavebit.fdtd._shapes._bodies as bodies
from wavebit.fdtd._shapes import Cylinder, occupancy


def switch():
    rng = np.random.default_rng(7)
    dx = .1
    worst = 0.
    n = 0
    for trial in range(40):
        R = rng.uniform(.2, .6)
        H = rng.uniform(.3, .9)
        c = rng.uniform(-.05, .05, 3)
        body = Cylinder(R, H, 'z', tuple(c))
        p = rng.uniform(-1, 1, (5000, 3)) * np.array([R + dx, R + dx, H / 2 + 2 * dx]) + c
        near, area, diag, off, (pa, pm) = body._split_tensor(p, dx)
        far = ~near
        n += far.sum()
        worst = max(worst, np.abs(area - pa)[far].max() / dx ** 2,
                    np.abs(diag - pm)[far].max() / dx ** 2)
    print(f'switch: {n} voxels beyond reach; worst |split - plane| in area or moment: '
          f'{worst:.2e} dx^2')


def w_of(R, H, points, dx):
    _, w = occupancy(Cylinder(R, H, 'z', (0., 0., 0.)), points, dx)
    return w


def sweep():
    dx = .1
    step = 1e-4 * dx
    offs = [(.5, 0, 0), (0, .5, 0), (0, 0, .5), (0, .5, .5), (.5, 0, .5), (.5, .5, 0)]
    k = np.arange(-5, 6) * dx
    pts = np.concatenate([np.stack(np.meshgrid(k + o[0] * dx, k + o[1] * dx, k + o[2] * dx,
                                               indexing='ij'), -1).reshape(-1, 3) for o in offs])
    for mode in ('plane', 'split'):
        bodies._CYLINDER_MOMENT = mode
        for label, var in (('height through a face', 'H'), ('radius through a face', 'R')):
            worst1 = worst2 = 0.
            meds = []
            for R0 in (.3, .3137, .35):          # .3: aligned rim; others: the cap tie alone
                for tie in (.3, .35):            # cap on a z face (.3) and mid-way (.35)
                    s = np.arange(-60, 61) * step
                    if var == 'H':
                        ws = np.stack([w_of(R0, 2 * (tie + x), pts, dx) for x in s])
                    else:
                        ws = np.stack([w_of(.3 + x, 2 * tie, pts, dx) for x in s])
                    d1 = np.abs(np.diff(ws, axis=0)).max(axis=(1, 2))
                    d2 = np.abs(np.diff(ws, 2, axis=0)).max(axis=(1, 2)) / step
                    worst1, worst2 = max(worst1, d1.max()), max(worst2, d2.max())
                    meds.append(np.median(d1))
                    if var == 'R':
                        break
            print(f'{mode:5s} {label}: worst |dw| per {step / dx:.0e} dx step {worst1:.2e} '
                  f'(median {np.median(meds):.2e}); worst jump in dw/dvar (2nd diff / step) '
                  f'{worst2:.2e}', flush=True)
    bodies._CYLINDER_MOMENT = 'plane'


def ad():
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    import torch
    dx = .1
    offs = [(.5, 0, 0), (0, .5, 0), (0, 0, .5), (0, .5, .5), (.5, 0, .5), (.5, .5, 0)]
    k = np.arange(-5, 6) * dx
    pts = np.concatenate([np.stack(np.meshgrid(k + o[0] * dx, k + o[1] * dx, k + o[2] * dx,
                                               indexing='ij'), -1).reshape(-1, 3) for o in offs])

    def loss(xp, v):
        fill, w = occupancy(Cylinder(v[0], v[1], 'z', (0., 0., 0.)), xp.asarray(pts), dx)
        weight = fill * (1 - fill)
        return (weight[..., None] * w * w).sum()

    for mode in ('plane', 'split'):
        bodies._CYLINDER_MOMENT = mode
        for delta in (1e-6, 1e-9, 1e-16, 0., -1e-16, -1e-9, -1e-6):
            v = np.array([.3, .6 - 2 * delta])
            gj = np.asarray(jax.grad(lambda t: loss(jnp, t))(jnp.asarray(v)))
            t = torch.tensor(v, requires_grad=True)
            loss(torch, t).backward()
            gt = t.grad.numpy()
            line = (f'{mode:5s} H tie {delta:+.0e}: JAX {gj[0]:+.6e} {gj[1]:+.6e} '
                    f'|JAX-Torch| {np.abs(gj - gt).max():.1e}')
            if abs(delta) >= 1e-6 or delta == 0:
                h = 1e-8
                cd = np.array([(loss(np, v + h * e) - loss(np, v - h * e)) / (2 * h)
                               for e in np.eye(2)])
                line += f'  CD {cd[0]:+.6e} {cd[1]:+.6e}'
            print(line, flush=True)
    bodies._CYLINDER_MOMENT = 'plane'


if __name__ == '__main__':
    {'switch': switch, 'sweep': sweep, 'ad': ad}[sys.argv[1]]()
