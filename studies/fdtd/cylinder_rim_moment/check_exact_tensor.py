"""Check the study's exact cylinder moment against brute-force quadrature of the surface.

Random voxels near a cylinder's side wall and cap rim; the reference integrates n n^T over
the true surface (side wall: 2000001 arc samples x axial overlap; caps: the clipped disc area)
independently of the divergence-theorem form. Prints the worst absolute error in the
moments, relative to dx^2.
"""
import numpy as np

import wavebit.fdtd._shapes._bodies as bodies
from wavebit.fdtd._shapes import Cylinder
from wavebit.fdtd._shapes._exact import clipped_disc_area, overlap, plane_share


def reference(cyl, p, dx):
    R = cyl.radius
    cx, cy, cz = cyl.center
    th = np.linspace(0, 2 * np.pi, 2000001)[:-1]
    ds = 2 * np.pi * R / th.size
    x, y = cx + R * np.cos(th), cy + R * np.sin(th)
    inb = (x > p[0] - dx / 2) & (x < p[0] + dx / 2) & (y > p[1] - dx / 2) & (y < p[1] + dx / 2)
    along = float(overlap(np.array(cz - cyl.height / 2), np.array(cz + cyl.height / 2),
                          np.array(p[2]), dx, np))
    c, s = np.cos(th[inb]), np.sin(th[inb])
    muu, mvv, muv = (c * c).sum() * ds, (s * s).sum() * ds, (c * s).sum() * ds
    a2 = float(clipped_disc_area(cx, cy, R, p[0] - dx / 2, p[0] + dx / 2,
                                 p[1] - dx / 2, p[1] + dx / 2, np))
    caps = sum(float(plane_share(np.array(cz + h), np.array(p[2]), dx, np))
               for h in (-cyl.height / 2, cyl.height / 2))
    diag = np.array([muu * along, mvv * along, caps * a2])
    off = np.array([muv * along, 0., 0.])
    return diag.sum(), diag, off


rng = np.random.default_rng(7)
worst = 0.
for trial in range(400):
    dx = 0.1
    R = rng.uniform(0.12, 0.6)
    h = rng.uniform(0.2, 0.8)
    c = rng.uniform(-0.05, 0.05, 3)
    cyl = Cylinder(R, h, 'z', tuple(c))
    t = rng.uniform(0, 2 * np.pi)
    p = c + np.array([(R + rng.uniform(-.08, .08)) * np.cos(t), (R + rng.uniform(-.08, .08)) * np.sin(t),
                      rng.choice([-1, 1]) * h / 2 + rng.uniform(-.08, .08)])
    _, area, diag, off = cyl._exact_tensor(p[None], dx)
    ra, rd, ro = reference(cyl, p, dx)
    err = max(abs(area[0] - ra), np.abs(diag[0] - rd).max(), np.abs(off[0] - ro).max()) / dx ** 2
    worst = max(worst, err)
print(f'worst |exact - quadrature| / dx^2 over 400 voxels: {worst:.3e}')
