"""Which Yee points switch their nearest cylinder face (wall vs cap) between the sweep values
where T jumps? A point at transverse distance rho from the axis and height z switches when
rho - r equals its distance past the nearer cap. Only points whose voxel the pillar cuts
(fill strictly between the clip round-off and one) are counted, and junction-written points
are reported separately. Run with the studies directory as the first argument."""
import sys
import warnings

import numpy as np

sys.path.insert(0, sys.argv[1])
import fdtd_traced_junction_probe as P  # noqa: E402
import wavebit.fd as fd  # noqa: E402
from wavebit.fd._lower.occupancy import yee_points  # noqa: E402
from wavebit.fd._shapes import occupancy  # noqa: E402

warnings.filterwarnings('ignore')
s = P.scene()
p = fd.plan(s)
shape = s.structures[1].shape
box = shape.support(p.grid)
r, h = P.POINT[0], P.POINT[1]
cx, cy = P.POINT[2], -.007
for field in ('E', 'H'):
    for c in range(3):
        pts = np.asarray(yee_points(p.grid, box, field, c)).reshape(-1, 3)
        fill = np.asarray(occupancy(shape.body(), pts, p.dx)[0])
        live = (fill > 1e-9) & (fill < 1 - 1e-9)
        rho = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        z = pts[:, 2]
        # radius sweep (height and base fixed): cap distances to base and top
        top, base = P.PLANE + h, P.PLANE
        cap = np.maximum(base - z, z - top)
        r_star = rho - cap
        hit_r = live & (r_star > .12006) & (r_star < .12007)
        # height sweep (base fixed): the top moves with h
        h_star = z - P.PLANE - (rho - r)
        hit_h = live & (h_star > .29993) & (h_star < .29994)
        for name, hit, star in (('radius', hit_r, r_star), ('height', hit_h, h_star)):
            for i in np.flatnonzero(hit):
                print(field + 'xyz'[c], name, 'switch at', repr(float(star[i])), 'point',
                      pts[i].round(5).tolist(), 'fill %.4f' % fill[i], flush=True)
