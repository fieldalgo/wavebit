"""Where do the lowered coefficients jump between radius .12006 and .12007? Second differences of
every coefficient array over three radii a quarter of that interval apart on either side, so a
smooth change cancels and a jump stands out. Run with the studies directory as an argument."""
import sys
import warnings

import numpy as np

sys.path.insert(0, sys.argv[1])
import fdtd_traced_junction_probe as P  # noqa: E402
import wavebit.fd as fd  # noqa: E402
from wavebit.fd._runtime import runtime  # noqa: E402

warnings.filterwarnings('ignore')
rt = runtime('numpy')


def lowered(radius, raw=False):
    v = P.POINT.copy()
    v[0] = radius
    s = P.scene(v)
    p = fd.plan(s)
    if raw:
        from wavebit.fd._lower.coefficients import build
        return build(s, p, rt), p
    return fd.lower(s, p, rt), p


for raw in (True, False):
    radii = (.1200575, .1200625, .1200675, .1200725)
    arrays = [lowered(r, raw)[0] for r in radii]
    keys = [k for k in ('be',) if k in arrays[0]]
    if not raw:
        keys += ['constitutive:' + k for k in arrays[0]['constitutive']]

    def get(a, key):
        return np.asarray(a['constitutive'][key.split(':', 1)[1]] if key.startswith('constitutive:')
                          else a[key])
    for key in keys:
        smooth = get(arrays[1], key) - get(arrays[0], key)       # both on one side
        across = get(arrays[2], key) - get(arrays[1], key)       # across .1200645?
        after = get(arrays[3], key) - get(arrays[2], key)
        jump = across - (smooth + after) / 2
        at = np.unravel_index(np.argmax(abs(jump)), jump.shape)
        print('raw' if raw else 'lifted', key, 'max |jump|', float(abs(jump).max()),
              'at', at, 'smooth step', float(abs(smooth).max()), flush=True)

# The node behind the constitutive jump: its coordinates, fills, nearest-face normal on both
# sides, and whether the junction writes it.
from wavebit.fd._lower.constitutive import common_points  # noqa: E402
from wavebit.fd._lower.junction import live_write  # noqa: E402
from wavebit.fd._shapes import occupancy  # noqa: E402
for radius in (.1200625, .1200675):
    v = P.POINT.copy()
    v[0] = radius
    s = P.scene(v)
    p = fd.plan(s)
    term = p.constitutive[0]
    box = tuple(reversed(term.box))
    pts = np.asarray(common_points(p.grid, box, 'E'))
    node = pts[8, 7, 5]
    body = s.structures[1].shape.body()
    fill = float(np.asarray(occupancy(body, node[None], p.dx)[0])[0])
    alpha = float(np.asarray(occupancy(s.structures[0].shape.body(), node[None], p.dx)[0])[0])
    normal = np.asarray(body.surface(node[None])[1])[0]
    (record,) = p.junctions
    written = bool(np.asarray(live_write(rt, record, s, p.grid, node[None]))[0])
    print('radius', radius, 'node', node.round(5).tolist(), 'pillar fill %.4f' % fill,
          'substrate fill %.4f' % alpha, 'nearest normal', normal.round(4).tolist(),
          'junction writes it', written, flush=True)
