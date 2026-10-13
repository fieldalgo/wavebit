"""Where do the lowered coefficients jump between height .29993 and .29994 (base on the face)?
The same second-difference locator as jump_locator.py, along the height. Run with the studies
directory as the first argument."""
import sys
import warnings

import numpy as np

sys.path.insert(0, sys.argv[1])
import fdtd_traced_junction_probe as P  # noqa: E402
import wavebit.fd as fd  # noqa: E402
from wavebit.fd._runtime import runtime  # noqa: E402
from wavebit.fd._lower.constitutive import common_points  # noqa: E402
from wavebit.fd._lower.junction import live_write  # noqa: E402
from wavebit.fd._shapes import occupancy  # noqa: E402

warnings.filterwarnings('ignore')
rt = runtime('numpy')


def lowered(height):
    v = P.POINT.copy()
    v[1] = height
    s = P.scene(v)
    p = fd.plan(s)
    return fd.lower(s, p, rt), s, p


heights = (.2999275, .2999325, .2999375, .2999425)
arrays = [lowered(h)[0] for h in heights]
keys = ['be'] + ['constitutive:' + k for k in arrays[0].get('constitutive', {})]


def get(a, key):
    return np.asarray(a['constitutive'][key.split(':', 1)[1]] if key.startswith('constitutive:')
                      else a[key])


for key in keys:
    smooth = get(arrays[1], key) - get(arrays[0], key)
    across = get(arrays[2], key) - get(arrays[1], key)
    after = get(arrays[3], key) - get(arrays[2], key)
    jump = across - (smooth + after) / 2
    at = np.unravel_index(np.argmax(abs(jump)), jump.shape)
    print(key, 'max |jump|', float(abs(jump).max()), 'at', tuple(int(i) for i in at),
          'smooth step', float(abs(smooth).max()), flush=True)
    if key.startswith('constitutive:'):
        for h in heights[1:3]:
            _, s, p = lowered(h)
            term = p.constitutive[0]
            node = np.asarray(common_points(p.grid, tuple(reversed(term.box)), 'E'))[at[1:]]
            body = s.structures[1].shape.body()
            (record,) = p.junctions
            print('  height', h, 'node', node.round(5).tolist(),
                  'pillar fill %.4f' % float(np.asarray(occupancy(body, node[None], p.dx)[0])[0]),
                  'nearest normal', np.asarray(body.surface(node[None])[1])[0].round(4).tolist(),
                  'junction writes it',
                  bool(np.asarray(live_write(rt, record, s, p.grid, node[None]))[0]), flush=True)
