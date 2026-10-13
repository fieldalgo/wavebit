"""The first fill points (.3 and .4 cell offsets) put a voxel face at -.22, the radius. Central
differences of the weighted fill there converge to reverse mode only linearly in the step,
from both one-sided slopes: a jump of the second derivative, not a wrong first derivative.
Run with the studies directory as the first argument."""
import sys
import warnings

import numpy as np

sys.path.insert(0, sys.argv[1])
import fdtd_sphere_center_tie_probe as P  # noqa: E402

warnings.filterwarnings('ignore')


def first_points(axis=2):
    base = np.array(P.OFF)
    offsets = np.arange(-2, 3) * P.DX
    pts = []
    for s in (-.5, .5):
        for u in offsets:
            for w in offsets:
                p = base.copy()
                others = [j for j in range(3) if j != axis]
                p[axis] += s * P.DX
                p[others[0]] += u + .3 * P.DX
                p[others[1]] += w + .4 * P.DX
                pts.append(p)
    return np.array(pts)


points = first_points()
weights = np.linspace(1., 2., len(points))
leaves = np.array((P.RADIUS, *P.OFF))


def objective(v):
    return float((np.asarray(P.fills('numpy', v[0], (v[1], v[2], v[3]), points))
                  * weights).sum())


import torch  # noqa: E402
v = torch.tensor(leaves, dtype=torch.float64, requires_grad=True)
g = torch.autograd.grad((P.fills('torch', v[0], (v[1], v[2], v[3]), points)
                         * torch.tensor(weights)).sum(), v)[0].numpy()
for k, name in ((0, 'radius'), (1, 'centre x')):
    for h in (1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
        d = np.zeros(4)
        d[k] = h
        print(f'{name} h {h:g}: central {(objective(leaves + d) - objective(leaves - d)) / (2 * h)!r}'
              f' reverse {g[k]!r} forward-sided {(objective(leaves + d) - objective(leaves)) / h!r}'
              f' backward-sided {(objective(leaves) - objective(leaves - d)) / h!r}', flush=True)
