"""(b) How far each study moment moves w (and the constitutive node tensor) from the plane's.

For cylinder scenes the tests, examples and C7 use: every voxel of every E and H Yee
staggering over the cylinder's support (plus a cell), dx as in the scene. Reports, over voxels
with a fractional fill (1e-12 < f < 1 - 1e-12), how many change, and the max and RMS of
max_c |w_c - w_c(plane)|; and the same for the node tensor <n n^T> the default discretisation's
cross term uses (E nodes; plane: n n^T of the nearest-surface normal). The fill never changes.
"""
import numpy as np

import wavebit.fdtd._shapes._bodies as bodies
from wavebit.fdtd._shapes import Cylinder, occupancy

SCENES = {
    # name: radius, height, centre, dx
    'pillar_gradient example': (.12, .3, (.011, -.007, -.2375 + .15), .05),
    'test_fd_cylinder_grad POINT': (.22, .47, (-.073, .017, -.009), .1),
    'C7 M1 @20 (offsets as C7)': (.2, .6, None, 1 / 20),
    'C7 M1 @80': (.2, .6, None, 1 / 80),
    'C7 M2 @100': (.16, .6, None, 1 / 100),
    'aligned R=3dx H=6dx, centre on node': (.3, .6, (0., 0., 0.), .1),
}


def m_centre(o, n, P):
    return tuple(-p / 2 + (round(p * n) // 2 + oo) / n for p, oo in zip(P, o))


def points(radius, height, centre, dx, offset):
    c = np.asarray(centre)
    span = np.array([radius, radius, height / 2]) + 2 * dx
    lo = np.floor((c - span) / dx) * dx
    hi = np.ceil((c + span) / dx) * dx
    axes = [np.arange(l, h + dx / 2, dx) + o * dx for l, h, o in zip(lo, hi, offset)]
    return np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)


def tensor(body, p, dx, mode):
    bodies._CYLINDER_MOMENT = mode
    d, n = body.surface(p)
    full = np.einsum('ni,nj->nij', n, n)
    st = body.study_tensor(p, dx)
    if st is not None:
        use, area, diag, off = st
        live = use & (area > 1e-12 * dx * dx)
        safe = np.where(live, area, 1.)[:, None]
        e = np.zeros_like(full)
        for a in range(3):
            e[:, a, a] = diag[:, a]
        for k, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
            e[:, i, j] = e[:, j, i] = off[:, k]
        full = np.where(live[:, None, None], e / safe[:, :, None], full)
    return full


def main():
    print('scene | mode | fractional voxels | changed (>1e-12) | max |dw| | rms |dw| over changed | '
          'node tensor: max |dP| | rms |dP| over changed')
    for name, (radius, height, centre, dx) in SCENES.items():
        if centre is None:
            n = round(1 / dx)
            if 'M1' in name:
                xc, yc = m_centre((.25, .40), n, (.85, 1.0))
            else:
                xc, yc = m_centre((.25, .25), n, (.6, .6))
            centre = (xc, yc, height / 2)
        body = Cylinder(radius, height, 'z', centre)
        staggers = [(.5, 0, 0), (0, .5, 0), (0, 0, .5), (0, .5, .5), (.5, 0, .5), (.5, .5, 0)]
        for mode in tuple(__import__('os').environ.get('STUDY_MODES', 'rim,split,exact').split(',')):
            frac = changed = 0
            dws, dps = [], []
            for off in staggers:
                p = points(radius, height, centre, dx, off)
                bodies._CYLINDER_MOMENT = 'plane'
                f0, w0 = occupancy(body, p, dx)
                bodies._CYLINDER_MOMENT = mode
                f1, w1 = occupancy(body, p, dx)
                assert np.array_equal(f0, f1)
                fr = (f0 > 1e-12) & (f0 < 1 - 1e-12)
                dw = np.abs(w1 - w0).max(-1)[fr]
                frac += fr.sum()
                changed += (dw > 1e-12).sum()
                dws.append(dw[dw > 1e-12])
            p = points(radius, height, centre, dx, (0, 0, 0))
            bodies._CYLINDER_MOMENT = 'plane'
            f0, _ = occupancy(body, p, dx)
            fr = (f0 > 1e-12) & (f0 < 1 - 1e-12)
            dP = np.abs(tensor(body, p, dx, mode) - tensor(body, p, dx, 'plane')).max((1, 2))[fr]
            dps = dP[dP > 1e-12]
            dw = np.concatenate(dws)
            print(f'{name} | {mode} | {frac} | {changed} | {dw.max(initial=0):.3e} | '
                  f'{np.sqrt((dw ** 2).mean()) if dw.size else 0:.3e} | {dps.max(initial=0):.3e} | '
                  f'{np.sqrt((dps ** 2).mean()) if dps.size else 0:.3e}', flush=True)
    bodies._CYLINDER_MOMENT = 'plane'


if __name__ == '__main__':
    main()
