"""(a) The rim gradient spike, before ('plane') and after ('rim', 'exact') the study's moment.

Three levels, each against a NumPy central difference where the difference is resolvable:

1. ``voxel``: d117e9bb's single voxel centred on the rim (dx = 0.25, radius 0.5 - delta,
   height 1 - 3 delta; delta < 0 puts the point inside the corner and the circle past a
   voxel face, loss sum(w^2) + sum(fill^2)).
2. ``lattice``: every Yee staggering (8 offsets) of a dx = 0.1 lattice around a grid-aligned
   cylinder (radius 0.3 - delta, height 0.6 - 3 delta, centre on a node) -- the common design.
   Same loss, summed. Rim and voxel-face tangency both land on sample points here.
3. ``fdtd``: tests/fdtd/test_fd_cylinder_grad.py's PEC scene (24 steps, Ey dipole,
   loss sum(E^2)) with that same aligned cylinder, end to end through the default
   discretisation (diagonal mix + constitutive cross term).

Usage: python spike.py voxel|lattice|fdtd [w2|mix]

``w2`` is d117e9bb's loss, sum(w^2) + sum(fill^2). It weights w the same in a voxel the
body barely touches as in a half-full one. ``mix`` sums (1/eps_c)^2 from the default mixing
rule (eps 12 in vacuum), where w enters multiplied by the contrast term, i.e. by f(1 - f).
"""
import os
import sys
import warnings

import numpy as np

import wavebit.fdtd._shapes._bodies as bodies
from wavebit.fdtd._shapes import Cylinder, occupancy

DELTAS = (1e-3, 1e-5, 1e-7, 1e-9, 1e-11, 1e-13, 1e-15, 1e-16, 0.,
          -1e-16, -1e-15, -1e-13, -1e-11, -1e-9, -1e-7, -1e-5, -1e-3)
# STUDY_MODES=split,plane reruns a subset (the blend study reran 'split' and 'plane').
MODES = tuple(os.environ.get('STUDY_MODES', 'plane,rim,exact,split').split(','))


def jax_setup():
    import jax
    jax.config.update('jax_enable_x64', True)
    return jax


LOSS = 'w2'


def occupancy_loss(xp, radius, height, points, dx):
    fill, w = occupancy(Cylinder(radius, height, 'z', (0., 0., 0.)), xp.asarray(points), dx)
    if LOSS == 'w2':
        return (w * w).sum() + (fill * fill).sum()
    # 'mix': what the solver sees -- the default rule's 1/eps_c, eps 12 in vacuum, per component
    from wavebit.fdtd._prepare.coefficients import mix
    total = 0.
    for c in range(3):
        inverse = mix(xp, 1.0, 12.0, fill, w[..., c])
        total = total + (inverse * inverse).sum()
    return total


def grads(loss, v):
    jax = jax_setup()
    import jax.numpy as jnp
    import torch
    g_jax = np.asarray(jax.grad(lambda t: loss(jnp, t))(jnp.asarray(v)))
    t = torch.tensor(v, dtype=torch.float64, requires_grad=True)
    loss(torch, t).backward()
    return g_jax, t.grad.numpy()


def central(loss, v, step):
    return np.array([(loss(np, v + step * e) - loss(np, v - step * e)) / (2 * step)
                     for e in np.eye(len(v))])


def lattice_points(dx, n):
    k = np.arange(-n, n + 1) * dx
    pts = []
    for off in np.ndindex(2, 2, 2):
        o = np.array(off) * dx / 2
        g = np.stack(np.meshgrid(k + o[0], k + o[1], k + o[2], indexing='ij'), -1).reshape(-1, 3)
        pts.append(g)
    return np.concatenate(pts)


def run_occupancy(kind):
    if kind == 'voxel':
        dx, points, base = .25, np.array([[0.5, 0.0, 0.5]]), (0.5, 1.0)
    else:
        dx, points, base = .1, lattice_points(.1, 5), (0.3, 0.6)
    print(f'# {kind}: {len(points)} points, dx={dx}; columns: mode delta '
          'dL/dR dL/dH (JAX) |JAX-Torch|/|g| CD(step=delta/100) relerr')
    for mode in MODES:
        bodies._CYLINDER_MOMENT = mode
        for delta in DELTAS:
            v = np.array([base[0] - delta, base[1] - 3 * delta])
            loss = lambda xp, t: occupancy_loss(xp, t[0], t[1], points, dx)
            gj, gt = grads(loss, v)
            agree = np.abs(gj - gt).max() / max(np.abs(gj).max(), 1e-300)
            value = float(loss(np, v))
            line = f'{mode:5s} {delta:7.0e} L={value:.15e} {gj[0]:+.6e} {gj[1]:+.6e} {agree:.1e}'
            if abs(delta) >= 1e-9:
                cd = central(lambda xp, t: loss(np, t), v, abs(delta) / 100)
                rel = np.abs(gj - cd).max() / max(np.abs(cd).max(), 1e-300)
                line += f'  CD {cd[0]:+.6e} {cd[1]:+.6e} rel {rel:.1e}'
            print(line, flush=True)
    bodies._CYLINDER_MOMENT = 'plane'


def run_fdtd():
    import wavebit.fdtd._legacy as fd
    from wavebit.fdtd._runtime import runtime
    jax = jax_setup()
    import jax.numpy as jnp
    runtime('jax')

    def scene(v):
        return fd.Scene(
            domain=fd.Domain(size=(1.6,) * 3), resolution=fd.Resolution(dx=.1),
            boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()),
            run=fd.Run(steps=24), structures=(fd.Structure(
                fd.Cylinder(radius=v[0], height=v[1], center=(0., 0., 0.), axis='z',
                            range=((-.36, .36),) * 3),
                fd.Material(eps=2.25, mu=1.2)),), sources=(fd.Dipole(
                    'drive', center=(-.55, .03, -.02), component='Ey',
                    band=fd.Band(wavelength=3., df=.03), delay=-18.),))

    print('# fdtd: mode delta dT/dR dT/dH (JAX)  CD(step=delta/100 or 1e-6) rel')
    for mode in MODES:
        bodies._CYLINDER_MOMENT = mode
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            plan = fd.plan(scene(np.array([.3, .6])))

            def objective(v, backend):
                r = fd.run(scene(v), backend, plan=plan)
                return (r.final.E ** 2).sum()

            for delta in DELTAS:
                v = np.array([.3 - delta, .6 - 3 * delta])
                g = np.asarray(jax.grad(lambda t: objective(t, 'jax'))(jnp.asarray(v)))
                line = f'{mode:5s} {delta:7.0e} L={float(objective(v, "numpy")):.15e} {g[0]:+.6e} {g[1]:+.6e}'
                if abs(delta) >= 1e-7:
                    s = max(abs(delta) / 100, 1e-8)
                    cd = np.array([(float(objective(v + s * e, 'numpy'))
                                    - float(objective(v - s * e, 'numpy'))) / (2 * s)
                                   for e in np.eye(2)])
                    rel = np.abs(g - cd).max() / np.abs(cd).max()
                    line += f'  CD {cd[0]:+.6e} {cd[1]:+.6e} rel {rel:.1e}'
                print(line, flush=True)
    bodies._CYLINDER_MOMENT = 'plane'


if __name__ == '__main__':
    kind = sys.argv[1]
    if len(sys.argv) > 2:
        LOSS = sys.argv[2]
    print(f'# loss: {LOSS}')
    run_fdtd() if kind == 'fdtd' else run_occupancy(kind)
