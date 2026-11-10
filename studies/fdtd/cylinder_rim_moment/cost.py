"""(d) Cost of each study moment: geometry setup alone, and one forward + gradient end to end.

1. ``occupancy``: ``component_occupancy`` for all six E/H components of a cylinder of radius
   12 cells and height 24 cells over its support box (dx = 0.025), NumPy / JAX / Torch, best
   of 5 after one warm-up; peak RSS growth read from /proc (ru_maxrss of a child process).
2. ``fdtd``: tests/fdtd/test_fd_cylinder_grad.py's PEC scene (24 steps) with an off-grid
   cylinder: JAX value_and_grad over (radius, height), best of 3 after a warm-up (compile).

Usage: python cost.py occupancy|fdtd MODE   (one mode per process, so peak RSS is its own)
"""
import resource
import sys
import time
import warnings

import numpy as np

import wavebit.fdtd._shapes._bodies as bodies


def best(fn, n):
    fn()
    times = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return min(times)


def occupancy(mode):
    from wavebit.fdtd._kernel.grid import Grid
    from wavebit.fdtd._prepare.occupancy import component_occupancy
    from wavebit.fdtd._runtime import runtime
    from wavebit.fdtd._shapes import Cylinder
    bodies._CYLINDER_MOMENT = mode
    dx = .025
    grid = Grid.from_domain((1.0,) * 3, (0.,) * 3, dx)
    body = Cylinder(.3, .6, 'z', (.0031, -.0047, .0013))
    box = ((7, 33), (7, 33), (7, 33))
    out = []
    for backend in ('numpy', 'jax', 'torch'):
        rt = runtime(backend)

        def go():
            res = [component_occupancy(body, grid, box, f, c, rt) for f in 'EH' for c in range(3)]
            [rt.numpy(a) for pair in res for a in pair]
        out.append(f'{backend} {best(go, 5):.4f}s')
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f'occupancy {mode:5s} ' + '  '.join(out) + f'  peak RSS {rss:.0f} MB', flush=True)


def fdtd(mode):
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    import wavebit.fdtd._legacy as fd
    bodies._CYLINDER_MOMENT = mode

    def scene(v):
        return fd.Scene(
            domain=fd.Domain(size=(1.6,) * 3), resolution=fd.Resolution(dx=.05),
            boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()),
            run=fd.Run(steps=48), structures=(fd.Structure(
                fd.Cylinder(radius=v[0], height=v[1], center=(.013, -.007, .004), axis='z',
                            range=((-.4, .4),) * 3),
                fd.Material(eps=2.25)),), sources=(fd.Dipole(
                    'drive', center=(-.55, .03, -.02), component='Ey',
                    band=fd.Band(wavelength=3., df=.03), delay=-18.),))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        t = time.perf_counter()
        plan = fd.plan(scene(np.array([.27, .51])))
        t_plan = time.perf_counter() - t
        f = jax.value_and_grad(lambda v: (fd.run(scene(v), 'jax', plan=plan).final.E ** 2).sum())
        v = jnp.asarray([.27, .51])
        t = time.perf_counter()
        jax.block_until_ready(f(v))
        first = time.perf_counter() - t
        t_run = best(lambda: jax.block_until_ready(f(v)), 3)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f'fdtd {mode:5s} plan {t_plan:.3f}s  first value_and_grad {first:.2f}s  '
          f'repeat {t_run:.3f}s  peak RSS {rss:.0f} MB  grad {np.asarray(f(v)[1])}', flush=True)


if __name__ == '__main__':
    {'occupancy': occupancy, 'fdtd': fdtd}[sys.argv[1]](sys.argv[2])
