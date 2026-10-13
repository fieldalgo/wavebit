"""A lone ranged sphere whose centre z lies on a node plane, in a PEC box: does reverse mode
match central differences for the centre derivative? Run against any checkout via PYTHONPATH."""
import warnings
import numpy as np
import wavebit.fd as fd
from wavebit.fd._runtime import runtime
warnings.filterwarnings('ignore')
print(fd.__file__)
DX = .1


def scene(v):
    ball = fd.Sphere(radius=v[0], center=(.013, -.021, v[1]), range=((-.36, .36),) * 3)
    return fd.Scene(domain=fd.Domain(size=(1.6,) * 3), resolution=fd.Resolution(dx=DX),
                    boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()),
                    run=fd.Run(steps=32), structures=(fd.Structure(ball, fd.Material(eps=2.25)),),
                    sources=(fd.Dipole('d', center=(.05, -.55, -.02), component='Ey',
                                       band=fd.Band(wavelength=3., df=.03), delay=-18.),))


def f(v, backend, p):
    return (fd.run(scene(v), backend, plan=p).final.E[:, 5:11, 5:11, :] ** 2).sum()


for zc in (0., .0013, .05):        # node plane (low -.8 + 8 * .1), off it, a half-node plane
    point = np.array((.22, zc))
    p = fd.plan(scene(point))
    h = 3e-6
    d = np.array((0., h))
    fdz = (float(f(point + d, 'numpy', p)) - float(f(point - d, 'numpy', p))) / (2 * h)
    import torch
    v = torch.tensor(point, dtype=torch.float64, requires_grad=True)
    gt = torch.autograd.grad(f(v, 'torch', p), v)[0][1].item()
    runtime('jax')
    import jax, jax.numpy as jnp
    gj = float(jax.grad(lambda x: f(x, 'jax', p))(jnp.asarray(point))[1])
    print(f'centre z {zc}: FD {fdz:.10g} torch {gt:.10g} jax {gj:.10g}', flush=True)
