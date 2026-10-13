"""Does a ranged Cylinder have the sphere's centre-tie defect? Its axis on half-node planes in x
and y (voxel faces through the axis), caps off every plane; reverse mode against central
differences of the final electric energy in a PEC box, radius, height and centre."""
import warnings

import numpy as np

import wavebit.fd as fd
from wavebit.fd._runtime import runtime

warnings.filterwarnings('ignore')
DX = .1


def scene(v):
    body = fd.Cylinder(radius=v[0], height=v[1], axis='z', center=(v[2], v[3], v[4]),
                       range=((-.36, .36),) * 3)
    return fd.Scene(domain=fd.Domain(size=(1.6,) * 3), resolution=fd.Resolution(dx=DX),
                    boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()),
                    run=fd.Run(steps=32), structures=(fd.Structure(body, fd.Material(eps=2.25)),),
                    sources=(fd.Dipole('d', center=(.05, -.55, -.02), component='Ey',
                                       band=fd.Band(wavelength=3., df=.03), delay=-18.),))


def energy(v, backend, plan):
    return (fd.run(scene(v), backend, plan=plan).final.E[:, 5:11, 5:11, :] ** 2).sum()


for name, centre in (('off', (.013, -.021, .017)), ('axis on half-node planes', (.05, .05, .017)),
                     ('axis on node planes', (0., 0., .017))):
    point = np.array((.22, .41, *centre))
    p = fd.plan(scene(point))
    h = 3e-6
    ref = np.array([(float(energy(point + d, 'numpy', p)) - float(energy(point - d, 'numpy', p)))
                    / (2 * h) for d in np.eye(5) * h])
    import torch
    v = torch.tensor(point, dtype=torch.float64, requires_grad=True)
    gt = torch.autograd.grad(energy(v, 'torch', p), v)[0].numpy()
    runtime('jax')
    import jax
    import jax.numpy as jnp
    gj = np.asarray(jax.grad(lambda x: energy(x, 'jax', p))(jnp.asarray(point)))
    print(name, 'central', ref, 'torch rel', np.abs(gt - ref) / np.abs(ref),
          'jax rel', np.abs(gj - ref) / np.abs(ref), flush=True)
