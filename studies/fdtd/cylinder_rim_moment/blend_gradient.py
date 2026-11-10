"""What the cap blend's width eps costs in gradient: dL/dH while a cap crosses the ramp.

spike.py's end-to-end scene (PEC box, 24 steps, sum E^2, JAX) with the aligned cylinder
(R = 0.3, dx = 0.1) and the height raised so that both caps sit a depth t inside their voxels,
t from 0 to 1.5 eps in 13 steps (t = 0 is the tie). The blend is steepest at t = eps/2.
Prints, for the eps the process was started with, dL/dR and dL/dH by JAX and the central
difference (step 1e-4 eps dx), and the largest |dL/dH| over the sweep. Away from any ramp,
|dL/dH| is about 5.7 (records/blend_spike_fdtd.txt).

Usage: WAVEBIT_STUDY_CYLINDER_MOMENT=split WAVEBIT_STUDY_CYLINDER_CAP_BLEND=eps python blend_gradient.py
"""
import warnings

import numpy as np

import wavebit.fdtd._legacy as fd
import wavebit.fdtd._shapes._bodies as bodies
from wavebit.fdtd._runtime import runtime

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp  # noqa: E402

runtime('jax')
dx, eps = .1, bodies._CAP_BLEND


def scene(v):
    return fd.Scene(
        domain=fd.Domain(size=(1.6,) * 3), resolution=fd.Resolution(dx=dx),
        boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()),
        run=fd.Run(steps=24), structures=(fd.Structure(
            fd.Cylinder(radius=v[0], height=v[1], center=(0., 0., 0.), axis='z',
                        range=((-.36, .36),) * 3),
            fd.Material(eps=2.25, mu=1.2)),), sources=(fd.Dipole(
                'drive', center=(-.55, .03, -.02), component='Ey',
                band=fd.Band(wavelength=3., df=.03), delay=-18.),))


print(f'# mode {bodies._CYLINDER_MOMENT} ramp {bodies._CAP_RAMP} eps {eps} dx; '
      'columns: t/(eps dx)  dL/dR dL/dH (JAX)  CD  rel')
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    plan = fd.plan(scene(np.array([.3, .6])))

    def objective(v, backend):
        return (fd.run(scene(v), backend, plan=plan).final.E ** 2).sum()

    worst = 0.
    for frac in np.linspace(0, 1.5, 13):
        # Caps at +/-(0.3 - t) are each a depth t inside a voxel whose face is at 0.3 (the
        # H-component voxels); other staggerings see them elsewhere in their own cells.
        v = np.array([.3137, .6 - 2 * frac * eps * dx])
        g = np.asarray(jax.grad(lambda t: objective(t, 'jax'))(jnp.asarray(v)))
        s = 1e-4 * eps * dx
        cd = np.array([(float(objective(v + s * e, 'numpy')) - float(objective(v - s * e, 'numpy')))
                       / (2 * s) for e in np.eye(2)])
        rel = np.abs(g - cd).max() / np.abs(cd).max()
        worst = max(worst, abs(g[1]))
        print(f'{frac:.3f} {g[0]:+.6e} {g[1]:+.6e}  CD {cd[0]:+.6e} {cd[1]:+.6e} rel {rel:.1e}',
              flush=True)
    print(f'max |dL/dH| over the ramp: {worst:.3e}')
