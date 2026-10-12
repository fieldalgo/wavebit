"""Compare two-pole ADE Yee updates and oscillator-strength gradients with FDTDX.

FDTDX is executed as an external package, not copied into the Wavebit solver.
Uses its low-level update_E/update_H functions, not a geometry or source test.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)
SHAPE=(9, 7, 5)
STEPS=53
DT=.2


def worker(backend):
    rng=np.random.default_rng(2419)
    e=jnp.asarray(rng.normal(size=(3,)+SHAPE)*.1)
    h=jnp.asarray(rng.normal(size=(3,)+SHAPE)*.1)
    pattern=jnp.asarray(rng.uniform(.1, .9, size=(3,)+SHAPE))
    mu=jnp.asarray(rng.uniform(1., 1.7, size=(3,)+SHAPE))
    def swap(value):
        return jnp.transpose(value,(0,3,2,1))
    if backend == 'fdtdx':
        import fdtdx
        import fdtdx.fdtd.update as update
        from fdtdx.fdtd.container import ArrayContainer, FieldState, ObjectContainer
        config=fdtdx.SimulationConfig(time=1e-7,grid=fdtdx.UniformGrid(spacing=1.),
                                      backend='cpu',dtype=jnp.float64,courant_factor=DT*np.sqrt(3))
        volume=fdtdx.SimulationVolume(partial_grid_shape=SHAPE[::-1]).place_on_grid(
            tuple((0,n) for n in SHAPE[::-1]),config,jax.random.PRNGKey(0))
        boundaries=[fdtdx.BlochBoundary(axis=a,direction=d) for a in range(3) for d in ('-','+')]
        objects=ObjectContainer(object_list=[volume]+boundaries,volume_idx=0)
        from fdtdx.constants import c
        from fdtdx.dispersion import compute_pole_coefficients
        external_poles=(fdtdx.DrudePole(plasma_frequency=np.sqrt(.7)*c,damping=.3*c),
                        fdtdx.LorentzPole(resonance_frequency=1.2*c,damping=.2*c,delta_epsilon=.8/1.2**2))
        c1,c2,c3=compute_pole_coefficients(external_poles,DT/c)
        c1,c2,c3=[jnp.asarray(v)[:,None,None,None,None] for v in (c1,c2,c3)]
        def run(parameter):
            p=jnp.zeros((2,3)+SHAPE[::-1])
            arrays=ArrayContainer(fields=FieldState(E=swap(e),H=swap(h),psi_E={},psi_H={},
                                                   dispersive_P_curr=p,dispersive_P_prev=p),
                                  inv_permittivities=swap(1/(2+.2*pattern)),
                                  inv_permeabilities=swap(1/mu),detector_states={},recording_state=None,
                                  dispersive_c1=c1,dispersive_c2=c2,
                                  dispersive_c3=c3*swap(1+parameter*pattern)[None])
            def step(arr,n):
                arr=update.update_H(n,arr,objects,config,True)
                return update.update_E(n,arr,objects,config,True),None
            arrays=jax.lax.scan(step,arrays,jnp.arange(STEPS))[0]
            return swap(arrays.fields.E),swap(arrays.fields.H)
        metadata=dict(version=importlib.metadata.version('fdtdx'),
                      update_sha256=hashlib.sha256(Path(update.__file__).read_bytes()).hexdigest())
    else:
        import wavebit
        def run(parameter):
            out=wavebit.fdtd_grid(2+.2*pattern,mu=mu,spacing=(1.,)*3,dt=DT,
                                 poles=((0.,.3,.7*(1+parameter*pattern)),
                                        (1.2,.2,.8*(1+parameter*pattern))),
                                 steps=STEPS,electric=e,magnetic=h,backend='jax',checkpoint_steps=8)
            return out['E'],out['H']
        metadata=dict(version=wavebit.__version__)
    def objective(parameter):
        ef,hf=run(parameter)
        # An asymmetric local field objective detects axis/sign/placement errors.
        return jnp.sum(pattern*jnp.abs(ef)**2)+.3*jnp.sum(jnp.abs(hf[:,2:5,1:4,1:3])**2)
    fields=jax.jit(run)(.2)
    value,gradient=jax.jit(jax.value_and_grad(objective))(.2)
    delta=1e-5
    finite=(objective(.2+delta)-objective(.2-delta))/(2*delta)
    return dict(backend=backend,**metadata,shape=SHAPE,steps=STEPS,dt=DT,
                E=np.asarray(fields[0]).real.tolist(),H=np.asarray(fields[1]).real.tolist(),
                value=float(value),gradient=float(gradient),finite_difference=float(finite))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=['fdtdx','wavebit'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.write_text(json.dumps(worker(args.backend),indent=2)+'\n')


if __name__ == '__main__':
    main()
