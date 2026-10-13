"""Differentiate a transmitted spectral plane with respect to a slab's permittivity.

Uses the optional JAX backend on its configured device. This native-grid example
uses a graded conductivity absorber, distinct from the scene solver's CPML.
"""

from wavebit.compat import legacy as _legacy_wavebit
import numpy as np
import jax
import jax.numpy as jnp


def main():
    shape=(64,2,2)
    dt=.01
    steps=800
    source=np.zeros((3,)+shape)
    source[1,14]=1.
    t=(np.arange(steps)+1)*dt
    pulse=np.exp(-((t-2.)/.5)**2)*np.sin(2*np.pi*t/.5)
    mask=(jnp.arange(64)[:,None,None]>=26)&(jnp.arange(64)[:,None,None]<34)
    def objective(parameter):
        # Bounded material stays above vacuum throughout differentiation.
        epsilon=jnp.broadcast_to(1.+3.*jax.nn.sigmoid(parameter)*mask,shape)
        result=_legacy_wavebit.fdtd_grid(epsilon,spacing=(.025,.025,.025),dt=dt,steps=steps,
                                source=source,pulse=pulse,wavelengths=[.5],monitor_z=[48],
                                absorber_cells=10,backend='jax',checkpoint_steps=32)
        return result['power_z'][0,0]
    value,gradient=jax.jit(jax.value_and_grad(objective))(jnp.array(0.))
    print(f'Devices: {jax.devices()}')
    print(f'Spectral plane flux: {float(value):.8g}; material gradient: {float(gradient):.8g}')
    assert np.isfinite(float(value)) and np.isfinite(float(gradient))
    assert abs(float(gradient))>0


if __name__=='__main__':
    main()
