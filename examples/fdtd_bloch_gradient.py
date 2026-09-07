"""Differentiate oblique plane power through both Bloch phase and source phase."""
import jax
import jax.numpy as jnp
import numpy as np

import wavebit


def main():
    shape=(64,2,32);spacing=.05;dt=.02;steps=700;wavelength=.8
    t=(jnp.arange(steps)+.5)*dt
    pulse=jnp.exp(-((t-3.2)/.8)**2)*jnp.exp(-2j*jnp.pi*t/wavelength)
    def objective(angle):
        kx=2*jnp.pi*jnp.sin(angle)/wavelength
        phase=jnp.stack((jnp.array(0.),jnp.array(0.),kx*shape[2]*spacing))
        source=jnp.zeros((3,)+shape,dtype=jnp.complex64)
        source=source.at[1,16].set(jnp.exp(1j*kx*jnp.arange(shape[2])*spacing)[None,:])
        out=wavebit.fdtd_grid(jnp.ones(shape),spacing=(spacing,)*3,dt=dt,steps=steps,
                             source=source,pulse=pulse,bloch_phase=phase,pml_cells=10,
                             pml_wavelength=wavelength,wavelengths=[wavelength],monitor_z=[44],
                             backend='jax',checkpoint_steps=32)
        return out['power_z'][0,0]
    value,derivative=jax.jit(jax.value_and_grad(objective))(jnp.array(.35))
    print(f'Oblique spectral plane power: {float(value):.8g}')
    print(f'Derivative per incident-angle radian: {float(derivative):.8g}')
    assert np.isfinite(float(value)) and np.isfinite(float(derivative))
    assert abs(float(derivative))>0


if __name__=='__main__':
    main()
