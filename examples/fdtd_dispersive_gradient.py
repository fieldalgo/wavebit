"""Differentiate Lorentz oscillator strength with CPML at both vacuum z ends."""
import jax
import jax.numpy as jnp
import numpy as np

import wavebit


def main():
    shape=(80,2,2)
    mask=np.zeros(shape);mask[32:40]=1
    source=np.zeros((3,)+shape);source[0,16]=1
    dt=.02;steps=800;t=(np.arange(steps)+.5)*dt
    pulse=np.exp(-((t-3.2)/.8)**2)*np.sin(2*np.pi*t/.8)
    def objective(parameter):
        strength=9*jax.nn.sigmoid(parameter)*mask
        result=wavebit.fdtd_grid(1+.5*mask,spacing=(.05,)*3,dt=dt,steps=steps,
                                source=source,pulse=pulse,poles=[(10.,1.,strength)],
                                pml_cells=12,pml_wavelength=.8,backend='jax',
                                wavelengths=[.8],monitor_z=[56],checkpoint_steps=32)
        return result['power_z'][0,0]
    value,gradient=jax.jit(jax.value_and_grad(objective))(jnp.array(0.))
    print(f'CPML Lorentz spectral power: {float(value):.8g}')
    print(f'Oscillator-strength parameter derivative: {float(gradient):.8g}')
    assert np.isfinite(float(value)) and np.isfinite(float(gradient))
    assert abs(float(gradient))>0


if __name__=='__main__':
    main()
