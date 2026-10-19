"""Differentiate a transmitted spectral plane with respect to a slab's permittivity.

Uses the optional JAX backend on its configured device. This native-grid example
uses a graded conductivity absorber, distinct from the scene solver's CPML.
"""

from wavebit import Units
from wavebit.fdtd import native, ConductiveAbsorber
from wavebit.units import Z0
import numpy as np
import jax
import jax.numpy as jnp


UNITS = Units(length='um')
TIME = UNITS.time_seconds
# Scale the SI Fourier plane power to a dimensionless optimization objective.
POWER = TIME**2 * UNITS.length_metres**2 / Z0


def main():
    jax.config.update('jax_enable_x64', True)
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
        problem=native.Problem(units=UNITS,medium=native.Medium(epsilon=epsilon),
            current=native.Current(density=source.astype(jnp.complex128)/(Z0*UNITS.length_metres),samples=pulse),
            outputs=native.Outputs(spectrum=native.Spectrum(wavelengths=(.5,),planes=(48,))))
        result=native.run(problem,backend='jax',numerics=native.Numerics(
            grid=native.Grid(spacing=(.025,)*3),time_step_seconds=dt*TIME,steps=steps,
            absorber=ConductiveAbsorber(cells=10),checkpoint_steps=32))
        return result.spectral_power_z[0,0]/POWER
    value,gradient=jax.jit(jax.value_and_grad(objective))(jnp.array(0.))
    print(f'Devices: {jax.devices()}')
    print(f'Spectral plane flux: {float(value):.8g}; material gradient: {float(gradient):.8g}')
    assert np.isfinite(float(value)) and np.isfinite(float(gradient))
    assert abs(float(gradient))>0


if __name__=='__main__':
    main()
