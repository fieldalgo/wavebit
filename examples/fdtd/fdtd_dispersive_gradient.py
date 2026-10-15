"""Differentiate Lorentz oscillator strength with CPML at both vacuum z ends."""

from wavebit import Units
from wavebit.fdtd import native
from wavebit.units import Z0
import jax
import jax.numpy as jnp
import numpy as np



UNITS = Units(length='um')
TIME = UNITS.time_seconds
# Scale the SI Fourier plane power to a dimensionless optimization objective.
POWER = TIME**2 * UNITS.length_metres**2 / Z0


def main():
    jax.config.update('jax_enable_x64', True)
    shape=(80,2,2)
    mask=np.zeros(shape);mask[32:40]=1
    source=np.zeros((3,)+shape);source[0,16]=1
    dt=.02;steps=800;t=(np.arange(steps)+.5)*dt
    pulse=np.exp(-((t-3.2)/.8)**2)*np.sin(2*np.pi*t/.8)
    def objective(parameter):
        strength=9*jax.nn.sigmoid(parameter)*mask
        problem=native.Problem(units=UNITS,medium=native.Medium(epsilon=1+.5*mask,
            poles=(native.Oscillator(omega_0=10./TIME,gamma=1./TIME,strength=strength/TIME**2),)),
            current=native.Current(density=source.astype(jnp.complex128)/(Z0*UNITS.length_metres),samples=pulse),
            outputs=native.Outputs(spectrum=native.Spectrum(wavelengths=(.8,),planes=(56,))))
        result=native.run(problem,backend='jax',numerics=native.Numerics(
            grid=native.Grid(spacing=(.05,)*3),time_step_seconds=dt*TIME,steps=steps,
            absorber=native.PML(cells=12,wavelength=.8),checkpoint_steps=32))
        return result.spectral_power_z[0,0]/POWER
    value,gradient=jax.jit(jax.value_and_grad(objective))(jnp.array(0.))
    print(f'CPML Lorentz spectral power: {float(value):.8g}')
    print(f'Oscillator-strength parameter derivative: {float(gradient):.8g}')
    assert np.isfinite(float(value)) and np.isfinite(float(gradient))
    assert abs(float(gradient))>0


if __name__=='__main__':
    main()
