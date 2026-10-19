"""Differentiate oblique plane power through both Bloch phase and source phase."""

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
    shape=(64,2,32);spacing=.05;dt=.02;steps=700;wavelength=.8
    t=(jnp.arange(steps)+.5)*dt
    pulse=jnp.exp(-((t-3.2)/.8)**2)*jnp.exp(-2j*jnp.pi*t/wavelength)
    def objective(angle):
        kx=2*jnp.pi*jnp.sin(angle)/wavelength
        phase=jnp.stack((jnp.array(0.),jnp.array(0.),kx*shape[2]*spacing))
        source=jnp.zeros((3,)+shape,dtype=jnp.complex64)
        source=source.at[1,16].set(jnp.exp(1j*kx*jnp.arange(shape[2])*spacing)[None,:])
        problem=native.Problem(units=UNITS,medium=native.Medium(epsilon=jnp.ones(shape)),
            current=native.Current(density=source.astype(jnp.complex128)/(Z0*UNITS.length_metres),samples=pulse),
            bloch_phase=phase,outputs=native.Outputs(spectrum=native.Spectrum(
                wavelengths=(wavelength,),z_indices=(44,))))
        out=native.run(problem,backend='jax',numerics=native.Numerics(
            grid=native.Grid(spacing=(spacing,)*3),time_step_seconds=dt*TIME,steps=steps,
            absorber=native.PML(cells=10,wavelength=wavelength),checkpoint_steps=32))
        return out.spectral_power_z[0,0]/POWER
    value,derivative=jax.jit(jax.value_and_grad(objective))(jnp.array(.35))
    print(f'Oblique spectral plane power: {float(value):.8g}')
    print(f'Derivative per incident-angle radian: {float(derivative):.8g}')
    assert np.isfinite(float(value)) and np.isfinite(float(derivative))
    assert abs(float(derivative))>0


if __name__=='__main__':
    main()
