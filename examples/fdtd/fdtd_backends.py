"""One FDTD objective, NumPy/JAX/Torch execution and matching derivatives."""

from wavebit import Units
from wavebit.fdtd import native
from wavebit.units import Z0
import array_api_compat as ac
import numpy as np



UNITS = Units(length='um')
TIME = UNITS.time_seconds
# Scale the SI Fourier plane power to a dimensionless optimization objective.
POWER = TIME**2 * UNITS.length_metres**2 / Z0


def objective(strength, backend):
    xp=ac.array_namespace(strength)
    shape=(48,2,2)
    mask=np.zeros(shape);mask[20:28]=1
    mask=xp.asarray(mask,dtype=strength.dtype,device=ac.device(strength))
    source=np.zeros((3,)+shape);source[0,12]=1
    dt=.05;steps=200;t=(np.arange(steps)+.5)*dt
    problem=native.Problem(units=UNITS,
        medium=native.Medium(epsilon=1+.2*mask, poles=(native.Oscillator(
            omega_0=3./TIME,gamma=.5/TIME,strength=strength*mask/TIME**2),)),
        current=native.Current(density=source/(Z0*UNITS.length_metres),
            samples=np.exp(-((t-2)/.5)**2)*np.sin(3*t)),
        outputs=native.Outputs(spectrum=native.Spectrum(wavelengths=(2.4,),planes=(32,))))
    result=native.run(problem,backend=backend,numerics=native.Numerics(
        grid=native.Grid(spacing=(.2,)*3),time_step_seconds=dt*TIME,steps=steps,
        absorber=native.PML(cells=8,wavelength=2.4)))
    return xp.sum(result.spectral_power_z)/POWER


def main():
    import jax
    import torch
    jax.config.update('jax_enable_x64', True)
    value_numpy=objective(np.array(4.),'numpy')
    value_jax,gradient_jax=jax.jit(jax.value_and_grad(lambda p:objective(p,'jax')))(jax.numpy.array(4.))
    parameter=torch.tensor(4.,dtype=torch.float64,requires_grad=True)
    value_torch=objective(parameter,'torch')
    gradient_torch=torch.autograd.grad(value_torch,parameter)[0]
    finite=(objective(np.array(4.0001),'numpy')-objective(np.array(3.9999),'numpy'))/.0002
    print('Power (NumPy/JAX/Torch):',value_numpy,float(value_jax),float(value_torch.detach()))
    print('Derivative (finite difference/JAX/Torch):',finite,float(gradient_jax),float(gradient_torch.detach()))
    np.testing.assert_allclose([float(value_jax),float(value_torch.detach())],value_numpy,rtol=1e-10)
    np.testing.assert_allclose([float(gradient_jax),float(gradient_torch.detach())],finite,rtol=2e-6)


if __name__=='__main__':
    main()
