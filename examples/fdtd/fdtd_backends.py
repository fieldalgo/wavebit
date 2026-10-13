"""One FDTD objective, NumPy/JAX/Torch execution and matching derivatives."""

from wavebit.compat import legacy as _legacy_wavebit
import array_api_compat as ac
import numpy as np



def objective(strength, backend):
    xp=ac.array_namespace(strength)
    shape=(48,2,2)
    mask=np.zeros(shape);mask[20:28]=1
    mask=xp.asarray(mask,dtype=strength.dtype,device=ac.device(strength))
    source=np.zeros((3,)+shape);source[0,12]=1
    dt=.05;steps=200;t=(np.arange(steps)+.5)*dt
    result=_legacy_wavebit.fdtd_grid(1+.2*mask,spacing=(.2,)*3,dt=dt,steps=steps,
                            source=source,pulse=np.exp(-((t-2)/.5)**2)*np.sin(3*t),
                            poles=[(3.,.5,strength*mask)],pml_cells=8,pml_wavelength=2.4,
                            wavelengths=[2.4],monitor_z=[32],backend=backend)
    return xp.sum(result['power_z'])


def main():
    import jax
    import torch
    _legacy_wavebit.enable_x64()
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
