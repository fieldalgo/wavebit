"""Differentiate a pillar on an explicitly continued, smoothed substrate."""
import argparse
import numpy as np
from wavebit import Material, Units, fdtd
from wavebit.units import C0

PLANE = -.2375


def problem(radius=.12, height=.3, epsilon=2.25):
    return fdtd.Problem(units=Units(length='um'), scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.6,.6,1.3), center=(0.,0.,.05)),
        boundaries=fdtd.Boundaries.periodic(z=fdtd.Open()), structures=(
            fdtd.Structure('substrate', shape=fdtd.Box(size=(10.,10.,2.), center=(0.,0.,PLANE-1.)),
                           material=Material(2.1), continuation=('z-',)),
            fdtd.Structure('pillar', shape=fdtd.Cylinder(radius=radius,height=height,
                center=(.011,-.007,PLANE+height/2)), material=Material(epsilon)))),
        sources=(fdtd.PlaneWave('pw',plane=fdtd.Plane(z=.45),
            pulse=fdtd.GaussianPulse(wavelength=1.,frequency_sigma_hz=.15*C0/1e-6)),),
        outputs=fdtd.Outputs(monitors=(fdtd.DiffractionMonitor('T',plane=fdtd.Plane(z=-.55),
            source='pw',wave='transmitted',wavelengths=(1.,),orders=0),)))


def main(backend='torch'):
    numerics=fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.05),duration=fdtd.Duration(steps=600),
        absorber=fdtd.PML(cells=8,wavelength=1.,allow_thin=True,
                         reference_media={'z-':Material(2.1)}),
        geometry_ranges={'pillar':((-.145,.145),(-.145,.145),(PLANE-.001,PLANE+.401))})
    if backend=='jax':
        import jax
        import jax.numpy as jnp
        jax.config.update('jax_enable_x64',True)
        values=jnp.asarray([.12,.3,2.25])
    else:
        import torch
        values=torch.tensor([.12,.3,2.25],dtype=torch.float64,requires_grad=True)
    prepared=fdtd.prepare(problem(),numerics=numerics,backend=backend)
    def transmission(v): return prepared.run(problem(*v))['T'].efficiency[0,0,0]
    if backend=='jax':
        value,gradient=jax.value_and_grad(transmission)(values)
    else:
        value=transmission(values)
        gradient,=torch.autograd.grad(value,values)
        value,gradient=value.detach(),gradient.detach()
    assert np.isfinite(np.asarray(gradient)).all()
    assert 0<float(value)<1
    # This reproduces the original scene's derivatives; it is not a claim of
    # optical convergence. Refine spacing, duration and absorber independently.
    np.testing.assert_allclose(np.asarray(gradient),(.1933,-.0818,.00625),atol=5e-5,rtol=0)
    print(f'T00={float(value):.9f}; dT/d(radius,height,epsilon)={np.asarray(gradient)}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=('torch','jax'),default='torch')
    main(parser.parse_args().backend)
