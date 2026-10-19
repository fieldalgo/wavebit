"""Measure physical near fields and spectral plane power without a time history."""
from wavebit import Units
from wavebit.fdtd import periodic as fdtd
import numpy as np


def main():
    units=Units(length='um')
    problem=fdtd.Problem(units=units,
        stack=fdtd.Stack(period=(.2,),layers=(fdtd.Layer(thickness=.1,epsilon=np.ones((1,2))),)),
        excitation=fdtd.Excitation(wavelength=(.85,.9,.95),sources=(fdtd.PlaneWave(),)),
        outputs=fdtd.Outputs(planes={'reflection':.75,'transmission':-.2}))
    numerics=fdtd.Numerics(spacing=1/30,duration_seconds=90*units.time_seconds,algorithm='reference')
    reference=fdtd.run(problem,numerics=numerics)
    film=problem.stack.layers[0].replace(epsilon=np.full((1,2),2.25))
    result=fdtd.run(problem.replace(stack=problem.stack.replace(layers=(film,))),numerics=numerics)
    # +z points up. An invariant-y calculation reports W/m of transverse width.
    incident=reference.plane_power_z['transmission']
    reflected=-result.plane_power_z['reflection']/incident
    transmitted=result.plane_power_z['transmission']/incident
    print('Plane reflection:',reflected)
    print('Plane transmission:',transmitted)
    print('Near-field Ey shape:',result.planes['transmission'].values[...,1].shape)
    np.testing.assert_allclose(reflected,result.R.sum(axis=(-2,-1)),atol=.01)
    np.testing.assert_allclose(transmitted,result.T.sum(axis=(-2,-1)),atol=.01)
    assert not result.snapshots


if __name__=='__main__':
    main()
