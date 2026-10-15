"""Two physical sources: boundary phase controls output and absorption."""
import numpy as np
from wavebit import Material, Units, rcwa


def problem(sources, region=None):
    return rcwa.Problem(units=Units(length='nm'),
        stack=rcwa.Stack(period=(900.,850.), bottom=Material(2.25),
            layers=(rcwa.Layer(thickness=100., background=Material(4.+.6j)),)),
        excitation=rcwa.Excitation(wavelength=600., sources=sources),
        outputs=rcwa.Outputs(diffraction=False, fields=True, power_region=region))


def main():
    numerics = rcwa.Numerics()
    top = rcwa.PlaneWave(side='top', kx=.002, ky=0.)
    bottom = rcwa.PlaneWave(side='bottom', kx=.002, ky=0.)
    prepared = rcwa.prepare(problem((top,bottom)), numerics=numerics)
    print('bottom phase [deg]    output_top    output_bottom    absorption')
    for phase in (0.,90.,180.):
        result = prepared.run(problem((top,bottom.replace(amplitude=np.exp(1j*np.deg2rad(phase))))))
        power_top, power_bottom = result.outputs['top'], result.outputs['bottom']
        incoming = power_top.incoming+power_bottom.incoming
        out_top, out_bottom = float((power_top.outgoing/incoming)[0]), float((power_bottom.outgoing/incoming)[0])
        absorption = float(result.absorption[0])
        print(f'{phase:18.0f}    {out_top:.6f}      {out_bottom:.6f}         {absorption:.6f}')
        np.testing.assert_allclose(out_top+out_bottom+absorption, 1., atol=1e-12)
    independent = problem((rcwa.PlaneWave(side='top',theta=np.deg2rad(20.),phi=0.),
        rcwa.PlaneWave(side='bottom',theta=np.deg2rad(35.),phi=0.,amplitude=.7j)), region='infinite')
    result = rcwa.run(independent, numerics=numerics)
    assert result.diagnostics['source_groups'] == (2,)
    fields = result.boundary_fields(side='top', x=np.linspace(-400.,400.,81), y=[0.])
    assert fields.values.shape == (1,1,81,6)
    regional = rcwa.run(independent.replace(outputs=independent.outputs.replace(
        power_region=((-400.,400.),(-300.,300.)))), numerics=numerics)
    assert 'absorption' not in regional.outputs
    assert regional.outputs['top'].units == 'W'
    print('Independent directions:', result.diagnostics['source_groups'][0], 'calculation groups')


if __name__ == '__main__':
    main()
