"""Two-sided illumination: boundary phase controls output and absorption."""
import numpy as np
import wavebit


def main():
    solver = wavebit.rcwa(
        wavelength=600., period=(900., 850.), top=1., bot=2.25,
        structure=[dict(background=4.+.6j, shapes=[])], thickness=[100.],
        truncation_order=0, answers='field', kx=.002, ky=0.,
        excitation={'top': wavebit.PlaneWave(), 'bottom': wavebit.PlaneWave()},
    )
    print('bottom phase [deg]    output_top    output_bottom    absorption')
    for phase in (0., 90., 180.):
        result = solver.solve(excitation={
            'top': wavebit.PlaneWave(),
            'bottom': wavebit.PlaneWave(amplitude=np.exp(1j*np.deg2rad(phase))),
        })
        print(f'{phase:18.0f}    {result.output_top:.6f}      '
              f'{result.output_bottom:.6f}         {result.absorption:.6f}')
        np.testing.assert_allclose(result.output_top+result.output_bottom+result.absorption, 1., atol=1e-12)

    result = solver.solve(excitation={
        'top': wavebit.PlaneWave(theta=np.deg2rad(20.), phi=0.),
        'bottom': wavebit.PlaneWave(theta=np.deg2rad(35.), phi=0., amplitude=.7j),
    }, power_region='infinite')
    assert result.group_count == 2
    fields = result.boundary_fields('top', np.linspace(-400., 400., 81), [0.])
    assert fields.shape == (1, 81, 6)
    regional = result.boundary_power(((-400., 400.), (-300., 300.)))
    assert regional.absorption is None
    print('Independent directions:', result.group_count, 'calculation groups')


if __name__ == '__main__':
    main()
