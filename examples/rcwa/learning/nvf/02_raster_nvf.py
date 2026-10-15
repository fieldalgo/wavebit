"""Pixelated disk: raster NVF uses the image grid, not shape_nvf_grid_size."""
import json
import numpy as np
from wavebit import Material, Units, rcwa


def main():
    pixels = 64
    centers = ((np.arange(pixels)+.5)/pixels-.5)*700.
    x, y = np.meshgrid(centers, centers, indexing='xy')
    epsilon = np.where(x*x+y*y < 160.**2, 4., 1.)
    problem = rcwa.Problem(units=Units(length='nm'),
        stack=rcwa.Stack(period=(700., 700.), bottom=Material(2.25), layers=(
            rcwa.Layer(thickness=180., epsilon=epsilon),)),
        excitation=rcwa.Excitation(wavelength=900., sources=(
            rcwa.PlaneWave(theta=.2, phi=.3, polarization='p'),)))
    result = rcwa.run(problem, numerics=rcwa.Numerics(orders=(2, 2),
        fourier_coefficients='integrated', factorization='nvf',
        raster_normal_method='smooth', nvf_projector='linear'))
    r, t = float(result.R.sum()), float(result.T.sum())
    np.testing.assert_allclose(r+t, 1., atol=2e-12)
    print(json.dumps(dict(pixels=pixels, R_p=r, T_p=t,
                         energy_balance_residual=abs(r+t-1.)), indent=2))


if __name__ == '__main__':
    main()
