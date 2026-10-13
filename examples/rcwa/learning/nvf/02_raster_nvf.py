"""Pixelated disk: raster NVF uses the image grid, not shape_nvf_grid_size."""

import json
import numpy as np
from wavebit.compat import legacy as wavebit


def main():
    pixels = 64
    centers = ((np.arange(pixels)+.5)/pixels-.5)*700.
    x, y = np.meshgrid(centers, centers, indexing='xy')
    epsilon = np.where(x*x+y*y < 160.**2, 4., 1.)[None, :, :]
    result = wavebit.rcwa(
        backend='numpy', complex_dtype=0,
        period=(700., 700.), wavelength=900., thickness=(180.,),
        top=1., bot=2.25, theta=.2, phi=.3, psi=0.,
        truncation_order=(2, 2), structure=epsilon,
        fourier_coefficients='integrated', factorization='nvf',
        raster_normal_method='smooth', nvf_projector='linear').solve()
    r, t = float(result.R.sum()), float(result.T.sum())
    print(json.dumps(dict(pixels=pixels, R_p=r, T_p=t,
                         energy_balance_residual=abs(r+t-1.)), indent=2))


if __name__ == '__main__':
    main()
