"""Select Li or NVF for the same physical Cartesian rectangle."""
import json
import numpy as np
from wavebit import Material, Units, rcwa


def run(method):
    options = dict(nvf_projector='linear', shape_nvf_grid_size=128) if method == 'nvf' else {}
    problem = rcwa.Problem(units=Units(length='nm'),
        stack=rcwa.Stack(period=(700., 700.), bottom=Material(2.25), layers=(
            rcwa.Layer(thickness=180., regions=(rcwa.Region(
                geometry=rcwa.Rectangle(size=(300., 220.)), material=Material(4.)),)),)),
        excitation=rcwa.Excitation(wavelength=900., sources=(
            rcwa.PlaneWave(theta=.2, phi=.3, polarization='p'),)))
    result = rcwa.run(problem, numerics=rcwa.Numerics(orders=(2, 2), factorization=method, **options))
    r, t = float(result.R.sum()), float(result.T.sum())
    np.testing.assert_allclose(r+t, 1., atol=2e-12)
    return dict(method=method, R_p=r, T_p=t, energy_balance_residual=abs(r+t-1.))


if __name__ == '__main__':
    print(json.dumps([run('li'), run('nvf')], indent=2))
