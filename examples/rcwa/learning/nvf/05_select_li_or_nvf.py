"""Select Li or NVF on the same supported Cartesian rectangle."""

import json
from wavebit.compat import legacy as wavebit


def run(method):
    options = dict(nvf_projector='linear', shape_nvf_grid_size=128) if method == 'nvf' else {}
    result = wavebit.rcwa(
        backend='numpy', complex_dtype=0,
        period=(700., 700.), wavelength=900., thickness=(180.,),
        top=1., bot=2.25, theta=.2, phi=.3, psi=0.,
        truncation_order=(2, 2), factorization=method,
        structure=[dict(background=1., shapes=[
            dict(type='rectangle', cx=0., cy=0., lx=300., ly=220., material=4.)])],
        **options).solve()
    r, t = float(result.R.sum()), float(result.T.sum())
    return dict(method=method, R_p=r, T_p=t, energy_balance_residual=abs(r+t-1.))


if __name__ == '__main__':
    print(json.dumps([run('li'), run('nvf')], indent=2))
