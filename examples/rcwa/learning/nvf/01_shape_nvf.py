"""One circular inclusion: explicit shape NVF, NumPy, double precision."""

import json
import numpy as np
from wavebit.compat import legacy as wavebit


def make_solver(order=(2, 2), shape_nvf_grid_size=128):
    # Lengths are in nm; material numbers are relative permittivities.
    return wavebit.rcwa(
        backend='numpy', complex_dtype=0,
        period=(700., 700.), wavelength=900., thickness=(180.,),
        top=1., bot=2.25, theta=.2, phi=.3, psi=0.,
        truncation_order=order,
        factorization='nvf', nvf_projector='linear', shape_nvf_grid_size=shape_nvf_grid_size,
        structure=[dict(background=1., shapes=[
            dict(type='circle', cx=0., cy=0., lx=320., material=4.)])])


def summarize(result):
    output = {}
    for name, incident in [('p', result.p_inc), ('s', result.s_inc)]:
        r = float(np.sum(incident.R))
        t = float(np.sum(incident.T))
        output[name] = dict(R=r, T=t, energy_balance_residual=abs(r+t-1.))
    output['array_shape'] = list(result.T.shape)
    output['orders_u'] = np.asarray(result.orders_u).tolist()
    output['orders_v'] = np.asarray(result.orders_v).tolist()
    return output


if __name__ == '__main__':
    print(json.dumps(summarize(make_solver().solve()), indent=2))
