"""One circular inclusion: physical shape NVF, NumPy and double precision."""
import json
import numpy as np
from wavebit import Material, Units, rcwa


def problem(polarization='p'):
    return rcwa.Problem(units=Units(length='nm'),
        stack=rcwa.Stack(period=(700., 700.), bottom=Material(2.25), layers=(
            rcwa.Layer(thickness=180., regions=(rcwa.Region(
                geometry=rcwa.Circle(radius=160.), material=Material(4.)),)),)),
        excitation=rcwa.Excitation(wavelength=900., sources=(
            rcwa.PlaneWave(theta=.2, phi=.3, polarization=polarization),)))


def run(order=(2, 2), shape_nvf_grid_size=128, polarization='p'):
    return rcwa.run(problem(polarization), numerics=rcwa.Numerics(orders=order,
        factorization='nvf', nvf_projector='linear', shape_nvf_grid_size=shape_nvf_grid_size))


def summarize():
    output = {}
    for polarization in ('p', 's'):
        result = run(polarization=polarization)
        r, t = float(result.R.sum()), float(result.T.sum())
        np.testing.assert_allclose(r+t, 1., atol=2e-12)
        output[polarization] = dict(R=r, T=t, energy_balance_residual=abs(r+t-1.))
    output['array_shape'] = list(result.T.shape)
    output['orders_u'] = np.asarray(result.transmission.orders_u).tolist()
    output['orders_v'] = np.asarray(result.transmission.orders_v).tolist()
    return output


if __name__ == '__main__':
    print(json.dumps(summarize(), indent=2))
