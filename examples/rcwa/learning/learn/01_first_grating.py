"""One periodic grating; explicit nm units and relative permittivities."""
import json
import numpy as np
from wavebit import Material, Units, rcwa

problem = rcwa.Problem(units=Units(length='nm'),
    stack=rcwa.Stack(period=(700., 700.), bottom=Material(2.25), layers=(
        rcwa.Layer(thickness=300., epsilon=np.array([[1., 1., 12.25, 12.25]])),)),
    excitation=rcwa.Excitation(wavelength=900., sources=(rcwa.PlaneWave(polarization='s'),)))
result = rcwa.run(problem, numerics=rcwa.Numerics(orders=(10, 0)))
np.testing.assert_allclose(result.R.sum()+result.T.sum(), 1., atol=2e-12)
print(json.dumps(dict(R=float(result.R.sum()), T=float(result.T.sum()),
    T00=float(result.transmission.order(0, 0).efficiency[0]), shape=list(result.T.shape),
    orders_u=np.asarray(result.transmission.orders_u).tolist(),
    orders_v=np.asarray(result.transmission.orders_v).tolist(),
    T_orders=np.asarray(result.T).tolist()), indent=2))
