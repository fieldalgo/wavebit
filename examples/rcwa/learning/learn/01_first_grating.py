"""One periodic grating; all lengths in nm, all materials are epsilon_r."""

import json
import numpy as np
from wavebit.compat import legacy as wavebit

# Enter relative permittivity directly, matching the diagram.
structure = np.array([[[1., 1., 12.25, 12.25]]])
solver = wavebit.rcwa(
    backend='numpy', wavelength=900., period=[700.],
    truncation_order=dict(x=10, y=0),
    top=1., bot=2.25,
    theta=0., phi=0., tetm='te',
    structure=structure, thickness=[300.],
)
result = solver.solve()
print(json.dumps(dict(R=float(result.R.sum()), T=float(result.T.sum()),
    T00=float(result.order(0, 0).T), shape=list(result.T.shape),
    orders_u=np.asarray(result.orders_u).tolist(),
    orders_v=np.asarray(result.orders_v).tolist(),
    T_orders=np.asarray(result.T).tolist()), indent=2))
