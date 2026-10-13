"""A deliberately small truncation study, not a production convergence certificate."""

import json
import numpy as np
from wavebit.compat import legacy as wavebit

rows = []
for n in (1, 3, 5, 7, 11):
    solver = wavebit.rcwa(
        backend='numpy', wavelength=900., period=(700., 700.),
        structure=np.array([[[1., 1., 12.25, 12.25]]]),
        thickness=[300.], top=1., bot=2.25, tetm='tm',
        truncation_order=(n, 0), fourier_coefficients='integrated',
        factorization='li',
    )
    r = solver.solve()
    rows.append(dict(Nx=n, R=float(r.R.sum()), T=float(r.T.sum()),
        energy_residual=float(abs(r.R.sum()+r.T.sum()-1))))
print(json.dumps(dict(rows=rows), indent=2))
