"""Uniform TE film: the field layout is (z, y, x, Ex/Ey/Ez/Z0Hx/Z0Hy/Z0Hz)."""

import json
import numpy as np
from wavebit.compat import legacy as wavebit

solver = wavebit.rcwa(
    backend='numpy', wavelength=900., period=(700., 700.),
    structure=np.array([[[4.]]]), thickness=[120.],
    top=1., bot=2.25, tetm='te', truncation_order=(0, 0),
    answers='field',
)
result = solver.solve()
z = np.linspace(0., 120., 25)
cell = result.reconstruct_field(sample_x=[0.], sample_y=[0.], sample_z=z)
Ey = np.asarray(cell[:, 0, 0, 1])
print(json.dumps(dict(shape=list(cell.shape), components=['Ex', 'Ey', 'Ez', 'Z0Hx', 'Z0Hy', 'Z0Hz'],
    R=float(result.R.sum()), T=float(result.T.sum()), z_nm=z.tolist(),
    Ey_real=Ey.real.tolist(), Ey_imag=Ey.imag.tolist(),
    Ey_abs2=(np.abs(Ey)**2).tolist()), indent=2))
