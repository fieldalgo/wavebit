"""A circular pillar. Small truncation/grid for teaching, not a converged design."""

import json
from wavebit.compat import legacy as wavebit

structure = [dict(background=1., shapes=[
    dict(type='circle', cx=0., cy=0., lx=240., material=4.)
])]
solver = wavebit.rcwa(
    backend='numpy', period=(600., 600.), wavelength=850.,
    structure=structure, thickness=[180.], top=1., bot=2.25,
    theta=0.12, phi=0.27, polarization=(1., 0.),
    truncation_order=dict(x=2, y=2),
    factorization='nvf', nvf_projector='linear', shape_nvf_grid_size=64,
)
result = solver.solve()
print(json.dumps(dict(R=float(result.R.sum()), T=float(result.T.sum()),
    R_plus_T=float(result.R.sum()+result.T.sum()),
    shape=list(result.T.shape), coefficients=str(solver.fourier_coefficients),
    factorization=str(solver.factorization)), indent=2))
