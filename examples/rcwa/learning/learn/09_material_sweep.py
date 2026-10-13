"""Assign a dispersive material once; RCWA reevaluates it at each wavelength."""

import json
from wavebit.compat import legacy as wavebit

silicon = wavebit.materials.catalog.load('Si')
solver = wavebit.rcwa(
    wavelength=900., period=(600., 600.), thickness=[120.],
    structure=[dict(background=1., shapes=[
        dict(type='circle', cx=0., cy=0., lx=240., material=silicon),
    ])],
    truncation_order=(1, 1), shape_nvf_grid_size=64,
    polarization=(1., 0.), top=1., bot=2.25,
)
rows = []
for wavelength in (800., 900., 1000.):
    result = solver.solve(wavelength=wavelength)
    epsilon = complex(solver.structure[0]['shapes'][0]['material'].epsilon)
    rows.append(dict(wavelength=wavelength, epsilon=[epsilon.real, epsilon.imag],
                     R=float(result.R.sum()), T=float(result.T.sum())))
print(json.dumps(dict(material_type=type(silicon).__name__, rows=rows), indent=2))
