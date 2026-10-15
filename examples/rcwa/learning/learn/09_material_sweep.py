"""Assign a physical table once; reevaluate its response at every wavelength."""
import json
import numpy as np
from wavebit import Material, Units, materials, rcwa

silicon = materials.load('Si')
problem = rcwa.Problem(units=Units(length='nm'),
    stack=rcwa.Stack(period=(600., 600.), bottom=Material(2.25), layers=(
        rcwa.Layer(thickness=120., regions=(rcwa.Region(
            geometry=rcwa.Circle(radius=120.), material=silicon),)),)),
    excitation=rcwa.Excitation(wavelength=900., sources=(rcwa.PlaneWave(polarization='s'),)))
prepared = rcwa.prepare(problem, numerics=rcwa.Numerics(orders=(1, 1), shape_nvf_grid_size=64))
rows = []
for wavelength in (800., 900., 1000.):
    live = problem.replace(excitation=problem.excitation.replace(wavelength=wavelength))
    result = prepared.run(live)
    epsilon = complex(silicon.at(wavelength, units=problem.units).epsilon)
    assert np.isfinite(result.T).all()
    rows.append(dict(wavelength=wavelength, epsilon=[epsilon.real, epsilon.imag],
                     R=float(result.R.sum()), T=float(result.T.sum())))
print(json.dumps(dict(material_type=type(silicon).__name__, rows=rows), indent=2))
