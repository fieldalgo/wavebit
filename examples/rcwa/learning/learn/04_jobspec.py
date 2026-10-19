"""Serialize physical Jobs and expand an explicit three-point wavelength sweep.

The filename is retained for course links. This writes wavebit.job/v2, not JobSpec.
"""
import json
import numpy as np
from wavebit import Job, Material, Units, rcwa, run

problem = rcwa.Problem(units=Units(length='nm'),
    stack=rcwa.Stack(period=(700., 700.), bottom=Material(2.25), layers=(
        rcwa.Layer(thickness=300., epsilon=np.array([[1., 1., 12.25, 12.25]])),)),
    excitation=rcwa.Excitation(wavelength=900., sources=(rcwa.PlaneWave(polarization='s'),)))
job = Job(solver='rcwa', problem=problem, numerics=rcwa.Numerics(orders=(10, 0)))
restored = Job.from_json(job.to_json())
assert restored.key == job.key
rows = []
for wavelength in np.linspace(800., 1000., 3):
    point = restored.replace(problem=restored.problem.replace(
        excitation=restored.problem.excitation.replace(wavelength=float(wavelength))))
    # Each point is independently portable, including the units and numerical policy.
    point = Job.from_json(point.to_json())
    result = run(point)
    np.testing.assert_allclose(result.R.sum()+result.T.sum(), 1., atol=2e-12)
    rows.append(dict(wavelength=float(point.problem.excitation.wavelength),
                     R=float(result.R.sum()), T=float(result.T.sum())))
print(json.dumps(dict(schema=json.loads(job.to_json())['schema'], round_trip=True, rows=rows), indent=2))
