"""Serialize a scene and run a three-point wavelength sweep."""
import json
import numpy as np
from wavebit.compat import legacy as wavebit

spec = wavebit.JobSpec(
    stack=wavebit.Stack(
        layers=[wavebit.Layer(thickness=300.,
            structure=np.array([[1., 1., 12.25, 12.25]]))],
        top=1., bot=2.25, period=(700., 700.)),
    incidence=wavebit.Incidence(wavelength=900., theta=0., phi=0., tetm='te'),
    numerics=wavebit.Numerics(truncation_order=(10, 0)),
    sweep=wavebit.Sweep.from_dict({'wavelength': {'linspace': [800., 1000., 3]}}),
)
restored = wavebit.JobSpec.from_json(spec.to_json())
assert restored.key() == spec.key()
rows = []
for point in restored.expand():
    result = wavebit.run(point)
    rows.append(dict(wavelength=float(point.incidence.wavelength),
        R=float(result.R.sum()), T=float(result.T.sum())))
print(json.dumps(dict(schema=spec.version, round_trip=True, rows=rows), indent=2))
