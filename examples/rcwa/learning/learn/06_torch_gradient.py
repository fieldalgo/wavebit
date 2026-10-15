"""The same physical film and derivative as example 05, using eager PyTorch."""
import json
import torch
from wavebit import Material, Units, rcwa

q = torch.tensor(.10, dtype=torch.float64, requires_grad=True)
problem = rcwa.Problem(units=Units(length='nm'),
    stack=rcwa.Stack(period=(700., 700.), bottom=Material(2.25), layers=(
        rcwa.Layer(thickness=900.*q, background=Material(4.)),)),
    excitation=rcwa.Excitation(wavelength=900., sources=(rcwa.PlaneWave(polarization='s'),)))
R = rcwa.run(problem, numerics=rcwa.Numerics(orders=(0, 0)), backend='torch').R.sum()
R.backward()
assert torch.isfinite(q.grad) and abs(q.grad) > 0
print(json.dumps(dict(R=float(R.detach()), gradient=float(q.grad)), indent=2))
