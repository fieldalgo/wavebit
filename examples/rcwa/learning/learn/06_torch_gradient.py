"""The same film and derivative as example 05, using eager PyTorch."""

import json
import torch
from wavebit.compat import legacy as wavebit

q = torch.tensor(0.10, dtype=torch.float64, requires_grad=True)
solver = wavebit.rcwa(
    backend='torch', wavelength=900., period=(700., 700.),
    structure=torch.tensor([[[4.]]], dtype=torch.float64),
    thickness=(900.*q).reshape(1), top=1., bot=2.25,
    tetm='te', truncation_order=(0, 0),
)
R = solver.solve().R.sum()
R.backward()
print(json.dumps(dict(R=float(R.detach()), gradient=float(q.grad)), indent=2))
