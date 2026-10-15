"""A circular pillar. Small orders/grid teach the API, not optical convergence."""
import json
import numpy as np
from wavebit import Material, Units, rcwa

problem = rcwa.Problem(units=Units(length='nm'),
    stack=rcwa.Stack(period=(600., 600.), bottom=Material(2.25), layers=(
        rcwa.Layer(thickness=180., regions=(rcwa.Region(
            geometry=rcwa.Circle(radius=120.), material=Material(4.)),)),)),
    excitation=rcwa.Excitation(wavelength=850., sources=(
        rcwa.PlaneWave(theta=.12, phi=.27, polarization='s'),)))
prepared = rcwa.prepare(problem, numerics=rcwa.Numerics(orders=(2, 2),
    factorization='nvf', nvf_projector='linear', shape_nvf_grid_size=64))
result = prepared.run(problem)
np.testing.assert_allclose(result.R.sum()+result.T.sum(), 1., atol=2e-12)
print(json.dumps(dict(R=float(result.R.sum()), T=float(result.T.sum()),
    R_plus_T=float(result.R.sum()+result.T.sum()), shape=list(result.T.shape),
    coefficients=prepared.report.resolved['fourier_coefficients'],
    factorization=prepared.report.resolved['factorization']), indent=2))
