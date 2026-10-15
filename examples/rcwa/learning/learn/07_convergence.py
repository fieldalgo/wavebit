"""A deliberately short Fourier-order study, not an optical convergence certificate."""
import json
import numpy as np
from wavebit import Material, Units, rcwa

problem = rcwa.Problem(units=Units(length='nm'),
    stack=rcwa.Stack(period=(700., 700.), bottom=Material(2.25), layers=(
        rcwa.Layer(thickness=300., epsilon=np.array([[1., 1., 12.25, 12.25]])),)),
    excitation=rcwa.Excitation(wavelength=900., sources=(rcwa.PlaneWave(polarization='p'),)))
rows = []
for n in (1, 3, 5, 7, 11):
    r = rcwa.run(problem, numerics=rcwa.Numerics(orders=(n, 0),
        fourier_coefficients='integrated', factorization='li'))
    np.testing.assert_allclose(r.R.sum()+r.T.sum(), 1., atol=2e-12)
    rows.append(dict(Nx=n, R=float(r.R.sum()), T=float(r.T.sum()),
                     energy_residual=float(abs(r.R.sum()+r.T.sum()-1))))
print(json.dumps(dict(rows=rows), indent=2))
