"""Uniform s-polarized film: explicit spectral axis, E in V/m and H in A/m."""
import json
import numpy as np
from wavebit import Material, Units, rcwa

problem = rcwa.Problem(units=Units(length='nm'),
    stack=rcwa.Stack(period=(700., 700.), bottom=Material(2.25), layers=(
        rcwa.Layer(thickness=120., background=Material(4.)),)),
    excitation=rcwa.Excitation(wavelength=900., sources=(rcwa.PlaneWave(polarization='s'),)),
    outputs=rcwa.Outputs(fields=True))
result = rcwa.run(problem, numerics=rcwa.Numerics(orders=(0, 0)))
z = np.linspace(0., 120., 25)
field = result.field(x=[0.], y=[0.], z=z)
Ey = np.asarray(field.values[0, :, 0, 0, 1])
np.testing.assert_allclose(result.R.sum()+result.T.sum(), 1., atol=2e-12)
assert field.units['Ey'] == 'V/m' and field.units['Hx'] == 'A/m' and field.units['coordinates'] == 'nm'
print(json.dumps(dict(shape=list(field.values.shape), components=list(field.components),
    units=dict(field.units), R=float(result.R.sum()), T=float(result.T.sum()), z_nm=z.tolist(),
    Ey_real=Ey.real.tolist(), Ey_imag=Ey.imag.tolist(), Ey_abs2=(np.abs(Ey)**2).tolist()), indent=2))
