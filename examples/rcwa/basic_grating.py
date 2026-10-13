"""Current immutable RCWA lifecycle: one lossless binary grating."""
import numpy as np
from wavebit import Material, Units, rcwa

problem = rcwa.Problem(
    units=Units(length="nm"),
    stack=rcwa.Stack(
        period=(700., 700.),
        bottom=Material(epsilon=2.25),
        layers=(rcwa.Layer(thickness=300., epsilon=np.array([[1., 1., 12.25, 12.25]])),),
    ),
    excitation=rcwa.Excitation(wavelength=900., sources=(rcwa.PlaneWave(polarization="s"),)),
)
numerics = rcwa.Numerics(orders=(10, 0))
result = rcwa.run(problem, numerics=numerics)
print(float(result.R.sum()), float(result.T.sum()))
print(result.transmission.order(0, 0).efficiency)

assert abs(float(result.R.sum()+result.T.sum())-1.) < 1e-10
