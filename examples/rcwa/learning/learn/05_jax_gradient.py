"""Differentiate film thickness / wavelength; eager JAX with prepared reuse."""
import json
import numpy as np
import jax
import jax.numpy as jnp
from wavebit import Material, Units, rcwa

jax.config.update('jax_enable_x64', True)


def problem(q):
    return rcwa.Problem(units=Units(length='nm'),
        stack=rcwa.Stack(period=(700., 700.), bottom=Material(2.25), layers=(
            rcwa.Layer(thickness=900.*q, background=Material(4.)),)),
        excitation=rcwa.Excitation(wavelength=900., sources=(rcwa.PlaneWave(polarization='s'),)))


prepared = rcwa.prepare(problem(.1), numerics=rcwa.Numerics(orders=(0, 0)), backend='jax')
def reflectance(q):
    return prepared.run(problem(q)).R.sum()

q = jnp.asarray(.10)
value, gradient = jax.value_and_grad(reflectance)(q)
h = 1e-5
fd = (reflectance(q+h)-reflectance(q-h))/(2*h)
np.testing.assert_allclose(gradient, fd, rtol=2e-7, atol=1e-9)
history = [dict(q=float(q), R=float(value))]
for _ in range(12):
    _, g = jax.value_and_grad(reflectance)(q)
    q = jnp.clip(q - .02*g, .03, .35)
    history.append(dict(q=float(q), R=float(reflectance(q))))
print(json.dumps(dict(q0=.1, R=float(value), gradient=float(gradient),
    finite_difference=float(fd), absolute_difference=float(abs(gradient-fd)),
    optimization=history), indent=2))
