"""Differentiate R with respect to film thickness / wavelength; eager JAX."""

import json
from wavebit.compat import legacy as wavebit
wavebit.enable_x64()
import jax
import jax.numpy as jnp

def reflectance(q):
    solver = wavebit.rcwa(
        backend='jax', wavelength=900., period=(700., 700.),
        structure=jnp.array([[[4.]]]), thickness=jnp.array([900.*q]),
        top=1., bot=2.25, tetm='te', truncation_order=(0, 0),
    )
    return jnp.sum(solver.solve().R)

q = jnp.asarray(0.10)
value, gradient = jax.value_and_grad(reflectance)(q)
h = 1e-5
fd = (reflectance(q+h)-reflectance(q-h))/(2*h)
history = [dict(q=float(q), R=float(value))]
for _ in range(12):
    _, g = jax.value_and_grad(reflectance)(q)
    q = jnp.clip(q - 0.02*g, 0.03, 0.35)
    history.append(dict(q=float(q), R=float(reflectance(q))))
print(json.dumps(dict(q0=0.1, R=float(value), gradient=float(gradient),
    finite_difference=float(fd), absolute_difference=float(abs(gradient-fd)),
    optimization=history), indent=2))
