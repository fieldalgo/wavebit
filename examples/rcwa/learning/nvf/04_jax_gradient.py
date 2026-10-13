"""Material derivative with fixed shape topology; compare two FD step sizes."""

import json
import jax
import jax.numpy as jnp
from wavebit.compat import legacy as wavebit

jax.config.update('jax_enable_x64', True)


def make_solver(epsilon):
    return wavebit.rcwa(
        backend='jax', complex_dtype=0,
        period=(700., 700.), wavelength=900., thickness=(180.,),
        top=1., bot=2.25, theta=.2, phi=.3, psi=0.,
        truncation_order=(2, 2),
        factorization='nvf', nvf_projector='circular', shape_nvf_grid_size=128,
        structure=[dict(background=1., shapes=[
            dict(type='rectangle', cx=0., cy=0., lx=300., ly=220.,
                 angle=.25, material=epsilon)])])


if __name__ == '__main__':
    # Take the discrete geometry decisions before JIT tracing.
    geometry = make_solver(4.).freeze_geometry()

    def transmission(epsilon):
        result = make_solver(epsilon).solve(geometry=geometry)
        return jnp.real(jnp.sum(result.T))

    value, gradient = jax.jit(jax.value_and_grad(transmission))(jnp.asarray(4.))
    # Rebuild eagerly for finite differences: no reuse of the frozen record.
    def eager(epsilon):
        return float(make_solver(epsilon).solve().T.sum())

    differences = []
    for h in (1e-3, 5e-4):
        fd = (eager(4.+h)-eager(4.-h))/(2*h)
        differences.append(dict(h=h, derivative=fd,
                                absolute_difference=abs(fd-float(gradient))))
    print(json.dumps(dict(T_p=float(value), dT_d_epsilon=float(gradient),
                         eager_T_p=eager(4.), finite_differences=differences), indent=2))
