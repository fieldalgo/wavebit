"""Material derivative at fixed topology, checked against two fresh differences."""
import json
import numpy as np
import jax
import jax.numpy as jnp
from wavebit import Material, Units, rcwa

jax.config.update('jax_enable_x64', True)


def problem(epsilon):
    return rcwa.Problem(units=Units(length='nm'),
        stack=rcwa.Stack(period=(700., 700.), bottom=Material(2.25), layers=(
            rcwa.Layer(thickness=180., regions=(rcwa.Region(
                geometry=rcwa.Rectangle(size=(300., 220.), angle=.25),
                material=Material(epsilon)),)),)),
        excitation=rcwa.Excitation(wavelength=900., sources=(
            rcwa.PlaneWave(theta=.2, phi=.3, polarization='p'),)))


NUMERICS = rcwa.Numerics(orders=(2, 2), factorization='nvf',
    nvf_projector='circular', shape_nvf_grid_size=128)


if __name__ == '__main__':
    prepared = rcwa.prepare(problem(4.), numerics=NUMERICS, backend='jax')
    def transmission(epsilon):
        return prepared.run(problem(epsilon)).T.sum()
    value, gradient = jax.jit(jax.value_and_grad(transmission))(jnp.asarray(4.))
    # Independent eager preparations on NumPy for each finite-difference point.
    def eager(epsilon):
        return float(rcwa.run(problem(epsilon), numerics=NUMERICS).T.sum())
    differences = []
    for h in (1e-3, 5e-4):
        fd = (eager(4.+h)-eager(4.-h))/(2*h)
        np.testing.assert_allclose(gradient, fd, rtol=2e-5, atol=1e-8)
        differences.append(dict(h=h, derivative=fd, absolute_difference=abs(fd-float(gradient))))
    np.testing.assert_allclose(value, eager(4.), atol=2e-12)
    print(json.dumps(dict(T_p=float(value), dT_d_epsilon=float(gradient),
                         eager_T_p=eager(4.), finite_differences=differences), indent=2))
