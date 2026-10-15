"""Overlapping curves: circular NVF follows the painting's polygon facets."""

from wavebit import Material, Units, rcwa
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)


def problem(width, epsilon):
    return rcwa.Problem(units=Units(length='nm'),
        stack=rcwa.Stack(period=(700., 700.), bottom=Material(2.25), layers=(
            rcwa.Layer(thickness=300., regions=(
                rcwa.Region(geometry=rcwa.Ellipse(radii=(width/2, 100.), angle=.17), material=Material(4.)),
                rcwa.Region(geometry=rcwa.Circle(radius=125., center=(130., 20.)), material=Material(epsilon)))),)),
        excitation=rcwa.Excitation(wavelength=900., sources=(
            rcwa.PlaneWave(theta=.2, phi=.13, polarization=(np.sin(.23), np.cos(.23))),)))


NUMERICS = rcwa.Numerics(orders=(2, 2), factorization='nvf', nvf_projector='circular',
                         shape_nvf_grid_size=64, topology_check='host')


def main():
    prepared = rcwa.prepare(problem(300., 9.), numerics=NUMERICS, backend='jax')

    def transmission(p):
        return prepared.run(problem(p[0], p[1])).T.sum().real

    value, derivative = jax.jit(jax.value_and_grad(transmission))(jnp.array([300., 9.]))
    print('Overlapping ellipse and circle: factorization=nvf, nvf_projector=circular')
    print(f'T={float(value):.12f}')
    print(f'dT/dwidth={float(derivative[0]):+.8e}; dT/d epsilon={float(derivative[1]):+.8e}')
    # Fresh eager solves check the live geometry without replaying its record.
    h = 1e-4
    fd = (rcwa.run(problem(300.+h, 9.), numerics=NUMERICS).T.sum()-rcwa.run(problem(300.-h, 9.), numerics=NUMERICS).T.sum())/(2*h)
    print(f'eager finite difference dT/dwidth={float(fd):+.8e}')
    np.testing.assert_allclose(derivative[0], fd, rtol=1e-5, atol=1e-9)
    # A changed painting plan fails validity under JIT; prepare the new concrete geometry.


if __name__ == '__main__':
    main()
