"""Overlapping curves: circular NVF follows the painting's polygon facets."""
import jax
import jax.numpy as jnp
import wavebit

jax.config.update('jax_enable_x64', True)


def solver(width, epsilon):
    return wavebit.rcwa(
        backend=1, period=(700., 700.), wavelength=900., top=1., bot=2.25,
        theta=.2, phi=.13, psi=.23, thickness=(300.,), truncation_order=(2, 2),
        factorization='nvf', projector='circular', normal_grid=64,
        structure=[dict(background=1., shapes=[
            dict(type='ellipse', lx=width, ly=200., angle=.17, material=4.),
            dict(type='circle', cx=130., cy=20., lx=250., material=epsilon)])])


def main():
    geometry = solver(300., 9.).freeze_geometry()

    def transmission(p):
        return solver(p[0], p[1]).solve(geometry=geometry).T.sum().real

    value, derivative = jax.jit(jax.value_and_grad(transmission))(jnp.array([300., 9.]))
    print('Overlapping ellipse and circle: factorization=nvf, projector=circular')
    print(f'T={float(value):.12f}')
    print(f'dT/dwidth={float(derivative[0]):+.8e}; dT/d epsilon={float(derivative[1]):+.8e}')
    # Fresh eager solves check the live geometry without replaying its record.
    h = 1e-4
    fd = (solver(300.+h, 9.).solve().T.sum()-solver(300.-h, 9.).solve().T.sum())/(2*h)
    print(f'eager finite difference dT/dwidth={float(fd):+.8e}')
    # Re-freeze when intersection topology changes; a traced solve cannot check it.


if __name__ == '__main__':
    main()
