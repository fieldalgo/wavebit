"""Exact Cartesian Li: guarded JIT derivatives of a closing rectangular gap."""
import jax
import jax.numpy as jnp
import numpy as np
from wavebit import Material, Units, rcwa

jax.config.update('jax_enable_x64', True)


def problem(gap, right_epsilon=9.):
    regions = tuple(rcwa.Region(geometry=rcwa.Rectangle(size=(300., 300.),
        center=(sign*(150.+gap/2), 0.)), material=Material(epsilon))
        for sign, epsilon in ((-1., 4.), (1., right_epsilon)))
    return rcwa.Problem(units=Units(length='nm'),
        stack=rcwa.Stack(period=(700., 700.), bottom=Material(2.25),
            layers=(rcwa.Layer(thickness=300., background=Material(1.), regions=regions),)),
        excitation=rcwa.Excitation(wavelength=900., sources=(rcwa.PlaneWave(theta=.2, phi=0., polarization='p'),)))


def main():
    numerics = rcwa.Numerics(orders=(3, 3), factorization='li')
    prepared = rcwa.prepare(problem(0.), numerics=numerics, backend='jax')
    def transmission(gap, epsilon):
        return prepared.run(problem(gap, epsilon)).T.sum().real
    evaluate = jax.jit(jax.value_and_grad(transmission, argnums=(0, 1)))
    print('gap            T              dT/dgap       dT/d(right epsilon)')
    for gap in (1., 1e-2, 1e-4, 1e-6, 0.):
        value, (dg, de) = evaluate(jnp.asarray(gap), jnp.asarray(9.))
        assert np.isfinite([value, dg, de]).all()
        print(f'{gap:8.1e}  {float(value):.12f}  {float(dg):+.8e}  {float(de):+.8e}')
    # Contact follows the opening side. Overlap invalidates the retained order.
    invalid = jax.jit(lambda gap: prepared.run(problem(gap)))(-1.)
    assert not bool(invalid.validity.checks['rectilinear_topology'])
    assert np.isnan(invalid.T).all()


if __name__ == '__main__':
    main()
