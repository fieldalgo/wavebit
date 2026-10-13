"""Exact Cartesian Li for a closing gap; run from the repository with PYTHONPATH=."""

from wavebit.compat import legacy as _legacy_wavebit
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)


def solver(gap, right_epsilon=9.):
    shapes = [dict(type='rectangle', cx=-150.-gap/2, lx=300., ly=300., material=4.),
              dict(type='rectangle', cx=150.+gap/2, lx=300., ly=300., material=right_epsilon)]
    return _legacy_wavebit.rcwa(
        backend='jax', period=(700., 700.), wavelength=900., top=1., bot=2.25,
        theta=.2, phi=0., psi=0., thickness=(300.,), truncation_order=(3, 3),
        factorization='li', structure=[dict(background=1., shapes=shapes)])


def main():
    # This edge order permits contact and positive gaps. Refreeze for overlaps.
    geometry = solver(0.).freeze_geometry()

    def transmission(gap, epsilon):
        return jnp.sum(solver(gap, epsilon).solve(geometry=geometry).T).real

    evaluate = jax.jit(jax.value_and_grad(transmission, argnums=(0, 1)))
    print('gap            T              dT/dgap       dT/d(right epsilon)')
    for gap in (1., 1e-2, 1e-4, 1e-6, 0.):
        value, (dg, de) = evaluate(jnp.asarray(gap), jnp.asarray(9.))
        print(f'{gap:8.1e}  {float(value):.12f}  {float(dg):+.8e}  {float(de):+.8e}')
    # The derivative at contact follows the gap-opening side of the frozen plan.


if __name__ == '__main__':
    main()
