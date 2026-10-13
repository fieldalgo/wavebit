"""NVF contact and material derivatives with the rectangular Jones projector."""

from wavebit.compat import legacy as _legacy_wavebit
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)


def solver(gap, epsilon):
    return _legacy_wavebit.rcwa(
        backend='jax', period=(700., 700.), wavelength=900., top=1., bot=2.25,
        theta=.2, phi=0., psi=0., thickness=(300.,), truncation_order=(3, 3),
        factorization='nvf', nvf_projector='circular', shape_nvf_grid_size=128,
        structure=[dict(background=1., shapes=[
            dict(type='rectangle', cx=-150.-gap/2, lx=300., ly=300., material=4.),
            dict(type='rectangle', cx=150.+gap/2, lx=300., ly=300., material=epsilon)])])


def main():
    # Circular NVF can freeze at contact and open the gap with the same record.
    # Crossing into overlap still requires a new concrete painting plan.
    geometry = solver(0., 9.).freeze_geometry()

    def transmission(gap, epsilon):
        return jnp.sum(solver(gap, epsilon).solve(geometry=geometry).T).real

    evaluate = jax.jit(jax.value_and_grad(transmission, argnums=(0, 1)))
    print('factorization=nvf, nvf_projector=circular')
    print('gap             T             dT/dgap        dT/d epsilon')
    for gap in (1e-2, 1e-4, 1e-6, 1e-8, 0.):
        value, (dg, de) = evaluate(jnp.asarray(gap), jnp.asarray(9.))
        print(f'{gap:8.1e}  {float(value):.12f}  {float(dg):+.8e}  {float(de):+.8e}')
    # Independent eager evaluation verifies the traced contact value.
    print(f'contact     {float(solver(0., 9.).solve().T.sum()):.12f}')


if __name__ == '__main__':
    main()
