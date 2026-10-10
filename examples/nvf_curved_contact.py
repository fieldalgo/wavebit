"""Different-material tangent ellipses with circular NVF and live derivatives."""
import jax
import jax.numpy as jnp
import wavebit

jax.config.update('jax_enable_x64', True)


def solver(gap, epsilon):
    return wavebit.rcwa(
        backend='jax', period=(700., 700.), wavelength=900., top=1., bot=2.25,
        theta=.2, phi=0., psi=0., thickness=(300.,), truncation_order=(3, 3),
        factorization='nvf', nvf_projector='circular', shape_nvf_grid_size=128,
        structure=[dict(background=1., shapes=[
            dict(type='ellipse', cx=-150.-gap/2, lx=300., ly=200., material=4.),
            dict(type='ellipse', cx=150.+gap/2, lx=300., ly=200., material=epsilon)])])


def main():
    geometry = solver(0., 9.).freeze_geometry()

    def transmission(gap, epsilon):
        return solver(gap, epsilon).solve(geometry=geometry).T.sum().real

    evaluate = jax.jit(jax.value_and_grad(transmission, argnums=(0, 1)))
    print('Tangent ellipses: factorization=nvf, nvf_projector=circular')
    print('gap             T             dT/dgap        dT/d epsilon')
    for gap in (1e-2, 1e-4, 1e-6, 0.):
        value, (dg, de) = evaluate(jnp.asarray(gap), jnp.asarray(9.))
        print(f'{gap:8.1e}  {float(value):.12f}  {float(dg):+.8e}  {float(de):+.8e}')
    # Independent eager solve and one-sided difference at the tangent geometry.
    contact = float(solver(0., 9.).solve().T.sum())
    fd = (float(solver(1e-4, 9.).solve().T.sum()) - contact) / 1e-4
    print(f'eager contact={contact:.12f}; one-sided dT/dgap={fd:+.8e}')


if __name__ == '__main__':
    main()
