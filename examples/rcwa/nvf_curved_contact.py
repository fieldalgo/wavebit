"""Different-material tangent ellipses with circular NVF and live derivatives."""

from wavebit import Material, Units, rcwa
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)


def problem(gap, epsilon):
    regions = tuple(rcwa.Region(geometry=rcwa.Ellipse(radii=(150., 100.), center=(center, 0.)), material=Material(material))
                    for center, material in ((-150.-gap/2, 4.), (150.+gap/2, epsilon)))
    return rcwa.Problem(units=Units(length='nm'),
        stack=rcwa.Stack(period=(700., 700.), bottom=Material(2.25),
            layers=(rcwa.Layer(thickness=300., regions=regions),)),
        excitation=rcwa.Excitation(wavelength=900., sources=(rcwa.PlaneWave(theta=.2, polarization='p'),)))


NUMERICS = rcwa.Numerics(orders=(3, 3), factorization='nvf', nvf_projector='circular',
                         shape_nvf_grid_size=128, topology_check='host')


def main():
    prepared = rcwa.prepare(problem(0., 9.), numerics=NUMERICS, backend='jax')

    def transmission(gap, epsilon):
        return prepared.run(problem(gap, epsilon)).T.sum().real

    evaluate = jax.jit(jax.value_and_grad(transmission, argnums=(0, 1)))
    print('Tangent ellipses: factorization=nvf, nvf_projector=circular')
    print('gap             T             dT/dgap        dT/d epsilon')
    for gap in (1e-2, 1e-4, 1e-6, 0.):
        value, (dg, de) = evaluate(jnp.asarray(gap), jnp.asarray(9.))
        print(f'{gap:8.1e}  {float(value):.12f}  {float(dg):+.8e}  {float(de):+.8e}')
    # Independent eager solve and one-sided difference at the tangent geometry.
    contact = float(rcwa.run(problem(0., 9.), numerics=NUMERICS).T.sum())
    fd = (float(rcwa.run(problem(1e-4, 9.), numerics=NUMERICS).T.sum()) - contact) / 1e-4
    np.testing.assert_allclose(value, contact, atol=2e-12)
    np.testing.assert_allclose(dg, fd, rtol=5e-6, atol=1e-9)
    print(f'eager contact={contact:.12f}; one-sided dT/dgap={fd:+.8e}')


if __name__ == '__main__':
    main()
