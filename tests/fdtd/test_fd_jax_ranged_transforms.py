"""Eager ``jax.vmap`` and ``jax.jacfwd`` over a ranged body's geometry.

``jax.vmap`` and ``jax.jacfwd`` of ``Prepared.run`` with respect to the radius or centre of a
body given a ``Numerics(geometry_ranges=...)`` support raised JAX's raw
``AttributeError: 'SingleDeviceSharding' object has no attribute 'spec'`` from the shape's
fill: a batch tracer was converted with an explicit ``device=``. ``jax.jit(jax.vmap(...))``,
``jax.grad`` and ``jax.jacrev`` were unaffected. The batch is checked against a Python loop of
single runs, and the forward Jacobian against ``jax.grad`` (the reverse Jacobian of a scalar)
and a central difference of NumPy runs.
"""
import warnings

import numpy as np
import pytest

from wavebit import Material, Units, fdtd
from wavebit.units import C0

jax = pytest.importorskip('jax')
jnp = jax.numpy

#: radius, then the centre's x and y.
POINT = np.array([.22, -.073, .017])


def problem(v):
    shape = fdtd.Sphere(radius=v[0], center=(v[1], v[2], -.009))
    body = fdtd.Structure('ball', shape=shape, material=Material(epsilon=2.25, mu=1.2))
    return fdtd.Problem(
        units=Units(length='um'),
        scene=fdtd.Scene(domain=fdtd.Domain(size=(1.6,)*3),
                         boundaries=fdtd.Boundaries(x=fdtd.PEC(), y=fdtd.PEC(), z=fdtd.PEC()),
                         structures=(body,)),
        sources=(fdtd.CurrentDipole('d', center=(-.55, .03, -.02), current_moment=1e-16,
                                    pulse=fdtd.GaussianPulse(wavelength=3., frequency_sigma_hz=.03*C0/1e-6),
                                    delay_seconds=-18e-6/C0),),
        outputs=fdtd.Outputs(final_fields=True))


NUMERICS = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.1), duration=fdtd.Duration(steps=6),
                         geometry_ranges={'ball': ((-.36, .36),)*3})


def objective(backend):
    jax.config.update('jax_enable_x64', True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        prepared = fdtd.prepare(problem(POINT), numerics=NUMERICS, backend=backend)

    def f(v):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return (prepared.run(problem(v)).final_fields.values['E']**2).sum()
    return f


def test_eager_jax_vmap_over_radius_and_centre_matches_a_loop_of_runs():
    f = objective('jax')
    batch = np.stack([POINT, POINT + [.01, .02, -.015]])
    loop = np.array([float(f(jnp.asarray(v))) for v in batch])
    batched = np.asarray(jax.vmap(f)(jnp.asarray(batch)))
    np.testing.assert_allclose(batched, loop, rtol=1e-12, atol=0)
    assert batched[0] != batched[1]


@pytest.mark.slow    # about 75 s: three eager forward-mode directions and a reverse pass
def test_eager_jax_jacfwd_over_radius_and_centre_matches_grad_and_numpy_differences():
    f = objective('jax')
    forward = np.asarray(jax.jacfwd(f)(jnp.asarray(POINT)))
    # jax.grad is jax.jacrev of this scalar objective, and far cheaper eagerly.
    reverse = np.asarray(jax.grad(f)(jnp.asarray(POINT)))
    np.testing.assert_allclose(forward, reverse, rtol=1e-10, atol=0)
    g = objective('numpy')
    h = 1e-6
    central = np.array([(float(g(POINT + d)) - float(g(POINT - d)))/(2*h) for d in np.eye(3)*h])
    assert np.all(np.abs(central) > 1e-5)
    np.testing.assert_allclose(forward, central, rtol=2e-6, atol=0)
