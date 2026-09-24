"""Shape declarations stay static while field solves trace their numeric values."""
import os
os.environ.setdefault('JAX_ENABLE_X64', '1')
import numpy as np
import pytest
import wavebit

jax = pytest.importorskip('jax')
import jax.numpy as jnp


def solver(width=230., kind='rectangle', painted=False, skew=False, backend=1, epsilon=4.):
    shape = dict(type=kind, cx=-31., cy=17., lx=width, ly=190., angle=.13,
                 material=wavebit.Material(epsilon, mu=1.1))
    shapes = [shape]
    if painted:
        shapes.append(dict(type='rectangle', cx=19., cy=-11., lx=110., ly=90.,
                           angle=.31, material=wavebit.Material(6., mu=1.2)))
    cell = dict(lattice=((700., 170.), (0., 700.))) if skew else dict(period=(700., 700.))
    return wavebit.rcwa(backend=backend, **cell, wavelength=850.,
                       truncation_order=(1, 1), thickness=(150.,), theta=.13,
                       phi=.21, psi=.37, shape_nvf_grid_size=32, answers='field',
                       structure=[dict(background=1., shapes=shapes)])


def test_shape_strings_are_static_but_dimensions_and_materials_are_leaves():
    a, b = solver(), solver(231.)
    leaves, tree = jax.tree_util.tree_flatten(a)
    assert not any(isinstance(v, str) for v in leaves)
    assert tree == jax.tree_util.tree_structure(b)
    assert tree != jax.tree_util.tree_structure(solver(kind='ellipse'))
    rebuilt = jax.tree_util.tree_unflatten(tree, leaves)
    assert rebuilt.structure[0]['shapes'][0]['type'] == 'rectangle'
    assert rebuilt.structure[0]['shapes'][0]['material'].mu == 1.1
    got = jax.jit(lambda s: s.structure[0]['shapes'][0]['lx'])(b)
    assert float(got) == 231.


@pytest.mark.parametrize('painted', [False, True])
@pytest.mark.parametrize('skew', [False, True])
def test_jitted_shape_field_matches_eager_without_leaving_tracers(painted, skew):
    s = solver(painted=painted, skew=skew)
    geometry = s.freeze_geometry()
    eager, ef = s.solve_field_eager(3, 3, 2, geometry=geometry)
    got, gf = s.solve_field(3, 3, 2, geometry=geometry)
    np.testing.assert_allclose(got.T, eager.T, rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose(gf, ef, rtol=2e-9, atol=2e-10)
    nr, nf = solver(painted=painted, skew=skew, backend=0).solve_field(3, 3, 2)
    np.testing.assert_allclose(got.T, nr.T, rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose(gf, nf, rtol=2e-9, atol=2e-10)
    assert not any(isinstance(v, jax.core.Tracer) for v in jax.tree_util.tree_leaves(s))


def test_jitted_field_retains_the_shape_width_derivative():
    geometry = solver().freeze_geometry()
    def loss(width):
        _, field = solver(width).solve_field(3, 3, 2, geometry=geometry)
        return jnp.sum(jnp.abs(field)**2).real
    got = float(jax.grad(loss)(jnp.asarray(230.)))
    h = 1e-3
    want = float((loss(230.+h)-loss(230.-h))/(2*h))
    assert np.isfinite(got) and abs(got) > 1e-6
    assert got == pytest.approx(want, rel=2e-4, abs=2e-7)


def test_jitted_field_retains_material_derivatives():
    geometry = solver().freeze_geometry()
    def loss(epsilon):
        _, field = solver(epsilon=epsilon).solve_field(3, 3, 2, geometry=geometry)
        return jnp.sum(jnp.abs(field)**2).real
    got = float(jax.grad(loss)(jnp.asarray(4.)))
    h = 1e-4
    want = float((loss(4.+h)-loss(4.-h))/(2*h))
    assert np.isfinite(got) and abs(got) > 1e-6
    assert got == pytest.approx(want, rel=2e-5, abs=2e-7)
