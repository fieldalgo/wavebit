"""
The cell basis as a design variable: numbers traced, ``axis_aligned`` not.

``lattice`` was eager NumPy until 2026-08-23, so ``jax.grad`` could not reach it -- only
``period`` was differentiable, which is two edge lengths of an axis-aligned cell. A hex angle
or a shear was not expressible as a design variable at all.

Splitting it takes a pytree, and the split is the whole point: the four numbers are children,
``axis_aligned`` is aux_data. That flag decides *which branch runs* -- the rectangular fold in
``analytic._shape_surface_normal`` reads it, and it is what keeps the orthogonal path
bit-identical -- and a traced value cannot be branched on. As a child it would kill every
jitted solve while staying invisible eagerly.
"""
import numpy as np
import pytest

import wavebit
from wavebit._core.lattice import Lattice, as_lattice, is_axis_aligned, periods_of

HEX = np.column_stack([[400.0, 0.0], [200.0, 400.0 * np.sqrt(3) / 2]])
SHEAR = np.column_stack([[400.0, 0.0], [350.0, 150.0]])


def test_both_spellings_of_a_cell_arrive_as_one_type():
    for A, aligned in ((as_lattice(period=[700.0, 1100.0]), True),
                       (as_lattice(period=[640.0]), True),
                       (as_lattice(lattice=np.diag([700.0, 1100.0])), True),
                       (as_lattice(lattice=HEX), False),
                       (as_lattice(lattice=SHEAR), False)):
        assert isinstance(A, Lattice)
        assert A.axis_aligned is aligned
        assert is_axis_aligned(A) is aligned          # the free function reads either spelling

    # indexing still delegates, so every `A[0][1]` in the tree reads as it always did
    A = as_lattice(lattice=HEX)
    assert (A[0][0], A[0][1], A[1][0]) == (400.0, 200.0, 0.0)
    assert np.array_equal(np.asarray(A), HEX)
    assert periods_of(A) == periods_of(HEX)
    assert as_lattice(lattice=A) is A                 # idempotent


def test_the_flag_is_aux_data_and_the_numbers_are_children():
    """A child is traced; aux_data is baked into the treedef and can be branched on.

    Two lattices holding the same numbers but a different flag must have *different* tree
    structures -- that is what stops jit reusing one branch's trace for the other.
    """
    jax = pytest.importorskip('jax')
    A = as_lattice(lattice=HEX)
    leaves, treedef = jax.tree_util.tree_flatten(A)
    assert len(leaves) == 1 and np.shape(leaves[0]) == (2, 2)
    back = jax.tree_util.tree_unflatten(treedef, leaves)
    assert back.axis_aligned is False and np.array_equal(np.asarray(back), HEX)
    assert jax.tree_util.tree_structure(A) != jax.tree_util.tree_structure(
        Lattice(A.columns, True))


def test_a_basis_handed_in_traced_takes_the_general_path():
    """``axis_aligned`` cannot be read off a tracer, so it is ``False`` -- always correct.

    The oblique path solves a rectangular cell correctly; it is only not bit-identical to the
    shortcut. Guessing ``True`` would be the unsafe direction, and refusing would make the
    feature useless. A ``period`` stays ``True`` because it *spells* a diagonal.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    seen = []

    def f(M):
        seen.append(as_lattice(lattice=M).axis_aligned)
        return jnp.sum(M)

    jax.grad(f)(jnp.diag(jnp.array([700.0, 1100.0])))
    assert seen == [False]


def _analytic_loss(jnp, M, order=3):
    s = wavebit.rcwa(backend=1, wavelength=900.0, lattice=M, truncation_order=[order, order],
                     top=1.0, bot=1.0, theta=0.07, phi=0.31, psi=0.2, thickness=[300.0],
                     structure=[dict(background=1.0, shapes=[
                         dict(type='circle', cx=250.0, cy=150.0, lx=140.0, material=5.76)])],
                     factorization='nvf', normal_grid=64)
    return jnp.sum(s.solve().R)


def test_the_cell_geometry_is_a_design_variable():
    """All four basis components carry a gradient, the off-diagonals included.

    The off-diagonals are the new reach: they *are* the shear and the angle, and no spelling of
    ``period`` can express them.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    g = np.asarray(jax.grad(lambda M: _analytic_loss(jnp, M))(
        jnp.array([[400.0, 200.0], [0.0, 400.0 * np.sqrt(3) / 2]])))
    assert np.all(np.isfinite(g))
    assert np.count_nonzero(np.abs(g) > 1e-12) == 4, g


def test_on_a_diagonal_cell_it_is_the_period_gradient():
    """The one place the two spellings must agree, so one checks the other.

    ``grad`` with respect to ``period`` predates this and was already verified, which makes it
    an oracle: on ``diag(Lx, Ly)`` the derivative in ``a1x`` is the derivative in ``period[0]``.
    A point-wise finite difference is *not* usable here -- the normal field is sampled on a
    raster, so the loss carries a staircase and a two-sided difference differentiates the step
    rather than the physics. Measured: FD swung over 1.3e-4 to 2.3e-5 with resolution while AD
    held at 7.31e-6, and integrating AD across a window matched the true loss change to 0.1%.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    Lx, Ly = 700.0, 1100.0
    by_lattice = np.asarray(jax.grad(lambda M: _analytic_loss(jnp, M))(
        jnp.diag(jnp.array([Lx, Ly]))))
    by_period = np.asarray(jax.grad(
        lambda p: _analytic_loss(jnp, jnp.diag(p)))(jnp.array([Lx, Ly])))
    assert by_lattice[0, 0] == pytest.approx(by_period[0], rel=1e-9)
    assert by_lattice[1, 1] == pytest.approx(by_period[1], rel=1e-9)


def test_the_raster_normal_field_says_why_it_cannot():
    """Correct-or-loud: a distance transform needs concrete numbers, so it refuses.

    The discrete half -- which pixel is nearest -- has a zero derivative almost everywhere, so
    nothing is lost by holding it fixed; it is the plumbing that is missing, not the maths.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    eps = np.ones((1, 32, 32))
    u = (np.arange(32) + 0.5) / 32
    U, V = np.meshgrid(u, u)
    eps[0][(U - 0.5) ** 2 + (V - 0.5) ** 2 < 0.1] = 12.25

    def loss(M):
        s = wavebit.rcwa(backend=1, wavelength=900.0, lattice=M, truncation_order=[3, 3],
                         top=1.0, bot=1.0, theta=0.0, phi=0.0, psi=0.0, thickness=[300.0],
                         structure=eps, factorization='nvf')
        return jnp.sum(s.solve().R)

    with pytest.raises(NotImplementedError, match='distance transform'):
        jax.grad(loss)(jnp.array([[400.0, 200.0], [0.0, 346.41]]))
