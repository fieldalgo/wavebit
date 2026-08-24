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


def test_the_basis_is_two_named_vectors():
    """``u`` and ``v``, the shape and the names fmmax's ``LatticeVectors`` uses.

    ``columns`` is derived, so the ``A[0][1]`` this package is written in still reads ``a2x``.
    Named because these are the two directions everything else is named after -- ``half_width_u``
    counts orders along ``u``, ``n_u`` counts raster samples along it. ``A[:, 0]`` needs the
    reader to know that the columns are the vectors and the rows are x and y; ``A.u`` does not.
    """
    A = as_lattice(lattice=HEX)
    assert np.allclose(A.u, HEX[:, 0]) and np.allclose(A.v, HEX[:, 1])
    assert np.allclose(np.asarray(A.columns), HEX)
    assert A[0][1] == pytest.approx(HEX[0, 1])          # a2x, as it always read
    assert np.array_equal(np.asarray(A), HEX)


def test_the_flag_is_aux_data_and_the_numbers_are_children():
    """A child is traced; aux_data is baked into the treedef and can be branched on.

    Two lattices holding the same numbers but a different flag must have *different* tree
    structures -- that is what stops jit reusing one branch's trace for the other.
    """
    jax = pytest.importorskip('jax')
    A = as_lattice(lattice=HEX)
    leaves, treedef = jax.tree_util.tree_flatten(A)
    assert len(leaves) == 2 and all(np.shape(x) == (2,) for x in leaves)
    back = jax.tree_util.tree_unflatten(treedef, leaves)
    assert back.axis_aligned is False and np.array_equal(np.asarray(back), HEX)
    assert jax.tree_util.tree_structure(A) != jax.tree_util.tree_structure(
        Lattice(A.u, A.v, True))


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


# ----------------------------------------------------------- spelling a cell by its angle

def test_a_cell_can_be_given_as_two_lengths_and_an_angle():
    from wavebit._core.lattice import periods_of

    A = wavebit.oblique(400.0, 520.0, np.radians(105.0))
    assert periods_of(A) == pytest.approx((400.0, 520.0))
    assert A.axis_aligned is False
    # a1 along x, a2 at the stated angle to it
    a1, a2 = np.asarray(A)[:, 0], np.asarray(A)[:, 1]
    cos = float(a1 @ a2 / (np.linalg.norm(a1) * np.linalg.norm(a2)))
    assert np.degrees(np.arccos(cos)) == pytest.approx(105.0)

    assert np.allclose(np.asarray(wavebit.hexagonal(400.0)),
                       np.asarray(wavebit.oblique(400.0, 400.0, 2 * np.pi / 3)))
    assert np.asarray(wavebit.rectangular(700.0, 1100.0)).tobytes() == \
        np.diag([700.0, 1100.0]).tobytes()
    assert np.asarray(wavebit.rectangular(640.0)).tobytes() == np.diag([640.0, 640.0]).tobytes()


def test_a_right_angle_reports_itself_as_one():
    """``axis_aligned`` is read off the basis rather than assumed from which function built it.

    To 1e-16 and not to the bit: ``cos(pi/2)`` is 6.1e-17, so the off-diagonal is ~7e-14 rather
    than 0. That is why :func:`rectangular` exists beside this.
    """
    A = wavebit.oblique(700.0, 1100.0, np.pi / 2)
    assert A.axis_aligned is True
    assert np.allclose(np.asarray(A), np.diag([700.0, 1100.0]), atol=1e-12)
    with pytest.raises(ValueError, match='same line'):
        wavebit.oblique(400.0, 400.0, 0.0)
    with pytest.raises(ValueError, match='same line'):
        wavebit.oblique(400.0, 400.0, np.pi)


def test_the_angle_itself_is_a_design_variable():
    """Build the basis inside the differentiated function and the angle is a parameter.

    Nothing stores an angle -- ``oblique`` is a spelling that returns the same
    :class:`Lattice` -- so this works for the ordinary reason any expression does.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp

    def loss(angle):
        return _analytic_loss(jnp, jnp.asarray(wavebit.oblique(400.0, 400.0, angle).columns))

    g = float(jax.grad(loss)(jnp.asarray(np.pi / 3)))
    assert np.isfinite(g) and abs(g) > 1e-12


def test_the_raster_path_differentiates_the_basis_through_its_index_offsets():
    """The distance transform runs on the host; only the pixel *choice* is held fixed.

    Returning the distance and stopping the gradient there drops the metric's smooth half --
    ``d = hypot(step_j*dj, step_i*di)`` varies smoothly with the steps even when the nearest
    pixel does not change. Measured, that cost ~10% of ``d/d a2x`` on this cell and did not
    shrink with raster resolution, which is what said it was systematic. Rebuilding the
    distance from the returned offsets brings every component under 0.5%.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    A = np.asarray(wavebit.hexagonal(400.0))
    n = 48
    t = (np.arange(n) + 0.5) / n
    U, V = np.meshgrid(t, t)
    X, Y = U * A[0, 0] + V * A[0, 1], U * A[1, 0] + V * A[1, 1]
    cx, cy = 0.5 * (A[0, 0] + A[0, 1]), 0.5 * (A[1, 0] + A[1, 1])
    eps = np.where((X - cx) ** 2 + (Y - cy) ** 2 < 150.0 ** 2, 12.25, 1.0)

    def loss(M):
        s = wavebit.rcwa(backend=1, wavelength=900.0, lattice=M, truncation_order=[4, 4],
                         top=1.0, bot=1.0, theta=0.1, phi=0.3, psi=0.2, thickness=[300.0],
                         structure=eps[None], factorization='nvf')
        return jnp.sum(s.solve().R)

    g = np.asarray(jax.grad(loss)(jnp.asarray(A)))
    assert np.all(np.isfinite(g)), g
    assert np.count_nonzero(np.abs(g) > 1e-12) == 4, g


def test_the_two_hexagonal_spellings_are_one_lattice_and_two_calculations():
    """60 and 120 degrees generate the same lattice; they do not truncate the same way.

    The default is 120, the crystallographic setting, and the reason is not taste: the
    three-fold rotation is *visible in the primitive vectors* there and is not at 60. Measured
    below -- at 120 a 120-degree rotation sends ``a1`` straight to ``a2``; at 60 it sends it to
    ``a2 - a1``. Both are lattice vectors, as they must be for one lattice, but only one of
    them lets you read the symmetry off the basis.

    60 remains the alternative basis whose cell is two equilateral triangles, and is what
    ``docs/oblique-lattice.md`` and the S4 cross-check in it are written on.
    """
    a = 400.0
    A120 = np.asarray(wavebit.hexagonal(a))
    A60 = np.asarray(wavebit.hexagonal(a, np.radians(60.0)))
    assert np.allclose(A60[:, 1] - A60[:, 0], A120[:, 1])          # a2_120 = a2_60 - a1

    # the reason for the default, measured rather than asserted
    t = 2 * np.pi / 3
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    assert np.allclose(rot @ A120[:, 0], A120[:, 1])               # a1 -> a2, exactly
    assert not np.allclose(rot @ A60[:, 0], A60[:, 1])             # a1 -> a2 - a1

    def points(A):
        return {tuple(np.round(m * A[:, 0] + n * A[:, 1], 6))
                for m in range(-3, 4) for n in range(-3, 4)}

    assert len(points(A60) & points(A120)) >= 37                   # one lattice

    def g_set(A, N):
        det = np.linalg.det(A)
        b1 = 2 * np.pi / det * np.array([A[1, 1], -A[0, 1]])
        b2 = 2 * np.pi / det * np.array([-A[1, 0], A[0, 0]])
        return {tuple(np.round(m * b1 + n * b2, 9))
                for m in range(-N, N + 1) for n in range(-N, N + 1)}

    shared = len(g_set(A60, 4) & g_set(A120, 4))
    assert 55 < shared < 81, shared                                # two truncations, not one

    # and the omitted centre lands elsewhere, because the fundamental cell does
    assert 0.5 * (A120[0, 0] + A120[0, 1]) == pytest.approx(100.0)
    assert 0.5 * (A60[0, 0] + A60[0, 1]) == pytest.approx(300.0)


def test_the_solver_always_has_a_basis_to_give():
    """``solver.lattice`` materialises ``diag(period)`` rather than answering ``None``.

    One place to ask what the cell is, one type back. It answered ``None`` for the
    axis-aligned default until 2026-08-23, on the grounds that "a basis has to be concrete"
    and ``period`` may be a tracer -- which stopped being true when :class:`Lattice` split its
    numbers (pytree children, traceable) from ``axis_aligned`` (aux_data, and known from which
    keyword was used rather than from a number).

    That the ``None`` cost something is visible in the tree: ``draw.slice_structure`` reaches
    past this property to ``_geometry_lattice_for`` to get the materialised basis "instead of
    a lookalike".
    """
    eps = np.ones((1, 12, 20))
    eps[0, 3:8, 2:7] = 12.25
    kw = dict(backend=0, wavelength=900.0, truncation_order=dict(u=4, v=1), top=1.0, bot=1.0,
              theta=0.1, phi=0.3, psi=0.2, structure=eps, thickness=[300.0])

    by_period = wavebit.rcwa(period=[700.0, 1100.0], **kw).lattice
    assert isinstance(by_period, Lattice) and by_period.axis_aligned is True
    assert np.allclose(np.asarray(by_period), np.diag([700.0, 1100.0]))

    by_lattice = wavebit.rcwa(lattice=HEX, factorization='nvf', **kw).lattice
    assert isinstance(by_lattice, Lattice) and by_lattice.axis_aligned is False
    assert np.allclose(np.asarray(by_lattice), HEX)

    # a lone period is the square it has always been
    assert np.allclose(np.asarray(wavebit.rcwa(period=[640.0], **kw).lattice),
                       np.diag([640.0, 640.0]))


def test_a_traced_period_still_gives_a_basis():
    """The case the ``None`` existed for. Its shape is a diagonal whether or not it is readable.

    So ``axis_aligned`` is ``True`` without reading a number -- which is exactly what aux_data
    is for, and why materialising is now available where it once was not.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    from wavebit._core.lattice import as_lattice as _as

    seen = []

    def f(p):
        A = _as(period=p)
        seen.append((type(A).__name__, A.axis_aligned))
        return jnp.sum(A.u) + jnp.sum(A.v)

    g = jax.grad(f)(jnp.array([700.0, 1100.0]))
    assert seen == [('Lattice', True)]
    assert np.allclose(np.asarray(g), [1.0, 1.0])        # d(Lx + Ly)/d(period)


def test_a_traced_period_is_still_a_cell():
    """``jax.grad`` over ``period`` solved a *different structure* until 2026-08-24.

    ``_geometry_lattice_for`` returned ``None`` when ``period`` could not be read as concrete --
    which is exactly when it is the variable under differentiation. ``None`` downstream is not
    "unknown": it is the identity metric, i.e. **square pixels**. So on a raster whose pixels
    are not square in real space, the forward solve inside ``value_and_grad`` answered for a cell
    nobody asked for, and the gradient was the exact derivative of that other function.

    Measured on a 200 x 400 cell at 32 x 32: eager ``sum(T)`` 0.964225033797 against
    0.962499142191 returned by ``value_and_grad`` -- and that second number was *bit-identical*
    to forcing ``_geometry_lattice_for`` to ``None``, which is what named the cause.

    The oracle is the definition of ``value_and_grad`` and not a physical claim: the value it
    returns is the value of the function. A square cell cannot show it -- with square pixels the
    identity metric *is* the right one -- which is why this scene is 200 x 400 on a square
    raster.

    ``rel=1e-7`` sits between two measured numbers. The suite runs JAX at its default precision
    (nothing here enables x64), so the two calls differ by 1.1e-09 relative on correct code; the
    defect moved the value by 1.8e-03. Four orders of margin below the one, two above the other.

    Nothing had to be built for the fix. :func:`~wavebit._core.lattice.as_lattice` has taken a
    traced ``period`` since the basis became a design variable -- a diagonal shape is known to be
    axis-aligned without reading a number, which is why that flag is aux_data -- so the call was
    there and simply was not made.
    """
    jax = pytest.importorskip('jax')
    jnp = pytest.importorskip('jax.numpy')

    n = 32
    t = (np.arange(n) + 0.5) / n
    U, V = np.meshgrid(t, t)
    structure = np.where(((U - 0.5) ** 2 + (V - 0.5) ** 2) < 0.09, 5.76, 1.0)[None, ...]

    def loss(period):
        solver = wavebit.rcwa(backend=1, structure=jnp.asarray(structure),
                              truncation_order=[4, 4], thickness=[150.0], wavelength=900.0,
                              top=1.0, bot=1.0, theta=0.2, phi=0.3, psi=0.4,
                              factorization='nvf', period=period)
        return jnp.sum(jnp.asarray(solver.solve().T))

    period = jnp.asarray([200.0, 400.0])                 # non-square cell, square raster
    eager = float(loss(period))
    value, grad = jax.value_and_grad(loss)(period)
    assert eager == pytest.approx(0.5, abs=0.5)          # a degenerate loss would pass the rest
    assert float(value) == pytest.approx(eager, rel=1e-7)
    assert np.all(np.isfinite(np.asarray(grad)))


def _rod_on_cell(basis, n=48, radius=150.0):
    """A disc on the parametric grid of ``basis`` -- the same object whatever the spelling."""
    A = np.asarray(basis)
    t = (np.arange(n) + 0.5) / n
    U, V = np.meshgrid(t, t)
    X, Y = U * A[0, 0] + V * A[0, 1], U * A[1, 0] + V * A[1, 1]
    cx, cy = 0.5 * (A[0, 0] + A[0, 1]), 0.5 * (A[1, 0] + A[1, 1])
    return np.where((X - cx) ** 2 + (Y - cy) ** 2 < radius ** 2, 12.25, 1.0)


@pytest.mark.parametrize('cell,tol', [
    ('oblique', 1e-9),                      # unequal edge lengths: machine precision
    ('hex', 1e-2),                          # equal lengths: genuine EDT ties, see the docstring
], ids=['oblique-400x537', 'hex-400'])
def test_the_basis_gradient_obeys_the_unimodular_chain_rule(cell, tol):
    """``grad L(A @ U) == grad L(A) @ inv(U).T`` -- a chain rule, not a golden and not an FD.

    ``U = [[0, 1], [1, 0]]`` swaps ``a1`` and ``a2``, ``det U = -1``, so ``A @ U`` is the same
    point lattice written the other way round. Writing the raster transposed states the same
    physical object, so the loss is invariant and its gradient must transform by ``inv(U).T``.
    Nothing here compares the code to itself, and nothing here needs a step size.

    **This is the check the shipped basis-gradient test could not make.** That one asserts
    ``isfinite`` and that four components exceed 1e-12, which a gradient of the right *shape*
    and the wrong *size* passes. On its own scene -- ``hexagonal(400)``, 48x48, a 150 nm rod --
    ``d(sum R)/d a1x`` read **-2.70e+08** against the loss's own slope of -8.12e-04 over a 4 nm
    window: eleven orders out, finite, four components nonzero, green. The cause was an
    exact-zero guard in ``normal_vector_field``'s normalisation (fixed 2026-08-24); this law
    reads 1.44e-01 with it live and 5.4e-04 without.

    The two arms have different tolerances and both numbers are measured. Unequal edge lengths
    give 1.2e-13 -- the identity is exact and the arithmetic is all that is left. Equal lengths
    do not: 37 of 2304 pixels sit at a genuine distance-transform tie, where which interface
    pixel is nearest is a choice rather than a fact, and that choice does not transpose with the
    raster. 1e-2 is above that residual and two orders below what the defect produced.

    The primal equality is the loose one deliberately. It checks that the two spellings
    really are one crystal; it is not what this test is for. Its residual moves with whether
    JAX is running in x64 -- 1.2e-13 with, 2.4e-08 without -- and which of those holds
    depends on import order within the process, so a tight bound here passes in a whole-file
    run and fails when the test is selected alone. The chain rule's own residual does not
    move with x64 at all, which is why it is the one carrying the argument.
    """
    jax = pytest.importorskip('jax')
    jnp = pytest.importorskip('jax.numpy')

    if cell == 'hex':
        basis = np.asarray(wavebit.hexagonal(400.0))
    else:
        angle = np.radians(105.0)
        basis = np.column_stack(([400.0, 0.0],
                                 [537.3 * np.cos(angle), 537.3 * np.sin(angle)]))
    structure = _rod_on_cell(basis)
    swap = np.array([[0.0, 1.0], [1.0, 0.0]])

    def grad_of(cell_basis, raster):
        def loss(M):
            solver = wavebit.rcwa(backend=1, wavelength=900.0, lattice=M,
                                  truncation_order=[4, 4], top=1.0, bot=1.0, theta=0.1,
                                  phi=0.3, psi=0.2, thickness=[300.0],
                                  structure=raster[None], factorization='nvf')
            return jnp.sum(solver.solve().R)
        return (float(loss(jnp.asarray(cell_basis))),
                np.asarray(jax.grad(loss)(jnp.asarray(cell_basis))))

    primal, g = grad_of(basis, structure)
    swapped, g_swapped = grad_of(basis @ swap, structure.T)

    scale = float(np.max(np.abs(g)))
    assert scale > 1e-12, scale                       # a dead gradient would pass the rest
    assert swapped == pytest.approx(primal, rel=1e-5)  # the two spellings are one crystal
    want = g @ np.linalg.inv(swap).T
    assert float(np.max(np.abs(g_swapped - want))) / scale < tol, (cell, g, g_swapped)


def test_a_traced_scene_attribute_takes_the_live_arm_and_says_nothing():
    """The tracer arms in the JAX ``period`` and ``thickness`` setters were unreachable.

    Each setter tested ``type(x) is DynamicJaxprTracer`` *after* an arm that already accepts
    ``isinstance(x, jnp.ndarray)`` -- which is ``True`` for a tracer inside ``jax.jit``. So the
    live arm caught every one, and the arm written for tracers ran never.

    It was not inert-looking code. Both copies carried ``print('init period')`` and
    ``jax.debug.print('init period')``, and the copy in the *thickness* setter printed that same
    wrong word: a debug line left in a shipped library, naming the wrong attribute, in a branch
    that could not execute. Nobody saw it because nobody reached it.

    Two things are asserted, because deleting the arm is only safe if both hold: the live arm
    keeps the autograd graph (``jnp.array(tracer, dtype=...)`` does), and it stores at
    ``float_dtype`` rather than passing the tracer through raw as the dead arm did. Neither is
    visible in a gradient's *value* -- a wrong dtype still differentiates -- so the dtype is
    read directly.
    """
    jax = pytest.importorskip('jax')
    jnp = pytest.importorskip('jax.numpy')
    import contextlib
    import io as _io

    structure = np.ones((1, 8, 8))
    structure[0, 2:5, 1:4] = 5.76

    def build(period, thickness):
        return wavebit.rcwa(backend=1, structure=jnp.asarray(structure),
                            truncation_order=[3, 3], thickness=thickness, wavelength=900.0,
                            top=1.0, bot=1.0, theta=0.2, phi=0.3, psi=0.4,
                            factorization='nvf', period=period)

    seen = {}

    def loss(x):
        solver = build([x[0], x[1]], [150.0])
        seen['period_dtype'] = solver.period.dtype
        seen['float_dtype'] = solver.float_dtype
        return jnp.sum(solver.solve().T)

    def loss_thickness(t):
        solver = build([400.0, 400.0], t)
        seen['thickness_dtype'] = solver.thickness.dtype
        return jnp.sum(solver.solve().T)

    chatter = _io.StringIO()
    with contextlib.redirect_stdout(chatter):
        g_period = jax.grad(loss)(jnp.asarray([400.0, 400.0]))
        g_thickness = jax.grad(loss_thickness)(jnp.asarray([150.0]))

    assert chatter.getvalue() == ''                       # the deleted arm was the only printer
    assert np.all(np.isfinite(np.asarray(g_period)))
    assert np.all(np.isfinite(np.asarray(g_thickness)))
    assert np.count_nonzero(np.asarray(g_period)) == 2     # a severed graph gives zeros
    assert seen['period_dtype'] == seen['float_dtype']
    assert seen['thickness_dtype'] == seen['float_dtype']
