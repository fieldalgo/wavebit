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
                         dict(type='circle', cx=-50.0, cy=-20.0, lx=140.0, material=5.76)])],
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

    **Which derivative this is.** ``grid`` is called once, at ``A``, and then ``loss(M)`` is
    differentiated -- so the array is held in *lattice* coordinates while the cell moves, and
    the pattern shears along with it. That is not an oversight here and it is not fixable
    there: a raster is numbers on indices, so a fixed array has no choice but to drag its
    pixels when the basis moves. It does mean this path answers a **different question** from
    the analytic one -- *"tilt the cell and shear the pattern with it"* against *"tilt the cell
    only"* -- and the two differ by 3.6x on a plain scene. Named and pinned by
    ``test_the_raster_basis_gradient_is_the_shear_along_one``; see also
    ``docs/oblique-lattice.md``, "Differentiating the cell".

    This test asserts finiteness and four live components, which is a statement about the
    *plumbing*. The size of the gradient is checked by
    ``test_the_basis_gradient_obeys_the_unimodular_chain_rule`` and its meaning by the test
    named above.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    A = np.asarray(wavebit.hexagonal(400.0))
    n = 48
    X, Y = (np.asarray(c) for c in wavebit.grid((n, n), lattice=A))
    eps = np.where(X ** 2 + Y ** 2 < 150.0 ** 2, 12.25, 1.0)

    def loss(M):
        s = wavebit.rcwa(backend=1, wavelength=900.0, lattice=M, truncation_order=[4, 4],
                         top=1.0, bot=1.0, theta=0.1, phi=0.3, psi=0.2, thickness=[300.0],
                         structure=eps[None], factorization='nvf')
        return jnp.sum(s.solve().R)

    g = np.asarray(jax.grad(loss)(jnp.asarray(A)))
    assert np.all(np.isfinite(g)), g
    assert np.count_nonzero(np.abs(g) > 1e-12) == 4, g


# ------------------------------------------------- one lattice, two uses: `grid` and `rcwa`

#: The scene both of the next two tests are sited on: a genuine Cartesian circle in a cell that
#: is square at 90 degrees and strongly skewed at 20, where a raster read on the wrong basis is
#: an ellipse rather than a slightly-wrong circle.
_L, _N, _R, _EPS = 640.0, 64, 100.0, 12.25


def _cell_at(gamma_deg, length=_L):
    """``|a1| = |a2| = length`` with ``gamma`` between them -- one knob, so a derivative is one."""
    g = np.radians(gamma_deg)
    return wavebit.oblique(length, length, g)


def _sum_t(basis, eps, truncation=6):
    s = wavebit.rcwa(backend=0, wavelength=900.0, lattice=basis, truncation_order=[truncation] * 2,
                     top=1.0, bot=1.0, theta=0.0, phi=0.0, psi=0.0, thickness=[300.0],
                     structure=np.asarray(eps)[None], factorization='nvf')
    return float(np.sum(np.asarray(s.solve().T)))


def _aspect_of_the_lit_pixels(eps, basis):
    """How round the *solved* structure is: the lit pixels placed on ``basis``, as a ratio.

    Read off the covariance rather than by fitting a circle, so it needs no centre and no
    threshold. A round feature gives 1; the number is the axis ratio of the ellipse of inertia.
    """
    A = np.asarray(basis)
    t = (np.arange(_N) + 0.5) / _N - 0.5
    U, V = np.meshgrid(t, t)
    X, Y = U * A[0, 0] + V * A[0, 1], U * A[1, 0] + V * A[1, 1]
    lit = np.asarray(eps) > 1.0
    w = np.linalg.eigvalsh(np.cov(np.stack([X[lit], Y[lit]])))
    return float(np.sqrt(w[1] / w[0]))


def test_the_two_uses_of_a_lattice_have_to_be_the_same_one():
    """The cell is needed twice -- to place the pixels and to solve them -- and only sameness saves it.

    ``grid`` returns genuine Cartesian coordinates; the lattice is not in the output, it is the
    *conversion*, because the array holds pixel indices and ``r = (u-0.5)*a1 + (v-0.5)*a2`` is
    the only thing that places them. So the same array means a different structure under a
    different basis, and the failure mode is the two uses diverging. Measured here, on a
    radius-100 Cartesian circle in a 640-unit cell at 64 px::

        grid's cell   solve's cell   aspect   sum(T)
        -----------   ------------   ------   -----
        90 deg        90 deg          1.00    0.157
        90 deg        20 deg          5.67    0.997
        20 deg        90 deg          5.60    0.532
        20 deg        20 deg          1.01    0.288

    Only the diagonal is round -- **the angle is irrelevant and sameness is the whole rule.**
    The top-right row is the dangerous one because it is what a caller who swaps ``lattice=``
    and reuses the array gets: ``sum(T)`` 0.157 -> 0.997, no error, no warning, and energy
    conserved throughout, which is why nothing downstream can be made to catch it.

    :meth:`Lattice.grid` exists for this and only this. It is a nudge, not a guarantee -- the
    free function still takes any ``lattice=`` -- and provenance cannot be attached to the
    array either, because the first ``np.where`` turns the coordinates into permittivity and
    there is nothing left to carry a tag.
    """
    drawn = {g: np.where(np.asarray(_cell_at(g).grid((_N, _N))[0]) ** 2
                         + np.asarray(_cell_at(g).grid((_N, _N))[1]) ** 2 < _R ** 2, _EPS, 1.0)
             for g in (90, 20)}

    aspect = {(g, s): _aspect_of_the_lit_pixels(drawn[g], _cell_at(s))
              for g in (90, 20) for s in (90, 20)}
    assert aspect[(90, 90)] == pytest.approx(1.00, abs=0.02)
    assert aspect[(20, 20)] == pytest.approx(1.01, abs=0.02)
    assert aspect[(90, 20)] == pytest.approx(5.67, rel=0.02)      # an ellipse, not a circle
    assert aspect[(20, 90)] == pytest.approx(5.60, rel=0.02)

    # ... and it is not a small perturbation of the answer either
    agree = _sum_t(_cell_at(90), drawn[90])
    diverge = _sum_t(_cell_at(20), drawn[90])
    assert agree == pytest.approx(0.157, abs=0.01)
    assert diverge == pytest.approx(0.997, abs=0.01)


def test_the_method_and_the_free_function_are_one_call():
    """``A.grid(shape)`` is ``wavebit.grid(shape, lattice=A)``, to the byte, on every spelling."""
    for A in (wavebit.hexagonal(400.0), wavebit.rectangular(700.0, 1100.0),
              wavebit.oblique(400.0, 520.0, np.radians(105.0)), as_lattice(lattice=SHEAR)):
        by_method = [np.asarray(c).tobytes() for c in A.grid((9, 13))]
        by_function = [np.asarray(c).tobytes() for c in wavebit.grid((9, 13), lattice=A)]
        assert by_method == by_function, A

    # the keywords ride through rather than being quietly dropped
    A = wavebit.hexagonal(400.0)
    assert [np.asarray(c).tobytes() for c in A.grid((5, 7), dtype=np.float32)] == \
        [np.asarray(c).tobytes() for c in wavebit.grid((5, 7), lattice=A, dtype=np.float32)]
    with pytest.raises(ValueError, match='backend'):
        A.grid((4, 4), backend=9)
    with pytest.raises(ValueError, match='rows, columns'):
        A.grid((4,))

    # The first parameter is `grid_shape`. Plain `shape` collides with geometry -- this package
    # spells a piece of geometry a shape -- but the word is also what says the pair is
    # rows-first, the way an array's shape is; `sample_x` / `sample_y` are the x-first spelling
    # and mean counts per named axis. So the prefix disambiguates rather than the word changing.
    assert [np.asarray(c).tobytes() for c in A.grid(grid_shape=(9, 13))] == \
        [np.asarray(c).tobytes() for c in wavebit.grid(grid_shape=(9, 13), lattice=A)]
    with pytest.raises(TypeError):
        A.grid(shape=(9, 13))


def test_naming_both_period_and_lattice_is_refused():
    """Two spellings of one cell, not two cells -- and `lattice` used to win in silence.

    ``period`` is ``(|a1|, |a2|)``, the angle already gone, so on a skewed cell the two cannot
    both be honoured. Until 2026-09-01 the basis won and the period was dropped without a word,
    which reads as a lattice effect and is not one.

    **The refusal has to key on the keyword being *named*, not on the values disagreeing**,
    and that is what makes it delicate: three internal paths legitimately hold both and hand
    them on. Each is exercised below, because a refusal that broke any of them would take jit,
    the typed container, or the document format with it.
    """
    A = wavebit.hexagonal(400.0)
    eps = np.where(np.asarray(A.grid((16, 16))[0]) ** 2 < 100.0 ** 2, 5.76, 1.0)[None]
    base = dict(backend=0, structure=eps, thickness=[300.0], wavelength=900.0,
                truncation_order=[2, 2], factorization='nvf')

    with pytest.raises(ValueError, match='period and lattice both given'):
        wavebit.rcwa(**base, period=[700.0, 700.0], lattice=A)
    # ...and not only when they disagree: the values here are exactly `periods_of(A)`
    with pytest.raises(ValueError, match='period and lattice both given'):
        wavebit.rcwa(**base, period=[400.0, 400.0], lattice=A)
    wavebit.rcwa(**base, lattice=A)                       # one or the other is fine
    wavebit.rcwa(**base, period=[400.0, 400.0])

    # 1. the JAX pytree rebuild: `period` is a child, `lattice` is aux data, so `__init__`
    #    sees both on every jitted call. `_tree_unflatten` drops the period, which is a no-op
    #    because assigning the basis overwrites the period slot anyway -- asserted here.
    jax = pytest.importorskip('jax')
    solver = wavebit.rcwa(**{**base, 'backend': 1}, lattice=A)
    leaves, treedef = jax.tree_util.tree_flatten(solver)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert np.allclose(np.asarray(rebuilt.lattice), np.asarray(A))
    assert np.allclose(np.asarray(rebuilt.period), np.asarray(solver.period))

    # 2. the typed container, which holds both by design (`period` is its derived readout)
    from wavebit._core.domain import Stack
    # a plain 2x2, which is what a `Stack` stores -- a `Lattice` object is not JSON-
    # representable, which is its own gap and not this one's business.
    stack = Stack.from_structure(eps, [300.0], lattice=np.asarray(A).tolist())
    kw = stack.as_kwargs()
    assert 'lattice' in kw and 'period' not in kw
    wavebit.rcwa(**kw, backend=0, wavelength=900.0, truncation_order=[2, 2],
                 factorization='nvf')

    # 3. the document format, which *writes* `period` into every saved stack (`to_dict` emits
    #    it unconditionally), so a skewed document carries both and must still load and run.
    from wavebit import Incidence
    from wavebit._core.spec import JobSpec
    spec = JobSpec(stack=stack, incidence=Incidence(wavelength=900.0, tetm='te'))
    reloaded = JobSpec.from_json(spec.to_json())
    assert 'period' in spec.to_dict()['stack']            # the writer still emits it
    assert np.allclose(np.asarray(as_lattice(lattice=reloaded.stack.lattice)), np.asarray(A))
    assert reloaded.key() == spec.key()


def test_a_structure_with_no_cell_is_refused_rather_than_solved_on_the_unit_one():
    """``period=(1., 1.)`` is a real cell; assuming it for a structure nobody sized is not.

    The default is the coherent partner of ``wavelength=1.`` and 35 call sites in this
    repository rely on it -- every one of them building a solver with **no structure**, to
    exercise the spec plumbing. Those still build. What is refused is *solving a structure* on
    a cell nobody chose.

    Measured, and it is why this is a refusal and not a warning: the same hex scene with
    ``lattice=`` forgotten answers **0.989790** against the correct **0.927664**, no error and
    energy conserved. The analytic path was already safe (a shape's lengths are absolute, so
    the omission raises "larger than the period"); the raster was the only silent one, because
    a raster carries no length of its own.
    """
    A = wavebit.hexagonal(400.0)
    X, Y = (np.asarray(c) for c in A.grid((64, 64)))
    eps = np.where(X ** 2 + Y ** 2 < 90.0 ** 2, 12.25, 1.0)[None]
    base = dict(backend=0, thickness=[300.0], wavelength=900.0, truncation_order=[6, 6],
                top=1.0, bot=1.0, theta=0.0, phi=0.0, psi=0.0, factorization='nvf')

    with pytest.raises(ValueError, match='no cell was given'):
        wavebit.rcwa(**base, structure=eps)

    # the door the constructor cannot watch: a structure assigned afterwards
    late = wavebit.rcwa(**base)
    late.structure = eps
    with pytest.raises(ValueError, match='no cell was given'):
        late.solve()

    # a solver with no structure still builds on the default -- the 35 sites
    wavebit.rcwa(**base)
    # ...and naming the cell, by either spelling, is all it takes
    stated = float(np.sum(np.asarray(wavebit.rcwa(**base, structure=eps, lattice=A).solve().T)))
    assert stated == pytest.approx(0.927664, abs=1e-4)

    # what the refusal is standing in front of: the unit cell answers, plausibly and wrongly
    silent = wavebit.rcwa(**base, structure=eps, period=[1.0, 1.0])
    got = float(np.sum(np.asarray(silent.solve().T)))
    assert got == pytest.approx(0.989790, abs=1e-4)
    assert abs(got - stated) > 0.05                       # not a rounding difference


def test_the_raster_basis_gradient_is_the_shear_along_one():
    """Which of the two lattice-derivatives the raster path computes, named by measurement.

    A raster is held in *lattice* coordinates, so a fixed array drags its pixels when the basis
    moves: differentiating through one asks *"tilt the cell and shear the pattern with it"*.
    The analytic path asks *"tilt the cell only"*, because its lengths are absolute Cartesian
    quantities. Both are legitimate questions and the code says which is which nowhere, so this
    test says it.

    **The identification is made against an independent path, not against the raster itself.**
    Shearing the circle along the cell turns it into an ellipse whose semi-axes and angle are
    the singular values and left singular vectors of ``A(gamma) @ A(gamma0)^-1`` -- and *that*
    is expressible analytically, staircase-free. At gamma=75 on the scene above, per degree::

        raster jax.grad                              -3.3806e-02
        analytic FD, shape sheared with the cell     -3.0704e-02     <- 10% away
        analytic FD, shape held fixed                -9.4996e-03     <- 3.6x away

    Verified stable rather than assumed: over ``truncation_order`` 6/8/10/12/14 crossed with
    raster 64/128/256, the raster gradient sits **2.8% to 23.7%** from the sheared derivative
    and **211% to 323%** from the fixed one. It is not stable below that -- at 48 px and
    ``truncation_order=5`` the two candidates cross (-9.72e-03 against -8.95e-03 and
    -3.35e-02), which is truncation error rather than a change of meaning, and is why this test
    is sited at 6/64 and not lower.

    The raster arm is ``jax.grad``, and it agrees with a finite difference over the **frozen**
    array to 0.7% (-3.3806e-02 against -3.3555e-02), so the AD is a faithful derivative of the
    loss it was handed. Redrawing the array at each gamma -- the other question, asked of the
    raster -- does not converge at all: -2.12e-01, -3.25e-01, -7.64e-01 at h = 0.5, 0.25, 0.1
    degrees, because the redraw flips pixels discretely and the quotient differentiates the
    staircase. **So the fixed-shape derivative is not reachable from a raster by any means**,
    which is why this is documented rather than warned about or refused: there is nothing to
    fall back to on this path, and the fallback is the other path.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp

    gamma, h = 75.0, 0.5
    A0 = np.asarray(_cell_at(gamma))
    X, Y = (np.asarray(c) for c in _cell_at(gamma).grid((_N, _N)))
    eps = np.where(X ** 2 + Y ** 2 < _R ** 2, _EPS, 1.0)

    def loss(M):
        s = wavebit.rcwa(backend=1, wavelength=900.0, lattice=M, truncation_order=[6, 6],
                         top=1.0, bot=1.0, theta=0.0, phi=0.0, psi=0.0, thickness=[300.0],
                         structure=jnp.asarray(eps)[None], factorization='nvf')
        return jnp.sum(s.solve().T)

    # d/d gamma, in degrees: the chain rule onto the one knob `_cell_at` turns
    rad = np.radians(gamma)
    dA = np.array([[0.0, -_L * np.sin(rad)], [0.0, _L * np.cos(rad)]]) * np.pi / 180.0
    raster = float(np.sum(np.asarray(jax.grad(loss)(jnp.asarray(A0))) * dA))

    def analytic(gamma_deg, shear):
        A = np.asarray(_cell_at(gamma_deg))
        if shear:
            U, sv, _ = np.linalg.svd(A @ np.linalg.inv(A0))
            shape = dict(type='ellipse', cx=0.0, cy=0.0, lx=2 * _R * sv[0], ly=2 * _R * sv[1],
                         angle=float(np.arctan2(U[1, 0], U[0, 0])), material=_EPS)
        else:
            shape = dict(type='circle', cx=0.0, cy=0.0, lx=2 * _R, material=_EPS)
        s = wavebit.rcwa(backend=0, wavelength=900.0, lattice=A, truncation_order=[6, 6],
                         top=1.0, bot=1.0, theta=0.0, phi=0.0, psi=0.0, thickness=[300.0],
                         structure=[dict(background=1.0, shapes=[shape])])
        return float(np.sum(np.asarray(s.solve().T)))

    def slope(shear):
        return (analytic(gamma + h, shear) - analytic(gamma - h, shear)) / (2 * h)

    sheared, fixed = slope(True), slope(False)
    # a degenerate scene, where the two questions had the same answer, would pass the rest
    assert abs(sheared - fixed) / abs(fixed) > 1.0, (sheared, fixed)
    assert abs(raster - sheared) / abs(sheared) < 0.25, (raster, sheared)
    assert abs(raster - fixed) / abs(fixed) > 1.5, (raster, fixed)


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
    X, Y = (np.asarray(c) for c in wavebit.grid((n, n), lattice=basis))
    return np.where(X ** 2 + Y ** 2 < radius ** 2, 12.25, 1.0)


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


@pytest.mark.parametrize('backend', [1, 2])
def test_a_period_written_as_two_separate_leaves_still_carries_gradient(backend):
    """`period=[a, b]` where a and b are arrays, not a single array of two.

    Rebuilding the pair from such a list is the obvious spelling and it is the one that loses
    the graph -- `torch.tensor([leaf, leaf])` raises outright, and the working version stacks
    instead, casting each element first so nothing is created at single precision.

    Torch had that branch and the other two did not, because Torch is where it was hit. Folding
    the three setters into one gave it to JAX as well, which is correct -- `jnp.stack` and
    `jnp.array` agree on values and on gradients for a list of scalars, measured -- but it was
    a branch no JAX test reached, since every JAX caller in this suite passes `period` as one
    whole `jnp.asarray([...])`. This is that case.
    """
    structure = np.ones((1, 8, 8))
    structure[0, 2:6, 2:6] = 4.0
    cfg = dict(truncation_order=[3, 3], thickness=[150.0], wavelength=900.0, top=1.0, bot=1.0,
               theta=0.2, phi=0.3, psi=0.4)

    if backend == 1:
        jax = pytest.importorskip('jax')
        jnp = pytest.importorskip('jax.numpy')

        def loss(p):
            solver = wavebit.rcwa(backend=1, structure=jnp.asarray(structure),
                                  period=[p, 2.0 * p], **cfg)
            return jnp.sum(jnp.asarray(solver.solve().T))

        p = jnp.asarray(400.0)
        grad = jax.grad(loss)(p)
        assert np.isfinite(float(grad)) and float(grad) != 0.0
        # and the pair really was built from both entries, not from the first twice
        solver = wavebit.rcwa(backend=1, structure=jnp.asarray(structure),
                              period=[p, 2.0 * p], **cfg)
    else:
        torch = pytest.importorskip('torch')
        leaf = torch.tensor(400.0, dtype=torch.float64, requires_grad=True)
        solver = wavebit.rcwa(backend=2, structure=torch.asarray(structure),
                              period=[leaf, 2.0 * leaf], **cfg)
        torch.real(solver.solve().T).sum().backward()
        assert leaf.grad is not None and float(leaf.grad) != 0.0

    got = np.asarray(solver.period, dtype=float) if backend == 1 else \
        solver.period.detach().numpy()
    assert got.tolist() == [400.0, 800.0], 'the second entry was dropped'
    assert str(np.asarray(got).dtype) == str(np.dtype(np.float64))
