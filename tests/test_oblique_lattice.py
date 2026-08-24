"""Non-orthogonal (oblique) unit cells: the ``lattice`` basis.

The load-bearing test here is :func:`test_unimodular_shear_is_the_same_physics`. Energy
conservation proves nothing on this path — the NVF assembly is Hermitian-symmetrised whatever
the normal field says, so ``R+T == 1`` holds to 1e-15 even when the cell is being read wrongly
— so every check below compares against a *second representation of the same structure* or a
closed form, never against a self-consistency identity.

The other half of the file pins the refusals. Each unsupported combination runs happily on a
skewed cell and returns a plausible wrong number; the tests assert it raises instead.
"""
import numpy as np
import pytest

import wavebit
from wavebit._core import lattice as lat
from wavebit._core import draw
from wavebit._core.factorization import (normal_vector_field,
                                         to_conv_mat_raster_pn_eps)
from wavebit._core.geometry import StaleGeometryError

P = 400.0
WL = 900.0
H = 150.0
EPS_ROD, EPS_AIR = 5.76, 1.0                       # 2.4**2 and 1.0, as permittivity
R = 90.0

ORTHO = np.column_stack(([P, 0.0], [0.0, P]))
SHEAR = np.column_stack(([P, 0.0], [P, P]))       # a2' = a2 + a1: the *same* point lattice
HEX = np.column_stack(([P, 0.0], [P / 2, P * np.sqrt(3) / 2]))


def raster(basis, n=128, radius=R):
    """A rod on the parametric grid of ``basis``: ``cell[j, i]`` at ``(i+.5)/n*a1+(j+.5)/n*a2``."""
    a1, a2 = np.asarray(basis)[:, 0], np.asarray(basis)[:, 1]
    t = (np.arange(n) + 0.5) / n
    U, V = np.meshgrid(t, t)
    pts = U[..., None] * a1 + V[..., None] * a2
    centre = 0.5 * (a1 + a2)
    inside = np.zeros(pts.shape[:2], dtype=bool)
    for i in (-1, 0, 1):                           # minimum image over the lattice
        for j in (-1, 0, 1):
            s = i * a1 + j * a2
            inside |= (((pts[..., 0] - centre[0] - s[0]) ** 2
                        + (pts[..., 1] - centre[1] - s[1]) ** 2) <= radius ** 2)
    cell = np.full((1, n, n), EPS_AIR)
    cell[0][inside] = EPS_ROD
    return cell


def solver(cell, truncation_order=8, lattice=None, **kw):
    kw.setdefault('factorization', 'nvf')
    return wavebit.rcwa(backend=0, structure=cell, period=[P, P], truncation_order=[truncation_order, truncation_order],
                          thickness=[H], wavelength=WL, top=1.0, bot=1.0,
                          theta=0.0, phi=0.0, psi=0.0, lattice=lattice, **kw)


def totals(cell, truncation_order=8, lattice=None, **kw):
    res = solver(cell, truncation_order, lattice, **kw).solve()
    return float(np.sum(np.asarray(res.R))), float(np.sum(np.asarray(res.T)))


# ------------------------------------------------------------------- the basis itself

def test_the_default_cell_materialises_as_diag_period():
    """A cell given as ``period`` still *is* a basis, and this reports it as one.

    It answered ``None`` here until 2026-08-23, on the grounds that "``period`` may be a JAX
    tracer" and a basis had to be concrete. That stopped being true when ``Lattice`` split its
    numbers from its shape: the numbers are pytree children and trace, while ``axis_aligned``
    -- the part that decides control flow -- is aux_data and follows from *which keyword was
    used*, with no number read. So there is one place to ask what the cell is and one type
    back, and `draw.slice_structure` no longer has to reach past this property for the
    materialised basis it wanted.
    """
    s = solver(raster(ORTHO))
    assert isinstance(s.lattice, lat.Lattice)
    assert s.lattice.axis_aligned and s.lattice_is_axis_aligned
    assert np.allclose(np.asarray(s.lattice), np.diag(np.asarray(s.period)))


def test_assigning_a_basis_reports_the_edge_lengths_as_period():
    s = solver(raster(HEX), lattice=HEX)
    assert not s.lattice_is_axis_aligned
    assert np.allclose(s.period, [P, P])           # both hex basis vectors have length P


def test_a_diagonal_basis_is_still_axis_aligned():
    """Diagonal, not merely orthogonal — a rotated orthogonal basis is *not* separable."""
    assert lat.is_axis_aligned(np.diag([3.0, 7.0]))
    rot = np.array([[1.0, -1.0], [1.0, 1.0]])      # orthogonal columns, rotated 45 degrees
    assert not lat.is_axis_aligned(rot)


@pytest.mark.parametrize('bad, match', [
    (np.zeros((2, 3)), 'columns are the basis vectors'),
    (np.array([[1.0, 2.0], [2.0, 4.0]]), 'parallel'),      # det == 0
])
def test_a_degenerate_basis_is_refused(bad, match):
    with pytest.raises(ValueError, match=match):
        lat.as_lattice(lattice=bad)


def test_the_sample_grid_gets_the_two_axes_the_right_way_round():
    """``sample_grid`` stretches ``u`` -- which indexes Nx -- by ``period[0]``, and only that.

    The one reader of the ``period`` pair for the two non-raster normal-field builders, which is
    why it is worth pinning here rather than only through them: ``analytic`` and ``boundary``
    each carried a copy of this block, the copy in ``boundary`` kept a reversed unpack, and on a
    square cell -- which every test that reached it used -- the two orders are the same number.
    So this cell is 2:1.

    **Both branches, and the transposition rather than the pair, for the second one.** A basis
    makes ``period`` unread, so reversing the pair cannot move the lattice branch and a check
    written that way would pin nothing; the mistake that branch *can* make is the same
    transposition spelled as ``diag(Py, Px)``. Measured on this cell, both move X by 0.9 and Y by
    5/6 -- so the last assertion pair is not inferred from reading the code, it is the number the
    reversed call actually returns. And the two branches have to agree with each other exactly,
    since they describe the same cell. Which one the solver takes was *counted*, not read:
    ``engine`` hands the shape paths ``as_lattice(period=...)`` whenever the period is readable,
    so the basis branch is the ordinary one and ``lattice=None`` survives for direct callers and
    for a ``grad`` with respect to ``period``, where the pair arrives as a tracer.
    """
    Px, Py = 2.0, 1.0
    Ny, Nx = 3, 5
    u = (np.arange(Nx) + 0.5) / Nx
    v = (np.arange(Ny) + 0.5) / Ny
    want_x = u[None, :] * Px + 0.0 * v[:, None]
    want_y = v[:, None] * Py + 0.0 * u[None, :]

    X0, Y0 = lat.sample_grid(np, (Ny, Nx), (Px, Py), np.float64)
    Xb, Yb = lat.sample_grid(np, (Ny, Nx), (Px, Py), np.float64, np.diag([Px, Py]))
    for X, Y in ((X0, Y0), (Xb, Yb)):
        assert X.shape == (Ny, Nx) and Y.shape == (Ny, Nx)
        assert np.array_equal(X, want_x)
        assert np.array_equal(Y, want_y)

    Xr, Yr = lat.sample_grid(np, (Ny, Nx), (Py, Px), np.float64)             # pair reversed
    Xt, Yt = lat.sample_grid(np, (Ny, Nx), (Px, Py), np.float64, np.diag([Py, Px]))  # transposed
    for X, Y in ((Xr, Yr), (Xt, Yt)):
        assert abs(np.abs(X - X0).max() - 0.9) < 1e-15
        assert abs(np.abs(Y - Y0).max() - 5.0 / 6.0) < 1e-15


def test_order_kxy_reduces_to_the_separable_formula():
    """The generalisation must be exactly the old expression on a diagonal basis."""
    half_width_u, half_width_v, lx, ly = 3, 2, 300.0, 700.0
    kx, ky = lat.order_kxy(np, np.diag([lx, ly]), half_width_u, half_width_v, WL,
                           0.25, -0.5, dtype=np.complex128)
    m = np.arange(-half_width_u, half_width_u + 1)[None, :]
    n = np.arange(-half_width_v, half_width_v + 1)[:, None]
    # Not bit-identical: the general form divides by det once and multiplies back, so the
    # rounding differs in the last place. Equal to well under any physical significance.
    assert np.allclose(np.asarray(kx), 0.25 + m * (WL / lx), rtol=1e-14, atol=0)
    assert np.allclose(np.asarray(ky), -0.5 + n * (WL / ly), rtol=1e-14, atol=0)


# ------------------------------------------------------------------- the physics

def test_an_axis_aligned_basis_changes_nothing():
    """Naming the cell explicitly must not select a different path from leaving it implicit.

    Asserted structurally rather than by comparing numbers. The two runs *are* the same
    computation, but ``eig`` / ``inv`` are not bit-reproducible run to run — an identical call
    repeated six times here spreads by 3.5e-18 — so an equality check would be testing LAPACK's
    reduction order, not this. The invariants below are exact and deterministic: both spellings
    hand the geometry steps the same basis, and the k-vectors stay two separable 1-D vectors.

    That first invariant once read ``_geometry_lattice is None`` for both. It is
    ``diag(period)`` for both now — the metric reaches every non-square raster, not only a
    skewed cell (see :meth:`~wavebit._core.engine.CoreSolveMixin._geometry_lattice_for`).
    What the test is *for* is unchanged, and is what it still says.
    """
    cell = raster(ORTHO)
    plain, explicit = solver(cell), solver(cell, lattice=ORTHO)
    assert explicit.lattice_is_axis_aligned
    assert np.array_equal(plain._geometry_lattice_for(None),
                          explicit._geometry_lattice_for(None))
    for s in (plain, explicit):
        kx, ky = s.order_kxy()
        assert kx.ndim == 1 and ky.ndim == 1
    assert totals(cell) == pytest.approx(totals(cell, lattice=ORTHO), abs=1e-12)


def test_the_public_wavevectors_are_the_ones_the_skewed_solve_uses():
    """``order_kxy`` answers for the cell that was given, not for a rectangle.

    Until 2026-08-20 the public accessor was ``get_kx_ky_vector``, which builds only the
    separable 1-D pair ``(kx[m], ky[n])``. That form cannot express a skewed cell at all --
    there each order's components mix both indices -- so it returned the rectangular problem's
    answer, silently, while the solve used the private lattice-aware build. Measured on this
    60-degree cell at ``truncation_order=(2, 2)``: ``ky`` out by **0.805 k0** across a retained band
    of about ``+-2``. Nothing raised, and no document said so.

    The two swapped places. This pins the swap from both sides: the skewed answer must *not*
    be the separable one, and on a rectangular cell it must be exactly the separable one --
    the second half is what says the fix did not move any existing result.
    """
    from wavebit._core.solve_2d import order_grid_kxy

    skewed = solver(raster(HEX), truncation_order=2, lattice=HEX)
    grid = order_grid_kxy(np, *(np.asarray(v) for v in skewed.order_kxy()))
    flat = order_grid_kxy(np, *(np.asarray(v) for v in skewed._separable_kxy(skewed.wavelength)))
    assert grid[0].shape == grid[1].shape == (5, 5)
    apart = max(float(np.max(np.abs(a - b))) for a, b in zip(grid, flat))
    assert apart > 0.5, apart          # the separable pair is a different problem, not a rounding

    aligned = solver(raster(ORTHO), truncation_order=2, lattice=ORTHO)
    for public, separable in zip(aligned.order_kxy(),
                                 aligned._separable_kxy(aligned.wavelength)):
        assert np.array_equal(np.asarray(public), np.asarray(separable))

# ------------------------------------------------- a 1D grating whose grooves are not along y

#: One raster row: eps varies along ``a1`` and is invariant along ``a2``. On any lattice that
#: is a 1D grating -- but its grooves run along ``a2``, and its period is measured along ``b1``,
#: which is perpendicular to ``a2`` rather than parallel to ``a1``. Both facts are invisible in
#: the raster, which is why this shape got routed and factorized as though the cell were square.
_GROOVE_N = 64
_GROOVE = np.where((np.arange(_GROOVE_N) >= _GROOVE_N // 4)
                   & (np.arange(_GROOVE_N) < 3 * _GROOVE_N // 4), 5.76, 1.0)[None, None, :]


def _skewed(deg, rows=1, **kw):
    a2 = P * np.array([np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))])
    basis = np.column_stack((np.array([P, 0.0]), a2))
    return wavebit.rcwa(backend=0, structure=np.tile(_GROOVE, (1, rows, 1)), thickness=[H],
                        wavelength=WL, lattice=basis, factorization='nvf', theta=0.0,
                        truncation_order=(6, 0), **kw)


def _rotated_frame_reference(deg):
    """The same grating written in its own frame, which shares no code with the skewed path.

    Invariant along ``a2`` means the permittivity depends only on ``b1 . r``, so in the frame
    whose x runs along ``b1`` this is an ordinary rectangular 1D grating of period
    ``1/|b1| = P sin(gamma)``, grooves along y, at normal incidence. Its TE and TM decouple
    exactly there (classical mount, ``k_par = 0``), and the scene's ``E || y`` decomposes onto
    that pair as ``(-cos g, sin g)`` — so the per-order power is ``sin^2 g`` of one plus
    ``cos^2 g`` of the other. Nothing in this reference is skewed, and nothing in it is a
    second spelling of the code under test.
    """
    g = np.deg2rad(deg)
    out = {}
    for tetm in ('te', 'tm'):
        solved = wavebit.rcwa(backend=0, structure=_GROOVE, thickness=[H], wavelength=WL,
                              period=[P * np.sin(g)], theta=0.0, phi=None, tetm=tetm,
                              truncation_order=(6, 0)).solve()
        out[tetm] = (np.asarray(solved.R).ravel(), np.asarray(solved.T).ravel())
    w_te, w_tm = np.sin(g) ** 2, np.cos(g) ** 2
    return tuple(w_te * out['te'][i] + w_tm * out['tm'][i] for i in (0, 1))


@pytest.mark.parametrize('rows', [1, 8], ids=['one_row', 'tiled'])
@pytest.mark.parametrize('deg', [90.0, 75.0, 60.0, 45.0])
def test_a_one_row_skewed_cell_is_the_grating_it_draws(deg, rows):
    """It answered the *rectangular* grating, twice over, and conserved energy while doing it.

    Two independent defects, both fixed 2026-08-20, and both needing a skewed cell **and** a
    structure invariant along ``a2`` to show:

    * **Routing.** ``_assign_grating_type`` read "one raster row" as "the 1D operators can
      express this". They cannot on a skewed cell: order ``m`` carries ``m*b1_y`` of ``ky``
      that the separable k layout has no slot for. Off by R 0.132 / T 0.090 at 60 degrees.
    * **The constant normal.** ``normal_vector_field``'s one-cell-axis shortcut returned
      ``x_hat`` — right in *index* space, right in Cartesian space only for a square cell,
      where the general path's ``J^-T`` pullback would have given ``b1_hat``. Off by a further
      0.061 on the same scene, and it survived the routing fix because it is a different bug.

    ``rows=8`` tiles the identical row: the same physical structure, routed to the 2D path
    whatever the lattice, so it is the half of this test that was already right. The two halves
    must land on the same numbers, and both on the reference.

    At ``deg=90`` the cell is square, the routing sends it to the 1D path as before, and the
    reference is the same grating with no rotation at all — that entry is what says neither fix
    moved a rectangular answer.
    """
    R_ref, T_ref = _rotated_frame_reference(deg)
    result = _skewed(deg, rows=rows, psi=np.pi / 2, phi=0.0).solve()
    assert np.allclose(np.asarray(result.R).ravel(), R_ref, atol=1e-12, rtol=0)
    assert np.allclose(np.asarray(result.T).ravel(), T_ref, atol=1e-12, rtol=0)


def test_the_skewed_one_row_answer_is_not_the_rectangular_one():
    """The guard the assertion above cannot make: that the reference *moved*.

    Every number here would also pass if the solver ignored the lattice and the reference
    ignored it too. It does not -- the effective period is ``P sin(gamma)`` -- so pin the size
    of the effect once, at the scene the docstrings above quote.
    """
    square = _rotated_frame_reference(90.0)[1]
    skewed = _rotated_frame_reference(60.0)[1]
    assert np.max(np.abs(square - skewed)) > 5e-2, np.max(np.abs(square - skewed))


def test_tetm_is_refused_on_a_one_row_skewed_cell():
    """``tetm`` names a pure pair this grating does not have, so it cannot be honoured.

    The grooves run along ``a2``; ``tetm`` means E along y. Those coincide only on a square
    cell, so on a skewed one the declaration is false rather than merely unhelpful -- E along
    y is a mixture of the grating's two decoupled polarizations, which is exactly what the
    reference above has to weight by ``sin^2`` and ``cos^2``.

    Refused rather than rerouted: ``psi`` and ``polarization`` carry a mixture and ``tetm``
    does not, so there is nothing for it to mean here.
    """
    with pytest.raises(ValueError, match='grooves run along a2'):
        _skewed(60.0, tetm='te', phi=None).solve()
    # ...and it is still accepted on the square cell, where the claim holds
    assert _skewed(90.0, tetm='te', phi=None).solve().T is not None

def test_unimodular_shear_is_the_same_physics():
    """``a2' = a2 + a1`` generates the identical lattice, so the answers must agree.

    Not an approximation of each other: same point lattice, same cell area, same reciprocal
    set. The residual is the two rasters staircasing the rod differently, which is why the
    tolerance is 1e-3 rather than machine epsilon and why it tightens with resolution.

    Guarding this specifically because it fails *quietly*. With the metric correction removed
    from the normal field the same comparison drifts to ~8.5e-4 at n=512/truncation_order=14 and ~7x that
    here, while both runs still conserve energy to 1e-15.
    """
    r_o, t_o = totals(raster(ORTHO))
    r_s, t_s = totals(raster(SHEAR), lattice=SHEAR)
    assert t_o == pytest.approx(t_s, abs=1e-3)
    assert r_o == pytest.approx(r_s, abs=1e-3)


@pytest.mark.parametrize('method', ['contour', 'spectral', 'smooth'])
def test_the_shear_is_the_same_physics_under_every_normal_method(method):
    """The same statement for the three constructions a skewed cell gained on 2026-08-21.

    Each carries the cell metric its own way -- ``'contour'`` maps its level set and its pixel
    centres to Cartesian coordinates, ``'spectral'`` states both of its frequency-domain
    quantities on the Cartesian wavevector, ``'smooth'`` swaps scipy's two per-axis routines
    for a kd-tree and a frequency-domain kernel -- so "the metric is right" is three separate
    claims and this is where each is made end to end. The test above is the same statement for
    ``'distance'``, and the failure mode is the one it names: quiet, energy-conserving, and
    invisible to every self-consistency check.

    Measured 3.9e-5 to 9.7e-5 across the three, against 1.2e-4 for ``'distance'``; the shared
    1e-3 is the two rasters' staircase, not the method's.

    ``'contour'`` needs contourpy, which is optional, so a minimal install skips its case
    rather than failing it.
    """
    if method == 'contour':
        pytest.importorskip('contourpy',
                            reason="normal_method='contour' needs the optional dependency")
    r_o, t_o = totals(raster(ORTHO), normal_method=method)
    r_s, t_s = totals(raster(SHEAR), lattice=SHEAR, normal_method=method)
    assert t_o == pytest.approx(t_s, abs=1e-3)
    assert r_o == pytest.approx(r_s, abs=1e-3)


def test_a_uniform_layer_is_lattice_independent():
    """No interfaces, so only the k-mapping is under test — and it must not shift the answer.

    A homogeneous slab has one propagating order whatever the cell is; a wrong reciprocal
    basis moves the *other* orders' kz and shows up as leaked efficiency.
    """
    flat = np.full((1, 64, 64), 1.7)
    base = totals(flat)
    for basis in (SHEAR, HEX):
        assert totals(flat, lattice=basis) == pytest.approx(base, abs=1e-9)


def test_hex_lattice_matches_its_rectangular_supercell():
    """The workaround this feature replaces, at low resolution: same array, two cells."""
    a = P
    sq3 = np.sqrt(3.0)
    hex_cell = raster(HEX, n=96, radius=80.0)

    nx = 96
    ny = round(nx * sq3)
    x = (np.arange(nx) + 0.5) / nx * a
    y = (np.arange(ny) + 0.5) / ny * (a * sq3)
    X, Y = np.meshgrid(x, y)
    inside = np.zeros(X.shape, dtype=bool)
    for cx, cy in ((0.25 * a, 0.25 * a * sq3), (0.75 * a, 0.75 * a * sq3)):
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                inside |= (((X - cx - i * a) ** 2
                            + (Y - cy - j * a * sq3) ** 2) <= 80.0 ** 2)
    sup = np.full((1, ny, nx), EPS_AIR)
    sup[0][inside] = EPS_ROD

    _, t_hex = totals(hex_cell, truncation_order=8, lattice=HEX)
    sup_solver = wavebit.rcwa(backend=0, structure=sup, period=[a, a * sq3], truncation_order=[8, 14],
                                thickness=[H], wavelength=WL, top=1.0, bot=1.0,
                                theta=0.0, phi=0.0, psi=0.0, factorization='nvf')
    t_sup = float(np.sum(np.asarray(sup_solver.solve().T)))
    assert t_hex == pytest.approx(t_sup, abs=5e-3)


@pytest.mark.parametrize('backend', [0, 1, 2])
def test_oblique_runs_on_every_backend(backend):
    """The basis is resolved in the shared core, so all three backends get it at once."""
    cell = raster(HEX, n=64)
    if backend == 1:
        pytest.importorskip('jax')
        import jax
        jax.config.update('jax_enable_x64', True)
        cell = __import__('jax.numpy', fromlist=['x']).asarray(cell)
    elif backend == 2:
        cell = pytest.importorskip('torch').tensor(cell, dtype=pytest.importorskip(
            'torch').float64)
    s = wavebit.rcwa(backend=backend, structure=cell, period=[P, P], truncation_order=[4, 4],
                       thickness=[H], wavelength=WL, top=1.0, bot=1.0,
                       theta=0.0, phi=0.0, psi=0.0, factorization='nvf', lattice=HEX)
    res = s.solve()
    total = float(np.sum(np.asarray(res.R))) + float(np.sum(np.asarray(res.T)))
    assert total == pytest.approx(1.0, abs=1e-6)


# ------------------------------------------------------------------- the refusals

def test_li_is_refused_on_a_skewed_cell():
    """Li's nested rule factorizes along the raster axes, which stop being x and y."""
    with pytest.raises(NotImplementedError, match="factorization='nvf'"):
        totals(raster(HEX), lattice=HEX, factorization='li')


def test_the_vector_modeler_is_refused_on_a_skewed_cell():
    with pytest.raises(NotImplementedError, match='block modeler'):
        s = wavebit.rcwa(backend=0, structure=[[1.0, [['rectangle', 100.0, 200.0, 50.0, 50.0,
                                                       5.76, 0, 0, 0]]]],
                           period=[P, P], truncation_order=[4, 4], thickness=[H], wavelength=WL,
                           top=1.0, bot=1.0, theta=0.0, phi=0.0, psi=0.0,
                           factorization='nvf', lattice=HEX)
        s.solve()


def test_a_painted_overlap_survives_a_unimodular_shear():
    """Resolving an overlap into polygons works skewed: the clipper never sees the cell.

    ``resolve_painting`` takes vertex arrays and does Cartesian polygon booleans — no period,
    no wrapping, and none needed: ``exp(-i G.r)`` carries the lattice periodicity, so a polygon
    straddling an edge transforms correctly wherever it sits. What did need the basis was the
    same three things the summed path needed, plus periodic-image enumeration in
    :func:`~wavebit._core.overlap.overlapping_pairs`, which decides whether a layer has an
    overlap at all and shifted by ``(ox*Px, oy*Py)`` rather than along the lattice.
    """
    def painted(lattice):
        structure = [{'background': 1.0, 'combine': 'paint', 'shapes': [
            {'type': 'circle', 'cx': P / 2, 'cy': P / 2, 'lx': 160.0, 'material': 5.76},
            {'type': 'circle', 'cx': P / 2 + 40.0, 'cy': P / 2, 'lx': 160.0, 'material': 9.0}]}]
        s = wavebit.rcwa(backend=0, structure=structure, period=[P, P], truncation_order=[6, 6], thickness=[H],
                           wavelength=WL, top=1.0, bot=1.0, theta=0.0, phi=0.0, psi=0.0,
                           factorization='nvf', lattice=lattice)
        res = s.solve()
        return float(np.sum(np.asarray(res.T)))

    assert painted(None) == pytest.approx(painted(SHEAR), abs=1e-3)


def test_a_painted_overlap_is_not_the_same_as_summing_it():
    """Guards the test above from being vacuous: the two rules must give different answers.

    If painting and summing agreed here, agreeing across the shear would say nothing about the
    resolver having run.
    """
    shapes = [{'type': 'circle', 'cx': P / 2, 'cy': P / 2, 'lx': 160.0, 'material': 5.76},
              {'type': 'circle', 'cx': P / 2 + 40.0, 'cy': P / 2, 'lx': 160.0, 'material': 9.0}]

    def run(combine):
        s = wavebit.rcwa(backend=0, structure=[{'background': 1.0, 'combine': combine,
                                              'shapes': shapes}],
                           period=[P, P], truncation_order=[6, 6], thickness=[H], wavelength=WL,
                           top=1.0, bot=1.0, theta=0.0, phi=0.0, psi=0.0,
                           factorization='nvf', lattice=SHEAR)
        return float(np.sum(np.asarray(s.solve().T)))

    assert abs(run('paint') - run('add')) > 1e-3


# --------------------------------------------------- paths a skew turned out not to break

def _analytic_totals(g, lattice=None):
    """One circle, stated in absolute Cartesian lengths — the same disc on the same lattice."""
    structure = [{'background': 1.0, 'shapes': [
        {'type': 'circle', 'cx': P / 2, 'cy': P / 2, 'lx': 2 * R, 'material': EPS_ROD}]}]
    s = wavebit.rcwa(backend=0, structure=structure, period=[P, P], truncation_order=[g, g], thickness=[H],
                       wavelength=WL, top=1.0, bot=1.0, theta=0.0, phi=0.0, psi=0.0,
                       factorization='nvf', lattice=lattice)
    return float(np.sum(np.asarray(s.solve().T)))


def test_analytic_shapes_survive_a_unimodular_shear():
    """Shape lengths are absolute Cartesian, so nothing about the shape deforms under a skew.

    Only the frequency grid its transform is sampled at stops being rectangular. This converges
    an order tighter than the raster comparison (2e-6 against 1.2e-4) because there is no
    staircase in it at all — the residual is the normal-field sampling grid alone.
    """
    assert _analytic_totals(12) == pytest.approx(_analytic_totals(12, SHEAR), abs=1e-4)


def test_integrated_coefficients_survive_a_unimodular_shear():
    """``fourier2d_integrated`` integrates over the *parametric* cell, where there is no metric to get wrong.

    The Jacobian cancels against the ``1/area`` normalisation, so this path was lattice-general
    before anyone asked it to be. Refused for a while on the assumption it was not.
    """
    r_o, t_o = totals(raster(ORTHO), fourier_coefficients='integrated')
    r_s, t_s = totals(raster(SHEAR), lattice=SHEAR, fourier_coefficients='integrated')
    assert t_o == pytest.approx(t_s, abs=1e-3)
    assert r_o == pytest.approx(r_s, abs=1e-3)


def test_field_reconstruction_runs_on_a_skewed_cell():
    """Samples land on the cell's parallelogram, so the grid is indexed by ``(v, u)``."""
    s = solver(raster(HEX), lattice=HEX, answers='field')
    s.solve()
    field = np.asarray(s.reconstruct_field(sample_x=6, sample_y=6, sample_z=4))
    assert field.shape[-1] == 6 and np.all(np.isfinite(field))


def test_the_reconstructed_field_agrees_across_a_unimodular_shear():
    """Compared where the comparison is meaningful: in a homogeneous region.

    The two bases describe the same lattice, so the field at the same *Cartesian* point must
    agree. Their parametric grids differ, but for a given ``v`` the sheared row visits exactly
    the orthogonal row's x-values, cyclically shifted — ``x = (i+j)/(N-1) * P`` against
    ``i'/(N-1) * P`` — so the alignment below is exact rather than interpolated.

    The catch, and the reason this samples away from the structure: a unimodular change of basis
    keeps the lattice but **retains a different set of orders** — a sheared parallelogram in
    ``(m, n)`` where the other has a square. Converged quantities cannot tell (the efficiencies
    agree to 2e-6), but a field is a finite sum over exactly that set, and near a material
    discontinuity the tails the two sets disagree about decay far too slowly to cancel. Measured
    on this structure the interface slice sits at 1.1e-1 and does **not** improve from truncation_order 4 to
    12 — which is the truncation shape, not an error. In the homogeneous cap, where every
    surviving order is propagating and both sets hold all of them, the same comparison is
    2.0e-4: a 500x drop across the same field.
    """
    n = 21
    shapes = [{'type': 'circle', 'cx': P / 2, 'cy': P / 2, 'lx': 2 * R, 'material': EPS_ROD}]
    structure = [{'background': 1.0, 'shapes': []}, {'background': 1.0, 'shapes': shapes}]

    def field(lattice):
        s = wavebit.rcwa(backend=0, structure=structure, period=[P, P], truncation_order=[8, 8],
                           thickness=[600.0, H], wavelength=WL, top=1.0, bot=1.0,
                           theta=0.0, phi=0.0, psi=0.0, factorization='nvf', lattice=lattice,
                           answers='field')
        s.solve()
        return np.asarray(s.reconstruct_field(sample_x=n, sample_y=n, sample_z=12))

    f_o, f_s = field(None), field(SHEAR)
    aligned = np.empty_like(f_s)
    for j in range(n):
        for i in range(n):
            aligned[..., j, i, :] = f_o[..., j, (i + j) % (n - 1), :]

    scale = np.abs(f_o).max()
    top = np.abs(f_s[:, 0, :, :n - 1, :] - aligned[:, 0, :, :n - 1, :]).max() / scale
    assert top < 1e-3, f'homogeneous region should agree: {top:.2e}'


def test_field_reconstruction_still_works_when_axis_aligned():
    s = solver(raster(ORTHO), answers='field')
    s.solve()
    assert s.reconstruct_field(sample_x=4, sample_y=4, sample_z=4) is not None


def test_a_frozen_normal_field_is_refused_across_a_lattice_change():
    """The basis is an input to the normal field, so a replay onto another cell is stale.

    Without this the replay is accepted in silence: the raster is unchanged, so every other
    tell agrees, and only the direction the interfaces point in space has moved.
    """
    cell = raster(HEX)
    s = solver(cell, lattice=HEX)
    frozen = s.freeze_geometry()
    s2 = solver(cell, lattice=SHEAR)
    with pytest.raises(StaleGeometryError, match='cell basis'):
        s2.solve(geometry=frozen)


def test_a_frozen_normal_field_replays_onto_the_same_lattice():
    cell = raster(HEX)
    s = solver(cell, lattice=HEX)
    frozen = s.freeze_geometry()
    replayed = s.solve(geometry=frozen)
    # Not bit-identity: `eig` is not bit-reproducible run to run (see the axis-aligned test).
    assert float(np.sum(np.asarray(replayed.T))) == pytest.approx(
        totals(cell, lattice=HEX)[1], abs=1e-12)




# ------------------------------------------------- against numbers from the literature

# Li, JOSA A 14(10), 2758 (1997), as tabulated by Demesy et al., "Gratings: Theory and Numeric
# Applications" (2012), Table 5.4. phi0 = theta0 = 0, psi0 = 45 deg measured from the cell axes,
# dx = dy = 5*lambda*sqrt(2)/4, h = lambda, eps+ = 2.25, eps- = 1. Those are permittivities
# in the paper too, and now also in the arguments below -- no conversion either way.
LI_FEM = {(-1, -1): 0.04333, (-1, 0): 0.12845, (-1, 1): 0.06176,
          (0, -1): 0.12838, (0, 0): 0.17577, (0, 1): 0.12839,
          (1, -1): 0.06177, (1, 0): 0.12843, (1, 1): 0.04332}
LI_SIDE = 1.25                                     # block edge; cell edge is LI_SIDE*sqrt(2)


def test_li_1997_checkerboard_matches_the_published_table():
    """The published crossed-grating benchmark, on the oblique cell it is naturally posed in.

    A checkerboard needs two blocks on a square cell and one on its own primitive cell — a
    square rotated 45 degrees, which is *not* axis-aligned — so this exercises the oblique path
    against numbers that did not come from this codebase. The paper's
    ``dx = dy = 5*sqrt(2)/4 = 1.767767`` is that primitive cell's edge.

    ``psi0 = 45 deg`` in the paper is measured from its own cell axes, which are the rotated
    basis, so here it is ``psi = pi/2``. That matters more than it looks: the wrong frame leaves
    T(0,0) almost unchanged (it is polarisation-independent on a C4v cell) while swapping which
    pairs of orders are degenerate, and the worst error grows to 1.9e-2. Hence the whole table
    is compared and not the zeroth order.
    """
    a1 = np.array([LI_SIDE, LI_SIDE])
    a2 = np.array([LI_SIDE, -LI_SIDE])
    basis = np.column_stack((a1, a2))
    assert not lat.is_axis_aligned(basis)                       # the point of the test
    assert np.linalg.norm(a1) == pytest.approx(5 * np.sqrt(2) / 4)

    centre = 0.5 * (a1 + a2)
    structure = [{'background': 1.0, 'shapes': [
        {'type': 'rectangle', 'cx': float(centre[0]), 'cy': float(centre[1]),
         'lx': LI_SIDE, 'ly': LI_SIDE, 'material': 2.25}]}]
    s = wavebit.rcwa(backend=0, structure=structure, truncation_order=[12, 12], thickness=[1.0], wavelength=1.0,
                       top=2.25, bot=1.0, theta=0.0, phi=0.0, psi=np.pi / 2,
                       factorization='nvf', lattice=basis)
    res = s.solve().given
    ti = np.asarray(res.T)
    c = (ti.shape[0] // 2, ti.shape[1] // 2)

    worst = max(abs(float(ti[c[0] + n, c[1] + m]) - v) for (n, m), v in LI_FEM.items())
    assert worst < 1e-3, f'worst deviation from the published table: {worst:.2e}'
    assert float(np.asarray(res.R).sum()) == pytest.approx(0.10040, abs=1e-3)


# --------------------------------------------- a lit order that is not the centre of the window

#: `excitation` moves the retained window onto the lit orders, so its centre `(cx, cy)` stops
#: being `(0, 0)`. Nothing on the rectangular path notices -- `_separable_kxy` builds the whole
#: window in one expression -- but the skewed path assembles `k_inc + m*b1 + n*b2` from an
#: incidence term it asked the backend for, and it asked with the wrong index.
SHIFTED = np.column_stack(([2000.0, 0.0], [600.0, 2200.0]))


def _shifted_solver(excitation, basis=SHIFTED, theta=0.2, phi=0.4):
    cell = np.full((1, 8, 8), EPS_AIR)
    cell[0, 2:5, 1:4] = 12.25
    return wavebit.rcwa(backend=0, structure=cell, thickness=[300.0], top=1.0, bot=1.0,
                        lattice=basis.tolist(), wavelength=500.0, theta=theta, phi=phi,
                        truncation_order=[3, 2], factorization='nvf', excitation=excitation)


def _closed_form_kxy(basis, order_window, wavelength, theta, phi, n_inc=1.0):
    """``k_inc + m*b1 + n*b2`` written out here, from the basis, in the test's own arithmetic."""
    (cx, hx), (cy, hy) = order_window
    B = wavelength * np.linalg.inv(np.asarray(basis)).T      # columns b1, b2, in k/k0 units
    m = np.arange(cx - hx, cx + hx + 1)[None, :]
    n = np.arange(cy - hy, cy + hy + 1)[:, None]
    return (n_inc * np.sin(theta) * np.cos(phi) + B[0, 0] * m + B[0, 1] * n,
            n_inc * np.sin(theta) * np.sin(phi) + B[1, 0] * m + B[1, 1] * n)


@pytest.mark.parametrize('lit', [(0, 0), (2, 0), (0, 1), (1, 1), (2, 1)])
def test_a_shifted_order_window_does_not_count_its_centre_twice(lit):
    """``order_kxy`` against the closed form, for a window that is not centred on zero.

    Until 2026-08-23 the incidence term was read back as ``kx[half_width_u]``, described in the
    docstring as the ``m = 0`` entry. It is the *window centre* entry, so it already carried
    ``cx * lambda / |a1|`` -- the rectangular step, edge lengths with the angle gone -- which
    :func:`wavebit._core.lattice.order_kxy` then added again as ``m*b1``. At ``lit=(2, 0)`` on
    this cell ``kx`` came out ``1.183`` where the basis says ``0.683``.

    Compared against arithmetic written here rather than against the solver's own separable
    build: the two agree by construction on a rectangular cell, which is the case that was
    never wrong.
    """
    s = _shifted_solver({lit: 1.0})
    want_x, want_y = _closed_form_kxy(SHIFTED, s.order_window, 500.0, 0.2, 0.4)
    kx, ky = (np.real(np.asarray(v)) for v in s.order_kxy())
    assert np.allclose(kx, want_x, rtol=0, atol=1e-13), np.max(np.abs(kx - want_x))
    assert np.allclose(ky, want_y, rtol=0, atol=1e-13), np.max(np.abs(ky - want_y))


def test_the_wrong_centre_pushed_a_propagating_order_outside_the_light_line():
    """The loud half of the same bug: the run refused instead of answering.

    ``(2, 0)`` sits at ``kx = 0.683`` against ``n_inc = 1``, comfortably propagating. The extra
    ``cx * lambda / |a1| = 0.5`` put it at ``1.183``, and `_incident_flux` refuses to measure
    efficiencies against an evanescent excitation -- correctly, on the wavevector it was given.

    The quiet half is ``(1, 1)``, which stayed inside the light line and answered:
    ``sum(R) = 0.173069`` before, ``0.059662`` after, energy conserving to 1e-15 either way.
    """
    r = _shifted_solver({(2, 0): 1.0}).solve()
    assert float(np.sum(r.R) + np.sum(r.T)) == pytest.approx(1.0, abs=1e-12)

    quiet = _shifted_solver({(1, 1): 1.0}).solve()
    assert float(np.sum(quiet.R)) == pytest.approx(0.059662278, abs=1e-8)
    assert float(np.sum(quiet.R)) != pytest.approx(0.173069452, abs=1e-6)


def test_the_skewed_branch_never_calls_the_separable_build():
    """Reachability, counted rather than argued.

    ``_separable_kxy`` divides by ``period``, which on a skewed cell is ``(|a1|, |a2|)`` with the
    angle gone -- exactly what ``tests/test_cell_shape_gate.py`` refuses everywhere it can be
    reached. Its exemption there rests on this: the skewed branch of ``order_kxy`` does not call
    it, and the 1D operators that do are routed to only when the cell is axis-aligned. A count,
    because the previous version of that argument was "it is called but its result is unused",
    which is a claim about a line that had already stopped being true.
    """
    calls = []
    for basis, expected in ((SHIFTED, 0), (np.diag([2000.0, 2200.0]), 1)):
        s = _shifted_solver({(1, 1): 1.0}, basis=basis)
        original = type(s)._separable_kxy
        try:
            type(s)._separable_kxy = lambda self, wl, _o=original: (calls.append(1), _o(self, wl))[1]
            calls.clear()
            s.order_kxy()
            assert len(calls) == expected, (basis.tolist(), len(calls))
        finally:
            type(s)._separable_kxy = original


def test_the_reconstructed_field_obeys_bloch_along_both_lattice_vectors():
    """The same fix seen in the field rather than in the efficiencies, and one half of it is blind.

    Bloch's theorem: ``E(r + a) = E(r) * exp(i k_inc . a)`` for either lattice vector, because
    every retained order differs from the incidence by a reciprocal vector ``G`` with
    ``G . a_j = 2*pi*n``. The skewed reconstruction samples the cell's own parallelogram at
    ``linspace(0, 1, n)`` in each of ``u`` and ``v``, so both endpoints are already in the grid
    and the check costs one solve.

    **The ``a1`` half cannot fail here, and that is the point of asserting it anyway.** The bug
    this pins put an extra ``cx * lambda / |a1|`` on every order's ``kx``. In phase across ``a1``
    that is ``k0 * cx * lambda / |a1| * a1x = 2*pi*cx`` exactly -- a whole number of turns, so
    the ``a1`` endpoint agrees to 1.2e-15 with the bug live. It is ``a2`` that separates them,
    because ``a2x / |a1| = 0.3`` is not an integer: 1.418 before, 2.2e-16 after, against a field
    of order 1.

    An efficiency test cannot see this either way -- ``R + T == 1`` to 1e-15 on both sides of the
    fix. The field is where a wrong per-order wavevector is a wrong *shape* rather than a wrong
    total, which is the reason to reconstruct one here at all.
    """
    basis = SHIFTED
    a1, a2 = np.asarray(basis)[:, 0], np.asarray(basis)[:, 1]
    s = _shifted_solver({(1, 1): 1.0}, basis=basis)
    s.answers = 'field'
    s.solve()
    field = np.asarray(s.reconstruct_field(sample_x=9, sample_y=7, sample_z=4))
    assert field.ndim == 5                                  # (incidence, z, v, u, component)
    scale = float(np.max(np.abs(field)))
    assert scale > 1e-6, scale                              # a zero field would pass anything

    k_inc = (2 * np.pi / 500.0) * np.array([np.sin(0.2) * np.cos(0.4),
                                            np.sin(0.2) * np.sin(0.4)])
    for name, a, lo, hi in (('a1', a1, field[:, :, :, 0, :], field[:, :, :, -1, :]),
                            ('a2', a2, field[:, :, 0, :, :], field[:, :, -1, :, :])):
        want = lo * np.exp(1j * float(np.dot(k_inc, a)))
        off = float(np.max(np.abs(hi - want))) / scale
        assert off < 1e-12, (name, off)


# ------------------------------------------------- the plan check that did not get the basis

@pytest.mark.parametrize('lattice,omit_centre', [(None, False), (None, True),
                                                 (HEX, False), (HEX, True)])
def test_a_frozen_painting_plan_replays_on_a_skewed_cell(lattice, omit_centre):
    """``check_plan_still_fits`` was the one sibling in its family not handed the basis.

    ``to_conv_mat_boundary_layer`` forwards ``lattice`` to ``shape_vertices``, ``plan_layer`` and
    ``conv_mats``, and the analytic twin forwards it to ``check_no_plan_still_fits``,
    ``check_layer_combination`` and ``layer_needs_resolving``. The frozen branch's own check did
    not get it, though its signature has had the parameter all along -- so it defaulted an
    omitted centre to ``(Px/2, Py/2)``, the centre of ``diag(|a1|, |a2|)``, while the eager pass
    beside it used the cell's. The two then disagreed about where the shape was, and the check
    reported the plan stale.

    This is ``2d7ca08``'s bug at the one call site that missed the fix, and the parametrisation
    is what separates the two halves: the cell must be skewed *and* a centre omitted. Either
    alone leaves the two spellings the same arithmetic, which is why nothing caught it.

    Only the frozen replay was affected -- the eager solve was right throughout, so an
    energy or convergence check could not have found this. The oracle has to be the eager pass.
    """
    def build():
        a = {'type': 'circle', 'lx': 160.0, 'material': EPS_ROD}
        if not omit_centre:
            a = dict(a, cx=P / 2, cy=P / 2)
        b = {'type': 'circle', 'cx': P / 2 + 40.0, 'cy': P / 2, 'lx': 160.0, 'material': 9.0}
        return wavebit.rcwa(backend=0, period=[P, P], truncation_order=[3, 3], thickness=[H],
                            structure=[{'background': EPS_AIR, 'combine': 'paint',
                                        'shapes': [a, b]}],
                            wavelength=WL, top=1.0, bot=1.0, theta=0.0, phi=0.0, psi=0.0,
                            factorization='nvf', lattice=lattice, normal_grid=64)

    eager = np.asarray(build().solve().T)
    replayer = build()
    replayed = np.asarray(replayer.solve(geometry=replayer.freeze_geometry()).T)
    assert float(np.sum(eager)) > 1e-6                    # an all-zero T would pass anything
    assert replayed.tobytes() == eager.tobytes()          # a replay is the same solve, 0 ulp


# ------------------------------------------------------- the drawing door around the refusal

#: A block instruction: the one modeling path `_reject_unsupported_lattice` refuses outright.
BLOCK = [[EPS_AIR, [['ellipse', P / 2, P / 2, 0.6 * P, 0.6 * P, EPS_ROD, 0.0, 24, 24]]]]


def _block_solver(lattice):
    return wavebit.rcwa(backend=0, structure=BLOCK, period=[P, P], truncation_order=[4, 4],
                        thickness=[H], wavelength=WL, top=1.0, bot=1.0,
                        theta=0.0, phi=0.0, psi=0.0, lattice=lattice)


def test_the_block_modeller_is_refused_at_the_drawing_door_too():
    """``solve()`` refused a skewed block layer; ``slice_structure`` drew one anyway.

    ``_reject_unsupported_lattice`` is reached from ``_check_before_building`` and
    ``freeze_geometry``, and neither is on the path to :func:`~wavebit._core.draw.slice_structure`.
    So the block modeller *drew* a skewed cell -- and the picture was
    ``np.array_equal`` to the same instruction on a square, because the modeller grids on a
    rectangle and nothing inside it reads a basis. ``plot_structure`` then sheared that
    rectangular staircase through the lattice and showed a structure the solve refuses to build.

    Pinned as a pair. The refusal has to reach every door, and it has to leave the axis-aligned
    picture untouched -- a refusal that also changed what a square cell draws would be a worse
    trade than the bug.
    """
    skewed = _block_solver(HEX)
    with pytest.raises(NotImplementedError, match='block modeler'):
        skewed.solve()
    for call in (lambda s: draw.slice_structure(s, 0, n=32),
                 lambda s: draw.slice_structure(s, 0, n=32, band_limited=True),
                 lambda s: draw.plot_structure(s, 0, n=32)):
        with pytest.raises(NotImplementedError, match='block modeler'):
            call(_block_solver(HEX))

    square = _block_solver(None)
    stated = _block_solver(np.diag([P, P]))          # the same cell, named explicitly
    a = np.asarray(draw.slice_structure(square, 0, n=64))
    b = np.asarray(draw.slice_structure(stated, 0, n=64))
    assert np.array_equal(a, b)
    assert len(np.unique(a)) > 1                     # a constant picture would pass anything


# ---------------------------------------------------- the incidence, which was always normal

def _rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def _rod_on(basis, n=64, radius=R):
    """The same physical rod, stated in the cell's own coordinates.

    Built at ``u*a1 + v*a2`` rather than on an ``(x, y)`` mesh, so rotating the basis rotates
    the object rigidly instead of resampling a different one. That is what makes the comparison
    below exact rather than approximate.
    """
    return raster(basis, n=n, radius=radius)


@pytest.mark.parametrize('degrees', [13.0, 47.0, 90.0])
@pytest.mark.parametrize('theta,phi,psi,chi', [
    (0.0, 0.0, 0.0, 0.0),                       # normal: what every oblique test used to be
    (0.35, 0.0, 0.4, 0.0),                      # oblique incidence, linear
    (0.35, 0.9, 0.4, 0.0),                      # with an azimuth of its own
    (0.35, 0.9, 0.0, np.pi / 4),                # circular
    (0.35, 0.9, 0.4, np.pi / 8),                # elliptical, at an orientation
])
def test_the_answer_rotates_with_the_cell(degrees, theta, phi, psi, chi):
    """Turn the cell and the azimuth together and nothing may move.

    Rotating the whole problem about ``z`` is a change of observer. The structure is stated in
    the cell's own coordinates, so ``R @ A`` carries the same object; ``phi`` is measured from
    ``x``, so it takes the same turn; ``psi`` and ``chi`` are measured against the plane of
    incidence, which rotates with everything else, and so are unchanged. Every efficiency must
    come back identical -- the per-order *amplitudes* rotate, the powers do not.

    This is the axis the oblique tests never had. Before 2026-08-24 every one of them ran at
    ``theta=0, phi=0``: ``phi`` and the cell orientation were never varied against each other,
    though ``phi=0`` means "along x" while a skewed ``a1`` is the one thing that need not be.
    ``chi`` appeared nowhere in this file at all.

    Note which rotations are worth asking for. 90 degrees maps ``x`` onto ``y`` and so survives
    an axis-aligned assumption unharmed; it agreed to 1e-14 while the bug in `b7c4645` was live.
    13 and 47 do not, which is why they are here.
    """
    basis = HEX
    turned = _rotation(np.radians(degrees)) @ basis
    common = dict(backend=0, period=[P, P], truncation_order=[5, 5], thickness=[H],
                  wavelength=WL, top=1.0, bot=1.0, theta=theta, psi=psi, chi=chi,
                  factorization='nvf')
    a = wavebit.rcwa(structure=_rod_on(basis), lattice=basis, phi=phi, **common).solve()
    b = wavebit.rcwa(structure=_rod_on(turned), lattice=turned,
                     phi=phi + np.radians(degrees), **common).solve()
    scale = float(np.sum(np.asarray(a.R)) + np.sum(np.asarray(a.T)))
    assert scale == pytest.approx(1.0, abs=1e-12)          # a dead solve would pass the rest
    for name in ('R', 'T'):
        off = float(np.max(np.abs(np.asarray(getattr(a, name)) - np.asarray(getattr(b, name)))))
        assert off < 1e-12, (name, degrees, off)


def _circular_dichroism(basis, truncation, normal_grid=512):
    """``sum(R)`` for one handedness minus the other, on a rod of isotropic dielectric.

    The *shape* path, not a raster. A pixel grid is itself only mirror-symmetric when the cell
    is, so a rastered rod on a shear carries a second asymmetry that answers to resolution
    rather than to truncation -- measured, it falls 1.2e-05 to 5.3e-06 between
    ``truncation_order`` 5 and 11 and then stops, which would hide the effect this is about.
    Stating the disc exactly leaves the retained order set as the only asymmetry left.
    """
    centre = 0.5 * (np.asarray(basis)[:, 0] + np.asarray(basis)[:, 1])
    shapes = [{'background': EPS_AIR,
               'shapes': [{'type': 'circle', 'cx': float(centre[0]), 'cy': float(centre[1]),
                           'lx': 2 * R, 'material': EPS_ROD}]}]
    common = dict(backend=0, structure=shapes, period=[P, P], thickness=[H],
                  truncation_order=[truncation, truncation], wavelength=WL, top=1.0, bot=1.0,
                  theta=0.35, phi=0.0, psi=0.0, factorization='nvf', lattice=basis,
                  normal_grid=normal_grid)
    lcp = wavebit.rcwa(chi=np.pi / 4, **common).solve()
    rcp = wavebit.rcwa(chi=-np.pi / 4, **common).solve()
    return float(np.sum(np.asarray(lcp.R)) - np.sum(np.asarray(rcp.R)))


def test_a_symmetry_oracle_on_a_skewed_cell_is_only_exact_in_the_limit():
    """Circular dichroism is machine-zero on a rectangle and merely *converging* on a shear.

    An array of round rods in an isotropic, non-magnetic dielectric has no circular dichroism:
    mirroring the scene about the plane of incidence maps one handedness onto the other and the
    structure onto a translate of itself, and a rigid in-plane translation cannot change a
    total. So ``sum(R)`` must be handedness-blind, and on an orthogonal cell it is -- 1e-15, at
    any truncation.

    On a skewed cell it is not, and the cause is the *retained order set* rather than the
    physics. The window is a rectangle in ``(m, n)``; the mirror that proves the symmetry sends
    ``a2`` to ``a1 - a2``, so it sends ``(m, n)`` to ``(m + n, -n)`` and carries that rectangle
    off itself. The symmetry is exact in the structure and broken in the truncation, so it comes
    back only as the window grows: measured on ``hexagonal(400)``, 5.4e-06 at
    ``truncation_order=5`` and 1.8e-08 at 17, with the sign changing on the way -- which is
    what truncation error looks like and what a defect does not.

    Pinned so the next reader does not go fix it. A symmetry argument is a valid oracle on this
    path (``docs``: energy conservation is not), but on a skewed cell it holds to the
    *truncation*, not to the machine.
    """
    assert abs(_circular_dichroism(ORTHO, 5)) < 1e-12       # exact, and at low order already
    assert abs(_circular_dichroism(ORTHO, 8)) < 1e-12

    coarse, finer = abs(_circular_dichroism(HEX, 5)), abs(_circular_dichroism(HEX, 11))
    assert coarse > 1e-7, coarse                            # the artifact is real, not noise
    assert finer < coarse / 5.0, (coarse, finer)            # and it is going away with the order


@pytest.mark.parametrize('backend', [0, 1, 2])
def test_a_skewed_cell_answers_in_the_width_it_was_asked_for(backend):
    """``complex_dtype=1`` was silently a no-op once a basis was named.

    The basis is *read* rather than converted -- that is what lets it be a design variable -- so
    ``det`` arrives at whatever width the caller built the cell in, which is float64. Under
    NEP 50 a ``complex64 / float64`` is a ``complex128``, so ``lam`` widened inside
    :func:`~wavebit._core.lattice.order_kxy` and carried the whole return with it. The
    separable build ends in ``.astype(self.complex_dtype)`` and never had the problem, so the
    two paths disagreed exactly when the cell stopped being a rectangle.

    The tell was not the width. It was the **missing warning**: ``UnresolvedBranchWarning``
    fires on the float32 path and the skewed cell was not raising it, because the solve had
    quietly become a float64 one. A user asking for single precision got double -- slower and
    larger, not wrong, but nothing said so.

    Asserted as a *pair* against the rectangle rather than as a literal dtype: the claim is that
    naming the cell does not change the arithmetic width, which is what the bug broke.
    """
    common = dict(backend=backend, structure=_rod_on(ORTHO, n=32), truncation_order=[4, 4],
                  thickness=[H], wavelength=WL, top=1.0, bot=1.0, theta=0.3, phi=0.2, psi=0.1,
                  factorization='nvf', period=[P, P])
    for complex_dtype in (0, 1):
        rect = wavebit.rcwa(complex_dtype=complex_dtype, **common).solve()
        skew = wavebit.rcwa(complex_dtype=complex_dtype, lattice=HEX,
                            **dict(common, structure=_rod_on(HEX, n=32))).solve()
        assert np.asarray(rect.R).dtype == np.asarray(skew.R).dtype, complex_dtype
    assert np.asarray(wavebit.rcwa(complex_dtype=1, lattice=HEX,
                                   **dict(common, structure=_rod_on(HEX, n=32))
                                   ).solve().R).dtype == np.float32


# --------------------------------------------- which Cartesian coordinate names which fraction

def _field_solver(basis, n=24):
    s = wavebit.rcwa(backend=0, structure=raster(basis, n=n, radius=110.0),
                     period=[P, P], lattice=np.asarray(basis).tolist(), truncation_order=[5, 5],
                     thickness=[220.0], wavelength=WL, top=1.0, bot=1.0,
                     theta=0.21, phi=0.58, psi=np.pi / 4, factorization='nvf', answers='field')
    s.solve()
    return s


@pytest.mark.parametrize('basis', [HEX, np.column_stack(([P, 0.0], [-P / 2, P * np.sqrt(3) / 2]))],
                         ids=['hex60', 'hex120'])
def test_a_skewed_cell_can_name_the_coordinate_its_basis_answers(basis):
    """``r = u*a1 + v*a2``, so it is the *other* axis's component that decides.

    ``x = u*a1x + v*a2x``: an ``x`` names a ``u`` only where ``a2x == 0``. ``y = u*a1y +
    v*a2y``: a ``y`` names a ``v`` only where ``a1y == 0``. Until 2026-08-24 the guard in
    :func:`~wavebit._core.fields._inv_fourier` was handed the two the other way round, on a
    premise written above it -- that this path could not meet a skewed cell -- which stopped
    being true when oblique cells landed.

    Both hex spellings put ``a1`` along ``x``, so both were wrong in *both* directions at once:
    an explicit ``sample_x`` was accepted and answered with the nearest ``u`` column, a line
    running across ``|a2x| = 200`` nm of the cell rather than the constant ``x`` asked for --
    about 90% of the field magnitude out where it was read -- while an explicit ``sample_y``,
    which these cells answer exactly, was refused.

    Pinned from both sides and on both spellings, because a guard that is merely *strict* is not
    the same as one that is right: swapping the two arguments back flips both assertions.
    """
    s = _field_solver(basis)
    a2y = float(np.asarray(basis)[1, 1])

    with pytest.raises(NotImplementedError, match='axis-aligned cell'):
        s.reconstruct_field(sample_x=np.array([250.0]), sample_y=3, sample_z=3)

    grid = np.asarray(s.reconstruct_field(sample_x=9, sample_y=3, sample_z=3))
    row = np.asarray(s.reconstruct_field(sample_x=9, sample_y=np.array([0.5 * a2y]),
                                         sample_z=3))
    assert float(np.max(np.abs(grid))) > 1e-6           # a zero field would pass anything
    assert row.shape[-3] == 1
    assert np.array_equal(row, grid[..., 1:2, :, :])    # v = 0.5 is the middle of `linspace`

# ------------------------------------ backside illumination crossed with a skewed cell

_BS_P = 400.0
_BS_WL = 900.0
#: ``a2' = a2 + a1``: det U = +1, so this is the *same* point lattice written another way.
#: A shear rather than a rotation on purpose -- a 90-degree rotation maps x onto y and an
#: axis-aligned assumption survives it unharmed.
_BS_SHEAR = np.column_stack(([_BS_P, 0.0], [_BS_P, _BS_P]))
_BS_ROD = (3.0 + 0.3j) ** 2                # a lossy rod, as permittivity
_BS_SUB = 4.0                              # substrate n = 2, against air on top


def _backside_totals(lattice, side, truncation_order=10):
    """``(sum R, sum T)`` for one rod on the lattice ``lattice`` names, lit from ``side``.

    The rod is a *shape* in absolute Cartesian lengths, so the two bases carry the same disc
    rather than re-discretizing it. A raster would not: re-basing re-staircases it, and that
    term converges only as O(1/n) -- 7.4% relative at n=32, still 2.0% at n=512 -- which is
    far larger than anything this test is trying to see.
    """
    structure = [{'background': 1.0, 'shapes': [
        {'type': 'circle', 'cx': _BS_P / 2, 'cy': _BS_P / 2, 'lx': 180.0,
         'material': _BS_ROD}]}]
    s = wavebit.rcwa(backend=0, structure=structure, period=[_BS_P, _BS_P],
                     truncation_order=[truncation_order, truncation_order],
                     thickness=[300.0], wavelength=_BS_WL, top=1.0, bot=_BS_SUB,
                     theta=0.5, phi=0.6, psi=np.pi / 4, factorization='nvf',
                     lattice=lattice, incidence_side=side)
    res = s.solve()
    return float(np.sum(np.asarray(res.R))), float(np.sum(np.asarray(res.T)))


def test_backside_illumination_survives_a_unimodular_shear():
    """Lit from below, a skewed cell answers what the same lattice answers unskewed.

    The crossing this file and ``tests/test_oblique_lattice.py`` each cover only one half of:
    every backside test runs on a rectangular cell, every skewed-cell test is lit from above,
    so the skewed branch of ``RCWA.order_kxy`` -- the per-order ``(kx, ky)`` a non-orthogonal
    basis needs, since ``period`` on such a cell is ``(|a1|, |a2|)`` with the angle gone -- has
    no test reaching it from ``incidence_side='bottom'``. Make that branch fall back to the
    separable pair when the light comes from below and **nothing in either file notices**:
    ``tests/test_backside_incidence.py`` 101 passed, ``tests/test_oblique_lattice.py`` 47
    passed. On this scene it moves ``sum(T)`` 0.492174 -> 0.177587 and ``sum(R)`` 0.229320 ->
    0.290383, i.e. it silently reads the rectangular problem instead of the skewed one.

    **The oracle is a second representation, not a self-consistency check.** ``a2' = a2 + a1``
    is unimodular, so ``_BS_SHEAR`` generates the identical point lattice with the identical
    cell area, and the rod is stated in the cell's own Cartesian coordinates -- the same disc
    at the same place, carried rather than resampled. Energy would prove nothing: the NVF
    assembly is Hermitian-symmetrised whatever the k-map says, and the rod absorbs, so the
    mutation above just moves the absorbed fraction 0.2785 -> 0.5320 with no identity broken.

    **Why this scene.** Lossy rod, ``top != bot``, oblique incidence -- all three deliberate.
    On the symmetric lossless variant (``top = bot = 1``, real rod, everything else equal) the
    two sides return *bit-identical* arrays: top-vs-bottom ``dR`` and ``dT`` are 0.0 exactly,
    per order, on both bases. A solver that ignored ``incidence_side`` altogether would pass
    such a scene, so the last assertion pins the asymmetry that makes 'bottom' mean something
    here: measured top-vs-bottom 0.232 in ``sum(T)`` and 0.153 in ``sum(R)``.

    **Numbers.** Clean residual at ``truncation_order=10``: 3.8e-4 in ``T``, 6.2e-5 in ``R``
    (worst across truncation 8/10/12/14: 5.5e-4 and 2.3e-4). It does not fall monotonically
    with truncation and it is not meant to: the two bases retain *different order sets* -- a
    square in ``(m, n)`` against a sheared parallelogram -- so this agrees to the truncation,
    not to the machine. The mutation's effect is 3.1e-1 in ``T`` and 6.1e-2 in ``R``. The 5e-3
    tolerance sits 13x above the residual and 12x below the smaller of the two effects.

    ``layer_algorithm`` is the default ``'ETM'`` (the backside path that solves the z-mirrored
    scene); ``'SMM'`` returns the same numbers to 9 digits on this scene.
    """
    r_o, t_o = _backside_totals(None, 'bottom')
    r_s, t_s = _backside_totals(_BS_SHEAR, 'bottom')
    assert t_s == pytest.approx(t_o, abs=5e-3), f'sum(T) from below: {t_o!r} vs {t_s!r}'
    assert r_s == pytest.approx(r_o, abs=5e-3), f'sum(R) from below: {r_o!r} vs {r_s!r}'

    # ...and the scene really is two-sided, so the pair above is a backside statement.
    _, t_top = _backside_totals(None, 'top')
    assert abs(t_top - t_o) > 0.1, f'scene is not two-sided: {t_top!r} vs {t_o!r}'


# ------------------------- the Popov-Neviere (patterned anisotropic) build on a skewed cell

#: ``a2' = a2 + 2*a1``. Unimodular (det U = +1), so the *same point lattice* as ``ORTHO`` --
#: and the shear is **even**, which is what puts the two parametric grids on the same physical
#: points: sample ``(i, j)`` of one is sample ``(i + 2j + 1 mod n, j)`` of the other. The odd
#: ``SHEAR`` above re-discretizes the rod instead, and the staircase term that opens up costs
#: an order of magnitude here -- 5.3e-4 to 6.7e-4 on ``sum(R)`` against the 1.6e-5 to 1.0e-4
#: below.
SHEAR_2A1 = np.column_stack(([P, 0.0], [2 * P, P]))

PN_N = 192
PN_ORDERS = (4, 12)                                # tall on purpose; see the docstring
PN_CENTRE = (0.31 * P, 0.44 * P)
PN_RADIUS = 0.40 * P
PN_EO, PN_EE = 4.0, 7.0                            # ordinary / extraordinary permittivity
PN_AXIS_DEG = 23.0


def _pn_disc(basis, n=PN_N):
    """Indicator of one disc, stated in the cell's own Cartesian coordinates.

    Written out here rather than taken from a helper because the **shape** is what has to
    cross between the two cells, not an array: the disc has a Cartesian centre and a Cartesian
    radius, and each cell samples that same disc on its own parametric grid.
    :func:`wavebit._core.draw._shape_inside` cannot serve -- its periodic image loop is fixed
    at ``(-1, 0, 1)``, which does not reach across a cell sheared by two lattice vectors: it
    draws 0.262 of ``SHEAR_2A1`` filled where the disc covers 0.503.
    """
    a1 = np.asarray(basis, dtype=float)[:, 0]
    a2 = np.asarray(basis, dtype=float)[:, 1]
    t = (np.arange(n) + 0.5) / n
    x = t[None, :] * a1[0] + t[:, None] * a2[0]
    y = t[None, :] * a1[1] + t[:, None] * a2[1]
    inside = np.zeros((n, n), dtype=bool)
    for i in range(-4, 5):                         # minimum image over the lattice
        for j in range(-4, 5):
            ox = PN_CENTRE[0] + i * a1[0] + j * a2[0]
            oy = PN_CENTRE[1] + i * a1[1] + j * a2[1]
            inside |= ((x - ox) ** 2 + (y - oy) ** 2) <= PN_RADIUS ** 2
    return inside


def _pn_tensor(inside):
    """A uniaxial rod in vacuum, optic axis in plane and 23 degrees off x.

    ``eps = eo I + (ee - eo) c c^T`` with ``c = (cos a, sin a, 0)``: real, symmetric, and with
    a **patterned** ``eps_xy``, which is the slot the Popov-Neviere rearrangement exists to
    fill. 23 degrees rather than 0, 45 or 90 -- an axis on x, on y or on the diagonal is
    carried by an axis swap, and a quarter turn maps x onto y unharmed.
    """
    c = np.array([np.cos(np.deg2rad(PN_AXIS_DEG)), np.sin(np.deg2rad(PN_AXIS_DEG)), 0.0])
    rod = PN_EO * np.eye(3) + (PN_EE - PN_EO) * np.outer(c, c)
    eps33 = np.zeros(inside.shape + (3, 3), dtype=complex)
    eps33[..., 0, 0] = eps33[..., 1, 1] = eps33[..., 2, 2] = 1.0
    eps33[inside] = rod
    return eps33


def _pn_totals(basis, method, lattice_blind=False):
    """``(sum(R), sum(T))`` for the rod on ``basis``, through the Popov-Neviere builder.

    The tensor path has no front door -- ``structure`` is a scalar raster -- so this is the
    documented route in: build the five slots and hand them to ``solve_conv``
    (``docs/patterned-anisotropic-factorization.md`` section 9).
    """
    inside = _pn_disc(basis)
    eps33 = _pn_tensor(inside)
    normal_field = None
    if lattice_blind:
        # exactly what the builder computes if it does not pass the basis on
        normal_field = [normal_vector_field(0.5 * (eps33[..., 0, 0] + eps33[..., 1, 1]),
                                            lattice=None, method=method)]
    slots = to_conv_mat_raster_pn_eps(eps33[None], PN_ORDERS[0], PN_ORDERS[1],
                                      normal_field=normal_field, lattice=basis,
                                      normal_method=method)
    solver = wavebit.rcwa(backend=0, structure=np.where(inside, PN_EO, 1.0)[None],
                          lattice=basis, truncation_order=PN_ORDERS, thickness=[H],
                          wavelength=WL, top=1.0, bot=1.0, theta=np.deg2rad(25.0),
                          phi=np.deg2rad(37.0), psi=np.deg2rad(20.0))
    result = solver.solve_conv(*(np.stack([slot[0]]) for slot in slots))
    return float(np.sum(np.asarray(result.R))), float(np.sum(np.asarray(result.T)))


@pytest.mark.parametrize('method', ['contour', 'spectral', 'smooth'])
def test_a_patterned_tensor_is_the_same_physics_under_a_unimodular_shear(method):
    """The Popov-Neviere build has to carry the cell basis into its own normal field.

    The crossing nothing covered: a **patterned anisotropic** layer on a **non-orthogonal**
    cell. :func:`~wavebit._core.factorization.to_conv_mat_raster_pn_eps` reads its normal field
    off the trace of the in-plane block, and that read is the one Cartesian-geometric step in
    the whole build -- an index-space gradient is a Cartesian one only where the pixels are
    square in real space. Drop the ``lattice=`` from that single call and 218 tests stay green,
    the 36 of the two fmmax patterned goldens included, because every scene they are stated on
    is square. Measured on this cell, that drop moves ``epxy`` and ``epyx`` -- the slots the
    rearrangement exists for -- by 0.67 to 1.00 relative, and moves ``epz_i`` by exactly zero,
    which is the tell: ``epz_i`` is the one slot that never reads the normal field.

    **Oracle: the same rod, twice.** ``a2 + 2*a1`` is unimodular, so it generates the identical
    point lattice, and the rod crosses between the two runs as a *shape* rather than as an
    array. Two things about that are asserted before any physics runs: the disc's area comes
    out right (a periodic image loop too short for the shear clips it silently), and the
    sheared raster is the square one re-indexed pixel for pixel -- an even shear puts both
    parametric grids on the same physical points, so the staircase is not merely similar, it is
    the same staircase. Energy conservation would say nothing here: the tensor is Hermitian at
    every pixel, so ``R + T`` comes back 1 to 1e-15 however the cell is read.

    **The retained window is deliberately tall.** The orders kept are a rectangle in
    ``(m, n)`` and the shear carries that rectangle off itself, so the two runs hold different
    physical order sets -- a residual no resolution removes. At ``(4, 4)`` it is 3.5e-3 to
    6.8e-3 and swamps everything; ``(4, 8)`` gives 4.6e-4 to 1.4e-3; ``(4, 12)`` shrinks the
    mismatched band to orders whose amplitude is already negligible, and ``(4, 16)`` does not
    improve on it.

    **Measured**, ``|d sum(R)|`` at ``n = 192``, ``truncation_order = (4, 12)``:

    ==========  ===============  ==========================
    method      clean            normal field on ``None``
    ==========  ===============  ==========================
    contour     8.6e-5           1.46e-2
    spectral    1.0e-4           1.46e-2
    smooth      1.6e-5           1.51e-2
    ==========  ===============  ==========================

    ``1e-3`` therefore sits 10x above the worst clean residual and 15x below the smallest
    effect it has to catch. ``'distance'`` is **not** in the list and cannot be: it separates
    1.37e-2 from 1.47e-2, a factor of 1.07, because its own index-step gradient differs between
    the two cells by more than the basis correction is worth. That is a finding about the
    method rather than about this wiring, and it is the same three constructions
    :func:`test_the_shear_is_the_same_physics_under_every_normal_method` names for the scalar
    path, for the same reason.

    The last assertion is the guard the first cannot make: that the comparison *moved*. Every
    number above would also pass on a scene with no skew in it to see.
    """
    if method == 'contour':
        pytest.importorskip('contourpy',
                            reason="normal_method='contour' needs the optional dependency")

    square, sheared = _pn_disc(ORTHO), _pn_disc(SHEAR_2A1)
    assert abs(square.mean() - np.pi * (PN_RADIUS / P) ** 2) < 2e-3, square.mean()
    for j in range(PN_N):
        assert np.array_equal(sheared[j], np.roll(square[j], -(2 * j + 1))), j

    r_o, t_o = _pn_totals(ORTHO, method)
    r_s, t_s = _pn_totals(SHEAR_2A1, method)
    assert r_s == pytest.approx(r_o, abs=1e-3)
    assert t_s == pytest.approx(t_o, abs=1e-3)

    r_blind, _ = _pn_totals(SHEAR_2A1, method, lattice_blind=True)
    assert abs(r_blind - r_o) > 5e-3, r_blind - r_o
