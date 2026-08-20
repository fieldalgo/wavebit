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
from wavebit._core.geometry import StaleGeometryError

P = 400.0
WL = 900.0
H = 150.0
N_ROD, N_AIR = 2.4, 1.0
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
    cell = np.full((1, n, n), N_AIR)
    cell[0][inside] = N_ROD
    return cell


def solver(cell, fto=8, lattice=None, **kw):
    kw.setdefault('factorization', 'nvf')
    return wavebit.rcwa(backend=0, ucell=cell, period=[P, P], fto=[fto, fto],
                          thickness=[H], wavelength=WL, n_top=1.0, n_bot=1.0,
                          theta=0.0, phi=0.0, psi=0.0, lattice=lattice, **kw)


def totals(cell, fto=8, lattice=None, **kw):
    res = solver(cell, fto, lattice, **kw).conv_solve()
    return float(np.sum(np.asarray(res.R))), float(np.sum(np.asarray(res.T)))


# ------------------------------------------------------------------- the basis itself

def test_default_lattice_is_none_and_reads_as_axis_aligned():
    """``None`` rather than a materialised ``diag(period)``: ``period`` may be a JAX tracer."""
    s = solver(raster(ORTHO))
    assert s.lattice is None
    assert s.lattice_is_axis_aligned


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
    fto_x, fto_y, lx, ly = 3, 2, 300.0, 700.0
    kx, ky = lat.order_kxy(np, np.diag([lx, ly]), fto_x, fto_y, WL,
                           0.25, -0.5, dtype=np.complex128)
    m = np.arange(-fto_x, fto_x + 1)[None, :]
    n = np.arange(-fto_y, fto_y + 1)[:, None]
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
        kx, ky = s._kxy_2d(WL)
        assert kx.ndim == 1 and ky.ndim == 1
    assert totals(cell) == pytest.approx(totals(cell, lattice=ORTHO), abs=1e-12)


def test_unimodular_shear_is_the_same_physics():
    """``a2' = a2 + a1`` generates the identical lattice, so the answers must agree.

    Not an approximation of each other: same point lattice, same cell area, same reciprocal
    set. The residual is the two rasters staircasing the rod differently, which is why the
    tolerance is 1e-3 rather than machine epsilon and why it tightens with resolution.

    Guarding this specifically because it fails *quietly*. With the metric correction removed
    from the normal field the same comparison drifts to ~8.5e-4 at n=512/fto=14 and ~7x that
    here, while both runs still conserve energy to 1e-15.
    """
    r_o, t_o = totals(raster(ORTHO))
    r_s, t_s = totals(raster(SHEAR), lattice=SHEAR)
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
    sup = np.full((1, ny, nx), N_AIR)
    sup[0][inside] = N_ROD

    _, t_hex = totals(hex_cell, fto=8, lattice=HEX)
    sup_solver = wavebit.rcwa(backend=0, ucell=sup, period=[a, a * sq3], fto=[8, 14],
                                thickness=[H], wavelength=WL, n_top=1.0, n_bot=1.0,
                                theta=0.0, phi=0.0, psi=0.0, factorization='nvf')
    t_sup = float(np.sum(np.asarray(sup_solver.conv_solve().T)))
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
    s = wavebit.rcwa(backend=backend, ucell=cell, period=[P, P], fto=[4, 4],
                       thickness=[H], wavelength=WL, n_top=1.0, n_bot=1.0,
                       theta=0.0, phi=0.0, psi=0.0, factorization='nvf', lattice=HEX)
    res = s.conv_solve()
    total = float(np.sum(np.asarray(res.R))) + float(np.sum(np.asarray(res.T)))
    assert total == pytest.approx(1.0, abs=1e-6)


# ------------------------------------------------------------------- the refusals

def test_li_is_refused_on_a_skewed_cell():
    """Li's nested rule factorizes along the raster axes, which stop being x and y."""
    with pytest.raises(NotImplementedError, match="factorization='nvf'"):
        totals(raster(HEX), lattice=HEX, factorization='li')


def test_the_vector_modeler_is_refused_on_a_skewed_cell():
    with pytest.raises(NotImplementedError, match='block modeler'):
        s = wavebit.rcwa(backend=0, ucell=[[1.0, [['rectangle', 100.0, 200.0, 50.0, 50.0,
                                                     2.4, 0, 0, 0]]]],
                           period=[P, P], fto=[4, 4], thickness=[H], wavelength=WL,
                           n_top=1.0, n_bot=1.0, theta=0.0, phi=0.0, psi=0.0,
                           factorization='nvf', lattice=HEX)
        s.conv_solve()


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
        ucell = [{'n_base': 1.0, 'combine': 'paint', 'shapes': [
            {'type': 'circle', 'cx': P / 2, 'cy': P / 2, 'lx': 160.0, 'n_index': 2.4},
            {'type': 'circle', 'cx': P / 2 + 40.0, 'cy': P / 2, 'lx': 160.0, 'n_index': 3.0}]}]
        s = wavebit.rcwa(backend=0, ucell=ucell, period=[P, P], fto=[6, 6], thickness=[H],
                           wavelength=WL, n_top=1.0, n_bot=1.0, theta=0.0, phi=0.0, psi=0.0,
                           factorization='nvf', lattice=lattice)
        res = s.conv_solve()
        return float(np.sum(np.asarray(res.T)))

    assert painted(None) == pytest.approx(painted(SHEAR), abs=1e-3)


def test_a_painted_overlap_is_not_the_same_as_summing_it():
    """Guards the test above from being vacuous: the two rules must give different answers.

    If painting and summing agreed here, agreeing across the shear would say nothing about the
    resolver having run.
    """
    shapes = [{'type': 'circle', 'cx': P / 2, 'cy': P / 2, 'lx': 160.0, 'n_index': 2.4},
              {'type': 'circle', 'cx': P / 2 + 40.0, 'cy': P / 2, 'lx': 160.0, 'n_index': 3.0}]

    def run(combine):
        s = wavebit.rcwa(backend=0, ucell=[{'n_base': 1.0, 'combine': combine,
                                              'shapes': shapes}],
                           period=[P, P], fto=[6, 6], thickness=[H], wavelength=WL,
                           n_top=1.0, n_bot=1.0, theta=0.0, phi=0.0, psi=0.0,
                           factorization='nvf', lattice=SHEAR)
        return float(np.sum(np.asarray(s.conv_solve().T)))

    assert abs(run('paint') - run('add')) > 1e-3


# --------------------------------------------------- paths a skew turned out not to break

def _analytic_totals(g, lattice=None):
    """One circle, stated in absolute Cartesian lengths — the same disc on the same lattice."""
    ucell = [{'n_base': 1.0, 'shapes': [
        {'type': 'circle', 'cx': P / 2, 'cy': P / 2, 'lx': 2 * R, 'n_index': N_ROD}]}]
    s = wavebit.rcwa(backend=0, ucell=ucell, period=[P, P], fto=[g, g], thickness=[H],
                       wavelength=WL, n_top=1.0, n_bot=1.0, theta=0.0, phi=0.0, psi=0.0,
                       factorization='nvf', lattice=lattice)
    return float(np.sum(np.asarray(s.conv_solve().T)))


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
    s.conv_solve()
    field = np.asarray(s.calculate_field(res_x=6, res_y=6, res_z=4))
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
    on this structure the interface slice sits at 1.1e-1 and does **not** improve from fto 4 to
    12 — which is the truncation shape, not an error. In the homogeneous cap, where every
    surviving order is propagating and both sets hold all of them, the same comparison is
    2.0e-4: a 500x drop across the same field.
    """
    n = 21
    shapes = [{'type': 'circle', 'cx': P / 2, 'cy': P / 2, 'lx': 2 * R, 'n_index': N_ROD}]
    ucell = [{'n_base': 1.0, 'shapes': []}, {'n_base': 1.0, 'shapes': shapes}]

    def field(lattice):
        s = wavebit.rcwa(backend=0, ucell=ucell, period=[P, P], fto=[8, 8],
                           thickness=[600.0, H], wavelength=WL, n_top=1.0, n_bot=1.0,
                           theta=0.0, phi=0.0, psi=0.0, factorization='nvf', lattice=lattice,
                           answers='field')
        s.conv_solve()
        return np.asarray(s.calculate_field(res_x=n, res_y=n, res_z=12))

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
    s.conv_solve()
    assert s.calculate_field(res_x=4, res_y=4, res_z=4) is not None


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
        s2.conv_solve(geometry=frozen)


def test_a_frozen_normal_field_replays_onto_the_same_lattice():
    cell = raster(HEX)
    s = solver(cell, lattice=HEX)
    frozen = s.freeze_geometry()
    replayed = s.conv_solve(geometry=frozen)
    # Not bit-identity: `eig` is not bit-reproducible run to run (see the axis-aligned test).
    assert float(np.sum(np.asarray(replayed.T))) == pytest.approx(
        totals(cell, lattice=HEX)[1], abs=1e-12)




# ------------------------------------------------- against numbers from the literature

# Li, JOSA A 14(10), 2758 (1997), as tabulated by Demesy et al., "Gratings: Theory and Numeric
# Applications" (2012), Table 5.4. phi0 = theta0 = 0, psi0 = 45 deg measured from the cell axes,
# dx = dy = 5*lambda*sqrt(2)/4, h = lambda, eps+ = 2.25, eps- = 1.
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
    ucell = [{'n_base': 1.0, 'shapes': [
        {'type': 'rectangle', 'cx': float(centre[0]), 'cy': float(centre[1]),
         'lx': LI_SIDE, 'ly': LI_SIDE, 'n_index': 1.5}]}]
    s = wavebit.rcwa(backend=0, ucell=ucell, fto=[12, 12], thickness=[1.0], wavelength=1.0,
                       n_top=1.5, n_bot=1.0, theta=0.0, phi=0.0, psi=np.pi / 2,
                       factorization='nvf', lattice=basis)
    res = s.conv_solve().res
    ti = np.asarray(res.T)
    c = (ti.shape[0] // 2, ti.shape[1] // 2)

    worst = max(abs(float(ti[c[0] + n, c[1] + m]) - v) for (n, m), v in LI_FEM.items())
    assert worst < 1e-3, f'worst deviation from the published table: {worst:.2e}'
    assert float(np.asarray(res.R).sum()) == pytest.approx(0.10040, abs=1e-3)
