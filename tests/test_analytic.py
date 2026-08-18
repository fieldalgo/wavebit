"""
Verification tests for the ANALYTIC (staircase-free) modeling path.

Covers: Fourier-coefficient convention vs ``fourier2d_pointwise``, the ``n -> eps`` convention
under complex
indices, analytic formulas vs a fine midpoint DFT, axis-aligned equivalence with the
raster-NVF path, energy conservation, cross-backend agreement, differentiability, and the
convergence advantage over the staircase path.
"""
import warnings

import numpy as np
import pytest

import wavebit
import array_api_compat.numpy as xnp
from wavebit._core.analytic import (
    shape_ft as _shape_ft, conv_matrix_2d as _conv_matrix_2d,
    assemble_eps_coeffs as _assemble_eps_coeffs, normal_vector_field_analytical as _nvf_analytic,
    _sampled_coeffs)
from wavebit._core.backend import UnphysicalResultWarning
from wavebit._core.factorization import fourier2d_pointwise as _pointwise


# The core builders are namespace-generic; bind NumPy for these convention checks.
def raw_dft(cell, fx, fy):
    """The raw DFT build, which is what this convention probe wants -- no pixel correction."""
    return _pointwise(xnp, cell, fx, fy)


def shape_ft(shape, ky, kx, period):
    return _shape_ft(xnp, shape, ky, kx, period)


def conv_matrix_2d(coeff2d, fto_x, fto_y):
    return _conv_matrix_2d(xnp, coeff2d, fto_x, fto_y)


def sampled_coeffs(field, fx, fy):
    """The analytic path's spelling of the sampled transform's coefficients."""
    return _sampled_coeffs(xnp, field, fx, fy)


def assemble_eps_coeffs(shapes, n_base, fto_x, fto_y, period):
    return _assemble_eps_coeffs(xnp, shapes, n_base, fto_x, fto_y, period, xnp.complex128)


def to_conv_mat_smooth(shapes, n_base, fto_x, fto_y, period):
    """Direct-rule (eps, 1/eps) Laurent pair — the NVF builder's inputs."""
    eps_c, inv_c = assemble_eps_coeffs(shapes, n_base, fto_x, fto_y, period)
    return conv_matrix_2d(eps_c, fto_x, fto_y), conv_matrix_2d(inv_c, fto_x, fto_y)


# ----------------------------------------------------------------------
# 1. coefficient-builder conventions: ordering, and n -> eps
# ----------------------------------------------------------------------
def test_conv_matrix_matches_grid_convention():
    """The sampled transform is written twice; the two spellings must stay one transform.

    ``fourier2d_pointwise`` (raster path) and ``_sampled_coeffs`` + ``conv_matrix_2d`` (analytic
    path) are the same Fourier transform of the same kind of data, reached by different routes.
    Nothing forces them to agree, and the NVF projectors go through one on a raster layer and
    the other on a shape layer -- so a change applied to one alone silently splits the two
    modelling paths apart while every per-path test stays green.

    That is not hypothetical. This test used to compare ``fourier2d_pointwise`` against a
    hand-rolled ``np.fft.fft2`` here in the test file, which made the *test* the second
    implementation rather than the analytic path. When the half-pixel cell-centre phase was
    added to the raster side on 2026-08-16 this failed, correctly reporting that the two had
    diverged -- and it was briefly misread as a stale convention probe rather than as the
    finding it was. Comparing the two shipped spellings instead is what makes the failure
    legible: it names the path that was missed.
    """
    rng = np.random.default_rng(0)
    cell = (rng.random((12, 10)) + 1.0).astype(complex)
    fto_x, fto_y = 2, 3

    M_raster = raw_dft(cell, fto_x, fto_y)
    M_analytic = conv_matrix_2d(sampled_coeffs(cell, fto_x, fto_y), fto_x, fto_y)

    gap = np.abs(M_raster - M_analytic).max()
    assert gap < 1e-12, (
        f'the raster and analytic spellings of the sampled transform differ by {gap:.2e}; '
        f'a correction was applied to one of them and not the other')


@pytest.mark.parametrize('where', ['n_base', 'n_index'])
def test_complex_index_is_loss_not_gain(where):
    """``Im(n) > 0`` must come out as ``Im(eps) > 0``, on both the background and the shape.

    The builder conjugated on the way in (``conj(n)**2``), which is the convention
    ``_core/conventions.py`` says Wavebit removed -- so an absorbing index became a gain
    medium, and only on this path: the raster builder (``factorization._layer_eps``) always
    used a raw ``n**2``. Nothing caught it because every case in this file was real, where
    ``conj(n)**2 == n**2``.

    The check is on the DC coefficient, which for a uniform background *is* ``eps``, and for a
    shape covering a known area fraction is the exact ``eps`` average. Sign of the imaginary
    part is the whole claim; the magnitude is there so a stray ``abs`` could not pass it.
    """
    N_METAL = 0.05 + 3.5j                       # a metal: conj() flips it to gain
    eps_metal = N_METAL ** 2                    # (0.0025 - 12.25) + 0.35j -- Im > 0 is loss
    fto_x = fto_y = 2
    period = (1.0, 1.0)
    dc = (2 * fto_y, 2 * fto_x)                 # [ky+2*fto_y, kx+2*fto_x] at ky = kx = 0

    if where == 'n_base':
        eps_c, inv_c = assemble_eps_coeffs([], N_METAL, fto_x, fto_y, period)
        assert complex(eps_c[dc]) == pytest.approx(eps_metal, rel=1e-14)
        assert complex(inv_c[dc]) == pytest.approx(1.0 / eps_metal, rel=1e-14)
    else:
        # A square of side 1/2 covers a quarter of the cell, so the DC term is the exact
        # area-weighted mean and the shape's contribution is separable from the background's.
        shape = dict(type='rectangle', cx=0.5, cy=0.5, lx=0.5, ly=0.5, angle=0.0,
                     n_index=N_METAL)
        eps_c, inv_c = assemble_eps_coeffs([shape], 1.0, fto_x, fto_y, period)
        assert complex(eps_c[dc]) == pytest.approx(1.0 + 0.25 * (eps_metal - 1.0), rel=1e-12)
        assert complex(inv_c[dc]) == pytest.approx(1.0 + 0.25 * (1.0 / eps_metal - 1.0),
                                                   rel=1e-12)

    assert complex(eps_c[dc]).imag > 0          # loss, not gain


# ----------------------------------------------------------------------
# 2. analytic shape transforms match a fine midpoint DFT
# ----------------------------------------------------------------------
@pytest.mark.parametrize("shape", [
    dict(type='rectangle', cx=0.55, cy=0.45, lx=0.5, ly=0.3, angle=0.6),
    dict(type='circle', cx=0.5, cy=0.5, lx=0.6, angle=0.0),
    dict(type='ellipse', cx=0.55, cy=0.45, lx=0.7, ly=0.4, angle=0.9),
])
def test_shape_ft_matches_fine_dft(shape):
    period = (1.0, 1.0)
    N = 2400
    idx = np.arange(N)
    x = (idx + 0.5) / N
    X, Y = np.meshgrid(x, x)
    dx = X - shape['cx']
    dy = Y - shape['cy']
    ct, st = np.cos(shape['angle']), np.sin(shape['angle'])
    xr = dx * ct + dy * st
    yr = -dx * st + dy * ct
    if shape['type'] == 'rectangle':
        ind = ((np.abs(xr) <= shape['lx'] / 2) & (np.abs(yr) <= shape['ly'] / 2)).astype(float)
    elif shape['type'] == 'circle':
        ind = ((xr ** 2 + yr ** 2) <= (shape['lx'] / 2) ** 2).astype(float)
    else:
        ind = (((xr / (shape['lx'] / 2)) ** 2 + (yr / (shape['ly'] / 2)) ** 2) <= 1).astype(float)

    for (ky, kx) in [(0, 0), (1, 0), (0, 2), (3, -2), (-4, 1)]:
        a = shape_ft(shape, np.array(ky), np.array(kx), period)
        phase = np.exp(-1j * 2 * np.pi * (ky * (idx[:, None] + 0.5) / N + kx * (idx[None, :] + 0.5) / N))
        m = (ind * phase).sum() / (N * N)
        assert abs(complex(a) - m) < 5e-5


# ----------------------------------------------------------------------
# 3. axis-aligned analytic == raster-NVF (same validated machinery)
# ----------------------------------------------------------------------
def test_axis_aligned_matches_raster_nvf():
    """The analytic rectangle and the same rectangle rastered give the same physics.

    Two things this pins that the earlier version did not. The period is not 1, so the shape
    parameters have to be *absolute* — they are lengths in the same units as ``period``, not
    fractions of it — and passing fractions here would solve a different, larger rod and fail.
    And it checks TM at oblique incidence over the summed orders, not TE at normal incidence on
    the zeroth order; the normal component of E dominates TM, so that is where a wrong normal
    field or a wrong shape would show first.

    The rod edges land exactly on pixel boundaries (``L * N / 2`` integral), so the raster is
    not an approximation of the rectangle -- it is the rectangle. What is left is the
    factorization difference alone, and it does not move with N.
    """
    P, NR, L, N = 0.77, 3.5, 0.5, 128
    cfg = dict(n_top=1.0, n_bot=1.5, theta=0.35, phi=0.4, period=(P, P),
               wavelength=0.777, thickness=(0.25,), fto=(8, 8))

    xx, yy = ((np.mgrid[0:N, 0:N] + 0.5) / N)[::-1]
    ras = np.where((np.abs(xx - 0.5) < L / 2) & (np.abs(yy - 0.5) < L / 2), NR, 1.0)[None]
    analytic = [dict(n_base=1.0, shapes=[dict(type='rectangle', cx=0.5 * P, cy=0.5 * P,
                                              lx=L * P, ly=L * P, angle=0.0, n_index=NR)])]

    def total(ucell, psi, **kw):
        res = wavebit.rcwa(backend=0, ucell=ucell, psi=psi, **cfg, **kw).conv_solve()
        return float(np.sum(np.asarray(res.R)))

    for psi in (np.pi / 2, 0.0):                       # TE, then TM
        want = total(analytic, psi)
        assert abs(total(ras, psi, factorization='nvf') - want) < 1e-3  # raster NVF
        assert abs(total(ras, psi) - want) < 2e-3                       # raster, sampled


def test_the_analytic_normal_field_gets_the_two_axes_the_right_way_round():
    """A disc's normal is radial in REAL space -- true only if the sample cell is not transposed.

    ``period[0]`` is Px everywhere in this package (:func:`shape_ft`,
    :func:`~wavebit._core.analytic.shape_geometry`, ``polygon.polygon_ft``,
    ``boundary.normal_projector_coeffs``), so the sample grid has to stretch ``u`` -- which
    indexes Nx -- by ``period[0]``. ``normal_vector_field_analytical`` read the pair backwards
    until 2026-08-11 and its one production caller reversed it back, so the path was right and
    nothing in the suite constrained the convention: **a square cell hides the swap completely**,
    and every existing caller here and in ``test_nvf_vector``,
    ``test_nvf_zero_norm_gradients`` and ``test_shape_position_gradients`` uses one. So this
    cell is 2:1, and a disc is the shape that says which axis is which, because its correct
    normal is the radial direction and nothing else.

    Unlike the polygon path's distance gradient, this is a closed form -- for a circle
    ``g = 2*(dx, dy)/a**2``, exactly parallel to the radius at every sample -- so there is no
    polygonisation angle to allow for and the tolerance is machine epsilon rather than the 1e-3
    its counterpart in ``test_boundary`` needs. The centre pixel is the one exclusion: ``g``
    vanishes there, :func:`~wavebit._core.analytic._hypot_safe` returns ``(0, 0)`` by design,
    and the grid being odd on both axes lands a sample exactly on it. A swapped pair is not a
    small effect: it takes the worst deviation to 1.0, i.e. somewhere in the cell the field
    comes out square to the radius instead of along it.

    **Both branches are checked, and the second one is the ordinary one.** ``lattice=None``
    builds the sample grid from Px / Py directly (in
    :func:`~wavebit._core.lattice.sample_grid`, since 2026-08-11 -- one reader for both shape
    paths), and a basis makes that read unreachable. Counting the calls says how much a check
    written only that way would pin: ``engine`` hands the shape paths ``as_lattice(period=...)``
    whenever the period is *readable*, so what is left for ``lattice=None`` is direct callers
    and a ``grad`` with respect to ``period``, where the pair arrives as a tracer with no primal
    to read. Rare, not dead -- and not the path this test is here to protect.

    But the pair is read *outside* the branch as well, at :func:`shape_geometry`, which is where
    a shape that does not name ``cx`` / ``cy`` gets its centre -- so the natural way to write a
    centred disc stays sensitive with a basis in hand. Measured on this cell, reversing the pair
    moves the field by 1.96 with an explicit centre and no lattice, and by 1.7 with the centre
    defaulted and a lattice; the remaining combination, explicit centre *and* a lattice, moves
    by exactly 0. That last cell is why the defaulted case is here: on its own the first
    assertion would be pinning the rare branch only.
    """
    Px, Py, r = 2.0, 1.0, 0.3
    Ny, Nx = 129, 257
    shape = dict(type='circle', cx=Px / 2.0, cy=Py / 2.0, lx=2 * r, n_index=2.0)
    nx, ny = _nvf_analytic(xnp, (Ny, Nx), [shape], (Px, Py), xnp.float64)

    X, Y = np.meshgrid((np.arange(Nx) + 0.5) / Nx * Px, (np.arange(Ny) + 0.5) / Ny * Py)
    dx, dy = X - Px / 2.0, Y - Py / 2.0
    rad = np.hypot(dx, dy)

    def radial_alignment(fx, fy):
        return (np.asarray(fx) * dx + np.asarray(fy) * dy) / np.where(rad > 0, rad, 1.0)

    assert np.abs(np.abs(radial_alignment(nx, ny)[rad > 0]) - 1.0).max() < 1e-12

    # And the same statement read off one pixel, exactly, where a swap is plainest. Both counts
    # are odd, so the disc's centre is a sample and row 64 sits *on* the centre line: due east
    # of it the normal is x_hat with nothing else in it. On the transposed cell that row is not
    # the centre line and this pixel is not even due east of the disc.
    i, j = Ny // 2, int((Px / 2.0 + 1.3 * r) / Px * Nx)
    assert abs(abs(float(nx[i, j])) - 1.0) < 1e-14
    assert abs(float(ny[i, j])) < 1e-14

    # The branch the solver actually takes: a basis in hand, and the centre left to default out
    # of Px / Py. Same disc, so the same radial statement holds and the same tolerance applies.
    centred = dict(type='circle', lx=2 * r, n_index=2.0)
    nx_l, ny_l = _nvf_analytic(xnp, (Ny, Nx), [centred], (Px, Py), xnp.float64,
                               ((Px, 0.0), (0.0, Py)))
    assert np.abs(np.abs(radial_alignment(nx_l, ny_l)[rad > 0]) - 1.0).max() < 1e-12


@pytest.mark.parametrize('psi', [np.pi / 2, 0.0], ids=['te', 'tm'])
def test_metal_rod_absorbs_end_to_end(psi):
    """A metal rod through the whole solve must absorb, not amplify.

    The coefficient test above pins ``n -> eps``; this pins that nothing downstream undoes it.
    ``conj(n)**2`` turned this rod into a gain medium, and a passive stack that returns
    ``R + T > 1`` is exactly what ``check_physical`` exists to catch — so the warning being
    absent is as much the assertion as the number is.

    Why this rather than a comparison against the same rod rastered: the two NVF schemes do not
    agree closely for a metal. Their gap here is ~3e-3 on ``sum(T)`` at ``fto=8`` and does
    not shrink when the raster is refined (128 -> 256 moves it by 1%), because they are
    different factorizations that meet only as ``fto -> inf``, and ``|eps|`` this large converges
    slowly. A tolerance loose enough to pass would be looser than the effect being guarded.
    The tight cross-path statement lives in ``test_lossy_media`` instead, on uniform layers
    where an absolute oracle exists.
    """
    P, L = 0.77, 0.5
    NR = 0.05 + 3.5j                                   # metal rod: conj() would make it gain
    ucell = [dict(n_base=1.0, shapes=[dict(type='rectangle', cx=0.5 * P, cy=0.5 * P,
                                           lx=L * P, ly=L * P, angle=0.0, n_index=NR)])]

    with warnings.catch_warnings():
        warnings.simplefilter('error', UnphysicalResultWarning)
        res = wavebit.rcwa(backend=0, ucell=ucell, psi=psi, n_top=1.0, n_bot=1.5,
                             theta=0.35, phi=0.4, period=(P, P), wavelength=0.777,
                             thickness=(0.25,), fto=(8, 8)).conv_solve()

    total = float(np.sum(np.asarray(res.R)).real + np.sum(np.asarray(res.T)).real)
    assert 0.5 < total < 0.95, total       # absorbing, and absorbing by a visible margin


def test_analytic_shape_parameters_are_absolute_lengths():
    """``cx`` / ``lx`` are lengths, not fractions of the period.

    Nothing else says so: the defaults (``Px / 2``) are the only statement in the code, and
    every other test here uses ``period = 1`` where the two readings coincide. Reading them as
    fractions builds a rod ``1/period`` times too big, which stays silent -- it just solves a
    different structure.
    """
    P, NR, L, N = 0.77, 3.5, 0.5, 128
    cfg = dict(n_top=1.0, n_bot=1.5, theta=0.35, phi=0.0, period=(P, P),
               wavelength=0.777, thickness=(0.25,), fto=(6, 6), psi=0.0)

    xx, yy = ((np.mgrid[0:N, 0:N] + 0.5) / N)[::-1]
    ras = np.where((np.abs(xx - 0.5) < L / 2) & (np.abs(yy - 0.5) < L / 2), NR, 1.0)[None]

    def shaped(lx, cx):
        return [dict(n_base=1.0, shapes=[dict(type='rectangle', cx=cx, cy=cx, lx=lx, ly=lx,
                                              angle=0.0, n_index=NR)])]

    def total(ucell, **kw):
        res = wavebit.rcwa(backend=0, ucell=ucell, **cfg, **kw).conv_solve()
        return float(np.sum(np.asarray(res.R)))

    # The two readings separate by ~90x, so the threshold is not load-bearing; it is set
    # above the analytic-vs-raster gap and far below the misreading. That gap widened from
    # 1e-3 when the NVF normal field stopped being taken on an over-refined grid (2026-08-07,
    # see docs/nvf-upsampling.md) -- a raster reference with sampled coefficients sits at 1.7e-3 on this structure
    # too, so it is the reference that moved, not the shape parameters.
    reference = total(ras, factorization='nvf')
    assert abs(total(shaped(L * P, 0.5 * P)) - reference) < 5e-3        # absolute: agrees
    assert abs(total(shaped(L, 0.5)) - reference) > 1e-2               # fractions: does not


# ----------------------------------------------------------------------
# 4. energy conservation for a lossless tilted/curved multi-shape stack
# ----------------------------------------------------------------------
def test_energy_conservation_lossless():
    cfg = dict(n_top=1.0, n_bot=1.0, theta=np.deg2rad(5), phi=np.deg2rad(20),
               period=(1.0, 1.2), wavelength=1.5, thickness=(0.3, 0.2), psi=0.35 * np.pi)
    ucell = [
        dict(n_base=1.0, shapes=[dict(type='ellipse', cx=0.5, cy=0.6, lx=0.5, ly=0.3,
                                      angle=np.deg2rad(40), n_index=2.4)]),
        dict(n_base=1.5, shapes=[dict(type='rectangle', cx=0.4, cy=0.5, lx=0.4, ly=0.5,
                                      angle=np.deg2rad(25), n_index=1.0),
                                 dict(type='circle', cx=0.8, cy=0.9, lx=0.25,
                                      n_index=3.0)]),
    ]
    for fto in (5, 8):
        res = wavebit.rcwa(backend=0, ucell=ucell, fto=(fto, fto), **cfg).conv_solve()
        tot = np.array(res.R).real.sum() + np.array(res.T).real.sum()
        assert abs(tot - 1.0) < 1e-9


# ----------------------------------------------------------------------
# 5. cross-backend agreement (numpy == jax tight, numpy == torch loose)
# ----------------------------------------------------------------------
def test_cross_backend_agreement():
    cfg = dict(n_top=1.0, n_bot=1.0, theta=np.deg2rad(5), phi=np.deg2rad(20),
               period=(1.0, 1.2), wavelength=1.5, thickness=(0.3,), psi=0.35 * np.pi)
    ucell = [dict(n_base=1.0, shapes=[dict(type='ellipse', cx=0.5, cy=0.6, lx=0.5, ly=0.3,
                                           angle=np.deg2rad(40), n_index=2.4),
                                      dict(type='circle', cx=0.3, cy=0.3, lx=0.2,
                                           n_index=3.0)])]

    def T(bk):
        return float(np.array(wavebit.rcwa(backend=bk, ucell=ucell, fto=(6, 6), **cfg)
                              .conv_solve().T).real.sum())

    tn, tj, tt = T(0), T(1), T(2)
    assert abs(tn - tj) < 1e-10
    assert abs(tn - tt) < 1e-7


# ----------------------------------------------------------------------
# 6. differentiability: gradient wrt shape params matches finite difference
# ----------------------------------------------------------------------
def _fd_grad_lx(shape_type, lx0=0.5):
    cfg = dict(n_top=1.0, n_bot=1.0, theta=0.0, phi=np.deg2rad(10),
               period=(1.0, 1.0), wavelength=1.5, thickness=(0.4,), psi=np.pi / 4)

    def T(lx):
        s = dict(type=shape_type, cx=0.5, cy=0.5, lx=lx, n_index=2.5,
                 **({} if shape_type == 'circle' else dict(ly=0.3, angle=0.4)))
        uc = [dict(n_base=1.0, shapes=[s])]
        return float(np.array(wavebit.rcwa(backend=0, ucell=uc, fto=(6, 6), **cfg)
                              .conv_solve().T).real.sum())

    h = 1e-5
    return (T(lx0 + h) - T(lx0 - h)) / (2 * h), cfg


@pytest.mark.parametrize("shape_type", ["circle", "ellipse"])
def test_torch_gradient(shape_type):
    torch = pytest.importorskip("torch")
    fd, cfg = _fd_grad_lx(shape_type)
    lx = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    s = dict(type=shape_type, cx=0.5, cy=0.5, lx=lx, n_index=2.5,
             **({} if shape_type == 'circle' else dict(ly=0.3, angle=0.4)))
    res = wavebit.rcwa(backend=2, ucell=[dict(n_base=1.0, shapes=[s])], fto=(6, 6), **cfg).conv_solve()
    T = res.T.real.sum()
    T.backward()
    assert np.isfinite(float(lx.grad))
    assert abs(float(lx.grad) - fd) < 1e-3


@pytest.mark.parametrize("shape_type", ["circle", "ellipse"])
def test_jax_gradient(shape_type):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    fd, cfg = _fd_grad_lx(shape_type)

    def loss(lx):
        s = dict(type=shape_type, cx=0.5, cy=0.5, lx=lx, n_index=2.5,
                 **({} if shape_type == 'circle' else dict(ly=0.3, angle=0.4)))
        r = wavebit.rcwa(backend=1, ucell=[dict(n_base=1.0, shapes=[s])], fto=(6, 6), **cfg).conv_solve()
        return jnp.real(r.T).sum()

    g = float(jax.grad(loss)(jnp.array(0.5)))
    assert np.isfinite(g)
    assert abs(g - fd) < 1e-3


# ----------------------------------------------------------------------
# 7. analytic converges faster than the staircase path (tilted dielectric)
# ----------------------------------------------------------------------
def test_analytic_beats_staircase_tilted():
    cfg = dict(n_top=1.0, n_bot=1.0, theta=0.0, phi=None,
               period=(1.0, 1.0), wavelength=1.6, thickness=(0.6,), psi=0.0)
    NR, ang, L, c = 3.5, np.deg2rad(30), 0.5, 0.5
    analytic = [dict(n_base=1.0, shapes=[dict(type='rectangle', cx=c, cy=c, lx=L, ly=L,
                                              angle=ang, n_index=NR)])]
    stair = [[1.0, [['rectangle', c, c, L, L, NR, ang, 24, 24]]]]

    def t0(res):
        d = np.array(res.T).real
        return d[d.shape[0] // 2, d.shape[0] // 2]

    ref = t0(wavebit.rcwa(backend=0, ucell=analytic, fto=(15, 15), **cfg).conv_solve())
    a11 = t0(wavebit.rcwa(backend=0, ucell=analytic, fto=(11, 11), **cfg).conv_solve())
    s11 = t0(wavebit.rcwa(backend=0, ucell=stair, fto=(11, 11), **cfg).conv_solve())

    err_analytic = abs(a11 - ref)
    err_stair = abs(s11 - ref)
    assert err_analytic < 5e-3
    assert err_analytic < err_stair / 5  # analytic is far closer to the converged answer


CIRCLE_P, CIRCLE_NR = 0.77, 3.5
CIRCLE_CFG = dict(period=[CIRCLE_P, CIRCLE_P], thickness=[0.25], theta=0.35, phi=0.4, psi=0.0,
                  wavelength=0.777, n_top=1.0, n_bot=1.5, fto=[6, 6])


def test_a_circle_is_written_with_one_length():
    """A circle is round, so one diameter is the whole of it — and the only spelling taken.

    The transform read ``lx`` as the diameter while the normal field read ``ly`` as a second
    semi-axis, and the two disagreed about the default (``lx`` versus ``500.0``). So passing
    ``ly != lx`` built one structure for ``eps`` and another for the projector — 14% on
    ``sum(R)`` — and *omitting* ``ly``, the natural way to write a circle, degenerated the
    normal field by 6%. Both resolve through ``shape_geometry`` now.

    That fix made ``ly`` harmless rather than absent, which this test used to pin: all three
    spellings returned the same number. It is now refused instead (``check_shape_spec``), for
    the reason in ``test_a_circle_that_names_a_second_length_is_refused`` — a length that is
    accepted and dropped still has a zero gradient, and silence is what the whole class of bug
    was made of. The anchor moves here, onto the one spelling there is.
    """
    shape = dict(type='circle', cx=0.5 * CIRCLE_P, cy=0.5 * CIRCLE_P, lx=0.5 * CIRCLE_P,
                 n_index=CIRCLE_NR)
    got = wavebit.rcwa(backend=0, ucell=[dict(n_base=1.0, shapes=[shape])],
                         **CIRCLE_CFG).conv_solve()
    # Only a regression anchor, and it rides on ``normal_grid`` — it read
    # 0.161262158771 while that default was 256, and moved 1.1e-4 when the default rose to 1024.
    # 0.161153064902 until 2026-08-16, when the sampled transform gained the half-pixel
    # cell-centre phase it had been missing (`_half_pixel_phase`). The analytic normal field is
    # sampled at `(i + 0.5)/N` by `sample_grid`, so its coefficients were half a pixel off the
    # permittivity they multiply; correcting it moves this anchor by 3.6e-5, which is the size an
    # O(1/N) term has at this `normal_grid`. Self-captured either way -- it says the number has
    # not drifted, not that it is right.
    assert float(np.sum(np.asarray(got.R))) == pytest.approx(0.161117311218, abs=1e-9)


@pytest.mark.parametrize('ly, why', [(0.5 * CIRCLE_P, 'a second length that agrees'),
                                     (0.3 * CIRCLE_P, 'a second length that does not')])
def test_a_circle_that_names_a_second_length_is_refused(ly, why):
    """Refused on the *key*, so both of these go the same way — which is the point.

    Accepting the agreeing one would mean reading the value to tell them apart, and a value is
    the one thing not readable under ``jax.grad``. See ``check_shape_spec``: the tolerance and
    identity handling that a value rule needs are exactly what phrasing it on the key removes.
    """
    shape = dict(type='circle', cx=0.5 * CIRCLE_P, cy=0.5 * CIRCLE_P, lx=0.5 * CIRCLE_P,
                 ly=ly, n_index=CIRCLE_NR)
    with pytest.raises(ValueError, match="'ly' would be dropped"):
        wavebit.rcwa(backend=0, ucell=[dict(n_base=1.0, shapes=[shape])],
                       **CIRCLE_CFG).conv_solve()


def test_shapes_superpose_where_they_overlap():
    """Overlapping shapes are added, not painted over — what ``combine='add'`` now selects.

    Each shape contributes ``(eps_shape - eps_base) * FT(indicator)``, so an overlap receives
    both. This is the coefficient builder underneath the modeling layer, so it superposes
    unconditionally; the layer above refuses to reach it without being told that is the
    intent (:func:`check_layer_combination`).

    The block path paints instead — later object over earlier — and agrees with a raster of
    the same instruction to eight digits, which is what makes the analytic value identifiable
    as the superposition rather than either of them being noise.
    """
    period, n_base = [300.0, 300.0], 1.0
    rods = [dict(type='rectangle', cx=130.0, cy=150.0, lx=120.0, ly=120.0, angle=0.0,
                 n_index=2.0),
            dict(type='rectangle', cx=190.0, cy=150.0, lx=120.0, ly=120.0, angle=0.0,
                 n_index=3.0)]

    eps_c, _ = _assemble_eps_coeffs(xnp, rods, n_base, 3, 3, period, np.complex128)
    dc = complex(eps_c[2 * 3, 2 * 3]).real

    cell = period[0] * period[1]
    fraction = 120.0 * 120.0 / cell
    superposed = n_base ** 2 + (2.0 ** 2 - n_base ** 2) * fraction \
        + (3.0 ** 2 - n_base ** 2) * fraction
    overlap = 60.0 * 120.0 / cell
    painted = superposed - (3.0 ** 2 - n_base ** 2) * overlap

    assert dc == pytest.approx(superposed, rel=1e-12)
    assert dc != pytest.approx(painted, rel=1e-6)      # 2.76 against 2.52


def test_overlapping_shapes_move_the_answer_while_the_vector_path_does_not():
    """The user-visible half of it: analytic parts from the raster, the block path does not.

    This is why the layer has to be asked. ``combine='add'`` is the answer that keeps the
    superposition, and it is a different structure from the one the same recipe builds on the
    block path — which is what the two assertions measure.
    """
    period, n_base = [300.0, 300.0], 1.0
    cfg = dict(period=period, thickness=[205.0], fto=[6, 6], n_top=1.0, n_bot=1.0,
               wavelength=900.0, psi=np.pi / 2, theta=0.0, phi=0.0)
    boxes = [(130.0, 2.0), (190.0, 3.0)]

    n_pix = 900
    axis = (np.arange(n_pix) + 0.5) / n_pix * period[0]
    X, Y = np.meshgrid(axis, axis)
    raster = np.full((n_pix, n_pix), n_base)
    for cx, n_index in boxes:                            # later box paints over the earlier
        raster = np.where((np.abs(X - cx) <= 60.0) & (np.abs(Y - 150.0) <= 60.0),
                          n_index, raster)

    def total(ucell, **extra):
        return float(np.sum(np.asarray(
            wavebit.rcwa(backend=0, ucell=ucell, **extra, **cfg).conv_solve().T)))

    want = total(raster[None], fourier_coefficients='sampled')
    vector = total([[n_base, [['rectangle', cx, 150.0, 120.0, 120.0, n, 0, 0, 0]
                              for cx, n in boxes]]])
    analytic = total([dict(n_base=n_base, combine='add',
                           shapes=[dict(type='rectangle', cx=cx, cy=150.0, lx=120.0, ly=120.0,
                                        angle=0.0, n_index=n) for cx, n in boxes])])

    assert vector == pytest.approx(want, abs=1e-8)
    assert abs(analytic - want) > 1e-3, (analytic, want)


# ------------------------------------------------------- the layer has to say how shapes combine

CHECK_CFG = dict(period=[300.0, 300.0], thickness=[205.0], fto=[2, 2], n_top=1.0, n_bot=1.0,
                 wavelength=900.0, psi=np.pi / 2, theta=0.0, phi=0.0)


def rod(cx, cy=150.0, lx=120.0, ly=120.0, angle=0.0, n_index=2.0, type='rectangle'):
    s = dict(type=type, cx=cx, cy=cy, lx=lx, ly=ly, angle=angle, n_index=n_index)
    if type in ('circle', 'disk', 'square'):
        # A round rod has one length. The helper's `ly` default is not the caller asking for a
        # second one, and passing it on would be this helper inventing an ambiguity to hand to
        # `check_shape_spec` -- `rod(80.0, lx=60.0, type='circle')` would arrive as 60 by 120.
        del s['ly']
    return s


def solve(shapes, **layer):
    return wavebit.rcwa(backend=0, ucell=[dict(n_base=1.0, shapes=shapes, **layer)],
                          **CHECK_CFG).conv_solve()


def test_overlapping_shapes_are_refused_until_the_layer_says_how_to_combine():
    """The hazard this whole check exists for: the same recipe means two structures.

    A vector recipe moved onto the smooth path used to come back superposed, with nothing
    said. Now the layer is refused until it names the rule, and the message has to carry both
    shapes and both ways forward — a user who cannot act on the error is no better off than
    one who never saw it.
    """
    shapes = [rod(130.0, n_index=2.0), rod(190.0, n_index=3.0)]

    with pytest.raises(ValueError) as raised:
        solve(shapes)
    message = str(raised.value)
    assert '0 (rectangle)' in message and '1 (rectangle)' in message
    assert "combine='add'" in message and 'block path' in message

    # And the escape hatch actually escapes, to the superposition that was always computed.
    assert float(np.sum(np.asarray(solve(shapes, combine='add').T))) > 0.0


def test_disjoint_shapes_never_have_to_answer_the_question():
    """No overlap, no difference between the rules — so no ``combine``, and any spelling works.

    This is the case the smooth path is built for and the one that must stay free: the check
    is only worth having if it fires on actual overlap, since pushing people off this path
    costs them the staircase-free rotation it exists to provide.
    """
    shapes = [rod(70.0, lx=100.0), rod(220.0, lx=100.0, n_index=3.0)]
    want = float(np.sum(np.asarray(solve(shapes).T)))

    for combine in ('add', 'paint'):
        assert float(np.sum(np.asarray(solve(shapes, combine=combine).T))) == want


def test_touching_is_not_overlapping():
    """Abutting shapes are the same structure either way, and designs are full of them.

    Two rectangles sharing an edge, two tangent circles, and a rod filling the period exactly
    all sit at a gap of zero. Reporting them would refuse ordinary geometry — a checkerboard,
    a full-width slab — for a difference of measure zero.
    """
    solve([rod(70.0), rod(190.0, n_index=3.0)])                       # edges meet at x = 130
    solve([rod(80.0, lx=60.0, type='circle'), rod(140.0, lx=60.0, type='circle')])
    solve([rod(150.0, lx=300.0)])                                     # exactly one period wide


def test_paint_now_builds_and_is_not_the_sum():
    """``combine='paint'`` used to raise here; the boundary path builds it, and it differs.

    It has to differ: the overlap holds the later shape's material under painting and the sum
    of both contributions under superposition, so agreeing would mean one of them is wrong.
    What this pins is that asking for painting no longer silently returns the sum -- the
    failure the ``combine`` keyword exists to prevent, now in the direction that used to be
    unbuildable rather than refused.
    """
    shapes = [rod(130.0), rod(190.0, n_index=3.0)]
    painted = float(np.sum(np.asarray(solve(shapes, combine='paint').T)))
    added = float(np.sum(np.asarray(solve(shapes, combine='add').T)))
    assert 0.0 < painted < 1.0
    # These rods overlap over a small part of the cell, so the two structures are close and
    # the gap is ~1e-3 -- four orders above the 1e-12 the two spellings agree to when nothing
    # intersects (`test_disjoint_shapes_never_have_to_answer_the_question`). That contrast is
    # the point; the size of the gap is the geometry's business, not this test's.
    assert abs(painted - added) > 1e-4, (painted, added)


def test_unknown_combine_modes_are_refused_even_without_an_overlap():
    """A typo in the one word that picks the semantics must not read as the default."""
    with pytest.raises(ValueError, match='combine='):
        solve([rod(150.0)], combine='superpose')


# ------------------------------------------- which side of `combine` each builder sits on
#
# `assemble_eps_coeffs` sums. `to_conv_mat_smooth_layers` reads the layer and routes a painted
# overlap to the boundary path instead. Reading only the first of those looks exactly like a
# solver that ignores `combine`, which is a diagnosis worth being able to settle with a number:
# `<eps>` is the DC coefficient, so the two rules give it in closed form and nothing about
# truncation, factorization or the normal field is in the way.

PAINT_PERIOD = (1000.0, 1000.0)

#: 300x300 rods of index 2 and 3, overlapping over 100x300 of a 1000x1000 cell. Summed that is
#: 1 + 3*0.09 + 8*0.09 = 1.99; painted, the earlier rod keeps only its unshared 60000, giving
#: 1 + 3*0.06 + 8*0.09 = 1.90. A 4.7% gap, and both are plausible numbers.
PAINT_RODS = [dict(type='rectangle', cx=400.0, cy=500.0, lx=300.0, ly=300.0, n_index=2.0),
              dict(type='rectangle', cx=600.0, cy=500.0, lx=300.0, ly=300.0, n_index=3.0)]
EPS_SUMMED, EPS_PAINTED = 1.99, 1.90


def _solver_mean_eps(combine, shapes=PAINT_RODS):
    """``<eps>`` as the solve reads it: the DC entry of ``ep_dir``, recovered from ``epz_i``."""
    from wavebit._core.analytic import to_conv_mat_smooth_layers
    layer = dict(n_base=1.0, shapes=shapes, combine=combine)
    mats = to_conv_mat_smooth_layers([layer], 6, 6, PAINT_PERIOD, normal_grid=256,
                                     type_complex=np.complex128, xp=xnp)
    return float(np.real(np.linalg.inv(np.asarray(mats.epz_i[0]))[0, 0]))


def test_the_solve_path_integrates_the_painted_structure_not_the_summed_one():
    """The solver honours ``combine``; only the bare coefficient builder does not.

    Both numbers are exact, not converged-to: ``<eps>`` is one Fourier coefficient and each
    rule fixes it in closed form. So this is not "close enough" — a solver that ignored
    ``combine`` would land on 1.99 for both rows and this would fail by 4.7%.
    """
    assert _solver_mean_eps('paint') == pytest.approx(EPS_PAINTED, abs=1e-9)
    assert _solver_mean_eps('add') == pytest.approx(EPS_SUMMED, abs=1e-9)


def test_assemble_eps_coeffs_sums_whatever_it_is_handed():
    """The precondition, pinned as behaviour so the docstring cannot drift off it.

    This is the misuse that reads as a solver bug: pull ``shapes`` out of a painted layer, hand
    them here, and get the summed structure back with nothing said. Both spellings give 1.99
    because the layer's rule was never in the argument list.
    """
    eps_c, _ = assemble_eps_coeffs(PAINT_RODS, 1.0, 6, 6, PAINT_PERIOD)
    assert float(np.real(eps_c[12, 12])) == pytest.approx(EPS_SUMMED, abs=1e-12)


def test_assemble_eps_coeffs_refuses_a_painted_overlap_when_told_the_rule():
    """Told what it cannot build, it refuses instead of answering plausibly.

    The message has to name the shapes and the path that *can* build it, for the same reason
    the layer-level refusal does: an error a caller cannot act on leaves them where they were.
    """
    with pytest.raises(ValueError) as raised:
        _assemble_eps_coeffs(xnp, PAINT_RODS, 1.0, 6, 6, PAINT_PERIOD, xnp.complex128,
                             combine='paint')
    message = str(raised.value)
    assert '0 (rectangle)' in message and '1 (rectangle)' in message
    assert 'boundary' in message


def test_a_disjoint_painted_layer_still_reaches_the_closed_form_builder():
    """The guard must not push the cheap exact path away from layers it handles correctly.

    Nothing intersects, so painting and summing are the same structure and this builder is
    both right and cheaper than resolving polygons that have nothing to resolve — which is
    also why the shipped dispatch sends such a layer here with ``combine='paint'`` still on it.
    """
    disjoint = [dict(PAINT_RODS[0], cx=200.0), dict(PAINT_RODS[1], cx=700.0)]
    eps_c, _ = _assemble_eps_coeffs(xnp, disjoint, 1.0, 6, 6, PAINT_PERIOD, xnp.complex128,
                                    combine='paint')
    assert float(np.real(eps_c[12, 12])) == pytest.approx(EPS_SUMMED, abs=1e-12)
    assert _solver_mean_eps('paint', disjoint) == pytest.approx(EPS_SUMMED, abs=1e-9)


# ------------------------------------------------------- what a shape dict may say
def _circle(**over):
    return dict(dict(type='circle', cx=150.0, cy=150.0, lx=120.0, n_index=2.0), **over)


@pytest.mark.parametrize('shape, match', [
    # a key nobody reads is a key nobody complains about -- every one of these was silent
    (_circle(ly=200.0), "'ly' would be dropped"),             # meant an ellipse
    (dict(_circle(), type='square', ly=200.0), 'rectangle'),  # meant a rectangle
    (_circle(radius=60.0), 'radius'),                         # meant lx, and got a size-0 shape
    (_circle(angel=0.7), "did you mean 'angle'"),             # meant angle, and got no rotation
    (_circle(type='Circle'), 'Known types'),                   # meant circle, and got ly = 0
])
def test_a_shape_dict_that_says_something_unreadable_is_refused(shape, match):
    """Every default is a ``.get``, so an unread key is dropped in silence. Not any more.

    ``shape_geometry`` stays total on purpose -- it has to be usable under a trace -- so this
    is charged once at the funnel instead. All five of these built a structure and returned a
    number before; two of them (``radius``, ``Circle``) built a shape of size zero.
    """
    with pytest.raises(ValueError, match=match):
        solve([shape])


@pytest.mark.parametrize('shape', [
    _circle(),                                       # the natural spelling: one diameter
    dict(type='circle', cx=150.0, cy=150.0, l=120.0, n_index=2.0),   # `l` says the same
    dict(_circle(), type='ellipse', ly=200.0),       # two lengths, on the type that has two
    dict(_circle(), type='rectangle', ly=200.0, angle=0.4),
])
def test_a_shape_dict_that_says_one_thing_is_accepted(shape):
    """The refusal is for saying more than the type has room for, not for using the path."""
    assert float(np.sum(np.asarray(solve([shape]).T))) > 0.0


def test_a_traced_ly_on_a_circle_is_refused_where_a_comparison_cannot_reach():
    """The dead-gradient case, and the reason this checks the *key* and not the values.

    ``jax.grad`` hands ``ly`` in as a tracer, which no comparison can read -- so a rule phrased
    as "refuse when lx != ly" is blind here, eagerly correct and silently off under the one
    workflow that matters most. And ignored means ``d(loss)/d(ly)`` is exactly 0.0, which an
    optimiser cannot tell from a converged direction.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    from wavebit._core.analytic import check_shape_spec

    def f(ly):
        check_shape_spec(_circle(ly=ly))
        return ly

    with pytest.raises(ValueError, match="'ly' would be dropped"):
        jax.grad(f)(jnp.asarray(120.0))


def test_overlap_is_found_across_the_period_boundary():
    """The layer is a unit cell: shapes that miss inside it can still meet through the edge."""
    shapes = [rod(20.0, lx=100.0), rod(280.0, lx=100.0, n_index=3.0)]   # meet through x = 0

    with pytest.raises(ValueError, match='period boundary'):
        solve(shapes)
    solve(shapes, combine='add')


def test_a_shape_wider_than_the_period_runs_into_its_own_image():
    """Same silent-wrong-structure class: one shape, no pair, and still not what it says.

    Its transform is a single sinc that no longer describes a periodic indicator, so the layer
    is not the structure the parameters name. One shape is enough to report.
    """
    with pytest.raises(ValueError, match='larger than the period'):
        solve([rod(150.0, lx=340.0)])


def test_detection_follows_the_shapes_and_not_their_bounding_boxes():
    """Rotation and curvature are where a lazy test would refuse a perfectly good layer.

    Both pairs have overlapping axis-aligned bounding boxes and are disjoint: two rectangles
    at +/- 45 degrees passing corner to corner, and two ellipses crossing broadside to
    end-on. Nudged into contact, both are reported — so the pass is not the check being
    asleep.
    """
    diagonal = [rod(110.0, 110.0, lx=120.0, ly=40.0, angle=np.pi / 4),
                rod(190.0, 190.0, lx=120.0, ly=40.0, angle=-np.pi / 4, n_index=3.0)]
    crossing = [rod(120.0, 150.0, lx=160.0, ly=30.0, type='ellipse'),
                rod(225.0, 150.0, lx=30.0, ly=160.0, type='ellipse', n_index=3.0)]

    for shapes in (diagonal, crossing):
        solve(shapes)                                          # disjoint: no question to ask
        closed = [dict(shapes[0]), dict(shapes[1], cx=shapes[0]['cx'] + 10.0)]
        with pytest.raises(ValueError, match='overlap'):
            solve(closed)


def test_the_check_reports_the_exact_geometry_it_should():
    """The gap itself, against closed forms — the numbers the report is a threshold on."""
    from wavebit._core.overlap import primitive, signed_gap

    circle = primitive('circle', 0.0, 0.0, 20.0, 20.0, 0.0)
    assert signed_gap(circle, primitive('circle', 24.0, 0.0, 20.0, 20.0, 0.0)) \
        == pytest.approx(4.0, abs=1e-12)                       # centres 24 apart, radii 10
    assert signed_gap(circle, primitive('rectangle', 30.0, 0.0, 20.0, 20.0, 0.0)) \
        == pytest.approx(10.0, abs=1e-12)                      # face at 20, circle ends at 10
    assert signed_gap(primitive('rectangle', 0.0, 0.0, 20.0, 20.0, 0.0),
                      primitive('rectangle', 16.0, 0.0, 20.0, 20.0, 0.0)) \
        == pytest.approx(-4.0, abs=1e-12)                       # penetration depth, not distance

    # The searched path (a genuine ellipse) has to land on the same answers.
    assert signed_gap(primitive('ellipse', 0.0, 0.0, 20.0, 10.0, 0.0),
                      primitive('ellipse', 45.0, 0.0, 20.0, 10.0, 0.0)) \
        == pytest.approx(25.0, rel=1e-9)                        # semi-axes 10 and 10 along x
    assert signed_gap(primitive('ellipse', 0.0, 0.0, 20.0, 10.0, np.pi / 2),
                      primitive('circle', 0.0, 30.0, 20.0, 20.0, 0.0)) \
        == pytest.approx(10.0, rel=1e-9)                        # rotated: semi-axis 10 along y


def test_the_check_steps_aside_for_a_traced_solve():
    """It compares numbers, and a tracer is not one — so tracing must pass through, not fail.

    A jitted analytic solve is the reason this cannot simply be an unconditional assertion:
    turning a differentiable sweep into an exception would cost more than the silence it
    replaces. ``freeze_geometry`` is the eager pass such a sweep already makes, and the same
    layer with a concrete ``cx`` right beside it is what still gets checked.
    """
    jax = pytest.importorskip('jax')
    jnp = pytest.importorskip('jax.numpy')

    def layer(cx):
        return [dict(n_base=1.0, shapes=[rod(cx), rod(190.0, n_index=3.0)])]

    frozen = wavebit.rcwa(backend=1, ucell=layer(60.0), **CHECK_CFG).freeze_geometry()

    def total(cx):
        return jnp.sum(wavebit.rcwa(backend=1, ucell=layer(cx),
                                      **CHECK_CFG).conv_solve(geometry=frozen).T)

    assert float(jax.jit(total)(jnp.asarray(130.0))) > 0.0     # overlapping, and traced: allowed
    with pytest.raises(ValueError, match='overlap'):
        total(130.0)                                           # same layer, concrete: refused


# ------------------------------------------------------------------ the block path's half

VECTOR_CFG = dict(period=[300.0, 300.0], thickness=[205.0], fto=[6, 6], n_top=1.0, n_bot=1.0,
                  wavelength=900.0, psi=np.pi / 2, theta=0.0, phi=0.0)


def _vector_box(cx, n_index=2.0):
    return ['rectangle', cx, 150.0, 120.0, 120.0, n_index, 0.0, 0, 0]


def _vector_total(objects, *combine):
    layer = [1.0, objects, *combine]
    return float(np.sum(np.asarray(
        wavebit.rcwa(backend=0, ucell=[layer], **VECTOR_CFG).conv_solve().T)))


def test_the_vector_path_refuses_the_superposition_it_cannot_draw():
    """The mirror of the analytic refusal, and the other half of making `combine` portable.

    Writing ``combine='add'`` and handing it to this path used to paint it, silently -- the
    same trap in the other direction. It cannot be drawn here: this path lays down a staircase
    of blocks and each block holds one index, so an overlap can only take one of them.
    """
    with pytest.raises(NotImplementedError, match='cannot add them'):
        _vector_total([_vector_box(140.0), _vector_box(180.0, 3.0)], 'add')


def test_the_vector_path_still_paints_what_it_always_painted():
    """`combine` is not a new tax: an unlabelled layer, and `'paint'`, are what this path did.

    Painting is this path's own rule and a raster of the same recipe agrees with it, so unlike
    the analytic side there is nothing surprising here to refuse.
    """
    objects = [_vector_box(140.0), _vector_box(180.0, 3.0)]
    silent = _vector_total(objects)
    assert _vector_total(objects, 'paint') == silent


def test_combine_is_free_while_nothing_overlaps():
    """Disjoint shapes mean one structure, so either spelling is accepted -- on both paths.

    That is what keeps `combine` from taxing the case the smooth path exists for, and it is
    also the case in which the two paths have to agree: same recipe, same answer.
    """
    objects = [_vector_box(80.0), _vector_box(220.0)]
    disjoint = _vector_total(objects)
    assert _vector_total(objects, 'add') == disjoint
    assert _vector_total(objects, 'paint') == disjoint

    shapes = [dict(type='rectangle', cx=cx, cy=150.0, lx=120.0, ly=120.0, angle=0.0,
                   n_index=2.0) for cx in (80.0, 220.0)]
    analytic = float(np.sum(np.asarray(
        wavebit.rcwa(backend=0, ucell=[dict(n_base=1.0, shapes=shapes)],
                       **VECTOR_CFG).conv_solve().T)))
    assert analytic == pytest.approx(disjoint, abs=2e-3)


def test_a_misspelled_combine_is_refused_on_the_vector_path_too():
    with pytest.raises(ValueError, match='is not one of'):
        _vector_total([_vector_box(80.0)], 'union')


def test_the_tiled_blend_is_bit_identical_to_the_whole_grid_one():
    """The tiled traversal is an optimization and must be invisible in the answer.

    ``normal_vector_field_analytical`` evaluates the multi-shape blend one cache-sized block at
    a time on NumPy, because each of the ~50 array operations per shape otherwise reads and
    writes the whole grid: at N = 1024 a two-shape layer moves roughly 800 MB through memory to
    do a few tens of megaflops. Blocking it is worth 1.4x at that size and 1.6x at 2048.

    The claim that makes it acceptable is that it changes nothing — same operations, same
    values, same order, different traversal — so this compares the two paths bit-for-bit rather
    than to a tolerance. A tolerance here would pass for a version that had quietly become an
    approximation.

    Both branches are exercised by flipping the pixel threshold rather than the grid size, so
    the two arms see an identical layer.
    """
    from wavebit._core import analytic as _an

    shapes = [dict(type='rectangle', cx=0.31, cy=0.44, lx=0.30, ly=0.22, angle=0.4, n_index=3.5),
              dict(type='circle', cx=0.70, cy=0.62, lx=0.22, n_index=2.1),
              dict(type='ellipse', cx=0.20, cy=0.80, lx=0.18, ly=0.11, angle=1.1, n_index=1.8)]
    guard = _an._TILE_MIN_PIXELS
    try:
        for n in (256, 512, 640):                      # 640 is not a multiple of the tile
            _an._TILE_MIN_PIXELS = 10 ** 12            # force the whole-grid path
            whole = _an.normal_vector_field_analytical(xnp, (n, n), shapes, (1.0, 1.0),
                                                       np.float64)
            _an._TILE_MIN_PIXELS = 0                   # force the tiled path
            tiled = _an.normal_vector_field_analytical(xnp, (n, n), shapes, (1.0, 1.0),
                                                       np.float64)
            for a, b in zip(whole, tiled):
                assert np.array_equal(np.asarray(a), np.asarray(b)), n
    finally:
        _an._TILE_MIN_PIXELS = guard


def test_only_numpy_takes_the_tiled_path():
    """JAX and Torch must keep the whole-grid path, and not because it is faster there.

    A Python loop over 256 blocks unrolls into a trace, so on those backends the optimization
    would trade a memory win it does not need — XLA and the compiler already fuse the chain
    into one pass — for a compile time that scales with the tile count.
    """
    from wavebit._core import analytic as _an

    numpy_grid = np.zeros((4, 4))
    assert _an._tiles_pay(numpy_grid, 1024, 1024, 2)
    assert not _an._tiles_pay(numpy_grid, 1024, 1024, 1)   # a lone shape never blends
    assert not _an._tiles_pay(numpy_grid, 64, 64, 2)       # already cache-resident

    # The decision reads the array, not the namespace's name -- `jax.numpy` ends with `numpy`,
    # so the obvious spelling would have switched the tiling ON for the one backend that must
    # not take it. This assertion is the one that caught it.
    jax = pytest.importorskip('jax')
    assert not _an._tiles_pay(jax.numpy.zeros((4, 4)), 1024, 1024, 2)
