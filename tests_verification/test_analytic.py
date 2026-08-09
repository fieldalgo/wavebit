"""
Verification tests for the ANALYTIC (staircase-free) modeling path.

Covers: Fourier-coefficient convention vs dfs2d, the ``n -> eps`` convention under complex
indices, analytic formulas vs a fine midpoint DFT, axis-aligned equivalence with the
raster-NVF path, energy conservation, cross-backend agreement, differentiability, and the
convergence advantage over the staircase path.
"""
import warnings

import numpy as np
import pytest

import fieldwave
import array_api_compat.numpy as xnp
from fieldwave._core.analytic import (
    shape_ft as _shape_ft, laurent_conv_2d as _laurent_conv_2d,
    assemble_eps_coeffs as _assemble_eps_coeffs, to_conv_mat_analytic_nvf)
from fieldwave._core.backend import UnphysicalResultWarning
from fieldwave._core.factorization import dfs2d as _dfs2d


# The core builders are namespace-generic; bind NumPy for these convention checks.
def dfs2d(cell, cx, cy, fx, fy):
    return _dfs2d(xnp, cell, cx, cy, fx, fy)


def shape_ft(shape, ky, kx, period):
    return _shape_ft(xnp, shape, ky, kx, period)


def laurent_conv_2d(coeff2d, fto_x, fto_y):
    return _laurent_conv_2d(xnp, coeff2d, fto_x, fto_y)


def assemble_eps_coeffs(shapes, n_base, fto_x, fto_y, period):
    return _assemble_eps_coeffs(xnp, shapes, n_base, fto_x, fto_y, period, xnp.complex128)


def to_conv_mat_analytic(shapes, n_base, fto_x, fto_y, period):
    """Direct-rule (eps, 1/eps) Laurent pair — the NVF builder's inputs."""
    eps_c, inv_c = assemble_eps_coeffs(shapes, n_base, fto_x, fto_y, period)
    return laurent_conv_2d(eps_c, fto_x, fto_y), laurent_conv_2d(inv_c, fto_x, fto_y)


# ----------------------------------------------------------------------
# 1. coefficient-builder conventions: ordering, and n -> eps
# ----------------------------------------------------------------------
def test_laurent_matches_dfs2d_convention():
    rng = np.random.default_rng(0)
    Ny, Nx = 12, 10
    cell = rng.random((Ny, Nx)) + 1.0
    fto_x, fto_y = 2, 3

    M_dfs = dfs2d(cell.astype(complex), 1, 1, fto_x, fto_y)

    F = np.fft.fft2(cell) / (Nx * Ny)
    KY = np.arange(-2 * fto_y, 2 * fto_y + 1)
    KX = np.arange(-2 * fto_x, 2 * fto_x + 1)
    coeff2d = np.array([[F[ky % Ny, kx % Nx] for kx in KX] for ky in KY])
    M_lau = laurent_conv_2d(coeff2d, fto_x, fto_y)

    assert np.abs(M_dfs - M_lau).max() < 1e-12


@pytest.mark.parametrize('where', ['n_base', 'n_index'])
def test_complex_index_is_loss_not_gain(where):
    """``Im(n) > 0`` must come out as ``Im(eps) > 0``, on both the background and the shape.

    The builder conjugated on the way in (``conj(n)**2``), which is the convention
    ``_core/conventions.py`` says FieldWave removed -- so an absorbing index became a gain
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
    dict(type='circle', cx=0.5, cy=0.5, lx=0.6, ly=0.6, angle=0.0),
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
        res = fieldwave.rcwa(backend=0, ucell=ucell, psi=psi, **cfg, **kw).conv_solve()
        return float(np.sum(np.asarray(res.de_ri)))

    for psi in (np.pi / 2, 0.0):                       # TE, then TM
        want = total(analytic, psi)
        assert abs(total(ras, psi, factorization='nvf') - want) < 1e-3  # raster NVF
        assert abs(total(ras, psi) - want) < 2e-3                       # raster DFS


@pytest.mark.parametrize('psi', [np.pi / 2, 0.0], ids=['te', 'tm'])
def test_metal_rod_absorbs_end_to_end(psi):
    """A metal rod through the whole solve must absorb, not amplify.

    The coefficient test above pins ``n -> eps``; this pins that nothing downstream undoes it.
    ``conj(n)**2`` turned this rod into a gain medium, and a passive stack that returns
    ``R + T > 1`` is exactly what ``check_physical`` exists to catch — so the warning being
    absent is as much the assertion as the number is.

    Why this rather than a comparison against the same rod rastered: the two NVF schemes do not
    agree closely for a metal. Their gap here is ~3e-3 on ``sum(de_ti)`` at ``fto=8`` and does
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
        res = fieldwave.rcwa(backend=0, ucell=ucell, psi=psi, n_top=1.0, n_bot=1.5,
                             theta=0.35, phi=0.4, period=(P, P), wavelength=0.777,
                             thickness=(0.25,), fto=(8, 8)).conv_solve()

    total = float(np.sum(np.asarray(res.de_ri)).real + np.sum(np.asarray(res.de_ti)).real)
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
        res = fieldwave.rcwa(backend=0, ucell=ucell, **cfg, **kw).conv_solve()
        return float(np.sum(np.asarray(res.de_ri)))

    # The two readings separate by ~90x, so the threshold is not load-bearing; it is set
    # above the analytic-vs-raster gap and far below the misreading. That gap widened from
    # 1e-3 when the NVF normal field stopped being taken on an over-refined grid (2026-08-07,
    # see docs/nvf-upsampling.md) -- a raster DFS reference sits at 1.7e-3 on this structure
    # too, so it is the reference that moved, not the shape parameters.
    reference = total(ras, factorization='nvf')
    assert abs(total(shaped(L * P, 0.5 * P)) - reference) < 5e-3        # absolute: agrees
    assert abs(total(shaped(L, 0.5)) - reference) > 1e-2               # fractions: does not


# ----------------------------------------------------------------------
# 4. energy conservation for a lossless tilted/curved multi-shape stack
# ----------------------------------------------------------------------
def test_energy_conservation_lossless():
    cfg = dict(n_top=1.0, n_bot=1.0, theta=np.deg2rad(5), phi=np.deg2rad(20),
               period=(1.0, 1.2), wavelength=1.5, thickness=(0.3, 0.2), pol=0.3)
    ucell = [
        dict(n_base=1.0, shapes=[dict(type='ellipse', cx=0.5, cy=0.6, lx=0.5, ly=0.3,
                                      angle=np.deg2rad(40), n_index=2.4)]),
        dict(n_base=1.5, shapes=[dict(type='rectangle', cx=0.4, cy=0.5, lx=0.4, ly=0.5,
                                      angle=np.deg2rad(25), n_index=1.0),
                                 dict(type='circle', cx=0.8, cy=0.9, lx=0.25, ly=0.25,
                                      n_index=3.0)]),
    ]
    for fto in (5, 8):
        res = fieldwave.rcwa(backend=0, ucell=ucell, fto=(fto, fto), **cfg).conv_solve()
        tot = np.array(res.de_ri).real.sum() + np.array(res.de_ti).real.sum()
        assert abs(tot - 1.0) < 1e-9


# ----------------------------------------------------------------------
# 5. cross-backend agreement (numpy == jax tight, numpy == torch loose)
# ----------------------------------------------------------------------
def test_cross_backend_agreement():
    cfg = dict(n_top=1.0, n_bot=1.0, theta=np.deg2rad(5), phi=np.deg2rad(20),
               period=(1.0, 1.2), wavelength=1.5, thickness=(0.3,), pol=0.3)
    ucell = [dict(n_base=1.0, shapes=[dict(type='ellipse', cx=0.5, cy=0.6, lx=0.5, ly=0.3,
                                           angle=np.deg2rad(40), n_index=2.4),
                                      dict(type='circle', cx=0.3, cy=0.3, lx=0.2, ly=0.2,
                                           n_index=3.0)])]

    def T(bk):
        return float(np.array(fieldwave.rcwa(backend=bk, ucell=ucell, fto=(6, 6), **cfg)
                              .conv_solve().de_ti).real.sum())

    tn, tj, tt = T(0), T(1), T(2)
    assert abs(tn - tj) < 1e-10
    assert abs(tn - tt) < 1e-7


# ----------------------------------------------------------------------
# 6. differentiability: gradient wrt shape params matches finite difference
# ----------------------------------------------------------------------
def _fd_grad_lx(shape_type, lx0=0.5):
    cfg = dict(n_top=1.0, n_bot=1.0, theta=0.0, phi=np.deg2rad(10),
               period=(1.0, 1.0), wavelength=1.5, thickness=(0.4,), pol=0.5)

    def T(lx):
        s = dict(type=shape_type, cx=0.5, cy=0.5, lx=lx, ly=(lx if shape_type == 'circle' else 0.3),
                 angle=(0.0 if shape_type == 'circle' else 0.4), n_index=2.5)
        uc = [dict(n_base=1.0, shapes=[s])]
        return float(np.array(fieldwave.rcwa(backend=0, ucell=uc, fto=(6, 6), **cfg)
                              .conv_solve().de_ti).real.sum())

    h = 1e-5
    return (T(lx0 + h) - T(lx0 - h)) / (2 * h), cfg


@pytest.mark.parametrize("shape_type", ["circle", "ellipse"])
def test_torch_gradient(shape_type):
    torch = pytest.importorskip("torch")
    fd, cfg = _fd_grad_lx(shape_type)
    lx = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    s = dict(type=shape_type, cx=0.5, cy=0.5, lx=lx, ly=(lx if shape_type == 'circle' else 0.3),
             angle=(0.0 if shape_type == 'circle' else 0.4), n_index=2.5)
    res = fieldwave.rcwa(backend=2, ucell=[dict(n_base=1.0, shapes=[s])], fto=(6, 6), **cfg).conv_solve()
    T = res.de_ti.real.sum()
    T.backward()
    assert np.isfinite(float(lx.grad))
    assert abs(float(lx.grad) - fd) < 1e-3


@pytest.mark.parametrize("shape_type", ["circle", "ellipse"])
def test_jax_gradient(shape_type):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    fd, cfg = _fd_grad_lx(shape_type)

    def loss(lx):
        s = dict(type=shape_type, cx=0.5, cy=0.5, lx=lx, ly=(lx if shape_type == 'circle' else 0.3),
                 angle=(0.0 if shape_type == 'circle' else 0.4), n_index=2.5)
        r = fieldwave.rcwa(backend=1, ucell=[dict(n_base=1.0, shapes=[s])], fto=(6, 6), **cfg).conv_solve()
        return jnp.real(r.de_ti).sum()

    g = float(jax.grad(loss)(jnp.array(0.5)))
    assert np.isfinite(g)
    assert abs(g - fd) < 1e-3


# ----------------------------------------------------------------------
# 7. analytic converges faster than the staircase path (tilted dielectric)
# ----------------------------------------------------------------------
def test_analytic_beats_staircase_tilted():
    cfg = dict(n_top=1.0, n_bot=1.0, theta=0.0, phi=None,
               period=(1.0, 1.0), wavelength=1.6, thickness=(0.6,), pol=1.0)
    NR, ang, L, c = 3.5, np.deg2rad(30), 0.5, 0.5
    analytic = [dict(n_base=1.0, shapes=[dict(type='rectangle', cx=c, cy=c, lx=L, ly=L,
                                              angle=ang, n_index=NR)])]
    stair = [[1.0, [['rectangle', c, c, L, L, NR, ang, 24, 24]]]]

    def t0(res):
        d = np.array(res.de_ti).real
        return d[d.shape[0] // 2, d.shape[0] // 2]

    ref = t0(fieldwave.rcwa(backend=0, ucell=analytic, fto=(15, 15), **cfg).conv_solve())
    a11 = t0(fieldwave.rcwa(backend=0, ucell=analytic, fto=(11, 11), **cfg).conv_solve())
    s11 = t0(fieldwave.rcwa(backend=0, ucell=stair, fto=(11, 11), **cfg).conv_solve())

    err_analytic = abs(a11 - ref)
    err_stair = abs(s11 - ref)
    assert err_analytic < 5e-3
    assert err_analytic < err_stair / 5  # analytic is far closer to the converged answer


@pytest.mark.parametrize('ly', [None, 'same', 'different'])
def test_circle_ignores_ly_everywhere(ly):
    """A circle is round: ``ly`` must not change the answer, and omitting it must be safe.

    The transform read ``lx`` as the diameter while the normal field read ``ly`` as a second
    semi-axis, and the two disagreed about the default (``lx`` versus ``500.0``). So passing
    ``ly != lx`` built one structure for ``eps`` and another for the projector — 14% on
    ``sum(de_ri)`` — and *omitting* ``ly``, the natural way to write a circle, degenerated the
    normal field by 6%. Both resolve through ``shape_geometry`` now.
    """
    P, NR = 0.77, 3.5
    shape = dict(type='circle', cx=0.5 * P, cy=0.5 * P, lx=0.5 * P, n_index=NR)
    if ly == 'same':
        shape['ly'] = 0.5 * P
    elif ly == 'different':
        shape['ly'] = 0.3 * P

    cfg = dict(period=[P, P], thickness=[0.25], theta=0.35, phi=0.4, psi=0.0,
               wavelength=0.777, n_top=1.0, n_bot=1.5, fto=[6, 6])
    got = fieldwave.rcwa(backend=0, ucell=[dict(n_base=1.0, shapes=[shape])], **cfg).conv_solve()
    # The three spellings agreeing is the claim; the absolute value is only a regression
    # anchor, and it rides on ``analytic_normal_grid`` — it read 0.161262158771 while that
    # default was 256, and moved 1.1e-4 when the default rose to 1024.
    assert float(np.sum(np.asarray(got.de_ri))) == pytest.approx(0.161153064902, abs=1e-9)


def test_shapes_superpose_where_they_overlap():
    """Overlapping shapes are added, not painted over — what ``combine='add'`` now selects.

    Each shape contributes ``(eps_shape - eps_base) * FT(indicator)``, so an overlap receives
    both. This is the coefficient builder underneath the modeling layer, so it superposes
    unconditionally; the layer above refuses to reach it without being told that is the
    intent (:func:`check_layer_combination`).

    The vector path paints instead — later object over earlier — and agrees with a raster of
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
    """The user-visible half of it: analytic parts from the raster, the vector path does not.

    This is why the layer has to be asked. ``combine='add'`` is the answer that keeps the
    superposition, and it is a different structure from the one the same recipe builds on the
    vector path — which is what the two assertions measure.
    """
    period, n_base = [300.0, 300.0], 1.0
    cfg = dict(period=period, thickness=[205.0], fto=[6, 6], n_top=1.0, n_bot=1.0,
               wavelength=900.0, pol=0, theta=0.0, phi=0.0)
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
            fieldwave.rcwa(backend=0, ucell=ucell, **extra, **cfg).conv_solve().de_ti)))

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
                 wavelength=900.0, pol=0, theta=0.0, phi=0.0)


def rod(cx, cy=150.0, lx=120.0, ly=120.0, angle=0.0, n_index=2.0, type='rectangle'):
    return dict(type=type, cx=cx, cy=cy, lx=lx, ly=ly, angle=angle, n_index=n_index)


def solve(shapes, **layer):
    return fieldwave.rcwa(backend=0, ucell=[dict(n_base=1.0, shapes=shapes, **layer)],
                          **CHECK_CFG).conv_solve()


def test_overlapping_shapes_are_refused_until_the_layer_says_how_to_combine():
    """The hazard this whole check exists for: the same recipe means two structures.

    A vector recipe moved onto the analytic path used to come back superposed, with nothing
    said. Now the layer is refused until it names the rule, and the message has to carry both
    shapes and both ways forward — a user who cannot act on the error is no better off than
    one who never saw it.
    """
    shapes = [rod(130.0, n_index=2.0), rod(190.0, n_index=3.0)]

    with pytest.raises(ValueError) as raised:
        solve(shapes)
    message = str(raised.value)
    assert '0 (rectangle)' in message and '1 (rectangle)' in message
    assert "combine='add'" in message and 'vector path' in message

    # And the escape hatch actually escapes, to the superposition that was always computed.
    assert float(np.sum(np.asarray(solve(shapes, combine='add').de_ti))) > 0.0


def test_disjoint_shapes_never_have_to_answer_the_question():
    """No overlap, no difference between the rules — so no ``combine``, and any spelling works.

    This is the case the analytic path is built for and the one that must stay free: the check
    is only worth having if it fires on actual overlap, since pushing people off this path
    costs them the staircase-free rotation it exists to provide.
    """
    shapes = [rod(70.0, lx=100.0), rod(220.0, lx=100.0, n_index=3.0)]
    want = float(np.sum(np.asarray(solve(shapes).de_ti)))

    for combine in ('add', 'paint'):
        assert float(np.sum(np.asarray(solve(shapes, combine=combine).de_ti))) == want


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
    painted = float(np.sum(np.asarray(solve(shapes, combine='paint').de_ti)))
    added = float(np.sum(np.asarray(solve(shapes, combine='add').de_ti)))
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
    from fieldwave._core.overlap import primitive, signed_gap

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

    frozen = fieldwave.rcwa(backend=1, ucell=layer(60.0), **CHECK_CFG).freeze_geometry()

    def total(cx):
        return jnp.sum(fieldwave.rcwa(backend=1, ucell=layer(cx),
                                      **CHECK_CFG).conv_solve(geometry=frozen).de_ti)

    assert float(jax.jit(total)(jnp.asarray(130.0))) > 0.0     # overlapping, and traced: allowed
    with pytest.raises(ValueError, match='overlap'):
        total(130.0)                                           # same layer, concrete: refused


# ------------------------------------------------------------------ the vector path's half

VECTOR_CFG = dict(period=[300.0, 300.0], thickness=[205.0], fto=[6, 6], n_top=1.0, n_bot=1.0,
                  wavelength=900.0, pol=0, theta=0.0, phi=0.0)


def _vector_box(cx, n_index=2.0):
    return ['rectangle', cx, 150.0, 120.0, 120.0, n_index, 0.0, 0, 0]


def _vector_total(objects, *combine):
    layer = [1.0, objects, *combine]
    return float(np.sum(np.asarray(
        fieldwave.rcwa(backend=0, ucell=[layer], **VECTOR_CFG).conv_solve().de_ti)))


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

    That is what keeps `combine` from taxing the case the analytic path exists for, and it is
    also the case in which the two paths have to agree: same recipe, same answer.
    """
    objects = [_vector_box(80.0), _vector_box(220.0)]
    disjoint = _vector_total(objects)
    assert _vector_total(objects, 'add') == disjoint
    assert _vector_total(objects, 'paint') == disjoint

    shapes = [dict(type='rectangle', cx=cx, cy=150.0, lx=120.0, ly=120.0, angle=0.0,
                   n_index=2.0) for cx in (80.0, 220.0)]
    analytic = float(np.sum(np.asarray(
        fieldwave.rcwa(backend=0, ucell=[dict(n_base=1.0, shapes=shapes)],
                       **VECTOR_CFG).conv_solve().de_ti)))
    assert analytic == pytest.approx(disjoint, abs=2e-3)


def test_a_misspelled_combine_is_refused_on_the_vector_path_too():
    with pytest.raises(ValueError, match='is not one of'):
        _vector_total([_vector_box(80.0)], 'union')
