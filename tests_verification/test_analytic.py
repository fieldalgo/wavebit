"""
Verification tests for the ANALYTIC (staircase-free) modeling path.

Covers: Fourier-coefficient convention vs dfs2d, analytic formulas vs a fine
midpoint DFT, axis-aligned equivalence with the raster-NVF path, energy conservation,
cross-backend agreement, differentiability, and the convergence advantage over the
staircase path.
"""
import numpy as np
import pytest

import fieldwave
import array_api_compat.numpy as xnp
from fieldwave._core.analytic import (
    shape_ft as _shape_ft, laurent_conv_2d as _laurent_conv_2d,
    assemble_eps_coeffs as _assemble_eps_coeffs, to_conv_mat_analytic_nvf)
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
# 1. laurent_conv_2d builds the exact same ordering as dfs2d
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
        assert abs(total(ras, psi, fourier_type=2) - want) < 1e-3      # raster NVF
        assert abs(total(ras, psi, fourier_type=0) - want) < 2e-3      # raster DFS


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

    reference = total(ras, fourier_type=2)
    assert abs(total(shaped(L * P, 0.5 * P)) - reference) < 1e-3        # absolute: agrees
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
    """Overlapping shapes are added, not painted over — pinned because nothing says so at runtime.

    Each shape contributes ``(eps_shape - eps_base) * FT(indicator)``, so an overlap receives
    both. That is fine for the non-overlapping layers this path is built for and silently a
    different structure otherwise, which is worth a test rather than a surprise: someone moving
    a vector recipe over gets no error, just another answer.

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
    """The user-visible half of it: analytic parts from the raster, the vector path does not."""
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

    want = total(raster[None], fourier_type=0)
    vector = total([[n_base, [['rectangle', cx, 150.0, 120.0, 120.0, n, 0, 0, 0]
                              for cx, n in boxes]]])
    analytic = total([dict(n_base=n_base,
                           shapes=[dict(type='rectangle', cx=cx, cy=150.0, lx=120.0, ly=120.0,
                                        angle=0.0, n_index=n) for cx, n in boxes])])

    assert vector == pytest.approx(want, abs=1e-8)
    assert abs(analytic - want) > 1e-3, (analytic, want)
