"""
Verification tests for the ANALYTIC (staircase-free) modeling path.

Covers: the Fourier-coefficient convention vs ``fourier2d_pointwise``, the permittivity
convention under complex materials, analytic formulas vs a fine midpoint DFT, axis-aligned
equivalence with the raster-NVF path, energy conservation, cross-backend agreement,
differentiability, and the convergence advantage over the staircase path.
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


def conv_matrix_2d(coeff2d, half_width_u, half_width_v):
    return _conv_matrix_2d(xnp, coeff2d, half_width_u, half_width_v)


def sampled_coeffs(field, fx, fy):
    """The analytic path's spelling of the sampled transform's coefficients."""
    return _sampled_coeffs(xnp, field, fx, fy)


def assemble_eps_coeffs(shapes, background, half_width_u, half_width_v, period):
    return _assemble_eps_coeffs(xnp, shapes, background, half_width_u, half_width_v, period, xnp.complex128)


def to_conv_mat_smooth(shapes, background, half_width_u, half_width_v, period):
    """Direct-rule (eps, 1/eps) Laurent pair — the NVF builder's inputs."""
    eps_c, inv_c = assemble_eps_coeffs(shapes, background, half_width_u, half_width_v, period)
    return conv_matrix_2d(eps_c, half_width_u, half_width_v), conv_matrix_2d(inv_c, half_width_u, half_width_v)


# ----------------------------------------------------------------------
# 1. coefficient-builder conventions: ordering, and the complex-permittivity branch
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
    half_width_u, half_width_v = 2, 3

    M_raster = raw_dft(cell, half_width_u, half_width_v)
    M_analytic = conv_matrix_2d(sampled_coeffs(cell, half_width_u, half_width_v), half_width_u, half_width_v)

    gap = np.abs(M_raster - M_analytic).max()
    assert gap < 1e-12, (
        f'the raster and analytic spellings of the sampled transform differ by {gap:.2e}; '
        f'a correction was applied to one of them and not the other')


@pytest.mark.parametrize('where', ['background', 'material'])
def test_complex_permittivity_is_loss_not_gain(where):
    """``Im(eps) > 0`` must come out as ``Im(eps) > 0``, on both the background and the shape.

    The builder used to square an index on the way in and conjugated while doing it
    (``conj(n)**2``), which is the convention ``_core/conventions.py`` says Wavebit removed --
    so an absorbing medium became a gain medium, and only on this path. Nothing caught it
    because every case in this file was real, where ``conj(n)**2 == n**2``.

    The squaring is gone with the flip to permittivity, and the conjugation hazard is not: a
    material value still travels from the API through ``as_material`` into the coefficient sum,
    and any ``conj`` on that route flips loss to gain exactly as before. What changed is that
    the wrong answer is now one arithmetic step away instead of two, which is a reason to keep
    the probe rather than to retire it -- an absorbing medium is where the two signs are
    distinguishable at all.

    The check is on the DC coefficient, which for a uniform background *is* ``eps``, and for a
    shape covering a known area fraction is the exact ``eps`` average. Sign of the imaginary
    part is the whole claim; the magnitude is there so a stray ``abs`` could not pass it.
    """
    # A metal, stated as the permittivity the API now takes: this is (0.05 + 3.5j)**2, and
    # conj() would flip it to gain.
    eps_metal = -12.2475 + 0.35000000000000003j
    half_width_u = half_width_v = 2
    period = (1.0, 1.0)
    dc = (2 * half_width_v, 2 * half_width_u)                 # [ky+2*half_width_v, kx+2*half_width_u] at ky = kx = 0

    if where == 'background':
        eps_c, inv_c = assemble_eps_coeffs([], eps_metal, half_width_u, half_width_v, period)
        assert complex(eps_c[dc]) == pytest.approx(eps_metal, rel=1e-14)
        assert complex(inv_c[dc]) == pytest.approx(1.0 / eps_metal, rel=1e-14)
    else:
        # A square of side 1/2 covers a quarter of the cell, so the DC term is the exact
        # area-weighted mean and the shape's contribution is separable from the background's.
        shape = dict(type='rectangle', cx=0.5, cy=0.5, lx=0.5, ly=0.5, angle=0.0,
                     material=eps_metal)
        eps_c, inv_c = assemble_eps_coeffs([shape], 1.0, half_width_u, half_width_v, period)
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
    P, EPS_R, L, N = 0.77, 12.25, 0.5, 128       # eps 12.25 is the old n = 3.5
    cfg = dict(top=1.0, bot=2.25, theta=0.35, phi=0.4, period=(P, P),
               wavelength=0.777, thickness=(0.25,), truncation_order=(8, 8))

    xx, yy = ((np.mgrid[0:N, 0:N] + 0.5) / N)[::-1]
    ras = np.where((np.abs(xx - 0.5) < L / 2) & (np.abs(yy - 0.5) < L / 2), EPS_R, 1.0)[None]
    analytic = [dict(background=1.0, shapes=[dict(type='rectangle', cx=0.5 * P, cy=0.5 * P,
                                              lx=L * P, ly=L * P, angle=0.0, material=EPS_R)])]

    def total(structure, psi, **kw):
        res = wavebit.rcwa(backend=0, structure=structure, psi=psi, **cfg, **kw).solve()
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
    shape = dict(type='circle', cx=Px / 2.0, cy=Py / 2.0, lx=2 * r, material=4.0)
    normal_x, normal_y = _nvf_analytic(xnp, (Ny, Nx), [shape], (Px, Py), xnp.float64)

    X, Y = np.meshgrid((np.arange(Nx) + 0.5) / Nx * Px, (np.arange(Ny) + 0.5) / Ny * Py)
    dx, dy = X - Px / 2.0, Y - Py / 2.0
    rad = np.hypot(dx, dy)

    def radial_alignment(fx, fy):
        return (np.asarray(fx) * dx + np.asarray(fy) * dy) / np.where(rad > 0, rad, 1.0)

    assert np.abs(np.abs(radial_alignment(normal_x, normal_y)[rad > 0]) - 1.0).max() < 1e-12

    # And the same statement read off one pixel, exactly, where a swap is plainest. Both counts
    # are odd, so the disc's centre is a sample and row 64 sits *on* the centre line: due east
    # of it the normal is x_hat with nothing else in it. On the transposed cell that row is not
    # the centre line and this pixel is not even due east of the disc.
    i, j = Ny // 2, int((Px / 2.0 + 1.3 * r) / Px * Nx)
    assert abs(abs(float(normal_x[i, j])) - 1.0) < 1e-14
    assert abs(float(normal_y[i, j])) < 1e-14

    # The branch the solver actually takes: a basis in hand, and the centre left to default out
    # of Px / Py. Same disc, so the same radial statement holds and the same tolerance applies.
    centred = dict(type='circle', lx=2 * r, material=4.0)
    nx_l, ny_l = _nvf_analytic(xnp, (Ny, Nx), [centred], (Px, Py), xnp.float64,
                               ((Px, 0.0), (0.0, Py)))
    assert np.abs(np.abs(radial_alignment(nx_l, ny_l)[rad > 0]) - 1.0).max() < 1e-12


@pytest.mark.parametrize('psi', [np.pi / 2, 0.0], ids=['te', 'tm'])
def test_metal_rod_absorbs_end_to_end(psi):
    """A metal rod through the whole solve must absorb, not amplify.

    The coefficient test above pins the material convention; this pins that nothing
    downstream undoes it.
    ``conj(n)**2`` turned this rod into a gain medium, and a passive stack that returns
    ``R + T > 1`` is exactly what ``check_physical`` exists to catch — so the warning being
    absent is as much the assertion as the number is.

    Why this rather than a comparison against the same rod rastered: the two NVF schemes do not
    agree closely for a metal. Their gap here is ~3e-3 on ``sum(T)`` at ``truncation_order=8`` and does
    not shrink when the raster is refined (128 -> 256 moves it by 1%), because they are
    different factorizations that meet only as ``truncation_order -> inf``, and ``|eps|`` this large converges
    slowly. A tolerance loose enough to pass would be looser than the effect being guarded.
    The tight cross-path statement lives in ``test_lossy_media`` instead, on uniform layers
    where an absolute oracle exists.
    """
    P, L = 0.77, 0.5
    # (0.05 + 3.5j)**2: a metal rod, and conj() would make it gain.
    EPS_METAL = -12.2475 + 0.35000000000000003j
    structure = [dict(background=1.0, shapes=[dict(type='rectangle', cx=0.5 * P, cy=0.5 * P,
                                                   lx=L * P, ly=L * P, angle=0.0,
                                                   material=EPS_METAL)])]

    with warnings.catch_warnings():
        warnings.simplefilter('error', UnphysicalResultWarning)
        res = wavebit.rcwa(backend=0, structure=structure, psi=psi, top=1.0, bot=2.25,
                             theta=0.35, phi=0.4, period=(P, P), wavelength=0.777,
                             thickness=(0.25,), truncation_order=(8, 8)).solve()

    total = float(np.sum(np.asarray(res.R)).real + np.sum(np.asarray(res.T)).real)
    assert 0.5 < total < 0.95, total       # absorbing, and absorbing by a visible margin


def test_analytic_shape_parameters_are_absolute_lengths():
    """``cx`` / ``lx`` are lengths, not fractions of the period.

    Nothing else says so: the defaults (``Px / 2``) are the only statement in the code, and
    every other test here uses ``period = 1`` where the two readings coincide. Reading them as
    fractions builds a rod ``1/period`` times too big, which stays silent -- it just solves a
    different structure.
    """
    P, EPS_R, L, N = 0.77, 12.25, 0.5, 128       # eps 12.25 is the old n = 3.5
    cfg = dict(top=1.0, bot=2.25, theta=0.35, phi=0.0, period=(P, P),
               wavelength=0.777, thickness=(0.25,), truncation_order=(6, 6), psi=0.0)

    xx, yy = ((np.mgrid[0:N, 0:N] + 0.5) / N)[::-1]
    ras = np.where((np.abs(xx - 0.5) < L / 2) & (np.abs(yy - 0.5) < L / 2), EPS_R, 1.0)[None]

    def shaped(lx, cx):
        return [dict(background=1.0, shapes=[dict(type='rectangle', cx=cx, cy=cx, lx=lx, ly=lx,
                                              angle=0.0, material=EPS_R)])]

    def total(structure, **kw):
        res = wavebit.rcwa(backend=0, structure=structure, **cfg, **kw).solve()
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
    cfg = dict(top=1.0, bot=1.0, theta=np.deg2rad(5), phi=np.deg2rad(20),
               period=(1.0, 1.2), wavelength=1.5, thickness=(0.3, 0.2), psi=0.35 * np.pi)
    structure = [
        dict(background=1.0, shapes=[dict(type='ellipse', cx=0.5, cy=0.6, lx=0.5, ly=0.3,
                                      angle=np.deg2rad(40), material=5.76)]),
        dict(background=2.25, shapes=[dict(type='rectangle', cx=0.4, cy=0.5, lx=0.4, ly=0.5,
                                       angle=np.deg2rad(25), material=1.0),
                                  dict(type='circle', cx=0.8, cy=0.9, lx=0.25,
                                       material=9.0)]),
    ]
    for truncation_order in (5, 8):
        res = wavebit.rcwa(backend=0, structure=structure, truncation_order=(truncation_order, truncation_order), **cfg).solve()
        tot = np.array(res.R).real.sum() + np.array(res.T).real.sum()
        assert abs(tot - 1.0) < 1e-9


# ----------------------------------------------------------------------
# 5. cross-backend agreement (numpy == jax tight, numpy == torch loose)
# ----------------------------------------------------------------------
def test_cross_backend_agreement():
    cfg = dict(top=1.0, bot=1.0, theta=np.deg2rad(5), phi=np.deg2rad(20),
               period=(1.0, 1.2), wavelength=1.5, thickness=(0.3,), psi=0.35 * np.pi)
    structure = [dict(background=1.0, shapes=[dict(type='ellipse', cx=0.5, cy=0.6, lx=0.5, ly=0.3,
                                           angle=np.deg2rad(40), material=5.76),
                                      dict(type='circle', cx=0.3, cy=0.3, lx=0.2,
                                           material=9.0)])]

    def T(bk):
        return float(np.array(wavebit.rcwa(backend=bk, structure=structure, truncation_order=(6, 6), **cfg)
                              .solve().T).real.sum())

    tn, tj, tt = T(0), T(1), T(2)
    assert abs(tn - tj) < 1e-10
    assert abs(tn - tt) < 1e-7


# ----------------------------------------------------------------------
# 6. differentiability: gradient wrt shape params matches finite difference
# ----------------------------------------------------------------------
def _fd_grad_lx(shape_type, lx0=0.5):
    cfg = dict(top=1.0, bot=1.0, theta=0.0, phi=np.deg2rad(10),
               period=(1.0, 1.0), wavelength=1.5, thickness=(0.4,), psi=np.pi / 4)

    def T(lx):
        s = dict(type=shape_type, cx=0.5, cy=0.5, lx=lx, material=6.25,
                 **({} if shape_type == 'circle' else dict(ly=0.3, angle=0.4)))
        uc = [dict(background=1.0, shapes=[s])]
        return float(np.array(wavebit.rcwa(backend=0, structure=uc, truncation_order=(6, 6), **cfg)
                              .solve().T).real.sum())

    h = 1e-5
    return (T(lx0 + h) - T(lx0 - h)) / (2 * h), cfg


@pytest.mark.parametrize("shape_type", ["circle", "ellipse"])
def test_torch_gradient(shape_type):
    torch = pytest.importorskip("torch")
    fd, cfg = _fd_grad_lx(shape_type)
    lx = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    s = dict(type=shape_type, cx=0.5, cy=0.5, lx=lx, material=6.25,
             **({} if shape_type == 'circle' else dict(ly=0.3, angle=0.4)))
    res = wavebit.rcwa(backend=2, structure=[dict(background=1.0, shapes=[s])], truncation_order=(6, 6), **cfg).solve()
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
        s = dict(type=shape_type, cx=0.5, cy=0.5, lx=lx, material=6.25,
                 **({} if shape_type == 'circle' else dict(ly=0.3, angle=0.4)))
        r = wavebit.rcwa(backend=1, structure=[dict(background=1.0, shapes=[s])], truncation_order=(6, 6), **cfg).solve()
        return jnp.real(r.T).sum()

    g = float(jax.grad(loss)(jnp.array(0.5)))
    assert np.isfinite(g)
    assert abs(g - fd) < 1e-3


# ----------------------------------------------------------------------
# 7. analytic converges faster than the staircase path (tilted dielectric)
# ----------------------------------------------------------------------
def test_analytic_beats_staircase_tilted():
    cfg = dict(top=1.0, bot=1.0, theta=0.0, phi=None,
               period=(1.0, 1.0), wavelength=1.6, thickness=(0.6,), psi=0.0)
    EPS_R, ang, L, c = 12.25, np.deg2rad(30), 0.5, 0.5    # eps 12.25 is the old n = 3.5
    analytic = [dict(background=1.0, shapes=[dict(type='rectangle', cx=c, cy=c, lx=L, ly=L,
                                              angle=ang, material=EPS_R)])]
    # The block path's positional slot 5 is a material too, so one value serves both paths.
    stair = [[1.0, [['rectangle', c, c, L, L, EPS_R, ang, 24, 24]]]]

    def t0(res):
        d = np.array(res.T).real
        return d[d.shape[0] // 2, d.shape[0] // 2]

    ref = t0(wavebit.rcwa(backend=0, structure=analytic, truncation_order=(15, 15), **cfg).solve())
    a11 = t0(wavebit.rcwa(backend=0, structure=analytic, truncation_order=(11, 11), **cfg).solve())
    s11 = t0(wavebit.rcwa(backend=0, structure=stair, truncation_order=(11, 11), **cfg).solve())

    err_analytic = abs(a11 - ref)
    err_stair = abs(s11 - ref)
    assert err_analytic < 5e-3
    assert err_analytic < err_stair / 5  # analytic is far closer to the converged answer


CIRCLE_P, CIRCLE_EPS = 0.77, 12.25               # eps 12.25 is the old n = 3.5
CIRCLE_CFG = dict(period=[CIRCLE_P, CIRCLE_P], thickness=[0.25], theta=0.35, phi=0.4, psi=0.0,
                  wavelength=0.777, top=1.0, bot=2.25, truncation_order=[6, 6])


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
                 material=CIRCLE_EPS)
    got = wavebit.rcwa(backend=0, structure=[dict(background=1.0, shapes=[shape])],
                         **CIRCLE_CFG).solve()
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
                 ly=ly, material=CIRCLE_EPS)
    with pytest.raises(ValueError, match="'ly' would be dropped"):
        wavebit.rcwa(backend=0, structure=[dict(background=1.0, shapes=[shape])],
                       **CIRCLE_CFG).solve()


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
    period, eps_base = [300.0, 300.0], 1.0
    rods = [dict(type='rectangle', cx=130.0, cy=150.0, lx=120.0, ly=120.0, angle=0.0,
                 material=4.0),
            dict(type='rectangle', cx=190.0, cy=150.0, lx=120.0, ly=120.0, angle=0.0,
                 material=9.0)]

    eps_c, _ = _assemble_eps_coeffs(xnp, rods, eps_base, 3, 3, period, np.complex128)
    dc = complex(eps_c[2 * 3, 2 * 3]).real

    cell = period[0] * period[1]
    fraction = 120.0 * 120.0 / cell
    # Nothing is squared in the oracle any more: 4.0 and 9.0 are the materials as written,
    # and they name the same structure the pre-flip literals 2.0 and 3.0 did.
    superposed = eps_base + (4.0 - eps_base) * fraction + (9.0 - eps_base) * fraction
    overlap = 60.0 * 120.0 / cell
    painted = superposed - (9.0 - eps_base) * overlap

    assert dc == pytest.approx(superposed, rel=1e-12)
    assert dc != pytest.approx(painted, rel=1e-6)      # 2.76 against 2.52


def test_overlapping_shapes_move_the_answer_while_the_vector_path_does_not():
    """The user-visible half of it: analytic parts from the raster, the block path does not.

    This is why the layer has to be asked. ``combine='add'`` is the answer that keeps the
    superposition, and it is a different structure from the one the same recipe builds on the
    block path — which is what the two assertions measure.
    """
    period, eps_base = [300.0, 300.0], 1.0
    cfg = dict(period=period, thickness=[205.0], truncation_order=[6, 6], top=1.0, bot=1.0,
               wavelength=900.0, psi=np.pi / 2, theta=0.0, phi=0.0)
    boxes = [(130.0, 4.0), (190.0, 9.0)]

    n_pix = 900
    axis = (np.arange(n_pix) + 0.5) / n_pix * period[0]
    X, Y = np.meshgrid(axis, axis)
    raster = np.full((n_pix, n_pix), eps_base)
    for cx, eps in boxes:                                # later box paints over the earlier
        raster = np.where((np.abs(X - cx) <= 60.0) & (np.abs(Y - 150.0) <= 60.0),
                          eps, raster)

    def total(structure, **extra):
        return float(np.sum(np.asarray(
            wavebit.rcwa(backend=0, structure=structure, **extra, **cfg).solve().T)))

    want = total(raster[None], fourier_coefficients='sampled')
    vector = total([[eps_base, [['rectangle', cx, 150.0, 120.0, 120.0, eps, 0, 0, 0]
                                for cx, eps in boxes]]])
    analytic = total([dict(background=eps_base, combine='add',
                           shapes=[dict(type='rectangle', cx=cx, cy=150.0, lx=120.0, ly=120.0,
                                        angle=0.0, material=eps) for cx, eps in boxes])])

    assert vector == pytest.approx(want, abs=1e-8)
    assert abs(analytic - want) > 1e-3, (analytic, want)


# ------------------------------------------------------- the layer has to say how shapes combine

CHECK_CFG = dict(period=[300.0, 300.0], thickness=[205.0], truncation_order=[2, 2], top=1.0, bot=1.0,
                 wavelength=900.0, psi=np.pi / 2, theta=0.0, phi=0.0)


def rod(cx, cy=150.0, lx=120.0, ly=120.0, angle=0.0, material=4.0, type='rectangle'):
    s = dict(type=type, cx=cx, cy=cy, lx=lx, ly=ly, angle=angle, material=material)
    if type in ('circle', 'disk', 'square'):
        # A round rod has one length. The helper's `ly` default is not the caller asking for a
        # second one, and passing it on would be this helper inventing an ambiguity to hand to
        # `check_shape_spec` -- `rod(80.0, lx=60.0, type='circle')` would arrive as 60 by 120.
        del s['ly']
    return s


def solve(shapes, **layer):
    return wavebit.rcwa(backend=0, structure=[dict(background=1.0, shapes=shapes, **layer)],
                          **CHECK_CFG).solve()


def test_overlapping_shapes_are_refused_until_the_layer_says_how_to_combine():
    """The hazard this whole check exists for: the same recipe means two structures.

    A vector recipe moved onto the smooth path used to come back superposed, with nothing
    said. Now the layer is refused until it names the rule, and the message has to carry both
    shapes and both ways forward — a user who cannot act on the error is no better off than
    one who never saw it.
    """
    shapes = [rod(130.0, material=4.0), rod(190.0, material=9.0)]

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
    shapes = [rod(70.0, lx=100.0), rod(220.0, lx=100.0, material=9.0)]
    want = float(np.sum(np.asarray(solve(shapes).T)))

    for combine in ('add', 'paint'):
        assert float(np.sum(np.asarray(solve(shapes, combine=combine).T))) == want


def test_touching_is_not_overlapping():
    """Abutting shapes are the same structure either way, and designs are full of them.

    Two rectangles sharing an edge, two tangent circles, and a rod filling the period exactly
    all sit at a gap of zero. Reporting them would refuse ordinary geometry — a checkerboard,
    a full-width slab — for a difference of measure zero.
    """
    solve([rod(70.0), rod(190.0, material=9.0)])                       # edges meet at x = 130
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
    shapes = [rod(130.0), rod(190.0, material=9.0)]
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

#: 300x300 rods of permittivity 4 and 9, overlapping over 100x300 of a 1000x1000 cell. Summed
#: that is 1 + 3*0.09 + 8*0.09 = 1.99; painted, the earlier rod keeps only its unshared 60000,
#: giving 1 + 3*0.06 + 8*0.09 = 1.90. A 4.7% gap, and both are plausible numbers.
PAINT_RODS = [dict(type='rectangle', cx=400.0, cy=500.0, lx=300.0, ly=300.0, material=4.0),
              dict(type='rectangle', cx=600.0, cy=500.0, lx=300.0, ly=300.0, material=9.0)]
EPS_SUMMED, EPS_PAINTED = 1.99, 1.90


def _solver_mean_eps(combine, shapes=PAINT_RODS):
    """``<eps>`` as the solve reads it: the DC entry of ``ep_dir``, recovered from ``epz_i``."""
    from wavebit._core.analytic import to_conv_mat_smooth_layers
    layer = dict(background=1.0, shapes=shapes, combine=combine)
    mats = to_conv_mat_smooth_layers([layer], 6, 6, PAINT_PERIOD, normal_grid=256,
                                     complex_dtype=np.complex128, xp=xnp)
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
    return dict(dict(type='circle', cx=150.0, cy=150.0, lx=120.0, material=4.0), **over)


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
    dict(type='circle', cx=150.0, cy=150.0, l=120.0, material=4.0),   # `l` says the same
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
    shapes = [rod(20.0, lx=100.0), rod(280.0, lx=100.0, material=9.0)]   # meet through x = 0

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
                rod(190.0, 190.0, lx=120.0, ly=40.0, angle=-np.pi / 4, material=9.0)]
    crossing = [rod(120.0, 150.0, lx=160.0, ly=30.0, type='ellipse'),
                rod(225.0, 150.0, lx=30.0, ly=160.0, type='ellipse', material=9.0)]

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
        return [dict(background=1.0, shapes=[rod(cx), rod(190.0, material=9.0)])]

    frozen = wavebit.rcwa(backend=1, structure=layer(60.0), **CHECK_CFG).freeze_geometry()

    def total(cx):
        return jnp.sum(wavebit.rcwa(backend=1, structure=layer(cx),
                                      **CHECK_CFG).solve(geometry=frozen).T)

    assert float(jax.jit(total)(jnp.asarray(130.0))) > 0.0     # overlapping, and traced: allowed
    with pytest.raises(ValueError, match='overlap'):
        total(130.0)                                           # same layer, concrete: refused


# ------------------------------------------------------------------ the block path's half

VECTOR_CFG = dict(period=[300.0, 300.0], thickness=[205.0], truncation_order=[6, 6], top=1.0, bot=1.0,
                  wavelength=900.0, psi=np.pi / 2, theta=0.0, phi=0.0)


def _vector_box(cx, material=4.0):
    return ['rectangle', cx, 150.0, 120.0, 120.0, material, 0.0, 0, 0]


def _vector_total(objects, *combine):
    layer = [1.0, objects, *combine]
    return float(np.sum(np.asarray(
        wavebit.rcwa(backend=0, structure=[layer], **VECTOR_CFG).solve().T)))


def test_the_vector_path_refuses_the_superposition_it_cannot_draw():
    """The mirror of the analytic refusal, and the other half of making `combine` portable.

    Writing ``combine='add'`` and handing it to this path used to paint it, silently -- the
    same trap in the other direction. It cannot be drawn here: this path lays down a staircase
    of blocks and each block holds one index, so an overlap can only take one of them.
    """
    with pytest.raises(NotImplementedError, match='cannot add them'):
        _vector_total([_vector_box(140.0), _vector_box(180.0, 9.0)], 'add')


def test_the_vector_path_still_paints_what_it_always_painted():
    """`combine` is not a new tax: an unlabelled layer, and `'paint'`, are what this path did.

    Painting is this path's own rule and a raster of the same recipe agrees with it, so unlike
    the analytic side there is nothing surprising here to refuse.
    """
    objects = [_vector_box(140.0), _vector_box(180.0, 9.0)]
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
                   material=4.0) for cx in (80.0, 220.0)]
    analytic = float(np.sum(np.asarray(
        wavebit.rcwa(backend=0, structure=[dict(background=1.0, shapes=shapes)],
                       **VECTOR_CFG).solve().T)))
    assert analytic == pytest.approx(disjoint, abs=2e-3)


def test_the_vector_path_refuses_a_shape_larger_than_the_period():
    """The one overlap painting cannot reach: a shape running into its own image.

    An in-cell overlap is painted exactly here (the test above), and a shape that merely
    *crosses* the cell boundary is drawn exactly too (the test below). Past one full period the
    same machinery stops describing either: the block is gridded in unwrapped coordinates and
    its wrapped part is a **signed** segment, so the image adds to the shape instead of covering
    it, and painting a shape over its own image can only mean their union.

    The default, an explicit ``'paint'``, and ``'add'`` all have to say so -- the default is
    where it was silent.
    """
    big = ['rectangle', 150.0, 150.0, 340.0, 340.0, 4.0, 0.0, 0, 0]

    for combine in ((), ('paint',)):
        with pytest.raises(NotImplementedError, match='larger than the period'):
            _vector_total([big], *combine)
    with pytest.raises(NotImplementedError, match='larger than the period'):
        _vector_total([big], 'add')

    ellipse = ['ellipse', 150.0, 150.0, 340.0, 200.0, 4.0, 0.0, 0, 0]
    with pytest.raises(NotImplementedError, match='larger than the period'):
        _vector_total([ellipse])

    # A turned shape has to be judged by the shape and not by its bounding box. A 440 x 40
    # bar at 45 degrees does run into its own image -- diagonally, which is the only direction
    # that catches it -- and is refused.
    with pytest.raises(NotImplementedError, match='larger than the period'):
        _vector_total([['rectangle', 150.0, 150.0, 440.0, 40.0, 4.0, np.pi / 4, 0, 0]])


def test_a_shape_up_to_the_period_is_still_drawn_by_the_vector_path():
    """The refusal must stop exactly where painting stops working, and not one step earlier.

    Two cases sit right against it. A shape *exactly* the period is touching, not overlapping,
    and is the filled cell -- built here to the last digit of a uniform slab of the same
    material. A sub-period shape straddling the cell boundary is the case the signed-width
    reading exists for, and is unaffected by where it sits.
    """
    filled = _vector_total([['rectangle', 150.0, 150.0, 300.0, 300.0, 4.0, 0.0, 0, 0]])
    slab = float(np.sum(np.asarray(
        wavebit.rcwa(backend=0, structure=[[4.0, []]], **VECTOR_CFG).solve().T)))
    assert filled == pytest.approx(slab, abs=1e-12)

    straddling = [_vector_total([_vector_box(cx)]) for cx in (140.0, 20.0, 290.0, 0.0)]
    assert max(straddling) - min(straddling) < 1e-12

    # And the bound follows the shape, not its bounding box: the same bar 40 shorter misses
    # its own image in every direction while its axis-aligned box is 311 across a 300 cell,
    # so it is built -- and what gets built holds the two materials of the recipe and no
    # others, which is the property the refusal above protects.
    thin = wavebit.rcwa(backend=0, structure=[[1.0, [
        ['rectangle', 150.0, 150.0, 400.0, 40.0, 4.0, np.pi / 4, 0, 0]]]], **VECTOR_CFG)
    drawn = np.unique(np.round(
        np.real(np.asarray(wavebit._core.draw.slice_structure(thin, layer=0, n=301))), 6))
    assert drawn.tolist() == [1.0, 4.0]


def test_what_the_block_grid_built_instead_was_not_a_structure_at_all():
    """Why this is refused rather than documented as a second meaning of ``paint``.

    Read as a superposition the old build was self-consistent on one side only. This takes the
    edge list the modeler produces for the case in the report -- a 1.3-wide rod of eps 5.76 at
    the centre of a unit cell, whose columns come back as ``[-0.15, 1.15, 1.0]``, outside the
    cell -- and transforms ``eps`` and ``1/eps`` through it exactly as
    :func:`wavebit._core.factorization.fourier2d_integrated` does. Against the profile that
    edge list actually describes (cover count 1 or 2, so eps 5.76 or 10.52 along a row):

    * the ``eps`` coefficients are that profile's, on every mode and down to the accuracy of
      the quadrature it is being compared against;
    * the ``1/eps`` coefficients are not, and not by a little -- the mean reciprocal
      permittivity comes back **negative** on a layer whose permittivity is everywhere above 5.

    Li's rule feeds both halves into the same solve, so the number that came out was neither
    the union's nor the superposition's. That is the claim the docstring of
    :func:`wavebit._core.modeler.check_block_layer_combination` makes, and it is the reason
    there is nothing here to keep.
    """
    from wavebit._core.factorization import _fourier1d_integrated
    from wavebit._core.modeler import Modeling

    modeler = Modeling(period=[1.0, 1.0])
    cell, edges_u, _ = modeler.block_per_layer_numeric(
        [1.0, modeler.rectangle(0.5, 0.5, 1.3, 1.3, 5.76)])

    edges = np.real(np.asarray(edges_u)).reshape(-1)
    assert edges.min() < 0.0 and edges.max() > 1.0, edges      # the grid leaves the cell

    row = np.asarray(cell)[1:2]                                # one row straight through the rod
    assert sorted(np.real(row).reshape(-1).tolist()) == [1.0, 1.0, 5.76]

    truncation_order, period = 6, np.asarray(1.0, dtype=complex)
    got_eps = np.asarray(_fourier1d_integrated(xnp, edges_u, row, truncation_order, period))[0]
    got_inv = np.asarray(_fourier1d_integrated(xnp, edges_u, 1 / row, truncation_order, period))[0]

    # The profile that edge list describes, by dense quadrature: 1 + 4.76 * (cover count).
    s = (np.arange(400_000) + 0.5) / 400_000
    covers = sum((np.abs(s - 0.5 - m) <= 0.65).astype(float) for m in (-1, 0, 1))
    profile = 1.0 + 4.76 * covers
    modes = np.arange(-2 * truncation_order, 2 * truncation_order + 1)
    truth = lambda f: np.array([np.mean(f * np.exp(-1j * 2 * np.pi * m * s)) for m in modes])

    assert np.abs(got_eps - truth(profile)).max() < 1e-8      # quadrature, not the transform
    assert np.abs(got_inv - truth(1.0 / profile)).max() > 0.2

    dc = 2 * truncation_order
    assert np.real(truth(1.0 / profile)[dc]) > 0.0            # it must be: eps >= 5.76 > 0
    assert np.real(got_inv[dc]) < 0.0                         # what the grid handed the solve


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

    shapes = [dict(type='rectangle', cx=0.31, cy=0.44, lx=0.30, ly=0.22, angle=0.4,
                   material=12.25),
              dict(type='circle', cx=0.70, cy=0.62, lx=0.22, material=4.41),
              dict(type='ellipse', cx=0.20, cy=0.80, lx=0.18, ly=0.11, angle=1.1,
                   material=3.24)]
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


PROJ_KW = dict(period=(1.0, 1.0), thickness=(0.32,), wavelength=0.94, top=1.0, bot=2.1025,
               theta=0.2, phi=0.3, psi=0.0, truncation_order=(4, 4), normal_grid=512)

#: 3.4**2 and 2.7**2, spelled to the bit rather than tidied. These reach a cache key, and a
#: rounded literal is a different key from the one the arithmetic produces.
EPS_ROD, EPS_ROD_ALT = 11.559999999999999, 7.290000000000001


def _one(cx, material=EPS_ROD):
    return [dict(background=1.0, shapes=[
        dict(type='rectangle', cx=cx, cy=0.35, lx=0.30, ly=0.20, angle=0.3,
             material=material)])]


def _two(cx, material=EPS_ROD):
    return [dict(background=1.0, shapes=[
        dict(type='rectangle', cx=cx, cy=0.30, lx=0.30, ly=0.18, angle=0.3, material=material),
        dict(type='circle', cx=0.72, cy=0.80, lx=0.20, material=5.289999999999999)])]


@pytest.mark.parametrize('build', [_one, _two], ids=['one-shape', 'two-shapes'])
def test_the_projector_cache_never_changes_an_answer(build):
    """A hit and a miss have to return the same bits, or the cache is a second implementation.

    ``normal_projector_coeffs`` is the expensive half of the analytic NVF build -- 83% of a
    five-point sweep at ``normal_grid = 1024, truncation_order = 4`` for one shape, 93% for six -- and none of
    it reads a material value, so a material sweep rebuilds a field it already had. What makes
    the reuse safe rather than merely fast is that the *build* is position-free for a single
    shape: it is done at the centre of the cell and carried to ``(cx, cy)`` by a phase computed
    fresh every call.
    """
    import wavebit
    from wavebit._core import analytic as _an

    def total(cell):
        return np.asarray(wavebit.rcwa(backend=0, structure=cell, **PROJ_KW).solve().T)

    for cx in (0.30, 0.45, 0.60):
        for eps in (EPS_ROD, EPS_ROD_ALT):
            cell = build(cx, eps)
            _an._PROJECTOR_CACHE.clear()
            cold = total(cell)                       # nothing cached: the field is built
            assert np.array_equal(cold, total(cell))  # warm: the same entry, the same bits


def test_the_cache_holds_no_material_value_so_a_material_sweep_reuses_one_field():
    """The point of the key: the build reads geometry only, so only geometry may enter it."""
    import wavebit
    from wavebit._core import analytic as _an

    built = []
    original = _an.normal_vector_field_analytical
    _an.normal_vector_field_analytical = lambda *a, **k: (built.append(1), original(*a, **k))[1]
    try:
        _an._PROJECTOR_CACHE.clear()
        for eps in (11.559999999999999, 9.0, 6.760000000000001, 4.840000000000001, 3.24):
            wavebit.rcwa(backend=0, structure=_one(0.40, eps), **PROJ_KW).solve()
        assert len(built) == 1, f'the field was rebuilt {len(built)} times for one geometry'

        for cx in (0.41, 0.42, 0.43):                # and a *single* shape sliding is the same
            wavebit.rcwa(backend=0, structure=_one(cx), **PROJ_KW).solve()
        assert len(built) == 1, 'position enters through the phase, not through the build'

        wavebit.rcwa(backend=0, structure=_one(0.40, material=EPS_ROD)                 # a real change
                     , **PROJ_KW).solve()
        _an._PROJECTOR_CACHE.clear()
        wavebit.rcwa(backend=0, structure=_one(0.40), **PROJ_KW).solve()
        assert len(built) == 2, 'a cleared cache must rebuild'
    finally:
        _an.normal_vector_field_analytical = original


def test_a_traced_or_autograd_value_is_never_served_from_the_cache():
    """The gradient runs through the build, so a differentiated value must not key it.

    Two refusals, and only one of them is automatic. A JAX tracer cannot be turned into a key at
    all. A PyTorch tensor *can* -- the trap ``layer_needs_resolving`` documents -- so
    ``requires_grad`` is refused by hand. Pinned here on the quantity that shows it: sliding a
    lone shape cannot change any order's efficiency, so ``d(sum T)/dcx`` is zero, and a field
    frozen at the wrong place reports a force instead -- measured +0.046 when the projector
    coefficients were held fixed across the move, against a numerical floor of 1.7e-9 here. The
    floor is not the cache's doing: with the cache disabled outright the same process gives 0.0
    on its first ``jax.grad`` and 1.7e-9 on every one after, so ``ZERO_FORCE`` is set to separate
    zero from the failure rather than to chase that.
    """
    import wavebit
    from wavebit._core import analytic as _an
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp

    def total(cx):
        return jnp.sum(wavebit.rcwa(backend=1, structure=_one(cx), **PROJ_KW).solve().T).real

    ZERO_FORCE = 1e-6          # 1.7e-9 measured, 0.046 for the failure it has to catch
    _an._PROJECTOR_CACHE.clear()
    assert abs(float(jax.grad(total)(jnp.asarray(0.40)))) < ZERO_FORCE
    assert abs(float(jax.grad(total)(jnp.asarray(0.40)))) < ZERO_FORCE   # again, now warm

    def pair(cx):
        return jnp.sum(wavebit.rcwa(backend=1, structure=_two(cx), **PROJ_KW).solve().T).real

    h = 1e-5
    fd = (float(pair(jnp.asarray(0.40 + h))) - float(pair(jnp.asarray(0.40 - h)))) / (2 * h)
    assert float(jax.grad(pair)(jnp.asarray(0.40))) == pytest.approx(fd, rel=1e-3)

    torch = pytest.importorskip('torch')
    _an._PROJECTOR_CACHE.clear()
    cx = torch.tensor(0.40, requires_grad=True)
    cell = [dict(background=1.0, shapes=[
        dict(type='rectangle', cx=cx, cy=0.35, lx=0.30, ly=0.20, angle=0.3,
             material=EPS_ROD)])]
    torch.sum(wavebit.rcwa(backend=2, structure=cell, **PROJ_KW).solve().T).real.backward()
    assert abs(float(cx.grad)) < ZERO_FORCE

    lx = torch.tensor(0.30, requires_grad=True)      # a length *does* reach the build
    assert _an._projector_cache_key(
        None, [dict(type='rectangle', cx=0.4, cy=0.35, lx=lx, ly=0.2, angle=0.3)],
        (512, 512), (1.0, 1.0), 4, 4, 'complex128', None) is None


def test_the_cache_key_carries_everything_the_build_depends_on():
    """One geometry, several *builds* — the entries must not collide.

    The key is a promise that two calls sharing it produce the same coefficients. Geometry is the
    interesting half and the easy half to get right; the rest of the arguments are the half that
    is easy to leave out, and leaving one out is not a slow path but a wrong answer served from
    the wrong entry. Both of the ones that change the result independently of the shapes are
    swept here against a cache that has been cleared, which is the only reference that cannot
    itself be a cache hit.
    """
    import wavebit
    from wavebit._core import analytic as _an

    def total(truncation_order, normal_grid):
        kw = dict(PROJ_KW, truncation_order=(truncation_order, truncation_order), normal_grid=normal_grid)
        return np.asarray(wavebit.rcwa(backend=0, structure=_one(0.40), **kw).solve().T)

    _an._PROJECTOR_CACHE.clear()
    alone = {(truncation_order, ng): total(truncation_order, ng)                       # each built with nothing cached
             for truncation_order in (3, 4, 6) for ng in (256, 512)
             if not _an._PROJECTOR_CACHE.clear()}

    for (truncation_order, ng), expected in alone.items():                # now with all six entries resident
        assert np.array_equal(total(truncation_order, ng), expected), (truncation_order, ng)
    assert len(_an._PROJECTOR_CACHE) == 6, 'the six builds shared an entry'


def test_one_geometry_solved_on_three_backends_gets_three_entries():
    """A cached array belongs to the backend that made it, so the three must not share an entry.

    Nothing about the *shapes* changes between backends, so a key built from geometry alone would
    hand a NumPy array to a JAX trace and a JAX array to Torch. The three answers below agree to
    double precision — that is the point, and it is also why a collision would not announce
    itself as a type error every time. What actually separates them today is ``str(dtype)``, which
    spells the namespace into it (``numpy.complex128`` against ``torch.complex128``); the
    namespace goes in the key as well because relying on that is an accident, and this test pins
    the outcome rather than either mechanism.
    """
    import wavebit
    from wavebit._core import analytic as _an
    jax = pytest.importorskip('jax')
    torch = pytest.importorskip('torch')

    _an._PROJECTOR_CACHE.clear()
    answers = {}
    for name, backend in (('numpy', 0), ('jax', 1), ('torch', 2)):
        out = wavebit.rcwa(backend=backend, structure=_one(0.40), **PROJ_KW).solve().T
        answers[name] = np.asarray(out.detach() if hasattr(out, 'detach') else out, dtype=float)

    assert len(_an._PROJECTOR_CACHE) == 3, 'the backends shared one entry'
    assert answers['jax'] == pytest.approx(answers['numpy'], rel=1e-10, abs=1e-12)
    assert answers['torch'] == pytest.approx(answers['numpy'], rel=1e-10, abs=1e-12)


# ---------------------------------------------------------- the periodic fold on a skewed cell

def _skewed_ellipse_scene():
    """A hex cell and one ellipse that is asymmetric under every reflection this could get wrong.

    Deliberately **not** symmetric top-to-bottom or left-to-right, and not centred: a scene with
    a mirror in it cannot see a flip, and a centred one cannot see a translation. That blindness
    is not hypothetical here -- every 2D raster in this repository's external goldens was
    y-symmetric, which is exactly why an axis bug once survived all of them.

    It also has to be *large*. The fold only misreports where a neighbouring image's surface
    reaches into the sampled cell, so a small feature on a skewed cell agrees either way.
    """
    A = np.column_stack([[400.0, 0.0],
                         [400.0 * np.cos(np.pi / 3), 400.0 * np.sin(np.pi / 3)]])
    cx0, cy0 = 0.5 * (A[0, 0] + A[0, 1]), 0.5 * (A[1, 0] + A[1, 1])
    shape = dict(type='ellipse', cx=cx0 - 45.0, cy=cy0 + 40.0,
                 lx=385.0, ly=300.0, angle=0.5, material=5.76)
    return A, shape


def _band_misalignment(A, shape, n=512):
    """``1 - |n_analytic . n_raster|`` over the surface band, against the raster contour field.

    The arbiter is the *raster* normal field, which reaches its neighbours by tiling in index
    space and is therefore lattice-correct by construction. It is independent of the analytic
    path in every part except the shape it is handed -- which is what a golden could not be.
    """
    from scipy import ndimage
    from wavebit._core.factorization import normal_vector_field
    from wavebit._core.lattice import periods_of

    nx_a, ny_a = _nvf_analytic(np, (n, n), [shape], periods_of(A), np.float64, A)

    u = (np.arange(n) + 0.5) / n
    U, V = np.meshgrid(u, u)
    X, Y = U * A[0, 0] + V * A[0, 1], U * A[1, 0] + V * A[1, 1]
    ca, sa = np.cos(shape['angle']), np.sin(shape['angle'])
    inside = np.zeros((n, n), bool)
    for m in (-1, 0, 1):                       # the true images, at m*a1 + k*a2
        for k in (-1, 0, 1):
            dx = X - shape['cx'] - (m * A[0, 0] + k * A[0, 1])
            dy = Y - shape['cy'] - (m * A[1, 0] + k * A[1, 1])
            xr, yr = dx * ca + dy * sa, -dx * sa + dy * ca
            inside |= (xr / (shape['lx'] / 2)) ** 2 + (yr / (shape['ly'] / 2)) ** 2 <= 1.0

    nx_r, ny_r = normal_vector_field(np.where(inside, 5.76, 1.0), lattice=A, method='contour')
    band = (ndimage.binary_dilation(inside, iterations=3)
            & ~ndimage.binary_erosion(inside, iterations=3))
    return (1.0 - np.abs(nx_a * nx_r + ny_a * ny_r))[band]


def test_analytic_normal_field_folds_on_the_cells_own_lattice():
    """The nearest periodic image sits at ``m*a1 + n*a2``, not at ``(m*Px, n*Py)``.

    ``periods_of`` reports ``(|a1|, |a2|)``, so a fold written as ``d % Px`` reduces towards a
    point where no image is -- on a 60-degree cell it looks straight up while the neighbour is
    up and to the right. Until 2026-08-23 the analytic path did exactly that, and
    ``docs/oblique-lattice.md`` asserted the opposite ("Analytic shapes do not have this
    problem"), which is true of their *lengths* and was never true of their *fold*.

    Measured on this scene before the fix: 21% of the surface band more than 0.05 out, worst
    0.62 (about 39 degrees). It conserved energy to 1e-12 throughout.
    """
    band = _band_misalignment(*_skewed_ellipse_scene())
    assert band.max() < 0.05, f'worst surface-band misalignment {band.max():.4f}'
    assert band.mean() < 5e-3


def test_the_skew_is_what_that_test_is_sensitive_to():
    """The control: the same scene on a rectangular cell agreed before the fix and still does.

    Without this, a failure of the test above could be read as the scene being too demanding
    for the raster arbiter rather than as the fold being wrong.
    """
    A = np.diag([400.0, 400.0 * np.sin(np.pi / 3)])
    _, shape = _skewed_ellipse_scene()
    shape = dict(shape, cx=0.5 * A[0, 0] - 45.0, cy=0.5 * A[1, 1] + 40.0)
    band = _band_misalignment(A, shape)
    assert band.max() < 0.05


def test_an_axis_aligned_cell_keeps_the_rectangular_fold_bit_for_bit():
    """``diag(Lx, Ly)`` is the one case where the two folds are the same fold.

    So it keeps the cheaper spelling, and every result predating the lattice branch is
    unchanged to the bit -- not to a tolerance. Passing the lattice explicitly must agree with
    passing only the period, which is what says the branch is on the *shape* of the cell and
    not on which keyword the caller happened to use.
    """
    from wavebit._core.lattice import as_lattice, periods_of
    shapes = [dict(type='circle', cx=0.37, cy=0.29, lx=0.34, material=5.76),
              dict(type='rectangle', cx=0.72, cy=0.66, lx=0.3, ly=0.18, angle=0.4, material=3.0)]
    lat = as_lattice(period=[0.9, 1.4])
    a = _nvf_analytic(np, (96, 96), shapes, (0.9, 1.4), np.float64, None)
    b = _nvf_analytic(np, (96, 96), shapes, periods_of(lat), np.float64, lat)
    for got, want in zip(b, a):
        assert np.asarray(got).tobytes() == np.asarray(want).tobytes()


# ------------------------------------------------ the omitted centre is the cell's, on any cell

_CELLS = [
    ('square', np.diag([400.0, 400.0])),
    ('rect', np.diag([700.0, 1100.0])),
    ('tall', np.diag([180.0, 1300.0])),
    ('hex60', np.column_stack([[400.0, 0.0], [200.0, 400.0 * np.sqrt(3) / 2]])),
    ('shear-unimodular', np.column_stack([[400.0, 0.0], [400.0, 400.0]])),
    ('shear-shallow', np.column_stack([[400.0, 0.0], [350.0, 150.0]])),
    ('obtuse-105', np.column_stack([[400.0, 0.0], [520.0 * np.cos(np.radians(105)),
                                                   520.0 * np.sin(np.radians(105))]])),
    ('acute-32', np.column_stack([[900.0, 0.0], [260.0 * np.cos(np.radians(32)),
                                                 260.0 * np.sin(np.radians(32))]])),
    ('a1-off-axis', np.column_stack([[380.0, 120.0], [-90.0, 430.0]])),
    ('negative-det', np.column_stack([[400.0, 0.0], [200.0, -346.41]])),
]

_SHAPES = [
    dict(type='circle', lx=140.0),
    dict(type='ellipse', lx=190.0, ly=95.0, angle=0.63),
    dict(type='rectangle', lx=150.0, ly=80.0, angle=-1.1),
    dict(type='square', lx=120.0),
]


@pytest.mark.parametrize('cell_name, A', _CELLS, ids=[c[0] for c in _CELLS])
@pytest.mark.parametrize('shape', _SHAPES, ids=['circle', 'ellipse', 'rectangle', 'square'])
def test_an_omitted_centre_is_the_cell_centre_on_every_cell(cell_name, A, shape):
    """``shape_geometry``'s default must equal ``0.5*(a1 + a2)``, to the bit.

    It read ``(Px/2, Py/2)`` until 2026-08-23 -- the centre of ``diag(|a1|, |a2|)``, which is a
    different point the moment the cell is not axis-aligned, and on ``shear-shallow`` is not
    even inside the cell. The cases here are deliberately ugly: obtuse and acute cells, one
    whose ``a1`` is itself off-axis (so *neither* basis vector is x), and one with a negative
    determinant (a left-handed basis, which nothing forbids).

    The comparison is against an explicit centre rather than against a stored number, because a
    frozen expectation would have to be recomputed by the same formula it is checking.
    """
    from wavebit._core.analytic import shape_geometry
    from wavebit._core.lattice import periods_of

    Px, Py = periods_of(A)
    want = (0.5 * (A[0, 0] + A[0, 1]), 0.5 * (A[1, 0] + A[1, 1]))
    _, cx, cy, _, _, _ = shape_geometry(shape, Px, Py, A)
    assert (cx, cy) == want

    # a stated centre still wins, and each coordinate defaults on its own
    stated = dict(shape, cx=37.5)
    _, cx2, cy2, _, _, _ = shape_geometry(stated, Px, Py, A)
    assert cx2 == 37.5 and cy2 == want[1]
    stated = dict(shape, cy=-12.25)
    _, cx3, cy3, _, _, _ = shape_geometry(stated, Px, Py, A)
    assert cx3 == want[0] and cy3 == -12.25


@pytest.mark.parametrize('cell_name, A', _CELLS, ids=[c[0] for c in _CELLS])
def test_every_reader_of_the_default_moves_together(cell_name, A):
    """The three readers must agree about where an omitted centre is.

    ``shape_ft`` (the transform), ``boundary.shape_vertices`` (the polygon path) and
    ``draw._shape_inside`` (the picture) each resolve the shape through ``shape_geometry``.
    There is one default, so correcting it moves them together -- and the picture agreeing with
    the solve is the load-bearing half, since ``draw()`` is what a user checks a structure with.

    **Against the closed form, not against `shape_geometry`.** Asking the readers to match what
    that function returned would pass while every one of them is in the wrong place -- there is
    one default, so a wrong one moves them together just as neatly as a right one. That exact
    weakness is what ``tests/test_boundary.py:355`` was already recording about the old default,
    and a mutation run caught this test committing it on four oblique cells.
    """
    from wavebit._core import boundary as _b
    from wavebit._core import draw as _d
    from wavebit._core.lattice import periods_of

    Px, Py = periods_of(A)
    shape = dict(type='rectangle', lx=150.0, ly=80.0, angle=0.4)
    cx, cy = 0.5 * (A[0, 0] + A[0, 1]), 0.5 * (A[1, 0] + A[1, 1])

    verts = np.asarray(_b.shape_vertices(xnp, _b._concrete_shape(shape), (Px, Py), lattice=A),
                       dtype=float)
    assert np.allclose(verts.mean(axis=0), (cx, cy), atol=1e-9)

    # the picture: the sample nearest the resolved centre must be inside the shape
    X, Y = wavebit.grid((129, 129), lattice=A)
    X, Y = np.asarray(X), np.asarray(Y)
    cover = np.asarray(_d._shape_inside(shape, X, Y, A, Px, Py))
    k = np.argmin((X - cx) ** 2 + (Y - cy) ** 2)
    assert cover.ravel()[k] >= 1


@pytest.mark.parametrize('cell_name, A', [c for c in _CELLS if c[0] in
                                          ('square', 'hex60', 'shear-shallow', 'obtuse-105')],
                         ids=['square', 'hex60', 'shear-shallow', 'obtuse-105'])
@pytest.mark.parametrize('factorization', ['nvf', 'boundary'])
def test_omitting_the_centre_solves_what_stating_it_solves(cell_name, A, factorization):
    """End to end: the two spellings must be one structure, not two.

    The unit test above pins the resolved number; this pins that nothing downstream
    reinterprets it. Both shape paths, because they resolve the shape separately and
    ``shape_geometry`` exists precisely because they once disagreed.

    Note what this can*not* be: an invariance under ``a2' = a2 + a1``. That re-bases the same
    point lattice onto a *different fundamental cell*, so "the centre of the cell" legitimately
    moves with it -- ``square`` and ``shear-unimodular`` here have centres ``(200, 200)`` and
    ``(400, 200)``. A default defined against the basis cannot be basis-independent, and a test
    demanding that would be demanding the wrong thing.
    """
    centre = dict(cx=float(0.5 * (A[0, 0] + A[0, 1])), cy=float(0.5 * (A[1, 0] + A[1, 1])))
    fixed = dict(type='circle', cx=70.0, cy=95.0, lx=90.0, material=5.76)

    def solve(second):
        kw = dict(backend=0, wavelength=900.0, lattice=A, truncation_order=[6, 6],
                  top=1.0, bot=1.0, theta=0.11, phi=0.37, psi=0.29, thickness=[300.0],
                  structure=[dict(background=1.0, combine='add', shapes=[fixed, second])])
        if factorization == 'boundary':
            kw['structure'][0]['combine'] = 'paint'
        else:
            kw['factorization'] = 'nvf'
        res = wavebit.rcwa(**kw).solve()
        return np.asarray(res.R), np.asarray(res.T)

    bare = dict(type='ellipse', lx=130.0, ly=70.0, angle=0.5, material=5.76)
    a_r, a_t = solve(bare)
    b_r, b_t = solve(dict(bare, **centre))
    assert a_r.tobytes() == b_r.tobytes() and a_t.tobytes() == b_t.tobytes()


# --------------------------------------------------------------------- wavebit.grid

@pytest.mark.parametrize('cell_name, A', _CELLS, ids=[c[0] for c in _CELLS])
def test_grid_places_samples_on_the_cell_not_a_rectangle(cell_name, A):
    """``grid`` returns ``r = (i+0.5)/n_u * a1 + (j+0.5)/n_v * a2``, in Cartesian coordinates.

    Checked against the closed form rather than against ``sample_grid``, which is the function
    it wraps -- comparing a wrapper to its callee proves only that the call happened.
    """
    n_v, n_u = 7, 11
    X, Y = wavebit.grid((n_v, n_u), lattice=A)
    u = (np.arange(n_u) + 0.5) / n_u
    v = (np.arange(n_v) + 0.5) / n_v
    U, V = np.meshgrid(u, v)
    assert np.allclose(np.asarray(X), U * A[0, 0] + V * A[0, 1], atol=1e-12)
    assert np.allclose(np.asarray(Y), U * A[1, 0] + V * A[1, 1], atol=1e-12)


def test_grid_period_and_lattice_are_two_spellings_of_one_cell():
    """``period=[Lx, Ly]`` must land exactly where ``diag(Lx, Ly)`` does, and a lone period is square."""
    a = [np.asarray(v).tobytes() for v in wavebit.grid((9, 13), period=[700.0, 1100.0])]
    b = [np.asarray(v).tobytes() for v in wavebit.grid((9, 13), lattice=np.diag([700.0, 1100.0]))]
    assert a == b
    c = [np.asarray(v).tobytes() for v in wavebit.grid((9, 9), period=[640.0])]
    d = [np.asarray(v).tobytes() for v in wavebit.grid((9, 9), lattice=np.diag([640.0, 640.0]))]
    assert c == d


def test_grid_refuses_what_it_cannot_answer():
    with pytest.raises(ValueError, match='rows, columns'):
        wavebit.grid((8,), period=[1.0])
    with pytest.raises(ValueError, match='rows, columns'):
        wavebit.grid((8, 8, 8), period=[1.0])
    with pytest.raises(ValueError, match='positive'):
        wavebit.grid((0, 8), period=[1.0])
    with pytest.raises(ValueError, match='one of period or lattice'):
        wavebit.grid((8, 8))
    with pytest.raises(ValueError, match='backend'):
        wavebit.grid((8, 8), period=[1.0], backend=9)
    with pytest.raises(ValueError, match='parallel'):
        wavebit.grid((8, 8), lattice=np.column_stack([[1.0, 0.0], [2.0, 0.0]]))


def test_a_structure_built_through_grid_is_the_one_that_gets_solved():
    """The point of the helper: coordinates in, and the layout follows on its own.

    Built through ``grid`` and built by hand from the same closed form must be the same raster
    and therefore the same answer -- on a skewed cell, where hand-rolling is where the
    ``u*a1 + v*a2`` step is easy to omit.
    """
    A = np.column_stack([[400.0, 0.0], [200.0, 400.0 * np.sqrt(3) / 2]])
    n = 96
    cx, cy = 0.5 * (A[0, 0] + A[0, 1]), 0.5 * (A[1, 0] + A[1, 1])

    X, Y = wavebit.grid((n, n), lattice=A)
    eps_helper = np.where((np.asarray(X) - cx) ** 2 + (np.asarray(Y) - cy) ** 2 < 150.0 ** 2, 5.76, 1.0)

    t = (np.arange(n) + 0.5) / n
    U, V = np.meshgrid(t, t)
    Xh, Yh = U * A[0, 0] + V * A[0, 1], U * A[1, 0] + V * A[1, 1]
    eps_hand = np.where((Xh - cx) ** 2 + (Yh - cy) ** 2 < 150.0 ** 2, 5.76, 1.0)

    assert eps_helper.tobytes() == eps_hand.tobytes()

    def solve(eps):
        s = wavebit.rcwa(backend=0, wavelength=900.0, lattice=A, truncation_order=[6, 6],
                         top=1.0, bot=1.0, theta=0.0, phi=0.0, psi=0.0, thickness=[300.0],
                         structure=eps[None], factorization='nvf')
        return float(np.sum(s.solve().R) + np.sum(s.solve().T))
    assert abs(solve(eps_helper) - 1.0) < 1e-9
