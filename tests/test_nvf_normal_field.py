"""The normal field's defining invariant — and the record of why SDF is the only method.

``nvf_conv_matrices`` derives the Li/Schuster factorization from a normal field that is
**unit magnitude throughout the unit cell**, not just at the interface. That assumption is
load-bearing: it is what makes ``C_xx + C_yy = <nx^2> + <ny^2> = I``, so the projector splits E
into normal and tangential parts and nothing else. Nothing else in the suite pins it down, so a
normal-field change that quietly stopped producing unit vectors would still pass every parity
and energy test while silently degrading the factorization.

The invariant is stated as a *convergence* property rather than a fixed tolerance, because the
discrete SDF field genuinely has zero gradient on the medial axis of the distance map (the
ridge midway between two interfaces). That ridge is a lower-dimensional set, so its share of
the cell — and with it the defect ``||C_xx + C_yy - I||`` — falls as the raster refines.

The legacy ``on_numpy/emsolver/nvf.py`` offered ``method='sobel'`` alongside ``'sdf'``. It was
dropped in the core rewrite, and ``test_sobel_style_gradient_field_violates_the_invariant``
records why that was correct rather than an oversight: a raw ``grad(eps)`` field is nonzero
only in the one-pixel band at the interface, so its defect runs the *wrong way* under
refinement — toward 1, as the band becomes an ever smaller share of the cell. It was never
reachable from the solver (the legacy entry point ``to_conv_mat_raster_nvf`` had no ``method``
parameter) and only ever appeared in a tutorial comparison illustrating that SDF is better.

The third legacy method, ``'analytical'``, was *not* dropped: it lives on as
``wavebit._core.analytic.normal_vector_field_analytical`` (see ``test_nvf_vector.py``).
"""
import numpy as np
import pytest

import array_api_compat.numpy as xnp

from wavebit._core.factorization import (fourier2d_piecewise_constant, fourier2d_pointwise,
                                           normal_vector_field, nvf_conv_matrices)


def _circle(N, radius_frac=0.3, n_in=2.0, n_out=1.0):
    gy, gx = np.ogrid[:N, :N]
    eps = np.full((N, N), n_out ** 2, dtype=complex)
    eps[np.sqrt((gx - N / 2) ** 2 + (gy - N / 2) ** 2) <= radius_frac * N] = n_in ** 2
    return eps


def _rect(N, half=0.25, n_in=2.0, n_out=1.0):
    eps = np.full((N, N), n_out ** 2, dtype=complex)
    lo, hi = int(N * (0.5 - half)), int(N * (0.5 + half))
    eps[lo:hi, lo:hi] = n_in ** 2
    return eps


def _stripe(N, n_in=2.0, n_out=1.0):
    """A 1D grating — the worst case for the ridge defect: **two** full medial columns.

    One inside the stripe, one outside it. The outside ridge exists because the cell is
    periodic: the gap between the stripe's right face and the *next copy's* left face has a
    midline of its own. Before the distance transform was made periodic (2026-08-17) only the
    inside ridge appeared, because the unwrapped transform let the distance grow monotonically
    to the array border instead of turning over halfway to the next copy — so this geometry's
    zero-gradient share was understated by exactly one column.
    """
    eps = np.full((N, N), n_out ** 2, dtype=complex)
    eps[:, N // 4:N // 2] = n_in ** 2
    return eps


GEOMETRIES = [('circle', _circle), ('rectangle', _rect), ('stripe', _stripe)]
IDS = [g[0] for g in GEOMETRIES]


def _sobel_normal_field(eps):
    """The dropped legacy ``method='sobel'`` field, kept here only as the negative control.

    Equivalent to the deleted ``on_numpy/emsolver/nvf.py`` branch: a Sobel gradient of ``eps``
    with periodic wrap, normalized where the gradient is non-negligible.
    """
    p = np.pad(eps.real, ((1, 1), (1, 1)), mode='wrap')
    gx = (p[1:-1, 2:] - p[1:-1, :-2]) * 2 / 8.0 + (p[:-2, 2:] - p[:-2, :-2]) / 8.0 \
        + (p[2:, 2:] - p[2:, :-2]) / 8.0
    gy = (p[2:, 1:-1] - p[:-2, 1:-1]) * 2 / 8.0 + (p[2:, :-2] - p[:-2, :-2]) / 8.0 \
        + (p[2:, 2:] - p[:-2, 2:]) / 8.0
    norm = np.hypot(gx, gy)
    mask = norm > 1e-8
    nx, ny = np.zeros_like(gx), np.zeros_like(gy)
    nx[mask] = gx[mask] / norm[mask]
    ny[mask] = gy[mask] / norm[mask]
    return nx, ny


def _unit_fraction(nx, ny):
    return float(np.isclose(np.hypot(nx, ny), 1.0, atol=1e-8).mean())


def _projector_defect(nx, ny, truncation_order=3):
    """``||C_xx + C_yy - I|| / ||I||`` in Fourier space — 0 iff ``nx^2 + ny^2 == 1``."""
    C_xx = fourier2d_pointwise(xnp, (nx ** 2).astype(complex), truncation_order, truncation_order)
    C_yy = fourier2d_pointwise(xnp, (ny ** 2).astype(complex), truncation_order, truncation_order)
    I = np.eye(C_xx.shape[0])
    return float(np.linalg.norm(C_xx + C_yy - I) / np.linalg.norm(I))


@pytest.mark.parametrize('name,geom', GEOMETRIES, ids=IDS)
def test_sdf_normal_field_is_unit_magnitude(name, geom):
    """The SDF field is a unit vector essentially everywhere, not only at the interface.

    The bound is stated as *a few medial columns' worth* rather than as a bare fraction,
    because that is the shape of the only legitimate exception. Zero-gradient pixels live on
    the medial axis of the distance map, and for the worst geometry here — a stripe, whose
    medial set is straight — that axis is whole columns: two of them in a periodic cell (see
    :func:`_stripe`), so ``2 * N`` of ``N**2`` pixels, i.e. ``1 - 2/N``. A fixed 0.98 encoded
    ``N = 100`` and one column, both by accident; ``1 - 3/N`` allows three columns and tightens
    as the raster refines, which is the property actually being claimed.

    It is also parity-sensitive in a way a fixed number hides: at ``N = 100`` the stripe's two
    ridges land exactly on pixel columns (200 non-unit pixels) and at ``N = 200`` they fall
    between them (zero). Curved geometries never approach the bound — circle and rectangle sit
    at 4 pixels total, whatever ``N``.
    """
    N = 100
    frac = _unit_fraction(*normal_vector_field(geom(N)))
    floor = 1.0 - 3.0 / N
    assert frac > floor, (f'{name}: only {frac:.4f} of cells carry a unit normal, below the '
                          f'{floor:.4f} that three medial columns would leave')


@pytest.mark.parametrize('name,geom', GEOMETRIES, ids=IDS)
def test_sdf_projector_identity_converges_under_refinement(name, geom):
    """``C_xx + C_yy -> I``: the identity the NVF factorization is derived from."""
    coarse = _projector_defect(*normal_vector_field(geom(100)))
    fine = _projector_defect(*normal_vector_field(geom(200)))

    assert coarse < 0.06, f'{name}: defect {coarse:.4f} too large even at N=100'
    assert fine <= coarse, f'{name}: defect grew under refinement, {coarse:.5f} -> {fine:.5f}'
    assert fine < 1e-3, f'{name}: defect {fine:.5f} does not approach the identity at N=200'


@pytest.mark.parametrize('name,geom', GEOMETRIES, ids=IDS)
def test_sobel_style_gradient_field_violates_the_invariant(name, geom):
    """Why ``method='sobel'`` did not survive the port.

    A raw ``grad(eps)`` field is nonzero only in the one-pixel band at the interface, so it is
    not a unit normal field over the cell. Its projector collapses toward 0 instead of ``I``,
    and — unlike the SDF ridge defect — refining the raster makes it *worse*, because the band
    shrinks relative to the cell. That degenerates the factorization back toward the plain
    direct rule: a different, weaker scheme wearing the NVF name, not an alternative normal
    field. This test fails the day someone reintroduces it as a supported method without
    addressing that.
    """
    coarse_nx, coarse_ny = _sobel_normal_field(geom(100))
    fine_nx, fine_ny = _sobel_normal_field(geom(200))

    frac = _unit_fraction(coarse_nx, coarse_ny)
    assert frac < 0.25, f'{name}: sobel field is unit over {frac:.3f} of the cell, unexpected'

    coarse = _projector_defect(coarse_nx, coarse_ny)
    fine = _projector_defect(fine_nx, fine_ny)
    assert coarse > 0.5, f'{name}: sobel projector defect only {coarse:.4f}'
    assert fine > coarse, f'{name}: sobel defect improved under refinement, {coarse} -> {fine}'

    # ... and it is decisively worse than the field that shipped.
    sdf = _projector_defect(*normal_vector_field(geom(100)))
    assert sdf < coarse / 10, f'{name}: sdf {sdf:.4g} not decisively better than sobel {coarse:.4g}'


def test_the_projector_is_built_from_the_sampled_transform():
    """``nvf_conv_matrices`` must take the normal field as samples, not as a staircase.

    The projector grids are a continuous unit-vector field evaluated at grid points, so their
    coefficients are the plain DFT; applying the piecewise-constant pixel correction there
    asserts a shape the data does not have. Which build is used was a boolean argument until
    2026-08-12 and is now the choice of entry point, but neither spelling is self-checking:
    the projector identity ``C_xx + C_yy = I`` that the rest of this file pins holds under
    *either* build, so nothing above would notice the swap.

    Both halves matter. The first says the sampled build is what runs. The second says that
    is a decision rather than a coincidence -- if the two builds ever agreed here, the first
    assertion would be pinning nothing.
    """
    eps = _circle(40)
    nx, ny = normal_vector_field(eps)
    truncation_order = 3

    got = np.asarray(nvf_conv_matrices(xnp, eps, nx, ny, truncation_order, truncation_order)[0])

    def assembled(build):
        c_xx, c_xy, c_yy = (np.asarray(build(f.astype(complex)))
                            for f in (nx ** 2, nx * ny, ny ** 2))
        return c_xx, c_xy, c_yy

    sampled = assembled(lambda f: fourier2d_pointwise(xnp, f, truncation_order, truncation_order))
    staircase = assembled(lambda f: fourier2d_piecewise_constant(xnp, f, truncation_order, truncation_order, inverse_rule=None))

    # The projector that shipped is the sampled one, to the last bit of the assembly.
    from wavebit._core.factorization import _nvf_assemble
    want = np.asarray(_nvf_assemble(xnp, *(np.asarray(fourier2d_piecewise_constant(
        xnp, e.astype(complex), truncation_order, truncation_order, inverse_rule=None)) for e in (eps, 1.0 / eps)), *sampled)[0])
    assert np.allclose(got, want, atol=1e-12, rtol=0), np.abs(got - want).max()

    # ... and the other build is far enough away that the check above has teeth.
    #
    # 1e-4 rather than the 1e-3 this carried until 2026-08-16. The two builds used to be
    # separated by *two* factors, the box transform and the half-pixel cell-centre phase, and
    # only the first of those is a real difference between a staircase and a point sample. The
    # phase belongs to both and is now applied to both
    # (:func:`~wavebit._core.factorization._half_pixel_phase`), which closed a genuine defect --
    # the normal field sat half a pixel off the permittivity -- and left the legitimate gap of
    # 4.8e-04 here. Lowered because the separation shrank for a good reason, not to accommodate
    # a regression: the remaining gap is the ``sinc`` alone, which is what this is checking for.
    gap = max(np.abs(a - b).max() for a, b in zip(sampled, staircase))
    assert gap > 1e-4, (
        f'the two builds of the projector differ by only {gap:.2e}, so the assertion above '
        f'no longer distinguishes them')
