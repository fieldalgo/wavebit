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
``fieldwave._core.analytic.normal_vector_field_analytical`` (see ``test_nvf_vector.py``).
"""
import numpy as np
import pytest

import array_api_compat.numpy as xnp

from fieldwave._core.factorization import dfs2d_sampled, normal_vector_field


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
    """A 1D grating — the worst case for the ridge defect: the medial axis is a full column."""
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


def _projector_defect(nx, ny, fto=3):
    """``||C_xx + C_yy - I|| / ||I||`` in Fourier space — 0 iff ``nx^2 + ny^2 == 1``."""
    C_xx = dfs2d_sampled(xnp, (nx ** 2).astype(complex), fto, fto)
    C_yy = dfs2d_sampled(xnp, (ny ** 2).astype(complex), fto, fto)
    I = np.eye(C_xx.shape[0])
    return float(np.linalg.norm(C_xx + C_yy - I) / np.linalg.norm(I))


@pytest.mark.parametrize('name,geom', GEOMETRIES, ids=IDS)
def test_sdf_normal_field_is_unit_magnitude(name, geom):
    """The SDF field is a unit vector essentially everywhere, not only at the interface."""
    frac = _unit_fraction(*normal_vector_field(geom(100)))
    assert frac > 0.98, f'{name}: only {frac:.4f} of cells carry a unit normal'


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
