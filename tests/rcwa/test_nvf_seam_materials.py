"""Material continuity and the geometric meaning of the seam study's opening.

These gates cross equal moduli with unequal permittivities; an equality ramp
between positive real materials alone cannot detect a modulus-ranking switch.
"""

from wavebit.compat import legacy as _legacy_wavebit
import numpy as np
import pytest

import wavebit
from wavebit.rcwa._engine import parametric as P
from wavebit.lattice import as_lattice


CELL = as_lattice(period=(700., 700.))
MATERIALS = (2. + np.sqrt(12.) * 1j, -4., 4j)
KW = dict(lattice=CELL, wavelength=900., top=1., bot=2.25, theta=.2,
          phi=.13, psi=.2, thickness=(300.,), truncation_order=(2, 2),
          shape_nvf_grid_size=128, factorization='nvf')


def pair(right, left=4.):
    return [dict(type='rectangle', cx=-150., cy=0., lx=300., ly=300., material=left),
            dict(type='rectangle', cx=150., cy=0., lx=300., ly=280., material=right)]


def structure(right):
    return [dict(background=1., shapes=pair(right))]


def total(right, **kwargs):
    P._PROJECTOR_CACHE.clear()
    return float(np.sum(_legacy_wavebit.rcwa(backend='numpy', structure=structure(right),
                                     **dict(KW, **kwargs)).solve().T))


@pytest.mark.parametrize('right', MATERIALS)
def test_equal_moduli_are_continuous_in_coefficients_and_transmission(right):
    def change(h):
        fields = [P.normal_projector_coeffs(np, pair(right * (1 + s*h)),
                  (128, 128), CELL, 2, 2, np.complex128) for s in (-1, 1)]
        return (max(np.max(np.abs(a-b)) for a, b in zip(*fields)),
                abs(total(right*(1+h)) - total(right*(1-h))))
    big, small = change(1e-6), change(1e-8)
    # A 100x smaller perturbation must reduce the change; a fixed jump cannot pass.
    assert small[0] < .02 * big[0] + 1e-13
    assert small[1] < .02 * big[1] + 1e-12
    assert small[0] < 1e-7 and small[1] < 1e-6


@pytest.mark.parametrize('right', MATERIALS)
def test_equal_moduli_do_not_choose_the_listing_order(right):
    # The 300-unit anchor offset is 60 whole pixels at N=140, removing sampling jitter.
    kw = dict(KW, shape_nvf_grid_size=140)
    values = [_legacy_wavebit.rcwa(backend='numpy', structure=[dict(background=1., shapes=shapes)],
                           **kw).solve().T for shapes in (pair(right), pair(right)[::-1])]
    np.testing.assert_allclose(*values, rtol=0., atol=2e-11)


@pytest.mark.parametrize('right', MATERIALS)
def test_complex_seam_is_the_exact_normal_and_has_a_hermitian_rank_one_projector(right):
    shapes = pair(right)
    # Turn a vertical seam into a horizontal one: a hard-coded x-axis tie cannot pass.
    shapes = [dict(s, cx=s['cy'], cy=s['cx'], lx=s['ly'], ly=s['lx']) for s in shapes]
    x = np.linspace(300., 400., 21)
    nx, ny = P._piece_field(np, x, np.full_like(x, 350.), shapes,
                           P.plan_gamma(shapes, CELL), None, CELL, np.float64)
    np.testing.assert_allclose(np.abs(nx), 0., atol=1e-14)
    np.testing.assert_allclose(np.abs(ny), 1., atol=1e-14)
    nx, ny = P.normal_vector_field_analytical(np, (65, 71), shapes, np.float64, CELL)
    n = np.stack([nx, ny], axis=-1)
    nvf_projector = n[..., :, None] * n[..., None, :].conj()
    np.testing.assert_allclose(nvf_projector, nvf_projector.swapaxes(-1, -2).conj(), atol=1e-14)
    np.testing.assert_allclose(nvf_projector @ nvf_projector, nvf_projector, atol=1e-14)
    np.testing.assert_allclose(np.trace(nvf_projector, axis1=-2, axis2=-1), 1., atol=1e-14)


def test_tiling_keeps_the_complex_seam_and_its_precision(monkeypatch):
    for dtype in (np.float32, np.float64):
        shapes = pair(MATERIALS[0])
        monkeypatch.setattr(P, '_tiles_pay', lambda *args: False)
        full = P.normal_vector_field_analytical(np, (73, 81), shapes, dtype, CELL)
        monkeypatch.setattr(P, '_tiles_pay', lambda *args: True)
        tiled = P.normal_vector_field_analytical(np, (73, 81), shapes, dtype, CELL)
        assert np.max(np.abs(full[0].imag)) > .01
        for a, b in zip(full, tiled):
            assert a.dtype == b.dtype
            np.testing.assert_array_equal(a, b)


def test_a_cached_real_spelling_does_not_change_the_complex_component_contract():
    P._PROJECTOR_CACHE.clear()
    def coefficients(right):
        return P.normal_projector_coeffs(np, pair(right), (64, 64), CELL, 2, 2, np.complex128)
    real = coefficients(9.)
    warm = coefficients(9.+0j)
    P._PROJECTOR_CACHE.clear()
    cold = coefficients(9.+0j)
    assert len(real) == 3 and len(warm) == len(cold) == 4
    for a, b in zip(warm, cold):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(real, cold):
        np.testing.assert_allclose(a, b, rtol=0., atol=1e-14)


def test_fourier_components_keep_hermitian_symmetry_and_unit_trace():
    P._PROJECTOR_CACHE.clear()
    xx, xy, yy, yx = P.normal_projector_coeffs(np, pair(MATERIALS[0]), (128, 128),
                                             CELL, 2, 2, np.complex128)
    for a, b in ((xx, xx), (yy, yy), (xy, yx)):
        np.testing.assert_allclose(a, b[::-1, ::-1].conj(), rtol=0., atol=1e-14)
    unit = np.zeros_like(xx)
    unit[4, 4] = 1.
    np.testing.assert_allclose(xx+yy, unit, rtol=0., atol=1e-14)
    assert np.max(np.abs(xy-yx)) > 1e-4


def test_jitted_material_gradient_crosses_equal_moduli_and_material_equality():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    _legacy_wavebit.enable_x64()
    for right in (MATERIALS[0], -4., 4., 4.+0j):
        solver = _legacy_wavebit.rcwa(backend='jax', structure=structure(right), **KW)
        geometry = solver.freeze_geometry()
        def response(scale):
            return jnp.sum(solver.solve(structure=structure(right*scale),
                                        geometry=geometry).T).real
        eager = float(response(jnp.asarray(1.)))
        assert float(jax.jit(response)(1.)) == pytest.approx(eager, abs=2e-11)
        assert eager == pytest.approx(total(right), abs=2e-11)
        grad = float(jax.jit(jax.grad(response))(1.))
        secant = (float(response(1.+1e-6))-float(response(1.-1e-6)))/2e-6
        assert np.isfinite(grad)
        assert grad == pytest.approx(secant, rel=2e-4, abs=1e-7)
        jax.clear_caches()


def test_torch_material_gradient_matches_the_same_secant():
    torch = pytest.importorskip('torch')
    right = MATERIALS[0]
    scale = torch.tensor(1., dtype=torch.float64, requires_grad=True)
    response = _legacy_wavebit.rcwa(backend='torch', structure=structure(right*scale), **KW).solve().T.sum()
    response.backward()
    assert float(response.detach()) == pytest.approx(total(right), abs=2e-11)
    secant = (total(right*(1.+1e-6))-total(right*(1.-1e-6)))/2e-6
    assert float(scale.grad) == pytest.approx(secant, rel=2e-4, abs=1e-7)


@pytest.mark.parametrize('backend', ('numpy', 'jax', 'torch'))
@pytest.mark.parametrize('precision', (0, 1))
def test_complex_electric_and_magnetic_seams_keep_backend_and_precision(backend, precision):
    pytest.importorskip(backend)
    shapes = pair(wavebit.Material(MATERIALS[0], mu=.7+.8j),
                  left=wavebit.Material(4., mu=1.1+.1j))
    kw = dict(KW, shape_nvf_grid_size=64, truncation_order=(1, 1),
              structure=[dict(background=1., shapes=shapes)])
    want = _legacy_wavebit.rcwa(backend='numpy', **kw).solve()
    solver = _legacy_wavebit.rcwa(backend=backend, complex_dtype=precision, **kw)
    got = solver.solve()
    for a, b in ((got.R, want.R), (got.T, want.T)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0.,
                                   atol=2e-5 if precision else 2e-11)


@pytest.mark.parametrize('gap', (1., 1e-9))
def test_the_study_opens_a_positive_gap_without_resizing_shapes(gap):
    from studies.rcwa.nvf_seam_contrast.nvf_seam_contrast import K, opened
    a, b = opened(K, gap)
    actual = b['cx']-b['lx']/2 - (a['cx']+a['lx']/2)
    assert actual > 0.
    assert actual == pytest.approx(gap, rel=2e-5, abs=0.)
    assert a['cx'] < K[0]['cx'] and b['cx'] > K[1]['cx']
    for old, new in zip(K, (a, b)):
        assert {k: v for k, v in old.items() if k != 'cx'} == {
            k: v for k, v in new.items() if k != 'cx'}
