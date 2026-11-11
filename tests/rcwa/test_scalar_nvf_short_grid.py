"""The scalar shape NVF projector reads its normal field through the trigonometric interpolant.

Until 2026-11-11 ``parametric._sampled_coeffs`` took the projector's ``|k| <= 2M`` coefficients
as wrapped DFT bins (``k % n``), and both shape paths refused a grid below ``4M+1`` per axis.
On ``n <= 2M`` points that fold is wrong in kind -- a constant field picks up off-diagonals of
1.0 -- and between ``2M+1`` and ``4M`` it puts the orders past Nyquist on the wrong bin. The
coefficients now come from the interpolant (weight 1 below Nyquist, 1/2 at it, 0 past it), any
positive grid runs, and at or above ``4M+1``, where every weight is 1, the multiply is skipped
so the transform is the old one to the bit.
"""
import numpy as np
import pytest

from wavebit import Lattice, Material, rcwa
from wavebit.rcwa._engine.parametric import _device, _sampled_coeffs, conv_matrix_2d

BACKENDS = ['numpy', 'jax', 'torch']


def _xp(backend):
    """The array namespace for a backend, as the engine uses it, and a constructor."""
    if backend == 'numpy':
        import array_api_compat.numpy as xp
        return xp, np.asarray
    if backend == 'jax':
        pytest.importorskip('jax')
        import jax.numpy as xp
        return xp, xp.asarray
    pytest.importorskip('torch')
    import torch
    import array_api_compat.torch as xp
    return xp, torch.asarray


def _host(value):
    return np.asarray(value.detach().cpu() if hasattr(value, 'detach') else value)


@pytest.fixture
def x64(request):
    """JAX in double precision for the one test; a no-op otherwise."""
    if request.node.callspec.params.get('backend') != 'jax':
        yield
        return
    import jax
    with jax.enable_x64(True):
        yield


# --------------------------------------------------------------------------- the transform
@pytest.mark.parametrize('grid', [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (6, 2), (1, 5)])
@pytest.mark.parametrize('backend', BACKENDS)
def test_a_constant_field_transforms_to_the_identity_on_a_short_grid(backend, grid, x64):
    """``n <= 2M``: the wrapped bins folded orders onto ``k = 0`` and gave off-diagonals of 1.0.

    For NVF to collapse onto Li's rules where the normal field is constant, its projector must
    transform to the identity (times the constant). The interpolant's coefficients past Nyquist
    are zero, so that holds on any grid.
    """
    xp, asarray = _xp(backend)
    M = 3
    value = 0.37 - 0.11j
    field = asarray(np.full(grid, value, dtype=complex))
    matrix = _host(conv_matrix_2d(xp, _sampled_coeffs(xp, field, M, M), M, M))
    gap = np.abs(matrix - value * np.eye((2*M + 1)**2)).max()
    assert gap < 1e-15, f'a constant field on a {grid} grid is {gap:.1e} from the identity'


def _wrapped_bins(xp, field, half_width_u, half_width_v):
    """``_sampled_coeffs`` as it was before 2026-11-11, transcribed op for op (the reference)."""
    Ny, Nx = field.shape
    dev = _device(field)
    F = xp.fft.fft2(field) / (Ny * Nx)
    iy = xp.asarray([k % Ny for k in range(-2 * half_width_v, 2 * half_width_v + 1)], device=dev)
    ix = xp.asarray([k % Nx for k in range(-2 * half_width_u, 2 * half_width_u + 1)], device=dev)
    coeff = F[iy[:, None], ix[None, :]]
    ky = xp.asarray([float(k) for k in range(-2 * half_width_v, 2 * half_width_v + 1)], device=dev)
    kx = xp.asarray([float(k) for k in range(-2 * half_width_u, 2 * half_width_u + 1)], device=dev)
    pi = 3.141592653589793
    phase = (xp.exp(-1j * pi * xp.astype(ky, coeff.dtype) / Ny)[:, None]
             * xp.exp(-1j * pi * xp.astype(kx, coeff.dtype) / Nx)[None, :])
    return coeff * phase


AT_OR_ABOVE = [((13, 13), (3, 3)), ((14, 14), (3, 3)), ((13, 9), (2, 3)), ((14, 10), (2, 3)),
               ((17, 9), (2, 4)), ((1, 21), (5, 0)), ((64, 64), (6, 6)), ((25, 26), (6, 6))]


@pytest.mark.parametrize('grid,orders', AT_OR_ABOVE)
@pytest.mark.parametrize('backend', BACKENDS)
def test_at_or_above_four_m_plus_one_the_transform_is_the_wrapped_bins_to_the_bit(backend, grid, orders, x64):
    """Every weight is 1 there and the multiply is skipped: the old transform, byte for byte."""
    xp, asarray = _xp(backend)
    hu, hv = orders
    rng = np.random.default_rng(7)
    field = rng.standard_normal(grid) + 1j * rng.standard_normal(grid)
    got = _host(_sampled_coeffs(xp, asarray(field), hu, hv))
    want = _host(_wrapped_bins(xp, asarray(field), hu, hv))
    assert got.dtype == want.dtype and got.shape == want.shape
    assert got.tobytes() == want.tobytes()


@pytest.mark.parametrize('grid,orders', AT_OR_ABOVE + [((12, 12), (3, 3)), ((13, 8), (2, 3))])
def test_the_weights_are_applied_only_below_four_m_plus_one(grid, orders):
    """The skip is structural: the traced program is the old one at or above ``4M+1``.

    Equal values on a few fields cannot show that the multiply was skipped -- a complex-by-real
    multiply by exactly 1 changes a bit only on rare patterns (the sign of an exact zero), which
    an FFT of ordinary data does not produce. The program JAX traces can: one more multiply is
    one more equation, so at or above ``4M+1`` it must match the old transform's, and below it
    must not.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    hu, hv = orders
    with jax.enable_x64(True):
        field = jnp.zeros(grid, dtype=jnp.complex128)
        new = str(jax.make_jaxpr(lambda f: _sampled_coeffs(jnp, f, hu, hv))(field))
        old = str(jax.make_jaxpr(lambda f: _wrapped_bins(jnp, f, hu, hv))(field))
    short = min(grid[0] - (4*hv + 1), grid[1] - (4*hu + 1)) < 0
    assert (new != old) == short


@pytest.mark.parametrize('n', [1, 2, 3, 6, 7, 11, 12])
def test_below_four_m_plus_one_the_orders_past_nyquist_read_zero(n):
    """The interpolant's coefficients: the bin below Nyquist, half of it at Nyquist, zero past it."""
    import array_api_compat.numpy as xp
    M = 3
    rng = np.random.default_rng(n)
    field = rng.standard_normal((n, 1)) + 1j * rng.standard_normal((n, 1))
    got = np.asarray(_sampled_coeffs(xp, field, 0, M))[:, 0]
    want = np.asarray(_wrapped_bins(xp, field, 0, M))[:, 0]
    for k, (a, b) in zip(range(-2*M, 2*M + 1), zip(got, want)):
        weight = 1.0 if 2*abs(k) < n else (0.5 if 2*abs(k) == n else 0.0)
        assert a == weight * b


# --------------------------------------------------------------------------- solves
def _region(geometry, eps=4.):
    return rcwa.Region(geometry=geometry, material=Material(eps))


SHORT_GRID_LAYERS = {
    'rectangle': (dict(period=(1., 1.)), (_region(rcwa.Rectangle(size=(.3, .4), angle=.35)),)),
    'circle': (dict(period=(1., 1.)), (_region(rcwa.Circle(radius=.3), 12.25),)),
    'ellipse': (dict(period=(1., 1.)), (_region(rcwa.Ellipse(radii=(.35, .2), angle=.4), 6.),)),
    # Two overlapping circles take the painted (boundary) route, which shares the transform.
    'overlap': (dict(period=(1., 1.)), (_region(rcwa.Circle(radius=.25, center=(-.12, 0.))),
                                        _region(rcwa.Circle(radius=.22, center=(.15, .05)), 9.))),
    'hexagonal': (dict(lattice=Lattice.hexagonal(1.)), (_region(rcwa.Circle(radius=.28)),)),
}


@pytest.mark.parametrize('polarization', ['s', 'p'])
@pytest.mark.parametrize('grid', [1, 2, 3, 5, 8, 12])
@pytest.mark.parametrize('case', sorted(SHORT_GRID_LAYERS))
def test_a_lossless_scalar_shape_layer_conserves_energy_below_four_m_plus_one(case, grid, polarization):
    """Below ``4M+1`` the answer is a coarser one, not a broken one: ``R + T = 1`` still.

    The projector stays Hermitian (the interpolant of a real field has conjugate-symmetric
    coefficients), so a lossless layer conserves energy at every grid, as at the default one.
    """
    import warnings
    cell, regions = SHORT_GRID_LAYERS[case]
    problem = rcwa.Problem(
        stack=rcwa.Stack(layers=(rcwa.Layer(thickness=.25, background=Material(1.5), regions=regions),),
                         bottom=Material(2.25), **cell),
        excitation=rcwa.Excitation(wavelength=1.3, sources=(
            rcwa.PlaneWave(theta=.2, phi=.3, polarization=polarization),)))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', rcwa.diagnostics.ShapeGridWarning)
        result = rcwa.run(problem, numerics=rcwa.Numerics(orders=(3, 3), shape_nvf_grid_size=grid))
    R, T = np.asarray(result.R), np.asarray(result.T)
    assert np.isfinite(R).all() and np.isfinite(T).all()
    balance = abs(float(R.sum() + T.sum()) - 1.)
    assert balance < 1e-12, f'{case} at grid {grid}: |1 - R - T| = {balance:.1e}'
