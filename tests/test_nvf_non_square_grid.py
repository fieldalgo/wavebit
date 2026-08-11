"""The NVF normal field on a raster whose pixels are not square in real space.

``normal_vector_field`` builds its normal from a ``distance_transform_edt`` and an
``np.gradient``, both of which count *index steps*. That is a Cartesian normal only when one
index step is the same physical length on both axes — ``period[0]/Nx == period[1]/Ny``. When it
is not, every normal is rotated toward the finely-sampled axis by a fixed angle, and the whole
factorization is built on it.

Nothing detected that. ``_nvf_assemble`` stays Hermitian whatever direction the normal points,
so a lossless stack still conserves energy to 1e-15 (``test_energy_is_conserved_anyway`` below
pins that the tell really is absent, which is why the fix needed a test written *for* it). The
stored goldens could not catch it either: every RETICOLO case is a 1D grating, where a one-cell
axis short-circuits the construction before the metric is read, and the one NVF entry in
``golden_legacy.npz`` is a 5x5 raster on a square cell, where the correction is a no-op.

So the arbiter here is the geometry itself. A circle's normal is ``(r - r0)/|r - r0|`` in closed
form, everywhere in the cell, which is exactly what the distance transform is trying to extend
into the bulk — no solver, no truncation, no staircase-versus-continuum argument.

The decisive part is not the size of the error but its behaviour under refinement:
``test_the_error_is_a_bias_refinement_cannot_remove`` shows the uncorrected field converging to
the *wrong* direction, which is what separates a bug from a discretization residual.
"""
import numpy as np
import pytest

import fieldwave
from fieldwave._core.factorization import normal_vector_field

PX, PY, R = 200.0, 400.0, 70.0          # a 2:1 cell — square pixels need Ny == 2*Nx


def _disc(nx, ny, px=PX, py=PY, r=R):
    """A centred circular rod, sampled on an ``(ny, nx)`` raster of a ``px`` by ``py`` cell."""
    x = (np.arange(nx) + 0.5) / nx * px
    y = (np.arange(ny) + 0.5) / ny * py
    X, Y = np.meshgrid(x, y)
    return X - px / 2, Y - py / 2, np.where((X - px / 2) ** 2 + (Y - py / 2) ** 2 <= r ** 2,
                                            3.0, 1.0)


def _alignment(nx, ny, lattice):
    """Mean ``|cos|`` between the computed normal and the circle's exact one, over the cell."""
    dx, dy, ucell = _disc(nx, ny)
    rho = np.hypot(dx, dy)
    rho[rho == 0] = 1.0
    n_x, n_y = normal_vector_field(ucell ** 2, lattice=lattice)
    return float(np.abs(n_x * dx / rho + n_y * dy / rho).mean())


@pytest.mark.parametrize('n', [64, 128, 256])
def test_square_pixels_are_untouched_by_the_metric(n):
    """The one case that must not move: an isotropic step cancels in the normalisation.

    Bit-for-bit, not merely to round-off. The step is normalised to one x-pixel, so a square
    grid lands on ``sampling=(1.0, 1.0)`` — bit-identical to passing no sampling at all — and a
    division by ``1.0``, which is exact. Nothing a square-pixel user has ever run can move.
    """
    dx, dy, ucell = _disc(n, 2 * n)
    legacy = normal_vector_field(ucell ** 2, lattice=None)
    metric = normal_vector_field(ucell ** 2, lattice=np.diag([PX, PY]))
    for a, b in zip(legacy, metric):
        assert np.array_equal(a, b)


@pytest.mark.parametrize('n', [64, 128, 256])
def test_the_metric_is_what_makes_a_non_square_grid_point_the_right_way(n):
    """Same raster, same shape: only the physical step of an index differs."""
    legacy = _alignment(n, n, lattice=None)
    metric = _alignment(n, n, lattice=np.diag([PX, PY]))
    assert legacy < 0.975, f'the uncorrected field is not wrong here; test is vacuous ({legacy})'
    assert metric > legacy + 0.02
    assert metric > 0.99


def test_the_error_is_a_bias_refinement_cannot_remove():
    """The point of the whole change: the uncorrected field converges to the *wrong* normal.

    A discretization residual shrinks as the raster refines. This does not — the angular error
    is set by the pixel aspect ratio, which refinement holds fixed. Meanwhile the corrected
    field converges to the same value the square-pixel grid of matching x-resolution reaches,
    which is the independent statement that it converges to the *right* one.
    """
    grids = (64, 128, 256)
    legacy = [_alignment(n, n, lattice=None) for n in grids]
    metric = [_alignment(n, n, lattice=np.diag([PX, PY])) for n in grids]

    assert max(legacy) - min(legacy) < 0.01, f'legacy error is not a fixed bias: {legacy}'
    assert max(legacy) < 0.98
    assert metric[0] < metric[1] < metric[2], f'the correction should converge: {metric}'

    # ... and it converges to where square pixels already sit at the same x-resolution.
    square = _alignment(grids[-1], 2 * grids[-1], lattice=np.diag([PX, PY]))
    assert metric[-1] == pytest.approx(square, abs=2e-3)


def test_energy_is_conserved_anyway():
    """Why nothing caught this. The wrong normal costs nothing the usual tell can see."""
    _, _, ucell = _disc(96, 96)
    cfg = dict(backend=0, period=[PX, PY], fto=[4, 4], wavelength=300.0, thickness=[220.0],
               n_top=1.0, n_bot=1.0, theta=0.2, phi=0.3, psi=0.4, factorization='nvf',
               ucell=ucell[None, :, :])
    res = fieldwave.rcwa(**cfg).conv_solve()
    total = float(np.sum(np.asarray(res.R)) + np.sum(np.asarray(res.T)))
    assert total == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize('coefficients', ['sampled', 'integrated'])
def test_a_frozen_normal_field_replays_onto_a_non_square_grid(coefficients):
    """Both NVF builders must be handed the same basis the freeze recorded.

    The basis is an input to the normal field, so it is part of the record's staleness key.
    Freezing with it and rebuilding without — which is what an unplumbed builder does — reads
    as a cell change and refuses the replay outright.
    """
    _, _, ucell = _disc(64, 64)
    cfg = dict(backend=0, period=[PX, PY], fto=[3, 3], wavelength=300.0, thickness=[220.0],
               n_top=1.0, n_bot=1.5, theta=0.2, factorization='nvf',
               fourier_coefficients=coefficients, ucell=ucell[None, :, :])
    solver = fieldwave.rcwa(**cfg)
    replayed = solver.conv_solve(geometry=solver.freeze_geometry()).T
    eager = fieldwave.rcwa(**cfg).conv_solve().T
    # Not bit-identity: these are two solves, and `eig` is not bit-reproducible run to run
    # under multithreaded LAPACK. The failure this guards against is not subtle — a basis the
    # replay cannot see either raises StaleGeometryError or moves the answer by ~8e-3.
    assert np.abs(np.asarray(replayed) - np.asarray(eager)).max() < 1e-12


@pytest.mark.parametrize('coefficients', ['sampled', 'integrated'])
def test_the_solver_hands_the_cell_basis_to_the_factorization(coefficients):
    """The wiring, pinned against the two builds it has to agree with.

    Everything above tests ``normal_vector_field`` in isolation, which was already correct once
    handed a basis — the defect was that the solver never handed it one. So this asserts the
    engine's convolution matrices *are* the ones built on ``diag(period)``, and are not the ones
    built on the index grid. Without the second half the test passes either way.
    """
    import fieldwave._core.factorization as fac

    _, _, ucell = _disc(64, 64)
    layers = ucell[None, :, :]
    cfg = dict(backend=0, period=[PX, PY], fto=[3, 3], wavelength=300.0, thickness=[220.0],
               n_top=1.0, n_bot=1.5, theta=0.2, factorization='nvf',
               fourier_coefficients=coefficients, ucell=layers)
    build = (fac.to_conv_mat_raster_nvf if coefficients == 'sampled'
             else fac.to_conv_mat_raster_continuous_nvf)

    on_pixel_shape = build(layers, 3, 3, lattice=np.diag([PX, PY]))[0]
    on_index_grid = build(layers, 3, 3, lattice=None)[0]
    assert np.abs(np.asarray(on_pixel_shape) - np.asarray(on_index_grid)).max() > 1e-6, \
        'the two builds coincide here, so this test could not tell them apart'

    epx = np.asarray(fieldwave.rcwa(**cfg)._conv_matrices()[0])
    assert np.abs(epx - np.asarray(on_pixel_shape)).max() == 0.0
    assert np.abs(epx - np.asarray(on_index_grid)).max() > 1e-6


def test_the_frozen_record_carries_the_pixel_shape():
    """The record stores the cell's shape, and it is not ``None``.

    ``None`` used to be recorded for every structure, which is the same thing as asserting the
    raster's pixels were square. On a 200x400 cell at 64x64 they are 1:2.

    Held as the basis scaled to ``|a1| == 1`` rather than as a lone aspect ratio, because the
    same slot has to describe a skewed cell too — where two numbers cannot. For an axis-aligned
    cell it carries exactly the aspect and nothing more, which is what keeps a uniform rescale
    of the cell from reading as a change (see
    :func:`~fieldwave._core.factorization._lattice_key`).
    """
    _, _, ucell = _disc(64, 64)
    solver = fieldwave.rcwa(backend=0, period=[PX, PY], fto=[3, 3], wavelength=300.0,
                            thickness=[220.0], n_top=1.0, n_bot=1.5, theta=0.2,
                            factorization='nvf', ucell=ucell[None, :, :])
    recorded = solver.freeze_geometry().normal_field_source[0][2]
    assert recorded == pytest.approx((1.0, 0.0, 0.0, PY / PX))   # (400/64) / (200/64) == 2

    # A cell twice the size with the same shape is the same normal field, so the same key.
    doubled = fieldwave.rcwa(backend=0, period=[2 * PX, 2 * PY], fto=[3, 3], wavelength=300.0,
                             thickness=[220.0], n_top=1.0, n_bot=1.5, theta=0.2,
                             factorization='nvf', ucell=ucell[None, :, :])
    assert doubled.freeze_geometry().normal_field_source[0][2] == pytest.approx(recorded)


def test_differentiating_period_does_not_read_as_a_cell_change():
    """``period`` is a design variable, and a grad tracer will not narrow to a float.

    So the live basis is unreadable exactly when ``period`` is under differentiation. The basis
    the record was frozen on is the right answer there — the field being replayed is the one
    built on it — and the alternative is that freezing a structure and then differentiating it
    refuses its own record.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp

    _, _, ucell = _disc(48, 48)
    base = dict(backend=1, fto=[3, 3], wavelength=300.0, thickness=[220.0], n_top=1.0,
                n_bot=1.5, theta=0.2, factorization='nvf', ucell=jnp.asarray(ucell[None, :, :]))
    frozen = fieldwave.rcwa(period=[PX, PY], **base).freeze_geometry()

    def objective(py):
        return fieldwave.rcwa(period=[PX, py], **base).conv_solve(geometry=frozen).T.sum()

    assert np.isfinite(float(jax.grad(objective)(jnp.float64(PY))))
    assert np.isfinite(float(jax.jit(objective)(jnp.float64(PY))))
