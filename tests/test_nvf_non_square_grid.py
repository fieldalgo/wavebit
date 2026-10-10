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

import wavebit
from wavebit._core.factorization import normal_vector_field

PX, PY, R = 200.0, 400.0, 70.0          # a 2:1 cell — square pixels need Ny == 2*Nx


def _disc(nx, ny, px=PX, py=PY, r=R):
    """A centred circular rod, sampled on an ``(ny, nx)`` raster of a ``px`` by ``py`` cell.

    **Relative permittivity**, which is what both consumers of this array take: the solver's
    ``structure`` keyword and ``normal_vector_field``'s ``eps_matrix``. The rod used to be
    written as the index ``3.0`` and squared at the two direct ``normal_vector_field`` calls,
    back when only one of the two consumers wanted a permittivity. Squaring it here instead
    keeps the scene identical -- ``eps = 9`` is the same rod ``n = 3`` was -- and, more to the
    point, leaves one array with one meaning: the alternative spelling after the flip is a
    file where the same variable is a permittivity of 9 three lines up and a permittivity of 3
    at every solver call, which is what it briefly was.

    The alignment figures quoted in the docstrings below were re-measured across the change and
    are bit-identical at all three ``n``, both lattices and both bands. That is not luck: the
    normal field is a geometry quantity, so it reads *where* the interface is and never the
    contrast across it.
    """
    x = (np.arange(nx) + 0.5) / nx * px
    y = (np.arange(ny) + 0.5) / ny * py
    X, Y = np.meshgrid(x, y)
    return X - px / 2, Y - py / 2, np.where((X - px / 2) ** 2 + (Y - py / 2) ** 2 <= r ** 2,
                                            9.0, 1.0)


def _alignment(nx, ny, lattice, band=None):
    """Mean ``|cos|`` between the computed normal and the circle's exact one.

    ``band`` restricts the average to pixels within that many x-pixels of the circle, and the
    restriction matters for a reason worth stating: **the radial-from-the-centre reference is
    only the right answer near the shape.** The distance field's bulk extension is periodic —
    far from the disc a pixel's nearest interface belongs to whichever *copy* of the disc is
    closest, which near the cell walls is not the one at the centre. Averaged over the whole
    cell, those wall pixels are counted as misalignment against an oracle that does not apply
    to them, and the score drifts with how much empty cell the case happens to have.

    Since the transform was made periodic (2026-08-17) that drift is visible: whole-cell
    alignment fell from 0.9903 to 0.9887 at ``n = 64`` while the *banded* figure did not move a
    single bit (0.938835 at +-2 px, 0.962047 at +-5 px, identical before and after at all three
    ``n``). Nothing changed at the interface; only the bulk did, and it changed to the
    periodically correct extension. The banded form is therefore the one that measures what
    this file is about.
    """
    dx, dy, structure = _disc(nx, ny)
    rho = np.hypot(dx, dy)
    rho[rho == 0] = 1.0
    n_u, n_y = normal_vector_field(structure, lattice=lattice)
    cos = np.abs(n_u * dx / rho + n_y * dy / rho)
    if band is None:
        return float(cos.mean())
    return float(cos[np.abs(rho - R) <= band * (PX / nx)].mean())


@pytest.mark.parametrize('n', [64, 128, 256])
def test_square_pixels_are_untouched_by_the_metric(n):
    """The one case that must not move: an isotropic step cancels in the normalisation.

    Bit-for-bit, not merely to round-off. The step is normalised to one x-pixel, so a square
    grid lands on ``sampling=(1.0, 1.0)`` — bit-identical to passing no sampling at all — and a
    division by ``1.0``, which is exact. Nothing a square-pixel user has ever run can move.
    """
    dx, dy, structure = _disc(n, 2 * n)
    legacy = normal_vector_field(structure, lattice=None)
    metric = normal_vector_field(structure, lattice=np.diag([PX, PY]))
    for a, b in zip(legacy, metric):
        assert np.array_equal(a, b)


@pytest.mark.parametrize('n', [64, 128, 256])
def test_the_metric_is_what_makes_a_non_square_grid_point_the_right_way(n):
    """Same raster, same shape: only the physical step of an index differs.

    Measured over the whole cell, which is the instrument that separates the two arms here: the
    aspect-ratio rotation is a *bulk* effect, so restricting to the interface band buries it
    under the staircase quantization both arms share (0.930 vs 0.939 at +-2 px, against 0.966
    vs 0.989 over the cell).

    The 0.99 floor this used to carry became 0.9887 at ``n = 64`` when the distance transform
    was made periodic on 2026-08-17, and that is a change in the *oracle's* domain of validity
    rather than in the field's accuracy — radial-from-the-centre stops describing the bulk once
    pixels near the wall point at the nearest periodic copy (see :func:`_alignment`). The
    interface band is bit-identical across that change, which is the evidence that nothing got
    worse; the banded assertion below is what keeps that claim under test rather than in a
    comment.
    """
    legacy = _alignment(n, n, lattice=None)
    metric = _alignment(n, n, lattice=np.diag([PX, PY]))
    assert legacy < 0.975, f'the uncorrected field is not wrong here; test is vacuous ({legacy})'
    assert metric > legacy + 0.02
    assert metric > 0.985

    # The correction reaches the interface too, where the oracle is unambiguous.
    band_legacy = _alignment(n, n, lattice=None, band=5)
    band_metric = _alignment(n, n, lattice=np.diag([PX, PY]), band=5)
    assert band_metric > band_legacy + 0.01, (
        f'the metric buys nothing at the interface: {band_legacy:.6f} -> {band_metric:.6f}')


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
    _, _, structure = _disc(96, 96)
    cfg = dict(backend='numpy', period=[PX, PY], truncation_order=[4, 4], wavelength=300.0, thickness=[220.0],
               top=1.0, bot=1.0, theta=0.2, phi=0.3, psi=0.4, factorization='nvf',
               structure=structure[None, :, :])
    res = wavebit.rcwa(**cfg).solve()
    total = float(np.sum(np.asarray(res.R)) + np.sum(np.asarray(res.T)))
    assert total == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize('coefficients', ['sampled', 'integrated'])
def test_a_frozen_normal_field_replays_onto_a_non_square_grid(coefficients):
    """Both NVF builders must be handed the same basis the freeze recorded.

    The basis is an input to the normal field, so it is part of the record's staleness key.
    Freezing with it and rebuilding without — which is what an unplumbed builder does — reads
    as a cell change and refuses the replay outright.
    """
    _, _, structure = _disc(64, 64)
    cfg = dict(backend='numpy', period=[PX, PY], truncation_order=[3, 3], wavelength=300.0, thickness=[220.0],
               top=1.0, bot=2.25, theta=0.2, factorization='nvf',
               fourier_coefficients=coefficients, structure=structure[None, :, :])
    solver = wavebit.rcwa(**cfg)
    replayed = solver.solve(geometry=solver.freeze_geometry()).T
    eager = wavebit.rcwa(**cfg).solve().T
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
    import wavebit._core.factorization as fac

    _, _, structure = _disc(64, 64)
    layers = structure[None, :, :]
    cfg = dict(backend='numpy', period=[PX, PY], truncation_order=[3, 3], wavelength=300.0, thickness=[220.0],
               top=1.0, bot=2.25, theta=0.2, factorization='nvf',
               fourier_coefficients=coefficients, structure=layers)
    build = (fac.to_conv_mat_raster_nvf if coefficients == 'sampled'
             else fac.to_conv_mat_raster_integrated_nvf)

    on_pixel_shape = build(layers, 3, 3, lattice=np.diag([PX, PY]))[0]
    on_index_grid = build(layers, 3, 3, lattice=None)[0]
    assert np.abs(np.asarray(on_pixel_shape) - np.asarray(on_index_grid)).max() > 1e-6, \
        'the two builds coincide here, so this test could not tell them apart'

    epx = np.asarray(wavebit.rcwa(**cfg)._eps_conv_matrices()[0])
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
    :func:`~wavebit._core.factorization._lattice_key`).
    """
    _, _, structure = _disc(64, 64)
    solver = wavebit.rcwa(backend='numpy', period=[PX, PY], truncation_order=[3, 3], wavelength=300.0,
                            thickness=[220.0], top=1.0, bot=2.25, theta=0.2,
                            factorization='nvf', structure=structure[None, :, :])
    recorded = solver.freeze_geometry().normal_field_source[0][2]
    assert recorded == pytest.approx((1.0, 0.0, 0.0, PY / PX))   # (400/64) / (200/64) == 2

    # A cell twice the size with the same shape is the same normal field, so the same key.
    doubled = wavebit.rcwa(backend='numpy', period=[2 * PX, 2 * PY], truncation_order=[3, 3], wavelength=300.0,
                             thickness=[220.0], top=1.0, bot=2.25, theta=0.2,
                             factorization='nvf', structure=structure[None, :, :])
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

    _, _, structure = _disc(48, 48)
    base = dict(backend='jax', truncation_order=[3, 3], wavelength=300.0, thickness=[220.0], top=1.0,
                bot=2.25, theta=0.2, factorization='nvf', structure=jnp.asarray(structure[None, :, :]))
    frozen = wavebit.rcwa(period=[PX, PY], **base).freeze_geometry()

    def objective(py):
        return wavebit.rcwa(period=[PX, py], **base).solve(geometry=frozen).T.sum()

    assert np.isfinite(float(jax.grad(objective)(jnp.float64(PY))))
    assert np.isfinite(float(jax.jit(objective)(jnp.float64(PY))))
