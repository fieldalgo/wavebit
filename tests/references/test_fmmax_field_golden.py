"""The in-cell field against fmmax — including on a **patterned** layer, which S4 cannot reach.

``test_s4_field_golden.py`` is exact but stops at uniform stacks: on a grating S4 and this
project disagree at 1e-3, and up to 7.6e-1 on ``E_z``, because their truncation and
factorization cannot be matched through S4's interface. Both knobs are reachable in fmmax, so
this file is where patterned field reconstruction finally gets an outside opinion.

**What each geometry reaches.** Two of them, and the distinction is the point:

* ``grating`` varies along **x only**. It pins the transverse assignment, but its reconstructed
  field is y-constant to machine precision (2.5e-15 of ``max|F|``), so no ``ky != 0`` order
  carries amplitude and the ``Ky`` terms of ``Ez`` / ``Hz`` multiply nothing.
* ``patch`` is an x/y-asymmetric crossed patch whose field varies in both axes (y-variation
  3.6e-1), compared on a full ``(z, y, x)`` grid. It reaches both halves, the structure's own
  ``ky`` orders, and which end of the raster is ``-y``.

Three tiers, and the file keeps them apart on purpose:

* **uniform**, **uniform2d** — 1e-13. There is no truncation to disagree about, so this is
  exactness, and it is what proves the harness's conventions (axis order, cell-centred x and y,
  the raster transpose, the corner-to-centre shift) rather than the physics.
* **grating** — 1e-4. Converged agreement, not machine precision: the two codes' vector
  factorizations still differ in detail once a discontinuity exists. That floor is measured
  (1.9e-5 for s, 7.2e-5 for p) and stated, not tuned until green.
* **patch** — 5e-2, a much weaker oracle, and the reason is stated rather than hidden: the
  residual does **not** shrink from 49 to 361 orders, so it is the two vector factorizations
  disagreeing near the rectangle's corners. Both truncations are frozen so that claim stays
  checkable. A defect in the field assembly moves this cell by order one, far above the floor.

**What is deliberately not here.** No internal curl-``z`` residual: ``Hz`` is *defined* from
``Ex`` and ``Ey`` here, so a curl-z check is satisfied by construction, and a solver that
paired the wrong ``K`` with each component would be self-consistent in the same way.

A 1e-4 oracle is worth having precisely because of what this repository has actually shipped
wrong. The sign-convention defect fixed in ``9824f0d`` moved fields by 2e0; a transposed raster
reverses the s/p split outright; a swapped component is order-1. Every one of those is four
orders above this floor. What 1e-4 cannot see is slow numerical drift — that is
``tests/fixtures/legacy_rcwa/golden_field.npz``'s job, and it is a different claim.

**The amplitude convention is fitted, and then checked against a closed form.** fmmax's "unit
mode amplitude" is not this project's "unit incident flux". Each case fits one complex scale by
least squares — and the fit is then required to equal ``-cos(theta)`` for s and ``-1`` for p,
which is what that ratio turned out to be, times the stated phase between the two codes' unit
incidence points (fmmax's corner, this library's centre; :func:`expected_scale`). Stating the
*dependence* rather than tabulating
three constants is deliberate: a table is satisfied by anything that reproduces the numbers,
whereas a case added at a new angle is genuinely checked by the formula.
"""

from wavebit.compat import legacy as _legacy_wavebit

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))
from tools.repository_paths import repository_path
import functools
import sys

import numpy as np
import pytest


HERE = repository_path('tests/references')
sys.path.insert(0, str(HERE))
from tests.references._fmmax_field_cases import (CASES, CASES_2D, MAX_ORDER, PERIOD, PERIOD_2D,
                                                 RES_Z, RES_Z_2D, THETA_2D, THICK, THICK_2D, WL,
                                                 WL_2D, fine_res_x, raster, raster_2d, x_samples,
                                                 x_samples_2d, y_samples_2d, z_samples_2d)

GOLDEN = np.load(HERE / 'golden_fmmax_field.npz')

TOL = {'uniform': 1e-13, 'grating': 1e-4, 'uniform2d': 1e-13, 'patch': 5e-2}
NAMES = ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')

def expected_scale(theta, pol, wavelength=WL, period=PERIOD):
    """What the fitted amplitude ratio must be — a closed form, not a table of constants.

    fmmax's "unit mode amplitude" is not this project's "unit incident flux", and the ratio
    turned out to be exactly ``-cos(theta)`` for s and ``-1`` for p. Measured across 0, 24 and
    41 degrees: -1.0000, -0.9135, -0.7547 against cos of 1.0000, 0.9135, 0.7547.

    Writing it as a formula rather than three frozen numbers is the point. A table would be
    satisfied by any solver that happened to hit those values; this says the *dependence* is
    right, so a case added at a new angle is checked rather than merely accommodated.

    Times one phase since 2026-09-27, where each code's incident wave is unit. fmmax's is unit
    at its ``x = 0``, the cell corner -- which is why the ratio came out real while this
    library's was too -- and this library's at the cell centre since then: the same wave unit
    half a period further along is ``exp(-1j * k0 * sin(theta) * PERIOD/2)`` times fmmax's (lit
    from air at ``phi = 0``). Also a closed form, and also checked at every angle.
    """
    ratio = -np.cos(np.deg2rad(theta)) if pol == 's' else -1.0
    return ratio * np.exp(-1j * 2 * np.pi / wavelength * np.sin(np.deg2rad(theta)) * period / 2)


def _reconstruct(name, theta, pol):
    """This project's field at fmmax's own sample points — no interpolation.

    ``fine_res_x()`` is ``2*RES_X + 1`` so that index ``2j+1`` of an inclusive ``linspace``
    lands exactly on fmmax's ``j``-th cell-centred sample.

    **One layer, no spacers.** An earlier version padded the stack with 1e-9-thick vacuum to
    mirror fmmax's zero-thickness ambient layers; that put a ~1e-11 phase into the fitted
    amplitude — ``k0 * 1e-9`` exactly — which is small but is an artefact of the harness rather
    than of either solver. ``top = bot = 1`` describes the same scene with nothing to
    propagate through — vacuum, where the permittivity and the index are the same 1.0 and so
    the flip is invisible; the layer is where it is not.
    """
    structure = raster(name)[None, :, :]
    solver = _legacy_wavebit.rcwa(backend='numpy', top=1.0, bot=1.0, period=(PERIOD, PERIOD),
                          truncation_order=(MAX_ORDER, MAX_ORDER), wavelength=WL, thickness=[THICK],
                          structure=structure, theta=np.deg2rad(theta), phi=0.0,
                          psi=np.pi / 2 if pol == 's' else 0.0, answers='field')
    solver.solve()
    cell = np.asarray(solver.reconstruct_field(sample_x=fine_res_x(), sample_y=1, sample_z=RES_Z))
    if cell.ndim == 5:
        cell = cell[0]
    return cell[::-1, 0, 1::2, :] * np.array([1, 1, -1, -1, -1, 1])                      # the only layer, at fmmax's x points


def _fit(ref, got):
    """The single complex scale that best takes ``ref`` onto ``got``, and what is left."""
    scale = np.vdot(ref.ravel(), got.ravel()) / np.vdot(ref.ravel(), ref.ravel())
    return scale, np.max(np.abs(got - scale * ref)) / np.max(np.abs(ref))


@pytest.mark.parametrize('name,theta,pol', CASES,
                         ids=[f'{n}-{t:g}-{p}' for n, t, p in CASES])
def test_the_field_matches_fmmax(name, theta, pol):
    ref = GOLDEN[f'{name}__{theta:g}__{pol}']
    got = _reconstruct(name, theta, pol)
    assert got.shape == ref.shape, (got.shape, ref.shape)

    scale, resid = _fit(ref, got)
    assert resid < TOL[name], f'{name} {theta:g} {pol}: residual {resid:.2e}'
    want = expected_scale(theta, pol)
    # The fitted scale inherits the same floor as the residual: exact where nothing truncates,
    # and carrying the factorization difference where something does.
    assert abs(scale - want) < TOL[name], (
        f'{name} {theta:g} {pol}: amplitude scale {scale:.7f}, expected {want:.7f}')


@pytest.mark.parametrize('name,theta,pol', [c for c in CASES if c[0] == 'uniform'],
                         ids=[f'{n}-{t:g}-{p}' for n, t, p in CASES if n == 'uniform'])
def test_every_component_agrees_on_the_exact_cases(name, theta, pol):
    """On the uniform cases the agreement is per component, not just in aggregate.

    A least-squares fit over the whole cell is dominated by the largest component; ``E_z`` runs
    well below ``H_y`` at these incidences, so it could be wholly wrong and still leave a small
    overall residual. That is exactly the failure mode the 2026-08-14 sign defect had.
    """
    ref = GOLDEN[f'{name}__{theta:g}__{pol}']
    got = _reconstruct(name, theta, pol)
    scale, _ = _fit(ref, got)
    cell_scale = np.max(np.abs(ref))
    for i, comp in enumerate(NAMES):
        if np.max(np.abs(ref[..., i])) < 1e-9 * cell_scale:
            continue
        gap = np.max(np.abs(got[..., i] - scale * ref[..., i])) / np.max(np.abs(ref[..., i]))
        assert gap < 1e-12, f'{comp}: {gap:.2e}'


def test_the_patterned_cases_are_actually_patterned():
    """Guard the premise: the grating rows must differ from the uniform ones.

    If ``raster`` ever returned a flat cell for both, every assertion above would still pass
    and the file would silently stop testing the thing it was built for.
    """
    assert not np.allclose(raster('grating'), raster('uniform'))
    a = GOLDEN['grating__24__s']
    b = GOLDEN['uniform__24__s']
    assert np.max(np.abs(a - b)) / np.max(np.abs(b)) > 1e-2


def test_the_sample_grids_are_the_ones_the_cases_describe():
    """The comparison rests on two grids coinciding exactly; state it rather than trust it."""
    assert np.allclose(GOLDEN['x'], x_samples(), atol=1e-9)
    fine = np.linspace(0, PERIOD, fine_res_x())
    assert np.allclose(fine[1::2], GOLDEN['x'], atol=1e-9)
    assert np.allclose(GOLDEN['z'], np.linspace(0, THICK, RES_Z), atol=1e-9)


# ---------------------------------------------------------------------------------------
# The 2D crossed patch. Everything above is y-invariant; this is what reaches the `Ky` half.

FORMULATIONS = ('second_order', 'first_order')


@functools.lru_cache(maxsize=None)
def _reconstruct_2d(name, order, pol, reverse_rows=False, formulation='second_order'):
    """This project's field at fmmax's own (x, y, z) points, in fmmax's frame.

    x and y are handed over as coordinates: fmmax's corner-origin grid, shifted by ``-P/2`` into
    this library's centre frame. z is a count, which samples the one layer at both faces from
    the inside, as fmmax does; an explicit height on a face would take the layer above it.
    Reversed and given the mirror parity, as in :func:`_reconstruct`. Cached: the tests below
    read each case more than once, and 361 orders is the slow part of this file.

    ``reverse_rows`` solves the raster upside down, which is how the row-order test below asks
    what a reversed cell would have produced. ``formulation`` picks the field assembly: the
    first-order path builds ``Ez`` / ``Hz`` with its own copy of the ``Ky`` terms.
    """
    structure = raster_2d(name)[::-1] if reverse_rows else raster_2d(name)
    solver = _legacy_wavebit.rcwa(backend='numpy', top=1.0, bot=1.0,
                                  period=(PERIOD_2D, PERIOD_2D), truncation_order=(order, order),
                                  wavelength=WL_2D, thickness=[THICK_2D],
                                  structure=structure[None, :, :], theta=np.deg2rad(THETA_2D),
                                  phi=0.0, psi=np.pi / 2 if pol == 's' else 0.0, answers='field')
    solver.formulation = formulation
    solver.solve()
    assert solver.formulation_assigned == formulation
    cell = np.asarray(solver.reconstruct_field(sample_x=x_samples_2d() - PERIOD_2D / 2,
                                               sample_y=y_samples_2d() - PERIOD_2D / 2,
                                               sample_z=RES_Z_2D))
    if cell.ndim == 5:
        cell = cell[0]
    cell = cell[::-1] * np.array([1, 1, -1, -1, -1, 1])
    cell.flags.writeable = False
    return cell


def _scale_2d(pol):
    return expected_scale(THETA_2D, pol, wavelength=WL_2D, period=PERIOD_2D)


@pytest.mark.parametrize('formulation', FORMULATIONS)
@pytest.mark.parametrize('name,order,pol', CASES_2D,
                         ids=[f'{n}-{o}-{p}' for n, o, p in CASES_2D])
def test_the_2d_field_matches_fmmax(name, order, pol, formulation):
    """A genuinely 2D patterned field against an outside code.

    The uniform control shares this harness -- the ``(z, y, x)`` axis order, the cell-centred y
    grid, the raster transpose, the corner-to-centre shift -- and lands at machine precision,
    which is what licenses reading the patterned number as physics rather than wiring. Both
    formulations meet the same golden at the same floor; they agree with each other to 1e-12.
    """
    ref = GOLDEN[f'2d__{name}__{order}__{pol}']
    got = _reconstruct_2d(name, order, pol, formulation=formulation)
    assert got.shape == ref.shape, (got.shape, ref.shape)

    scale, resid = _fit(ref, got)
    assert resid < TOL[name], f'{name} order {order} {pol}: residual {resid:.2e}'
    want = _scale_2d(pol)
    assert abs(scale - want) < TOL[name], (
        f'{name} order {order} {pol}: amplitude scale {scale:.7f}, expected {want:.7f}')


@pytest.mark.parametrize('formulation', FORMULATIONS)
@pytest.mark.parametrize('name,order,pol', CASES_2D,
                         ids=[f'{n}-{o}-{p}' for n, o, p in CASES_2D])
def test_the_2d_case_pins_both_halves_of_the_component_assignment(name, order, pol, formulation):
    """Per component, so a wrong ``Ez`` / ``Hz`` cannot hide behind a right ``Ex`` / ``Hy``.

    The aggregate residual above is a max over the whole cell and is carried by whichever
    component is largest; ``Ez`` runs several times below that here.

    **Normalised by the cell max, not by each component's own max**, the opposite of the uniform
    test above and on purpose: in s polarisation ``Hy`` is a near-null component that collects
    the truncation error of the large ones, so against its own max no honest tolerance exists.
    The cost is that this cannot see a small component wrong by a fraction of itself; the
    uniform tier is where that claim is made.
    """
    ref = GOLDEN[f'2d__{name}__{order}__{pol}']
    got = _reconstruct_2d(name, order, pol, formulation=formulation)
    scale, _ = _fit(ref, got)
    cell_scale = np.max(np.abs(ref))
    for i, comp in enumerate(NAMES):
        gap = np.max(np.abs(got[..., i] - scale * ref[..., i])) / cell_scale
        assert gap < TOL[name], f'{name} order {order} {pol}: {comp} {gap:.2e}'


def test_the_2d_case_actually_carries_ky_amplitude():
    """Guard the premise this block exists for.

    If the patch were ever tiled, mirrored or otherwise made y-invariant, every assertion above
    would still pass and the file would silently go back to testing only the transverse half.
    """
    assert not np.allclose(raster_2d('patch'), raster_2d('uniform2d'))
    patch = raster_2d('patch') != 1.0
    assert patch.sum(axis=0).max() != patch.sum(axis=1).max(), 'patch is x/y symmetric'
    assert not np.array_equal(patch, patch[::-1]) and not np.array_equal(patch, patch[:, ::-1])
    for order in (3, 9):
        for pol in ('s', 'p'):
            f = GOLDEN[f'2d__patch__{order}__{pol}']
            spread = np.max(np.abs(f - f.mean(axis=1, keepdims=True))) / np.max(np.abs(f))
            assert spread > 1e-2, f'patch order {order} {pol}: y-variation only {spread:.1e}'


@pytest.mark.parametrize('name,order,pol', [c for c in CASES_2D if c[0] == 'patch'],
                         ids=[f'{n}-{o}-{p}' for n, o, p in CASES_2D if n == 'patch'])
def test_fmmax_can_see_which_end_of_the_raster_is_minus_y(name, order, pol):
    """Row 0 of a raster ``structure`` is the ``-y`` end, at ``y = -Py/2 + Py/(2*Ny)``.

    ``docs/reference/common/conventions.md`` states it; this is where an outside code says so.
    The ``patch`` cases above would fail too if the rows were read reversed, but their names
    are about the component assignment, so a later tolerance retune, or a patch made symmetric
    in y, could drop the coverage without any name changing meaning. The pinning table cites
    this one.

    Written as an A/B rather than a threshold: on a 2D patterned field both codes sit at a 3e-2
    floor that does not shrink, and a floor can raise both sides of a ratio but cannot
    manufacture one. ``uniform2d`` is the control, not a case: its raster is y-invariant, so
    reversing it is a no-op and the ratio is exactly 1.
    """
    ref = GOLDEN[f'2d__{name}__{order}__{pol}']
    _, shipped = _fit(ref, _reconstruct_2d(name, order, pol))
    _, reversed_ = _fit(ref, _reconstruct_2d(name, order, pol, reverse_rows=True))
    assert reversed_ > 5 * shipped, (
        f'{name} order {order} {pol}: shipped row order {shipped:.2e}, rows reversed '
        f'{reversed_:.2e} -- fmmax no longer distinguishes them')


def test_the_2d_sample_grids_are_the_ones_the_cases_describe():
    """Same statement as for the 1D grids, now including the y axis they never had."""
    assert np.allclose(GOLDEN['2d_x'], x_samples_2d(), atol=1e-9)
    assert np.allclose(GOLDEN['2d_y'], y_samples_2d(), atol=1e-9)
    assert np.allclose(GOLDEN['2d_z'], z_samples_2d(), atol=1e-9)
    # Both truncations are frozen on the *same* points, which is what makes the residuals
    # comparable across truncations and so makes "the floor does not shrink" checkable.
    for order, n in ((3, 49), (9, 361)):
        assert GOLDEN[f'2d_orders__{order}'].shape == (n, 2)
