"""The in-cell field against fmmax — including on a **patterned** layer, which S4 cannot reach.

``test_s4_field_golden.py`` is exact but stops at uniform stacks: on a grating S4 and this
project disagree at 1e-3, and up to 7.6e-1 on ``E_z``, because their truncation and
factorization cannot be matched through S4's interface. Both knobs are reachable in fmmax, so
this file is where patterned field reconstruction finally gets an outside opinion.

Two tiers, and the file keeps them apart on purpose:

* **uniform** — 1e-13. There is no truncation to disagree about, so this is exactness, and it
  is what proves the generator's conventions (axis order, cell-centred x, the raster
  transpose) rather than the physics.
* **patterned** — 1e-4. Converged agreement, not machine precision: the two codes' vector
  factorizations still differ in detail once a discontinuity exists. That floor is measured
  (1.9e-5 for s, 7.2e-5 for p) and stated, not tuned until green.

A 1e-4 oracle is worth having precisely because of what this repository has actually shipped
wrong. The sign-convention defect fixed in ``9824f0d`` moved fields by 2e0; a transposed raster
reverses the s/p split outright; a swapped component is order-1. Every one of those is four
orders above this floor. What 1e-4 cannot see is slow numerical drift — that is
``tests/data/golden_field.npz``'s job, and it is a different claim.

**The amplitude convention is fitted, and then checked against a closed form.** fmmax's "unit
mode amplitude" is not this project's "unit incident flux". Each case fits one complex scale by
least squares — and the fit is then required to equal ``-cos(theta)`` for s and ``-1`` for p,
which is what that ratio turned out to be, times the stated phase between the two codes' unit
incidence points (fmmax's corner, this library's centre; :func:`expected_scale`). Stating the
*dependence* rather than tabulating
three constants is deliberate: a table is satisfied by anything that reproduces the numbers,
whereas a case added at a new angle is genuinely checked by the formula.
"""
import pathlib
import sys

import numpy as np
import pytest

import wavebit

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _fmmax_field_cases import (CASES, MAX_ORDER, PERIOD, RES_Z,  # noqa: E402
                                THICK, WL, fine_res_x, raster, x_samples)

GOLDEN = np.load(HERE / 'golden_fmmax_field.npz')

TOL = {'uniform': 1e-13, 'grating': 1e-4}
NAMES = ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')

def expected_scale(theta, pol):
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
    return ratio * np.exp(-1j * 2 * np.pi / WL * np.sin(np.deg2rad(theta)) * PERIOD / 2)


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
    solver = wavebit.rcwa(backend='numpy', top=1.0, bot=1.0, period=(PERIOD, PERIOD),
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
