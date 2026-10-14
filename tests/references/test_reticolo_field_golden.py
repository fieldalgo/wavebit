"""The in-cell field of the **pure TE/TM 1D** path, against RETICOLO.

``test_s4_field_golden.py`` and ``test_fmmax_field_golden.py`` both reach field reconstruction
through the coupled 6-component routes. Neither touches ``src/wavebit/rcwa/_engine/fields.py::field_1d``
(``grating_type == 0``), which returns 3 components in its own layout -- TE ``(Ey, Hx, Hz)``,
TM ``(Hy, Ex, Ez)`` -- and applies its own signs. That path had no outside opinion at all until
this file. RETICOLO returns exactly those 3 components for a 1D grating, so nothing is padded
or discarded to make the shapes meet.

**This is the strongest of the three field oracles, and patterned layers are why.** Every
component except one agrees at **1e-14 on gratings**, where fmmax reaches 1e-4 and S4 cannot
go at all. The reason is not luck and is worth being precise about: RETICOLO and this project
share the 1D formulation exactly, the bridge hands RETICOLO an analytic piecewise-constant
texture, and the solver is given ``fourier_coefficients='integrated'`` -- so both assemble the
*same* convolution matrices, and every remaining difference is implementation and convention.
That is exactly what a field oracle is for (frame, component layout, z direction, amplitude
normalisation, order-index sign), and it is also the limit of the claim: agreement here does
not independently confirm the formulation, because nothing sharing a formulation can.

**The frame transform is the claim under test**, so the golden stores RETICOLO's raw cell and
the conversion lives here, in :func:`_in_our_frame`. Two parts, both measured (see the
generator) when this library's ``+z`` still pointed down into the stack:

* RETICOLO's ``+z`` points back out of the stack, so the cell reverses **as a whole**, layer
  order included -- not per layer. ``bilayer`` is what separates those two readings.
* The reversal carries the component parity ``E_z, H_x, H_y`` flip, ``E_t, H_z`` do not, which
  is the rule ``src/wavebit/rcwa/_engine/fields.py`` already states for its own frame.

Since 2026-09-21 the public frame *is* RETICOLO's: ``+z`` up from ``z = 0`` at the bottom
interface, bottom layer first. The comparison is still made in the downward depth frame
``STACKS`` is written in, so :func:`_in_our_frame` reverses RETICOLO's cell into it and
:func:`_reconstruct` applies the identical reversal and parity to ours. The two transforms
cancel: what the assertions pin is that RETICOLO's raw cell and the public cell agree entry for
entry on every slot held exactly (TM ``E_x`` on patterned stacks excepted).

With those applied there is **no free amplitude**: not a fitted scale as the fmmax golden
needs, but exactly 1, to 2.7e-14. That is asserted rather than fitted away, because a scale
free to absorb anything absorbs errors too. It is 1 after one stated phase, which is where each
code puts its unit incident wave: RETICOLO at its ``x = 0``, the cell corner, and this library
at the cell centre since 2026-09-27 (see :func:`_in_our_frame`).

**One component is excluded, and it is not swept under the rug.** TM's ``E_x`` is discontinuous
across the grating's vertical interfaces and the two codes resolve it differently at any finite
truncation -- 7.0e-2 to 1.0e-1 here, shrinking with ``truncation_order`` but non-monotonically, in the shape
of Gibbs ringing near the jump rather than of a defect. Which code is closer to the true field
is not established, and cannot be from inside this directory.
:func:`test_the_discontinuous_tm_component_is_the_known_exception` pins that gap from both
sides so it cannot silently grow, and cannot silently vanish without someone updating the
story here.
"""

from wavebit.compat import legacy as _legacy_wavebit

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))
from tools.repository_paths import repository_path
import sys

import numpy as np
import pytest

import wavebit

HERE = repository_path('tests/references')
sys.path.insert(0, str(HERE))
from tests.references._reticolo_field_cases import CASES, MAX_ORDER, NAMES, PERIOD, RES_X, RES_Z, SLOW_SLOT, STACKS, WL, Z_PARITY, key, layer_rows, n_layers, x_samples

GOLDEN_PATH = HERE / 'golden_reticolo_field.npz'

# Measured across all 24 cases: the worst exact-component gap is 6.25e-14 and the worst
# departure of the amplitude from 1 is 2.7e-14. This is round-off room above that, not slack.
TOL = 1e-12

# TM `E_x` on a patterned stack, measured over the same 24 cases: 7.0e-2 to 1.0e-1 at truncation_order=8.
# The band is deliberately loose at both ends -- it exists to keep the exception visible and
# bounded, not to pin a number. See the generator for the truncation_order convergence behind it.
SLOW_BAND = (1e-3, 3e-1)

IDS = [f'{s}-{t:g}-{p}' for s, t, p in CASES]


def _golden():
    if not GOLDEN_PATH.exists():
        pytest.skip(f'{GOLDEN_PATH.name} is missing — regenerate with '
                    'gen_goldens_reticolo_field.py')
    return np.load(GOLDEN_PATH)


def _in_our_frame(e, pol, theta):
    """RETICOLO's raw cell, in the depth frame ``STACKS`` is written in. The claim under test.

    Whole-cell reversal (RETICOLO's ``+z`` points out of the stack, so its first layer is the
    last one ``STACKS`` lists) plus the z-parity of each slot. No conjugation, and no scale.
    That depth frame was this library's until 2026-09-21; the public frame is RETICOLO's own
    now, and :func:`_reconstruct` brings ours into the depth frame the same way.

    **One stated phase, the lateral reference.** RETICOLO's ``x = 0`` is the cell's corner --
    its texture starts there (``o`` settled that) and its incident wave is unit there -- while
    this library's incidence has been unit at the cell **centre**, ``x = y = 0`` in public
    coordinates, since 2026-09-27. The same plane wave unit half a period further along is
    RETICOLO's times ``exp(-1j * k0 * sin(theta) * PERIOD/2)`` (every case is lit from air, so
    ``n_inc = 1``): a closed form, applied here once, not a fitted scale.

    A ``.conj()`` *was* applied by hand in a script deleted on 2026-08-15, unaudited and
    alongside a per-layer flip. Both were wrong, which is what an unaudited convention patch
    tends to be; the measurement that replaced them is in the generator's docstring.
    """
    centre = np.exp(-1j * 2 * np.pi / WL * np.sin(np.deg2rad(theta)) * PERIOD / 2)
    return e[::-1] * Z_PARITY[pol] * centre


def _reconstruct(stack, theta, pol):
    # STACKS and the comparison use downward depth, top layer first; public Wavebit uses
    # height, bottom layer first, as RETICOLO's raw archive does.
    thickness, structure = STACKS[stack]
    solver = _legacy_wavebit.rcwa(backend='numpy', top=1.0, bot=1.0, period=[PERIOD], truncation_order=MAX_ORDER,
                          wavelength=WL, thickness=thickness[::-1], structure=structure[::-1],
                          theta=np.deg2rad(theta), phi=None, tetm=pol,
                          # RETICOLO's texture is analytic piecewise-constant, so the exact
                          # coefficients are the ones that describe the same layer. Without
                          # this the two truncations differ and nothing lands at 1e-14.
                          fourier_coefficients='integrated', answers='field')
    solver.solve()
    cell = np.asarray(solver.reconstruct_field(sample_x=RES_X, sample_y=1, sample_z=RES_Z))
    return cell[::-1, 0, :, :] * Z_PARITY[pol]                       # (sample_z*n_layers, sample_x, 3)


def _exact_slots(stack, pol):
    """Which of the three slots both codes are held to entry for entry."""
    slow = SLOW_SLOT[pol] if stack != 'uniform' else None
    return [c for c in range(3) if c != slow]


@pytest.mark.parametrize('stack,theta,pol', CASES, ids=IDS)
def test_the_field_matches_reticolo(stack, theta, pol):
    """Every component the two codes can both resolve, entry for entry, at 1e-12.

    Component by component rather than in aggregate: a whole-cell norm is dominated by the
    largest slot, and a component running well below it could be wholly wrong and still leave
    a small total. That is the shape the 2026-08-14 sign defect had.
    """
    g = _golden()
    ref = _in_our_frame(g[key(stack, theta, pol)], pol, theta)
    got = _reconstruct(stack, theta, pol)
    assert got.shape == ref.shape, (got.shape, ref.shape)

    cell = np.max(np.abs(ref))
    for c in _exact_slots(stack, pol):
        amp = np.max(np.abs(ref[..., c]))
        if amp < 1e-9 * cell:
            # A slot that is identically zero on a uniform slab at normal incidence (TE `Hz`,
            # TM `Ez`): there is no relative gap to take, so require ours to vanish too rather
            # than skipping and claiming a pass.
            assert np.max(np.abs(got[..., c])) < 1e-9 * cell, NAMES[pol][c]
            continue
        gap = np.max(np.abs(got[..., c] - ref[..., c])) / amp
        assert gap < TOL, f'{NAMES[pol][c]}: {gap:.2e}'


@pytest.mark.parametrize('stack,theta,pol', CASES, ids=IDS)
def test_the_amplitude_needs_no_fitted_scale(stack, theta, pol):
    """RETICOLO's incident normalisation is this project's, exactly — assert it, don't fit it.

    The fmmax field golden has to fit one complex scale per case because "unit mode amplitude"
    is not "unit incident flux". Here the two conventions coincide, so the honest test is that
    the best-fit scale *is* 1. A scale left free to absorb a factor absorbs a defect just as
    happily, and this is the assertion that keeps that door shut.

    **For TM they coincide because every case is lit from air.** The bridge's ``res3`` call
    passes ``einc = 1``, which RETICOLO's own 1D TM examples fill with ``PlaneWave_H(2)``: its TM
    input is the incident ``H_y``. This project's pure TM solve is lit by unit incident **E**
    (since 2026-09-27, as every other path is), and the two are the same number only where the
    incidence medium's ``n/mu`` is 1. A future case lit through glass needs ``x n_inc/mu_inc``
    on RETICOLO's side; a scale of ``n_inc`` there is that conversion, not a solver regression.
    """
    g = _golden()
    ref = _in_our_frame(g[key(stack, theta, pol)], pol, theta)
    got = _reconstruct(stack, theta, pol)

    keep = _exact_slots(stack, pol)               # TM `E_x` would bias the fit by ~1e-2
    r, q = ref[..., keep].ravel(), got[..., keep].ravel()
    scale = np.vdot(r, q) / np.vdot(r, r)
    assert abs(scale - 1.0) < TOL, f'amplitude scale {scale:.14f}'


@pytest.mark.parametrize('stack,theta,pol',
                         [c for c in CASES if SLOW_SLOT[c[2]] is not None
                          and c[0] != 'uniform'],
                         ids=[f'{s}-{t:g}-{p}' for s, t, p in CASES
                              if SLOW_SLOT[p] is not None and s != 'uniform'])
def test_the_discontinuous_tm_component_is_the_known_exception(stack, theta, pol):
    """TM's ``E_x`` on a patterned layer: bounded from both sides, on purpose.

    The upper bound says the gap is a slow truncation difference and not a swapped component
    or a sign — either of those is O(1). The lower bound says the exception is still real: if
    a change ever makes ``E_x`` agree to machine precision, this fails and whoever did it has
    to come here and rewrite the story rather than inherit a stale caveat.
    """
    g = _golden()
    c = SLOW_SLOT[pol]
    ref = _in_our_frame(g[key(stack, theta, pol)], pol, theta)[..., c]
    got = _reconstruct(stack, theta, pol)[..., c]

    gap = np.max(np.abs(got - ref)) / np.max(np.abs(ref))
    lo, hi = SLOW_BAND
    assert gap < hi, f'{NAMES[pol][c]} gap {gap:.2e} — too large to be truncation alone'
    assert gap > lo, (f'{NAMES[pol][c]} gap {gap:.2e} — the documented exception appears to '
                      'have gone away; update this file and the generator rather than the bound')


def test_the_golden_is_reticolos_raw_cell_in_reticolos_own_frame():
    """The stored data must still need the transform — otherwise the transform is untested.

    If a regeneration ever pre-applied the frame conversion, every assertion above would keep
    passing while :func:`_in_our_frame` quietly stopped being checked by anything.
    """
    g = _golden()
    raw = g[key('bilayer', 24.0, 'te')]
    got = _reconstruct('bilayer', 24.0, 'te')
    assert np.max(np.abs(got - raw)) / np.max(np.abs(raw)) > 1e-2, (
        'the golden already matches without the frame transform')


def test_the_stored_grids_are_the_ones_the_cases_describe():
    """x, z and the layer-row rule, stated rather than trusted."""
    g = _golden()
    assert np.allclose(g['x'], x_samples(), atol=1e-9)
    for stack in STACKS:
        thickness = STACKS[stack][0]
        z = g[f'{stack}__z']
        assert z.shape == (RES_Z * n_layers(stack),), z.shape
        assert len(layer_rows(n_layers(stack))) == z.shape[0]
        # RETICOLO's z runs 0 .. sum(thickness) from the far side back; each layer contributes
        # `RES_Z` samples inclusive of both its faces, which is this project's sampling too.
        edges = np.concatenate([[0.0], np.cumsum(thickness[::-1])])
        want = np.concatenate([np.linspace(edges[k], edges[k + 1], RES_Z)
                               for k in range(len(thickness))])
        assert np.allclose(z, want, atol=1e-9), (z, want)


@pytest.mark.parametrize('stack', [s for s in STACKS if s != 'uniform'])
def test_the_patterned_stacks_are_actually_patterned(stack):
    """Guard the premise, on RETICOLO's own index map rather than on our raster.

    ``o`` is what settled the x origin in the first place; asserting against it here means the
    two codes are still describing the same structure, not merely the same numbers.

    **It is also the check on the quantity, and that is new.** ``o`` is a refractive index and
    ``STACKS`` is a permittivity, so this comparison only closes if the bridge really does take
    the root on the way out -- a bridge that forwarded ``eps`` unconverted would put ``4`` where
    RETICOLO reports ``2`` and fail here by O(1), *before* any field is compared. That matters
    because the failure it stands in front of is otherwise silent: an unconverted permittivity
    is a legal medium, and the field tests would read the disagreement as a solver defect.
    """
    g = _golden()
    o = g[f'{stack}__o']
    assert o.shape == (RES_Z * n_layers(stack), RES_X)
    assert np.abs(o - o[:, :1]).max() > 1e-9, 'RETICOLO saw a uniform layer'

    # and the index map must be our own structure, rooted and sampled on the same x, **complex**
    # — the `lossy` stack is where that matters, and it doubles as a convention check on the
    # branch: `refractive_index_from_eps(3.9975+0.2j)` is `2+0.05j`, which is what RETICOLO stores,
    # so both codes spell loss as `Im(n) > 0` and the principal root is the one that lands there.
    structure = STACKS[stack][1]
    ours_index = wavebit.refractive_index_from_eps(structure)
    idx = np.clip((x_samples() / (PERIOD / structure.shape[2])).astype(int), 0,
                  structure.shape[2] - 1)
    for k in range(n_layers(stack)):
        # RETICOLO's layer order is reversed relative to ours
        ours = ours_index[n_layers(stack) - 1 - k, 0, idx]
        theirs = o[k * RES_Z:(k + 1) * RES_Z]
        assert np.allclose(theirs, ours[None, :], atol=1e-9), (k, theirs[0], ours)
