"""The in-cell field on the **conical** path, against RETICOLO.

``field_1d_conical`` is a 1D grating at a non-zero azimuth: s and p no longer decouple, and one
coupled mode set produces all six components. Before this file, a **patterned** layer on that
path had no outside opinion at all — ``test_s4_field_golden.py`` runs the conical path but only
on uniform stacks, and ``test_fmmax_field_golden.py`` reaches a patterned layer through the 2D
route. This is the case with an azimuth *and* a discontinuity, where the s/p mixing and the
factorization have to be right at the same time.

Two tiers, measured across all 24 cases and kept apart on purpose:

* **uniform** — 1e-12. Nothing truncates, so this is the conventions: frame, component order,
  z direction, the ``(z, x, y)`` axis order, amplitude normalisation, and the s/p rotation.
* **patterned** — the same 1e-12 on five components of six. ``E_x`` is the exception; it gets
  its own bound below.

**The calibration is the pure TE/TM golden's, unchanged, and that is the finding.** Whole-cell
z reversal plus the declared frame's z-parity, and **no fitted amplitude** — the scale is 1 to
4.2e-14. Nothing conical-specific had to be discovered. The generator records the search that
established there was nothing to discover: over all 720 component permutations x z-flip x
conjugation, the identity permutation with no conjugation is the unique best fit.

What *is* conical-specific is the array shape, and it was measured: ``res3``'s conical branch
returns ``(z, x, y, 6)`` — **x before y**, the transpose of this project's ``(z, y, x, 6)``.

**Two structural facts this file also pins**, both free consequences of having the oracle:

* At ``phi = 0`` the conical solve must decouple, leaving the complementary slots *identically*
  zero — and it does, exactly, not approximately.
* At ``phi = 0`` it must also agree with the pure TE/TM golden beside it, which ties the two
  RETICOLO goldens together and checks both harnesses at once.
"""
import pathlib
import sys

import numpy as np
import pytest

import wavebit

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _reticolo_conical_field_cases import (CASES, MAX_ORDER, NAMES, PERIOD, RES_X,  # noqa: E402
                                           RES_Z, SLOW_SLOT, STACKS, WL, Z_PARITY,
                                           key, layer_rows, n_layers, x_samples)

GOLDEN_PATH = HERE / 'golden_reticolo_conical_field.npz'

# Measured across all 24 cases: the worst exact-component gap is 9.75e-14 and the worst
# departure of the amplitude from 1 is 4.2e-14. Round-off room above that, not slack.
TOL = 1e-12

# `E_x` on a patterned stack, measured over the same cases: 8.6e-2 to 1.2e-1 at truncation_order=8. Loose at
# both ends on purpose -- the band keeps the exception visible and bounded, it does not pin a
# number. See the generator for the truncation_order behaviour behind it.
SLOW_BAND = (1e-3, 3e-1)

IDS = [f'{s}-{t:g}-{p:g}-{q}' for s, t, p, q in CASES]


def _golden():
    if not GOLDEN_PATH.exists():
        pytest.skip(f'{GOLDEN_PATH.name} is missing — regenerate with '
                    'gen_goldens_reticolo_conical_field.py')
    return np.load(GOLDEN_PATH)


def _in_our_frame(e):
    """RETICOLO's raw conical cell, in the depth frame of ``STACKS``. The claim under test.

    Three parts, none of them guessed: transpose ``(z, x, y, 6)`` to this project's
    ``(z, y, x, 6)``; reverse the cell as a whole, because RETICOLO's ``+z`` points back out of
    the stack and its first layer is the last one ``STACKS`` lists; and apply the z-parity,
    ``E_z, H_x, H_y`` flip and ``E_t, H_z`` do not. No conjugation and no scale.

    That depth frame was this library's until 2026-09-21. The public frame is RETICOLO's own
    now (``+z`` up, bottom layer first), and :func:`_reconstruct` brings ours into the depth
    frame by the same reversal and parity, so the two cancel: after the transpose, RETICOLO's
    raw cell and the public cell agree entry for entry on every slot held exactly (``E_x`` on
    patterned stacks excepted).
    """
    return e.transpose(0, 2, 1, 3)[::-1] * Z_PARITY


def _reconstruct(stack, theta, phi, pol):
    # STACKS and the comparison use downward depth, top layer first; public Wavebit uses
    # height, bottom layer first, as RETICOLO's raw archive does.
    thickness, structure = STACKS[stack]
    solver = wavebit.rcwa(backend=0, top=1.0, bot=1.0, period=[PERIOD], truncation_order=MAX_ORDER,
                          wavelength=WL, thickness=thickness[::-1], structure=structure[::-1],
                          theta=np.deg2rad(theta), phi=np.deg2rad(phi),
                          psi=np.pi / 2 if pol == 's' else 0.0,
                          # RETICOLO's texture is analytic piecewise-constant, so the exact
                          # coefficients describe the same layer. Without this the two
                          # truncations differ and nothing lands at 1e-13.
                          fourier_coefficients='integrated', answers='field')
    solver.solve()
    cell = np.asarray(solver.reconstruct_field(sample_x=RES_X, sample_y=1, sample_z=RES_Z))
    return cell[0, ::-1] * Z_PARITY                                 # the one incidence requested


def _classify(ref, stack):
    """Split the six slots into the ones held exactly, the zero ones, and the slow one.

    Order matters: a slot that is identically zero is zero *first*, even when it is also the
    slow slot. At ``phi = 0`` the conical solve decouples and ``E_x`` vanishes for s incidence,
    so on those cases slot 0 is a zero slot rather than the known exception.
    """
    cell = np.max(np.abs(ref))
    zero = [c for c in range(6) if np.max(np.abs(ref[..., c])) < 1e-9 * cell]
    slow = SLOW_SLOT if (stack != 'uniform' and SLOW_SLOT not in zero) else None
    exact = [c for c in range(6) if c not in zero and c != slow]
    return exact, zero, slow


@pytest.mark.parametrize('stack,theta,phi,pol', CASES, ids=IDS)
def test_the_conical_field_matches_reticolo(stack, theta, phi, pol):
    """Every component both codes can resolve, entry for entry, at 1e-12.

    Component by component rather than in aggregate: a whole-cell norm is dominated by the
    largest slot, so a small component could be wholly wrong and still leave a small total.
    """
    g = _golden()
    ref = _in_our_frame(g[key(stack, theta, phi, pol)])
    got = _reconstruct(stack, theta, phi, pol)
    assert got.shape == ref.shape, (got.shape, ref.shape)

    exact, zero, _ = _classify(ref, stack)
    cell = np.max(np.abs(ref))
    for c in exact:
        gap = np.max(np.abs(got[..., c] - ref[..., c])) / np.max(np.abs(ref[..., c]))
        assert gap < TOL, f'{NAMES[c]}: {gap:.2e}'
    for c in zero:
        # No relative gap exists against zero, so require ours to vanish too rather than
        # skipping and calling that a pass.
        assert np.max(np.abs(got[..., c])) < 1e-9 * cell, f'{NAMES[c]} should vanish'


@pytest.mark.parametrize('stack,theta,phi,pol', CASES, ids=IDS)
def test_the_amplitude_needs_no_fitted_scale(stack, theta, phi, pol):
    """RETICOLO's incident normalisation is this project's on the conical path too.

    The fmmax field golden fits one complex scale per case because its "unit mode amplitude" is
    not "unit incident flux". Here the conventions coincide, so the honest test is that the
    best-fit scale *is* 1 — a scale free to absorb a factor absorbs a defect just as happily.
    """
    g = _golden()
    ref = _in_our_frame(g[key(stack, theta, phi, pol)])
    got = _reconstruct(stack, theta, phi, pol)

    exact, _, _ = _classify(ref, stack)            # `E_x` would bias the fit by ~1e-1
    r, q = ref[..., exact].ravel(), got[..., exact].ravel()
    scale = np.vdot(r, q) / np.vdot(r, r)
    assert abs(scale - 1.0) < TOL, f'amplitude scale {scale:.14f}'


@pytest.mark.parametrize('stack,theta,phi,pol',
                         [c for c in CASES if c[0] != 'uniform' and c[2] != 0.0],
                         ids=[f'{s}-{t:g}-{p:g}-{q}' for s, t, p, q in CASES
                              if s != 'uniform' and p != 0.0])
def test_the_discontinuous_component_is_the_known_exception(stack, theta, phi, pol):
    """``E_x`` on a patterned layer: bounded from both sides, on purpose.

    ``E_x`` is normal to the grating's vertical interfaces, so it is the discontinuous one and
    the two codes resolve it differently at any finite truncation — the same slot, and the same
    behaviour, as the pure TM path shows in ``test_reticolo_field_golden.py``.

    The upper bound says it is a truncation difference and not a swapped component or a sign,
    either of which is O(1). The lower bound says the exception is still real: if it ever
    agrees to machine precision, this fails and whoever did it has to rewrite the story rather
    than inherit a stale caveat.
    """
    g = _golden()
    ref = _in_our_frame(g[key(stack, theta, phi, pol)])
    got = _reconstruct(stack, theta, phi, pol)
    c = SLOW_SLOT

    gap = np.max(np.abs(got[..., c] - ref[..., c])) / np.max(np.abs(ref[..., c]))
    lo, hi = SLOW_BAND
    assert gap < hi, f'{NAMES[c]} gap {gap:.2e} — too large to be truncation alone'
    assert gap > lo, (f'{NAMES[c]} gap {gap:.2e} — the documented exception appears to have '
                      'gone away; update this file and the generator, not the bound')


@pytest.mark.parametrize('stack', list(STACKS))
@pytest.mark.parametrize('pol', ['s', 'p'])
def test_a_zero_azimuth_decouples_into_pure_te_or_tm(stack, pol):
    """At ``phi = 0`` the conical solve must fall back to the pure layouts — exactly.

    s incidence leaves ``(Ey, Hx, Hz)`` and p leaves ``(Ex, Ez, Hy)``; the other three are not
    small, they are zero. Asserting that against RETICOLO's own numbers is what says the s/p
    rotation is right at the one azimuth where the answer is independently known.
    """
    g = _golden()
    ref = _in_our_frame(g[key(stack, 24.0, 0.0, pol)])
    got = _reconstruct(stack, 24.0, 0.0, pol)
    cell = np.max(np.abs(ref))

    vanish = [0, 2, 4] if pol == 's' else [1, 3, 5]     # Ex,Ez,Hy  /  Ey,Hx,Hz
    alive = [c for c in range(6) if c not in vanish]
    for c in vanish:
        assert np.max(np.abs(ref[..., c])) < 1e-9 * cell, f'RETICOLO: {NAMES[c]} not zero'
        assert np.max(np.abs(got[..., c])) < 1e-9 * cell, f'ours: {NAMES[c]} not zero'
    for c in alive:
        assert np.max(np.abs(ref[..., c])) > 1e-3 * cell, f'{NAMES[c]} should carry the field'


@pytest.mark.parametrize('stack', list(STACKS))
@pytest.mark.parametrize('pol', ['s', 'p'])
def test_at_zero_azimuth_the_two_reticolo_goldens_agree(stack, pol):
    """The conical golden at ``phi = 0`` against the pure TE/TM golden beside it.

    Two different RETICOLO code paths (``grating_type`` 1 and 0) on the same structure, so this
    checks both harnesses at once — the row rule, the frame transform and the two different
    array shapes — against something other than this project. It is the one comparison here
    that does not involve the solver at all.

    Two things make RETICOLO's two paths the same field here, and neither is general. The
    angle is positive, so the incident order's ``s_hat`` is ``+y`` and the two paths' incident
    fields are one field (RETICOLO's 1D and conical *amplitudes* differ by
    ``sign(kx_m) * sign(kx_inc)`` per order, measured on V10 on 2026-09-27; a field sums over
    the orders and does not see that). And the stacks are lit from air, where RETICOLO's 1D TM
    input -- the incident ``H_y`` -- is also a unit incident E.
    """
    import _reticolo_field_cases as flat

    conical = _golden()
    flat_path = HERE / 'golden_reticolo_field.npz'
    if not flat_path.exists():
        pytest.skip('golden_reticolo_field.npz is missing')
    flat_g = np.load(flat_path)

    # the two case lists must describe the same scene, or this compares unrelated numbers
    assert (flat.WL, flat.PERIOD, flat.MAX_ORDER, flat.RES_X, flat.RES_Z) == \
           (WL, PERIOD, MAX_ORDER, RES_X, RES_Z)
    assert np.array_equal(flat.STACKS[stack][1], STACKS[stack][1])
    assert flat.STACKS[stack][0] == STACKS[stack][0]

    tetm = 'te' if pol == 's' else 'tm'
    ref3 = flat_g[flat.key(stack, 24.0, tetm)][::-1] * flat.Z_PARITY[tetm]   # (z, x, 3)
    ref6 = _in_our_frame(conical[key(stack, 24.0, 0.0, pol)])[:, 0]          # (z, x, 6)

    # TE is (Ey, Hx, Hz) -> slots 1, 3, 5;  TM is (Hy, Ex, Ez) -> slots 4, 0, 2
    slots = (1, 3, 5) if pol == 's' else (4, 0, 2)
    for j, c in enumerate(slots):
        scale = np.max(np.abs(ref3[..., j]))
        gap = np.max(np.abs(ref6[..., c] - ref3[..., j])) / scale
        assert gap < 1e-12, f'{NAMES[c]}: {gap:.2e}'


def test_the_golden_is_reticolos_raw_cell_in_reticolos_own_frame():
    """The stored data must still need the transform, or the transform is tested by nothing."""
    g = _golden()
    raw = g[key('bilayer', 24.0, 15.0, 's')]
    got = _reconstruct('bilayer', 24.0, 15.0, 's')
    assert raw.shape != got.shape or np.max(np.abs(got - raw)) / np.max(np.abs(raw)) > 1e-2


def test_the_stored_grids_are_the_ones_the_cases_describe():
    """x, z and the layer-row rule, stated rather than trusted."""
    g = _golden()
    assert np.allclose(g['x'], x_samples(), atol=1e-9)
    for stack in STACKS:
        thickness = STACKS[stack][0]
        z = g[f'{stack}__z']
        assert z.shape == (RES_Z * n_layers(stack),), z.shape
        assert len(layer_rows(n_layers(stack))) == z.shape[0]
        edges = np.concatenate([[0.0], np.cumsum(thickness[::-1])])
        want = np.concatenate([np.linspace(edges[k], edges[k + 1], RES_Z)
                               for k in range(len(thickness))])
        assert np.allclose(z, want, atol=1e-9), (z, want)


@pytest.mark.parametrize('stack', [s for s in STACKS if s != 'uniform'])
def test_the_patterned_stacks_are_actually_patterned(stack):
    """Guard the premise on RETICOLO's own index map, not on our raster.

    And on the *quantity* too, as its pure-1D twin explains: ``o`` is an index, ``STACKS`` is a
    permittivity, so the comparison closes only if the bridge took the root on the way out.
    """
    g = _golden()
    o = g[f'{stack}__o']
    assert o.shape == (RES_Z * n_layers(stack), RES_X)
    assert np.abs(o - o[:, :1]).max() > 1e-9, 'RETICOLO saw a uniform layer'

    structure = STACKS[stack][1]
    ours_index = wavebit.index_from_eps(structure)
    idx = np.clip((x_samples() / (PERIOD / structure.shape[2])).astype(int), 0,
                  structure.shape[2] - 1)
    for k in range(n_layers(stack)):
        ours = ours_index[n_layers(stack) - 1 - k, 0, idx]  # RETICOLO's layer order is reversed
        theirs = o[k * RES_Z:(k + 1) * RES_Z]
        assert np.allclose(theirs, ours[None, :], atol=1e-9), (k, theirs[0], ours)
