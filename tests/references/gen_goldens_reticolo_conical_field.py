"""Freeze RETICOLO's in-layer field on the **conical** path into
``golden_reticolo_conical_field.npz``.

Run once, on a machine with Octave and a RETICOLO V10 tree;
``test_reticolo_conical_field_golden.py`` then runs anywhere.

    RETICOLO_PATH=... conda run -n fieldwave python tests/references/gen_goldens_reticolo_conical_field.py

Rerun it only to add cases -- never to make a failing test pass.

**Why this exists.** ``field_1d_conical`` is a 1D grating at a non-zero azimuth: s and p no
longer decouple, and one coupled mode set produces all six components. Of the three field
oracles that existed before this, S4 runs the conical path but only on *uniform* stacks, and
fmmax reaches a patterned layer only through the **2D** route. A patterned layer on the
conical path -- an azimuth and a discontinuity at once, where the s/p mixing and the
factorization both have to be right -- was checked by nothing outside this repository.

**What it is worth, measured 2026-08-15.** The same two tiers as the pure TE/TM golden beside
it. Uniform stacks land at **1e-15** on every component that is not identically zero, which is
what proves the conventions rather than the physics. Patterned stacks land at **1e-13 on five
of the six components**; ``E_x``, the one normal to the grating's vertical interfaces and
therefore discontinuous, sits at 7.0e-2 to 7.5e-2 and does not converge cleanly -- 1.3e-1,
1.4e-1, 4.8e-2, 5.9e-2 at ``truncation_order = 4, 8, 16, 32``. That is the same Gibbs-limited disagreement
the pure TM path shows in the same slot, and it is excluded from the tight comparison and
bounded on its own rather than hidden.

**The calibration is the same one, which is the point of checking it here.** Whole-cell z
reversal plus the declared frame's z-parity, and **no free amplitude at all** -- the fitted
scale is 1 to ~1e-15 (after one stated lateral phase since 2026-09-27, which the test applies:
RETICOLO's incident wave is unit at the cell corner, this library's at the centre). Nothing new
had to be discovered: the transform established on the pure
TE/TM path transfers unchanged to the coupled one, and a full search over all 720 component
permutations x z-flip x conjugation confirms the identity permutation with no conjugation as
the unique best fit. Had the conical path carried its own frame quirk, that search is what
would have found it.

**What is new is the array shape**, and it is measured rather than assumed: ``res3``'s conical
branch returns ``(z, x, y, 6)`` -- **x before y**, the transpose of this project's
``(z, y, x, 6)`` -- with ``y`` of length 1 because the ``.m`` sets ``y = 0`` for
``grating_type == 1``. The transpose lives in the test with the rest of the frame conversion;
this file stores what RETICOLO returned.

**Citation.** RETICOLO's terms require Hugonin & Lalanne (arXiv:2101.00901), *and* P. Lalanne
and M.P. Jurek, J. Mod. Opt. **45**, 1357-1374 (1998) whenever near-field distributions are
shown -- which a field golden is. See ``README.md``.
"""
import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))   # repo root, for `import wavebit`
sys.path.insert(0, str(HERE))              # the bridge and the cases module beside it

RETICOLO_PATH = os.environ.get('RETICOLO_PATH')
if not RETICOLO_PATH:
    raise SystemExit('set RETICOLO_PATH to a reticolo_allege_v10 directory to regenerate this '
                     'golden; the committed .npz is what the tests read.')
RETICOLO_PATH = os.path.expanduser(RETICOLO_PATH)

OUT = HERE / 'golden_reticolo_conical_field.npz'


def main():
    from oct2py import octave
    octave.addpath(octave.genpath(RETICOLO_PATH))

    from interface.Reticolo import Reticolo
    from _reticolo_conical_field_cases import (CASES, MAX_ORDER, PERIOD, RES_X, RES_Z, STACKS,
                                               WL, key, layer_rows, n_layers, x_samples)

    reti = Reticolo()
    out = {}
    for stack, theta, phi, pol in CASES:
        thickness, structure = STACKS[stack]
        # The `.m`'s conical branch drives RETICOLO with `einc = [0, 1]` for TE and `[1, 0]`
        # for TM, which is what `pol` selects -- the same two incidences `psi = pi/2` and
        # `psi = 0` name on this side.
        #
        # `structure` is permittivity and the bridge converts it to RETICOLO's index; see
        # `interface/Reticolo.py::_index`.
        *_, e, z, o = reti.run_res3(
            grating_type=1, period=[PERIOD], truncation_order=MAX_ORDER, structure=structure, thickness=thickness,
            theta=np.deg2rad(theta), phi=np.deg2rad(phi), pol=0 if pol == 's' else 1,
            wavelength=WL, top=1.0, bot=1.0, res3_npts=RES_Z)
        e, z, o = np.asarray(e), np.asarray(z).ravel(), np.asarray(o)

        rows = (n_layers(stack) + 2) * RES_Z
        assert e.shape == (rows, RES_X, 1, 6), e.shape
        assert z.shape == (rows,) and o.shape == (rows, RES_X), (z.shape, o.shape)

        # Same guard the pure 1D generator carries: every row kept must be inside a layer and
        # every row dropped must be ambient, checked against `res3`'s own index map.
        keep = layer_rows(n_layers(stack))
        amb = np.setdiff1d(np.arange(rows), keep)
        assert np.allclose(o[amb], 1.0), 'a dropped row is not ambient -- the row rule moved'
        assert not np.allclose(o[keep], 1.0), 'a kept row is ambient -- the row rule moved'

        # Stored RAW: RETICOLO's z direction, its component order and its (z, x, y) axes. The
        # frame transform is the claim under test and belongs in the test.
        out[key(stack, theta, phi, pol)] = e[keep]
        out[f'{stack}__z'] = z[keep]
        out[f'{stack}__o'] = o[keep]

    out['x'] = np.linspace(0, PERIOD, RES_X)     # what the .m hardcodes
    assert np.allclose(out['x'], x_samples())
    np.savez_compressed(str(OUT), **out)
    print(f'wrote {len(out)} arrays to {OUT.name}; {len(CASES)} cases, truncation_order={MAX_ORDER}, '
          f'{2 * MAX_ORDER + 1} orders')


if __name__ == '__main__':
    try:
        import oct2py  # noqa: F401
    except ImportError:
        sys.exit('needs oct2py and Octave; the committed .npz is what the tests read.')
    main()
