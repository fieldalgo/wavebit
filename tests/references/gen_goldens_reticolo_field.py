"""Freeze RETICOLO's in-layer **field** into ``golden_reticolo_field.npz``.

Run once, on a machine with Octave and a RETICOLO V10 tree; ``test_reticolo_field_golden.py``
then runs anywhere.

    RETICOLO_PATH=... conda run -n fieldwave python tests/references/gen_goldens_reticolo_field.py

Rerun it only to add cases -- never to make a failing test pass, since the whole value of this
oracle is that its numbers did not come from this codebase.

**Why this exists.** ``golden_s4_field.npz`` and ``golden_fmmax_field.npz`` both check field
reconstruction through the coupled 6-component paths. Neither touches the pure TE/TM route
(``grating_type == 0``, ``field_1d``), which has its own 3-component layout and its own signs,
and which was therefore checked by nothing outside this repository. RETICOLO returns exactly 3
components for a 1D grating, so it fits the shape with nothing to pad or discard.

**What it is worth, measured 2026-08-15.** Better than either of the other two, and by a lot:
**1e-14 on every component of the uniform stacks, and on five of the six on the patterned
ones** -- patterned included, which is what fmmax could only reach at 1e-4 and S4 not at all.
That is not luck. RETICOLO and this project share the 1D formulation exactly (Li's rules), and
the bridge hands RETICOLO an analytic piecewise-constant texture while the solver is given
``fourier_coefficients='integrated'``, so both build the *same* convolution matrices. What
agreement at 1e-14 then establishes is the implementation and every convention around it --
frame, component layout, z direction, amplitude normalisation, order-index sign -- which is
precisely what a field oracle is for. It does not independently confirm the formulation
choice; nothing that shares a formulation can.

**The exception, and it is a real one.** TM's ``E_x`` is discontinuous across the grating's
vertical interfaces, and the two codes resolve it differently at any finite truncation:
7.0e-2 to 1.0e-1 across these cases at ``max_order=8``. On the ``grating`` case at 24 degrees it
decreases with truncation but **not monotonically** -- 3.1e-1, 2.3e-1, 9.7e-2, 1.1e-1, 3.8e-2,
4.4e-2 at ``max_order = 2, 4, 8, 16, 32, 64``, roughly 8x for a 32x increase in orders. That is Gibbs
ringing: the gap is set by the samples nearest a discontinuity (10 nm away here) and the local
error there oscillates with the number of retained terms. It is **not** a rescaling (a
per-component complex fit does not remove it) and **not** a disagreement about which side of
the interface to divide by -- forming ``D_x = eps*E_x`` leaves a gap of the same order,
1.7e-1 / 1.6e-1 / 8.1e-2 / 4.8e-2 at ``max_order = 4, 8, 16, 32``.

Which code is closer to the true ``E_x`` is **not** established here, and nothing in this
directory can settle it: that needs a reference finer than either. The test excludes the slot
on patterned stacks and bounds it from both sides rather than hiding it.

**Three things that had to be measured before any of this lined up**, each of which leaves an
O(1) residual when guessed wrong:

* ``res3``'s second and third outputs were being discarded by the wrapper. ``z`` alone cannot
  locate a layer -- a single-slab cell has eight rows at ``z == 0``, of which exactly one is
  the slab's top face -- and ``o``, the refractive index at every sample, says which. The
  wrapper now forwards both. Aligning on the six *nonzero* z values instead, which is the
  reading ``z`` invites, drops the slab's top face and leaves the whole comparison O(1) out.
* **RETICOLO's ``+z`` points out of the stack**, so the cell reverses as a whole, layer order
  included. On one layer that is indistinguishable from a per-layer reversal; ``bilayer``
  separates them (1.3e0 the wrong way round, 2.8e-14 the right way).
* The reversal carries the **component parity** ``E_z, H_x, H_y`` flip / ``E_t, H_z`` do not,
  which is the rule ``_core/fields.py`` already states for its own frame. With the parity and
  the reversal applied there is **no free amplitude at all** -- not a fitted scale, as the
  fmmax golden needs, but exactly 1.

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

# No default: where RETICOLO is installed is a property of the machine, not of this project.
RETICOLO_PATH = os.environ.get('RETICOLO_PATH')
if not RETICOLO_PATH:
    raise SystemExit('set RETICOLO_PATH to a reticolo_allege_v10 directory to regenerate this '
                     'golden; the committed .npz is what the tests read.')
RETICOLO_PATH = os.path.expanduser(RETICOLO_PATH)

OUT = HERE / 'golden_reticolo_field.npz'


def main():
    from oct2py import octave
    octave.addpath(octave.genpath(RETICOLO_PATH))

    from interface.Reticolo import Reticolo
    from _reticolo_field_cases import (CASES, MAX_ORDER, PERIOD, RES_X, RES_Z, STACKS, WL,
                                       key, layer_rows, n_layers, x_samples)

    reti = Reticolo()
    out = {}
    for stack, theta, pol in CASES:
        thickness, structure = STACKS[stack]
        # Permittivity in, index out: the bridge converts (`interface/Reticolo.py::_index`), so
        # the numbers here are the same ones the test hands the solver. `top` / `bot` are
        # vacuum, where eps and n happen to be the same 1.0 -- which is exactly why they are a
        # bad place to notice a missing conversion, and why the layers are not.
        *_, e, z, o = reti.run_res3(
            grating_type=0, period=[PERIOD], max_order=MAX_ORDER, structure=structure, thickness=thickness,
            theta=np.deg2rad(theta), phi=None, pol=0 if pol == 'te' else 1,
            wavelength=WL, top=1.0, bot=1.0, res3_npts=RES_Z)
        e, z, o = np.asarray(e), np.asarray(z).ravel(), np.asarray(o)

        rows = (n_layers(stack) + 2) * RES_Z
        assert e.shape == (rows, RES_X, 3), e.shape
        assert z.shape == (rows,) and o.shape == (rows, RES_X), (z.shape, o.shape)

        # Guard the row rule at generation time rather than trusting it downstream: every row
        # this golden keeps must sit inside a layer, and every row it drops must be ambient.
        keep = layer_rows(n_layers(stack))
        amb = np.setdiff1d(np.arange(rows), keep)
        assert np.allclose(o[amb], 1.0), 'a dropped row is not ambient -- the row rule moved'
        assert not np.allclose(o[keep], 1.0), 'a kept row is ambient -- the row rule moved'

        # Stored RAW, in RETICOLO's own z direction and its own component order. The frame
        # transform is the claim under test and belongs in the test, where it is documented.
        out[key(stack, theta, pol)] = e[keep]
        out[f'{stack}__z'] = z[keep]
        out[f'{stack}__o'] = o[keep]

    out['x'] = np.linspace(0, PERIOD, RES_X)     # what the .m hardcodes
    assert np.allclose(out['x'], x_samples())
    np.savez_compressed(str(OUT), **out)
    print(f'wrote {len(out)} arrays to {OUT.name}; {len(CASES)} cases, max_order={MAX_ORDER}, '
          f'{2 * MAX_ORDER + 1} orders')


if __name__ == '__main__':
    try:
        import oct2py  # noqa: F401
    except ImportError:
        sys.exit('needs oct2py and Octave; the committed .npz is what the tests read.')
    main()
