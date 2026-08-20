"""Freeze the legacy block modeler's output as reference data.

Run while wavebit/on_numpy/modeler/modeling.py still exists.

The two ``ellipse`` entries it writes are kept for the record but are no longer asserted
against: the legacy ellipse decomposition did not converge under refinement, so
test_modeler_core.py holds only the rectangles to parity and test_modeler_ellipse.py checks
the ellipse against geometry instead.

**``INSTRUCTIONS`` is stated in permittivity, and the golden it produced is not.** The eleven
``*__cell`` arrays in ``golden_legacy_modeler.npz`` literally hold refractive index, because
that is what a block modeler painted in 2026-08 and this file cannot be rerun to say otherwise.
They are squared at the comparison instead (``tests/test_modeler_core.py``), which is the trade
this directory's README prescribes for a golden that has outlived its generator.

The list is migrated all the same, for the same reason as ``gen_goldens.py``: it has a twin in
``test_modeler_core.py`` marked "must stay identical", and that twin is live code driving
today's modeler. The pair is worth more than the fiction that these lines are a transcript of
the 2026-08 run — the refusal below already says they are not.
"""
# ---------------------------------------------------------------------------------------
# REFUSES TO RUN, and the refusal is the point.
#
# This was written to capture the legacy block modeler, which the rewrite deleted. Nothing here fails on its own
# any more: it would run to completion and write *the current implementation's* output into a
# file named for the legacy one -- turning the repo's only legacy-parity evidence into a change
# detector, silently, with every test still green.
#
# That is not hypothetical twice over. It is how a wrong sign convention survived in the field
# golden until 2026-08-14 (tests/data/gen_goldens_field.py records it); and while this guard was
# being added the author ran this script "to check that it refuses" -- before the guard was
# actually in the file -- and overwrote the golden on the spot. It was restored from git and
# verified byte-identical. The window between "obviously safe to run" and "silently destroyed
# the reference" is one command wide.
#
# Unconditional, because nothing can make it correct again. If the output must change on
# purpose, transform the golden AT THE COMPARISON and say why in the commit -- the way
# tests/test_field_core.py does for the 2026-08-14 sign fix.
import sys as _sys
_sys.exit('refusing to run: the legacy block modeler no longer exists, so this would re-freeze the current '
          'implementation as its own reference. See the note at the top of this file.')
# ---------------------------------------------------------------------------------------

import pathlib

import numpy as np

import wavebit

PERIOD = [300.0, 300.0]
# Permittivities, at the indices these three materials have at 900 nm: n = 3.638751670074983,
# 1.4518 and 2.0056 respectively. Written as the squared doubles rather than as
# `eps_from_index(n)` because the copy in `test_modeler_core.py` has to match this one
# character for character, and a helper call is one more thing for the two to disagree about.
SI, SIO2, SI3N4 = 13.24051371647348, 2.10772324, 4.02243136

INSTRUCTIONS = {
    'rect_1d': [[SI3N4, [['rectangle', 50, 150, 31, 300, SI, 0, 0, 0]]], [SI, []]],
    'rect_2d': [[SI3N4, [['rectangle', 80, 120, 60, 90, SI, 0, 0, 0],
                         ['rectangle', 200, 180, 50, 70, SIO2, 0, 0, 0]]], [SI, []]],
    'rect_rot': [[SI3N4, [['rectangle', 150, 150, 80, 60, SI, 0.4, 4, 4]]], [SIO2, []]],
    'ellipse': [[SI3N4, [['ellipse', 150, 150, 90, 60, SI, 0.0, 4, 4]]], [SIO2, []]],
    'ellipse_rot': [[SI3N4, [['ellipse', 140, 160, 100, 50, SI, 0.5, 4, 4]]], [SIO2, []]],
}

out = {}
for name, instr in INSTRUCTIONS.items():
    solver = wavebit.rcwa(backend=0, period=PERIOD, thickness=[205.0, 100.0], fto=[3, 3],
                          top=1.0, bot=1.0, wavelength=900.0, psi=np.pi / 4, structure=instr)
    info = solver.modeling_block_instruction(solver.structure)
    out[f'{name}__n_layers'] = np.array(len(info))
    for i, (cell, x, y) in enumerate(info):
        out[f'{name}__L{i}__cell'] = np.asarray(cell)
        out[f'{name}__L{i}__x'] = np.asarray(x)
        out[f'{name}__L{i}__y'] = np.asarray(y)
    res = solver.conv_solve()
    out[f'{name}__de_ri'] = np.asarray(res.res.R)
    out[f'{name}__de_ti'] = np.asarray(res.res.T)

# rectangle_no_approximation / draw, exercised by test_fixes.
# `material=4.0` is the index-2 rectangle this has always drawn; the keyword was `base=` and
# took an index. The `draw__L0__cell` array in the golden therefore holds 1 and 2, not 1 and 4
# — the eleventh of the eleven index-valued cells named in the docstring above.
solver = wavebit.rcwa(backend=0, period=[1000.0, 1000.0], thickness=[100.0], fto=[2, 2],
                      wavelength=900.0, structure=[[1.0, []]])
elem = solver.modeler.rectangle_no_approximation(cx=500, cy=500, lx=200, ly=200, material=4.0)
drawn = solver.modeler.draw([[1.0, elem]])
for i, (cell, x, y) in enumerate(drawn):
    out[f'draw__L{i}__cell'] = np.asarray(cell)
    out[f'draw__L{i}__x'] = np.asarray(x)
    out[f'draw__L{i}__y'] = np.asarray(y)

np.savez_compressed(str(pathlib.Path(__file__).parent / 'golden_legacy_modeler.npz'), **out)
print(f'wrote {len(out)} arrays for {len(INSTRUCTIONS)} instruction sets')
