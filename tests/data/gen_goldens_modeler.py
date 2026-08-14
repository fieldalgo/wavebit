"""Freeze the legacy block modeler's output as reference data.

Run while wavebit/on_numpy/modeler/modeling.py still exists.

The two ``ellipse`` entries it writes are kept for the record but are no longer asserted
against: the legacy ellipse decomposition did not converge under refinement, so
test_modeler_core.py holds only the rectangles to parity and test_modeler_ellipse.py checks
the ellipse against geometry instead.
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
SI, SIO2, SI3N4 = 3.638751670074983, 1.4518, 2.0056

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
                          n_top=1.0, n_bot=1.0, wavelength=900.0, psi=np.pi / 4, ucell=instr)
    info = solver.modeling_block_instruction(solver.ucell)
    out[f'{name}__n_layers'] = np.array(len(info))
    for i, (cell, x, y) in enumerate(info):
        out[f'{name}__L{i}__cell'] = np.asarray(cell)
        out[f'{name}__L{i}__x'] = np.asarray(x)
        out[f'{name}__L{i}__y'] = np.asarray(y)
    res = solver.conv_solve()
    out[f'{name}__de_ri'] = np.asarray(res.res.R)
    out[f'{name}__de_ti'] = np.asarray(res.res.T)

# rectangle_no_approximation / draw, exercised by test_fixes
solver = wavebit.rcwa(backend=0, period=[1000.0, 1000.0], thickness=[100.0], fto=[2, 2],
                      wavelength=900.0, ucell=[[1.0, []]])
elem = solver.modeler.rectangle_no_approximation(cx=500, cy=500, lx=200, ly=200, base=2.0)
drawn = solver.modeler.draw([[1.0, elem]])
for i, (cell, x, y) in enumerate(drawn):
    out[f'draw__L{i}__cell'] = np.asarray(cell)
    out[f'draw__L{i}__x'] = np.asarray(x)
    out[f'draw__L{i}__y'] = np.asarray(y)

np.savez_compressed(str(pathlib.Path(__file__).parent / 'golden_legacy_modeler.npz'), **out)
print(f'wrote {len(out)} arrays for {len(INSTRUCTIONS)} instruction sets')
