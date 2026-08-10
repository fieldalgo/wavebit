"""Freeze RETICOLO's anisotropic slab efficiencies into ``golden_reticolo_anisotropic.npz``.

Run once, on a machine with Octave and RETICOLO V10; ``test_reticolo_anisotropic_golden.py``
then runs anywhere. Rerun it only to add cases, never to make a failing test pass — the whole
value of this oracle is that its numbers did not come from this codebase.

    conda run -n fieldwave python tests/references/gen_goldens_reticolo_anisotropic.py

Cases are imported from the test module rather than restated, for the reason
``gen_goldens_reticolo.py`` gives: a copy that must stay identical is a copy that will not.

Setup notes:

* Needs **V10**, not the V9 tree the isotropic generator uses. The anisotropic texture form
  (a cell holding a 3x3 tensor) is a V9-and-later feature, and V10_2025 is what was verified
  here. Override with ``RETICOLO_ANISO_PATH``.
* ``oct2py`` does not carry complex matrices across reliably, so ``interface/reti_aniso.m``
  takes the real and imaginary parts separately.
* Energy conservation is checked at generation time, **for the Hermitian cases only**. Those
  are lossless, so a golden that does not sum to one there means the bridge is wired wrong —
  better to fail here than to freeze it and have the test measure the wiring instead of the
  solver. A real antisymmetric tensor is *not* Hermitian: it is an active medium, and R+T
  above one is the correct answer for it rather than a symptom. This guard caught that
  distinction being got wrong the first time round, which is the argument for having it.
"""
import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))    # tests/, for the case list
sys.path.insert(0, str(HERE))

RETICOLO_PATH = os.environ.get(
    'RETICOLO_ANISO_PATH',
    os.path.expanduser('$RETICOLO_PATH/'
                       'V10_2025/reticolo_allege_v10'))

OUT = HERE / 'golden_reticolo_anisotropic.npz'


def main():
    from oct2py import octave

    from test_reticolo_anisotropic_golden import ALL_CASES, WL, PERIOD, THICK, is_hermitian

    if not pathlib.Path(RETICOLO_PATH).is_dir():
        raise SystemExit(f'RETICOLO not found at {RETICOLO_PATH}; set RETICOLO_ANISO_PATH')

    octave.addpath(RETICOLO_PATH)
    octave.addpath(str(HERE / 'interface'))

    out = {}
    for name, (eps, theta_deg) in ALL_CASES.items():
        e = np.asarray(eps, dtype=complex)
        vals = np.asarray(octave.reti_aniso(
            WL, PERIOD, THICK,
            np.real(e).astype(float), np.imag(e).astype(float), float(theta_deg))).ravel()
        if vals.shape != (4,):
            raise SystemExit(f'{name}: expected 4 efficiencies, got {vals.shape}')
        r_te, t_te, r_tm, t_tm = vals
        herm = is_hermitian(e)
        if herm:
            for pol, (r, t) in (('TE', (r_te, t_te)), ('TM', (r_tm, t_tm))):
                if abs(r + t - 1.0) > 1e-9:
                    raise SystemExit(f'{name} {pol}: R+T = {r + t:.12f}, expected 1 for a '
                                     'Hermitian tensor — the bridge is wired wrong, do not '
                                     'freeze this')
        out[name] = vals
        tag = 'lossless' if herm else 'active (non-Hermitian)'
        print(f'  {name:<30} {np.array2string(vals, precision=9)}  [{tag}]')

    np.savez(OUT, **out)
    print(f'\nwrote {OUT} ({len(out)} cases)')


if __name__ == '__main__':
    main()
