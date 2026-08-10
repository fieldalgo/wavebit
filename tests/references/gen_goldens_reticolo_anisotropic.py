"""Freeze RETICOLO's anisotropic slab efficiencies into ``golden_reticolo_anisotropic.npz``.

Run once, on a machine with Octave and RETICOLO V10; ``test_reticolo_anisotropic_golden.py``
then runs anywhere. Rerun it only to add cases, never to make a failing test pass — the whole
value of this oracle is that its numbers did not come from this codebase.

    conda run -n fieldwave python tests/references/gen_goldens_reticolo_anisotropic.py

Cases are imported from the test module rather than restated, for the reason
``gen_goldens_reticolo.py`` gives: a copy that must stay identical is a copy that will not.

Setup notes:

* Reads the same **V10** snapshot as ``gen_goldens_reticolo.py``, through the same
  ``RETICOLO_PATH``. The anisotropic texture form (a cell holding a 3x3 tensor) is a
  V9-and-later feature; V10_2025 is what both were verified against.
* ``oct2py`` does not carry complex matrices across reliably, so ``interface/reti_aniso.m``
  takes the real and imaginary parts separately.
* What RETICOLO will and will not take, from its V10 manual — read this before adding cases:
  a **uniform** layer accepts an arbitrary 3x3 tensor (page 1), a **patterned** one only a
  diagonal tensor via ``parm.res1.change_index`` (pages 1 and 7), and the **substrate and
  superstrate cannot be anisotropic at all** (page 1). Every case below is a uniform layer
  between vacuum, which is the intersection of those. A magneto-optic substrate, or an
  off-diagonal tensor inside a grating, has to go to a different oracle — see
  ``docs/off-diagonal-slot-convention.md`` §8.
* **"Accepts an arbitrary 3x3" is narrower in practice than the manual reads**, and the
  boundary was measured here on 2026-08-11 rather than assumed. Feeding a *real symmetric*
  tensor — Hermitian, hence provably lossless, hence ``R + T = 1`` by theorem — through this
  same bridge:

  ===============================  ==========  =====================================
  tensor                           ``delta0``  RETICOLO's own ``R + T``
  ===============================  ==========  =====================================
  diagonal                         0 and 40    1 to 1e-15 (and matches 4ff to 1e-15)
  one off-diagonal pair            0           1 to 1e-15 (and matches 4ff to 1e-15)
  one off-diagonal pair            1 / 40      off by 5.8e-6 / 6.3e-3
  two or three off-diagonal pairs  0           off by 2.5e-3 to 1.2e-2
  ===============================  ==========  =====================================

  The deviation is equal and opposite between TE and TM, and does not move when ``nn`` is
  raised from 0 to 8, so it is converged rather than truncated. It cannot be a convention
  difference on this side: a real symmetric matrix stays real symmetric under transposition and
  under any permutation of axes, so no reading of the tensor this bridge could get wrong makes
  the medium lossy. The shipped ``exemple_V9_0D_anisotrope.m`` uses a single ``xz`` pair at
  ``delta0 = 0`` — inside the exact region that holds.

  Consequence: **RETICOLO is not an oracle for a general anisotropic tensor.** It is one for
  diagonal tensors, and for a single off-diagonal pair at classical mounting, which is where
  every case in this file sits. The tilted-magnetisation and fully dense cases were written,
  measured, and then removed rather than frozen. This is not a claim about RETICOLO's internals,
  which have not been read — only about the numbers it returns through this bridge.
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
    'RETICOLO_PATH',
    os.path.expanduser('$RETICOLO_PATH/'
                       'V10_2025/reticolo_allege_v10'))

OUT = HERE / 'golden_reticolo_anisotropic.npz'


def main():
    from oct2py import octave

    from test_reticolo_anisotropic_golden import ALL_CASES, WL, PERIOD, THICK, is_hermitian

    if not pathlib.Path(RETICOLO_PATH).is_dir():
        raise SystemExit(f'RETICOLO not found at {RETICOLO_PATH}; set RETICOLO_PATH')

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
