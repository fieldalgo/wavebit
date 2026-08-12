"""Freeze RETICOLO's diffraction orders into ``golden_reticolo.npz``.

Run once, on a machine that has Octave and RETICOLO; ``test_reticolo_golden.py`` then runs
anywhere. Rerun it only to add cases or after deliberately changing what is compared —
never to make a failing test pass, since the whole value of this oracle is that its numbers
did not come from this codebase.

    conda run -n fieldwave python tests/references/gen_goldens_reticolo.py

The cases are imported from the test module rather than restated here. The other generators
in this directory keep a copy with a "must stay identical" comment, and a copy that must
stay identical is a copy that will not.

Setup notes, all of them things that cost an afternoon once:

* Points at a **version-pinned V10 tree**, given by ``RETICOLO_PATH``. RETICOLO is licensed
  and not redistributable, so it is kept outside this repository beside its licence and
  provenance notes rather than vendored in.

  It used to point at a ``V9_pristine`` tree under an unrelated project directory, chosen
  because the modified ``V9`` beside it failed inside ``retb`` under Octave 9.4 with
  nonconformant operands. Both generators in this directory now read the same tree. The
  switch was checked rather than assumed: regenerating this golden against V10 reproduces
  all 48 arrays **bit for bit**, so nothing in the checked-in data moved.
* ``fto`` reaches RETICOLO as a scalar for the 1D branches; handing it the solver's
  ``[fto_x, fto_y]`` pair returns an empty efficiency list rather than an error.
* RETICOLO reports only propagating orders, in increasing order index. The order indices are
  recomputed here from the grating equation and checked against that count, so a silent
  disagreement about which orders propagate fails at generation time instead of becoming a
  golden that quietly compares the wrong entries.
"""
import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))   # repo root, for `import fieldwave`
sys.path.insert(0, str(HERE))              # the bridge and the test module beside it

# No default: where RETICOLO is installed is a property of the machine, not of this project,
# and a default that happens to match one checkout silently resolves to nothing everywhere else.
RETICOLO_PATH = os.environ.get('RETICOLO_PATH')
if not RETICOLO_PATH:
    raise SystemExit('set RETICOLO_PATH to a reticolo_allege_v10 directory to regenerate this '
                     'golden; the committed .npz is what the tests read.')
RETICOLO_PATH = os.path.expanduser(RETICOLO_PATH)

OUT = HERE / 'golden_reticolo.npz'


def _propagating_indices(kw, n_side):
    """Indices into the solver's flat order axis whose order propagates in ``n_side``.

    1D grating equation: order ``m`` propagates when ``|n_top*sin(theta) + m*wl/period| < n``.
    The solver's axis runs ``m = -fto .. +fto``, so the index is ``m + fto``.
    """
    fto = kw['fto']
    fto = fto if np.isscalar(fto) else fto[0]
    kpar = kw['n_top'] * np.sin(kw['theta'])
    if kw.get('phi'):                       # conical: the azimuth tilts k_parallel off x
        kpar = kpar * np.cos(kw['phi'])
    step = kw['wavelength'] / kw['period'][0]
    return np.array([m + fto for m in range(-fto, fto + 1)
                     if abs(kpar + m * step) < n_side - 1e-12], dtype=np.int64)


def main():
    from oct2py import octave
    octave.addpath(octave.genpath(RETICOLO_PATH))

    from interface.Reticolo import Reticolo
    from test_reticolo_golden import CASES

    reti = Reticolo()
    out = {}
    for name, kw in CASES.items():
        kw = dict(kw)
        conical = kw.get('phi') is not None
        # RETICOLO's conical branch is driven by TE/TM incidence, which is what the solver's
        # psi = pi/2 and psi = 0 select; anything between has no counterpart to compare to.
        pol = kw.get('pol', 0 if kw.get('psi', 0.0) > 0.1 else 1)

        reti_kw = dict(period=kw['period'], fto=kw['fto'], ucell=kw['ucell'],
                       thickness=kw['thickness'], theta=kw['theta'],
                       phi=kw.get('phi'), wavelength=kw['wavelength'],
                       n_top=kw['n_top'], n_bot=kw['n_bot'])
        # One res2 call answers both illumination sides: RETICOLO's inc_bottom_* results are
        # its own bottom-incidence solve at the same k_parallel = n_top*sin(theta). The
        # bottom-side reflection propagates in n_bot and the transmission in n_top, so the
        # roles of the two index sets swap.
        R, T, R_bot, T_bot = reti.run_res2(grating_type=1 if conical else 0, pol=pol,
                                           **reti_kw)
        R = np.asarray(R, dtype=float).ravel()
        T = np.asarray(T, dtype=float).ravel()
        R_bot = np.asarray(R_bot, dtype=float).ravel()
        T_bot = np.asarray(T_bot, dtype=float).ravel()

        idx_ri = _propagating_indices(kw, kw['n_top'])
        idx_ti = _propagating_indices(kw, kw['n_bot'])
        if (len(idx_ri) != len(R) or len(idx_ti) != len(T)
                or len(idx_ti) != len(R_bot) or len(idx_ri) != len(T_bot)):
            raise SystemExit(
                f'{name}: RETICOLO reported {len(R)}/{len(T)} propagating orders top-side '
                f'and {len(R_bot)}/{len(T_bot)} bottom-side, the grating equation says '
                f'{len(idx_ri)}/{len(idx_ti)}. Freezing this would compare the wrong entries.')

        out[f'{name}__de_ri'] = R
        out[f'{name}__de_ti'] = T
        out[f'{name}__idx_ri'] = idx_ri
        out[f'{name}__idx_ti'] = idx_ti
        out[f'{name}__de_ri_bot'] = R_bot
        out[f'{name}__de_ti_bot'] = T_bot
        print(f'{name:16s} orders R={len(R)} T={len(T)}  '
              f'sum={R.sum() + T.sum():.12f}  bottom sum={R_bot.sum() + T_bot.sum():.12f}')

    np.savez_compressed(OUT, **out)
    print(f'\nwrote {OUT} ({OUT.stat().st_size} bytes, {len(CASES)} cases)')


if __name__ == '__main__':
    main()
