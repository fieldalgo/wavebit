"""Freeze RETICOLO's diffraction orders into ``golden_reticolo.npz``.

Run once, on a machine that has Octave and RETICOLO; ``test_reticolo_golden.py`` then runs
anywhere. Rerun it only to add cases or after deliberately changing what is compared —
never to make a failing test pass, since the whole value of this oracle is that its numbers
did not come from this codebase.

    conda run -n wavebit python tests/references/gen_goldens_reticolo.py

The cases are imported from the test module rather than restated here. The other generators
in this directory keep a copy with a "must stay identical" comment, and a copy that must
stay identical is a copy that will not.

Setup notes, all of them things that cost an afternoon once:

* Points at a **version-pinned V10 tree**, given by ``RETICOLO_PATH``. RETICOLO is kept
  outside this repository beside its licence and provenance notes rather than vendored in.
  This used to say "not redistributable", which was an assumption rather than a finding: the
  V10 Zenodo record is CC BY 4.0 and *does* permit redistribution with attribution, V10's
  documentation calls RETICOLO a "freeware", and the V8, V9 and V9a records declare no licence
  at all. The tree is kept out by choice, and pinning by path is what we wanted anyway. The full check is in ``README.md``, and the citation
  RETICOLO's terms *do* require is there too.

  It used to point at a ``V9_pristine`` tree under an unrelated project directory, chosen
  because the modified ``V9`` beside it failed inside ``retb`` under Octave 9.4 with
  nonconformant operands. Both generators in this directory now read the same tree. The
  switch was checked rather than assumed: regenerating this golden against V10 reproduces
  all 48 arrays **bit for bit**, so nothing in the checked-in data moved.
* ``truncation_order`` reaches RETICOLO as a scalar for the 1D branches; handing it the solver's
  ``[half_width_u, half_width_v]`` pair returns an empty efficiency list rather than an error.
* **The polarisation is read from whichever spelling the case uses.** The pure cases declare
  ``tetm='te'`` / ``'tm'`` and carry no ``psi``; the conical ones carry ``psi`` and no
  ``tetm``. Selecting on ``psi`` alone -- which is what this did until 2026-08-15 -- returns
  TM for *every* pure case, the five ``te_*`` included, so this generator could not reproduce
  its own committed golden: rerunning it moved 20 arrays by up to 4.6e-1 and failed 17 tests
  against numbers that had just been called the reference. ``R + T`` stayed exactly 1
  throughout, so the guard below could not see it either -- solving the wrong polarisation of
  a lossless stack is still a perfectly lossless answer. Verified by regenerating: with the
  polarisation read correctly, all 72 arrays come back byte-identical to the committed file.
* **The cases are permittivities and RETICOLO takes an index**, since 2026-08-20. The one
  conversion is in the bridge (``interface/Reticolo.py::_index``), so this file hands the same
  numbers to Octave that ``test_reticolo_golden.py`` hands to the solver -- which is the whole
  point of the two sharing a case list. Verified the only way it can be: with the flip applied
  on both sides, regenerating reproduces all 72 committed arrays **bit for bit**, so nothing in
  the frozen data moved and the physical scene is the one it always was.
* RETICOLO reports only propagating orders, in increasing order index. The order indices are
  recomputed here from the grating equation and checked against that count, so a silent
  disagreement about which orders propagate fails at generation time instead of becoming a
  golden that quietly compares the wrong entries.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))
from tools.repository_paths import repository_path
import os
import sys

import numpy as np

HERE = repository_path('tests/references')
sys.path.insert(0, str(HERE.parents[1]))   # repo root, for `import wavebit`
sys.path.insert(0, str(HERE))              # the bridge and the test module beside it

from wavebit import index_from_eps         # noqa: E402  (after the sys.path lines above)

# No default: where RETICOLO is installed is a property of the machine, not of this project,
# and a default that happens to match one checkout silently resolves to nothing everywhere else.
RETICOLO_PATH = os.environ.get('RETICOLO_PATH')
if not RETICOLO_PATH:
    raise SystemExit('set RETICOLO_PATH to a reticolo_allege_v10 directory to regenerate this '
                     'golden; the committed .npz is what the tests read.')
RETICOLO_PATH = os.path.expanduser(RETICOLO_PATH)

OUT = HERE / 'golden_reticolo.npz'


def _index(eps):
    """The refractive index of a half-space stated as a permittivity.

    The grating equation below is written in indices, and the cases carry permittivities, so
    the conversion has to happen somewhere. Doing it here rather than reaching for the bridge's
    own ``_index`` keeps this file readable without Octave loaded, and both are the same
    principal root -- see ``interface/Reticolo.py`` for why that branch is the right one.

    ``.real`` is safe for exactly these cases and is not a general rule: every half-space in
    ``CASES`` is a real, non-absorbing medium, so its index is real and the imaginary part is
    an exact zero. An absorbing ambient would need the complex index, and the count below would
    need a different criterion than ``|k_par| < n`` anyway.
    """
    return float(np.real(index_from_eps(eps)))


def _propagating_indices(kw, n_side):
    """Indices into the solver's flat order axis whose order propagates in ``n_side``.

    1D grating equation: order ``m`` propagates when ``|n_top*sin(theta) + m*wl/period| < n``.
    The solver's axis runs ``m = -truncation_order .. +truncation_order``, so the index is ``m + truncation_order``.
    """
    truncation_order = kw['truncation_order']
    truncation_order = truncation_order if np.isscalar(truncation_order) else truncation_order[0]
    kpar = _index(kw['top']) * np.sin(kw['theta'])
    if kw.get('phi'):                       # conical: the azimuth tilts k_parallel off x
        kpar = kpar * np.cos(kw['phi'])
    step = kw['wavelength'] / kw['period'][0]
    return np.array([m + truncation_order for m in range(-truncation_order, truncation_order + 1)
                     if abs(kpar + m * step) < n_side - 1e-12], dtype=np.int64)


def main():
    from oct2py import octave
    octave.addpath(octave.genpath(RETICOLO_PATH))

    from interface.Reticolo import Reticolo
    from tests.references.test_reticolo_golden import CASES

    reti = Reticolo()
    out = {}
    for name, kw in CASES.items():
        kw = dict(kw)
        conical = kw.get('phi') is not None
        # RETICOLO's conical branch is driven by TE/TM incidence, which is what the solver's
        # psi = pi/2 and psi = 0 select; anything between has no counterpart to compare to.
        # The pure cases declare `tetm` instead and carry no `psi` at all, so reading `psi`
        # alone silently returned TM for every one of them -- including the five `te_*`. That
        # made this generator unable to reproduce its own committed golden: regenerating moved
        # 20 arrays by up to 4.6e-1 while `R + T` stayed exactly 1, so energy conservation
        # could not see it, and 17 tests failed against freshly generated "reference" numbers.
        # The two spellings are read separately here; `tetm` is authoritative when present.
        if 'pol' in kw:
            pol = kw['pol']
        elif 'tetm' in kw:
            pol = 0 if kw['tetm'] == 'te' else 1
        else:
            pol = 0 if kw.get('psi', 0.0) > 0.1 else 1

        # Permittivity straight through: the bridge is what converts to RETICOLO's index, so
        # the same numbers reach Octave and the solver and neither can be migrated without the
        # other. See `interface/Reticolo.py::_index`.
        reti_kw = dict(period=kw['period'], truncation_order=kw['truncation_order'], structure=kw['structure'],
                       thickness=kw['thickness'], theta=kw['theta'],
                       phi=kw.get('phi'), wavelength=kw['wavelength'],
                       top=kw['top'], bot=kw['bot'])
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

        idx_ri = _propagating_indices(kw, _index(kw['top']))
        idx_ti = _propagating_indices(kw, _index(kw['bot']))
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
