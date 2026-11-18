"""Freeze RETICOLO's diffraction efficiencies of tapered line gratings into ``golden_reticolo_taper.npz``.

Run once, on a machine with Octave, oct2py and a RETICOLO V10 tree; ``test_reticolo_taper_golden.py``
then runs anywhere.

    RETICOLO_PATH=... python tests/references/gen_goldens_reticolo_taper.py

Rerun it only to add cases -- never to make a failing test pass.

**What it checks.** ``rcwa.ProfiledLayer`` slices a :class:`~wavebit.rcwa.Trapezoid` into
rectangles. This golden holds RETICOLO's answer for the same staircase, built here from the
geometry in ``_reticolo_taper_cases.py`` (which does not import the solver's slicer), at two
uniform slice counts, and for three finer ones from which the test extrapolates the limit of
the staircase. RETICOLO gets each slice as an exact 1D texture, its edges as numbers, so it and
the solver build the same convolution matrices: on the same staircase the two should agree to
round-off. The finer ladder checks the other half -- that the solver's automatic slice count
lands within its stated tolerance of the converged profile.

The generator process imports NumPy, oct2py and the cases module only, never Wavebit, and drives
``interface/reticolo_res2.m`` itself. RETICOLO's ``+z`` points out of the stack, so its profile is the solver's slices reversed, top
first. Its texture runs over ``0 .. period``; a line centred on the cell is centred at
``period/2`` there, which leaves every diffraction efficiency unchanged.

**Citation.** J. P. Hugonin and P. Lalanne, RETICOLO software for grating analysis,
arXiv:2101.00901 (V10, Zenodo 10.5281/zenodo.14631951, CC BY 4.0). See ``README.md``.
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
sys.path.insert(0, str(HERE))             # the cases module beside it; the product is not imported

RETICOLO_PATH = os.environ.get('RETICOLO_PATH')
if not RETICOLO_PATH:
    raise SystemExit('set RETICOLO_PATH to a reticolo_allege_v10 directory to regenerate this '
                     'golden; the committed .npz is what the tests read.')
RETICOLO_PATH = os.path.expanduser(RETICOLO_PATH)

OUT = HERE / 'golden_reticolo_taper.npz'


def main():
    from oct2py import octave
    octave.addpath(octave.genpath(RETICOLO_PATH))
    octave.addpath(str(HERE / 'interface'))

    from _reticolo_taper_cases import (EPS_LINE, EPS_SUBSTRATE, LADDER, LIGHT, ORDERS, PERIOD,
                                       PROFILES, SAME_STAIRCASE, WAVELENGTH, key, staircase)
    # RETICOLO takes refractive indices: the principal root is the passive branch, Im(n) >= 0.
    n_air, n_line, n_sub = 1.0, complex(np.sqrt(complex(EPS_LINE))), complex(np.sqrt(complex(EPS_SUBSTRATE)))
    out = {}
    for name in PROFILES:
        for count in SAME_STAIRCASE + LADDER:
            slices = staircase(name, count)
            textures = [n_air]
            for _, left, right in reversed(slices):
                textures.append([np.array([PERIOD/2 + left, PERIOD/2 + right, PERIOD]),
                                 np.array([n_air, n_line, n_air], dtype=complex)])
            textures.append(n_sub)
            thickness = [t for t, _, _ in reversed(slices)]
            profile = np.array([[0, *thickness, 0], range(1, len(thickness) + 3)])
            for theta, pol in LIGHT:
                refl, tran, _, _ = octave.reticolo_res2(0 if pol == 'te' else 1, theta, 0, PERIOD, n_air,
                                                        ORDERS, textures, profile, WAVELENGTH, 0, nout=4)
                k = key(name, count, theta, pol)
                out[k + '__R'] = np.ravel(np.asarray(refl.efficiency, dtype=float))
                out[k + '__R_orders'] = np.ravel(np.asarray(refl.order, dtype=int))
                out[k + '__T'] = np.ravel(np.asarray(tran.efficiency, dtype=float))
                out[k + '__T_orders'] = np.ravel(np.asarray(tran.order, dtype=int))
        print(name, 'done', flush=True)
    np.savez_compressed(str(OUT), **out)
    print(f'wrote {len(out)} arrays to {OUT.name}')


if __name__ == '__main__':
    try:
        import oct2py  # noqa: F401
    except ImportError:
        sys.exit('needs oct2py and Octave; the committed .npz is what the tests read.')
    main()
