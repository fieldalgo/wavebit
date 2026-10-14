"""Freeze the 2024-09-09 RETICOLO (kx, ky) sweep into ``golden_reticolo_conical_sweep.npz``.

Unlike the other RETICOLO generators, this one does not run Octave. The sweep was run once,
on 2024-09-09, by a study that lived outside this repository, and its raw output is a set of
7-19 MB ``.mat`` files. All this script does is pull out the one slice the test needs -- the
zeroth-order reflected amplitude for TE incidence -- and commit it at a few kilobytes.

The study directory is not in this repository and is not needed to run the tests -- the golden
is committed. It holds::

    run_reti.m               the sweep driver
    Reticolo_RCWA_1D.m       the per-point wrapper
    reti_res_80.mat          truncation_order=80   (reti_res2.mat is a second run: equal on the
                                                    (0,0) reflection to 3e-14, ~1e-6 elsewhere)
    reti_res_200.mat         truncation_order=200

Pass that directory as the first argument and run this file with Octave-free Python; the golden
it writes is what CI reads. There is no default: the path is machine-specific, and a stale
default that silently resolves to nothing is worse than being asked for it.

**The scene is written in permittivity, under the solver's names** ``structure`` / ``top`` /
``bot``. RETICOLO takes a refractive index, so the constants below state what ``run_reti.m``
swept in its own quantity and are squared once, at the write. Until 2026-10-15 the golden held
those indices as ``ucell`` / ``n_top`` / ``n_bot`` and ``test_reticolo_conical_sweep._scene``
squared them at the read, because the study directory was not at hand to rerun this; rerun
from it then, the golden reproduced every amplitude and grid array bit for bit, and the squaring
moved here. It lives in one place: squaring at the read as well would solve ``eps**2``.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))
from tools.repository_paths import repository_path
import pathlib
import sys

import numpy as np
import scipy.io

if len(sys.argv) < 2:
    raise SystemExit(f'usage: {pathlib.Path(__file__).name} <study-directory>\n'
                     '  the directory holding reti_res_80.mat and reti_res_200.mat; see the '
                     'module docstring')
SOURCE = pathlib.Path(sys.argv[1]).expanduser()
OUT = repository_path('tests/references/golden_reticolo_conical_sweep.npz')

# The grid and structure `run_reti.m` swept. Lengths are metres here; the .m file works in nm.
KX = np.linspace(-2, 2, 21)
KY = np.linspace(-2, 2, 21)
# Refractive indices, as RETICOLO took them: run_reti.m's ucell_1d_s, n_I and n_II.
UCELL = np.array([[[0, 1, 0, 1, 1, 1]]]) * 4 + 1
PERIOD = [2000e-9]
WAVELENGTH = 900e-9
THICKNESS = [100e-9]
N_TOP, N_BOT = 1.0, 5.0

payload = dict(kx=KX, ky=KY, structure=UCELL ** 2, period=PERIOD, wavelength=WAVELENGTH,
               thickness=THICKNESS, top=N_TOP ** 2, bot=N_BOT ** 2)

for truncation_order, name in ((80, 'reti_res_80.mat'), (200, 'reti_res_200.mat')):
    mat = scipy.io.loadmat(SOURCE / name)
    arr = mat['dc_ri_TETE_Reticolo_arr']
    assert arr.shape == (len(KX), len(KY), 2 * truncation_order + 1), (name, arr.shape)
    # [:, :, truncation_order] is the (0,0) order: the array runs -truncation_order .. +truncation_order.
    payload[f'amp_fto{truncation_order}'] = arr[:, :, truncation_order].astype(np.complex128)

np.savez(OUT, **payload)
print('wrote', OUT, sorted(payload))
