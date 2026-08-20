"""Freeze the 2024-09-09 RETICOLO (kx, ky) sweep into ``golden_reticolo_conical_sweep.npz``.

Unlike the other RETICOLO generators, this one does not run Octave. The sweep was run once,
on 2024-09-09, by a study that lived outside this repository, and its raw output is a set of
7-19 MB ``.mat`` files. All this script does is pull out the one slice the test needs -- the
zeroth-order reflected amplitude for TE incidence -- and commit it at a few kilobytes.

The study directory is not in this repository and is not needed to run the tests -- the golden
is committed. It holds::

    run_reti.m               the sweep driver
    Reticolo_RCWA_1D.m       the per-point wrapper
    reti_res_80.mat          max_order=80   (reti_res2.mat is the same run, to 3e-14)
    reti_res_200.mat         max_order=200

Pass that directory as the first argument and run this file with Octave-free Python; the golden
it writes is what CI reads. There is no default: the path is machine-specific, and a stale
default that silently resolves to nothing is worse than being asked for it.

**The scene keys below hold a refractive INDEX, and they stay that way.** ``ucell`` / ``n_top``
/ ``n_bot`` are the spellings and the quantity of 2024-09-09; wavebit took permittivity from
2026-08-20. Renaming them here would be a lie about a file that cannot be regenerated to match
-- the ``.mat`` inputs are not in this repository -- and squaring them would leave the committed
golden and this script describing different substrates. So the conversion lives at the read
instead, in ``test_reticolo_conical_sweep._scene``, which says the same thing from the other
end. If the study directory ever comes back and this is rerun, move the squaring here and
delete it there; do not have both.
"""
import pathlib
import sys

import numpy as np
import scipy.io

if len(sys.argv) < 2:
    raise SystemExit(f'usage: {pathlib.Path(__file__).name} <study-directory>\n'
                     '  the directory holding reti_res_80.mat and reti_res_200.mat; see the '
                     'module docstring')
SOURCE = pathlib.Path(sys.argv[1]).expanduser()
OUT = pathlib.Path(__file__).parent / 'golden_reticolo_conical_sweep.npz'

# The grid and structure `run_reti.m` swept. Lengths are metres here; the .m file works in nm.
KX = np.linspace(-2, 2, 21)
KY = np.linspace(-2, 2, 21)
UCELL = np.array([[[0, 1, 0, 1, 1, 1]]]) * 4 + 1
PERIOD = [2000e-9]
WAVELENGTH = 900e-9
THICKNESS = [100e-9]
N_TOP, N_BOT = 1.0, 5.0

payload = dict(kx=KX, ky=KY, ucell=UCELL, period=PERIOD, wavelength=WAVELENGTH,
               thickness=THICKNESS, n_top=N_TOP, n_bot=N_BOT)

for max_order, name in ((80, 'reti_res_80.mat'), (200, 'reti_res_200.mat')):
    mat = scipy.io.loadmat(SOURCE / name)
    arr = mat['dc_ri_TETE_Reticolo_arr']
    assert arr.shape == (len(KX), len(KY), 2 * max_order + 1), (name, arr.shape)
    # [:, :, max_order] is the (0,0) order: the array runs -max_order .. +max_order.
    payload[f'amp_fto{max_order}'] = arr[:, :, max_order].astype(np.complex128)

np.savez(OUT, **payload)
print('wrote', OUT, sorted(payload))
