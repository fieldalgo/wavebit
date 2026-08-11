"""Freeze the 2024-09-09 RETICOLO (kx, ky) sweep into ``golden_reticolo_conical_sweep.npz``.

Unlike the other RETICOLO generators, this one does not run Octave. The sweep was run once,
on 2024-09-09, by a study that lived outside this repository, and its raw output is a set of
7-19 MB ``.mat`` files. All this script does is pull out the one slice the test needs -- the
zeroth-order reflected amplitude for TE incidence -- and commit it at a few kilobytes.

Source (not in this repository)::

    <the study directory>/20240910/
        run_reti.m               the sweep driver
        Reticolo_RCWA_1D.m       the per-point wrapper
        reti_res_80.mat          fto=80   (reti_res2.mat is the same run, to 3e-14)
        reti_res_200.mat         fto=200

Point ``SOURCE`` at that directory and run this file with Octave-free Python; the golden it
writes is what CI reads.
"""
import pathlib
import sys

import numpy as np
import scipy.io

SOURCE = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                      '<the study directory>/20240910').expanduser()
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

for fto, name in ((80, 'reti_res_80.mat'), (200, 'reti_res_200.mat')):
    mat = scipy.io.loadmat(SOURCE / name)
    arr = mat['dc_ri_TETE_Reticolo_arr']
    assert arr.shape == (len(KX), len(KY), 2 * fto + 1), (name, arr.shape)
    # [:, :, fto] is the (0,0) order: the array runs -fto .. +fto.
    payload[f'amp_fto{fto}'] = arr[:, :, fto].astype(np.complex128)

np.savez(OUT, **payload)
print('wrote', OUT, sorted(payload))
