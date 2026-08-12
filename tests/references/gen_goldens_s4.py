"""Freeze S4's anisotropic slab efficiencies into ``golden_s4.npz``.

Run once, in an environment with the S4 extension built; ``test_s4_golden.py`` then runs
anywhere. Rerun it only to add cases, never to make a failing test pass.

    conda run -n fieldwave python tests/references/gen_goldens_s4.py

**Why a fourth oracle, and what it is not.** fmmax generalises the S4 paper (Liu & Fan 2012,
eq. 28) — so these two are the *same formulation
lineage*, and this file does not add an independent derivation the way RETICOLO does. What it
adds is a second **implementation** of that lineage, in C++ against LAPACK rather than JAX, and
one predating the other by a decade. Formulation errors survive it; implementation errors do
not. fmmax's own comments record fixing several equation errors in the S4 paper, which is
exactly the kind of place two implementations of one formulation can part company.

**Two conventions that cost an afternoon each, and are the real reason this file exists.**

* S4 indexes the permittivity tensor with **x and y exchanged** relative to this repo. Passing
  a tensor unconjugated puts S4 4e-1 away from Berreman on a symmetric case and looks for all
  the world like a physics disagreement. ``SWAP @ eps @ SWAP`` brings it to 5.6e-16. This was
  first misdiagnosed as an angle-convention problem, which it is not: ``IncidenceAngles`` is
  ``(polar, azimuth)`` exactly as written, checked against an analytic slab at 0, 20 and 45
  degrees.
* ANALYSIS.md §2.1 records that S4 is **silently wrong when polarisation decomposition and
  anisotropy are combined**. So this reads total power flux per incident polarisation and never
  order-resolved s/p amplitudes. Do not "improve" it into the latter.

**Real tensors only.** This build rejects a complex ``Epsilon``, so gyrotropy proper is out of
reach here — the real antisymmetric cases below are active media rather than physical ones,
which is fine: they discriminate the same algebra. Complex off-diagonal belongs to the fmmax
and RETICOLO goldens.
"""
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

OUT = HERE / 'golden_s4.npz'

WL = 580.0
PERIOD = 260.0
THICK = 240.0

SWAP = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def tensor(exx, eyy, ezz, exy=0.0, eyx=0.0):
    return np.array([[exx, exy, 0.0], [eyx, eyy, 0.0], [0.0, 0.0, ezz]], dtype=float)


CASES = {
    'isotropic_normal': (tensor(2.9, 2.9, 2.9), 0.0),
    'isotropic_41deg': (tensor(2.9, 2.9, 2.9), 41.0),
    'biaxial_23deg': (tensor(2.4, 3.3, 2.8), 23.0),
    'symmetric_offdiag_normal': (tensor(2.6, 3.4, 2.5, 0.45, 0.45), 0.0),
    'symmetric_offdiag_53deg': (tensor(2.6, 3.4, 2.5, 0.45, 0.45), 53.0),
    'antisymmetric_normal': (tensor(2.6, 3.4, 2.5, 0.45, -0.45), 0.0),
    'antisymmetric_53deg': (tensor(2.6, 3.4, 2.5, 0.45, -0.45), 53.0),
    'antisymmetric_strong_17deg': (tensor(3.0, 2.2, 2.5, 0.95, -0.95), 17.0),
}


def s4_rt(eps, theta_deg, pol):
    import S4
    e = SWAP @ np.asarray(eps, dtype=float) @ SWAP
    S = S4.New(Lattice=((PERIOD, 0.0), (0.0, PERIOD)), NumBasis=1)
    S.SetMaterial(Name='vac', Epsilon=1.0)
    S.SetMaterial(Name='mat', Epsilon=tuple(tuple(float(v) for v in row) for row in e))
    S.AddLayer(Name='top', Thickness=0.0, Material='vac')
    S.AddLayer(Name='slab', Thickness=THICK, Material='mat')
    S.AddLayer(Name='bot', Thickness=0.0, Material='vac')
    S.SetExcitationPlanewave(IncidenceAngles=(theta_deg, 0.0),
                             sAmplitude=1.0 if pol == 's' else 0.0,
                             pAmplitude=0.0 if pol == 's' else 1.0, Order=0)
    S.SetFrequency(1.0 / WL)
    f_top, b_top = S.GetPowerFlux(Layer='top', zOffset=0.0)
    f_bot, _ = S.GetPowerFlux(Layer='bot', zOffset=0.0)
    return abs(b_top.real) / f_top.real, f_bot.real / f_top.real


def main():
    out = {}
    for name, (eps, theta_deg) in CASES.items():
        e = np.asarray(eps, dtype=float)
        vals = []
        for pol in ('s', 'p'):
            vals += list(s4_rt(e, theta_deg, pol))
        vals = np.asarray(vals, dtype=float)      # [R_s, T_s, R_p, T_p]

        # A real tensor is Hermitian only when it is symmetric; the antisymmetric cases are
        # active media and R + T above 1 is their correct answer, not a symptom.
        if np.abs(e - e.T).max() <= 1e-12:
            for pol, (r, t) in (('s', vals[:2]), ('p', vals[2:])):
                if abs(r + t - 1.0) > 1e-9:
                    raise SystemExit(f'{name} {pol}: R+T = {r + t:.12f} for a lossless '
                                     'symmetric tensor — the bridge is wired wrong, do not '
                                     'freeze this')
            tag = 'lossless'
        else:
            tag = 'active (non-Hermitian)'
        out[name] = vals
        print(f'  {name:<30} {np.array2string(vals, precision=9)}  [{tag}]')

    np.savez(OUT, **out)
    print(f'\nwrote {OUT} ({len(out)} cases)')


if __name__ == '__main__':
    main()
