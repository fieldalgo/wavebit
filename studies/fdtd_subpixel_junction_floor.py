"""Diagnosis of the anchor's oblique floor: the bare-substrate control K0 at 45 degrees p, with
the default z PML (24 cells, reflection 1e-8) against stricter ones. If the floor is the PML's
round-trip return, it moves with the target and not with the rung; if it is anything else, it
does not move. Read-only on the study's logs; nothing is appended to them."""
import dataclasses
import sys
import time


import fdtd_subpixel_junction as S
import wavebit.fd as fd
from wavebit.fd import _run
from wavebit.fd._runtime import runtime

S.PROV = S.provenance()
exact_rows = S.load(S.RCWA_LOG)


def k0(film, n, zeta, pml, reflection, periods=80):
    name = S.FILMS[film]
    scene, info = S.build(name, n, zeta, variant='bare', film=True, theta=S.OBLIQUE, pol='p')
    glass = fd.Material(eps=S.SCENES[name]['eps_s'])
    faces = fd.Bloch.from_source(scene.sources[0], axis1d='x')
    z = (fd.PML(cells=pml, background=glass, reflection=reflection),
         fd.PML(cells=pml, reflection=reflection))
    if pml != 24:
        # keep the physical layout; only the layer's cells and grading change
        pass
    scene = dataclasses.replace(scene, boundaries=fd.Boundaries(**faces, z=z),
                                run=fd.Run(periods=periods))
    t0 = time.perf_counter()
    p, rows = _run._planned(scene)
    res = _run._solve(scene, p, runtime('jax' if n >= 48 else 'numpy'), rows)
    amp = S.amplitudes(res, 'p')
    err = 0.0
    for i, wl in enumerate(S.WAVELENGTHS):
        row = exact_rows[S.rcwa_key(name, 'bare', 0, wl, S.OBLIQUE)]
        for m in ('r', 't'):
            err = max(err, abs(complex(*amp[m][i]) - S._rc(row, 'p', m)))
    return err, time.perf_counter() - t0


if __name__ == '__main__':
    for film in sys.argv[1:] or ["FT'", "FC'"]:
        for n in (80, 160):
            for pml, refl in ((24, 1e-8), (24, 1e-12), (24, 1e-16)):
                e, wall = k0(film, n, 0.25, pml, refl)
                print(f'{film} n {n} PML {pml} cells reflection {refl:g}: K0 {e:.3e} '
                      f'({wall:.0f} s)', flush=True)
