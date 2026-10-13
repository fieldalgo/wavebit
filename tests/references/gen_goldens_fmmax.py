"""Freeze fmmax's anisotropic slab efficiencies into ``golden_fmmax.npz``.

Run once, in an environment that has fmmax; ``test_fmmax_golden.py`` then runs anywhere.
Rerun it only to add cases, never to make a failing test pass — the whole value of this oracle
is that its numbers did not come from this codebase.

    conda run -n fwref python tests/references/gen_goldens_fmmax.py

**A separate environment, deliberately.** fmmax carries its own JAX floor and the project env
sits well ahead of it, so installing fmmax there would drag JAX back and break the backend the
suite tests. ``fwref`` exists for reference solvers and nothing else:

    conda create -n fwref python=3.11 && conda run -n fwref pip install fmmax

**JAX defaults to float32.** Without ``jax_enable_x64`` this agrees with an analytic slab only
to ~3e-8, which looks like a mediocre solver and is really a mediocre configuration. With it,
1e-17. The line is at the top of this file for that reason.

Why a third oracle, on top of Berreman and RETICOLO. fmmax (MIT) installs with pip, so
regenerating needs no Octave; and it reaches cases the others cannot — a **patterned** complex off-diagonal tensor,
which RETICOLO takes only as diagonal and S4 only as real. None of the cases below are patterned
yet, because nothing in the solver can express one, but the wiring will be here when that
changes.

Its anisotropic entry point takes the same five slots this solver does —
``permittivity_xx, xy, yx, yy, zz`` with no z row — so it is a second-order formulation as well, and
it cannot arbitrate the Voigt geometry any better than we can. RETICOLO remains the oracle there.

Sign convention: ``_fmmax_cases`` states the *physical* tensor. fmmax takes it as-is; the
solver's slots take the negation. Note that power efficiencies cannot see that sign either way
(``docs/theory/rcwa/off-diagonal-slot-convention.md`` §3), so this golden does not pin it — the amplitude
test in ``tests/rcwa/test_uniform_anisotropic.py`` does.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))
from tools.repository_paths import repository_path
import sys

import numpy as np

HERE = repository_path('tests/references')
sys.path.insert(0, str(HERE))

OUT = HERE / 'golden_fmmax.npz'


def main():
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    import fmmax

    from tests.references._fmmax_cases import CASES, WL, PERIOD, THICK

    lat = fmmax.LatticeVectors(u=jnp.array([PERIOD, 0.0]), v=jnp.array([0.0, PERIOD]))
    expansion = fmmax.generate_expansion(lat, approximate_num_terms=1,
                                         truncation=fmmax.Truncation.CIRCULAR)

    def uniform(v):
        return jnp.asarray(v, dtype=complex).reshape(1, 1)

    out = {}
    for name, (eps, theta_deg) in CASES.items():
        e = np.asarray(eps, dtype=complex)
        kx = jnp.asarray([2 * np.pi / WL * np.sin(np.deg2rad(theta_deg)), 0.0])
        wl = jnp.asarray(WL)

        vac = fmmax.eigensolve_isotropic_media(wl, kx, lat, uniform(1.0), expansion,
                                               formulation=fmmax.Formulation.FFT)
        lay = fmmax.eigensolve_anisotropic_media(
            wl, kx, lat, uniform(e[0, 0]), uniform(e[0, 1]), uniform(e[1, 0]),
            uniform(e[1, 1]), uniform(e[2, 2]), expansion,
            formulation=fmmax.Formulation.FFT)
        smat = fmmax.stack_s_matrix(
            [vac, lay, vac],
            [jnp.asarray(0.0), jnp.asarray(THICK), jnp.asarray(0.0)])

        n = expansion.num_terms
        vals = []
        for idx in (0, n):                       # s incidence, then p
            fwd = jnp.zeros((2 * n, 1), dtype=complex).at[idx, 0].set(1.0)
            f_in, b_in = fmmax.amplitude_poynting_flux(fwd, smat.s21 @ fwd, vac)
            f_out, _ = fmmax.amplitude_poynting_flux(smat.s11 @ fwd,
                                                     jnp.zeros((2 * n, 1), dtype=complex), vac)
            inc = float(jnp.sum(f_in))
            vals += [abs(float(jnp.sum(b_in))) / inc, float(jnp.sum(f_out)) / inc]

        vals = np.asarray(vals, dtype=float)     # [R_s, T_s, R_p, T_p]
        # Hermitian is the whole criterion for lossless. An earlier version also demanded a
        # real tensor, which excluded exactly the gyrotropic cases -- Hermitian, lossless, and
        # imaginary off-diagonal -- so the guard skipped the ones most worth guarding.
        lossless = np.abs(e - e.conj().T).max() <= 1e-12
        if lossless:
            for pol, (r, t) in (('s', vals[:2]), ('p', vals[2:])):
                if abs(r + t - 1.0) > 1e-9:
                    raise SystemExit(f'{name} {pol}: R+T = {r + t:.12f} for a lossless '
                                     'Hermitian tensor — the bridge is wired wrong, do not '
                                     'freeze this')
        out[name] = vals
        tag = 'lossless' if lossless else 'lossy/active'
        print(f'  {name:<32} {np.array2string(vals, precision=9)}  [{tag}]')

    np.savez(OUT, **out)
    print(f'\nwrote {OUT} ({len(out)} cases)')


if __name__ == '__main__':
    main()
