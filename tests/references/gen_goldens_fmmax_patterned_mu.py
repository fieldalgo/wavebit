"""Freeze fmmax's **patterned-permeability** efficiencies into ``golden_fmmax_patterned_mu.npz``.

    conda run -n fwref python tests/references/gen_goldens_fmmax_patterned_mu.py

Read ``gen_goldens_fmmax.py`` for the environment and float64 caveats and
``gen_goldens_fmmax_patterned.py`` for why a patterned comparison is a *convergence* one; both
apply here unchanged.

**Run before ``mu`` had a factorization.** The magnetic work landed with ``mu`` reaching the
layer operator as convolution matrices the caller supplies, and nothing built those from a
patterned raster. So these rows could not be fitted to anything, which is the whole reason to
generate them now rather than after.

fmmax was gated first, the same way the patterned ``eps`` oracle was, and on three questions
rather than two:

* a patterned ``mu`` moves its eigenvalues by 8.8e-2 — it is used, not accepted and dropped;
* patterning it differs from its own mean by 7.3e-2 — the *pattern* is used, not the average;
* a patterned ``mu_xy`` moves them by 7.2e-2.

Plus one that only exists for permeability: exchanging ``eps`` and ``mu`` must **trade the s and
p channels**, which fmmax obeys on patterned tensors to 3e-14. That is the check that says its
``mu`` handling is not merely present but dual-correct. (Attempting the same comparison on
eigenvalue *sets* is what it looks like when a check is fragile rather than failing —
``np.sort_complex`` is not an assignment between two spectra, and it reported a 6.8e-2
disagreement that the R/T form shows is not there.)

Both fmmax formulations are frozen — ``FFT``, the plain direct rule, and ``JONES_DIRECT``, a
vector one — because the tolerance the test may claim is the spread between them and nothing
tighter.
"""
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

OUT = HERE / 'golden_fmmax_patterned_mu.npz'


def main():
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    import fmmax

    from _fmmax_patterned_mu_cases import CASES, MAX_ORDER, PERIOD, THETA, THICK, WL, tensors

    lat = fmmax.LatticeVectors(u=jnp.array([PERIOD, 0.0]), v=jnp.array([0.0, PERIOD]))
    expansion = fmmax.generate_expansion(lat, approximate_num_terms=(2 * MAX_ORDER + 1) ** 2,
                                         truncation=fmmax.Truncation.PARALLELOGRAMIC)
    kx = jnp.asarray([2 * np.pi / WL * np.sin(np.deg2rad(THETA)), 0.0])
    wl = jnp.asarray(WL)

    def g(a):
        """This project's ``[y, x]`` raster as fmmax's ``[x, y]`` one. Components untouched."""
        return jnp.asarray(np.asarray(a).T)

    def u(v):
        return jnp.asarray(v, dtype=complex).reshape(1, 1)

    def efficiencies(eps, eps_xy, mu, mu_xy, formulation):
        vac = fmmax.eigensolve_general_anisotropic_media(
            wl, kx, lat, u(1), u(0), u(0), u(1), u(1), u(1), u(0), u(0), u(1), u(1),
            expansion, formulation=fmmax.Formulation.FFT)
        lay = fmmax.eigensolve_general_anisotropic_media(
            wl, kx, lat,
            g(eps), g(eps_xy), g(-eps_xy), g(eps), g(eps),
            g(mu), g(mu_xy), g(-mu_xy), g(mu), g(mu),
            expansion, formulation=formulation)
        smat = fmmax.stack_s_matrix([vac, lay, vac],
                                    [jnp.asarray(0.0), jnp.asarray(THICK), jnp.asarray(0.0)])
        n = expansion.num_terms
        vals = []
        for idx in (0, n):                        # s incidence, then p
            fwd = jnp.zeros((2 * n, 1), dtype=complex).at[idx, 0].set(1.0)
            f_in, b_in = fmmax.amplitude_poynting_flux(fwd, smat.s21 @ fwd, vac)
            f_out, _ = fmmax.amplitude_poynting_flux(
                smat.s11 @ fwd, jnp.zeros((2 * n, 1), dtype=complex), vac)
            inc = float(jnp.sum(f_in))
            vals += [abs(float(jnp.sum(b_in))) / inc, float(jnp.sum(f_out)) / inc]
        return np.asarray(vals, dtype=float)

    out = {}
    for name, case in CASES().items():
        eps, eps_xy, mu, mu_xy = tensors(case)
        fft = efficiencies(eps, eps_xy, mu, mu_xy, fmmax.Formulation.FFT)
        jones = efficiencies(eps, eps_xy, mu, mu_xy, fmmax.Formulation.JONES_DIRECT)

        # Every tensor is Hermitian at every pixel, so both are lossless whatever the pattern
        # is. Enforced sharply on FFT, where a Hermitian input gives an exactly Hermitian
        # convolution matrix; JONES_DIRECT is a vector formulation whose truncation leaves a
        # residual that shrinks with truncation_order rather than vanishing (same as the patterned eps case).
        for pol, (r, t) in (('s', fft[:2]), ('p', fft[2:])):
            if abs(r + t - 1.0) > 1e-9:
                raise SystemExit(f'{name} FFT {pol}: R+T = {r + t:.12f} for Hermitian eps and '
                                 'mu — the bridge is wired wrong, do not freeze this')
        drift = max(abs(jones[0] + jones[1] - 1.0), abs(jones[2] + jones[3] - 1.0))
        if drift > 1e-4:
            raise SystemExit(f'{name} JONES_DIRECT: R+T off by {drift:.2e}, past the truncation '
                             'residual this formulation is expected to carry')

        out[f'{name}__fft'] = fft
        out[f'{name}__jones'] = jones
        print(f'  {name:<42} spread {np.abs(fft - jones).max():.2e}   '
              f'JONES drift {drift:.2e}')
        print(f'    FFT   {np.array2string(fft, precision=9)}')
        print(f'    JONES {np.array2string(jones, precision=9)}')

    np.savez(OUT, **out)
    print(f'\nwrote {OUT} ({len(out)} rows)')


if __name__ == '__main__':
    main()
