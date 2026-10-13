"""Freeze fmmax's *patterned* anisotropic efficiencies into ``golden_fmmax_patterned.npz``.

    conda run -n fwref python tests/references/gen_goldens_fmmax_patterned.py

Read ``gen_goldens_fmmax.py`` first for the environment and float64 caveats; they apply here
unchanged. What is different is what this oracle can and cannot assert.

**fmmax is the only reference solver that reaches this case.** RETICOLO takes an anisotropic
layer only as diagonal, S4 only as real, and inkstone accepts a full 3x3 and returns results
bit-identical to the z-row-zeroed tensor — i.e. it drops what it was given. fmmax was gated the
same way before any of this was written (does a *patterned* off-diagonal change the
eigenvalues, and does the pattern itself matter) and it passed both.

**It arbitrates by convergence, not by identity.** ``pn_eps_conv_matrices`` is a Popov-Neviere
rearrangement; fmmax's ``FFT`` is the plain direct rule and its ``JONES_DIRECT`` is a vector
formulation of a different construction. Three different formulations of the same physics agree
only in the limit, so this generator freezes **both** fmmax formulations and the test asserts
against their spread. A tolerance tighter than the spread between two of the oracle's own
answers would be a claim about noise.

Measured, at truncation_order=14: the spread is 4.8e-6 / 3.0e-5 / 1.7e-4 across the three contrasts, and the
solver's answer sits inside it every time — landing on ``JONES_DIRECT`` to 3e-7, 1.4e-7 and
2.0e-5 respectively, which is the agreement one expects between two vector formulations with
the direct rule as the outlier.

The transpose in ``_grid`` below is load-bearing; ``_fmmax_patterned_cases`` explains it.
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

OUT = HERE / 'golden_fmmax_patterned.npz'


def main():
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    import fmmax

    from tests.references._fmmax_patterned_cases import CASES, MAX_ORDER, PERIOD, THETA, THICK, WL, tensor_rasters

    lat = fmmax.LatticeVectors(u=jnp.array([PERIOD, 0.0]), v=jnp.array([0.0, PERIOD]))
    expansion = fmmax.generate_expansion(lat, approximate_num_terms=(2 * MAX_ORDER + 1) ** 2,
                                         truncation=fmmax.Truncation.PARALLELOGRAMIC)
    kx = jnp.asarray([2 * np.pi / WL * np.sin(np.deg2rad(THETA)), 0.0])
    wl = jnp.asarray(WL)

    def _grid(a):
        """This project's ``[y, x]`` raster as fmmax's ``[x, y]`` one. Components untouched."""
        return jnp.asarray(np.asarray(a).T)

    def efficiencies(eps, gyro, formulation):
        vac = fmmax.eigensolve_isotropic_media(wl, kx, lat, jnp.ones((1, 1), dtype=complex),
                                               expansion, formulation=fmmax.Formulation.FFT)
        lay = fmmax.eigensolve_anisotropic_media(
            wl, kx, lat, _grid(eps), _grid(gyro), _grid(-gyro), _grid(eps), _grid(eps),
            expansion, formulation=formulation)
        smat = fmmax.stack_s_matrix([vac, lay, vac],
                                    [jnp.asarray(0.0), jnp.asarray(THICK), jnp.asarray(0.0)])
        n = expansion.num_terms
        vals = []
        for idx in (0, n):                       # s incidence, then p
            fwd = jnp.zeros((2 * n, 1), dtype=complex).at[idx, 0].set(1.0)
            f_in, b_in = fmmax.amplitude_poynting_flux(fwd, smat.s21 @ fwd, vac)
            f_out, _ = fmmax.amplitude_poynting_flux(
                smat.s11 @ fwd, jnp.zeros((2 * n, 1), dtype=complex), vac)
            inc = float(jnp.sum(f_in))
            vals += [abs(float(jnp.sum(b_in))) / inc, float(jnp.sum(f_out)) / inc]
        return np.asarray(vals, dtype=float)     # [R_s, T_s, R_p, T_p]

    out = {}
    for name, (n_in, g) in CASES.items():
        eps, gyro = tensor_rasters(n_in, g)
        fft = efficiencies(eps, gyro, fmmax.Formulation.FFT)
        jones = efficiencies(eps, gyro, fmmax.Formulation.JONES_DIRECT)

        # The tensor is Hermitian everywhere, so the layer is lossless whatever the pattern is
        # and R + T = 1 is a constraint on the frozen numbers, not a hope. But the two
        # formulations owe it to different precision, and conflating them would either miss a
        # mis-wiring or refuse a sound number:
        #
        #   FFT is the plain direct rule, so a Hermitian eps gives an exactly Hermitian
        #     convolution matrix and energy closes to machine precision. That makes it a sharp
        #     wiring check -- if the bridge is wrong, it shows here.
        #   JONES_DIRECT is a vector formulation; truncating it breaks the Hermiticity slightly
        #     and the residual shrinks with truncation_order rather than vanishing. Measured at 1.4e-5 for
        #     truncation_order=4 and ~1e-6 here. That is a property of the oracle, to be recorded, not a
        #     defect to refuse.
        drift = max(abs(jones[0] + jones[1] - 1.0), abs(jones[2] + jones[3] - 1.0))
        for pol, (r, t) in (('s', fft[:2]), ('p', fft[2:])):
            if abs(r + t - 1.0) > 1e-9:
                raise SystemExit(f'{name} FFT {pol}: R+T = {r + t:.12f} for a lossless '
                                 'Hermitian tensor - the bridge is wired wrong, do not '
                                 'freeze this')
        if drift > 1e-4:
            raise SystemExit(f'{name} JONES_DIRECT: R+T off by {drift:.2e}, far past the '
                             'truncation residual this formulation is expected to carry')

        out[f'{name}__fft'] = fft
        out[f'{name}__jones'] = jones
        print(f'  {name:<26} spread {np.abs(fft - jones).max():.2e}   '
              f'JONES energy drift {drift:.2e}')
        print(f'    FFT          {np.array2string(fft, precision=9)}')
        print(f'    JONES_DIRECT {np.array2string(jones, precision=9)}')

    np.savez(OUT, **out)
    print(f'\nwrote {OUT} ({len(CASES)} cases, two formulations each)')


if __name__ == '__main__':
    main()
