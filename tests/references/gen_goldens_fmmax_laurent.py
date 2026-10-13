"""Freeze fmmax's ``Formulation.FFT`` answers into ``golden_fmmax_laurent.npz``.

    conda run -n fwref python tests/references/gen_goldens_fmmax_laurent.py

Read ``gen_goldens_fmmax.py`` first for the environment and float64 caveats; they apply here
unchanged. ``_fmmax_laurent_cases.py`` says what the comparison is and why it can be one of
identity: fmmax's FFT formulation assembles the operator ``factorization='laurent'`` does, and
the only difference left -- the midpoint rule fmmax's DFT applies to each pixel -- is removed by
extrapolating over the samples per pixel.

What is frozen, per case and per level ``R`` of ``LEVELS``: the total ``[R_s, T_s, R_p, T_p]``
and the per-order reflected and transmitted power for each polarization, keyed by fmmax's own
``(m, n)`` order indices (``__orders``), so the test can place them without assuming a layout.
The extrapolation is left to the test, which is arithmetic on these numbers; the raw levels are
what make the premise checkable there.

**Twice: fmmax as shipped, and with its frequency grid in double precision.** fmmax 1.7.1 forms
the pixel-centring phase of ``fmmax.fft._fft2`` from ``jnp.fft.fftfreq(..., dtype=jnp.float32)``,
so that phase is a complex64 number in an otherwise float64 pipeline. It is a translation in
exact arithmetic and would leave every efficiency alone; rounded to single precision it is not
one, and it is the whole of the difference left once the quadrature is extrapolated away --
measured 3.8e-8 in the ``[[eps]]`` entries of the rod case at order 3 (entries of size 4), and
9e-9 to 1e-7 in the efficiencies of the three cases, while a weak-contrast twin of the rod
(eps 1.21) agrees to 8.5e-13 with the same phase in place. So the generator runs each case a
second time (``__phase64``) with fmmax's ``fft`` module reading a namespace whose ``fftfreq``
returns float64 -- nothing of fmmax's is rewritten or copied, only the dtype its own call asks
for is overridden -- and the test holds Wavebit to that run tightly and to the stock run within
the gap between the two.

The order set is handed to fmmax as ``|m|, |n| <= ORDER`` rather than generated (see
``freeze``). Refuses to freeze if a lossless case does not close its energy to 1e-9 -- the FFT
formulation's matrices are exactly Hermitian on a Hermitian permittivity, so a miss there is
the bridge, not the oracle.
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

OUT = HERE / 'golden_fmmax_laurent.npz'


def _float64_frequencies(fft_module, jnp):
    """Point ``fft_module``'s ``jnp`` at a namespace whose ``fft.fftfreq`` ignores a float32
    request. Everything else resolves to jax.numpy itself, and nothing outside that module sees
    the change. Returns the undo."""
    import types

    def fftfreq(n, d=1.0, *, dtype=None, **kwargs):
        return jnp.fft.fftfreq(n, d, dtype=jnp.float64, **kwargs)

    fft = types.SimpleNamespace(**{k: getattr(jnp.fft, k) for k in dir(jnp.fft)
                                   if not k.startswith('_')})
    fft.fftfreq = fftfreq
    proxy = types.SimpleNamespace(**{k: getattr(jnp, k) for k in dir(jnp) if not k.startswith('_')})
    proxy.fft = fft
    original = fft_module.jnp
    fft_module.jnp = proxy
    return lambda: setattr(fft_module, 'jnp', original)


def main():
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    import fmmax
    from fmmax import fft as fmmax_fft

    out = {}
    freeze(jnp, fmmax, out, flavour='stock')
    undo = _float64_frequencies(fmmax_fft, jnp)
    try:
        freeze(jnp, fmmax, out, flavour='phase64')
    finally:
        undo()
    from tests.references._fmmax_laurent_cases import CASES, LEVELS
    out['__levels'] = np.asarray(LEVELS, dtype=np.int64)
    np.savez(OUT, **out)
    print(f'\nwrote {OUT} ({len(CASES)} cases, {len(LEVELS)} sampling levels, two flavours)')


def freeze(jnp, fmmax, out, flavour):
    from tests.references._fmmax_laurent_cases import CASES, LEVELS, ORDER, THETA, THICK, WL, lossless, upsampled

    k0 = 2 * np.pi / WL
    wl = jnp.asarray(WL)
    for name, (basis, raster, top, bot) in CASES.items():
        lat = fmmax.LatticeVectors(u=jnp.asarray(basis[:, 0]), v=jnp.asarray(basis[:, 1]))
        # Stated, not generated: fmmax's parallelogramic truncation balances the two sides in
        # k-space, so on the 700 x 560 cell it keeps 17 x 13 orders where Wavebit's
        # truncation_order=(7, 7) keeps 15 x 15. The zeroth order goes first, as
        # `generate_expansion` places it, because the incident amplitude is indexed by it.
        coefficients = sorted(((m, n) for m in range(-ORDER, ORDER + 1)
                               for n in range(-ORDER, ORDER + 1)),
                              key=lambda mn: (mn != (0, 0), abs(mn[0]) + abs(mn[1]), mn))
        expansion = fmmax.Expansion(basis_coefficients=np.asarray(coefficients, dtype=np.int64))
        orders = np.asarray(expansion.basis_coefficients)
        if tuple(orders[0]) != (0, 0) or len(set(map(tuple, orders))) != (2 * ORDER + 1) ** 2:
            raise SystemExit(f'{name}: the expansion is not |m|,|n| <= {ORDER} with the '
                             'zeroth order first')
        # Incidence from the top medium, in the plane of a1 (phi = 0): (kx, ky) in rad/nm.
        kxy = jnp.asarray([k0 * np.sqrt(top).real * np.sin(np.deg2rad(THETA)), 0.0])
        n = expansion.num_terms

        def uniform(eps):
            return fmmax.eigensolve_isotropic_media(
                wl, kxy, lat, jnp.full((1, 1), eps, dtype=complex), expansion,
                formulation=fmmax.Formulation.FFT)

        ambient_in, ambient_out = uniform(top), uniform(bot)
        levels = []
        for samples in LEVELS:
            # This project's [v, u] raster as fmmax's [u, v] one -- a relabelling, not a mirror.
            grid = jnp.asarray(upsampled(raster, samples).T)
            layer = fmmax.eigensolve_isotropic_media(wl, kxy, lat, grid, expansion,
                                                     formulation=fmmax.Formulation.FFT)
            smat = fmmax.stack_s_matrix([ambient_in, layer, ambient_out],
                                        [jnp.asarray(0.0), jnp.asarray(THICK), jnp.asarray(0.0)])
            totals, per_order = [], []
            for idx in (0, n):                       # s incidence, then p
                fwd = jnp.zeros((2 * n, 1), dtype=complex).at[idx, 0].set(1.0)
                f_in, b_in = fmmax.amplitude_poynting_flux(fwd, smat.s21 @ fwd, ambient_in)
                f_out, _ = fmmax.amplitude_poynting_flux(
                    smat.s11 @ fwd, jnp.zeros((2 * n, 1), dtype=complex), ambient_out)
                inc = float(jnp.sum(f_in))
                r = -np.asarray(b_in)[:, 0].real / inc
                t = np.asarray(f_out)[:, 0].real / inc
                r, t = r[:n] + r[n:], t[:n] + t[n:]  # both outgoing polarizations of an order
                totals += [float(r.sum()), float(t.sum())]
                per_order.append(np.stack([r, t]))
            totals = np.asarray(totals)
            if lossless(name):
                for pol, (rr, tt) in (('s', totals[:2]), ('p', totals[2:])):
                    if abs(rr + tt - 1.0) > 1e-9:
                        raise SystemExit(f'{name} R={samples} {pol}: R+T = {rr + tt:.12f} on a '
                                         'lossless layer -- the bridge is wired wrong')
            out[f'{name}__{flavour}__R{samples}__totals'] = totals
            # [polarization, R/T, order], the orders in `__orders`' sequence
            out[f'{name}__{flavour}__R{samples}__per_order'] = np.stack(per_order)
            levels.append(totals)
            print(f'  {flavour:<8} {name:<16} R={samples:<4} '
                  f'{np.array2string(totals, precision=12)}')
        out[f'{name}__orders'] = orders.astype(np.int64)
        gaps = np.abs(np.diff(np.asarray(levels), axis=0)).max(axis=1)
        print(f'  {flavour:<8} {name:<16} successive level gaps '
              f'{np.array2string(gaps, precision=3)}')


if __name__ == '__main__':
    main()
