"""Freeze fmmax's answers for gratings on, in and under **lossy double negatives**.

Writes ``golden_fmmax_dng.npz``; run it as::

    conda run -n fwref python tests/references/gen_goldens_fmmax_dng.py

Read ``gen_goldens_fmmax.py`` first for the environment and float64 caveats, and
``gen_goldens_fmmax_laurent.py`` for the two corrections this generator shares with it;
``_fmmax_dng_cases.py`` says what the cases are and why the comparison is basis-free.

Frozen with fmmax 1.7.1 (MIT), jax 0.10.2, numpy 2.4.6, in the ``fwref`` env (2026-10-04). The
version is checked below: a different fmmax is refused, because the single-precision phase this
generator removes in its second run is a property of 1.7.1's ``fmmax.fft._fft2``.

What is frozen, per case, per flavour (``stock`` / ``phase64``) and per level ``R`` of
``LEVELS``: the **unpolarized** per-order reflected and transmitted efficiencies,
``[R/T, order]`` in the order sequence of ``__orders``, and their totals. Unpolarized means
``trace(H_in^-1 H_out) / 2``: the incident order-0 modes of the lit medium -- whatever mix of
polarizations fmmax's eigensolver returns them in -- are lit one at a time and in the two
combinations that fix the Hermitian incident flux form ``H_in``, and the same four solves give
each order's output form ``H_out``. The trace is the mean over any two incident polarizations
orthonormal in incident flux, so no basis has to be matched between the codes.

Each medium goes to fmmax as its own tensors: ``eigensolve_isotropic_media`` for glass and for
the non-magnetic gratings, whose analytic or isotropic modes avoid the general eigensolver's
sign choice on a lossless medium, and ``eigensolve_general_anisotropic_media`` with ``eps`` and
``mu`` for every double negative and for the third case's layer, where both are patterned.
Every negative-index medium here is lossy, which is where fmmax's ``Im(kz) > 0`` mode choice is
the physical one.

Refuses to freeze if the order-0 modes of the lit medium are not exactly two, if an incident
flux form is not positive definite, or if an order comes out negative or the total above one.
Every medium is passive; lit from glass that bounds ``R + T`` by one exactly, and lit from the
absorbing double negative it is a sanity check (incident and reflected waves interfere in the
flux there), which the frozen row passes with 1.2e-3 to spare.
"""
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

OUT = HERE / 'golden_fmmax_dng.npz'
FMMAX_VERSION = '1.7.1'


def main():
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    import fmmax
    from fmmax import fft as fmmax_fft
    from importlib.metadata import version
    from gen_goldens_fmmax_laurent import _float64_frequencies

    if version('fmmax') != FMMAX_VERSION:
        raise SystemExit(f'fmmax {version("fmmax")} is not {FMMAX_VERSION}; the phase64 flavour '
                         'removes a property of that release -- re-derive it before freezing')
    out = {}
    freeze(jnp, fmmax, out, flavour='stock')
    undo = _float64_frequencies(fmmax_fft, jnp)
    try:
        freeze(jnp, fmmax, out, flavour='phase64')
    finally:
        undo()
    from _fmmax_dng_cases import CASES, LEVELS
    out['__levels'] = np.asarray(LEVELS, dtype=np.int64)
    np.savez(OUT, **out)
    print(f'\nwrote {OUT} ({len(CASES)} cases, {len(LEVELS)} sampling levels, two flavours)')


def freeze(jnp, fmmax, out, flavour):
    from _fmmax_dng_cases import CASES, LEVELS, THICK, WL, incident_kxy, orders
    from _fmmax_laurent_cases import upsampled

    k0 = 2 * np.pi / WL
    wl = jnp.asarray(WL)

    def u1(v):
        return jnp.asarray(v, dtype=complex).reshape(1, 1)

    for name, case in CASES.items():
        period = case['period']
        px, py = period[0], (period[1] if len(period) > 1 else period[0])
        lat = fmmax.LatticeVectors(u=jnp.asarray([px, 0.0]), v=jnp.asarray([0.0, py]))
        basis = np.asarray(orders(name), dtype=np.int64)
        expansion = fmmax.Expansion(basis_coefficients=basis)
        n = expansion.num_terms
        kx, ky = incident_kxy(name)
        kxy = jnp.asarray([k0 * kx, k0 * ky])

        def medium(eps, mu):
            if complex(mu) == 1 and complex(eps).imag == 0:
                return fmmax.eigensolve_isotropic_media(
                    wl, kxy, lat, u1(eps), expansion, formulation=fmmax.Formulation.FFT)
            return fmmax.eigensolve_general_anisotropic_media(
                wl, kxy, lat, u1(eps), u1(0), u1(0), u1(eps), u1(eps),
                u1(mu), u1(0), u1(0), u1(mu), u1(mu), expansion,
                formulation=fmmax.Formulation.FFT)

        lit, far = medium(*case['top']), medium(*case['bot'])
        # The order-0 modes of the lit medium: the two eigenvectors whose weight is on the
        # zeroth Fourier term of both tangential components.
        vectors = np.asarray(lit.eigenvectors)
        owner = np.argmax(np.abs(vectors[:n]) ** 2 + np.abs(vectors[n:]) ** 2, axis=0)
        incident = [j for j in range(2 * n) if owner[j] == 0]
        if len(incident) != 2:
            raise SystemExit(f'{name}: {len(incident)} order-0 modes in the lit medium, not 2')
        combos = {'11': (1, 0), '22': (0, 1), '12': (1, 1), '1i2': (1, 1j)}

        levels = []
        for samples in LEVELS:
            # This project's [y, x] rasters as fmmax's [x, y] ones -- a relabelling, not a mirror.
            eps = jnp.asarray(upsampled(case['eps'], samples).T)
            if case['mu'] is None:
                layer = fmmax.eigensolve_isotropic_media(wl, kxy, lat, eps, expansion,
                                                         formulation=fmmax.Formulation.FFT)
            else:
                mu = jnp.asarray(upsampled(case['mu'], samples).T)
                zero = jnp.zeros_like(eps)
                layer = fmmax.eigensolve_general_anisotropic_media(
                    wl, kxy, lat, eps, zero, zero, eps, eps, mu, zero, zero, mu, mu, expansion,
                    formulation=fmmax.Formulation.FFT)
            smat = fmmax.stack_s_matrix([lit, layer, far], [jnp.asarray(0.0), jnp.asarray(THICK),
                                                            jnp.asarray(0.0)])
            nothing = jnp.zeros((2 * n, 1), dtype=complex)
            flux = {}
            for tag, (c1, c2) in combos.items():
                a = np.zeros((2 * n, 1), dtype=complex)
                a[incident[0], 0], a[incident[1], 0] = c1, c2
                a = jnp.asarray(a)
                f_in, _ = fmmax.amplitude_poynting_flux(a, nothing, lit)
                _, f_r = fmmax.amplitude_poynting_flux(nothing, smat.s21 @ a, lit)
                f_t, _ = fmmax.amplitude_poynting_flux(smat.s11 @ a, nothing, far)
                per_order = [np.asarray(f)[:n, 0] + np.asarray(f)[n:, 0] for f in (f_in, f_r, f_t)]
                flux[tag] = (per_order[0].sum(), -per_order[1], per_order[2])

            def form(k):
                f11, f22, f12, f1i2 = (np.asarray(flux[t][k]) for t in ('11', '22', '12', '1i2'))
                re, im = (f12 - f11 - f22) / 2, -(f1i2 - f11 - f22) / 2
                h = np.empty(np.shape(f11) + (2, 2), dtype=complex)
                h[..., 0, 0], h[..., 1, 1] = f11, f22
                h[..., 0, 1], h[..., 1, 0] = re + 1j * im, re - 1j * im
                return h

            h_in = form(0)
            if np.linalg.eigvalsh(h_in).min() <= 0:
                raise SystemExit(f'{name} R={samples}: the incident flux form is not positive '
                                 'definite -- the lit medium is not lit')
            inverse = np.linalg.inv(h_in)
            per_order = np.stack([np.real(np.trace(inverse[None] @ form(k), axis1=-2, axis2=-1))
                                  for k in (1, 2)]) / 2           # [R/T, order], unpolarized
            totals = per_order.sum(axis=1)
            if totals.sum() > 1 + 1e-9 or per_order.min() < -1e-12:
                raise SystemExit(f'{name} R={samples}: R + T = {totals.sum():.12f} on a passive '
                                 'stack, or a negative order -- the bridge is wired wrong')
            out[f'{name}__{flavour}__R{samples}__totals'] = totals
            out[f'{name}__{flavour}__R{samples}__per_order'] = per_order
            levels.append(totals)
            print(f'  {flavour:<8} {name:<20} R={samples:<4} '
                  f'{np.array2string(totals, precision=13)}')
        out[f'{name}__orders'] = basis
        out[f'{name}__kxy'] = np.asarray([kx, ky])
        gaps = np.abs(np.diff(np.asarray(levels), axis=0)).max(axis=1)
        print(f'  {flavour:<8} {name:<20} successive level gaps '
              f'{np.array2string(gaps, precision=3)}')


if __name__ == '__main__':
    main()
