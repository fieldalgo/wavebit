"""Freeze fmmax's *magnetic* slab efficiencies into ``golden_fmmax_magnetic.npz``.

    conda run -n fwref python tests/references/gen_goldens_fmmax_magnetic.py

**Nothing in this solver can consume the mu rows yet.** ``docs/anisotropic-materials.md`` §5
records that permeability has no slot anywhere: ``layer_modes_2d`` writes ``I`` where a
``mu_conv`` would go, and ``boundary_kz`` takes one material argument. This golden is laid down
*before* that work rather than after, for one reason — an oracle chosen after the fact is an
oracle chosen to agree.

So the file is built to be trustworthy on its own terms today:

* Every case is also generated at ``mu = 1``, and those rows **are** checked against the solver
  (``test_fmmax_magnetic_golden.py``). If the harness, conventions or units were wrong, that
  subset would say so now, while the answer is knowable. It is the same convention gate the
  other three oracles went through, applied to an oracle whose interesting half cannot be
  checked yet.
* Every magnetic case is Hermitian in both ``eps`` and ``mu``, hence lossless, so ``R + T = 1``
  is a real constraint on the frozen numbers and is enforced here.

``fmmax.eigensolve_general_anisotropic_media`` takes five permittivity and five permeability
components — ``xx, xy, yx, yy, zz`` each, no z row — so it is the same second-order tensor class
this solver uses, extended to mu. A magnetic Voigt geometry is out of its reach as much as ours;
RETICOLO is the oracle if that is ever wanted.

Environment and float64 caveats are the same as ``gen_goldens_fmmax.py``; read that first.
"""
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

OUT = HERE / 'golden_fmmax_magnetic.npz'

WL = 700.0
PERIOD = 400.0
THICK = 210.0


def _t(xx, yy, zz, xy=0.0, yx=0.0):
    return np.array([[xx, xy, 0.0], [yx, yy, 0.0], [0.0, 0.0, zz]], dtype=complex)


ONE = _t(1.0, 1.0, 1.0)

# name -> (eps, mu, theta_deg). Each entry is also run with mu = 1 under the name + '__mu1',
# and those are the rows the solver can be held to today.
CASES = {
    'iso_eps_scalar_mu': (_t(2.6, 2.6, 2.6), _t(1.7, 1.7, 1.7), 0.0),
    'iso_eps_scalar_mu_39deg': (_t(2.6, 2.6, 2.6), _t(1.7, 1.7, 1.7), 39.0),
    'iso_eps_uniaxial_mu_39deg': (_t(2.6, 2.6, 2.6), _t(1.4, 1.4, 2.2), 39.0),
    'biaxial_eps_biaxial_mu_21deg': (_t(2.3, 3.0, 2.6), _t(1.3, 1.8, 1.5), 21.0),
    'gyroelectric_46deg': (_t(2.6, 2.6, 2.6, 0.35j, -0.35j), ONE, 46.0),
    'gyromagnetic_46deg': (_t(2.6, 2.6, 2.6), _t(1.6, 1.6, 1.6, 0.40j, -0.40j), 46.0),
    'gyroelectric_and_gyromagnetic_46deg': (_t(2.6, 2.6, 2.6, 0.35j, -0.35j),
                                            _t(1.6, 1.6, 1.6, 0.40j, -0.40j), 46.0),
}


def main():
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    import fmmax

    lat = fmmax.LatticeVectors(u=jnp.array([PERIOD, 0.0]), v=jnp.array([0.0, PERIOD]))
    expansion = fmmax.generate_expansion(lat, approximate_num_terms=1,
                                         truncation=fmmax.Truncation.CIRCULAR)

    def u(v):
        return jnp.asarray(v, dtype=complex).reshape(1, 1)

    def run(eps, mu, theta_deg):
        e, m = np.asarray(eps, complex), np.asarray(mu, complex)
        kx = jnp.asarray([2 * np.pi / WL * np.sin(np.deg2rad(theta_deg)), 0.0])
        wl = jnp.asarray(WL)
        vac = fmmax.eigensolve_isotropic_media(wl, kx, lat, u(1.0), expansion,
                                               formulation=fmmax.Formulation.FFT)
        lay = fmmax.eigensolve_general_anisotropic_media(
            wl, kx, lat,
            u(e[0, 0]), u(e[0, 1]), u(e[1, 0]), u(e[1, 1]), u(e[2, 2]),
            u(m[0, 0]), u(m[0, 1]), u(m[1, 0]), u(m[1, 1]), u(m[2, 2]),
            expansion, formulation=fmmax.Formulation.FFT)
        smat = fmmax.stack_s_matrix([vac, lay, vac],
                                    [jnp.asarray(0.0), jnp.asarray(THICK), jnp.asarray(0.0)])
        n = expansion.num_terms
        vals = []
        for idx in (0, n):
            fwd = jnp.zeros((2 * n, 1), dtype=complex).at[idx, 0].set(1.0)
            f_in, b_in = fmmax.amplitude_poynting_flux(fwd, smat.s21 @ fwd, vac)
            f_out, _ = fmmax.amplitude_poynting_flux(
                smat.s11 @ fwd, jnp.zeros((2 * n, 1), dtype=complex), vac)
            inc = float(jnp.sum(f_in))
            vals += [abs(float(jnp.sum(b_in))) / inc, float(jnp.sum(f_out)) / inc]
        return np.asarray(vals, dtype=float)

    def hermitian(a):
        a = np.asarray(a, complex)
        return np.abs(a - a.conj().T).max() <= 1e-12

    out = {}
    for name, (eps, mu, theta_deg) in CASES.items():
        for suffix, m in (('', mu), ('__mu1', ONE)):
            key = name + suffix
            vals = run(eps, m, theta_deg)
            if hermitian(eps) and hermitian(m):
                for pol, (r, t) in (('s', vals[:2]), ('p', vals[2:])):
                    if abs(r + t - 1.0) > 1e-9:
                        raise SystemExit(
                            f'{key} {pol}: R+T = {r + t:.12f} for a lossless Hermitian '
                            'eps/mu pair — the bridge is wired wrong, do not freeze this')
            out[key] = vals
            tag = 'mu = 1, checkable today' if suffix else 'magnetic, no consumer yet'
            print(f'  {key:<40} {np.array2string(vals, precision=9)}  [{tag}]')

    np.savez(OUT, **out)
    print(f'\nwrote {OUT} ({len(out)} rows)')


if __name__ == '__main__':
    main()
