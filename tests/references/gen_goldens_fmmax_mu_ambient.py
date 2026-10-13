"""Freeze fmmax's **magnetic half-space** efficiencies into ``golden_fmmax_mu_ambient.npz``.

    conda run -n fwref python tests/references/gen_goldens_fmmax_mu_ambient.py

Read ``gen_goldens_fmmax.py`` first for the environment and float64 caveats; they apply here
unchanged. ``_fmmax_mu_ambient_cases`` states what these cases are and why the bare-interface
half exists.

**Run before the solver could express a magnetic half-space**, deliberately. ``boundary_kz``
takes one material argument and the p-channel admittance is spelled ``/ eps_bot``; there is
nowhere to put a ``mu``, so none of these rows could be fitted to anything. That is the same
discipline ``gen_goldens_fmmax_magnetic.py`` was written under, and on that occasion it caught a
real error on the first run.

Two guards on the frozen numbers themselves, because a magnetic ambient has no ``mu = 1`` twin
that would expose a mis-wiring the way the layer golden's did:

* **the closed form.** Every ``ZERO_THICKNESS_CASES`` row is one bare interface, whose Fresnel
  coefficients follow from the admittances ``kz/mu`` and ``kz/eps`` with no solver involved.
  Enforced here to 1e-12. If fmmax's ambient handling, flux normalisation or units were wrong,
  these rows say so at generation time.
* **energy.** Every medium in every case is real and lossless, so ``R + T = 1`` holds for both
  polarizations and both halves of the file.

Note the flux normalisation: the transmitted flux is taken against the **substrate's** solve,
not the superstrate's. With asymmetric ambients those differ, and using the incidence medium for
both would silently rescale ``T`` — which the energy guard would then catch, which is why the
guard is worth having even though the line above it looks obvious.
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

OUT = HERE / 'golden_fmmax_mu_ambient.npz'


def main():
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    import fmmax

    from tests.references._fmmax_mu_ambient_cases import CASES, PERIOD, THICK, WL, ZERO_THICKNESS_CASES, fresnel

    lat = fmmax.LatticeVectors(u=jnp.array([PERIOD, 0.0]), v=jnp.array([0.0, PERIOD]))
    expansion = fmmax.generate_expansion(lat, approximate_num_terms=1,
                                         truncation=fmmax.Truncation.CIRCULAR)
    u = lambda v: jnp.asarray(v, dtype=complex).reshape(1, 1)

    def medium(eps, mu, kx):
        """A uniform half-space or layer, from its two tensors."""
        e = np.asarray(eps, dtype=complex) if np.ndim(eps) else np.eye(3) * eps
        m = np.asarray(mu, dtype=complex) if np.ndim(mu) else np.eye(3) * mu
        return fmmax.eigensolve_general_anisotropic_media(
            jnp.asarray(WL), kx, lat,
            u(e[0, 0]), u(e[0, 1]), u(e[1, 0]), u(e[1, 1]), u(e[2, 2]),
            u(m[0, 0]), u(m[0, 1]), u(m[1, 0]), u(m[1, 1]), u(m[2, 2]),
            expansion, formulation=fmmax.Formulation.FFT)

    def efficiencies(top, bot, layer_eps, layer_mu, theta_deg, thick):
        eps_t, mu_t = top
        # The in-plane wavevector is set by the INCIDENCE medium, so a magnetic superstrate
        # moves it: n_inc = sqrt(eps*mu), not sqrt(eps).
        n_inc = np.sqrt(complex(eps_t * mu_t)).real
        kx = jnp.asarray([2 * np.pi / WL * n_inc * np.sin(np.deg2rad(theta_deg)), 0.0])

        sup = medium(eps_t, mu_t, kx)
        sub = medium(bot[0], bot[1], kx)
        lay = medium(layer_eps, layer_mu, kx)
        smat = fmmax.stack_s_matrix(
            [sup, lay, sub],
            [jnp.asarray(0.0), jnp.asarray(thick), jnp.asarray(0.0)])

        n = expansion.num_terms
        vals = []
        for idx in (0, n):                        # s incidence, then p
            fwd = jnp.zeros((2 * n, 1), dtype=complex).at[idx, 0].set(1.0)
            f_in, b_in = fmmax.amplitude_poynting_flux(fwd, smat.s21 @ fwd, sup)
            f_out, _ = fmmax.amplitude_poynting_flux(
                smat.s11 @ fwd, jnp.zeros((2 * n, 1), dtype=complex), sub)
            inc = float(jnp.sum(f_in))
            vals += [abs(float(jnp.sum(b_in))) / inc, float(jnp.sum(f_out)) / inc]
        return np.asarray(vals, dtype=float)

    out = {}

    print('bare interfaces -- checked against the closed form, no solver involved:')
    for name, ((eps_s, mu_s), theta) in ZERO_THICKNESS_CASES.items():
        vals = efficiencies((1.0, 1.0), (eps_s, mu_s), np.eye(3), np.eye(3), theta, 0.0)
        want = fresnel(eps_s, mu_s, theta)
        gap = np.abs(vals - want).max()
        if gap > 1e-12:
            raise SystemExit(f'{name}: {gap:.2e} from the closed-form Fresnel answer — fmmax is '
                             'not handling this ambient the way the formula says, do not freeze')
        out[name] = vals
        print(f'  {name:<40} {np.array2string(vals, precision=9)}  (closed form {gap:.1e})')

    print('\nstacks:')
    for name, (top, bot, layer_eps, layer_mu, theta) in CASES.items():
        vals = efficiencies(top, bot, layer_eps, layer_mu, theta, THICK)
        for pol, (r, t) in (('s', vals[:2]), ('p', vals[2:])):
            if abs(r + t - 1.0) > 1e-9:
                raise SystemExit(f'{name} {pol}: R+T = {r + t:.12f} for a lossless stack — the '
                                 'bridge is wired wrong, do not freeze this')
        out[name] = vals
        print(f'  {name:<40} {np.array2string(vals, precision=9)}')

    np.savez(OUT, **out)
    print(f'\nwrote {OUT} ({len(out)} rows)')


if __name__ == '__main__':
    main()
