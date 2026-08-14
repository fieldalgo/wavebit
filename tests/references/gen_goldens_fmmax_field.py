"""Freeze fmmax's in-layer **field** into ``golden_fmmax_field.npz``.

Run once, in the ``fwref`` environment; ``test_fmmax_field_golden.py`` then runs anywhere.

    conda run -n fwref python tests/references/gen_goldens_fmmax_field.py

Rerun it only to add cases — never to make a failing test pass, since the whole value of this
oracle is that its numbers did not come from this codebase.

**Why this exists.** ``golden_s4_field.npz`` gave field reconstruction its first external
check, but only on *uniform* stacks: on a patterned layer S4 and this project disagree at 1e-3
on the continuous components and up to 7.6e-1 on the discontinuous ``E_z``, because their
truncations and factorizations cannot be matched through S4's interface. Patterned fields were
therefore checked by nothing outside this repository. fmmax closes that, because both of those
knobs *are* reachable here — see ``_fmmax_field_cases.py`` for the two settings and what each
was worth.

**What agreement to expect, measured 2026-08-15.** Uniform layers land at 7.8e-16, which is the
same exactness the S4 field golden has and is what proves this generator's conventions are
right. Patterned layers land at **1.9e-5 (s) and 7.2e-5 (p)** — converged agreement, not
machine precision, because the two codes' vector factorizations still differ in detail. That is
the honest ceiling of this oracle and the test's tolerance says so out loud. It is worth having
anyway: every structural error this repository has actually shipped — a reversed sign, a
transposed raster, a swapped component — moved things by 1e-1, four orders above the floor.

**Axis order is measured, not assumed.** ``layer_fields_3d`` returns E and H each shaped
``(component, gx, gy, znum, nsrc)``. With ``RES_X == RES_Y == RES_Z`` that is unreadable; it was
established by asking for distinct sizes.

**JAX defaults to float32.** Without ``jax_enable_x64`` the uniform cases come back at ~1e-7 and
read as a mediocre solver rather than a mediocre configuration.
"""
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / 'golden_fmmax_field.npz'


def main():
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    import fmmax

    from _fmmax_field_cases import (CASES, FTO, NUM_TERMS, PERIOD, RES_X, RES_Y, RES_Z,
                                    THICK, WL, raster, x_samples)

    lat = fmmax.LatticeVectors(u=jnp.array([PERIOD, 0.0]), v=jnp.array([0.0, PERIOD]))
    expansion = fmmax.generate_expansion(lat, approximate_num_terms=NUM_TERMS,
                                         truncation=fmmax.Truncation.PARALLELOGRAMIC)
    n = expansion.num_terms
    assert n == NUM_TERMS, f'expansion returned {n} terms, cases assume {NUM_TERMS}'
    wl = jnp.asarray(WL)

    out = {}
    solved = {}
    for name, theta, pol in CASES:
        key = (name, theta)
        if key not in solved:
            kx = jnp.asarray([2 * np.pi / WL * np.sin(np.deg2rad(theta)), 0.0])
            vac = fmmax.eigensolve_isotropic_media(
                wl, kx, lat, jnp.ones((1, 1), dtype=complex), expansion,
                formulation=fmmax.Formulation.FFT)
            # `.T`: this project's [y, x] raster is fmmax's [x, y].
            lay = fmmax.eigensolve_isotropic_media(
                wl, kx, lat, jnp.asarray(raster(name).T ** 2, dtype=complex), expansion,
                formulation=fmmax.Formulation.JONES_DIRECT)
            th = [jnp.asarray(0.0), jnp.asarray(THICK), jnp.asarray(0.0)]
            solved[key] = (lay,
                           fmmax.stack_s_matrix([vac, lay], th[:2]),
                           fmmax.stack_s_matrix([lay, vac], th[1:]))
        lay, before, after = solved[key]

        idx = 0 if pol == 's' else n
        fwd0 = jnp.zeros((2 * n, 1), dtype=complex).at[idx, 0].set(1.0)
        f_in, b_in = fmmax.fields.layer_amplitudes_interior(
            before, after, fwd0, jnp.zeros((2 * n, 1), dtype=complex))
        E, H, (xg, _, zg) = fmmax.fields.layer_fields_3d(
            f_in, b_in, lay, jnp.asarray(THICK), layer_znum=RES_Z,
            grid_shape=(RES_X, RES_Y))
        # (component, gx, gy, znum, nsrc) -> (znum, gx, 6) at y index 0, source 0
        six = [np.asarray(A)[c, :, 0, :, 0].T for A in (E, H) for c in range(3)]
        out[f'{name}__{theta:g}__{pol}'] = np.stack(six, axis=-1)

    got_x = np.asarray(xg)[:, 0]
    assert np.allclose(got_x, x_samples(), atol=1e-9), (got_x, x_samples())
    out['x'] = got_x
    out['z'] = np.asarray(zg)
    out['orders'] = np.asarray(expansion.basis_coefficients)
    np.savez_compressed(str(OUT), **out)
    print(f'wrote {len(out)} arrays to {OUT.name}; {n} orders, '
          f'field shape {out[f"{CASES[0][0]}__{CASES[0][1]:g}__{CASES[0][2]}"].shape}')


if __name__ == '__main__':
    try:
        import fmmax  # noqa: F401
    except ImportError:
        sys.exit('needs the `fwref` environment (see README.md): conda run -n fwref ...')
    main()
