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

**The 2D crossed patch, added 2026-08-15.** The cases above are y-invariant, so they freeze a
field with no ``ky`` content at all. ``CASES_2D`` is an x/y-asymmetric patch whose field varies
in y (measured 3.6e-1), stored on a full ``(z, y, x)`` grid rather than at one y index, at two
truncations (49 and 361 orders). See ``_fmmax_field_cases.py`` for why it is dielectric rather
than gold and why ``phi`` stays 0.

**Axis order is measured, not assumed.** ``layer_fields_3d`` returns E and H each shaped
``(component, gx, gy, znum, nsrc)``. With ``RES_X == RES_Y == RES_Z`` that is unreadable; it was
established by asking for distinct sizes. The 2D block relies on all three axes being
distinguishable, so it uses ``9 x 7 x 5``.

**JAX defaults to float32.** Without ``jax_enable_x64`` the uniform cases come back at ~1e-7 and
read as a mediocre solver rather than a mediocre configuration.
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
OUT = HERE / 'golden_fmmax_field.npz'


def main():
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    import fmmax

    from tests.references._fmmax_field_cases import (
        CASES, CASES_2D, NUM_TERMS, PERIOD, PERIOD_2D, RES_X, RES_Y, RES_Z, RES_Z_2D, THETA_2D,
        THICK, THICK_2D, WL, WL_2D, grid_2d, raster, raster_2d, x_samples, x_samples_2d,
        y_samples_2d, z_samples_2d)

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
            # `.T`: this project's [y, x] raster is fmmax's [x, y]. No squaring: `raster`
            # states a permittivity now, which is what this entry point always wanted -- the
            # `** 2` that used to sit here was converting an index case list, and that case
            # list is gone rather than the conversion having moved somewhere else.
            lay = fmmax.eigensolve_isotropic_media(
                wl, kx, lat, jnp.asarray(raster(name).T, dtype=complex), expansion,
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

    # --- the 2D crossed patch: a full (z, y, x) grid, and one expansion per truncation ---
    lat2 = fmmax.LatticeVectors(u=jnp.array([PERIOD_2D, 0.0]), v=jnp.array([0.0, PERIOD_2D]))
    wl2 = jnp.asarray(WL_2D)
    kxy = jnp.asarray([2 * np.pi / WL_2D * np.sin(np.deg2rad(THETA_2D)), 0.0])
    shape2, ix, iy = grid_2d()
    solved2 = {}
    for name, order, pol in CASES_2D:
        if (name, order) not in solved2:
            nt = (2 * order + 1) ** 2
            exp2 = fmmax.generate_expansion(lat2, approximate_num_terms=nt,
                                            truncation=fmmax.Truncation.PARALLELOGRAMIC)
            assert exp2.num_terms == nt, (order, exp2.num_terms, nt)
            vac2 = fmmax.eigensolve_isotropic_media(
                wl2, kxy, lat2, jnp.ones((1, 1), dtype=complex), exp2,
                formulation=fmmax.Formulation.FFT)
            # `.T` as above; `raster_2d` is a permittivity, which is what this entry point takes.
            lay2 = fmmax.eigensolve_isotropic_media(
                wl2, kxy, lat2, jnp.asarray(raster_2d(name).T, dtype=complex), exp2,
                formulation=fmmax.Formulation.JONES_DIRECT)
            th2 = [jnp.asarray(0.0), jnp.asarray(THICK_2D), jnp.asarray(0.0)]
            solved2[(name, order)] = (
                nt, exp2, lay2, fmmax.stack_s_matrix([vac2, lay2], th2[:2]),
                fmmax.stack_s_matrix([lay2, vac2], th2[1:]))
        nt, exp2, lay2, before2, after2 = solved2[(name, order)]
        fwd0 = jnp.zeros((2 * nt, 1), dtype=complex).at[0 if pol == 's' else nt, 0].set(1.0)
        f_in, b_in = fmmax.fields.layer_amplitudes_interior(
            before2, after2, fwd0, jnp.zeros((2 * nt, 1), dtype=complex))
        E, H, (xg2, yg2, zg2) = fmmax.fields.layer_fields_3d(
            f_in, b_in, lay2, jnp.asarray(THICK_2D), layer_znum=RES_Z_2D, grid_shape=shape2)
        # (component, gx, gy, znum, nsrc) -> (znum, gy, gx), then onto the stored points
        six = [np.transpose(np.asarray(A)[c, :, :, :, 0], (2, 1, 0))
               for A in (E, H) for c in range(3)]
        out[f'2d__{name}__{order}__{pol}'] = np.stack(six, axis=-1)[:, iy][:, :, ix]
        out[f'2d_orders__{order}'] = np.asarray(exp2.basis_coefficients)

    gx2, gy2 = np.asarray(xg2)[ix, 0], np.asarray(yg2)[0, iy]
    assert np.allclose(gx2, x_samples_2d(), atol=1e-9), (gx2, x_samples_2d())
    assert np.allclose(gy2, y_samples_2d(), atol=1e-9), (gy2, y_samples_2d())
    assert np.allclose(np.asarray(zg2), z_samples_2d(), atol=1e-9), (zg2, z_samples_2d())
    out['2d_x'], out['2d_y'], out['2d_z'] = gx2, gy2, np.asarray(zg2)

    np.savez_compressed(str(OUT), **out)
    print(f'wrote {len(out)} arrays to {OUT.name}; {n} orders, '
          f'field shape {out[f"{CASES[0][0]}__{CASES[0][1]:g}__{CASES[0][2]}"].shape}')


if __name__ == '__main__':
    try:
        import fmmax  # noqa: F401
    except ImportError:
        sys.exit('needs the `fwref` environment (see README.md): conda run -n fwref ...')
    main()
