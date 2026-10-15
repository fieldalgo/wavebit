# Five slots out of nine — the shape every second-order RCWA has, and what happens at its edge

Written: 2026-08-11. Measured on CPU, float64: NumPy 2.5.1 for this solver, and each reference
solver at its current release (fmmax 1.7.1, inkstone 0.3.15, EMpy 2.2.3, RETICOLO V10 2025-01),
versions checked against PyPI and Zenodo rather than against notes.

Targets:
- `src/wavebit/rcwa/_engine/solve_2d.py::layer_modes_2d` — the five blocks it reads
- `src/wavebit/rcwa/_engine/solve_2d_first_order.py` — the formulation that reads all nine
- `tests/rcwa/test_anisotropic_berreman.py::test_voigt_geometry_matches_the_oracle` — and its
  `::test_a_z_row_is_refused_by_the_formulation_that_cannot_hold_it` twin
- `tests/references/README.md` — the per-solver measurements summarised in §4

Companion to two documents that answer neighbouring questions.
[`anisotropic-materials.md`](../../guide/rcwa/anisotropic-materials.md) asks what the solver accepts and what each
gap costs. [`off-diagonal-slot-convention.md`](off-diagonal-slot-convention.md) asks what to put
in the slots by hand. **This one asks a question neither does: the five-slot shape is not a
Wavebit decision — every second-order RCWA has it — so what do the other four components mean,
and what does each solver do when handed them?**

---

## TL;DR

1. A permittivity tensor has nine components. A second-order RCWA reads **five**: the
   `xx, xy, yx, yy` block plus `zz`. The other four — `xz, yz, zx, zy` — have no slot, and that
   is structural, not an omission.
2. `zz` survives *inverted* because eliminating `E_z` needs `eps_zz⁻¹`. The elimination closes
   into a second-order equation **only if `xz` and `yz` are zero**; otherwise a first-derivative
   term survives and there is no second-order operator to write.
3. The four missing components are the **Voigt geometry** — a magnetisation lying in the plane,
   which is what an ordinary magnetised film has.
4. Four reference solvers were run at their current release. **Three of the four that claim a
   full 3×3 do not deliver one**, and they fail in three different ways: refuse, silently drop,
   diverge.
5. **Energy conservation cannot detect the silent drop.** The reduced tensor is a valid lossless
   medium, so `R + T = 1` holds to machine precision while the answer is wrong by up to 6e-2.
   This is why Wavebit never drops: the four components select the first-order formulation,
   and the one way to ask for the second-order one over a z row (`formulation='second_order'`)
   raises.

---

## 1. Why five, and why those five

Under `e^{-iwt}` with `d/dx -> i k0 Kx`, the `z` component of `curl H = -i k0 eps E` gives

```
    D_z = ezx Ex + ezy Ey + ezz Ez  =  Ky Hx - Kx Hy
```

so `E_z` is not an independent unknown — it can be solved for and substituted away:

```
    Ez = ezz⁻¹ (Ky Hx - Kx Hy - ezx Ex - ezy Ey)
                 └──────┬──────┘  └───────┬──────┘
              always present         only if the z row is populated
```

That is where the fifth slot comes from, and why it is stored inverted (`epz_conv_i`): the solver
never wants `eps_zz`, only `eps_zz⁻¹`, and that inverse is the position where Li's inverse rule
applies. Substituting back leaves

```
                        E_x         E_y         E_z
                   ┌───────────┬───────────┬───────────┐
           D_x     │    epx    │   epxy    │     ○     │
                   ├───────────┼───────────┼───────────┤
           D_y     │   epyx    │    epy    │     ○     │
                   ├───────────┼───────────┼───────────┤
           D_z     │     ○     │     ○     │  epz_i⁻¹  │
                   └───────────┴───────────┴───────────┘

    named  = a slot layer_modes_2d reads      ○ = assumed zero, no slot exists
```

With the `○` cells zero the substitution produces `d²E_t/dz² = Omega2 E_t` — second order, no
first-derivative term, forward and backward modes sharing one eigenvector. **That last property
is what the whole second-order machinery is built on**, and it is what the missing four take away.

## 2. What the missing four are

`eps_xz` and `eps_yz` couple the in-plane field to the stack normal. For a magneto-optic medium
`eps_ij = eps·δ_ij + i·ε_ijk·g_k`, so the magnetisation direction picks which cells fill:

| magnetisation | populated | second-order? | name |
|---|---|---|---|
| along `z` (stack normal) | `xy`, `yx` | yes | Faraday |
| in the plane (`x` or `y`) | `yz, zy` or `zx, xz` | **no** | Voigt |
| tilted | all six off-diagonal | **no** | — |

The middle row is the ordinary case. A magnetised film magnetises *in its own plane* unless it has
perpendicular anisotropy or an applied field holding it out — so the geometry the second-order
formulation cannot express is not exotic, it is the default. `anisotropic-materials.md` §6 has the
physics; this document is about the boundary itself.

## 3. The shape is not ours

It is tempting to read the five slots as a Wavebit simplification. It is not — it is what a
second-order formulation *is*, and the same five appear wherever one is used:

| | how the five appear |
|---|---|
| Wavebit second-order | `solve_conv(epx, epy, epz_i, epxy, epyx)` |
| fmmax | `eigensolve_anisotropic_media(..., permittivity_xx, xy, yx, yy, zz, ...)` |
| S4 | `xy` block plus `zz` |

fmmax's `eigensolve_general_anisotropic_media` adds five more for `mu` and nothing for the z row.
**"General" there means magnetic, not z-coupled** — a reading worth stating because this repo's
own survey once recorded it as "fully anisotropic" and was wrong for a while.

## 4. Four ways to meet the edge

Each solver below was handed a tensor with a populated z row, at its current release. Two checks
decide every row, and neither needs another solver to agree:

- **Lossless.** A real symmetric `eps` is Hermitian, hence lossless, hence `R + T = 1` per
  incident polarisation *by theorem*. A real symmetric matrix stays real symmetric under
  transposition and under any permutation of axes, so no convention difference can explain a
  violation.
- **Used or not.** Zero `xz, zx, yz, zy` and compare. Identical output means they were ignored.

| solver | behaviour | evidence |
|---|---|---|
| **Wavebit second-order** | refuses | `ValueError` naming the slot, only reachable by asking for `formulation='second_order'` |
| **Wavebit patterned (PN)** | refused; handles it since 2026-09-24 | `ValueError` naming the z row from 2026-08-27; before that it **accepted and silently dropped** it, which is the row below, in this table, about this solver. Since 2026-09-24 a patterned z-coupled tensor takes the first-order path under PN or tensor Li ([full tensor patterns](../../guide/rcwa/full-tensor-patterns.md)) |
| **fmmax** 1.7.1 | cannot be asked | no `xz/yz/zx/zy` arguments exist |
| **inkstone** 0.3.15 | **accepts, silently drops** | takes a `(3,3)` array; output bit-identical to the z row zeroed |
| **EMpy** 2.2.3 | reads it, diverges | uses the z row, but returns `R + T = 1.6e13` for a plain Faraday tensor at normal incidence |
| **RETICOLO** V10 | handles it, within a bound | exact for one off-diagonal pair at `delta0 = 0`; past that `R + T` leaves 1 by 2.5e-3 to 1.2e-2 |
| **Wavebit first-order** | handles it, and is the default here | `solve_2d_first_order.py`, agreeing with RETICOLO to 1e-15 inside RETICOLO's bound — through the public API as well as directly |

The per-solver numbers, and how each was established, are in
[`tests/references/README.md`](https://github.com/fieldalgo/wavebit/blob/develop/tests/references/README.md).

## 5. Why the silent drop is the dangerous one

Dropping the z row does not produce nonsense. It produces **a different, perfectly valid problem**
— a lossless medium if the original was lossless — and every cheap check a caller would run comes
back clean:

| check | on a Voigt tensor with the z row dropped |
|---|---|
| solve completes | yes |
| `R + T = 1` | yes, to machine precision |
| reciprocity | yes |
| the answer | **wrong by 2e-2 to 6e-2** |

That table is why `test_a_z_row_is_refused_by_the_formulation_that_cannot_hold_it` exists, and
why the same numbers are quoted in the test that now *answers* the geometry rather than
refusing it: the failure being guarded against is not a crash. It is also
exactly what inkstone was measured doing — thirty lossless configurations conserving energy to
1.3e-15, every z-decoupled case matching Wavebit to 1e-15, and only the z-coupled ones quietly
4e-4 to 3e-2 out. Without the refusal, Wavebit would look identical.

**Energy conservation is a necessary check, not a sufficient one.** The same caution appears in
`tests/rcwa/test_anisotropic_berreman.py` and in
[`off-diagonal-slot-convention.md`](off-diagonal-slot-convention.md) §3, where a wrong *sign*
is likewise invisible to every power efficiency.

## 6. What crossing the edge costs

Reading all nine means abandoning the second-order reduction for the first-order system

```
    d/dz (Ex, Ey, Hx, Hy) = i k0 M (Ex, Ey, Hx, Hy)          M is 4N x 4N for N orders
```

which is roughly **8x** on the layer solve (`eig` is cubic and the problem doubles), and which
gives up the free `±` mode pairing every second-order consumer assumes — forward and backward
become genuinely different mode sets. `src/wavebit/rcwa/_engine/solve_2d_first_order.py` implements it, and
since 2026-08-13 the public entry does reach it: `solve_conv` takes nine slots rather than five,
and giving any of the four z-row slots selects the first-order path (`formulation='first_order'`
forces it, `'second_order'` refuses a z row rather than dropping it). What crossing the edge also
cost, beyond the 8x, was `mu` and `layer_algorithm='SMM'`; both have run on the first-order path
since 2026-09-24, and the module's docstring keeps what it supports.
Field reconstruction was on it until 2026-08-14 and is not any more —
`reconstruct_field` reads the formulation, since the grating type cannot tell the two paths apart.

Because no outside solver covers the general z-coupled case, the first-order path is judged partly
by a theorem rather than by an oracle: turning the tensor and the incidence azimuth together about
the stack normal must leave the efficiencies unchanged
(`tests/rcwa/test_solve_2d_first_order.py::test_rotating_the_whole_problem_about_z_changes_nothing`).

## References

- D. W. Berreman, "Optics in Stratified and Anisotropic Media: 4×4-Matrix Formulation",
  *J. Opt. Soc. Am.* **62**(4), 502 (1972). doi:10.1364/JOSA.62.000502
- L. Li, "Fourier modal method for crossed anisotropic gratings with arbitrary permittivity and
  permeability tensors", *J. Opt. A* **5**, 345 (2003).
- J. P. Hugonin and P. Lalanne, "RETICOLO software for grating analysis", arXiv:2101.00901.
- V. Liu and S. Fan, "S4: A free electromagnetic solver for layered periodic structures",
  *Comput. Phys. Commun.* **183**, 2233 (2012).
