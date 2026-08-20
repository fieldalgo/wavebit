# Magnetic materials — `mu` on layers and half-spaces

Landed 2026-08-14. [`anisotropic-materials.md`](anisotropic-materials.md) §5 had scoped this and
got the location right and the content wrong; §3 below is the difference.

Target: `wavebit/_core/solve_2d.py` — `layer_modes_2d`, `solve_2d`;
`wavebit/_core/scattering.py` — `smm_2d`.

**Scope.** The **2D path** — layers (§1–§4), half-spaces (§5–§6), and patterned layers (§7).
Neither the 1D/conical operators nor the first-order 4ff solve take a `mu`; both refuse rather
than ignore.

---

## 1. Where it goes

Under `e^{-iωt}` the two curl equations are

```
∇ × E  = +i k0 μ H̃                                   (M1)
∇ × H̃  = -i k0 ε E                                   (M2)
```

and the substitution `(E, H̃, ε, μ) → (H̃, -E, μ, ε)` maps the pair onto itself. That is the whole
design: `mu` is not a second kind of input, it is the same five slots on the other side.

Eliminating the longitudinal components — `H̃z` through `μzz` now, `Ez` through `εzz` as before —
and writing `S = [Ex; Ey]`, `U = +i[H̃x; H̃y]`:

```
dS/dz̃ = -Ω_L U          Ω_L = [[-Cc,  B ],        B  = Kx εz⁻¹ Kx - μy
                                [-D,   E ]]        D  = Ky εz⁻¹ Ky - μx
                                                   Cc = Kx εz⁻¹ Ky - μyx̂
                                                   E  = Ky εz⁻¹ Kx - μxŷ

dU/dz̃ = -Ω_R S          Ω_R = [[-Kx μz⁻¹ Ky + εyx̂,  Kx μz⁻¹ Kx - εy ],
                                [ εx - Ky μz⁻¹ Ky,   Ky μz⁻¹ Kx - εxŷ]]
```

So `Ω_L` — which eliminates `Ez` through `εzz` — carries the *transverse* `mu`, and `Ω_R` — which
eliminates `H̃z` through `μzz` — carries the transverse `eps`. The two hats mark the slot
convention, which is the permittivity's unchanged: diagonals as they are, `zz` inverted,
off-diagonals negated.

**The derivation was not trusted on sight.** It was required to reproduce the shipped `Omega2_LR`
term for term at `mu = 1` before anything downstream was written — a mismatch there would have
meant the derivation described a different operator, and every magnetic number built on it would
have been wrong in a way no test states.

## 2. The cross pairing

`B` takes `μ_yy` and `D` takes `μ_xx`, not the other way round. It reads like a transposition
error and is not: `B` sits in `Ω_L[0,1]`, the coefficient of `Uy` in `dEx/dz̃`, which (M1)'s
`y`-row pairs with `μ_yy`.

Getting it backwards is invisible on everything except a **uniaxial or biaxial** `mu`. A scalar
permeability is symmetric under the swap, so the isotropic case — the one anybody checks first —
passes either way.

## 3. What `Omega2_LR`'s bare `K @ K` terms actually are

`layer_modes_2d` does not build `Ω_L` and multiply. It carries the product hand-expanded:

```python
Omega2_LR[0,0] = Ky @ Ky + B @ epx_conv - Cc @ epyx_conv
```

That `Ky @ Ky` is **not** geometry that passed through the multiplication untouched. Written out,
the block is

```
  Cc·(Kx μz⁻¹ Ky) - B·(Ky μz⁻¹ Ky) + B·εx - Cc·εyx̂
```

and the first two terms mostly cancel — the `εz⁻¹` halves commute past each other because `Kx`
and `Ky` are diagonal — leaving `μy·(Ky μz⁻¹ Ky) + μyx̂·(Kx μz⁻¹ Ky)`. That is `Ky @ Ky` **only**
when `μy = 1`, `μyx̂ = 0` and `μz⁻¹ = 1`.

So the expanded form is valid at `mu = 1` and nowhere else, and `layer_modes_2d` branches: no
`mu` keeps it verbatim, a `mu` multiplies `Ω_L @ Ω_R` out. Keeping the branch rather than always
multiplying is what makes a non-magnetic solve bit-untouched by this feature.

The cost of missing this is recorded because of how it presented. With `Ω_R` corrected and the
expanded form left alone:

| case | result |
|---|---|
| scalar `mu`, normal **and** 39° | **exact** (1e-16) |
| uniaxial `mu` | wrong by 1e-2, **in the s channel only** |
| biaxial `mu` | wrong by 1e-3, s only |
| gyromagnetic `mu` | `R + T = 1.035` — **energy created** |

A scalar `mu` being exact at oblique incidence is what makes this dangerous: it is the case that
looks like proof the `mu_zz` path works.

## 4. What checks it

**The golden, which predates the code.** `tests/references/golden_fmmax_magnetic.npz` was frozen
while the solver had nowhere to put a `mu`, on the explicit reasoning that an oracle chosen after
the work is an oracle chosen to agree. All seven magnetic cases now match fmmax to ~2e-15, under
both connecting algorithms and all three backends. `tests/references/test_fmmax_magnetic_golden.py`.

**The duality, which needs no oracle.** Exchanging `eps` and `mu` trades the s and p channels —
exactly, at any incidence, for scalar, uniaxial, biaxial and gyromagnetic tensors alike (7e-16).
It reaches *only* the `mu` path, so no solver that quietly ignores permeability can satisfy it,
and it fails if the transverse `mu` lands anywhere but its dual position — including under the
uncrossed pairing of §2, which survives energy conservation and reciprocity untouched.

Stated as a **trade** rather than invariance. Invariance is the natural guess and is wrong even
at normal incidence: a biaxial in-plane tensor still distinguishes x from y, so the swap moves
the answer by 4.7e-2 there while trading it exactly.

**The `__mu1` half**, which is now a regression gate. Every golden case exists twice, and the
`mu = 1` rows say the non-magnetic path is untouched — against numbers generated before the
feature.

`tests/test_magnetic_layer.py` carries the oracle-free half.

## 5. Half-spaces

Landed the same day, against its own golden. `n_top` / `n_bot` keep meaning the **refractive
index** `sqrt(eps*mu)`, and `mu_top` / `mu_bot` are new. That choice is
what makes the change small: `kz` depends only on the index, so Snell, the order grid and
`boundary_kz` are untouched — `boundary_kz` was already taking `n^2` under the name `eps`.

Since the material flip (2026-08-20) the half-space *is* a `Material`: `top` and `bot` hold
`(epsilon, mu)`, and `n_top` / `n_bot` are read-only properties derived from them,
`sqrt(eps*mu)`, taken once at ingest so the ~43 places that read an index keep working
([`conventions.md`](conventions.md)). The engine therefore passes `eps_top` / `eps_bot`
alongside `mu_top` / `mu_bot`, and both 2D solves (`solve_2d` and `smm_2d`) read them where
they used to reconstruct. The old recovery `eps = n^2/mu` is still
there and still correct, but it is now the *fallback* for a caller who reaches
`solve_for_conv` holding only an index — going through a square root and back loses digits the
material was carrying exactly.

Everything that changes is one pair of admittances:

```
  s (electric amplitude):  Y = kz / mu
  p (magnetic amplitude):  Y = kz / eps  =  kz * mu / n^2
```

and it appears in four places that must agree: the substrate seed (`big_F` / `big_G`), the
superstrate rows of `final_A`, the incident column, and the efficiency normalisation.

**The incident normalisation is the part that is easy to miss.** `a_s` and `a_p` are defined at
*equal incident flux*; once the s flux carries `kz/mu`, the p amplitude's magnetic scale must be
`n/mu` rather than `n` for the two to still match. Get it wrong and s and p are normalised to
different incident powers — and **`R + T = 1` still holds within each polarization**, so no
energy check sees it.

That is the general shape of the risk here, and why the golden carries a closed form rather than
leaning on energy: a boundary that used `kz/eps` for *both* polarizations conserves energy
perfectly. A bare interface has an exact answer, so §6 can check the two admittances separately.

### The gap layer

The SMM cascade inserts a fictitious "gap" medium so every interface is computed the same way.
It is left non-magnetic deliberately. It is a bookkeeping device, not a medium in the scene, and
it cancels out of the result — giving it a material property would be a category error that
happens to cancel, which is harder to notice than one that does not.

## 6. What checks the half-spaces

**A golden frozen before the boundary could hold a `mu`**, same discipline as the layer half:
`golden_fmmax_mu_ambient.npz`, generated while `boundary_kz` took one material argument.
Sixteen rows, matched to ~1e-15 under both connecting algorithms.

**A closed form**, which is what replaces the layer golden's `__mu1` gate — there is no
non-magnetic twin of an ambient under test. Every bare-interface row is checkable by hand:

```
  R = |(Y_inc - Y_sub)/(Y_inc + Y_sub)|^2,   Y = kz/mu (s), kz/eps (p)
```

The solver matches it to 4e-16. Both the generator and the test enforce it, so a wrong
admittance is caught without fmmax being trusted at all.

**One case that cannot be passed by accident.** `impedance_only_mismatch_30deg` is
`eps = 0.5, mu = 2.0`: index-matched to vacuum, impedance-mismatched. A boundary carrying only
`n` sees no interface and returns `R = 0`. The answer is 0.111.

**Reciprocity** for the backside mirror, where `mu_top` and `mu_bot` must swap. Transmittance is
direction-independent for a lossless reciprocal stack — but the **angle has to be converted**,
because `theta` is stated against the incidence medium's normal. Comparing top-at-33° with
bottom-at-33° puts `T` out by 9e-3 and reads exactly like a broken mirror; what reciprocity
equates is one `k_par` seen from both sides, `n_top sin(th_top) = n_bot sin(th_bot)`.

### What a mutation survey added

Two mutants of the incident column survived the first survey, and measuring them showed they
changed *nothing* — they were not redundant code, they were in the copy of that column the
tests never reached. Every magnetic assertion at that point read `res_te_inc` / `res_tm_inc`,
built from a different block than `res`, the psi-specified incidence. The gap was not subtle:
the psi path returns `R + T = 1.575` with the incident s-flux left non-magnetic. Deleting the
backside swap survived everything too. With the psi and reciprocity tests added, 15 of 15
mutants die.

## 7. Patterned `mu`

**No new factorization was needed**, and that is the finding rather than a convenience.
`pn_conv_matrices` — the Popov-Nevière rearrangement written for a patterned tensor `eps`
([`patterned-anisotropic-factorization.md`](patterned-anisotropic-factorization.md)) — serves
`mu` verbatim. The rearrangement is driven entirely by which quantities are continuous at an
interface:

```
  tangential E and normal D    ->    tangential H and normal B
```

and the second line is the first with every symbol replaced. `D = eps E` becomes `B = mu H`, so
`a = n.mu.n`, `p = mu n`, `q = mu^T n`, and every expression after them is unchanged. A separate
derivation would have been the same derivation.

What is genuinely new is two things that are easy to get wrong and quiet when you do.

**The slot order used to be inverted between the two sides, and the repair is worth recording.**
`pn_conv_matrices` returned `(x, y, xy, yx, z_i)` while `layer_modes_2d` takes `mu` as
`(x, y, z_i, xy, yx)` — the inverted z slot *last* there and *third* here. Passing one straight
into the other put `mu_zz^-1` where `mu_xy` belongs, which is a plausible magnetic layer rather
than an error, and `pn_mu_conv_matrices` existed to hold that reorder in one place.

Holding it in one place was not enough. The same mismatch bit a second time on the permittivity
side, through `solve_for_conv`, which a wrapper here could not reach: splatting a five-slot
builder into it returned `R + T = 1.61` on a lossless passive pillar. So on 2026-08-18 the order
was made one order everywhere — every builder now returns `solve_for_conv`'s own parameter order
— and the reorder inside `pn_mu_conv_matrices` went away. The function remains as the name for
the `mu` half of the duality; it rearranges nothing. `tests/test_slot_order.py` pins the
property for every builder, and the mismatch cannot be reintroduced in one place only.

**A layer's two tensors may have different boundaries.** The rearrangement is written around
*one* interface normal, so each tensor needs the normal field of its own boundary — a magnetic
inclusion need not sit where the dielectric one does. Sharing the permittivity's field is the
natural shortcut and passes every case where the two shapes coincide, which is most of them.
`_fmmax_patterned_mu_cases.different_shapes` is the case that does not.

(The normal field is read off `mu`'s in-plane trace. Reading `mu_zz` would give the *same*
field for any tensor whose components share a boundary — `normal_vector_field` reads where the
interface is, not what value sits either side — so a mutation of that line is equivalent, not
uncovered. Measured at exactly 0.)

### What checks it

`golden_fmmax_patterned_mu.npz`, frozen before `mu` had a builder. Six cases, judged by
convergence inside the band fmmax's two formulations leave (2.7e-5 to 1.3e-4); the solver lands
7e-6 to 5e-5 from the nearer, on the **vector** side in five of six — which is the assertion that
says the factorization reached the answer and not merely the code. A build feeding `mu` through
the direct rule unfactorized conserves energy, stays inside the band, and sits on the `FFT` side.

fmmax was gated on four questions first, the fourth being the duality: exchanging `eps` and `mu`
must trade the s and p channels, which it obeys on patterned tensors to 3e-14.

**One thing that is not true**, recorded because it is plausible enough to be re-asserted: the
three formulations do **not** bracket each other. An earlier test demanded the solver sit
componentwise between fmmax's two answers, on the reasoning that bracketing is the sharper
claim; three of six cases sit just outside, near the vector answer but past it. Three
formulations converging to one limit approach it from whatever side their own truncation puts
them on.

## 8. Reaching it

**Through the front door, since 2026-08-20.** A permeability travels with the material it
belongs to, so on the three *assignment* paths there is nothing extra to pass:

```python
structure = [dict(background=1.0, shapes=[
    dict(type='circle', cx=150., cy=150., lx=180.,
         material=wavebit.Material(12.25, mu=2.0))])]
solver = wavebit.rcwa(backend=0, structure=structure,
                      top=1.0, bot=wavebit.Material(2.25, mu=1.5), ...)
```

The raster is the exception, and it is the same asymmetry the material type is built around: a
raster is a sampled *field*, not an assignment, and a field holds one quantity, so its
permeability is a second array of the same shape.

```python
solver = wavebit.rcwa(backend=0, structure=eps_raster, mu=mu_raster, ...)
```

`mu=` is refused by name on the two shape paths rather than ignored — there is no region for a
layer-shaped array to attach itself to there — and a mismatched raster shape is refused too,
since the two fields are sampled on one grid.

What happens underneath is one sentence: `_conv_matrices` builds the permeability slots
whenever anything in the scene carries `mu != 1`, by running the **same builder a second
time** on a view of the structure with each material's `mu` where its `epsilon` was. That is
why the permeability slots are the E/H dual of the permittivity ones by construction rather
than by a second derivation, and why the two quantities cannot land on different geometry —
the block path replays the same grid plans, the shape paths the same boundary plans. A
non-magnetic scene builds nothing and takes the arithmetic that existed before `mu` did, bit
for bit.

A `mu` the chosen path cannot express is refused by name, never dropped: the 1D and conical
operators carry no permeability slot at all, and the first-order (4ff) solve a z-coupled `eps`
routes to does not implement one. A knob the solve silently ignored is the failure this
repository refuses elsewhere too.

**By hand.** `mu` is also one keyword on `solve_for_conv`, taking the five stacks in the
permittivity's convention:

```python
mu = (mux_all, muy_all, muz_i_all, muxy_all, muyx_all)      # any entry may be None
res = mee.solve_for_conv(wl, epx, epy, epz_i, epxy, epyx, mu=mu)
```

Each entry is `(n_layers, ff_xy, ff_xy)`, so a stack with `mu` on some layers and not others is
expressible — the layers without it take the original arithmetic. `mux` / `muy` fall back to the
identity and the other three to nothing.

A *tensor* `mu` still has no front door, exactly as a tensor `eps` has none: `structure` is a
scalar quantity per region or per pixel. A patterned magnetic tensor goes through
`to_conv_mat_raster_pn_mu`; see §7.

## 9. What is not implemented, and why

**1D and conical paths.** No permeability slot in either operator. Refused, not ignored.

**Magnetic `mu` with a z-coupled `eps`.** That combination routes to the first-order 4ff solve,
which has no `mu`. Both features work and the combination refuses, which is the point — landing
on whichever path the dispatch picks and silently dropping the other input is the failure mode.



## 10. References

- fmmax's `eigensolve_general_anisotropic_media` — five `eps` and five `mu` components, no z row;
  the same 2ff tensor class this implements, which is why it can arbitrate this and not a
  magnetic Voigt geometry.
- [`formulation.md`](formulation.md) §4 — the non-magnetic first-order system this extends, in
  the code's own convention.
