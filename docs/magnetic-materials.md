# Magnetic materials — `mu` on layers and half-spaces

Landed 2026-08-14. [`anisotropic-materials.md`](anisotropic-materials.md) §5 had scoped this and
got the location right and the content wrong; §3 below is the difference.

Target: `wavebit/_core/solve_2d.py` — `layer_modes_2d`, `solve_2d`;
`wavebit/_core/scattering.py` — `smm_2d`; `wavebit/_core/fields.py` — `_field_coupled`.

**Scope.** Layers (§1–§4), half-spaces (§5–§6), patterned layers (§7) and field
reconstruction (§9, added 2026-08-20) on the **2D path**; the **1D and conical paths** since
2026-09-01 (§11), diagonal slots only. The first-order solve still takes no `mu` and
refuses rather than ignores, as do the two off-diagonal slots on the 1D paths.

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
`solve_conv` holding only an index — going through a square root and back loses digits the
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

**A negative-index half-space, on the far side.** When both real parts are negative and the
medium propagates (`Re(eps*mu) > 0`), `eps*mu` looks like an ordinary forward medium, and
Maystre's root advances its phase away from the interface, the wrong way for this medium.
Glass onto `eps = -2+0.1i`, `mu = -1+0.1i` returned `R = 929` against a true `0.0011`, and the
closed form above is how that was measured; such a half-space was refused on 2026-10-03.
The kernels now hand `boundary_kz` the half-space's own `(eps, mu)` whenever the scene is
magnetic, and a double negative takes the mirror of Maystre's cut, the decaying root
([`branch-cuts.md`](branch-cuts.md) §9): the limiting-absorption root with the loss on `eps` and
`mu` separately, which M. Cassier, C. Hazard and P. Joly (Commun. PDE 42, 1707 (2017)) prove for
a dispersive double-negative half-space. Its index is `n = sqrt(eps)*sqrt(mu)`, each factor's
root on Maystre's cut (the principal root for a passive factor), the negative of the principal
`sqrt(eps*mu)`, so `n/mu`, the admittance, keeps a positive real part and `t_p = t_s` at normal
incidence. Measured against the closed-form interface (s and p, `r` and
`t`, 0 to 80 degrees, both sides, every path and both connecting algorithms) to `3.2e-15`, and
against a thick lossy slab of the medium -- a layer keeps both roots, so that comparison depends
on no root choice -- to `4e-16` in every order's `R` (`tests/test_negative_index_half_space.py`;
the numbers are `studies/negative_index_media.py`'s), and against fmmax per order
(`tests/references/test_fmmax_dng_golden.py`). **Lit** from such a
medium it is solved too: `theta` names the light's direction and its in-plane wavevector points
the other way ([`negative-index-media.md`](negative-index-media.md) §7). Still refused: an
active one that propagates (where it is set), and two lossless half-spaces that are each
other's negation, `(eps, mu)` and `(-eps, -mu)`, which have no time-harmonic solution on an
evanescent order -- refused wherever the truncation retains one
([`negative-index-media.md`](negative-index-media.md) §9). A pair too lossy to propagate
(`Re(eps*mu) <= 0`, such as `eps = mu = -1+1.2i`) takes the same root under both rules, so its
`R` and `T` are unchanged; its index is now negative too, which flips its `t_p` (exactly).

**A negative-index layer.** The same sign question has a layer form. A layer's modes come as
`+-q` pairs, and ETM has to call the root the region below receives "down"; Maystre's root does
that for an ordinary layer and does the opposite for a double negative, so a double-negative
layer resting on its own medium made ETM's interface factor singular. A uniform double negative
now takes the mirror of the rule, as its half-space does, and a patterned layer holding one is
labelled per mode by its power flux ([`connecting-algorithms.md`](connecting-algorithms.md) §11).

**A magnetic metal is a negative-index medium too.** A passive magnetic medium can multiply to
`Im(eps*mu) < 0` with one real part negative: `eps = -2+0.01i`, `mu = 0.1+0.5i`. Its principal
root `0.638-0.783i` reads as gain, and its index is the negative of it,
`sqrt(eps)*sqrt(mu) = -0.638+0.783i`: it is a negative-phase-velocity medium although not a
double negative. No `kz` is decided on that `n`; every `kz` comes from `n**2`, and `R` and `T`
match the closed form either way. The public `n_top` / `n_bot`, a complex angle's formal `kx`,
a real angle's `k_par` from it, and the sign of a transmitted `t_p` read it. Until 2026-10-04
the index was the principal root, and that medium's `t_p` was `-t_s` at normal incidence; it is
`t_s` now (`0.8319-0.4608i` for both, glass onto it), as for every other medium.
[`negative-index-media.md`](negative-index-media.md) §6 tabulates which media are negated.

**Reciprocity** for the backside mirror, where `mu_top` and `mu_bot` must swap (the kernels'
backside: public bottom incidence when this was written, public top incidence since
2026-09-21; comparing the two sides reaches it either way). Transmittance is
direction-independent for a lossless reciprocal stack — but the **angle has to be converted**,
because `theta` is stated against the incidence medium's normal. Comparing top-at-33° with
bottom-at-33° puts `T` out by 9e-3 and reads exactly like a broken mirror; what reciprocity
equates is one `k_par` seen from both sides, `n_top sin(th_top) = n_bot sin(th_bot)`.

### What a mutation survey added

Two mutants of the incident column survived the first survey, and measuring them showed they
changed *nothing* — they were not redundant code, they were in the copy of that column the
tests never reached. Every magnetic assertion at that point read `s_inc` / `p_inc`,
built from a different block than `incident`, the psi-specified incidence. The gap was not subtle:
the psi path returns `R + T = 1.575` with the incident s-flux left non-magnetic. Deleting the
backside swap survived everything too. With the psi and reciprocity tests added, 15 of 15
mutants die. (Since 2026-09-21 the swap sits on the public top-incidence path, where every
top-lit case reaches it: deleting the 2D ETM swap now fails 16 tests in
`tests/references/test_fmmax_mu_ambient_golden.py`, not only the reciprocity ones.)

## 7. Patterned `mu`

**No new factorization was needed**, and that is the finding rather than a convenience.
`pn_eps_conv_matrices` — the Popov-Nevière rearrangement written for a patterned tensor `eps`
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
`pn_eps_conv_matrices` returned `(x, y, xy, yx, z_i)` while `layer_modes_2d` takes `mu` as
`(x, y, z_i, xy, yx)` — the inverted z slot *last* there and *third* here. Passing one straight
into the other put `mu_zz^-1` where `mu_xy` belongs, which is a plausible magnetic layer rather
than an error, and `pn_mu_conv_matrices` existed to hold that reorder in one place.

Holding it in one place was not enough. The same mismatch bit a second time on the permittivity
side, through `solve_conv`, which a wrapper here could not reach: splatting a five-slot
builder into it returned `R + T = 1.61` on a lossless passive pillar. So on 2026-08-18 the order
was made one order everywhere — every builder now returns `solve_conv`'s own parameter order
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
    dict(type='circle', cx=0., cy=0., lx=180.,
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

What happens underneath is one sentence: `_eps_conv_matrices` builds the permeability slots
whenever anything in the scene carries `mu != 1`, by running the **same builder a second
time** on a view of the structure with each material's `mu` where its `epsilon` was. That is
why the permeability slots are the E/H dual of the permittivity ones by construction rather
than by a second derivation, and why the two quantities cannot land on different geometry —
the shape path replays the same boundary plans. A
non-magnetic scene builds nothing and takes the arithmetic that existed before `mu` did, bit
for bit.

Full tensor epsilon and mu use the first-order path when either has xz/yz/zx/zy
coupling. Both longitudinal fields are reconstructed from their full z rows.
The separate 1D operators carry diagonal mu; tensors use the coupled path.
See [full tensor patterns](full-tensor-patterns.md).

**By hand.** `mu` is also one keyword on `solve_conv`, taking the five stacks in the
permittivity's convention, about the cell centre as `mu_conv_matrices()` returns them:

```python
mu = (mux_conv_all, muy_conv_all, muz_conv_i_all,
      muxy_conv_all, muyx_conv_all)                 # any entry may be None
res = mee.solve_conv(epx_conv_all, epy_conv_all, epz_conv_i_all,
                     epxy_conv_all, epyx_conv_all, mu=mu)
```

Slot for slot, the two tuples are now spelled the same way, which is the point: `mu` is
convolved by the *same* builder as `eps` (§7), so the `_conv` in the name is as true of one as
of the other. It was `(mux, muy, muz_i, ...)` until 2026-08-20, sitting beside `epz_conv_i` in
the same expressions, and read as though the permeability skipped the factorization.

Each entry is `(n_layers, n_orders, n_orders)`, so a stack with `mu` on some layers and not others is
expressible — the layers without it take the original arithmetic. `mux_conv` / `muy_conv`
fall back to the identity and the other three to nothing.

Use `Material(epsilon, mu)` in shape/background assignments, or a tensor `mu`
raster beside `structure`. Full tensors produce nine slots; pass the returned
permeability tuple as `mu=solver.mu_conv_matrices()` to `solve_conv`.

## 9. Field reconstruction

Added 2026-08-20, and it is the part of this feature that shipped wrong for six days.

`R` and `T` were right from the start; so were `E`, and the transverse `H`, inside the cell.
What was wrong was `H_z`, and only `H_z`. The transverse field is what the solve hands over —
it comes out of mode matrices that do account for `mu` — but the two **longitudinal**
components are not propagated at all. They are recovered algebraically at the end, each by
dividing out the z slot of its own tensor:

```
eps_zz E_z  =  Ky Hx - Kx Hy          Ampere's z row      ->  epz_conv_i
mu_zz  H_z  =  Kx Ey - Ky Ex          Faraday's z row     ->  muz_conv_i
```

One line is the other under the same `(E, H, eps, mu) -> (H, -E, mu, eps)` duality §1 places
`mu` by. `_field_coupled` carried the first and, for the second, the `mu = 1` form
`H_z = Kx Ey - Ky Ex`, with nothing marking it as conditional. The condition was written down
one function away — `field_2d_first_order`'s docstring said `H_z` "is unchanged while `mu = 1`,
which is all this path admits" — and that sentence was true of the path it was written on and false
of the one beside it, with no refusal anywhere making either claim hold. Before the front door
(`Material(epsilon, mu)`, and `mu=` beside a raster) the combination was barely reachable;
after it, it was ordinary user input.

**Why nothing caught it.** A magnetic layer's `H_z` came back scaled by `mu_zz` while every
other component stayed exact. `R + T` was still 1, the duality of §4 still closed, both
connecting algorithms still agreed, and fmmax still matched — none of those read the interior
`H_z`. The proximate gap is simpler than any of that: `tests/test_magnetic_layer.py` never
called field reconstruction at all.

**What pins it now**, in `tests/test_magnetic_layer.py`:

- a **closed form**. A uniform magnetic slab's s-polarized interior field comes out of a
  characteristic-matrix recursion in `(Ey, Hx)` with the admittance `kz/mu` — no wavebit
  anywhere in it, so it cannot be co-migrated the way a regenerated golden can. The comparison
  is absolute: unit incident amplitude, no fitted scale or phase. The oracle measures depth down
  from the top of the stack; the test asks the solver for the matching heights from the bottom,
  `sum(thickness) - depth`, and applies the z-mirror parity to its cell. It reaches `mu_zz` as a
  *scalar*.
- the **duality, on the cell**. A patterned `mu_zz^-1` is a convolution matrix, and omitting it
  is a different mistake from applying it on the wrong side. Exchanging the two patterns and
  trading the incidence maps `E` onto `H` entry for entry with unit scale, which checks the
  patterned `muz_conv_i` in `H_z` against the patterned `epz_conv_i` in `E_z` — a line that
  predates permeability and is pinned from outside by `test_fmmax_field_golden.py`. This is
  the one that catches an operand-order error; the uniform closed form cannot, because a
  scalar commutes.

`muz_conv_i` reaches the reconstruction as an eighth slot on each `layer_info_list` entry, beside
the `epz_conv_i` it is dual to, `None` where the layer is non-magnetic. Both connecting
algorithms build that list by their own route, so the slot had to be added twice —
`solve_2d` and `_smm_sp` — and a stack whose `H_z` was magnetic under `'ETM'` and
non-magnetic under `'SMM'` would have had identical `R` and `T` under both.

Two consequences of it being per-layer state rather than a solver attribute. It has to travel
with its layer through the **backside** reversal — the propagation kernels' backside, which
since 2026-09-21 serves public *top* incidence — and the two algorithms reverse by different
means — the ETM mirrors the scene before solving, the SMM reverses the state it kept — so a
`muz_conv_i` left out of either gives each layer the other's `mu_zz`: still magnetic, still
energy-conserving, `R` and `T` untouched. And the slot is *appended*, not inserted, because the
legacy goldens name the first seven and `fields.py` unpacks both algorithms' records by the same
positions (a ninth slot, after it, holds the SMM's per-layer amplitudes); `_z_plan`'s thickness
lookup, which used to infer the layout from the entry's length, now takes the index from its
caller instead.

## 10. What is not implemented, and why

**Magnetic `mu` on the first-order solve.** No `mu` in the operator, and
`fields.field_2d_first_order`'s `H_z = Kx Ey - Ky Ex` likewise has no `mu_zz` in it. Two ways to
land there — a z-coupled `eps`, or `formulation='first_order'` asked for directly — and the refusal
names which one happened, because blaming a z row that was never given sends the reader to look for
one. Both features work and the combination refuses, which is the point: landing on whichever
path the dispatch picks and silently dropping the other input is the failure mode. Should `mu`
ever land there it needs the `muz_conv_i` factor *and* the two z-row terms `mu_zx H_x` /
`mu_zy H_y` that formulation would then be the one able to express — exactly as its `E_z`
already keeps `ezx` / `ezy`.



## 11. The 1D and conical paths

Landed 2026-09-01. §10 used to carry these as "no permeability slot in either operator,
refused not ignored"; this is what replaced that.

### What the operators became

The duality of §1 does the placing here too, and on this path it lands somewhere sharper: it
carries the **TE branch onto the TM branch**. Eliminating each polarization's longitudinal
component through the `zz` slot of its own tensor,

```
TE:  d2(E_y)/dz~2 = mu_x  (Kx muz_i Kx - eps_y) E_y        V = mu_x^-1  W q
TM:  d2(H_y)/dz~2 = eps_x (Kx epz_i Kx - mu_y ) H_y        V = eps_x^-1 W q
```

and each line is the other with every symbol replaced. At `mu = 1` the TE prefactor is the
identity and drops out — which is exactly why that branch had no left factor and no inverse in
`V`, while the TM branch already carried both and its `- I` had been `- mu_y` all along.

**A magnetic TE layer therefore costs what a TM layer always cost**, one matmul and one
inverse: measured 0.70 ms against 0.78 ms at `truncation_order=(15, 0)`, four layers. That is
the whole arithmetic price. What this change *removes* is larger: a magnetic 1D scene had to be
routed through the 2D path, 3.01 ms against 0.78 ms on the same problem.

The conical operator is the same two lines with a `ky**2` shift and a right-multiplication
rather than a left one, because its mode matrices expand `H_x` and `E_x` rather than `E_y` and
`H_y`:

```
family 1 (H_x):  ky**2 + (Kx muz_i Kx - eps_y) mu_x
family 2 (E_x):  ky**2 + (Kx epz_i Kx - mu_y ) eps_x
```

and its `V` blocks split by the same duality — the **E-side** pair `V_11` / `V_12` takes the
`mu`, the **H-side** pair `V_21` / `V_22` is untouched. `V_21` already carried the `epz_conv_i`
that `V_12` now mirrors with `muz_conv_i`; the two lines had been each other's reflection since
before there was a permeability to put in one of them.

### The decoupling premise, which had to be checked rather than assumed

The conical path solves two `n`-sized eigenproblems where the 2D path solves one `2n` one. That
split is not free: written out, the coupled `Omega_L @ Omega_R` has off-diagonal blocks
proportional to `Kx muz_i mu_y - eps_y epz_i Kx`, so it needs `eps_y = eps_z` **and**
`mu_y = mu_z` — as matrices, not as materials.

Both hold exactly on a 1D grating, for one reason: `E_y`, `E_z`, `H_y` and `H_z` are all
*tangential* to a boundary perpendicular to x, so all four slots are built by the direct
(Laurent) rule from one profile and only the `x` slots take the inverse rule. Measured on a
random 1D grating at `truncation_order=5`: `|epy - inv(epz_i)| = 9e-16` against a matrix of
scale 1.9, while `|epx - inv(epz_i)| = 0.16`. The permeability inherits it because the same
builder runs a second time (§8), so a scene that breaks the premise for one tensor breaks it
for both.

**The premise is older than this feature and is not new risk.** Fed convolution matrices that
violate it, the conical path already disagreed with the 2D path by 6.8e-4 *without any* `mu`.
That control is what identified the 2.6e-4 magnetic disagreement as the same pre-existing
effect rather than a defect in the new operator: with the premise held, magnetic conical and 2D
agree to 4.4e-16.

### The reconstruction, which is the reason this was not a small change

Lifting the refusal makes an existing, correct line wrong. `field_1d`'s TE branch was
`Mz = 1j * Kx @ My` — Faraday's z row at `mu = 1` — and §10 said in as many words that it was
"correct because that refusal holds". It now divides by `muz_conv_i`, the dual of the
`epz_conv_i` the TM branch always had:

```
TE:  mu_zz  H_z = Kx E_y      Faraday's z row
TM:  eps_zz E_z = Kx H_y      Ampere's z row
```

The conical path had the same hole in a different shape: its `layer_info_list` appended a
**hard-coded `None`** in the `muz_conv_i` slot, with a comment saying it "structurally must
be". Both are the 2D path's 2026-08-20 defect waiting to happen — `H_z` returned scaled by
`mu_zz` while every other component stays exact, `R + T` still 1, both connecting algorithms
still agreeing.

### What checks it

`tests/test_magnetic_1d.py`, and its module docstring ranks the four instruments. The duality
(3.3e-16) is first because it is a theorem across two separately written branches. A closed-form
uniform slab is second, carrying the two cases an index-only solver cannot express at all:
`eps = 2, mu = 2` has the vacuum *impedance* with a non-vacuum index and must reflect **0**
where an index-only solver says 0.111, and `eps = 0.5, mu = 2` has the vacuum *index* with a
non-vacuum impedance and reflects 0.346 where an index-only solver sees no interface and says
0. The 2D path is third and is the only instrument that reaches the conical coupling blocks.
The two connecting algorithms are fourth, compared **on the reconstructed field as well as the
efficiencies** — `muz_conv_i` is per-layer state each assembles by its own route, so an
efficiency-only comparison cannot see one of them dropping it.

**Energy conservation is deliberately absent from that list**, for the reason §5 gives.

### Two things the mutation survey found

Sixteen mutants, one per new term. Three survived the first run — the conical `muy_conv`, the
conical `V_12`'s `muz_conv_i`, and the SMM's field slot — and none of the three was a code
defect.

The first two had one cause: **the oracles ran at `phi = 0`**, where `ky = 0`, the conical
problem decouples exactly, and every block carrying a `ky` factor or belonging to the second
family is unreachable. A conical test at zero azimuth is not exercising the conical path. Moved
off-axis, both mutants die.

The third was a genuine coverage gap of the kind §9 warns about: the comparison between the two
connecting algorithms read efficiencies only, and the `muz_conv_i` each carries for `H_z` is
invisible to those. With a field comparison added, all sixteen die.

### The half-spaces, and one asymmetry worth stating

`kz/mu` on the electric amplitude (TE) and `kz/eps` on the magnetic one (TM), the same pair §5
places on the 2D boundary — `sp_efficiencies` had been spelling both correctly since 2026-08-14
against callers that could not yet reach it.

**Reflection needs no permeability and transmission needs two**, which looks like a dropped term
and is not. A reflected order sits in the *incidence* medium, so it is measured against that
medium's own incident admittance and the slot cancels algebraically **inside the `Re`** —
exactly, for a complex `mu_top` as much as a real one. A transmitted order sits in the far
half-space, so its slot and the superstrate's both survive.

### Non-magnetic solves are untouched, and it was measured rather than asserted

Every branch is skipped rather than multiplied by an identity, so a non-magnetic 1D solve runs
the arithmetic it ran before. Checked by byte comparison against `git archive` of the parent
commit: ETM and SMM, top and bottom incidence, TE / TM / conical, two truncation orders,
`R` / `T` / `r_s` / `t_p` and a reconstructed field cell — 120 arrays, 232 KB, **one sha256**.

The digest was validated before being believed: it is stable across two runs of one tree, it
moves under one ulp of wavelength, and it moves when the same probe is given a `mu` — the last
being what says the probe reaches the changed code at all, so the equality is a result rather
than a probe that misses.

## 12. References

- fmmax's `eigensolve_general_anisotropic_media` — five `eps` and five `mu` components, no z row;
  the same second-order tensor class this implements, which is why it can arbitrate this and not a
  magnetic Voigt geometry.
- [`formulation.md`](formulation.md) §4 — the non-magnetic first-order system this extends, in
  the code's own convention.
