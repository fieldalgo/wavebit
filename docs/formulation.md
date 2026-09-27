# The formulation, derived in the code's own convention

Written: 2026-08-11.

Targets:
- `wavebit/_core/solve_1d.py` — `layer_modes_1d`, `layer_modes_1d_conical`, `solve_1d`,
  `final_matching`
- `wavebit/_core/solve_2d.py` — `layer_modes_2d`
- `wavebit/_core/factorization.py` — `_circulant_index`, `fourier2d_piecewise_constant`
- `wavebit/_core/conventions.py` — every square root below
- the paper draft (`e^{+jωt}`), via the mapping table in §10

---

## TL;DR

Wavebit's algebra was originally written against a paper draft in the engineering
convention `e^{+jωt}` and later flipped to `e^{-iωt}` to match RETICOLO. The flip is complex
conjugation of the entire problem, and doing it equation-by-equation is exactly how sign
errors breed — three shipped as gain-media bugs, one as a branch-cut defect
(`docs/time-convention-audit.md`). This document removes the translation step: **the full
derivation, from Maxwell's equations to the diffraction efficiencies, natively in
`e^{-iωt}`**, with every intermediate quantity named after the variable that holds it in
`_core/`. Each boxed result was checked against the code the day this was written — the
check is stated next to the result.

Conventions, all of them, up front:

| quantity | convention |
| --- | --- |
| time factor | `e^{-iωt}`; a field is `Re[E(r) e^{-iωt}]` |
| loss | `Im(n) > 0`, `Im(eps) > 0` (`n + ik` as databases tabulate) |
| upward (+z) wave | `exp(+i k0 kz z)`, `Re(kz) > 0` |
| decay along +z | `Im(kz) > 0` |
| plane-wave phase | `exp(+i k0 (kx x + ky y + kz z))` |
| Fourier series of `eps` | `eps(x) = Σ_p c_p exp(+i 2π p x / Λ)` |
| order vectors | `kx_m = n_I sinθ cosφ + m λ/Λx`, `m = -M..M` (same for `ky_n`) |
| normalized H | `H̃ = Z0 H`, `Z0 = sqrt(μ0/ε0)` |
| partner variable | `U = +i H̃_t` for an E-primary, `Q = -i E_t` for an H-primary |

Everything below is unit-free: lengths are normalized by `k0` (`z̃ = k0 z`), so `d/dz̃`
replaces `d/dz` and wavevectors are in units of `k0`.

**Which way is down.** The derivation keeps the propagation kernels' own words:
"superstrate" is the kernel's entry medium, `z̃ = 0` its entry face, and "down" is `+z̃`,
into the stack. Since 2026-09-21 the kernels' axis *is* the public `+z`, which points up,
and their entry port is bound to the public **bottom** medium
(`_core/engine.py`, `_kernel_incidence_side`); public top incidence runs the kernels' mirrored
path. So read "down" as `+z̃` and "superstrate" as the lit medium of that unmirrored solve,
not as the public words. The algebra did not change with the public frame; only the names
around it did.

---

## 1. Maxwell's equations under `e^{-iωt}`

With `∂/∂t → -iω` on `Re[E e^{-iωt}]`:

```
∇ × E = -∂B/∂t = +iω μ0 H
∇ × H = +∂D/∂t = -iω ε0 ε E
```

Substituting `H̃ = Z0 H` and `k0 = ω/c`:

```
∇ × E  = +i k0 H̃                                    (M1)
∇ × H̃  = -i k0 ε E                                  (M2)
```

Sanity anchor: a vacuum plane wave `E = ŷ exp(i k0 z)`, i.e. travelling `+z`. (M1) gives
`H̃x = -exp(i k0 z)`, and the time-averaged Poynting flux
`S_z = ½ Re(E × H̃*)_z = ½ Re(-Ey H̃x*) = +½` — energy flows `+z`. The pair
`(Ey, H̃x) = (1, -kz)` is what `tests/test_anisotropic_berreman.py::_vacuum_modes` writes,
which is the code-side check on (M1)/(M2) themselves.

**`reconstruct_field` returns these same variables**, so this anchor is a statement about its
output too, not only about the solve. That was worth leaving unsaid until it wasn't: this
document had no reference to the reconstruction, the reconstruction had none to it, and from
2026-08-14 back to the rewrite it returned `E_z`, `H̃x` and `H̃y` negated — a `+z` vacuum wave
came back with `H̃x = +1` and its energy flowing `-z`, contradicting the anchor above while
every check in the suite stayed green. See `docs/conventions.md` and
`tests/test_field_core.py::test_every_path_returns_a_maxwell_field_in_the_declared_frame`.

Loss check: `eps = 1 + iδ`, `kz = sqrt(eps) ≈ 1 + iδ/2`, so `exp(i k0 kz z)` **decays**
toward `+z`. `Im(eps) > 0` is absorption under this sign of (M1)/(M2) — the two statements
are one convention, not two.

---

## 2. Fourier machinery

### 2.1 Orders and the coupling sign

A field in a periodic layer is a Bloch sum over orders `m` (1D shown; `y` is identical):

```
Ey(x̃, z̃) = Σ_m  S_m(z̃) exp(+i kx_m x̃),      kx_m = n_I sinθ cosφ + m λ/Λx
```

with `m` running `-M..M` — the sign of `m` here and the `+i` in the phase are one choice.
Code: `_separable_kxy` builds exactly this vector (`max_order_x_range = arange(-M, M+1)`).

The permittivity expands in the **same** sign:

```
eps(x) = Σ_p c_p exp(+i 2π p x/Λx)
```

so that `eps · E` couples order `n` into order `m` through `c_{m-n}`. The convolution
("Toeplitz") matrix is therefore

```
[E]_{m,n} = c_{m-n}                                  (F1)
```

Code: `_circulant_index` returns `(r - c)`, and its docstring records why this sign is a
trap: pairing (F1) with the opposite-sign order vector **mirrors the diffraction orders
while conserving energy**, so only a grating oracle (RETICOLO) can catch it. Nothing in
this document changes that; it is stated so the derivation carries the sign explicitly.

### 2.2 Factorization rules

Products of discontinuous factors truncate badly (Li). The rules, per field component of a
layer varying in `x`:

| component | continuity across the `x`-jump | rule | code matrix |
| --- | --- | --- | --- |
| `Ey`, `Ez`-free products | continuous | Laurent (direct) | `epy_conv` |
| `Ex` | discontinuous (`Dx` continuous) | inverse rule | `epx_conv` |
| `Ez` via `Dz = -Kx H̃y`-type eliminations | — | inverse of `eps`-conv | `epz_conv_i` |

`factorization='laurent'` puts the direct rule in the `Ex` row too -- `[[eps]]` in both in-plane
slots, `epz_conv_i` unchanged -- which is the formulation this table corrects; see
`docs/factorization-terms.md` §9.

For 2D-patterned layers the same logic produces the five slots
`(epx, epy, epz_i, epxy, epyx)`; the off-diagonal slot convention (`-eps_xy` in the slot)
is documented in `docs/off-diagonal-slot-convention.md` and is not re-derived here. The
`eps` these builders read is the layer material **as written** — since 2026-08-20 the front
door takes relative permittivity, so there is no `n → eps` map left at
`factorization._layer_eps`, `parametric.assemble_eps_coeffs` or `boundary.conv_mats`, and
there is **no conjugation** either (the last of the three held the conjugate until
2026-08-11; `docs/time-convention-audit.md` §6). Loss is `Im(eps) > 0`, as written, in
`e^{-iωt}`.

---

## 3. One 1D layer: TE and TM

Geometry: `eps = eps(x)`, no `y` variation, `ky = 0`. In Fourier space `∂x̃ → i Kx`
(diagonal matrix of `kx_m`), `∂z → d/dz̃`.

### 3.1 TE (`tetm_index = 0`)

Fields: `(Ey, H̃x, H̃z)`. Primary `S = Ey` (vector over orders). From (M1):

```
-S' = i H̃x            (x-row)          H̃z = Kx S          (z-row)
```

Define the partner `U ≡ +i H̃x = -S'`. From (M2)'s `y`-row,
`H̃x' - i Kx H̃z = -i εy S` with `εy =` `epy_conv`:

```
S'' = (Kx² - εy) S  ≡  A S                            (TE1)
```

Eigendecompose `A = W diag(q²) W⁻¹`; take `q` with `Re(q) ≥ 0` (`mode_q`; the label is
free — `docs/time-convention-audit.md` §5 — and this one keeps `exp(-q d̃)` bounded). The
general solution and its partner:

```
S(z̃) = W [ e^{-q z̃} c⁺ + e^{-q(d̃-z̃)} c⁻ ]
U(z̃) = V [ e^{-q z̃} c⁺ - e^{-q(d̃-z̃)} c⁻ ],        V = W q          (TE2)
```

(`U = -S'` applied termwise.) **Code check**: `layer_modes_1d`, `tetm_index == 0`:
`A = diag(kx)² - epy_conv`, `V = W * q`. Verbatim.

A downward wave in a homogeneous half-space, `S = t exp(+i kz z̃)`, has
`U = -i kz S` — the substrate seed

```
G = -i Kz_bot                                          (TE3)
```

(code: `G = -1j * _diag(xp, kz_bot)`), and the homogeneous identification is `q = -i kz`.

### 3.2 TM (`tetm_index = 1`)

Fields: `(H̃y, Ex, Ez)`. Primary `P = H̃y`. From (M2):

```
P' = i εx Ex     →  Ex = -i εx⁻¹ P'          (x-row, εx = epx_conv, inverse rule)
i Kx P = -i εz Ez →  Ez = -εz⁻¹ Kx P          (z-row, εz⁻¹ = epz_conv_i)
```

Substituting both into (M1)'s `y`-row `Ex' - i Kx Ez = i P`:

```
P'' = εx (Kx εz⁻¹ Kx - I) P  ≡  εx B P                (TM1)
```

Partner `Q ≡ -i Ex = -εx⁻¹ P'`, so for `P = W e^{-q z̃} c⁺`:

```
V = εx⁻¹ W q                                           (TM2)
```

and the substrate seed from `P = t e^{+i kz z̃}`: `Q = -i (kz/εbot) P`, i.e.

```
G = -i Kz_bot / eps_bot                                (TM3)
```

**Code check**: `layer_modes_1d`, `tetm_index == 1`: `B = kx εz⁻¹ kx - I`, `eig(epx_conv @ B)`,
`V = inv(epx_conv) @ W * q`; `solve_1d` stage 1: `G = -1j * diag(kz_bot / eps_bot)`.
Verbatim, including which side `εx` multiplies.

Note the symmetry that makes the two cases one pattern: primary `(S | P)`, partner
`(+i H̃x | -i Ex)`, and in both the partner equals `-(mass matrix)⁻¹ d(primary)/dz̃` with
mass `I` for TE and `εx` for TM. That symmetry is what the conical and 2D systems block
together.

---

## 4. One 2D layer: the first-order system

Geometry: `eps = eps(x, y)`, both transverse field components coupled. Collect
`S = [Ex; Ey]` and `U = +i [H̃x; H̃y]` (each block a vector over the `n_orders_v·n_orders_u` order
grid). Eliminate the longitudinal components using (M1)/(M2)'s `z`-rows:

```
H̃z = Kx Ey - Ky Ex
Ez  = -εz⁻¹ (Kx H̃y - Ky H̃x) = i εz⁻¹ (Kx Uy - Ky Ux)        (using H̃_t = -i U)
```

Substituting into the four remaining rows and writing `B = Kx εz⁻¹ Kx - I`,
`D = Ky εz⁻¹ Ky - I`, `C = Kx εz⁻¹ Ky`, `E = Ky εz⁻¹ Kx` (the code's `B, D, Cc, E`):

```
dS/dz̃ = -Ω_L U          Ω_L = [[ -C,      B     ],
                                [ -D,      E     ]]           (2D1)

dU/dz̃ = -Ω_R S          Ω_R = [[ -Kx Ky + εyx̂,   Kx² - εy ],
                                [ εx - Ky²,        Ky Kx - εxŷ ]]   (2D2)
```

(`εyx̂`/`εxŷ` are the off-diagonal slots; zero for isotropic layers.) Eliminating `U`:

```
d²S/dz̃² = Ω_L Ω_R S ≡ Ω²_LR S                          (2D3)
```

with, written out, `Ω²_LR = [[Ky² + B εx - C εyx̂, C εy - Kx Ky - B εxŷ],
[E εx - Ky Kx - D εyx̂, Kx² + D εy - E εxŷ]]` — the code's `Omega2_LR` in
`layer_modes_2d`, term for term. The mode matrix is `Ω²_LR = W diag(q²) W⁻¹` and the
partner follows from (2D2) applied to `S = W e^{-q z̃}`:

```
V = Ω_R W / q                                           (2D4)
```

**Code check**: `layer_modes_2d` computes exactly (2D3) and (2D4); the sign pattern of
`Ω_R`'s four blocks matches line for line. Isotropic sanity limit: scalar `eps`, plane
wave — `Ω²_LR` collapses to `(kx² + ky² - eps) I₂`, so `q² = k_par² - eps`, i.e.
`q = -i kz`. The off-diagonal ordering between `Ω²_LR` and `Ω_R` (which pairs `εyx̂` with
which column) is **not** free — see `docs/off-diagonal-slot-convention.md` §4.

Why `U = +i H̃_t` and not `H̃_t`: it makes every propagation factor real-positive
(`exp(-q z̃)`) and every admittance come out as `-i (…)`, so the `i`'s live in exactly two
places — the seeds (TE3)/(TM3) and the final match (§7) — instead of being smeared through
the recursion. It is the `e^{-iωt}` counterpart of Moharam's `-j` normalization.

---

## 5. The conical 1D layer

Geometry: `eps = eps(x)` again, but `ky = ky0 ≠ 0` (one order; `Ky = ky0 I` commutes with
everything). The 2D system (2D1)–(2D2) then decouples into two scalar-Helmholtz families:

```
family 1 (E-like):   d²S1/dz̃² = (Ky² + A) S1,      A = Kx² - εy       (C1)
family 2 (H-like):   d²S2/dz̃² = (Ky² + B εx) S2,   B = Kx εz⁻¹ Kx - I  (C2)
```

— at `ky0 = 0` these are (TE1) and (TM1). The partners do not decouple: each family's `U`
picks up a `Ky`-proportional admixture of the other, which is where the four coupling
blocks come from. Carrying (2D2) through with `Ky = ky0 I`:

```
V_11 = A⁻¹ W1 q1              V_12 = Ky A⁻¹ Kx W2
V_21 = Ky B⁻¹ Kx εz⁻¹ W1      V_22 = B⁻¹ W2 q2                    (C3)
```

**Code check**: `layer_modes_1d_conical` — `w1, W1 = eig(Ky² + A)`,
`w2, W2 = eig(Ky² + B @ epx_conv)`, and the four `V` blocks verbatim. (Note (C2) is
`B εx`, transposed order relative to 1D TM's `εx B`; the two are similar matrices —
same `q` — and the eigenvector convention difference is absorbed by the `V` blocks.
Bit-parity with the legacy chain and the RETICOLO conical sweep pin this jointly.)

The boundary works in the s/p basis, rotated per order by `varphi = atan(ky/kx)`
(`F_c = diag(cos varphi)`, `F_s = diag(sin varphi)`); `sp_modes_1d_conical` /
`sp_modes_2d` assemble `big_W` (mode → boundary primaries) and `big_V` (mode → boundary
partners) from (C3)/(2D4). The rotation is pure geometry and carries no convention
content beyond `varphi`'s sign, which is fixed by the `ky` order vector of §2.1.

---

## 6. Joining layers: the transfer recursion

State per interface: the pair `(F, G)` such that the field below is `S = F t`,
`U = G t` for the (yet unknown) transmission amplitudes `t`. Seed: the substrate,
`F = I`, `G` from (TE3)/(TM3) (or their block forms `big_F`, `big_G`).

Add one layer of thickness `d̃` on top. Continuity of `(S, U)` at the layer's **bottom**
interface, with the layer solution (TE2):

```
W (X c⁺ + c⁻) = F t          V (X c⁺ - c⁻) = G t,        X = e^{-q d̃}
```

Define

```
a = ½ (W⁻¹ F + V⁻¹ G)         b = ½ (W⁻¹ F - V⁻¹ G)
```

so `X c⁺ = a t` and `c⁻ = b t`. Reparametrize `t = a⁻¹ X t'` (this is the step that keeps
every exponential decaying — `X⁻¹` never appears):

```
c⁺ = t'          c⁻ = b a⁻¹ X t'
```

and the field at the layer's **top** interface becomes the new state:

```
F ← W (I + Y)        G ← V (I - Y)        Y = X b a⁻¹ X        T ← T a⁻¹ X    (J1)
```

`T` accumulates the reparametrizations so that the substrate amplitudes are `T t_top` at
the end. **Code check**: `solve_1d` stage 2 is (J1) symbol for symbol (`a_i = a⁻¹`); the
conical/2D loops are the block form. Nothing in (J1) contains a convention sign — every
`i` cancelled in `a` and `b` — which is why the recursion survived the convention flip
untouched. The seeds and the final match did not, which is where all four historical bugs
lived.

---

## 7. Closing the system: the incident wave

Superstrate, `z̃ < 0`: incident (down) plus reflected (up),

```
S(z̃) = δ e^{+i kz_inc z̃} + R e^{-i Kz_top z̃}
U(z̃) = -i kz_inc δ e^{+i kz_inc z̃} + i Kz_top R e^{-i Kz_top z̃}
```

`δ` is the incident column in the primary's Cartesian component (`E_y` for TE, `H_y` for
TM): `sigma_inc` on the incident order, or `sigma_j w_j` on each lit order under
`excitation`, where `sigma = sign(Re kx)` is `s_hat . y_hat` on the `ky == 0` row
(`order_frame_sign`). The reported amplitudes are `D R` and `D T T1` with `D = diag(sigma)`
(TM transmission after its impedance ratio), which puts every order in its own s/p frame,
the coupled paths' (`docs/amplitude-sp-frame.md` §6); the TM field state handed to
reconstruction is `T1 · n_inc/mu_inc`, a unit incident E. `kz_inc` **must be read off the same vector**
`Kz_top` is built from (`incident_kz`): the spelling `n_top cosθ` is an identity for real
`θ` and the wrong branch for complex `θ` (`docs/branch-cuts.md`). Continuity at `z̃ = 0`
against `(F T1, G T1)`:

```
δ + R = F T1
-i kz_inc δ + i Kz_top R = G T1
```

Eliminating `R`:

```
(G - i Kz_top F) T1 = -i Kz_top δ - i kz_inc δ                     (TE match)
(G - i (Kz_top/n_top²) F) T1 = -i (Kz_top/n_top²) δ - i (kz_inc/n_top²) δ   (TM match)
```

**Code check**: `solve_1d` stage 3, both branches, verbatim (`inc = -1j*kz_inc*δ`).
Then `R = F T1 - δ` and the substrate amplitudes are `T T1`.

The conical/2D final match (`final_matching`) is the same elimination in the s/p block
basis, solved as one `4·n_orders` system (`final_A`, `final_B`). The boundary variables it uses
are **definitions carried over from the legacy chain** — s carries the electric amplitude
(`E_s`), p carries the magnetic one (an incident p wave of unit E has `H̃ = n_top`), which
is why `final_B`'s rows read `(-sinψ, cosψ kz_inc/n_top, i sinψ kz_inc, i n_top cosψ)`.
The magnetic p variable becomes the Fresnel `r_p` through `/(-i n_top)`; since 2026-08-12
that conversion lives at the pack sites, so the public `r_p`/`t_p` are Fresnel-normalized
and the amplitude oracle compares them directly. The facts are pinned by
`test_reflection_amplitude` and `test_transmission_amplitude` across six path/algorithm
combinations, which is the honest statement of where the s/p sign conventions are
*checked* rather than *derived*.

---

## 8. Efficiencies

Time-averaged `z`-flux of one order, from `S_z = ½ Re(E × H̃*)_z`:

- TE order with electric amplitude `R_m`: `S_z ∝ |R_m|² Re(kz_m)`
- TM order with magnetic amplitude `R_m`: `S_z ∝ |R_m|² Re(kz_m / eps)`

Normalizing by the incident flux (`n_top cosθ` for unit incident E, real `θ`):

```
de_ri[m] = |R_m|² · Re(kz_top,m) / Re(n_top cosθ)
de_ti[m] = |T_m|² · Re(kz_bot,m) / Re(n_top cosθ)              (TE)
de_ti[m] = |T_m|² · Re(kz_bot,m / n_bot²) / Re(cosθ / n_top)   (TM)
```

**Code check**: `solve_1d` stage 3 and `sp_efficiencies`, verbatim. Two convention facts
worth stating: it is `Re(kz/eps)`, **not** `Re(conj(kz)/eps)` — the conjugate is a
leftover of the old formulation and returned negative transmitted power into gold
(`test_lossy_media` module docstring) — and for a complex `n_top` the incident-power
normalization is not convention-free, so `de_ri` is not compared against anything there
(the complex amplitude is; `tests/test_lossy_media.py`, amplitude oracle).

---

## 9. Square roots

Every `sqrt` above is a physical decision, centralized in `_core/conventions.py`:

- **layer** `q`: `Re(q) ≥ 0` (`mode_q`) — a conditioning choice on a free label;
- **half-space** `kz`: principal root (`Re > 0`, tie `Im > 0`) plus the branch-sheet test
  `_off_principal_branch` for the both-sides-absorbing regime.

The derivations and measurements are in `docs/time-convention-audit.md` and
`docs/branch-cuts.md`; nothing there is repeated here.

---

## 10. Mapping to the `e^{+jωt}` paper draft

A separate paper draft, kept outside this repository, is written in the engineering
convention. The two formulations are exact complex conjugates (`j ↔ -i`);
every physical magnitude is identical. For cross-reading:

| | paper (`e^{+jωt}`) | this document / the code (`e^{-iωt}`) |
| --- | --- | --- |
| Maxwell | `∇×E = -jωμ0H` | `∇×E = +ik0H̃` |
| +z wave | `exp(-j kz z̃)` | `exp(+i kz z̃)` |
| loss | `n - jκ`, `Im(eps) < 0` | `n + iκ`, `Im(eps) > 0` |
| decay in +z | `Im(kz) < 0` | `Im(kz) > 0` |
| partner normalization | `U = -j Z0 H_t` | `U = +i Z0 H_t` |
| layer root | `Re(q) ≥ 0` | `Re(q) ≥ 0` — **identical**, conjugation-invariant |
| complex `r`, `t`, fields | conjugates of the code's | |
| `\|r\|²`, efficiencies, `\|E\|` | identical | |

Practical consequences: efficiency and intensity figures computed with Wavebit drop
into the paper unchanged; any **complex amplitude or field phase** must be conjugated (or
the paper's convention statement changed); and the paper currently states its convention
nowhere — the single sentence "we adopt `e^{+jωt}`; a lossy medium has `Im(ε) < 0`" in
the theory chapter's first paragraph is the cheapest bug repellent this codebase's
history has to offer.

A LaTeX version of this derivation (paper-voice, no code references) is kept outside this
repository, ready to `\input` into the paper for the case where the paper is instead flipped
to match the code.
