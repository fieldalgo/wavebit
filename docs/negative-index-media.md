# Negative-index media — the signed index, the mirror root, and what `theta` means

Written: 2026-10-04. Measured on CPU (Apple Accelerate), NumPy complex128 unless stated, JAX and
PyTorch where named.

This page is meant to be read on its own, by someone who was not there when the decisions were
made. It explains what a negative-index medium is and why it is hard for a solver, how the
solver went wrong, the three rules that replaced the old behaviour and why each is the right
one, what the incidence angles mean, what is refused and why, how all of it was checked, what
other tools do, where it stops, how to use it, how the decisions were reached, and what
published work says about all of it.

**Every number on this page is printed by one script**, `studies/negative_index_media.py`, whose
output is committed as `studies/results/negative_index_media.log`. A number is tagged with the
section of that log it comes from, as *(study §3)*; section numbers in such a tag refer to the
log, not to this page. Run it from the repository root with
`python studies/negative_index_media.py --develop 0dfc4ce`. It takes about a minute and a half
on a laptop CPU and needs NumPy, JAX and PyTorch (sections 9 and 11 say so and skip themselves
if JAX or PyTorch is missing, and section 12 then runs on NumPy alone). The `--develop` part
extracts develop `0dfc4ce` with `git archive` and re-solves a harness there for §13's
bit-identity count; everything else runs without it, and `--only 3 4` runs single sections. A
rerun reproduces the log byte for byte on the same NumPy and BLAS (the log's second line names
the NumPy version, 2.5.1); on another BLAS the last digit of a round-off-sized number can differ.
Measurements this page does not quote but only points to -- the ETM/SMM tables of
[`connecting-algorithms.md`](connecting-algorithms.md) §11 -- live on that page. Two kinds of
statement are not the study's: PyMoosh's own outputs (§14), which it cannot rerun without
PyMoosh and which `tests/references/README.md` recorded when that golden was made; and what
published work says (§5, §18), each read in its source -- where a number illustrates such a
statement, the study's section 13 computes it.

Targets:
- `wavebit/_core/conventions.py` — `signed_index`, `boundary_kz`, `double_negative`,
  `incident_phase_index`, `gain_cutoff_band`, `mode_q`, `etm_mode_q`, `layer_label_rule`,
  `power_down_labels`, `order_azimuth`, `order_frame_sign`
- `wavebit/_core/engine.py` — `_half_space_index`, `_refuse_negative_index_half_space`,
  `_refuse_negative_index_pair`, `_perfect_lens`,
  `CoreFactorizationMixin._refuse_half_space_negation_pair`, `CoreSolveMixin._incident_kxy`,
  `_lit_index_is_negative`, `_frame_azimuth`, `_indices_signed_by_pairs`,
  `CoreFactorizationMixin._layer_double_negative`
- `wavebit/_core/domain.py` — `index_from_eps`; `wavebit/_core/backend.py` — `principal_sqrt`
- `wavebit/_core/illumination.py` — `_plane_basis`, the coherent door's ports
- `wavebit/_core/source_expansion.py` — `resolve_wave`, `gaussian_waves`
- `wavebit/_core/solve_2d_4ff.py` — `split_forward_backward`
- `studies/negative_index_media.py` — every number below
- `tests/test_negative_index_half_space.py`, `tests/test_negative_index_layers.py`,
  `tests/references/test_fmmax_dng_golden.py`, `tests/references/test_pymoosh_dng_golden.py`,
  `tests/test_material.py`

---

## TL;DR

* **Is `theta` the direction the light comes in? Yes.** `theta` and `phi` give the direction in
  which the incident light -- its power -- travels toward the structure, from the inward normal
  of the lit half-space. That is what they always meant, and in every ordinary medium nothing
  changes, because there the phase travels with the power. A negative-index medium is the one
  place where phase and power travel *opposite* ways, so there the same light has its in-plane
  wavevector pointing the other way:

  ```text
  k_par / k0 = s * N * sin(theta) * (cos(phi), sin(phi)),     s = sign(Re(n)) = -1
  ```

  In a lossless medium this is `kx = n sin(theta) cos(phi)` with the **signed** index `n`, which
  is Snell's law as Veselago and Pendry write it. Lit from `n = -sqrt(2)` at 20 degrees onto
  glass, the light leaves at **-18.8117 degrees**, on the same side of the normal it came from:
  negative refraction *(study §4)*.
* **A double negative** (`Re(eps) < 0` and `Re(mu) < 0`) is solved as the substrate, as the lit
  superstrate, as a uniform layer and as a region of a patterned layer, on every path, with ETM
  and SMM, on NumPy, JAX and PyTorch, under `jit` and autodiff.
* **Three rules make that possible.**
  1. The half-space root of a double negative is the *mirror* of Maystre's cut,
     `Im(kz) - Re(kz) > 0` (§5). Maystre's cut, right for every ordinary medium, picks the root
     that grows away from the interface for a double negative: glass onto
     `eps = -2+0.1i`, `mu = -1+0.1i` gave `R = 928.78` instead of `0.0010767`, with `R + T = 1`
     holding exactly, so no energy check could see it (§3).
  2. The index of a magnetic medium is `n = sqrt(eps) * sqrt(mu)`, each root on Maystre's cut --
     the principal root for every passive factor (§6). It is the old principal `sqrt(eps*mu)` to
     the bit for every ordinary medium, and its exact negation for the negative-phase-velocity
     media: every passive double negative and the passive magnetic metals.
  3. `theta` is the light's direction (§7), as above.
* **ETM's layer labels** follow where a mode's power goes in a double-negative layer (§9).
* **Refused:** an *active* double negative that propagates, because no rule at a single
  frequency can say which root is physical in it (Skaar 2006; Nistad and Skaar 2008). A
  `GaussianBeam` launched from a negative-index side is refused too (§10), and so are two
  lossless half-spaces that are each other's negation, `(eps, mu)` and `(-eps, -mu)`, wherever
  the truncation retains an evanescent order: on such an order the pair has no time-harmonic
  solution at all (§9).
* **Every medium with `s = +1` -- every non-magnetic and every ordinary magnetic one -- gives
  develop's numbers to the bit**, except in the places §13 names: 26,090 of 26,124 harness arrays
  are byte-identical to develop `0dfc4ce`, and the 34 that are not (the coherent door at exact
  normal incidence, and `jax.jit` over a traced magnetic half-space) move by at most 2.8e-15
  relative; two gradients at exactly `theta = 0` that were wrong on develop are now right; and
  some refusal texts were reworded. Media with `s = -1` -- double negatives and passive magnetic
  metals -- change on purpose, as §13 lists.
* **Checked against** closed-form Fresnel coefficients to 3.2e-15, a thick lossy slab (`R` per
  order to 4.0e-16, `T` to 1.7e-15), the lossless limit, unitarity, reciprocity, the Poynting
  vector of the solved field, the mirror medium of Lakhtakia's conjugation symmetry (bit for bit
  at a bare interface), fmmax 1.7.1 to 2.7e-14 .. 6.1e-12 per diffraction order for lossy double
  negatives, and PyMoosh 4.0.1 to 8.8e-15 for lossless ones in flat stacks (§12).
* **What the literature says** (§18). The rule is *the limiting-absorption root, with the loss
  applied to `eps` and `mu` separately*: proved for a dispersive double-negative half-space by
  Cassier, Hazard and Joly (2017, 2022), stated for a modal method by Depine and Lakhtakia (2005),
  and written into PyMoosh's code. For an ordinary medium it is Maystre's cut; the mirror cut has
  no published name (§5). Nothing newer contradicts it for passive media; for active media no
  single-frequency rule exists, which is also why Wavebit's answer for an ordinary *gain*
  half-space is a conditional one (§18).

**How to read this page.** §1 is the vocabulary; §2–§3 the physics and the failure; §4–§7 the
rules; §8–§11 amplitudes, layers, refusals and backends; §12 the evidence and the tests; §13 what
changed; §14 other tools; §15 the limits; §16 how to use it; §17 how the decisions were reached;
§18 what published work supports, refines and leaves open.

---

## 1. Words

| word | meaning on this page |
| --- | --- |
| time convention | `e^{-iwt}`: a field is `Re[E e^{-iwt}]`. Loss is `Im(eps) > 0`, `Im(mu) > 0`; gain the opposite sign |
| passive, active | passive: `Im(eps) >= 0` and `Im(mu) >= 0` (lossless or absorbing). Active: either is negative (gain) |
| units | wavevectors are divided by `k0 = 2*pi/wavelength`; `H` means `Z0 H`, so a plane wave has `H = k x E / mu` |
| `k_par`, `kz` | the in-plane and normal parts of a plane wave's wavevector. `k_par` is shared by every medium a wave meets (phase matching); `kz` is each medium's own |
| **double negative (DNG)** | a medium with `Re(eps) < 0` **and** `Re(mu) < 0`. Also called left-handed (Veselago). A half-space is always isotropic (a scalar `eps` and `mu`); for a tensor in a layer see §9 |
| double positive | `Re(eps) > 0` and `Re(mu) > 0`: every ordinary dielectric. The *double-positive twin* of a lossless `(eps, mu)` is `(-eps, -mu)`; the *mirror medium* of any `(eps, mu)` is `(-conj(eps), -conj(mu))` |
| **propagating** | `Re(eps*mu) > 0`, so a wave at normal incidence carries power through the medium. A double negative with `Re(eps*mu) <= 0`, such as `eps = mu = -1+1.2i`, is too lossy to propagate at any angle |
| **negative-phase-velocity (NPV) medium** | one in which a plane wave's phase advances against its power flow (§6). Also called a **negative-index medium (NIM)** on this page and in the code: the two words mean `s = -1` |
| **magnetic metal** | here: a passive NPV medium that is *not* a double negative, with one real part negative, such as `eps = -2+0.01i`, `mu = 0.1+0.5i`. It never propagates (§6) |
| **signed index** `n` | `sqrt(eps) * sqrt(mu)`, each root on Maystre's cut, which for a passive factor is the principal root. `Re(n) < 0` exactly for an NPV medium |
| **`s`** | `sign(Re(n))`: `-1` for an NPV medium, `+1` for every other one, including every medium that names no permeability |
| **phase index `N`** | `\|Re(k)\|` of the incident plane wave at a real angle; `N = \|n\|` in a lossless medium (§7) |
| lit side, far side | the half-space the light arrives from (`incidence_side`), and the other one |
| root, cut | `kz**2` has two square roots; a *cut* is a rule that picks one from `kz**2` alone. Its *branch line* is where the pick jumps; a *tie* is what it does exactly on that line |
| Maystre's cut | the root with `Re(kz) + Im(kz) > 0`; the half-space rule for every ordinary medium ([`branch-cuts.md`](branch-cuts.md) §8). The name is RETICOLO's, from its source code; in print the convention is attributed to R. Petit's 1980 book (§5) |
| the mirror cut | the root with `Im(kz) - Re(kz) > 0`; the rule for a double-negative half-space (§5). The name is this page's: the literature has none for it |
| limiting absorption | fix an undecided root by adding a little loss, taking the root that decays, and following it as the loss is removed. *Where* the loss goes (on `eps*mu`, or on `eps` and `mu`) decides the answer for a double negative (§5). Wavebit's half-space rule, both cuts together, is *the limiting-absorption root, with the loss applied to `eps` and `mu` separately* |
| ETM, SMM | the two ways the solver joins its layers ([`connecting-algorithms.md`](connecting-algorithms.md)): the enhanced transmittance matrix recursion (the default `layer_algorithm`) and the scattering-matrix cascade |
| the five paths | the solver's formulations: pure 1D TE, pure 1D TM (`tetm='te'` / `'tm'`), 1D conical, 2D, and 2D `formulation='4ff'` |
| raster path, shape path | a layer given as a sampled array (`structure` with `mu` beside it), or as a `background` with `shapes`, each with its `material` |
| `q`, `+-q` | a layer mode varies as `exp(-k0 q z)` in the kernels' downward `z`, so `q = -i*kz`: `Re(q) > 0` decays downward, and on a lossless propagating mode `Im(q) < 0` advances the phase downward. Each mode has `+q` and `-q`; which one the recursion calls `+` is its *label* (§9) |
| growth cap | ETM's guard: where the root its rule labels `+` would grow across the layer by more than the machine epsilon to the power -1/4 (8192 in double precision), the decaying root is kept instead (`etm_mode_q`; [`connecting-algorithms.md`](connecting-algorithms.md) §10) |
| factorization | how a layer's convolution matrices are built: `'li'` (Li's rules), `'nvf'` (normal-vector field) or `'laurent'` (the direct rule for every product, as fmmax's `FFT` formulation; opt-in) |
| the per-order frame | each diffraction order's s/p basis: `s_hat = z_hat x k_hat_par`, with `k_hat_par` the order's in-plane direction (§8). Amplitudes `r_s`, `r_p`, `t_s`, `t_p` are quoted in it |
| `psi`, `polarization` | the incident polarization: `psi` an angle in radians (`pi/2` is s, `0` is p), or `polarization=(a_s, a_p)`, its Jones components, in the incident order's frame |
| `power_entering` | the power that crosses the lit interface into the structure, per unit incident power. It is `T` plus what the stack absorbs; it is *not* `1 - R` when the lit medium absorbs (§7) |
| doors | the ways light or data reaches a solve: the **theta door** (`theta`, `phi`), the **direct-k door** (`kx`, `ky`), the **coherent door** (`excitation={side: PlaneWave(...)}`), and the **field door** (`reconstruct_field`, `boundary_fields`) |
| harness | the fixed set of scenes §13 solves in this tree and in develop's, to compare every array byte for byte |

---

## 2. What a negative-index medium is

### Phase and power

A plane wave in a medium of relative permittivity `eps` and permeability `mu` varies as
`exp(i k0 k.r)` with `k.k = eps*mu`. Faraday's law gives its magnetic field, `H = k x E / mu`,
and the time-averaged power flow is the Poynting vector:

```text
S = Re(E x H*) / 2  =  |E|**2 Re(k/mu) / 2         (a transverse wave, k real)
```

In an ordinary lossless medium `mu > 0`, so `S` points along `k`: the power goes where the
phase goes. In a lossless double negative `mu < 0` while `k` is still real (`eps*mu > 0`), so
**`S` points against `k`**. Power still flows away from a source -- energy has to go somewhere --
but the phase fronts move *toward* the source. `E`, `H` and `k` form a left-handed triad, which
is why V. G. Veselago, who worked this out in 1968, called such media left-handed
(V. G. Veselago, *Sov. Phys. Usp.* **10**, 509 (1968)). For the p wave the same argument runs on
`Re(k/eps)`, with the same conclusion, since a propagating double negative has both negative.

### The index and its sign

`n**2 = eps*mu` fixes the index only up to a sign. For an **absorbing** passive medium passivity
fixes it: the index must have `Im(n) > 0`, and only one of the two roots does. For a **lossless**
medium both `+n` and `-n` are real and both are passive; there the index is the limit of the
lossy one as its loss goes to zero. One expression gives both answers. Each of `eps`, `mu` lies
in the closed upper half plane, so each principal square root lies in the closed first quadrant
and their product in the closed upper half plane:

```text
n = sqrt(eps) * sqrt(mu)          each root principal (for a passive factor);   Im(n) >= 0 for every passive medium
```

and it is continuous as the loss in `eps` or `mu` goes to zero, so it is that limit. It is
J. Skaar's `n = sqrt(|eps||mu|) exp(i(arg eps + arg mu)/2)` (*Phys. Rev. E* **73**, 026605 (2006)),
and for a passive medium it is the causal index too: Skaar shows that this product can be continued
analytically into the upper half of the complex-frequency plane, which no rule at one frequency can
promise for an active medium (J. Skaar, URSI EMTS 2010, arXiv:1008.2874). For `eps = -2`, `mu = -1`
it is `-sqrt(2)`: Veselago's negative index, and the `n < 0` of J. B. Pendry's perfect lens
(*Phys. Rev. Lett.* **85**, 3966 (2000)). D. R. Smith and N. Kroll (*Phys. Rev. Lett.* **85**, 2933
(2000)) show that a current source radiating into a one-dimensional left-handed medium forces
`Re(n) < 0`; R. W. Ziolkowski and E. Heyman (*Phys. Rev. E* **64**, 056625 (2001)) fix the index and
impedance roots by analyticity in the complex frequency plane and confirm the negative index with a
positive impedance by finite-difference time-domain runs of a lossy Drude medium. `Re(n) < 0`
exactly when `arg(eps) + arg(mu) > pi`, which is the negative-phase-velocity condition of §6.

### Negative refraction

At a flat interface `k_par` is shared by both media (phase matching). In glass the transmitted
power leaves on the side of the normal that `k_par` points to; in a negative-index medium on the
other side. Written with beam angles -- the angles a measurement sees, signed toward `+x` -- this
is Snell's law with the signed index:

```text
n1 sin(theta1) = n2 sin(theta2)
```

as Veselago (1968) and Pendry (2000) write it, and as R. A. Shelby, D. R. Smith and S. Schultz
measured it at microwave frequencies (*Science* **292**, 77 (2001)). Light from
`n1 = -sqrt(2)` at 20 degrees enters glass (`n2 = 1.5`) at
`arcsin(-sqrt(2) sin(20 deg) / 1.5) = -18.8117` degrees *(study §4)*:

```text
                              normal
  lit half-space, n1 = -sqrt(2) |
                           \    |
  incident light (power),   \   |  the incident wave's phase (k) runs
  20 deg from the normal     v  |  along the same line the other way:
                              \ |  up and back, away from the structure
                               \|
  ==============================O=============================
                               /|  k_par = -sqrt(2) sin(20 deg) = -0.4837,
                              / |  the same on both sides of the interface
                             /  |  (from n1 = +sqrt(2) the light would go on
                            v   |   to the right, at +18.81 deg)
  transmitted light, -18.81 deg:|  glass, n2 = 1.5
  back on the side it came from |
```

---

## 3. The two roots, and why Maystre's cut fails here

### The two roots at a half-space

Below an interface, a half-space carries one plane wave per diffraction order. Its `k_par` is
fixed by the order, so `kz**2 = eps*mu - k_par**2` is fixed too, and `kz` has two roots, `+v`
and `-v`. One is the wave the interface sends into the half-space; the other would be a wave
arriving from infinity, which is not there. Physics names the right one without any cut:

* it does not grow away from the interface: `Im(kz) >= 0`;
* it carries its power away from the interface: `Re(kz/mu) > 0` for an s wave (whose flux is
  `|E|**2 Re(kz/mu)/2`) and `Re(kz/eps) > 0` for a p wave (`|H|**2 Re(kz/eps)/2`).

For a passive medium at a real `k_par` the two criteria pick the same root: in a lossy medium
exactly one root has `Im(kz) > 0`, and passivity makes its flux outgoing; in a lossless medium
both roots are either real, and the flux decides, or imaginary, and the decay decides. That a
lossless double negative's root is the one carrying power away is how Veselago, Pendry and
J. A. Kong (*Prog. Electromagn. Res.* **35**, 1 (2002)) treat it. In a double-positive medium
that root has `Re(kz) > 0` -- the phase leaves the interface too. **In a double negative it has
`Re(kz) < 0`: the phase comes toward the interface while the power leaves it.**

**One rule serves both sides.** The solver's half-space root (`conventions.boundary_kz`) is the
wave that travels along the kernels' downward axis, from their entry face into the stack and on
out of it. In the far half-space that wave is the transmitted one, travelling away from the
interface. In the lit half-space it is the *incident* wave, travelling toward the interface; the
reflected wave there uses `-kz`. Each must decay along its own direction of travel and carry its
power that way, so the transmitted wave decays away from the structure and the incident wave
decays as it approaches it -- one condition on one root, and the same rule (§5) fixes both
sides.

A solver needs a *cut*: a rule that reads `kz**2` and returns a root. This is where double
negatives break things, because `kz**2` does not carry enough information.

### Maystre's cut, and why it is right for ordinary media

Every ordinary half-space in Wavebit takes Maystre's cut, the root with
`Re(kz) + Im(kz) > 0`, whose branch line is the negative imaginary axis of `kz**2`. It is the
root that *limiting absorption* reaches: give the half-space a little loss, take the decaying
root, and remove the loss at fixed `k_par`. Loss raises `Im(kz**2)` in an ordinary medium, so as
it is removed `kz**2` moves straight down and can reach the branch point only on the negative
imaginary axis. For a gain medium the same walk is "the gain switched on from zero". RETICOLO
takes the same cut by default; its source (`retsqrt.m`) calls it the "coupure de Maystre"
(RETICOLO: J. P. Hugonin and P. Lalanne, arXiv:2101.00901), a name its manual does not use. In
print the convention is written `Re(sqrt(x)) + Im(sqrt(x)) > 0`: P. Lalanne and T. Wu
(arXiv:2508.10742 (2025)) attribute it to R. Petit (ed.), *Electromagnetic Theory of Gratings*
(Springer, 1980), and P. Lalanne and J.-P. Hugonin (arXiv:2609.10076 (2026)) cite H. Benisty,
J.-J. Greffet and P. Lalanne, *Introduction to Nanophotonics* (Oxford University Press, 2022),
§18.4.3, for it. [`branch-cuts.md`](branch-cuts.md) §8 has the measurements.

### What it does to a double negative

`Im(eps*mu) = Re(eps) Im(mu) + Im(eps) Re(mu)`. With both real parts negative, **loss lowers
`kz**2`**: the passive pair `eps = -2+0.1i`, `mu = -1+0.1i` multiplies to `1.99-0.30i`, below the
real axis, exactly where a *gain* medium's product sits. Maystre's cut reads that product as gain
and takes the root that grows. S. A. Ramakrishna and O. J. F. Martin (*Opt. Lett.* **30**, 2626
(2005)) map the same thing: with the cut on the negative imaginary axis, a passive propagating
double negative (their "region 4") needs the root on the other sheet, while a gain medium with a
positive index sits in the same quadrant of `kz**2` on the first sheet. Glass onto that pair at
normal incidence *(study §1)*:

| root | `kz` | away from the interface | phase | power `Re(kz/mu)` | `r` | `R` | `T` | `R + T - 1` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Maystre's cut | `+1.4147-0.1060i` | grows | away | `-1.4111` (toward) | `28.209627+11.532417i` | **928.7797034** | -927.7797034 | -1.1e-13 |
| mirror cut (physical) | `-1.4147+0.1060i` | decays | toward | `+1.4111` (away) | `0.030373-0.012417i` | **0.001076681582** | 0.9989233184 | -1.1e-16 |

The solver now returns `R = 0.0010766816` on ETM and SMM *(study §1)*.

**How long it was there.** The wrong root is older than Maystre's cut. From 2026-08-12 until
`6bb364b` (2026-10-03) the half-space took the principal root of `kz**2`, continued from normal
incidence ([`branch-cuts.md`](branch-cuts.md) §8); at a real `k_par` that is the same
fourth-quadrant root for this medium, so a magnetic double-negative half-space was solved with the
root that grows under that rule too. Until `601ba85` a
generic result check, `check_physical`, flagged its negative `T` with an
`UnphysicalResultWarning` -- by the sign of `T`, not by energy. `601ba85` removed that check (it
also called correct answers for active media failed solves; [`result-guards.md`](result-guards.md)),
so for one commit develop returned `R = 928.78` silently, and `d421aa2` then refused the medium.
§17 has the whole sequence.

**Why `r` becomes `1/r`.** At an interface `r = (Y1 - Y2)/(Y1 + Y2)` with `Y = kz/mu` (s) or
`kz/eps` (p). The wrong root flips the sign of `Y2`, so `r -> (Y1 + Y2)/(Y1 - Y2) = 1/r` exactly
(`r_wrong * r_true - 1 = 2.0e-18`, *study §1*).

**Why no energy check could see it.** For a lossless lit medium (`Y1` real),
`|Y1 + Y2|**2 - |Y1 - Y2|**2 = 4 Y1 Re(Y2)`, so `R + T = 1` is an *algebraic identity* of the
interface conditions, whichever root `Y2` came from. The wrong root does not break energy
conservation; it reverses the transmitted flux (`T < 0`): power flowing out of the substrate
into the interface, fed by a wave arriving from infinity. `R > 1` cannot be the alarm either:
it is a legitimate answer for a gain medium, which is why a generic "physicality" check was
removed ([`result-guards.md`](result-guards.md)). A wrong root is invisible to every
power-balance test and visible only to a closed form, a second oracle, or the sign of `T`.

### The same product, three media

At one frequency, three media share `eps*mu = 1.99-0.30i` and need three different answers
*(study §1)*:

| medium | which root is physical | `R`, glass onto it |
| --- | --- | --- |
| the passive double negative `(-2+0.1i, -1+0.1i)` | the decaying root, fixed by passivity | 0.001076681582 |
| `eps = 1.99-0.3i`, `mu = 1`, a gain medium whose gain is switched on from zero | Maystre's root | 0.002178010778 |
| `eps = 1.99-0.3i`, `mu = 1`, the active medium Skaar constructs with the double negative's dispersion | the decaying root | 459.1345507 |

So a rule reading only `eps*mu` cannot be right for all three; a rule reading `eps` and `mu`
separately can be right for the first (it is passive, so limiting absorption decides). For the
other two, `eps` and `mu` at one frequency do not decide either: Skaar showed that which root an
active medium takes depends on its dispersion (J. Skaar, *Phys. Rev. E* **73**, 026605 (2006),
and *Opt. Lett.* **31**, 3372 (2006), which disputes the contrary claim for active media in
Ramakrishna and Martin; B. Nistad and J. Skaar, *Phys. Rev. E* **78**, 036603 (2008)). That is
why §10 refuses active double negatives that propagate rather than guessing.

### The standard table, and what a test row has to contain

Glass (`n = 1.5`) onto the double negative, `R_s / R_p`, both roots *(study §1)*. §14 compares
other tools against these columns.

| medium | root | 0 deg | 30 deg | 60 deg |
| --- | --- | --- | --- | --- |
| `(-2+0.1i, -1+0.1i)` | physical | 0.001076681582 / 0.001076681582 | 0.001516744535 / 0.001195717263 | 0.02662274906 / 0.0297091977 |
| `(-2+0.1i, -1+0.1i)` | Maystre | 928.7797034 / 928.7797034 | 659.3068093 / 836.3181086 | 37.56186102 / 33.65960973 |
| `(-2, -1)` | physical | 0.0008665517772 / 0.0008665517772 | 0.001605140572 / 0.0003535768269 | 0.02128623625 / 0.007713804549 |
| `(-2, -1)` | Maystre | 1153.999133 / 1153.999133 | 622.9983949 / 2828.239646 | 46.97871376 / 129.6377156 |

Three traps for whoever writes a test *(study §1)*:

* **A lossless double negative reflects exactly as its double-positive twin**: `(-2, -1)` and
  `(2, 1)` both give `R = 0.0008665517772`; `(-3, -0.5)` and `(3, 0.5)` both `0.0577961054`.
  Same `|n|`, same impedance. A solver that silently replaces the medium by its twin passes every
  lossless planar row.
* **With `|mu| = 1` an impedance-blind solver passes too.** Treating `eps*mu` as a permittivity
  with `mu = 1` gives `(-2, -1)`'s answer by accident; `(-3, -0.5)` exposes it: `0.0577961054`
  against `0.01020514434` for `eps = 1.5`, `mu = 1` (and `17.30220389` for Maystre's root).
* **The mirror medium** `(-conj(eps), -conj(mu))` of the lossy pair has the same `R` at 0, 30
  and 60 degrees to the bit, with `r_s = conj(r_s(mirror))` exactly -- while `r_s` itself differs
  by up to 0.291. Planar `R` alone cannot tell a double negative from its mirror; a phase, a
  grating, or a lit side can.

The tests here and the fmmax golden therefore use lossy rows with `|mu| != 1`, asymmetric
gratings and oblique azimuths.

---

## 4. Overview of the rules

| decision | ordinary medium (`s = +1`) | double negative | negative-phase-velocity, not double negative (passive magnetic metal) |
| --- | --- | --- | --- |
| half-space root (§5) | Maystre's cut | **mirror cut** | Maystre's cut (it is right there, §5) |
| index (§6) | principal `sqrt(eps*mu)` (unchanged bits) | `sqrt(eps)*sqrt(mu)` = its negation | `sqrt(eps)*sqrt(mu)` = its negation |
| real-angle `k_par` on the lit side (§7) | `+N sin(theta)` (unchanged bits) | `-N sin(theta)` | `-N sin(theta)` |
| incident s/p basis (§8) | the frame of `k_par`, along `phi` | the frame of `k_par`, along `phi + pi` | the frame of `k_par`, along `phi + pi` |
| frame at `theta = 0` (§7) | `phi` | `phi + pi` | `phi + pi` |
| uniform layer labels, ETM (§9) | Maystre's cut, capped (unchanged) | mirror cut, capped | unchanged |
| patterned layer labels, ETM (§9) | unchanged | per mode, by power flux | unchanged |
| active (gain) | solved; `GainCutoffWarning` near grazing | **refused** if it propagates; solved if it does not | solved; its `s` is a convention (§6) |

Every decision is a selection (`where`), never a Python branch on a value, so all of it traces
under `jax.jit` and differentiates (§11).

---

## 5. The half-space root: the mirror of Maystre's cut

### The rule

```text
Re(eps) < 0 and Re(mu) < 0:  kz = the root of eps*mu - k_par**2 with  Im(kz) - Re(kz) > 0
                             (on its own cut, Im(kz) - Re(kz) == 0, the root with Re(kz) < 0)
any other medium:            Maystre's root, Re(kz) + Im(kz) > 0
                             (on his cut, Re(kz) + Im(kz) == 0, the root with Re(kz) > 0)
```

`conventions.boundary_kz(xp, eps*mu, k_par**2, medium=(eps, mu))` decides on the **material**,
never on the product, which (§3) cannot tell a passive double negative from a gain medium. The
kernels pass `medium` whenever the scene names a permeability (`half_space_medium`); a
non-magnetic solve passes nothing and runs the old arithmetic. A half-space is always a scalar
medium, so "double negative" there is the two real parts' signs.

### Why: limiting absorption on `eps` and `mu`

The physical root of a passive medium is the one limiting absorption reaches, provided the
absorption is put where the material puts it: on `eps` and `mu`, not on their product. Adding
loss `delta` to both gives `d(eps*mu)/d(delta) = i (eps + mu)`, whose imaginary part is
`Re(eps) + Re(mu)`: positive for an ordinary medium, **negative for a double negative**. So:

1. Give the double negative enough loss: `kz**2` moves into the lower half plane.
2. Take the decaying root there. On a propagating order (`Re(kz**2) > 0`) it lies in the second
   quadrant: `Re(kz) < 0`, `Im(kz) > 0`.
3. Remove the loss at fixed `k_par`. `kz**2` moves straight *up* to the real axis, and the root
   moves continuously with it -- as long as `kz**2` never crosses the branch line. Moving up from
   the lower half plane, it can reach the branch point only through the *positive* imaginary
   axis. Put the cut there, and it is never crossed.

The cut on the positive imaginary axis of `kz**2` whose root agrees with the decaying one is
exactly `Im(kz) - Re(kz) > 0`. The picture, with a lossy ordinary medium and the lossy double
negative at `k_par = 0`:

```text
                                 ^ Im(kz**2)
                                 |  the mirror cut: Im(kz) - Re(kz) = 0
                                 |  (the positive imaginary axis)
                                 |
  <---- k_par grows -------------|----------------x (2+0.1i, 1+0.1i): 1.99+0.30i
                                 |                ^ loss on eps and mu raises it
                                 |                |
  -------------------------------+----------------o-----------------> Re(kz**2)
                                 |                | (the lossless (2, 1) and (-2, -1) sit on the axis)
                                 |                v loss on eps and mu lowers it
  <---- k_par grows -------------|----------------x (-2+0.1i, -1+0.1i): 1.99-0.30i
                                 |
                                 |  Maystre's cut: Re(kz) + Im(kz) = 0
                                 |  (the negative imaginary axis)
                                 |
```

Taking the loss away moves each `x` vertically onto the real axis, and neither walk meets the
cut of its own rule. Increasing `k_par` moves each `x` to the left, and each track crosses the
*other* rule's cut at grazing (`k_par**2 = 1.99`): that is where Maystre's root of the double
negative jumps (below), and where the mirror root of the ordinary medium would. No single cut
serves both media, which is why the rule has to read `eps` and `mu`.

Measured on the lossless double negative `(-2, -1)` *(study §2)*:

| `k_par` | loss on | distance to the mirror root at `delta` = 1e-2 / 1e-4 / 1e-6 | lossless mirror root | Maystre's root, distance at 1e-6 |
| --- | --- | --- | --- | --- |
| 0.0 | `eps` and `mu` | 1.1e-02 / 1.1e-04 / 1.1e-06 | `-1.4142` | `+1.4142`, 2.8 |
| 0.0 | `eps` only | 3.5e-03 / 3.5e-05 / 3.5e-07 | `-1.4142` | 2.8 |
| 0.0 | `mu` only | 7.1e-03 / 7.1e-05 / 7.1e-07 | `-1.4142` | 2.8 |
| 0.8 | `eps` and `mu` | 1.3e-02 / 1.3e-04 / 1.3e-06 | `-1.1662` | `+1.1662`, 2.3 |
| 1.3 | `eps` and `mu` | 2.7e-02 / 2.7e-04 / 2.7e-06 | `-0.5568` | `+0.5568`, 1.1 |
| 1.7 (evanescent) | `eps` and `mu` | 1.6e-02 / 1.6e-04 / 1.6e-06 | `+0.9434i` | the same root |

Every placement of the loss converges linearly onto the mirror root. Loss put on the **product**
instead (`eps*mu + i*delta`) converges onto Maystre's root, at the same rates (3.5e-07 at
`k_par = 0`): "Maystre's cut is limiting absorption" is true for loss on the product, which is
the same thing as loss on the material only when the material is ordinary.

### The same rule in print

The argument above is not new.

* **Proved, for a dispersive half-space.** M. Cassier, C. Hazard and P. Joly (*Commun. PDE*
  **42**, 1707 (2017); Part II, **47**, 1217 (2022)) solve vacuum against a lossless Drude
  half-space whose `eps` and `mu` are both dispersive. Their root is the limit taken as the
  frequency gains a vanishing imaginary part, which puts `omega*eps` and `omega*mu` *each* in the
  upper half plane -- loss on each factor, not on the product -- and it changes sign, relative
  to an ordinary band, exactly in the bands where the medium propagates with `eps` and `mu` both
  negative ("inverse zones", whose waves they call inverse propagative). Part II justifies that
  choice of root by a limiting absorption process in so many words. Their Drude half-space is
  this page's double negative, and their root is the mirror cut's.
* **Loss on `eps` and `mu` separately**, written as such: H.-M. Nguyen and S. Sil (*Commun. Math.
  Phys.* **379**, 145 (2020)) give a negative-index inclusion `(eps + i delta, mu + i delta)` in
  the `e^{-iwt}` convention, and H.-M. Nguyen (*J. Math. Pures Appl.* **106**, 342 (2016)) does
  the same for the scalar equation. Both treat a bounded inclusion, not a half-space; the phrase
  is theirs, the half-space is not.
* **The trigger.** For a passive, non-dissipative, local material, the frequencies that carry
  backward waves are exactly those with `eps < 0` and `mu < 0` (M. Cassier, P. Joly and
  M. Kachanovska, *Comput. Math. Appl.* **74**, 2792 (2017), Theorem 3.34; their Remark 3.35
  restricts the test to passive materials). That is the lossless limit of "both real parts
  negative", and the reason the rule is keyed on it (below).
* **In a modal method.** R. A. Depine and A. Lakhtakia (*Optik* **116**, 31 (2005)), solving a
  grating over a negative-phase-velocity half-space by the C method, state the per-medium rule
  for its eigenvalues: in the lower medium a positive-phase-velocity material takes the real
  negative one (or `Im < 0`), a negative-phase-velocity material the real *positive* one (or
  `Im < 0`) -- in their sign convention, the lossless mirror. It is the earliest explicit
  statement of it that the 2026-10 review found.
* **Stated again, independently.** M. Lequime, B. Gralak, S. Guenneau, M. Zerrad and C. Amra
  (arXiv:1312.6288 (2013)) take the minus sign of the root formula for a negative-index medium,
  which for every passive double negative is the mirror root, evanescent orders included (a
  lossless evanescent order aside, where the sign their formula reads is zero), and note that
  every approach agrees for media that refract negatively and do not amplify. PyMoosh
  (D. Langevin et al., *J. Opt. Soc. Am. B* **41**, A67 (2024)) writes the same switch into its
  code -- the principal root, negated in the lit and the last medium when both real parts are
  negative -- though not into its paper.
* **Sorted by power.** Codes that sort a half-space's waves by decay and, on a lossless order,
  by the sign of the Poynting flux reach the same root for every passive medium: P. Lalanne and
  J.-P. Hugonin (arXiv:2609.10076 (2026), §2.3) and E. Yang, Q. Song and W. Cai
  (arXiv:2608.06185 (2026), citing L. Li).

The mirror *symmetry* that the identity below implies for a solve is A. Lakhtakia's conjugation
symmetry (below).

### The mirror identity

The rule is Maystre's run on the mirror medium `(-conj(eps), -conj(mu))`, whose `kz**2` is
`conj(kz**2)`:

```text
kz_mirror(z) = -conj(maystre(conj(z)))
```

`boundary_kz` satisfies this **bit for bit** on 26,005 of 26,005 inputs -- random `kz**2`, both
axes, both diagonals and the signed zeros *(study §2)*. The ties are each other's mirror:
Maystre's cut keeps the root with `Re(kz) > 0` on his line (`kz**2 = -2i` gives `1-1i`, with a
strict `<` in the code), the mirror keeps `Re(kz) < 0` on its own (`kz**2 = +2i` gives `-1-1i`,
with `<=`).

**What it does to a solve.** The mirror medium is A. Lakhtakia's conjugation symmetry (*Microwave
Opt. Technol. Lett.* **40**, 160 (2004)): replacing `eps` and `mu` everywhere by the negatives of
their conjugates conjugates both `E` and `H`. V. Grunhut and R. A. Depine (*Eur. Phys. J. D* **62**,
227 (2011)), citing it, point out what it means at a flat interface lit at a real angle: the Fresnel
coefficients change their phase but not their magnitude, so `|r|` and `|t|` cannot tell a positive
medium from its double-negative mirror. Lit from a lossless medium, with only the far side mirrored,
the change is exactly a conjugation: the root goes to `-conj(kz)`, the admittances `kz/mu` and
`kz/eps` to their conjugates, the signed index to `-conj(n)`, so `r -> conj(r)` and `t -> conj(t)`.
Through the solver, with no layer between, this holds **bit for bit** on every path (TE, TM,
conical, 2D, 4ff), ETM and SMM, either side lit, lossy and lossless far sides, propagating and
evanescent in them, at 0, 0.3 rad and 60 degrees: every operation on the bare interface has the
conjugate operation as its conjugate. The same identity holds layer by layer -- a layer's
first-order system goes to its conjugate under the mirror -- so with every layer and the far side
mirrored, and every retained order propagating in the lit medium (whose admittances must stay real),
the reflection conjugates again: two lossy uniform layers (a positive one and a double negative,
which swap) and a grating symmetric about the cell centre, every path, to 7.8e-15; at the bare
interface 2,376 of 2,376 arrays are exactly conjugate *(study §8;
`test_the_mirror_medium_conjugates_every_reflection`,
`test_mirroring_everything_below_the_light_conjugates_it_through_layers`)*. Either fails if a double
negative takes Maystre's root or the principal index. The layer labels of §9 do not show in it: they
move no exact answer.

### Where the two cuts differ, and why everything else keeps its bits

* **They agree wherever `Re(kz**2) < 0`** -- 10,960 of 10,960 random samples -- and disagree
  wherever `Re(kz**2) > 0` (0 of 13,040 agree) *(study §2)*. So evanescent orders of a double
  negative, and a double negative too lossy to propagate anywhere (`Re(eps*mu) <= 0`), keep the
  root they always had; only propagating orders (and Maystre's tie line) change.
* **Every other medium keeps its bits.** `boundary_kz` with `medium` selects with `where`, and for
  a medium that is not a double negative the selection hands back Maystre's own decision and
  comparison: 6,416 of 6,416 roots are bit-identical with and without `medium`, for eight media
  covering gain, metals, a magnetic metal and a mu-negative medium, at real and complex `k_par`
  *(study §2)*.
* **The decision is never in round-off.** For a passive double negative at a real `k_par`,
  `kz**2` lies in the closed lower half plane, where `|Im(v) - Re(v)| = |Re(v)| + |Im(v)| >= |v|`.
  Measured at least `1.000043 |v|` on 200,000 random passive double negatives, where the mirror
  root was the physical one (decaying; outgoing power if lossless) on 200,000 of 200,000
  *(study §2)*.
* **It is continuous through grazing.** Sweeping `k_par` through `sqrt(Re(eps*mu))` in steps of
  1e-4, the mirror root moves by at most 2.6e-04 per step; Maystre's root jumps by 1.10, because
  its cut runs right through the passive double negative's path *(study §2)*.

### The tie that was a live bug: `eps = mu = -1+1i`

At normal incidence this medium has `kz**2 = -2i`, exactly **on Maystre's cut**, where his tie
keeps `1-1i`, which grows. `Re(eps*mu) = 0`, so the refusal of `d421aa2` (which asked for
`Re(eps*mu) > 0`) let it through. Glass onto it *(study §2, §10)*:

| | `kz` | `R` |
| --- | --- | --- |
| Maystre's tie (develop `0dfc4ce`, ETM) | `+1-1i` | 25 |
| develop, SMM | -- | `LinAlgError` |
| the mirror cut (now, ETM and SMM) | `-1+1i` | 0.040000000000 |

### Why "both real parts negative" and not "s = -1"

The rule is keyed on the double negative, not on the negative-phase-velocity class of §6. A
passive NPV medium that is not a double negative -- a magnetic metal such as
`eps = -2+0.01i`, `mu = 0.1+0.5i` -- has `kz**2` in the open third quadrant at every real `k_par`,
where Maystre's root is already the decaying one -- on 1,514,874 of 1,514,874 (24,834 random
passive magnetic metals, 61 values of `k_par` from 0 to 6) *(study §2)*. Such media need no
mirror, and taking it would move their bits. One edge is left, and §15 records it. In the
lossless limit the key is also the theory's: a passive, non-dissipative local medium carries
backward waves exactly where `eps < 0` and `mu < 0` (Cassier, Joly and Kachanovska 2017,
Theorem 3.34, above).

### The alternatives, and why each was not taken

| rule | what it gets wrong |
| --- | --- |
| Maystre's cut for every medium (develop) | every propagating order of a passive double negative: the growing root, `R = 928.78` (§3) |
| "negate the root for every double negative" (applied to Maystre's root) | negates the evanescent orders and the non-propagating double negatives too, which Maystre's cut already had right: `eps = mu = -1+1.2i` would get `+1-1.2i`, which grows |
| `Im(kz) > 0` for every medium (fmmax's and inkstone's rule, §14) | right for every *lossy* medium of either sign, but it has nothing to read on a lossless propagating order (`Im(kz) = 0`), where the tie must be broken by the power; and for gain media it is not Maystre's root, which Wavebit keeps there ([`branch-cuts.md`](branch-cuts.md) §8) |
| key the mirror on `s = -1` instead of on the double negative | the same answers except on one knife edge (§15), and it would move the bits of the passive magnetic metals, which need no mirror (above). It was decided to key on the double negative |
| "negate the principal root when both real parts are negative", on the index (PyMoosh's rule, §14) | a rule on the index, not a cut: it leaves the root of a passive magnetic metal to the principal root, which grows (glass onto `(-10+0.1i, 1+0.5i)`: `R = 1.463` against `0.6834`, which Maystre's cut takes *(study §1)*) |

### What to call it

The literature has names for neighbouring ideas and none for this rule as a whole, nor for its
double-negative half. The names used here, and the ones avoided:

* **The rule: the limiting-absorption root, with the loss applied to `eps` and `mu`
  separately.** That is how the rigorous sources justify the same choice (Cassier, Hazard and
  Joly; Nguyen and Sil; above), and it carries its own scope: passive media, which is what
  Wavebit solves without refusal. For a gain medium the phrase reads as "the gain switched on
  from zero", which is the physical answer only for weak, conventional gain (§18).
* **For an ordinary medium it takes the form of Maystre's cut**, `Re(sqrt) + Im(sqrt) > 0`. The
  name is RETICOLO's, from its source code (`retsqrt.m`); the published attribution found is to
  R. Petit (ed.), *Electromagnetic Theory of Gratings* (Springer, 1980), as cited by Lalanne and
  Wu (§3). This page keeps "Maystre's cut" because RETICOLO, the outside code that shares it,
  uses it.
* **For a double negative it takes the form of the mirror of Maystre's cut**,
  `Im(sqrt) - Re(sqrt) > 0`. That is a description, not a borrowed name: no source the
  2026-10 review read writes the double-negative rule as a cut on the positive imaginary axis of
  `kz**2`. Every one either sorts the waves by decay and Poynting flux, or negates the root when
  the medium is a double negative.

Names avoided, and why:

* **"The causal root."** Causality fixes a root through `eps(omega)` and `mu(omega)` continued
  into the upper half of the complex-frequency plane, not through their values at one real
  frequency; for passive media the two agree, for active ones they need not (J. Skaar, URSI EMTS
  2010, arXiv:1008.2874). And causality-based criteria for a negative index test the group
  velocity, not the Poynting vector this rule's lossless limit reads (P. Kinsler and M. W.
  McCall, *Phys. Rev. Lett.* **101**, 167401 (2008)).
* **"Petit's cut."** RETICOLO's source already gives that name to a different cut, the decaying
  one, `Im(kz) > 0` ([`branch-cuts.md`](branch-cuts.md) §8), while the published attribution to
  Petit's book is for the `Re + Im > 0` convention. The name would point both ways.
* **"Mandelstam's radiation principle."** That is an energy principle -- the outgoing wave is the
  one carrying energy away -- and A.-S. Bonnet-Ben Dhia, L. Chesnel and M. Rihani (*Pure Appl.
  Anal.* **7**, 127 (2025)) note that it and limiting absorption can disagree in rare cut-off
  cases. This rule is the limiting-absorption one.

---

## 6. The signed index `n = sqrt(eps) * sqrt(mu)`

### The rule, and which media it negates

`conventions.signed_index(eps, mu)` is the one place an `(eps, mu)` pair becomes an index. The
public `n_top` / `n_bot` / `n_inc` read it, and so do the kernels' `n/mu`, the incident `k_par`,
the field door and the coherent door's plane-wave basis. A medium with no permeability never
reaches it: its index is the principal `sqrt(eps)` (`index_from_eps`), as always *(study §3)*:

| medium `(eps, mu)` | principal `sqrt(eps*mu)` (develop's index) | signed `n` (now) | `s` |
| --- | --- | --- | --- |
| `2.25`, no `mu` | `+1.5000` | the same bits | `+1` |
| `(2.25, 1.3)` | `+1.7103` | the same bits | `+1` |
| `(-2+0.1i, 1)`, a metal | `+0.0353+1.4147i` | the same bits | `+1` |
| `(2+0.1i, -1+0.1i)`, mu-negative | `+0.0353+1.4182i` | the same bits | `+1` |
| `(-2+0.1i, -1+0.1i)` | `+1.4147-0.1060i` | `-1.4147+0.1060i` | `-1` |
| `(-2, -1)` | `+1.4142` | `-1.4142` | `-1` |
| `(-3+0.1i, -0.5+0.05i)` | `+1.2254-0.0816i` | `-1.2254+0.0816i` | `-1` |
| `(-1+1.2i, -1+1.2i)`, too lossy to propagate | `+1.0000-1.2000i` | `-1.0000+1.2000i` | `-1` |
| `(-2+0.01i, 0.1+0.5i)`, a passive magnetic metal | `+0.6383-0.7826i` | `-0.6383+0.7826i` | `-1` |

The principal root of the product has `Im < 0` in every `s = -1` row: it reads a passive medium
as gain. The signed index keeps `Im(n) >= 0` on 1,000,000 of 1,000,000 random passive pairs
*(study §3)*.

### The negative-phase-velocity class

`Re(n) < 0` is the negative-phase-velocity condition. On 1,000,000 random passive pairs (a
quarter with a lossless `mu`, a quarter with a lossless `eps`, none with both lossless), `s = -1`
agrees with each of these on every pair *(study §3)*:

| condition | source |
| --- | --- |
| `(abs(eps) - Re(eps)) (abs(mu) - Re(mu)) > Im(eps) Im(mu)` | M. W. McCall, A. Lakhtakia and W. S. Weiglhofer, *Eur. J. Phys.* **23**, 353 (2002) |
| `Re(eps) abs(mu) + Re(mu) abs(eps) < 0` | R. A. Depine and A. Lakhtakia, *Microwave Opt. Technol. Lett.* **41**, 315 (2004) |
| `Im(eps*mu) < 0` | R. Ruppin's form, which the same paper shows equal to the others for passive media **off the real axis of `eps*mu`** |

The third needs its qualifier. **A lossless double negative has `Im(eps*mu) = 0`** and is
negative-phase-velocity all the same: `(-2, -1)` and `(-3, -0.5)` have `s = -1`, and both of the
first two conditions hold for them, while `Im(eps*mu) < 0` does not *(study §3)*. The random
sample above has no pair on that axis, so it says nothing there; the first two conditions are the
ones that cover every passive medium.

Of those pairs 499,905 have `s = -1`: 249,498 double negatives and 250,407 with one real part
non-negative (passive magnetic metals); no double negative has `s = +1`. Among media that
propagate (`Re(eps*mu) > 0`), `s = -1` holds exactly for the double negatives: 462,415 of 462,415
*(study §3)*. No non-magnetic medium is ever negated.

**Active media: each factor's root is Maystre's.** For an active medium the sign is a *convention*,
not physics: which index an active medium has depends on its dispersion, and no rule at one
frequency decides it (Skaar 2006, 2010; §3). It reaches the solve in only two places, the sign of
`t_p` and the direction of a lit `k_par` (below), and §10 lists the active media that are not
refused. The convention is A. A. Govyadinov, V. A. Podolskiy and M. A. Noginov's
(*Appl. Phys. Lett.* **91**, 191103 (2007)): `n = sqrt(eps) * sqrt(mu)` with each factor's root on
Maystre's cut, `Re + Im >= 0`, the root of switching that factor's loss or gain on from zero. They
note that their table of which media decay and which grow cannot be reduced to a single cut in the
complex plane; the recipe cuts each factor instead. For a passive factor, and for an active one with
`Re > 0`, that is the principal root, so nothing above moves; it differs only for a *gain-metal*
factor (`Re < 0`, `Im < 0`), and a medium with exactly one such factor gets the exact negation of
the index the principal factors gave it.

Why this one: at normal incidence the half-space's `kz` is `+-n`, and `t_p = t_s` there exactly
when `n` points along the `kz` it is solved with. On 400,000 random media (`Re` in `(-3, 3)`,
`Im` in `(-1, 1)`), the share of `n` pointing along that `kz` *(study §3)*:

| media | count | principal factors (until 2026-10) | Maystre's factors (now) |
| --- | --- | --- | --- |
| passive | 99,848 | 100% | 100% |
| active, no gain-metal factor | 124,941 | 97.6% | 97.6% |
| active, a gain-metal factor, not a double negative, evanescent | 93,797 | 0% | 100% |
| active, a gain-metal factor, not a double negative, propagating | 5,839 | 100% | 0% |
| active double negative, evanescent (accepted) | 2,903 | 0% | 0% |
| active double negative, propagating (refused) | 72,672 | 30.4% | 100% |

So it fixes the common case -- a gain metal, `t_p = -t_s` at normal incidence under principal
factors -- and breaks a rare one, a gain metal paired with a strongly absorbing factor that
together propagate. No rule built from `eps` and `mu` separately gets every class; the sign of the
normal-incidence `kz` itself would, by construction. The magnetic metal with gain
`(-2-0.01i, 0.1-0.5i)` now gets `n = +0.6383+0.7826i`, `s = +1`, and lit at `theta = 0.3`,
`kx = +0.1939` (until 2026-10: `-0.6383-0.7826i`, `s = -1`, `kx = -0.1939`) *(study §7)*.

**Naming `mu = 1` no longer keeps a gain metal's index.** A non-magnetic medium's index is still
the principal `sqrt(eps)` (`index_from_eps`, unchanged), so `Material(-2-0.1i, mu=1.0)` gets the
opposite `n` -- and `t_p` -- to `-2-0.1i` unnamed. Every power quantity, `r` and `t_s` agree.

### How it is computed, and why it keeps every other medium's bits

```text
r = principal sqrt(eps*mu)                    the expression the index always was
m(z) = -sqrt(z) where Re(z) < 0 and Im(sqrt(z)) < 0, else sqrt(z)  Maystre's root of a factor
q = m(eps + 0j) * m(mu + 0j)                  only steers
n = -r  where Re(conj(r) * q) < 0  and  Re(r) != 0;   otherwise r
```

* **Bit identity.** On the 500,095 pairs with `s = +1` the result is `r` to the bit; on the
  499,905 with `s = -1` it is `-r` to the bit, an exact negation, so `n**2` keeps its bits and no
  `kz` can move *(study §3)*.
* **Why the projection and not `sign(Re(q))`.** `q` equals `+-r` up to round-off, so
  `Re(conj(r) q)` is `+-|n|**2` and its sign is never in doubt. `Re(q)` alone is round-off where
  `eps*mu` sits on the negative real axis (`n` nearly imaginary): on 199,930 passive pairs placed
  within 3 ulp of it, reading `Re(q)` alone returned the gain-reading root for 35,855, the
  projection for none *(study §3)*.
* **Maystre's root of a factor, decided on the factor.** `m(z)` is the root with
  `Re + Im > 0`; for the principal root that sum is negative exactly when `Re(z) < 0` and the
  root's imaginary part is negative, two signs no round-off moves, so that is what is read. The
  cut's own tie, a factor exactly on the negative imaginary axis, keeps the principal root. Read
  off the computed `Re + Im` of the root, as the rule was first written, that tie was round-off of
  either sign: with `mu = -3` and `eps = -0.01i .. -3.00i`, `n` pointed against the `kz` it is
  solved with (`t_p = -t_s` at normal incidence) for 67 of the 300 on NumPy, JAX and Python
  numbers and for none on PyTorch; now for none anywhere *(study §3)*. Off the tie the two
  readings agree to the bit.
* **Signed zeros.** `q` is formed from `eps + 0j` and `mu + 0j`, so `(complex(-2, -0.0), -1)` is
  the lossless `-1.4142`, not `+1.4142`.
* **Exactly on the negative real axis** (a lossless medium that carries no wave) `r` and `-r` are
  equally lossless, and `r` is returned unturned -- the root a non-magnetic medium takes. So
  `Material(-(4+0j), mu=1.0)` has the bits of `sqrt(-(4+0j)) = -2i`, as naming `mu = 1` promises
  ([`conventions.md`](conventions.md), "Materials") *(study §3)*.
* **Types.** A Python number in gives a Python complex out (through `cmath`), a NumPy scalar a
  NumPy scalar; arrays and tracers go through their own namespace with `where`.

### Continuity: why not "both real parts negative"?

`s` changes only where `eps*mu` crosses the negative real axis, and there `N = 0`: nothing
propagates, so nothing physical jumps. A sign keyed on "both real parts negative" would flip
where `Re(eps)` or `Re(mu)` crosses zero, inside media that propagate. Measured on a conical
grating whose far side crosses each seam by `+-1e-9`, the factor that crosses having a fixed
imaginary part *(study §3)*:

| seam | the far side's `(eps, mu)` | `n_bot` there | step in `n_bot` | step in every `R`, `T`, `r`, `t` | step under a "both negative" rule |
| --- | --- | --- | --- | --- | --- |
| `Re(eps) = 0` | `(+-1e-9 + 0.3i, -1+0.1i)` | `-0.3684+0.4071i` | 1.8e-09 | 1.4e-09 | 1.0982 |
| `Re(mu) = 0` | `(-2+0.1i, +-1e-9 + 0.4i)` | `-0.6168+0.6485i` | 2.2e-09 | 1.3e-09 | 1.7900 |

### What the sign decides, and what it does not

`n**2 = eps*mu` does not depend on the sign, so **no `kz`, `R` or `T` does**: every root is taken
from `n**2` by §5's rules. The sign sets:

1. **`n/mu`, the medium's admittance**, whose real part a passive medium needs positive. It
   enters the per-layer modal amplitudes and half-space spectra the solver keeps to rebuild
   fields (its *field state*), and the kernels quote a `p` amplitude along a `p_hat` normalised
   by `n`, so the sign of a transmitted `t_p` follows it.
2. **The direction of the incident `k_par`** for a real angle on the lit side (§7).
3. **The incident `p_hat`** and every order's `p_hat` in the lit or far medium (§8).

**The magnetic metal's `t_p`.** Glass onto `(-2+0.01i, 0.1+0.5i)` at normal incidence, closed
form: `t_s = t_p = 0.8319-0.4608i` with the signed index. The solver, through a 37 nm glass
spacer, returns `t_s = t_p = 0.9336+0.1811i` (`|t_p - t_s| = 3.3e-16`); with the principal index
`t_p` was `-0.9336-0.1811i = -t_s` *(study §3)*. At normal incidence every isotropic interface
has `t_p = t_s`; the principal index broke that for every far-side NPV medium.

---

## 7. What `theta` and `phi` mean

### The rule

For real `theta` and `phi`, measured from the lit half-space's inward normal (`-z` lit from the
top, `+z` from the bottom) and from `+x` toward `+y`:

```text
k_par / k0 = s * N * sin(theta) * (cos(phi), sin(phi))

eps*mu = a + i b
N      = |Re(k)| = sqrt((a + hypot(a, b / cos(theta))) / 2)     incident_phase_index
s      = sign(Re(n)),  n = sqrt(eps) * sqrt(mu)                   +1 unless NPV
kz_inc = the lit half-space's own root (§5) at that k_par
```

`N` follows from writing the incident normal component as `kz = s N cos(theta) + i alpha` and
imposing both parts of `kz**2 + k_par**2 = eps*mu`: `N**2 - alpha**2 = a` and
`2 s N cos(theta) alpha = b`. `b` enters `N` only squared, so one `N` serves both signs, and
`alpha = b / (2 s N cos(theta))` is positive for every passive medium -- an ordinary one has
`b >= 0` and `s = +1`, an NPV one `b <= 0` and `s = -1` -- so the incident wave decays as it
travels toward the structure.

`theta` and `phi` are the angles of **`s * Re(k)`**: the incident wave's phase direction,
turned to point where its light goes.

* **Positive-index medium (`s = +1`)**: `s Re(k) = Re(k)`, the phase direction. In a lossless
  medium that is also the light's direction; in an absorbing one the light departs from it (below).
  This is the meaning the angle has had since real angles in absorbing media were introduced, and
  it is unchanged to the bit.
* **Negative-index medium (`s = -1`)**: the phase runs against the light, so `k_par` points
  along `phi + pi` for light travelling along `phi`.
* **Lossless medium of either sign**: `s Re(k)` is the Poynting vector's direction, so `theta`
  and `phi` are *exactly* the direction the incident power travels, and
  `kx = n sin(theta) cos(phi)` with the signed `n`.

Seen from above, with the light travelling along `phi`:

```text
                        ^ +y
                        |         the light travels along phi, from either medium;
                        |       / k_par from a positive-index lit medium (n > 0)
                        |     /   points the same way
                        |   /
                        | /  phi
  ----------------------+-------------------------> +x
                      / |
                    /   |
                  /     |  k_par from a negative-index lit medium (n < 0):
                /       |  along phi + pi, against the light -- and the
                        |  direction the frame takes at theta = 0
```

### Why the light's direction and not the phase's

A real angle has to name *something*, and in a negative-index medium phase and power disagree.
The alternatives, and why they were not taken:

| reading of `theta` | what `theta = 20 deg` from `n = -sqrt(2)` would mean | verdict |
| --- | --- | --- |
| **the light's direction** (adopted) | a beam arriving at 20 deg; it leaves into glass at -18.8 deg | Snell's law with beam angles and a signed `n`, as Veselago and Pendry state it; what an experiment measures (Shelby, Smith and Schultz 2001) |
| the phase direction, literally | the incident phase fronts move *away* from the structure (§8), so an incident wave needs `theta` between 90 and 180 deg, and every acute angle describes an outgoing wave | every ordinary script's angle would mean something different here; refusing acute angles would be needed |
| `k_par = +N sin(theta)`, the formula develop applied to every lit medium | a beam arriving at **-20** deg: neither the phase nor the power direction | mirrors every asymmetric grating's order set |

Other RCWA codes do not define it: RETICOLO's manual writes `k_parallel = n_inc sin(theta)` and
says of `n_inc`: "One expects that it is a positive real number"; fmmax builds the
incident wavevector from `sqrt(Re(eps))` alone. Both take the in-plane wavevector as the primary
input, as Wavebit's direct `kx` / `ky` do.

**What kind of statement the angle is.** It is a convention Wavebit chose: the direction of the
incident *power*, the Poynting vector -- exact in a lossless medium, and the angle of `s Re(k)` in
an absorbing one (below). It is not a causal statement. P. Kinsler and M. W. McCall
(*Phys. Rev. Lett.* **101**, 167401 (2008)) show that causality-based criteria for a negative index
test the group velocity, not the Poynting vector, and that the two are not equivalent; in a
companion paper (*Microwave Opt. Technol. Lett.* **50**, 1804 (2008)) the published
negative-phase-velocity conditions agree for passive media and part for active ones. The angle is
therefore a definition everywhere, and it is the light's direction exactly where the lit medium is
lossless. No source the 2026-10 review read defines an incidence angle inside a double negative at
all. Several codes can be lit from one (the last column of §14's table), but the one found that
solves a lossless lit double negative physically, PyMoosh (D. Langevin et al.,
*J. Opt. Soc. Am. B* **41**, A67 (2024)), takes the *phase's* angle: its in-plane wavevector is the
principal `sqrt(eps*mu)` times `sin(theta)`, so from a lossless double negative its `kx` at a
given `theta` is the negative of Wavebit's (from an absorbing one it is complex, with a real part
of the opposite sign), and its golden is compared at its own `kx` for that reason (§14).

### Lossless lit media: exactly the power direction

The incident field the solve returns (`boundary_fields(..., part='incoming')`), its Poynting
vector against `(theta, phi = 0.6)`, `theta` from 0.1 to 1.3 rad, `psi` 0 / `pi/2` / 0.7, both
sides, the 2D, conical and 4ff paths *(study §4)*:

| lit medium | polar angle | azimuth | `(kx, ky) = n sin(theta) (cos phi, sin phi)` with signed `n` |
| --- | --- | --- | --- |
| `(-3, -0.5)`, `n = -sqrt(1.5)` | 3.8e-14 deg | 5.6e-16 rad | 1.1e-16 |
| `(-2, -1)`, `n = -sqrt(2)` | 3.8e-14 deg | 5.6e-16 rad | 0 |
| `(2.25, 1.3)`, `n = +1.71` | 3.3e-14 deg | 1.1e-16 rad | 0 |

**Negative refraction.** Lit from `(-2, -1)` at 20 degrees onto glass, `kx = -0.483690`, the
transmitted field's Poynting vector leaves at `-18.8117` degrees (to 7.1e-15 deg, both sides, 2D
path at `phi = 0.6`), and `R = 0.0010947`, `T = 0.9989053` *(study §4)*.

### Absorbing lit media: what the angle does not promise

In an absorbing medium the incident intensity is uniform along the surface (the in-plane
wavevector is real), and the Poynting vector departs from `s Re(k)` by an amount set by the loss
and the polarization. The angle does not try to undo that: a power direction that depends on the
polarization cannot fix one `k_par` for both. At `theta = 45.8` degrees, `phi = 0.6`, the incident
Poynting vector's polar angle, signed in the plane of incidence *(study §4)*:

| lit medium | `s` | p light | s light |
| --- | --- | --- | --- |
| `2+0.5i`, lossy dielectric | `+1` | 44.07 | 45.80 |
| `-10+1i`, metal | `+1` | **-45.51** (across the normal) | 45.80 |
| `(-3+0.1i, -0.5+0.05i)`, lossy double negative | `-1` | 45.67 | 45.41 |
| `(-1+1i, -1+0.5i)`, absorbing double negative | `-1` | 25.67 | 33.23 |
| `(-1+0.2i, 0.5+2i)`, magnetic metal | `-1` | 36.36 | -8.43 |

The first two rows are develop's behaviour for ordinary absorbing media, unchanged; they are why
the angle is defined on the wavevector at all. The normal component of the incident flux points
into the structure in every row.

**A worked absorbing lit example, and why `R + T` is not 1 there.** Lit from the lossy double
negative `(-2+0.1i, -1+0.1i)` at 20 degrees, `phi = 0`, onto glass, one interface *(study §4)*:
`N = 1.415176` (against `|n| = 1.418621`), `kx = -N sin(20 deg) = -0.484019`, the incident wave's
public `k_z = +1.329830-0.112796i` (its phase moves up, away from the structure, and it decays as
it comes down toward it), and the light leaves into glass at `-18.825` degrees.

| polarization | `R` | `T` | `R + T` | `power_entering` | incident power travels at |
| --- | --- | --- | --- | --- | --- |
| s | 0.0011723371 | 0.9990539670 | 1.0002263042 | 0.9990539670 | 19.845 deg |
| p | 0.0010474959 | 1.0001534647 | 1.0012009606 | 1.0001534647 | 19.922 deg |

`T > 1` and `R + T > 1` are not errors. In an absorbing lit medium the incident and reflected
waves overlap, and the power through the interface is not the incident flux minus the reflected
one: there is a cross term. `power_entering` is the power that actually crosses into the
structure; with a lossless stack and a lossless far side it equals `T`, as here. Read
`power_entering`, not `1 - R`, whenever the lit medium absorbs -- whatever its sign of index.

### Normal incidence: the frame follows the limit

At `theta = 0` the zeroth order has no in-plane direction, and the per-order s/p frame -- in
which the incident polarization is written -- takes one by convention: the limit `theta -> 0+` of
`k_par`'s own direction. From a negative-index side that is `phi + pi`, so the frame is taken
there (`CoreSolveMixin._frame_azimuth`; the pure 1D frame sign `order_frame_sign` and the coherent
door's `resolve_wave` read the same azimuth). A grating lit from `(-3+0.1i, -0.5+0.05i)` at
`phi = 0.6`, `theta = 0` against `1e-9`, every amplitude, `r_xyz` / `t_xyz`, the reconstructed
field and the incident field, 5 paths x 2 sides x ETM / SMM: they move by at most 8.7e-09. With
the frame left at `phi` they would step by at least 4.670 on the conical, 2D and 4ff paths
*(study §4)*.

### Complex angles and direct wavevectors

* **A complex `theta` or `phi`** keeps the formal `k_par = n sin(theta) (cos(phi), sin(phi))`,
  with the same **signed** `n`. In a lossless medium the two parameterisations are one: at
  `theta = 0.35 + 1e-9i` against `0.35`, `kx` and every amplitude move by 1.9e-09. In an absorbing
  medium they differ by design (`N` against the complex `n`): `kx` moves by 0.028, staying on the
  same side of the normal (`-0.4203` real, `-0.4202` complex) *(study §4)*. Their `R` and `T` are
  formal ratios, as for any complex angle ([`conventions.md`](conventions.md#real-angle-incidence)).
* **Direct `kx`, `ky`** are physical wavenumbers and carry no angle, so they are taken as given
  whatever the lit medium. Set to the `theta` door's own `(kx, ky)`, they reproduce every
  amplitude to 3.1e-15 *(study §4)*.
* **At exactly normal incidence the two doors take different frames from a negative-index side.**
  The direct-`k` door has no `phi`, so at `kx = ky = 0` its s/p basis uses azimuth zero; the
  `theta` door at `theta = 0, phi = 0` uses `phi + pi`. From `(-2, -1)` with `polarization=(1, 0)`
  the `theta` door's incident field is `E = (0, -1, 0)` and the direct-`k` door's `(0, +1, 0)`,
  while both report `r_s = -0.029437` *(study §4)*: each amplitude is quoted in its own door's
  frame. From a positive-index side the two doors coincide. A comparison of phases across the two
  doors at normal incidence needs that sign.

### The `s = +-1` seam of a lit absorbing medium

For a passive lit medium `s` changes where `eps*mu` crosses the negative real axis. There `N = 0`:
the medium carries no wave, every `theta` gives `k_par = 0`, and a real angle exactly on the axis
is refused. Beside it `k_par` passes through zero continuously. Lit from `eps = -2+0.1i`,
`mu = 1 + (0.05 +- 1e-9)i` at `theta = 0.3`, `phi = 0.6` *(study §4)*:

| `Im(mu)` | `n_top` | `s` | `kx` | incident `E` |
| --- | --- | --- | --- | --- |
| `0.05 - 1e-9` | `+7.06e-10+1.4160i` | `+1` | `+1.80e-10` | `(-0.955, -0.296, 0)` |
| `0.05 + 1e-9` | `-7.06e-10+1.4160i` | `-1` | `-1.80e-10` | `(+0.955, +0.296, 0)` |

`R` steps by 2.1e-09; `T` by 2.1e-08 relative (it is a ratio to an incident power that vanishes on
the seam, here up to 17.2). The incident field changes sign because the frame follows `k_par`
through zero -- the same convention every medium has when `theta` itself crosses zero: glass at
`theta = -1e-9` and `+1e-9` has `E_inc = (+0.955, +0.296, 0)` and `(-0.955, -0.296, 0)`, with `R`
and `T` continuous to 5.5e-10 *(study §4)*. A phase-sensitive sweep across either should multiply
by the sign of the incident `kx`, as [`conventions.md`](conventions.md#the-result-surface)
describes for a sweep through `theta = 0`.

---

## 8. Amplitudes, polarization and the incident wave

The s/p frame convention is unchanged ([`conventions.md`](conventions.md#the-result-surface));
every `n` in it is now the signed one.

* Each order is quoted in its own frame. With `(ca, sa) = (cos, sin)` of the order's azimuth (the
  direction of its `k_par`), `rho = kx ca + ky sa = |k_par|`, `d = -1` lit from the top and `+1`
  lit from the bottom, `kz` the half-space's own root and `n` its signed index:

  ```text
  s_hat      = (-sa, ca, 0)                              = z_hat x k_hat_par
  p_hat^ref  = ( kz_i ca/n_i,  kz_i sa/n_i, d*rho/n_i)   in-plane part +(kz_i/n_i) k_hat_par
  p_hat^trn  = (-kz_t ca/n_t, -kz_t sa/n_t, d*rho/n_t)   in-plane part -(kz_t/n_t) k_hat_par

  r_xyz[m] = r_s[m] s_hat_m + r_p[m] p_hat_m^ref         (incidence medium: n_i, kz_i)
  t_xyz[m] = t_s[m] s_hat_m + t_p[m] p_hat_m^trn         (exit medium: n_t, kz_t)
  ```

  On a propagating order of a lossless medium of either sign `kz/n = cos(theta_m)`: a double
  negative's root has `Re(kz) < 0` and its `n < 0`, and the ratio stays positive. So `r_xyz`, `t_xyz` and every field
  are physical vectors whatever the signs.
* **At normal incidence `r_p = -r_s` and `t_p = t_s` at every interface.** Measured to 4.7e-16 and
  7.8e-16 for lit and far double negatives and a far magnetic metal, on the conical, 2D and 4ff
  paths, ETM and SMM *(study §5)*.
* **The scattering matrix.** Its p block holds port variables -- the p wave's magnetic field,
  `-1j (n/mu) a_p` in its own medium -- so with `delta` the unit vector on the incident order's p
  channel, `S21 @ delta` (the transmitted column that channel excites) is
  `t_p (n/mu)_exit / (n/mu)_lit`, with the signed indices
  ([`conventions.md`](conventions.md#the-result-surface)). A glass grating between a lossless
  `(-3, -0.5)` lit side and glass, its public matrix scaled to flux units on the open channels, is
  unitary to 1.6e-15 (conical, 8 + 8 channels) and 7.5e-15 (2D, 16 + 18) *(study §5)*.

**The incident s/p basis from a negative-index side.** The incident polarization is written in
the incident order's frame, which follows `k_par`; from an NPV side `k_par` points along
`phi + pi`, so at the same `(theta, phi)` both `s_hat` and `p_hat` are the negatives of a
positive-index medium's. The same `polarization=(a_s, a_p)` therefore gives the same light -- the
same power direction and the same polarization ellipse -- with the field negated. Lit at 20
degrees, `phi = 0`, from `(-2, -1)` and from its double-positive twin `(2, 1)` *(study §4)*:

| `polarization` | incident `E` from `(-2, -1)` | incident `E` from `(2, 1)` | Poynting vectors agree to |
| --- | --- | --- | --- |
| `(1, 0)`, s | `(0, -1, 0)` | `(0, +1, 0)` | 3.0e-17 |
| `(0, 1)`, p | `(+0.9397, 0, +0.3420)` | `(-0.9397, 0, -0.3420)` | 0 |

To ask for a given field *vector* from an NPV side, negate both Jones components.

**The incident wave from a negative-index side** is `E = a_s s_hat + a_p p_hat`, `H = k x E / mu`,
with `k_par` from §7 and `kz` the lit half-space's own root. In a double negative its phase moves
*away* from the structure while its power arrives. Lit from the top at `theta = 0.4`, `phi = 0.6`,
`psi = 0.7` *(study §5)*:

| lit medium | `Re(k_z)` (public frame) | `Im(k_z)` | `S_z` | `abs(H - k x E/mu)` | `abs(k.E)` |
| --- | --- | --- | --- | --- | --- |
| `(-3, -0.5)` | `+1.1281`: phase up | `0` | `-1.1281`: power down | 1.7e-16 | 4.5e-18 |
| `(-3+0.1i, -0.5+0.05i)` | `+1.1291` | `-0.0886`: decays toward the structure | `-1.1262` | 4.2e-17 | 8.5e-17 |

The coherent door (`excitation={side: PlaneWave(...)}`) builds the same wave with its own basis
(`illumination._plane_basis`) and agrees with the ordinary solve to 3.7e-15 in power and 1.5e-14
in field, lit from a lossy or lossless double negative on either side, at `(0.35, 0.6)` and at
normal incidence with `phi = 0.6` and `2.5` *(study §8)*.

**Where each diffraction order's light goes.** `order_kxy()` returns each order's *phase*
wavevector `k_par,m`. In a lossless half-space of signed index `n`, order `m` carries its power
away from the interface at the polar angle `theta_m` with

```text
sin(theta_m) = k_par,m / n          (signed in the plane, positive toward +x)
```

so in a negative-index half-space each beam leaves on the *other* side of the normal from where
its `k_par,m` points: the transmitted orders of a far double negative, and the reflected orders
in a lit one. Measured from the field the solve returns -- the outgoing field sampled over one
period, one Fourier component per order, each order's own Poynting vector -- on a TE grating
(700 nm period, `theta = 10` degrees) with the lossless `(-2, -1)`, `n = -sqrt(2)`
*(study §5)*:

| far side, transmitted | `m = -2` | `m = -1` | `m = 0` | `m = +1` |
| --- | --- | --- | --- | --- |
| `kx` | -1.1681 | -0.4538 | +0.2605 | +0.9748 |
| beam angle (deg) | +55.69 | +18.72 | -10.61 | -43.57 |

| lit side, reflected | `m = -1` | `m = 0` | `m = +1` | `m = +2` |
| --- | --- | --- | --- | --- |
| `kx` | -0.9599 | -0.2456 | +0.4687 | +1.1830 |
| beam angle (deg) | +42.74 | +10.00 | -19.36 | -56.77 |

Each matches `arcsin(kx / n)` to 3.6e-14 and 1.4e-14 deg, and every order carries its power away
from the interface. The specular reflection lit at 10 degrees along `+x` leaves at `+10.00`
degrees: it goes on along `+x`, as the light did.

---

## 9. Layers: which of a mode's `+-q` is "down"

A layer keeps both roots of every mode, so which one the recursion calls `+` is a choice of basis:
in exact arithmetic it moves no answer, only conditioning. SMM's cascade's labels are free that
way. ETM's are not, because ETM writes each layer through the *transmission* from its downward
modes into the region below ([`connecting-algorithms.md`](connecting-algorithms.md) §10): its `+`
modes must be the waves that region receives -- and so must those of the field state SMM hands to
reconstruction, which is ETM's state and takes ETM's labels. For an ordinary layer that is
Maystre's root, the half-space's own. For a double negative Maystre's root advances its phase
downward and carries its power *up*, so a double-negative layer resting on its own medium -- no
interface at all -- made ETM's interface factor exactly singular.

In the language of `q = -i*kz` (§1) the two cuts read: Maystre's `Re(kz) + Im(kz) > 0` is
`Re(q) - Im(q) > 0`, and the mirror `Im(kz) - Re(kz) > 0` is `Re(q) + Im(q) > 0` -- the form
[`connecting-algorithms.md`](connecting-algorithms.md) §11 writes. A "tie" below is the choice on
a lossless propagating mode, where `Re(q) = 0` and only the sign of `Im(q)` is left.

| layer | ETM's `+` root | SMM's `+` root |
| --- | --- | --- |
| no double-negative region | Maystre's cut, with the growth cap of [`connecting-algorithms.md`](connecting-algorithms.md) §10 (unchanged) | decaying; tie `Im(q) <= 0` (unchanged) |
| one uniform double negative | the mirror cut -- the half-space's own rule -- with the cap | decaying; tie `Im(q) >= 0` |
| patterned, holding a double-negative region | per mode: the root carrying power down, where the mode is propagation-dominated (`\|Im(q)\| > \|Re(q)\|`) and its flux resolvable; otherwise decaying; then the cap | decaying; a lossless tie broken by the flux |

The engine reads which rule applies from each layer's materials
(`CoreFactorizationMixin._layer_double_negative`) and hands the kernels one
`DoubleNegativeLayer(holds, uniform)` per layer; a layer without a double negative keeps its
arithmetic structurally. **For a tensor material** the test reads its three diagonal entries in
the lab frame (or the three principal values when the material is given as three values): a
region counts as double negative when all three of `eps`'s and all three of `mu`'s have negative
real parts. A rotated tensor is read by its lab-frame diagonal, not by its eigenvalues, so a
hyperbolic tensor whose lab diagonal happens to be all negative counts as double negative.
Measured with the rules on and switched off (every layer reported free of double negatives,
which is develop's labelling) *(study §6)*:

| scenes | rules | ETM `LinAlgError` | SMM `LinAlgError` | `abs(ETM - SMM)` where both solved |
| --- | --- | --- | --- | --- |
| 56: a uniform double-negative layer on its own medium, 4 media, TE/TM/conical/2D/4ff | on | 0 | 0 | 5.6e-16 |
| | off | 20 | 4 | 3.6e-15 |
| 96 random patterned layers (double-negative regions in glass or a metal and the reverse, on the double negative or glass, `theta` 0.3, 0.5+0.2i, 0.7) | on | 0 | 0 | 9.7e-13 |
| | off | 0 | 0 | 2.0e-10 |

and on [`connecting-algorithms.md`](connecting-algorithms.md) §10's grating with its block made
the double negative `(-25, -1)` -- a wide block resting on its own medium; every amplitude of every
polarization the result carries:

| `abs(ETM - SMM)` | TE | TM | conical, `phi = 0.3` |
| --- | --- | --- | --- |
| `theta = 0.5`, rules on / off | 1.2e-14 / 6.6e-07 | 5.5e-15 / 1.0e-05 | 6.4e-14 / 1.4e-05 |
| `theta = 0.5+0.3i`, rules on / off | 4.0e-14 / 7.5e-07 | 1.2e-14 / 2.7e-04 | 1.5e-13 / 1.2e-04 |

Why per mode for a patterned layer, the five label rules that were compared, and the cost:
[`connecting-algorithms.md`](connecting-algorithms.md) §11. The 4ff path has always split a
layer's modes by flux (`split_forward_backward`); its threshold is now counted in ulp of the
working precision, so a lossless patterned double negative is no longer refused at single
precision on JAX and PyTorch.

### Pendry's perfect-lens pair

A uniform layer of `(eps, mu)` beside a medium of `(-eps, -mu)` -- the perfect lens of Pendry
(2000), `(-1, -1)` in vacuum being the textbook case -- has, on every evanescent order, two
admittances that are exact negatives. **A single interface of such a pair is singular**: `r` has
a pole at every evanescent `k_par` (a surface mode on each). **A finite layer is not**: it
transmits each evanescent order with Pendry's finite amplification, and SMM reproduces that closed
form. Two limits must be kept apart:

* **Per order, at a fixed `k_par`**, the lossless plane-wave answer *is* the limit of the lossy
  one: the answer is linear in the loss as the loss goes to zero.
* **For the whole problem, as `truncation_order` grows**, there is no limit: the amplification
  grows without bound with `k_par`, so wherever a patterned layer excites evanescent orders the
  lossless answer keeps moving with the truncation, and only loss bounds it (D. R. Smith et al.,
  *Appl. Phys. Lett.* **82**, 1506 (2003)).

ETM's interface factor is exactly zero on those orders; where round-off leaves it non-zero ETM
still agrees with SMM, and where the zero is exact it raises a `ValueError` that names the pair
and points to SMM (`engine._perfect_lens`) -- whichever of the two is the double negative, the
layer or the half-space it rests on. The measurements are in
[`connecting-algorithms.md`](connecting-algorithms.md) §11.

### Two half-spaces that are each other's negation: refused

The single interface is a problem of its own when both sides are semi-infinite: `(eps, mu)` above
`(-eps, -mu)`, both lossless, glass `(2.25, 1)` on `(-2.25, -1)`. The two media share `kz**2`, so
an order is evanescent in both or in neither, and on an evanescent order both take the same
decaying root (the two cuts agree wherever `Re(kz**2) < 0`), so `kz/mu + kz/(-mu) = 0`: the
reflection has a pole at every evanescent `k_par` and the scattering matrix does not exist. The
time-harmonic problem has no solution. B. Gralak and D. Maystre (*C. R. Physique* **13**, 786
(2012)) show it for the `-1` half-space against vacuum: with absorption `gamma` the reflection and
a surface mode scale like `1/gamma`, there is no `gamma -> 0` limit, and in time the surface
modes grow linearly; B. Gralak (*C. R. Physique* **21**, 343 (2020)) reviews why the low-absorption
limit fails at that contrast. M. Cassier, C. Hazard and P. Joly (*Commun. PDE* **42**, 1707 (2017);
**47**, 1217 (2022)) and M. Cassier and P. Joly (arXiv:2504.00235 (2025)) prove the transmission
problem ill-posed when both contrasts are `-1`.

Uniform layers of the two media change nothing. A layer of `(-eps, -mu)` cancels an equal
thickness of `(eps, mu)` (J. B. Pendry and S. A. Ramakrishna, *J. Phys.: Condens. Matter* **15**,
6345 (2003)): its first-order system is the negative of the other's, so its transfer over `d` is
the other's over `-d`, and any stack of the two media between the two half-spaces is the bare
interface, moved.

**Only the evanescent orders are singular.** On a propagating order the two cuts take opposite
roots, the mirror's `-kz` against Maystre's `kz`, so the admittances `kz/mu` and `(-kz)/(-mu)`
are equal and the order is matched: `r = 0`, `t = 1`. Uniform layers couple no two orders, so a
truncation that retains no evanescent order has a regular system and an exact answer.

**What a solve does without the refusal.** Glass on `(-2.25, -1)`, wavelength 500 nm, period
300 nm, `theta = 0.2`: of the retained orders 5 of 7 (TE, TM, conical) and 7 of 9 (2D, 4ff) are
evanescent in both media, while one first order propagates in both (the `-1` order lit from
glass, the `+1` lit from the double negative) *(study §12)*. Every path, ETM and SMM, either side
lit; nothing between the half-spaces, or a glass layer, a `(-2.25, -1)` layer, or both. With the
refusal switched off *(study §12)*:

| between the half-spaces | NumPy | JAX | PyTorch |
| --- | --- | --- | --- |
| nothing (`solve_conv` with no layer), 20 scenes | `LinAlgError`, 20 | `FloatingPointError` (non-finite), 20 | `LinAlgError` 16; a finite answer 4 (2D and 4ff SMM) |
| layers of the two media, 60 scenes | SMM a finite answer, 30; ETM a finite answer 16 (conical, 2D, 4ff), the perfect-lens message 14 (TE, TM, 2D) | the same | SMM a finite answer, 30; ETM a finite answer 14, the perfect-lens message 16 |

The finite answers look like the matched propagating answer -- `R` at most 9e-20 and `T` within
1e-06 of 1 on NumPy -- beside evanescent amplitudes that are round-off: on the 4ff path, five of
SMM's six finite NumPy answers carry 0.0089 to 2.7 in orders a uniform stack does not excite
(PyTorch: four, 1.1 to 21). Until the refusal landed, ETM's perfect-lens message also called such
a stack finite; it now never does (`test_the_lens_message_never_calls_two_half_spaces_finite`).

With a loss `delta` in the double negative and its layer, every one of the 80 scenes is solved
*(study §12)*. The bare interface's `R` falls as `delta**2` -- 1.5e-26 .. 2.4e-26 at
`delta = 1e-12`, 1.5e-08 .. 2.4e-08 at 1e-3 -- with `T = 1` to 7.1e-08 at most: the limit of the
propagating orders is `r = 0`, `t = 1`. ETM and SMM agree to 1.6e-15 everywhere except on the 4ff
path with both media's layers between, where the nearly singular system parts them by 4.6e-10 at
`delta = 1e-12` and 3.1e-08 at 1e-9. The evanescent orders have no limit. Lit by an evanescent
wave from glass (`theta = pi/2 - 0.5i`, a glass layer between, SMM) the incident order sits on the
pole, and `delta * max|r_s|` is 0.0766 on the TE path and 0.0226 on the 2D and 4ff paths, the same
to the four digits printed from `delta = 1e-3` to 1e-12; at `delta = 0`, with the refusal
switched off, the solve returns `max|r_s|` of 2e+15 (TE) and 4e+14 (2D, 4ff).

So the exact lossless pair is refused before any algorithm runs wherever the truncation retains
an evanescent order -- 80 of the 80 scenes above on each of NumPy, JAX and PyTorch *(study §12)*
-- naming both media, how many of the retained orders are evanescent, and the sources above
(`engine.CoreFactorizationMixin._refuse_half_space_negation_pair`):

```
ValueError: the top half-space (eps = 2.25+0j, mu = 1+0j) and the bottom half-space
(eps = -2.25+0j, mu = -1+0j) are lossless and each other's exact negation, with no layer between
them, and 5 of the 7 retained orders are evanescent. On an evanescent order that pair has no
time-harmonic solution. ...
```

The same 80 scenes with no evanescent order retained -- the specular order alone
(`truncation_order=0`), or a 3000 nm period at the same truncation -- are solved, not refused:
160 of 160 on each backend, `R` at most 6e-30 and `T` within 2e-15 of 1, every amplitude within
1.4e-12 of the same scene with a loss of 1e-12 *(study §12)*. Until the second follow-up (§17)
the refusal stopped these too, saying that retained evanescent orders made the system singular
when there were none.

The test is the values' -- both half-spaces lossless (`Im == 0` exactly) and each other's negation
to 1e-12, every layer between them a uniform layer of one of the two, and at least one retained
order with `Re(kz**2) < 0` on the solve's own in-plane wavevectors, which is also the test for a
complex `theta` (the two cuts take the same root exactly there); an order exactly at `kz**2 = 0`
is a Rayleigh point and left to the solve's handling of one -- so a pair with any loss, a real
mismatch of 1e-9, a third medium between them (vacuum, a layer with the right `eps` and the wrong
`mu`, a uniaxial or sheared tensor, a grating) and Pendry's lens between other media are all
solved as before. It reads values, so it runs eagerly and under an un-jitted `jax.grad`; under
`jax.jit`, where the media or the in-plane wavevectors are traced, there is nothing to read, and
the traced solve returns whatever the singular system gives. A non-magnetic scene cannot hold the
pair and reads nothing.

---

## 10. What is refused, and what is only flagged

| input | what happens | why |
| --- | --- | --- |
| an **active** double negative that propagates: `Re(eps) < 0`, `Re(mu) < 0`, `Re(eps*mu) > 0`, and `Im(eps) < 0` or `Im(mu) < 0` | `ValueError` when the solver is built (`wavebit.rcwa(...)`) or when `top` / `bot` is assigned, either side, lit or not; also as a `solve_conv` half-space pair, when that solve is called | which root is physical depends on the medium's dispersion, not on `eps` and `mu` at one frequency (§3; Skaar 2006, 2010; Nistad and Skaar 2008). The limiting absorption that fixes the passive root says nothing about gain, and the backward-wave test behind the trigger holds for passive media only (Cassier, Joly and Kachanovska 2017, Remark 3.35) |
| the same, under `jax.jit` | no refusal can read a tracer, so the medium is solved, every order with the mirror root; `solver.gain_cutoff_band()` flags only the orders in the band near grazing (below). The other orders are solved too, and their numbers have no physical basis either, with nothing to say so | check such a material once eagerly, outside the trace: building the solver with concrete values raises |
| an active double negative that does not propagate, `Re(eps*mu) <= 0` (`(-1-1.2i, -1-1.2i)`: `R = 25`) | accepted, as before; may raise `GainCutoffWarning` | both cuts take the same root wherever `Re(kz**2) < 0` (§5), so there is no choice between the rules to make. The root is Maystre's, "the gain switched on from zero", which is the convention every gain half-space gets ([`branch-cuts.md`](branch-cuts.md) §8), not a physical guarantee: `R = 25` is what that convention gives (§18 says how far it can be trusted) |
| an active medium that is not a propagating double negative (a magnetic metal with gain, `(-2-0.01i, 0.1-0.5i)`; the double negative of the row above) | accepted; its sign, `s = +1` and `s = -1`, sets its `t_p` and, lit, its `k_par` (`kx = +0.1939` and `-0.3036` at `theta = 0.3`) *(study §7)* | its root is Maystre's (it does not propagate, or is not a double negative), and its sign is the convention of §6 |
| a `GaussianBeam` launched from a negative-index side | `ValueError` naming the alternatives | its angular spectrum, transversality and focal phase are written for waves whose phase and power travel together. Build the beam from `PlaneWave` sources, which are supported. (A beam needs a lossless incident medium, and a lossless negative-index medium is always a double negative, which is what the check reads) |
| a real `theta` in a lit medium that carries no wave (`Im(eps*mu) = 0`, `Re(eps*mu) <= 0`) | `ValueError` | `N = 0`: no real angle names a direction there |
| two lossless half-spaces that are each other's negation, `(eps, mu)` and `(-eps, -mu)`, with nothing between them but uniform layers of the two | `ValueError` when it is solved with an evanescent order retained, every path, either algorithm, naming both media and how many orders are evanescent; silent under `jax.jit`. A truncation that retains only propagating orders is solved (`r = 0`, `t = 1`) | the interface reflects with a pole at every evanescent `k_par` and the time-harmonic problem has no solution there (§9; Gralak and Maystre 2012; Cassier, Hazard and Joly 2017, 2022; Cassier and Joly 2025). Any loss is solved |

The messages *(study §7)*:

```text
`bot` is an active negative-index half-space: Re(eps) < 0 and Re(mu) < 0, Re(eps*mu) > 0 so
waves propagate in it, and Im(eps) < 0 or Im(mu) < 0. A passive double negative is solved with
its own root ... (J. Skaar, Phys. Rev. E 73, 026605 (2006); B. Nistad and J. Skaar, Phys. Rev.
E 78, 036603 (2008)). Put it in `structure` as a layer -- a layer keeps both roots -- with an
ordinary half-space beyond it.

GaussianBeam requires a lossless positive-index isotropic incident medium; the top half-space is
a negative-index one (Re(eps) < 0 and Re(mu) < 0). ...

real-angle incidence requires a propagating incident wave: a lossless lit half-space must carry
one (Re(eps*mu) > 0), and in an absorbing one abs(theta) must be less than pi/2 radians
```

Gain in `mu` alone (`(-2+0.1i, -1-0.01i)`) is refused the same way. So is every double negative
whose `Im(eps)` and `Im(mu)` have opposite signs: with both real parts negative,
`Re(eps*mu) = Re(eps) Re(mu) - Im(eps) Im(mu)` is then positive, so it always propagates. The
message calls such a medium active, and it need not be: B. Gralak (*C. R. Physique* **21**, 343
(2020), §4) shows that the effective `Im(mu)` of a passive, homogenised medium can take either
sign. It is outside the local passive model the root rule assumes, and no root serves it: for
`eps = -2+0.1i`, `mu = -1-0.5i` (`kz**2 = 2.05+0.90i`) the mirror root `-1.4644-0.3073i` grows
away from the interface while it carries power away, and Maystre's `+1.4644+0.3073i` decays but
carries power toward it *(study §13)*. Refusing it is the honest answer for either reading.

**Why an active propagating double negative has no answer here.** J. Skaar (URSI EMTS 2010,
arXiv:1008.2874) shows that the sign of an active medium's index is not fixed by `eps` and `mu`
at one frequency, and that even weak gain at oblique incidence can leave `kz` without a meaning
at a real frequency. Y.-F. Chen, P. Fischer and F. W. Wise (*J. Opt. Soc. Am. B* **23**, 45
(2006)) find, from a causality analysis, that adding gain to a medium with negative `eps` and
`mu` can make its index positive. No rule in this solver could read that from two numbers.

**Why a layer of an active double negative is solved.** A finite layer keeps both roots of every
mode, so no root has to be chosen for it, and the sign of an active medium's index does not
matter for a slab (B. Nistad and J. Skaar 2008; A. Lakhtakia, J. B. Geddes III and T. G. Mackay,
*Opt. Express* **15**, 17709 (2007), who show that the choice matters only for the half-space,
which they call "not realistic"). That is why the refusal message suggests a layer.

**The gain band of a double negative is the mirror image.** Near grazing, a gain half-space's
plane-wave answer stops describing a finite beam, and `GainCutoffWarning` /
`gain_cutoff_band()` mark the band `abs(Re(kz**2)) < 2 abs(Im(kz**2))` on the gain side of the real
axis ([`branch-cuts.md`](branch-cuts.md#gain-near-grazing)). Gain *lowers* `kz**2` in an ordinary
medium and *raises* it in a double negative, so the gain side is
`where(double_negative, Im(kz**2) > 0, Im(kz**2) < 0)`. Over `k_par**2` from 0 to 9, 300 random
double negatives (either sign of loss in either factor) have exactly the band of their
double-positive mirrors; the passive ones are never flagged; the active `(-2-0.1i, -1-0.1i)` is
flagged on 121 of 901 orders, all with `Im(kz**2) > 0` -- where testing `Im(kz**2) < 0` as for an
ordinary medium flagged none *(study §7)*. Under `jax.jit`, glass at 60 degrees onto that active
medium, `gain_cutoff_band()` flags 1 of 7 orders, while eagerly the same medium is refused
*(study §9)*. The other six orders are solved, unflagged, with no physical answer behind them --
which is why the band is not a substitute for the eager refusal.

---

## 11. Backends, tracing and gradients

* **Every decision is a `where`.** The signed index, the root, the sign of `k_par`, the frame
  azimuth at normal incidence and a layer's label rule are all selections, never Python branches
  on values. A scene whose materials are concrete and hold no double negative stages no `where`
  for one: what is concrete stays concrete inside a trace. A *traced* magnetic half-space does
  stage the signed index's and the lit `k_par` sign's `where`s, even when its value is ordinary
  (§13 counts what that costs: round-off).
* **One compile serves both signs of index.** `jax.jit` over the lit medium's `Re(eps)` on a
  conical grating at 0.35 rad matches NumPy to 1.9e-15 at `Re(eps) = -3` (a double negative) and
  to 3.8e-15 at `+3` (a mu-negative medium, `s = +1`), from one compilation *(study §9)*.
* **Backends agree.** Eager JAX and PyTorch against NumPy on a conical grating, far double
  negative / lit / lit at `theta = 0`: 2.0e-15, 2.5e-15, 1.9e-15 (JAX) and 7.5e-15, 2.7e-15,
  2.6e-15 (PyTorch) *(study §9)*.
* **Gradients agree with finite differences** (`h = 1e-6`): `jax.grad` of the far side's
  `dR/dRe(eps)`, `+0.050237` (to 6.6e-12), of a lit double negative's `dR/dtheta`, `+0.111502`
  (to 1.8e-10), and PyTorch `backward` of the first (to 6.6e-12) *(study §9)*.
* **Two gradients at exactly normal incidence** were wrong on develop for every medium and are
  right now: AD through the frame of an order with `kx == 0` dropped the frame's derivative on
  the 2D paths (`order_azimuth` now carries `-kx/ky`), and PyTorch gradients through a
  normal-incidence `PlaneWave` were NaN (`_plane_basis` now reads `|k_par|` as a projection). The
  values were and are the same; `tests/test_order_azimuth.py` and
  `tests/test_coherent_illumination.py` pin the derivatives.

---

## 12. How it was checked

Every oracle below needs **no root rule of its own**, or one fixed by physics alone (decay, and
outgoing power when lossless):

| oracle | what it pins | agreement |
| --- | --- | --- |
| closed-form Fresnel `r`, `t`, `R`, `T`, s and p, far side, 0 to 80 deg, 5 paths x 2 sides x ETM / SMM | the root, the index sign (through `t_p`) | `(-2+0.1i, -1+0.1i)` 2.0e-15; `(-3+0.1i, -0.5+0.05i)` 1.6e-15; lossless `(-2, -1)` 3.2e-15; `(-6+0.3i, -1.5+0.05i)` 2.3e-15; magnetic metal 1.0e-15; `eps = mu = -1+1i` 6.7e-16 *(study §8)* |
| the same, **lit** side, 0/20/35/60 deg | the lit root, the incident normalisation, `t_p` through the signed `n1` | `(-3+0.1i, -0.5+0.05i)` 1.3e-15; lossless `(-3, -0.5)` 1.0e-15; `(-2+0.1i, -1+0.1i)` 8.9e-16 *(study §8)* |
| a grating on the half-space against the same grating on a **16 um lossy slab** of the medium (a layer keeps both roots) | the half-space root, order by order | `R` per order 4.0e-16; the half-space's `T` against the power the slab takes in 1.7e-15 *(study §8)* |
| the **lossless limit**: slabs of `(-2+i delta, -1+i delta)` | that the lossless half-space is the limit of the lossy one | `delta = 1e-5` (a 128 mm slab) is 1.9e-05 away; Richardson on 1e-4 and 1e-5, 2.3e-08 *(study §8)* |
| the incident field's **Poynting vector** | that `theta`, `phi` are the light's direction | 3.8e-14 deg; negative refraction to 7.1e-15 deg *(study §4)* |
| each order's **Poynting vector** in a negative-index half-space | where diffracted light goes | 3.6e-14 deg (transmitted), 1.4e-14 deg (reflected) *(study §5)* |
| **reciprocity**: lit from a lossless `(-3, -0.5)`, order `+-1` into glass, against glass lit along that order's reverse | the lit side's direction and normalisation, by a theorem | 1.7e-15 *(study §8)* |
| **unitarity** of the public scattering matrix between lossless ports | the port normalisation with signed indices | 1.6e-15 conical, 7.5e-15 2D *(study §5)* |
| Maxwell on the incident wave, `H = k x E / mu`, `k.E = 0` | the incident basis | 1.7e-16 *(study §5)* |
| **fmmax 1.7.1**, `tests/references/test_fmmax_dng_golden.py`: unpolarized `R` and `T` of every order, Richardson over 16/32/64/128 samples per pixel | a solver that made none of these choices (its rule, `Im(kz) > 0`, is right for every lossy medium) | worst order per case: 1D grating on a DNG 5.9e-14 (bound 4.7e-12); 2D 7.5e-14 (5.6e-12); DNG pixels in the grating on a DNG 6.1e-12 (3.9e-10); lit from a DNG 2.7e-14 (2.7e-12); ETM and SMM, the `theta` and direct-`k` doors *(study §8)* |
| the **mirror medium** (A. Lakhtakia's conjugation symmetry, 2004, as V. Grunhut and R. A. Depine 2011 apply it to an interface): the far side `(eps, mu)` against `(-conj(eps), -conj(mu))`, glass lit at 0, 0.3 rad and 60 deg, lossy, lossless and evanescent far sides, 5 paths x 2 sides x ETM / SMM | that a double negative takes the mirror root and the signed index, by a theorem | no layer between: 2,376 of 2,376 arrays exactly the conjugate, bit for bit; every layer mirrored too: 7.8e-15 *(study §8)* |
| **PyMoosh 4.0.1**, `tests/references/test_pymoosh_dng_golden.py`: flat stacks with a **lossless** double negative as substrate, lit superstrate, inner layer and all three; `r`, `t`, `R`, `T`, TE and TM, four `kx` each | the lossless root and the index sign in `t_p`, where fmmax has nothing to read; PyMoosh's rule is Wavebit's on every lossless medium | 8.8e-15 over 32 rows x the TE/TM, 2D and 4ff paths x ETM / SMM (bound 1e-13); each double negative beyond the lit side replaced by its positive twin misses by 1.8 to 2.9 *(study §8)* |
| ETM against SMM on double-negative layers | the label rules | §9 |
| backends, `jit`, gradients | traceability | §11 |

fmmax as shipped differs from its corrected run by 2.3e-09 .. 3.9e-09 *(study §8)*: its centring
phase is computed in float32, which the golden's generator removes. Every negative-index medium in
the golden is lossy with `|mu| != 1`, the regime where fmmax's rule is physical (§14 says why
lossless ones are not). The bound is the extrapolation's own spread plus 1e-12 of round-off, the
tolerance the test applies. The golden's test file records the run made when it was frozen
(numpy 2.4.6, in the env that has fmmax), which differs from the study's numpy 2.5.1 only in the
last digit (7.6e-14 and 6.2e-12 there).

### The tests, by behaviour

Run them one file at a time, from the repository root:
`python -m pytest tests/test_negative_index_half_space.py` (and likewise for the others). The fmmax
golden's test needs no fmmax: the numbers are frozen in `tests/references/golden_fmmax_dng.npz`;
only regenerating them needs fmmax 1.7.1 and its environment (`tests/references/README.md`). The
same holds for PyMoosh 4.0.1 and `tests/references/golden_pymoosh_dng.npz`.
`tests/test_gain_cutoff.py` changed only in two comments.

| behaviour | tests (`tests/test_negative_index_half_space.py` unless named) |
| --- | --- |
| the far-side root against closed forms, every path, side and algorithm | `test_a_negative_index_far_side_matches_the_closed_form`, `test_other_negative_index_media_match_the_closed_form`, `test_a_passive_far_side_says_nothing` |
| the lit-side root and normalisation | `test_a_lit_negative_index_half_space_matches_the_closed_form` |
| the mirror rule: bits, tie, sweep, other media untouched | `test_the_mirror_medium_conjugates_every_reflection`, `test_mirroring_everything_below_the_light_conjugates_it_through_layers`, `test_the_double_negative_rule_is_the_mirror_of_maystres_to_the_bit`, `test_maystres_tie_on_a_double_negative_takes_the_decaying_root`, `test_the_mirror_rule_on_a_sweep`, `test_one_negative_real_part_keeps_maystres_root`, `test_a_double_negative_too_lossy_to_propagate_is_solved` |
| slabs and the lossless limit (no root rule needed) | `test_a_negative_index_slab_matches_airy`, `test_a_grating_on_the_half_space_is_the_thick_slab_limit`, `test_the_lossless_half_space_is_the_vanishing_loss_limit`, `test_the_field_below_the_interface_is_the_slab_field` |
| the signed index, its axis case and its seams | `test_the_signed_index_is_the_product_of_the_two_roots`, `test_each_factors_root_is_maystres`, `test_t_p_is_t_s_at_normal_incidence_where_the_index_points_along_kz`, `test_on_the_axis_the_signed_index_is_the_principal_root`, `test_naming_mu_one_keeps_the_non_magnetic_index_and_amplitudes`, `test_t_p_is_continuous_across_a_real_part_seam`, `test_the_index_is_negative_and_t_p_follows_t_s`, `test_a_factor_on_maystres_cut_keeps_its_root_whatever_the_round_off` |
| a signed zero's root is the same on every backend | `tests/test_material.py::test_a_signed_zero_names_the_same_root_on_every_backend` |
| `theta` names the light: Poynting, refraction, reciprocity | `test_theta_and_phi_are_the_direction_the_incident_power_travels`, `test_light_from_a_negative_index_medium_refracts_negatively`, `test_a_grating_lit_from_a_negative_index_side_is_reciprocal` |
| continuity at `theta = 0` and for complex angles | `test_the_incident_field_is_continuous_through_normal_incidence`, `test_a_complex_angle_meets_the_real_one` |
| the coherent door | `test_the_coherent_door_agrees_with_the_ordinary_solve`, `test_the_coherent_door_lights_a_negative_index_side_as_the_ordinary_solve` |
| refusals and the gain band | `test_an_active_negative_index_half_space_is_refused`, `test_a_gaussian_beam_from_a_negative_index_side_is_refused`, `test_the_refusals_read_torch_values_eagerly`, `test_the_gain_band_of_a_double_negative_is_the_mirror_image`, `test_the_gain_band_flags_an_active_double_negative_under_jit` |
| `solve_conv` half-space pairs | `test_solve_conv_half_space_terms_are_read`, `test_a_solve_conv_pair_signs_the_index_it_is_solved_with`, `test_under_jit_the_scenes_own_pairs_sign_nothing` |
| backends, `jit`, gradients | `test_the_backends_agree`, `test_jit_through_a_lit_negative_index_superstrate`, `test_jit_and_grad_through_a_negative_index_substrate` |
| layer labels (`tests/test_negative_index_layers.py`) | `test_a_double_negative_layer_on_its_own_medium_is_that_medium`, `test_the_mirror_labels_are_the_half_spaces_roots`, `test_a_complex_angle_takes_the_mirror_too`, `test_a_patterned_double_negative_is_labelled_by_its_power`, `test_a_high_contrast_double_negative_block_on_its_own_medium`, `test_the_flux_labels_carry_power_down`, `test_smm_breaks_a_patterned_layers_tie_by_the_flux_too`, `test_the_flux_rule_keeps_decay_where_decay_dominates_and_the_cap`, `test_smm_no_longer_meets_a_backward_mode_at_its_gap`, `test_an_ordinary_layer_takes_no_new_rule` |
| what the engine reads from materials (`tests/test_negative_index_layers.py`) | `test_the_engine_reads_each_layer_from_its_materials`, `test_a_shape_layer_is_read_from_its_materials`, `test_three_principal_values_are_one_flag_and_solve_as_their_tensor`, `test_the_labels_travel_with_their_layers_through_the_mirrored_scene`, `test_solve_conv_with_the_scenes_own_matrices_is_solve`, `test_a_traced_double_negative_layer_is_selected_not_branched_on`, `test_the_4ff_split_reads_flux_at_the_working_precision` |
| Pendry's pair (`tests/test_negative_index_layers.py`) | `test_the_perfect_lens_pair_is_finite_and_smm_has_it`, `test_etm_names_the_perfect_lens_pair_where_it_cannot_write_it`, `test_a_lossy_lens_is_not_the_pair`, `test_etm_names_the_pair_when_the_double_negative_is_the_substrate`, `test_the_lens_message_never_calls_two_half_spaces_finite` |
| two half-spaces that are each other's negation (`tests/test_negative_index_layers.py`) | `test_two_half_spaces_that_are_each_others_lossless_negation_are_refused`, `test_the_negation_pair_with_no_evanescent_order_retained_is_solved`, `test_the_negation_pair_is_refused_on_every_backend`, `test_a_lossy_negation_pair_is_solved_and_has_no_lossless_limit`, `test_a_third_medium_between_the_pair_is_another_stack` |
| the lossless outside golden (`tests/references/test_pymoosh_dng_golden.py`) | `test_every_row_is_pymooshs`, `test_the_theta_door_from_a_double_negative_meets_the_same_rows`, `test_the_golden_tells_a_double_negative_from_its_mirror`, `test_the_golden_holds_the_scenes_the_cases_state` |
| the outside golden (`tests/references/test_fmmax_dng_golden.py`) | `test_every_order_is_fmmax_fft`, `test_the_only_gap_is_the_quadrature`, `test_fmmax_as_shipped_differs_by_its_single_precision_phase_only`, `test_the_golden_tells_a_double_negative_from_its_mirror`, `test_the_golden_tells_the_rules_apart`, `test_energy_is_accounted_for`, `test_the_golden_holds_the_scenes_the_cases_state` |
| derivatives at `theta = 0` (every medium) | `tests/test_order_azimuth.py::test_the_column_carries_its_derivative_along_kx`, `tests/test_order_azimuth.py::test_the_theta_derivative_at_normal_incidence_on_a_2d_grating`, `tests/test_coherent_illumination.py::test_torch_gradients_at_normal_incidence_are_the_one_sided_limit` |
| the refusal of a real angle in a medium with no wave | `tests/test_phase_incidence.py::test_a_lossless_lit_medium_that_carries_no_wave_is_refused_by_name` |

**Each test was checked against a mutant**: every new behaviour was reverted one at a time -- no
sign, a sign keyed on "double negative", `sign(Re(q))` alone, a strict tie, an unsigned `k_par`,
the frame not turned in each of the places that read it, the coherent basis on the principal
root, the gain band, the mirror cut switched off, each label rule -- and each revert fails at
least one test. The commit messages record the counts.

---

## 13. What changed against develop

Develop is `0dfc4ce`, the commit this work branched from.

### Media with `s = +1`: every number is develop's

A harness of 12 media with `s = +1` -- glass, a lossy
dielectric, a metal, a lossless metal, gain, ordinary magnetic, lossy magnetic, a magnetic metal
with `s = +1`, mu-negative lossy and lossless, magnetic gain, and `mu = 1` named -- run in this
tree and in a `git archive` of develop by the same code, every array compared byte for byte
*(study §10)*:

| group | arrays (refusal records) | byte-identical | not |
| --- | --- | --- | --- |
| NumPy: 5 paths x ETM / SMM x lit / far x `theta` 0, 0.3, 0.3+0.05i, 1.4 | 20,796 (90) | 20,796 | 0 |
| JAX eager: conical and 2D x ETM / SMM x lit / far, `theta = 0.3` | 2,616 (12) | 2,616 | 0 |
| PyTorch eager: the same | 2,616 (12) | 2,616 | 0 |
| the coherent door, `PlaneWave` from the medium, `theta` 0.3 and 0 | 60 (6) | 34 | 26, all at `theta = 0`, at most 2.8e-15 relative |
| `jax.jit`, `Re(eps)` of a magnetic half-space traced, lit and far | 36 (0) | 28 | 8, at most 7.6e-16 relative |

A refusal record is the class name of what was raised, so a reworded message does not count as a
change. Both exceptions are deliberate: at exact normal incidence the coherent door now turns the
frame rather than the polarization, which is what makes its gradients right (§11); and under
`jit` a traced *magnetic* half-space's signed index and lit-side `k_par` sign are runtime
selections, which XLA compiles with different rounding. Removing those `where`s restores
identity; they are required for a traced material.

Further changes for `s = +1` media outside the harness: at `complex_dtype=1` on the 4ff path a
mode whose flux lies between 1e-12 and the single-precision threshold of its energy is now decided
by attenuation (§9); AD at exactly `theta = 0` is now right on the 2D paths and through a
`PlaneWave` on PyTorch (values unchanged, §11); a value exactly on the negative real axis with a
`-0.0` imaginary part, handed in as a JAX array, now takes the root that sign names, `-i` times
`sqrt(-eps)`, as it always did on NumPy, PyTorch and Python numbers, where develop's JAX took
`+i` times it (§15); and some warning and refusal texts were reworded to the new semantics.

### Media with `s = -1`

| what | develop `0dfc4ce` | now |
| --- | --- | --- |
| a propagating double negative, far side | refused (`d421aa2`); before that, Maystre's root (`R = 928.78`, §3) | solved |
| a propagating double negative, lit side | refused | solved; `k_par = -N sin(theta)` |
| a double-negative layer under ETM | `LinAlgError` or a degraded answer (§9) | the SMM answer |
| `n_top`, `n_bot`, `n_inc` | principal `sqrt(eps*mu)` | its exact negation, `sqrt(eps)*sqrt(mu)` (except exactly on the negative real axis) |
| `t_p` of a far NPV half-space | `-t_s` at normal incidence | exactly negated: `t_s`. `r`, `t_s`, `R`, `T`, `r_xyz`, `t_xyz`, fields unchanged |
| incident `kx`, `ky` from a lit NPV medium at a real angle (a magnetic metal; develop refused a lit double negative) | `+N sin(theta)` | negated. Planar `R`, `T` unchanged; an asymmetric grating sees the mirrored order set |
| incident `p_hat` from a lit NPV medium | from the principal index | from the signed one; mixed polarizations differ |
| `theta = 0` from a lit NPV medium | frame at `phi` | frame at `phi + pi`: incident field and amplitudes negated |
| complex `theta` from a lit NPV medium | principal `n` | signed `n` |
| `eps = mu = -1+1i` | `R = 25` (ETM), `LinAlgError` (SMM) | `R = 0.04`, the closed form |
| an active double negative under `jit` | not flagged | `gain_cutoff_band()` flags its orders near grazing |
| `solve_conv` with an NPV `(eps, mu)` half-space pair on a non-magnetic scene | the scene's principal index | the pair's signed index, for that solve |
| a `GaussianBeam` from a negative-index side | refused: the half-space itself was | refused, with a message naming the alternatives |
| `n` of a magnetic medium with exactly one gain-metal factor (`Re < 0`, `Im < 0`) | principal `sqrt(eps*mu)` | `sqrt(eps)*sqrt(mu)` with each factor's root on Maystre's cut (§6): the principal root or its negation, whichever that product points along -- `(-2-0.01i, 0.1-0.5i)` keeps develop's `+0.6383+0.7826i`, a gain metal named with `mu = 1` is negated. `t_p` and a lit `k_par` follow; `kz`, `R`, `T`, `r`, `t_s` do not |

The log's section 10 prints what develop does with the double negatives: refused (`ValueError`)
on the far side for `(-2+0.1i, -1+0.1i)` and `(-2, -1)` and on the lit side for
`(-3+0.1i, -0.5+0.05i)`, `R = 25` / `LinAlgError` for `eps = mu = -1+1i`; and what this tree
returns: 0.001076681582, 0.0008665517772, 0.04 and 0.05284956599 (ETM and SMM agreeing).

---

## 14. What other tools do

Built from a survey of the tools below in 2026-10, each finding re-run independently before it
was kept: some **run** as black boxes on the standard table of §3, some read or documented only.
The columns refer to §3's table: *physical* is the right answer, *Maystre* is the growing root's
column, and *the gain twin* is `eps*mu` treated as a permittivity with `mu = 1` under Maystre's
root (0.002178010778 at 0 degrees, §3). Licences matter here: copyleft tools (GPL, LGPL, AGPL)
were studied by their documentation and their outputs only, never their source, so a root rule
stated for one of them is **inferred from its outputs** (marked *inferred*).

| tool (licence) | studied | permeability in a half-space | root rule | lossy DNG half-space | lossless DNG half-space | lit from a DNG |
| --- | --- | --- | --- | --- | --- | --- |
| RETICOLO V10 (CC-BY-4.0) | run | only by a route its manual forbids for half-spaces; runs silently | Maystre's cut on `eps*mu`; a global switch to the decaying cut | Maystre column by default; physical with the decaying cut, which then breaks gain media globally | Maystre column; the decaying cut right only by round-off | same as far side |
| fmmax 1.7.1 (MIT) | run; the frozen golden | general anisotropic solver only | `Im(kz) > 0`, ties keep the forward root | **physical** | Maystre column, with `T < 0` | lossy right; lossless incident mode carries negative flux |
| inkstone 0.3.15 (AGPL-3.0+) | run | yes | *inferred*: `Im(kz) > 0`, ties `Re(kz) > 0` (read off its transmitted field) | **physical** | Maystre column (the forward root), `T < 0`; `(-3, -0.5)` gives Maystre's 17.30 | lossy right; lossless `R = 1/R_true` |
| torcwa 0.1.4.2 (LGPL-3.0) | run | accepted, but `mu` acts only through `eps*mu` | *inferred*: `abs(Re q) + i abs(Im q)` (a model that reproduces its outputs) | the gain twin; lossy double-negative *layers* wrong too | right by coincidence (`\|mu\| = 1`); `(-3, -0.5)` gives `eps = 1.5`'s 0.0102 | impedance-blind, as on the far side |
| nannos 2.6.4 (GPL-3.0+) | run | `mu != 1` raises `NotImplementedError` | -- | refuses | refuses | refuses |
| S4 1.1 (GPL-2.0+), grcwa 0.1.2 (GPL-3.0) | documented, API checked | none | -- | cannot express | cannot express | cannot express |
| edmundsj `rcwa` (MIT), chiuhans `rcwa`, rcwa4d | run | yes (rcwa4d only after a patch) | principal root of `eps*mu` | Maystre column; edmundsj's `T` is consistent with it (`T < 0`, `R + T = 1`), chiuhans and rcwa4d normalise `T` without `mu` | Maystre column | Maystre column (edmundsj, chiuhans) |
| meent (MIT) | run | none (an index squared) | conjugated principal root | `R_s` the gain twin's; `R_p` not: its TM half-space admittance conjugates `kz` but not `eps`, which gives 0.01348 / 0.01657 / 0.07686 at 0 / 30 / 60 deg *(study §1)* | right by coincidence | -- |
| EMpy 2.2.3 (MIT) | run | none | none (principal `arcsin`) | the gain twin, or with `n < 0` the decaying root on the gain twin (459.13 at 0 deg, §3) | Maystre column with `n < 0` | -- |
| A_FMM | read, not run | none | Maystre's cut, prints a warning | -- | -- | -- |
| PyMoosh 4.0.1 (MIT) | run; the frozen golden for lossless flat stacks | yes | principal root negated when both real parts are negative (lit and last medium); inner layers `Im(kz) >= 0` | **physical** | **physical** | **physical**; its `theta` is the phase's, so from a lossless lit medium its `kx` is the opposite of Wavebit's |
| Byrnes `tmm` 0.2.0, `tmm_fast` 0.3.0 (MIT) | run | none (an index) | decay first, flux tie-break; asserts against gain | refuses (reads it as gain) | replaced silently by the positive twin (right at a half-space, wrong in a mixed stack) | refuses |
| gyptis 1.2.0 (MIT, FEM) | run | yes, per subdomain | principal root in its analytic annex and post-processing | Maystre column, `T < 0`, no warning; the FEM field crossing a DNG and its PML follows the forward root | -- | -- |
| Meep (GPL) | documented and run as a black box | dispersive `mu` | none: time domain | a long lossy Drude double negative ended by metal converges slowly toward the physical column; every absorbing layer inside the double negative (PML, absorber) diverges | -- | -- |
| COMSOL | documented | yes | the user's: its negative-index example sets the port's propagation constant and a PML scaling of -1 by hand | -- | -- | -- |

What the survey taught, and where Wavebit's rules come from:

* **No RCWA code defines what an angle means in a negative-index medium**, and none documents
  negative-index media. RETICOLO, fmmax and Wavebit's direct-`k` door all take the in-plane
  wavevector as the primary input.
* **`Im(kz) > 0` is the physical rule for every lossy medium of either sign** (fmmax, inkstone,
  and Depine and Lakhtakia's grating papers: R. A. Depine and A. Lakhtakia, *Phys. Rev. E* **69**,
  057602 (2004)). It has nothing to read on a lossless order, where the tie must be broken by the
  outgoing power, `Re(kz/mu) > 0` -- which none of the RCWA codes does. It also cannot be
  Wavebit's rule for every medium: for gain media it is not Maystre's root, which Wavebit keeps
  there ([`branch-cuts.md`](branch-cuts.md) §8, "the alternatives").
* **Only a rule that reads `eps` and `mu` separately can be right for a passive double negative**.
  PyMoosh's "negate when both real parts are negative" is the closest to Wavebit's, but it is
  applied to the index, not to a cut, and so it leaves every other medium on the principal root.
  For a passive NPV medium with one positive real part that root *grows*: glass onto
  `(-10+0.1i, 1+0.5i)` gives `R = 1.463` against the true `0.6834` *(study §1)*. That is a root
  failure, not a sign of the index -- Wavebit gets it right because Maystre's cut already takes
  the decaying root there (§5), and the signed index of §6 changes only `t_p` and `k_par`, never
  `R`. One of PyMoosh's solver paths also reads `eps` without `mu`.
* **PyMoosh is the oracle for lossless double negatives in flat stacks**
  (`tests/references/test_pymoosh_dng_golden.py`): substrate, lit superstrate, inner layer and
  all three, TE and TM, four in-plane wavevectors each, every path and both algorithms, to
  8.8e-15 *(study §8)*, compared at PyMoosh's `kx` because its angle is the phase's (§7). Its
  inner layers keep the principal root, so -- in PyMoosh's own outputs, which
  `tests/references/README.md` records -- a lossless `(-3, -0.5)` layer on the same medium
  returns `NaN`, and a `(-6, -1.5)` layer beside the impedance-matched `(4, 1)` returns
  `R + T = 3.10` at normal incidence: the label failure §9's rules fixed here.
* **Energy conservation catches none of these failures**; `T < 0`, a negative incident flux, a
  closed form or a lossy slab does (§3).

---

## 15. Where it stops

* **Lossless double negatives at and near normal incidence beside an ordinary or metal region**
  can be numerically singular on both algorithms: a few such patterned scenes leave ETM and SMM
  apart, identically before and after the label rules. It is not a labelling question, and
  `auto_detune` does not catch it, because nothing raises. **Lossless half-and-half mixtures on
  the conical path** (a double negative beside a positive-index medium) also leave ETM and SMM
  measurably apart; there a little loss is measured to close the gap. No remedy is measured for
  the first. [`connecting-algorithms.md`](connecting-algorithms.md) §11, "What it does not
  cover", has the numbers.
* **Li's inverse rule on a lossless double negative mixed with its sign-flipped neighbour.** A
  lossless `(-2, -1)` layer patterned with glass on the 2D path, where one column of `mu` is
  `+1` and `-1` in equal parts, raises `LinAlgError` under `factorization='li'`: the Toeplitz
  matrix of `1/mu` it inverts is exactly singular. `'laurent'` and `'nvf'` solve the same layer
  *(study §6)*.
* **The lossless perfect lens under a grating** has no limit in `truncation_order` (§9); its
  answer is a fixed-truncation answer and no warning says so. Under `jit` an exactly singular ETM
  step returns NaN rather than the `ValueError` it raises eagerly. If a grazing-order failure ever
  coincided with an exact lens pair, the lens message would be raised where `auto_detune` would
  otherwise have retried.
* **Hyperbolic and other partly negative tensors** keep Maystre's root and labels; so does a
  backward mode of an ordinary plasmonic layer. A layer's tensor is read by its lab-frame diagonal
  (§9), so a rotated hyperbolic tensor whose diagonal is all negative is labelled as a double
  negative.
* **A non-double-negative NPV medium with a lossless partner** at `k_par = 0` -- `eps = 0.1i`,
  `mu = -1` -- sits exactly on Maystre's cut and takes his tie, which disagrees with its signed
  index. Keying the mirror cut on `s = -1` instead of on "double negative" would fix it and move
  other media's bits; it was decided to key on the double negative.
* **Active media whose index points against their `kz`.** With each factor's root on Maystre's
  cut (§6), a gain-metal factor beside a strongly absorbing one that together propagate --
  `(-0.1-2i, 0.1+2i)` -- and an active double negative too lossy to propagate --
  `(-1-1.2i, -1-1.2i)` -- have `n` pointing against the normal-incidence `kz`, so `t_p = -t_s`
  at normal incidence: 5,839 and 2,903 of 400,000 random media *(study §3)*. A sign taken from
  that `kz` itself would agree everywhere (and keep every passive medium's bits); it was not
  adopted here. Non-magnetic gain metals keep the principal `index_from_eps`, and with it
  `t_p = -t_s` at normal incidence.
* **Near the negation pair nothing warns.** The refusal of §9 reads only an *exact* lossless
  pair. A tiny loss or a tiny real mismatch is solved without a word, and an excited evanescent
  order then grows like `1/delta`; on the 4ff path with layers of both media between, ETM and
  SMM part by 3.1e-08 at `delta = 1e-9` *(study §12)*. No warning marks the approach.
* **A gain-compensated negation pair is not refused.** A lossy double negative under a gain medium
  holding its exact negation -- imaginary parts negated too -- is singular in the same way: without
  a layer SMM raises `LinAlgError` on the TE, 2D and 4ff paths, and with a layer of either medium on
  the 2D and 4ff paths it returns round-off of order one and more in the evanescent orders (on TE it
  returns zeros there, or with a double-negative layer and a loss of 0.1 raises `LinAlgError`). The
  refusal tests `Im == 0` on all four values and so lets it through; beside a layer, ETM's
  perfect-lens message can still say that SMM solves it.
* **Each backend's square root is its own.** On the cut they now agree: the principal root of
  `complex(-1, -0.0)` is `-1j` in `cmath`, NumPy and PyTorch -- the lip the sign of the zero names
  (W. Kahan 1987) -- and `+1j` in JAX 0.11.0, whose `sqrt` puts both zeros on the upper lip, and
  `index_from_eps` turns JAX's root there (`backend.principal_sqrt`), so a value on the negative
  real axis carrying a `-0.0` imaginary part -- `eps`, or `eps*mu` on the axis for a magnetic
  medium -- gets the lower lip's root and the same sign of `t_p` on every backend *(study §13)*.
  Off the cut they are each their library's: `cmath.sqrt` and `np.sqrt` differ on 45 of 2,000
  random products `eps*mu`, by one ulp in a component on 38 and by two on 7 *(study §13)*, so the
  same medium handed in as a Python number and as a NumPy scalar can give indices an ulp or two
  apart (the mirror test of §12 keeps both as Python numbers for that reason).
* **Anisotropic half-spaces.** A half-space is isotropic. If tensor half-spaces are ever allowed,
  a switch per medium will not do: `eps = diag(1, 1, -1)`, `mu = diag(-1, -1, 1)` at `kx = 1.5`
  has `kz**2 = 1.25` for both s and p, and the root carrying power away is `-1.1180` for s and
  `+1.1180` for p *(study §13)*. Such a half-space needs its waves sorted one by one, by
  Poynting flux and decay (E. Yang, Q. Song and W. Cai, arXiv:2608.06185 (2026); M. M. Bay,
  S. Vignolini and K. Vynck, *Comput. Phys. Commun.* **273**, 108256 (2022), who find the sign of
  `Re(kz)` and the sign of the Poynting flux disagreeing even in a passive positive uniaxial
  crystal near total internal reflection).
* **Complex frequency.** Wavebit solves at a real frequency. A pole search at a complex one would
  need the per-medium rule kept, not one cut: for a lossless Drude double negative at
  `omega = 0.5+0.01i`, `Re + Im > 0` -- the cut P. Lalanne and J.-P. Hugonin state for complex
  frequency (arXiv:2609.10076 (2026)) -- takes `+1.499-0.050i`, which grows, while the mirror
  takes the decaying `-1.499+0.050i` *(study §13)*.
* **A purely imaginary `theta` from an NPV medium** turns the frame by `+1` (`Re(kx) = -0.0`)
  where `theta = 0` turns it by `-1`; the real-axis limit is continuous.
* **Gaussian beams from a negative-index side** are refused (§10), and **active double negatives**
  that propagate are refused (§10) -- deliberately, since no single-frequency answer exists. Under
  `jit` the refusal cannot run and only the grazing band is flagged (§10).
* **A caller holding only `eps*mu`.** A kernel called with `mu_bot` and no `eps_bot` takes
  Maystre's root ([`branch-cuts.md`](branch-cuts.md) §9). The engine always passes both on a
  magnetic scene.
* **`solve_conv` half-space pairs.** A pair that turns the lit side's sign holds only for that
  solve: afterwards the solver's own `reconstruct_field`, which reads the incidence live, refuses
  rather than mix the two scenes; the result's field door still works. `solve_conv` given
  matrices that are not the scene's, with a matching layer count, uses the scene's labels; that
  costs conditioning only, never the answer.
* **Untested:** the 2D path's relabelling of a cluster of grazing orders has no test of its own.
* **Not yet compared:** the published grating efficiencies on the substrate
  `(-5+0.01i, -1+0.01i)` -- Depine and Lakhtakia's sinusoidal grating (*Phys. Rev. E* **69**,
  057602 (2004)) and Depine, Lakhtakia and Smith's rectangular one, computed with the C method
  (*Phys. Lett. A* **337**, 155 (2005)) -- are candidates for a second outside golden.

---

## 16. Using it

A double negative is a `Material` with both real parts negative. Everything else is as for any
medium. Every value in the comments below is what the code prints *(study §11, which runs these
examples as written)*.

**As the substrate** (the worked example of §3). `result.R` and `result.T` are the requested
polarization's per-order efficiencies; at normal incidence s and p agree:

```python
import numpy as np
import wavebit

solver = wavebit.rcwa(
    top=2.25, bot=wavebit.Material(-2 + 0.1j, mu=-1 + 0.1j),
    wavelength=500., period=(1000., 1000.), truncation_order=(0, 0),
    structure=np.full((1, 2, 2), 2.25), thickness=[37.],
)
result = solver.solve()
solver.n_bot                                      # (-1.4147+0.1060j): the signed index
float(np.sum(result.R)), float(np.sum(result.T))  # 0.0010766816, 0.9989233184 -- not 928.78
```

**Lit from it** -- negative refraction (§7):

```python
solver = wavebit.rcwa(
    top=wavebit.Material(-2.0, mu=-1.0), bot=2.25,
    theta=np.deg2rad(20.), phi=0., polarization=(1., 0.),
    wavelength=500., period=[500.], truncation_order=0,
    structure=np.full((1, 1, 1), 2.25), thickness=[0.],
)
result = solver.solve()
kx = solver.order_kxy()[0][0].real          # -0.483690 = -sqrt(2) sin(20 deg)
np.degrees(np.arcsin(kx / 1.5))             # -18.8117: back across the normal
float(np.sum(result.R)), float(np.sum(result.T))   # 0.0010947, 0.9989053
```

**An absorbing lit medium**: `R + T` is not 1 there, and `power_entering` is what enters (§7):

```python
solver.top = wavebit.Material(-2 + 0.1j, mu=-1 + 0.1j)
result = solver.solve()
float(np.sum(result.R)) + float(np.sum(result.T))  # 1.0002263 (R 0.0011723, T 0.9990540)
float(np.sum(result.power_entering))              # 0.9990540
```

**The same light by its wavevector**: direct `kx` / `ky` are taken as given, in physical units
(here 1/nm), whatever the lit medium (§7). `kx = +0.48369*k0` would be the mirror image: light
arriving from the other side.

```python
k0 = 2 * np.pi / 500.
direct = wavebit.rcwa(
    top=wavebit.Material(-2.0, mu=-1.0), bot=2.25, kx=-0.48369 * k0,
    polarization=(1., 0.), wavelength=500., period=[500.], truncation_order=0,
    structure=np.full((1, 1, 1), 2.25), thickness=[0.],
)
float(np.sum(direct.solve().R))             # 0.0010947, as through theta
```

**A double-negative layer**, uniform or patterned, on the raster path: give `mu` beside
`structure`, and leave `layer_algorithm` as it is (ETM labels the layer's modes itself, §9):

```python
eps = np.full((1, 4, 4), 2.25, dtype=complex)
mu = np.ones((1, 4, 4), dtype=complex)
eps[0, 1:3, 1:3], mu[0, 1:3, 1:3] = -3 + 0.1j, -0.5 + 0.05j   # a double-negative pillar
solver = wavebit.rcwa(
    top=1., bot=2.25, structure=eps, mu=mu, thickness=[200.],
    wavelength=500., period=(1000., 900.), truncation_order=(2, 2),
    theta=0.3, phi=0.6,
)
result = solver.solve()
float(np.sum(result.R)), float(np.sum(result.T))   # 0.21002, 0.60802
```

**On a shape path** the double negative is a shape's (or a background's) `material`:

```python
layer = {'background': 2.25,
         'shapes': [{'type': 'circle', 'cx': 0., 'cy': 0., 'lx': 400.,     # 400 nm across
                     'material': wavebit.Material(-3 + 0.1j, mu=-0.5 + 0.05j)}]}
solver = wavebit.rcwa(
    top=1., bot=2.25, structure=[layer], thickness=[200.],
    wavelength=500., period=(1000., 900.), truncation_order=(2, 2),
    theta=0.3, phi=0.6,
)
result = solver.solve()
float(np.sum(result.R)), float(np.sum(result.T))   # 0.11623, 0.74861
```

**Through the coherent door**: a `PlaneWave` from a negative-index side means what `theta` means,
and returns the same light (§8). A coherent result reports powers per side, not `R`:

```python
grating = np.full((1, 4, 4), 2.25, dtype=complex)
grating[0, :2, :3] = 4.0
scene = dict(top=wavebit.Material(-3 + 0.1j, mu=-0.5 + 0.05j), bot=2.25,
             wavelength=500., period=(1000., 900.), truncation_order=(2, 2),
             structure=grating, thickness=[120.])
coherent = wavebit.rcwa(**scene, excitation={'top': wavebit.PlaneWave(
    theta=np.deg2rad(35.), phi=0.6, psi=0.3)}).solve()
plain = wavebit.rcwa(**scene, theta=np.deg2rad(35.), phi=0.6, psi=0.3).solve()
float(coherent.output_top), float(coherent.output_bottom)   # 0.06712053, 0.93318075
float(np.sum(plain.R)), float(np.sum(plain.T))              # 0.06712053, 0.93318075
# their sum is 1.00030128: the lit medium absorbs, as above; coherent.absorption is round-off
```

**A `solve_conv` half-space pair**: the half-space terms handed to `solve_conv` may name an
`(eps, mu)` pair the scene does not hold; the solve then uses that pair's root and signed index,
and afterwards the solver is its own scene again. If the pair turns the *lit* side's sign, the
solver's own `reconstruct_field` refuses afterwards; the result's field door still works (§15):

```python
nim = (-3 + 0.1j, -0.5 + 0.05j)
common = dict(top=2.25, wavelength=500., period=(1000., 900.), truncation_order=(2, 2),
              structure=grating, thickness=[120.], theta=0.3, phi=0.6)
scene = wavebit.rcwa(**common, bot=nim[0] * nim[1])          # non-magnetic: eps*mu only
paired = scene.solve_conv(*scene.eps_conv_matrices(), eps_bot=nim[0], mu_bot=nim[1])
magnetic = wavebit.rcwa(**common, bot=wavebit.Material(*nim)).solve()
np.array_equal(paired.t_p, magnetic.t_p)   # True: solved as that medium, signed index included
scene.n_bot                                # (1.2254-0.0816j): the scene's own index again
```

**Differentiating through it** -- every decision is a `where` (§11):

```python
import jax, jax.numpy as jnp
wavebit.enable_x64()

def reflectance(re_eps):
    medium = wavebit.Material(re_eps + 0.1j, mu=-1 + 0.1j)
    s = wavebit.rcwa(backend=1, top=2.25, bot=medium, wavelength=500.,
                     period=(1000., 1000.), truncation_order=(0, 0),
                     structure=np.full((1, 2, 2), 2.25), thickness=[37.])
    return s.solve().R.sum().real

jax.grad(reflectance)(jnp.asarray(-2.0))    # 0.0074111 (a central difference agrees)
jax.jit(reflectance)(jnp.asarray(-2.0))     # 0.0010766816
```

**Under `jax.jit`, read the gain band** -- the refusals cannot run on a tracer (§10):

```python
def band(re_eps):
    s = wavebit.rcwa(backend=1, top=2.25, bot=wavebit.Material(re_eps - 0.1j, mu=-1 - 0.1j),
                     theta=np.deg2rad(60.), wavelength=500., period=[700.],
                     truncation_order=3, thickness=[180.],
                     structure=np.array([[[4., 4., 1., 1., 2.25, 2.25]]]))
    return s.gain_cutoff_band()['bottom']

jax.jit(band)(jnp.asarray(-2.0))   # [False False False True False False False]: one order
band(-2.0)                         # ValueError: `bot` is an active negative-index half-space ...
```

The six unflagged orders under `jit` are solved too, with no physical answer behind them, so
check an active material once outside the trace.

**Checking what you asked for.** `solver.n_top` / `n_bot` show the index sign the solve uses;
`solver.order_kxy()` shows the incident wavevector a real angle produced (negative `kx` for light
travelling along `+x` from a negative-index side) and each order's phase wavevector (§8 says
where its light goes); `result.boundary_fields(side, x, y, part='incoming')` returns the incident
field, whose Poynting vector is the direction `theta` named (exactly, in a lossless medium).
Direct `kx`, `ky` bypass the angle semantics entirely.

---

## 17. How the decisions were reached

The history, for whoever has to change any of it. Develop commits are named by id; the work that
followed is named by commit title, since its ids change if the branch is rebased.

1. **2026-08-12 to 2026-10-03.** A half-space took the principal root of `kz**2`, continued
   from normal incidence ([`branch-cuts.md`](branch-cuts.md) §8). For a passive double negative at a real `k_par` that is the growing
   root of §3, so such a half-space was always solved wrongly; the generic result check
   `check_physical` flagged it, by its negative `T`, with an `UnphysicalResultWarning`.
2. **`6bb364b`** (2026-10-03): half-spaces take Maystre's cut, the limit of their own absorption
   ([`branch-cuts.md`](branch-cuts.md) §8). For a double negative it is the same growing root.
3. **`601ba85`**: `check_physical` removed, because it called correct answers for active media
   failed solves ([`result-guards.md`](result-guards.md)). The double negative's `R = 928.78` was
   now returned with no warning.
4. **`d421aa2`**: a propagating double-negative half-space refused where it is set (`Re(eps) < 0`,
   `Re(mu) < 0`, `Re(eps*mu) > 0`). `eps = mu = -1+1i` (`Re(eps*mu) = 0`) slipped through it (§5).
   `0dfc4ce` (`GainCutoffWarning`) is the develop commit this work started from.
5. **The prototype**, "solve a negative-index far side; its root is the mirror of Maystre's cut":
   the mirror cut keyed on the material, `n = -sqrt(eps*mu)` for a double negative, the far side
   solved and the lit side refused. A review found five problems in it: ETM's layer labels for a
   double-negative layer (singular or wrong); the index sign jumping where `Re(eps)` or `Re(mu)`
   crosses zero; complex `theta` from a lit double negative turning `kx` inconsistently with real
   `theta`; the gain band never flagging an active double negative; and a `solve_conv` pair
   keeping the old sign of `t_p`.
6. **The decisions of 2026-10-04**, made by the maintainer from that review:
   * **incidence**: `theta` and `phi` give the direction the incident light (its power) travels,
     as always; a lit double negative is solved, not refused (§7);
   * **the index**: `n = sqrt(eps)*sqrt(mu)`, one shared function, bit-identical for every
     `s = +1` medium, continuous across the real-part seams (§6);
   * **the root**: keep the mirror cut for double-negative half-spaces, with its own tie (§5);
   * **refusals**: keep refusing an active propagating double negative, with the citations; flag
     it under `jit` through the gain band; drop every other refusal of a double negative (§10);
   * **ETM**: give the SMM answer for double-negative layers, leaving every other layer
     bit-identical (§9).
7. **The work**, in order: one signed index for every magnetic half-space; the mirror cut's own
   tie; `theta` as the direction of the light, with a lit negative-index side solved; the sign of
   a `solve_conv` pair; the gain band above the axis for a double negative; reciprocity of a
   grating lit from a double negative; the docstrings; the layer labels by power, and the message
   for Pendry's pair; then a review round: three principal values as one flag per material, the
   principal root exactly on the negative real axis, the scene's own pairs under `jit`, finite
   PyTorch gradients through a normal-incidence `PlaneWave`, the derivative of the `kx == 0`
   frame column, the 4ff flux threshold in ulp; the fmmax golden; this page and its study.
8. **The follow-up** (2026-10-04), from a review of what the work had left open and of the
   literature since 2005: two lossless half-spaces that are each other's negation refused (§9);
   ETM's perfect-lens message for a double-negative substrate; each factor of the signed index
   on Maystre's cut (§6); the mirror-medium test and the PyMoosh golden for lossless double
   negatives (§12); and the citations, the names (§5) and the literature status (§18).
9. **The second follow-up** (2026-10-04), from a verification of the first: the negation pair is
   refused only where a retained order is evanescent, since a truncation of propagating orders is
   regular and was solved exactly before the refusal (§9); a factor's Maystre root is decided on
   the factor's real part, not on the round-off of its root's `Re + Im`, which had decided the
   cut's tie by backend (§6); `index_from_eps` takes the lip a signed zero names on JAX too
   (§15); and corrections to this page -- the mirror symmetry is Lakhtakia's, which Grunhut and
   Depine cite; §12's scene has one first order propagating; PyMoosh is not the only code lit
   from a double negative; the gaps between `cmath` and NumPy are one or two ulp.

---

## 18. What newer research says

A review made in 2026-10 of the work published from 2005 to 2026 -- numerical methods, the
mathematics of limiting absorption and radiation conditions, causality and active media, and the
names in use -- with each item confirmed to exist and read in full unless this section says
otherwise. None of it gives a different root for a passive medium. Where a number illustrates an
item, the study's section 13 computes it.

### What it supports

* **The passive rule.** Every source that treats a passive double-negative half-space either
  sorts its waves by decay and Poynting flux, or flips the root where both real parts are
  negative, and both give Wavebit's root (§5, "The same rule in print"). The rigorous support is
  M. Cassier, C. Hazard and P. Joly's (2017, 2022): the limiting-absorption root of a dispersive
  double-negative half-space, with the loss on each factor, flipped exactly in its backward-wave
  bands. H.-M. Nguyen and S. Sil (2020) write the loss on `eps` and `mu` separately; R. A. Depine
  and A. Lakhtakia (2005), M. Lequime et al. (2013) and PyMoosh's code state the same switch
  independently.
* **The trigger.** "Both real parts negative" is, in the lossless limit, exactly where a passive,
  non-dissipative, local material carries backward waves -- and the theorem is for passive
  materials only (M. Cassier, P. Joly and M. Kachanovska 2017, Theorem 3.34 and Remark 3.35).
  That is where Wavebit applies the mirror without refusing anything.
* **The lossless limit.** That a lossless half-space takes the limit of the lossy root is what
  the limiting-absorption theorems say; Depine and Lakhtakia state the lossless branch for a
  modal method in so many words. The codes that keep `Im(kz) > 0` (fmmax, inkstone, §14) are the
  lossy half of the same rule, and sorting a lossless order by its Poynting flux (Lalanne and
  Hugonin 2026, §2.3) is the other half.
* **The mirror symmetry**, A. Lakhtakia's conjugation symmetry (2004), which V. Grunhut and
  R. A. Depine (2011) apply to a flat interface, now a bit-exact test (§12).
* **The refusals.** The lossless negation pair has no time-harmonic answer (B. Gralak and
  D. Maystre 2012; B. Gralak 2020; Cassier, Hazard and Joly 2022; M. Cassier and P. Joly 2025;
  §9). An active double negative has no single-frequency rule (J. Skaar 2006, 2010; B. Nistad and
  J. Skaar 2008; Y.-F. Chen, P. Fischer and F. W. Wise 2006, abstract read; §10).
* **The signed index, per factor.** A. A. Govyadinov, V. A. Podolskiy and M. A. Noginov (2007)
  take `n = sqrt(eps) * sqrt(mu)` with each factor's root on Maystre's cut, and note that their
  table of decay and growth cannot be reduced to one cut (§6); for passive media the product of
  principal roots is the causal index (Skaar 2010).

### What it refines

* **`Im(mu)` of a homogenised medium can be negative** without any gain (B. Gralak, *C. R.
  Physique* **21**, 343 (2020), §4). "Limiting absorption on `eps` and `mu` separately" assumes
  `Im(eps) >= 0` and `Im(mu) >= 0`, as the local passive model does. A double negative whose
  imaginary parts have opposite signs lies outside that model, no root serves it, and it is
  refused (§10; *study §13*).
* **Tensors need their waves sorted one by one.** For full `eps` and `mu` tensors, current RCWA
  classifies propagating modes by the direction of their longitudinal Poynting flux and evanescent
  ones by their decay (E. Yang, Q. Song and W. Cai 2026), and the sign of `Re(kz)` and the sign of
  the flux can disagree even in a passive, positive uniaxial crystal (M. M. Bay, S. Vignolini and
  K. Vynck 2022). A switch per medium cannot serve a hyperbolic half-space (*study §13*; §15).
  Wavebit's half-spaces are isotropic, so this is a condition on future work, not a defect.
* **Complex frequency.** P. Lalanne and J.-P. Hugonin (2026) sort real-frequency modes by
  Poynting flux and decay -- Wavebit's root -- and state `Re + Im > 0` for the whole paper, which
  they need at complex frequency. For a double negative the two statements disagree, and at a
  complex frequency the single cut takes a growing root where the per-medium rule takes the
  decaying one (*study §13*; §15).
* **What `theta` is.** A Poynting (power) direction, which Wavebit chose; not a causal statement,
  because causality criteria test the group velocity (P. Kinsler and M. W. McCall 2008; §7).

### Where it is silent

* **No name** for the mirror cut, or for the rule as a whole (§5).
* **No definition of an incidence angle inside a double negative.** Wavebit's is its own (§7).
  Several codes can be lit from one (§14); PyMoosh, the one found that solves a lossless lit
  double negative physically, uses the phase's angle, so from a lossless double negative its `kx`
  is the negative of Wavebit's at the same `theta`.
* **No rule for an active double negative** -- only the demonstration that none at a single
  frequency exists (§10).
* **No discussion of a tie-break** on either cut. Wavebit's ties (§5) are its own, chosen so that
  the mirror identity holds to the bit.
* **No RCWA paper since 2005 that states the half-space rule for a grating on a negative-index
  half-space.** Depine and Lakhtakia's grating papers (2004, 2005) use the Rayleigh and C methods.
  M. Cuevas and R. A. Depine (*Opt. Commun.* **284**, 5242 (2011)) treat a grating on an isotropic
  negative-index half-space, but the review read only the abstract and could not see their root
  choice.

### Gain media that are not double negatives

The same literature reaches a branch Wavebit already had. An ordinary gain half-space
(`Im(eps) < 0` with positive real parts) takes Maystre's root, "the gain switched on from zero",
which beyond the critical angle gives amplified total internal reflection, `R > 1`
([`branch-cuts.md`](branch-cuts.md#gain-near-grazing)). That answer is conditional. Nothing in the
solver changes for it; what changes is what the documentation claims.

* **It is the weak, conventional-gain answer.** J. O. Grepstad and J. Skaar (*Opt. Express*
  **19**, 21404 (2011)), in this time convention, obtain evanescent gain for weak conventional
  gain by continuing from high frequency. Their `kz` changes sign only at the critical point,
  where `Re(kz**2) = 0` -- which is where Maystre's cut jumps in a gain medium, and the band
  `GainCutoffWarning` marks.
* **It is contested.** T. S. Mansuripur and M. Mansuripur (*Appl. Phys. Lett.* **104**, 121106
  (2014)) argue that a semi-infinite gain medium gives no single-surface amplified total internal
  reflection. A. E. Siegman (*Opt. Photon. News* **21**(1), 38 (2010), a magazine article; the
  review read only the abstract of a related talk) argues against the amplification too.
* **It can be the wrong root for a strongly dispersive active medium.** Y.-F. Chen, P. Fischer
  and F. W. Wise (*Phys. Rev. Lett.* **95**, 067402 (2005)) propose an active, non-magnetic medium
  that refracts negatively, at frequencies where Maystre's cut takes the root of positive
  refraction; B. Nistad and J. Skaar (*Opt. Express* **15**, 10935 (2007); abstract read) find a
  negative index in an active right-handed medium. Which root such a medium takes is fixed by its
  dispersion (Skaar 2010), which a solve at one frequency does not have.

---

## References

- V. G. Veselago, "The electrodynamics of substances with simultaneously negative values of
  epsilon and mu", *Sov. Phys. Usp.* **10**, 509 (1968).
- J. B. Pendry, "Negative refraction makes a perfect lens", *Phys. Rev. Lett.* **85**, 3966 (2000).
- D. R. Smith and N. Kroll, "Negative refractive index in left-handed materials", *Phys. Rev.
  Lett.* **85**, 2933 (2000).
- R. A. Shelby, D. R. Smith and S. Schultz, "Experimental verification of a negative index of
  refraction", *Science* **292**, 77 (2001).
- R. W. Ziolkowski and E. Heyman, "Wave propagation in media having negative permittivity and
  permeability", *Phys. Rev. E* **64**, 056625 (2001).
- J. A. Kong, "Electromagnetic wave interaction with stratified negative isotropic media",
  *Prog. Electromagn. Res.* **35**, 1 (2002).
- M. W. McCall, A. Lakhtakia and W. S. Weiglhofer, "The negative index of refraction
  demystified", *Eur. J. Phys.* **23**, 353 (2002).
- R. A. Depine and A. Lakhtakia, "A new condition to identify isotropic dielectric-magnetic
  materials displaying negative phase velocity", *Microwave Opt. Technol. Lett.* **41**, 315 (2004).
- R. A. Depine and A. Lakhtakia, "Plane-wave diffraction at the periodically corrugated boundary
  of vacuum and a negative-phase-velocity material", *Phys. Rev. E* **69**, 057602 (2004).
- R. A. Depine, A. Lakhtakia and D. R. Smith, "Enhanced diffraction by a rectangular grating made
  of a negative phase-velocity (or negative index) material", *Phys. Lett. A* **337**, 155 (2005).
- D. R. Smith, D. Schurig, M. Rosenbluth, S. Schultz, S. A. Ramakrishna and J. B. Pendry,
  "Limitations on subdiffraction imaging with a negative refractive index slab", *Appl. Phys.
  Lett.* **82**, 1506 (2003).
- S. A. Ramakrishna and O. J. F. Martin, "Resolving the wave vector in negative refractive index
  media", *Opt. Lett.* **30**, 2626 (2005).
- J. Skaar, "Fresnel equations and the refractive index of active media", *Phys. Rev. E* **73**,
  026605 (2006).
- J. Skaar, "On resolving the refractive index and the wave vector", *Opt. Lett.* **31**, 3372
  (2006).
- A. Lakhtakia, J. B. Geddes III and T. G. Mackay, "When does the choice of the refractive index
  of a linear, homogeneous, isotropic, active, dielectric medium matter?", *Opt. Express* **15**,
  17709 (2007).
- B. Nistad and J. Skaar, "Causality and electromagnetic properties of active media", *Phys. Rev.
  E* **78**, 036603 (2008).
- A. A. Govyadinov, V. A. Podolskiy and M. A. Noginov, "Active metamaterials: Sign of refractive
  index and gain-assisted dispersion management", *Appl. Phys. Lett.* **91**, 191103 (2007),
  arXiv:0704.3765.
- B. Gralak and D. Maystre, "Negative index materials and time-harmonic electromagnetic field",
  *C. R. Physique* **13**, 786 (2012).
- J. B. Pendry and S. A. Ramakrishna, "Focusing light using negative refraction", *J. Phys.:
  Condens. Matter* **15**, 6345 (2003).
- A. Lakhtakia, "Conjugation symmetry in linear electromagnetism in extension of materials with
  negative real permittivity and permeability scalars", *Microwave Opt. Technol. Lett.* **40**,
  160 (2004), doi:10.1002/mop.11315, arXiv:physics/0307130.
- V. Grunhut and R. A. Depine, "Influence of the sign of the refractive index in the reflectivity
  of a metamaterial surface with localized roughness", *Eur. Phys. J. D* **62**, 227 (2011),
  arXiv:1102.2639.
- M. Cassier, C. Hazard and P. Joly, "Spectral theory for Maxwell's equations at the interface of
  a metamaterial. Part I: Generalized Fourier transform", *Commun. PDE* **42**, 1707 (2017),
  arXiv:1610.03021; "Part II: Limiting absorption, limiting amplitude principles and interface
  resonance", *Commun. PDE* **47**, 1217 (2022), arXiv:2110.06579.
- M. Cassier and P. Joly, "An operator approach to the analysis of electromagnetic wave
  propagation in dispersive media. Part 2: transmission problems", arXiv:2504.00235 (2025).
- D. Langevin, P. Bennet, A. Khaireh-Walieh, P. Wiecha, O. Teytaud and A. Moreau, "PyMoosh: a
  comprehensive numerical toolkit for computing the optical properties of multilayered
  structures", *J. Opt. Soc. Am. B* **41**, A67 (2024).
- R. Petit (ed.), *Electromagnetic Theory of Gratings*, Topics in Current Physics (Springer,
  Berlin, 1980), doi:10.1007/978-3-642-81500-3. Not opened; cited by Lalanne and Wu for the
  `Re + Im > 0` convention.
- H. Benisty, J.-J. Greffet and P. Lalanne, *Introduction to Nanophotonics* (Oxford University
  Press, 2022), §18.4.3. Not opened; cited by Lalanne and Hugonin for the same convention.
- P. Lalanne and T. Wu, "Theory of superlensing with complex frequency illuminations",
  arXiv:2508.10742 (2025).
- P. Lalanne and J.-P. Hugonin, "Expanding RCWA capabilities with advanced S-matrix algorithms",
  arXiv:2609.10076 (2026).
- R. A. Depine and A. Lakhtakia, "Diffraction gratings of isotropic negative-phase velocity
  materials", *Optik* **116**, 31 (2005), doi:10.1016/j.ijleo.2004.11.004, arXiv:physics/0408050.
- M. Lequime, B. Gralak, S. Guenneau, M. Zerrad and C. Amra, "Optical properties of multilayer
  optics including negative index materials", arXiv:1312.6288 (2013).
- M. Cassier, P. Joly and M. Kachanovska, "Mathematical models for dispersive electromagnetic
  waves: an overview", *Comput. Math. Appl.* **74**, 2792 (2017), doi:10.1016/j.camwa.2017.07.025.
- H.-M. Nguyen, "Limiting absorption principle and well-posedness for the Helmholtz equation with
  sign changing coefficients", *J. Math. Pures Appl.* **106**, 342 (2016),
  doi:10.1016/j.matpur.2016.02.013.
- H.-M. Nguyen and S. Sil, "Limiting absorption principle and well-posedness for the time-harmonic
  Maxwell equations with anisotropic sign-changing coefficients", *Commun. Math. Phys.* **379**,
  145 (2020), doi:10.1007/s00220-020-03805-1.
- B. Gralak, "Negative index materials: at the frontier of macroscopic electromagnetism",
  *C. R. Physique* **21**, 343 (2020), doi:10.5802/crphys.29.
- A.-S. Bonnet-Ben Dhia, L. Chesnel and M. Rihani, "Maxwell's equations with hypersingularities at
  a negative index material conical tip", *Pure Appl. Anal.* **7**, 127 (2025),
  doi:10.2140/paa.2025.7.127.
- J. Skaar, "Causality and its implications for passive and active media", Proc. URSI Int. Symp.
  on Electromagnetic Theory (EMTS 2010), pp. 29-32, doi:10.1109/URSI-EMTS.2010.5636996,
  arXiv:1008.2874.
- P. Kinsler and M. W. McCall, "Causality-based criteria for a negative refractive index must be
  used with care", *Phys. Rev. Lett.* **101**, 167401 (2008); "Criteria for negative refraction in
  active and passive media", *Microwave Opt. Technol. Lett.* **50**, 1804 (2008).
- Y.-F. Chen, P. Fischer and F. W. Wise, "Negative refraction at optical frequencies in
  nonmagnetic two-component molecular media", *Phys. Rev. Lett.* **95**, 067402 (2005); "Sign of
  the refractive index in a gain medium with negative permittivity and permeability", *J. Opt.
  Soc. Am. B* **23**, 45 (2006).
- B. Nistad and J. Skaar, "Simulations and realizations of active right-handed metamaterials with
  negative refractive index", *Opt. Express* **15**, 10935 (2007).
- J. O. Grepstad and J. Skaar, "Total internal reflection and evanescent gain", *Opt. Express*
  **19**, 21404 (2011).
- T. S. Mansuripur and M. Mansuripur, "Fresnel reflection from a cavity with net roundtrip gain",
  *Appl. Phys. Lett.* **104**, 121106 (2014).
- A. E. Siegman, "Fresnel reflection, Lenserf reflection and evanescent gain", *Opt. Photon. News*
  **21**(1), 38 (2010).
- E. Yang, Q. Song and W. Cai, "Differentiable eigendecomposition-free RCWA for full-tensor
  anisotropic photonics", arXiv:2608.06185 (2026).
- M. M. Bay, S. Vignolini and K. Vynck, "PyLlama: a stable and versatile Python toolkit for the
  electromagnetic modelling of multilayered anisotropic media", *Comput. Phys. Commun.* **273**,
  108256 (2022), arXiv:2012.05945.
- M. Cuevas and R. A. Depine, "Dispersive characteristics of surface plasmon polaritons on negative
  refractive index gratings", *Opt. Commun.* **284**, 5242 (2011).
- W. Kahan, "Branch cuts for complex elementary functions, or Much ado about nothing's sign bit",
  in *The State of the Art in Numerical Analysis*, A. Iserles and M. J. D. Powell (eds.)
  (Clarendon Press, Oxford, 1987), pp. 165-211.
- J. P. Hugonin and P. Lalanne, "RETICOLO software for grating analysis", arXiv:2101.00901. The
  name "coupure de Maystre" is that of RETICOLO's source, `retsqrt.m`, not of this paper.
