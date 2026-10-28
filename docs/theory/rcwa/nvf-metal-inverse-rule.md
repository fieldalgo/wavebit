# The inverse rule on a metal: the pole in `<1/eps>^-1`, and the floor that closes it

Written: 2026-09-27. Measured on CPU, NumPy backend, float64, single-threaded. Everything was
first measured on 2026-09-27 with wavebit at `6fe4b7b`, with and without the floor; the spectra,
the comparisons with fmmax, S4 and RETICOLO, and the points at 441 harmonics and above other than
the holes at 807 nm are as measured then. The single-wavelength points that need no other solver
were re-measured on 2026-10-01 on the commit that adds the floor: the holes at 807 nm / 441, the
disks at 852 nm / 121, the clip counts and the anti-Hermitian minima. Of those, only the
unfloored holes had moved since `6fe4b7b`, from `R + T` 1.0019 (`T` 0.0188) and −16.95 to the
1.0017 (`T` 0.0189) and −16.94 given below. The derivatives were measured on 2026-10-01; the
oblique-incidence derivative, the second-derivative refusal, forward mode, the kinks, the
lossless jump, the hull bound with gain and the experimental passive closure's two points on
2026-10-02. Gold is Johnson & Christy (1972), interpolated linearly in `n` and `k`.

Targets:
- `src/wavebit/rcwa/_engine/factorization.py::_nvf_assemble` — the closing, and the floor it applies first
- `src/wavebit/rcwa/_engine/factorization.py::_floor_singular_values` — the floor
- `src/wavebit/rcwa/_engine/factorization.py::_floor_derivative` — its derivative, in closed form
- `src/wavebit/rcwa/_engine/factorization.py::_jax_floor`, `_torch_floor` — where that derivative enters autodiff (reverse and forward mode), and where a second derivative is refused
- `src/wavebit/_common/backend.py::forward_tangent` — how a PyTorch forward-mode dual, which has no `requires_grad`, is recognised (`carries_autodiff` asks it)
- `src/wavebit/rcwa/_engine/factorization.py::INVERSE_RULE_BOUND` — the one constant, and why it is 2
- `src/wavebit/rcwa/_engine/factorization.py::inverse_rule_bound` — the bound every builder hands over: `None` (no floor) on a readable layer of passive dielectrics, else `max |eps|`
- `src/wavebit/rcwa/_engine/factorization.py::nvf_conv_matrices`, `nvf_conv_matrices_integrated` — the raster builders' bound, the raster's largest `|eps|`
- `src/wavebit/rcwa/_engine/boundary.py::conv_mats` — the boundary path's bound, taken region by region
- `src/wavebit/rcwa/_engine/parametric.py::nvf_from_coeffs` — now delegates to `_nvf_assemble` rather than carrying its own copy of the formula
- `src/wavebit/rcwa/_engine/parametric.py::layer_epsilons` — the permittivities the shape path decides its bound from
- `studies/rcwa/nvf_resonant_spectrum/resonant_nvf_probe.py::direct_from_coeffs` — the diagnostic stand-in for `nvf_from_coeffs`, which takes the bound and ignores it
- `tests/rcwa/test_nvf_metal_inverse_rule.py`

---

## TL;DR

On a layer where `1/eps` changes sign across a boundary — gold beside air or glass — the
inverse-rule matrix `<1/eps>^-1` of the NVF tensor has a pole: at any finite truncation some
eigenvalue of `<1/eps>` passes the origin, and the inverse answers with `1/Im(1/eps)`, about 400
on gold whose `|eps|` is 25. The tensor then carries a plasmon of the *smeared* interface which the
sharp one does not have. Its symptoms, measured against fmmax, S4 and RETICOLO on a gold nanohole
array and a gold nanodisk array before this change: narrow spectral features no other code has,
that move with the truncation; and, through the symmetrized closing, gain — `R + T` up to 1.27 on
the holes and 1.13 (121 harmonics) to 45 (625 harmonics) on the disks.

**The fix is a floor** on the singular values of `<1/eps>` at `1 / (2 max|eps|)`, applied before
the inverse. It binds on a handful of modes on a metal (6 of 441 on the holes at 807 nm) and never
on a passive dielectric, lossy or not, where the tensor is bit-identical to what it was. On the
holes at 441 harmonics it takes the 796–840 nm spectrum from 0.0094 to 0.0015 RMS in `T` from
fmmax's Jones-direct at the same square truncation, removes the spike, and takes `R + T` at 807 nm
from 1.0017 to 0.9404; on the disks at 121 harmonics, from 43 of 51 points above 1 to none. It is
differentiated exactly, so a gradient through a floored layer is the gradient of what is solved.

The closing itself is unchanged. Three closings that are passive *by construction* were built and
measured against it on the same matrices; each costs accuracy or consistency somewhere the current
one does not, and the floor is what all of them needed anyway. The residual non-passivity of the
symmetrized closing after the floor is measured (the tensor's anti-Hermitian part reaches −1.8 to
−2.4 on gold, against −17 to −25 before; `R + T` at most 0.958 over the windows swept) and is
documented as a residual, not proved away. The repository's experimental passive closure was not
in that comparison; it is measured at two points below.

## The mechanism

Write the layer's `1/eps` as `1/eps_m + (1/eps_d − 1/eps_m) chi_d` with `chi_d` the dielectric's
indicator. Then

    <1/eps> = (1/eps_m) I + (1/eps_d − 1/eps_m) K,        K = <chi_d>

and `K`, the compression of an indicator onto the truncated Fourier basis, has eigenvalues `kappa`
that cluster at 0 and 1 with a transition band between them — the Slepian modes that straddle the
boundary — whose count grows with the boundary length and the truncation. `<1/eps>`'s eigenvalues
are `1/eps_m + (1/eps_d − 1/eps_m) kappa`, a line from `1/eps_m` (−0.040 − 0.0025i for gold at
807 nm) to `1/eps_d` (1 in air), and that line passes the origin at

    kappa* = eps_d / (eps_d − eps_m)          at 807 nm: 0.039 in air, 0.083 in n = 1.5

within `Im(1/eps_m)`. The inverse rule's effective permittivity for a transition mode is the
series-capacitor formula `1 / (kappa/eps_d + (1 − kappa)/eps_m)`: the laminate with a fraction
`kappa` of dielectric in series with the metal, which has a genuine plasmon at `kappa*`. So the
truncated basis represents the interface as a laminate, and the laminate resonates where the sharp
interface cannot. With loss the pole is a near-pole of height `1/Im(1/eps_m)` — 400 on gold, 2000
on silver — and it is met whenever some `kappa_j` of `K` (a property of geometry and truncation
only) lands where the wavelength puts `kappa*`:

    Re eps_m(lambda) = eps_d (1 − 1/kappa_j)

That prediction, from the eigenvalues of `K` alone, against the spurious features measured before
the change:

| Case | Harmonics | Eigenvalues `kappa_j` in the window | Predicted (nm) | Measured feature (nm) |
| --- | --- | --- | --- | --- |
| B disks | 121 | 0.0747 | 845 | 828–860, `R + T` 1.13 at 852 |
| B disks | 441 | 0.0816, 0.0911 | 815, 780 | 776 (where `E − Delta C` broke, 1.81) |
| B disks | 625 | 0.0922, 0.0988, 0.0990 | 776, 755, 754 | 755–765, `R + T` 45 at 760 |
| B disks | 1225 | 0.0867, 0.0901 | 796, 783 | 788–792 kink |
| A holes | 1369 | 0.0384, 0.0348 | 812, 834 | 816–819, `R + T` 1.27 at 818 |
| A holes | 1681 | 0.0374, 0.0361 | 820, 832 | 822.5–823 narrow peak |
| A holes | 441 | 0.0361, 0.0312 | 831, 881 | 804–808 (the one miss: 25 nm off) |

The coupled solve shifts the resonance from the bare pole by a few nanometres, more where the
field is strong (A at 441 sits on an extraordinary-transmission peak); the correspondence is
otherwise within 5 nm.

**Why the symmetrized closing turns the pole into gain.** The tensor's anti-Hermitian part is
`Im<eps> − ½ (Im(Delta) C + C Im(Delta))`, and a Jordan product of a positive matrix with a
projector is indefinite: for a rank-one `x |v><v|` and `<v|C|v> = c` it contributes as little as
`−x (sqrt(c) − c) / 2`, i.e. `−x/8` at `c = 1/4`. With `x` the pole, 400, the most negative
eigenvalue of that part was measured at −16.94 on the holes (807 nm, 441) and −25.4 on the disks
(852 nm, 121). The plain product `E − Delta C` that S4 and fmmax write has no Jordan structure and
is worse: −32.6 and −30.0 on the same matrices, `R + T = 1.81` on the disks at 441 and 16 in fmmax
with square truncation.

## The floor

`_floor_singular_values` raises every singular value of `<1/eps>` below `1 / (INVERSE_RULE_BOUND ·
max|eps|)` to that value and keeps the singular vectors. On a two-material layer `<1/eps>` is a
normal matrix (a linear function of the Hermitian `K`), so this is exactly a clip on the magnitude
of each eigenvalue with its phase untouched: the mode at the pole keeps its loss angle and loses
its excursion. It is bit-transparent where nothing is below the floor. Since 2026-10-28 it is decided
from the materials first: a layer whose permittivities can all be read and are passive dielectrics
(`Re eps > 0`, `Im eps ≥ 0`, the case proved below) is not floored at all, so no singular value is
computed. Any other layer computes the singular values alone, and the full SVD only where one is
below the floor. A metal, a gain medium, `Re eps = 0` and a `jax.jit` trace all take that second
route. The full SVD was not only the costlier half. PyTorch's (MKL's) failed to converge on the
`<1/eps>` of a dielectric circle at 961 harmonics (two tight clusters of singular values,
condition 12.25), which this floor would never have touched.

Why it never binds on a passive dielectric: the numerical range of `<1/eps>` lies in the convex
hull of the values `1/eps` takes, so the smallest singular value is at least the hull's distance
from the origin. For passive materials (`Im eps ≥ 0`) with every `Re eps > 0`, each `1/eps` sits at
least `1 / max|eps|` from the origin at minus its loss angle, so all of them lie in a sector at
most 90 degrees wide, and the hull stays at least `cos 45° / max|eps| = 0.71 / max|eps|` away
whatever the losses — `cos 22.5° / max|eps| = 0.92 / max|eps|` while no loss angle exceeds 45
degrees. The floor sits at `0.5 / max|eps|`. On the widest passive pair, loss angles of 89.9 and
0.1 degrees (`0.01 + 4i` disks in `4 + 0.01i`), the smallest singular value times `max|eps|` is
0.717, 0.721 and 0.709 at 81, 121 and 289 harmonics. The hull reaches the origin when some
`Re eps < 0` — a metal.

The argument needs passivity. A dielectric layer that mixes gain (`Im eps < 0`) with loss spreads
its `1/eps` over both sides of the real axis, the sector opens towards 180 degrees, and the floor
can bind with every `Re eps > 0`: `0.2 + 2i` disks in `0.2 − 2i` bring the smallest singular value
to `0.21 / max|eps|` at 121 harmonics and clip 6 modes (`2 ± 2i`: 0.72, not clipped). Such a layer
is floored like a metal; nothing here measures whether that is the right answer for it.

Why 2 and not 1: the series formula legitimately exceeds `max|eps|` for modes almost entirely
inside the metal (a thin dielectric gap in series reads *more* negative than the metal), and a
bound at 1 clips 356 of the 441 singular values on the holes at 807 nm, where 2 clips 6 and 3
clips 4. Measured on the holes' 796–840 nm spectrum at 441 against fmmax's Jones-direct at the
same truncation: RMS 0.0015 in `T` at 2, 0.0021 at 3.

### Its derivative

The floored matrix is `U max(S, f) V^H`, `f = 1 / (2 max|eps|)`, and autodiff is not let through
the SVD that computes it: the SVD's derivative divides by `s_i − s_j`, which is 0/0 on a uniform
layer and noise on the near-degenerate clusters of a two-material one. `_floor_derivative`
supplies the floored matrix's own derivative in closed form instead, entered as a JAX
`custom_jvp` rule (so forward and reverse mode both see it) and a PyTorch `autograd.Function`
with a `backward` and a `jvp` (so reverse mode and `torch.autograd.forward_ad` both see it).
With `X = U^H dB V` it is `U Y V^H`,

    Y     = ½ [D1 ∘ (X + X^H) + D2 ∘ (X − X^H)] + diag(clipped) df
    D1_ij = (g_i − g_j) / (s_i − s_j),   D2_ij = (g_i + g_j) / (s_i + s_j),   g = max(s, f)

from the SVD's first-order perturbation (J. Townsend, *Differentiating the Singular Value
Decomposition*, 2016; Z.-Q. Wan and S.-X. Zhang, arXiv:1909.02659, for the complex case). `D1` is
exactly 1 between two unclipped values and 0 between two clipped ones, so no difference of
degenerate values is ever divided by. The last term is the floor's own dependence on the
permittivity, which every clipped mode carries. `D2` divides by `s_i + s_j`, which vanishes only
between two zero singular values of an exactly singular `<1/eps>` — a lossless metal at its pole —
where the floored matrix has no derivative; the entry is NaN there rather than an invented finite
value.

It changes nothing in the forward pass, which does not read it. The convolution matrices and
`R`, `T` are byte-identical to the same floor with the lift added as a bare constant, on NumPy,
JAX and PyTorch, eagerly and under autodiff, on 19 scenes: every NVF builder (shape, boundary,
both raster coefficient routes, the circular projector) with gold, where up to 6 modes are
clipped, and with dielectrics.
What it changes is the gradient wherever the floor binds on a mode that reaches the far field.
`d(R + T)/d(diameter)`, against a central difference extrapolated in the step:

| Case | Clipped | Exact (this) | Central difference | Lift as a bare constant | Vectors frozen |
| --- | --- | --- | --- | --- | --- |
| B disks, 852 nm | 1 of 121 | 2.969526e-04 | 2.969526e-04 | −1.142507e-02 | −7.105029e-03 |
| A holes, 807 nm | 6 of 169 | 3.065110e-04 | 3.065110e-04 | −1.842889e-03 | 1.130864e-04 |
| B disks, 852 nm, oblique | 2 of 81 | −4.314524e-03 | −4.314524e-03 | −5.198912e-03 | −5.271193e-03 |

A bare constant returns the *unfloored* closing's derivative at the floored point: 39x and of
the wrong sign on the disks. Holding the clipped modes at the floor with their singular vectors
frozen differentiates the lift's magnitude but not its rotation, and the rotation is most of the
answer. In the permittivity, `d(R + T)/d Re eps` on the disks is 1.073187e-02 against a
difference of 1.073187e-02; without the floor's own term it would be 6.054986e-03. The first two
rows are at normal incidence. There the disks at 81 harmonics give the same derivative either
way: two modes are clipped, and symmetry keeps them dark, so `R + T` does not see them. The
oblique row (θ 0.3, φ 0.4, ψ 0.7) is the same structure with those two modes lit, and there the
bare constant is 20% off.

**Forward mode is the same derivative.** `jax.jvp` and `jax.jacfwd` go through the same rule as
`jax.grad`. On PyTorch a `torch.autograd.forward_ad` dual has `requires_grad` False, so the floor
asks for a forward-mode tangent as well; until it did, a dual went down the plain branch, whose
floored part is a constant, and its tangent came back zero, with no error. Through
`eps_conv_matrices()` on the disks at 81 harmonics (read-out `Re Σ conj(W) ∘ eps_xx`) that gave
−0.538854 in the diameter and +3.423643 in `Re eps`, where reverse mode and a central difference
read −27.626431 and −17.338962; it now gives the reverse-mode values to 1e-14. `torch.autograd`
through `torch.func.vmap` -- a batch of wavelengths or designs, differentiated by `backward()` or
`forward_ad` -- is exact too, with the lanes binding or not: the floor's `autograd.Function` has a
`vmap` rule, and a per-lane `where` hands a lane where the floor does not bind `<1/eps>` itself,
bit for bit. The `torch.func` grad and jvp transforms (`grad`, `vjp`, `jvp`, `jacrev`, `jacfwd`,
`hessian`, alone or composed with `vmap`) would reach the rule's first-order guard, which cannot
tell such a level from a second derivative, so where the floor binds and their input reaches the
floored matrix they raise a `NotImplementedError` that says so and points at `torch.autograd`
(a captured `requires_grad` permittivity beside a transform over something else is not their
level, and passes); before, `torch.func.jvp` returned the same −0.538854. On a layer where the
floor does not bind it is never reached and `torch.func` is exact -- without `vmap`: under a
`vmap` whose batch reaches the floored matrix, with a grad or jvp level whose input reaches it
too, whether the floor binds is a per-lane answer that cannot be read, so a dielectric layer is
refused there too. A whole `solve()` is exact in forward mode too: its modal operator, which
asked `requires_grad` as well and gave a wrong `forward_ad` derivative of `R + T` on a
dielectric, floor or none, has carried a forward rule since 2026-10-02 (the autodiff guide's
"Forward mode").

**First derivatives only.** The closed form is linear in `dB` with coefficients from the
detached SVD, so a second derivative taken through it would lack the floor's curvature. Through
`eps_conv_matrices()` on the disks at 81 harmonics, with the read-out `Re Σ conj(W) ∘ eps_xx`, it
returned +4.193117 on JAX and PyTorch alike, where a central difference of the exact first
derivative reads −17.546: the wrong sign, and no error. A second derivative through a layer where
the floor binds now raises, as the modal operator's does — `NotImplementedError` on JAX,
`RuntimeError` on PyTorch for `create_graph=True` and for either mixed order, a `forward_ad`
tangent carried through a backward pass or a backward pass through one (each read the same
+4.193117 unrefused). On a passive dielectric layer the floor is the identity and the Hessian
stays exact (−2.125e-3 in every mode on that read-out). Under `jax.jit`, and under
`jax.checkpoint`, which stages the function the same way, whether the floor binds is not known
when the trace is built, so there a second derivative through any NVF layer is refused. Until
2026-10-02 the refusal under `jax.checkpoint` was dropped: it was a call whose result went
unread, which the checkpoint's partial evaluation removes as dead code, and `jax.hessian` of the
checkpointed read-out returned −0.8494 where −17.546 is right. `solve()` itself refused second derivatives before the floor
existed, at the modal operator; `eps_conv_matrices()` is the route this closes.

**A kink where a singular value crosses the floor.** The floored matrix is continuous there and
its derivative is not: that mode's gain steps from 0 (held at the floor) to 1 (its own). A value
exactly at the floor counts as unclipped, but at a crossing rounding decides which side of the
floor a computed singular value lands on, so either one-sided slope can come back there. The
steps are neither small nor rare. On the disks at 121 harmonics and oblique incidence,
`d(R + T)/d(diameter)` goes from 3.86e-03 to 2.70e-02 per nm across the crossing at
`lx` = 132.397 nm, 7x, autodiff matching a one-sided difference on each side; and the clipped set
changes every few nanometres of diameter, 0, 2, 2, 0, 1, 3, 2, 2, 2, 2, 3, 3, 2 modes at `lx` =
100, 110, …, 220 nm. A design step that carries a mode across the floor sees the slope change
there.

**On a lossless metal the floored tensor jumps.** With a real `eps_m`, `<1/eps>` is Hermitian and
its eigenvalues are real; the floor holds a clipped one's magnitude while its sign follows the
eigenvalue, so where that eigenvalue changes sign the floored value goes from `−f` to `+f`. A
magnitude floor on a real eigenvalue cannot be continuous through zero. On the disks at 121
harmonics (`lx` 140 nm, oblique incidence), `R` steps from 0.197922 to 0.060704 at
`Re eps_m` = −27.856868 and by −0.086 at −11.6395, at a relative step of 1e-9 as at 1e-6, with
`R + T = 1` to 1e-12 on both sides. Autodiff returns the slope on one side and cannot report
the step. Any loss makes the turn continuous but steep: with `Im eps_m` = 1e-3, `R` changes by
0.133 within ±0.028 of that `Re eps_m`; with 0.1, by 0.020.

## What it does, measured

Same matrices through `solve_conv`, so the closing is the only variable.

| Case | Harmonics | nm | `R + T` before | after | fmmax Jones-direct, square |
| --- | --- | --- | --- | --- | --- |
| A holes | 441 | 807 | 1.0017 (`T` 0.0189) | 0.9404 (`T` 0.0086) | `T` 0.007–0.010 |
| A holes | 1369 | 818 | 1.2726 (`T` 0.1815, absorbed −0.273) | 0.8896 (`T` 0.0592, absorbed 0.110) | `T` 0.069, absorbed 0.105 |
| B disks | 121 | 852 | 1.1259 (`T` 1.0283) | 0.8158 | — |
| B disks | 441 | 776 | 0.6745 | 0.5719 | — |
| D TiO2 nanofin | 121 | 532 | 0.999999 | 0.999999, bit-identical | — |
| E a-Si ellipse `P_yx` | 121–961 | 915 | 0.2814 / 0.3026 / 0.2907 | identical | — |

Spectra, the holes at 441 over 796–840 nm and the disks at 121 over 820–870 nm, 1 nm steps, with
wavebit at `6fe4b7b` (`rough` is the largest second difference of `T`, the signature of a spike):

| Closing | A: max `R + T` | RMS `T` vs fmmax square | rough | B: points above 1 | max `R + T` | rough |
| --- | --- | --- | --- | --- | --- | --- |
| symmetrized, no floor (before) | 1.0019 | 0.0094 | 0.0084 | 43 of 51 | 1.1259 | 0.0016 |
| **symmetrized + floor (now)** | 0.9581 | **0.0015** | 0.0011 | 0 | 0.8746 | 0.0006 |
| `E − Delta C` (S4, fmmax) | 0.9610 | 0.0027 | 0.0008 | 0 | 0.8696 | 0.0003 |
| congruence `T E T + C A C` + floor | 0.9619 | 0.0024 | 0.0017 | 0 | 0.8723 | 0.0003 |
| Herm(symmetrized) + antiHerm(congruence) + floor | 0.9577 | 0.0022 | 0.0011 | 0 | 0.9263 | 0.0002 |
| Herm(symmetrized) + `Im<eps>` + floor | 0.9570 | 0.0058 | 0.0008 | 0 | 0.9048 | 0.0002 |

fmmax's own second difference on the same window is 0.0045 (square) and 0.0104 (circle): the
floored closing is smoother than either reference at equal truncation.

The disks at the two truncations where the pole was worst, floored against unfloored, with wavebit
at `6fe4b7b`:

| Harmonics | Window (nm) | max `R + T` before | after | rough before | after | RMS `T` vs fmmax circle, before | after |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 441 | 740–830, 2 nm | 0.9318 (absorption bump 748–764) | 0.8701 | 0.0563 | 0.0053 | 0.111 | 0.056 |
| 625 | 745–775, 1 nm | 45.26 | 0.8652 | 31.3 | 0.0016 | — | — |

(fmmax's square truncation at 441 has `R + T = 16` on this window and is no reference; the
circle at 437 is the one that settled, and the floored closing halves the distance to it.)

## The passive closings, and why they were not taken

Each of the last three rows is passive by theorem — its anti-Hermitian part is a sum of
congruences of positive semidefinite matrices (`<Im eps>` is the compression of a non-negative
function; `Im(<1/eps>^-1) = −B^-1 Im<1/eps> B^-H ≥ 0`), measured at +1e-10 or better on every
gold point where the symmetrized closing is at −2. On a lossless layer all three reduce to the
symmetrized closing exactly.

- **Congruence, `T E T + C A C` with `T = I − C`.** Consistent only if `C^2 = C`, and a truncated
  projector series is not a projector: `||C^2 − C||_2 = 0.250` at every truncation and shape
  tried (an eigenvalue sits at exactly ½), with 10% of the eigenvalues between 0.1 and 0.9. On the
  a-Si ellipse it puts `P_yx` at 0.4406 against 0.2814 at 121 harmonics (converged 0.2935).
  Rounding `C` to the nearest projector by its spectrum (`congp`) recovers 0.3472 and is still the
  least accurate on the holes (0.0085).
- **Hermitian part of the symmetrized closing + anti-Hermitian part of the congruence closing.**
  Keeps the accuracy (0.0022 on the holes) and the theorem, but its loss term is `T Im<eps> T +
  C Im(A) C`, which scales the loss by `T^2 + C^2` — ½ on the modes where `C` has eigenvalue ½.
  On a painted *uniform* gold slab (two overlapping rectangles of the same index, the boundary
  path's test) it moves `R` from 0.07477 to 0.07405, 1%, where the answer is Airy's and exact.
  A closing that is wrong on a slab is not a default.
- **Hermitian part of the symmetrized closing + `Im<eps>`.** Exact on the slab, passive, and
  4x worse on the holes (0.0058): the inverse rule's imaginary part is part of the jump condition
  on a metal, not decoration. Without the floor it is also rough (0.030 on the disks), because
  the pole is still in the Hermitian part.

The floor alone keeps what all four have in common where they are smooth, and the closing it
leaves is the most accurate of them. The cost is that `R + T ≤ 1` on a metal remains a
measurement: −1.76 (holes), −2.09 (disks at 121) and −2.37 (disks at 441) on the anti-Hermitian
part after the floor, none of which reached the far field over the windows swept.

**The repository's experimental passive closure was not among them.**
[`nvf-passive-closure.md`](../../validation/rcwa/nvf-passive-closure.md) (2026-09-13) records
`passive_nvf.passive_nvf_assemble`, the `K^(−1/2)`-normalized congruence
`Z^H [Q^H E Q + G^H A G] Z`, which stays consistent when `C` is not a projector — the defect that
ruled out the congruence closing above — and is passive by theorem. No solve selects it, and its
spectra were not compared with fmmax here. At the two single points, on the same matrices
(measured 2026-10-02):

| Case | Closing | `R + T` | `T` |
| --- | --- | --- | --- |
| B disks, 852 nm / 121 | symmetrized + floor (this page) | 0.8158 | 0.7460 |
| | passive, unfloored | 0.8340 | 0.6836 |
| | passive, `<1/eps>` floored | 0.8563 | 0.7373 |
| A holes, 807 nm / 441 | symmetrized + floor (this page) | 0.9404 | 0.0086 |
| | passive, unfloored | 0.9512 | 0.0088 |
| | passive, `<1/eps>` floored | 0.9503 | 0.0088 |

Its anti-Hermitian part is non-negative to rounding in all four, and on the holes it lands inside
fmmax's 0.007–0.010. On the disks the floor still moves its `T` by 0.054: their pole sits at
845 nm, and passivity turns the pole's gain into loss without removing the pole. Whether it is as
accurate as the floored symmetrized closing over the spectra is open; if a passive closing
replaces this one, the floor stays.

## What this does not touch

- **The projector's own transition band.** `||C^2 − C|| = 0.25` is a statement about the normal
  field's series, and the shape path's analytic field is the one whose `P_yx` on the a-Si ellipse
  zigzags with truncation (0.2814 / 0.3026 / 0.2907 / 0.2956 / 0.2936 at 121 / 441 / 961 / 1225 /
  1681) where the raster path with `raster_normal_method='contour'` does not (0.2926 → 0.2926).
  Not this change's subject; the numbers are unchanged by it.
- **Truncation shape.** fmmax's Jones-direct with *circular* truncation is the one formulation that
  settled monotonically on both gold cases; every square truncation, wavebit's included, swings by
  several nanometres between neighbouring truncations. See `fourier-truncation-shape.md` for why
  the circle is not implemented; the gold results are a reason to re-measure that decision.
- **Li's rule and the Laurent rule**, which do not invert `<1/eps>` this way and had no pole to
  close.
- **The experimental passive closure**, `passive_nvf.passive_nvf_assemble`, which inverts
  `<1/eps>` itself, unfloored. No solve selects it; a caller who hands it a metal's matrices gets
  the unfloored inverse, pole included (table above).
- **Tensor layers.** A scene with a tensor material anywhere in it — a shape, a background or a
  tensor raster, on any layer — is built entirely by the Popov–Nevière route,
  `[[A]] @ inv([[B]])` (`tensor_shapes.build` for shapes, `pn_eps_conv_matrices` for rasters),
  or under `factorization='li'` by tensor Li, and neither reaches `_nvf_assemble`. So **no
  layer of such a scene is floored**, a metal included. Whether that inverse carries the same
  pole was not swept. At the one point checked,
  the gold disks at 852 nm / 121 with the glass background spelled `diag(2.25, 2.25, 2.25)`, the
  tensor route returns `R + T = 0.9682` where the floored scalar route returns 0.8158. That is
  not gain, but it is not the same answer either.
