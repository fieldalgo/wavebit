# Boundary conditions and power in complex half-spaces, audited

> **Historical API:** snippets and descriptions using `wavebit.compat` on this page require a historical checkout.
> The namespace is absent from current installations; the pre-retirement baseline is
> [revision `3c8a50f8`](https://github.com/fieldalgo/wavebit/tree/3c8a50f8a6506240169538957e99648a4b18ca3a).
> Reproduce recorded experiments with their own cited commit and environment; their measurements have not been rerun.


Written: 2026-09-22; the implementation stages at the end were added through 2026-09-23.
Measured on the working tree over `a1d23fc`. The investigation compared the paper, the
implementation and independent calculations without modifying the solver. Every section up
to and including "Where the code has to change" records the state **before** the fix; the
"Stage" sections record the fix. Probe scripts, raw outputs and run logs were not committed;
the numbers quoted here are the record.

**2026-10-03:** `check_physical` and `UnphysicalResultWarning`, which several sections below
fix or describe, were removed afterwards; see [`result-guards.md`](../../reference/common/result-guards.md). The power
quantities this audit set up (`power_entering`, the NaN for a complex transverse wavevector)
are unchanged.

Targets:
- `src/wavebit/observations/power.py` — `wave_power`, `net_power`, `incident_power`, `transverse_k_is_real`
- `src/wavebit/_common/conventions.py` — `incident_phase_index`
- `src/wavebit/rcwa/_engine/engine.py` — `_incident_kxy`, `_half_space_index`
- `src/wavebit/rcwa/_engine/solve_1d.py`, `src/wavebit/rcwa/_engine/scattering.py` — the efficiency call sites
- `tests/rcwa/test_half_space_power.py`, `tests/rcwa/test_direct_wavevector.py`, `tests/rcwa/test_lossy_media.py`

## Verdict

For a general passive, isotropic, non-magnetic medium, the case where **the incidence side is
lossless and only the exit side absorbs** is physically well defined. In this single-interface
check Wavebit's reflectance and interface transmitted power agreed with the Fresnel and
Poynting calculation. Transmitted power here means the power **entering** the substrate, not
the power that survives an infinite traverse of the absorbing substrate.

**An absorbing incidence medium** also admits a power definition once a reference plane and the
incident wave are fixed. But Wavebit's current `R`/`T` are not implemented to that definition.
That some amplitude checks pass is a separate matter from computing the power correctly.

The statement in `docs/reference/common/conventions.md` that "it depends on the definition, so differences
between computation paths are allowed" does not justify this defect. With the same input and
the same reference plane, every computation path must return the same physical quantity.

## What the paper covers

Moharam et al., *Formulation for stable and efficient implementation of the rigorous
coupled-wave analysis of binary gratings*, JOSA A 12, 1068–1076 (1995). Read from the PDF, with
the pages carrying equations checked as images.

- p.1069 eq. (7) and p.1072 eq. (52) separate the propagating and evanescent branches of the
  outgoing diffracted waves by comparing real magnitudes. That classification does not carry
  over to a complex refractive index as it stands.
- The TM transmission efficiency of p.1072 eq. (45) keeps the real part of the numerator
  separate from the incidence-side normalization factor. Substituting a complex incident index
  into it is not a derivation of its own.
- Treating absorption in the grating material and defining power in an absorbing incidence
  medium are different scopes.
- The universal claim "the refractive index of every half-space must physically be real" does
  not follow from the paper. It is a reading of the conditions under which the given equations
  apply, not a quotation of a sentence saying that the two outer indices must be real.

NIST SCATMECH is a real RCWA implementation worth comparing. In its mode that computes only
reflection for top incidence it allows an absorbing substrate; its other modes require a
non-absorbing substrate. It is an example of drawing the support boundary by observable.
[NIST RCW_Model](https://pages.nist.gov/SCATMECH/docs/rcwmodel.htm)

## The independent reference

At a single planar interface, the power of the incident wave alone, extrapolated to the
interface, was taken as the reference. With `q = kz/k0` and `β = k_parallel/k0`:

```text
q_j² = ε_j μ_j − β²
TE: use the E amplitude, Y_j = q_j / μ_j
TM: use the H amplitude, Z_j = q_j / ε_j
```

Writing `a` for whichever of `Y` or `Z` applies to the polarization, the boundary conditions
give

```text
r = (a₁ − a₂) / (a₁ + a₂)
t = 2 a₁ / (a₁ + a₂)
R = |r|²
T = |t|² Re(a₂) / Re(a₁)
```

The TM `t` here is an H-amplitude ratio; to compare it with Wavebit's public E amplitude it was
multiplied by `(n₁/μ₁)/(n₂/μ₂)`. These formulas were used in this check under three
conditions: a real transverse wavevector, isotropic media, and nonzero incident power.

In an absorbing incidence medium the net power actually entering at the interface generally
differs from `1−R`, because the incident and reflected waves interfere. For the same amplitude
variables it is

```text
P_enter / P_inc = Re[(1+r)* a₁ (1−r)] / Re(a₁)
```

where the star on `(1+r)*` is complex conjugation. At a single interface this equals the `T`
above. So `R+T≠1` alone was not taken as evidence of an error in an absorbing-incidence
problem. [Byrnes, §4.3–4.4 and Appendix B](https://arxiv.org/html/1603.02720v5)

## Reproduction

NumPy complex128, wavelength 500 nm, zeroth diffraction order only. A single interface was
represented as a uniform layer of zero thickness. TE/TM, 1D/2D and ETM/SMM were compared.
Angles are in radians.

| Case | Independent | Wavebit |
|---|---|---|
| air → ε=4+0.6i, μ=1, θ=0.45, TE | R=0.139374746, T=0.860625254 | agrees |
| same interface, TM | R=0.090828202, T=0.909171798 | agrees |
| air → n=0.18+3i, μ=1, θ=0.45, TM | R=0.922979811, T=0.077020189 | agrees |
| n=1.5+0.3i → air, μ=1, normal incidence, TE | R=0.053627760, T=0.984227129 | R agrees, T=0.946372240 |
| same interface, 2D TM | R=0.053627760, T=0.984227129 | R=0.049502548, T=0.946372240 |
| same absorbing medium on both sides, zero thickness, TM | R=0, T=1 | 1D T=1, 2D T=0.923076923 |

To rule out the zero-thickness representation as the cause, the power between two planes 75 nm
apart inside a uniform absorbing medium, with no reflecting interface, was also checked. With
`ε=2.16+0.9i`, `μ=1`, i.e. `n=1.5+0.3i`, at normal incidence

```text
T = exp(−4π Im(n) d / λ) = 0.568083605878
```

1D TE/TM and 2D TE agreed. 2D TM gave `0.524384866964`, about 7.69% low. The same defect
reproduced for top and bottom incidence and for the second- and first-order paths. There is no
reflected wave in this case, so a difference in how interference is normalized cannot explain it.

Minimal reproduction:

```python
import numpy as np
from wavebit.compat import legacy as wavebit

common = dict(
    backend='numpy',
    top=2.16 + 0.9j,
    bot=2.16 + 0.9j,
    structure=np.full((1, 2, 2), 2.16 + 0.9j),
    thickness=[75.], wavelength=500., period=[1000., 1000.],
    truncation_order=0, theta=0., phi=0.,
)
for psi in (np.pi / 2, 0.):
    result = wavebit.rcwa(psi=psi, **common).solve()
    print(result.T.sum())
# TE: 0.5680836058777344
# TM: 0.5243848669640628
```

Since 2026-10-13 this solver interface is `wavebit.compat.legacy`, imported as above; run after
the fix, both lines print the TE value.

What this investigation checked:

- 11 single-interface setups × 2 polarizations × 2 paths × 2 algorithms = 88 calls. The
  negative-index cases among them are described separately below.
- Two reference planes in a uniform absorbing medium: 8 combinations of incidence side,
  formulation (second- or first-order) and TE/TM.
- 273 passed and 70 deselected from the existing absorbing-media tests. Selected: substrate
  reflection, conservation of the power entering the substrate, reflection/transmission
  amplitudes, and the test that pins the path difference under absorbing incidence.
- This investigation does not vouch for every diffraction order of a general pattern or for
  every backend.

Why the existing test passed was also checked. The then-current
`test_absorbing_superstrate_efficiency_is_convention_bound` checked the 1D-to-2D TM difference
factor as given instead of the correct power. In stage 1 it was replaced by
`tests/rcwa/test_lossy_media.py::test_absorbing_incident_reflectance_matches_airy`, which checks, at
normal incidence and at a real transverse wavevector `n_top·sinθ = 0.5`, for both ETM and SMM,
that the 1D and 2D TM reflectances agree with the Airy `|r_p|²` within `1e-12` and that the two
paths' `r_p` amplitudes agree within `1e-12`.

## Causes in the implementation

1. `src/wavebit/rcwa/_engine/solve_1d.py::sp_efficiencies` uses the form `Re(a_out / a_inc)`. That differs
   from `Re(a_out) / Re(a_inc)`, which a power ratio needs, whenever the incident factor is
   complex. The pure-1D path and `scattering.py` have the same form.
2. The coupled s/p path applies one complex normalization factor to both polarizations even in
   a complex incidence medium. The true incident power has to be computed from the E/H
   amplitudes and the per-polarization admittance. At a complex incidence angle the s/p
   weighting needs checking too.
3. `check_physical` (removed 2026-10-03) skipped the power check when the incident index is complex. The
   general absorbing-incidence cases reproduced above returned values with no warning.
4. In `engine.py::_incident_kxy`, a complex refractive index times a real incidence angle makes
   the transverse wavevector complex too. That is an incident wave whose intensity also varies
   along the surface. Per-order power summation, which assumes a real transverse wavevector,
   cannot simply be extended to it. The power error also reproduced at a single interface with
   a complex incidence angle chosen to make the transverse wavevector real, so this defect is
   not caused by the complex transverse wavevector alone.

## A branch-selection problem found in magnetic half-spaces

The claim in `engine.py::_half_space_index` that "for passive ε, μ, εμ lies in the upper
complex half-plane" does not hold in general.

```text
ε = μ = −1 + 0.1i
ε μ = 0.99 − 0.2i
principal sqrt(ε μ) = +1 − 0.1i
index of the passive outgoing wave = −1 + 0.1i
```

The outgoing wave must decay away from the structure and carry energy in that direction. In
this example the physical admittance `n/μ=1` equals air's, so at normal incidence R=0 and the
interface T=1. The current implementation picks the opposite branch and a single-interface
solve raises `Singular matrix`.

Whether exact impedance matching is the cause was also checked. With the exit medium changed
to `ε=−2+0.1i`, `μ=−1+0.1i`, the independent calculation gives `R=0.029282057`,
`T=0.970717943`. The current result is `R=34.150606141`, `T=−33.150606141`, with an
unphysical-result warning. A substrate with positive real parts, `ε=4+0.2i`, `μ=1.5+0.1i`,
passed.

Since 2026-10-04 a magnetic half-space takes the signed index `sqrt(ε)*sqrt(μ)`, and both
examples above solve to the independent values (`R=0`, `T=1`; `R=0.029282057`,
`T=0.970717943`); see [`negative-index-media.md`](../../theory/rcwa/negative-index-media.md).

## Scope of support and direction of the fix

- There is ground for keeping a passive absorbing substrate under a general lossless incidence
  medium. The reference plane of `T` has to be stated as the interface between the structure
  and the substrate.
- For an absorbing incidence medium the current `R`/`T` must not be offered as verified
  efficiencies. The actual incidence face depends on `incidence_side`, so checking only `top`
  is not enough either.
- Supporting it means unifying the normalization on the Poynting power at the reference plane
  and separating `power_entering` from layer absorption. It should be tested against an
  independent power reference while preserving the cases where the existing amplitudes are
  right.
- Separate incidence at a real transverse wavevector from incidence at a complex one, and define
  the per-order power and mode branches of the latter separately.
- Branch selection for a magnetic outer medium is not guaranteed by the principal value of
  `sqrt(ε μ)` alone.
- Anisotropic `top`/`bot` is currently refused at input. Implementing it means designing the
  outer-medium eigenmodes, the incident mode and the power normalization together; it is not a
  matter of deleting a scalar check. This investigation did not verify such an implementation.

No external solver code or new dependency was brought in.

## Direct runs of fmmax and RETICOLO

The same day, on a follow-up request, fmmax 1.7.1 and RETICOLO V10 were run directly in the
existing external-validation environment. The Wavebit solver code was not modified.

### Defining the physical problem

Light travelling from a source at a finite position to the interface is physically possible
even in an absorbing incidence medium. But even the incident wave alone loses power as
`exp(−αL)` with propagation distance `L`. Holding the source power finite and sending `L→∞`,
the incident power that reaches the interface goes to 0. So a constant incident amplitude at
infinity cannot serve as the reference the way it does for a lossless outer medium.

This experiment normalized by **the power of the incident wave alone, extrapolated to the
interface**. The net power combined with the reflected wave was computed separately to check
the boundary conditions and the energy flow. General support for a complex incidence medium
needs this reference plane, wavevector and normalization definition to be explicit in the API.
[Byrnes, Appendix B](https://arxiv.org/html/1603.02720v5)

### fmmax 1.7.1

TE and TM were each computed for 12 structure/incidence conditions: an air–glass interface,
interfaces with one or both sides absorbing, 75 nm of propagation inside one absorbing medium,
a finite-thickness film under an absorbing incidence medium, and an incidence medium with the
optical constants of gold. Normal incidence and a real transverse wavevector `kx/k0=0.5` were
used.

The mode amplitudes from `eigensolve_isotropic_media` and `stack_s_matrix` were converted to E/H
with `fields_from_wave_amplitudes`, and `0.5 Re(Ex Hy* − Ey Hx*)` was computed directly. The
power evaluated separately for the incident, reflected and transmitted waves was kept distinct
from the net power of the incident and reflected waves together. Evaluating each separately
with `amplitude_poynting_flux` agreed as well.

| Case | Independent theory | fmmax |
|---|---|---|
| uniform medium n=1.5+0.3i, 75 nm of propagation, λ=500 nm | T=0.568083606 | 0.568083606 for both TE and TM |
| n=1.5+0.3i → air, normal incidence | R=0.053627760, T=0.984227129 | agrees for TE and TM |
| same interface, kx/k0=0.5, TE | R=0.076235897, T=0.970032766 | agrees |
| same interface, kx/k0=0.5, TM | R=0.034812863, T=0.995225744 | agrees |

Over the 24 cases the largest errors were `4.45e−16` in R, `1.07e−14` in T, `3.56e−15` in the
interface net power and `5.67e−16` in the complex amplitudes. So under these conditions fmmax
can compute the local power of an absorbing incidence medium correctly. This does not verify
arbitrary patterns or complex transverse wavevectors in general.

It was also confirmed that different uses of the function give different physical quantities.
Passing the incident and reflected waves together to `amplitude_poynting_flux(a,b,...)` puts
the interference term into the returned components; the first return value must not be taken
as the incident-only power. Dividing that way at the normal-incidence interface above gives
R=0.034055728, T=0.965944272, a different normalization from this table. The denominator in
this comparison always came from `amplitude_poynting_flux(a,0,...)` or from the incident-only
E/H.

### RETICOLO V10

40 runs in GNU Octave 9.4.0: 10 conditions × 1D/2D × TE/TM. A real `k_parallel` was given to
`res1` and the standard incidence and diffraction-efficiency results of `res2` were read. An
empty result array was recorded as not returned, not summed as 0.

- The 4 lossless air → glass controls agreed at R=0.04, T=0.96.
- In the 8 air → absorbing medium cases the reflectance agreed with theory. The standard
  transmission-efficiency array on the absorbing exit side was empty.
- The 28 absorbing-incidence-medium cases raised no run error, but the standard incident order
  and the reflection and transmission efficiency arrays were all empty.

This matches the RETICOLO manual's premise of a positive real incident index. The observation
that the standard efficiency results are empty is not extended to a claim that every internal
field computation or separate feature of RETICOLO is impossible.

SHA-256 of the `res1.m` and `res2.m` used:

```text
res1.m b2c79f61342c3796ca04f06a075e0d5adbfb06295454a2af8590072830192528
res2.m 3a2e19a6b543db0e888dc16b732d695a24bf5053e776791e0a7b2bc4a2935575
```

### Environment and provenance

No new package was installed and no external source copied. fmmax was 1.7.1 in the existing
`fwref` environment, MIT licence confirmed. The runtime was JAX/JAXlib 0.10.2 (Apache-2.0),
NumPy 2.4.6, SciPy 1.17.1 (BSD-family and bundled licences), opt_einsum 3.4.0 (MIT) and
ml_dtypes 0.5.4 (Apache-2.0). The compiler-runtime notices bundled with NumPy/SciPy were
checked too. This is an internal numerical check that redistributes none of those libraries or
executables.

For RETICOLO V10, `license.id=cc-by-4.0` was confirmed through the official Zenodo API. The
existing GPL Octave was used as an external runner; no RETICOLO/Octave source or executable was
integrated into Wavebit or redistributed. The authors of the RETICOLO used here are
J. P. Hugonin and P. Lalanne.
[RETICOLO V10 official release](https://zenodo.org/records/14631951),
[RETICOLO manual](https://arxiv.org/abs/2101.00901),
[fmmax licence](https://github.com/invrs-io/fmmax/blob/main/LICENSE).

## Follow-up check that scoped the fix

2026-09-22, same working tree: **without modifying the solver, only the power of the existing
public amplitudes was re-evaluated**. The aim was to separate errors in solving for the fields
from errors in converting them to power. No new dependency or external implementation was
introduced.

### What re-evaluating the existing amplitudes already fixes

Eight reference structures with a real transverse wavevector, isotropic outer media and uniform
layers were checked: a lossless control, an absorbing exit medium, an absorbing incidence
medium, 75 nm of propagation in one absorbing medium, a finite film under an absorbing
incidence medium, and an incidence medium with complex ε·μ of positive real part. Top/bottom
incidence, TE/TM/mixed polarization, 1D·conical·2D·first-order and the supported ETM/SMM
combinations came to **304** solves. The first-order path included only ETM, the one currently
supported.

- Largest error of the public complex reflection/transmission amplitudes against the
  independent boundary-condition/Airy calculation: below `8.25e-16`.
- Largest error of R/T recomputed from those amplitudes against the independent calculation:
  below `1.00e-15`.
- For the lossless-incidence control the existing R/T agreed as well.
- This verifies the reflection/transmission amplitudes. It does not mean the full internal
  field, convergence of general patterns, or every backend and derivative was verified.

| Case, 2D TM | Returned now | Existing amplitudes, power re-evaluated | Independent |
|---|---:|---:|---:|
| n=1.5+0.3i → air, normal incidence: R | 0.049502548 | 0.053627760 | 0.053627760 |
| same interface: T | 0.946372240 | 0.984227129 | 0.984227129 |
| 75 nm of propagation in the same absorbing medium: T | 0.524384867 | 0.568083606 | 0.568083606 |

The last case has no reflected wave. So this error is not a missing interference term but
**the power conversion/normalization of amplitudes that were already right**.

Using the public s/p amplitudes as they are, the power weight of one order with `q=kz/k0`,
dropping the common unit factor, is

```text
w_s = Re(q / μ)
w_p = |n / μ|² Re(q / ε)
P_inc ∝ w_s,inc |a_s|² + w_p,inc |a_p|²
```

So each outgoing order's `w_s |r_s|² + w_p |r_p|²` or `w_s |t_s|² + w_p |t_p|²` just has to be
divided by **the same true incident power**. The p amplitude in these formulas follows
Wavebit's current public amplitude definition; when the internal TM H amplitude is used
directly, the weight is `Re(q/ε)`. Replacing every `real(a / denom)` with
`real(a) / real(denom)` does not by itself settle the s/p and mixed-polarization normalization.

Separately, a uniform absorbing medium with two diffraction orders incident at once was checked
8 times: normal fundamental incidence, `(0,0)` amplitude 1 and `(1,0)` amplitude `0.5i`, period
1000 nm, wavelength 500 nm, propagation distance 75 nm. The theoretical TM transmittance is
`0.564563926081`. Currently 1D returns `0.564589751100` and conical/2D/first-order return
`0.565551868830`. Re-evaluating the per-order power of the existing amplitudes brought every
path within `1e-15` of theory. The plain `Σ|a_m|² Re(q_m)` in `_incident_flux` likewise has to
become the true power weight for the polarization and the medium.

### What needs the incidence setup itself to change

`engine.py::_incident_kxy` uses `n_inc sin(theta)`. A real angle in an absorbing incidence
medium generally makes the transverse wavevector complex. At `n=1.5+0.3i`, `theta=0.4`,
`phi=0` the value actually returned now is

```text
kx/k0 = 0.584127513463 + 0.116825502693 i
```

At wavelength 500 nm, moving 1000 nm along x multiplies the incident wave's `|E|²` by about
`0.053069913`. That is a physical incident wave whose intensity also varies across the
surface; simply dropping the imaginary part solves a different problem.

An incidence input that can specify a real transverse wavevector is a suitable first scope for
the fix. The 304-solve check above produced that condition with `theta=arcsin(beta/n_inc)`.
Supporting complex transverse wavevectors as standard efficiencies needs the power integral
over the unit cell, and the reference plane and transverse energy flow, to be defined
separately. Because the common Bloch factor decays, power cross terms between different Fourier
orders do not in general vanish in the cell integral either, so the per-order diagonal weights
for a real transverse wavevector must not be reused.

### Confirmed counterexamples that re-evaluating efficiency does not fix

Branch selection in magnetic outer media also reproduced.

- `ε=μ=-1+0.1i`: the current n is `1-0.1i`; the passive outgoing wave's n is `-1+0.1i`. The
  current solver raises `Singular matrix`. The correct interface is impedance-matched, R=0, T=1.
- `ε=-2+0.1i`, `μ=-1+0.1i`: currently R=`34.150606141`, T=`-33.150606141`. The correct values
  are R=`0.029282057`, T=`0.970717943`.

Here the boundary conditions that produce the reflection/transmission amplitudes are already
wrong, so renormalizing the result cannot fix it. `_half_space_index` and `boundary_kz` have
to be changed to take ε·μ and the true power direction into account together. For a passive
half-space at a real transverse wavevector the rule must pick the wave that decays outward, and
for an undamped propagating wave pick by power direction, not by the sign of phase advance.
This rule is not extended unconditionally to complex transverse wavevectors or gain media.

### Required changes, in order

1. **Reference plane and returned quantities:** take the incident-only power as the denominator
   at the top face of the structure for top incidence and at the bottom face for bottom
   incidence. Keep R/T as per-wave power ratios, and compute separately `power_entering`, the
   net power that the combined incident and reflected field sends into the structure. At a real
   transverse wavevector with one-sided incidence the finite structure's absorption is
   `power_entering - T`; under absorbing incidence it is not replaced by `1-R-T`. Here
   `power_entering` too is normalized by the incident power.
2. **Shared power computation:** unify the pure-1D efficiency formula, `sp_efficiencies`, the
   corresponding SMM formula and the multi-order `_incident_flux` on one E/H power definition.
   At a real transverse wavevector this can be computed straight from mode amplitudes; the
   field need not be redrawn on a full spatial grid for it. There is ground for preserving the
   amplitudes already shown to be right, the NVF/Fourier matrices and the internal layer
   connections.
3. **Absorbing-incidence input:** define an input that specifies a real transverse wavevector
   and keep it distinct from complex-transverse-wavevector computation. `polarization`
   currently normalizes the Euclidean norm of the electric-amplitude pair, so it must not be
   described as normalizing to unit true power. The API has to decide explicitly whether to
   keep the current incident-amplitude definition and compute a separate true-power
   denominator, or to add a power-normalized mode input.
4. **Magnetic outer media:** fix the branches of n and kz and the power-direction check of the
   incident mode. Fix `_incident_flux` judging power direction by `Re(kz)>0` alone at the same
   time.
5. **Replace checks and explanations:** fix `check_physical` skipping every check under complex
   incidence, and check the budget of net power and finite-layer absorption. Do not judge
   absorbing incidence to be an error by `R+T>1` alone. Replace the existing test and documents
   that pin the path difference as the right answer with agreement checks on the same physical
   quantity.

After implementation, multiple diffraction orders of general patterns, the full s/p response,
top/bottom incidence, complex μ, NumPy/JAX/PyTorch and derivatives have to be verified. This
check scoped the fix; it did not complete the solver fix or the verification of that whole
scope.

## Where the code has to change, and what depends on what

Further investigation, 2026-09-22. What follows is **the specification written before the
implementation**. The solver and the existing tests were not modified. Line numbers refer to
the working tree at the time of this investigation.

### 1. Merge the efficiency conversion into one power computation

The efficiency computation does not live in one function. The actual call relations are:

```text
ETM, pure 1D ────────────────────── the efficiency formula inside solve_1d()
ETM, conical / 2D / first-order ─── final_matching() → sp_efficiencies()
SMM, pure 1D ────────────────────── the efficiency formula inside smm_1d()
SMM, conical / 2D ───────────────── _smm_sp() → sp_efficiencies()
```

| Location | Current problem | Required change |
|---|---|---|
| `src/wavebit/rcwa/_engine/solve_1d.py:474–486`, `solve_1d` | both reflection and transmission form the complex ratio first, then take the real part | compute the true incident power first and divide each outgoing wave's real power by it |
| `src/wavebit/rcwa/_engine/solve_1d.py:653–684`, `_incident_flux` | the weight for multiple incident orders is only `Re(kz)`; it takes no polarization, ε or μ | take each incident order's actual s/p amplitudes and ε·μ and sum the power |
| `src/wavebit/rcwa/_engine/solve_1d.py:838–875`, `final_matching` | the specified-polarization and pure s/p results share one denominator | compute the denominator from the incident amplitudes of each of `incident`, `s_inc`, `p_inc` |
| `src/wavebit/rcwa/_engine/solve_1d.py:886–905`, `sp_efficiencies` | uses `Re(admittance / complex_denom)` | compute `Re(admittance) / real_incident_power`, taking the power of the actual incident polarization as the denominator |
| `src/wavebit/rcwa/_engine/scattering.py:525–549`, `smm_1d` | the pure-1D efficiency formula is duplicated separately | call the same shared power computation as ETM |
| `src/wavebit/rcwa/_engine/scattering.py:700–747`, `_smm_sp` | builds three incident columns but uses one efficiency denominator | compute the power for each column of the actual `c_inc` and divide output column by output column |

The p input of `sp_efficiencies` is not the public `r_p` but the phase-multiplied **H
amplitude**. With `q=kz/k0` and internal interface amplitudes `c_s`, `c_p`, the power of one
separated one-way wave in an isotropic half-space at a real transverse wavevector, dropping the
common unit factor, is

```text
P_order = |c_s|² Re(q/μ) + |c_p|² Re(q/ε)
P_inc   = sum(P_order for incident orders)
R_order = P_reflected_order / P_inc
T_order = P_transmitted_order / P_inc
```

Applying it to the public p amplitude needs the `|n/μ|²` conversion of the previous section.
Pure-1D TM sets the incident H amplitude to 1 while the coupled path converts the incident E
coefficient to H, so each path has to pass **the incident amplitude it actually used** to the
shared function. (Still true of the pure TM solve's power arithmetic; since 2026-09-27 the
field state it hands to reconstruction is rescaled to a unit incident E, as the coupled
path's is.) Changing only the formula while keeping today's interface, where each path
passes a different complex denominator, leaves the same problem.

A separate internal module (e.g. `src/wavebit/observations/power.py`) is the right home for the shared function.
Its inputs are the media, the per-order wavevectors and the actual incident, reflected and
transmitted amplitudes. Computing the power needs neither a reconstruction of the full spatial
field nor a second eigenproblem. NumPy/JAX/PyTorch use the same formula, and the finiteness and
positivity of the incident power are checked when the value can be read.

This is not a TM-only fix. In the earlier reference case `n_inc=1.5+0.3i → air` at normal
incidence, **the TE transmittance is also currently 0.946372239748 where the correct value is
0.984227129338**. TM served as the representative case because under this condition its
reflectance is wrong too, and the error also shows in the uniform-medium propagation example
above.

### 2. Fix the incident columns of the pure-polarization results returned alongside

A separate error was also confirmed. For multi-order incidence the general incident column of
`final_matching` uses `Kz_top @ delta` (`solve_1d.py:802–807`), but the columns for the pure
s/p results always use `kz_inc * delta` (`:817–823`), which applies the central order's
wavevector to every incident order. **Fixing the denominator leaves this amplitude error in
place.**

- Setup: all one material, two incident orders `(0,0): 1`, `(1,0): 0.5i`, wavelength 500 nm,
  period 1000 nm, thickness 75 nm. There should be no reflection.
- Lossless ε=2.25 and absorbing ε=(1.5+0.3i)², TE/TM, conical/2D/first-order and the supported
  ETM/SMM: 20 checks in all.
- In lossless ETM, R of `incident` is about `1e-31`, while R of `s_inc`/`p_inc`, which mean the
  same pure polarization, is `0.0001754677013`. The complex amplitudes differ by up to about
  `0.0151650`.
- The same mismatch occurs in absorbing ETM. In the 8 SMM comparisons the amplitude difference
  is 0.

The fix is to build `incident` and the pure s/p columns with the same incident-column
constructor: apply the per-order wavevectors and the same `delta` to every column, then compute
each column's true incident power separately. The current multi-incidence comparison in
`tests/rcwa/test_excitation.py` mostly checks `incident` only, so a regression check is needed that
`s_inc`/`p_inc` match an `incident` with the same polarization specified directly.

The 304-solve amplitude agreement of the previous section is for `incident` under single
incidence, and the 8-solve multi-incidence power check is also for `incident`. This finding is
an error in result columns those checks did not examine. It occurs even without loss, and must
not be confused with the efficiency error of absorbing incidence.

### 3. Connect the net entering power to the returned values

The recommendation is to add to `ResultSub` the power ratio that the combined incident and
reflected field sends into the structure. Its provisional name is `power_entering`; it is not
an attribute implemented now. It is computed by summing each order's tangential E/H at the
interface and then taking `0.5 Re(Ex Hy* - Ey Hx*)`, which includes the incident–reflected
interference term. The incidence direction counts as positive, and the result is divided by the
incident-only power at the same interface. The value is a sum over the whole cell; the
interference term is not distributed into the R/T arrays.

Places to change with it:

- `src/wavebit/rcwa/_engine/solve_1d.py:908`, `_pack`: pass on each result column's net entering power.
- `src/wavebit/observations/diffraction.py:41`, `ResultSub`: connect the result in the constructor and
  `_FIELDS`. The specified-polarization and unpolarized-average delegation of `Result` follows
  the same definition.
- `src/wavebit/rcwa/_engine/engine.py:278`, `_efficiency_only`, `:1845`, `_finish`, `:2870–2874`, result
  construction: tidy the return contract together so that the new value is not lost on the
  efficiency-only path. Whether to change the existing tuple return or to state it per
  supported stage needs a decision.
- JAX's `ResultSubJax` uses the shared class's pytree handling, so check the `jit` return,
  serialization and differentiation boundary with the new field as well.
- `src/wavebit/_common/backend.py:361`, `check_physical`: do not skip the whole check for a complex
  incident index. For a passive problem at a real transverse wavevector, check that R/T are
  non-negative and check the absorption budget `power_entering - sum(T)`. Treat gain media
  separately.

The reference is the public structure's top face for top incidence and the bottom face at z=0
for bottom incidence. Parts of the kernel use propagation coordinates with the public top/bot
swapped, so the power has to be computed by incidence-side/exit-side role while keeping the
medium swap of `engine.py:2080–2138` and `_kernel_incidence_side`.

### 4. Separate fixes for oblique-incidence input and wavevector branches

The centre of **real transverse wavevector input** is `engine.py:1727`, `_incident_kxy`. Keep
the two inputs, converting from an angle and using a directly specified wavevector, clearly
distinct. The API has to state the wavevector's unit (a true inverse length, or a value
normalized by k0), the check when it is given together with an angle, and the rule that fixes
the s/p reference direction at normal incidence. The imaginary part of a complex wavevector
produced in an absorbing medium must not be silently discarded.

A new input has to be carried through to:

- `domain.py:632`, `Incidence` and `as_kwargs`.
- `spec.py:874`, `_incidence_from_dict`, `:972`, `_incidence_to_dict` and the spec keys.
- the `engine.py` constructor, `_incident_kxy`, the 1D selection and incidence validation, and
  `:2883`, the record of solve conditions in `_reconstruction_scene`.
- `src/wavebit/rcwa/_backends/jax_backend.py:253`, `CHILD_NAMES` and pytree restoration.
- a check through `order_kxy`/`_separable_kxy` that the solve and the field reconstruction read
  the same wavevector, and a check that incidence with a y component is not sent to pure-1D
  TE/TM.

This is not a prerequisite for fixing absorbing **normal incidence only**. The normal-incidence
efficiency regression can be fixed first and oblique-incidence support added afterwards.
General cell-integral support for complex transverse wavevectors is separate work from this
real-transverse-wavevector power formula.

**Wavevector branches in magnetic media** are fixed together in `engine.py:177`,
`_half_space_index` and `conventions.py:195`, `boundary_kz`. The parameter of `boundary_kz` is
currently named `eps`, but every solver call site actually passes `n**2=εμ`. The interface,
which cannot tell ε from μ, has to be sorted out first. A power-direction check that accounts
for permeability is needed, and for passive problems at a real transverse wavevector the
decaying branch and the lossless-limit power direction have to be chosen consistently.

The call sites are the two solvers in `solve_1d.py`, `solve_2d.py`, `solve_2d_first_order.py`, and
the two computation paths in `scattering.py`. The `Re(kz)>0` check in `_incident_flux` has to
change too. Changing the meaning of a general conversion function such as
`refractive_index_from_eps` unconditionally, or substituting the same rule into `mode_q` for
internal finite layers, is outside the scope of the fix. The boundary conditions of half-spaces and
the mode representation of internal finite layers have to be kept distinct.

### 5. Order of the fix and pass criteria

1. Add a regression that catches the existing error with independent boundary-condition/Airy
   power. At the time of the investigation (`a1d23fc`),
   `test_absorbing_superstrate_efficiency_is_convention_bound` at
   `tests/rcwa/test_lossy_media.py:424` pinned the wrong inter-path ratio, and the follow-up check
   also saw **1 passed**. Replace that check with a path-agreement check under the new power
   definition. (Replaced in stage 1 by
   `tests/rcwa/test_lossy_media.py::test_absorbing_incident_reflectance_matches_airy`.)
2. Fix the shared power function and every efficiency call site. Fix the multi-incidence column
   error of the pure-polarization results with its own regression. Efficiencies for the same
   input must agree across `answers='efficiency'/'amplitude'/'field'`.
3. Connect the net entering power and the absorption budget. Check
   `power_entering = 1 - sum(R)` under lossless incidence, `power_entering = sum(T)` at a single
   interface under absorbing incidence, and `T = exp(-2 Im(kz_physical) d)` over a uniform
   stretch of one medium.
4. Connect the real-transverse-wavevector input and the branch fix for magnetic half-spaces.
   Check agreement for the same problem with the incidence side swapped, and between an angle
   input and the equivalent real wavevector input.
5. Widen verification to multiple diffraction orders of real patterns, s/p conversion, oblique
   lattices, NumPy/JAX/PyTorch, JIT and derivatives. There is currently no ground for changing
   the internal NVF/Fourier matrices; this item preserves the existing amplitude computation.
6. Correct `docs/reference/common/conventions.md:408`, the explanation in `result.py`, and the part of
   `docs/validation/rcwa/time-convention-audit.md` that read the TM difference as a difference of convention.
   Also fix the wording in `conventions.py:277`, `engine.py` and the docs that describes the
   length normalization of the polarization coefficients as normalization to unit true power.

No new dependency or external code was introduced. What this investigation additionally ran
was 20 small reproduction calculations and 1 existing test. It does not amount to a full
regression run or a completed solver implementation.

## Stage 1: power and efficiency of separated waves

2026-09-22, following the investigation above, **the efficiency conversion and the
incident-power denominator** were fixed. The investigation results and fix specification in
the preceding sections record the state before the fix.

The following computation was gathered in `src/wavebit/observations/power.py`:

```python
wave_power = real(amplitude * conj(amplitude)) * real(admittance)
efficiency = wave_power / incident_power
```

Here the admittance is `kz/mu` for internal electric-field amplitudes and `kz/epsilon` for
internal magnetic-field amplitudes. The incident power sums the power of the actual incident
orders. The `n/mu` factor that converts the public electric p coefficient to the internal
magnetic amplitude is included. The complex ratio is never formed first with its real part
taken afterwards.

- Pure 1D ETM/SMM, conical/2D ETM/SMM and first-order ETM use the same power definition.
- The denominators of the specified-polarization and pure s/p results are computed separately.
- The incident columns of the pure s/p results returned alongside also use each order's
  wavevector. The single-incidence amplitude computation and the internal NVF/Fourier matrices
  were not changed.
- The exit-side epsilon is passed through to the shared efficiency function, so the epsilon
  provided is used rather than the refractive index squared again.
- The test that pinned the per-path efficiency difference under absorbing incidence as the
  right answer was replaced by an agreement check against the Airy reflectance.

The values returned after the fix are below. They are what the fixed solver returned directly,
not values converted afterwards.

| 2D TM, normal incidence | Before | After | Independent |
|---|---:|---:|---:|
| n=1.5+0.3i → air, R | 0.049502548 | 0.053627760252366 | 0.053627760252366 |
| same interface, T | 0.946372240 | 0.984227129337540 | 0.984227129337540 |
| 75 nm of propagation in the same absorbing medium, T | 0.524384867 | 0.568083605877734 | 0.568083605877734 |

The new `tests/rcwa/test_half_space_power.py` checks independent boundary conditions, exponential
decay within one medium, identity of the pure-polarization and specified-polarization results,
agreement across return stages, and JAX/JIT and PyTorch results differentiated with respect to
an absorbing incidence material. The existing whole-result wiring test was also fixed to
compare with the public/kernel medium swap and the incidence side matched: its previous direct
call had the two layers in a different order from the public API call, so it compared
different structures.

The supported and verified scope of this stage is **real transverse wavevectors and isotropic
half-spaces**. Cell power integrals for general complex transverse wavevectors, a new
transverse-wavevector input API, and branch selection at negative ε·μ were not fixed.
Returning `power_entering`, and the absorption-budget check that uses it, are also left to the
next stage. `1-R-T` must not be used as the structure's absorption under absorbing incidence on
the strength of this efficiency fix.

Final verification: **1,134 passed, 32 skipped** across 10 related test files. The 32 skips are
the unsupported combinations that apply mixed polarization to pure 1D. After the new power
tests' amplitude comparison was tightened to an absolute `2e-12`, that file was rerun on its
own: **391 passed, 32 skipped**. The 10 derivative checks of absorbing-incidence power, JAX JIT
and PyTorch included, also passed. The full repository suite was not run.

Commits were not made per stage; they are made together once the whole fix and its
verification are in order.

## Stage 2: net power at the boundary and absorption of the whole structure

`result.power_entering` is a scalar: the net power at the boundary, computed with the incident
and reflected waves together, divided by the incident-only power. It is positive when power
enters the structure, for top and bottom incidence alike. `result.absorption` is:

```python
result.absorption == result.power_entering - result.T.sum()
```

This is the absorption of every finite-thickness layer combined. Absorption in the `top` and
`bot` half-spaces is excluded. If the incidence medium is lossless,
`power_entering = 1 - R.sum()`. A lossless incidence medium alone does not make the structure's
absorption 0; the finite layers must be lossless as well.

The shared `net_power` function combines the boundary fields first:

```python
primary = incident + reflected
partner = admittance * (incident - reflected)
net = real(primary * conj(partner))
```

The primary amplitude is E for TE and H for TM. Swapping the two factors in the Poynting
vector leaves the real part unchanged, so one function serves both. The complex admittance has
to be kept through the product for the interference term to survive. That term is not mixed
into the separated waves' `R`, `T`. The distinction was checked against
[Byrnes, §4.3–4.4 and Appendix B](https://arxiv.org/html/1603.02720v5#A2). For magnetic media
the same E/H boundary relations use the actual ε and μ. No external implementation code was
copied and no dependency added.

Scope and return contract:

- Connected in pure 1D ETM/SMM, conical ETM/SMM, 2D ETM/SMM and first-order ETM.
- Uses the incident power and interference of each of `incident`, `s_inc`, `p_inc`.
- The two scalars are provided under `answers='efficiency'` too. This stage uses the boundary
  amplitudes right after the solve; it does not escalate to a higher answer level or
  reconstruct the internal field to keep them in the result.
- The low-level efficiency return is `(R, T, power_entering)`; the public `Result` shape is
  unchanged. The corresponding low-level call tests and benchmarks were updated.
- `power_entering` is part of `ResultSub`'s JAX pytree, and `absorption` is a property computed
  from it. The differentiation path is not cut.
- `UnphysicalResultWarning` checks `T.sum() > power_entering` when the net power is available.
  It does not warn merely because `R+T>1` under absorbing incidence. The check assumes a
  passive structure, so gain materials need separate interpretation.
- General cell integrals for complex transverse wavevectors were not implemented in this
  stage. In that case the two new scalars are `NaN` rather than a wrong absorption. The
  real-transverse-wavevector test allows only floating-point rounding error.

Representative returned values for normal TM incidence, wavelength 500, finite-layer thickness
75:

| Incidence → finite layer → exit medium | R | T | power_entering | absorption |
|---|---:|---:|---:|---:|
| n=1.5+0.3i → zero thickness → air | 0.053627760252366 | 0.984227129337540 | 0.984227129337539 | ≈ 0 |
| ε=2.25 → ε=4+0.8i → air | 0.139756850733125 | 0.578016970599071 | 0.860243149266875 | 0.282226178667804 |
| n=1.5+0.3i → ε=4+0.8i → air | 0.100542006662143 | 0.575339770196985 | 0.856258762263978 | 0.280918992066993 |

Verification covers power continuity at a single interface, exponential decay in one absorbing
medium, heat absorption from a separate integral of `ω ε0 Im(ε) |E|² / 2` inside the layer, the
power budget of a lossless pattern with two incident orders, the total absorption of a layer
split into three, and the JAX JIT return and derivatives and PyTorch derivatives of the public
result. The references are the independent boundary-condition linear systems and numerical
integrals inside the tests.

Result: **1,638 passed, 48 skipped** across 16 related test files. The 14 three-layer-split
checks added after that run started also all passed. The 4 JAX JIT and PyTorch
public-result/derivative checks were rerun alongside; excluding duplicates that is
**1,652 passed, 48 skipped**. The skips are pure-1D mixed-polarization combinations. This is not
a full repository run.

The regression run had 3 warnings. One is the existing field-reconstruction state record
reading a PyTorch differentiable tensor as a Python complex; the actual power derivatives
agreed with independent finite differences. Two are unphysical-power warnings from the existing
branch-tolerance and NVF derivative checks. This stage neither hid the related warnings nor
loosened test tolerances.

After this stage, an explicit real-transverse-wavevector input API and passive half-space
branch selection at negative ε·μ still remain. It must not be read as completing support for
cell integrals of general complex transverse wavevectors. Not yet committed.

## Stage 3: phase incidence angle for a lossy incidence half-space

> **Extended on 2026-10-04.** This stage defined a real angle as the direction of phase advance,
> which is the direction the light travels in every positive-index medium and is unchanged for
> all of them. A negative-index lit side, which did not exist here then, runs its phase against
> its power, and `theta` names the light's direction there too: `k_par = s*N*sin(theta)` with
> `s = sign(Re(sqrt(eps)*sqrt(mu)))`, `-1` for such a medium. Read `docs/reference/common/conventions.md`
> ("Real-angle incidence") and `docs/theory/rcwa/negative-index-media.md` for the current statement; the
> record below is the stage as it was.

Real `theta`, `phi` are defined as **the direction of phase advance inside the selected
incidence half-space**. The incident wave's intensity is uniform along the structure surface,
so the transverse wavevector is real. No fictitious air layer or finite absorbing layer is
added above the incidence face, and no approximation discards the material's imaginary part.
The existing `theta`, `phi` input specifies this wave; no separate direct
transverse-wavevector input key was added.

With `epsilon_r*mu_r = a+i*b`, `N=|Re(k)|/k0`, and the inward normal component written as
`q/k0 = N*cos(theta)+i*alpha`, the following two equations have to hold together:

```text
N² - alpha² = a
2 N cos(theta) alpha = b

N² = (a + sqrt(a² + b²/cos²(theta))) / 2
kx/k0 = N sin(theta) cos(phi)
ky/k0 = N sin(theta) sin(phi)
```

For a lossless positive-index medium `b=0, a>0`, so this reduces exactly to the familiar
`kx/k0=n*sin(theta)*cos(phi)`. In a general lossy medium the direction of phase advance and the
direction of the Poynting vector can differ, so `theta` is not read as the angle of power flow.
The Maxwell relations for the phase and attenuation vectors were checked against
[Mishchenko et al. on inhomogeneous plane waves, eqs. (11)–(15)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7580800/).
The scope that assumes uniform incident intensity on the surface is the same as
[Byrnes §2](https://arxiv.org/html/1603.02720v5#S2). The implementation was written by deriving
the relations above directly; no external code was copied and no dependency added.

`conventions.incident_phase_index` computes the small root in rationalized form to avoid
cancellation at negative `a`. `engine._incident_kxy` calls it, so NumPy, JAX, PyTorch, every
ETM/SMM path of either formulation and field reconstruction use the same incident wavevector. With
`incidence_side='bottom'` the actual ε·μ of `bot` is used, and a change of wavelength in a
dispersive material comes through the evaluated material.

The complex incidence angle feature is kept. If `theta` or `phi` has a nonzero imaginary part,
the existing formal `n*sin(theta)` parameterization is used. That is not an analytic
continuation of the real phase-angle model into the complex plane. Differentiating through a
real phase angle needs a real design variable. Cell-integrated power for general complex
transverse wavevectors remains out of scope.

For lossy phase incidence with `abs(theta)>=pi/2`, uniform surface incidence and a finite
wavevector cannot both hold, so a concrete input raises an error. Inside JAX JIT, where the
value cannot be read, NaN is returned rather than a wrong finite result. The existing grazing
treatment for lossless and complex incidence angles, and the separate branch-selection problem
at negative ε·μ, remain.

For the example ε=4+3i, μ=1, λ=1,000 nm, θ=30°, φ=0: `kx=0.006771390373184058 nm⁻¹`, `ky=0`.
An s-incidence solve with a lossless finite layer ε=2.25 and the same exit half-space gives:

| R | T | power_entering | absorption |
|---:|---:|---:|---:|
| 0.1452272767102451 | 1.0131886090510112 | 1.0131886090510107 | ≈ 0 |

Even with T above 1 under absorbing incidence, the net power including the incident–reflected
interference matches the outgoing power, so the finite structure absorbs nothing.

Verification uses materials built from prescribed phase and attenuation vectors, an
independent bisection inversion of the dispersion relation, Fresnel/Airy amplitudes, E/H
boundary power, the heat-absorption integral inside layers, the field's phase along the surface
and decay along the normal, JAX JIT/vmap, and PyTorch derivatives. The reference checks that
had computed lossy incidence angles in the old form were switched to the independent inversion
so that they compare the same incident wave. Derivatives are not taken along the imaginary
direction of a real incidence angle, and continuity of the first and second derivatives with
respect to material loss at the lossless point is also checked.

### Verification results

- New phase-incidence-angle checks: **165 passed**. Run on their own, and also included in the
  regression below.
- Related regression: **2,052 passed, 48 skipped**. It covered lossy media, multiple incidence,
  polarization, magnetic materials, top/bottom incidence, dispersive materials, ETM/SMM, the
  result API and derivatives, and transverse-wavevector propagation on oblique lattices. Long
  high-order checks of shape integration itself were not in the regression scope for this
  incidence change.
- Sphinx HTML: the full build passed with warnings as errors. intersphinx, which fetches the
  inventories of external sites, was disabled for this verification run only, because of
  network restrictions. Equations were confirmed to render as math elements, not as raw source.
- `git diff --check` passed. No new dependency and no copied external source.

The regression run's 7 warnings: 1 scalar conversion in PyTorch's field-reconstruction state
record, 2 unphysical-power warnings from the existing branch-tolerance and NVF derivative
checks, 1 single-precision branch warning, and 3 input-precision warnings from the
material-precision-change check. Power derivatives agreed with independent finite differences;
no warning was hidden and no tolerance loosened.

## Direct transverse-wavevector input (2026-09-23)

`wavebit.rcwa(kx=..., ky=...)` takes a real transverse wavevector in the fixed xyz frame
directly. With lengths in nm the input is in nm⁻¹; internally it is multiplied by
`wavelength/(2*pi)` to feed the existing solver's dimensionless wavevector. The
reciprocal-lattice components of the diffraction orders are added after that. `theta`·`phi` and
a direct wavevector are mutually exclusive, and omitting the direction means normal incidence.
Giving only one component makes the other 0. The directly specified physical transverse
wavevector is kept when the material or the wavelength changes. The normal wavevector and the
power are computed from the ε·μ of the actual incidence half-space.

The public input, `Incidence`, `JobSpec` serialization and sweeps, the JAX pytree,
NumPy/JAX/PyTorch, automatic differentiation with respect to the wavevector, and field
reconstruction all carry the same condition. Complex, non-finite and array inputs are refused.
An invalid value that cannot be read during JIT is passed on as NaN. A direct incident wave
goes through the same positive-incident-power check as multi-order incidence, so no efficiency
is attached to an evanescent wave that carries no incident power. General cell integrals for
complex transverse wavevectors were not added.

- `tests/rcwa/test_direct_wavevector.py`: 141 checks. It verifies that angle and direct-wavevector
  input agree in reflection, transmission and fields on patterned structures; independent
  Fresnel boundary calculations with magnetism and loss; seven computation paths; top/bottom
  incidence; wavevector derivatives of an independent heat-absorption integral; multi-order
  incidence power; oblique lattices; wavelength sweeps of dispersive media;
  JIT/vmap/pytree/field computation; and invalid input.
- Related regression: **2,015 passed, 48 skipped**. That run included the first 128 of the new
  checks. The final run, bundling the added checks and the spec input checks, was
  **191 passed**; it overlaps the earlier regression, so the two numbers are not added.
- The 3 regression warnings are the scalar conversion in the existing Torch state record, a
  single-precision eigenvalue branch, and a notice about the requires_grad default in Torch
  array conversion. No tolerance was loosened.
- Two worked examples were run: for lossless ε=2.25, θ=30° and `kx=0.75*k0, ky=0` both give
  `T=0.7395557697684563`.
- No new dependency or external source code was introduced.
