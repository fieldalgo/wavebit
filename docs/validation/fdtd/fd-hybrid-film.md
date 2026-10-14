# C8: a mixed Debye and Drude/Lorentz film

Written: 2026-10-07

Targets: `src/wavebit/fdtd/_prepare/hybrid_film.py`,
`src/wavebit/fdtd/_prepare/hybrid_resolution.py`, `src/wavebit/materials/time_domain.py`,
`src/wavebit/fdtd/_kernel/poles.py`, `src/wavebit/fdtd/_kernel/observe.py`,
`src/wavebit/fdtd/_prepare/checks.py`, `src/wavebit/fdtd/_prepare/plan.py`,
`src/wavebit/fdtd/_prepare/report.py`.

## TL;DR

**The bounded implementation, six-row calibration, 918 targeted checks
and 13 actual mutation witnesses are complete.** Exact source restoration
is followed by a passing 234-case baseline. The registered 40-worker
campaign is complete: 47/48 self-orders are second and one mixed; 47/48
final direct pairs are second and one noise-unresolved. Full regression
passes 3,669 cases with eight skips, the separate slow tier passes all
81, and all four frozen-reference gates are bit-exact.
The feature covers one isolated flat film with both passive
Debye relaxations and passive Drude/Lorentz oscillators, finite declared
counts and unequal oscillator damping. The calibration is disclosed known
data, not a convergence result or an arbitrary-material accuracy claim.

The implementation couples physical polarization coordinates while retaining
the bilinear Debye and centred oscillator frequency symbols. A half-step
Debye state and an affine electric-field split make the update explicit.
The observer needs a different discrete storage, including a negative
force-square correction. Four vector arrays per declared pole preserve
the required clock without inverting a possibly singular time update.

An exact-arithmetic closed-system energy identity supports a sufficient
curl-plus-oscillator CFL condition. The implementation deliberately retains an
additional nonnegative Debye policy budget, and uses physical high-frequency
permittivity in the curl bound. Neither that proof nor the conservative
bulk spectral floor establishes floating-point passivity, PML stability,
finite-time settling or scattering accuracy.

Earlier single-family routes and their evidence keep their existing scope.
This bounded mixed-family extension does not admit curved dispersive
interfaces, dispersive/lossy hosts, junctions or geometry/material/band
derivatives. It does not complete C8 or alter the unpassed C7 verdict.

## Bounded contract

| Item | Bounded admission |
| --- | --- |
| Geometry | Exactly one concrete public `Box`, `angle=0`, `range=None`, voxel discretization, no partner or junction. |
| Extent | Exactly two axes uniformly span the full grid with the inherited one-cell transverse overhang; finite normal thickness at least `dx`; normal support strictly inside the domain. |
| Absorbers | Every support cell stays clear of every absorbing face. No interface wraps across a wall or enters an absorbing layer. |
| Material | Exact `Dispersive`, positive finite isotropic high-frequency permittivity, `mu=1`, with at least one declared exact `Debye` and one exact `Drude` or `Lorentz`. |
| Poles | Debye `delta_eps>=0`, `tau>0`; Drude positive plasma frequency; Lorentz nonnegative strength; nonnegative oscillator damping, with existing constructor restrictions retained. |
| Host | Plain exact positive isotropic nonmagnetic lossless `Material`. |
| Values | Concrete finite geometry, material values and spectral requests. Unrepresentable derived active values receive a named refusal. |
| State | Public initial fields excluded; material memory begins at rest. Private seeded-state tests exercise the algebraic update and energy identities; they do not admit public initial fields. |
| Backends | NumPy, JAX and Torch have six disclosed calibration rows and targeted checks; arbitrary-material accuracy and campaign convergence do not follow from backend agreement. |

Dispatch is based on the declared families, including exact zero-strength
Debye or Lorentz declarations. Slots retain their declared order and count;
duplicate parameters are not coalesced. A mixed declaration with one wholly
inactive family remains on this new realization's endpoint. Pure declared
Debye and pure declared oscillator lists retain their earlier routes.
Traced zero is not accepted as a concrete inactive declaration. No claim
about source-amplitude differentiation is made by the concrete geometry,
material and band guards.

## Coupled physical response

Let `f` be the component's film fraction, `g=1-f`, `a` the film's physical
high-frequency permittivity and `b` the host permittivity. Set

```text
A = f*b + g*a
eta = a*b/A,       alpha = g/A,       kappa = f*b*b/A**2       (normal)
eta = f*a + g*b,   alpha = 0,         kappa = f               (tangent).
```

For Debye strengths `Delta_i` and oscillator strengths `S_j`, define

```text
chi(s) = sum_D Delta_i/(1+tau_i*s)
         + sum_O S_j/(s*s+gamma_j*s+d_j)
epsilon_eff(s) = eta + kappa*chi(s)/(1+alpha*chi(s)).
```

Here `d_j=0` and `S_j=omega_p**2` for Drude, while Lorentz uses
`d_j=omega_0**2` and its declared strength. The normal expression equals
`1/(f/(a+chi)+g/b)`; the tangent expression is `f*(a+chi)+g*b`.
This reduction couples the complete susceptibility sum and does not
assume that the families can be averaged independently.

With physical Debye positions `m_i`, oscillator positions `q_j`, and
`P=sum(m)+sum(q)`, the continuous realization is

```text
tau_i*m_i' + m_i + alpha*Delta_i*P = kappa*Delta_i*E
q_j'' + gamma_j*q_j' + d_j*q_j + alpha*S_j*P = kappa*S_j*E.
```

Only active slots at positive fill enter weighted equations. Put
`rho=alpha/kappa`; the corresponding weights are

```text
R_D = diag(tau_i/(kappa*Delta_i))
M_O = diag(1/(kappa*S_j)),     R_O = diag(gamma_j/(kappa*S_j))
K = diag(1/(kappa*Delta_i), d_j/(kappa*S_j)) + rho*1*1^T.
```

`K` is positive semidefinite, with a positive definite Debye principal
block. The rank-one potential is included once over both families.
Unequal diagonal damping need not commute with this stiffness.

## Two discrete symbols and an explicit predictor

Write `h=dt`. Debye auxiliary positions `D` live at half steps, and their
physical integer-time output is
`m^n=(D^(n+1/2)+D^(n-1/2))/2`. Oscillator positions remain at integer steps:

```text
R_D*(D^(n+1/2)-D^(n-1/2))/h + K_DD*m^n + K_DO*q^n = 1_D*E^n

M_O*(q^(n+1)-2*q^n+q^(n-1))/h**2
 + R_O*(q^(n+1)-q^(n-1))/(2*h)
 + K_OO*q^n + K_OD*m^n = 1_O*E^n.
```

Averaging adjacent Debye equations gives its trapezoidal equation for
physical `m`. For `z=exp(-i*omega*h)`, the intended susceptibility is

```text
s_D = (2/h)*(z-1)/(z+1)
Q_j = (z-2+1/z)/h**2 + gamma_j*(z-1/z)/(2*h) + d_j
chi_h = sum_D Delta_i/(1+tau_i*s_D) + sum_O S_j/Q_j
P/E = kappa*chi_h/(1+alpha*chi_h).
```

At `z=-1`, use the Debye physical-output zero limit; this does not require
every auxiliary coordinate to vanish. Singular steady-state denominators
must not be treated as finite comparison values. A common midpoint update
for every family would change the oscillator symbol, so it is not this
candidate's update.

The positive Debye local matrix is `L_D=2*R_D/h+K_DD`. Its diagonal plus
rank-one form admits linear work in the declared count. A concrete scalar
form used by the implementation is

```text
t_i = 2*tau_i/(2*tau_i+h)
bare_i = kappa*Delta_i*h/(2*tau_i+h)
c_i = bare_i/(1+rho*sum_D bare_i)
c = sum_D c_i
T = sum_D t_i*D_i^(n-1/2)
m_i^n = t_i*D_i^(n-1/2) + c_i*(E^n-rho*(T+sum_O q_j^n)).
```

Set `y_i^n=m_i^n-c_i*E^n`, predict
`D_i^(n+1/2)=2*m_i^n-D_i^(n-1/2)`, and advance every oscillator using the
same old physical `P^n`. With `den_j=1+gamma_j*h/2`,

```text
q_j^(n+1) = ((2-d_j*h*h)*q_j^n - (1-gamma_j*h/2)*q_j^(n-1)
             + kappa*S_j*h*h*E^n - alpha*S_j*h*h*P^n)/den_j.
```

Then predict the next transformed Debye output using the new half-step
state and the complete new oscillator sum:

```text
y_i^(n+1) = t_i*D_i^(n+1/2)
            - rho*c_i*(sum_D t_k*D_k^(n+1/2)+sum_O q_j^(n+1)).
```

No future electric field enters this predictor. Ampere becomes

```text
(eta+c)*(E^(n+1)-E^n) + sum_D(y^(n+1)-y^n)
                         + sum_O(q^(n+1)-q^n) = h*C*H^(n+1/2).
```

The existing `before_E` scale remains `dx/dt`; the update permittivity is
`eta+c`, while physical high-frequency permittivity remains `eta`.
All old sums must be completed before any corresponding history is replaced.

## Four arrays per pole and the observer clock

The proposed term reserves `P`, `previous`, `delta` and `aux` tuples, each
with one three-component compact-region array per declared pole. At the
observer reading associated with `E^n`:

| Memory | Debye slot | Oscillator slot |
| --- | --- | --- |
| `P` | Predicted `y^(n+1)` | Predicted `q^(n+1)` |
| `previous` | `y^n` | `q^n` |
| `delta` | `y^(n+1)-y^n` | `q^(n+1)-q^n` |
| `aux` | `D^(n+1/2)` | `vbar^n=(q^(n+1)-q^(n-1))/(2*h)` |

Reconstruct physical Debye output from **previous** plus `c_i*E^n`, not
from predicted `P`. After the electric update the advance uses `E^(n+1)`
to construct the next prediction and centred velocity before overwriting
older oscillator history. It never divides by `1-gamma*h/2`, which can
vanish at a valid `gamma*h=2`.

Thus the material payload is `4*(N_D+N_O)` vector arrays, or
`12*(N_D+N_O)*number_of_support_cells*dtype_itemsize` logical bytes.
Exact inactive declarations still reserve their arrays. This is a concrete
layout, not a minimum-memory theorem or total process/device memory.
The separate proposed report correction counts ordinary three-array pole
memory; the hybrid integration additionally needs the four-array metadata
and an allocation oracle. Source, observer, diagnostic, temporary and
allocator overhead remain outside that report field.

## Modified storage and its limits

Let `theta=h*h/4`, `P^n=sum(m^n)+sum(q^n)`, and define the weighted
oscillator force residual

```text
F_j^n = d_j*q_j^n/(kappa*S_j) + rho*P^n - E^n.
```

In the repository's doubled energy convention, the new material correction
to the electric weight `eta+c` is

```text
sum_D |m_i^n|**2/(kappa*Delta_i)
 + sum_O d_j*|q_j^n|**2/(kappa*S_j)
 + rho*|P^n|**2
 + sum_O (1+theta*gamma_j**2)*|vbar_j^n|**2/(kappa*S_j)
 - theta*sum_O kappa*S_j*|F_j^n|**2
 - c*|E^n|**2.
```

Every weight and sum acts locally before reducing cells or components.
Apply inactive/family masks before forming the shared physical sum and
force, and apply the same component/region selection to fields, memory and
broadcast weights. The shared rank-one potential occurs once. Reusing the
old oscillator `-Re(E*Delta P)` correction or an all-midpoint energy would
not produce this storage.

The full closed-lattice storage `W^n` adds physical `eta*|E^n|**2` and
`Re(<H^(n-1/2),H^(n+1/2)>)` to the positive material potential, modified
kinetic and negative force-square terms above. For source-free adjoint Yee
curls its exact arithmetic loss identity is

```text
W^(n+1)-W^n = -2*h*sum_D tau_i/(kappa*Delta_i)
                         * |(m_i^(n+1)-m_i^n)/h|**2
             -h*sum_O gamma_j/(kappa*S_j)
                         * (|vbar_j^(n+1)|**2+|vbar_j^n|**2).
```

For complex fields use Hermitian products and real cross terms. This is a
global closed-system identity; a cropped observation region can exchange
flux. Nonnegative unequal damping preserves the loss sign. Drude dark
polarization directions may have zero restoring potential; the statement
does not guarantee a bounded arbitrary dark position or strict positivity
on every memory coordinate. An active Debye family can dissipate energy
even when all oscillator damping coefficients are zero. The lossless
conservation control therefore has an inactive Debye family and zero
oscillator damping; it is distinct from the active-Debye, zero-oscillator-
damping dissipation control.

At an inactive-Debye endpoint the physical oscillator recurrence reduces
to the old centred recurrence. Its energy reading generally differs:

```text
W_hybrid - T_existing = h*sum_O gamma_j/(kappa*S_j)*|vbar_j|**2.
```

The two coincide for zero oscillator damping. Equality of recurrence or
physical transfer therefore must not be replaced by equality of damped
observer values. At an inactive-oscillator endpoint the physical Debye
transfer retains the old bilinear law, but modal reductions and internal
memory coordinates need not be bit-identical. Old declared-family routes
are preserved separately.

## Physical-permittivity CFL and retained Debye policy

A sufficient exact-arithmetic positivity condition is

```text
h*h*(c_C+r_O) < 4
c_C = ||eta^(-1/2)*C||**2
r_O = max_active d_j + sum_active S_j/a.
```

The oscillator part follows from the normal identity
`alpha+kappa/eta=1/a`; the tangent coefficient is no larger. The full
modified-potential proof factors its positive-semidefinite quadratic-form
matrix and does not invert the Drude null directions. The physical `eta`
must be used in the curl bound; substituting the larger update `eta+c`
could weaken the condition incorrectly.

The implementation keeps a stricter policy rather than using the minimal sufficient
material budget from that proof. Define Debye moments
`M1=sum_active Delta_i/tau_i` and `M2=sum_active Delta_i/tau_i**2`, then

```text
r_D = max(M2/a, max_actual_Yee_nodes [kappa/eta*(M2+alpha*M1*M1)])
r_policy = r_O + r_D.
```

The added `r_D` is a conservative inherited policy, not an extra term
required by the hybrid storage proof. It is a maximum over the actual
component fills and the bulk Debye sum; individual pole maxima do not
replace the sum. Both inactive families contribute exactly zero without
evaluating unused reciprocals or frequency squares. Policy sums use exact
rational arithmetic followed by upward float64 conversion.

This condition concerns the closed Yee/material system. It does not extend
the absorber/PML stability contract, guarantee temporal frequency accuracy,
or promise that a chosen source duration settles a stiff relaxation.

## Representability and the bulk spectral floor

The implementation evaluates each distinct concrete fill's scalar coefficients as
exact fractions before final float conversion. Active positive drives,
Debye shifts, inverse energy weights and required force/coupling weights
must remain positive finite values. Unused endpoints branch before
reciprocal arithmetic; a positive underflow is not silently pruned.

An active Debye gain `2*t_i-1` must round strictly between -1 and 1. If it
rounds to either endpoint, the implementation refuses it by name, even though the
input `tau` is positive and finite. This rule has no tolerance cutoff and
does not merge active and inactive declarations. Representability checks
alone do not certify computed matrix spectra, cancellation in small output
increments, or float64 nonnegativity of the modified storage.

For `x=omega**2` a sufficient continuum bulk wavenumber bound is

```text
K_bound**2 = a*x_high
 + sum_D Delta_i*x_high/sqrt(1+tau_i**2*x_high)
 + sum_O S_j*max_band[x/sqrt((d_j-x)**2+gamma_j**2*x)].
```

Each Debye summand increases with positive x. Each oscillator maximum
uses both band endpoints and the interior candidate
`2*d_j*d_j/(2*d_j-gamma_j*gamma_j)` when its denominator is positive and
it lies inside the band. The combined mixed-family modulus is not assumed
monotone. Candidate decisions and squared bounds use exact rational
arithmetic; square-root upper conversions are checked outward. An active
undamped in-band resonance is refused, while an exact zero strength is
skipped before its singular denominator.

Every positive concrete source band must satisfy `dx*K_bound<=2*pi/8`
in film and host. Every requested monitor, Resolution and explicit PML
wavelength separately needs at least eight phase cells and one skin-depth
cell, retaining the twelve/two warning policy. Geometry/material/band
tracers remain excluded. With neither source nor requested wavelength,
a warning must disclose that no spectral scale was certified.
This is a bulk scale floor, not a bound on all normal-interface resonances,
oblique longitudinal scales or measured r/t error.

## Targeted checks and actual mutation witnesses

Targeted attempt 1 collects and passes 918 cases, with exit zero and no
skips, expected failures or unexpected passes. The new hybrid file has
156 parameterized cases across nine functions; the state-byte file has
78 cases, including the mixed-film four-array allocation control. Earlier
film tests and the existing stencil refusal regression are also included.
Collection reports 3.21 pytest seconds (4.095501 wrapper seconds); the run
reports 106.64 pytest seconds (107.128116 wrapper seconds), with 1,045
warnings retained. The source identity is unchanged before and after.

The nine hybrid contracts exercised are:


| Test function | Independent witness |
| --- | --- |
| `test_hybrid_actual_recurrence_preserves_both_symbols_and_nyquist` | Actual state map and dense predictor versus the declared dual-symbol response, including the Debye physical Nyquist limit. |
| `test_hybrid_inactive_family_endpoints_order_and_memory_are_exact` | Fractions, inactive-family reductions, declared ordering and four-array allocation. |
| `test_hybrid_observer_clock_and_closed_yee_dissipation` | Weighted/cropped complex storage, actual observer clock and closed Yee loss identity: active Debye with zero oscillator damping still dissipates; pure lossless oscillators provide the conservation control. Also checks the damped old-storage distinction. |
| `test_oblique_hybrid_matches_complex_airy_on_each_backend` | Six disclosed backend rows against independently evaluated continuum amplitudes, after measured ceilings are accepted. |
| `test_unsupported_hybrid_films_are_refused_by_name` | Geometry/material doors and named derived-value range refusals. |
| `test_prepared_hybrid_plan_refuses_changed_identity_or_memory` | Either family's changed signature and missing/extra/wrong material memory. |
| `test_traced_hybrid_parameters_cannot_bypass_the_concrete_door` | JAX and Torch reverse/forward traced material parameters. |
| `test_hybrid_cfl_uses_physical_eta_and_both_conservative_rates` | Physical-permittivity curl and both retained material budgets, including inactive limits. |
| `test_hybrid_spectral_guard_covers_both_band_families_and_each_sample` | Separately decisive Debye band contribution, oscillator interior peak and declared wavelength floors. |

Mutation attempt 1 accepts all 13 actual source-defect witnesses across
the nine hybrid functions and the report-allocation function. Every
witness exits one with the intended assertion failure; original/mutant
snapshots, logs and restoration hashes are retained. The source is
restored exactly. The final clean hybrid-plus-allocation baseline passes
all 234 collected cases, exit zero, no skips, in 118.17 pytest seconds
(121.766595 wrapper seconds), retaining 313 warnings.

| Deliberate defect | Witness |
| --- | --- |
| Scale the Debye half-step gain | Actual dual-symbol recurrence |
| Double the oscillator damping clock | Unequal-damping oscillator recurrence |
| Shift the all-inactive permittivity | Exact inactive endpoint |
| Omit the negative force-square energy | Modified discrete storage |
| Observe the predicted coordinate instead of the current one | Observer clock |
| Quarter the oscillator drive | Disclosed oblique Airy regression |
| Remove the initial-field refusal | Unsupported-state door |
| Remove the prepared signature check | Changed-plan refusal |
| Detach traced Debye time | Concrete-input door |
| Omit oscillator strengths from the CFL rate | Both retained material budgets |
| Omit the Debye band contribution | Debye spectral floor |
| Omit the interior oscillator peak | Whole-band oscillator floor |
| Lose the four-array report override | Actual initialized-state allocation |

A fixture, import or runtime error does not count as a caught defect.
The targeted manifest, `targeted_attempt1.json`, and its raw logs/source
snapshots (the study's `fdtd_hybrid_film_controls` directory) are archived
outside the repository. So is the mutation manifest,
`fdtd_hybrid_film_mutations.json`, together with the exact before/after
sources, assertion logs and the restored baseline of its first attempt
(`fdtd_hybrid_film_mutations_attempt1`).


## Disclosed calibration and accepted regression ceilings

The two disclosed synthetic materials both have `eps_inf=1`:

| Family | Debye `(Delta,tau)` | Oscillators |
| --- | --- | --- |
| Mixed metal | `(0.25,0.2)` | Drude `omega_p=9`, `gamma=1`; Lorentz `(omega_0,gamma,S)=(8,2,10)` |
| Mixed dielectric | `(2,0.08)`, `(1,0.25)` | Lorentz `(5,0.8,10)`, `(8,1.5,20)`, `(11,2,15)` |

Their oscillator constituents relate to the preceding oscillator examples;
that history is not an unseen-outcome validation. The analytic preflight
passes before any hybrid field response. It checks the 40-cell whole-band
floor, passive sampled loss, independent Airy identities and negative
sampled real permittivity for the mixed-metal example. Its conservative
Courant-vacuum `analytic_n40_angular_dt_upper_bound` is 0.069180674, below
Nyquist. The bulk wave-number bounds are 14.946359 and 21.965800 for the
two families, respectively. These are bulk scale certificates, not
measured temporal-accuracy or interface-convergence claims.

Calibration attempt 1 completes six sequential runs: both families on
NumPy/JAX/Torch, vacuum host, thickness 0.2, 35-degree centre incidence,
p polarization, face fraction 0.3, exact transverse `Axis1D`, 40 cells,
120 periods, PML24 at reflection `1e-12`, Courant 0.5, source wavelength 1,
`df=0.1`, cutoff `1e-6`, and sampled wavelengths 0.9/1.0/1.1. Independent
continuum epsilon is the directly declared rational sum, followed by
slab admittances; production lowering is not the reference.

All six workers exit zero with stable fields, finite energy and unchanged
source identity, from 20:14:30 to 20:16:00 UTC on 2026-10-06. Each records
1,041 finite energy samples, 16,628 time steps and `dt=0.007216878365`.
The actual step and saved source band permit reconstruction of its
`omega*dt` range. Maximum absolute complex errors over the three sampled
wavelengths and all three backends are:

| Family | Maximum r error | Maximum t error | Maximum r decay | Maximum t decay |
| --- | --- | --- | --- | --- |
| Mixed metal | 0.002103337008 | 0.0009106264580 | 2.932265e-9 | 1.617117e-13 |
| Mixed dielectric | 0.002903755513 | 0.005908102518 | 1.052043e-8 | 1.494392e-12 |

The largest scaled backend difference is 2.336255e-15, below the fixed
128-double-epsilon budget 2.842171e-14. This checks agreement, not continuum
accuracy. Python 3.12.14, NumPy 2.5.1, JAX/jaxlib 0.11.0 and Torch 2.13.0
are recorded with one-thread settings. All rows retain the bounded-scope
accuracy warning. Mixed-metal rows also retain `AbsorberProximityWarning`:
clearance 0.775 is less than the local half-wavelength 0.840433. The fixed
geometry is not moved to remove that warning; registered duration/PML
controls are required to assess response sensitivity.

At 20:17:36 UTC, after these measurements were disclosed and reviewed,
the inherited candidate ceilings 0.005 for r and 0.010 for t were accepted
unchanged. Margins over the observed global maxima are 0.002096244487 and
0.004091897482. Acceptance is restricted to the two known 40-cell,
p-polarized, fraction-0.3 configurations. These thresholds do not decide
other campaign rows or establish a convergence rate. The test's pending
ceiling constant was then replaced with those accepted values; the exact
pre-acceptance test is archived. Production and the probe did not change.

The acceptance record is
`studies/fdtd/hybrid_film/records/fdtd_hybrid_film_calibration_acceptance.json`.
The calibration record `fdtd_hybrid_film_calibration_attempt1.json`, the
test-update record `fdtd_hybrid_film_calibration_acceptance_test_update.json`,
and all six raw result/log pairs with the pre-response helper/controller
snapshots (the `fdtd_hybrid_film_calibration_attempt1` directory) are archived
outside the repository.

The FD and package digests are respectively
`5aa63f0f692d85cf83b988d6f97cfa0b5835ec876e890fca976628df3fc08838` and
`d0d1be0ce8e1e7220a07d842abfd2bf438ac9525d1cb7005c7c84cc964910f94`,
using sorted relative POSIX paths, NUL separators and LF-normalized bytes.
Calibration, targeted checks, mutation restoration and registration record
that same production identity.

## Registered campaign and complete results

Registration at 20:25:32 UTC on 2026-10-06 freezes fingerprint
`f244a5875fe988f20065e06060af3bf0ba0ce26f9d6478f69bdf8d6dcf3126d4`.
It follows completed calibration, targeted checks and restored mutation
witnesses. It fixes the two families, fractions 0.3/0.1, s/p and five
sequential workers per group: 40/120/PML24, 80/120/PML24,
80/240/PML24, 80/240/PML48, 160/120/PML24. There are 40 base workers,
48 cellwise self-orders, 96 adjacent direct-error intervals and 48 final
direct intervals. The two matching coarse p/fraction-0.3 configurations
are previously exposed calibration responses, not unseen corroboration.

The registration discloses 83 known development artifact records and
five frozen source snapshots. Those 83 artifacts include calibration,
targeted/mutation exposures, raw logs, snapshots and outer execution
records; they are not 83 independent physical experiments. Before the
first registration, the collector was extended to include every completed
development record required by the final auditor. The old collector is
archived in the annotation
`fdtd_hybrid_film_registration_collection_fix.json`.
This provenance-only change precedes registration and changes no production,
probe, tests, mutation harness, numerical criteria or response. The
registration itself is the run record `fdtd_hybrid_film_registration.json`.
Both records are archived outside the repository.
The completed campaign and its cellwise exceptions are recorded below.

Only an otherwise valid unsettled 120-period base row may receive a single
240-period replacement, reusing its fixed long control when available;
an invalid capped row or control stops the campaign. The registered
order rule is an entire sensitivity interval within `[1.8,2.3]` for second,
an upper endpoint at most 1.75 for low, and other resolved intervals mixed.
It retains the `1e-10` signal floor and the one-tenth proxy-share gate.
The per-cell proxy is the 80-cell duration difference plus PML difference;
self changes use plus/minus twice that proxy, and direct errors use
plus/minus the proxy. These are empirical sensitivity intervals, not
rigorous fine-grid uncertainty bounds. Report every unresolved cell and
stop at the declared cap, without assuming all-second outcomes.

## Measured outcomes and the two exceptional cells

All 40 sequential workers complete with exit zero, without a replacement
or registered stop, from 20:25:49 to 20:42:40 UTC on 2026-10-06. Summed
worker wall time is 1,006.502191 seconds. Every raw result records stable
fields, finite energy and unchanged source identity. Maximum decay is
1.200072e-8, below the registered 1e-5 line. The completed procedure does
not imply that every order is second.

| Family | Self-orders: second / mixed / unresolved | All direct pairs: second / mixed / unresolved | Final direct pairs: second / unresolved |
| --- | --- | --- | --- |
| Mixed metal | 23 / 1 / 0 | 47 / 0 / 1 | 23 / 1 |
| Mixed dielectric | 24 / 0 / 0 | 48 / 0 / 0 | 24 / 0 |
| Total | 47 / 1 / 0 | 95 / 0 / 1 | 47 / 1 |

There are no low classifications and no vanishing self/direct signals.
All self intervals span 1.771050 to 2.092461. Resolved direct intervals
span 1.810054 to 2.064540; resolved final direct intervals span 1.881377
to 2.053072. The latter ranges exclude the unresolved pair and do not
replace the cellwise verdicts.

The sole mixed self-order is mixed-metal s reflection at fraction 0.3,
wavelength 1.1. Its interval is [1.771050189, 1.771278906], point order
1.771164545. The noise share is only 6.130636e-5, so the mixed verdict is
resolved; the interval is below the second-order band and above the low
cutoff. Its direct intervals are [1.810054080, 1.810137565] for 40-to-80
and [1.912753934, 1.913063512] for 80-to-160, both second. Improved direct
orders do not establish an asymptotic rate beyond this measured ladder.

The sole unresolved final direct pair is mixed-metal p reflection at
fraction 0.3, wavelength 0.9. Its errors at 80 and 160 are
3.504629827e-4 and 8.639503103e-5. The time proxy is 4.294664523e-6 and
the PML proxy 8.999052616e-8, giving `u=4.384655049e-6`. Consequently
`2*u/min(error80,error160)=0.1015024822`, above the fixed 0.1 limit. The
point order 2.020241825 is descriptive only; no resolved interval or
second-order acceptance is assigned. Its self interval
[1.941655846, 2.061439961] and coarse direct interval
[1.982706313, 2.027800109] are second. The noise gate is not relaxed for
this near-threshold cell.

The largest proxy across all cells is 4.394895248e-6, with separate maxima
4.304673197e-6 for duration and 9.911309316e-8 for PML sensitivity. These
empirical n80 controls are not rigorous uncertainty bounds on the n160
error. The campaign stops at its registered cap with the mixed and
unresolved outcomes intact.

| Family and coefficient | Maximum error at 40 | At 80 | At 160 |
| --- | --- | --- | --- |
| Mixed-metal reflection | 0.004051170484 | 0.001027802848 | 0.0002583865636 |
| Mixed-metal transmission | 0.001764240663 | 0.0004417767579 | 0.0001104888880 |
| Mixed-dielectric reflection | 0.006596787190 | 0.001642823080 | 0.0004104645604 |
| Mixed-dielectric transmission | 0.006504847998 | 0.001621706091 | 0.0004051403677 |

The coarse reflection maximum occurs in mixed-dielectric s reflection at
fraction 0.1, wavelength 0.9. It exceeds 0.005; that calibration ceiling
belongs only to the two disclosed p/fraction-0.3 regression configurations
and is not a campaign accuracy gate. The fixed campaign accepts cellwise
order intervals, not these global maxima. Both matching NumPy coarse
p/fraction-0.3 physical payloads repeat the known calibration exactly;
that repeat documents provenance rather than independent corroboration.

The complete report is
`studies/fdtd/hybrid_film/records/fdtd_hybrid_film_report.json`. The raw
40-row sequence (`fdtd_hybrid_film_campaign.jsonl`) and the 80 raw result/log
files (the `fdtd_hybrid_film_campaign_jobs` directory) are archived outside
the repository. The controller's saved-data audit exits zero at 20:44:14 UTC,
recorded in `fdtd_hybrid_film_campaign_audit_execution.json`, also archived
outside the repository.

An independent standard-library audit exits zero at 20:44:23 UTC. It
verifies 280 input hashes and reconstructs every cell without importing
the solver, probe, controller or criteria. Its independent cos/sin
transfer matrix differs from stored continuum references by at most
2.618456e-16; reconstructed cell arithmetic differs by at most
8.881784e-16. Five proof groups check the coupled response, affine clock,
modified energy/dissipation, material CFL identity and spectral critical
points. Those include finite exact-rational checks and the stated
exact-arithmetic algebra, with closed-system, representability and bulk
scale limitations; they add no PML energy or universal scattering theorem.

The independent report and observed child-exit record are
`fdtd_hybrid_film_independent_audit_attempt1.json` and
`fdtd_hybrid_film_independent_audit_attempt1_execution.json`; they, the raw
log and the exact auditor/runner archives are archived outside the repository.
Procedural and independent audits verify the records, including their
exceptions. They do not promote mixed or unresolved classifications to
second. These two three- and five-pole physical examples support only the
registered material/geometry/placement/polarization/wavelength scope,
not all declared counts or material parameters.

## Regression gates and evidence preservation

The completed evidence includes the analytic preflight, six disclosed
calibration rows, accepted known-row ceilings, 918 targeted passes,
13 mutation witnesses with a restored 234-case baseline, and the completed
40-worker campaign with controller and independent saved-data audits.
The one mixed self-order and one noise-unresolved final direct pair remain
explicit. The numerical validation pipeline exits zero on its fixed source:

| Gate | Collected | Passed | Skipped | Retained warnings | Pytest seconds | Wrapper seconds |
| --- | --- | --- | --- | --- | --- | --- |
| Full | 3,677 | 3,669 | 8 | 1,510 | 2,337.83 | 2,338.465899 |
| Separate slow | 81 | 81 | 0 | 50 | 1,133.04 | 1,133.551794 |

Both tiers have no expected failures or unexpected passes. All four
read-only freezers exit zero and reproduce 143/288/140/570 arrays bit for
bit, 1,141 in total, without writing the committed golden files. Stage 2
and stage 3 retain notices about historical case/freezer provenance;
these notices are distinct from their exact array comparisons.

The eight collection/run/freezer checks are recorded in the run record
`fdtd_hybrid_film_validation.json`; it and all raw logs (the
`fdtd_hybrid_film_controls` directory) are archived outside the repository.
The validation wrapper records
FD digest
`de8ebf7f5ae5fa58ee1b786a96f937236c02ed771d67b05d52b38457ceaa2e48`,
using the inherited sorted-relative-path plus LF-byte algorithm without
delimiters. It is a different hash definition from the NUL-delimited
campaign identity above. Completed regression gates do not change the
campaign's mixed or noise-unresolved scientific classifications.

Final documentation and preservation acceptance is a separate release
requirement. Its prescribed outputs are named
`fdtd_hybrid_film_final_docs.json`,
`fdtd_hybrid_film_final_evidence_audit.json` and
`fdtd_hybrid_film_final_evidence_audit_execution.json`.
Acceptance requires observed zero exits for documentation collection/tests
and Sphinx `-W`, recorded lint checks, unchanged document-source hashes
during verification, and a complete saved-data audit with an independently
recorded zero process exit. The preserved audit and documentation-runner
sources are `final_evidence_audit.py.txt.gz` and
`final_documentation_gates.py.txt.gz`. This article specifies the evidence
requirements without predeclaring their result. The execution records, which
are archived outside the repository with those sources, report it for
2026-10-06: documentation collection, the documentation contract (73
collected, 73 passed) and the warning-as-error Sphinx build each exited zero,
and the saved-data audit completed with exit zero.

The preceding oscillator milestone retains its frozen source evidence
and registered mixed/unresolved results. This extension preserves those
records and the old single-family routes. General C8 smoothing remains
incomplete and the earlier C7 investigation remains unpassed.
