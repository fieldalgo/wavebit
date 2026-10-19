# C8: a finite Drude/Lorentz sum in one film

Written: 2026-10-07

Targets: `src/wavebit/fdtd/_prepare/oscillator_sum_film.py`,
`src/wavebit/fdtd/_prepare/oscillator_sum_resolution.py`, `src/wavebit/materials/_time_domain.py`,
`src/wavebit/fdtd/_kernel/poles.py`, `src/wavebit/fdtd/_kernel/observe.py`,
`src/wavebit/fdtd/_prepare/checks.py`.

## TL;DR

**The bounded realization is implemented for two or more passive
Drude/Lorentz poles in one isolated flat film, including unequal damping
rates. Six disclosed calibration rows complete on NumPy, JAX and Torch,
and all 684 targeted cases pass. All ten actual mutation witnesses are
accepted, with exact restoration and a passing 148-case baseline. The
40-worker campaign is complete: 45/48 self-orders are second, two mixed
and one unresolved; 44/48 final direct pairs are second and four unresolved.
Full regression passes 3,434 cases with eight skips, the separate slow
tier passes all 81, and all four frozen-reference gates are bit-exact.** The normal
interface response couples physical polarization memories through their
sum. Keeping those coordinates avoids assuming that the restoring and
damping matrices can be diagonalized together.

One declared pole retains one memory slot. The recurrence adds a rank-one
coupling in work proportional to the declared count. Energy uses the
unshifted per-pole resonance terms and one additional term for the summed
polarization. With nonnegative damping, the corresponding discrete energy
dissipates; the existing summed material CFL bound remains sufficient
within the stated lossless-host film contract.

The scope keeps the earlier concrete, axis-aligned film geometry and
preserves the existing single-oscillator route. It does not combine Debye
and second-order poles, admit a dispersive host, or validate curved bodies,
junctions or geometry/material/band differentiation; mixed Debye/oscillator
films later got their own scope ([record](fd-hybrid-film.md), 2026-10-07). The prior
[single-oscillator](fd-single-pole-film.md),
[finite-Debye-sum](fd-debye-sum-film.md) and
[conductive-film](fd-conductive-film.md) records retain their own scope and
limits. General C8 smoothing remains incomplete and the
[C7 metasurface verdict](fd-metasurface-validation.md) remains unpassed.

## Bounded contract

| Item | Admission |
| --- | --- |
| Geometry | Exactly one concrete public `Box`, `angle=0`, `range=None`, with voxel discretization; no partner or junction. |
| Extent | Exactly two axes span the complete grid with at least one `dx` of transverse overhang; normal thickness is at least `dx`. |
| Clearance | Normal support is strictly inside the domain; all support stays clear of every absorbing layer. |
| Film | Exact `Dispersive`, finite positive isotropic `eps_inf`, `mu=1`, and at least two exact public `Drude` or `Lorentz` poles. |
| Poles | Concrete finite passive parameters satisfying their public constructors. Every damping rate is nonnegative. An inactive zero-strength Lorentz pole retains a slot but contributes no drive or energy. |
| Host | Exact plain lossless `Material`, finite positive isotropic permittivity and `mu=1`. |
| State | No initial field. Prepared geometry, material, grid, declared order/count and all memory slots must match at lowering. |

The geometry rule is invariant under permutation of x, y and z. Occupancy
is evaluated at each electric component's own Yee dual cell. Exact empty
and full endpoints are preserved; a small nonzero fill is not snapped away.
This is not a promise to represent every mathematically finite combination
of parameters: active coefficients, reciprocal energy weights and planning
rates must fit the numerical representation or receive a named refusal.

The one-pole Drude/Lorentz case continues through the existing implementation.
The new path is not a fit to independent effective Lorentz modes, and does
not need a different implementation for each finite count. Repeated pole
parameters may describe a reducible physical response without changing the
declared memory allocation. Existing source and monitor restrictions still
apply. No source-amplitude gradient correctness is claimed.

## The exact coupled harmonic response

Use `s=-i*omega` for the repository's `exp(-i*omega*t)` convention. The
declared bulk film is

```text
eps_film(s) = a + chi(s)
chi(s) = sum_j S_j/(s*s + gamma_j*s + d_j).
```

For a Drude pole, `d_j=0` and `S_j=omega_p_j**2`. For a Lorentz pole,
`d_j=omega0_j**2` and `S_j=strength_j`. Let `f` be the film fraction,
`g=1-f`, `b` the host permittivity, and define

```text
A = f*b + g*a
alpha = g/A
kappa = f*b*b/(A*A)
eps_inf_normal = a*b/A.
```

The harmonic normal response is

```text
eps_n(s) = 1/(f/(a+chi(s)) + g/b)
         = eps_inf_normal + kappa*chi(s)/(1+alpha*chi(s)).
```

Introduce physical component polarizations `p_j`, with `P=sum_j p_j`, and
keep the declared damping coordinates:

```text
p_j'' + gamma_j*p_j' + d_j*p_j + alpha*S_j*P = kappa*S_j*E.
```

In the frequency domain, summation gives
`P=chi*(kappa*E-alpha*P)`, hence precisely the required harmonic response.
The tangential law has `alpha=0`, `kappa=f` and
`eps_inf_tangent=f*a+g*b`; its independent memories give the arithmetic
susceptibility `f*chi`.

For active poles define `u_j=sqrt(S_j)`,
`q_j=p_j/sqrt(kappa*S_j)`, `D=diag(d_j)` and
`Gamma=diag(gamma_j)`. The normal equation becomes

```text
q'' + Gamma*q' + J*q = v*E
J = D + alpha*u*u^T
v = sqrt(kappa)*u.
```

`J` is symmetric positive semidefinite, and `Gamma` is diagonal positive
semidefinite. Unequal damping generally prevents one modal rotation from
diagonalizing both, which is why this construction keeps coupled physical
memories. The exact mixture identity needs neither equal damping nor a
residue fit.

## Recurrence and the kernel's time shift

Centre both derivatives in each physical polarization equation. With
`den_j=1+gamma_j*dt/2`, the implemented recurrence is

```text
gain_j = (2-d_j*dt*dt)/den_j
hold_j = (1-gamma_j*dt/2)/den_j
drive_j = kappa*S_j*dt*dt/den_j
coupling_j = alpha*S_j*dt*dt/den_j

p_j^(m+1) = gain_j*p_j^m - hold_j*p_j^(m-1)
          + drive_j*E^m - coupling_j*sum_k p_k^m.
```

Compute the sum from the entire old memory before advancing any pole.
Replacing old values with newly advanced ones while forming the sum would
introduce an ordering-dependent update. Computing one shared sum and then
one update per pole costs linear work in the declared count; no dense
per-cell solve is required at each time step.

The existing `before_E` operation subtracts
`(dx/dt)*sum_j(p_j^(m+1)-p_j^m)` from the curl accumulator. The field
coefficient already includes `dt/(dx*eps_inf)`, so no new `dx` factor or
instantaneous permittivity shift belongs in the oscillator equation.

The kernel advances memories at the end of a step. At the energy reading
associated with `E^n`, memory `P` holds `p^(n+1)` and `previous` holds
`p^n`. This same pair supplies the next electric update. The recurrence
must preserve that shift on NumPy, JAX and Torch.

The explicit oscillator energy metadata must be retained even when
`gamma_j*dt=2` makes `hold_j=0`; that value does not turn a second-order
pole into a Debye relaxation. Zero-strength and empty cells must be masked
before forming reciprocal `kappa*S_j` or the rank-one energy weight.

The lowering evaluates coefficient expressions with exact rational
arithmetic for each distinct concrete fill, then converts finite results
to float64. Active drive and inverse-strength weights must remain positive
and finite; so must the potential when `d_j>0`, coupling when `alpha>0`, and
rank-one weight when required. Unrepresentable active quantities receive
a named refusal. Empty/full and inactive endpoints branch before any
otherwise-unused division. The targeted checks below exercise this contract;
the full and separate slow tiers pass, and all four frozen-reference
gates are bit-exact. The measured propagation limits are below.

## Discrete energy: one rank-one term

At the observer's `E^n` level, put
`delta_p_j=p_j^(n+1)-p_j^n` and `P^n=sum_j p_j^n`.
In the repository's energy normalization, the material correction is

```text
sum_active_j [
    |delta_p_j/dt|**2/(kappa*S_j)
  + d_j*Re(p_j^(n+1)*conj(p_j^n))/(kappa*S_j)
  - Re(E^n*conj(delta_p_j))
]
+ (alpha/kappa)*Re(P^(n+1)*conj(P^n)).
```

There is no factor of one half: the existing field and oscillator energies
already use this doubled normalization. The per-pole potential uses the
unshifted declared `d_j`. If it used `d_j+alpha*S_j` and also added the
rank-one term, the coupling's diagonal would be counted twice, already
wrong in a one-active-pole reduction.

The final rank-one term is added once per pole set, after summing its
polarizations at each cell. Its spatial coefficient must be applied before
reducing cells, components or clipped observation regions. It is not a
product of separately reduced polarization sums. The existing per-pole
field-coupling terms sum to `-Re(E^n*conj(P^(n+1)-P^n))`.

With `J_j^(n+1/2)=delta_p_j/dt`, the damping contribution to the energy
difference is

```text
-(dt/2)*sum_active_j gamma_j/(kappa*S_j)
         * |J_j^(n+1/2)+J_j^(n-1/2)|**2.
```

It is nonpositive for unequal as well as equal nonnegative damping rates.
This is the material dissipation identity; a complete energy balance must
also include source work and boundary flux.
The weighted stiffness is
`diag(d_j/(kappa*S_j))+(alpha/kappa)*1*1^T`, which is symmetric; the
weighted damping is diagonal positive semidefinite. No commutation between
their normalized matrices is needed for this energy identity.

Drude poles have zero restoring frequency. Static polarization combinations
orthogonal to the coupled sum may therefore carry zero restoring energy.
The energy is semidefinite on those memory directions, not strictly
positive definite over every possible polarization state. This is distinct
from growth of a physical field or current. The bounded scope starts
memories at rest and still requires explicit finite-energy and propagation
validation.

## The existing summed CFL bound

Normalize electric fields by the local high-frequency permittivity and
let `C_eps` denote the corresponding discrete curl. For the lossless
coupled system, the relevant block operator can be written

```text
B = [[C_eps, -v^T/sqrt(eps_inf_eff)],
     [0,      sqrt(J)                ]].
```

Splitting `B*B^*` into its curl block and cell-local factors gives

```text
||B||**2 <= ||C_eps||**2
         + max_cells lambda_max(J + v*v^T/eps_inf_eff).
```

For the normal component,

```text
alpha + kappa/eps_inf_normal = 1/a
J + v*v^T/eps_inf_normal = D + u*u^T/a
lambda_max(D + u*u^T/a) <= max_j d_j + sum_j S_j/a.
```

For a tangent component, `kappa/eps_inf_tangent=f/(f*a+g*b)<=1/a`, so
the same bound applies. The scoped policy uses `max_active d_j + sum_active S_j/a`,
evaluated as an exact rational sum and rounded upward to float64. An
all-inactive declaration contributes exactly zero; unused extreme
frequencies and damping remain in the identity without entering this rate.
The usual strict `dt*||B||<2` condition
controls the undamped leapfrog energy, and centred nonnegative damping
adds the dissipative term above without a new damping-rate CFL penalty.
This argument does not extend the existing absorber/PML stability contract.

## Conservative bulk spectral floor

A Drude/Lorentz sum can have resonances within a source band; the monotone
upper-endpoint theorem for passive Debye sums does not apply. The new
spectral guard bounds the entire declared sum over each positive
source interval before allowing a mesh. Its purpose is bulk phase/skin
resolution, not a theorem of interface or scattering accuracy.

A conservative whole-band bound uses `x=omega**2` and

```text
R_j(x) = x/sqrt((d_j-x)**2 + gamma_j**2*x)
K_bound**2 = a*x_high + sum_active_j S_j*max_band R_j(x)
|k(omega)|**2 <= K_bound**2.
```

For each pole, test both endpoints and the possible interior critical point

```text
x_critical = 2*d_j*d_j/(2*d_j-gamma_j*gamma_j),
```

when its denominator is positive and the point lies in the band. This
follows by differentiating `R_j**2`; its derivative has the sign of
`x*((gamma_j**2-2*d_j)*x+2*d_j**2)` away from a pole. A positive strength at
an undamped in-band pole has no finite bound and is explicitly refused.
A zero-strength pole is removed before testing its denominator.

The implementation computes critical points and squared ratios
with exact rational arithmetic and rounds square-root upper bounds
outward before combining them. It does not approximate an interior
resonance by sampling a few frequencies. The triangle inequality can make
the bound conservative because it does not rely on cancellation among
complex susceptibilities. Targeted peak/sample checks and the interior-peak
mutation below exercise this guard; they do not establish scattering order.

Every source band and declared monitor, `Resolution` and explicit PML
wavelength must remain concrete and positive. The prior eight-cell phase
and one-cell skin-depth refusal floors, with twelve/two warnings, are the
inherited policy. This spectral floor does not control all effective
interface resonances or oblique longitudinal scales.

## Disclosed six-row calibration

The first calibration attempt completes six sequential workers, all with
exit zero and unchanged production identity, from 2026-10-06 17:13:45 to
17:15:05 UTC. These responses were observed before a convergence
registration. They are calibration data, not unseen corroboration or a
spatial-order result. The two synthetic material families are:

| Family | High-frequency permittivity | Declared poles |
| --- | --- | --- |
| Mixed metal | 1 | Drude `omega_p=9`, `gamma=1`; Lorentz `omega_0=8`, `gamma=2`, `strength=10` |
| Multiple Lorentz | 1 | `(omega_0,gamma,strength)=(5,0.8,10), (8,1.5,20), (11,2,15)` |

Each family runs on NumPy, JAX and Torch. The configuration is a film of
thickness 0.2 in vacuum, p polarization at 35-degree centre incidence,
exact transverse `Axis1D`, face fraction 0.3, 40 cells per central
wavelength, 120 periods and 24 PML cells. The source band has wavelength
1, `df=0.1` and cutoff `1e-6`; the sampled wavelengths are 0.9/1.0/1.1.
At Courant 0.5 the time step is 0.007216878364870323, with 16,628 steps
and fixed transverse wavenumber 3.6038870374253196. The sampled angles
are approximately 31.079, 35 and 39.119 degrees.

The analytic preflight uses no field solve. Its independent Airy controls
pass, its whole-band bulk bounds are 14.5866231 and 17.3572660, both
admitted at 40 cells, and the mixed-metal film has negative real
permittivity at all three sampled wavelengths. The continuum comparison
evaluates the declared susceptibility sum and the complex film response.
Maximum absolute complex-amplitude errors over all three wavelengths and
all three backends are:

| Family | Reflection error | Transmission error | Largest reflected decay | Largest transmitted decay |
| --- | --- | --- | --- | --- |
| Mixed metal | 0.002272591954 | 0.0009530150183 | 2.401781e-9 | 6.597181e-13 |
| Multiple Lorentz | 0.0006533071788 | 0.002072997834 | 5.796089e-8 | 2.894161e-10 |

All six rows report stable fields and 1,041 finite energy samples. The
largest scaled JAX/Torch difference from its family's NumPy r/t values is
2.8288909673081856e-15, below the prior budget of
128 double-precision epsilons, 2.842170943040401e-14, using
`max(1,abs(NumPy value))` as the scale. This comparison concerns the two
known configurations; it is not an accuracy claim for other parameters.

The inherited candidate ceilings, 0.005 for reflection and 0.010 for
transmission, were accepted unchanged at 17:15:42 UTC, **after** these
measurements. Their margins against the observed maxima are
0.002727408046 and 0.007927002166. Acceptance applies only to the two
40-cell p-polarized, fraction-0.3 regression configurations. The registered
campaign's convergence criterion is separate, and its registration discloses
these known responses before new data.

All rows retain `SubpixelAccuracyWarning`. The three mixed-metal rows
also retain `AbsorberProximityWarning`: the film is 0.775 from the inner
z-minus absorber, less than the reported half material wavelength 1.13248.
The warning was not removed; later matched duration/absorber controls are
still needed. The multiple-Lorentz rows carry only the subpixel warning.
Stable finite energy and small decay here do not establish a convergence
order or a general absorber guarantee.

The saved environment is Windows AMD64, Python 3.12.14, NumPy 2.5.1,
JAX/JAXlib 0.11.0 and Torch 2.13.0, with OMP, OpenBLAS and MKL thread
counts set to one. Worker wall times include process/backend setup:

| Family | NumPy seconds | JAX seconds | Torch seconds |
| --- | --- | --- | --- |
| Mixed metal | 7.790750 | 4.948190 | 26.647968 |
| Multiple Lorentz | 8.247971 | 5.357459 | 27.282940 |

These timings are not a controlled backend performance comparison.
The record is `fdtd_oscillator_sum_film_calibration_attempt1.json`; its directory retains
six result/raw-log pairs and frozen probe/controller sources. The record and its
directory are archived outside the repository. All 14 linked artifact hashes match.
The explicit acceptance is
`studies/fdtd/oscillator_sum_film/records/fdtd_oscillator_sum_film_calibration_acceptance.json`.
The recorded FD and package digests are respectively
`86e2f7078c49a546d63f20b9acfc822d7c3c519363577599b56659572dcf193f` and
`419700870cb89df8555115f0b6754744eacd846d6c782721a11736123589b297`,
using sorted relative POSIX paths, NUL separators and LF-normalized source
bytes. Every worker records the same before/after source identity.

## Targeted checks

The first targeted attempt collects and passes all 684 cases with exit
zero, no skips, expected failures or unexpected passes. It includes 148
new parameterized cases across nine test functions, all five earlier film
test files, and the existing stencil refusal-routing regression. Collection
takes 3.05 pytest seconds (3.916261 wrapper seconds); the run takes 95.00
pytest seconds (95.508062 wrapper seconds). The raw log retains 732
warnings. Production and probe identities match before and after the run.

The new functions cover the actual coupled recurrence against the declared
discrete harmonic mixture; endpoints, inactive strength and pole ordering;
spatial energy and conservation/dissipation under actual advances;
three-backend oblique Airy controls; unsupported geometry/material cases;
prepared-plan identity and missing/extra memory; traced-input bypasses;
all active strengths and the largest resonance in the CFL rate; and
whole-band peaks plus requested spectral samples. The Airy cases repeat
the disclosed calibration configurations and are regression checks, not
unseen physical corroboration.

The manifest, collection/run logs, runner and exact test snapshots are
retained in the study's controls directory, `fdtd_oscillator_sum_film_controls`, with
the result in `targeted_attempt1.json` there; the directory is archived outside the
repository.
The eight snapshots and both logs match their recorded hashes.

## Actual mutation witnesses and restoration

The first mutation attempt runs from 17:18:00 to 17:21:01 UTC. All ten
intentional production mutations produce the intended assertion failure,
covering all nine new test functions; the discrete-response function has
two separate witnesses. The harness accepts every witness and restores
all production bytes exactly.

| Mutated behavior | Assertion witness |
| --- | --- |
| Omit the rank-one runtime coupling | Actual discrete harmonic response |
| Use the first damping rate for every pole | Unequal-damping response |
| Shift the zero-strength lossless permittivity | Endpoint reduction |
| Omit the rank-one energy cross term | Spatial energy and advance identity |
| Quarter the final memory's drive | Oblique Airy regression |
| Remove the initial-field refusal | Unsupported-input guard |
| Remove the prepared signature check | Changed-material identity |
| Detach traced final damping | Traced-input refusal |
| Omit the final strength from the CFL rate | Summed-rate policy |
| Omit an interior spectral peak | Whole-band resolution guard |

The exactly restored new-file baseline passes all 148 cases with exit
zero, no skips or expected/unexpected failures, in 79.85 pytest seconds
(82.014707 wrapper seconds); its raw log retains 170 warnings. The before
and after source identities agree. The manifest is
`fdtd_oscillator_sum_film_mutations.json`; original/mutant snapshots, assertion logs and
the restored baseline remain in the attempt directory
`fdtd_oscillator_sum_film_mutations_attempt1`. Both are archived outside the
repository. All 32 linked
snapshot, collection-log, assertion-log and restored-log hashes match.
These witnesses test the specified faults; they are not an exhaustive
proof against every possible implementation defect.

## Registered convergence measurement

After the disclosed calibration, targeted tests and real production
mutation witnesses, the protocol was registered at 17:21:31 UTC on
2026-10-06. Its record, archived outside the repository, is
`fdtd_oscillator_sum_film_registration.json`, with fingerprint
`81cb9013510869cfea71b56f398039a89817f015fd19368c32c153e4b0aaaf10`.
All 66 linked known-exposure records and four frozen controller, probe,
criteria and test snapshots match their recorded hashes. The registration
fixes the two families above, face fractions 0.3/0.1, both polarizations
and five sequential NumPy rows per group:

| Cells | Periods | PML cells |
| --- | --- | --- |
| 40 | 120 | 24 |
| 80 | 120 | 24 |
| 80 | 240 | 24 |
| 80 | 240 | 48 |
| 160 | 120 | 24 |

This gives 40 scheduled workers, 48 complex r/t cells and
96 adjacent direct-error intervals. Only an otherwise valid unsettled
120-period base row may receive one 240-period replacement, reusing the
fixed 80/240/PML24 control when already available. An invalid capped row
or fixed control stops the campaign. The decay limit is `1e-5`. No duration
beyond 240 periods, extra rung, geometry change or adaptation to an order
classification is admitted.

For each complex cell the empirical proxy is
`u=abs(v80,long24-v80,base24)+abs(v80,long48-v80,long24)`, retaining the
120-period base response in the proxy even if the selected 80-cell result
uses 240 periods. This middle-grid proxy is not a rigorous fine-grid
error bound. The self-order is the base-two logarithm of the ratio of
adjacent grid changes, with a sensitivity interval formed by adding and
subtracting `2*u` from each change. Both changes must exceed `1e-10`, and
`2*u` must be at most one tenth of the smaller change. Each adjacent
direct-error order similarly adds/subtracts `u` from each error; both
errors must exceed `1e-10`, and `2*u` must be at most one tenth of the
smaller error. A resolved interval wholly inside `[1.8,2.3]` is second;
an upper endpoint at most 1.75 is low, and other resolved intervals are
mixed. Failed noise gates, vanishing signals or missing/invalid rows are
unresolved. Every cell is reported independently.

The reference evaluates the declared rational susceptibility and continuum
Airy admittances independently of production `eps_at` and memory helpers.
Measured `1-R-T` and continuum absorptance are retained; physical loss is
not treated as an energy-closure error. Both coarse p-polarized,
fraction-0.3 NumPy rows repeat disclosed calibration responses. They are
known-response checks, not independent corroboration. All warnings remain
recorded, and the fixed geometry is not adapted to their presence.

## Registered results and unresolved cells

All 40 sequential workers complete with exit zero, without a replacement
or registered stop, from 17:21:54 to 17:37:06 UTC on 2026-10-06. Their
combined worker wall time is 908.05 seconds. Every group retains its
120-period base rows at 40/80/160 cells. All raw rows record stable fields,
finite energy and unchanged source identity; the maximum recorded decay
is 6.109033e-8, below the fixed `1e-5` line. All 80 worker result/log hashes
match. Completion of the procedure does not mean every order is second.

| Family | Self-orders: second / mixed / unresolved | All direct pairs: second / mixed / unresolved | Final direct pairs: second / unresolved |
| --- | --- | --- | --- |
| Mixed metal | 23 / 1 / 0 | 48 / 0 / 0 | 24 / 0 |
| Multiple Lorentz | 22 / 1 / 1 | 42 / 1 / 5 | 20 / 4 |
| Total | 45 / 2 / 1 | 90 / 1 / 5 | 44 / 4 |

No resolved interval is classified low. The resolved self intervals span
1.751656 to 2.102740; resolved direct intervals span 1.788854 to 2.090531,
and resolved final direct intervals span 1.894705 to 2.090531. These ranges
exclude unresolved cells and cannot replace the cellwise counts.

Two reflected s-polarized cells at fraction 0.3 and wavelength 1.1 have
resolved self intervals below the second-order band. Their noise shares
are small, so they are mixed, not noise-unresolved. Both final direct
intervals enter the second-order band, but this does not establish an
asymptotic order beyond the measured ladder:

| Family | Self interval, 40/80/160 | Direct interval, 40 to 80 | Direct interval, 80 to 160 | `2*u/min(grid changes)` |
| --- | --- | --- | --- | --- |
| Mixed metal | [1.762812, 1.763027], mixed | [1.803546, 1.803624], second | [1.910102, 1.910392], second | 5.762717e-5 |
| Multiple Lorentz | [1.751656, 1.752522], mixed | [1.788854, 1.789169], mixed | [1.898797, 1.899953], second | 2.314187e-4 |

The four unresolved final direct pairs are all multiple-Lorentz reflected
p-polarized cells. Each has nonzero errors, but the empirical time/PML
proxy is too large relative to its finer error: the rule requires
`2*u/min(error80,error160)<=0.1`. Their point orders are descriptive only;
no accepted sensitivity interval is reported for these pairs.

| Face fraction | Wavelength | Error at 160 | Proxy `u` | Final proxy share | Final point order |
| --- | --- | --- | --- | --- | --- |
| 0.3 | 0.9 | 4.499855e-5 | 3.256662e-6 | 0.144745 | 1.924689 |
| 0.3 | 1.0 | 8.351744e-6 | 8.133213e-7 | 0.194767 | 1.839859 |
| 0.3 | 1.1 | 1.126168e-5 | 2.675716e-6 | 0.475190 | 2.180072 |
| 0.1 | 1.1 | 4.671513e-5 | 2.670544e-6 | 0.114333 | 2.021096 |

For fraction 0.3, wavelength 1.1, the self-order is also unresolved:
its proxy share is 0.134097, above 0.1 (point order 2.227096). The same
cell's 40-to-80 direct pair is unresolved with share 0.104858 (point
2.220137). These account for the one unresolved self-order and the fifth
unresolved direct pair. The other three cells in the table have second
self-orders and second coarse direct pairs. No self/direct signal is zero.

The largest empirical proxy across all cells is 4.596175e-6; the separate
maximum duration and PML changes are 4.501747e-6 and 1.203836e-7. The
middle-grid proxy measures those two controls. It is not a rigorous
fine-grid uncertainty bound, and a small final point error does not
remove a failed noise gate. The completed experiment stops at its
registered cap; these unresolved results remain explicit.

Maximum absolute complex errors over placements, polarizations and
wavelengths decrease as follows:

| Family and coefficient | 40 cells | 80 cells | 160 cells |
| --- | --- | --- | --- |
| Mixed-metal reflection | 0.004404282404 | 0.001118648834 | 0.0002814617724 |
| Mixed-metal transmission | 0.001848931292 | 0.0004629315836 | 0.0001157764782 |
| Multiple-Lorentz reflection | 0.002917135702 | 0.0007319839793 | 0.0001835327259 |
| Multiple-Lorentz transmission | 0.002545033567 | 0.0006347910525 | 0.0001586058334 |

These maxima describe accuracy but do not decide the registered cellwise
order classifications. The calibration's 0.005/0.010 ceilings remain
specific to its two disclosed regression rows. Both matching NumPy
40-cell p/fraction-0.3 physical payloads repeat their known calibration
payloads exactly, which is provenance evidence rather than a new
independent accuracy check. The retained absorber-proximity warnings and
physical absorptance also remain part of the record.

The complete cell report is
`studies/fdtd/oscillator_sum_film/records/fdtd_oscillator_sum_film_report.json`, with the fixed
sequence in `fdtd_oscillator_sum_film_campaign.jsonl` and raw pairs in the job directory
`fdtd_oscillator_sum_film_campaign_jobs`; the sequence and the job directory are
archived outside the repository. The separate saved-data audit is
`fdtd_oscillator_sum_film_independent_audit_attempt1.json`, also archived outside the
repository.
It checks all 259 recorded input hashes and reconstructs every cell
without importing the solver, probe, controller or criteria. Its separate
cos/sin transfer-matrix continuum differs by at most 3.510833e-16;
reconstructed cell arithmetic differs by at most 6.661338e-16. The raw
log and auditor/runner source archives are retained, and
`fdtd_oscillator_sum_film_independent_audit_attempt1_execution.json`
records the observed child exit zero and unchanged source identity; these too are
archived outside the repository.
Passing this audit verifies the record and its arithmetic, including the
mixed and unresolved outcomes; it does not convert those outcomes to passes.

The bounded implementation is therefore supported by exact-response,
energy, guard and backend tests, and by this completed experiment with
its stated exceptions. Two- and three-pole propagation examples cannot
establish measured accuracy for all finite counts. Higher-count targeted
constitutive/energy checks, including eight poles, remain distinct from
propagation evidence. General C8 smoothing is incomplete and C7 remains
unpassed.

## Regression gates and evidence preservation

The completed numerical validation uses the same fixed production source
and exits zero. The full and separate slow tiers report:

| Gate | Collected | Passed | Skipped | Retained warnings | Pytest seconds | Wrapper seconds |
| --- | --- | --- | --- | --- | --- | --- |
| Full | 3,442 | 3,434 | 8 | 1,198 | 2,578.99 | 2,579.705001 |
| Separate slow | 81 | 81 | 0 | 50 | 1,153.09 | 1,153.610707 |

Both have no expected failures or unexpected passes. All four read-only
freezers exit zero and reproduce 143/288/140/570 arrays bit for bit,
1,141 in total, without writing the committed golden files. The stage-2
and stage-3 logs retain notices about historical case/freezer provenance;
those notices are distinct from the exact array comparisons.

The observed outcomes are recorded in `fdtd_oscillator_sum_film_validation.json`; raw
collection, run and freezer logs are retained in the controls directory. Both are archived
outside the repository.
The wrapper records FD source digest
`15002f0d0c251979103dace7866f02f29a3e22b9aea99c81926497c1aa17a6b8`.
That inherited helper hashes sorted relative paths and LF-normalized
source bytes without delimiters; it differs by definition from the
NUL-delimited campaign identity above. Completed regression gates do not
change the campaign's mixed or noise-unresolved physical classifications.

Final documentation and saved-evidence acceptance is a separate release
requirement. Its prescribed outputs are named
`fdtd_oscillator_sum_film_final_docs.json`,
`fdtd_oscillator_sum_film_final_evidence_audit.json` and
`fdtd_oscillator_sum_film_final_evidence_audit_execution.json`; all three are archived
outside the repository.
Acceptance requires actual zero exits for documentation collection/tests
and Sphinx `-W`, unchanged document-source hashes during those checks,
and a complete saved-data audit with an independently recorded zero
process exit. The preserved auditor and documentation-runner sources are
`final_evidence_audit.py.txt.gz` and `final_documentation_gates.py.txt.gz` in the
controls directory, archived outside the repository.
This article specifies the required evidence without predeclaring their
result. The execution records, archived with those sources, report it for
2026-10-06: documentation collection, the documentation contract (72
collected, 72 passed) and the warning-as-error Sphinx build each exited zero.
The saved-data audit's first attempt exited one because a controller audit
log was missing; the second, with the same audit source and no change to
numerical source or records, completed with exit zero.
