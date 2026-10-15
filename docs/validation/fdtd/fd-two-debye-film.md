# C8: one film with two Debye poles

Written: 2026-10-06

Targets: `src/wavebit/fdtd/_prepare/two_debye_film.py`,
`src/wavebit/fdtd/_prepare/two_debye_resolution.py`, `src/wavebit/materials/time_domain.py`,
`src/wavebit/fdtd/_kernel/poles.py`, `src/wavebit/fdtd/_kernel/observe.py`,
`src/wavebit/fdtd/_prepare/checks.py`.

## TL;DR

**The registered 40/80/160-cell study classifies all 24 self-orders and all
48 direct pairs as second within its fixed sensitivity intervals.** All 20
workers complete without a replacement or stop. This conclusion applies to
the disclosed material and two placements. All 374 targeted cases and nine
production mutation witnesses pass, followed by 127 passing restored cases.
The full tier passes 3,122 cases with eight skips, the slow tier passes all
81 cases, and all 1,141 frozen-reference arrays reproduce bit for bit.

The bounded implementation and required numerical gates are complete on the
feature branch. Its harmonic law
couples the declared poles; reducing each pole separately would give a
different material. Up to two passive normal relaxation modes use two reserved
memory slots, the existing split recurrence and local energy weights. The
material-rate policy sums both modes at each cell before taking the maximum
over cells and the original bulk rate.

The unchanged calibration-row ceilings of 0.005 for reflection and 0.010
for transmission apply to that one known configuration. Other campaign
rows exceed those coarse-grid errors; their registered criterion is
convergence under refinement, not those calibration thresholds.
The contract is exactly two passive Debye poles in one concrete
flat `Box` in a plain dielectric host. It does not admit an arbitrary pole
count or general dispersive smoothing.

The earlier [single-Debye film](fd-debye-film.md),
[Drude/Lorentz film](fd-single-pole-film.md) and
[conductive film](fd-conductive-film.md) retain their recorded material
scopes and outcomes. General C8 smoothing remains incomplete, and the
[C7 metasurface verdict](fd-metasurface-validation.md) remains unpassed.

## Bounded contract

The public declaration is one
`fd.Structure(fd.Box(...), fd.Dispersive(eps_inf=..., poles=(pole1, pole2)))`,
where both poles are exact public `Debye` objects and discretization is the
default voxel path. The following restrictions apply together:

| Item | Bounded case |
| --- | --- |
| Geometry | Exactly one concrete public `Box`, `angle=0`, `range=None`, without a partner structure or junction. |
| Extent | Exactly two complete grid axes are uniform through the film, with at least one `dx` of overhang beyond each transverse domain face. |
| Thickness | The remaining normal extent is at least one `dx`, keeping the two faces out of the interior of the same electric dual cell. |
| Clearance | Normal support is strictly inside the domain, and all support stays clear of every absorbing layer. A transverse absorber is outside this scope. |
| Film | Exact `Dispersive` with exactly two exact `Debye` poles, finite positive isotropic high-frequency permittivity and `mu=1`. |
| Poles | Concrete finite increments `delta_eps >= 0` and relaxation times `tau > 0`. Zero increments and equal relaxation times are admitted. |
| Host | Exact plain lossless `Material`, concrete finite positive isotropic permittivity and `mu=1`. |
| State and reuse | No initial field; prepared geometry, material, grid and both reserved memory slots must match. A different plan cannot bypass admission. |

The contract is invariant under permutation of x, y and z. Occupancy is the
analytic interval overlap of each electric component's own Yee dual cell.
Exact geometric empty/full endpoints are retained without a tolerance band
that prunes small nonzero mixed fractions. Existing source, monitor,
feature-size and absorber restrictions still apply.

This scope excludes other pole counts or pole types, subclasses, a lossy or
dispersive host, anisotropy, magnetism, curved or rotated bodies, junctions,
and traced geometry, materials or source bands. No gradient correctness is
claimed, including for source amplitudes. Existing snapped-material and
stencil paths retain their own contracts. Later scopes admit three or more
Debye poles ([record](fd-debye-sum-film.md)) and mixed Debye/oscillator poles
([record](fd-hybrid-film.md)) in one film (2026-10-07), and separated Debye
films ([record](fd-debye-stack.md), 2026-10-09).

## Coupled harmonic reduction

Use the repository's `exp(-i*omega*t)` convention and put `s=-i*omega`.
Let `f` be the film fraction, `g=1-f`, `a` the film high-frequency
permittivity, `b` the host permittivity, and `delta_i,tau_i` the two declared
Debye parameters. For active poles define

```text
r_i = 1/tau_i
u_i = sqrt(delta_i*r_i)
D = diag(r_1, r_2)
chi(s) = sum_i delta_i/(1 + tau_i*s)
       = u^T * (s*I + D)^(-1) * u
A = f*b + g*a.
```

The arithmetic tangential response is

```text
eps_t(s) = f*a + g*b + f*chi(s).
```

The normal harmonic response instead reduces as a whole:

```text
eps_n(s) = 1/(f/(a + chi(s)) + g/b)
         = a*b/A + (f*b*b/(A*A))*chi(s)/(1 + (g/A)*chi(s))

J = D + (g/A)*u*u^T
eps_n(s) = a*b/A + (f*b*b/(A*A))*u^T*(s*I + J)^(-1)*u.
```

The last identity follows from the rank-one inverse identity. Since `D` is
positive definite and `(g/A)*u*u^T` is positive semidefinite, `J` has positive
eigenvalues. Let `J*q_j=lambda_j*q_j` with orthonormal eigenvectors, and put
`w_j=q_j^T*u`. Then

```text
eps_inf_n = a*b/A
tau_n_j   = 1/lambda_j
delta_n_j = (f*b*b/(A*A))*w_j*w_j/lambda_j

eps_n(s) = eps_inf_n + sum_j delta_n_j/(1 + tau_n_j*s).
```

Every modal increment is nonnegative. Tangential coefficients are simply
`eps_inf_t=f*a+g*b`, `delta_t_i=f*delta_i` and `tau_t_i=tau_i`.
No fitted pole, residual conductivity or off-diagonal constitutive operator
is needed for this axis-aligned contract. In particular, normal residues and
relaxation times depend on both original poles through `J`.

## Degenerate cases and two reserved slots

The physical number of active relaxation modes can be smaller than two.
The memory allocation nevertheless reserves exactly two slots, so the
prepared-plan identity does not depend on whether a particular increment
vanishes.

| Limit | Required physical response |
| --- | --- |
| Empty cell, `f=0` | Host permittivity with no active memory. |
| Full cell, `f=1` | The declared bulk two-Debye response. |
| Both increments zero | Lossless harmonic/arithmetic mixtures; no active memory or material rate. |
| One increment zero | The corresponding single-active-pole mixture. |
| Equal relaxation times | One physical pole with the sum of the two increments before normal transformation. |

For equal rates `r_1=r_2=r`, `J=r*I+(g/A)*u*u^T`. Only the mode parallel
to `u` is driven; the orthogonal mode has zero residue. Thus its normal law
is exactly the single-Debye formula with `delta=delta_1+delta_2`, not a
pair of distinct resonances created by rounding.

Active normal modes are ordered by increasing effective relaxation time.
Exact zero-strength and equal-time collapses use slot zero for the remaining
active response while retaining the second slot. Inactive slots carry zero
drive and energy weights. Small nonzero residues are not discarded by an
epsilon threshold. Only exact equality of relaxation times triggers collapse;
distinct adjacent float64 times remain distinct. The implementation computes
the small final modal increment with binary-scaled products, so an intermediate
normalized weight need not erase a representable final pole. Non-finite or
underflowed active coefficients and inverse energy weights are refused
explicitly; this does not promise admission of every mathematically finite
parameter combination.

Zero increments are handled before forming inactive reciprocal times or
time squares. A finite but extreme unused `tau` therefore cannot create
spurious active memory. The targeted tests below exercise these
representation requirements separately from physical convergence.

## Split recurrence and local energy

Each active effective mode with increment `delta` and time `T` uses the
existing trapezoidal Debye split. Define

```text
m = max(T, dt/2)
left = T/m
right = (dt/2)/m
u = left/(left+right)
v = right/(left+right)

gain = u-v
c = delta*v
drive = 2*delta*u*v
hold = 0
Q = P-c*E
Q^(n+1) = gain*Q^n + drive*E^n.
```

Here the scalar split ratios `u,v` are unrelated to the two-component
coupling vector in the harmonic derivation. The field-update permittivity
is `eps_inf_eff+sum_j c_j`. All mode increments enter Ampere's law through
the existing polarization recurrence. The full trapezoidal term involving
`E^(n+1)` is preserved by the instantaneous shifts; it is not dropped from
the physical Debye equation.

At the observer time level, the local energy correction is the sum over
active slots

```text
sum_j (|Q_j^n + c_j*E^n|**2/delta_j - c_j*|E^n|**2).
```

The existing spatial Debye branch applies each inverse increment and
instantaneous shift before reducing cells, components or clipped regions.
The subtraction cancels the shift already included in the electric-field
weight. Inactive weights are zero. This is the discrete material-energy
identity, not a propagation or long-time stability measurement.

The coefficient-level frequency oracle evaluates the same exact mixture
at `omega_hat=2*tan(omega*dt/2)/dt`. A physical film study must instead use
the continuum material at the actual frequency and include the solver's
spatial and temporal errors.

## Summed material-rate policy

The repository's conservative Debye policy sums the active poles of a cell:

```text
R_bulk = (delta_1*r_1*r_1 + delta_2*r_2*r_2)/a
R_cell = sum_j delta_eff_j/(eps_inf_eff*tau_eff_j*tau_eff_j)
R_film = max(R_bulk, max_actual_cells_and_components(R_cell))
dt_bound = 1/sqrt(1/dt_nondispersive**2 + R_film/4).
```

Taking only the largest individual pole would omit a simultaneously active
contribution. Conversely, adding the bulk and transformed rates would count
the same film twice. The maximum uses actual Yee component occupancies.
For a normal cell, the exact identity is

```text
W = sum_i delta_i*r_i
R_n = (f*b/(a*A)) * (sum_i delta_i*r_i*r_i + (g/A)*W*W).
```

It follows from `sum_j w_j*w_j*lambda_j=u^T*J*u` and avoids interpreting
modal contributions as unrelated materials. The normal rate can exceed the
bulk sum. The tangential sum is
`f*sum_i(delta_i*r_i*r_i)/(f*a+g*b)`, no greater than the bulk value.

Only this matching film bypasses the ordinary declared-pole rate path in
favour of the scoped summed policy. Zero increments contribute zero before
any inactive reciprocal or square is evaluated. The policy is conservative;
passive trapezoidal relaxation alone does not intrinsically require this
extra time-step restriction. It does not extend the absorber stability
contract.

## Monotone bulk spectral guard

For the positive-frequency bulk response and `x=omega*omega`, direct
expansion gives

```text
|eps(omega)|**2 = a*a + C_1/(1 + tau_1*tau_1*x)
                     + C_2/(1 + tau_2*tau_2*x)

C_i = delta_i*delta_i + 2*a*delta_i
    + 2*delta_i*delta_other*tau_i/(tau_i+tau_other).
```

Both coefficients are nonnegative, including zero increments. Each
nonconstant term in `|k|**4=x*x*|eps|**2` has nonnegative derivative:

```text
d[x*x*C_i/(1 + tau_i*tau_i*x)]/dx
  = C_i*x*(2 + tau_i*tau_i*x)/(1 + tau_i*tau_i*x)**2.
```

The `a*a*x*x` term makes the total strictly increasing for `x>0`. The
largest bulk modulus on every positive source band is therefore at its
upper endpoint. The guard evaluates that endpoint's actual two-pole
response and requires `dx*|k_max|<=2*pi/8` for both film and host.

For a source with `f0=1/band.wavelength`, width `df` and
`m=band.sigmas`, the angular-frequency interval is
`[2*pi*(f0-m*df), 2*pi*(f0+m*df)]`, with a positive lower endpoint.
The scoped contract also checks every declared monitor wavelength,
the `Resolution` wavelength and explicit PML wavelengths. The sampled
minima are eight cells per phase wavelength and one per skin depth;
existing warnings below twelve and two remain. Bands and wavelengths must
be concrete, finite and positive.

This is a bulk scale-resolution floor. It is not a theorem of scattering
accuracy, a certificate for every effective interface or oblique
longitudinal scale, or a substitute for duration and absorber controls.
With no usable spectral metadata, the planner can only warn that the bulk
mesh could not be assessed.

## Disclosed development algebra check

A separate 10,000-case algebra exploration records maximum relative
response and material-moment differences of 1.046299e-15 and 1.051992e-15.
It ran no field solve and imported no production package. The identical
script and observed output were preserved afterward in
`fdtd_two_debye_film_algebra_attempt1.py.txt` and
`fdtd_two_debye_film_algebra_attempt1.json`, both archived outside the repository; the source hash
was captured after execution without an intervening source edit. This is a
development calculation, not a pytest gate, registered physical result or
measurement of production propagation accuracy.

## Disclosed three-backend calibration

The first calibration completed three sequential runs on 2026-10-06,
13:31:29 to 13:32:12 UTC, all exit 0, without adapting a row. Each uses
`eps_inf=2`, increments `(2,4)`, relaxation times `(0.05,0.2)`, film
thickness 0.2 and a vacuum host. The grid has 40 cells per centre vacuum
wavelength, face fraction 0.3, p polarization and 35-degree incidence at
centre wavelength 1. An exact `Axis1D` derivative keeps the same transverse
wavevector at wavelengths 0.9, 1.0 and 1.1. Each run lasts 120 periods with
24-cell PMLs, Courant factor 0.5 and source `df=0.1`, cutoff `1e-6`.

The reference in `studies/fdtd/two_debye_film/fdtd_two_debye_film_probe.py` evaluates the explicit
two-Debye rational permittivity and complex Airy film response independently
of the production material evaluator. Its analytic preflight controls pass
without field solves. The NumPy measurements are:

| Wavelength | Absolute complex `r` error | Absolute complex `t` error |
| --- | --- | --- |
| 0.9 | 0.004011955551 | 0.009808685985 |
| 1.0 | 0.004119745238 | 0.008412073742 |
| 1.1 | 0.003727428224 | 0.007064898435 |

The maxima across all three backends are 0.004119745237675937 for `r`
and 0.009808685984934997 for `t`. These are absolute complex-amplitude
errors, not relative errors or power errors. The inherited candidate
ceilings of 0.005 for `r` and 0.010 for `t` were accepted unchanged after
reviewing these disclosed observations. The transmission margin is only
0.00019131401506500278. The thresholds were not widened; their acceptance
is not presented as a test of previously unknown data. The decision and
its calibration checksum are recorded in
`studies/fdtd/two_debye_film/records/fdtd_two_debye_film_calibration_acceptance.json`.

At each complex sample, JAX and Torch are compared with NumPy using
`abs(value-reference)/max(1,abs(reference))`. The largest scaled difference
is 1.1826807284728509e-15, within the inherited
`128*float64_epsilon` budget of 2.842170943040401e-14. This is agreement on
one configuration, not evidence for every admitted material, orientation
or derivative.

All three rows report stable fields and finite energy, with 1,041 energy
samples each. The largest reflection and transmission decay ratios are
9.959172e-9 and 1.753847e-11. Those measurements describe settling of the
recorded responses; they neither bound absorber reflection nor determine
a convergence order.

Both warning categories remain recorded on every backend. The
`CoarseMeshWarning` reports approximately 12 cells per phase wavelength
and 7.55 per skin depth at wavelength 0.655457, with the displayed phase
count rounded. The `SubpixelAccuracyWarning` retains the bulk-only scale
qualification and calls for refinement, duration and absorber controls.
Neither warning was suppressed or converted into a numerical pass.

The recorded runtime is Windows with Python 3.12.14, NumPy 2.5.1,
JAX/JAXlib 0.11.0 and Torch 2.13.0. The OMP, OpenBLAS and MKL thread limits
are each one. These records establish no hardware performance comparison.

The manifest is `fdtd_two_debye_film_calibration_attempt1.json`. Raw results, logs and
frozen probe/controller copies remain in the attempt directory
`fdtd_two_debye_film_calibration_attempt1`. Both are archived outside the repository. All six raw
result/log hashes and both frozen-source hashes match the manifest. Each
worker confirms unchanged before/after source identity, and the controller
records `source_unchanged=true`. Its FD digest is
`d61cafe1d30073c857fcd70611d69ad979174c9a288f0121be2019e07c65b6ae`,
using sorted paths relative to `src/wavebit/fd`, NUL separators and
LF-normalized source bytes. These responses are known exposure for any
subsequent registered study.

## Targeted checks and actual mutation witnesses

The combined targeted run collected and passed all 374 cases, with no skips,
xfailed or xpassed, exit 0, in 65.30 seconds as reported by pytest. It
comprises 127 new cases in nine test functions, 99 single-Debye cases,
94 Drude/Lorentz film cases, 53 conductive-film cases and one existing
stencil refusal case. The manifest, collection/run logs and test/controller
snapshots are retained as `targeted_attempt1.json` and the rest of the study's controls
directory, `fdtd_two_debye_film_controls`, archived outside the repository.

Nine actual production mutations each produce an assertion or missing-refusal
failure, exit 1:

| Mutated behaviour | Required witness |
| --- | --- |
| Omit the normal relaxation transformation | Coupled material-response check fails. |
| Do not sum equal-time increments | Collapse-limit check fails. |
| Omit the second memory's energy weight | Local energy check fails. |
| Reduce the second memory's drive to one quarter | Complex Airy film check fails. |
| Remove the initial-field refusal | Unsupported-scene check detects a missing refusal. |
| Remove the prepared signature check | Plan-reuse check detects a missing refusal. |
| Detach the second traced increment | Traced-material check detects a missing refusal. |
| Omit the second local material rate | Summed-rate check fails. |
| Omit the second spectral pole | Bulk spectral guard check fails. |

All modified production bytes are restored, and before/after source identities
match. The restored new-test file passes all 127 cases, exit 0, in 50.67
seconds as reported by pytest. The canonical manifest is
`fdtd_two_debye_film_mutations.json`; attempt 1, its raw logs, test/harness snapshots and
original/mutant source copies remain under the corresponding
`fdtd_two_debye_film_mutations_attempt1` names. All of them are archived outside the
repository. These witnesses check the tests' ability to detect the
specified errors; they do not establish arbitrary-material accuracy.

## Registration and retained packaging correction

The accepted registration, archived outside the repository, is
`fdtd_two_debye_film_registration.json`, dated
2026-10-06T13:43:47 UTC, with fingerprint
`3c883d48c26ee57c91cae5c42a69d8da02622b46367de77974c2aed0d8d64fbe`.
It follows an invalid first registration that is preserved rather than
silently replaced.

The first registration command exited 0, but its launcher redirected stdout
to a file that the registration itself included as known exposure. It
therefore recorded the empty-file checksum before its own output was
written. The completed log no longer matched that checksum. This was found
before any campaign worker or response. The unchanged controller was invoked
again after the first log had completed, without another in-flight redirected
launcher file.

The initial fingerprint was
`a0945089d0b78adf039184fbe166f6e10fa3626abf9cf63c0eca620c142b7fda`.
Only the registration timestamp, the completed launcher-log exposure hash
and the resulting fingerprint differ. Production, tests, controller, probe,
criteria and protocol are unchanged. The initial registration and four
frozen source copies remain in the attempt directory
`fdtd_two_debye_film_registration_attempt1`; the correction and checksums are recorded in
`fdtd_two_debye_film_registration_packaging.json`. Both are archived outside the
repository.
All preserved hashes and all 61 accepted exposure/frozen-source links match.
The first campaign worker started at 13:44:05 UTC, after the accepted
registration. No campaign response preceded the correction.

## Registered physical campaign

The campaign uses the disclosed material and thickness, face fractions 0.3
and 0.1, both s and p polarization, three wavelengths 0.9, 1.0 and 1.1,
and NumPy with the exact `Axis1D` transverse derivative. Each of four
placement/polarization groups has five fixed rows:

| Role | Cells per centre wavelength | Periods | PML cells |
| --- | --- | --- | --- |
| Coarse base | 40 | 120 | 24 |
| Middle base | 80 | 120 | 24 |
| Duration control | 80 | 240 | 24 |
| Absorber control | 80 | 240 | 48 |
| Fine base | 160 | 120 | 24 |

All 24 placement/polarization/wavelength/complex-coefficient cells are
classified independently. Their middle-grid empirical proxy is

```text
u = abs(v80_240_PML24 - v80_120_PML24)
  + abs(v80_240_PML48 - v80_240_PML24).
```

Self-order is `log2(abs(v40-v80)/abs(v80-v160))`; its sensitivity interval
adds and subtracts `2*u` from each adjacent change. Direct order is
`log2(error_coarse/error_fine)`; its interval adds and subtracts `u` from
each absolute complex error. Both changes or errors must exceed `1e-10`,
with `2*u/min(adjacent changes or errors)<=0.1`. A resolved interval is
second only when entirely inside `[1.8,2.3]`, low when its upper endpoint
is at most 1.75, and otherwise mixed. A failed noise gate is unresolved.
The proxy is not a rigorous fine-grid error bound. Replacements and stops
follow the fixed registration; order labels cannot trigger additional
mesh, duration or geometry choices.

### Measured result and remaining limits

All 20 workers exit 0, report stable fields and finite energy, and retain
the recorded source identity. Every group selects its original 120-period
base rows: no replacement or stop occurs. All 40 raw result/log hashes
match the ledger, and the procedural audit recomputes all 24 cells.

| Classification | Self-order cells, 40/80/160 | Direct pairs, both mesh pairs | Final direct cells, 80 to 160 |
| --- | --- | --- | --- |
| Second | 24 | 48 | 24 |
| Low, mixed or unresolved | 0 | 0 | 0 |
| Total | 24 | 48 | 24 |

Self point estimates range from 1.993797 to 2.215318. The following envelopes
show the smallest lower and largest upper endpoints among six independent
cell sensitivity intervals per group; they are not pooled confidence
intervals.

| Face fraction | Polarization | Self-interval envelope | Final direct-interval envelope |
| --- | --- | --- | --- |
| 0.3 | s | [2.006009, 2.176121] | [2.001216, 2.085152] |
| 0.3 | p | [1.989901, 2.229082] | [1.989488, 2.128853] |
| 0.1 | s | [2.006844, 2.046215] | [2.001372, 2.017351] |
| 0.1 | p | [1.994057, 2.057532] | [1.986162, 2.027798] |

Across the registered cells, the largest absolute complex errors are:

| Cells per centre wavelength | Reflection `r` | Transmission `t` |
| --- | --- | --- |
| 40 | 0.007719271817 | 0.010084340268 |
| 80 | 0.001921250878 | 0.002518270695 |
| 160 | 0.000479775649 | 0.000630297828 |

The coarse maxima exceed the calibration-row regression ceilings of
0.005/0.010. Those ceilings belong only to the known p/fraction-0.3
configuration; they are not campaign accuracy thresholds. The largest
transmission error is p polarization at fraction 0.1, wavelength 0.9.
The campaign's unchanged acceptance criterion is its per-cell convergence
interval with the declared noise gate, not an absolute-error ceiling shared
by every placement.

The largest proxy is 2.820159e-6, for p reflection at fraction 0.3,
wavelength 1.1: its duration contribution is 2.714259e-6 and its absorber
contribution 1.059001e-7. The largest self-order noise share is 0.00825754
at that same cell; the largest direct-order share is 0.02586359 at p
reflection, fraction 0.3, wavelength 0.9. Both remain below the fixed 0.1
limit. The largest monitor decay ratio across workers is 1.173587e-8.
These diagnostics support the declared finite-run classification without
making the empirical proxy a rigorous error bound.

The known coarse p/fraction-0.3 response matches the NumPy calibration
exactly in all recorded physical values; only timestamps and wall time
differ. It remains a disclosed repeat, not an independent corroborating
arm. The registration also discloses targeted and mutation exposures.

Full cell values, sensitivity intervals and noise terms are in
`studies/fdtd/two_debye_film/records/fdtd_two_debye_film_report.json`. The worker ledger is
`fdtd_two_debye_film_campaign.jsonl`, with raw files in the job directory
`fdtd_two_debye_film_campaign_jobs`; the procedural audit is
`fdtd_two_debye_film_campaign_audit.log`. The ledger, the job directory and the audit log
are archived outside the repository.

A separate read-only audit reconstructs the continuum film with a complex
cosine/sine transfer matrix, without importing the solver, probe, campaign
or classification helper. It agrees with the stored continuum coefficients
within 8.881784197001252e-16 and reconstructs all cell arithmetic within
6.661338147750939e-16. Its 128-double-epsilon comparison budget concerns
rounded audit quantities only; the scientific gates are unchanged. It
checks 206 input hashes and independently confirms all 24 self-orders and
48 direct pairs as second, with the following envelopes of interval
endpoints:

| Intervals included | Smallest lower to largest upper endpoint |
| --- | --- |
| All 24 self-order intervals | [1.989901, 2.229082] |
| All 48 direct-order intervals | [1.986162, 2.195470] |
| Final 24 direct-order intervals | [1.986162, 2.128853] |

The audit script, log and result are retained as
`fdtd_two_debye_film_independent_audit_attempt1.py.txt`,
`fdtd_two_debye_film_independent_audit_attempt1.log` and
`fdtd_two_debye_film_independent_audit_attempt1.json`, archived outside the repository.
This independent reconstruction checks recorded quantities and reference
algebra; it adds no new field response or physical acceptance criterion.

The measured conclusion is second-order behaviour under the registered
criterion for these material values, thickness, incidence, placements,
wavelengths and mesh ladder. It is not a universal convergence theorem or
evidence for other materials, geometries, gradients or absorber stability.
The bulk spectral guard retains its narrower scale-resolution meaning.

## Regression gates and evidence preservation

The final numerical validation completes with every command exiting zero:

| Gate | Recorded result |
| --- | --- |
| Full tier | 3,130 collected; 3,122 passed, eight skipped |
| Separate slow tier | 81 collected; 81 passed, none skipped |
| Frozen stage 0 | 143 of 143 arrays bit-exact |
| Frozen stage 1 | 288 of 288 arrays bit-exact |
| Frozen stage 2 | 140 of 140 arrays bit-exact |
| Frozen stage 3 | 570 of 570 arrays bit-exact |

The full and slow pytest runs take 2,932.64 and 1,345.28 seconds respectively
(wrapper times 2,933.608247 and 1,345.920876 seconds). Neither tier reports
an expected failure or unexpected pass. All 1,141 reference arrays match;
the read-only freezers write no golden files. Stage 2/3 logs also note
changes to historical case-list or freezer-source provenance. Those notices
are distinct from numerical mismatches: the arrays and committed golden
bytes remain unchanged.

The final manifest is `fdtd_two_debye_film_validation.json`, with raw logs in the
controls directory; both are archived outside the repository. The validation wrapper records
FD source digest
`3395a14fbb4910ce2dca0b77ea2dff6f0dbea97a7a03ee1edbcc2716684190bd`
using sorted relative paths and LF-normalized source bytes without delimiters.
This is a different serialization from the campaign's NUL-delimited digest,
not a change of tested source. The preserved prevalidation maps cover 97
production files, 81 FD test files, four golden files and 607 earlier film
evidence files; all remain unchanged through these gates. The maps and
frozen validation inputs are recorded in
`fdtd_two_debye_film_prevalidation.json` and
`fdtd_two_debye_film_validation_setup.json`, archived outside the repository.

Earlier measurements retain their original material scopes and open limits.
General C8 smoothing is incomplete and C7 remains unpassed.
