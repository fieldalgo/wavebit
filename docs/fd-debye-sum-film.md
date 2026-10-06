# C8: a finite Debye sum in one film

Written: 2026-10-07

Targets: `wavebit/fd/_lower/debye_sum_film.py`,
`wavebit/fd/_lower/debye_sum_resolution.py`, `wavebit/fd/materials.py`,
`wavebit/fd/_kernel/poles.py`, `wavebit/fd/_kernel/observe.py`,
`wavebit/fd/_lower/checks.py`.

## TL;DR

**The registered four-pole study classifies all 24 self-orders and all
48 direct pairs as second under its fixed sensitivity-interval criterion.**
All 20 workers complete without replacement or stop. This result concerns
the specified material, two placements and 40/80/160-cell ladder; it is not
a convergence guarantee for every admitted finite sum.

The finite passive Debye-sum realization is implemented for the existing
isolated flat-film geometry. All 536 targeted cases pass; nine actual
production mutations are caught, followed by 162 passing restored cases.
The full tier passes 3,285 cases with eight skips, the separate slow tier
passes all 81, and all 1,141 frozen-reference arrays reproduce bit for bit.
The bounded implementation and its required numerical gates are complete.
The new path handles three or more declared Debye poles.
The existing [single-pole](fd-debye-film.md) and
[two-pole](fd-two-debye-film.md) paths retain their own arithmetic and
recorded evidence.

The normal interface law couples the complete susceptibility sum. Applying
the single-pole transform independently to each declared pole would describe
a different material. A positive symmetric relaxation operator gives a
finite passive realization of the exact harmonic mixture. The tangential
law remains arithmetic. Every active effective relaxation contributes to
the existing split memory, local energy and summed material-rate policy.

The scope remains one concrete, axis-aligned film in a plain dielectric
host. Finite Debye count does not mean arbitrary pole families or arbitrary
geometry. Curved dispersive bodies, mixed-family sums, lossy or dispersive
hosts, junctions and geometry/material/band differentiation remain outside
this extension. General C8 smoothing remains incomplete, and the
[C7 metasurface verdict](fd-metasurface-validation.md) remains unpassed.

## Bounded contract

The declaration remains a single
`fd.Structure(fd.Box(...), fd.Dispersive(eps_inf=..., poles=(...)))`.
The declared tuple contains a finite positive number of exact public
`Debye` objects. Counts one and two select their existing implementations;
the new realization applies from count three onward.

| Item | Bounded case |
| --- | --- |
| Geometry | One exact concrete `Box`, `angle=0`, `range=None`, using voxel discretization; no partner or junction. |
| Extent | Exactly two complete grid axes are uniform through the film, with at least one `dx` of overhang beyond each transverse domain face. |
| Thickness | Normal extent is at least one `dx`, keeping the two faces out of the interior of the same electric dual cell. |
| Clearance | Normal support lies strictly inside the domain; all support stays clear of every absorbing layer. |
| Film | Exact `Dispersive`, finite positive isotropic high-frequency permittivity, and `mu=1`. |
| Relaxations | Every pole is exact `Debye`, with concrete finite `delta_eps >= 0` and `tau > 0`. Zero increments and exactly equal times are allowed. |
| Host | Exact plain lossless `Material`, finite positive isotropic permittivity, and `mu=1`. |
| State | No initial field. A reused plan must match the geometry, material, grid, declared pole count and reserved memory representation. |

The restrictions are invariant under permutation of the grid axes.
Occupancy uses the analytic overlap at each electric component's own Yee
dual cell. Empty and full endpoints are exact; a tolerance must not erase a
small positive mixed fraction or pole strength.

The declared count fixes memory allocation before a run. Zero-strength or
equal-time reductions may reduce the number of physically active modes,
but must not silently change a prepared plan's slot count. This is a
concrete material transformation, not a traced eigenvalue calculation.
Existing source, monitor, feature-size and absorber restrictions still
apply. There is no gradient-correctness claim, including for source
amplitudes.

## The harmonic law couples the whole sum

With the repository convention `exp(-i*omega*t)`, let `s=-i*omega`, film
fraction `f`, host fraction `g=1-f`, film high-frequency permittivity `a`,
and host permittivity `b`. For each active declared relaxation, define

```text
r_i = 1/tau_i
u_i = sqrt(delta_i*r_i)
D = diag(r_i)
chi(s) = sum_i delta_i/(1 + tau_i*s)
       = u^T * (s*I + D)^(-1) * u
eps_film(s) = a + chi(s)
A = f*b + g*a.
```

The arithmetic tangent and harmonic normal responses are

```text
eps_t(s) = f*a + g*b + f*chi(s)
eps_n(s) = 1/(f/(a + chi(s)) + g/b)
         = a*b/A + (f*b*b/A**2)*chi(s)/(1 + (g/A)*chi(s)).
```

The rank-one inverse identity gives

```text
J = D + (g/A)*u*u^T
eps_n(s) = a*b/A + (f*b*b/A**2)*u^T*(s*I + J)^(-1)*u.
```

For positive relaxation times, `D` is positive definite. The rank-one term
is positive semidefinite, so `J` is positive definite. If `lambda_j` and
`q_j` are its eigenvalues and orthonormal eigenvectors, the exact passive
modal representation is

```text
tau_eff_j = 1/lambda_j
delta_eff_j = (f*b*b/A**2)*(q_j^T*u)**2/lambda_j
eps_inf_normal = a*b/A.
```

Each effective increment is nonnegative. A zero projection carries no
physical relaxation. Equal declared times can be combined exactly before
the reduction; distinct times must not be merged merely because they are
close. The tangential representation retains the declared times and uses
increments `f*delta_i` with `eps_inf_tangent=f*a+g*b`.

This matrix identity establishes the material law, not the numerical
accuracy of a particular eigensolver or root finder. The implementation
must preserve small active residues, distinguish exact reductions from
numerical loss of a mode, and refuse active derived quantities that its
finite-precision representation cannot carry. Admission of every
mathematically finite parameter combination is not promised.

## Certified numerical realization

The implementation treats each supplied binary floating-point value as
an exact rational number during the preparation-time reduction. Exact zero
strengths are removed from the active calculation, and exactly equal
relaxation times are combined. This does not use approximate equality to
merge distinct poles. The declared count still reserves the slots; active
effective modes are ordered by increasing relaxation time.

For a genuinely mixed normal cell, put `alpha=g/A`,
`kappa=f*b*b/A**2` and `w_i=delta_i*r_i`. Rather than recovering a small
residue by subtracting nearly parallel floating-point eigenvectors, solve
the scalar secular equation with rational brackets:

```text
F(lambda) = 1 + alpha*sum_i w_i/(r_i-lambda) = 0
delta_eff = kappa/(lambda*alpha**2*sum_i w_i/(r_i-lambda)**2)
tau_eff = 1/lambda.
```

After the exact reductions, the active rates are distinct and the weights
positive. The roots interlace the rates; the last root is no greater than
`max(r_i)+alpha*sum_i w_i`. Bisection can therefore refine disjoint root
brackets. The admission check encloses both the resulting time
and increment and accepts a mode only when each enclosure rounds to one
identical positive finite float64 value. A small residue is retained if it
is representable; a missing representable result is a refusal, not a
zero-strength approximation.

The precision budget is 4,096 bisections per active mode, followed
by a named refusal if an enclosure is still unresolved. This is a numerical
work bound, not a cap on declared pole count. Pure-host/full-film endpoints
and exact reductions avoid evaluating the mixed-cell formula at `alpha=0`.
Up to 128 immutable normal-mode reductions are cached by concrete film/host
permittivities, face fraction and canonical active-pole tuple. Neither cell
arrays nor traced values enter that cache. Tangential and full-film cells
need no secular solve.

## Split memory and local energy

For each effective increment `delta` and relaxation time `T`, use the
existing scaled trapezoidal split:

```text
m = max(T, dt/2)
left = T/m
right = (dt/2)/m
u_split = left/(left+right)
v_split = right/(left+right)
gain = u_split-v_split
c = delta*v_split
drive = 2*delta*u_split*v_split
Q = P-c*E
Q^(n+1) = gain*Q^n + drive*E^n.
```

The electric update uses `eps_inf_eff+sum_j c_j`. At the observer time
level, the material correction is

```text
sum_active_j (|Q_j^n + c_j*E^n|**2/delta_j - c_j*|E^n|**2).
```

Weights apply at their own spatial cells before reducing components or
clipped observation regions. Inactive slots have zero weights and drive.
The subtraction cancels the instantaneous shift already included in the
electric energy. The algebraic identity does not replace propagation,
finite-energy or long-time checks.

A recurrence-frequency oracle evaluates the exact mixture at
`omega_hat=2*tan(omega*dt/2)/dt`. A physical continuum reference instead
uses the actual frequency, leaving temporal and spatial discretization
errors visible.

## Summed material-rate policy

The conservative policy takes the maximum of the bulk summed rate and the
actual transformed sum at each Yee component:

```text
R_bulk = sum_i delta_i*r_i*r_i/a
R_cell = sum_j delta_eff_j/(eps_inf_eff*tau_eff_j*tau_eff_j)
R_film = max(R_bulk, max_actual_cells_and_components(R_cell))
dt_bound = 1/sqrt(1/dt_nondispersive**2 + R_film/4).
```

All simultaneously active modes contribute. Taking only the largest pole
would drop part of the material response; adding bulk and transformed
rates would count the same film twice. For a normal cell the exact moment
identity is

```text
W = sum_i delta_i*r_i
R_normal = (f*b/(a*A))*(sum_i delta_i*r_i*r_i + (g/A)*W*W).
```

It follows from `u^T*J*u`. The normal rate can exceed the bulk sum.
Only the matching admitted film replaces the ordinary declared-pole rate
path with this policy. A zero increment contributes zero before evaluating
its unused reciprocal time or squared rate. This conservative repository
policy is not a necessary stability condition for passive trapezoidal
relaxation alone, and does not extend the absorber stability contract.

## Bulk spectral floor for a finite sum

For `x=omega*omega`, expansion of the complete passive sum gives

```text
|eps(omega)|**2 = a*a + sum_i C_i/(1 + tau_i*tau_i*x)
C_i = delta_i*delta_i + 2*a*delta_i
    + 2*sum_(j != i) delta_i*delta_j*tau_i/(tau_i+tau_j).
```

Every `C_i` is nonnegative. Each term in
`|k|**4=x*x*|eps|**2` has nonnegative derivative, and `a*a*x*x` makes the
sum strictly increasing for positive `x`. Therefore the largest bulk
wavevector modulus on a positive source band occurs at its upper endpoint.
The guard evaluates the entire declared response there, for both film and
host, and requires `dx*|k_max| <= 2*pi/8`.

Every declared monitor wavelength, the `Resolution` wavelength and explicit
PML wavelengths also retain the phase/skin floors of eight cells per phase
wavelength and one per skin depth. Existing warnings below twelve and two
remain. Bands and sampled wavelengths must be concrete, finite and positive.

This is a bulk scale-resolution floor, not a scattering-accuracy theorem
or a certificate for every interface or oblique longitudinal scale.
Duration and absorber controls remain necessary for measured convergence.

## Disclosed algebra and preparation-cost measurement

The first standalone algebra probe completes 23 fixed cases with declared
counts 3, 4, 8 and 12. It covers ordinary and reversed tuples, exact equal
times, adjacent distinct floating-point times, tiny active strengths,
all-zero and one-active reductions, and a subnormal-range refusal. Twenty-two
cases return the expected active modes; the subnormal case receives the
expected explicit refusal. No mode is accepted by clipping a small residue.

At angular frequencies 0.5, `2*pi` and 25, the largest relative difference
from the independently evaluated rational harmonic response is
1.8613182691395824e-16. The first two high-frequency moments agree within
3.7834299533191136e-16. The comparison budget, fixed before the responses,
is 512 double-precision epsilons relative to the reference magnitude, with
the smallest normal float as its denominator floor. This budget concerns
rounded algebraic comparisons, not field accuracy.

The largest recorded mode-reduction time is 0.216402 seconds, for the
12-pole tiny-strength case with a cold cache. Reversed declarations reuse
the canonical cache entry. These are local preparation measurements with
no timing pass criterion; they do not establish scaling to every finite
count or whole-solver performance.

The probe executes selected production functions extracted from source,
without importing the solver package or performing a field solve. Its
script and all 99 production Python files were archived, and the initial
manifest written, before those functions executed. All 100 recorded raw
source hashes and the archive hash remain unchanged. The complete record
and source archive are
`studies/results/fdtd_debye_sum_film_algebra_attempt1.json` and
`studies/results/fdtd_debye_sum_film_algebra_attempt1_sources.zip`.
This is a disclosed development calculation, not a pytest gate, full-scene
admission test or registered physical convergence result.

## Disclosed three-backend calibration

The first calibration runs three sequential workers on 2026-10-06,
15:38:23 to 15:39:04 UTC. All exit zero, with no adapted row and unchanged
production source. The fixed configuration is:

| Setting | Value |
| --- | --- |
| Film and host | `eps_inf=2`, increments `(1,1,2,2)`, relaxation times `(0.04,0.08,0.16,0.32)`; vacuum host |
| Geometry | Thickness 0.2; face fraction 0.3; 40 cells per centre wavelength |
| Excitation | p polarization, 35 degrees at centre wavelength 1; exact transverse `Axis1D` with fixed `kx=3.6038870374253196` |
| Samples | Wavelengths 0.9, 1.0 and 1.1, each at its own fixed-`kx` incidence angle |
| Duration and absorber | 120 periods, 24 PML cells, Courant 0.5 |
| Source band | Centre wavelength 1, `df=0.1`, cutoff `1e-6` |
| Time grid | `dt=0.007216878364870323`, 16,628 steps |

The continuum reference evaluates the declared Debye sum independently
and uses a complex film/Airy calculation. The NumPy complex-amplitude
errors are:

| Wavelength | Reflection `r` | Transmission `t` |
| --- | --- | --- |
| 0.9 | 0.003993837708 | 0.009375278239 |
| 1.0 | 0.003951685462 | 0.007943278066 |
| 1.1 | 0.003479144091 | 0.006599623867 |

Across all three backends, the maxima are 0.003993837707944514 and
0.00937527823855372. The inherited candidate ceilings of 0.005 for
reflection and 0.010 for transmission were accepted unchanged after this
disclosure. The transmission margin is 0.000624721761446281. They apply
only to this known 40-cell p/fraction-0.3 configuration, not to unseen
materials, other campaign rows or a universal accuracy claim.

The largest backend difference, scaled by `max(1,abs(NumPy value))`, is
9.992970910418892e-16 against the existing 128-double-epsilon budget
2.842170943040401e-14. Every worker reports stable fields and 1,041 finite
energy samples. The largest reflection/transmission decay ratios are
1.0177013480271628e-8 and 1.4673743010763767e-11. The
`SubpixelAccuracyWarning` remains in every raw record: the spectral guard
certifies bulk scale resolution, not scattering accuracy.

The environment is Windows AMD64, Python 3.12.14, NumPy 2.5.1,
JAX/JAXlib 0.11.0 and Torch 2.13.0, with OMP, OpenBLAS and MKL thread
counts set to one. Worker wall times are 7.7645, 5.3392 and 27.7193 seconds
for NumPy, JAX and Torch respectively, including their worker setup; this
is not a backend performance comparison.

The manifest and accepted calibration ceilings are
`studies/results/fdtd_debye_sum_film_calibration_attempt1.json` and
`studies/results/fdtd_debye_sum_film_calibration_acceptance.json`.
The raw three-backend JSON/log pairs and frozen probe/controller are under
`studies/results/fdtd_debye_sum_film_calibration_attempt1/`.
The calibration and algebra records share FD source digest
`27a90c01598447fa9354873141e37d5e0b0490120246211ac71213309f4dc261`
and package digest
`b8e000e3b7613dba5e7319e76738f4b8ac7ed831b29a595ebf793e841246cd9d`,
using sorted relative POSIX paths, NUL separators and LF-normalized bytes.

## Targeted checks and mutation witnesses

The first targeted invocation collects and passes all 536 cases with exit
zero, no skips and no expected failures or unexpected passes. This includes
162 new parameterized cases across nine test functions, the four earlier
film suites and the existing stencil-refusal compatibility case. Pytest
takes 51.84 seconds; the wrapper records 52.404453 seconds. Production and
probe identities are unchanged before and after the run.

The manifest is
`studies/results/fdtd_debye_sum_film_controls/targeted_attempt1.json`.
Its collection/run logs and frozen runner/test snapshots remain in the
same directory. These checks separate response algebra, numerical
representation, recurrence energy, summed planning rates and scope
refusals from physical propagation.

Mutation attempt 1 catches all nine actual production changes on the
intended assertion or missing-refusal check, each with exit one:

| Deliberate production change | Observable failure |
| --- | --- |
| Replace the harmonic normal reduction with arithmetic poles | Declared-sum discrete response differs. |
| Overwrite rather than sum equal-time increments | Exact-collapse response differs. |
| Remove the last memory's inverse energy weight | Local summed energy differs. |
| Reduce the last memory drive to one quarter | Complex film/Airy regression fails. |
| Remove the initial-field restriction | Unsupported initial-field scene is admitted. |
| Remove prepared-plan identity checks | A changed last strength reuses old memory. |
| Silently detach traced pole strengths | A traced last strength bypasses the concrete-material refusal. |
| Keep only the first local CFL contribution | The summed local-rate check fails. |
| Keep only the first bulk spectral pole | The full-band resolution refusal is lost. |

Every changed source is restored byte for byte; the complete source
identities before and after agree. The restored suite collects and passes
all 162 cases, with no skip, failure or expected/unexpected pass, exit zero.
It takes 46.76 pytest seconds (48.852146 wrapper seconds). The canonical
manifest is `studies/results/fdtd_debye_sum_film_mutations.json`; original
and mutant source snapshots, raw witness logs and the restored baseline
remain under `studies/results/fdtd_debye_sum_film_mutations_attempt1/`.
The witnesses establish sensitivity to these faults, not arbitrary-material
accuracy or a physical convergence result.

## Registered physical campaign

Registration completes on its first attempt at 2026-10-06
15:48:00.431256 UTC, before campaign responses. Its fingerprint is
`f113b590a811d1d0653d49eb53d897118f3f356fe6a8cca9e16d44d4dcc9d498`.
The fixed record is
`studies/results/fdtd_debye_sum_film_registration.json`, which links 59
known-exposure records and four frozen source files. Calibration,
standalone algebra, targeted tests and mutation exposures are disclosed.
The matching coarse p/fraction-0.3 campaign row is a repeated known
response, not independent corroboration.

The registered material is exactly the four-pole calibration material,
with thickness 0.2 and 35-degree centre incidence in vacuum. NumPy runs
both s and p polarizations at face fractions 0.3 and 0.1, using the same
exact transverse derivative, fixed `kx`, source band and three wavelengths.
Each of these four groups has the following five scheduled workers:

| Order | Cells per centre wavelength | Periods | PML cells |
| --- | --- | --- | --- |
| 1 | 40 | 120 | 24 |
| 2 | 80 | 120 | 24 |
| 3 | 80 | 240 | 24 |
| 4 | 80 | 240 | 48 |
| 5 | 160 | 120 | 24 |

This schedules 20 base/control workers. An otherwise valid but unsettled
120-period base row may receive one 240-period replacement; an existing
80/240/PML24 control is reused if available. A failed fixed control or
replacement stops the whole campaign. There is no duration beyond 240,
extra grid rung, geometry change or adaptation to the measured order. The
decay limit is `1e-5`.

For each complex reflection/transmission cell, define the empirical
middle-grid sensitivity proxy

```text
u = |v80,long24 - v80,base24| + |v80,long48 - v80,long24|.
```

The 120-period base remains in this definition even if a long replacement
is selected. Self-order is `log2(|v40-v80|/|v80-v160|)`; a sensitivity
interval is constructed by adding and subtracting `2*u` from each adjacent
change. Both changes must exceed `1e-10`, and `2*u` must not exceed one
tenth of the smaller change.

Direct order is `log2(error_coarse/error_fine)` against the independently
evaluated declared Debye sum and continuum complex film/Airy response.
Its sensitivity interval adds and subtracts `u` from each error. Both
errors must exceed `1e-10`, and `2*u` must not exceed one tenth of the
smaller error. The proxy is empirical; these are not rigorous fine-grid
error bounds or statistical confidence intervals.

The fixed classifications are second when the entire interval lies in
`[1.8,2.3]`, low when its upper endpoint is at most 1.75, mixed for other
resolved intervals, and unresolved when a validity/noise gate prevents a
classification. All 24 placement/polarization/wavelength/coefficient cells
are reported individually, including both direct rung pairs; a worst-cell
summary cannot replace them. Measured and reference `1-R-T` are retained
as absorptance, not treated as a lossless energy-closure error.

### Measured result and its limits

All 20 workers exit zero with stable fields, finite recorded energy and
unchanged source. Every group uses its original 120-period base rows;
there is no replacement or global stop. The frozen controller's procedural
audit recomputes every cell and passes. All 24 self-orders, all 48 direct
pairs and all 24 final direct pairs are second. There are no low, mixed or
unresolved cells under the registered rule. Self-order point estimates
range from 1.993512 to 2.191648.

The following are envelopes over each group's six independent intervals,
not pooled estimates or confidence intervals:

| Face fraction | Polarization | Self-order interval envelope | Final direct-order interval envelope |
| --- | --- | --- | --- |
| 0.3 | s | [2.006001, 2.156687] | [2.001214, 2.074189] |
| 0.3 | p | [1.989275, 2.205619] | [1.982360, 2.114193] |
| 0.1 | s | [2.006241, 2.037664] | [2.000610, 2.013917] |
| 0.1 | p | [1.993781, 2.048403] | [1.984888, 2.024376] |

Maximum absolute complex errors across the registered cells are:

| Cells per centre wavelength | Reflection `r` | Transmission `t` |
| --- | --- | --- |
| 40 | 0.007525227466 | 0.009708960243 |
| 80 | 0.001875193613 | 0.002425240488 |
| 160 | 0.000468591928 | 0.000607048711 |

The reflection maximum is s polarization at fraction 0.1, wavelength 1.1;
the transmission maximum is p at fraction 0.1, wavelength 0.9. The coarse
reflection maximum exceeds the calibration's 0.005 reflection ceiling.
That ceiling belongs only to the known p/fraction-0.3 regression case;
the campaign criterion is per-cell convergence and its noise gate, not a
shared absolute-error ceiling. Neither the calibration ceiling nor the
campaign criterion was widened to accommodate this result.

The largest proxy is 2.806531e-6 at p reflection, fraction 0.3, wavelength
1.1: duration contributes 2.701434e-6 and the PML control 1.050962e-7.
The largest `2*u/min(signal)` shares are 0.00874647 for self-orders and
0.02640575 for direct pairs, both below 0.1. No self/direct signal vanishes.
The largest monitor decay ratio is 1.190624e-8. These diagnostics support
the fixed finite-run classification while leaving the proxy's empirical
character unchanged.

The known coarse p/fraction-0.3 row repeats every NumPy calibration physical
value exactly. It remains disclosed prior knowledge, not an independent
arm. Raw workers are under
`studies/results/fdtd_debye_sum_film_campaign_jobs/`, the ledger is
`studies/results/fdtd_debye_sum_film_campaign.jsonl`, and the full per-cell
values and procedural audit are
`studies/results/fdtd_debye_sum_film_report.json` and
`studies/results/fdtd_debye_sum_film_campaign_audit.log`.

A separate standard-library audit reads the saved data without importing
the solver, probe, campaign or classification helper. Its independently
written complex cosine/sine transfer matrix agrees with the continuum
coefficients within 8.881784197001252e-16; rederived cell quantities agree
within the same amount. Its 128-double-epsilon rounded-comparison budget
does not alter the scientific criteria. It checks 210 input hashes and
independently confirms all 24/48/final-24 second classifications. Across
all cells, interval endpoints have these envelopes:

| Included intervals | Smallest lower to largest upper endpoint |
| --- | --- |
| 24 self-order intervals | [1.989275, 2.205619] |
| 48 direct-order intervals | [1.982360, 2.173370] |
| 24 final direct-order intervals | [1.982360, 2.114193] |

The independent result, source and observed execution record are
`studies/results/fdtd_debye_sum_film_independent_audit_attempt1.json`,
`studies/results/fdtd_debye_sum_film_independent_audit_attempt1.py.txt` and
`studies/results/fdtd_debye_sum_film_independent_audit_attempt1_execution.json`.
The raw log and runner snapshot are retained alongside them. This audit
adds no field solve or additional physical acceptance claim.

The measured conclusion applies to this four-pole material and fixed
experiment, not every admitted count, parameter set or geometry. The
upper-band bulk spectral result remains a scale floor rather than a
scattering theorem. Existing one- and two-pole measurements keep their
original scopes.

## Regression gates and evidence preservation

The final numerical validation completes with every command exiting zero:

| Gate | Recorded result |
| --- | --- |
| Full tier | 3,293 collected; 3,285 passed, eight skipped |
| Separate slow tier | 81 collected; 81 passed, none skipped |
| Frozen stage 0 | 143 of 143 arrays bit-exact |
| Frozen stage 1 | 288 of 288 arrays bit-exact |
| Frozen stage 2 | 140 of 140 arrays bit-exact |
| Frozen stage 3 | 570 of 570 arrays bit-exact |

The full and slow pytest runs take 2,403.12 and 1,156.13 seconds respectively
(wrapper times 2,403.832170 and 1,156.648677 seconds). Neither tier reports
an expected failure or unexpected pass. The raw logs retain 1,063 full-tier
and 50 slow-tier warnings. All 1,141 reference arrays match, and the
read-only freezers write no golden files. Stage 2/3 logs also retain
historical case-list or freezer-source provenance notices. These are
distinct from numerical mismatches; the committed golden bytes remain
unchanged.

The final manifest is
`studies/results/fdtd_debye_sum_film_validation.json`, with raw logs under
`studies/results/fdtd_debye_sum_film_controls/`. The validation wrapper
records FD source digest
`fe58bb53f8e9d0bb046d9a33f7b02d82075692d26e009ca989ca79e9b786918a`
using sorted paths relative to `wavebit/fd` and LF-normalized source bytes
without delimiters. The campaign uses its separately described
NUL-delimited serialization; these different digest values do not indicate
a change of numerical source. The preserved raw-byte prevalidation maps
cover 99 production files, 82 FD test files, four golden files and 748
earlier film evidence files. All remain unchanged after the gates. The
maps and frozen validation inputs are recorded in
`studies/results/fdtd_debye_sum_film_prevalidation.json` and
`studies/results/fdtd_debye_sum_film_validation_setup.json`.

This extension concerns the finite passive relaxation family in one flat
film. It does not settle curved interfaces, mixed Drude/Lorentz/Debye sums,
lossy or dispersive hosts, junctions, gradients or the earlier C7 failure.
