# C8: separated single-oscillator films

Written: 2026-10-09

Targets: `src/wavebit/fdtd/_prepare/oscillator_stack.py`,
`src/wavebit/fdtd/_prepare/single_pole_film.py`,
`src/wavebit/fdtd/_prepare/single_pole_resolution.py`,
`src/wavebit/fdtd/_prepare/plan.py`, `src/wavebit/fdtd/_prepare/params.py`,
`src/wavebit/fdtd/_prepare/coefficients.py`, `src/wavebit/fdtd/_prepare/checks.py`,
`tests/fdtd/test_fd_oscillator_stack.py`.

## TL;DR

**This bounded extension composes separated flat films, with exactly one passive
Drude or Lorentz pole in each film. Its mandatory numerical gates pass.** Different
films may explicitly use different oscillator kinds. Every member reuses the
existing [single-film reduction](fd-single-pole-film.md), recurrence and energy
weights. Its padded support is disjoint from the others, and its memory remains
separate. No within-film sum or new oscillator reduction is introduced.

The material stability rate is the maximum of the complete member rates; it
is not a sum over film positions. Every member receives the existing whole-band
bulk check, including possible interior Lorentz resonances. These analytic
identities justify the composition but do not establish measured accuracy,
backend agreement or convergence on their own. The completed registered experiment
retains all 48 complex cells: 47 self-orders and 43 final direct
orders are classified as second under its fixed sensitivity rule. Every other
classification remains visible below. General C8 remains incomplete, and the
[C7 metasurface verdict](fd-metasurface-validation.md) remains unpassed.

## Bounded contract

| Item | Admitted case |
| --- | --- |
| Declaration | The ordinary `Scene(structures=(...))` tuple contains at least two exact `Dispersive` materials, each with exactly one exact `Drude` or `Lorentz` pole. Drude-only, Lorentz-only and cross-film mixtures use the same route. |
| Geometry | Each member is a concrete, unturned, unranged voxel `Box`, spans the same two transverse axes with the singleton's overhang, and is at least one `dx` thick along its common normal. x-, y- and z-normal films are supported structurally. |
| Separation | Actual padded half-open grid supports are pairwise disjoint along the common normal. Physical non-contact or a nominal gap measured in cells is insufficient. Two support intervals may meet at their half-open boundary without sharing a cell. |
| Host and clearance | One plain isotropic lossless nonmagnetic `Material` host. Each support clears the normal domain walls and every absorbing layer. Existing source, feature-size and TFSF shell restrictions remain. |
| Parameters | Concrete finite positive isotropic high-frequency permittivity, `mu=1`, nonnegative damping and strength, and positive declared Drude/Lorentz frequencies. The existing singleton's derived-value range checks still apply. |
| Duration | Fixed `Run` time, steps or periods. `Decay` and public initial fields are refused. |
| Differentiation | Carried derivatives in geometry, materials, sources or spectral declarations are outside this scope, including Torch forward duals. |
| Prepared reuse | The whole ordered stack, exact boundary declarations and complete ordered material/source term tuple must match. Changing members, adding sources or omitting memories requires a new plan. |

The new route excludes within-film multipole sums, Debye-containing stacks,
ordinary dielectric or conductive members, physically touching films or overlapping supports,
curved bodies, dispersive or lossy hosts, anisotropy, magnetism and junctions.
The existing singleton and cell-snapped routes retain their own contracts.
A zero-strength Lorentz declaration keeps its identity and memory slot. Its
nominal frequency must still square to a finite positive float64 value and
continues to enter the conservative stability bound, just as in the reused
singleton. This extension does not admit extreme inactive frequencies that
the singleton refuses.

## Local response and composition

For a member, write `d=omega_0**2` and `S=strength` for Lorentz, or `d=0` and
`S=omega_p**2` for Drude. With `s=-i*omega`, film fraction `f`, `g=1-f`, film
high-frequency permittivity `a`, host permittivity `b` and `A=f*b+g*a`,

```text
L(s) = s*s + gamma*s + d
eps_film(s) = a + S/L(s)

eps_t(s) = f*a + g*b + f*S/L(s)
eps_n(s) = b*(a*L(s)+S)/(A*L(s)+g*S)
         = a*b/A + (f*b*b*S/A**2)/(L(s)+g*S/A).
```

| Electric component | `eps_inf_eff` | `S_eff` | `Omega_squared` |
| --- | --- | --- | --- |
| Tangential | `f*a+g*b` | `f*S` | `d` |
| Normal, mixed | `a*b/A` | `f*b*b*S/A**2` | `d+g*S/A` |

The harmonic/arithmetic interface law and its derivation are documented in the
[single-film record](fd-single-pole-film.md#exact-complex-mixture). Exact empty
and full fractions use the host and original film, respectively; positive
mixed fractions are not snapped away. The composer pastes complete singleton
profiles into the host on disjoint supports, avoiding subtraction and readdition
of large host values. Spatially separated Drude and Lorentz members introduce
no additional local material coupling.

Each `oscillator_stack[i]` term has one ordinary oscillator slot and the existing
`P`, `previous` and `delta` arrays. Coefficients use the same centered recurrence:

```text
h = gamma*dt/2
step_gain = (2-Omega_squared*dt**2)/(1+h)
hold = (1-h)/(1+h)
drive = S_eff*dt**2/(1+h)
P_next = step_gain*P - hold*P_previous + drive*E.
```

The usual `-(dx/dt)*delta` term reaches the electric curl before the field update.
The source, kernel, energy observer and singleton numerical reducers are not
changed by the composition.

## Storage and time-step bound

On an active member cell, the existing electric storage adds

```text
(|P_next-P|**2/dt**2 + Omega_squared*Re(P_next*conj(P)))/S_eff
    - Re(E*conj(P_next-P))
```

to `eps_inf_eff*|E|**2`, under the observer's existing normalization and indexing.
The spatial inverse-strength, resonance-over-strength and active weights are
supplied directly by the singleton reducer. Inactive slots contribute zero
storage, including at their nominal resonance; no `0/0` is evaluated. The global
electromagnetic term is counted once and the disjoint local storage terms add.
A zero recurrence `hold` at `gamma*dt=2` remains an oscillator, not a Debye pole.

With normalized coupling `q**2=S_eff/eps_inf_eff`, a local material block obeys
`Omega_squared+q**2 <= d+S/a`. The normal value is exactly `d+S/a`; the tangent
value is no larger. Disjoint supports make the local material blocks a direct
sum. Consequently, the existing curl/material norm argument gives

```text
rate = max_i(d_i + S_i/a_i)
dt < 1/sqrt(1/dt_vacuum**2 + rate/4).
```

The curl bound uses physical high-frequency permittivity, including the host,
and the existing boundary derivative bounds. Each member sum is formed as an
exact rational from the singleton descriptor's float values before upward
rounding to a finite float64 rate. The inactive Lorentz `d_i` contribution is
retained conservatively. The rule neither recomputes the requested `dt` nor
extends the existing PML stability contract.

## Bulk spectral floors

For `x=omega**2`, the singleton triangle bound is

```text
|k|**2 <= a*x_max + S*max R(x)
R(x) = x/sqrt((d-x)**2 + gamma**2*x).
```

Besides the two endpoints, the only possible interior maximum is
`x=2*d**2/(2*d-gamma**2)` when the denominator is positive and the candidate
lies inside the band. Every member and the host are checked. An active undamped
Lorentz resonance inside the band has no finite bound and is refused.
Inactive Lorentz poles use the plain dielectric response even at that frequency.
This reuses the existing analytic bound evaluated in ordinary finite floating
arithmetic; it is not a new outward-rounded spectral certificate.

Source bands must define concrete finite positive frequency intervals. Monitor,
`Resolution` and explicit PML wavelengths are checked individually for at least
eight cells per phase wavelength and one per skin depth; fewer than twelve and
two, respectively, produce scoped warnings. No declared wavelength and no source
band produces a warning, not an inferred certificate. The bulk floors do not
bound every interface resonance, scattering error or asymptotic convergence.

## Registered numerical admission

All 161 new cases and 1363 related cases pass. The 14 actual
production-source defects each fail their intended assertion, and exact restoration
passes all 161 new cases. Full non-slow regression
passes 4486 cases with 8 unchanged historical skips; the
separate slow tier passes 81. All 1141 frozen arrays reproduce
bit for bit through four read-only comparisons. These counts are replayed from
raw node inventories, logs and completed parent process receipts.

The registered physical roster has two film families, two placements, both
polarizations and three wavelengths. It completes 40 worker
runs, including 0 allowed duration replacements;
all 48 complex cells remain in the report. A worker count is not concurrency.

| Item | Registered scope |
| --- | --- |
| Families | One Drude/Lorentz pair and one Lorentz triple, the latter retaining an inactive member. |
| Spatial rungs | `n=40,80,160` vacuum cells per reference wavelength. |
| Placements and samples | Fractions `0.3` and `0.1`; `s` and `p`; wavelengths `0.9`, `1.0`, `1.1`; complex `r` and `t`. |
| Base roster | Forty worker runs: five runs for each family/placement/polarization combination. This is a run count, not concurrent execution. |
| Five runs per combination | `40/base120/PML24`, `80/base120/PML24`, `80/long240/PML24`, `80/long240/PML48`, `160/base120/PML24`. |
| Reported cells | All 48 family/placement/polarization/wavelength/amplitude cells, including slow, mixed or unresolved classifications. |
| Known-case calibration | Six coarse `p`, fraction `0.3` rows: both families on NumPy, JAX and Torch. Absolute complex-amplitude ceilings were accepted after these rows were disclosed and remain known-case checks. |
| Reference | Declared continuum materials and geometry in a multilayer reference, checked independently with recursive scattering or single-film Airy controls; no fitted field response. |
| Sensitivity | Separate time and absorber differences at the middle rung, retained as an empirical proxy under the inherited order rule; not a rigorous fine-grid error bound. |

The roster and classification criteria were registered after the disclosed
calibration and targeted/mutation exposures, before the physical campaign.
The numerical evidence is complete; production integration additionally requires
the strict documentation build and the final saved-evidence audit with their
actual completed parent exits. Neither later gate is claimed complete here.
Original failures, warnings, raw records and frozen references remain preserved.
A successful plan or stable field alone cannot replace these measurements.

## Targeted verification obligations

Every collected case of these nine test functions passes in the accepted new-test roster.

| Test target | Status |
| --- | --- |
| `tests/fdtd/test_fd_oscillator_stack.py::test_each_member_advance_matches_the_declared_series_parallel_response` | 27 cases passed. |
| `tests/fdtd/test_fd_oscillator_stack.py::test_composition_preserves_singletons_declared_slots_and_actual_allocation` | 15 cases passed. |
| `tests/fdtd/test_fd_oscillator_stack.py::test_all_members_storage_and_closed_yee_dissipation_use_independent_oscillators` | 30 cases passed. |
| `tests/fdtd/test_fd_oscillator_stack.py::test_oblique_oscillator_stack_matches_declared_multilayer_on_each_backend` | 6 cases passed. |
| `tests/fdtd/test_fd_oscillator_stack.py::test_every_member_must_pass_scope_and_exact_padded_support_separation` | 27 cases passed. |
| `tests/fdtd/test_fd_oscillator_stack.py::test_prepared_stack_binds_all_members_memories_and_source_allocation` | 23 cases passed. |
| `tests/fdtd/test_fd_oscillator_stack.py::test_traced_later_poles_geometry_and_host_cannot_detach_into_stack_admission` | 18 cases passed. |
| `tests/fdtd/test_fd_oscillator_stack.py::test_cfl_uses_each_complete_oscillator_rate_then_the_spatial_maximum` | 5 cases passed. |
| `tests/fdtd/test_fd_oscillator_stack.py::test_spectral_certificate_includes_later_member_interior_resonances_and_samples` | 10 cases passed. |

## Disclosed calibration and execution environment

All 6 calibration rows completed with finite positive energy
and stable diagnostics. The maximum absolute complex-amplitude errors were
0.00287460301 in reflection and
0.00465698766 in transmission. Post-disclosure
known-case ceilings are 0.005 and
0.01, respectively. They are not relative power
errors, an unseen accuracy gate or a campaign-wide maximum-error requirement.
Matching coarse campaign rows are known repeats, not independent corroboration.
All earlier development, calibration and mutation exposures remain in registration.

Execution platform: Windows-11-10.0.22631-SP0. Python version:
3.12.14 | packaged by Anaconda, Inc. | (main, Aug 27 2026, 14:37:13) [MSC v.1942 64 bit (AMD64)]. Installed backend versions recorded
with the numerical gates: {"jax": "0.11.0", "jaxlib": "0.11.0", "numpy": "2.5.1", "pytest": "9.1.1", "pytest-xdist": "3.8.0", "torch": "2.13.0"}.
No hardware speed comparison is inferred from these runs.

## Measured results

The complete registered interval determines each classification: second means
the entire interval lies in [1.8, 2.3], low means its upper endpoint is at most
1.75, and mixed retains every other resolved interval. Unresolved includes the
fixed noise/vanishing-error rule. The middle-grid duration/absorber proxy is
empirical sensitivity, not a rigorous fine-grid error bound or an asymptotic
order theorem. No requirement that every cell be second is imposed.

| Comparison | Second | Low | Mixed | Unresolved |
| --- | ---: | ---: | ---: | ---: |
| Self-order (40/80/160) | 47 | 0 | 1 | 0 |
| Direct order (40 to 80) | 48 | 0 | 0 | 0 |
| Direct order (80 to 160) | 43 | 0 | 0 | 5 |

Maximum absolute complex-amplitude errors over every registered cell:

| Coefficient | 40 | 80 | 160 |
| --- | ---: | ---: | ---: |
| r | 0.0057695709 | 0.00144217553 | 0.00035967287 |
| t | 0.00465698766 | 0.00116631524 | 0.000292278424 |

Each row retains its point estimate, full interval when resolved, and verdict.
Displayed values are rounded; verdicts use the unrounded saved values.
Self-order uses successive complex-response differences; direct order uses
continuum errors.

| Family | Placement | Pol. | Wavelength | Coefficient | Self-order | Direct 40 to 80 | Direct 80 to 160 |
| --- | ---: | --- | ---: | --- | --- | --- | --- |
| mixed-pair | 0.3 | s | 0.9 | r | 1.8862 [1.8862, 1.8863]; second | 1.9031 [1.9031, 1.9031]; second | 1.9525 [1.9524, 1.9525]; second |
| mixed-pair | 0.3 | s | 1.0 | r | 1.8292 [1.8291, 1.8293]; second | 1.8572 [1.8572, 1.8572]; second | 1.9334 [1.9332, 1.9335]; second |
| mixed-pair | 0.3 | s | 1.1 | r | 1.7723 [1.7721, 1.7726]; mixed | 1.8108 [1.8107, 1.8108]; second | 1.9128 [1.9125, 1.9131]; second |
| mixed-pair | 0.3 | s | 0.9 | t | 2.0530 [2.0530, 2.0530]; second | 2.0450 [2.0450, 2.0450]; second | 2.0208 [2.0208, 2.0208]; second |
| mixed-pair | 0.3 | s | 1.0 | t | 2.1221 [2.1220, 2.1221]; second | 2.1060 [2.1059, 2.1060]; second | 2.0544 [2.0543, 2.0545]; second |
| mixed-pair | 0.3 | s | 1.1 | t | 2.2346 [2.2343, 2.2349]; second | 2.2081 [2.2080, 2.2083]; second | 2.1177 [2.1171, 2.1182]; second |
| mixed-pair | 0.3 | p | 0.9 | r | 1.9710 [1.9373, 2.0051]; second | 1.9774 [1.9648, 1.9901]; second | 1.9985 [1.9484, 2.0496]; second |
| mixed-pair | 0.3 | p | 1.0 | r | 1.9506 [1.9451, 1.9562]; second | 1.9579 [1.9558, 1.9600]; second | 1.9798 [1.9716, 1.9880]; second |
| mixed-pair | 0.3 | p | 1.1 | r | 1.9249 [1.9146, 1.9351]; second | 1.9378 [1.9340, 1.9416]; second | 1.9752 [1.9603, 1.9901]; second |
| mixed-pair | 0.3 | p | 0.9 | t | 2.0465 [2.0372, 2.0560]; second | 2.0385 [2.0350, 2.0421]; second | 2.0152 [2.0009, 2.0296]; second |
| mixed-pair | 0.3 | p | 1.0 | t | 2.1211 [2.1056, 2.1367]; second | 2.1061 [2.1002, 2.1121]; second | 2.0619 [2.0371, 2.0870]; second |
| mixed-pair | 0.3 | p | 1.1 | t | 2.1155 [2.0218, 2.2132]; second | 2.1101 [2.0736, 2.1472]; second | 2.1078; unresolved |
| mixed-pair | 0.1 | s | 0.9 | r | 1.9982 [1.9982, 1.9983]; second | 1.9979 [1.9979, 1.9979]; second | 1.9970 [1.9970, 1.9970]; second |
| mixed-pair | 0.1 | s | 1.0 | r | 1.9795 [1.9795, 1.9795]; second | 1.9826 [1.9826, 1.9826]; second | 1.9919 [1.9918, 1.9919]; second |
| mixed-pair | 0.1 | s | 1.1 | r | 1.9670 [1.9670, 1.9671]; second | 1.9722 [1.9722, 1.9723]; second | 1.9876 [1.9875, 1.9877]; second |
| mixed-pair | 0.1 | s | 0.9 | t | 2.0313 [2.0313, 2.0313]; second | 2.0257 [2.0257, 2.0257]; second | 2.0090 [2.0090, 2.0090]; second |
| mixed-pair | 0.1 | s | 1.0 | t | 2.0264 [2.0264, 2.0265]; second | 2.0217 [2.0217, 2.0217]; second | 2.0083 [2.0082, 2.0084]; second |
| mixed-pair | 0.1 | s | 1.1 | t | 1.9816 [1.9815, 1.9817]; second | 1.9843 [1.9843, 1.9844]; second | 1.9931 [1.9929, 1.9932]; second |
| mixed-pair | 0.1 | p | 0.9 | r | 1.9992 [1.9821, 2.0163]; second | 2.0002 [1.9938, 2.0066]; second | 2.0035 [1.9779, 2.0293]; second |
| mixed-pair | 0.1 | p | 1.0 | r | 1.9884 [1.9855, 1.9914]; second | 1.9904 [1.9893, 1.9915]; second | 1.9963 [1.9919, 2.0007]; second |
| mixed-pair | 0.1 | p | 1.1 | r | 1.9781 [1.9726, 1.9837]; second | 1.9824 [1.9804, 1.9845]; second | 1.9952 [1.9870, 2.0035]; second |
| mixed-pair | 0.1 | p | 0.9 | t | 2.0201 [2.0135, 2.0268]; second | 2.0158 [2.0133, 2.0183]; second | 2.0031 [1.9931, 2.0131]; second |
| mixed-pair | 0.1 | p | 1.0 | t | 1.9926 [1.9849, 2.0003]; second | 1.9936 [1.9907, 1.9964]; second | 1.9973 [1.9858, 2.0088]; second |
| mixed-pair | 0.1 | p | 1.1 | t | 1.9760 [1.9548, 1.9973]; second | 1.9803 [1.9724, 1.9883]; second | 1.9932 [1.9617, 2.0250]; second |
| lorentz-triple | 0.3 | s | 0.9 | r | 1.9217 [1.9215, 1.9219]; second | 1.9324 [1.9324, 1.9325]; second | 1.9645 [1.9642, 1.9648]; second |
| lorentz-triple | 0.3 | s | 1.0 | r | 1.9451 [1.9449, 1.9453]; second | 1.9529 [1.9529, 1.9530]; second | 1.9761 [1.9758, 1.9764]; second |
| lorentz-triple | 0.3 | s | 1.1 | r | 1.9731 [1.9726, 1.9737]; second | 1.9768 [1.9766, 1.9770]; second | 1.9879 [1.9871, 1.9887]; second |
| lorentz-triple | 0.3 | s | 0.9 | t | 2.0024 [2.0024, 2.0025]; second | 2.0017 [2.0016, 2.0017]; second | 1.9993 [1.9992, 1.9994]; second |
| lorentz-triple | 0.3 | s | 1.0 | t | 2.0009 [2.0008, 2.0010]; second | 2.0004 [2.0004, 2.0004]; second | 1.9989 [1.9988, 1.9990]; second |
| lorentz-triple | 0.3 | s | 1.1 | t | 2.0013 [2.0011, 2.0016]; second | 2.0008 [2.0007, 2.0009]; second | 1.9991 [1.9987, 1.9995]; second |
| lorentz-triple | 0.3 | p | 0.9 | r | 1.9348 [1.8540, 2.0184]; second | 1.9359 [1.9057, 1.9665]; second | 1.9438; unresolved |
| lorentz-triple | 0.3 | p | 1.0 | r | 1.9258 [1.9044, 1.9473]; second | 1.9359 [1.9280, 1.9439]; second | 1.9657 [1.9349, 1.9968]; second |
| lorentz-triple | 0.3 | p | 1.1 | r | 2.0122 [1.9302, 2.0971]; second | 2.0160 [1.9847, 2.0477]; second | 2.0286; unresolved |
| lorentz-triple | 0.3 | p | 0.9 | t | 1.9977 [1.9909, 2.0046]; second | 1.9974 [1.9949, 2.0000]; second | 1.9965 [1.9863, 2.0068]; second |
| lorentz-triple | 0.3 | p | 1.0 | t | 1.9956 [1.9902, 2.0011]; second | 1.9972 [1.9952, 1.9992]; second | 2.0020 [1.9938, 2.0101]; second |
| lorentz-triple | 0.3 | p | 1.1 | t | 1.9955 [1.9710, 2.0203]; second | 1.9989 [1.9897, 2.0082]; second | 2.0091 [1.9722, 2.0466]; second |
| lorentz-triple | 0.1 | s | 0.9 | r | 1.9532 [1.9529, 1.9535]; second | 1.9587 [1.9586, 1.9588]; second | 1.9758 [1.9754, 1.9762]; second |
| lorentz-triple | 0.1 | s | 1.0 | r | 1.9574 [1.9570, 1.9579]; second | 1.9624 [1.9622, 1.9626]; second | 1.9781 [1.9774, 1.9787]; second |
| lorentz-triple | 0.1 | s | 1.1 | r | 1.9826 [1.9820, 1.9833]; second | 1.9849 [1.9847, 1.9852]; second | 1.9921 [1.9911, 1.9930]; second |
| lorentz-triple | 0.1 | s | 0.9 | t | 2.0055 [2.0054, 2.0056]; second | 2.0043 [2.0043, 2.0043]; second | 2.0006 [2.0005, 2.0007]; second |
| lorentz-triple | 0.1 | s | 1.0 | t | 1.9989 [1.9988, 1.9991]; second | 1.9989 [1.9988, 1.9989]; second | 1.9987 [1.9985, 1.9989]; second |
| lorentz-triple | 0.1 | s | 1.1 | t | 1.9998 [1.9993, 2.0002]; second | 1.9996 [1.9994, 1.9998]; second | 1.9991 [1.9984, 1.9997]; second |
| lorentz-triple | 0.1 | p | 0.9 | r | 1.9815 [1.8959, 2.0703]; second | 1.9794 [1.9471, 2.0122]; second | 1.9751; unresolved |
| lorentz-triple | 0.1 | p | 1.0 | r | 1.9723 [1.9312, 2.0141]; second | 1.9631 [1.9478, 1.9785]; second | 1.9402 [1.8818, 2.0000]; second |
| lorentz-triple | 0.1 | p | 1.1 | r | 2.0044 [1.9217, 2.0899]; second | 2.0142 [1.9826, 2.0462]; second | 2.0448; unresolved |
| lorentz-triple | 0.1 | p | 0.9 | t | 2.0004 [1.9920, 2.0089]; second | 1.9995 [1.9964, 2.0027]; second | 1.9969 [1.9843, 2.0095]; second |
| lorentz-triple | 0.1 | p | 1.0 | t | 1.9959 [1.9890, 2.0029]; second | 1.9979 [1.9953, 2.0005]; second | 2.0039 [1.9934, 2.0144]; second |
| lorentz-triple | 0.1 | p | 1.1 | t | 1.9962 [1.9662, 2.0267]; second | 2.0000 [1.9887, 2.0114]; second | 2.0114 [1.9661, 2.0577]; second |

Warnings retained across completed workers: AbsorberProximityWarning: 20, SubpixelAccuracyWarning: 40.
Selected base-rung durations (counting each group/rung selection): 120 periods: 24.
The largest middle-grid sensitivity proxy is 5.13136627e-06.

## Preserved evidence

- [Disclosed calibration and raw backend rows](https://github.com/fieldalgo/wavebit/blob/develop/studies/fdtd/oscillator_stack/records/fdtd_oscillator_stack_calibration_attempt1.json).
- [Accepted known-case ceilings and preserved test transition](https://github.com/fieldalgo/wavebit/blob/develop/studies/fdtd/oscillator_stack/records/fdtd_oscillator_stack_calibration_acceptance.json).
- [Exact new and related test rosters](https://github.com/fieldalgo/wavebit/blob/develop/studies/fdtd/oscillator_stack/records/fdtd_oscillator_stack_targeted.json).
- [Actual source witnesses and exact restored baseline](https://github.com/fieldalgo/wavebit/blob/develop/studies/fdtd/oscillator_stack/records/fdtd_oscillator_stack_mutations.json).
- [Registered protocol, source identities and prior exposures](https://github.com/fieldalgo/wavebit/blob/develop/studies/fdtd/oscillator_stack/records/fdtd_oscillator_stack_registration.json).
- [Every physical cell, verdict and selected duration](https://github.com/fieldalgo/wavebit/blob/develop/studies/fdtd/oscillator_stack/records/fdtd_oscillator_stack_report.json).
- [Full/slow regressions and all four frozen-reference comparisons](https://github.com/fieldalgo/wavebit/blob/develop/studies/fdtd/oscillator_stack/records/fdtd_oscillator_stack_validation.json).

The final auditor separately verifies saved source, raw process and numerical
records after documentation validation. This publication replay does not claim
those future exits, universal convergence, arbitrary-stack support, general C8
completion or a passed C7 verdict.
