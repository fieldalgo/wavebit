# C8: separated Debye films

Written: 2026-10-09

Targets: `src/wavebit/fdtd/_prepare/debye_stack.py`, `src/wavebit/fdtd/_prepare/plan.py`, `src/wavebit/fdtd/_prepare/params.py`, `src/wavebit/fdtd/_prepare/coefficients.py`, `src/wavebit/fdtd/_prepare/checks.py`.

## Current status

The bounded implementation's numerical gates are complete. All 147 new
cases and 1314 related cases pass. Thirteen actual production-source
mutations fail their intended assertions, then exact restoration passes all
147 new cases. The full FDTD non-slow tier passes
4324 cases with 8 existing skips, the
separate slow tier passes 81, and all 1141 frozen arrays
reproduce bit for bit.

The registered study completes 40 workers and retains all 48
complex cells. Its fixed sensitivity-interval criterion classifies
42/48 self-orders and 48/48 final direct orders as second.
The table below retains every other outcome. These measured fixtures do not
establish second-order convergence for every admitted stack.

The new route composes two or more parallel Debye films whose padded grid
supports are disjoint. Each member retains its existing one-, two- or
finite-sum Debye implementation. General C8 smoothing and the
[C7 metasurface verdict](fd-metasurface-validation.md) remain open.

## Bounded public contract

Use the ordinary Scene structures tuple with Box geometry and Dispersive
materials whose poles are all exact Debye declarations. There is no new
public stack constructor.

| Item | Admitted case |
| --- | --- |
| Members | At least two exact Dispersive materials, each containing one or more exact passive Debye declarations. Different declared counts may share a scene. |
| Geometry | Concrete unturned, unranged voxel Box films, all with the same normal axis. |
| Extent | Each film spans the two transverse axes with the existing one-cell overhang, has normal thickness at least dx, and clears domain walls and all absorbing layers. |
| Separation | Actual padded half-open support intervals are pairwise disjoint; physical non-contact alone does not establish admission. |
| Material | Finite positive isotropic high-frequency permittivity, finite nonnegative increments, finite positive relaxation times, and mu=1. |
| Host | One exact plain positive isotropic lossless nonmagnetic Material. |
| Runtime | Fixed duration, no initial field, no traced geometry/material/band values. The new route refuses Decay. |
| Plan | The ordered member signature, declared slot counts and boundary declarations must match. Missing, reordered, duplicate or foreign memory terms are refused. |

Zero-strength and exactly equal-time declarations retain their reserved slots.
The geometry and separation restrictions also apply to wholly inactive members.
Original structure indices appear in diagnostics even when spatial order differs
from declaration order. Existing source and monitor contracts remain in force;
in particular this does not extend the layered-background scope of TFSF.

Curved lossy bodies, touching films, junctions, lossy/dispersive hosts, stacks
mixing Debye and other material families, and differentiation are outside this
extension. Each singleton reducer retains its finite-range and work-budget
refusals.

## Interface law and composition

Use the repository convention exp(-i*omega*t) and let s=-i*omega.
For one member, write epsilon(s)=a+chi(s),
chi(s)=sum(delta_i/(1+tau_i*s)), host permittivity b, film fraction f
and host fraction g=1-f. Then

~~~text
A = f*b + g*a
alpha = g/A
kappa = f*b*b/A**2
epsilon_t(s) = f*(a+chi(s)) + g*b
epsilon_n(s) = a*b/A + kappa*chi(s)/(1+alpha*chi(s)).
~~~

The harmonic normal law transforms the entire susceptibility sum. Existing
[single-Debye](fd-debye-film.md), [two-Debye](fd-two-debye-film.md) and
[finite-sum](fd-debye-sum-film.md) reducers supply its passive memories unchanged.
The selected reducer follows the declared count, not the number of active modes.
There is no approximate merging or independent single-pole approximation of a
multi-pole normal law.

Each electric component retains its singleton Yee-cell fraction. Because padded
supports are disjoint, the composer can paste each complete singleton update
profile into the host and reuse its cropped pole parameters. No E node receives
two material updates. The magnetic medium remains uniform and nonmagnetic.

The existing split Debye recurrence and storage observer are reused. Material
storage corrections add over disjoint supports while global electric/magnetic
storage is counted once. Each declared pole retains three histories with three
components on its own cropped region. The actual allocator, rather than a
nominal slot count alone, must confirm the state-byte report.

## Stability and bulk spectral checks

Let M1=sum(delta_i/tau_i) and M2=sum(delta_i/tau_i**2). The independent
local material-rate identities are

~~~text
bulk:       M2/a
tangential: f*M2/(f*a+g*b)
normal:     kappa*(M2+alpha*M1**2)/(a*b/A).
~~~

The implementation reuses each singleton policy, including its rounded
effective coefficients, then takes the maximum across members. Rates of
different poles within one member are summed before the spatial maximum.
Separated member maxima are not summed. The existing physical high-frequency
permittivity supplies the curl bound. This retains the conservative repository
policy; it is not a new necessary stability condition or PML theorem.

Every member and the host is checked at all requested monitor, Resolution and
explicit PML wavelengths and over every source band. For passive Debye sums,
put x=omega**2. The identity

~~~text
|epsilon|**2 = a**2 + sum(C_i/(1+tau_i**2*x))
C_i = delta_i**2 + 2*a*delta_i
      + sum(j != i, 2*delta_i*delta_j*tau_i/(tau_i+tau_j))
~~~

has nonnegative C_i. Each x**2/(1+t*x) increases for positive x, so the
bulk |k| is largest at the upper source-band endpoint. Endpoint arithmetic
uses exact rationals for the represented public floating-point constants.
Two square roots, each verified upward, bound the resulting fourth root.
Nonrepresentable intermediate values are refused explicitly; finite inputs
alone do not promise admission at every extreme scale.

The inherited phase/skin hard floors are 8/1 cells; warning floors are 12/2.
No requested spectral data means no inferred bulk certificate. These checks
do not establish interface accuracy, stack resonance resolution, adequate
duration, absorber quality or a universal convergence order.

## Independent verification and registered physical study

The tests cover all normal axes, declared counts 1/2/N, independent rational
series/parallel responses through actual recurrences, independently computed
cell fills, storage and closed-Yee loss, actual state allocation, singleton
equivalence, every-member CFL/spectral checks and complete scope/plan refusals.
Actual production-source mutations must trigger the intended assertions before
byte-exact restoration.

The fixed physical instrument compares complex reflection and transmission
against the existing continuum multilayer characteristic matrix. Declared
Debye permittivities, every host gap and the outer-face reference phase are
explicit. Independent backward Fresnel recursion and complex single-film Airy
controls check the reference before field measurements.

Two asymmetric fixtures contain respectively 1+2 and 3+4+1 declared poles;
the latter includes zero-strength declarations and a wholly inactive member.
The disclosed calibration uses the two coarse p-polarized configurations on
NumPy, JAX and Torch. All six rows complete with finite positive energy and
stable diagnostics. The maximum absolute complex-amplitude errors are
0.00413897832 in reflection and
0.00792802502 in transmission.
After disclosure and an independent scalar-reference replay, the known-case
regression ceilings were set to 0.005 and
0.01. They are not relative power errors or
unseen accuracy gates. Matching coarse campaign rows are known repeats,
not independent corroboration. Before/after tests and the sole
docstring/ceiling transition remain in the acceptance record.

The separate 40-worker study fixes wavelengths 0.9/1.0/1.1, central angle 35
degrees with fixed transverse wavevector, s/p polarizations, placements 0.3/0.1
and resolutions 40/80/160. Each placement/polarization includes the two
middle-grid controls: 240 periods and 48 PML cells. The base run uses 120
periods and 24 PML cells. Only the preregistered unsettled-duration replacement
is allowed; no result-selected geometry, wavelength or extra rung is permitted.

All 48 complex cells, 48 self-orders and 96 direct pairs remain visible.
The inherited duration/PML sensitivity intervals may leave mixed, low or
unresolved cases. Completion of the instrument is distinct from a
second-order verdict.

Production integration additionally requires the strict documentation build
and independent saved-evidence audit, with their actual parent process exits.
The numerical counts above are taken from completed raw records. Existing
evidence and reference arrays remain immutable.

## Measured results

The classifications use the complete registered sensitivity interval: second
means the entire interval lies in [1.8, 2.3]. A middle-grid duration/PML
proxy supplies empirical sensitivity, not a rigorous numerical error bound.
Low, mixed and unresolved outcomes are retained when present.

| Comparison | Second | Low | Mixed | Unresolved |
| --- | ---: | ---: | ---: | ---: |
| Self-order (40/80/160) | 42 | 5 | 1 | 0 |
| Direct order (40 to 80) | 43 | 3 | 2 | 0 |
| Direct order (80 to 160) | 48 | 0 | 0 | 0 |

Maximum absolute complex-amplitude errors across every family, placement,
polarization and wavelength in the registered ladder:

| Coefficient | 40 | 80 | 160 |
| --- | ---: | ---: | ---: |
| r | 0.0077602694 | 0.0019720314 | 0.00050025416 |
| t | 0.007928025 | 0.0019937653 | 0.00050018356 |

Each cell below retains its point estimate, full sensitivity interval when
resolved, and classification. Direct columns compare continuum errors;
self-order compares successive complex-response differences. Displayed estimates
and intervals are rounded; classifications use the unrounded registered values.

| Family | Placement | Pol. | Wavelength | Coefficient | Self-order | Direct 40 to 80 | Direct 80 to 160 |
| --- | ---: | --- | ---: | --- | --- | --- | --- |
| debye-pair | 0.3 | s | 0.9 | r | 1.6980 [1.6980, 1.6981]; low | 1.7514 [1.7514, 1.7514]; mixed | 1.8878 [1.8877, 1.8878]; second |
| debye-pair | 0.3 | s | 1.0 | r | 1.6958 [1.6958, 1.6958]; low | 1.7500 [1.7500, 1.7500]; low | 1.8868 [1.8867, 1.8868]; second |
| debye-pair | 0.3 | s | 1.1 | r | 1.7439 [1.7438, 1.7440]; low | 1.7868 [1.7867, 1.7868]; mixed | 1.8996 [1.8995, 1.8997]; second |
| debye-pair | 0.3 | s | 0.9 | t | 2.0145 [2.0144, 2.0145]; second | 2.0114 [2.0114, 2.0114]; second | 2.0025 [2.0025, 2.0025]; second |
| debye-pair | 0.3 | s | 1.0 | t | 2.0145 [2.0145, 2.0145]; second | 2.0116 [2.0116, 2.0116]; second | 2.0029 [2.0029, 2.0029]; second |
| debye-pair | 0.3 | s | 1.1 | t | 2.0168 [2.0167, 2.0168]; second | 2.0135 [2.0135, 2.0136]; second | 2.0039 [2.0039, 2.0040]; second |
| debye-pair | 0.3 | p | 0.9 | r | 1.6271 [1.6193, 1.6349]; low | 1.6968 [1.6940, 1.6997]; low | 1.8675 [1.8576, 1.8775]; second |
| debye-pair | 0.3 | p | 1.0 | r | 1.6659 [1.6611, 1.6706]; low | 1.7257 [1.7240, 1.7274]; low | 1.8773 [1.8711, 1.8834]; second |
| debye-pair | 0.3 | p | 1.1 | r | 1.8114 [1.7842, 1.8388]; mixed | 1.8403 [1.8303, 1.8504]; second | 1.9272 [1.8899, 1.9651]; second |
| debye-pair | 0.3 | p | 0.9 | t | 1.9995 [1.9954, 2.0036]; second | 1.9984 [1.9969, 2.0000]; second | 1.9953 [1.9892, 2.0014]; second |
| debye-pair | 0.3 | p | 1.0 | t | 1.9935 [1.9922, 1.9948]; second | 1.9940 [1.9935, 1.9945]; second | 1.9958 [1.9938, 1.9978]; second |
| debye-pair | 0.3 | p | 1.1 | t | 1.9876 [1.9826, 1.9926]; second | 1.9890 [1.9871, 1.9909]; second | 1.9932 [1.9857, 2.0007]; second |
| debye-pair | 0.1 | s | 0.9 | r | 1.9758 [1.9758, 1.9758]; second | 1.9760 [1.9760, 1.9760]; second | 1.9797 [1.9797, 1.9797]; second |
| debye-pair | 0.1 | s | 1.0 | r | 1.9940 [1.9940, 1.9940]; second | 1.9912 [1.9912, 1.9912]; second | 1.9867 [1.9867, 1.9868]; second |
| debye-pair | 0.1 | s | 1.1 | r | 2.0263 [2.0263, 2.0264]; second | 2.0195 [2.0195, 2.0195]; second | 2.0020 [2.0019, 2.0021]; second |
| debye-pair | 0.1 | s | 0.9 | t | 2.0264 [2.0264, 2.0264]; second | 2.0211 [2.0211, 2.0211]; second | 2.0050 [2.0050, 2.0050]; second |
| debye-pair | 0.1 | s | 1.0 | t | 2.0214 [2.0213, 2.0214]; second | 2.0170 [2.0170, 2.0170]; second | 2.0040 [2.0040, 2.0040]; second |
| debye-pair | 0.1 | s | 1.1 | t | 2.0166 [2.0166, 2.0166]; second | 2.0132 [2.0132, 2.0132]; second | 2.0030 [2.0030, 2.0031]; second |
| debye-pair | 0.1 | p | 0.9 | r | 1.9768 [1.9725, 1.9811]; second | 1.9764 [1.9748, 1.9780]; second | 1.9789 [1.9727, 1.9852]; second |
| debye-pair | 0.1 | p | 1.0 | r | 1.9829 [1.9801, 1.9858]; second | 1.9810 [1.9800, 1.9821]; second | 1.9799 [1.9757, 1.9840]; second |
| debye-pair | 0.1 | p | 1.1 | r | 1.9944 [1.9720, 2.0170]; second | 1.9879 [1.9796, 1.9964]; second | 1.9745 [1.9417, 2.0078]; second |
| debye-pair | 0.1 | p | 0.9 | t | 2.0176 [2.0141, 2.0212]; second | 2.0133 [2.0120, 2.0147]; second | 2.0004 [1.9951, 2.0058]; second |
| debye-pair | 0.1 | p | 1.0 | t | 2.0112 [2.0098, 2.0126]; second | 2.0088 [2.0083, 2.0093]; second | 2.0015 [1.9994, 2.0036]; second |
| debye-pair | 0.1 | p | 1.1 | t | 2.0011 [1.9941, 2.0080]; second | 2.0008 [1.9982, 2.0034]; second | 2.0002 [1.9897, 2.0106]; second |
| debye-triple | 0.3 | s | 0.9 | r | 1.8361 [1.8360, 1.8361]; second | 1.8601 [1.8601, 1.8601]; second | 1.9287 [1.9286, 1.9287]; second |
| debye-triple | 0.3 | s | 1.0 | r | 1.9857 [1.9857, 1.9858]; second | 1.9842 [1.9842, 1.9843]; second | 1.9841 [1.9840, 1.9842]; second |
| debye-triple | 0.3 | s | 1.1 | r | 2.1039 [2.1038, 2.1040]; second | 2.0893 [2.0892, 2.0893]; second | 2.0429 [2.0426, 2.0431]; second |
| debye-triple | 0.3 | s | 0.9 | t | 2.0042 [2.0041, 2.0042]; second | 2.0026 [2.0026, 2.0026]; second | 1.9981 [1.9981, 1.9982]; second |
| debye-triple | 0.3 | s | 1.0 | t | 2.0082 [2.0081, 2.0082]; second | 2.0062 [2.0062, 2.0062]; second | 2.0004 [2.0004, 2.0005]; second |
| debye-triple | 0.3 | s | 1.1 | t | 2.0131 [2.0131, 2.0132]; second | 2.0105 [2.0105, 2.0106]; second | 2.0029 [2.0028, 2.0030]; second |
| debye-triple | 0.3 | p | 0.9 | r | 1.8868 [1.8791, 1.8945]; second | 1.9022 [1.8994, 1.9051]; second | 1.9497 [1.9388, 1.9607]; second |
| debye-triple | 0.3 | p | 1.0 | r | 2.0333 [2.0284, 2.0382]; second | 2.0280 [2.0262, 2.0299]; second | 2.0136 [2.0062, 2.0211]; second |
| debye-triple | 0.3 | p | 1.1 | r | 2.1193 [2.0844, 2.1547]; second | 2.1046 [2.0912, 2.1180]; second | 2.0569 [2.0015, 2.1136]; second |
| debye-triple | 0.3 | p | 0.9 | t | 1.9913 [1.9879, 1.9947]; second | 1.9917 [1.9904, 1.9930]; second | 1.9932 [1.9881, 1.9983]; second |
| debye-triple | 0.3 | p | 1.0 | t | 1.9903 [1.9888, 1.9918]; second | 1.9915 [1.9909, 1.9920]; second | 1.9950 [1.9928, 1.9972]; second |
| debye-triple | 0.3 | p | 1.1 | t | 1.9879 [1.9829, 1.9929]; second | 1.9896 [1.9877, 1.9915]; second | 1.9946 [1.9872, 2.0020]; second |
| debye-triple | 0.1 | s | 0.9 | r | 2.0138 [2.0137, 2.0139]; second | 2.0060 [2.0060, 2.0061]; second | 1.9904 [1.9903, 1.9905]; second |
| debye-triple | 0.1 | s | 1.0 | r | 2.0579 [2.0579, 2.0580]; second | 2.0478 [2.0477, 2.0478]; second | 2.0180 [2.0179, 2.0181]; second |
| debye-triple | 0.1 | s | 1.1 | r | 2.0289 [2.0288, 2.0290]; second | 2.0236 [2.0236, 2.0237]; second | 2.0083 [2.0081, 2.0085]; second |
| debye-triple | 0.1 | s | 0.9 | t | 2.0235 [2.0234, 2.0235]; second | 2.0186 [2.0186, 2.0186]; second | 2.0039 [2.0039, 2.0040]; second |
| debye-triple | 0.1 | s | 1.0 | t | 2.0093 [2.0093, 2.0093]; second | 2.0070 [2.0070, 2.0070]; second | 2.0000 [1.9999, 2.0001]; second |
| debye-triple | 0.1 | s | 1.1 | t | 1.9963 [1.9962, 1.9964]; second | 1.9963 [1.9962, 1.9963]; second | 1.9962 [1.9960, 1.9964]; second |
| debye-triple | 0.1 | p | 0.9 | r | 1.9650 [1.9510, 1.9791]; second | 1.9630 [1.9578, 1.9682]; second | 1.9681 [1.9477, 1.9887]; second |
| debye-triple | 0.1 | p | 1.0 | r | 2.1775 [2.1630, 2.1921]; second | 2.1490 [2.1434, 2.1545]; second | 2.0562 [2.0329, 2.0797]; second |
| debye-triple | 0.1 | p | 1.1 | r | 2.0487 [2.0194, 2.0783]; second | 2.0372 [2.0261, 2.0482]; second | 2.0022 [1.9580, 2.0471]; second |
| debye-triple | 0.1 | p | 0.9 | t | 2.0106 [2.0057, 2.0155]; second | 2.0074 [2.0055, 2.0092]; second | 1.9976 [1.9902, 2.0049]; second |
| debye-triple | 0.1 | p | 1.0 | t | 1.9901 [1.9873, 1.9929]; second | 1.9915 [1.9904, 1.9925]; second | 1.9955 [1.9914, 1.9997]; second |
| debye-triple | 0.1 | p | 1.1 | t | 1.9869 [1.9788, 1.9950]; second | 1.9892 [1.9862, 1.9922]; second | 1.9961 [1.9841, 2.0082]; second |

The first prevalidation attempt rejected equivalent Windows and POSIX report
path spellings. Its failure, all original helpers and all numerical records
remain preserved; the raw run material among them is archived outside the
repository. Archived prepare/auditor variants resolve that report path
inside the repository and bind their exact source changes and process arguments.
The numerical criteria and results are unchanged; the repair proof, the
`manifest.json` of the study's path repair (archived outside the repository),
records the original failure and both exact variant sources.

## Preserved evidence

- [Six disclosed backend measurements and all raw worker links](../../../studies/fdtd/debye_stack/records/fdtd_debye_stack_calibration_attempt1.json).
- [Post-disclosure optical ceilings and before/after test bytes](../../../studies/fdtd/debye_stack/records/fdtd_debye_stack_calibration_acceptance.json).
- [Exact collected/reported new and related test rosters](../../../studies/fdtd/debye_stack/records/fdtd_debye_stack_targeted.json).
- Thirteen actual source witnesses, restoration and final baseline: the run record
  `fdtd_debye_stack_mutations.json`, archived outside the repository.
- Fixed protocol, prior exposures and source fingerprints: the run record
  `fdtd_debye_stack_registration.json`, archived outside the repository.
- [Every registered physical cell and selected duration](../../../studies/fdtd/debye_stack/records/fdtd_debye_stack_report.json).
- Full/slow rosters and four read-only frozen-reference comparisons: the run record
  `fdtd_debye_stack_validation.json`, archived outside the repository.

The saved-data auditor independently replays the scalar continuum reference,
worker roster, every classification, raw pytest outcomes and source-mutation
bytes. A process cannot certify its own future exit; completed parent receipts
supply that final link. No Meep implementation or prohibited projection source
was used for this extension.
