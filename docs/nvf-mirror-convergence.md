# Faster optical convergence checks using exact mirror sectors

The new NumPy research path uses the rectangular fixture's y-reflection
symmetry to reduce the cost of its existing NVF energy closure and Maxwell
solve. At outer order 16 the measured run decreases from 13.10 to 3.77 seconds,
with p/s transmission differences below 1.4e-12. It enables higher-order
directional comparisons while retaining the previous material field and
factorization. It does not establish optical convergence or change production
NVF dispatch.

## Meaning and status of the convergence target

The `1e-5` absolute transmission-increment threshold used below is a
provisional precision-study target. The reviewed project records do not
establish it as a product accuracy requirement or justify making it a
mandatory gate for all NVF development. For incident-power-normalized T,
`1e-5` means 10 ppm of incident power, or 0.001 percentage points of
transmittance; it is not a fixed relative error in T.

Varying the retained Fourier order is a sound convergence diagnostic, as
described by [NIST's RCW_Model documentation](https://pages.nist.gov/SCATMECH/docs/rcwmodel.htm).
Separate x/y refinements and mixed-order corners help expose directional
cancellation, and independent auxiliary-cell refinements help distinguish
cell discretization from outer Fourier truncation. None of these finite
differences alone bounds the error relative to the infinite-order solution.
For example, the s-y 48-to-64 change of 1.48e-5 does not prove that the
order-64 result has error above 1e-5; its 56-to-64 change of 2.93e-6 also
does not prove error below 1e-5. A recent multi-interval window is useful
evidence of stability, but its spread is not a necessary or sufficient
condition for a specified true error.

Keep numerical integrity checks, accuracy qualification and feature
integration separate. The existing algebra, symmetry, finite-value and
lossless-energy checks retain their thresholds. Accuracy qualification must
state the intended observables, geometries and tolerances, and report the
evidence and unresolved uncertainty for those cases. Total T/R stability
does not qualify individual diffraction orders, near fields or derivatives.
The present 1e-5 diagnostic remains visible without silently relaxing it or
treating it as an established release requirement. Further high-order runs
need a stated accuracy question and bounded computational budget; crossing
this provisional threshold alone does not require indefinite refinement or
deferring otherwise justified implementation work.

Production already exposes the circular shape NVF described in
[the integration record](nvf-contact-followup.md#production-nvf-integration-and-correction-of-the-representation-premise).
The separate integration work mentioned here concerns the newer conforming
energy construction used by this research path.

## What is reduced

For an outer set `(X,Y)`, let H=(2X+1)(2Y+1). The previous transverse system
has 2H unknowns. At zero y Bloch wavevector and a y-mirror-symmetric material,
form orthonormal even and odd combinations of each pair of Fourier waves with
y indices n and -n. The n=0 waves appear only in the even space.

The p-incident sector has Ex even and Ey odd; its Hx is odd and Hy even.
The s-incident sector has Ex odd and Ey even, with the opposite magnetic
parities. Both sectors still contain coupled vector fields and all x orders.
Each sector has H unknowns. The largest dense matrix therefore has one quarter
as many entries, and the sectors are solved sequentially.

`nvf_mirror_convergence.py` constructs the scalar parity convolutions directly
from the original Fourier coefficients. It restricts both the field-change Q
and energy G before solving

```text
effective = Q^(-H) G Q^(-1).
```

This is the restriction of the same finite closure, not the cropping of a
larger inverse. It also inverts the longitudinal material convolution in the
appropriate scalar parity space. Multiplication by ky exchanges the even and
odd scalar spaces; kx stays within each. The two first-order Maxwell factors
are assembled in the electric and magnetic parity bases before forming the
smaller eigenproblem.

Exterior matching uses tangential electric field coordinates and sparse
outgoing H/E admittances. Transmission and reflection are evaluated through
the corresponding Poynting form, including the normalization of the incident
p or s field. This removes the need for a full 2H-by-2H exterior rotation or
inverse. The modal branch convention remains `mode_q`.

Large assembly intermediates are released after use, and LAPACK may overwrite
its Fortran-order eigenproblem input. Neither change adds regularization or
changes the physical equations. Progress records expose the closure, mode,
matching and completed-power stages for long calculations.

The later [memory reduction](nvf-energy-closure.md) evaluates the same energy
congruence with two solves, tiles Fourier assembly and builds its final blocks
in Fortran order. NumPy modal propagation also uses sparse ky exchange and
an algebraically reduced bottom-interface elimination. These changes preserve
the finite closure and physical acceptance criteria.

The subsequent [modal/interface memory change](nvf-mirror-modal-memory.md)
builds LR directly in LAPACK storage, applies R without storing it, and
eliminates the saved round-trip and electric-front matrices using row panels.
All modes participate. At order 32, measured full p-sector peak process memory
falls from 2.12 to 1.43 GB; T changes by 3.27e-13. The 76 passing controls
include direct coupled-interface solves and propagating-root sign invariance.

## Applicability checks

The input rows retain the convention

```text
Bxx, Bxy, Byx, Byy, (B.H epsilon B)xx/xy/yy, epsilon.
```

Their y-reflection signs are `+,-,-,+,+,-,+,+`. The code checks a relative
coefficient reflection defect no larger than 1e-10, real-field Fourier
conjugacy, finite values and adequate coefficient bandwidth. Non-symmetric
inputs are refused. The existing analytic rectangular research cell has a
reflection defect of 1.90e-11 at its highest measured resolution, attributable
to its numerical cell solve; measured agreement with the full solve checks
the effect at the optical level.

The existing unstructured fitted FEM exports have defects around 3e-8, and
are refused by this path. They have not been silently symmetrized or accepted
under a relaxed threshold. Their previous full-system measurements remain
independent auxiliary-field controls.

A subsequent [half-cell FEM reference](nvf-mirror-reference.md) instead
constructs even/odd corrector spaces with the physical mirror boundary
conditions. Its fitted mesh and paired half-cell quadrature have zero
reported reflection defect through cap 96. The original full-cell exports
and rejection threshold remain unchanged. The new reference agrees with
the former full-cell optical control within 7e-9 at order 16.

The default modal instrument is NumPy-only, uses the existing fixed one-layer optical
fixture and zero y Bloch incidence, and excludes near-grazing layer clusters
and exactly grazing exterior orders. The production coupled-cluster treatment
has not been replaced. The later explicit doubling path supports JAX and
layer graze under the same coefficient symmetry checks; it is documented in
[the combined implementation](nvf-mirror-doubling.md). Neither path inserts
its results into production NVF dispatch.

## Independent Cartesian control

`nvf_mirror_cartesian.py` separately accelerates the existing Cartesian
material-interval comparison. Within each x interval, the y material
convolution commutes with reflection. Its one-dimensional inversion therefore
splits into even and odd parts before x integration. The outer x-rule inverse
and the longitudinal material inverse are then taken in those same scalar
parity spaces.

The resulting matrices are the restrictions of `profile_li` at exactly the
same retained order. The x, y and longitudinal rules retain their original
order of operations. Centered Fourier phases reproduce the original interval
coordinate convention. This control shares the Maxwell matching instrument,
but does not use the NVF cell field or energy closure. It is not an NVF fallback
and is not an independent 3D Maxwell reference.

## Measurements

The physical fixture is unchanged: period 700 nm, width 350 nm, gap 17 nm,
seam 31 nm, relative permittivities 4/9, wavelength 900 nm, thickness 300 nm,
incidence .2 radians, top 1 and bottom 2.25. The NVF cell uses body degree 4,
gap degree 6, base rows 8, x subdivisions 3, edge fraction 1e-5 and grading
8/.01, for 7559 scalar cell unknowns.

The paired order-16 timing holds the saved field, retained order and process
environment settings fixed. Full p/s T is 0.907166395579 / 0.941482300435; the mirror
path differs by +6.41e-13 / -1.34e-12. The measured 3.47 ratio is a timing for
this fixture and run, not a universal speedup. NumPy/SciPy use Apple
Accelerate in this environment; the supplied OpenBLAS/OpenMP variables do
not establish its actual thread count. A separate order-24 control
reuses the preceding 5039-unknown field: the p/s differences from its recorded
full-system results are 7.44e-15 / 1.13e-13.

The higher-resolution cell gives:

| x half width | y half width | NVF p T | NVF s T |
| --- | --- | --- | --- |
| 24 | 24 | 0.906794640677 | 0.941569618541 |
| 32 | 24 | 0.906753615722 | 0.941585293399 |
| 24 | 32 | 0.906738240327 | 0.941519632772 |
| 32 | 32 | 0.906697183008 | 0.941534973306 |
| 40 | 32 | 0.906654312952 | 0.941543122710 |
| 32 | 40 | 0.906687195223 | 0.941516900659 |
| 40 | 40 | 0.906644280612 | 0.941524987696 |
| 48 | 40 | 0.906621686474 | 0.941528265861 |
| 40 | 48 | 0.906648336978 | 0.941534151033 |
| 48 | 48 | 0.906625743004 | 0.941537507320 |

The separate p increments are approximately -4.10e-5 in x and -5.64e-5 in y;
both remain material. For s they are +1.57e-5 and -5.00e-5. Thus directional
cancellation remains relevant at the higher order, and the square sequence
cannot certify convergence.

The four-corner comparison from `(32,32)` to `(40,40)` confirms that this is
still relevant at order 40:

| increment | p T change | s T change |
| --- | ---: | ---: |
| x only | -4.28700557e-5 | +8.14940349e-6 |
| y only | -9.98778483e-6 | -1.80726467e-5 |
| both | -5.29023963e-5 | -9.98561021e-6 |
| mixed: both minus x minus y | -4.45556950e-8 | -6.23669852e-8 |

The s square increment is slightly below 1e-5, but its y-only increment is
still larger than 1e-5 and has the opposite sign from the x increment. The
p square increment remains about 5.3e-5. Neither polarization is declared
converged. Lossless residuals across this table stay below 1.7e-11; that is a
matching check, not a truncation-error estimate.

The new square order-48 result changes p/s transmission by -1.85376078e-5 /
+1.25196239e-5 from the recorded order 40. Both magnitudes exceed 1e-5, and
the s increment reverses direction again. This run used the same trial space
in local cell coordinates at cap 96. A repeat at order 40 with that export
and the final reduced-memory algebra gives 0.906644280607 / 0.941524987694,
within 4.7e-12 of the preceding table. The coordinate/storage changes are
therefore much smaller than the observed outer-order drift.

The four corners using that same cap-96 export give:

| increment from (40,40) | p T change | s T change |
| --- | ---: | ---: |
| x only to (48,40) | -2.25941328e-5 | +3.27816685e-6 |
| y only to (40,48) | +4.05637093e-6 | +9.16333860e-6 |
| both to (48,48) | -1.85376031e-5 | +1.25196257e-5 |
| mixed: both minus x minus y | +1.58717928e-10 | +7.81202212e-8 |

For p, the two directional increments partially cancel. For s, each separate
increment is below 1e-5, but their combined change exceeds it. Both individual
directions and the combined change must therefore be inspected. Neither
polarization passes this finite increment check, and even a pass would not
constitute a rigorous optical error bound. The largest power-balance residual
in these four corners is 2.35e-11.

The final reduced-memory implementation repeats order 40 and the two
directional order-48 cases in one process, whose peak resident memory is
5.18 GB. The order-48 square run preceded the later sparse modal/entrance
storage changes, while using the same cell export and equations. The full
system, JAX and order-40 comparisons test these algebraic changes. Raw records
are in `nvf_mirror_order48.log`, `nvf_mirror_order48_directional.log` and
`nvf_mirror_order48_summary.log` under `studies/results/`.

The separately factorized Cartesian control gives:

| square half width | Cartesian p T | Cartesian s T | NVF minus Cartesian p | NVF minus Cartesian s |
| --- | ---: | ---: | ---: | ---: |
| 24 | 0.906805437055 | 0.941572093232 | -1.07963777e-5 | -2.47469166e-6 |
| 32 | 0.906699960647 | 0.941538496827 | -2.77763905e-6 | -3.52352100e-6 |
| 40 | 0.906645520373 | 0.941520979916 | -1.23976179e-6 | +4.00777995e-6 |

Both formulations retain a similar p outer-order drift. Their small
same-order differences therefore do not bound either one's optical error.
The s discrepancy also changes sign rather than decreasing monotonically.

The cap-64 export retains the preceding cap-48 coefficients bit-for-bit. The
next cap-80 export changes the retained cap-64 portion by at most 2.63e-12,
consistent with projection roundoff rather than a different cell solve.

## Reproduction and verification

Run from the repository root with NumPy/SciPy. A saved projection fixes the
cell field; without one, the CLI constructs the polynomial-body cell using
the explicit settings above. `--save-projection` stores its coefficients and
physical metadata. `--progress` emits stage records and completed polarization
powers during long solves.

Set `NVF_CACHE` to an existing scratch directory for generated projection files.

```bash
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_mirror_convergence --orders 24 32:24 24:32 32 --save-projection "$NVF_CACHE/wavebit-mirror-cap64.npz" --progress
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_mirror_convergence --orders 40 --save-projection "$NVF_CACHE/wavebit-mirror-cap80.npz" --progress
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_mirror_convergence --projection "$NVF_CACHE/wavebit-mirror-cap80.npz" --orders 32:40 40:32 --progress
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_mirror_cartesian --orders 24 32 40 --progress
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests/test_nvf_mirror_convergence.py tests/test_nvf_mirror_cartesian.py
```

The initial seventeen NVF tests cover explicit orthonormal basis projection, restriction of
the full finite closure, an independent uniform-slab formula, directional and
closed/nonzero-gap comparisons against unreduced Maxwell, refusal of invalid
or asymmetric inputs and the unchanged independent p/s contact derivative
criterion at the final 7559-unknown cell setting. Six Cartesian tests compare
its matrices and powers against unreduced interval factorization, including
zero half widths, multiple materials, contact and redundant partitions.

Validation on 2026-09-08: **23 passed** for these two instruments. Together
with the separate scattering-doubling selection, **41 new tests passed**.

Raw records are in `studies/results/nvf_mirror_timing.log`,
`nvf_mirror_order24_control.log`, `nvf_mirror_projection.log`,
`nvf_mirror_order32.log`, `nvf_mirror_order40.log`,
`nvf_mirror_order40_directional.log` and `nvf_mirror_cartesian.log`.

The [combined mirror/doubling implementation](nvf-mirror-doubling.md) now
supports the same live closure under JAX, admits exact layer graze and checks
the new propagation through order 24. It reduces the measured cost of the
doubling calculation while the modal path remains faster for the tested
NumPy forward sweep. The order-40 measurements above used the modal path.

The [independent half-cell FEM comparison](nvf-mirror-reference.md) now also
reaches orders 40 and 48. Its p/s transmission differs from the candidate by
at most 9.58e-7, while both discretizations retain 40-to-48 changes above
1e-5. This supports prioritizing outer truncation for the present fixture.
The independent component is the auxiliary cell discretization; the Maxwell
propagation is shared, and no 3D optical error bound is obtained.

The optical convergence measurements remain research diagnostics. Agreement
between formulations at one order, a small combined increment or a small
power residual is not an optical error bound. General geometry and production
NVF integration remain separate outstanding work.

## Targeted directional continuation through 56

The same 7559-DOF local cell is exported to cap 112. Its overlapping cap-96
coefficients are **bit-identical**. This extends the retained Fourier range
without changing the auxiliary trial space, grading or field at lower modes.
After the modal/interface memory changes, each selected 56-directional
sector completes with peak process memory about 5.62 GB.

| Polarization and refined direction | Fixed other order | T at 40 | T at 48 | T at 56 | 48-to-56 change |
| --- | ---: | ---: | ---: | ---: | ---: |
| p, x | y=40 | 0.90664428060684 | 0.90662168647408 | 0.90661453679769 | -7.14967639e-6 |
| s, y | x=40 | 0.94152498769412 | 0.94153415103273 | 0.94154602323666 | +1.18722039e-5 |

The p increment decreases from 2.2594e-5 in magnitude to 7.1497e-6. The s
increment instead increases from 9.1633e-6 to 1.1872e-5 and exceeds 1e-5.
The p step alone passes that finite increment check; the s step does not.
These are deliberately selected polarization/direction pairs, not a square
or four-corner convergence test. No estimate formed by summing them is a
computed square result or an optical error bound.

Lossless residuals are 4.12e-11 / 2.19e-12. The single-sector runs take
628 / 608 seconds, respectively. Independent FEM projection work overlaps
part of each run, so these are not isolated throughput measurements. Raw
records: `nvf_mirror_order56_directional.log`,
`nvf_mirror_order56_summary.log` and `nvf_mirror_cap112_projection.log` in
`studies/results/`. The remaining s-y change motivates the next directional
refinement before spending memory on the larger square system.

## The next s-y interval, through 64

The remaining s-y drift is followed with `(x,y)=(40,64)`. The cap-128 local
cell export overlaps cap 112 bit-for-bit, again retaining the same 7559-DOF
trial space. The completed result is

```
s T = 0.9415489501558668
s R = 0.0584510498446252
|T + R - 1| = 4.9205e-13
```

The 56-to-64 change is **+2.92691921e-6**, compared with +1.18722039e-5 for
48-to-56. The latest interval is therefore below 1e-5. However, the entire
48/56/64 window spans **1.47991231e-5**, still above 1e-5. With the earlier
oscillation in this sequence, the small last interval alone is not evidence
of resolved optical convergence. Neither the wider s window nor both axes
have been closed by this calculation.

The optical calculation takes 790.21 s after the 46.39-s projection. Peak
process RSS is 6,038,323,200 bytes (6.04 GB) across that combined process;
it is an observed resident-memory peak, not an allocation bound for larger
orders. There is no concurrent large optical solve. Logs are
`nvf_mirror_order64_directional.log` and `nvf_mirror_order64_summary.log` in
`studies/results/`.

If further qualification against the provisional 1e-5 target is needed,
the next bounded checks are:

1. Extend s at fixed x=40 through y=72 to distinguish a local flattening
   from a stable multi-interval window. Extend the projection first and
   verify overlap with the current cap-128 field.
2. Extend p at fixed y=40 through x=64 to check the 48/56/64 window rather
   than accepting its single small 48-to-56 increment.
3. Once the directional windows settle, complete the missing mixed-order
   corners for each polarization and compare against the corrected fitted
   FEM cell at matching orders. Passing an empirical window is not itself
   a rigorous error bound.

These are conditional precision-study followups, not prerequisites imposed
by a documented product requirement. General geometry and integration of
the new construction remain separate open work. These computational
improvements do not change the production NVF dispatch.
