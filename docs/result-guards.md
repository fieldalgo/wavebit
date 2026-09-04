# What a solve checks about its own result

Written: 2026-09-03. Measured on CPU (10-core, macOS), NumPy / JAX / PyTorch, float64.

Targets:
- `wavebit/_core/backend.py::check_finite` — the one result guard that survives
- `wavebit/_core/solve_1d.py::final_matching` — where it runs on the 1D path
- `wavebit/_core/scattering.py` — where it runs on the 2D path
- `tests/test_fp_warnings.py` — the unit-level contract
- `tests/test_rayleigh_guard.py` — the end-to-end case it was written for

---

## TL;DR

**One guard, not two.** A NaN or infinity in `R` / `T` raises `FloatingPointError` naming the
contaminated array. There is deliberately **no** generic "`R + T > 1` on a passive stack" check.

There was one — `check_physical`, removed 2026-09-03. It assumed the stack was passive and had
no way to verify it, so it could not tell a broken solve from a structure that legitimately
amplifies. On the FDTD path it fired on 35 % of all solves, none of them defects.

| | `check_finite` | the removed `check_physical` |
| --- | --- | --- |
| reads | the result | the result |
| premise | a finite float is required by arithmetic | the stack is passive, the incidence is a propagating plane wave, `n_top` is real |
| can it verify its premise? | nothing to verify | **no** — it inspected `n_top` only, never the structure |
| false positives measured | none | 85 of 243 FDTD calls; 4 RCWA test cases |

## 1. Why the premise cannot be verified

`check_physical` skipped the check when `n_top` was complex with a nonzero imaginary part —
a *normalization* guard, since incident power is convention-dependent there. That was its only
look at any material. It never saw the structure, `bot`, or `mu`, and it did not distinguish
gain from loss.

So three ordinary situations produced a warning that said `treat it as a failed solve`:

| situation | what `R + T > 1` means there | measured |
| --- | --- | --- |
| **an active medium in the structure** | the correct answer | 4 test cases: the real-antisymmetric tensors in `tests/references/test_s4_golden.py` and `tests/references/test_reticolo_anisotropic_golden.py`, whose own docstring says `R + T` above 1 is their correct answer |
| **a time-domain discretisation** | second-order truncation error | 85 of 243 calls in `tests/test_fdtd.py`, all within `[1.001, 1.0214]`, zero negative efficiencies |
| **an interior source** | undefined — there is no incident flux to divide by | not reachable today; `wavebit/_core/solve_1d.py` refuses an evanescent lit order for this reason and names the dipole case |

The library had already settled this epistemic question the other way once, in
`wavebit/_core/draw.py`: `index_from_eps` takes the principal root and deliberately does not
repair the branch, because *a helper cannot know whether a tiny negative `Im` is noise or a
deliberately active medium*. It hands that judgement to its caller. `check_physical` made the
judgement instead, and made it wrong on every active structure.

## 2. The FDTD measurement

`check_physical` was instrumented to record `sum(R) + sum(T)` at every call across the 134
tests of `tests/test_fdtd.py`.

| | |
| --- | --- |
| calls | 243 (0 skipped — no complex `n_top` in that suite) |
| calls exceeding `atol = 1e-3` | **85 (35 %)** |
| range of those | `[1.001, 1.0214]` |
| negative efficiencies | 0 |

The hits included `test_a_uniform_slab_matches_the_closed_form` at 1.00128 — a planar slab
against an exact transfer matrix, which that file's docstring calls the one test that still
means something if every other part of the package is wrong.

**The excess is truncation error, and it converges.** Uniform slab, TE, normal incidence:

| dx | `R + T` | excess | observed order |
| --- | --- | --- | --- |
| 40 nm | 1.005611 | +5.6e-3 | |
| 20 nm | 1.001282 | +1.3e-3 | 2.13 |
| 10 nm | 1.000308 | +3.1e-4 | 2.06 |
| 5 nm | 1.000099 | +9.9e-5 | 1.63 |

A Yee grid does not conserve energy exactly; it conserves it to `O(dz^2)`. "`R + T > 1` means
gain out of nothing" is a theorem about the continuum and about RCWA's algebra, where the
scattering matrix is unitary for a lossless stack. It is not a theorem about a time-domain
discretisation, so the check was applying a frequency-domain tolerance to a method with
different mathematics.

**This is why the warning could not do its job.** On 2026-08-31 a defect in the conical mount
returned `R + T = 2.0042` at `theta = 30 deg, phi = 0` — both polarisation legs solving the
same problem and being added. `check_physical` ran on that result and emitted its message. The
defect was found by reading the code. A signal present on 35 % of correct solves carries no
information when it appears on the 36th.

## 3. What replaces it

Nothing generic, deliberately. Energy is asserted **per scene**, where the expected number is
known and the tolerance can be chosen for the method:

| assertion | file | tolerance, and why |
| --- | --- | --- |
| `abs(R + T - 1) < 2e-2` at the conical knife edge | `tests/test_fdtd.py` | the FDTD scenes here run coarse; 2.0042 fails it by two orders |
| `abs(R + T - 1) < 5e-3` on a lossless grating | `tests/test_fdtd.py` | a liveness check, deliberately demoted — it catches a dead run, not a wrong one |
| `abs(R + T - 1) < 1e-6` on a lossless passive layer | `tests/test_manifest.py`, `tests/test_slot_order.py` | RCWA, where unitarity is exact and the bound can be tight |
| finite, and no negative power | `tests/test_stability.py` | on a named extreme scene rather than on every solve |

**What this gives up, stated plainly.** `check_physical` ran on every user solve, including
scenes nobody wrote a test for; the per-scene assertions do not. Three FDTD defects are on
record as having announced themselves through it first — an over-conducting PML, a mis-scaled
CFS pole, an out-of-band DFT — as `R + T` between 1.1 and 106. A user hitting something of that
shape on their own structure now gets a plausible number and no warning.

That was judged the better trade: a check that fires on a third of correct results is not
coverage, and the remedies its message recommended (`truncation_order`, thinner slices) do
nothing for any of the three causes above.

## 4. What the check never covered, and still does not

`R + T < 1` was never flagged, because that is ordinary absorption. Two defects from the same
period were therefore always outside its remit, and neither is evidence about the removal:

| defect | symptom | why no result-level check can see it |
| --- | --- | --- |
| a dropped diffraction order at oblique incidence | `R + T = 0.928` | a short sum is what absorption looks like |
| a wrong incident reference under a graded mesh | `R + T = 0.974` | `R` and `T` scale together, so the ratio is untouched |

Both are caught by comparison against a reference solve, which is what `tests/test_fdtd.py`
layer 3 exists to do.
