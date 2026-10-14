# What a solve checks about its own result

Updated: 2026-10-13. The RCWA guard policy and the FDTD measurement below apply
to the integrated solvers; their numerical error tolerances remain different.

Targets:
- `src/wavebit/_common/backend.py::check_finite` — the one result guard that survives
- `src/wavebit/rcwa/_engine/solve_1d.py::final_matching` — where it runs on the 1D path
- `src/wavebit/rcwa/_engine/scattering.py` — where it runs on the 2D path
- `src/wavebit/rcwa/_engine/engine.py::_refuse_negative_index_half_space` — the one input the removed
  check flagged that no test covered, refused for a while and now solved on the far side
- `tests/rcwa/test_fp_warnings.py` — the unit-level contract
- `tests/rcwa/test_rayleigh_guard.py` — the end-to-end case it was written for
- `tests/rcwa/test_lossy_media.py` — the closed-form tests that replaced it where it caught defects

---

## TL;DR

**One guard, not two.** A NaN or infinity in `R` / `T` raises `FloatingPointError` naming the
contaminated array. There is deliberately **no** generic "`R + T > 1` on a passive stack" check.

There was one, `check_physical`, and `UnphysicalResultWarning` with it. It warned on a negative
efficiency, or on transmitted power above the net power entering. Its premise was a passive
stack lit by a propagating plane wave, and it never verified that premise. It read the result
and the net power entering, and nothing else; it skipped itself only when that power was not
finite. (A second branch looked at `n_top` when no entering power was given, but every caller
gave one, so it never ran.) So it could not tell a broken solve from a structure whose right
answer is above one, and it told the user to treat correct answers as failed solves.

| | `check_finite` | the removed `check_physical` |
| --- | --- | --- |
| reads | the result | the result and the net power entering |
| premise | a finite float is required by arithmetic | the stack is passive, the incidence is a propagating plane wave |
| can it verify its premise? | nothing to verify | **no**: it never saw a material, so not the structure, either half-space, `mu`, or the sign of any `Im(eps)` |

## 1. The correct answers it called failed solves

| situation | what the check saw | why that is the right answer |
| --- | --- | --- |
| a **gain half-space under total internal reflection** | glass onto `eps = 1 - 0.05i`, 45 deg, TE: `R = 1.2646`, `T = -0.2646` | evanescent gain. Since 2026-10-02 the half-space takes Maystre's cut ([`branch-cuts.md`](../../theory/rcwa/branch-cuts.md) §8), which gives this order the decaying root with `Re(kz) < 0`: power flows toward the interface, so its transmitted efficiency is negative by construction. In a grating the same holds for every order past its cutoff in the gain medium |
| a **gain layer** between passive half-spaces | `R + T > 1` | amplification; this was a documented false positive the user was told to filter |
| the **real-antisymmetric tensor** reference cases | `R + T > 1` | these media are active; `tests/references/test_s4_golden.py` and `gen_goldens_reticolo_anisotropic.py` say so in their own docstrings |

A message that says "treat it as a failed solve" on a correct answer costs more than the
silence it replaced: the remedies it recommended (`truncation_order`, thinner slices) change
nothing in any of these cases. And a check that cannot verify its own premise cannot be trusted
in either direction. If `1.1` can be the correct value, a warning on `1.1` carries no
information about whether this `1.1` is.

The library had already settled the same question the other way once, in
`src/wavebit/rcwa/plotting.py`: `refractive_index_from_eps` takes the principal root and
deliberately does not repair the branch, because a helper cannot know whether a tiny negative `Im`
is noise or a deliberately active medium. It hands that judgement to its caller. `check_physical`
made the judgement instead, and made it wrong on every active structure.

## 2. What it did catch, and what pins those now

Four defects on record produced results it flags. In two of them (`ed81de1`, `7571fe3`) the
commit names its warning as the only objection there was.

| defect | symptom | pinned by |
| --- | --- | --- |
| the five-slot builders returned `epz_i` last while `solve_conv` took it third (`ed81de1`, 2026-08-18) | `R + T = 1.61` on a lossless passive pillar | `tests/rcwa/test_slot_order.py`: `abs(R + T - 1) < 1e-6` for every builder |
| the SMM 1D TM transmitted efficiency kept a stray conjugate (`445b71a`, 2026-08-05) | `T = -0.263` into an absorbing substrate, TM only | `tests/rcwa/test_lossy_media.py::test_lossy_substrate_transmission_agrees_between_algorithms` and `::test_lossless_layer_on_absorbing_substrate_conserves_energy` |
| the analytic (now parametric) builder wrote `conj(n)**2` (`b96256b`, 2026-08-05) | an absorbing film read as gain: `R + T` above 1, about 9% on `R` | `tests/rcwa/test_lossy_media.py::test_absorbing_layer_via_the_parametric_path` (Airy, `1e-12`); `tests/rcwa/test_parametric.py`: `0.5 < R + T < 0.95` |
| the painted builder still wrote `conj(n)**2` (`7571fe3`, 2026-08-12) | `R` off by 0.15 on gold, `R + T` up to 1.11 | `tests/rcwa/test_lossy_media.py::test_absorbing_layer_via_the_painted_path` (Airy) |

These were not one-time slips. A conjugate that turns absorption into gain came back three
times: in the SMM TM efficiency normalisation (`conj(kz)`) and the analytic builder
(`conj(n)**2`) on the same day, and in the painted builder a week later, after the builder fix
was taken as complete. Each survived because its tests never looked at the quantity it changed
on an absorbing medium: the substrate tests compared only `R`, and the builder tests used real
indices, where `conj(n)**2 == n**2`. What ended it is a closed-form test per path, the Airy
answer of an absorbing stack through each, at `1e-12`. A closed form also catches a wrong
answer below one, which `check_physical` never could.

One input it flagged had no test at all: a **negative-index half-space** (`Re(eps) < 0`,
`Re(mu) < 0`, and `Re(eps*mu) > 0` so that it propagates). The half-space root reads only
`eps*mu`, and for these media it takes the root that advances its phase away from the
interface, which is the wrong one. Glass onto
`eps = -2+0.1i`, `mu = -1+0.1i` returned `R = 929`, `T = -928` at normal incidence against a
true `0.0011` -- with `R + T = 1` exactly, an identity of the interface conditions whichever root
is taken, so no energy test could have caught it
([`negative-index-media.md`](../../theory/rcwa/negative-index-media.md) §3). After the removal that half-space was
refused where it was set. It is now
solved with its own root, the mirror of Maystre's cut ([`branch-cuts.md`](../../theory/rcwa/branch-cuts.md) §9),
on either side and lit or not ([`negative-index-media.md`](../../theory/rcwa/negative-index-media.md)), and
refused only where it is active
(`src/wavebit/rcwa/_engine/engine.py::_refuse_negative_index_half_space`).
`tests/rcwa/test_negative_index_half_space.py` pins it against closed forms and against a thick
lossy slab of the same medium, a layer, which keeps both roots and so needs no root rule.

## 3. What replaces it

Nothing generic, deliberately. Energy is asserted **per scene**, where the expected number is
known and the tolerance can be chosen for the case:

| assertion | file | why there |
| --- | --- | --- |
| `abs(R + T - 1) < 1e-6` on a lossless passive layer | `tests/api/test_manifest.py`, `tests/rcwa/test_slot_order.py` | RCWA's scattering matrix is unitary for a lossless stack, so the bound can be tight |
| an absorbing rod lands inside `(0.5, 0.95)` | `tests/rcwa/test_parametric.py` | gain would leave the band upward |
| finite, and no negative power | `tests/rcwa/test_stability.py` | on a named extreme scene rather than on every solve |

**What this gives up, stated plainly.** `check_physical` ran on every user solve, including
scenes nobody wrote a test for; the per-scene assertions do not. A user whose own passive
structure hits a defect of the slot-order shape now gets a plausible number and no warning.
That is the trade, judged worth it: a warning that fires on correct answers for every active
medium is not coverage, and each defect it flagged now has a closed-form test on the path where
it happened.

## 4. What the check never covered, and still does not

`R + T < 1` was never flagged, because that is ordinary absorption. A defect that loses energy,
such as a dropped diffraction order, looks exactly like an absorbing structure, and only a
comparison against a reference solve can see it. That is what the oracle tests in
`tests/references/` exist to do.

## 5. The FDTD measurement

`check_physical` was instrumented to record `sum(R) + sum(T)` at every call across the 134
tests of `tests/fdtd/test_fdtd.py`.

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
