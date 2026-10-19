# Examples

Wavebit applied to problems, in plain scripts. No datasets or GPU are required.
Some examples use optional JAX/PyTorch backends.

Install `wavebit[plot]` for the FDTD image example; the shape examples use the
optional JAX/PyTorch backends.

## Current physical API

Start here for shared `Material` models, explicit `Units`, immutable problems and
`check` / `prepare` / `run`. These scripts use only the current public interface.
All checks live alongside the example so a copied script retains its verification.

| Example | Workflow and check | Dependencies |
| --- | --- | --- |
| [`record_run.py`](common/record_run.py) | Save inputs, measured arrays, units and source; verify exact round trips without repeating the solver. Use `--output` to retain a record. | Base install |
| [`basic_grating.py`](rcwa/basic_grating.py) | A binary grating and its lossless power balance. | Base install |
| [`shapes_1d.py`](rcwa/shapes_1d.py), [`shapes_2d.py`](rcwa/shapes_2d.py) | Typed regions with explicit mixed/s/p illumination and lossless power checks. | PyTorch / JAX respectively |
| [`rectilinear_contact.py`](rcwa/rectilinear_contact.py) | Cartesian Li gap/material derivatives under JIT with checked topology and NaN for overlap. | JAX |
| [`anisotropic_material.py`](rcwa/anisotropic_material.py), [`coherent_illumination.py`](rcwa/coherent_illumination.py) | Physical tensor spectra and two-source boundary power. | Base install |
| [`nvf_method_selection.py`](rcwa/nvf_method_selection.py) | Explicit ordinary NVF/Li and experimental auxiliary operators for the same pair. | Base install |
| [`learning/`](rcwa/learning/) | 15 teaching scripts: physical geometry, SI fields, Job round trips, sweeps, lossless/Airy checks and gradients. | Base install; JAX/PyTorch for gradients |
| [`nvf_contact.py`](rcwa/nvf_contact.py), [`nvf_curved_contact.py`](rcwa/nvf_curved_contact.py), [`nvf_curved_painting.py`](rcwa/nvf_curved_painting.py) | Scalar-shape JIT derivatives with explicit host topology checks and fresh finite differences. | JAX |
| [`pillar_gradient.py`](fdtd/pillar_gradient.py) | Physical pillar gradients with smoothed substrate continuation through outside PML. | PyTorch or JAX |
| [`pulse_slab.py`](fdtd/pulse_slab.py) | A physical pulse, outside PML padding and a named time record. | Base install |
| [`film_optimization.py`](rcwa/film_optimization.py) | Reuse a prepared RCWA calculation, differentiate film thickness with JAX and minimize reflectance; check the derivative against an independent Airy formula and the design against the quarter-wave thickness. | `wavebit[jax]` |
| [`dispersive_fields.py`](fdtd/dispersive_fields.py) | A Lorentz film with SI rates, time-domain E and raw Fourier E/H; check passive power balance and linear field scaling under prepared reuse. | Base install; `wavebit[plot]` only for `--plot` |
| [`film_and_grating.py`](cross_solver/film_and_grating.py) | Declare the same film or shifted grating for both solvers; compare powers and complex odd-order amplitudes at matched reference faces while refining RCWA order, FDTD spacing and duration separately. | Base install |

For example, from the repository root:

```bash
python examples/rcwa/film_optimization.py
python examples/fdtd/dispersive_fields.py
python examples/cross_solver/film_and_grating.py
python examples/cross_solver/film_and_grating.py --polarization p
```

The cross-solver example starts with a deliberately coarse grid, so its mesh
warnings are expected and remain visible. Its printed errors belong to those two
specific scenes; they do not establish an accuracy bound for other structures.
The dispersive example labels Fourier fields in V·s/m and A·s/m, keeping them
separate from monochromatic amplitudes. Add `--plot /path/pulse.png` to save the
physical time trace; no script opens a plotting window during the default run.

## Advanced physical workflows

The advanced examples also use the current physical API. Periodic stacks use
`fdtd.periodic`; explicit Yee arrays use `fdtd.native`. Length units and SI currents,
fields and time integrals are declared at the boundary. See the
[migration guide](../docs/migration/index.md) for the old-to-current mapping.

| Entry | Question it answers |
| --- | --- |
| [`fdtd_multilayer_optimization.py`](fdtd/fdtd_multilayer_optimization.py) | How can three coating thicknesses and permittivities be optimized together while maintaining minimum lengths and checking the final spectrum independently? |
| [`fdtd_backends.py`](fdtd/fdtd_backends.py) | Evaluate one FDTD objective on NumPy/JAX/Torch and compare automatic derivatives with finite differences. |
| [`fdtd_bloch_gradient.py`](fdtd/fdtd_bloch_gradient.py) | Differentiate oblique plane power through both the source phase and the transverse Bloch boundary. |
| [`fdtd_dispersive_gradient.py`](fdtd/fdtd_dispersive_gradient.py) | Differentiate Lorentz oscillator strength while vacuum-end CPML absorbs outgoing waves. |
| [`fdtd_layer_optimization.py`](fdtd/fdtd_layer_optimization.py) | Optimize a coating thickness in a typed Stack with optional C1 averaging, then check smoothed and sharp designs at twice the resolution and against quarter-wave optics. |
| [`fdtd_frequency_planes.py`](fdtd/fdtd_frequency_planes.py) | Measure six frequency-domain fields and integrated slab R/T without keeping time histories. |
| [`fdtd_grid_gradient.py`](fdtd/fdtd_grid_gradient.py) | Differentiate spectral transmission through a native Yee grid with JAX; uses a graded absorber, not CPML. |
| [`fdtd_poynting.py`](fdtd/fdtd_poynting.py) | How do you measure instantaneous energy flow at a point with staggered E/H snapshots? Saves a Sz time trace inside a film; requires the plot extra. |
| [`fdtd_record.py`](fdtd/fdtd_record.py) | How do you define a typed FDTD scene, take field snapshots of a conical pulse and save an Ex slice? Run `python fdtd_record.py --output /path/to/field.png`; add `--chi 0.7853981633974483` for circular incidence and `--component Hy` to view the magnetic field. FDTD is experimental; see the [research-alpha scope](../docs/guide/fdtd/fdtd-alpha.md). |
| [`shapes_1d_verification.py`](rcwa/shapes_1d_verification.py), [`shapes_2d_verification.py`](rcwa/shapes_2d_verification.py) | Shape/raster comparisons through physical Problems, preserving the original convergence sweeps. |

## Shape derivative examples using the current API

[`rectilinear_contact.py`](rcwa/rectilinear_contact.py) selects exact Cartesian shape Li
and prints JIT transmission, gap derivatives, and material derivatives through contact, using the current physical API and a runtime edge-order guard.
It requires the JAX extra. Unlike the default NVF examples, it uses fixed zero-angle
rectangles wholly inside the cell.

[`nvf_contact.py`](rcwa/nvf_contact.py) keeps `factorization='nvf'` and uses the
rectangular circular projector for contact and material gradients, including a
prepared plan at contact. These three NVF examples require JAX and explicitly select
`topology_check='host'` for CPU callbacks that check each live geometry; under vmap
checks run sequentially. See [the policy and its costs](../docs/guide/rcwa/guarded-shape-jit.md) and
[shape NVF projector details](../docs/guide/rcwa/parametric-modeling.md#complex-nvf-projection-for-rectangular-contacts).

[`nvf_curved_contact.py`](rcwa/nvf_curved_contact.py) exercises the same NVF options on
tangent ellipses and compares the contact derivative with an eager one-sided
difference. See [curved shape support](../docs/guide/rcwa/parametric-modeling.md#disjoint-circles-and-ellipses).

[`nvf_curved_painting.py`](rcwa/nvf_curved_painting.py) uses circular NVF on overlapping
curves and checks the live width derivative. The projector follows the polygon
facets used by the painting; see [scope and approximation limits](../docs/guide/rcwa/parametric-modeling.md#overlapping-circles-and-ellipses).

## How maintained is this?

**Covered by CI, as a smoke job only.** If something here disagrees with
[`../tests/`](../tests), the tests are right — the tests check the physics, and this only checks
that the examples still run.

The `examples smoke` job in [`../.github/workflows/tests.yml`](../.github/workflows/tests.yml)
runs every script from its own directory with `MPLBACKEND=Agg` and fails on a non-zero exit or
on `Traceback` in the output. This is a separate smoke job; runtime depends on the machine and installed backends.

The smoke job checks successful execution and the assertions each script contains.
The current examples above include checks of power balance, an independent analytic
answer, derivatives or explicit refinement. These are checks of the demonstrated
scenes, not a substitute for the numerical validation in `tests/`. Printed values
without assertions are not checked. The former `UnphysicalResultWarning` is no
longer a smoke criterion; see [`../README.md`](../README.md).

Nor does it check what a verification example is *for*. Until 2026-08-11 the 2D one compared a
shape `structure` against a raster that was the vertical mirror of it — a defect that survived
because a mirrored structure radiates into the specular order exactly like the original, so the
two paths agreed to 1e-15 for as long as the only propagating order was the zeroth. Energy was
conserved throughout. If you add a case here, give it a propagating higher order.

The verification pair also stopped being able to lean on a bit-identity. They compared the
**block** path against a commensurate raster, and on such a grating the block path *was* the
raster's arithmetic — Li's rule over the same segments — so it agreed to the last bit while
saying nothing about whether either was right. That path was removed on 2026-08-26. The shape
path runs NVF off an analytic normal field instead. In 2D the same commensurate geometry now
gives two answers that converge on each other rather than one number twice. In 1D each rod spans
the cell, so its covered face pair carries no normal, the field is constant, and the NVF
projector *is* Li's rule: the two agree to 4e-16, which is a check of the face-pair rule rather
than a repeat of the raster. Each file ends with the truncation sweep that shows it.

**It cannot see a material read as the wrong quantity either**, and that is worth stating next
to the mirror, because it is the same shape of defect. `structure` is **relative
permittivity** everywhere — raster pixels, `Material(epsilon)`, `top` and `bot`. A refractive
index left in one of those slots is a legal medium, so the run exits 0, prints numbers and
conserves energy; only the structure is different. The pre-rename spellings (`ucell`, `n_top`,
`n_bot`, `n_base`, `n_index`) were refused by name for that reason until the refusals were
deleted on 2026-08-26 — nothing has been released, so there is no recipe in the world still
using them — and a *value* nobody renamed was always invisible here anyway.
`wavebit.eps_from_refractive_index(n)` is the conversion, and these scripts write it out at the
point the datasheet number arrives rather than folding it into a literal.

The same check runs locally, from the repository root:

```bash
python tools/run_examples_smoke.py
```

It takes a path to narrow the run — `python tools/run_examples_smoke.py examples/rcwa/shapes_2d.py`.
