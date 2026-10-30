# Current implementation map

This map follows the current immutable `Problem` → `check` → `prepare` → `run`
interfaces. Use it to find a change location and the first regression checks to
run. Start with [source checkout and debugging](source-checkout.md) to ensure
that the interpreter imports the copy you are editing.

The public modules are `wavebit.rcwa` and `wavebit.fdtd`. Both reuse established
numerical implementations behind their input and result adapters. Reusing an
internal engine does not make its mutable objects or normalized units part of
the new public contract. The imported [course code map](../learn/fundamentals/code-map.md)
and [periodic integration checkpoint](solver-integration.md) describe the
compatibility workflow; new application code uses the current modules.

## Shared declarations and units

| Responsibility | Start here | Boundary to preserve |
| --- | --- | --- |
| Physical material response | [materials/models.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/materials/models.py), [materials/response.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/materials/response.py) | Evaluate the declared physical model; relative permittivity is not refractive index. |
| Geometry units and SI conversions | [units.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/units.py), [materials/_lower.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/materials/_lower.py) | Length follows `Units`; seconds, Hz/rad/s and S/m are converted exactly once when lowering FDTD. A table is not silently fitted into a causal time-domain model. |
| Immutable replacement, execution and errors | [_common/declarations.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/_common/declarations.py), [_common/execution.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/_common/execution.py), [errors.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/errors.py) | Keep physical inputs, numerics and execution separate; retain structured errors rather than silently changing the model. |
| Named observations and validity | [observations/contracts.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/observations/contracts.py) | Preserve spectral axes, units, calibration and failed validity checks. |
| Reproducible input declarations | [jobs.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/jobs.py) | A job serializes supported physical inputs and execution policy, not a solver's last mutable state. |

Run records have a separate boundary from executable Jobs:
[records.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/records.py) manages execution,
archive completion and loading;
[_common/record_io.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/_common/record_io.py)
encodes numerical observations; and
[_common/provenance.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/_common/provenance.py)
captures the installed package identity. See [the record guide](../guide/run-records.md).
Loading retains data and metadata without recreating a solver or executing archived source.

## RCWA: physical problem to spectral result

Read these stages in order for a single plane wave. Coherent beams and internal
sources branch through the source assembly noted below.

| Stage | Implementation | What to inspect |
| --- | --- | --- |
| Declare | [rcwa/problem.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/problem.py), [rcwa/numerics.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/numerics.py) | Bottom-to-top layers, excitation, requested fields/scattering and numerical choices. |
| Check and prepare | [rcwa/_lifecycle.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_lifecycle.py) | `_resolve`, `prepare` and `Prepared._solvers` admit inputs, freeze geometry and reject incompatible reuse. `check` does not solve scattering. |
| Adapt live values | [rcwa/_adapt.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_adapt.py) | `build_solver` evaluates materials at each wavelength and carries live arrays into the internal factory. Avoid host conversions that sever derivatives. |
| Build and solve | [rcwa/_engine/engine.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_engine/engine.py), [parametric.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_engine/parametric.py), [factorization.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_engine/factorization.py) | Input dispatch, frozen geometry checks, material matrices and solve orchestration. Follow the selected geometry/factorization path instead of adding another front-door special case. |
| Propagate and connect layers | [solve_1d.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_engine/solve_1d.py), [solve_2d.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_engine/solve_2d.py), [scattering.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_engine/scattering.py) | Modal algebra, propagation and interface composition. Changes here require independent physical checks, not only agreement between backends. |
| Assemble coherent sources | [illumination.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_engine/illumination.py) | Source grouping and boundary power. This route is eager; it does not invent a unique single-port R/T for a general coherent field. |
| Package and sample | [rcwa/results.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/results.py), [fields.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_engine/fields.py) | `pack` retains the wavelength axis and result-owned reconstruction state; public E/H are V/m and A/m. |

A prepared RCWA object retains structural constraints and frozen geometry, not a
replacement problem for the next call. Trace `Prepared.run(problem)` through
`_solvers`, the selected engine solve and `pack`. A later solve must not change
the field returned by an earlier result. See [lifecycle](../guide/common/lifecycle.md)
for the supported live/static boundary and [differentiation](../guide/common/autodiff.md)
for transformations and their limits.

## FDTD: physical domain to time stepping and observations

| Stage | Implementation | What to inspect |
| --- | --- | --- |
| Declare | [fdtd/_problem.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/fdtd/_problem.py), [fdtd/_numerics.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/fdtd/_numerics.py) | Physical domain, structures, sources and named observations; grid, duration and PML are numerical choices. |
| Check and prepare | [fdtd/_api.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/fdtd/_api.py) | Public `check`, `prepare` and `Prepared._lower`; reuse depends on the resolved grid and time-step structure as well as declaration types. |
| Lower physical inputs | [fdtd/_lower_public.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/fdtd/_lower_public.py) | `lower_problem` converts units and places numerical padding outside the physical domain. Current moments, pulse widths and source positions must retain their physical meaning. |
| Prepare numerical data | [fdtd/_prepare/plan.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/fdtd/_prepare/plan.py), [fdtd/_prepare/__init__.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/fdtd/_prepare/__init__.py) | Separate the static plan from live coefficients. Material admission, occupancy, sources and monitors have their own modules in this directory. |
| Advance fields | [fdtd/_execution.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/fdtd/_execution.py), [fdtd/_runtime.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/fdtd/_runtime.py), [fdtd/_kernel/step.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/fdtd/_kernel/step.py) | Execution lowers coefficients, runs the selected runtime and advances the Yee state. Preserve staggered field times, pole/PML memory and observer ordering. |
| Assemble and convert results | [fdtd/_assemble.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/fdtd/_assemble.py), [fdtd/_result.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/fdtd/_result.py) | `adapt_result` exposes seconds, physical coordinates and SI field units. Raw spectral amplitudes require their declared reference normalization before becoming efficiencies. |

The public FDTD boundary currently accepts CPU and double precision only. A
successful check reports admission and estimates; it does not advance fields or
establish spatial, temporal or absorber convergence. The [FDTD guide](../guide/fdtd/general-fdtd.md)
explains supported sources, materials and observations. Historical periodic
FDTD adapters under `_periodic` have a separate coordinate/normalization contract;
do not route a new physical-domain API change through them merely because both
paths report diffraction.

## Choose the first regression by what changes

These are starting checks, not a claim that a small file covers every affected
physical regime. Run the narrow case while editing, then the relevant complete
file and the appropriate verification tier. Naming a file opts into its slow and
research cases; inspect its scope before launching a long calculation.

| Change | First checks | What they distinguish |
| --- | --- | --- |
| Material response or SI conversion | [test_common_material_models.py](https://github.com/fieldalgo/wavebit/blob/main/tests/materials/test_common_material_models.py), [test_lifecycle.py](https://github.com/fieldalgo/wavebit/blob/main/tests/fdtd/test_lifecycle.py) | Equivalent geometry units, SI conductivity/poles, supported causal material admission and intact derivative graphs. |
| RCWA preparation, live values or result ownership | [test_unified_rcwa.py](https://github.com/fieldalgo/wavebit/blob/main/tests/api/test_unified_rcwa.py) | Reuse versus fresh preparation, independent Fresnel fields/powers, invalid transformed inputs and earlier-result independence. |
| Coherent sources and absolute power | [test_unified_rcwa_sources.py](https://github.com/fieldalgo/wavebit/blob/main/tests/api/test_unified_rcwa_sources.py) | SI source strength, finite observation regions and independently evaluated fields. |
| FDTD domain, source/monitor or reuse | [test_lifecycle.py](https://github.com/fieldalgo/wavebit/blob/main/tests/fdtd/test_lifecycle.py) | PML outside the physical domain, unchanged laboratory coordinates, live source/material reuse and resolved duration guards. |
| RCWA modal or reconstruction algebra | [test_smm_field_state.py](https://github.com/fieldalgo/wavebit/blob/main/tests/rcwa/test_smm_field_state.py), [test_lossy_media.py](https://github.com/fieldalgo/wavebit/blob/main/tests/rcwa/test_lossy_media.py) | Independent interior fields and interface responses rather than a regenerated solver golden. |
| FDTD waveguide modes | [test_fd_waveguide_modes.py](https://github.com/fieldalgo/wavebit/blob/main/tests/fdtd/test_fd_waveguide_modes.py) | Modes of the discrete Yee system against closed forms, exact slab dispersion and a published corner benchmark, and a solved mode advanced by the time stepper itself. |
| FDTD mode source | [test_fd_mode_source.py](https://github.com/fieldalgo/wavebit/blob/main/tests/fdtd/test_fd_mode_source.py) | A one-way mode sheet's injection and leakage against their exact grid predictions, absolute power, convergence, a butt joint against mode matching, backends, derivatives and refusals. |
| FDTD time step or constitutive update | [test_fd_stage3_backends.py](https://github.com/fieldalgo/wavebit/blob/main/tests/fdtd/test_fd_stage3_backends.py), [test_fd_constitutive_absorber.py](https://github.com/fieldalgo/wavebit/blob/main/tests/fdtd/test_fd_constitutive_absorber.py) | Unchanged raw fields under observation-only changes, discrete energy/work identities and backend agreement. |
| Cross-solver phase, port or diffraction normalization | [test_public_lifecycle_diffraction.py](https://github.com/fieldalgo/wavebit/blob/main/tests/cross_solver/test_public_lifecycle_diffraction.py) | Current public APIs, unequal structures and complex nonzero orders; equal total power alone can hide a phase error. |
| Recorded results and source identity | [test_run_records.py](https://github.com/fieldalgo/wavebit/blob/main/tests/api/test_run_records.py) | Exact arrays/units, corrupt or incomplete archive refusal, clean input association and loading without the producing backend. |
| Job serialization | [test_unified_jobs.py](https://github.com/fieldalgo/wavebit/blob/main/tests/api/test_unified_jobs.py) | Physical declarations, units, dtype/shape and unsupported serialization inputs. |
| Refusal codes or an `orders=` window | [test_error_codes.py](https://github.com/fieldalgo/wavebit/blob/main/tests/api/test_error_codes.py) | One `area.item` code at every public refusal, `prepared.structure` naming what changed, engine refusals raised with the check's code, and the explicit `(Mu, Mv)` pair everywhere. |

For example, these two checks isolate preparation from stepping/scattering:

```sh
python -m pytest tests/api/test_unified_rcwa.py::test_check_does_not_run_scattering -q
python -m pytest tests/fdtd/test_lifecycle.py::test_check_never_advances_fields -q
```

Use the [verification runner](verification.md) for recorded quick/full/research runs and
strict resumption. Read the [test guide](https://github.com/fieldalgo/wavebit/blob/main/tests/README.md)
for default, slow and research selection. If a numerical change alters a
reference result, diagnose the convention, arithmetic or convergence difference
before changing a tolerance or regenerating evidence. Historical records remain
historical; a new calculation gets a new record.
