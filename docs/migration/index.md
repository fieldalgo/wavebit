# Migrating to the integrated API

The integrated package uses `src/wavebit`, shared physical material definitions and separate public solver modules. Existing scientific engines and their measured evidence remain available through an explicit compatibility namespace.

| Earlier interface | Current interface |
| --- | --- |
| `wavebit.rcwa(...)` mutable factory | `rcwa.Problem`, `rcwa.Numerics`, `rcwa.check/prepare/run` |
| General `wavebit.fd` scene/plan/run | `fdtd.Problem`, physical `fdtd.Scene`, separate `fdtd.Numerics` |
| `wavebit.fdtd(...)` periodic factory | Retained as `wavebit.compat.legacy.fdtd(...)` |
| Scene-specific normalized material parameters | Shared physical `Material`, `ConductiveMaterial`, `DispersiveMaterial` and explicit `Units` |
| `JobSpec` | `Job(solver=..., problem=..., numerics=..., execution=...)` |
| Hidden previous-input state | `prepared.run(problem)` requires the complete current input |
| `solver.slice_structure(...)` for physical material pictures | `rcwa.sample_structure(problem, layer=..., quantity='epsilon')`; use `quantity='mu'` for permeability |
| Public normalized magnetic fields | Current field observations use H in A/m |

For a historical script, make its dependency explicit:

```python
from wavebit.compat import legacy as wavebit
from wavebit.compat.legacy import fd
```

This preserves the earlier interfaces and their original units. It does **not** convert an old general-FDTD normalized time, angular rate or magnetic amplitude into SI. Rewrite physical declarations deliberately when adopting the current module; start from the [RCWA](../getting-started/rcwa.md) and [FDTD](../getting-started/fdtd.md) examples.

Layered stacks use physical bottom-to-top order. `top` and `bottom` retain their physical meanings, angles are radians in the current API, and explicit `Units` applies to lengths while time and material rates use SI. Keep original phase, coordinates and field normalization in mind when comparing archived complex amplitudes.

Historical serialized measurements retain their bytes, names and original conventions. Published relocation/provenance maps connect them to the reorganized repository; moving documentation is not a rerun of an experiment.

## Removing a compatibility dependency

Move a caller when its physical inputs and required outputs have current equivalents.
The [structure inspection guide](../guide/rcwa/scene.md#inspecting-a-structure-without-solving)
and the plain [1D](../../examples/rcwa/shapes_1d.py) and
[2D](../../examples/rcwa/shapes_2d.py) shape examples use the current interface.
The shape examples retain their original geometry, materials, Fourier orders, ETM
algorithm and backend choices; mixed, s and p illumination are now explicit problems.
Their individual diffraction efficiencies were compared before and after migration,
in addition to their lossless-energy checks.

The remaining compatibility namespace is **not deprecated as a whole yet**. Its
removal depends on these capabilities, rather than the age of an import:

| Retained workflow | Condition before its compatibility entry can retire |
| --- | --- |
| JIT/vmap over changing shape geometry | Current eager JAX gradients exist, but traced geometry is still refused with `prepared.host_topology`. A current traced path must retain topology checks and derivative validation. See [parametric modeling](../guide/rcwa/parametric-modeling.md). |
| Experimental conforming/auxiliary NVF constructors | Define a current public declaration and retain the research acceptance criteria; ordinary `Numerics` is not a substitute. See [method selection](../guide/rcwa/nvf-method-selection.md). |
| Periodic FDTD, native/graded grids and their derivative workflows | Provide physical declarations and matching fields, phases, units and derivatives in the current FDTD API, or explicitly end support for the individual feature. See the [integration checkpoint](../development/solver-integration.md). |
| General-FDTD scenes extending smoothed substrates through PML | Preserve the scene and boundary semantics when adding current declarations. See [FDTD differentiation](../guide/fdtd/fd-differentiation.md). |
| Historical JobSpec and capability-manifest consumers | Migrate the schema and consumers with explicit units; current `Job` files are not interchangeable with old serialized inputs. |
| Published experiments, course records and frozen reference generators | Retain the original source commit, dependency environment and input/output bytes. A replay of the old experiment and validation of the new API must remain separately identifiable. |

Port numerical assertions and external-reference comparisons with each caller;
deleting a test or changing its expected result is not migration evidence. Shared
internal engines remain implementation dependencies of the current API, even
after a compatibility entry point can be removed. Moving those engines under a
different name does not remove an API dependency or establish feature parity.

Complete earlier manuals remain available for [API declarations](legacy-api.md), [general FDTD](general-fdtd-v1.md), [materials](materials-v1.md), [jobs](jobs-v1.md), [backends](backends-v1.md) and [autodiff](autodiff-v1.md). Advanced compatibility guides state their scope explicitly. The [design document](../design/unified-api.md) records the accepted design and implementation checkpoint; the generated reference and validation report establish implemented behavior.
