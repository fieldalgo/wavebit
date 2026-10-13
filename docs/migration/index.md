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
| Public normalized magnetic fields | Current field observations use H in A/m |

For a historical script, make its dependency explicit:

```python
from wavebit.compat import legacy as wavebit
from wavebit.compat.legacy import fd
```

This preserves the earlier interfaces and their original units. It does **not** convert an old general-FDTD normalized time, angular rate or magnetic amplitude into SI. Rewrite physical declarations deliberately when adopting the current module; start from the [RCWA](../getting-started/rcwa.md) and [FDTD](../getting-started/fdtd.md) examples.

Layered stacks use physical bottom-to-top order. `top` and `bottom` retain their physical meanings, angles are radians in the current API, and explicit `Units` applies to lengths while time and material rates use SI. Keep original phase, coordinates and field normalization in mind when comparing archived complex amplitudes.

Historical serialized measurements retain their bytes, names and original conventions. Published relocation/provenance maps connect them to the reorganized repository; moving documentation is not a rerun of an experiment.

Complete earlier manuals remain available for [API declarations](legacy-api.md), [general FDTD](general-fdtd-v1.md), [materials](materials-v1.md), [jobs](jobs-v1.md), [backends](backends-v1.md) and [autodiff](autodiff-v1.md). Advanced compatibility guides state their scope explicitly. The [design document](../design/unified-api.md) records the accepted design and implementation checkpoint; the generated reference and validation report establish implemented behavior.
