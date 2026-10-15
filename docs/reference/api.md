# API reference

> **Historical API:** snippets and descriptions using `wavebit.compat` on this page require a historical checkout.
> The namespace is absent from current installations; the pre-retirement baseline is
> [revision `3c8a50f8`](https://github.com/fieldalgo/wavebit/tree/3c8a50f8a6506240169538957e99648a4b18ca3a).
> Reproduce recorded experiments with their own cited commit and environment; their measurements have not been rerun.


The public namespaces are `wavebit.rcwa` and `wavebit.fdtd`. Both consume immutable problems and explicit numerical/execution settings. Shared materials, units, observation types and structured errors live at the package root.

- [Shared types](common/index.md): material response, physical units, execution, jobs and errors.
- [RCWA](rcwa/index.md): periodic geometry, excitation, numerics and results.
- [FDTD](fdtd/index.md): physical domains, time-domain sources, numerics and monitors.
- [Lifecycle](../guide/common/lifecycle.md): `check`, `prepare`, `run` and compatible reuse.
- [Earlier API](../migration/legacy-api.md): historical normalized engine details, with pinned source for retired imports.

The pages below are generated from current public declarations. Private implementation modules are source references, not alternate user APIs.
