# A smoothed substrate through an open boundary

Use `Structure(..., continuation=('z-',))` for a fixed, unturned `fdtd.Box`
substrate that reaches the bottom `Open()` face. It must fill both transverse
directions with more than one grid cell of clearance. Its interior surface stays
where it was declared; its outside face is extended past the allocated PML and
its smoothing halo. The same declaration works on another single axis, including
both ends of that axis. Orthogonal open faces, ranged substrates, rotated boxes
and finite curved scatterers are refused.

The physical `Domain` excludes PML. Increasing `PML.cells` grows the allocation
outside that domain without moving the source, monitor or substrate surface.
Ordinary finite bodies still cannot touch an open face without a supported
continuation declaration. A `SnappedSlab` retains its separate cell-snapped semantics.

`PML(reference_media={'z-': Material(2.1)})` explicitly fixes the numerical stretch
profile. This matters when differentiating a substrate material: its physical
permittivity may vary while the absorber reference remains concrete. Omitting
the entry follows the boundary medium and therefore cannot provide a static
profile for a traced material. Reference media do not replace physical materials.
The engine's stability, PML incidence and junction checks continue to apply.
For a PlaneWave with `DiffractionMonitor` outputs, a traced physical medium in a pure-stretch
PML is checked against the same grazing-order rule as an eager run, including scenes
without smoothed bodies. Invalid transformed values produce NaN fields and derivatives
and failed runtime validity. A traced medium filling a stabilized PML layer is refused:
the exact-return survey is host-only, even with a fixed `reference_media` entry.
Keep that medium concrete or use `PML(stabilizing_loss=0)`.

The following example expresses the original pillar scene in micrometres and SI
pulse bandwidth. The physical height is 1.3 µm; eight cells at 0.05 µm on each end
make the allocated height 2.1 µm. The pillar's base follows the substrate surface
as its height changes. Its reserved geometry range is a numerical declaration.

```{literalinclude} ../../../examples/fdtd/pillar_gradient.py
:language: python
```

Run it with `python examples/fdtd/pillar_gradient.py` (PyTorch) or add
`--backend jax`. The derivative check reproduces this scene's earlier engine
calculation; it does not certify optical convergence. Coarse-mesh and smoothing
warnings remain visible. The regression suite also compares full E and SI H
against an independently declared normalized scene, checks nm/µm invariance,
and compares geometry and material derivatives with fresh finite differences.

See [differentiation](fd-differentiation.md) for the detailed retained engine
admission rules and their historical evidence.
