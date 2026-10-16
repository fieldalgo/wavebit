# FDTD validation

> **Historical API:** snippets and descriptions using `wavebit.compat` on this page require a historical checkout.
> The namespace is absent from current installations; the pre-retirement baseline is
> [revision `3c8a50f8`](https://github.com/fieldalgo/wavebit/tree/3c8a50f8a6506240169538957e99648a4b18ca3a).
> Reproduce recorded experiments with their own cited commit and environment; their measurements have not been rerun.


Spatial, temporal, material and boundary studies. Each report states its measured configuration and remaining limits.

Current public API: [automatic stopping with incident source columns](fd-source-decay.md).

These records predate the public `wavebit.fdtd` module (2026-10-13) and name the interfaces they
measured. The general engine they call `wavebit.fd` (`fd.Scene`, `fd.plan`, `fd.run`, `fd.check`)
is retained as `wavebit.compat.legacy.fd`, which alone keeps `Absorber` and `Decay`; the periodic
stack solver the `fdtd-*` records measure is retained as `wavebit.compat.legacy.fdtd`. For current
usage see the [FDTD guide](../../guide/fdtd/general-fdtd.md) and the
[earlier general-engine manual](../../migration/general-fdtd-v1.md). Source digests quoted
"relative to `src/wavebit/fd`" describe how those runs hashed the tree before it moved to
`src/wavebit/fdtd/`.

- [Mixed material parameters and pole arithmetic refusals](fd-pole-arithmetic.md)
- [C8: one conductive film in a dielectric](fd-conductive-film.md)
- [C8: separated parallel conductive films](fd-conductive-stack.md)
- [Matched absorption of commuting coupled dielectrics](fd-constitutive-absorber.md)
- [C8: one Debye film in a dielectric](fd-debye-film.md)
- [C8: separated Debye films](fd-debye-stack.md)
- [C8: a finite Debye sum in one film](fd-debye-sum-film.md)
- [Automatic stopping after a transient](fd-decay-stopping.md)
- [C8: a mixed Debye and Drude/Lorentz film](fd-hybrid-film.md)
- [Radiation consistent with the Yee lattice](fd-lattice-radiation.md)
- [A periodic metasurface against RCWA (stage 4, C7)](fd-metasurface-validation.md)
- [Mie reference and the curved-interface accuracy gate](fd-mie-validation.md)
- [C8: separated single-oscillator films](fd-oscillator-stack.md)
- [C8: a finite Drude/Lorentz sum in one film](fd-oscillator-sum-film.md)
- [C8: one Drude or Lorentz film in a dielectric](fd-single-pole-film.md)
- [Sphere clipping across a JIT boundary](fd-sphere-clip-jit.md)
- [Smoothed occupancy and the tilted-interface measurement](fd-subpixel-occupancy.md)
- [C8: one film with two Debye poles](fd-two-debye-film.md)
- [Adjoint, discrete derivatives and physical accuracy](fdtd-adjoint-validation.md)
- [Bragg cavity temporal and spatial convergence](fdtd-bragg-cavity.md)
- [Circular FDTD shapes and diameter derivatives](fdtd-circle-gradients.md)
- [Frequency-monitor subsampling cost and error](fdtd-dft-sampling.md)
- [Directional FDTD extraction and persistent near-Rayleigh modes](fdtd-directional.md)
- [Rotated elliptical FDTD shapes and derivatives](fdtd-ellipse-gradients.md)
- [Lossy ENZ films and calibrated scattering convergence](fdtd-enz.md)
- [Half-space derivatives without changing the experiment](fdtd-halfspace-gradients.md)
- [Separating lattice, boundary and finite-time scattering errors](fdtd-lattice-reference.md)
- [Fitting planar dispersive layer boundaries](fdtd-layer-alignment.md)
- [Internal layer derivatives on a fixed mesh](fdtd-layer-gradients.md)
- [Continuous internal-interface sampling](fdtd-layer-smoothing.md)
- [Residual Lorentz motion and dispersive settling](fdtd-lorentz-decay.md)
- [Low-loss mixed Drude/Lorentz metal](fdtd-mixed-metal.md)
- [Constrained multilayer optimization](fdtd-multilayer-optimization.md)
- [Oblique pulse bandwidth and metal interfaces](fdtd-oblique-pulse.md)
- [Periodic painting of multiple rotated FDTD rectangles](fdtd-painted-shapes.md)
- [FDTD CPML strength near a Rayleigh threshold](fdtd-pml-strength.md)
- [Pulse settling versus spatial accuracy](fdtd-pulse-convergence.md)
- [Source spectral zeros and remaining cutoff error](fdtd-pulse-notches.md)
- [Near-Rayleigh channels and diffraction-plane placement](fdtd-rayleigh.md)
- [Transverse rectangle derivatives](fdtd-rectangle-gradients.md)
- [FDTD ringdown fits and a nonexponential cutoff tail](fdtd-ringdown.md)
- [Rotated rectangle area integration](fdtd-rotation-gradients.md)
- [Exact translation averaging for rotated rectangles](fdtd-rotation-smoothing.md)
- [FDTD route parity: do the two engines make the same error?](fdtd-route-parity.md)
- [Fixed-grid incidence-angle sensitivities](fdtd-scene-angles.md)
- [Azimuth derivatives on one transverse mesh](fdtd-scene-azimuth.md)
- [Independent source position and boundary clearance](fdtd-source-clearance.md)
- [Independent transverse meshes and axis-consistent material averaging](fdtd-transverse-resolution.md)
- [Total-depth derivatives with moving exterior reference planes](fdtd-variable-depth.md)

## Public input and preparation controls

`tests/fdtd/test_public_input_followup.py` exercises the current public API;
it does not depend on the historical namespace used by the studies above.
Run it from a source checkout with that checkout's `src` on `PYTHONPATH`:

```sh
python -m pytest tests/fdtd/test_public_input_followup.py -q
```

The tests cover non-Plane input to `PlaneWave` through `check`, `prepare` and
`run`, a valid plane and a plane outside the physical domain, stencil memory
limits and invalid/unused settings, and the public remedy for multiple
structures. `Numerics(memory_bound=...)` passes a limit in bytes to the existing
host prediction guard. The memory routing test stops at entry to sparse geometry
assembly; it verifies the admission boundary and the unchanged default source,
not an assembled operator or scattering accuracy. The multi-structure remedy
is also applied through `check` to confirm that `Numerics(discretisation="voxel")`
removes that refusal.

See the [current input and memory contract](../../guide/fdtd/model.md).

### Public grazing guidance

`tests/fdtd/test_public_grazing_guidance.py` checks the actual messages from
`check`, `prepare` and `run` on narrow-band oblique columns. It executes each
printed public candidate and rechecks the same physical problem, including
stronger existing controls, a quoted source name, the pure and stabilized
refusals, different fixed reference media, the measured floor, and unchanged
private engine guidance after the public call. It verifies that extra absorber
cells grow external padding and preserve the other absorber options.

```sh
python -m pytest tests/fdtd/test_public_grazing_guidance.py -q
```

This validates diagnostics and preparation, without time stepping or new
scattering-accuracy claims. The existing grazing return and PML stability
measurements remain the evidence for the numerical rule. Function-level product
mutations and exact source restoration are recorded in
`tests/fdtd/public_grazing_mutation_evidence.json`.
