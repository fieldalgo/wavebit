# Full tensor materials in patterned layers

Wavebit accepts all nine components of relative permittivity and permeability
in a layer background, shape material, or tensor raster. This includes
`xz`, `yz`, `zx` and `zy` coupling, which occurs when a material's principal
axes tilt out of the xy plane. Components are expressed in the public xyz
frame. Rotating a material changes its response, not the shape geometry.

```python
from wavebit import Material, Units, rcwa

crystal = Material(
    epsilon=[2.4, 3.2, 4.1],
    mu=[1.1, 1.3, 1.4],       # omit for relative permeability one
).rotated(0.4, axis='y')

problem = rcwa.Problem(
    units=Units(length='nm'),
    stack=rcwa.Stack(period=(600., 600.), layers=(rcwa.Layer(
        thickness=100., background=Material(epsilon=1.), regions=(
            rcwa.Region(geometry=rcwa.Rectangle(size=(240., 310.)), material=crystal),)),)),
    excitation=rcwa.Excitation(wavelength=730., sources=(rcwa.PlaneWave(side='top'),)))
result = rcwa.run(problem, numerics=rcwa.Numerics(orders=(3, 3), factorization='li'))
```

The tilted axes give this layer a z coupling, so it is solved by the first-order formulation.
The current `rcwa.Result` does not report which formulation ran yet; on the earlier interface
`solver.formulation_assigned` reads `'first_order'` for this layer.

## Geometry and numerical choices

| Geometry | Factorization |
| --- | --- |
| Tensor raster on an axis-aligned rectangular cell | Li (`sampled` or `integrated` coefficients), or linear NVF (`sampled`) |
| Axis-aligned rectangle paintings within a rectangular cell | Li with integrated coefficients, or linear NVF |
| Curved shapes, rotated shapes, or oblique cells | Linear NVF; shape coefficients default to integrated |
| Spatially uniform full tensor | Exact uniform blocks; no normal-field construction |

The existing [shape integration restrictions](tensor-shape-integration.md)
remain applicable to periodic self-overlap and painted contours. The tensor
projector is `linear`. Anisotropic `Stack(top=..., bottom=...)` half-spaces remain unsupported;
scalar epsilon and mu are accepted there. The public full-tensor stack solver
supports `rcwa.Numerics(algorithm='smm')` (default) and `'etm'`. Both support incidence
from either side, complex amplitudes, power and six-component fields. With
SMM and `rcwa.Outputs(scattering=True)`, `result.scattering` also exposes
the scattering matrix, one per wavelength, in the same scalar half-space port basis as the
second-order solver, so compatible stacks can be joined with `.cascade()`.

## Constitutive operators and the solver

The three field directions must be factorized together when z coupling exists.
For Li, the normal-displacement/tangential-field exchange includes both tangent
components at each interface. Successive transforms run along v and then u: the
fixed nesting, the only one tensor Li builds (`li_nesting='symmetric'` is refused).
For NVF, the pointwise continuous mixed-field relation gives full 3×3 blocks
`A` and `B`; the effective operator is `conv(A) @ inv(conv(B))`.
Material-region integration precedes this closure for integrated shapes.

The sampled NVF path samples the projector-dependent factors. It retains the
exact pixel-box transform of the bare zz material term and samples its normal
field correction. This reduces continuously to the existing five-slot
construction when z coupling vanishes. Refining the raster and Fourier order
controls the remaining sampling approximation.

`TensorConvMatrices` has nine slots:

```text
xx, yy, inverse(zz), -xy, -yx, -xz, -yz, -zx, -zy
```

The same layout holds for epsilon and mu. It represents nonlocal Fourier
operators, not a pointwise 3×3 material matrix. A z coupling in either quantity
selects the first-order solver. It eliminates the longitudinal fields as

```text
Ez = inverse(ezz) (Ky Hx - Kx Hy - ezx Ex - ezy Ey)
Hz = inverse(mzz) (Kx Ey - Ky Ex - mzx Hx - mzy Hy)
```

These relations are also used during reconstruction of all six field
components. For bottom incidence, layer order and both tensors' z coupling
signs are transformed consistently.

Concrete tensors without z coupling retain the smaller five-slot, second-order path.
A traced full tensor or one carrying derivatives keeps all components, even
when a coupling is zero at the evaluation point: its derivative may be nonzero.
Consequently a full-tensor differentiated or jitted solve can cost more than
an eager solve at a z-decoupled value. An explicit `rcwa.Numerics(formulation='second_order')`
rejects nine-slot input instead of discarding its components.

## Validation and convergence

`tests/rcwa/test_full_tensor_patterns.py` checks:

- Exact laminate operators independently integrated from continuity of normal
  D/B and tangential E/H, for x and y interfaces and complex full tensors.
- Reflection and transmission against an independent Maxwell matrix exponential
  and plane-wave boundary match, with loss and from both incidence sides.
- Uniform-film reduction, magnetic half-spaces, electric/magnetic duality, and
  both curl equations in reconstructed fields.
- NumPy/JAX/PyTorch agreement, JIT fields, and material-rotation derivatives
  against finite differences, including a zero-coupling evaluation point.
- Convergence of different factorizations for a two-dimensional rectangle and
  sampled/integrated NVF for a circle on an oblique lattice.

The older fmmax and RETICOLO goldens continue to cover their original
in-plane-pattern or uniform full-tensor cases. They are not independent
full-tensor two-dimensional pattern references and have not been regenerated.

For the test rectangle (237×310 nm in a 600 nm square cell, 73 nm thickness,
730 nm wavelength, theta=0.23, phi=0.19, the full epsilon/mu pair defined in the
test), the s-incident powers were:

| Fourier half-order | Li R | Li T | NVF R | NVF T |
| --- | --- | --- | --- | --- |
| 1 | 0.03503484 | 0.96496516 | 0.03359187 | 0.96335680 |
| 3 | 0.03424091 | 0.96575909 | 0.03408849 | 0.96490794 |
| 5 | 0.03438028 | 0.96561972 | 0.03432622 | 0.96596868 |
| 7 | 0.03439952 | 0.96560048 | 0.03443170 | 0.96541664 |

NVF used a 64×64 normal grid. This is a convergence example, not a universal
accuracy bound. Finite-order PN is not necessarily Hermitian for lossless
materials and may show a power deficit or excess. The first-order mode split
uses attenuation for decay-dominated eigenvalues and energy flow for
propagation-dominated ones, so small spurious evanescent flux does not reverse
its direction. An unbalanced split raises eagerly and produces NaN modes under JAX tracing;
increase the Fourier order and normal resolution rather than ignoring it. In
a fixed-seed check of 40 tilted positive-definite epsilon/mu rectangle patterns,
three refused at half-order 1 and all three ran at half-orders 3 and 5. This
checks recovery from a coarse discretization, not universal convergence.
Near grazing, strongly
active media and poorly converged operators require separate scrutiny.
Check observables against Fourier order and normal resolution; energy balance
alone cannot establish accuracy.

The full-tensor algebra is independently implemented from Maxwell's equations
and Fourier factorization; no third-party solver code or dependency was added.
The mathematical reference is [Li, *Fourier Modal Method*, §§13.3.3.1 and 13.4.3](https://www.fresnel.fr/files/gratings/Second-Edition/GratingsTheoryandNumericApplicationsSecondEdition.pdf#page=564).
