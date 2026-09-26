# Li factorization for anisotropic materials

Set `factorization='li'` to use successive Cartesian Fourier factorization for
full permittivity and permeability tensors. Tensor inputs default to NVF.
Both choices emit five slots for z-decoupled tensors or nine for full tensors.

```python
import wavebit

crystal = wavebit.Material(epsilon=[2.25, 4., 2.8]).rotated(0.3, axis='y')
solver = wavebit.rcwa(
    wavelength=730., period=(600., 600.), thickness=[173.],
    structure=[dict(background=1., shapes=[
        dict(type='rectangle', lx=240., ly=300., material=crystal),
    ])],
    factorization='li', truncation_order=dict(x=3, y=3),
)
result = solver.solve()
```

## Supported inputs

- An axis-aligned rectangular cell, so lattice u/v coincide with Cartesian x/y.
- Tensor rasters with `fourier_coefficients='sampled'` (default) or `'integrated'`.
  Both describe the supplied piecewise-constant pixels. The integrated builder
  retains every pixel, including its independent derivative.
- Axis-aligned rectangle paintings wholly inside the cell, including its edges,
  with `'integrated'` coefficients. Later rectangles paint over earlier ones.
  Boundaries follow the live rectangle coordinates; no normal grid is used.
- Full complex 3×3 epsilon and mu, including independent xy/yx/xz/yz/zx/zy entries.
- Uniform full tensors through exact scaled-identity blocks. A z coupling in
  either quantity selects the first-order 4ff solver, with ETM or SMM layer connection.

Li does not read NVF normal-field or projector settings; explicit NVF settings
are rejected. It builds the fixed nesting only (v inside u for every slot):
`li_nesting='fixed'` is accepted, and `li_nesting='symmetric'` is rejected by name. Use NVF for general curved or rotated shapes and oblique cells.
Anisotropic half-spaces remain outside the public material interface.

## Construction

For `D = εE` at a planar interface, tangential E and normal D are continuous.
Writing the normal and tangential blocks of the material operator as

```text
Dₙ = a Eₙ + b Eₜ
Dₜ = c Eₙ + d Eₜ
```

eliminates the discontinuous normal field:

```text
Eₙ = a⁻¹ Dₙ − a⁻¹ b Eₜ
Dₜ = c a⁻¹ Dₙ + (d − c a⁻¹ b) Eₜ
```

This exchange is its own inverse. The implementation exchanges the y variables,
Fourier-transforms along v, and exchanges back. It then repeats the process along
u with the resulting matrix-valued blocks, preserving multiplication order.
For a full tensor, the two tangent components participate in each exchange;
the zz operator includes the coupling through x and y. For a z-decoupled tensor,
zz reduces to the direct rule because E_z is tangential to vertical walls.

This is Li's successive two-dimensional construction in the scalar path's fixed
nesting: v inside, u outside, for every component. Another nesting gives a
different finite-truncation approximation; the scalar path's `'symmetric'`
nesting, which puts each slot's own inverse-rule axis inside, has no tensor
counterpart here. Tensor components that reduce to a scalar recover the scalar
Li matrices of the fixed nesting.
See [Li, *Fourier Modal Method*, §§13.3.3.1 and 13.4.3, Eq. (13.132)](https://www.fresnel.fr/files/gratings/Second-Edition/GratingsTheoryandNumericApplicationsSecondEdition.pdf#page=572).

The mathematical reference informed an independently written implementation.
No reference software, source code, or additional dependency is included. Its
one-dimensional row transform called the scalar module's earlier Fourier helpers
until the scalar Li builders were rewritten from Li's papers (2026-09-25); it has
run on the rewritten series since, and `LICENSE` records that history.

## Differentiation and verification

NumPy, JAX and PyTorch use the same builder. Raster values are differentiable
without constructing or freezing normals. For rectangle geometry derivatives,
call `freeze_geometry()` before JAX tracing and replay the stored edge ordering.
Positions, lengths, periods and material values remain live. Re-freeze when the
rectangle edge ordering or painting topology changes. Frozen Li and NVF records
are not interchangeable.

`tests/test_tensor_li.py` checks the scalar reduction, independent exact stripe
relations, complex reflection/transmission amplitudes against stripe NVF,
magnetic painting versus raster input, lossless/passive behavior, backend
agreement and finite-difference checks of material and geometry gradients.
`tests/references/test_tensor_li_golden.py` compares three patterned gyrotropic
pillars against the existing independent fmmax data at Fourier order 14.
The maximum R/T discrepancies from the nearer stored formulation are
1.29e-6, 3.10e-6 and 1.24e-5; its two stored formulations differ by
4.84e-6, 2.97e-5 and 1.67e-4 respectively. Those reference spreads set the
test tolerances and limit the accuracy claim.
Passing these checks does not remove the need to converge the Fourier order
for a new structure.

Full-tensor validation, field reconstruction, mode selection and convergence
are described in [full tensor patterns](full-tensor-patterns.md).
