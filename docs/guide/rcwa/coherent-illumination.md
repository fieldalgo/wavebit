# Coherent illumination and internal sources

> **Historical API:** snippets and descriptions using `wavebit.compat` on this page require a historical checkout.
> The namespace is absent from current installations; the pre-retirement baseline is
> [revision `3c8a50f8`](https://github.com/fieldalgo/wavebit/tree/3c8a50f8a6506240169538957e99648a4b18ca3a).
> Reproduce recorded experiments with their own cited commit and environment; their measurements have not been rerun.


:::{note}
Every example below uses the current `rcwa` lifecycle. The number of calculation groups is
`result.diagnostics['source_group_counts']`, one count per wavelength. One quantity is still
reported only by the earlier `wavebit.compat.legacy` interface, because the current result does
not carry it yet: a finite region's net flux with its incoming/outgoing interference
(`net_into_stack`). The examples say where.
:::

Put several sources in `rcwa.Excitation(sources=...)` when plane waves need their own direction,
polarization or phase, or when light arrives from both half-spaces. Every beam shares the
excitation's vacuum wavelength. The calculation assumes linear materials.

```python
import numpy as np
from wavebit import Material, Units, rcwa

problem = rcwa.Problem(
    units=Units(length='nm'),                       # all lengths in nm
    stack=rcwa.Stack(
        period=(900., 850.),
        top=Material(epsilon=1.), bottom=Material(epsilon=2.25),
        layers=(rcwa.Layer(thickness=100., background=Material(epsilon=4.+.6j)),)),
    excitation=rcwa.Excitation(
        wavelength=600.,
        sources=(
            rcwa.PlaneWave(side='top', polarization='s', kx=.002, ky=0.),  # nm^-1
            rcwa.PlaneWave(side='bottom', polarization='s', kx=.002, ky=0.,
                           amplitude=np.exp(1j*np.deg2rad(90.))))),
    outputs=rcwa.Outputs(diffraction=False, fields=True))
result = rcwa.run(problem, numerics=rcwa.Numerics(orders=(0, 0)))  # a uniform slab
top, bottom = result.outputs['top'], result.outputs['bottom']
incident = top.incoming + bottom.incoming
output_top, output_bottom = top.outgoing / incident, bottom.outgoing / incident
print(output_top, output_bottom, result.absorption)
# [0.73987951] [0.05936313] [0.20075736]
```

`Stack.top` and `Stack.bottom` still describe the materials above and below the layer stack.
Each source names the side it arrives from with `side='top'` or `'bottom'`. Layer order stays
bottom to top, starting at `z=0`. A run with more than one source, or with any source other than
a single `PlaneWave`, reports boundary power and requested fields rather than per-order `R`/`T`,
so it needs `Outputs(diffraction=False)`. `result.outputs['top']` and `result.outputs['bottom']`
are `BoundaryPower` records with `incoming` and `outgoing` power; an output fraction is their
ratio.

## What each PlaneWave specifies

| PlaneWave argument | Meaning |
| --- | --- |
| `side` | `'top'` (the default) or `'bottom'`: the half-space the beam arrives from. |
| `theta`, `phi` | Direction angles in radians, evaluated using the material on that beam's incident side. |
| `kx`, `ky` | Real Cartesian transverse wavevector components, in inverse lengths of the problem's `Units`. Do not combine with angles on the same PlaneWave. |
| `polarization` | `'s'`, `'p'`, or electric s/p coefficients `(a_s, a_p)`, normalized by their Euclidean length. Defaults to `'s'`. |
| `amplitude` | Complex multiplier of that beam's normalized s/p pair, in V/m. `0.5*np.exp(1j*np.deg2rad(45.))` specifies magnitude 0.5 and phase +45°. |

The earlier interface's `psi`, `chi` polarization angles are not a `PlaneWave` field; give the
Jones pair instead.

Angles use the normal pointing into the stack: −z from the top, +z from the bottom.
The azimuth sets the transverse direction in the same xyz frame on both sides.
Normal incidence with no azimuth uses the φ=0 s/p basis. Relative s/p signs follow
Wavebit's electric-amplitude convention; at normal incidence its basis vectors are
s=+y and p=−x on both incident sides.

A beam with no direction is normal incidence. There are no shared defaults: each `PlaneWave`
states its own direction and polarization, which is why the example repeats `kx`. Several plane
waves on one side are several sources with the same `side`.

PlaneWave amplitudes are referenced at **x=y=0 on that beam's outside interface**:
`z=sum(thickness)` for top, `z=0` for bottom. Their phase difference is between those
two reference points. Translating either reference plane would require a propagation
phase factor. Equal amplitude magnitudes do not in general mean equal normal powers,
because the two media, angles and polarizations can have different admittances.

### Different outer media, common kx and ky

In the example, the top index is 1 and the bottom index is 1.5. The same
`kx=0.002 nm^-1` gives a top angle of about 11.01° and a bottom angle of 7.31°.
Both beams are valid inputs on the same diffraction grid; their normal wavenumbers
and power weights are calculated from their respective materials.

At the reference points, changing the bottom beam phase gives:

| Bottom phase | `output_top` | `output_bottom` | `absorption` |
| --- | --- | --- | --- |
| 0° | 0.481181 | 0.242506 | 0.276313 |
| 90° | 0.739880 | 0.059363 | 0.200757 |
| 180° | 0.393406 | 0.411418 | 0.195176 |

The material and the two individual incident powers are unchanged. Interference
changes the outgoing fields and the field inside the absorbing slab.

## Several beams from the same side

Give several `PlaneWave` sources the same `side`. Leave out the other side when it has no
incident beam; both outer materials still belong in `Stack.top` and `Stack.bottom`. Every beam
shares `wavelength`, but can have its own direction, polarization, amplitude magnitude and
phase.

This complete example illuminates a uniform absorbing slab from the top with
two s-polarized beams: one normal, the other tilted 20° toward +x.

```python
import numpy as np
from wavebit import Material, Units, rcwa

beams = (
    rcwa.PlaneWave(
        side='top', theta=0., phi=0., polarization='s', amplitude=1.,
    ),
    rcwa.PlaneWave(
        side='top', theta=np.deg2rad(20.), phi=0., polarization='s',
        amplitude=0.5 * np.exp(1j * np.deg2rad(90.)),
    ),
)
same_side = rcwa.Problem(
    units=Units(length='nm'),
    stack=rcwa.Stack(
        period=(900., 850.), top=Material(epsilon=1.), bottom=Material(epsilon=2.25),
        layers=(rcwa.Layer(thickness=100., background=Material(epsilon=4.+.6j)),)),
    excitation=rcwa.Excitation(wavelength=600., sources=beams),
    outputs=rcwa.Outputs(diffraction=False, power_region='infinite'))
numerics = rcwa.Numerics(orders=(0, 0))  # uniform slab
same_side_result = rcwa.run(same_side, numerics=numerics)
top, bottom = same_side_result.outputs['top'], same_side_result.outputs['bottom']
print(top.outgoing / top.incoming)     # total reflectance
print(bottom.outgoing / top.incoming)  # total transmittance
print(same_side_result.absorption)
```

The indices in `beams[0]` and `beams[1]` are tuple positions, not diffraction orders.
Add more `PlaneWave` objects to specify more inputs. For bottom-only illumination,
`output_bottom` is reflectance and `output_top` is transmittance.

The second beam has half the electric amplitude magnitude and +90° phase relative
to the first at **x=y=0 on the top interface**. Different transverse wavevectors
make this phase difference position-dependent, while it stays fixed in time at
each position. `polarization` sets the relative s/p components within a beam;
`amplitude` multiplies the whole normalized pair. Change only the second beam's
`polarization` to `'p'` to make that input p polarized.

Amplitude magnitude is not a power fraction. For this lossless isotropic incident
medium, each beam's inward normal flux is proportional to `abs(amplitude)**2 *
cos(theta)`. The second beam therefore carries about `0.25*cos(20°) = 0.235` times
the first beam's normal incident flux.

These directions form two calculation groups, so an explicit `power_region` is
required. Their cross terms vanish in the infinite-area average, even though the
local fields remain coherent. To include interference over a finite rectangle, run the
problem again with that rectangle as its `power_region`:

```python
window = same_side.replace(outputs=same_side.outputs.replace(
    power_region=((-450., 450.), (-425., 425.)),  # x and y bounds, nm
))
window_power = rcwa.run(window, numerics=numerics).outputs
print(window_power['top'].outgoing / window_power['top'].incoming,
      window_power['bottom'].outgoing / window_power['top'].incoming)
```

These fractions use the incident-only flux averaged over that same rectangle.
See the observation-area section below for finite-window power and absorption.

### PlaneWave lists and illuminated orders

Use separate sources for independently chosen directions and polarizations. Use `weights`
within an `IncidentOrders` for directions offset from a reference by integer reciprocal
lattice vectors, all with the same s/p coefficients:

```python
order_problem = same_side.replace(
    excitation=same_side.excitation.replace(sources=(
        rcwa.IncidentOrders(
            side='top', theta=0., phi=0., polarization='s', amplitude=1.,
            weights=(((0, 0), 1.), ((1, 0), 0.5 * np.exp(1j * np.deg2rad(90.))))),
    )),
    outputs=rcwa.Outputs(diffraction=False, power_region='cell'),
)
order_result = rcwa.run(order_problem, numerics=rcwa.Numerics(orders=(1, 0)))
```

Here the labels `(m, 0)` are x orders: `kx_m = kx_reference + m*2*pi/Px`, with `ky=0`.
At wavelength 600 nm and period 900 nm in air, the +1 input has angle about 41.81°,
not the 20° angle of the earlier list. In 2D the labels `(m, n)` multiply the first
and second reciprocal lattice vectors. The full electric s/p amplitude is
`amplitude * weight * normalized_polarization`, with each order's weight from `weights`.
Specify separate plane waves to assign different polarizations to different orders. Order
labels must fit `Numerics(orders=...)`, and each input must carry positive inward normal power.

## Independent angles and the calculation flow

You may specify independent angles on both sides:

```python
two_angles = problem.replace(
    excitation=rcwa.Excitation(wavelength=600., sources=(
        rcwa.PlaneWave(side='top', theta=np.deg2rad(20.), phi=0.),
        rcwa.PlaneWave(
            side='bottom', theta=np.deg2rad(35.), phi=0., amplitude=0.7j,
        ),
    )),
    outputs=rcwa.Outputs(diffraction=False, fields=True, power_region='infinite'),
)
result = rcwa.run(two_angles, numerics=rcwa.Numerics(orders=(0, 0)))
assert result.diagnostics['source_group_counts'] == (2,)   # two calculation groups
```

The solver first evaluates each beam's physical transverse wavevector. Two inputs
can share a diffraction grid when their difference equals `m*b1+n*b2`, with integer
m,n and reciprocal lattice vectors b1,b2. Such inputs form a **Bloch sector**, or
calculation group. Inputs that do not satisfy this relation require separate RCWA
matrices. `result.diagnostics['source_group_counts']` reports their number, one count per
wavelength.

![PlaneWave directions are grouped, scattering matrices produce complex boundary amplitudes, and summed E/H fields determine observation-area power.](../../_static/coherent-illumination.svg)

Within each group, the solver builds one two-port scattering matrix. The retained
rectangular order window includes `Numerics(orders=...)` neighbours about every input
in that group. `IncidentOrders.weights` labels must be within the declared order range;
separate PlaneWave directions can widen the combined window further. Widely separated
inputs can therefore cost more than a single normally incident solve.

The two outgoing amplitudes have contributions from both sides:

```{math}
\begin{pmatrix}b_\mathrm{bottom}\\ b_\mathrm{top}\end{pmatrix}
=
\begin{pmatrix}S_{11}&S_{12}\\S_{21}&S_{22}\end{pmatrix}
\begin{pmatrix}a_\mathrm{bottom}\\ a_\mathrm{top}\end{pmatrix}.
```

Here a denotes incoming electric s/p amplitudes and b outgoing amplitudes in this
calculation's boundary basis. For different groups, the physical Cartesian fields
are reconstructed with their respective spatial phases before addition:

```{math}
\mathbf E=\sum_j\mathbf E_j,\qquad
\mathbf H=\sum_j\mathbf H_j,\qquad
\langle\mathbf S\rangle=\tfrac12\operatorname{Re}(\mathbf E\times\mathbf H^*).
```

This retains coherent cross terms. Computing efficiencies from each solve and
adding them would discard those terms. Only boundary fields are needed for output
power; recovering fields throughout the layers is optional.

## Choosing the area over which power is averaged

For multiple groups, explicitly set `Outputs(power_region=...)`. Their superposition generally
does not repeat on one structural cell, so the averaging area is part of the question.

| `power_region` | Average |
| --- | --- |
| `"cell"` | The centred lattice parallelogram. Default only when there is one group. |
| `"infinite"` | Infinite transverse-plane average. Distinct transverse wavevectors have zero cross contribution to this average. |
| `((xmin, xmax), (ymin, ymax))` | A finite Cartesian rectangle in the problem's length unit, including all cross terms. |

The integrations use analytic integrals of the Fourier exponentials. Infinite-area
averaging removes some interference contributions **spatially**, even for coherent
beams. It does not change the local field or declare the beams incoherent. Finite
rectangles can show interference that disappears from the infinite-area average.

## Reading the result

The result names its outputs by their **exit side**. Light leaving the top can
contain reflection of the top input and transmission of the bottom input, so it has
no single reflection/transmission label.

| Result | Meaning |
| --- | --- |
| `outputs['top'].incoming`, `outputs['bottom'].incoming` | Incoming-only power, positive into the stack. |
| `outputs['top'].outgoing`, `outputs['bottom'].outgoing` | Outgoing-only power, positive away from the stack. |
| `output_top` | Outgoing top power divided by the sum of incident-only powers on both sides, computed as in the first example. |
| `output_bottom` | Corresponding outgoing bottom fraction. |
| `absorption` | Net power into the stack divided by the incident power, for one periodic group averaged over a cell, or for an infinite-plane average. For finite rectangles or a multi-group cell average, reading it raises `UnsupportedFeatureError`. |

The earlier interface also reports `power.net_into_stack`, the bottom total z flux minus the
top total z flux, including incoming/outgoing interference; the current result does not.

The power values are SI. With amplitudes in V/m, `incoming` and `outgoing` are watts through
the cell or the rectangle (`units='W'`), and a mean in W/m² for `'infinite'`
(`units='W/m^2'`). Output fractions cancel the units.

With lossless outer media and a periodic or infinite-plane average,
`absorption = 1 - output_top - output_bottom`. With absorbing outer media, use
`absorption`: the net flux can also contain incoming/outgoing interference.
A finite observation rectangle can exchange power through its lateral faces, so
its two horizontal boundary fluxes alone are not assigned a total absorption.
An outgoing-only flux through a finite rectangle can even be locally negative;
these regional ratios are not bounded probability-like efficiencies.

```python
regional = rcwa.run(two_angles.replace(outputs=two_angles.outputs.replace(
    power_region=((-400., 400.), (-300., 300.)))), numerics=rcwa.Numerics(orders=(0, 0)))
top, bottom = regional.outputs['top'], regional.outputs['bottom']
print(top.outgoing / (top.incoming + bottom.incoming))
# net_into_stack is reported only by the earlier interface's result.boundary_power(region).
```

## Boundary and internal fields

Fields give the complex electric and magnetic components at each position,
including the coherent sum of all input beams. They are not power fractions.
Boundary and internal fields both require `Outputs(fields=True)`, which retains the modal data
needed for later sampling. It does not evaluate a spatial grid in advance.

Without it, the result still holds the power evaluated over the declared `power_region`. It
does not retain complex boundary fields, and the current result has no method that integrates
them over another region: a new region is a new run, as above.

### Boundary fields

The following examples continue the 100 nm slab above.

```python
x = np.linspace(-300., 300., 61)
y = np.array([0.])

incoming = result.boundary_fields(side='top', x=x, y=y, part='incoming')
outgoing = result.boundary_fields(side='top', x=x, y=y, part='outgoing')
above = result.boundary_fields(side='top', x=x, y=y, part='total')

print(above.values.shape)  # (1, 1, 61, 6); above = incoming + outgoing
```

| Argument | Meaning |
| --- | --- |
| `side` | Required: `"top"` or `"bottom"`. Returns the field at that outer interface, approached from the corresponding half-space. |
| `x`, `y` | Required: positive integer sample counts or one-dimensional physical coordinate arrays. `[0.]` is the cell centre. |
| `part="incoming"` | Only waves entering the stack from the selected side. |
| `part="outgoing"` | All waves leaving the stack through that side. In two-sided illumination these combine reflection of inputs on that side and transmission of inputs on the other side. |
| `part="total"` | Sum of incoming and outgoing fields; the default. |

For this slab, `"top"` samples z=100 nm from the upper half-space and
`"bottom"` samples z=0 from the lower half-space. No arbitrary half-space
height is accepted by this method.

### Internal fields

The problem above already requests `Outputs(fields=True)`. A result run without it cannot be
sampled; run the problem again with it before sampling inside.

```python
inside = result.field(x=x, y=y, z=np.array([0., 50., 100.]))

print(inside.values.shape)  # (1, 3, 1, 61, 6)
middle = inside.values[0, 1]  # z=50 nm plane, shape (1, 61, 6)
```

| Argument | Meaning |
| --- | --- |
| `x`, `y` | Required: positive integer sample counts or one-dimensional coordinate arrays. |
| `z` | Required: a positive integer count per layer, or an array of absolute heights in `[0, sum(thickness)]`, measured upward from the stack bottom. |

Internal fields include both propagation directions and every beam's contribution.
There is no `part` argument. For layer thicknesses 100 and 200, the layers occupy
z=0–100 and z=100–300; z does not restart in each layer. An internal interface
point such as z=100 uses the layer above the interface. At the two outer
endpoints, the bottommost and topmost layer values are returned.

Thus `boundary_fields(side='top', ...)` and `field(..., z=[sum(thickness)])` approach the
same interface from different materials. Normal E/H components can jump; tangential Ex, Ey,
Hx and Hy are continuous in the retained Fourier representation.

### Coordinates, array axes and components

Both methods share the [common field sampling rules](fields.md#coordinates).
Integer counts sample the cell; a z count samples each layer separately.
Explicit coordinates, such as `[0.]` or `np.linspace(...)`, use the problem's length unit and
cell-centred Cartesian x,y, including on an oblique lattice. All combinations
of the x and y coordinates are evaluated, and internal sampling repeats the
plane at each supplied z in the requested order.

| Method | Shape of `.values` |
| --- | --- |
| `boundary_fields` | `(n_wavelengths, Ny, Nx, 6)` |
| `field` | `(n_wavelengths, Nz, Ny, Nx, 6)` |

Both return complex `(Ex, Ey, Ez, Hx, Hy, Hz)` in the public xyz frame, with E in V/m and
H in A/m. Wavebit formulates the equations using the normalized magnetic field
`H_tilde = Z0 H` from the outset, with vacuum impedance Z0 ≈ 376.73 Ω, and divides the
magnetic components by Z0 before returning them. The earlier interface returned `Z0 H`.

```python
from wavebit.units import Z0

Ey = inside.values[0, 1, 0, :, 1]  # z=50 nm, y=0, all requested x positions
Ey_amplitude = np.abs(Ey)
Ey_phase_deg = np.angle(Ey, deg=True)
E_squared = np.sum(np.abs(inside.values[..., :3])**2, axis=-1)  # (1, 3, 1, 61)
H = inside.values[..., 3:]  # A/m, shape (1, 3, 1, 61, 3)
H_normalized = Z0 * H  # the normalized Z0 H the earlier interface returned
```

`Ey.real` is the instantaneous field at t=0; at another time use
`Re[Ey exp(-i omega t)]`. `E_squared` is squared electric-field magnitude.
Power flux requires both E and H through `Re(E × H*)/2`.

Fields are not area-averaged or divided by incident power. Doubling every
input beam amplitude doubles these fields. Changing only `power_region` does
not change the fields at a fixed position. See [Fields](fields.md) for
phasor interpretation; a single plane wave's result has the same sampling methods.

### Magnetic-field normalization and physical units

Using `H_tilde = Z0 H` inside the solver puts electric and magnetic fields on the same unit
scale. The public result undoes it: with the input electric amplitude in V/m, the returned
magnetic components are in A/m. For example, a normalized magnetic component equal to 1 for
an input amplitude of 1 V/m is returned as about 0.00265 A/m. Relative input amplitudes
require a physical electric-field reference before both E and H are absolute.

Division by the positive constant Z0 changes the magnitude and units while
preserving the spatial pattern and phase. Time-averaged power flux is

```{math}
\langle\mathbf S\rangle
=\frac12\operatorname{Re}(\mathbf E\times\mathbf H^*)
=\frac{1}{2Z_0}\operatorname{Re}
  (\mathbf E\times\widetilde{\mathbf H}^{\,*}).
```

With E in V/m and H in A/m, this yields W/m². The `BoundaryPower` values already include the
`1/Z0` factor and the area of the cell or rectangle: they are watts, or W/m² for an
infinite-plane mean. The factor cancels in power ratios: do not divide returned R/T or
`output_top`/`output_bottom` by Z0.

Other RCWA packages return magnetic fields in normalized units, as the earlier interface did,
often using simply the name `H`. Multiply the current H by Z0 to compare with them:

| Package | Magnetic-field convention |
| --- | --- |
| <span style="white-space:nowrap">Meent</span> | The normalized scale follows from its [material matrices](https://github.com/kc-ml2/meent/blob/main/meent/on_numpy/emsolver/transfer_method.py) and [field reconstruction](https://github.com/kc-ml2/meent/blob/main/meent/on_numpy/emsolver/field_distribution.py); the returned magnetic components are not converted to A/m. |
| <span style="white-space:nowrap">S4</span> | The [original paper, §3.1](https://web.stanford.edu/group/fan/publication/Liu_ComputerPhysicsCommunications_183_2233_2012.pdf), explicitly defines the change from SI magnetic fields to `Z0 H_SI`. |
| <span style="white-space:nowrap">FMMAX</span> | The [documented conventions](https://facebookresearch.github.io/fmmax/#fmm-conventions) set vacuum permittivity, permeability and c to 1, giving the same relative E/H unit scale. |
| <span style="white-space:nowrap">RETICOLO</span> | The [manual, p. 4](https://arxiv.org/pdf/2101.00901#page=4), sets vacuum permittivity, permeability and c to 1 in Maxwell's equations; the H components returned by `res3()` use that unit system. |

Comparing complex field values across packages also requires matching the input
amplitude scale, coordinate axes and phase conventions.

## Execution scope and validation

A run with more than one source, or with any source other than a single `PlaneWave`, uses the
Cartesian **first-order formulation with SMM**. NumPy, JAX and PyTorch are supported in eager
execution. PlaneWave grouping requires concrete directions, so an input traced under `jax.jit`
or `jax.vmap` is refused (code `rcwa.coherent_transform`); eager `jax.grad` reads its values, as
PyTorch does, and differentiates. Under
`torch.compile` it runs eagerly inside the compiled caller, as every public wavebit callable
does ([`torch-compile.md`](../../development/torch-compile.md) §6). Eager derivatives with
respect to a region permittivity, a layer thickness and a radius were compared with a central
difference on both backends for two plane waves (`tests/api/test_eager_grad_parity.py`); other
inputs and sources have not been compared. It does not expose the single-plane-wave `R`, `T`, per-order
amplitudes or scattering matrix: `Outputs(diffraction=True)` is refused (code
`rcwa.coherent_diffraction`), and so is `Outputs(scattering=True)` (code
`rcwa.coherent_scattering`). A prepared problem fixes the cell, the wavelengths, the source
directions and sampling, and the half-space responses; changing one needs a new `prepare`
(code `prepared.source_grouping`). Amplitudes, polarizations, dipole moments and layer
materials stay live.

External plane waves have one wavelength, real kx/ky and positive incident normal power.
The outer half-spaces remain uniform and isotropic; passive absorbing and magnetic
half-spaces are supported. This feature does not establish a gain-medium branch
prescription. Layer materials retain their Li/NVF and anisotropy restrictions.
This path does not detune a wavelength at a Rayleigh singularity, and refuses
`Numerics(detuning=...)` (code `rcwa.coherent_detuning`); choose and verify a nearby
common parameter point if the matching becomes singular.

The earlier interface's order-keyed map, `excitation={0: 1., 1: .5j}`, runs the ordinary
single-sided solver. Its complex weights use the centre reference stated
above, exactly as `IncidentOrders.weights` does, so the two forms take the same
weights and light the same light (since 2026-09-27; until then the ordinary weights
were referenced at the cell corner). That door also answers per-order `R`,
`T` and amplitudes, which a current `IncidentOrders` run does not yet; its fields and
boundary waves equal this path's for the same weights on the same side.

Validation covers independent four-equation slab matching for both polarizations,
unequal magnetic and absorbing half-spaces, phase-dependent slab absorption,
legacy one-sided field and efficiency limits, rotated tensor layers, coherent
field superposition on oblique patterned cells, reciprocal-order grouping,
finite-area integration, and a thick absorber without growing exponentials.
See [the tests](../../../tests/rcwa/test_coherent_illumination.py) and
[the executable example](../../../examples/rcwa/coherent_illumination.py), which uses the
current `rcwa` lifecycle and reads the group count from `source_group_counts`.

## Source objects

`rcwa.Excitation(sources=...)` takes a nonempty tuple of sources. A `PlaneWave`
represents exactly one incident plane wave. `IncidentOrders` represents several
plane waves via reciprocal-order weights. A `GaussianBeam` describes a localized
external incident field; an `ElectricDipole` describes a prescribed internal current.
Every source in one run shares the wavelength and is combined coherently.

```python
sources = (
    rcwa.PlaneWave(side='top', kx=.002, ky=0., amplitude=1.),
    rcwa.PlaneWave(side='bottom', kx=.002, ky=0., amplitude=.5j),
)
# Pass rcwa.Excitation(wavelength=600., sources=sources) to rcwa.Problem.
```

`side` defaults to `'top'`. An `ElectricDipole` has no side: its `position` places it inside
a layer.

## Gaussian angular spectrum

`GaussianBeam` uses a propagating angular spectrum weighted by
`exp(-waist**2 * (ku**2 + kv**2)/4)`, where ku and kv are perpendicular to the
central propagation direction. Each electric vector is projected perpendicular
to its wavevector, and its magnetic field follows from Maxwell's equations.
Components that cannot propagate or point away from the stack are excluded.
This defines a vector beam; at large angular spread it need not have an exactly
scalar Gaussian envelope.

```python
gaussian = rcwa.Problem(
    units=Units(length='nm'),
    stack=rcwa.Stack(
        period=(900., 850.), top=Material(epsilon=1.), bottom=Material(epsilon=2.25),
        layers=(rcwa.Layer(thickness=100., background=Material(epsilon=4.)),)),
    excitation=rcwa.Excitation(wavelength=600., sources=(
        rcwa.GaussianBeam(
            side='top', theta=0., phi=0., polarization='s',
            waist=1500., focus=(0., 0., 100.), sampling=(9, 9),
        ),
    )),
    outputs=rcwa.Outputs(diffraction=False,
                         power_region=((-3000., 3000.), (-3000., 3000.))),
)
gaussian_result = rcwa.run(gaussian, numerics=rcwa.Numerics(orders=(0, 0)))
```

`waist` is the scalar envelope's 1/e amplitude radius. `focus` is global xyz in
the problem's length unit; `amplitude` is the electric vector norm and common phase at the
focus in the absence of the structure. The incident medium must be isotropic,
lossless, and have positive epsilon and mu: a beam launched from a negative-index side is
refused, because its angular spectrum, transversality and focal phase are written for waves
whose phase and power travel together. A `PlaneWave` from a negative-index side is supported,
with `theta` and `phi` meaning what they mean to a single plane wave -- the light's direction
([`negative-index-media.md`](../../theory/rcwa/negative-index-media.md) §7) -- so such a beam can be built from
`PlaneWave` sources lit from that side. `sampling` gives the midpoint grid
sizes; `angular_spectrum_extent` (default 4) covers +/-angular_spectrum_extent/waist in each beam
transverse direction. Converge both parameters. A finite k spacing produces
spatial replicas, so increase sampling to move them away from the observation area.

A finite rectangular `power_region` is required. Output fractions compare powers
collected in that same rectangle; enlarge the collection area to approach total
beam power. These are coherent fields, not an incoherent average of plane-wave
intensities. No scalar paraxial H approximation is used in the angular components.

## Electric dipoles inside a layer

An `ElectricDipole` specifies the complex electric moment p in C m, not a
self-consistently induced particle polarizability. With exp(-i omega t), its
current is J=-i omega p delta(r-r0). All position coordinates are global xyz;
z=0 is the bottom of the first layer and +z points upward.

```python
dipole = rcwa.Problem(
    units=Units(length='nm'),
    stack=rcwa.Stack(
        period=(900., 850.), top=Material(epsilon=1.), bottom=Material(epsilon=2.25),
        layers=(rcwa.Layer(thickness=300., background=Material(epsilon=1.)),)),
    excitation=rcwa.Excitation(wavelength=600., sources=(
        rcwa.ElectricDipole(
            position=(0., 0., 150.), moment=(0., 1e-29, 0.),
            repetition='periodic', bloch_k=(0., 0.),
        ),
    )),
    outputs=rcwa.Outputs(diffraction=False, fields=True, power_region='cell'),
)
dipole_result = rcwa.run(dipole, numerics=rcwa.Numerics(orders=(1, 1)))
emitted_up = dipole_result.outputs['top'].outgoing  # W through one cell
print(emitted_up / (900e-9 * 850e-9))  # mean W/m^2
```

`Units` fixes the metres per length unit, which SI moments need. The electric field is
V/m and the magnetic field A/m. For a coherent mixture with external waves, give their
electric amplitudes in V/m too. `BoundaryPower` reports watts through the cell or the
rectangle; divide by its area in m^2 for a mean W/m^2.

The host layer must have no regions, or a raster with identical material values
in every spatial cell, for both epsilon and mu; uniform tensor
materials are supported, and other layers may be patterned. The source must be
strictly inside a layer. The solver splits at its height, inserts the integrated
current jump, and cascades affine scattering maps (scattering matrices plus
source-generated outgoing amplitudes). This includes multiple reflection of
emission and supports several internal and external sources together. Fields on
the singular source plane itself are rejected; sample above or below it.

`repetition="periodic"` places a copy in every lattice cell with phase
exp(i bloch_k dot R); omitted bloch_k is zero. `repetition="isolated"` instead
uses Brillouin-zone midpoint quadrature with `sampling=(5,5)` by default. Do not
supply bloch_k in that case. A finite observation rectangle is required. Finite
quadrature is equivalent to a repeated larger supercell: converge the grid and
Fourier order and keep replicas outside the region of interest. Guided-mode or
Rayleigh singularities still require a well-defined limiting prescription; the
source interface does not silently change wavelength or add loss.

With any internal source, output fractions are undefined and reading `result.absorption`
raises `UnsupportedFeatureError`. The unnormalized incoming/outgoing boundary power remains
available (the earlier interface also reports `net_into_stack`). Boundary net flux includes
source supply and material loss; it is not material absorption alone. Source work,
regularized self-fields, and Purcell factors are not returned. All typed sources use eager
first-order SMM; `jit` and autodiff are outside this API's supported scope.

Validation includes homogeneous dyadic Green-function spectra for all three
electric-moment directions, SI length scaling, tangential current jumps,
internal/external superposition, Bloch quadrature versus an explicit supercell,
and Gaussian focal profiles and vacuum transmission on NumPy, JAX and PyTorch.
See [source tests](../../../tests/rcwa/test_sources.py).
