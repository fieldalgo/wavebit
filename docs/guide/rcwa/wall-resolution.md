# Resolving walls with stretched coordinates

`rcwa.Numerics(wall_resolution=G)` solves a stack of axis-aligned rectangles in coordinates whose lines crowd at the rectangles' walls. The retained orders then resolve the field at a wall more finely than their count says, which is what the near field beside a metal needs: at an order where the unstretched solve is tens of percent from a finite-element field, the stretched one is within a few percent or less ([the measurements](../../validation/rcwa/wall-resolution.md)).

```python
from wavebit import Material, Units, rcwa

problem = rcwa.Problem(
    units=Units(length='nm'),
    stack=rcwa.Stack(period=(400.0, 400.0), layers=[rcwa.Layer(thickness=150.0, regions=[
        rcwa.Region(geometry=rcwa.Rectangle(size=(200.0, 200.0)), material=Material(epsilon=-20 + 1.5j))])]),
    excitation=rcwa.Excitation(wavelength=900.0, sources=[rcwa.PlaneWave(side='top', polarization='p')]),
    outputs=rcwa.Outputs(fields=True))
result = rcwa.run(problem, numerics=rcwa.Numerics(orders=(12, 12), factorization='li', wall_resolution=0.02))
```

The default is `None`: the cell's own coordinates, and every solve bit for bit as it is without the keyword.

:::{admonition} Experimental
:class: warning

`wall_resolution` is experimental, and every solve that uses it emits `wavebit.rcwa.diagnostics.ExperimentalWarning` on the calling line.

- **What sets it apart.** The unstretched solve is checked two ways: against independent implementations of the same method at the same truncation, and against finite-element fields. The stretched solve has the second check only so far.
- **What the measurements cover.** Seven layers of axis-aligned rectangles against finite-element fields, one wavelength each, at normal incidence. At oblique incidence and with several propagating orders it is checked for energy conservation and convergence, not against a reference.
- **What may change.** Its values, what it accepts and its name, between releases and without the deprecation period the rest of the interface gets.
- **How to use it.** Compare two orders before relying on a value, as with any solve.

A script that has taken this in can silence the one category:

```python
import warnings
from wavebit.rcwa import diagnostics

warnings.simplefilter('ignore', diagnostics.ExperimentalWarning)
```

A scene that is refused raises and does not warn.
:::

## What the value is

`G` is the slope of the solve's coordinates at a wall, a number in `(0, 1]`. With `x = F(u)` along each axis, `F'(u) = G` at every rectangle edge, so `0.02` places the coordinate lines fifty times more densely there than in the cell's own coordinates, and `1.0` on a structure whose walls are already evenly spaced is no stretch at all.

- **`0.02` is the recommended value.** It is the value the measurements were taken at.
- A weaker stretch still helps: `0.1` was measured on two of the layers and was closer than the unstretched solve on both, though not as close as `0.02`.
- The walls are placed at equal distances in the stretched cell, and one function per axis serves the whole stack: every rectangle edge of every layer is a wall of it. Nothing about the placement is a keyword.
- `rcwa.check(...).resolved['wall_resolution']` reports the value, `None` when unset.

## What it accepts

Rectangles along the cell's axes with `factorization='li'` in a rectangular cell; isotropic, non-magnetic materials in the layers, metals included; lossless dielectric half-spaces; one incident plane wave from either side, at any angle; any number of layers, uniform ones included; the scattering-matrix algorithm (`algorithm='smm'`, the default) and the second-order formulation. NumPy, JAX and PyTorch, run eagerly.

Everything else is refused by name with the code `rcwa.unsupported`, at `rcwa.check` as at `rcwa.run`:

- another factorization (the default on regions is `'nvf'`, so `factorization='li'` has to be named), a raster, a shape that is not an axis-aligned rectangle, a skewed cell;
- `algorithm='etm'`, `formulation='first_order'`;
- tensor materials, a material that names a permeability;
- a half-space that absorbs or has a negative permittivity;
- several incident waves or an internal source;
- an axis that has walls while `orders` retains a single harmonic along it;
- two walls too close for the stretch to hold between them (a piece of the stretched cell is `period / number of walls` wide whatever its physical width);
- `Outputs(scattering=True)`.

A rectangle as long as the period along one axis has no wall across that axis, and that axis is not stretched: a lamellar grating is stretched along its one direction.

`jax.jit`, `jax.vmap` and gradients are not supported yet. Which orders propagate, and where a field's samples sit, are read as numbers.

## Reading the result

- **Efficiencies and the amplitudes of propagating orders** mean what they do without the stretch.
- **An evanescent order's amplitude is reported as 0.** In the stretched coordinates only a propagating order of a half-space is a plane wave with an amplitude; what decays beside the stack is the half-space's own modes, which are not orders. Zero is also what an evanescent order contributes to every power.
- **`result.field(x=..., y=..., z=...)`** samples inside the layers at the points asked for, in the cell's own coordinates. Only `reconstruction='plain'` is available: the coordinates already resolve the walls the other rules are for.
- **`result.boundary_fields(...)` is not available**, for the reason the evanescent amplitudes are not. For the field above or below the structure, add a uniform layer of the half-space's medium to the stack and sample inside it.
- **`result.scattering` is not available**: it has a port per order.

## What it costs

The matrices are the size they are without the stretch. Each half-space takes an eigenproblem where the unstretched solve has plane waves, and so does every uniform layer; a uniform layer is no longer merged into a half-space. At order 16 on one patterned layer the stretched solve took 1.7 to 1.8 times as long as the unstretched one in eight of nine measured cases, and 3.7 times in the ninth.

## Where to be careful

- **Just above a Rayleigh wavelength.** On a dielectric block of period 400 nm at 405 nm the stretched solve's `R` read 0.02 at order 8 and 0.31 from order 12 on, where it settled; the unstretched solve was slow there too (0.15, 0.27, 0.30 at orders 8, 12, 16). Just below, at 395 nm, the stretched solve was settled from order 8. Do not read a low order near a Rayleigh wavelength.
- **A weak metal's `R` and `T`.** The measurements were of fields on rectangles. On a circle, which this version does not take, a strong stretch left a 1% offset in `R` for a weakly negative permittivity; a block of the same material was within 0.6%. If only the efficiencies of a dielectric or weakly metallic structure are wanted, the unstretched solve is the checked one.
- **Many walls.** Each piece between two walls gets the same share of the stretched cell. A structure with many walls along an axis leaves each piece few harmonics, and that case was not measured.
