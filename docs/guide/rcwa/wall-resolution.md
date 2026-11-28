# Resolving walls with stretched coordinates

`rcwa.Numerics(wall_resolution=G)` solves a stack of axis-aligned rectangles in coordinates whose lines crowd at the rectangles' walls, and a stack of one circle or one axis-aligned ellipse in coordinates matched to it ({ref}`Circles and ellipses <wall-resolution-circles>`). The retained orders then resolve the field at a wall more finely than their count says, which is what the near field beside a metal needs: at an order where the unstretched solve is tens of percent from a finite-element field, the stretched one is within a few percent or less ([the measurements](../../validation/rcwa/wall-resolution.md)).

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

## How it is checked

- **Against finite-element fields.** Seven layers of axis-aligned rectangles and twelve of one circle or ellipse at normal incidence; and twenty-four metal cases and four dielectric ones beyond it: oblique incidence (p and s, theta 20 and 40 deg), five propagating orders, a glass substrate, two patterned layers, and for a circle a non-square cell, and an ellipse at oblique incidence, on glass and in two layers. On those twenty-eight, at order 12, `R` and `T` are within 0.41% of the references and the near field is 0.02 to 0.11 of the default solve's error, against criteria written before measuring ([the measurements](../../validation/rcwa/wall-resolution.md)). On thirteen of those cases the references themselves move by 0.1 to 0.43% between their two mesh levels, and the comparison is only as tight as that. Four more layers, with four walls along an axis, met the same criteria in two cases and missed them in two: a small `R` was 1.04% off at order 12, and on a metal frame `R`, `T` and the field were outside them ({ref}`the measurement <wall-resolution-many-walls>`). Of eight cases with shapes away from the cell's centre -- circles, an ellipse and a block moved across the cell's edge or corner, and two layers with no mirror along x -- six met them ({ref}`the measurement <wall-resolution-off-centre>`).
- **Against other codes without the stretch.** On dielectric blocks, in air and on glass, the unstretched Li solve, run in RETICOLO to order 30, agrees with it to that solve's own uncertainty. On metals the unstretched Li solve -- RETICOLO's and Wavebit's alike -- has not settled by order 40; that slowness is what the stretch removes.
- **What no one has.** No other implementation of the stretched or the matched solve was found to compare with at the same truncation.
- **Spurious modes.** A strong stretch is reported in the literature to add spurious layer modes. None were found at slopes down to 0.005 and orders 4 to 16. In a uniform medium every computed mode is a plane wave. At the slope 0.02 a patterned layer has no more growing roots than the default solve, and in a lossless layer those roots come in conjugate pairs that carry no net power. Turning the exceptional-point handling off moves `R` and `T` by at most 7.7e-11 ({ref}`the measurement <wall-resolution-spurious>`).
- **How to use it.** Compare two orders before relying on a value, as with any solve.

## What the value is

`G` is the slope of the solve's coordinates at a wall, a number in `(0, 1]`. With `x = F(u)` along each axis, `F'(u) = G` at every wall, so `0.02` places the coordinate lines fifty times more densely there than in the cell's own coordinates, and `1.0` on a structure whose walls are already evenly spaced is no stretch at all.

- **`0.02` is the recommended value.** It is the value the measurements were taken at.
- A weaker stretch still helps: `0.1` was measured on two of the layers and was closer than the unstretched solve on both, though not as close as `0.02`.
- The walls are placed at equal distances in the stretched cell, and one function per axis serves the whole stack: every wall of every layer is a wall of it. A wall is a rectangle edge with different materials on its two sides. Nothing about the placement is a keyword.
- Two rectangles of one material that touch make no wall where they touch, so a shape written as several touching rectangles is solved as the one shape. A rectangle may not cross the cell's edge; one that would is written as the pieces the edge cuts it into, and is then solved as the shape moved ({ref}`the measurement <wall-resolution-off-centre>`). Until 2026-11-28 every rectangle edge was a wall, and a block written as two halves moved the near field by 4.9%.
- `rcwa.check(...).resolved['wall_resolution']` reports the value, `None` when unset.
- For a circle or an ellipse the walls of the stretch are the four lines of the matched mesh ({ref}`below <wall-resolution-circles>`), and the same value applies there.

## What it accepts

Rectangles along the cell's axes, or one circle or axis-aligned ellipse ({ref}`below <wall-resolution-circles>`), with `factorization='li'` in a rectangular cell; isotropic, non-magnetic materials in the layers, metals included; lossless dielectric half-spaces; one incident plane wave from either side, at any angle; any number of layers, uniform ones included; the scattering-matrix algorithm (`algorithm='smm'`, the default) and the second-order formulation. NumPy, JAX and PyTorch, at double precision; on JAX and PyTorch, derivatives with respect to the materials, the wavelength and the incident angle, and on JAX `jax.jit` and `jax.vmap` over them ({ref}`below <wall-resolution-transforms>`).

Everything else is refused by name with the code `rcwa.unsupported`, at `rcwa.check` as at `rcwa.run`:

- another factorization (the default on regions is `'nvf'`, so `factorization='li'` has to be named), a raster, a polygon, a skewed cell;
- `algorithm='etm'`, `formulation='first_order'`;
- tensor materials, a material that names a permeability;
- a half-space that absorbs or has a negative permittivity;
- several incident waves or an internal source;
- an axis that has walls while `orders` retains a single harmonic along it;
- two walls too close for the stretch to hold between them (a piece of the stretched cell is `period / number of walls` wide whatever its physical width);
- `Outputs(scattering=True)`;
- `precision='single'`: the stretch makes the layer operators large where they meet the walls, and single-precision rounding then moved `R` and `T` by up to a few percent in the measurements ({ref}`why <wall-resolution-single-precision>`).
- a traced or differentiated geometry -- a size, centre, radius, angle or the cell -- under `jax.grad`, `jax.jit`, `jax.vmap` or a PyTorch tensor that requires grad, refused at its path ({ref}`below <wall-resolution-transforms>`);
- under `jax.jit` or `jax.vmap`, a traced half-space permittivity.

A rectangle as long as the period along one axis has no wall across that axis, and that axis is not stretched: a lamellar grating is stretched along its one direction.

(wall-resolution-transforms)=
## Derivatives, `jax.jit` and `jax.vmap`

The materials of the layers, the wavelength and the incident angle may be traced or differentiated; the geometry is read as numbers.

```python
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

def design(epsilon, wavelength):
    return rcwa.Problem(
        units=Units(length='nm'),
        stack=rcwa.Stack(period=(400.0, 400.0), layers=[rcwa.Layer(thickness=150.0, regions=[
            rcwa.Region(geometry=rcwa.Rectangle(size=(200.0, 200.0)), material=Material(epsilon=epsilon))])]),
        excitation=rcwa.Excitation(wavelength=wavelength, sources=[rcwa.PlaneWave(polarization='p', theta=0.35)]))

numerics = rcwa.Numerics(orders=(3, 3), factorization='li', wall_resolution=0.02)
prepared = rcwa.prepare(design(-20 + 1.5j, 900.0), numerics=numerics, backend='jax')
reflectance = lambda eps, wl: jnp.sum(prepared.run(design(eps + 1.5j, wl)).R)
value, slope = jax.jit(jax.value_and_grad(reflectance))(-20.0, 900.0)
sweep = jax.vmap(reflectance, in_axes=(None, 0))(-20.0, jnp.linspace(850.0, 950.0, 5))
```

- **What may move.** The permittivities of the regions and backgrounds, the wavelength and `theta`: under `jax.grad` (eagerly or inside `jax.jit`), `jax.jit`, `jax.vmap`, and PyTorch autograd. `phi` takes the same code and was not measured. A half-space's permittivity may be differentiated eagerly (measured once); under `jax.jit` or `jax.vmap` it has to be a constant `Material`.
- **What may not.** The coordinates are built from the walls' positions as numbers, so a size, centre, radius, angle or the cell that is traced or carries a derivative is refused with `rcwa.unsupported` at its path, on JAX and PyTorch alike. Write the geometry as numbers inside the transformed function, as above. A derivative with respect to the geometry would have to move the coordinates with the walls; it is not implemented.
- **Under `jax.jit` and `jax.vmap`** the run makes the decisions an eager run makes on numbers -- which orders propagate, which of the half-space's modes their plane waves replace -- as array operations, and agrees with the eager run to rounding. Where an eager run would refuse (too few orders to hold a half-space's plane waves), the transformed run answers NaN and `result.validity.checks['numerical_finite']` is false.
- **Touching rectangles of one material** make no wall, as without a transform. Under `jax.jit` or `jax.vmap` a traced material cannot be compared by value: one traced value written for several rectangles (the same variable, or the same `Material`) is one material, and two different traced values are two. Where two different traced values turn out equal on a touching edge, the run answers NaN rather than solve in coordinates the eager run would not take.
- **How it is checked.** Derivatives against central differences on a metal block and a metal circle, jitted and vmapped runs against eager ones, and JAX against PyTorch ({ref}`the measurements <wall-resolution-transforms-measured>`).

(wall-resolution-circles)=
## Circles and ellipses

```python
layer = rcwa.Layer(thickness=150.0, regions=[
    rcwa.Region(geometry=rcwa.Ellipse(radii=(150.0, 50.0)), material=Material(epsilon=-20 + 1.5j))])
numerics = rcwa.Numerics(orders=(12, 12), factorization='li', wall_resolution=0.02)
```

A stack whose patterned layers hold one circle, or one ellipse with its axes along the cell's, is solved in coordinates matched to it. A mesh carries a rectangle onto the ellipse (Weiss et al., Opt. Express 17, 8051 (2009), as restated by Weiss's thesis and by Küchenmeister, Zebrowski and Busch, Opt. Express 20, 17319 (2012)): the rectangle whose corners are the ellipse's 45-degree points, so that its four sides become the ellipse's four arcs. Outside it four side patches stretch only across the wall and four corner patches are left alone, and so is the cell's edge. In front of the mesh the same stretch as for rectangles crowds the coordinate lines at the rectangle's four sides, where they advance at the slope `G`. In these coordinates every layer, the uniform ones and the half-spaces included, is a full in-plane tensor.

- **What it takes.** One circle or one axis-aligned ellipse per patterned layer, the same one (centre and size) in every patterned layer, anywhere in the cell, across its edge as well; uniform layers between them. Each semi-axis has to be shorter than half the period along it, so that the mesh has room between the wall and the cell's edge. `factorization='li'` names it as it does for rectangles; the rest is as for rectangles.
- **What it refuses**, by name with `rcwa.unsupported`: several shapes in a layer, different circles or ellipses in two layers, a circle in a stack that also holds rectangles, a turned ellipse, an ellipse that reaches the cell's edge, and a single harmonic along either axis.
- **How the field is read.** The mesh's four lines run across the whole cell, and on them the map bends, so the component of the solve's field normal to a line jumps there even in a uniform medium; its plain sum rings. Within half a resolution unit, `period / (2 N + 1)`, of a line the reading takes that component from the normal displacement (`D` for `E`, `B` for `H`), which does not jump; beyond one and a half units it keeps the plain sum, and between the two it blends them. This is what `reconstruction='plain'` reads here; the other rules are refused, as for rectangles.
- **What it costs.** The materials are sampled on a raster of the stretched cell, 512 by 512 pixels whose edges are the mesh's lines (more when the orders need them), and go through the tensor Li factorization. Measured at order 12 on one layer, a solve took about 1.6 times the default NVF solve's time.
- **How close it is.** On six metal circles, three metal ellipses and three dielectric shapes at normal incidence the field is several to tens of times closer to the finite-element reference than the default solve's at the same order, and `R` at order 12 within 0.2% of the reference's on all but the weakest metal ({ref}`the measurements <wall-resolution-circles-measured>`). The 3:1 ellipse, on which the default solve's `R` is 12% off at order 12, is within 0.01%. A metal circle at theta 20 and 40 deg, on glass, in a 400 by 300 nm cell, in two layers and at 350 nm, and a 130 by 80 nm metal ellipse at theta 20 and 40 deg, on glass and in two layers, are within 0.28% in `R` and `T`.

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
- **A weak metal's `R` and `T`.** On a circle of permittivity -5+1j the matched solve's `R` is 1.2% above the reference's at order 12, where the field is ten times closer than the default solve's; a block of the same material was within 0.6%. If only the efficiencies of a dielectric or weakly metallic structure are wanted, the unstretched solve is the checked one.
- **An `ExceptionalPointWarning` at an isolated wavelength.** In sweeps of 138 wavelengths the stretched solve warned at three points, from its stretched air half-spaces, where the default solve did not. There the answer was the same to the bit with the handling off and lay on the curve through the neighbouring wavelengths. Check such a point against its neighbours or a second order before discarding it.
- **Many walls.** Each piece between two walls gets the same share of the stretched cell, so a structure with many walls along an axis leaves each piece fewer harmonics. With four walls along each axis, a metal frame read `R` and `T` 0.35 and 0.59% off a finite-element reference at order 12 and within 0.10% at order 16. With four along one axis, two metal bars in p read a small `R` (0.07) 1.0% off at order 12, 0.44% at order 16 and 0.09% at order 20. Compare such a layer at two orders. More than four walls along an axis was not measured.
