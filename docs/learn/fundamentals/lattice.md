# 06. Beyond a square: lattices and two coordinate systems

> **Historical API:** snippets and descriptions using `wavebit.compat` on this page require a historical checkout.
> The namespace is absent from current installations; the pre-retirement baseline is
> [revision `3c8a50f8`](https://github.com/fieldalgo/wavebit/tree/3c8a50f8a6506240169538957e99648a4b18ca3a).
> Reproduce recorded experiments with their own cited commit and environment; their measurements have not been rerun.


:::{note}
The downloadable teaching scripts use the current physical Problem/Job API. Some explanatory snippets and figures retain earlier API spellings; compare the [current getting-started guide](../../getting-started/index.md) and [migration guide](../../migration/index.md). Historical measurement files and their source provenance remain unchanged; they are distinct from validation of the migrated scripts.
:::

**A raster assigns material to lattice cells; a shape defines an object in physical space.** Below, both side lengths stay at 600 nm while only the angle `α` between the two lattice vectors changes from 90° to 60°. We are changing the angle between the sides, which is different from rotating the entire structure as one object.

`period=(600., 600.)` specifies an orthogonal cell. An oblique cell is represented by a `Lattice` with two primitive vectors. `period` and `lattice` are two representations of one cell, so do not supply both.

```python
A90 = wavebit.Lattice.rectangular(600., 600.)
A60 = wavebit.Lattice.oblique(600., 600., np.deg2rad(60.))
X, Y = A60.meshgrid_xy((64, 96))
# X and Y: physical x, y coordinates of pixel centers in the 60° cell, in nm
```

## Application: repeating cylinders on a triangular lattice

**An oblique lattice can represent a triangular array of cylinders with a single small cell.** Such an array is a standard photonic-crystal example; the [official MPB tutorial](https://mpb.readthedocs.io/en/latest/Python_Data_Analysis_Tutorial/#triangular-lattice-of-rods) also studies this type of array. The geometry diagram below was drawn for Wavebit's coordinate representation.

In a triangular array, each cylinder has **six nearest neighbors**, all at center-to-center distance `a`. Choose translation vectors `a₁` of length `a` to the right and `a₂` of length `a` at 60° above the horizontal to represent this arrangement directly.

```{figure} ../../figures/learn/learn/06d-triangular-cell.svg
:alt: The same triangular array is represented by a 60-degree oblique cell containing one cylinder on the left and a rectangular cell containing two cylinders on the right. The oblique cell has half the area; all six nearest-neighbor center distances are a.
:figclass: learning-figure

Figure 6-1. The same triangular array is represented by a 60-degree oblique cell containing one cylinder on the left and a rectangular cell containing two cylinders on the right. The oblique cell has half the area; all six nearest-neighbor center distances are a.
```

The cylinder arrangement is identical in both diagrams. The left uses a **60° rhombic cell containing one cylinder**, while the right uses a **rectangular cell containing two cylinders**. The right-hand cell combines two primitive cells: it is a supercell. The advantage of the oblique lattice here is that it directly specifies the smallest repeating unit.

| Representation of the same triangular array | 60° oblique cell | Rectangular cell |
| --- | --- | --- |
| Translation vectors | `a₁=(a,0)`, `a₂=(a/2, √3a/2)` | `(a,0)`, `(0,√3a)` |
| Cylinders per cell | 1 | 2 |
| Cell area | `√3a²/2` | `√3a²` |

The repeating cell and cylinder cross-section on the left are entered as follows. The angle is in radians, and the circle diameter is a physical xy length.

```python
import numpy as np
from wavebit.compat import legacy as wavebit

a = 600.  # Nearest-neighbor center distance, nm
lattice = wavebit.Lattice.oblique(a, a, np.deg2rad(60.))
structure = [{
    "background": 1.,
    "shapes": [{
        "type": "circle", "cx": 0., "cy": 0.,
        "lx": 240., "material": 4.,  # Diameter 240 nm, relative permittivity 4
    }],
}]
# Pass to rcwa(lattice=lattice, structure=structure, thickness=[...], ...)
```

`Lattice.hexagonal(a, np.deg2rad(60.))` creates the same cell. The default angle of `Lattice.hexagonal(a)` is 120°, so specify 60° to obtain the two vectors shown above. The 60° and 120° bases generate the same triangular lattice, but truncation at the same order indices does not retain the same set of Fourier wavevectors.

## Diagram A: how does the same lattice position move in physical space?

`u` and `v` specify relative position within the cell. If one cell spans 0 to 1, `(u,v)=(0.5,0.5)` is its center. Subtract `1/2` from each coordinate to place the cell center at physical position `(x,y)=(0,0)`.

```text
r = (u − 1/2) a₁ + (v − 1/2) a₂

Example: (u, v) = (0.75, 0.75)
    → From the cell center, move 1/4 along a₁ and 1/4 along a₂
    → r = ¼a₁ + ¼a₂
```

```{figure} ../../figures/learn/learn/06-lattice.svg
:alt: Keeping the lattice coordinates P=(0.75,0.75) fixed while changing the angle from 90° to 60° moves the physical position from (150,150) to (225,129.9) nm. The blue and green arrows are ¼a₁ and ¼a₂.
:figclass: learning-figure

Figure 6-2. Keeping the lattice coordinates P=(0.75,0.75) fixed while changing the angle from 90° to 60° moves the physical position from (150,150) to (225,129.9) nm. The blue and green arrows are ¼a₁ and ¼a₂.
```

On the left, `a₂` points upward; on the right, it tilts upward and to the right. Adding the same `¼a₂` therefore reaches a different position. **Keeping the same array index does not mean keeping the same physical xy position.**

## Diagram B: a circle with a fixed array versus a circle with a fixed diameter

The blue pattern at the top approximates a circle with a 24×24 raster. **Exactly the same array elements contain material in both diagrams.** However, the cells change from squares to parallelograms, deforming the pattern in physical space. Applying this transformation to an ideal smooth circle produces an ellipse; the diagram shows its stepped approximation.

The green circle below is a shape with `cx=0, cy=0, lx=240`. Its diameter of 240 nm is a physical xy length, so it remains circular when the cell tilts.

```{figure} ../../figures/learn/learn/06b-lattice-shapes.svg
:alt: Top: fixing the same raster array deforms both its cells and the circle pattern. Bottom: fixing a shape with physical diameter 240 nm preserves the circle while changing its repetition lattice. All four scenes use the same xy scale.
:figclass: learning-figure

Figure 6-3. Top: fixing the same raster array deforms both its cells and the circle pattern. Bottom: fixing a shape with physical diameter 240 nm preserves the circle while changing its repetition lattice. All four scenes use the same xy scale.
```

**Even in the lower diagrams, the entire structure does not stay unchanged.** The central circle remains at `(0,0)`, but the copies in other cells move with the lattice. A neighbor displaced by `a₂` lies at `(0,600)` nm for the 90° lattice and approximately `(300,519.6)` nm for the 60° lattice. The center distance remains 600 nm, but the direction changes. Consequently, the optical response can change even when the size of one circle is fixed.

| What changes from 90° to 60°? | Same raster array | Fixed physical center and diameter of the circle |
| --- | --- | --- |
| Cell and repetition of neighboring patterns | Changes | Changes |
| Physical shape of the object within the cell | Deforms with the cells | Remains circular |
| Physical cross-sectional area of the object per cell | About 86.6% of its initial value | Unchanged circle area |
| Fraction of the cell occupied by the object | Unchanged | About 1.155 times its initial value |

These ratios follow directly from geometry, not an optical calculation. With fixed side lengths, the cell area is `600²·sin α`. A fixed raster retains the fraction of colored cells, while a fixed circle retains its own area. The circle in this example does not touch the cell boundary, so clipping and overlap with neighbors need not be considered.

## Why “differentiate the lattice angle” can mean different questions

Imagine changing the angle slightly and asking for the transmission derivative `dT/dα`. **The rule used to construct the next structure** determines what this derivative means.

| Quantity held fixed | Physical question |
| --- | --- |
| Every raster-array element | How much does T change when cells and objects deform together? |
| Physical center and diameter of the circular shape | How much does T change when repetition changes but the circle size stays fixed? |

These are different physical variations, so their gradients may differ. To optimize the lattice angle while preserving the circle size, specifying the shape in physical lengths expresses the intended question directly.

With a raster, you can also recompute the physical `X, Y` coordinates at each angle and redraw the circle. This is **different from keeping the same array**. Moreover, rebuilding a binary raster by abruptly flipping pixels with a condition such as `X²+Y²<r²` does not immediately yield a smooth shape derivative. See the [oblique-lattice document](../../theory/rcwa/oblique-lattice.md) for derivative scope and comparisons.

(c-1-0-x)=
## Diagram C: why does each Fourier component have two indices?

So far we have considered where shapes repeat along a lattice. Next, consider **how the calculation represents that periodic distribution**. As in Chapter 2, RCWA expresses a complicated distribution as a sum of simple wave-like components. The indices distinguishing those Fourier components are called **orders**.

**1. First, consider where zeroth, first and second orders come from in one direction.** Count how many spatial oscillations fit in a 600 nm cell. We are counting variation with position, not oscillation in time.

| Variation of the basis component within one cell | Order index |
| --- | --- |
| Constant with position | 0 |
| One oscillation over 600 nm | 1 |
| Two oscillations over 600 nm | 2 |

Moving by one cell must continue the same periodic component, so the number of oscillations in a cell is an integer. This explains the indices 0, 1 and 2. Negative orders describe components whose phase—the stage of their oscillation—varies in the opposite direction. **An index does not specify the component's magnitude.** A separate coefficient determines how much of that component is added.

**2. Repetition in two directions requires two indices.** Write `m` for the number along the first translation vector `a₁` and `n` for the number along the second vector `a₂`, and combine them as `(m,n)`.

| Component index `(m,n)` | Moving one cell along `a₁` | Moving one cell along `a₂` |
| --- | --- | --- |
| `(0,0)` | No variation | No variation |
| `(1,0)` | One oscillation | No variation |
| `(0,1)` | No variation | One oscillation |
| `(2,0)` | Two oscillations | No variation |
| `(1,1)` | One oscillation | One oscillation |

**We will use the `(1,0)` component from the table as our example.** The first index, `1`, means one oscillation along `a₁`; the second, `0`, means no variation along `a₂`. A zero index does not mean that the component is absent. `(1,0)` identifies a component; it is not a point's physical coordinates `(x,y)`.

At this stage we are considering the Fourier components of the periodic part. Actual diffracted waves also include the incident wave's phase variation, which we connect to this picture in C2 below.

**3. Connect these indices to physical xy directions.** The phase of the `(1,0)` component does not change anywhere along `a₂`. Lines joining equal-phase points are therefore parallel to `a₂`. Moving along one line preserves phase; crossing the lines changes it.

The arrow representing the direction and rate of fastest phase increase is the wavevector `b₁`. It crosses equal-phase lines normally, so `b₁` is perpendicular to `a₂`.

```{figure} ../../figures/learn/learn/06c-phase-lines.svg
:alt: Both diagrams show the physical xy plane. Equal-phase lines are parallel to a2, and b1 is perpendicular to them. In an orthogonal cell b1 points right; in a 60-degree oblique cell it points down and right. These lines are not material boundaries.
:figclass: learning-figure

Figure 6-4. Both diagrams show the physical xy plane. Equal-phase lines are parallel to a2, and b1 is perpendicular to them. In an orthogonal cell b1 points right; in a 60-degree oblique cell it points down and right. These lines are not material boundaries.
```

On the left, equal-phase lines are vertical, so `b₁` points right. On the right, they slope at 60° upward and to the right, so the perpendicular `b₁` points at −30°, downward and to the right. **Even if `a₁` points right, `b₁` need not point right.** `b₁` and `b₂` are called reciprocal-lattice vectors: the basic spatial-frequency units corresponding to the repetition lattice.

**First, a definition: what is a transverse wavevector?**

Here, the **transverse wavevector groups the x and y components of the wavevector**. “Transverse” means perpendicular to the layer-stacking z axis: the **xy directions parallel to the layer surfaces**.

**Full wavevector:** k = (k<sub>x</sub>, k<sub>y</sub>, k<sub>z</sub>)

**Transverse wavevector:** k<sub>∥</sub> = (k<sub>x</sub>, k<sub>y</sub>)

A wavevector describes how quickly the wave's phase changes with displacement. At fixed time and height z, `kₓ` is the phase variation per distance in x, and `kᵧ` is the corresponding variation in y. The `∥` symbol denotes components parallel to the layer surface. Here `k∥` is a two-component vector, not just its magnitude.

Consider a plane wave incident through a uniform isotropic medium.

| Incidence | Incident transverse wavevector | Viewed along a surface at fixed height |
| --- | --- | --- |
| Normal to the layers | `kₓ=0`, `kᵧ=0` | Phase is unchanged when moving in x or y |
| Oblique in the xz plane | `kₓ≠0`, `kᵧ=0` | Phase changes when moving in x |

At normal incidence, a zero transverse wavevector still permits light to propagate in z. It means that phase does not vary along the surface.

The **Δ symbol below means a difference**. `Δk∥=(Δkₓ, Δkᵧ)` is the order-dependent increment added to the incident wave's x and y wavevector components.

**4. The same indices distinguish diffracted waves.** The wavevector corresponding to the `(1,0)` component is `b₁`, and that corresponding to `(0,1)` is `b₂`. A general `(m,n)` component corresponds to `m b₁ + n b₂`. Add the incident transverse wavevector to obtain the actual diffracted wave's transverse wavevector.

| Order `(m,n)` | Increment `Δk∥` added to the incident transverse wavevector |
| --- | --- |
| `(0,0)` | `0` |
| `(1,0)` | `b₁` |
| `(0,1)` | `b₂` |
| `(1,1)` | `b₁ + b₂` |

**First, consider the meaning of 2π/L.**

Let one side of the lattice have length `L`. For a component that completes one oscillation while moving by `L` along that side, the phase changes by one cycle, or `2π` radians. Thus the **phase change per unit length is 2π/L**.

```text
Phase change in one oscillation = 2π
Distance traveled              = L
Phase change per unit length   = 2π/L

For L = 600 nm:
2π/L = 2π/(600 nm) ≈ 0.010472 nm⁻¹
```

Thus `2π/L` is the **reference value for one full phase cycle over distance L**. Here `L` is the structural repeat length. Phase is measured in radians, which are dimensionless, so using nm for length gives units of `nm⁻¹`.

When `a₁` points along +x and `a₂` tilts at 60°, as above, the reciprocal vector of the `(1,0)` component can be written directly using this quantity. Because `b₁` points at −30°, perpendicular to `a₂`, its y component is `−1/√3` times its x component.

```text
b₁ = (2π/L, −2π/(√3 L))
```

**Now introduce the name g to abbreviate the repeated factor 2π/L.** The two expressions below denote the same vector.

```text
g = 2π/L

b₁ = (2π/L, −2π/(√3 L))
   = (g,    −g/√3)
```

Diagram C2 below uses this abbreviation. `g` is a scalar; `b₁` is a vector with direction and magnitude. In this oblique lattice, `g` is the x component of `b₁`, not its full magnitude.

```{figure} ../../figures/learn/learn/06c2-order-wavevectors.svg
:alt: In the wavevector-difference coordinates on the right, order (1,0) adds b1 once. For the 60-degree cell, b1 has positive x component g and negative y component minus g divided by the square root of 3. Add the incident transverse wavevector to obtain the actual transverse wavevector.
:figclass: learning-figure

Figure 6-5. In the wavevector-difference coordinates on the right, order (1,0) adds b1 once. For the 60-degree cell, b1 has positive x component g and negative y component minus g divided by the square root of 3. Add the incident transverse wavevector to obtain the actual transverse wavevector.
```

In C2, `(1,0)` denotes a wavevector difference with x component `+g` and y component `−g/√3`. Reading `(1,0)` simply as “first order in physical x” misses this y component.

C1 plots **equal-phase lines in physical xy space**, while C2 plots **transverse wavevector differences**. Do not read the arrows in C2 directly as outgoing light directions. An actual diffracted wave includes the incident transverse wavevector.

```text
Order-dependent increment:       Δk∥ = m b₁ + n b₂
Diffracted transverse wavevector: k∥(m,n) = k∥(incident) + Δk∥
```

To interpret the outgoing direction, also determine whether that order can propagate in the medium. In code, retrieve results by order index, as in `result.order(1,0)`. `orders_u` corresponds to `m`, and `orders_v` to `n`. Unlike `u, v`, which previously denoted position within the cell, `m, n` are integer order indices.

Algebraically, `b₁·a₁=2π` and `b₁·a₂=0`. The second equation is exactly the perpendicular relation shown above; the first says that moving by `a₁` changes phase by one cycle. Conversely, `b₂` corresponds to a component whose phase is constant along `a₁` and changes by one cycle along `a₂`.

**How should g be defined when the two side lengths differ?**

In the diagrams above, the common constant `g=(2π/600) nm⁻¹` abbreviates formulas for an example with two 600 nm sides. If their lengths are `L₁` and `L₂`, define separate scales `g₁=2π/L₁` and `g₂=2π/L₂`. The actual reciprocal vectors must also account for **the angle between the sides**.

As in this chapter's diagrams, suppose `a₁` points along +x and `a₂` is rotated counterclockwise by `α` from +x. Then:

```text
Lattice vectors:    a₁ = (L₁, 0)
                    a₂ = (L₂ cos α, L₂ sin α)

Basic scales:       g₁ = 2π/L₁,  g₂ = 2π/L₂

Reciprocal vectors: b₁ = (g₁, −g₁ cos α / sin α)
                    b₂ = (0, g₂ / sin α)
```

For example, with **side lengths 600 nm and 800 nm and an included angle of 60°**:

```text
g₁ = (2π/600) nm⁻¹ ≈ 0.010472 nm⁻¹
g₂ = (2π/800) nm⁻¹ ≈ 0.007854 nm⁻¹

b₁ = (g₁, −g₁/√3) ≈ (0.010472, −0.006046) nm⁻¹
b₂ = (0, 2g₂/√3)   ≈ (0,         0.009069) nm⁻¹
```

`g₁` and `g₂` are the basic scales used in these expressions. For an oblique lattice, they differ from the magnitudes of the reciprocal vectors themselves. For `0°<α<180°`, those magnitudes are `g₁/sin α` and `g₂/sin α`. At `α=90°`, the orthogonal case simplifies to `b₁=(g₁,0)` and `b₂=(0,g₂)`.
