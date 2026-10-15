# 04. Boundary normal n → auxiliary vector J → projector P

:::{note}
The downloadable teaching scripts use the current physical Problem/Job API. Some explanatory snippets and figures retain earlier API spellings; compare the [current getting-started guide](../../getting-started/index.md) and [migration guide](../../migration/index.md). Historical measurement files and their source provenance remain unchanged; they are distinct from validation of the migrated scripts.
:::

**n describes the boundary direction, J extends that information across the domain as a computational reference, and P extracts components using that reference.** The electric field E is solved afterward, using equations constructed from this information. We do not first calculate the electric field to discover the normals.

```text
Shape and material information → boundary normal n → auxiliary vector J throughout the domain
                                                        ↓
                                                    projector P
                                                        ↓
                                    Fourier coefficients and material matrices
                                                        ↓
                           solve with incident-light conditions → electric field E
```

## n: the actual normal defined at a smooth shape boundary

A normal is perpendicular to the boundary. In this course's xy cross sections it is represented by two numbers, `n=(nx, ny)`, usually as a unit vector satisfying `nx²+ny²=1`. For a circle centered at the origin, the rightmost point has `n=(1,0)`, the top has `n=(0,1)`, and the point at 45° in the upper right has `n=(1/√2, 1/√2)`. All follow from the shape alone.

**The center of a circle has no actual boundary normal**, because it is not on the boundary. A nonsmooth boundary point, such as a rectangle's corner, also has no unique normal. Directions may be assigned inside a shape, but they must be distinguished from actual boundary normals.

## J: an auxiliary vector extending boundary-normal information throughout the domain

NVF stands for Normal Vector Field. Fourier calculations in RCWA require **a decomposition reference defined throughout a periodic domain**, not just at a few boundary points. We therefore construct a reference matching the boundary normals and fill the shape interior and exterior background with a computational vector J. This course distinguishes actual boundary normals n from the extended auxiliary vector J. Implementations and other references may call the extended field n as well, so check the definition rather than the symbol alone.

| Symbol | Meaning | Where and how it is defined | Is the electric field E needed? |
| --- | --- | --- | --- |
| `n` | Actual unit normal at a boundary | Determined by the geometry of a smooth material boundary | No |
| `J` | Auxiliary vector defining a component-decomposition reference | Matched to boundary normals and extended throughout the domain | No |
| `P` | A 2×2 projector applying that reference | Calculated from the components of J | No |

```{figure} ../../figures/learn/nvf/04-field.svg
:alt: Boundary n, domain-wide J, and component-extracting P play different roles
:figclass: learning-figure

Figure 4-1. Boundary n, domain-wide J, and component-extracting P play different roles
```

At a boundary, J is constructed to produce the same projection as n. In a real-valued example we can set `J=n`; in a complex representation, multiplying n by a common complex phase also gives the same projection. **Inside a region, J need not be an actual normal.** It is computational information constructed with periodicity, multiple boundaries, and smooth connections in mind.

## P: a matrix combining the component decomposition defined by J

J is a vector; P is a matrix. For nonzero J, construct P as follows. `H` denotes the conjugate transpose: complex conjugation followed by exchanging rows and columns. For real values it is the ordinary transpose.

```text
P = J Jᴴ / (Jᴴ J)

If J has been normalized to unit length:
J = (a, b)ᵀ,  |a|² + |b|² = 1

P = [ |a|²    a b*  ]     * denotes complex conjugation
    [ b a*    |b|²  ]
```

Extracting a component with unit-length J takes two steps. First calculate its coefficient `c = JᴴE`. The scalar c may be complex. Then calculate `Jc` to turn that component back into a vector. For unit J, the combined operation is `J(JᴴE) = (JJᴴ)E = PE`. **P does not choose another direction; it packages the calculation defined by J.**

The following hand calculation shows a horizontal decomposition. The example E is supplied to demonstrate what P does; it was not used to obtain the normal.

```text
J = (1, 0)ᵀ  →  P = [ 1  0 ]
                     [ 0  0 ]

Example E = (3, 2)ᵀ  →  PE = (3, 0)ᵀ
```

```{figure} ../../figures/learn/nvf/05-projector.svg
:alt: J defines P, and applying P retains the selected component
:figclass: learning-figure

Figure 4-2. J defines P, and applying P retains the selected component
```

P can be constructed before E is known. Wavebit converts the spatially varying `Pxx, Pxy, Pyx, Pyy` to Fourier coefficients, then combines them with coefficients of ε and 1/ε to construct RCWA material matrices. E follows from solving the equations containing those matrices. **The method does more than decompose an already computed E afterward.** Chapter 5 examines the assembly formula.

## Why the tangential part is I−P

I is the identity matrix, which leaves a vector unchanged. Subtracting the selected component from the whole leaves the remaining component: `(I−P)E = E−PE`. At a smooth boundary, PE is the normal component and the remainder is tangential in the xy plane.

```text
E = PE + (I−P)E
(3, 2)ᵀ = (3, 0)ᵀ + (0, 2)ᵀ
            normal    tangential   ← example of a vertical boundary
```

This decomposition holds for complex J as well. Away from the boundary, however, the two parts should be read as **computational components extending the boundary decomposition into the interior**, rather than actual normal and tangential components. The normalized projector for nonzero J satisfies `P²=P`, `Pᴴ=P`, and `trace(P)=1`. Zero vectors and degenerate points in actual code require separate handling beyond the conditions under which this formula applies.

For real normals, `P=nnᵀ` and `P(n)=P(−n)`: opposite signs give the same decomposition. Adding opposite vectors before constructing the projector can nevertheless give zero. This property alone therefore does not solve cancellation when multiple vectors are combined.

```{raw} html
<section class="exercise" aria-labelledby="projection-title"><h3 id="projection-title">Try it · What happens to P when the boundary normal n = J rotates?</h3><p class="explain">This example sets J equal to the real unit boundary normal n, then constructs P=JJᵀ. P is determined without supplying E. Compare θ with θ+180°.</p><div class="controls"><label for="angle">Normal angle θ (degrees) <output id="angle-value" for="angle">30°</output><input id="angle" type="range" min="-180" max="180" step="1" value="30"></label></div><svg class="mini-svg" viewBox="0 0 400 260" role="img" aria-label="Normal direction as a function of angle"><circle cx="200" cy="130" r="90" fill="white" stroke="#cad7df"/><path d="M80 130H320 M200 20V240" stroke="#cad7df"/><line id="normal-line" x1="200" y1="130" x2="278" y2="85" stroke="#117c77" stroke-width="4"/><circle id="normal-tip" cx="278" cy="85" r="7" fill="#117c77"/><text x="330" y="135" font-size="16">x</text><text x="208" y="23" font-size="16">y</text></svg><div id="projection-result" class="readout" aria-live="polite">θ=30°: Pxx=0.7500, Pxy=Pyx=0.4330, Pyy=0.2500.</div><noscript><p>The equations and the 0° and 90° examples in Chapter 4 illustrate the same principle.</p></noscript></section>
```

## Linear and circular: two ways to construct the interior decomposition reference

In the basic real-material example, **linear extends real line directions to construct a projector**. J matches the boundary normal and represents a real direction in the interior too. Wavebit's shape path for complex materials can use complex components and Hermitian projection even with linear, so these API names must not simply be interpreted as “real array” versus “complex array.”

**Circular is a separate construction using a complex auxiliary vector.** At the boundary it produces the same projection as the actual normal; inside, relative magnitudes and phases between its two components help connect the decomposition reference. It neither uses a purely circular basis at every point nor applies only to circular shapes.

## Why the difference appears at a circle's center

Consider a real-valued example that extends a circle's normals straight inward. Approaching the center from the right leaves a horizontal direction; approaching from above leaves a vertical direction. Assigning an arbitrary vector at the center cannot remove **the different limiting projectors obtained along different approaches**. Opposite approaches from the right and left produce the same projection, which is why this example compares the right with the top.

```{figure} ../../figures/learn/nvf/04b-center.svg
:alt: A circle's center has no actual normal, but a complex auxiliary vector can connect the projector smoothly
:figclass: learning-figure

Figure 4-3. A circle's center has no actual normal, but a complex auxiliary vector can connect the projector smoothly
```

Allowing complex values makes an extension using `J=(1,i)/√2` at the center possible. This J is not a single fixed real arrow, but its two components and the resulting P are well defined.

```text
One choice at the center:
J = (1, i)ᵀ / √2

P = ½ [ 1  −i ]
      [ i   1 ]
```

**“Joining at the center” means joining to the same P, not the same real arrow.** This P is not simply `I/2`. The off-diagonal ±i/2 entries are needed to represent this complex decomposition. The i encodes a 90° phase relation between the auxiliary basis's x and y components; it does not mean that the actual incident light rotates.

The figure is an educational model of one possible extension. On a unit disk, set `v=(x,y)ᵀ`, `r²=x²+y²`, `M=vvᵀ+(1−r²)I`, and `J=M(1,i)ᵀ`. At the boundary, `M=nnᵀ`, so P is the actual normal projector. At the center, `M=I`, giving the complex projector above. This is not a plot of a field actually computed by Wavebit or a literal rendering of its implementation throughout a periodic cell.

## Where Wavebit uses this, and what it is independent of

This projection construction is used in **both raster NVF and shape NVF**. Raster estimates a direction field from a pixel material pattern; shape constructs it from boundary geometry. Both subsequently incorporate the Fourier components of P into the material matrices. The Li path does not construct this NVF projector field.

Implementation details differ between the two inputs. Raster circular combines a direction and its perpendicular with a relative phase i, using field magnitude information to determine their mixture. Shape circular builds a matrix from positively weighted boundary-normal dyads `nnᵀ`, then forms a complex combination corresponding to J and constructs the projector. J is a shared explanatory symbol; it does not mean every code path must explicitly store an array named J.

**`nvf_projector='linear'` and `'circular'` are not incident-polarization settings.** Incident light is specified separately with settings such as `psi`, `chi`, and `polarization`. You can compare NVF projectors while keeping the same linearly polarized incident light. Circular is not always more accurate either; compare convergence of the quantity you need at finite Fourier order.
