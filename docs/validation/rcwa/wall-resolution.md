# The solve in stretched coordinates against finite-element fields — measurements

What [`Numerics(wall_resolution=G)`](../../guide/rcwa/wall-resolution.md) was measured against, what came out, and where it has not been checked. The option is experimental: the stretched solve has no independent implementation to be compared with at the same truncation, so everything here is either a comparison with a finite-element field, which says how accurate the method is, or a property the construction has to satisfy.

## The method in one paragraph

Along each axis the solve uses a coordinate `u` with `x = F(u)`, where `F` is, between each pair of neighbouring walls, a straight line plus one period of a sine (Granet's adaptive spatial resolution in the form of Vallius and Honkanen, Opt. Express 10, 24 (2002)). `F'` equals the option's value `G` at every wall, and the walls sit at equal distances in `u`. In `(u, v, z)` Maxwell's equations keep their form with `eps~ = eps diag(g/f, f/g, f g)` and `mu~` alike (`f = F'`, `g` the other axis's slope), so each layer is solved by the same eigenproblem as before with slope-weighted Fourier integrals. The whole stack is solved in these coordinates, the half-spaces included: their modes are computed like a layer's, and only the propagating orders are replaced by the exact plane waves written in the stretched basis.

## Near fields on seven layers

One layer of thickness 150 nm in a 400 nm square cell, in air, lit at normal incidence at 900 nm with `E` along x; `factorization='li'`. Each entry is the RMS of the in-plane vector error of `E` over 2304 samples of the mid-height plane, wall samples included, over the reference's own plane RMS, at orders 8 / 12 / 16. The references are 3D finite-element fields (gyptis 1.2.0, second-order edge elements).

| Layer | Unstretched | `wall_resolution=0.02` | `R`: stretched (reference) |
| --- | --- | --- | --- |
| metal block 200 nm, `eps` -20+1.5j | 32.1 / 25.8 / 22.5% | 4.0 / 0.63 / 0.52% | 0.35555 / 0.35553 / 0.35547 (0.35615) |
| square hole 200 nm in a film of that metal | 36.8 / 30.9 / 29.0% | 6.5 / 3.3 / 1.8% | 0.89270 / 0.89331 / 0.89344 (0.89281) |
| dielectric square 200 nm, `eps` 12.25 | 14.8 / 11.5 / 9.1% | 0.71 / 0.41 / 0.39% | 0.03650 / 0.03650 / 0.03650 (0.03662) |
| metal rectangle 250 by 150 nm | 37.9 / 36.4 / 26.5% | 4.5 / 3.6 / 3.4% | 0.63826 / 0.63888 / 0.63885 (0.64043) |
| metal block, `eps` -100+5j | 85.3 / 34.7 / 27.4% | 5.0 / 3.3 / 0.70% | 0.28785 / 0.28750 / 0.28783 (0.28827) |
| metal block, `eps` -5+1j | 21.8 / 17.0 / 13.4% | 2.8 / 1.4 / 1.0% | 0.37259 / 0.37308 / 0.37361 (0.37454) |
| dielectric rectangle 250 by 150 nm | 16.0 / 12.0 / 9.8% | 1.7 / 1.6 / 1.6% | 0.07651 / 0.07651 / 0.07651 (0.07671) |

- **The first four references are stored** (`tests/references/golden_gyptis_rectangles_field.npz`, with its record) and the order 8 rows are tests; the order 12 rows are in the slow tier. The unstretched order 16 entries of those four were measured with a test implementation of the same solve. The last three references are not stored in this repository, and no test reads them.
- **How good the references are.** The block's and the hole's coarser mesh levels are 0.37% and 0.20% of the plane RMS from the stored ones. The other five have one mesh level. Where the stretched solve stops improving -- 3.4% on the metal rectangle, 1.6% on the dielectric one, 0.4% on the dielectric square -- its `R` is settled to five digits and 0.2 to 0.3% from the reference's, and what is left sits beside the walls outside the shape. That floor is taken to be those references' accuracy at one mesh level; it was not confirmed with a finer mesh.
- **The hole** is the slowest of the seven: 6.5% at order 8 and 1.8% at order 16.
- **The unstretched solve's `R`** on the metal block is 0.3505, 0.3506, 0.3512 at the three orders; the stretched one is within 0.2% of the reference from order 8.

### Criteria written before the implementation

These were set from a test implementation's values with about a factor of two of margin, before the product code existed. At order 12, as a fraction of the unstretched plane error:

| Criterion | Limit | Measured |
| --- | --- | --- |
| metal block, field | 0.06 | 0.024 |
| metal block, `R` against the reference | 0.3% | 0.17% |
| dielectric square, field | 0.1 | 0.036 |
| dielectric square, `R` (from order 8) | 0.5% | 0.33% |
| metal rectangle, field | 0.25 | 0.10 |
| hole, field | 0.2 | 0.11 |

At order 16 the block is at 0.023 and the rectangle at 0.13. One criterion is not met as it was written: the three backends were to agree to 1e-10, and they do in `R` and `T` (3e-11 at worst) but the field of a metal layer at 350 nm differs by 4e-10 between them.

## What the construction has to satisfy

Each of these is a test in `tests/rcwa/test_wall_resolution.py`.

- **The limit without a stretch.** With `wall_resolution=1.0` on a block half the period wide, whose walls are already evenly spaced, the coordinates are the cell's own. `R`, `T`, the specular amplitudes and all six field components then agree with the unstretched solve to 1e-12 -- on four layers over glass, two of them uniform, lit obliquely from either side -- although the half-spaces and the uniform layers went through an eigenproblem instead of plane waves.
- **Moving the structure.** A rectangle moved by (37.5, -25) nm in the cell gives the same `R` and `T` to 1e-11 and the same field at the moved samples. The walls' places in the stretched cell depend on their order alone.
- **Energy.** On the lossless block `R + T = 1` to 1e-4 at order 8, at normal and oblique incidence and with four and five propagating orders.
- **Backends.** NumPy, JAX and PyTorch, eager, agree to 3e-11 in `R` and `T` and to 4e-10 of the field's scale.

## Oblique incidence, several propagating orders

There is no finite-element reference here: the model above is for normal incidence at one wavelength. What can be seen is whether the stretched solve settles, and whether the unstretched one moves toward it as its order rises. The 200 nm block (or hole) again; `R` at orders 8 / 12 / 16, and for the unstretched solve also at 20.

| Case | Propagating orders | `wall_resolution=0.02` | Unstretched |
| --- | --- | --- | --- |
| dielectric, 900 nm, theta 20 deg, phi 30 deg, p | 1 | 0.02594 / 0.02594 / 0.02594 | 0.02615 / 0.02605 / 0.02601 / 0.02599 |
| the same, s | 1 | 0.05394 / 0.05393 / 0.05393 | 0.05418 / 0.05406 / 0.05402 / 0.05399 |
| dielectric, 350 nm, normal | 5 | 0.24577 / 0.24502 / 0.24497 | 0.23188 / 0.23990 / 0.24262 / 0.24372 |
| dielectric, 350 nm, theta 20 deg, p | 4 | 0.12974 / 0.16194 / 0.16203 | 0.13188 / 0.15125 / 0.15719 / 0.15948 |
| dielectric, 405 nm, normal | 1 | 0.02123 / 0.31136 / 0.31217 | 0.14721 / 0.27412 / 0.30218 / 0.31042 |
| dielectric, 395 nm, normal | 5 | 0.63451 / 0.64034 / 0.64034 | 0.61329 / 0.63145 / 0.63660 / 0.63850 |
| metal, 900 nm, theta 20 deg, phi 30 deg, p | 1 | 0.32910 / 0.32900 / 0.32902 | 0.32154 / 0.32339 / 0.32445 / 0.32565 |
| metal, 350 nm, normal | 5 | 0.36196 / 0.36280 / 0.36279 | 0.34676 / 0.35414 / 0.35748 / 0.35884 |
| hole, 900 nm, theta 20 deg, p | 1 | 0.88329 / 0.88395 / 0.88409 | 0.87830 / 0.87979 / 0.87982 / 0.87956 |

- In the lossless cases `R + T = 1` to 2e-6 at order 8 and to 1e-9 from order 12.
- On the eight block cases the stretched solve moves less between orders 12 and 16 (at most 8e-4, beside the Rayleigh wavelength; 9e-5 elsewhere) than the unstretched one between 16 and 20, and the unstretched one approaches it monotonically.
- On the hole the stretched solve moves by 1.4e-4 between orders 12 and 16; the unstretched one has not settled by order 20, where it turns back.
- **This is not evidence of accuracy.** It shows that the scheme -- half-spaces solved in the stretched coordinates, each propagating order replaced by its plane wave -- does not break at oblique incidence or with several propagating orders.

### Beside a Rayleigh wavelength

The cell's period is 400 nm, so 405 nm is just above a Rayleigh wavelength and 395 nm just below.

- At 395 nm, with five propagating orders, the stretched solve is settled from order 8.
- **At 405 nm it is far off at order 8** (`R` 0.021 against 0.312) and settles from order 12; the unstretched solve is slow there too. Replacing the orders just past cutoff by plane waves as well, in a test implementation, changed nothing: the low orders are not short of a better half-space, the layer itself converges slowly at that wavelength.

## Why the half-spaces are solved in the stretched coordinates

The alternative is to leave the half-spaces unstretched and carry every one of their plane waves, evanescent ones included, into the layers' basis. A test implementation did that first. Where the coordinate runs fast the high orders then oscillate beyond the retained harmonics, and the matrix that carries them is nearly singular: its condition number was 1e4, 5e6 and 2e9 at orders 8, 12 and 16 with a slope of 0.02.

- On the dielectric layers the answer was the same either way, to four digits, even at 2e9.
- On the metal ones the reflectance stayed right while the field broke down as the order rose: the metal rectangle read 5.7%, 17% and 27% at orders 8, 12, 16 where the solve above reads 4.5%, 3.6%, 3.4%, and the block with `eps` -100+5j was 143% off at order 16.

Solving the half-spaces in the same coordinates needs only the propagating orders' plane waves, a handful of well-conditioned columns, and no inverse of that matrix anywhere.

## Why this function and this placement

- **A bump at each wall did less.** A profile that slows the coordinate only within a fixed band around each wall was measured first, with the slope at the wall as low as 0.02; on the metal rectangle it read 19%, 14% and 4.2% at the three orders, against 4.5%, 3.6%, 3.4% for the sine. With it a dielectric layer's `R` also got worse at low order (3.5% off at order 8, against 0.26% here). The shape of the function matters more than the slope at the wall.
- **Equal distances between the walls.** With each piece instead as wide in `u` as it is in `x`, the metal rectangle read 6.7% and 6.3% at orders 8 and 12, and on a uniform slab with two walls 37.5 nm apart the field was wrong by more than its own size; with equal distances the same slab's field was uniform to 1.5%.
- **The slope.** On the metal rectangle a slope of 0.1 read 8.6 / 4.6 / 4.1% and on the dielectric square 0.93 / 0.75 / 0.51%, between the unstretched solve and 0.02.

These three comparisons were made with a test implementation that the product code reproduces to five digits in `R` and `T` on thirteen cases.

(wall-resolution-circles-measured)=
## Circles and ellipses

A stack whose patterned layers hold one circle or one axis-aligned ellipse is solved in coordinates matched to it ([the guide](../../guide/rcwa/wall-resolution.md)): the mesh of Weiss et al. (Opt. Express 17, 8051 (2009)) carries the rectangle whose corners are the ellipse's 45-degree points onto the ellipse, and the stretch above puts its four lines half a period apart at the slope `G`. The materials become full in-plane tensors, `eps~ = eps det(J) (J^T J)^-1` with `J` the map's Jacobian, sampled on a 512 by 512 raster of the stretched cell whose pixel edges are those lines and put through the tensor Li factorization. The half-spaces are solved in the same coordinates, as for rectangles, and their propagating orders replaced by the exact plane waves written in the matched basis.

### Near fields on twelve layers

Normal incidence with `E` along x. The cell is the one above (400 nm, a 150 nm layer, 900 nm) except the resonant pair, a circle of radius 280 nm in a 1000 nm cell, 200 nm thick, at 1363 nm. Plane errors as above, wall samples included, at orders 8 / 12, against 3D finite-element fields (gyptis 1.2.0); the default solve is the product's NVF with the plain sum.

| Layer | Default solve | `wall_resolution=0.02` | `R`: matched (reference) |
| --- | --- | --- | --- |
| metal circle, radius 100 nm, `eps` -20+1.5j | 27.4 / 21.5% | 2.1 / 0.42% | 0.18804 / 0.18792 (0.18799) |
| metal circle, radius 60 nm | 17.7 / 19.8% | 3.9 / 0.43% | 0.01656 / 0.01657 (0.01658) |
| metal circle, radius 140 nm | 32.1 / 24.6% | 1.1 / 0.32% | 0.55233 / 0.55230 (0.55237) |
| metal circle, `eps` -100+5j | 29.7 / 28.2% | 1.1 / 0.49% | 0.15754 / 0.15749 (0.15755) |
| metal circle, `eps` -5+1j | 20.9 / 17.3% | 2.1 / 1.8% | 0.37027 / 0.36986 (0.36538) |
| metal circle, resonant cell | 38.6 / 27.3% | 5.5 / 1.2% | 0.76081 / 0.76214 (0.76235) |
| metal ellipse, 130 by 80 nm | 30.6 / 27.5% | 2.1 / 0.61% | 0.45155 / 0.45145 (0.45173) |
| metal ellipse, 150 by 50 nm | 58.2 / 39.2% | 3.1 / 1.4% | 0.76094 / 0.76122 (0.76133) |
| metal ellipse, 110 by 90 nm | 24.0 / 24.8% | 2.2 / 0.62% | 0.25519 / 0.25515 (0.25553) |
| dielectric circle, radius 100 nm, `eps` 12.25 | 13.6 / 11.2% | 0.24 / 0.14% | 0.02429 / 0.02428 (0.02431) |
| dielectric circle, resonant cell | 9.1 / 7.1% | 1.2 / 0.14% | 0.77225 / 0.76715 (0.76816) |
| dielectric ellipse, 130 by 80 nm | 15.5 / 12.8% | 0.37 / 0.25% | 0.05278 / 0.05277 (0.05282) |

- **Three references are stored**: the two circles of radius 100 nm (`golden_gyptis_cylinder_field.npz`) and the dielectric ellipse (`golden_gyptis_shapes_field.npz`), read by `tests/references/test_gyptis_matched_field_golden.py`, order 8 in the default run and order 12 in the slow tier. The other nine are in a private research repository, and no test reads them.
- **How good the references are.** The stored ones' next mesh levels are 0.15%, 0.11% and 0.38% of the plane RMS from them. The resonant pair is at the third level of its ladder, the 130 by 80 metal ellipse at the second (against the first, the matched solve reads 0.70% at order 12, against the second 0.61%), the 150 by 50 and 110 by 90 ellipses at the first, the others at the second. Below about 0.3 to 0.6% a difference is within the references' own step between levels and is not read.
- **The default solve's `R`** is 4% low on the resonant metal cell and 12% low on the 150 by 50 ellipse at order 12 (0.73033 and 0.66875); the matched solve is within 0.03% and 0.01% there.
- **The weak metal.** On `eps` -5+1j the matched solve's `R` is 1.3% and 1.2% above the reference's at orders 8 and 12. Two finite-element models of that circle agree at their finest levels (0.36548 and 0.36520; the reference is 0.36538), and the default solve's `R` (0.36643 at order 12) is on their side; in a test implementation the offset grew as the stretch was made stronger, while the field kept improving. It is taken to be this method's error on a weak metal, and the guide says so.

### Criteria for circles and ellipses, written before the implementation

Set from a test implementation's values with about a factor of two of margin (design of the second stage), at order 12, with slope 0.02:

| Criterion | Limit | Measured |
| --- | --- | --- |
| five metal circles (radius 60, 100, 140 nm; `eps` -100+5j; the resonant cell): field as a fraction of the default solve's | 0.1 | 0.013 to 0.044 |
| the same five: `R` against the reference | 0.3% | 0.01 to 0.07% |
| three metal ellipses: field as a fraction of the default solve's | 0.1 | 0.022 to 0.036 |
| the same three: `R` against the reference | 0.3% | 0.01 to 0.15% |
| dielectric circle: field | 0.5% | 0.14% |
| the circle of `eps` -5+1j: field as a fraction / `R` | 0.2 / 1.5% | 0.10 / 1.2% |
| gold cylinder of Küchenmeister et al. (2012): `R` from order 12 | 0.823 +- 0.001 | 0.8231, 0.8234 (orders 12, 14) |

The gold cylinder is the published case of the mesh (Küchenmeister, Zebrowski and Busch, Opt. Express 20, 17319 (2012), section 9.2: gold of their Drude model, `eps` -26.37+1.80j at 829 nm, radius 150 nm, height 50 nm, period 700 nm, in air). The paper prints no table; its figure shows the mesh's `R` flat at about 0.823 from roughly 500 plane waves on, read off the plot. Here `R` is 0.8129, 0.8210, 0.8231, 0.8234 at orders 8, 10, 12, 14 (289 to 841 plane waves; the paper's truncation is circular, this one square).

All are met. The field reading had to change to meet the first: read as a plain sum, the resonant metal cell was 4.0% off at order 12, 0.147 of the default solve's error; read through the window below, 1.2%.

### How the field is read

The map bends on its four lines, which cross the whole cell, so the component of the solve's `E~` normal to a line jumps there even in a uniform medium, and its plain sum rings. The reading takes that component from the normal `D~ = [[eps~]] E~` within half a resolution unit `period / (2 N + 1)` of a line, keeps the plain sum beyond one and a half units, and blends the two with a `cos^2` between (`H` the same with `B`). At order 12 the plain sum against this window:

| Layer | Plain sum | Window |
| --- | --- | --- |
| metal circles: radius 100, 60, 140 nm; `eps` -100+5j, -5+1j | 0.51, 0.55, 1.48, 0.83, 1.80% | 0.42, 0.43, 0.32, 0.49, 1.77% |
| resonant cell: metal, dielectric | 4.02, 1.07% | 1.20, 0.14% |
| metal ellipses: 130 by 80, 150 by 50, 110 by 90 nm | 1.09, 2.20, 1.04% | 0.61, 1.39, 0.62% |
| dielectric circle, dielectric ellipse | 0.31, 0.34% | 0.14, 0.25% |

The window's widths are those a test implementation chose on a weaker stretch; they were not tuned again here. The product code reproduces that implementation's `R` to five digits and its plane errors to the digits that implementation printed, on all twelve layers.

### What the matched construction has to satisfy

Each is a test in `tests/rcwa/test_wall_resolution_matched.py`.

- **The map.** The rectangle's sides land on the ellipse to 1e-12, the map is periodic and does not fold, its analytic Jacobian agrees with finite differences, and its closed-form inverse returns a point to 1e-8 nm, on a circle, on ellipses, one of them across the cell's corner, and in a rectangular cell. The raster's pixel edges are the lines.
- **The plane waves in the matched basis** agree with a brute-force sum over the cell to the sum's own error.
- **A uniform stack.** With the shape of the background's own material the physics is a stack of uniform layers, yet the map bends every line and every layer is a full tensor. At order 6, over glass, lit obliquely from either side, `R`, `T` and the specular amplitudes are the plane-wave answer to 6e-9 on a circle and 6e-7 on a 150 by 60 ellipse; the electric field is within 0.15% and 0.41% RMS.
- **Moving the shape**, also across the cell's edge, gives the same `R` and `T` to 1e-9 and the same field at the moved samples.
- **Energy.** On a lossless ellipse `R + T = 1` to 1e-4 at order 6, at normal and oblique incidence and with four and five propagating orders.
- **Backends.** NumPy, JAX and PyTorch agree to 1e-9 in `R` and `T` and 1e-8 of the field's scale on a metal and a dielectric ellipse.

### What the matched solve costs

At order 12 on one layer the matched solve took 16 to 17 seconds where the default NVF solve took 10 to 11, on four cores; at order 8, 3 seconds against 2.

## Not measured

- Any reference at oblique incidence, with several propagating orders, on a substrate, or on several patterned layers; for circles and ellipses, also any reference with a shape away from the cell's centre or in a non-square cell.
- An independent implementation of the matched mesh: none was found to compare with.
- Layers with more than two walls along an axis against a reference. A uniform slab through four unevenly placed walls reads the right `R` to 1e-5.
- A second mesh level for five of the seven references.
- Single precision; `jax.jit`, `jax.vmap` and gradients, which are refused.
- The interaction with the exceptional-point handling: a strong stretch is reported in the literature to add spurious layer modes that move away as the order rises, and none was looked for.
